import { useEffect, useState } from "react";
import {
  portfoliosList,
  autonomousStatus,
  autonomousStart,
  autonomousStop,
  autonomousReplay,
  autonomousGate,
  openPath,
  strategyList,
  strategyBlacklist,
  parityCheck,
  tailRisk,
  challengeSim,
  type PortfolioEntry,
  type GateVerdict,
  type ParityReport,
  type TailRiskReport,
  type ChallengeReport,
} from "../api";
import { usePoll } from "../hooks";
import { HelpPanel, HelpStep, Tip } from "../components/Help";
import { FilterChips, ago, stamp, tfRank, toggleIn } from "../components/filters";

function StatGrid({ data }: { data: any }) {
  if (!data || typeof data !== "object") return null;
  const rows = Object.entries(data).filter(([, v]) => typeof v !== "object" || v === null);
  return (
    <table className="tbl">
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td style={{ color: "#9ca3af" }}>{k.replace(/_/g, " ")}</td>
            <td>{typeof v === "number" ? (Number.isInteger(v) ? (v as number).toLocaleString() : (v as number).toFixed(4)) : String(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function GatePanel({ gate }: { gate: GateVerdict | null }) {
  if (!gate) return null;
  const blocked = gate.enforced && !gate.eligible;
  return (
    <div className="ticket" style={{ marginTop: 12, borderColor: blocked ? "#b91c1c" : gate.eligible ? "#15803d" : "#a16207" }}>
      <h2 style={{ marginTop: 0 }}>
        Demo forward-test gate{" "}
        <span className={`badge ${gate.enforced ? "live" : "demo"}`}>
          {gate.enforced ? "LIVE — enforced" : "DEMO — informational"}
        </span>{" "}
        <span className={`badge ${gate.eligible ? "demo" : ""}`} style={{ background: gate.eligible ? "#15803d" : "#a16207" }}>
          {gate.eligible ? "ELIGIBLE" : "NOT YET"}
        </span>
      </h2>
      <p className="muted small">{gate.summary}</p>
      {!gate.enforced && (
        <p className="muted small">
          Active account is a <b>Demo</b> environment — running here builds the demo track record. The gate only
          blocks <b>real-money</b> (Live) accounts.
        </p>
      )}
      {gate.criteria.length > 0 && (
        <table className="tbl">
          <thead><tr><th></th><th>Criterion</th><th>Live</th><th></th><th>Backtest floor/cap</th></tr></thead>
          <tbody>
            {gate.criteria.map((c) => (
              <tr key={c.name}>
                <td>{c.passed ? "✅" : "❌"}</td>
                <td>{c.name}</td>
                <td>{c.actual.toFixed(2)}</td>
                <td style={{ color: "#9ca3af" }}>{c.comparison}</td>
                <td>{c.threshold.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

const fileTail = (p: string) => p.split(/[\\/]/).pop() ?? p;

export default function Autopilot() {
  const { data: list, error, reload } = usePoll(portfoliosList, 0);
  const { data: status, reload: reloadStatus } = usePoll(autonomousStatus, 3000);
  const [selected, setSelected] = useState<string[]>([]);
  const [focus, setFocus] = useState<PortfolioEntry | null>(null);
  const [gate, setGate] = useState<GateVerdict | null>(null);
  const [replay, setReplay] = useState<any>(null);
  const [parity, setParity] = useState<ParityReport | null>(null);
  const [risk, setRisk] = useState<TailRiskReport | null>(null);
  const [chal, setChal] = useState<ChallengeReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  // Filters — same set as Strategy Report (operator request 2026-07-21), sharing
  // the chip/format helpers so the two screens can never drift apart.
  const [modeFilter, setModeFilter] = useState<"all" | "risky" | "prop">("all");
  const [onlyValidated, setOnlyValidated] = useState(false);
  const [validatedKeys, setValidatedKeys] = useState<Set<string>>(new Set());
  const [symFilter, setSymFilter] = useState<string[]>([]);
  const [tfFilter, setTfFilter] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<"discovered" | "symbol" | "genes">("discovered");
  // Auto-cull: retire a strategy after this many consecutive losing trades.
  const [cullLosses, setCullLosses] = useState(6);
  // Auto-cull, rolling window: min win-rate % over the last N closed trades.
  const [cullMinWr, setCullMinWr] = useState(57);
  const [cullWindow, setCullWindow] = useState(10);

  const { data: blacklist } = usePoll(strategyBlacklist, 0);
  const retired = Array.isArray(blacklist) ? blacklist : [];

  const engines: any[] = status?.engines ?? [];
  const running = !!status?.running;
  const runningPaths = new Set(engines.map((e) => e.portfolioPath));
  const allPortfolios = list?.portfolios ?? [];
  const selectable = allPortfolios.filter((p) => !p.blacklisted); // retired are never selectable
  const symbolOpts = Array.from(new Set(selectable.map((p) => p.symbol ?? "?"))).sort();
  const tfOpts = Array.from(new Set(selectable.map((p) => p.baseTf ?? "?"))).sort(
    (a, b) => tfRank(a) - tfRank(b),
  );
  const q = search.trim().toUpperCase();
  const portfolios = selectable
    .filter((p) => {
      const isProp = p.path.toLowerCase().includes("propfirm");
      if (modeFilter === "risky" && isProp) return false;
      if (modeFilter === "prop" && !isProp) return false;
      if (onlyValidated && !validatedKeys.has(`${p.symbol ?? ""}|${p.baseTf ?? ""}`)) return false;
      if (symFilter.length && !symFilter.includes(p.symbol ?? "?")) return false;
      if (tfFilter.length && !tfFilter.includes(p.baseTf ?? "?")) return false;
      if (q && !`${p.symbol ?? ""} ${p.baseTf ?? ""} ${p.fileName}`.toUpperCase().includes(q)) return false;
      return true;
    })
    .sort((a, b) => {
      if (sortBy === "discovered") return (b.modifiedMs ?? 0) - (a.modifiedMs ?? 0);
      if (sortBy === "genes") return (b.geneCount ?? 0) - (a.geneCount ?? 0);
      return (a.symbol ?? "").localeCompare(b.symbol ?? "") || tfRank(a.baseTf ?? "") - tfRank(b.baseTf ?? "");
    });
  // Rollup of what is currently shown, so "what do I have per timeframe" is one
  // glance instead of counting rows.
  const byTf = (() => {
    const m = new Map<string, { n: number; live: number }>();
    for (const p of portfolios) {
      const k = p.baseTf ?? "?";
      const e = m.get(k) ?? { n: 0, live: 0 };
      e.n += 1;
      if (runningPaths.has(p.path)) e.live += 1;
      m.set(k, e);
    }
    return [...m.entries()].sort((a, b) => tfRank(a[0]) - tfRank(b[0]));
  })();
  const newest = selectable.reduce<number | null>(
    (acc, p) => (p.modifiedMs && (!acc || p.modifiedMs > acc) ? p.modifiedMs : acc),
    null,
  );
  const filtersOn =
    symFilter.length > 0 || tfFilter.length > 0 || modeFilter !== "all" || onlyValidated || q !== "";
  const clearFilters = () => {
    setSymFilter([]); setTfFilter([]); setModeFilter("all"); setOnlyValidated(false); setSearch("");
  };

  // Validated set (passed CPCV + Walkforward) keyed by symbol|timeframe, from the
  // strategy report — powers the "only validated" filter.
  useEffect(() => {
    let live = true;
    strategyList()
      .then((r) => {
        if (!live) return;
        const keys = new Set<string>();
        for (const s of r.strategies) {
          if (s.cpcvPassed && s.walkforwardPassed) keys.add(`${s.symbol}|${s.timeframe}`);
        }
        setValidatedKeys(keys);
      })
      .catch(() => {});
    return () => { live = false; };
  }, []);

  // Demo forward-test verdict for the focused strategy.
  useEffect(() => {
    setGate(null);
    if (!focus?.path) return;
    let live = true;
    autonomousGate(focus.path)
      .then((v) => { if (live) setGate(v); })
      .catch(() => { if (live) setGate(null); });
    return () => { live = false; };
  }, [focus?.path]);

  const toggle = (path: string) =>
    setSelected((s) => (s.includes(path) ? s.filter((x) => x !== path) : [...s, path]));

  const startSelected = async () => {
    if (selected.length === 0) { setMsg("Tick at least one strategy to run."); return; }
    setBusy(true);
    setMsg(`Starting ${selected.length} engine${selected.length === 1 ? "" : "s"}…`);
    try {
      const r: any = await autonomousStart({
        portfolio_paths: selected,
        cull_after_consecutive_losses: cullLosses,
        cull_min_win_rate_pct: cullMinWr,
        cull_window_trades: cullWindow,
      });
      const s = r?.started?.length ?? 0;
      const sk = r?.skipped?.length ?? 0;
      const bl = r?.blacklisted?.length ?? 0;
      const f = r?.failed ?? [];
      let m = `✓ Started ${s}${sk ? `, ${sk} already running` : ""}${bl ? `, ${bl} retired (skipped)` : ""}${f.length ? `, ${f.length} blocked/failed` : ""}.`;
      if (f.length) m += " — " + f.map((x: any) => `${fileTail(x.portfolio)}: ${x.error}`).join("; ");
      setMsg(m);
      await reloadStatus();
    } catch (e) {
      setMsg(`Start failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const stopAll = async () => {
    setBusy(true);
    setMsg("Stopping all engines…");
    try {
      await autonomousStop();
      setMsg("✓ All engines stopped.");
      await reloadStatus();
    } catch (e) {
      setMsg(`Stop failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const doReplay = async () => {
    if (!focus) { setMsg("Click a strategy name to focus it, then Replay."); return; }
    setBusy(true);
    setMsg(`Replaying ${focus.symbol ?? ""} ${focus.baseTf ?? ""}…`);
    try {
      const r = await autonomousReplay({ symbol: focus.symbol ?? undefined, base_tf: focus.baseTf ?? undefined });
      setReplay(r);
      setMsg("✓ Replay done.");
    } catch (e) {
      setMsg(`Replay failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const doTailRisk = async () => {
    if (!focus) { setMsg("Click a strategy name to focus it, then Tail risk."); return; }
    setBusy(true);
    setRisk(null);
    setMsg(`Monte-Carlo tail risk for ${focus.symbol ?? ""}… (2000 reshuffles)`);
    try {
      const r = await tailRisk(focus.path);
      setRisk(r);
      setMsg(r.ruinProbabilityPct >= 1 ? "⚠ Tail risk: DANGER — see the report." : "✓ Tail risk computed.");
    } catch (e) {
      setMsg(`Tail risk failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const doChallenge = async () => {
    if (!focus) { setMsg("Click a strategy name to focus it, then Challenge sim."); return; }
    setBusy(true);
    setChal(null);
    setMsg(`Challenge first-passage Monte Carlo for ${focus.symbol ?? ""}… (2000 paths × 6 sizes × 2 phases)`);
    try {
      const r = await challengeSim(focus.path);
      setChal(r);
      setMsg(r.bestFundedPct < 5 ? "⚠ Challenge sim: this edge barely clears prop-firm barriers." : "✓ Challenge sim computed.");
    } catch (e) {
      setMsg(`Challenge sim failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const doParity = async () => {
    if (!focus) { setMsg("Click a strategy name to focus it, then Check parity."); return; }
    setBusy(true);
    setParity(null);
    setMsg(`Checking live↔backtest parity for ${focus.symbol ?? ""}… (fetches broker bars, ~10-30s)`);
    try {
      const r = await parityCheck(focus.path);
      setParity(r);
      setMsg(r.verdict === "PASS" ? "✓ Parity PASS — live signals are window-invariant." : "✗ Parity FAIL — see the report below.");
    } catch (e) {
      setMsg(`Parity check failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>
        Autopilot{" "}
        <span className={`badge ${running ? "live" : "demo"}`}>
          {running ? `LIVE · ${engines.length} engine${engines.length === 1 ? "" : "s"}` : "STOPPED"}
        </span>
      </h1>
      <p className="sub">Run one or MANY discovered strategies at once — each is internally multi-timeframe and trades concurrently</p>

      <HelpPanel id="autopilot">
        <p>Autopilot runs your discovered strategies <b>for you</b>. Each strategy already reads every higher timeframe internally; running several together is like a trader watching multiple setups at once.</p>
        <HelpStep n={1}>Tick the strategies you want to run (use <b>Select all</b> to deploy your whole library). Click a name to <b>focus</b> it for Replay + the demo-gate detail.</HelpStep>
        <HelpStep n={2}><b>Replay (dry-run)</b> tests the focused strategy on stored history with zero broker calls.</HelpStep>
        <HelpStep n={3}><b>Start selected (live)</b> launches one concurrent engine per ticked strategy. On a <b>Demo</b> account they always run (building the track record); on a <b>Live</b> account each is blocked until its demo gate passes — blocked ones are reported, the rest still start.</HelpStep>
        <HelpStep n={4}><b>Stop all</b> halts every running engine.</HelpStep>
        <HelpStep n={5}><b>Auto-cull</b>: set a consecutive-loss limit. When a running strategy hits it, the engine stops itself and the strategy is <b>permanently retired</b> — blacklisted so it can never be selected or re-discovered again (kept as a record, never deleted). Retired ones appear at the bottom.</HelpStep>
      </HelpPanel>

      <div className="btn-row" style={{ flexWrap: "wrap", alignItems: "center" }}>
        <button onClick={reload} disabled={busy}>Refresh strategies</button>
        <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
          Search
          <input value={search} placeholder="EURUSD, M5…" onChange={(e) => setSearch(e.target.value)} style={{ width: 140 }} />
        </label>
        <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
          Mode
          <select value={modeFilter} onChange={(e) => setModeFilter(e.target.value as any)}>
            <option value="all">All</option>
            <option value="risky">🚀 Risky</option>
            <option value="prop">🛡 Prop-firm</option>
          </select>
        </label>
        <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
          Sort
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value as typeof sortBy)}>
            <option value="discovered">Newest first</option>
            <option value="genes">Most genes</option>
            <option value="symbol">Symbol · TF</option>
          </select>
        </label>
        <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }} title="Only strategies whose backtest passed CPCV + Walkforward out-of-sample">
          <input type="checkbox" checked={onlyValidated} onChange={(e) => setOnlyValidated(e.target.checked)} /> Only validated (passed OOS)
        </label>
        {filtersOn && <button className="link" onClick={clearFilters}>clear filters</button>}
        <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
          Auto-cull after <Tip text="After this many CONSECUTIVE losing trades, the engine stops itself and permanently retires the strategy (blacklist) — it can never be selected or re-discovered again. 0 = off." />
          <input type="number" min={0} max={50} value={cullLosses} onChange={(e) => setCullLosses(Math.max(0, Number(e.target.value)))} style={{ width: 56 }} /> losses
        </label>
        <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
          or WR &lt; <Tip text="Rolling-window cull: once the last N closed trades are in, the win rate must stay at or above this percent or the strategy retires. Catches chronic losers (e.g. 40% WR) that never streak. 0 = off." />
          <input type="number" min={0} max={100} value={cullMinWr} onChange={(e) => setCullMinWr(Math.min(100, Math.max(0, Number(e.target.value))))} style={{ width: 56 }} />% per last
          <input type="number" min={4} max={100} value={cullWindow} onChange={(e) => setCullWindow(Math.max(4, Number(e.target.value)))} style={{ width: 52 }} /> trades
        </label>
        <span className="muted small">
          {portfolios.length} of {allPortfolios.length} shown · {selected.length} selected
          {retired.length ? ` · ${retired.length} retired` : ""}
          {newest ? ` · newest ${stamp(newest)}` : ""}
        </span>
      </div>
      <FilterChips label="Pairs" opts={symbolOpts} sel={symFilter} onToggle={toggleIn(setSymFilter)} />
      <FilterChips label="Timeframes" opts={tfOpts} sel={tfFilter} onToggle={toggleIn(setTfFilter)} />
      {byTf.length > 1 && (
        <div className="cards" style={{ gridTemplateColumns: `repeat(${Math.min(6, byTf.length)}, 1fr)` }}>
          {byTf.map(([tf, e]) => (
            <div className="card" key={tf} title={`${e.n} strategies on ${tf}${e.live ? `, ${e.live} running live` : ""}`}>
              <div className="card-label">{tf}</div>
              <div className="card-value">{e.n}</div>
              {e.live > 0 && <div className="muted small">{e.live} live</div>}
            </div>
          ))}
        </div>
      )}
      {error && <div className="banner warn">{error}</div>}

      {portfolios.length === 0 ? (
        <p className="muted">{allPortfolios.length === 0 ? "No discovered strategies yet — run Discovery, then promote in Strategy Lab." : "No strategies match the current filters."}</p>
      ) : (
        <>
          <div className="btn-row" style={{ marginBottom: 6 }}>
            <button className="link" onClick={() => setSelected(portfolios.map((p) => p.path))}>Select all</button>
            <button className="link" onClick={() => setSelected([])}>None</button>
          </div>
          <table className="tbl">
            <thead><tr><th></th><th>Discovered</th><th>Symbol</th><th>Base TF</th><th>Genes</th><th>File</th><th></th></tr></thead>
            <tbody>
              {portfolios.map((p) => {
                const live = runningPaths.has(p.path);
                return (
                  <tr key={p.path} className={focus?.path === p.path ? "row-sel" : ""}>
                    <td><input type="checkbox" checked={selected.includes(p.path)} onChange={() => toggle(p.path)} /></td>
                    <td className="muted small" style={{ whiteSpace: "nowrap" }} title={ago(p.modifiedMs)}>
                      {stamp(p.modifiedMs)}
                    </td>
                    <td>
                      <button className="link" onClick={() => setFocus(p)} style={{ fontWeight: 700 }}>{p.symbol ?? "?"}</button>
                      {live && <span className="badge live" style={{ marginLeft: 6, fontSize: 9 }}>LIVE</span>}
                    </td>
                    <td>{p.baseTf ?? "?"}</td>
                    <td>{p.geneCount ?? "—"}</td>
                    <td style={{ fontFamily: "monospace", fontSize: 11, color: "#9ca3af", maxWidth: 300, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={p.path}>{p.fileName}</td>
                    <td><button onClick={() => openPath(p.path).catch(() => {})}>Open</button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}

      <div className="ticket" style={{ marginTop: 14 }}>
        <div className="btn-row">
          <button disabled={busy || !focus} onClick={doReplay}>Replay focused (dry-run)</button>
          <button disabled={busy || !focus} onClick={doParity} title="Does the live bar-window produce the SAME signals as the full history? FAIL = live won't match the validated backtest.">Check parity</button>
          <button disabled={busy || !focus} onClick={doTailRisk} title="Monte-Carlo the trade sequence: worst-case drawdown distribution + probability of losing half the account at YOUR current risk %. The pre-Start number.">Tail risk</button>
          <button disabled={busy || !focus} onClick={doChallenge} title="First-passage Monte Carlo of a prop-firm challenge (FTMO-style +10% target vs −10% max / −5% daily loss): pass & funded probability per risk size, and the challenge-optimal size.">Challenge sim</button>
          <button className="primary" disabled={busy || selected.length === 0} onClick={startSelected}>
            Start selected (live) · {selected.length}
          </button>
          <button className="danger" disabled={busy || !running} onClick={stopAll}>Stop all</button>
        </div>
        {focus && <p className="muted small" style={{ marginTop: 8 }}>Focused: {focus.symbol} {focus.baseTf} — {focus.path}</p>}
        {msg && <div className="banner info">{msg}</div>}
      </div>

      <GatePanel gate={gate} />

      {risk && (
        <div className="ticket" style={{ marginTop: 12, borderColor: risk.ruinProbabilityPct >= 1 ? "#b91c1c" : risk.maxDdP95Pct > 30 ? "#a16207" : "#15803d" }}>
          <h2 style={{ marginTop: 0 }}>
            Tail risk (Monte-Carlo ×{risk.iterations}){" "}
            <span className="badge" style={{ background: risk.ruinProbabilityPct >= 1 ? "#b91c1c" : risk.maxDdP95Pct > 30 ? "#a16207" : "#15803d" }}>
              {risk.ruinProbabilityPct >= 1 ? "DANGER" : risk.maxDdP95Pct > 30 ? "CAUTION" : "SURVIVABLE"}
            </span>
          </h2>
          <p className="muted small">{risk.note}</p>
          <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
            <div className="card" title="Risk-constrained Kelly (Busseti/Ryu/Boyd), solved on the portfolio's FULL R-multiple distribution — rare catastrophic losses (fat left tail) shrink it automatically, like a CVaR-aware sizer. The largest risk-per-trade with ≤5% chance of EVER losing half the account. Set it as Risk per trade in Settings to trade at the survival-constrained optimum.">
              <div className="card-label">RCK risk/trade</div>
              <div className="card-value" style={{ color: "#4ade80" }}>{risk.rckRiskPct != null ? `${risk.rckRiskPct.toFixed(2)}%` : "—"}</div>
            </div>
            <div className="card" title="95th percentile of the LONGEST stretch below the equity peak (in trades) across the reshuffles — how long the portfolio can drag underwater before a new high. Prop-firm challenges implicitly test this recovery speed.">
              <div className="card-label">Underwater p95</div>
              <div className="card-value">{risk.underwaterP95Trades} trades</div>
            </div>
            <div className="card"><div className="card-label">DD p50</div><div className="card-value">{risk.maxDdP50Pct.toFixed(0)}%</div></div>
            <div className="card"><div className="card-label">DD p95</div><div className="card-value" style={{ color: risk.maxDdP95Pct > 30 ? "#ef5350" : undefined }}>{risk.maxDdP95Pct.toFixed(0)}%</div></div>
            <div className="card"><div className="card-label">DD p99</div><div className="card-value">{risk.maxDdP99Pct.toFixed(0)}%</div></div>
            <div className="card"><div className="card-label">Ruin (≥{risk.ruinThresholdPct.toFixed(0)}%)</div><div className="card-value" style={{ color: risk.ruinProbabilityPct >= 1 ? "#ef5350" : undefined }}>{risk.ruinProbabilityPct.toFixed(1)}%</div></div>
            <div className="card"><div className="card-label">Median final ×</div><div className="card-value">{risk.medianFinalMultiple.toFixed(2)}</div></div>
          </div>
          <p className="muted small" style={{ marginTop: 6 }}>
            {risk.trades} trades · risk {(risk.riskFraction * 100).toFixed(2)}%/trade · source: {risk.mode} · shuffling assumes independent trades — real streaks can cluster worse.
          </p>
        </div>
      )}

      {chal && (
        <div className="ticket" style={{ marginTop: 12, borderColor: chal.bestFundedPct < 5 ? "#b91c1c" : chal.bestFundedPct < 25 ? "#a16207" : "#15803d" }}>
          <h2 style={{ marginTop: 0 }}>
            Prop-firm challenge sim (first-passage ×{chal.iterations}){" "}
            <span className="badge" style={{ background: chal.bestFundedPct < 5 ? "#b91c1c" : chal.bestFundedPct < 25 ? "#a16207" : "#15803d" }}>
              {chal.bestFundedPct < 5 ? "NO REAL EDGE" : chal.bestFundedPct < 25 ? "MARGINAL" : "VIABLE"}
            </span>
          </h2>
          <p className="muted small">{chal.note}</p>
          <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
            <div className="card"><div className="card-label">Best risk/trade</div><div className="card-value">{chal.bestRiskPct.toFixed(2)}%</div></div>
            <div className="card"><div className="card-label">Funded / attempt</div><div className="card-value" style={{ color: chal.bestFundedPct < 5 ? "#ef5350" : undefined }}>{chal.bestFundedPct.toFixed(0)}%</div></div>
            <div className="card"><div className="card-label">Attempts for ≥90%</div><div className="card-value">{chal.attemptsFor90Pct > 0 ? chal.attemptsFor90Pct : "—"}</div></div>
            <div className="card"><div className="card-label">Trade cadence</div><div className="card-value">{chal.tradesPerDay.toFixed(1)}/day</div></div>
          </div>
          <table className="tbl" style={{ marginTop: 6 }}>
            <thead><tr><th>Risk/trade</th><th>Pass phase 1</th><th>Funded (1×2)</th><th>Bust</th><th>Timeout</th><th>Median days</th></tr></thead>
            <tbody>
              {chal.sweep.map((s) => (
                <tr key={s.riskPct} style={s.riskPct === chal.bestRiskPct ? { background: "rgba(21,128,61,0.15)" } : undefined}>
                  <td><b>{s.riskPct.toFixed(2)}%</b></td>
                  <td>{s.passPhase1Pct.toFixed(0)}%</td>
                  <td className={s.fundedPct >= 25 ? "buy" : s.fundedPct < 5 ? "sell" : ""}>{s.fundedPct.toFixed(0)}%</td>
                  <td className="muted">{s.bustPct.toFixed(0)}%</td>
                  <td className="muted">{s.timeoutPct.toFixed(0)}%</td>
                  <td className="muted">{s.medianDaysPhase1 > 0 ? s.medianDaysPhase1.toFixed(0) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="muted small" style={{ marginTop: 6 }}>
            Rules: +{chal.profitTargetPct.toFixed(0)}% target (phase 2: +{chal.phase2TargetPct.toFixed(0)}%) vs −{chal.maxLossPct.toFixed(0)}% max / −{chal.dailyLossPct.toFixed(0)}% daily loss ·
            {" "}{chal.dayLimitPhase1}/{chal.dayLimitPhase2}-day windows (conservative — FTMO dropped time limits) · {chal.trades} source trades ·
            bootstrap assumes iid trades — treat pass% as an upper bound.
          </p>
        </div>
      )}

      {parity && (
        <div className="ticket" style={{ marginTop: 12, borderColor: parity.verdict === "PASS" ? "#15803d" : "#b91c1c" }}>
          <h2 style={{ marginTop: 0 }}>
            Live↔backtest parity{" "}
            <span className="badge" style={{ background: parity.verdict === "PASS" ? "#15803d" : "#b91c1c" }}>{parity.verdict}</span>
          </h2>
          <p className="muted small">{parity.note}</p>
          <p className="muted small">
            {parity.symbol} {parity.baseTf} · window {parity.windowBars} vs reference {parity.referenceBars} bars · compared {parity.comparedBars} ·
            mismatches <b>{parity.directionMismatches}</b> · max ΔSL {parity.maxSlDeltaPips.toFixed(3)} pips · max ΔTP {parity.maxTpDeltaPips.toFixed(3)} pips
          </p>
          {parity.mismatchSamples.length > 0 && (
            <table className="tbl">
              <thead><tr><th>Bar</th><th>Reference</th><th>Live window</th></tr></thead>
              <tbody>
                {parity.mismatchSamples.map((m) => (
                  <tr key={m.barTsMs}>
                    <td className="muted">{new Date(m.barTsMs).toLocaleString()}</td>
                    <td>{m.reference}</td>
                    <td className="sell">{m.window}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {engines.length > 0 && (
        <>
          <h2>Running engines <span className="muted">({engines.length})</span></h2>
          <table className="tbl">
            <thead><tr><th>Symbol</th><th>Base TF</th><th>Genes</th><th>Bars eval</th><th>Last signal</th><th>Open pos</th><th>Loss streak</th><th>Window WR</th></tr></thead>
            <tbody>
              {engines.map((e, i) => {
                const losses = Number(e.consecutiveLosses ?? 0);
                const wr = typeof e.windowWinRatePct === "number" ? e.windowWinRatePct : null;
                const wrDanger = wr != null && Number(e.windowTrades ?? 0) >= cullWindow - 2 && wr < cullMinWr + 5;
                return (
                  <tr key={e.portfolioPath ?? i}>
                    <td><b>{e.symbol ?? "?"}</b>{e.retired && <span className="badge" style={{ marginLeft: 6, background: "#7f1d1d", fontSize: 9 }}>RETIRED</span>}</td>
                    <td>{e.baseTf ?? "?"}</td>
                    <td>{e.genes ?? "—"}</td>
                    <td>{typeof e.barsEvaluated === "number" ? e.barsEvaluated.toLocaleString() : "—"}</td>
                    <td>{e.lastSignal ?? "—"}</td>
                    <td>{e.openPositionId ?? "—"}</td>
                    <td className={losses >= Math.max(1, cullLosses - 1) ? "sell" : ""}>{losses > 0 ? `${losses} in a row` : "—"}</td>
                    <td className={wrDanger ? "sell" : ""}>{wr != null ? `${wr.toFixed(0)}% (${e.windowTrades}/${cullWindow})` : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}

      {retired.length > 0 && (
        <>
          <h2>Retired strategies <span className="muted">({retired.length})</span> <Tip text="Permanently blacklisted by auto-cull. Never traded or re-discovered again. Kept as a record — nothing is deleted." /></h2>
          <table className="tbl">
            <thead><tr><th>Symbol</th><th>Reason</th><th>Net P/L</th><th>Retired</th><th>File</th></tr></thead>
            <tbody>
              {retired.map((r) => (
                <tr key={r.fingerprint}>
                  <td><b>{r.symbol ?? "?"}</b></td>
                  <td className="muted small">{r.reason}</td>
                  <td className={r.netPnl < 0 ? "sell" : "buy"}>{r.netPnl.toFixed(2)}</td>
                  <td className="muted small">{r.retiredAtUnixMs ? new Date(r.retiredAtUnixMs).toLocaleString() : "—"}</td>
                  <td className="muted small" style={{ fontFamily: "monospace", maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={r.portfolioPath}>{fileTail(r.portfolioPath)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {replay && (<><h2>Replay result</h2><StatGrid data={replay} /></>)}
    </div>
  );
}
