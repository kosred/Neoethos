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
  type PortfolioEntry,
  type GateVerdict,
  type ParityReport,
} from "../api";
import { usePoll } from "../hooks";
import { HelpPanel, HelpStep, Tip } from "../components/Help";

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
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  // Filters
  const [modeFilter, setModeFilter] = useState<"all" | "risky" | "prop">("all");
  const [onlyValidated, setOnlyValidated] = useState(false);
  const [validatedKeys, setValidatedKeys] = useState<Set<string>>(new Set());
  // Auto-cull: retire a strategy after this many consecutive losing trades.
  const [cullLosses, setCullLosses] = useState(6);

  const { data: blacklist } = usePoll(strategyBlacklist, 0);
  const retired = Array.isArray(blacklist) ? blacklist : [];

  const engines: any[] = status?.engines ?? [];
  const running = !!status?.running;
  const runningPaths = new Set(engines.map((e) => e.portfolioPath));
  const allPortfolios = list?.portfolios ?? [];
  const portfolios = allPortfolios.filter((p) => {
    if (p.blacklisted) return false; // retired strategies are never selectable
    const isProp = p.path.toLowerCase().includes("propfirm");
    if (modeFilter === "risky" && isProp) return false;
    if (modeFilter === "prop" && !isProp) return false;
    if (onlyValidated && !validatedKeys.has(`${p.symbol ?? ""}|${p.baseTf ?? ""}`)) return false;
    return true;
  });

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
      const r: any = await autonomousStart({ portfolio_paths: selected, cull_after_consecutive_losses: cullLosses });
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
          Mode
          <select value={modeFilter} onChange={(e) => setModeFilter(e.target.value as any)}>
            <option value="all">All</option>
            <option value="risky">🚀 Risky</option>
            <option value="prop">🛡 Prop-firm</option>
          </select>
        </label>
        <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }} title="Only strategies whose backtest passed CPCV + Walkforward out-of-sample">
          <input type="checkbox" checked={onlyValidated} onChange={(e) => setOnlyValidated(e.target.checked)} /> Only validated (passed OOS)
        </label>
        <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
          Auto-cull after <Tip text="After this many CONSECUTIVE losing trades, the engine stops itself and permanently retires the strategy (blacklist) — it can never be selected or re-discovered again. 0 = off." />
          <input type="number" min={0} max={50} value={cullLosses} onChange={(e) => setCullLosses(Math.max(0, Number(e.target.value)))} style={{ width: 56 }} /> losses
        </label>
        <span className="muted small">{portfolios.length} of {allPortfolios.length} shown · {selected.length} selected{retired.length ? ` · ${retired.length} retired` : ""}</span>
      </div>
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
            <thead><tr><th></th><th>Symbol</th><th>Base TF</th><th>Genes</th><th>File</th><th></th></tr></thead>
            <tbody>
              {portfolios.map((p) => {
                const live = runningPaths.has(p.path);
                return (
                  <tr key={p.path} className={focus?.path === p.path ? "row-sel" : ""}>
                    <td><input type="checkbox" checked={selected.includes(p.path)} onChange={() => toggle(p.path)} /></td>
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
          <button className="primary" disabled={busy || selected.length === 0} onClick={startSelected}>
            Start selected (live) · {selected.length}
          </button>
          <button className="danger" disabled={busy || !running} onClick={stopAll}>Stop all</button>
        </div>
        {focus && <p className="muted small" style={{ marginTop: 8 }}>Focused: {focus.symbol} {focus.baseTf} — {focus.path}</p>}
        {msg && <div className="banner info">{msg}</div>}
      </div>

      <GatePanel gate={gate} />

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
            <thead><tr><th>Symbol</th><th>Base TF</th><th>Genes</th><th>Bars eval</th><th>Last signal</th><th>Open pos</th><th>Loss streak</th></tr></thead>
            <tbody>
              {engines.map((e, i) => {
                const losses = Number(e.consecutiveLosses ?? 0);
                return (
                  <tr key={e.portfolioPath ?? i}>
                    <td><b>{e.symbol ?? "?"}</b>{e.retired && <span className="badge" style={{ marginLeft: 6, background: "#7f1d1d", fontSize: 9 }}>RETIRED</span>}</td>
                    <td>{e.baseTf ?? "?"}</td>
                    <td>{e.genes ?? "—"}</td>
                    <td>{typeof e.barsEvaluated === "number" ? e.barsEvaluated.toLocaleString() : "—"}</td>
                    <td>{e.lastSignal ?? "—"}</td>
                    <td>{e.openPositionId ?? "—"}</td>
                    <td className={losses >= Math.max(1, cullLosses - 1) ? "sell" : ""}>{losses > 0 ? `${losses} in a row` : "—"}</td>
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
