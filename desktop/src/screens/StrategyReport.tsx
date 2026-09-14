import { useMemo, useState } from "react";
import { strategyList, strategyReport, type StrategyEntry, type StrategyReport as Report } from "../api";
import { usePoll } from "../hooks";
import { FilterChips } from "../components/FilterChips";
import { ago, stamp, tfRank, toggleIn } from "../components/filterUtils";

const normalizedUnits = (v: number) => `${v.toLocaleString(undefined, { maximumFractionDigits: 0 })} units`;
const recordedModeLabels: Record<string, string> = { risky: "Risky", prop_firm: "Prop-firm", strict: "Strict" };
const recordedModeLabel = (mode: string) => recordedModeLabels[mode] ?? "Unavailable";
const recordedPercent = (fraction: number) =>
  `${(fraction * 100).toLocaleString(undefined, { maximumFractionDigits: 3 })}%`;
type SortKey = "discovered" | "cagr" | "dd" | "trades" | "symbol";

export default function StrategyReport() {
  const { data, error, loading, reload } = usePoll(strategyList, 0);
  const [rep, setRep] = useState<Report | null>(null);
  const [busy, setBusy] = useState(false);
  const [reportError, setReportError] = useState("");

  // ── Filters (operator request: "I can't tell what happened per pair / per
  // timeframe, and I can't see WHEN anything was discovered") ──────────────
  const [symFilter, setSymFilter] = useState<string[]>([]);
  const [tfFilter, setTfFilter] = useState<string[]>([]);
  const [modeFilter, setModeFilter] = useState<"all" | "risky" | "prop_firm" | "strict" | "unknown">("all");
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<SortKey>("discovered");

  const all: StrategyEntry[] = useMemo(() => data?.strategies ?? [], [data]);

  // Option lists come from the DATA, so they only ever offer real choices.
  const symbols = useMemo(
    () => Array.from(new Set(all.map((s) => s.symbol))).sort(),
    [all],
  );
  const timeframes = useMemo(
    () => Array.from(new Set(all.map((s) => s.timeframe))).sort((a, b) => tfRank(a) - tfRank(b)),
    [all],
  );

  const rows = useMemo(() => {
    const q = search.trim().toUpperCase();
    const out = all.filter((s) => {
      if (symFilter.length && !symFilter.includes(s.symbol)) return false;
      if (tfFilter.length && !tfFilter.includes(s.timeframe)) return false;
      if (modeFilter !== "all" && s.mode !== modeFilter) return false;
      if (q && !`${s.symbol} ${s.timeframe} ${s.mode} ${s.strategyId}`.toUpperCase().includes(q)) return false;
      return true;
    });
    const cmp: Record<SortKey, (a: StrategyEntry, b: StrategyEntry) => number> = {
      discovered: (a, b) => (b.discoveredAtMs ?? 0) - (a.discoveredAtMs ?? 0),
      cagr: (a, b) => a.cagrPct == null ? (b.cagrPct == null ? 0 : 1) : b.cagrPct == null ? -1 : b.cagrPct - a.cagrPct,
      dd: (a, b) => a.maxDdPct - b.maxDdPct,
      trades: (a, b) => b.trades - a.trades,
      symbol: (a, b) => a.symbol.localeCompare(b.symbol) || tfRank(a.timeframe) - tfRank(b.timeframe),
    };
    return [...out].sort(cmp[sortBy]);
  }, [all, symFilter, tfFilter, modeFilter, search, sortBy]);

  // Per-timeframe rollup of the FILTERED set — answers "what is happening per
  // group" without reading every row.
  const byTf = useMemo(() => {
    const m = new Map<string, { n: number; best: number }>();
    for (const s of rows) {
      const e = m.get(s.timeframe) ?? { n: 0, best: -Infinity };
      e.n += 1;
      if (s.cagrPct != null && Math.abs(s.cagrPct) <= 1000) e.best = Math.max(e.best, s.cagrPct);
      m.set(s.timeframe, e);
    }
    return [...m.entries()].sort((a, b) => tfRank(a[0]) - tfRank(b[0]));
  }, [rows]);

  const newest = useMemo(
    () => all.reduce<number | null>((acc, s) => (s.discoveredAtMs && (!acc || s.discoveredAtMs > acc) ? s.discoveredAtMs : acc), null),
    [all],
  );

  const clearAll = () => {
    setSymFilter([]); setTfFilter([]); setModeFilter("all");
    setSearch("");
  };
  const filtersOn =
    symFilter.length > 0 || tfFilter.length > 0 || modeFilter !== "all" ||
    search.trim() !== "";

  const open = async (s: StrategyEntry) => {
    if (busy) return;
    setBusy(true);
    setRep(null);
    setReportError("");
    try {
      setRep(await strategyReport(s.dir, s.base, s.strategyId, s.exactGeneHash));
    } catch (reason) {
      setReportError(`Could not load ${s.symbol} ${s.timeframe} (${s.base}): ${reason}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Strategy Report</h1>
      <p className="sub">
        One selected representative per portfolio · unsealed IS journal · normalized 1,000-unit curve, not portfolio PnL
        {newest && <> · newest discovery <b>{stamp(newest)}</b> ({ago(newest)})</>}
      </p>
      <div className="btn-row"><button disabled={loading} onClick={() => void reload()}>{loading ? "Refreshing…" : "Refresh strategies"}</button></div>
      {error && <div className="banner warn" role="alert">Strategy inventory could not refresh. Retained entries may be stale. {error}</div>}
      {(data?.unavailable?.length ?? 0) > 0 && <div className="banner warn" role="alert">
        {data!.unavailable.length} stored reports unavailable: missing, inconsistent or ambiguous evidence is not replaced with another strategy's results.
        <details><summary>Details</summary>{data!.unavailable.map((reason, index) => <div key={index}>{reason}</div>)}</details>
      </div>}
      {reportError && <div className="banner warn" role="alert">{reportError} Retry the report from its row.</div>}
      {busy && <p className="muted" role="status">Loading the selected strategy report…</p>}

      {/* ── Filters ───────────────────────────────────────────────────────── */}
      <div className="ticket">
        <div className="ticket-row" style={{ flexWrap: "wrap", alignItems: "center", gap: 12 }}>
          <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
            Search
            <input
              value={search}
              placeholder="EURUSD, M5…"
              onChange={(e) => setSearch(e.target.value)}
              style={{ width: 150 }}
            />
          </label>
          <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
            Recorded mode
            <select value={modeFilter} onChange={(e) => setModeFilter(e.target.value as typeof modeFilter)}>
              <option value="all">All</option>
              <option value="risky">Risky</option>
              <option value="prop_firm">Prop-firm</option>
              <option value="strict">Strict</option>
              <option value="unknown">Unavailable</option>
            </select>
          </label>
          <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
            Sort
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value as SortKey)}>
              <option value="discovered">Newest first</option>
              <option value="cagr">Best CAGR</option>
              <option value="dd">Lowest drawdown</option>
              <option value="trades">Most trades</option>
              <option value="symbol">Symbol · TF</option>
            </select>
          </label>
          {filtersOn && <button className="link" onClick={clearAll}>clear filters</button>}
          <span className="muted small">{data ? `${rows.length} of ${all.length}` : "Inventory unknown"}</span>
        </div>

        <FilterChips label="Pairs" options={symbols} selected={symFilter} onToggle={toggleIn(setSymFilter)} />
        <FilterChips label="Timeframes" options={timeframes} selected={tfFilter} onToggle={toggleIn(setTfFilter)} />
      </div>

      {/* ── Per-timeframe rollup of what is currently shown ────────────────── */}
      {byTf.length > 1 && (
        <div className="cards" style={{ gridTemplateColumns: `repeat(${Math.min(6, byTf.length)}, 1fr)` }}>
          {byTf.map(([tf, e]) => (
            <div className="card" key={tf} title={`${e.n} selected-strategy IS diagnostics on ${tf}`}>
              <div className="card-label">{tf}</div>
              <div className="card-value">{e.n}</div>
              <div className="muted small">
                Unsealed IS{isFinite(e.best) ? ` · best diagnostic CAGR ${e.best.toFixed(0)}%` : ""}
              </div>
            </div>
          ))}
        </div>
      )}

      {rows.length === 0 ? (
        <p className="muted">
          {error ? "Strategy inventory unavailable." : !data ? "Loading strategy inventory…" : all.length === 0 && data.unavailable.length > 0
            ? "Stored reports are unavailable; see the reasons above."
            : all.length === 0
            ? "No strategies stored yet — run Discovery first."
            : "No strategies match the current filters."}
        </p>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>Discovered</th><th>Mode</th><th>Symbol</th><th>TF</th><th>Selected strategy</th><th>Trades</th><th>Win%</th>
              <th>CAGR%</th><th>maxDD%</th><th>1k units→</th><th>Research evidence</th><th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => (
              <tr key={JSON.stringify([s.dir, s.base, s.exactGeneHash])} className={rep?.exactGeneHash === s.exactGeneHash && rep?.base === s.base && rep?.dir === s.dir ? "row-sel" : ""}>
                <td className="muted small" style={{ whiteSpace: "nowrap" }} title={ago(s.discoveredAtMs)}>
                  {stamp(s.discoveredAtMs)}
                </td>
                <td><span className="badge" title="Recorded search mode, not live-trading status">{recordedModeLabel(s.mode)}</span></td>
                <td><b>{s.symbol}</b></td>
                <td>{s.timeframe}</td>
                <td className="mono small" title={s.exactGeneHash}>{s.strategyId}</td>
                <td>{s.trades}</td>
                <td>{s.winRate != null ? (s.winRate * 100).toFixed(1) : "—"}</td>
                <td className={s.cagrPct == null ? "muted" : s.cagrPct >= 0 ? "buy" : "sell"}>{s.cagrPct == null ? "—" : Math.abs(s.cagrPct) > 1000 ? "🚩" : s.cagrPct.toFixed(1)}</td>
                <td>{s.maxDdPct.toFixed(1)}</td>
                <td>{normalizedUnits(s.finalFrom1000)}</td>
                <td>
                  <span className="badge">Unsealed IS</span>
                  {s.flags.length > 0 && <span className="sell small" title={s.flags.join("\n")}> 🚩{s.flags.length}</span>}
                </td>
                <td><button disabled={busy} onClick={() => open(s)}>Report</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {rep && (
        <>
          <h2>{rep.symbol} {rep.timeframe} <span className="badge" title="Recorded search mode, not live-trading status">{recordedModeLabel(rep.mode)}</span></h2>
          <p className="muted small">
            <span className="mono" title={rep.exactGeneHash}>{rep.strategyId}</span> ·
            {rep.spanStart} → {rep.spanEnd} · {rep.years}y · {rep.trades} trades
            {rep.discoveredAtMs ? <> · discovered {stamp(rep.discoveredAtMs)}</> : null}
          </p>

          <div className="banner info">
            Unsealed IS diagnostics only. This report does not establish whether independent final-window
            validation passed or failed; it cannot authorize promotion. Read saved final-window research
            for an exact selected handoff in the Final evaluation tab.
          </div>
          {rep.flags.map((f, i) => <div className="banner warn" key={i}>🚩 {f}</div>)}

          <h3>Recorded evaluation settings</h3>
          <p className="muted small">Saved with this research run; today's settings are not substituted. The recorded mode is a search label, not live-trading or promotion approval.</p>
          {rep.recordedEvaluation ? <>
            <div className="cards">
              <div className="card"><div className="card-label">STARTING CAPITAL</div><div className="card-value" style={{ fontSize: 18 }}>{rep.recordedEvaluation.initialCapital.toLocaleString(undefined, { maximumFractionDigits: 2 })} {rep.recordedEvaluation.accountCurrency}</div></div>
              <div className="card"><div className="card-label">RISK PER TRADE</div><div className="card-value" style={{ fontSize: 18 }}>{recordedPercent(rep.recordedEvaluation.riskPerTradeMin)}–{recordedPercent(rep.recordedEvaluation.riskPerTradeMax)}</div><div className="muted small">Planned risk at the stop, as a fraction of entry equity. Costs and execution can change the realized loss.</div></div>
              <div className="card"><div className="card-label">FULL-RISK CONFIDENCE</div><div className="card-value" style={{ fontSize: 18 }}>{rep.recordedEvaluation.highQualityConfidence.toLocaleString(undefined, { maximumFractionDigits: 3 })}</div><div className="muted small">Signal-strength score on a 0–1 scale, not a win probability.</div></div>
            </div>
            <p className="muted small">
              Confidence measures the weighted signal's distance beyond its entry threshold, divided by the long/short threshold gap and clamped to 0–1.
              Risk scales from the recorded minimum to maximum as confidence / full-risk confidence reaches 1.
              This does not estimate the probability of a profitable trade.
            </p>
            <p className="muted small mono">Saved policy: {rep.recordedEvaluation.policyIdentityHash}</p>
          </> : <p className="muted">Unavailable: this saved policy does not record starting capital, currency, risk band or signal-confidence settings.</p>}
          {rep.recordedEvaluation?.growthGoal ? <p className="muted small">
            Reference growth goal: {rep.recordedEvaluation.growthGoal.referenceStartBalance.toLocaleString()} → {rep.recordedEvaluation.growthGoal.targetBalance.toLocaleString()} reference units in {rep.recordedEvaluation.growthGoal.horizonDays.toLocaleString()} days.
            This capital ratio guides the realized-balance pace score; it does not change the simulated starting capital.
            It is not a probability or evidence that the target was reached.
          </p> : <p className="muted small">Growth target and horizon: unavailable in the saved evaluation policy.</p>}
          <p className="muted small">Diagnostic return base: {rep.diagnosticInitialCapital.toLocaleString(undefined, { maximumFractionDigits: 2 })} account units from the selected journal; the chart below is rescaled to 1,000 units.</p>

          <div className="cards">
            <div className="card"><div className="card-label">DIAGNOSTIC CAGR</div><div className="card-value">{rep.cagrPct == null ? "—" : Math.abs(rep.cagrPct) > 1000 ? "🚩 extreme" : `${rep.cagrPct.toFixed(1)}%`}</div></div>
            <div className="card accent"><div className="card-label">1,000 UNITS →</div><div className="card-value" style={{ fontSize: 18 }}>{normalizedUnits(rep.finalFrom1000)}</div></div>
            <div className="card"><div className="card-label">MAX DD</div><div className="card-value">{rep.maxDdPct.toFixed(1)}%</div></div>
            <div className="card"><div className="card-label">WIN RATE</div><div className="card-value">{rep.winRate != null ? `${(rep.winRate * 100).toFixed(1)}%` : "—"}</div></div>
          </div>

          {rep.yearly.length > 0 && (
            <>
              <h2>Year-end balance (from 1,000 normalized units)</h2>
              <div className="ticker" style={{ flexWrap: "wrap" }}>
                {rep.yearly.map((y) => <span className="tick" key={y.month}>{y.month}: <b>{normalizedUnits(y.balance)}</b></span>)}
              </div>
              <h2>Monthly journal</h2>
              <table className="tbl">
                <thead><tr><th>Month</th><th>Return%</th><th>Balance</th><th>Trades</th></tr></thead>
                <tbody>
                  {rep.monthly.slice(-24).reverse().map((m) => (
                    <tr key={m.month}>
                      <td>{m.month}</td>
                      <td className={m.returnPct >= 0 ? "buy" : "sell"}>{m.returnPct >= 0 ? "+" : ""}{m.returnPct.toFixed(1)}</td>
                      <td className="mono">{normalizedUnits(m.balance)}</td>
                      <td>{m.trades}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="muted small">Showing last 24 months of {rep.monthly.length}.</p>
            </>
          )}
        </>
      )}
    </div>
  );
}
