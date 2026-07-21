import { useMemo, useState } from "react";
import { strategyList, strategyReport, type StrategyEntry, type StrategyReport as Report } from "../api";
import { usePoll } from "../hooks";

const eur = (v: number) => `€${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
const Badge = ({ ok, label }: { ok: boolean | null; label: string }) => (
  <span className={`badge ${ok ? "demo" : "live"}`} title={label}>{ok ? "✓" : "✗"} {label}</span>
);

// "2026-07-21 01:08" — sortable at a glance, no locale surprises.
const stamp = (ms: number | null) => {
  if (!ms) return "—";
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
};
const ago = (ms: number | null) => {
  if (!ms) return "";
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
};

// Fastest-first, so timeframe filters read in the order runs actually go.
const TF_ORDER = ["MN1", "W1", "D1", "H12", "H8", "H6", "H4", "H3", "H2", "H1", "M30", "M20", "M15", "M12", "M10", "M6", "M5", "M4", "M3", "M2", "M1"];
const tfRank = (t: string) => {
  const i = TF_ORDER.indexOf(t);
  return i < 0 ? 999 : i;
};

type SortKey = "discovered" | "cagr" | "dd" | "trades" | "symbol";

export default function StrategyReport() {
  const { data, error } = usePoll(strategyList, 0);
  const [rep, setRep] = useState<Report | null>(null);
  const [busy, setBusy] = useState(false);

  // ── Filters (operator request: "I can't tell what happened per pair / per
  // timeframe, and I can't see WHEN anything was discovered") ──────────────
  const [symFilter, setSymFilter] = useState<string[]>([]);
  const [tfFilter, setTfFilter] = useState<string[]>([]);
  const [modeFilter, setModeFilter] = useState<"all" | "risky" | "prop_firm">("all");
  const [validOnly, setValidOnly] = useState(false);
  const [hideFlagged, setHideFlagged] = useState(false);
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
      if (validOnly && !(s.cpcvPassed && s.walkforwardPassed)) return false;
      if (hideFlagged && s.flags.length > 0) return false;
      if (q && !`${s.symbol} ${s.timeframe} ${s.mode}`.toUpperCase().includes(q)) return false;
      return true;
    });
    const cmp: Record<SortKey, (a: StrategyEntry, b: StrategyEntry) => number> = {
      discovered: (a, b) => (b.discoveredAtMs ?? 0) - (a.discoveredAtMs ?? 0),
      cagr: (a, b) => b.cagrPct - a.cagrPct,
      dd: (a, b) => a.maxDdPct - b.maxDdPct,
      trades: (a, b) => b.trades - a.trades,
      symbol: (a, b) => a.symbol.localeCompare(b.symbol) || tfRank(a.timeframe) - tfRank(b.timeframe),
    };
    return [...out].sort(cmp[sortBy]);
  }, [all, symFilter, tfFilter, modeFilter, validOnly, hideFlagged, search, sortBy]);

  // Per-timeframe rollup of the FILTERED set — answers "what is happening per
  // group" without reading every row.
  const byTf = useMemo(() => {
    const m = new Map<string, { n: number; valid: number; best: number }>();
    for (const s of rows) {
      const e = m.get(s.timeframe) ?? { n: 0, valid: 0, best: -Infinity };
      e.n += 1;
      if (s.cpcvPassed && s.walkforwardPassed) e.valid += 1;
      if (Math.abs(s.cagrPct) <= 1000) e.best = Math.max(e.best, s.cagrPct);
      m.set(s.timeframe, e);
    }
    return [...m.entries()].sort((a, b) => tfRank(a[0]) - tfRank(b[0]));
  }, [rows]);

  const newest = useMemo(
    () => all.reduce<number | null>((acc, s) => (s.discoveredAtMs && (!acc || s.discoveredAtMs > acc) ? s.discoveredAtMs : acc), null),
    [all],
  );

  const toggle = (set: React.Dispatch<React.SetStateAction<string[]>>) => (v: string) =>
    set((cur) => (cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v]));
  const clearAll = () => {
    setSymFilter([]); setTfFilter([]); setModeFilter("all");
    setValidOnly(false); setHideFlagged(false); setSearch("");
  };
  const filtersOn =
    symFilter.length > 0 || tfFilter.length > 0 || modeFilter !== "all" ||
    validOnly || hideFlagged || search.trim() !== "";

  const open = async (s: StrategyEntry) => {
    setBusy(true);
    try {
      setRep(await strategyReport(s.dir, s.base));
    } catch {
      setRep(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Strategy Report</h1>
      <p className="sub">
        Monthly journal · €1000 growth · validation verdict · honest flags — from the stored backtests
        {newest && <> · newest discovery <b>{stamp(newest)}</b> ({ago(newest)})</>}
      </p>
      {error && <div className="banner warn">{error}</div>}

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
            Mode
            <select value={modeFilter} onChange={(e) => setModeFilter(e.target.value as typeof modeFilter)}>
              <option value="all">All</option>
              <option value="risky">🚀 Risky</option>
              <option value="prop_firm">🛡 Prop-firm</option>
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
          <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }} title="Passed CPCV + Walkforward out-of-sample">
            <input type="checkbox" checked={validOnly} onChange={(e) => setValidOnly(e.target.checked)} /> Validated only
          </label>
          <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }} title="Hide anything carrying an honesty flag">
            <input type="checkbox" checked={hideFlagged} onChange={(e) => setHideFlagged(e.target.checked)} /> Hide flagged
          </label>
          {filtersOn && <button className="link" onClick={clearAll}>clear filters</button>}
          <span className="muted small">{rows.length} of {all.length}</span>
        </div>

        {symbols.length > 1 && (
          <>
            <div className="muted small" style={{ marginTop: 8 }}>
              Pairs <span className="muted">({symFilter.length || "all"})</span>
            </div>
            <div className="chip-row">
              {symbols.map((s) => (
                <button key={s} type="button" className={`chip ${symFilter.includes(s) ? "on" : ""}`} onClick={() => toggle(setSymFilter)(s)}>
                  {s}
                </button>
              ))}
            </div>
          </>
        )}
        {timeframes.length > 1 && (
          <>
            <div className="muted small" style={{ marginTop: 8 }}>
              Timeframes <span className="muted">({tfFilter.length || "all"})</span>
            </div>
            <div className="chip-row">
              {timeframes.map((t) => (
                <button key={t} type="button" className={`chip ${tfFilter.includes(t) ? "on" : ""}`} onClick={() => toggle(setTfFilter)(t)}>
                  {t}
                </button>
              ))}
            </div>
          </>
        )}
      </div>

      {/* ── Per-timeframe rollup of what is currently shown ────────────────── */}
      {byTf.length > 1 && (
        <div className="cards" style={{ gridTemplateColumns: `repeat(${Math.min(6, byTf.length)}, 1fr)` }}>
          {byTf.map(([tf, e]) => (
            <div className="card" key={tf} title={`${e.n} strategies on ${tf}, ${e.valid} passed out-of-sample`}>
              <div className="card-label">{tf}</div>
              <div className="card-value">{e.n}</div>
              <div className="muted small">
                {e.valid} validated{isFinite(e.best) ? ` · best ${e.best.toFixed(0)}%` : ""}
              </div>
            </div>
          ))}
        </div>
      )}

      {rows.length === 0 ? (
        <p className="muted">
          {all.length === 0
            ? "No strategies stored yet — run Discovery first."
            : "No strategies match the current filters."}
        </p>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>Discovered</th><th>Mode</th><th>Symbol</th><th>TF</th><th>Trades</th><th>Win%</th>
              <th>CAGR%</th><th>maxDD%</th><th>€1k→</th><th>Validation</th><th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((s) => (
              <tr key={s.dir + s.base} className={rep?.base === s.base && rep?.dir === s.dir ? "row-sel" : ""}>
                <td className="muted small" style={{ whiteSpace: "nowrap" }} title={ago(s.discoveredAtMs)}>
                  {stamp(s.discoveredAtMs)}
                </td>
                <td><span className={`badge ${s.mode === "risky" ? "live" : "demo"}`}>{s.mode}</span></td>
                <td><b>{s.symbol}</b></td>
                <td>{s.timeframe}</td>
                <td>{s.trades}</td>
                <td>{s.winRate != null ? (s.winRate * 100).toFixed(1) : "—"}</td>
                <td className={s.cagrPct >= 0 ? "buy" : "sell"}>{Math.abs(s.cagrPct) > 1000 ? "🚩" : s.cagrPct.toFixed(1)}</td>
                <td>{s.maxDdPct.toFixed(1)}</td>
                <td>{Math.abs(s.cagrPct) > 1000 ? "—" : eur(s.finalFrom1000)}</td>
                <td>
                  <span className={`badge ${s.cpcvPassed ? "demo" : "live"}`} title="CPCV">C{s.cpcvPassed ? "✓" : "✗"}</span>{" "}
                  <span className={`badge ${s.walkforwardPassed ? "demo" : "live"}`} title="Walkforward (OOS)">W{s.walkforwardPassed ? "✓" : "✗"}</span>
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
          <h2>{rep.symbol} {rep.timeframe} <span className={`badge ${rep.mode === "risky" ? "live" : "demo"}`}>{rep.mode}</span></h2>
          <p className="muted small">
            {rep.spanStart} → {rep.spanEnd} · {rep.years}y · {rep.trades} trades
            {rep.discoveredAtMs ? <> · discovered {stamp(rep.discoveredAtMs)}</> : null}
          </p>

          {(() => {
            const oos = !!rep.cpcvPassed && !!rep.walkforwardPassed;
            const full = oos && !!rep.validationComplete;
            const txt = full
              ? "✅ FULLY VALIDATED — passed CPCV + Walkforward, evidence complete"
              : oos
                ? "✅ PASSED out-of-sample (CPCV + Walkforward) — full-evidence check still pending"
                : "⚠️ NOT validated out-of-sample — not safe to trade live";
            return (
              <div className={`banner ${oos ? "info" : "warn"}`} style={{ fontWeight: 600 }}>
                {txt}
              </div>
            );
          })()}

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", margin: "8px 0" }}>
            <Badge ok={rep.cpcvPassed} label="CPCV" />
            <Badge ok={rep.walkforwardPassed} label="Walkforward (out-of-sample)" />
            <Badge ok={rep.validationComplete} label="Full evidence" />
          </div>
          {rep.flags.map((f, i) => <div className="banner warn" key={i}>🚩 {f}</div>)}

          <div className="cards">
            <div className="card"><div className="card-label">CAGR</div><div className="card-value">{Math.abs(rep.cagrPct) > 1000 ? "🚩 bug" : `${rep.cagrPct.toFixed(1)}%`}</div></div>
            <div className="card accent"><div className="card-label">€1000 →</div><div className="card-value" style={{ fontSize: 18 }}>{Math.abs(rep.cagrPct) > 1000 ? "—" : eur(rep.finalFrom1000)}</div></div>
            <div className="card"><div className="card-label">MAX DD</div><div className="card-value">{rep.maxDdPct.toFixed(1)}%</div></div>
            <div className="card"><div className="card-label">WIN RATE</div><div className="card-value">{rep.winRate != null ? `${(rep.winRate * 100).toFixed(1)}%` : "—"}</div></div>
          </div>

          {Math.abs(rep.cagrPct) <= 1000 && rep.yearly.length > 0 && (
            <>
              <h2>Year-end balance (from €1000)</h2>
              <div className="ticker" style={{ flexWrap: "wrap" }}>
                {rep.yearly.map((y) => <span className="tick" key={y.month}>{y.month}: <b>{eur(y.balance)}</b></span>)}
              </div>
              <h2>Monthly journal</h2>
              <table className="tbl">
                <thead><tr><th>Month</th><th>Return%</th><th>Balance</th><th>Trades</th></tr></thead>
                <tbody>
                  {rep.monthly.slice(-24).reverse().map((m) => (
                    <tr key={m.month}>
                      <td>{m.month}</td>
                      <td className={m.returnPct >= 0 ? "buy" : "sell"}>{m.returnPct >= 0 ? "+" : ""}{m.returnPct.toFixed(1)}</td>
                      <td className="mono">{eur(m.balance)}</td>
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
