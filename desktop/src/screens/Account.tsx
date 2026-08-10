import { useState } from "react";
import {
  brokerProfile,
  brokerVersion,
  ordersHistory,
  cashFlow,
  expectedMargin,
  journalStats,
  journalTrades,
  journalAnalytics,
  type BucketSummary,
  type JournalAnalytics,
} from "../api";
import { usePoll } from "../hooks";

const fmt = (v: unknown) =>
  typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(5)) : v == null ? "—" : String(v);

const fmt2 = (v: unknown) =>
  typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2)) : v == null ? "—" : String(v);

const num = (v: any, d = 2) => (typeof v === "number" && isFinite(v) ? v.toFixed(d) : "—");
const price = (v: any) => (typeof v === "number" && isFinite(v) ? v.toString() : "—");
const fmtTime = (ms: any) => (typeof ms === "number" && ms > 0 ? new Date(ms).toLocaleString() : "—");

// "winRatePct" / "max_drawdown_pct" → "WIN RATE PCT"
const label = (k: string) =>
  k.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/_/g, " ").toUpperCase();

// Audit #124 — the per-trade analytics table.
//
// `GET /journal/analytics` has existed since 2026-07-30 and its ONLY caller
// repo-wide was `mcp/ops.rs:864`: an LLM tool call. The operator could not
// reach it from anywhere in the app. This is the view that reports realised
// PAYOFF in pips and R per trade, plus MFE/MAE — i.e. the numbers that say
// "the winners were there and we gave them back". A total P/L cannot say that,
// which is why the 1.08-vs-2.0 payoff gap survived sixteen months.
function Bucket({ title, rows, note }: { title: string; rows: BucketSummary[]; note?: string }) {
  if (!rows || rows.length === 0) return null;
  return (
    <>
      <h2>{title}</h2>
      {note && <p className="muted small">{note}</p>}
      <table className="tbl">
        <thead>
          <tr>
            <th>{title.replace(/^By /, "")}</th>
            <th>Trades</th>
            <th>Win %</th>
            <th>Expectancy / trade</th>
            <th>Net pips</th>
            <th>Net P/L</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((b) => (
            <tr key={b.bucket}>
              <td><b>{b.bucket}</b></td>
              <td>{b.trades}</td>
              <td>{num(b.winRatePct, 1)}%</td>
              <td className={b.expectancy >= 0 ? "buy" : "sell"}>{num(b.expectancy)}</td>
              <td className={b.netPips >= 0 ? "buy" : "sell"}>{num(b.netPips, 1)}</td>
              <td className={b.netProfit >= 0 ? "buy" : "sell"}>{num(b.netProfit)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function AnalyticsTab({ data, error }: { data: JournalAnalytics | null; error?: unknown }) {
  if (error) return <div className="banner warn">{String(error)}</div>;
  if (!data) return <p className="muted">Loading analytics…</p>;
  const trades = data.trades ?? [];
  if (trades.length === 0) {
    return (
      <p className="muted">
        No closed trades recorded yet — this view derives everything from the journal, so it fills
        in as trades close.
      </p>
    );
  }
  // Realised payoff, computed from the same rows shown below, so the headline
  // number and the table can never disagree. In PIPS, because that is the unit
  // the backtest's payoff floor is expressed in — money-weighted averages hide
  // it behind position size (memory: "the payoff 2.21 was WRONG; in pips 1.08").
  const withPips = trades.filter((t) => typeof t.pips === "number" && isFinite(t.pips as number));
  const winPips = withPips.filter((t) => (t.pips as number) > 0).map((t) => t.pips as number);
  const lossPips = withPips.filter((t) => (t.pips as number) < 0).map((t) => -(t.pips as number));
  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
  const avgWin = mean(winPips);
  const avgLoss = mean(lossPips);
  const payoff = isFinite(avgWin) && isFinite(avgLoss) && avgLoss > 0 ? avgWin / avgLoss : NaN;
  const winRate = withPips.length ? (winPips.length / withPips.length) * 100 : NaN;
  // The win rate this payoff needs just to break even. Below it, more trades
  // lose more money.
  const breakEven = isFinite(payoff) ? 100 / (1 + payoff) : NaN;

  return (
    <>
      <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
        <div className="card">
          <div className="card-label">REALISED PAYOFF (PIPS)</div>
          <div className={`card-value ${isFinite(payoff) && payoff >= 2 ? "buy" : "sell"}`} style={{ fontSize: 20 }}>
            {num(payoff)}
          </div>
        </div>
        <div className="card">
          <div className="card-label">WIN RATE</div>
          <div className="card-value" style={{ fontSize: 20 }}>{num(winRate, 1)}%</div>
        </div>
        <div className="card">
          <div className="card-label">BREAK-EVEN WIN RATE</div>
          <div className={`card-value ${winRate >= breakEven ? "buy" : "sell"}`} style={{ fontSize: 20 }}>
            {num(breakEven, 1)}%
          </div>
        </div>
        <div className="card">
          <div className="card-label">AVG KEPT OF BEST (MFE)</div>
          <div className="card-value" style={{ fontSize: 20 }}>
            {data.avgCaptureRatio != null ? `${num(data.avgCaptureRatio * 100, 0)}%` : "—"}
          </div>
        </div>
      </div>
      <p className="muted small">
        Payoff = average winning move ÷ average losing move, in <b>pips</b>, over the{" "}
        {withPips.length} closed trades whose pip move could be computed. Break-even win rate is
        what that payoff needs to stop losing money. <b>Avg kept of best</b> is the mean capture
        ratio: how much of the favourable excursion each trade actually kept — a low number with a
        healthy MFE means winners are being given back, not that entries are wrong.
        {data.avgMfePips != null && <> Mean best-ever excursion: <b>{num(data.avgMfePips, 1)} pips</b>.</>}
      </p>

      <Bucket title="By symbol" rows={data.bySymbol} />
      <Bucket title="By direction" rows={data.bySide} />
      <Bucket
        title="By entry hour (UTC)"
        rows={data.byHourUtc}
        note={
          data.inactiveHoursUtc?.length
            ? `Never traded in ${data.inactiveHoursUtc.length} of 24 hours: ${data.inactiveHoursUtc.join(", ")} UTC.`
            : undefined
        }
      />
      <Bucket title="By weekday" rows={data.byWeekday} />

      <h2>Per trade ({trades.length})</h2>
      <table className="tbl">
        <thead>
          <tr>
            <th>Closed</th>
            <th>Symbol</th>
            <th>Side</th>
            <th>Pips</th>
            <th>R</th>
            <th>MFE (pips)</th>
            <th>MAE (pips)</th>
            <th>Kept of best</th>
            <th>Held (h)</th>
            <th>Net P/L</th>
          </tr>
        </thead>
        <tbody>
          {[...trades]
            .sort((a, b) => (b.exitTsMs ?? 0) - (a.exitTsMs ?? 0))
            .slice(0, 300)
            .map((t, i) => (
              <tr key={t.positionId ?? i}>
                <td className="muted">{fmtTime(t.exitTsMs)}</td>
                <td><b>{t.symbol}</b></td>
                <td className={String(t.side).toUpperCase().includes("BUY") ? "buy" : "sell"}>{t.side}</td>
                <td className={(t.pips ?? 0) >= 0 ? "buy" : "sell"}>{num(t.pips, 1)}</td>
                <td className={(t.rMultiple ?? 0) >= 0 ? "buy" : "sell"}>{num(t.rMultiple)}</td>
                <td>{num(t.mfePips, 1)}</td>
                <td>{num(t.maePips, 1)}</td>
                <td>{t.captureRatio != null ? `${num(t.captureRatio * 100, 0)}%` : "—"}</td>
                <td className="muted">{num(t.durationHours, 1)}</td>
                <td className={t.netProfit >= 0 ? "buy" : "sell"}>{num(t.netProfit)}</td>
              </tr>
            ))}
        </tbody>
      </table>
      <p className="muted small">
        Empty cells are honest gaps, not zeros: <b>R</b> needs a risk estimate for the symbol, and{" "}
        <b>MFE/MAE</b> need local price bars covering the trade's window. A missing excursion never
        reads as "the trade never went anywhere".
      </p>
    </>
  );
}

export default function Account() {
  // One screen for everything account-shaped: the closed-trade journal
  // (day-to-day view), the per-trade analytics (#124), plus broker identity,
  // order history, cash flow, margin.
  const [tab, setTab] = useState<"journal" | "analytics" | "broker">("journal");

  const { data: profile } = usePoll(brokerProfile, 0);
  const { data: version } = usePoll(brokerVersion, 0);
  const { data: hist, error: he } = usePoll(ordersHistory, 0);
  const { data: cash } = usePoll(cashFlow, 0);
  const { data: stats, error: e1 } = usePoll(journalStats, 0);
  const { data: trades, error: e2 } = usePoll(journalTrades, 0);
  const { data: analytics, error: e3 } = usePoll(journalAnalytics, 0);

  const [symId, setSymId] = useState("1");
  const [vol, setVol] = useState("100000");
  const [margin, setMargin] = useState<any>(null);
  const [mErr, setMErr] = useState("");

  const calcMargin = async () => {
    setMErr("");
    try {
      setMargin(await expectedMargin(Number(symId), Number(vol)));
    } catch (e) {
      setMErr(String(e));
      setMargin(null);
    }
  };

  const orders: any[] = hist?.orders ?? [];
  const ocols = orders.length ? ["orderId", "side", "orderType", "orderStatus", "volumeLots", "limitPrice", "stopPrice"] : [];
  const entries: any[] = cash?.entries ?? [];

  const statEntries =
    stats && typeof stats === "object"
      ? Object.entries(stats).filter(([, v]) => typeof v !== "object" || v === null)
      : [];
  const tradeRows: any[] = Array.isArray(trades) ? trades : (trades?.trades ?? []);
  // newest first
  const rows = [...tradeRows].sort(
    (a, b) => (b.exitTsMs ?? b.recordedAtUnixMs ?? 0) - (a.exitTsMs ?? a.recordedAtUnixMs ?? 0),
  );

  return (
    <div className="screen">
      <h1>Account &amp; Journal</h1>
      <p className="sub">Closed-trade log &amp; stats · per-trade pips/R/MFE · broker identity · order history · cash flow · margin</p>

      <div className="settings-grid">
        <div className="kv"><span>cTID user</span><b>{profile?.userId ?? "—"}</b></div>
        <div className="kv"><span>Broker API</span><b>v{version?.version ?? "—"}</b></div>
        <div className="kv"><span>Account</span><b>{hist?.accountId ?? "—"}</b></div>
      </div>

      <div className="seg" style={{ margin: "12px 0" }}>
        <button className={tab === "journal" ? "on" : ""} onClick={() => setTab("journal")}>Journal</button>
        <button className={tab === "analytics" ? "on" : ""} onClick={() => setTab("analytics")}>Analytics (pips · R · MFE)</button>
        <button className={tab === "broker" ? "on" : ""} onClick={() => setTab("broker")}>Broker &amp; history</button>
      </div>

      {tab === "analytics" ? (
        <AnalyticsTab data={analytics} error={e3} />
      ) : tab === "journal" ? (
        <>
          {(e1 || e2) && <div className="banner warn">{e1 || e2}</div>}

          {statEntries.length > 0 && (
            <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
              {statEntries.slice(0, 8).map(([k, v]) => (
                <div className="card" key={k}>
                  <div className="card-label">{label(k)}</div>
                  <div className="card-value" style={{ fontSize: 18 }}>{fmt2(v)}</div>
                </div>
              ))}
            </div>
          )}

          <h2>Trades ({rows.length})</h2>
          {rows.length === 0 ? (
            <p className="muted">No closed trades recorded yet.</p>
          ) : (
            <table className="tbl">
              <thead>
                <tr>
                  <th>Closed</th>
                  <th>Symbol</th>
                  <th>Side</th>
                  <th>Lots</th>
                  <th>Entry</th>
                  <th>Exit</th>
                  <th>Costs</th>
                  <th>Net P/L</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {rows.slice(0, 300).map((r, i) => {
                  const net = Number(r.netProfit ?? 0);
                  const costs = Number(r.commission ?? 0) + Number(r.swap ?? 0);
                  const buy = String(r.side ?? "").toUpperCase().includes("BUY");
                  const cls = net >= 0 ? "buy" : "sell";
                  return (
                    <tr key={r.positionId ?? i}>
                      <td className="muted">{fmtTime(r.exitTsMs ?? r.recordedAtUnixMs)}</td>
                      <td><b>{r.symbol ?? "?"}</b></td>
                      <td className={buy ? "buy" : "sell"}>{r.side ?? "—"}</td>
                      <td>{num(r.lots)}</td>
                      <td>{price(r.entryPrice)}</td>
                      <td>{price(r.exitPrice)}</td>
                      <td className="muted">{num(costs)}</td>
                      <td className={cls}><b>{net >= 0 ? "+" : ""}{num(net)}</b></td>
                      <td>{net > 0 ? "✓ win" : net < 0 ? "✗ loss" : "— BE"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </>
      ) : (
        <>
          <h2>Margin calculator</h2>
          <div className="ticket">
            <div className="ticket-row">
              <label>Symbol id<input value={symId} onChange={(e) => setSymId(e.target.value)} style={{ width: 80 }} /></label>
              <label>Volume (units)<input value={vol} onChange={(e) => setVol(e.target.value)} style={{ width: 120 }} /></label>
              <button className="primary" onClick={calcMargin}>Compute</button>
            </div>
            {mErr && <div className="banner warn">{mErr}</div>}
            {margin && (
              <table className="tbl">
                <tbody>
                  {Object.entries(margin).map(([k, v]) => (
                    <tr key={k}><td style={{ color: "#9ca3af" }}>{k}</td><td>{fmt(v)}</td></tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          <h2>Order history ({orders.length})</h2>
          {he && <div className="banner warn">{he}</div>}
          {orders.length === 0 ? (
            <p className="muted">No order history.</p>
          ) : (
            <table className="tbl">
              <thead><tr>{ocols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {orders.slice(0, 200).map((o, i) => (
                  <tr key={i}>{ocols.map((c) => <td key={c}>{fmt(o[c])}</td>)}</tr>
                ))}
              </tbody>
            </table>
          )}

          <h2>Cash flow ({entries.length})</h2>
          {entries.length === 0 ? (
            <p className="muted">No deposits / withdrawals / swaps recorded.</p>
          ) : (
            <table className="tbl">
              <thead><tr>{Object.keys(entries[0]).map((c) => <th key={c}>{c}</th>)}</tr></thead>
              <tbody>
                {entries.slice(0, 200).map((e, i) => (
                  <tr key={i}>{Object.keys(entries[0]).map((c) => <td key={c}>{fmt(e[c])}</td>)}</tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}
