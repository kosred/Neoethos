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
  type BrokerExpectedMargin,
  type BucketSummary,
  type JournalAnalytics,
} from "../api";
import { usePoll } from "../hooks";
import { useBrokerUi } from "../brokerUiContext";
import BrokerUnavailable from "../components/BrokerUnavailable";

const fmt = (v: unknown) =>
  typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(5)) : v == null ? "—" : String(v);

const fmt2 = (v: unknown) =>
  typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2)) : v == null ? "—" : String(v);

const num = (v: unknown, d = 2) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—";
const price = (v: unknown) =>
  typeof v === "number" && Number.isFinite(v) ? v.toString() : "—";
const quantity = (v: unknown) =>
  typeof v === "number" && Number.isFinite(v)
    ? v.toLocaleString(undefined, { maximumSignificantDigits: 15 }) : "—";
const fmtTime = (ms: unknown) =>
  typeof ms === "number" && ms > 0 ? new Date(ms).toLocaleString() : "—";

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
            <th>Observed price-move pips</th>
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
              <td className={b.netPips == null || !b.pipsTrades ? "muted" : b.netPips >= 0 ? "buy" : "sell"}>
                {b.pipsTrades > 0 ? num(b.netPips, 1) : "—"}
                <div className="muted small">{b.pipsTrades ?? "unknown"}/{b.trades} measured</div>
              </td>
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
        No closed trades in the local journal. This does not confirm that the broker has no
        trading history; broker reconciliation may be unavailable or incomplete.
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
          <div className="card-label">PRICE-MOVE PAYOFF (PIPS)</div>
          <div className={`card-value ${!isFinite(payoff) ? "muted" : payoff >= 2 ? "buy" : "sell"}`} style={{ fontSize: 20 }}>
            {num(payoff)}
          </div>
        </div>
        <div className="card">
          <div className="card-label">WIN RATE</div>
          <div className="card-value" style={{ fontSize: 20 }}>{Number.isFinite(winRate) ? `${num(winRate, 1)}%` : "—"}</div>
        </div>
        <div className="card">
          <div className="card-label">GROSS BREAK-EVEN WIN RATE</div>
          <div className={`card-value ${!Number.isFinite(breakEven) ? "muted" : winRate >= breakEven ? "buy" : "sell"}`} style={{ fontSize: 20 }}>
            {Number.isFinite(breakEven) ? `${num(breakEven, 1)}%` : "—"}
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
        {withPips.length} closed trades whose pip move could be computed. The gross break-even rate
        excludes commissions, swaps and other cash costs; it is not an account-profitability threshold. <b>Avg kept of best</b> is the mean capture
        ratio: how much of the favourable excursion each trade actually kept — a low number with a
        healthy MFE means winners are being given back, not that entries are wrong.
        {data.avgMfePips != null && <> Mean best-ever excursion: <b>{num(data.avgMfePips, 1)} pips</b>.</>}
      </p>

      <Bucket title="By symbol" rows={data.bySymbol} />
      {data.coverage && <div className="banner info">
        Coverage: pips {data.coverage.withPips}/{data.coverage.tradesTotal} · excursion {data.coverage.withExcursion}/{data.coverage.tradesTotal} · estimated R {data.coverage.withRMultiple}/{data.coverage.tradesTotal}.
        {data.coverage.missingEntryTime > 0 && <> {data.coverage.missingEntryTime} trades lack an entry timestamp.</>}
        {data.coverage.missingPriceSeries > 0 && <> {data.coverage.missingPriceSeries} trade windows lack usable price evidence.</>}
        {data.coverage.symbolsUsingFallbackRisk.length > 0 && <> R uses a pooled, cross-symbol risk estimate for {data.coverage.symbolsUsingFallbackRisk.join(", ")}.</>}
      </div>}
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
            <th>R estimate</th>
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
                <td className={t.rMultiple == null ? "muted" : t.rMultiple >= 0 ? "buy" : "sell"} title={`Estimated risk per lot: ${num(t.riskPerLot)} · basis: ${t.riskBasis ?? "unknown"}`}>{num(t.rMultiple)}</td>
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
        Empty cells are honest gaps, not zeros: <b>R</b> is inferred from historical losses per lot,
        not the trade's original stop risk. <b>MFE/MAE</b> are bar-based estimates and need local
        price evidence covering the trade's window. A missing excursion never
        reads as "the trade never went anywhere".
      </p>
    </>
  );
}

export default function Account() {
  const { access } = useBrokerUi();
  // Never carry a margin result or broker history into another account context.
  return <AccountContent key={access.key} brokerEnabled={access.requestsEnabled} brokerKey={access.key} />;
}

function AccountContent({ brokerEnabled, brokerKey }: { brokerEnabled: boolean; brokerKey: string }) {
  // One screen for everything account-shaped: the closed-trade journal
  // (day-to-day view), the per-trade analytics (#124), plus broker identity,
  // order history, cash flow, margin.
  const [tab, setTab] = useState<"journal" | "analytics" | "broker">("journal");

  const { data: profile, error: pe, loading: pl, reload: reloadProfile } = usePoll(brokerProfile, 0, brokerKey, brokerEnabled);
  const { data: version, error: ve, loading: vl, reload: reloadVersion } = usePoll(brokerVersion, 0, brokerKey, brokerEnabled);
  const { data: hist, error: he, loading: hl, reload: reloadHistory } = usePoll(ordersHistory, 0, brokerKey, brokerEnabled);
  const { data: cash, error: ce, loading: cl, reload: reloadCash } = usePoll(cashFlow, 0, brokerKey, brokerEnabled);
  const { data: stats, error: e1, loading: sl, reload: reloadStats } = usePoll(journalStats, 0);
  const { data: trades, error: e2, loading: tl, reload: reloadTrades } = usePoll(journalTrades, 0);
  const { data: analytics, error: e3, loading: al, reload: reloadAnalytics } = usePoll(journalAnalytics, 0);
  const refreshing = pl || vl || hl || cl || sl || tl || al;

  const [symId, setSymId] = useState("1");
  const [vol, setVol] = useState("100000");
  const [margin, setMargin] = useState<BrokerExpectedMargin | null>(null);
  const [mErr, setMErr] = useState("");
  const [marginBusy, setMarginBusy] = useState(false);
  const marginValid = Number.isSafeInteger(Number(symId)) && Number(symId) > 0
    && Number.isSafeInteger(Number(vol)) && Number(vol) > 0;

  const calcMargin = async () => {
    if (!brokerEnabled || marginBusy || !marginValid) return;
    setMarginBusy(true);
    setMErr("");
    setMargin(null);
    try {
      setMargin(await expectedMargin(Number(symId), Number(vol)));
    } catch (e) {
      setMErr(String(e));
      setMargin(null);
    } finally {
      setMarginBusy(false);
    }
  };

  const orders = hist?.orders ?? [];
  const entries = cash?.entries ?? [];
  const statEntries = stats
    ? [
        ["TOTAL TRADES", fmt2(stats.totalTrades)],
        ["WINS", fmt2(stats.wins)],
        ["LOSSES", fmt2(stats.losses)],
        ["WIN RATE", stats.totalTrades > 0 ? `${num(stats.winRatePct, 1)}%` : "—"],
        ["NET PROFIT", stats.totalTrades > 0 ? fmt2(stats.netProfit) : "—"],
        ["PROFIT FACTOR", stats.totalTrades > 0 ? num(stats.profitFactor) : "—"],
        ["EXPECTANCY", stats.totalTrades > 0 ? fmt2(stats.expectancy) : "—"],
        ["MAX DRAWDOWN", stats.totalTrades > 0 ? `${num(stats.maxDrawdownPct, 1)}%` : "—"],
      ] as const
    : [];
  const tradeRows = trades ?? [];
  // newest first
  const rows = [...tradeRows].sort(
    (a, b) => (b.exitTsMs ?? b.recordedAtUnixMs ?? 0) - (a.exitTsMs ?? a.recordedAtUnixMs ?? 0),
  );

  return (
    <div className="screen">
      <h1>Account &amp; Journal</h1>
      <p className="sub">Closed-trade log &amp; stats · per-trade pips/R/MFE · broker identity · order history · cash flow · margin</p>

      <div className="btn-row"><button disabled={refreshing} onClick={() => void Promise.all([
        reloadProfile(), reloadVersion(), reloadHistory(), reloadCash(), reloadStats(), reloadTrades(), reloadAnalytics(),
      ])}>{refreshing ? "Refreshing…" : brokerEnabled ? "Refresh account & journal" : "Refresh local journal"}</button></div>
      <BrokerUnavailable />
      {pe && <div className="banner warn" role="alert">Broker profile unavailable: {pe}</div>}
      {ve && <div className="banner warn" role="alert">Broker version unavailable: {ve}</div>}
      {he && <div className="banner warn" role="alert">Broker order history unavailable. Local journal figures do not confirm current account performance. {he}</div>}
      {ce && <div className="banner warn" role="alert">Broker cash-flow history unavailable: {ce}</div>}

      <div className="settings-grid">
        <div className="kv"><span>cTID user</span><b>{profile?.userId ?? "—"}</b></div>
        <div className="kv"><span>Broker API</span><b>v{version?.version ?? "—"}</b></div>
        <div className="kv"><span>Account</span><b>{hist?.accountId ?? "—"}</b></div>
      </div>

      <div className="seg" style={{ margin: "12px 0" }}>
        <button className={tab === "journal" ? "on" : ""} onClick={() => setTab("journal")}>Local journal</button>
        <button className={tab === "analytics" ? "on" : ""} onClick={() => setTab("analytics")}>Analytics (pips · R · MFE)</button>
        <button className={tab === "broker" ? "on" : ""} onClick={() => setTab("broker")}>Broker &amp; history</button>
      </div>

      {tab !== "broker" && <div className="banner info" role="status">
        These figures describe the local journal only. They do not establish a complete,
        broker-reconciled trading history. Missing records are not proof of zero profit or loss.
      </div>}

      {tab === "analytics" ? (
        <AnalyticsTab data={analytics} error={e3} />
      ) : tab === "journal" ? (
        <>
          {e1 && <div className="banner warn" role="alert">Journal statistics could not refresh. Retained figures may be stale. {e1}</div>}
          {e2 && <div className="banner warn" role="alert">Journal trades could not refresh. Retained rows may be stale. {e2}</div>}

          {statEntries.length > 0 && (
            <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
              {statEntries.map(([statLabel, value]) => (
                <div className="card" key={statLabel}>
                  <div className="card-label">{statLabel}</div>
                  <div className="card-value" style={{ fontSize: 18 }}>{value}</div>
                </div>
              ))}
            </div>
          )}

          <h2>Trades ({trades ? rows.length : "unknown"})</h2>
          {rows.length === 0 ? (
            <p className="muted">{e2 ? "Local closed-trade history unavailable." : !trades ? "Loading local closed trades…" : "No closed trades in the local journal. Broker history may be unavailable or not yet reconciled."}</p>
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
                  const net = typeof r.netProfit === "number" ? r.netProfit : NaN;
                  const costs = typeof r.commission === "number" && typeof r.swap === "number" ? r.commission + r.swap : NaN;
                  const buy = String(r.side ?? "").toUpperCase().includes("BUY");
                  const cls = !Number.isFinite(net) ? "" : net >= 0 ? "buy" : "sell";
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
                      <td>{!Number.isFinite(net) ? "Unknown" : net > 0 ? "✓ win" : net < 0 ? "✗ loss" : "— BE"}</td>
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
              <label>Symbol id<input disabled={!brokerEnabled || marginBusy} type="number" min={1} step={1} value={symId} onChange={(e) => { setSymId(e.target.value); setMargin(null); }} style={{ width: 80 }} /></label>
              <label>Volume (0.01 units)<input disabled={!brokerEnabled || marginBusy} type="number" min={1} step={1} value={vol} onChange={(e) => { setVol(e.target.value); setMargin(null); }} style={{ width: 160 }} /></label>
              <button className="primary" disabled={!brokerEnabled || marginBusy || !marginValid} onClick={calcMargin}>{marginBusy ? "Computing…" : "Compute"}</button>
            </div>
            <p className="muted small">This broker endpoint takes integer hundredths of a unit, not lots: 100,000 means 1,000 units. Lot size depends on the symbol.</p>
            {mErr && <div className="banner warn">{mErr}</div>}
            {margin && (
              <>
                <p className="muted small">Broker account #{margin.accountId} · symbol #{margin.symbolId ?? "unknown"}. Lots use the current broker lotSize, read at {fmtTime(margin.lotSizeObservedAtUnixMs)}.</p>
                {margin.lotSizeError && <div className="banner warn" role="alert">{margin.lotSizeError}</div>}
                <table className="tbl">
                  <thead><tr><th>Volume (units)</th><th>Volume (lots)</th><th>Buy margin</th><th>Sell margin</th></tr></thead>
                  <tbody>
                    {margin.entries.map((entry) => (
                      <tr key={entry.volumeRawCentiUnits}>
                        <td>{quantity(entry.volumeUnits)}</td>
                        <td title={`Broker lotSize: ${entry.lotSizeRawCentiUnits ?? "unavailable"} centi-units`}>{entry.lotSizeRawCentiUnits != null ? quantity(entry.volumeLots) : "—"}</td>
                        <td>{fmt(entry.buyMargin)}</td>
                        <td>{fmt(entry.sellMargin)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </div>

          <h2>Order history ({hist ? orders.length : "unknown"})</h2>
          {hist?.hasMore && <div className="banner info">The broker reports more orders in the requested interval. This response is incomplete.</div>}
          {hist?.lotSizeError && <div className="banner warn" role="alert">{hist.lotSizeError}</div>}
          {orders.length > 0 && <p className="muted small">Units are the recorded order quantities. Lots use the current broker lotSize, read at {fmtTime(hist?.lotSizeObservedAtUnixMs)}; this does not prove the contract size at the historical fill. Requested and filled volumes remain separate.</p>}
          {orders.length === 0 ? (
            <p className="muted">{!brokerEnabled ? "Broker order history is unknown until broker setup is available." : he ? "Order history unavailable." : !hist ? "Loading order history…" : "No orders in the returned history window."}</p>
          ) : (
            <div style={{ overflowX: "auto", maxHeight: 480 }}><table className="tbl">
              <thead><tr><th>Order / symbol</th><th>Side</th><th>Type</th><th>Status</th><th>Units</th><th>Lots (current contract)</th><th>Filled units</th><th>Filled lots</th><th>Limit</th><th>Stop</th></tr></thead>
              <tbody>
                {orders.map((order) => (
                  <tr key={order.orderId}>
                    <td>#{order.orderId}<div className="muted small">Symbol #{order.symbolId}</div></td>
                    <td>{order.side}</td>
                    <td>{order.orderType}</td>
                    <td>{order.orderStatus}</td>
                    <td>{quantity(order.volumeUnits)}</td>
                    <td title={`Broker lotSize: ${order.lotSizeRawCentiUnits ?? "unavailable"} centi-units`}>{order.lotSizeRawCentiUnits != null ? quantity(order.volumeLots) : "—"}</td>
                    <td>{quantity(order.executedVolumeUnits)}</td>
                    <td>{order.lotSizeRawCentiUnits != null ? quantity(order.executedVolumeLots) : "—"}</td>
                    <td>{price(order.limitPrice)}</td>
                    <td>{price(order.stopPrice)}</td>
                  </tr>
                ))}
              </tbody>
            </table></div>
          )}

          <h2>Cash flow ({cash ? entries.length : "unknown"})</h2>
          {entries.length === 0 ? (
            <p className="muted">{!brokerEnabled ? "Broker cash-flow history is unknown until broker setup is available." : ce ? "Cash-flow history unavailable." : !cash ? "Loading cash flow…" : "No deposits / withdrawals / swaps in the returned history window."}</p>
          ) : (
            <table className="tbl">
              <thead><tr><th>When</th><th>Operation</th><th>Delta</th><th>Balance</th><th>Equity</th><th>Note</th></tr></thead>
              <tbody>
                {entries.slice(0, 200).map((entry) => (
                  <tr key={entry.balanceHistoryId}>
                    <td className="muted">{fmtTime(entry.changeBalanceTimestampMs)}</td>
                    <td>{entry.operationType}</td>
                    <td className={entry.delta >= 0 ? "buy" : "sell"}>{fmt2(entry.delta)}</td>
                    <td>{fmt2(entry.balance)}</td>
                    <td>{fmt2(entry.equity)}</td>
                    <td className="muted">{entry.externalNote || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </>
      )}
    </div>
  );
}
