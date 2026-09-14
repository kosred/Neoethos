import { useEffect, useState } from "react";
import KChart from "../components/KChart";
import { KLINE_DISPLAY_INDICATORS } from "../components/chartOptions";
import PositionsTable from "../components/PositionsTable";
import {
  serverSymbols,
  brokerTimeframes,
  getWatchlist,
  setWatchlist,
  placeOrder,
  closePosition,
  refreshAccount,
  amendProtection,
  type BrokerSymbol,
  type ExecResult,
} from "../api";
import { useSpotStream, useAccountStream, usePoll } from "../hooks";
import { CANONICAL_BROKER_TIMEFRAMES } from "../timeframes";
import { useBrokerUi } from "../brokerUiContext";
import BrokerUnavailable from "../components/BrokerUnavailable";

const fmt = (v: number | undefined, d = 2) =>
  v === undefined ? "—" : v.toLocaleString(undefined, { maximumFractionDigits: d });

export default function Cockpit() {
  const { access } = useBrokerUi();
  if (!access.requestsEnabled) return (
    <div className="screen">
      <h1>Market &amp; positions</h1>
      <BrokerUnavailable />
      <p className="muted">Balance, equity and positions are unknown. Broker charts and streams are not requested until setup is available.</p>
      <button type="button" disabled>Place order</button>
    </div>
  );
  // Reset account/quote state when the selected account or environment changes.
  return <ConfiguredCockpit key={access.key} />;
}

function ConfiguredCockpit() {
  const { access } = useBrokerUi();
  const { ticks, connected, error: quoteError } = useSpotStream();
  const { snap, error: accountStreamError } = useAccountStream(access.scope);
  const { error: accountRefreshError, reload: reloadAccount } = usePoll(refreshAccount, 5000);
  const [metadataError, setMetadataError] = useState("");
  const [universe, setUniverse] = useState<BrokerSymbol[]>([]);
  const [symbol, setSymbol] = useState("EURUSD");
  const [tf, setTf] = useState("H1");
  const [tfs, setTfs] = useState<string[]>([
    ...CANONICAL_BROKER_TIMEFRAMES,
  ]);
  const [indicator, setIndicator] = useState("");
  const [filter, setFilter] = useState("");
  const [watchlist, setLocalWatchlist] = useState<Set<string>>(new Set());
  const [watchlistLoaded, setWatchlistLoaded] = useState(false);
  const [manageWatchlist, setManageWatchlist] = useState(false);
  const [watchBusy, setWatchBusy] = useState(false);
  const [watchMsg, setWatchMsg] = useState("");

  // order ticket
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [lots, setLots] = useState(0.01);
  const [sl, setSl] = useState<number | "">(20);
  const [tp, setTp] = useState<number | "">(40);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  // Modify-protection editor (merged from the old Positions screen): click
  // Edit on an open position → set SL/TP as PRICE LEVELS (breakeven, trailing).
  const [editId, setEditId] = useState<number | null>(null);
  const [editSl, setEditSl] = useState<number | "">("");
  const [editTp, setEditTp] = useState<number | "">("");
  const [editTrail, setEditTrail] = useState<"unchanged" | "enable" | "disable">("unchanged");
  const [protectionChanged, setProtectionChanged] = useState({ sl: false, tp: false });
  const positions = snap?.positions ?? [];

  useEffect(() => {
    serverSymbols().then((u) => setUniverse(u.symbols)).catch((error) => setMetadataError(`Broker symbol list unavailable: ${error}`));
    brokerTimeframes().then((r) => r.timeframes.length && setTfs(r.timeframes)).catch((error) => setMetadataError(`Broker timeframe list unavailable: ${error}`));
    getWatchlist()
      .then((result) => {
        const symbols: string[] = Array.isArray(result) ? result : (result?.symbols ?? []);
        setLocalWatchlist(new Set(symbols.map((value) => value.toUpperCase())));
      })
      .catch((error) => setWatchMsg(`Could not load watchlist: ${error}`))
      .finally(() => setWatchlistLoaded(true));
  }, []);

  const place = async () => {
    if (busy || !snap || accountRefreshError || accountStreamError || !Number.isFinite(lots) || lots <= 0) return;
    setBusy(true); setMsg("");
    try {
      const r: ExecResult = await placeOrder(symbol, side, lots, sl === "" ? undefined : Number(sl), tp === "" ? undefined : Number(tp));
      setMsg(`${r.status}${r.positionId ? ` · #${r.positionId}` : ""}${r.message ? ` · ${r.message}` : ""}`);
      await reloadAccount();
    } catch (e) { setMsg(`Error: ${e}`); } finally { setBusy(false); }
  };
  const onClose = async (id: number, vol: number) => {
    if (busy) return;
    setBusy(true);
    try {
      const response = await closePosition(id, Math.round(vol));
      setMsg(`Close response: ${response.status}${response.message ? ` · ${response.message}` : ""}`);
      await reloadAccount();
    }
    catch (e) { setMsg(`Close error: ${e}`); } finally { setBusy(false); }
  };

  // Selecting a position pre-fills the inline editor with its current stops.
  const onEdit = (positionId: number) => {
    const p = positions.find((x) => x.positionId === positionId);
    if (!p) return;
    setEditId(positionId);
    setEditSl(p.stopLoss ?? "");
    setEditTp(p.takeProfit ?? "");
    setEditTrail("unchanged");
    setProtectionChanged({ sl: false, tp: false });
  };

  const saveProtection = async () => {
    if (busy || editId == null) return;
    setBusy(true);
    setMsg("Updating SL/TP…");
    try {
      const r = await amendProtection(
        editId,
        !protectionChanged.sl || editSl === "" ? null : Number(editSl),
        !protectionChanged.tp || editTp === "" ? null : Number(editTp),
        editTrail === "unchanged" ? undefined : editTrail === "enable",
      );
      setMsg(`Protection response: ${r.status}${r.message ? ` · ${r.message}` : ""}`);
      await reloadAccount();
    } catch (e) {
      setMsg(`SL/TP update failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const editPos = editId != null ? positions.find((p) => p.positionId === editId) : undefined;

  const toggleWatch = (name: string) => {
    setLocalWatchlist((current) => {
      const next = new Set(current);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const saveWatchlist = async () => {
    setWatchBusy(true);
    setWatchMsg(`Updating ${watchlist.size} live subscriptions…`);
    try {
      await setWatchlist([...watchlist]);
      setWatchMsg(`Live watchlist updated · ${watchlist.size} symbols subscribed.`);
      setManageWatchlist(false);
    } catch (error) {
      setWatchMsg(`Watchlist update failed: ${error}`);
    } finally {
      setWatchBusy(false);
    }
  };

  const visibleUniverse = manageWatchlist || !watchlistLoaded || watchlist.size === 0
    ? universe
    : universe.filter((entry) => watchlist.has(entry.symbolName) || entry.symbolName === symbol);
  const groups: Record<string, BrokerSymbol[]> = {};
  for (const s of visibleUniverse) {
    if (filter && !s.symbolName.toUpperCase().includes(filter.toUpperCase())) continue;
    (groups[s.assetClass || "Other"] ??= []).push(s);
  }
  const cur = snap?.currency ?? "";
  const pnl = snap ? snap.equity - snap.balance : undefined;

  return (
    <div className="cockpit">
      {metadataError && <div className="banner warn" role="alert">{metadataError}</div>}
      {quoteError && <div className="banner warn" role="alert">{quoteError}</div>}
      {(accountRefreshError || accountStreamError) && <div className="banner warn" role="alert">
        <div className="section-heading-row"><span>Account values and open positions are not confirmed current. Orders remain blocked while account validation fails.</span><button onClick={() => void reloadAccount()}>Refresh account</button></div>
        <details><summary>Account refresh error details</summary><div className="break-anywhere small">{accountRefreshError || accountStreamError}</div></details>
      </div>}
      <div className="ck-grid">
        {/* Market Watch */}
        <div className="ck-watch">
          <div className="ck-watch-head">
            <span className="ck-label">Watchlist · {watchlist.size}</span>
            <button
              type="button"
              className="link"
              onClick={() => setManageWatchlist((current) => !current)}
            >
              {manageWatchlist ? "Done" : "Manage"}
            </button>
          </div>
          <input className="ck-filter" placeholder="filter…" value={filter} onChange={(e) => setFilter(e.target.value)} />
          {manageWatchlist && (
            <div className="ck-watch-actions">
              <button type="button" className="primary" disabled={watchBusy} onClick={saveWatchlist}>Save</button>
              <button type="button" disabled={watchBusy} onClick={() => setLocalWatchlist(new Set(universe.map((entry) => entry.symbolName)))}>All</button>
              <button type="button" disabled={watchBusy} onClick={() => setLocalWatchlist(new Set())}>Clear</button>
            </div>
          )}
          {watchMsg && <div className="ck-msg" role="status">{watchMsg}</div>}
          <div className="ck-watch-list">
            {Object.keys(groups).sort().map((cls) => (
              <div key={cls}>
                <div className="ck-group">{cls}</div>
                {groups[cls].map((s) => {
                  const t = ticks[s.symbolName];
                  return (
                    <div key={s.symbolId} className={`ck-sym-row${symbol === s.symbolName ? " on" : ""}`}>
                      {manageWatchlist && (
                        <input
                          type="checkbox"
                          checked={watchlist.has(s.symbolName)}
                          aria-label={`Subscribe ${s.symbolName}`}
                          onChange={() => toggleWatch(s.symbolName)}
                        />
                      )}
                      <button type="button" className="ck-sym" onClick={() => setSymbol(s.symbolName)}>
                        <span>{s.symbolName}</span>
                        <span className="mono">{t?.midPrice?.toFixed(5) ?? "—"}</span>
                      </button>
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Chart */}
        <div className="ck-chart">
          <div className="ck-chart-bar">
            <b>{symbol}</b>
            <select value={tf} onChange={(e) => setTf(e.target.value)}>
              {tfs.map((t) => <option key={t}>{t}</option>)}
            </select>
            <select value={indicator} onChange={(e) => setIndicator(e.target.value)}>
              <option value="">indicator</option>
              {KLINE_DISPLAY_INDICATORS.map((indicatorOption) => (
                <option key={indicatorOption.value} value={indicatorOption.value}>
                  {indicatorOption.label}
                </option>
              ))}
            </select>
            <span className="spacer" />
            {watchlistLoaded && !watchlist.has(symbol) && (
              <button
                type="button"
                className="link"
                disabled={watchBusy}
                onClick={() => {
                  const next = new Set(watchlist);
                  next.add(symbol);
                  setLocalWatchlist(next);
                  setWatchBusy(true);
                  setWatchMsg(`Subscribing ${symbol}…`);
                  setWatchlist([...next])
                    .then(() => setWatchMsg(`${symbol} added to the live watchlist.`))
                    .catch((error) => setWatchMsg(`Subscribe failed: ${error}`))
                    .finally(() => setWatchBusy(false));
                }}
              >
                Add live stream
              </button>
            )}
            <span className={`stream-pill ${connected ? "on" : ""}`} title="This is the UI event-stream connection, not a broker-freshness guarantee.">{connected ? "● UI update stream connected" : "○ UI update stream disconnected"}</span>
          </div>
          <div className="ck-chart-host">
            {!symbol || !tf
              ? <div className="empty">Select a symbol…</div>
              : <KChart symbol={symbol} timeframe={tf} indicator={indicator} liveTick={ticks[symbol] ?? null} />}
          </div>
        </div>

        {/* Order + Account */}
        <div className="ck-side">
          <div className="ck-order">
            <div className="ck-label">Order</div>
            <div className="seg" style={{ width: "100%" }}>
              <button className={side === "buy" ? "on buy" : ""} style={{ flex: 1 }} onClick={() => setSide("buy")}>BUY</button>
              <button className={side === "sell" ? "on sell" : ""} style={{ flex: 1 }} onClick={() => setSide("sell")}>SELL</button>
            </div>
            {/* min guards: without one the spinner arrows step below zero, and
                this ticket places REAL orders on the live account. */}
            <label className="ck-field">Lots<input type="number" min="0.01" step="0.01" value={lots} onChange={(e) => setLots(Math.max(0, Number(e.target.value)))} /></label>
            <label className="ck-field">SL pips<input type="number" min="0" value={sl} onChange={(e) => setSl(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} /></label>
            <label className="ck-field">TP pips<input type="number" min="0" value={tp} onChange={(e) => setTp(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} /></label>
            <button className="primary" style={{ width: "100%", marginTop: 6 }} disabled={busy || !snap || !!accountRefreshError || !!accountStreamError || !Number.isFinite(lots) || lots <= 0} onClick={place}>
              {busy ? "…" : `${side.toUpperCase()} ${symbol} ${lots}`}
            </button>
            {msg && <div className="ck-msg">{msg}</div>}
          </div>
          <div className="ck-account">
            <div className="ck-label">Account</div>
            <p className="muted small">{snap ? `Snapshot: ${new Date(snap.fetchedAtUnixMs).toLocaleString()}` : "No account snapshot received."}</p>
            <div className="ck-kv"><span>Balance</span><b className="mono">{fmt(snap?.balance)} {cur}</b></div>
            <div className="ck-kv"><span>Equity</span><b className="mono">{fmt(snap?.equity)} {cur}</b></div>
            <div className="ck-kv"><span>Used margin</span><b className="mono">{fmt(snap?.usedMargin)} {cur}</b></div>
            <div className="ck-kv"><span>Free margin</span><b className="mono">{fmt(snap?.freeMargin)} {cur}</b></div>
            <div className="ck-kv"><span>P/L</span><b className={`mono ${pnl !== undefined && pnl < 0 ? "sell" : "buy"}`}>{fmt(pnl)} {cur}</b></div>
          </div>
        </div>
      </div>

      {/* Trade Watch — open positions with Edit (SL/TP as price levels) + Close */}
      <div className="ck-tradewatch">
        <div className="ck-label">Positions {positions.length > 0 && <span className="badge live">{positions.length}</span>}</div>
        {editPos && (
          <div className="ticket" style={{ marginBottom: 8 }}>
            <div className="ticket-row">
              <b style={{ alignSelf: "center" }}>{editPos.symbol} {editPos.side} #{editPos.positionId}</b>
              {/* NO min here, unlike the pips fields above: these are ABSOLUTE
                  price levels, and a price can legitimately be negative on
                  commodities — WTI crude settled at -$37.63 on 2020-04-20, and
                  XTIUSD / XBRUSD / NAT.GAS are in the operator's watchlist.
                  Clamping these at zero would block a valid stop in exactly the
                  market where a stop matters most. */}
              <label>SL price<input type="number" step="0.00001" value={editSl} onChange={(e) => { setEditSl(e.target.value === "" ? "" : Number(e.target.value)); setProtectionChanged((current) => ({ ...current, sl: true })); }} /></label>
              <label>TP price<input type="number" step="0.00001" value={editTp} onChange={(e) => { setEditTp(e.target.value === "" ? "" : Number(e.target.value)); setProtectionChanged((current) => ({ ...current, tp: true })); }} /></label>
              <label>Broker trailing
                <select value={editTrail} onChange={(e) => setEditTrail(e.target.value as typeof editTrail)}>
                  <option value="unchanged">Leave unchanged</option><option value="enable">Enable</option><option value="disable">Disable</option>
                </select>
              </label>
              <button className="primary" disabled={busy || (!protectionChanged.sl && !protectionChanged.tp)} onClick={saveProtection}>Update SL/TP</button>
              <button disabled={busy} onClick={() => setEditId(null)}>Cancel</button>
            </div>
            <p className="muted small">Only edited prices are sent. Blank means leave unchanged, not remove protection. To change broker trailing, explicitly edit the SL price as well.</p>
          </div>
        )}
        {snap ? <PositionsTable live={positions} currency={cur} onClose={onClose} onEdit={onEdit} busy={busy} />
          : <p className="banner warn">Open positions are unknown until an account snapshot is received.</p>}
      </div>
    </div>
  );
}
