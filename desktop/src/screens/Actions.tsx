import { useEffect, useState } from "react";
import {
  pendingActions,
  confirmAction,
  rejectAction,
  brokerPendingOrders,
  placePendingOrder,
  cancelOrder,
  streamSpots,
  type PendingOrder,
  type Tick,
} from "../api";
import { usePoll } from "../hooks";
import { SymbolSelect } from "../components/Select";
import { HelpPanel, HelpStep, Tip } from "../components/Help";

const price = (v: any) => (typeof v === "number" && isFinite(v) ? String(v) : "—");
const fmtTime = (ms: any) => (typeof ms === "number" && ms > 0 ? new Date(ms).toLocaleString() : "—");

export default function Actions() {
  // Broker-side resting (limit/stop) orders — the "trade when price hits X" list.
  const { data: pendingData, error: pErr, reload: reloadPending } = usePoll(brokerPendingOrders, 5000);
  // AI-approval queue (LLM-proposed close actions) — kept for when a proposer fires.
  const { data: actionsData, error: aErr, reload: reloadActions } = usePoll(pendingActions, 3000);

  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  // New conditional-order form.
  const [symbol, setSymbol] = useState("EURUSD");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [otype, setOtype] = useState<"limit" | "stop">("limit");
  const [lots, setLots] = useState(0.01);
  const [trigger, setTrigger] = useState<number | "">("");
  const [sl, setSl] = useState<number | "">(20);
  const [tp, setTp] = useState<number | "">(40);
  const [expiry, setExpiry] = useState(""); // datetime-local; empty = Good-Till-Cancel

  // Live prices to anchor the trigger against the current market.
  const [ticks, setTicks] = useState<Record<string, Tick>>({});
  useEffect(() => {
    let stop: (() => void) | undefined;
    let disposed = false;
    streamSpots((t) => setTicks((m) => ({ ...m, [t.symbolName]: t }))).then((fn) => {
      if (disposed) fn();
      else stop = fn;
    });
    return () => {
      disposed = true;
      stop?.();
    };
  }, []);
  const spot = ticks[symbol.toUpperCase()];

  const orders: PendingOrder[] = Array.isArray(pendingData) ? pendingData : [];
  const actions: any[] = Array.isArray(actionsData) ? actionsData : (actionsData?.actions ?? actionsData?.pending ?? []);
  const liveActions = actions.filter((a) => (a.status ?? "pending") === "pending");

  // Non-blocking sanity hint: which side of the market this order type usually rests.
  const dirHint = (() => {
    if (!spot || trigger === "" || !(Number(trigger) > 0)) return null;
    const px = spot.midPrice;
    const t = Number(trigger);
    const wantAbove = (side === "buy" && otype === "stop") || (side === "sell" && otype === "limit");
    const wantBelow = (side === "buy" && otype === "limit") || (side === "sell" && otype === "stop");
    if (wantAbove && t <= px) return { warn: true, text: `⚠ A ${side} ${otype} normally triggers ABOVE the current price (now ${px}).` };
    if (wantBelow && t >= px) return { warn: true, text: `⚠ A ${side} ${otype} normally triggers BELOW the current price (now ${px}).` };
    return { warn: false, text: `✓ Trigger is ${t > px ? "above" : "below"} the market (now ${px}).` };
  })();

  const submit = async () => {
    if (trigger === "" || !(Number(trigger) > 0)) {
      setMsg("Set a trigger price first.");
      return;
    }
    setBusy(true);
    setMsg("Placing conditional order…");
    try {
      const r: any = await placePendingOrder({
        symbol: symbol.toUpperCase(),
        side,
        orderType: otype,
        volumeLots: lots,
        triggerPrice: Number(trigger),
        stopLossPips: sl === "" ? null : Number(sl),
        takeProfitPips: tp === "" ? null : Number(tp),
        expiryUnixMs: expiry ? new Date(expiry).getTime() : null,
      });
      setMsg(`✓ ${r.status ?? "placed"}${r.orderId ? ` · order #${r.orderId}` : ""}${r.message ? ` · ${r.message}` : ""}`);
      await reloadPending();
    } catch (e) {
      setMsg(`Failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (orderId: number) => {
    setBusy(true);
    setMsg(`Cancelling order #${orderId}…`);
    try {
      await cancelOrder(orderId);
      setMsg(`✓ cancelled order #${orderId}`);
      await reloadPending();
    } catch (e) {
      setMsg(`Cancel failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const decide = async (id: string, ok: boolean) => {
    setBusy(true);
    setMsg(ok ? `Confirming ${id}…` : `Rejecting ${id}…`);
    try {
      await (ok ? confirmAction(id) : rejectAction(id));
      setMsg(`✓ ${ok ? "confirmed" : "rejected"} ${id}`);
      await reloadActions();
    } catch (e) {
      setMsg(`Failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>
        Actions
        {orders.length > 0 && <span className="badge live" style={{ marginLeft: 8 }}>{orders.length} resting</span>}
        {liveActions.length > 0 && <span className="badge live" style={{ marginLeft: 8 }}>{liveActions.length} to approve</span>}
      </h1>
      <p className="sub">Place a trade that fires when the price hits your level · manage resting orders · approve AI proposals</p>

      <HelpPanel id="actions">
        <p>A <b>conditional (pending) order</b> rests at the broker and fills automatically the moment the market reaches your <b>trigger price</b> — you don't have to be watching. It survives closing the app.</p>
        <HelpStep n={1}><b>Limit</b> = enter at a <i>better</i> price than now (BUY below / SELL above the market). <b>Stop</b> = enter on a <i>breakout</i> (BUY above / SELL below).</HelpStep>
        <HelpStep n={2}>Set Symbol, side, lots, the <b>trigger price</b>, and optional SL/TP in pips. Leave expiry blank for Good-Till-Cancel.</HelpStep>
        <HelpStep n={3}>Resting orders appear below — <b>Cancel</b> any that haven't filled. The broker validates the price/side combination and rejects invalid ones with a reason.</HelpStep>
        <p className="muted small">The <b>AI approvals</b> section stays empty unless the assistant proposes a trade-management action for your one-click confirmation. Automated entries live in <b>Autopilot</b>.</p>
      </HelpPanel>

      {msg && <div className="banner info">{msg}</div>}

      {/* ── Place a conditional order ── */}
      <h2>New conditional order <Tip text="A limit/stop order the broker holds until the market reaches your trigger price, then fills automatically." /></h2>
      <div className="ticket">
        <div className="ticket-row" style={{ flexWrap: "wrap", gap: 12 }}>
          <label>
            Symbol
            <SymbolSelect value={symbol} onChange={setSymbol} style={{ width: 120 }} />
          </label>
          <div className="seg">
            <button className={side === "buy" ? "on buy" : ""} onClick={() => setSide("buy")}>BUY</button>
            <button className={side === "sell" ? "on sell" : ""} onClick={() => setSide("sell")}>SELL</button>
          </div>
          <label>
            Type <Tip text="Limit = fill at your price or better (BUY below / SELL above market). Stop = fill on breakout (BUY above / SELL below)." />
            <select value={otype} onChange={(e) => setOtype(e.target.value as "limit" | "stop")}>
              <option value="limit">Limit</option>
              <option value="stop">Stop</option>
            </select>
          </label>
          {/* Lots and pip DISTANCES are floored at zero — cTrader takes SL/TP as
              a positive distance and derives the side ("BUY: entry - SL,
              SELL: entry + SL"), so a sell's stop sitting above entry is still
              a positive number. The trigger PRICE is deliberately unfloored:
              prices can go negative on commodities (WTI settled at -$37.63 on
              2020-04-20) and XTIUSD / XBRUSD / NAT.GAS are watchlisted. */}
          <label>Lots<input type="number" min="0.01" step="0.01" value={lots} onChange={(e) => setLots(Math.max(0, Number(e.target.value)))} style={{ width: 80 }} /></label>
          <label>
            Trigger price <Tip text="The price at which the order activates. This is your 'when the criteria are met' level." />
            <input type="number" step="0.00001" value={trigger} placeholder={spot ? String(spot.midPrice) : "price"} onChange={(e) => setTrigger(e.target.value === "" ? "" : Number(e.target.value))} style={{ width: 110 }} />
          </label>
          <label>SL pips<input type="number" min="0" value={sl} onChange={(e) => setSl(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} style={{ width: 80 }} /></label>
          <label>TP pips<input type="number" min="0" value={tp} onChange={(e) => setTp(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} style={{ width: 80 }} /></label>
          <label>
            Expiry <Tip text="Optional. Order auto-cancels at this time (Good-Till-Date). Leave blank to rest until filled or cancelled." />
            <input type="datetime-local" value={expiry} onChange={(e) => setExpiry(e.target.value)} />
          </label>
          <button className="primary" onClick={submit} disabled={busy}>{busy ? "…" : `Place ${side.toUpperCase()} ${otype}`}</button>
        </div>
        <div className="muted small" style={{ marginTop: 8 }}>
          {spot ? <>Current {symbol.toUpperCase()}: <b>{spot.midPrice}</b> (bid {spot.bid} / ask {spot.ask}). </> : <>Live price loading… </>}
          {dirHint && <span style={{ color: dirHint.warn ? "var(--warn, #d08700)" : "var(--pos, #16a34a)" }}>{dirHint.text}</span>}
        </div>
      </div>

      {/* ── Resting broker orders ── */}
      <h2>Resting orders ({orders.length})</h2>
      {pErr && <div className="banner warn">{String(pErr).slice(0, 180)}</div>}
      {orders.length === 0 ? (
        <p className="muted">No resting orders. Any limit/stop order you place will wait here until it fills or you cancel it.</p>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>Order</th>
              <th>Symbol</th>
              <th>Side</th>
              <th>Type</th>
              <th>Lots</th>
              <th>Trigger</th>
              <th>SL</th>
              <th>TP</th>
              <th>Placed</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {orders.map((o) => {
              const buy = String(o.side ?? "").toUpperCase().includes("BUY");
              return (
                <tr key={o.orderId}>
                  <td className="muted">#{o.orderId}</td>
                  <td><b>{o.symbol}</b></td>
                  <td className={buy ? "buy" : "sell"}>{o.side}</td>
                  <td>{o.orderType}</td>
                  <td>{o.volumeLots != null ? o.volumeLots.toFixed(2) : o.volume}</td>
                  <td><b>{price(o.triggerPrice)}</b></td>
                  <td className="muted">{price(o.stopLoss)}</td>
                  <td className="muted">{price(o.takeProfit)}</td>
                  <td className="muted">{fmtTime(o.openTimestampMs)}</td>
                  <td><button className="danger" disabled={busy} onClick={() => cancel(o.orderId)}>Cancel</button></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      {/* ── AI-proposed actions ── */}
      <h2>AI approvals ({liveActions.length})</h2>
      {aErr && <div className="banner warn">{String(aErr).slice(0, 160)}</div>}
      {liveActions.length === 0 ? (
        <p className="muted">Nothing awaiting approval. When the assistant proposes a trade-management action, it appears here for your one-click confirm/reject.</p>
      ) : (
        <div className="news-list">
          {liveActions.map((a, i) => {
            const id = String(a.id ?? a.actionId ?? i);
            return (
              <div className="news-item" key={id}>
                <div className="news-title">{a.kind ?? a.type ?? a.action ?? "Action"} — {a.symbol ?? ""}</div>
                <div className="muted small" style={{ whiteSpace: "pre-wrap" }}>{a.reason ?? a.summary ?? a.description ?? JSON.stringify(a)}</div>
                <div className="btn-row">
                  <button className="primary" disabled={busy} onClick={() => decide(id, true)}>Confirm</button>
                  <button className="danger" disabled={busy} onClick={() => decide(id, false)}>Reject</button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
