import { useState } from "react";
import {
  pendingActions,
  confirmAction,
  rejectAction,
  brokerPendingOrders,
  placePendingOrder,
  amendOrder,
  cancelOrder,
  type PendingAction,
  type PendingOrder,
} from "../api";
import { usePoll, useSpotStream } from "../hooks";
import { SymbolSelect } from "../components/Select";
import { HelpPanel, HelpStep, Tip } from "../components/Help";
import { useBrokerUi } from "../brokerUiContext";
import BrokerUnavailable from "../components/BrokerUnavailable";

const price = (value: unknown) =>
  typeof value === "number" && Number.isFinite(value) ? String(value) : "—";
const fmtTime = (ms: unknown) =>
  typeof ms === "number" && ms > 0 ? new Date(ms).toLocaleString() : "—";

function actionTitle(action: PendingAction): string {
  if (action.kind.kind === "close_position") {
    const volume = action.kind.volume_units > 0
      ? `${action.kind.volume_units.toLocaleString()} units of `
      : "all of ";
    const symbol = action.kind.symbol_hint ? ` (${action.kind.symbol_hint})` : "";
    return `Close ${volume}position #${action.kind.position_id}${symbol}`;
  }
  return `Run MCP tool ${action.kind.server}/${action.kind.tool}`;
}

export default function Actions() {
  const { access } = useBrokerUi();
  return <ActionsContent key={access.key} brokerEnabled={access.requestsEnabled} brokerKey={access.key} />;
}

function ActionsContent({ brokerEnabled, brokerKey }: { brokerEnabled: boolean; brokerKey: string }) {
  // Broker-side resting (limit/stop) orders — the "trade when price hits X" list.
  const { data: pendingData, error: pErr, reload: reloadPending } = usePoll(brokerPendingOrders, 5000, brokerKey, brokerEnabled);
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
  const validOrder = symbol.trim() !== "" && Number.isFinite(lots) && lots > 0
    && trigger !== "" && Number.isFinite(trigger) && trigger > 0
    && (sl === "" || (Number.isFinite(sl) && sl >= 0))
    && (tp === "" || (Number.isFinite(tp) && tp >= 0))
    && (!expiry || Number.isFinite(new Date(expiry).getTime()));

  // Live prices to anchor the trigger against the current market.
  const { ticks, error: quoteError } = useSpotStream(brokerEnabled);
  const spot = ticks[symbol.toUpperCase()];

  const orders: PendingOrder[] = Array.isArray(pendingData) ? pendingData : [];
  const actions = actionsData?.actions ?? [];
  const liveActions = actions.filter((action) => action.status === "pending");

  // Non-blocking sanity hint: which side of the market this order type usually rests.
  const dirHint = (() => {
    if (!spot || trigger === "" || !(Number(trigger) > 0)) return null;
    const px = spot.midPrice;
    if (px == null || !Number.isFinite(px)) return null;
    const t = Number(trigger);
    const wantAbove = (side === "buy" && otype === "stop") || (side === "sell" && otype === "limit");
    const wantBelow = (side === "buy" && otype === "limit") || (side === "sell" && otype === "stop");
    if (wantAbove && t <= px) return { warn: true, text: `⚠ A ${side} ${otype} normally triggers ABOVE the current price (now ${px}).` };
    if (wantBelow && t >= px) return { warn: true, text: `⚠ A ${side} ${otype} normally triggers BELOW the current price (now ${px}).` };
    return { warn: false, text: `✓ Trigger is ${t > px ? "above" : "below"} the market (now ${px}).` };
  })();

  const submit = async () => {
    if (!brokerEnabled) return;
    if (busy || !validOrder || (expiry && new Date(expiry).getTime() <= Date.now())) {
      setMsg("Set positive lots and a trigger price, non-negative stop distances and a future expiry (or leave expiry empty).");
      return;
    }
    setBusy(true);
    setMsg("Placing conditional order…");
    try {
      const r = await placePendingOrder({
        symbol: symbol.toUpperCase(),
        side,
        orderType: otype,
        volumeLots: lots,
        triggerPrice: Number(trigger),
        stopLossPips: sl === "" ? null : Number(sl),
        takeProfitPips: tp === "" ? null : Number(tp),
        expiryUnixMs: expiry ? new Date(expiry).getTime() : null,
      });
      setMsg(`Broker response: ${r.status}${r.orderId ? ` · order #${r.orderId}` : ""}${r.message ? ` · ${r.message}` : ""}`);
      await reloadPending();
    } catch (e) {
      setMsg(`Failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  // ── Modify a resting order in place (audit #236) ──────────────────────────
  // Until 2026-08-10 the only way to change a trigger price was Cancel then
  // re-place: two broker round trips, a new order id, and a stretch in which
  // the level the operator is waiting for has no order behind it at all.
  // `editId` is the row being edited; blank fields mean LEAVE UNCHANGED, which
  // is what the endpoint does with an omitted field.
  const [editId, setEditId] = useState<number | null>(null);
  const [eLots, setELots] = useState<number | "">("");
  const [eTrigger, setETrigger] = useState<number | "">("");
  const [eSl, setESl] = useState<number | "">("");
  const [eTp, setETp] = useState<number | "">("");

  const beginEdit = (o: PendingOrder) => {
    setEditId(o.orderId);
    // Seed with what the order carries today so the operator edits a value
    // rather than retyping one, and an untouched field round-trips unchanged.
    setELots(o.volumeLots ?? "");
    setETrigger(o.triggerPrice ?? "");
    // SL/TP come back from the broker as ABSOLUTE prices while the amend takes
    // pip DISTANCES, so they are deliberately left blank: pre-filling a price
    // into a pips box would send a stop thousands of pips away.
    setESl("");
    setETp("");
  };

  const cancelEdit = () => {
    setEditId(null);
    setELots("");
    setETrigger("");
    setESl("");
    setETp("");
  };

  const saveEdit = async (o: PendingOrder) => {
    if (!brokerEnabled || busy) return;
    const otypeOfRow = String(o.orderType ?? "").toUpperCase().includes("STOP") ? "stop" : "limit";
    if (eLots === "" && eTrigger === "" && eSl === "" && eTp === "") {
      setMsg("Nothing to change — set at least one of lots, trigger, SL or TP.");
      return;
    }
    setBusy(true);
    setMsg(`Modifying order #${o.orderId}…`);
    try {
      const response = await amendOrder({
        orderId: o.orderId,
        symbol: o.symbol,
        orderType: otypeOfRow,
        volumeLots: eLots === "" ? null : Number(eLots),
        triggerPrice: eTrigger === "" ? null : Number(eTrigger),
        stopLossPips: eSl === "" ? null : Number(eSl),
        takeProfitPips: eTp === "" ? null : Number(eTp),
      });
      setMsg(`Modify #${o.orderId}: ${response.status}${response.message ? ` · ${response.message}` : ""}`);
      await reloadPending();
    } catch (e) {
      setMsg(`Modify failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (orderId: number) => {
    if (!brokerEnabled || busy) return;
    setBusy(true);
    setMsg(`Cancelling order #${orderId}…`);
    try {
      const response = await cancelOrder(orderId);
      setMsg(`Cancel #${orderId}: ${response.status}${response.message ? ` · ${response.message}` : ""}`);
      await reloadPending();
    } catch (e) {
      setMsg(`Cancel failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const decide = async (id: string, ok: boolean) => {
    if (busy || aErr) return;
    const action = actions.find((entry) => entry.id === id);
    if (ok && action?.kind.kind === "close_position" && !brokerEnabled) return;
    setBusy(true);
    setMsg(ok ? `Confirming ${id}…` : `Rejecting ${id}…`);
    try {
      if (ok) {
        const response = await confirmAction(id);
        setMsg(`Confirmation response for ${id}: ${response.status} · accepted: ${response.ok}`);
      } else {
        const response = await rejectAction(id);
        setMsg(`Rejection response for ${id}: ${response.action.status} · ${response.action.result_note || "no further detail"}`);
      }
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
        <HelpStep n={3}>Resting orders appear below — <b>Modify</b> one to change its lots, trigger price or bracket in place (leave a box blank to keep what it has), or <b>Cancel</b> any that haven't filled. The broker validates the price/side combination and rejects invalid ones with a reason.</HelpStep>
        <p className="muted small">The <b>AI approvals</b> section stays empty unless the assistant proposes a trade-management action for your one-click confirmation. Automated entries live in <b>Autopilot</b>.</p>
      </HelpPanel>

      <BrokerUnavailable />
      {msg && <div className="banner info">{msg}</div>}

      <fieldset disabled={!brokerEnabled} style={{ border: 0, padding: 0, margin: 0, minWidth: 0 }}>
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
          {/* This endpoint currently requires positive trigger prices and lot
              sizes; SL/TP are non-negative pip distances, not absolute prices. */}
          <label>Lots<input type="number" min="0.01" step="0.01" value={lots} onChange={(e) => setLots(Math.max(0, Number(e.target.value)))} style={{ width: 80 }} /></label>
          <label>
            Trigger price <Tip text="The price at which the order activates. This is your 'when the criteria are met' level." />
            <input type="number" step="0.00001" value={trigger} placeholder={spot?.midPrice != null ? String(spot.midPrice) : "price"} onChange={(e) => setTrigger(e.target.value === "" ? "" : Number(e.target.value))} style={{ width: 110 }} />
          </label>
          <label>SL pips<input type="number" min="0" value={sl} onChange={(e) => setSl(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} style={{ width: 80 }} /></label>
          <label>TP pips<input type="number" min="0" value={tp} onChange={(e) => setTp(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} style={{ width: 80 }} /></label>
          <label>
            Expiry <Tip text="Optional. Order auto-cancels at this time (Good-Till-Date). Leave blank to rest until filled or cancelled." />
            <input type="datetime-local" value={expiry} onChange={(e) => setExpiry(e.target.value)} />
          </label>
          <button className="primary" onClick={submit} disabled={busy || !validOrder}>{busy ? "…" : `Place ${side.toUpperCase()} ${otype}`}</button>
        </div>
        <div className="muted small" style={{ marginTop: 8 }}>
          {spot ? <>Last observed {symbol.toUpperCase()}: <b>{price(spot.midPrice)}</b> (bid {price(spot.bid)} / ask {price(spot.ask)}). </> : <>No live price observed. </>}
          {dirHint && <span style={{ color: dirHint.warn ? "var(--warn, #d08700)" : "var(--pos, #16a34a)" }}>{dirHint.text}</span>}
        </div>
        {quoteError && <div className="banner warn" role="alert">{quoteError}</div>}
      </div>

      {/* ── Resting broker orders ── */}
      <h2>Resting orders ({pendingData ? orders.length : "unknown"})</h2>
      {pErr && <div className="banner warn">{String(pErr)}</div>}
      {orders.length === 0 ? (
        <p className="muted">{pendingData && !pErr ? "No resting orders in the latest broker snapshot." : "Resting orders are not confirmed. Wait for a successful broker response."}</p>
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
                  <td style={{ whiteSpace: "nowrap" }}>
                    <button disabled={busy} onClick={() => (editId === o.orderId ? cancelEdit() : beginEdit(o))}>
                      {editId === o.orderId ? "Close" : "Modify"}
                    </button>{" "}
                    <button className="danger" disabled={busy} onClick={() => cancel(o.orderId)}>Cancel</button>
                  </td>
                </tr>
              );
            })}
            {orders
              .filter((o) => o.orderId === editId)
              .map((o) => (
                <tr key={`edit-${o.orderId}`}>
                  <td colSpan={10}>
                    <div className="ticket-row" style={{ flexWrap: "wrap", gap: 12 }}>
                      <span className="muted">Modify #{o.orderId} — blank = leave unchanged</span>
                      <label>Lots<input type="number" min="0.01" step="0.01" value={eLots} onChange={(e) => setELots(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} style={{ width: 80 }} /></label>
                      {/* Unfloored on purpose: commodity prices can be negative. */}
                      <label>Trigger<input type="number" step="0.00001" value={eTrigger} onChange={(e) => setETrigger(e.target.value === "" ? "" : Number(e.target.value))} style={{ width: 110 }} /></label>
                      <label>SL pips <Tip text="A DISTANCE in pips, not a price. The broker reports the existing stop as a price, so this box starts empty — type a distance to replace it, or leave it blank to keep the stop the order already has." /><input type="number" min="0" value={eSl} onChange={(e) => setESl(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} style={{ width: 80 }} /></label>
                      <label>TP pips<input type="number" min="0" value={eTp} onChange={(e) => setETp(e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))} style={{ width: 80 }} /></label>
                      <button className="primary" disabled={busy} onClick={() => saveEdit(o)}>{busy ? "…" : "Save changes"}</button>
                      <button disabled={busy} onClick={cancelEdit}>Discard</button>
                    </div>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      )}

      </fieldset>

      {/* ── AI-proposed actions ── */}
      <h2>AI approvals ({actionsData ? liveActions.length : "unknown"})</h2>
      {aErr && <div className="banner warn">{String(aErr).slice(0, 160)}</div>}
      {liveActions.length === 0 ? (
        <p className="muted">{actionsData && !aErr ? "Nothing awaiting approval. AI proposals appear here for your confirmation." : "The approval queue is not confirmed. Waiting for a successful response."}</p>
      ) : (
        <div className="news-list">
          {liveActions.map((action) => {
            return (
              <div className="news-item" key={action.id}>
                <div className="news-title">{actionTitle(action)}</div>
                <div className="muted small" style={{ whiteSpace: "pre-wrap" }}>{action.reason}</div>
                <div className="muted small">Expires {fmtTime(action.expires_at_unix_ms)}</div>
                <div className="btn-row">
                  <button className="primary" disabled={busy || (!brokerEnabled && action.kind.kind === "close_position")} onClick={() => decide(action.id, true)}>Confirm</button>
                  <button className="danger" disabled={busy} onClick={() => decide(action.id, false)}>Reject</button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
