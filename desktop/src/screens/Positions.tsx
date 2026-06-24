import { useEffect, useState } from "react";
import {
  accountSnapshot,
  closePosition,
  placeOrder,
  refreshAccount,
  type AccountSnapshot,
  type ExecResult,
} from "../api";
import { useAccountStream } from "../hooks";
import PositionsTable from "../components/PositionsTable";

export default function Positions() {
  const [acct, setAcct] = useState<AccountSnapshot | null>(null);
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");
  const { snap } = useAccountStream(); // live positions + P/L (push)

  // order ticket
  const [symbol, setSymbol] = useState("EURUSD");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [lots, setLots] = useState(0.01);
  const [sl, setSl] = useState<number | "">(20);
  const [tp, setTp] = useState<number | "">(40);
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      setAcct(await accountSnapshot());
      setErr("");
    } catch (e) {
      setErr(String(e));
    }
    // also nudge the backend's account bridge so the SSE stream (Dashboard)
    // reflects the new state within ~1s instead of waiting for the safety poll.
    refreshAccount().catch(() => {});
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  const show = (r: ExecResult) =>
    setMsg(`${r.status}${r.positionId ? ` · pos #${r.positionId}` : ""}${r.fillPrice ? ` @ ${r.fillPrice}` : ""}${r.message ? ` · ${r.message}` : ""}`);

  const submit = async () => {
    setBusy(true);
    setMsg("");
    try {
      const r = await placeOrder(
        symbol.toUpperCase(),
        side,
        lots,
        sl === "" ? undefined : Number(sl),
        tp === "" ? undefined : Number(tp),
      );
      show(r);
      await refresh();
    } catch (e) {
      setMsg(`Error: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const onClose = async (positionId: number, volume: number) => {
    setBusy(true);
    setMsg("");
    try {
      // reconcile volume is the broker wire volume; pass it through.
      const r = await closePosition(positionId, Math.round(volume));
      show(r);
      await refresh();
    } catch (e) {
      setMsg(`Error: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Positions</h1>

      <div className="ticket">
        <h2>New market order</h2>
        <div className="ticket-row">
          <label>
            Symbol
            <input value={symbol} onChange={(e) => setSymbol(e.target.value)} />
          </label>
          <div className="seg">
            <button className={side === "buy" ? "on buy" : ""} onClick={() => setSide("buy")}>
              BUY
            </button>
            <button className={side === "sell" ? "on sell" : ""} onClick={() => setSide("sell")}>
              SELL
            </button>
          </div>
          <label>
            Lots
            <input type="number" step="0.01" value={lots} onChange={(e) => setLots(Number(e.target.value))} />
          </label>
          <label>
            SL pips
            <input type="number" value={sl} onChange={(e) => setSl(e.target.value === "" ? "" : Number(e.target.value))} />
          </label>
          <label>
            TP pips
            <input type="number" value={tp} onChange={(e) => setTp(e.target.value === "" ? "" : Number(e.target.value))} />
          </label>
          <button className="primary" onClick={submit} disabled={busy}>
            {busy ? "…" : `${side.toUpperCase()} ${lots}`}
          </button>
        </div>
        {msg && <div className="banner info">{msg}</div>}
      </div>

      <h2>Open positions</h2>
      {err && <div className="banner warn">{err.slice(0, 160)}</div>}
      <PositionsTable
        live={snap?.positions ?? []}
        detail={acct?.positions}
        currency={snap?.currency ?? acct?.currency ?? ""}
        onClose={onClose}
        busy={busy}
      />
    </div>
  );
}
