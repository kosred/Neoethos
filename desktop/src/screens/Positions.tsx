import { useEffect, useState } from "react";
import {
  accountSnapshot,
  closePosition,
  placeOrder,
  refreshAccount,
  amendProtection,
  type AccountSnapshot,
  type ExecResult,
} from "../api";
import { useAccountStream } from "../hooks";
import PositionsTable from "../components/PositionsTable";
import { SymbolSelect } from "../components/Select";
import { HelpPanel, HelpStep } from "../components/Help";

export default function Positions() {
  const [acct, setAcct] = useState<AccountSnapshot | null>(null);
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");
  const { snap } = useAccountStream(); // live positions + P/L (push)

  // Unified order ticket: one card, two modes.
  const [mode, setMode] = useState<"new" | "modify">("new");

  // New-order fields
  const [symbol, setSymbol] = useState("EURUSD");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [lots, setLots] = useState(0.01);
  const [sl, setSl] = useState<number | "">(20);
  const [tp, setTp] = useState<number | "">(40);
  const [busy, setBusy] = useState(false);

  // Modify-protection fields
  const [protId, setProtId] = useState<number | "">("");
  const [protSl, setProtSl] = useState<number | "">("");
  const [protTp, setProtTp] = useState<number | "">("");
  const [protTrail, setProtTrail] = useState(false);
  const positions = snap?.positions ?? [];

  const refresh = async () => {
    try {
      setAcct(await accountSnapshot());
      setErr("");
    } catch (e) {
      setErr(String(e));
    }
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
      const r = await closePosition(positionId, Math.round(volume));
      show(r);
      await refresh();
    } catch (e) {
      setMsg(`Error: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  // Selecting a position in the table jumps to Modify mode pre-filled.
  const editPosition = (positionId: number) => {
    const p = positions.find((x) => x.positionId === positionId);
    if (!p) return;
    setMode("modify");
    setProtId(positionId);
    setProtSl(p.stopLoss ?? "");
    setProtTp(p.takeProfit ?? "");
    setProtTrail(false);
  };

  const saveProtection = async () => {
    if (protId === "") {
      setMsg("Pick a position first.");
      return;
    }
    setBusy(true);
    setMsg("Updating SL/TP…");
    try {
      const r: any = await amendProtection(
        Number(protId),
        protSl === "" ? null : Number(protSl),
        protTp === "" ? null : Number(protTp),
        protTrail,
      );
      setMsg(`✓ Protection updated${r?.message ? ` · ${r.message}` : ""}`);
      await refresh();
    } catch (e) {
      setMsg(`SL/TP update failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Positions</h1>

      <HelpPanel id="positions">
        <p>Place and manage <b>manual</b> trades on your connected cTrader account, and watch open positions with live P/L.</p>
        <HelpStep n={1}>One <b>order ticket</b> with two modes. <b>New order:</b> Symbol, BUY/SELL, Lots and optional SL/TP <i>in pips</i> → send. These are real orders on the active account (Demo or Live — check the header).</HelpStep>
        <HelpStep n={2}><b>Modify open:</b> pick a position (or click <b>Edit</b> in the table) and set SL/TP <i>as price levels</i> — e.g. move a stop to breakeven, or enable trailing.</HelpStep>
        <p className="muted small">Manual orders here are not risk-gated — you are in control. Automated trading lives in <b>Autopilot</b> and is protected by the demo gate.</p>
      </HelpPanel>

      {/* ── Unified order ticket ── */}
      <div className="ticket">
        <div className="seg" style={{ marginBottom: 12 }}>
          <button className={mode === "new" ? "on" : ""} onClick={() => setMode("new")}>New order</button>
          <button className={mode === "modify" ? "on" : ""} onClick={() => setMode("modify")}>Modify open</button>
        </div>

        {mode === "new" ? (
          <div className="ticket-row">
            <label>
              Symbol
              <SymbolSelect value={symbol} onChange={setSymbol} style={{ width: 120 }} />
            </label>
            <div className="seg">
              <button className={side === "buy" ? "on buy" : ""} onClick={() => setSide("buy")}>BUY</button>
              <button className={side === "sell" ? "on sell" : ""} onClick={() => setSide("sell")}>SELL</button>
            </div>
            <label>Lots<input type="number" step="0.01" value={lots} onChange={(e) => setLots(Number(e.target.value))} /></label>
            <label>SL pips<input type="number" value={sl} onChange={(e) => setSl(e.target.value === "" ? "" : Number(e.target.value))} /></label>
            <label>TP pips<input type="number" value={tp} onChange={(e) => setTp(e.target.value === "" ? "" : Number(e.target.value))} /></label>
            <button className="primary" onClick={submit} disabled={busy}>{busy ? "…" : `${side.toUpperCase()} ${lots}`}</button>
          </div>
        ) : positions.length === 0 ? (
          <p className="muted">No open positions to modify.</p>
        ) : (
          <div className="ticket-row">
            <label>
              Position
              <select
                value={protId}
                onChange={(e) => {
                  const id = e.target.value === "" ? "" : Number(e.target.value);
                  setProtId(id);
                  const p = positions.find((x) => x.positionId === id);
                  setProtSl(p?.stopLoss ?? "");
                  setProtTp(p?.takeProfit ?? "");
                }}
              >
                <option value="">— pick —</option>
                {positions.map((p) => (
                  <option key={p.positionId} value={p.positionId}>{p.symbol} {p.side} #{p.positionId}</option>
                ))}
              </select>
            </label>
            <label>SL price<input type="number" step="0.00001" value={protSl} onChange={(e) => setProtSl(e.target.value === "" ? "" : Number(e.target.value))} /></label>
            <label>TP price<input type="number" step="0.00001" value={protTp} onChange={(e) => setProtTp(e.target.value === "" ? "" : Number(e.target.value))} /></label>
            <label style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
              <input type="checkbox" checked={protTrail} onChange={(e) => setProtTrail(e.target.checked)} /> Trailing
            </label>
            <button className="primary" disabled={busy} onClick={saveProtection}>Update SL/TP</button>
          </div>
        )}
        {msg && <div className="banner info">{msg}</div>}
      </div>

      <h2>Open positions</h2>
      {err && <div className="banner warn">{err.slice(0, 160)}</div>}
      <PositionsTable
        live={snap?.positions ?? []}
        currency={snap?.currency ?? acct?.currency ?? ""}
        onClose={onClose}
        onEdit={editPosition}
        busy={busy}
      />
    </div>
  );
}
