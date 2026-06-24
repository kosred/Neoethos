import { useState } from "react";
import { brokerProfile, brokerVersion, ordersHistory, cashFlow, expectedMargin } from "../api";
import { usePoll } from "../hooks";

const fmt = (v: unknown) =>
  typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(5)) : v == null ? "—" : String(v);

export default function Account() {
  const { data: profile } = usePoll(brokerProfile, 0);
  const { data: version } = usePoll(brokerVersion, 0);
  const { data: hist, error: he } = usePoll(ordersHistory, 0);
  const { data: cash } = usePoll(cashFlow, 0);

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

  return (
    <div className="screen">
      <h1>Account</h1>
      <p className="sub">Broker identity · order history · cash flow · margin</p>

      <div className="settings-grid">
        <div className="kv"><span>cTID user</span><b>{profile?.userId ?? "—"}</b></div>
        <div className="kv"><span>Broker API</span><b>v{version?.version ?? "—"}</b></div>
        <div className="kv"><span>Account</span><b>{hist?.accountId ?? "—"}</b></div>
      </div>

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
    </div>
  );
}
