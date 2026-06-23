import { journalStats, journalTrades } from "../api";
import { usePoll } from "../hooks";

const fmt = (v: unknown) =>
  typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v.toFixed(2)) : String(v);

export default function Journal() {
  const { data: stats, error: e1 } = usePoll(journalStats, 0);
  const { data: trades, error: e2 } = usePoll(journalTrades, 0);

  const statEntries =
    stats && typeof stats === "object"
      ? Object.entries(stats).filter(([, v]) => typeof v !== "object" || v === null)
      : [];
  const tradeRows: any[] = Array.isArray(trades) ? trades : (trades?.trades ?? []);
  const cols = tradeRows.length > 0 ? Object.keys(tradeRows[0]).slice(0, 8) : [];

  return (
    <div className="screen">
      <h1>Trade Journal</h1>
      <p className="sub">Closed-trade log &amp; computed stats (MyFxbook-style)</p>
      {(e1 || e2) && <div className="banner warn">{e1 || e2}</div>}

      {statEntries.length > 0 && (
        <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
          {statEntries.slice(0, 8).map(([k, v]) => (
            <div className="card" key={k}>
              <div className="card-label">{k.replace(/_/g, " ").toUpperCase()}</div>
              <div className="card-value" style={{ fontSize: 18 }}>{fmt(v)}</div>
            </div>
          ))}
        </div>
      )}

      <h2>Trades ({tradeRows.length})</h2>
      {tradeRows.length === 0 ? (
        <p className="muted">No closed trades recorded yet.</p>
      ) : (
        <table className="tbl">
          <thead><tr>{cols.map((c) => <th key={c}>{c.replace(/_/g, " ")}</th>)}</tr></thead>
          <tbody>
            {tradeRows.slice(0, 200).map((r, i) => (
              <tr key={i}>{cols.map((c) => <td key={c}>{fmt(r[c])}</td>)}</tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
