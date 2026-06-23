import { useEffect, useState } from "react";
import { getWatchlist, setWatchlist, liveSpots } from "../api";
import { usePoll } from "../hooks";

export default function MarketWatch() {
  const [edit, setEdit] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const { data: spots } = usePoll(liveSpots, 2000);

  useEffect(() => {
    getWatchlist()
      .then((w) => {
        const syms: string[] = Array.isArray(w) ? w : (w?.symbols ?? []);
        setEdit(syms.join(", "));
      })
      .catch((e) => setMsg(String(e)));
  }, []);

  const save = async () => {
    const symbols = edit.split(",").map((s) => s.trim().toUpperCase()).filter(Boolean);
    setBusy(true);
    setMsg("Saving watchlist…");
    try {
      await setWatchlist(symbols);
      setMsg(`✓ Saved ${symbols.length} symbols — live stream re-subscribes within ~5s.`);
    } catch (e) {
      setMsg(`Save failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const rows: any[] = Array.isArray(spots) ? spots : (spots?.spots ?? []);

  return (
    <div className="screen">
      <h1>Market Watch</h1>
      <p className="sub">Editable watchlist with live bid/ask</p>

      <div className="ticket">
        <label style={{ display: "block", fontSize: 11, color: "#6b7280" }}>
          Symbols (comma-separated)
          <input value={edit} onChange={(e) => setEdit(e.target.value)} style={{ width: "100%", marginTop: 4 }} placeholder="EURUSD, GBPUSD, XAUUSD" />
        </label>
        <div className="btn-row">
          <button className="primary" disabled={busy} onClick={save}>Save watchlist</button>
        </div>
        {msg && <div className="banner info">{msg}</div>}
      </div>

      <h2>Live prices</h2>
      {rows.length === 0 ? (
        <p className="muted">No live ticks yet (needs broker connection + spot streamer).</p>
      ) : (
        <table className="tbl">
          <thead><tr><th>Symbol</th><th>Bid</th><th>Ask</th><th>Mid</th></tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>{r.symbolName ?? r.symbolId}</td>
                <td>{r.bid ?? "—"}</td>
                <td>{r.ask ?? "—"}</td>
                <td>{r.midPrice ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
