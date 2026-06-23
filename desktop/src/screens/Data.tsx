import { useState } from "react";
import { dataBootstrap, dataFetch } from "../api";
import { usePoll } from "../hooks";

export default function Data() {
  const { data, error, reload } = usePoll(dataBootstrap, 0);
  const [symbol, setSymbol] = useState("EURUSD");
  const [tf, setTf] = useState("H1");
  const [from, setFrom] = useState("2020-01-01");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const fetchData = async () => {
    const fromMs = Date.parse(from);
    if (Number.isNaN(fromMs)) {
      setMsg("Invalid 'from' date (use YYYY-MM-DD).");
      return;
    }
    setBusy(true);
    setMsg(`Fetching ${symbol} ${tf} from broker…`);
    try {
      const r: any = await dataFetch({ symbol: symbol.trim().toUpperCase(), timeframe: tf.trim().toUpperCase(), from_ms: fromMs });
      setMsg(`✓ Fetched ${r?.bar_count ?? "?"} bars → ${r?.written_path ?? ""}${r?.has_more ? " (more available)" : ""}`);
      await reload();
    } catch (e) {
      setMsg(`Fetch failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Data</h1>
      <p className="sub">Local dataset status &amp; download historical bars from the broker</p>
      {error && <div className="banner warn">{error}</div>}

      {data && (
        <div className="cards">
          <div className="card"><div className="card-label">SYMBOLS</div><div className="card-value">{data.symbols.length}</div></div>
          <div className="card"><div className="card-label">FILES</div><div className="card-value">{data.file_count}</div></div>
          <div className="card" style={{ gridColumn: "span 2" }}>
            <div className="card-label">DATA DIR</div>
            <div className="card-value" style={{ fontSize: 12 }}>{data.data_dir} {data.data_dir_exists ? "" : "(missing)"}</div>
          </div>
        </div>
      )}

      <h2>Download bars</h2>
      <div className="ticket">
        <div className="ticket-row">
          <label>Symbol<input value={symbol} onChange={(e) => setSymbol(e.target.value)} style={{ width: 110 }} /></label>
          <label>Timeframe<input value={tf} onChange={(e) => setTf(e.target.value)} style={{ width: 80 }} /></label>
          <label>From<input value={from} onChange={(e) => setFrom(e.target.value)} style={{ width: 120 }} placeholder="YYYY-MM-DD" /></label>
        </div>
        <div className="btn-row">
          <button className="primary" disabled={busy} onClick={fetchData}>Fetch from broker</button>
        </div>
        {msg && <div className="banner info">{msg}</div>}
      </div>

      {data && data.symbols.length > 0 && (
        <>
          <h2>Local symbols</h2>
          <div className="ticker" style={{ flexWrap: "wrap" }}>
            {data.symbols.map((s) => <span className="tick" key={s}><b>{s}</b></span>)}
          </div>
        </>
      )}
    </div>
  );
}
