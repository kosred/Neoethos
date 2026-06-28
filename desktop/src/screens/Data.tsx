import { useState } from "react";
import { dataBootstrap, dataFetch, refreshBrokerCosts } from "../api";
import { usePoll } from "../hooks";
import { SymbolSelect, TimeframeSelect, invalidateSymbolCache } from "../components/Select";
import { HelpPanel, HelpStep } from "../components/Help";

export default function Data() {
  const { data, error, reload } = usePoll(dataBootstrap, 0);
  const [symbol, setSymbol] = useState("EURUSD");
  const [tf, setTf] = useState("H1");
  const [from, setFrom] = useState("2020-01-01");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [costBusy, setCostBusy] = useState(false);
  const [costMsg, setCostMsg] = useState("");

  const refreshCosts = async () => {
    setCostBusy(true);
    setCostMsg("Fetching real per-symbol costs from the broker… (can take a minute)");
    try {
      setCostMsg(`✓ ${await refreshBrokerCosts()}`);
    } catch (e) {
      setCostMsg(`Failed: ${e}`);
    } finally {
      setCostBusy(false);
    }
  };

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
      setMsg(`✓ Fetched ${r?.barCount ?? "?"} bars → ${r?.writtenPath ?? ""}${r?.hasMore ? " (more available)" : ""}`);
      invalidateSymbolCache();
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

      <HelpPanel id="data">
        <p>This screen manages the <b>price history</b> the engine trains and backtests on. Everything is stored locally under your data folder (see <b>Files &amp; Storage</b>).</p>
        <HelpStep n={1}><b>Download bars:</b> pick a <b>Symbol</b> and <b>Timeframe</b> from the lists, choose a <b>From</b> date, and press <b>Fetch from broker</b>. The deeper the date, the longer it takes (a 10-year M1 pull is millions of bars). It replaces that symbol+timeframe file with the fetched range.</HelpStep>
        <HelpStep n={2}><b>Broker costs:</b> press <b>Refresh broker costs</b> once so backtests use your account's real commission/swap/spread instead of a generic table — this keeps results honest vs live.</HelpStep>
        <HelpStep n={3}><b>Local symbols:</b> the chips at the bottom show what data you already have. These are the symbols available in every dropdown across the app.</HelpStep>
        <p className="muted small">Tip: discovery uses a base timeframe plus higher ones, so download the same date range for each timeframe you plan to search.</p>
      </HelpPanel>

      {error && <div className="banner warn">{error}</div>}

      {data && (
        <div className="cards">
          <div className="card"><div className="card-label">SYMBOLS</div><div className="card-value">{data.symbols.length}</div></div>
          <div className="card"><div className="card-label">FILES</div><div className="card-value">{data.fileCount}</div></div>
          <div className="card" style={{ gridColumn: "span 2" }}>
            <div className="card-label">DATA DIR</div>
            <div className="card-value" style={{ fontSize: 12 }}>{data.dataDir} {data.dataDirExists ? "" : "(missing)"}</div>
          </div>
        </div>
      )}

      <h2>Download bars</h2>
      <div className="ticket">
        <div className="ticket-row">
          <label>Symbol<SymbolSelect value={symbol} onChange={setSymbol} style={{ width: 120 }} /></label>
          <label>Timeframe<TimeframeSelect value={tf} onChange={setTf} style={{ width: 90 }} /></label>
          <label>From<input type="date" value={from} onChange={(e) => setFrom(e.target.value)} style={{ width: 150 }} /></label>
        </div>
        <div className="btn-row">
          <button className="primary" disabled={busy} onClick={fetchData}>Fetch from broker</button>
        </div>
        {msg && <div className="banner info">{msg}</div>}
      </div>

      <h2>Broker costs (for accurate backtests)</h2>
      <div className="ticket">
        <p className="muted small">
          Pull this account's real per-lot commission, swap and spread from cTrader and rebuild the
          cost model. Without it, discovery uses a static table that may not match your broker — making
          backtests over-optimistic vs live.
        </p>
        <div className="btn-row">
          <button className="primary" disabled={costBusy} onClick={refreshCosts}>
            {costBusy ? "Refreshing…" : "Refresh broker costs"}
          </button>
        </div>
        {costMsg && <div className="banner info">{costMsg}</div>}
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
