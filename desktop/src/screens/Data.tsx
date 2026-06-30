import { useState } from "react";
import { dataBootstrap, dataFetch, refreshBrokerCosts } from "../api";
import { usePoll } from "../hooks";
import { useSymbolOptions, useTimeframeOptions, invalidateSymbolCache } from "../components/Select";
import { HelpPanel, HelpStep } from "../components/Help";

const TF_SPEED = ["MN1", "W1", "D1", "H12", "H4", "H1", "M30", "M15", "M5", "M3", "M1"];
const tfRank = (t: string) => {
  const i = TF_SPEED.indexOf(t);
  return i < 0 ? 99 : i;
};

function Chips({ opts, sel, onToggle }: { opts: string[]; sel: string[]; onToggle: (v: string) => void }) {
  return (
    <div className="chip-row">
      {opts.map((o) => (
        <button key={o} type="button" className={`chip ${sel.includes(o) ? "on" : ""}`} onClick={() => onToggle(o)}>{o}</button>
      ))}
    </div>
  );
}

export default function Data() {
  const { data, error, reload } = usePoll(dataBootstrap, 0);
  const symOpts = useSymbolOptions();
  const tfOpts = useTimeframeOptions();
  const [selSyms, setSelSyms] = useState<string[]>([]);
  const [selTfs, setSelTfs] = useState<string[]>([]);
  const [from, setFrom] = useState("2015-01-01");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [costBusy, setCostBusy] = useState(false);
  const [costMsg, setCostMsg] = useState("");

  const toggle = (set: React.Dispatch<React.SetStateAction<string[]>>) => (v: string) =>
    set((c) => (c.includes(v) ? c.filter((x) => x !== v) : [...c, v]));

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

  const fetchAll = async () => {
    const fromMs = Date.parse(from);
    if (Number.isNaN(fromMs)) {
      setMsg("Invalid 'from' date.");
      return;
    }
    if (selSyms.length === 0 || selTfs.length === 0) {
      setMsg("Pick at least one symbol and one timeframe.");
      return;
    }
    const tfs = [...selTfs].sort((a, b) => tfRank(a) - tfRank(b));
    const combos = selSyms.flatMap((s) => tfs.map((t) => ({ s, t })));
    setBusy(true);
    let done = 0;
    let failed = 0;
    const fails: string[] = [];
    for (const { s, t } of combos) {
      setMsg(`Downloading ${done + failed + 1}/${combos.length}: ${s} ${t}…`);
      try {
        await dataFetch({ symbol: s.toUpperCase(), timeframe: t.toUpperCase(), from_ms: fromMs });
        done++;
      } catch (e) {
        failed++;
        fails.push(`${s} ${t}`);
      }
    }
    invalidateSymbolCache();
    await reload();
    setMsg(`✓ Downloaded ${done}/${combos.length}${failed ? ` · ${failed} failed (${fails.slice(0, 4).join(", ")}${fails.length > 4 ? "…" : ""})` : ""}.`);
    setBusy(false);
  };

  const nCombos = selSyms.length * selTfs.length;

  return (
    <div className="screen">
      <h1>Data</h1>
      <p className="sub">Local dataset status &amp; download historical bars from the broker (for Discovery + Training)</p>

      <HelpPanel id="data">
        <p>This screen manages the <b>price history</b> the engine searches and trains on. Everything is stored locally under your data folder (see <b>Files &amp; Storage</b>).</p>
        <HelpStep n={1}><b>Download bars:</b> tick one or more <b>Symbols</b> and <b>Timeframes</b>, pick a <b>From</b> date, and press <b>Fetch</b>. It downloads every symbol × timeframe combination in turn (deeper dates = millions of bars, slower). Each replaces that symbol+timeframe file with the fetched range.</HelpStep>
        <HelpStep n={2}><b>Broker costs:</b> press <b>Refresh broker costs</b> once so backtests use your account's real commission/swap/spread instead of a generic table.</HelpStep>
        <HelpStep n={3}><b>Local symbols:</b> the chips at the bottom show what data you already have — available in every dropdown across the app.</HelpStep>
        <p className="muted small">Tip: Discovery searches a base timeframe plus higher ones, so download the same From date across the timeframes you plan to use.</p>
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
        <label className="picker-label">
          Symbols <span className="muted">({selSyms.length || "none"})</span>
          <div className="picker-actions">
            <button type="button" className="link" onClick={() => setSelSyms(symOpts)}>all</button>
            <button type="button" className="link" onClick={() => setSelSyms([])}>none</button>
          </div>
        </label>
        <Chips opts={symOpts} sel={selSyms} onToggle={toggle(setSelSyms)} />

        <label className="picker-label" style={{ marginTop: 12 }}>
          Timeframes <span className="muted">({selTfs.length || "none"})</span>
          <div className="picker-actions">
            <button type="button" className="link" onClick={() => setSelTfs(["H1", "M30", "M15", "M5", "M3", "M1"])}>discovery set</button>
            <button type="button" className="link" onClick={() => setSelTfs(tfOpts)}>all</button>
            <button type="button" className="link" onClick={() => setSelTfs([])}>none</button>
          </div>
        </label>
        <Chips opts={tfOpts} sel={selTfs} onToggle={toggle(setSelTfs)} />

        <div className="ticket-row" style={{ marginTop: 12, alignItems: "flex-end" }}>
          <label>From<input type="date" value={from} onChange={(e) => setFrom(e.target.value)} style={{ width: 150 }} /></label>
          <button className="primary" disabled={busy || nCombos === 0} onClick={fetchAll}>
            {busy ? "Downloading…" : `Fetch ${nCombos || ""} from broker`}
          </button>
          <span className="muted small">{selSyms.length} × {selTfs.length} = {nCombos} download{nCombos === 1 ? "" : "s"}</span>
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
