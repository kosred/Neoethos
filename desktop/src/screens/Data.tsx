import { useEffect, useRef, useState } from "react";
import {
  ApiResponseError,
  dataBootstrap,
  dataFetch,
  dataFetchBody,
  dataFetchStatus,
  refreshBrokerCosts,
  serverSymbols,
  spreadStats,
  stopActiveDataFetch,
  type BrokerSymbol,
  type SpreadStats,
} from "../api";
import { usePoll } from "../hooks";
import { useSymbolOptions, useTimeframeOptions, invalidateSymbolCache } from "../components/Select";
import { HelpPanel, HelpStep } from "../components/Help";
import {
  brokerFetchSelectionKey,
  dataBatchResultText,
  brokerBlockedRetryAfterSeconds,
  dataOperationErrorText,
  selectedDatasetGenerationForBrokerFetch,
  type CanonicalDatasetIdentity,
  type DataFetchStopOutcome,
} from "../apiContracts";
import { CANONICAL_BROKER_TIMEFRAMES } from "../timeframes";

const TF_SPEED: string[] = [...CANONICAL_BROKER_TIMEFRAMES].reverse();
const tfRank = (t: string) => {
  const i = TF_SPEED.indexOf(t);
  return i < 0 ? 99 : i;
};

function Chips({
  opts,
  sel,
  onToggle,
  local,
}: {
  opts: string[];
  sel: string[];
  onToggle: (v: string) => void;
  /** Symbols that already have local data — marked with ✓ on the chip. */
  local?: Set<string>;
}) {
  return (
    <div className="chip-row">
      {opts.map((o) => (
        <button
          key={o}
          type="button"
          className={`chip ${sel.includes(o) ? "on" : ""}`}
          title={local?.has(o.toUpperCase()) ? `${o} — local data already downloaded` : o}
          onClick={() => onToggle(o)}
        >
          {o}{local?.has(o.toUpperCase()) ? " ✓" : ""}
        </button>
      ))}
    </div>
  );
}

export default function Data() {
  const { data, error, reload } = usePoll(dataBootstrap, 0);
  const localSyms = useSymbolOptions();
  const tfOpts = useTimeframeOptions();
  // The FULL broker symbol universe (dozens — forex/metals/indices), so NEW
  // pairs can be downloaded, not just the ones that already have local data.
  const [brokerSyms, setBrokerSyms] = useState<BrokerSymbol[]>([]);
  useEffect(() => {
    serverSymbols()
      .then((u) => setBrokerSyms(u.symbols))
      .catch(() => {}); // broker offline → fall back to local list below
  }, []);
  // Grouped by asset class; local-only symbols (imported files etc.) that the
  // broker list doesn't carry get their own group so nothing disappears.
  const localSet = new Set(localSyms.map((s) => s.toUpperCase()));
  const groups: [string, string[]][] = (() => {
    if (brokerSyms.length === 0) return localSyms.length ? [["Local", localSyms]] : [];
    const byClass: Record<string, string[]> = {};
    const brokerNames = new Set<string>();
    for (const s of brokerSyms) {
      (byClass[s.assetClass || "Other"] ??= []).push(s.symbolName);
      brokerNames.add(s.symbolName.toUpperCase());
    }
    const localOnly = localSyms.filter((s) => !brokerNames.has(s.toUpperCase()));
    const out: [string, string[]][] = Object.entries(byClass)
      .sort()
      .map(([c, list]) => [c, list.sort()]);
    if (localOnly.length) out.push(["Local only", localOnly.sort()]);
    return out;
  })();
  const symOpts = groups.flatMap(([, list]) => list);
  const [selSyms, setSelSyms] = useState<string[]>([]);
  const [selTfs, setSelTfs] = useState<string[]>([]);
  const [selectedBrokerDatasetIds, setSelectedBrokerDatasetIds] = useState<
    Record<string, CanonicalDatasetIdentity>
  >({});
  const [from, setFrom] = useState("2015-01-01");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const stopRequested = useRef(false);
  const {
    data: fetchStatus,
    reload: reloadFetchStatus,
  } = usePoll(dataFetchStatus, busy ? 250 : 0, [busy]);
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
    stopRequested.current = false;
    setBusy(true);
    let done = 0;
    let failed = 0;
    const acknowledgements: string[] = [];
    const fails: string[] = [];
    for (const { s, t } of combos) {
      if (stopRequested.current) break;
      setMsg(`Downloading ${done + failed + 1}/${combos.length}: ${s} ${t}…`);
      const symbol = s.toUpperCase();
      const timeframe = t.toUpperCase();
      try {
        const selectedDatasetIdentity =
          selectedBrokerDatasetIds[brokerFetchSelectionKey(symbol, timeframe)] ?? null;
        const datasetSelection = selectedDatasetGenerationForBrokerFetch(
          data?.datasets ?? [],
          symbol,
          timeframe,
          selectedDatasetIdentity,
        );
        const outcome = await dataFetch(
          dataFetchBody(
            symbol,
            timeframe,
            fromMs,
            undefined,
            datasetSelection,
          ),
        );
        acknowledgements.push(`${outcome.datasetIdentity} @ ${outcome.generation}`);
        done++;
      } catch (e) {
        failed++;
        fails.push(`${symbol} ${timeframe}: ${dataOperationErrorText(e)}`);
        if (stopRequested.current) break;
        const retryAfter = brokerBlockedRetryAfterSeconds(
          e instanceof ApiResponseError ? e.payload : null,
        );
        if (retryAfter !== undefined) {
          stopRequested.current = true;
          const wait = retryAfter === null
            ? "the broker supplied no retryAfter"
            : `wait at least ${retryAfter}s`;
          fails.push(`Batch stopped after broker BLOCKED_PAYLOAD_TYPE; ${wait}.`);
          break;
        }
      }
    }
    invalidateSymbolCache();
    await reload();
    setMsg(dataBatchResultText(combos.length, acknowledgements, fails));
    setBusy(false);
  };

  const stopFetch = async () => {
    if (!fetchStatus?.active) return;
    stopRequested.current = true;
    const renderOutcome = (outcome: DataFetchStopOutcome) => {
      switch (outcome.outcome) {
        case "cancelled":
          setMsg(`Cancellation requested for exact download run ${outcome.runId}.`);
          break;
        case "publication_in_progress":
          setMsg(`Run ${outcome.runId} is already committing atomically and cannot be cancelled.`);
          break;
        case "stale_run":
          setMsg(
            `Run ${outcome.requestedRunId} is stale; active run is ${outcome.activeRunId}.`,
          );
          break;
        case "no_active_fetch":
          setMsg("There is no active broker download to stop.");
          break;
      }
    };
    try {
      renderOutcome(await stopActiveDataFetch(fetchStatus.runId));
    } catch (error) {
      setMsg(`Stop failed: ${dataOperationErrorText(error)}`);
    } finally {
      await reloadFetchStatus();
    }
  };

  const nCombos = selSyms.length * selTfs.length;
  const brokerDatasetCounts = new Map<string, number>();
  for (const entry of data?.datasets ?? []) {
    if (entry.sourceKind !== "ctrader" || entry.symbol === null || entry.timeframe === null) {
      continue;
    }
    const key = brokerFetchSelectionKey(entry.symbol, entry.timeframe);
    brokerDatasetCounts.set(key, (brokerDatasetCounts.get(key) ?? 0) + 1);
  }

  return (
    <div className="screen">
      <h1>Data</h1>
      <p className="sub">Local dataset status &amp; download historical bars from the broker (for Discovery + Training)</p>

      <HelpPanel id="data">
        <p>This screen manages the <b>price history</b> the engine searches and trains on. Everything is stored locally under your data folder (see <b>Files &amp; Storage</b>).</p>
        <HelpStep n={1}><b>Download bars:</b> the symbol list shows the broker's <b>full universe</b> (forex, metals, indices — grouped by class), so you can bring in <b>brand-new pairs</b>, not just refresh existing ones (✓ marks pairs with canonical data). Tick Symbols + Timeframes, pick a <b>From</b> date, press <b>Fetch</b>. Every timeframe is downloaded directly and published as its own canonical Vortex generation.</HelpStep>
        <HelpStep n={2}><b>Broker costs:</b> press <b>Refresh broker costs</b> once so backtests use your account's real commission/swap/spread instead of a generic table.</HelpStep>
        <HelpStep n={3}><b>Local symbols:</b> the chips at the bottom show what data you already have — available in every dropdown across the app.</HelpStep>
        <p className="muted small">Discovery requires direct data for its base and every selected higher timeframe. Download or import each one explicitly; missing data fails visibly.</p>
      </HelpPanel>

      {error && <div className="banner warn">{error}</div>}

      {data && (
        <div className="cards">
          <div className="card"><div className="card-label">SYMBOLS</div><div className="card-value">{data.symbols.length}</div></div>
          <div className="card"><div className="card-label">DATASETS</div><div className="card-value">{data.datasetCount}</div></div>
          <div className="card" style={{ gridColumn: "span 2" }}>
            <div className="card-label">DATA DIR</div>
            <div className="card-value" style={{ fontSize: 12 }}>{data.dataDir} {data.dataDirExists ? "" : "(missing)"}</div>
          </div>
        </div>
      )}

      {data && data.datasets.length > 0 && (
        <>
          <h2>Canonical dataset inventory</h2>
          <p className="muted small">
            When several cTrader identities publish the same symbol/timeframe, choose the exact
            identity that a broker refresh must advance. Its current generation and manifest binding
            are read from this inventory at the moment Fetch is pressed.
          </p>
          <table className="tbl">
            <thead>
              <tr><th>Symbol</th><th>TF</th><th>Current generation</th><th>Exact identity</th><th>Verification</th><th>Broker refresh</th></tr>
            </thead>
            <tbody>
              {data.datasets.map((entry) => {
                const selectionKey = entry.symbol !== null && entry.timeframe !== null
                  ? brokerFetchSelectionKey(entry.symbol, entry.timeframe)
                  : null;
                const multipleBrokerIdentities = entry.sourceKind === "ctrader" &&
                  selectionKey !== null &&
                  (brokerDatasetCounts.get(selectionKey) ?? 0) > 1;
                return (
                  <tr key={entry.datasetIdentity}>
                    <td>{entry.symbol ?? "missing"}</td>
                    <td>{entry.timeframe ?? "missing"}</td>
                    <td><code>{entry.generation}</code></td>
                    <td><code style={{ overflowWrap: "anywhere" }}>{entry.datasetIdentity}</code></td>
                    <td>{entry.verification}</td>
                    <td>
                      {multipleBrokerIdentities && selectionKey !== null ? (
                        <label>
                          <input
                            type="radio"
                            name={`broker-refresh-${selectionKey}`}
                            checked={selectedBrokerDatasetIds[selectionKey] === entry.datasetIdentity}
                            onChange={() => setSelectedBrokerDatasetIds((current) => ({
                              ...current,
                              [selectionKey]: entry.datasetIdentity,
                            }))}
                          />
                          use this identity
                        </label>
                      ) : entry.sourceKind === "ctrader" ? "automatic" : "not broker data"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}

      {data && data.skipped.length > 0 && (
        <div className="banner warn">
          <b>Import/download required or rejected data:</b>
          <ul>
            {data.skipped.map((item) => (
              <li key={`${item.path}:${item.category}:${item.detail}`}>
                <code>{item.path}</code> — {item.category}: {item.detail}
              </li>
            ))}
          </ul>
        </div>
      )}

      <h2>Download bars</h2>
      <div className="ticket">
        <label className="picker-label">
          Symbols <span className="muted">({selSyms.length || "none"} selected · {symOpts.length} available{brokerSyms.length ? " from broker" : " — broker offline, local only"} · ✓ = has local data)</span>
          <div className="picker-actions">
            <button type="button" className="link" onClick={() => setSelSyms(symOpts)}>all</button>
            <button type="button" className="link" onClick={() => setSelSyms(localSyms)}>with data</button>
            <button type="button" className="link" onClick={() => setSelSyms([])}>none</button>
          </div>
        </label>
        {groups.map(([cls, list]) => (
          <div key={cls} style={{ marginTop: 6 }}>
            <div className="muted small" style={{ marginBottom: 2 }}>
              {cls}{" "}
              <button type="button" className="link" onClick={() => setSelSyms((c) => Array.from(new Set([...c, ...list])))}>+all</button>
            </div>
            <Chips opts={list} sel={selSyms} onToggle={toggle(setSelSyms)} local={localSet} />
          </div>
        ))}

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
          {fetchStatus?.active && (
            <button
              type="button"
              className="danger"
              disabled={fetchStatus.phase !== "capturing"}
              onClick={stopFetch}
            >
              {fetchStatus.phase === "publication_in_progress"
                ? "Atomic publication in progress"
                : fetchStatus.phase === "cancellation_requested"
                  ? "Cancellation requested"
                  : `Stop run ${fetchStatus.runId}`}
            </button>
          )}
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

      <SpreadStatsPanel />

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

/** The broker's REAL spread by UTC hour, recorded from the live tick stream.
 *  Shows why a flat backtest spread is optimistic and what value to set. */
function SpreadStatsPanel() {
  const [stats, setStats] = useState<SpreadStats | null>(null);
  useEffect(() => {
    spreadStats().then(setStats).catch(() => {});
  }, []);
  const symbols = Object.entries(stats?.symbols ?? {}).filter(([, v]) => v.hourly?.some((h) => h.samples > 0));
  if (symbols.length === 0) {
    return (
      <>
        <h2>Real spread by hour (recorded)</h2>
        <p className="muted small">
          Recording started — the app samples your broker's live bid/ask once a minute and builds a
          per-hour spread profile here (used to sanity-check the backtest's cost assumption). Come
          back after a few hours of the app running with the tick stream live.
        </p>
      </>
    );
  }
  return (
    <>
      <h2>Real spread by hour (recorded from your broker)</h2>
      <p className="muted small">
        Mean pips per UTC hour · red = ≥2× the tightest hour (times a flat backtest spread underprices).
        Use this to set an honest <code>backtest_spread_pips</code>.
      </p>
      <table className="tbl" style={{ fontSize: 11 }}>
        <thead>
          <tr>
            <th>Symbol</th>
            {Array.from({ length: 24 }, (_, h) => <th key={h}>{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {symbols.map(([sym, v]) => {
            const means = v.hourly.map((h) => (h.samples > 0 ? h.meanPips : null));
            const tightest = Math.min(...means.filter((m): m is number => m != null && m > 0));
            return (
              <tr key={sym}>
                <td><b>{sym}</b></td>
                {means.map((m, h) => (
                  <td key={h} className={m != null && isFinite(tightest) && m >= tightest * 2 ? "sell" : ""} title={m != null ? `${sym} ${h}:00 UTC — mean ${m.toFixed(2)} pips (max ${v.hourly[h].maxPips.toFixed(1)}, n=${v.hourly[h].samples})` : "no samples"}>
                    {m != null ? m.toFixed(1) : "·"}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}
