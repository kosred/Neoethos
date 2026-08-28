import { useEffect, useState, useSyncExternalStore } from "react";
import {
  dataBootstrap,
  dataImport,
  enginesStatus,
  pickDataFile,
  riskInfo,
  settings,
  updateSettings,
} from "../api";
import { usePoll } from "../hooks";
import { TimeframeSelect } from "../components/Select";
import { HelpPanel, HelpStep, Tip } from "../components/Help";
import {
  dataImportIdentityKey,
  dataOperationErrorText,
  expectedGenerationFor,
  recordDatasetGeneration,
  type CanonicalDatasetIdentity,
  type DataImportBody,
  type DataImportSourceFormat,
  type DatasetInventoryEntry,
  type DatasetGenerationReceipts,
} from "../apiContracts";
import {
  subscribe,
  getSnapshot,
  setQueue,
  startQueue,
  stopQueue,
  clearQueue,
  drive,
  labelFor,
  type QItem,
} from "../discoveryQueue";
import { CANONICAL_BROKER_TIMEFRAMES } from "../timeframes";

// Fast-first TF order so the cheap, high-yield timeframes run before the
// dense ones (M5/M3 take hours) — you get strategies quickly and the slow
// units land last. Lower index = runs earlier.
const TF_SPEED: string[] = [...CANONICAL_BROKER_TIMEFRAMES].reverse();
const tfRank = (t: string) => {
  const i = TF_SPEED.indexOf(t);
  return i < 0 ? 99 : i;
};

const num = (s: string) => (s.trim() === "" ? undefined : Number(s));
const statusIcon: Record<QItem["status"], string> = {
  pending: "⏳",
  running: "▶",
  done: "✓",
  failed: "✗",
};

export default function Discovery() {
  const { data: st, error } = usePoll(enginesStatus, 2000);
  const { data: cfg, reload: reloadCfg } = usePoll(settings, 0);
  const {
    data: inventory,
    error: inventoryError,
    reload: reloadInventory,
  } = usePoll(dataBootstrap, 5_000);
  const q = useSyncExternalStore(subscribe, getSnapshot);
  const [selectedDatasetIds, setSelectedDatasetIds] =
    useState<CanonicalDatasetIdentity[]>([]);
  const [riskPct, setRiskPct] = useState<number | null>(null);
  const [adv, setAdv] = useState(false);
  const [population, setPopulation] = useState("");
  const [generations, setGenerations] = useState("");
  const [targets, setTargets] = useState("");
  const [portfolio, setPortfolio] = useState("");
  const [msg, setMsg] = useState("");

  // Import data file (lives here because data is only for search + training).
  const [impSrc, setImpSrc] = useState("");
  const [impFormat, setImpFormat] = useState<DataImportSourceFormat>("csv");
  const [impNamespace, setImpNamespace] = useState("operator-upload");
  const [impSym, setImpSym] = useState("EURUSD");
  const [impTf, setImpTf] = useState("H1");
  const [impTimestampConvention, setImpTimestampConvention] =
    useState<DataImportBody["barTimestampConvention"]>("bar_open");
  const [impMsg, setImpMsg] = useState("");
  const [impBusy, setImpBusy] = useState(false);
  const [importReceipts, setImportReceipts] = useState<DatasetGenerationReceipts>({});

  const browse = async () => {
    try {
      const p = await pickDataFile();
      if (p) setImpSrc(p);
    } catch (e) {
      setImpMsg(String(e));
    }
  };
  const doImport = async () => {
    if (!impSrc) { setImpMsg("Choose a file first (Browse…)."); return; }
    if (!impNamespace.trim()) { setImpMsg("Source namespace must be non-empty."); return; }
    const sourceNamespace = impNamespace.trim();
    const symbol = impSym.trim();
    const requestIdentity = dataImportIdentityKey(
      sourceNamespace,
      symbol,
      impTf,
      impTimestampConvention,
    );
    setImpBusy(true);
    setImpMsg("Importing…");
    try {
      const outcome = await dataImport(
        impSrc,
        impFormat,
        sourceNamespace,
        symbol,
        impTf,
        impTimestampConvention,
        expectedGenerationFor(importReceipts, requestIdentity),
      );
      setImportReceipts((current) =>
        recordDatasetGeneration(current, requestIdentity, outcome),
      );
      await reloadInventory();
      setImpMsg(
        `✓ Imported ${outcome.rowCount} rows as ${outcome.datasetIdentity} @ ${outcome.generation} → ${outcome.writtenPath}`,
      );
    } catch (e) {
      setImpMsg(`Import failed: ${dataOperationErrorText(e)}`);
    } finally {
      setImpBusy(false);
    }
  };

  const state = st?.discovery ?? "…";
  const running = state === "Running";
  const stage = st?.discoveryStage ?? st?.discovery_stage ?? "";
  const percent = st?.discoveryPercent ?? st?.discovery_percent ?? 0;
  const summary = st?.discoverySummary ?? st?.discovery_summary ?? "";
  const counters = st?.discoveryCounters ?? st?.discovery_counters ?? [];

  // RAM / disk readout (operator visibility).
  const ramTotal = st?.ramTotalGb ?? 0;
  const ramAvail = st?.ramAvailableGb ?? 0;
  const ramUsedPct = ramTotal > 0 ? ((ramTotal - ramAvail) / ramTotal) * 100 : 0;
  const diskMb = st?.featureStoreMb ?? 0;

  // Drive the queue forward on every poll tick.
  useEffect(() => {
    if (st) void drive(st.discovery, summary);
  }, [st]); // eslint-disable-line react-hooks/exhaustive-deps

  // Pre-flight risk visibility is independent from the exact data selector.
  useEffect(() => {
    let live = true;
    riskInfo().then((r) => { if (live) setRiskPct(r.riskPerTrade); }).catch(() => {});
    return () => { live = false; };
  }, []);

  const applyMode = async (m: "risky" | "prop_firm") => {
    try {
      await updateSettings({ tradingMode: m });
      reloadCfg?.();
    } catch { /* ignore */ }
  };
  const applyRisk = async (pctStr: string) => {
    const pct = Number(pctStr);
    if (!Number.isFinite(pct) || pct <= 0) return;
    try {
      await updateSettings({ riskPerTrade: pct / 100 });
      const r = await riskInfo();
      setRiskPct(r.riskPerTrade);
    } catch { /* ignore */ }
  };

  const inventoryEntries = inventory ? inventory.datasets : [];
  const selectedDatasets = inventoryEntries
    .filter((entry) => selectedDatasetIds.includes(entry.datasetIdentity))
    .slice()
    .sort((left, right) => {
      const symbolOrder = (left.symbol ?? "").localeCompare(right.symbol ?? "");
      if (symbolOrder !== 0) return symbolOrder;
      const timeframeOrder = tfRank(left.timeframe ?? "") - tfRank(right.timeframe ?? "");
      return timeframeOrder || left.datasetIdentity.localeCompare(right.datasetIdentity);
    });

  const toggleDataset = (entry: DatasetInventoryEntry) => {
    setSelectedDatasetIds((current) =>
      current.includes(entry.datasetIdentity)
        ? current.filter((identity) => identity !== entry.datasetIdentity)
        : [...current, entry.datasetIdentity],
    );
  };

  const queued = q.items.length;
  const done = q.items.filter((i) => i.status === "done").length;
  const failed = q.items.filter((i) => i.status === "failed").length;

  const launch = () => {
    if (selectedDatasets.length === 0) {
      setMsg("Select at least one exact canonical dataset generation from Data.");
      return;
    }
    setQueue(selectedDatasets, {
      population: num(population),
      generations: num(generations),
      target_candidates: num(targets),
      portfolio_size: num(portfolio),
    });
    startQueue();
    setMsg(
      `Queued ${selectedDatasets.length} exact dataset run${selectedDatasets.length === 1 ? "" : "s"}.`,
    );
  };

  const stop = async () => {
    await stopQueue();
    setMsg("Stopped — current run cancelled, queue cleared.");
  };

  return (
    <div className="screen">
      <h1>
        Discovery{" "}
        {cfg?.tradingMode && (
          <span className={`badge ${cfg.tradingMode === "risky" ? "live" : "demo"}`}>
            {cfg.tradingMode === "risky" ? "🚀 RISKY MODE" : "🛡 PROP-FIRM MODE"}
          </span>
        )}
      </h1>
      <p className="sub">
        Genetic strategy search · queue many pairs · <b>mode + tuning in Settings</b>
      </p>

      <HelpPanel id="discovery">
        <p>
          Discovery is the <b>strategy factory</b>. Select one or more exact canonical datasets from
          the authoritative Data inventory, press <b>Start queue</b>, and it runs each identity in turn.
        </p>
        <HelpStep n={1}>
          Select the exact <b>identity + current generation</b>. Symbol and timeframe labels are shown
          only as consistency assertions; they never choose a file.
        </HelpStep>
        <HelpStep n={2}>
          Every timeframe must be downloaded from the broker or imported directly at that same
          timeframe. Missing higher-timeframe data stops that run and reports the backend error.
        </HelpStep>
        <HelpStep n={3}>
          The <b>RAM / disk</b> strip shows what the run is consuming: cubes that fit in RAM use no
          disk; large ones stream to disk and are freed as each timeframe finishes. Results appear in{" "}
          <b>Strategy Lab</b> / <b>Autopilot</b>.
        </HelpStep>
        <p className="muted small">
          The engine runs in-process, so keep the app open while a queue runs. Leaving this screen is
          fine — the queue resumes when you return.
        </p>
      </HelpPanel>

      {/* ── Live machine-resource strip ── */}
      <div className="res-strip">
        <div className="res-item">
          <div className="res-label">
            RAM {ramAvail.toFixed(1)} GB free of {ramTotal.toFixed(0)} GB
          </div>
          <div className="res-bar">
            <div className="res-fill" style={{ width: `${Math.min(100, ramUsedPct)}%` }} />
          </div>
        </div>
        <div className="res-item res-disk">
          <div className="res-label">Discovery disk</div>
          <div className="res-value">
            {diskMb > 0 ? `${(diskMb / 1024).toFixed(2)} GB` : "0 (all in RAM)"}
          </div>
        </div>
      </div>

      {/* ── Currently running ── */}
      <div className="engine-status">
        <span className={`badge ${running ? "live" : "demo"}`}>
          {running ? "RUNNING" : state.toUpperCase()}
        </span>
        {stage && <span className="muted">{stage}</span>}
        {running && (
          <div className="progress">
            <div className="progress-bar" style={{ width: `${Math.min(100, percent)}%` }} />
            <span className="progress-label">{percent.toFixed(0)}%</span>
          </div>
        )}
      </div>
      {summary && <div className="banner info">{summary}</div>}
      {error && <div className="banner warn">{error}</div>}
      {inventoryError && <div className="banner warn">{inventoryError}</div>}

      {counters.length > 0 && (
        <div
          className="cards"
          style={{ gridTemplateColumns: `repeat(${Math.min(4, counters.length)}, 1fr)` }}
        >
          {counters.map((c) => (
            <div className="card" key={c.name}>
              <div className="card-label">{c.name.toUpperCase()}</div>
              <div className="card-value">{c.value.toLocaleString()}</div>
            </div>
          ))}
        </div>
      )}

      {/* ── Queue ── */}
      {queued > 0 && (
        <>
          <h2>
            Queue{" "}
            <span className="muted">
              — {done} done · {queued - done - failed} left
              {failed ? ` · ${failed} failed` : ""}
            </span>
          </h2>
          <div className="queue-list">
            {q.items.map((it) => (
              <div className={`q-item q-${it.status}`} key={it.id}>
                <span className="q-icon">{statusIcon[it.status]}</span>
                <span className="q-name">{labelFor(it.symbol ?? "?", it.timeframe ?? "?", it.generation)}</span>
                {it.status === "running" && (
                  <span className="q-prog">
                    <span className="q-bar">
                      <span className="q-fill" style={{ width: `${Math.min(100, percent)}%` }} />
                    </span>
                    {stage || "running"} · {percent.toFixed(0)}%
                  </span>
                )}
                {it.note && it.status !== "running" && (
                  <span className="q-note muted">{it.note}</span>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {/* ── Build a queue ── */}
      <h2>Build a queue</h2>
      <div className="ticket">
        <label className="picker-label">
          Exact canonical datasets <span className="muted">({selectedDatasets.length} selected)</span>
          <div className="picker-actions">
            <button
              type="button"
              className="link"
              onClick={() => setSelectedDatasetIds(
                inventoryEntries
                  .filter((entry) => entry.symbol !== null && entry.timeframe !== null)
                  .map((entry) => entry.datasetIdentity),
              )}
            >all</button>
            <button type="button" className="link" onClick={() => setSelectedDatasetIds([])}>none</button>
          </div>
        </label>
        {inventoryEntries.length === 0 ? (
          <div className="banner warn">
            No canonical dataset generation is available. Download or import each exact timeframe in Data.
          </div>
        ) : (
          <table className="tbl">
            <thead>
              <tr><th>Select</th><th>Symbol</th><th>TF</th><th>Current generation</th><th>Exact identity</th><th>Verification</th></tr>
            </thead>
            <tbody>
              {inventoryEntries.map((entry) => {
                const assertionMetadataMissing = entry.symbol === null || entry.timeframe === null;
                return (
                  <tr key={entry.datasetIdentity}>
                    <td>
                      <input
                        type="checkbox"
                        checked={selectedDatasetIds.includes(entry.datasetIdentity)}
                        disabled={assertionMetadataMissing}
                        onChange={() => toggleDataset(entry)}
                        aria-label={`Select exact dataset ${entry.datasetIdentity}`}
                      />
                    </td>
                    <td>{entry.symbol ?? "missing"}</td>
                    <td>{entry.timeframe ?? "missing"}</td>
                    <td><code>{entry.generation}</code></td>
                    <td><code style={{ overflowWrap: "anywhere" }}>{entry.datasetIdentity}</code></td>
                    <td>{assertionMetadataMissing ? "missing assertion metadata" : entry.verification}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}

        {(inventory?.skipped.length ?? 0) > 0 && (
          <div className="banner warn">
            <b>Import/download required or rejected data:</b>
            <ul>
              {inventory!.skipped.map((item) => (
                <li key={`${item.path}:${item.category}:${item.detail}`}>
                  <code>{item.path}</code> — {item.category}: {item.detail}
                </li>
              ))}
            </ul>
          </div>
        )}

        <label style={{ flexDirection: "row", alignItems: "center", gap: 6, marginTop: 12 }}>
          <input type="checkbox" checked={adv} onChange={(e) => setAdv(e.target.checked)} /> Advanced knobs
        </label>
        {adv && (
          <div className="ticket-row" style={{ marginTop: 8 }}>
            <label>Population<input type="number" min="0" step="50" value={population} placeholder="default" onChange={(e) => setPopulation(e.target.value)} /></label>
            <label>Generations<input type="number" min="0" step="10" value={generations} placeholder="default" onChange={(e) => setGenerations(e.target.value)} /></label>
            <label>Target candidates<input type="number" min="0" step="10" value={targets} placeholder="default" onChange={(e) => setTargets(e.target.value)} /></label>
            <label>Portfolio size<input type="number" min="0" step="1" value={portfolio} placeholder="default" onChange={(e) => setPortfolio(e.target.value)} /></label>
          </div>
        )}

        {/* ── Pre-flight: choose mode/risk + see EXACTLY what will run ── */}
        <h2 style={{ marginTop: 12 }}>Before you start — what THIS search will use</h2>
        <div className="ticket-row" style={{ alignItems: "flex-end" }}>
          <label>
            Mode <Tip text="Applies to THIS search (saved to config as system.trading_mode). Risky = aggressive account-multiplication, drawdown-agnostic, ranks by fastest compounding. Prop-firm = robust FTMO-style rules (low drawdown / daily-loss limits)." />
            <select value={cfg?.tradingMode ?? "risky"} onChange={(e) => applyMode(e.target.value as "risky" | "prop_firm")}>
              <option value="risky">🚀 Risky</option>
              <option value="prop_firm">🛡 Prop-firm</option>
            </select>
            {/* models.discovery_mode can override this switch and reach a mode
                (`strict`) it cannot express. The backend resolves and ships the
                effective value; showing only the switch would let this
                pre-flight promise one regime and run another. */}
            {cfg?.tradingModeDivergent && (
              <span className="sell small">
                ⚠ overridden — this search will run as <b>{String(cfg.effectiveDiscoveryMode)}</b>{" "}
                (models.discovery_mode = {String(cfg.discoveryMode)})
              </span>
            )}
          </label>
          {/* 💰 The old tooltip promised "this search + live sizing". It sets
              neither. `grep '\.risk_per_trade\b' crates/neoethos-search` →
              zero hits: the search samples from the risk BANDS
              (discovery.rs:813-821). And risky-mode live sizing substitutes
              its own ladder (live_trading.rs:1664-1680). It binds live sizing
              in prop-firm mode, and nothing else. */}
          <label>
            Risk %/trade{" "}
            <Tip
              text={
                cfg?.tradingMode === "risky"
                  ? "Does NOT apply to this search, and does NOT apply to live sizing while the mode is Risky. The search samples the configured risk bands, not this field; risky live sizing uses the engine's 30–50% ladder capped by Max portfolio risk. Saved to config.yaml, where it will bind if you switch to Prop-firm."
                  : "Live position sizing in Prop-firm mode: fraction of the account risked per trade, clamped to the account's max risk. It does NOT change what THIS search explores — the search samples the configured risk bands, not this field."
              }
            />
            <input
              type="number"
              step="0.1"
              min="0"
              style={{ width: 80 }}
              key={riskPct ?? "risk"}
              defaultValue={riskPct != null ? (riskPct * 100).toFixed(1) : ""}
              onBlur={(e) => applyRisk(e.target.value)}
            />
          </label>
          <div className="muted small" style={{ paddingBottom: 6 }}>
            <b>{selectedDatasets.length} exact run{selectedDatasets.length === 1 ? "" : "s"}</b>
            {cfg?.searchGenerations != null && (
              <> · <b>gen</b> {cfg.searchGenerations} · <b>pop</b> {cfg.searchPopulation} · <b>prefilter</b> {cfg.prefilterTopK}</>
            )}
          </div>
        </div>
        {selectedDatasets.length > 0 && (
          <table className="tbl">
            <thead><tr><th>Pair</th><th>TF</th><th>Current generation</th><th>Exact identity</th></tr></thead>
            <tbody>
              {selectedDatasets.map((entry) => (
                  <tr key={entry.datasetIdentity}>
                    <td><b>{entry.symbol}</b></td>
                    <td>{entry.timeframe}</td>
                    <td><code>{entry.generation}</code></td>
                    <td><code style={{ overflowWrap: "anywhere" }}>{entry.datasetIdentity}</code></td>
                  </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="muted small">
          The queue uses these exact opaque identities. Required higher timeframes are separate direct broker downloads/imports; missing ones fail visibly.
        </p>

        <div className="btn-row">
          <button className="primary" disabled={q.active || selectedDatasets.length === 0} onClick={launch}>
            {q.active ? "Queue running…" : "Start queue"}
          </button>
          <button className="danger" disabled={!q.active && !running} onClick={stop}>
            Stop
          </button>
          {queued > 0 && !q.active && (
            <button className="ghost" onClick={() => { clearQueue(); setMsg(""); }}>Clear</button>
          )}
        </div>
        {msg && <div className="banner info">{msg}</div>}
      </div>

      {/* ── Import data file (data is only for search + training) ── */}
      <h2>Import data file</h2>
      <div className="ticket">
        <p className="muted small">Bring in CSV, TSV, JSON, Parquet, Arrow IPC, or Vortex data. Choose Format explicitly; the filename extension never changes it automatically. A successful import publishes a verified canonical Vortex generation; runtime never opens the source format.</p>
        <div className="ticket-row">
          <button onClick={browse} disabled={impBusy}>Browse…</button>
          <label style={{ flex: 1 }}>
            File
            <input value={impSrc} onChange={(e) => setImpSrc(e.target.value)} placeholder="(choose a file with Browse…)" style={{ width: "100%" }} />
          </label>
          <label>
            Format
            <select value={impFormat} onChange={(e) => setImpFormat(e.target.value as DataImportSourceFormat)}>
              <option value="csv">CSV</option>
              <option value="tsv">TSV</option>
              <option value="json-array">JSON array</option>
              <option value="json-lines">JSON lines</option>
              <option value="parquet">Parquet</option>
              <option value="arrow-ipc-file">Arrow IPC file</option>
              <option value="arrow-ipc-stream">Arrow IPC stream</option>
              <option value="vortex">Vortex</option>
            </select>
          </label>
          <label>Source namespace<input value={impNamespace} onChange={(e) => setImpNamespace(e.target.value)} style={{ width: 130 }} placeholder="operator-upload" /></label>
          <label>Symbol<input value={impSym} onChange={(e) => setImpSym(e.target.value)} style={{ width: 90 }} placeholder="EURUSD" /></label>
          <label>TF<TimeframeSelect value={impTf} onChange={setImpTf} style={{ width: 80 }} /></label>
          <label>
            Timestamp meaning
            <select value={impTimestampConvention} onChange={(e) => setImpTimestampConvention(e.target.value as DataImportBody["barTimestampConvention"])}>
              <option value="bar_open">Bar open</option>
              <option value="bar_close">Bar close (rejected)</option>
              <option value="bar_end">Bar end (rejected)</option>
              <option value="unknown">Unknown (rejected)</option>
            </select>
          </label>
          <button className="primary" disabled={impBusy || !impSrc} onClick={doImport}>Import</button>
        </div>
        {impMsg && <div className="banner info">{impMsg}</div>}
      </div>
    </div>
  );
}
