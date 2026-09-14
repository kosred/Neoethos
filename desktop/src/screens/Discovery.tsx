import { useMemo, useState } from "react";
import {
  dataBootstrap,
  discoveryStart,
  discoveryStop,
  enginesStatus,
} from "../api";
import {
  discoveryStartBody,
  sameDiscoveryDatasetGeneration,
  toggleDiscoveryDatasetSelection,
  type DatasetInventoryEntry,
} from "../apiContracts";
import { DiscoveryParameters } from "../components/DiscoveryParameters";
import { tfRank } from "../components/filterUtils";
import { discoveryInventoryPage } from "../discoveryInventory";
import { discoveryCounterRows, discoveryGenerationProgress, researchEngineStatus } from "../runtimeStatus";
import { HelpPanel, HelpStep } from "../components/Help";
import { usePoll } from "../hooks";
import { CANONICAL_BROKER_TIMEFRAMES } from "../timeframes";

const stateClass = (state: string) =>
  state === "Running" || state === "Queued"
    ? "live"
    : state === "Succeeded" || state === "Published"
      ? "demo"
      : "";

export default function Discovery() {
  const { data: engines, error: engineError } = usePoll(enginesStatus, 2_000);
  const { data: inventory, error: inventoryError, reload: reloadInventory } =
    usePoll(dataBootstrap, 5_000);
  const [selectedEntries, setSelectedEntries] = useState<DatasetInventoryEntry[]>([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [parametersReady, setParametersReady] = useState(false);
  const [inventoryQuery, setInventoryQuery] = useState("");
  const [inventoryTimeframe, setInventoryTimeframe] = useState("");
  const [inventoryPage, setInventoryPage] = useState(0);

  const inventoryEntries = useMemo(() => inventory?.datasets ?? [], [inventory]);
  const visibleInventory = useMemo(
    () => discoveryInventoryPage(inventoryEntries, inventoryQuery, inventoryTimeframe, inventoryPage),
    [inventoryEntries, inventoryQuery, inventoryTimeframe, inventoryPage],
  );
  const selectedDatasets = useMemo(
    () => selectedEntries
      .slice()
      .sort((left, right) => {
        const symbolOrder = (left.symbol ?? "").localeCompare(right.symbol ?? "");
        if (symbolOrder !== 0) return symbolOrder;
        return tfRank(left.timeframe ?? "") - tfRank(right.timeframe ?? "");
      }),
    [selectedEntries],
  );
  const unavailableDatasets = useMemo(
    () => selectedDatasets.filter((selected) =>
      !inventoryEntries.some((entry) => sameDiscoveryDatasetGeneration(selected, entry)),
    ),
    [inventoryEntries, selectedDatasets],
  );

  const discoveryState = engines?.discovery ?? "Unknown";
  const discoveryStatus = researchEngineStatus(discoveryState);
  const discoveryRunning = discoveryState === "Running";
  const researchRoute = engines?.discoveryStartMode === "ResearchOnly";
  const discoveryAvailable = researchRoute && engines?.discoveryStartAvailable === true;
  const discoveryPercent = typeof engines?.discoveryPercent === "number"
    && Number.isFinite(engines.discoveryPercent)
    ? Math.max(0, Math.min(100, engines.discoveryPercent))
    : undefined;
  const native = engines?.canonicalNativeResearch;
  const generationProgress = discoveryGenerationProgress(engines?.discoveryCounters);
  const counterRows = discoveryCounterRows(engines?.discoveryCounters);
  const workingSetBatch = counterRows.find((counter) => counter.name === "working_set_batch");
  const progressLabel = workingSetBatch ? "Current batch progress" : "Overall progress";
  const nativeRunning = native?.state === "Running" || native?.state === "Queued";
  const ramTotal = engines?.ramTotalGb ?? 0;
  const ramAvailable = engines?.ramAvailableGb ?? 0;
  const ramUsedPercent = ramTotal > 0
    ? Math.max(0, Math.min(100, ((ramTotal - ramAvailable) / ramTotal) * 100))
    : 0;

  const toggleDataset = (entry: DatasetInventoryEntry) => {
    setSelectedEntries((current) => toggleDiscoveryDatasetSelection(current, entry));
  };

  const startResearchRun = async () => {
    const selected = selectedDatasets[0];
    if (!selected || selectedDatasets.length !== 1 || unavailableDatasets.length > 0 || busy || discoveryRunning || !discoveryAvailable || !parametersReady || engineError || inventoryError) return;
    setBusy(true);
    setMessage(`Requesting exact ${selected.symbol} ${selected.timeframe} run…`);
    try {
      await discoveryStart(discoveryStartBody(selected, {}));
      setMessage("Research request accepted for the exact dataset generation. Input preparation and search progress appear above; this does not authorize trading.");
    } catch (error) {
      setMessage(`Search start failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const stopCurrentRun = async () => {
    setBusy(true);
    try {
      const result = await discoveryStop();
      setMessage(result.running
        ? "Cancellation requested. The worker remains active until it has stopped safely."
        : "No active research run remains to stop.");
    } catch (error) {
      setMessage(`Stop failed: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>Strategy search</h1>
      <p className="sub">
        Indicator + SMC search · exact data · separate execution validation
      </p>

      <HelpPanel id="discovery">
        <p>
          Search combines indicators and SMC using the saved objectives and risk settings.
          Select an exact dataset generation, then start the research run.
        </p>
        <HelpStep n={1}>Select exact canonical generations, not a symbol/timeframe filename guess.</HelpStep>
        <HelpStep n={2}>Discovery prepares inputs, searches candidates and screens the calibration window using research cost assumptions. The reserved final window is evaluated separately from the selected handoff. ResearchOnly means the result is not approved for live trading.</HelpStep>
        <HelpStep n={3}>Exact bid/ask fills and historical execution costs need their own validation. The CUDA Generation-0 lane below is a separate kernel-research path, not the complete strategy search.</HelpStep>
      </HelpPanel>

      <DiscoveryParameters onReadinessChange={setParametersReady} />

      <div className="res-strip" aria-label="Current machine resources">
        <div className="res-item">
          <div className="res-label">{engines ? `RAM ${ramAvailable.toFixed(1)} GB free of ${ramTotal.toFixed(1)} GB` : "RAM availability unknown"}</div>
          <div className="res-bar"><div className="res-fill" style={{ width: `${ramUsedPercent}%` }} /></div>
        </div>
        <div className="res-item res-disk">
          <div className="res-label">Active Vortex scratch</div>
          <div className="res-value">{engines ? `${((engines.featureStoreMb ?? 0) / 1024).toFixed(2)} GiB` : "—"}</div>
        </div>
      </div>

      <h2>Execution lanes</h2>
      <div className="engine-lanes">
        <section className="ticket engine-lane" aria-labelledby="native-research-lane">
          <div className="engine-lane-head">
            <div>
              <h3 id="native-research-lane">Canonical CUDA · ResearchOnly</h3>
              <p className="muted small">Sealed contract → device Generation 0 → evidence receipt</p>
            </div>
            <span className={`badge ${stateClass(native?.state ?? "Unknown")}`}>{native?.state ?? "Unknown"}</span>
          </div>
          <p className={native?.available ? "buy" : "muted"}>
            {native?.availabilityDetail ?? "Reading native runtime capability…"}
          </p>
          {native?.stage && <p><b>Stage:</b> {native.stage} · {native.percent.toFixed(1)}%</p>}
          {nativeRunning && <progress max={100} value={native?.percent ?? 0}>{native?.percent ?? 0}%</progress>}
          {native?.failureDetail && (
            <div className="banner warn" role="alert">
              {native.failureStage ?? "native research"} · {native.failureCode ?? "failed"}: {native.failureDetail}
            </div>
          )}
          {native?.published && (
            <div className="banner info">
              Published <code>{native.published.relativePath}</code><br />
              <span className="small">engine {native.published.engine} · population {native.published.resolvedPopulation.toLocaleString()} · device {native.published.selectedDeviceOrdinal}</span>
            </div>
          )}
          <p className="muted small">
            Dataset rows on this screen are not yet wired to the sealed-contract builder, so the UI
            does not fabricate a native start request from incomplete inputs.
          </p>
        </section>

        <section className="ticket engine-lane" aria-labelledby="strategy-discovery-lane">
          <div className="engine-lane-head">
            <div>
              <h3 id="strategy-discovery-lane">Strategy discovery · ResearchOnly</h3>
              <p className="muted small">Input preparation → indicator + SMC search → calibration screening</p>
            </div>
            <span className={`badge ${discoveryStatus.badgeClass}`}>{discoveryStatus.label}</span>
          </div>
          <p className={discoveryAvailable ? "buy" : "muted"}>
            {discoveryRunning
              ? "Research is running. Input preparation and search progress appear below."
              : discoveryAvailable
                ? "Research requests are available. Selected data and cost sources are checked at start."
                : engines ? "New research requests are currently unavailable." : "Waiting for backend status."}
          </p>
          {engines && !researchRoute && (
            <div className="banner warn" role="alert">This backend does not advertise the ResearchOnly start contract. Restart with the matching application build before starting.</div>
          )}
          {!discoveryRunning && !discoveryAvailable && engines?.discoveryStartUnavailableReason && (
            <div className="banner warn" role="alert">
              {engines.discoveryStartUnavailableReason}
            </div>
          )}
          {(discoveryStatus.notice || engines?.discoverySummary) && (
            <div className={`banner ${discoveryStatus.warning ? "warn" : "info"}`} role={discoveryStatus.warning ? "alert" : "status"}>
              {discoveryStatus.notice && <p>{discoveryStatus.notice}</p>}
              <div style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{engines?.discoverySummary}</div>
            </div>
          )}
          {discoveryRunning && (
            <>
              <p><b>{progressLabel}:</b> {discoveryPercent === undefined ? "Progress not yet measurable" : `${discoveryPercent.toFixed(1)}%`} · <b>Stage:</b> {engines?.discoveryStage || "running"}</p>
              <progress aria-label={progressLabel} max={100} value={discoveryPercent}>{discoveryPercent === undefined ? "Indeterminate" : `${discoveryPercent}%`}</progress>
              {engines?.discoveryStage === "search_generations" && generationProgress && (
                <p className="small"><b>Generations:</b> {generationProgress.completed.toLocaleString()} / {generationProgress.total.toLocaleString()} · {generationProgress.percent.toFixed(1)}% of generations</p>
              )}
              <button type="button" className="danger" disabled={busy} onClick={stopCurrentRun}>Stop current run</button>
            </>
          )}
          {counterRows.length > 0 && (
            <details open>
              <summary>Search counts · {discoveryRunning ? "current run" : "last reported run"}</summary>
              {workingSetBatch && (
                <p className="muted small">
                  Batch {workingSetBatch.value === null ? "unknown" : workingSetBatch.value.toLocaleString()}.
                  Population, generation and validation counts refer to this batch; batch totals refer to the whole run.
                  Completed batches include empty results. Saved reports are not trading approval.
                </p>
              )}
              <p className="muted small">Planned evaluations, unique candidates and validation tests are different counts. Capped or not-tested candidates did not fail OOS. Unreported stages remain unknown.</p>
              <table className="tbl" aria-label="Discovery candidate and validation counts">
                <thead><tr><th scope="col">Observation</th><th scope="col">Count</th></tr></thead>
                <tbody>
                  {counterRows.map((counter) => (
                    <tr key={counter.name}>
                      <th scope="row">{counter.label}</th>
                      <td>{counter.value === null ? "Unavailable" : counter.value.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          )}
          <details>
            <summary>Historical execution validation</summary>
            <p className="muted small">The Discovery start button does not yet run the reviewed bid/ask replay or its execution-cost ledger. Completing research is not proof of validated fills, profit, or readiness for live trading.</p>
            <p className="small">{engines?.historicalEvaluationAvailable === true ? "Historical financial capability is available separately." : "Historical financial capability is not available for this process."}</p>
            {engines?.historicalEvaluationUnavailableReason && <p className="muted small break-anywhere">{engines.historicalEvaluationUnavailableReason}</p>}
          </details>
        </section>
      </div>

      {engineError && <div className="banner warn" role="alert">{engineError}</div>}
      {inventoryError && <div className="banner warn" role="alert">{inventoryError}</div>}

      <div className="section-heading-row">
        <div>
          <h2>Research dataset plan</h2>
          <p className="muted small">{selectedDatasets.length} exact generation{selectedDatasets.length === 1 ? "" : "s"} selected</p>
        </div>
        <div className="btn-row">
          <button type="button" onClick={() => void reloadInventory()}>Refresh inventory</button>
          <button type="button" onClick={() => setSelectedEntries([])}>Clear</button>
        </div>
      </div>

      {unavailableDatasets.length > 0 && (
        <div className="banner warn" role="alert">
          <b>Selected data changed or is no longer available.</b>
          <p>Start is blocked. Select the current row again, or clear the unavailable selections below. Your selected versions are not replaced automatically.</p>
          <ul>
            {unavailableDatasets.map((entry) => (
              <li key={entry.datasetIdentity}>
                {entry.symbol} {entry.timeframe} · <code className="break-anywhere">{entry.generation}</code>
                <details>
                  <summary>Selected identity and manifest binding</summary>
                  <code className="break-anywhere">{entry.datasetIdentity}</code><br />
                  <code className="break-anywhere">{entry.manifestBindingSha256}</code>
                </details>
              </li>
            ))}
          </ul>
          <button type="button" onClick={() => setSelectedEntries((current) =>
            current.filter((selected) => inventoryEntries.some((entry) => sameDiscoveryDatasetGeneration(selected, entry))),
          )}>Clear unavailable selections</button>
        </div>
      )}

      <div className="ticket research-admission">
        {!parametersReady && <p className="banner warn">Search parameters must be loaded and saved before starting. Unsaved edits are not used by the engine.</p>}
        {selectedDatasets.length > 1 ? (
          <div className="banner warn">The backend has no parallel batch admission for these {selectedDatasets.length} exact generations. Select one generation to start.</div>
        ) : selectedDatasets.length === 1 ? (
          <p className="muted small">Start uses the saved search settings and the exact selected generation; it does not start training or trading.</p>
        ) : (
          <p className="muted small">Select one generation below to start research.</p>
        )}
        {selectedDatasets.map((entry) => (
          <div key={entry.datasetIdentity}>
            <b>{entry.symbol} · {entry.timeframe}</b> · {entry.sourceKind}
            <details><summary>Selected generation</summary><code className="break-anywhere">{entry.generation}</code></details>
            <button type="button" onClick={() => toggleDataset(entry)}>Remove {entry.symbol} {entry.timeframe}</button>
          </div>
        ))}
        <button
          type="button"
          className="primary"
          disabled={busy || discoveryRunning || !discoveryAvailable || !parametersReady || !!engineError || !!inventoryError || selectedDatasets.length !== 1 || unavailableDatasets.length > 0}
          onClick={startResearchRun}
        >
          {discoveryAvailable ? "Start strategy research" : "Research start unavailable"}
        </button>
        {message && <div className="banner info" role="status">{message}</div>}
      </div>

      <div className="btn-row" role="search" aria-label="Filter research datasets">
        <label>Find dataset
          <input type="search" value={inventoryQuery} placeholder="Symbol, source or exact identity"
            onChange={(event) => { setInventoryQuery(event.target.value); setInventoryPage(0); }} />
        </label>
        <label>Timeframe
          <select value={inventoryTimeframe} onChange={(event) => { setInventoryTimeframe(event.target.value); setInventoryPage(0); }}>
            <option value="">All timeframes</option>
            {CANONICAL_BROKER_TIMEFRAMES.map((tf) => <option key={tf} value={tf}>{tf}</option>)}
          </select>
        </label>
        <span className="muted small" role="status">{visibleInventory.matched} of {inventoryEntries.length} datasets · selections stay pinned when filtered</span>
      </div>

      {inventoryEntries.length === 0 ? (
        <div className="banner warn">{inventory && !inventoryError ? "No verified canonical datasets are available. Use Data to download or import them." : "Dataset inventory is not confirmed. Waiting for a successful backend response."}</div>
      ) : visibleInventory.matched === 0 ? (
        <div className="banner info">No datasets match these filters. Existing selections are unchanged.</div>
      ) : (
        <div className="table-scroll" tabIndex={0} aria-label="Canonical research dataset inventory">
          <table className="tbl">
            <thead>
              <tr><th>Select</th><th>Symbol</th><th>TF</th><th>Source</th><th>Exact generation details</th><th>Verification</th></tr>
            </thead>
            <tbody>
              {visibleInventory.entries.map((entry) => {
                const assertionMetadataMissing = entry.symbol === null || entry.timeframe === null;
                return (
                  <tr key={entry.datasetIdentity}>
                    <td>
                      <input
                        type="checkbox"
                        checked={selectedDatasets.some((selected) => sameDiscoveryDatasetGeneration(selected, entry))}
                        disabled={assertionMetadataMissing}
                        onChange={() => toggleDataset(entry)}
                        aria-label={`Select exact dataset ${entry.datasetIdentity}`}
                      />
                    </td>
                    <td>{entry.symbol ?? "missing"}</td>
                    <td>{entry.timeframe ?? "missing"}</td>
                    <td>{entry.sourceKind}</td>
                    <td>
                      <details>
                        <summary>Identity, generation and binding</summary>
                        <code className="break-anywhere">{entry.datasetIdentity}</code><br />
                        <code className="break-anywhere">{entry.generation}</code><br />
                        <code className="break-anywhere">{entry.manifestBindingSha256}</code>
                      </details>
                    </td>
                    <td>{assertionMetadataMissing ? "missing assertion metadata" : entry.verification}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {visibleInventory.pageCount > 1 && (
        <div className="btn-row" aria-label="Research dataset pages">
          <button type="button" disabled={visibleInventory.page === 0} onClick={() => setInventoryPage(visibleInventory.page - 1)}>Previous page</button>
          <span>Page {visibleInventory.page + 1} of {visibleInventory.pageCount}</span>
          <button type="button" disabled={visibleInventory.page + 1 >= visibleInventory.pageCount} onClick={() => setInventoryPage(visibleInventory.page + 1)}>Next page</button>
        </div>
      )}

      {(inventory?.skipped.length ?? 0) > 0 && (
        <details className="banner warn">
          <summary>Rejected or non-canonical entries ({inventory?.skipped.length})</summary>
          <ul>
            {inventory?.skipped.map((item) => (
              <li key={`${item.path}:${item.category}:${item.detail}`}>
                <code>{item.path}</code> — {item.category}: {item.detail}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
