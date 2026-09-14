import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

const tauriSource = readFileSync(
  new URL("../src-tauri/src/lib.rs", import.meta.url),
  "utf8",
);
const smokeSource = readFileSync(
  new URL("../src-tauri/examples/smoke.rs", import.meta.url),
  "utf8",
);
const discoverySource = readFileSync(
  new URL("../src/screens/Discovery.tsx", import.meta.url),
  "utf8",
);
const discoveryParametersSource = readFileSync(
  new URL("../src/components/DiscoveryParameters.tsx", import.meta.url),
  "utf8",
);
const settingsSource = readFileSync(
  new URL("../src/screens/Settings.tsx", import.meta.url),
  "utf8",
);
const advancedSource = readFileSync(
  new URL("../src/screens/Advanced.tsx", import.meta.url),
  "utf8",
);
const trainingSource = readFileSync(
  new URL("../src/screens/Training.tsx", import.meta.url),
  "utf8",
);
const dataSource = readFileSync(
  new URL("../src/screens/Data.tsx", import.meta.url),
  "utf8",
);
const dataImportPanelSource = readFileSync(
  new URL("../src/components/DataImportPanel.tsx", import.meta.url),
  "utf8",
);
const apiSource = readFileSync(
  new URL("../src/api.ts", import.meta.url),
  "utf8",
);
const apiContractsSource = readFileSync(
  new URL("../src/apiContracts.ts", import.meta.url),
  "utf8",
);
const timeframeContractUrl = new URL("../src/timeframes.ts", import.meta.url);
const timeframeContractSource = existsSync(timeframeContractUrl)
  ? readFileSync(timeframeContractUrl, "utf8")
  : "";
const selectSource = readFileSync(
  new URL("../src/components/Select.tsx", import.meta.url),
  "utf8",
);
const selectOptionsSource = readFileSync(
  new URL("../src/components/selectOptions.ts", import.meta.url),
  "utf8",
);
const filtersSource = readFileSync(
  new URL("../src/components/filterUtils.ts", import.meta.url),
  "utf8",
);
const cockpitSource = readFileSync(
  new URL("../src/screens/Cockpit.tsx", import.meta.url),
  "utf8",
);
const kChartSource = readFileSync(
  new URL("../src/components/KChart.tsx", import.meta.url),
  "utf8",
);

function between(source: string, start: string, end: string): string {
  const startAt = source.indexOf(start);
  const endAt = source.indexOf(end, startAt + start.length);
  assert.notEqual(startAt, -1, `missing start marker: ${start}`);
  assert.notEqual(endAt, -1, `missing end marker: ${end}`);
  return source.slice(startAt, endAt);
}

test("native picker advertises only the eight explicit import format filters", () => {
  const filters = between(
    tauriSource,
    "const IMPORT_PICKER_FILTERS",
    "async fn pick_data_file",
  );

  for (const name of [
    "CSV",
    "TSV",
    "JSON array",
    "JSON Lines",
    "Parquet",
    "Arrow IPC file",
    "Arrow IPC stream",
    "Vortex",
  ]) {
    assert.match(filters, new RegExp(`name: \\"${name}\\"`));
  }
  assert.equal((filters.match(/ImportPickerFilter \{/g) ?? []).length, 8);
  assert.doesNotMatch(filters, /"txt"|"\*"|All files/);

  const picker = between(tauriSource, "async fn pick_data_file", "struct SymbolCoverage");
  assert.match(picker, /for filter in IMPORT_PICKER_FILTERS/);
  assert.doesNotMatch(picker, /"txt"|"\*"|All files/);
});

test("native browse leaves the operator-selected import format authoritative", () => {
  const browse = between(dataImportPanelSource, "const browse = async", "const importSelectedFile = async");
  assert.doesNotMatch(browse, /setFormat|inferred|extension/);
});

test("every desktop timeframe picker consumes one exact 14-period broker contract", () => {
  const contract = between(
    timeframeContractSource,
    "export const CANONICAL_BROKER_TIMEFRAMES = [",
    "] as const",
  );
  const periods = [...contract.matchAll(/"([A-Z0-9]+)"/g)].map((match) => match[1]);
  assert.deepEqual(periods, [
    "M1", "M2", "M3", "M4", "M5", "M10", "M15", "M30",
    "H1", "H4", "H12", "D1", "W1", "MN1",
  ]);

  for (const source of [
    selectOptionsSource,
    filtersSource,
    cockpitSource,
    dataSource,
    discoverySource,
    kChartSource,
  ]) {
    assert.match(source, /CANONICAL_BROKER_TIMEFRAMES/);
    assert.doesNotMatch(source, /\b(?:CANON_TFS|BROKER_TFS)\b/);
    assert.doesNotMatch(source, /"(?:M6|M12|M20|H2|H3|H6|H8)"/);
  }
  assert.match(
    filtersSource,
    /TF_ORDER:\s*string\[\]\s*=\s*\[\.\.\.CANONICAL_BROKER_TIMEFRAMES\]\.reverse\(\)/,
  );
  assert.match(selectOptionsSource, /data\.timeframes\.length > 0/);
  assert.match(dataSource, /TF_SPEED:\s*string\[\]\s*=\s*\[\.\.\.CANONICAL_BROKER_TIMEFRAMES\]\.reverse\(\)/);
  assert.match(discoverySource, /import \{ tfRank \} from "\.\.\/components\/filterUtils"/);
  assert.match(discoverySource, /tfRank\(left\.timeframe \?\? ""\) - tfRank\(right\.timeframe \?\? ""\)/);
  assert.doesNotMatch(discoverySource, /const TF_SPEED|const tfRank/);
  assert.match(kChartSource, /Record<CanonicalBrokerTimeframe,/);
  assert.match(kChartSource, /isCanonicalBrokerTimeframe/);
  assert.doesNotMatch(kChartSource, /PERIOD\[timeframe\]\s*\?\?|TF_SECONDS\[timeframe\]\s*\?\?/);
});

test("Tauri data commands require exact identity plus generation and fully verify it", () => {
  assert.doesNotMatch(
    tauriSource,
    /async fn list_symbols\(|\bdiscover_symbols\(/,
    "the exact dataset inventory supersedes the symbol-only Tauri command",
  );

  const receipt = between(
    tauriSource,
    "struct ExactDatasetGenerationReceipt",
    "async fn list_timeframes",
  );
  assert.match(receipt, /CanonicalDatasetIdentity/);
  assert.match(receipt, /deserialize_with = "deserialize_canonical_dataset_identity"/);
  assert.match(receipt, /generation: String/);
  assert.match(receipt, /load_canonical_timeframe/);
  assert.match(receipt, /artifact\(\)\.generation_id\(\)/);
  assert.match(receipt, /selected generation receipt/);

  const timeframes = between(tauriSource, "async fn list_timeframes", "/// One OHLC bar");
  assert.match(timeframes, /selection: ExactDatasetGenerationReceipt/);
  assert.match(timeframes, /load_exact_dataset_generation/);
  assert.doesNotMatch(timeframes, /symbol: String|discover_timeframes/);

  const chart = between(tauriSource, "async fn chart", "struct ImportPickerFilter");
  assert.match(chart, /selection: ExactDatasetGenerationReceipt/);
  assert.match(chart, /load_exact_dataset_generation/);
  assert.doesNotMatch(chart, /symbol: String|timeframe: String|load_symbol_timeframe/);
});

test("coverage routes every exact receipt through typed summarization", () => {
  const command = between(tauriSource, "async fn data_coverage", "mod gate1_desktop_contract_tests");
  const summary = between(tauriSource, "fn summarize_exact_dataset_coverage", "async fn data_coverage");
  assert.match(command, /selections: Vec<ExactDatasetGenerationReceipt>/);
  assert.match(command, /load_exact_dataset_generation/);
  assert.match(command, /summarize_exact_dataset_coverage/);
  assert.doesNotMatch(command, /symbols: Vec<String>|timeframe: String|load_symbol_timeframe/);
  assert.doesNotMatch(command, /Err\(_\).*bars:\s*0/s);
  assert.doesNotMatch(command, /Err\(_\)/);
  assert.match(summary, /SymbolCoverage::Verified/);
  assert.match(summary, /SymbolCoverage::Failed/);
  assert.match(summary, /dataset_identity/);
  assert.match(summary, /generation/);
  assert.match(summary, /kind: "load_failed"/);
  assert.match(summary, /detail: format!\("\{error:#\}"\)/);
});

test("smoke selects and verifies one exact generation or exits with an error", () => {
  assert.match(smokeSource, /DatasetDiscovery::scan_metadata/);
  assert.match(smokeSource, /CanonicalDatasetIdentity::from_path_component/);
  assert.match(smokeSource, /load_canonical_timeframe/);
  assert.match(smokeSource, /artifact\(\)\.generation_id\(\)/);
  assert.match(smokeSource, /dataset_identity=/);
  assert.match(smokeSource, /generation=/);
  assert.match(smokeSource, /Result<\(\), Box<dyn std::error::Error>>/);
  assert.doesNotMatch(smokeSource, /discover_timeframes|load_symbol_timeframe|resampl/i);
});

test("desktop does not start training or Discovery from an ambiguous symbol-only selection", () => {
  assert.match(trainingSource, /trainingStart\(selected\.identity, mode\)/);
  assert.match(trainingSource, /trainingHandoffs/);
  assert.match(trainingSource, /oosCutoffMs/);
  assert.match(apiSource, /training_handoff: identity/);
  assert.doesNotMatch(trainingSource, /SymbolSelect|dataCoverage/);
  assert.doesNotMatch(discoverySource, /dataCoverage|symbolCoverageFailureText/);
});

test("bootstrap inventory carries exact identity, current generation, binding, and diagnostics", () => {
  const bootstrap = between(apiSource, "export type DataBootstrap =", "export const dataBootstrap");
  for (const field of ["datasetCount:", "datasets:", "skipped:"]) {
    assert.match(bootstrap, new RegExp(field));
  }
  const inventoryTypes = between(
    apiContractsSource,
    "export type DatasetInventoryEntry",
    "export type DiscoveryKnobs",
  );
  for (const field of [
    "datasetIdentity:",
    "generation:",
    "manifestBindingSha256:",
    "verification:",
    "category:",
    "detail:",
  ]) {
    assert.match(inventoryTypes, new RegExp(field));
  }
});

test("Discovery selects only exact inventory entries and renders skipped diagnostics verbatim", () => {
  assert.match(discoverySource, /dataBootstrap/);
  assert.match(discoverySource, /inventory\?\.datasets/);
  assert.match(discoverySource, /datasetIdentity/);
  assert.match(discoverySource, /entry\.generation/);
  assert.match(discoverySource, /entry\.datasetIdentity/);
  assert.match(discoverySource, /skipped\.map/);
  assert.match(discoverySource, /item\.detail/);

  const launch = between(
    discoverySource,
    "const startResearchRun = async",
    "const stopCurrentRun = async",
  );
  assert.doesNotMatch(launch, /config default|\[""\]|toUpperCase|toLowerCase/);
  assert.match(launch, /selectedDatasets\.length !== 1/);
  assert.match(launch, /discoveryStartBody\(selected, \{\}\)/);
});

test("Data displays authoritative current generations and all skipped diagnostics", () => {
  assert.match(dataSource, /data\.datasets\.map/);
  assert.match(dataSource, /entry\.datasetIdentity/);
  assert.match(dataSource, /entry\.generation/);
  assert.match(dataSource, /data\.skipped\.map/);
  assert.match(dataSource, /item\.detail/);
  assert.doesNotMatch(dataSource, /fileCount/);
});

test("Discovery keeps selected revision snapshots and blocks stale selections before dispatch", () => {
  assert.match(discoverySource, /useState<DatasetInventoryEntry\[\]>/);
  assert.doesNotMatch(discoverySource, /selectedDatasetIds|CanonicalDatasetIdentity\[\]/);
  const selection = between(discoverySource, "const selectedDatasets", "const discoveryState");
  assert.match(selection, /\(\) => selectedEntries/);
  assert.match(selection, /sameDiscoveryDatasetGeneration\(selected, entry\)/);
  const launch = between(discoverySource, "const startResearchRun = async", "const stopCurrentRun = async");
  assert.match(launch, /unavailableDatasets\.length > 0/);
  assert.match(discoverySource, /disabled=\{[^}]*unavailableDatasets\.length > 0\}/);
  assert.match(discoverySource, /toggleDiscoveryDatasetSelection\(current, entry\)/);
  assert.match(discoverySource, /checked=\{selectedDatasets\.some/);
  assert.match(discoverySource, /Your selected versions are not replaced automatically/);
  assert.match(discoverySource, /Clear unavailable selections/);
});

test("Data refreshes only an exact bootstrap receipt and exposes exact-run status and stop", () => {
  assert.match(apiContractsSource, /datasetSelection: SelectedDatasetGenerationV1 \| null/);
  assert.match(apiContractsSource, /selectedDatasetGenerationForBrokerFetch/);
  assert.match(apiContractsSource, /stopDataFetchFollowingActiveRun/);
  assert.doesNotMatch(apiContractsSource, /dataFetchIdentityKey/);
  assert.doesNotMatch(dataSource, /expectedGenerationFor/);
  assert.match(apiSource, /dataFetchStatus/);
  assert.match(apiSource, /stopActiveDataFetch/);
  assert.match(apiSource, /dataFetchStopOutcomeFromPayload/);
  assert.match(dataSource, /data\.datasets/);
  assert.match(dataSource, /selectedDatasetGenerationForBrokerFetch/);
  assert.match(dataSource, /selectedBrokerDatasetIds/);
  assert.match(dataSource, /type="radio"/);
  assert.match(dataSource, /selectedDatasetIdentity/);
  assert.match(dataSource, /fetchStatus\.runId/);
  assert.match(dataSource, /stopActiveDataFetch\(fetchStatus\.runId\)/);
  assert.doesNotMatch(dataSource, /stopDataFetch\(fetchStatus\.runId\)/);
  assert.match(dataSource, /brokerBlockedRetryAfterSeconds/);
});

test("single research admission sends the selected opaque identity and exact assertions", () => {
  assert.match(apiContractsSource, /manifest_binding_sha256/);
  assert.match(apiContractsSource, /dataset_selection/);
  assert.match(discoverySource, /discoveryStartBody/);
  assert.doesNotMatch(discoverySource, /resolve from config/);
});

test("Discovery exposes capability truth and contains no client-side serial queue", () => {
  assert.match(apiSource, /type EngineRunState/);
  assert.match(apiSource, /discoveryStartAvailable/);
  assert.match(apiSource, /discoveryStartMode: "ResearchOnly"/);
  assert.match(apiSource, /historicalEvaluationAvailable: boolean/);
  assert.match(apiSource, /canonicalNativeResearch/);
  assert.match(discoverySource, /discoveryStartAvailable/);
  assert.match(discoverySource, /discoveryStartMode === "ResearchOnly"/);
  assert.match(discoverySource, /Start strategy research/);
  assert.match(discoverySource, /historicalEvaluationUnavailableReason/);
  assert.match(discoverySource, /canonicalNativeResearch/);
  assert.match(discoverySource, /no parallel batch admission/);
  assert.doesNotMatch(discoverySource, /setQueue|startQueue|queueTerminalOutcome|\bdrive\(/);
  assert.equal(existsSync(new URL("../src/discoveryQueue.ts", import.meta.url)), false);
  assert.equal(existsSync(new URL("../src/discoveryQueueState.ts", import.meta.url)), false);
});

test("Discovery keeps unknown progress indeterminate and renders the worker counters", () => {
  assert.match(apiSource, /discoveryPercent\?: number \| null/);
  assert.match(discoverySource, /Number\.isFinite\(engines\.discoveryPercent\)/);
  assert.match(discoverySource, /Progress not yet measurable/);
  assert.match(discoverySource, /<progress[^>]*value=\{discoveryPercent\}/);
  assert.match(discoverySource, /const counterRows = discoveryCounterRows\(engines\?\.discoveryCounters\)/);
  const renderedCounters = between(
    discoverySource,
    "{counterRows.map((counter) => (",
    "</tbody>",
  );
  assert.match(renderedCounters, /<tr key=\{counter\.name\}>/);
  assert.match(renderedCounters, /<th scope="row">\{counter\.label\}<\/th>/);
  assert.match(renderedCounters, /<td>\{counter\.value === null \? "Unavailable" : counter\.value\.toLocaleString\(\)\}<\/td>/);
  assert.doesNotMatch(discoverySource, /discoveryPercent\s*\?\?\s*0/);
});

test("Discovery does not label an active run as waiting for admission or an unknown backend as idle", () => {
  assert.match(discoverySource, /discoveryRunning\s*\? "Research is running\./);
  assert.match(discoverySource, /!discoveryRunning && !discoveryAvailable && engines\?\.discoveryStartUnavailableReason/);
  assert.match(discoverySource, /engines\?\.discovery \?\? "Unknown"/);
  assert.doesNotMatch(discoverySource, /Waiting for research admission\./);
});

test("Discovery owns the editable search objective, sizing band, and breadth controls", () => {
  assert.match(discoverySource, /<DiscoveryParameters onReadinessChange=\{setParametersReady\} \/>/);
  assert.match(discoverySource, /!parametersReady \|\| engineError \|\| inventoryError/);
  for (const contractField of [
    "riskySearchRiskMin",
    "riskySearchRiskMax",
    "propFirmSearchRiskMin",
    "propFirmSearchRiskMax",
    "searchHighQualityConfidence",
    "propFirmSearchProfitTargetPct",
    "propFirmSearchMaxDailyLossPct",
    "propFirmSearchMaxDrawdownPct",
    "searchDevice",
    "searchPopulationAuto",
    "searchGenerations",
    "searchMaxIndicators",
    "prefilterTopK",
  ]) {
    assert.match(discoveryParametersSource, new RegExp(contractField));
    assert.match(apiSource, new RegExp(contractField));
  }
  assert.match(discoveryParametersSource, /Today this band is a run-wide sizing scenario, not a gene/);
  assert.match(discoveryParametersSource, /full financial-evaluation lane uses the[\s\S]*same resolved band in search and back\/forward validation/);
  assert.match(discoveryParametersSource, /canonical native lane uses[\s\S]*only for Generation 0 until its validation consumer is connected/);
  assert.match(discoveryParametersSource, /Fixed live risk is separate/);
  assert.match(discoveryParametersSource, /No research contract was sealed and no run was started/);
  assert.doesNotMatch(settingsSource, /<h2>Discovery mode<\/h2>|<h2>Risky goal<\/h2>|<h2>Search tuning/);
  assert.doesNotMatch(advancedSource, /Risky goal|Discovery search|Anti-stagnation \(GA tuning\)|riskPerTrade|searchPopulation/);
  assert.match(discoveryParametersSource, /CUDA required \(fail closed\)/);
  assert.match(settingsSource, /Search-time risk, objectives and validation windows now live in/);
});

test("desktop contains no resample UI, help, or fallback", () => {
  for (const source of [
    discoverySource,
    trainingSource,
    dataSource,
    dataImportPanelSource,
    apiSource,
    timeframeContractSource,
    selectSource,
    cockpitSource,
    kChartSource,
  ]) {
    assert.doesNotMatch(source, /resampl/i);
  }
  assert.doesNotMatch(discoverySource, /config default/i);
  assert.match(discoverySource, /exact canonical generations/i);
  assert.match(discoverySource, /Use Data to download or import them/i);
});

test("Cockpit visibly distinguishes the UI update stream from broker freshness", () => {
  const streamStatus = between(cockpitSource, '<span className={`stream-pill', "</span>");
  assert.match(streamStatus, /connected \? "● UI update stream connected" : "○ UI update stream disconnected"/);
  assert.match(streamStatus, /title="This is the UI event-stream connection, not a broker-freshness guarantee\."/);
  assert.doesNotMatch(streamStatus, /"[●○] stream (?:connected|disconnected)"/);
});

test("broker history and margin distinguish units from metadata-backed lots", () => {
  const account = readFileSync(new URL("../src/screens/Account.tsx", import.meta.url), "utf8");
  for (const field of ["volumeRawCentiUnits", "volumeUnits", "lotSizeRawCentiUnits", "lotSizeObservedAtUnixMs", "lotSizeError"]) {
    assert.match(apiSource, new RegExp(field));
    assert.match(account, new RegExp(field));
  }
  assert.match(account, /entry\.lotSizeRawCentiUnits != null \? quantity\(entry\.volumeLots\) : "—"/);
  assert.match(account, /order\.lotSizeRawCentiUnits != null \? quantity\(order\.volumeLots\) : "—"/);
  assert.match(account, /order\.executedVolumeUnits/);
  assert.match(account, /order\.executedVolumeLots/);
  assert.match(account, /does not prove the contract size at the historical fill/);
  assert.match(account, /This response is incomplete/);
  assert.doesNotMatch(account, /orders\.slice/);
});

test("Research inventory labels handoff targets separately from saved Discovery reports", () => {
  const inventory = readFileSync(new URL("../src/screens/Intelligence.tsx", import.meta.url), "utf8");
  const workspace = readFileSync(new URL("../src/screens/ResearchWorkspace.tsx", import.meta.url), "utf8");
  assert.match(inventory, /HANDOFF TARGETS<\/div><div className="card-value">\{data\.discoveryTargets\.length\}/);
  assert.match(inventory, /<h2>Training handoff targets<\/h2>/);
  assert.match(inventory, /published Discovery-to-Training handoffs/);
  assert.match(inventory, /For saved Discovery reports, open the Results tab; those reports are not counted here\./);
  const empty = between(inventory, "{data.discoveryTargets.length === 0 ? (", ") : (");
  assert.match(empty, /No published training handoff targets found\./);
  assert.doesNotMatch(inventory, /Discovered strategies|No discovered strategies yet|run Discovery first/);
  assert.doesNotMatch(inventory, /WF SPLITS|WF ACCURACY|walkforwardSplits|walkforwardAvgAccuracy/);
  assert.doesNotMatch(apiSource, /walkforwardSplits|walkforwardAvgAccuracy/);
  assert.match(workspace, /description: "Inspect stored model artifacts and published training handoff targets\."/);
  assert.match(workspace, /id: "results",\s+label: "Results"/);
});

test("Settings separates configured data from resolved storage and labels binary RAM units", () => {
  assert.match(settingsSource, /Configured historical-data directory:/);
  assert.match(settingsSource, /Models, cache and logs can use separate directories/);
  assert.doesNotMatch(settingsSource, /journal all live under/);
  const hardware = readFileSync(new URL("../src/screens/Hardware.tsx", import.meta.url), "utf8");
  assert.match(hardware, /mib \/ 1024/);
  assert.match(hardware, /GiB/);
  assert.doesNotMatch(hardware, /\)\} GB`/);
  for (const field of ["totalMb", "usedMb", "availableMb"]) {
    assert.ok(hardware.includes(`gib(data.ram.${field})`));
  }
});

test("Settings distinguishes sizing calibration from final OOS and explicit model-off behavior", () => {
  assert.match(settingsSource, /selection-validation calibration/);
  assert.match(settingsSource, /not the independent\s+final OOS test/);
  assert.doesNotMatch(settingsSource, /stored OOS evidence|trading continues gene-only/);
  assert.match(settingsSource, /Missing or mismatched models refuse engine startup/);
  assert.match(settingsSource, /no automatic gene-only fallback/);
  assert.match(settingsSource, /When disabled, entries use strategy genes without model gating/);
  assert.match(settingsSource, /All other trading checks\s+still apply/);
  assert.match(settingsSource, /Manual orders require a stop-loss when/);
  assert.match(settingsSource, /When OFF, new manual orders still require SL or TP unless submitted with the explicit/);
  assert.doesNotMatch(settingsSource, /without a stop is <b>refused/);
});

test("Supervisor exposes account scope and exact research selectors rather than legacy starts", () => {
  const supervisor = readFileSync(new URL("../src/screens/Supervisor.tsx", import.meta.url), "utf8");
  const observation = between(apiSource, "export type SupervisorObservation =", "export type SupervisorStatus =");
  assert.match(observation, /account: AccountStreamSnap \| null/);
  assert.match(supervisor, /account\.sourceAccountId/);
  assert.match(supervisor, /account\.sourceEnvironment/);
  assert.match(supervisor, /exact saved dataset generation/);
  assert.match(supervisor, /exact published Discovery handoff/);
  assert.match(supervisor, /Research results are not trading permission/);
  assert.match(supervisor, /engine account ownership is unverified/);
  assert.doesNotMatch(supervisor, /No running local engine reports ownership/);
});

test("Storage distinguishes complete, partial and unavailable metadata and hides stale rows on error", () => {
  const files = readFileSync(new URL("../src/screens/Files.tsx", import.meta.url), "utf8");
  assert.match(apiSource, /scanStatus: "complete" \| "partial" \| "missing" \| "unavailable"/);
  assert.match(files, /e\.scanError/);
  assert.match(files, /error \? undefined : data/);
  assert.match(files, /e\.scanStatus === "partial" \? "Observed "/);
  assert.match(files, /e\.kind === "secret" \|\| !measured/);
});

test("Results exposes IS diagnostics without dead OOS filters or invented validation verdicts", () => {
  const report = readFileSync(new URL("../src/screens/StrategyReport.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(report, /validOnly|hideFlagged|OOS-screened|cpcvPassed|walkforwardPassed|validationComplete/);
  assert.match(report, /rep\.flags\.map/);
  assert.match(report, /s\.flags\.join/);
  assert.match(report, /Unsealed IS diagnostics only/);
  assert.match(report, /does not establish whether independent final-window/);
  assert.match(report, /Final evaluation tab/);
  assert.match(discoverySource, /screens the calibration window/);
  assert.match(discoverySource, /reserved final window is evaluated separately/);
  assert.doesNotMatch(discoverySource, /checks the holdout|holdout diagnostics/);
});

test("one unavailable handoff remains diagnostic without hiding valid evaluation choices", () => {
  const inventory = readFileSync(new URL("../src/screens/Intelligence.tsx", import.meta.url), "utf8");
  const workspace = readFileSync(new URL("../src/screens/ResearchWorkspace.tsx", import.meta.url), "utf8");
  assert.match(apiSource, /trainingHandoffUnavailable: \{ identity: string; reason: string \}\[\]/);
  assert.match(trainingSource, /inventory\?\.trainingHandoffUnavailable \?\? \[\]/);
  assert.match(trainingSource, /unavailableHandoffs\.map/);
  assert.match(trainingSource, /Valid selections remain available/);
  assert.match(trainingSource, /No verified handoff is available/);
  assert.match(inventory, /data!\.trainingHandoffUnavailable\.map/);
  assert.match(workspace, /label: "Final evaluation"/);
  const startGuard = between(trainingSource, "const canStart =", "useEffect(");
  assert.match(startGuard, /Boolean\(selected\)/);
  assert.doesNotMatch(startGuard, /unavailableHandoffs/, "an unrelated invalid file cannot veto a valid selected handoff");
});

test("Autopilot distinguishes in-sample research screening from independent OOS", () => {
  const autopilot = readFileSync(new URL("../src/screens/Autopilot.tsx", import.meta.url), "utf8");
  assert.match(autopilot, /Research-screened only/);
  assert.match(autopilot, /Passed stored in-sample CPCV \/ walk-forward checks; not independent OOS or promotion approval\./);
  assert.doesNotMatch(autopilot, /OOS-screened|[Oo]osScreened/);
  assert.ok(autopilot.includes('if (s.cpcvPassed && s.walkforwardPassed) keys.add(`${s.symbol}|${s.timeframe}`);'));
  assert.ok(autopilot.includes('if (onlyResearchScreened && !researchScreenedKeys.has(`${p.symbol ?? ""}|${p.baseTf ?? ""}`)) return false;'));
  assert.match(autopilot, /setResearchScreenedKeys\(keys\)/);
  assert.match(autopilot, /checked=\{onlyResearchScreened\} onChange=\{\(e\) => setOnlyResearchScreened\(e.target.checked\)\}/);
  assert.match(autopilot, /setOnlyResearchScreened\(false\)/);
});
