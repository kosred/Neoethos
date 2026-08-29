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
const trainingSource = readFileSync(
  new URL("../src/screens/Training.tsx", import.meta.url),
  "utf8",
);
const dataSource = readFileSync(
  new URL("../src/screens/Data.tsx", import.meta.url),
  "utf8",
);
const helpSource = readFileSync(
  new URL("../src/screens/Help.tsx", import.meta.url),
  "utf8",
);
const apiSource = readFileSync(
  new URL("../src/api.ts", import.meta.url),
  "utf8",
);
const queueSource = readFileSync(
  new URL("../src/discoveryQueue.ts", import.meta.url),
  "utf8",
);
const queueStateSource = readFileSync(
  new URL("../src/discoveryQueueState.ts", import.meta.url),
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
const filtersSource = readFileSync(
  new URL("../src/components/filters.tsx", import.meta.url),
  "utf8",
);
const marketsSource = readFileSync(
  new URL("../src/screens/Markets.tsx", import.meta.url),
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
  const browse = between(discoverySource, "const browse = async", "const doImport = async");
  assert.doesNotMatch(browse, /setImpFormat|inferred|extension/);
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
    selectSource,
    filtersSource,
    marketsSource,
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
  assert.match(selectSource, /d\.timeframes\?\.length\s*\?\s*d\.timeframes/);
  assert.match(marketsSource, /tfs\.length\s*\?\s*tfs/);
  assert.match(dataSource, /TF_SPEED:\s*string\[\]\s*=\s*\[\.\.\.CANONICAL_BROKER_TIMEFRAMES\]\.reverse\(\)/);
  assert.match(discoverySource, /TF_SPEED:\s*string\[\]\s*=\s*\[\.\.\.CANONICAL_BROKER_TIMEFRAMES\]\.reverse\(\)/);
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

test("legacy Training coverage renders typed failure while Discovery never uses ambiguous coverage", () => {
  assert.match(trainingSource, /status === "failed"/);
  assert.match(trainingSource, /symbolCoverageFailureText/);
  assert.match(trainingSource, /coverage failed/);
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

test("Discovery queues only exact inventory entries and renders skipped diagnostics verbatim", () => {
  assert.match(discoverySource, /dataBootstrap/);
  assert.match(discoverySource, /inventory\.datasets/);
  assert.match(discoverySource, /datasetIdentity/);
  assert.match(discoverySource, /entry\.generation/);
  assert.match(discoverySource, /entry\.datasetIdentity/);
  assert.match(discoverySource, /skipped\.map/);
  assert.match(discoverySource, /item\.detail/);

  const launch = between(discoverySource, "const launch =", "const stop =");
  assert.doesNotMatch(launch, /config default|\[""\]|toUpperCase|toLowerCase/);
  assert.match(launch, /selectedDatasets/);
});

test("Data displays authoritative current generations and all skipped diagnostics", () => {
  assert.match(dataSource, /data\.datasets\.map/);
  assert.match(dataSource, /entry\.datasetIdentity/);
  assert.match(dataSource, /entry\.generation/);
  assert.match(dataSource, /data\.skipped\.map/);
  assert.match(dataSource, /item\.detail/);
  assert.doesNotMatch(dataSource, /fileCount/);
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

test("queue sends the selected opaque identity and keeps legacy fields as exact assertions", () => {
  assert.match(queueSource, /datasetIdentity:/);
  assert.match(queueSource, /generation:/);
  assert.match(apiContractsSource, /manifest_binding_sha256/);
  assert.match(apiContractsSource, /dataset_selection/);
  assert.match(queueSource, /discoveryStartBody/);
  assert.doesNotMatch(queueSource, /resolve from config/);
});

test("queue consumes the typed backend terminal state instead of a running boolean", () => {
  assert.match(queueStateSource, /type EngineRunState/);
  assert.match(queueSource, /queueTerminalOutcome/);
  assert.doesNotMatch(queueSource, /backendRunning:\s*boolean/);
  assert.match(discoverySource, /drive\(st\.discovery, summary\)/);
});

test("desktop contains no resample UI, help, or fallback", () => {
  for (const source of [
    discoverySource,
    trainingSource,
    dataSource,
    helpSource,
    apiSource,
    queueSource,
    timeframeContractSource,
    selectSource,
    marketsSource,
    cockpitSource,
    kChartSource,
  ]) {
    assert.doesNotMatch(source, /resampl/i);
  }
  const discoveryHelp = between(helpSource, '{ id: "discovery"', '{ id: "training"');
  assert.doesNotMatch(discoveryHelp, /config default/i);
  assert.match(discoveryHelp, /exact canonical dataset/i);
  assert.match(discoveryHelp, /downloaded or imported directly/i);
});
