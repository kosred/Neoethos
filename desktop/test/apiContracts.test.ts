import assert from "node:assert/strict";
import test from "node:test";

import {
  amendProtectionBody,
  brokerBlockedRetryAfterSeconds,
  brokerFetchSelectionKey,
  dataBatchResultText,
  dataFetchBody,
  dataFetchStopOutcomeFromPayload,
  dataImportBody,
  dataImportIdentityKey,
  dataOperationErrorText,
  discoveryStartBody,
  promoteStrategyBody,
  selectedDatasetGenerationForBrokerFetch,
  stopDataFetchFollowingActiveRun,
  symbolCoverageFailureText,
  type CanonicalDatasetIdentity,
  type DiscoveryKnobs,
} from "../src/apiContracts.ts";

test("amendProtectionBody serializes the camelCase server fixture byte-for-byte", () => {
  const body = amendProtectionBody(42, 1.07125, 1.0845, true);

  assert.equal(
    JSON.stringify(body),
    '{"positionId":42,"stopLossPrice":1.07125,"takeProfitPrice":1.0845,"trailingStopLoss":true}',
  );
});

test("dataImportBody serializes the camelCase server fixture byte-for-byte", () => {
  const body = dataImportBody(
    "C:/market-data/EURUSD.csv",
    "csv",
    "operator-upload",
    "EURUSD",
    "M5",
    "bar_open",
    null,
  );

  assert.equal(
    JSON.stringify(body),
    '{"sourcePath":"C:/market-data/EURUSD.csv","sourceFormat":"csv","sourceNamespace":"operator-upload","symbol":"EURUSD","timeframe":"M5","barTimestampConvention":"bar_open","expectedGeneration":null}',
  );
});

test("dataFetchBody serializes the camelCase server fixture byte-for-byte", () => {
  const body = dataFetchBody("EURUSD", "M5", 1_700_000_000_000, undefined, null);

  assert.equal(
    JSON.stringify(body),
    '{"symbol":"EURUSD","timeframe":"M5","fromMs":1700000000000,"datasetSelection":null}',
  );
});

test("dataFetchBody preserves the complete exact generation receipt on refresh", () => {
  const generation = `g1-${"ab".repeat(32)}.vortex`;
  const manifestBindingSha256 = "cd".repeat(32);
  const body = dataFetchBody(
    "EURUSD",
    "M5",
    1_700_000_000_000,
    undefined,
    {
      schema: "neoethos.selected-dataset-generation.v1",
      version: 1,
      dataset_identity: "d1-exact-broker-identity" as CanonicalDatasetIdentity,
      generation_id: generation,
      manifest_binding_sha256: manifestBindingSha256,
    },
  );

  assert.equal(
    JSON.stringify(body),
    `{"symbol":"EURUSD","timeframe":"M5","fromMs":1700000000000,"datasetSelection":{"schema":"neoethos.selected-dataset-generation.v1","version":1,"dataset_identity":"d1-exact-broker-identity","generation_id":"${generation}","manifest_binding_sha256":"${manifestBindingSha256}"}}`,
  );
});

test("data import request identity keys preserve every exact identity field", () => {
  assert.notEqual(
    dataImportIdentityKey("operator-A", "EURUSD", "M5", "bar_open"),
    dataImportIdentityKey("operator-B", "EURUSD", "M5", "bar_open"),
  );
  assert.notEqual(
    dataImportIdentityKey("operator-A", "EURUSD", "M5", "bar_open"),
    dataImportIdentityKey("operator-A", "eurusd", "M5", "bar_open"),
  );
});

test("broker refresh selection is rebuilt from bootstrap and never uses an external or ambiguous identity", () => {
  const generation = `g1-${"11".repeat(32)}.vortex`;
  const manifestBindingSha256 = "22".repeat(32);
  const broker = {
    datasetIdentity: "d1-broker" as CanonicalDatasetIdentity,
    generation,
    manifestBindingSha256,
    symbol: "EURUSD",
    timeframe: "M5",
    verification: "generation_verified" as const,
    sourceKind: "ctrader" as const,
  };
  const external = {
    ...broker,
    datasetIdentity: "d1-external" as CanonicalDatasetIdentity,
    sourceKind: "external" as const,
  };

  assert.deepEqual(
    selectedDatasetGenerationForBrokerFetch([external, broker], "eurusd", "m5"),
    {
      schema: "neoethos.selected-dataset-generation.v1",
      version: 1,
      dataset_identity: broker.datasetIdentity,
      generation_id: generation,
      manifest_binding_sha256: manifestBindingSha256,
    },
  );
  assert.equal(selectedDatasetGenerationForBrokerFetch([external], "EURUSD", "M5"), null);
  assert.throws(
    () => selectedDatasetGenerationForBrokerFetch(
      [broker, { ...broker, datasetIdentity: "d1-other-account" as CanonicalDatasetIdentity }],
      "EURUSD",
      "M5",
    ),
    /multiple exact cTrader dataset identities/i,
  );
});

test("an explicit cTrader identity selects its exact current generation receipt", () => {
  const first = {
    datasetIdentity: "d1-account-A" as CanonicalDatasetIdentity,
    generation: `g1-${"11".repeat(32)}.vortex`,
    manifestBindingSha256: "22".repeat(32),
    symbol: "EURUSD",
    timeframe: "M5",
    verification: "generation_verified" as const,
    sourceKind: "ctrader" as const,
  };
  const selected = {
    ...first,
    datasetIdentity: "d1-account-B" as CanonicalDatasetIdentity,
    generation: `g1-${"33".repeat(32)}.vortex`,
    manifestBindingSha256: "44".repeat(32),
  };

  assert.deepEqual(
    selectedDatasetGenerationForBrokerFetch(
      [first, selected],
      "eurusd",
      "m5",
      selected.datasetIdentity,
    ),
    {
      schema: "neoethos.selected-dataset-generation.v1",
      version: 1,
      dataset_identity: selected.datasetIdentity,
      generation_id: selected.generation,
      manifest_binding_sha256: selected.manifestBindingSha256,
    },
  );
  assert.throws(
    () => selectedDatasetGenerationForBrokerFetch(
      [first, selected],
      "EURUSD",
      "M5",
      "d1-not-in-this-pair" as CanonicalDatasetIdentity,
    ),
    /selected cTrader dataset identity.*does not match/i,
  );
  assert.equal(
    brokerFetchSelectionKey(" eurusd ", "m5"),
    brokerFetchSelectionKey("EURUSD", " M5 "),
  );
});

test("stale stop follows the exact active run once and never chases indefinitely", async () => {
  const calls: number[] = [];
  const cancelled = await stopDataFetchFollowingActiveRun(41, async (runId) => {
    calls.push(runId);
    return calls.length === 1
      ? { outcome: "stale_run", requestedRunId: 41, activeRunId: 52 }
      : { outcome: "cancelled", runId: 52 };
  });
  assert.deepEqual(calls, [41, 52]);
  assert.deepEqual(cancelled, { outcome: "cancelled", runId: 52 });

  const boundedCalls: number[] = [];
  const stillStale = await stopDataFetchFollowingActiveRun(61, async (runId) => {
    boundedCalls.push(runId);
    return runId === 61
      ? { outcome: "stale_run", requestedRunId: 61, activeRunId: 62 }
      : { outcome: "stale_run", requestedRunId: 62, activeRunId: 63 };
  });
  assert.deepEqual(boundedCalls, [61, 62]);
  assert.deepEqual(stillStale, {
    outcome: "stale_run",
    requestedRunId: 62,
    activeRunId: 63,
  });
});

test("only a structurally valid typed stop payload can drive the stale follow-up", () => {
  assert.deepEqual(
    dataFetchStopOutcomeFromPayload({
      outcome: "stale_run",
      requestedRunId: 7,
      activeRunId: 8,
    }),
    { outcome: "stale_run", requestedRunId: 7, activeRunId: 8 },
  );
  assert.equal(
    dataFetchStopOutcomeFromPayload({
      outcome: "stale_run",
      requestedRunId: 7,
      activeRunId: "8",
    }),
    null,
  );
  assert.equal(dataFetchStopOutcomeFromPayload({ outcome: "cancelled" }), null);
  assert.equal(dataFetchStopOutcomeFromPayload({ outcome: "unknown" }), null);
});

test("BLOCKED payload metadata stops a desktop batch and preserves broker retryAfter", () => {
  assert.equal(
    brokerBlockedRetryAfterSeconds({
      code: "BLOCKED_PAYLOAD_TYPE",
      retryAfterSeconds: 7,
    }),
    7,
  );
  assert.equal(
    brokerBlockedRetryAfterSeconds({ code: "BLOCKED_PAYLOAD_TYPE", retryAfterSeconds: null }),
    null,
  );
  assert.equal(brokerBlockedRetryAfterSeconds({ code: "MARKET_CLOSED" }), undefined);
  assert.throws(
    () => brokerBlockedRetryAfterSeconds({
      code: "BLOCKED_PAYLOAD_TYPE",
      retryAfterSeconds: -1,
    }),
    /non-negative integer/i,
  );
});

test("discovery request preserves the opaque inventory identity and uses symbol/base only as assertions", () => {
  const generation = `g1-${"ab".repeat(32)}.vortex`;
  const manifestBindingSha256 = "cd".repeat(32);
  const selected = {
    datasetIdentity: "d1-cTrAdEr-Exact-MiXeD-Identity" as CanonicalDatasetIdentity,
    generation,
    manifestBindingSha256,
    symbol: "eurUsd.raw",
    timeframe: "m5",
    verification: "manifest_only" as const,
  };

  const body = discoveryStartBody(selected, {
    population: 512,
    higher_tfs: ["M15", "H1"],
  });

  assert.equal(
    JSON.stringify(body),
    `{"dataset_selection":{"schema":"neoethos.selected-dataset-generation.v1","version":1,"dataset_identity":"d1-cTrAdEr-Exact-MiXeD-Identity","generation_id":"${generation}","manifest_binding_sha256":"${manifestBindingSha256}"},"symbol":"eurUsd.raw","base_tf":"m5","population":512,"higher_tfs":["M15","H1"]}`,
  );
});

test("discovery knobs cannot override the authoritative identity assertions at runtime", () => {
  const selected = {
    datasetIdentity: "d1-authoritative" as CanonicalDatasetIdentity,
    generation: `g1-${"11".repeat(32)}.vortex`,
    manifestBindingSha256: "22".repeat(32),
    symbol: "EURUSD",
    timeframe: "M5",
    verification: "manifest_only" as const,
  };
  const hostile = {
    dataset_identity: "d1-forged",
    symbol: "GBPUSD",
    base_tf: "H1",
    generations: 25,
  } as unknown as DiscoveryKnobs;

  const body = discoveryStartBody(selected, hostile);

  assert.equal(body.dataset_selection.dataset_identity, selected.datasetIdentity);
  assert.equal(body.dataset_selection.generation_id, selected.generation);
  assert.equal(
    body.dataset_selection.manifest_binding_sha256,
    selected.manifestBindingSha256,
  );
  assert.equal(body.symbol, selected.symbol);
  assert.equal(body.base_tf, selected.timeframe);
  assert.equal(body.generations, 25);
});

test("discovery request rejects a missing or malformed generation receipt", () => {
  const base = {
    datasetIdentity: "d1-authoritative" as CanonicalDatasetIdentity,
    generation: `g1-${"33".repeat(32)}.vortex`,
    manifestBindingSha256: "44".repeat(32),
    symbol: "EURUSD",
    timeframe: "M5",
    verification: "manifest_only" as const,
  };

  assert.throws(
    () => discoveryStartBody({ ...base, generation: "" }, {}),
    /generation receipt/i,
  );
  assert.throws(
    () => discoveryStartBody({ ...base, manifestBindingSha256: "not-a-sha256" }, {}),
    /manifest binding/i,
  );
});

test("discovery request cannot be constructed without an authoritative inventory selection", () => {
  assert.throws(
    () => discoveryStartBody(undefined, {}),
    /Select an exact canonical dataset generation from Data/,
  );
});

test("discovery request fails closed when inventory omits assertion metadata", () => {
  const selected = {
    datasetIdentity: "d1-authoritative" as CanonicalDatasetIdentity,
    generation: `g1-${"55".repeat(32)}.vortex`,
    manifestBindingSha256: "66".repeat(32),
    symbol: null,
    timeframe: "M5",
    verification: "manifest_only" as const,
  };

  assert.throws(
    () => discoveryStartBody(selected, {}),
    /lacks authoritative symbol\/timeframe assertions/,
  );
});

test("data operation errors keep the full typed conflict response visible", () => {
  const conflict =
    '409 Conflict — {"error":"generation conflict","detail":"generation conflict: expected g1-old.vortex, current is g1-new.vortex"}';

  assert.equal(dataOperationErrorText(new Error(conflict)), conflict);
  assert.equal(dataOperationErrorText(conflict), conflict);
});

test("a failed data batch cannot be formatted as a successful download", () => {
  const conflict =
    'EURUSD M5: 409 Conflict — {"error":"generation conflict","detail":"current is g1-new.vortex"}';

  const text = dataBatchResultText(1, [], [conflict]);
  assert.match(text, /^Download failed 0\/1/);
  assert.doesNotMatch(text, /^✓/);
  assert.match(text, /generation conflict/);
  assert.match(text, /g1-new\.vortex/);
});

test("a typed coverage failure preserves the complete loader diagnostic", () => {
  const coverage = {
    status: "failed" as const,
    symbol: "EURUSD",
    error: {
      kind: "load_failed" as const,
      detail: "canonical dataset rejected: ambiguous identity: found 2 verified sources",
    },
  };

  assert.equal(
    symbolCoverageFailureText(coverage),
    "load_failed: canonical dataset rejected: ambiguous identity: found 2 verified sources",
  );
});

test("verified empty coverage is not rendered as a loader failure", () => {
  const coverage = {
    status: "verified" as const,
    symbol: "EURUSD",
    bars: 0,
    firstMs: 0,
    lastMs: 0,
    years: 0,
  };

  assert.equal(symbolCoverageFailureText(coverage), null);
});

test("promoteStrategyBody serializes the camelCase server fixture byte-for-byte", () => {
  const body = promoteStrategyBody("EURUSD", "M5");

  assert.equal(JSON.stringify(body), '{"symbol":"EURUSD","baseTf":"M5"}');
});
