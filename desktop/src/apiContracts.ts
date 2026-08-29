export type AmendProtectionBody = Readonly<{
  positionId: number;
  stopLossPrice: number | null;
  takeProfitPrice: number | null;
  trailingStopLoss: boolean | null;
}>;

export function amendProtectionBody(
  positionId: number,
  stopLossPrice?: number | null,
  takeProfitPrice?: number | null,
  trailingStopLoss?: boolean,
): AmendProtectionBody {
  return {
    positionId,
    stopLossPrice: stopLossPrice ?? null,
    takeProfitPrice: takeProfitPrice ?? null,
    trailingStopLoss: trailingStopLoss ?? null,
  };
}

export type DataImportSourceFormat =
  | "csv"
  | "tsv"
  | "json-array"
  | "json-lines"
  | "parquet"
  | "arrow-ipc-file"
  | "arrow-ipc-stream"
  | "vortex";

declare const canonicalDatasetIdentityBrand: unique symbol;

/**
 * Opaque canonical identity returned by the backend inventory/publication
 * receipts. The desktop must never build, normalize, or decode this value.
 */
export type CanonicalDatasetIdentity = string & {
  readonly [canonicalDatasetIdentityBrand]: "CanonicalDatasetIdentity";
};

export type DatasetInventoryEntry = Readonly<{
  datasetIdentity: CanonicalDatasetIdentity;
  generation: string;
  manifestBindingSha256: string;
  /** Authoritative backend classification; the desktop never decodes identity bytes. */
  sourceKind: "ctrader" | "external";
  symbol: string | null;
  timeframe: string | null;
  verification: "manifest_only" | "generation_verified";
}>;

export type DatasetInventorySkipped = Readonly<{
  path: string;
  category: string;
  detail: string;
}>;

export type DiscoveryKnobs = Readonly<{
  higher_tfs?: string[];
  population?: number;
  generations?: number;
  max_indicators?: number;
  target_candidates?: number;
  portfolio_size?: number;
}>;

export type SelectedDatasetGenerationV1 = Readonly<{
  schema: "neoethos.selected-dataset-generation.v1";
  version: 1;
  dataset_identity: CanonicalDatasetIdentity;
  generation_id: string;
  manifest_binding_sha256: string;
}>;

export type DiscoveryStartBody = DiscoveryKnobs &
  Readonly<{
    dataset_selection: SelectedDatasetGenerationV1;
    /** Compatibility assertions only; never selectors. */
    symbol: string;
    /** Compatibility assertion only; never a selector. */
    base_tf: string;
  }>;

function selectedDatasetGenerationReceipt(
  selected: DatasetInventoryEntry,
): SelectedDatasetGenerationV1 {
  if (!/^g1-[0-9a-f]{64}\.vortex$/i.test(selected.generation)) {
    throw new Error(
      `Inventory entry ${selected.datasetIdentity} has an invalid generation receipt.`,
    );
  }
  if (!/^[0-9a-f]{64}$/i.test(selected.manifestBindingSha256)) {
    throw new Error(
      `Inventory entry ${selected.datasetIdentity} has an invalid manifest binding SHA-256.`,
    );
  }
  return {
    schema: "neoethos.selected-dataset-generation.v1",
    version: 1,
    dataset_identity: selected.datasetIdentity,
    generation_id: selected.generation,
    manifest_binding_sha256: selected.manifestBindingSha256,
  };
}

export function discoveryStartBody(
  selected: DatasetInventoryEntry | undefined,
  knobs: DiscoveryKnobs,
): DiscoveryStartBody {
  if (!selected) {
    throw new Error("Select an exact canonical dataset generation from Data before Discovery.");
  }
  if (
    selected.symbol === null ||
    selected.symbol.length === 0 ||
    selected.timeframe === null ||
    selected.timeframe.length === 0
  ) {
    throw new Error(
      `Inventory entry ${selected.datasetIdentity} lacks authoritative symbol/timeframe assertions.`,
    );
  }
  return {
    dataset_selection: selectedDatasetGenerationReceipt(selected),
    symbol: selected.symbol,
    base_tf: selected.timeframe,
    ...(knobs.population === undefined ? {} : { population: knobs.population }),
    ...(knobs.generations === undefined ? {} : { generations: knobs.generations }),
    ...(knobs.max_indicators === undefined
      ? {}
      : { max_indicators: knobs.max_indicators }),
    ...(knobs.target_candidates === undefined
      ? {}
      : { target_candidates: knobs.target_candidates }),
    ...(knobs.portfolio_size === undefined
      ? {}
      : { portfolio_size: knobs.portfolio_size }),
    ...(knobs.higher_tfs === undefined ? {} : { higher_tfs: [...knobs.higher_tfs] }),
  };
}

export type DataImportBody = Readonly<{
  sourcePath: string;
  sourceFormat: DataImportSourceFormat;
  sourceNamespace: string;
  symbol: string;
  timeframe: string;
  barTimestampConvention: "bar_open" | "bar_close" | "bar_end" | "unknown";
  expectedGeneration: string | null;
}>;

export function dataImportBody(
  sourcePath: string,
  sourceFormat: DataImportSourceFormat,
  sourceNamespace: string,
  symbol: string,
  timeframe: string,
  barTimestampConvention: DataImportBody["barTimestampConvention"],
  expectedGeneration: string | null,
): DataImportBody {
  return {
    sourcePath,
    sourceFormat,
    sourceNamespace,
    symbol,
    timeframe,
    barTimestampConvention,
    expectedGeneration,
  };
}

export type DataFetchBody = Readonly<{
  symbol: string;
  timeframe: string;
  fromMs: number;
  toMs: number | undefined;
  datasetSelection: SelectedDatasetGenerationV1 | null;
}>;

export function dataFetchBody(
  symbol: string,
  timeframe: string,
  fromMs: number,
  toMs?: number,
  datasetSelection: SelectedDatasetGenerationV1 | null = null,
): DataFetchBody {
  return { symbol, timeframe, fromMs, toMs, datasetSelection };
}

/** Stable UI key for choosing one opaque cTrader identity for a symbol/TF. */
export function brokerFetchSelectionKey(symbol: string, timeframe: string): string {
  return JSON.stringify([
    symbol.trim().toUpperCase(),
    timeframe.trim().toUpperCase(),
  ]);
}

/**
 * Select the one exact cTrader generation from the authoritative bootstrap.
 * External imports are never a broker-refresh CAS base. Multiple broker
 * identities (for example two accounts) are deliberately ambiguous and must
 * be selected explicitly rather than collapsed to symbol/timeframe text.
 */
export function selectedDatasetGenerationForBrokerFetch(
  inventory: readonly DatasetInventoryEntry[],
  symbol: string,
  timeframe: string,
  selectedDatasetIdentity: CanonicalDatasetIdentity | null = null,
): SelectedDatasetGenerationV1 | null {
  const normalizedSymbol = symbol.trim().toUpperCase();
  const normalizedTimeframe = timeframe.trim().toUpperCase();
  const matches = inventory.filter((entry) =>
    entry.sourceKind === "ctrader" &&
    entry.symbol?.trim().toUpperCase() === normalizedSymbol &&
    entry.timeframe?.trim().toUpperCase() === normalizedTimeframe
  );
  if (selectedDatasetIdentity !== null) {
    const selected = matches.find(
      (entry) => entry.datasetIdentity === selectedDatasetIdentity,
    );
    if (!selected) {
      throw new Error(
        `Selected cTrader dataset identity ${selectedDatasetIdentity} does not match ${normalizedSymbol} ${normalizedTimeframe}.`,
      );
    }
    return selectedDatasetGenerationReceipt(selected);
  }
  if (matches.length > 1) {
    throw new Error(
      `Multiple exact cTrader dataset identities match ${normalizedSymbol} ${normalizedTimeframe}; select one account identity explicitly.`,
    );
  }
  return matches.length === 0 ? null : selectedDatasetGenerationReceipt(matches[0]);
}

export type DatasetGenerationAcknowledgement = Readonly<{
  datasetIdentity: CanonicalDatasetIdentity;
  generation: string;
}>;

export type DataFetchOutcome = DatasetGenerationAcknowledgement &
  Readonly<{
    symbol: string;
    timeframe: string;
    barCount: number;
    hasMore: boolean;
    writtenPath: string;
    oldestMs: number | null;
    durableCommitId: string;
  }>;

export type DataFetchStatus =
  | Readonly<{ active: false; runId: null; phase: null }>
  | Readonly<{
      active: true;
      runId: number;
      phase: "capturing" | "cancellation_requested" | "publication_in_progress";
    }>;

export type DataFetchStopOutcome =
  | Readonly<{ outcome: "cancelled"; runId: number }>
  | Readonly<{ outcome: "publication_in_progress"; runId: number }>
  | Readonly<{
      outcome: "stale_run";
      requestedRunId: number;
      activeRunId: number;
    }>
  | Readonly<{ outcome: "no_active_fetch" }>;

function isRunId(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

/** Parse only the exact typed stop outcomes accepted by the desktop. */
export function dataFetchStopOutcomeFromPayload(
  payload: unknown,
): DataFetchStopOutcome | null {
  if (typeof payload !== "object" || payload === null) return null;
  const record = payload as Record<string, unknown>;
  switch (record.outcome) {
    case "cancelled":
    case "publication_in_progress":
      return isRunId(record.runId) ? payload as DataFetchStopOutcome : null;
    case "stale_run":
      return isRunId(record.requestedRunId) && isRunId(record.activeRunId)
        ? payload as DataFetchStopOutcome
        : null;
    case "no_active_fetch":
      return payload as DataFetchStopOutcome;
    default:
      return null;
  }
}

/**
 * Stop the run the operator can currently see. If status polling raced ahead
 * of the click, follow the backend's exact activeRunId once. A second stale
 * response is returned as-is, so this can never chase an unbounded stream of
 * newly-started runs.
 */
export async function stopDataFetchFollowingActiveRun(
  requestedRunId: number,
  requestStop: (runId: number) => Promise<DataFetchStopOutcome>,
): Promise<DataFetchStopOutcome> {
  const first = await requestStop(requestedRunId);
  return first.outcome === "stale_run"
    ? requestStop(first.activeRunId)
    : first;
}

export type DataImportOutcome = DatasetGenerationAcknowledgement &
  Readonly<{
    symbol: string;
    timeframe: string;
    sourceFormat: DataImportSourceFormat;
    writtenPath: string;
    rowCount: number;
    durableCommitId: string;
    sourceSha256: string;
  }>;

export type DatasetGenerationReceipts = Readonly<
  Record<string, DatasetGenerationAcknowledgement>
>;

export function dataImportIdentityKey(
  sourceNamespace: string,
  symbol: string,
  timeframe: string,
  barTimestampConvention: DataImportBody["barTimestampConvention"],
): string {
  return JSON.stringify([
    "external",
    sourceNamespace,
    symbol,
    timeframe,
    barTimestampConvention,
  ]);
}

export function recordDatasetGeneration(
  receipts: DatasetGenerationReceipts,
  requestIdentity: string,
  acknowledgement: DatasetGenerationAcknowledgement,
): DatasetGenerationReceipts {
  return {
    ...receipts,
    [requestIdentity]: {
      datasetIdentity: acknowledgement.datasetIdentity,
      generation: acknowledgement.generation,
    },
  };
}

export function expectedGenerationFor(
  receipts: DatasetGenerationReceipts,
  requestIdentity: string,
): string | null {
  return receipts[requestIdentity]?.generation ?? null;
}

export function dataOperationErrorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Return the broker-mandated wait for a typed BLOCKED response. */
export function brokerBlockedRetryAfterSeconds(payload: unknown): number | null | undefined {
  if (typeof payload !== "object" || payload === null) return undefined;
  const record = payload as Record<string, unknown>;
  if (record.code !== "BLOCKED_PAYLOAD_TYPE") return undefined;
  const retryAfter = record.retryAfterSeconds;
  if (retryAfter === null || retryAfter === undefined) return null;
  if (!Number.isSafeInteger(retryAfter) || (retryAfter as number) < 0) {
    throw new Error("Broker retryAfterSeconds must be a non-negative integer or null.");
  }
  return retryAfter as number;
}

export function dataBatchResultText(
  total: number,
  acknowledgements: readonly string[],
  failures: readonly string[],
): string {
  const completed = acknowledgements.length;
  const headline = failures.length === 0
    ? `✓ Downloaded ${completed}/${total}`
    : completed === 0
      ? `Download failed ${completed}/${total}`
      : `⚠ Downloaded ${completed}/${total} with ${failures.length} failed`;
  const acknowledged = acknowledgements.length
    ? ` · ${acknowledgements.join(" · ")}`
    : "";
  const failureDetails = failures.length ? ` · ${failures.join(" | ")}` : "";
  return `${headline}${acknowledged}${failureDetails}.`;
}

export type SymbolCoverageFailure = Readonly<{
  status: "failed";
  symbol: string;
  error: Readonly<{
    kind: "load_failed" | "command_failed";
    detail: string;
  }>;
}>;

export type SymbolCoverageVerified = Readonly<{
  status: "verified";
  symbol: string;
  bars: number;
  firstMs: number;
  lastMs: number;
  years: number;
}>;

export type SymbolCoverage = SymbolCoverageVerified | SymbolCoverageFailure;

export function symbolCoverageFailureText(coverage: SymbolCoverage): string | null {
  return coverage.status === "failed"
    ? `${coverage.error.kind}: ${coverage.error.detail}`
    : null;
}

export type PromoteStrategyBody = Readonly<{
  symbol: string | undefined;
  baseTf: string | undefined;
}>;

export function promoteStrategyBody(symbol?: string, baseTf?: string): PromoteStrategyBody {
  return {
    symbol: symbol || undefined,
    baseTf: baseTf || undefined,
  };
}
