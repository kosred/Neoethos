// Typed wrappers over every Tauri command. The UI never touches `invoke`
// directly — it calls these, so the command surface is one auditable file.
import { invoke } from "@tauri-apps/api/core";

import {
  amendProtectionBody,
  dataFetchStopOutcomeFromPayload,
  dataImportBody,
  promoteStrategyBody,
  stopDataFetchFollowingActiveRun,
  type DatasetInventoryEntry,
  type DatasetInventorySkipped,
  type DiscoveryStartBody,
  type DataFetchBody,
  type DataFetchOutcome,
  type DataFetchStatus,
  type DataFetchStopOutcome,
  type DataImportBody,
  type DataImportOutcome,
  type DataImportSourceFormat,
  type SupervisorConfig,
} from "./apiContracts";
export { dataFetchBody } from "./apiContracts";
export type { SupervisorConfig } from "./apiContracts";

// ── In-process backend (full neoethos-app axum API over loopback) ─────────────
// The Tauri shell runs the whole backend in-process and tells us the port via
// the `api_base` command. We resolve it once and reuse it for every call.
let _apiBase: string | null = null;
export async function apiBaseUrl(): Promise<string> {
  if (_apiBase) return _apiBase;
  _apiBase = await invoke<string>("api_base");
  return _apiBase;
}

export class ApiResponseError extends Error {
  readonly status: number;
  readonly statusText: string;
  readonly payload: unknown;

  constructor(
    status: number,
    statusText: string,
    payload: unknown,
    detail: string,
  ) {
    super(`${status} ${statusText}${detail ? ` — ${detail}` : ""}`);
    this.name = "ApiResponseError";
    this.status = status;
    this.statusText = statusText;
    this.payload = payload;
  }
}

async function _check(r: Response): Promise<Response> {
  if (!r.ok) {
    const detail = await r.text().catch(() => "");
    let payload: unknown = null;
    if (detail) {
      try {
        payload = JSON.parse(detail);
      } catch {
        payload = null;
      }
    }
    throw new ApiResponseError(r.status, r.statusText, payload, detail);
  }
  return r;
}
export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  const base = await apiBaseUrl();
  const r = await _check(await fetch(`${base}${path}`, { signal }));
  return r.json() as Promise<T>;
}
export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const base = await apiBaseUrl();
  const r = await _check(
    await fetch(`${base}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  );
  // some POSTs return empty bodies
  const txt = await r.text();
  return (txt ? JSON.parse(txt) : null) as T;
}

// ── Server-Sent Events (push) ─────────────────────────────────────────────
// The backend pushes ticks + account snapshots over SSE. We open an
// EventSource against the in-process server; the browser auto-reconnects.
// Returns a disposer that closes the stream.
export async function openSse<T>(
  path: string,
  eventName: string,
  onData: (data: T) => void,
  onStatus?: (connected: boolean) => void,
): Promise<() => void> {
  const base = await apiBaseUrl();
  const es = new EventSource(`${base}${path}`);
  es.addEventListener("open", () => onStatus?.(true));
  es.addEventListener("error", () => onStatus?.(false));
  es.addEventListener(eventName, (e) => {
    try {
      onData(JSON.parse((e as MessageEvent<string>).data) as T);
    } catch {
      /* ignore malformed frame */
    }
  });
  return () => es.close();
}

export type Tick = {
  symbolId: number;
  symbolName: string;
  bid: number | null;
  ask: number | null;
  midPrice: number | null;
  brokerTimestampMs: number | null;
  receivedAtUnixMs: number;
  freshnessSeconds: number;
};
export type StreamPosition = {
  positionId: number;
  volumeUnits: number; // wire volume (pass to closePosition)
  symbol: string; // human name, e.g. "EURUSD"
  side: string; // BUY / SELL
  volume: number;
  openTimestampMs: number | null;
  pnlPips: number | null;
  pnlUsd: number; // live P/L in the ACCOUNT currency (field name is legacy)
  entryPrice: number | null; // all server-provided — client does NO conversion
  stopLoss: number | null;
  takeProfit: number | null;
  volumeLots: number | null; // cTrader-parity lots (1.17), not raw units
};
export type AccountSnapshotValues = {
  balance: number;
  equity: number;
  freeMargin: number;
  usedMargin: number;
  currency: string;
  fetchedAtUnixMs: number;
  positions: StreamPosition[];
};
/** Response-derived identity, not the currently selected account stamped onto cached values. */
export type AccountStreamSnap = AccountSnapshotValues & {
  sourceAccountId: string;
  sourceEnvironment: "Demo" | "Live";
};
export const streamSpots = (onTick: (t: Tick) => void, onStatus?: (c: boolean) => void) =>
  openSse("/live/spots/stream", "tick", onTick, onStatus);
export const streamAccount = (onSnap: (s: AccountStreamSnap) => void, onStatus?: (c: boolean) => void) =>
  openSse("/account/snapshot/stream", "account", onSnap, onStatus);
export const refreshAccount = () => apiPost("/account/snapshot/refresh");

// ── Types (mirror the Rust serde DTOs, camelCase) ─────────────────────────────
export type AppInfo = { version: string; data_root: string; data_root_exists: boolean };
export type Candle = { time: number; open: number; high: number; low: number; close: number };

export type BrokerStatus = {
  configured: boolean;
  hasToken: boolean;
  environment: string;
  accountId: string | null;
};

export type AccountInfo = {
  accountId: string;
  brokerTitle: string;
  accountName: string;
  isLive: boolean | null;
  login: number | null;
  enabled: boolean;
  label: string; // e.g. "DEMO · Spotware · login 5789955"
};
export type ExecResult = {
  status: string;
  orderId: number | null;
  positionId: number | null;
  dealId: number | null;
  side: string | null;
  fillPrice: number | null;
  message: string;
};
export type ReauthResult = {
  callbackPort: number;
  refreshTokenPresent: boolean;
  accessTokenLen: number;
  message: string;
};

// ── Local vortex data (works offline, no broker) ──────────────────────────────
export const appInfo = () => invoke<AppInfo>("app_info");
/** Native OS file picker for data import; returns the chosen path or null. */
export const pickDataFile = () => invoke<string | null>("pick_data_file");
/** Per-symbol local-history coverage (years + bars) for the given base TF. */

// ── Live cTrader (in-process, auto-auth) ──────────────────────────────────────
export const brokerStatus = () => invoke<BrokerStatus>("broker_status");
export const brokerChart = (symbol: string, timeframe: string, limit = 1000) =>
  invoke<Candle[]>("broker_chart", { symbol, timeframe, limit });
export const brokerAccounts = () => invoke<AccountInfo[]>("broker_accounts");
// Full broker symbol universe (dozens — forex/metals/indices) WITH asset class,
// straight from the server. Use this for selection/watchlist, not the 7 defaults.
export type BrokerSymbol = {
  symbolId: number;
  symbolName: string;
  enabled: boolean;
  description: string | null;
  assetClass: string | null;
};
export const serverSymbols = () =>
  apiGet<{ symbolCount: number; symbols: BrokerSymbol[] }>("/broker/symbols");
export const selectAccount = (accountId: string, live: boolean, label?: string) =>
  invoke<BrokerStatus>("select_account", { accountId, live, label: label ?? null });
export const placeOrder = (
  symbol: string,
  side: "buy" | "sell",
  volumeLots: number,
  stopLossPips?: number,
  takeProfitPips?: number,
) =>
  invoke<ExecResult>("place_order", {
    symbol,
    side,
    volumeLots,
    stopLossPips: stopLossPips ?? null,
    takeProfitPips: takeProfitPips ?? null,
  });
export const closePosition = (positionId: number, volume: number) =>
  invoke<ExecResult>("close_position", { positionId, volume });
export const reauthBroker = () => invoke<ReauthResult>("reauth_broker");

// ── cTrader API credentials (audit #119) ──────────────────────────────────
// Until 2026-08-09 the Dashboard told the operator to "go to Settings and add
// cTrader credentials" and NO such form existed anywhere in `desktop/src`.
// Credentials are baked into the binary by `neoethos-app/build.rs`, so a
// revoked client_id locked him out of his own broker until someone rebuilt it.
// The backend has always had the endpoints; only the screen was missing.
//
// SECRET HANDLING: the GET never returns the secret — only
// `clientSecretMask` ("****abcd (length 32)") and `clientSecretConfigured`.
// The POST treats an empty `clientSecret` as "keep the saved one". Never log,
// echo or persist the typed value anywhere in the UI.
export type BrokerCredentials = {
  clientId: string;
  clientSecretMask: string;
  clientSecretConfigured: boolean;
  redirectUri: string;
  environment: string;
  accountId: string;
};
export const brokerCredentials = () => apiGet<BrokerCredentials>("/broker/credentials");
export const saveBrokerCredentials = (b: {
  clientId: string;
  clientSecret: string;
  redirectUri?: string;
  environment?: string;
  accountId?: string;
}) =>
  apiPost<{ ok: boolean; message: string }>("/broker/credentials", {
    clientId: b.clientId,
    clientSecret: b.clientSecret,
    redirectUri: b.redirectUri ?? "",
    environment: b.environment ?? "Demo",
    accountId: b.accountId ?? "",
  });

// ══════════════════════════════════════════════════════════════════════════
// Full backend API (in-process axum server) — every old Flutter feature.
// ══════════════════════════════════════════════════════════════════════════

// ── Engines: Discovery + Training ─────────────────────────────────────────
export type EngineRunState = "Idle" | "Running" | "Succeeded" | "Degraded" | "Failed" | "Cancelled";
export type EngineCounter = { name: string; value: number };
export type CanonicalNativeResearchPublished = {
  relativePath: string;
  byteCount: number;
  fileSha256: string;
  evidenceIdentitySha256: string;
  configuredPopulation: number;
  resolvedPopulation: number;
  populationCap: number;
  hardGrowthCap: number;
  termCap: number;
  selectedDeviceOrdinal: number;
  engine: string;
  parentH2dBytes: number;
  adaptiveH2dBytes: number;
  metricRows: number;
  metricBytes: number;
  consumerCompletionConfirmed: boolean;
  replayIdentitySealed: boolean;
};
export type CanonicalNativeResearchStatus = {
  available: boolean;
  availabilityDetail: string;
  state: string;
  stage: string;
  percent: number;
  leaseToken: string | null;
  cancellationRequested: boolean;
  failureStage: string | null;
  failureCode: string | null;
  failureDetail: string | null;
  published: CanonicalNativeResearchPublished | null;
};
export type EnginesStatus = {
  discovery: EngineRunState;
  discoveryStartAvailable: boolean;
  discoveryStartUnavailableReason: string | null;
  discoveryStartMode: "ResearchOnly";
  historicalEvaluationAvailable: boolean;
  historicalEvaluationUnavailableReason: string | null;
  training: EngineRunState;
  canonicalNativeResearch: CanonicalNativeResearchStatus;
  autoTrader?: string;
  auto_trader?: string;
  discoverySummary?: string;
  discovery_summary?: string;
  trainingSummary?: string;
  training_summary?: string;
  discoveryStage?: string;
  discovery_stage?: string;
  discoveryPercent?: number | null;
  discovery_percent?: number | null;
  discoveryCounters?: EngineCounter[];
  discovery_counters?: EngineCounter[];
  ramTotalGb?: number;
  ramAvailableGb?: number;
  featureStoreMb?: number;
};
export const enginesStatus = () => apiGet<EnginesStatus>("/engines/status");
export const discoveryStart = (b: DiscoveryStartBody) => apiPost("/engines/discovery/start", b);
export const discoveryStop = () => apiPost<{ running: boolean; kind: string }>("/engines/discovery/stop");
export const trainingStop = () => apiPost<{ running: boolean; kind: string }>("/engines/training/stop");
export type ResearchEvaluationMode = "train_models" | "strategy_only";
export const trainingStart = (identity: string, mode?: ResearchEvaluationMode) =>
  apiPost<{ started: boolean; symbol: string; base_tf: string }>("/engines/training/start", {
    training_handoff: identity,
    ...(mode === undefined ? {} : { mode }),
  });

// ── Strategy Lab ──────────────────────────────────────────────────────────
export type PromotionMetricSet = {
  sharpe: number;
  winRate: number;
  profitFactor: number;
  maxDrawdownPct: number;
  trades: number;
};
export type PromotionCriterion = {
  name: string;
  passed: boolean;
  actual: number;
  threshold: number;
  comparison: string;
};
export type PromotionStatus = {
  symbol: string;
  baseTf: string;
  portfolioSize: number;
  aggregate: PromotionMetricSet | null;
  decision: {
    promoted: boolean;
    criteria: PromotionCriterion[];
    summary: string;
  };
  config: Record<string, boolean | number>;
};
export type PromoteResult = {
  promoted: boolean;
  symbol: string;
  baseTf: string;
  liveModelsPath: string | null;
  filesCopied: number;
  message: string;
};
const qs = (o: Record<string, string | undefined>) => {
  const p = Object.entries(o).filter(([, v]) => v && v.trim() !== "");
  return p.length ? "?" + p.map(([k, v]) => `${k}=${encodeURIComponent(v!)}`).join("&") : "";
};
export const promotionStatus = (symbol?: string, baseTf?: string) =>
  apiGet<PromotionStatus>("/strategy_lab/promotion" + qs({ symbol, base_tf: baseTf }));
export const promoteStrategy = (symbol?: string, baseTf?: string) =>
  apiPost<PromoteResult>("/strategy_lab/promote", promoteStrategyBody(symbol, baseTf));

// ── Autonomous trader ─────────────────────────────────────────────────────
export type LiveEngineStatus = {
  running: boolean;
  portfolioPath: string | null;
  symbol: string | null;
  baseTf: string | null;
  genes: number;
  barsEvaluated: number;
  lastSignal: string | null;
  openPositionId: number | null;
  consecutiveLosses: number;
  windowWinRatePct: number | null;
  windowTrades: number;
  retired: boolean;
  protectionPolicyIdentity: string | null;
  trailingEnabled: boolean | null;
  protectionState: string | null;
  positionEntryPrice: number | null;
  initialStopPips: number | null;
  favorableExtremePrice: number | null;
  favorableMoveR: number | null;
  confirmedStopPrice: number | null;
  lastProtectionError: string | null;
  lastExitReason: string | null;
};
export type AutonomousStatus = {
  running: boolean;
  engineCount: number;
  engines: LiveEngineStatus[];
};
export type AutonomousStartResult = {
  started: string[];
  skipped: string[];
  blacklisted: string[];
  failed: Array<{ portfolio: string; error: string }>;
  overview: AutonomousStatus;
};
export type ReplayStats = {
  bars_processed: number;
  signals_evaluated: number;
  intents_emitted: number;
  intents_executed: number;
  intents_blocked: number;
  positions_opened: number;
  positions_closed: number;
  open_positions: number;
  realized_pnl: number;
  equity: number;
  commission_paid: number;
  max_hold_exits: number;
  fidelity_warnings: string[];
};
export const autonomousStatus = () => apiGet<AutonomousStatus>("/autonomous/status");
export const autonomousStart = (b?: unknown) => apiPost<AutonomousStartResult>("/autonomous/start", b ?? {});
export const autonomousStop = () => apiPost<{ stopped: boolean }>("/autonomous/stop");
export const autonomousReplay = (b?: unknown) => apiPost<ReplayStats>("/autonomous/replay", b ?? {});

export type GateCriterion = { name: string; passed: boolean; actual: number; threshold: number; comparison: string };
export type GateVerdict = {
  envIsLive: boolean;
  enforced: boolean;
  eligible: boolean;
  summary: string;
  criteria: GateCriterion[];
};
export const autonomousGate = (portfolioPath: string) =>
  apiGet<GateVerdict>(`/autonomous/gate?portfolio=${encodeURIComponent(portfolioPath)}`);

// ── Risk ──────────────────────────────────────────────────────────────────
export type PresetSummary = {
  id: string;
  displayName: string;
  maxDailyLossPct: number;
  maxOverallDrawdownPct: number;
  challengeProfitTargetPct: number;
  minTradingDays: number;
};
export type RiskInfo = {
  riskPerTrade: number;
  minRiskPerTrade: number;
  maxRiskPerTrade: number;
  dailyDrawdownLimit: number;
  totalDrawdownLimit: number;
  maxLotSize: number;
  requireStopLoss: boolean;
  preset: string;
  presetDisplayName: string;
  availablePresets: PresetSummary[];
  // `propFirmRulesEnabled` was HERE and is deleted. It mirrored
  // `risk.prop_firm_rules`, a config field with exactly one write
  // (`server/risk.rs:166`, from the preset dropdown), one read
  // (`:244`, into this DTO) and ZERO decisions anywhere in the engine —
  // every discovery call passes a hardcoded `PropFirmRiskRules::default()`.
  // Risk.tsx and RiskyMode.tsx rendered it as "currently active rules", so the
  // app could announce "Prop-firm" while `system.trading_mode` was `risky`.
  // Both screens now derive that display from `system.trading_mode`.
  //
  // UPDATE 2026-08-10 — D6 is DONE. The config field
  // `RiskConfig::prop_firm_rules` is DELETED from `neoethos-core`, and
  // `risk.prop_firm_rules` is now a `load_seal::RETIRED_KEYS` entry, so a
  // config file that still carries the key loads with the key NAMED at WARN
  // instead of being refused.
  //
  // The DTO still carries `propFirmRulesEnabled`
  // (`crates/neoethos-app/src/server/risk.rs`) — but it is no longer a mirror
  // of anything: it is DERIVED from `system.trading_mode` by
  // `derive_prop_firm_rules_active`, the same switch the engine reads. It is
  // omitted from this type because both screens compute the display from
  // `system.trading_mode` directly; there is no longer a second opinion to
  // disagree with.
  riskyModeCooldownRemainingSecs: number | null;
};
export const riskInfo = () => apiGet<RiskInfo>("/risk");
export const setRiskPreset = (preset: string) => apiPost("/risk/preset", { preset });
// ── Hardware ──────────────────────────────────────────────────────────────
export type HardwareInfo = {
  cpu: { model: string; coresLogical: number; coresPhysical: number; loadAvg: number };
  ram: { totalMb: number; usedMb: number; availableMb: number };
  gpu: { name: string; available: boolean; kind: string };
  /** Whether THIS BUILD has a GPU lane compiled in (`gpu.available` is about
   *  the machine; this is about the binary). Both must be true for GPU work. */
  gpuSupport: { compiled: boolean; backend: string; detail: string };
};
export const hardwareInfo = () => apiGet<HardwareInfo>("/hardware");

// ── Intelligence ──────────────────────────────────────────────────────────
export type DiscoveryTarget = {
  symbol: string;
  baseTf: string;
  strategyId: string;
  sharpe: number | null;
  winRate: number | null;
};
export type IntelligenceInfo = {
  modelsDir: string;
  modelsDirExists: boolean;
  artifactCount: number;
  artifacts: string[];
  lastTouchedUnixMs: number | null;
  discoveryTargets: DiscoveryTarget[];
  trainingHandoffs: TrainingHandoff[];
  trainingHandoffUnavailable: { identity: string; reason: string }[];
};

export type TrainingHandoff = {
  identity: string;
  symbol: string;
  baseTf: string;
  datasetIdentity: string;
  generation: string;
  strategyCount: number;
  plannedModels: string[];
  oosCutoffMs: number;
  purgeBars: number;
};
export const intelligence = () => apiGet<IntelligenceInfo>("/intelligence");

export type SavedResearchAccount = {
  netProfit: number | null;
  sharpe: number | null;
  winRate: number | null;
  profitFactor: number | null;
  expectancy: number | null;
  tradeCount: number;
  maxDrawdownFraction: number | null;
  endingRealizedBalance: number;
  terminalOpen: boolean;
  grossUnrealizedAccount: number | null;
  pendingRoundTripCommissionAccount: number | null;
  belowMinEntries: number;
};
export type SavedResearchReport = {
  evaluationMode: ResearchEvaluationMode;
  reportId: string;
  reportSha256: string;
  rawFinalScopeSha256: string;
  lockedFinalInputsSha256: string;
  firstLockedFinalInputsSha256: string | null;
  holdoutUse: string;
  historicalExposure: string;
  symbol: string;
  baseTimeframe: string;
  accountCurrency: string;
  rowStart: number;
  rowEnd: number;
  rows: number;
  timestampStartMs: number;
  timestampEndMs: number;
  trainingCutoffMs: number;
  blendMode: string | null;
  blendGateFloor: number | null;
  blendVetoBelow: number | null;
  modelHistoryRows: number | null;
  invalidModelSignalRows: number | null;
  configuredLiveMlGate: boolean | null;
  promotionEligible: boolean;
  geneOnly: SavedResearchAccount;
  combined: SavedResearchAccount | null;
};
export type SavedTrainingResearch = {
  trainingHandoff: string;
  status: "candidate_not_ready" | "no_completed_result" | "completed_results";
  reports: SavedResearchReport[];
  unavailable: { reportId: string; reason: string }[];
};
/** Read existing reports only. This endpoint never starts training or evaluation. */
export const savedTrainingResearch = (identity: string, signal?: AbortSignal) => {
  if (!/^[0-9a-f]{64}$/.test(identity)) {
    return Promise.reject(new Error("Select an exact saved training handoff (64 lowercase hexadecimal characters)."));
  }
  return apiGet<SavedTrainingResearch>(`/intelligence/research?training_handoff=${identity}`, signal);
};

// ── Journal ───────────────────────────────────────────────────────────────
export type JournalStats = {
  totalTrades: number;
  wins: number;
  losses: number;
  breakeven: number;
  winRatePct: number;
  netProfit: number;
  grossProfit: number;
  grossLoss: number;
  profitFactor: number | null;
  avgWin: number;
  avgLoss: number;
  payoffRatio: number | null;
  expectancy: number;
  largestWin: number;
  largestLoss: number;
  maxConsecutiveWins: number;
  maxConsecutiveLosses: number;
  maxDrawdownAbs: number;
  maxDrawdownPct: number;
  recoveryFactor: number | null;
  sharpe: number | null;
};
export type ClosedTrade = {
  schemaVersion: number;
  recordedAtUnixMs: number;
  positionId: number;
  symbol: string;
  side: string;
  lots: number;
  accountId: string | null;
  environment: string | null;
  entryTsMs: number | null;
  entryPrice: number | null;
  exitTsMs: number | null;
  exitPrice: number | null;
  grossProfit: number;
  commission: number;
  swap: number;
  netProfit: number;
  balanceAfter: number | null;
};
export const journalStats = () => apiGet<JournalStats>("/journal/stats");
export const journalTrades = () => apiGet<ClosedTrade[]>("/journal/trades");

// Per-trade pips / R / MFE-MAE + the breakdowns that locate a problem.
// Audit #124: this endpoint has existed since 2026-07-30 and until now its
// ONLY caller repo-wide was `mcp/ops.rs:864` — an LLM tool call. The operator
// could not reach the one view that would have surfaced the payoff-1.08
// problem (a 1.08 realised payoff against a 2.0 floor) sixteen months earlier.
export type DerivedTrade = {
  positionId: number;
  symbol: string;
  side: string;
  lots: number;
  entryTsMs: number | null;
  exitTsMs: number | null;
  netProfit: number;
  durationHours: number | null;
  pips: number | null;
  rMultiple: number | null;
  mfePips: number | null;
  maePips: number | null;
  captureRatio: number | null;
  entryHourUtc: number | null;
  entryWeekday: string | null;
  riskPerLot: number | null;
  riskBasis: string;
};
export type BucketSummary = {
  bucket: string;
  trades: number;
  wins: number;
  winRatePct: number;
  netProfit: number;
  expectancy: number;
  netPips: number | null;
  pipsTrades: number;
};
export type JournalAnalytics = {
  trades: DerivedTrade[];
  bySymbol: BucketSummary[];
  byHourUtc: BucketSummary[];
  byWeekday: BucketSummary[];
  bySide: BucketSummary[];
  avgMfePips: number | null;
  avgCaptureRatio: number | null;
  inactiveHoursUtc: number[];
  coverage: {
    tradesTotal: number;
    withPips: number;
    withExcursion: number;
    withRMultiple: number;
    withDuration: number;
    missingEntryTime: number;
    missingPriceSeries: number;
    riskPerLotBySymbol: Record<string, number>;
    riskPerLotAllSymbols: number | null;
    symbolsUsingFallbackRisk: string[];
    minLossesForSymbolRisk: number;
  };
};
export const journalAnalytics = () => apiGet<JournalAnalytics>("/journal/analytics");

// ── News ──────────────────────────────────────────────────────────────────
export type NewsItem = {
  title: string;
  link: string;
  source: string;
  publishedMs: number | null;
  blurb: string;
};
export type NewsFeed = {
  items: NewsItem[];
  aiSummary: string;
  aiAvailable: boolean;
  generatedAtMs: number;
  notice: string;
};
export const newsFeed = (force = false) =>
  apiGet<NewsFeed>(`/news/feed${force ? "?force=true" : ""}`);

// ── Data ──────────────────────────────────────────────────────────────────
export type DataBootstrap = {
  dataDir: string;
  dataDirExists: boolean;
  symbols: string[];
  datasetCount: number;
  lastTouchedUnixMs: number | null;
  datasets: DatasetInventoryEntry[];
  skipped: DatasetInventorySkipped[];
};
export const dataBootstrap = () => apiGet<DataBootstrap>("/data/bootstrap");
export const dataFetch = (body: DataFetchBody) =>
  apiPost<DataFetchOutcome>("/data/fetch", body);
export const dataFetchStatus = () => apiGet<DataFetchStatus>("/data/fetch/status");
export const stopDataFetch = (runId: number) =>
  apiPost<DataFetchStopOutcome>("/data/fetch/stop", { runId });

async function requestDataFetchStopOutcome(runId: number): Promise<DataFetchStopOutcome> {
  try {
    return await stopDataFetch(runId);
  } catch (error) {
    const outcome = error instanceof ApiResponseError
      ? dataFetchStopOutcomeFromPayload(error.payload)
      : null;
    if (outcome) return outcome;
    throw error;
  }
}

export const stopActiveDataFetch = (runId: number) =>
  stopDataFetchFollowingActiveRun(runId, requestDataFetchStopOutcome);

// ── Market Watch / watchlist ──────────────────────────────────────────────
export type WatchlistView = { symbols: string[] };
export type WatchlistUpdateResult = {
  saved: number;
  symbols: string[];
  restarted: boolean;
};
export const getWatchlist = () => apiGet<WatchlistView>("/watchlist");
export const setWatchlist = (symbols: string[]) =>
  apiPost<WatchlistUpdateResult>("/watchlist", { symbols });

// ── AI Desk (Codex / ChatGPT subscription) ────────────────────────────────
export type CodexStatus = {
  authenticated: boolean;
  email: string | null;
  loginInProgress: boolean;
  lastError: string | null;
  authPath: string;
};
export type CodexStartResult = { authorizeUrl: string; callbackPort: number };
export const codexStatus = () => apiGet<CodexStatus>("/auth/codex/status");
// `email` (optional) becomes the OAuth `login_hint` so the operator picks
// WHICH ChatGPT account to connect. Empty ⇒ the issuer shows its picker.
export const codexStart = (email?: string) =>
  apiPost<CodexStartResult>("/auth/codex/start", email ? { email } : {});
export const codexLogout = () => apiPost<{ ok: boolean }>("/auth/codex/logout");
export const codexChat = (prompt: string, model?: string) =>
  apiPost<{ model: string; response: string; total_tokens: number }>("/codex/chat", { prompt, model });

// ── MCP sidecar (external tool servers for the Supervisor) ────────────────
export type McpTool = {
  server: string;
  name: string;
  description: string | null;
};
export type McpStatus = {
  reachable: boolean;
  url?: string;
  error?: string;
  note?: string;
  health?: unknown;
  tools?: McpTool[];
};
export type McpConfigSaveResult = { saved: boolean; path: string; note: string };
export const mcpStatus = () => apiGet<McpStatus>("/mcp/status");
export const mcpConfigGet = () => apiGet<{ exists: boolean; path: string; content: string }>("/mcp/config");
export const mcpConfigSave = (content: string) =>
  apiPost<McpConfigSaveResult>("/mcp/config", { content });

// ── Account / broker detail (history, profile, margin, cashflow) ───────────
export const brokerProfile = () => apiGet<{ userId: number }>("/broker/profile");
export const brokerVersion = () => apiGet<{ version: string }>("/broker/version");
export type BrokerHistoricalOrder = {
  orderId: number;
  positionId: number | null;
  symbolId: number;
  side: string;
  orderType: string;
  orderStatus: string;
  volumeRawCentiUnits: number;
  executedVolumeRawCentiUnits: number | null;
  volumeUnits: number;
  executedVolumeUnits: number | null;
  volumeLots: number | null;
  executedVolumeLots: number | null;
  lotSizeRawCentiUnits: number | null;
  executionPrice: number | null;
  limitPrice: number | null;
  stopPrice: number | null;
  stopLoss: number | null;
  takeProfit: number | null;
  timeInForce: string | null;
  label: string | null;
  comment: string | null;
  clientOrderId: string | null;
  closingOrder: boolean;
  isStopOut: boolean;
  trailingStopLoss: boolean;
  openTimestampMs: number | null;
  closeTimestampMs: number | null;
  expirationTimestampMs: number | null;
  utcLastUpdateTimestampMs: number | null;
};
export type BrokerOrderHistory = {
  accountId: number;
  orders: BrokerHistoricalOrder[];
  hasMore: boolean;
  lotSizeObservedAtUnixMs: number | null;
  lotSizeError: string | null;
};
export type BrokerCashFlowEntry = {
  balanceHistoryId: number;
  operationType: string;
  operationTypeCode: number;
  balance: number;
  delta: number;
  equity: number | null;
  changeBalanceTimestampMs: number;
  externalNote: string | null;
  balanceVersion: number | null;
};
export type BrokerCashFlow = { accountId: number; entries: BrokerCashFlowEntry[] };
export type BrokerExpectedMarginEntry = {
  volumeRawCentiUnits: number;
  volumeUnits: number;
  volumeLots: number | null;
  lotSizeRawCentiUnits: number | null;
  buyMargin: number;
  sellMargin: number;
};
export type BrokerExpectedMargin = {
  accountId: number;
  symbolId: number | null;
  entries: BrokerExpectedMarginEntry[];
  lotSizeObservedAtUnixMs: number | null;
  lotSizeError: string | null;
};
export const ordersHistory = () => apiGet<BrokerOrderHistory>("/broker/orders/history");
export const cashFlow = () => apiGet<BrokerCashFlow>("/broker/cashflow");
export const expectedMargin = (symbolId: number, volume: number) =>
  apiGet<BrokerExpectedMargin>(`/broker/margin/expected?symbolId=${symbolId}&volume=${volume}`);

// ── Position protection (move SL/TP, breakeven, trailing) ──────────────────
export const amendProtection = (
  positionId: number,
  stopLossPrice?: number | null,
  takeProfitPrice?: number | null,
  trailingStopLoss?: boolean,
) =>
  apiPost<ExecResult>(
    "/positions/protection",
    amendProtectionBody(positionId, stopLossPrice, takeProfitPrice, trailingStopLoss),
  );

// ── Pending / conditional orders (limit & stop — "trade when price hits X") ──
export type PendingOrder = {
  orderId: number;
  symbol: string;
  side: string;
  orderType: string;
  volume: number;
  volumeLots: number | null;
  triggerPrice: number | null;
  limitPrice: number | null;
  stopPrice: number | null;
  stopLoss: number | null;
  takeProfit: number | null;
  openTimestampMs: number | null;
  comment: string | null;
};
export const brokerPendingOrders = () => apiGet<PendingOrder[]>("/orders/pending");
export const placePendingOrder = (body: {
  symbol: string;
  side: "buy" | "sell";
  orderType: "limit" | "stop";
  volumeLots: number;
  triggerPrice: number;
  stopLossPips?: number | null;
  takeProfitPips?: number | null;
  expiryUnixMs?: number | null;
  comment?: string | null;
}) => apiPost<ExecResult>("/orders/pending", body);
export const cancelOrder = (orderId: number) => apiPost<ExecResult>("/orders/cancel", { orderId });
/**
 * Modify a RESTING limit/stop order in place (audit #236).
 *
 * Every optional field means LEAVE UNCHANGED. `symbol` and `orderType` are
 * required: the server converts lots with the symbol's lot size and pips with
 * its pip size, and the trigger price goes to limitPrice or stopPrice according
 * to the order's own type.
 */
export const amendOrder = (body: {
  orderId: number;
  symbol: string;
  orderType: "limit" | "stop";
  volumeLots?: number | null;
  triggerPrice?: number | null;
  stopLossPips?: number | null;
  takeProfitPips?: number | null;
  expiryUnixMs?: number | null;
}) => apiPost<ExecResult>("/orders/amend", body);

// ── Advanced settings / diagnostics / data import ─────────────────────────
// ── Chart scroll-back ─────────────────────────────────────────────────────
// (Indicator overlays are computed client-side by KLineChart since the v10
// migration; the server /indicators endpoint remains for CLI/API users.)
export const chartHistory = (symbol: string, timeframe: string, beforeMs: number, limit = 500) =>
  apiGet<{ candles: { tsMs: number | null; open: number; high: number; low: number; close: number }[]; hasMore: boolean }>(
    `/chart/history?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}&beforeMs=${beforeMs}&limit=${limit}`,
  );

export type SettingsView = {
  dataDir: string;
  tradingMode: string;
  discoveryMode: string;
  effectiveDiscoveryMode: "risky" | "prop_firm" | "strict";
  tradingModeDivergent: boolean;
  computeMode: string;
  searchDevice: string;
  riskyStartBalance: number;
  riskyTargetBalance: number;
  riskyHorizonDays: number;
  sharedSearchRiskMin: number;
  sharedSearchRiskMax: number;
  riskySearchRiskMin: number;
  riskySearchRiskMax: number;
  propFirmSearchRiskMin: number;
  propFirmSearchRiskMax: number;
  searchHighQualityConfidence: number;
  propFirmSearchProfitTargetPct: number;
  propFirmSearchMaxDailyLossPct: number;
  propFirmSearchMaxDrawdownPct: number;
  propFirmSearchMinTradingDays: number;
  propFirmSearchWindowDays: number;
  propFirmSearchWindowCount: number;
  propFirmSearchPassRate: number;
  autoRediscoverOnCull: boolean;
  newsCalendarEnabled: boolean;
  newsCalendarSource: string;
  newsTradingMode: "block_on_news" | "allow_always" | "warn_only";
  newsTradingModeDisplayName: string;
  searchPopulation: number;
  searchPopulationAuto: boolean;
  searchGenerations: number;
  searchMaxHours: number;
  searchMaxIndicators: number;
  searchPortfolioSize: number;
  searchCorrThreshold: number;
  searchMaxRows: number;
  prefilterTopK: number;
  convergencePatience: number;
  stagnationPatience: number;
  noveltyWeight: number;
  disableSmcGate: boolean;
  maxPortfolioRisk: number;
  liveMlGate: boolean;
  blendGateFloor: number;
  blendVetoBelow: number;
};
export const settings = () => apiGet<SettingsView>("/settings");

export type SettingsUpdate = {
  dataDir?: string;
  tradingMode?: "risky" | "prop_firm";
  computeMode?: "auto" | "cpu" | "gpu";
  searchDevice?: "" | "auto" | "cpu" | "cuda_required";
  riskPerTrade?: number;
  maxPortfolioRisk?: number;
  riskyStartBalance?: number;
  riskyTargetBalance?: number;
  riskyHorizonDays?: number;
  sharedSearchRiskMin?: number;
  sharedSearchRiskMax?: number;
  riskySearchRiskMin?: number;
  riskySearchRiskMax?: number;
  propFirmSearchRiskMin?: number;
  propFirmSearchRiskMax?: number;
  searchHighQualityConfidence?: number;
  propFirmSearchProfitTargetPct?: number;
  propFirmSearchMaxDailyLossPct?: number;
  propFirmSearchMaxDrawdownPct?: number;
  propFirmSearchMinTradingDays?: number;
  propFirmSearchWindowDays?: number;
  propFirmSearchWindowCount?: number;
  propFirmSearchPassRate?: number;
  // Discovery search knobs (models.prop_search_*)
  searchPopulation?: number;
  // SEARCH-MORE knob: true + CUDA card raises the GA population to the card's
  // fits ceiling (max 16384) at run start. Changes what is searched.
  searchPopulationAuto?: boolean;
  searchGenerations?: number;
  searchMaxHours?: number;
  searchMaxIndicators?: number;
  searchPortfolioSize?: number;
  searchCorrThreshold?: number;
  searchMaxRows?: number;
  // GA anti-stagnation tuning (models.discovery_runtime / models.search_runtime)
  prefilterTopK?: number;
  convergencePatience?: number;
  stagnationPatience?: number;
  noveltyWeight?: number;
  disableSmcGate?: boolean;
  liveMlGate?: boolean;
  blendGateFloor?: number;
  blendVetoBelow?: number;
  // Auto-cull retirement → automatic rediscovery on the same symbol+TF.
  autoRediscoverOnCull?: boolean;
  // News gate config
  newsCalendarEnabled?: boolean;
  newsCalendarSource?: string;
  newsTradingMode?: "block_on_news" | "allow_always" | "warn_only";
};
export const updateSettings = (payload: SettingsUpdate) => apiPost<SettingsView>("/settings", payload);

export const brokerTimeframes = () => apiGet<{ count: number; timeframes: string[] }>("/broker/timeframes");
export type KnobEntry = {
  id: string;
  section: string;
  label: string;
  kind: "Int" | "Float" | "Bool" | "Text" | "Enum" | "Path";
  min?: number;
  max?: number;
  enumChoices?: string[];
  default: string;
  current: string;
  helpShort: string;
  helpLong: string;
  presetConservative: string;
  presetBalanced: string;
  presetAggressive: string;
};
export type KnobCatalog = {
  schemaVersion: number;
  generatedAtUnixMs: number;
  knobs: KnobEntry[];
};
export type RawSettingsView = { yaml: string; path: string };
export type RawSettingsSaveResult = {
  ok: boolean;
  path: string;
  backupPath: string;
};
export type DiagnosticsReport = {
  zipPath: string;
  totalBytes: number;
  filesIncluded: string[];
  emailSubject: string;
  emailBody: string;
  emailRecipient: string;
};
export const knobCatalog = () => apiGet<KnobCatalog>("/settings/knob-catalog");
export const settingsRaw = () => apiGet<RawSettingsView>("/settings/raw");
export const saveSettingsRaw = (yaml: string) =>
  apiPost<RawSettingsSaveResult>("/settings/raw", { yaml });
export const diagnosticsReport = () => apiPost<DiagnosticsReport>("/diagnostics/report", {});
export const dataImport = (
  sourcePath: string,
  sourceFormat: DataImportSourceFormat,
  sourceNamespace: string,
  symbol: string,
  timeframe: string,
  barTimestampConvention: DataImportBody["barTimestampConvention"],
  expectedGeneration: string | null = null,
) =>
  apiPost<DataImportOutcome>(
    "/data/import",
    dataImportBody(
      sourcePath,
      sourceFormat,
      sourceNamespace,
      symbol,
      timeframe,
      barTimestampConvention,
      expectedGeneration,
    ),
  );

// ── Storage transparency (where every file lives) ─────────────────────────
export type StorageEntry = {
  key: string;
  label: string;
  path: string;
  exists: boolean;
  isDir: boolean;
  sizeBytes: number;
  itemCount: number;
  lastModifiedMs: number | null;
  kind: string;
  scanStatus: "complete" | "partial" | "missing" | "unavailable";
  scanError: string | null;
};
export const storagePaths = () => apiGet<{ entries: StorageEntry[] }>("/storage/paths");
export const openPath = (path: string) => invoke("open_path", { path });
// Refresh real per-symbol costs (commission/swap/spread) from cTrader → cost model.
export const refreshBrokerCosts = () => invoke<string>("refresh_broker_costs");

// ── Autopilot: existing strategy portfolios ───────────────────────────────
export type PortfolioEntry = {
  path: string;
  fileName: string;
  symbol: string | null;
  baseTf: string | null;
  geneCount: number | null;
  sizeBytes: number;
  modifiedMs: number | null;
  blacklisted?: boolean;
};
export const portfoliosList = () => apiGet<{ count: number; portfolios: PortfolioEntry[] }>("/portfolios/list");

// Permanent auto-cull blacklist — strategies retired after too many losses.
export type BlacklistEntry = {
  fingerprint: string;
  portfolioPath: string;
  symbol: string | null;
  reason: string;
  consecutiveLosses: number;
  netPnl: number;
  retiredAtUnixMs: number;
};
export const strategyBlacklist = () => apiGet<BlacklistEntry[]>("/strategy/blacklist");

// Live↔backtest parity harness: does the live bar-window reproduce the
// long-history signals for this portfolio? FAIL = live ≠ validated backtest.
export type ParityReport = {
  symbol: string;
  baseTf: string;
  referenceBars: number;
  windowBars: number;
  comparedBars: number;
  directionMismatches: number;
  mismatchSamples: { barTsMs: number; reference: string; window: string }[];
  maxSlDeltaPips: number;
  maxTpDeltaPips: number;
  verdict: "PASS" | "FAIL";
  note: string;
};
export const parityCheck = (portfolio: string, window?: number) =>
  apiGet<ParityReport>(
    `/autonomous/parity?portfolio=${encodeURIComponent(portfolio)}${window ? `&window=${window}` : ""}`,
  );

// Monte-Carlo tail risk: the max-drawdown DISTRIBUTION of the portfolio's
// trade sequence at the CURRENT risk sizing (p95 + ruin probability).
export type TailRiskReport = {
  portfolio: string;
  trades: number;
  iterations: number;
  riskFraction: number;
  mode: string;
  maxDdP50Pct: number;
  maxDdP90Pct: number;
  maxDdP95Pct: number;
  maxDdP99Pct: number;
  worstDdPct: number;
  ruinThresholdPct: number;
  ruinProbabilityPct: number;
  medianFinalMultiple: number;
  /** Risk-constrained Kelly (Busseti/Ryu/Boyd), solved on the FULL empirical
   *  R-multiple distribution (fat left tails shrink it — CVaR-aware): largest
   *  risk-per-trade (%) with ≤5% chance of ever losing half the account.
   *  null = no edge / no R data. */
  rckRiskPct: number | null;
  /** p95 of the longest time-under-water streak (trades below the equity
   *  peak) across reshuffled paths — the recovery number prop firms test. */
  underwaterP95Trades: number;
  note: string;
};
export const tailRisk = (portfolio: string) =>
  apiGet<TailRiskReport>(`/autonomous/tailrisk?portfolio=${encodeURIComponent(portfolio)}`);

// First-passage prop-firm challenge Monte Carlo: pass/funded probability per
// risk-per-trade size against FTMO-style barriers, + the challenge-optimal size.
export type ChallengeSweepPoint = {
  riskPct: number;
  passPhase1Pct: number;
  fundedPct: number;
  bustPct: number;
  timeoutPct: number;
  medianDaysPhase1: number;
};
export type ChallengeReport = {
  portfolio: string;
  trades: number;
  iterations: number;
  profitTargetPct: number;
  phase2TargetPct: number;
  dailyLossPct: number;
  maxLossPct: number;
  dayLimitPhase1: number;
  dayLimitPhase2: number;
  tradesPerDay: number;
  sweep: ChallengeSweepPoint[];
  bestRiskPct: number;
  bestFundedPct: number;
  attemptsFor90Pct: number;
  note: string;
};
export const challengeSim = (portfolio: string) =>
  apiGet<ChallengeReport>(`/autonomous/challenge?portfolio=${encodeURIComponent(portfolio)}`);

// ── Federation Phase 0 — serverless shared discovery (SETI@home-style) ─────
export type FedJob = { symbol: string; baseTf: string };
export type FedStatus = {
  jobsQueued: number;
  leases: { job: FedJob; worker: string; leasedAtUnixMs: number }[];
  received: { worker: string; symbol: string; baseTf: string; savedPath: string; receivedAtUnixMs: number }[];
  tokenRequired: boolean;
  workerRunning: boolean;
  workerStatus: string;
};
export const federationStatus = () => apiGet<FedStatus>("/federation/status");
export const federationSetJobs = (combos: FedJob[], token?: string) =>
  apiPost<{ queued: number }>("/federation/jobs", { combos, token });
export const federationWorkerStart = (coordinatorUrl: string, workerId?: string, token?: string) =>
  apiPost<{ started: boolean }>("/federation/worker/start", { coordinatorUrl, workerId, token });
export const federationWorkerStop = () => apiPost<{ stopped: boolean }>("/federation/worker/stop");

// The mesh sidecar's aggregated swarm capacity — the network as one machine.
export type SwarmCapacity = {
  running: boolean;
  nodes: number;
  totalCores: number;
  totalRamGb?: number;
  totalGpus?: number;
};
export const swarmCapacity = () => apiGet<SwarmCapacity>("/mesh/swarm");

// Mesh opt-in toggle — the desktop shell owns the sidecar lifecycle (Tauri
// commands, not the HTTP backend). `enabled` = the operator opted in (auto-start
// with the app); `running` = the sidecar process is alive right now.
export type MeshStatus = { enabled: boolean; running: boolean };
export const meshStatus = () => invoke<MeshStatus>("mesh_status");
export const meshSetEnabled = (enabled: boolean) =>
  invoke<MeshStatus>("mesh_set_enabled", { enabled });

// ── Autonomous LLM supervisor ───────────────────────────────────────────────
export type SupervisorLogEntry = {
  tsMs: number;
  kind: string; // tick | action | error | note
  detail: string;
  action?: unknown;
  result?: string;
};
export type SupervisorObservation = {
  observedAtUnixMs: number;
  // Same scope-checked account DTO as HTTP/SSE, including response-derived identity.
  account: AccountStreamSnap | null;
  accountFailure: { code: string; detail: string; observedAtUnixMs: number } | null;
  market: { spots: Tick[]; snapshotAtUnixMs: number; symbolCount: number };
  liveEngines: LiveEngineStatus[];
  liveEngineError: string | null;
};
export type SupervisorStatus = {
  config: SupervisorConfig;
  log: SupervisorLogEntry[];
  cycleRunning: boolean;
  observation: SupervisorObservation;
};
export const supervisorStatus = () =>
  apiGet<SupervisorStatus>("/supervisor/status");
export const supervisorConfig = (b: Partial<SupervisorConfig>) =>
  apiPost<{ config: SupervisorConfig }>("/supervisor/config", b);
export const supervisorTick = () => apiPost<{ summary: string }>("/supervisor/tick");
export const supervisorChat = (message: string) =>
  apiPost<{ reply: string; summary: string }>("/supervisor/chat", { message });

// Offline learning report from live experience (never touches live trading).
export type ExperienceGroup = {
  portfolio: string; records: number; trainN: number; testN: number;
  baselinePct: number; oosAccuracyPct: number; edgePct: number; verdict: string;
};
export const experienceTrain = () =>
  apiPost<{ totalRecords: number; usableRecords: number; groups: ExperienceGroup[]; note: string }>("/experience/train");

// ── Session-aware spread stats (recorded from the broker's own ticks) ──────
export type HourSpread = { samples: number; meanPips: number; maxPips: number };
export type SpreadStats = {
  symbols: Record<string, { hourly: HourSpread[] }>;
  updatedMs: number;
};
export const spreadStats = () => apiGet<SpreadStats>("/data/spread-stats");

// ── Strategy report: monthly journal + validation verdict + flags ─────────
export type RecordedStrategyEvaluation = {
  policyIdentityHash: string;
  accountCurrency: string;
  initialCapital: number;
  riskPerTradeMin: number;
  riskPerTradeMax: number;
  highQualityConfidence: number;
  confidenceBasis: "signal_threshold_margin";
  growthGoal: {
    referenceStartBalance: number;
    targetBalance: number;
    horizonDays: number;
  } | null;
};
export type StrategyEntry = {
  mode: string;
  /** Archived evaluator only; null when the saved policy lacks these fields. */
  recordedEvaluation: RecordedStrategyEvaluation | null;
  /** Selected unsealed journal denominator, not a replacement policy balance. */
  diagnosticInitialCapital: number;
  dir: string;
  symbol: string;
  timeframe: string;
  base: string;
  strategyId: string;
  exactGeneHash: string;
  trades: number;
  winRate: number | null;
  profitFactor: number | null;
  sharpe: number | null;
  cpcvPassed: boolean | null;
  walkforwardPassed: boolean | null;
  validationComplete: boolean | null;
  spanStart: string | null;
  spanEnd: string | null;
  years: number;
  cagrPct: number | null;
  finalFrom1000: number;
  maxDdPct: number;
  flags: string[];
  /** When the strategy was discovered (unix ms), from the artifact's mtime. */
  discoveredAtMs: number | null;
};
export type MonthRow = { month: string; balance: number; returnPct: number; trades: number };
export type StrategyReport = StrategyEntry & { monthly: MonthRow[]; yearly: MonthRow[] };
export const strategyList = () => apiGet<{ count: number; strategies: StrategyEntry[]; unavailable: string[] }>("/strategy/list");
export const strategyReport = (dir: string, base: string, strategyId: string, exactGeneHash: string) =>
  apiGet<StrategyReport>(`/strategy/report?dir=${encodeURIComponent(dir)}&base=${encodeURIComponent(base)}&strategy_id=${encodeURIComponent(strategyId)}&exact_gene_hash=${encodeURIComponent(exactGeneHash)}`);

// ── Trade-confirmation / actions queue ────────────────────────────────────
export type PendingActionKind =
  | {
      kind: "close_position";
      position_id: number;
      volume_units: number;
      symbol_hint: string | null;
    }
  | {
      kind: "mcp_call";
      server: string;
      tool: string;
      args: unknown;
    };
export type PendingAction = {
  id: string;
  schema_version: number;
  kind: PendingActionKind;
  reason: string;
  proposed_at_unix_ms: number;
  expires_at_unix_ms: number;
  status: "pending" | "confirmed" | "rejected" | "expired" | "executed" | "failed";
  result_note: string;
};
export type PendingActionsView = { actions: PendingAction[] };
export const pendingActions = () => apiGet<PendingActionsView>("/actions/pending");
export const confirmAction = (id: string, volumeUnitsOverride?: number) =>
  apiPost<{ ok: boolean; action_id: string; status: string }>(
    `/actions/${encodeURIComponent(id)}/confirm`,
    volumeUnitsOverride === undefined ? undefined : { volumeUnitsOverride },
  );
export const rejectAction = (id: string, reason?: string) =>
  apiPost<{ ok: boolean; action: PendingAction }>(
    `/actions/${encodeURIComponent(id)}/reject`,
    reason === undefined ? undefined : { reason },
  );
