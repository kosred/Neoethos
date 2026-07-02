// Typed wrappers over every Tauri command. The UI never touches `invoke`
// directly — it calls these, so the command surface is one auditable file.
import { invoke } from "@tauri-apps/api/core";

// ── In-process backend (full neoethos-app axum API over loopback) ─────────────
// The Tauri shell runs the whole backend in-process and tells us the port via
// the `api_base` command. We resolve it once and reuse it for every call.
let _apiBase: string | null = null;
export async function apiBaseUrl(): Promise<string> {
  if (_apiBase) return _apiBase;
  _apiBase = await invoke<string>("api_base");
  return _apiBase;
}
async function _check(r: Response): Promise<Response> {
  if (!r.ok) {
    const detail = await r.text().catch(() => "");
    throw new Error(`${r.status} ${r.statusText}${detail ? ` — ${detail}` : ""}`);
  }
  return r;
}
export async function apiGet<T>(path: string): Promise<T> {
  const base = await apiBaseUrl();
  const r = await _check(await fetch(`${base}${path}`));
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
export async function openSse(
  path: string,
  eventName: string,
  onData: (data: any) => void,
  onStatus?: (connected: boolean) => void,
): Promise<() => void> {
  const base = await apiBaseUrl();
  const es = new EventSource(`${base}${path}`);
  es.addEventListener("open", () => onStatus?.(true));
  es.addEventListener("error", () => onStatus?.(false));
  es.addEventListener(eventName, (e) => {
    try {
      onData(JSON.parse((e as MessageEvent).data));
    } catch {
      /* ignore malformed frame */
    }
  });
  return () => es.close();
}

export type Tick = {
  symbolId: number;
  symbolName: string;
  bid: number;
  ask: number;
  midPrice: number;
  brokerTimestampMs: number;
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
  pnlPips: number;
  pnlUsd: number; // live P/L in the ACCOUNT currency (field name is legacy)
  entryPrice: number | null; // all server-provided — client does NO conversion
  stopLoss: number | null;
  takeProfit: number | null;
  volumeLots: number | null; // cTrader-parity lots (1.17), not raw units
};
export type AccountStreamSnap = {
  balance: number;
  equity: number;
  freeMargin: number;
  usedMargin: number;
  currency: string;
  fetchedAtUnixMs: number;
  positions: StreamPosition[];
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
export type Position = {
  positionId: number;
  symbolId: number;
  side: string;
  volume: number;
  volumeUnits: number; // raw wire volume — pass THIS to closePosition
  price: number | null;
  stopLoss: number | null;
  takeProfit: number | null;
};
export type AccountSnapshot = {
  accountId: number;
  balance: number;
  equity: number;
  unrealizedPnl: number;
  currency: string;
  openPositions: number;
  positions: Position[];
  live: boolean;
  brokerName: string | null;
  leverage: number | null;
  login: number | null;
  accountType: string | null;
  label: string; // e.g. "LIVE · FTMO · 200k USD · 1:30"
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
export type SymbolCoverage = { symbol: string; bars: number; firstMs: number; lastMs: number; years: number };
/** Per-symbol local-history coverage (years + bars) for the given base TF. */
export const dataCoverage = (symbols: string[], timeframe: string) =>
  invoke<SymbolCoverage[]>("data_coverage", { symbols, timeframe });

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
export const accountSnapshot = () => invoke<AccountSnapshot>("account_snapshot");
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

// ══════════════════════════════════════════════════════════════════════════
// Full backend API (in-process axum server) — every old Flutter feature.
// ══════════════════════════════════════════════════════════════════════════

// ── Engines: Discovery + Training ─────────────────────────────────────────
export type EngineCounter = { name: string; value: number };
export type EnginesStatus = {
  discovery: string;
  training: string;
  autoTrader?: string;
  auto_trader?: string;
  discoverySummary?: string;
  discovery_summary?: string;
  trainingSummary?: string;
  training_summary?: string;
  discoveryStage?: string;
  discovery_stage?: string;
  discoveryPercent?: number;
  discovery_percent?: number;
  discoveryCounters?: EngineCounter[];
  discovery_counters?: EngineCounter[];
  ramTotalGb?: number;
  ramAvailableGb?: number;
  featureStoreMb?: number;
};
export type StartJob = {
  symbol?: string;
  base_tf?: string;
  higher_tfs?: string[];
  population?: number;
  generations?: number;
  max_indicators?: number;
  target_candidates?: number;
  portfolio_size?: number;
};
export const enginesStatus = () => apiGet<EnginesStatus>("/engines/status");
export const discoveryStart = (b: StartJob) => apiPost("/engines/discovery/start", b);
export const discoveryStop = () => apiPost("/engines/discovery/stop");
export const trainingStart = (b: StartJob) => apiPost("/engines/training/start", b);
export const trainingStop = () => apiPost("/engines/training/stop");

// ── Strategy Lab ──────────────────────────────────────────────────────────
const qs = (o: Record<string, string | undefined>) => {
  const p = Object.entries(o).filter(([, v]) => v && v.trim() !== "");
  return p.length ? "?" + p.map(([k, v]) => `${k}=${encodeURIComponent(v!)}`).join("&") : "";
};
export const promotionStatus = (symbol?: string, baseTf?: string) =>
  apiGet<any>("/strategy_lab/promotion" + qs({ symbol, base_tf: baseTf }));
export const promoteStrategy = (symbol?: string, baseTf?: string) =>
  apiPost<any>("/strategy_lab/promote", { symbol: symbol || undefined, base_tf: baseTf || undefined });

// ── Autonomous trader ─────────────────────────────────────────────────────
export const autonomousStatus = () => apiGet<any>("/autonomous/status");
export const autonomousStart = (b?: unknown) => apiPost("/autonomous/start", b ?? {});
export const autonomousStop = () => apiPost("/autonomous/stop");
export const autonomousReplay = (b?: unknown) => apiPost("/autonomous/replay", b ?? {});

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
  propFirmRulesEnabled: boolean;
  riskyModeCooldownRemainingSecs: number | null;
};
export const riskInfo = () => apiGet<RiskInfo>("/risk");
export const setRiskPreset = (preset: string) => apiPost("/risk/preset", { preset });
export type RiskyParams = {
  startingUsd?: number;
  targetUsd?: number;
  riskFraction?: number;
  winRate?: number;
  rewardToRisk?: number;
  tradesPerDay?: number;
};
export type RiskyScenario = {
  startingUsd: number;
  targetUsd: number;
  riskFraction: number;
  winRate: number;
  rewardToRisk: number;
  tradesPerDay: number;
  bestCaseDays: number | null;
  expectedDays: number | null;
  conservativeDays: number | null;
  ruinProbability: number;
  riskFractionMin: number;
  riskFractionMax: number;
};
export const riskyScenarios = (p: RiskyParams = {}) => {
  const q = Object.entries(p)
    .filter(([, v]) => v !== undefined && v !== null && !Number.isNaN(v))
    .map(([k, v]) => `${k}=${v}`)
    .join("&");
  return apiGet<RiskyScenario>(`/risky/scenarios${q ? `?${q}` : ""}`);
};

// ── Hardware ──────────────────────────────────────────────────────────────
export type HardwareInfo = {
  cpu: { model: string; coresLogical: number; coresPhysical: number; loadAvg: number };
  ram: { totalMb: number; usedMb: number; availableMb: number };
  gpu: { name: string; available: boolean; kind: string };
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
  walkforwardSplits: number | null;
  walkforwardAvgAccuracy: number | null;
};
export const intelligence = () => apiGet<IntelligenceInfo>("/intelligence");

// ── Journal ───────────────────────────────────────────────────────────────
export const journalStats = () => apiGet<any>("/journal/stats");
export const journalTrades = () => apiGet<any>("/journal/trades");

// ── News ──────────────────────────────────────────────────────────────────
export const newsFeed = (force = false) => apiGet<any>(`/news/feed${force ? "?force=true" : ""}`);

// ── Data ──────────────────────────────────────────────────────────────────
export type DataBootstrap = {
  dataDir: string;
  dataDirExists: boolean;
  symbols: string[];
  fileCount: number;
  lastTouchedUnixMs: number | null;
};
export const dataBootstrap = () => apiGet<DataBootstrap>("/data/bootstrap");
export const dataFetch = (b: unknown) => apiPost("/data/fetch", b);

// ── Market Watch / watchlist ──────────────────────────────────────────────
export const getWatchlist = () => apiGet<any>("/watchlist");
export const setWatchlist = (symbols: string[]) => apiPost("/watchlist", { symbols });

// ── AI Desk (Codex / ChatGPT subscription) ────────────────────────────────
export const codexStatus = () => apiGet<any>("/auth/codex/status");
export const codexStart = () => apiPost<any>("/auth/codex/start");
export const codexLogout = () => apiPost("/auth/codex/logout");
export const codexChat = (prompt: string, model?: string) =>
  apiPost<{ model: string; response: string; total_tokens: number }>("/codex/chat", { prompt, model });

// ── Account / broker detail (history, profile, margin, cashflow) ───────────
export const brokerProfile = () => apiGet<{ userId: number }>("/broker/profile");
export const brokerVersion = () => apiGet<{ version: string }>("/broker/version");
export const ordersHistory = () => apiGet<any>("/broker/orders/history");
export const cashFlow = () => apiGet<any>("/broker/cashflow");
export const expectedMargin = (symbolId: number, volume: number) =>
  apiGet<any>(`/broker/margin/expected?symbolId=${symbolId}&volume=${volume}`);

// ── Position protection (move SL/TP, breakeven, trailing) ──────────────────
export const amendProtection = (
  positionId: number,
  stopLossPrice?: number | null,
  takeProfitPrice?: number | null,
  trailingStopLoss?: boolean,
) =>
  apiPost("/positions/protection", {
    position_id: positionId,
    stop_loss_price: stopLossPrice ?? null,
    take_profit_price: takeProfitPrice ?? null,
    trailing_stop_loss: trailingStopLoss ?? null,
  });

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

// ── Advanced settings / diagnostics / data import ─────────────────────────
// ── Chart scroll-back ─────────────────────────────────────────────────────
// (Indicator overlays are computed client-side by KLineChart since the v10
// migration; the server /indicators endpoint remains for CLI/API users.)
export const chartHistory = (symbol: string, timeframe: string, beforeMs: number, limit = 500) =>
  apiGet<{ candles: { tsMs: number | null; open: number; high: number; low: number; close: number }[]; hasMore: boolean }>(
    `/chart/history?symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}&beforeMs=${beforeMs}&limit=${limit}`,
  );

export const settings = () => apiGet<any>("/settings");

export type SettingsUpdate = {
  dataDir?: string;
  uiLocale?: "en" | "el";
  tradingMode?: "risky" | "prop_firm";
  computeMode?: "auto" | "cpu" | "gpu";
  riskPerTrade?: number;
  maxPortfolioRisk?: number;
  riskyStartBalance?: number;
  riskyTargetBalance?: number;
  riskyHorizonDays?: number;
  // Discovery search knobs (models.prop_search_*)
  searchPopulation?: number;
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
  // News gate config
  newsCalendarEnabled?: boolean;
  newsCalendarSource?: string;
  newsTradingMode?: "block_on_news" | "allow_always" | "warn_only";
};
export const updateSettings = (payload: SettingsUpdate) => apiPost<any>("/settings", payload);

export const brokerTimeframes = () => apiGet<{ count: number; timeframes: string[] }>("/broker/timeframes");
export const knobCatalog = () => apiGet<any>("/settings/knob-catalog");
export const settingsRaw = () => apiGet<any>("/settings/raw");
export const saveSettingsRaw = (yaml: string) => apiPost("/settings/raw", { yaml });
export const diagnosticsReport = () => apiPost<any>("/diagnostics/report", {});
export const dataImport = (sourcePath: string, symbol: string, timeframe: string) =>
  apiPost<any>("/data/import", { source_path: sourcePath, symbol, timeframe });

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

// ── Autonomous LLM supervisor ───────────────────────────────────────────────
export type SupervisorConfig = { enabled: boolean; intervalMinutes: number; maxActionsPerTick: number; directives: string[] };
export type SupervisorLogEntry = {
  tsMs: number;
  kind: string; // tick | action | error | note
  detail: string;
  action?: any;
  result?: string;
};
export const supervisorStatus = () =>
  apiGet<{ config: SupervisorConfig; log: SupervisorLogEntry[] }>("/supervisor/status");
export const supervisorConfig = (b: Partial<SupervisorConfig>) =>
  apiPost<{ config: SupervisorConfig }>("/supervisor/config", b);
export const supervisorTick = () => apiPost<{ summary: string }>("/supervisor/tick");
export const supervisorChat = (message: string) =>
  apiPost<{ reply: string; summary: string }>("/supervisor/chat", { message });

// ── Session-aware spread stats (recorded from the broker's own ticks) ──────
export type HourSpread = { samples: number; meanPips: number; maxPips: number };
export type SpreadStats = {
  symbols: Record<string, { hourly: HourSpread[] }>;
  updatedMs: number;
};
export const spreadStats = () => apiGet<SpreadStats>("/data/spread-stats");

// ── Strategy report: monthly journal + validation verdict + flags ─────────
export type StrategyEntry = {
  mode: string;
  dir: string;
  symbol: string;
  timeframe: string;
  base: string;
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
  cagrPct: number;
  finalFrom1000: number;
  maxDdPct: number;
  flags: string[];
};
export type MonthRow = { month: string; balance: number; returnPct: number; trades: number };
export type StrategyReport = StrategyEntry & { monthly: MonthRow[]; yearly: MonthRow[] };
export const strategyList = () => apiGet<{ count: number; strategies: StrategyEntry[] }>("/strategy/list");
export const strategyReport = (dir: string, base: string) =>
  apiGet<StrategyReport>(`/strategy/report?dir=${encodeURIComponent(dir)}&base=${encodeURIComponent(base)}`);

// ── Trade-confirmation / actions queue ────────────────────────────────────
export const pendingActions = () => apiGet<any>("/actions/pending");
export const confirmAction = (id: string) => apiPost(`/actions/${id}/confirm`);
export const rejectAction = (id: string) => apiPost(`/actions/${id}/reject`);
