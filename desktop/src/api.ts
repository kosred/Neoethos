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

// ── Types (mirror the Rust serde DTOs, camelCase) ─────────────────────────────
export type AppInfo = { version: string; data_root: string; data_root_exists: boolean };
export type Candle = { time: number; open: number; high: number; low: number; close: number };

export type BrokerStatus = {
  configured: boolean;
  hasToken: boolean;
  environment: string;
  accountId: string | null;
};

export type SymbolInfo = { symbolId: number; name: string; enabled: boolean };
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
export type SpotPrice = {
  symbolId: number;
  name: string;
  bid: number | null;
  ask: number | null;
  mid: number | null;
};

// ── Local vortex data (works offline, no broker) ──────────────────────────────
export const appInfo = () => invoke<AppInfo>("app_info");
export const listSymbols = () => invoke<string[]>("list_symbols");
export const listTimeframes = (symbol: string) => invoke<string[]>("list_timeframes", { symbol });
export const localChart = (symbol: string, timeframe: string, limit = 1500) =>
  invoke<Candle[]>("chart", { symbol, timeframe, limit });

// ── Live cTrader (in-process, auto-auth) ──────────────────────────────────────
export const brokerStatus = () => invoke<BrokerStatus>("broker_status");
export const brokerChart = (symbol: string, timeframe: string, limit = 1000) =>
  invoke<Candle[]>("broker_chart", { symbol, timeframe, limit });
export const brokerSymbols = () => invoke<SymbolInfo[]>("broker_symbols");
export const brokerAccounts = () => invoke<AccountInfo[]>("broker_accounts");
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
export const spotPrices = () => invoke<SpotPrice[]>("spot_prices");

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

// ── Risk ──────────────────────────────────────────────────────────────────
export type PresetSummary = {
  id: string;
  displayName?: string;
  display_name?: string;
  maxDailyLossPct?: number;
  max_daily_loss_pct?: number;
  maxOverallDrawdownPct?: number;
  max_overall_drawdown_pct?: number;
  challengeProfitTargetPct?: number;
  challenge_profit_target_pct?: number;
  minTradingDays?: number;
  min_trading_days?: number;
};
export type RiskInfo = {
  risk_per_trade: number;
  min_risk_per_trade: number;
  max_risk_per_trade: number;
  daily_drawdown_limit: number;
  total_drawdown_limit: number;
  max_lot_size: number;
  require_stop_loss: boolean;
  preset: string;
  preset_display_name: string;
  available_presets: PresetSummary[];
  prop_firm_rules_enabled: boolean;
  risky_mode_cooldown_remaining_secs: number | null;
};
export const riskInfo = () => apiGet<RiskInfo>("/risk");
export const setRiskPreset = (preset: string) => apiPost("/risk/preset", { preset });
export const riskyScenarios = () => apiGet<any>("/risky/scenarios");

// ── Hardware ──────────────────────────────────────────────────────────────
export type HardwareInfo = {
  cpu: { model: string; cores_logical: number; cores_physical: number; load_avg: number };
  ram: { total_mb: number; used_mb: number; available_mb: number };
  gpu: { name: string; available: boolean; kind: string };
};
export const hardwareInfo = () => apiGet<HardwareInfo>("/hardware");

// ── Intelligence ──────────────────────────────────────────────────────────
export type DiscoveryTarget = {
  symbol: string;
  base_tf: string;
  strategy_id: string;
  sharpe: number | null;
  win_rate: number | null;
};
export type IntelligenceInfo = {
  models_dir: string;
  models_dir_exists: boolean;
  artifact_count: number;
  artifacts: string[];
  last_touched_unix_ms: number | null;
  discovery_targets: DiscoveryTarget[];
  walkforward_splits: number | null;
  walkforward_avg_accuracy: number | null;
};
export const intelligence = () => apiGet<IntelligenceInfo>("/intelligence");

// ── Journal ───────────────────────────────────────────────────────────────
export const journalStats = () => apiGet<any>("/journal/stats");
export const journalTrades = () => apiGet<any>("/journal/trades");

// ── News ──────────────────────────────────────────────────────────────────
export const newsFeed = (force = false) => apiGet<any>(`/news/feed${force ? "?force=true" : ""}`);

// ── Data ──────────────────────────────────────────────────────────────────
export type DataBootstrap = {
  data_dir: string;
  data_dir_exists: boolean;
  symbols: string[];
  file_count: number;
  last_touched_unix_ms: number | null;
};
export const dataBootstrap = () => apiGet<DataBootstrap>("/data/bootstrap");
export const dataFetch = (b: unknown) => apiPost("/data/fetch", b);

// ── Market Watch / watchlist ──────────────────────────────────────────────
export const getWatchlist = () => apiGet<any>("/watchlist");
export const setWatchlist = (symbols: string[]) => apiPost("/watchlist", { symbols });
export const liveSpots = () => apiGet<any>("/live/spots");

// ── AI Desk (Codex / ChatGPT subscription) ────────────────────────────────
export const codexStatus = () => apiGet<any>("/auth/codex/status");
export const codexStart = () => apiPost<any>("/auth/codex/start");
export const codexLogout = () => apiPost("/auth/codex/logout");
export const codexChat = (prompt: string, model?: string) =>
  apiPost<{ model: string; response: string; total_tokens: number }>("/codex/chat", { prompt, model });
