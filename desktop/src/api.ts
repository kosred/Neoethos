// Typed wrappers over every Tauri command. The UI never touches `invoke`
// directly — it calls these, so the command surface is one auditable file.
import { invoke } from "@tauri-apps/api/core";

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
