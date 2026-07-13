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

export type DataImportBody = Readonly<{
  sourcePath: string;
  symbol: string;
  timeframe: string;
}>;

export function dataImportBody(
  sourcePath: string,
  symbol: string,
  timeframe: string,
): DataImportBody {
  return { sourcePath, symbol, timeframe };
}

export type DataFetchBody = Readonly<{
  symbol: string;
  timeframe: string;
  fromMs: number;
  toMs: number | undefined;
}>;

export function dataFetchBody(
  symbol: string,
  timeframe: string,
  fromMs: number,
  toMs?: number,
): DataFetchBody {
  return { symbol, timeframe, fromMs, toMs };
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
