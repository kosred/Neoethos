import { useEffect, useState } from "react";
import { brokerTimeframes, dataBootstrap } from "../api";
import { CANONICAL_BROKER_TIMEFRAMES } from "../timeframes";

let symbolsCache: string[] | null = null;
let symbolRequest: Promise<string[]> | null = null;
let timeframesCache: string[] | null = null;
let timeframeRequest: Promise<string[]> | null = null;

const loadSymbols = (): Promise<string[]> => {
  if (symbolsCache) return Promise.resolve(symbolsCache);
  if (!symbolRequest) {
    symbolRequest = dataBootstrap()
      .then((data) => {
        symbolsCache = [...data.symbols].sort();
        return symbolsCache;
      })
      .finally(() => {
        symbolRequest = null;
      });
  }
  return symbolRequest;
};

const loadTimeframes = (): Promise<string[]> => {
  if (timeframesCache) return Promise.resolve(timeframesCache);
  if (!timeframeRequest) {
    timeframeRequest = brokerTimeframes()
      .then((data) => {
        timeframesCache = data.timeframes.length > 0
          ? data.timeframes
          : [...CANONICAL_BROKER_TIMEFRAMES];
        return timeframesCache;
      })
      .catch(() => {
        timeframesCache = [...CANONICAL_BROKER_TIMEFRAMES];
        return timeframesCache;
      })
      .finally(() => {
        timeframeRequest = null;
      });
  }
  return timeframeRequest;
};

/** Force a re-fetch after a data operation adds a canonical symbol. */
export function invalidateSymbolCache() {
  symbolsCache = null;
  symbolRequest = null;
}

export function useSymbolOptions(): string[] {
  const [options, setOptions] = useState<string[]>(() => symbolsCache ?? []);
  useEffect(() => {
    let active = true;
    void loadSymbols().then((loaded) => {
      if (active) setOptions(loaded);
    });
    return () => {
      active = false;
    };
  }, []);
  return options;
}

export function useTimeframeOptions(): string[] {
  const [options, setOptions] = useState<string[]>(
    () => timeframesCache ?? [...CANONICAL_BROKER_TIMEFRAMES],
  );
  useEffect(() => {
    let active = true;
    void loadTimeframes().then((loaded) => {
      if (active) setOptions(loaded);
    });
    return () => {
      active = false;
    };
  }, []);
  return options;
}
