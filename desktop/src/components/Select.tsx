import { useEffect, useState, type CSSProperties } from "react";
import { dataBootstrap, brokerTimeframes } from "../api";
import { CANONICAL_BROKER_TIMEFRAMES } from "../timeframes";

// Module-level caches so every dropdown on every screen shares one fetch.
let symbolsCache: string[] | null = null;
let tfsCache: string[] | null = null;

/** Force a re-fetch of the symbol list (call after a data download adds pairs). */
export function invalidateSymbolCache() {
  symbolsCache = null;
}

/** Shared option sources (one fetch, cached) for building custom pickers
 *  like the multi-select queue builder on the Discovery screen. */
export function useSymbolOptions(): string[] {
  const [opts, setOpts] = useState<string[]>(symbolsCache ?? []);
  useEffect(() => {
    if (symbolsCache) {
      setOpts(symbolsCache);
      return;
    }
    dataBootstrap()
      .then((d) => {
        symbolsCache = (d.symbols ?? []).slice().sort();
        setOpts(symbolsCache);
      })
      .catch(() => {});
  }, []);
  return opts;
}

export function useTimeframeOptions(): string[] {
  const [opts, setOpts] = useState<string[]>(
    tfsCache ?? [...CANONICAL_BROKER_TIMEFRAMES],
  );
  useEffect(() => {
    if (tfsCache) {
      setOpts(tfsCache);
      return;
    }
    brokerTimeframes()
      .then((d) => {
        const options = d.timeframes?.length
          ? d.timeframes
          : [...CANONICAL_BROKER_TIMEFRAMES];
        tfsCache = options;
        setOpts(options);
      })
      .catch(() => setOpts([...CANONICAL_BROKER_TIMEFRAMES]));
  }, []);
  return opts;
}

type Common = {
  value: string;
  onChange: (v: string) => void;
  style?: CSSProperties;
  /** Add a leading "(from config)" option that maps to empty string. */
  allowConfig?: boolean;
  className?: string;
  title?: string;
};

/** Scrollable dropdown of the symbols that actually have local data. */
export function SymbolSelect({ value, onChange, style, allowConfig, className, title }: Common) {
  const [opts, setOpts] = useState<string[]>(symbolsCache ?? []);
  useEffect(() => {
    if (symbolsCache) {
      setOpts(symbolsCache);
      return;
    }
    dataBootstrap()
      .then((d) => {
        symbolsCache = (d.symbols ?? []).slice().sort();
        setOpts(symbolsCache);
      })
      .catch(() => {});
  }, []);
  return (
    <select className={className} title={title} value={value} onChange={(e) => onChange(e.target.value)} style={style}>
      {allowConfig && <option value="">(from config)</option>}
      {/* keep a current value that isn't in the list visible rather than silently dropping it */}
      {value && !opts.includes(value) && <option value={value}>{value}</option>}
      {opts.map((s) => (
        <option key={s} value={s}>{s}</option>
      ))}
    </select>
  );
}

/** Scrollable dropdown of the broker's canonical timeframes. */
export function TimeframeSelect({ value, onChange, style, allowConfig, className, title }: Common) {
  const [opts, setOpts] = useState<string[]>(
    tfsCache ?? [...CANONICAL_BROKER_TIMEFRAMES],
  );
  useEffect(() => {
    if (tfsCache) {
      setOpts(tfsCache);
      return;
    }
    brokerTimeframes()
      .then((d) => {
        const options = d.timeframes?.length
          ? d.timeframes
          : [...CANONICAL_BROKER_TIMEFRAMES];
        tfsCache = options;
        setOpts(options);
      })
      .catch(() => setOpts([...CANONICAL_BROKER_TIMEFRAMES]));
  }, []);
  return (
    <select className={className} title={title} value={value} onChange={(e) => onChange(e.target.value)} style={style}>
      {allowConfig && <option value="">(from config)</option>}
      {value && !opts.includes(value) && <option value={value}>{value}</option>}
      {opts.map((t) => (
        <option key={t} value={t}>{t}</option>
      ))}
    </select>
  );
}
