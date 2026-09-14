import type { CSSProperties } from "react";
import { useSymbolOptions, useTimeframeOptions } from "./selectOptions";

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
  const opts = useSymbolOptions();
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
  const opts = useTimeframeOptions();
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
