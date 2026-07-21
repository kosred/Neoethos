// Shared filtering pieces for the strategy-facing screens (Strategy Report,
// Autopilot). They live here so the two screens can never drift: the operator
// asked for "the same filters" in both, and a copy-paste pair would diverge on
// the first change.

/** Fastest-first, so timeframe chips read in the order runs actually go. */
export const TF_ORDER = [
  "MN1", "W1", "D1", "H12", "H8", "H6", "H4", "H3", "H2", "H1",
  "M30", "M20", "M15", "M12", "M10", "M6", "M5", "M4", "M3", "M2", "M1",
];

export const tfRank = (t: string) => {
  const i = TF_ORDER.indexOf(t);
  return i < 0 ? 999 : i;
};

/** "2026-07-21 21:47" — sortable at a glance, no locale surprises. */
export const stamp = (ms: number | null | undefined) => {
  if (!ms) return "—";
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
};

/** "3h ago" — relative age, for tooltips and secondary lines. */
export const ago = (ms: number | null | undefined) => {
  if (!ms) return "";
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
};

/** Toggle-chip row. Empty selection means "all" — never "none". */
export function FilterChips({
  label,
  opts,
  sel,
  onToggle,
}: {
  label: string;
  opts: string[];
  sel: string[];
  onToggle: (v: string) => void;
}) {
  if (opts.length < 2) return null;
  return (
    <>
      <div className="muted small" style={{ marginTop: 8 }}>
        {label} <span className="muted">({sel.length || "all"})</span>
      </div>
      <div className="chip-row">
        {opts.map((o) => (
          <button
            key={o}
            type="button"
            className={`chip ${sel.includes(o) ? "on" : ""}`}
            onClick={() => onToggle(o)}
          >
            {o}
          </button>
        ))}
      </div>
    </>
  );
}

/** Curried toggle for a `string[]` filter state. */
export const toggleIn =
  (set: React.Dispatch<React.SetStateAction<string[]>>) => (v: string) =>
    set((cur) => (cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v]));
