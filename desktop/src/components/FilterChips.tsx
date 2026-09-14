/** Toggle-chip row. An empty selection means "all", never "none". */
export function FilterChips({
  label,
  options,
  selected,
  onToggle,
}: {
  label: string;
  options: string[];
  selected: string[];
  onToggle: (value: string) => void;
}) {
  if (options.length < 2) return null;
  return (
    <>
      <div className="muted small" style={{ marginTop: 8 }}>
        {label} <span className="muted">({selected.length || "all"})</span>
      </div>
      <div className="chip-row">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            className={`chip ${selected.includes(option) ? "on" : ""}`}
            onClick={() => onToggle(option)}
          >
            {option}
          </button>
        ))}
      </div>
    </>
  );
}
