import type { StreamPosition } from "../api";

const ago = (ms: number | null) => {
  if (!ms) return "—";
  const s = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h`;
  return `${Math.floor(s / 86400)}d`;
};

/**
 * The ONE open-positions table used everywhere (Dashboard, Positions, …).
 * EVERY field — symbol, volume, entry, SL, TP, live P/L, pips — comes straight
 * from the server account stream. The client does NO conversion or merging.
 * Pass `onClose` to show a Close column (trading screens only).
 */
export default function PositionsTable({
  live,
  currency = "",
  onClose,
  onEdit,
  busy,
}: {
  live: StreamPosition[];
  currency?: string;
  onClose?: (positionId: number, volumeUnits: number) => void;
  onEdit?: (positionId: number) => void;
  busy?: boolean;
}) {
  if (!live || live.length === 0) return <p className="muted">No open positions.</p>;
  const hasActions = !!(onClose || onEdit);
  let total = 0;

  return (
    <table className="tbl">
      <thead>
        <tr>
          <th>Side</th>
          <th>Symbol</th>
          <th>Lots</th>
          <th>Entry</th>
          <th>SL</th>
          <th>TP</th>
          <th>Opened</th>
          <th>P/L{currency ? ` (${currency})` : ""}</th>
          <th>Pips</th>
          {hasActions && <th></th>}
        </tr>
      </thead>
      <tbody>
        {live.map((p) => {
          const cls = p.pnlUsd >= 0 ? "buy" : "sell";
          total += p.pnlUsd;
          return (
            <tr key={p.positionId}>
              <td className={p.side.toLowerCase().includes("buy") ? "buy" : "sell"}>{p.side}</td>
              <td>{p.symbol}</td>
              <td>{p.volumeLots != null ? p.volumeLots.toFixed(2) : p.volume}</td>
              <td>{p.entryPrice ?? "—"}</td>
              <td>{p.stopLoss ?? "—"}</td>
              <td>{p.takeProfit ?? "—"}</td>
              <td className="muted">{ago(p.openTimestampMs)}</td>
              <td className={cls}><b>{p.pnlUsd >= 0 ? "+" : ""}{p.pnlUsd.toFixed(2)}</b></td>
              <td className={cls}>{p.pnlPips >= 0 ? "+" : ""}{p.pnlPips.toFixed(1)}</td>
              {hasActions && (
                <td style={{ whiteSpace: "nowrap" }}>
                  {onEdit && (
                    <button disabled={busy} onClick={() => onEdit(p.positionId)} style={{ marginRight: 4 }}>
                      Edit
                    </button>
                  )}
                  {onClose && (
                    <button className="danger" disabled={busy} onClick={() => onClose(p.positionId, p.volumeUnits)}>
                      Close
                    </button>
                  )}
                </td>
              )}
            </tr>
          );
        })}
      </tbody>
      <tfoot>
        <tr>
          <td colSpan={7} style={{ textAlign: "right", color: "#6b7280" }}>Total P/L</td>
          <td className={total >= 0 ? "buy" : "sell"}><b>{total >= 0 ? "+" : ""}{total.toFixed(2)}</b></td>
          <td colSpan={hasActions ? 2 : 1}></td>
        </tr>
      </tfoot>
    </table>
  );
}
