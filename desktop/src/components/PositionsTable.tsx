import type { StreamPosition, Position } from "../api";

/**
 * The ONE open-positions table used everywhere (Dashboard, Positions, …).
 * Live symbol name + P/L (currency) + pips come from the SSE account stream;
 * entry/SL/TP are merged in from the Tauri snapshot by position id.
 * Pass `onClose` to show a Close column (trading screens only).
 */
export default function PositionsTable({
  live,
  detail,
  currency = "",
  onClose,
  busy,
}: {
  live: StreamPosition[];
  detail?: Position[];
  currency?: string;
  onClose?: (positionId: number, volumeUnits: number) => void;
  busy?: boolean;
}) {
  if (!live || live.length === 0) return <p className="muted">No open positions.</p>;
  const byId = new Map((detail ?? []).map((p) => [p.positionId, p]));
  let total = 0;

  return (
    <table className="tbl">
      <thead>
        <tr>
          <th>Side</th>
          <th>Symbol</th>
          <th>Volume</th>
          <th>Entry</th>
          <th>SL</th>
          <th>TP</th>
          <th>P/L{currency ? ` (${currency})` : ""}</th>
          <th>Pips</th>
          {onClose && <th></th>}
        </tr>
      </thead>
      <tbody>
        {live.map((p) => {
          const d = byId.get(p.positionId);
          const cls = p.pnlUsd >= 0 ? "buy" : "sell";
          total += p.pnlUsd;
          return (
            <tr key={p.positionId}>
              <td className={p.side.toLowerCase().includes("buy") ? "buy" : "sell"}>{p.side}</td>
              <td>{p.symbol}</td>
              <td>{p.volume}</td>
              <td>{d?.price ?? "—"}</td>
              <td>{d?.stopLoss ?? "—"}</td>
              <td>{d?.takeProfit ?? "—"}</td>
              <td className={cls}><b>{p.pnlUsd >= 0 ? "+" : ""}{p.pnlUsd.toFixed(2)}</b></td>
              <td className={cls}>{p.pnlPips >= 0 ? "+" : ""}{p.pnlPips.toFixed(1)}</td>
              {onClose && (
                <td>
                  <button className="danger" disabled={busy} onClick={() => onClose(p.positionId, p.volumeUnits)}>
                    Close
                  </button>
                </td>
              )}
            </tr>
          );
        })}
      </tbody>
      <tfoot>
        <tr>
          <td colSpan={6} style={{ textAlign: "right", color: "#6b7280" }}>Total P/L</td>
          <td className={total >= 0 ? "buy" : "sell"}><b>{total >= 0 ? "+" : ""}{total.toFixed(2)}</b></td>
          <td colSpan={onClose ? 2 : 1}></td>
        </tr>
      </tfoot>
    </table>
  );
}
