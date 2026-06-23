import { intelligence } from "../api";
import { usePoll } from "../hooks";

export default function Intelligence() {
  const { data, error } = usePoll(intelligence, 0);

  return (
    <div className="screen">
      <h1>Intelligence</h1>
      <p className="sub">Trained model artifacts &amp; discovered strategy targets</p>
      {error && <div className="banner warn">{error}</div>}
      {data && (
        <>
          <div className="cards">
            <div className="card"><div className="card-label">ARTIFACTS</div><div className="card-value">{data.artifact_count}</div></div>
            <div className="card"><div className="card-label">TARGETS</div><div className="card-value">{data.discovery_targets.length}</div></div>
            <div className="card"><div className="card-label">WF SPLITS</div><div className="card-value">{data.walkforward_splits ?? "—"}</div></div>
            <div className="card"><div className="card-label">WF ACCURACY</div><div className="card-value">{data.walkforward_avg_accuracy != null ? `${(data.walkforward_avg_accuracy * 100).toFixed(1)}%` : "—"}</div></div>
          </div>

          <h2>Discovered strategies</h2>
          {data.discovery_targets.length === 0 ? (
            <p className="muted">No discovered strategies yet — run Discovery first.</p>
          ) : (
            <table className="tbl">
              <thead><tr><th>Symbol</th><th>Base TF</th><th>Strategy</th><th>Sharpe</th><th>Win rate</th></tr></thead>
              <tbody>
                {data.discovery_targets.map((t, i) => (
                  <tr key={i}>
                    <td>{t.symbol}</td>
                    <td>{t.base_tf}</td>
                    <td style={{ fontFamily: "monospace", fontSize: 11 }}>{t.strategy_id}</td>
                    <td>{t.sharpe != null ? t.sharpe.toFixed(2) : "—"}</td>
                    <td>{t.win_rate != null ? `${(t.win_rate * 100).toFixed(1)}%` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h2>Model store</h2>
          <p className="muted small">{data.models_dir} {data.models_dir_exists ? "" : "(missing)"}</p>
          {data.artifacts.length > 0 && (
            <ul className="file-list">
              {data.artifacts.slice(0, 50).map((a) => <li key={a}>{a}</li>)}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
