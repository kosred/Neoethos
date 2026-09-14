import { intelligence } from "../api";
import { usePoll } from "../hooks";

export default function Intelligence() {
  const { data, error, loading, reload } = usePoll(intelligence, 0);

  return (
    <div className="screen">
      <h1>Research inventory</h1>
      <p className="sub">Stored model artifacts &amp; published training handoff targets</p>
      <div className="btn-row"><button disabled={loading} onClick={() => void reload()}>{loading ? "Refreshing…" : "Refresh inventory"}</button></div>
      {error && <div className="banner warn">{error}</div>}
      {(data?.trainingHandoffUnavailable?.length ?? 0) > 0 && <div className="banner warn" role="alert">
        Some saved handoffs could not be verified. Valid entries remain visible; no files were deleted.
        <details><summary>Unavailable handoffs</summary><ul>
          {data!.trainingHandoffUnavailable.map((item) => <li key={item.identity} style={{ overflowWrap: "anywhere" }}>{item.identity}: {item.reason}</li>)}
        </ul></details>
      </div>}
      {data && (
        <>
          <div className="cards">
            <div className="card"><div className="card-label">ARTIFACTS</div><div className="card-value">{data.artifactCount}</div></div>
            <div className="card"><div className="card-label">HANDOFF TARGETS</div><div className="card-value">{data.discoveryTargets.length}</div></div>
          </div>

          <h2>Training handoff targets</h2>
          <p className="muted small">These strategy rows come from published Discovery-to-Training handoffs. For saved Discovery reports, open the Results tab; those reports are not counted here.</p>
          {data.discoveryTargets.length === 0 ? (
            <p className="muted">No published training handoff targets found.</p>
          ) : (
            <table className="tbl">
              <thead><tr><th>Symbol</th><th>Base TF</th><th>Strategy</th><th>Sharpe</th><th>Win rate</th></tr></thead>
              <tbody>
                {data.discoveryTargets.map((t, i) => (
                  <tr key={i}>
                    <td>{t.symbol}</td>
                    <td>{t.baseTf}</td>
                    <td style={{ fontFamily: "monospace", fontSize: 11 }}>{t.strategyId}</td>
                    <td>{t.sharpe != null ? t.sharpe.toFixed(2) : "—"}</td>
                    <td>{t.winRate != null ? `${(t.winRate * 100).toFixed(1)}%` : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h2>Model store</h2>
          <p className="muted small">File presence does not prove compatibility with the current pipeline, validation or permission to trade.</p>
          <p className="muted small">{data.modelsDir} {data.modelsDirExists ? "" : "(missing)"}</p>
          {data.artifacts.length > 0 && (
            <ul className="file-list">
              {data.artifacts.map((a) => <li key={a}>{a}</li>)}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
