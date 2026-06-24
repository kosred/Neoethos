import { storagePaths, openPath, type StorageEntry } from "../api";
import { usePoll } from "../hooks";

const human = (b: number) => {
  if (b <= 0) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0, v = b;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${u[i]}`;
};
const when = (ms: number | null) => (ms ? new Date(ms).toLocaleString() : "—");

const KIND_ICON: Record<string, string> = {
  data: "🗄", models: "🧠", strategies: "⚗", journal: "📒",
  logs: "📜", config: "⚙", secret: "🔑", cache: "♻",
};

export default function Files() {
  const { data, error, reload } = usePoll(storagePaths, 0);

  return (
    <div className="screen">
      <h1>Files &amp; Storage</h1>
      <p className="sub">Exactly where everything the app downloads, trains, or logs is kept — click to open</p>

      <div className="btn-row">
        <button onClick={reload}>Refresh</button>
      </div>
      {error && <div className="banner warn">{error}</div>}

      <table className="tbl">
        <thead>
          <tr><th></th><th>What</th><th>Path</th><th>Size</th><th>Items</th><th>Modified</th><th></th></tr>
        </thead>
        <tbody>
          {(data?.entries ?? []).map((e: StorageEntry) => (
            <tr key={e.key}>
              <td>{KIND_ICON[e.kind] ?? "📁"}</td>
              <td><b>{e.label}</b>{!e.exists && <span className="sell small"> (missing)</span>}</td>
              <td style={{ fontFamily: "monospace", fontSize: 11, color: "#9ca3af", maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={e.path}>{e.path}</td>
              <td>{e.kind === "secret" ? "—" : human(e.sizeBytes)}</td>
              <td>{e.itemCount || "—"}</td>
              <td className="muted small">{when(e.lastModifiedMs)}</td>
              <td>
                <button disabled={!e.exists} onClick={() => openPath(e.path).catch(() => {})}>Open</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 10 }}>
        Downloaded market data lands in <b>data</b>; imported files too. Discovered strategies + trained
        models live in <b>cache</b> / <b>models</b>. Secrets show the path only.
      </p>
    </div>
  );
}
