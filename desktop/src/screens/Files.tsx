import { useState } from "react";
import { storagePaths, openPath, type StorageEntry } from "../api";
import { usePoll } from "../hooks";
import { HelpPanel } from "../components/Help";

const human = (b: number) => {
  if (!Number.isFinite(b) || b < 0) return "—";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0, v = b;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${u[i]}`;
};
const when = (ms: number | null) => (ms ? new Date(ms).toLocaleString() : "—");

const KIND_ICON: Record<string, string> = {
  data: "🗄", models: "🧠", journal: "📒",
  logs: "📜", config: "⚙", secret: "🔑", cache: "♻",
};

export default function Files() {
  const { data, error, loading, reload } = usePoll(storagePaths, 0);
  const [openError, setOpenError] = useState("");
  const [opening, setOpening] = useState<string | null>(null);
  const reveal = async (path: string) => {
    if (opening) return;
    setOpening(path);
    setOpenError("");
    try {
      await openPath(path);
    } catch (reason) {
      setOpenError(`Could not open ${path}: ${reason}`);
    } finally {
      setOpening(null);
    }
  };

  return (
    <div className="screen">
      <h1>Files &amp; Storage</h1>
      <p className="sub">Configured storage locations — bounded metadata snapshots, with incomplete reads marked</p>

      <HelpPanel id="files">
        <p>Each row shows an absolute configured location. Size totals regular-file bytes; Items counts immediate children (or one file), not strategies or datasets. Links and reparse points are skipped. Partial values are observed lower bounds, not complete totals.</p>
        <p>Press <b>Open</b> on any row to reveal it in Windows Explorer: <b>data</b> = downloaded price history, <b>models</b> = trained AI, <b>cache</b> = discovered strategies, <b>journal</b> = closed trades, <b>logs</b> = diagnostics, <b>config</b> = your settings file. Secrets (broker credentials) show the path only — never the contents.</p>
      </HelpPanel>

      <div className="btn-row">
        <button disabled={loading} onClick={() => void reload()}>{loading ? "Refreshing…" : "Refresh"}</button>
      </div>
      {error && <div className="banner warn">{error}</div>}
      {openError && <div className="banner warn" role="alert">{openError}</div>}
      {!data && !error && <p className="muted" role="status">Reading storage paths…</p>}

      <table className="tbl">
        <thead>
          <tr><th></th><th>What</th><th>Path</th><th>Size</th><th>Items</th><th>Observed modified</th><th></th></tr>
        </thead>
        <tbody>
          {((error ? undefined : data)?.entries ?? []).map((e: StorageEntry) => {
            const measured = e.scanStatus === "complete" || e.scanStatus === "partial";
            const prefix = e.scanStatus === "partial" ? "Observed " : "";
            return (
              <tr key={e.key}>
                <td>{KIND_ICON[e.kind] ?? "📁"}</td>
                <td><b>{e.label}</b>{e.scanStatus !== "complete" && <span className="sell small"> ({e.scanStatus ?? "unavailable"})</span>}
                  {e.scanError && <div className="muted small">{e.scanError}</div>}</td>
                <td style={{ fontFamily: "monospace", fontSize: 11, color: "#9ca3af", maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={e.path}>{e.path}</td>
                <td>{e.kind === "secret" || !measured ? "—" : `${prefix}${human(e.sizeBytes)}`}</td>
                <td>{e.kind === "secret" || !measured ? "—" : `${prefix}${e.itemCount}`}</td>
                <td className="muted small">{e.kind === "secret" || !measured ? "—" : when(e.lastModifiedMs)}</td>
                <td>
                  <button disabled={!e.exists || !measured || opening !== null} onClick={() => void reveal(e.path)}>{opening === e.path ? "Opening…" : "Open"}</button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="muted small" style={{ marginTop: 10 }}>
        Downloaded market data lands in <b>data</b>; imported files too. Discovered strategies + trained
        models live in <b>cache</b> / <b>models</b>. Secrets show the path only.
      </p>
    </div>
  );
}
