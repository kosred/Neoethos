import { useEffect, useState } from "react";
import { settingsRaw, saveSettingsRaw, knobCatalog, diagnosticsReport, dataImport } from "../api";
import { usePoll } from "../hooks";

export default function Advanced() {
  const { data: catalog } = usePoll(knobCatalog, 0);
  const [yaml, setYaml] = useState("");
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  // data import
  const [src, setSrc] = useState("");
  const [iSym, setISym] = useState("EURUSD");
  const [iTf, setITf] = useState("H1");

  useEffect(() => {
    settingsRaw()
      .then((r: any) => { setYaml(r?.yaml ?? ""); setPath(r?.path ?? ""); })
      .catch((e) => setMsg(String(e)));
  }, []);

  const save = async () => {
    setBusy(true);
    setMsg("Saving config.yaml…");
    try {
      await saveSettingsRaw(yaml);
      setMsg("✓ config.yaml saved (written verbatim).");
    } catch (e) {
      setMsg(`Save failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const runDiag = async () => {
    setBusy(true);
    setMsg("Running diagnostics…");
    try {
      const r: any = await diagnosticsReport();
      setMsg(`✓ Diagnostics: ${typeof r === "string" ? r : JSON.stringify(r).slice(0, 300)}`);
    } catch (e) {
      setMsg(`Diagnostics failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const doImport = async () => {
    setBusy(true);
    setMsg("Importing…");
    try {
      const r: any = await dataImport(src, iSym.toUpperCase(), iTf.toUpperCase());
      setMsg(`✓ Imported → ${r?.writtenPath ?? r?.written_path ?? "done"}`);
    } catch (e) {
      setMsg(`Import failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  // group knobs by section
  const knobs: any[] = catalog?.knobs ?? [];
  const sections = Array.from(new Set(knobs.map((k) => k.section)));

  return (
    <div className="screen">
      <h1>Advanced</h1>
      <p className="sub">Raw config · knob catalog · diagnostics · data import</p>
      {msg && <div className="banner info">{msg}</div>}

      <div className="btn-row">
        <button onClick={runDiag} disabled={busy}>Run diagnostics</button>
      </div>

      <h2>Import data file</h2>
      <div className="ticket">
        <div className="ticket-row">
          <label style={{ flex: 1 }}>Source path<input value={src} onChange={(e) => setSrc(e.target.value)} placeholder="C:\path\to\EURUSD.csv" style={{ width: "100%" }} /></label>
          <label>Symbol<input value={iSym} onChange={(e) => setISym(e.target.value)} style={{ width: 90 }} /></label>
          <label>TF<input value={iTf} onChange={(e) => setITf(e.target.value)} style={{ width: 70 }} /></label>
          <button className="primary" disabled={busy || !src} onClick={doImport}>Import</button>
        </div>
      </div>

      <h2>config.yaml</h2>
      <p className="muted small">{path}</p>
      <textarea
        className="yaml-editor"
        value={yaml}
        onChange={(e) => setYaml(e.target.value)}
        spellCheck={false}
      />
      <div className="btn-row">
        <button className="primary" disabled={busy} onClick={save}>Save config.yaml</button>
      </div>

      <h2>Knob catalog ({knobs.length})</h2>
      {sections.map((sec) => (
        <details key={sec} className="knob-section">
          <summary>{sec}</summary>
          <table className="tbl">
            <thead><tr><th>Knob</th><th>Current</th><th>Default</th><th>Help</th></tr></thead>
            <tbody>
              {knobs.filter((k) => k.section === sec).map((k) => (
                <tr key={k.id}>
                  <td title={k.id}>{k.label}</td>
                  <td><b>{k.current}</b></td>
                  <td className="muted">{k.default}</td>
                  <td className="muted small">{k.helpShort}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      ))}
    </div>
  );
}
