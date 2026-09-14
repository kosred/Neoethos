import { useEffect, useState } from "react";
import {
  settingsRaw, saveSettingsRaw, knobCatalog, diagnosticsReport,
  federationStatus, federationSetJobs, federationWorkerStart, federationWorkerStop, swarmCapacity,
  meshStatus, meshSetEnabled,
  type FedStatus, type SwarmCapacity, type MeshStatus, type KnobEntry,
} from "../api";
import { usePoll } from "../hooks";
import { HelpPanel, HelpStep } from "../components/Help";

// Federation Phase 0 — share compute with other NeoEthos users, no server:
// one instance plays COORDINATOR (sets a work plan, receives results); any
// number of WORKERS point at its URL and contribute their cores.
function FederationPanel() {
  const { data: fed, error: fedError, reload } = usePoll<FedStatus>(federationStatus, 15000);
  const { data: swarm, error: swarmError } = usePoll<SwarmCapacity>(swarmCapacity, 15000);
  const { data: mesh, error: meshError, reload: reloadMesh } = usePoll<MeshStatus>(meshStatus, 10000);

  const toggleMesh = async () => {
    if (busy || !mesh || meshError) return;
    setBusy(true);
    try {
      const s = await meshSetEnabled(!mesh?.enabled);
      setMsg(
        s.enabled
          ? "Mesh enabled. Check process and peer status below; this does not prove distributed work has run."
          : "Mesh OFF — this machine left the swarm.",
      );
      await reloadMesh();
    } catch (e) { setMsg(`Mesh toggle failed: ${e}`); } finally { setBusy(false); }
  };
  const [combosText, setCombosText] = useState("EURUSD M15\nGBPUSD M15\nUSDJPY H1");
  const [token, setToken] = useState("");
  const [coordUrl, setCoordUrl] = useState("");
  const [workerId, setWorkerId] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const publishJobs = async () => {
    const combos = combosText
      .split("\n")
      .map((l) => l.trim().split(/[\s,]+/))
      .filter((p) => p.length >= 2)
      .map(([symbol, baseTf]) => ({ symbol, baseTf }));
    if (combos.length === 0) { setMsg("Write one combo per line, e.g. EURUSD M15"); return; }
    setBusy(true);
    try {
      const r = await federationSetJobs(combos, token.trim() || undefined);
      setMsg(`✓ Work plan published — ${r.queued} combos queued for workers.`);
      await reload();
    } catch (e) { setMsg(`Publish failed: ${e}`); } finally { setBusy(false); }
  };

  const startWorker = async () => {
    if (!coordUrl.trim()) { setMsg("Enter the coordinator URL first (e.g. http://100.x.y.z:PORT)."); return; }
    setBusy(true);
    try {
      await federationWorkerStart(coordUrl.trim(), workerId.trim() || undefined, token.trim() || undefined);
      setMsg("✓ Worker started — this machine now contributes its cores.");
      await reload();
    } catch (e) { setMsg(`Worker start failed: ${e}`); } finally { setBusy(false); }
  };

  const stopWorker = async () => {
    setBusy(true);
    try { await federationWorkerStop(); setMsg("Worker stopping…"); await reload(); }
    catch (e) { setMsg(`Stop failed: ${e}`); } finally { setBusy(false); }
  };

  return (
    <div>
      <h2>Federation <span className="badge demo">PHASE 0</span></h2>
      <p className="muted small">
        SETI@home for strategy discovery — share compute with people you trust, no server needed.
        One instance is the <b>coordinator</b> (publishes a work plan below and receives results into
        <code> cache/federation_inbox</code> — they appear in the normal strategy list and still pass every
        local gate before any real money). Others run as <b>workers</b>: they fetch a combo, run their own
        Discovery on it, and send the result back. Expose the coordinator with Tailscale / port-forward;
        set a shared token so only your group can submit.
      </p>
      {msg && <div className="banner info">{msg}</div>}
      {fedError && <div className="banner warn" role="alert">Federation state unavailable: {fedError}</div>}
      {swarmError && <div className="banner warn" role="alert">Swarm capacity unavailable: {swarmError}</div>}
      {meshError && <div className="banner warn" role="alert">Mesh state unavailable: {meshError}</div>}

      <div className="ticket" style={{ borderColor: mesh?.running ? "#295c3a" : undefined }}>
        <div className="ticket-row" style={{ alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 12 }}>
          <div>
            <b>🌐 Mesh — pool your computers as one</b>
            <div className="muted small" style={{ marginTop: 4, maxWidth: 560 }}>
              Turns this machine into a swarm node: it discovers your other NeoEthos
              machines automatically (no server, no port-forwarding) and pools their
              CPUs so discovery covers more ground in the same time. Off by default —
              pooling compute over the internet is your choice.
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <button className={mesh?.enabled ? "danger" : "primary"} disabled={busy || !mesh || Boolean(meshError)} onClick={toggleMesh}>
              {mesh?.enabled ? "Turn mesh OFF" : "Turn mesh ON"}
            </button>
            <div className="muted small" style={{ marginTop: 4 }}>
              {!mesh || meshError ? "State unknown" : mesh.enabled
                ? (mesh?.running ? "● running" : "● enabled (starting…)")
                : "○ off"}
            </div>
          </div>
        </div>
      </div>

      {swarm?.running && (
        <div className="ticket" style={{ borderColor: "#295c3a", background: "#0e1a12" }}>
          <b>🖥 Your swarm — the network as one machine</b>
          <div className="cards" style={{ gridTemplateColumns: "repeat(4, 1fr)", marginTop: 8 }}>
            <div className="card"><div className="card-label">Nodes</div><div className="card-value">{swarm.nodes}</div></div>
            <div className="card"><div className="card-label">Total cores</div><div className="card-value" style={{ color: "#4ade80" }}>{swarm.totalCores}</div></div>
            <div className="card"><div className="card-label">Total RAM</div><div className="card-value">{swarm.totalRamGb ? `${swarm.totalRamGb.toFixed(0)} GB` : "—"}</div></div>
            <div className="card"><div className="card-label">GPUs</div><div className="card-value">{swarm.totalGpus ?? 0}</div></div>
          </div>
          <p className="muted small" style={{ marginTop: 6 }}>
            Capacity reported by the P2P mesh sidecar. Each job still needs per-node memory admission;
            aggregated capacity is not proof that a distributed search completed.
          </p>
        </div>
      )}

      <div className="ticket">
        <b>Coordinator — publish a work plan</b>
        <div className="ticket-row" style={{ alignItems: "flex-end", flexWrap: "wrap", gap: 12 }}>
          <label>
            Combos (one per line: SYMBOL TF)
            <textarea value={combosText} onChange={(e) => setCombosText(e.target.value)} spellCheck={false}
              style={{ minWidth: 240, minHeight: 70, fontFamily: "inherit", fontSize: 13 }} />
          </label>
          <label>Shared token (optional)
            <input type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} style={{ width: 160 }} />
          </label>
          <button className="primary" disabled={busy || !fed || Boolean(fedError)} onClick={publishJobs}>Publish work plan</button>
        </div>
        {fed && (
          <p className="muted small" style={{ marginTop: 6 }}>
            Queue: <b>{fed.jobsQueued}</b> · leased: <b>{fed.leases.length}</b> · received: <b>{fed.received.length}</b>
            {fed.tokenRequired ? " · token required" : " · open (no token)"}
          </p>
        )}
        {fed && fed.received.length > 0 && (
          <table className="tbl">
            <thead><tr><th>When</th><th>Worker</th><th>Combo</th><th>Saved</th></tr></thead>
            <tbody>
              {fed.received.slice(0, 10).map((r, i) => (
                <tr key={i}>
                  <td className="muted small">{new Date(r.receivedAtUnixMs).toLocaleString()}</td>
                  <td>{r.worker}</td>
                  <td>{r.symbol} {r.baseTf}</td>
                  <td className="muted small" style={{ maxWidth: 320, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={r.savedPath}>{r.savedPath.split(/[\\/]/).pop()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="ticket" style={{ marginTop: 10 }}>
        <b>Worker — contribute this machine {fed?.workerRunning && <span className="badge live">RUNNING</span>}</b>
        <div className="ticket-row" style={{ alignItems: "flex-end", flexWrap: "wrap", gap: 12 }}>
          <label>Coordinator URL
            <input type="text" placeholder="http://100.x.y.z:PORT" value={coordUrl} onChange={(e) => setCoordUrl(e.target.value)} style={{ width: 230 }} />
          </label>
          <label>Worker name (optional)
            <input type="text" placeholder="konstantinos-minipc" value={workerId} onChange={(e) => setWorkerId(e.target.value)} style={{ width: 170 }} />
          </label>
          {fed?.workerRunning
            ? <button className="danger" disabled={busy} onClick={stopWorker}>Stop worker</button>
            : <button className="primary" disabled={busy || !fed || Boolean(fedError)} onClick={startWorker}>Start worker</button>}
        </div>
        {fed?.workerStatus && <p className="muted small" style={{ marginTop: 6 }}>{fed.workerStatus}</p>}
      </div>
    </div>
  );
}

// Temporary client-side truth marker until the catalog reports this itself.
// Delete this list when each row includes a backend-owned `currentIsLive`.
const CURRENT_IS_A_SHIPPED_LITERAL = new Set<string>([
  "ctrader.read_timeout_secs",
  "ctrader.max_attempts",
  "ctrader.backoff_base_ms",
  "ctrader.allow_partial_fill",
  "ctrader.chart_merge_side",
  "ctrader.stream_max_attempts",
  "ctrader.stream_backoff_base_ms",
  "paths.symbol_metadata_override",
  "paths.user_data_dir_override",
  "risk.prop_firm_preset",
  "risk.require_stop_loss",
  "log.rust_log",
  "log.log_dir",
  "server.bind_addr",
]);

function knobTypeLabel(knob: KnobEntry): string {
  if (knob.kind === "Enum") return `enum: ${(knob.enumChoices ?? []).join(" | ")}`;
  if (knob.kind === "Int" || knob.kind === "Float") {
    const lo = knob.min ?? null;
    const hi = knob.max ?? null;
    const range =
      lo != null && hi != null ? `${lo} … ${hi}` : lo != null ? `≥ ${lo}` : hi != null ? `≤ ${hi}` : "unbounded";
    return `${knob.kind.toLowerCase()} (${range})`;
  }
  return knob.kind.toLowerCase();
}

export default function Advanced() {
  const { data: catalog, error: catalogError } = usePoll(knobCatalog, 0);
  const [yaml, setYaml] = useState("");
  const [path, setPath] = useState("");
  const [rawLoaded, setRawLoaded] = useState(false);
  const [rawError, setRawError] = useState("");
  const [loadVersion, setLoadVersion] = useState(0);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [showYaml, setShowYaml] = useState(false);

  useEffect(() => {
    let active = true;
    void settingsRaw()
      .then((result) => {
        if (!active) return;
        setYaml(result.yaml);
        setPath(result.path);
        setRawLoaded(true);
        setRawError("");
      })
      .catch((error: unknown) => {
        if (active) setRawError(`Could not read config.yaml: ${String(error)}`);
      });
    return () => { active = false; };
  }, [loadVersion]);

  const saveYaml = async () => {
    if (busy || !rawLoaded) return;
    setBusy(true);
    setMsg("Saving config.yaml…");
    try {
      const result = await saveSettingsRaw(yaml);
      setMsg(
        `✓ config.yaml saved after schema validation. Previous file: ${result.backupPath}`,
      );
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
      const result = await diagnosticsReport();
      setMsg(`✓ Diagnostic bundle ready: ${result.zipPath} (${result.totalBytes.toLocaleString()} bytes)`);
    } catch (e) {
      setMsg(`Diagnostics failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const knobs = catalog?.knobs ?? [];
  const sections = Array.from(new Set(knobs.map((knob) => knob.section)));

  return (
    <div className="screen">
      <h1>Advanced</h1>
      <p className="sub">Diagnostics, distributed compute and the complete configuration source</p>

      <HelpPanel id="advanced">
        <p>This screen is intentionally not a second settings form.</p>
        <HelpStep n={1}>Search objectives, search risk and GA/SMC breadth live only in <b>Research → Strategy search</b>.</HelpStep>
        <HelpStep n={2}>Broker, training and live-trading safeguards live in <b>General</b>.</HelpStep>
        <HelpStep n={3}>Use the raw YAML only for knobs that still lack a typed control. The backend rejects unknown keys and wrong types before replacing the file.</HelpStep>
      </HelpPanel>

      {msg && <div className="banner info">{msg}</div>}
      {catalogError && <div className="banner warn">Could not load knob catalog: {catalogError}</div>}
      {rawError && <div className="banner warn" role="alert">{rawError} <button onClick={() => setLoadVersion((value) => value + 1)}>Retry loading config</button></div>}

      <div className="btn-row">
        <button onClick={runDiag} disabled={busy}>Run diagnostics</button>
      </div>

      <FederationPanel />

      <h2>
        Raw config.yaml + knob catalog
        <button className="link" style={{ marginLeft: 10 }} onClick={() => setShowYaml((s) => !s)}>{showYaml ? "hide" : "show"}</button>
      </h2>
      {showYaml && (
        <>
          <p className="muted small">
            {path || "Loading config path…"} — full source of runtime configuration. Search-related
            controls exposed by the typed API belong on Discovery and should not be edited twice here.
          </p>
          <textarea className="yaml-editor" disabled={busy || !rawLoaded} value={yaml} onChange={(e) => setYaml(e.target.value)} spellCheck={false} />
          <div className="btn-row"><button className="primary" disabled={busy || !rawLoaded} onClick={saveYaml}>Save config.yaml</button></div>

          <h2>Knob catalog ({knobs.length})</h2>
          <div className="banner info">
            <b>Read-only.</b> These {knobs.length} knobs are documented here with their type, legal
            range and advisory preset columns. The backend has no catalog write endpoint; use an
            existing typed control or edit the raw <code>config.yaml</code> above.
          </div>
          <div className="banner warn">
            <b>The Conservative / Balanced / Aggressive columns are advice, not a setting.</b>{" "}
            No endpoint applies those columns as a bundle. The live prop-firm preset in General is
            a different, backend-owned control.
          </div>
          <div className="banner warn">
            <b>
              {knobs.filter((k) => CURRENT_IS_A_SHIPPED_LITERAL.has(k.id)).length} of{" "}
              {knobs.length} rows do not read their “Current” value from the running process.
            </b>{" "}
            The backend builds those cells from a fixed string, so they show the shipped value
            forever — change the knob and the cell does not move. They are marked{" "}
            <span className="sell small">⚠ shipped value — not read live</span> in the table
            below. Every other row is a live reading. This is a backend gap
            (<code>knob_catalog.rs</code>), not a display choice, and marking it is the honest
            stand-in until those rows read the runtime.
          </div>
          {sections.map((sec) => (
            <details key={sec} className="knob-section">
              <summary>{sec}</summary>
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Knob</th>
                    <th>Type / range</th>
                    <th>Current<div className="muted small" style={{ fontWeight: 400 }}>live unless marked</div></th>
                    <th>Default</th>
                    <th>Conservative<div className="muted small" style={{ fontWeight: 400 }}>advice only</div></th>
                    <th>Balanced<div className="muted small" style={{ fontWeight: 400 }}>advice only</div></th>
                    <th>Aggressive<div className="muted small" style={{ fontWeight: 400 }}>advice only</div></th>
                    <th>Help</th>
                  </tr>
                </thead>
                <tbody>
                  {knobs.filter((k) => k.section === sec).map((k) => (
                    <tr key={k.id}>
                      <td title={k.id}>
                        {k.label}
                        <div className="muted small"><code>{k.id}</code></div>
                      </td>
                      <td className="muted small">{knobTypeLabel(k)}</td>
                      {CURRENT_IS_A_SHIPPED_LITERAL.has(k.id) ? (
                        <td
                          title={
                            "The backend serves this cell as a fixed string, not a reading from the " +
                            "running process. If you changed this knob, THIS NUMBER WILL NOT MOVE — " +
                            "check config.yaml, not here."
                          }
                        >
                          <span className="muted">{k.current}</span>
                          <div className="sell small">⚠ shipped value — not read live</div>
                        </td>
                      ) : (
                        <td><b>{k.current}</b></td>
                      )}
                      <td className="muted">{k.default}</td>
                      <td className="muted small">{k.presetConservative || "—"}</td>
                      <td className="muted small">{k.presetBalanced || "—"}</td>
                      <td className="muted small">{k.presetAggressive || "—"}</td>
                      <td className="muted small" title={k.helpLong}>{k.helpShort}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          ))}
        </>
      )}
    </div>
  );
}
