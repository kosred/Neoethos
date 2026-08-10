import { useEffect, useState } from "react";
import {
  settings, updateSettings, settingsRaw, saveSettingsRaw, knobCatalog, diagnosticsReport, riskInfo,
  federationStatus, federationSetJobs, federationWorkerStart, federationWorkerStop, swarmCapacity,
  meshStatus, meshSetEnabled,
  type FedStatus, type SwarmCapacity, type MeshStatus,
} from "../api";
import { usePoll } from "../hooks";
import { HelpPanel, HelpStep, Tip } from "../components/Help";

// Federation Phase 0 — share compute with other NeoEthos users, no server:
// one instance plays COORDINATOR (sets a work plan, receives results); any
// number of WORKERS point at its URL and contribute their cores.
function FederationPanel() {
  const { data: fed, reload } = usePoll<FedStatus>(federationStatus, 15000);
  const { data: swarm } = usePoll<SwarmCapacity>(swarmCapacity, 15000);
  const { data: mesh, reload: reloadMesh } = usePoll<MeshStatus>(meshStatus, 10000);

  const toggleMesh = async () => {
    setBusy(true);
    try {
      const s = await meshSetEnabled(!mesh?.enabled);
      setMsg(
        s.enabled
          ? "✓ Mesh ON — this machine joins the swarm and pools compute with your other nodes."
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
            <button className={mesh?.enabled ? "danger" : "primary"} disabled={busy} onClick={toggleMesh}>
              {mesh?.enabled ? "Turn mesh OFF" : "Turn mesh ON"}
            </button>
            <div className="muted small" style={{ marginTop: 4 }}>
              {mesh?.enabled
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
            Aggregated by the P2P mesh sidecar. Each node stays never-OOM (memory capped to its own hardware);
            more nodes = broader search the app can scale into.
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
            <input type="text" value={token} onChange={(e) => setToken(e.target.value)} style={{ width: 160 }} />
          </label>
          <button className="primary" disabled={busy} onClick={publishJobs}>Publish work plan</button>
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
            : <button className="primary" disabled={busy} onClick={startWorker}>Start worker</button>}
        </div>
        {fed?.workerStatus && <p className="muted small" style={{ marginTop: 6 }}>{fed.workerStatus}</p>}
      </div>
    </div>
  );
}

// Form-driven config editor over the SAFE, typed /settings DTO — no raw-YAML
// hand-editing needed for the common knobs. Each field maps to a DTO key
// (camelCase) that update_settings validates + clamps server-side.
type Opt = { v: string; l: string };
type Field = {
  key: string;
  label: string;
  kind: "num" | "text" | "bool" | "enum";
  help: string;
  options?: Opt[];
  pct?: boolean; // stored as 0..1 fraction, shown/edited as %
  step?: number;
};
type Group = { title: string; fields: Field[] };

// Render a knob-catalog entry's TYPE the way the backend describes it. The
// wire shape is flat: `{ kind: "Int", min, max }`, `{ kind: "Enum",
// enumChoices: [...] }`, `{ kind: "Bool" }`, … (`knob_catalog.rs:139-175`).
// Audit #113: every one of these fields was already on the wire and the table
// showed none of them, so a knob's legal range was invisible to the operator.
// ── The "Current" column must not print a compile-time constant as a live
// value ───────────────────────────────────────────────────────────────────
//
// 16 of the 52 catalog rows build their `current` from a string LITERAL in
// `knob_catalog.rs`, not from the runtime. `ctrader.max_attempts` renders "3"
// forever; set `app_runtime.ctrader_max_attempts: 5` and this screen still
// says 3, while the live value sits one function call away in
// `env_overrides::ctrader_max_attempts()`. A number under a heading that says
// "Current" is a claim about what the bot is doing right now, and for these
// rows the claim is false.
//
// The client cannot tell a literal from a genuine match on its own — the wire
// carries no "is this live?" flag — so the 16 are named here and rendered as
// what they are: the shipped value, unverified against the running process.
//
// ⚠ DELETE THIS LIST the moment `knob_catalog.rs` emits a `currentIsLive`
// boolean per row. It is a mirror of another file's defect and will rot.
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
  "risk.pnl_audit_drift_fraction",
  "risk.pnl_circuit_breaker_fraction",
  "risk.require_stop_loss",
  "log.rust_log",
  "log.log_dir",
  "server.bind_addr",
]);

function knobTypeLabel(k: any): string {
  const kind = String(k?.kind ?? "Text");
  if (kind === "Enum") return `enum: ${(k.enumChoices ?? []).join(" | ")}`;
  if (kind === "Int" || kind === "Float") {
    const lo = k.min ?? null;
    const hi = k.max ?? null;
    const range =
      lo != null && hi != null ? `${lo} … ${hi}` : lo != null ? `≥ ${lo}` : hi != null ? `≤ ${hi}` : "unbounded";
    return `${kind.toLowerCase()} (${range})`;
  }
  return kind.toLowerCase();
}

const GROUPS: Group[] = [
  {
    title: "Mode & risk",
    fields: [
      { key: "tradingMode", label: "Trading mode", kind: "enum", options: [{ v: "risky", l: "🚀 Risky (multiply)" }, { v: "prop_firm", l: "🛡 Prop-firm (robust)" }], help: "Risky = aggressive account-multiplication, drawdown-agnostic. Prop-firm = FTMO-style strict rules. Drives discovery ranking + risk orientation." },
      // 💰 The old help said "Percent of the account risked per trade". In
      // risky mode that is not what happens: live sizing does not read
      // risk.risk_per_trade at all (live_trading.rs:1664-1680 substitutes the
      // risky ladder), and the discovery search reads the risk BANDS, not this
      // field (grep '.risk_per_trade' across neoethos-search → zero hits). It
      // binds in prop-firm mode. Saying which mode honours it is the whole
      // difference between a setting and a decoration.
      { key: "riskPerTrade", label: "Risk per trade (%)", kind: "num", pct: true, step: 0.1, help: "PROP-FIRM MODE ONLY. Percent of the account risked per trade, clamped to the account's max risk on save. In RISKY mode live sizing IGNORES this field and uses the engine's own ladder (30–50%, capped by Max portfolio risk) — see the Risky Mode screen for the band actually in force. The discovery search never reads it in either mode; it samples from the risk bands." },
      // 💰 The old help said "entries pause once open positions risk ~5%".
      // It does not pause: live_trading.rs:1749-1770 computes
      // effective_risk = base_risk.min(remaining), so with nothing open the
      // FIRST entry is resized down to this cap. And 0 is not "off" in the
      // protective sense — it removes the cap entirely.
      { key: "maxPortfolioRisk", label: "Max portfolio risk (%)", kind: "num", pct: true, step: 0.5, help: "Ceiling on TOTAL concurrent risk across ALL running autopilot engines. It does NOT pause entries — it SIZES THEM DOWN: each entry is cut to whatever headroom is left, so with nothing open the first entry is capped at this number. 0 means NO CAP AT ALL, not 'no additional risk'. In risky mode this is the number that decides the first entry's size." },
      // Two axes, deliberately: enable_gpu_preference gates TRAINING;
      // models.prop_search_device REPLACES the global for the SEARCH whenever
      // it is non-empty (backend.rs:126-130). The refuters established these
      // cannot be merged — cpu training + gpu search is a real configuration.
      // So this control must not claim to choose "the" device.
      { key: "computeMode", label: "Compute (training)", kind: "enum", options: [{ v: "auto", l: "Auto" }, { v: "cpu", l: "CPU" }, { v: "gpu", l: "GPU" }], help: "TRAINING device preference (system.enable_gpu_preference). This does NOT decide the discovery search device: models.prop_search_device overrides it for the search whenever it is set, and both shipped config files set it — so choosing CPU here can still give you a GPU search. The device each run actually used is printed in the run's device-summary log line. prop_search_device is not editable from this screen; change it in the raw config.yaml below." },
    ],
  },
  {
    title: "Risky goal",
    fields: [
      { key: "riskyStartBalance", label: "Start balance", kind: "num", help: "Starting capital the risky goal-ranking compounds from." },
      { key: "riskyTargetBalance", label: "Target balance", kind: "num", help: "The goal the risky mode ranks strategies toward (fastest compounder wins)." },
      { key: "riskyHorizonDays", label: "Horizon (days)", kind: "num", help: "Time budget for reaching the target — used by the goal-based ranking." },
    ],
  },
  {
    title: "Discovery search",
    fields: [
      { key: "searchPopulation", label: "Population", kind: "num", help: "GA population size per generation. Bigger = wider search, slower." },
      { key: "searchPopulationAuto", label: "Population auto (GPU)", kind: "bool", help: "ON + NVIDIA card: raise the GA population to what the card fits in one launch (max 16384), logged at run start. SEARCHES MORE — different candidates, different results. Leave OFF until the population experiment settles that bigger finds better." },
      { key: "searchGenerations", label: "Generations", kind: "num", help: "Max GA generations (early-stop applies). Bigger = deeper search." },
      { key: "searchMaxHours", label: "Max hours", kind: "num", step: 0.5, help: "Wall-clock cap per (symbol, timeframe) unit before it advances to the next." },
      { key: "searchMaxIndicators", label: "Max indicators", kind: "num", help: "Max indicators a single gene may combine." },
      { key: "searchPortfolioSize", label: "Portfolio size", kind: "num", help: "How many surviving strategies to keep in the exported portfolio." },
      { key: "searchCorrThreshold", label: "Correlation cap", kind: "num", step: 0.01, help: "Prune strategies whose returns correlate above this (0..1) — keeps the portfolio diversified." },
      { key: "searchMaxRows", label: "Max rows (0=all)", kind: "num", help: "Cap the bars per unit. 0 = full history. Set (e.g. 600000) to make dense TFs (M3/M5) finish faster." },
    ],
  },
  {
    title: "Anti-stagnation (GA tuning)",
    fields: [
      { key: "prefilterTopK", label: "Indicator pool", kind: "num", help: "How many prefiltered indicators the GA may use. RAISE first if the search stalls — the #1 lever. Auto-capped at the number of available indicators + SMC." },
      { key: "convergencePatience", label: "Explore patience", kind: "num", help: "Flat generations before the GA gives up. Raise to search longer." },
      { key: "stagnationPatience", label: "Diversity kick", kind: "num", help: "Flat generations before heavier mutation + fresh genes kick in. Lower = reacts sooner." },
      { key: "noveltyWeight", label: "Novelty reward", kind: "num", step: 0.05, help: "0 = off. 0.1–0.3 rewards DIFFERENT genes → more market-regime variety." },
      { key: "disableSmcGate", label: "Disable SMC gate", kind: "bool", help: "Turn off the structural (SMC) gate if it over-constrains a pair." },
    ],
  },
  {
    title: "News gate",
    fields: [
      { key: "newsCalendarEnabled", label: "Calendar enabled", kind: "bool", help: "Pull the economic calendar to gate trading around high-impact events." },
      { key: "newsCalendarSource", label: "Calendar source", kind: "enum", options: [{ v: "forexfactory", l: "ForexFactory" }], help: "Economic-calendar provider. ForexFactory is the only one this build implements; the backend rejects anything else rather than silently serving ForexFactory under another name." },
      { key: "newsTradingMode", label: "Around news", kind: "enum", options: [{ v: "block_on_news", l: "Pause on news" }, { v: "allow_always", l: "Always allow" }, { v: "warn_only", l: "Warn only" }], help: "What automated trading does around high-impact events." },
    ],
  },
  {
    title: "Data",
    fields: [
      { key: "dataDir", label: "Data directory", kind: "text", help: "Where local price history + models live." },
      // The "Language" picker that used to sit here was removed: this app ships
      // no i18n layer, so `uiLocale` was validated, persisted and echoed back
      // while every string stayed English. A control that moves nothing is
      // worse than a missing one — it is indistinguishable from working.
    ],
  },
];

export default function Advanced() {
  const { data: catalog } = usePoll(knobCatalog, 0);
  const [form, setForm] = useState<Record<string, any>>({});
  const [yaml, setYaml] = useState("");
  const [path, setPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [showYaml, setShowYaml] = useState(false);

  const load = async () => {
    try {
      const s: any = await settings();
      let rpt: number | undefined;
      try { rpt = (await riskInfo()).riskPerTrade; } catch { /* optional */ }
      // Store pct fields as PERCENT in the form (fraction × 100); saveForm
      // converts back to a fraction exactly once.
      const form: Record<string, any> = { ...s };
      for (const g of GROUPS) {
        for (const f of g.fields) {
          if (f.pct && typeof form[f.key] === "number") form[f.key] = form[f.key] * 100;
        }
      }
      form.riskPerTrade = rpt != null ? rpt * 100 : undefined;
      setForm(form);
    } catch (e) {
      setMsg(String(e));
    }
  };
  useEffect(() => {
    load();
    settingsRaw().then((r: any) => { setYaml(r?.yaml ?? ""); setPath(r?.path ?? ""); }).catch(() => {});
  }, []);

  const setField = (k: string, v: any) => setForm((f) => ({ ...f, [k]: v }));

  const saveForm = async () => {
    setBusy(true);
    setMsg("Saving settings…");
    const payload: Record<string, any> = {};
    for (const g of GROUPS) {
      for (const f of g.fields) {
        const v = form[f.key];
        if (v === undefined || v === null || v === "") continue;
        payload[f.key] = f.pct ? Number(v) / 100 : v;
      }
    }
    try {
      await updateSettings(payload);
      setMsg("✓ Settings saved.");
      await load();
    } catch (e) {
      setMsg(`Save failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const saveYaml = async () => {
    setBusy(true);
    setMsg("Saving config.yaml…");
    try {
      await saveSettingsRaw(yaml);
      // "(verbatim)" used to be the whole claim, and the schema check behind
      // it accepted any misspelled key — `trailing_enabeld:` saved and
      // reported success. `Settings` now denies unknown fields, so the check
      // this message implies is finally the check that runs; say what it
      // actually covers rather than leaving the operator to assume.
      setMsg(
        "✓ config.yaml saved verbatim — YAML parsed, schema-checked (unknown keys and " +
          "wrong types are rejected), previous file backed up. A key the engine ignores " +
          "would have been refused, not saved.",
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
      const r: any = await diagnosticsReport();
      setMsg(`✓ Diagnostics: ${typeof r === "string" ? r : JSON.stringify(r).slice(0, 300)}`);
    } catch (e) {
      setMsg(`Diagnostics failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const knobs: any[] = catalog?.knobs ?? [];
  const sections = Array.from(new Set(knobs.map((k) => k.section)));

  const renderField = (f: Field) => {
    const raw = form[f.key];
    const val = raw ?? ""; // pct fields already hold the % value (see load)
    return (
      <label key={f.key} style={{ minWidth: 150 }}>
        <span>{f.label} <Tip text={f.help} /></span>
        {f.kind === "bool" ? (
          <input type="checkbox" checked={!!raw} onChange={(e) => setField(f.key, e.target.checked)} />
        ) : f.kind === "enum" ? (
          <select value={raw ?? ""} onChange={(e) => setField(f.key, e.target.value)}>
            {f.options!.map((o) => <option key={o.v} value={o.v}>{o.l}</option>)}
          </select>
        ) : f.kind === "num" ? (
          // Every numeric knob here is a non-negative quantity (a balance, a
          // count, a percentage). `min` blocks the spinner arrows and clamping
          // on change blocks typed/pasted negatives — a negative risk or
          // population is meaningless and must never reach the backend.
          <input
            type="number"
            min={0}
            step={f.step ?? 1}
            value={val}
            onChange={(e) =>
              setField(f.key, e.target.value === "" ? "" : Math.max(0, Number(e.target.value)))
            }
            style={{ width: 110 }}
          />
        ) : (
          <input type="text" value={raw ?? ""} onChange={(e) => setField(f.key, e.target.value)} style={{ width: 180 }} />
        )}
      </label>
    );
  };

  return (
    <div className="screen">
      <h1>Advanced</h1>
      {/* Audit #113/#114: the old subtitle claimed "Every engine setting as a
          form — no raw YAML needed". That was false by two orders of magnitude
          — the form below carries ~24 controls against ~390 knobs, and the
          catalog is read-only. A UI that overstates its own coverage is how the
          operator concludes a knob he set is in force when it never was. */}
      <p className="sub">The common engine settings as a form · full knob catalog (read-only) · diagnostics · raw YAML fallback</p>

      <HelpPanel id="advanced">
        <p>Power-user configuration. The common knobs are grouped below as friendly controls (each writes <code>config.yaml</code> safely, validated + clamped by the backend). The raw YAML editor + full knob catalog are kept as a fallback.</p>
        <HelpStep n={1}>Edit any field, then <b>Save settings</b>. Hover the ⓘ next to a control for what it does.</HelpStep>
        <HelpStep n={2}><b>Diagnostics</b> runs a health report. The <b>knob catalog</b> documents every option (incl. ones not surfaced here).</HelpStep>
        <p className="muted small">Data import moved to <b>Discovery</b>; discovery mode/risk can also be set on the <b>Discovery</b> pre-flight.</p>
      </HelpPanel>

      {msg && <div className="banner info">{msg}</div>}

      <div className="btn-row">
        <button className="primary" disabled={busy} onClick={saveForm}>Save settings</button>
        <button onClick={runDiag} disabled={busy}>Run diagnostics</button>
      </div>

      {GROUPS.map((g) => (
        <div key={g.title}>
          <h2>{g.title}</h2>
          <div className="ticket">
            <div className="ticket-row" style={{ flexWrap: "wrap", gap: 14 }}>
              {g.fields.map(renderField)}
            </div>
          </div>
        </div>
      ))}

      <FederationPanel />

      <h2>
        Raw config.yaml + knob catalog
        <button className="link" style={{ marginLeft: 10 }} onClick={() => setShowYaml((s) => !s)}>{showYaml ? "hide" : "show"}</button>
      </h2>
      {showYaml && (
        <>
          {/* The old text said "~200 long-tail knobs". The measured surface is
              390, of which this form reaches ~24 and the catalog documents 52.
              Understating the gap is how an operator concludes the form is
              nearly complete and stops looking in the YAML for the knob that
              actually decided his run. */}
          <p className="muted small">
            {path} — the ONLY write path for the ~340 knobs neither the form above nor the
            catalog below covers (390 knobs exist; the form reaches ~24, the catalog documents{" "}
            {knobs.length}).
          </p>
          <textarea className="yaml-editor" value={yaml} onChange={(e) => setYaml(e.target.value)} spellCheck={false} />
          <div className="btn-row"><button className="primary" disabled={busy} onClick={saveYaml}>Save config.yaml</button></div>

          <h2>Knob catalog ({knobs.length})</h2>
          {/* Audit #113/#114. The backend has always emitted the full widget
              schema for every knob — `kind`, `min`, `max`, `enumChoices`,
              `helpLong`, `envVar` and three preset values, flattened to
              top-level JSON keys (`knob_catalog.rs:85-100`). This table rendered
              exactly four of them and threw the rest away, so the operator could
              not even SEE a knob's legal range.
              It now shows the whole schema. It is still READ-ONLY, and that is
              a backend gap, not a UI choice: THERE IS NO WRITE ENDPOINT. The
              catalog module says so in its own doc — "Write path (future):
              POST /settings/knobs will write the operator's changes to
              config.yaml" (`knob_catalog.rs:34-38`) — and the ids here
              (`ga.seed`, `cost.spread_pips`) are catalog names, not config.yaml
              paths, so no client can map them to a key on its own. Rendering
              editable inputs against a missing endpoint would produce 53
              controls that move nothing, which is the exact failure this
              codebase keeps repeating (see the deleted Language picker above).
              The raw YAML editor is the honest write path until the endpoint
              lands. DO NOT make these inputs before then. */}
          <div className="banner info">
            <b>Read-only.</b> These {knobs.length} knobs are documented here with their type, legal
            range and presets, but the backend has no knob write endpoint yet
            (<code>POST /settings/knobs</code>). Change them in the raw <code>config.yaml</code>{" "}
            editor above, or use the typed form at the top of this screen for the common ones.
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
                    <th>Conservative</th>
                    <th>Balanced</th>
                    <th>Aggressive</th>
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
