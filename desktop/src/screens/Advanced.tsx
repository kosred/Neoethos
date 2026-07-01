import { useEffect, useState } from "react";
import { settings, updateSettings, settingsRaw, saveSettingsRaw, knobCatalog, diagnosticsReport, riskInfo } from "../api";
import { usePoll } from "../hooks";
import { HelpPanel, HelpStep, Tip } from "../components/Help";

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

const GROUPS: Group[] = [
  {
    title: "Mode & risk",
    fields: [
      { key: "tradingMode", label: "Trading mode", kind: "enum", options: [{ v: "risky", l: "🚀 Risky (multiply)" }, { v: "prop_firm", l: "🛡 Prop-firm (robust)" }], help: "Risky = aggressive account-multiplication, drawdown-agnostic. Prop-firm = FTMO-style strict rules. Drives discovery ranking + risk orientation." },
      { key: "riskPerTrade", label: "Risk per trade (%)", kind: "num", pct: true, step: 0.1, help: "Percent of the account risked per trade (position sizing). Clamped to the account's max risk on save." },
      { key: "maxPortfolioRisk", label: "Max portfolio risk (%)", kind: "num", pct: true, step: 0.5, help: "Cap on TOTAL concurrent risk across ALL running autopilot engines (e.g. 5% = entries pause once open positions already risk ~5% of the balance). 0 = off. Protects a small account when many strategies run at once." },
      { key: "computeMode", label: "Compute", kind: "enum", options: [{ v: "auto", l: "Auto" }, { v: "cpu", l: "CPU" }, { v: "gpu", l: "GPU" }], help: "Which hardware discovery/training prefers. Auto detects; GPU can OOM on a shared-RAM iGPU." },
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
      { key: "prefilterTopK", label: "Indicator pool", kind: "num", help: "How many prefiltered indicators the GA may use. RAISE first if the search stalls — the #1 lever." },
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
      { key: "newsCalendarSource", label: "Calendar source", kind: "text", help: "Calendar provider id (e.g. forexfactory)." },
      { key: "newsTradingMode", label: "Around news", kind: "enum", options: [{ v: "block_on_news", l: "Pause on news" }, { v: "allow_always", l: "Always allow" }, { v: "warn_only", l: "Warn only" }], help: "What automated trading does around high-impact events." },
    ],
  },
  {
    title: "Data & locale",
    fields: [
      { key: "dataDir", label: "Data directory", kind: "text", help: "Where local price history + models live." },
      { key: "uiLocale", label: "Language", kind: "enum", options: [{ v: "en", l: "English" }, { v: "el", l: "Ελληνικά" }], help: "UI language." },
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
      setMsg("✓ config.yaml saved (verbatim).");
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
          <input type="number" step={f.step ?? 1} value={val} onChange={(e) => setField(f.key, e.target.value === "" ? "" : Number(e.target.value))} style={{ width: 110 }} />
        ) : (
          <input type="text" value={raw ?? ""} onChange={(e) => setField(f.key, e.target.value)} style={{ width: 180 }} />
        )}
      </label>
    );
  };

  return (
    <div className="screen">
      <h1>Advanced</h1>
      <p className="sub">Every engine setting as a form — no raw YAML needed · diagnostics · raw fallback</p>

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

      <h2>
        Raw config.yaml + knob catalog
        <button className="link" style={{ marginLeft: 10 }} onClick={() => setShowYaml((s) => !s)}>{showYaml ? "hide" : "show"}</button>
      </h2>
      {showYaml && (
        <>
          <p className="muted small">{path} — power-user fallback for the ~200 long-tail knobs not in the form above.</p>
          <textarea className="yaml-editor" value={yaml} onChange={(e) => setYaml(e.target.value)} spellCheck={false} />
          <div className="btn-row"><button className="primary" disabled={busy} onClick={saveYaml}>Save config.yaml</button></div>

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
        </>
      )}
    </div>
  );
}
