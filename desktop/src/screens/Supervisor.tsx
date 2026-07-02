import { useEffect, useState } from "react";
import {
  supervisorStatus,
  supervisorConfig,
  supervisorTick,
  supervisorChat,
  experienceTrain,
  type SupervisorLogEntry,
  type ExperienceGroup,
} from "../api";
import { usePoll } from "../hooks";
import { HelpPanel, HelpStep, Tip } from "../components/Help";

const fmtTime = (ms: number) => (ms > 0 ? new Date(ms).toLocaleString() : "—");

const KIND_BADGE: Record<string, { label: string; bg: string }> = {
  tick: { label: "TICK", bg: "#374151" },
  action: { label: "ACTION", bg: "#1d4ed8" },
  note: { label: "NOTE", bg: "#15803d" },
  chat: { label: "CHAT", bg: "#7c3aed" },
  error: { label: "ERROR", bg: "#b91c1c" },
};

export default function Supervisor() {
  const { data, error, reload } = usePoll(supervisorStatus, 10000);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  // Chat with the supervisor (same brain, same guard-rails, your steering).
  const [chatInput, setChatInput] = useState("");
  const [chatReply, setChatReply] = useState("");
  // Standing directives editor (one per line).
  const [directivesText, setDirectivesText] = useState("");
  const [directivesLoaded, setDirectivesLoaded] = useState(false);
  // Live-experience learnability report.
  const [expGroups, setExpGroups] = useState<ExperienceGroup[] | null>(null);
  const [expNote, setExpNote] = useState("");

  const cfg = data?.config;
  const log: SupervisorLogEntry[] = data?.log ?? [];

  useEffect(() => {
    if (cfg && !directivesLoaded) {
      setDirectivesText((cfg.directives ?? []).join("\n"));
      setDirectivesLoaded(true);
    }
  }, [cfg, directivesLoaded]);

  const sendChat = async () => {
    const message = chatInput.trim();
    if (!message) return;
    setBusy(true);
    setChatReply("…σκέφτεται (διαβάζει όλη την κατάσταση του συστήματος)…");
    try {
      const r = await supervisorChat(message);
      setChatReply(r.reply);
      setChatInput("");
      await reload();
    } catch (e) {
      setChatReply(`Failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const runExperienceTrain = async () => {
    setBusy(true);
    setMsg("Training from live experience (time-ordered OOS)…");
    try {
      const r = await experienceTrain();
      setExpGroups(r.groups);
      setExpNote(`${r.usableRecords}/${r.totalRecords} usable records · ${r.note}`);
      setMsg("✓ Experience report ready.");
    } catch (e) {
      setMsg(`Experience training failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const saveDirectives = async () => {
    setBusy(true);
    try {
      await supervisorConfig({ directives: directivesText.split("\n").map((s) => s.trim()).filter(Boolean) } as any);
      setMsg("✓ Directives saved — every future cycle follows them.");
      await reload();
    } catch (e) {
      setMsg(`Save failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const setCfg = async (patch: any) => {
    setBusy(true);
    try {
      await supervisorConfig(patch);
      await reload();
    } catch (e) {
      setMsg(`Config failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  const runNow = async () => {
    setBusy(true);
    setMsg("Running a supervisor cycle… (gathers state, asks the AI, executes)");
    try {
      const r = await supervisorTick();
      setMsg(`✓ ${r.summary}`);
      await reload();
    } catch (e) {
      setMsg(`Tick failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="screen">
      <h1>
        Supervisor{" "}
        {cfg && (
          <span className={`badge ${cfg.enabled ? "live" : "demo"}`}>
            {cfg.enabled ? `AUTONOMOUS · every ${cfg.intervalMinutes}m` : "PAUSED"}
          </span>
        )}
      </h1>
      <p className="sub">An AI co-pilot that watches engines, journal, autopilot & configs — and acts</p>

      <HelpPanel id="supervisor">
        <p>The Supervisor periodically feeds the whole system state (engines, live autopilot, 7-day journal stats, portfolios, blacklist, account) to <b>your ChatGPT sign-in</b> (AI Desk) and executes what it decides — with hard guard-rails.</p>
        <HelpStep n={1}><b>Autonomous:</b> observations (notes), web research, starting/stopping <b>Discovery</b> and <b>Training</b>, starting/stopping live engines (the demo gate still protects real-money accounts), and settings changes through the same validated path as the UI.</HelpStep>
        <HelpStep n={2}><b>Never autonomous:</b> closing a position — that lands in <b>Actions</b> for YOUR click.</HelpStep>
        <HelpStep n={3}>Every decision + result is journaled below. <b>Run now</b> triggers one cycle on demand; the toggle enables the recurring loop.</HelpStep>
        <p className="muted small">Requires the AI Desk to be signed in (ChatGPT). Guard-rails: max {cfg?.maxActionsPerTick ?? 3} actions per cycle, whitelisted actions only, blacklisted strategies never started, every config change server-clamped.</p>
      </HelpPanel>

      {error && <div className="banner warn">{String(error).slice(0, 180)}</div>}
      {msg && <div className="banner info">{msg}</div>}

      <div className="ticket">
        <div className="ticket-row" style={{ flexWrap: "wrap", gap: 14 }}>
          <label style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
            <input
              type="checkbox"
              checked={cfg?.enabled ?? false}
              disabled={busy || !cfg}
              onChange={(e) => setCfg({ enabled: e.target.checked })}
            />
            <b>Autonomous loop</b> <Tip text="When on, the Supervisor runs a cycle on the interval below. When off, nothing happens unless you click Run now." />
          </label>
          <label>
            Interval (min) <Tip text="Minutes between autonomous cycles (5–240). Each cycle costs one ChatGPT request." />
            <input
              type="number" min={5} max={240} style={{ width: 70 }}
              value={cfg?.intervalMinutes ?? 30}
              disabled={busy || !cfg}
              onChange={(e) => setCfg({ intervalMinutes: Number(e.target.value) })}
            />
          </label>
          <label>
            Max actions/cycle <Tip text="Hard cap on how many actions one cycle may execute (1–5)." />
            <input
              type="number" min={1} max={5} style={{ width: 56 }}
              value={cfg?.maxActionsPerTick ?? 3}
              disabled={busy || !cfg}
              onChange={(e) => setCfg({ maxActionsPerTick: Number(e.target.value) })}
            />
          </label>
          <button className="primary" disabled={busy} onClick={runNow}>{busy ? "…" : "▶ Run now"}</button>
        </div>
      </div>

      <h2>Talk to the Supervisor <Tip text="Your message steers the next cycle: it reads the full system state + your standing directives, answers you, and can act (same whitelisted actions + guard-rails as the autonomous loop)." /></h2>
      <div className="ticket">
        <div className="ticket-row" style={{ alignItems: "flex-end" }}>
          <label style={{ flex: 1 }}>
            Message
            <input
              type="text"
              value={chatInput}
              placeholder="π.χ. Τι βλέπεις στα live engines; Ξεκίνα discovery στο GBPUSD M15…"
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") sendChat(); }}
              style={{ width: "100%" }}
            />
          </label>
          <button className="primary" disabled={busy || !chatInput.trim()} onClick={sendChat}>Send</button>
        </div>
        {chatReply && (
          <div className="banner info" style={{ whiteSpace: "pre-wrap", marginTop: 8 }}>{chatReply}</div>
        )}
      </div>

      <h2>Standing directives <Tip text="One per line. Injected into EVERY autonomous cycle and chat — your strategy stays in force between conversations. E.g. 'focus discovery on EURUSD+GBPUSD', 'never start live engines without asking me'." /></h2>
      <div className="ticket">
        <textarea
          value={directivesText}
          onChange={(e) => setDirectivesText(e.target.value)}
          placeholder={"π.χ.\nΕστίασε το discovery σε EURUSD και GBPUSD M15\nΠοτέ μην ξεκινάς live engines χωρίς να με ρωτήσεις"}
          spellCheck={false}
          style={{ width: "100%", minHeight: 90, fontFamily: "inherit", fontSize: 13 }}
        />
        <div className="btn-row" style={{ marginTop: 8 }}>
          <button className="primary" disabled={busy} onClick={saveDirectives}>Save directives</button>
        </div>
      </div>

      <h2>Live-experience learnability <Tip text="Trains a model on the EXACT feature rows your live entries acted on, tested on a strictly time-ordered holdout (the future). Answers honestly: do live outcomes carry learnable signal yet? Report only — never touches live trading." /></h2>
      <div className="ticket">
        <div className="btn-row">
          <button disabled={busy} onClick={runExperienceTrain}>🧪 Train from live experience</button>
        </div>
        {expNote && <p className="muted small" style={{ marginTop: 6 }}>{expNote}</p>}
        {expGroups && expGroups.length > 0 && (
          <table className="tbl" style={{ marginTop: 6 }}>
            <thead><tr><th>Portfolio</th><th>Records</th><th>Baseline</th><th>OOS acc</th><th>Edge</th><th>Verdict</th></tr></thead>
            <tbody>
              {expGroups.map((g) => (
                <tr key={g.portfolio}>
                  <td className="muted small" style={{ maxWidth: 220, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={g.portfolio}>{g.portfolio.split(/[\\/]/).pop()}</td>
                  <td>{g.records}</td>
                  <td>{g.testN > 0 ? `${g.baselinePct.toFixed(0)}%` : "—"}</td>
                  <td>{g.testN > 0 ? `${g.oosAccuracyPct.toFixed(0)}%` : "—"}</td>
                  <td className={g.edgePct > 5 ? "buy" : g.edgePct < 0 ? "sell" : ""}>{g.testN > 0 ? `${g.edgePct >= 0 ? "+" : ""}${g.edgePct.toFixed(1)}%` : "—"}</td>
                  <td className="muted small" style={{ maxWidth: 320 }}>{g.verdict}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <h2>Decision log <span className="muted">({log.length})</span></h2>
      {log.length === 0 ? (
        <p className="muted">No activity yet. Click <b>Run now</b> for a first cycle — it will observe the system and write its findings here.</p>
      ) : (
        <table className="tbl">
          <thead><tr><th>When</th><th>Kind</th><th>What</th><th>Result</th></tr></thead>
          <tbody>
            {log.map((e, i) => {
              const badge = KIND_BADGE[e.kind] ?? KIND_BADGE.tick;
              return (
                <tr key={`${e.tsMs}-${i}`}>
                  <td className="muted small" style={{ whiteSpace: "nowrap" }}>{fmtTime(e.tsMs)}</td>
                  <td><span className="badge" style={{ background: badge.bg, fontSize: 9 }}>{badge.label}</span></td>
                  <td className="small" style={{ maxWidth: 420, overflowWrap: "anywhere" }}>{e.detail}</td>
                  <td className="muted small" style={{ maxWidth: 380, overflowWrap: "anywhere" }}>{e.result ?? "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
