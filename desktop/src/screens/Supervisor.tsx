import { useState } from "react";
import {
  supervisorStatus,
  supervisorConfig,
  supervisorTick,
  type SupervisorLogEntry,
} from "../api";
import { usePoll } from "../hooks";
import { HelpPanel, HelpStep, Tip } from "../components/Help";

const fmtTime = (ms: number) => (ms > 0 ? new Date(ms).toLocaleString() : "—");

const KIND_BADGE: Record<string, { label: string; bg: string }> = {
  tick: { label: "TICK", bg: "#374151" },
  action: { label: "ACTION", bg: "#1d4ed8" },
  note: { label: "NOTE", bg: "#15803d" },
  error: { label: "ERROR", bg: "#b91c1c" },
};

export default function Supervisor() {
  const { data, error, reload } = usePoll(supervisorStatus, 10000);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const cfg = data?.config;
  const log: SupervisorLogEntry[] = data?.log ?? [];

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
