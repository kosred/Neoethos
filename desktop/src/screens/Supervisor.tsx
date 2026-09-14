import { useState } from "react";
import {
  supervisorStatus,
  supervisorConfig,
  supervisorTick,
  experienceTrain,
  type SupervisorLogEntry,
  type SupervisorConfig,
  type SupervisorObservation,
  type ExperienceGroup,
} from "../api";
import { usePoll } from "../hooks";
import { sameSupervisorConfig } from "../apiContracts";
import { HelpPanel, HelpStep, Tip } from "../components/Help";

const fmtTime = (ms: number) => (ms > 0 ? new Date(ms).toLocaleString() : "—");

const KIND_BADGE: Record<string, { label: string; bg: string }> = {
  tick: { label: "TICK", bg: "#374151" },
  action: { label: "ACTION", bg: "#1d4ed8" },
  note: { label: "NOTE", bg: "#15803d" },
  chat: { label: "CHAT", bg: "#7c3aed" },
  error: { label: "ERROR", bg: "#b91c1c" },
};

function SupervisorControls({
  config,
  busy,
  canRun,
  onSave,
  onRun,
}: {
  config: SupervisorConfig;
  busy: boolean;
  canRun: boolean;
  onSave: (config: SupervisorConfig) => Promise<SupervisorConfig>;
  onRun: () => Promise<void>;
}) {
  const [baseline, setBaseline] = useState(config);
  const [enabled, setEnabled] = useState(config.enabled);
  const [interval, setInterval] = useState(String(config.intervalMinutes));
  const [maxActions, setMaxActions] = useState(String(config.maxActionsPerTick));
  const [text, setText] = useState(() => config.directives.join("\n"));
  const [error, setError] = useState("");
  const directives = text.split("\n").map((line) => line.trim()).filter(Boolean);
  const draft = { enabled, intervalMinutes: Number(interval), maxActionsPerTick: Number(maxActions), directives };
  const dirty = !sameSupervisorConfig(draft, baseline);
  const conflict = !sameSupervisorConfig(config, baseline);
  const valid = Number.isInteger(draft.intervalMinutes) && draft.intervalMinutes >= 5 && draft.intervalMinutes <= 240
    && Number.isInteger(draft.maxActionsPerTick) && draft.maxActionsPerTick >= 1 && draft.maxActionsPerTick <= 5
    && directives.length <= 20;
  const reset = (saved: SupervisorConfig) => {
    setBaseline(saved);
    setEnabled(saved.enabled);
    setInterval(String(saved.intervalMinutes));
    setMaxActions(String(saved.maxActionsPerTick));
    setText(saved.directives.join("\n"));
    setError("");
  };
  return (
    <form className="ticket" onSubmit={(event) => {
      event.preventDefault();
      if (busy || !dirty || !valid || conflict) return;
      setError("");
      void onSave(draft).then(reset).catch((reason) => setError(`Save failed: ${reason}`));
    }}>
      <fieldset disabled={busy} style={{ border: 0, padding: 0, margin: 0 }}>
        <legend>Cycle controls &amp; standing directives</legend>
        <div className="ticket-row" style={{ flexWrap: "wrap", gap: 14, marginTop: 10 }}>
          <label style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
            <input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} /> Enable scheduled cycles
          </label>
          <label>Interval (minutes)
            <input type="number" min={5} max={240} step={1} value={interval} onChange={(event) => setInterval(event.target.value)} />
          </label>
          <label>Maximum actions per cycle
            <input type="number" min={1} max={5} step={1} value={maxActions} onChange={(event) => setMaxActions(event.target.value)} />
          </label>
        </div>
        <label style={{ display: "block", marginTop: 12 }}>Standing directives · one per line, up to 20
          <textarea value={text} onChange={(event) => setText(event.target.value)} spellCheck={false}
            placeholder="π.χ. Εστίασε το discovery σε EURUSD και GBPUSD M15"
            style={{ width: "100%", minHeight: 90, fontFamily: "inherit", fontSize: 13 }} />
        </label>
        {!valid && <p className="banner warn" role="alert">Use 5–240 whole minutes, 1–5 actions and at most 20 directives.</p>}
        {conflict && <p className="banner warn" role="alert">Saved settings changed elsewhere. Your draft is preserved; reload the saved settings before saving.</p>}
        {error && <p className="banner warn" role="alert">{error}</p>}
        <div className="btn-row" style={{ marginTop: 8 }}>
          <button type="submit" className="primary" disabled={!dirty || !valid || conflict}>Save settings</button>
          <button type="button" disabled={!dirty && !conflict} onClick={() => reset(config)}>Reload saved settings</button>
          <button type="button" disabled={!canRun || dirty || conflict} onClick={() => void onRun()}>Run one cycle</button>
          {dirty && <span className="muted small">Unsaved changes · save before running a cycle</span>}
        </div>
      </fieldset>
    </form>
  );
}

function Observation({ value }: { value: SupervisorObservation }) {
  const account = value.account;
  const number = (n: number | null | undefined, decimals = 5) =>
    n != null && Number.isFinite(n) ? n.toLocaleString(undefined, { maximumFractionDigits: decimals }) : "—";
  return (
    <section aria-label="Supervisor observations">
      <h2>Market &amp; open positions</h2>
      <p className="muted small">Observed {fmtTime(value.observedAtUnixMs)}. This is the same account, quote and engine snapshot included in the next AI cycle, refreshed when that cycle begins.</p>
      {value.accountFailure && <div className="banner warn" role="alert" style={{ overflowWrap: "anywhere" }}>
        Account refresh failed at {fmtTime(value.accountFailure.observedAtUnixMs)}. Any retained positions below are stale.<br />
        {value.accountFailure.code}: {value.accountFailure.detail}
      </div>}
      {!account ? <div className="banner warn">Account snapshot unavailable — open positions and account protection are unknown, not zero.</div> : <>
        <div className="settings-grid">
          <div className="kv"><span>Observed account</span><b>{account.sourceEnvironment} · {account.sourceAccountId}</b></div>
          <div className="kv"><span>Balance</span><b>{number(account.balance, 2)} {account.currency}</b></div>
          <div className="kv"><span>Equity</span><b>{number(account.equity, 2)} {account.currency}</b></div>
          <div className="kv"><span>Account snapshot</span><b>{fmtTime(account.fetchedAtUnixMs)}</b></div>
        </div>
        {account.positions.length === 0 ? <p className="muted">No open positions in this account snapshot.</p> : <div style={{ overflowX: "auto" }}>
          <table className="tbl"><thead><tr><th>Position</th><th>Entry</th><th>Broker SL / TP</th><th>P/L · {account.currency}</th><th>Engine report · scope unverified</th></tr></thead>
            <tbody>{account.positions.map((position) => {
              const engine = value.liveEngines.find((entry) => entry.running && entry.openPositionId === position.positionId);
              return <tr key={position.positionId}>
                <td>{position.symbol} · {position.side}<div className="muted small">#{position.positionId} · {number(position.volumeLots, 2)} lots</div></td>
                <td>{number(position.entryPrice)}</td>
                <td>{number(position.stopLoss)} / {number(position.takeProfit)}{position.stopLoss == null && <div className="sell small">No broker stop reported</div>}</td>
                <td className={position.pnlUsd < 0 ? "sell" : "buy"}>{number(position.pnlUsd, 2)}</td>
                <td>{engine ? <>
                  <div className="sell small">Position-ID match only; engine account ownership is unverified.</div>
                  {engine.protectionState ?? "State unknown"}
                  <div className="muted small">Favourable move {number(engine.favorableMoveR, 2)} R · confirmed SL {number(engine.confirmedStopPrice)}</div>
                  {engine.lastProtectionError && <div className="sell small">{engine.lastProtectionError}</div>}
                </> : <span className="sell">No matching running engine report</span>}</td>
              </tr>;
            })}</tbody>
          </table>
        </div>}
      </>}
      {value.liveEngineError && <div className="banner warn">Engine state unavailable: {value.liveEngineError}</div>}
      <h3>Quotes · {value.market.spots.length} cached symbols</h3>
      {value.market.spots.length === 0 ? <p className="banner warn">No quotes observed. Check the broker connection and live watchlist in Trading.</p> : <div style={{ maxHeight: 230, overflow: "auto" }}>
        <table className="tbl"><thead><tr><th>Symbol</th><th>Bid</th><th>Ask</th><th>Broker timestamp</th><th>Last event age</th></tr></thead>
          <tbody>{value.market.spots.map((quote) => <tr key={quote.symbolId}>
            <td>{quote.symbolName}</td><td>{number(quote.bid)}</td><td>{number(quote.ask)}</td>
            <td className="small">{quote.brokerTimestampMs == null ? "Unknown" : fmtTime(quote.brokerTimestampMs)}</td>
            <td>{number(quote.freshnessSeconds, 1)} s</td>
          </tr>)}</tbody>
        </table>
      </div>}
      <p className="muted small">Event age measures the last received update; it does not prove that both quote sides are fresh. A periodic AI cycle is not a broker stop or continuous position protection.</p>
    </section>
  );
}

export default function Supervisor({ aiReady = false }: { aiReady?: boolean }) {
  const { data, error, loading, reload } = usePoll(supervisorStatus, 3000);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  // The workspace embeds these controls before the unified AI Desk chat.
  // Live-experience learnability report.
  const [expGroups, setExpGroups] = useState<ExperienceGroup[] | null>(null);
  const [expNote, setExpNote] = useState("");

  const cfg = data?.config;
  const log: SupervisorLogEntry[] = data?.log ?? [];

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

  const saveConfig = async (config: SupervisorConfig) => {
    setBusy(true);
    try {
      const saved = await supervisorConfig(config);
      setMsg("Settings saved. Scheduled cycles follow the saved enable/interval settings.");
      await reload();
      return saved.config;
    } finally {
      setBusy(false);
    }
  };

  const runNow = async () => {
    setBusy(true);
    setMsg("Running a supervisor cycle… (gathers state, asks the AI, executes)");
    try {
      const r = await supervisorTick();
      setMsg(r.summary);
      await reload();
    } catch (e) {
      setMsg(`Tick failed: ${e}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <h1>
        Supervisor{" "}
        {cfg && (
          <span className={`badge ${cfg.enabled ? "live" : "demo"}`}>
            {data?.cycleRunning ? "CYCLE RUNNING" : cfg.enabled ? `SCHEDULED · every ${cfg.intervalMinutes}m` : "SCHEDULE PAUSED"}
          </span>
        )}
      </h1>
      <p className="sub">Market observations, position protection and an accountable decision log</p>

      <HelpPanel id="supervisor">
        <p>The Supervisor sends account positions, cached quotes with timestamps, engine protection state, journal and research state to <b>your ChatGPT sign-in</b>. It can request only the implemented actions; backend gates still decide whether an action is allowed.</p>
        <HelpStep n={1}><b>Autonomous:</b> observations (notes), web research, <b>Discovery</b> on an exact saved dataset generation, <b>Training</b> from an exact published Discovery handoff, starting/stopping live engines, and settings changes through the same validated paths as the UI. Research results are not trading permission; backend promotion and execution checks still apply.</HelpStep>
        <HelpStep n={2}><b>Never autonomous:</b> closing a position — that lands in <b>Orders &amp; approvals</b> for YOUR click.</HelpStep>
        <HelpStep n={3}>Every decision + result is journaled below. <b>Run one cycle</b> triggers one cycle on demand; enable and save scheduled cycles for the recurring loop.</HelpStep>
        <p><b>Stopping an engine does not close its broker positions.</b> It also stops that engine's local monitoring. Position closes proposed by the AI require confirmation in <b>Trading → Orders &amp; approvals</b>.</p>
        <p className="muted small">Requires the AI Desk to be signed in (ChatGPT). Guard-rails: max {cfg?.maxActionsPerTick ?? 3} actions per cycle, whitelisted actions only, blacklisted strategies never started, every config change server-clamped.</p>
      </HelpPanel>

      {loading && !data && <p className="muted" role="status">Loading supervisor state…</p>}
      {error && <div className="banner warn" role="alert">Cannot refresh supervisor state. Any retained snapshot below may be stale. {error} <button onClick={() => void reload()}>Retry</button></div>}
      {msg && <div className="banner info" role="status" style={{ whiteSpace: "pre-wrap" }}>{msg}</div>}
      {data?.observation && <Observation value={data.observation} />}
      {!aiReady && <p className="banner warn">AI connection is not confirmed. Market observations remain available; connect ChatGPT before requesting a cycle.</p>}
      {cfg && <SupervisorControls config={cfg} busy={busy || Boolean(error)} canRun={aiReady && !data?.cycleRunning}
        onSave={saveConfig} onRun={runNow} />}

      <details className="parameter-group" style={{ marginTop: 18 }}>
        <summary>Live-experience learnability</summary>
        <div className="details-body">
        <p className="muted small">Optional training report from recorded live entries. This does not deploy a model or start trading. <Tip text="Trains a model on the EXACT feature rows your live entries acted on, tested on a strictly time-ordered holdout (the future). Answers honestly: do live outcomes carry learnable signal yet? Report only — never touches live trading." /></p>
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
      </details>

      <h2>Decision log <span className="muted">({log.length})</span></h2>
      {log.length === 0 ? (
        <p className="muted">No recorded cycles. A requested cycle observes state and may execute allowed actions; its decisions and results appear here.</p>
      ) : (
        <div className="table-scroll supervisor-log">
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
        </div>
      )}
    </div>
  );
}
