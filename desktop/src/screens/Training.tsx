import { useEffect, useState } from "react";
import { enginesStatus, intelligence, savedTrainingResearch, trainingStart, trainingStop, type ResearchEvaluationMode } from "../api";
import { HelpPanel, HelpStep } from "../components/Help";
import { usePoll } from "../hooks";
import { researchEngineStatus } from "../runtimeStatus";
import {
  createResearchReader, researchAccountRows, researchEvaluationLabel, researchNumber, researchTimestamp, researchUseLabel,
  type ResearchReadState,
} from "../trainingResearch";

export default function Training() {
  const { data: engines, error, reload } = usePoll(enginesStatus, 2_000);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const { data: inventory, error: inventoryError, reload: reloadInventory } = usePoll(intelligence, 10_000);
  const [selectedIdentity, setSelectedIdentity] = useState("");
  const [researchReader] = useState(createResearchReader);
  const [researchRead, setResearchRead] = useState<ResearchReadState | null>(null);
  const handoffs = inventory?.trainingHandoffs ?? [];
  const unavailableHandoffs = inventory?.trainingHandoffUnavailable ?? [];
  const selected = handoffs.find((handoff) => handoff.identity === selectedIdentity);
  const research = selected && researchRead?.identity === selectedIdentity ? researchRead : null;
  const state = engines?.training ?? "Unknown";
  const status = researchEngineStatus(state);
  const running = state === "Running";
  const summary = engines?.trainingSummary ?? engines?.training_summary ?? "";
  const canStart = !busy && !running && engines?.discovery !== "Running"
    && Boolean(engines) && !error && !inventoryError && Boolean(selected);

  useEffect(() => () => researchReader.cancel(), [researchReader]);

  const selectHandoff = (identity: string) => {
    researchReader.cancel();
    setResearchRead(null);
    setSelectedIdentity(identity);
  };

  const readSavedResearch = () => {
    if (!selected) return;
    void researchReader.load(selected.identity, savedTrainingResearch, setResearchRead);
  };

  const stop = async () => {
    setBusy(true);
    try {
      const result = await trainingStop();
      setMessage(result.running
        ? "Cancellation requested. Waiting for the active research job to stop safely."
        : "No active research job remains to stop.");
      await reload();
    } catch (stopError) {
      setMessage(`Stop failed: ${stopError instanceof Error ? stopError.message : String(stopError)}`);
    } finally {
      setBusy(false);
    }
  };

  const start = async (mode: ResearchEvaluationMode) => {
    if (!selected || !canStart) return;
    setBusy(true);
    try {
      const result = await trainingStart(selected.identity, mode);
      setMessage(`${mode === "strategy_only" ? "Strategy-only final-window evaluation" : "Model training and final-window evaluation"} accepted for ${result.symbol} ${result.base_tf}. Follow the research job status; no trading has been enabled.`);
      await reload();
    } catch (startError) {
      setMessage(`Research job could not start: ${startError instanceof Error ? startError.message : String(startError)}`);
    } finally { setBusy(false); }
  };

  return (
    <div className="screen">
      <h1>Final-window research &amp; model handoff</h1>
      <p className="sub">Exact Discovery result → locked strategies → explicit final-window evaluation</p>

      <HelpPanel id="training">
        <p>
          Evaluate the selected strategies without models, or train the configured candidate models
          and compare both on the same final window. Model features are fitted on the purged training
          prefix, never the held-out window.
        </p>
        <HelpStep n={1}>Lock one ResearchOnly result and its immutable input receipt.</HelpStep>
        <HelpStep n={2}>Select its published handoff, which binds the data, cutoff, purge and model plan.</HelpStep>
        <HelpStep n={3}>Choose one explicit evaluation action. Neither action grants promotion or live authority.</HelpStep>
      </HelpPanel>

      <div className="engine-status">
        <span className={`badge ${status.badgeClass}`}>
          {status.label}
        </span>
        <span className="muted">Research job · training or strategy-only evaluation</span>
      </div>
      {(status.notice || summary) && (
        <div className={`banner ${status.warning ? "warn" : "info"}`} role={status.warning ? "alert" : "status"}>
          {status.notice && <p>{status.notice}</p>}
          <div style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{summary}</div>
        </div>
      )}
      {error && <div className="banner warn" role="alert">{error}</div>}

      {inventoryError && <div className="banner warn" role="alert">{inventoryError}</div>}
      {unavailableHandoffs.length > 0 && <div className="banner warn" role="alert">
        Some saved handoffs could not be verified. Valid selections remain available; no files were deleted.
        <details><summary>Unavailable handoffs ({unavailableHandoffs.length})</summary>
          <ul>{unavailableHandoffs.map((item) => <li key={item.identity} style={{ overflowWrap: "anywhere" }}>{item.identity}: {item.reason}</li>)}</ul>
        </details>
      </div>}

      <div className="ticket">
        <h2>Choose a Discovery result</h2>
        <p>Both actions use the exact selected portfolio and its immutable data. Strategy-only evaluation
          does not train or load models. The model action trains the complete configured set before evaluating.</p>
        {!handoffs.length && !inventoryError && <p>{unavailableHandoffs.length > 0
          ? "No verified handoff is available. See the saved-file errors above."
          : "No evaluation-ready result yet. Complete Discovery with a calibration-surviving portfolio first."}</p>}
        <label htmlFor="training-handoff">Exact research result</label>
        <select id="training-handoff" value={selectedIdentity} onChange={(event) => selectHandoff(event.target.value)} disabled={busy || running}>
          <option value="">Select a result…</option>
          {handoffs.map((handoff) => <option key={handoff.identity} value={handoff.identity}>
            {handoff.symbol} {handoff.baseTf} · {handoff.strategyCount} strategies · {handoff.identity.slice(0, 12)}
          </option>)}
        </select>
        {selected && <div className="banner info">
          <p>Models: {selected.plannedModels.join(", ")}</p>
          <p>Training cutoff (before calibration): {researchTimestamp(selected.oosCutoffMs)} · purge: {selected.purgeBars} bars</p>
          <details><summary>Exact data identity</summary><p style={{ overflowWrap: "anywhere" }}>{selected.datasetIdentity}<br />{selected.generation}<br />{selected.identity}</p></details>
        </div>}
        <p className="banner warn">Either action uses the reserved final window. Repeating it is research-only;
          first recorded local use does not prove the data was never seen before. Starting does not activate trading.</p>
        <button type="button" disabled={!canStart} onClick={() => void start("strategy_only")}>
          Evaluate strategies only
        </button>
        <button type="button" disabled={!canStart} onClick={() => void start("train_models")}>
          Train models + evaluate final window
        </button>
        <button type="button" disabled={busy} onClick={() => void reloadInventory()}>Refresh results</button>
        {running && (
          <button type="button" className="danger" disabled={busy || Boolean(error)} onClick={stop}>
            Stop research job
          </button>
        )}
        {message && <div className="banner info" role="status">{message}</div>}
      </div>

      <section className="ticket" aria-labelledby="saved-research-heading" aria-busy={Boolean(research?.loading)}>
        <h2 id="saved-research-heading">Saved final-window research</h2>
        <p>Inspect strategies alone, or their comparison with candidate models. Reading saved reports
          does not run training or evaluation. Every matching completed attempt is shown, without profit ranking.</p>
        <p className="banner warn">Research only — bar-based execution, not broker fills. Completion is not proof of
          profitability, independent out-of-sample success, or permission to trade. First recorded local use
          does not prove the data was never seen before.</p>
        <button type="button" disabled={!selected || Boolean(inventoryError) || Boolean(research?.loading)} onClick={readSavedResearch}>
          {research?.loading ? "Reading saved research…" : "Read saved research for selected result"}
        </button>
        {!selected && <p>Select an exact research result above.</p>}
        {selected && !research && <p>No report has been loaded for this selection.</p>}
        {research?.loading && <p role="status">Checking saved reports and their exact identities. No evaluation is being started.</p>}
        {research?.error && <p className="banner warn" role="alert">Saved research could not be read: {research.error}</p>}
        {research?.data && <>
          {research.data.status === "candidate_not_ready" && <p>No saved result is ready for this handoff yet.</p>}
          {research.data.status !== "candidate_not_ready" && research.data.reports.length === 0 && <p>No verified completed report is available for this handoff.</p>}
          {research.data.reports.map((report) => (
            <section key={report.reportId} aria-label={`Saved research attempt ${report.reportId}`}>
              <h3>{report.symbol} {report.baseTimeframe} · {researchUseLabel(report.holdoutUse)}</h3>
              <p>{researchEvaluationLabel(report.evaluationMode)}</p>
              <p>Final window: {researchTimestamp(report.timestampStartMs)} → {researchTimestamp(report.timestampEndMs)} (last included bar).
                {" "}{researchNumber(report.rows, 0)} bars · all monetary values in {report.accountCurrency || "unknown account currency"}.</p>
              <div style={{ overflowX: "auto" }}>
                <table>
                  <caption>Saved final-window account results — research only</caption>
                  <thead><tr><th scope="col">Measure</th><th scope="col">Strategies only</th>{report.combined != null && <th scope="col">Strategies + models</th>}</tr></thead>
                  <tbody>{researchAccountRows(report.geneOnly, report.combined).map((row) => (
                    <tr key={row.label}><th scope="row">{row.label}</th>{row.values.map((value, index) => <td key={index}>{value}</td>)}</tr>
                  ))}</tbody>
                </table>
              </div>
              {report.evaluationMode !== "strategy_only" && <p>Invalid model prediction rows: {researchNumber(report.invalidModelSignalRows, 0)} · model history: {researchNumber(report.modelHistoryRows, 0)} bars.</p>}
              <p>Unknown means no finite saved value, not zero. Open exposure is separate from realized results. Sharpe and profit factor retain the saved calculation; win rate is shown as a percentage and expectancy as account currency per closed trade.</p>
              <details>
                <summary>Attempt identity and saved policy</summary>
                <dl style={{ overflowWrap: "anywhere" }}>
                  <dt>Report</dt><dd>{report.reportId}</dd>
                  <dt>Report SHA-256</dt><dd>{report.reportSha256}</dd>
                  <dt>Final scope SHA-256</dt><dd>{report.rawFinalScopeSha256}</dd>
                  <dt>Locked final inputs</dt><dd>{report.lockedFinalInputsSha256}</dd>
                  <dt>First locally recorded locked inputs</dt><dd>{report.firstLockedFinalInputsSha256 ?? "Unknown"}</dd>
                  <dt>Local use classification</dt><dd>{report.holdoutUse}</dd>
                  <dt>Earlier historical exposure</dt><dd>{report.historicalExposure}</dd>
                  <dt>Rows [start, end)</dt><dd>{report.rowStart} → {report.rowEnd}</dd>
                  <dt>Training cutoff</dt><dd>{researchTimestamp(report.trainingCutoffMs)}</dd>
                  {report.evaluationMode !== "strategy_only" && <>
                    <dt>Saved blend policy</dt><dd>{report.blendMode ?? "Unknown"} · floor {researchNumber(report.blendGateFloor)} · veto below {researchNumber(report.blendVetoBelow)}</dd>
                    <dt>Saved ML entry gate</dt><dd>{report.configuredLiveMlGate === true ? "Enabled" : report.configuredLiveMlGate === false ? "Disabled" : "Unknown"}</dd>
                  </>}
                </dl>
              </details>
              {report.promotionEligible && <p className="banner warn" role="alert">Unexpected promotion flag in research evidence; this view grants no trading permission.</p>}
            </section>
          ))}
          {research.data.unavailable.length > 0 && <div className="banner warn">
            <h3>Unavailable evidence</h3>
            <p>These saved attempts could not be verified; they are not counted as failed or successful evaluations.</p>
            <ul>{research.data.unavailable.map((item, index) => <li key={`${item.reportId}:${index}`} style={{ overflowWrap: "anywhere" }}>
              {item.reportId}: {item.reason}
            </li>)}</ul>
          </div>}
        </>}
      </section>
    </div>
  );
}
