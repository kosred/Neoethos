import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  createResearchReader, researchAccountRows, researchEvaluationLabel, researchNumber, researchTimestamp, researchUseLabel,
  type ResearchReadState,
} from "../src/trainingResearch.ts";
import type { SavedResearchAccount, SavedResearchReport, SavedTrainingResearch } from "../src/api.ts";

const firstIdentity = "a".repeat(64);
const secondIdentity = "b".repeat(64);
const account: SavedResearchAccount = {
  netProfit: -40,
  sharpe: -1.25,
  winRate: 0.25,
  profitFactor: 0.6,
  expectancy: -20,
  tradeCount: 2,
  maxDrawdownFraction: 0.125,
  endingRealizedBalance: 960,
  terminalOpen: true,
  grossUnrealizedAccount: 12,
  pendingRoundTripCommissionAccount: 3,
  belowMinEntries: 1,
};
const report: SavedResearchReport = {
  evaluationMode: "train_models",
  reportId: "first-attempt", reportSha256: "c".repeat(64), rawFinalScopeSha256: "d".repeat(64),
  lockedFinalInputsSha256: "e".repeat(64), firstLockedFinalInputsSha256: "e".repeat(64),
  holdoutUse: "first_recorded_local_use_of_reserved_final_scope",
  historicalExposure: "unknown_before_this_local_journal_not_never_ever_seen_evidence",
  symbol: "EURGBP", baseTimeframe: "M5", accountCurrency: "GBP",
  rowStart: 90, rowEnd: 100, rows: 10, timestampStartMs: 1_000_000, timestampEndMs: 3_700_000,
  trainingCutoffMs: 900_000, blendMode: "gate", blendGateFloor: 0.1, blendVetoBelow: 0.2,
  modelHistoryRows: 256, invalidModelSignalRows: 3, configuredLiveMlGate: true,
  promotionEligible: false, geneOnly: account, combined: { ...account, netProfit: -60, endingRealizedBalance: 940 },
};
const response = (identity = firstIdentity): SavedTrainingResearch => ({
  trainingHandoff: identity, status: "completed_results", reports: [report], unavailable: [],
});
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => { resolve = resolvePromise; reject = rejectPromise; });
  return { promise, resolve, reject };
}

test("saved research reader is idle until requested, aborts superseded reads, and keeps all attempts in returned order", async () => {
  const reader = createResearchReader();
  const states: ResearchReadState[] = [];
  const first = deferred<SavedTrainingResearch>();
  const second = deferred<SavedTrainingResearch>();
  const signals: AbortSignal[] = [];
  let calls = 0;
  const fetcher = (_identity: string, signal: AbortSignal) => {
    signals.push(signal);
    return ++calls === 1 ? first.promise : second.promise;
  };
  assert.equal(calls, 0);
  const oldRead = reader.load(firstIdentity, fetcher, (state) => states.push(state));
  const newRead = reader.load(secondIdentity, fetcher, (state) => states.push(state));
  assert.equal(signals[0].aborted, true);
  assert.equal(signals[1].aborted, false);
  const saved = response(secondIdentity);
  saved.reports.push({ ...report, reportId: "later-more-profitable", combined: { ...account, netProfit: 90 } });
  saved.unavailable.push({ reportId: "unreadable-third", reason: "Report hash mismatch" });
  second.resolve(saved);
  await newRead;
  first.resolve(response());
  await oldRead;
  assert.equal(states.length, 3, "the late old success must publish nothing");
  assert.equal(states[2].identity, secondIdentity);
  assert.equal(states[2].loading, false);
  assert.equal(states[2].data, saved);
  assert.deepEqual(states[2].data?.reports.map((item) => item.reportId), ["first-attempt", "later-more-profitable"]);
  assert.equal(states[2].data?.unavailable[0].reason, "Report hash mismatch");
});

test("late errors and cancellation cannot overwrite a newer handoff or publish after unmount", async () => {
  const reader = createResearchReader();
  const states: ResearchReadState[] = [];
  const pending = deferred<SavedTrainingResearch>();
  const oldRead = reader.load(firstIdentity, () => pending.promise, (state) => states.push(state));
  await reader.load(secondIdentity, async () => response(secondIdentity), (state) => states.push(state));
  pending.reject(new Error("old handoff failed"));
  await oldRead;
  assert.equal(states.at(-1)?.data?.trainingHandoff, secondIdentity);
  assert.equal(states.at(-1)?.error, "");
  const unmounted = deferred<SavedTrainingResearch>();
  let signal: AbortSignal | undefined;
  const finalRead = reader.load(firstIdentity, (_identity, incoming) => {
    signal = incoming;
    return unmounted.promise;
  }, (state) => states.push(state));
  const countBeforeUnmount = states.length;
  reader.cancel();
  assert.equal(signal?.aborted, true);
  unmounted.resolve(response());
  await finalRead;
  assert.equal(states.length, countBeforeUnmount);
});

test("response identity mismatch is an error, and a retry clears failed state without starting another operation", async () => {
  const reader = createResearchReader();
  const states: ResearchReadState[] = [];
  await reader.load(firstIdentity, async () => response(secondIdentity), (state) => states.push(state));
  assert.equal(states.at(-1)?.data, null);
  assert.equal(states.at(-1)?.loading, false);
  assert.match(states.at(-1)!.error, /different training handoff/);
  await reader.load(firstIdentity, async () => { throw new Error("404 Not Found"); }, (state) => states.push(state));
  assert.equal(states.at(-1)?.error, "404 Not Found");
  const empty: SavedTrainingResearch = { trainingHandoff: firstIdentity, status: "candidate_not_ready", reports: [], unavailable: [] };
  await reader.load(firstIdentity, async () => empty, (state) => states.push(state));
  assert.equal(states.at(-2)?.error, "");
  assert.equal(states.at(-2)?.loading, true);
  assert.equal(states.at(-1)?.data, empty);
  assert.equal(states.at(-1)?.error, "");
});

test("account comparison preserves losses and separates realized money, open exposure, commission and drawdown percent", () => {
  const rows = researchAccountRows(account, { ...account, terminalOpen: false, grossUnrealizedAccount: null, pendingRoundTripCommissionAccount: null });
  const values = (label: string) => rows.find((row) => row.label === label)!.values;
  assert.deepEqual(values("Closed-trade net P&L"), [researchNumber(-40), researchNumber(-40)]);
  assert.deepEqual(values("Ending realized balance (excludes open P&L)"), [researchNumber(960), researchNumber(960)]);
  assert.deepEqual(values("Maximum modeled drawdown (%)"), [`${researchNumber(12.5)}%`, `${researchNumber(12.5)}%`]);
  assert.deepEqual(values("Position still open at window end"), ["Yes", "No"]);
  assert.deepEqual(values("Open gross unrealized P&L (before pending commission)"), [researchNumber(12), "Unknown"]);
  assert.deepEqual(values("Open pending round-trip commission"), [researchNumber(3), "Unknown"]);
  assert.deepEqual(values("Closed trades"), [researchNumber(2, 0), researchNumber(2, 0)]);
  assert.equal(rows.length, 12);
});

test("strategy-only research has one real account column, not zero or invented model results", async () => {
  const strategyOnly: SavedResearchReport = {
    ...report,
    reportId: "strategies-alone",
    evaluationMode: "strategy_only",
    combined: null,
    blendMode: null,
    blendGateFloor: null,
    blendVetoBelow: null,
    modelHistoryRows: null,
    invalidModelSignalRows: null,
    configuredLiveMlGate: null,
  };
  const saved = response();
  saved.reports = [strategyOnly, report];
  const states: ResearchReadState[] = [];
  await createResearchReader().load(firstIdentity, async () => saved, (state) => states.push(state));
  assert.equal(states.at(-1)?.data, saved);
  assert.deepEqual(states.at(-1)?.data?.reports.map((item) => item.evaluationMode), ["strategy_only", "train_models"]);
  const rows = researchAccountRows(strategyOnly.geneOnly, strategyOnly.combined);
  assert.equal(rows.length, 12);
  assert.ok(rows.every((row) => row.values.length === 1));
  assert.deepEqual(rows[0].values, [researchNumber(-40)]);
  assert.deepEqual(rows[3].values, [`${researchNumber(12.5)}%`]);
  assert.deepEqual(rows[5].values, [researchNumber(12)]);
  assert.equal(strategyOnly.invalidModelSignalRows, null);
  assert.equal(strategyOnly.configuredLiveMlGate, null);
  assert.equal(researchEvaluationLabel(strategyOnly.evaluationMode), "Strategies only — no model training or inference");
  assert.equal(researchEvaluationLabel(report.evaluationMode), "Strategies + candidate models");
  assert.equal(researchEvaluationLabel(undefined), "Strategies + candidate models", "older combined responses omitted mode");
  assert.match(researchEvaluationLabel("unexpected"), /^Unknown evaluation mode:/);
});

test("null, missing and nonfinite money stay unknown; explicit zero stays zero and local first use is not an OOS verdict", () => {
  for (const value of [null, undefined, NaN, Infinity, -Infinity]) assert.equal(researchNumber(value), "Unknown");
  assert.equal(researchNumber(0), (0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
  for (const value of [null, NaN, Infinity, Number.MAX_VALUE]) {
    const rows = researchAccountRows({ ...account, netProfit: value, maxDrawdownFraction: value }, account);
    if (value !== Number.MAX_VALUE) assert.equal(rows[0].values[0], "Unknown");
    assert.equal(rows[3].values[0], "Unknown");
  }
  assert.equal(researchUseLabel(report.holdoutUse), "First recorded local use");
  assert.equal(researchUseLabel("reused_reserved_final_scope_research_only"), "Reused final window — research only");
  assert.match(researchUseLabel("unexpected"), /^Unknown/);
  assert.equal(researchTimestamp(NaN), "Unknown");
});

test("saved account metrics retain each account's ratios, win-rate fraction and currency expectancy", () => {
  const combined = { ...account, sharpe: 0.75, winRate: 0.75, profitFactor: 1.4, expectancy: 12.5 };
  const rows = researchAccountRows(account, combined);
  const values = (label: string) => rows.find((row) => row.label === label)!.values;
  assert.deepEqual(values("Saved Sharpe"), [researchNumber(-1.25), researchNumber(0.75)]);
  assert.deepEqual(values("Win rate (%)"), [`${researchNumber(25)}%`, `${researchNumber(75)}%`]);
  assert.deepEqual(values("Saved profit factor"), [researchNumber(0.6), researchNumber(1.4)]);
  assert.deepEqual(values("Expectancy (account currency / closed trade)"), [researchNumber(-20), researchNumber(12.5)]);
  assert.deepEqual(researchAccountRows(account, null).slice(8).map((row) => row.values), rows.slice(8).map((row) => [row.values[0]]));
});

test("unavailable saved metrics remain unknown in both modes, while explicit zero is preserved", () => {
  for (const value of [null, NaN, Infinity, -Infinity]) {
    const unknown = { ...account, sharpe: value, winRate: value, profitFactor: value, expectancy: value };
    for (const combined of [null, unknown]) {
      assert.ok(researchAccountRows(unknown, combined).slice(8).every((row) => row.values.every((item) => item === "Unknown")));
    }
  }
  // A response from an older backend can omit the newly exposed fields.
  const missing = { ...account };
  for (const field of ["sharpe", "winRate", "profitFactor", "expectancy"]) Reflect.deleteProperty(missing, field);
  assert.ok(researchAccountRows(missing, null).slice(8).every((row) => row.values[0] === "Unknown"));
  const zero = { ...account, sharpe: 0, winRate: 0, profitFactor: 0, expectancy: 0 };
  assert.deepEqual(researchAccountRows(zero, null).slice(8).map((row) => row.values[0]), [
    researchNumber(0), `${researchNumber(0)}%`, researchNumber(0), researchNumber(0),
  ]);
  const overflow = researchAccountRows({ ...account, winRate: Number.MAX_VALUE }, null);
  assert.equal(overflow.find((row) => row.label === "Win rate (%)")!.values[0], "Unknown");
});

test("Training connects the explicit saved GET, selection and unmount cancellation, all reports and unavailable evidence", () => {
  const api = readFileSync(new URL("../src/api.ts", import.meta.url), "utf8");
  const screen = readFileSync(new URL("../src/screens/Training.tsx", import.meta.url), "utf8");
  assert.match(api, /apiGet<T>\(path: string, signal\?: AbortSignal\)/);
  assert.match(api, /fetch\(`\$\{base\}\$\{path\}`, \{ signal \}\)/);
  assert.match(api, /apiGet<SavedTrainingResearch>\(`\/intelligence\/research\?training_handoff=\$\{identity\}`, signal\)/);
  assert.match(api, /\^\[0-9a-f\]\{64\}\$/);
  assert.match(screen, /onClick=\{readSavedResearch\}/);
  assert.match(screen, /researchReader\.load\(selected\.identity, savedTrainingResearch, setResearchRead\)/);
  assert.match(screen, /useEffect\(\(\) => \(\) => researchReader\.cancel\(\), \[researchReader\]\)/);
  assert.match(screen, /const selectHandoff = \(identity: string\) => \{\s*researchReader\.cancel\(\);\s*setResearchRead\(null\);\s*setSelectedIdentity\(identity\)/);
  assert.match(screen, /researchRead\?\.identity === selectedIdentity \? researchRead : null/);
  assert.match(screen, /research\.data\.reports\.map\(\(report\) =>/);
  assert.match(screen, /researchAccountRows\(report\.geneOnly, report\.combined\)\.map/);
  const accountContract = api.slice(api.indexOf("export type SavedResearchAccount ="), api.indexOf("export type SavedResearchReport ="));
  for (const field of ["sharpe", "winRate", "profitFactor", "expectancy"]) {
    assert.ok(accountContract.includes(`${field}: number | null;`), field);
  }
  assert.match(screen, /win rate is shown as a percentage and expectancy as account currency per closed trade/);
  assert.match(screen, /report\.combined != null && <th scope="col">Strategies \+ models/);
  assert.match(screen, /row\.values\.map\(/);
  assert.match(screen, /report\.evaluationMode !== "strategy_only"/);
  assert.match(screen, /report\.configuredLiveMlGate === true \? "Enabled" : report\.configuredLiveMlGate === false \? "Disabled" : "Unknown"/);
  assert.match(screen, /research\.data\.unavailable\.map/);
  assert.match(screen, /report\.invalidModelSignalRows/);
  assert.match(screen, /report\.accountCurrency/);
  assert.match(screen, /Research only — bar-based execution/);
  assert.match(screen, /does not prove the data was never seen before/);
  assert.match(screen, /role="alert">Saved research could not be read/);
  assert.match(screen, /No verified completed report is available/);
  assert.doesNotMatch(screen, /usePoll\(savedTrainingResearch|reports\.sort|reports\.slice|reports\.filter|\$USD/);
  const readStart = screen.indexOf("const readSavedResearch =");
  const readEnd = screen.indexOf("const stop =", readStart);
  assert.ok(readStart >= 0 && readEnd > readStart);
  assert.doesNotMatch(screen.slice(readStart, readEnd), /trainingStart|apiPost|autonomous|evaluate/);
});

test("final-window actions pass explicit modes for one selected handoff and never start on selection or mount", () => {
  const api = readFileSync(new URL("../src/api.ts", import.meta.url), "utf8");
  const screen = readFileSync(new URL("../src/screens/Training.tsx", import.meta.url), "utf8");
  assert.match(api, /ResearchEvaluationMode = "train_models" \| "strategy_only"/);
  assert.match(api, /trainingStart = \(identity: string, mode\?: ResearchEvaluationMode\)/);
  assert.match(api, /mode === undefined \? \{\} : \{ mode \}/, "omission preserves the backend's training default");
  assert.match(screen, /if \(!selected \|\| !canStart\) return/);
  assert.match(screen, /trainingStart\(selected\.identity, mode\)/);
  assert.match(screen, /disabled=\{!canStart\} onClick=\{\(\) => void start\("strategy_only"\)\}/);
  assert.match(screen, /disabled=\{!canStart\} onClick=\{\(\) => void start\("train_models"\)\}/);
  assert.match(screen, /Evaluate strategies only/);
  assert.match(screen, /Train models \+ evaluate final window/);
  assert.match(screen, /Either action uses the reserved final window/);
  assert.match(screen, /Repeating it is research-only/);
  assert.match(screen, /const result = await trainingStop\(\)/);
  assert.match(screen, /No active research job remains to stop/);
  assert.equal((screen.match(/await trainingStart\(/g) ?? []).length, 1);
  const selection = screen.slice(screen.indexOf("const selectHandoff ="), screen.indexOf("const readSavedResearch ="));
  assert.doesNotMatch(selection, /trainingStart|apiPost|start\(/);
  assert.doesNotMatch(screen, /useEffect\([^;]*trainingStart|useEffect\([^;]*\bstart\(/s);
});
