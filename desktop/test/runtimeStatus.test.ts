import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { brokerConnectionView, discoveryCounterRows, discoveryGenerationProgress, researchEngineStatus } from "../src/runtimeStatus.ts";

const credentials = { configured: true, hasToken: true, accountId: "42", environment: "Demo" };
const observation = { connected: true, accountId: "42", environment: "Demo", lastSnapshotAtUnixMs: 100_000 };

test("degraded research is a warning, not full success or a crashed job", () => {
  const degraded = researchEngineStatus("Degraded");
  assert.equal(degraded.label, "Degraded — completed with limitations");
  assert.equal(degraded.badgeClass, "");
  assert.equal(degraded.warning, true);
  assert.match(degraded.notice, /reported reason and saved results/);
  assert.match(degraded.notice, /not full success or trading approval/);
  for (const state of ["Idle", "Running", "Succeeded", "Failed", "Cancelled", "Unknown"] as const) {
    const status = researchEngineStatus(state);
    assert.equal(status.label, state);
    assert.equal(status.notice, "");
    assert.equal(status.warning, state === "Failed");
    assert.equal(status.badgeClass, state === "Running" ? "live" : state === "Succeeded" ? "demo" : "");
  }
});

test("both research screens consume the distinct wire state and preserve diagnostic text", () => {
  const api = readFileSync(new URL("../src/api.ts", import.meta.url), "utf8");
  assert.match(api, /export type EngineRunState = [^;]*"Degraded"/);
  for (const [screen, status, state, summary] of [
    ["Discovery", "discoveryStatus", "discoveryState", "engines?.discoverySummary"],
    ["Training", "status", "state", "summary"],
  ]) {
    const source = readFileSync(new URL(`../src/screens/${screen}.tsx`, import.meta.url), "utf8");
    assert.ok(source.includes(`const ${status} = researchEngineStatus(${state})`));
    assert.ok(source.includes(`${status}.label`));
    assert.ok(source.includes(`${status}.notice && <p>{${status}.notice}</p>`));
    assert.ok(source.includes(`role={${status}.warning ? "alert" : "status"}`));
    assert.ok(source.includes(`whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{${summary}}`));
  }
});

test("connection uses a recent observation for the same configured account, not a stored token", () => {
  assert.equal(brokerConnectionView(credentials, null, 100_000).connected, false);
  assert.equal(brokerConnectionView(credentials, { ...observation, lastSnapshotAtUnixMs: undefined }, 100_000).connected, false);
  const healthy = brokerConnectionView(credentials, observation, 101_000);
  assert.equal(healthy.connected, true);
  assert.equal(healthy.label, "Demo · account connected");
  assert.match(healthy.detail, /Quote freshness and trading readiness are separate/);
});

test("a retained observation expires even without another backend response", () => {
  assert.equal(brokerConnectionView(credentials, observation, 115_000).connected, true);
  const stalled = brokerConnectionView(credentials, observation, 115_001);
  assert.equal(stalled.connected, false);
  assert.match(stalled.label, /stale/);
});

test("account changes, disconnects and poll failures cannot retain a green indicator", () => {
  for (const changed of [
    { ...observation, connected: false },
    { ...observation, accountId: "another-account" },
    { ...observation, environment: "Live" },
  ]) assert.equal(brokerConnectionView(credentials, changed, 101_000).connected, false);
  assert.equal(brokerConnectionView(credentials, observation, 101_000, "config failed").connected, false);
  assert.equal(brokerConnectionView(credentials, observation, 101_000, "", "backend failed").connected, false);
});

test("missing configuration, authentication or an account never proves connection", () => {
  assert.equal(brokerConnectionView(null, observation, 101_000).connected, false);
  for (const changed of [
    { ...credentials, configured: false },
    { ...credentials, hasToken: false },
    { ...credentials, accountId: null },
  ]) assert.equal(brokerConnectionView(changed, observation, 101_000).connected, false);
});

test("invalid or future timestamps are not current broker observations", () => {
  for (const timestamp of [null, undefined, NaN, Infinity, -1, 0, 100_000.5, 100_001]) {
    assert.equal(brokerConnectionView(credentials, { ...observation, lastSnapshotAtUnixMs: timestamp }, 100_000).connected, false);
  }
  assert.equal(brokerConnectionView(credentials, observation, NaN).connected, false);
});

test("generation percentage comes from the generation counters, not weighted overall progress", () => {
  const progress = discoveryGenerationProgress([
    { name: "generation", value: 681 }, { name: "generations", value: 1000 },
    { name: "overall_progress", value: 86.8 },
  ]);
  assert.equal(progress?.completed, 681);
  assert.equal(progress?.total, 1000);
  assert.ok(Math.abs(progress!.percent - 68.1) < 1e-12);
  assert.equal(discoveryGenerationProgress([{ name: "generation", value: 0 }, { name: "generations", value: 1000 }])?.percent, 0);
});

test("missing, contradictory or malformed generation counters remain unknown", () => {
  assert.equal(discoveryGenerationProgress(undefined), undefined);
  assert.equal(discoveryGenerationProgress([{ name: "generations", value: 1000 }]), undefined);
  for (const [done, total] of [[1, 0], [-1, 1000], [1001, 1000], [NaN, 1000], [1, Infinity], [1.5, 1000]]) {
    assert.equal(discoveryGenerationProgress([{ name: "generation", value: done }, { name: "generations", value: total }]), undefined);
  }
  assert.equal(discoveryGenerationProgress([
    { name: "generation", value: 1 }, { name: "generation", value: 2 }, { name: "generations", value: 1000 },
  ]), undefined);
});

test("search counters keep plans, admitted candidates, capacity and actual walk-forward verdicts separate", () => {
  const rows = discoveryCounterRows([
    { name: "planned_ga_evaluations", value: 200000 },
    { name: "ga_returned_candidates", value: 10000 },
    { name: "validation_candidate_limit", value: 200 },
    { name: "validation_candidates_admitted", value: 200 },
    { name: "validation_candidates_capped", value: 9800 },
    { name: "quality_evaluated", value: 180 },
    { name: "quality_screened", value: 120 },
    { name: "walkforward_tested", value: 120 },
    { name: "walkforward_passed", value: 100 },
    { name: "walkforward_failed", value: 20 },
    { name: "walkforward_not_tested", value: 80 },
    { name: "portfolio_capacity_not_selected", value: 96 },
    { name: "robustness_removed", value: 2 },
    { name: "portfolio_selected", value: 2 },
  ]);
  const row = (name: string) => rows.find((entry) => entry.name === name)!;
  assert.equal(row("planned_ga_evaluations").value, 200000);
  assert.match(row("planned_ga_evaluations").label, /Planned.*not generated/);
  assert.equal(row("ga_returned_candidates").label, "GA returned candidate pool");
  assert.match(row("validation_candidates_capped").label, /Not tested.*budget cap/);
  assert.equal(row("walkforward_failed").value, 20);
  assert.match(row("walkforward_not_tested").label, /not tested/);
  assert.match(row("portfolio_capacity_not_selected").label, /Not selected.*capacity/);
  assert.deepEqual(row("robustness_removed"), {
    name: "robustness_removed", label: "Removed by robustness checks", value: 2,
  });
  assert.deepEqual(row("portfolio_selected"), {
    name: "portfolio_selected", label: "Selected at reported stage", value: 2,
  });
  assert.match(row("quality_screened").label, /survivors.*not evaluations/);
  assert.match(row("quality_evaluated").label, /backtests completed/);
  assert.equal(rows.length, 14, "presentation must not invent any absent stage count");
  assert.deepEqual(discoveryCounterRows([{ name: "portfolio_selected", value: 4 }]), [
    { name: "portfolio_selected", label: "Selected at reported stage", value: 4 },
  ], "an earlier selection snapshot must not claim a final result or fabricate robustness coverage");
  for (const admitted of [10000, 200]) {
    const capped = 10000 - admitted;
    assert.deepEqual(discoveryCounterRows([
      { name: "candidates", value: admitted },
      { name: "validation_candidates_admitted", value: admitted },
      { name: "validation_candidates_capped", value: capped },
    ]), [
      { name: "candidates", label: "Candidates admitted after ranking", value: admitted },
      { name: "validation_candidates_admitted", label: "Candidates admitted to validation", value: admitted },
      { name: "validation_candidates_capped", label: "Not tested — validation budget cap", value: capped },
    ], "all-retained and capped pools have one skipped-by-cap observation, never the retained count");
  }
  const labels = readFileSync(new URL("../src/runtimeStatus.ts", import.meta.url), "utf8");
  assert.doesNotMatch(labels, /truncated_candidates\s*:/, "the obsolete duplicate cap label is removed");
});

test("unreported and invalid census values stay unknown while explicit zero is preserved", () => {
  assert.deepEqual(discoveryCounterRows(undefined), []);
  assert.deepEqual(discoveryCounterRows(null), []);
  assert.deepEqual(discoveryCounterRows([{ name: "walkforward_failed", value: 0 }]), [
    { name: "walkforward_failed", label: "Walk-forward candidates failed", value: 0 },
  ]);
  for (const value of [NaN, Infinity, -1, 0.25, Number.MAX_SAFE_INTEGER + 1]) {
    assert.equal(discoveryCounterRows([{ name: "walkforward_tested", value }])[0].value, null);
  }
  assert.equal(discoveryCounterRows([
    { name: "walkforward_tested", value: 4 },
    { name: "walkforward_tested", value: 5 },
  ])[0].value, null, "ambiguous duplicate observations are not summed or silently chosen");
  const future = discoveryCounterRows([{ name: "cpcv_gene_fold_tests", value: 25 }]);
  assert.deepEqual(future, [{ name: "cpcv_gene_fold_tests", label: "CPCV gene fold tests", value: 25 }]);
});

test("zero validation limits mean all returned candidates, not zero completed work", () => {
  for (const name of ["target_candidates", "validation_candidate_limit"]) {
    for (const value of [0, 1, 200]) {
      const [row] = discoveryCounterRows([{ name, value }]);
      assert.equal(row.value, value);
      assert.match(row.label, /0 = all GA-returned candidates/);
    }
  }
  const [completed] = discoveryCounterRows([{ name: "quality_evaluated", value: 0 }]);
  assert.equal(completed.value, 0);
  assert.doesNotMatch(completed.label, /all GA-returned candidates/);
});

test("working-set counters separate completed batches, research reports and evaluation handoffs", () => {
  const observations = [
    { name: "working_set_batch", value: 3 },
    { name: "working_set_completed_batches", value: 2 },
    { name: "working_set_completed_entries", value: 40 },
    { name: "working_set_total_entries", value: 342 },
    { name: "working_set_saved_results", value: 2 },
    { name: "working_set_training_handoffs", value: 0 },
    { name: "working_set_publication_failures", value: 1 },
  ];
  const rows = discoveryCounterRows(observations);
  assert.deepEqual(rows.map(({ name, value }) => ({ name, value })), observations);
  assert.deepEqual(rows.map(({ label }) => label), [
    "Current batch", "Completed batches", "Selection entries in completed batches",
    "Total selection entries", "Saved batch research reports",
    "Results available for final evaluation", "Portfolio/handoff publication failures",
  ]);
  assert.equal(rows.length, 7, "no unobserved success or coverage count is fabricated");
  assert.equal(discoveryCounterRows([
    { name: "working_set_completed_entries", value: 40 },
    { name: "working_set_completed_entries", value: 50 },
  ])[0].value, null, "conflicting coverage remains unknown");
});

test("streamed Discovery labels its percentage per batch, not completion of the whole search", () => {
  const discovery = readFileSync(new URL("../src/screens/Discovery.tsx", import.meta.url), "utf8");
  assert.match(discovery, /const workingSetBatch = counterRows\.find\(\(counter\) => counter\.name === "working_set_batch"\)/);
  assert.match(discovery, /const progressLabel = workingSetBatch \? "Current batch progress" : "Overall progress"/);
  assert.match(discovery, /aria-label=\{progressLabel\}/);
  assert.match(discovery, /batch totals refer to the whole run/);
  assert.match(discovery, /Completed batches include empty results/);
  assert.match(discovery, /workingSetBatch\.value === null \? "unknown"/);
});

test("Discovery renders reported counts after terminal state independently of running-only controls", () => {
  const discovery = readFileSync(new URL("../src/screens/Discovery.tsx", import.meta.url), "utf8");
  assert.match(discovery, /const counterRows = discoveryCounterRows\(engines\?\.discoveryCounters\)/);
  assert.match(discovery, /Stop current run<\/button>\s*<\/>\s*\)\}\s*\{counterRows\.length > 0 &&/);
  assert.match(discovery, /discoveryRunning \? "current run" : "last reported run"/);
  assert.match(discovery, /counter\.value === null \? "Unavailable"/);
  assert.doesNotMatch(discovery, /discoveryRunning && counterRows/);
});

test("the shell consumes backend connection evidence and refreshes its display clock with cleanup", () => {
  const app = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
  assert.match(app, /apiGet<BrokerConnectionObservation>\("\/broker\/status"\)/);
  assert.match(app, /brokerConnectionView\(status, connection, nowMs, statusError, connectionError\)/);
  assert.match(app, /setInterval\(\(\) => setNowMs\(Date.now\(\)\), 1000\)/);
  assert.match(app, /clearInterval\(interval\)/);
  assert.match(app, /setNowMs\(Date.now\(\)\);\s*return observation/);
  assert.match(app, /const WorkspacePane = memo\(function WorkspacePane/);
  assert.match(app, /<WorkspacePane active=\{active\} \/>/);
  assert.doesNotMatch(app, /className="dot off"/);
  const discovery = readFileSync(new URL("../src/screens/Discovery.tsx", import.meta.url), "utf8");
  assert.match(discovery, /"Current batch progress" : "Overall progress"/);
  assert.match(discovery, /discoveryGenerationProgress\(engines\?\.discoveryCounters\)/);
  assert.match(discovery, /of generations/);
});
