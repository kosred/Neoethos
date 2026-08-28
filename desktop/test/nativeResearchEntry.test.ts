import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  boundedNativeResearchText,
  nativeResearchFailureSummary,
  nativeResearchPublishedSummary,
  nativeResearchStartBody,
  type CanonicalNativeResearchStatus,
} from "../src/nativeResearch.ts";

const apiSource = readFileSync(new URL("../src/api.ts", import.meta.url), "utf8");
const appSource = readFileSync(new URL("../src/App.tsx", import.meta.url), "utf8");
const screenSource = readFileSync(
  new URL("../src/screens/NativeResearch.tsx", import.meta.url),
  "utf8",
);
const tauriSource = readFileSync(
  new URL("../src-tauri/src/lib.rs", import.meta.url),
  "utf8",
);

test("native start body preserves the exact artifact reference and explicit overrides", () => {
  const body = nativeResearchStartBody(
    "contracts/EURUSD-M1.json",
    "ab".repeat(32),
    4096,
    false,
    17,
  );

  assert.equal(
    JSON.stringify(body),
    JSON.stringify({
      contractArtifact: {
        relativePath: "contracts/EURUSD-M1.json",
        expectedSha256: "ab".repeat(32),
      },
      population: 4096,
      populationAuto: false,
      maxIndicators: 17,
    }),
  );
});

test("failure display keeps the stable stage and bounds untrusted detail", () => {
  const status: CanonicalNativeResearchStatus = {
    state: "Failed",
    stage: "generation_zero_evaluation",
    percent: 12.5,
    leaseToken: null,
    cancellationRequested: false,
    failureStage: "GenerationZeroEvaluation",
    failureCode: "evaluation_failed",
    failureDetail: "x".repeat(2_000),
    published: null,
  };

  const summary = nativeResearchFailureSummary(status);
  assert.match(summary, /GenerationZeroEvaluation/);
  assert.match(summary, /evaluation_failed/);
  assert.ok(summary.length <= 640, `failure summary was ${summary.length} characters`);
  assert.equal(boundedNativeResearchText("abc", 8), "abc");
});

test("published display exposes the bounded canonical evidence summary", () => {
  const summary = nativeResearchPublishedSummary({
    relativePath: "generation-zero/result.json",
    byteCount: 1234,
    fileSha256: "cd".repeat(32),
    evidenceIdentitySha256: "ef".repeat(32),
    configuredPopulation: 2048,
    resolvedPopulation: 1024,
    populationCap: 1024,
    hardGrowthCap: 1024,
    termCap: 96,
    selectedDeviceOrdinal: 0,
    engine: "cuda",
    parentH2dBytes: 40,
    adaptiveH2dBytes: 2,
    metricRows: 18,
    metricBytes: 144,
    consumerCompletionConfirmed: true,
    replayIdentitySealed: true,
  });

  for (const expected of [
    "generation-zero/result.json",
    "resolved P=1024",
    "hard cap=1024",
    "T=96",
    "device=0",
    "consumer=yes",
    "replay=yes",
  ]) {
    assert.match(summary, new RegExp(expected.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
  }
  assert.ok(summary.length <= 800, `published summary was ${summary.length} characters`);
});

test("desktop start cancel and status stay on the canonical native lane", () => {
  for (const marker of [
    "canonicalNativeResearchStart",
    "canonicalNativeResearchCancel",
    "canonicalNativeResearch",
    '"/engines/native-research/start"',
    '"/engines/native-research/cancel"',
  ]) {
    assert.match(apiSource + screenSource, new RegExp(marker));
  }
  for (const retired of [
    "discoveryStart(",
    "discoveryStop(",
    "trainingStart(",
    "trainingStop(",
  ]) {
    assert.ok(!screenSource.includes(retired), `native screen still invokes ${retired}`);
  }
  assert.match(appSource, /id: "nativeResearch"/);
  assert.match(appSource, /<NativeResearch \/>/);
});

test("Tauri close guard treats a queued native worker as active", () => {
  const guardStart = tauriSource.indexOf("pub fn engine_running() -> bool");
  const guardEnd = tauriSource.indexOf("\n}\n\n/// Base URL", guardStart);
  assert.notEqual(guardStart, -1, "missing close guard");
  assert.notEqual(guardEnd, -1, "missing close guard boundary");
  const guard = tauriSource.slice(guardStart, guardEnd);
  assert.match(guard, /body\.contains\("\\\"Running"\)/);
  assert.match(guard, /body\.contains\("\\\"Queued"\)/);
});
