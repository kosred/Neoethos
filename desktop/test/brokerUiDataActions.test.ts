import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

test("Data keeps local inventory/import/recorded diagnostics and exact-run stop independent of broker setup", () => {
  const data = source("../src/screens/Data.tsx");
  assert.match(data, /usePoll\(dataBootstrap, 0\)/);
  assert.match(data, /const localSyms = useSymbolOptions\(\)/);
  assert.match(data, /<DataImportPanel onImported=\{reload\} \/>/);
  assert.match(data, /usePoll\(spreadStats, 60000\)/);
  assert.match(data, /usePoll\(dataFetchStatus, busy \? 250 : 2000\)/);
  const stop = data.slice(data.indexOf("  const stopFetch ="), data.indexOf("  const nCombos ="));
  assert.match(stop, /stopActiveDataFetch\(fetchStatus\.runId\)/);
  assert.doesNotMatch(stop, /requestsEnabled|brokerAccessRef|brokerEnabled/);
});

test("Data guards catalog, cost refresh and every future batch dispatch without replacing backend error details", () => {
  const data = source("../src/screens/Data.tsx");
  assert.match(data, /usePoll\(serverSymbols, 0, access\.key, access\.requestsEnabled\)/);
  assert.match(data, /if \(!access\.requestsEnabled \|\| costBusy\) return/);
  assert.match(data, /disabled=\{!access\.requestsEnabled \|\| costBusy\} onClick=\{refreshCosts\}/);
  assert.match(data, /if \(!access\.requestsEnabled \|\| busy \|\| !data/);
  assert.match(data, /const batchBrokerKey = access\.key/);
  assert.match(data, /if \(!currentBroker\?\.requestsEnabled \|\| currentBroker\.key !== batchBrokerKey\)/);
  assert.match(data, /return \(\) => \{ brokerAccessRef\.current = null; \}/);
  assert.ok(data.indexOf("currentBroker.key !== batchBrokerKey") < data.indexOf("const outcome = await dataFetch("));
  assert.match(data, /costMsg && costKey === access\.key/);
  assert.match(data, /Already-dispatched work is not cancelled by this check/);
  assert.match(data, /dataOperationErrorText\(e\)/);
  assert.match(data, /Broker symbol catalog unavailable; only known local symbols are shown\. \{brokerError\}/);
});

test("Actions disables only broker operations while keeping the local approval queue and rejections available", () => {
  const actions = source("../src/screens/Actions.tsx");
  assert.match(actions, /<ActionsContent key=\{access\.key\}/);
  assert.match(actions, /usePoll\(brokerPendingOrders, 5000, brokerKey, brokerEnabled\)/);
  assert.match(actions, /useSpotStream\(brokerEnabled\)/);
  assert.match(actions, /usePoll\(pendingActions, 3000\)/);
  assert.match(actions, /<fieldset disabled=\{!brokerEnabled\}/);
  assert.ok(actions.indexOf("</fieldset>") < actions.indexOf("{/* ── AI-proposed actions"));
  for (const start of ["const submit =", "const saveEdit =", "const cancel ="]) {
    const block = actions.slice(actions.indexOf(start), actions.indexOf(start) + 170);
    assert.match(block, /if \(!brokerEnabled(?: \|\| busy)?\) return/);
  }
  assert.match(actions, /if \(ok && action\?\.kind\.kind === "close_position" && !brokerEnabled\) return/);
  assert.match(actions, /disabled=\{busy \|\| \(!brokerEnabled && action\.kind\.kind === "close_position"\)\}/);
  assert.match(actions, /className="danger" disabled=\{busy\} onClick=\{\(\) => decide\(action\.id, false\)\}/);
  assert.match(actions, /await rejectAction\(id\)/);
  assert.match(actions, /\{pErr && <div className="banner warn">\{String\(pErr\)\}/);
});

test("a disabled quote hook never opens its event source and existing cleanup closes deferred opens", () => {
  const hooks = source("../src/hooks.ts");
  const stream = hooks.slice(hooks.indexOf("export function useSpotStream"), hooks.indexOf("/** Live account snapshot"));
  assert.match(stream, /useSpotStream\(enabled = true\)/);
  assert.ok(stream.indexOf("if (!enabled) return;") < stream.indexOf("    streamSpots("));
  assert.match(stream, /else c\(\)/);
  assert.match(stream, /alive = false;\s*close\(\)/);
  assert.match(stream, /const visibleTicks: Record<string, Tick> = enabled \? ticks : \{\}/);
  assert.match(stream, /\}, \[enabled\]\)/);
  const options = source("../src/components/selectOptions.ts");
  const local = options.slice(options.indexOf("const loadSymbols ="), options.indexOf("const loadTimeframes ="));
  assert.match(local, /dataBootstrap\(\)/);
  assert.doesNotMatch(local, /serverSymbols|brokerAccounts|invoke\(/);
});
