import assert from "node:assert/strict";
import test from "node:test";
import { discoveryInventoryPage } from "../src/discoveryInventory.ts";
import type { DatasetInventoryEntry } from "../src/apiContracts.ts";

const entry = (index: number, symbol: string | null = "EURUSD", timeframe: string | null = "M5"): DatasetInventoryEntry => Object.freeze({
  datasetIdentity: `d1-test-${index}` as DatasetInventoryEntry["datasetIdentity"],
  generation: `g1-exact-${index}`, manifestBindingSha256: `binding-${index}`,
  sourceKind: "ctrader", symbol, timeframe, verification: "generation_verified",
});

test("dataset filtering matches all words and the exact timeframe without decoding identities", () => {
  const entries = [entry(1), entry(2, "EURUSD", "M15"), entry(3, "GBPUSD")];
  assert.deepEqual(discoveryInventoryPage(entries, " eurUSD  cTrader ", "M5", 0).entries, [entries[0]]);
  assert.deepEqual(discoveryInventoryPage(entries, "binding-2", "", 0).entries, [entries[1]]);
  assert.equal(discoveryInventoryPage(entries, "EURUSD", "M1", 0).matched, 0);
  assert.equal(discoveryInventoryPage([entry(4, null, null)], "", "", 0).matched, 1);
});

test("pagination bounds rendered rows and clamps after a smaller inventory refresh", () => {
  const entries = Object.freeze(Array.from({ length: 178 }, (_, index) => entry(index)));
  const last = discoveryInventoryPage(entries, "", "", 8);
  assert.equal(last.pageCount, 9);
  assert.equal(last.entries.length, 18);
  assert.equal(last.entries[0], entries[160]);
  const refreshed = discoveryInventoryPage(entries.slice(0, 3), "", "", 8);
  assert.equal(refreshed.page, 0);
  assert.equal(refreshed.entries.length, 3);
  assert.equal(discoveryInventoryPage(entries, "", "", NaN).page, 0);
});

test("filtering does not replace a pinned generation or mutate the inventory", () => {
  const pinned = entry(7);
  const replacement = Object.freeze({ ...pinned, generation: "g1-new-generation" });
  const entries = Object.freeze([replacement]);
  assert.equal(discoveryInventoryPage(entries, "GBPUSD", "M5", 0).matched, 0);
  assert.equal(pinned.generation, "g1-exact-7");
  assert.equal(entries[0], replacement);
  assert.equal(discoveryInventoryPage(entries, "", "", 0).entries[0], replacement);
});
