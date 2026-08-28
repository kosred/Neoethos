import assert from "node:assert/strict";
import test from "node:test";

import { queueTerminalOutcome } from "../src/discoveryQueueState.ts";

test("queue preserves every backend terminal outcome", () => {
  assert.deepEqual(queueTerminalOutcome("Succeeded", "portfolio saved"), {
    status: "done",
    note: "portfolio saved",
  });
  assert.deepEqual(queueTerminalOutcome("Failed", "data receipt is stale"), {
    status: "failed",
    note: "data receipt is stale",
  });
  assert.deepEqual(queueTerminalOutcome("Cancelled", ""), {
    status: "failed",
    note: "cancelled",
  });
  assert.equal(queueTerminalOutcome("Running", "still working"), null);
  assert.equal(queueTerminalOutcome("Idle", ""), null);
});
