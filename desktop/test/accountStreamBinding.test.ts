import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { accountStreamViewForScope, createAccountStreamBinding, type AccountStreamView } from "../src/accountStreamBinding.ts";
import type { AccountStreamSnap } from "../src/api.ts";
import type { BrokerAccountScope } from "../src/brokerUi.ts";

const demo42: BrokerAccountScope = { accountId: "42", environment: "Demo" };
const demo99: BrokerAccountScope = { accountId: "99", environment: "Demo" };
function snapshot(scope: BrokerAccountScope, balance = 123.5): AccountStreamSnap {
  return {
    sourceAccountId: scope.accountId, sourceEnvironment: scope.environment,
    balance, equity: balance + 1, freeMargin: balance - 2, usedMargin: 3,
    currency: "EUR", fetchedAtUnixMs: 1000, positions: [],
  };
}
function viewRecorder() {
  const views: AccountStreamView[] = [];
  return { views, publish: (view: AccountStreamView) => { views.push(view); } };
}

test("matching account snapshot preserves values; connected alone never supplies account values", () => {
  const { views, publish } = viewRecorder();
  const binding = createAccountStreamBinding(demo42, publish);
  binding.status(true);
  assert.equal(views.at(-1)?.snap, null);
  const payload = snapshot(demo42);
  binding.receive(payload);
  assert.equal(views.at(-1)?.snap, payload);
  assert.equal(views.at(-1)?.error, "");
  assert.equal(views.at(-1)?.scope?.accountId, "42");
});

test("missing or foreign account/environment clears stale values and reconnect cannot erase identity refusal", () => {
  const { views, publish } = viewRecorder();
  const binding = createAccountStreamBinding(demo99, publish);
  for (const invalid of [
    null,
    { ...snapshot(demo99), sourceAccountId: undefined } as unknown as AccountStreamSnap,
    { ...snapshot(demo99), sourceAccountId: 99 } as unknown as AccountStreamSnap,
    snapshot(demo42),
    snapshot({ ...demo99, environment: "Live" }),
  ]) {
    binding.receive(snapshot(demo99));
    assert.ok(views.at(-1)?.snap);
    binding.receive(invalid);
    assert.equal(views.at(-1)?.snap, null, "unavailable is not a zero balance or empty verified account");
    assert.match(views.at(-1)?.error ?? "", /identity.*unknown/);
    binding.status(true);
    assert.equal(views.at(-1)?.snap, null);
    assert.match(views.at(-1)?.error ?? "", /identity/);
  }
  binding.receive(snapshot(demo99, 750.5));
  assert.equal(views.at(-1)?.snap?.balance, 750.5);
  assert.equal(views.at(-1)?.error, "");
});

test("a delayed old-account completion cannot publish after unmount or masquerade on the new connection", async () => {
  const old = viewRecorder();
  const next = viewRecorder();
  const oldBinding = createAccountStreamBinding(demo42, old.publish);
  const nextBinding = createAccountStreamBinding(demo99, next.publish);
  let finish!: (payload: AccountStreamSnap) => void;
  const pending = new Promise<AccountStreamSnap>((resolve) => { finish = resolve; });
  const detachedCompletion = pending.then(oldBinding.receive);
  const lateOnNewConnection = pending.then(nextBinding.receive);
  oldBinding.receive(snapshot(demo42));
  oldBinding.stop();
  nextBinding.receive(snapshot(demo99, 999));
  const oldPublications = old.views.length;
  finish(snapshot(demo42, 42));
  await Promise.all([detachedCompletion, lateOnNewConnection]);
  assert.equal(old.views.length, oldPublications, "detached callback publishes nothing");
  assert.equal(next.views.at(-1)?.snap, null, "bridge's old-account broadcast is not accepted by new view");
  assert.match(next.views.at(-1)?.error ?? "", /identity/);
  oldBinding.status(true);
  assert.equal(old.views.length, oldPublications);
  nextBinding.receive(snapshot(demo99, 1001));
  assert.equal(next.views.at(-1)?.snap?.balance, 1001);
});

test("render hides retained account before effect cleanup on unknown status, account change or environment change", () => {
  const { views, publish } = viewRecorder();
  const binding = createAccountStreamBinding(demo42, publish);
  binding.receive(snapshot(demo42));
  const saved = views.at(-1)!;
  assert.equal(accountStreamViewForScope(saved, { ...demo42 }), saved);
  for (const scope of [null, demo99, { ...demo42, environment: "Live" as const }]) {
    const hidden = accountStreamViewForScope(saved, scope);
    assert.equal(hidden.snap, null);
    assert.equal(hidden.connected, false);
    assert.equal(hidden.error, "", "broker setup status is rendered by the existing wrapper");
  }
});

test("string identities above JavaScript safe integer retain exact distinction", () => {
  const { views, publish } = viewRecorder();
  const scope = { ...demo42, accountId: "9007199254740993" };
  const binding = createAccountStreamBinding(scope, publish);
  binding.receive(snapshot({ ...scope, accountId: "9007199254740992" }));
  assert.equal(views.at(-1)?.snap, null);
  binding.receive(snapshot(scope));
  assert.equal(views.at(-1)?.snap?.sourceAccountId, "9007199254740993");
});

test("transport failures remain visible even when an identity-matching payload arrives", () => {
  const { views, publish } = viewRecorder();
  const binding = createAccountStreamBinding(demo42, publish);
  binding.status(false, "synthetic connection failure");
  binding.receive(snapshot(demo42));
  assert.equal(views.at(-1)?.error, "synthetic connection failure");
  binding.status(true);
  assert.equal(views.at(-1)?.error, "");
});

test("the account hook consumes the binding with cleanup and explicit current scope without extra polling", () => {
  const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");
  const hooks = source("../src/hooks.ts");
  const hook = hooks.slice(hooks.indexOf("export function useAccountStream("), hooks.indexOf("/**\n * Fetch once"));
  assert.match(hook, /if \(accountId === null \|\| environment === null\) return/);
  assert.match(hook, /createAccountStreamBinding\(\{ accountId, environment \}, setView\)/);
  assert.match(hook, /streamAccount\(binding.receive, binding.status\)/);
  assert.match(hook, /binding.stop\(\);\s*close\(\)/);
  assert.match(hook, /else c\(\)/);
  assert.match(hook, /\[accountId, environment\]/);
  assert.match(hook, /return accountStreamViewForScope\(view, scope\)/);
  assert.doesNotMatch(hook, /setInterval|refreshAccount|brokerStatus|apiPost/);
  const cockpit = source("../src/screens/Cockpit.tsx");
  assert.match(cockpit, /useAccountStream\(access.scope\)/);
  assert.match(cockpit, /Open positions are unknown until an account snapshot is received/);
});
