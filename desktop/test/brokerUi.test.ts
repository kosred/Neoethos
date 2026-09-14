import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { brokerUiAccess } from "../src/brokerUi.ts";
import type { BrokerStatus } from "../src/api.ts";

const configured: BrokerStatus = { configured: true, hasToken: true, accountId: "42", environment: "Demo" };
const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");

test("unknown status and failed status reads are not reported as unconfigured", () => {
  const checking = brokerUiAccess(null);
  assert.equal(checking.phase, "checking");
  assert.equal(checking.requestsEnabled, false);
  assert.equal(checking.scope, null);
  assert.equal(checking.setupAvailable, false);
  const detail = "Status read failed: configuration is unavailable";
  for (const retained of [null, configured, { ...configured, configured: false }]) {
    const failed = brokerUiAccess(retained, detail);
    assert.equal(failed.phase, "unavailable");
    assert.equal(failed.requestsEnabled, false);
    assert.equal(failed.scope, null);
    assert.equal(failed.detail, detail);
  }
});

test("only an explicit configured=false becomes the unconfigured empty state", () => {
  const unavailable = brokerUiAccess({ ...configured, configured: false });
  assert.equal(unavailable.phase, "unconfigured");
  assert.equal(unavailable.requestsEnabled, false);
  assert.equal(unavailable.setupAvailable, true);
  assert.match(unavailable.detail, /local research and journal data remain readable/);
  assert.doesNotMatch(unavailable.detail, /could not reach|connection failed|balance.*0/i);
  assert.equal(brokerUiAccess({ ...configured, configured: undefined } as unknown as BrokerStatus).phase, "unavailable");
});

test("missing token and missing account remain distinct setup states, without enabling requests", () => {
  const token = brokerUiAccess({ ...configured, hasToken: false });
  const account = brokerUiAccess({ ...configured, accountId: null });
  assert.equal(token.phase, "missing_token");
  assert.equal(account.phase, "missing_account");
  for (const access of [token, account, brokerUiAccess({ ...configured, accountId: " " })]) {
    assert.equal(access.requestsEnabled, false);
    assert.equal(access.setupAvailable, true);
  }
  assert.match(token.detail, /does not start authentication/);
});

test("configured setup allows existing request/error paths but proves no broker connectivity or authority", () => {
  const access = brokerUiAccess(configured);
  assert.equal(access.phase, "configured");
  assert.equal(access.requestsEnabled, true);
  assert.deepEqual(access.scope, { accountId: "42", environment: "Demo" });
  assert.equal(access.setupAvailable, false);
  assert.equal("connected" in access, false);
  assert.equal("tradingAllowed" in access, false);
});

test("account or environment changes replace retained broker-view state", () => {
  const first = brokerUiAccess(configured);
  assert.equal(first.key, brokerUiAccess({ ...configured }).key);
  assert.notEqual(first.key, brokerUiAccess({ ...configured, accountId: "99" }).key);
  assert.notEqual(first.key, brokerUiAccess({ ...configured, environment: "Live" }).key);
  assert.notEqual(first.key, brokerUiAccess({ ...configured, configured: false }).key);
  assert.equal(brokerUiAccess({ ...configured, accountId: 42 } as unknown as BrokerStatus).requestsEnabled, false);
  assert.equal(brokerUiAccess({ ...configured, environment: "" }).requestsEnabled, false);
  assert.equal(brokerUiAccess({ ...configured, environment: "unrecognised" }).scope, null);
});

test("Market's structural mount boundary precedes every broker stream, chart and polling hook", () => {
  const cockpit = source("../src/screens/Cockpit.tsx");
  const wrapper = cockpit.slice(cockpit.indexOf("export default function Cockpit()"), cockpit.indexOf("function ConfiguredCockpit()"));
  assert.match(wrapper, /if \(!access\.requestsEnabled\) return/);
  assert.match(wrapper, /<BrokerUnavailable \/>/);
  assert.match(wrapper, /disabled>Place order/);
  assert.match(wrapper, /<ConfiguredCockpit key=\{access\.key\} \/>/);
  assert.doesNotMatch(wrapper, /useSpotStream|useAccountStream|usePoll|serverSymbols|<KChart|placeOrder\(/);
  assert.match(cockpit, /usePoll\(refreshAccount, 5000\)/);
});

test("Performance gates only broker reads and margin while retaining local journal reads", () => {
  const account = source("../src/screens/Account.tsx");
  for (const fetcher of ["brokerProfile", "brokerVersion", "ordersHistory", "cashFlow"]) {
    assert.ok(account.includes("usePoll(" + fetcher + ", 0, brokerKey, brokerEnabled)"));
  }
  for (const fetcher of ["journalStats", "journalTrades", "journalAnalytics"]) {
    assert.ok(account.includes("usePoll(" + fetcher + ", 0)"));
  }
  assert.match(account, /<AccountContent key=\{access\.key\}/);
  assert.match(account, /if \(!brokerEnabled \|\| marginBusy \|\| !marginValid\) return/);
  assert.match(account, /disabled=\{!brokerEnabled \|\| marginBusy \|\| !marginValid\} onClick=\{calcMargin\}/);
  assert.match(account, /Refresh local journal/);
  assert.match(account, /Broker order history is unknown until broker setup is available/);
  assert.match(account, /Broker cash-flow history is unknown until broker setup is available/);
  assert.match(account, /Missing records are not proof of zero profit or loss/);
  assert.match(account, /Broker profile unavailable: \{pe\}/);
  assert.match(account, /setMErr\(String\(e\)\)/);
});

test("the optional polling guard covers manual reload, interval creation, stale completions and hidden data", () => {
  const hooks = source("../src/hooks.ts");
  assert.match(hooks, /enabled = true/);
  assert.match(hooks, /const reload = useCallback\(\(\) => lifecycleRef\.current\.run\(/);
  assert.match(hooks, /lifecycle\.start\(\);\s*requests\.current =/);
  assert.match(hooks, /return \(\) => \{\s*lifecycle\.stop\(\);\s*requests\.current =/);
  assert.match(hooks, /if \(!enabledRef\.current\) return Promise\.resolve\(\)/);
  assert.match(hooks, /const isCurrent = \(\) => lifecycleRef\.current\.isActive\(\) && enabledRef\.current && group === requests\.current/);
  assert.match(hooks, /if \(enabled\) void reload\(\)/);
  assert.match(hooks, /if \(enabled && intervalMs > 0\) id = setInterval/);
  assert.match(hooks, /\[reload, intervalMs, dependencyKey, enabled\]/);
  assert.match(hooks, /if \(!enabled\) return \{ data: null, error: "", loading: false, reload \}/);
});

test("the shell reuses its status read and setup navigation does not invoke authentication", () => {
  const app = source("../src/App.tsx");
  assert.equal((app.match(/usePoll\(brokerStatus, 5000\)/g) ?? []).length, 1);
  assert.match(app, /access: brokerUiAccess\(status, statusError\)/);
  assert.match(app, /<BrokerUiContext.Provider value=\{brokerUi\}>/);
  assert.match(app, /const openBrokerSetup = useCallback\(\(\) => setView\("settings"\), \[\]\)/);
  const empty = source("../src/components/BrokerUnavailable.tsx");
  assert.match(empty, /onClick=\{openSetup\}>Open Broker Setup/);
  assert.doesNotMatch(empty, /reauth|brokerAccounts|selectAccount|fetch\(|apiPost|invoke\(/);
  const settings = source("../src/screens/Settings.tsx");
  const mount = settings.slice(settings.indexOf("  useEffect(() => {"), settings.indexOf("  // ── cTrader API credentials"));
  assert.match(mount, /void refresh\(\)/);
  assert.doesNotMatch(mount, /doReauth|loadAccounts|loadCreds|activateAccount/);
});
