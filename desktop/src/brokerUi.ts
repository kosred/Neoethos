import type { BrokerStatus } from "./api";

export type BrokerAccountScope = Readonly<{ accountId: string; environment: "Demo" | "Live" }>;

export type BrokerUiAccess = Readonly<{
  phase: "checking" | "unavailable" | "unconfigured" | "missing_token" | "missing_account" | "configured";
  requestsEnabled: boolean;
  scope: BrokerAccountScope | null;
  key: string;
  title: string;
  detail: string;
  setupAvailable: boolean;
}>;

/** Request/display availability only: stored setup is not broker connectivity or trading authority. */
export function brokerUiAccess(status: BrokerStatus | null, error = ""): BrokerUiAccess {
  const unavailable = (phase: BrokerUiAccess["phase"], title: string, detail: string, setupAvailable = false): BrokerUiAccess => ({
    phase, requestsEnabled: false, scope: null, key: "inactive", title, detail, setupAvailable,
  });
  // A failed read must not reuse a retained status as evidence of either configuration state.
  if (error) return unavailable("unavailable", "Broker setup status unavailable", error);
  if (!status) return unavailable("checking", "Checking broker setup…", "Broker data requests are paused while setup is checked.");
  if (status.configured === false) return unavailable(
    "unconfigured", "Broker is not configured",
    "Configure cTrader in Settings → General → Broker connection. Broker data and trading controls are unavailable; local research and journal data remain readable.",
    true,
  );
  if (status.configured !== true || typeof status.hasToken !== "boolean") {
    return unavailable("unavailable", "Broker setup status unavailable", "The setup response is incomplete.");
  }
  if (!status.hasToken) return unavailable(
    "missing_token", "No broker access token reported",
    "Open Broker Setup to inspect authentication. This screen does not start authentication.",
    true,
  );
  if (status.accountId == null || (typeof status.accountId === "string" && !status.accountId.trim())) return unavailable(
    "missing_account", "No broker account selected",
    "Select an account in Broker Setup before requesting account or market data.",
    true,
  );
  if (typeof status.accountId !== "string" || (status.environment !== "Demo" && status.environment !== "Live")) {
    return unavailable("unavailable", "Broker setup status unavailable", "The broker account or environment is invalid.");
  }
  return {
    phase: "configured", requestsEnabled: true,
    scope: { accountId: status.accountId, environment: status.environment },
    key: JSON.stringify([status.environment, status.accountId]),
    title: "", detail: "", setupAvailable: false,
  };
}
