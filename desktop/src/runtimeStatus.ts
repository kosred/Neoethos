import type { BrokerStatus, EngineRunState } from "./api";

/** Display only: completion with limitations is neither full success nor a crash. */
export function researchEngineStatus(state: EngineRunState | "Unknown") {
  const degraded = state === "Degraded";
  return {
    label: degraded ? "Degraded — completed with limitations" : state,
    badgeClass: state === "Running" ? "live" : state === "Succeeded" ? "demo" : "",
    warning: degraded || state === "Failed",
    notice: degraded
      ? "The job completed with warnings or errors. Inspect the reported reason and saved results; this is not full success or trading approval."
      : "",
  };
}

/** Account-channel observation, not a quote-freshness or trading permit. */
export type BrokerConnectionObservation = {
  environment: string;
  accountId: string;
  connected: boolean;
  // Older backends have no freshness proof. Credentials alone cannot replace it.
  lastSnapshotAtUnixMs?: number | null;
};

export function brokerConnectionView(
  credentials: BrokerStatus | null,
  observation: BrokerConnectionObservation | null,
  nowMs: number,
  credentialsError = "",
  observationError = "",
): { connected: boolean; label: string; detail: string } {
  const off = (label: string, detail: string) => ({ connected: false, label, detail });
  if (credentialsError) return off("status unavailable", credentialsError);
  if (!credentials) return off("checking…", "Reading broker configuration.");
  if (!credentials.configured) return off("not configured", "Broker credentials are not configured.");
  const environment = credentials.environment;
  if (!credentials.accountId) return off(`${environment} · no account`, "Select an execution account.");
  if (!credentials.hasToken) return off(`${environment} · needs auth`, "No saved access token is available.");
  if (observationError) return off(`${environment} · status unavailable`, observationError);
  if (!observation) return off(`${environment} · checking connection`, "Stored credentials do not prove an active connection.");
  if (observation.accountId !== credentials.accountId || observation.environment !== environment) {
    return off(`${environment} · account not confirmed`, "The backend observation belongs to a different account or environment.");
  }
  const observedAt = observation.lastSnapshotAtUnixMs;
  if (typeof observedAt !== "number" || !Number.isSafeInteger(observedAt) || observedAt <= 0
    || !Number.isSafeInteger(nowMs) || nowMs < observedAt) {
    return off(`${environment} · connection unverified`, "No valid broker account observation timestamp is available. Use the matching backend build.");
  }
  // Mirrors the backend's display-only 15-second horizon. Re-evaluate against
  // a local clock even if the next request hangs; never refresh age on receipt.
  const ageMs = nowMs - observedAt;
  if (ageMs > 15_000) return off(`${environment} · account data stale`, "The last broker account response is more than 15 seconds old.");
  if (observation.connected !== true) return off(`${environment} · account disconnected`, "The backend has not confirmed a healthy account connection.");
  return {
    connected: true,
    label: `${environment} · account connected`,
    detail: `Broker account response ${Math.floor(ageMs / 1000)}s ago. Quote freshness and trading readiness are separate.`,
  };
}

export function discoveryGenerationProgress(
  counters?: readonly { name: string; value: number }[] | null,
): { completed: number; total: number; percent: number } | undefined {
  const completed = counters?.filter((counter) => counter.name === "generation");
  const total = counters?.filter((counter) => counter.name === "generations");
  if (completed?.length !== 1 || total?.length !== 1) return undefined;
  const done = completed[0].value;
  const limit = total[0].value;
  if (!Number.isSafeInteger(done) || !Number.isSafeInteger(limit)
    || done < 0 || limit <= 0 || done > limit) return undefined;
  return { completed: done, total: limit, percent: (done / limit) * 100 };
}

export type DiscoveryCounterRow = {
  name: string;
  label: string;
  value: number | null;
};

// Display the producer's observations, never derive "failed OOS" from a
// population plan minus a small final portfolio. Missing stages are unknown.
const DISCOVERY_COUNTER_LABELS: Readonly<Record<string, string>> = {
  working_set_batch: "Current batch",
  working_set_completed_batches: "Completed batches",
  working_set_completed_entries: "Selection entries in completed batches",
  working_set_total_entries: "Total selection entries",
  working_set_saved_results: "Saved batch research reports",
  working_set_training_handoffs: "Results available for final evaluation",
  working_set_publication_failures: "Portfolio/handoff publication failures",
  population: "Population",
  generations: "Generation limit",
  generation: "Generations completed",
  target_candidates: "Post-search validation target (0 = all GA-returned candidates)",
  planned_ga_evaluations: "Planned GA evaluations (not generated candidates)",
  ga_returned_candidates: "GA returned candidate pool",
  validation_candidate_limit: "Post-search validation limit (0 = all GA-returned candidates)",
  validation_candidates_admitted: "Candidates admitted to validation",
  validation_candidates_capped: "Not tested — validation budget cap",
  walkforward_tested: "Walk-forward candidates tested",
  walkforward_passed: "Walk-forward candidates passed",
  walkforward_failed: "Walk-forward candidates failed",
  walkforward_not_tested: "Walk-forward candidates not tested",
  portfolio_selected: "Selected at reported stage",
  portfolio_capacity_not_selected: "Not selected — portfolio capacity",
  robustness_removed: "Removed by robustness checks",
  correlation_tested: "Candidates checked for correlation",
  archived_profitable: "Profitable candidates archived",
  stagnant_generations: "Generations without improvement",
  candidates: "Candidates admitted after ranking",
  filtered_candidates: "Base-filter survivors",
  min_trades_required: "Minimum trades required",
  quality_screened: "Quality survivors (not evaluations)",
  quality_evaluated: "Full in-sample quality backtests completed",
  quality_scored: "Candidates with quality records",
  opportunistic_candidates: "Opportunistic quality survivors",
  trade_logs: "Saved trade logs",
  portfolio: "Portfolio at reported stage",
  rejected: "Not selected (not an OOS failure count)",
  rejected_by_correlation: "Rejected by correlation",
};

export function discoveryCounterRows(
  counters?: readonly { name: string; value: number }[] | null,
): DiscoveryCounterRow[] {
  const byName = new Map<string, number[]>();
  for (const counter of counters ?? []) {
    const values = byName.get(counter.name) ?? [];
    values.push(counter.value);
    byName.set(counter.name, values);
  }
  return [...byName].map(([name, values]) => ({
    name,
    label: typeof DISCOVERY_COUNTER_LABELS[name] === "string"
      ? DISCOVERY_COUNTER_LABELS[name]
      : name.replaceAll("_", " ").replace(/\bcpcv\b/g, "CPCV").replace(/\boos\b/g, "OOS"),
    value: values.length === 1 && Number.isSafeInteger(values[0]) && values[0] >= 0
      ? values[0]
      : null,
  }));
}
