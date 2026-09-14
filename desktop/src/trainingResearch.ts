import type { SavedResearchAccount, SavedTrainingResearch } from "./api.ts";

export type ResearchReadState = {
  identity: string;
  loading: boolean;
  data: SavedTrainingResearch | null;
  error: string;
};

/** One explicit read at a time; abort plus request identity also handles loaders ignoring abort. */
export function createResearchReader() {
  let current: AbortController | null = null;
  const cancel = () => {
    current?.abort();
    current = null;
  };
  return {
    cancel,
    async load(
      identity: string,
      fetcher: (identity: string, signal: AbortSignal) => Promise<SavedTrainingResearch>,
      publish: (state: ResearchReadState) => void,
    ) {
      cancel();
      const request = new AbortController();
      current = request;
      publish({ identity, loading: true, data: null, error: "" });
      try {
        const data = await fetcher(identity, request.signal);
        if (current !== request || request.signal.aborted) return;
        if (data.trainingHandoff !== identity) {
          throw new Error("Saved research response belongs to a different training handoff.");
        }
        publish({ identity, loading: false, data, error: "" });
      } catch (error) {
        if (current !== request || request.signal.aborted) return;
        publish({ identity, loading: false, data: null, error: error instanceof Error ? error.message : String(error) });
      } finally {
        if (current === request) current = null;
      }
    },
  };
}

export function researchNumber(value: number | null | undefined, digits = 2): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
    : "Unknown";
}

export function researchTimestamp(value: number): string {
  const timestamp = new Date(value);
  return Number.isFinite(timestamp.getTime()) ? timestamp.toISOString() : "Unknown";
}

export function researchUseLabel(value: string): string {
  if (value === "first_recorded_local_use_of_reserved_final_scope") return "First recorded local use";
  if (value === "reused_reserved_final_scope_research_only") return "Reused final window — research only";
  return `Unknown use classification: ${value}`;
}

/** Older combined-report responses omitted mode; never infer a model run from null metrics. */
export function researchEvaluationLabel(mode: string | undefined): string {
  if (mode === "strategy_only") return "Strategies only — no model training or inference";
  if (mode === "train_models" || mode === undefined) return "Strategies + candidate models";
  return `Unknown evaluation mode: ${mode}`;
}

/** Keep realized results and unclosed exposure separate; never derive or add missing money. */
export function researchAccountRows(geneOnly: SavedResearchAccount, combined: SavedResearchAccount | null) {
  const accounts = combined == null ? [geneOnly] : [geneOnly, combined];
  const money = (field: "netProfit" | "endingRealizedBalance" | "grossUnrealizedAccount" | "pendingRoundTripCommissionAccount" | "expectancy") =>
    accounts.map((account) => researchNumber(account[field]));
  const count = (value: number) => Number.isSafeInteger(value) && value >= 0 ? researchNumber(value, 0) : "Unknown";
  const percentage = (value: number | null) => typeof value === "number" && Number.isFinite(value * 100)
    ? `${researchNumber(value * 100)}%` : "Unknown";
  const open = (value: boolean) => value === true ? "Yes" : value === false ? "No" : "Unknown";
  return [
    { label: "Closed-trade net P&L", values: money("netProfit") },
    { label: "Closed trades", values: accounts.map((account) => count(account.tradeCount)) },
    { label: "Ending realized balance (excludes open P&L)", values: money("endingRealizedBalance") },
    { label: "Maximum modeled drawdown (%)", values: accounts.map((account) => percentage(account.maxDrawdownFraction)) },
    { label: "Position still open at window end", values: accounts.map((account) => open(account.terminalOpen)) },
    { label: "Open gross unrealized P&L (before pending commission)", values: money("grossUnrealizedAccount") },
    { label: "Open pending round-trip commission", values: money("pendingRoundTripCommissionAccount") },
    { label: "Entries skipped below broker minimum", values: accounts.map((account) => count(account.belowMinEntries)) },
    { label: "Saved Sharpe", values: accounts.map((account) => researchNumber(account.sharpe)) },
    { label: "Win rate (%)", values: accounts.map((account) => percentage(account.winRate)) },
    { label: "Saved profit factor", values: accounts.map((account) => researchNumber(account.profitFactor)) },
    { label: "Expectancy (account currency / closed trade)", values: money("expectancy") },
  ];
}
