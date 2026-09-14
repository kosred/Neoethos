import { useEffect, useMemo, useState } from "react";
import {
  settings as getSettings,
  updateSettings,
  type SettingsUpdate,
  type SettingsView,
} from "../api";

type NumericDraft = {
  riskyStartBalance: string;
  riskyTargetBalance: string;
  riskyHorizonDays: string;
  sharedSearchRiskMinPct: string;
  sharedSearchRiskMaxPct: string;
  riskySearchRiskMinPct: string;
  riskySearchRiskMaxPct: string;
  propFirmSearchRiskMinPct: string;
  propFirmSearchRiskMaxPct: string;
  searchHighQualityConfidencePct: string;
  propFirmSearchProfitTargetPct: string;
  propFirmSearchMaxDailyLossPct: string;
  propFirmSearchMaxDrawdownPct: string;
  propFirmSearchMinTradingDays: string;
  propFirmSearchWindowDays: string;
  propFirmSearchWindowCount: string;
  propFirmSearchPassRatePct: string;
  searchPopulation: string;
  searchGenerations: string;
  searchMaxHours: string;
  searchMaxIndicators: string;
  searchPortfolioSize: string;
  searchCorrThresholdPct: string;
  searchMaxRows: string;
  prefilterTopK: string;
  convergencePatience: string;
  stagnationPatience: string;
  noveltyWeightPct: string;
};

type NumericDraftKey = keyof NumericDraft;
type DiscoveryDraft = NumericDraft & {
  tradingMode: "risky" | "prop_firm";
  searchDevice: string;
  searchPopulationAuto: boolean;
  disableSmcGate: boolean;
};

type SearchDevice = NonNullable<SettingsUpdate["searchDevice"]>;
const SEARCH_DEVICES: readonly SearchDevice[] = ["", "auto", "cpu", "cuda_required"];
const isSearchDevice = (value: string): value is SearchDevice =>
  SEARCH_DEVICES.some((candidate) => candidate === value);

const percentText = (fraction: number) => String(Number((fraction * 100).toFixed(6)));

const draftFromSettings = (config: SettingsView): DiscoveryDraft => ({
  tradingMode: config.tradingMode === "risky" ? "risky" : "prop_firm",
  searchDevice: config.searchDevice,
  riskyStartBalance: String(config.riskyStartBalance),
  riskyTargetBalance: String(config.riskyTargetBalance),
  riskyHorizonDays: String(config.riskyHorizonDays),
  sharedSearchRiskMinPct: percentText(config.sharedSearchRiskMin),
  sharedSearchRiskMaxPct: percentText(config.sharedSearchRiskMax),
  riskySearchRiskMinPct: percentText(config.riskySearchRiskMin),
  riskySearchRiskMaxPct: percentText(config.riskySearchRiskMax),
  propFirmSearchRiskMinPct: percentText(config.propFirmSearchRiskMin),
  propFirmSearchRiskMaxPct: percentText(config.propFirmSearchRiskMax),
  searchHighQualityConfidencePct: percentText(config.searchHighQualityConfidence),
  propFirmSearchProfitTargetPct: percentText(config.propFirmSearchProfitTargetPct),
  propFirmSearchMaxDailyLossPct: percentText(config.propFirmSearchMaxDailyLossPct),
  propFirmSearchMaxDrawdownPct: percentText(config.propFirmSearchMaxDrawdownPct),
  propFirmSearchMinTradingDays: String(config.propFirmSearchMinTradingDays),
  propFirmSearchWindowDays: String(config.propFirmSearchWindowDays),
  propFirmSearchWindowCount: String(config.propFirmSearchWindowCount),
  propFirmSearchPassRatePct: percentText(config.propFirmSearchPassRate),
  searchPopulation: String(config.searchPopulation),
  searchGenerations: String(config.searchGenerations),
  searchMaxHours: String(config.searchMaxHours),
  searchMaxIndicators: String(config.searchMaxIndicators),
  searchPortfolioSize: String(config.searchPortfolioSize),
  searchCorrThresholdPct: percentText(config.searchCorrThreshold),
  searchMaxRows: String(config.searchMaxRows),
  prefilterTopK: String(config.prefilterTopK),
  convergencePatience: String(config.convergencePatience),
  stagnationPatience: String(config.stagnationPatience),
  noveltyWeightPct: percentText(config.noveltyWeight),
  searchPopulationAuto: config.searchPopulationAuto,
  disableSmcGate: config.disableSmcGate,
});

const asFinite = (value: string, label: string): number => {
  const parsed = Number(value);
  if (value.trim() === "" || !Number.isFinite(parsed)) {
    throw new Error(`${label} must be a number.`);
  }
  return parsed;
};

const asNonNegativeInteger = (value: string, label: string): number => {
  const parsed = asFinite(value, label);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative whole number.`);
  }
  return parsed;
};

const asPercentFraction = (value: string, label: string): number => {
  const parsed = asFinite(value, label);
  if (parsed < 0 || parsed > 100) {
    throw new Error(`${label} must be between 0% and 100%.`);
  }
  return parsed / 100;
};

export function DiscoveryParameters({ onReadinessChange }: { onReadinessChange?: (ready: boolean) => void }) {
  const [config, setConfig] = useState<SettingsView | null>(null);
  const [draft, setDraft] = useState<DiscoveryDraft | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [loadVersion, setLoadVersion] = useState(0);

  useEffect(() => {
    let active = true;
    void getSettings()
      .then((loaded) => {
        if (!active) return;
        setConfig(loaded);
        setDraft(draftFromSettings(loaded));
      })
      .catch((error: unknown) => {
        if (active) setMessage(`Could not read Discovery parameters: ${String(error)}`);
      });
    return () => {
      active = false;
    };
  }, [loadVersion]);

  const plannedCandidateBudget = useMemo(() => {
    if (!draft) return 0;
    const population = Number(draft.searchPopulation);
    const generations = Number(draft.searchGenerations);
    return Number.isFinite(population) && Number.isFinite(generations)
      ? Math.max(0, population) * Math.max(0, generations)
      : 0;
  }, [draft]);
  const draftDirty = useMemo(
    () => config !== null && draft !== null
      && JSON.stringify(draft) !== JSON.stringify(draftFromSettings(config)),
    [config, draft],
  );

  useEffect(() => {
    onReadinessChange?.(config !== null && draft !== null && !busy && !draftDirty);
  }, [onReadinessChange, config, draft, busy, draftDirty]);

  if (!config || !draft) {
    return (
      <section aria-labelledby="discovery-parameters-heading">
        <h2 id="discovery-parameters-heading">Objective &amp; search constraints</h2>
        <div className="ticket">{message || "Reading the exact persisted Discovery configuration…"}</div>
        {message && <button onClick={() => { setMessage(""); setLoadVersion((value) => value + 1); }}>Retry loading parameters</button>}
      </section>
    );
  }

  const setNumber = (key: NumericDraftKey, value: string) => {
    setDraft((current) => current ? { ...current, [key]: value } : current);
  };

  const applyServerView = (next: SettingsView) => {
    setConfig(next);
    setDraft(draftFromSettings(next));
  };

  const save = async () => {
    setBusy(true);
    setMessage("Validating and saving the next-run Discovery parameters…");
    try {
      const riskyStartBalance = asFinite(draft.riskyStartBalance, "Risky start balance");
      const riskyTargetBalance = asFinite(draft.riskyTargetBalance, "Risky target balance");
      const riskyHorizonDays = asNonNegativeInteger(draft.riskyHorizonDays, "Risky horizon");
      if (riskyStartBalance <= 0 || riskyTargetBalance <= riskyStartBalance || riskyHorizonDays === 0) {
        throw new Error("Risky objective requires a positive start, a larger target, and at least one day.");
      }

      const sharedMin = asPercentFraction(draft.sharedSearchRiskMinPct, "Shared risk minimum");
      const sharedMax = asPercentFraction(draft.sharedSearchRiskMaxPct, "Shared risk maximum");
      const riskyMin = asPercentFraction(draft.riskySearchRiskMinPct, "Risky risk minimum");
      const riskyMax = asPercentFraction(draft.riskySearchRiskMaxPct, "Risky risk maximum");
      const propMin = asPercentFraction(draft.propFirmSearchRiskMinPct, "Prop-firm risk minimum");
      const propMax = asPercentFraction(draft.propFirmSearchRiskMaxPct, "Prop-firm risk maximum");
      if (sharedMin > sharedMax || riskyMin > riskyMax || propMin > propMax) {
        throw new Error("Every search-risk minimum must be less than or equal to its maximum.");
      }
      if (!isSearchDevice(draft.searchDevice)) {
        throw new Error(`Search device '${draft.searchDevice}' is a legacy alias. Select a canonical device policy.`);
      }

      const payload: SettingsUpdate = {
        tradingMode: draft.tradingMode,
        searchDevice: draft.searchDevice,
        riskyStartBalance,
        riskyTargetBalance,
        riskyHorizonDays,
        sharedSearchRiskMin: sharedMin,
        sharedSearchRiskMax: sharedMax,
        riskySearchRiskMin: riskyMin,
        riskySearchRiskMax: riskyMax,
        propFirmSearchRiskMin: propMin,
        propFirmSearchRiskMax: propMax,
        searchHighQualityConfidence: asPercentFraction(
          draft.searchHighQualityConfidencePct,
          "Maximum-risk confidence",
        ),
        propFirmSearchProfitTargetPct: asPercentFraction(
          draft.propFirmSearchProfitTargetPct,
          "Prop-firm window target",
        ),
        propFirmSearchMaxDailyLossPct: asPercentFraction(
          draft.propFirmSearchMaxDailyLossPct,
          "Prop-firm daily-loss limit",
        ),
        propFirmSearchMaxDrawdownPct: asPercentFraction(
          draft.propFirmSearchMaxDrawdownPct,
          "Prop-firm drawdown limit",
        ),
        propFirmSearchMinTradingDays: asNonNegativeInteger(
          draft.propFirmSearchMinTradingDays,
          "Prop-firm minimum trading days",
        ),
        propFirmSearchWindowDays: asNonNegativeInteger(
          draft.propFirmSearchWindowDays,
          "Prop-firm window length",
        ),
        propFirmSearchWindowCount: asNonNegativeInteger(
          draft.propFirmSearchWindowCount,
          "Prop-firm window count",
        ),
        propFirmSearchPassRate: asPercentFraction(
          draft.propFirmSearchPassRatePct,
          "Prop-firm pass-rate floor",
        ),
        searchPopulation: asNonNegativeInteger(draft.searchPopulation, "Population"),
        searchPopulationAuto: draft.searchPopulationAuto,
        searchGenerations: asNonNegativeInteger(draft.searchGenerations, "Generations"),
        searchMaxHours: asFinite(draft.searchMaxHours, "Time cap"),
        searchMaxIndicators: asNonNegativeInteger(draft.searchMaxIndicators, "Indicators per gene"),
        searchPortfolioSize: asNonNegativeInteger(draft.searchPortfolioSize, "Portfolio size"),
        searchCorrThreshold: asPercentFraction(draft.searchCorrThresholdPct, "Correlation ceiling"),
        searchMaxRows: asNonNegativeInteger(draft.searchMaxRows, "Maximum rows"),
        prefilterTopK: asNonNegativeInteger(draft.prefilterTopK, "Feature-pool floor"),
        convergencePatience: asNonNegativeInteger(draft.convergencePatience, "Convergence patience"),
        stagnationPatience: asNonNegativeInteger(draft.stagnationPatience, "Stagnation patience"),
        noveltyWeight: asPercentFraction(draft.noveltyWeightPct, "Novelty weight"),
        disableSmcGate: draft.disableSmcGate,
      };

      if (payload.searchMaxHours !== undefined && payload.searchMaxHours < 0) {
        throw new Error("Time cap cannot be negative; use 0 for no time cap.");
      }
      if (payload.searchPopulation !== undefined && payload.searchPopulation < 10) {
        throw new Error("Population must be at least 10.");
      }
      if (payload.searchGenerations === 0 || payload.searchPortfolioSize === 0) {
        throw new Error("Generations and portfolio size must be at least 1.");
      }
      if (payload.prefilterTopK !== undefined && payload.prefilterTopK < 10) {
        throw new Error("Feature-pool floor must be at least 10.");
      }

      applyServerView(await updateSettings(payload));
      setMessage(
        "Saved to config.yaml for the next admitted run. No research contract was sealed and no run was started.",
      );
    } catch (error) {
      setMessage(`Discovery parameters were not saved: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setBusy(false);
    }
  };

  const effectiveMode = config.tradingModeDivergent
    ? config.effectiveDiscoveryMode
    : draft.tradingMode;
  const riskMinKey: NumericDraftKey = effectiveMode === "risky"
    ? "riskySearchRiskMinPct"
    : effectiveMode === "prop_firm"
      ? "propFirmSearchRiskMinPct"
      : "sharedSearchRiskMinPct";
  const riskMaxKey: NumericDraftKey = effectiveMode === "risky"
    ? "riskySearchRiskMaxPct"
    : effectiveMode === "prop_firm"
      ? "propFirmSearchRiskMaxPct"
      : "sharedSearchRiskMaxPct";

  return (
    <section aria-labelledby="discovery-parameters-heading">
      <div className="section-heading-row">
        <div>
          <h2 id="discovery-parameters-heading">Objective &amp; search constraints</h2>
          <p className="muted small">One persisted pre-flight for what the search is trying to discover and how candidates are sized.</p>
        </div>
        <span className={`badge ${draftDirty ? "live" : "demo"}`}>
          {draftDirty ? "UNSAVED" : "SAVED · NEXT RUN"}
        </span>
      </div>

      <fieldset className="ticket discovery-parameters" disabled={busy} aria-label="Next-run Discovery parameters">
        <div className="engine-lane-head">
          <div>
            <h3>Research objective</h3>
            <p className="muted small">This changes fitness and validation criteria; it is not a display preference.</p>
          </div>
          <div className="seg" aria-label="Discovery objective">
            <button
              type="button"
              className={draft.tradingMode === "prop_firm" ? "on" : ""}
              disabled={busy}
              onClick={() => setDraft({ ...draft, tradingMode: "prop_firm" })}
            >Prop-firm</button>
            <button
              type="button"
              className={draft.tradingMode === "risky" ? "on buy" : ""}
              disabled={busy}
              onClick={() => setDraft({ ...draft, tradingMode: "risky" })}
            >Growth target</button>
          </div>
        </div>

        <p className="banner warn">The objective selector currently changes the shared application trading mode, not just this screen. Saving a different mode can also change the mode used by live risk guards.</p>

        <p>
          Effective search mode: <b>{effectiveMode}</b>. In the full financial-evaluation lane the
          objective shapes candidate ranking and the same run's backtest / walk-forward gates. The
          canonical native lane currently stops at Generation 0.
        </p>
        {config.tradingModeDivergent && (
          <div className="banner warn" role="alert">
            <code>models.discovery_mode={config.discoveryMode}</code> overrides the selected mode,
            so the engine will run <b>{effectiveMode}</b>. Clear the strict/legacy escape hatch in
            Settings → Advanced before changing the objective here.
          </div>
        )}

        {effectiveMode === "risky" && (
          <fieldset className="parameter-group">
            <legend>Growth objective</legend>
            <div className="ticket-row parameter-grid">
              <label>Start balance
                <input type="number" min="1" step="50" value={draft.riskyStartBalance} onChange={(event) => setNumber("riskyStartBalance", event.target.value)} />
              </label>
              <label>Target balance
                <input type="number" min="1" step="1000" value={draft.riskyTargetBalance} onChange={(event) => setNumber("riskyTargetBalance", event.target.value)} />
              </label>
              <label>Horizon (days)
                <input type="number" min="1" step="1" value={draft.riskyHorizonDays} onChange={(event) => setNumber("riskyHorizonDays", event.target.value)} />
              </label>
            </div>
            <p className="muted small">The growth objective ranks whether the measured edge can compound from start to target within this horizon.</p>
          </fieldset>
        )}

        {effectiveMode === "prop_firm" && (
          <fieldset className="parameter-group">
            <legend>Prop-firm search window</legend>
            <div className="ticket-row parameter-grid">
              <label>Net target / window (%)
                <input type="number" min="0" max="100" step="0.1" value={draft.propFirmSearchProfitTargetPct} onChange={(event) => setNumber("propFirmSearchProfitTargetPct", event.target.value)} />
              </label>
              <label>Max daily loss (%)
                <input type="number" min="0" max="100" step="0.1" value={draft.propFirmSearchMaxDailyLossPct} onChange={(event) => setNumber("propFirmSearchMaxDailyLossPct", event.target.value)} />
              </label>
              <label>Max drawdown (%)
                <input type="number" min="0" max="100" step="0.1" value={draft.propFirmSearchMaxDrawdownPct} onChange={(event) => setNumber("propFirmSearchMaxDrawdownPct", event.target.value)} />
              </label>
              <label>Window length (days)
                <input type="number" min="1" step="1" value={draft.propFirmSearchWindowDays} onChange={(event) => setNumber("propFirmSearchWindowDays", event.target.value)} />
              </label>
              <label>Windows (0 = adaptive)
                <input type="number" min="0" step="1" value={draft.propFirmSearchWindowCount} onChange={(event) => setNumber("propFirmSearchWindowCount", event.target.value)} />
              </label>
              <label>Required pass rate (%)
                <input type="number" min="0" max="100" step="1" value={draft.propFirmSearchPassRatePct} onChange={(event) => setNumber("propFirmSearchPassRatePct", event.target.value)} />
              </label>
              <label>Minimum trading days
                <input type="number" min="0" step="1" value={draft.propFirmSearchMinTradingDays} onChange={(event) => setNumber("propFirmSearchMinTradingDays", event.target.value)} />
              </label>
            </div>
            <p className="muted small">These are Discovery screening rules. Promotion and live account guards remain later, independent authorities.</p>
          </fieldset>
        )}

        <fieldset className="parameter-group">
          <legend>Candidate sizing during search and validation</legend>
          <div className="ticket-row parameter-grid">
            <label>Risk floor / trade (%)
              <input type="number" min="0" max="100" step="0.1" value={draft[riskMinKey] as string} onChange={(event) => setNumber(riskMinKey, event.target.value)} />
            </label>
            <label>Risk ceiling / trade (%)
              <input type="number" min="0" max="100" step="0.1" value={draft[riskMaxKey] as string} onChange={(event) => setNumber(riskMaxKey, event.target.value)} />
            </label>
            <label>Confidence for ceiling (%)
              <input type="number" min="0.01" max="100" step="1" value={draft.searchHighQualityConfidencePct} onChange={(event) => setNumber("searchHighQualityConfidencePct", event.target.value)} />
            </label>
          </div>
          <p className="muted small">
            Today this band is a run-wide sizing scenario, not a gene. Each signal is sized between
            the floor and ceiling from its confidence. The full financial-evaluation lane uses the
            same resolved band in search and back/forward validation; the canonical native lane uses
            it only for Generation 0 until its validation consumer is connected. Fixed live risk is separate.
          </p>
        </fieldset>

        <fieldset className="parameter-group">
          <legend>Search breadth &amp; stopping</legend>
          <div className="ticket-row parameter-grid">
            <label>Search device
              <select
                value={draft.searchDevice}
                onChange={(event) => setDraft({ ...draft, searchDevice: event.target.value })}
              >
                {!isSearchDevice(draft.searchDevice) && (
                  <option value={draft.searchDevice}>Legacy value: {draft.searchDevice}</option>
                )}
                <option value="">Inherit training preference</option>
                <option value="auto">Auto (run preflight decides)</option>
                <option value="cpu">CPU canonical</option>
                <option value="cuda_required">CUDA required (fail closed)</option>
              </select>
            </label>
            <label>Population floor
              <input type="number" min="10" step="10" value={draft.searchPopulation} onChange={(event) => setNumber("searchPopulation", event.target.value)} />
            </label>
            <label>Generations
              <input type="number" min="1" step="10" value={draft.searchGenerations} onChange={(event) => setNumber("searchGenerations", event.target.value)} />
            </label>
            <label>Time cap (hours; 0 = none)
              <input type="number" min="0" step="0.25" value={draft.searchMaxHours} onChange={(event) => setNumber("searchMaxHours", event.target.value)} />
            </label>
            <label>Indicators per gene (0 = all)
              <input type="number" min="0" step="1" value={draft.searchMaxIndicators} onChange={(event) => setNumber("searchMaxIndicators", event.target.value)} />
            </label>
            <label>Feature-pool floor
              <input type="number" min="10" step="10" value={draft.prefilterTopK} onChange={(event) => setNumber("prefilterTopK", event.target.value)} />
            </label>
            <label>Rows (0 = full dataset)
              <input type="number" min="0" step="1000" value={draft.searchMaxRows} onChange={(event) => setNumber("searchMaxRows", event.target.value)} />
            </label>
            <label>Final portfolio capacity
              <input type="number" min="1" step="10" value={draft.searchPortfolioSize} onChange={(event) => setNumber("searchPortfolioSize", event.target.value)} />
            </label>
            <label>Correlation ceiling (%)
              <input type="number" min="0" max="100" step="1" value={draft.searchCorrThresholdPct} onChange={(event) => setNumber("searchCorrThresholdPct", event.target.value)} />
            </label>
          </div>
          <p className="muted small">
            This is the Discovery device policy, independent of model training. <b>CUDA required</b>
            refuses the run if a real CUDA device cannot be sealed; it never substitutes CPU. ROCm/HIP
            is deliberately absent until a supported native route exists.
          </p>
          <label className="inline-check">
            <input
              type="checkbox"
              checked={draft.searchPopulationAuto}
              onChange={(event) => setDraft({ ...draft, searchPopulationAuto: event.target.checked })}
            />
            <span><b>Adaptive GPU population</b> — use the configured population as a floor, then raise it to the admitted card/dataset capacity. The resolved value must appear in the run receipt.</span>
          </label>
          <div className="banner info">
            Configured population × generations: <b>{plannedCandidateBudget.toLocaleString()}</b> planned candidate evaluations
            ({Number(draft.searchPopulation || 0).toLocaleString()} × {Number(draft.searchGenerations || 0).toLocaleString()});
            adaptive population can increase that; convergence or time limits can stop the run earlier. Feature-pool floor and indicators-per-gene are different limits.
          </div>
        </fieldset>

        <details className="parameter-group">
          <summary>Advanced diversity and SMC controls</summary>
          <div className="ticket-row parameter-grid details-body">
            <label>Convergence patience
              <input type="number" min="10" step="10" value={draft.convergencePatience} onChange={(event) => setNumber("convergencePatience", event.target.value)} />
            </label>
            <label>Stagnation kick
              <input type="number" min="1" step="1" value={draft.stagnationPatience} onChange={(event) => setNumber("stagnationPatience", event.target.value)} />
            </label>
            <label>Novelty weight (%)
              <input type="number" min="0" max="100" step="1" value={draft.noveltyWeightPct} onChange={(event) => setNumber("noveltyWeightPct", event.target.value)} />
            </label>
          </div>
          <label className="inline-check">
            <input type="checkbox" checked={draft.disableSmcGate} onChange={(event) => setDraft({ ...draft, disableSmcGate: event.target.checked })} />
            <span><b>Disable SMC gate</b> — SMC features remain available, but SMC confirmation stops rejecting candidates.</span>
          </label>
        </details>

        <div className="btn-row">
          <button type="button" className="primary" disabled={busy} onClick={() => void save()}>
            {busy ? "Saving…" : "Save next-run parameters"}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              setDraft(draftFromSettings(config));
              setMessage("Unsaved edits discarded.");
            }}
          >Reset unsaved edits</button>
        </div>
        {message && <div className="banner info" role="status">{message}</div>}
      </fieldset>
    </section>
  );
}
