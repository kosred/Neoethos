//! The goal constants — **read-only**, named by field path.
//!
//! `docs/autoresearch-loop.md` §1.
//!
//! The goals are already in the operator's config. This module **loads** them
//! and **hashes** them. It has no code path that writes them, and that is
//! structural rather than promised: [`GoalSet`]'s fields are private, it
//! exposes getters only, and no `SearchConfigDelta` variant names any of them
//! ([`crate::space::FROZEN_FIELDS`], enforced by a test in that module).
//!
//! *A loop that can edit its own objective will reach any goal you like and
//! mean nothing.*
//!
//! ## The refusals — a zero goal is not a goal
//!
//! The operator's installed `config.yaml` carries
//! `risk.monthly_profit_target_pct: 0.0` and `models.prop_firm_min_pass_rate:
//! 0.0` against shipped defaults of `0.04` and `0.40`. A loop that judged a
//! prop-firm sweep against a 0% monthly target and a 0% pass rate would promote
//! anything at all. So G3 and G4 below **abort the session**, naming the field
//! path and the resolved value.
//!
//! **The loop does not fix them.** Fixing them is writing the goals. The edit
//! is written out in `docs/autoresearch-loop.md` §14.5 for whoever owns that
//! file.

use neoethos_core::config::Settings;
use neoethos_core::domain::prop_firm::{PropFirmConstraints, PropFirmPreset};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Which family of goal a scenario belongs to. Mirrors `system.trading_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioKind {
    /// Capital multiplication: start → target within a horizon.
    Risky,
    /// Up to 4% each month, under daily-loss and overall-drawdown path
    /// constraints.
    PropFirm,
}

impl ScenarioKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Risky => "risky",
            Self::PropFirm => "prop_firm",
        }
    }
}

/// One risky scenario: *start → target, by when*.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RiskyScenario {
    /// Human label, e.g. `"100 -> 50k (x500)"`. Rendered in the verdict.
    pub label: String,
    pub start_balance_usd: f64,
    pub target_balance_usd: f64,
    pub horizon_days: u32,
    /// The config field path each number came from, so a reader can check it
    /// against the file rather than trust this struct.
    ///
    /// `Cow<'static, str>` and not `&'static str`: `Scenario` is an internally
    /// tagged enum, so serde buffers the variant content and requires the
    /// variant type to be `DeserializeOwned`. A borrowed `&'static str` field
    /// forces `'de: 'static` and the derive fails. `Cow` deserializes as owned
    /// with no lifetime bound, and every construction site here is still a
    /// zero-cost `Cow::Borrowed` over a literal.
    pub source: std::borrow::Cow<'static, str>,
}

impl RiskyScenario {
    /// The multiple the operator is asking for. Rendered because "x500" is the
    /// number that makes the ask legible, and it is not in the config.
    pub fn multiple(&self) -> f64 {
        if self.start_balance_usd > 0.0 {
            self.target_balance_usd / self.start_balance_usd
        } else {
            f64::NAN
        }
    }
}

/// The prop-firm goal: a monthly bar plus the path constraints that make it
/// meaningful. Every field is a config value, never a literal chosen here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropFirmScenario {
    pub label: String,
    /// `risk.monthly_profit_target_pct`.
    pub monthly_profit_target_pct: f64,
    /// `models.prop_firm_min_pass_rate` — the fraction of months that must
    /// clear the bar.
    pub min_pass_rate: f64,
    /// `models.discovery_runtime.prop_firm_gate.max_daily_loss_pct`, resolved
    /// (`None` → the active preset's baseline).
    pub max_daily_loss_pct: f64,
    /// `…prop_firm_gate.max_overall_drawdown_pct`, resolved.
    pub max_overall_drawdown_pct: f64,
    /// `…prop_firm_gate.profit_target_pct`, resolved.
    pub profit_target_pct: f64,
    /// `…prop_firm_gate.min_trading_days`, resolved.
    pub min_trading_days: usize,
    /// `…prop_firm_gate.window_days`.
    pub window_days: usize,
    /// `risk.preset` — which firm's baseline filled the `None`s above.
    pub preset: String,
}

/// A scenario the loop may optimise toward. Never constructed outside
/// [`GoalSet::load`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Scenario {
    Risky(RiskyScenario),
    PropFirm(PropFirmScenario),
}

impl Scenario {
    pub fn kind(&self) -> ScenarioKind {
        match self {
            Self::Risky(_) => ScenarioKind::Risky,
            Self::PropFirm(_) => ScenarioKind::PropFirm,
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Self::Risky(s) => &s.label,
            Self::PropFirm(s) => &s.label,
        }
    }
}

/// Why the session refused to start. `docs/autoresearch-loop.md` §1.3.
///
/// **None of these is a warning.** Each names the field path and the resolved
/// value, because the operator has to be able to go and look at the number that
/// stopped the run.
#[derive(Debug, Clone, PartialEq)]
pub enum GoalRefusal {
    /// G1 — risky mode without a well-ordered start/target pair.
    G1RiskyTargetNotAboveStart { start: f64, target: f64 },
    /// G2 — risky mode with a zero horizon.
    G2RiskyHorizonZero,
    /// G3 — prop-firm mode with a non-positive monthly target.
    G3MonthlyTargetNotPositive { value: f64 },
    /// G4 — prop-firm mode with a non-positive pass rate.
    G4PassRateNotPositive { value: f64 },
    /// G5 — `system.trading_mode` is neither `risky` nor `prop_firm`.
    G5UnknownTradingMode { value: String },
    /// G6 — a goal field is NaN or infinite.
    G6NonFiniteGoalField { field: &'static str, value: f64 },
}

impl fmt::Display for GoalRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::G1RiskyTargetNotAboveStart { start, target } => write!(
                f,
                "G1 REFUSED: system.trading_mode = \"risky\" but the goal is not well-ordered — \
                 system.risky_start_balance_usd = {start}, system.risky_target_balance_usd = \
                 {target}. The loop requires target > start > 0. It will NOT choose these for \
                 you: setting a goal is the operator's act, not the optimiser's \
                 (docs/autoresearch-loop.md §1.3)."
            ),
            Self::G2RiskyHorizonZero => write!(
                f,
                "G2 REFUSED: system.trading_mode = \"risky\" but system.risky_horizon_days = 0. \
                 \"Reach the target eventually\" is not a goal — every positive-drift path \
                 reaches every target given unbounded time, so a zero horizon makes the goal \
                 metric vacuous (docs/autoresearch-loop.md §1.3)."
            ),
            Self::G3MonthlyTargetNotPositive { value } => write!(
                f,
                "G3 REFUSED: system.trading_mode = \"prop_firm\" but \
                 risk.monthly_profit_target_pct = {value}. A loop judging against a 0% monthly \
                 target promotes ANYTHING. The FTMO preset's own bar is \
                 min_monthly_net_profit_pct = 0.04 (neoethos-core/src/domain/prop_firm.rs). \
                 Restore it in config.yaml, or state on the record that prop-firm mode is out of \
                 scope for autoresearch — the loop will not choose for you, because choosing \
                 would be writing the goal (docs/autoresearch-loop.md §14.5)."
            ),
            Self::G4PassRateNotPositive { value } => write!(
                f,
                "G4 REFUSED: system.trading_mode = \"prop_firm\" but \
                 models.prop_firm_min_pass_rate = {value}, against a shipped default of 0.40. A \
                 0% pass rate means \"no month has to clear the bar\", which is the same defect \
                 as G3 wearing a different hat (docs/autoresearch-loop.md §14.5)."
            ),
            Self::G5UnknownTradingMode { value } => write!(
                f,
                "G5 REFUSED: system.trading_mode = \"{value}\" is neither \"risky\" nor \
                 \"prop_firm\". The loop has no goal to optimise toward and will not invent one \
                 (docs/autoresearch-loop.md §1.3)."
            ),
            Self::G6NonFiniteGoalField { field, value } => write!(
                f,
                "G6 REFUSED: {field} = {value} is not finite. Every downstream comparison \
                 against a NaN goal silently returns false, so the loop would run for days and \
                 promote nothing without ever saying why (docs/autoresearch-loop.md §1.3)."
            ),
        }
    }
}

impl std::error::Error for GoalRefusal {}

/// The search-side mirror disagreed with the `system.*` authority.
///
/// `DiscoveryConfig::risky_start_balance` / `risky_target_balance` /
/// `risky_horizon_days` mirror the `system.*` fields
/// (`neoethos-search/src/discovery.rs:667-669`). The loop reads `system.*` as
/// the authority and asserts the mirror agrees; a disagreement means two parts
/// of the process are optimising toward two different goals, which is worse
/// than either being wrong.
#[derive(Debug, Clone, PartialEq)]
pub struct MirrorDisagreement {
    pub field: &'static str,
    pub authority_path: &'static str,
    pub authority_value: f64,
    pub mirror_path: &'static str,
    pub mirror_value: f64,
}

impl fmt::Display for MirrorDisagreement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GOAL MIRROR DISAGREEMENT on {}: the authority {} = {} but the search mirror {} = {}. \
             Two parts of this process would optimise toward two different goals. Refusing to \
             start (docs/autoresearch-loop.md §1.1).",
            self.field,
            self.authority_path,
            self.authority_value,
            self.mirror_path,
            self.mirror_value
        )
    }
}

impl std::error::Error for MirrorDisagreement {}

/// The operator's goals, loaded once and frozen for the life of the session.
///
/// **Every field is private.** There are getters and there is no setter, no
/// `&mut` accessor and no public constructor other than [`GoalSet::load`],
/// which reads a [`Settings`] and never writes one. That is non-negotiable 1
/// made structural.
#[derive(Debug, Clone, PartialEq)]
pub struct GoalSet {
    trading_mode: String,
    scenarios: Vec<Scenario>,
    scenarios_source: &'static str,
    field_values: Vec<(&'static str, String)>,
    goal_hash: String,
}

impl GoalSet {
    /// Load and validate. Returns the FIRST refusal, so the message names one
    /// field rather than a list the operator has to triage.
    pub fn load(settings: &Settings) -> Result<Self, GoalRefusal> {
        let mode = settings.system.trading_mode.trim().to_ascii_lowercase();

        // G6 first, on every goal field of both modes: a non-finite value makes
        // every later comparison quietly false, so it must be caught before the
        // mode-specific checks read any of them.
        for (path, value) in [
            (
                "system.risky_start_balance_usd",
                settings.system.risky_start_balance_usd,
            ),
            (
                "system.risky_target_balance_usd",
                settings.system.risky_target_balance_usd,
            ),
            (
                "risk.monthly_profit_target_pct",
                settings.risk.monthly_profit_target_pct,
            ),
            (
                "models.prop_firm_min_pass_rate",
                settings.models.prop_firm_min_pass_rate,
            ),
        ] {
            if !value.is_finite() {
                return Err(GoalRefusal::G6NonFiniteGoalField { field: path, value });
            }
        }

        let (scenarios, scenarios_source, field_values) = match mode.as_str() {
            "risky" => {
                let start = settings.system.risky_start_balance_usd;
                let target = settings.system.risky_target_balance_usd;
                let horizon = settings.system.risky_horizon_days;

                if !(target > start && start > 0.0) {
                    return Err(GoalRefusal::G1RiskyTargetNotAboveStart { start, target });
                }
                if horizon == 0 {
                    return Err(GoalRefusal::G2RiskyHorizonZero);
                }

                let primary = RiskyScenario {
                    label: format!(
                        "{start:.0} -> {target:.0} (x{:.0})",
                        if start > 0.0 {
                            target / start
                        } else {
                            f64::NAN
                        }
                    ),
                    start_balance_usd: start,
                    target_balance_usd: target,
                    horizon_days: horizon,
                    source: std::borrow::Cow::Borrowed(
                        "system.risky_start_balance_usd / system.risky_target_balance_usd / \
                         system.risky_horizon_days",
                    ),
                };

                // ── THE SECOND SCENARIO IS NOT IN THE CONFIG ────────────────
                //
                // The operator stated two: 100 -> 50,000 AND 1,000 -> 200,000,
                // both in six months. Only the first is a config citizen; the
                // second exists solely as a demo constant at
                // `neoethos-search/examples/goal_frontier_demo.rs:36`.
                //
                // The loop MAY NOT INVENT IT and may not hardcode it — a goal
                // the loop wrote is a goal the loop can reach. Until
                // `system.risky_scenarios` lands (docs/autoresearch-loop.md
                // §14.1) this returns exactly ONE scenario and the session
                // header says so, in the header, where a reader of the verdict
                // will see it.
                //
                // When §14.1 lands, extend here by MAPPING
                // `settings.system.risky_scenarios` — never by adding a
                // literal.
                let source = "system.risky_* (single) — system.risky_scenarios not present; \
                              the operator's second scenario (1,000 -> 200,000) is NOT \
                              enumerated, see docs/autoresearch-loop.md §14.1";

                let fields = vec![
                    ("system.trading_mode", mode.clone()),
                    ("system.risky_start_balance_usd", format!("{start}")),
                    ("system.risky_target_balance_usd", format!("{target}")),
                    ("system.risky_horizon_days", format!("{horizon}")),
                ];
                (vec![Scenario::Risky(primary)], source, fields)
            }
            "prop_firm" => {
                let monthly = settings.risk.monthly_profit_target_pct;
                if monthly <= 0.0 {
                    return Err(GoalRefusal::G3MonthlyTargetNotPositive { value: monthly });
                }
                let pass_rate = settings.models.prop_firm_min_pass_rate;
                if pass_rate <= 0.0 {
                    return Err(GoalRefusal::G4PassRateNotPositive { value: pass_rate });
                }

                let preset: PropFirmPreset = settings.risk.preset;
                let baseline = PropFirmConstraints::for_preset(preset);
                let gate = &settings.models.discovery_runtime.prop_firm_gate;

                // `None` resolves to the ACTIVE preset's baseline, not to a
                // literal FTMO number: the operator can be on The5ers, and a
                // hardcoded 0.05 would judge them against a firm they do not
                // trade with.
                let scenario = PropFirmScenario {
                    label: format!("{:.1}%/month under {}", monthly * 100.0, preset.as_str()),
                    monthly_profit_target_pct: monthly,
                    min_pass_rate: pass_rate,
                    max_daily_loss_pct: gate
                        .max_daily_loss_pct
                        .unwrap_or(f64::from(baseline.max_daily_loss_pct)),
                    max_overall_drawdown_pct: gate
                        .max_overall_drawdown_pct
                        .unwrap_or(f64::from(baseline.max_overall_drawdown_pct)),
                    profit_target_pct: gate
                        .profit_target_pct
                        .unwrap_or(f64::from(baseline.challenge_profit_target_pct)),
                    min_trading_days: gate
                        .min_trading_days
                        .unwrap_or(baseline.min_trading_days as usize),
                    window_days: gate.window_days,
                    preset: preset.as_str().to_string(),
                };

                for (path, value) in [
                    (
                        "models.discovery_runtime.prop_firm_gate.max_daily_loss_pct",
                        scenario.max_daily_loss_pct,
                    ),
                    (
                        "models.discovery_runtime.prop_firm_gate.max_overall_drawdown_pct",
                        scenario.max_overall_drawdown_pct,
                    ),
                    (
                        "models.discovery_runtime.prop_firm_gate.profit_target_pct",
                        scenario.profit_target_pct,
                    ),
                ] {
                    if !value.is_finite() {
                        return Err(GoalRefusal::G6NonFiniteGoalField { field: path, value });
                    }
                }

                let fields = vec![
                    ("system.trading_mode", mode.clone()),
                    ("risk.preset", scenario.preset.clone()),
                    ("risk.monthly_profit_target_pct", format!("{monthly}")),
                    ("models.prop_firm_min_pass_rate", format!("{pass_rate}")),
                    (
                        "models.discovery_runtime.prop_firm_gate.max_daily_loss_pct",
                        format!("{}", scenario.max_daily_loss_pct),
                    ),
                    (
                        "models.discovery_runtime.prop_firm_gate.max_overall_drawdown_pct",
                        format!("{}", scenario.max_overall_drawdown_pct),
                    ),
                    (
                        "models.discovery_runtime.prop_firm_gate.profit_target_pct",
                        format!("{}", scenario.profit_target_pct),
                    ),
                    (
                        "models.discovery_runtime.prop_firm_gate.min_trading_days",
                        format!("{}", scenario.min_trading_days),
                    ),
                    (
                        "models.discovery_runtime.prop_firm_gate.window_days",
                        format!("{}", scenario.window_days),
                    ),
                ];
                (
                    vec![Scenario::PropFirm(scenario)],
                    "risk.* + models.prop_firm_min_pass_rate + \
                     models.discovery_runtime.prop_firm_gate.*",
                    fields,
                )
            }
            other => {
                return Err(GoalRefusal::G5UnknownTradingMode {
                    value: other.to_string(),
                });
            }
        };

        // The hash covers the field PATHS as well as the values, so renaming a
        // field is a new experiment rather than a silent continuation of the
        // old one under a name that no longer means the same thing.
        let canonical: String = field_values
            .iter()
            .map(|(path, value)| format!("{path}={value}"))
            .collect::<Vec<_>>()
            .join(";");
        let goal_hash = crate::fnv64(&canonical);

        Ok(Self {
            trading_mode: mode,
            scenarios,
            scenarios_source,
            field_values,
            goal_hash,
        })
    }

    pub fn trading_mode(&self) -> &str {
        &self.trading_mode
    }

    pub fn scenarios(&self) -> &[Scenario] {
        &self.scenarios
    }

    /// The scenario the session optimises toward when the operator named none.
    /// There is always at least one: [`GoalSet::load`] refuses rather than
    /// returning an empty set.
    pub fn primary(&self) -> &Scenario {
        &self.scenarios[0]
    }

    /// Select by label, for `--scenario`. Returns `None` rather than falling
    /// back to the primary: silently optimising toward a different goal than
    /// the one the operator typed is the whole failure mode this crate exists
    /// to avoid.
    pub fn scenario_by_label(&self, label: &str) -> Option<&Scenario> {
        self.scenarios.iter().find(|s| s.label() == label)
    }

    /// What the scenario list was built from — rendered into the session header
    /// and the verdict, because *"one scenario"* and *"the operator's two
    /// scenarios"* are different claims.
    pub fn scenarios_source(&self) -> &'static str {
        self.scenarios_source
    }

    /// `(field path, resolved value)` for every goal constant, in load order.
    /// Written into `SessionOpened` so the verdict can be checked against the
    /// config file by hand.
    pub fn field_values(&self) -> &[(&'static str, String)] {
        &self.field_values
    }

    pub fn goal_hash(&self) -> &str {
        &self.goal_hash
    }

    /// Assert the search-side mirror agrees with the `system.*` authority
    /// (§1.1). Called at startup, before any sweep.
    ///
    /// Takes the three mirrored numbers rather than a `&DiscoveryConfig` so
    /// this stays pure and testable without building a config, and so that a
    /// future rename on the search side surfaces at the ONE call site in
    /// [`crate::contracts`] rather than here.
    pub fn assert_mirror_agrees(
        &self,
        mirror_start: f64,
        mirror_target: f64,
        mirror_horizon_days: f64,
    ) -> Result<(), MirrorDisagreement> {
        let Scenario::Risky(primary) = self.primary() else {
            // Prop-firm mode has no risky mirror to disagree with.
            return Ok(());
        };
        for (field, authority_path, authority, mirror_path, mirror) in [
            (
                "risky start balance",
                "system.risky_start_balance_usd",
                primary.start_balance_usd,
                "DiscoveryConfig::risky_start_balance",
                mirror_start,
            ),
            (
                "risky target balance",
                "system.risky_target_balance_usd",
                primary.target_balance_usd,
                "DiscoveryConfig::risky_target_balance",
                mirror_target,
            ),
            (
                "risky horizon",
                "system.risky_horizon_days",
                f64::from(primary.horizon_days),
                "DiscoveryConfig::risky_horizon_days",
                mirror_horizon_days,
            ),
        ] {
            if (authority - mirror).abs() > 1e-9 {
                return Err(MirrorDisagreement {
                    field,
                    authority_path,
                    authority_value: authority,
                    mirror_path,
                    mirror_value: mirror,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn risky_settings() -> Settings {
        let mut s = Settings::default();
        s.system.trading_mode = "risky".to_string();
        s.system.risky_start_balance_usd = 100.0;
        s.system.risky_target_balance_usd = 50_000.0;
        s.system.risky_horizon_days = 180;
        s
    }

    fn prop_firm_settings() -> Settings {
        let mut s = Settings::default();
        s.system.trading_mode = "prop_firm".to_string();
        s.risk.monthly_profit_target_pct = 0.04;
        s.models.prop_firm_min_pass_rate = 0.40;
        s
    }

    #[test]
    fn the_operators_installed_risky_goal_loads() {
        let goals = GoalSet::load(&risky_settings()).expect("risky goals load");
        assert_eq!(goals.trading_mode(), "risky");
        assert_eq!(goals.scenarios().len(), 1);
        let Scenario::Risky(s) = goals.primary() else {
            panic!("expected a risky scenario");
        };
        assert_eq!(s.start_balance_usd, 100.0);
        assert_eq!(s.target_balance_usd, 50_000.0);
        assert_eq!(s.horizon_days, 180);
        assert!((s.multiple() - 500.0).abs() < 1e-9);
    }

    #[test]
    fn the_second_scenario_is_absent_and_the_header_says_so() {
        // The operator named TWO scenarios. Only one is in the config. The loop
        // must not invent the other, and it must not stay quiet about it.
        let goals = GoalSet::load(&risky_settings()).unwrap();
        assert_eq!(goals.scenarios().len(), 1);
        assert!(goals.scenarios_source().contains("§14.1"));
        assert!(goals.scenarios_source().contains("1,000 -> 200,000"));
    }

    #[test]
    fn a_zero_monthly_target_aborts_naming_the_field() {
        // The operator's installed config.yaml really does carry 0.0 here.
        let mut s = prop_firm_settings();
        s.risk.monthly_profit_target_pct = 0.0;
        let err = GoalSet::load(&s).expect_err("a 0% monthly target is not a goal");
        assert!(matches!(
            err,
            GoalRefusal::G3MonthlyTargetNotPositive { .. }
        ));
        assert!(err.to_string().contains("risk.monthly_profit_target_pct"));
    }

    #[test]
    fn a_zero_pass_rate_aborts_naming_the_field() {
        let mut s = prop_firm_settings();
        s.models.prop_firm_min_pass_rate = 0.0;
        let err = GoalSet::load(&s).expect_err("a 0% pass rate is not a goal");
        assert!(matches!(err, GoalRefusal::G4PassRateNotPositive { .. }));
        assert!(err.to_string().contains("models.prop_firm_min_pass_rate"));
    }

    #[test]
    fn a_target_below_the_start_aborts() {
        let mut s = risky_settings();
        s.system.risky_target_balance_usd = 50.0;
        assert!(matches!(
            GoalSet::load(&s),
            Err(GoalRefusal::G1RiskyTargetNotAboveStart { .. })
        ));
    }

    #[test]
    fn a_zero_horizon_aborts() {
        let mut s = risky_settings();
        s.system.risky_horizon_days = 0;
        assert!(matches!(
            GoalSet::load(&s),
            Err(GoalRefusal::G2RiskyHorizonZero)
        ));
    }

    #[test]
    fn an_unknown_trading_mode_aborts_rather_than_defaulting() {
        let mut s = risky_settings();
        s.system.trading_mode = "aggressive".to_string();
        assert!(matches!(
            GoalSet::load(&s),
            Err(GoalRefusal::G5UnknownTradingMode { .. })
        ));
    }

    #[test]
    fn a_non_finite_goal_aborts_before_any_comparison_can_swallow_it() {
        let mut s = risky_settings();
        s.system.risky_target_balance_usd = f64::NAN;
        assert!(matches!(
            GoalSet::load(&s),
            Err(GoalRefusal::G6NonFiniteGoalField { .. })
        ));
    }

    #[test]
    fn the_goal_hash_moves_when_a_goal_moves_and_not_otherwise() {
        let a = GoalSet::load(&risky_settings()).unwrap();
        let b = GoalSet::load(&risky_settings()).unwrap();
        assert_eq!(a.goal_hash(), b.goal_hash());

        let mut moved = risky_settings();
        moved.system.risky_target_balance_usd = 200_000.0;
        let c = GoalSet::load(&moved).unwrap();
        assert_ne!(a.goal_hash(), c.goal_hash());
    }

    #[test]
    fn a_mirror_that_disagrees_with_the_authority_is_refused() {
        let goals = GoalSet::load(&risky_settings()).unwrap();
        assert!(goals.assert_mirror_agrees(100.0, 50_000.0, 180.0).is_ok());

        let err = goals
            .assert_mirror_agrees(100.0, 25_000.0, 180.0)
            .expect_err("a mirror at half the target must be refused");
        assert!(err.to_string().contains("system.risky_target_balance_usd"));
        assert!(
            err.to_string()
                .contains("DiscoveryConfig::risky_target_balance")
        );
    }
}
