//! Prop firm hard constraints + named-preset registry.
//!
//! NeoEthos is not locked to a single firm. The runtime reads its rule
//! set from a named preset (`PropFirmPreset`) — FTMO, MyForexFunds,
//! FundedNext, The5%ers, or `None` (own-money / personal account, no
//! external caps). FTMO is *one* preset, not "the" rule set. Operators
//! pick the preset that matches their account via `config.yaml`'s
//! `risk.preset` field, the `NEOETHOS_PROP_FIRM_PRESET` env var, or
//! the Flutter Risk Settings screen.
//!
//! Numbers in built-in presets are approximate as of writing
//! (2026-05). Prop firms revise their rule sheets routinely —
//! operators are responsible for verifying the active values against
//! their current contract. The runtime never enforces a number it
//! didn't read from the active preset; users can override any field
//! by editing `config.yaml`'s `risk:` block (preset values are seeds,
//! not locks).
//!
//! Source for FTMO numbers: https://ftmo.com/en/trading-objectives/.
//! Other firms cross-checked from each firm's published rule pages.
//! External prop-firm numbers belong in [`PropFirmConstraints`]. Local
//! neoethos policy defaults live beside it so search, validation, and
//! live risk code do not carry duplicate literals.

use serde::{Deserialize, Serialize};

/// Named prop-firm preset. Drives the default values in `RiskConfig`,
/// `PropFirmRules`, and the discovery-side challenge gate. The runtime
/// itself is firm-agnostic — it just reads numeric thresholds from
/// whichever preset is active.
///
/// `None` is the own-money / personal-account preset: no external
/// drawdown caps, no profit target, no minimum trading days. Use it
/// when the operator's account is theirs and not a funded challenge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PropFirmPreset {
    #[default]
    Ftmo,
    MyForexFunds,
    FundedNext,
    The5ers,
    /// Own-money / personal account. No external caps; the runtime
    /// still respects per-trade risk-management settings the operator
    /// dialled in, but the prop-firm gate is effectively bypassed.
    None,
}

impl PropFirmPreset {
    /// Human-readable preset label for UI / CLI output.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ftmo => "ftmo",
            Self::MyForexFunds => "myforexfunds",
            Self::FundedNext => "fundednext",
            Self::The5ers => "the5ers",
            Self::None => "none",
        }
    }

    /// Display name (Title Case) suitable for UI dropdowns / labels.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Ftmo => "FTMO",
            Self::MyForexFunds => "MyForexFunds",
            Self::FundedNext => "FundedNext",
            Self::The5ers => "The5%ers",
            Self::None => "None (own account)",
        }
    }

    /// Parse `--prop-firm-preset` / `NEOETHOS_PROP_FIRM_PRESET` / the
    /// `risk.preset` YAML field. Case-insensitive. Returns `None` for
    /// unknown values so callers can decide to default vs. reject.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "ftmo" => Some(Self::Ftmo),
            "myforexfunds" | "mff" => Some(Self::MyForexFunds),
            "fundednext" | "funded_next" => Some(Self::FundedNext),
            "the5ers" | "the_5ers" | "5ers" => Some(Self::The5ers),
            "none" | "personal" | "own" | "ownmoney" | "own_money" => Some(Self::None),
            _ => None,
        }
    }

    /// All known presets — used by the Flutter dropdown + CLI listing.
    pub fn all() -> &'static [Self] {
        &[
            Self::Ftmo,
            Self::MyForexFunds,
            Self::FundedNext,
            Self::The5ers,
            Self::None,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropFirmConstraints {
    /// Maximum daily loss as fraction of account equity (FTMO: 5%).
    /// A loss exceeding this in a single trading day fails the challenge.
    pub max_daily_loss_pct: f32,
    /// Maximum overall drawdown as fraction of account equity (FTMO: 10%).
    /// Trailing or static depending on firm; FTMO uses static-from-initial-balance.
    pub max_overall_drawdown_pct: f32,
    /// Profit target as fraction of account equity to clear the challenge
    /// (FTMO Phase 1: 10%, Phase 2: 5%).
    pub challenge_profit_target_pct: f32,
    /// Operator-mandated minimum monthly net profit target (4% per
    /// directive 2026-05-14). Live strategies that drop below this
    /// monthly should be flagged for review.
    pub min_monthly_net_profit_pct: f32,
    /// Minimum trading days per challenge cycle (FTMO: 4 trading days
    /// for the Aggressive variant, 10 for Standard).
    pub min_trading_days: u32,
}

impl PropFirmConstraints {
    /// Canonical FTMO Trader Challenge values plus operator's 4%
    /// monthly profit floor.
    pub const FTMO_STANDARD: Self = Self {
        max_daily_loss_pct: 0.05,
        max_overall_drawdown_pct: 0.10,
        challenge_profit_target_pct: 0.10,
        min_monthly_net_profit_pct: 0.04, // operator directive
        min_trading_days: 10,
    };

    /// MyForexFunds Rapid / Evaluation defaults (approximate).
    /// MFF historically used 5% daily DD, 12% overall, 8% Phase 1
    /// profit target, 5 minimum trading days.
    pub const MYFOREXFUNDS_STANDARD: Self = Self {
        max_daily_loss_pct: 0.05,
        max_overall_drawdown_pct: 0.12,
        challenge_profit_target_pct: 0.08,
        min_monthly_net_profit_pct: 0.04,
        min_trading_days: 5,
    };

    /// FundedNext Stellar Lite defaults (approximate).
    pub const FUNDEDNEXT_STANDARD: Self = Self {
        max_daily_loss_pct: 0.05,
        max_overall_drawdown_pct: 0.10,
        challenge_profit_target_pct: 0.08,
        min_monthly_net_profit_pct: 0.04,
        min_trading_days: 5,
    };

    /// The5%ers Bootcamp / Hyper-Growth defaults (approximate).
    /// The5%ers historically uses tighter daily DD (4%) and a 6%
    /// total drawdown ceiling.
    pub const THE5ERS_STANDARD: Self = Self {
        max_daily_loss_pct: 0.04,
        max_overall_drawdown_pct: 0.06,
        challenge_profit_target_pct: 0.06,
        min_monthly_net_profit_pct: 0.02,
        min_trading_days: 3,
    };

    /// Permissive "own-money / personal-account" preset. No external
    /// challenge target. The 10% / 20% drawdown caps are *recommended
    /// ceilings* the system surfaces in the UI as warnings — not
    /// challenge-failure conditions. Operators trading their own
    /// capital still benefit from a kill switch.
    pub const NONE_OWN_MONEY: Self = Self {
        max_daily_loss_pct: 0.10,
        max_overall_drawdown_pct: 0.20,
        challenge_profit_target_pct: 0.0,
        min_monthly_net_profit_pct: 0.0,
        min_trading_days: 0,
    };

    /// Look up the constraint set for a preset. Always returns
    /// something — `None` (personal account) is a valid preset, not
    /// an absence of rules.
    pub fn for_preset(preset: PropFirmPreset) -> Self {
        match preset {
            PropFirmPreset::Ftmo => Self::FTMO_STANDARD,
            PropFirmPreset::MyForexFunds => Self::MYFOREXFUNDS_STANDARD,
            PropFirmPreset::FundedNext => Self::FUNDEDNEXT_STANDARD,
            PropFirmPreset::The5ers => Self::THE5ERS_STANDARD,
            PropFirmPreset::None => Self::NONE_OWN_MONEY,
        }
    }
}

/// Local operating defaults for challenge-cycle planning.
///
/// These are not external prop-firm rules. They are neoethos runtime
/// defaults that need one canonical owner because the search optimizer,
/// validation artifacts, and live risk presets all reason about the same
/// challenge window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropFirmChallengeDefaults {
    /// Denominator used to turn a full challenge profit target into a daily
    /// pacing target.
    pub daily_target_trading_days: u32,
    /// Short-cycle fixture/window minimum used by local validation flows.
    pub relaxed_min_trading_days: u32,
    /// Planning horizon used by phase-specific risk presets.
    pub target_trading_days: u32,
    /// Upper bound used for challenge-cycle pacing.
    pub max_trading_days: u32,
}

impl PropFirmChallengeDefaults {
    pub const FTMO_STANDARD: Self = Self {
        daily_target_trading_days: 20,
        relaxed_min_trading_days: 5,
        target_trading_days: 22,
        max_trading_days: 60,
    };

    /// "Own money / personal" — no challenge cycle. Use the FTMO
    /// values as a sane pacing default since the search code still
    /// needs a denominator. The runtime ignores these when the
    /// preset is `None`.
    pub const NONE_OWN_MONEY: Self = Self::FTMO_STANDARD;

    /// All non-FTMO funded challenges in our preset list compress
    /// the cycle window vs. FTMO's 60-day max. 30 days is a tight
    /// default that matches FundedNext / MFF Evaluation cadence.
    pub const COMPACT_30_DAY: Self = Self {
        daily_target_trading_days: 15,
        relaxed_min_trading_days: 3,
        target_trading_days: 18,
        max_trading_days: 30,
    };

    pub fn for_preset(preset: PropFirmPreset) -> Self {
        match preset {
            PropFirmPreset::Ftmo => Self::FTMO_STANDARD,
            PropFirmPreset::MyForexFunds
            | PropFirmPreset::FundedNext
            | PropFirmPreset::The5ers => Self::COMPACT_30_DAY,
            PropFirmPreset::None => Self::NONE_OWN_MONEY,
        }
    }
}

/// Local runtime defaults layered under the hard prop-firm constraints.
///
/// These numbers are guard-rail policy, not FTMO facts. Keeping them here
/// prevents duplicated risk bands and trade caps from drifting across crates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropFirmRuntimeDefaults {
    pub max_lot_size: f64,
    pub daily_dd_warning_pct: f64,
    pub daily_dd_stop_trading_pct: f64,
    pub daily_profit_lock_pct: f64,
    pub max_trades_per_day: usize,
    pub recovery_halt_drawdown_pct: f64,
    pub recovery_top_strategy_drawdown_pct: f64,
    pub recovery_min_sharpe_drawdown_pct: f64,
    pub recovery_top_three_drawdown_pct: f64,
    pub recovery_top_strategy_rank: usize,
    pub recovery_caution_strategy_rank: usize,
    pub recovery_max_trades_per_day: usize,
    pub recovery_min_strategy_sharpe: f64,
    pub recovery_mode_risk_multiplier: f64,
    pub defensive_mode_risk_multiplier: f64,
    pub caution_mode_risk_multiplier: f64,
}

impl PropFirmRuntimeDefaults {
    pub const FTMO_STANDARD: Self = Self {
        max_lot_size: 10.0,
        daily_dd_warning_pct: 0.035,
        daily_dd_stop_trading_pct: 0.040,
        daily_profit_lock_pct: 0.03,
        max_trades_per_day: 15,
        recovery_halt_drawdown_pct: 0.05,
        recovery_top_strategy_drawdown_pct: 0.04,
        recovery_min_sharpe_drawdown_pct: 0.03,
        recovery_top_three_drawdown_pct: 0.02,
        recovery_top_strategy_rank: 1,
        recovery_caution_strategy_rank: 3,
        recovery_max_trades_per_day: 2,
        recovery_min_strategy_sharpe: 1.0,
        recovery_mode_risk_multiplier: 0.25,
        defensive_mode_risk_multiplier: 0.50,
        caution_mode_risk_multiplier: 0.75,
    };

    /// The5%ers compresses the tolerated daily DD from FTMO's 5%
    /// down to 4%, so the runtime stop-trading threshold needs to
    /// move with it (kept at 75% of the firm's ceiling).
    pub const THE5ERS_TIGHTER_DAILY_DD: Self = Self {
        daily_dd_warning_pct: 0.028,
        daily_dd_stop_trading_pct: 0.032,
        daily_profit_lock_pct: 0.02,
        ..Self::FTMO_STANDARD
    };

    /// Personal account — looser guard-rails than the funded
    /// presets, but still finite. Doubles the daily DD tolerance
    /// vs. FTMO so the operator's own bad day doesn't trip the
    /// kill switch unless something is genuinely wrong.
    pub const NONE_OWN_MONEY: Self = Self {
        daily_dd_warning_pct: 0.07,
        daily_dd_stop_trading_pct: 0.08,
        daily_profit_lock_pct: 0.05,
        max_trades_per_day: 30,
        ..Self::FTMO_STANDARD
    };

    pub fn for_preset(preset: PropFirmPreset) -> Self {
        match preset {
            PropFirmPreset::Ftmo | PropFirmPreset::MyForexFunds | PropFirmPreset::FundedNext => {
                Self::FTMO_STANDARD
            }
            PropFirmPreset::The5ers => Self::THE5ERS_TIGHTER_DAILY_DD,
            PropFirmPreset::None => Self::NONE_OWN_MONEY,
        }
    }
}

/// Phase-specific strategy defaults for FTMO-style challenge operation.
///
/// These are local strategy tunables, not published prop-firm rules. They
/// live beside the challenge/runtime defaults so the live risk preset builder
/// does not carry a separate set of phase literals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropFirmPhaseRiskDefaults {
    pub risk_per_trade: f64,
    pub max_risk_per_trade: f64,
    pub min_confidence_threshold: f64,
    pub max_trades_per_day: usize,
    pub daily_profit_lock_pct: f64,
}

impl PropFirmPhaseRiskDefaults {
    pub const FTMO_PHASE_1: Self = Self {
        risk_per_trade: 0.0030,
        max_risk_per_trade: 0.0050,
        min_confidence_threshold: 0.66,
        max_trades_per_day: 3,
        daily_profit_lock_pct: 0.015,
    };

    pub const FTMO_PHASE_2: Self = Self {
        risk_per_trade: 0.0025,
        max_risk_per_trade: 0.0040,
        min_confidence_threshold: 0.68,
        max_trades_per_day: 3,
        daily_profit_lock_pct: 0.012,
    };

    pub const FTMO_FUNDED: Self = Self {
        risk_per_trade: 0.0030,
        max_risk_per_trade: 0.0050,
        min_confidence_threshold: 0.65,
        max_trades_per_day: 4,
        daily_profit_lock_pct: 0.0,
    };

    /// Phase-1 numbers for any non-FTMO preset that doesn't have
    /// its own table. The5%ers tightens by ~25 % because the daily
    /// DD ceiling is lower; the rest stay close to FTMO.
    pub const THE5ERS_PHASE_1: Self = Self {
        risk_per_trade: 0.0020,
        max_risk_per_trade: 0.0035,
        min_confidence_threshold: 0.70,
        max_trades_per_day: 2,
        daily_profit_lock_pct: 0.010,
    };

    /// "Own money" — wider risk-per-trade band, lower confidence
    /// threshold (the operator wears the loss, not a prop firm).
    pub const NONE_OWN_MONEY: Self = Self {
        risk_per_trade: 0.0050,
        max_risk_per_trade: 0.0100,
        min_confidence_threshold: 0.55,
        max_trades_per_day: 10,
        daily_profit_lock_pct: 0.0,
    };

    /// Phase 1 / 2 / Funded selector. Most firms run a Phase 1 →
    /// Phase 2 → Funded ladder; The5%ers has a single Bootcamp
    /// phase that we slot in as Phase 1.
    pub fn for_preset(preset: PropFirmPreset, challenge_phase: &str) -> Self {
        let phase_norm = challenge_phase.trim().to_ascii_lowercase();
        match preset {
            PropFirmPreset::Ftmo | PropFirmPreset::MyForexFunds | PropFirmPreset::FundedNext => {
                match phase_norm.as_str() {
                    "phase_1" | "phase1" | "p1" => Self::FTMO_PHASE_1,
                    "phase_2" | "phase2" | "p2" => Self::FTMO_PHASE_2,
                    "funded" => Self::FTMO_FUNDED,
                    _ => Self::FTMO_PHASE_1,
                }
            }
            PropFirmPreset::The5ers => Self::THE5ERS_PHASE_1,
            PropFirmPreset::None => Self::NONE_OWN_MONEY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PropFirmChallengeDefaults, PropFirmConstraints, PropFirmPhaseRiskDefaults,
        PropFirmPreset, PropFirmRuntimeDefaults,
    };

    #[test]
    fn preset_parse_is_case_insensitive_and_aliased() {
        assert_eq!(PropFirmPreset::parse("ftmo"), Some(PropFirmPreset::Ftmo));
        assert_eq!(PropFirmPreset::parse("FTMO"), Some(PropFirmPreset::Ftmo));
        assert_eq!(
            PropFirmPreset::parse("mff"),
            Some(PropFirmPreset::MyForexFunds)
        );
        assert_eq!(
            PropFirmPreset::parse("MyForexFunds"),
            Some(PropFirmPreset::MyForexFunds)
        );
        assert_eq!(
            PropFirmPreset::parse("funded_next"),
            Some(PropFirmPreset::FundedNext)
        );
        assert_eq!(PropFirmPreset::parse("5ers"), Some(PropFirmPreset::The5ers));
        assert_eq!(PropFirmPreset::parse("none"), Some(PropFirmPreset::None));
        assert_eq!(
            PropFirmPreset::parse("personal"),
            Some(PropFirmPreset::None)
        );
        assert_eq!(PropFirmPreset::parse("nonsense"), None);
    }

    #[test]
    fn preset_default_is_ftmo_for_backwards_compatibility() {
        // The CURRENT default has been FTMO since the project began;
        // bumping this would silently change every existing operator's
        // active rule set on a routine upgrade. Anyone who wants a
        // different preset sets it explicitly in `config.yaml` or via
        // the env var.
        assert_eq!(PropFirmPreset::default(), PropFirmPreset::Ftmo);
    }

    #[test]
    fn for_preset_returns_distinct_constraints() {
        let ftmo = PropFirmConstraints::for_preset(PropFirmPreset::Ftmo);
        let the5ers = PropFirmConstraints::for_preset(PropFirmPreset::The5ers);
        let none = PropFirmConstraints::for_preset(PropFirmPreset::None);
        assert!(the5ers.max_overall_drawdown_pct < ftmo.max_overall_drawdown_pct);
        assert!(none.max_daily_loss_pct > ftmo.max_daily_loss_pct);
        assert_eq!(none.challenge_profit_target_pct, 0.0);
        assert_eq!(none.min_trading_days, 0);
    }

    #[test]
    fn the5ers_runtime_caps_stay_under_the5ers_constraints() {
        let constraints = PropFirmConstraints::for_preset(PropFirmPreset::The5ers);
        let runtime = PropFirmRuntimeDefaults::for_preset(PropFirmPreset::The5ers);
        assert!(runtime.daily_dd_stop_trading_pct <= constraints.max_daily_loss_pct as f64);
        assert!(runtime.daily_dd_warning_pct < runtime.daily_dd_stop_trading_pct);
    }

    #[test]
    fn for_preset_phase_picks_funded_when_requested() {
        let funded = PropFirmPhaseRiskDefaults::for_preset(PropFirmPreset::Ftmo, "funded");
        assert_eq!(funded.daily_profit_lock_pct, 0.0);

        let phase_2 = PropFirmPhaseRiskDefaults::for_preset(PropFirmPreset::Ftmo, "phase_2");
        let phase_1 = PropFirmPhaseRiskDefaults::for_preset(PropFirmPreset::Ftmo, "phase_1");
        assert!(phase_2.risk_per_trade <= phase_1.risk_per_trade);

        // Unknown phase string falls back to phase 1 (safer default).
        let fallback = PropFirmPhaseRiskDefaults::for_preset(PropFirmPreset::Ftmo, "garbage");
        assert_eq!(fallback.risk_per_trade, phase_1.risk_per_trade);
    }

    #[test]
    fn challenge_defaults_compact_window_is_shorter_than_ftmo() {
        let ftmo = PropFirmChallengeDefaults::for_preset(PropFirmPreset::Ftmo);
        let mff = PropFirmChallengeDefaults::for_preset(PropFirmPreset::MyForexFunds);
        assert!(mff.max_trading_days < ftmo.max_trading_days);
    }

    #[test]
    fn all_presets_lookup_resolves_every_variant() {
        // Smoke test — every variant of the enum returns a real
        // constraint set so the lookup table can never silently miss
        // a preset.
        for &preset in PropFirmPreset::all() {
            let c = PropFirmConstraints::for_preset(preset);
            assert!(c.max_daily_loss_pct >= 0.0);
            let r = PropFirmRuntimeDefaults::for_preset(preset);
            assert!(r.max_trades_per_day > 0);
            let _ = PropFirmChallengeDefaults::for_preset(preset);
            let _ = PropFirmPhaseRiskDefaults::for_preset(preset, "phase_1");
        }
    }

    #[test]
    fn ftmo_runtime_defaults_stay_inside_hard_constraints() {
        let constraints = PropFirmConstraints::FTMO_STANDARD;
        let challenge = PropFirmChallengeDefaults::FTMO_STANDARD;
        let runtime = PropFirmRuntimeDefaults::FTMO_STANDARD;

        assert!(challenge.relaxed_min_trading_days < constraints.min_trading_days);
        assert!(challenge.max_trading_days > constraints.min_trading_days);
        assert!(challenge.daily_target_trading_days <= challenge.target_trading_days);
        assert!(runtime.daily_dd_warning_pct < runtime.daily_dd_stop_trading_pct);
        assert!(runtime.daily_dd_stop_trading_pct <= constraints.max_daily_loss_pct as f64);
        assert!(runtime.recovery_top_three_drawdown_pct < runtime.recovery_min_sharpe_drawdown_pct);
        assert!(
            runtime.recovery_min_sharpe_drawdown_pct < runtime.recovery_top_strategy_drawdown_pct
        );
        assert!(runtime.recovery_top_strategy_drawdown_pct < runtime.recovery_halt_drawdown_pct);
        assert!(runtime.recovery_halt_drawdown_pct <= constraints.max_overall_drawdown_pct as f64);
        assert!(runtime.recovery_mode_risk_multiplier < runtime.defensive_mode_risk_multiplier);
        assert!(runtime.defensive_mode_risk_multiplier < runtime.caution_mode_risk_multiplier);
        assert!(runtime.caution_mode_risk_multiplier <= 1.0);
    }

    #[test]
    fn ftmo_phase_risk_defaults_preserve_phase_ordering() {
        let phase_1 = PropFirmPhaseRiskDefaults::FTMO_PHASE_1;
        let phase_2 = PropFirmPhaseRiskDefaults::FTMO_PHASE_2;
        let funded = PropFirmPhaseRiskDefaults::FTMO_FUNDED;

        assert!(phase_2.risk_per_trade <= phase_1.risk_per_trade);
        assert!(phase_2.max_risk_per_trade <= phase_1.max_risk_per_trade);
        assert!(phase_2.min_confidence_threshold > funded.min_confidence_threshold);
        assert!(funded.max_trades_per_day >= phase_1.max_trades_per_day);
        assert_eq!(funded.daily_profit_lock_pct, 0.0);
    }
}
