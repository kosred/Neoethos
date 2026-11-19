//! The named scoring-table selector for the GA (docs/autoresearch-loop.md
//! §14.7; neo-671 B9).
//!
//! Today the GA's fitness is chosen by one bool
//! (`EvaluationConfig::growth_objective`, set from `DiscoveryMode`), so the
//! named tables in `scoring/named.rs` cannot be compared and the axis-B
//! objective variants cannot be expressed. This selector replaces that implicit
//! choice with an explicit one.
//!
//! `GaFitness` is the default and keeps the canonical three-way dispatch
//! (growth goal -> growth -> `ga_fitness`) byte-for-byte. Only
//! The four named tables and `MonthlyUpsideV1` are bound. The latter routes through
//! `crate::scoring::ga_fitness_monthly_upside_v1` using the evaluator's
//! completed-month return vector. NetPerBarInMarket and ExpectancyTStat remain
//! declared pending their own evaluation inputs and binding.
use super::monthly_upside::MonthlyGoalProfileV1;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FitnessTable {
    /// The canonical prop-firm consistency scorer (`ga_fitness`) and the
    /// unchanged growth-goal/growth boolean chain. THE DEFAULT.
    #[default]
    GaFitness,
    ArchiveScore,
    WindowScore,
    QualityScore,
    /// B1's "net per bar in market" table — declared, NOT bound yet.
    NetPerBarInMarket,
    /// B7's expectancy t-stat table — declared, NOT bound yet.
    ExpectancyTStat,
    /// B9's asymmetric monthly-upside table — BOUND to
    /// `crate::scoring::ga_fitness_monthly_upside_v1` in
    /// `genetic::evolution_math::apply_metrics`.
    MonthlyUpsideV1,
    /// A validated goal and bounded weight preset, transported with the recipe.
    MonthlyGoalProgressV1(MonthlyGoalProfileV1),
}

impl FitnessTable {
    /// The snake_case selector name, stable with the serde representation and
    /// with the B0 table level strings in the autoresearch crate.
    pub fn name(self) -> &'static str {
        match self {
            Self::GaFitness => "ga_fitness",
            Self::ArchiveScore => "archive_score",
            Self::WindowScore => "window_score",
            Self::QualityScore => "quality_score",
            Self::NetPerBarInMarket => "net_per_bar_in_market",
            Self::ExpectancyTStat => "expectancy_t_stat",
            Self::MonthlyUpsideV1 => "monthly_upside_v1",
            Self::MonthlyGoalProgressV1(_) => "monthly_goal_progress_v1",
        }
    }

    /// Parses a parameter-free selector name. Goal profiles require their
    /// explicit validated target and preset. Unknown names fall back to the
    /// canonical default rather than inventing a table.
    pub fn from_name(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "archive_score" => Self::ArchiveScore,
            "window_score" => Self::WindowScore,
            "quality_score" => Self::QualityScore,
            "net_per_bar_in_market" => Self::NetPerBarInMarket,
            "expectancy_t_stat" => Self::ExpectancyTStat,
            "monthly_upside_v1" => Self::MonthlyUpsideV1,
            _ => Self::GaFitness,
        }
    }

    /// Whether this selector is the canonical default, for serde's
    /// skip_serializing_if so default configs serialize exactly as before
    /// the field existed.
    pub fn is_default(&self) -> bool {
        *self == Self::GaFitness
    }

    pub fn requires_monthly_returns(self) -> bool {
        matches!(self, Self::MonthlyUpsideV1 | Self::MonthlyGoalProgressV1(_))
    }

    pub fn require_bound(self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !matches!(self, Self::NetPerBarInMarket | Self::ExpectancyTStat),
            "fitness table {} has no implemented evaluator",
            self.name()
        );
        if let Self::MonthlyGoalProgressV1(profile) = self {
            profile.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_the_canonical_ga_fitness_table() {
        assert_eq!(FitnessTable::default(), FitnessTable::GaFitness);
        assert!(FitnessTable::default().is_default());
        assert!(!FitnessTable::MonthlyUpsideV1.is_default());
    }

    #[test]
    fn from_name_round_trips_every_selector_under_the_snake_case_config_names() {
        for (name, table) in [
            ("ga_fitness", FitnessTable::GaFitness),
            ("archive_score", FitnessTable::ArchiveScore),
            ("window_score", FitnessTable::WindowScore),
            ("quality_score", FitnessTable::QualityScore),
            ("net_per_bar_in_market", FitnessTable::NetPerBarInMarket),
            ("expectancy_t_stat", FitnessTable::ExpectancyTStat),
            ("monthly_upside_v1", FitnessTable::MonthlyUpsideV1),
        ] {
            assert_eq!(FitnessTable::from_name(name), table, "{name}");
            assert_eq!(table.name(), name, "{name}");
        }
        assert_eq!(FitnessTable::from_name("unknown"), FitnessTable::GaFitness);
    }

    #[test]
    fn serde_names_match_from_name_so_config_json_round_trips() {
        let json = serde_json::to_string(&FitnessTable::MonthlyUpsideV1).unwrap();
        assert_eq!(json, "\"monthly_upside_v1\"");
        let parsed: FitnessTable =
            serde_json::from_str(&json).expect("snake_case selector deserializes");
        assert_eq!(parsed, FitnessTable::MonthlyUpsideV1);
    }

    #[test]
    fn parameterized_goal_table_preserves_the_preset_and_frozen_target_on_the_wire() {
        let profile = MonthlyGoalProfileV1::new(20, 0.04).unwrap();
        let table = FitnessTable::MonthlyGoalProgressV1(profile);
        let json = serde_json::to_string(&table).unwrap();
        assert_eq!(serde_json::from_str::<FitnessTable>(&json).unwrap(), table);
        assert!(table.requires_monthly_returns());
        table.require_bound().unwrap();
        assert!(FitnessTable::ExpectancyTStat.require_bound().is_err());
        assert!(FitnessTable::NetPerBarInMarket.require_bound().is_err());
        assert!(
            serde_json::from_str::<FitnessTable>(
                r#"{"monthly_goal_progress_v1":{"preset":255,"monthly_target_ppm":40000}}"#
            )
            .is_err()
        );
    }
}
