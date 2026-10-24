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
//! `MonthlyUpsideV1` is bound today: it routes fitness through
//! `crate::scoring::ga_fitness_monthly_upside_v1` using the evaluator's
//! completed-month return vector. The remaining variants are declared so the
//! autoresearch knobs can name them, but they do not change the GA until their
//! own §14.7 follow-ups land.
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
        }
    }

    /// Parses a snake_case selector name. Unknown names fall back to the
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
}
