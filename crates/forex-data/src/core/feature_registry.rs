use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::core::all_indicators::ALL_INDICATORS;
use crate::core::resample::parse_timeframe_to_minutes;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureSource {
    SmartMoneyConcept,
    ClassicTechnicalAnalysis,
    Quantitative,
    Session,
    Regime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureValueDtype {
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureValueKind {
    Continuous,
    Binary,
    SignedSignal,
    Ratio,
    Distance,
    State,
    Count,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureParameterKind {
    Timeframe,
    IndicatorId,
    ParameterSet,
    Period,
    LagBars,
    WindowBars,
    OutputLine,
    Session,
    Formula,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureOutputSchema {
    pub dtype: FeatureValueDtype,
    pub kind: FeatureValueKind,
    pub nullable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureParameterMetadata {
    pub name: String,
    pub kind: FeatureParameterKind,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureColumnMetadata {
    pub name: String,
    pub source: FeatureSource,
    pub output: FeatureOutputSchema,
    pub parameters: Vec<FeatureParameterMetadata>,
    pub requires_volume: bool,
}

const SMC_FEATURE_NAMES: &[&str] = &[
    "smc_ob",
    "smc_fvg",
    "smc_ifvg",
    "smc_liq_sweep",
    "smc_pd_array",
    "smc_killzone",
    "smc_displacement",
    "smc_breaker_block",
    "smc_mitigation_block",
    "smc_mss",
    "smc_volume_imbalance",
    "smc_bos",
    "smc_eqh",
    "smc_eql",
    "smc_inducement",
    "smc_asian_range",
    "smc_silver_bullet",
    "smc_judas_swing",
    "smc_nwog",
    "smc_ndog",
    "smc_ict_macro",
    "smc_fvg_strength",
    "smc_dealing_range_width",
    "smc_swing_range_pct",
    "smc_ob_strength",
    "smc_trend_bias",
    "smc_unicorn_model",
    "smc_rejection_block",
    "smc_propulsion_block",
    "smc_fib_time_ratio",
    "smc_fib_236",
    "smc_fib_382",
    "smc_fib_500",
    "smc_fib_618",
    "smc_fib_705",
    "smc_fib_786",
    "smc_fib_886",
    "smc_fib_1272",
    "smc_fib_1414",
    "smc_fib_1618",
    "smc_fib_2000",
    "smc_fib_2618",
];

const SESSION_FEATURE_NAMES: &[&str] = &[
    "session_london_open_dist",
    "session_london_high_dist",
    "session_london_low_dist",
    "session_london_range",
    "session_london_vwap_dist",
    "session_ny_open_dist",
    "session_ny_high_dist",
    "session_ny_low_dist",
    "session_ny_range",
    "session_ny_vwap_dist",
    "session_asian_open_dist",
    "session_asian_close_dist",
    "session_asian_range_norm",
    "session_london_ny_overlap",
    "session_vol_ratio",
    "session_prev_close_dist",
    "session_open_gap",
    "daily_range_pct",
    "daily_body_pct",
    "daily_position",
    "daily_high_dist",
    "daily_low_dist",
    "daily_vwap_dist",
];

const REGIME_FEATURE_NAMES: &[&str] = &[
    "regime_vol_state",
    "regime_vol_zscore",
    "regime_trend_strength",
    "regime_trend_direction",
    "regime_trend_state",
    "regime_squeeze",
    "regime_squeeze_momentum",
    "regime_mr_vs_momentum",
    "regime_rei",
    "regime_choppiness",
    "regime_cusum_up",
    "regime_cusum_down",
    "regime_change_signal",
    "regime_entropy",
];

const CLASSIC_MULTI_PERIOD_IDS: &[&str] = &[
    "rsi",
    "ema",
    "sma",
    "atr",
    "adx",
    "cci",
    "stoch",
    "macd",
    "bollinger_bands",
    "keltner",
    "supertrend",
    "willr",
    "roc",
    "mom",
    "tsi",
    "mfi",
    "obv",
    "vwap",
];

const CLASSIC_ALT_PERIODS: &[usize] = &[7, 21, 50, 100, 200];

const QUANT_EXACT_FEATURES: &[(&str, bool)] = &[
    ("quant_log_return", false),
    ("quant_vol_ratio", false),
    ("quant_hurst_100", false),
    ("quant_skewness_30", false),
    ("quant_kurtosis_30", false),
    ("quant_kyle_lambda", true),
    ("quant_vpin", true),
    ("quant_amihud_illiquidity", true),
    ("quant_roll_spread", false),
    ("quant_consec_up", false),
    ("quant_consec_down", false),
    ("quant_inside_bar", false),
    ("quant_outside_bar", false),
    ("quant_body_ratio", false),
    ("quant_upper_shadow", false),
    ("quant_lower_shadow", false),
    ("quant_prev_day_h_dist", false),
    ("quant_prev_day_l_dist", false),
    ("quant_prev_week_h_dist", false),
    ("quant_prev_week_l_dist", false),
    ("quant_amd_phase", false),
    ("quant_wyckoff", false),
    ("quant_engulfing_vol", true),
    ("quant_pivot_dist", false),
    ("quant_r1_dist", false),
    ("quant_r2_dist", false),
    ("quant_s1_dist", false),
    ("quant_s2_dist", false),
    ("quant_cam_r3_dist", false),
    ("quant_cam_s3_dist", false),
    ("quant_fractal_dim", false),
    ("quant_delta_volume", true),
    ("quant_cum_delta_zscore", true),
];

pub fn feature_column_metadata(name: &str) -> Option<FeatureColumnMetadata> {
    let (base_name, timeframe) = strip_timeframe_prefix(name);
    let mut metadata = feature_column_metadata_unprefixed(base_name)?;
    metadata.name = name.to_string();

    if let Some(timeframe) = timeframe {
        metadata.parameters.insert(
            0,
            parameter("timeframe", FeatureParameterKind::Timeframe, timeframe),
        );
    }

    Some(metadata)
}

pub fn feature_metadata_for_names(names: &[String]) -> Result<Vec<FeatureColumnMetadata>> {
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let Some(metadata) = feature_column_metadata(name) else {
            bail!("unregistered feature column: {name}");
        };
        out.push(metadata);
    }
    Ok(out)
}

pub fn validate_feature_names(names: &[String]) -> Result<()> {
    let unknown = unknown_feature_names(names);
    if !unknown.is_empty() {
        bail!("unregistered feature columns: {}", unknown.join(", "));
    }
    Ok(())
}

pub fn unknown_feature_names(names: &[String]) -> Vec<String> {
    names
        .iter()
        .filter(|name| feature_column_metadata(name).is_none())
        .cloned()
        .collect()
}

fn feature_column_metadata_unprefixed(name: &str) -> Option<FeatureColumnMetadata> {
    if SMC_FEATURE_NAMES.contains(&name) {
        return Some(group_metadata(
            name,
            FeatureSource::SmartMoneyConcept,
            infer_value_kind(name),
            false,
            smc_parameters(name),
        ));
    }

    if SESSION_FEATURE_NAMES.contains(&name) {
        return Some(group_metadata(
            name,
            FeatureSource::Session,
            infer_value_kind(name),
            false,
            session_parameters(name),
        ));
    }

    if REGIME_FEATURE_NAMES.contains(&name) {
        return Some(group_metadata(
            name,
            FeatureSource::Regime,
            infer_value_kind(name),
            false,
            regime_parameters(name),
        ));
    }

    if let Some(metadata) = quant_metadata(name) {
        return Some(metadata);
    }

    classic_ta_metadata(name)
}

fn strip_timeframe_prefix(name: &str) -> (&str, Option<&str>) {
    let Some((candidate, rest)) = name.split_once('_') else {
        return (name, None);
    };

    if parse_timeframe_to_minutes(candidate).is_ok() {
        (rest, Some(candidate))
    } else {
        (name, None)
    }
}

fn quant_metadata(name: &str) -> Option<FeatureColumnMetadata> {
    for (candidate, requires_volume) in QUANT_EXACT_FEATURES {
        if name == *candidate {
            return Some(group_metadata(
                name,
                FeatureSource::Quantitative,
                infer_value_kind(name),
                *requires_volume,
                quant_exact_parameters(name),
            ));
        }
    }

    let parameterized = [
        (
            "quant_return_",
            FeatureParameterKind::LagBars,
            &[1, 2, 3, 5, 8, 13, 21][..],
            false,
        ),
        (
            "quant_realized_vol_",
            FeatureParameterKind::WindowBars,
            &[5, 10, 20, 50][..],
            false,
        ),
        (
            "quant_gk_vol_",
            FeatureParameterKind::WindowBars,
            &[10, 20][..],
            false,
        ),
        (
            "quant_parkinson_vol_",
            FeatureParameterKind::WindowBars,
            &[10, 20][..],
            false,
        ),
        (
            "quant_autocorr_",
            FeatureParameterKind::LagBars,
            &[1, 5, 10][..],
            false,
        ),
        (
            "quant_efficiency_ratio_",
            FeatureParameterKind::WindowBars,
            &[10, 20][..],
            false,
        ),
        (
            "quant_orb_",
            FeatureParameterKind::WindowBars,
            &[4, 8, 12][..],
            false,
        ),
        (
            "quant_zscore_",
            FeatureParameterKind::WindowBars,
            &[20, 50][..],
            false,
        ),
        (
            "quant_rvol_",
            FeatureParameterKind::WindowBars,
            &[10, 20, 50][..],
            true,
        ),
    ];

    for (prefix, kind, allowed_values, requires_volume) in parameterized {
        if let Some(value) = numeric_suffix(name, prefix, allowed_values) {
            return Some(group_metadata(
                name,
                FeatureSource::Quantitative,
                infer_value_kind(name),
                requires_volume,
                vec![parameter(parameter_name(kind), kind, value.to_string())],
            ));
        }
    }

    None
}

fn classic_ta_metadata(name: &str) -> Option<FeatureColumnMetadata> {
    if let Some((indicator_id, period, line)) = classic_multi_period_parts(name) {
        let mut parameters = vec![
            parameter(
                "indicator_id",
                FeatureParameterKind::IndicatorId,
                indicator_id,
            ),
            parameter("period", FeatureParameterKind::Period, period.to_string()),
        ];
        if let Some(line) = line {
            parameters.push(parameter(
                "output_line",
                FeatureParameterKind::OutputLine,
                line.to_string(),
            ));
        }

        return Some(group_metadata(
            name,
            FeatureSource::ClassicTechnicalAnalysis,
            infer_value_kind(name),
            classic_indicator_requires_volume(indicator_id),
            parameters,
        ));
    }

    if let Some((indicator_id, line)) = classic_default_parts(name) {
        let mut parameters = vec![
            parameter(
                "indicator_id",
                FeatureParameterKind::IndicatorId,
                indicator_id,
            ),
            parameter("params", FeatureParameterKind::ParameterSet, "default"),
        ];
        if let Some(line) = line {
            parameters.push(parameter(
                "output_line",
                FeatureParameterKind::OutputLine,
                line.to_string(),
            ));
        }

        return Some(group_metadata(
            name,
            FeatureSource::ClassicTechnicalAnalysis,
            infer_value_kind(name),
            classic_indicator_requires_volume(indicator_id),
            parameters,
        ));
    }

    None
}

fn classic_multi_period_parts(name: &str) -> Option<(&'static str, usize, Option<usize>)> {
    for indicator_id in CLASSIC_MULTI_PERIOD_IDS {
        for period in CLASSIC_ALT_PERIODS {
            let exact = format!("{indicator_id}_{period}");
            if name == exact {
                return Some((indicator_id, *period, None));
            }

            let line_prefix = format!("{indicator_id}_{period}_line");
            if let Some(line) = name
                .strip_prefix(line_prefix.as_str())
                .and_then(|suffix| suffix.parse::<usize>().ok())
            {
                return Some((indicator_id, *period, Some(line)));
            }
        }
    }
    None
}

fn classic_default_parts(name: &str) -> Option<(&'static str, Option<usize>)> {
    for indicator_id in ALL_INDICATORS {
        if name == *indicator_id {
            return Some((indicator_id, None));
        }

        let line_prefix = format!("{indicator_id}_line");
        if let Some(line) = name
            .strip_prefix(line_prefix.as_str())
            .and_then(|suffix| suffix.parse::<usize>().ok())
        {
            return Some((indicator_id, Some(line)));
        }
    }
    None
}

fn numeric_suffix(name: &str, prefix: &str, allowed_values: &[usize]) -> Option<usize> {
    let value = name.strip_prefix(prefix)?.parse::<usize>().ok()?;
    allowed_values.contains(&value).then_some(value)
}

fn group_metadata(
    name: &str,
    source: FeatureSource,
    kind: FeatureValueKind,
    requires_volume: bool,
    parameters: Vec<FeatureParameterMetadata>,
) -> FeatureColumnMetadata {
    FeatureColumnMetadata {
        name: name.to_string(),
        source,
        output: FeatureOutputSchema {
            dtype: FeatureValueDtype::F64,
            kind,
            nullable: false,
        },
        parameters,
        requires_volume,
    }
}

fn parameter(
    name: &str,
    kind: FeatureParameterKind,
    value: impl ToString,
) -> FeatureParameterMetadata {
    FeatureParameterMetadata {
        name: name.to_string(),
        kind,
        value: value.to_string(),
    }
}

fn parameter_name(kind: FeatureParameterKind) -> &'static str {
    match kind {
        FeatureParameterKind::LagBars => "lag_bars",
        FeatureParameterKind::WindowBars => "window_bars",
        FeatureParameterKind::Period => "period",
        _ => "parameter",
    }
}

fn smc_parameters(name: &str) -> Vec<FeatureParameterMetadata> {
    let mut parameters = Vec::new();
    if name.contains("_fib_") || name == "smc_pd_array" || name == "smc_dealing_range_width" {
        parameters.push(parameter(
            "lookback_bars",
            FeatureParameterKind::WindowBars,
            40,
        ));
    }
    if matches!(name, "smc_eqh" | "smc_eql" | "smc_bos") {
        parameters.push(parameter(
            "swing_fractal",
            FeatureParameterKind::Formula,
            "5_bar",
        ));
    }
    parameters
}

fn session_parameters(name: &str) -> Vec<FeatureParameterMetadata> {
    let mut parameters = vec![parameter(
        "timestamp_policy",
        FeatureParameterKind::Formula,
        "utc_session_windows",
    )];

    if name.contains("london") {
        parameters.push(parameter(
            "session",
            FeatureParameterKind::Session,
            "London",
        ));
    } else if name.contains("_ny_") {
        parameters.push(parameter(
            "session",
            FeatureParameterKind::Session,
            "NewYork",
        ));
    } else if name.contains("asian") {
        parameters.push(parameter("session", FeatureParameterKind::Session, "Asian"));
    } else if name.starts_with("daily_") {
        parameters.push(parameter("session", FeatureParameterKind::Session, "Daily"));
    }

    parameters
}

fn regime_parameters(name: &str) -> Vec<FeatureParameterMetadata> {
    let mut parameters = Vec::new();
    if name.contains("trend") {
        parameters.push(parameter("period", FeatureParameterKind::Period, 14));
    } else if name.contains("squeeze") {
        parameters.push(parameter(
            "window_bars",
            FeatureParameterKind::WindowBars,
            20,
        ));
    } else if name.contains("entropy") {
        parameters.push(parameter(
            "window_bars",
            FeatureParameterKind::WindowBars,
            50,
        ));
    }
    parameters
}

fn quant_exact_parameters(name: &str) -> Vec<FeatureParameterMetadata> {
    match name {
        "quant_hurst_100" => vec![parameter(
            "window_bars",
            FeatureParameterKind::WindowBars,
            100,
        )],
        "quant_skewness_30" | "quant_kurtosis_30" | "quant_fractal_dim" => {
            vec![parameter(
                "window_bars",
                FeatureParameterKind::WindowBars,
                30,
            )]
        }
        "quant_kyle_lambda" | "quant_amihud_illiquidity" | "quant_roll_spread" => {
            vec![parameter(
                "window_bars",
                FeatureParameterKind::WindowBars,
                20,
            )]
        }
        "quant_vpin" => vec![
            parameter("bucket_size_bars", FeatureParameterKind::WindowBars, 50),
            parameter("bucket_count", FeatureParameterKind::Formula, 10),
        ],
        "quant_prev_day_h_dist"
        | "quant_prev_day_l_dist"
        | "quant_pivot_dist"
        | "quant_r1_dist"
        | "quant_r2_dist"
        | "quant_s1_dist"
        | "quant_s2_dist"
        | "quant_cam_r3_dist"
        | "quant_cam_s3_dist" => {
            vec![parameter(
                "window_bars",
                FeatureParameterKind::WindowBars,
                24,
            )]
        }
        "quant_prev_week_h_dist" | "quant_prev_week_l_dist" => {
            vec![parameter(
                "window_bars",
                FeatureParameterKind::WindowBars,
                120,
            )]
        }
        "quant_amd_phase" => vec![parameter(
            "window_bars",
            FeatureParameterKind::WindowBars,
            20,
        )],
        "quant_wyckoff" => vec![parameter(
            "window_bars",
            FeatureParameterKind::WindowBars,
            30,
        )],
        _ => Vec::new(),
    }
}

fn classic_indicator_requires_volume(indicator_id: &str) -> bool {
    indicator_id.contains("volume")
        || indicator_id.contains("vwap")
        || matches!(
            indicator_id,
            "ad" | "adosc" | "mfi" | "obv" | "vpt" | "vosc" | "vpci" | "vwma" | "vwmacd"
        )
}

fn infer_value_kind(name: &str) -> FeatureValueKind {
    if name.contains("overlap")
        || name.contains("killzone")
        || name.contains("silver_bullet")
        || name.contains("ict_macro")
        || name.contains("inside_bar")
        || name.contains("outside_bar")
        || name.contains("squeeze")
    {
        FeatureValueKind::Binary
    } else if name.contains("state") || name.contains("phase") || name.contains("regime") {
        FeatureValueKind::State
    } else if name.contains("signal")
        || name.contains("direction")
        || name.contains("bias")
        || name.contains("swing")
        || name.ends_with("_bos")
        || name.ends_with("_mss")
    {
        FeatureValueKind::SignedSignal
    } else if name.contains("ratio")
        || name.contains("_pct")
        || name.contains("_range")
        || name.contains("vol_")
        || name.contains("_vol")
    {
        FeatureValueKind::Ratio
    } else if name.contains("dist") || name.contains("zscore") || name.contains("z_score") {
        FeatureValueKind::Distance
    } else if name.contains("count") || name.contains("consec") {
        FeatureValueKind::Count
    } else {
        FeatureValueKind::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_explicit_feature_groups() {
        for name in SMC_FEATURE_NAMES
            .iter()
            .chain(SESSION_FEATURE_NAMES)
            .chain(REGIME_FEATURE_NAMES)
        {
            assert!(
                feature_column_metadata(name).is_some(),
                "{name} should have registry metadata"
            );
        }
    }

    #[test]
    fn resolves_parameterized_quant_features() {
        let rvol = feature_column_metadata("quant_rvol_20").expect("rvol metadata");
        assert_eq!(rvol.source, FeatureSource::Quantitative);
        assert!(rvol.requires_volume);
        assert_eq!(rvol.parameters[0].name, "window_bars");

        let prefixed = feature_column_metadata("H1_quant_return_13").expect("prefixed metadata");
        assert_eq!(prefixed.name, "H1_quant_return_13");
        assert_eq!(prefixed.parameters[0].kind, FeatureParameterKind::Timeframe);
        assert_eq!(prefixed.parameters[1].kind, FeatureParameterKind::LagBars);
    }

    #[test]
    fn resolves_vector_ta_defaults_and_period_variants() {
        let default_rsi = feature_column_metadata("rsi").expect("rsi metadata");
        assert_eq!(default_rsi.source, FeatureSource::ClassicTechnicalAnalysis);

        let period_line =
            feature_column_metadata("bollinger_bands_21_line2").expect("period line metadata");
        assert_eq!(period_line.parameters[1].kind, FeatureParameterKind::Period);
        assert_eq!(
            period_line.parameters[2].kind,
            FeatureParameterKind::OutputLine
        );
    }

    #[test]
    fn rejects_unknown_feature_names() {
        let names = vec!["quant_return_1".to_string(), "quant_return_4".to_string()];
        let unknown = unknown_feature_names(&names);
        assert_eq!(unknown, vec!["quant_return_4".to_string()]);
        assert!(validate_feature_names(&names).is_err());
    }
}
