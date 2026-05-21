use super::evolution_math::gene_signature_hash;
use super::search_engine::evaluate_genes;
use super::strategy_gene::{EvaluationConfig, Gene};
use anyhow::{Result, bail};
use neoethos_data::{FeatureFrame, Ohlcv};
use ndarray::s;
use serde::{Deserialize, Serialize};

const MILLIS_PER_DAY: i64 = 86_400_000;

pub type EvalMetrics = [f64; 11];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeWindow {
    pub window_id: usize,
    pub start_idx: usize,
    pub end_idx: usize,
    pub start_ts: i64,
    pub end_ts: i64,
    pub bars: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowPerformanceLabel {
    pub window_id: usize,
    pub start_ts: i64,
    pub end_ts: i64,
    pub bars: usize,
    pub net_profit: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub trades: f64,
    pub consistency: f64,
    pub quality_score: f64,
    pub active: bool,
    pub profitable: bool,
    pub tradable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyRegimeProfile {
    pub strategy_id: String,
    pub signature_hash: u64,
    pub evaluated_windows: usize,
    pub active_windows: usize,
    pub profitable_windows: usize,
    pub tradable_windows: usize,
    pub hit_rate: f64,
    pub active_rate: f64,
    pub tradable_rate: f64,
    pub best_window_score: f64,
    pub median_window_score: f64,
    pub avg_positive_window_score: f64,
    pub fragility_score: f64,
    pub regime_specialist_score: f64,
    pub always_on_score: f64,
    pub deployment_candidate: bool,
    pub training_candidate: bool,
    pub specialist_candidate: bool,
    pub labels: Vec<WindowPerformanceLabel>,
}

#[derive(Debug, Clone, Copy)]
pub struct RegimeLabelPolicy {
    pub window_days: i64,
    pub step_days: i64,
    pub min_bars_per_window: usize,
    pub min_trades_per_window: f64,
    pub min_window_profit: f64,
    pub min_profit_factor: f64,
    pub max_drawdown: f64,
    pub min_quality_score: f64,
    pub min_specialist_windows: usize,
    pub min_specialist_score: f64,
    pub min_always_on_hit_rate: f64,
}

impl Default for RegimeLabelPolicy {
    fn default() -> Self {
        Self {
            window_days: 90,
            step_days: 30,
            min_bars_per_window: 500,
            min_trades_per_window: 8.0,
            min_window_profit: 0.0,
            min_profit_factor: 1.05,
            max_drawdown: 0.20,
            min_quality_score: 0.05,
            min_specialist_windows: 2,
            min_specialist_score: 0.30,
            min_always_on_hit_rate: 0.55,
        }
    }
}

// `RegimeLabelPolicy::from_env` and its `env_i64` / `env_usize` / `env_f64`
// helpers were retired during Phase 22 of the consolidated audit
// follow-on: the constructor had no callers in this crate or any of its
// dependents, so the only behavior was reading 11 `FOREX_BOT_REGIME_LABEL_*`
// env vars on demand. Production code constructs `RegimeLabelPolicy`
// through its struct fields directly; if a future feature needs
// env-driven defaults, route them through a typed `*RuntimeOverrides`
// boundary like the others under `genetic::runtime_overrides`.

use neoethos_core::utils::finite_or;

pub fn build_rolling_regime_windows(
    timestamps: &[i64],
    window_days: i64,
    step_days: i64,
    min_bars_per_window: usize,
) -> Vec<RegimeWindow> {
    if timestamps.len() < min_bars_per_window || timestamps.is_empty() {
        return Vec::new();
    }

    let window_ms = window_days.max(1) * MILLIS_PER_DAY;
    let step_ms = step_days.max(1) * MILLIS_PER_DAY;
    let mut windows = Vec::new();
    let mut window_id = 0usize;
    let mut start_idx = 0usize;
    let mut start_ts = timestamps[0];
    let last_ts = *timestamps.last().unwrap_or(&timestamps[0]);

    while start_idx < timestamps.len() && start_ts <= last_ts {
        let end_ts = start_ts.saturating_add(window_ms);
        let mut end_idx = start_idx;
        while end_idx < timestamps.len() && timestamps[end_idx] < end_ts {
            end_idx += 1;
        }

        let bars = end_idx.saturating_sub(start_idx);
        if bars >= min_bars_per_window {
            windows.push(RegimeWindow {
                window_id,
                start_idx,
                end_idx,
                start_ts,
                end_ts: timestamps[end_idx.saturating_sub(1)],
                bars,
            });
            window_id += 1;
        }

        let next_start_ts = start_ts.saturating_add(step_ms);
        while start_idx < timestamps.len() && timestamps[start_idx] < next_start_ts {
            start_idx += 1;
        }
        if start_idx >= timestamps.len() {
            break;
        }
        start_ts = timestamps[start_idx];
    }

    windows
}

pub fn label_strategies_by_regime_windows(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    genes: &[Gene],
    eval_config: &EvaluationConfig,
    policy: RegimeLabelPolicy,
) -> Result<Vec<StrategyRegimeProfile>> {
    if genes.is_empty() {
        return Ok(Vec::new());
    }
    if features.timestamps.len() != features.data.nrows() {
        bail!("feature timestamp count does not match feature rows");
    }
    if ohlcv.close.len() != features.data.nrows() {
        bail!("OHLCV rows do not match feature rows");
    }

    let windows = build_rolling_regime_windows(
        &features.timestamps,
        policy.window_days,
        policy.step_days,
        policy.min_bars_per_window,
    );
    if windows.is_empty() {
        return Ok(genes.iter().map(|gene| empty_profile(gene)).collect());
    }

    let mut per_gene_labels: Vec<Vec<WindowPerformanceLabel>> = vec![Vec::new(); genes.len()];

    for window in &windows {
        let wf = slice_feature_frame(features, window.start_idx, window.end_idx);
        let wo = slice_ohlcv(
            ohlcv,
            window.start_idx,
            window.end_idx,
            &features.timestamps,
        );
        let metrics = evaluate_genes(&wf, &wo, genes, eval_config)?;

        for (gene_idx, metric) in metrics.iter().enumerate() {
            per_gene_labels[gene_idx].push(window_label(window, metric, &policy));
        }
    }

    Ok(genes
        .iter()
        .zip(per_gene_labels)
        .map(|(gene, labels)| summarize_profile(gene, labels, &policy))
        .collect())
}

fn slice_feature_frame(features: &FeatureFrame, start: usize, end: usize) -> FeatureFrame {
    FeatureFrame {
        timestamps: features.timestamps[start..end].to_vec(),
        names: features.names.clone(),
        data: features.data.slice(s![start..end, ..]).to_owned(),
    }
}

fn slice_ohlcv(ohlcv: &Ohlcv, start: usize, end: usize, fallback_timestamps: &[i64]) -> Ohlcv {
    neoethos_data::slice_ohlcv(ohlcv, start, end, Some(fallback_timestamps))
}

fn window_label(
    window: &RegimeWindow,
    metrics: &EvalMetrics,
    policy: &RegimeLabelPolicy,
) -> WindowPerformanceLabel {
    let net_profit = finite_or(metrics[0], 0.0);
    let sharpe = finite_or(metrics[1], 0.0);
    let max_drawdown = finite_or(metrics[3], 1.0).max(0.0);
    let win_rate = finite_or(metrics[4], 0.0).clamp(0.0, 1.0);
    let profit_factor = finite_or(metrics[5], 0.0).max(0.0);
    let expectancy = finite_or(metrics[6], 0.0);
    let trades = finite_or(metrics[8], 0.0).max(0.0);
    let consistency = finite_or(metrics[9], 0.0).clamp(0.0, 1.0);
    let active = trades >= policy.min_trades_per_window;
    let profitable = net_profit > policy.min_window_profit;
    let quality_score = window_quality_score(metrics);
    let tradable = active
        && profitable
        && profit_factor >= policy.min_profit_factor
        && max_drawdown <= policy.max_drawdown
        && quality_score >= policy.min_quality_score;

    WindowPerformanceLabel {
        window_id: window.window_id,
        start_ts: window.start_ts,
        end_ts: window.end_ts,
        bars: window.bars,
        net_profit,
        sharpe,
        max_drawdown,
        win_rate,
        profit_factor,
        expectancy,
        trades,
        consistency,
        quality_score,
        active,
        profitable,
        tradable,
    }
}

pub fn window_quality_score(metrics: &EvalMetrics) -> f64 {
    let net = finite_or(metrics[0], 0.0);
    let sharpe = finite_or(metrics[1], 0.0);
    let max_drawdown = finite_or(metrics[3], 1.0).max(0.0);
    let win_rate = finite_or(metrics[4], 0.0).clamp(0.0, 1.0);
    let profit_factor = finite_or(metrics[5], 0.0).max(0.0);
    let expectancy = finite_or(metrics[6], 0.0);
    let trades = finite_or(metrics[8], 0.0).max(0.0);
    let consistency = finite_or(metrics[9], 0.0).clamp(0.0, 1.0);

    let trade_confidence = (trades.sqrt() / 8.0).min(1.0);
    let net_component = (net / 2_500.0).clamp(-3.0, 3.0) * 0.20;
    let sharpe_component = sharpe.clamp(-2.0, 4.0) * 0.25 * trade_confidence;
    let pf_component = ((profit_factor - 1.0) * 0.80).clamp(-1.5, 2.5) * 0.20;
    let consistency_component = consistency * 0.15;
    let win_component = ((win_rate - 0.45) * 2.0).clamp(0.0, 1.0) * 0.10;
    let expectancy_component = (expectancy / 50.0).clamp(-1.0, 1.0) * 0.10;
    let drawdown_penalty = (max_drawdown * 8.0).min(3.0);

    net_component
        + sharpe_component
        + pf_component
        + consistency_component
        + win_component
        + expectancy_component
        - drawdown_penalty
}

fn summarize_profile(
    gene: &Gene,
    labels: Vec<WindowPerformanceLabel>,
    policy: &RegimeLabelPolicy,
) -> StrategyRegimeProfile {
    if labels.is_empty() {
        return empty_profile(gene);
    }

    let evaluated_windows = labels.len();
    let active_windows = labels.iter().filter(|label| label.active).count();
    let profitable_windows = labels.iter().filter(|label| label.profitable).count();
    let tradable_windows = labels.iter().filter(|label| label.tradable).count();
    let hit_rate = profitable_windows as f64 / evaluated_windows as f64;
    let active_rate = active_windows as f64 / evaluated_windows as f64;
    let tradable_rate = tradable_windows as f64 / evaluated_windows as f64;

    let mut scores: Vec<f64> = labels.iter().map(|label| label.quality_score).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let best_window_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let median_window_score = scores[scores.len() / 2];
    let positive_scores: Vec<f64> = scores
        .iter()
        .copied()
        .filter(|score| *score > 0.0)
        .collect();
    let avg_positive_window_score = if positive_scores.is_empty() {
        0.0
    } else {
        positive_scores.iter().sum::<f64>() / positive_scores.len() as f64
    };
    let negative_windows = labels
        .iter()
        .filter(|label| label.quality_score < 0.0)
        .count();
    let fragility_score = negative_windows as f64 / evaluated_windows as f64;

    let specialist_candidate = tradable_windows >= policy.min_specialist_windows
        && best_window_score >= policy.min_specialist_score;
    let regime_specialist_score = if specialist_candidate {
        (best_window_score.max(0.0) * (tradable_windows as f64).sqrt())
            * (1.0 - fragility_score * 0.35).max(0.25)
    } else {
        0.0
    };

    let always_on_score = (hit_rate * 0.35)
        + (tradable_rate * 0.35)
        + (median_window_score.max(0.0) * 0.20)
        + ((1.0 - fragility_score).max(0.0) * 0.10);
    let deployment_candidate = hit_rate >= policy.min_always_on_hit_rate
        && tradable_rate >= policy.min_always_on_hit_rate * 0.75
        && fragility_score <= 0.35;
    let training_candidate =
        specialist_candidate || deployment_candidate || avg_positive_window_score > 0.0;

    StrategyRegimeProfile {
        strategy_id: gene.strategy_id.clone(),
        signature_hash: gene_signature_hash(gene),
        evaluated_windows,
        active_windows,
        profitable_windows,
        tradable_windows,
        hit_rate,
        active_rate,
        tradable_rate,
        best_window_score,
        median_window_score,
        avg_positive_window_score,
        fragility_score,
        regime_specialist_score,
        always_on_score,
        deployment_candidate,
        training_candidate,
        specialist_candidate,
        labels,
    }
}

fn empty_profile(gene: &Gene) -> StrategyRegimeProfile {
    StrategyRegimeProfile {
        strategy_id: gene.strategy_id.clone(),
        signature_hash: gene_signature_hash(gene),
        evaluated_windows: 0,
        active_windows: 0,
        profitable_windows: 0,
        tradable_windows: 0,
        hit_rate: 0.0,
        active_rate: 0.0,
        tradable_rate: 0.0,
        best_window_score: 0.0,
        median_window_score: 0.0,
        avg_positive_window_score: 0.0,
        fragility_score: 1.0,
        regime_specialist_score: 0.0,
        always_on_score: 0.0,
        deployment_candidate: false,
        training_candidate: false,
        specialist_candidate: false,
        labels: Vec::new(),
    }
}

pub fn rank_training_profiles(
    mut profiles: Vec<StrategyRegimeProfile>,
) -> Vec<StrategyRegimeProfile> {
    profiles.sort_by(|a, b| {
        let score_a = a.regime_specialist_score.max(a.always_on_score);
        let score_b = b.regime_specialist_score.max(b.always_on_score);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.tradable_windows.cmp(&a.tradable_windows))
            .then_with(|| a.signature_hash.cmp(&b.signature_hash))
    });
    profiles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(
        net: f64,
        sharpe: f64,
        dd: f64,
        wr: f64,
        pf: f64,
        exp: f64,
        trades: f64,
        consistency: f64,
    ) -> EvalMetrics {
        [
            net,
            sharpe,
            100_000.0 + net,
            dd,
            wr,
            pf,
            exp,
            0.0,
            trades,
            consistency,
            0.0,
        ]
    }

    #[test]
    fn rolling_windows_can_find_sub_two_year_regimes() {
        let start = 1_700_000_000_000_i64;
        let timestamps: Vec<i64> = (0..730).map(|day| start + day * MILLIS_PER_DAY).collect();
        let windows = build_rolling_regime_windows(&timestamps, 90, 30, 30);

        assert!(windows.len() > 12);
        assert!(windows.iter().all(|w| w.bars >= 30));
    }

    #[test]
    fn specialist_profile_survives_even_with_low_full_period_hit_rate() {
        let gene = Gene {
            strategy_id: "specialist".to_string(),
            indices: vec![0, 1],
            weights: vec![1.0, 0.5],
            long_threshold: 0.2,
            short_threshold: -0.2,
            ..Default::default()
        };
        let policy = RegimeLabelPolicy {
            min_specialist_windows: 2,
            min_specialist_score: 0.20,
            min_always_on_hit_rate: 0.60,
            ..Default::default()
        };
        let labels = vec![
            window_label(
                &RegimeWindow {
                    window_id: 0,
                    start_idx: 0,
                    end_idx: 10,
                    start_ts: 0,
                    end_ts: 9,
                    bars: 10,
                },
                &metrics(2_000.0, 1.2, 0.04, 0.55, 1.5, 25.0, 20.0, 0.6),
                &policy,
            ),
            window_label(
                &RegimeWindow {
                    window_id: 1,
                    start_idx: 10,
                    end_idx: 20,
                    start_ts: 10,
                    end_ts: 19,
                    bars: 10,
                },
                &metrics(1_500.0, 1.1, 0.05, 0.53, 1.4, 18.0, 18.0, 0.6),
                &policy,
            ),
            window_label(
                &RegimeWindow {
                    window_id: 2,
                    start_idx: 20,
                    end_idx: 30,
                    start_ts: 20,
                    end_ts: 29,
                    bars: 10,
                },
                &metrics(-800.0, -0.5, 0.15, 0.42, 0.8, -20.0, 12.0, 0.2),
                &policy,
            ),
            window_label(
                &RegimeWindow {
                    window_id: 3,
                    start_idx: 30,
                    end_idx: 40,
                    start_ts: 30,
                    end_ts: 39,
                    bars: 10,
                },
                &metrics(-500.0, -0.2, 0.12, 0.45, 0.9, -10.0, 9.0, 0.2),
                &policy,
            ),
        ];

        let profile = summarize_profile(&gene, labels, &policy);
        assert!(profile.specialist_candidate);
        assert!(profile.training_candidate);
        assert!(!profile.deployment_candidate);
    }
}
