use crate::genetic::{evolve_search_with_progress, signals_for_gene, Gene};
use anyhow::Result;
use chrono::{Datelike, TimeZone, Utc};
use forex_data::{FeatureFrame, Ohlcv};
use serde::Serialize;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub population: usize,
    pub generations: usize,
    pub max_indicators: usize,
    pub candidate_count: usize,
    pub portfolio_size: usize,
    pub corr_threshold: f64,
    pub min_trades_per_day: f64,
    pub filtering: crate::genetic::FilteringConfig,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            population: 1000,
            generations: 10,
            max_indicators: 12,
            candidate_count: 5000,
            portfolio_size: 2000,
            corr_threshold: 0.85,
            min_trades_per_day: 0.2,
            filtering: crate::genetic::FilteringConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    pub portfolio: Vec<Gene>,
    pub candidates: Vec<Gene>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiscoveryProgress {
    SearchStarted {
        population: usize,
        generations: usize,
        max_indicators: usize,
    },
    GenerationCompleted {
        generation: usize,
        total_generations: usize,
        best_fitness: f64,
        stagnant_generations: usize,
        archived_profitable: usize,
    },
    CandidatesRanked {
        candidate_count: usize,
        truncated_to: usize,
    },
    CandidatesFiltered {
        passed_filters: usize,
        evaluated_candidates: usize,
        min_trades_required: usize,
    },
    PortfolioSelected {
        portfolio_size: usize,
        rejected_by_correlation: usize,
        target_portfolio: usize,
    },
    Completed {
        candidate_count: usize,
        filtered_count: usize,
        portfolio_size: usize,
    },
}

pub fn ensure_non_empty_portfolio(result: &DiscoveryResult, context: &str) -> Result<()> {
    if result.portfolio.is_empty() {
        anyhow::bail!(
            "Discovery produced an empty portfolio for {} (candidates={})",
            context,
            result.candidates.len()
        );
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct GeneExport<'a> {
    strategy_id: &'a str,
    indicators: Vec<&'a str>,
    indices: Vec<usize>,
    weights: Vec<f32>,
    long_threshold: f32,
    short_threshold: f32,
    fitness: f64,
    sharpe_ratio: f64,
    win_rate: f64,
    tp_pips: f64,
    sl_pips: f64,
}

pub fn run_discovery_cycle(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
) -> Result<DiscoveryResult> {
    run_discovery_cycle_with_progress(features, ohlcv, config, |_| {})
}

pub fn run_discovery_cycle_with_progress<F>(
    features: &FeatureFrame,
    ohlcv: &Ohlcv,
    config: &DiscoveryConfig,
    mut progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    progress_fn(DiscoveryProgress::SearchStarted {
        population: config.population,
        generations: config.generations,
        max_indicators: config.max_indicators,
    });
    let search = evolve_search_with_progress(
        features,
        ohlcv,
        config.population,
        config.generations,
        config.max_indicators,
        |generation, total_generations, best_fitness, stagnant_generations, archived_profitable| {
            progress_fn(DiscoveryProgress::GenerationCompleted {
                generation,
                total_generations,
                best_fitness,
                stagnant_generations,
                archived_profitable,
            });
        },
    )?;

    finalize_candidates_with_progress(search.genes, features, config, progress_fn)
}

fn finalize_candidates_with_progress<F>(
    mut candidates: Vec<Gene>,
    features: &FeatureFrame,
    config: &DiscoveryConfig,
    mut progress_fn: F,
) -> Result<DiscoveryResult>
where
    F: FnMut(DiscoveryProgress),
{
    // Sort by a weighted combo of fitness and consistency to find reliably profitable ones
    candidates.sort_by(|a, b| {
        let score_a = a.fitness + (a.consistency * 0.5);
        let score_b = b.fitness + (b.consistency * 0.5);
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let max_candidates = config.candidate_count.max(100).min(candidates.len());
    candidates.truncate(max_candidates);
    progress_fn(DiscoveryProgress::CandidatesRanked {
        candidate_count: candidates.len(),
        truncated_to: max_candidates,
    });

    let min_trades = min_trades_required(
        &features.timestamps,
        config.min_trades_per_day,
        features.data.nrows(),
    );
    let mut filtered = Vec::new();
    let mut signals_map = Vec::new();
    for gene in &candidates {
        if !gene.passes_filter(&config.filtering) {
            continue;
        }
        let sig = signals_for_gene(features, gene);
        let trade_count = sig.iter().filter(|v| **v != 0).count() as f64;
        if trade_count >= min_trades as f64 {
            filtered.push(gene.clone());
            signals_map.push(sig);
        }
    }
    progress_fn(DiscoveryProgress::CandidatesFiltered {
        passed_filters: filtered.len(),
        evaluated_candidates: candidates.len(),
        min_trades_required: min_trades,
    });

    let filtered_count = filtered.len();
    let mut portfolio = Vec::new();
    let mut portfolio_signals: Vec<Vec<i8>> = Vec::new();
    let mut rejected_by_correlation = 0usize;
    for (gene, sig) in filtered.into_iter().zip(signals_map.into_iter()) {
        if portfolio.len() >= config.portfolio_size {
            break;
        }
        let mut ok = true;
        for existing in &portfolio_signals {
            let corr = pearson_corr_i8(&sig, existing);
            if corr.abs() >= config.corr_threshold {
                ok = false;
                rejected_by_correlation += 1;
                break;
            }
        }
        if ok {
            portfolio_signals.push(sig);
            portfolio.push(gene);
        }
    }
    progress_fn(DiscoveryProgress::PortfolioSelected {
        portfolio_size: portfolio.len(),
        rejected_by_correlation,
        target_portfolio: config.portfolio_size,
    });
    progress_fn(DiscoveryProgress::Completed {
        candidate_count: candidates.len(),
        filtered_count,
        portfolio_size: portfolio.len(),
    });

    Ok(DiscoveryResult {
        portfolio,
        candidates,
    })
}

fn min_trades_required(timestamps: &[i64], min_trades_per_day: f64, n_rows: usize) -> usize {
    if timestamps.is_empty() {
        let days = (n_rows as f64 / 1440.0).max(1.0);
        return (days * min_trades_per_day).ceil() as usize;
    }
    let mut days = HashSet::new();
    for ts in timestamps {
        if let Some(dt) = Utc.timestamp_millis_opt(*ts).single() {
            if dt.weekday().num_days_from_monday() < 5 {
                let key = (dt.year() as i64) * 10000 + (dt.month() as i64) * 100 + dt.day() as i64;
                days.insert(key);
            }
        }
    }
    let day_count = days.len().max(1) as f64;
    (day_count * min_trades_per_day).ceil() as usize
}

fn pearson_corr_i8(a: &[i8], b: &[i8]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    for i in 0..n {
        sum_a += a[i] as f64;
        sum_b += b[i] as f64;
    }
    let mean_a = sum_a / n as f64;
    let mean_b = sum_b / n as f64;
    let mut num = 0.0;
    let mut denom_a = 0.0;
    let mut denom_b = 0.0;
    for i in 0..n {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    if denom_a <= 1e-12 || denom_b <= 1e-12 {
        return 0.0;
    }
    num / (denom_a.sqrt() * denom_b.sqrt())
}

pub fn save_portfolio_json(
    path: impl AsRef<Path>,
    portfolio: &[Gene],
    feature_names: &[String],
) -> Result<()> {
    let mut exports = Vec::new();
    for gene in portfolio {
        let mut names = Vec::new();
        for idx in &gene.indices {
            if let Some(name) = feature_names.get(*idx) {
                names.push(name.as_str());
            }
        }
        exports.push(GeneExport {
            strategy_id: &gene.strategy_id,
            indicators: names,
            indices: gene.indices.clone(),
            weights: gene.weights.clone(),
            long_threshold: gene.long_threshold,
            short_threshold: gene.short_threshold,
            fitness: gene.fitness,
            sharpe_ratio: gene.sharpe_ratio,
            win_rate: gene.win_rate,
            tp_pips: gene.tp_pips,
            sl_pips: gene.sl_pips,
        });
    }
    let payload = serde_json::to_string_pretty(&exports)?;
    fs::write(path, payload)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FilteringConfig;
    use ndarray::Array2;

    fn sample_feature_frame() -> FeatureFrame {
        let start = 1_704_067_200_000_i64;
        FeatureFrame {
            timestamps: (0..10).map(|idx| start + idx * 60_000).collect(),
            names: vec!["signal".to_string()],
            data: Array2::from_shape_vec(
                (10, 1),
                vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            )
            .expect("feature frame shape should be valid"),
        }
    }

    fn profitable_gene(strategy_id: &str) -> Gene {
        Gene {
            strategy_id: strategy_id.to_string(),
            indices: vec![0],
            weights: vec![1.0],
            long_threshold: 0.5,
            short_threshold: -0.5,
            fitness: 150.0,
            sharpe_ratio: 1.4,
            win_rate: 0.61,
            max_drawdown: 0.04,
            profit_factor: 1.3,
            trades_count: 10,
            consistency: 0.8,
            ..Gene::default()
        }
    }

    #[test]
    fn empty_portfolio_is_an_explicit_error() {
        let result = DiscoveryResult {
            portfolio: Vec::new(),
            candidates: vec![Gene::default()],
        };

        let err = ensure_non_empty_portfolio(&result, "EURUSD M1")
            .expect_err("expected empty discovery portfolio to fail");
        let msg = err.to_string();
        assert!(msg.contains("empty portfolio"), "unexpected error: {msg}");
        assert!(msg.contains("candidates=1"), "unexpected error: {msg}");
    }

    #[test]
    fn non_empty_portfolio_is_accepted() {
        let result = DiscoveryResult {
            portfolio: vec![Gene::default()],
            candidates: vec![Gene::default()],
        };

        ensure_non_empty_portfolio(&result, "EURUSD M1")
            .expect("expected non-empty portfolio to pass");
    }

    #[test]
    fn finalize_candidates_with_progress_emits_filter_and_portfolio_milestones() {
        let features = sample_feature_frame();
        let config = DiscoveryConfig {
            candidate_count: 2,
            portfolio_size: 2,
            corr_threshold: 0.9,
            min_trades_per_day: 1.0,
            filtering: FilteringConfig {
                min_profit: 1.0,
                min_trades: 1.0,
                min_sharpe: 0.1,
                min_win_rate: 0.5,
                min_profit_factor: 1.01,
                max_dd: 0.2,
                anomaly_guard: false,
                elite_mode: false,
            },
            ..DiscoveryConfig::default()
        };
        let candidates = vec![profitable_gene("alpha-1"), profitable_gene("alpha-2")];
        let mut progress_events = Vec::new();

        let result = finalize_candidates_with_progress(candidates, &features, &config, |event| {
            progress_events.push(event)
        })
        .expect("candidate finalization should succeed");

        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.portfolio.len(), 1);
        assert!(progress_events.iter().any(|event| matches!(
            event,
            DiscoveryProgress::CandidatesRanked { candidate_count, truncated_to }
                if *candidate_count == 2 && *truncated_to == 2
        )));
        assert!(progress_events.iter().any(|event| matches!(
            event,
            DiscoveryProgress::CandidatesFiltered { passed_filters, evaluated_candidates, min_trades_required }
                if *passed_filters == 2 && *evaluated_candidates == 2 && *min_trades_required == 1
        )));
        assert!(progress_events.iter().any(|event| matches!(
            event,
            DiscoveryProgress::PortfolioSelected { portfolio_size, rejected_by_correlation, target_portfolio }
                if *portfolio_size == 1 && *rejected_by_correlation == 1 && *target_portfolio == 2
        )));
        assert!(progress_events.iter().any(|event| matches!(
            event,
            DiscoveryProgress::Completed { candidate_count, filtered_count, portfolio_size }
                if *candidate_count == 2 && *filtered_count == 2 && *portfolio_size == 1
        )));
    }
}
