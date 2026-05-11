use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub symbol: String,
    pub weight: f64,
    pub kelly_size: f64,
    pub risk_budget: f64,
    pub correlation_score: f64,
    pub sharpe: f64,
}

#[derive(Debug, Clone)]
pub struct SymbolMetrics {
    pub returns: Vec<f64>,
    pub sharpe: f64,
    pub win_rate: f64,
    pub avg_win_pct: f64,
    pub avg_loss_pct: f64,
}

#[derive(Debug, Clone)]
pub struct PortfolioOptimizer {
    pub lookback_days: usize,
    pub max_weight: f64,
    pub kelly_fraction: f64,
}

impl Default for PortfolioOptimizer {
    fn default() -> Self {
        Self {
            lookback_days: 30,
            max_weight: 0.35,
            kelly_fraction: 0.25,
        }
    }
}

impl PortfolioOptimizer {
    pub fn new(lookback_days: usize, max_weight: f64, kelly_fraction: f64) -> Self {
        Self {
            lookback_days,
            max_weight,
            kelly_fraction,
        }
    }

    pub fn get_optimal_allocation(
        &self,
        symbols: &[String],
        metrics_map: &HashMap<String, SymbolMetrics>,
    ) -> HashMap<String, AllocationResult> {
        if symbols.is_empty() {
            return HashMap::new();
        }

        let mut rets = Vec::new();
        let mut names = Vec::new();
        for s in symbols {
            if let Some(metrics) = metrics_map.get(s) {
                let returns = bounded_lookback_returns(&metrics.returns, self.lookback_days);
                if returns.len() > 5 {
                    rets.push(returns);
                    names.push(s.clone());
                }
            }
        }

        let mut weights = HashMap::new();
        let min_corr_samples = if self.lookback_days == 0 {
            30
        } else {
            self.lookback_days.clamp(6, 30)
        };

        if rets.len() >= 2 {
            let min_len = rets.iter().map(|r| r.len()).min().unwrap_or(0);
            if min_len < min_corr_samples {
                let w = 1.0 / names.len().max(1) as f64;
                for s in names {
                    weights.insert(
                        s.clone(),
                        AllocationResult {
                            symbol: s,
                            weight: w,
                            kelly_size: 0.0,
                            risk_budget: w,
                            correlation_score: 0.0,
                            sharpe: 0.0,
                        },
                    );
                }
                return weights;
            }

            for r in &mut rets {
                if r.len() > min_len {
                    *r = r[(r.len() - min_len)..].to_vec();
                }
            }

            let n_assets = names.len();
            let mut means = vec![0.0; n_assets];
            let mut vols = vec![0.0; n_assets];
            for (i, r) in rets.iter().enumerate() {
                let mean = mean(r);
                let std = stddev(r, mean);
                means[i] = mean;
                vols[i] = std.max(1e-6);
            }

            // Avoid building a full NxN correlation matrix when we only need per-asset averages.
            let mut corr_sums = vec![0.0_f64; n_assets];
            for i in 0..n_assets {
                for j in (i + 1)..n_assets {
                    let c = cov(&rets[i], means[i], &rets[j], means[j]);
                    let v = c / (vols[i] * vols[j]).max(1e-9);
                    corr_sums[i] += v;
                    corr_sums[j] += v;
                }
            }

            let mut sharpe_map = HashMap::new();
            let mut win_map = HashMap::new();
            for s in &names {
                if let Some(m) = metrics_map.get(s) {
                    sharpe_map.insert(s.clone(), m.sharpe);
                    win_map.insert(s.clone(), m.win_rate);
                }
            }

            let mut avg_corr = vec![0.0; n_assets];
            for i in 0..n_assets {
                avg_corr[i] = corr_sums[i] / (n_assets.saturating_sub(1).max(1) as f64);
            }

            let mut raw = vec![0.0; n_assets];
            for i in 0..n_assets {
                let s = &names[i];
                // The sharpe_map is built from metrics_map.get() above,
                // so a name without metrics gets silently skipped. The
                // previous `.expect("…always resolve…")` would panic
                // when that happened. We treat missing metrics as a
                // zero-sharpe allocation (effectively excluded from the
                // ranking) which is what the downstream `.max(0.0)`
                // already enforces.
                let sharpe = sharpe_map.get(s).copied().unwrap_or_else(|| {
                    tracing::warn!(
                        target: "forex_search::portfolio",
                        strategy = %s,
                        "ranked allocation name has no sharpe metric; \
                         falling back to zero-weight allocation"
                    );
                    0.0
                });
                let div_score = if avg_corr[i] >= 0.0 {
                    1.0 / (1.0 + avg_corr[i])
                } else {
                    1.0 + (-avg_corr[i]).min(1.0) * 0.5
                };
                raw[i] = (1.0 / vols[i]) * sharpe.max(0.0) * div_score;
            }

            if raw.iter().all(|v| *v <= 0.0) {
                raw = vec![1.0; n_assets];
            }
            let sum_raw: f64 = raw.iter().sum();
            let mut norm: Vec<f64> = raw.iter().map(|v| v / sum_raw).collect();

            for w in &mut norm {
                *w = w.clamp(0.0, self.max_weight);
            }
            let sum_capped: f64 = norm.iter().sum();
            let norm = if sum_capped > 0.0 {
                norm.iter().map(|v| v / sum_capped).collect::<Vec<f64>>()
            } else {
                vec![1.0 / n_assets as f64; n_assets]
            };

            for (i, s) in names.iter().enumerate() {
                let win_rate = *win_map
                    .get(s)
                    .expect("ranked allocation names should always resolve to win-rate metrics");
                // Proper Kelly criterion: f* = p - (1-p)/b where b = avg_win / avg_loss
                let metrics = metrics_map
                    .get(s)
                    .expect("allocation names should always resolve to source metrics");
                let avg_win = metrics.avg_win_pct.max(0.0);
                let avg_loss_mag = metrics.avg_loss_pct.abs().max(1e-9);
                let b = avg_win / avg_loss_mag;
                let kelly_raw = win_rate - (1.0 - win_rate) / b.max(1e-9);
                let kelly_val = self.kelly_fraction * kelly_raw.clamp(0.0, 0.5);
                weights.insert(
                    s.clone(),
                    AllocationResult {
                        symbol: s.clone(),
                        weight: norm[i],
                        kelly_size: kelly_val,
                        risk_budget: norm[i],
                        correlation_score: avg_corr[i],
                        sharpe: metrics.sharpe,
                    },
                );
            }
        } else {
            let w = 1.0 / symbols.len().max(1) as f64;
            for s in symbols {
                let sharpe = metrics_map
                    .get(s)
                    .map(|m| m.sharpe)
                    .filter(|value| value.is_finite())
                    .unwrap_or_default();
                weights.insert(
                    s.clone(),
                    AllocationResult {
                        symbol: s.clone(),
                        weight: w,
                        kelly_size: 0.0,
                        risk_budget: w,
                        correlation_score: 0.0,
                        sharpe,
                    },
                );
            }
        }

        weights
    }
}

use forex_core::utils::{mean, stddev_sample};

// Local `stddev` wrapper retains the previous one-arg call shape used
// throughout this module while routing the math through the canonical
// sample-stddev helper in `forex-core::utils::stats` (Phase 64).
fn stddev(values: &[f64], mean: f64) -> f64 {
    stddev_sample(values, mean)
}

fn bounded_lookback_returns(returns: &[f64], lookback_days: usize) -> Vec<f64> {
    let finite_returns = returns
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if lookback_days == 0 || finite_returns.len() <= lookback_days {
        return finite_returns;
    }
    finite_returns[(finite_returns.len() - lookback_days)..].to_vec()
}

fn cov(a: &[f64], mean_a: f64, b: &[f64], mean_b: f64) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 0.0;
    }
    let n = a.len().min(b.len());
    let mut sum = 0.0;
    for i in 0..n {
        sum += (a[i] - mean_a) * (b[i] - mean_b);
    }
    sum / (n as f64 - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(returns: Vec<f64>, sharpe: f64, win_rate: f64) -> SymbolMetrics {
        SymbolMetrics {
            returns,
            sharpe,
            win_rate,
            avg_win_pct: 0.0,
            avg_loss_pct: 0.0,
        }
    }

    #[test]
    fn lookback_days_limits_allocation_to_recent_returns() {
        let optimizer = PortfolioOptimizer::new(6, 0.95, 0.25);
        let symbols = vec!["EURUSD".to_string(), "GBPUSD".to_string()];
        let mut eur_returns = (0..60)
            .map(|i| if i % 2 == 0 { 0.10 } else { -0.10 })
            .collect::<Vec<_>>();
        eur_returns.extend([0.001; 6]);
        let mut gbp_returns = vec![0.001; 60];
        gbp_returns.extend([0.10, -0.10, 0.10, -0.10, 0.10, -0.10]);

        let mut map = HashMap::new();
        map.insert("EURUSD".to_string(), metrics(eur_returns, 1.0, 0.55));
        map.insert("GBPUSD".to_string(), metrics(gbp_returns, 1.0, 0.55));

        let alloc = optimizer.get_optimal_allocation(&symbols, &map);
        let eur_weight = alloc.get("EURUSD").expect("missing EURUSD").weight;
        let gbp_weight = alloc.get("GBPUSD").expect("missing GBPUSD").weight;
        assert!(
            eur_weight > gbp_weight,
            "recent low-volatility EURUSD tail should receive higher weight"
        );
    }

    #[test]
    fn equal_weight_fallback_when_not_enough_corr_samples() {
        let optimizer = PortfolioOptimizer::default();
        let symbols = vec!["EURUSD".to_string(), "GBPUSD".to_string()];
        let mut map = HashMap::new();
        map.insert("EURUSD".to_string(), metrics(vec![0.01; 20], 1.2, 0.55));
        map.insert("GBPUSD".to_string(), metrics(vec![0.02; 20], 1.0, 0.52));

        let alloc = optimizer.get_optimal_allocation(&symbols, &map);
        assert_eq!(alloc.len(), 2);
        for symbol in symbols {
            let w = alloc.get(&symbol).expect("missing allocation").weight;
            assert!((w - 0.5).abs() < 1e-12);
        }
    }

    #[test]
    fn correlation_score_is_stable_for_identical_series() {
        let optimizer = PortfolioOptimizer::new(30, 0.8, 0.25);
        let symbols = vec!["EURUSD".to_string(), "GBPUSD".to_string()];
        let rets: Vec<f64> = (0..64).map(|i| (i as f64 * 0.0001).sin()).collect();
        let mut map = HashMap::new();
        map.insert("EURUSD".to_string(), metrics(rets.clone(), 1.5, 0.56));
        map.insert("GBPUSD".to_string(), metrics(rets, 1.3, 0.54));

        let alloc = optimizer.get_optimal_allocation(&symbols, &map);
        assert_eq!(alloc.len(), 2);
        let eur_corr = alloc
            .get("EURUSD")
            .expect("missing EURUSD allocation")
            .correlation_score;
        let gbp_corr = alloc
            .get("GBPUSD")
            .expect("missing GBPUSD allocation")
            .correlation_score;
        assert!((eur_corr - 1.0).abs() < 1e-6);
        assert!((gbp_corr - 1.0).abs() < 1e-6);
    }
}
