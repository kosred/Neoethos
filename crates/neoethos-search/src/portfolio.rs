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
                        target: "neoethos_search::portfolio",
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
            // Audit D08 (2026-07-13): the previous code clamped each weight
            // to `max_weight` and THEN renormalized (÷ sum) — which scales
            // the capped weights back UP above the cap (e.g. [0.6,0.3,0.1],
            // cap 0.35 → clamp [0.35,0.3,0.1] → renorm [0.467,0.4,0.133],
            // both cap-violated). Project onto the capped simplex instead:
            // weights are nonnegative, sum to 1, and none exceeds the cap.
            let sum_raw: f64 = raw.iter().sum();
            let base: Vec<f64> = if sum_raw > 0.0 {
                raw.iter().map(|v| v.max(0.0) / sum_raw).collect()
            } else {
                vec![1.0 / n_assets as f64; n_assets]
            };
            let norm = project_to_capped_simplex(&base, self.max_weight);

            for (i, s) in names.iter().enumerate() {
                // F-053 fix (2026-05-25 — task #218 unwrap audit):
                // the previous code `.expect()`-panicked on any name
                // without win-rate / metrics — the panic was a runtime
                // crash in the portfolio-allocation path. Following
                // the same pattern as `sharpe_map.get(s).copied()
                // .unwrap_or_else(...)` above (lines 147-155), we now
                // fall back to a zero-weight / zero-Kelly allocation
                // and emit a structured warning. The downstream
                // `.max(0.0)` already filters zero-Kelly allocations.
                let win_rate = win_map.get(s).copied().unwrap_or_else(|| {
                    tracing::warn!(
                        target: "neoethos_search::portfolio",
                        strategy = %s,
                        "ranked allocation name has no win-rate metric; \
                         falling back to zero-weight allocation"
                    );
                    0.0
                });
                let metrics = match metrics_map.get(s) {
                    Some(m) => m,
                    None => {
                        tracing::warn!(
                            target: "neoethos_search::portfolio",
                            strategy = %s,
                            "ranked allocation name has no source metrics; \
                             falling back to zero-weight allocation"
                        );
                        weights.insert(
                            s.clone(),
                            AllocationResult {
                                symbol: s.clone(),
                                weight: 0.0,
                                kelly_size: 0.0,
                                risk_budget: 0.0,
                                correlation_score: avg_corr[i],
                                sharpe: 0.0,
                            },
                        );
                        continue;
                    }
                };
                // Proper Kelly criterion: f* = p - (1-p)/b where b = avg_win / avg_loss
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
                // DOCUMENTED-DEFAULT: this branch is the equal-weight
                // fallback used when no per-symbol Sharpe is known.
                // Sharpe is descriptive only here (allocation weights
                // are `w`, not derived from sharpe), so 0.0 / missing
                // is a safe display fallback.
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

use neoethos_core::utils::{mean, stddev_sample};

// Local `stddev` wrapper retains the previous one-arg call shape used
// throughout this module while routing the math through the canonical
// sample-stddev helper in `neoethos-core::utils::stats` (Phase 64).
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

/// Project nonnegative `weights` onto the capped simplex: the returned
/// weights are nonnegative, sum to 1 (when the cap allows it), and none
/// exceeds `cap`. Uses water-filling — cap the over-cap weights and
/// redistribute their excess mass proportionally to the uncapped ones,
/// repeating until none exceed the cap (audit D08).
///
/// Infeasible cap (`n * cap < 1`, i.e. too tight for this many assets to
/// still sum to 1): no allocation can both sum to 1 and honor the cap, so
/// fall back to equal weights `1/n` (the least-concentrated choice) rather
/// than silently exceeding the cap on a subset.
fn project_to_capped_simplex(weights: &[f64], cap: f64) -> Vec<f64> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }
    let cap = cap.max(0.0);
    if cap <= 0.0 || (n as f64) * cap < 1.0 - 1e-12 {
        return vec![1.0 / n as f64; n];
    }
    // Normalize the nonnegative input to sum 1 as the starting point.
    let mut w: Vec<f64> = {
        let s: f64 = weights.iter().map(|v| v.max(0.0)).sum();
        if s > 0.0 {
            weights.iter().map(|v| v.max(0.0) / s).collect()
        } else {
            vec![1.0 / n as f64; n]
        }
    };
    // Each pass freezes at least one more weight at the cap, so this
    // terminates in <= n passes; the guard bounds it regardless. A weight
    // AT the cap is frozen (not eligible for more mass) — only weights
    // STRICTLY below the cap absorb redistributed excess, otherwise a
    // just-capped weight would be pushed back over.
    for _ in 0..=n {
        let mut excess = 0.0;
        let mut free_sum = 0.0;
        for &wi in &w {
            if wi >= cap - 1e-15 {
                excess += (wi - cap).max(0.0);
            } else {
                free_sum += wi;
            }
        }
        if excess <= 1e-15 {
            break;
        }
        if free_sum <= 1e-15 {
            // No room left to redistribute: clamp everything to the cap.
            for wi in &mut w {
                *wi = wi.min(cap);
            }
            break;
        }
        for wi in &mut w {
            if *wi >= cap - 1e-15 {
                *wi = cap;
            } else {
                *wi += excess * (*wi / free_sum);
            }
        }
    }
    w
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

    #[test]
    fn capped_simplex_respects_cap_sums_to_one_and_is_nonneg() {
        // Audit D08: the exact case the old clamp+renormalize violated.
        let cap = 0.35;
        let out = project_to_capped_simplex(&[0.6, 0.3, 0.1], cap);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "must sum to 1, got {sum}");
        for w in &out {
            assert!(*w >= -1e-12, "weights must be nonnegative: {w}");
            assert!(*w <= cap + 1e-9, "weight {w} exceeds cap {cap}");
        }
        // Mass conserved: the two under-cap assets absorb the excess.
        assert!(out[0] <= cap + 1e-9 && out[0] >= cap - 1e-9, "biggest hits cap");
    }

    #[test]
    fn capped_simplex_cascades_when_redistribution_re_violates() {
        // FEASIBLE (n*cap = 4*0.30 = 1.2 >= 1) but redistributing the big
        // asset's excess pushes a medium one over the cap, and again — the
        // water-fill must iterate (3 passes here) until all obey the cap.
        let cap = 0.30;
        let out = project_to_capped_simplex(&[0.5, 0.29, 0.19, 0.02], cap);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum {sum}");
        for w in &out {
            assert!(*w <= cap + 1e-9, "weight {w} exceeds cap {cap}");
            assert!(*w >= -1e-12, "weight {w} negative");
        }
    }

    #[test]
    fn capped_simplex_infeasible_cap_falls_back_to_equal_weights() {
        // n*cap < 1 (3 * 0.2 = 0.6 < 1): impossible to sum to 1 under the
        // cap — fall back to equal weights rather than silently violating.
        let out = project_to_capped_simplex(&[0.5, 0.4, 0.1], 0.2);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
        for w in &out {
            assert!((*w - 1.0 / 3.0).abs() < 1e-9, "expected equal weights, got {w}");
        }
    }
}
