use anyhow::{Result, bail};
use ndarray::Array2;
use std::collections::HashMap;

/// Portfolio Strategy Manager and Correlation Risk Math.
pub struct PortfolioManager {
    max_exposure: f64,
    correlation_threshold: f64,
    strategy_weights: HashMap<String, f64>,
}

impl Default for PortfolioManager {
    fn default() -> Self {
        Self {
            max_exposure: 1.0,
            correlation_threshold: 0.7,
            strategy_weights: HashMap::new(),
        }
    }
}

impl PortfolioManager {
    pub fn new(max_exposure: f64, correlation_threshold: f64) -> Self {
        Self {
            max_exposure: max_exposure.max(0.0),
            correlation_threshold: correlation_threshold.clamp(0.0, 1.0),
            strategy_weights: HashMap::new(),
        }
    }

    /// Calculate Pearson correlation matrix between multiple strategy returns
    pub fn calculate_correlation_matrix(&self, returns: &Array2<f64>) -> Result<Array2<f64>> {
        let n_strats = returns.ncols();
        let n_obs = returns.nrows();
        if n_strats == 0 {
            return Ok(Array2::<f64>::zeros((0, 0)));
        }
        if n_obs < 2 {
            bail!("at least two return observations are required for correlation");
        }

        let mut cov_matrix = Array2::<f64>::zeros((n_strats, n_strats));
        let mut means = vec![0.0; n_strats];
        let mut std_devs = vec![0.0; n_strats];

        for j in 0..n_strats {
            let col = returns.column(j);
            let mean = col.sum() / n_obs as f64;
            means[j] = mean;
            let variance =
                col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_obs as f64 - 1.0);
            std_devs[j] = variance.max(0.0).sqrt();
        }

        for i in 0..n_strats {
            cov_matrix[[i, i]] = 1.0;
            for j in (i + 1)..n_strats {
                let col_i = returns.column(i);
                let col_j = returns.column(j);
                let covariance: f64 = col_i
                    .iter()
                    .zip(col_j.iter())
                    .map(|(&x, &y)| (x - means[i]) * (y - means[j]))
                    .sum::<f64>()
                    / (n_obs as f64 - 1.0);

                let corr = if std_devs[i] > 0.0 && std_devs[j] > 0.0 {
                    covariance / (std_devs[i] * std_devs[j])
                } else {
                    0.0
                }
                .clamp(-1.0, 1.0);
                cov_matrix[[i, j]] = corr;
                cov_matrix[[j, i]] = corr;
            }
        }

        Ok(cov_matrix)
    }

    /// Optimize allocations with inverse volatility, positive return bias, and correlation penalty.
    pub fn optimize_weights(
        &mut self,
        strategy_names: &[String],
        returns: &Array2<f64>,
    ) -> Result<()> {
        let n_strats = returns.ncols();
        let n_obs = returns.nrows();
        if strategy_names.len() != n_strats {
            bail!(
                "strategy name count ({}) must match returns columns ({})",
                strategy_names.len(),
                n_strats
            );
        }
        if n_strats == 0 {
            self.strategy_weights.clear();
            return Ok(());
        }
        if n_obs < 2 {
            bail!("at least two return observations are required for portfolio optimization");
        }

        let correlations = self.calculate_correlation_matrix(returns)?;
        let mut means = vec![0.0; n_strats];
        let mut std_devs = vec![0.0; n_strats];
        for j in 0..n_strats {
            let col = returns.column(j);
            let mean = col.sum() / n_obs as f64;
            means[j] = mean;
            let variance =
                col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_obs as f64 - 1.0);
            std_devs[j] = variance.max(0.0).sqrt().max(1e-9);
        }

        let has_positive_edge = means.iter().any(|mean| *mean > 0.0);
        let mut raw_scores = vec![0.0; n_strats];
        for j in 0..n_strats {
            let edge_score = if has_positive_edge {
                means[j].max(0.0)
            } else {
                1.0
            };
            let mut excess_corr_sum = 0.0;
            for k in 0..n_strats {
                if j == k {
                    continue;
                }
                let excess = correlations[[j, k]].abs() - self.correlation_threshold;
                excess_corr_sum += excess.max(0.0);
            }
            let avg_excess_corr = excess_corr_sum / n_strats.saturating_sub(1).max(1) as f64;
            let diversification_penalty = 1.0 / (1.0 + avg_excess_corr);
            raw_scores[j] = (edge_score / std_devs[j]) * diversification_penalty;
        }

        if raw_scores
            .iter()
            .all(|score| *score <= 0.0 || !score.is_finite())
        {
            raw_scores.fill(1.0);
        }

        let total_score: f64 = raw_scores
            .iter()
            .copied()
            .filter(|score| score.is_finite() && *score > 0.0)
            .sum();
        self.strategy_weights.clear();
        for (i, name) in strategy_names.iter().enumerate() {
            let score = raw_scores[i];
            let weight = if total_score > 0.0 && score.is_finite() && score > 0.0 {
                (score / total_score) * self.max_exposure
            } else {
                0.0
            };
            self.strategy_weights.insert(name.clone(), weight);
        }

        Ok(())
    }

    pub fn get_weight(&self, strategy_name: &str) -> f64 {
        // FIXME(silent-fallback): caller plumbing required (F-CORE2-003)
        // Public API returns f64, so a missing strategy quietly produces 0.0.
        // Callers placing real orders should use `try_get_weight` once added,
        // or check `has_strategy(name)` before relying on this value.
        match self.strategy_weights.get(strategy_name) {
            Some(weight) => *weight,
            None => {
                tracing::warn!(
                    target: "forex_core::portfolio",
                    strategy = strategy_name,
                    "get_weight: unknown strategy, returning 0.0 (F-CORE2-003)"
                );
                0.0
            }
        }
    }

    /// Returns `Some(weight)` if the strategy is known, otherwise `None`.
    /// Prefer this in safety-critical paths (sizing, risk gate).
    pub fn try_get_weight(&self, strategy_name: &str) -> Option<f64> {
        self.strategy_weights.get(strategy_name).copied()
    }

    /// Quick membership check used by callers that need to validate a name
    /// before sizing a position.
    pub fn has_strategy(&self, strategy_name: &str) -> bool {
        self.strategy_weights.contains_key(strategy_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn optimizer_penalizes_highly_correlated_strategy_cluster() {
        let returns = array![
            [0.01, 0.010, 0.03],
            [0.02, 0.021, -0.01],
            [0.03, 0.031, 0.02],
            [0.04, 0.041, -0.02],
            [0.05, 0.051, 0.01],
        ];
        let names = vec![
            "trend_a".to_string(),
            "trend_b".to_string(),
            "mean_rev".to_string(),
        ];
        let mut manager = PortfolioManager::new(1.0, 0.5);

        manager
            .optimize_weights(&names, &returns)
            .expect("portfolio optimization should succeed");

        let trend_a = manager.get_weight("trend_a");
        let trend_b = manager.get_weight("trend_b");
        let mean_rev = manager.get_weight("mean_rev");
        assert!(trend_a > 0.0);
        assert!(trend_b > 0.0);
        assert!(mean_rev > 0.0);
        assert!(
            (trend_a + trend_b + mean_rev - 1.0).abs() < 1e-9,
            "weights should use the configured exposure budget"
        );
        assert!(
            trend_a + trend_b < 0.95,
            "correlated strategies should not consume the entire budget"
        );
    }
}
