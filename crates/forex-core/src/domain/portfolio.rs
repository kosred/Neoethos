use anyhow::Result;
use ndarray::{Array1, Array2};
use nalgebra::{DMatrix, DVector};
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
            max_exposure,
            correlation_threshold,
            strategy_weights: HashMap::new(),
        }
    }

    /// Calculate Pearson correlation matrix between multiple strategy returns
    pub fn calculate_correlation_matrix(&self, returns: &Array2<f64>) -> Result<Array2<f64>> {
        let n_strats = returns.ncols();
        let n_obs = returns.nrows() as f64;
        
        let mut cov_matrix = Array2::<f64>::zeros((n_strats, n_strats));
        let mut std_devs = Array1::<f64>::zeros(n_strats);

        for j in 0..n_strats {
            let col = returns.column(j);
            let mean = col.sum() / n_obs;
            let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_obs - 1.0);
            std_devs[j] = variance.sqrt();
        }

        for i in 0..n_strats {
            for j in 0..n_strats {
                let col_i = returns.column(i);
                let col_j = returns.column(j);
                let mean_i = col_i.sum() / n_obs;
                let mean_j = col_j.sum() / n_obs;
                
                let covariance: f64 = col_i.iter().zip(col_j.iter())
                    .map(|(&x, &y)| (x - mean_i) * (y - mean_j))
                    .sum::<f64>() / (n_obs - 1.0);
                
                let corr = if std_devs[i] > 0.0 && std_devs[j] > 0.0 {
                    covariance / (std_devs[i] * std_devs[j])
                } else {
                    0.0
                };
                cov_matrix[[i, j]] = corr;
            }
        }
        
        Ok(cov_matrix)
    }

    /// Optimize strategy allocations based on Markowitz mean-variance optimization 
    /// (Simplified proportional allocation inversely weighted by volatility and correlation)
    pub fn optimize_weights(&mut self, strategy_names: &[String], returns: &Array2<f64>) -> Result<()> {
        let n_strats = returns.ncols();
        let n_obs = returns.nrows() as f64;
        
        let mut std_devs = vec![0.0; n_strats];
        for j in 0..n_strats {
            let col = returns.column(j);
            let mean = col.sum() / n_obs;
            let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_obs - 1.0);
            std_devs[j] = variance.sqrt();
        }

        // Inverse volatility weighting (risk parity)
        let mut total_inv_vol = 0.0;
        let mut inv_vols = vec![0.0; n_strats];
        for j in 0..n_strats {
            if std_devs[j] > 0.0 {
                inv_vols[j] = 1.0 / std_devs[j];
                total_inv_vol += inv_vols[j];
            }
        }

        let mut weights = vec![0.0; n_strats];
        if total_inv_vol > 0.0 {
            for j in 0..n_strats {
                weights[j] = (inv_vols[j] / total_inv_vol) * self.max_exposure;
            }
        }

        self.strategy_weights.clear();
        for (i, name) in strategy_names.iter().enumerate() {
            self.strategy_weights.insert(name.clone(), weights[i]);
        }

        Ok(())
    }

    pub fn get_weight(&self, strategy_name: &str) -> f64 {
        *self.strategy_weights.get(strategy_name).unwrap_or(&0.0)
    }
}
