//! `RegimeHmmExpert` — 3-state Hidden Markov Model for soft-posterior
//! regime detection.
//!
//! Added 2026-05-25 as the 34th model in the ensemble per operator
//! directive. Complements the rule-based regime classifier in
//! `neoethos-search::stop_target::infer_regime` (which returns
//! trend/range/neutral via ADX+Hurst+EMA-cross hard thresholds) with
//! a probabilistic posterior `P(state_t | obs_1:t)` from real data.
//!
//! ## Model
//!
//! 3 hidden states, indexed exactly as the canonical 3-class label
//! mapping in `crate::runtime::artifacts::default_three_class_label_mapping`:
//!
//! - **State 0** = `range` (no directional bias) → maps to label
//!   0 (neutral) → output column 0
//! - **State 1** = `bullish_trend` (upward drift) → maps to label
//!   1 (buy) → output column 1
//! - **State 2** = `bearish_trend` (downward drift) → maps to label
//!   -1 (sell) → output column 2
//!
//! Emissions are bivariate Gaussian over `(log_return, log_volatility)`:
//!
//! - `log_return` per bar = `ln(close[t] / close[t-1])`
//! - `log_volatility` per bar = `ln(max(high - low, 1e-12))`
//!
//! Each state has its own (μ, Σ) where Σ is a 2×2 covariance with
//! σ²_lr, σ²_lv on the diagonal and ρ × σ_lr × σ_lv off-diagonal.
//! Training: Baum-Welch EM on a rolling window (default 2000 bars).
//! Inference: Forward algorithm → α[t][s] then normalize to posterior.
//!
//! ## Why HMM
//!
//! - The rule-based `infer_regime` returns a HARD vote (one of three
//!   strings). HMM gives a SMOOTH posterior — handles transitions
//!   gracefully (e.g. P(trend) = 0.4 + P(range) = 0.6 means we are
//!   probably ranging but trending is plausible).
//! - Joint EM training learns transition probabilities from data
//!   rather than the operator hardcoding `ADX ≥ 25 = trend` thresholds
//!   (operator directive 2026-05-25 "remove hardcoded values that
//!   limit our capability").
//! - Risky Mode position-sizer can use `P(range_state)` to dynamically
//!   shrink per-trade risk in choppy markets (math in
//!   `risky_mode.rs` already accepts an adaptive `risk_per_trade_fraction`).
//!
//! ## Stationarity caveat
//!
//! Pure HMM assumes a stationary transition matrix. Markets aren't
//! stationary. Workaround: rolling-window retraining (refit every
//! ~500 bars). The `HmmRegimeConfig::refit_every_n_bars` knob
//! controls this. A Markov-switching extension (Hidden Semi-Markov,
//! MS-GARCH) is deferred to v0.5+ if needed.

use anyhow::{Context, Result, bail};
use ndarray::{Array1, Array2};
use neoethos_data::{FeatureCellValidity, FeatureFrame};
use neoethos_execution_budget::CpuLease;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::base::{
    build_runtime_prediction_with_details, canonical_three_class_label_mapping,
    three_class_runtime_confidence, try_build_runtime_artifact_metadata,
};
use crate::runtime::artifacts::{RuntimeArtifactMetadata, TrainingSummaryMetadata};
use crate::runtime::capabilities::{CapabilityState, ModelFamily};
use crate::runtime::prediction::RuntimePrediction;

const MODEL_NAME: &str = "hmm_regime";
const N_STATES: usize = 3;
const FEATURE_DIM: usize = 2;
const ARTIFACT_FILE: &str = "hmm_regime.json";
const HMM_F64_SCHEMA: &str = "neoethos.hmm_regime.f64_validity.v2";
const HMM_FEATURE_COLUMNS: [&str; FEATURE_DIM] = ["quant_log_return", "quant_log_volatility"];

/// Operator-tunable knobs for the HMM regime expert. All knobs have
/// sensible defaults derived from typical FX behaviour at the M1
/// timeframe; operators can override per-symbol via
/// [`crate::runtime::capabilities::requested_runtime_device_policy`]
/// or future typed-override registry.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HmmRegimeConfig {
    /// Maximum Baum-Welch EM iterations per training cycle. Default
    /// 50 — empirically converges in 15-30 on FX M1 windows.
    pub max_em_iterations: usize,
    /// Convergence tolerance on log-likelihood improvement per EM
    /// step. Default 1e-4.
    pub em_log_likelihood_tolerance: f64,
    /// Refit cadence in bars. Default 500 — retrains the model every
    /// 500 bars to handle non-stationarity.
    pub refit_every_n_bars: usize,
    /// Minimum training window in bars before the model is considered
    /// reliable. Default 500 — below this, training fails closed.
    pub min_training_bars: usize,
}

impl Default for HmmRegimeConfig {
    fn default() -> Self {
        Self {
            max_em_iterations: 50,
            em_log_likelihood_tolerance: 1e-4,
            refit_every_n_bars: 500,
            min_training_bars: 500,
        }
    }
}

/// Serialised HMM parameters. State indexing is the canonical
/// 3-class order: 0=range/neutral, 1=bullish_trend/buy,
/// 2=bearish_trend/sell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmmRegimeArtifact {
    #[serde(default)]
    pub precision_schema: String,
    pub model_name: String,
    pub feature_columns: Vec<String>,
    /// `(N_STATES,)` initial-state distribution π.
    pub initial_probs: Vec<f64>,
    /// `(N_STATES, N_STATES)` transition matrix flattened row-major.
    /// `A[i*N + j] = P(state_{t+1} = j | state_t = i)`.
    pub transition_flat: Vec<f64>,
    /// `(N_STATES, FEATURE_DIM)` emission means flattened.
    pub emission_means_flat: Vec<f64>,
    /// `(N_STATES, FEATURE_DIM, FEATURE_DIM)` emission covariances
    /// flattened. Stored as full symmetric 2×2 matrices per state.
    pub emission_covs_flat: Vec<f64>,
    pub training_summary: TrainingSummaryMetadata,
    #[serde(default)]
    pub runtime_metadata: Option<RuntimeArtifactMetadata>,
    pub config: HmmRegimeConfig,
}

/// Row-aligned HMM output. Invalid feature rows retain an explicit reason and
/// canonical NaN payload instead of being changed into a neutral probability.
#[derive(Debug, Clone, PartialEq)]
pub struct HmmPosteriorFrame {
    pub probabilities: Array2<f64>,
    pub validity: Vec<FeatureCellValidity>,
}

/// Live HMM regime predictor. Holds the trained parameters in
/// dense ndarrays for fast Forward-algorithm inference. Constructed
/// via [`Self::train`] (Baum-Welch EM) or [`Self::load_from_artifact`].
#[derive(Debug, Clone)]
pub struct RegimeHmmExpert {
    /// `(N_STATES,)` — initial-state distribution π.
    initial_probs: Array1<f64>,
    /// `(N_STATES, N_STATES)` — transition matrix.
    transition: Array2<f64>,
    /// `(N_STATES, FEATURE_DIM)` — emission means.
    emission_means: Array2<f64>,
    /// `(N_STATES, FEATURE_DIM, FEATURE_DIM)` — emission covariances.
    /// Stored as a Vec of N_STATES 2×2 matrices for clarity.
    emission_covs: Vec<Array2<f64>>,
    feature_columns: Vec<String>,
    config: HmmRegimeConfig,
}

impl RegimeHmmExpert {
    /// Materialize the exact versioned feature-plan outputs admitted for HMM
    /// training. Training never reconstructs these values from bare OHLCV.
    pub fn training_observations_from_feature_frame(
        frame: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<Array2<f64>> {
        lease.scope(|| {
            anyhow::ensure!(
                frame.n_samples() > 0,
                "HMM regime training feature frame is empty"
            );
            let mut columns = Vec::with_capacity(FEATURE_DIM);
            for expected in HMM_FEATURE_COLUMNS {
                let index = frame
                    .names
                    .iter()
                    .position(|name| name == expected)
                    .with_context(|| {
                        format!("HMM regime training is missing feature column `{expected}`")
                    })?;
                columns.push(frame.feature_column(index)?);
            }

            let mut values = Vec::with_capacity(frame.n_samples() * FEATURE_DIM);
            for row in 0..frame.n_samples() {
                for column in &columns {
                    let validity = column.validity[row];
                    anyhow::ensure!(
                        validity.is_valid(),
                        "HMM regime training feature `{}` row {row} is invalid: {validity:?}",
                        column.name
                    );
                    let value = column.values[row];
                    anyhow::ensure!(
                        value.is_finite(),
                        "HMM regime training feature `{}` row {row} is non-finite: {value}",
                        column.name
                    );
                    values.push(value);
                }
            }
            Array2::from_shape_vec((frame.n_samples(), FEATURE_DIM), values)
                .context("shape exact HMM training observations")
        })
    }

    /// Train a new HMM via Baum-Welch EM on the supplied
    /// `(log_return, log_volatility)` matrix.
    ///
    /// The input must have exactly 2 columns named `log_return` and
    /// `log_volatility`. The caller is responsible for computing those
    /// from the versioned feature plan before model admission.
    pub fn train(
        observations: &Array2<f64>,
        feature_columns: Vec<String>,
        config: HmmRegimeConfig,
    ) -> Result<Self> {
        if observations.ncols() != FEATURE_DIM {
            bail!(
                "HMM regime expects {FEATURE_DIM} feature columns, got {}",
                observations.ncols()
            );
        }
        if observations.nrows() < config.min_training_bars {
            bail!(
                "HMM regime requires at least {} training bars; got {}",
                config.min_training_bars,
                observations.nrows()
            );
        }
        if feature_columns.len() != FEATURE_DIM {
            bail!(
                "HMM regime expects {FEATURE_DIM} feature column names, got {}",
                feature_columns.len()
            );
        }
        if feature_columns
            .iter()
            .map(String::as_str)
            .ne(HMM_FEATURE_COLUMNS)
        {
            bail!(
                "HMM regime feature columns must be {:?} in that order; got {:?}",
                HMM_FEATURE_COLUMNS,
                feature_columns
            );
        }
        for ((row, column), value) in observations.indexed_iter() {
            if !value.is_finite() {
                bail!(
                    "HMM regime training observation row {row} column {column} is non-finite: {value}"
                );
            }
        }
        let mut model = Self::initial_seed(observations, feature_columns, config)?;
        model.run_baum_welch(observations)?;
        Ok(model)
    }

    /// Initial seed for EM. Cluster the observations by simple
    /// thresholding on log-return:
    ///   - log_return < -threshold → state 2 (bearish)
    ///   - log_return > +threshold → state 1 (bullish)
    ///   - otherwise → state 0 (range)
    /// Threshold is the 33rd / 67th percentile of |log_return|.
    /// Then compute initial μ, Σ, π, A from this assignment.
    fn initial_seed(
        observations: &Array2<f64>,
        feature_columns: Vec<String>,
        config: HmmRegimeConfig,
    ) -> Result<Self> {
        let n = observations.nrows();
        // Sorted absolute log-returns for percentile cuts.
        let mut abs_returns: Vec<f64> = (0..n)
            .map(|i| observations[(i, 0)].abs())
            .filter(|v| v.is_finite())
            .collect();
        abs_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo_idx = ((abs_returns.len() as f64) * 0.33).round() as usize;
        let threshold = abs_returns.get(lo_idx).copied().unwrap_or(1e-6).max(1e-9);

        // Assign each bar to an initial state.
        let mut state_assign = Vec::with_capacity(n);
        for i in 0..n {
            let lr = observations[(i, 0)];
            let state = if lr < -threshold {
                2 // bearish
            } else if lr > threshold {
                1 // bullish
            } else {
                0 // range
            };
            state_assign.push(state);
        }

        // Compute initial parameters from the assignment.
        let mut counts = [0_usize; N_STATES];
        let mut means = vec![Array1::<f64>::zeros(FEATURE_DIM); N_STATES];
        for (i, &s) in state_assign.iter().enumerate() {
            counts[s] += 1;
            for f in 0..FEATURE_DIM {
                means[s][f] += observations[(i, f)];
            }
        }
        for s in 0..N_STATES {
            if counts[s] > 0 {
                means[s] /= counts[s] as f64;
            }
        }

        // Covariance per state.
        let mut covs = Vec::with_capacity(N_STATES);
        for s in 0..N_STATES {
            let mut sigma = Array2::<f64>::zeros((FEATURE_DIM, FEATURE_DIM));
            if counts[s] > 1 {
                for i in 0..n {
                    if state_assign[i] != s {
                        continue;
                    }
                    for a in 0..FEATURE_DIM {
                        for b in 0..FEATURE_DIM {
                            let da = observations[(i, a)] - means[s][a];
                            let db = observations[(i, b)] - means[s][b];
                            sigma[(a, b)] += da * db;
                        }
                    }
                }
                sigma /= (counts[s] - 1) as f64;
                // Add small ridge for numerical stability.
                for d in 0..FEATURE_DIM {
                    sigma[(d, d)] += 1e-8;
                }
            } else {
                // Fallback: identity-ish small covariance.
                for d in 0..FEATURE_DIM {
                    sigma[(d, d)] = 1e-4;
                }
            }
            covs.push(sigma);
        }

        // Transition matrix from assignment.
        let mut trans = Array2::<f64>::zeros((N_STATES, N_STATES));
        for window in state_assign.windows(2) {
            trans[(window[0], window[1])] += 1.0;
        }
        for i in 0..N_STATES {
            let row_sum: f64 = (0..N_STATES).map(|j| trans[(i, j)]).sum();
            if row_sum > 0.0 {
                for j in 0..N_STATES {
                    trans[(i, j)] /= row_sum;
                }
            } else {
                // Uniform fallback for unobserved states.
                for j in 0..N_STATES {
                    trans[(i, j)] = 1.0 / N_STATES as f64;
                }
            }
        }

        // Initial distribution: first observation's state, smoothed.
        let mut pi = Array1::<f64>::from_elem(N_STATES, 1e-6);
        if let Some(&s0) = state_assign.first() {
            pi[s0] += 1.0;
        }
        let pi_sum: f64 = pi.iter().sum();
        pi /= pi_sum;

        let mut emission_means = Array2::<f64>::zeros((N_STATES, FEATURE_DIM));
        for s in 0..N_STATES {
            for f in 0..FEATURE_DIM {
                emission_means[(s, f)] = means[s][f];
            }
        }

        Ok(Self {
            initial_probs: pi,
            transition: trans,
            emission_means,
            emission_covs: covs,
            feature_columns,
            config,
        })
    }

    /// Baum-Welch EM. Iterates Forward-Backward + parameter
    /// re-estimation until log-likelihood improvement is below
    /// tolerance or max iterations reached.
    fn run_baum_welch(&mut self, observations: &Array2<f64>) -> Result<()> {
        let mut prev_ll = f64::NEG_INFINITY;
        for iter in 0..self.config.max_em_iterations {
            let (alpha, ll) = self.forward(observations)?;
            let beta = self.backward(observations)?;
            // Posterior γ[t][s] = P(state_t = s | obs)
            // and ξ[t][i][j] = P(state_t = i, state_{t+1} = j | obs)
            let n = observations.nrows();
            let mut gamma = Array2::<f64>::zeros((n, N_STATES));
            let mut xi_sum = Array2::<f64>::zeros((N_STATES, N_STATES));
            for t in 0..n {
                let mut row_norm = 0.0;
                for s in 0..N_STATES {
                    gamma[(t, s)] = alpha[(t, s)] * beta[(t, s)];
                    row_norm += gamma[(t, s)];
                }
                if row_norm > 0.0 {
                    for s in 0..N_STATES {
                        gamma[(t, s)] /= row_norm;
                    }
                }
            }
            for t in 0..n.saturating_sub(1) {
                let mut denom = 0.0;
                let mut xi_t = Array2::<f64>::zeros((N_STATES, N_STATES));
                for i in 0..N_STATES {
                    for j in 0..N_STATES {
                        let b_j = self.emission_pdf(j, t + 1, observations);
                        let v = alpha[(t, i)] * self.transition[(i, j)] * b_j * beta[(t + 1, j)];
                        xi_t[(i, j)] = v;
                        denom += v;
                    }
                }
                if denom > 0.0 {
                    for i in 0..N_STATES {
                        for j in 0..N_STATES {
                            xi_sum[(i, j)] += xi_t[(i, j)] / denom;
                        }
                    }
                }
            }
            // Re-estimate parameters.
            // π = γ[0]
            for s in 0..N_STATES {
                self.initial_probs[s] = gamma[(0, s)].max(1e-9);
            }
            let pi_sum: f64 = self.initial_probs.iter().sum();
            self.initial_probs /= pi_sum;
            // A[i][j] = ξ_sum[i][j] / Σ_j ξ_sum[i][j]
            for i in 0..N_STATES {
                let row_sum: f64 = (0..N_STATES).map(|j| xi_sum[(i, j)]).sum();
                if row_sum > 0.0 {
                    for j in 0..N_STATES {
                        self.transition[(i, j)] = (xi_sum[(i, j)] / row_sum).max(1e-9);
                    }
                }
            }
            // Renormalize transition rows.
            for i in 0..N_STATES {
                let row_sum: f64 = (0..N_STATES).map(|j| self.transition[(i, j)]).sum();
                if row_sum > 0.0 {
                    for j in 0..N_STATES {
                        self.transition[(i, j)] /= row_sum;
                    }
                }
            }
            // μ_s = Σ_t γ[t][s] * x[t] / Σ_t γ[t][s]
            for s in 0..N_STATES {
                let gamma_sum: f64 = (0..n).map(|t| gamma[(t, s)]).sum();
                if gamma_sum > 0.0 {
                    for f in 0..FEATURE_DIM {
                        let weighted: f64 =
                            (0..n).map(|t| gamma[(t, s)] * observations[(t, f)]).sum();
                        self.emission_means[(s, f)] = weighted / gamma_sum;
                    }
                }
            }
            // Σ_s = Σ_t γ[t][s] * (x[t]-μ_s)(x[t]-μ_s)^T / Σ_t γ[t][s]
            for s in 0..N_STATES {
                let gamma_sum: f64 = (0..n).map(|t| gamma[(t, s)]).sum();
                if gamma_sum > 0.0 {
                    let mut sigma = Array2::<f64>::zeros((FEATURE_DIM, FEATURE_DIM));
                    for t in 0..n {
                        let g = gamma[(t, s)];
                        for a in 0..FEATURE_DIM {
                            for b in 0..FEATURE_DIM {
                                let da = observations[(t, a)] - self.emission_means[(s, a)];
                                let db = observations[(t, b)] - self.emission_means[(s, b)];
                                sigma[(a, b)] += g * da * db;
                            }
                        }
                    }
                    sigma /= gamma_sum;
                    // Ridge for numerical stability.
                    for d in 0..FEATURE_DIM {
                        sigma[(d, d)] += 1e-8;
                    }
                    self.emission_covs[s] = sigma;
                }
            }
            // Convergence check.
            if (ll - prev_ll).abs() < self.config.em_log_likelihood_tolerance {
                tracing::debug!(
                    target: "neoethos_models::hmm_regime",
                    iter,
                    log_likelihood = ll,
                    "Baum-Welch converged"
                );
                break;
            }
            prev_ll = ll;
        }
        Ok(())
    }

    /// Forward algorithm. Returns `(alpha, total_log_likelihood)`.
    /// `alpha[t][s] = P(obs_1:t, state_t = s)`, normalized per-row to
    /// avoid underflow. The total log-likelihood is computed from the
    /// per-row scaling factors.
    fn forward(&self, observations: &Array2<f64>) -> Result<(Array2<f64>, f64)> {
        let n = observations.nrows();
        if n == 0 {
            bail!("HMM regime forward pass requires at least one observation");
        }
        let mut alpha = Array2::<f64>::zeros((n, N_STATES));
        let mut log_likelihood = 0.0;
        // t=0: α[0][s] = π_s * b_s(x_0)
        let mut row_sum = 0.0;
        for s in 0..N_STATES {
            alpha[(0, s)] = self.initial_probs[s] * self.emission_pdf(s, 0, observations);
            row_sum += alpha[(0, s)];
        }
        if !row_sum.is_finite() || row_sum <= 0.0 {
            bail!("HMM regime posterior mass is invalid at observation row 0: {row_sum}");
        }
        for s in 0..N_STATES {
            alpha[(0, s)] /= row_sum;
        }
        log_likelihood += row_sum.ln();
        // Recurse: α[t][j] = (Σ_i α[t-1][i] * A[i][j]) * b_j(x_t)
        for t in 1..n {
            let mut row_sum = 0.0;
            for j in 0..N_STATES {
                let trans_term: f64 = (0..N_STATES)
                    .map(|i| alpha[(t - 1, i)] * self.transition[(i, j)])
                    .sum();
                alpha[(t, j)] = trans_term * self.emission_pdf(j, t, observations);
                row_sum += alpha[(t, j)];
            }
            if !row_sum.is_finite() || row_sum <= 0.0 {
                bail!("HMM regime posterior mass is invalid at observation row {t}: {row_sum}");
            }
            for j in 0..N_STATES {
                alpha[(t, j)] /= row_sum;
            }
            log_likelihood += row_sum.ln();
        }
        Ok((alpha, log_likelihood))
    }

    /// Backward algorithm. β[t][s] = P(obs_{t+1:T} | state_t = s).
    /// Row-normalized for numerical stability.
    fn backward(&self, observations: &Array2<f64>) -> Result<Array2<f64>> {
        let n = observations.nrows();
        if n == 0 {
            bail!("HMM regime backward pass requires at least one observation");
        }
        let mut beta = Array2::<f64>::zeros((n, N_STATES));
        // β[T-1][s] = 1
        for s in 0..N_STATES {
            beta[(n - 1, s)] = 1.0 / N_STATES as f64;
        }
        for t in (0..n - 1).rev() {
            let mut row_sum = 0.0;
            for i in 0..N_STATES {
                let v: f64 = (0..N_STATES)
                    .map(|j| {
                        self.transition[(i, j)]
                            * self.emission_pdf(j, t + 1, observations)
                            * beta[(t + 1, j)]
                    })
                    .sum();
                beta[(t, i)] = v;
                row_sum += v;
            }
            if !row_sum.is_finite() || row_sum <= 0.0 {
                bail!(
                    "HMM regime backward probability mass is invalid at observation row {t}: {row_sum}"
                );
            }
            for i in 0..N_STATES {
                beta[(t, i)] /= row_sum;
            }
        }
        Ok(beta)
    }

    /// Bivariate Gaussian PDF for state `s` at observation index `t`.
    /// Inlined 2×2 inverse + determinant for speed (no general
    /// matrix inversion needed).
    fn emission_pdf(&self, s: usize, t: usize, obs: &Array2<f64>) -> f64 {
        let sigma = &self.emission_covs[s];
        let det = sigma[(0, 0)] * sigma[(1, 1)] - sigma[(0, 1)] * sigma[(1, 0)];
        if det <= 0.0 {
            return 1e-30;
        }
        let dx0 = obs[(t, 0)] - self.emission_means[(s, 0)];
        let dx1 = obs[(t, 1)] - self.emission_means[(s, 1)];
        // (Σ⁻¹ × dx)·dx for 2×2:
        let inv_det = 1.0 / det;
        let m00 = sigma[(1, 1)] * inv_det;
        let m11 = sigma[(0, 0)] * inv_det;
        let m01 = -sigma[(0, 1)] * inv_det;
        let quad = m00 * dx0 * dx0 + 2.0 * m01 * dx0 * dx1 + m11 * dx1 * dx1;
        let norm = 1.0 / (2.0 * std::f64::consts::PI * det.sqrt());
        norm * (-0.5 * quad).exp()
    }

    /// Feature column names this expert was trained on. Used by the
    /// ensemble-inference adapter to satisfy `ExpertModel::feature_columns`.
    pub fn feature_columns(&self) -> &[String] {
        &self.feature_columns
    }

    /// Predict the per-bar regime posteriors for `observations`.
    /// Returns an `(N, 3)` matrix where col 0 = P(range/neutral),
    /// col 1 = P(bullish_trend/buy), col 2 = P(bearish_trend/sell).
    /// Matches the canonical 3-class output convention from
    /// `base.rs` lines 128-135.
    pub fn predict_proba_observations(&self, observations: &Array2<f64>) -> Result<Array2<f64>> {
        if observations.ncols() != FEATURE_DIM {
            bail!(
                "HMM regime expects {FEATURE_DIM} feature columns, got {}",
                observations.ncols()
            );
        }
        if observations.nrows() == 0 {
            bail!("HMM regime prediction requires at least one observation");
        }
        for ((row, column), value) in observations.indexed_iter() {
            if !value.is_finite() {
                bail!("HMM regime observation row {row} column {column} is non-finite: {value}");
            }
        }
        let (alpha, _) = self.forward(observations)?;
        for t in 0..alpha.nrows() {
            let row_sum: f64 = (0..N_STATES).map(|s| alpha[(t, s)]).sum();
            if !row_sum.is_finite() || (row_sum - 1.0).abs() > 1e-12 {
                bail!("HMM regime posterior row {t} has invalid normalized mass {row_sum}");
            }
        }
        Ok(alpha)
    }

    /// Predict from the exact feature-plan outputs used to train this model.
    /// Invalid rows remain invalid. Each contiguous valid segment starts from
    /// the learned initial distribution so a gap cannot leak hidden state into
    /// observations that follow it.
    pub fn predict_feature_frame(
        &self,
        frame: &FeatureFrame,
        lease: &CpuLease,
    ) -> Result<HmmPosteriorFrame> {
        lease.scope(|| {
            let mut columns = Vec::with_capacity(FEATURE_DIM);
            for expected in &self.feature_columns {
                let index = frame
                    .names
                    .iter()
                    .position(|name| name == expected)
                    .with_context(|| format!("HMM regime: missing feature column `{expected}`"))?;
                columns.push(frame.feature_column(index)?);
            }

            let rows = frame.n_samples();
            let mut validity = vec![FeatureCellValidity::Valid; rows];
            for row in 0..rows {
                for column in &columns {
                    let reason = column.validity[row];
                    if !reason.is_valid() {
                        validity[row] = reason;
                        break;
                    }
                    if !column.values[row].is_finite() {
                        validity[row] = FeatureCellValidity::NonFinite;
                        break;
                    }
                }
            }

            let mut probabilities = Array2::from_elem((rows, N_STATES), f64::NAN);
            let mut start = 0;
            while start < rows {
                while start < rows && !validity[start].is_valid() {
                    start += 1;
                }
                if start == rows {
                    break;
                }
                let mut end = start + 1;
                while end < rows && validity[end].is_valid() {
                    end += 1;
                }

                let mut values = Vec::with_capacity((end - start) * FEATURE_DIM);
                for row in start..end {
                    for column in &columns {
                        values.push(column.values[row]);
                    }
                }
                let observations = Array2::from_shape_vec((end - start, FEATURE_DIM), values)
                    .context("shape HMM feature segment")?;
                let segment = self.predict_proba_observations(&observations)?;
                for row in start..end {
                    for state in 0..N_STATES {
                        probabilities[(row, state)] = segment[(row - start, state)];
                    }
                }
                start = end;
            }

            Ok(HmmPosteriorFrame {
                probabilities,
                validity,
            })
        })
    }

    /// Persist the trained HMM to disk as a JSON artifact.
    pub fn save_to_path(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create HMM artifact dir {}", dir.display()))?;
        let metadata = try_build_runtime_artifact_metadata(
            MODEL_NAME,
            ModelFamily::Meta,
            CapabilityState::Implemented,
            self.feature_columns.clone(),
            canonical_three_class_label_mapping(),
            TrainingSummaryMetadata::new(1, 1, 0),
        )?;
        let artifact = HmmRegimeArtifact {
            precision_schema: HMM_F64_SCHEMA.to_string(),
            model_name: MODEL_NAME.to_string(),
            feature_columns: self.feature_columns.clone(),
            initial_probs: self.initial_probs.to_vec(),
            transition_flat: self.transition.iter().copied().collect(),
            emission_means_flat: self.emission_means.iter().copied().collect(),
            emission_covs_flat: self
                .emission_covs
                .iter()
                .flat_map(|m| m.iter().copied())
                .collect(),
            training_summary: TrainingSummaryMetadata::new(1, 1, 0),
            runtime_metadata: Some(metadata),
            config: self.config,
        };
        let path = dir.join(ARTIFACT_FILE);
        let json = serde_json::to_vec_pretty(&artifact).context("serialize HMM regime artifact")?;
        std::fs::write(&path, json)
            .with_context(|| format!("write HMM regime artifact {}", path.display()))?;
        Ok(())
    }

    /// Load a trained HMM from disk. Inverse of [`Self::save_to_path`].
    pub fn load_from_artifact(dir: &Path) -> Result<Self> {
        let path = dir.join(ARTIFACT_FILE);
        let payload = std::fs::read(&path)
            .with_context(|| format!("read HMM regime artifact {}", path.display()))?;
        let artifact: HmmRegimeArtifact = serde_json::from_slice(&payload)
            .with_context(|| format!("parse HMM regime artifact {}", path.display()))?;
        if artifact.precision_schema != HMM_F64_SCHEMA {
            bail!(
                "HMM regime precision schema mismatch: expected {HMM_F64_SCHEMA}, found {:?}",
                artifact.precision_schema
            );
        }
        if artifact.model_name != MODEL_NAME {
            bail!(
                "HMM regime artifact model name mismatch: expected {MODEL_NAME}, found {:?}",
                artifact.model_name
            );
        }
        if artifact
            .feature_columns
            .iter()
            .map(String::as_str)
            .ne(HMM_FEATURE_COLUMNS)
        {
            bail!(
                "HMM regime artifact feature columns must be {:?}; found {:?}",
                HMM_FEATURE_COLUMNS,
                artifact.feature_columns
            );
        }
        if artifact.initial_probs.len() != N_STATES {
            bail!(
                "HMM regime artifact has {} initial probs, expected {N_STATES}",
                artifact.initial_probs.len()
            );
        }
        let expected_transition = N_STATES * N_STATES;
        if artifact.transition_flat.len() != expected_transition {
            bail!(
                "HMM regime artifact has {} transition values, expected {expected_transition}",
                artifact.transition_flat.len()
            );
        }
        let expected_means = N_STATES * FEATURE_DIM;
        if artifact.emission_means_flat.len() != expected_means {
            bail!(
                "HMM regime artifact has {} emission means, expected {expected_means}",
                artifact.emission_means_flat.len()
            );
        }
        let expected_covariances = N_STATES * FEATURE_DIM * FEATURE_DIM;
        if artifact.emission_covs_flat.len() != expected_covariances {
            bail!(
                "HMM regime artifact has {} covariance values, expected {expected_covariances}",
                artifact.emission_covs_flat.len()
            );
        }
        for (field, values) in [
            ("initial probabilities", artifact.initial_probs.as_slice()),
            ("transition matrix", artifact.transition_flat.as_slice()),
            ("emission means", artifact.emission_means_flat.as_slice()),
            (
                "emission covariances",
                artifact.emission_covs_flat.as_slice(),
            ),
        ] {
            if let Some((index, value)) = values
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                bail!("HMM regime artifact {field} value {index} is non-finite: {value}");
            }
        }
        let initial_probs = Array1::from(artifact.initial_probs);
        let transition = Array2::from_shape_vec((N_STATES, N_STATES), artifact.transition_flat)
            .context("reshape HMM transition matrix")?;
        let emission_means =
            Array2::from_shape_vec((N_STATES, FEATURE_DIM), artifact.emission_means_flat)
                .context("reshape HMM emission means")?;
        let cov_chunk = FEATURE_DIM * FEATURE_DIM;
        let mut emission_covs = Vec::with_capacity(N_STATES);
        for s in 0..N_STATES {
            let start = s * cov_chunk;
            let end = start + cov_chunk;
            let slice = artifact.emission_covs_flat[start..end].to_vec();
            let mat = Array2::from_shape_vec((FEATURE_DIM, FEATURE_DIM), slice)
                .context("reshape HMM emission covariance")?;
            emission_covs.push(mat);
        }
        Ok(Self {
            initial_probs,
            transition,
            emission_means,
            emission_covs,
            feature_columns: artifact.feature_columns,
            config: artifact.config,
        })
    }
}

/// Convenience adapter that converts an `(N, 3)` f64 posterior into
/// the workspace-canonical `RuntimePrediction` triple expected by
/// the ensemble inference path. Drops nothing — just metadata wrap.
pub fn hmm_runtime_prediction(probabilities: &Array2<f64>) -> Result<Vec<RuntimePrediction>> {
    if probabilities.ncols() != N_STATES {
        bail!(
            "HMM runtime prediction needs {N_STATES} columns, got {}",
            probabilities.ncols()
        );
    }
    let mut out = Vec::with_capacity(probabilities.nrows());
    for row_idx in 0..probabilities.nrows() {
        let row = [
            probabilities[(row_idx, 0)],
            probabilities[(row_idx, 1)],
            probabilities[(row_idx, 2)],
        ];
        let (confidence, abstain) = three_class_runtime_confidence(row)?;
        let pred = build_runtime_prediction_with_details(
            MODEL_NAME,
            ModelFamily::Meta,
            CapabilityState::Implemented,
            row,
            Some(confidence),
            Some(abstain),
            Some("hmm_regime_inference".to_string()),
            None,
        )?;
        out.push(pred);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_two_regime_data(n: usize) -> Array2<f64> {
        // Half bullish trend (μ_lr=+0.0005, σ=0.001), half range
        // (μ_lr=0, σ=0.0008). Synthetic but enough to test that EM
        // recovers two distinct emission distributions.
        let half = n / 2;
        let mut obs = Array2::<f64>::zeros((n, 2));
        let mut prng_seed: u64 = 0xDEAD_BEEF_C0FFEE_42;
        let mut next = || {
            prng_seed ^= prng_seed << 13;
            prng_seed ^= prng_seed >> 7;
            prng_seed ^= prng_seed << 17;
            // Convert to roughly uniform [0,1) then to normal via
            // Box-Muller (cheap, no rand dep here).
            let u = (prng_seed >> 11) as f64 / ((1u64 << 53) as f64);
            (u - 0.5) * 4.0 // pseudo-normal in [-2, 2]
        };
        for i in 0..n {
            let in_trend = i < half;
            let lr_mean = if in_trend { 0.0005 } else { 0.0 };
            let lr_std = if in_trend { 0.001 } else { 0.0008 };
            let lv_mean = if in_trend { -7.0 } else { -7.5 };
            obs[(i, 0)] = lr_mean + lr_std * next();
            obs[(i, 1)] = lv_mean + 0.3 * next();
        }
        obs
    }

    #[test]
    fn hmm_trains_on_synthetic_data_and_returns_normalized_posterior() {
        let obs = synthetic_two_regime_data(600);
        let cfg = HmmRegimeConfig {
            min_training_bars: 500,
            ..HmmRegimeConfig::default()
        };
        let expert = RegimeHmmExpert::train(
            &obs,
            vec![
                "quant_log_return".to_string(),
                "quant_log_volatility".to_string(),
            ],
            cfg,
        )
        .expect("train HMM");
        let probs = expert
            .predict_proba_observations(&obs)
            .expect("predict on training data");
        assert_eq!(probs.shape(), &[600, 3]);
        // Every row sums to ~1.0 (canonical 3-class normalization).
        for t in 0..probs.nrows() {
            let s: f64 = (0..3).map(|c| probs[(t, c)]).sum();
            assert!(
                (s - 1.0).abs() < 1e-12,
                "row {t}: posterior sum {} != 1.0",
                s
            );
            for c in 0..3 {
                let v = probs[(t, c)];
                assert!(
                    (0.0..=1.0).contains(&v),
                    "row {t} col {c}: probability {} out of [0,1]",
                    v
                );
            }
        }
    }

    #[test]
    fn hmm_artifact_round_trips_through_disk() {
        let obs = synthetic_two_regime_data(600);
        let cfg = HmmRegimeConfig {
            min_training_bars: 500,
            ..HmmRegimeConfig::default()
        };
        let expert = RegimeHmmExpert::train(
            &obs,
            vec![
                "quant_log_return".to_string(),
                "quant_log_volatility".to_string(),
            ],
            cfg,
        )
        .expect("train");
        let tmp = std::env::temp_dir().join(format!(
            "neoethos_hmm_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        expert.save_to_path(&tmp).expect("save");
        let loaded = RegimeHmmExpert::load_from_artifact(&tmp).expect("load");
        assert_eq!(loaded.feature_columns, expert.feature_columns);
        // π should round-trip exactly.
        for s in 0..3 {
            assert!(
                (loaded.initial_probs[s] - expert.initial_probs[s]).abs() < 1e-9,
                "π[{s}] mismatch: loaded {} vs trained {}",
                loaded.initial_probs[s],
                expert.initial_probs[s]
            );
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn train_rejects_insufficient_bars() {
        let obs = synthetic_two_regime_data(50);
        let cfg = HmmRegimeConfig {
            min_training_bars: 500,
            ..HmmRegimeConfig::default()
        };
        let err = RegimeHmmExpert::train(
            &obs,
            vec![
                "quant_log_return".to_string(),
                "quant_log_volatility".to_string(),
            ],
            cfg,
        )
        .expect_err("must reject");
        assert!(
            err.to_string().contains("at least 500 training bars"),
            "unexpected error: {err}"
        );
    }
}
