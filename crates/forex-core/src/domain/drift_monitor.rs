use std::collections::VecDeque;
use tracing::{info, warn};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct DriftMonitorStatus {
    pub drift_detected: bool,
    pub drift_magnitude: f64,
    pub drift_method: String,
    pub errors_tracked: usize,
    pub last_drift_at: Option<u64>,
    pub ks_statistic: f64,
    pub psi_score: f64,
    pub kl_divergence: f64,
}

pub struct ConceptDriftMonitor {
    pub window_size: usize,
    pub threshold: f64,
    pub error_stream: VecDeque<f64>,
    pub mean_error: f64,
    pub variance: f64,
    pub drift_detected: bool,
    pub last_drift_at: Option<u64>,

    pub drift_magnitude: f64,
    pub drift_method_used: String,
    pub ks_statistic: f64,
    pub psi_score: f64,
    pub kl_divergence: f64,

    // Feature monitor stats 
    pub feature_stats: std::collections::HashMap<String, FeatureStat>,
    pub alpha: f64,
}

#[derive(Debug, Clone)]
pub struct FeatureStat {
    pub mean: f64,
    pub std: f64,
    pub initialized: bool,
}

impl ConceptDriftMonitor {
    pub fn new(window_size: Option<usize>, threshold: Option<f64>) -> Self {
        let window_size = window_size.unwrap_or(100);
        Self {
            window_size,
            threshold: threshold.unwrap_or(0.05),
            error_stream: VecDeque::with_capacity(window_size * 2),
            mean_error: 0.0,
            variance: 0.0,
            drift_detected: false,
            last_drift_at: None,
            drift_magnitude: 0.0,
            drift_method_used: String::new(),
            ks_statistic: 0.0,
            psi_score: 0.0,
            kl_divergence: 0.0,
            feature_stats: std::collections::HashMap::new(),
            alpha: 0.01,
        }
    }

    pub fn update(&mut self, y_true: i32, y_pred_prob: &[f64]) -> bool {
        let idx = match y_true {
            0 => 0,
            1 => 1,
            -1 => 2,
            _ => 0,
        };

        let prob_correct = if y_pred_prob.is_empty() {
            0.0
        } else if idx < y_pred_prob.len() {
            y_pred_prob[idx]
        } else {
            y_pred_prob[0]
        };

        let error = 1.0 - prob_correct;

        if self.error_stream.len() >= self.window_size * 2 {
            self.error_stream.pop_front();
        }
        self.error_stream.push_back(error);

        let n = self.error_stream.len() as f64;
        let sum: f64 = self.error_stream.iter().sum();
        self.mean_error = sum / n;
        
        if n > 1.0 {
            let var_sum: f64 = self.error_stream.iter().map(|&x| {
                let diff = x - self.mean_error;
                diff * diff
            }).sum();
            self.variance = var_sum / (n - 1.0); // sample variance
        } else {
            self.variance = 0.0;
        }

        if self.error_stream.len() >= self.window_size {
            self.drift_detected = self.check_drift();
            if self.drift_detected {
                self.last_drift_at = Some(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());
            }
        }

        self.drift_detected
    }

    fn check_drift(&mut self) -> bool {
        let n = self.error_stream.len();
        if n < self.window_size * 2 { return false; }

        let mid = n / 2;
        let mut slice1: Vec<f64> = self.error_stream.iter().take(mid).copied().collect();
        let mut slice2: Vec<f64> = self.error_stream.iter().skip(mid).copied().collect();

        // 1. Z-Score Mean Shift
        let err_std = variance(&slice1).sqrt() + 1e-6;
        let mu1: f64 = slice1.iter().sum::<f64>() / mid as f64;
        let mu2: f64 = slice2.iter().sum::<f64>() / (n - mid) as f64;

        let z_shift = (mu1 - mu2).abs() / err_std;
        let variance_drift = z_shift > 3.0;

        // 2. KS Test
        let mut ks_drift = false;
        slice1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        slice2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let (ks_stat, ks_pval) = Self::ks_2samp(&slice1, &slice2);
        self.ks_statistic = ks_stat;
        if ks_pval < 0.001 {
            ks_drift = true;
        }

        // 3. PSI
        let mut psi_drift = false;
        self.psi_score = self.calculate_psi(&slice1, &slice2, 10);
        if self.psi_score > 0.40 {
            psi_drift = true;
        }

        let mut drift_votes = 0;
        if variance_drift { drift_votes += 1; }
        if ks_drift { drift_votes += 1; }
        if psi_drift { drift_votes += 1; }

        let drift_detected = drift_votes >= 2;
        if drift_detected {
            warn!("REAL Drift Detected (Z-Shift={:.2})", z_shift);
        }

        drift_detected
    }

    fn ks_2samp(data1: &[f64], data2: &[f64]) -> (f64, f64) {
        let n1 = data1.len() as f64;
        let n2 = data2.len() as f64;
        let mut max_d = 0.0f64;

        let mut i = 0;
        let mut j = 0;

        while i < data1.len() && j < data2.len() {
            let val1 = data1[i];
            let val2 = data2[j];

            let d = if val1 <= val2 {
                while i < data1.len() && data1[i] <= val1 { i += 1; }
                let cdf1 = i as f64 / n1;
                while j < data2.len() && data2[j] <= val1 { j += 1; }
                let cdf2 = j as f64 / n2;
                (cdf1 - cdf2).abs()
            } else {
                while j < data2.len() && data2[j] <= val2 { j += 1; }
                let cdf2 = j as f64 / n2;
                while i < data1.len() && data1[i] <= val2 { i += 1; }
                let cdf1 = i as f64 / n1;
                (cdf1 - cdf2).abs()
            };

            if d > max_d {
                max_d = d;
            }
        }
        
        // P-value approximation
        let en = (n1 * n2) / (n1 + n2);
        let arg = (en.sqrt() + 0.12 + 0.11 / en.sqrt()) * max_d;
        let p_val = if arg < 0.0 { 1.0 } else { smirnov_p(arg) };
        
        (max_d, p_val.clamp(0.0, 1.0))
    }

    fn calculate_psi(&self, expected: &[f64], actual: &[f64], bins: usize) -> f64 {
        if expected.is_empty() || actual.is_empty() { return 0.0; }

        let min_val = expected.first().unwrap();
        let max_val = expected.last().unwrap();
        
        if (max_val - min_val).abs() < 1e-9 { return 0.0; }

        let step = (max_val - min_val) / bins as f64;
        let mut expected_counts = vec![0.0; bins];
        let mut actual_counts = vec![0.0; bins];

        for &val in expected {
            let mut b = ((val - min_val) / step).floor() as usize;
            if b >= bins { b = bins - 1; }
            expected_counts[b] += 1.0;
        }

        for &val in actual {
            let mut b = ((val - min_val) / step).floor() as usize;
            if b >= bins { b = bins - 1; }
            actual_counts[b] += 1.0;
        }

        let e_sum: f64 = expected_counts.iter().sum();
        let a_sum: f64 = actual_counts.iter().sum();
        
        let mut psi = 0.0;
        for i in 0..bins {
            let e_pct = (expected_counts[i] + 1e-6) / (e_sum + bins as f64 * 1e-6);
            let a_pct = (actual_counts[i] + 1e-6) / (a_sum + bins as f64 * 1e-6);
            psi += (a_pct - e_pct) * (a_pct / e_pct).ln();
        }

        psi.abs()
    }

    pub fn should_retrain(&self) -> bool {
        self.drift_detected
    }

    pub fn reset_after_retrain(&mut self) {
        self.drift_detected = false;
        self.error_stream.clear();
        self.mean_error = 0.0;
        self.variance = 0.0;
        self.drift_magnitude = 0.0;
        self.drift_method_used = String::new();
        info!("Drift monitor reset after retraining.");
    }
}

fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / (data.len() as f64 - 1.0)
}

fn smirnov_p(x: f64) -> f64 {
    // Asymptotic p-value for KS test
    if x < 0.27 { return 1.0; }
    if x > 3.0 { return 0.0; }
    
    let mut sum = 0.0;
    for k in 1..=5 {
        let f = k as f64;
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        sum += sign * f64::exp(-2.0 * f * f * x * x);
    }
    
    let prob = 2.0 * sum;
    prob.clamp(0.0, 1.0)
}
