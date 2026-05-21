use crate::core::feature_registry::{
    FeatureColumnMetadata, feature_metadata_for_names, validate_feature_names,
};
use anyhow::Result;
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FeatureProfile {
    #[default]
    Standard,
    Full,
    HPC,
    Adaptive,
}

impl FromStr for FeatureProfile {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard" => Ok(Self::Standard),
            "full" => Ok(Self::Full),
            "hpc" => Ok(Self::HPC),
            "adaptive" => Ok(Self::Adaptive),
            _ => Err(format!("unknown feature profile: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureBuildOptions {
    pub profile: FeatureProfile,
    pub include_smc: bool,
    pub include_hpc_ta: bool,
    pub include_regime: bool,
    pub include_quant: bool,
    pub prefix_base_features: bool,
    pub higher_tfs: Vec<String>,
}

impl Default for FeatureBuildOptions {
    fn default() -> Self {
        Self {
            profile: FeatureProfile::Standard,
            include_smc: true,
            include_hpc_ta: true,
            include_regime: true,
            include_quant: true,
            prefix_base_features: false,
            higher_tfs: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FeatureFrame {
    pub timestamps: Vec<i64>,
    pub names: Vec<String>,
    pub data: Array2<f32>,
}

impl FeatureFrame {
    pub fn column_metadata(&self) -> Result<Vec<FeatureColumnMetadata>> {
        feature_metadata_for_names(&self.names)
    }

    pub fn validate_registry(&self) -> Result<()> {
        validate_feature_names(&self.names)
    }
}

pub fn align_features_by_ns(
    base_ns: &[i64],
    feature_ns: &[i64],
    feature_data: &Array2<f32>,
    ffill: bool,
) -> Array2<f32> {
    let n_base = base_ns.len();
    let n_feat = feature_ns.len();
    let n_cols = feature_data.ncols();
    let mut out = Array2::from_elem((n_base, n_cols), f32::NAN);

    if n_feat == 0 {
        return out;
    }

    let mut feat_idx = 0usize;
    for i in 0..n_base {
        let ts = base_ns[i];
        while feat_idx < n_feat && feature_ns[feat_idx] <= ts {
            feat_idx += 1;
        }

        let best_idx = if feat_idx > 0 {
            let prev = feat_idx - 1;
            if feature_ns[prev] == ts || ffill {
                Some(prev)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(idx) = best_idx {
            for j in 0..n_cols {
                out[(i, j)] = feature_data[(idx, j)];
            }
        }
    }
    out
}
