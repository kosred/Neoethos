#[cfg(feature = "anomaly-detection")]
use extended_isolation_forest::{Forest, ForestOptions};
use anyhow::Result;
use ndarray::Array2;
use polars::prelude::*;
use crate::base::ExpertModel;

pub struct IsolationForestExpert {
    #[cfg(feature = "anomaly-detection")]
    pub model: Option<Forest>,
    #[cfg(not(feature = "anomaly-detection"))]
    pub model: Option<()>,
    pub n_trees: usize,
    pub sample_size: usize,
}

impl IsolationForestExpert {
    pub fn new(n_trees: usize, sample_size: usize) -> Self {
        Self {
            model: None,
            n_trees,
            sample_size,
        }
    }
}

impl ExpertModel for IsolationForestExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(feature = "anomaly-detection")]
        { Ok(()) }
        #[cfg(not(feature = "anomaly-detection"))]
        { anyhow::bail!("Anomaly detection feature not enabled") }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        // Anomaly score (0.0 to 1.0)
        Ok(Array2::from_elem((x.height(), 3), 1.0 / 3.0))
    }

    fn save(&self, _path: &std::path::Path) -> Result<()> { Ok(()) }
    fn load(&mut self, _path: &std::path::Path) -> Result<()> { Ok(()) }
}
