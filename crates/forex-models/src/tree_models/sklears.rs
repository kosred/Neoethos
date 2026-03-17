#[cfg(feature = "sklears-tree")]
use sklears::tree::DecisionTreeClassifier;
#[cfg(feature = "sklears-tree")]
use sklears::traits::{Fit, Predict};
use anyhow::{Result, Context};
use ndarray::Array2;
use polars::prelude::*;
use crate::base::ExpertModel;

pub struct SklearsTreeExpert {
    #[cfg(feature = "sklears-tree")]
    model: Option<DecisionTreeClassifier>,
    #[cfg(not(feature = "sklears-tree"))]
    model: Option<()>,
}

impl SklearsTreeExpert {
    pub fn new() -> Self {
        Self { model: None }
    }
}

impl ExpertModel for SklearsTreeExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(feature = "sklears-tree")]
        {
            // Conversion and training logic
            Ok(())
        }
        #[cfg(not(feature = "sklears-tree"))]
        { anyhow::bail!("sklears-tree feature not enabled") }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        Ok(Array2::zeros((x.height(), 3)))
    }

    fn save(&self, _path: &std::path::Path) -> Result<()> { Ok(()) }
    fn load(&mut self, _path: &std::path::Path) -> Result<()> { Ok(()) }
}
