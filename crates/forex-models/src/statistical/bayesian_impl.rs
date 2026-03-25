#[cfg(feature = "statistical-models")]
use variational_regression::BayesianLinearRegression;
use anyhow::Result;
use ndarray::Array2;
use polars::prelude::*;
use crate::base::ExpertModel;

pub struct BayesianLogitExpert {
    #[cfg(feature = "statistical-models")]
    pub model: Option<BayesianLinearRegression>,
    #[cfg(not(feature = "statistical-models"))]
    pub model: Option<()>,
}

impl BayesianLogitExpert {
    pub fn new() -> Self {
        Self {
            model: None,
        }
    }
}

impl Default for BayesianLogitExpert {
    fn default() -> Self {
        Self::new()
    }
}

impl ExpertModel for BayesianLogitExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(feature = "statistical-models")]
        { Ok(()) }
        #[cfg(not(feature = "statistical-models"))]
        { anyhow::bail!("Statistical models feature not enabled") }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        Ok(Array2::from_elem((x.height(), 3), 1.0 / 3.0))
    }

    fn save(&self, _path: &std::path::Path) -> Result<()> { Ok(()) }
    fn load(&mut self, _path: &std::path::Path) -> Result<()> { Ok(()) }
}
