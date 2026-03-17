#[cfg(feature = "lightgbm")]
use lightgbm3;
use anyhow::{Result, Context};
use ndarray::Array2;
use polars::prelude::*;
use crate::base::ExpertModel;

pub struct LightGBMExpert {
    pub idx: usize,
    #[cfg(feature = "lightgbm")]
    model: Option<lightgbm3::Booster>,
    #[cfg(not(feature = "lightgbm"))]
    model: Option<()>,
}

impl LightGBMExpert {
    pub fn new(idx: usize) -> Self {
        Self { idx, model: None }
    }
}

impl ExpertModel for LightGBMExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(not(feature = "lightgbm"))]
        { anyhow::bail!("LightGBM feature not enabled") }
        #[cfg(feature = "lightgbm")]
        { Ok(()) }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        Ok(Array2::zeros((x.height(), 3)))
    }

    fn save(&self, _path: &std::path::Path) -> Result<()> { Ok(()) }
    fn load(&mut self, _path: &std::path::Path) -> Result<()> { Ok(()) }
}
