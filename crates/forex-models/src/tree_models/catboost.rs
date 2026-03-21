#[cfg(feature = "catboost")]
use catboost_rust as catboost;
use anyhow::Result;
use ndarray::Array2;
use polars::prelude::*;
use crate::base::ExpertModel;

pub struct CatBoostExpert {
    pub idx: usize,
    #[cfg(feature = "catboost")]
    _model: Option<catboost::Model>,
    #[cfg(not(feature = "catboost"))]
    _model: Option<()>,
}

impl CatBoostExpert {
    pub fn new(idx: usize) -> Self {
        Self { idx, _model: None }
    }
}

impl ExpertModel for CatBoostExpert {
    fn fit(&mut self, _x: &DataFrame, _y: &Series) -> Result<()> {
        #[cfg(not(feature = "catboost"))]
        { anyhow::bail!("CatBoost feature not enabled") }
        #[cfg(feature = "catboost")]
        { Ok(()) }
    }

    fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        Ok(Array2::zeros((x.height(), 3)))
    }

    fn save(&self, _path: &std::path::Path) -> Result<()> { Ok(()) }
    fn load(&mut self, _path: &std::path::Path) -> Result<()> { Ok(()) }
}
