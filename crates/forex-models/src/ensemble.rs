use anyhow::{Context, Result};
use ndarray::Array2;
use std::path::Path;
use serde::{Deserialize, Serialize};
use crate::base::ExpertModel;
use crate::tree_models::XGBoostExpert;
use polars::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct MetaBlender {
    #[serde(skip)]
    pub model: Option<XGBoostExpert>,
    pub feature_columns: Vec<String>,
    pub fitted: bool,
}

impl MetaBlender {
    pub fn new() -> Self {
        Self {
            model: None,
            feature_columns: Vec::new(),
            fitted: false,
        }
    }

    pub fn fit(&mut self, x: &DataFrame, y: &Series) -> Result<()> {
        let mut model = XGBoostExpert::new(0, None);
        model.fit(x, y)?;
        self.model = Some(model);
        self.feature_columns = x.get_column_names().iter().map(|s| s.to_string()).collect();
        self.fitted = true;
        Ok(())
    }

    pub fn predict_proba(&self, x: &DataFrame) -> Result<Array2<f32>> {
        let model = self.model.as_ref().context("MetaBlender not fitted")?;
        model.predict_proba(x)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(m) = &self.model {
            m.save(path)?;
        }
        Ok(())
    }
}

impl Default for MetaBlender {
    fn default() -> Self {
        Self::new()
    }
}
