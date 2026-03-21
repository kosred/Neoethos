#[cfg(feature = "adaptive-models")]
use irithyll::{SGBT, SGBTConfig, Sample};
use anyhow::Result;
#[cfg(feature = "adaptive-models")]
use anyhow::Context;

pub struct AdaptiveGradientBooster {
    #[cfg(feature = "adaptive-models")]
    pub inner: Option<SGBT>,
    #[cfg(not(feature = "adaptive-models"))]
    pub inner: Option<()>,
}

impl AdaptiveGradientBooster {
    pub fn new() -> Self {
        #[cfg(feature = "adaptive-models")]
        {
            let config = SGBTConfig::default();
            Self {
                inner: Some(SGBT::new(config)),
            }
        }
        #[cfg(not(feature = "adaptive-models"))]
        {
            Self { inner: None }
        }
    }

    pub fn learn_one(&mut self, _x: Vec<f64>, _y: f64) -> Result<()> {
        #[cfg(feature = "adaptive-models")]
        {
            let model = self.inner.as_mut().context("Model not initialized")?;
            let sample = Sample::new(_x, _y);
            model.train_one(&sample);
            Ok(())
        }
        #[cfg(not(feature = "adaptive-models"))]
        {
            anyhow::bail!("Adaptive models feature not enabled")
        }
    }

    pub fn predict_one(&self, _x: &Vec<f64>) -> Result<f64> {
        #[cfg(feature = "adaptive-models")]
        {
            let model = self.inner.as_ref().context("Model not initialized")?;
            Ok(model.predict_one(_x))
        }
        #[cfg(not(feature = "adaptive-models"))]
        {
            anyhow::bail!("Adaptive models feature not enabled")
        }
    }
}

impl Default for AdaptiveGradientBooster {
    fn default() -> Self {
        Self::new()
    }
}
