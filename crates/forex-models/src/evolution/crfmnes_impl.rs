#[cfg(feature = "neuro-evolution")]
use crfmnes::CrfmnesOptimizer;
#[cfg(feature = "neuro-evolution")]
use nalgebra::DVector;
use anyhow::{Result, Context};
use rand::SeedableRng;
use rand_xoshiro::Xoroshiro128PlusPlus;

pub struct NeuroEvoOptimizer {
    #[cfg(feature = "neuro-evolution")]
    pub optimizer: Option<CrfmnesOptimizer<Xoroshiro128PlusPlus>>,
    #[cfg(not(feature = "neuro-evolution"))]
    pub optimizer: Option<()>,
    pub dim: usize,
}

impl NeuroEvoOptimizer {
    pub fn new(dim: usize, _sigma: f64) -> Self {
        #[cfg(feature = "neuro-evolution")]
        {
            let mut rng = Xoroshiro128PlusPlus::from_entropy();
            let start_m = DVector::from_element(dim, 0.0);
            let opt = CrfmnesOptimizer::new(start_m, _sigma, None, &mut rng);
            Self {
                optimizer: Some(opt),
                dim,
            }
        }
        #[cfg(not(feature = "neuro-evolution"))]
        {
            Self { optimizer: None, dim }
        }
    }

    pub fn ask(&mut self) -> Result<Vec<Vec<f64>>> {
        #[cfg(feature = "neuro-evolution")]
        {
            let opt = self.optimizer.as_mut().context("Optimizer not initialized")?;
            let mut rng = Xoroshiro128PlusPlus::from_entropy();
            let samples = opt.ask(&mut rng);
            Ok(samples.into_iter().map(|v| v.as_slice().to_vec()).collect())
        }
        #[cfg(not(feature = "neuro-evolution"))]
        {
            anyhow::bail!("Neuro-evolution feature not enabled")
        }
    }

    pub fn tell(&mut self, _fitness_values: Vec<f64>) -> Result<()> {
        #[cfg(feature = "neuro-evolution")]
        {
            let opt = self.optimizer.as_mut().context("Optimizer not initialized")?;
            opt.tell(_fitness_values).map_err(|e| anyhow::anyhow!("CRFMNES tell failed: {}", e))
        }
        #[cfg(not(feature = "neuro-evolution"))]
        {
            anyhow::bail!("Neuro-evolution feature not enabled")
        }
    }

    pub fn best_weights(&self) -> Result<Vec<f64>> {
        #[cfg(feature = "neuro-evolution")]
        {
            let opt = self.optimizer.as_ref().context("Optimizer not initialized")?;
            Ok(opt.best_x().as_slice().to_vec())
        }
        #[cfg(not(feature = "neuro-evolution"))]
        {
            anyhow::bail!("Neuro-evolution feature not enabled")
        }
    }
}
