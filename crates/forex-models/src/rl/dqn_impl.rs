#[cfg(feature = "reinforcement-learning")]
use rlkit::algs::dqn::{DQN, DNQStateMode};
#[cfg(feature = "reinforcement-learning")]
use rlkit::types::{EnvTrait, Status, Reward, Action, TrainArgs};
#[cfg(feature = "reinforcement-learning")]
use rlkit::policies::EpsilonGreedy;
use anyhow::{Result, Context};
use ndarray::Array2;

pub struct TradingReinforcementLearner {
    #[cfg(feature = "reinforcement-learning")]
    model: Option<DQN<u16, u16>>,
}

impl TradingReinforcementLearner {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "reinforcement-learning")]
            model: None,
        }
    }

    // This will be expanded once we have the Environment trait implemented for our forex data
    pub fn train(&mut self) -> Result<()> {
        #[cfg(not(feature = "reinforcement-learning"))]
        { anyhow::bail!("Reinforcement learning feature not enabled") }
        #[cfg(feature = "reinforcement-learning")]
        { Ok(()) }
    }
}
