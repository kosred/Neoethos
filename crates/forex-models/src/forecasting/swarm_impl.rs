#[cfg(feature = "swarm-forecasting")]
use ruv_swarm_ml::agent_forecasting::AgentForecastingManager;
use anyhow::Result;

pub struct SwarmForecaster {
    #[cfg(feature = "swarm-forecasting")]
    pub manager: Option<AgentForecastingManager>,
    #[cfg(not(feature = "swarm-forecasting"))]
    pub manager: Option<()>,
}

impl SwarmForecaster {
    pub fn new(memory_limit_mb: f64) -> Self {
        #[cfg(feature = "swarm-forecasting")]
        {
            Self {
                manager: Some(AgentForecastingManager::new(memory_limit_mb)),
            }
        }
        #[cfg(not(feature = "swarm-forecasting"))]
        {
            let _ = memory_limit_mb;
            Self { manager: None }
        }
    }

    pub fn train(&mut self) -> Result<()> {
        #[cfg(not(feature = "swarm-forecasting"))]
        { anyhow::bail!("Swarm forecasting feature not enabled") }
        #[cfg(feature = "swarm-forecasting")]
        { Ok(()) }
    }
}
