//! Policy module, containing implementations of various action selection policies.

use crate::{types::{Action, QValue}};
use candle_core::{Tensor, Result};  
use rand::{Rng, rng};
use candle_nn::ops; 

/// Policy configuration
#[deprecated(
    since = "0.0.3",
    note = "The enum PolicyConfig is deprecated, please directly instantiated the policy."
)]
#[derive(Debug, Default, Clone)]
pub enum PolicyConfig {
    EpsilonGreedy {
        epsilon_start: f32,
        epsilon_min: f32,
        epsilon_decay: f32,
    },
    Boltzmann {
        temperature_start: f32,
        temperature_min: f32,
        temperature_decay: f32,
    },
    OrnsteinUhlenbeck {
        mu: f32,
        theta: f32,
        sigma: f32,
        action_dim: usize,
    },
    GaussianNoise {
        mean: f32,
        std_dev: f32,
        decay_rate: f32,
    },
    #[default]
    DeterministicPolicy,
}

impl PolicyConfig {
    /// Return the default ε-贪婪策略参数配置，常用在 DQN 中
    pub const fn dqn_epsilon_greedy() -> Self {
        Self::EpsilonGreedy {
            epsilon_start: 1.0,
            epsilon_min: 0.01,
            epsilon_decay: 0.995,
        }
    }

    /// Return the default Boltzmann strategy parameter configuration, commonly used in DDPG.
    pub const fn default_boltzmann() -> Self {
        Self::Boltzmann {
            temperature_start: 1.0,
            temperature_min: 0.1,
            temperature_decay: 0.99,
        }
    }

    /// Return the default Ornstein-Uhlenbeck process parameter configuration, commonly used in DDPG.
    pub const fn ddpg_ornstein_uhlenbeck(action_dim: usize) -> Self {
        Self::OrnsteinUhlenbeck {
            mu: 0.0,
            theta: 0.15,
            sigma: 0.2,
            action_dim,
        }
    }

    /// Return the default Gaussian noise strategy parameter configuration, commonly used in DDPG.
    pub const fn default_gaussian_noise() -> Self {
        Self::GaussianNoise {
            mean: 0.0,
            std_dev: 0.2,
            decay_rate: 0.99,
        }
    }
}

impl PolicyConfig {
    /// Create a policy instance based on the configuration.
    pub fn create_policy<T>(&self, action_dim: usize) -> Result<Box<dyn Policy<T>>>
    where
        T: Copy + From<f32> + std::ops::Add<Output = T>
            + rand::distr::uniform::SampleUniform + Default + std::cmp::PartialOrd + std::fmt::Display,
    {
        match self {
            Self::EpsilonGreedy { epsilon_start, epsilon_min, epsilon_decay } => {
                Ok(Box::new(EpsilonGreedy::new(*epsilon_start, *epsilon_min, *epsilon_decay)))
            }
            Self::Boltzmann { temperature_start, temperature_min, temperature_decay } => {
                Ok(Box::new(Boltzmann::new(*temperature_start, *temperature_min, *temperature_decay)))
            }
            Self::OrnsteinUhlenbeck { mu, theta, sigma, action_dim: _ } => {
                Ok(Box::new(OrnsteinUhlenbeck::new(*mu, *theta, *sigma, action_dim)))
            }
            Self::GaussianNoise { mean, std_dev, decay_rate } => {
                Ok(Box::new(GaussianNoise::new(*mean, *std_dev, *decay_rate)))
            }
            Self::DeterministicPolicy => {
                Ok(Box::new(DeterministicPolicy))
            }
        }
    }
}

/// Policy interface, defining methods for action selection.
pub trait Policy<T = u16> {
    /// Select an action based on the network output.
    fn select_action(&mut self, q_value: &QValue<T>) -> Result<Action<T>>;
    
    /// Update the policy parameters (e.g., ε value).
    fn update(&mut self);
    
    /// Get a string representation of the current policy parameters.
    fn get_params(&self) -> String;
}

/// ε-Greedy policy, commonly used in DQN for exploration.
pub struct EpsilonGreedy {
    /// Current ε value
    pub epsilon: f32,
    /// Minimum ε value
    pub epsilon_min: f32,
    /// ε decay rate
    pub epsilon_decay: f32,
}

impl EpsilonGreedy {
    /// Create a new ε-Greedy policy.
    /// 
    /// # Arguments
    /// * `epsilon_start` - Initial ε value
    /// * `epsilon_min` - Minimum ε value
    /// * `epsilon_decay` - ε decay rate
    pub fn new(epsilon_start: f32, epsilon_min: f32, epsilon_decay: f32) -> Self {
        Self {
            epsilon: epsilon_start,
            epsilon_min,
            epsilon_decay,
        }
    }
}

impl<T> Policy<T> for EpsilonGreedy
where
    T: Copy + rand::distr::uniform::SampleUniform + Default + std::cmp::PartialOrd,
{
    /// Select an action based on the ε-Greedy policy.
    /// 
    /// # Arguments
    /// * `q_values` - Q-value distribution for the current state
    fn select_action(&mut self, q_values: &QValue<T>) -> Result<Action<T>> {
        let mut rng = rng();
        
        match q_values {
            QValue::Deterministic(action) => {
                if rng.random::<f32>() < self.epsilon {
                    Ok(action.random(&mut rng))
                } else {
                    Ok(action.clone())
                }
            },
            QValue::Stochastic(actions_with_values) => {
                // 获取最好的动作
                let best_action = q_values.best_action().clone();

                if rng.random::<f32>() < self.epsilon {
                    // 从所有可用动作中随机选择一个
                    let random_idx = rng.random_range(0..actions_with_values.len());
                    Ok(actions_with_values[random_idx].0.clone())
                } else {
                    Ok(best_action.clone())
                }
            }
        }
    }
    
    /// Update the ε value according to the decay rate.
    fn update(&mut self) {
        // 衰减ε值
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
        }
    }
    
    /// Get a string representation of the current ε value.
    fn get_params(&self) -> String {
        format!("ε={:.4}", self.epsilon)
    }
}

/// Boltzmann policy, commonly used in DQN for exploration.
pub struct Boltzmann {
    /// Current temperature value
    pub temperature: f32,
    /// Minimum temperature value
    pub temperature_min: f32,
    /// Temperature decay rate
    pub temperature_decay: f32,
}

impl Boltzmann {
    /// Create a new Boltzmann policy.
    /// 
    /// # Arguments
    /// * `temperature_start` - Initial temperature value
    /// * `temperature_min` - Minimum temperature value
    /// * `temperature_decay` - Temperature decay rate
    pub fn new(temperature_start: f32, temperature_min: f32, temperature_decay: f32) -> Self {
        Self {
            temperature: temperature_start,
            temperature_min,
            temperature_decay,
        }
    }
}

impl<T> Policy<T> for Boltzmann
where
    T: Copy,
{
    /// Select an action based on the Boltzmann policy.
    /// 
    /// # Arguments
    /// * `q_values` - Q-value distribution for the current state
    fn select_action(&mut self, q_values: &QValue<T>) -> Result<Action<T>> {
        match q_values {
            QValue::Deterministic(action) => {
                // 对于确定性Q值，直接返回对应的动作
                Ok(action.clone())
            },
            QValue::Stochastic(actions_with_values) => {
                // 对于随机Q值，从动作集合中基于softmax概率采样
                let mut rng = rng();
                
                // 提取所有动作的Q值
                let values: Vec<f32> = actions_with_values.iter()
                    .map(|(_, q_val)| *q_val)
                    .collect();
                
                // 计算softmax概率
                let values_tensor = Tensor::new(values.as_slice(), &candle_core::Device::Cpu)?;
                let temperature_tensor = Tensor::new(self.temperature, &candle_core::Device::Cpu)?;
                let scaled_values = values_tensor.div(&temperature_tensor)?;
                let probabilities = ops::softmax(&scaled_values, 0)?;
                
                // 从概率分布中采样动作索引
                let probabilities_vec = probabilities.to_vec1::<f32>()?;
                let sample = rng.random::<f32>();
                let mut cumulative = 0.0;
                
                for (i, &prob) in probabilities_vec.iter().enumerate() {
                    cumulative += prob;
                    if sample < cumulative {
                        // 返回选中的动作
                        return Ok(actions_with_values[i].0.clone());
                    }
                }
                
                // 以防数值精度问题，返回最后一个动作
                Ok(actions_with_values.last().unwrap().0.clone())
            }
        }
    }
    
    /// Update the temperature value according to the decay rate.
    fn update(&mut self) {
        // 衰减温度参数
        if self.temperature > self.temperature_min {
            self.temperature *= self.temperature_decay;
        }
    }
    
    /// Get a string representation of the current temperature value.
    fn get_params(&self) -> String {
        format!("T={:.4}", self.temperature)
    }
}

/// Ornstein-Uhlenbeck process noise, commonly used in DDPG for exploration.
pub struct OrnsteinUhlenbeck {
    /// Mean value
    pub mu: f32,
    /// Theta parameter
    pub theta: f32,
    /// Sigma parameter
    pub sigma: f32,
    /// Action dimension
    pub action_dim: usize,
    /// Current state
    pub state: Option<Vec<f32>>,
}

impl OrnsteinUhlenbeck {
    /// Create a new Ornstein-Uhlenbeck process noise.
    /// 
    /// # Arguments
    /// * `mu` - Mean value
    /// * `theta` - Theta parameter
    /// * `sigma` - Sigma parameter
    /// * `action_dim` - Action dimension
    pub fn new(mu: f32, theta: f32, sigma: f32, action_dim: usize) -> Self {
        Self {
            mu,
            theta,
            sigma,
            action_dim,
            state: None,
        }
    }
    
    fn sample(&mut self) -> Vec<f32> {
        let mut rng = rng();
        
        match &mut self.state {
            Some(state) => {
                for i in 0..self.action_dim {
                    let dx = self.theta * (self.mu - state[i]) + self.sigma * rng.random_range(-1.0..1.0);
                    state[i] += dx;
                }
                state.clone()
            },
            None => {
                // 初始状态
                let state = vec![self.mu; self.action_dim];
                self.state = Some(state.clone());
                state
            }
        }
    }
}

impl<T> Policy<T> for OrnsteinUhlenbeck
where
    T: Copy + From<f32> + std::ops::Add<Output = T>,
{
    /// Select an action based on the Ornstein-Uhlenbeck process noise.
    /// 
    /// # Arguments
    /// * `q_values` - Q-value distribution for the current state
    fn select_action(&mut self, q_values: &QValue<T>) -> Result<Action<T>> {
        match q_values {
            QValue::Deterministic(action) => {
                // 对于确定性Q值，基于动作添加噪声
                let mut action_data = action.value.clone();
                
                // 添加噪声
                let noise = self.sample();
                for i in 0..action_data.len() {
                    action_data[i] = action_data[i] + T::from(noise[i]);
                }
                
                // 返回带噪声的动作，使用相同的上界
                Ok(Action::new(action_data, action.uppers.clone()))
            },
            QValue::Stochastic(_actions_with_values) => {
                // 对于随机Q值，选择最佳动作并添加噪声
                let best_action = q_values.best_action();
                let mut action_data = best_action.value.clone();
                
                // 添加噪声
                let noise = self.sample();
                for i in 0..action_data.len() {
                    action_data[i] = action_data[i] + T::from(noise[i]);
                }
                
                // 返回带噪声的动作，使用相同的上界
                Ok(Action::new(action_data, best_action.uppers.clone()))
            }
        }
    }
    
    fn update(&mut self) {
        // 对于OU过程，不需要特定的更新
    }
    
    fn get_params(&self) -> String {
        format!("μ={:.4}, θ={:.4}, σ={:.4}", self.mu, self.theta, self.sigma)
    }
}

/// Gaussian noise policy for exploration in DDPG.
pub struct GaussianNoise {
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Decay rate for standard deviation
    pub decay_rate: f32,
}

impl GaussianNoise {
    pub fn new(mean: f32, std_dev: f32, decay_rate: f32) -> Self {
        Self {
            mean,
            std_dev,
            decay_rate,
        }
    }
    
    fn sample(&self, size: usize) -> Vec<f32> {
        let mut rng = rng();
        (0..size).map(|_| rng.random_range(-1.0..1.0) * self.std_dev + self.mean).collect()
    }
}

impl<T> Policy<T> for GaussianNoise
where
    T: Copy + From<f32> + std::ops::Add<Output = T> + std::fmt::Display,
{
    fn select_action(&mut self, q_values: &QValue<T>) -> Result<Action<T>> {
        match q_values {
            QValue::Deterministic(action) => {
                // 对于确定性Q值，基于动作添加噪声
                let mut action_data = action.value.clone();
                
                // 添加高斯噪声
                let noise = self.sample(action_data.len());
                for i in 0..action_data.len() {
                    action_data[i] = action_data[i] + T::from(noise[i]);
                }
                
                // 返回带噪声的动作，使用相同的上界
                Ok(Action::new(action_data, action.uppers.clone()))
            },
            QValue::Stochastic(_actions_with_values) => {
                // 对于随机Q值，选择最佳动作并添加噪声
                let best_action = q_values.best_action();
                let mut action_data = best_action.value.clone();
                
                // 添加高斯噪声
                let noise = self.sample(action_data.len());
                for i in 0..action_data.len() {
                    action_data[i] = action_data[i] + T::from(noise[i]);
                }
                
                // 返回带噪声的动作，使用相同的上界
                Ok(Action::new(action_data, best_action.uppers.clone()))
            }
        }
    }
    
    fn update(&mut self) {
        // 衰减标准差
        self.std_dev = self.std_dev * self.decay_rate;
    }
    
    fn get_params(&self) -> String {
        format!("μ={:.4}, σ={:.4}", self.mean, self.std_dev)
    }
}

/// Deterministic policy, directly using the network output as the action.
pub struct DeterministicPolicy;

impl DeterministicPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl<T> Policy<T> for DeterministicPolicy
where
    T: Copy,
{
    fn select_action(&mut self, q_values: &QValue<T>) -> Result<Action<T>> {
        match q_values {
            QValue::Deterministic(action) => {
                // 对于确定性Q值，直接返回对应的动作
                Ok(action.clone())
            },
            QValue::Stochastic(_actions_with_values) => {
                // 对于随机Q值，返回最佳动作
                Ok(q_values.best_action().clone())
            }
        }
    }
    
    fn update(&mut self) {
        // 确定性策略不需要更新参数
    }
    
    fn get_params(&self) -> String {
        "Deterministic".to_string()
    }
}
