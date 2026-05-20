//! 强化学习模型模块

use crate::algs::AlgorithmError;

use super::types::{Status, Action, Sample, EnvTrait};
use super::network::NeuralNetwork;
use super::replay_buffer::ReplayBuffer;
use super::policies::{PolicyConfig, Policy};
use super::algs::{Algorithm, AlgorithmType, QLearning};
use candle_core::{Device, Tensor, DType, Module};
use candle_nn::{self as nn, optim, Optimizer, ops};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use rand::Rng;
use indicatif::{ProgressBar, ProgressStyle};

type Result<T> = std::result::Result<T, ModelError>;

#[derive(Debug)]
pub enum ModelError {
    /// 策略创建失败
    PolicyCreationFailed,
    /// 算法创建失败
    AlgorithmCreationFailed,
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::PolicyCreationFailed => write!(f, "Policy creation failed"),
            ModelError::AlgorithmCreationFailed => write!(f, "Algorithm creation failed"),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<AlgorithmError> for ModelError {
    fn from(err: AlgorithmError) -> Self {
        match err {
            AlgorithmError::ModelUpdateFailed(_) => ModelError::AlgorithmCreationFailed,
            AlgorithmError::InvalidParameters(_) => ModelError::AlgorithmCreationFailed,
            AlgorithmError::DeviceError(_) => ModelError::AlgorithmCreationFailed,
            AlgorithmError::InsufficientSamples(_) => ModelError::AlgorithmCreationFailed,
            AlgorithmError::InternalError(_) => ModelError::AlgorithmCreationFailed,
        }
    }
}

/// 强化学习模型模块
pub struct RLModel<Env, S, A>
where
    Env: EnvTrait<S, A>,
    S: Clone + 'static,
    A: Clone + 'static
{
    /// 策略
    policy: Box<dyn Policy>,
    /// 算法
    algorithm: Box<dyn Algorithm<Env, S, A>>,
}

impl<Env, S, A> RLModel<Env, S, A>
where
    Env: EnvTrait<S, A>,
    S: Clone + 'static,
    A: Clone + 'static
{
    /// 创建一个新的DQN模块
    pub fn new(policy_config: PolicyConfig, algorithm: Box<dyn Algorithm<Env, S, A>>) -> Result<Self> {
        
        // // 创建优化器
        // let optimizer = optim::AdamW::new_lr(
        //     q_network.varmap.all_vars(),
        //     config.learning_rate,
        // )?;
        
        // 创建策略
        let policy = policy_config.create_policy(Env::action_space().len()).unwrap();
        
        Ok(Self {
            policy,
            algorithm,
        })
    }
    
    // /// 保存模型
    // pub fn save(&self, path: &str) -> Result<()> {
    //     self.q_network.save(path)
    // }
    
    // /// 加载模型
    // pub fn load(config: RLConfig, path: &str) -> Result<Self> {
    //     let q_network = NeuralNetwork::load(
    //         path,
    //         config.state_dim,
    //         &config.hidden_dims,
    //         config.action_dim,
    //         &config.device,
    //     )?;
        
    //     let target_network = q_network.clone();
        
    //     // let optimizer = optim::AdamW::new_lr(
    //     //     q_network.varmap.all_vars(),
    //     //     config.learning_rate,
    //     // )?;
        
    //     let replay_buffer = Arc::new(Mutex::new(ReplayBuffer::<A>::new(
    //         config.replay_buffer_capacity,
    //     )));
        
    //     // 创建策略
    //     let policy = PolicyConfig::default_gaussian_noise()
    //         .create_policy(q_network.action_dim())?;
        
    //     Ok(Self {
    //         q_network,
    //         target_network,
    //         replay_buffer,
    //         step_count: 0,
    //         config,
    //         policy,
    //     })
    // }
    
    /// 根据当前状态获取动作
    pub fn get_action(&mut self, state: &Status<S>) -> Result<Action<A>> {
        self.algorithm.get_action(state, &mut self.policy).map_err(Into::into)
    }
    
}
