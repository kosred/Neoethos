//! 强化学习算法模块，封装了强化学习算法的训练和使用方法

pub mod q_learning;
pub mod dqn;

pub use q_learning::QLearning;
pub use dqn::DQN;

use crate::policies::Policy;
use crate::{Action, EnvTrait, Status};
use std::any::Any;

pub type Result<T> = std::result::Result<T, AlgorithmError>;

/// 算法错误
#[derive(Debug, PartialEq)]
pub enum AlgorithmError {
    /// 模型更新失败
    ModelUpdateFailed(String),
    /// 参数无效
    InvalidParameters(String),
    /// 设备错误
    DeviceError(String),
    /// 样本不足
    InsufficientSamples(String),
    /// 内部错误
    InternalError(String),
}

impl std::fmt::Display for AlgorithmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgorithmError::ModelUpdateFailed(msg) => write!(f, "模型更新失败: {}", msg),
            AlgorithmError::InvalidParameters(msg) => write!(f, "参数无效: {}", msg),
            AlgorithmError::DeviceError(msg) => write!(f, "设备错误: {}", msg),
            AlgorithmError::InsufficientSamples(msg) => write!(f, "样本不足: {}", msg),
            AlgorithmError::InternalError(msg) => write!(f, "内部错误: {}", msg),
        }
    }
}

impl std::error::Error for AlgorithmError {}

impl From<candle_core::Error> for AlgorithmError {
    fn from(e: candle_core::Error) -> Self {
        Self::ModelUpdateFailed(e.to_string())
    }
}

impl From<AlgorithmError> for candle_core::Error {
    fn from(e: AlgorithmError) -> Self {
        Self::Msg(e.to_string())
    }
}

pub trait Algorithm<Env, S, A>
where
    Env: EnvTrait<S, A>,
    S: Clone + 'static,
    A: Clone + 'static
{

    /// 训练算法
    fn train(&mut self, env: &mut Env, policy: &mut dyn Policy<A>, args: TrainArgs) -> Result<()>;

    /// 获取动作
    fn get_action(&self, state: &Status<S>, policy: &mut dyn Policy<A>) -> Result<Action<A>>;

    /// 获取训练变量
    fn vars_any(&self) -> Box<dyn Any + Send>;

   /// 将算法转换为 Any 类型，用于动态 dispatch
    fn as_any(&self) -> &dyn Any;
}


/// 训练参数
#[derive(Copy, Clone, Debug)]
pub struct TrainArgs {
    /// 训练轮数
    pub epochs: usize,
    /// 每个轮次的最大步数
    pub max_steps: usize,
    /// 每次更新目标网络的样本数量
    pub update_interval: usize,
    /// 每次更新网络学习的次数
    pub update_freq: usize,
    /// 批次大小
    pub batch_size: usize,
    /// 学习率
    pub learning_rate: f64,
    /// 折扣因子
    pub gamma: f32,
}

impl Default for TrainArgs {
    fn default() -> Self {
        Self {
            epochs: 1000,
            max_steps: 200,
            update_interval: 100,
            update_freq: 10,
            batch_size: 64,
            learning_rate: 1e-3,
            gamma: 0.99,
        }
    }   
}

pub enum AlgorithmType {
    QLearning,
    DQN,
}
