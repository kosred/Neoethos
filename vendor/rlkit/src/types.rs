//! Define the core types and environment interfaces in reinforcement learning.

use std::fmt::Debug;
use crate::{algs::AlgorithmError, utils::{dim_product, encode_mixed_radix, IntoF32}};

use super::algs::Result;
use candle_core::{Device, Tensor};

/// Q-value in reinforcement learning, categorized into deterministic and stochastic policies.
#[derive(Debug, Clone)]
pub enum QValue<T = u16> {
    /// Deterministic policy - single action
    Deterministic(Action<T>),
    /// Stochastic policy - vector of actions with corresponding Q-values
    Stochastic(Vec<(Action<T>, f32)>),
}

impl<T> QValue<T> {
    /// Get the best action based on the highest Q-value.
    /// 
    /// Will be changed to Result later
    pub fn try_best_action(&self) -> Option<&Action<T>> {
        match self {
            QValue::Deterministic(action) => Some(action),
            QValue::Stochastic(actions_with_values) => {
                if actions_with_values.is_empty() {
                    None
                } else {
                    actions_with_values.iter()
                        .max_by(|(_, val1), (_, val2)| val1.total_cmp(val2))
                        .map(|(action, _)| action)
                }
            },
        }
    }
    
    /// Get the best action based on the highest Q-value, panicking if no action is available.
    pub fn best_action(&self) -> &Action<T> {
        self.try_best_action().expect("No action available in QValue")
    }
}

/// Action in reinforcement learning, categorized into deterministic and stochastic policies.
#[derive(Debug, Clone)]
pub struct Action<T = u16> {
    pub(crate) value: Vec<T>,
    pub(crate) uppers: Vec<T>,
}

impl<T> Action<T> {
    /// Create a new generic action.
    pub fn new(value: Vec<T>, uppers: Vec<T>) -> Self {
        Self { value, uppers }
    }
    
    /// Get a reference to the internal vector of the action.
    pub fn as_slice(&self) -> &[T] {
        &self.value
    }
    
    /// Get a mutable reference to the internal vector of the action.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.value
    }
    
    /// Get the dimension of the action.
    pub fn dim(&self) -> usize {
        self.value.len()
    }
    
    /// Get the upper bounds of the action space for each dimension.
    pub fn upper_bound(&self) -> &[T] {
        &self.uppers
    }
}

impl<T> Action<T>
where
    T: Copy + rand::distr::uniform::SampleUniform + Default + std::cmp::PartialOrd
{
    /// Sample a random action from the action space in each dimension.
    pub fn random(&self, rng: &mut impl rand::Rng) -> Self {
        let mut value = self.value.clone();
        for (v, &up) in value.iter_mut().zip(&self.uppers) {
            *v = rng.random_range(T::default()..up);
        }
        Self::new(value, self.uppers.clone())
    }
}

/// State in reinforcement learning, with a vector of values and upper bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct Status<T = f32> {
    pub(crate) value: Vec<T>,
    pub(crate) uppers: Vec<T>,
}

impl<T: Clone + Eq + std::hash::Hash> std::hash::Hash for Status<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<T> Status<T> {
    /// Create a new status with a vector of values and upper bounds.
    pub fn new(values: Vec<T>, uppers: Vec<T>) -> Self {
        Self { value: values, uppers }
    }
    
    /// Get a reference to the internal vector of the status.
    pub fn as_slice(&self) -> &[T] {
        &self.value
    }
    
    /// Get a mutable reference to the internal vector of the status.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.value
    }
    
    /// Get a clone of the internal vector of the status.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.value.clone()
    }
    
    /// Get the dimension of the status.
    pub fn len(&self) -> usize {
        self.value.len()
    }
    
    /// Check if the status is empty.
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }
}

/// Reward in reinforcement learning.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Reward(pub f32);

/// Experience sample, containing state, action, reward, and next state.
#[derive(Clone, Debug)]
pub struct Sample<S: Clone = u16, A: Clone = u16> {
    pub state: Status<S>,
    pub action: Action<A>,
    pub reward: Reward,
    pub next_state: Status<S>,
    pub done: bool,
}

impl<T: Clone + IntoF32> Action<T> {
    /// Convert the action to a Tensor. 
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let values: Vec<f32> = self.as_slice()
            .iter()
            .map(|a| a.clone().into_f32())
            .collect::<Result<Vec<f32>>>()?;
        let len = values.len();
        Ok(Tensor::from_vec(values, (1, len), device)?)
    }
}

impl<T> Action<T> {
    /// Convert an action of type A to an action of type T.
    pub fn from_actions<A>(action: Action<A>) -> Result<Self>
    where
        A: TryInto<T> + Clone,
    {
        Ok(Self::new(
            action.value.into_iter()
                .map(|v| v.try_into().map_err(|_| AlgorithmError::InvalidParameters("动作转换为T失败".to_string())))
                .collect::<Result<Vec<T>>>()?,
            action.uppers.into_iter()
                .map(|v| v.try_into().map_err(|_| AlgorithmError::InvalidParameters("上界转换为T失败".to_string())))
                .collect::<Result<Vec<T>>>()?,
        ))
    }
}

impl<T: Copy + IntoF32> Status<T> {
    /// Convert the status to a Tensor.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let values: Vec<f32> = self.as_slice()
            .iter()
            .map(|&s| s.into_f32())
            .collect::<Result<Vec<f32>>>()?;
        let len = values.len();
        Ok(Tensor::from_vec(values, len, device)?)
    }
    
    /// Convert the status to a normalized Tensor.
    /// 
    /// # Arguments
    /// 
    /// * `uppers` - A slice of upper bounds for each dimension of the status.
    /// * `device` - The device to create the Tensor on.
    /// 
    /// # Returns
    /// 
    /// A normalized Tensor of the status.
    pub fn to_tensor_normalized(&self, uppers: &[T], device: &Device) -> Result<Tensor>
    {
        let values: Vec<f32> = self.as_slice()
            .iter()
            .zip(uppers)
            .map(|(&s, &up)| {
                let s_f32 = s.into_f32().map_err(|_| AlgorithmError::InvalidParameters("状态值转换为f32失败".to_string()))?;
                let up_f32 = up.into_f32().map_err(|_| AlgorithmError::InvalidParameters("上界值转换为f32失败".to_string()))?;
                if up_f32 == 0.0 {
                    return Err(AlgorithmError::InvalidParameters("上界值不能为0，无法进行归一化".to_string()));
                }
                Ok(s_f32 / up_f32)
            })
            .collect::<Result<Vec<f32>>>()?;
        let len = values.len();
        Ok(Tensor::from_vec(values, len, device)?)
    }

    /// Convert the status to a one-hot encoded Tensor.
    /// 
    /// # Arguments
    /// 
    /// * `uppers` - A slice of upper bounds for each dimension of the status.
    /// * `device` - The device to create the Tensor on.
    /// 
    /// # Returns
    /// 
    /// A one-hot encoded Tensor of the status.
    pub fn to_one_hot_flat(&self, uppers: &[T], device: &Device) -> Result<Tensor>
    where
        T: Copy + TryInto<usize> + TryFrom<usize> + std::iter::Product,
    {
        let dim = dim_product(uppers)?;
        let index = encode_mixed_radix(&self.value, uppers)?;
        let one_hot = Tensor::new(
            (0..dim).map(|i| if i == index { 1.0 } else { 0.0 }).collect::<Vec<f32>>(),
            device,
        )?;
        Ok(one_hot)
    }
}

impl<T> Status<T> {
    /// Convert a status of type S to a status of type T.
    pub fn from_status<S>(status: Status<S>) -> Result<Self>
    where
        S: TryInto<T> + Clone,
    {
        Ok(Self::new(
            status.value.into_iter()
                .map(|v| v.try_into().map_err(|_| AlgorithmError::InvalidParameters("状态转换为T失败".to_string())))
                .collect::<Result<Vec<T>>>()?,
            status.uppers.into_iter()
                .map(|v| v.try_into().map_err(|_| AlgorithmError::InvalidParameters("上界转换为T失败".to_string())))
                .collect::<Result<Vec<T>>>()?,
        ))
    }
}

impl Reward {
    /// Convert the reward to a Tensor.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        Ok(Tensor::from_slice(&[self.0], (1, 1), device)?)
    }
}

/// The environment interface, defining the methods for interacting with the environment.
/// 
/// # Type Parameters
/// 
/// * `S` - The type of the state space. Defaults to `u16`.
/// * `A` - The type of the action space. Defaults to `u16`.
pub trait EnvTrait<S: Clone = u16, A: Clone = u16> {
    /// Execute the action and return the next state and reward.
    fn step(&mut self, state: &Status<S>, action: &Action<A>) -> (Status<S>, Reward, bool);
    
    /// Reset the environment and return the initial state.
    fn reset(&mut self) -> Status<S>;
    
    /// Get the number of possible actions in the environment (for discrete action spaces).
    fn action_space(&self) -> &[A];
    
    /// Get the dimensions of the state space.
    fn state_space(&self) -> &[S];

    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}
