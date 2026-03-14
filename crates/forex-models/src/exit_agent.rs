// Exit Agent - Pure Rust Burn RL-Based Trade Exit Expert
// Ported from src/forex_bot/models/exit_agent.py
//
// Lightweight RL Network for Trade Exit Decisions.
// Learns to balance Greed (Holding for TP) vs Fear (Cutting Loss/Stall).

#![cfg(feature = "burn-backend")]

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use anyhow::{Result, anyhow};
use burn::nn;
use burn::prelude::*;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn_ndarray::NdArray;
use burn::backend::Autodiff;

use rand::Rng;
use tracing::info;

pub type TrainBackend = Autodiff<NdArray>;

// ============================================================================
// BURN Q-NETWORK
// ============================================================================

#[derive(Module, Debug)]
pub struct ExitAgentNet<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    output: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct ExitAgentNetConfig {
    #[config(default = 6)]
    pub input_dim: usize,
    #[config(default = 64)]
    pub hidden_dim: usize,
}

impl ExitAgentNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ExitAgentNet<B> {
        ExitAgentNet {
            fc1: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            output: nn::LinearConfig::new(self.hidden_dim, 2).init(device),
        }
    }
}

impl<B: Backend> ExitAgentNet<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = burn::tensor::activation::relu(self.fc1.forward(x));
        let x = burn::tensor::activation::relu(self.fc2.forward(x));
        self.output.forward(x)
    }
}

// ============================================================================
// AGENT STATE & EXPERIENCE
// ============================================================================

#[derive(Clone, Debug)]
pub struct Experience {
    pub state: Vec<f32>,
    pub action: i64,
    pub reward: f32,
    pub done: bool,
}

#[derive(Clone, Debug)]
pub struct PendingRegret {
    pub state: Vec<f32>,
    pub action: i64,
    pub exit_price: f64,
    pub time: i64,
    pub direction: i32,
}

/// Pure-Rust Burn ExitAgent
pub struct ExitAgent {
    model: ExitAgentNet<TrainBackend>,
    optim: burn::optim::AdamW<TrainBackend>,
    memory: VecDeque<Experience>,
    pending_regret: HashMap<i32, PendingRegret>,
    gamma: f32,
    epsilon: f32,
    device: <TrainBackend as Backend>::Device,
}

impl ExitAgent {
    pub fn new() -> Self {
        let device = <TrainBackend as Backend>::Device::default();
        let model = ExitAgentNetConfig::new().init(&device);
        let optim = AdamWConfig::new().with_weight_decay(1e-4).init();
        
        Self {
            model,
            optim,
            memory: VecDeque::with_capacity(10000),
            pending_regret: HashMap::new(),
            gamma: 0.99,
            epsilon: 0.2,
            device,
        }
    }

    /// Returns 0 (Hold) or 1 (Close).
    pub fn get_action(&self, state: &[f32], eval_mode: bool) -> i32 {
        let mut rng = rand::rng();
        if !eval_mode && rng.random::<f32>() < self.epsilon {
            return rng.random_range(0..=1);
        }

        // Forward pass
        let state_tensor = Tensor::<TrainBackend, 1>::from_data(
            TensorData::new(state.to_vec(), [state.len()]), &self.device
        ).unsqueeze::<2>();
        
        let logits = self.model.forward(state_tensor);
        let action = logits.argmax(1).into_data().to_vec::<i64>().unwrap_or(vec![0])[0];
        action as i32
    }

    pub fn observe_exit(
        &mut self,
        ticket: i32,
        state: &[f32],
        action: i32,
        current_price: f64,
        timestamp: i64,
    ) {
        // Infer direction from the sign of the first state element (PnL typically)
        let direction = if !state.is_empty() && state[0] > 0.0 { 1 } else { -1 };
        
        self.pending_regret.insert(ticket, PendingRegret {
            state: state.to_vec(),
            action: action as i64,
            exit_price: current_price,
            time: timestamp,
            direction,
        });
    }

    pub fn process_regret(
        &mut self,
        ticket: i32,
        future_price_trace: &[f64],
        direction: i32,
    ) {
        if let Some(data) = self.pending_regret.remove(&ticket) {
            let exit_price = data.exit_price;
            let action = data.action;

            if future_price_trace.is_empty() {
                return;
            }

            let mut min_future_price = f64::MAX;
            let mut max_future_price = f64::MIN;
            for &p in future_price_trace {
                if p < min_future_price { min_future_price = p; }
                if p > max_future_price { max_future_price = p; }
            }

            let (potential_gain, potential_loss) = if direction == 1 {
                (max_future_price - exit_price, exit_price - min_future_price)
            } else {
                (exit_price - min_future_price, max_future_price - exit_price)
            };

            let mut reward = 0.0f32;
            if action == 1 {
                if potential_gain > (potential_loss * 1.5) {
                    reward = -1.0;
                } else if potential_loss > (potential_gain * 1.5) {
                    reward = 1.0;
                }
            } else {
                if potential_gain > (potential_loss * 1.5) {
                    reward = 1.0;
                } else if potential_loss > (potential_gain * 1.5) {
                    reward = -1.0;
                } else {
                    reward = 0.1;
                }
            }

            if self.memory.len() >= 10000 {
                self.memory.pop_front();
            }
            self.memory.push_back(Experience {
                state: data.state,
                action,
                reward,
                done: true,
            });
        }
    }

    pub fn train_step(&mut self) {
        if self.memory.len() < 32 { return; }

        let mut rng = rand::rng();
        let mut batch_indices: Vec<usize> = (0..self.memory.len()).collect();
        // Naive shuffling strategy for sampling
        use rand::seq::SliceRandom;
        batch_indices.shuffle(&mut rng);
        let batch_indices = &batch_indices[0..32];

        let mut states_flat = Vec::with_capacity(32 * 6);
        let mut actions = Vec::with_capacity(32);
        let mut rewards = Vec::with_capacity(32);

        for &idx in batch_indices {
            let exp = &self.memory[idx];
            states_flat.extend_from_slice(&exp.state);
            actions.push(exp.action);
            rewards.push(exp.reward);
        }

        let states_tensor: Tensor<TrainBackend, 2> = Tensor::from_data(
            TensorData::new(states_flat, [32, 6]), &self.device
        );
        let actions_tensor: Tensor<TrainBackend, 1, Int> = Tensor::from_data(
            TensorData::new(actions, [32]), &self.device
        );
        let rewards_tensor: Tensor<TrainBackend, 1> = Tensor::from_data(
            TensorData::new(rewards, [32]), &self.device
        );

        let q_values = self.model.forward(states_tensor);
        let q_value = q_values.gather(1, actions_tensor.unsqueeze_dim(1)).squeeze(1);

        let loss = burn::nn::loss::MseLoss::new().forward(
            q_value, 
            rewards_tensor, 
            burn::nn::loss::Reduction::Mean
        );

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optim.step(1e-4, self.model.clone(), grads_params);

        self.epsilon = 0.05f32.max(self.epsilon * 0.999);
    }

    pub fn get_epsilon(&self) -> f32 { self.epsilon }
    pub fn set_epsilon(&mut self, e: f32) { self.epsilon = e; }
    pub fn memory_size(&self) -> usize { self.memory.len() }
}
