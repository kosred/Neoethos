//! # RLKit
//! 
//! A multi-algorithm deep reinforcement learning library based on Rust and Candle.
//! 
//! ## Features
//! 
//! ### Core Algorithms
//! - **Q-Learning**: A classic tabular reinforcement learning algorithm
//! - **DQN (Deep Q-Network)**: Q-value function estimation based on deep neural networks
//! 
//! ### Key Features
//! - Supports multiple action selection policies (ε-greedy, Boltzmann, Gaussian noise, etc.)
//! - Implements experience replay buffer to improve sample efficiency
//! - Supports target network updates (DQN) for improved training stability
//! - Generic design supporting different types of state and action spaces
//! - Comprehensive error handling mechanisms
//! 
//! ### Hardware Acceleration
//! - **CUDA Support**: Enable GPU acceleration using the `cuda` feature flag
//! - CPU-only mode available for systems without GPU
//! 
//! ## Core Components
//! 
//! - **Environment Interface**: Defines standard methods for interacting with environments
//! - **Algorithm Implementations**: Contains implementations of various reinforcement learning algorithms
//! - **Policies**: Provides multiple action selection strategies
//! - **Networks**: Neural network structures (used by DQN and other algorithms)
//! - **Replay Buffer**: Stores and samples experiences
//! 
//! ## Getting Started
//! 
//! To use this library, simply implement the `EnvTrait` interface to create a custom environment, then select appropriate algorithms and policies for training.
//! 
//! ## Usage Examples
//! 
//! Here are simple examples of using Q-Learning and DQN algorithms:
//! 
//! ```rust
//! // 1. Import necessary components
//! use rlkit::{
//!     Algorithm,
//!     QLearning,
//!     DQN,
//!     TrainArgs,
//!     EpsilonGreedy,
//!     DeterministicPolicy,
//!     EnvTrait,
//!     Status,
//!     Reward,
//!     Action,
//!     DNQStateMode,
//!     Policy,
//! };
//! use candle_core::Device;
//! 
//! // 2. Create a simple environment example
//! struct SimpleEnv {
//!     state: u16,
//!     max_steps: usize,
//!     current_step: usize,
//! }
//! 
//! impl SimpleEnv {
//!     fn new(max_steps: usize) -> Self {
//!         Self {
//!             state: 0,
//!             max_steps,
//!             current_step: 0,
//!         }
//!     }
//! }
//! 
//! impl EnvTrait<u16, u16> for SimpleEnv {
//!     fn step(&mut self, _state: &Status<u16>, action: &Action<u16>) -> (Status<u16>, Reward, bool) {
//!         // Simple environment logic
//!         self.current_step += 1;
//!         let next_state = (self.state + action.as_slice()[0]) % 10;
//!         self.state = next_state;
//!         
//!         // Simple reward function
//!         let reward = if next_state == 5 {
//!             Reward(10.0)
//!         } else {
//!             Reward(-1.0)
//!         };
//!         
//!         // Check for termination
//!         let done = self.current_step >= self.max_steps || next_state == 5;
//!         
//!         // Use the correct Status::new constructor
//!         (Status::new(vec![next_state], vec![10]), reward, done)
//!     }
//!     
//!     fn reset(&mut self) -> Status<u16> {
//!         self.state = 0;
//!         self.current_step = 0;
//!         Status::new(vec![self.state], vec![10])
//!     }
//!     
//!     fn action_space(&self) -> &[u16] {
//!         &[1]
//!     }
//!     
//!     fn state_space(&self) -> &[u16] {
//!         &[10]
//!     }
//!     
//!     fn as_any(&self) -> &dyn std::any::Any {
//!         self
//!     }
//!     
//!     fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
//!         self
//!     }
//! }
//! 
//! fn main() -> anyhow::Result<()> {
//!     // Create environment
//!     let mut env = SimpleEnv::new(100);
//!     
//!     // Configure training parameters
//!     let train_args = TrainArgs {
//!         epochs: 100,
//!         max_steps: 50,
//!         batch_size: 64,
//!         learning_rate: 0.1,
//!         gamma: 0.99,
//!         update_freq: 1,
//!         update_interval: 10,
//!     };
//!     
//!     // 3. Create Q-Learning agent
//!     let mut q_agent = QLearning::new(&env, 10000)?;
//!     let mut q_policy = EpsilonGreedy::new(1.0, 0.01, 0.995); // Initial exploration rate, minimum exploration rate, decay rate
//!     
//!     // 4. Train Q-Learning agent
//!     println!("Training Q-Learning agent...");
//!     q_agent.train(&mut env, &mut q_policy, train_args)?;
//!     
//!     println!("Training complete!");
//!     
//!     // 5. Use the trained agent for inference (demonstrate get_action usage)
//!     println!("\nTesting the trained agent...");
//!     let mut current_state = env.reset();
//!     let mut total_reward = 0.0;
//!     let mut steps = 0;
//!     
//!     // Reset policy's exploration rate for testing
//!     let mut q_policy = DeterministicPolicy;
//!     
//!     // Run a test episode
//!     while steps < 50 {
//!         // Use get_action method to select action
//!         let action = <rlkit::QLearning as Algorithm<SimpleEnv, u16, u16>>::get_action(&q_agent, &current_state, &mut q_policy)?;
//!         
//!         // Execute action
//!         let (next_state, reward, done) = env.step(&current_state, &action);
//!         
//!         // Update state and cumulative reward
//!         let (next_state, reward, done) = env.step(&current_state, &action);
//!         
//!         // Update state and accumulated reward
//!         current_state = next_state;
//!         total_reward += reward.0 as f64;
//!         steps += 1;
//!         
//!         println!("Step {}: Action = {:?}, Reward = {}, Cumulative Reward = {:.2}", 
//!                  steps, action.as_slice(), reward.0, total_reward);
//!         
//!         if done {
//!             println!("Termination condition reached!");
//!             break;
//!         }
//!     }
//!     
//!     println!("\nTest completed! Total Reward: {:.2}, Total Steps: {}", total_reward, steps);
//!     Ok(())
//! }
//! ```
//! 
//! For more detailed examples, please check the examples directory in the project.
//! 
//! ## Cargo Features
//! 
//! - `cuda`: Enable CUDA GPU acceleration for faster training
//! - Default features: CPU-only mode

pub mod types;
pub mod network;
pub mod replay_buffer;
pub mod policies;
pub mod algs;
pub mod utils;

// Re-export commonly used types and structs
pub use types::{Status, Reward, Sample, EnvTrait, Action, QValue};
pub use algs::{Algorithm, QLearning, DQN, TrainArgs};
pub use policies::{Policy, EpsilonGreedy, Boltzmann, GaussianNoise, DeterministicPolicy};
pub use algs::dqn::DNQStateMode;

/// Version number
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
