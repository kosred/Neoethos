//! The DQN algorithm module encapsulates the training and usage methods of the Deep Q-Learning algorithm.

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::optim::{self, Optimizer};
use candle_nn::{self as nn, AdamW};
use indicatif::{ProgressBar, ProgressStyle};

use super::*;
use crate::network::NeuralNetwork;
use crate::replay_buffer::ReplayBuffer;
use crate::utils::{IntoF32, decode_mixed_radix, dim_product, encode_mixed_radix};
use crate::{Action, EnvTrait, QValue, Sample, Status};
use std::any::Any;
use std::iter::Product;

fn cast_tensor_dtype(tensor: Tensor, dtype: DType) -> Result<Tensor> {
    if tensor.dtype() == dtype {
        Ok(tensor)
    } else {
        Ok(tensor.to_dtype(dtype)?)
    }
}

/// Copy every parameter from `src` into `dst`'s own storage.
///
/// `NeuralNetwork` is `#[derive(Clone)]` over a candle `VarMap`, whose inner
/// `HashMap<String, Var>` sits behind an `Arc` — so `clone()` SHARES the
/// variables. A target network built that way is never frozen: every
/// optimizer step on the online network moves the "target" too, which
/// destroys the whole point of a DQN target network. The target must own
/// independent variables and be synced by copying tensors.
fn copy_network_weights(src: &NeuralNetwork, dst: &NeuralNetwork) -> Result<()> {
    let src_map = src
        .varmap
        .data()
        .lock()
        .map_err(|e| AlgorithmError::InternalError(format!("source varmap poisoned: {e}")))?;
    let dst_map = dst
        .varmap
        .data()
        .lock()
        .map_err(|e| AlgorithmError::InternalError(format!("target varmap poisoned: {e}")))?;
    for (name, src_var) in src_map.iter() {
        let dst_var = dst_map.get(name).ok_or_else(|| {
            AlgorithmError::ModelUpdateFailed(format!(
                "target network is missing parameter '{name}' during weight sync"
            ))
        })?;
        dst_var.set(&src_var.as_tensor().detach())?;
    }
    Ok(())
}

/// DQN state encoding mode.
#[derive(Default, Debug, Eq, PartialEq)]
pub enum DNQStateMode {
    /// One-hot encoding.
    #[default]
    OneHot,
    /// Naive encoding.
    Naive,
    /// Normalized encoding.
    Normalized,
}

/// DQN algorithm implementation for continuous/discrete state spaces and discrete action spaces.
pub struct DQN<S, A>
where
    S: Copy + Clone + 'static,
    A: Copy + Clone + TryInto<usize> + TryFrom<usize> + 'static,
{
    /// Q-network.
    q_network: NeuralNetwork,
    /// Target Q-network.
    target_network: NeuralNetwork,
    /// Optimizer.
    optimizer: AdamW,
    /// Replay buffer.
    replay_buffer: ReplayBuffer<S, A>,
    /// Buffer capacity.
    buffer_capacity: usize,
    /// Minimum buffer size to start batch updates.
    min_buffer_size: usize,
    /// Action space.
    action_space: Vec<A>,
    /// Action dimension.
    action_dim: usize,
    /// State encoding mode.
    state_mode: DNQStateMode,
    /// State space.
    state_space: Vec<S>,
    /// Current step count.
    step_count: usize,
}

impl<S, A> DQN<S, A>
where
    S: IntoF32 + Copy + TryInto<usize> + TryFrom<usize> + Product + 'static,
    A: Copy + Clone + TryInto<usize> + TryFrom<usize> + 'static,
{
    /// Creates a new DQN algorithm instance.
    pub fn new<Env>(
        env: &Env,
        buffer_capacity: usize,
        hidden_dims: &[usize],
        state_mode: DNQStateMode,
        device: &Device,
    ) -> Result<Self>
    where
        Env: EnvTrait<S, A>,
        A: Copy + Clone + Product + TryInto<usize> + 'static,
    {
        Self::new_with_dtype(
            env,
            buffer_capacity,
            hidden_dims,
            state_mode,
            DType::F32,
            device,
        )
    }

    /// Creates a new DQN algorithm instance with an explicit network dtype.
    pub fn new_with_dtype<Env>(
        env: &Env,
        buffer_capacity: usize,
        hidden_dims: &[usize],
        state_mode: DNQStateMode,
        dtype: DType,
        device: &Device,
    ) -> Result<Self>
    where
        Env: EnvTrait<S, A>,
        A: Copy + Clone + Product + TryInto<usize> + 'static,
    {
        let action_space = env.action_space().to_vec();
        let action_dim = dim_product(&action_space)?;
        let state_space = env.state_space().to_vec();

        // 根据状态编码模式计算状态维度
        let state_dim = match state_mode {
            DNQStateMode::OneHot => dim_product(&state_space)?,
            DNQStateMode::Naive | DNQStateMode::Normalized => state_space.len(),
        };

        // 缓冲区容量和最小批量大小设置
        let min_buffer_size = 64;

        // 创建Q网络和目标网络
        let q_network =
            NeuralNetwork::new_with_dtype(state_dim, hidden_dims, action_dim, dtype, device)?;

        // The target network must own INDEPENDENT variables (see
        // `copy_network_weights`); `q_network.clone()` would share them.
        let target_network =
            NeuralNetwork::new_with_dtype(state_dim, hidden_dims, action_dim, dtype, device)?;
        copy_network_weights(&q_network, &target_network)?;

        // 创建优化器
        let optimizer = optim::AdamW::new(q_network.varmap.all_vars(), nn::ParamsAdamW::default())?;

        Ok(Self {
            q_network,
            target_network,
            optimizer,
            replay_buffer: ReplayBuffer::new(buffer_capacity),
            buffer_capacity,
            min_buffer_size,
            action_space,
            action_dim,
            state_mode,
            state_space,
            step_count: 0,
        })
    }

    /// Updates the target network to match the Q-network (hard sync into the
    /// target's OWN variables — never an Arc-sharing clone).
    fn update_target_network(&mut self) -> Result<()> {
        copy_network_weights(&self.q_network, &self.target_network)
    }

    /// Updates the Q-network using a batch of samples from the replay buffer.
    fn update_q_network_batch(&mut self, batch_size: usize, gamma: f32) -> Result<()> {
        let samples = self.replay_buffer.sample(batch_size);
        if samples.len() != batch_size {
            eprintln!(
                "回放缓冲区采样数量异常：期望 {} 个样本，实际 {} 个，缓冲区当前大小：{}",
                batch_size,
                samples.len(),
                self.replay_buffer.len()
            );
            return Ok(());
        }

        // 提取批次数据
        let mut states = Vec::with_capacity(batch_size);
        let mut next_states = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut actions = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);

        let device = &self.q_network.device();

        // 收集样本数据
        for sample in &samples {
            match self.state_mode {
                DNQStateMode::OneHot => {
                    states.push(sample.state.to_one_hot_flat(&self.state_space, device)?);
                    next_states.push(
                        sample
                            .next_state
                            .to_one_hot_flat(&self.state_space, device)?,
                    );
                }
                DNQStateMode::Naive => {
                    states.push(sample.state.to_tensor(device)?);
                    next_states.push(sample.next_state.to_tensor(device)?);
                }
                DNQStateMode::Normalized => {
                    states.push(
                        sample
                            .state
                            .to_tensor_normalized(&self.state_space, device)?,
                    );
                    next_states.push(
                        sample
                            .next_state
                            .to_tensor_normalized(&self.state_space, device)?,
                    );
                }
            }
            rewards.push(sample.reward.0);
            dones.push(sample.done);

            let action_idx = encode_mixed_radix(sample.action.as_slice(), &self.action_space)?;
            actions.push(action_idx as i64);
        }

        // 创建状态批次张量
        let network_dtype = self.q_network.dtype();
        let state_tensor = cast_tensor_dtype(Tensor::stack(&states, 0)?, network_dtype)?;
        let next_state_tensor = cast_tensor_dtype(Tensor::stack(&next_states, 0)?, network_dtype)?;

        // 计算当前状态的Q值。The target-network pass is detached: the TD
        // target is a constant in the DQN loss — gradients must only flow
        // through the online network's Q(s,a).
        let current_q_values = self.q_network.forward(&state_tensor)?;
        let next_q_values = self.target_network.forward(&next_state_tensor)?.detach();

        // 计算最大Q值
        let max_next_q_values = next_q_values.max_keepdim(1)?.squeeze(1)?;

        // 创建奖励张量
        let reward_tensor = cast_tensor_dtype(
            Tensor::from_vec(rewards, batch_size, device)?,
            network_dtype,
        )?;

        // 创建done掩码张量
        let done_mask: Vec<f32> = dones
            .iter()
            .map(|&done| if done { 0.0 } else { 1.0 })
            .collect();
        let done_mask_tensor = cast_tensor_dtype(
            Tensor::from_vec(done_mask, batch_size, device)?,
            network_dtype,
        )?;

        // 计算目标Q值
        let gamma = cast_tensor_dtype(Tensor::new(gamma, device)?, network_dtype)?;
        let target_q_values = reward_tensor.add(
            &max_next_q_values
                .mul(&done_mask_tensor)?
                .broadcast_mul(&gamma)?,
        )?;

        // 准备动作索引张量用于选择对应的Q值
        let action_indices = Tensor::from_vec(actions, batch_size, device)?;

        // 选择当前状态下对应动作的Q值
        let selected_q_values = current_q_values
            .gather(&action_indices.unsqueeze(1)?, 1)?
            .squeeze(1)?;

        // 计算MSE损失
        let loss = selected_q_values.sub(&target_q_values)?.sqr()?.mean_all()?;

        // 反向传播和优化
        self.optimizer.backward_step(&loss)?;

        Ok(())
    }

    /// Get Q-values for a given state.
    fn get_q_values(&self, state: &Status<S>) -> Result<Vec<f32>> {
        let state_tensor = match self.state_mode {
            DNQStateMode::OneHot => {
                state.to_one_hot_flat(&self.state_space, &self.q_network.device())?
            }
            DNQStateMode::Naive => state.to_tensor(&self.q_network.device())?,
            DNQStateMode::Normalized => {
                state.to_tensor_normalized(&self.state_space, &self.q_network.device())?
            }
        }
        .unsqueeze(0)?;
        let state_tensor = cast_tensor_dtype(state_tensor, self.q_network.dtype())?;
        let q_values = self
            .q_network
            .forward(&state_tensor)
            .map_err(|e| AlgorithmError::ModelUpdateFailed(e.to_string()))?;

        // 关键：校验输出形状（单个状态的批次，形状应为 (1, 动作总数)）
        let expected_shape = candle_core::Shape::from_dims(&[1, self.action_dim]);
        if q_values.shape() != &expected_shape {
            return Err(AlgorithmError::ModelUpdateFailed(format!(
                "Q网络输出形状错误！期望 {:?}，实际 {:?}（动作总数: {}）",
                expected_shape,
                q_values.shape(),
                self.action_dim
            )));
        }

        // 转换为Vec<f32>
        let q_values_flat = q_values
            .squeeze(0)? // 移除批次维度，得到 (action_dim,)
            .to_vec1::<f32>()?; // 直接转 Vec<f32>（Candle 1.0+ 支持）
        Ok(q_values_flat)
    }

    /// Train the DQN agent for a single episode.
    fn train_episode<Env: EnvTrait<S, A>>(
        &mut self,
        env: &mut Env,
        policy: &mut dyn Policy<A>,
        args: &TrainArgs,
    ) -> Result<(f32, usize)>
    where
        S: std::fmt::Debug + IntoF32 + Copy,
    {
        let mut state = env.reset();

        let mut total_reward = 0.0;
        let mut steps = 0;

        for _ in 0..args.max_steps {
            steps += 1;
            self.step_count += 1;

            // 获取当前状态的Q值
            let q_values = self.get_q_values(&state)?;
            // 调试
            if self.step_count % 10000 == 0 {
                println!(
                    "当前步数: {}, 状态: {:?}, Q值: {:?}",
                    self.step_count, state.value, q_values
                );
            }

            // 创建QValue对象用于策略选择
            let q_value = QValue::Stochastic(
                q_values
                    .iter()
                    .enumerate()
                    .map(|(i, q_val)| -> Result<(Action<A>, f32)> {
                        let action_value = decode_mixed_radix(i, env.action_space())?;
                        let action_uppers = env.action_space().to_vec();
                        Ok((Action::new(action_value, action_uppers), *q_val))
                    })
                    .collect::<Result<Vec<_>>>()?,
            );

            // 使用策略系统选择动作
            let action = policy.select_action(&q_value)?;

            // 执行动作
            let (next_state, reward, done) = env.step(&state, &action);

            // 将样本添加到经验回放缓冲区
            let sample = Sample {
                state: state.clone(),
                action: action.clone(),
                reward: reward.clone(),
                next_state: next_state.clone(),
                done,
            };
            self.replay_buffer.push(sample);

            // 当缓冲区足够大时，进行批量更新
            if (self.replay_buffer.len() >= self.min_buffer_size)
                && (self.step_count % args.update_freq == 0)
            {
                if let Err(e) = self.update_q_network_batch(args.batch_size, args.gamma) {
                    eprintln!("更新失败: {:?}，继续训练...", e);
                }
            }

            if self.step_count % args.update_interval == 0 {
                self.update_target_network()?;
            }

            total_reward += reward.0;
            state = next_state;
            if done {
                break;
            }
        }

        // 更新策略参数
        policy.update();

        Ok((total_reward, steps))
    }
}

impl<Env, S, A> Algorithm<Env, S, A> for DQN<S, A>
where
    Env: EnvTrait<S, A>,
    S: std::fmt::Debug
        + IntoF32
        + Copy
        + TryInto<usize>
        + TryFrom<usize>
        + Product
        + 'static + 'static,
    A: Copy + Clone + TryInto<usize> + TryFrom<usize> + 'static,
{
    fn train(&mut self, env: &mut Env, policy: &mut dyn Policy<A>, args: TrainArgs) -> Result<()> {
        // 执行多轮训练
        let mut total_reward = 0.0;
        let mut total_steps = 0;

        // 更新学习率
        let adam_params = nn::ParamsAdamW {
            lr: args.learning_rate as f64,
            ..Default::default()
        };
        self.optimizer.set_params(adam_params);
        self.min_buffer_size = args.batch_size;

        // 重置回放缓冲区
        self.replay_buffer = ReplayBuffer::new(self.buffer_capacity);
        self.step_count = 0;

        // 初始化目标网络
        self.update_target_network()?;

        // 创建进度条
        let pb = ProgressBar::new(args.epochs as u64);
        pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta_precise}) - {msg}")
            .unwrap()
            .progress_chars(">= "));

        for episode in 0..args.epochs {
            let (episode_reward, steps) = self.train_episode::<Env>(env, policy, &args)?;
            total_reward += episode_reward;
            total_steps += steps;

            let avg_reward = total_reward / (episode as f32 + 1.0);
            let avg_steps = total_steps as f32 / (episode as f32 + 1.0);

            // 创建包含所有信息的单一消息字符串
            let status_msg = format!(
                "平均奖励: {:.2}, 平均步长: {:.2}, 策略参数: {}",
                avg_reward,
                avg_steps,
                policy.get_params()
            );

            // 设置进度条消息
            pb.set_message(status_msg);
            pb.inc(1);
        }

        pb.finish_with_message("训练完成");

        Ok(())
    }

    fn get_action(&self, state: &Status<S>, policy: &mut dyn Policy<A>) -> Result<Action<A>> {
        // 获取当前状态的Q值
        let q_values = self.get_q_values(state)?;

        // 创建QValue对象用于策略选择
        let q_value = QValue::Stochastic(
            q_values
                .iter()
                .enumerate()
                .map(|(i, q_val)| -> Result<(Action<A>, f32)> {
                    let action_value = decode_mixed_radix(i, &self.action_space)?;
                    let action_uppers = self.action_space.clone();
                    Ok((Action::new(action_value, action_uppers), *q_val))
                })
                .collect::<Result<Vec<_>>>()?,
        );

        // 使用策略选择动作
        policy
            .select_action(&q_value)
            .map_err(|e| AlgorithmError::ModelUpdateFailed(e.to_string()))
    }

    fn vars_any(&self) -> Box<dyn Any + Send> {
        // 返回Q网络参数的克隆
        Box::new(self.q_network.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
