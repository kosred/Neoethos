//! The Q-learning algorithm module encapsulates the training and usage methods of the Q-learning algorithm.

use std::any::{Any};
use std::iter::Product;
use rand::Rng;
use indicatif::{ProgressBar, ProgressStyle};
use crate::{Action, EnvTrait, Status, QValue, Sample};
use super::*;
use crate::utils::{dim_product, encode_mixed_radix, decode_mixed_radix};
use crate::replay_buffer::ReplayBuffer;

/// Q-learning algorithm implementation, suitable for discrete action spaces.
#[derive(Debug)]
pub struct QLearning<T: Clone + 'static = u16> {
    /// Q-table
    q_table: Vec<Vec<f32>>,
    action_space: Vec<T>,
    state_space: Vec<T>,

    /// Learning rate
    learning_rate: f32,
    lr_decay: f32,    // Decay factor (e.g., 0.995)
    
    /// Experience replay buffer
    replay_buffer: ReplayBuffer<T, T>,
    /// Experience replay buffer capacity
    buffer_capacity: usize,
    /// Minimum buffer size to start batch updates
    min_buffer_size: usize,
}

impl<T> QLearning<T>
where
    T: Copy + Clone + TryInto<usize> + TryFrom<usize> + Product<T> + 'static,
{
    /// Create a new Q-learning algorithm instance.
    pub fn new<Env>(env: &Env, buffer_capacity: usize) -> Result<Self>
    where
        Env: EnvTrait<T, T>,
    {
        let action_dim = dim_product(env.action_space())?;
        let state_dim = dim_product(env.state_space())?;
        let action_space = env.action_space().to_vec();
        let state_space = env.state_space().to_vec();
        
        // Buffer capacity and minimum batch size settings
        let buffer_capacity = buffer_capacity;
        let min_buffer_size = 64;

        let mut rng = rand::rng();
        let q_table: Vec<Vec<f32>> = (0..state_dim)
            .map(|_| {
                (0..action_dim)
                    .map(|_| rng.random_range(-0.1..0.1))  // Small random values
                    .collect()
            })
            .collect();

        Ok(Self {
            q_table,
            action_space,
            state_space,
            learning_rate: 0.1,
            lr_decay: 0.995,
            replay_buffer: ReplayBuffer::new(buffer_capacity),
            buffer_capacity,
            min_buffer_size,
        })
    }
    
    /// Update the Q-table using a batch of samples from the replay buffer.
    fn update_q_table_batch(&mut self, batch_size: usize, gamma: f32) -> Result<()> {
        // Sample a batch of transitions from the replay buffer
        let samples = self.replay_buffer.sample(batch_size);
        
        for sample in samples {
            let state = sample.state;
            let action = sample.action;
            let reward = sample.reward;
            let next_state = sample.next_state;
            let done = sample.done;
            
            // 将动作转换为索引
            let a_idx = encode_mixed_radix(action.as_slice(), &self.action_space)?;
            
            // 当前 Q(s,a)
            let state_idx = encode_mixed_radix(state.as_slice(), &self.state_space)?;
            let old_q = self.q_table[state_idx][a_idx];
            
            // 计算目标Q值
            let target_q = if done {
                reward.0  // 终止状态没有未来奖励
            } else {
                // max_{a'} Q(s',a')
                let next_state_idx = encode_mixed_radix(next_state.as_slice(), &self.state_space)?;
                let max_next_q = *self.q_table[next_state_idx].iter().max_by(|a, b| a.total_cmp(b)).unwrap_or(&0.0);
                reward.0 + gamma * max_next_q
            };
            
            // Q-Learning 更新
            self.q_table[state_idx][a_idx] = old_q + self.learning_rate * (target_q - old_q);
        }

        Ok(())
    }
    
    fn get_q(&mut self, state: &Status<T>) -> Result<&[f32]> {
        let idx = encode_mixed_radix(state.as_slice(), self.state_space.as_slice())?;
        Ok(&self.q_table[idx])
    }
    
    /// Train the Q-learning agent for one episode.
    fn train_episode<Env: EnvTrait<T, T>>(
        &mut self,
        env: &mut Env,
        policy: &mut dyn Policy<T>,
        args: &TrainArgs,
    ) -> Result<(f32, usize)> {
        let mut state = env.reset();

        let mut total_reward = 0.0;
        let mut steps = 0;

        let uppers = self.action_space.to_vec();

        for _ in 0..args.max_steps {
            steps += 1;

            // 获取当前状态的Q值向量
            let q_vec = self.get_q(&state)?;
            
            // 创建QValue对象用于策略选择
            let q_value = QValue::Stochastic(
                q_vec.iter().enumerate()
                    .map(|(i, q_val)| -> Result<(Action<T>, f32)> {
                        // 将索引转换为动作值
                        let action_value = decode_mixed_radix(i, &uppers)?;
                        Ok((Action::new(action_value, uppers.clone()), *q_val))
                    })
                    .collect::<Result<Vec<_>>>()?
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
            if self.replay_buffer.len() >= self.min_buffer_size && steps % args.update_freq == 0 {
                self.update_q_table_batch(args.batch_size, args.gamma)?;
            }

            total_reward += reward.0;
            state = next_state;
            if done {
                break;
            }
        }
        
        // 更新策略参数
        policy.update();

        // 更新学习率
        self.learning_rate *= self.lr_decay;
        
        Ok((total_reward, steps))
    }
}



impl<Env, T> Algorithm<Env, T, T> for QLearning<T>
where
    Env: EnvTrait<T, T>,
    T: Copy + Clone + Product<T> + TryInto<usize> + TryFrom<usize> + 'static,
{
    /// Train the Q-learning agent for multiple episodes.
    fn train(&mut self, env: &mut Env, policy: &mut dyn Policy<T>, args: TrainArgs) -> Result<()> {
        // 执行多轮训练
        let mut total_reward = 0.0;
        let mut total_steps = 0;

        // 初始化学习率
        self.learning_rate = args.learning_rate as f32;
        self.min_buffer_size = args.batch_size as usize;
        
        // 重置回放缓冲区
        self.replay_buffer = ReplayBuffer::new(self.buffer_capacity);
        
        // 创建进度条
        let pb = ProgressBar::new(args.epochs as u64);
        pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta_precise}) - {msg}")
            .unwrap()
            .progress_chars("=> "));
        
        for episode in 0..args.epochs {
            let (episode_reward, steps) = self.train_episode::<Env>(env, policy, &args)?;
            total_reward += episode_reward;
            total_steps += steps;
            
            let avg_reward = total_reward / (episode as f32 + 1.0);
            let avg_steps = total_steps as f32 / (episode as f32 + 1.0);
            
            // 创建包含所有信息的单一消息字符串，避免临时值引用问题
            let status_msg = format!("平均奖励: {:.2}, 平均步长: {:.2}, 策略参数: {}", 
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

    fn get_action(&self, state: &Status<T>, policy: &mut dyn Policy<T>) -> Result<Action<T>> {
        let uppers = self.action_space.to_vec();

        // 获取当前状态的Q值
        let state_idx = encode_mixed_radix(state.as_slice(), &self.state_space)?;
        let q_vec = if let Some(entry) = self.q_table.get(state_idx) {
            entry
        } else {
            return Err(AlgorithmError::InvalidParameters("状态超出范围".to_string()));
        };
        
        // 创建QValue对象用于策略选择
        let q_value = QValue::Stochastic(
            q_vec.iter().enumerate()
                .map(|(i, q_val)| -> Result<(Action<T>, f32)> {
                    // 将索引转换为动作值
                    let action_value = decode_mixed_radix(i, &uppers)?;
                    Ok((Action::new(action_value, uppers.to_vec()), *q_val))
                })
                .collect::<Result<Vec<_>>>()?
        );
        
        // 使用策略选择动作
        policy.select_action(&q_value)
            .map_err(|e| AlgorithmError::ModelUpdateFailed(e.to_string()))
    }

    fn vars_any(&self) -> Box<dyn Any + Send> {
        Box::new(self.q_table.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
