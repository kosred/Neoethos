//! Experience replay buffer, used for storing and sampling training samples.

use super::types::{Sample};
use rand::Rng;
use std::collections::VecDeque;

/// Experience replay buffer, used for storing and sampling training samples.
#[derive(Debug)]
pub struct ReplayBuffer<S = u16, A = u16>
where
    S: Clone + 'static,
    A: Clone + 'static,
{
    buffer: VecDeque<Sample<S, A>>,
    capacity: usize,
}

impl<S, A> ReplayBuffer<S, A>
where
    S: Clone + 'static,
    A: Clone + 'static,
{
    /// Create a new experience replay buffer.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
    
    /// Add a sample to the buffer.
    pub fn push(&mut self, sample: Sample<S, A>) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(sample);
    }
    
    /// Sample a batch of samples from the buffer.
    pub fn sample(&self, batch_size: usize) -> Vec<Sample<S, A>> {
        let mut rng = rand::rng();
        let mut samples = Vec::with_capacity(batch_size);
        let len = self.buffer.len();
        
        if len == 0 {
            return samples;
        }
        
        for _ in 0..batch_size {
            let idx = rng.random_range(0..len);
            samples.push(self.buffer[idx].clone());
        }
        
        samples
    }
    
    /// Get the current size of the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}
