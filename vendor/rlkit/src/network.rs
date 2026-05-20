//! Network implemented using the Candle library.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn as nn;
use candle_nn::{VarBuilder, VarMap};
use std::collections::HashMap;

/// Network structure implemented using the Candle library.
#[derive(Clone)]
pub struct NeuralNetwork {
    /// Variable map for storing network parameters.
    pub varmap: VarMap,
    /// Vector of linear layers in the network.
    layers: Vec<nn::Linear>,
    /// Computing device (CPU or GPU) used to create and execute network operations.
    device: Device,
    /// Dimension of the input layer, corresponding to the dimension of the state space.
    input_dim: usize,
    /// Vector of hidden layer dimensions, defining the depth and width of the network.
    hidden_dims: Vec<usize>,
    /// Dimension of the output layer, corresponding to the size of the action space.
    output_dim: usize,
    /// Floating-point dtype used by the network parameters.
    dtype: DType,
}

impl NeuralNetwork {
    /// Creates a new Neural Network instance.
    ///
    /// This method creates a fully connected neural network for computing the state-action value function (Q-values). The network structure includes:
    /// - An input layer with dimension `input_dim`.
    /// - An arbitrary number of hidden layers with dimensions specified by `hidden_dims`.
    /// - An output layer with dimension `output_dim`, representing the Q-values for each possible action.
    ///
    /// # Arguments
    /// - `input_dim`: Dimension of the input layer, corresponding to the dimension of the state space.
    /// - `hidden_dims`: An array of hidden layer dimensions, defining the depth and width of the network.
    /// - `output_dim`: Dimension of the output layer, corresponding to the size of the action space.
    /// - `device`: The computing device (CPU or GPU) used to create and execute network operations.
    ///
    /// # Returns
    /// - `Result<Self>`: Returns a configured QNetwork instance if creation is successful, otherwise returns an error message.
    ///
    /// # Algorithm Details
    /// The network uses the following architecture:
    /// 1. Initialize weight and bias parameters in VarMap.
    /// 2. Create fully connected layers equal to the number of hidden layers.
    /// 3. Use ReLU activation function for all layers except the output layer.
    /// 4. The output layer does not use an activation function and directly outputs the raw Q-values.
    ///
    /// # Example
    /// ```
    /// use candle_core::{Device, Result};
    /// use rlkit::network::NeuralNetwork;
    ///
    /// fn create_q_network() -> Result<()> {
    ///     // Create a Q-network with state dimension 4, action dimension 2, and hidden layers [64, 64].
    ///     let device = Device::Cpu;
    ///     let q_network = NeuralNetwork::new(4, &[64, 64], 2, &device)?;
    ///     
    ///     // The network is created successfully and can be used in the DQN algorithm.
    ///     println!("Q-network created successfully. State dimension: {}, Action dimension: {}",
    ///              q_network.state_dim(), q_network.action_dim());
    ///     
    ///     Ok(())
    /// }
    /// ```
    pub fn new(
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_dtype(input_dim, hidden_dims, output_dim, DType::F32, device)
    }

    /// Creates a new Neural Network instance with an explicit parameter dtype.
    pub fn new_with_dtype(
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        Self::new_with_varbuilder(
            varmap,
            input_dim,
            hidden_dims,
            output_dim,
            device,
            dtype,
            vb,
        )
    }

    /// 使用VarBuilder创建Q网络
    pub fn new_with_varbuilder(
        varmap: VarMap,
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        device: &Device,
        dtype: DType,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        let mut in_dim = input_dim;

        // 创建隐藏层
        for (i, &hidden_dim) in hidden_dims.iter().enumerate() {
            let layer = nn::linear(in_dim, hidden_dim, vb.pp(&format!("layer_{}", i)))?;
            layers.push(layer);
            in_dim = hidden_dim;
        }

        // 创建输出层
        let output_layer = nn::linear(in_dim, output_dim, vb.pp("output_layer"))?;
        layers.push(output_layer);

        Ok(Self {
            varmap,
            layers,
            device: device.clone(),
            input_dim,
            hidden_dims: hidden_dims.to_vec(),
            output_dim,
            dtype,
        })
    }

    /// Load the parameter of the network from a file.
    pub fn load(
        path: &str,
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::load_with_dtype(path, input_dim, hidden_dims, output_dim, DType::F32, device)
    }

    /// Load the parameter of the network from a file using an explicit dtype.
    pub fn load_with_dtype(
        path: &str,
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        let mut qnetwork = Self::new_with_varbuilder(
            varmap,
            input_dim,
            hidden_dims,
            output_dim,
            device,
            dtype,
            vb,
        )?;
        qnetwork.varmap.load(path)?;

        Ok(qnetwork)
    }
}

impl NeuralNetwork {
    /// Get all parameters of the network.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            // Get weight parameters
            params.push(layer.weight().clone());
            // Get bias parameters (assuming they always exist)
            if let Some(bias) = layer.bias() {
                params.push(bias.clone());
            }
        }
        params
    }

    /// Get the dimension of the action space.
    pub fn action_dim(&self) -> usize {
        self.output_dim
    }

    /// Get the dimension of the state space.
    pub fn state_dim(&self) -> usize {
        self.input_dim
    }

    /// Get the device used by the network.
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    pub fn hidden_dims(&self) -> &[usize] {
        &self.hidden_dims
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Save the parameter of the network to a file.
    pub fn save(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    /// Save the network parameters after converting them to a specific dtype.
    pub fn save_with_dtype(&self, path: &str, dtype: DType) -> Result<()> {
        let tensors = self
            .varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, var)| {
                let tensor = var.as_tensor();
                let tensor = if tensor.dtype() == dtype {
                    tensor.clone()
                } else {
                    tensor.to_dtype(dtype)?
                };
                Ok((name.clone(), tensor))
            })
            .collect::<Result<HashMap<String, Tensor>>>()?;
        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }
}

impl Module for NeuralNetwork {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            // 除了输出层，其他层都使用ReLU激活函数
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }
        Ok(x)
    }
}
