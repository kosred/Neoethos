// Pure-Rust Neural Network Models via RusTorch
//
// These replace the PyO3/Python wrappers in neural_networks.rs,
// eliminating GIL contention, OOM from Python memory copies, and
// the Python runtime dependency for inference.
//
// Gated behind `#[cfg(feature = "rustorch")]`.
// Fallback to PyO3 wrappers when feature is disabled.

#[cfg(feature = "rustorch")]
use rus_torch::core::{Tensor, GradMode};
#[cfg(feature = "rustorch")]
use rustorch_nn::{
    linear::Linear,
    dropout::Dropout,
    norm::{BatchNorm1d, LayerNorm},
    transformer::TransformerEncoderLayer,
    optim::{Adam, Optimizer},
    loss::CrossEntropyLoss,
    module::Module,
};

use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Serialize, Deserialize};
use std::path::Path;
use tracing::{info, warn};

// ============================================================================
// MODEL CONFIGURATION
// ============================================================================

/// Configuration for RusTorch MLP model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RusTorchMLPConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_classes: usize,
    pub dropout: f64,
    pub lr: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
}

impl Default for RusTorchMLPConfig {
    fn default() -> Self {
        Self {
            input_dim: 96,
            hidden_dim: 256,
            n_layers: 3,
            n_classes: 3, // [neutral, buy, sell]
            dropout: 0.3,
            lr: 1e-3,
            batch_size: 256,
            max_epochs: 100,
            patience: 10,
        }
    }
}

/// Configuration for RusTorch Transformer model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RusTorchTransformerConfig {
    pub input_dim: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub dim_feedforward: usize,
    pub n_classes: usize,
    pub dropout: f64,
    pub lr: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
}

impl Default for RusTorchTransformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 96,
            d_model: 128,
            n_heads: 8,
            n_layers: 4,
            dim_feedforward: 512,
            n_classes: 3,
            dropout: 0.1,
            lr: 1e-4,
            batch_size: 64,
            max_epochs: 50,
            patience: 8,
        }
    }
}

/// Configuration for RusTorch N-BEATS model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RusTorchNBeatsConfig {
    pub input_dim: usize,
    pub n_stacks: usize,
    pub n_blocks_per_stack: usize,
    pub hidden_dim: usize,
    pub n_classes: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
}

impl Default for RusTorchNBeatsConfig {
    fn default() -> Self {
        Self {
            input_dim: 96,
            n_stacks: 2,
            n_blocks_per_stack: 3,
            hidden_dim: 256,
            n_classes: 3,
            lr: 1e-3,
            batch_size: 128,
            max_epochs: 80,
            patience: 10,
        }
    }
}

/// Configuration for RusTorch TiDE model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RusTorchTiDEConfig {
    pub input_dim: usize,
    pub encoder_dim: usize,
    pub decoder_dim: usize,
    pub n_encoder_layers: usize,
    pub n_decoder_layers: usize,
    pub n_classes: usize,
    pub dropout: f64,
    pub lr: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
}

impl Default for RusTorchTiDEConfig {
    fn default() -> Self {
        Self {
            input_dim: 96,
            encoder_dim: 128,
            decoder_dim: 128,
            n_encoder_layers: 2,
            n_decoder_layers: 2,
            n_classes: 3,
            dropout: 0.1,
            lr: 1e-3,
            batch_size: 128,
            max_epochs: 60,
            patience: 10,
        }
    }
}

// ============================================================================
// RUSTORCH MLP MODEL — Pure Rust, no GIL, no Python
// ============================================================================

#[cfg(feature = "rustorch")]
pub struct RusTorchMLP {
    config: RusTorchMLPConfig,
    layers: Vec<Linear>,
    bn_layers: Vec<BatchNorm1d>,
    dropout: Dropout,
    output_layer: Linear,
    trained: bool,
}

#[cfg(feature = "rustorch")]
impl RusTorchMLP {
    pub fn new(config: RusTorchMLPConfig) -> Self {
        let mut layers = Vec::new();
        let mut bn_layers = Vec::new();

        // Build hidden layers: input → hidden → hidden → ... → output
        let mut prev_dim = config.input_dim;
        for _ in 0..config.n_layers {
            layers.push(Linear::new(prev_dim, config.hidden_dim, true));
            bn_layers.push(BatchNorm1d::new(config.hidden_dim));
            prev_dim = config.hidden_dim;
        }

        let dropout = Dropout::new(config.dropout);
        let output_layer = Linear::new(config.hidden_dim, config.n_classes, true);

        Self {
            config,
            layers,
            bn_layers,
            dropout,
            output_layer,
            trained: false,
        }
    }

    /// Forward pass: x → Linear → BN → ReLU → Dropout → ... → Linear → Softmax
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        for (linear, bn) in self.layers.iter().zip(self.bn_layers.iter()) {
            out = linear.forward(&out);
            out = bn.forward(&out);
            out = out.relu();
            out = self.dropout.forward(&out);
        }
        out = self.output_layer.forward(&out);
        out.softmax(-1)
    }

    /// Train the MLP on feature matrix and labels.
    /// x: (n_samples, n_features), y: (n_samples,) with values in {0, 1, 2}
    pub fn fit(&mut self, x_data: &Array2<f32>, y_data: &[i32]) -> Result<()> {
        let n_samples = x_data.nrows();
        let n_features = x_data.ncols();
        info!(
            "RusTorchMLP training: {} samples, {} features, {} hidden, {} layers",
            n_samples, n_features, self.config.hidden_dim, self.config.n_layers
        );

        // Convert ndarray to RusTorch tensors
        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor = Tensor::from_vec(x_flat, &[n_samples, n_features]);
        let y_labels: Vec<i64> = y_data.iter().map(|&v| v as i64).collect();
        let y_tensor = Tensor::from_vec(y_labels, &[n_samples]);

        let loss_fn = CrossEntropyLoss::new();
        let mut optimizer = Adam::new(self.parameters(), self.config.lr);

        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            // Mini-batch training
            let mut start = 0;
            while start < n_samples {
                let end = (start + self.config.batch_size).min(n_samples);
                let x_batch = x_tensor.narrow(0, start, end - start);
                let y_batch = y_tensor.narrow(0, start, end - start);

                optimizer.zero_grad();
                let pred = self.forward(&x_batch);
                let loss = loss_fn.forward(&pred, &y_batch);
                loss.backward();
                optimizer.step();

                epoch_loss += loss.to_f64();
                n_batches += 1;
                start = end;
            }

            let avg_loss = epoch_loss / n_batches as f64;

            // Early stopping
            if avg_loss < best_loss - 1e-4 {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    info!("Early stopping at epoch {} (loss: {:.6})", epoch, avg_loss);
                    break;
                }
            }

            if epoch % 10 == 0 {
                info!("Epoch {}: loss = {:.6}", epoch, avg_loss);
            }
        }

        self.trained = true;
        info!("RusTorchMLP training complete. Best loss: {:.6}", best_loss);
        Ok(())
    }

    /// Predict probabilities: returns (n_samples, 3) array for [neutral, buy, sell]
    pub fn predict_proba(&self, x_data: &Array2<f32>) -> Result<Array2<f32>> {
        let n_samples = x_data.nrows();
        let n_features = x_data.ncols();

        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor = Tensor::from_vec(x_flat, &[n_samples, n_features]);

        let _guard = GradMode::no_grad();
        let proba = self.forward(&x_tensor);

        // Convert tensor back to ndarray
        let proba_vec: Vec<f32> = proba.to_vec();
        let result = Array2::from_shape_vec((n_samples, self.config.n_classes), proba_vec)
            .context("Failed to reshape RusTorch output to Array2")?;
        Ok(result)
    }

    /// Collect all trainable parameters.
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        for bn in &self.bn_layers {
            params.extend(bn.parameters());
        }
        params.extend(self.output_layer.parameters());
        params
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let config_path = path.join("rustorch_mlp_config.json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(&config_path, config_json)?;
        // TODO: Save weights when RusTorch adds serde support
        info!("RusTorchMLP config saved to {:?}", config_path);
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let config_path = path.join("rustorch_mlp_config.json");
        if config_path.exists() {
            let config_json = std::fs::read_to_string(&config_path)?;
            self.config = serde_json::from_str(&config_json)?;
            info!("RusTorchMLP config loaded from {:?}", config_path);
        }
        Ok(())
    }
}

// ============================================================================
// RUSTORCH TRANSFORMER MODEL — Pure Rust
// ============================================================================

#[cfg(feature = "rustorch")]
pub struct RusTorchTransformer {
    config: RusTorchTransformerConfig,
    input_projection: Linear,
    encoder_layers: Vec<TransformerEncoderLayer>,
    output_layer: Linear,
    trained: bool,
}

#[cfg(feature = "rustorch")]
impl RusTorchTransformer {
    pub fn new(config: RusTorchTransformerConfig) -> Self {
        let input_projection = Linear::new(config.input_dim, config.d_model, true);

        let mut encoder_layers = Vec::new();
        for _ in 0..config.n_layers {
            encoder_layers.push(TransformerEncoderLayer::new(
                config.d_model,
                config.n_heads,
                config.dim_feedforward,
                config.dropout,
            ));
        }

        let output_layer = Linear::new(config.d_model, config.n_classes, true);

        Self {
            config,
            input_projection,
            encoder_layers,
            output_layer,
            trained: false,
        }
    }

    /// Forward: project → transformer encoder stack → mean pool → classify
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // x: (batch, features)
        // Project to d_model, unsqueeze for sequence dim (seq_len=1)
        let mut out = self.input_projection.forward(x);
        out = out.unsqueeze(1); // (batch, 1, d_model)

        for encoder in &self.encoder_layers {
            out = encoder.forward(&out);
        }

        // Mean pool over sequence dimension
        out = out.mean(1); // (batch, d_model)
        out = self.output_layer.forward(&out);
        out.softmax(-1)
    }

    pub fn fit(&mut self, x_data: &Array2<f32>, y_data: &[i32]) -> Result<()> {
        let n_samples = x_data.nrows();
        let n_features = x_data.ncols();
        info!(
            "RusTorchTransformer training: {} samples, d_model={}, heads={}, layers={}",
            n_samples, self.config.d_model, self.config.n_heads, self.config.n_layers
        );

        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor = Tensor::from_vec(x_flat, &[n_samples, n_features]);
        let y_labels: Vec<i64> = y_data.iter().map(|&v| v as i64).collect();
        let y_tensor = Tensor::from_vec(y_labels, &[n_samples]);

        let loss_fn = CrossEntropyLoss::new();
        let mut optimizer = Adam::new(self.parameters(), self.config.lr);

        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            let mut start = 0;
            while start < n_samples {
                let end = (start + self.config.batch_size).min(n_samples);
                let x_batch = x_tensor.narrow(0, start, end - start);
                let y_batch = y_tensor.narrow(0, start, end - start);

                optimizer.zero_grad();
                let pred = self.forward(&x_batch);
                let loss = loss_fn.forward(&pred, &y_batch);
                loss.backward();
                optimizer.step();

                epoch_loss += loss.to_f64();
                n_batches += 1;
                start = end;
            }

            let avg_loss = epoch_loss / n_batches as f64;
            if avg_loss < best_loss - 1e-4 {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    info!("Transformer early stopping at epoch {} (loss: {:.6})", epoch, avg_loss);
                    break;
                }
            }

            if epoch % 5 == 0 {
                info!("Transformer epoch {}: loss = {:.6}", epoch, avg_loss);
            }
        }

        self.trained = true;
        Ok(())
    }

    pub fn predict_proba(&self, x_data: &Array2<f32>) -> Result<Array2<f32>> {
        let n_samples = x_data.nrows();
        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor = Tensor::from_vec(x_flat, &[n_samples, x_data.ncols()]);

        let _guard = GradMode::no_grad();
        let proba = self.forward(&x_tensor);
        let proba_vec: Vec<f32> = proba.to_vec();
        let result = Array2::from_shape_vec((n_samples, self.config.n_classes), proba_vec)
            .context("Failed to reshape transformer output")?;
        Ok(result)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.input_projection.parameters());
        for enc in &self.encoder_layers {
            params.extend(enc.parameters());
        }
        params.extend(self.output_layer.parameters());
        params
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let p = path.join("rustorch_transformer_config.json");
        std::fs::write(&p, serde_json::to_string_pretty(&self.config)?)?;
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let p = path.join("rustorch_transformer_config.json");
        if p.exists() {
            self.config = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
        }
        Ok(())
    }
}

// ============================================================================
// RUSTORCH N-BEATS MODEL — Pure Rust
// Residual stacking of fully-connected blocks for time-series classification
// ============================================================================

#[cfg(feature = "rustorch")]
pub struct RusTorchNBeats {
    config: RusTorchNBeatsConfig,
    /// Each block: [Linear, Linear, Linear, Linear] (4 FC layers per block)
    blocks: Vec<[Linear; 4]>,
    /// Output projection per stack
    stack_outputs: Vec<Linear>,
    /// Final classification head
    classifier: Linear,
    trained: bool,
}

#[cfg(feature = "rustorch")]
impl RusTorchNBeats {
    pub fn new(config: RusTorchNBeatsConfig) -> Self {
        let mut blocks = Vec::new();
        let mut stack_outputs = Vec::new();

        for _ in 0..config.n_stacks {
            for _ in 0..config.n_blocks_per_stack {
                blocks.push([
                    Linear::new(config.input_dim, config.hidden_dim, true),
                    Linear::new(config.hidden_dim, config.hidden_dim, true),
                    Linear::new(config.hidden_dim, config.hidden_dim, true),
                    Linear::new(config.hidden_dim, config.input_dim, true), // backcast
                ]);
            }
            stack_outputs.push(Linear::new(config.input_dim, config.input_dim, true));
        }

        let classifier = Linear::new(config.input_dim, config.n_classes, true);

        Self {
            config,
            blocks,
            stack_outputs,
            classifier,
            trained: false,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut residual = x.clone();
        let mut forecast = Tensor::zeros_like(x);
        let blocks_per_stack = self.config.n_blocks_per_stack;

        for (stack_idx, stack_out) in self.stack_outputs.iter().enumerate() {
            for block_idx in 0..blocks_per_stack {
                let block = &self.blocks[stack_idx * blocks_per_stack + block_idx];
                let mut h = block[0].forward(&residual).relu();
                h = block[1].forward(&h).relu();
                h = block[2].forward(&h).relu();
                let backcast = block[3].forward(&h);
                residual = &residual - &backcast; // residual connection
            }
            let stack_forecast = stack_out.forward(&residual);
            forecast = &forecast + &stack_forecast;
        }

        let out = self.classifier.forward(&forecast);
        out.softmax(-1)
    }

    pub fn fit(&mut self, x_data: &Array2<f32>, y_data: &[i32]) -> Result<()> {
        let n_samples = x_data.nrows();
        info!("RusTorchNBeats training: {} samples, {} stacks × {} blocks",
              n_samples, self.config.n_stacks, self.config.n_blocks_per_stack);

        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor = Tensor::from_vec(x_flat, &[n_samples, x_data.ncols()]);
        let y_labels: Vec<i64> = y_data.iter().map(|&v| v as i64).collect();
        let y_tensor = Tensor::from_vec(y_labels, &[n_samples]);

        let loss_fn = CrossEntropyLoss::new();
        let mut optimizer = Adam::new(self.parameters(), self.config.lr);

        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            let mut start = 0;
            while start < n_samples {
                let end = (start + self.config.batch_size).min(n_samples);
                let x_batch = x_tensor.narrow(0, start, end - start);
                let y_batch = y_tensor.narrow(0, start, end - start);

                optimizer.zero_grad();
                let pred = self.forward(&x_batch);
                let loss = loss_fn.forward(&pred, &y_batch);
                loss.backward();
                optimizer.step();

                epoch_loss += loss.to_f64();
                n_batches += 1;
                start = end;
            }

            let avg_loss = epoch_loss / n_batches as f64;
            if avg_loss < best_loss - 1e-4 {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    info!("NBeats early stopping at epoch {}", epoch);
                    break;
                }
            }
        }

        self.trained = true;
        Ok(())
    }

    pub fn predict_proba(&self, x_data: &Array2<f32>) -> Result<Array2<f32>> {
        let n = x_data.nrows();
        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor = Tensor::from_vec(x_flat, &[n, x_data.ncols()]);

        let _guard = GradMode::no_grad();
        let proba = self.forward(&x_tensor);
        let proba_vec: Vec<f32> = proba.to_vec();
        Ok(Array2::from_shape_vec((n, self.config.n_classes), proba_vec)?)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for block in &self.blocks {
            for layer in block {
                params.extend(layer.parameters());
            }
        }
        for so in &self.stack_outputs {
            params.extend(so.parameters());
        }
        params.extend(self.classifier.parameters());
        params
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let p = path.join("rustorch_nbeats_config.json");
        std::fs::write(&p, serde_json::to_string_pretty(&self.config)?)?;
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let p = path.join("rustorch_nbeats_config.json");
        if p.exists() {
            self.config = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
        }
        Ok(())
    }
}

// ============================================================================
// RUSTORCH TiDE MODEL — Pure Rust
// Time-series Dense Encoder: encoder → decoder → classification
// ============================================================================

#[cfg(feature = "rustorch")]
pub struct RusTorchTiDE {
    config: RusTorchTiDEConfig,
    encoder_layers: Vec<Linear>,
    encoder_norms: Vec<LayerNorm>,
    decoder_layers: Vec<Linear>,
    decoder_norms: Vec<LayerNorm>,
    dropout: Dropout,
    classifier: Linear,
    trained: bool,
}

#[cfg(feature = "rustorch")]
impl RusTorchTiDE {
    pub fn new(config: RusTorchTiDEConfig) -> Self {
        let mut encoder_layers = Vec::new();
        let mut encoder_norms = Vec::new();
        let mut prev = config.input_dim;
        for _ in 0..config.n_encoder_layers {
            encoder_layers.push(Linear::new(prev, config.encoder_dim, true));
            encoder_norms.push(LayerNorm::new(config.encoder_dim));
            prev = config.encoder_dim;
        }

        let mut decoder_layers = Vec::new();
        let mut decoder_norms = Vec::new();
        prev = config.encoder_dim;
        for _ in 0..config.n_decoder_layers {
            decoder_layers.push(Linear::new(prev, config.decoder_dim, true));
            decoder_norms.push(LayerNorm::new(config.decoder_dim));
            prev = config.decoder_dim;
        }

        let dropout = Dropout::new(config.dropout);
        let classifier = Linear::new(config.decoder_dim, config.n_classes, true);

        Self {
            config,
            encoder_layers,
            encoder_norms,
            decoder_layers,
            decoder_norms,
            dropout,
            classifier,
            trained: false,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Encoder: Linear → LayerNorm → ReLU → Dropout (with residual when dims match)
        let mut h = x.clone();
        for (linear, norm) in self.encoder_layers.iter().zip(self.encoder_norms.iter()) {
            let projected = linear.forward(&h);
            let normed = norm.forward(&projected);
            h = self.dropout.forward(&normed.relu());
        }

        // Decoder: same pattern
        for (linear, norm) in self.decoder_layers.iter().zip(self.decoder_norms.iter()) {
            let projected = linear.forward(&h);
            let normed = norm.forward(&projected);
            // Residual connection (dims match within decoder)
            let out = self.dropout.forward(&normed.relu());
            if h.shape() == out.shape() {
                h = &h + &out;
            } else {
                h = out;
            }
        }

        let logits = self.classifier.forward(&h);
        logits.softmax(-1)
    }

    pub fn fit(&mut self, x_data: &Array2<f32>, y_data: &[i32]) -> Result<()> {
        let n_samples = x_data.nrows();
        info!("RusTorchTiDE training: {} samples, enc={}, dec={}",
              n_samples, self.config.n_encoder_layers, self.config.n_decoder_layers);

        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor = Tensor::from_vec(x_flat, &[n_samples, x_data.ncols()]);
        let y_labels: Vec<i64> = y_data.iter().map(|&v| v as i64).collect();
        let y_tensor = Tensor::from_vec(y_labels, &[n_samples]);

        let loss_fn = CrossEntropyLoss::new();
        let mut optimizer = Adam::new(self.parameters(), self.config.lr);

        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            let mut start = 0;
            while start < n_samples {
                let end = (start + self.config.batch_size).min(n_samples);
                optimizer.zero_grad();
                let pred = self.forward(&x_tensor.narrow(0, start, end - start));
                let loss = loss_fn.forward(&pred, &y_tensor.narrow(0, start, end - start));
                loss.backward();
                optimizer.step();
                epoch_loss += loss.to_f64();
                n_batches += 1;
                start = end;
            }

            let avg_loss = epoch_loss / n_batches as f64;
            if avg_loss < best_loss - 1e-4 {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    info!("TiDE early stopping at epoch {}", epoch);
                    break;
                }
            }
        }

        self.trained = true;
        Ok(())
    }

    pub fn predict_proba(&self, x_data: &Array2<f32>) -> Result<Array2<f32>> {
        let n = x_data.nrows();
        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor = Tensor::from_vec(x_flat, &[n, x_data.ncols()]);

        let _guard = GradMode::no_grad();
        let proba = self.forward(&x_tensor);
        let proba_vec: Vec<f32> = proba.to_vec();
        Ok(Array2::from_shape_vec((n, self.config.n_classes), proba_vec)?)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for l in &self.encoder_layers { params.extend(l.parameters()); }
        for n in &self.encoder_norms { params.extend(n.parameters()); }
        for l in &self.decoder_layers { params.extend(l.parameters()); }
        for n in &self.decoder_norms { params.extend(n.parameters()); }
        params.extend(self.classifier.parameters());
        params
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let p = path.join("rustorch_tide_config.json");
        std::fs::write(&p, serde_json::to_string_pretty(&self.config)?)?;
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let p = path.join("rustorch_tide_config.json");
        if p.exists() {
            self.config = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
        }
        Ok(())
    }
}

// ============================================================================
// RUSTORCH TABNET MODEL — Pure Rust (with custom GLU since RusTorch lacks it)
// ============================================================================

#[cfg(feature = "rustorch")]
pub struct RusTorchTabNet {
    config: RusTorchTabNetConfig,
    initial_bn: BatchNorm1d,
    shared_fc: Linear,
    step_attentive: Vec<Linear>,
    step_fc: Vec<Linear>,
    classifier: Linear,
    trained: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RusTorchTabNetConfig {
    pub input_dim: usize,
    pub n_steps: usize,
    pub n_classes: usize,
    pub hidden_dim: usize,
    pub lr: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
}

impl Default for RusTorchTabNetConfig {
    fn default() -> Self {
        Self {
            input_dim: 96,
            n_steps: 5,
            n_classes: 3,
            hidden_dim: 128,
            lr: 2e-2,
            batch_size: 256,
            max_epochs: 100,
            patience: 15,
        }
    }
}

#[cfg(feature = "rustorch")]
impl RusTorchTabNet {
    /// GLU: Gated Linear Unit = x[:, :half] * sigmoid(x[:, half:])
    fn glu(x: &Tensor) -> Tensor {
        let half = x.size(-1) / 2;
        let (a, b) = x.split(half, -1);
        &a * &b.sigmoid()
    }

    pub fn new(config: RusTorchTabNetConfig) -> Self {
        let initial_bn = BatchNorm1d::new(config.input_dim);
        let shared_fc = Linear::new(config.input_dim, config.hidden_dim * 2, true);

        let mut step_attentive = Vec::new();
        let mut step_fc = Vec::new();
        for _ in 0..config.n_steps {
            step_attentive.push(Linear::new(config.hidden_dim, config.input_dim, true));
            step_fc.push(Linear::new(config.input_dim, config.hidden_dim * 2, true));
        }

        let classifier = Linear::new(config.hidden_dim, config.n_classes, true);

        Self {
            config,
            initial_bn,
            shared_fc,
            step_attentive,
            step_fc,
            classifier,
            trained: false,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let normed = self.initial_bn.forward(x);
        let shared = Self::glu(&self.shared_fc.forward(&normed));
        let mut aggregated = Tensor::zeros_like(&shared);

        for step_idx in 0..self.config.n_steps {
            // Attention mask
            let attention = self.step_attentive[step_idx].forward(&shared).softmax(-1);
            let masked = &normed * &attention;
            let step_out = Self::glu(&self.step_fc[step_idx].forward(&masked));
            aggregated = &aggregated + &step_out;
        }

        let logits = self.classifier.forward(&aggregated);
        logits.softmax(-1)
    }

    pub fn fit(&mut self, x_data: &Array2<f32>, y_data: &[i32]) -> Result<()> {
        let n_samples = x_data.nrows();
        info!("RusTorchTabNet training: {} samples, {} steps", n_samples, self.config.n_steps);

        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor = Tensor::from_vec(x_flat, &[n_samples, x_data.ncols()]);
        let y_labels: Vec<i64> = y_data.iter().map(|&v| v as i64).collect();
        let y_tensor = Tensor::from_vec(y_labels, &[n_samples]);

        let loss_fn = CrossEntropyLoss::new();
        let mut optimizer = Adam::new(self.parameters(), self.config.lr);

        let mut best_loss = f64::INFINITY;
        let mut patience_counter = 0;

        for epoch in 0..self.config.max_epochs {
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;

            let mut start = 0;
            while start < n_samples {
                let end = (start + self.config.batch_size).min(n_samples);
                optimizer.zero_grad();
                let pred = self.forward(&x_tensor.narrow(0, start, end - start));
                let loss = loss_fn.forward(&pred, &y_tensor.narrow(0, start, end - start));
                loss.backward();
                optimizer.step();
                epoch_loss += loss.to_f64();
                n_batches += 1;
                start = end;
            }

            let avg_loss = epoch_loss / n_batches as f64;
            if avg_loss < best_loss - 1e-4 {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    info!("TabNet early stopping at epoch {}", epoch);
                    break;
                }
            }
        }

        self.trained = true;
        Ok(())
    }

    pub fn predict_proba(&self, x_data: &Array2<f32>) -> Result<Array2<f32>> {
        let n = x_data.nrows();
        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor = Tensor::from_vec(x_flat, &[n, x_data.ncols()]);
        let _guard = GradMode::no_grad();
        let proba = self.forward(&x_tensor);
        Ok(Array2::from_shape_vec((n, self.config.n_classes), proba.to_vec())?)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.initial_bn.parameters());
        params.extend(self.shared_fc.parameters());
        for a in &self.step_attentive { params.extend(a.parameters()); }
        for f in &self.step_fc { params.extend(f.parameters()); }
        params.extend(self.classifier.parameters());
        params
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let p = path.join("rustorch_tabnet_config.json");
        std::fs::write(&p, serde_json::to_string_pretty(&self.config)?)?;
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<()> {
        let p = path.join("rustorch_tabnet_config.json");
        if p.exists() {
            self.config = serde_json::from_str(&std::fs::read_to_string(&p)?)?;
        }
        Ok(())
    }
}

// ============================================================================
// RUSTORCH RL MODELS — PPO/SAC value and policy networks
// ============================================================================

#[cfg(feature = "rustorch")]
pub struct RusTorchRLNetwork {
    config: RusTorchRLConfig,
    shared_layers: Vec<Linear>,
    policy_head: Linear,
    value_head: Linear,
    trained: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RusTorchRLConfig {
    pub state_dim: usize,
    pub action_dim: usize, // 3 for [hold, buy, sell]
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub lr: f64,
}

impl Default for RusTorchRLConfig {
    fn default() -> Self {
        Self {
            state_dim: 96,
            action_dim: 3,
            hidden_dim: 128,
            n_layers: 2,
            lr: 3e-4,
        }
    }
}

#[cfg(feature = "rustorch")]
impl RusTorchRLNetwork {
    pub fn new(config: RusTorchRLConfig) -> Self {
        let mut shared_layers = Vec::new();
        let mut prev = config.state_dim;
        for _ in 0..config.n_layers {
            shared_layers.push(Linear::new(prev, config.hidden_dim, true));
            prev = config.hidden_dim;
        }

        let policy_head = Linear::new(config.hidden_dim, config.action_dim, true);
        let value_head = Linear::new(config.hidden_dim, 1, true);

        Self {
            config,
            shared_layers,
            policy_head,
            value_head,
            trained: false,
        }
    }

    /// Returns (action_probs, state_value)
    pub fn forward(&self, state: &Tensor) -> (Tensor, Tensor) {
        let mut h = state.clone();
        for layer in &self.shared_layers {
            h = layer.forward(&h).relu();
        }
        let policy = self.policy_head.forward(&h).softmax(-1);
        let value = self.value_head.forward(&h);
        (policy, value)
    }

    pub fn predict_action_probs(&self, x_data: &Array2<f32>) -> Result<Array2<f32>> {
        let n = x_data.nrows();
        let x_flat: Vec<f32> = x_data.iter().copied().collect();
        let x_tensor = Tensor::from_vec(x_flat, &[n, x_data.ncols()]);

        let _guard = GradMode::no_grad();
        let (probs, _value) = self.forward(&x_tensor);
        let probs_vec: Vec<f32> = probs.to_vec();
        Ok(Array2::from_shape_vec((n, self.config.action_dim), probs_vec)?)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for l in &self.shared_layers { params.extend(l.parameters()); }
        params.extend(self.policy_head.parameters());
        params.extend(self.value_head.parameters());
        params
    }
}
