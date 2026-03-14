// Production Burn Neural Network Models
//
// Pure-Rust deep learning models using Burn 0.20 with NdArray backend.
// Replaces Python models (deep.py, mlp.py) — no Python, no GIL.
//
// Gated behind #[cfg(feature = "burn-backend")]
//
// Production features matching Python:
// - Class-weighted loss for imbalanced data
// - Time-series aware train/val split with embargo (no look-ahead bias)
// - Early stopping with configurable patience
// - Mini-batch training with shuffling
// - Label protocol mapping (-1 → 2)

use burn::nn;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn_ndarray::NdArray;
use burn::backend::Autodiff;

use ndarray::Array2;
use tracing::info;

/// Backend types
pub type TrainBackend = Autodiff<NdArray>;
pub type InferBackend = NdArray;

// ============================================================================
// SHARED UTILITIES — matching Python base.py
// ============================================================================

/// Map labels from {-1, 0, 1} to {2, 0, 1} matching Python protocol.
fn map_labels(y: &[i32]) -> Vec<i64> {
    y.iter().map(|&v| match v {
        -1 => 2i64,
        0 => 0i64,
        1 => 1i64,
        other => other.clamp(0, 2) as i64,
    }).collect()
}

/// Compute class weights (inverse frequency) matching Python compute_class_weights().
fn compute_class_weights(y: &[i64], n_classes: usize) -> Vec<f32> {
    let mut counts = vec![0usize; n_classes];
    for &label in y {
        let idx = label as usize;
        if idx < n_classes { counts[idx] += 1; }
    }
    let total = y.len() as f32;
    counts.iter().map(|&c| {
        if c == 0 { 1.0 } else { total / (n_classes as f32 * c as f32) }
    }).collect()
}

/// Time-series train/val split with embargo gap. No shuffling, no look-ahead.
fn time_series_split(
    n_samples: usize, val_ratio: f32, min_train: usize, embargo: usize,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    let val_size = ((n_samples as f32) * val_ratio).ceil() as usize;
    let val_size = val_size.max(1).min(n_samples.saturating_sub(min_train + embargo));
    let train_end = n_samples.saturating_sub(val_size + embargo).max(min_train.min(n_samples));
    let val_start = (train_end + embargo).min(n_samples);
    (0..train_end, val_start..n_samples)
}

/// Early stopping tracker matching Python EarlyStopper.
struct EarlyStopper {
    patience: usize,
    min_delta: f32,
    counter: usize,
    best_loss: Option<f32>,
}

impl EarlyStopper {
    fn new(patience: usize, min_delta: f32) -> Self {
        Self { patience, min_delta, counter: 0, best_loss: None }
    }
    fn check(&mut self, val_loss: f32) -> bool {
        match self.best_loss {
            None => { self.best_loss = Some(val_loss); false }
            Some(best) if val_loss > best - self.min_delta => {
                self.counter += 1;
                self.counter >= self.patience
            }
            _ => { self.best_loss = Some(val_loss); self.counter = 0; false }
        }
    }
}

/// Convert ndarray::Array2<f32> to Burn Tensor<B, 2>
fn array2_to_tensor<B: Backend>(data: &Array2<f32>, device: &B::Device) -> Tensor<B, 2> {
    let (rows, cols) = (data.nrows(), data.ncols());
    let flat: Vec<f32> = data.iter().copied().collect();
    Tensor::from_data(TensorData::new(flat, [rows, cols]), device)
}

/// Convert i64 labels to Burn Int Tensor<B, 1>
fn labels_to_tensor<B: Backend>(labels: &[i64], device: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::from_data(TensorData::new(labels.to_vec(), [labels.len()]), device)
}

// ============================================================================
// BURN MLP — matches Python MLPExpert (mlp.py)
// Uses LayerNorm instead of BatchNorm to avoid 3D reshape complications
// ============================================================================

#[derive(Module, Debug)]
pub struct BurnMLP<B: Backend> {
    layers: Vec<nn::Linear<B>>,
    norms: Vec<nn::LayerNorm<B>>,
    dropout: nn::Dropout,
    output: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct BurnMLPConfig {
    pub input_dim: usize,
    #[config(default = 256)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub n_layers: usize,
    #[config(default = 3)]
    pub n_classes: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl BurnMLPConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnMLP<B> {
        let mut layers = Vec::new();
        let mut norms = Vec::new();
        let mut dim = self.input_dim;
        for _ in 0..self.n_layers.max(1) {
            layers.push(nn::LinearConfig::new(dim, self.hidden_dim).init(device));
            norms.push(nn::LayerNormConfig::new(self.hidden_dim).init(device));
            dim = self.hidden_dim;
        }
        BurnMLP {
            layers,
            norms,
            dropout: nn::DropoutConfig::new(self.dropout).init(),
            output: nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device),
        }
    }
}

impl<B: Backend> BurnMLP<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = x;
        for (linear, norm) in self.layers.iter().zip(self.norms.iter()) {
            h = linear.forward(h);
            h = norm.forward(h);
            h = burn::tensor::activation::relu(h);
            h = self.dropout.forward(h);
        }
        self.output.forward(h)
    }
}

// ============================================================================
// BURN N-BEATS — matches Python NBeatsExpert (deep.py)
// ============================================================================

#[derive(Module, Debug)]
pub struct NBeatsBlock<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    fc3: nn::Linear<B>,
    fc4: nn::Linear<B>,
    theta_b: nn::Linear<B>,
    theta_f: nn::Linear<B>,
}

impl<B: Backend> NBeatsBlock<B> {
    fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let h = burn::tensor::activation::relu(self.fc1.forward(x));
        let h = burn::tensor::activation::relu(self.fc2.forward(h));
        let h = burn::tensor::activation::relu(self.fc3.forward(h));
        let h = burn::tensor::activation::relu(self.fc4.forward(h));
        (self.theta_b.forward(h.clone()), self.theta_f.forward(h))
    }
}

#[derive(Module, Debug)]
pub struct BurnNBeats<B: Backend> {
    embed: nn::Linear<B>,
    blocks: Vec<NBeatsBlock<B>>,
    output: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct BurnNBeatsConfig {
    pub input_dim: usize,
    #[config(default = 64)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub n_blocks: usize,
    #[config(default = 3)]
    pub n_classes: usize,
}

impl BurnNBeatsConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnNBeats<B> {
        let embed = nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device);
        let blocks = (0..self.n_blocks).map(|_| NBeatsBlock {
            fc1: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            fc3: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            fc4: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            theta_b: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim)
                .with_bias(false).init(device),
            theta_f: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim)
                .with_bias(false).init(device),
        }).collect();
        let output = nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device);
        BurnNBeats { embed, blocks, output }
    }
}

impl<B: Backend> BurnNBeats<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut residual = self.embed.forward(x);
        let mut forecast = Tensor::zeros(residual.dims(), &residual.device());
        for block in &self.blocks {
            let (backcast, fore) = block.forward(residual.clone());
            residual = residual - backcast;
            forecast = forecast + fore;
        }
        self.output.forward(forecast)
    }
}

// ============================================================================
// BURN TiDE — matches Python TiDEExpert (deep.py)
// ============================================================================

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    fc: nn::Linear<B>,
    norm: nn::LayerNorm<B>,
    dropout: nn::Dropout,
}

impl<B: Backend> ResidualBlock<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = burn::tensor::activation::relu(self.fc.forward(x.clone()));
        self.norm.forward(x + self.dropout.forward(out))
    }
}

#[derive(Module, Debug)]
pub struct BurnTiDE<B: Backend> {
    feature_proj: nn::Linear<B>,
    enc1: ResidualBlock<B>,
    enc2: ResidualBlock<B>,
    temporal_link: nn::Linear<B>,
    dec1: ResidualBlock<B>,
    dec2: ResidualBlock<B>,
    output: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct BurnTiDEConfig {
    pub input_dim: usize,
    #[config(default = 128)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub n_classes: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl BurnTiDEConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnTiDE<B> {
        let mk_res = || ResidualBlock {
            fc: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            norm: nn::LayerNormConfig::new(self.hidden_dim).init(device),
            dropout: nn::DropoutConfig::new(self.dropout).init(),
        };
        BurnTiDE {
            feature_proj: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            enc1: mk_res(), enc2: mk_res(),
            temporal_link: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            dec1: mk_res(), dec2: mk_res(),
            output: nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device),
        }
    }
}

impl<B: Backend> BurnTiDE<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.feature_proj.forward(x);
        let h = self.enc1.forward(h);
        let h = self.enc2.forward(h);
        let h = self.temporal_link.forward(h);
        let h = self.dec1.forward(h);
        let h = self.dec2.forward(h);
        self.output.forward(h)
    }
}

// ============================================================================
// BURN TabNet — matches Python TabNetExpert (deep.py)
// Uses GLU activation via manual split
// ============================================================================

#[derive(Module, Debug)]
pub struct BurnTabNet<B: Backend> {
    initial_norm: nn::LayerNorm<B>,
    feat_fc1: nn::Linear<B>,
    feat_fc2: nn::Linear<B>,
    attn_fc: nn::Linear<B>,
    attn_norm: nn::LayerNorm<B>,
    output: nn::Linear<B>,
    n_steps: usize,
    hidden_dim: usize,
    input_dim: usize,
}

#[derive(Config, Debug)]
pub struct BurnTabNetConfig {
    pub input_dim: usize,
    #[config(default = 64)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub n_steps: usize,
    #[config(default = 3)]
    pub n_classes: usize,
}

impl BurnTabNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnTabNet<B> {
        BurnTabNet {
            initial_norm: nn::LayerNormConfig::new(self.input_dim).init(device),
            feat_fc1: nn::LinearConfig::new(self.input_dim, self.hidden_dim * 2).init(device),
            feat_fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim * 2).init(device),
            attn_fc: nn::LinearConfig::new(self.hidden_dim, self.input_dim)
                .with_bias(false).init(device),
            attn_norm: nn::LayerNormConfig::new(self.input_dim).init(device),
            output: nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device),
            n_steps: self.n_steps,
            hidden_dim: self.hidden_dim,
            input_dim: self.input_dim,
        }
    }
}

/// GLU: split last dim in half, sigmoid-gate one half
fn glu<B: Backend>(x: Tensor<B, 2>, half: usize) -> Tensor<B, 2> {
    let a = x.clone().slice([0..x.dims()[0], 0..half]);
    let b = x.clone().slice([0..x.dims()[0], half..(half * 2)]);
    a * burn::tensor::activation::sigmoid(b)
}

impl<B: Backend> BurnTabNet<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch = x.dims()[0];
        let x_norm = self.initial_norm.forward(x.clone());
        let mut prior: Tensor<B, 2> = Tensor::ones([batch, self.input_dim], &x.device());
        let mut out_accum: Tensor<B, 2> = Tensor::zeros([batch, self.hidden_dim], &x.device());

        for _ in 0..self.n_steps {
            // Feature transform with GLU
            let feat = glu(self.feat_fc1.forward(x_norm.clone()), self.hidden_dim);

            // Attention mask
            let attn = self.attn_fc.forward(feat.clone());
            let attn = self.attn_norm.forward(attn * prior.clone());
            let mask = burn::tensor::activation::softmax(attn, 1);

            // Update prior (decay attention)
            let threshold = Tensor::<B, 2>::ones_like(&mask) * 1.5 - mask.clone();
            prior = prior * threshold.clamp(0.0, 1.0);

            // Masked feature → transform → accumulate
            let masked = x_norm.clone() * mask;
            let step_out = glu(self.feat_fc1.forward(masked), self.hidden_dim);
            out_accum = out_accum + step_out;
        }

        self.output.forward(out_accum)
    }
}

// ============================================================================
// BURN KAN — matches Python KANExpert (deep.py)
// ============================================================================

#[derive(Module, Debug)]
pub struct KANLayer<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
}

impl<B: Backend> KANLayer<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = burn::tensor::activation::gelu(self.fc1.forward(x));
        burn::tensor::activation::gelu(self.fc2.forward(h))
    }
}

#[derive(Module, Debug)]
pub struct BurnKAN<B: Backend> {
    layer1: KANLayer<B>,
    layer2: KANLayer<B>,
    output: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct BurnKANConfig {
    pub input_dim: usize,
    #[config(default = 32)]
    pub hidden_dim: usize,
    #[config(default = 3)]
    pub n_classes: usize,
}

impl BurnKANConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnKAN<B> {
        BurnKAN {
            layer1: KANLayer {
                fc1: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
                fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            },
            layer2: KANLayer {
                fc1: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
                fc2: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            },
            output: nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device),
        }
    }
}

impl<B: Backend> BurnKAN<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = self.layer1.forward(x);
        let h = self.layer2.forward(h);
        self.output.forward(h)
    }
}

// ============================================================================
// BURN TRANSFORMER — matches Python TransformerExpert (transformers.py)
// Manual multi-head attention (Burn has no built-in transformer module for 2D)
// ============================================================================

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    q_proj: nn::Linear<B>,
    k_proj: nn::Linear<B>,
    v_proj: nn::Linear<B>,
    out_proj: nn::Linear<B>,
    ff1: nn::Linear<B>,
    ff2: nn::Linear<B>,
    norm1: nn::LayerNorm<B>,
    norm2: nn::LayerNorm<B>,
    dropout: nn::Dropout,
    n_heads: usize,
}

impl<B: Backend> TransformerBlock<B> {
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let [batch, d_model] = x.dims();
        let head_dim = d_model / self.n_heads.max(1);

        // Multi-head self-attention (tabular: treat each sample independently)
        let q = self.q_proj.forward(x.clone()).reshape([batch, self.n_heads, head_dim]);
        let k = self.k_proj.forward(x.clone()).reshape([batch, self.n_heads, head_dim]);
        let v = self.v_proj.forward(x.clone()).reshape([batch, self.n_heads, head_dim]);

        let scale = (head_dim as f32).sqrt();
        let scores = q.matmul(k.swap_dims(1, 2)) / scale;
        let attn = burn::tensor::activation::softmax(scores, 2);
        let attn_out = attn.matmul(v).reshape([batch, d_model]);
        let attn_proj = self.out_proj.forward(attn_out);

        let h = self.norm1.forward(x + self.dropout.forward(attn_proj));
        let ff = self.ff2.forward(self.dropout.forward(
            burn::tensor::activation::gelu(self.ff1.forward(h.clone()))
        ));
        self.norm2.forward(h + self.dropout.forward(ff))
    }
}

#[derive(Module, Debug)]
pub struct BurnTransformer<B: Backend> {
    input_proj: nn::Linear<B>,
    encoder: Vec<TransformerBlock<B>>,
    final_norm: nn::LayerNorm<B>,
    output: nn::Linear<B>,
}

#[derive(Config, Debug)]
pub struct BurnTransformerConfig {
    pub input_dim: usize,
    #[config(default = 128)]
    pub hidden_dim: usize,
    #[config(default = 8)]
    pub n_heads: usize,
    #[config(default = 4)]
    pub n_layers: usize,
    #[config(default = 512)]
    pub dim_ff: usize,
    #[config(default = 3)]
    pub n_classes: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl BurnTransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BurnTransformer<B> {
        let encoder = (0..self.n_layers).map(|_| TransformerBlock {
            q_proj: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            k_proj: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            v_proj: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            out_proj: nn::LinearConfig::new(self.hidden_dim, self.hidden_dim).init(device),
            ff1: nn::LinearConfig::new(self.hidden_dim, self.dim_ff).init(device),
            ff2: nn::LinearConfig::new(self.dim_ff, self.hidden_dim).init(device),
            norm1: nn::LayerNormConfig::new(self.hidden_dim).init(device),
            norm2: nn::LayerNormConfig::new(self.hidden_dim).init(device),
            dropout: nn::DropoutConfig::new(self.dropout).init(),
            n_heads: self.n_heads,
        }).collect();
        BurnTransformer {
            input_proj: nn::LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            encoder,
            final_norm: nn::LayerNormConfig::new(self.hidden_dim).init(device),
            output: nn::LinearConfig::new(self.hidden_dim, self.n_classes).init(device),
        }
    }
}

impl<B: Backend> BurnTransformer<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut h = self.input_proj.forward(x);
        for block in &self.encoder {
            h = block.forward(h);
        }
        h = self.final_norm.forward(h);
        self.output.forward(h)
    }
}

// ============================================================================
// BURN EXPERT TRAIT — uniform interface for all models
// ============================================================================

/// Trait for all Burn models providing a consistent forward pass.
pub trait BurnForward<B: Backend> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2>;
}

impl<B: Backend> BurnForward<B> for BurnMLP<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> { self.forward(x) }
}
impl<B: Backend> BurnForward<B> for BurnNBeats<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> { self.forward(x) }
}
impl<B: Backend> BurnForward<B> for BurnTiDE<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> { self.forward(x) }
}
impl<B: Backend> BurnForward<B> for BurnTabNet<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> { self.forward(x) }
}
impl<B: Backend> BurnForward<B> for BurnKAN<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> { self.forward(x) }
}
impl<B: Backend> BurnForward<B> for BurnTransformer<B> {
    fn forward_pass(&self, x: Tensor<B, 2>) -> Tensor<B, 2> { self.forward(x) }
}

// ============================================================================
// TRAINING CONFIGURATION
// ============================================================================

/// Training configuration matching Python deep.py defaults.
pub struct TrainConfig {
    pub lr: f64,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub patience: usize,
    pub n_classes: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self { lr: 1e-3, batch_size: 64, max_epochs: 100, patience: 8, n_classes: 3 }
    }
}

// ============================================================================
// CROSS-ENTROPY LOSS
// ============================================================================

/// Weighted cross-entropy loss matching Python nn.CrossEntropyLoss(weight=...).
fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
    class_weights: &[f32],
    device: &B::Device,
) -> Tensor<B, 1> {
    let n_classes = class_weights.len();
    let batch_size = logits.dims()[0];
    let log_probs = burn::tensor::activation::log_softmax(logits, 1);
    let one_hot = targets.one_hot(n_classes).float();
    let weights = Tensor::<B, 1>::from_data(
        TensorData::new(class_weights.to_vec(), [n_classes]), device,
    );
    let weighted = log_probs * one_hot * weights.unsqueeze_dim(0);
    weighted.sum().neg() / (batch_size as f32)
}

// ============================================================================
// TRAINING LOOP — generic over all Burn models
// Matches Python deep.py: class weights, time-series split, early stopping
// ============================================================================

use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use rand::seq::SliceRandom;

/// Train any Burn model with production features.
///
/// Includes: time-series split with embargo, class-weighted loss,
/// early stopping, mini-batch, Adam optimizer, label protocol mapping.
pub fn train_model<B, M>(
    model: M,
    x_data: &Array2<f32>,
    y_raw: &[i32],
    config: &TrainConfig,
) -> (M, f32)
where
    B: AutodiffBackend,
    M: burn::module::AutodiffModule<B> + BurnForward<B>,
{
    let n_samples = x_data.nrows();
    let device = B::Device::default();

    // 1. Label mapping: -1 → 2
    let y_mapped = map_labels(y_raw);

    // 2. Time-series split with embargo
    let embargo = (n_samples as f32 * 0.005).ceil() as usize;
    let (train_range, val_range) = time_series_split(n_samples, 0.15, 100, embargo.max(10));
    let n_train = train_range.len();
    info!("Burn training: {} train, {} val, embargo={}", n_train, val_range.len(), embargo);

    // 3. Class weights
    let train_labels: Vec<i64> = y_mapped[train_range.clone()].to_vec();
    let class_weights = compute_class_weights(&train_labels, config.n_classes);

    // 4. Build tensors
    let x_train = array2_to_tensor::<B>(
        &x_data.slice(ndarray::s![train_range.clone(), ..]).to_owned(), &device,
    );
    let y_train = labels_to_tensor::<B>(&train_labels, &device);

    // 5. Optimizer + early stopping
    // Use AdamW with 5e-4 decoupled weight decay as recommended for noisy time-series
    let mut optim = AdamWConfig::new().with_weight_decay(5e-4).init();
    let mut early_stop = EarlyStopper::new(config.patience, 1e-4);
    let mut best_loss = f32::INFINITY;
    let mut model = model;
    let mut rng = rand::rng();
    let mut indices: Vec<usize> = (0..n_train).collect();

    for epoch in 0..config.max_epochs {
        indices.shuffle(&mut rng);
        let mut epoch_loss = 0.0f32;
        let mut n_batches = 0usize;

        let mut start = 0;
        while start < n_train {
            let end = (start + config.batch_size).min(n_train);

            // Gather batch using shuffled indices
            let batch_idx: Vec<i64> = indices[start..end].iter().map(|&i| i as i64).collect();
            let idx_tensor = Tensor::<B, 1, Int>::from_data(
                TensorData::new(batch_idx, [end - start]), &device,
            );
            let x_batch = x_train.clone().select(0, idx_tensor.clone());
            let y_batch = y_train.clone().select(0, idx_tensor);

            let logits = BurnForward::forward_pass(&model, x_batch);
            let loss = cross_entropy_loss(logits, y_batch, &class_weights, &device);
            let loss_val = loss.clone().into_data().to_vec::<f32>().map(|v| v[0]).unwrap_or(0.0);

            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optim.step(config.lr, model, grads_params);

            epoch_loss += loss_val;
            n_batches += 1;
            start = end;
        }

        // Validation on holdout (sequential, no shuffle)
        if !val_range.is_empty() {
            let x_val = array2_to_tensor::<B>(
                &x_data.slice(ndarray::s![val_range.clone(), ..]).to_owned(), &device,
            );
            let y_val = labels_to_tensor::<B>(&y_mapped[val_range.clone()], &device);

            let val_logits = BurnForward::forward_pass(&model, x_val);
            let val_loss = cross_entropy_loss(val_logits, y_val, &class_weights, &device);
            let vl = val_loss.into_data().to_vec::<f32>().map(|v| v[0]).unwrap_or(0.0);

            if vl < best_loss { best_loss = vl; }
            if early_stop.check(vl) {
                info!("Early stop at epoch {} (val_loss={:.6})", epoch, vl);
                break;
            }
            if epoch % 10 == 0 {
                info!("Epoch {}: train={:.6} val={:.6}", epoch, epoch_loss / n_batches.max(1) as f32, vl);
            }
        }
    }

    (model, best_loss)
}

// ============================================================================
// PREDICTION HELPER
// ============================================================================

/// Run inference and return probabilities as (n_samples, 3) array.
pub fn predict_proba<B: Backend, M: BurnForward<B>>(
    model: &M,
    x_data: &Array2<f32>,
    batch_size: usize,
) -> Array2<f32> {
    let device = B::Device::default();
    let n_samples = x_data.nrows();
    let mut all_probs: Vec<f32> = Vec::with_capacity(n_samples * 3);

    let mut start = 0;
    while start < n_samples {
        let end = (start + batch_size).min(n_samples);
        let batch = array2_to_tensor::<B>(
            &x_data.slice(ndarray::s![start..end, ..]).to_owned(), &device,
        );
        let logits = model.forward_pass(batch);
        let probs = burn::tensor::activation::softmax(logits, 1);
        let data: Vec<f32> = probs.into_data().to_vec().unwrap_or_default();
        all_probs.extend(data);
        start = end;
    }

    let n_cols = if n_samples > 0 { all_probs.len() / n_samples } else { 3 };
    Array2::from_shape_vec((n_samples, n_cols), all_probs)
        .unwrap_or_else(|_| Array2::zeros((n_samples, 3)))
}

