use anyhow::{Context, Result, bail};
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*;
use ndarray::Array2;
use std::collections::HashMap;
use symbios_neat::{Activation, GraphTopology, NeatGenome, NodeId, NodeType};

const CLASS_COUNT: usize = 3;

#[derive(Debug, Clone, Copy)]
pub(crate) struct NeatCudaMetrics {
    pub fitness: f32,
    pub loss: f32,
    pub accuracy: f32,
}

struct FlattenedNeatBatch {
    node_counts: Vec<i32>,
    node_offsets: Vec<i32>,
    edge_offsets: Vec<i32>,
    edge_sources: Vec<i32>,
    edge_weights: Vec<f32>,
    activation_codes: Vec<i32>,
    biases: Vec<f32>,
    input_indices: Vec<i32>,
    output_indices: Vec<i32>,
    bias_indices: Vec<i32>,
    eval_offsets: Vec<i32>,
    eval_indices: Vec<i32>,
    complexity_penalties: Vec<f32>,
    max_nodes: usize,
}

#[cube]
fn clamp_f32(value: f32, min_value: f32, max_value: f32) -> f32 {
    let mut out = value;
    if out < min_value {
        out = min_value;
    }
    if out > max_value {
        out = max_value;
    }
    out
}

#[cube]
fn apply_activation(code: i32, x: f32) -> f32 {
    let mut out: f32 = 0.0;
    if code == 0 {
        out = clamp_f32(x, -1000.0, 1000.0);
    } else if code == 1 {
        let clamped = clamp_f32(x, -88.0, 88.0);
        out = 1.0 / (1.0 + (-clamped).exp());
    } else if code == 2 {
        out = x.tanh();
    } else if code == 3 {
        out = clamp_f32(x, 0.0, 1000.0);
    } else if code == 4 {
        out = x.sin();
    } else if code == 5 {
        out = x.cos();
    } else if code == 6 {
        let abs_x = x.abs();
        out = (-(x * x)).exp();
        if abs_x > 26.0 {
            out = 0.0;
        }
    } else if code == 7 {
        let abs_x = x.abs();
        out = abs_x;
        if abs_x > 1000.0 {
            out = 1000.0;
        }
    } else if code == 8 {
        if x >= 0.0 {
            out = 1.0;
        } else {
            out = 0.0;
        }
    } else {
        let mut result: f32 = 0.01 * x;
        if x > 0.0 {
            result = x;
        }
        out = clamp_f32(result, -1000.0, 1000.0);
    }
    out
}

#[cube(launch)]
fn neat_population_metrics_kernel(
    features: &Array<f32>,
    labels: &Array<i32>,
    node_counts: &Array<i32>,
    node_offsets: &Array<i32>,
    edge_offsets: &Array<i32>,
    edge_sources: &Array<i32>,
    edge_weights: &Array<f32>,
    activation_codes: &Array<i32>,
    biases: &Array<f32>,
    input_indices: &Array<i32>,
    output_indices: &Array<i32>,
    bias_indices: &Array<i32>,
    eval_offsets: &Array<i32>,
    eval_indices: &Array<i32>,
    complexity_penalties: &Array<f32>,
    scratch: &mut Array<f32>,
    metrics_out: &mut Array<f32>,
    n_rows: u32,
    input_dim: u32,
    max_nodes: u32,
) {
    let candidate_count = metrics_out.len() / CLASS_COUNT;
    if ABSOLUTE_POS < candidate_count {
        let candidate = ABSOLUTE_POS;
        let n_rows_us = n_rows as usize;
        let input_dim_us = input_dim as usize;
        let max_nodes_us = max_nodes as usize;
        let node_count = node_counts[candidate] as usize;
        let node_offset = node_offsets[candidate] as usize;
        let scratch_base = candidate * max_nodes_us;
        let metric_base = candidate * CLASS_COUNT;

        if n_rows == 0 || node_count == 0 {
            metrics_out[metric_base] = -1000000.0;
            metrics_out[metric_base + 1] = 1000000.0;
            metrics_out[metric_base + 2] = 0.0;
            terminate!();
        }

        let mut log_loss: f32 = 0.0;
        let mut correct: u32 = 0;
        let mut confidence_sum: f32 = 0.0;
        let mut row = 0usize;
        while row < n_rows_us {
            let mut reset_idx = 0usize;
            while reset_idx < node_count {
                scratch[scratch_base + reset_idx] = 0.0;
                reset_idx += 1;
            }

            let input_base = candidate * input_dim_us;
            let mut input = 0usize;
            while input < input_dim_us {
                let node_idx = input_indices[input_base + input] as usize;
                scratch[scratch_base + node_idx] = features[row * input_dim_us + input];
                input += 1;
            }

            let bias_idx = bias_indices[candidate];
            if bias_idx >= 0 {
                scratch[scratch_base + bias_idx as usize] = 1.0;
            }

            let eval_start = eval_offsets[candidate] as usize;
            let eval_end = eval_offsets[candidate + 1] as usize;
            let mut eval_pos = eval_start;
            while eval_pos < eval_end {
                let node_idx = eval_indices[eval_pos] as usize;
                let absolute_node = node_offset + node_idx;
                let mut sum = biases[absolute_node];
                let edge_start = edge_offsets[absolute_node] as usize;
                let edge_end = edge_offsets[absolute_node + 1] as usize;
                let mut edge = edge_start;
                while edge < edge_end {
                    let source = edge_sources[edge] as usize;
                    sum += scratch[scratch_base + source] * edge_weights[edge];
                    edge += 1;
                }
                let activation = activation_codes[absolute_node];
                scratch[scratch_base + node_idx] = apply_activation(activation, sum);
                eval_pos += 1;
            }

            let output_base = candidate * CLASS_COUNT;
            let logit0 = scratch[scratch_base + output_indices[output_base] as usize];
            let logit1 = scratch[scratch_base + output_indices[output_base + 1] as usize];
            let logit2 = scratch[scratch_base + output_indices[output_base + 2] as usize];
            let mut max_logit = logit0;
            if logit1 > max_logit {
                max_logit = logit1;
            }
            if logit2 > max_logit {
                max_logit = logit2;
            }
            let e0 = (logit0 - max_logit).exp();
            let e1 = (logit1 - max_logit).exp();
            let e2 = (logit2 - max_logit).exp();
            let denom = e0 + e1 + e2;
            let label = labels[row];
            let mut p0 = e0 / denom;
            let mut p1 = e1 / denom;
            let mut p2 = e2 / denom;
            p0 = clamp_f32(p0, 0.000001, 0.999999);
            p1 = clamp_f32(p1, 0.000001, 0.999999);
            p2 = clamp_f32(p2, 0.000001, 0.999999);

            let mut expected_probability = p2;
            if label == 0 {
                expected_probability = p0;
            } else if label == 1 {
                expected_probability = p1;
            }
            log_loss -= expected_probability.ln();
            confidence_sum += expected_probability;

            let mut best_idx = 0i32;
            let mut best_value = p0;
            if p1 > best_value {
                best_value = p1;
                best_idx = 1;
            }
            if p2 > best_value {
                best_idx = 2;
            }
            if best_idx == label {
                correct += 1;
            }
            row += 1;
        }

        let rows = n_rows as f32;
        let avg_loss = log_loss / rows;
        let accuracy = correct as f32 / rows;
        let confidence = confidence_sum / rows;
        let fitness = (accuracy * 3.0 + confidence) - avg_loss - complexity_penalties[candidate];
        metrics_out[metric_base] = fitness;
        metrics_out[metric_base + 1] = avg_loss;
        metrics_out[metric_base + 2] = accuracy;
    }
}

pub(crate) fn neat_cuda_kernel_enabled(policy: &str) -> bool {
    let normalized = policy.trim().to_ascii_lowercase();
    let requested_gpu = normalized == "gpu" || normalized.starts_with("gpu:");
    let env_enabled = !matches!(
        std::env::var("FOREX_BOT_NEAT_CUDA_KERNEL")
            .ok()
            .map(|value| value.trim().to_ascii_lowercase()),
        Some(value) if matches!(value.as_str(), "0" | "false" | "off" | "disable" | "disabled")
    );
    requested_gpu && env_enabled
}

fn cuda_device_id(policy: &str) -> usize {
    std::env::var("FOREX_BOT_NEAT_CUDA_DEVICE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .or_else(|| {
            policy
                .trim()
                .to_ascii_lowercase()
                .strip_prefix("gpu:")
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(0)
}

fn kernel_units(client: &ComputeClient<CudaRuntime>) -> u32 {
    let max_units = client.properties().hardware.max_units_per_cube.max(1);
    std::env::var("FOREX_BOT_NEAT_KERNEL_UNITS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(max_units)
        .min(max_units)
        .max(1)
}

fn activation_code(activation: Activation) -> i32 {
    match activation {
        Activation::Identity => 0,
        Activation::Sigmoid => 1,
        Activation::Tanh => 2,
        Activation::ReLU => 3,
        Activation::Sine => 4,
        Activation::Cosine => 5,
        Activation::Gaussian => 6,
        Activation::Abs => 7,
        Activation::Step => 8,
        Activation::LeakyReLU => 9,
    }
}

fn flatten_features(features: &Array2<f32>, input_dim: usize) -> Result<Vec<f32>> {
    crate::common::cuda_flatten_features(features, input_dim, "NEAT")
}

fn flatten_labels(labels: &[usize], expected_rows: usize) -> Result<Vec<i32>> {
    if labels.len() != expected_rows {
        bail!(
            "NEAT cuda labels mismatch: {} labels for {} feature rows",
            labels.len(),
            expected_rows
        );
    }
    if labels.iter().any(|label| *label >= CLASS_COUNT) {
        bail!("NEAT cuda labels must be in 0..3");
    }
    Ok(labels.iter().map(|label| *label as i32).collect())
}

fn complexity_penalty(genome: &NeatGenome) -> f32 {
    let hidden_nodes = genome.hidden_ids().len() as f32;
    let enabled_connections = genome.num_enabled_connections() as f32;
    hidden_nodes * 0.003 + enabled_connections * 0.0006
}

fn flatten_population(population: &[NeatGenome], input_dim: usize) -> Result<FlattenedNeatBatch> {
    if population.is_empty() {
        bail!("NEAT cuda population must not be empty");
    }

    let mut node_counts = Vec::with_capacity(population.len());
    let mut node_offsets = Vec::with_capacity(population.len() + 1);
    let mut edge_offsets = Vec::<i32>::new();
    let mut edge_sources = Vec::<i32>::new();
    let mut edge_weights = Vec::<f32>::new();
    let mut activation_codes = Vec::<i32>::new();
    let mut biases = Vec::<f32>::new();
    let mut input_indices = Vec::with_capacity(population.len().saturating_mul(input_dim));
    let mut output_indices = Vec::with_capacity(population.len().saturating_mul(CLASS_COUNT));
    let mut bias_indices = Vec::with_capacity(population.len());
    let mut eval_offsets = Vec::with_capacity(population.len() + 1);
    let mut eval_indices = Vec::<i32>::new();
    let mut complexity_penalties = Vec::with_capacity(population.len());
    let mut max_nodes = 0usize;

    node_offsets.push(0);
    eval_offsets.push(0);
    edge_offsets.push(0);

    for genome in population {
        if genome.config.num_outputs != CLASS_COUNT || genome.output_ids.len() != CLASS_COUNT {
            bail!(
                "NEAT cuda requires {} output nodes, got config={} ids={}",
                CLASS_COUNT,
                genome.config.num_outputs,
                genome.output_ids.len()
            );
        }
        if genome.config.num_inputs != input_dim || genome.input_ids.len() != input_dim {
            bail!(
                "NEAT cuda input dimension mismatch: expected {}, config={} ids={}",
                input_dim,
                genome.config.num_inputs,
                genome.input_ids.len()
            );
        }

        let topo = GraphTopology::from_genome(genome);
        let depths = topo
            .compute_depths()
            .context("compile NEAT cuda topology depths")?;
        let depth_map: HashMap<NodeId, u32> = (0..topo.node_count())
            .filter_map(|idx| topo.node_id(idx).map(|id| (id, depths[idx])))
            .collect();

        let mut nodes = genome.nodes.iter().collect::<Vec<_>>();
        nodes.sort_by_key(|(id, _)| depth_map.get(id).copied().unwrap_or(0));
        let mut node_id_to_idx = HashMap::<NodeId, usize>::with_capacity(nodes.len());
        for (idx, (node_id, _)) in nodes.iter().enumerate() {
            node_id_to_idx.insert(*node_id, idx);
        }

        let node_count = nodes.len();
        if node_count == 0 {
            bail!("NEAT cuda genome contains no nodes");
        }
        max_nodes = max_nodes.max(node_count);
        node_counts.push(node_count as i32);

        let node_base = activation_codes.len();
        for (_, node) in &nodes {
            activation_codes.push(activation_code(node.activation));
            biases.push(node.bias);
        }

        let mut local_eval_indices = Vec::new();
        for (node_id, node) in &nodes {
            let idx = *node_id_to_idx
                .get(node_id)
                .context("NEAT cuda node index missing")?;
            if matches!(node.node_type, NodeType::Hidden | NodeType::Output) {
                local_eval_indices.push(idx as i32);
            }
        }
        eval_indices.extend(local_eval_indices);
        eval_offsets.push(eval_indices.len() as i32);

        for input_id in &genome.input_ids {
            input_indices.push(
                *node_id_to_idx
                    .get(input_id)
                    .context("NEAT cuda input index missing")? as i32,
            );
        }
        for output_id in &genome.output_ids {
            output_indices.push(
                *node_id_to_idx
                    .get(output_id)
                    .context("NEAT cuda output index missing")? as i32,
            );
        }
        bias_indices.push(match genome.bias_id {
            Some(id) => *node_id_to_idx
                .get(&id)
                .context("NEAT cuda bias index missing")? as i32,
            None => -1,
        });

        let (csr_offsets, csr_sources, csr_weights) =
            topo.get_csr_for_evaluation(genome, &node_id_to_idx);
        if csr_offsets.len() != node_count + 1 {
            bail!("NEAT cuda CSR offset length mismatch");
        }
        let edge_base = edge_sources.len();
        for offset in csr_offsets.iter().skip(1) {
            edge_offsets.push((edge_base + *offset) as i32);
        }
        edge_sources.extend(csr_sources.into_iter().map(|idx| idx as i32));
        edge_weights.extend(csr_weights);

        complexity_penalties.push(complexity_penalty(genome));
        debug_assert_eq!(node_base + node_count, activation_codes.len());
        node_offsets.push(activation_codes.len() as i32);
    }

    Ok(FlattenedNeatBatch {
        node_counts,
        node_offsets,
        edge_offsets,
        edge_sources,
        edge_weights,
        activation_codes,
        biases,
        input_indices,
        output_indices,
        bias_indices,
        eval_offsets,
        eval_indices,
        complexity_penalties,
        max_nodes,
    })
}

fn launch_population_metrics_kernel(
    client: &ComputeClient<CudaRuntime>,
    batch: &FlattenedNeatBatch,
    features: &Array2<f32>,
    labels: &[usize],
    input_dim: usize,
) -> Result<Vec<NeatCudaMetrics>> {
    if features.nrows() == 0 {
        return Ok(batch
            .node_counts
            .iter()
            .map(|_| NeatCudaMetrics {
                fitness: -1_000_000.0,
                loss: 1_000_000.0,
                accuracy: 0.0,
            })
            .collect());
    }
    let candidate_count = batch.node_counts.len();
    let features_flat = flatten_features(features, input_dim)?;
    let labels = flatten_labels(labels, features.nrows())?;

    let features_handle = client.create_from_slice(f32::as_bytes(&features_flat));
    let labels_handle = client.create_from_slice(i32::as_bytes(&labels));
    let node_counts_handle = client.create_from_slice(i32::as_bytes(&batch.node_counts));
    let node_offsets_handle = client.create_from_slice(i32::as_bytes(&batch.node_offsets));
    let edge_offsets_handle = client.create_from_slice(i32::as_bytes(&batch.edge_offsets));
    let edge_sources_handle = client.create_from_slice(i32::as_bytes(&batch.edge_sources));
    let edge_weights_handle = client.create_from_slice(f32::as_bytes(&batch.edge_weights));
    let activation_handle = client.create_from_slice(i32::as_bytes(&batch.activation_codes));
    let biases_handle = client.create_from_slice(f32::as_bytes(&batch.biases));
    let input_indices_handle = client.create_from_slice(i32::as_bytes(&batch.input_indices));
    let output_indices_handle = client.create_from_slice(i32::as_bytes(&batch.output_indices));
    let bias_indices_handle = client.create_from_slice(i32::as_bytes(&batch.bias_indices));
    let eval_offsets_handle = client.create_from_slice(i32::as_bytes(&batch.eval_offsets));
    let eval_indices_handle = client.create_from_slice(i32::as_bytes(&batch.eval_indices));
    let complexity_handle = client.create_from_slice(f32::as_bytes(&batch.complexity_penalties));

    let scratch_len = candidate_count.saturating_mul(batch.max_nodes);
    let scratch_handle = client.empty(scratch_len.saturating_mul(std::mem::size_of::<f32>()));
    let metrics_len = candidate_count.saturating_mul(CLASS_COUNT);
    let metrics_handle = client.empty(metrics_len.saturating_mul(std::mem::size_of::<f32>()));

    let units = kernel_units(client);
    let cubes = (candidate_count as u32).div_ceil(units);
    neat_population_metrics_kernel::launch::<CudaRuntime>(
        client,
        CubeCount::Static(cubes, 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts::<f32>(&features_handle, features_flat.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&labels_handle, labels.len(), 1) },
        unsafe { ArrayArg::from_raw_parts::<i32>(&node_counts_handle, batch.node_counts.len(), 1) },
        unsafe {
            ArrayArg::from_raw_parts::<i32>(&node_offsets_handle, batch.node_offsets.len(), 1)
        },
        unsafe {
            ArrayArg::from_raw_parts::<i32>(&edge_offsets_handle, batch.edge_offsets.len(), 1)
        },
        unsafe {
            ArrayArg::from_raw_parts::<i32>(&edge_sources_handle, batch.edge_sources.len(), 1)
        },
        unsafe {
            ArrayArg::from_raw_parts::<f32>(&edge_weights_handle, batch.edge_weights.len(), 1)
        },
        unsafe {
            ArrayArg::from_raw_parts::<i32>(&activation_handle, batch.activation_codes.len(), 1)
        },
        unsafe { ArrayArg::from_raw_parts::<f32>(&biases_handle, batch.biases.len(), 1) },
        unsafe {
            ArrayArg::from_raw_parts::<i32>(&input_indices_handle, batch.input_indices.len(), 1)
        },
        unsafe {
            ArrayArg::from_raw_parts::<i32>(&output_indices_handle, batch.output_indices.len(), 1)
        },
        unsafe {
            ArrayArg::from_raw_parts::<i32>(&bias_indices_handle, batch.bias_indices.len(), 1)
        },
        unsafe {
            ArrayArg::from_raw_parts::<i32>(&eval_offsets_handle, batch.eval_offsets.len(), 1)
        },
        unsafe {
            ArrayArg::from_raw_parts::<i32>(&eval_indices_handle, batch.eval_indices.len(), 1)
        },
        unsafe {
            ArrayArg::from_raw_parts::<f32>(&complexity_handle, batch.complexity_penalties.len(), 1)
        },
        unsafe { ArrayArg::from_raw_parts::<f32>(&scratch_handle, scratch_len, 1) },
        unsafe { ArrayArg::from_raw_parts::<f32>(&metrics_handle, metrics_len, 1) },
        ScalarArg::new(features.nrows() as u32),
        ScalarArg::new(input_dim as u32),
        ScalarArg::new(batch.max_nodes as u32),
    )
    .context("launch NEAT cuda population metrics kernel")?;

    let bytes = client.read_one(metrics_handle);
    let flat = f32::from_bytes(&bytes);
    if flat.len() != metrics_len {
        bail!(
            "NEAT cuda metrics length mismatch: expected {}, received {}",
            metrics_len,
            flat.len()
        );
    }
    Ok(flat
        .chunks_exact(CLASS_COUNT)
        .map(|chunk| NeatCudaMetrics {
            fitness: chunk[0],
            loss: chunk[1],
            accuracy: chunk[2],
        })
        .collect())
}

pub(crate) fn try_population_scores_cuda(
    population: &[NeatGenome],
    train_features: &Array2<f32>,
    train_labels: &[usize],
    val_features: &Array2<f32>,
    val_labels: &[usize],
    policy: &str,
) -> Result<Vec<NeatCudaMetrics>> {
    if population.is_empty() {
        return Ok(Vec::new());
    }
    let input_dim = train_features.ncols();
    if input_dim == 0 {
        bail!("NEAT cuda requires at least one feature column");
    }
    if val_features.ncols() != input_dim {
        bail!(
            "NEAT cuda validation feature dimension mismatch: train {}, val {}",
            input_dim,
            val_features.ncols()
        );
    }

    let batch = flatten_population(population, input_dim)?;
    let device = CudaDevice::new(cuda_device_id(policy));
    let client = CudaRuntime::client(&device);
    let train_metrics =
        launch_population_metrics_kernel(&client, &batch, train_features, train_labels, input_dim)?;
    let val_metrics = if val_labels.is_empty() {
        train_metrics.clone()
    } else {
        launch_population_metrics_kernel(&client, &batch, val_features, val_labels, input_dim)?
    };

    Ok(train_metrics
        .into_iter()
        .zip(val_metrics)
        .map(|(train, val)| {
            if val_labels.is_empty() {
                train
            } else {
                NeatCudaMetrics {
                    fitness: 0.65 * train.fitness + 0.35 * val.fitness,
                    loss: 0.65 * train.loss + 0.35 * val.loss,
                    accuracy: 0.65 * train.accuracy + 0.35 * val.accuracy,
                }
            }
        })
        .collect())
}
