//! Separate tiny-fixture signal trace specialization for parity levels 1-3.
//!
//! Production kernels remain compact-output only. This module is compiled only
//! with a GPU backend and writes diagnostic score/signal/confidence/SMC buffers.

use anyhow::{Context, Result, bail};
#[cfg(feature = "gpu-cuda")]
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
use cubecl::wgpu::WgpuRuntime;

const SMC_WIDTH: usize = 11;

#[derive(Debug, Clone)]
pub struct SignalTraceInput {
    pub indicators_feature_major: Vec<f32>,
    pub feature_count: usize,
    pub sample_count: usize,
    pub gene_offsets: Vec<i32>,
    pub gene_indices: Vec<i32>,
    pub gene_weights: Vec<f32>,
    pub long_thresholds: Vec<f32>,
    pub short_thresholds: Vec<f32>,
    pub smc_data: Vec<[i8; SMC_WIDTH]>,
    pub gene_smc_flags: Vec<[i8; SMC_WIDTH]>,
    pub smc_weights: [f32; SMC_WIDTH],
    pub gate_threshold: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SignalTraceOutput {
    pub scores_before_threshold: Vec<f32>,
    pub raw_signals: Vec<i32>,
    pub confidences: Vec<f32>,
    pub smc_scores: Vec<f32>,
    pub smc_gates: Vec<f32>,
    pub final_signals: Vec<i32>,
}

impl SignalTraceInput {
    pub fn gene_count(&self) -> usize {
        self.long_thresholds.len()
    }

    fn validate(&self) -> Result<()> {
        let genes = self.gene_count();
        if self.feature_count == 0 || self.sample_count == 0 || genes == 0 {
            bail!("signal trace requires non-empty features, samples and genes");
        }
        if self.indicators_feature_major.len() != self.feature_count * self.sample_count {
            bail!("signal trace indicator shape mismatch");
        }
        if self.short_thresholds.len() != genes
            || self.gene_smc_flags.len() != genes
            || self.gene_offsets.len() != genes + 1
            || self.gene_offsets.first().copied() != Some(0)
            || self.gene_offsets.last().copied() != Some(self.gene_indices.len() as i32)
            || self.gene_indices.len() != self.gene_weights.len()
            || self.smc_data.len() != self.sample_count
        {
            bail!("signal trace gene/SMC shape mismatch");
        }
        if self
            .gene_indices
            .iter()
            .any(|index| *index < 0 || (*index as usize) >= self.feature_count)
        {
            bail!("signal trace gene index outside feature range");
        }
        if self
            .indicators_feature_major
            .iter()
            .chain(&self.gene_weights)
            .chain(&self.long_thresholds)
            .chain(&self.short_thresholds)
            .chain(&self.smc_weights)
            .any(|value| !value.is_finite())
            || !self.gate_threshold.is_finite()
        {
            bail!("signal trace rejects non-finite values");
        }
        Ok(())
    }
}

#[cube(launch)]
fn signal_trace_kernel(
    indicators: &Array<f32>,
    gene_offsets: &Array<i32>,
    gene_indices: &Array<i32>,
    gene_weights: &Array<f32>,
    long_thresholds: &Array<f32>,
    short_thresholds: &Array<f32>,
    smc_data: &Array<i32>,
    gene_smc_flags: &Array<i32>,
    smc_weights: &Array<f32>,
    scores_before_threshold: &mut Array<f32>,
    raw_signals: &mut Array<i32>,
    confidences: &mut Array<f32>,
    smc_scores: &mut Array<f32>,
    smc_gates: &mut Array<f32>,
    final_signals: &mut Array<i32>,
    sample_count: u32,
    gate_threshold: f32,
) {
    let pos = ABSOLUTE_POS;
    if pos < final_signals.len() {
        let samples = sample_count as usize;
        let gene = pos / samples;
        let sample = pos % samples;
        let start = gene_offsets[gene] as usize;
        let end = gene_offsets[gene + 1] as usize;
        let combined = RuntimeCell::<f32>::new(0.0);
        for term in start..end {
            let feature = gene_indices[term] as usize;
            combined.store(
                combined.read() + gene_weights[term] * indicators[feature * samples + sample],
            );
        }
        let score = combined.read();
        scores_before_threshold[pos] = score;
        let raw = RuntimeCell::<i32>::new(0);
        if score >= long_thresholds[gene] {
            raw.store(1);
        } else if score <= short_thresholds[gene] {
            raw.store(-1);
        }
        let raw_value = raw.read();
        raw_signals[pos] = raw_value;

        let confidence = RuntimeCell::<f32>::new(0.0);
        if raw_value != 0 {
            let gap_raw = long_thresholds[gene] - short_thresholds[gene];
            let gap_abs = RuntimeCell::<f32>::new(gap_raw);
            if gap_raw < 0.0 {
                gap_abs.store(0.0 - gap_raw);
            }
            let gap = RuntimeCell::<f32>::new(gap_abs.read());
            if gap_abs.read() < 1.0e-6 {
                gap.store(1.0e-6);
            }
            let margin = RuntimeCell::<f32>::new(short_thresholds[gene] - score);
            if raw_value == 1 {
                margin.store(score - long_thresholds[gene]);
            }
            let ratio = margin.read() / gap.read();
            confidence.store(ratio);
            if ratio < 0.0 {
                confidence.store(0.0);
            } else if ratio > 1.0 {
                confidence.store(1.0);
            }
        }
        confidences[pos] = confidence.read();

        let flag_base = gene * SMC_WIDTH;
        let smc_base = sample * SMC_WIDTH;
        let active_sum = RuntimeCell::<f32>::new(0.0);
        for index in 0..SMC_WIDTH {
            if gene_smc_flags[flag_base + index] != 0 {
                active_sum.store(active_sum.read() + smc_weights[index]);
            }
        }
        let gate = RuntimeCell::<f32>::new(gate_threshold);
        if active_sum.read() < gate_threshold {
            gate.store(active_sum.read());
        }
        smc_gates[pos] = gate.read();

        let smc_score = RuntimeCell::<f32>::new(0.0);
        if raw_value != 0 {
            for index in 0..SMC_WIDTH {
                if gene_smc_flags[flag_base + index] != 0 {
                    let value = smc_data[smc_base + index];
                    if index == 5 {
                        if value == 1 {
                            smc_score.store(smc_score.read() + smc_weights[index]);
                        }
                    } else if value == raw_value {
                        smc_score.store(smc_score.read() + smc_weights[index]);
                    }
                }
            }
        }
        smc_scores[pos] = smc_score.read();

        let final_signal = RuntimeCell::<i32>::new(0);
        if raw_value != 0 {
            if active_sum.read() <= 0.0 || smc_score.read() >= gate.read() {
                final_signal.store(raw_value);
            }
        }
        final_signals[pos] = final_signal.read();
        if final_signal.read() == 0 {
            confidences[pos] = 0.0;
        }
    }
}

pub fn cpu_signal_trace(input: &SignalTraceInput) -> Result<SignalTraceOutput> {
    input.validate()?;
    let total = input.gene_count() * input.sample_count;
    let mut output = SignalTraceOutput {
        scores_before_threshold: vec![0.0; total],
        raw_signals: vec![0; total],
        confidences: vec![0.0; total],
        smc_scores: vec![0.0; total],
        smc_gates: vec![0.0; total],
        final_signals: vec![0; total],
    };
    for gene in 0..input.gene_count() {
        for sample in 0..input.sample_count {
            let pos = gene * input.sample_count + sample;
            let start = input.gene_offsets[gene] as usize;
            let end = input.gene_offsets[gene + 1] as usize;
            let mut score = 0.0_f32;
            for term in start..end {
                let feature = input.gene_indices[term] as usize;
                score += input.gene_weights[term]
                    * input.indicators_feature_major[feature * input.sample_count + sample];
            }
            output.scores_before_threshold[pos] = score;
            let raw = if score >= input.long_thresholds[gene] {
                1
            } else if score <= input.short_thresholds[gene] {
                -1
            } else {
                0
            };
            output.raw_signals[pos] = raw;
            if raw == 0 {
                continue;
            }
            let gap = (input.long_thresholds[gene] - input.short_thresholds[gene])
                .abs()
                .max(1.0e-6);
            let margin = if raw == 1 {
                score - input.long_thresholds[gene]
            } else {
                input.short_thresholds[gene] - score
            };
            let confidence = (margin / gap).clamp(0.0, 1.0);
            let mut active_sum = 0.0_f32;
            let mut smc_score = 0.0_f32;
            for index in 0..SMC_WIDTH {
                if input.gene_smc_flags[gene][index] == 0 {
                    continue;
                }
                active_sum += input.smc_weights[index];
                let value = input.smc_data[sample][index] as i32;
                if (index == 5 && value == 1) || (index != 5 && value == raw) {
                    smc_score += input.smc_weights[index];
                }
            }
            let gate = active_sum.min(input.gate_threshold);
            output.smc_scores[pos] = smc_score;
            output.smc_gates[pos] = gate;
            if active_sum <= 0.0 || smc_score >= gate {
                output.final_signals[pos] = raw;
                output.confidences[pos] = confidence;
            }
        }
    }
    Ok(output)
}

#[cfg(feature = "gpu-cuda")]
pub fn gpu_signal_trace(input: &SignalTraceInput) -> Result<SignalTraceOutput> {
    let client = crate::cubecl_eval::create_gpu_client(None)?;
    launch_signal_trace::<CudaRuntime>(&client, input)
}

#[cfg(all(feature = "gpu-vulkan", not(feature = "gpu-cuda")))]
pub fn gpu_signal_trace(input: &SignalTraceInput) -> Result<SignalTraceOutput> {
    let client = crate::cubecl_eval::create_gpu_client(None)?;
    launch_signal_trace::<WgpuRuntime>(&client, input)
}

fn launch_signal_trace<R: Runtime>(
    client: &ComputeClient<R>,
    input: &SignalTraceInput,
) -> Result<SignalTraceOutput> {
    input.validate()?;
    let total = input.gene_count() * input.sample_count;
    let smc_data: Vec<i32> = input
        .smc_data
        .iter()
        .flat_map(|row| row.iter().map(|value| *value as i32))
        .collect();
    let flags: Vec<i32> = input
        .gene_smc_flags
        .iter()
        .flat_map(|row| row.iter().map(|value| *value as i32))
        .collect();
    let indicators = client.create_from_slice(f32::as_bytes(&input.indicators_feature_major));
    let offsets = client.create_from_slice(i32::as_bytes(&input.gene_offsets));
    let indices = client.create_from_slice(i32::as_bytes(&input.gene_indices));
    let weights = client.create_from_slice(f32::as_bytes(&input.gene_weights));
    let long = client.create_from_slice(f32::as_bytes(&input.long_thresholds));
    let short = client.create_from_slice(f32::as_bytes(&input.short_thresholds));
    let smc = client.create_from_slice(i32::as_bytes(&smc_data));
    let flag_handle = client.create_from_slice(i32::as_bytes(&flags));
    let smc_weights = client.create_from_slice(f32::as_bytes(&input.smc_weights));
    let score_out = client.empty(total * core::mem::size_of::<f32>());
    let raw_out = client.empty(total * core::mem::size_of::<i32>());
    let confidence_out = client.empty(total * core::mem::size_of::<f32>());
    let smc_score_out = client.empty(total * core::mem::size_of::<f32>());
    let smc_gate_out = client.empty(total * core::mem::size_of::<f32>());
    let final_out = client.empty(total * core::mem::size_of::<i32>());
    let units = client.properties().hardware.max_units_per_cube.clamp(1, 64);
    signal_trace_kernel::launch::<R>(
        client,
        CubeCount::Static((total as u32).div_ceil(units), 1, 1),
        CubeDim::new_1d(units),
        unsafe { ArrayArg::from_raw_parts(indicators, input.indicators_feature_major.len()) },
        unsafe { ArrayArg::from_raw_parts(offsets, input.gene_offsets.len()) },
        unsafe { ArrayArg::from_raw_parts(indices, input.gene_indices.len()) },
        unsafe { ArrayArg::from_raw_parts(weights, input.gene_weights.len()) },
        unsafe { ArrayArg::from_raw_parts(long, input.gene_count()) },
        unsafe { ArrayArg::from_raw_parts(short, input.gene_count()) },
        unsafe { ArrayArg::from_raw_parts(smc, smc_data.len()) },
        unsafe { ArrayArg::from_raw_parts(flag_handle, flags.len()) },
        unsafe { ArrayArg::from_raw_parts(smc_weights, SMC_WIDTH) },
        unsafe { ArrayArg::from_raw_parts(score_out.clone(), total) },
        unsafe { ArrayArg::from_raw_parts(raw_out.clone(), total) },
        unsafe { ArrayArg::from_raw_parts(confidence_out.clone(), total) },
        unsafe { ArrayArg::from_raw_parts(smc_score_out.clone(), total) },
        unsafe { ArrayArg::from_raw_parts(smc_gate_out.clone(), total) },
        unsafe { ArrayArg::from_raw_parts(final_out.clone(), total) },
        input.sample_count as u32,
        input.gate_threshold,
    );

    let score_bytes = client.read_one(score_out).context("read trace scores")?;
    let raw_bytes = client.read_one(raw_out).context("read trace raw signals")?;
    let confidence_bytes = client
        .read_one(confidence_out)
        .context("read trace confidences")?;
    let smc_score_bytes = client
        .read_one(smc_score_out)
        .context("read trace SMC scores")?;
    let smc_gate_bytes = client
        .read_one(smc_gate_out)
        .context("read trace SMC gates")?;
    let final_bytes = client
        .read_one(final_out)
        .context("read trace final signals")?;
    Ok(SignalTraceOutput {
        scores_before_threshold: f32::from_bytes(score_bytes.as_ref()).to_vec(),
        raw_signals: i32::from_bytes(raw_bytes.as_ref()).to_vec(),
        confidences: f32::from_bytes(confidence_bytes.as_ref()).to_vec(),
        smc_scores: f32::from_bytes(smc_score_bytes.as_ref()).to_vec(),
        smc_gates: f32::from_bytes(smc_gate_bytes.as_ref()).to_vec(),
        final_signals: i32::from_bytes(final_bytes.as_ref()).to_vec(),
    })
}

#[cfg(all(test, any(feature = "gpu-cuda", feature = "gpu-vulkan")))]
mod tests {
    use super::*;

    #[test]
    fn direct_gpu_trace_matches_cpu_when_an_adapter_is_available() {
        let input = SignalTraceInput {
            indicators_feature_major: vec![0.0, 0.4, -0.4, 0.2, 0.0, 0.1, -0.1, 0.3],
            feature_count: 2,
            sample_count: 4,
            gene_offsets: vec![0, 2],
            gene_indices: vec![0, 1],
            gene_weights: vec![1.0, 0.5],
            long_thresholds: vec![0.2],
            short_thresholds: vec![-0.2],
            smc_data: vec![[0; SMC_WIDTH]; 4],
            gene_smc_flags: vec![[0; SMC_WIDTH]],
            smc_weights: [0.0; SMC_WIDTH],
            gate_threshold: 0.0,
        };
        let expected = cpu_signal_trace(&input).unwrap();
        let actual = match gpu_signal_trace(&input) {
            Ok(actual) => actual,
            Err(error) => {
                eprintln!("signal trace GPU test skipped: {error:#}");
                return;
            }
        };
        assert_eq!(actual, expected);
    }
}
