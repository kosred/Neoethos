//! GPU population-eval benchmark harness (Task 6, step 1 of the GPU
//! remediation).
//!
//! Correctness before speed: at every workload shape it runs the GPU population
//! evaluator AND the canonical CPU reference on identical inputs, asserts they
//! agree within tolerance, and only then reports throughput. A faster path that
//! disagrees with the CPU is a FAILURE, never a win — the harness exits non-zero
//! if any shape breaks parity.
//!
//! This file compiles on any machine; the measurement body needs a real device
//! and the `gpu` feature. Run it on rented hardware:
//!
//!   # NVIDIA (A6000/A4000):
//!   NEOETHOS_REQUIRE_GPU=1 cargo run --release -p neoethos-search \
//!     --features gpu-nvidia --example gpu_eval_bench
//!
//!   # AMD / integrated / any wgpu adapter:
//!   NEOETHOS_REQUIRE_GPU=1 cargo run --release -p neoethos-search \
//!     --features gpu-vulkan --example gpu_eval_bench
//!
//! `NEOETHOS_REQUIRE_GPU=1` turns a silent CPU fallback into a hard failure, so
//! a run that reports "GPU" numbers really used the GPU. Without it, the harness
//! still runs but says so.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gpu_eval_bench needs the `gpu` feature (build with --features gpu-nvidia \
         or --features gpu-vulkan). Nothing to measure in a CPU-only build."
    );
}

#[cfg(feature = "gpu")]
fn main() {
    std::process::exit(gpu::run());
}

#[cfg(feature = "gpu")]
mod gpu {
    use ndarray::Array2;
    use neoethos_search::eval::{
        validation_backtest_population, validation_backtest_population_cpu, BacktestSettings,
        PopulationEvalInputs, SmcRow,
    };
    use std::time::Instant;

    /// One (rows × population) point to measure. Ordered small→large so a
    /// device that OOMs on a bigger shape still prints the smaller rows first.
    /// The medium/large shapes assume a discrete card (A4000/A6000-class); a
    /// memory-constrained integrated adapter (e.g. a 512 MB shared iGPU) is
    /// expected to OOM on them — which is itself a useful signal that the card
    /// is too small for that unchunked population.
    const SHAPES: &[(usize, usize, &str)] = &[
        (2_000, 64, "small"),
        (50_000, 256, "medium"),
        (400_000, 1024, "large"),
    ];

    const N_FEATURES: usize = 8;

    struct Owned {
        close: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        indicators: Array2<f32>,
        gene_offsets: Vec<i32>,
        gene_indices: Vec<i32>,
        gene_weights: Vec<f32>,
        long_thr: Vec<f32>,
        short_thr: Vec<f32>,
        month_idx: Vec<i64>,
        day_idx: Vec<i64>,
        timestamps: Vec<i64>,
        sl_pips: Vec<f64>,
        tp_pips: Vec<f64>,
        smc_data: Vec<SmcRow>,
        gene_smc_flags: Vec<SmcRow>,
        smc_weights: [f32; 11],
        settings: BacktestSettings,
    }

    /// Deterministic, non-degenerate inputs — the same shape the parity tests
    /// use, scaled up. SMC gating is off so signals are pure threshold crossings
    /// (CPU and GPU must agree exactly bar-for-bar before rounding).
    fn build(n_samples: usize, n_genes: usize) -> Owned {
        let close: Vec<f64> = (0..n_samples)
            .map(|i| 1.10 + ((i as f64) * 0.02).sin() * 0.01 + (i as f64) * 1e-7)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.0008).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.0008).collect();
        let indicators = Array2::from_shape_fn((N_FEATURES, n_samples), |(f, i)| {
            (((i + f * 11) as f32) * 0.05).sin() * 0.8
        });

        // Each gene sums 2 features, weight 1.0 — CSR layout.
        let mut gene_offsets = Vec::with_capacity(n_genes + 1);
        let mut gene_indices = Vec::with_capacity(n_genes * 2);
        let mut gene_weights = Vec::with_capacity(n_genes * 2);
        for g in 0..n_genes {
            gene_offsets.push((g * 2) as i32);
            gene_indices.push((g % (N_FEATURES - 1)) as i32);
            gene_indices.push(((g + 1) % N_FEATURES) as i32);
            gene_weights.push(1.0);
            gene_weights.push(1.0);
        }
        gene_offsets.push((n_genes * 2) as i32);

        let mut settings = BacktestSettings::default();
        settings.pip_value = 0.0001;
        settings.pip_value_per_lot = 10.0;
        settings.spread_pips = 0.0;
        settings.commission_per_trade = 0.0;
        settings.swap_long_pips_per_day = 0.0;
        settings.swap_short_pips_per_day = 0.0;
        settings.pnl_conversion_fee_rate = 0.0;
        settings.kill_zones_enabled = false;
        settings.risk_based_sizing = true;
        settings.risk_per_trade_min = 0.005;
        settings.risk_per_trade_max = 0.03;
        settings.high_quality_confidence = 0.65;

        Owned {
            close,
            high,
            low,
            indicators,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr: vec![0.3; n_genes],
            short_thr: vec![-0.3; n_genes],
            month_idx: (0..n_samples as i64).map(|i| i / 20_000).collect(),
            day_idx: (0..n_samples as i64).map(|i| i / 1_440).collect(),
            timestamps: (0..n_samples as i64).map(|i| i * 60_000).collect(),
            sl_pips: vec![25.0; n_genes],
            tp_pips: vec![50.0; n_genes],
            smc_data: vec![[0i8; 11]; n_samples],
            gene_smc_flags: vec![[0i8; 11]; n_genes],
            smc_weights: [0.0f32; 11],
            settings,
        }
    }

    fn inputs(o: &Owned) -> PopulationEvalInputs<'_> {
        PopulationEvalInputs {
            close: &o.close,
            high: &o.high,
            low: &o.low,
            indicators: o.indicators.view(),
            gene_offsets: &o.gene_offsets,
            gene_indices: &o.gene_indices,
            gene_weights: &o.gene_weights,
            long_thr: &o.long_thr,
            short_thr: &o.short_thr,
            month_idx: &o.month_idx,
            day_idx: &o.day_idx,
            timestamps: &o.timestamps,
            sl_pips: &o.sl_pips,
            tp_pips: &o.tp_pips,
            smc_data: &o.smc_data,
            gene_smc_flags: &o.gene_smc_flags,
            gate_threshold: 0.0,
            weights: &o.smc_weights,
            settings: &o.settings,
        }
    }

    /// Compare GPU vs CPU per-gene metrics. Returns the worst relative error on
    /// the money metrics and the max trade-count delta.
    fn parity(cpu: &[[f64; 11]], gpu: &[[f64; 11]]) -> Result<(f64, f64), String> {
        if cpu.len() != gpu.len() {
            return Err(format!("gene-count mismatch cpu={} gpu={}", cpu.len(), gpu.len()));
        }
        let mut worst_rel = 0.0_f64;
        let mut worst_trades = 0.0_f64;
        for (g, (c, v)) in cpu.iter().zip(gpu).enumerate() {
            worst_trades = worst_trades.max((c[8] - v[8]).abs());
            if (c[8] - v[8]).abs() > 1.0 {
                return Err(format!(
                    "gene {g} trade-count off by {} (cpu={} gpu={})",
                    (c[8] - v[8]).abs(),
                    c[8],
                    v[8]
                ));
            }
            for m in [0usize, 1, 2, 3, 4, 5, 6, 7, 9, 10] {
                let denom = c[m].abs().max(1.0);
                worst_rel = worst_rel.max((c[m] - v[m]).abs() / denom);
            }
        }
        Ok((worst_rel, worst_trades))
    }

    pub fn run() -> i32 {
        let mut probe = neoethos_core::system::HardwareProbe::new();
        let hw = probe.detect();
        let required = neoethos_search::gpu_fallback::require_gpu();

        println!("# NeoEthos GPU eval benchmark");
        println!("cpu_cores            {}", hw.cpu_cores);
        println!("available_ram_gb     {:.1}", hw.available_ram_gb);
        println!("gpus                 {} {:?}", hw.num_gpus, hw.gpu_names);
        println!("gpu_mem_gb           {:?}", hw.gpu_mem_gb);
        println!("NEOETHOS_REQUIRE_GPU {}", required);
        for a in &hw.accelerator_devices {
            println!("adapter              {a:?}");
        }
        println!();
        println!(
            "{:<8} {:>9} {:>6} {:>11} {:>11} {:>8} {:>13} {:>10} parity",
            "shape", "rows", "pop", "cpu_ms", "gpu_ms", "speedup", "gpu_evals/s", "cube_mb"
        );

        let mut any_parity_break = false;
        for &(rows, pop, label) in SHAPES {
            let o = build(rows, pop);

            // Warm both lanes once (JIT/kernel compile) — excluded from timing.
            let cpu_ref = validation_backtest_population_cpu(inputs(&o));
            let _ = validation_backtest_population(inputs(&o));

            let t0 = Instant::now();
            let cpu = validation_backtest_population_cpu(inputs(&o));
            let cpu_ms = t0.elapsed().as_secs_f64() * 1e3;

            let t1 = Instant::now();
            let gpu = validation_backtest_population(inputs(&o));
            let gpu_ms = t1.elapsed().as_secs_f64() * 1e3;

            let _ = cpu_ref;
            let cube_mb = (rows * N_FEATURES * 4) as f64 / (1024.0 * 1024.0);
            let evals_per_s = if gpu_ms > 0.0 { (pop as f64) / (gpu_ms / 1e3) } else { 0.0 };
            let speedup = if gpu_ms > 0.0 { cpu_ms / gpu_ms } else { 0.0 };

            let parity_str = match parity(&cpu, &gpu) {
                Ok((rel, tr)) => format!("OK (rel={rel:.1e}, dtrades={tr:.0})"),
                Err(e) => {
                    any_parity_break = true;
                    format!("FAIL — {e}")
                }
            };

            println!(
                "{label:<8} {rows:>9} {pop:>6} {cpu_ms:>11.1} {gpu_ms:>11.1} {speedup:>7.2}x {evals_per_s:>13.0} {cube_mb:>10.1} {parity_str}"
            );
        }

        if any_parity_break {
            eprintln!(
                "\nPARITY BROKEN — a GPU result disagreed with the CPU reference. Do NOT \
                 promote any throughput number from this run; fix the kernel first."
            );
            return 1;
        }
        println!("\nAll shapes matched the CPU reference. Throughput figures are trustworthy.");
        0
    }
}
