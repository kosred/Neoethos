from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)

mod_path = Path("crates/neoethos-search/src/gpu_native/mod.rs")
mods = mod_path.read_text(encoding="utf-8")
mods = replace_once(
    mods,
    "pub mod parity_hierarchy;\n",
    "pub mod parity_hierarchy;\npub mod population_fixture;\n",
    "population fixture module export",
)
mod_path.write_text(mods, encoding="utf-8")

proto_path = Path("crates/neoethos-search/src/gpu_native/prototype_a.rs")
proto = proto_path.read_text(encoding="utf-8")
proto = replace_once(
    proto,
    """            h2d_bytes: self
                .resident_upload_bytes
                .saturating_add(self.streamed_dataset_upload_bytes)
                .saturating_add(self.gene_upload_bytes),
""",
    """            h2d_bytes: self
                .resident_upload_bytes
                .saturating_add(self.streamed_dataset_upload_bytes)
                .saturating_add(self.gene_upload_bytes)
                .saturating_add(if self.chained_reuploads > 0 {
                    self.full_readback_bytes
                } else {
                    0
                }),
""",
    "Prototype A chained reupload bytes",
)
proto_path.write_text(proto, encoding="utf-8")

bench_path = Path("crates/neoethos-cli/src/gpu_bench.rs")
bench = bench_path.read_text(encoding="utf-8")
bench = replace_once(
    bench,
    "    let identity = identity(args)?;\n    let sweep = sweep(args)?;\n\n",
    "    let identity = identity(args)?;\n    let sweep = sweep(args)?;\n\n    if has_flag(args, \"--execute-tiny\") {\n        return run_tiny_population_benchmark(args, output, identity, sweep);\n    }\n\n",
    "executable tiny benchmark dispatch",
)
marker = "#[derive(Debug, Serialize)]\nstruct BenchmarkPlan {\n"
if marker not in bench:
    raise RuntimeError("BenchmarkPlan marker missing")
function = '''fn run_tiny_population_benchmark(
    args: &[String],
    output: PathBuf,
    identity: BenchmarkIdentity,
    requested_sweep: SweepPoint,
) -> Result<()> {
    use neoethos_search::backend::EvaluationBackend;
    use neoethos_search::gpu_native::capability::{
        GpuCapabilityManifest, PipelineStage, gpu_pipeline_preflight,
    };
    use neoethos_search::gpu_native::cpu_strategy::CpuStrategyAuditContext;
    use neoethos_search::gpu_native::population_fixture::TinyPopulationFixture;
    use neoethos_search::gpu_native::prototype_a::{
        disable_prototype_a_telemetry, prototype_a_status, prototype_a_telemetry,
        reset_prototype_a_telemetry,
    };
    use std::time::Instant;

    if identity.prototype != PrototypeId::A {
        anyhow::bail!("--execute-tiny currently executes Prototype A only");
    }
    if identity.fixture != FixtureMode::Tiny {
        anyhow::bail!("--execute-tiny requires --fixture tiny");
    }
    let backend = EvaluationBackend::parse(&identity.backend)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if backend != EvaluationBackend::GPU_REQUIRED {
        anyhow::bail!(
            "--execute-tiny requires --backend gpu_required so a CPU fallback cannot contaminate the measurement"
        );
    }
    gpu_pipeline_preflight(
        backend,
        &GpuCapabilityManifest::stage1_baseline(),
        &[PipelineStage::PopulationEvaluation],
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    let fixture = TinyPopulationFixture::new(
        requested_sweep.population,
        requested_sweep.bar_count,
        requested_sweep.feature_count,
    );
    let sweep = SweepPoint {
        population: fixture.population(),
        batch_size: requested_sweep.batch_size,
        bar_count: fixture.bars(),
        feature_count: fixture.features(),
        scenario_count: 1,
        calendar_days: requested_sweep.calendar_days,
    };

    let reference_audit = CpuStrategyAuditContext::validation_reference(0x4350_5552_4546);
    let reference = fixture
        .evaluate(EvaluationBackend::CPU_CANONICAL, &reference_audit)
        .map_err(anyhow::Error::msg)?;

    let warmups = parse_usize(args, "--warmups", 2)?;
    let repetitions = parse_usize(args, "--repetitions", 5)?.max(1);
    let transfer_instrumented = identity.pass == BenchmarkPass::Diagnostics;
    disable_prototype_a_telemetry();
    for warmup in 0..warmups {
        let audit = CpuStrategyAuditContext::production(0x5741_524d_0000 + warmup as u64);
        fixture
            .evaluate(backend, &audit)
            .map_err(anyhow::Error::msg)?;
        audit
            .snapshot()
            .assert_zero_executed()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }

    // Warm-up populates resident device buffers. Transfer atomics are enabled only
    // for the diagnostic pass so clean/profiler timing is not contaminated.
    if transfer_instrumented {
        reset_prototype_a_telemetry();
    } else {
        disable_prototype_a_telemetry();
    }
    let mut wall_samples = Vec::with_capacity(repetitions);
    let mut candidate = Vec::new();
    for repetition in 0..repetitions {
        let audit = CpuStrategyAuditContext::production(0x4d45_4153_0000 + repetition as u64);
        let started = Instant::now();
        candidate = fixture
            .evaluate(backend, &audit)
            .map_err(anyhow::Error::msg)?;
        wall_samples.push(started.elapsed().as_secs_f64());
        audit
            .snapshot()
            .assert_zero_executed()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }

    let parity_report = TinyPopulationFixture::compare_final_metrics(&reference, &candidate);
    let parity = match parity_report.first_divergence {
        None => ParityStatus {
            matched: true,
            first_divergent_level: None,
            detail: None,
        },
        Some(divergence) => ParityStatus {
            matched: false,
            first_divergent_level: Some(divergence.level as u8),
            detail: Some(format!(
                "{}[{}] expected={} actual={}",
                divergence.field, divergence.index, divergence.expected, divergence.actual
            )),
        },
    };
    let total_wall_seconds = DistributionSummary::from_samples(&wall_samples)
        .context("tiny benchmark produced no finite wall-time samples")?;
    let elapsed = total_wall_seconds.median.max(f64::EPSILON);
    let total_trades: f64 = candidate.iter().map(|metrics| metrics[8].max(0.0)).sum();
    let telemetry = prototype_a_telemetry();
    let transfers = telemetry.transfer_snapshot();
    let coverage = CapabilityCoverage {
        total_candidates: fixture.population(),
        supported_candidates: fixture.population(),
        unsupported_candidates: 0,
        unsupported_reasons: BTreeMap::new(),
    };
    let throughput = ThroughputMetrics {
        candidates_per_second: Some(fixture.population() as f64 / elapsed),
        candidate_bars_per_second: Some(fixture.candidate_bars() as f64 / elapsed),
        trades_per_second: (total_trades > 0.0).then_some(total_trades / elapsed),
        peak_vram_bytes: parse_optional_u64(args, "--peak-vram-bytes")?,
        event_density: (fixture.candidate_bars() > 0)
            .then_some(total_trades / fixture.candidate_bars() as f64),
        hold_bars: None,
    };
    let mut notes = vec![
        "Workload execution occurred inside neoethos-cli bench; CPU reference parity ran before timing.".to_string(),
        "Every measured GPU-required work unit asserted zero executed CPU strategy operations.".to_string(),
        format!(
            "CubeCL calls={} resident_hits={} resident_misses={} streamed_dataset_bytes={}",
            telemetry.gpu_calls,
            telemetry.resident_cache_hits,
            telemetry.resident_cache_misses,
            telemetry.streamed_dataset_upload_bytes,
        ),
    ];
    if transfer_instrumented {
        if telemetry.satisfies_no_dense_roundtrip() {
            notes.push("No dense signal/confidence D2H roundtrip was observed.".to_string());
        } else {
            notes.push(
                "Dense signal/confidence readback or chained reupload was observed; Prototype A residency acceptance is not satisfied on this run."
                    .to_string(),
            );
        }
    } else {
        notes.push(
            "Transfer counters were intentionally disabled for this non-diagnostic pass; zero transfer fields mean not collected, not zero physical traffic."
                .to_string(),
        );
    }
    notes.push(
        "Engine status remains not_benchmarked until the attributed discrete-NVIDIA matrix is complete."
            .to_string(),
    );

    let report = BenchmarkReport::new(
        identity,
        hardware(args),
        prototype_a_status(),
        warmups,
        repetitions,
        sweep,
        total_wall_seconds,
        BTreeMap::new(),
        throughput,
        transfers,
        coverage,
        parity,
        notes,
    );
    report.write_json(&output)?;
    println!("Executable tiny GPU benchmark written to {}", output.display());
    Ok(())
}
'''
bench = bench.replace(marker, function + "\n" + marker, 1)
bench_path.write_text(bench, encoding="utf-8")
print("added deterministic engine-only executable tiny benchmark")
