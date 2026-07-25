use anyhow::{Context, Result};
use neoethos_search::gpu_native::benchmark::{
    BenchmarkIdentity, BenchmarkPass, BenchmarkReport, CapabilityCoverage, DistributionSummary,
    FixtureMode, HardwareMetadata, ParityStatus, PrototypeId, SweepPoint, ThroughputMetrics,
};
use neoethos_search::gpu_native::engine::{EngineStatus, TransferSnapshot};
use neoethos_search::gpu_native::semantics::{
    DISCOVERY_SEMANTICS_VERSION, TRADING_SEMANTICS_VERSION, discovery_semantics_hash,
    semantics_hash_hex, trading_semantics_hash,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<()> {
    if args.iter().any(|arg| arg == "--execute-snapshot") {
        return crate::gpu_bench_snapshot::run(args);
    }
    let output = flag(args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("cache/gpu-bench/report.json"));
    let identity = identity(args)?;
    let sweep = sweep(args)?;

    if has_flag(args, "--execute-tiny") {
        return run_tiny_population_benchmark(args, output, identity, sweep);
    }

    if has_flag(args, "--dry-run") || flag(args, "--wall-samples").is_none() {
        let plan = BenchmarkPlan {
            schema_version: 1,
            identity,
            sweep,
            warmup_count: parse_usize(args, "--warmups", 2)?,
            repetitions: parse_usize(args, "--repetitions", 5)?,
            trading_semantics_version: TRADING_SEMANTICS_VERSION,
            trading_semantics_hash: semantics_hash_hex(trading_semantics_hash()),
            discovery_semantics_version: DISCOVERY_SEMANTICS_VERSION,
            discovery_semantics_hash: semantics_hash_hex(discovery_semantics_hash()),
            required_passes: vec![
                BenchmarkPass::CleanTiming,
                BenchmarkPass::Diagnostics,
                BenchmarkPass::NsightSystems,
                BenchmarkPass::NsightCompute,
            ],
            notes: vec![
                "This is a benchmark execution plan, not a performance result.".to_string(),
                "Clean timing, diagnostics and profiler passes must run separately.".to_string(),
            ],
        };
        write_json(&output, &plan)?;
        println!("GPU benchmark plan written to {}", output.display());
        return Ok(());
    }

    let wall_samples = parse_f64_csv(
        &flag(args, "--wall-samples").expect("checked above"),
        "--wall-samples",
    )?;
    let total_wall_seconds = DistributionSummary::from_samples(&wall_samples)
        .context("--wall-samples must contain finite values")?;
    let stage_wall_seconds = parse_stage_samples(args)?;
    let coverage = coverage(args)?;
    let hardware = hardware(args);
    let transfers = transfer_snapshot(args)?;
    let parity = parity_status(args)?;
    let engine_status = parse_engine_status(flag(args, "--engine-status").as_deref())?;

    let elapsed = total_wall_seconds.median.max(f64::EPSILON);
    let candidates = parse_usize(args, "--candidates", sweep.population)?;
    let trades = parse_usize(args, "--trades", 0)?;
    let throughput = ThroughputMetrics {
        candidates_per_second: Some(candidates as f64 / elapsed),
        candidate_bars_per_second: Some(candidates as f64 * sweep.bar_count as f64 / elapsed),
        trades_per_second: (trades > 0).then_some(trades as f64 / elapsed),
        peak_vram_bytes: parse_optional_u64(args, "--peak-vram-bytes")?,
        event_density: parse_optional_f64(args, "--event-density")?,
        hold_bars: flag(args, "--hold-samples")
            .map(|raw| parse_f64_csv(&raw, "--hold-samples"))
            .transpose()?
            .as_deref()
            .and_then(DistributionSummary::from_samples),
    };

    let report = BenchmarkReport::new(
        identity,
        hardware,
        engine_status,
        parse_usize(args, "--warmups", 2)?,
        wall_samples.len(),
        sweep,
        total_wall_seconds,
        stage_wall_seconds,
        throughput,
        transfers,
        coverage,
        parity,
        vec!["Report collated by neoethos-cli bench; workload execution is external and separately attributed.".to_string()],
    );
    report.write_json(&output)?;
    println!("GPU benchmark report written to {}", output.display());
    Ok(())
}

fn run_tiny_population_benchmark(
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
    println!(
        "Executable tiny GPU benchmark written to {}",
        output.display()
    );
    Ok(())
}

#[derive(Debug, Serialize)]
struct BenchmarkPlan {
    schema_version: u32,
    identity: BenchmarkIdentity,
    sweep: SweepPoint,
    warmup_count: usize,
    repetitions: usize,
    trading_semantics_version: u32,
    trading_semantics_hash: String,
    discovery_semantics_version: u32,
    discovery_semantics_hash: String,
    required_passes: Vec<BenchmarkPass>,
    notes: Vec<String>,
}

fn identity(args: &[String]) -> Result<BenchmarkIdentity> {
    Ok(BenchmarkIdentity {
        git_sha: flag(args, "--git-sha").unwrap_or_else(|| "unresolved".to_string()),
        baseline_sha: flag(args, "--baseline-sha")
            .unwrap_or_else(|| "2be1408ee3986026fdbb2a5a74aaaf6ac67e5209".to_string()),
        dataset_hash: flag(args, "--dataset-hash").unwrap_or_else(|| "unresolved".to_string()),
        config_hash: flag(args, "--config-hash").unwrap_or_else(|| "unresolved".to_string()),
        seed: parse_u64(args, "--seed", 0)?,
        timeframe: flag(args, "--timeframe").unwrap_or_else(|| "M1".to_string()),
        backend: flag(args, "--backend").unwrap_or_else(|| "unresolved".to_string()),
        prototype: parse_prototype(flag(args, "--prototype").as_deref())?,
        fixture: parse_fixture(flag(args, "--fixture").as_deref())?,
        pass: parse_pass(flag(args, "--pass").as_deref())?,
    })
}

fn sweep(args: &[String]) -> Result<SweepPoint> {
    Ok(SweepPoint {
        population: parse_usize(args, "--population", 4000)?,
        batch_size: parse_usize(args, "--batch-size", 256)?,
        bar_count: parse_usize(args, "--bars", 0)?,
        feature_count: parse_usize(args, "--features", 0)?,
        scenario_count: parse_usize(args, "--scenarios", 1)?,
        calendar_days: parse_optional_u64(args, "--calendar-days")?.map(|value| value as u32),
    })
}

fn hardware(args: &[String]) -> HardwareMetadata {
    HardwareMetadata {
        cpu: flag(args, "--cpu").unwrap_or_else(|| "unresolved".to_string()),
        ram_bytes: flag(args, "--ram-bytes").and_then(|value| value.parse().ok()),
        gpu: flag(args, "--gpu"),
        device_class: flag(args, "--device-class"),
        driver_version: flag(args, "--driver"),
        cuda_runtime_version: flag(args, "--cuda-runtime"),
        cuda_toolkit_version: flag(args, "--cuda-toolkit"),
        gpu_clock_mhz: flag(args, "--gpu-clock-mhz").and_then(|value| value.parse().ok()),
        power_limit_watts: flag(args, "--power-limit-watts").and_then(|value| value.parse().ok()),
        temperature_celsius: flag(args, "--temperature-celsius")
            .and_then(|value| value.parse().ok()),
        dedicated_vram_bytes: flag(args, "--vram-bytes").and_then(|value| value.parse().ok()),
        cuda_occupancy: flag(args, "--cuda-occupancy").and_then(|value| value.parse().ok()),
    }
}

fn coverage(args: &[String]) -> Result<CapabilityCoverage> {
    let total = parse_usize(
        args,
        "--coverage-total",
        parse_usize(args, "--candidates", 0)?,
    )?;
    let supported = parse_usize(args, "--coverage-supported", total)?;
    if supported > total {
        anyhow::bail!("--coverage-supported cannot exceed --coverage-total");
    }
    Ok(CapabilityCoverage {
        total_candidates: total,
        supported_candidates: supported,
        unsupported_candidates: total - supported,
        unsupported_reasons: BTreeMap::new(),
    })
}

fn transfer_snapshot(args: &[String]) -> Result<TransferSnapshot> {
    Ok(TransferSnapshot {
        dataset_uploads: parse_u64(args, "--dataset-uploads", 0)?,
        gene_uploads: parse_u64(args, "--gene-uploads", 0)?,
        scenario_uploads: parse_u64(args, "--scenario-uploads", 0)?,
        full_d2h_readbacks: parse_u64(args, "--full-readbacks", 0)?,
        compact_d2h_readbacks: parse_u64(args, "--compact-readbacks", 0)?,
        chained_reuploads: parse_u64(args, "--chained-reuploads", 0)?,
        synchronization_events: parse_u64(args, "--sync-events", 0)?,
        workspace_allocations: parse_u64(args, "--workspace-allocations", 0)?,
        h2d_bytes: parse_u64(args, "--h2d-bytes", 0)?,
        d2h_bytes: parse_u64(args, "--d2h-bytes", 0)?,
    })
}

fn parity_status(args: &[String]) -> Result<ParityStatus> {
    let raw = flag(args, "--parity").unwrap_or_else(|| "unknown".to_string());
    match raw.trim().to_ascii_lowercase().as_str() {
        "matched" | "pass" | "true" => Ok(ParityStatus {
            matched: true,
            first_divergent_level: None,
            detail: None,
        }),
        "unknown" => Ok(ParityStatus {
            matched: false,
            first_divergent_level: None,
            detail: Some("parity was not supplied".to_string()),
        }),
        "failed" | "fail" | "false" => Ok(ParityStatus {
            matched: false,
            first_divergent_level: flag(args, "--parity-level")
                .and_then(|value| value.parse().ok()),
            detail: flag(args, "--parity-detail"),
        }),
        other => anyhow::bail!("unknown --parity value `{other}`"),
    }
}

fn parse_stage_samples(args: &[String]) -> Result<BTreeMap<String, DistributionSummary>> {
    let mut output = BTreeMap::new();
    let Some(raw) = flag(args, "--stage-samples") else {
        return Ok(output);
    };
    for stage in raw.split(';').filter(|value| !value.trim().is_empty()) {
        let (name, samples) = stage
            .split_once('=')
            .with_context(|| format!("invalid stage sample `{stage}`; expected name=v1,v2"))?;
        let samples = parse_f64_csv(samples, "--stage-samples")?;
        let summary = DistributionSummary::from_samples(&samples)
            .with_context(|| format!("stage `{name}` contains invalid samples"))?;
        output.insert(name.trim().to_string(), summary);
    }
    Ok(output)
}

fn parse_prototype(raw: Option<&str>) -> Result<PrototypeId> {
    match raw.unwrap_or("legacy").trim().to_ascii_lowercase().as_str() {
        "legacy" => Ok(PrototypeId::Legacy),
        "a" | "prototype_a" | "prototype-a" => Ok(PrototypeId::A),
        "b" | "prototype_b" | "prototype-b" => Ok(PrototypeId::B),
        "c" | "prototype_c" | "prototype-c" => Ok(PrototypeId::C),
        other => anyhow::bail!("unknown --prototype `{other}`"),
    }
}

fn parse_fixture(raw: Option<&str>) -> Result<FixtureMode> {
    match raw.unwrap_or("tiny").trim().to_ascii_lowercase().as_str() {
        "tiny" => Ok(FixtureMode::Tiny),
        "snapshot" => Ok(FixtureMode::Snapshot),
        other => anyhow::bail!("unknown --fixture `{other}`"),
    }
}

fn parse_pass(raw: Option<&str>) -> Result<BenchmarkPass> {
    match raw
        .unwrap_or("clean_timing")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "clean" | "clean_timing" | "clean-timing" => Ok(BenchmarkPass::CleanTiming),
        "diagnostics" => Ok(BenchmarkPass::Diagnostics),
        "nsight_systems" | "nsight-systems" | "nsys" => Ok(BenchmarkPass::NsightSystems),
        "nsight_compute" | "nsight-compute" | "ncu" => Ok(BenchmarkPass::NsightCompute),
        other => anyhow::bail!("unknown --pass `{other}`"),
    }
}

fn parse_engine_status(raw: Option<&str>) -> Result<EngineStatus> {
    match raw
        .unwrap_or("not_benchmarked")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "ready" => Ok(EngineStatus::Ready),
        "not_implemented" | "not-implemented" => Ok(EngineStatus::NotImplemented),
        "not_benchmarked" | "not-benchmarked" => Ok(EngineStatus::NotBenchmarked),
        "unsupported" | "unsupported_capability" => Ok(EngineStatus::UnsupportedCapability),
        other => anyhow::bail!("unknown --engine-status `{other}`"),
    }
}

fn parse_f64_csv(raw: &str, name: &str) -> Result<Vec<f64>> {
    let values: Vec<f64> = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<f64>()
                .with_context(|| format!("invalid {name} sample `{value}`"))
        })
        .collect::<Result<_>>()?;
    if values.is_empty() {
        anyhow::bail!("{name} requires at least one sample");
    }
    Ok(values)
}

fn parse_usize(args: &[String], name: &str, default: usize) -> Result<usize> {
    flag(args, name)
        .map(|value| {
            value
                .parse()
                .with_context(|| format!("invalid {name} `{value}`"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_u64(args: &[String], name: &str, default: u64) -> Result<u64> {
    flag(args, name)
        .map(|value| {
            value
                .parse()
                .with_context(|| format!("invalid {name} `{value}`"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_optional_u64(args: &[String], name: &str) -> Result<Option<u64>> {
    flag(args, name)
        .map(|value| {
            value
                .parse()
                .with_context(|| format!("invalid {name} `{value}`"))
        })
        .transpose()
}

fn parse_optional_f64(args: &[String], name: &str) -> Result<Option<f64>> {
    flag(args, name)
        .map(|value| {
            value
                .parse()
                .with_context(|| format!("invalid {name} `{value}`"))
        })
        .transpose()
}

fn flag(args: &[String], name: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            return iter.next().cloned();
        }
    }
    None
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn write_json(path: &PathBuf, value: &impl Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_run_plan_uses_versioned_semantics() {
        let args = vec!["--dry-run".to_string()];
        let identity = identity(&args).unwrap();
        assert_eq!(identity.prototype, PrototypeId::Legacy);
        assert_eq!(TRADING_SEMANTICS_VERSION, 1);
        assert_eq!(DISCOVERY_SEMANTICS_VERSION, 1);
    }

    #[test]
    fn invalid_coverage_is_rejected() {
        let args = vec![
            "--coverage-total".to_string(),
            "10".to_string(),
            "--coverage-supported".to_string(),
            "11".to_string(),
        ];
        assert!(coverage(&args).is_err());
    }
}
