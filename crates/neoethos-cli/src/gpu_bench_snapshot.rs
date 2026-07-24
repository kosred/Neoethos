use anyhow::{Context, Result};
use neoethos_search::backend::EvaluationBackend;
use neoethos_search::gpu_native::benchmark::{
    BenchmarkIdentity, BenchmarkPass, BenchmarkReport, CapabilityCoverage, DistributionSummary,
    FixtureMode, HardwareMetadata, ParityStatus, PrototypeId, SweepPoint, ThroughputMetrics,
};
use neoethos_search::gpu_native::capability::{
    GpuCapabilityManifest, PipelineStage, gpu_pipeline_preflight,
};
use neoethos_search::gpu_native::cpu_strategy::CpuStrategyAuditContext;
use neoethos_search::gpu_native::prototype_a::{
    disable_prototype_a_telemetry, prototype_a_status, prototype_a_telemetry,
    reset_prototype_a_telemetry,
};
use neoethos_search::gpu_native::snapshot_fixture::SnapshotPopulationFixture;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

pub fn run(args: &[String]) -> Result<()> {
    let snapshot_path = PathBuf::from(
        flag(args, "--snapshot").context("--execute-snapshot requires --snapshot <file>")?,
    );
    let output = PathBuf::from(
        flag(args, "--out").unwrap_or_else(|| "cache/gpu-bench/snapshot-report.json".into()),
    );
    let fixture = SnapshotPopulationFixture::from_json_path(&snapshot_path)
        .map_err(anyhow::Error::msg)?;
    let prototype = parse_prototype(flag(args, "--prototype").as_deref())?;
    if prototype != PrototypeId::A {
        anyhow::bail!("full snapshot execution currently supports Prototype A only");
    }
    let backend_raw = flag(args, "--backend").unwrap_or_else(|| "gpu_required".into());
    let backend = EvaluationBackend::parse(&backend_raw)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if backend != EvaluationBackend::GPU_REQUIRED {
        anyhow::bail!("snapshot timing requires --backend gpu_required");
    }
    gpu_pipeline_preflight(
        backend,
        &GpuCapabilityManifest::stage1_baseline(),
        &[PipelineStage::PopulationEvaluation],
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    let reference_audit = CpuStrategyAuditContext::validation_reference(0x5350_4e41_5052);
    let reference = fixture
        .evaluate(EvaluationBackend::CPU_CANONICAL, &reference_audit)
        .map_err(anyhow::Error::msg)?;

    let warmups = parse_usize(args, "--warmups", 2)?;
    let repetitions = parse_usize(args, "--repetitions", 7)?.max(1);
    let pass = parse_pass(flag(args, "--pass").as_deref())?;
    let transfer_instrumented = pass == BenchmarkPass::Diagnostics;
    disable_prototype_a_telemetry();
    for warmup in 0..warmups {
        let audit = CpuStrategyAuditContext::production(0x5350_5741_0000 + warmup as u64);
        fixture.evaluate(backend, &audit).map_err(anyhow::Error::msg)?;
        audit
            .snapshot()
            .assert_zero_executed()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }
    if transfer_instrumented {
        reset_prototype_a_telemetry();
    } else {
        disable_prototype_a_telemetry();
    }

    let mut wall_samples = Vec::with_capacity(repetitions);
    let mut candidate = Vec::new();
    for repetition in 0..repetitions {
        let audit = CpuStrategyAuditContext::production(0x5350_4d45_0000 + repetition as u64);
        let started = Instant::now();
        candidate = fixture.evaluate(backend, &audit).map_err(anyhow::Error::msg)?;
        wall_samples.push(started.elapsed().as_secs_f64());
        audit
            .snapshot()
            .assert_zero_executed()
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    }

    let comparison = SnapshotPopulationFixture::compare_final_metrics(&reference, &candidate);
    let parity = match comparison.first_divergence {
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
    let distribution = DistributionSummary::from_samples(&wall_samples)
        .context("snapshot benchmark produced no finite wall-time samples")?;
    let elapsed = distribution.median.max(f64::EPSILON);
    let total_trades: f64 = candidate.iter().map(|metrics| metrics[8].max(0.0)).sum();
    let telemetry = prototype_a_telemetry();
    let transfers = telemetry.transfer_snapshot();
    let population = fixture.population();
    let bars = fixture.bars();
    let identity = BenchmarkIdentity {
        git_sha: flag(args, "--git-sha").unwrap_or_else(|| "unresolved".into()),
        baseline_sha: flag(args, "--baseline-sha")
            .unwrap_or_else(|| "2be1408ee3986026fdbb2a5a74aaaf6ac67e5209".into()),
        dataset_hash: flag(args, "--dataset-hash").unwrap_or_else(|| "unresolved".into()),
        config_hash: flag(args, "--config-hash").unwrap_or_else(|| "unresolved".into()),
        seed: parse_u64(args, "--seed", 0)?,
        timeframe: fixture.timeframe().to_string(),
        backend: backend_raw,
        prototype,
        fixture: FixtureMode::Snapshot,
        pass,
    };
    let report = BenchmarkReport::new(
        identity,
        hardware(args),
        prototype_a_status(),
        warmups,
        repetitions,
        SweepPoint {
            population,
            batch_size: parse_usize(args, "--batch-size", population)?,
            bar_count: bars,
            feature_count: fixture.features(),
            scenario_count: 1,
            calendar_days: None,
        },
        distribution,
        BTreeMap::new(),
        ThroughputMetrics {
            candidates_per_second: Some(population as f64 / elapsed),
            candidate_bars_per_second: Some(fixture.candidate_bars() as f64 / elapsed),
            trades_per_second: (total_trades > 0.0).then_some(total_trades / elapsed),
            peak_vram_bytes: optional_u64(args, "--peak-vram-bytes")?,
            event_density: (fixture.candidate_bars() > 0)
                .then_some(total_trades / fixture.candidate_bars() as f64),
            hold_bars: None,
        },
        transfers,
        CapabilityCoverage {
            total_candidates: population,
            supported_candidates: population,
            unsupported_candidates: 0,
            unsupported_reasons: BTreeMap::new(),
        },
        parity,
        vec![
            format!("Versioned real-data snapshot: {}", fixture.source_description()),
            "CPU reference parity ran before timing; measured work units asserted zero executed CPU strategy operations.".into(),
            if transfer_instrumented {
                "Transfer telemetry was enabled for the diagnostics pass.".into()
            } else {
                "Transfer telemetry was disabled to avoid contaminating timing/profiling.".into()
            },
        ],
    );
    report.write_json(&output)?;
    println!("Executable snapshot GPU benchmark written to {}", output.display());
    Ok(())
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn parse_usize(args: &[String], name: &str, default: usize) -> Result<usize> {
    flag(args, name)
        .map(|value| value.parse::<usize>().with_context(|| format!("parse {name}")))
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_u64(args: &[String], name: &str, default: u64) -> Result<u64> {
    flag(args, name)
        .map(|value| value.parse::<u64>().with_context(|| format!("parse {name}")))
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn optional_u64(args: &[String], name: &str) -> Result<Option<u64>> {
    flag(args, name)
        .map(|value| value.parse::<u64>().with_context(|| format!("parse {name}")))
        .transpose()
}

fn parse_prototype(raw: Option<&str>) -> Result<PrototypeId> {
    match raw.unwrap_or("a").trim().to_ascii_lowercase().as_str() {
        "a" => Ok(PrototypeId::A),
        "b" => Ok(PrototypeId::B),
        "c" => Ok(PrototypeId::C),
        other => anyhow::bail!("unsupported prototype {other}"),
    }
}

fn parse_pass(raw: Option<&str>) -> Result<BenchmarkPass> {
    match raw.unwrap_or("clean_timing").trim().to_ascii_lowercase().as_str() {
        "clean_timing" | "clean" => Ok(BenchmarkPass::CleanTiming),
        "diagnostics" => Ok(BenchmarkPass::Diagnostics),
        "nsight_systems" | "nsys" => Ok(BenchmarkPass::NsightSystems),
        "nsight_compute" | "ncu" => Ok(BenchmarkPass::NsightCompute),
        other => anyhow::bail!("unsupported benchmark pass {other}"),
    }
}

fn hardware(args: &[String]) -> HardwareMetadata {
    HardwareMetadata {
        cpu: flag(args, "--cpu").unwrap_or_else(|| std::env::consts::ARCH.into()),
        ram_bytes: flag(args, "--ram-bytes").and_then(|value| value.parse().ok()),
        gpu: flag(args, "--gpu"),
        device_class: flag(args, "--device-class"),
        driver_version: flag(args, "--driver-version"),
        cuda_runtime_version: flag(args, "--cuda-runtime-version"),
        cuda_toolkit_version: flag(args, "--cuda-toolkit-version"),
        gpu_clock_mhz: flag(args, "--gpu-clock-mhz").and_then(|value| value.parse().ok()),
        power_limit_watts: flag(args, "--power-limit-watts").and_then(|value| value.parse().ok()),
        temperature_celsius: flag(args, "--temperature-celsius").and_then(|value| value.parse().ok()),
        dedicated_vram_bytes: flag(args, "--dedicated-vram-bytes").and_then(|value| value.parse().ok()),
        cuda_occupancy: flag(args, "--cuda-occupancy").and_then(|value| value.parse().ok()),
    }
}
