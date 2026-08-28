//! Executable Prototype B/C population benchmarks.
//!
//! Both prototypes share one measurement protocol: the canonical oracle runs
//! first, outside timing; the engine then uploads once and every repetition is
//! timed end to end. Dispatch never falls back — a missing backend is a typed
//! unsupported status and a non-zero CLI exit, because a paid measurement that
//! silently ran somewhere else is worse than no measurement.

use anyhow::{Context, Result};
#[cfg(any(
    feature = "gpu-nvidia",
    feature = "gpu-b-native",
    feature = "gpu-bench-cuda",
    feature = "gpu-vulkan"
))]
use neoethos_search::gpu_native::benchmark::execute_population_benchmark;
use neoethos_search::gpu_native::benchmark::{
    BenchmarkIdentity, BenchmarkReport, DistributionSummary, ParityStatus,
    PopulationBenchmarkOptions, PopulationBenchmarkOutcome, PrototypeId, SweepPoint,
    ThroughputMetrics,
};
use neoethos_search::gpu_native::engine::EngineStatus;
use neoethos_search::gpu_native::population_fixture::TinyPopulationFixture;
use neoethos_search::gpu_native::prototype_bc::PrototypeKind;
use neoethos_search::gpu_native::prototype_population::{
    PropFirmRequirement, PrototypeBcRequirements, PrototypePopulationWorkload,
};
use neoethos_search::gpu_native::prototype_population_oracle::evaluate_population_oracle;
use neoethos_search::gpu_native::snapshot_fixture::SnapshotPopulationFixture;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The two prices the trade-invariant net needs to turn a price distance into
/// the same money the kernel reported, carried as one value so a call site
/// cannot supply one and forget the other.
/// Read only by the Prototype B arm, which is feature-gated. The value is still
/// BUILT unconditionally so the two arms take the same argument list and cannot
/// drift apart.
#[cfg_attr(
    not(any(feature = "gpu-nvidia", feature = "gpu-b-native")),
    allow(dead_code)
)]
#[derive(Debug, Clone, Copy)]
struct DevicePricing {
    pip_value: f64,
    pip_value_per_lot: f64,
}

/// Tiny deterministic fixture path for `--execute-tiny --prototype b|c`.
pub fn run_tiny(
    args: &[String],
    output: PathBuf,
    identity: BenchmarkIdentity,
    requested_sweep: SweepPoint,
) -> Result<()> {
    let fixture = TinyPopulationFixture::new(
        requested_sweep.population,
        requested_sweep.bar_count,
        requested_sweep.feature_count,
    );
    let workload = fixture
        .population_workload(PrototypeBcRequirements {
            prop_firm_state: PropFirmRequirement::NotRequested,
        })
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let sweep = SweepPoint {
        population: fixture.population(),
        batch_size: requested_sweep.batch_size,
        bar_count: fixture.bars(),
        feature_count: fixture.features(),
        scenario_count: 1,
        calendar_days: requested_sweep.calendar_days,
    };
    execute(args, output, identity, sweep, workload, None)
}

/// Real-data snapshot path for `--execute-snapshot --prototype b|c`.
pub fn run_snapshot(
    args: &[String],
    output: PathBuf,
    identity: BenchmarkIdentity,
    fixture: &SnapshotPopulationFixture,
) -> Result<()> {
    let workload = fixture
        .population_workload(PrototypeBcRequirements {
            prop_firm_state: PropFirmRequirement::NotRequested,
        })
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let sweep = SweepPoint {
        population: fixture.population(),
        batch_size: parse_usize(args, "--batch-size", fixture.population())?,
        bar_count: fixture.bars(),
        feature_count: fixture.features(),
        scenario_count: 1,
        calendar_days: None,
    };
    execute(
        args,
        output,
        identity,
        sweep,
        workload,
        Some(fixture.source_description().to_string()),
    )
}

fn execute(
    args: &[String],
    output: PathBuf,
    identity: BenchmarkIdentity,
    sweep: SweepPoint,
    workload: PrototypePopulationWorkload,
    source_description: Option<String>,
) -> Result<()> {
    let kind = match identity.prototype {
        PrototypeId::B => PrototypeKind::BWarpCooperative,
        PrototypeId::C => PrototypeKind::CSparseFirstHit,
        other => anyhow::bail!("{other:?} is not a population prototype"),
    };
    let eligibility = workload.common_bc_intersection(kind);
    let coverage = eligibility.coverage();
    if coverage.supported_candidates == 0 {
        anyhow::bail!(
            "no candidate is inside the common B/C intersection for {:?}; the workload is a \
             coverage gap, not a measurement",
            identity.prototype
        );
    }
    // Only the supported subset is measured. The unsupported remainder is
    // reported as a coverage gap and is never evaluated by another engine.
    let measured = workload
        .supported_partition(&eligibility)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let measured_eligibility = measured.common_bc_intersection(kind);

    // Reference work happens before any timed region.
    let reference = evaluate_population_oracle(&measured)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let reference_metrics = reference
        .metrics
        .iter()
        .map(|row| row.values)
        .collect::<Vec<_>>();

    let options = PopulationBenchmarkOptions {
        warmups: parse_usize(args, "--warmups", 2)?,
        repetitions: parse_usize(args, "--repetitions", 5)?.max(1),
    };
    let device = parse_optional_usize(args, "--device")?;
    let max_events = parse_usize(
        args,
        "--max-events",
        measured
            .genes
            .population()
            .saturating_mul(measured.dataset.bars()),
    )?;
    let session_id = parse_u64(args, "--seed", 0)?.wrapping_add(0x4243_0000_0000_0000);

    let outcome = match identity.prototype {
        PrototypeId::B => run_prototype_b(
            &measured,
            &measured_eligibility,
            &options,
            device,
            session_id,
            max_events,
            DevicePricing {
                pip_value: reference.settings.pip_value,
                pip_value_per_lot: reference.settings.pip_value_per_lot,
            },
        )?,
        PrototypeId::C => run_prototype_c(
            &measured,
            &measured_eligibility,
            &options,
            device,
            session_id,
            max_events,
        )?,
        other => anyhow::bail!("{other:?} is not a population prototype"),
    };

    write_report(
        args,
        output,
        identity,
        sweep,
        &options,
        &outcome,
        &reference_metrics,
        &workload,
        source_description,
    )
}

#[cfg(any(feature = "gpu-nvidia", feature = "gpu-b-native"))]
fn run_prototype_b(
    workload: &PrototypePopulationWorkload,
    eligibility: &neoethos_search::gpu_native::prototype_population::CommonBcEligibility,
    options: &PopulationBenchmarkOptions,
    device: Option<usize>,
    session_id: u64,
    max_events: usize,
    pricing: DevicePricing,
) -> Result<PopulationBenchmarkOutcome> {
    use neoethos_search::gpu_native::prototype_b_engine::create_prototype_b_engine;
    use neoethos_search::gpu_native::trade_invariants::{PriceSeries, audit_device_outcomes};

    let mut engine = create_prototype_b_engine(device, session_id, max_events)
        .map_err(|error| anyhow::anyhow!("Prototype B is unavailable: {error}"))?;
    let outcome = execute_population_benchmark(&mut engine, workload, eligibility, options)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    // The correctness net over the trades the card actually settled. Parity
    // against the oracle only proves the two agree; these properties hold by
    // definition, so they catch the case parity is blind to — both wrong.
    //
    // Outside every timed repetition: `read_diagnostics` is a device-to-host
    // copy, and it refuses itself above a 1 GB host budget rather than taking
    // the machine down. A refusal is reported, not swallowed: a benchmark that
    // could not check its own trades must say so.
    match engine.read_diagnostics() {
        Ok(diagnostics) => {
            let aggregates = outcome
                .candidate_ids
                .iter()
                .copied()
                .zip(outcome.metrics.iter().map(|values| values[0]))
                .collect::<Vec<_>>();
            let complaints = audit_device_outcomes(
                &diagnostics.events,
                &diagnostics.outcomes,
                &aggregates,
                PriceSeries {
                    high: &workload.dataset.high,
                    low: &workload.dataset.low,
                },
                pricing.pip_value,
                pricing.pip_value_per_lot,
            );
            if !complaints.is_empty() {
                let shown = complaints
                    .iter()
                    .take(20)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n  ");
                anyhow::bail!(
                    "Prototype B settled {} trade(s) that violate properties holding by \
                     definition — the measurement is void, not slow. First {} of {}:\n  {shown}",
                    diagnostics.outcomes.len(),
                    complaints.len().min(20),
                    complaints.len()
                );
            }
        }
        Err(error) => {
            eprintln!(
                "warning: the trade-invariant net did not run — the diagnostic readback was \
                 refused: {error}"
            );
        }
    }

    Ok(outcome)
}

#[cfg(not(any(feature = "gpu-nvidia", feature = "gpu-b-native")))]
fn run_prototype_b(
    _workload: &PrototypePopulationWorkload,
    _eligibility: &neoethos_search::gpu_native::prototype_population::CommonBcEligibility,
    _options: &PopulationBenchmarkOptions,
    _device: Option<usize>,
    _session_id: u64,
    _max_events: usize,
    _pricing: DevicePricing,
) -> Result<PopulationBenchmarkOutcome> {
    anyhow::bail!(
        "Prototype B is a native-CUDA engine: rebuild the CLI with --features gpu-b-native \
         (rustc + nvcc only) or --features gpu-nvidia (full stack) on a CUDA device. No other \
         engine may stand in for it."
    )
}

#[cfg(any(
    feature = "gpu-nvidia",
    feature = "gpu-bench-cuda",
    feature = "gpu-vulkan"
))]
fn run_prototype_c(
    workload: &PrototypePopulationWorkload,
    eligibility: &neoethos_search::gpu_native::prototype_population::CommonBcEligibility,
    options: &PopulationBenchmarkOptions,
    device: Option<usize>,
    session_id: u64,
    max_events: usize,
) -> Result<PopulationBenchmarkOutcome> {
    use neoethos_search::gpu_native::prototype_c_engine::create_prototype_c_engine;

    let mut engine = create_prototype_c_engine(device, session_id, max_events)
        .map_err(|error| anyhow::anyhow!("Prototype C is unavailable: {error}"))?;
    execute_population_benchmark(&mut engine, workload, eligibility, options)
        .map_err(|error| anyhow::anyhow!(error.to_string()))
}

#[cfg(not(any(
    feature = "gpu-nvidia",
    feature = "gpu-bench-cuda",
    feature = "gpu-vulkan"
)))]
fn run_prototype_c(
    _workload: &PrototypePopulationWorkload,
    _eligibility: &neoethos_search::gpu_native::prototype_population::CommonBcEligibility,
    _options: &PopulationBenchmarkOptions,
    _device: Option<usize>,
    _session_id: u64,
    _max_events: usize,
) -> Result<PopulationBenchmarkOutcome> {
    anyhow::bail!(
        "Prototype C needs a CubeCL runtime: rebuild the CLI with --features gpu-nvidia or \
         --features gpu-vulkan. No CPU path may stand in for it."
    )
}

#[allow(clippy::too_many_arguments)]
fn write_report(
    args: &[String],
    output: PathBuf,
    identity: BenchmarkIdentity,
    sweep: SweepPoint,
    options: &PopulationBenchmarkOptions,
    outcome: &PopulationBenchmarkOutcome,
    reference_metrics: &[[f64; 11]],
    full_workload: &PrototypePopulationWorkload,
    source_description: Option<String>,
) -> Result<()> {
    let parity = parity_status(reference_metrics, &outcome.metrics);
    let distribution = DistributionSummary::from_samples(&outcome.wall_samples)
        .context("the population benchmark produced no finite wall-time samples")?;
    let elapsed = distribution.median.max(f64::EPSILON);
    let measured_candidates = outcome.candidate_ids.len();
    let candidate_bars = (measured_candidates as u64).saturating_mul(sweep.bar_count as u64);

    let mut notes = vec![
        "Workload execution occurred inside neoethos-cli bench; the canonical oracle ran before \
         timing."
            .to_string(),
        format!(
            "Measured {measured_candidates} of {} candidates; the remainder is a declared \
             capability gap and was not evaluated by any other engine.",
            outcome.coverage.total_candidates
        ),
        "Engine status remains not_benchmarked until the attributed discrete-NVIDIA matrix is \
         complete."
            .to_string(),
    ];
    if let Some(description) = source_description {
        notes.push(format!("Versioned real-data snapshot: {description}"));
    }
    if !parity.matched {
        notes.push(
            "Parity failed: this run is a correctness failure, not a performance result."
                .to_string(),
        );
    }
    if outcome.accepted_trades <= 0.0 {
        notes.push(
            "No trade was accepted in the measured population; treat the throughput figures as \
             a lower bound on an empty workload."
                .to_string(),
        );
    }
    let _ = full_workload;

    let report = BenchmarkReport::new(
        identity,
        crate::gpu_bench::hardware(args),
        EngineStatus::NotBenchmarked,
        options.warmups,
        outcome.wall_samples.len(),
        sweep,
        distribution,
        BTreeMap::new(),
        ThroughputMetrics {
            candidates_per_second: Some(measured_candidates as f64 / elapsed),
            candidate_bars_per_second: Some(candidate_bars as f64 / elapsed),
            trades_per_second: (outcome.accepted_trades > 0.0)
                .then_some(outcome.accepted_trades / elapsed),
            peak_vram_bytes: parse_optional_u64(args, "--peak-vram-bytes")?,
            event_density: (candidate_bars > 0)
                .then_some(outcome.accepted_trades / candidate_bars as f64),
            hold_bars: None,
        },
        outcome.transfers,
        outcome.coverage.clone(),
        parity,
        notes,
    );
    report.write_json(&output)?;
    println!(
        "Executable {} population benchmark written to {}",
        match report.identity.prototype {
            PrototypeId::B => "Prototype B",
            PrototypeId::C => "Prototype C",
            _ => "population",
        },
        output.display()
    );
    Ok(())
}

/// A report may only claim a match when a candidate actually executed and every
/// row agrees with the canonical reference.
fn parity_status(reference: &[[f64; 11]], candidate: &[[f64; 11]]) -> ParityStatus {
    if candidate.is_empty() {
        return ParityStatus {
            matched: false,
            first_divergent_level: Some(10),
            detail: Some("no candidate executed; a match cannot be claimed".to_string()),
        };
    }
    let comparison = TinyPopulationFixture::compare_final_metrics(reference, candidate);
    match comparison.first_divergence {
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
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].clone())
}

fn parse_usize(args: &[String], name: &str, default: usize) -> Result<usize> {
    flag(args, name)
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("invalid {name} `{value}`"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_optional_usize(args: &[String], name: &str) -> Result<Option<usize>> {
    flag(args, name)
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("invalid {name} `{value}`"))
        })
        .transpose()
}

fn parse_u64(args: &[String], name: &str, default: u64) -> Result<u64> {
    flag(args, name)
        .map(|value| {
            value
                .parse::<u64>()
                .with_context(|| format!("invalid {name} `{value}`"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn parse_optional_u64(args: &[String], name: &str) -> Result<Option<u64>> {
    flag(args, name)
        .map(|value| {
            value
                .parse::<u64>()
                .with_context(|| format!("invalid {name} `{value}`"))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use neoethos_search::gpu_native::benchmark::{BenchmarkPass as Pass, FixtureMode};

    fn identity(prototype: PrototypeId) -> BenchmarkIdentity {
        BenchmarkIdentity {
            git_sha: "test".into(),
            baseline_sha: "test".into(),
            dataset_hash: "test".into(),
            config_hash: "test".into(),
            seed: 7,
            timeframe: "M1".into(),
            backend: "gpu_required".into(),
            prototype,
            fixture: FixtureMode::Tiny,
            pass: Pass::CleanTiming,
        }
    }

    #[test]
    fn a_report_cannot_claim_a_match_without_an_executed_candidate() {
        let reference = vec![[1.0_f64; 11]];
        let status = parity_status(&reference, &[]);
        assert!(!status.matched);
        assert!(status.detail.unwrap().contains("no candidate executed"));
    }

    #[test]
    fn identical_metrics_are_reported_as_a_match() {
        let reference = vec![[1.0_f64; 11], [2.0; 11]];
        let status = parity_status(&reference, &reference);
        assert!(status.matched, "identical rows must match");
    }

    #[test]
    fn divergent_metrics_are_reported_with_their_level() {
        let reference = vec![[1.0_f64; 11]];
        let mut candidate = reference.clone();
        candidate[0][0] = 500.0;
        let status = parity_status(&reference, &candidate);
        assert!(!status.matched);
        assert_eq!(status.first_divergent_level, Some(10));
    }

    #[test]
    fn a_fully_unsupported_population_is_refused_before_any_engine_is_created() {
        let fixture = TinyPopulationFixture::new(2, 128, 4);
        let (mut dataset, genes, scenarios) = fixture.prototype_a_uploads();
        dataset.settings.trailing_enabled = true;
        let workload = PrototypePopulationWorkload::from_uploads(
            dataset,
            genes,
            scenarios,
            PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::NotRequested,
            },
        )
        .unwrap();

        let error = execute(
            &[],
            PathBuf::from("cache/gpu-bench/unused.json"),
            identity(PrototypeId::C),
            SweepPoint {
                population: 2,
                batch_size: 2,
                bar_count: 128,
                feature_count: 4,
                scenario_count: 1,
                calendar_days: None,
            },
            workload,
            None,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("coverage gap"),
            "expected a coverage-gap refusal, got {error}"
        );
    }

    #[cfg(not(any(feature = "gpu-nvidia", feature = "gpu-b-native")))]
    #[test]
    fn prototype_b_is_refused_without_a_cuda_build_rather_than_substituted() {
        let workload = TinyPopulationFixture::new(2, 128, 4)
            .population_workload(PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::NotRequested,
            })
            .unwrap();
        let eligibility = workload.common_bc_intersection(PrototypeKind::BWarpCooperative);
        // This test exercises only the typed missing-CUDA refusal. Reading the
        // workload's own fixed-width settings avoids running the financial
        // oracle, which is production-gated until exact broker replay exists.
        let reference_settings = &workload.dataset.settings;
        let error = run_prototype_b(
            &workload,
            &eligibility,
            &PopulationBenchmarkOptions::default(),
            None,
            1,
            4096,
            DevicePricing {
                pip_value: reference_settings.pip_value,
                pip_value_per_lot: reference_settings.pip_value_per_lot,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("gpu-b-native"), "{error}");
    }

    #[cfg(not(any(feature = "gpu-nvidia", feature = "gpu-vulkan")))]
    #[test]
    fn prototype_c_is_refused_without_a_cubecl_runtime_rather_than_substituted() {
        let workload = TinyPopulationFixture::new(2, 128, 4)
            .population_workload(PrototypeBcRequirements {
                prop_firm_state: PropFirmRequirement::NotRequested,
            })
            .unwrap();
        let eligibility = workload.common_bc_intersection(PrototypeKind::CSparseFirstHit);
        let error = run_prototype_c(
            &workload,
            &eligibility,
            &PopulationBenchmarkOptions::default(),
            None,
            1,
            4096,
        )
        .unwrap_err();
        assert!(error.to_string().contains("CubeCL runtime"), "{error}");
    }
}
