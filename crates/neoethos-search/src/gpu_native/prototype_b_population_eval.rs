//! Prototype B as the discovery pipeline's population evaluator.
//!
//! Until now B existed only inside the benchmark harness, while the shipped
//! discovery lane ran Prototype A in f32. A 2026-07-28 measurement on one card
//! settled which of those belongs in production:
//!
//! | bars | A-f32 | A-f64 | B |
//! |---|---|---|---|
//! | 4 096 | wrong, 95.0 M cand-bars/s | wrong, 13.5 M | exact, 49.7 M |
//! | 20 000 | wrong, 136.4 M | wrong, 15.4 M | exact, 47.8 M |
//! | 200 000 | wrong by 54 %, 138.9 M | wrong by 0.19 %, 11.5 M | exact, 47.4 M |
//!
//! B is the only lane that reproduces the canonical CPU engine, and it is also
//! 3-4x faster than A once A is given the double precision it needs to be even
//! approximately right. This module is the adapter that lets the discovery
//! pipeline call it, presenting exactly the signature the existing CubeCL entry
//! point uses so the call sites do not have to care which engine runs.
//!
//! Nothing here changes what is computed — B was proven bit-exact against the
//! CPU at 4 096, 20 000 and 200 000 bars on real EURUSD data. It changes which
//! engine computes it.

use anyhow::{Context, Result, bail};

// `NeoPopulationEvent` is deliberately not imported. It was here only so the
// deleted event budget could take its size, and the population lane touches no
// event record anywhere else — the type stays alive for the diagnostic readback
// contract, which is a different lane.
use neoethos_gpu_contracts::device::{GeneDescriptor, ScenarioDescriptor};

use crate::eval::SmcRow;
use crate::gpu_native::prototype_a::PrototypeAGeneUpload;
use crate::gpu_native::prototype_population_oracle::population_settings_for_settings;
use crate::gpu_native::scenario;
use crate::population_auto_sizing_receipt_v1::{
    NativePopulationAutoPlanFactsV1, POPULATION_AUTO_ALLOCATOR_RESERVE_BYTES_V1,
    POPULATION_AUTO_HARD_GROWTH_CAP_V1, PopulationAutoSizingErrorCodeV1,
    PopulationAutoSizingErrorV1, sizing_error_v1,
};
use crate::population_execution_evidence_v1::UnsplittablePopulationAllocationV1;

use neoethos_gpu_cuda::{
    CudaPopulationError, PopulationGeneStorePlanV1, PopulationGeneView,
    PopulationMetricsOnlyPlanV1, PopulationParentDevicePlanV1, PopulationResidencyCountersV1,
};

/// Metric row shape shared with the CPU and CubeCL lanes.
const ZERO_METRICS: [f64; 11] = [0.0; 11];

struct NativePopulationBatchV1 {
    rows: Vec<[f64; 11]>,
    counters: PopulationResidencyCountersV1,
}

/// Is a CUDA device present and the native population engine usable?
pub(crate) fn prototype_b_available() -> bool {
    neoethos_gpu_cuda::runtime_available() && neoethos_gpu_cuda::device_count() > 0
}

/// Sustained device throughput, in scenario-bars per second.
///
/// Measured 2026-08 on an RTX 3090 at populations 16 384 and 131 072: 843-966 M
/// candidate-bars/s. The lower end is used, so the time estimate errs toward
/// SMALLER launches — an underestimate costs one extra launch, an overestimate
/// costs an unobservable hour.
///
/// PRE-FUSION MEASUREMENT — RE-MEASURE ON THE CARD. That number was taken while
/// the walk READ a precomputed `signal_values[bar]`. The fused walk now runs the
/// CSR accumulation, both thresholds and the SMC gate inside the same thread, so
/// its throughput is unmeasured and every launch-length estimate below is
/// arithmetic against a stale constant. It is only ever used to LOWER the
/// approved count, so a wrong value cannot cause an out-of-memory — it can only
/// make a launch longer or shorter than the target claims.
const SCENARIO_BARS_PER_SECOND: u64 = 843_000_000;

/// How long one submission should take.
///
/// The active strict path is metrics-only: at the default 240-month capacity it
/// allocates exactly 4 000 bytes per scenario and zero outcome records. That
/// makes time, rather than scenario VRAM, the normal launch bound. At about
/// 1 000 scenarios/second a multi-million-scenario launch can run for more than
/// an hour with no host observation point: no progress, no telemetry line, no
/// chance to cancel, and a failure anywhere in it discards all of it.
///
/// So sizing gained a second term. Memory says what the card can HOLD; this
/// says what the operator can WATCH. The launch takes the smaller.
const TARGET_LAUNCH_SECONDS: u64 = 20;

/// Below this a launch stops filling the card, so the time term must never push
/// under it — the occupancy knee measured on the 3090.
///
/// It cannot cause an out-of-memory: the time term is only ever combined with
/// the memory term by `min`, so the memory ceiling still wins outright.
const OCCUPANCY_KNEE: u64 = 16_384;

/// Scenarios that keep one launch inside [`TARGET_LAUNCH_SECONDS`].
///
/// Peak memory is untouched by this: it can only lower the count.
///
/// The occupancy floor WINS over the target when the series is long enough, and
/// it says so rather than leaving the operator to infer it: at 5 270 000 bars
/// the honest term is ~3 200 scenarios and the floor lifts it to 16 384, an
/// estimated 102 s launch against a 20 s target. That is the right trade — a
/// launch that does not fill the card wastes more than it saves — but an
/// unobservable 102 s is exactly what this term exists to stop being a surprise.
pub(crate) fn checked_candidates_for_target_launch_v1(
    bars: usize,
) -> std::result::Result<(u64, u64, bool), CudaPopulationError> {
    let bars = u64::try_from(bars)
        .map_err(|_| {
            CudaPopulationError::InvalidInput(
                "evaluation rows do not fit the strict u64 time plan".to_owned(),
            )
        })?
        .max(1);
    let raw = SCENARIO_BARS_PER_SECOND
        .checked_mul(TARGET_LAUNCH_SECONDS)
        .ok_or_else(|| {
            CudaPopulationError::InvalidInput(
                "strict target-launch numerator overflows u64".to_owned(),
            )
        })?
        / bars;
    let floor_overrode = raw < OCCUPANCY_KNEE;
    if raw < OCCUPANCY_KNEE {
        tracing::warn!(
            target: "neoethos_search::eval",
            bars,
            target_seconds = TARGET_LAUNCH_SECONDS,
            time_term = raw,
            floor = OCCUPANCY_KNEE,
            estimated_seconds = OCCUPANCY_KNEE.saturating_mul(bars) / SCENARIO_BARS_PER_SECOND,
            "the occupancy floor overrides the launch-length target — this launch will run \
             longer than the target with no host observation point"
        );
    }
    Ok((raw, raw.max(OCCUPANCY_KNEE), floor_overrode))
}

fn max_effective_scenario_window_rows_v1(
    scenarios: &[ScenarioDescriptor],
    evaluation_rows: usize,
) -> Result<usize> {
    let mut maximum = 0usize;
    for (index, scenario) in scenarios.iter().enumerate() {
        let offset = usize::try_from(scenario.window_offset)
            .with_context(|| format!("scenario {index} window offset does not fit this process"))?;
        let length = if scenario.window_len == 0 {
            evaluation_rows.checked_sub(offset).ok_or_else(|| {
                anyhow::anyhow!(
                    "scenario {index} starts at {offset}, past the {evaluation_rows}-row evaluation view"
                )
            })?
        } else {
            usize::try_from(scenario.window_len).with_context(|| {
                format!("scenario {index} window length does not fit this process")
            })?
        };
        let end = offset.checked_add(length).ok_or_else(|| {
            anyhow::anyhow!("scenario {index} effective window extent overflows usize")
        })?;
        if length == 0 || end > evaluation_rows {
            bail!(
                "scenario {index} effective window {offset}..{end} is outside the {evaluation_rows}-row evaluation view"
            );
        }
        maximum = maximum.max(length);
    }
    if maximum == 0 {
        bail!("strict time sizing requires at least one non-empty scenario window");
    }
    Ok(maximum)
}

/// What the sizing arithmetic concluded.
///
/// THE TWO "NO ANSWER" CASES ARE NOT THE SAME ANSWER. This was an `Option`, and
/// `None` meant both "free VRAM is unreadable" and "this card has no room for
/// the dataset at all" — so the branch taken when the card is provably too small
/// was the branch that re-used the LAST SUCCESSFUL LARGE BATCH. That is the
/// `unwrap_or(usize::MAX)` incident rebuilt with more steps: a 12 GB card asked
/// for an M1 dataset it cannot hold would have launched whatever an earlier M5
/// run had fitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sizing {
    /// The card can host this many scenarios in one launch.
    Fits {
        memory_ceiling: usize,
        time_ceiling: usize,
        chosen: usize,
    },
    /// The admitted card has no usable room. Scenario splitting cannot repair
    /// an immutable parent or gene-store allocation that does not fit.
    NoRoom {
        fixed_device_bytes: u64,
        budget_bytes: u64,
    },
    /// The immutable parent and exact gene store fit, but even one strict
    /// metrics-only scenario does not fit in the remaining admitted budget.
    NoScenarioRoom {
        available_device_bytes: u64,
        one_scenario_device_bytes: u64,
    },
}

const ALLOCATOR_RESERVE_BYTES_V1: u64 = POPULATION_AUTO_ALLOCATOR_RESERVE_BYTES_V1;

fn admitted_budget_bytes_v1(
    pre_parent_free_memory_bytes: u64,
) -> std::result::Result<u64, CudaPopulationError> {
    (pre_parent_free_memory_bytes / 10)
        .checked_mul(7)
        .ok_or_else(|| {
            CudaPopulationError::InvalidInput(
                "strict admitted memory budget overflows u64".to_owned(),
            )
        })
}

/// Exact strict runtime plan against the free-memory snapshot captured on the
/// selected ordinal before this run allocated its resident parent.
///
/// `resident_parent_rows` and `evaluation_rows` are deliberately different:
/// the former sizes immutable VRAM, while the latter sizes the observable
/// launch-duration ceiling. Stage 1 may evaluate 25% of a parent that must
/// remain resident in full.
fn candidates_for_pre_parent_free_memory(
    pre_parent_free_memory_bytes: u64,
    resident_parent_rows: usize,
    evaluation_rows: usize,
    feature_count: usize,
    month_capacity: usize,
    gene_count: usize,
    gene_term_count: usize,
) -> std::result::Result<Sizing, CudaPopulationError> {
    let month_capacity = u32::try_from(month_capacity).map_err(|_| {
        CudaPopulationError::InvalidInput(
            "month capacity does not fit the strict native u32 plan".to_owned(),
        )
    })?;
    let parent = PopulationParentDevicePlanV1::checked_from_parent_extents_v1(
        resident_parent_rows,
        feature_count,
    )?;
    let genes =
        PopulationGeneStorePlanV1::checked_from_gene_extents_v1(gene_count, gene_term_count)?;
    let one_scenario =
        PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(1, month_capacity)?;
    let budget_bytes = admitted_budget_bytes_v1(pre_parent_free_memory_bytes)?;
    let fixed_device_bytes = parent
        .total_device_bytes()
        .checked_add(genes.total_device_bytes())
        .and_then(|bytes| bytes.checked_add(ALLOCATOR_RESERVE_BYTES_V1))
        .ok_or_else(|| {
            CudaPopulationError::InvalidInput(
                "strict fixed device allocation plan overflows u64".to_owned(),
            )
        })?;
    if fixed_device_bytes > budget_bytes {
        return Ok(Sizing::NoRoom {
            fixed_device_bytes,
            budget_bytes,
        });
    }
    let available_device_bytes = budget_bytes
        .checked_sub(fixed_device_bytes)
        .expect("fixed bytes were checked within the admitted budget");
    let one_scenario_device_bytes = one_scenario.total_device_bytes();
    let memory_ceiling = available_device_bytes / one_scenario_device_bytes;
    if memory_ceiling == 0 {
        return Ok(Sizing::NoScenarioRoom {
            available_device_bytes,
            one_scenario_device_bytes,
        });
    }
    let (_, time_ceiling, _) = checked_candidates_for_target_launch_v1(evaluation_rows)?;
    let chosen = memory_ceiling.min(time_ceiling);
    let memory_ceiling = usize::try_from(memory_ceiling).map_err(|_| {
        CudaPopulationError::InvalidInput(
            "strict memory ceiling does not fit this process".to_owned(),
        )
    })?;
    let time_ceiling = usize::try_from(time_ceiling).map_err(|_| {
        CudaPopulationError::InvalidInput(
            "strict time ceiling does not fit this process".to_owned(),
        )
    })?;
    let chosen = usize::try_from(chosen).map_err(|_| {
        CudaPopulationError::InvalidInput(
            "strict submission ceiling does not fit this process".to_owned(),
        )
    })?;
    Ok(Sizing::Fits {
        memory_ceiling,
        time_ceiling,
        chosen,
    })
}

fn auto_sizing_error_from_cuda_v1(
    context: &'static str,
    source: CudaPopulationError,
) -> PopulationAutoSizingErrorV1 {
    sizing_error_v1(
        PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
        format!("{context}: {source}"),
    )
}

/// One admission-bound auto plan. Every byte comes from the same checked CUDA
/// allocation plans as runtime submission sizing; this function never probes a
/// device or reads live free memory.
#[allow(clippy::too_many_arguments)]
pub(crate) fn population_auto_plan_for_pre_parent_free_memory_v1(
    pre_parent_free_memory_bytes: u64,
    resident_parent_rows: usize,
    evaluation_rows: usize,
    feature_count: usize,
    month_capacity: usize,
    configured_population: usize,
    term_cap: usize,
) -> std::result::Result<NativePopulationAutoPlanFactsV1, PopulationAutoSizingErrorV1> {
    let month_capacity_u32 = u32::try_from(month_capacity).map_err(|_| {
        sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
            "month capacity does not fit the authoritative native u32 plan",
        )
    })?;
    let configured_terms = configured_population.checked_mul(term_cap).ok_or_else(|| {
        sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
            "configured population × sealed term cap overflows usize",
        )
    })?;
    let parent = PopulationParentDevicePlanV1::checked_from_parent_extents_v1(
        resident_parent_rows,
        feature_count,
    )
    .map_err(|source| auto_sizing_error_from_cuda_v1("strict parent plan", source))?;
    let one_gene = PopulationGeneStorePlanV1::checked_from_gene_extents_v1(1, term_cap)
        .map_err(|source| auto_sizing_error_from_cuda_v1("one-gene plan", source))?;
    let two_terms = term_cap.checked_mul(2).ok_or_else(|| {
        sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
            "two sealed-cap genes overflow the term extent",
        )
    })?;
    let two_genes = PopulationGeneStorePlanV1::checked_from_gene_extents_v1(2, two_terms)
        .map_err(|source| auto_sizing_error_from_cuda_v1("two-gene plan", source))?;
    let configured_genes = PopulationGeneStorePlanV1::checked_from_gene_extents_v1(
        configured_population,
        configured_terms,
    )
    .map_err(|source| auto_sizing_error_from_cuda_v1("configured gene plan", source))?;
    let one_scenario =
        PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(1, month_capacity_u32)
            .map_err(|source| auto_sizing_error_from_cuda_v1("one-scenario plan", source))?;
    let configured_scenarios = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(
        configured_population,
        month_capacity_u32,
    )
    .map_err(|source| auto_sizing_error_from_cuda_v1("configured scenario plan", source))?;
    let budget = admitted_budget_bytes_v1(pre_parent_free_memory_bytes)
        .map_err(|source| auto_sizing_error_from_cuda_v1("admitted budget", source))?;
    let parent_and_reserve = parent
        .total_device_bytes()
        .checked_add(ALLOCATOR_RESERVE_BYTES_V1)
        .ok_or_else(|| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
                "parent plus allocator reserve overflows u64",
            )
        })?;
    if parent_and_reserve > budget {
        return Err(sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ParentNoRoom,
            format!(
                "admitted budget {budget} B cannot host parent+reserve {} B",
                parent_and_reserve
            ),
        ));
    }
    let configured_fixed = parent_and_reserve
        .checked_add(configured_genes.total_device_bytes())
        .ok_or_else(|| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
                "parent plus configured gene store overflows u64",
            )
        })?;
    if configured_fixed > budget {
        return Err(sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::GeneNoRoom,
            format!(
                "configured unsplittable gene store needs {configured_fixed} B with parent/reserve against {budget} B"
            ),
        ));
    }
    let configured_one_launch = configured_fixed
        .checked_add(one_scenario.total_device_bytes())
        .ok_or_else(|| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
                "configured fixed bytes plus one scenario overflow u64",
            )
        })?;
    if configured_one_launch > budget {
        return Err(sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ScenarioNoRoom,
            format!(
                "configured parent/gene store fits but one strict scenario raises it to {configured_one_launch} B against {budget} B"
            ),
        ));
    }

    // Derive the affine gene slope and fixed overhead from two authoritative
    // plans instead of restating the native layout formula here.
    let gene_slope = two_genes
        .total_device_bytes()
        .checked_sub(one_gene.total_device_bytes())
        .ok_or_else(|| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
                "authoritative gene plan is not monotone",
            )
        })?;
    let gene_fixed_overhead = one_gene
        .total_device_bytes()
        .checked_sub(gene_slope)
        .ok_or_else(|| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
                "authoritative gene fixed overhead underflows",
            )
        })?;
    let available_after_parent_reserve_and_gene_overhead = budget
        .checked_sub(parent_and_reserve)
        .and_then(|bytes| bytes.checked_sub(gene_fixed_overhead))
        .ok_or_else(|| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::GeneNoRoom,
                "admitted budget leaves no room for one authoritative gene-store header",
            )
        })?;
    let fixed_gene_bytes = available_after_parent_reserve_and_gene_overhead
        .checked_sub(one_scenario.total_device_bytes())
        .ok_or_else(|| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::ScenarioNoRoom,
                "admitted budget leaves no room for a gene store plus one strict scenario",
            )
        })?;
    let fixed_gene_capacity_u64 = fixed_gene_bytes / gene_slope;
    let combined_slope = gene_slope
        .checked_add(one_scenario.total_device_bytes())
        .ok_or_else(|| {
            sizing_error_v1(
                PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
                "combined gene/scenario slope overflows u64",
            )
        })?;
    let memory_population_cap_u64 =
        available_after_parent_reserve_and_gene_overhead / combined_slope;
    let fixed_gene_capacity = usize::try_from(fixed_gene_capacity_u64).map_err(|_| {
        sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
            "fixed gene capacity does not fit this process",
        )
    })?;
    let memory_population_cap = usize::try_from(memory_population_cap_u64).map_err(|_| {
        sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
            "memory population cap does not fit this process",
        )
    })?;
    let (raw_time_cap, effective_time_cap, floor_overrode) =
        checked_candidates_for_target_launch_v1(evaluation_rows)
            .map_err(|source| auto_sizing_error_from_cuda_v1("strict time plan", source))?;
    let raw_time_cap = usize::try_from(raw_time_cap).map_err(|_| {
        sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
            "raw time cap does not fit this process",
        )
    })?;
    let effective_time_cap = usize::try_from(effective_time_cap).map_err(|_| {
        sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
            "effective time cap does not fit this process",
        )
    })?;
    let hard_growth_cap = POPULATION_AUTO_HARD_GROWTH_CAP_V1;
    let growth_cap = memory_population_cap
        .min(effective_time_cap)
        .min(hard_growth_cap);
    Ok(NativePopulationAutoPlanFactsV1 {
        admitted_budget_bytes: budget,
        parent_device_bytes: parent.total_device_bytes(),
        gene_bytes_per_candidate_at_term_cap: gene_slope,
        gene_fixed_overhead_bytes: gene_fixed_overhead,
        scenario_device_bytes_per_candidate: one_scenario.total_device_bytes(),
        configured_gene_device_bytes: configured_genes.total_device_bytes(),
        configured_scenario_device_bytes: configured_scenarios.total_device_bytes(),
        fixed_gene_capacity,
        memory_population_cap,
        raw_time_cap,
        effective_time_cap,
        occupancy_floor_overrode_time_target: floor_overrode,
        hard_growth_cap,
        growth_cap,
    })
}

pub(crate) fn population_auto_resolved_bytes_v1(
    population: usize,
    term_cap: usize,
    month_capacity: usize,
) -> std::result::Result<(u64, u64), PopulationAutoSizingErrorV1> {
    let terms = population.checked_mul(term_cap).ok_or_else(|| {
        sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
            "resolved population × term cap overflows usize",
        )
    })?;
    let month_capacity = u32::try_from(month_capacity).map_err(|_| {
        sizing_error_v1(
            PopulationAutoSizingErrorCodeV1::ArithmeticOverflow,
            "month capacity does not fit native u32 plan",
        )
    })?;
    let genes = PopulationGeneStorePlanV1::checked_from_gene_extents_v1(population, terms)
        .map_err(|source| auto_sizing_error_from_cuda_v1("resolved gene plan", source))?;
    let scenarios =
        PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(population, month_capacity)
            .map_err(|source| auto_sizing_error_from_cuda_v1("resolved scenario plan", source))?;
    Ok((genes.total_device_bytes(), scenarios.total_device_bytes()))
}

pub(crate) fn runtime_submission_ceiling_for_admitted_ordinal_v1(
    admitted: &crate::ExactCudaDeviceOrdinalV1,
    resident_parent_rows: usize,
    evaluation_rows: usize,
    feature_count: usize,
    month_capacity: usize,
    gene_count: usize,
    gene_term_count: usize,
) -> Result<usize> {
    match candidates_for_pre_parent_free_memory(
        admitted.pre_parent_free_memory_bytes(),
        resident_parent_rows,
        evaluation_rows,
        feature_count,
        month_capacity,
        gene_count,
        gene_term_count,
    )
    .map_err(anyhow::Error::new)?
    {
        Sizing::Fits { chosen, .. } => Ok(chosen),
        Sizing::NoRoom {
            fixed_device_bytes,
            budget_bytes,
        } => bail!(
            "prototype B: the admitted pre-parent snapshot cannot host the immutable parent and exact unsplittable gene store: fixed {fixed_device_bytes} B against {budget_bytes} B usable budget"
        ),
        Sizing::NoScenarioRoom {
            available_device_bytes,
            one_scenario_device_bytes,
        } => bail!(
            "prototype B: the immutable parent and exact unsplittable gene store fit, but only {available_device_bytes} B remains in the admitted budget and one strict metrics-only scenario needs {one_scenario_device_bytes} B"
        ),
    }
}

/// The `max_events` the native session is created with — a formality.
///
/// There WAS an `event_capacity` here. It read free VRAM, subtracted the
/// dataset and the per-candidate workspaces, divided the remainder by
/// `size_of::<NeoPopulationEvent>() + size_of::<NeoPopulationOutcome>()`, and
/// `bail!`ed the whole evaluation when the answer came out under 1 024.
///
/// Every one of those bytes was imaginary. `session->events` is declared in the
/// native session and freed in `release_workspace()`, and there is no
/// `device_alloc` for it anywhere in the allocation block — the only kernel
/// that ever filled it has been commented out since the reduce started opening
/// positions from the signal directly. The engine has not allocated one event
/// record in a long time.
///
/// So the budget guarded nothing and cost three real things: it could abort an
/// evaluation over a buffer that does not exist; it was half the session-reuse
/// predicate, so a session was rebuilt whenever the imaginary number moved; and
/// because it SUBTRACTS `population * bars * 5`, a larger population asked for
/// a smaller capacity — the arithmetic ran backwards with respect to the one
/// quantity that actually matters.
///
/// What now bounds device memory is the admitted pre-parent snapshot together
/// with exact parent, gene-store, and metrics-only plans. This constant just
/// satisfies the ABI's "must be non-zero" check; the device stores it nowhere
/// and reads it never.
/// What a learned launch size is a fact ABOUT.
///
/// A fit is measured in scenarios, and the room available for scenarios is
/// `budget - dataset(bars, features)` — which changes by an order of magnitude
/// between an M5 and an M1 dataset, and between a 12 GB and a 24 GB card. A
/// single process-wide `AtomicUsize` keyed to nothing therefore replayed one
/// symbol/timeframe's fit as another's launch size, and one dataset's FAILURE
/// permanently capped every other dataset's launches for the life of the
/// process.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct LimitKey {
    device: usize,
    cuda_device_identity_sha256: String,
    parent_dataset_identity_sha256: String,
    pre_parent_free_memory_bytes: u64,
    resident_parent_rows: usize,
    evaluation_rows: usize,
    feature_count: usize,
    month_capacity: usize,
    gene_count: usize,
    gene_term_count: usize,
}

/// Largest work list known to have fitted THIS (device, dataset shape).
///
/// Discovering a fragmentation/runtime-pressure limit by failing costs the
/// whole attempt: the allocation is discarded before the halves are retried.
/// Paying that once is unavoidable; paying it every generation is waste. A
/// 2026-07-29 M3 run spent 391 s on a single population evaluation that way,
/// against a benchmark rate that would place it near 4 s.
///
/// So the size that worked is remembered and used as the starting point — for
/// the shape it was learned on, and no other. An absent entry means "not yet
/// learned": the first call tries the whole work list, as it should.
fn learned_batch_limits() -> &'static std::sync::Mutex<std::collections::HashMap<LimitKey, usize>> {
    static LIMITS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<LimitKey, usize>>,
    > = std::sync::OnceLock::new();
    LIMITS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn learned_batch_limit(key: &LimitKey) -> usize {
    learned_batch_limits()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(key)
        .copied()
        .unwrap_or(usize::MAX)
}

/// Raise the learned ceiling: this size was accepted.
fn learn_batch_success(key: &LimitKey, scenarios: usize) {
    let mut limits = learned_batch_limits()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = limits.entry(key.clone()).or_insert(scenarios);
    if *entry != usize::MAX {
        *entry = (*entry).max(scenarios);
    }
}

/// Lower the learned ceiling: this size was refused for capacity.
fn learn_batch_failure(key: &LimitKey, ceiling: usize) {
    let mut limits = learned_batch_limits()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = limits.entry(key.clone()).or_insert(ceiling);
    *entry = (*entry).min(ceiling);
}

/// Evaluate one full-series scenario per gene — the identity work list.
///
/// This is what every caller outside the quality screen wants, and it is the
/// parity floor for the whole scenario change: a descriptor array of nothing but
/// `base_scenario` describes exactly the evaluation the engine performed before
/// scenarios existed. The zeroed fields the old code wrote by hand are now
/// written by [`scenario::base_scenario`], which matters for one of them —
/// `spread_ticks` and `commission_micros` are now `-1` ("no override") where the
/// literal was `0`, and `0` has since become a REAL override meaning "charge no
/// spread". Constructing descriptors anywhere but through the builders is how
/// that would silently become a free-trading backtest.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_evaluate_population_b(
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
) -> Result<Vec<[f64; 11]>> {
    evidence
        .validate_population_layout(evidence.row_count(), evidence.feature_count())
        .map_err(anyhow::Error::new)?;
    let expected = long_thr.len();
    let rows = require_exact_native_population_rows_v1(
        evaluate_population_b_raw_v1(
            evidence,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            sl_pips,
            tp_pips,
            stop_vol_mult,
            gene_smc_flags,
            gate_threshold,
            smc_weights,
        ),
        expected,
    )?;
    if expected > 0 {
        evidence.record_successful_native_population_v1(
            expected,
            rows.rows.len(),
            rows.counters,
        )?;
        evidence
            .record_successful_population(
                crate::engine_identity::PopulationEvalEngine::CudaNativeF64,
                expected,
                rows.rows.len(),
            )
            .map_err(anyhow::Error::new)?;
    }
    Ok(rows.rows)
}

fn require_exact_native_population_rows_v1(
    outcome: Result<NativePopulationBatchV1>,
    expected: usize,
) -> Result<NativePopulationBatchV1> {
    let rows = outcome?;
    if rows.rows.len() != expected {
        bail!(
            "prototype B returned {} metric rows; expected {expected}",
            rows.rows.len()
        );
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_population_b_raw_v1(
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
) -> Result<NativePopulationBatchV1> {
    let n_genes = long_thr.len();
    let bars = evidence.row_count();
    if n_genes == 0 || bars == 0 {
        return Ok(NativePopulationBatchV1 {
            rows: vec![ZERO_METRICS; n_genes],
            counters: PopulationResidencyCountersV1::default(),
        });
    }
    let scenarios: Vec<ScenarioDescriptor> = (0..n_genes as u64)
        .map(|candidate| scenario::base_scenario(candidate, candidate, bars))
        .collect();
    evaluate_scenarios_b_raw_v1(
        evidence,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
        &scenarios,
    )
}

/// Evaluate an arbitrary work list on Prototype B, splitting when a
/// scenario-sized allocation cannot fit despite the admitted exact plan.
///
/// THE SCENARIO IS THE UNIT OF WORK. The gene arrays stay whole and resident;
/// what is sized, split and submitted is the DESCRIPTOR ARRAY. That is the
/// difference that turns the quality screen's seven launches — six Monte-Carlo
/// chunks and one sensitivity pass — into one.
///
/// The device allocation is bounded by the run's admitted pre-parent snapshot
/// and exact allocation plans. Fragmentation can still exhaust a scenario-sized
/// allocation, so the engine reports that condition distinctly and the work
/// list is cut and each part evaluated in turn. Parent and gene allocation
/// failures carry an unsplittable marker and never enter this recursion.
///
/// Scenarios are independent — each is one thread reading shared, read-only gene
/// and dataset arrays — so a split result equals the whole one. This is the
/// never-OOM rule applied as intended: peak memory follows the hardware, and an
/// oversized workload gets slower instead of failing.
///
/// Splitting the SCENARIOS rather than the genes is also strictly simpler than
/// what it replaces. The old split sliced the CSR gene arrays and rebased the
/// tail's offsets — get that wrong and it evaluates the wrong terms silently
/// rather than erroring. Slicing a descriptor array cannot be got wrong that
/// way: every descriptor carries its own gene index, so a slice is still a
/// correct work list whatever it contains.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_evaluate_scenarios_b(
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
    scenarios: &[ScenarioDescriptor],
) -> Result<Vec<[f64; 11]>> {
    evidence
        .validate_population_layout(evidence.row_count(), evidence.feature_count())
        .map_err(anyhow::Error::new)?;
    let expected = scenarios.len();
    let rows = require_exact_native_population_rows_v1(
        evaluate_scenarios_b_raw_v1(
            evidence,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            sl_pips,
            tp_pips,
            stop_vol_mult,
            gene_smc_flags,
            gate_threshold,
            smc_weights,
            scenarios,
        ),
        expected,
    )?;
    if expected > 0 {
        evidence.record_successful_native_population_v1(
            expected,
            rows.rows.len(),
            rows.counters,
        )?;
        evidence
            .record_successful_population(
                crate::engine_identity::PopulationEvalEngine::CudaNativeF64,
                expected,
                rows.rows.len(),
            )
            .map_err(anyhow::Error::new)?;
    }
    Ok(rows.rows)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_scenarios_b_raw_v1(
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
    scenarios: &[ScenarioDescriptor],
) -> Result<NativePopulationBatchV1> {
    let n_genes = long_thr.len();
    // What the launch is sized by, from here down.
    let n_scenarios = scenarios.len();
    if n_scenarios == 0 {
        return Ok(NativePopulationBatchV1 {
            rows: Vec::new(),
            counters: PopulationResidencyCountersV1::default(),
        });
    }
    // Refused here, before anything is uploaded, because the device cannot
    // detect either fault: a gene index past the end of the population is an
    // out-of-bounds read of thresholds and CSR offsets that still produces a
    // metric row, and a window past the end of the series is an out-of-bounds
    // read of prices. The native side checks again — two independent guards,
    // like the workspace-population pair — but this one can name the index.
    if let Err(detail) = scenario::validate_scenarios(scenarios, n_genes, evidence.row_count()) {
        bail!("prototype B scenario list: {detail}");
    }
    let evaluation_rows = max_effective_scenario_window_rows_v1(scenarios, evidence.row_count())?;

    let admitted_ordinal = evidence.require_exact_cuda_device_ordinal_v1()?;
    let selected_ordinal = admitted_ordinal.selected_ordinal();
    let device = usize::try_from(selected_ordinal).map_err(|_| {
        anyhow::anyhow!("sealed CUDA ordinal {selected_ordinal} does not fit this process")
    })?;

    // Everything the launch size is a fact about, in one key.
    let limit_key = LimitKey {
        device,
        cuda_device_identity_sha256: admitted_ordinal.cuda_device_identity_sha256().to_owned(),
        parent_dataset_identity_sha256: evidence.parent_dataset_identity_sha256().to_owned(),
        pre_parent_free_memory_bytes: admitted_ordinal.pre_parent_free_memory_bytes(),
        resident_parent_rows: evidence.parent_row_count(),
        evaluation_rows,
        feature_count: evidence.feature_count(),
        month_capacity: crate::eval::current_backtest_runtime_overrides().month_capacity,
        gene_count: n_genes,
        gene_term_count: gene_indices.len(),
    };
    // Start at the size already known to fit THIS shape rather than
    // rediscovering the limit by throwing away a full evaluation every
    // generation.
    let learned = learned_batch_limit(&limit_key);
    // What the card can hold is knowable before asking it, so ask first. The
    // retry below still exists for fragmentation or external runtime pressure,
    // but it should never be reached for a size that was arithmetic all along.
    let fits = runtime_submission_ceiling_for_admitted_ordinal_v1(
        admitted_ordinal,
        limit_key.resident_parent_rows,
        limit_key.evaluation_rows,
        limit_key.feature_count,
        limit_key.month_capacity,
        limit_key.gene_count,
        limit_key.gene_term_count,
    )?;
    if fits < n_scenarios {
        tracing::info!(
            target: "neoethos_search::eval",
            n_genes,
            n_scenarios,
            fits,
            "work list exceeds what the card can host — splitting before asking"
        );
    }
    let learned = learned.min(fits);
    if n_scenarios > learned && n_scenarios > 1 {
        return split_and_evaluate(
            evidence,
            gene_offsets,
            gene_indices,
            gene_weights,
            long_thr,
            short_thr,
            sl_pips,
            tp_pips,
            stop_vol_mult,
            gene_smc_flags,
            gate_threshold,
            smc_weights,
            scenarios,
            learned,
        );
    }

    let attempt = evaluate_population_b_batch(
        evidence,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
        device,
        scenarios,
    );
    let Err(error) = attempt else {
        // This size fits; keep it as the starting point for the next call over
        // this same shape.
        learn_batch_success(&limit_key, n_scenarios);
        return attempt;
    };
    // Only a capacity exhaustion is worth retrying smaller. Anything else is a
    // fault, and halving the work would just hide it behind a slower failure.
    //
    // "Capacity exhaustion" now excludes immutable parent and gene uploads.
    // They report the same `STATUS_ALLOCATION_FAILED` as a scenario-sized
    // workspace exhaustion, so without the marker an unchanged multi-gigabyte
    // allocation was retried at every recursive leaf.
    if !is_capacity_exhaustion(&error) || n_scenarios < 2 {
        return Err(error);
    }
    // Remember the ceiling so the next generation does not pay this again.
    learn_batch_failure(&limit_key, n_scenarios / 2);
    tracing::info!(
        target: "neoethos_search::eval",
        n_genes,
        n_scenarios,
        learned = learned_batch_limit(&limit_key),
        "a scenario-sized allocation hit runtime capacity pressure — splitting, and \
         remembering the limit so later launches start there"
    );
    return split_and_evaluate(
        evidence,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
        scenarios,
        n_scenarios / 2,
    );
}

/// Whether the card ran out of room, so halving the work is worth a retry.
///
/// This compared the message text against `"exceeded the session capacity"`,
/// which is the event buffer's wording. Removing the event buffer moved the
/// first allocation to fail to the outcome array, which says `"device
/// allocation failed"` — so the retry stopped firing and every oversized
/// population went to the CPU. The optimisation disabled its own safety net,
/// quietly, because the two agreed on a string rather than a type.
///
/// Asking the error what it is removes that coupling.
///
/// There was a second arm here matching the string `"leaves no room for events
/// on this device"` — the host-side event budget's own `bail!`. That budget was
/// sizing a buffer the kernel never allocated, so it is gone and so is the
/// phrase; a substring match against a message no code can produce is dead
/// weight that reads like a live path.
///
/// This only works because the call sites attach context rather than formatting
/// the error into a new one — `anyhow!("...: {error}")` renders the source and
/// throws the value away, which left this check unable to find anything and
/// made it dead code the moment it was written.
fn is_capacity_exhaustion(error: &anyhow::Error) -> bool {
    // An immutable parent or gene upload that ran out of memory is never a
    // work-list size problem. A split halves only the descriptor/workspace
    // extent, so
    // retrying a failed immutable parent or gene upload smaller re-attempts the
    // identical `cudaMalloc` at every leaf of the recursion. See
    // [`UnsplittablePopulationAllocationV1`].
    //
    // `anyhow::Error::downcast_ref` is what searches the context chain.
    // `error.chain()` does NOT work here: it yields anyhow's internal
    // `ContextError` wrapper as `&dyn Error`, and downcasting THAT to the
    // marker fails — the check compiles, always answers "no", and the marker
    // silently does nothing. `an_unsplittable_allocation_failure_is_never_split`
    // caught exactly that.
    if error
        .downcast_ref::<UnsplittablePopulationAllocationV1>()
        .is_some()
    {
        return false;
    }
    error
        .downcast_ref::<CudaPopulationError>()
        .is_some_and(CudaPopulationError::is_capacity_exhausted)
}

#[cfg(test)]
mod capacity_detection_tests {
    use super::*;

    /// Every sizing test drives the arithmetic at the DEFAULT month capacity
    /// unless it is testing that knob, so the number under test is the one
    /// production computes.
    const MONTHS: usize = 240;

    const DEFAULT_GENES: usize = 200;
    const DEFAULT_TERMS: usize = 3_200;

    fn sizing(
        free: u64,
        resident_parent_rows: usize,
        evaluation_rows: usize,
        features: usize,
        months: usize,
        genes: usize,
        terms: usize,
    ) -> Sizing {
        candidates_for_pre_parent_free_memory(
            free,
            resident_parent_rows,
            evaluation_rows,
            features,
            months,
            genes,
            terms,
        )
        .expect("checked sizing inputs")
    }

    fn fits(
        free: u64,
        resident_parent_rows: usize,
        evaluation_rows: usize,
        features: usize,
        months: usize,
        genes: usize,
        terms: usize,
    ) -> (usize, usize, usize) {
        match sizing(
            free,
            resident_parent_rows,
            evaluation_rows,
            features,
            months,
            genes,
            terms,
        ) {
            Sizing::Fits {
                memory_ceiling,
                time_ceiling,
                chosen,
            } => (memory_ceiling, time_ceiling, chosen),
            other => panic!("expected a fit, got {other:?}"),
        }
    }

    #[test]
    fn strict_default_plan_is_4000_bytes_per_scenario_with_zero_outcomes() {
        let plan = PopulationMetricsOnlyPlanV1::checked_from_session_extents_v1(1, 240)
            .expect("default strict plan");
        assert_eq!(plan.metric_rows_bytes(), 104);
        assert_eq!(plan.monthly_pnls_bytes(), 1_920);
        assert_eq!(plan.month_start_equities_bytes(), 1_920);
        assert_eq!(plan.scenario_descriptor_bytes(), 56);
        assert_eq!(plan.outcome_bytes(), 0);
        assert_eq!(plan.total_device_bytes(), 4_000);
        assert_ne!(plan.total_device_bytes(), 8_192 * 72 + 4_000);
    }

    #[test]
    fn full_resident_parent_not_stage1_rows_controls_admission() {
        const RESIDENT_PARENT_ROWS: usize = 1_049_160;
        const STAGE1_ROWS: usize = 262_290;
        const FEATURES: usize = 1_800;
        const SNAPSHOT: u64 = 20 * 1024 * 1024 * 1024;

        assert!(matches!(
            sizing(
                SNAPSHOT,
                RESIDENT_PARENT_ROWS,
                STAGE1_ROWS,
                FEATURES,
                MONTHS,
                DEFAULT_GENES,
                DEFAULT_TERMS,
            ),
            Sizing::NoRoom { .. }
        ));
        assert!(matches!(
            sizing(
                SNAPSHOT,
                STAGE1_ROWS,
                STAGE1_ROWS,
                FEATURES,
                MONTHS,
                DEFAULT_GENES,
                DEFAULT_TERMS,
            ),
            Sizing::Fits { .. }
        ));

        let full = PopulationParentDevicePlanV1::checked_from_parent_extents_v1(
            RESIDENT_PARENT_ROWS,
            FEATURES,
        )
        .expect("full parent plan");
        assert_eq!(full.total_device_bytes(), 15_187_640_160);
    }

    #[test]
    fn resident_parent_and_evaluation_extents_change_independent_terms() {
        const GIB: u64 = 1024 * 1024 * 1024;

        let (memory, time, chosen) = fits(
            24 * GIB,
            100_000,
            843_456,
            64,
            MONTHS,
            DEFAULT_GENES,
            DEFAULT_TERMS,
        );
        let (same_memory, shorter_time, shorter_chosen) = fits(
            24 * GIB,
            100_000,
            1_686_912,
            64,
            MONTHS,
            DEFAULT_GENES,
            DEFAULT_TERMS,
        );
        assert_eq!(memory, same_memory, "evaluation rows changed VRAM sizing");
        assert_ne!(
            time, shorter_time,
            "evaluation rows did not change time sizing"
        );
        assert_eq!(chosen, time);
        assert_eq!(shorter_chosen, shorter_time);

        let (small_parent_memory, same_time, small_parent_chosen) = fits(
            2 * GIB,
            1_000,
            4_096,
            1_800,
            MONTHS,
            DEFAULT_GENES,
            DEFAULT_TERMS,
        );
        let (large_parent_memory, same_time_again, large_parent_chosen) = fits(
            2 * GIB,
            50_000,
            4_096,
            1_800,
            MONTHS,
            DEFAULT_GENES,
            DEFAULT_TERMS,
        );
        assert_eq!(
            same_time, same_time_again,
            "parent rows changed time sizing"
        );
        assert!(small_parent_memory > large_parent_memory);
        assert_eq!(small_parent_chosen, small_parent_memory);
        assert_eq!(large_parent_chosen, large_parent_memory);
    }

    #[test]
    fn time_extent_is_the_maximum_effective_scenario_window() {
        let scenarios = [
            ScenarioDescriptor {
                window_offset: 100,
                window_len: 500,
                ..ScenarioDescriptor::default()
            },
            ScenarioDescriptor {
                window_offset: 200,
                window_len: 0,
                ..ScenarioDescriptor::default()
            },
        ];
        assert_eq!(
            max_effective_scenario_window_rows_v1(&scenarios, 1_000)
                .expect("validated effective window"),
            800
        );
    }

    #[test]
    fn exact_gene_store_is_unsplittable_and_can_exceed_the_old_reserve() {
        const QUALITY_SCREEN_GENES: usize = 262_144;
        const TERMS_PER_GENE: usize = 20;
        let term_count = QUALITY_SCREEN_GENES * TERMS_PER_GENE;
        let genes = PopulationGeneStorePlanV1::checked_from_gene_extents_v1(
            QUALITY_SCREEN_GENES,
            term_count,
        )
        .expect("quality-screen gene store");
        assert_eq!(genes.total_device_bytes(), 79_429_724);
        assert!(genes.total_device_bytes() > ALLOCATOR_RESERVE_BYTES_V1);

        assert!(matches!(
            sizing(128 * 1024 * 1024, 1, 1, 1, MONTHS, 1, 1),
            Sizing::Fits { .. }
        ));
        assert!(matches!(
            sizing(
                128 * 1024 * 1024,
                1,
                1,
                1,
                MONTHS,
                QUALITY_SCREEN_GENES,
                term_count,
            ),
            Sizing::NoRoom { .. }
        ));
    }

    #[test]
    fn month_capacity_is_charged_by_the_authoritative_scenario_plan() {
        const FREE: u64 = 1024 * 1024 * 1024;
        let (default_memory, _, _) =
            fits(FREE, 1_000, 4_096, 64, MONTHS, DEFAULT_GENES, DEFAULT_TERMS);
        let (doubled_memory, _, _) = fits(
            FREE,
            1_000,
            4_096,
            64,
            2 * MONTHS,
            DEFAULT_GENES,
            DEFAULT_TERMS,
        );
        assert!(doubled_memory < default_memory);
    }

    #[test]
    fn smaller_admitted_snapshot_produces_a_smaller_memory_ceiling() {
        let (large_memory, large_time, large_chosen) = fits(
            4 * 1024 * 1024 * 1024,
            1_000,
            4_096,
            64,
            MONTHS,
            DEFAULT_GENES,
            DEFAULT_TERMS,
        );
        let (small_memory, small_time, small_chosen) = fits(
            2 * 1024 * 1024 * 1024,
            1_000,
            4_096,
            64,
            MONTHS,
            DEFAULT_GENES,
            DEFAULT_TERMS,
        );
        assert_eq!(large_time, small_time);
        assert!(small_memory < large_memory);
        assert_eq!(large_chosen, large_memory);
        assert_eq!(small_chosen, small_memory);
    }

    #[test]
    fn one_to_fifteen_scenario_capacity_is_a_real_fit_not_immutable_no_room() {
        let fixed = PopulationParentDevicePlanV1::checked_from_parent_extents_v1(1, 1)
            .expect("one-row parent")
            .total_device_bytes()
            + PopulationGeneStorePlanV1::checked_from_gene_extents_v1(1, 1)
                .expect("one-gene store")
                .total_device_bytes()
            + ALLOCATOR_RESERVE_BYTES_V1;
        let desired_budget = fixed + 15 * 4_000;
        let snapshot = desired_budget.div_ceil(7) * 10;
        let (memory, _, chosen) = fits(snapshot, 1, 1, 1, MONTHS, 1, 1);
        assert_eq!(memory, 15);
        assert_eq!(chosen, 15);
    }

    #[test]
    fn zero_scenario_capacity_has_a_distinct_fail_loud_classification() {
        let fixed = PopulationParentDevicePlanV1::checked_from_parent_extents_v1(1, 1)
            .expect("one-row parent")
            .total_device_bytes()
            + PopulationGeneStorePlanV1::checked_from_gene_extents_v1(1, 1)
                .expect("one-gene store")
                .total_device_bytes()
            + ALLOCATOR_RESERVE_BYTES_V1;
        let snapshot = fixed.div_ceil(7) * 10;
        match sizing(snapshot, 1, 1, 1, MONTHS, 1, 1) {
            Sizing::NoScenarioRoom {
                available_device_bytes,
                one_scenario_device_bytes,
            } => assert!(available_device_bytes < one_scenario_device_bytes),
            other => panic!("expected distinct scenario-space refusal, got {other:?}"),
        }
    }

    /// Short series make the time term generous, and the occupancy knee is the
    /// floor that keeps it from ever being the thing that starves the card.
    #[test]
    fn the_time_term_never_starves_a_short_series() {
        assert_eq!(
            checked_candidates_for_target_launch_v1(usize::MAX)
                .expect("checked time ceiling")
                .1,
            OCCUPANCY_KNEE,
            "even an absurd bar count must not push the launch under the knee"
        );
        assert!(
            checked_candidates_for_target_launch_v1(4_096)
                .expect("checked time ceiling")
                .1
                > OCCUPANCY_KNEE
        );
        // The floor really does OVERRIDE the target on a long series, and that
        // is a deliberate trade rather than a bound: at EURUSD M1 dimensions the
        // honest time term is ~3 200 scenarios and the floor lifts it to 16 384,
        // an estimated 102 s launch against a 20 s target. It is pinned here so
        // the occupancy override remains explicit in the checked result, and so
        // that nobody reads the target as a guarantee.
        const M1_BARS: u64 = 5_270_000;
        assert!(
            SCENARIO_BARS_PER_SECOND * TARGET_LAUNCH_SECONDS / M1_BARS < OCCUPANCY_KNEE,
            "the long-series case is what the override warning exists for"
        );
    }

    #[test]
    fn previous_strict_transients_are_released_before_a_new_gene_allocation() {
        let source = include_str!("../../../neoethos-gpu-cuda/native/prototype_b_population.cu");
        let start = source
            .find("neoethos_gpu_cuda_population_upload_genes(")
            .expect("native gene upload entry point");
        let rest = &source[start..];
        let end = rest
            .find("neoethos_gpu_cuda_population_upload_scenarios(")
            .expect("next native upload entry point");
        let upload = &rest[..end];
        let release_scenarios = upload
            .find("session->release_scenarios();")
            .expect("old scenario arrays must be released");
        let release_workspace = upload
            .find("session->release_workspace();")
            .expect("old strict workspace must be released");
        let first_gene_allocation = upload
            .find("device_alloc(&session->candidate_ids")
            .expect("first new gene allocation");
        assert!(release_scenarios < first_gene_allocation);
        assert!(release_workspace < first_gene_allocation);
    }

    fn base_limit_key() -> LimitKey {
        LimitKey {
            device: 7,
            cuda_device_identity_sha256: "cuda-device-a".to_owned(),
            parent_dataset_identity_sha256: "parent-a".to_owned(),
            pre_parent_free_memory_bytes: 23_000_000_000,
            resident_parent_rows: 1_049_160,
            evaluation_rows: 262_290,
            feature_count: 1_800,
            month_capacity: 240,
            gene_count: 200,
            gene_term_count: 3_200,
        }
    }

    #[test]
    fn learned_limit_binds_parent_snapshot_and_exact_gene_shape() {
        let base = base_limit_key();
        learn_batch_success(&base, 26_777);
        assert_eq!(learned_batch_limit(&base), 26_777);

        let mut variants = Vec::new();
        let mut different_parent_rows = base.clone();
        different_parent_rows.resident_parent_rows += 1;
        variants.push(different_parent_rows);
        let mut different_parent = base.clone();
        different_parent.parent_dataset_identity_sha256 = "parent-b".to_owned();
        variants.push(different_parent);
        let mut different_snapshot = base.clone();
        different_snapshot.pre_parent_free_memory_bytes -= 1;
        variants.push(different_snapshot);
        let mut different_gene_terms = base.clone();
        different_gene_terms.gene_term_count += 1;
        variants.push(different_gene_terms);
        let mut different_device_identity = base.clone();
        different_device_identity.cuda_device_identity_sha256 = "cuda-device-b".to_owned();
        variants.push(different_device_identity);

        for variant in &variants {
            assert_eq!(
                learned_batch_limit(variant),
                usize::MAX,
                "a learned fit leaked across a distinct sizing receipt fact: {variant:?}"
            );
        }
        learn_batch_failure(&variants[0], 512);
        assert_eq!(learned_batch_limit(&variants[0]), 512);
        assert_eq!(learned_batch_limit(&base), 26_777);
    }

    #[test]
    fn an_unsplittable_allocation_failure_is_never_split() {
        let workspace: Result<()> = Err(native_error(neoethos_gpu_cuda::STATUS_ALLOCATION_FAILED))
            .map_err(anyhow::Error::new)
            .context("prototype B evaluate");
        assert!(
            is_capacity_exhaustion(&workspace.expect_err("constructed as an error")),
            "a workspace exhaustion IS worth retrying smaller"
        );

        let parent: Result<()> = Err(native_error(neoethos_gpu_cuda::STATUS_ALLOCATION_FAILED))
            .map_err(anyhow::Error::new)
            .context(UnsplittablePopulationAllocationV1(
                "the immutable native population parent upload",
            ))
            .context("upload immutable native population parent");
        let error = parent.expect_err("constructed as an error");
        assert!(
            !is_capacity_exhaustion(&error),
            "a parent upload failure is the same size at every leaf: {error:#}"
        );
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("upload immutable native population parent"),
            "{rendered}"
        );
        assert!(
            rendered.contains("does not depend on the work list size"),
            "{rendered}"
        );
    }

    #[cfg(feature = "gpu-b-native")]
    #[test]
    #[ignore = "requires a real CUDA device"]
    fn real_admitted_cuda_snapshot_sizes_full_parent_and_exact_genes() {
        let admission =
            crate::acquire_strict_discovery_device_admission_v1().expect("real CUDA admission");
        let route = admission.into_route_v1();
        let admitted = route
            .require_exact_cuda_device_ordinal_v1()
            .expect("admission must seal a CUDA ordinal");
        assert!(admitted.pre_parent_free_memory_bytes() > 0);
        let ceiling = runtime_submission_ceiling_for_admitted_ordinal_v1(
            admitted, 1_049_160, 262_290, 1_800, 240, 200, 3_200,
        )
        .expect("the admitted RTX route must fit the exact strict plan");
        assert!(ceiling >= 16, "admitted strict ceiling is {ceiling}");
    }

    /// Built exactly as the engine builds it, so the test exercises the real
    /// shape rather than a convenient stand-in.
    fn native_error(status: i32) -> CudaPopulationError {
        CudaPopulationError::Native {
            operation: "evaluate",
            status,
            message: neoethos_gpu_cuda::population_status_message(status),
        }
    }

    /// The call sites attach context; this proves the typed error survives it.
    ///
    /// It did not: every site formatted the error into a fresh `anyhow!`, so
    /// the retry-smaller check could never find it. The population that could
    /// not fit went to the CPU instead of being halved, and a measured run put
    /// 770 500 of 778 205 validation items there with the card idle.
    #[test]
    fn out_of_memory_survives_the_context_the_call_sites_attach() {
        let native: Result<()> = Err(native_error(neoethos_gpu_cuda::STATUS_ALLOCATION_FAILED))
            .map_err(anyhow::Error::new)
            .context("prototype B evaluate");
        let error = native.expect_err("constructed as an error");
        assert!(
            is_capacity_exhaustion(&error),
            "an out-of-memory has to stay recognisable through the context: {error:#}"
        );

        // The rendering the log uses must still name the cause, or the next
        // investigation starts blind again.
        let rendered = format!("{error:#}");
        assert!(rendered.contains("prototype B evaluate"), "{rendered}");
        assert!(rendered.contains("device allocation failed"), "{rendered}");
    }

    /// A launch failure is a fault, not a size problem.
    #[test]
    fn a_launch_failure_is_not_retried_smaller() {
        let error = anyhow::Error::new(native_error(neoethos_gpu_cuda::STATUS_LAUNCH_FAILED))
            .context("prototype B evaluate");
        assert!(!is_capacity_exhaustion(&error));
    }

    #[test]
    fn resident_reuse_is_owned_by_the_run_scoped_sealed_native_boundary() {
        let source = include_str!("prototype_b_population_eval.rs");
        let production = &source[source
            .rfind("\nfn evaluate_population_b_batch(")
            .expect("native population adapter boundary")..];
        assert!(production.contains("bind_exact_native_population_view_v1"));
        assert!(!production.contains("fn resident_slot()"));
        assert!(!production.contains("static RESIDENT"));
        assert!(!production.contains("sample_hash"));
        assert!(!production.contains("dataset_key"));
    }

    #[test]
    fn prototype_b_f64_adapter_contract_has_no_cubecl_diversion() {
        let source = include_str!("../cubecl_eval.rs").to_ascii_lowercase();
        for forbidden in [
            "prototype_b",
            "prototype b",
            "prototype-b",
            "gpu-b-adapter",
            "try_evaluate_population_b",
        ] {
            assert!(
                !source.contains(forbidden),
                "CubeCL must remain a separate engine; found stale diversion token {forbidden}"
            );
        }
    }

    #[test]
    fn prototype_b_f64_adapter_contract_real_parity_is_fail_loud() {
        let source = include_str!("../eval.rs");
        let start = source
            .find("fn gpu_matches_cpu_with_a_trailing_stop()")
            .expect("real Prototype B parity test must exist");
        let rest = &source[start..];
        let end = rest
            .find("fn uniform_buckets_are_a_scalar_by_another_name()")
            .expect("next parity test must delimit the real trailing-stop test");
        let parity_test = &rest[..end];

        assert!(
            !parity_test.contains("skipping trailing parity"),
            "a device error must fail the paid real-device parity test, never report a skip"
        );
        assert!(
            parity_test.contains("Prototype B real-device parity failed"),
            "the parity test must surface a device failure with an explicit fail-loud diagnostic"
        );
    }
}

/// Cut the WORK LIST and evaluate each part.
///
/// This used to slice the CSR gene arrays and rebase the tail's offsets so its
/// first gene started at zero — a manoeuvre that, done wrong, evaluates the
/// wrong terms silently rather than erroring. None of that is needed any more:
/// every descriptor carries its own gene index, so the genes stay whole and
/// resident and a slice of the descriptor array is still a correct work list
/// whatever it contains. The class of bug is gone rather than guarded.
#[allow(clippy::too_many_arguments)]
fn split_and_evaluate(
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
    scenarios: &[ScenarioDescriptor],
    head_len: usize,
) -> Result<NativePopulationBatchV1> {
    // Everything launched below this point is a split leaf. Without it the
    // telemetry sees only the outer entry: a `calls=7` line covered an unknown
    // number of real launches, and "unknown" is what let the recursive split be
    // mistaken for a single submission twice over. The guard nests, so a leaf
    // three halvings deep still reports as a leaf.
    let _split = crate::eval_telemetry::SplitScope::enter();
    let n_scenarios = scenarios.len();
    // Cut at the size that fits rather than in half.
    //
    // Halving overshoots whenever the work list is a little over the limit: a
    // measured run had 25 600 items against a card holding 12 334, halved to
    // 12 800, halved again to 6 400 — four launches each using half the card,
    // where three full ones would do. The caller knows the limit when the split
    // is pre-emptive; after a capacity failure it does not, and passes half.
    let cut = head_len.clamp(1, n_scenarios.saturating_sub(1));

    // The gene arrays are passed through UNCHANGED to both halves. That is the
    // point: a descriptor's `base_candidate_id` indexes the full population, so
    // slicing the genes would invalidate every index in the work list. Uploading
    // the whole gene array twice costs a few megabytes; slicing it wrongly costs
    // a run that reports numbers from the wrong strategies.
    let mut head = evaluate_scenarios_b_raw_v1(
        evidence,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
        &scenarios[..cut],
    )?;
    let tail = evaluate_scenarios_b_raw_v1(
        evidence,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        sl_pips,
        tp_pips,
        stop_vol_mult,
        gene_smc_flags,
        gate_threshold,
        smc_weights,
        &scenarios[cut..],
    )?;
    head.rows.extend(tail.rows);
    head.counters = tail.counters;
    Ok(head)
}

/// Evaluate a population on Prototype B.
///
/// This is the canonical f64 native-CUDA adapter. CubeCL remains a separate
/// engine and never diverts into this path.
#[allow(clippy::too_many_arguments)]
fn evaluate_population_b_batch(
    evidence: &crate::population_execution_evidence_v1::ExactPopulationEvaluationV1,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f64],
    long_thr: &[f64],
    short_thr: &[f64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    stop_vol_mult: &[f64],
    gene_smc_flags: &[SmcRow],
    gate_threshold: f64,
    smc_weights: &[f64; 11],
    device: usize,
    scenarios: &[ScenarioDescriptor],
) -> Result<NativePopulationBatchV1> {
    // Host prep starts here, not at the first upload.
    //
    // Everything between this line and the submission below is the card
    // waiting: hashing the dataset key, sizing the session, staging genes and
    // scenarios, and — when the session is not reusable — freeing and
    // re-uploading the entire workspace. It was never measured, so "the launch
    // took 23 s" and "the kernel took 0.30 s" could both be reported without
    // anyone being able to say where the other 22.7 s went.
    let host_prep_started = std::time::Instant::now();
    let n_genes = long_thr.len();
    let n_scenarios = scenarios.len();
    let bars = evidence.row_count();
    if n_genes == 0 || bars == 0 || n_scenarios == 0 {
        return Ok(NativePopulationBatchV1 {
            rows: vec![ZERO_METRICS; n_scenarios],
            counters: PopulationResidencyCountersV1::default(),
        });
    }
    // Same optional-contract handling as the CubeCL lane: an empty
    // `stop_vol_mult` means "no adaptive stops", and every downstream slice
    // would otherwise index out of range.
    let stop_vol_fallback = crate::eval::normalized_stop_vol_mult(stop_vol_mult, n_genes);
    let stop_vol_mult = stop_vol_fallback.as_deref().unwrap_or(stop_vol_mult);

    let native_device = i32::try_from(device)
        .map_err(|_| anyhow::anyhow!("sealed CUDA ordinal {device} does not fit the native ABI"))?;
    let native_settings = population_settings_for_settings(evidence.settings())
        .map_err(anyhow::Error::new)
        .context("prototype B settings")?;

    // Genes and scenarios are what actually change between calls, so they are
    // the only things rebuilt and re-uploaded.
    let genes = PrototypeAGeneUpload {
        candidate_ids: (0..n_genes as u64).collect(),
        offsets: gene_offsets.to_vec(),
        indices: gene_indices.to_vec(),
        weights: gene_weights.to_vec(),
        long_thresholds: long_thr.to_vec(),
        short_thresholds: short_thr.to_vec(),
        stop_pips: sl_pips.to_vec(),
        target_pips: tp_pips.to_vec(),
        stop_vol_multipliers: stop_vol_mult.to_vec(),
        smc_flags: gene_smc_flags.to_vec(),
        smc_weights: *smc_weights,
        gate_threshold,
    };
    let descriptors = genes
        .candidate_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(index, candidate_id)| {
            let start = genes.offsets[index] as u32;
            let end = genes.offsets[index + 1] as u32;
            GeneDescriptor {
                candidate_id,
                term_offset: start,
                term_count: end.saturating_sub(start),
                long_threshold: genes.long_thresholds[index],
                short_threshold: genes.short_thresholds[index],
                stop_ticks: 0,
                target_ticks: 0,
                stop_vol_multiplier: genes.stop_vol_multipliers[index],
                flags: 0,
                reserved: 0,
            }
        })
        .collect::<Vec<_>>();
    let smc_flags = genes
        .smc_flags
        .iter()
        .flatten()
        .copied()
        .collect::<Vec<_>>();

    // The work list, as the caller built it.
    //
    // THIS BLOCK USED TO CONSTRUCT THE DESCRIPTORS ITSELF — one per gene, with
    // every field except `scenario_id` a literal zero, and a comment explaining
    // that costs were carried by the settings "so every per-scenario knob stays
    // zero". The knobs were zero because nothing read them; the engine did not
    // "reject anything else", it ignored everything else. That is what made a
    // screen wanting three treatments of the same genes need three launches.
    //
    // The caller now decides. `try_evaluate_population_b` still builds one
    // full-series `base_scenario` per gene and that path is bit-identical to the
    // old literal — with one field genuinely changed: `spread_ticks` and
    // `commission_micros` are `-1` ("no override") where they were `0`, and `0`
    // now means "charge nothing". Any descriptor built outside
    // `scenario::base_scenario` / `cost_scenario` / `perturb_scenario` is how
    // that becomes a free-trading backtest, which is why there is no third
    // construction site.
    let ((rows, counters, host_prep, device_elapsed), residency_counters) = evidence
        .bind_exact_native_population_view_v1(native_device, |session| {
            session
                .upload_genes(PopulationGeneView {
                    descriptors: &descriptors,
                    offsets: &genes.offsets,
                    indices: &genes.indices,
                    weights: &genes.weights,
                    stop_pips: &genes.stop_pips,
                    target_pips: &genes.target_pips,
                    stop_vol_multipliers: &genes.stop_vol_multipliers,
                    smc_flags: &smc_flags,
                    smc_weights: &genes.smc_weights,
                    gate_threshold: genes.gate_threshold,
                    smc_gate_disabled: crate::genetic::smc_gate_disabled(),
                })
                .map_err(anyhow::Error::new)
                .context(UnsplittablePopulationAllocationV1("the gene upload"))
                .context("prototype B gene upload")?;
            session
                .upload_scenarios(scenarios)
                .map_err(anyhow::Error::new)
                .context("prototype B scenario upload")?;

            // Parent/view bind plus changing gene/scenario staging are host
            // prep. Evaluation itself uses the strict metrics-only workspace:
            // no outcome ledger, seed kernel, or accepted-total scalar exists.
            let host_prep = host_prep_started.elapsed();
            let device_started = std::time::Instant::now();
            let host_metrics = session
                .enqueue_metrics_only_v1(&native_settings)?
                .consume_host_metrics_v1()?;
            let counters = host_metrics.counters();
            let rows = host_metrics.into_metric_rows();
            Ok((rows, counters, host_prep, device_started.elapsed()))
        })?;
    let session_rebuilt = residency_counters.parent_upload_count() == 1
        && residency_counters.view_binding_count() == 1;

    crate::eval_telemetry::record_launch(
        crate::eval_telemetry::current_lane(),
        crate::eval_telemetry::LaunchRecord {
            host_prep,
            device: device_elapsed,
            kernel_submissions: counters.kernel_submissions,
            synchronization_events: counters.synchronization_events,
            session_rebuilt,
            split_leaf: crate::eval_telemetry::inside_split(),
        },
    );

    // The strict workspace has no outcome ledger or accepted-trade scalar.
    // Slot 8 is the authoritative per-scenario trade count already present in
    // the single bounded metric readback.
    let trade_counts = rows
        .iter()
        .map(|row| row.values[8])
        .filter(|count| count.is_finite());
    let (peak, total) = trade_counts.fold((0.0f64, 0.0f64), |(peak, total), count| {
        (peak.max(count), total + count)
    });
    tracing::info!(
        target: "neoethos_search::eval",
        n_genes,
        n_scenarios,
        busiest_candidate = peak as u64,
        mean_trades = (total / (rows.len() as f64).max(1.0)) as u64,
        "strict metrics-only trade counts — no diagnostic outcome slots allocated"
    );

    // Where this launch's time went.
    //
    // This was behind a `std::sync::Once` — "it is a property of the data, not
    // of the call". That was wrong twice over. The population size changes on
    // every recursive split, so the first launch is the LEAST representative one
    // there is; and the number that actually matters is not the event budget at
    // all but the ratio below. A once-per-process line cannot show a ratio that
    // varies per launch, and a lane whose cost is per-call is exactly the lane
    // where a single sample is worthless.
    //
    // So it is per launch now. The GA drives a few hundred of these in a run,
    // which is the same order as the per-call `trade slot usage` line already
    // emitted above, and INFO rather than DEBUG because the app installs its own
    // subscriber and never enables DEBUG — a diagnostic nobody sees is not a
    // diagnostic.
    //
    // Three fields are GONE from this line: `event_capacity`, `emitted_events`
    // and `capacity_used_pct`. They were the most confidently wrong numbers in
    // the run. `emitted_events` is not a count of events — nothing emits events
    // — it is `population * MAX_TRADES_PER_CANDIDATE`, the size of the outcome
    // array, so it was a constant multiple of the population dressed as a
    // measurement. `event_capacity` was a budget for a buffer with no
    // allocation. Their ratio, printed as `capacity_used_pct`, was therefore a
    // pure function of the population and told an investigator precisely
    // nothing while looking exactly like the answer. What a candidate actually
    // records is the `trade slot usage` line above, which reads real metric
    // rows.
    {
        let host_prep_ms = host_prep.as_secs_f64() * 1e3;
        let device_ms = device_elapsed.as_secs_f64() * 1e3;
        let accounted = host_prep_ms + device_ms;
        tracing::info!(
            target: "neoethos_search::eval",
            lane = crate::eval_telemetry::current_lane(),
            n_genes,
            // Both, because they are no longer the same number and the ratio is
            // the measurement: 174 genes carrying 17 574 scenarios is one launch
            // where it used to be seven.
            n_scenarios,
            bars,
            session_rebuilt,
            split_leaf = crate::eval_telemetry::inside_split(),
            kernel_submissions = counters.kernel_submissions,
            sync_events = counters.synchronization_events,
            host_prep_ms = format!("{host_prep_ms:.1}"),
            device_ms = format!("{device_ms:.1}"),
            device_pct = format!(
                "{:.1}",
                if accounted > 0.0 { 100.0 * device_ms / accounted } else { 0.0 }
            ),
            "launch anatomy — host prep against device time for THIS launch"
        );
    }

    if rows.len() != n_scenarios {
        bail!(
            "prototype B returned {} metric rows for {} scenarios",
            rows.len(),
            n_scenarios
        );
    }
    // Rows come back in scenario order, and each one is checked against the
    // descriptor that asked for it.
    //
    // The old check demultiplexed by `candidate_id`, which worked only while
    // there was exactly one scenario per gene: a Monte-Carlo work list has 100
    // scenarios sharing one `candidate_id`, so "seen this id twice" would fire
    // on a correct result and the id would no longer identify a row. Matching
    // `scenario_id` positionally is a STRICTLY STRONGER check — it catches the
    // same permutation without requiring gene ids to be unique — and it is the
    // reason the descriptor carries an id the host chose rather than a position
    // the host has to trust.
    let mut out = vec![ZERO_METRICS; n_scenarios];
    for (index, row) in rows.iter().enumerate() {
        let expected = scenarios[index].scenario_id;
        if row.scenario_id != expected {
            bail!(
                "prototype B returned scenario {} at position {index}, where the work list \
                 asked for scenario {expected} — the results are permuted and every metric \
                 would be attributed to the wrong run",
                row.scenario_id
            );
        }
        out[index] = row.values;
    }
    Ok(NativePopulationBatchV1 {
        rows: out,
        counters: residency_counters,
    })
}
