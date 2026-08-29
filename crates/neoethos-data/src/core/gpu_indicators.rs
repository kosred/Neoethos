//! vector-ta CUDA indicator lane — device-resident, **f64**, multi-period.
//!
//! Everything in this file is `#[cfg(feature = "gpu-cuda")]`-gated at the
//! module declaration in `core/mod.rs`, so a card-less build never compiles a
//! byte of it.
//!
//! # What this replaces
//!
//! `hpc_ta::compute_classic_ta_columns` runs a configured multi-indicator ×
//! multi-period sweep on the CPU: independent full-series scans, each re-reading the same OHLCV
//! from host memory. This module uploads the series ONCE and keeps it resident
//! while every indicator in the table below is swept against it, so there is no
//! device→host→device round trip between indicators. That round trip — not the
//! kernel time — is what makes a host stage expensive; "it is only 5.6% of
//! runtime" is not a reason to leave work on the CPU, because the 5.6% is
//! measured on the stage, not on the residency it breaks.
//!
//! # THE PRECISION CHANGE (2026-08-09)
//!
//! This lane used to narrow f64 → f32 on upload and widen f32 → f64 on return,
//! because vector-ta's shared device layer (`CudaDeviceVectorF32`,
//! `upload_f32`, `IndicatorCudaSeries::HostF32`) has no f64 in it at all, and
//! because NO indicator in the table below had an f64 kernel: their
//! `*_kernel.cu` files are f32 in and f32 out. (`sma_kernel.cu` LOOKS f64
//! because five entry points carry a `_f64` suffix, but that names an f64
//! ACCUMULATOR inside an f32 pipeline — `sma_prefix_stage1_scan_f64` takes
//! `const float* prices`.)
//!
//! Both halves are now fixed in the vendored crate:
//!
//! * `vendor/vector-ta-0.2.9-patched/src/cuda/device_types_f64.rs` adds the
//!   f64 device vocabulary alongside the f32 one, and
//!   `CudaRuntime::upload_ohlcv_f64` / `upload_f64` upload without narrowing;
//! * `vendor/vector-ta-0.2.9-patched/kernels/cuda/neoethos_f64_kernels.cu`
//!   holds `*_batch_f64` kernels written against this crate's own f64 CPU
//!   implementations, compiled with `-prec-div=true -prec-sqrt=true
//!   -fmad=false -ftz=false` and NEVER with `--use_fast_math` regardless of
//!   `CUDA_FAST_MATH`.
//!
//! So both documented divergence sources are gone and this lane is now a
//! PARITY claim rather than a measured-divergence claim.
//! `hpc_ta::gpu_cpu_indicator_sweep_parity` is what proves it on real bars;
//! until that has run on a card, treat the claim as UNVERIFIED, not as fact.
//!
//! # The three things that decide whether this lane may run
//!
//! 1. **The arch trap — closed fail-closed.** vector-ta emits one verified
//!    native cubin per kernel and exact requested or visible architecture.
//!    Runtime selection accepts only the current device's exact `sm_X` entry;
//!    a missing architecture is an error, never a nearest-architecture choice
//!    or a different execution lane. [`GpuIndicatorEngine::new`] still proves
//!    the indicator path by loading a module and launching a kernel.
//! 2. **No silent fallback.** Every failure here is an `Err` carrying the
//!    device error. Nothing in this module computes a CPU value. The caller
//!    (`hpc_ta::compute_classic_ta_columns_with_policy`) decides, and whatever
//!    it decides is recorded by name in `indicator_telemetry`.
//! 3. **No silent precision drop.** An indicator with no f64 kernel produces
//!    `IndicatorDispatchError::CudaF64KernelMissing` naming it. It is never
//!    served by the f32 kernel.
//!
//! # Why the historical table below is short
//!
//! Only indicators whose DEVICE contract is verified — same input series, same
//! parameter meaning, single output — are listed, and the input contract is
//! now CROSS-CHECKED against `vector_ta::indicators::dispatch::F64_KERNELS` at
//! test time and again on every sweep, so the two cannot drift.
//!
//! Two entries need their price series chosen explicitly:
//!
//! * `cci` — the CPU path sources `hlc3` (`cpu_batch.rs:3401`), so it is fed an
//!   explicit `Slice` over the resident hlc3 upload;
//! * `mfi` — same (`cpu_batch.rs:2867`), fed an explicit `CloseVolume` built
//!   from hlc3 and volume.
//!
//! Passing the OHLCV ref instead would hand the kernel `close` and compute a
//! DIFFERENT INDICATOR, not a less precise one.
//!
//! That table is retained only for the old host-returning parity sweep while
//! the complete resident plan is built. It is not the `GpuOnly` capability
//! authority: [`f64_primary_device_route_for`] resolves every primary row in
//! the vector-ta f64 registry,
//! and `GpuOnly` fails before work until every requested output and downstream
//! node is resident and promoted. No id outside the table is sent to the CPU
//! inside a GPU run. The remaining historical multi-period limitation is:
//!
//! * `stoch`, `macd`, `bollinger_bands`, `keltner`, `supertrend` are
//!   MULTI-OUTPUT. `resolve_output_id` (cpu_batch.rs:2185) returns
//!   `Err(InvalidParam)` for a multi-output indicator when `output_id` is
//!   `None`, which is how `hpc_ta` calls it, and `hpc_ta.rs:291` swallows that.
//!   So these five emit ZERO columns on EITHER lane today. A device kernel for
//!   them would have no CPU column to be checked against; the thing to fix
//!   first is the CPU call, not the kernel.
//!
//! There used to be a second reason and it is GONE (2026-08-10). `vwap` was
//! withheld because vector-ta answered "what is vwap" two ways — a second
//! implementation, `vwap_row_scalar_pv`, reached only by the `Kernel::Scalar`
//! arm of `vwap_batch_inner`, accumulated `price * volume` with TWO roundings
//! where `vwap_scalar` uses one `mul_add` — so there was no single CPU answer
//! for the device to match. vector-ta fixed that AT THE SOURCE rather than by
//! tolerance: `vwap_row_scalar_pv` was deleted, both batch arms now call
//! `vwap_row_scalar`, `vwap` (and `wilders`, the other withheld id, whose
//! warm-up seed association was settled the same way) carry rows in
//! `cuda_f64::F64_KERNELS`, and `cuda_f64::WITHHELD_PENDING_CPU_SELF_CONSISTENCY`
//! is now `&[]`. The card-less test
//! `tests/f64_lane_cpu_reference.rs::scalar_and_auto_agree_for_every_claimed_indicator`
//! is what measures the agreement, on clean AND on gapped bars.
//!
//! So `tsi`, `obv` and now `vwap` all have real f64 primary routes. OBV and
//! VWAP have no window and are therefore reached only through the complete
//! primary registry, while every single-output id that is actually in the
//! historical period sweep can launch on the device.
//! The multi-output five still need explicit output routes; that gap keeps the
//! complete `GpuOnly` feature plan unavailable rather than creating a hybrid.
//!
//! No count is written here, and none should be added. That relationship is
//! ASSERTED instead, by
//! `tests::every_reachable_multi_period_id_with_an_f64_kernel_is_claimed`,
//! which fails the day vector-ta registers a kernel this table has not picked
//! up. Counts rot; assertions do not.

use super::super::Ohlcv;
use crate::core::indicator_telemetry::{
    IndicatorLane, VECTOR_TA_ARCH_SOURCE, VECTOR_TA_NATIVE_ARCHS,
};
use anyhow::{Context, Result, bail};

use vector_ta::cuda::{
    CudaDeviceCloseVolumeF64Ref, CudaDeviceHighLowF64Ref, CudaDeviceOhlcvF64, CudaDeviceVectorF64,
    CudaDeviceVectorI64, CudaF64Indicators, CudaRuntime, F64NamedOutputsResult, cuda_available,
};
use vector_ta::indicators::adaptive_bounds_rsi::AdaptiveBoundsRsiParams;
use vector_ta::indicators::adaptive_schaff_trend_cycle::AdaptiveSchaffTrendCycleParams;
use vector_ta::indicators::adjustable_ma_alternating_extremities::AdjustableMaAlternatingExtremitiesParams;
use vector_ta::indicators::alligator::AlligatorParams;
use vector_ta::indicators::alphatrend::AlphaTrendParams;
use vector_ta::indicators::bulls_v_bears::BullsVBearsParams;
use vector_ta::indicators::candle_strength_oscillator::CandleStrengthOscillatorParams;
use vector_ta::indicators::chandelier_exit::ChandelierExitParams;
use vector_ta::indicators::cksp::CkspParams;
use vector_ta::indicators::coppock::CoppockParams;
use vector_ta::indicators::dispatch::{
    CudaOutputTargetF64, F64FirstValidRule, F64InputKind, IndicatorCudaDeviceDataRefF64,
    IndicatorCudaDeviceRequestF64, IndicatorCudaOutputF64, IndicatorCudaSeriesF64,
    compute_cuda_device_f64, f64_kernel_for, has_f64_resident_output_route,
};
use vector_ta::indicators::fibonacci_entry_bands::FibonacciEntryBandsBatchRange;
use vector_ta::indicators::hema_trend_levels::HemaTrendLevelsParams;
use vector_ta::indicators::ichimoku_oscillator::IchimokuOscillatorBatchRange;
use vector_ta::indicators::ict_propulsion_block::IctPropulsionBlockParams;
use vector_ta::indicators::kase_peak_oscillator_with_divergences::KasePeakOscillatorWithDivergencesParams;
use vector_ta::indicators::market_structure_confluence::MarketStructureConfluenceParams;
use vector_ta::indicators::pivot::PivotParams;
use vector_ta::indicators::range_filtered_trend_signals::RangeFilteredTrendSignalsParams;
use vector_ta::indicators::range_oscillator::RangeOscillatorParams;
use vector_ta::indicators::registry::get_indicator;
use vector_ta::indicators::vdubus_divergence_wave_pattern_generator::VdubusDivergenceWavePatternGeneratorParams;

/// Which host series the device kernel must be fed so it computes the SAME
/// indicator the CPU path computes — not merely a lower-precision one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceInput {
    /// Kernel reads a single price series and the CPU default source is
    /// `close`. Served by the resident OHLCV ref (its `prices()` accessor is
    /// `source().unwrap_or(close())`, and we upload `source: None`).
    CloseFromOhlcv,
    /// Kernel reads high/low/close and the CPU path uses the same. Served by
    /// the resident OHLCV ref.
    Ohlc,
    /// Kernel reads a single price series but the CPU default source is
    /// `hlc3`. Served by an explicit `Slice` ref over the resident hlc3.
    Hlc3Slice,
    /// Kernel reads (typical price, volume) and the CPU path uses hlc3 as the
    /// typical price. Served by an explicit `CloseVolume` ref built from the
    /// resident hlc3 and volume.
    Hlc3CloseVolume,
    /// Kernel reads (close, volume) — `obv`, `vwma`, `efi`. Same device shape
    /// as [`Self::Hlc3CloseVolume`] and a DIFFERENT price series in it. The
    /// pair is built here from close rather than hlc3, so the distinction is
    /// made once, at the upload, instead of being re-derived per indicator.
    CloseVolume,
    /// Kernel reads (high, low) with no close — `medprice`, `midprice`. These
    /// carry their own first-valid rule, scanning high and low only, because
    /// their CPU `*_prepare` does.
    HighLow,
    /// Kernel reads (timestamps, close, volume) — `vwap` alone, whose anchor
    /// is a calendar bucket.
    TimestampCloseVolume,
    // -------------------------------------------------------------- shard 1
    /// Kernel reads a single price series whose CPU default source is `hl2` --
    /// `kurtosis` (cpu_batch.rs:3522) and `alligator` (:13912). A THIRD source
    /// alongside close and hlc3. Served by an explicit `Slice` over a resident
    /// hl2 upload, for exactly the reason `Hlc3Slice` exists: handing these
    /// kernels close computes a different indicator, not a less precise one.
    Hl2Slice,
    /// Kernel reads (high, low, volume) with NO close -- `emv` alone, whose CPU
    /// reference never touches close and whose first-valid scan covers the
    /// three series it does read.
    HighLowVolume,
    /// Kernel reads (high, low, close, volume) -- `kvo` alone.
    Hlcv,
    /// Kernel reads the FULL bar (open, high, low, close, volume) --
    /// `trend_flow_trail` and every other indicator whose CPU batch calls
    /// `extract_ohlcv_full_input`. Distinct from [`Self::Hlcv`] because OPEN
    /// gates validity: a bar with a non-finite open resets the CPU cascade,
    /// and a kernel handed the four-pointer shape would never see it.
    Ohlcv5,
    // -------------------------------------------------------------- shard 4
    /// Kernel reads (open, close, volume) -- `dvdiqqe` alone. Its CPU
    /// reference takes high and low as `_high` / `_low` and never reads them
    /// (`dvdiqqe.rs:447-448`), while `open` is read at every bar. Served by
    /// the resident OHLCV ref, the only upload carrying open.
    OpenCloseVolume,
    // -------------------------------------------------------------- closer 5
    /// Kernel reads a single price series whose CPU default source is
    /// `hlcc4` = (h + l + 2c) / 4 -- the `velocity` family. A FOURTH source
    /// alongside close, hlc3 and hl2, served by an explicit `Slice` over a
    /// resident hlcc4 upload for exactly the reason `Hl2Slice` exists:
    /// handing these kernels close computes a DIFFERENT indicator, not a
    /// less precise one.
    Hlcc4Slice,
    /// Kernel reads VOLUME alone with no price series at all -- `vosc`,
    /// whose CPU batch calls `extract_volume_input`. Served by an explicit
    /// `Slice` over the volume already resident inside the OHLCV upload, so
    /// this shape costs no extra transfer.
    VolumeSlice,
    // ------------------------------------------------------ closer 6, round 3
    /// Kernel reads (hlcc4, volume). The SAME device shape as
    /// [`Self::Hlc3CloseVolume`] and [`Self::CloseVolume`] with a THIRD price
    /// series in it; the pair is built here from the resident hlcc4 upload, so
    /// the distinction is made once, at the upload, and a kernel expecting
    /// hlcc4 can never be handed close.
    Hlcc4CloseVolume,
}

impl DeviceInput {
    /// Translate vector-ta's declaration of the kernel's source requirement.
    ///
    /// Deriving this rather than restating it means the day an f64 kernel is
    /// added for another hlc3-sourced indicator, this lane cannot forget to
    /// feed it hlc3.
    fn from_vector_ta(kind: F64InputKind) -> Self {
        match kind {
            F64InputKind::CloseSlice => DeviceInput::CloseFromOhlcv,
            F64InputKind::Hlc => DeviceInput::Ohlc,
            F64InputKind::Hlc3Slice => DeviceInput::Hlc3Slice,
            F64InputKind::Hlc3Volume => DeviceInput::Hlc3CloseVolume,
            F64InputKind::CloseVolume => DeviceInput::CloseVolume,
            F64InputKind::HighLow => DeviceInput::HighLow,
            F64InputKind::TimestampCloseVolume => DeviceInput::TimestampCloseVolume,
            F64InputKind::Hl2Slice => DeviceInput::Hl2Slice,
            F64InputKind::HighLowVolume => DeviceInput::HighLowVolume,
            F64InputKind::Hlcv => DeviceInput::Hlcv,
            F64InputKind::Ohlcv5 => DeviceInput::Ohlcv5,
            F64InputKind::OpenCloseVolume => DeviceInput::OpenCloseVolume,
            // closer 5
            F64InputKind::Hlcc4Slice => DeviceInput::Hlcc4Slice,
            F64InputKind::VolumeSlice => DeviceInput::VolumeSlice,
            // closer 6, round 3
            F64InputKind::Hlcc4Volume => DeviceInput::Hlcc4CloseVolume,
            // shard 3: `aso` reads OPEN as well as high/low/close. The resident
            // OHLCV ref already carries open, so no new upload shape is needed —
            // only the declaration that this kernel takes four price pointers,
            // which vector-ta's `resolve_inputs` turns into `F64Inputs::Ohlc4`.
            F64InputKind::Ohlc4 => DeviceInput::Ohlc,
        }
    }
}

/// One indicator that may be swept on the device.
#[derive(Debug, Clone, Copy)]
pub struct GpuSweepSpec {
    /// vector-ta indicator id — must match the id the CPU sweep uses, because
    /// the emitted column name is `{id}_{period}` either way.
    pub id: &'static str,
    pub input: DeviceInput,
}

/// The indicators of `hpc_ta`'s `MULTI_PERIOD_IDS` that have a real f64 CUDA
/// kernel.
///
/// THE RELATIONSHIP, NOT A COUNT. This table is the INTERSECTION of two sets
/// that are each maintained elsewhere:
///
/// ```text
///   GPU_SWEEP_SPECS  ==  MULTI_PERIOD_IDS  ∩  single-output  ∩  F64_KERNELS
/// ```
///
/// `vector_ta::indicators::dispatch::cuda_f64::F64_KERNELS` is far LARGER than
/// this table and grows independently of it — it registers every indicator in
/// vector-ta with an f64 kernel, most of which this crate never sweeps by
/// period. `hpc_ta::MULTI_PERIOD_IDS` is what this crate sweeps. So a row here
/// is a member of both, and neither number is written down: both directions of
/// the equation above are asserted in `tests` below, which is the only form
/// that cannot rot. A previous version of this comment claimed "exactly the ten
/// in F64_KERNELS"; the table held 12 and F64_KERNELS held 338.
///
/// This is also why `wilders` is NOT here even though vector-ta registered its
/// f64 kernel in the same round as `vwap`'s: `wilders` is not in
/// `MULTI_PERIOD_IDS`, so this crate never period-sweeps it, so a row here
/// would name an indicator no caller asks for. It reaches the CPU through the
/// base vocabulary (`all_indicators.rs`), which has no device lane at all.
///
/// ORDER IS LOAD-BEARING but only within this table: the caller re-assembles
/// columns in `MULTI_PERIOD_IDS` order regardless, so the pure-CPU column order
/// is preserved exactly whichever lane produced each column.
pub const GPU_SWEEP_SPECS: &[GpuSweepSpec] = &[
    GpuSweepSpec {
        id: "sma",
        input: DeviceInput::CloseFromOhlcv,
    },
    GpuSweepSpec {
        id: "ema",
        input: DeviceInput::CloseFromOhlcv,
    },
    GpuSweepSpec {
        id: "rsi",
        input: DeviceInput::CloseFromOhlcv,
    },
    GpuSweepSpec {
        id: "roc",
        input: DeviceInput::CloseFromOhlcv,
    },
    GpuSweepSpec {
        id: "mom",
        input: DeviceInput::CloseFromOhlcv,
    },
    GpuSweepSpec {
        id: "atr",
        input: DeviceInput::Ohlc,
    },
    GpuSweepSpec {
        id: "adx",
        input: DeviceInput::Ohlc,
    },
    GpuSweepSpec {
        id: "willr",
        input: DeviceInput::Ohlc,
    },
    GpuSweepSpec {
        id: "cci",
        input: DeviceInput::Hlc3Slice,
    },
    GpuSweepSpec {
        id: "mfi",
        input: DeviceInput::Hlc3CloseVolume,
    },
    // The final windowed single-output row in the historical period sweep.
    // `every_reachable_multi_period_id_with_an_f64_kernel_is_claimed` holds
    // the exact relationship to `hpc_ta::MULTI_PERIOD_IDS`; indicators without
    // a window (OBV/VWAP) deliberately remain outside this table. Their real
    // f64 kernels are reached through the complete primary-kernel registry,
    // not by manufacturing five duplicate `_7`/`_21`/... aliases.
    GpuSweepSpec {
        id: "tsi",
        input: DeviceInput::CloseFromOhlcv,
    },
];

/// Is this indicator id served by the device lane?
pub fn spec_for(id: &str) -> Option<&'static GpuSweepSpec> {
    GPU_SWEEP_SPECS.iter().find(|s| s.id == id)
}

/// The device-resident route declared by vector-ta's complete f64 registry.
///
/// Unlike [`GpuSweepSpec`], this is not a second manually maintained list. A
/// route exists exactly when vector-ta has registered a real f64 kernel, and
/// it carries the input and warm-up contracts from that same authoritative
/// row. This is the metadata boundary the complete `GpuOnly` feature-plan
/// builder consumes before it launches any work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct F64DeviceRoute {
    pub indicator_id: &'static str,
    pub output_id: &'static str,
    pub input: DeviceInput,
    pub first_valid: F64FirstValidRule,
    pub entry_point: &'static str,
    /// Whether CUDA may receive an inert anchor for a base CPU request with no
    /// window parameter.  This comes from the kernel authority, not from a
    /// downstream id list.
    pub period_invariant: bool,
}

/// Resolve one indicator through the complete f64 registry without consulting
/// the historical multi-period sweep table.
pub fn f64_primary_device_route_for(indicator_id: &str) -> Option<F64DeviceRoute> {
    let spec = f64_kernel_for(indicator_id)?;
    Some(F64DeviceRoute {
        indicator_id: spec.indicator_id,
        output_id: spec.primary_output_id()?,
        input: DeviceInput::from_vector_ta(spec.input),
        first_valid: spec.first_valid,
        entry_point: spec.kernel.entry_point(),
        period_invariant: spec.kernel.is_period_invariant(),
    })
}

/// Resolve the exact feature-output identity produced by an indicator's
/// registered f64 primary kernel.
///
/// The authoritative mapping lives with vector-ta's f64 kernel specification;
/// downstream code never infers a name from the output shape or from a second
/// handwritten table.
pub fn f64_primary_output_id_for(indicator_id: &str) -> Option<&'static str> {
    f64_kernel_for(indicator_id)?.primary_output_id()
}

/// Why one current production feature output cannot yet enter `GpuOnly`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuOnlyOutputGapReason {
    /// No registered f64 value kernel exists for this production id.
    MissingF64Kernel,
    /// The primary f64 kernel exists, but this additional named output has no
    /// resident f64 route of its own.
    MissingNamedOutputRoute,
    /// The output is a discrete matrix and needs its own typed resident route;
    /// pretending it is an f64 value column would corrupt the schema.
    MissingDiscreteMatrixRoute,
}

impl std::fmt::Display for GpuOnlyOutputGapReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::MissingF64Kernel => "missing_f64_kernel",
            Self::MissingNamedOutputRoute => "missing_named_output_route",
            Self::MissingDiscreteMatrixRoute => "missing_discrete_matrix_route",
        })
    }
}

/// One exact output-level blocker found by the no-work `GpuOnly` preflight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuOnlyOutputGap {
    pub indicator_id: &'static str,
    pub output_id: &'static str,
    pub reason: GpuOnlyOutputGapReason,
}

fn production_output_ids(indicator_id: &'static str) -> Vec<&'static str> {
    if let Some(info) = get_indicator(indicator_id) {
        return info.outputs.iter().map(|output| output.id).collect();
    }
    if let Some((_, outputs)) = crate::core::indicator_ledger::UNREGISTERED_MULTI_OUTPUTS
        .iter()
        .find(|(id, _)| *id == indicator_id)
    {
        return outputs.to_vec();
    }
    // This is the exact by-name dispatch contract for unregistered
    // single-output rows in vector-ta's cpu_batch dispatcher.
    vec!["value"]
}

/// Inventory every output in the current production classic-TA vocabulary
/// that does not have its own exact resident f64/discrete route.
///
/// This function performs registry/table inspection only. It opens no CUDA
/// context, allocates no host/device buffer, and launches no work, so strict
/// admission can reject an incomplete graph atomically.
pub fn gpu_only_classic_ta_output_gaps() -> Vec<GpuOnlyOutputGap> {
    let mut gaps = Vec::new();
    for &indicator_id in crate::core::all_indicators::ALL_INDICATORS {
        for output_id in production_output_ids(indicator_id) {
            if indicator_id == "pattern_recognition" {
                gaps.push(GpuOnlyOutputGap {
                    indicator_id,
                    output_id,
                    reason: GpuOnlyOutputGapReason::MissingDiscreteMatrixRoute,
                });
                continue;
            }

            let Some(_) = f64_primary_output_id_for(indicator_id) else {
                gaps.push(GpuOnlyOutputGap {
                    indicator_id,
                    output_id,
                    reason: GpuOnlyOutputGapReason::MissingF64Kernel,
                });
                continue;
            };
            if !has_f64_resident_output_route(indicator_id, output_id) {
                gaps.push(GpuOnlyOutputGap {
                    indicator_id,
                    output_id,
                    reason: GpuOnlyOutputGapReason::MissingNamedOutputRoute,
                });
            }
        }
    }
    gaps
}

/// Number of f64 primary rows registered by the custom vector-ta fork.
pub fn registered_f64_primary_route_count() -> usize {
    vector_ta::indicators::dispatch::cuda_f64::F64_KERNELS.len()
}

/// The compute capability of CUDA device `ordinal`, spelled `sm_XY`.
///
/// Read from the driver via `cust` rather than inferred from a build-time
/// string, because the whole point of the preflight is to compare what we
/// COMPILED against what is actually PRESENT.
fn device_arch(ordinal: u32) -> Result<String> {
    use cust::device::{Device, DeviceAttribute};
    // Idempotent: `cuda_available()` has already initialised the driver, and
    // `cust::init` is safe to call again.
    cust::init(cust::CudaFlags::empty()).context("cust::init failed")?;
    let dev = Device::get_device(ordinal).with_context(|| format!("no CUDA device {ordinal}"))?;
    let major = dev
        .get_attribute(DeviceAttribute::ComputeCapabilityMajor)
        .context("query ComputeCapabilityMajor")?;
    let minor = dev
        .get_attribute(DeviceAttribute::ComputeCapabilityMinor)
        .context("query ComputeCapabilityMinor")?;
    Ok(format!("sm_{major}{minor}"))
}

fn device_name(ordinal: u32) -> String {
    use cust::device::Device;
    Device::get_device(ordinal)
        .and_then(|d| d.name())
        .unwrap_or_else(|_| "<unknown device>".to_string())
}

/// First index whose value is not NaN — the CPU's
/// `data.iter().position(|x| !x.is_nan())`.
fn first_valid_1(series: &[f64]) -> Option<usize> {
    series.iter().position(|x| !x.is_nan())
}

/// First index where none of the three series is NaN — the CPU's
/// `first_valid_hlc` / `first_valid_triple_checked`.
fn first_valid_3(a: &[f64], b: &[f64], c: &[f64]) -> Option<usize> {
    (0..c.len()).find(|&i| !a[i].is_nan() && !b[i].is_nan() && !c[i].is_nan())
}

/// First index where both series are non-NaN — the CPU's `mfi_prepare`.
fn first_valid_2(a: &[f64], b: &[f64]) -> Option<usize> {
    (0..a.len()).find(|&i| !a[i].is_nan() && !b[i].is_nan())
}

/// First index where the price selected by `input` and volume are both finite.
/// EVWMA's f64 row uses `CloseVolume`; the retained HLCC4 device shape must
/// still scan its own price series when another row declares it.
fn first_valid_price_volume_finite_for_input(
    input: DeviceInput,
    close: &[f64],
    hlcc4: &[f64],
    volume: &[f64],
) -> Option<usize> {
    let price = match input {
        DeviceInput::CloseVolume => close,
        DeviceInput::Hlcc4CloseVolume => hlcc4,
        _ => return None,
    };
    price
        .iter()
        .zip(volume)
        .position(|(&price, &volume)| price.is_finite() && volume.is_finite())
}

#[inline]
fn longest_true_run(values: impl Iterator<Item = bool>) -> usize {
    let mut longest = 0usize;
    let mut current = 0usize;
    for valid in values {
        if valid {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 0;
        }
    }
    longest
}

/// A CUDA indicator lane bound to one frame's OHLCV, with the series resident
/// on the device in f64 for the whole life of the engine.
///
/// Construction is where the lane is PROVEN. `new` does not merely check that
/// a device exists — it uploads, then loads the f64 module and launches a real
/// kernel. The shared availability probe already uses the exact native
/// registry; this additional launch proves the real f64 indicator route.
pub struct GpuIndicatorEngine {
    runtime: CudaRuntime,
    /// The f64 kernel module. ONE load for the whole frame — the f32
    /// dispatcher constructs a fresh wrapper and loads its verified cubin per
    /// call, roughly fifty times per frame.
    f64_engine: CudaF64Indicators,
    /// Resident open/high/low/close/volume in f64, uploaded once.
    ohlcv: CudaDeviceOhlcvF64,
    /// Resident hlc3 = (h+l+c)/3 in f64, uploaded once. Required so `cci` and
    /// `mfi` compute the same indicator the CPU computes rather than a
    /// close-priced impostor.
    hlc3: CudaDeviceVectorF64,
    /// Resident hl2 = (h + l) / 2 in f64, uploaded once. Required so `kurtosis`
    /// and `alligator` compute the same indicator the CPU computes -- their
    /// batch arms call `extract_slice_input(..., "hl2")`, and close would be a
    /// different series, not a rounder one.
    hl2: CudaDeviceVectorF64,
    /// Resident hlcc4 = (h + l + 2c) / 4 in f64, uploaded once. Required by
    /// the `velocity` family, whose CPU batch arms resolve their source with
    /// `source_type(candles, "hlcc4")`; close would be a different series,
    /// not a rounder one. Formed the way `Candles::compute_hlcc4` does.
    hlcc4: CudaDeviceVectorF64,
    /// Resident bar timestamps in `i64`, uploaded once. Required by `vwap`,
    /// whose anchor is a calendar bucket rather than a rolling window, so the
    /// timestamps are an INPUT. Built with `hpc_ta`'s own rule for a frame with
    /// no timestamps — `vec![0i64; n]` — so the device and the CPU agree about
    /// where sessions begin even in that degenerate case.
    timestamps: CudaDeviceVectorI64,
    /// Per-RULE first-valid index, computed on the host with the same rule the
    /// CPU `*_prepare` uses. Precomputed once because each is a property of the
    /// frame, not of the period.
    ///
    /// There are SEVEN of these and only six input shapes, because the
    /// high/low/close bucket carries three different rules — see
    /// [`vector_ta::indicators::dispatch::F64FirstValidRule`]. Deriving one
    /// index per SHAPE is what this used to do and it was wrong for `adx`,
    /// `natr` and `adxr`.
    first_valid_close: usize,
    first_valid_hlc: usize,
    /// First bar where high, low and close are simultaneously finite. The
    /// Ichimoku Oscillator rejects infinities at admission; the older
    /// `first_valid_hlc` is a non-NaN scan and cannot be substituted.
    first_valid_hlc_finite: usize,
    /// `fh.max(fl).max(fc)` — the max of three INDEPENDENT scans, which is a
    /// different index from `first_valid_hlc` whenever high, low and close
    /// start at different bars. `adx.rs:201-219`, `natr.rs:226-235`.
    first_valid_hlc_max_of_firsts: usize,
    first_valid_hlc3: usize,
    first_valid_hlc3_volume: usize,
    first_valid_close_volume: usize,
    /// Close AND volume both `is_finite` at the same index -- the f64 EVWMA
    /// production row's close-source admission rule.
    first_valid_close_volume_finite: usize,
    first_valid_high_low: usize,
    // ------------------------------------------------------------ shard 1
    /// hl2 non-NaN. Not derivable from `first_valid_high_low`: hl2 is formed
    /// before the scan, so a bar where exactly one of high/low is NaN is NaN in
    /// hl2 too -- the same index in this case, but stated separately because
    /// "the same today" is not a contract.
    first_valid_hl2: usize,
    /// First finite hl2 value. Ehlers Adaptive Cyber Cycle rejects infinity
    /// while several older hl2 consumers accept any non-NaN value, so this is
    /// intentionally not an alias of `first_valid_hl2`.
    first_valid_hl2_finite: usize,
    // ------------------------------------------------------------ closer 5
    /// hlcc4 non-NaN. Stated separately from `first_valid_close` for the
    /// same reason `first_valid_hl2` is: hlcc4 is formed BEFORE the scan, so
    /// a bar where any one of high/low/close is NaN is NaN in hlcc4 too.
    first_valid_hlcc4: usize,
    /// Volume non-NaN -- `vosc.rs:361` scans the volume series alone.
    first_valid_volume: usize,
    /// AVSL's `first_valid_max3`: the max of independent first-non-NaN
    /// scans over close, low and volume. It is not the first bar where all
    /// three happen to be valid simultaneously.
    first_valid_avsl: usize,
    // ------------------------------------------------------ closer 6, round 3
    /// hlcc4 AND volume both `is_finite` at the same index.
    ///
    /// Stated separately from `first_valid_hlcc4` and from a non-NaN pair scan
    /// for the reason every narrow field here exists: `is_finite` REJECTS an
    /// infinity that `!is_nan` accepts, and this index sets both the NaN prefix
    /// and the bar a future strict HLCC4/volume row would seed from.
    first_valid_hlcc4_volume_finite: usize,
    /// Volume `is_finite` -- `volume_zone_oscillator.rs:271`. NOT the same
    /// as `first_valid_volume`: that one is a `!is_nan` scan and accepts an
    /// infinity this one rejects.
    first_valid_volume_finite: usize,
    /// The first index `i >= 1` at which high, low and close are non-NaN at
    /// BOTH `i - 1` and `i` -- `ultosc.rs:391-401`, whose true range reads
    /// `close[i-1]`. At least one bar later than `first_valid_hlc`.
    first_valid_hlc_consecutive_pair: usize,
    /// high, low and volume simultaneously non-NaN -- `emv.rs:219`. Close is
    /// deliberately absent from this scan.
    first_valid_high_low_volume: usize,
    /// high, low, close and volume simultaneously non-NaN -- `kvo.rs:292-297`.
    first_valid_hlcv: usize,
    // ------------------------------------------------------------ shard 6
    // Four more, because four of shard 6's indicators read their first bar by
    // a rule none of the seven above use. Each is a property of the frame, so
    // each is computed once here rather than per period.
    /// high and low both `is_finite` at the same index -- `aroonosc.rs:16-20`.
    /// Different from `first_valid_high_low` on any bar carrying an infinity.
    first_valid_high_low_finite: usize,
    /// high and low both finite AND `> 0` at the same index --
    /// `parkinson_volatility.rs:214-223`. The indicator takes `ln(high/low)`.
    first_valid_high_low_finite_positive: usize,
    /// `fh.max(fl)` over INDEPENDENT scans -- `donchian.rs:183-188`. The
    /// two-series twin of `first_valid_hlc_max_of_firsts`.
    first_valid_high_low_max_of_firsts: usize,
    /// The first index that can form a RETURN: `close[i-1]` and `close[i]`
    /// finite and `close[i-1] != 0.0` -- `historical_volatility.rs:334-355`.
    first_valid_close_return_pair: usize,
    // ------------------------------------------------------------ closer 1
    /// open, high, low and close ALL `is_finite` at the same index --
    /// `accumulation_swing_index.rs:245`, `daily_factor.rs:258`. Open is an
    /// INPUT to both, so `first_valid_hlc` would seed `prev_open` from a bar
    /// the CPU skips.
    first_valid_ohlc4_finite: usize,
    /// The same four series scanned with `!is_nan` -- `bop.rs:209-211`. A
    /// SEPARATE field from the one above because `bop` accepts an infinite bar
    /// that `accumulation_swing_index` rejects, so on a frame carrying an
    /// infinity the two indicators start at different bars.
    first_valid_ohlc4_non_nan: usize,
    /// OPEN and CLOSE both `is_finite` at the same index --
    /// `andean_oscillator.rs:244`. Distinct from `first_valid_open_close`,
    /// which is the `!is_nan` scan `qstick` uses.
    first_valid_open_close_finite: usize,
    // ------------------------------------------------------------ shard 4
    /// close alone, `is_finite` -- `dvdiqqe.rs:385`. NOT `first_valid_close`:
    /// that one is a `!is_nan` scan and accepts an infinity this one rejects.
    first_valid_close_finite: usize,
    /// Number of finite close values. Adaptive Bandpass Trigger Oscillator's
    /// public CPU contract admits the frame only when this count is at least
    /// twelve; carrying the once-computed metadata avoids any device→host
    /// scan or a late, partially-executed GpuOnly failure.
    finite_close_count: usize,
    /// Longest uninterrupted run of finite close values. Adaptive Bounds RSI
    /// resets its Wilder seed at every non-finite value, so total finite-count
    /// admission can otherwise accept a row that can only emit NaN.
    max_consecutive_finite_close: usize,
    /// Longest uninterrupted run of finite, strictly-positive closes. Dual
    /// Ulcer Index validates `2 * period - 1` such bars before computing and
    /// resets both extrema and square-sum state at every other value.
    max_consecutive_valid_dual_ulcer_close: usize,
    /// Number of bars whose high, low and close are simultaneously finite.
    /// Market Structure Confluence's public CPU admission rule counts these
    /// bars (it does not require one consecutive run), so the resident route
    /// carries the exact same metadata into pre-allocation validation.
    finite_hlc_count: usize,
    /// Bit `mode` is set only when at least one current-period Pivot output can
    /// be formed from that mode's exact previous/current inputs. This lets
    /// `GpuOnly` reject an impossible formula row before allocating outputs.
    pivot_valid_mode_mask: u8,
    /// Longest uninterrupted run with finite high, low and close. Fibonacci
    /// Entry Bands validates this before launch for sources that do not read
    /// open, so an invalid sweep fails before allocating any device output.
    max_consecutive_finite_hlc: usize,
    /// Number of bars from the first ASTC-valid HLC bar through frame end.
    /// The CPU admission rule uses this suffix length (not a consecutive-run
    /// count) and additionally requires `high >= low`; later invalid bars
    /// reset the recurrence without changing admission.
    adaptive_schaff_valid_suffix_len: usize,
    /// Longest uninterrupted run with all four OHLC inputs finite. Fibonacci
    /// Entry Bands uses this stricter value for `open` and `ohlc4` sources.
    max_consecutive_finite_ohlc: usize,
    /// Longest uninterrupted run with all five OHLCV inputs finite. Trend
    /// Flow Trail resets every internal HMA/EMA/MFI state when any one input
    /// is non-finite, so validating a looser price-only run would permit a
    /// late partial launch that can never produce a valid row.
    max_consecutive_finite_ohlcv: usize,
    /// open, close and volume simultaneously non-NaN -- the common rule for
    /// this shape. Present so `AllInputsNonNan` has an answer for it even
    /// though `dvdiqqe`, the only indicator using the shape today, declares
    /// `CloseFinite` instead.
    first_valid_open_close_volume: usize,
    // ----------------------------------------------------------- closer 4
    /// open and close simultaneously non-NaN -- `qstick.rs:235-243`.
    ///
    /// qstick is declared `Ohlc4` so the kernel receives the four price
    /// pointers the resident upload already carries, but it reads only open
    /// and close. Under `AllInputsNonNan` that shape would resolve to
    /// `first_valid_hlc`, which names a LATER bar on any frame where high or
    /// low starts after open and close -- and `first_valid` sets both the NaN
    /// prefix and the seed window, so that is a different set of windows, not
    /// a rounding difference.
    first_valid_open_close: usize,
    /// Bar count. Every device result is checked against this before any
    /// indexing.
    n: usize,
    device_ordinal: u32,
    device_arch: String,
    device_name: String,
    /// Number of host→device uploads performed. Should always be 9 (o,h,l,c,v,
    /// hlc3, hl2, hlcc4 and timestamps) for the lifetime of the engine — if
    /// this grows, residency broke.
    uploads: u64,
}

impl GpuIndicatorEngine {
    /// Upload `ohlcv` in f64 and prove the lane end to end.
    ///
    /// FAIL LOUD, ALWAYS. There is no path through this function that returns
    /// a working-looking engine on a card that cannot run the kernels.
    pub fn new(ohlcv: &Ohlcv, device_ordinal: u32) -> Result<Self> {
        let n = ohlcv.len();
        if n == 0 {
            bail!("GpuIndicatorEngine: empty frame");
        }
        // Fail loud on a ragged frame rather than panicking on an index
        // further down, or — worse — uploading buffers of different lengths
        // and letting the kernel read whatever follows.
        if ohlcv.open.len() != n || ohlcv.high.len() != n || ohlcv.low.len() != n {
            bail!(
                "GpuIndicatorEngine: ragged OHLCV — open={} high={} low={} close={}",
                ohlcv.open.len(),
                ohlcv.high.len(),
                ohlcv.low.len(),
                n
            );
        }
        let source_volume = ohlcv.volume.as_ref().context(
            "GpuIndicatorEngine: volume is required by the complete resident feature plan; \
             missing volume is not fabricated as numeric zero",
        )?;
        if source_volume.len() != n {
            bail!(
                "GpuIndicatorEngine: volume has {} entries for {n} bars",
                source_volume.len()
            );
        }

        if !cuda_available() {
            bail!(
                "GpuIndicatorEngine: vector_ta::cuda::cuda_available() == false — no usable exact \
                 native CUDA lane. This binary carries verified cubins for \
                 {VECTOR_TA_NATIVE_ARCHS:?} (source={VECTOR_TA_ARCH_SOURCE})."
            );
        }

        let device_arch = device_arch(device_ordinal)?;
        let device_name = device_name(device_ordinal);

        // NO NARROWING. The host series are f64, the device buffers are f64,
        // the kernels are f64. This is the change that makes the lane a parity
        // claim instead of a measured-divergence one.
        let volume = source_volume.clone();
        // hlc3 computed exactly as `Candles::compute_hlc3` does
        // (`data_loader.rs:169 (h + l + c) / 3.0`) — same expression, same
        // order, same precision, so the device and CPU sources are the same
        // numbers rather than merely the same formula.
        let hlc3: Vec<f64> = (0..n)
            .map(|i| (ohlcv.high[i] + ohlcv.low[i] + ohlcv.close[i]) / 3.0)
            .collect();

        // Where each series actually begins, by the CPU's own rules. An
        // entirely-NaN series here is a data fault, not a fallback condition.
        let first_valid_close =
            first_valid_1(&ohlcv.close).context("GpuIndicatorEngine: close is entirely NaN")?;
        let first_valid_hlc = first_valid_3(&ohlcv.high, &ohlcv.low, &ohlcv.close)
            .context("GpuIndicatorEngine: no bar has all of high, low and close")?;
        let first_valid_hlc_finite = (0..n)
            .find(|&i| {
                ohlcv.high[i].is_finite() && ohlcv.low[i].is_finite() && ohlcv.close[i].is_finite()
            })
            .context("GpuIndicatorEngine: no bar has finite high, low and close")?;
        // NOT the same number. `adx.rs::first_valid_triple_checked` and
        // `natr.rs:226-235` scan the three series INDEPENDENTLY and take the
        // max of the three answers, which can name a bar at which one of them
        // is still NaN. Computing it here rather than reusing `first_valid_hlc`
        // is the difference between the device reproducing the CPU's series and
        // shifting it.
        let first_valid_hlc_max_of_firsts = {
            let fh =
                first_valid_1(&ohlcv.high).context("GpuIndicatorEngine: high is entirely NaN")?;
            let fl =
                first_valid_1(&ohlcv.low).context("GpuIndicatorEngine: low is entirely NaN")?;
            fh.max(fl).max(first_valid_close)
        };
        // hl2 formed the way `Candles::compute_hl2` does -- `(h + l) / 2.0`,
        // one expression, one rounding -- so the device source and the CPU
        // source are the same numbers and not merely the same formula.
        let hl2: Vec<f64> = (0..n)
            .map(|i| (ohlcv.high[i] + ohlcv.low[i]) / 2.0)
            .collect();
        let first_valid_hl2 =
            first_valid_1(&hl2).context("GpuIndicatorEngine: hl2 is entirely NaN")?;
        let first_valid_hl2_finite = hl2
            .iter()
            .position(|value| value.is_finite())
            .context("GpuIndicatorEngine: hl2 has no finite value")?;
        // closer 5: hlcc4 formed the way `Candles::compute_hlcc4` does --
        // `(h + l + 2c) / 4`, one expression, one rounding -- so the device
        // source and the CPU source are the same NUMBERS and not merely the
        // same formula.
        let hlcc4: Vec<f64> = (0..n)
            .map(|i| (ohlcv.high[i] + ohlcv.low[i] + 2.0 * ohlcv.close[i]) / 4.0)
            .collect();
        let first_valid_hlcc4 =
            first_valid_1(&hlcc4).context("GpuIndicatorEngine: hlcc4 is entirely NaN")?;
        // Price/volume finite admission is source-sensitive. EVWMA declares
        // close; the retained HLCC4 device shape keeps its own independent
        // scan for any row that explicitly declares it.
        let first_valid_close_volume_finite = first_valid_price_volume_finite_for_input(
            DeviceInput::CloseVolume,
            &ohlcv.close,
            &hlcc4,
            &volume,
        )
        .context("GpuIndicatorEngine: no bar has both a finite close and a finite volume")?;
        let first_valid_hlcc4_volume_finite = first_valid_price_volume_finite_for_input(
            DeviceInput::Hlcc4CloseVolume,
            &ohlcv.close,
            &hlcc4,
            &volume,
        )
        .context("GpuIndicatorEngine: no bar has both a finite hlcc4 and a finite volume")?;
        let first_valid_volume =
            first_valid_1(&volume).context("GpuIndicatorEngine: volume is entirely NaN")?;
        let first_valid_avsl = first_valid_close
            .max(first_valid_1(&ohlcv.low).context("GpuIndicatorEngine: low is entirely NaN")?)
            .max(first_valid_volume);
        let first_valid_volume_finite = (0..n)
            .find(|&i| volume[i].is_finite())
            .context("GpuIndicatorEngine: no bar has a finite volume")?;
        // ultosc.rs:391-401 -- the scan starts at 1 and requires BOTH bars.
        let first_valid_hlc_consecutive_pair = (1..n)
            .find(|&i| {
                !ohlcv.high[i - 1].is_nan()
                    && !ohlcv.low[i - 1].is_nan()
                    && !ohlcv.close[i - 1].is_nan()
                    && !ohlcv.high[i].is_nan()
                    && !ohlcv.low[i].is_nan()
                    && !ohlcv.close[i].is_nan()
            })
            .context(
                "GpuIndicatorEngine: no consecutive pair of bars has all of \
                 high, low and close",
            )?;
        let first_valid_high_low_volume = (0..n)
            .find(|&i| !ohlcv.high[i].is_nan() && !ohlcv.low[i].is_nan() && !volume[i].is_nan())
            .context("GpuIndicatorEngine: no bar has all of high, low and volume")?;
        let first_valid_hlcv = (0..n)
            .find(|&i| {
                !ohlcv.high[i].is_nan()
                    && !ohlcv.low[i].is_nan()
                    && !ohlcv.close[i].is_nan()
                    && !volume[i].is_nan()
            })
            .context("GpuIndicatorEngine: no bar has all of high, low, close and volume")?;
        let first_valid_hlc3 =
            first_valid_1(&hlc3).context("GpuIndicatorEngine: hlc3 is entirely NaN")?;
        let first_valid_hlc3_volume = first_valid_2(&hlc3, &volume)
            .context("GpuIndicatorEngine: no bar has both hlc3 and volume")?;
        // `obv`/`vwma`/`efi` pair CLOSE with volume, not hlc3. Same shape, a
        // different first-valid index whenever high or low is the late series.
        let first_valid_close_volume = first_valid_2(&ohlcv.close, &volume)
            .context("GpuIndicatorEngine: no bar has both close and volume")?;
        // `medprice`/`midprice` never read close, and their CPU `*_prepare`
        // scans high and low only. Using the hlc index here would push their
        // warmup out by however long close lags.
        let first_valid_high_low = first_valid_2(&ohlcv.high, &ohlcv.low)
            .context("GpuIndicatorEngine: no bar has both high and low")?;

        // ------------------------------------------------------------ shard 6
        // Four rules read out of the CPU `*_prepare` of the indicators that
        // use them. None is a variation on "non-NaN": each one names a
        // different bar on real data, and `first_valid` sets both the NaN
        // prefix and the seed window, so the wrong one shifts the series.

        // `aroonosc.rs:16-20` -- `h.is_finite() && l.is_finite()`.
        let first_valid_high_low_finite = (0..n)
            .find(|&i| ohlcv.high[i].is_finite() && ohlcv.low[i].is_finite())
            .context("GpuIndicatorEngine: no bar has both high and low finite")?;

        // `parkinson_volatility.rs:214-216` -- finite AND strictly positive.
        let first_valid_high_low_finite_positive = (0..n)
            .find(|&i| {
                ohlcv.high[i].is_finite()
                    && ohlcv.low[i].is_finite()
                    && ohlcv.high[i] > 0.0
                    && ohlcv.low[i] > 0.0
            })
            .context(
                "GpuIndicatorEngine: no bar has high and low both finite and strictly positive",
            )?;

        // `donchian.rs:183-188` -- the MAX of two INDEPENDENT scans, not the
        // first index at which both are non-NaN.
        let first_valid_high_low_max_of_firsts = {
            let fh =
                first_valid_1(&ohlcv.high).context("GpuIndicatorEngine: high is entirely NaN")?;
            let fl =
                first_valid_1(&ohlcv.low).context("GpuIndicatorEngine: low is entirely NaN")?;
            fh.max(fl)
        };

        // `historical_volatility.rs:334-355` -- the first index at which a
        // percentage return can be FORMED. Starts at 1, and rejects a zero
        // previous price that a non-NaN scan would accept and then divide by.
        // dvdiqqe.rs:385 -- `c.iter().position(|x| x.is_finite())`.
        let first_valid_close_finite = (0..n)
            .find(|&i| ohlcv.close[i].is_finite())
            .context("GpuIndicatorEngine: no close value is finite")?;
        let finite_close_count = ohlcv.close.iter().filter(|value| value.is_finite()).count();
        let max_consecutive_finite_close =
            longest_true_run(ohlcv.close.iter().map(|value| value.is_finite()));
        let max_consecutive_valid_dual_ulcer_close = longest_true_run(
            ohlcv
                .close
                .iter()
                .map(|value| value.is_finite() && *value > 0.0),
        );
        let finite_hlc_count = (0..n)
            .filter(|&i| {
                ohlcv.high[i].is_finite() && ohlcv.low[i].is_finite() && ohlcv.close[i].is_finite()
            })
            .count();
        let mut pivot_valid_mode_mask = 0u8;
        for index in 1..n {
            let previous = index - 1;
            let previous_hlc = ohlcv.high[previous].is_finite()
                && ohlcv.low[previous].is_finite()
                && ohlcv.close[previous].is_finite();
            if previous_hlc {
                pivot_valid_mode_mask |= (1 << 0) | (1 << 1) | (1 << 3);
                if ohlcv.open[previous].is_finite() {
                    pivot_valid_mode_mask |= 1 << 2;
                }
            }
            if ohlcv.high[previous].is_finite()
                && ohlcv.low[previous].is_finite()
                && ohlcv.open[index].is_finite()
            {
                pivot_valid_mode_mask |= 1 << 4;
            }
        }
        let max_consecutive_finite_hlc = longest_true_run((0..n).map(|i| {
            ohlcv.high[i].is_finite() && ohlcv.low[i].is_finite() && ohlcv.close[i].is_finite()
        }));
        let adaptive_schaff_valid_suffix_len = (0..n)
            .find(|&i| {
                ohlcv.high[i].is_finite()
                    && ohlcv.low[i].is_finite()
                    && ohlcv.close[i].is_finite()
                    && ohlcv.high[i] >= ohlcv.low[i]
            })
            .map_or(0, |first_valid| n - first_valid);
        let max_consecutive_finite_ohlc = longest_true_run((0..n).map(|i| {
            ohlcv.open[i].is_finite()
                && ohlcv.high[i].is_finite()
                && ohlcv.low[i].is_finite()
                && ohlcv.close[i].is_finite()
        }));
        let max_consecutive_finite_ohlcv = longest_true_run((0..n).map(|i| {
            ohlcv.open[i].is_finite()
                && ohlcv.high[i].is_finite()
                && ohlcv.low[i].is_finite()
                && ohlcv.close[i].is_finite()
                && volume[i].is_finite()
        }));
        let first_valid_open_close_volume = (0..n)
            .find(|&i| !ohlcv.open[i].is_nan() && !ohlcv.close[i].is_nan() && !volume[i].is_nan())
            .context("GpuIndicatorEngine: no bar has all of open, close and volume")?;
        // qstick.rs:235-243 -- open and close, neither NaN, at the same index.
        let first_valid_open_close = (0..n)
            .find(|&i| !ohlcv.open[i].is_nan() && !ohlcv.close[i].is_nan())
            .context("GpuIndicatorEngine: no bar has both open and close")?;
        // ------------------------------------------------------- closer 1
        // accumulation_swing_index.rs:245 / daily_factor.rs:258 -- all four
        // `is_finite`, open included.
        let first_valid_ohlc4_finite = (0..n)
            .find(|&i| {
                ohlcv.open[i].is_finite()
                    && ohlcv.high[i].is_finite()
                    && ohlcv.low[i].is_finite()
                    && ohlcv.close[i].is_finite()
            })
            .context(
                "GpuIndicatorEngine: no bar has all four of open, high, low and close finite",
            )?;
        // bop.rs:209-211 -- the SAME four series, but `!is_nan`, which accepts
        // an infinity the scan above rejects.
        let first_valid_ohlc4_non_nan = (0..n)
            .find(|&i| {
                !ohlcv.open[i].is_nan()
                    && !ohlcv.high[i].is_nan()
                    && !ohlcv.low[i].is_nan()
                    && !ohlcv.close[i].is_nan()
            })
            .context(
                "GpuIndicatorEngine: no bar has all four of open, high, low and close non-NaN",
            )?;
        // andean_oscillator.rs:244 -- open and close, both finite.
        let first_valid_open_close_finite = (0..n)
            .find(|&i| ohlcv.open[i].is_finite() && ohlcv.close[i].is_finite())
            .context("GpuIndicatorEngine: no bar has both open and close finite")?;
        let first_valid_close_return_pair = (1..n)
            .find(|&i| {
                ohlcv.close[i - 1].is_finite()
                    && ohlcv.close[i].is_finite()
                    && ohlcv.close[i - 1] != 0.0
            })
            .context("GpuIndicatorEngine: no consecutive pair of close values forms a return")?;

        // `hpc_ta::compute_classic_ta_columns_with_policy` builds its `Candles`
        // with exactly this fallback (`hpc_ta.rs:99`). Mirroring it rather than
        // erroring keeps the device and the CPU on the same timestamps for a
        // frame that carries none — the alternative is a lane that computes a
        // different vwap from the reference it is checked against.
        let timestamps: Vec<i64> = match &ohlcv.timestamp {
            Some(t) if t.len() == n => t.clone(),
            Some(t) => bail!(
                "GpuIndicatorEngine: timestamp has {} entries for {n} bars",
                t.len()
            ),
            None => vec![0i64; n],
        };

        let runtime = CudaRuntime::new(device_ordinal as usize).with_context(|| {
            format!("CudaRuntime::new({device_ordinal}) failed on {device_name} ({device_arch})")
        })?;

        // ONE upload of the whole frame, in f64. Every indicator below reads
        // these buffers in place.
        //
        // `source: None` so the OHLCV ref's price accessor resolves to
        // `close`, matching the CPU default source for the price-series
        // indicators in the table.
        let device_ohlcv = runtime
            .upload_ohlcv_f64(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                &volume,
                None,
            )
            .with_context(|| format!("upload_ohlcv_f64({n} bars) failed on {device_name}"))?;
        let device_hlc3 = runtime
            .upload_f64(&hlc3)
            .with_context(|| format!("upload_f64(hlc3, {n} bars) failed on {device_name}"))?;
        let device_hl2 = runtime
            .upload_f64(&hl2)
            .with_context(|| format!("upload_f64(hl2, {n} bars) failed on {device_name}"))?;
        let device_hlcc4 = runtime
            .upload_f64(&hlcc4)
            .with_context(|| format!("upload_f64(hlcc4, {n} bars) failed on {device_name}"))?;
        let device_timestamps = runtime
            .upload_i64(&timestamps)
            .with_context(|| format!("upload_i64(timestamps, {n} bars) failed on {device_name}"))?;

        let f64_engine =
            CudaF64Indicators::from_session(runtime.session_arc()).with_context(|| {
                format!(
                    "CudaF64Indicators::from_session({device_ordinal}) failed on {device_name} \
                 ({device_arch}) — the f64 kernel module could not be loaded. vector-ta's module \
                 loader prints the device arch and the compiled arch set; the same text is \
                 available from vector_ta::cuda::module_loader::last_module_load_failure()."
                )
            })?;

        let engine = Self {
            runtime,
            f64_engine,
            ohlcv: device_ohlcv,
            hlc3: device_hlc3,
            hl2: device_hl2,
            hlcc4: device_hlcc4,
            timestamps: device_timestamps,
            first_valid_close,
            first_valid_hlc,
            first_valid_hlc_finite,
            first_valid_hlc_max_of_firsts,
            first_valid_hlc3,
            first_valid_hlc3_volume,
            first_valid_close_volume,
            first_valid_close_volume_finite,
            first_valid_high_low,
            first_valid_hl2,
            first_valid_hl2_finite,
            first_valid_hlcc4,
            first_valid_hlcc4_volume_finite,
            first_valid_volume,
            first_valid_avsl,
            first_valid_volume_finite,
            first_valid_hlc_consecutive_pair,
            first_valid_high_low_volume,
            first_valid_hlcv,
            first_valid_high_low_finite,
            first_valid_high_low_finite_positive,
            first_valid_high_low_max_of_firsts,
            first_valid_close_return_pair,
            first_valid_ohlc4_finite,
            first_valid_ohlc4_non_nan,
            first_valid_open_close_finite,
            first_valid_close_finite,
            finite_close_count,
            max_consecutive_finite_close,
            max_consecutive_valid_dual_ulcer_close,
            finite_hlc_count,
            pivot_valid_mode_mask,
            max_consecutive_finite_hlc,
            adaptive_schaff_valid_suffix_len,
            max_consecutive_finite_ohlc,
            max_consecutive_finite_ohlcv,
            first_valid_open_close_volume,
            first_valid_open_close,
            n,
            device_ordinal,
            device_arch,
            device_name,
            // o, h, l, c, v, hlc3, hl2, hlcc4, timestamps. If this number
            // ever grows at runtime, residency broke.
            uploads: 9,
        };

        engine.prove_module_loads()?;
        Ok(engine)
    }

    /// Load a real module and launch a real kernel.
    ///
    /// `cuda_available()` already loads and launches the exact-architecture
    /// native probe. This second gate proves a real indicator module and its
    /// resident f64 route, rather than only the shared loader.
    fn prove_module_loads(&self) -> Result<()> {
        // `sma` with period 1 over the resident close series: the cheapest
        // real kernel in the table, single row, no warmup arithmetic to argue
        // about. If this loads and launches, the exact cubin matches the device.
        let probe = self.sweep_periods(
            &GpuSweepSpec {
                id: "sma",
                input: DeviceInput::CloseFromOhlcv,
            },
            &[1],
        );
        match probe {
            Ok(_) => Ok(()),
            Err(e) => {
                let loader = vector_ta::cuda::module_loader::last_module_load_failure()
                    .unwrap_or_else(|| "<no module-load diagnostic recorded>".to_string());
                bail!(
                    "GpuIndicatorEngine: the CUDA f64 indicator lane is selected but cannot run \
                     on this device.\n\
                     \n\
                       device       : {} (compute capability {})\n\
                       kernels built: exact native cubins for {:?} (source={})\n\
                     \n\
                     This runtime accepts only an exact cubin for the current compute capability. \
                     If the device architecture is absent, rebuild naming it explicitly:\n\
                     \n\
                       CUDA_ARCHS={} cargo build -p neoethos-data --features \
                     gpu-cuda\n\
                     \n\
                     (CUDA_FAST_MATH is irrelevant to the f64 lane by construction: \
                     kernels/cuda/neoethos_f64_kernels.cu is listed in vector-ta build.rs's \
                     F64_LANE_SOURCES and is always compiled with -prec-div=true -prec-sqrt=true \
                     -fmad=false -ftz=false and never with --use_fast_math.)\n\
                     \n\
                     Refusing to fall back to the CPU silently, and refusing to fall back to the \
                     f32 kernels.\n\
                     \n\
                     vector-ta module loader said:\n{}\n\
                     \n\
                     Underlying error: {e:?}",
                    self.device_name,
                    self.device_arch,
                    VECTOR_TA_NATIVE_ARCHS,
                    VECTOR_TA_ARCH_SOURCE,
                    self.device_arch.trim_start_matches("sm_"),
                    loader,
                )
            }
        }
    }

    pub fn device_arch(&self) -> &str {
        &self.device_arch
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn uploads(&self) -> u64 {
        self.uploads
    }

    pub fn lane(&self) -> IndicatorLane {
        IndicatorLane::Gpu {
            arch: self.device_arch.clone(),
        }
    }

    /// The numeric class of this lane's results relative to the CPU reference,
    /// for telemetry.
    ///
    /// It is f64 because the upload, the kernels and the download are all f64
    /// and the kernels are compiled without fast math. The WORD is not the
    /// proof; `hpc_ta::gpu_cpu_indicator_sweep_parity` on real bars is.
    pub fn precision(&self) -> &'static str {
        "f64 (no narrowing, no fast math)"
    }

    fn data_ref(&self, input: DeviceInput) -> Result<IndicatorCudaDeviceDataRefF64> {
        Ok(match input {
            DeviceInput::CloseFromOhlcv | DeviceInput::Ohlc => {
                IndicatorCudaDeviceDataRefF64::Ohlcv(self.ohlcv.as_view())
            }
            DeviceInput::Hlc3Slice => IndicatorCudaDeviceDataRefF64::Slice {
                values: self.hlc3.as_view_f64(),
            },
            DeviceInput::Hlc3CloseVolume => IndicatorCudaDeviceDataRefF64::CloseVolume(
                CudaDeviceCloseVolumeF64Ref::new(
                    self.hlc3.as_view_f64(),
                    self.ohlcv.volume.as_view_f64(),
                )
                .map_err(|e| {
                    anyhow::anyhow!("CudaDeviceCloseVolumeF64Ref::new(hlc3, volume) failed: {e:?}")
                })?,
            ),
            // The SAME device shape as the arm above with a DIFFERENT price
            // series in it. This is the one place the two are distinguished,
            // and the vector-ta kernel table is what decides which arm an
            // indicator lands in — see the `declared_input != spec.input` check
            // in `sweep_periods`, which refuses a disagreement rather than
            // letting it become a plausible number.
            DeviceInput::CloseVolume => IndicatorCudaDeviceDataRefF64::CloseVolume(
                CudaDeviceCloseVolumeF64Ref::new(
                    self.ohlcv.close.as_view_f64(),
                    self.ohlcv.volume.as_view_f64(),
                )
                .map_err(|e| {
                    anyhow::anyhow!("CudaDeviceCloseVolumeF64Ref::new(close, volume) failed: {e:?}")
                })?,
            ),
            DeviceInput::HighLow => IndicatorCudaDeviceDataRefF64::HighLow(
                CudaDeviceHighLowF64Ref::new(
                    self.ohlcv.high.as_view_f64(),
                    self.ohlcv.low.as_view_f64(),
                )
                .map_err(|e| {
                    anyhow::anyhow!("CudaDeviceHighLowF64Ref::new(high, low) failed: {e:?}")
                })?,
            ),
            // shard 1: the same explicit-Slice treatment hlc3 gets, over the
            // resident hl2 upload.
            DeviceInput::Hl2Slice => IndicatorCudaDeviceDataRefF64::Slice {
                values: self.hl2.as_view_f64(),
            },
            // closer 5: the same explicit-Slice treatment hlc3 and hl2 get.
            DeviceInput::Hlcc4Slice => IndicatorCudaDeviceDataRefF64::Slice {
                values: self.hlcc4.as_view_f64(),
            },
            // closer 6, round 3: the same (price, volume) device shape the two
            // arms above use, built from the resident hlcc4 upload.
            DeviceInput::Hlcc4CloseVolume => IndicatorCudaDeviceDataRefF64::CloseVolume(
                CudaDeviceCloseVolumeF64Ref::new(
                    self.hlcc4.as_view_f64(),
                    self.ohlcv.volume.as_view_f64(),
                )
                .map_err(|e| {
                    anyhow::anyhow!("CudaDeviceCloseVolumeF64Ref::new(hlcc4, volume) failed: {e:?}")
                })?,
            ),
            // closer 5: volume is already resident inside the OHLCV upload, so
            // this shape is a VIEW rather than a ninth transfer.
            DeviceInput::VolumeSlice => IndicatorCudaDeviceDataRefF64::Slice {
                values: self.ohlcv.volume.as_view_f64(),
            },
            // shard 1: both of these are served by the resident OHLCV ref --
            // which series the kernel actually reads is settled by the vector-ta
            // input KIND, and the launch passes exactly those pointers.
            DeviceInput::HighLowVolume | DeviceInput::Hlcv | DeviceInput::Ohlcv5 => {
                IndicatorCudaDeviceDataRefF64::Ohlcv(self.ohlcv.as_view())
            }
            // shard 4: same reasoning -- the resident OHLCV ref carries open,
            // and vector-ta's `inputs_for` picks (open, close, volume) out of
            // it according to the declared input KIND.
            DeviceInput::OpenCloseVolume => {
                IndicatorCudaDeviceDataRefF64::Ohlcv(self.ohlcv.as_view())
            }
            DeviceInput::TimestampCloseVolume => {
                IndicatorCudaDeviceDataRefF64::TimestampCloseVolume {
                    timestamps: self.timestamps.as_view(),
                    close: self.ohlcv.close.as_view_f64(),
                    volume: self.ohlcv.volume.as_view_f64(),
                }
            }
        })
    }

    /// The index the kernel must start at, derived from the CPU rule
    /// vector-ta declares for THIS INDICATOR — not from the device input shape.
    ///
    /// The shape tells you which series are uploaded. It does NOT tell you how
    /// the CPU scanned them: three of the six high/low/close indicators use a
    /// rule none of the other three use, and `first_valid` sets both the NaN
    /// prefix and the seed window, so the wrong rule shifts the entire series
    /// rather than perturbing it. See
    /// [`vector_ta::indicators::dispatch::F64FirstValidRule`].
    fn first_valid_for(&self, input: DeviceInput, rule: F64FirstValidRule) -> usize {
        match rule {
            // adx.rs:201-219, natr.rs:226-235.
            F64FirstValidRule::HlcMaxOfIndependentFirsts => self.first_valid_hlc_max_of_firsts,
            // adxr.rs:255-258 — close alone.
            F64FirstValidRule::HlcCloseOnly => self.first_valid_close,
            // `vwap` has NO warmup prefix — `vwap_with_kernel` calls
            // `alloc_with_nan_prefix(n, 0)` — and its kernel ignores this
            // value. Reported as 0 rather than as close's index so telemetry
            // does not imply a warmup that does not exist.
            F64FirstValidRule::Ignored => 0,
            // ------------------------------------------------------- shard 6
            // These four ignore `input` for the same reason the three above
            // do: the rule is a property of the INDICATOR, and the whole point
            // of declaring it per indicator is that two indicators reading the
            // same pair of series may still start at different bars.
            // aroonosc.rs:16-20
            F64FirstValidRule::HighLowFinite => self.first_valid_high_low_finite,
            // fisher.rs: finite H/L are insufficient when H + L overflows.
            F64FirstValidRule::HighLowMidpointFinite => self.first_valid_hl2_finite,
            // parkinson_volatility.rs:214-223
            F64FirstValidRule::HighLowFiniteAndPositive => {
                self.first_valid_high_low_finite_positive
            }
            // donchian.rs:183-188
            F64FirstValidRule::MaxOfIndependentFirsts => self.first_valid_high_low_max_of_firsts,
            // historical_volatility.rs:334-355
            F64FirstValidRule::ConsecutiveValidReturnPair => self.first_valid_close_return_pair,
            // ------------------------------------------------------- shard 4
            // dvdiqqe.rs:385 -- `is_finite`, which also rejects an infinity
            // that the `!is_nan` scan every other close-reading indicator
            // uses would accept.
            F64FirstValidRule::CloseFinite => self.first_valid_close_finite,
            // ------------------------------------------------------- closer 5
            // ultosc.rs:391-401 -- a CONSECUTIVE PAIR, because the true range
            // reads close[i-1]. At least one bar later than first_valid_hlc.
            F64FirstValidRule::HlcConsecutivePairNonNan => self.first_valid_hlc_consecutive_pair,
            // volume_zone_oscillator.rs:271 -- VOLUME ALONE, is_finite. Close
            // is deliberately absent: a non-finite close is a signed-negative
            // bar inside the loop, not a skipped one.
            F64FirstValidRule::VolumeFiniteOnly => self.first_valid_volume_finite,
            // ------------------------------------------------------ closer 4
            // qstick.rs:235-243 -- open and close only. See the field's doc
            // comment for why this cannot be folded into `AllInputsNonNan`.
            F64FirstValidRule::OpenCloseNonNan => self.first_valid_open_close,
            // ------------------------------------------------------- closer 1
            // accumulation_swing_index.rs:245, daily_factor.rs:258
            F64FirstValidRule::Ohlc4AllFinite => self.first_valid_ohlc4_finite,
            // bop.rs:209-211
            F64FirstValidRule::Ohlc4AllNonNan => self.first_valid_ohlc4_non_nan,
            // andean_oscillator.rs:244
            F64FirstValidRule::OpenCloseFinite => self.first_valid_open_close_finite,
            // ----------------------------------------------- closer 6, round 3
            // elastic_volume_weighted_moving_average.rs -- price AND volume
            // both `is_finite`. NOT the `AllInputsNonNan` pair scan: that one
            // is `!is_nan` and would accept an INFINITE volume the CPU skips.
            // The selected price series must follow the row's DeviceInput.
            F64FirstValidRule::PriceVolumeFinite => match input {
                DeviceInput::CloseVolume => self.first_valid_close_volume_finite,
                DeviceInput::Hlcc4CloseVolume => self.first_valid_hlcc4_volume_finite,
                other => {
                    panic!("PriceVolumeFinite requires a price-volume device input, got {other:?}")
                }
            },
            F64FirstValidRule::AllInputsNonNan => match input {
                DeviceInput::CloseFromOhlcv => self.first_valid_close,
                DeviceInput::Ohlc => self.first_valid_hlc,
                DeviceInput::Hlc3Slice => self.first_valid_hlc3,
                DeviceInput::Hlc3CloseVolume => self.first_valid_hlc3_volume,
                DeviceInput::CloseVolume => self.first_valid_close_volume,
                DeviceInput::HighLow => self.first_valid_high_low,
                DeviceInput::TimestampCloseVolume => 0,
                DeviceInput::Hl2Slice => self.first_valid_hl2,
                DeviceInput::HighLowVolume => self.first_valid_high_low_volume,
                DeviceInput::Hlcv => self.first_valid_hlcv,
                DeviceInput::OpenCloseVolume => self.first_valid_open_close_volume,
                DeviceInput::Hlcc4Slice => self.first_valid_hlcc4,
                // closer 6, round 3. No row currently declares this pair with
                // the common non-NaN rule. The arm exists because the match is
                // exhaustive, and reports the stricter finite-pair index.
                DeviceInput::Hlcc4CloseVolume => self.first_valid_hlcc4_volume_finite,
                DeviceInput::VolumeSlice => self.first_valid_volume,
                // closer 5, round 2. NO ROW IN `F64_KERNELS` USES THIS PAIR
                // TODAY -- the only `Ohlcv5` indicator, `trend_flow_trail`, is
                // registered `Ignored` because its CPU row walks every bar from
                // index 0 and resets mid-series. The arm exists because the
                // match is exhaustive, and it reports the MAX of two
                // INDEPENDENT scans: `first_valid_hlcv` covers high/low/close/
                // volume and `first_valid_open_close_volume` covers open/close/
                // volume, so between them all five series are seen.
                //
                // That max is a LOWER BOUND on "the first index at which all
                // five are simultaneously non-NaN", not that index itself: on a
                // frame where high has a hole after open has started, the true
                // index is later. A future `Ohlcv5` indicator that genuinely
                // scans all five must add a real five-series field rather than
                // adopt this one, for exactly the reason the six high/low/close
                // indicators each declare their own rule.
                DeviceInput::Ohlcv5 => self
                    .first_valid_hlcv
                    .max(self.first_valid_open_close_volume),
            },
        }
    }

    /// Launch one registered f64 indicator and keep its primary output on the
    /// selected device.
    ///
    /// This is the generic resident door used by the complete `GpuOnly` plan.
    /// It resolves exclusively through vector-ta's authoritative f64 registry;
    /// the historical [`GPU_SWEEP_SPECS`] table is deliberately not consulted.
    /// Multi-output completion is tracked separately because this dispatcher
    /// currently exposes one explicitly defined primary output per kernel.
    pub fn compute_primary_device(
        &self,
        indicator_id: &str,
        periods: &[usize],
    ) -> Result<IndicatorCudaOutputF64> {
        if periods.is_empty() {
            bail!("{indicator_id}: GpuOnly device request has no periods");
        }
        if periods.iter().any(|&period| period == 0) {
            bail!("{indicator_id}: period must be >= 1");
        }

        let route = f64_primary_device_route_for(indicator_id).ok_or_else(|| {
            anyhow::anyhow!(
                "{indicator_id}: no registered f64 device route; GpuOnly never falls back to CPU \
                 or f32"
            )
        })?;
        let periods_i32: Vec<i32> = periods
            .iter()
            .copied()
            .map(|period| {
                i32::try_from(period).map_err(|_| {
                    anyhow::anyhow!("{indicator_id}: period {period} exceeds the CUDA i32 ABI")
                })
            })
            .collect::<Result<_>>()?;
        let request = IndicatorCudaDeviceRequestF64 {
            indicator_id: route.indicator_id,
            data: self.data_ref(route.input)?,
            periods: &periods_i32,
            first_valid: self.first_valid_for(route.input, route.first_valid),
            target: CudaOutputTargetF64::Device,
        };
        let output = compute_cuda_device_f64(&self.f64_engine, request).map_err(|error| {
            anyhow::anyhow!(
                "compute_cuda_device_f64({indicator_id}, periods={periods:?}) failed on {} \
                 ({}): {error}",
                self.device_name,
                self.device_arch
            )
        })?;

        if output.rows != periods.len() || output.cols != self.n {
            bail!(
                "{indicator_id}: resident result shape {}x{} != {}x{}",
                output.rows,
                output.cols,
                periods.len(),
                self.n
            );
        }
        if !matches!(&output.series, IndicatorCudaSeriesF64::DeviceF64(_)) {
            bail!("{indicator_id}: GpuOnly request unexpectedly materialized HostF64");
        }
        Ok(output)
    }

    /// Materialize one already-computed primary f64 matrix at the canonical
    /// host feature boundary.  A strict run never asks the dispatcher for a
    /// host target and never widens an f32 result after the fact.
    pub fn download_primary_output_f64(&self, output: IndicatorCudaOutputF64) -> Result<Vec<f64>> {
        let indicator_id = output.indicator_id.clone();
        let rows = output.rows;
        let cols = output.cols;
        let values = match output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => {
                self.runtime.download_matrix_f64(&matrix).map_err(|error| {
                    anyhow::anyhow!(
                        "download resident f64 matrix for {indicator_id} ({rows}x{cols}) failed on \
                         {} ({}): {error:?}",
                        self.device_name,
                        self.device_arch
                    )
                })?
            }
            IndicatorCudaSeriesF64::HostF64(_) => {
                bail!("{indicator_id}: strict resident output was already materialized on host")
            }
        };
        let expected = rows.checked_mul(cols).ok_or_else(|| {
            anyhow::anyhow!("{indicator_id}: resident result shape {rows}x{cols} overflows usize")
        })?;
        if values.len() != expected {
            bail!(
                "{indicator_id}: downloaded {} f64 values for a {rows}x{cols} resident result",
                values.len()
            );
        }
        Ok(values)
    }

    /// Materialize a proven all-output launch at the same host f64 feature
    /// boundary as a primary matrix.  The result remains borrowed so its
    /// asynchronous-launch parameter buffers stay alive through synchronization
    /// and every admitted output download. `planned_output_ids` is the exact
    /// canonical feature schema; a full-kernel auxiliary that production has
    /// already excluded remains resident and is never materialized on host.
    pub fn download_named_outputs_f64(
        &self,
        result: &F64NamedOutputsResult,
        planned_output_ids: &[&'static str],
    ) -> Result<Vec<(&'static str, Vec<f64>)>> {
        let expected = result.rows.checked_mul(result.cols).ok_or_else(|| {
            anyhow::anyhow!(
                "{}: named resident result shape {}x{} overflows usize",
                result.indicator_id,
                result.rows,
                result.cols
            )
        })?;
        let mut downloaded = Vec::with_capacity(planned_output_ids.len());
        for (planned_index, &planned_output_id) in planned_output_ids.iter().enumerate() {
            if planned_output_ids[..planned_index].contains(&planned_output_id) {
                bail!(
                    "{}: duplicate planned named output `{planned_output_id}`",
                    result.indicator_id
                );
            }
            let mut matching = result
                .outputs
                .iter()
                .filter(|output| output.output_id == planned_output_id);
            let Some(output) = matching.next() else {
                bail!(
                    "{}: planned named output `{planned_output_id}` is absent from the resident \
                     result",
                    result.indicator_id
                );
            };
            if matching.next().is_some() {
                bail!(
                    "{}: resident result contains duplicate named output `{planned_output_id}`",
                    result.indicator_id
                );
            }
            if output.matrix.rows() != result.rows || output.matrix.cols() != result.cols {
                bail!(
                    "{}.{}: resident matrix shape {}x{} != result shape {}x{}",
                    result.indicator_id,
                    output.output_id,
                    output.matrix.rows(),
                    output.matrix.cols(),
                    result.rows,
                    result.cols
                );
            }
            let values = self
                .runtime
                .download_matrix_f64(&output.matrix)
                .map_err(|error| {
                    anyhow::anyhow!(
                        "download resident f64 matrix for {}.{} ({}x{}) failed on {} ({}): \
                         {error:?}",
                        result.indicator_id,
                        output.output_id,
                        result.rows,
                        result.cols,
                        self.device_name,
                        self.device_arch
                    )
                })?;
            if values.len() != expected {
                bail!(
                    "{}.{}: downloaded {} f64 values for a {}x{} resident result",
                    result.indicator_id,
                    output.output_id,
                    values.len(),
                    result.rows,
                    result.cols
                );
            }
            downloaded.push((output.output_id, values));
        }
        Ok(downloaded)
    }

    /// Launch all three ASI oscillator outputs from the already-resident close
    /// series through the same CUDA session used by every other f64 route.
    pub fn compute_absolute_strength_index_oscillator_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .absolute_strength_index_oscillator_all_outputs(
                self.ohlcv.close.as_view_f64(),
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "absolute_strength_index_oscillator all-output resident launch failed on {} \
                     ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "absolute_strength_index_oscillator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch bull, bear, and signal from the frame's one resident open/close
    /// upload through the shared f64 CUDA session.
    pub fn compute_andean_oscillator_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .andean_oscillator_all_outputs(
                self.ohlcv.as_view(),
                self.first_valid_open_close_finite,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "andean_oscillator all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "andean_oscillator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Aroon's canonical up/down pair from the frame's one resident
    /// high/low upload through the shared f64 CUDA session.
    pub fn compute_aroon_outputs_device(&self, lengths: &[usize]) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .aroon_all_outputs(self.ohlcv.as_view(), lengths)
            .map_err(|error| {
                anyhow::anyhow!(
                    "aroon all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != lengths.len() || result.cols != self.n {
            bail!(
                "aroon resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                lengths.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch ASO's canonical bulls/bears pair from the frame's resident OHLC
    /// upload through the shared f64 CUDA session.
    pub fn compute_aso_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .aso_all_outputs(
                self.ohlcv.as_view(),
                self.first_valid_close,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "aso all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "aso resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Autocorrelation Indicator's canonical filtered/selected-
    /// correlation pair from the frame's resident close series. One tuple is
    /// one exact `(length, lag, use_test_signal)` state machine; the standalone
    /// all-lag wrapper is not part of this production route.
    pub fn compute_autocorrelation_indicator_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize, bool)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .autocorrelation_indicator_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.max_consecutive_finite_close,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "autocorrelation_indicator selected-output resident launch failed on {} \
                     ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "autocorrelation_indicator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch AVSL's canonical value matrix from the frame's resident close,
    /// low, and volume buffers. Each exact `(fast, slow, multiplier)` tuple is
    /// one sequential row inside a single shared-session CUDA launch.
    pub fn compute_avsl_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .avsl_production_output(
                self.ohlcv.close.as_view_f64(),
                self.ohlcv.low.as_view_f64(),
                self.ohlcv.volume.as_view_f64(),
                self.first_valid_avsl,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "avsl parameterized resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "avsl resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Bandpass's canonical bp/bp_normalized/signal/trigger matrices
    /// from the frame's one resident close upload. Every exact
    /// `(period, bandwidth)` tuple owns both IIR passes in one sequential CUDA
    /// thread and all four matrices remain resident until the FeatureFrame
    /// materialization boundary.
    pub fn compute_bandpass_outputs_device(
        &self,
        parameter_tuples: &[(usize, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .bandpass_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.first_valid_close_finite,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "bandpass four-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "bandpass resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Bollinger Bands' canonical upper/middle/lower matrices from the
    /// frame's one resident close upload. Every exact
    /// `(period, devup, devdn)` tuple owns the scalar rolling state in one
    /// sequential CUDA thread and all three matrices remain resident until the
    /// FeatureFrame materialization boundary.
    pub fn compute_bollinger_bands_outputs_device(
        &self,
        parameter_tuples: &[(usize, f64, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .bollinger_bands_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.first_valid_close,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "bollinger_bands three-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "bollinger_bands resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Buff Averages' canonical fast/slow matrices from the frame's
    /// one resident close/volume upload. Every exact `(fast_period,
    /// slow_period)` tuple owns both rolling volume-weighted states in one
    /// sequential CUDA thread and both matrices remain resident until the
    /// FeatureFrame materialization boundary.
    pub fn compute_buff_averages_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .buff_averages_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.ohlcv.volume.as_view_f64(),
                self.first_valid_close,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "buff_averages two-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "buff_averages resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Candle Strength Oscillator's canonical six matrices from the
    /// frame's one resident OHLC upload. Every exact parameter row owns the
    /// complete nested-HMA, level, and signal state in one sequential CUDA
    /// thread; all matrices stay resident until FeatureFrame materialization.
    pub fn compute_candle_strength_oscillator_outputs_device(
        &self,
        parameter_rows: &[CandleStrengthOscillatorParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .candle_strength_oscillator_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_ohlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "candle_strength_oscillator six-output resident launch failed on {} \
                     ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "candle_strength_oscillator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Cyberpunk Value Trend Analyzer's canonical six outputs from the
    /// frame's one resident OHLC upload. Every threshold tuple owns its exact
    /// rolling/filter state in one sequential CUDA thread, and all matrices
    /// stay resident until FeatureFrame materialization.
    pub fn compute_cyberpunk_value_trend_analyzer_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .cyberpunk_value_trend_analyzer_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_ohlc,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "cyberpunk_value_trend_analyzer six-output resident launch failed on {} \
                     ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "cyberpunk_value_trend_analyzer resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Cycle Channel Oscillator's canonical fast/slow pair from the
    /// frame's one resident default-close HLC view. Every admitted coupled
    /// length/multiplier tuple owns its sequential RMA/ATR/history state, and
    /// both matrices remain resident until FeatureFrame materialization.
    pub fn compute_cycle_channel_oscillator_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize, f64, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .cycle_channel_oscillator_all_outputs(
                self.ohlcv.as_view(),
                self.first_valid_hlc_finite,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "cycle_channel_oscillator two-output resident launch failed on {} ({}): \
                     {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "cycle_channel_oscillator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Daily Factor's canonical value/ema/signal triple from the one
    /// resident OHLC frame. The threshold array and all output matrices remain
    /// device-owned until FeatureFrame materialization.
    pub fn compute_daily_factor_outputs_device(
        &self,
        threshold_levels: &[f64],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .daily_factor_all_outputs(
                self.ohlcv.as_view(),
                self.first_valid_ohlc4_finite,
                threshold_levels,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "daily_factor three-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != threshold_levels.len() || result.cols != self.n {
            bail!(
                "daily_factor resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                threshold_levels.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Damiani Volatmeter's canonical vol/anti pair from the one
    /// resident close upload. Every admitted four-window/threshold tuple owns
    /// its sequential ATR, variance, and lag state until frame materialization.
    pub fn compute_damiani_volatmeter_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize, usize, usize, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .damiani_volatmeter_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.first_valid_close,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "damiani_volatmeter two-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "damiani_volatmeter resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch DI's canonical plus/minus pair from the frame's one resident HLC
    /// upload. Every admitted period owns one sequential Wilder state and both
    /// matrices remain resident until FeatureFrame materialization.
    pub fn compute_di_outputs_device(&self, periods: &[usize]) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .di_all_outputs(self.ohlcv.as_view(), self.first_valid_hlc, periods)
            .map_err(|error| {
                anyhow::anyhow!(
                    "di two-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != periods.len() || result.cols != self.n {
            bail!(
                "di resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                periods.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch DM's canonical plus/minus pair from the frame's one resident
    /// high/low upload. Every admitted period owns one sequential Wilder state
    /// and both matrices remain resident until FeatureFrame materialization.
    pub fn compute_dm_outputs_device(&self, periods: &[usize]) -> Result<F64NamedOutputsResult> {
        let high_low = CudaDeviceHighLowF64Ref::new(
            self.ohlcv.high.as_view_f64(),
            self.ohlcv.low.as_view_f64(),
        )
        .map_err(|error| {
            anyhow::anyhow!("DM resident high/low view construction failed: {error:?}")
        })?;
        let result = self
            .f64_engine
            .dm_all_outputs(high_low, self.first_valid_high_low, periods)
            .map_err(|error| {
                anyhow::anyhow!(
                    "dm two-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != periods.len() || result.cols != self.n {
            bail!(
                "dm resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                periods.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Fisher's canonical fisher/signal pair from the frame's one
    /// resident high/low upload. Every admitted period owns one sequential
    /// recurrence and both matrices stay resident until FeatureFrame download.
    pub fn compute_fisher_outputs_device(
        &self,
        periods: &[usize],
    ) -> Result<F64NamedOutputsResult> {
        let high_low = CudaDeviceHighLowF64Ref::new(
            self.ohlcv.high.as_view_f64(),
            self.ohlcv.low.as_view_f64(),
        )
        .map_err(|error| {
            anyhow::anyhow!("Fisher resident high/low view construction failed: {error:?}")
        })?;
        let result = self
            .f64_engine
            .fisher_all_outputs(high_low, self.first_valid_hl2_finite, periods)
            .map_err(|error| {
                anyhow::anyhow!(
                    "fisher two-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != periods.len() || result.cols != self.n {
            bail!(
                "fisher resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                periods.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch FBEO's canonical forward_backward/backward/histogram matrices
    /// from the frame's one resident close upload. Every admitted
    /// (length,smooth) tuple owns one sequential state and all outputs stay
    /// resident until FeatureFrame materialization.
    pub fn compute_forward_backward_exponential_oscillator_outputs_device(
        &self,
        parameter_rows: &[(usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .forward_backward_exponential_oscillator_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.finite_close_count,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "FBEO three-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "FBEO resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch FVG Trailing Stop's canonical upper/lower/upper_ts/lower_ts
    /// matrices from the frame's one resident HLC upload. Every admitted
    /// lookback/smoothing/reset tuple owns one sequential state machine and all
    /// four outputs stay resident until FeatureFrame materialization.
    pub fn compute_fvg_trailing_stop_outputs_device(
        &self,
        parameter_rows: &[(usize, usize, bool)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .fvg_trailing_stop_all_outputs(
                self.ohlcv.as_view(),
                self.first_valid_hlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "FVG Trailing Stop four-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "FVG Trailing Stop resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Gator Oscillator's canonical upper/lower/change quartet from the
    /// frame's one resident close upload. Each exact six-integer tuple owns one
    /// sequential EMA/ring state machine and all outputs remain device-resident.
    pub fn compute_gatorosc_outputs_device(
        &self,
        parameter_rows: &[(usize, usize, usize, usize, usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .gatorosc_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.first_valid_close,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "Gator Oscillator four-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "Gator Oscillator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch HalfTrend's complete canonical six-output state machine from the
    /// frame's existing resident high/low/close upload. No standalone wrapper,
    /// host materialization, or second CUDA session participates in this path.
    pub fn compute_halftrend_outputs_device(
        &self,
        parameter_rows: &[(usize, f64, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .halftrend_all_outputs(
                self.ohlcv.high.as_view_f64(),
                self.ohlcv.low.as_view_f64(),
                self.ohlcv.close.as_view_f64(),
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "HalfTrend six-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "HalfTrend resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Donchian's canonical upper/middle/lower matrices from the
    /// frame's one resident high/low upload. Every admitted period shares one
    /// bar-parallel launch and all outputs remain resident until FeatureFrame
    /// materialization.
    pub fn compute_donchian_outputs_device(
        &self,
        periods: &[usize],
    ) -> Result<F64NamedOutputsResult> {
        let high_low = CudaDeviceHighLowF64Ref::new(
            self.ohlcv.high.as_view_f64(),
            self.ohlcv.low.as_view_f64(),
        )
        .map_err(|error| {
            anyhow::anyhow!("Donchian resident high/low view construction failed: {error:?}")
        })?;
        let result = self
            .f64_engine
            .donchian_all_outputs(high_low, self.first_valid_high_low_max_of_firsts, periods)
            .map_err(|error| {
                anyhow::anyhow!(
                    "donchian three-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != periods.len() || result.cols != self.n {
            bail!(
                "donchian resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                periods.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Dual Ulcer Index's canonical long/short/threshold matrices from
    /// the frame's one resident close upload. Every admitted tuple owns one
    /// sequential extrema/square-sum/threshold state, and all outputs remain
    /// resident until FeatureFrame materialization.
    pub fn compute_dual_ulcer_index_outputs_device(
        &self,
        parameter_rows: &[(usize, bool, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .dual_ulcer_index_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.max_consecutive_valid_dual_ulcer_close,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "dual_ulcer_index three-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "dual_ulcer_index resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch DVDIQQE's canonical dvdi/fast/slow/center matrices from the
    /// frame's one resident open/close/volume upload. Every exact tuple owns
    /// one sequential PVI/NVI, six-EMA, ratchet and cumulative-center state.
    pub fn compute_dvdiqqe_outputs_device(
        &self,
        parameter_rows: &[(usize, usize, f64, f64, bool, bool, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .dvdiqqe_all_outputs(
                self.ohlcv.as_view(),
                self.first_valid_close_finite,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "dvdiqqe four-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "dvdiqqe resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Ehlers Autocorrelation Periodogram's canonical dominant-cycle
    /// and normalized-power matrices from the frame's resident close upload.
    /// Only immutable scalar-CPU coefficient/table bits cross to the device;
    /// every price-dependent operation remains in the one CUDA launch.
    pub fn compute_ehlers_autocorrelation_periodogram_outputs_device(
        &self,
        parameter_rows: &[(usize, usize, usize, bool)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .ehlers_autocorrelation_periodogram_all_outputs(
                self.ohlcv.close.as_view_f64(),
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "ehlers_autocorrelation_periodogram two-output resident launch failed on {} \
                     ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "ehlers_autocorrelation_periodogram resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Ehlers Linear Extrapolation Predictor's canonical five matrices
    /// from the frame's resident close upload. Only immutable scalar-CPU
    /// coefficient/Hann bits cross to the device; every price-dependent state
    /// transition remains in one sequential CUDA launch.
    pub fn compute_ehlers_linear_extrapolation_predictor_outputs_device(
        &self,
        parameter_rows: &[(usize, usize, f64, usize, i32)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .ehlers_linear_extrapolation_predictor_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.finite_close_count,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "ehlers_linear_extrapolation_predictor five-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "ehlers_linear_extrapolation_predictor resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Ehlers Undersampled Double Moving Average's canonical fast/slow
    /// pair from the frame's resident close upload. Only exact scalar-CPU Hann
    /// coefficients and integer tuple parameters cross to the device.
    pub fn compute_ehlers_undersampled_double_moving_average_outputs_device(
        &self,
        parameter_rows: &[(usize, usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .ehlers_undersampled_double_moving_average_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.first_valid_close,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "ehlers_undersampled_double_moving_average two-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "ehlers_undersampled_double_moving_average resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch EMA-Deviation-Corrected T3's canonical corrected/T3 pair from
    /// the frame's resident close upload. Only the bounded exact parameter
    /// tuples cross to the device; both price-dependent rows remain resident.
    pub fn compute_ema_deviation_corrected_t3_outputs_device(
        &self,
        parameter_rows: &[(usize, f64, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .ema_deviation_corrected_t3_all_outputs(
                self.ohlcv.close.as_view_f64(),
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "ema_deviation_corrected_t3 two-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "ema_deviation_corrected_t3 resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch EMD's canonical upper/middle/lower matrices from the frame's
    /// one resident high/low upload. Only immutable scalar-CPU coefficient
    /// bits and exact parameter tuples cross to the device; every
    /// price-dependent operation stays in the one shared-session launch.
    pub fn compute_emd_outputs_device(
        &self,
        parameter_rows: &[(usize, f64, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let high_low = CudaDeviceHighLowF64Ref::new(
            self.ohlcv.high.as_view_f64(),
            self.ohlcv.low.as_view_f64(),
        )
        .map_err(|error| {
            anyhow::anyhow!("EMD resident high/low view construction failed: {error:?}")
        })?;
        let result = self
            .f64_engine
            .emd_all_outputs(high_low, self.first_valid_high_low, parameter_rows)
            .map_err(|error| {
                anyhow::anyhow!(
                    "emd three-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "emd resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch EMD Trend's canonical direction/average/upper/lower matrices
    /// from the frame's one resident close upload. Only exact bounded
    /// `(length, mult)` tuples cross to the device; production never enters
    /// either compatibility wrapper.
    pub fn compute_emd_trend_outputs_device(
        &self,
        parameter_rows: &[(usize, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .emd_trend_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.first_valid_close,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "emd_trend four-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "emd_trend resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch ERI's canonical bull/bear pair from the frame's one resident
    /// OHLCV upload. The only new payload is the exact admitted period list;
    /// production never uploads a host-computed moving-average matrix.
    pub fn compute_eri_outputs_device(&self, periods: &[usize]) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .eri_all_outputs(self.ohlcv.as_view(), self.first_valid_hlc, periods)
            .map_err(|error| {
                anyhow::anyhow!(
                    "eri two-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != periods.len() || result.cols != self.n {
            bail!(
                "eri resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                periods.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Evasive Supertrend's canonical band/state/noisy/changed outputs
    /// from this frame's one resident OHLCV upload. Only the exact parameter
    /// tuples cross PCIe; production never constructs the legacy wrapper.
    pub fn compute_evasive_supertrend_outputs_device(
        &self,
        parameter_rows: &[(usize, f64, f64, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .evasive_supertrend_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_ohlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "evasive_supertrend four-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "evasive_supertrend resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch the complete Ehlers Data Sampling RSI triple from the frame's
    /// resident OHLC upload. Lengths are the only new device payload.
    pub fn compute_ehlers_data_sampling_rsi_outputs_device(
        &self,
        lengths: &[usize],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .ehlers_data_sampling_relative_strength_indicator_all_outputs(
                self.ohlcv.as_view(),
                lengths,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "ehlers_data_sampling_relative_strength_indicator three-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != lengths.len() || result.cols != self.n {
            bail!(
                "ehlers_data_sampling_relative_strength_indicator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                lengths.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Didi Index's canonical four outputs from the frame's one
    /// resident close upload. Every exact RegistryRatio tuple owns one
    /// sequential three-ring state and all matrices stay device-resident until
    /// FeatureFrame materialization.
    pub fn compute_didi_index_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .didi_index_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.finite_close_count,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "didi_index four-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "didi_index resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Directional Imbalance Index's canonical six outputs from the
    /// frame's one resident high/low upload. The exact simultaneous-finite
    /// receipt proves the scalar input contract before one shared-session
    /// launch retains all matrices to FeatureFrame materialization.
    pub fn compute_directional_imbalance_index_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .directional_imbalance_index_all_outputs(
                self.ohlcv.as_view(),
                self.first_valid_high_low_finite,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "directional_imbalance_index six-output resident launch failed on {} ({}): \
                     {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "directional_imbalance_index resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Disparity Index's sole canonical value matrix from the frame's
    /// already-resident close upload. Every exact four-parameter tuple owns
    /// both dynamic rings in one shared-session launch; the result remains on
    /// the selected device until FeatureFrame materialization.
    pub fn compute_disparity_index_output_device(
        &self,
        parameter_rows: &[(usize, usize, usize, bool)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .disparity_index_production_output(
                self.ohlcv.close.as_view_f64(),
                self.max_consecutive_finite_close,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "disparity_index parameterized resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "disparity_index resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Chandelier Exit's canonical long/short stop pair from the
    /// frame's one resident HLC upload. The shared f64 session retains the
    /// parameter and runtime deque buffers until FeatureFrame materialization.
    pub fn compute_chandelier_exit_outputs_device(
        &self,
        parameter_rows: &[ChandelierExitParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .chandelier_exit_all_outputs(
                self.ohlcv.as_view(),
                self.first_valid_close,
                self.first_valid_hlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "chandelier_exit two-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "chandelier_exit resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch CKSP's canonical long/short value pair from the frame's one
    /// resident HLC upload. Parameter and four-deque scratch ownership remains
    /// in the shared f64 session through FeatureFrame materialization.
    pub fn compute_cksp_outputs_device(
        &self,
        parameter_rows: &[CkspParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .cksp_all_outputs(self.ohlcv.as_view(), self.first_valid_close, parameter_rows)
            .map_err(|error| {
                anyhow::anyhow!(
                    "cksp two-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "cksp resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Coppock's canonical value matrix from the frame's one resident
    /// close upload. Each exact ROC/WMA tuple owns one sequential CUDA thread;
    /// the result remains resident until FeatureFrame materialization.
    pub fn compute_coppock_output_device(
        &self,
        parameter_rows: &[CoppockParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .coppock_production_output(
                self.ohlcv.close.as_view_f64(),
                self.first_valid_close,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "coppock parameterized resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "coppock resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch CVI's canonical value matrix from the frame's one resident
    /// high/low upload. The exact periods and result remain owned by the
    /// shared f64 session until FeatureFrame materialization.
    pub fn compute_cvi_output_device(&self, periods: &[usize]) -> Result<F64NamedOutputsResult> {
        let high_low = CudaDeviceHighLowF64Ref::new(
            self.ohlcv.high.as_view_f64(),
            self.ohlcv.low.as_view_f64(),
        )
        .map_err(|error| {
            anyhow::anyhow!("CVI resident high/low view construction failed: {error:?}")
        })?;
        let result = self
            .f64_engine
            .cvi_production_output(high_low, self.first_valid_high_low, periods)
            .map_err(|error| {
                anyhow::anyhow!(
                    "CVI parameterized resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != periods.len() || result.cols != self.n {
            bail!(
                "CVI resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                periods.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Correlation Cycle's canonical real/imag/angle/state matrices
    /// from the frame's one resident close upload. Every exact
    /// `(period, threshold)` tuple owns its complete sequential state in one
    /// shared-session CUDA launch.
    pub fn compute_correlation_cycle_outputs_device(
        &self,
        parameter_tuples: &[(usize, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .correlation_cycle_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.first_valid_close,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "correlation_cycle four-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "correlation_cycle resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch the canonical jaw/teeth/lips tuple from the frame's resident hl2
    /// upload through the one shared f64 CUDA session.
    pub fn compute_alligator_outputs_device(
        &self,
        parameter_rows: &[AlligatorParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .alligator_all_outputs(self.hl2.as_view_f64(), self.first_valid_hl2, parameter_rows)
            .map_err(|error| {
                anyhow::anyhow!(
                    "alligator all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "alligator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch AlphaTrend's canonical k1/k2 pair from the frame's resident
    /// OHLCV upload through the shared f64 CUDA session.
    pub fn compute_alphatrend_outputs_device(
        &self,
        parameter_rows: &[AlphaTrendParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .alphatrend_all_outputs(self.ohlcv.as_view(), self.first_valid_close, parameter_rows)
            .map_err(|error| {
                anyhow::anyhow!(
                    "alphatrend all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "alphatrend resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all three Adaptive MACD outputs from the frame's resident close
    /// series. The tuple list is the real indicator parameter space; signal
    /// and histogram are emitted by the same device state machine as MACD.
    pub fn compute_adaptive_macd_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize, usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .adaptive_macd_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.first_valid_close,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "adaptive_macd all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "adaptive_macd resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch both Adaptive Momentum Oscillator outputs from the resident
    /// close series through the frame's one CUDA session.
    pub fn compute_adaptive_momentum_oscillator_outputs_device(
        &self,
        parameter_tuples: &[(usize, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .adaptive_momentum_oscillator_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.first_valid_close,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "adaptive_momentum_oscillator all-output resident launch failed on {} \
                     ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "adaptive_momentum_oscillator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch both Adaptive Schaff Trend Cycle outputs from the frame's one
    /// resident HLC upload through the shared CUDA session.
    pub fn compute_adaptive_schaff_trend_cycle_outputs_device(
        &self,
        parameter_rows: &[AdaptiveSchaffTrendCycleParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .adaptive_schaff_trend_cycle_all_outputs(
                self.ohlcv.as_view(),
                self.adaptive_schaff_valid_suffix_len,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "adaptive_schaff_trend_cycle all-output resident launch failed on {} \
                     ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "adaptive_schaff_trend_cycle resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch both Ehlers Adaptive CG outputs from the resident `hl2` series
    /// through the frame's one CUDA session.
    pub fn compute_ehlers_adaptive_cg_outputs_device(
        &self,
        alphas: &[f64],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .ehlers_adaptive_cg_all_outputs(self.hl2.as_view_f64(), alphas)
            .map_err(|error| {
                anyhow::anyhow!(
                    "ehlers_adaptive_cg all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != alphas.len() || result.cols != self.n {
            bail!(
                "ehlers_adaptive_cg resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                alphas.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch both Ehlers Adaptive Cyber Cycle outputs from the resident
    /// `hl2` series through the frame's one CUDA session.
    pub fn compute_ehlers_adaptive_cyber_cycle_outputs_device(
        &self,
        alphas: &[f64],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .ehlers_adaptive_cyber_cycle_all_outputs(
                self.hl2.as_view_f64(),
                self.first_valid_hl2_finite,
                alphas,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "ehlers_adaptive_cyber_cycle all-output resident launch failed on {} \
                     ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != alphas.len() || result.cols != self.n {
            bail!(
                "ehlers_adaptive_cyber_cycle resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                alphas.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch both Ehlers Simple Cycle Indicator outputs from the resident
    /// `hl2` series through the frame's one CUDA session.
    pub fn compute_ehlers_simple_cycle_indicator_outputs_device(
        &self,
        alphas: &[f64],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .ehlers_simple_cycle_indicator_all_outputs(
                self.hl2.as_view_f64(),
                self.first_valid_hl2_finite,
                alphas,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "ehlers_simple_cycle_indicator all-output resident launch failed on {} \
                     ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != alphas.len() || result.cols != self.n {
            bail!(
                "ehlers_simple_cycle_indicator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                alphas.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch the one real, parameter-free Ehlers PMA row and retain both
    /// outputs in the frame's CUDA session.
    pub fn compute_ehlers_pma_outputs_device(&self) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .ehlers_pma_all_outputs(self.ohlcv.close.as_view_f64(), self.first_valid_close)
            .map_err(|error| {
                anyhow::anyhow!(
                    "ehlers_pma all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != 1 || result.cols != self.n {
            bail!(
                "ehlers_pma resident result shape {}x{} != 1x{}",
                result.rows,
                result.cols,
                self.n
            );
        }
        Ok(result)
    }

    /// Launch Fibonacci Trailing Stop's canonical four outputs from the
    /// frame's one resident OHLCV upload. Only exact parameter tuples cross
    /// PCIe; the legacy wrapper/context/upload path is not production-routable.
    pub fn compute_fibonacci_trailing_stop_outputs_device(
        &self,
        parameter_rows: &[(usize, usize, f64, i32)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .fibonacci_trailing_stop_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_hlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "fibonacci_trailing_stop four-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "fibonacci_trailing_stop resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all eighteen Fibonacci Entry Bands outputs from this frame's
    /// resident OHLC buffers through the one shared CUDA session.
    pub fn compute_fibonacci_entry_bands_outputs_device(
        &self,
        sweep: &FibonacciEntryBandsBatchRange,
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .fibonacci_entry_bands_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_hlc,
                self.max_consecutive_finite_ohlc,
                sweep,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "fibonacci_entry_bands all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.cols != self.n {
            bail!(
                "fibonacci_entry_bands resident result has {} bars, expected {}",
                result.cols,
                self.n
            );
        }
        Ok(result)
    }

    /// Launch both Adaptive Bandpass Trigger Oscillator outputs from the
    /// resident close series through the frame's one CUDA session.
    pub fn compute_adaptive_bandpass_trigger_oscillator_outputs_device(
        &self,
        parameter_tuples: &[(f64, f64)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .adaptive_bandpass_trigger_oscillator_all_outputs(
                self.ohlcv.close.as_view_f64(),
                self.finite_close_count,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "adaptive_bandpass_trigger_oscillator all-output resident launch failed on \
                     {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "adaptive_bandpass_trigger_oscillator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all seventeen Trend Flow Trail outputs from the frame's one
    /// resident OHLCV upload. The route borrows this engine's CUDA session;
    /// it never constructs the superseded wrapper/context and never
    /// materializes an intermediate host feature column.
    pub fn compute_trend_flow_trail_outputs_device(
        &self,
        parameter_tuples: &[(usize, f64, usize)],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .trend_flow_trail_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_ohlcv,
                parameter_tuples,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "trend_flow_trail all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_tuples.len() || result.cols != self.n {
            bail!(
                "trend_flow_trail resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_tuples.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all sixteen Market Structure Confluence outputs from the
    /// frame's one resident OHLC upload and shared CUDA session. The explicit
    /// parameter rows preserve caller order and both confirmation modes; no
    /// legacy range expansion, host feature computation or input re-upload is
    /// reachable from this route.
    pub fn compute_market_structure_confluence_outputs_device(
        &self,
        parameter_rows: &[MarketStructureConfluenceParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .market_structure_confluence_all_outputs(
                self.ohlcv.as_view(),
                self.finite_hlc_count,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "market_structure_confluence all-output resident launch failed on {} \
                     ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "market_structure_confluence resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all fifteen HEMA Trend Levels outputs from the frame's one
    /// resident OHLC upload and shared CUDA session.
    pub fn compute_hema_trend_levels_outputs_device(
        &self,
        parameter_rows: &[HemaTrendLevelsParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .hema_trend_levels_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_ohlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "hema_trend_levels all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "hema_trend_levels resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all thirteen Range Filtered Trend Signals outputs from the
    /// frame's one resident HLC upload and shared CUDA session.
    pub fn compute_range_filtered_trend_signals_outputs_device(
        &self,
        parameter_rows: &[RangeFilteredTrendSignalsParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .range_filtered_trend_signals_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_hlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "range_filtered_trend_signals all-output resident launch failed on {} \
                     ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "range_filtered_trend_signals resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all twelve ICT Propulsion Block outputs from the frame's one
    /// resident OHLC upload and shared CUDA session.
    pub fn compute_ict_propulsion_block_outputs_device(
        &self,
        parameter_rows: &[IctPropulsionBlockParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .ict_propulsion_block_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_ohlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "ict_propulsion_block all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "ict_propulsion_block resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all twelve Vdubus Divergence Wave Pattern Generator outputs
    /// from the frame's one resident HLC upload and shared CUDA session.
    pub fn compute_vdubus_divergence_wave_pattern_generator_outputs_device(
        &self,
        parameter_rows: &[VdubusDivergenceWavePatternGeneratorParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .vdubus_divergence_wave_pattern_generator_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_hlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "vdubus_divergence_wave_pattern_generator all-output resident launch failed \
                     on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "vdubus_divergence_wave_pattern_generator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all ten Adaptive Bounds RSI outputs from the frame's resident
    /// close upload and shared CUDA session. No host value is produced here;
    /// the returned matrices remain on the selected device.
    pub fn compute_adaptive_bounds_rsi_outputs_device(
        &self,
        parameter_rows: &[AdaptiveBoundsRsiParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .adaptive_bounds_rsi_all_outputs(
                self.ohlcv.as_view().close(),
                self.max_consecutive_finite_close,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "adaptive_bounds_rsi all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "adaptive_bounds_rsi resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all ten Adjustable MA & Alternating Extremities outputs from the
    /// frame's resident HLC upload. The shared CUDA session owns every result;
    /// no input is re-uploaded and no host result is produced by this call.
    pub fn compute_adjustable_ma_alternating_extremities_outputs_device(
        &self,
        parameter_rows: &[AdjustableMaAlternatingExtremitiesParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .adjustable_ma_alternating_extremities_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_hlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "adjustable_ma_alternating_extremities all-output resident launch failed on \
                     {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "adjustable_ma_alternating_extremities resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all ten Bulls v Bears outputs from the frame's resident HLC
    /// upload through the shared CUDA session. Parameter rows may independently
    /// select EMA/SMA/WMA and Normalized/Raw modes.
    pub fn compute_bulls_v_bears_outputs_device(
        &self,
        parameter_rows: &[BullsVBearsParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .bulls_v_bears_all_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_hlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "bulls_v_bears all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "bulls_v_bears resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all nine Range Oscillator outputs from the frame's one resident
    /// HLC upload. The common ATR state, bar-parallel weighted-window stage and
    /// sticky-trend stage all run on the shared CUDA stream; this call neither
    /// re-uploads HLC nor materializes an output on the host.
    pub fn compute_range_oscillator_outputs_device(
        &self,
        parameter_rows: &[RangeOscillatorParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .range_oscillator_all_outputs(
                self.ohlcv.as_view(),
                self.finite_hlc_count,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "range_oscillator all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "range_oscillator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all nine Pivot levels from the frame's one resident OHLC upload.
    /// Formula row `t` is causally sourced from period `t-1`; Woodie alone also
    /// reads `open[t]`. No host-side output or formula computation occurs.
    pub fn compute_pivot_outputs_device(
        &self,
        parameter_rows: &[PivotParams],
    ) -> Result<F64NamedOutputsResult> {
        for (row, params) in parameter_rows.iter().enumerate() {
            let mode = params.mode.unwrap_or(3);
            if mode > 4 {
                bail!("pivot row {row} has invalid formula mode {mode}; expected 0..=4");
            }
            if self.pivot_valid_mode_mask & (1u8 << mode) == 0 {
                bail!(
                    "pivot row {row} mode {mode} has no valid previous-period source in the resident frame"
                );
            }
        }
        let result = self
            .f64_engine
            .pivot_all_outputs(self.ohlcv.as_view(), parameter_rows)
            .map_err(|error| {
                anyhow::anyhow!(
                    "pivot all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "pivot resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch both ACOSC outputs from the frame's one resident high/low
    /// upload. The CUDA kernel owns formula and validity evaluation for every
    /// bar; this method performs orchestration and shape validation only.
    pub fn compute_acosc_outputs_device(&self) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .acosc_all_outputs(self.ohlcv.as_view())
            .map_err(|error| {
                anyhow::anyhow!(
                    "acosc all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != 1 || result.cols != self.n {
            bail!(
                "acosc resident result shape {}x{} != 1x{}",
                result.rows,
                result.cols,
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all ten mathematically distinct Kase outputs from the frame's
    /// one resident HLC upload and shared CUDA session. `hist` is intentionally
    /// absent: the scalar contract proves it is the exact oscillator display
    /// alias, so production search must not allocate or score it twice.
    pub fn compute_kase_peak_oscillator_with_divergences_outputs_device(
        &self,
        parameter_rows: &[KasePeakOscillatorWithDivergencesParams],
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .kase_peak_oscillator_with_divergences_unique_outputs(
                self.ohlcv.as_view(),
                self.max_consecutive_finite_hlc,
                parameter_rows,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "kase_peak_oscillator_with_divergences all-output resident launch failed \
                     on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        if result.rows != parameter_rows.len() || result.cols != self.n {
            bail!(
                "kase_peak_oscillator_with_divergences resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                parameter_rows.len(),
                self.n
            );
        }
        Ok(result)
    }

    /// Launch all thirteen Ichimoku Oscillator outputs against the resident
    /// high/low/close frame. The production batch contract's default source
    /// is close; the shared CUDA implementation accepts an explicit resident
    /// source view so non-default source routing can be added without an
    /// upload or a host-computed feature column.
    pub fn compute_ichimoku_oscillator_outputs_device(
        &self,
        sweep: &IchimokuOscillatorBatchRange,
    ) -> Result<F64NamedOutputsResult> {
        let result = self
            .f64_engine
            .ichimoku_oscillator_all_outputs(
                self.ohlcv.as_view(),
                self.ohlcv.close.as_view_f64(),
                self.first_valid_hlc_finite,
                sweep,
            )
            .map_err(|error| {
                anyhow::anyhow!(
                    "ichimoku_oscillator all-output resident launch failed on {} ({}): {error}",
                    self.device_name,
                    self.device_arch
                )
            })?;
        let expected_rows = vector_ta::indicators::ichimoku_oscillator::expand_grid(sweep)?.len();
        if result.rows != expected_rows || result.cols != self.n {
            bail!(
                "ichimoku_oscillator resident result shape {}x{} != {}x{}",
                result.rows,
                result.cols,
                expected_rows,
                self.n
            );
        }
        Ok(result)
    }

    /// Sweep ONE indicator across an explicit period list in ONE launch,
    /// device-resident in, host out. Returns one row per period, in order.
    ///
    /// # Why one launch for the whole list, and why the list is explicit
    ///
    /// The f32 lane went through vector-ta's `(start, end, step)` sweep API,
    /// which expands arithmetically. The periods this codebase wants —
    /// `[7, 21, 50, 100, 200]` — are not an arithmetic progression, so its only
    /// batched form was the contiguous `7..=200 step 1`: 194 rows of which 189
    /// are discarded, a 654 MB device allocation at 843k bars to keep 17 MB,
    /// and a size that is a function of the requested RANGE rather than of the
    /// hardware. The old workaround was one launch per period.
    ///
    /// The f64 lane takes the period LIST directly, so the whole sweep is one
    /// launch, the output is exactly `rows × n × 8` bytes, and nothing is
    /// computed that is thrown away. NEVER-OOM is held inside the wrapper,
    /// which chunks over rows against `mem_get_info` — a longer period list
    /// makes the sweep slower, never fatter.
    fn sweep_periods(&self, spec: &GpuSweepSpec, periods: &[usize]) -> Result<Vec<Vec<f64>>> {
        if periods.is_empty() {
            return Ok(Vec::new());
        }
        if periods.iter().any(|&p| p == 0) {
            bail!("{}: period must be >= 1", spec.id);
        }

        // Single-output only. Multi-output indicators require an explicit
        // output per series and would silently pick output 0 if we guessed.
        if let Some(info) = get_indicator(spec.id) {
            if info.outputs.len() > 1 {
                bail!(
                    "{}: multi-output indicator ({} outputs) is not in the single-output device \
                     sweep contract",
                    spec.id,
                    info.outputs.len()
                );
            }
        } else {
            bail!("{}: not in the vector-ta indicator registry", spec.id);
        }

        // vector-ta's f64 kernel table is the authority on which price series
        // this indicator needs. Disagreeing with it is a contract bug, not a
        // rounding difference, so it is checked here rather than left to the
        // parity tolerance to notice.
        let declared = f64_kernel_for(spec.id).ok_or_else(|| {
            anyhow::anyhow!(
                "{}: vector-ta has no f64 kernel for this indicator, so it must not be in \
                 GPU_SWEEP_SPECS. The f64 lane does not fall back to the f32 kernel and does not \
                 fall back to the CPU.",
                spec.id
            )
        })?;
        let declared_input = DeviceInput::from_vector_ta(declared.input);
        if declared_input != spec.input {
            bail!(
                "{}: GPU_SWEEP_SPECS says {:?} but vector-ta's f64 kernel table says {:?}. \
                 Feeding the wrong series computes a DIFFERENT indicator, not a less precise one.",
                spec.id,
                spec.input,
                declared_input
            );
        }

        let periods_i32: Vec<i32> = periods.iter().map(|&p| p as i32).collect();

        let req = IndicatorCudaDeviceRequestF64 {
            indicator_id: spec.id,
            data: self.data_ref(spec.input)?,
            periods: &periods_i32,
            // The RULE comes from vector-ta's table, not from `spec.input`.
            first_valid: self.first_valid_for(spec.input, declared.first_valid),
            target: CudaOutputTargetF64::Host,
        };

        let out = compute_cuda_device_f64(&self.f64_engine, req).map_err(|e| {
            anyhow::anyhow!(
                "compute_cuda_device_f64({}, periods={periods:?}) failed on {} ({}): {e}",
                spec.id,
                self.device_name,
                self.device_arch
            )
        })?;

        let host: Vec<f64> = match out.series {
            IndicatorCudaSeriesF64::HostF64(v) => v,
            IndicatorCudaSeriesF64::DeviceF64(_) => bail!(
                "{}: compute_cuda_device_f64 returned DeviceF64 despite a Host target",
                spec.id
            ),
        };

        // Shape is checked BEFORE any indexing. Anything other than
        // `periods.len() x n` means the contract for this indicator is not what
        // the table claims, and that is a correctness bug, not a fallback
        // condition.
        if out.rows != periods.len() {
            bail!(
                "{}: expected {} rows for a {}-period sweep, got rows={} cols={}",
                spec.id,
                periods.len(),
                periods.len(),
                out.rows,
                out.cols
            );
        }
        if out.cols != self.n {
            bail!(
                "{}: cols={} but the frame has {} bars",
                spec.id,
                out.cols,
                self.n
            );
        }
        let expected = periods.len().saturating_mul(self.n);
        if host.len() != expected {
            bail!(
                "{}: host buffer len {} != {} ({} rows x {} bars)",
                spec.id,
                host.len(),
                expected,
                periods.len(),
                self.n
            );
        }

        Ok(host.chunks_exact(self.n).map(|r| r.to_vec()).collect())
    }

    /// Sweep one indicator across `periods`, emitting `{id}_{period}` columns
    /// in the given order.
    ///
    /// Applies the SAME `(period as f64) * 1.25 >= n` pre-flight rule as the
    /// CPU sweep. Non-computable periods do not launch, but they still emit an
    /// all-NaN column in the requested position so the feature schema is frame
    /// invariant and identical across CPU/CUDA lanes.
    pub fn sweep_columns(
        &self,
        spec: &GpuSweepSpec,
        periods: &[usize],
    ) -> Result<Vec<(String, Vec<f64>)>> {
        let kept: Vec<usize> = periods
            .iter()
            .copied()
            .filter(|&period| period_fits_frame(period, self.n))
            .collect();
        let rows = if kept.is_empty() {
            Vec::new()
        } else {
            self.sweep_periods(spec, &kept)?
        };
        assemble_sweep_columns(spec.id, periods, self.n, rows)
    }

    /// Whether this request issues a kernel launch rather than returning only
    /// schema-preserving all-NaN placeholders.
    pub fn has_launchable_period(&self, periods: &[usize]) -> bool {
        periods
            .iter()
            .any(|&period| period_fits_frame(period, self.n))
    }

    /// Block until every launch issued so far has retired. Called once at the
    /// end of a frame so the recorded device time is real work, not an
    /// asynchronous queue depth.
    pub fn synchronize(&self) -> Result<()> {
        self.f64_engine
            .synchronize()
            .map_err(|e| anyhow::anyhow!("CudaF64Indicators::synchronize failed: {e}"))?;
        self.runtime
            .synchronize()
            .map_err(|e| anyhow::anyhow!("CudaRuntime::synchronize failed: {e:?}"))
    }

    pub fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }
}

#[inline]
fn period_fits_frame(period: usize, n: usize) -> bool {
    (period as f64) * 1.25 < n as f64
}

/// Reinsert frame-dependent preflight misses without changing the requested
/// schema. `computed_rows` contains exactly the fitting periods, in request
/// order; a length mismatch is a device-contract error rather than a partial
/// result.
fn assemble_sweep_columns(
    indicator_id: &str,
    periods: &[usize],
    n: usize,
    computed_rows: Vec<Vec<f64>>,
) -> Result<Vec<(String, Vec<f64>)>> {
    let expected_rows = periods
        .iter()
        .filter(|&&period| period_fits_frame(period, n))
        .count();
    if computed_rows.len() != expected_rows {
        bail!(
            "{indicator_id}: expected {expected_rows} computed period rows before schema assembly, got {}",
            computed_rows.len()
        );
    }

    let mut rows = computed_rows.into_iter();
    let mut columns = Vec::with_capacity(periods.len());
    for &period in periods {
        let values = if period_fits_frame(period, n) {
            rows.next().expect("computed row count checked above")
        } else {
            vec![f64::NAN; n]
        };
        columns.push((format!("{indicator_id}_{period}"), values));
    }
    debug_assert!(rows.next().is_none());
    Ok(columns)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS: [&str; 6] = [
        "value_trend",
        "value_trend_lag",
        "deviation_index",
        "overbought_signal",
        "buy_signal",
        "sell_signal",
    ];
    const CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS: [&str; 2] = ["fast", "slow"];
    const DAILY_FACTOR_OUTPUT_IDS: [&str; 3] = ["value", "ema", "signal"];
    const DAMIANI_VOLATMETER_OUTPUT_IDS: [&str; 2] = ["vol", "anti"];
    const DI_OUTPUT_IDS: [&str; 2] = ["plus", "minus"];
    const DM_OUTPUT_IDS: [&str; 2] = ["plus", "minus"];
    const DONCHIAN_OUTPUT_IDS: [&str; 3] = ["upper", "middle", "lower"];
    const DIDI_INDEX_OUTPUT_IDS: [&str; 4] = ["short", "long", "crossover", "crossunder"];
    const DUAL_ULCER_INDEX_OUTPUT_IDS: [&str; 3] = ["long_ulcer", "short_ulcer", "threshold"];
    const DVDIQQE_OUTPUT_IDS: [&str; 4] = ["dvdi", "fast_tl", "slow_tl", "center_line"];
    const EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS: [&str; 2] =
        ["dominant_cycle", "normalized_power"];
    const EHLERS_DATA_SAMPLING_RSI_OUTPUT_IDS: [&str; 3] = ["ds_rsi", "original_rsi", "signal"];
    const EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS: [&str; 5] =
        ["prediction", "filter", "state", "go_long", "go_short"];
    const EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS: [&str; 2] = ["fast", "slow"];
    const EMD_OUTPUT_IDS: [&str; 3] = ["upperband", "middleband", "lowerband"];
    const EMD_TREND_OUTPUT_IDS: [&str; 4] = ["direction", "average", "upper", "lower"];
    const ERI_OUTPUT_IDS: [&str; 2] = ["bull", "bear"];
    const EVASIVE_SUPERTREND_OUTPUT_IDS: [&str; 4] = ["band", "state", "noisy", "changed"];
    const FISHER_OUTPUT_IDS: [&str; 2] = ["fisher", "signal"];
    const FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS: [&str; 3] =
        ["forward_backward", "backward", "histogram"];
    const DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS: [&str; 6] =
        ["up", "down", "bulls", "bears", "upper", "lower"];

    const TREND_FLOW_TRAIL_OUTPUT_IDS: [&str; 17] = [
        "alpha_trail",
        "alpha_trail_bullish",
        "alpha_trail_bearish",
        "alpha_dir",
        "mfi",
        "tp_upper",
        "tp_lower",
        "alpha_trail_bullish_switch",
        "alpha_trail_bearish_switch",
        "mfi_overbought",
        "mfi_oversold",
        "mfi_cross_up_mid",
        "mfi_cross_down_mid",
        "price_cross_alpha_trail_up",
        "price_cross_alpha_trail_down",
        "mfi_above_90",
        "mfi_below_10",
    ];

    const MARKET_STRUCTURE_CONFLUENCE_OUTPUT_IDS: [&str; 16] = [
        "basis",
        "upper_band",
        "lower_band",
        "structure_direction",
        "bullish_arrow",
        "bearish_arrow",
        "bullish_change",
        "bearish_change",
        "hh",
        "lh",
        "hl",
        "ll",
        "bullish_bos",
        "bullish_choch",
        "bearish_bos",
        "bearish_choch",
    ];

    const HEMA_TREND_LEVELS_OUTPUT_IDS: [&str; 15] = [
        "fast_hema",
        "slow_hema",
        "trend_direction",
        "bar_state",
        "bullish_crossover",
        "bearish_crossunder",
        "box_offset",
        "bull_box_top",
        "bull_box_bottom",
        "bear_box_top",
        "bear_box_bottom",
        "bullish_test",
        "bearish_test",
        "bullish_test_level",
        "bearish_test_level",
    ];

    const ICHIMOKU_OSCILLATOR_OUTPUT_IDS: [&str; 13] = [
        "signal",
        "ma",
        "conversion",
        "base",
        "chikou",
        "current_kumo_a",
        "current_kumo_b",
        "future_kumo_a",
        "future_kumo_b",
        "max_level",
        "high_level",
        "low_level",
        "min_level",
    ];

    const RANGE_FILTERED_TREND_SIGNALS_OUTPUT_IDS: [&str; 13] = [
        "kalman",
        "supertrend",
        "upper_band",
        "lower_band",
        "trend",
        "kalman_trend",
        "state",
        "market_trending",
        "market_ranging",
        "short_term_bullish",
        "short_term_bearish",
        "long_term_bullish",
        "long_term_bearish",
    ];

    const ICT_PROPULSION_BLOCK_OUTPUT_IDS: [&str; 12] = [
        "bullish_high",
        "bullish_low",
        "bullish_kind",
        "bullish_active",
        "bullish_mitigated",
        "bullish_new",
        "bearish_high",
        "bearish_low",
        "bearish_kind",
        "bearish_active",
        "bearish_mitigated",
        "bearish_new",
    ];

    const VDUBUS_DIVERGENCE_WAVE_PATTERN_GENERATOR_OUTPUT_IDS: [&str; 12] = [
        "fast_standard",
        "fast_climax",
        "fast_rounded",
        "fast_predator",
        "slow_standard",
        "slow_climax",
        "slow_rounded",
        "slow_predator",
        "opposing_force",
        "macd",
        "signal",
        "hist",
    ];

    const KASE_PEAK_OSCILLATOR_WITH_DIVERGENCES_OUTPUT_IDS: [&str; 10] = [
        "oscillator",
        "max_peak_value",
        "min_peak_value",
        "market_extreme",
        "regular_bullish",
        "hidden_bullish",
        "regular_bearish",
        "hidden_bearish",
        "go_long",
        "go_short",
    ];

    const ADAPTIVE_BOUNDS_RSI_OUTPUT_IDS: [&str; 10] = [
        "rsi",
        "lower",
        "lower_mid",
        "middle",
        "upper_mid",
        "upper",
        "regime",
        "regime_flip",
        "lower_signal",
        "upper_signal",
    ];

    const ADJUSTABLE_MA_ALTERNATING_EXTREMITIES_OUTPUT_IDS: [&str; 10] = [
        "ma",
        "upper",
        "lower",
        "extremity",
        "state",
        "changed",
        "smoothed_open",
        "smoothed_high",
        "smoothed_low",
        "smoothed_close",
    ];

    const ALLIGATOR_OUTPUT_IDS: [&str; 3] = ["jaw", "teeth", "lips"];
    const ANDEAN_OSCILLATOR_OUTPUT_IDS: [&str; 3] = ["bull", "bear", "signal"];
    const AROON_OUTPUT_IDS: [&str; 2] = ["up", "down"];
    const ASO_OUTPUT_IDS: [&str; 2] = ["bulls", "bears"];
    const AUTOCORRELATION_INDICATOR_OUTPUT_IDS: [&str; 2] = ["filtered", "correlation"];
    const AVSL_OUTPUT_IDS: [&str; 1] = ["value"];
    const BANDPASS_OUTPUT_IDS: [&str; 4] = ["bp", "bp_normalized", "signal", "trigger"];
    const BOLLINGER_BANDS_OUTPUT_IDS: [&str; 3] = ["upper", "middle", "lower"];
    const BUFF_AVERAGES_OUTPUT_IDS: [&str; 2] = ["fast", "slow"];
    const CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS: [&str; 6] = [
        "strength",
        "highs",
        "lows",
        "mid",
        "long_signal",
        "short_signal",
    ];
    const CHANDELIER_EXIT_OUTPUT_IDS: [&str; 2] = ["long_stop", "short_stop"];
    const CKSP_OUTPUT_IDS: [&str; 2] = ["long_values", "short_values"];
    const CORRELATION_CYCLE_OUTPUT_IDS: [&str; 4] = ["real", "imag", "angle", "state"];

    const ALPHATREND_OUTPUT_IDS: [&str; 2] = ["k1", "k2"];

    const BULLS_V_BEARS_OUTPUT_IDS: [&str; 10] = [
        "value",
        "bull",
        "bear",
        "ma",
        "upper",
        "lower",
        "bullish_signal",
        "bearish_signal",
        "zero_cross_up",
        "zero_cross_down",
    ];

    const RANGE_OSCILLATOR_OUTPUT_IDS: [&str; 9] = [
        "oscillator",
        "ma",
        "upper_band",
        "lower_band",
        "range_width",
        "in_range",
        "trend",
        "break_up",
        "break_down",
    ];

    const PIVOT_OUTPUT_IDS: [&str; 9] = ["r4", "r3", "r2", "r1", "pp", "s1", "s2", "s3", "s4"];

    const ACOSC_OUTPUT_IDS: [&str; 2] = ["osc", "change"];

    /// Extend the canonical broker-captured EURUSD M1 trendbar fixture without
    /// inventing ticks or resampling it. Bandpass's largest admitted tuple
    /// needs 2,667 source bars before its exact high-pass stage can produce;
    /// cycling the captured bar values gives the parity oracle enough history
    /// while every price/volume observation still comes from the checked-in
    /// cTrader trendbar authority.
    fn repeated_ctrader_bandpass_fixture() -> Ohlcv {
        const BARS: usize = 3_200;
        let seed = crate::test_fixtures::ctrader_sample_ohlcv();
        let repeat = |values: &[f64]| {
            values
                .iter()
                .copied()
                .cycle()
                .take(BARS)
                .collect::<Vec<_>>()
        };
        Ohlcv {
            timestamp: None,
            open: repeat(&seed.open),
            high: repeat(&seed.high),
            low: repeat(&seed.low),
            close: repeat(&seed.close),
            volume: seed.volume.as_deref().map(repeat),
        }
    }

    fn range_oscillator_parity_fixture() -> Ohlcv {
        const BARS: usize = 451;
        const GAP: usize = 225;
        let mut open = Vec::with_capacity(BARS);
        let mut high = Vec::with_capacity(BARS);
        let mut low = Vec::with_capacity(BARS);
        let mut close = Vec::with_capacity(BARS);
        let mut volume = Vec::with_capacity(BARS);
        let changes = [0.0, 1.75, -0.875, 2.25, -1.625, 0.5, -0.25, 1.125];
        for index in 0..BARS {
            let close_value = 100.0 + index as f64 * 0.015625 + changes[index % changes.len()];
            open.push(close_value + (index % 5) as f64 * 0.03125 - 0.0625);
            high.push(close_value + 2.0 + (index % 3) as f64 * 0.0625);
            low.push(close_value - 2.0 - (index % 4) as f64 * 0.046875);
            close.push(close_value);
            volume.push(1_000.0 + (index % 17) as f64 * 29.0);
        }
        open[GAP] = f64::NAN;
        high[GAP] = f64::NAN;
        low[GAP] = f64::NAN;
        close[GAP] = f64::NAN;
        volume[GAP] = f64::NAN;

        Ohlcv {
            timestamp: None,
            open,
            high,
            low,
            close,
            volume: Some(volume),
        }
    }

    fn trend_flow_trail_parity_fixture() -> Ohlcv {
        let mut open = Vec::with_capacity(220);
        let mut high = Vec::with_capacity(220);
        let mut low = Vec::with_capacity(220);
        let mut close = Vec::with_capacity(220);
        let mut volume = Vec::with_capacity(220);
        let swings = [0.0, 1.4, -0.9, 2.1, -1.7, 0.8, -0.4, 1.2];
        for index in 0..220 {
            let close_value = 100.0 + index as f64 * 0.03125 + swings[index % swings.len()];
            let open_value = close_value + (index % 5) as f64 * 0.07 - 0.14;
            open.push(open_value);
            high.push(open_value.max(close_value) + 0.6 + (index % 3) as f64 * 0.03);
            low.push(open_value.min(close_value) - 0.5 - (index % 4) as f64 * 0.02);
            close.push(close_value);
            volume.push(900.0 + (index % 17) as f64 * 37.0);
        }

        for index in [0, 1] {
            open[index] = f64::NAN;
            high[index] = f64::NAN;
            low[index] = f64::NAN;
            close[index] = f64::NAN;
            volume[index] = f64::NAN;
        }
        // Two distinct mid-series reset causes. Each surrounding finite run is
        // long enough for the largest tuple, so neither lane may skip recovery.
        open[104] = f64::INFINITY;
        volume[105] = f64::NAN;
        // After the reset, force a zero-range segment. Alpha length 1 then
        // produces coincident upper/lower bands, the adversarial case where
        // comparing the previous trail to the previous upper band is not
        // interchangeable with guessing from a separately stored direction.
        for index in 106..132 {
            open[index] = close[index];
            high[index] = close[index];
            low[index] = close[index];
        }

        Ohlcv {
            timestamp: None,
            open,
            high,
            low,
            close,
            volume: Some(volume),
        }
    }

    fn market_structure_confluence_parity_fixture() -> Ohlcv {
        let mut open = Vec::with_capacity(260);
        let mut high = Vec::with_capacity(260);
        let mut low = Vec::with_capacity(260);
        let mut close = Vec::with_capacity(260);
        let mut volume = Vec::with_capacity(260);
        let impulses = [0.0, 1.7, -1.2, 2.4, -2.1, 0.8, -0.5, 1.3, -1.8, 0.4];
        for index in 0..260 {
            let center = 100.0
                + index as f64 * 0.021875
                + (index as f64 * 0.173).sin() * 2.1
                + impulses[index % impulses.len()];
            let open_value = center + (index % 7) as f64 * 0.06 - 0.18;
            let close_value = center - (index % 5) as f64 * 0.04 + 0.08;
            open.push(open_value);
            high.push(open_value.max(close_value) + 0.7 + (index % 4) as f64 * 0.09);
            low.push(open_value.min(close_value) - 0.6 - (index % 6) as f64 * 0.07);
            close.push(close_value);
            volume.push(1_000.0 + (index % 23) as f64 * 29.0);
        }
        // A finite close/low with a non-finite high exercises Rust f64::max
        // versus CUDA fmax and invalidates exactly the pivot windows that see
        // it without poisoning the WMA basis indefinitely.
        high[137] = f64::NAN;

        Ohlcv {
            timestamp: None,
            open,
            high,
            low,
            close,
            volume: Some(volume),
        }
    }

    fn hema_trend_levels_parity_fixture() -> Ohlcv {
        let mut open = Vec::with_capacity(280);
        let mut high = Vec::with_capacity(280);
        let mut low = Vec::with_capacity(280);
        let mut close = Vec::with_capacity(280);
        let mut volume = Vec::with_capacity(280);
        let impulses = [0.0, 1.8, -1.1, 2.3, -2.0, 0.7, -0.4, 1.4];
        for index in 0..280 {
            let segment = match index {
                0..=69 => -(index as f64) * 0.035,
                70..=139 => (index as f64 - 70.0) * 0.082 - 2.45,
                140..=209 => -(index as f64 - 140.0) * 0.091 + 3.2,
                _ => (index as f64 - 210.0) * 0.067 - 3.1,
            };
            let center = 100.0
                + segment
                + (index as f64 * 0.117).sin() * 0.9
                + impulses[index % impulses.len()];
            let open_value = center + (index % 5) as f64 * 0.11 - 0.22;
            let close_value = center - (index % 7) as f64 * 0.07 + 0.21;
            open.push(open_value);
            high.push(open_value.max(close_value) + 0.8 + (index % 4) as f64 * 0.05);
            low.push(open_value.min(close_value) - 0.7 - (index % 3) as f64 * 0.06);
            close.push(close_value);
            volume.push(1_100.0 + (index % 19) as f64 * 31.0);
        }
        // Both invalid forms are part of the scalar contract: every HEMA,
        // ATR and box state resets, and no box/test may leak across the gap.
        high[143] = f64::NAN;
        open[207] = f64::INFINITY;

        Ohlcv {
            timestamp: None,
            open,
            high,
            low,
            close,
            volume: Some(volume),
        }
    }

    fn ichimoku_oscillator_parity_fixture() -> Ohlcv {
        let mut open = Vec::with_capacity(260);
        let mut high = Vec::with_capacity(260);
        let mut low = Vec::with_capacity(260);
        let mut close = Vec::with_capacity(260);
        let mut volume = Vec::with_capacity(260);
        let impulses = [0.0, 1.3, -0.8, 2.0, -1.7, 0.6, -0.3, 1.1, -1.4];
        for index in 0..260 {
            let center = 100.0
                + index as f64 * 0.01875
                + (index as f64 * 0.091).sin() * 2.4
                + impulses[index % impulses.len()];
            let open_value = center + (index % 6) as f64 * 0.08 - 0.2;
            let close_value = center - (index % 5) as f64 * 0.09 + 0.16;
            open.push(open_value);
            high.push(open_value.max(close_value) + 0.75 + (index % 4) as f64 * 0.07);
            low.push(open_value.min(close_value) - 0.65 - (index % 3) as f64 * 0.05);
            close.push(close_value);
            volume.push(1_000.0 + (index % 17) as f64 * 41.0);
        }
        // Admission must reject this leading infinity and start at bar 1;
        // a non-NaN scan would incorrectly start the rolling windows at 0.
        high[0] = f64::INFINITY;
        // The scalar rolling-midpoint contract skips these bars without
        // clearing its monotonic deques; the CUDA path must reproduce that
        // per-output validity and state exactly.
        high[113] = f64::NAN;
        low[172] = f64::INFINITY;

        Ohlcv {
            timestamp: None,
            open,
            high,
            low,
            close,
            volume: Some(volume),
        }
    }

    fn range_filtered_trend_signals_parity_fixture() -> Ohlcv {
        let mut open = Vec::with_capacity(430);
        let mut high = Vec::with_capacity(430);
        let mut low = Vec::with_capacity(430);
        let mut close = Vec::with_capacity(430);
        let mut volume = Vec::with_capacity(430);
        let impulses = [0.0, 2.2, -1.8, 3.1, -2.7, 1.3, -0.6, 1.9, -2.4];
        for index in 0..430 {
            let segment = match index {
                0..=109 => index as f64 * 0.061,
                110..=219 => 6.7 - (index as f64 - 110.0) * 0.084,
                220..=329 => -2.5 + (index as f64 - 220.0) * 0.097,
                _ => 8.2 - (index as f64 - 330.0) * 0.073,
            };
            let center = 100.0
                + segment
                + (index as f64 * 0.137).sin() * 1.4
                + impulses[index % impulses.len()];
            let close_value = center + (index % 5) as f64 * 0.07 - 0.14;
            open.push(close_value - 0.05);
            high.push(close_value + 0.8 + (index % 4) as f64 * 0.11);
            low.push(close_value - 0.7 - (index % 3) as f64 * 0.09);
            close.push(close_value);
            volume.push(1_000.0 + (index % 23) as f64 * 31.0);
        }
        // Reset every carried recurrence, then leave more than 200 finite bars
        // so the CUDA lane must prove recovery instead of comparing only the
        // first run.
        high[101] = f64::NAN;
        // A flat zero-range block after the reset is the adversarial case in
        // which Supertrend's previous output cannot be replaced by a guessed
        // previous direction.
        let flat = close[239];
        for index in 240..271 {
            open[index] = flat;
            high[index] = flat;
            low[index] = flat;
            close[index] = flat;
        }

        Ohlcv {
            timestamp: None,
            open,
            high,
            low,
            close,
            volume: Some(volume),
        }
    }

    #[test]
    fn fisher_v2_finite_midpoint_admission_source_is_closed() {
        const SOURCE: &str = include_str!("gpu_indicators.rs");
        let high = [f64::INFINITY, f64::MAX, 1.0];
        let low = [0.0, f64::MAX, 1.0];
        let finite_pair_first = high
            .iter()
            .zip(&low)
            .position(|(high, low)| high.is_finite() && low.is_finite())
            .unwrap();
        let finite_midpoint_first = high
            .iter()
            .zip(&low)
            .position(|(high, low)| ((*high + *low) / 2.0).is_finite())
            .unwrap();
        assert_eq!(finite_pair_first, 1);
        assert_eq!(finite_midpoint_first, 2);

        let first_valid = SOURCE
            .split("fn first_valid_for(")
            .nth(1)
            .expect("first-valid resolver remains present")
            .split("pub fn resolve_device_route")
            .next()
            .unwrap();
        assert!(
            first_valid.contains(
                "F64FirstValidRule::HighLowMidpointFinite => self.first_valid_hl2_finite,"
            )
        );
        let fisher = SOURCE
            .split("pub fn compute_fisher_outputs_device(")
            .nth(1)
            .expect("Fisher resident route remains present")
            .split("/// Launch FBEO's canonical")
            .next()
            .unwrap();
        assert!(
            fisher.contains(".fisher_all_outputs(high_low, self.first_valid_hl2_finite, periods)")
        );
    }

    /// The table is only trustworthy if every id in it is real and
    /// single-output. This runs WITHOUT a card — it is a registry query, not a
    /// launch — so it catches a bad table entry at `cargo test` on the build
    /// host instead of mid-run on a rented box.
    #[test]
    fn every_spec_is_a_real_single_output_indicator() {
        for spec in GPU_SWEEP_SPECS {
            let info = get_indicator(spec.id)
                .unwrap_or_else(|| panic!("{}: not in the vector-ta registry", spec.id));
            assert!(
                info.outputs.len() <= 1,
                "{}: {} outputs — multi-output indicators are not in the single-output device \
                 sweep contract",
                spec.id,
                info.outputs.len()
            );
        }
    }

    /// THE precision gate, and it needs no card: every indicator this lane
    /// claims must have a real f64 kernel in vector-ta, and this table's idea
    /// of which price series it needs must match vector-ta's. A mismatch would
    /// either silently serve an f32 kernel or silently compute a different
    /// indicator.
    #[test]
    fn every_spec_has_an_f64_kernel_with_a_matching_input_contract() {
        for spec in GPU_SWEEP_SPECS {
            let declared = f64_kernel_for(spec.id).unwrap_or_else(|| {
                panic!(
                    "{}: no f64 CUDA kernel in vector-ta — this lane must not claim it, because \
                     it will never fall back to the f32 kernel",
                    spec.id
                )
            });
            assert!(
                declared.kernel.entry_point().ends_with("_f64"),
                "{}: resolves to {}, which is not an f64 entry point",
                spec.id,
                declared.kernel.entry_point()
            );
            assert_eq!(
                DeviceInput::from_vector_ta(declared.input),
                spec.input,
                "{}: this table and vector-ta's f64 kernel table disagree about the input series",
                spec.id
            );
        }
    }

    /// `hpc_ta` maps a requested TSI period to the named window pair
    /// `(long_period = period, short_period = round(period * 13 / 25))`.
    /// The CUDA row therefore has to consume the requested anchor.  The old
    /// invariant declaration discarded it and returned 25/13 for every row.
    #[test]
    fn tsi_cuda_contract_consumes_the_coupled_window_anchor() {
        let tsi = f64_kernel_for("tsi").expect("TSI must have an f64 CUDA kernel");
        assert!(
            !tsi.kernel.is_period_invariant(),
            "TSI is coupled-window swept in production; an invariant CUDA row silently ignores \
             long_period/short_period and computes a different feature"
        );
    }

    /// THE ANTI-ROT ASSERTION, and the reason no count appears in this file's
    /// prose any more.
    ///
    /// `GPU_SWEEP_SPECS` is the intersection of three sets maintained in three
    /// different places, and the failure mode this project keeps repeating is
    /// that one of them grows and the intersection does not — the kernel gets
    /// written, compiled, registered upstream, and then computed on the CPU
    /// anyway because nobody added the row.
    ///
    /// So the relationship is asserted in BOTH directions:
    ///
    /// * every id in `MULTI_PERIOD_IDS` that is single-output AND has an f64
    ///   kernel MUST be claimed — this is the direction that catches a
    ///   newly-registered kernel still being run on the CPU;
    /// * every id claimed MUST be in `MULTI_PERIOD_IDS` — this is the direction
    ///   that catches a row for an indicator no caller ever sweeps, which would
    ///   be dead weight the parity test still pays for.
    ///
    /// Registry queries and table lookups only. No card, no launch — this fails
    /// at `cargo test --features gpu-cuda` on the build host, not mid-run on a
    /// rented box.
    #[test]
    fn every_reachable_multi_period_id_with_an_f64_kernel_is_claimed() {
        use crate::core::hpc_ta::MULTI_PERIOD_IDS;

        let mut should_be_claimed: Vec<&str> = Vec::new();
        let mut multi_output: Vec<&str> = Vec::new();
        let mut no_kernel: Vec<&str> = Vec::new();

        for &id in MULTI_PERIOD_IDS.iter() {
            let info = get_indicator(id).unwrap_or_else(|| {
                panic!(
                    "{id}: in MULTI_PERIOD_IDS but not in the vector-ta \
                                           registry — the CPU sweep cannot compute it either"
                )
            });
            if info.outputs.len() > 1 {
                // Emits ZERO columns on EITHER lane (`hpc_ta.rs:291` swallows
                // the `InvalidParam` from `output_id: None`), so it is not a
                // device gap.
                multi_output.push(id);
                continue;
            }
            if f64_kernel_for(id).is_none() {
                no_kernel.push(id);
                continue;
            }
            should_be_claimed.push(id);
        }

        let claimed: Vec<&str> = GPU_SWEEP_SPECS.iter().map(|s| s.id).collect();

        let missing: Vec<&str> = should_be_claimed
            .iter()
            .copied()
            .filter(|id| !claimed.contains(id))
            .collect();
        assert!(
            missing.is_empty(),
            "these ids are period-swept by hpc_ta, are single-output, and vector-ta HAS an f64 \
             kernel for them — but GPU_SWEEP_SPECS does not claim them, so a card is present, a \
             working registered kernel exists, and hpc_ta computes them on the CPU anyway: {missing:?}\n\
             (multi-output, correctly not claimed: {multi_output:?}; no f64 kernel in vector-ta: \
             {no_kernel:?})\n\
             Add a GpuSweepSpec row. Do NOT add a justification comment instead."
        );

        let unswept: Vec<&str> = claimed
            .iter()
            .copied()
            .filter(|id| !MULTI_PERIOD_IDS.contains(id))
            .collect();
        assert!(
            unswept.is_empty(),
            "GPU_SWEEP_SPECS claims ids that hpc_ta never period-sweeps, so nothing ever launches \
             them: {unswept:?}. Either add them to MULTI_PERIOD_IDS or drop the rows. (`wilders` \
             is the live example of an f64 kernel this crate deliberately does NOT claim: it is \
             not in MULTI_PERIOD_IDS.)"
        );
    }

    /// The production GPU plan is driven by vector-ta's complete f64 registry,
    /// not by the historical windowed sweep table. `wilders` is the
    /// regression fixture: its real f64 kernel has long been registered but it
    /// never appeared in `GPU_SWEEP_SPECS` because that table only described
    /// one old period sweep.
    #[test]
    fn registered_f64_kernel_is_reachable_without_a_legacy_table_row() {
        assert!(
            spec_for("wilders").is_none(),
            "the fixture must remain outside the legacy historical sweep table"
        );
        let route = f64_primary_device_route_for("wilders")
            .expect("every registered real f64 kernel needs a generic device route");
        assert_eq!(route.indicator_id, "wilders");
        assert_eq!(route.input, DeviceInput::CloseFromOhlcv);
        assert!(route.entry_point.ends_with("_f64"));
    }

    /// A successful launch is not enough for a multi-output indicator: the
    /// resident matrix must identify the exact feature column it contains.
    /// Without this mapping a `keltner` launch, for example, could be labelled
    /// as any of its bands and silently duplicate or omit another output.
    #[test]
    fn every_registered_f64_primary_has_an_explicit_feature_output_identity() {
        use vector_ta::indicators::dispatch::cuda_f64::F64_KERNELS;

        let missing: Vec<&str> = F64_KERNELS
            .iter()
            .filter(|spec| f64_primary_output_id_for(spec.indicator_id).is_none())
            .map(|spec| spec.indicator_id)
            .collect();

        assert!(
            missing.is_empty(),
            "registered f64 primaries without an exact output identity: {missing:?}"
        );
    }

    /// Strict admission must see the complete output graph before any CUDA
    /// context or allocation exists. Primary-kernel reachability may reduce
    /// this list, but it may never hide additional/discrete outputs.
    #[test]
    fn gpu_only_preflight_names_current_output_level_gaps_without_launching() {
        let gaps = gpu_only_classic_ta_output_gaps();
        let missing_kernels = gaps
            .iter()
            .filter(|gap| gap.reason == GpuOnlyOutputGapReason::MissingF64Kernel)
            .count();
        let missing_named_outputs = gaps
            .iter()
            .filter(|gap| gap.reason == GpuOnlyOutputGapReason::MissingNamedOutputRoute)
            .count();
        let missing_discrete = gaps
            .iter()
            .filter(|gap| gap.reason == GpuOnlyOutputGapReason::MissingDiscreteMatrixRoute)
            .count();
        eprintln!(
            "NEOETHOS_GPU_ONLY_OUTPUT_PREFLIGHT total={} missing_f64_kernel={} \
             missing_named_output_route={} missing_discrete_matrix_route={} \
             registered_primary_candidates={}",
            gaps.len(),
            missing_kernels,
            missing_named_outputs,
            missing_discrete,
            registered_f64_primary_route_count(),
        );
        assert!(
            !gaps.is_empty(),
            "the full output graph is not complete yet"
        );
        assert!(gaps.iter().any(|gap| {
            gap.indicator_id == "pattern_recognition"
                && gap.reason == GpuOnlyOutputGapReason::MissingDiscreteMatrixRoute
        }));
        assert!(gaps.iter().any(|gap| {
            gap.indicator_id == "keltner"
                && gap.reason == GpuOnlyOutputGapReason::MissingNamedOutputRoute
        }));
        assert!(gaps.iter().any(|gap| {
            gap.indicator_id == "ma" && gap.reason == GpuOnlyOutputGapReason::MissingF64Kernel
        }));

        let keltner_primary =
            f64_primary_output_id_for("keltner").expect("keltner primary is registered");
        assert!(
            !gaps
                .iter()
                .any(|gap| { gap.indicator_id == "keltner" && gap.output_id == keltner_primary })
        );

        for output_id in ["oscillator", "signal", "histogram"] {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "absolute_strength_index_oscillator"
                        && gap.output_id == output_id
                }),
                "the proven resident ASI route is still reported missing for {output_id}"
            );
        }
        for output_id in ["amo", "ama"] {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "adaptive_momentum_oscillator" && gap.output_id == output_id
                }),
                "the proven resident AMO route is still reported missing for {output_id}"
            );
        }
        for output_id in ["macd", "signal", "hist"] {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "adaptive_macd" && gap.output_id == output_id
                }),
                "Adaptive MACD must keep every exact f64 output resident; {output_id} is still a \
                 strict-GPU gap"
            );
        }
        for output_id in ["cg", "trigger"] {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "ehlers_adaptive_cg" && gap.output_id == output_id
                }),
                "the proven resident Ehlers Adaptive CG route is still reported missing for \
                 {output_id}"
            );
        }
        for output_id in ["cycle", "trigger"] {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "ehlers_adaptive_cyber_cycle" && gap.output_id == output_id
                }),
                "the proven resident Ehlers Adaptive Cyber Cycle route is still reported \
                 missing for {output_id}"
            );
        }
        for output_id in ["cycle", "trigger"] {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "ehlers_simple_cycle_indicator"
                        && gap.output_id == output_id
                }),
                "the proven resident Ehlers Simple Cycle route is still reported missing for \
                {output_id}"
            );
        }
        for output_id in ["predict", "trigger"] {
            assert!(
                !gaps
                    .iter()
                    .any(|gap| { gap.indicator_id == "ehlers_pma" && gap.output_id == output_id }),
                "the proven resident Ehlers PMA route is still reported missing for {output_id}"
            );
        }
        for output_id in [
            "middle",
            "trend",
            "upper_0618",
            "upper_1000",
            "upper_1618",
            "upper_2618",
            "lower_0618",
            "lower_1000",
            "lower_1618",
            "lower_2618",
            "tp_long_band",
            "tp_short_band",
            "go_long",
            "go_short",
            "rejection_long",
            "rejection_short",
            "long_bounce",
            "short_bounce",
        ] {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "fibonacci_entry_bands" && gap.output_id == output_id
                }),
                "the proven resident Fibonacci Entry Bands route is still reported missing for \
                 {output_id}"
            );
        }
        for output_id in ["in_phase", "lead"] {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "adaptive_bandpass_trigger_oscillator"
                        && gap.output_id == output_id
                }),
                "the proven resident adaptive-bandpass route is still reported missing for \
                 {output_id}"
            );
        }
        for output_id in TREND_FLOW_TRAIL_OUTPUT_IDS {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "trend_flow_trail" && gap.output_id == output_id
                }),
                "the proven resident Trend Flow Trail route is still reported missing for \
                 {output_id}"
            );
        }
        for output_id in MARKET_STRUCTURE_CONFLUENCE_OUTPUT_IDS {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "market_structure_confluence" && gap.output_id == output_id
                }),
                "the proven resident Market Structure Confluence route is still reported missing \
                 for {output_id}"
            );
        }
        for output_id in HEMA_TREND_LEVELS_OUTPUT_IDS {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "hema_trend_levels" && gap.output_id == output_id
                }),
                "the proven resident HEMA Trend Levels route is still reported missing for \
                 {output_id}"
            );
        }
        for output_id in ICHIMOKU_OSCILLATOR_OUTPUT_IDS {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "ichimoku_oscillator" && gap.output_id == output_id
                }),
                "the proven resident Ichimoku Oscillator route is still reported missing for \
                 {output_id}"
            );
        }
        for output_id in RANGE_FILTERED_TREND_SIGNALS_OUTPUT_IDS {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "range_filtered_trend_signals" && gap.output_id == output_id
                }),
                "the proven resident Range Filtered Trend Signals route is still reported missing \
                 for {output_id}"
            );
        }
        for output_id in ADJUSTABLE_MA_ALTERNATING_EXTREMITIES_OUTPUT_IDS {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "adjustable_ma_alternating_extremities"
                        && gap.output_id == output_id
                }),
                "the proven resident Adjustable MA & Alternating Extremities route is still \
                 reported missing for {output_id}"
            );
        }
        for output_id in BULLS_V_BEARS_OUTPUT_IDS {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "bulls_v_bears" && gap.output_id == output_id
                }),
                "the proven resident Bulls v Bears route is still reported missing for \
                 {output_id}"
            );
        }
        for output_id in RANGE_OSCILLATOR_OUTPUT_IDS {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "range_oscillator" && gap.output_id == output_id
                }),
                "the proven resident Range Oscillator route is still reported missing for \
                 {output_id}"
            );
        }
        for output_id in PIVOT_OUTPUT_IDS {
            assert!(
                !gaps
                    .iter()
                    .any(|gap| { gap.indicator_id == "pivot" && gap.output_id == output_id }),
                "the proven resident Pivot route is still reported missing for {output_id}"
            );
        }
        for output_id in ACOSC_OUTPUT_IDS {
            assert!(
                !gaps
                    .iter()
                    .any(|gap| { gap.indicator_id == "acosc" && gap.output_id == output_id }),
                "the proven resident ACOSC route is still reported missing for {output_id}"
            );
        }
        for output_id in ADAPTIVE_BOUNDS_RSI_OUTPUT_IDS {
            assert!(
                !gaps.iter().any(|gap| {
                    gap.indicator_id == "adaptive_bounds_rsi" && gap.output_id == output_id
                }),
                "the proven resident Adaptive Bounds RSI route is still reported missing for \
                 {output_id}"
            );
        }

        let mut identities: Vec<(&str, &str)> = gaps
            .iter()
            .map(|gap| (gap.indicator_id, gap.output_id))
            .collect();
        let before = identities.len();
        identities.sort_unstable();
        identities.dedup();
        assert_eq!(identities.len(), before, "duplicate preflight gap identity");
    }

    /// Explicit audit command for selecting the next complete resident family
    /// without guessing from registry size. Ignored in normal suites because
    /// its value is the full line-oriented manifest, not another assertion.
    #[test]
    #[ignore = "run explicitly with --ignored --nocapture to print the GpuOnly gap manifest"]
    fn gpu_only_preflight_prints_complete_gap_manifest() {
        for gap in gpu_only_classic_ta_output_gaps() {
            eprintln!(
                "NEOETHOS_GPU_ONLY_GAP\t{}\t{}\t{}",
                gap.indicator_id, gap.output_id, gap.reason
            );
        }
    }

    /// A route is not GPU coverage until a real registered kernel leaves its
    /// result resident on the selected card. This fixture deliberately uses an
    /// id outside `GPU_SWEEP_SPECS`, so the legacy windowed path cannot
    /// satisfy the assertion accidentally.
    #[test]
    fn registered_f64_kernel_launches_to_resident_device_memory() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card test requires a launchable CUDA device 0");
        let output = engine
            .compute_primary_device("wilders", &[7, 21])
            .expect("wilders must launch through the generic f64 device route");

        assert_eq!(output.indicator_id, "wilders");
        assert_eq!(
            output.output_id,
            f64_primary_output_id_for("wilders").expect("wilders output identity")
        );
        assert_eq!(output.rows, 2);
        assert_eq!(output.cols, ohlcv.len());
        assert!(output.entry_point.ends_with("_f64"));
        assert!(
            matches!(output.series, IndicatorCudaSeriesF64::DeviceF64(_)),
            "GpuOnly must not materialize an intermediate host Vec"
        );
        engine
            .synchronize()
            .expect("the real kernel must finish successfully");
    }

    /// A resident pointer is only reusable when its uploader and every
    /// consumer belong to the exact same CUDA context and stream contract.
    /// Matching only the runtime ordinal is insufficient: two independently
    /// created sessions on device 0 can still introduce synchronization and
    /// ownership boundaries that make the advertised no-bounce graph false.
    #[test]
    fn uploads_and_f64_kernels_borrow_one_cuda_session() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card session test requires a launchable CUDA device 0");

        assert_eq!(
            engine.runtime.session_identity(),
            engine.f64_engine.session_identity(),
            "GpuOnly must upload and launch through one exact CUDA context/stream session"
        );
    }

    #[test]
    fn volume_dependent_gpu_plan_rejects_missing_volume_instead_of_fabricating_zeroes() {
        let mut ohlcv = trend_flow_trail_parity_fixture();
        ohlcv.volume = None;

        let error = match GpuIndicatorEngine::new(&ohlcv, 0) {
            Ok(_) => panic!("a volume-dependent resident feature plan must reject missing volume"),
            Err(error) => error,
        };
        assert!(
            format!("{error:#}").contains("volume"),
            "missing-volume failure must name the absent broker/data field: {error:#}"
        );
    }

    /// The ASI oscillator is the first repaired multi-output family. Its
    /// existing CUDA translation unit already computes all three f64 series;
    /// the production route must launch that kernel against the frame's
    /// resident close buffer instead of constructing the legacy host-upload
    /// wrapper or duplicating the formula.
    #[test]
    fn absolute_strength_index_oscillator_all_outputs_stay_resident() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card multi-output test requires a launchable CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_absolute_strength_index_oscillator_outputs_device(&[(21, 34), (42, 68)])
            .expect("all ASI outputs must launch through the shared resident session");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "absolute_strength_index_oscillator");
        assert_eq!(
            result.entry_point,
            "absolute_strength_index_oscillator_batch_f64"
        );
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ["oscillator", "signal", "histogram"]
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 2
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("all three resident outputs must retire successfully");
    }

    /// This is a CPU/GPU parity check, not an independent proof that the CPU
    /// formula is mathematically authoritative. It prevents the resident
    /// multi-output route from swapping columns, dropping a reset, collapsing
    /// a real parameter tuple into one `period`, or changing f64 arithmetic.
    #[test]
    fn absolute_strength_index_oscillator_all_outputs_match_cpu_bits() {
        use vector_ta::indicators::absolute_strength_index_oscillator::{
            AbsoluteStrengthIndexOscillatorInput, AbsoluteStrengthIndexOscillatorParams,
            absolute_strength_index_oscillator,
        };

        let close = vec![
            f64::NAN,
            0.0,
            -0.0,
            1.0e-300,
            1.0e-300,
            -1.0e-300,
            1.0,
            2.0,
            2.0,
            0.0,
            4.0,
            f64::INFINITY,
            8.0,
            8.0,
            4.0,
            f64::NEG_INFINITY,
            1.0e150,
            1.0e150,
            1.0e-150,
            3.0,
            f64::NAN,
            5.0,
            2.5,
            2.5,
            10.0,
        ];
        let ohlcv = Ohlcv {
            timestamp: None,
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            close: close.clone(),
            volume: Some(vec![0.0; close.len()]),
        };
        let parameter_tuples = [(2, 2), (3, 4), (21, 34)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the parity test requires a launchable CUDA device 0");
        let result = engine
            .compute_absolute_strength_index_oscillator_outputs_device(&parameter_tuples)
            .expect("the resident ASI route must accept exact parameter tuples");

        let mut expected_oscillator = Vec::with_capacity(result.rows * result.cols);
        let mut expected_signal = Vec::with_capacity(result.rows * result.cols);
        let mut expected_histogram = Vec::with_capacity(result.rows * result.cols);
        for &(ema_length, signal_length) in &parameter_tuples {
            let input = AbsoluteStrengthIndexOscillatorInput::from_slice(
                &close,
                AbsoluteStrengthIndexOscillatorParams {
                    ema_length: Some(ema_length),
                    signal_length: Some(signal_length),
                },
            );
            let expected = absolute_strength_index_oscillator(&input)
                .expect("the CPU parity reference must accept the same tuple");
            expected_oscillator.extend(expected.oscillator);
            expected_signal.extend(expected.signal);
            expected_histogram.extend(expected.histogram);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident output must download for parity inspection");
            let expected = match output.output_id {
                "oscillator" => &expected_oscillator,
                "signal" => &expected_signal,
                "histogram" => &expected_histogram,
                unexpected => panic!("unexpected ASI output identity {unexpected}"),
            };
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU reset/undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] differs: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    /// Adaptive MACD's signal state is part of the same recurrence as the
    /// primary MACD line. GpuOnly must therefore launch all three matrices in
    /// one resident CUDA call; reconstructing signal/hist on the host would
    /// reintroduce the CPU/GPU ping-pong this execution mode forbids.
    #[test]
    fn adaptive_macd_all_outputs_stay_resident() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card multi-output test requires a launchable CUDA device 0");
        let uploads_before = engine.uploads();
        let parameter_tuples = [(20, 10, 20, 9), (32, 8, 21, 5)];

        let result = engine
            .compute_adaptive_macd_outputs_device(&parameter_tuples)
            .expect("all Adaptive MACD outputs must launch through the resident session");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Adaptive MACD route re-uploaded close prices"
        );
        assert_eq!(result.indicator_id, "adaptive_macd");
        assert_eq!(result.entry_point, "adaptive_macd_neo_all_outputs_f64");
        assert_eq!(result.rows, parameter_tuples.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ["macd", "signal", "hist"]
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("all three resident Adaptive MACD outputs must retire successfully");
    }

    /// This establishes CPU/GPU parity for the current scalar state machine;
    /// the independent mathematical review remains a separate production
    /// promotion gate. NaN/Inf bars exercise the deliberately asymmetric
    /// correlation reset and held signal state.
    #[test]
    fn adaptive_macd_all_outputs_match_cpu_bits() {
        use vector_ta::indicators::adaptive_macd::{
            AdaptiveMacdInput, AdaptiveMacdParams, adaptive_macd,
        };

        let mut close = (0..96)
            .map(|index| {
                let x = index as f64;
                100.0 + x.mul_add(0.125, (x * 0.37).sin() * 4.0)
            })
            .collect::<Vec<_>>();
        close[0] = f64::NAN;
        close[3] = 0.0;
        close[4] = -0.0;
        close[18] = f64::INFINITY;
        close[19] = 99.25;
        close[37] = f64::NAN;
        close[38] = 101.5;
        close[61] = f64::NEG_INFINITY;
        close[62] = 102.75;
        close[79] = 1.0e-200;
        close[80] = -1.0e-200;

        let ohlcv = Ohlcv {
            timestamp: None,
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            close: close.clone(),
            volume: Some(vec![0.0; close.len()]),
        };
        let parameter_tuples = [(2, 2, 3, 2), (5, 3, 7, 4), (20, 10, 20, 9)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the parity test requires a launchable CUDA device 0");
        let result = engine
            .compute_adaptive_macd_outputs_device(&parameter_tuples)
            .expect("the resident Adaptive MACD route must accept exact parameter tuples");

        let mut expected_macd = Vec::with_capacity(result.rows * result.cols);
        let mut expected_signal = Vec::with_capacity(result.rows * result.cols);
        let mut expected_hist = Vec::with_capacity(result.rows * result.cols);
        for &(length, fast_period, slow_period, signal_period) in &parameter_tuples {
            let input = AdaptiveMacdInput::from_slice(
                &close,
                AdaptiveMacdParams {
                    length: Some(length),
                    fast_period: Some(fast_period),
                    slow_period: Some(slow_period),
                    signal_period: Some(signal_period),
                },
            );
            let expected = adaptive_macd(&input)
                .expect("the CPU parity reference must accept the same parameter tuple");
            expected_macd.extend(expected.macd);
            expected_signal.extend(expected.signal);
            expected_hist.extend(expected.hist);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Adaptive MACD output must download for parity inspection");
            let expected = match output.output_id {
                "macd" => &expected_macd,
                "signal" => &expected_signal,
                "hist" => &expected_hist,
                unexpected => panic!("unexpected Adaptive MACD output identity {unexpected}"),
            };
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] differs: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    /// Adaptive Momentum Oscillator already ships one CUDA kernel that writes
    /// both `amo` and `ama`. GpuOnly must connect both matrices to the same
    /// resident frame/session; the superseded wrapper's second context and
    /// host input upload are not an admissible route.
    #[test]
    fn adaptive_momentum_oscillator_all_outputs_stay_resident() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card multi-output test requires a launchable CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_adaptive_momentum_oscillator_outputs_device(&[(14, 9), (28, 7)])
            .expect("both AMO outputs must launch through the shared resident session");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "adaptive_momentum_oscillator");
        assert_eq!(result.entry_point, "adaptive_momentum_oscillator_batch_f64");
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ["amo", "ama"]
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 2
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("both resident AMO outputs must retire successfully");
    }

    /// CPU/GPU parity for both AMO columns. This checks the current scalar
    /// semantics only; the independent mathematical review remains a separate
    /// promotion gate.
    #[test]
    fn adaptive_momentum_oscillator_all_outputs_match_cpu_bits() {
        use vector_ta::indicators::adaptive_momentum_oscillator::{
            AdaptiveMomentumOscillatorInput, AdaptiveMomentumOscillatorParams,
            adaptive_momentum_oscillator,
        };

        let close = vec![
            10.0,
            10.0,
            9.0,
            11.0,
            8.0,
            12.0,
            f64::NAN,
            7.0,
            13.0,
            6.0,
            14.0,
            0.0,
            -0.0,
            1.0e-200,
            -1.0e-200,
            1.0e100,
            f64::INFINITY,
            5.0,
            15.0,
            4.0,
            16.0,
            f64::NEG_INFINITY,
            3.0,
            17.0,
            2.0,
            18.0,
            1.0,
            19.0,
            19.0,
            18.5,
            20.0,
            17.5,
            21.0,
            16.5,
            22.0,
            15.5,
            23.0,
            14.5,
            24.0,
            13.5,
        ];
        let ohlcv = Ohlcv {
            timestamp: None,
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            close: close.clone(),
            volume: Some(vec![0.0; close.len()]),
        };
        let parameter_tuples = [(3, 2), (5, 3), (14, 9)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the parity test requires a launchable CUDA device 0");
        let result = engine
            .compute_adaptive_momentum_oscillator_outputs_device(&parameter_tuples)
            .expect("the resident AMO route must accept exact parameter tuples");

        let mut expected_amo = Vec::with_capacity(result.rows * result.cols);
        let mut expected_ama = Vec::with_capacity(result.rows * result.cols);
        for &(length, smoothing_length) in &parameter_tuples {
            let input = AdaptiveMomentumOscillatorInput::from_slice(
                &close,
                AdaptiveMomentumOscillatorParams {
                    length: Some(length),
                    smoothing_length: Some(smoothing_length),
                },
            );
            let expected = adaptive_momentum_oscillator(&input)
                .expect("the CPU parity reference must accept the same tuple");
            expected_amo.extend(expected.amo);
            expected_ama.extend(expected.ama);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident output must download for parity inspection");
            let expected = match output.output_id {
                "amo" => &expected_amo,
                "ama" => &expected_ama,
                unexpected => panic!("unexpected AMO output identity {unexpected}"),
            };
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] differs: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    /// ASTC's histogram is produced by the same state machine as STC. Both
    /// canonical matrices must therefore stay in the frame's existing CUDA
    /// session; the superseded wrapper's private context and three input
    /// uploads are not a production route.
    #[test]
    fn adaptive_schaff_trend_cycle_all_outputs_stay_resident() {
        use vector_ta::indicators::adaptive_schaff_trend_cycle::AdaptiveSchaffTrendCycleParams;

        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card ASTC resident test requires a launchable CUDA device 0");
        let uploads_before = engine.uploads();
        let parameter_rows = [
            AdaptiveSchaffTrendCycleParams::default(),
            AdaptiveSchaffTrendCycleParams {
                adaptive_length: Some(21),
                stc_length: Some(8),
                smoothing_factor: Some(0.35),
                fast_length: Some(10),
                slow_length: Some(34),
            },
        ];

        let result = engine
            .compute_adaptive_schaff_trend_cycle_outputs_device(&parameter_rows)
            .expect("both ASTC outputs must launch through the shared resident session");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the ASTC route re-uploaded HLC input"
        );
        assert_eq!(result.indicator_id, "adaptive_schaff_trend_cycle");
        assert_eq!(result.entry_point, "adaptive_schaff_trend_cycle_batch_f64");
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ["stc", "histogram"]
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("both resident ASTC outputs must retire successfully");
    }

    /// Exact CPU/GPU parity for ASTC's full two-output recurrence, including
    /// state resets at NaN, infinity and a finite high<low bar.
    #[test]
    fn adaptive_schaff_trend_cycle_all_outputs_match_cpu_bits() {
        use vector_ta::indicators::adaptive_schaff_trend_cycle::{
            AdaptiveSchaffTrendCycleInput, AdaptiveSchaffTrendCycleParams,
            adaptive_schaff_trend_cycle,
        };

        let close = (0..144)
            .map(|index| {
                let x = index as f64;
                100.0 + x * 0.075 + (x * 0.31).sin() * 3.25 + (x * 0.07).cos()
            })
            .collect::<Vec<_>>();
        let mut high = close
            .iter()
            .enumerate()
            .map(|(index, value)| value + 0.75 + (index % 5) as f64 * 0.05)
            .collect::<Vec<_>>();
        let mut low = close
            .iter()
            .enumerate()
            .map(|(index, value)| value - 0.65 - (index % 7) as f64 * 0.04)
            .collect::<Vec<_>>();
        let mut close = close;
        for index in [17usize, 73] {
            high[index] = f64::NAN;
            low[index] = f64::NAN;
            close[index] = f64::NAN;
        }
        high[41] = f64::INFINITY;
        low[41] = f64::INFINITY;
        close[41] = f64::INFINITY;
        high[103] = close[103] - 1.0;
        low[103] = close[103] + 1.0;

        let ohlcv = Ohlcv {
            timestamp: None,
            open: close.clone(),
            high: high.clone(),
            low: low.clone(),
            close: close.clone(),
            volume: Some(vec![1.0; close.len()]),
        };
        let parameter_rows = [
            AdaptiveSchaffTrendCycleParams {
                adaptive_length: Some(3),
                stc_length: Some(2),
                smoothing_factor: Some(0.5),
                fast_length: Some(2),
                slow_length: Some(5),
            },
            AdaptiveSchaffTrendCycleParams {
                adaptive_length: Some(8),
                stc_length: Some(4),
                smoothing_factor: Some(0.35),
                fast_length: Some(3),
                slow_length: Some(9),
            },
            AdaptiveSchaffTrendCycleParams::default(),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the ASTC parity test requires a launchable CUDA device 0");
        let result = engine
            .compute_adaptive_schaff_trend_cycle_outputs_device(&parameter_rows)
            .expect("the resident ASTC route must accept the exact parameter rows");

        let mut expected_stc = Vec::with_capacity(result.rows * result.cols);
        let mut expected_histogram = Vec::with_capacity(result.rows * result.cols);
        for params in parameter_rows {
            let input = AdaptiveSchaffTrendCycleInput::from_slices(&high, &low, &close, params);
            let expected = adaptive_schaff_trend_cycle(&input)
                .expect("the CPU parity reference must accept the same parameter row");
            expected_stc.extend(expected.stc);
            expected_histogram.extend(expected.histogram);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident ASTC output must download for parity inspection");
            let expected = match output.output_id {
                "stc" => &expected_stc,
                "histogram" => &expected_histogram,
                unexpected => panic!("unexpected ASTC output identity {unexpected}"),
            };
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU reset/undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] differs: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    /// Ehlers Adaptive CG consumes the resident `hl2` source and writes both
    /// `cg` and `trigger`. A close-priced launch or a second upload would be a
    /// different indicator, not a compatible implementation.
    #[test]
    fn ehlers_adaptive_cg_all_outputs_stay_resident() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card multi-output test requires a launchable CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_ehlers_adaptive_cg_outputs_device(&[0.07, 0.2])
            .expect("both Ehlers Adaptive CG outputs must use the shared resident session");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "ehlers_adaptive_cg");
        assert_eq!(result.entry_point, "ehlers_adaptive_cg_batch_f64");
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ["cg", "trigger"]
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 2
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("both resident Ehlers Adaptive CG outputs must retire successfully");
    }

    /// CPU/GPU parity for the exact alpha sweep and both Ehlers Adaptive CG
    /// outputs. The alternating tail exercises near-cancelling windows and
    /// the f64 epsilon division guard repaired in the shared CUDA source.
    /// This remains a parity reference, not the independent formula review.
    #[test]
    fn ehlers_adaptive_cg_all_outputs_match_cpu_bits() {
        use vector_ta::indicators::ehlers_adaptive_cg::{
            EhlersAdaptiveCgInput, EhlersAdaptiveCgParams, ehlers_adaptive_cg,
        };

        let mut source = vec![
            f64::NAN,
            10.0,
            11.0,
            9.0,
            12.0,
            8.0,
            13.0,
            7.0,
            14.0,
            6.0,
            15.0,
            5.0,
            16.0,
            4.0,
            17.0,
            3.0,
            18.0,
            2.0,
            19.0,
            1.0,
        ];
        for index in 0..48 {
            let magnitude = 1.0 + (index % 3) as f64 * f64::EPSILON;
            source.push(if index % 2 == 0 {
                magnitude
            } else {
                -magnitude
            });
        }
        source.extend([0.0, -0.0, f64::EPSILON, -f64::EPSILON]);

        let ohlcv = Ohlcv {
            timestamp: None,
            open: source.clone(),
            high: source.clone(),
            low: source.clone(),
            close: source.clone(),
            volume: Some(vec![0.0; source.len()]),
        };
        let hl2 = ohlcv
            .high
            .iter()
            .zip(&ohlcv.low)
            .map(|(&high, &low)| (high + low) / 2.0)
            .collect::<Vec<_>>();
        let alphas = [0.03, 0.07, 0.2];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the parity test requires a launchable CUDA device 0");
        let result = engine
            .compute_ehlers_adaptive_cg_outputs_device(&alphas)
            .expect("the resident Ehlers Adaptive CG route must accept exact alphas");

        let mut expected_cg = Vec::with_capacity(result.rows * result.cols);
        let mut expected_trigger = Vec::with_capacity(result.rows * result.cols);
        for &alpha in &alphas {
            let input = EhlersAdaptiveCgInput::from_slice(
                &hl2,
                EhlersAdaptiveCgParams { alpha: Some(alpha) },
            );
            let expected = ehlers_adaptive_cg(&input)
                .expect("the CPU parity reference must accept the same alpha");
            expected_cg.extend(expected.cg);
            expected_trigger.extend(expected.trigger);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident output must download for parity inspection");
            let expected = match output.output_id {
                "cg" => &expected_cg,
                "trigger" => &expected_trigger,
                unexpected => panic!("unexpected Ehlers Adaptive CG output identity {unexpected}"),
            };
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] differs: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    /// Ehlers Adaptive Cyber Cycle consumes the resident `hl2` source and
    /// writes `cycle` and `trigger` in one CUDA launch. GpuOnly must not
    /// rebuild the trigger on the host or create a second CUDA context.
    #[test]
    fn ehlers_adaptive_cyber_cycle_all_outputs_stay_resident() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card multi-output test requires a launchable CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_ehlers_adaptive_cyber_cycle_outputs_device(&[0.07, 0.2])
            .expect("both Ehlers Adaptive Cyber Cycle outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "ehlers_adaptive_cyber_cycle");
        assert_eq!(result.entry_point, "ehlers_adaptive_cyber_cycle_batch_f64");
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ["cycle", "trigger"]
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 2
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("both resident Ehlers Adaptive Cyber Cycle outputs must retire");
    }

    /// CPU/GPU parity for the complete Cyber Cycle state machine, including
    /// the finite-only source rule and both inclusive alpha boundaries. This
    /// protects the resident connection; independent formula authority is a
    /// separate review gate.
    #[test]
    fn ehlers_adaptive_cyber_cycle_all_outputs_match_cpu_bits() {
        use vector_ta::indicators::ehlers_adaptive_cyber_cycle::{
            EhlersAdaptiveCyberCycleInput, EhlersAdaptiveCyberCycleParams,
            ehlers_adaptive_cyber_cycle,
        };

        let mut source = vec![
            f64::INFINITY,
            f64::NAN,
            10.0,
            11.0,
            9.0,
            12.0,
            8.0,
            13.0,
            7.0,
            14.0,
            6.0,
            15.0,
            5.0,
            16.0,
            4.0,
            17.0,
            3.0,
            18.0,
            2.0,
            19.0,
            1.0,
        ];
        for index in 0..64 {
            let epsilon = (1 + index % 4) as f64 * f64::EPSILON;
            source.push(if index % 2 == 0 {
                1.0 + epsilon
            } else {
                -1.0 - epsilon
            });
        }
        source.extend([
            f64::NEG_INFINITY,
            f64::NAN,
            0.0,
            -0.0,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            2.0,
            -2.0,
        ]);

        let ohlcv = Ohlcv {
            timestamp: None,
            open: source.clone(),
            high: source.clone(),
            low: source.clone(),
            close: source.clone(),
            volume: Some(vec![0.0; source.len()]),
        };
        let hl2 = ohlcv
            .high
            .iter()
            .zip(&ohlcv.low)
            .map(|(&high, &low)| (high + low) / 2.0)
            .collect::<Vec<_>>();
        let alphas = [0.0, 0.07, 0.2, 1.0];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the parity test requires a launchable CUDA device 0");
        let result = engine
            .compute_ehlers_adaptive_cyber_cycle_outputs_device(&alphas)
            .expect("the resident Cyber Cycle route must accept exact alphas");

        let mut expected_cycle = Vec::with_capacity(result.rows * result.cols);
        let mut expected_trigger = Vec::with_capacity(result.rows * result.cols);
        for &alpha in &alphas {
            let input = EhlersAdaptiveCyberCycleInput::from_slice(
                &hl2,
                EhlersAdaptiveCyberCycleParams { alpha: Some(alpha) },
            );
            let expected = ehlers_adaptive_cyber_cycle(&input)
                .expect("the CPU parity reference must accept the same alpha");
            expected_cycle.extend(expected.cycle);
            expected_trigger.extend(expected.trigger);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident output must download for parity inspection");
            let expected = match output.output_id {
                "cycle" => &expected_cycle,
                "trigger" => &expected_trigger,
                unexpected => panic!("unexpected Cyber Cycle output identity {unexpected}"),
            };
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] differs: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    /// Ehlers Simple Cycle Indicator has the same two-output device ABI as
    /// Cyber Cycle, but it is a distinct formula/kernel and must have its own
    /// exact route rather than borrowing another indicator's output.
    #[test]
    fn ehlers_simple_cycle_indicator_all_outputs_stay_resident() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card multi-output test requires a launchable CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_ehlers_simple_cycle_indicator_outputs_device(&[0.07, 0.2])
            .expect("both Ehlers Simple Cycle outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "ehlers_simple_cycle_indicator");
        assert_eq!(
            result.entry_point,
            "ehlers_simple_cycle_indicator_batch_f64"
        );
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ["cycle", "trigger"]
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 2
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("both resident Ehlers Simple Cycle outputs must retire");
    }

    /// CPU/GPU parity for both Simple Cycle outputs, including skipped
    /// non-finite bars and the inclusive alpha boundaries. This is a parity
    /// reference; formula authority remains an independent review.
    #[test]
    fn ehlers_simple_cycle_indicator_all_outputs_match_cpu_bits() {
        use vector_ta::indicators::ehlers_simple_cycle_indicator::{
            EhlersSimpleCycleIndicatorInput, EhlersSimpleCycleIndicatorParams,
            ehlers_simple_cycle_indicator,
        };

        let mut source = vec![
            f64::INFINITY,
            f64::NAN,
            10.0,
            11.0,
            9.0,
            12.0,
            8.0,
            13.0,
            7.0,
            14.0,
            6.0,
            15.0,
            5.0,
            16.0,
            4.0,
        ];
        for index in 0..64 {
            let epsilon = (1 + index % 4) as f64 * f64::EPSILON;
            source.push(if index % 2 == 0 {
                1.0 + epsilon
            } else {
                -1.0 - epsilon
            });
        }
        source.extend([
            f64::NEG_INFINITY,
            f64::NAN,
            0.0,
            -0.0,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            2.0,
            -2.0,
        ]);

        let ohlcv = Ohlcv {
            timestamp: None,
            open: source.clone(),
            high: source.clone(),
            low: source.clone(),
            close: source.clone(),
            volume: Some(vec![0.0; source.len()]),
        };
        let hl2 = ohlcv
            .high
            .iter()
            .zip(&ohlcv.low)
            .map(|(&high, &low)| (high + low) / 2.0)
            .collect::<Vec<_>>();
        let alphas = [0.0, 0.07, 0.2, 1.0];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the parity test requires a launchable CUDA device 0");
        let result = engine
            .compute_ehlers_simple_cycle_indicator_outputs_device(&alphas)
            .expect("the resident Simple Cycle route must accept exact alphas");

        let mut expected_cycle = Vec::with_capacity(result.rows * result.cols);
        let mut expected_trigger = Vec::with_capacity(result.rows * result.cols);
        for &alpha in &alphas {
            let input = EhlersSimpleCycleIndicatorInput::from_slice(
                &hl2,
                EhlersSimpleCycleIndicatorParams { alpha: Some(alpha) },
            );
            let expected = ehlers_simple_cycle_indicator(&input)
                .expect("the CPU parity reference must accept the same alpha");
            expected_cycle.extend(expected.cycle);
            expected_trigger.extend(expected.trigger);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident output must download for parity inspection");
            let expected = match output.output_id {
                "cycle" => &expected_cycle,
                "trigger" => &expected_trigger,
                unexpected => panic!("unexpected Simple Cycle output identity {unexpected}"),
            };
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] differs: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    /// Ehlers PMA has no numeric search parameter. The old benchmark wrapper
    /// manufactured hundreds of identical combo rows; production GpuOnly
    /// must launch the real formula once and retain its distinct `predict`
    /// and `trigger` outputs on the selected device.
    #[test]
    fn ehlers_pma_all_outputs_stay_resident_without_duplicate_rows() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card multi-output test requires a launchable CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_ehlers_pma_outputs_device()
            .expect("both Ehlers PMA outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "ehlers_pma");
        assert_eq!(result.entry_point, "ehlers_pma_bars_parallel_f64");
        assert_eq!(result.rows, 1, "parameter-free PMA emitted duplicate rows");
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ["predict", "trigger"]
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 1
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("both resident Ehlers PMA outputs must retire");
    }

    /// Exact parity for the parameter-free Ehlers PMA f64 recurrence and both
    /// outputs. It also proves that the single resident row represents the
    /// one real formula instead of an arbitrary duplicate-combo sweep.
    #[test]
    fn ehlers_pma_all_outputs_match_cpu_bits() {
        use vector_ta::indicators::moving_averages::ehlers_pma::{
            EhlersPmaInput, EhlersPmaParams, ehlers_pma,
        };

        let mut close = vec![f64::NAN, f64::NAN];
        close.extend([
            10.0,
            11.0,
            9.0,
            12.0,
            8.0,
            13.0,
            7.0,
            14.0,
            6.0,
            15.0,
            5.0,
            16.0,
            4.0,
            17.0,
            3.0,
            18.0,
            2.0,
            19.0,
            1.0,
            0.0,
            -0.0,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            1.0e150,
            -1.0e150,
            2.0,
            -2.0,
            3.0,
            -3.0,
        ]);
        for index in 0..64 {
            close.push((index as f64 * 0.125).sin() * 100.0 + index as f64 * 0.01);
        }

        let ohlcv = Ohlcv {
            timestamp: None,
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            close: close.clone(),
            volume: Some(vec![0.0; close.len()]),
        };
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the parity test requires a launchable CUDA device 0");
        let result = engine
            .compute_ehlers_pma_outputs_device()
            .expect("the resident Ehlers PMA route must launch");
        let expected = ehlers_pma(&EhlersPmaInput::from_slice(&close, EhlersPmaParams))
            .expect("the CPU parity reference must accept the same series");

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident output must download for parity inspection");
            let expected = match output.output_id {
                "predict" => &expected.predict,
                "trigger" => &expected.trigger,
                unexpected => panic!("unexpected Ehlers PMA output identity {unexpected}"),
            };
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] differs: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    fn assert_fibonacci_named_outputs_match_cpu_bits(
        engine: &GpuIndicatorEngine,
        result: &F64NamedOutputsResult,
        expected: &vector_ta::indicators::fibonacci_entry_bands::FibonacciEntryBandsBatchOutput,
        context: &str,
    ) {
        assert_eq!(
            (result.rows, result.cols),
            (expected.rows, expected.cols),
            "{context}: result shape differs"
        );
        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident output must download for parity inspection");
            let expected: &[f64] = match output.output_id {
                "middle" => &expected.basis,
                "trend" => &expected.trend,
                "upper_0618" => &expected.upper_0618,
                "upper_1000" => &expected.upper_1000,
                "upper_1618" => &expected.upper_1618,
                "upper_2618" => &expected.upper_2618,
                "lower_0618" => &expected.lower_0618,
                "lower_1000" => &expected.lower_1000,
                "lower_1618" => &expected.lower_1618,
                "lower_2618" => &expected.lower_2618,
                "tp_long_band" => &expected.tp_long_band,
                "tp_short_band" => &expected.tp_short_band,
                "go_long" => &expected.long_entry,
                "go_short" => &expected.short_entry,
                "rejection_long" => &expected.rejection_long,
                "rejection_short" => &expected.rejection_short,
                "long_bounce" => &expected.long_bounce,
                "short_bounce" => &expected.short_bounce,
                unexpected => panic!("unexpected Fibonacci Entry Bands output {unexpected}"),
            };
            assert_eq!(actual.len(), expected.len(), "{context}: output length");
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{context}: {}[{index}] lost CPU undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{context}: {}[{index}] differs: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    /// Large, deterministic real-card fixture for Nsight Compute. It is kept
    /// out of the ordinary suite because the profiler replays the kernel and
    /// needs a launch large enough to expose grid/SM occupancy honestly.
    #[test]
    #[ignore = "run explicitly under ncu to profile the bar-parallel PMA kernel"]
    fn profile_ehlers_pma_bars_parallel_full_card_fixture() {
        use cust::event::{Event, EventFlags};

        const BARS: usize = 4 * 1024 * 1024;
        const MEASURED_LAUNCHES: usize = 20;
        let close = (0..BARS)
            .map(|index| {
                100.0 + (index % 4096) as f64 * 0.000_1 + ((index / 4096) % 19) as f64 * 0.000_01
            })
            .collect::<Vec<_>>();
        let ohlcv = Ohlcv {
            timestamp: None,
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            close: close.clone(),
            volume: Some(vec![0.0; BARS]),
        };
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Nsight fixture requires a launchable CUDA device 0");

        let capacity = engine
            .f64_engine
            .ehlers_pma_bars_parallel_launch_capacity(BARS)
            .expect("query the exact PMA launch and occupancy capacity");
        assert_eq!(capacity.threads_per_block, 256);
        assert_eq!(capacity.grid_blocks, 16_384);
        assert_eq!(capacity.logical_work_items, BARS);
        assert!(capacity.grid_blocks >= capacity.multiprocessors);
        let resident_threads =
            capacity.max_active_blocks_per_multiprocessor * capacity.threads_per_block;
        let theoretical_occupancy =
            resident_threads as f64 / capacity.max_threads_per_multiprocessor as f64;
        assert_eq!(
            resident_threads, capacity.max_threads_per_multiprocessor,
            "the sm89 launch is resource-limited below full theoretical thread occupancy"
        );
        eprintln!(
            "NEOETHOS_EHLERS_PMA_LAUNCH grid_blocks={} threads_per_block={} \
             logical_work_items={} multiprocessors={} max_active_blocks_per_sm={} \
             resident_threads_per_sm={} max_threads_per_sm={} theoretical_occupancy={:.3}",
            capacity.grid_blocks,
            capacity.threads_per_block,
            capacity.logical_work_items,
            capacity.multiprocessors,
            capacity.max_active_blocks_per_multiprocessor,
            resident_threads,
            capacity.max_threads_per_multiprocessor,
            theoretical_occupancy,
        );

        let warmup = engine
            .compute_ehlers_pma_outputs_device()
            .expect("the PMA profiler warmup must launch");
        assert_eq!(warmup.entry_point, "ehlers_pma_bars_parallel_f64");
        assert_eq!((warmup.rows, warmup.cols), (1, BARS));
        engine
            .synchronize()
            .expect("the PMA profiler warmup must retire successfully");
        drop(warmup);

        let mut milliseconds = Vec::with_capacity(MEASURED_LAUNCHES);
        for _ in 0..MEASURED_LAUNCHES {
            let start = Event::new(EventFlags::DEFAULT).expect("create CUDA start event");
            let stop = Event::new(EventFlags::DEFAULT).expect("create CUDA stop event");
            start
                .record(engine.runtime.stream())
                .expect("record CUDA start event");
            let result = engine
                .compute_ehlers_pma_outputs_device()
                .expect("the measured PMA route must launch");
            stop.record(engine.runtime.stream())
                .expect("record CUDA stop event");
            stop.synchronize().expect("wait for measured PMA launch");
            let elapsed = stop
                .elapsed_time_f32(&start)
                .expect("measure PMA CUDA event interval");
            assert!(elapsed.is_finite() && elapsed > 0.0);
            std::hint::black_box(&result);
            milliseconds.push(elapsed);
        }

        milliseconds.sort_by(f32::total_cmp);
        let median_ms = milliseconds[milliseconds.len() / 2];
        let p95_index = (milliseconds.len() * 95).div_ceil(100) - 1;
        let p95_ms = milliseconds[p95_index];
        let median_mbars_per_second = BARS as f64 / median_ms as f64 / 1_000.0;
        eprintln!(
            "NEOETHOS_EHLERS_PMA_GPU_EVENT bars={BARS} launches={MEASURED_LAUNCHES} \
             median_ms={median_ms:.6} p95_ms={p95_ms:.6} \
             median_mbars_per_second={median_mbars_per_second:.3}"
        );
    }

    /// The existing f64 Fibonacci Entry Bands kernel already writes eighteen
    /// distinct matrices. GpuOnly must borrow this frame's resident OHLC and
    /// one CUDA session rather than constructing the legacy wrapper, context,
    /// stream and four duplicate host uploads.
    #[test]
    fn fibonacci_entry_bands_all_outputs_stay_resident() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card multi-output test requires a launchable CUDA device 0");
        let uploads_before = engine.uploads();
        let sweep = FibonacciEntryBandsBatchRange {
            length: (5, 9, 4),
            atr_length: (3, 7, 4),
            source: "hlc3".into(),
            use_atr: true,
            tp_aggressiveness: "medium".into(),
        };

        let result = engine
            .compute_fibonacci_entry_bands_outputs_device(&sweep)
            .expect("all Fibonacci Entry Bands outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded OHLC"
        );
        assert_eq!(result.indicator_id, "fibonacci_entry_bands");
        assert_eq!(result.entry_point, "fibonacci_entry_bands_batch_f64");
        assert_eq!((result.rows, result.cols), (4, ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            [
                "middle",
                "trend",
                "upper_0618",
                "upper_1000",
                "upper_1618",
                "upper_2618",
                "lower_0618",
                "lower_1000",
                "lower_1618",
                "lower_2618",
                "tp_long_band",
                "tp_short_band",
                "go_long",
                "go_short",
                "rejection_long",
                "rejection_short",
                "long_bounce",
                "short_bounce",
            ]
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 4
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident Fibonacci Entry Bands outputs must retire");
    }

    /// Parity for every value-bearing Fibonacci Entry Bands output. This is a
    /// CUDA/CPU implementation check, not independent formula authority; exact
    /// bits prevent a hidden output swap or different f64 operation order from
    /// being accepted merely because the bands look visually close.
    #[test]
    fn fibonacci_entry_bands_all_outputs_match_cpu_bits() {
        use vector_ta::indicators::fibonacci_entry_bands::fibonacci_entry_bands_batch_with_kernel;
        use vector_ta::utilities::enums::Kernel;

        // The admitted length-200 tuple needs more history than the canonical
        // 100-bar seed. Reuse the broker-captured values through the existing
        // deterministic repeated fixture instead of dropping the longest row.
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the parity test requires a launchable CUDA device 0");

        for length in [21, 7, 50, 100, 200] {
            let sweep = FibonacciEntryBandsBatchRange {
                length: (length, length, 0),
                atr_length: (14, 14, 0),
                source: "hlc3".into(),
                use_atr: true,
                tp_aggressiveness: "low".into(),
            };
            let expected = fibonacci_entry_bands_batch_with_kernel(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                &sweep,
                Kernel::ScalarBatch,
            )
            .unwrap_or_else(|error| {
                panic!("the scalar CPU reference rejected canonical length={length}: {error}")
            });
            let result = engine
                .compute_fibonacci_entry_bands_outputs_device(&sweep)
                .unwrap_or_else(|error| {
                    panic!("resident route rejected canonical length={length}: {error}")
                });

            assert_fibonacci_named_outputs_match_cpu_bits(
                &engine,
                &result,
                &expected,
                &format!("canonical length={length}"),
            );
        }
    }

    /// Every public source, volatility mode and TP policy must preserve the
    /// same reset/validity and f64 arithmetic. The fixture remains captured
    /// cTrader data; two bars are made invalid to exercise both HLC-wide resets
    /// and the stricter open-sensitive reset without inventing a price series.
    #[test]
    fn fibonacci_entry_bands_all_parameter_modes_match_cpu_bits_across_gaps() {
        use vector_ta::indicators::fibonacci_entry_bands::fibonacci_entry_bands_batch_with_kernel;
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        ohlcv.open[41] = f64::NAN;
        ohlcv.high[73] = f64::NAN;
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the parameter-mode parity test requires a launchable CUDA device 0");

        for source in [
            "open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4",
        ] {
            for use_atr in [false, true] {
                for tp_aggressiveness in ["low", "medium", "high"] {
                    let sweep = FibonacciEntryBandsBatchRange {
                        length: (5, 5, 0),
                        atr_length: (3, 3, 0),
                        source: source.into(),
                        use_atr,
                        tp_aggressiveness: tp_aggressiveness.into(),
                    };
                    let expected = fibonacci_entry_bands_batch_with_kernel(
                        &ohlcv.open,
                        &ohlcv.high,
                        &ohlcv.low,
                        &ohlcv.close,
                        &sweep,
                        Kernel::ScalarBatch,
                    )
                    .expect("the scalar CPU reference must accept every public parameter mode");
                    let result = engine
                        .compute_fibonacci_entry_bands_outputs_device(&sweep)
                        .unwrap_or_else(|error| {
                            panic!(
                                "resident route rejected source={source} use_atr={use_atr} \
                                 tp={tp_aggressiveness}: {error}"
                            )
                        });
                    let context =
                        format!("source={source} use_atr={use_atr} tp={tp_aggressiveness}");
                    assert_fibonacci_named_outputs_match_cpu_bits(
                        &engine, &result, &expected, &context,
                    );
                }
            }
        }
    }

    /// Adaptive Bandpass Trigger Oscillator writes `in_phase` and `lead` from
    /// the same carried state. Both matrices must stay on the selected card;
    /// reconstructing `lead` on the host is not an admissible GpuOnly route.
    #[test]
    fn adaptive_bandpass_trigger_oscillator_all_outputs_stay_resident() {
        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card multi-output test requires a launchable CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_adaptive_bandpass_trigger_oscillator_outputs_device(&[(0.1, 0.07), (0.2, 0.1)])
            .expect("both adaptive bandpass outputs must use the shared resident session");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "adaptive_bandpass_trigger_oscillator");
        assert_eq!(
            result.entry_point,
            "adaptive_bandpass_trigger_oscillator_batch_f64"
        );
        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ["in_phase", "lead"]
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 2
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("both resident adaptive bandpass outputs must retire successfully");
    }

    /// Exact CPU/GPU parity for both adaptive-bandpass columns across the
    /// complete registry parameter boundary and non-finite reset boundaries.
    /// The CPU and CUDA translation unit use the same bounded msun cosine, so
    /// no tolerance is admissible inside the recursive state.
    #[test]
    fn adaptive_bandpass_trigger_oscillator_matches_cpu_bits() {
        use vector_ta::indicators::adaptive_bandpass_trigger_oscillator::{
            AdaptiveBandpassTriggerOscillatorInput, AdaptiveBandpassTriggerOscillatorParams,
            adaptive_bandpass_trigger_oscillator,
        };

        let close = vec![
            f64::NAN,
            10.0,
            11.0,
            9.5,
            12.0,
            8.5,
            13.0,
            8.0,
            14.0,
            7.5,
            15.0,
            7.0,
            16.0,
            6.5,
            17.0,
            6.0,
            f64::INFINITY,
            20.0,
            18.0,
            21.0,
            17.0,
            22.0,
            16.0,
            23.0,
            15.0,
            24.0,
            14.0,
            25.0,
            13.0,
            26.0,
            12.0,
            27.0,
            11.0,
            28.0,
            10.0,
            29.0,
            9.0,
            30.0,
            f64::NEG_INFINITY,
            31.0,
            30.0,
            32.0,
            29.0,
            33.0,
            28.0,
            34.0,
            27.0,
            35.0,
            26.0,
            36.0,
            25.0,
            37.0,
            24.0,
            38.0,
        ];
        let ohlcv = Ohlcv {
            timestamp: None,
            open: close.clone(),
            high: close.clone(),
            low: close.clone(),
            close: close.clone(),
            volume: Some(vec![0.0; close.len()]),
        };
        let deltas = [0.0000001, 0.01, 0.05, 0.1, 0.4, 0.75, 0.9999999];
        let alphas = [0.0000001, 0.01, 0.03, 0.07, 0.2, 0.75, 0.9999999];
        let parameter_tuples: Vec<(f64, f64)> = deltas
            .into_iter()
            .flat_map(|delta| alphas.into_iter().map(move |alpha| (delta, alpha)))
            .collect();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the parity test requires a launchable CUDA device 0");
        let result = engine
            .compute_adaptive_bandpass_trigger_oscillator_outputs_device(&parameter_tuples)
            .expect("the resident adaptive-bandpass route must accept exact parameter tuples");

        let mut expected_in_phase = Vec::with_capacity(result.rows * result.cols);
        let mut expected_lead = Vec::with_capacity(result.rows * result.cols);
        for &(delta, alpha) in &parameter_tuples {
            let input = AdaptiveBandpassTriggerOscillatorInput::from_slice(
                &close,
                AdaptiveBandpassTriggerOscillatorParams {
                    delta: Some(delta),
                    alpha: Some(alpha),
                },
            );
            let expected = adaptive_bandpass_trigger_oscillator(&input)
                .expect("the CPU parity reference must accept the same tuple");
            expected_in_phase.extend(expected.in_phase);
            expected_lead.extend(expected.lead);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident output must download for parity inspection");
            let expected = match output.output_id {
                "in_phase" => &expected_in_phase,
                "lead" => &expected_lead,
                unexpected => panic!("unexpected adaptive-bandpass output identity {unexpected}"),
            };
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU reset/undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not bit-identical: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
        eprintln!(
            "ADAPTIVE_BANDPASS_F64_PARITY exact rows={} cells_per_output={}",
            parameter_tuples.len(),
            parameter_tuples.len() * close.len()
        );
    }

    /// Trend Flow Trail already has one CUDA entry point that computes all
    /// seventeen columns. GpuOnly must connect every matrix to the frame's one
    /// resident OHLCV upload/session instead of invoking its superseded wrapper
    /// (which creates another context, re-uploads five host arrays and syncs).
    #[test]
    fn trend_flow_trail_all_outputs_stay_resident() {
        let ohlcv = trend_flow_trail_parity_fixture();
        let parameter_tuples = [(1, 0.1, 1), (7, 1.7, 5), (33, 3.3, 14), (48, 6.5, 21)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card Trend Flow Trail test requires launchable CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_trend_flow_trail_outputs_device(&parameter_tuples)
            .expect("all Trend Flow Trail outputs must launch through the resident session");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "trend_flow_trail");
        assert_eq!(result.entry_point, "trend_flow_trail_batch_f64");
        assert_eq!(result.rows, parameter_tuples.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            TREND_FLOW_TRAIL_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("all resident Trend Flow Trail outputs must retire successfully");
    }

    /// Exact parity for the complete recursive state, all event columns,
    /// parameter boundaries and two different non-finite reset causes.
    /// Independent formula authority remains a separate promotion gate.
    #[test]
    fn trend_flow_trail_all_outputs_match_cpu_bits() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::trend_flow_trail::{
            TrendFlowTrailInput, TrendFlowTrailParams, trend_flow_trail,
        };

        let ohlcv = trend_flow_trail_parity_fixture();
        let volume = ohlcv.volume.as_ref().expect("fixture carries exact volume");
        let parameter_tuples = [(1, 0.1, 1), (7, 1.7, 5), (33, 3.3, 14), (48, 6.5, 21)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the parity test requires a launchable CUDA device 0");
        let result = engine
            .compute_trend_flow_trail_outputs_device(&parameter_tuples)
            .expect("the resident Trend Flow Trail route must accept exact tuples");

        let mut expected: BTreeMap<&str, Vec<f64>> = TREND_FLOW_TRAIL_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for &(alpha_length, alpha_multiplier, mfi_length) in &parameter_tuples {
            let input = TrendFlowTrailInput::from_slices(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                volume,
                TrendFlowTrailParams {
                    alpha_length: Some(alpha_length),
                    alpha_multiplier: Some(alpha_multiplier),
                    mfi_length: Some(mfi_length),
                },
            );
            let output = trend_flow_trail(&input)
                .expect("the CPU parity reference must accept the same tuple");
            expected
                .get_mut("alpha_trail")
                .unwrap()
                .extend(output.alpha_trail);
            expected
                .get_mut("alpha_trail_bullish")
                .unwrap()
                .extend(output.alpha_trail_bullish);
            expected
                .get_mut("alpha_trail_bearish")
                .unwrap()
                .extend(output.alpha_trail_bearish);
            expected
                .get_mut("alpha_dir")
                .unwrap()
                .extend(output.alpha_dir);
            expected.get_mut("mfi").unwrap().extend(output.mfi);
            expected
                .get_mut("tp_upper")
                .unwrap()
                .extend(output.tp_upper);
            expected
                .get_mut("tp_lower")
                .unwrap()
                .extend(output.tp_lower);
            expected
                .get_mut("alpha_trail_bullish_switch")
                .unwrap()
                .extend(output.alpha_trail_bullish_switch);
            expected
                .get_mut("alpha_trail_bearish_switch")
                .unwrap()
                .extend(output.alpha_trail_bearish_switch);
            expected
                .get_mut("mfi_overbought")
                .unwrap()
                .extend(output.mfi_overbought);
            expected
                .get_mut("mfi_oversold")
                .unwrap()
                .extend(output.mfi_oversold);
            expected
                .get_mut("mfi_cross_up_mid")
                .unwrap()
                .extend(output.mfi_cross_up_mid);
            expected
                .get_mut("mfi_cross_down_mid")
                .unwrap()
                .extend(output.mfi_cross_down_mid);
            expected
                .get_mut("price_cross_alpha_trail_up")
                .unwrap()
                .extend(output.price_cross_alpha_trail_up);
            expected
                .get_mut("price_cross_alpha_trail_down")
                .unwrap()
                .extend(output.price_cross_alpha_trail_down);
            expected
                .get_mut("mfi_above_90")
                .unwrap()
                .extend(output.mfi_above_90);
            expected
                .get_mut("mfi_below_10")
                .unwrap()
                .extend(output.mfi_below_10);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Trend Flow Trail output must download for parity");
            let cpu = expected.get(output.output_id).unwrap_or_else(|| {
                panic!("unexpected Trend Flow Trail output {}", output.output_id)
            });
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU reset/undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not bit-identical: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn market_structure_confluence_all_outputs_stay_resident() {
        use vector_ta::indicators::market_structure_confluence::MarketStructureConfluenceParams;

        let ohlcv = market_structure_confluence_parity_fixture();
        let parameter_rows = vec![
            MarketStructureConfluenceParams {
                swing_size: Some(2),
                bos_confirmation: Some("Candle Close".into()),
                basis_length: Some(1),
                atr_length: Some(1),
                atr_smooth: Some(1),
                vol_mult: Some(0.5),
            },
            MarketStructureConfluenceParams {
                swing_size: Some(2),
                bos_confirmation: Some("Wicks".into()),
                basis_length: Some(3),
                atr_length: Some(2),
                atr_smooth: Some(2),
                vol_mult: Some(0.75),
            },
            MarketStructureConfluenceParams {
                swing_size: Some(5),
                bos_confirmation: Some("Candle Close".into()),
                basis_length: Some(13),
                atr_length: Some(7),
                atr_smooth: Some(3),
                vol_mult: Some(2.0),
            },
            MarketStructureConfluenceParams {
                swing_size: Some(9),
                bos_confirmation: Some("Wicks".into()),
                basis_length: Some(31),
                atr_length: Some(14),
                atr_smooth: Some(21),
                vol_mult: Some(3.5),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Market Structure Confluence test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_market_structure_confluence_outputs_device(&parameter_rows)
            .expect("all Market Structure Confluence outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "market_structure_confluence");
        assert_eq!(result.entry_point, "market_structure_confluence_batch_f64");
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            MARKET_STRUCTURE_CONFLUENCE_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Market Structure Confluence launch must retire");
    }

    #[test]
    fn market_structure_confluence_all_outputs_match_cpu_bits() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::market_structure_confluence::{
            MarketStructureConfluenceInput, MarketStructureConfluenceParams,
            market_structure_confluence,
        };

        let ohlcv = market_structure_confluence_parity_fixture();
        let parameter_rows = vec![
            MarketStructureConfluenceParams {
                swing_size: Some(2),
                bos_confirmation: Some("Candle Close".into()),
                basis_length: Some(1),
                atr_length: Some(1),
                atr_smooth: Some(1),
                vol_mult: Some(0.5),
            },
            MarketStructureConfluenceParams {
                swing_size: Some(2),
                bos_confirmation: Some("Wicks".into()),
                basis_length: Some(3),
                atr_length: Some(2),
                atr_smooth: Some(2),
                vol_mult: Some(0.75),
            },
            MarketStructureConfluenceParams {
                swing_size: Some(5),
                bos_confirmation: Some("Candle Close".into()),
                basis_length: Some(13),
                atr_length: Some(7),
                atr_smooth: Some(3),
                vol_mult: Some(2.0),
            },
            MarketStructureConfluenceParams {
                swing_size: Some(9),
                bos_confirmation: Some("Wicks".into()),
                basis_length: Some(31),
                atr_length: Some(14),
                atr_smooth: Some(21),
                vol_mult: Some(3.5),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Market Structure Confluence parity test requires CUDA device 0");
        let result = engine
            .compute_market_structure_confluence_outputs_device(&parameter_rows)
            .expect("the resident route must accept the exact parameter rows");

        let mut expected: BTreeMap<&str, Vec<f64>> = MARKET_STRUCTURE_CONFLUENCE_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for params in parameter_rows {
            let output = market_structure_confluence(&MarketStructureConfluenceInput::from_slices(
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                params,
            ))
            .expect("the scalar parity reference must accept the same row");
            macro_rules! extend_output {
                ($id:literal, $field:ident) => {
                    expected.get_mut($id).unwrap().extend(output.$field)
                };
            }
            extend_output!("basis", basis);
            extend_output!("upper_band", upper_band);
            extend_output!("lower_band", lower_band);
            extend_output!("structure_direction", structure_direction);
            extend_output!("bullish_arrow", bullish_arrow);
            extend_output!("bearish_arrow", bearish_arrow);
            extend_output!("bullish_change", bullish_change);
            extend_output!("bearish_change", bearish_change);
            extend_output!("hh", hh);
            extend_output!("lh", lh);
            extend_output!("hl", hl);
            extend_output!("ll", ll);
            extend_output!("bullish_bos", bullish_bos);
            extend_output!("bullish_choch", bullish_choch);
            extend_output!("bearish_bos", bearish_bos);
            extend_output!("bearish_choch", bearish_choch);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident output must download for exact parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not bit-identical: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn hema_trend_levels_all_outputs_stay_resident() {
        let ohlcv = hema_trend_levels_parity_fixture();
        let parameter_rows = vec![
            HemaTrendLevelsParams {
                fast_length: Some(1),
                slow_length: Some(2),
            },
            HemaTrendLevelsParams {
                fast_length: Some(3),
                slow_length: Some(8),
            },
            HemaTrendLevelsParams {
                fast_length: Some(20),
                slow_length: Some(40),
            },
            HemaTrendLevelsParams {
                fast_length: Some(31),
                slow_length: Some(67),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the HEMA Trend Levels test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_hema_trend_levels_outputs_device(&parameter_rows)
            .expect("all HEMA Trend Levels outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "hema_trend_levels");
        assert_eq!(result.entry_point, "hema_trend_levels_batch_f64");
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            HEMA_TREND_LEVELS_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident HEMA Trend Levels launch must retire");
    }

    #[test]
    fn hema_trend_levels_all_outputs_match_cpu_bits() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::hema_trend_levels::{HemaTrendLevelsInput, hema_trend_levels};

        let ohlcv = hema_trend_levels_parity_fixture();
        let parameter_rows = vec![
            HemaTrendLevelsParams {
                fast_length: Some(1),
                slow_length: Some(2),
            },
            HemaTrendLevelsParams {
                fast_length: Some(3),
                slow_length: Some(8),
            },
            HemaTrendLevelsParams {
                fast_length: Some(20),
                slow_length: Some(40),
            },
            HemaTrendLevelsParams {
                fast_length: Some(31),
                slow_length: Some(67),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the HEMA Trend Levels parity test requires CUDA device 0");
        let result = engine
            .compute_hema_trend_levels_outputs_device(&parameter_rows)
            .expect("the resident route must accept the exact parameter rows");

        let mut expected: BTreeMap<&str, Vec<f64>> = HEMA_TREND_LEVELS_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for params in parameter_rows {
            let output = hema_trend_levels(&HemaTrendLevelsInput::from_slices(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                params,
            ))
            .expect("the scalar parity reference must accept the same row");
            macro_rules! extend_output {
                ($id:literal, $field:ident) => {
                    expected.get_mut($id).unwrap().extend(output.$field)
                };
            }
            extend_output!("fast_hema", fast_hema);
            extend_output!("slow_hema", slow_hema);
            extend_output!("trend_direction", trend_direction);
            extend_output!("bar_state", bar_state);
            extend_output!("bullish_crossover", bullish_crossover);
            extend_output!("bearish_crossunder", bearish_crossunder);
            extend_output!("box_offset", box_offset);
            extend_output!("bull_box_top", bull_box_top);
            extend_output!("bull_box_bottom", bull_box_bottom);
            extend_output!("bear_box_top", bear_box_top);
            extend_output!("bear_box_bottom", bear_box_bottom);
            extend_output!("bullish_test", bullish_test);
            extend_output!("bearish_test", bearish_test);
            extend_output!("bullish_test_level", bullish_test_level);
            extend_output!("bearish_test_level", bearish_test_level);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident HEMA Trend Levels output must download for exact parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not bit-identical: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn ichimoku_oscillator_all_outputs_stay_resident() {
        use vector_ta::indicators::ichimoku_oscillator::IchimokuOscillatorNormalizeMode;

        let ohlcv = ichimoku_oscillator_parity_fixture();
        let sweep = IchimokuOscillatorBatchRange {
            conversion_periods: (3, 7, 2),
            base_periods: (9, 9, 0),
            lagging_span_periods: (15, 15, 0),
            displacement: (5, 5, 0),
            ma_length: (3, 3, 0),
            smoothing_length: (2, 2, 0),
            window_size: (5, 5, 0),
            top_band: (2.0, 2.0, 0.0),
            mid_band: (1.5, 1.5, 0.0),
            extra_smoothing: true,
            normalize: IchimokuOscillatorNormalizeMode::Window,
            clamp: true,
        };
        let expected_rows = vector_ta::indicators::ichimoku_oscillator::expand_grid(&sweep)
            .unwrap()
            .len();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Ichimoku Oscillator test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_ichimoku_oscillator_outputs_device(&sweep)
            .expect("all Ichimoku Oscillator outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "ichimoku_oscillator");
        assert_eq!(result.entry_point, "ichimoku_oscillator_batch_f64");
        assert_eq!(result.rows, expected_rows);
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ICHIMOKU_OSCILLATOR_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == expected_rows
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Ichimoku Oscillator launch must retire");
    }

    #[test]
    fn ichimoku_oscillator_all_outputs_match_cpu_bits() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::ichimoku_oscillator::{
            IchimokuOscillatorInput, IchimokuOscillatorNormalizeMode, expand_grid,
            ichimoku_oscillator,
        };

        let ohlcv = ichimoku_oscillator_parity_fixture();
        let sweeps = [
            IchimokuOscillatorBatchRange {
                conversion_periods: (3, 5, 2),
                base_periods: (7, 7, 0),
                lagging_span_periods: (11, 11, 0),
                displacement: (4, 4, 0),
                ma_length: (3, 3, 0),
                smoothing_length: (2, 2, 0),
                window_size: (5, 5, 0),
                top_band: (2.0, 2.0, 0.0),
                mid_band: (1.5, 1.5, 0.0),
                extra_smoothing: true,
                normalize: IchimokuOscillatorNormalizeMode::Window,
                clamp: true,
            },
            IchimokuOscillatorBatchRange {
                conversion_periods: (4, 4, 0),
                base_periods: (8, 8, 0),
                lagging_span_periods: (13, 13, 0),
                displacement: (5, 5, 0),
                ma_length: (4, 4, 0),
                smoothing_length: (3, 3, 0),
                window_size: (7, 7, 0),
                top_band: (2.5, 2.5, 0.0),
                mid_band: (0.75, 0.75, 0.0),
                extra_smoothing: false,
                normalize: IchimokuOscillatorNormalizeMode::All,
                clamp: false,
            },
            IchimokuOscillatorBatchRange {
                conversion_periods: (1, 1, 0),
                base_periods: (1, 1, 0),
                lagging_span_periods: (1, 1, 0),
                displacement: (1, 1, 0),
                ma_length: (1, 1, 0),
                smoothing_length: (1, 1, 0),
                window_size: (5, 5, 0),
                top_band: (2.0, 2.0, 0.0),
                mid_band: (1.5, 1.5, 0.0),
                extra_smoothing: false,
                normalize: IchimokuOscillatorNormalizeMode::Disabled,
                clamp: true,
            },
            // Same smoothing path as the Window case, without normalization.
            // This distinguishes raw Chebyshev/Gaussian drift from RMS drift.
            IchimokuOscillatorBatchRange {
                conversion_periods: (3, 5, 2),
                base_periods: (7, 7, 0),
                lagging_span_periods: (11, 11, 0),
                displacement: (4, 4, 0),
                ma_length: (3, 3, 0),
                smoothing_length: (2, 2, 0),
                window_size: (5, 5, 0),
                top_band: (2.0, 2.0, 0.0),
                mid_band: (1.5, 1.5, 0.0),
                extra_smoothing: true,
                normalize: IchimokuOscillatorNormalizeMode::Disabled,
                clamp: true,
            },
            // Same row without Gaussian post-smoothing isolates the
            // Chebyshev coefficient and recurrence.
            IchimokuOscillatorBatchRange {
                conversion_periods: (3, 5, 2),
                base_periods: (7, 7, 0),
                lagging_span_periods: (11, 11, 0),
                displacement: (4, 4, 0),
                ma_length: (3, 3, 0),
                smoothing_length: (2, 2, 0),
                window_size: (5, 5, 0),
                top_band: (2.0, 2.0, 0.0),
                mid_band: (1.5, 1.5, 0.0),
                extra_smoothing: false,
                normalize: IchimokuOscillatorNormalizeMode::Disabled,
                clamp: true,
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Ichimoku Oscillator parity test requires CUDA device 0");

        // Exercise the unnormalized isolation rows first so a future failure
        // identifies raw smoothing before RMS/normalization.
        for sweep in sweeps.into_iter().rev() {
            let parameter_rows = expand_grid(&sweep).unwrap();
            let result = engine
                .compute_ichimoku_oscillator_outputs_device(&sweep)
                .expect("the resident route must accept the exact sweep");
            let mut expected: BTreeMap<&str, Vec<f64>> = ICHIMOKU_OSCILLATOR_OUTPUT_IDS
                .into_iter()
                .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
                .collect();

            for params in parameter_rows {
                let output = ichimoku_oscillator(&IchimokuOscillatorInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    &ohlcv.close,
                    params,
                ))
                .expect("the scalar parity reference must accept the same row");
                macro_rules! extend_output {
                    ($id:literal, $field:ident) => {
                        expected.get_mut($id).unwrap().extend(output.$field)
                    };
                }
                extend_output!("signal", signal);
                extend_output!("ma", ma);
                extend_output!("conversion", conversion);
                extend_output!("base", base);
                extend_output!("chikou", chikou);
                extend_output!("current_kumo_a", current_kumo_a);
                extend_output!("current_kumo_b", current_kumo_b);
                extend_output!("future_kumo_a", future_kumo_a);
                extend_output!("future_kumo_b", future_kumo_b);
                extend_output!("max_level", max_level);
                extend_output!("high_level", high_level);
                extend_output!("low_level", low_level);
                extend_output!("min_level", min_level);
            }

            for output in &result.outputs {
                let actual = engine
                    .runtime
                    .download_matrix_f64(&output.matrix)
                    .expect("each resident Ichimoku output must download for exact parity");
                let cpu = expected
                    .get(output.output_id)
                    .unwrap_or_else(|| panic!("unexpected output {}", output.output_id));
                assert_eq!(actual.len(), cpu.len());
                for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                    if cpu.is_nan() {
                        assert!(
                            gpu.is_nan(),
                            "{}[{index}] lost CPU undefined state: gpu={gpu:?}",
                            output.output_id
                        );
                    } else {
                        assert_eq!(
                            gpu.to_bits(),
                            cpu.to_bits(),
                            "{}[{index}] is not bit-identical in {:?}: gpu={gpu:?} cpu={cpu:?}",
                            output.output_id,
                            sweep.normalize
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn range_filtered_trend_signals_all_outputs_stay_resident() {
        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let parameter_rows = vec![
            RangeFilteredTrendSignalsParams::default(),
            RangeFilteredTrendSignalsParams {
                kalman_alpha: Some(0.03),
                kalman_beta: Some(0.2),
                kalman_period: Some(31),
                dev: Some(0.0),
                supertrend_factor: Some(0.0),
                supertrend_atr_period: Some(1),
            },
            RangeFilteredTrendSignalsParams {
                kalman_alpha: Some(0.25),
                kalman_beta: Some(0.0),
                kalman_period: Some(1),
                dev: Some(2.5),
                supertrend_factor: Some(1.3),
                supertrend_atr_period: Some(13),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Range Filtered Trend Signals test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_range_filtered_trend_signals_outputs_device(&parameter_rows)
            .expect("all Range Filtered Trend Signals outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "range_filtered_trend_signals");
        assert_eq!(result.entry_point, "range_filtered_trend_signals_batch_f64");
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            RANGE_FILTERED_TREND_SIGNALS_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Range Filtered Trend Signals launch must retire");
    }

    #[test]
    fn range_filtered_trend_signals_all_outputs_match_cpu_bits() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::range_filtered_trend_signals::{
            RangeFilteredTrendSignalsInput, range_filtered_trend_signals,
        };

        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let parameter_rows = vec![
            RangeFilteredTrendSignalsParams::default(),
            RangeFilteredTrendSignalsParams {
                kalman_alpha: Some(0.03),
                kalman_beta: Some(0.2),
                kalman_period: Some(31),
                dev: Some(0.0),
                supertrend_factor: Some(0.0),
                supertrend_atr_period: Some(1),
            },
            RangeFilteredTrendSignalsParams {
                kalman_alpha: Some(0.25),
                kalman_beta: Some(0.0),
                kalman_period: Some(1),
                dev: Some(2.5),
                supertrend_factor: Some(1.3),
                supertrend_atr_period: Some(13),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Range Filtered Trend Signals parity test requires CUDA device 0");
        let result = engine
            .compute_range_filtered_trend_signals_outputs_device(&parameter_rows)
            .expect("the resident route must accept the exact parameter rows");

        let mut expected: BTreeMap<&str, Vec<f64>> = RANGE_FILTERED_TREND_SIGNALS_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for params in parameter_rows {
            let output =
                range_filtered_trend_signals(&RangeFilteredTrendSignalsInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    params,
                ))
                .expect("the scalar parity reference must accept the same row");
            macro_rules! extend_output {
                ($id:literal, $field:ident) => {
                    expected.get_mut($id).unwrap().extend(output.$field)
                };
            }
            extend_output!("kalman", kalman);
            extend_output!("supertrend", supertrend);
            extend_output!("upper_band", upper_band);
            extend_output!("lower_band", lower_band);
            extend_output!("trend", trend);
            extend_output!("kalman_trend", kalman_trend);
            extend_output!("state", state);
            extend_output!("market_trending", market_trending);
            extend_output!("market_ranging", market_ranging);
            extend_output!("short_term_bullish", short_term_bullish);
            extend_output!("short_term_bearish", short_term_bearish);
            extend_output!("long_term_bullish", long_term_bullish);
            extend_output!("long_term_bearish", long_term_bearish);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Range Filtered Trend Signals output must download");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not bit-identical: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn range_filtered_trend_signals_coincident_bands_match_hand_derived_state() {
        use vector_ta::indicators::range_filtered_trend_signals::{
            RangeFilteredTrendSignalsInput, range_filtered_trend_signals,
        };

        const BARS: usize = 205;
        let mut values = vec![10.0; BARS];
        values[200..].fill(11.0);
        let ohlcv = Ohlcv {
            timestamp: None,
            open: values.clone(),
            high: values.clone(),
            low: values.clone(),
            close: values,
            volume: Some(vec![1.0; BARS]),
        };
        let params = RangeFilteredTrendSignalsParams {
            // `1.0 + MIN_POSITIVE == 1.0`, so the Kalman gain is exactly one
            // while the public strictly-positive alpha contract is preserved.
            kalman_alpha: Some(f64::MIN_POSITIVE),
            kalman_beta: Some(1.0),
            kalman_period: Some(1),
            dev: Some(0.0),
            supertrend_factor: Some(0.0),
            supertrend_atr_period: Some(1),
        };
        let cpu = range_filtered_trend_signals(&RangeFilteredTrendSignalsInput::from_slices(
            &ohlcv.high,
            &ohlcv.low,
            &ohlcv.close,
            params.clone(),
        ))
        .expect("the hand-derived coincident-band row is valid");
        let hand_derived = [-1.0_f64, 1.0, 1.0, -1.0];
        for (offset, expected) in hand_derived.into_iter().enumerate() {
            assert_eq!(
                cpu.kalman_trend[199 + offset].to_bits(),
                expected.to_bits(),
                "the scalar lane violates the hand-derived state at bar {}",
                199 + offset
            );
        }

        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the coincident-band state test requires CUDA device 0");
        let result = engine
            .compute_range_filtered_trend_signals_outputs_device(&[params])
            .expect("the resident route must accept the coincident-band row");
        let output = result
            .outputs
            .iter()
            .find(|output| output.output_id == "kalman_trend")
            .expect("kalman_trend must remain a named resident output");
        let gpu = engine
            .runtime
            .download_matrix_f64(&output.matrix)
            .expect("the resident state output must download for the oracle check");

        for index in 199..203 {
            assert_eq!(
                gpu[index].to_bits(),
                cpu.kalman_trend[index].to_bits(),
                "CUDA used a direction proxy instead of previous-output identity at bar {index}: gpu={:?} cpu={:?}",
                gpu[index],
                cpu.kalman_trend[index]
            );
        }
    }

    #[test]
    fn ict_propulsion_block_all_outputs_stay_resident() {
        use vector_ta::indicators::ict_propulsion_block::{
            IctPropulsionBlockMitigationPrice, IctPropulsionBlockParams,
        };

        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let parameter_rows = vec![
            IctPropulsionBlockParams::default(),
            IctPropulsionBlockParams {
                swing_length: Some(3),
                mitigation_price: Some(IctPropulsionBlockMitigationPrice::Wick),
            },
            IctPropulsionBlockParams {
                swing_length: Some(7),
                mitigation_price: Some(IctPropulsionBlockMitigationPrice::Close),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the ICT Propulsion Block test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_ict_propulsion_block_outputs_device(&parameter_rows)
            .expect("all ICT Propulsion Block outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(result.indicator_id, "ict_propulsion_block");
        assert_eq!(result.entry_point, "ict_propulsion_block_batch_f64");
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ICT_PROPULSION_BLOCK_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident ICT Propulsion Block launch must retire");
    }

    #[test]
    fn ict_propulsion_block_all_outputs_match_cpu_bits() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::ict_propulsion_block::{
            IctPropulsionBlockInput, IctPropulsionBlockMitigationPrice, IctPropulsionBlockParams,
            ict_propulsion_block,
        };

        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let parameter_rows = vec![
            IctPropulsionBlockParams::default(),
            IctPropulsionBlockParams {
                swing_length: Some(3),
                mitigation_price: Some(IctPropulsionBlockMitigationPrice::Wick),
            },
            IctPropulsionBlockParams {
                swing_length: Some(7),
                mitigation_price: Some(IctPropulsionBlockMitigationPrice::Close),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the ICT Propulsion Block parity test requires CUDA device 0");
        let result = engine
            .compute_ict_propulsion_block_outputs_device(&parameter_rows)
            .expect("the resident route must accept the exact parameter rows");

        let mut expected: BTreeMap<&str, Vec<f64>> = ICT_PROPULSION_BLOCK_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for params in parameter_rows {
            let output = ict_propulsion_block(&IctPropulsionBlockInput::from_slices(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                params,
            ))
            .expect("the scalar parity reference must accept the same row");
            macro_rules! extend_output {
                ($id:literal, $field:ident) => {
                    expected.get_mut($id).unwrap().extend(output.$field)
                };
            }
            extend_output!("bullish_high", bullish_high);
            extend_output!("bullish_low", bullish_low);
            extend_output!("bullish_kind", bullish_kind);
            extend_output!("bullish_active", bullish_active);
            extend_output!("bullish_mitigated", bullish_mitigated);
            extend_output!("bullish_new", bullish_new);
            extend_output!("bearish_high", bearish_high);
            extend_output!("bearish_low", bearish_low);
            extend_output!("bearish_kind", bearish_kind);
            extend_output!("bearish_active", bearish_active);
            extend_output!("bearish_mitigated", bearish_mitigated);
            extend_output!("bearish_new", bearish_new);
        }
        assert!(
            expected["bullish_new"].iter().any(|&value| value == 1.0)
                || expected["bearish_new"].iter().any(|&value| value == 1.0),
            "the parity fixture never exercised an order/propulsion-block insertion"
        );

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident ICT Propulsion Block output must download");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not bit-identical: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn ict_propulsion_block_cuda_matches_hand_derived_mitigation_state() {
        use vector_ta::indicators::ict_propulsion_block::{
            IctPropulsionBlockMitigationPrice, IctPropulsionBlockParams,
        };

        let ohlcv = Ohlcv {
            timestamp: None,
            open: vec![7.0, 8.0, 8.0, 10.0, 7.0],
            high: vec![10.0, 11.0, 9.0, 12.0, 8.0],
            low: vec![5.0, 6.0, 6.0, 9.0, 5.5],
            close: vec![8.0, 9.0, 7.0, 11.5, 6.5],
            volume: Some(vec![1.0; 5]),
        };
        let parameter_rows = [
            IctPropulsionBlockParams {
                swing_length: Some(1),
                mitigation_price: Some(IctPropulsionBlockMitigationPrice::Close),
            },
            IctPropulsionBlockParams {
                swing_length: Some(1),
                mitigation_price: Some(IctPropulsionBlockMitigationPrice::Wick),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the hand-derived ICT state test requires CUDA device 0");
        let result = engine
            .compute_ict_propulsion_block_outputs_device(&parameter_rows)
            .expect("the resident route must accept the hand-derived rows");
        let download = |output_id| {
            let output = result
                .outputs
                .iter()
                .find(|output| output.output_id == output_id)
                .unwrap_or_else(|| panic!("missing resident output {output_id}"));
            engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("download {output_id}: {error}"))
        };
        let bullish_high = download("bullish_high");
        let bullish_low = download("bullish_low");
        let bullish_kind = download("bullish_kind");
        let bullish_active = download("bullish_active");
        let bullish_mitigated = download("bullish_mitigated");
        let bullish_new = download("bullish_new");

        for row in 0..2 {
            let base = row * ohlcv.len();
            assert_eq!(bullish_high[base + 3].to_bits(), 9.0_f64.to_bits());
            assert_eq!(bullish_low[base + 3].to_bits(), 6.0_f64.to_bits());
            assert_eq!(bullish_kind[base + 3].to_bits(), 1.0_f64.to_bits());
            assert_eq!(bullish_active[base + 3].to_bits(), 1.0_f64.to_bits());
            assert_eq!(bullish_mitigated[base + 3].to_bits(), 0.0_f64.to_bits());
            assert_eq!(bullish_new[base + 3].to_bits(), 1.0_f64.to_bits());
            assert_eq!(bullish_new[base + 4].to_bits(), 0.0_f64.to_bits());
        }
        assert_eq!(bullish_mitigated[4].to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            bullish_mitigated[ohlcv.len() + 4].to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn vdubus_all_outputs_stay_resident() {
        use vector_ta::indicators::vdubus_divergence_wave_pattern_generator::VdubusDivergenceWavePatternGeneratorParams;

        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let parameter_rows = vec![
            VdubusDivergenceWavePatternGeneratorParams::default(),
            VdubusDivergenceWavePatternGeneratorParams {
                fast_depth: Some(5),
                slow_depth: Some(13),
                fast_length: Some(8),
                slow_length: Some(21),
                signal_length: Some(5),
                lookback: Some(3),
                ..Default::default()
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Vdubus resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_vdubus_divergence_wave_pattern_generator_outputs_device(&parameter_rows)
            .expect("all Vdubus outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the route re-uploaded input"
        );
        assert_eq!(
            result.indicator_id,
            "vdubus_divergence_wave_pattern_generator"
        );
        assert_eq!(
            result.entry_point,
            "vdubus_divergence_wave_pattern_generator_batch_f64"
        );
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            VDUBUS_DIVERGENCE_WAVE_PATTERN_GENERATOR_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Vdubus launch must retire");
    }

    #[test]
    fn kase_all_unique_outputs_stay_resident_without_the_histogram_alias() {
        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let parameter_rows = vec![
            KasePeakOscillatorWithDivergencesParams::default(),
            KasePeakOscillatorWithDivergencesParams {
                deviations: Some(1.5),
                short_cycle: Some(5),
                long_cycle: Some(34),
                sensitivity: Some(30.0),
                ..Default::default()
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Kase resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_kase_peak_oscillator_with_divergences_outputs_device(&parameter_rows)
            .expect("all unique Kase outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Kase route re-uploaded resident HLC input"
        );
        assert_eq!(result.indicator_id, "kase_peak_oscillator_with_divergences");
        assert_eq!(
            result.entry_point,
            "kase_peak_oscillator_with_divergences_batch_f64"
        );
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            KASE_PEAK_OSCILLATOR_WITH_DIVERGENCES_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Kase launch must retire");
    }

    #[test]
    fn kase_all_unique_outputs_match_the_reviewed_cpu_formula() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::kase_peak_oscillator_with_divergences::{
            KasePeakOscillatorWithDivergencesInput, kase_peak_oscillator_with_divergences,
        };

        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let parameter_rows = vec![
            KasePeakOscillatorWithDivergencesParams::default(),
            KasePeakOscillatorWithDivergencesParams {
                deviations: Some(1.5),
                short_cycle: Some(5),
                long_cycle: Some(34),
                sensitivity: Some(30.0),
                ..Default::default()
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Kase parity test requires CUDA device 0");
        let result = engine
            .compute_kase_peak_oscillator_with_divergences_outputs_device(&parameter_rows)
            .expect("the resident route must accept the exact Kase rows");

        let mut expected: BTreeMap<&str, Vec<f64>> =
            KASE_PEAK_OSCILLATOR_WITH_DIVERGENCES_OUTPUT_IDS
                .into_iter()
                .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
                .collect();
        for params in parameter_rows {
            let output = kase_peak_oscillator_with_divergences(
                &KasePeakOscillatorWithDivergencesInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    params,
                ),
            )
            .expect("the reviewed scalar Kase formula must accept the same row");
            macro_rules! extend_output {
                ($id:literal, $field:ident) => {
                    expected.get_mut($id).unwrap().extend(output.$field)
                };
            }
            extend_output!("oscillator", oscillator);
            extend_output!("max_peak_value", max_peak_value);
            extend_output!("min_peak_value", min_peak_value);
            extend_output!("market_extreme", market_extreme);
            extend_output!("regular_bullish", regular_bullish);
            extend_output!("hidden_bullish", hidden_bullish);
            extend_output!("regular_bearish", regular_bearish);
            extend_output!("hidden_bearish", hidden_bearish);
            extend_output!("go_long", go_long);
            extend_output!("go_short", go_short);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Kase output must download for parity inspection");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected Kase output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    let tolerance = 1e-10 * cpu.abs().max(1.0);
                    assert!(
                        (gpu - cpu).abs() <= tolerance,
                        "{}[{index}] exceeds f64 parity: gpu={gpu:?} cpu={cpu:?} tolerance={tolerance}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn adaptive_bounds_rsi_all_outputs_stay_resident() {
        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.close[104] = f64::NAN;
        let parameter_rows = vec![
            AdaptiveBoundsRsiParams::default(),
            AdaptiveBoundsRsiParams {
                rsi_length: Some(5),
                alpha: Some(0.35),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Adaptive Bounds RSI resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_adaptive_bounds_rsi_outputs_device(&parameter_rows)
            .expect("all Adaptive Bounds RSI outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Adaptive Bounds RSI route re-uploaded resident close input"
        );
        assert_eq!(result.indicator_id, "adaptive_bounds_rsi");
        assert_eq!(result.entry_point, "adaptive_bounds_rsi_batch_f64");
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ADAPTIVE_BOUNDS_RSI_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Adaptive Bounds RSI launch must retire");
    }

    #[test]
    fn adaptive_bounds_rsi_all_outputs_match_the_reviewed_cpu_formula() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::adaptive_bounds_rsi::{
            AdaptiveBoundsRsiInput, adaptive_bounds_rsi,
        };

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.close[104] = f64::NAN;
        let parameter_rows = vec![
            AdaptiveBoundsRsiParams::default(),
            AdaptiveBoundsRsiParams {
                rsi_length: Some(5),
                alpha: Some(0.35),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Adaptive Bounds RSI parity test requires CUDA device 0");
        let result = engine
            .compute_adaptive_bounds_rsi_outputs_device(&parameter_rows)
            .expect("the resident route must accept the exact parameter rows");

        let mut expected: BTreeMap<&str, Vec<f64>> = ADAPTIVE_BOUNDS_RSI_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for params in parameter_rows {
            let output =
                adaptive_bounds_rsi(&AdaptiveBoundsRsiInput::from_slice(&ohlcv.close, params))
                    .expect("the independently reviewed scalar formula must accept the same row");
            macro_rules! extend_output {
                ($id:literal, $field:ident) => {
                    expected.get_mut($id).unwrap().extend(output.$field)
                };
            }
            extend_output!("rsi", rsi);
            extend_output!("lower", lower_bound);
            extend_output!("lower_mid", lower_mid);
            extend_output!("middle", mid);
            extend_output!("upper_mid", upper_mid);
            extend_output!("upper", upper_bound);
            extend_output!("regime", regime);
            extend_output!("regime_flip", regime_flip);
            extend_output!("lower_signal", lower_signal);
            extend_output!("upper_signal", upper_signal);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Adaptive Bounds RSI output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    let tolerance = 1e-10 * cpu.abs().max(1.0);
                    assert!(
                        (gpu - cpu).abs() <= tolerance,
                        "{}[{index}] exceeds f64 parity: gpu={gpu:?} cpu={cpu:?} \
                         tolerance={tolerance}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn adjustable_ma_alternating_extremities_outputs_stay_resident() {
        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let parameter_rows = vec![
            AdjustableMaAlternatingExtremitiesParams::default(),
            AdjustableMaAlternatingExtremitiesParams {
                length: Some(8),
                mult: Some(1.75),
                alpha: Some(1.0),
                beta: Some(0.5),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Adjustable MA resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_adjustable_ma_alternating_extremities_outputs_device(&parameter_rows)
            .expect("all Adjustable MA outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Adjustable MA route re-uploaded resident HLC input"
        );
        assert_eq!(result.indicator_id, "adjustable_ma_alternating_extremities");
        assert_eq!(
            result.entry_point,
            "adjustable_ma_alternating_extremities_batch_f64"
        );
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ADJUSTABLE_MA_ALTERNATING_EXTREMITIES_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Adjustable MA launch must retire");
    }

    #[test]
    fn adjustable_ma_alternating_extremities_outputs_match_reviewed_cpu_after_gap() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::adjustable_ma_alternating_extremities::{
            AdjustableMaAlternatingExtremitiesInput, adjustable_ma_alternating_extremities,
        };

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let parameter_rows = vec![
            AdjustableMaAlternatingExtremitiesParams::default(),
            AdjustableMaAlternatingExtremitiesParams {
                length: Some(8),
                mult: Some(1.75),
                alpha: Some(1.0),
                beta: Some(0.5),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Adjustable MA parity test requires CUDA device 0");
        let result = engine
            .compute_adjustable_ma_alternating_extremities_outputs_device(&parameter_rows)
            .expect("the resident route must accept the exact parameter rows");

        let mut expected: BTreeMap<&str, Vec<f64>> =
            ADJUSTABLE_MA_ALTERNATING_EXTREMITIES_OUTPUT_IDS
                .into_iter()
                .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
                .collect();
        for params in parameter_rows {
            let output = adjustable_ma_alternating_extremities(
                &AdjustableMaAlternatingExtremitiesInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    params,
                ),
            )
            .expect("the reviewed scalar formula must accept the same row");
            macro_rules! extend_output {
                ($id:literal, $field:ident) => {
                    expected.get_mut($id).unwrap().extend(output.$field)
                };
            }
            extend_output!("ma", ma);
            extend_output!("upper", upper);
            extend_output!("lower", lower);
            extend_output!("extremity", extremity);
            extend_output!("state", state);
            extend_output!("changed", changed);
            extend_output!("smoothed_open", smoothed_open);
            extend_output!("smoothed_high", smoothed_high);
            extend_output!("smoothed_low", smoothed_low);
            extend_output!("smoothed_close", smoothed_close);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Adjustable MA output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    let tolerance = 1e-10 * cpu.abs().max(1.0);
                    assert!(
                        (gpu - cpu).abs() <= tolerance,
                        "{}[{index}] exceeds f64 parity: gpu={gpu:?} cpu={cpu:?} \
                         tolerance={tolerance}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn alligator_outputs_stay_resident() {
        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.high[0] = f64::NAN;
        ohlcv.low[0] = f64::NAN;
        let parameter_rows = vec![
            AlligatorParams::default(),
            AlligatorParams {
                jaw_period: Some(21),
                jaw_offset: Some(8),
                teeth_period: Some(13),
                teeth_offset: Some(5),
                lips_period: Some(8),
                lips_offset: Some(3),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Alligator resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_alligator_outputs_device(&parameter_rows)
            .expect("all three Alligator outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Alligator route re-uploaded resident hl2 input"
        );
        assert_eq!(result.indicator_id, "alligator");
        assert_eq!(result.entry_point, "alligator_outputs_f64");
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ALLIGATOR_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Alligator launch must retire");
    }

    #[test]
    fn alligator_outputs_match_cpu_bits_for_default_and_ratio_sweep() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::alligator::{AlligatorInput, alligator};

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.high[0] = f64::NAN;
        ohlcv.low[0] = f64::NAN;
        let hl2 = ohlcv
            .high
            .iter()
            .zip(&ohlcv.low)
            .map(|(high, low)| (high + low) / 2.0)
            .collect::<Vec<_>>();
        let parameter_rows = vec![
            AlligatorParams::default(),
            AlligatorParams {
                jaw_period: Some(21),
                jaw_offset: Some(8),
                teeth_period: Some(13),
                teeth_offset: Some(5),
                lips_period: Some(8),
                lips_offset: Some(3),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Alligator parity test requires CUDA device 0");
        let result = engine
            .compute_alligator_outputs_device(&parameter_rows)
            .expect("the resident route must accept the default and ratio-swept tuples");

        let mut expected: BTreeMap<&str, Vec<f64>> = ALLIGATOR_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for params in parameter_rows {
            let output = alligator(&AlligatorInput::from_slice(&hl2, params))
                .expect("the scalar Alligator formula must accept the same exact tuple");
            expected.get_mut("jaw").unwrap().extend(output.jaw);
            expected.get_mut("teeth").unwrap().extend(output.teeth);
            expected.get_mut("lips").unwrap().extend(output.lips);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Alligator output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected Alligator output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU NaN/shift state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact f64 CPU parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn alphatrend_outputs_stay_resident() {
        use vector_ta::indicators::alphatrend::AlphaTrendParams;

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.close[0] = f64::NAN;
        let volume = ohlcv.volume.as_mut().expect("fixture volume");
        volume[0] = f64::NAN;
        volume[1] = f64::NAN;
        volume[2] = f64::NAN;
        let parameter_rows = vec![
            AlphaTrendParams::default(),
            AlphaTrendParams {
                coeff: Some(1.0),
                period: Some(21),
                no_volume: Some(false),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the AlphaTrend resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_alphatrend_outputs_device(&parameter_rows)
            .expect("both AlphaTrend outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the AlphaTrend route re-uploaded resident OHLCV input"
        );
        assert_eq!(result.indicator_id, "alphatrend");
        assert_eq!(result.entry_point, "alphatrend_outputs_f64");
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ALPHATREND_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident AlphaTrend launch must retire");
    }

    #[test]
    fn alphatrend_outputs_match_cpu_bits_for_default_and_period_sweep() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::alphatrend::{
            AlphaTrendInput, AlphaTrendParams, alphatrend_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.close[0] = f64::NAN;
        let volume = ohlcv.volume.as_mut().expect("fixture volume");
        volume[0] = f64::NAN;
        volume[1] = f64::NAN;
        volume[2] = f64::NAN;
        let parameter_rows = vec![
            AlphaTrendParams::default(),
            AlphaTrendParams {
                coeff: Some(1.0),
                period: Some(21),
                no_volume: Some(false),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the AlphaTrend parity test requires CUDA device 0");
        let result = engine
            .compute_alphatrend_outputs_device(&parameter_rows)
            .expect("the resident route must accept the default and period-swept tuples");

        let volume = ohlcv.volume.as_deref().expect("fixture volume");
        let mut expected: BTreeMap<&str, Vec<f64>> = ALPHATREND_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for params in parameter_rows {
            let output = alphatrend_with_kernel(
                &AlphaTrendInput::from_slices(
                    &ohlcv.open,
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    volume,
                    params,
                ),
                Kernel::Scalar,
            )
            .expect("the scalar AlphaTrend formula must accept the same exact tuple");
            expected.get_mut("k1").unwrap().extend(output.k1);
            expected.get_mut("k2").unwrap().extend(output.k2);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident AlphaTrend output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected AlphaTrend output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/lag state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact f64 CPU parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn bulls_v_bears_outputs_stay_resident() {
        use vector_ta::indicators::bulls_v_bears::{
            BullsVBearsCalculationMethod, BullsVBearsMaType, BullsVBearsParams,
        };

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let parameter_rows = vec![
            BullsVBearsParams::default(),
            BullsVBearsParams {
                period: Some(8),
                ma_type: Some(BullsVBearsMaType::Wma),
                calculation_method: Some(BullsVBearsCalculationMethod::Raw),
                normalized_bars_back: Some(17),
                raw_rolling_period: Some(13),
                raw_threshold_percentile: Some(90.0),
                threshold_level: Some(45.0),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Bulls v Bears resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_bulls_v_bears_outputs_device(&parameter_rows)
            .expect("all Bulls v Bears outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Bulls v Bears route re-uploaded resident HLC input"
        );
        assert_eq!(result.indicator_id, "bulls_v_bears");
        assert_eq!(result.entry_point, "bulls_v_bears_batch_f64");
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            BULLS_V_BEARS_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Bulls v Bears launch must retire");
    }

    #[test]
    fn bulls_v_bears_outputs_match_reviewed_cpu_for_every_mode_after_gap() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::bulls_v_bears::{
            BullsVBearsCalculationMethod, BullsVBearsInput, BullsVBearsMaType, BullsVBearsParams,
            bulls_v_bears,
        };

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let parameter_rows = vec![
            BullsVBearsParams::default(),
            BullsVBearsParams {
                period: Some(7),
                ma_type: Some(BullsVBearsMaType::Ema),
                calculation_method: Some(BullsVBearsCalculationMethod::Raw),
                normalized_bars_back: Some(19),
                raw_rolling_period: Some(11),
                raw_threshold_percentile: Some(90.0),
                threshold_level: Some(40.0),
            },
            BullsVBearsParams {
                period: Some(5),
                ma_type: Some(BullsVBearsMaType::Sma),
                calculation_method: Some(BullsVBearsCalculationMethod::Normalized),
                normalized_bars_back: Some(15),
                raw_rolling_period: Some(9),
                raw_threshold_percentile: Some(85.0),
                threshold_level: Some(50.0),
            },
            BullsVBearsParams {
                period: Some(8),
                ma_type: Some(BullsVBearsMaType::Wma),
                calculation_method: Some(BullsVBearsCalculationMethod::Raw),
                normalized_bars_back: Some(17),
                raw_rolling_period: Some(13),
                raw_threshold_percentile: Some(95.0),
                threshold_level: Some(45.0),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Bulls v Bears parity test requires CUDA device 0");
        let result = engine
            .compute_bulls_v_bears_outputs_device(&parameter_rows)
            .expect("the resident route must accept every supported Bulls v Bears mode");

        let mut expected: BTreeMap<&str, Vec<f64>> = BULLS_V_BEARS_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for params in parameter_rows {
            let output = bulls_v_bears(&BullsVBearsInput::from_slices(
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                params,
            ))
            .expect("the reviewed scalar formula must accept the same row");
            macro_rules! extend_output {
                ($id:literal, $field:ident) => {
                    expected.get_mut($id).unwrap().extend(output.$field)
                };
            }
            extend_output!("value", value);
            extend_output!("bull", bull);
            extend_output!("bear", bear);
            extend_output!("ma", ma);
            extend_output!("upper", upper);
            extend_output!("lower", lower);
            extend_output!("bullish_signal", bullish_signal);
            extend_output!("bearish_signal", bearish_signal);
            extend_output!("zero_cross_up", zero_cross_up);
            extend_output!("zero_cross_down", zero_cross_down);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Bulls v Bears output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    let tolerance = 1e-10 * cpu.abs().max(1.0);
                    assert!(
                        (gpu - cpu).abs() <= tolerance,
                        "{}[{index}] exceeds f64 parity: gpu={gpu:?} cpu={cpu:?} \
                         tolerance={tolerance}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn range_oscillator_outputs_stay_resident_and_use_bar_parallel_work() {
        let ohlcv = range_oscillator_parity_fixture();
        let parameter_rows = vec![
            RangeOscillatorParams::default(),
            RangeOscillatorParams {
                length: Some(3),
                mult: Some(1.5),
            },
            RangeOscillatorParams {
                length: Some(17),
                mult: Some(2.75),
            },
            RangeOscillatorParams {
                length: Some(201),
                mult: Some(0.5),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Range Oscillator resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_range_oscillator_outputs_device(&parameter_rows)
            .expect("every Range Oscillator output must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Range Oscillator route re-uploaded its resident HLC frame"
        );
        assert_eq!(result.indicator_id, "range_oscillator");
        assert_eq!(result.entry_point, "range_oscillator_outputs_f64");
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            RANGE_OSCILLATOR_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        let capacity = engine
            .f64_engine
            .range_oscillator_outputs_launch_capacity(parameter_rows.len(), ohlcv.len())
            .expect("the real output kernel must expose launch-capacity evidence");
        assert_eq!(capacity.entry_point, "range_oscillator_outputs_f64");
        assert_eq!(
            capacity.logical_work_items,
            parameter_rows.len() * ohlcv.len(),
            "the heavy stage regressed to one sequential work item per parameter row"
        );
        assert!(
            capacity.grid_blocks > parameter_rows.len() as u32,
            "the heavy stage exposes only {} blocks for {} rows x {} bars",
            capacity.grid_blocks,
            parameter_rows.len(),
            ohlcv.len()
        );
        engine
            .synchronize()
            .expect("the resident Range Oscillator stages must retire");
    }

    #[test]
    fn range_oscillator_all_outputs_match_reviewed_cpu_bits_after_gap() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::range_oscillator::{
            RangeOscillatorInput, range_oscillator_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let ohlcv = range_oscillator_parity_fixture();
        let parameter_rows = vec![
            RangeOscillatorParams::default(),
            RangeOscillatorParams {
                length: Some(3),
                mult: Some(1.5),
            },
            RangeOscillatorParams {
                length: Some(17),
                mult: Some(2.75),
            },
            RangeOscillatorParams {
                length: Some(201),
                mult: Some(0.5),
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Range Oscillator parity test requires CUDA device 0");
        let result = engine
            .compute_range_oscillator_outputs_device(&parameter_rows)
            .expect("the resident route must accept every reviewed parameter row");

        let mut expected: BTreeMap<&str, Vec<f64>> = RANGE_OSCILLATOR_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for params in parameter_rows {
            let output = range_oscillator_with_kernel(
                &RangeOscillatorInput::from_slices(&ohlcv.high, &ohlcv.low, &ohlcv.close, params),
                Kernel::Scalar,
            )
            .expect("the independently reviewed scalar formula must accept the same row");
            macro_rules! extend_output {
                ($id:literal, $field:ident) => {
                    expected.get_mut($id).unwrap().extend(output.$field)
                };
            }
            extend_output!("oscillator", oscillator);
            extend_output!("ma", ma);
            extend_output!("upper_band", upper_band);
            extend_output!("lower_band", lower_band);
            extend_output!("range_width", range_width);
            extend_output!("in_range", in_range);
            extend_output!("trend", trend);
            extend_output!("break_up", break_up);
            extend_output!("break_down", break_down);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Range Oscillator output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] changed an exact f64 operation: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn pivot_outputs_stay_resident_and_use_bar_parallel_work() {
        let ohlcv = range_oscillator_parity_fixture();
        let parameter_rows: Vec<PivotParams> = (0..=4)
            .map(|mode| PivotParams { mode: Some(mode) })
            .collect();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Pivot resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_pivot_outputs_device(&parameter_rows)
            .expect("every published Pivot output must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Pivot route re-uploaded its resident OHLC frame"
        );
        assert_eq!(result.indicator_id, "pivot");
        assert_eq!(result.entry_point, "pivot_outputs_f64");
        assert_eq!(result.rows, parameter_rows.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            PIVOT_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        let capacity = engine
            .f64_engine
            .pivot_outputs_launch_capacity(parameter_rows.len(), ohlcv.len())
            .expect("the real Pivot kernel must expose launch-capacity evidence");
        assert_eq!(capacity.entry_point, "pivot_outputs_f64");
        assert_eq!(
            capacity.logical_work_items,
            parameter_rows.len() * ohlcv.len(),
            "Pivot regressed to one sequential work item per formula row"
        );
        assert!(
            capacity.grid_blocks > parameter_rows.len() as u32,
            "Pivot exposes only {} blocks for {} rows x {} bars",
            capacity.grid_blocks,
            parameter_rows.len(),
            ohlcv.len()
        );
        engine
            .synchronize()
            .expect("the resident Pivot launch must retire");
    }

    #[test]
    fn pivot_all_outputs_match_the_reviewed_cpu_bits_across_gaps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::pivot::{PivotInput, pivot_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let ohlcv = range_oscillator_parity_fixture();
        let parameter_rows: Vec<PivotParams> = (0..=4)
            .map(|mode| PivotParams { mode: Some(mode) })
            .collect();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Pivot parity test requires CUDA device 0");
        let result = engine
            .compute_pivot_outputs_device(&parameter_rows)
            .expect("the resident route must accept all five reviewed formulas");

        let mut expected: BTreeMap<&str, Vec<f64>> = PIVOT_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for params in parameter_rows {
            let output = pivot_with_kernel(
                &PivotInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    &ohlcv.open,
                    params,
                ),
                Kernel::Scalar,
            )
            .expect("the independently reviewed Pivot formula must accept the same row");
            macro_rules! extend_output {
                ($id:literal, $field:ident) => {
                    expected.get_mut($id).unwrap().extend(output.$field)
                };
            }
            extend_output!("r4", r4);
            extend_output!("r3", r3);
            extend_output!("r2", r2);
            extend_output!("r1", r1);
            extend_output!("pp", pp);
            extend_output!("s1", s1);
            extend_output!("s2", s2);
            extend_output!("s3", s3);
            extend_output!("s4", s4);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Pivot output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] changed an exact f64 operation: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn pivot_primary_route_is_previous_period_f64_and_not_a_sequential_bar_loop() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;
        use vector_ta::indicators::pivot::{PivotInput, PivotParams, pivot_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        assert!(
            !F64Kernel::Pivot.is_sequential(),
            "the period-invariant Pivot primary must launch one CUDA work item per bar"
        );

        let ohlcv = range_oscillator_parity_fixture();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Pivot primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("pivot", &[1])
            .expect("the generic Pivot primary route must stay resident");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the primary Pivot matrix must download for parity"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Pivot unexpectedly materialized a host result")
            }
        };
        let expected = pivot_with_kernel(
            &PivotInput::from_slices(
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                &ohlcv.open,
                PivotParams { mode: Some(3) },
            ),
            Kernel::Scalar,
        )
        .expect("the reviewed CPU Pivot formula must accept the fixture")
        .pp;

        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "pp[{index}] lost CPU undefined state: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "pp[{index}] is not the exact reviewed previous-period f64 result"
                );
            }
        }
    }

    #[test]
    fn acosc_outputs_stay_resident_and_use_one_exact_state_thread() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Acosc.is_sequential(),
            "ACOSC's rolling ring state must execute in exact row order"
        );
        let ohlcv = range_oscillator_parity_fixture();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the ACOSC resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_acosc_outputs_device()
            .expect("both published ACOSC outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the ACOSC route re-uploaded its resident high/low frame"
        );
        assert_eq!(result.indicator_id, "acosc");
        assert_eq!(result.entry_point, "acosc_outputs_f64");
        assert_eq!(result.rows, 1);
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ACOSC_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 1
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident ACOSC launch must retire");
    }

    #[test]
    fn acosc_outputs_match_cpu_bits_across_a_gap() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::acosc::{AcoscInput, AcoscParams, acosc_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let ohlcv = range_oscillator_parity_fixture();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the ACOSC parity test requires CUDA device 0");
        let result = engine
            .compute_acosc_outputs_device()
            .expect("the resident route must produce both reviewed ACOSC outputs");
        let cpu = acosc_with_kernel(
            &AcoscInput::from_slices(&ohlcv.high, &ohlcv.low, AcoscParams::default()),
            Kernel::Scalar,
        )
        .expect("the independently reviewed ACOSC formula must accept the fixture");
        let expected = BTreeMap::from([
            ("osc", cpu.osc.as_slice()),
            ("change", cpu.change.as_slice()),
        ]);

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident ACOSC output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(*cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not the exact CPU rolling-state f64 result",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn andean_oscillator_outputs_stay_resident_for_default_and_length_sweep() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::AndeanOscillator.is_sequential(),
            "each Andean parameter tuple owns one exact rolling-state thread"
        );
        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.open[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let parameter_tuples = [(50, 9), (100, 9)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Andean Oscillator resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_andean_oscillator_outputs_device(&parameter_tuples)
            .expect("all Andean Oscillator outputs must stay resident");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Andean Oscillator route re-uploaded resident open/close input"
        );
        assert_eq!(result.indicator_id, "andean_oscillator");
        assert_eq!(result.entry_point, "andean_oscillator_batch_f64");
        assert_eq!(result.rows, parameter_tuples.len());
        assert_eq!(result.cols, ohlcv.len());
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ANDEAN_OSCILLATOR_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident Andean Oscillator launch must retire");
    }

    #[test]
    fn andean_oscillator_outputs_match_cpu_bits_across_nonfinite_gaps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::andean_oscillator::{
            AndeanOscillatorInput, AndeanOscillatorParams, andean_oscillator,
        };

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.open[90] = f64::NAN;
        ohlcv.close[140] = f64::INFINITY;
        let parameter_tuples = [(50, 9), (100, 9)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Andean Oscillator parity test requires CUDA device 0");
        let result = engine
            .compute_andean_oscillator_outputs_device(&parameter_tuples)
            .expect("the resident route must accept default and length-only sweep tuples");

        let mut expected: BTreeMap<&str, Vec<f64>> = ANDEAN_OSCILLATOR_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (length, signal_length) in parameter_tuples {
            let output = andean_oscillator(&AndeanOscillatorInput::from_slices(
                &ohlcv.open,
                &ohlcv.close,
                AndeanOscillatorParams {
                    length: Some(length),
                    signal_length: Some(signal_length),
                },
            ))
            .expect("the scalar Andean formula must accept the exact production tuple");
            expected.get_mut("bull").unwrap().extend(output.bull);
            expected.get_mut("bear").unwrap().extend(output.bear);
            expected.get_mut("signal").unwrap().extend(output.signal);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Andean output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected Andean output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/state-carry behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn aroon_outputs_stay_resident_for_default_and_all_length_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Aroon.is_sequential(),
            "each Aroon length owns one exact rolling-extreme state thread"
        );
        assert!(
            !F64Kernel::Aroon.is_period_invariant(),
            "Aroon must consume every admitted length"
        );
        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let lengths = [14, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Aroon resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_aroon_outputs_device(&lengths)
            .expect("both Aroon outputs must stay resident for every admitted length");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Aroon route re-uploaded its resident high/low input"
        );
        assert_eq!(result.indicator_id, "aroon");
        assert_eq!(result.entry_point, "aroon_outputs_f64");
        assert_eq!((result.rows, result.cols), (lengths.len(), ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            AROON_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == lengths.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident Aroon launch must retire");
    }

    #[test]
    fn aroon_outputs_match_cpu_bits_for_default_and_all_sweeps_across_gaps_and_ties() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::aroon::{AroonInput, AroonParams, aroon};

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        for index in 60..=64 {
            ohlcv.high[index] = ohlcv.high[59];
            ohlcv.low[index] = ohlcv.low[59];
        }
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.high[280] = f64::INFINITY;
        let lengths = [14, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Aroon parity test requires CUDA device 0");
        let result = engine
            .compute_aroon_outputs_device(&lengths)
            .expect("the resident Aroon route must accept the default and all five sweeps");

        let mut expected: BTreeMap<&str, Vec<f64>> = AROON_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for length in lengths {
            let output = aroon(&AroonInput::from_slices_hl(
                &ohlcv.high,
                &ohlcv.low,
                AroonParams {
                    length: Some(length),
                },
            ))
            .expect("the scalar Aroon formula must accept every admitted length");
            expected.get_mut("up").unwrap().extend(output.aroon_up);
            expected.get_mut("down").unwrap().extend(output.aroon_down);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Aroon output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected Aroon output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn aso_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Aso.is_sequential(),
            "each ASO tuple owns one exact running-mean state thread"
        );
        assert!(
            !F64Kernel::Aso.is_period_invariant(),
            "ASO must consume every admitted period"
        );
        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let parameter_tuples = [(10, 0), (7, 0), (21, 0), (50, 0), (100, 0), (200, 0)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the ASO resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_aso_outputs_device(&parameter_tuples)
            .expect("both ASO outputs must stay resident for every admitted tuple");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the ASO route re-uploaded its resident OHLC input"
        );
        assert_eq!(result.indicator_id, "aso");
        assert_eq!(result.entry_point, "neoethos_aso_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_tuples.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ASO_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident ASO launch must retire");
    }

    #[test]
    fn aso_outputs_match_cpu_bits_for_default_and_all_sweeps_across_sentinels_and_ties() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::aso::{
            AsoInput, AsoOutputField, AsoParams, aso_output_into_slice,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.close[0] = f64::NAN;
        ohlcv.high[1] = f64::NAN;
        ohlcv.low[2] = f64::NAN;
        for index in 60..=64 {
            ohlcv.high[index] = ohlcv.high[59];
            ohlcv.low[index] = ohlcv.low[59];
        }
        ohlcv.high[150] = ohlcv.low[150];
        ohlcv.high[410] = f64::INFINITY;
        let parameter_tuples = [(10, 0), (7, 0), (21, 0), (50, 0), (100, 0), (200, 0)];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the ASO parity test requires CUDA device 0");
        let result = engine
            .compute_aso_outputs_device(&parameter_tuples)
            .expect("the resident ASO route must accept the default and all five sweeps");

        let mut expected: BTreeMap<&str, Vec<f64>> = ASO_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (period, mode) in parameter_tuples {
            let input = AsoInput::from_slices(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                AsoParams {
                    period: Some(period),
                    mode: Some(mode),
                },
            );
            let mut bulls = vec![f64::NAN; ohlcv.len()];
            let mut bears = vec![f64::NAN; ohlcv.len()];
            aso_output_into_slice(&mut bulls, &input, Kernel::Scalar, AsoOutputField::Bulls)
                .expect("the scalar ASO bulls formula must accept every admitted tuple");
            aso_output_into_slice(&mut bears, &input, Kernel::Scalar, AsoOutputField::Bears)
                .expect("the scalar ASO bears formula must accept every admitted tuple");
            expected.get_mut("bulls").unwrap().extend(bulls);
            expected.get_mut("bears").unwrap().extend(bears);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident ASO output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected ASO output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU sentinel/undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn autocorrelation_indicator_outputs_stay_resident_for_default_and_all_length_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::AutocorrelationIndicator.is_sequential(),
            "each ACI tuple owns one exact smoother/correlation state thread"
        );
        assert!(
            !F64Kernel::AutocorrelationIndicator.is_period_invariant(),
            "ACI must consume every admitted length"
        );
        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let parameter_tuples = [
            (20, 1, false),
            (7, 1, false),
            (21, 1, false),
            (50, 1, false),
            (100, 1, false),
            (200, 1, false),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the ACI resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_autocorrelation_indicator_outputs_device(&parameter_tuples)
            .expect("both ACI outputs must stay resident for every admitted tuple");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the ACI route re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "autocorrelation_indicator");
        assert_eq!(result.entry_point, "autocorrelation_indicator_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_tuples.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            AUTOCORRELATION_INDICATOR_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident ACI launch must retire");
    }

    #[test]
    fn autocorrelation_indicator_outputs_match_cpu_bits_for_default_and_all_length_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::autocorrelation_indicator::{
            AutocorrelationIndicatorInput, AutocorrelationIndicatorOutputField,
            AutocorrelationIndicatorParams, autocorrelation_indicator_output_into_slice,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.close[101] = f64::NAN;
        let parameter_tuples = [
            (20, 1, false),
            (7, 1, false),
            (21, 1, false),
            (50, 1, false),
            (100, 1, false),
            (200, 1, false),
        ];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the ACI parity test requires CUDA device 0");
        let result = engine
            .compute_autocorrelation_indicator_outputs_device(&parameter_tuples)
            .expect("the resident ACI route must accept the default and all five sweeps");

        let mut expected: BTreeMap<&str, Vec<f64>> = AUTOCORRELATION_INDICATOR_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (length, lag, use_test_signal) in parameter_tuples {
            let input = AutocorrelationIndicatorInput::from_slice(
                &ohlcv.close,
                AutocorrelationIndicatorParams {
                    length: Some(length),
                    max_lag: Some(lag),
                    use_test_signal: Some(use_test_signal),
                },
            );
            let mut filtered = vec![f64::NAN; ohlcv.len()];
            let mut correlation = vec![f64::NAN; ohlcv.len()];
            autocorrelation_indicator_output_into_slice(
                &mut filtered,
                &input,
                Kernel::Scalar,
                AutocorrelationIndicatorOutputField::Filtered,
            )
            .expect("the scalar ACI filter must accept every admitted tuple");
            autocorrelation_indicator_output_into_slice(
                &mut correlation,
                &input,
                Kernel::Scalar,
                AutocorrelationIndicatorOutputField::Correlation { lag },
            )
            .expect("the scalar ACI correlation must accept every admitted tuple");
            expected.get_mut("filtered").unwrap().extend(filtered);
            expected.get_mut("correlation").unwrap().extend(correlation);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident ACI output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected ACI output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn autocorrelation_indicator_primary_route_consumes_each_length_and_matches_filtered_cpu_bits()
    {
        use vector_ta::indicators::autocorrelation_indicator::{
            AutocorrelationIndicatorInput, AutocorrelationIndicatorOutputField,
            AutocorrelationIndicatorParams, autocorrelation_indicator_output_into_slice,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.close[101] = f64::NAN;
        let lengths = [20, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the ACI primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("autocorrelation_indicator", &lengths)
            .expect("the generic ACI primary route must consume every admitted length");
        assert_eq!(output.output_id, "filtered");
        assert_eq!((output.rows, output.cols), (lengths.len(), ohlcv.len()));
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident ACI primary matrix must download for parity"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly ACI unexpectedly materialized a host result")
            }
        };

        let mut expected = Vec::with_capacity(actual.len());
        for length in lengths {
            let input = AutocorrelationIndicatorInput::from_slice(
                &ohlcv.close,
                AutocorrelationIndicatorParams {
                    length: Some(length),
                    max_lag: Some(1),
                    use_test_signal: Some(false),
                },
            );
            let mut filtered = vec![f64::NAN; ohlcv.len()];
            autocorrelation_indicator_output_into_slice(
                &mut filtered,
                &input,
                Kernel::Scalar,
                AutocorrelationIndicatorOutputField::Filtered,
            )
            .expect("the scalar ACI primary must accept every admitted length");
            expected.extend(filtered);
        }

        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "filtered[{index}] lost CPU undefined/reset behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "filtered[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn avsl_outputs_stay_resident_for_default_and_all_ratio_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Avsl.is_sequential(),
            "each AVSL tuple owns one exact rolling-state thread"
        );
        assert!(
            !F64Kernel::Avsl.is_period_invariant(),
            "AVSL must consume every admitted slow-period anchor"
        );
        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let parameter_tuples = [
            (12, 26, 2.0),
            (3, 7, 2.0),
            (10, 21, 2.0),
            (23, 50, 2.0),
            (46, 100, 2.0),
            (92, 200, 2.0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the AVSL resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_avsl_outputs_device(&parameter_tuples)
            .expect("AVSL value must stay resident for every exact admitted tuple");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the AVSL route re-uploaded its resident close/low/volume inputs"
        );
        assert_eq!(result.indicator_id, "avsl");
        assert_eq!(result.entry_point, "avsl_production_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_tuples.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            AVSL_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident AVSL launch must retire");
    }

    #[test]
    fn avsl_outputs_match_cpu_bits_for_default_and_all_ratio_sweeps() {
        use vector_ta::indicators::avsl::{AvslInput, AvslParams, avsl_into_slice};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.close[..2].fill(f64::NAN);
        ohlcv.low[..4].fill(f64::NAN);
        ohlcv.volume.as_mut().unwrap()[..6].fill(f64::NAN);
        // Exercise Rust's saturating NaN-to-usize conversion in the adaptive
        // history-length branch after a valid prefix; CUDA must emit the same
        // undefined suffix without an invalid negative-index conversion.
        ohlcv.close[257] = f64::NAN;
        let parameter_tuples = [
            (12, 26, 2.0),
            (3, 7, 2.0),
            (10, 21, 2.0),
            (23, 50, 2.0),
            (46, 100, 2.0),
            (92, 200, 2.0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the AVSL parity test requires CUDA device 0");
        let result = engine
            .compute_avsl_outputs_device(&parameter_tuples)
            .expect("the resident AVSL route must accept the default and all five sweeps");

        let volume = ohlcv.volume.as_ref().unwrap();
        let mut expected = Vec::with_capacity(result.rows * result.cols);
        for (fast_period, slow_period, multiplier) in parameter_tuples {
            let input = AvslInput::from_slices(
                &ohlcv.close,
                &ohlcv.low,
                volume,
                AvslParams {
                    fast_period: Some(fast_period),
                    slow_period: Some(slow_period),
                    multiplier: Some(multiplier),
                },
            );
            let mut values = vec![f64::NAN; ohlcv.len()];
            avsl_into_slice(&mut values, &input, Kernel::Scalar)
                .expect("the scalar AVSL formula must accept every admitted tuple");
            expected.extend(values);
        }

        let output = &result.outputs[0];
        let actual = engine
            .runtime
            .download_matrix_f64(&output.matrix)
            .expect("the resident AVSL output must download for parity");
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "value[{index}] lost CPU sentinel behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "value[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn avsl_primary_route_consumes_slow_anchor_and_matches_cpu_bits() {
        use vector_ta::indicators::avsl::{AvslInput, AvslParams, avsl_into_slice};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = range_filtered_trend_signals_parity_fixture();
        ohlcv.close[..2].fill(f64::NAN);
        ohlcv.low[..4].fill(f64::NAN);
        ohlcv.volume.as_mut().unwrap()[..6].fill(f64::NAN);
        let slow_periods = [26, 7, 21, 50, 100, 200];
        let fast_periods = [12, 3, 10, 23, 46, 92];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the AVSL primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("avsl", &slow_periods)
            .expect("the generic AVSL primary route must consume every slow anchor");
        assert_eq!(output.output_id, "value");
        assert_eq!(
            (output.rows, output.cols),
            (slow_periods.len(), ohlcv.len())
        );
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident AVSL primary matrix must download for parity"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly AVSL unexpectedly materialized a host result")
            }
        };

        let volume = ohlcv.volume.as_ref().unwrap();
        let mut expected = Vec::with_capacity(actual.len());
        for (fast_period, slow_period) in fast_periods.into_iter().zip(slow_periods) {
            let input = AvslInput::from_slices(
                &ohlcv.close,
                &ohlcv.low,
                volume,
                AvslParams {
                    fast_period: Some(fast_period),
                    slow_period: Some(slow_period),
                    multiplier: Some(2.0),
                },
            );
            let mut values = vec![f64::NAN; ohlcv.len()];
            avsl_into_slice(&mut values, &input, Kernel::Scalar)
                .expect("the scalar AVSL primary must accept every derived tuple");
            expected.extend(values);
        }

        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "value[{index}] lost CPU sentinel behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "value[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn classic_bandpass_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Bandpass.is_sequential(),
            "each Bandpass tuple owns both exact IIR states in one thread"
        );
        assert!(
            !F64Kernel::Bandpass.is_period_invariant(),
            "Bandpass must consume every admitted period anchor"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_tuples = [
            (20, 0.3),
            (7, 0.3),
            (21, 0.3),
            (50, 0.3),
            (100, 0.3),
            (200, 0.3),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Bandpass resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_bandpass_outputs_device(&parameter_tuples)
            .expect("all Bandpass outputs must stay resident for every admitted tuple");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Bandpass route re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "bandpass");
        assert_eq!(result.entry_point, "bandpass_production_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_tuples.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            BANDPASS_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident Bandpass launch must retire");
    }

    #[test]
    fn classic_bandpass_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use vector_ta::indicators::bandpass::{
            BandPassInput, BandPassOutputField, BandPassParams, bandpass_output_into_slice,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let parameter_tuples = [
            (20, 0.3),
            (7, 0.3),
            (21, 0.3),
            (50, 0.3),
            (100, 0.3),
            (200, 0.3),
        ];
        let fields = [
            BandPassOutputField::Bp,
            BandPassOutputField::BpNormalized,
            BandPassOutputField::Signal,
            BandPassOutputField::Trigger,
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Bandpass parity test requires CUDA device 0");
        let result = engine
            .compute_bandpass_outputs_device(&parameter_tuples)
            .expect("the resident Bandpass route must accept the default and all five sweeps");

        for ((output, output_id), field) in
            result.outputs.iter().zip(BANDPASS_OUTPUT_IDS).zip(fields)
        {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("the resident Bandpass output must download for parity");
            let mut expected = Vec::with_capacity(result.rows * result.cols);
            for &(period, bandwidth) in &parameter_tuples {
                let input = BandPassInput::from_slice(
                    &ohlcv.close,
                    BandPassParams {
                        period: Some(period),
                        bandwidth: Some(bandwidth),
                    },
                );
                let mut values = vec![f64::NAN; ohlcv.len()];
                bandpass_output_into_slice(&mut values, &input, Kernel::Scalar, field)
                    .expect("the scalar Bandpass formula must accept every admitted tuple");
                expected.extend(values);
            }
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{output_id}[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{output_id}[{index}] is not exact scalar CPU f64 parity: \
                         gpu={gpu:?} cpu={cpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn classic_bandpass_primary_route_matches_cpu_bits_for_all_periods() {
        use vector_ta::indicators::bandpass::{
            BandPassInput, BandPassOutputField, BandPassParams, bandpass_output_into_slice,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let periods = [20, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Bandpass primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("bandpass", &periods)
            .expect("the generic Bandpass primary route must consume every period anchor");
        assert_eq!(output.output_id, "bp");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Bandpass primary matrix must download for parity"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Bandpass unexpectedly materialized a host result")
            }
        };

        let mut expected = Vec::with_capacity(actual.len());
        for period in periods {
            let input = BandPassInput::from_slice(
                &ohlcv.close,
                BandPassParams {
                    period: Some(period),
                    bandwidth: Some(0.3),
                },
            );
            let mut values = vec![f64::NAN; ohlcv.len()];
            bandpass_output_into_slice(
                &mut values,
                &input,
                Kernel::Scalar,
                BandPassOutputField::Bp,
            )
            .expect("the scalar Bandpass primary must accept every admitted period");
            expected.extend(values);
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "bp[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "bp[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn classic_bollinger_bands_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::BollingerBands.is_sequential(),
            "each Bollinger Bands tuple must preserve the scalar rolling sums in one thread"
        );
        assert!(
            !F64Kernel::BollingerBands.is_period_invariant(),
            "Bollinger Bands must consume every admitted period anchor"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_tuples = [
            (20, 2.0, 2.0),
            (7, 2.0, 2.0),
            (21, 2.0, 2.0),
            (50, 2.0, 2.0),
            (100, 2.0, 2.0),
            (200, 2.0, 2.0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Bollinger Bands resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_bollinger_bands_outputs_device(&parameter_tuples)
            .expect("all Bollinger Bands outputs must stay resident for every admitted tuple");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Bollinger Bands route re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "bollinger_bands");
        assert_eq!(result.entry_point, "bollinger_bands_production_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_tuples.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            BOLLINGER_BANDS_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident Bollinger Bands launch must retire");
    }

    #[test]
    fn classic_bollinger_bands_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use vector_ta::indicators::bollinger_bands::{
            BollingerBandsInput, BollingerBandsParams, bollinger_bands_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let parameter_tuples = [
            (20, 2.0, 2.0),
            (7, 2.0, 2.0),
            (21, 2.0, 2.0),
            (50, 2.0, 2.0),
            (100, 2.0, 2.0),
            (200, 2.0, 2.0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Bollinger Bands parity test requires CUDA device 0");
        let result = engine
            .compute_bollinger_bands_outputs_device(&parameter_tuples)
            .expect("the resident Bollinger Bands route must accept all admitted tuples");

        for (output, output_id) in result.outputs.iter().zip(BOLLINGER_BANDS_OUTPUT_IDS) {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("the resident Bollinger Bands output must download for parity");
            let mut expected = Vec::with_capacity(result.rows * result.cols);
            for &(period, devup, devdn) in &parameter_tuples {
                let input = BollingerBandsInput::from_slice(
                    &ohlcv.close,
                    BollingerBandsParams {
                        period: Some(period),
                        devup: Some(devup),
                        devdn: Some(devdn),
                        matype: Some("sma".to_string()),
                        devtype: Some(0),
                    },
                );
                let computed = bollinger_bands_with_kernel(&input, Kernel::Scalar)
                    .expect("the scalar Bollinger Bands formula must accept every admitted tuple");
                expected.extend(match output_id {
                    "upper" => computed.upper_band,
                    "middle" => computed.middle_band,
                    "lower" => computed.lower_band,
                    _ => unreachable!("the canonical output set is exhaustive"),
                });
            }
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{output_id}[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{output_id}[{index}] is not exact scalar CPU f64 parity: \
                         gpu={gpu:?} cpu={cpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn classic_bollinger_bands_primary_route_matches_cpu_bits_for_all_periods() {
        use vector_ta::indicators::bollinger_bands::{
            BollingerBandsInput, BollingerBandsParams, bollinger_bands_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let periods = [20, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Bollinger Bands primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("bollinger_bands", &periods)
            .expect("the generic Bollinger Bands primary route must consume every period anchor");
        assert_eq!(output.output_id, "upper");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Bollinger Bands primary matrix must download for parity"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Bollinger Bands unexpectedly materialized a host result")
            }
        };

        let mut expected = Vec::with_capacity(actual.len());
        for period in periods {
            let input = BollingerBandsInput::from_slice(
                &ohlcv.close,
                BollingerBandsParams {
                    period: Some(period),
                    devup: Some(2.0),
                    devdn: Some(2.0),
                    matype: Some("sma".to_string()),
                    devtype: Some(0),
                },
            );
            expected.extend(
                bollinger_bands_with_kernel(&input, Kernel::Scalar)
                    .expect("the scalar Bollinger Bands primary must accept every period")
                    .upper_band,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "upper[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "upper[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn buff_averages_outputs_stay_resident_for_default_and_all_ratio_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::BuffAverages.is_sequential(),
            "each Buff Averages tuple must preserve scalar rolling state in one thread"
        );
        assert!(
            !F64Kernel::BuffAverages.is_period_invariant(),
            "Buff Averages must consume both admitted window parameters"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_tuples = [(5, 20), (2, 7), (5, 21), (13, 50), (25, 100), (50, 200)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Buff Averages resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_buff_averages_outputs_device(&parameter_tuples)
            .expect("all Buff Averages outputs must stay resident for every admitted tuple");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Buff Averages route re-uploaded its resident close/volume input"
        );
        assert_eq!(result.indicator_id, "buff_averages");
        assert_eq!(result.entry_point, "buff_averages_production_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_tuples.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            BUFF_AVERAGES_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident Buff Averages launch must retire");
    }

    #[test]
    fn buff_averages_outputs_match_cpu_bits_for_default_and_all_ratio_sweeps() {
        use vector_ta::indicators::moving_averages::buff_averages::{
            BuffAveragesBatchRange, buff_averages_batch_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        ohlcv
            .volume
            .as_mut()
            .expect("the Buff Averages fixture requires volume")[17] = f64::NAN;
        let volume = ohlcv
            .volume
            .as_deref()
            .expect("the Buff Averages fixture requires volume");
        let parameter_tuples = [(5, 20), (2, 7), (5, 21), (13, 50), (25, 100), (50, 200)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Buff Averages parity test requires CUDA device 0");
        let result = engine
            .compute_buff_averages_outputs_device(&parameter_tuples)
            .expect("the resident Buff Averages route must accept all admitted tuples");

        for (output, output_id) in result.outputs.iter().zip(BUFF_AVERAGES_OUTPUT_IDS) {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("the resident Buff Averages output must download for parity");
            let mut expected = Vec::with_capacity(result.rows * result.cols);
            for &(fast_period, slow_period) in &parameter_tuples {
                let computed = buff_averages_batch_with_kernel(
                    &ohlcv.close,
                    volume,
                    &BuffAveragesBatchRange {
                        fast_period: (fast_period, fast_period, 0),
                        slow_period: (slow_period, slow_period, 0),
                    },
                    Kernel::ScalarBatch,
                )
                .expect("the scalar Buff Averages formula must accept every admitted tuple");
                expected.extend(match output_id {
                    "fast" => computed.fast,
                    "slow" => computed.slow,
                    _ => unreachable!("the canonical output set is exhaustive"),
                });
            }
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{output_id}[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{output_id}[{index}] is not exact scalar CPU f64 parity: \
                         gpu={gpu:?} cpu={cpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn buff_averages_primary_route_matches_cpu_bits_for_all_slow_periods() {
        use vector_ta::indicators::moving_averages::buff_averages::{
            BuffAveragesBatchRange, buff_averages_batch_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        ohlcv
            .volume
            .as_mut()
            .expect("the Buff Averages fixture requires volume")[17] = f64::NAN;
        let volume = ohlcv
            .volume
            .as_deref()
            .expect("the Buff Averages fixture requires volume");
        let slow_periods = [20, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Buff Averages primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("buff_averages", &slow_periods)
            .expect("the generic Buff Averages primary route must consume every slow period");
        assert_eq!(output.output_id, "fast");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Buff Averages primary matrix must download for parity"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Buff Averages unexpectedly materialized a host result")
            }
        };

        let mut expected = Vec::with_capacity(actual.len());
        for slow_period in slow_periods {
            expected.extend(
                buff_averages_batch_with_kernel(
                    &ohlcv.close,
                    volume,
                    &BuffAveragesBatchRange {
                        fast_period: (5, 5, 0),
                        slow_period: (slow_period, slow_period, 0),
                    },
                    Kernel::ScalarBatch,
                )
                .expect("the scalar Buff Averages primary must accept every slow period")
                .fast,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "fast[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "fast[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn candle_strength_oscillator_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;
        use vector_ta::indicators::candle_strength_oscillator::CandleStrengthOscillatorParams;

        assert!(
            F64Kernel::CandleStrengthOscillator.is_sequential(),
            "each Candle Strength Oscillator tuple must preserve every rolling state in one thread"
        );
        assert!(
            !F64Kernel::CandleStrengthOscillator.is_period_invariant(),
            "Candle Strength Oscillator must consume every admitted period anchor"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows =
            [50, 7, 21, 50, 100, 200].map(|period| CandleStrengthOscillatorParams {
                period: Some(period),
                atr_enabled: Some(false),
                atr_length: Some(50),
                mode: Some("bollinger".to_string()),
            });
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Candle Strength Oscillator resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_candle_strength_oscillator_outputs_device(&parameter_rows)
            .expect(
                "all Candle Strength Oscillator outputs must stay resident for every admitted tuple",
            );

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Candle Strength Oscillator route re-uploaded its resident OHLC input"
        );
        assert_eq!(result.indicator_id, "candle_strength_oscillator");
        assert_eq!(result.entry_point, "candle_strength_oscillator_batch_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident Candle Strength Oscillator launch must retire");
    }

    #[test]
    fn candle_strength_oscillator_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use vector_ta::indicators::candle_strength_oscillator::{
            CandleStrengthOscillatorInput, CandleStrengthOscillatorParams,
            candle_strength_oscillator_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows =
            [50, 7, 21, 50, 100, 200].map(|period| CandleStrengthOscillatorParams {
                period: Some(period),
                atr_enabled: Some(false),
                atr_length: Some(50),
                mode: Some("bollinger".to_string()),
            });
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Candle Strength Oscillator parity test requires CUDA device 0");
        let result = engine
            .compute_candle_strength_oscillator_outputs_device(&parameter_rows)
            .expect(
                "the resident Candle Strength Oscillator route must accept all admitted tuples",
            );

        for (output, output_id) in result
            .outputs
            .iter()
            .zip(CANDLE_STRENGTH_OSCILLATOR_OUTPUT_IDS)
        {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("the resident Candle Strength Oscillator output must download for parity");
            let mut expected = Vec::with_capacity(result.rows * result.cols);
            for params in &parameter_rows {
                let computed = candle_strength_oscillator_with_kernel(
                    &CandleStrengthOscillatorInput::from_slices(
                        &ohlcv.open,
                        &ohlcv.high,
                        &ohlcv.low,
                        &ohlcv.close,
                        params.clone(),
                    ),
                    Kernel::Scalar,
                )
                .expect(
                    "the scalar Candle Strength Oscillator formula must accept every admitted tuple",
                );
                expected.extend(match output_id {
                    "strength" => computed.strength,
                    "highs" => computed.highs,
                    "lows" => computed.lows,
                    "mid" => computed.mid,
                    "long_signal" => computed.long_signal,
                    "short_signal" => computed.short_signal,
                    _ => unreachable!("the canonical output set is exhaustive"),
                });
            }
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{output_id}[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{output_id}[{index}] is not exact scalar CPU f64 parity: \
                         gpu={gpu:?} cpu={cpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn candle_strength_oscillator_primary_route_matches_cpu_bits_for_all_periods() {
        use vector_ta::indicators::candle_strength_oscillator::{
            CandleStrengthOscillatorInput, CandleStrengthOscillatorParams,
            candle_strength_oscillator_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let ohlcv = repeated_ctrader_bandpass_fixture();
        let periods = [50, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Candle Strength Oscillator primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("candle_strength_oscillator", &periods)
            .expect(
                "the generic Candle Strength Oscillator primary route must consume every period",
            );
        assert_eq!(output.output_id, "strength");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect(
                    "the resident Candle Strength Oscillator primary matrix must download for parity",
                ),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!(
                    "GpuOnly Candle Strength Oscillator unexpectedly materialized a host result"
                )
            }
        };

        let mut expected = Vec::with_capacity(actual.len());
        for period in periods {
            expected.extend(
                candle_strength_oscillator_with_kernel(
                    &CandleStrengthOscillatorInput::from_slices(
                        &ohlcv.open,
                        &ohlcv.high,
                        &ohlcv.low,
                        &ohlcv.close,
                        CandleStrengthOscillatorParams {
                            period: Some(period),
                            atr_enabled: Some(false),
                            atr_length: Some(50),
                            mode: Some("bollinger".to_string()),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar Candle Strength Oscillator primary must accept every period")
                .strength,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "strength[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "strength[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn chandelier_exit_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;
        use vector_ta::indicators::chandelier_exit::ChandelierExitParams;

        assert!(
            F64Kernel::ChandelierExit.is_sequential(),
            "each Chandelier Exit tuple must preserve ATR, deque, and stop state in one thread"
        );
        assert!(
            !F64Kernel::ChandelierExit.is_period_invariant(),
            "Chandelier Exit must consume every admitted period anchor"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [22, 7, 21, 50, 100, 200].map(|period| ChandelierExitParams {
            period: Some(period),
            mult: Some(3.0),
            use_close: Some(true),
        });
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Chandelier Exit resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_chandelier_exit_outputs_device(&parameter_rows)
            .expect("both Chandelier Exit outputs must stay resident for every admitted tuple");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Chandelier Exit route re-uploaded its resident HLC input"
        );
        assert_eq!(result.indicator_id, "chandelier_exit");
        assert_eq!(result.entry_point, "chandelier_exit_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            CHANDELIER_EXIT_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident Chandelier Exit launch must retire");
    }

    #[test]
    fn chandelier_exit_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use vector_ta::indicators::chandelier_exit::{
            ChandelierExitInput, ChandelierExitParams, chandelier_exit_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [22, 7, 21, 50, 100, 200].map(|period| ChandelierExitParams {
            period: Some(period),
            mult: Some(3.0),
            use_close: Some(true),
        });
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Chandelier Exit parity test requires CUDA device 0");
        let result = engine
            .compute_chandelier_exit_outputs_device(&parameter_rows)
            .expect("the resident Chandelier Exit route must accept all admitted tuples");

        for (output, output_id) in result.outputs.iter().zip(CHANDELIER_EXIT_OUTPUT_IDS) {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("the resident Chandelier Exit output must download for parity");
            let mut expected = Vec::with_capacity(result.rows * result.cols);
            for params in &parameter_rows {
                let computed = chandelier_exit_with_kernel(
                    &ChandelierExitInput::from_slices(
                        &ohlcv.high,
                        &ohlcv.low,
                        &ohlcv.close,
                        params.clone(),
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar Chandelier Exit formula must accept every admitted tuple");
                expected.extend(match output_id {
                    "long_stop" => computed.long_stop,
                    "short_stop" => computed.short_stop,
                    _ => unreachable!("the canonical Chandelier Exit output set is exhaustive"),
                });
            }
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{output_id}[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{output_id}[{index}] is not exact scalar CPU f64 parity: \
                         gpu={gpu:?} cpu={cpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn chandelier_exit_primary_route_matches_cpu_bits_for_all_periods() {
        use vector_ta::indicators::chandelier_exit::{
            ChandelierExitInput, ChandelierExitParams, chandelier_exit_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let ohlcv = repeated_ctrader_bandpass_fixture();
        let periods = [22, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Chandelier Exit primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("chandelier_exit", &periods)
            .expect("the generic Chandelier Exit primary route must consume every period");
        assert_eq!(output.output_id, "long_stop");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Chandelier Exit primary matrix must download for parity"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Chandelier Exit unexpectedly materialized a host result")
            }
        };

        let mut expected = Vec::with_capacity(actual.len());
        for period in periods {
            expected.extend(
                chandelier_exit_with_kernel(
                    &ChandelierExitInput::from_slices(
                        &ohlcv.high,
                        &ohlcv.low,
                        &ohlcv.close,
                        ChandelierExitParams {
                            period: Some(period),
                            mult: Some(3.0),
                            use_close: Some(true),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar Chandelier Exit primary must accept every period")
                .long_stop,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "long_stop[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "long_stop[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn cksp_outputs_stay_resident_for_the_exact_default_tuple() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;
        use vector_ta::indicators::cksp::CkspParams;

        assert!(
            F64Kernel::Cksp.is_sequential(),
            "the CKSP tuple must preserve RMA and four deque states in one thread"
        );
        assert!(
            F64Kernel::Cksp.is_period_invariant(),
            "the preserved primary ABI is the current default-only admitted route"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [CkspParams::default()];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the CKSP resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_cksp_outputs_device(&parameter_rows)
            .expect("both CKSP outputs must stay resident for the exact default tuple");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the CKSP route re-uploaded its resident HLC input"
        );
        assert_eq!(result.indicator_id, "cksp");
        assert_eq!(result.entry_point, "cksp_outputs_f64");
        assert_eq!((result.rows, result.cols), (1, ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            CKSP_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 1
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident CKSP launch must retire");
    }

    #[test]
    fn cksp_outputs_match_cpu_bits_for_the_exact_default_tuple() {
        use vector_ta::indicators::cksp::{CkspInput, CkspParams, cksp_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [CkspParams::default()];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the CKSP parity test requires CUDA device 0");
        let result = engine
            .compute_cksp_outputs_device(&parameter_rows)
            .expect("the resident CKSP route must accept the exact default tuple");
        let expected = cksp_with_kernel(
            &CkspInput::from_slices(&ohlcv.high, &ohlcv.low, &ohlcv.close, CkspParams::default()),
            Kernel::Scalar,
        )
        .expect("the scalar CKSP formula must accept its registry defaults");

        for (output, (output_id, cpu)) in result.outputs.iter().zip([
            ("long_values", expected.long_values),
            ("short_values", expected.short_values),
        ]) {
            assert_eq!(output.output_id, output_id);
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("the resident CKSP output must download for parity");
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(&cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{output_id}[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{output_id}[{index}] is not exact scalar CPU f64 parity: \
                         gpu={gpu:?} cpu={cpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn cksp_primary_route_matches_cpu_bits_for_the_exact_default_tuple() {
        use vector_ta::indicators::cksp::{CkspInput, CkspParams, cksp_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let ohlcv = repeated_ctrader_bandpass_fixture();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the CKSP primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("cksp", &[1])
            .expect("the generic CKSP primary route must preserve its inert anchor");
        assert_eq!(output.output_id, "long_values");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident CKSP primary matrix must download for parity"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly CKSP unexpectedly materialized a host result")
            }
        };
        let expected = cksp_with_kernel(
            &CkspInput::from_slices(&ohlcv.high, &ohlcv.low, &ohlcv.close, CkspParams::default()),
            Kernel::Scalar,
        )
        .expect("the scalar CKSP primary must accept its defaults")
        .long_values;

        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "long_values[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "long_values[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn acosc_primary_route_is_reviewed_f64_and_sequential() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;
        use vector_ta::indicators::acosc::{AcoscInput, AcoscParams, acosc_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        assert!(
            F64Kernel::Acosc.is_sequential(),
            "the ACOSC primary must preserve rolling state in one thread per row"
        );

        let ohlcv = range_oscillator_parity_fixture();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the ACOSC primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("acosc", &[1])
            .expect("the generic ACOSC primary route must stay resident");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the primary ACOSC matrix must download for parity"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly ACOSC unexpectedly materialized a host result")
            }
        };
        let expected = acosc_with_kernel(
            &AcoscInput::from_slices(&ohlcv.high, &ohlcv.low, AcoscParams::default()),
            Kernel::Scalar,
        )
        .expect("the reviewed CPU ACOSC formula must accept the fixture")
        .osc;

        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "osc[{index}] lost CPU undefined/reset state: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "osc[{index}] is not the exact CPU rolling-state f64 result"
                );
            }
        }
    }

    #[test]
    fn prb_primary_route_uses_resident_f64_and_matches_the_scalar_reference() {
        use vector_ta::indicators::prb::{PrbInput, PrbParams, prb};

        let ohlcv = range_oscillator_parity_fixture();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the PRB primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("prb", &[14])
            .expect("the generic PRB primary route must stay resident");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident PRB matrix must download for parity"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly PRB unexpectedly materialized a host result")
            }
        };
        let expected = prb(&PrbInput::from_slice(&ohlcv.close, PrbParams::default()))
            .expect("the scalar PRB reference must accept the same f64 fixture")
            .values;

        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "PRB[{index}] lost CPU undefined/reset state: gpu={gpu:?}"
                );
            } else {
                let tolerance = 2.0e-12_f64.max(cpu.abs() * 2.0e-13);
                assert!(
                    (gpu - cpu).abs() <= tolerance,
                    "PRB[{index}] exceeds f64 parity: gpu={gpu:.17e} cpu={cpu:.17e} \
                     delta={:.3e} tolerance={tolerance:.3e}",
                    (gpu - cpu).abs()
                );
            }
        }
    }

    #[test]
    fn vdubus_all_outputs_match_cpu_bits() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::vdubus_divergence_wave_pattern_generator::{
            VdubusDivergenceWavePatternGeneratorInput, VdubusDivergenceWavePatternGeneratorParams,
            vdubus_divergence_wave_pattern_generator,
        };

        let ohlcv = range_filtered_trend_signals_parity_fixture();
        let parameter_rows = vec![
            VdubusDivergenceWavePatternGeneratorParams::default(),
            VdubusDivergenceWavePatternGeneratorParams {
                fast_depth: Some(5),
                slow_depth: Some(13),
                fast_length: Some(8),
                slow_length: Some(21),
                signal_length: Some(5),
                lookback: Some(3),
                ..Default::default()
            },
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Vdubus parity test requires CUDA device 0");
        let result = engine
            .compute_vdubus_divergence_wave_pattern_generator_outputs_device(&parameter_rows)
            .expect("the resident route must accept the exact Vdubus rows");

        let mut expected: BTreeMap<&str, Vec<f64>> =
            VDUBUS_DIVERGENCE_WAVE_PATTERN_GENERATOR_OUTPUT_IDS
                .into_iter()
                .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
                .collect();
        for params in parameter_rows {
            let output = vdubus_divergence_wave_pattern_generator(
                &VdubusDivergenceWavePatternGeneratorInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    params,
                ),
            )
            .expect("the scalar Vdubus parity reference must accept the same row");
            macro_rules! extend_output {
                ($id:literal, $field:ident) => {
                    expected.get_mut($id).unwrap().extend(output.$field)
                };
            }
            extend_output!("fast_standard", fast_standard);
            extend_output!("fast_climax", fast_climax);
            extend_output!("fast_rounded", fast_rounded);
            extend_output!("fast_predator", fast_predator);
            extend_output!("slow_standard", slow_standard);
            extend_output!("slow_climax", slow_climax);
            extend_output!("slow_rounded", slow_rounded);
            extend_output!("slow_predator", slow_predator);
            extend_output!("opposing_force", opposing_force);
            extend_output!("macd", macd);
            extend_output!("signal", signal);
            extend_output!("hist", hist);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Vdubus output must download");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected Vdubus output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined/reset state: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not bit-identical: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn vdubus_cuda_matches_hand_derived_concurrent_ema_seed() {
        use vector_ta::indicators::vdubus_divergence_wave_pattern_generator::VdubusDivergenceWavePatternGeneratorParams;

        let close = vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let ohlcv = Ohlcv {
            timestamp: None,
            open: close.clone(),
            high: close.iter().map(|value| value + 1.0).collect(),
            low: close.iter().map(|value| value - 1.0).collect(),
            close,
            volume: Some(vec![1.0; 6]),
        };
        let params = VdubusDivergenceWavePatternGeneratorParams {
            fast_depth: Some(1),
            slow_depth: Some(1),
            fast_length: Some(2),
            slow_length: Some(3),
            signal_length: Some(2),
            lookback: Some(1),
            ..Default::default()
        };
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Vdubus hand-oracle test requires CUDA device 0");
        let result = engine
            .compute_vdubus_divergence_wave_pattern_generator_outputs_device(&[params])
            .expect("the resident route must accept the hand-derived Vdubus row");
        let download = |output_id| {
            let output = result
                .outputs
                .iter()
                .find(|output| output.output_id == output_id)
                .unwrap_or_else(|| panic!("missing resident Vdubus output {output_id}"));
            engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("download {output_id}: {error}"))
        };
        let macd = download("macd");
        let signal = download("signal");
        let hist = download("hist");

        assert!(hist[..3].iter().all(|value| value.is_nan()));
        assert!((macd[3] - 11.0 / 9.0).abs() <= 1e-15);
        assert!((signal[3] - 37.0 / 36.0).abs() <= 1e-15);
        assert!((hist[3] - 7.0 / 36.0).abs() <= 1e-15);
    }

    /// Registry size is not execution coverage. Launch every registered f64
    /// row through the same resident session and keep all outputs on device
    /// until one final synchronization. Any resolving-but-unlaunchable row is
    /// reported by name instead of being hidden behind a count.
    #[test]
    fn every_registered_f64_primary_kernel_launches_resident_on_the_real_card() {
        use vector_ta::indicators::dispatch::cuda_f64::F64_KERNELS;

        let ohlcv = crate::test_fixtures::ctrader_sample_ohlcv();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the real-card registry test requires a launchable CUDA device 0");
        let mut resident_outputs = Vec::with_capacity(F64_KERNELS.len());
        let mut failures = Vec::new();

        for spec in F64_KERNELS {
            match engine.compute_primary_device(spec.indicator_id, &[14]) {
                Ok(output) => resident_outputs.push(output),
                Err(error) => failures.push(format!("{}: {error:#}", spec.indicator_id)),
            }
        }
        engine
            .synchronize()
            .expect("all registered f64 primary kernels must retire successfully");

        assert!(
            failures.is_empty(),
            "{} registered f64 kernel(s) failed a real resident launch:\n{}",
            failures.len(),
            failures.join("\n")
        );
        assert_eq!(resident_outputs.len(), F64_KERNELS.len());
        assert!(
            resident_outputs
                .iter()
                .all(|output| matches!(output.series, IndicatorCudaSeriesF64::DeviceF64(_)))
        );
    }

    /// No duplicate ids — a duplicate would emit the same column twice and
    /// silently change the feature-frame width.
    #[test]
    fn spec_ids_are_unique() {
        let mut ids: Vec<&str> = GPU_SWEEP_SPECS.iter().map(|s| s.id).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "duplicate id in GPU_SWEEP_SPECS");
    }

    /// The two indicators whose CPU source is hlc3 must NOT be served from the
    /// OHLCV ref, because that ref hands the kernel `close`. Getting this wrong
    /// computes a different indicator, not a less precise one.
    #[test]
    fn hlc3_sourced_indicators_do_not_use_the_close_ref() {
        for id in ["cci", "mfi"] {
            let spec = spec_for(id).unwrap_or_else(|| panic!("{id} missing from GPU_SWEEP_SPECS"));
            assert!(
                matches!(
                    spec.input,
                    DeviceInput::Hlc3Slice | DeviceInput::Hlc3CloseVolume
                ),
                "{id}: CPU path sources hlc3 (cpu_batch.rs:2867 / :3401) but the spec would feed \
                 the kernel close"
            );
        }
    }

    #[test]
    fn preflight_misses_remain_in_the_gpu_schema_as_nan_columns() {
        let periods = [7, 21, 50, 100, 200];
        let n = 100;
        let computed = vec![vec![7.0; n], vec![21.0; n], vec![50.0; n]];
        let columns = assemble_sweep_columns("sma", &periods, n, computed).unwrap();

        let names: Vec<&str> = columns.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["sma_7", "sma_21", "sma_50", "sma_100", "sma_200"]);
        assert!(columns[3].1.iter().all(|value| value.is_nan()));
        assert!(columns[4].1.iter().all(|value| value.is_nan()));
        assert!(columns[0].1.iter().all(|&value| value == 7.0));
    }

    fn canonical_coppock_parameter_rows() -> [vector_ta::indicators::coppock::CoppockParams; 6] {
        use vector_ta::indicators::coppock::CoppockParams;

        [
            (11, 14, 10),
            (6, 7, 5),
            (17, 21, 15),
            (39, 50, 36),
            (79, 100, 71),
            (157, 200, 143),
        ]
        .map(|(short, long, ma)| CoppockParams {
            short_roc_period: Some(short),
            long_roc_period: Some(long),
            ma_period: Some(ma),
            ma_type: Some("wma".to_string()),
        })
    }

    #[test]
    fn coppock_outputs_stay_resident_for_default_and_all_ratio_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Coppock.is_sequential(),
            "each Coppock tuple must own its exact ROC/WMA recurrence in one thread"
        );
        assert!(
            !F64Kernel::Coppock.is_period_invariant(),
            "Coppock must consume every admitted ratio tuple instead of replaying defaults"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = canonical_coppock_parameter_rows();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Coppock resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_coppock_output_device(&parameter_rows)
            .expect("all canonical Coppock tuples must stay resident in one launch");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Coppock route re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "coppock");
        assert_eq!(result.entry_point, "coppock_production_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].output_id, "value");
        assert_eq!(result.outputs[0].matrix.rows(), parameter_rows.len());
        assert_eq!(result.outputs[0].matrix.cols(), ohlcv.len());
        assert_eq!(
            result.outputs[0].matrix.device_id(),
            engine.device_ordinal()
        );

        engine
            .synchronize()
            .expect("the resident Coppock launch must retire");
    }

    #[test]
    fn coppock_outputs_match_cpu_bits_for_default_and_all_ratio_sweeps() {
        use vector_ta::indicators::coppock::{CoppockInput, coppock_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let parameter_rows = canonical_coppock_parameter_rows();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Coppock parity test requires CUDA device 0");
        let result = engine
            .compute_coppock_output_device(&parameter_rows)
            .expect("the resident Coppock route must accept all canonical tuples");
        let actual = engine
            .runtime
            .download_matrix_f64(&result.outputs[0].matrix)
            .expect("the resident Coppock output must download for parity");

        let mut expected = Vec::with_capacity(actual.len());
        for params in &parameter_rows {
            expected.extend(
                coppock_with_kernel(
                    &CoppockInput::from_slice(&ohlcv.close, params.clone()),
                    Kernel::Scalar,
                )
                .expect("the scalar Coppock authority must accept the canonical tuple")
                .values,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "value[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "value[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn coppock_primary_route_consumes_ratio_anchor_and_matches_cpu_bits() {
        use vector_ta::indicators::coppock::{CoppockInput, coppock_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let anchors = [14_usize, 7, 21, 50, 100, 200];
        let parameter_rows = canonical_coppock_parameter_rows();
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Coppock primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("coppock", &anchors)
            .expect("the preserved primary ABI must consume each ratio anchor");
        assert_eq!(output.output_id, "value");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Coppock primary matrix must download for parity"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Coppock unexpectedly materialized a host result")
            }
        };

        let mut expected = Vec::with_capacity(actual.len());
        for params in &parameter_rows {
            expected.extend(
                coppock_with_kernel(
                    &CoppockInput::from_slice(&ohlcv.close, params.clone()),
                    Kernel::Scalar,
                )
                .expect("the scalar Coppock authority must accept the anchor-derived tuple")
                .values,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary value[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary value[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn correlation_cycle_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::CorrelationCycle.is_sequential(),
            "each Correlation Cycle tuple must own its exact recursive state in one thread"
        );
        assert!(
            !F64Kernel::CorrelationCycle.is_period_invariant(),
            "Correlation Cycle must consume every admitted period anchor"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_tuples = [
            (20, 9.0),
            (7, 9.0),
            (21, 9.0),
            (50, 9.0),
            (100, 9.0),
            (200, 9.0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Correlation Cycle resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_correlation_cycle_outputs_device(&parameter_tuples)
            .expect("all Correlation Cycle outputs must stay resident in one launch");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the Correlation Cycle route re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "correlation_cycle");
        assert_eq!(result.entry_point, "correlation_cycle_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_tuples.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            CORRELATION_CYCLE_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));

        engine
            .synchronize()
            .expect("the resident Correlation Cycle launch must retire");
    }

    #[test]
    fn correlation_cycle_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use vector_ta::indicators::correlation_cycle::{
            CorrelationCycleInput, CorrelationCycleParams, correlation_cycle_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let parameter_tuples = [
            (20, 9.0),
            (7, 9.0),
            (21, 9.0),
            (50, 9.0),
            (100, 9.0),
            (200, 9.0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Correlation Cycle parity test requires CUDA device 0");
        let result = engine
            .compute_correlation_cycle_outputs_device(&parameter_tuples)
            .expect("the resident Correlation Cycle route must accept every canonical tuple");

        let references = parameter_tuples
            .iter()
            .map(|&(period, threshold)| {
                correlation_cycle_with_kernel(
                    &CorrelationCycleInput::from_slice(
                        &ohlcv.close,
                        CorrelationCycleParams {
                            period: Some(period),
                            threshold: Some(threshold),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar Correlation Cycle authority must accept every tuple")
            })
            .collect::<Vec<_>>();

        for (output, output_id) in result.outputs.iter().zip(CORRELATION_CYCLE_OUTPUT_IDS) {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("the resident Correlation Cycle output must download for parity");
            let mut expected = Vec::with_capacity(result.rows * result.cols);
            for reference in &references {
                let values = match output_id {
                    "real" => &reference.real,
                    "imag" => &reference.imag,
                    "angle" => &reference.angle,
                    "state" => &reference.state,
                    _ => unreachable!("typed Correlation Cycle output drifted"),
                };
                expected.extend_from_slice(values);
            }
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{output_id}[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{output_id}[{index}] is not exact scalar CPU f64 parity: \
                         gpu={gpu:?} cpu={cpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn correlation_cycle_primary_route_preserves_real_output_bits() {
        use vector_ta::indicators::correlation_cycle::{
            CorrelationCycleInput, CorrelationCycleParams, correlation_cycle_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let periods = [20_usize, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Correlation Cycle primary-route test requires CUDA device 0");
        let output = engine
            .compute_primary_device("correlation_cycle", &periods)
            .expect("the preserved primary ABI must consume every period anchor");
        assert_eq!(output.output_id, "real");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Correlation Cycle primary matrix must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Correlation Cycle unexpectedly materialized a host result")
            }
        };

        let mut expected = Vec::with_capacity(actual.len());
        for period in periods {
            expected.extend(
                correlation_cycle_with_kernel(
                    &CorrelationCycleInput::from_slice(
                        &ohlcv.close,
                        CorrelationCycleParams {
                            period: Some(period),
                            threshold: Some(9.0),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar Correlation Cycle primary must accept every period")
                .real,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary real[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary real[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn cvi_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::{CVI_MAX_PERIOD, F64Kernel};

        assert!(F64Kernel::Cvi.is_sequential());
        assert!(!F64Kernel::Cvi.is_period_invariant());
        assert_eq!(F64Kernel::Cvi.max_period(), Some(CVI_MAX_PERIOD));
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let periods = [10_usize, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the CVI resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let output = engine
            .compute_cvi_output_device(&periods)
            .expect("all canonical CVI periods must stay resident in one launch");

        assert_eq!(engine.uploads(), uploads_before, "CVI re-uploaded high/low");
        assert_eq!(output.indicator_id, "cvi");
        assert_eq!(output.entry_point, "cvi_batch_f64");
        assert_eq!((output.rows, output.cols), (periods.len(), ohlcv.len()));
        assert_eq!(output.outputs.len(), 1);
        assert_eq!(output.outputs[0].output_id, "value");
        let matrix = &output.outputs[0].matrix;
        assert_eq!(matrix.device_id(), engine.device_ordinal());
        engine
            .synchronize()
            .expect("the resident CVI launch must retire");
    }

    #[test]
    fn cvi_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use vector_ta::indicators::cvi::{CviInput, CviParams, cvi_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        let periods = [10_usize, 7, 21, 50, 100, 200];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the CVI parity test requires CUDA device 0");
        let output = engine
            .compute_cvi_output_device(&periods)
            .expect("the resident CVI route must accept every canonical period");
        assert_eq!(output.outputs.len(), 1);
        assert_eq!(output.outputs[0].output_id, "value");
        let actual = engine
            .runtime
            .download_matrix_f64(&output.outputs[0].matrix)
            .expect("the resident CVI matrix must download for parity");

        let mut expected = Vec::with_capacity(actual.len());
        for period in periods {
            expected.extend(
                cvi_with_kernel(
                    &CviInput::from_slices(
                        &ohlcv.high,
                        &ohlcv.low,
                        CviParams {
                            period: Some(period),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar CVI authority must accept every canonical period")
                .values,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "value[{index}] lost CPU NaN: gpu={gpu:?}");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "value[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn cvi_primary_route_preserves_value_output_bits() {
        use vector_ta::indicators::cvi::{CviInput, CviParams, cvi_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        let periods = [10_usize, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the CVI primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("cvi", &periods)
            .expect("the preserved CVI primary ABI must consume every canonical period");
        assert_eq!(output.output_id, "value");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident CVI primary matrix must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly CVI unexpectedly materialized a host primary result")
            }
        };

        let mut expected = Vec::with_capacity(actual.len());
        for period in periods {
            expected.extend(
                cvi_with_kernel(
                    &CviInput::from_slices(
                        &ohlcv.high,
                        &ohlcv.low,
                        CviParams {
                            period: Some(period),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar CVI primary must accept every canonical period")
                .values,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary value[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary value[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn cyberpunk_value_trend_analyzer_outputs_stay_resident_for_the_canonical_tuple() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(F64Kernel::CyberpunkValueTrendAnalyzer.is_sequential());
        assert!(F64Kernel::CyberpunkValueTrendAnalyzer.is_period_invariant());
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_tuples = [(30_usize, 75_usize)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Cyberpunk Value Trend Analyzer resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_cyberpunk_value_trend_analyzer_outputs_device(&parameter_tuples)
            .expect("all Cyberpunk Value Trend Analyzer outputs must stay resident in one launch");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Cyberpunk Value Trend Analyzer re-uploaded its resident OHLC input"
        );
        assert_eq!(result.indicator_id, "cyberpunk_value_trend_analyzer");
        assert_eq!(
            result.entry_point,
            "cyberpunk_value_trend_analyzer_batch_f64"
        );
        assert_eq!((result.rows, result.cols), (1, ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 1
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Cyberpunk Value Trend Analyzer launch must retire");
    }

    #[test]
    fn cyberpunk_value_trend_analyzer_outputs_match_cpu_bits() {
        use vector_ta::indicators::cyberpunk_value_trend_analyzer::{
            CyberpunkValueTrendAnalyzerInput, CyberpunkValueTrendAnalyzerParams,
            cyberpunk_value_trend_analyzer_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.open[90] = f64::NAN;
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let parameter_tuples = [(30_usize, 75_usize)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Cyberpunk Value Trend Analyzer parity test requires CUDA device 0");
        let result = engine
            .compute_cyberpunk_value_trend_analyzer_outputs_device(&parameter_tuples)
            .expect("the resident route must accept the canonical threshold tuple");
        let reference = cyberpunk_value_trend_analyzer_with_kernel(
            &CyberpunkValueTrendAnalyzerInput::from_slices(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                CyberpunkValueTrendAnalyzerParams {
                    entry_level: Some(30),
                    exit_level: Some(75),
                },
            ),
            Kernel::Scalar,
        )
        .expect("the scalar Cyberpunk Value Trend Analyzer authority must accept the tuple");

        for (output, output_id) in result
            .outputs
            .iter()
            .zip(CYBERPUNK_VALUE_TREND_ANALYZER_OUTPUT_IDS)
        {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("the resident Cyberpunk Value Trend Analyzer output must download");
            let expected = match output_id {
                "value_trend" => &reference.value_trend,
                "value_trend_lag" => &reference.value_trend_lag,
                "deviation_index" => &reference.deviation_index,
                "overbought_signal" => &reference.overbought_signal,
                "buy_signal" => &reference.buy_signal,
                "sell_signal" => &reference.sell_signal,
                _ => unreachable!("typed Cyberpunk Value Trend Analyzer output drifted"),
            };
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{output_id}[{index}] lost CPU undefined/reset state: gpu={gpu:?}"
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{output_id}[{index}] is not exact scalar CPU f64 parity: \
                         gpu={gpu:?} cpu={cpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn cyberpunk_value_trend_analyzer_primary_route_preserves_value_trend_bits() {
        use vector_ta::indicators::cyberpunk_value_trend_analyzer::{
            CyberpunkValueTrendAnalyzerInput, CyberpunkValueTrendAnalyzerParams,
            cyberpunk_value_trend_analyzer_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.open[90] = f64::NAN;
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Cyberpunk Value Trend Analyzer primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("cyberpunk_value_trend_analyzer", &[1])
            .expect("the preserved primary ABI must remain a resident value_trend route");
        assert_eq!(output.output_id, "value_trend");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Cyberpunk Value Trend Analyzer primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Cyberpunk Value Trend Analyzer materialized a host result")
            }
        };
        let expected = cyberpunk_value_trend_analyzer_with_kernel(
            &CyberpunkValueTrendAnalyzerInput::from_slices(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                CyberpunkValueTrendAnalyzerParams::default(),
            ),
            Kernel::Scalar,
        )
        .expect("the scalar Cyberpunk Value Trend Analyzer primary must accept defaults")
        .value_trend;
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary value_trend[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary value_trend[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn cycle_channel_oscillator_outputs_stay_resident_for_default_and_ratio_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(F64Kernel::CycleChannelOscillator.is_sequential());
        assert!(
            F64Kernel::CycleChannelOscillator.is_period_invariant(),
            "the preserved primary ABI remains the canonical default-only fast route"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_tuples = [
            (10_usize, 30_usize, 1.0_f64, 3.0_f64),
            (7, 21, 1.0, 3.0),
            (17, 50, 1.0, 3.0),
            (33, 100, 1.0, 3.0),
            (67, 200, 1.0, 3.0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Cycle Channel Oscillator resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_cycle_channel_oscillator_outputs_device(&parameter_tuples)
            .expect("both Cycle Channel Oscillator outputs must stay resident in one launch");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Cycle Channel Oscillator re-uploaded its resident default-close HLC input"
        );
        assert_eq!(result.indicator_id, "cycle_channel_oscillator");
        assert_eq!(result.entry_point, "cycle_channel_oscillator_batch_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_tuples.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Cycle Channel Oscillator launch must retire");
    }

    #[test]
    fn cycle_channel_oscillator_outputs_match_cpu_bits() {
        use vector_ta::indicators::cycle_channel_oscillator::{
            CycleChannelOscillatorInput, CycleChannelOscillatorParams,
            cycle_channel_oscillator_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let parameter_tuples = [
            (10_usize, 30_usize, 1.0_f64, 3.0_f64),
            (7, 21, 1.0, 3.0),
            (17, 50, 1.0, 3.0),
            (33, 100, 1.0, 3.0),
            (67, 200, 1.0, 3.0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Cycle Channel Oscillator parity test requires CUDA device 0");
        let result = engine
            .compute_cycle_channel_oscillator_outputs_device(&parameter_tuples)
            .expect("the resident route must accept every canonical tuple");

        for (output, output_id) in result
            .outputs
            .iter()
            .zip(CYCLE_CHANNEL_OSCILLATOR_OUTPUT_IDS)
        {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("the resident Cycle Channel Oscillator output must download");
            let mut expected = Vec::with_capacity(parameter_tuples.len() * ohlcv.len());
            for &(short_cycle_length, medium_cycle_length, short_multiplier, medium_multiplier) in
                &parameter_tuples
            {
                let reference = cycle_channel_oscillator_with_kernel(
                    &CycleChannelOscillatorInput::from_slices(
                        &ohlcv.close,
                        &ohlcv.high,
                        &ohlcv.low,
                        &ohlcv.close,
                        CycleChannelOscillatorParams {
                            short_cycle_length: Some(short_cycle_length),
                            medium_cycle_length: Some(medium_cycle_length),
                            short_multiplier: Some(short_multiplier),
                            medium_multiplier: Some(medium_multiplier),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar Cycle Channel Oscillator authority must accept the tuple");
                match output_id {
                    "fast" => expected.extend(reference.fast),
                    "slow" => expected.extend(reference.slow),
                    _ => unreachable!("typed Cycle Channel Oscillator output drifted"),
                }
            }

            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{output_id}[{index}] lost CPU undefined state: gpu={gpu:?}"
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{output_id}[{index}] is not exact scalar CPU f64 parity: \
                         gpu={gpu:?} cpu={cpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn cycle_channel_oscillator_primary_route_preserves_default_fast_bits() {
        use vector_ta::indicators::cycle_channel_oscillator::{
            CycleChannelOscillatorInput, CycleChannelOscillatorParams,
            cycle_channel_oscillator_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Cycle Channel Oscillator primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("cycle_channel_oscillator", &[1])
            .expect("the preserved primary ABI must remain a resident default fast route");
        assert_eq!(output.output_id, "fast");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Cycle Channel Oscillator primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Cycle Channel Oscillator materialized a host result")
            }
        };
        let expected = cycle_channel_oscillator_with_kernel(
            &CycleChannelOscillatorInput::from_slices(
                &ohlcv.close,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                CycleChannelOscillatorParams::default(),
            ),
            Kernel::Scalar,
        )
        .expect("the scalar Cycle Channel Oscillator primary must accept defaults")
        .fast;
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary fast[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary fast[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn daily_factor_outputs_stay_resident_for_the_canonical_threshold() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(F64Kernel::DailyFactor.is_sequential());
        assert!(F64Kernel::DailyFactor.is_period_invariant());
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let threshold_levels = [0.35_f64];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Daily Factor resident test requires CUDA device 0");
        let uploads_before = engine.uploads();

        let result = engine
            .compute_daily_factor_outputs_device(&threshold_levels)
            .expect("all Daily Factor outputs must stay resident in one launch");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Daily Factor re-uploaded its resident OHLC input"
        );
        assert_eq!(result.indicator_id, "daily_factor");
        assert_eq!(result.entry_point, "daily_factor_batch_f64");
        assert_eq!((result.rows, result.cols), (1, ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            DAILY_FACTOR_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 1
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("the resident Daily Factor launch must retire");
    }

    #[test]
    fn daily_factor_outputs_match_cpu_bits() {
        use vector_ta::indicators::daily_factor::{
            DailyFactorInput, DailyFactorParams, daily_factor_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.open[90] = f64::NAN;
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let threshold_levels = [0.35_f64];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Daily Factor parity test requires CUDA device 0");
        let result = engine
            .compute_daily_factor_outputs_device(&threshold_levels)
            .expect("the resident route must accept the canonical threshold");
        let reference = daily_factor_with_kernel(
            &DailyFactorInput::from_slices(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                DailyFactorParams {
                    threshold_level: Some(0.35),
                },
            ),
            Kernel::Scalar,
        )
        .expect("the scalar Daily Factor authority must accept the canonical threshold");

        for (output, output_id) in result.outputs.iter().zip(DAILY_FACTOR_OUTPUT_IDS) {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("the resident Daily Factor output must download");
            let expected = match output_id {
                "value" => &reference.value,
                "ema" => &reference.ema,
                "signal" => &reference.signal,
                _ => unreachable!("typed Daily Factor output drifted"),
            };
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{output_id}[{index}] lost CPU undefined state: gpu={gpu:?}"
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{output_id}[{index}] is not exact scalar CPU f64 parity: \
                         gpu={gpu:?} cpu={cpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn daily_factor_primary_route_preserves_default_value_bits() {
        use vector_ta::indicators::daily_factor::{
            DailyFactorInput, DailyFactorParams, daily_factor_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.open[90] = f64::NAN;
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Daily Factor primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("daily_factor", &[1])
            .expect("the preserved primary ABI must remain a resident default value route");
        assert_eq!(output.output_id, "value");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Daily Factor primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Daily Factor materialized a host result")
            }
        };
        let expected = daily_factor_with_kernel(
            &DailyFactorInput::from_slices(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                DailyFactorParams::default(),
            ),
            Kernel::Scalar,
        )
        .expect("the scalar Daily Factor primary must accept defaults")
        .value;
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary value[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary value[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn damiani_volatmeter_outputs_stay_resident_for_default_and_ratio_tuples() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(F64Kernel::DamianiVolatmeter.is_sequential());
        assert!(F64Kernel::DamianiVolatmeter.is_period_invariant());
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_tuples = [
            (13_usize, 20_usize, 40_usize, 100_usize, 1.4_f64),
            (1, 1, 3, 7, 1.4),
            (3, 4, 8, 21, 1.4),
            (7, 10, 20, 50, 1.4),
            (26, 40, 80, 200, 1.4),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Damiani Volatmeter resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_damiani_volatmeter_outputs_device(&parameter_tuples)
            .expect("both Damiani Volatmeter outputs must stay resident in one launch");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Damiani Volatmeter re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "damiani_volatmeter");
        assert_eq!(result.entry_point, "damiani_volatmeter_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_tuples.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            DAMIANI_VOLATMETER_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("both resident Damiani Volatmeter outputs must retire");
    }

    #[test]
    fn damiani_volatmeter_outputs_match_cpu_bits() {
        use vector_ta::indicators::damiani_volatmeter::{
            DamianiVolatmeterInput, DamianiVolatmeterParams, damiani_volatmeter_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[90] = f64::NAN;
        let parameter_tuples = [
            (13_usize, 20_usize, 40_usize, 100_usize, 1.4_f64),
            (1, 1, 3, 7, 1.4),
            (3, 4, 8, 21, 1.4),
            (7, 10, 20, 50, 1.4),
            (26, 40, 80, 200, 1.4),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Damiani Volatmeter parity test requires CUDA device 0");
        let result = engine
            .compute_damiani_volatmeter_outputs_device(&parameter_tuples)
            .expect("the resident route must accept every canonical Damiani tuple");

        for (output, output_id) in result.outputs.iter().zip(DAMIANI_VOLATMETER_OUTPUT_IDS) {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("the resident Damiani Volatmeter output must download");
            let mut expected = Vec::with_capacity(parameter_tuples.len() * ohlcv.len());
            for &(vis_atr, vis_std, sed_atr, sed_std, threshold) in &parameter_tuples {
                let reference = damiani_volatmeter_with_kernel(
                    &DamianiVolatmeterInput::from_slice(
                        &ohlcv.close,
                        DamianiVolatmeterParams {
                            vis_atr: Some(vis_atr),
                            vis_std: Some(vis_std),
                            sed_atr: Some(sed_atr),
                            sed_std: Some(sed_std),
                            threshold: Some(threshold),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar Damiani Volatmeter authority must accept the same tuple");
                expected.extend(match output_id {
                    "vol" => reference.vol,
                    "anti" => reference.anti,
                    other => panic!("unexpected Damiani Volatmeter output `{other}`"),
                });
            }
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{output_id}[{index}] lost CPU NaN: gpu={gpu:?}"
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{output_id}[{index}] is not exact scalar CPU f64 parity: \
                         gpu={gpu:?} cpu={cpu:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn damiani_volatmeter_primary_route_preserves_default_vol_bits() {
        use vector_ta::indicators::damiani_volatmeter::{
            DamianiVolatmeterInput, DamianiVolatmeterParams, damiani_volatmeter_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[90] = f64::NAN;
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Damiani Volatmeter primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("damiani_volatmeter", &[1])
            .expect("the preserved primary ABI must remain a resident default vol route");
        assert_eq!(output.output_id, "vol");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Damiani Volatmeter primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Damiani Volatmeter materialized a host result")
            }
        };
        let expected = damiani_volatmeter_with_kernel(
            &DamianiVolatmeterInput::from_slice(&ohlcv.close, DamianiVolatmeterParams::default()),
            Kernel::Scalar,
        )
        .expect("the scalar Damiani Volatmeter primary must accept defaults")
        .vol;
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary vol[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary vol[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn di_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Di.is_sequential(),
            "each DI tuple owns one exact Wilder state thread"
        );
        assert!(
            !F64Kernel::Di.is_period_invariant(),
            "DI must consume every admitted period"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let periods = [14, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the DI resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_di_outputs_device(&periods)
            .expect("both DI outputs must stay resident for every admitted period");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "DI re-uploaded its resident high/low/close input"
        );
        assert_eq!(result.indicator_id, "di");
        assert_eq!(result.entry_point, "di_outputs_f64");
        assert_eq!((result.rows, result.cols), (periods.len(), ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            DI_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == periods.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("both resident DI matrices must retire");
    }

    #[test]
    fn di_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::di::{
            DiInput, DiParams, di_minus_with_kernel, di_plus_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        for index in 0..3 {
            ohlcv.high[index] = f64::NAN;
            ohlcv.low[index] = f64::NAN;
            ohlcv.close[index] = f64::NAN;
        }
        for index in 60..=64 {
            ohlcv.high[index] = ohlcv.high[59];
            ohlcv.low[index] = ohlcv.low[59];
        }
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let periods = [14, 7, 21, 50, 100, 200];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the DI parity test requires CUDA device 0");
        let result = engine
            .compute_di_outputs_device(&periods)
            .expect("the resident DI route must accept the default and all five sweeps");

        let mut expected: BTreeMap<&str, Vec<f64>> = DI_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for period in periods {
            let input = DiInput::from_slices(
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                DiParams {
                    period: Some(period),
                },
            );
            expected.get_mut("plus").unwrap().extend(
                di_plus_with_kernel(&input, Kernel::Scalar)
                    .expect("the scalar DI plus authority must accept the admitted period"),
            );
            expected.get_mut("minus").unwrap().extend(
                di_minus_with_kernel(&input, Kernel::Scalar)
                    .expect("the scalar DI minus authority must accept the admitted period"),
            );
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident DI output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected DI output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn di_primary_route_preserves_default_plus_bits() {
        use vector_ta::indicators::di::{DiInput, DiParams, di_plus_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.close[90] = f64::NAN;
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the DI primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("di", &[14])
            .expect("the preserved primary ABI must remain a resident default plus route");
        assert_eq!(output.output_id, "plus");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident DI primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly DI materialized a host result")
            }
        };
        let expected = di_plus_with_kernel(
            &DiInput::from_slices(
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                DiParams { period: Some(14) },
            ),
            Kernel::Scalar,
        )
        .expect("the scalar DI primary must accept defaults");
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary plus[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary plus[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn dm_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Dm.is_sequential(),
            "each DM tuple owns one exact Wilder state thread"
        );
        assert!(
            !F64Kernel::Dm.is_period_invariant(),
            "DM must consume every admitted period"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let periods = [14, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the DM resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_dm_outputs_device(&periods)
            .expect("both DM outputs must stay resident for every admitted period");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "DM re-uploaded its resident high/low input"
        );
        assert_eq!(result.indicator_id, "dm");
        assert_eq!(result.entry_point, "dm_batch_f64");
        assert_eq!((result.rows, result.cols), (periods.len(), ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            DM_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == periods.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("both resident DM matrices must retire");
    }

    #[test]
    fn dm_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::dm::{DmInput, DmParams, dm_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        for index in 60..=64 {
            let expansion = (index - 59) as f64 * 0.125;
            ohlcv.high[index] = ohlcv.high[59] + expansion;
            ohlcv.low[index] = ohlcv.low[59] - expansion;
        }
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        let periods = [14, 7, 21, 50, 100, 200];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the DM parity test requires CUDA device 0");
        let result = engine
            .compute_dm_outputs_device(&periods)
            .expect("the resident DM route must accept the default and all five sweeps");

        let mut expected: BTreeMap<&str, Vec<f64>> = DM_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for period in periods {
            let output = dm_with_kernel(
                &DmInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    DmParams {
                        period: Some(period),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar DM authority must accept the admitted period");
            expected.get_mut("plus").unwrap().extend(output.plus);
            expected.get_mut("minus").unwrap().extend(output.minus);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident DM output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected DM output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn dm_primary_route_preserves_default_plus_bits() {
        use vector_ta::indicators::dm::{DmInput, DmParams, dm_plus_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the DM primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("dm", &[14])
            .expect("the preserved primary ABI must remain a resident default plus route");
        assert_eq!(output.output_id, "plus");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident DM primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly DM materialized a host result")
            }
        };
        let expected = dm_plus_with_kernel(
            &DmInput::from_slices(&ohlcv.high, &ohlcv.low, DmParams { period: Some(14) }),
            Kernel::Scalar,
        )
        .expect("the scalar DM primary must accept defaults");
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary plus[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary plus[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn didi_index_outputs_stay_resident_for_default_and_all_registry_ratio_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::DidiIndex.is_sequential(),
            "each Didi Index tuple owns one exact three-ring state thread"
        );
        assert!(
            F64Kernel::DidiIndex.is_period_invariant(),
            "the preserved generic primary remains fixed at canonical defaults"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_tuples = [
            (3, 8, 20),
            (1, 3, 7),
            (3, 8, 21),
            (8, 20, 50),
            (15, 40, 100),
            (30, 80, 200),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Didi Index resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_didi_index_outputs_device(&parameter_tuples)
            .expect("all Didi Index outputs must stay resident for every admitted tuple");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Didi Index re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "didi_index");
        assert_eq!(result.entry_point, "didi_index_batch_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_tuples.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            DIDI_INDEX_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident Didi Index matrices must retire");
    }

    #[test]
    fn didi_index_outputs_match_cpu_bits_for_default_and_all_registry_ratio_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::didi_index::{
            DidiIndexInput, DidiIndexParams, didi_index_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        for index in 0..3 {
            ohlcv.close[index] = f64::NAN;
        }
        ohlcv.close[90] = f64::NAN;
        let parameter_tuples = [
            (3, 8, 20),
            (1, 3, 7),
            (3, 8, 21),
            (8, 20, 50),
            (15, 40, 100),
            (30, 80, 200),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Didi Index parity test requires CUDA device 0");
        let result = engine
            .compute_didi_index_outputs_device(&parameter_tuples)
            .expect("the resident Didi Index route must accept every admitted tuple");

        let mut expected: BTreeMap<&str, Vec<f64>> = DIDI_INDEX_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (short_length, medium_length, long_length) in parameter_tuples {
            let output = didi_index_with_kernel(
                &DidiIndexInput::from_slice(
                    &ohlcv.close,
                    DidiIndexParams {
                        short_length: Some(short_length),
                        medium_length: Some(medium_length),
                        long_length: Some(long_length),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar Didi Index authority must accept the admitted tuple");
            expected.get_mut("short").unwrap().extend(output.short);
            expected.get_mut("long").unwrap().extend(output.long);
            expected
                .get_mut("crossover")
                .unwrap()
                .extend(output.crossover);
            expected
                .get_mut("crossunder")
                .unwrap()
                .extend(output.crossunder);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Didi Index output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected Didi Index output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn didi_index_primary_route_preserves_default_short_bits() {
        use vector_ta::indicators::didi_index::{
            DidiIndexInput, DidiIndexParams, didi_index_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[90] = f64::NAN;
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Didi Index primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("didi_index", &[20])
            .expect("the preserved primary ABI must remain a resident default short route");
        assert_eq!(output.output_id, "short");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Didi Index primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Didi Index materialized a host result")
            }
        };
        let expected = didi_index_with_kernel(
            &DidiIndexInput::from_slice(&ohlcv.close, DidiIndexParams::default()),
            Kernel::Scalar,
        )
        .expect("the scalar Didi Index primary must accept defaults")
        .short;
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary short[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary short[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn directional_imbalance_index_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::DirectionalImbalanceIndex.is_sequential(),
            "each Directional Imbalance Index tuple owns one exact four-ring state thread"
        );
        assert!(
            !F64Kernel::DirectionalImbalanceIndex.is_period_invariant(),
            "Directional Imbalance Index must consume every admitted period"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_tuples = [(10, 70), (10, 7), (10, 21), (10, 50), (10, 100), (10, 200)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Directional Imbalance Index resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_directional_imbalance_index_outputs_device(&parameter_tuples)
            .expect("all six outputs must stay resident for every admitted tuple");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Directional Imbalance Index re-uploaded resident high/low"
        );
        assert_eq!(result.indicator_id, "directional_imbalance_index");
        assert_eq!(result.entry_point, "directional_imbalance_index_batch_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_tuples.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_tuples.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all six resident Directional Imbalance Index matrices must retire");
    }

    #[test]
    fn directional_imbalance_index_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::directional_imbalance_index::{
            DirectionalImbalanceIndexInput, DirectionalImbalanceIndexParams,
            directional_imbalance_index_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        for index in 0..3 {
            ohlcv.high[index] = f64::NAN;
            ohlcv.low[index] = f64::NAN;
        }
        for index in 60..=64 {
            ohlcv.high[index] = ohlcv.high[59];
            ohlcv.low[index] = ohlcv.low[59];
        }
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.high[130] = f64::INFINITY;
        ohlcv.low[130] = f64::NEG_INFINITY;
        ohlcv.high[131] = -0.0;
        ohlcv.low[131] = -1.0;
        ohlcv.high[132] = 0.0;
        ohlcv.low[132] = -1.0;
        ohlcv.high[140] = f64::NAN;
        ohlcv.low[140] = f64::NAN;
        ohlcv.high[141] = 1.0;
        ohlcv.low[141] = 0.0;
        ohlcv.high[142] = 1.0;
        ohlcv.low[142] = -0.0;
        let parameter_tuples = [(10, 70), (10, 7), (10, 21), (10, 50), (10, 100), (10, 200)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Directional Imbalance Index parity test requires CUDA device 0");
        let result = engine
            .compute_directional_imbalance_index_outputs_device(&parameter_tuples)
            .expect("the resident route must accept the default and all five sweeps");

        let mut expected: BTreeMap<&str, Vec<f64>> = DIRECTIONAL_IMBALANCE_INDEX_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (length, period) in parameter_tuples {
            let output = directional_imbalance_index_with_kernel(
                &DirectionalImbalanceIndexInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    DirectionalImbalanceIndexParams {
                        length: Some(length),
                        period: Some(period),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar Directional Imbalance Index authority must accept the tuple");
            expected.get_mut("up").unwrap().extend(output.up);
            expected.get_mut("down").unwrap().extend(output.down);
            expected.get_mut("bulls").unwrap().extend(output.bulls);
            expected.get_mut("bears").unwrap().extend(output.bears);
            expected.get_mut("upper").unwrap().extend(output.upper);
            expected.get_mut("lower").unwrap().extend(output.lower);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Directional Imbalance Index output must download");
            let cpu = expected.get(output.output_id).unwrap_or_else(|| {
                panic!(
                    "unexpected Directional Imbalance Index output {}",
                    output.output_id
                )
            });
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn directional_imbalance_index_primary_route_preserves_default_up_bits() {
        use vector_ta::indicators::directional_imbalance_index::{
            DirectionalImbalanceIndexInput, DirectionalImbalanceIndexParams,
            directional_imbalance_index_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Directional Imbalance Index primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("directional_imbalance_index", &[70])
            .expect("the preserved primary ABI must remain a resident default up route");
        assert_eq!(output.output_id, "up");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Directional Imbalance Index primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Directional Imbalance Index materialized a host result")
            }
        };
        let expected = directional_imbalance_index_with_kernel(
            &DirectionalImbalanceIndexInput::from_slices(
                &ohlcv.high,
                &ohlcv.low,
                DirectionalImbalanceIndexParams::default(),
            ),
            Kernel::Scalar,
        )
        .expect("the scalar Directional Imbalance Index primary must accept defaults")
        .up;
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary up[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary up[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn disparity_index_outputs_stay_resident_for_default_and_all_lookback_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(F64Kernel::DisparityIndex.is_sequential());
        assert!(F64Kernel::DisparityIndex.is_period_invariant());
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (14_usize, 14_usize, 9_usize, false),
            (14, 7, 9, false),
            (14, 21, 9, false),
            (14, 50, 9, false),
            (14, 100, 9, false),
            (14, 200, 9, false),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Disparity Index resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let output = engine
            .compute_disparity_index_output_device(&parameter_rows)
            .expect("every canonical Disparity Index tuple must stay resident in one launch");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Disparity Index re-uploaded resident close"
        );
        assert_eq!(output.indicator_id, "disparity_index");
        assert_eq!(output.entry_point, "disparity_index_batch_f64");
        assert_eq!(
            (output.rows, output.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(output.outputs.len(), 1);
        assert_eq!(output.outputs[0].output_id, "value");
        assert_eq!(
            output.outputs[0].matrix.device_id(),
            engine.device_ordinal()
        );
        engine
            .synchronize()
            .expect("the resident Disparity Index matrix must retire");
    }

    #[test]
    fn disparity_index_outputs_match_cpu_bits_for_default_and_all_lookback_sweeps() {
        use vector_ta::indicators::disparity_index::{
            DisparityIndexInput, DisparityIndexParams, disparity_index_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        ohlcv.close[260] = f64::NAN;
        let parameter_rows = [
            (14_usize, 14_usize, 9_usize, false),
            (14, 7, 9, false),
            (14, 21, 9, false),
            (14, 50, 9, false),
            (14, 100, 9, false),
            (14, 200, 9, false),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Disparity Index parity test requires CUDA device 0");
        let output = engine
            .compute_disparity_index_output_device(&parameter_rows)
            .expect("the resident route must accept the default and all five sweeps");
        assert_eq!(output.outputs.len(), 1);
        assert_eq!(output.outputs[0].output_id, "value");
        let actual = engine
            .runtime
            .download_matrix_f64(&output.outputs[0].matrix)
            .expect("the resident Disparity Index matrix must download");

        let mut expected = Vec::with_capacity(actual.len());
        for &(ema_period, lookback_period, smoothing_period, smoothing_is_sma) in &parameter_rows {
            expected.extend(
                disparity_index_with_kernel(
                    &DisparityIndexInput::from_slice(
                        &ohlcv.close,
                        DisparityIndexParams {
                            ema_period: Some(ema_period),
                            lookback_period: Some(lookback_period),
                            smoothing_period: Some(smoothing_period),
                            smoothing_type: Some(
                                if smoothing_is_sma { "sma" } else { "ema" }.to_string(),
                            ),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar Disparity Index authority must accept every tuple")
                .values,
            );
        }

        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "value[{index}] lost CPU undefined behavior: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "value[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn disparity_index_primary_route_preserves_default_value_bits() {
        use vector_ta::indicators::disparity_index::{
            DisparityIndexInput, DisparityIndexParams, disparity_index_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[260] = f64::NAN;
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Disparity Index primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("disparity_index", &[14])
            .expect("the preserved primary ABI must remain a resident default value route");
        assert_eq!(output.output_id, "value");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Disparity Index primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Disparity Index materialized a host result")
            }
        };
        let expected = disparity_index_with_kernel(
            &DisparityIndexInput::from_slice(&ohlcv.close, DisparityIndexParams::default()),
            Kernel::Scalar,
        )
        .expect("the scalar Disparity Index primary must accept defaults")
        .values;
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary value[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary value[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn donchian_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            !F64Kernel::Donchian.is_sequential(),
            "each Donchian bar is an independent exact window reduction"
        );
        assert!(
            !F64Kernel::Donchian.is_period_invariant(),
            "Donchian must consume every admitted period"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let periods = [20, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Donchian resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_donchian_outputs_device(&periods)
            .expect("all Donchian outputs must stay resident for every admitted period");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Donchian re-uploaded its resident high/low input"
        );
        assert_eq!(result.indicator_id, "donchian");
        assert_eq!(result.entry_point, "donchian_all_outputs_batch_f64");
        assert_eq!((result.rows, result.cols), (periods.len(), ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            DONCHIAN_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == periods.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident Donchian matrices must retire");
    }

    #[test]
    fn donchian_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::donchian::{
            DonchianInput, DonchianParams, donchian_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        ohlcv.high[120] = f64::INFINITY;
        ohlcv.low[120] = f64::NEG_INFINITY;
        ohlcv.low[140] = 0.0;
        ohlcv.low[141] = -0.0;
        let periods = [20, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Donchian parity test requires CUDA device 0");
        let result = engine
            .compute_donchian_outputs_device(&periods)
            .expect("the resident Donchian route must accept default and all five sweeps");

        let mut expected: BTreeMap<&str, Vec<f64>> = DONCHIAN_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for period in periods {
            let output = donchian_with_kernel(
                &DonchianInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    DonchianParams {
                        period: Some(period),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar Donchian authority must accept the admitted period");
            expected.get_mut("upper").unwrap().extend(output.upperband);
            expected
                .get_mut("middle")
                .unwrap()
                .extend(output.middleband);
            expected.get_mut("lower").unwrap().extend(output.lowerband);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Donchian output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected Donchian output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn donchian_primary_route_preserves_default_upper_bits() {
        use vector_ta::indicators::donchian::{
            DonchianInput, DonchianParams, donchian_upper_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[90] = f64::NAN;
        ohlcv.low[90] = f64::NAN;
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Donchian primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("donchian", &[20])
            .expect("the preserved primary ABI must remain a resident default upper route");
        assert_eq!(output.output_id, "upper");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Donchian primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Donchian materialized a host result")
            }
        };
        let expected = donchian_upper_with_kernel(
            &DonchianInput::from_slices(
                &ohlcv.high,
                &ohlcv.low,
                DonchianParams { period: Some(20) },
            ),
            Kernel::Scalar,
        )
        .expect("the scalar Donchian primary must accept defaults");
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary upper[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary upper[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn dual_ulcer_index_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::DualUlcerIndex.is_sequential(),
            "Dual Ulcer Index carries window sums and cumulative threshold state"
        );
        assert!(
            !F64Kernel::DualUlcerIndex.is_period_invariant(),
            "Dual Ulcer Index must consume every admitted period"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (5, true, 0.1),
            (7, true, 0.1),
            (21, true, 0.1),
            (50, true, 0.1),
            (100, true, 0.1),
            (200, true, 0.1),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Dual Ulcer Index resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_dual_ulcer_index_outputs_device(&parameter_rows)
            .expect("all Dual Ulcer Index outputs must stay resident for every admitted period");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Dual Ulcer Index re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "dual_ulcer_index");
        assert_eq!(result.entry_point, "dual_ulcer_index_all_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            DUAL_ULCER_INDEX_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident Dual Ulcer Index matrices must retire");
    }

    #[test]
    fn dual_ulcer_index_outputs_match_cpu_bits_across_resets_and_period_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::dual_ulcer_index::{
            DualUlcerIndexInput, DualUlcerIndexParams, dual_ulcer_index_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[600] = f64::NAN;
        ohlcv.close[1_300] = 0.0;
        ohlcv.close[2_000] = f64::INFINITY;
        ohlcv.close[2_001] = -1.0;
        let parameter_rows = [
            (5, true, 0.1),
            (7, true, 0.1),
            (21, true, 0.1),
            (50, true, 0.1),
            (100, true, 0.1),
            (200, true, 0.1),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Dual Ulcer Index parity test requires CUDA device 0");
        let result = engine
            .compute_dual_ulcer_index_outputs_device(&parameter_rows)
            .expect("the resident Dual Ulcer Index route must accept all admitted tuples");

        let mut expected: BTreeMap<&str, Vec<f64>> = DUAL_ULCER_INDEX_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (period, auto_threshold, threshold) in parameter_rows {
            let output = dual_ulcer_index_with_kernel(
                &DualUlcerIndexInput::from_slice(
                    &ohlcv.close,
                    DualUlcerIndexParams {
                        period: Some(period),
                        auto_threshold: Some(auto_threshold),
                        threshold: Some(threshold),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar Dual Ulcer Index authority must accept the admitted tuple");
            expected
                .get_mut("long_ulcer")
                .unwrap()
                .extend(output.long_ulcer);
            expected
                .get_mut("short_ulcer")
                .unwrap()
                .extend(output.short_ulcer);
            expected
                .get_mut("threshold")
                .unwrap()
                .extend(output.threshold);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Dual Ulcer Index output must download for parity");
            let cpu = expected.get(output.output_id).unwrap_or_else(|| {
                panic!("unexpected Dual Ulcer Index output {}", output.output_id)
            });
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn dual_ulcer_index_primary_route_preserves_default_long_ulcer_bits() {
        use vector_ta::indicators::dual_ulcer_index::{
            DualUlcerIndexInput, DualUlcerIndexOutputField, DualUlcerIndexParams,
            dual_ulcer_index_output_into_slice,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[600] = f64::NAN;
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Dual Ulcer Index primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("dual_ulcer_index", &[5])
            .expect("the preserved primary ABI must remain a resident default long-ulcer route");
        assert_eq!(output.output_id, "long_ulcer");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident Dual Ulcer Index primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Dual Ulcer Index materialized a host result")
            }
        };
        let mut expected = vec![f64::NAN; ohlcv.len()];
        dual_ulcer_index_output_into_slice(
            &mut expected,
            &DualUlcerIndexInput::from_slice(&ohlcv.close, DualUlcerIndexParams::default()),
            Kernel::Scalar,
            DualUlcerIndexOutputField::LongUlcer,
        )
        .expect("the scalar Dual Ulcer Index primary must accept defaults");
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary long_ulcer[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary long_ulcer[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn dvdiqqe_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Dvdiqqe.is_sequential(),
            "DVDIQQE carries PVI/NVI, six EMA, ratchet, and center state"
        );
        assert!(
            !F64Kernel::Dvdiqqe.is_period_invariant(),
            "DVDIQQE must consume every admitted period"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (13, 6, 2.618, 4.236, false, true, 0.01),
            (7, 6, 2.618, 4.236, false, true, 0.01),
            (21, 6, 2.618, 4.236, false, true, 0.01),
            (50, 6, 2.618, 4.236, false, true, 0.01),
            (100, 6, 2.618, 4.236, false, true, 0.01),
            (200, 6, 2.618, 4.236, false, true, 0.01),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the DVDIQQE resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_dvdiqqe_outputs_device(&parameter_rows)
            .expect("all DVDIQQE outputs must stay resident for every admitted period");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "DVDIQQE re-uploaded its resident open/close/volume input"
        );
        assert_eq!(result.indicator_id, "dvdiqqe");
        assert_eq!(result.entry_point, "dvdiqqe_all_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            DVDIQQE_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident DVDIQQE matrices must retire");
    }

    #[test]
    fn dvdiqqe_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::dvdiqqe::{
            DvdiqqeInput, DvdiqqeOutputField, DvdiqqeParams, dvdiqqe_output_into_slice,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.open[1] = ohlcv.close[1];
        ohlcv.open[777] = ohlcv.close[777] - 0.001;
        let volume = ohlcv.volume.as_mut().expect("fixture volume");
        volume[700] = f64::NAN;
        volume[1_300] = f64::INFINITY;
        let parameter_rows = [
            (13, 6, 2.618, 4.236, false, true, 0.01),
            (7, 6, 2.618, 4.236, false, true, 0.01),
            (21, 6, 2.618, 4.236, false, true, 0.01),
            (50, 6, 2.618, 4.236, false, true, 0.01),
            (100, 6, 2.618, 4.236, false, true, 0.01),
            (200, 6, 2.618, 4.236, false, true, 0.01),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the DVDIQQE parity test requires CUDA device 0");
        let result = engine
            .compute_dvdiqqe_outputs_device(&parameter_rows)
            .expect("the resident DVDIQQE route must accept all admitted tuples");

        let mut expected: BTreeMap<&str, Vec<f64>> = DVDIQQE_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (period, smoothing, fast, slow, use_tick_only, dynamic_center, tick) in parameter_rows {
            let input = DvdiqqeInput::from_slices(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                ohlcv.volume.as_deref(),
                DvdiqqeParams {
                    period: Some(period),
                    smoothing_period: Some(smoothing),
                    fast_multiplier: Some(fast),
                    slow_multiplier: Some(slow),
                    volume_type: Some(if use_tick_only { "tick" } else { "default" }.to_string()),
                    center_type: Some(
                        if dynamic_center { "dynamic" } else { "static" }.to_string(),
                    ),
                    tick_size: Some(tick),
                },
            );
            for (output_id, field) in DVDIQQE_OUTPUT_IDS.into_iter().zip([
                DvdiqqeOutputField::Dvdi,
                DvdiqqeOutputField::FastTl,
                DvdiqqeOutputField::SlowTl,
                DvdiqqeOutputField::CenterLine,
            ]) {
                let mut row = vec![f64::NAN; ohlcv.len()];
                dvdiqqe_output_into_slice(&mut row, &input, Kernel::Scalar, field)
                    .expect("the scalar DVDIQQE authority must accept the admitted tuple");
                expected.get_mut(output_id).unwrap().extend(row);
            }
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident DVDIQQE output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected DVDIQQE output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn dvdiqqe_primary_route_preserves_default_dvdi_bits() {
        use vector_ta::indicators::dvdiqqe::{
            DvdiqqeInput, DvdiqqeOutputField, DvdiqqeParams, dvdiqqe_output_into_slice,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.volume.as_mut().expect("fixture volume")[700] = f64::NAN;
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the DVDIQQE primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("dvdiqqe", &[13])
            .expect("the preserved primary ABI must remain a resident default DVDI route");
        assert_eq!(output.output_id, "dvdi");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("the resident DVDIQQE primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly DVDIQQE materialized a host result")
            }
        };
        let mut expected = vec![f64::NAN; ohlcv.len()];
        dvdiqqe_output_into_slice(
            &mut expected,
            &DvdiqqeInput::from_slices(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                ohlcv.volume.as_deref(),
                DvdiqqeParams::default(),
            ),
            Kernel::Scalar,
            DvdiqqeOutputField::Dvdi,
        )
        .expect("the scalar DVDIQQE primary must accept defaults");
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary dvdi[{index}] lost CPU NaN: gpu={gpu:?}"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary dvdi[{index}] is not exact scalar CPU f64 parity: \
                     gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn ehlers_autocorrelation_periodogram_outputs_stay_resident_for_all_admitted_tuples() {
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (8, 48, 3, true),
            (4, 21, 1, true),
            (8, 50, 3, true),
            (17, 100, 6, true),
            (33, 200, 13, true),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the periodogram resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_ehlers_autocorrelation_periodogram_outputs_device(&parameter_rows)
            .expect("both periodogram outputs must stay resident for every admitted tuple");

        assert_eq!(
            engine.uploads(),
            uploads_before,
            "the periodogram route re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "ehlers_autocorrelation_periodogram");
        assert_eq!(
            result.entry_point,
            "ehlers_autocorrelation_periodogram_outputs_f64"
        );
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("both resident periodogram matrices must retire");
    }

    #[test]
    fn ehlers_autocorrelation_periodogram_outputs_match_cpu_bits_for_all_admitted_tuples() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::ehlers_autocorrelation_periodogram::{
            EhlersAutocorrelationPeriodogramInput, EhlersAutocorrelationPeriodogramParams,
            ehlers_autocorrelation_periodogram_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        ohlcv.close[700] = f64::NAN;
        ohlcv.close[1_300] = f64::INFINITY;
        let parameter_rows = [
            (8, 48, 3, true),
            (4, 21, 1, true),
            (8, 50, 3, true),
            (17, 100, 6, true),
            (33, 200, 13, true),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the periodogram parity test requires CUDA device 0");
        let result = engine
            .compute_ehlers_autocorrelation_periodogram_outputs_device(&parameter_rows)
            .expect("the resident periodogram route must accept every admitted tuple");

        let mut expected: BTreeMap<&str, Vec<f64>> = EHLERS_AUTOCORRELATION_PERIODOGRAM_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (min_period, max_period, avg_length, enhance) in parameter_rows {
            let output = ehlers_autocorrelation_periodogram_with_kernel(
                &EhlersAutocorrelationPeriodogramInput::from_slice(
                    &ohlcv.close,
                    EhlersAutocorrelationPeriodogramParams {
                        min_period: Some(min_period),
                        max_period: Some(max_period),
                        avg_length: Some(avg_length),
                        enhance: Some(enhance),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar periodogram authority must accept the admitted tuple");
            expected
                .get_mut("dominant_cycle")
                .expect("dominant-cycle oracle")
                .extend(output.dominant_cycle);
            expected
                .get_mut("normalized_power")
                .expect("normalized-power oracle")
                .extend(output.normalized_power);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("{} download failed: {error}", output.output_id));
            let expected = expected
                .get(output.output_id)
                .expect("every returned output needs one scalar oracle");
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn ehlers_data_sampling_rsi_outputs_stay_resident_for_default_and_length_sweeps() {
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let lengths = [14, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the EDSRSI resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_ehlers_data_sampling_rsi_outputs_device(&lengths)
            .expect("all EDSRSI outputs must stay resident for every admitted length");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "EDSRSI re-uploaded resident prices"
        );
        assert_eq!(
            result.indicator_id,
            "ehlers_data_sampling_relative_strength_indicator"
        );
        assert_eq!(
            result.entry_point,
            "ehlers_data_sampling_relative_strength_indicator_batch_f64"
        );
        assert_eq!((result.rows, result.cols), (lengths.len(), ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            EHLERS_DATA_SAMPLING_RSI_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == lengths.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("resident EDSRSI matrices must retire");
    }

    #[test]
    fn ehlers_data_sampling_rsi_outputs_match_cpu_bits_for_default_and_length_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::ehlers_data_sampling_relative_strength_indicator::{
            EhlersDataSamplingRelativeStrengthIndicatorInput,
            EhlersDataSamplingRelativeStrengthIndicatorParams,
            ehlers_data_sampling_relative_strength_indicator_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.open[..3].fill(f64::NAN);
        ohlcv.close[..3].fill(f64::NAN);
        ohlcv.open[700] = f64::NAN;
        ohlcv.close[1_300] = f64::INFINITY;
        let lengths = [14, 7, 21, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the EDSRSI parity test requires CUDA device 0");
        let result = engine
            .compute_ehlers_data_sampling_rsi_outputs_device(&lengths)
            .expect("the resident EDSRSI route must accept every admitted length");

        let mut expected: BTreeMap<&str, Vec<f64>> = EHLERS_DATA_SAMPLING_RSI_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for length in lengths {
            let output = ehlers_data_sampling_relative_strength_indicator_with_kernel(
                &EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
                    &ohlcv.open,
                    &ohlcv.close,
                    EhlersDataSamplingRelativeStrengthIndicatorParams {
                        length: Some(length),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("scalar EDSRSI must accept every admitted length");
            expected.get_mut("ds_rsi").unwrap().extend(output.ds_rsi);
            expected
                .get_mut("original_rsi")
                .unwrap()
                .extend(output.original_rsi);
            expected.get_mut("signal").unwrap().extend(output.signal);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("{} download failed: {error}", output.output_id));
            let expected = expected.get(output.output_id).unwrap();
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(gpu.is_nan(), "{}[{index}] lost CPU NaN", output.output_id);
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn ehlers_data_sampling_rsi_primary_consumes_requested_length_and_matches_cpu_bits() {
        use vector_ta::indicators::ehlers_data_sampling_relative_strength_indicator::{
            EhlersDataSamplingRelativeStrengthIndicatorInput,
            EhlersDataSamplingRelativeStrengthIndicatorParams,
            ehlers_data_sampling_relative_strength_indicator_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let ohlcv = repeated_ctrader_bandpass_fixture();
        let lengths = [7, 21];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the EDSRSI primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("ehlers_data_sampling_relative_strength_indicator", &lengths)
            .expect("the preserved primary ABI must consume requested lengths");
        assert_eq!(output.output_id, "ds_rsi");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident EDSRSI primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => panic!("GpuOnly EDSRSI materialized HostF64"),
        };
        let mut expected = Vec::with_capacity(actual.len());
        for length in lengths {
            let output = ehlers_data_sampling_relative_strength_indicator_with_kernel(
                &EhlersDataSamplingRelativeStrengthIndicatorInput::from_slices(
                    &ohlcv.open,
                    &ohlcv.close,
                    EhlersDataSamplingRelativeStrengthIndicatorParams {
                        length: Some(length),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("scalar EDSRSI primary must accept requested length");
            expected.extend(output.ds_rsi);
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "primary ds_rsi[{index}] lost CPU NaN");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary ds_rsi[{index}] is not exact scalar CPU f64 parity"
                );
            }
        }
    }

    #[test]
    fn ehlers_linear_extrapolation_predictor_outputs_stay_resident_for_all_admitted_tuples() {
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (125, 12, 0.7, 5, 0),
            (7, 1, 0.7, 5, 0),
            (21, 2, 0.7, 5, 0),
            (50, 5, 0.7, 5, 0),
            (100, 10, 0.7, 5, 0),
            (200, 19, 0.7, 5, 0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the ELEP resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_ehlers_linear_extrapolation_predictor_outputs_device(&parameter_rows)
            .expect("all five ELEP outputs must stay resident for every admitted tuple");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "ELEP re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "ehlers_linear_extrapolation_predictor");
        assert_eq!(
            result.entry_point,
            "ehlers_linear_extrapolation_predictor_outputs_f64"
        );
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident ELEP matrices must retire");
    }

    #[test]
    fn ehlers_linear_extrapolation_predictor_outputs_match_cpu_bits_for_all_admitted_tuples() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::ehlers_linear_extrapolation_predictor::{
            EhlersLinearExtrapolationPredictorInput, EhlersLinearExtrapolationPredictorParams,
            ehlers_linear_extrapolation_predictor_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        ohlcv.close[700] = f64::NAN;
        ohlcv.close[1_300] = f64::INFINITY;
        let parameter_rows = [
            (125, 12, 0.7, 5, 0),
            (7, 1, 0.7, 5, 0),
            (21, 2, 0.7, 5, 0),
            (50, 5, 0.7, 5, 0),
            (100, 10, 0.7, 5, 0),
            (200, 19, 0.7, 5, 0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the ELEP parity test requires CUDA device 0");
        let result = engine
            .compute_ehlers_linear_extrapolation_predictor_outputs_device(&parameter_rows)
            .expect("resident ELEP must accept every admitted tuple");

        let mut expected: BTreeMap<&str, Vec<f64>> =
            EHLERS_LINEAR_EXTRAPOLATION_PREDICTOR_OUTPUT_IDS
                .into_iter()
                .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
                .collect();
        for (high_pass_length, low_pass_length, gain, bars_forward, signal_mode) in parameter_rows {
            assert_eq!(
                signal_mode, 0,
                "the admitted graph pins the canonical default mode"
            );
            let output = ehlers_linear_extrapolation_predictor_with_kernel(
                &EhlersLinearExtrapolationPredictorInput::from_slice(
                    &ohlcv.close,
                    EhlersLinearExtrapolationPredictorParams {
                        high_pass_length: Some(high_pass_length),
                        low_pass_length: Some(low_pass_length),
                        gain: Some(gain),
                        bars_forward: Some(bars_forward),
                        signal_mode: Some("predict_filter_crosses".to_string()),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar ELEP authority must accept the admitted tuple");
            expected
                .get_mut("prediction")
                .unwrap()
                .extend(output.prediction);
            expected.get_mut("filter").unwrap().extend(output.filter);
            expected.get_mut("state").unwrap().extend(output.state);
            expected.get_mut("go_long").unwrap().extend(output.go_long);
            expected
                .get_mut("go_short")
                .unwrap()
                .extend(output.go_short);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("{} download failed: {error}", output.output_id));
            let expected = expected
                .get(output.output_id)
                .expect("every returned ELEP output needs one scalar oracle");
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn ehlers_undersampled_double_moving_average_outputs_stay_resident_for_all_admitted_tuples() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::EhlersUndersampledDoubleMovingAverage.is_sequential(),
            "each EUDMA tuple owns two recursive Hann rings"
        );
        assert!(
            F64Kernel::EhlersUndersampledDoubleMovingAverage.is_period_invariant(),
            "the preserved generic EUDMA primary remains fixed-default; typed production must bypass it"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (6, 12, 5),
            (4, 7, 3),
            (11, 21, 9),
            (25, 50, 21),
            (50, 100, 42),
            (100, 200, 83),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the EUDMA resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_ehlers_undersampled_double_moving_average_outputs_device(&parameter_rows)
            .expect("both EUDMA outputs must stay resident for every admitted tuple");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "EUDMA re-uploaded its resident close input"
        );
        assert_eq!(
            result.indicator_id,
            "ehlers_undersampled_double_moving_average"
        );
        assert_eq!(
            result.entry_point,
            "ehlers_undersampled_double_moving_average_outputs_f64"
        );
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("both resident EUDMA matrices must retire");
    }

    #[test]
    fn ehlers_undersampled_double_moving_average_outputs_match_cpu_bits_for_all_admitted_tuples() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::moving_averages::ehlers_undersampled_double_moving_average::{
            EhlersUndersampledDoubleMovingAverageInput,
            EhlersUndersampledDoubleMovingAverageParams,
            ehlers_undersampled_double_moving_average_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        ohlcv.close[700] = f64::NAN;
        ohlcv.close[1_300] = f64::INFINITY;
        let parameter_rows = [
            (6, 12, 5),
            (4, 7, 3),
            (11, 21, 9),
            (25, 50, 21),
            (50, 100, 42),
            (100, 200, 83),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the EUDMA parity test requires CUDA device 0");
        let result = engine
            .compute_ehlers_undersampled_double_moving_average_outputs_device(&parameter_rows)
            .expect("resident EUDMA must accept every admitted tuple");

        let mut expected: BTreeMap<&str, Vec<f64>> =
            EHLERS_UNDERSAMPLED_DOUBLE_MOVING_AVERAGE_OUTPUT_IDS
                .into_iter()
                .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
                .collect();
        for (fast_length, slow_length, sample_length) in parameter_rows {
            let output = ehlers_undersampled_double_moving_average_with_kernel(
                &EhlersUndersampledDoubleMovingAverageInput::from_slice(
                    &ohlcv.close,
                    EhlersUndersampledDoubleMovingAverageParams {
                        fast_length: Some(fast_length),
                        slow_length: Some(slow_length),
                        sample_length: Some(sample_length),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar EUDMA authority must accept the admitted tuple");
            expected.get_mut("fast").unwrap().extend(output.fast);
            expected.get_mut("slow").unwrap().extend(output.slow);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("{} download failed: {error}", output.output_id));
            let expected = expected
                .get(output.output_id)
                .expect("every returned EUDMA output needs one scalar oracle");
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn ema_deviation_corrected_t3_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        const OUTPUT_IDS: [&str; 2] = ["corrected", "t3"];
        assert!(
            F64Kernel::EmaDeviationCorrectedT3.is_sequential(),
            "each EDCT3 tuple owns one recursive T3/deviation state"
        );
        assert!(
            !F64Kernel::EmaDeviationCorrectedT3.is_period_invariant(),
            "EDCT3 must consume the canonical period-10 base and every admitted sweep"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (10, 0.7, 0),
            (7, 0.7, 0),
            (21, 0.7, 0),
            (50, 0.7, 0),
            (100, 0.7, 0),
            (200, 0.7, 0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the EDCT3 resident test requires CUDA device 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_ema_deviation_corrected_t3_outputs_device(&parameter_rows)
            .expect("both EDCT3 outputs must stay resident for every admitted tuple");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "EDCT3 re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "ema_deviation_corrected_t3");
        assert_eq!(result.entry_point, "ema_deviation_corrected_t3_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("both resident EDCT3 matrices must retire");
    }

    #[test]
    fn ema_deviation_corrected_t3_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::moving_averages::ema_deviation_corrected_t3::{
            EmaDeviationCorrectedT3Input, EmaDeviationCorrectedT3Params,
            ema_deviation_corrected_t3_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        const OUTPUT_IDS: [&str; 2] = ["corrected", "t3"];
        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        ohlcv.close[700] = f64::NAN;
        ohlcv.close[1_300] = f64::INFINITY;
        let parameter_rows = [
            (10, 0.7, 0),
            (7, 0.7, 0),
            (21, 0.7, 0),
            (50, 0.7, 0),
            (100, 0.7, 0),
            (200, 0.7, 0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the EDCT3 parity test requires CUDA device 0");
        let result = engine
            .compute_ema_deviation_corrected_t3_outputs_device(&parameter_rows)
            .expect("resident EDCT3 must accept every admitted tuple");

        let mut expected: BTreeMap<&str, Vec<f64>> = OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (period, hot, t3_mode) in parameter_rows {
            let output = ema_deviation_corrected_t3_with_kernel(
                &EmaDeviationCorrectedT3Input::from_slice(
                    &ohlcv.close,
                    EmaDeviationCorrectedT3Params {
                        period: Some(period),
                        hot: Some(hot),
                        t3_mode: Some(t3_mode),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar EDCT3 authority must accept the admitted tuple");
            expected
                .get_mut("corrected")
                .unwrap()
                .extend(output.corrected);
            expected.get_mut("t3").unwrap().extend(output.t3);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("{} download failed: {error}", output.output_id));
            let expected = expected
                .get(output.output_id)
                .expect("every returned EDCT3 output needs one scalar oracle");
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn ema_deviation_corrected_t3_primary_matches_cpu_bits_for_requested_periods() {
        use vector_ta::indicators::moving_averages::ema_deviation_corrected_t3::{
            EmaDeviationCorrectedT3Input, EmaDeviationCorrectedT3Params,
            ema_deviation_corrected_t3_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[700] = f64::NAN;
        let periods = [10, 7, 21];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the EDCT3 primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("ema_deviation_corrected_t3", &periods)
            .expect("the preserved EDCT3 primary ABI must consume requested periods");
        assert_eq!(output.output_id, "corrected");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident EDCT3 primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => panic!("GpuOnly EDCT3 materialized HostF64"),
        };
        let mut expected = Vec::with_capacity(actual.len());
        for period in periods {
            let output = ema_deviation_corrected_t3_with_kernel(
                &EmaDeviationCorrectedT3Input::from_slice(
                    &ohlcv.close,
                    EmaDeviationCorrectedT3Params {
                        period: Some(period),
                        hot: Some(0.7),
                        t3_mode: Some(0),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("scalar EDCT3 primary must accept requested period");
            expected.extend(output.corrected);
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "primary corrected[{index}] lost CPU NaN");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary corrected[{index}] is not exact scalar CPU f64 parity"
                );
            }
        }
    }

    #[test]
    fn emd_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Emd.is_sequential(),
            "each EMD tuple owns one bandpass recurrence and three running sums"
        );
        assert!(
            !F64Kernel::Emd.is_period_invariant(),
            "EMD must consume the canonical period-20 base and every admitted sweep"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (20, 0.5, 0.1),
            (7, 0.5, 0.1),
            (21, 0.5, 0.1),
            (50, 0.5, 0.1),
            (100, 0.5, 0.1),
            (200, 0.5, 0.1),
        ];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the EMD resident test requires CUDA 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_emd_outputs_device(&parameter_rows)
            .expect("all three EMD outputs must stay resident for every admitted tuple");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "EMD re-uploaded its resident high/low inputs"
        );
        assert_eq!(result.indicator_id, "emd");
        assert_eq!(result.entry_point, "emd_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            EMD_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident EMD matrices must retire");
    }

    #[test]
    fn emd_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::emd::{EmdInput, EmdParams, emd_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        let parameter_rows = [
            (20, 0.5, 0.1),
            (7, 0.5, 0.1),
            (21, 0.5, 0.1),
            (50, 0.5, 0.1),
            (100, 0.5, 0.1),
            (200, 0.5, 0.1),
        ];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the EMD parity test requires CUDA 0");
        let result = engine
            .compute_emd_outputs_device(&parameter_rows)
            .expect("resident EMD must accept every admitted tuple");

        let mut expected: BTreeMap<&str, Vec<f64>> = EMD_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (period, delta, fraction) in parameter_rows {
            let output = emd_with_kernel(
                &EmdInput::from_high_low_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    EmdParams {
                        period: Some(period),
                        delta: Some(delta),
                        fraction: Some(fraction),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar EMD authority must accept every admitted tuple");
            expected
                .get_mut("upperband")
                .unwrap()
                .extend(output.upperband);
            expected
                .get_mut("middleband")
                .unwrap()
                .extend(output.middleband);
            expected
                .get_mut("lowerband")
                .unwrap()
                .extend(output.lowerband);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("{} download failed: {error}", output.output_id));
            let expected = expected
                .get(output.output_id)
                .expect("every returned EMD output needs one scalar oracle");
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn emd_trend_outputs_stay_resident_for_default_and_all_length_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::EmdTrend.is_sequential(),
            "each EMD Trend tuple owns one exact SMA/deviation/direction state"
        );
        assert!(
            F64Kernel::EmdTrend.is_period_invariant(),
            "the preserved generic primary remains fixed at length 28; typed production must bypass it for every canonical length tuple"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (28, 1.0),
            (7, 1.0),
            (21, 1.0),
            (50, 1.0),
            (100, 1.0),
            (200, 1.0),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the EMD Trend resident test requires CUDA 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_emd_trend_outputs_device(&parameter_rows)
            .expect("all four EMD Trend outputs must stay resident for every admitted tuple");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "EMD Trend re-uploaded its resident close input"
        );
        assert_eq!(result.indicator_id, "emd_trend");
        assert_eq!(result.entry_point, "emd_trend_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            EMD_TREND_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident EMD Trend matrices must retire");
    }

    #[test]
    fn emd_trend_outputs_match_cpu_bits_for_default_and_all_length_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::emd_trend::{
            EmdTrendInput, EmdTrendParams, emd_trend_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let parameter_rows = [
            (28, 1.0),
            (7, 1.0),
            (21, 1.0),
            (50, 1.0),
            (100, 1.0),
            (200, 1.0),
        ];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the EMD Trend parity test requires CUDA 0");
        let result = engine
            .compute_emd_trend_outputs_device(&parameter_rows)
            .expect("resident EMD Trend must accept every admitted tuple");

        let mut expected: BTreeMap<&str, Vec<f64>> = EMD_TREND_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (length, mult) in parameter_rows {
            let output = emd_trend_with_kernel(
                &EmdTrendInput::from_slices(
                    &ohlcv.open,
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    EmdTrendParams {
                        source: Some("close".to_string()),
                        avg_type: Some("SMA".to_string()),
                        length: Some(length),
                        mult: Some(mult),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar EMD Trend authority must accept every admitted tuple");
            expected
                .get_mut("direction")
                .unwrap()
                .extend(output.direction);
            expected.get_mut("average").unwrap().extend(output.average);
            expected.get_mut("upper").unwrap().extend(output.upper);
            expected.get_mut("lower").unwrap().extend(output.lower);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("{} download failed: {error}", output.output_id));
            let expected = expected
                .get(output.output_id)
                .expect("every returned EMD Trend output needs one scalar oracle");
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn emd_trend_primary_preserves_fixed_default_average_cpu_bits() {
        use vector_ta::indicators::emd_trend::{
            EmdTrendInput, EmdTrendParams, emd_trend_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let requested_periods = [28, 7];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the EMD Trend primary test requires CUDA device 0");
        let output = engine
            .compute_primary_device("emd_trend", &requested_periods)
            .expect("the preserved EMD Trend primary ABI must remain available");
        assert_eq!(output.output_id, "average");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident EMD Trend primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly EMD Trend primary materialized HostF64")
            }
        };
        let canonical = emd_trend_with_kernel(
            &EmdTrendInput::from_slices(
                &ohlcv.open,
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                EmdTrendParams {
                    source: Some("close".to_string()),
                    avg_type: Some("SMA".to_string()),
                    length: Some(28),
                    mult: Some(1.0),
                },
            ),
            Kernel::Scalar,
        )
        .expect("the scalar EMD Trend authority must accept its fixed compatibility default")
        .average;
        let expected = canonical
            .iter()
            .copied()
            .cycle()
            .take(requested_periods.len() * canonical.len())
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "primary average[{index}] lost CPU NaN");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary average[{index}] is not exact fixed-default scalar CPU f64 parity"
                );
            }
        }
    }

    #[test]
    fn eri_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Eri.is_sequential(),
            "each ERI tuple owns one exact EMA recurrence"
        );
        assert!(
            !F64Kernel::Eri.is_period_invariant(),
            "ERI must consume the canonical period-13 base and every admitted sweep"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let periods = [13, 7, 21, 50, 100, 200];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the ERI resident test requires CUDA 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_eri_outputs_device(&periods)
            .expect("both ERI outputs must stay resident for every admitted period");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "ERI re-uploaded its resident high/low/close inputs"
        );
        assert_eq!(result.indicator_id, "eri");
        assert_eq!(result.entry_point, "eri_outputs_f64");
        assert_eq!((result.rows, result.cols), (periods.len(), ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ERI_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == periods.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("both resident ERI matrices must retire");
    }

    #[test]
    fn eri_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::eri::{EriInput, EriParams, eri_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        ohlcv.close[..3].fill(f64::NAN);
        let periods = [13, 7, 21, 50, 100, 200];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the ERI parity test requires CUDA 0");
        let result = engine
            .compute_eri_outputs_device(&periods)
            .expect("resident ERI must accept every admitted period");

        let mut expected: BTreeMap<&str, Vec<f64>> = ERI_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for period in periods {
            let output = eri_with_kernel(
                &EriInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    EriParams {
                        period: Some(period),
                        ma_type: Some("ema".to_string()),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar ERI authority must accept every admitted period");
            expected.get_mut("bull").unwrap().extend(output.bull);
            expected.get_mut("bear").unwrap().extend(output.bear);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("{} download failed: {error}", output.output_id));
            let expected = expected
                .get(output.output_id)
                .expect("every returned ERI output needs one scalar oracle");
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn eri_primary_matches_cpu_bits_for_requested_periods() {
        use vector_ta::indicators::eri::{EriInput, EriParams, eri_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        ohlcv.close[..3].fill(f64::NAN);
        let periods = [13, 7, 21];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the ERI primary test requires CUDA 0");
        let output = engine
            .compute_primary_device("eri", &periods)
            .expect("the preserved ERI primary ABI must consume requested periods");
        assert_eq!(output.output_id, "bull");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident ERI primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => panic!("GpuOnly ERI materialized HostF64"),
        };
        let mut expected = Vec::with_capacity(actual.len());
        for period in periods {
            let output = eri_with_kernel(
                &EriInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    EriParams {
                        period: Some(period),
                        ma_type: Some("ema".to_string()),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar ERI primary must accept requested period");
            expected.extend(output.bull);
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "primary bull[{index}] lost CPU NaN");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary bull[{index}] is not exact scalar CPU f64 parity"
                );
            }
        }
    }

    #[test]
    fn evasive_supertrend_outputs_stay_resident_for_default_and_all_atr_length_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::EvasiveSupertrend.is_sequential(),
            "each Evasive Supertrend tuple owns one exact ATR/trend recurrence"
        );
        assert!(
            !F64Kernel::EvasiveSupertrend.is_period_invariant(),
            "Evasive Supertrend must consume canonical atr_length 10 and every admitted sweep"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (10, 3.0, 1.0, 0.5),
            (7, 3.0, 1.0, 0.5),
            (21, 3.0, 1.0, 0.5),
            (50, 3.0, 1.0, 0.5),
            (100, 3.0, 1.0, 0.5),
            (200, 3.0, 1.0, 0.5),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Evasive Supertrend resident test requires CUDA 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_evasive_supertrend_outputs_device(&parameter_rows)
            .expect("all four Evasive Supertrend outputs must stay resident");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Evasive Supertrend re-uploaded its resident OHLC inputs"
        );
        assert_eq!(result.indicator_id, "evasive_supertrend");
        assert_eq!(result.entry_point, "evasive_supertrend_batch_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            EVASIVE_SUPERTREND_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident Evasive Supertrend matrices must retire");
    }

    #[test]
    fn evasive_supertrend_outputs_match_cpu_bits_for_default_and_all_atr_length_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::evasive_supertrend::{
            EvasiveSuperTrendInput, EvasiveSuperTrendParams, evasive_supertrend_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        for index in [3, 311] {
            ohlcv.open[index] = f64::NAN;
            ohlcv.high[index] = f64::NAN;
            ohlcv.low[index] = f64::NAN;
            ohlcv.close[index] = f64::NAN;
        }
        let parameter_rows = [
            (10, 3.0, 1.0, 0.5),
            (7, 3.0, 1.0, 0.5),
            (21, 3.0, 1.0, 0.5),
            (50, 3.0, 1.0, 0.5),
            (100, 3.0, 1.0, 0.5),
            (200, 3.0, 1.0, 0.5),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Evasive Supertrend parity test requires CUDA 0");
        let result = engine
            .compute_evasive_supertrend_outputs_device(&parameter_rows)
            .expect("resident Evasive Supertrend must accept every admitted atr_length");

        let mut expected: BTreeMap<&str, Vec<f64>> = EVASIVE_SUPERTREND_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (atr_length, base_multiplier, noise_threshold, expansion_alpha) in parameter_rows {
            let output = evasive_supertrend_with_kernel(
                &EvasiveSuperTrendInput::from_slices(
                    &ohlcv.open,
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    EvasiveSuperTrendParams {
                        atr_length: Some(atr_length),
                        base_multiplier: Some(base_multiplier),
                        noise_threshold: Some(noise_threshold),
                        expansion_alpha: Some(expansion_alpha),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar Evasive Supertrend authority must accept every admitted tuple");
            expected.get_mut("band").unwrap().extend(output.band);
            expected.get_mut("state").unwrap().extend(output.state);
            expected.get_mut("noisy").unwrap().extend(output.noisy);
            expected.get_mut("changed").unwrap().extend(output.changed);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("{} download failed: {error}", output.output_id));
            let expected = expected
                .get(output.output_id)
                .expect("every returned Evasive Supertrend output needs one scalar oracle");
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn evasive_supertrend_primary_matches_cpu_bits_for_requested_atr_lengths() {
        use vector_ta::indicators::evasive_supertrend::{
            EvasiveSuperTrendInput, EvasiveSuperTrendParams, evasive_supertrend_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        for index in [3, 311] {
            ohlcv.open[index] = f64::NAN;
            ohlcv.high[index] = f64::NAN;
            ohlcv.low[index] = f64::NAN;
            ohlcv.close[index] = f64::NAN;
        }
        let periods = [10, 7, 21];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Evasive Supertrend primary test requires CUDA 0");
        let output = engine
            .compute_primary_device("evasive_supertrend", &periods)
            .expect("the preserved Evasive Supertrend primary ABI must consume atr_length");
        assert_eq!(output.output_id, "band");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident Evasive Supertrend primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Evasive Supertrend materialized HostF64")
            }
        };
        let mut expected = Vec::with_capacity(actual.len());
        for atr_length in periods {
            let output = evasive_supertrend_with_kernel(
                &EvasiveSuperTrendInput::from_slices(
                    &ohlcv.open,
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    EvasiveSuperTrendParams {
                        atr_length: Some(atr_length),
                        base_multiplier: Some(3.0),
                        noise_threshold: Some(1.0),
                        expansion_alpha: Some(0.5),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar Evasive Supertrend primary must accept requested atr_length");
            expected.extend(output.band);
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "primary band[{index}] lost CPU NaN");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary band[{index}] is not exact scalar CPU f64 parity"
                );
            }
        }
    }

    #[test]
    fn fibonacci_entry_bands_primary_consumes_requested_lengths_and_matches_middle_cpu_bits() {
        use vector_ta::indicators::fibonacci_entry_bands::{
            FibonacciEntryBandsInput, FibonacciEntryBandsParams, fibonacci_entry_bands_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        for index in [3, 311] {
            ohlcv.open[index] = f64::NAN;
            ohlcv.high[index] = f64::NAN;
            ohlcv.low[index] = f64::NAN;
            ohlcv.close[index] = f64::NAN;
        }
        let lengths = [21, 7, 50, 100, 200];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Fibonacci Entry Bands primary test requires CUDA 0");
        let output = engine
            .compute_primary_device("fibonacci_entry_bands", &lengths)
            .expect("the preserved primary ABI must consume every requested length");
        assert_eq!(output.output_id, "middle");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident Fibonacci Entry Bands primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Fibonacci Entry Bands materialized HostF64")
            }
        };
        let mut expected = Vec::with_capacity(actual.len());
        for length in lengths {
            let output = fibonacci_entry_bands_with_kernel(
                &FibonacciEntryBandsInput::from_slices(
                    &ohlcv.open,
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    FibonacciEntryBandsParams {
                        source: Some("hlc3".into()),
                        length: Some(length),
                        atr_length: Some(14),
                        use_atr: Some(true),
                        tp_aggressiveness: Some("low".into()),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("scalar Fibonacci Entry Bands must accept every requested length");
            expected.extend(output.basis);
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "primary middle[{index}] lost CPU NaN");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary middle[{index}] is not exact scalar CPU f64 parity"
                );
            }
        }
    }

    #[test]
    fn fibonacci_trailing_stop_outputs_stay_resident_for_exact_default_tuple() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::FibonacciTrailingStop.is_sequential(),
            "one thread must own the complete pivot/ratchet state"
        );
        assert!(
            F64Kernel::FibonacciTrailingStop.is_period_invariant(),
            "the preserved primary ABI remains default-only and ignores generic periods"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [(20, 1, -0.382, 0)];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Fibonacci Trailing Stop resident test requires CUDA 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_fibonacci_trailing_stop_outputs_device(&parameter_rows)
            .expect("all four Fibonacci Trailing Stop outputs must stay resident");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Fibonacci Trailing Stop re-uploaded resident prices"
        );
        assert_eq!(result.indicator_id, "fibonacci_trailing_stop");
        assert_eq!(result.entry_point, "fibonacci_trailing_stop_batch_f64");
        assert_eq!((result.rows, result.cols), (1, ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            ["trailing_stop", "long_stop", "short_stop", "direction"]
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == 1
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident Fibonacci Trailing Stop matrices must retire");
    }

    #[test]
    fn fibonacci_trailing_stop_outputs_match_exact_scalar_cpu_bits() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::fibonacci_trailing_stop::{
            FibonacciTrailingStopInput, FibonacciTrailingStopParams,
            fibonacci_trailing_stop_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        for index in [3, 311] {
            ohlcv.high[index] = f64::NAN;
            ohlcv.low[index] = f64::NAN;
            ohlcv.close[index] = f64::NAN;
        }
        let parameter_rows = [(20, 1, -0.382, 0)];
        let expected = fibonacci_trailing_stop_with_kernel(
            &FibonacciTrailingStopInput::from_slices(
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                FibonacciTrailingStopParams {
                    left_bars: Some(20),
                    right_bars: Some(1),
                    level: Some(-0.382),
                    trigger: Some("close".to_string()),
                },
            ),
            Kernel::Scalar,
        )
        .expect("the scalar authority must accept the canonical default tuple");
        let expected = BTreeMap::from([
            ("trailing_stop", expected.trailing_stop),
            ("long_stop", expected.long_stop),
            ("short_stop", expected.short_stop),
            ("direction", expected.direction),
        ]);

        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Fibonacci Trailing Stop parity test requires CUDA 0");
        let result = engine
            .compute_fibonacci_trailing_stop_outputs_device(&parameter_rows)
            .expect("the resident route must accept the canonical default tuple");
        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .unwrap_or_else(|error| panic!("{} download failed: {error}", output.output_id));
            let expected = expected
                .get(output.output_id)
                .expect("every returned output needs one scalar oracle");
            assert_eq!(actual.len(), expected.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(expected).enumerate() {
                if cpu.is_nan() {
                    assert!(gpu.is_nan(), "{}[{index}] lost CPU NaN", output.output_id);
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn fibonacci_trailing_stop_primary_matches_exact_default_cpu_bits() {
        use vector_ta::indicators::fibonacci_trailing_stop::{
            FibonacciTrailingStopInput, FibonacciTrailingStopParams,
            fibonacci_trailing_stop_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        for index in [3, 311] {
            ohlcv.high[index] = f64::NAN;
            ohlcv.low[index] = f64::NAN;
            ohlcv.close[index] = f64::NAN;
        }
        let periods = [1];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the Fibonacci Trailing Stop primary test requires CUDA 0");
        let output = engine
            .compute_primary_device("fibonacci_trailing_stop", &periods)
            .expect("the preserved primary ABI must remain launchable");
        assert_eq!(output.output_id, "trailing_stop");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident Fibonacci Trailing Stop primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly Fibonacci Trailing Stop materialized HostF64")
            }
        };
        let expected_row = fibonacci_trailing_stop_with_kernel(
            &FibonacciTrailingStopInput::from_slices(
                &ohlcv.high,
                &ohlcv.low,
                &ohlcv.close,
                FibonacciTrailingStopParams::default(),
            ),
            Kernel::Scalar,
        )
        .expect("the scalar primary authority must accept defaults")
        .trailing_stop;
        let expected = expected_row;
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "primary trailing_stop[{index}] lost CPU NaN");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary trailing_stop[{index}] is not exact scalar CPU f64 parity"
                );
            }
        }
    }

    #[test]
    fn fisher_outputs_stay_resident_for_default_and_all_period_sweeps() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::Fisher.is_sequential(),
            "one thread must own each Fisher recurrence"
        );
        assert!(
            !F64Kernel::Fisher.is_period_invariant(),
            "Fisher must consume every admitted period"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let periods = [9, 7, 21, 50, 100, 200];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the Fisher resident test requires CUDA 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_fisher_outputs_device(&periods)
            .expect("both Fisher outputs must stay resident for every admitted period");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "Fisher re-uploaded its resident high/low input"
        );
        assert_eq!(result.indicator_id, "fisher");
        assert_eq!(result.entry_point, "fisher_outputs_f64");
        assert_eq!((result.rows, result.cols), (periods.len(), ohlcv.len()));
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            FISHER_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == periods.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("both resident Fisher matrices must retire");
    }

    #[test]
    fn fisher_outputs_match_cpu_bits_for_default_and_all_period_sweeps() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::fisher::{FisherInput, FisherParams, fisher_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        for index in [131, 709] {
            ohlcv.high[index] = f64::NAN;
            ohlcv.low[index] = f64::NAN;
        }
        let periods = [9, 7, 21, 50, 100, 200];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the Fisher parity test requires CUDA 0");
        let result = engine
            .compute_fisher_outputs_device(&periods)
            .expect("the resident Fisher route must accept the default and all five sweeps");

        let mut expected: BTreeMap<&str, Vec<f64>> = FISHER_OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for period in periods {
            let output = fisher_with_kernel(
                &FisherInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    FisherParams {
                        period: Some(period),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar Fisher authority must accept the admitted period");
            expected.get_mut("fisher").unwrap().extend(output.fisher);
            expected.get_mut("signal").unwrap().extend(output.signal);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident Fisher output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected Fisher output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn fisher_primary_matches_cpu_bits_for_requested_periods() {
        use vector_ta::indicators::fisher::{FisherInput, FisherParams, fisher_with_kernel};
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        let periods = [9, 7, 21];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the Fisher primary test requires CUDA 0");
        let output = engine
            .compute_primary_device("fisher", &periods)
            .expect("the preserved Fisher primary ABI must consume every requested period");
        assert_eq!(output.output_id, "fisher");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident Fisher primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => panic!("GpuOnly Fisher materialized HostF64"),
        };
        let mut expected = Vec::with_capacity(actual.len());
        for period in periods {
            expected.extend(
                fisher_with_kernel(
                    &FisherInput::from_slices(
                        &ohlcv.high,
                        &ohlcv.low,
                        FisherParams {
                            period: Some(period),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar Fisher primary must accept the requested period")
                .fisher,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "primary Fisher[{index}] lost CPU NaN");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary Fisher[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn forward_backward_exponential_oscillator_outputs_stay_resident_for_all_admitted_lengths() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        assert!(
            F64Kernel::ForwardBackwardExponentialOscillator.is_sequential(),
            "one thread must own each FBEO recurrence"
        );
        assert!(
            !F64Kernel::ForwardBackwardExponentialOscillator.is_period_invariant(),
            "FBEO must consume every admitted length"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [(20, 10), (7, 10), (21, 10), (50, 10), (100, 10), (200, 10)];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the FBEO resident test requires CUDA 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_forward_backward_exponential_oscillator_outputs_device(&parameter_rows)
            .expect("all three FBEO outputs must stay resident for every admitted tuple");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "FBEO re-uploaded its resident close input"
        );
        assert_eq!(
            result.indicator_id,
            "forward_backward_exponential_oscillator"
        );
        assert_eq!(
            result.entry_point,
            "forward_backward_exponential_oscillator_batch_f64"
        );
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident FBEO matrices must retire");
    }

    #[test]
    fn forward_backward_exponential_oscillator_outputs_match_cpu_bits_for_all_admitted_lengths() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::forward_backward_exponential_oscillator::{
            ForwardBackwardExponentialOscillatorInput, ForwardBackwardExponentialOscillatorParams,
            forward_backward_exponential_oscillator_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        for index in [311, 937] {
            ohlcv.close[index] = f64::NAN;
        }
        let parameter_rows = [(20, 10), (7, 10), (21, 10), (50, 10), (100, 10), (200, 10)];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the FBEO parity test requires CUDA 0");
        let result = engine
            .compute_forward_backward_exponential_oscillator_outputs_device(&parameter_rows)
            .expect("the resident FBEO route must accept every admitted tuple");

        let mut expected: BTreeMap<&str, Vec<f64>> =
            FORWARD_BACKWARD_EXPONENTIAL_OSCILLATOR_OUTPUT_IDS
                .into_iter()
                .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
                .collect();
        for (length, smooth) in parameter_rows {
            let output = forward_backward_exponential_oscillator_with_kernel(
                &ForwardBackwardExponentialOscillatorInput::from_slice(
                    &ohlcv.close,
                    ForwardBackwardExponentialOscillatorParams {
                        length: Some(length),
                        smooth: Some(smooth),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar FBEO authority must accept the admitted tuple");
            expected
                .get_mut("forward_backward")
                .unwrap()
                .extend(output.forward_backward);
            expected
                .get_mut("backward")
                .unwrap()
                .extend(output.backward);
            expected
                .get_mut("histogram")
                .unwrap()
                .extend(output.histogram);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident FBEO output must download for parity");
            let cpu = expected
                .get(output.output_id)
                .unwrap_or_else(|| panic!("unexpected FBEO output {}", output.output_id));
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn forward_backward_exponential_oscillator_primary_matches_cpu_bits_for_requested_lengths() {
        use vector_ta::indicators::forward_backward_exponential_oscillator::{
            ForwardBackwardExponentialOscillatorInput, ForwardBackwardExponentialOscillatorParams,
            forward_backward_exponential_oscillator_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let lengths = [20, 7, 21];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the FBEO primary test requires CUDA 0");
        let output = engine
            .compute_primary_device("forward_backward_exponential_oscillator", &lengths)
            .expect("the preserved FBEO primary ABI must consume every requested length");
        assert_eq!(output.output_id, "forward_backward");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident FBEO primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => panic!("GpuOnly FBEO materialized HostF64"),
        };
        let mut expected = Vec::with_capacity(actual.len());
        for length in lengths {
            expected.extend(
                forward_backward_exponential_oscillator_with_kernel(
                    &ForwardBackwardExponentialOscillatorInput::from_slice(
                        &ohlcv.close,
                        ForwardBackwardExponentialOscillatorParams {
                            length: Some(length),
                            smooth: Some(10),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar FBEO primary must accept the requested length")
                .forward_backward,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "primary FBEO[{index}] lost CPU NaN");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary FBEO[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn fvg_trailing_stop_outputs_stay_resident_for_all_admitted_smoothing_lengths() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        const OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_ts", "lower_ts"];
        assert!(
            F64Kernel::FvgTrailingStop.is_sequential(),
            "one thread must own each FVG Trailing Stop state machine"
        );
        assert!(
            !F64Kernel::FvgTrailingStop.is_period_invariant(),
            "FVG Trailing Stop must consume every admitted smoothing length"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (5, 9, false),
            (5, 7, false),
            (5, 21, false),
            (5, 50, false),
            (5, 100, false),
            (5, 200, false),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the FVG Trailing Stop resident test requires CUDA 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_fvg_trailing_stop_outputs_device(&parameter_rows)
            .expect("all four FVG Trailing Stop outputs must stay resident for every tuple");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "FVG Trailing Stop re-uploaded its resident HLC input"
        );
        assert_eq!(result.indicator_id, "fvg_trailing_stop");
        assert_eq!(result.entry_point, "fvg_trailing_stop_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident FVG Trailing Stop matrices must retire");
    }

    #[test]
    fn fvg_trailing_stop_outputs_match_cpu_bits_for_all_admitted_smoothing_lengths() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::fvg_trailing_stop::{
            FvgTrailingStopInput, FvgTrailingStopParams, fvg_trailing_stop_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        const OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_ts", "lower_ts"];
        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        ohlcv.close[..3].fill(f64::NAN);
        for index in [311, 937] {
            ohlcv.high[index] = f64::NAN;
            ohlcv.low[index] = f64::NAN;
            ohlcv.close[index] = f64::NAN;
        }
        let parameter_rows = [
            (5, 9, false),
            (5, 7, false),
            (5, 21, false),
            (5, 50, false),
            (5, 100, false),
            (5, 200, false),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the FVG Trailing Stop parity test requires CUDA 0");
        let result = engine
            .compute_fvg_trailing_stop_outputs_device(&parameter_rows)
            .expect("the resident FVG Trailing Stop route must accept every admitted tuple");

        let mut expected: BTreeMap<&str, Vec<f64>> = OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (lookback, smoothing_length, reset_on_cross) in parameter_rows {
            let output = fvg_trailing_stop_with_kernel(
                &FvgTrailingStopInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    FvgTrailingStopParams {
                        unmitigated_fvg_lookback: Some(lookback),
                        smoothing_length: Some(smoothing_length),
                        reset_on_cross: Some(reset_on_cross),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("the scalar FVG Trailing Stop authority must accept the admitted tuple");
            expected.get_mut("upper").unwrap().extend(output.upper);
            expected.get_mut("lower").unwrap().extend(output.lower);
            expected
                .get_mut("upper_ts")
                .unwrap()
                .extend(output.upper_ts);
            expected
                .get_mut("lower_ts")
                .unwrap()
                .extend(output.lower_ts);
        }

        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("each resident FVG Trailing Stop output must download for parity");
            let cpu = expected.get(output.output_id).unwrap_or_else(|| {
                panic!("unexpected FVG Trailing Stop output {}", output.output_id)
            });
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(
                        gpu.is_nan(),
                        "{}[{index}] lost CPU undefined behavior: gpu={gpu:?}",
                        output.output_id
                    );
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn fvg_trailing_stop_primary_matches_cpu_bits_for_requested_smoothing_lengths() {
        use vector_ta::indicators::fvg_trailing_stop::{
            FvgTrailingStopInput, FvgTrailingStopParams, fvg_trailing_stop_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        ohlcv.close[..3].fill(f64::NAN);
        let smoothing_lengths = [9, 7, 21];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the FVG Trailing Stop primary test requires CUDA 0");
        let output = engine
            .compute_primary_device("fvg_trailing_stop", &smoothing_lengths)
            .expect("the preserved primary ABI must consume every requested smoothing length");
        assert_eq!(output.output_id, "upper");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident FVG Trailing Stop primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => {
                panic!("GpuOnly FVG Trailing Stop materialized HostF64")
            }
        };
        let mut expected = Vec::with_capacity(actual.len());
        for smoothing_length in smoothing_lengths {
            expected.extend(
                fvg_trailing_stop_with_kernel(
                    &FvgTrailingStopInput::from_slices(
                        &ohlcv.high,
                        &ohlcv.low,
                        &ohlcv.close,
                        FvgTrailingStopParams {
                            unmitigated_fvg_lookback: Some(5),
                            smoothing_length: Some(smoothing_length),
                            reset_on_cross: Some(false),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("the scalar primary authority must accept the requested smoothing length")
                .upper,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(
                    gpu.is_nan(),
                    "primary FVG Trailing Stop[{index}] lost CPU NaN"
                );
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary FVG Trailing Stop[{index}] is not exact scalar CPU f64 parity: gpu={gpu:?} cpu={cpu:?}"
                );
            }
        }
    }

    #[test]
    fn gatorosc_outputs_stay_resident_for_all_admitted_length_ratios() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        const OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_change", "lower_change"];
        assert!(F64Kernel::Gatorosc.is_sequential());
        assert!(
            !F64Kernel::Gatorosc.is_period_invariant(),
            "Gator Oscillator must consume every admitted length ratio"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (13, 8, 8, 5, 5, 3),
            (7, 8, 4, 5, 3, 3),
            (21, 8, 13, 5, 8, 3),
            (50, 8, 31, 5, 19, 3),
            (100, 8, 62, 5, 38, 3),
            (200, 8, 123, 5, 77, 3),
        ];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the Gator resident test requires CUDA 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_gatorosc_outputs_device(&parameter_rows)
            .expect("all four Gator outputs must stay resident for every admitted tuple");
        assert_eq!(engine.uploads(), uploads_before, "Gator re-uploaded close");
        assert_eq!(result.indicator_id, "gatorosc");
        assert_eq!(result.entry_point, "gatorosc_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident Gator matrices must retire");
    }

    #[test]
    fn gatorosc_outputs_match_cpu_bits_for_all_admitted_length_ratios() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::gatorosc::{
            GatorOscInput, GatorOscParams, gatorosc_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        const OUTPUT_IDS: [&str; 4] = ["upper", "lower", "upper_change", "lower_change"];
        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        for index in [311, 937] {
            ohlcv.close[index] = f64::NAN;
        }
        let parameter_rows = [
            (13, 8, 8, 5, 5, 3),
            (7, 8, 4, 5, 3, 3),
            (21, 8, 13, 5, 8, 3),
            (50, 8, 31, 5, 19, 3),
            (100, 8, 62, 5, 38, 3),
            (200, 8, 123, 5, 77, 3),
        ];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the Gator parity test requires CUDA 0");
        let result = engine
            .compute_gatorosc_outputs_device(&parameter_rows)
            .expect("the resident Gator route must accept every admitted tuple");
        let mut expected: BTreeMap<&str, Vec<f64>> = OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (jaws_length, jaws_shift, teeth_length, teeth_shift, lips_length, lips_shift) in
            parameter_rows
        {
            let output = gatorosc_with_kernel(
                &GatorOscInput::from_slice(
                    &ohlcv.close,
                    GatorOscParams {
                        jaws_length: Some(jaws_length),
                        jaws_shift: Some(jaws_shift),
                        teeth_length: Some(teeth_length),
                        teeth_shift: Some(teeth_shift),
                        lips_length: Some(lips_length),
                        lips_shift: Some(lips_shift),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("scalar Gator authority must accept the admitted tuple");
            expected.get_mut("upper").unwrap().extend(output.upper);
            expected.get_mut("lower").unwrap().extend(output.lower);
            expected
                .get_mut("upper_change")
                .unwrap()
                .extend(output.upper_change);
            expected
                .get_mut("lower_change")
                .unwrap()
                .extend(output.lower_change);
        }
        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("resident Gator output must download for parity");
            let cpu = expected.get(output.output_id).unwrap();
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(gpu.is_nan(), "{}[{index}] lost CPU NaN", output.output_id);
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn gatorosc_primary_matches_cpu_bits_for_requested_length_anchors() {
        use vector_ta::indicators::gatorosc::{
            GatorOscInput, GatorOscParams, gatorosc_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.close[..3].fill(f64::NAN);
        let anchors = [13, 7, 21];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the Gator primary test requires CUDA 0");
        let output = engine
            .compute_primary_device("gatorosc", &anchors)
            .expect("the preserved Gator primary ABI must consume every anchor");
        assert_eq!(output.output_id, "upper");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident Gator primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => panic!("GpuOnly Gator materialized HostF64"),
        };
        let mut expected = Vec::with_capacity(actual.len());
        for anchor in anchors {
            let teeth_length = (8 * anchor + 6) / 13;
            let lips_length = (5 * anchor + 6) / 13;
            expected.extend(
                gatorosc_with_kernel(
                    &GatorOscInput::from_slice(
                        &ohlcv.close,
                        GatorOscParams {
                            jaws_length: Some(anchor),
                            jaws_shift: Some(8),
                            teeth_length: Some(teeth_length),
                            teeth_shift: Some(5),
                            lips_length: Some(lips_length),
                            lips_shift: Some(3),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("scalar Gator primary must accept the requested anchor")
                .upper,
            );
        }
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "primary Gator[{index}] lost CPU NaN");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary Gator[{index}] drifted"
                );
            }
        }
    }

    #[test]
    fn halftrend_outputs_stay_resident_for_all_admitted_atr_periods() {
        use vector_ta::cuda::neoethos_f64_wrapper::F64Kernel;

        const OUTPUT_IDS: [&str; 6] = [
            "halftrend",
            "trend",
            "atr_high",
            "atr_low",
            "buy_signal",
            "sell_signal",
        ];
        assert!(F64Kernel::Halftrend.is_sequential());
        assert!(
            !F64Kernel::Halftrend.is_period_invariant(),
            "HalfTrend must consume every admitted ATR period"
        );
        let ohlcv = repeated_ctrader_bandpass_fixture();
        let parameter_rows = [
            (2, 2.0, 100),
            (2, 2.0, 7),
            (2, 2.0, 21),
            (2, 2.0, 50),
            (2, 2.0, 200),
        ];
        let engine = GpuIndicatorEngine::new(&ohlcv, 0)
            .expect("the HalfTrend resident test requires CUDA 0");
        let uploads_before = engine.uploads();
        let result = engine
            .compute_halftrend_outputs_device(&parameter_rows)
            .expect("all six HalfTrend outputs must stay resident for every admitted tuple");
        assert_eq!(
            engine.uploads(),
            uploads_before,
            "HalfTrend re-uploaded OHLC"
        );
        assert_eq!(result.indicator_id, "halftrend");
        assert_eq!(result.entry_point, "halftrend_outputs_f64");
        assert_eq!(
            (result.rows, result.cols),
            (parameter_rows.len(), ohlcv.len())
        );
        assert_eq!(
            result
                .outputs
                .iter()
                .map(|output| output.output_id)
                .collect::<Vec<_>>(),
            OUTPUT_IDS
        );
        assert!(result.outputs.iter().all(|output| {
            output.matrix.rows() == parameter_rows.len()
                && output.matrix.cols() == ohlcv.len()
                && output.matrix.device_id() == engine.device_ordinal()
        }));
        engine
            .synchronize()
            .expect("all resident HalfTrend matrices must retire");
    }

    #[test]
    fn halftrend_outputs_match_cpu_bits_for_default_and_all_admitted_atr_periods() {
        use std::collections::BTreeMap;
        use vector_ta::indicators::halftrend::{
            HalfTrendInput, HalfTrendParams, halftrend_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        const OUTPUT_IDS: [&str; 6] = [
            "halftrend",
            "trend",
            "atr_high",
            "atr_low",
            "buy_signal",
            "sell_signal",
        ];
        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        ohlcv.close[..3].fill(f64::NAN);
        let parameter_rows = [
            (2, 2.0, 100),
            (2, 2.0, 7),
            (2, 2.0, 21),
            (2, 2.0, 50),
            (2, 2.0, 200),
        ];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the HalfTrend parity test requires CUDA 0");
        let result = engine
            .compute_halftrend_outputs_device(&parameter_rows)
            .expect("the resident HalfTrend route must accept every admitted tuple");
        let mut expected: BTreeMap<&str, Vec<f64>> = OUTPUT_IDS
            .into_iter()
            .map(|output_id| (output_id, Vec::with_capacity(result.rows * result.cols)))
            .collect();
        for (amplitude, channel_deviation, atr_period) in parameter_rows {
            let output = halftrend_with_kernel(
                &HalfTrendInput::from_slices(
                    &ohlcv.high,
                    &ohlcv.low,
                    &ohlcv.close,
                    HalfTrendParams {
                        amplitude: Some(amplitude),
                        channel_deviation: Some(channel_deviation),
                        atr_period: Some(atr_period),
                    },
                ),
                Kernel::Scalar,
            )
            .expect("scalar HalfTrend authority must accept the admitted tuple");
            expected
                .get_mut("halftrend")
                .unwrap()
                .extend(output.halftrend);
            expected.get_mut("trend").unwrap().extend(output.trend);
            expected
                .get_mut("atr_high")
                .unwrap()
                .extend(output.atr_high);
            expected.get_mut("atr_low").unwrap().extend(output.atr_low);
            expected
                .get_mut("buy_signal")
                .unwrap()
                .extend(output.buy_signal);
            expected
                .get_mut("sell_signal")
                .unwrap()
                .extend(output.sell_signal);
        }
        for output in &result.outputs {
            let actual = engine
                .runtime
                .download_matrix_f64(&output.matrix)
                .expect("resident HalfTrend output must download for parity");
            let cpu = expected.get(output.output_id).unwrap();
            assert_eq!(actual.len(), cpu.len());
            for (index, (&gpu, &cpu)) in actual.iter().zip(cpu).enumerate() {
                if cpu.is_nan() {
                    assert!(gpu.is_nan(), "{}[{index}] lost CPU NaN", output.output_id);
                } else {
                    assert_eq!(
                        gpu.to_bits(),
                        cpu.to_bits(),
                        "{}[{index}] is not exact scalar CPU f64 parity",
                        output.output_id
                    );
                }
            }
        }
    }

    #[test]
    fn halftrend_primary_matches_cpu_bits_for_requested_atr_periods() {
        use vector_ta::indicators::halftrend::{
            HalfTrendInput, HalfTrendParams, halftrend_with_kernel,
        };
        use vector_ta::utilities::enums::Kernel;

        let mut ohlcv = repeated_ctrader_bandpass_fixture();
        ohlcv.high[..3].fill(f64::NAN);
        ohlcv.low[..3].fill(f64::NAN);
        ohlcv.close[..3].fill(f64::NAN);
        let atr_periods = [100, 7, 21];
        let engine =
            GpuIndicatorEngine::new(&ohlcv, 0).expect("the HalfTrend primary test requires CUDA 0");
        let output = engine
            .compute_primary_device("halftrend", &atr_periods)
            .expect("the preserved HalfTrend primary ABI must consume every ATR period");
        assert_eq!(output.output_id, "halftrend");
        let actual = match &output.series {
            IndicatorCudaSeriesF64::DeviceF64(matrix) => engine
                .runtime
                .download_matrix_f64(matrix)
                .expect("resident HalfTrend primary must download"),
            IndicatorCudaSeriesF64::HostF64(_) => panic!("GpuOnly HalfTrend materialized HostF64"),
        };
        let mut expected = Vec::with_capacity(actual.len());
        for atr_period in atr_periods {
            expected.extend(
                halftrend_with_kernel(
                    &HalfTrendInput::from_slices(
                        &ohlcv.high,
                        &ohlcv.low,
                        &ohlcv.close,
                        HalfTrendParams {
                            amplitude: Some(2),
                            channel_deviation: Some(2.0),
                            atr_period: Some(atr_period),
                        },
                    ),
                    Kernel::Scalar,
                )
                .expect("scalar HalfTrend primary must accept the requested ATR period")
                .halftrend,
            );
        }
        assert_eq!(actual.len(), expected.len());
        for (index, (&gpu, &cpu)) in actual.iter().zip(&expected).enumerate() {
            if cpu.is_nan() {
                assert!(gpu.is_nan(), "primary HalfTrend[{index}] lost CPU NaN");
            } else {
                assert_eq!(
                    gpu.to_bits(),
                    cpu.to_bits(),
                    "primary HalfTrend[{index}] drifted"
                );
            }
        }
    }

    #[test]
    fn evwma_finite_pair_admission_follows_the_declared_price_source() {
        let close = [1.0, 2.0];
        let hlcc4 = [f64::INFINITY, 2.0];
        let volume = [10.0, 11.0];

        assert_eq!(
            first_valid_price_volume_finite_for_input(
                DeviceInput::CloseVolume,
                &close,
                &hlcc4,
                &volume,
            ),
            Some(0),
            "close-volume admission must not be delayed by an unrelated HLCC4 infinity"
        );
        assert_eq!(
            first_valid_price_volume_finite_for_input(
                DeviceInput::Hlcc4CloseVolume,
                &close,
                &hlcc4,
                &volume,
            ),
            Some(1),
            "the retained HLCC4-volume shape must still scan its own price series"
        );
    }
}
