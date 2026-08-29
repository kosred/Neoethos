//! vector-ta CUDA indicator lane — device-resident, **f64**, multi-period.
//!
//! Everything in this file is `#[cfg(feature = "gpu-cuda")]`-gated at the
//! module declaration in `core/mod.rs`, so a card-less build never compiles a
//! byte of it.
//!
//! # What this replaces
//!
//! `hpc_ta::compute_classic_ta_columns` runs an 18-indicator × 5-period sweep
//! on the CPU: 90 independent full-series scans, each re-reading the same OHLCV
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
//! 1. **The arch trap — now closed at the build.** vector-ta emits ONE
//!    multi-architecture fatbin per kernel carrying real SASS for sm_80,
//!    sm_86, sm_89 and sm_90 plus embedded PTX at the highest of them, so the
//!    same binary runs on an A100, a 3090, a 4090 and an H100 with no source
//!    change and no rebuild flag change, and a newer card JITs the embedded
//!    PTX forward. [`GpuIndicatorEngine::new`] still proves it at run time by
//!    loading a module and launching a kernel, because a build-time string is
//!    not evidence.
//! 2. **No silent fallback.** Every failure here is an `Err` carrying the
//!    device error. Nothing in this module computes a CPU value. The caller
//!    (`hpc_ta::compute_classic_ta_columns_with_policy`) decides, and whatever
//!    it decides is recorded by name in `indicator_telemetry`.
//! 3. **No silent precision drop.** An indicator with no f64 kernel produces
//!    `IndicatorDispatchError::CudaF64KernelMissing` naming it. It is never
//!    served by the f32 kernel.
//!
//! # Why the table below is short, and why it is a table at all
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
//! The rest of `MULTI_PERIOD_IDS` stays on the CPU and is reported as
//! [`IndicatorLane::CpuIndicatorNotPortable`] — enumerated up front from this
//! table, never discovered by a failed launch mid-run. There is now exactly ONE
//! reason an id is in that group:
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
//! So `tsi`, `obv` and now `vwap` have all left the CPU group, and EVERY
//! single-output id in `MULTI_PERIOD_IDS` is on the device — the whole
//! reachable sweep, with only the multi-output five left on the CPU emitting
//! nothing on either lane.
//!
//! No count is written here, and none should be added. That relationship is
//! ASSERTED instead, by
//! `tests::every_reachable_multi_period_id_with_an_f64_kernel_is_claimed`,
//! which fails the day vector-ta registers a kernel this table has not picked
//! up — which is exactly how `vwap` was allowed to sit written, compiled,
//! registered and still computed on the CPU. Counts rot; assertions do not.

use super::super::Ohlcv;
use crate::core::indicator_telemetry::{IndicatorLane, VECTOR_TA_ARCHS, VECTOR_TA_PTX_ARCH};
use anyhow::{Context, Result, bail};

use vector_ta::cuda::{
    CudaDeviceCloseVolumeF64Ref, CudaDeviceHighLowF64Ref, CudaDeviceOhlcvF64, CudaDeviceVectorF64,
    CudaDeviceVectorI64, CudaF64Indicators, CudaRuntime, cuda_available,
};
use vector_ta::indicators::dispatch::{
    CudaOutputTargetF64, F64FirstValidRule, F64InputKind, IndicatorCudaDeviceDataRefF64,
    IndicatorCudaDeviceRequestF64, IndicatorCudaSeriesF64, compute_cuda_device_f64, f64_kernel_for,
};
use vector_ta::indicators::registry::get_indicator;

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
    /// Kernel reads (hlcc4, volume) -- `elastic_volume_weighted_moving_average`
    /// alone. The SAME device shape as [`Self::Hlc3CloseVolume`] and
    /// [`Self::CloseVolume`] with a THIRD price series in it; the pair is built
    /// here from the resident hlcc4 upload, so the distinction is made once, at
    /// the upload, and a kernel expecting hlcc4 can never be handed close.
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
    // The three that complete the sweep. `hpc_ta::MULTI_PERIOD_IDS` lists 18
    // ids; five of them (stoch, macd, bollinger_bands, keltner, supertrend) are
    // multi-output, and `compute_cpu` with `output_id: None` returns
    // `Err(InvalidParam)` for those, which `hpc_ta.rs:291` swallows — so they
    // emit ZERO columns on either lane and cannot be swept. Every id that is
    // left is here, so a frame's period sweep no longer leaves the device at
    // any point. `every_reachable_multi_period_id_with_an_f64_kernel_is_claimed`
    // is what holds that, not this comment.
    GpuSweepSpec {
        id: "tsi",
        input: DeviceInput::CloseFromOhlcv,
    },
    GpuSweepSpec {
        id: "obv",
        input: DeviceInput::CloseVolume,
    },
    // `vwap` — CLAIMED 2026-08-10, and nothing but this row changed to claim
    // it. The engine has carried the whole shape since the `TimestampCloseVolume`
    // arm landed: the bar timestamps are uploaded once as i64 (`new`, the
    // `upload_i64` call), `data_ref` builds the three-pointer ref from them,
    // and `first_valid_for` answers `F64FirstValidRule::Ignored` with 0 because
    // `vwap_with_kernel` calls `alloc_with_nan_prefix(n, 0)` and has no warmup.
    // The kernel `neoethos_vwap_batch_f64` was written against `vwap_scalar`
    // and compiled all along; the row was the only thing missing, so the work
    // sat unreachable.
    //
    // What made it unclaimable was a SECOND CPU implementation, not a
    // precision question: `vwap_row_scalar_pv` consumed a precomputed
    // `pv[i] = price * volume` (two roundings) where `vwap_scalar` writes
    // `vol_price_sum = p.mul_add(v, vol_price_sum)` (one), and only the
    // `Kernel::Scalar` arm of `vwap_batch_inner` reached it. vector-ta deleted
    // it — both batch arms now call `vwap_row_scalar`, so `Kernel::Auto` (what
    // `hpc_ta` runs in production) and `Kernel::Scalar` (what the kernel was
    // written against, and what the parity oracle pins) are the same numbers.
    // `WITHHELD_PENDING_CPU_SELF_CONSISTENCY` is now `&[]`, and
    // `f64_lane_cpu_reference::scalar_and_auto_agree_for_every_claimed_indicator`
    // measures the agreement on clean and on gapped bars without a card.
    GpuSweepSpec {
        id: "vwap",
        input: DeviceInput::TimestampCloseVolume,
    },
];

/// Is this indicator id served by the device lane?
pub fn spec_for(id: &str) -> Option<&'static GpuSweepSpec> {
    GPU_SWEEP_SPECS.iter().find(|s| s.id == id)
}

/// Reassemble a device period sweep into the canonical, frame-independent
/// column schema.
///
/// The device computes only periods that clear the warmup preflight. Periods
/// that cannot exist on this frame are still represented by full-length NaN
/// columns, exactly like the scalar lane. Dropping them would make the feature
/// schema depend on frame length and would mix incompatible CPU/GPU frames.
fn assemble_sweep_columns(
    id: &str,
    periods: &[usize],
    n: usize,
    computed_rows: Vec<Vec<f64>>,
) -> Result<Vec<(String, Vec<f64>)>> {
    let expected_computed = periods
        .iter()
        .filter(|&&period| (period as f64) * 1.25 < n as f64)
        .count();
    if computed_rows.len() != expected_computed {
        bail!(
            "{id}: device returned {} computed period rows, expected {expected_computed}",
            computed_rows.len()
        );
    }

    let mut computed = computed_rows.into_iter();
    let mut columns = Vec::with_capacity(periods.len());
    for &period in periods {
        let values = if (period as f64) * 1.25 < n as f64 {
            let values = computed
                .next()
                .with_context(|| format!("{id}_{period}: missing computed device row"))?;
            if values.len() != n {
                bail!(
                    "{id}_{period}: device row has {} values, expected {n}",
                    values.len()
                );
            }
            values
        } else {
            vec![f64::NAN; n]
        };
        columns.push((format!("{id}_{period}"), values));
    }
    debug_assert!(computed.next().is_none());
    Ok(columns)
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

/// A CUDA indicator lane bound to one frame's OHLCV, with the series resident
/// on the device in f64 for the whole life of the engine.
///
/// Construction is where the lane is PROVEN. `new` does not merely check that
/// a device exists — it uploads, then loads the f64 module and launches a real
/// kernel. That is the only check that actually covers the arch trap, because
/// `vector_ta::cuda::cuda_available()` probes with its own `.target sm_52` PTX
/// (mod.rs:1199) and therefore returns `true` on a 3090 even when the real
/// kernels are unloadable.
pub struct GpuIndicatorEngine {
    runtime: CudaRuntime,
    /// The f64 kernel module. ONE load for the whole frame — the f32
    /// dispatcher constructs a fresh wrapper per call and therefore JITs once
    /// per indicator, roughly fifty times per frame.
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
    /// `fh.max(fl).max(fc)` — the max of three INDEPENDENT scans, which is a
    /// different index from `first_valid_hlc` whenever high, low and close
    /// start at different bars. `adx.rs:201-219`, `natr.rs:226-235`.
    first_valid_hlc_max_of_firsts: usize,
    first_valid_hlc3: usize,
    first_valid_hlc3_volume: usize,
    first_valid_close_volume: usize,
    first_valid_high_low: usize,
    // ------------------------------------------------------------ shard 1
    /// hl2 non-NaN. Not derivable from `first_valid_high_low`: hl2 is formed
    /// before the scan, so a bar where exactly one of high/low is NaN is NaN in
    /// hl2 too -- the same index in this case, but stated separately because
    /// "the same today" is not a contract.
    first_valid_hl2: usize,
    // ------------------------------------------------------------ closer 5
    /// hlcc4 non-NaN. Stated separately from `first_valid_close` for the
    /// same reason `first_valid_hl2` is: hlcc4 is formed BEFORE the scan, so
    /// a bar where any one of high/low/close is NaN is NaN in hlcc4 too.
    first_valid_hlcc4: usize,
    /// Volume non-NaN -- `vosc.rs:361` scans the volume series alone.
    first_valid_volume: usize,
    // ------------------------------------------------------ closer 6, round 3
    /// hlcc4 AND volume both `is_finite` at the same index --
    /// `elastic_volume_weighted_moving_average.rs:308-317`.
    ///
    /// Stated separately from `first_valid_hlcc4` and from a non-NaN pair scan
    /// for the reason every narrow field here exists: `is_finite` REJECTS an
    /// infinity that `!is_nan` accepts, and this index sets both the NaN prefix
    /// and the bar the EVWMA recurrence seeds `base` from.
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
    /// Number of host→device uploads performed. Should always be 6 (o,h,l,c,v
    /// + hlc3) for the lifetime of the engine — if this grows, residency broke.
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
        if let Some(v) = &ohlcv.volume {
            if v.len() != n {
                bail!(
                    "GpuIndicatorEngine: volume has {} entries for {n} bars",
                    v.len()
                );
            }
        }

        if !cuda_available() {
            bail!(
                "GpuIndicatorEngine: vector_ta::cuda::cuda_available() == false — no usable CUDA \
                 device. Set CUDA_PROBE_DEBUG=1 to see which probe stage failed. (This binary \
                 carries vector-ta SASS for {VECTOR_TA_ARCHS} plus {VECTOR_TA_PTX_ARCH} PTX.)"
            );
        }

        let device_arch = device_arch(device_ordinal)?;
        let device_name = device_name(device_ordinal);

        // NO NARROWING. The host series are f64, the device buffers are f64,
        // the kernels are f64. This is the change that makes the lane a parity
        // claim instead of a measured-divergence one.
        let volume: Vec<f64> = match &ohlcv.volume {
            Some(v) => v.clone(),
            None => vec![0.0f64; n],
        };
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
        // closer 5: hlcc4 formed the way `Candles::compute_hlcc4` does --
        // `(h + l + 2c) / 4`, one expression, one rounding -- so the device
        // source and the CPU source are the same NUMBERS and not merely the
        // same formula.
        let hlcc4: Vec<f64> = (0..n)
            .map(|i| (ohlcv.high[i] + ohlcv.low[i] + 2.0 * ohlcv.close[i]) / 4.0)
            .collect();
        let first_valid_hlcc4 =
            first_valid_1(&hlcc4).context("GpuIndicatorEngine: hlcc4 is entirely NaN")?;
        // closer 6, round 3: `is_finite` on BOTH, which is what
        // `elastic_volume_weighted_moving_average.rs:313-315` scans for.
        let first_valid_hlcc4_volume_finite = (0..n)
            .find(|&i| hlcc4[i].is_finite() && volume[i].is_finite())
            .context("GpuIndicatorEngine: no bar has both a finite hlcc4 and a finite volume")?;
        let first_valid_volume =
            first_valid_1(&volume).context("GpuIndicatorEngine: volume is entirely NaN")?;
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

        let f64_engine = CudaF64Indicators::new(device_ordinal as usize).with_context(|| {
            format!(
                "CudaF64Indicators::new({device_ordinal}) failed on {device_name} \
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
            first_valid_hlc_max_of_firsts,
            first_valid_hlc3,
            first_valid_hlc3_volume,
            first_valid_close_volume,
            first_valid_high_low,
            first_valid_hl2,
            first_valid_hlcc4,
            first_valid_hlcc4_volume_finite,
            first_valid_volume,
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
    /// This is the arch gate. `cuda_available()` cannot serve as one: it JITs
    /// its own `.target sm_52` probe PTX, so it succeeds on hardware where
    /// every shipped indicator kernel is unloadable. The only honest check is
    /// to run one.
    fn prove_module_loads(&self) -> Result<()> {
        // `sma` with period 1 over the resident close series: the cheapest
        // real kernel in the table, single row, no warmup arithmetic to argue
        // about. If this loads and launches, the fatbin matches the device.
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
                       kernels built: vector-ta fatbin SASS for {}, forward-JIT PTX at {}\n\
                     \n\
                     SASS and PTX both run FORWARD only. The vendored build emits ONE fatbin \
                     carrying sm_80 / sm_86 / sm_89 / sm_90 plus forward PTX, so any device at or \
                     above sm_80 should be served with no rebuild. If it is not, rebuild naming \
                     this device's architecture explicitly:\n\
                     \n\
                       NEOETHOS_CUDA_ARCHS={} cargo build -p neoethos-data --features \
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
                    VECTOR_TA_ARCHS,
                    VECTOR_TA_PTX_ARCH,
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
            // elastic_volume_weighted_moving_average.rs:308-317 -- price AND
            // volume both `is_finite`. NOT the `AllInputsNonNan` pair scan:
            // that one is `!is_nan` and would accept an INFINITE volume the
            // CPU skips, and the index it names seeds the recurrence.
            F64FirstValidRule::PriceVolumeFinite => self.first_valid_hlcc4_volume_finite,
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
                // closer 6, round 3. NO ROW IN `F64_KERNELS` USES THIS PAIR
                // WITH THIS RULE today -- the only `Hlcc4Volume` indicator,
                // `elastic_volume_weighted_moving_average`, is registered
                // `PriceVolumeFinite` and is answered above. The arm exists
                // because the match is exhaustive, and it reports the `is_finite`
                // index rather than a non-NaN one: an indicator reaching here
                // would be claiming the COMMON rule over a pair this engine only
                // ever scans the stricter way, and the stricter index is the
                // safe one to hand a kernel that divides by volume.
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
    /// Applies the SAME `(period as f64) * 1.25 >= n` pre-flight rule the CPU
    /// sweep applies (`hpc_ta::cpu_multi_period_columns`). Unsupported periods
    /// do not launch, but remain in the canonical schema as full-length NaN
    /// columns, so names/order/width are independent of frame length.
    pub fn sweep_columns(
        &self,
        spec: &GpuSweepSpec,
        periods: &[usize],
    ) -> Result<Vec<(String, Vec<f64>)>> {
        let kept: Vec<usize> = periods
            .iter()
            .copied()
            .filter(|&p| (p as f64) * 1.25 < self.n as f64)
            .collect();
        let rows = if kept.is_empty() {
            Vec::new()
        } else {
            self.sweep_periods(spec, &kept)?
        };
        assemble_sweep_columns(spec.id, periods, self.n, rows)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_skips_preserve_the_canonical_column_schema_without_launch_rows() {
        let periods = [7, 21, 50, 100, 200];
        let n = 100;
        let computed = vec![vec![7.0; n], vec![21.0; n], vec![50.0; n]];

        let columns = assemble_sweep_columns("sma", &periods, n, computed).unwrap();
        let names: Vec<&str> = columns.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["sma_7", "sma_21", "sma_50", "sma_100", "sma_200"]);
        assert!(columns[0].1.iter().all(|&value| value == 7.0));
        assert!(columns[1].1.iter().all(|&value| value == 21.0));
        assert!(columns[2].1.iter().all(|&value| value == 50.0));
        assert!(columns[3].1.iter().all(|value| value.is_nan()));
        assert!(columns[4].1.iter().all(|value| value.is_nan()));
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

    /// THE ANTI-ROT ASSERTION, and the reason no count appears in this file's
    /// prose any more.
    ///
    /// `GPU_SWEEP_SPECS` is the intersection of three sets maintained in three
    /// different places, and the failure mode this project keeps repeating is
    /// that one of them grows and the intersection does not — the kernel gets
    /// written, compiled, registered upstream, and then computed on the CPU
    /// anyway because nobody added the row. `vwap` sat like that: kernel
    /// written, `F64_KERNELS` row present, `WITHHELD_PENDING_CPU_SELF_CONSISTENCY`
    /// emptied upstream, and this table still excluded it on the stale
    /// justification.
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
             Add a GpuSweepSpec row. Do NOT add a justification comment instead — that is exactly \
             how `vwap` stayed on the CPU after its kernel was registered upstream."
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
}
