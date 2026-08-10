#![cfg(feature = "cuda")]

//! The f64 dispatch lane.
//!
//! # The gap this closes
//!
//! `compute_cuda` / `compute_cuda_device` — the only CUDA entry points
//! `neoethos-data` calls — are f32 from end to end: `IndicatorCudaDataRef`
//! carries `&[f32]`, `IndicatorCudaDeviceDataRef` carries
//! `CudaDeviceSliceF32Ref`, and the result is
//! `IndicatorCudaSeries::{DeviceF32, HostF32}`. So no matter how many `*_f64`
//! kernels the crate ships, a caller going through that door gets f32 back.
//!
//! This module is the f64 door. It is strictly ADDITIVE — the f32 types and
//! the 6,132-line generated f32 dispatcher are untouched, because 180 wrappers
//! still depend on them.
//!
//! # Kernel-name resolution
//!
//! [`resolve_f64_kernel`] maps an indicator id to the `*_f64` entry point that
//! serves it. Resolution is a TABLE, not a string transformation: there is no
//! code path that takes an f32 symbol and appends `_f64` to it, so an f64
//! request can never land on an f32 kernel because a name happened to exist.
//!
//! # What happens when there is no f64 kernel
//!
//! [`IndicatorDispatchError::CudaF64KernelMissing`], naming the indicator.
//! Never an f32 kernel, never the CPU. Both of those failure modes are already
//! present in this crate and both are the reason this rule is spelled out:
//!
//! * nine `*_f64` kernels are one-line empty stubs
//!   (`extern "C" __global__ void possible_rsi_batch_f64() {}`) whose wrappers
//!   call `get_function` on the stub purely so symbol resolution succeeds, then
//!   compute on the host via `Kernel::ScalarBatch` and upload the host answer
//!   so the caller cannot tell (`possible_rsi_wrapper.rs:104-152`);
//! * the f32 lane silently answers f64-shaped questions today, and was
//!   measured 54% wrong at 200k bars on the backtest.
//!
//! A loud `Err` is cheaper than either.

use super::error::IndicatorDispatchError;
use crate::cuda::device_types::CudaDeviceSliceI64Ref;
use crate::cuda::device_types_f64::{
    CudaDeviceCloseVolumeF64Ref, CudaDeviceHighLowF64Ref, CudaDeviceMatrixF64,
    CudaDeviceOhlcF64Ref, CudaDeviceOhlcvF64Ref, CudaDeviceSliceF64Ref,
};
use crate::cuda::neoethos_f64_wrapper::{CudaF64Indicators, F64Inputs, F64Kernel};

// ---------------------------------------------------------------------------
// Request / response vocabulary, mirroring the f32 side one for one
// ---------------------------------------------------------------------------

/// Host-side f64 input to a CUDA indicator. The f64 twin of
/// [`super::types::IndicatorCudaDataRef`].
#[derive(Debug, Clone, Copy)]
pub enum IndicatorCudaDataRefF64<'a> {
    Slice {
        values: &'a [f64],
    },
    Ohlc {
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        source: Option<&'a [f64]>,
    },
    Ohlcv {
        timestamp: Option<&'a [i64]>,
        open: &'a [f64],
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        volume: &'a [f64],
        source: Option<&'a [f64]>,
    },
    HighLow {
        high: &'a [f64],
        low: &'a [f64],
    },
    CloseVolume {
        close: &'a [f64],
        volume: &'a [f64],
    },
}

impl IndicatorCudaDataRefF64<'_> {
    pub fn len(&self) -> usize {
        match self {
            IndicatorCudaDataRefF64::Slice { values } => values.len(),
            IndicatorCudaDataRefF64::Ohlc { close, .. } => close.len(),
            IndicatorCudaDataRefF64::Ohlcv { close, .. } => close.len(),
            IndicatorCudaDataRefF64::HighLow { high, .. } => high.len(),
            IndicatorCudaDataRefF64::CloseVolume { close, .. } => close.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Device-resident f64 input. The f64 twin of
/// [`super::types::IndicatorCudaDeviceDataRef`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndicatorCudaDeviceDataRefF64 {
    Slice { values: CudaDeviceSliceF64Ref },
    Ohlc(CudaDeviceOhlcF64Ref),
    Ohlcv(CudaDeviceOhlcvF64Ref),
    HighLow(CudaDeviceHighLowF64Ref),
    CloseVolume(CudaDeviceCloseVolumeF64Ref),
    /// `vwap` only. The timestamps are `i64` because that is what a bar
    /// timestamp is; the anchor divides by 86_400_000 to find the session, so
    /// narrowing them would move session boundaries rather than round them.
    TimestampCloseVolume {
        timestamps: CudaDeviceSliceI64Ref,
        close: CudaDeviceSliceF64Ref,
        volume: CudaDeviceSliceF64Ref,
    },
}

impl IndicatorCudaDeviceDataRefF64 {
    pub fn len(&self) -> usize {
        match self {
            IndicatorCudaDeviceDataRefF64::Slice { values } => values.len(),
            IndicatorCudaDeviceDataRefF64::Ohlc(r) => r.len(),
            IndicatorCudaDeviceDataRefF64::Ohlcv(r) => r.len(),
            IndicatorCudaDeviceDataRefF64::HighLow(r) => r.len(),
            IndicatorCudaDeviceDataRefF64::CloseVolume(r) => r.len(),
            IndicatorCudaDeviceDataRefF64::TimestampCloseVolume { close, .. } => close.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn device_id(&self) -> u32 {
        match self {
            IndicatorCudaDeviceDataRefF64::Slice { values } => values.device_id(),
            IndicatorCudaDeviceDataRefF64::Ohlc(r) => r.device_id(),
            IndicatorCudaDeviceDataRefF64::Ohlcv(r) => r.device_id(),
            IndicatorCudaDeviceDataRefF64::HighLow(r) => r.device_id(),
            IndicatorCudaDeviceDataRefF64::CloseVolume(r) => r.device_id(),
            IndicatorCudaDeviceDataRefF64::TimestampCloseVolume { close, .. } => close.device_id(),
        }
    }
}

/// An f64 device sweep request.
///
/// `periods` is an EXPLICIT list rather than a `(start, end, step)` range: the
/// periods this codebase sweeps — `[7, 21, 50, 100, 200]` — are not an
/// arithmetic progression, and the range form's only batched shape would
/// compute 194 rows to keep 5.
#[derive(Debug, Clone, Copy)]
pub struct IndicatorCudaDeviceRequestF64<'a> {
    pub indicator_id: &'a str,
    pub data: IndicatorCudaDeviceDataRefF64,
    pub periods: &'a [i32],
    /// First index whose inputs are all non-NaN, computed by the caller with
    /// the SAME rule the CPU `*_prepare` uses. The warmup prefix and the seed
    /// window both hang off it.
    pub first_valid: usize,
    pub target: CudaOutputTargetF64,
}

/// Where an f64 result should land.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaOutputTargetF64 {
    /// Leave it on the device. This is the point of the lane — the next
    /// indicator reads the same resident upload with no round trip.
    Device,
    /// Copy back to the host. For the parity oracle and for callers that
    /// genuinely consume host values.
    Host,
}

/// An f64 result: a `rows x cols` matrix, one row per period.
#[derive(Debug)]
pub enum IndicatorCudaSeriesF64 {
    DeviceF64(CudaDeviceMatrixF64),
    HostF64(Vec<f64>),
}

/// One indicator's f64 output plus its shape.
#[derive(Debug)]
pub struct IndicatorCudaOutputF64 {
    pub indicator_id: String,
    pub series: IndicatorCudaSeriesF64,
    pub rows: usize,
    pub cols: usize,
    /// The `*_f64` entry point that produced this. Recorded so a caller can
    /// assert in telemetry WHICH kernel ran rather than inferring it.
    pub entry_point: &'static str,
}

// ---------------------------------------------------------------------------
// Kernel-name resolution
// ---------------------------------------------------------------------------

/// One row of the f64 kernel table.
#[derive(Debug, Clone, Copy)]
pub struct F64KernelSpec {
    pub indicator_id: &'static str,
    pub kernel: F64Kernel,
    /// Which host series the kernel needs so it computes the SAME indicator
    /// the CPU computes — not merely a lower-precision one.
    pub input: F64InputKind,
    /// How the CPU reference derives the index the series STARTS at. Declared
    /// per indicator rather than per input shape, because three of the
    /// high/low/close indicators do NOT use the rule the other three use.
    pub first_valid: F64FirstValidRule,
}

/// How the CPU reference computes `first_valid`.
///
/// # Why this is not a property of [`F64InputKind`]
///
/// `first_valid` is not a tolerance-sized detail. It sets BOTH the length of
/// the NaN warmup prefix AND the seed window, so getting it wrong does not
/// perturb the series by an ULP — it SHIFTS the whole series by however many
/// bars the two rules disagree by, and every value after the seed is a
/// different number.
///
/// Six indicators in this table read high/low/close, and they do not agree on
/// the rule:
///
/// * `atr.rs:197-206`, `willr.rs:300`, `wclprice.rs:176` — the first index at
///   which all three are non-NaN SIMULTANEOUSLY.
/// * `adx.rs:201-219` (`first_valid_triple_checked`) and `natr.rs:226-235` —
///   `fh.max(fl).max(fc)`, the MAX of three INDEPENDENT first-non-NaN scans.
///   That is a different index whenever the three series start at different
///   places, and it can name a bar at which one of them is still NaN.
/// * `adxr.rs:255-258` — `close.iter().position(|x| !x.is_nan())`. High and low
///   are never scanned at all.
///
/// Example, from the test fixture: `high = [1.10, NaN, 1.12]`,
/// `low = [1.09, 1.09, 1.09]`, `close = [NaN, 1.10, 1.11]` gives 2 for `atr`,
/// 1 for `adx`/`natr`, and 0 for `adxr`. One index for all six would be wrong
/// for at least three of them on any gapped symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F64FirstValidRule {
    /// The first index at which EVERY series named by [`F64InputKind`] is
    /// non-NaN. This is the common case: every single-series indicator, every
    /// (price, volume) pair, every (high, low) pair, and the three of the six
    /// Hlc indicators that scan the triple simultaneously.
    AllInputsNonNan,
    /// high/low/close, but the MAX of three INDEPENDENT first-non-NaN scans —
    /// `adx.rs:201-219`, `natr.rs:226-235`.
    HlcMaxOfIndependentFirsts,
    /// high/low/close, but the CPU scans CLOSE ALONE — `adxr.rs:255-258`.
    HlcCloseOnly,
    /// The kernel does not read the caller's value.
    ///
    /// Two kernels, for two different reasons:
    ///
    /// * `vwap` has no warmup prefix at all -- `vwap_with_kernel` calls
    ///   `alloc_with_nan_prefix(n, 0)`.
    /// * `garman_klass_volatility` DOES have a warmup, but its CPU rule
    ///   (`validity_summary`, :346) is "all four OHLC prices finite AND
    ///   strictly positive", which none of the rules here expresses. Rather
    ///   than add a variant that every consumer would have to grow a field
    ///   for, that kernel derives its own start index and declares the
    ///   caller's value unused. Declaring `AllInputsNonNan` instead would be
    ///   a claim the kernel does not honour.
    Ignored,

    // ------------------------------------------------------------------ shard 6
    //
    // Four more rules, all found by reading the CPU `*_prepare` rather than by
    // assuming. Each one shifts the whole series relative to `AllInputsNonNan`
    // on real data, which is why none of them is folded into it.

    /// The MAX of INDEPENDENT first-non-NaN scans over the series the input
    /// kind names -- the two-series twin of
    /// [`Self::HlcMaxOfIndependentFirsts`]. `donchian.rs:183-188` scans high
    /// and low separately and takes `h.max(l)`, which names a different bar
    /// from "the first index at which both are non-NaN" whenever one series
    /// has a hole after the other has started.
    MaxOfIndependentFirsts,

    /// high and low both `is_finite` at the same index -- `aroonosc.rs:16-20`
    /// (`first_valid_hilo`). Stricter than non-NaN: an INFINITE high is
    /// skipped by the CPU and would be accepted by `AllInputsNonNan`.
    HighLowFinite,

    /// high and low both finite AND strictly positive at the same index --
    /// `parkinson_volatility.rs:214-223` (`is_valid_high_low` /
    /// `first_valid_high_low`). Stricter again, because the indicator takes
    /// `ln(high / low)` and a non-positive price has no logarithm.
    HighLowFiniteAndPositive,

    /// The first index `i >= 1` at which a RETURN can be formed:
    /// `data[i-1].is_finite() && data[i].is_finite() && data[i-1] != 0.0` --
    /// `historical_volatility.rs:334-355` (`valid_return_pair` /
    /// `first_valid_return`). It names a bar at least one later than the first
    /// non-NaN value, and it rejects a zero previous price that
    /// `AllInputsNonNan` would accept and then divide by.
    ConsecutiveValidReturnPair,

    // ------------------------------------------------------------- closer 4
    /// OPEN and CLOSE both non-NaN at the same index -- `qstick.rs:235-243`.
    ///
    /// Declared rather than reusing [`Self::AllInputsNonNan`] because qstick
    /// is registered with [`F64InputKind::Ohlc4`], and under the common rule
    /// that shape resolves to the high/low/close scan. qstick never reads high
    /// or low (its CPU source pair is ("open", "close"), cpu_batch.rs:3709),
    /// so adopting the Hlc index would shift the WHOLE series on any frame
    /// where high or low starts later than open and close -- `first_valid`
    /// sets both the NaN prefix and the seed window, so the disagreement is
    /// not an ULP, it is a different set of windows.
    OpenCloseNonNan,

    /// ONE price series scanned with `is_finite`, not `!is_nan` -- an INFINITE
    /// bar is SKIPPED by the CPU and would be accepted by
    /// [`Self::AllInputsNonNan`]. `dvdiqqe.rs:385` and
    /// `l1_ehlers_phasor.rs:229` (`first_valid`) both scan this way.
    ///
    /// The variant was already REFERENCED by the `dvdiqqe` row and already
    /// handled by `first_valid_for` in
    /// `crates/neoethos-data/src/core/gpu_indicators.rs`; only the declaration
    /// was missing, so the crate did not compile.
    CloseFinite,

    // ------------------------------------------------------------ closer 5
    /// The first index `i >= 1` at which high, low AND close are non-NaN at
    /// BOTH `i - 1` and `i` -- `ultosc.rs:391-401`. The true range reads
    /// `close[i-1]`, so the CPU cannot start at the first bar where the
    /// triple alone is valid; this rule names a bar at least one later than
    /// [`Self::AllInputsNonNan`] over the same three series, and the index
    /// sets both the NaN prefix and the seed window.
    HlcConsecutivePairNonNan,
    /// VOLUME alone, `is_finite` -- `volume_zone_oscillator.rs:271-274`.
    ///
    /// Close is deliberately NOT in this scan. A non-finite close is handled
    /// INSIDE the loop by the `directed` branch (:296-301), which treats it
    /// as "not an up bar" and therefore signs the volume NEGATIVE. Folding
    /// close into first-valid would skip bars the CPU counts as down bars
    /// and shift both EMAs.
    VolumeFiniteOnly,

    // ------------------------------------------------------------ closer 1
    //
    // Three more, all read from the CPU prepare rather than assumed. The first
    // two scan the SAME four series and are still different rules -- which is
    // exactly why neither is folded into `AllInputsNonNan`.

    /// open, high, low and close ALL `is_finite` at the same index --
    /// `accumulation_swing_index.rs:245`, `daily_factor.rs:258`. Open is an
    /// INPUT to both, so the Hlc rules would seed `prev_open` from a bar the
    /// CPU skips.
    Ohlc4AllFinite,

    /// open, high, low and close all `!is_nan` at the same index --
    /// `bop.rs:209-211`. Deliberately NOT [`Self::Ohlc4AllFinite`]: `bop`
    /// ACCEPTS an infinite bar that `accumulation_swing_index` rejects, so on
    /// any frame carrying an infinity the two start at different bars.
    Ohlc4AllNonNan,

    /// OPEN and CLOSE both `is_finite` at the same index --
    /// `andean_oscillator.rs:244`. Distinct from [`Self::OpenCloseNonNan`]
    /// (`qstick.rs:235`) by `is_finite` vs `!is_nan`, and distinct from every
    /// Hlc rule because high and low are never scanned at all.
    OpenCloseFinite,

    // ------------------------------------------------------ closer 6, round 3
    /// The PRICE series and VOLUME both `is_finite` at the same index --
    /// `elastic_volume_weighted_moving_average.rs:308-317` (`find_first_valid`).
    ///
    /// Deliberately NOT [`Self::AllInputsNonNan`], which under a (price,
    /// volume) shape resolves to a `!is_nan` scan. EVWMA divides by the rolling
    /// volume sum at every bar, so an INFINITE volume is rejected by the CPU
    /// scan and would be accepted by the non-NaN one -- and `first_valid` sets
    /// both the NaN prefix and the point the recurrence seeds from, so the two
    /// rules do not perturb the series by an ULP, they shift it.
    PriceVolumeFinite,
}

/// The series shape an f64 kernel expects.
///
/// `Hlc3Slice` and `Hlc3Volume` exist because the CPU default source for `cci`
/// and `mfi` is `hlc3`, not `close`. Handing those kernels `close` computes a
/// DIFFERENT indicator, which a parity tolerance would report as a large
/// numeric error rather than as the contract bug it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F64InputKind {
    /// One price series; CPU default source is `close`.
    CloseSlice,
    /// high / low / close.
    Hlc,
    /// One price series; CPU default source is `hlc3`.
    Hlc3Slice,
    /// (hlc3, volume).
    Hlc3Volume,
    /// (close, volume) — `obv`, `vwma`, `efi`. Distinct from [`Self::
    /// Hlc3Volume`] on purpose: the two differ only in which price series is
    /// paired with volume, and swapping them computes a different indicator
    /// while passing every length and shape check on the way through.
    CloseVolume,
    /// (high, low) with NO close — `medprice`, `midprice`. Their CPU
    /// first-valid rule scans high and low only, so feeding an Ohlc ref would
    /// adopt close's first-valid and shift the whole series.
    HighLow,
    /// (timestamps, close, volume) — `vwap` alone. Its anchor is a calendar
    /// bucket, so the bar timestamps are an INPUT, not metadata.
    TimestampCloseVolume,
    /// One price series whose CPU default source is `hl2` -- `kurtosis`
    /// (cpu_batch.rs:3522) and `alligator` (:13912). A THIRD source alongside
    /// `close` and `hlc3`, declared for the same reason those two are: feeding
    /// this kernel close computes a different indicator and passes every shape
    /// check on the way through.
    Hl2Slice,
    /// (high, low, volume) with NO close -- `emv` alone. Its CPU reference
    /// destructures close as `_close` and scans high, low and volume for
    /// first-valid (emv.rs:196, :219), so an Hlc triple would adopt the wrong
    /// warmup as well as read the wrong series.
    HighLowVolume,
    /// (high, low, close, volume) -- `kvo` alone.
    Hlcv,
    /// (open, high, low, close, volume) -- the FULL bar, for every indicator
    /// whose CPU batch calls `extract_ohlcv_full_input`. Distinct from
    /// [`Self::Hlcv`] because OPEN is an input to the validity gate: a bar
    /// with a non-finite open RESETS `trend_flow_trail`'s whole cascade
    /// (trend_flow_trail.rs:506-516), and a kernel handed the four-pointer
    /// shape would never see it.
    Ohlcv5,
    /// open / high / low / close. `aso` alone: its per-bar value reads
    /// `open[i]` and `open[window_start]`, so open is an INPUT. Distinct
    /// from [`Self::Hlc`] on purpose -- handing aso an Hlc ref would drop
    /// the series the indicator is built on while passing every length
    /// check on the way through.
    Ohlc4,

    // ------------------------------------------------------------ closer 5
    /// One price series whose CPU default source is `hlcc4` = (h + l + 2c)/4
    /// -- the `velocity` family (velocity.rs:32, cpu_batch.rs:4178,
    /// velocity_acceleration_indicator.rs:32,
    /// velocity_acceleration_convergence_divergence_indicator.rs:8162).
    ///
    /// A FOURTH source alongside `close`, `hlc3` and `hl2`, declared for
    /// exactly the reason those three are: handing these kernels `close`
    /// computes a DIFFERENT indicator and passes every shape and length
    /// check on the way through.
    Hlcc4Slice,
    /// VOLUME alone, with no price series at all -- `vosc`, whose CPU batch
    /// calls `extract_volume_input` (cpu_batch.rs:3019) and whose scalar
    /// reference never reads a price. Distinct from [`Self::CloseSlice`]
    /// because a `Slice` shape says nothing about WHICH series is in it, and
    /// feeding vosc close would produce a plausible number from the wrong
    /// data.
    VolumeSlice,

    // ------------------------------------------------------------ closer 1
    /// (open, close, volume) with NO high and NO low -- `dvdiqqe` alone.
    ///
    /// Its CPU reference binds high and low as `_high` and `_low` and never
    /// reads either (`dvdiqqe.rs:445-446`), while `open` is read at every bar.
    /// Declared as its own kind, and NOT folded into [`Self::Hlcv`] or an Ohlcv
    /// shape, for the reason every other narrow kind here exists: a launch arm
    /// that happens to hold four pointers must not be able to hand this kernel
    /// `high` where it asked for `open`. That substitution passes every length
    /// and device check and shows up only as a plausible wrong number.
    ///
    /// The row that names this kind is `dvdiqqe` at :1353. Without the variant
    /// that row does not compile -- which is how the gap was found.
    OpenCloseVolume,

    // ------------------------------------------------------ closer 6, round 3
    /// (hlcc4, volume) -- `elastic_volume_weighted_moving_average` alone.
    ///
    /// A FIFTH price source paired with volume, and it is its own kind for the
    /// reason [`Self::CloseVolume`] and [`Self::Hlc3Volume`] are two kinds over
    /// one device shape: they differ only in WHICH price series is in the pair,
    /// and swapping them computes a different indicator while passing every
    /// length and device check on the way through.
    ///
    /// EVWMA's declared default source is `hlcc4`
    /// (`elastic_volume_weighted_moving_average.rs:113`,
    /// `with_default_candles`), the same evidence that put the `velocity`
    /// family on [`Self::Hlcc4Slice`].
    Hlcc4Volume,
}

/// Every indicator with a real f64 CUDA kernel in this crate, and the entry
/// point that serves it.
///
/// Deliberately short and deliberately explicit. An entry here is a claim that
/// the kernel was written against the named CPU reference and that its warmup,
/// seed window and accumulation order match — see the header of
/// `kernels/cuda/neoethos_f64_kernels.cu`, which lists the reference function
/// for each one. Adding a row without doing that work moves a silent wrongness
/// from the f32 lane into the f64 lane.
pub const F64_KERNELS: &[F64KernelSpec] = &[
    // ---------------------------------------------------- closer 5, round 3
    //
    // Nine indicators that had no reachable f64 entry point until this round.
    // The input kind and the first-valid rule below are each read from the CPU
    // `*_prepare` / `compute_*_batch`, not inferred from the shape:
    //
    // * `rsmk` and `macz` are `Ignored` because their CPU index is NOT a scan
    //   of the series this kind names. `rsmk` scans the LOG-RATIO series --
    //   NaN when either leg is NaN OR the divisor is exactly zero
    //   (rsmk.rs:322-334) -- and no rule here expresses the zero-divisor
    //   clause. `macz` scans CLOSE ALONE (macz.rs:678-681) and handles a NaN
    //   volume INSIDE the loop via `n_vwap_nan`, so adopting volume's first
    //   non-NaN would shift every window on a frame whose volume starts late.
    // * `corrected_moving_average` is `Ignored` because its CPU walks EVERY
    //   bar from index 0 and RESETS the rolling window on any non-finite value
    //   (corrected_moving_average.rs:236, :369) -- a start index would skip
    //   bars the CPU processes.
    // * `rsmk` pairs CLOSE with VOLUME as (main, compare). That is surprising
    //   for a relative-strength indicator and it is what `compute_rsmk_batch`
    //   does (cpu_batch.rs:16445-16447); the kernel computes what the CPU
    //   computes.
    F64KernelSpec {
        indicator_id: "rsmk",
        kernel: F64Kernel::Rsmk,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "squeeze_momentum",
        kernel: F64Kernel::SqueezeMomentum,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "uma",
        kernel: F64Kernel::Uma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "lpc",
        kernel: F64Kernel::Lpc,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "mab",
        kernel: F64Kernel::Mab,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "macz",
        kernel: F64Kernel::Macz,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "vwmacd",
        kernel: F64Kernel::Vwmacd,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "corrected_moving_average",
        kernel: F64Kernel::CorrectedMovingAverage,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "ehlers_undersampled_double_moving_average",
        kernel: F64Kernel::EhlersUndersampledDoubleMovingAverage,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // ------------------------------------------ closer 5, round 2 (adosc)
    F64KernelSpec {
        indicator_id: "adosc",
        kernel: F64Kernel::Adosc,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::Ignored,
    },
    // ------------------------------------------------------ closer 5, round 2
    F64KernelSpec {
        indicator_id: "smoothed_gaussian_trend_filter",
        kernel: F64Kernel::SmoothedGaussianTrendFilter,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "spearman_correlation",
        kernel: F64Kernel::SpearmanCorrelation,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "squeeze_index",
        kernel: F64Kernel::SqueezeIndex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "standardized_psar_oscillator",
        kernel: F64Kernel::StandardizedPsarOscillator,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "statistical_trailing_stop",
        kernel: F64Kernel::StatisticalTrailingStop,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "stochastic_adaptive_d",
        kernel: F64Kernel::StochasticAdaptiveD,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "stochastic_connors_rsi",
        kernel: F64Kernel::StochasticConnorsRsi,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "stochastic_distance",
        kernel: F64Kernel::StochasticDistance,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "stochastic_money_flow_index",
        kernel: F64Kernel::StochasticMoneyFlowIndex,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "supertrend",
        kernel: F64Kernel::Supertrend,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "supertrend_oscillator",
        kernel: F64Kernel::SupertrendOscillator,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "supertrend_recovery",
        kernel: F64Kernel::SupertrendRecovery,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "trend_flow_trail",
        kernel: F64Kernel::TrendFlowTrail,
        input: F64InputKind::Ohlcv5,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "twiggs_money_flow",
        kernel: F64Kernel::TwiggsMoneyFlow,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "volatility_quality_index",
        kernel: F64Kernel::VolatilityQualityIndex,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "vwap_deviation_oscillator",
        kernel: F64Kernel::VwapDeviationOscillator,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "vwap_zscore_with_signals",
        kernel: F64Kernel::VwapZscoreWithSignals,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "sma",
        kernel: F64Kernel::Sma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "ema",
        kernel: F64Kernel::Ema,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "rsi",
        kernel: F64Kernel::Rsi,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "roc",
        kernel: F64Kernel::Roc,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "mom",
        kernel: F64Kernel::Mom,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "atr",
        kernel: F64Kernel::Atr,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "adx",
        kernel: F64Kernel::Adx,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::HlcMaxOfIndependentFirsts,
    },
    F64KernelSpec {
        indicator_id: "willr",
        kernel: F64Kernel::Willr,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "cci",
        kernel: F64Kernel::Cci,
        input: F64InputKind::Hlc3Slice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "mfi",
        kernel: F64Kernel::Mfi,
        input: F64InputKind::Hlc3Volume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ---------------------------------------------------------------- batch 2
    //
    // WHY THESE, AND WHY NOT THE ONES THE BRIEF NAMED.
    //
    // Membership here was decided by MEASUREMENT, not by which indicators are
    // famous. Running the CPU dispatcher over all 342 `ALL_INDICATORS` ids the
    // way `hpc_ta` calls it (`output_id: None`, `params: &[]`) shows 232
    // succeed and 110 are silently dropped, because `resolve_output_id`
    // (cpu_batch.rs:2185) errors for every MULTI-OUTPUT indicator and
    // `hpc_ta.rs:291` swallows that error. So `bollinger_bands`, `donchian`,
    // `aroon`, `di`, `dm`, `chandelier_exit`, `devstop` and `emd` — most of the
    // brief's hint list — produce no CPU column at all today, and an f64 kernel
    // for any of them would have nothing to be checked against.
    //
    // The first three below are the ones that matter most: `hpc_ta`'s
    // 18-indicator period sweep has 13 reachable ids, 10 of which were already
    // served. tsi, obv and vwap were the remainder. With them the reachable
    // sweep is 13/13 on the device.
    F64KernelSpec {
        indicator_id: "tsi",
        kernel: F64Kernel::Tsi,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "obv",
        kernel: F64Kernel::Obv,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // `vwap` IS NOT REGISTERED, and `neoethos_vwap_batch_f64` exists anyway.
    // See WITHHELD below.

    // Core moving averages. Every one is a cross-bar recurrence or a running
    // window, so every one is sequential per column and written in the CPU's
    // exact accumulation order — including `wilders`, whose seed sum groups by
    // four, and `smma`, whose seed sum deliberately does not.
    F64KernelSpec {
        indicator_id: "wma",
        kernel: F64Kernel::Wma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // `wilders` IS NOT REGISTERED, and `neoethos_wilders_batch_f64` exists
    // anyway. See WITHHELD below.
    F64KernelSpec {
        indicator_id: "smma",
        kernel: F64Kernel::Smma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "dema",
        kernel: F64Kernel::Dema,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "tema",
        kernel: F64Kernel::Tema,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "zlema",
        kernel: F64Kernel::Zlema,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "vwma",
        kernel: F64Kernel::Vwma,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // Volatility / directional. `natr` and `adxr` are the two remaining
    // Wilder-family recurrences that are reachable; both are sequential.
    F64KernelSpec {
        indicator_id: "natr",
        kernel: F64Kernel::Natr,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::HlcMaxOfIndependentFirsts,
    },
    F64KernelSpec {
        indicator_id: "adxr",
        kernel: F64Kernel::Adxr,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::HlcCloseOnly,
    },
    F64KernelSpec {
        indicator_id: "efi",
        kernel: F64Kernel::Efi,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // Pointwise and windowed. These are the only ones in this batch that are
    // parallel over (combo, bar), because they are the only ones whose CPU
    // reference has no cross-bar state.
    F64KernelSpec {
        indicator_id: "medprice",
        kernel: F64Kernel::Medprice,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "wclprice",
        kernel: F64Kernel::Wclprice,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "midpoint",
        kernel: F64Kernel::Midpoint,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "midprice",
        kernel: F64Kernel::Midprice,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "rocp",
        kernel: F64Kernel::Rocp,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "rocr",
        kernel: F64Kernel::Rocr,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ---------------------------------------------------------------- shard 2
    //
    // `wilders` LEAVES THE WITHHELD LIST.
    //
    // It was withheld because vector-ta's own scalar and AVX CPU paths
    // disagreed by 1 ULP at the seed bar, so there was no single CPU answer for
    // a device result to be in parity with. That is no longer true, and it was
    // fixed in the CPU rather than worked around: `wilders_scalar`,
    // `wilders_avx2`, `wilders_avx512_short` and `wilders_avx512_long` now all
    // call `wilders_seed_sum` (moving_averages/wilders.rs), which is the 4-wide
    // scalar association.
    //
    // Which association, and why that one: the recurrence was never in dispute
    // — all four paths always ran `y = (x - y).mul_add(alpha, y)`, one rounding,
    // same order. Only the warm-up seed differed, summing the same `period`
    // values 4-wide (scalar), 8-wide (avx512_short) and 16-wide (avx512_long).
    // The 4-wide scalar tree wins because it is what `Kernel::ScalarBatch`
    // runs — the path `hpc_ta` takes and the one every existing fixture in this
    // lane was recorded against — and because the vector associations were an
    // artefact of register width. An indicator whose value changes with
    // `-C target-cpu` is a bug, not a tolerance.
    //
    // `neoethos_wilders_batch_f64` was already written against that same 4-wide
    // seed, so registering it is the entire change; no kernel was touched.
    //
    // `vwap`, the other withheld id, has the identical shape (the `mul_add`
    // recurrence agrees; only the chunked `volume_sum` / `vol_price_sum` seed
    // differs) and the identical remedy. It is not registered here because
    // `moving_averages/vwap_kernel.cu` and `vwap.rs` belong to another shard.
    F64KernelSpec {
        indicator_id: "wilders",
        kernel: F64Kernel::Wilders,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // `sqwma` — kernel written by this shard INTO ITS OWN FILE,
    // `kernels/cuda/moving_averages/sqwma_kernel.cu`, beside the f32 kernels
    // that 180 wrappers still call. Reference `sqwma_scalar`
    // (moving_averages/sqwma.rs:286).
    //
    // first_valid: `AllInputsNonNan` — `sqwma_with_kernel` takes
    // `data.iter().position(|x| !x.is_nan())` over the single source series, so
    // the common rule is the right one here. Note that the WARMUP is NOT the
    // common one: `warm = first + period + 1`, one bar later than the
    // `first + period - 1` most of this crate's moving averages use, because
    // the emit loop is `for j in (first + period + 1)..n`.
    F64KernelSpec {
        indicator_id: "sqwma",
        kernel: F64Kernel::Sqwma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ------------------------------------------------------------ shard 2
    //
    // Kernels written in the indicator's OWN `.cu` file, beside the f32 entry
    // points the 180 f32 wrappers still call. Each row is a claim that the
    // kernel was transcribed from the CPU reference named in that file's
    // `S2 f64 LANE` header, that its warmup and first-valid rule match, and
    // that its accumulation order reproduces the CPU's rounding count.
    //
    // `tradjema` is the one that is NOT `AllInputsNonNan`: `tradjema_prepare`
    // (:260) scans CLOSE ALONE, the `adxr` rule, so it declares
    // `HlcCloseOnly`. Reading high/low as well would shift the whole series
    // on any frame where either starts later than close.
    F64KernelSpec {
        indicator_id: "gaussian",
        kernel: F64Kernel::Gaussian,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "reflex",
        kernel: F64Kernel::Reflex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "jma",
        kernel: F64Kernel::Jma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "maaq",
        kernel: F64Kernel::Maaq,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "tradjema",
        kernel: F64Kernel::Tradjema,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::HlcCloseOnly,
    },
    F64KernelSpec {
        indicator_id: "pwma",
        kernel: F64Kernel::Pwma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "nama",
        kernel: F64Kernel::Nama,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "sama",
        kernel: F64Kernel::Sama,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "ehlers_kama",
        kernel: F64Kernel::EhlersKama,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "ehlers_itrend",
        kernel: F64Kernel::EhlersItrend,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "pvi",
        kernel: F64Kernel::Pvi,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "vpt",
        kernel: F64Kernel::Vpt,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "trix",
        kernel: F64Kernel::Trix,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "rsx",
        kernel: F64Kernel::Rsx,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ---------------------------------------------------------------- shard 6
    //
    // Every row below names an entry point in the INDICATOR'S OWN `.cu` file,
    // written against the CPU reference quoted in that file's `f64 LANE` header
    // and reachable through `F64Kernel::module_stem`. None of them lives in
    // `neoethos_f64_kernels.cu`: the instruction was to fix what the crate
    // already ships, in place.
    //
    // The first-valid rule on each row was read out of that indicator's
    // `*_prepare`, not inferred from its input shape. Four of them are NOT the
    // common rule and each would shift the whole series if it were.

    // --- single price series, CPU source `close`, common first-valid rule.
    F64KernelSpec {
        indicator_id: "fwma",
        kernel: F64Kernel::Fwma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "hwma",
        kernel: F64Kernel::Hwma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "jsa",
        kernel: F64Kernel::Jsa,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "nma",
        kernel: F64Kernel::Nma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "swma",
        kernel: F64Kernel::Swma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "trendflex",
        kernel: F64Kernel::Trendflex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "vpwma",
        kernel: F64Kernel::Vpwma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "cfo",
        kernel: F64Kernel::Cfo,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "var",
        kernel: F64Kernel::Var,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "bollinger_bands_width",
        kernel: F64Kernel::BollingerBandsWidth,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "dec_osc",
        kernel: F64Kernel::DecOsc,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "voss",
        kernel: F64Kernel::Voss,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "percentile_nearest_rank",
        kernel: F64Kernel::PercentileNearestRank,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // --- high/low/close, all three non-NaN at the SAME index.
    F64KernelSpec {
        indicator_id: "ttm_trend",
        kernel: F64Kernel::TtmTrend,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "vi",
        kernel: F64Kernel::Vi,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // --- (high, low) with no close.
    F64KernelSpec {
        indicator_id: "cvi",
        kernel: F64Kernel::Cvi,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "correl_hl",
        kernel: F64Kernel::CorrelHl,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // --- the four that do NOT use the common rule. Read the rule's doc
    //     comment for the CPU line each one came from.
    F64KernelSpec {
        indicator_id: "aroonosc",
        kernel: F64Kernel::Aroonosc,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::HighLowFinite,
    },
    F64KernelSpec {
        indicator_id: "parkinson_volatility",
        kernel: F64Kernel::ParkinsonVolatility,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::HighLowFiniteAndPositive,
    },
    F64KernelSpec {
        indicator_id: "donchian",
        kernel: F64Kernel::Donchian,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::MaxOfIndependentFirsts,
    },
    F64KernelSpec {
        indicator_id: "historical_volatility",
        kernel: F64Kernel::HistoricalVolatility,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::ConsecutiveValidReturnPair,
    },

    // --- `vwap` LEAVES THE WITHHELD LIST.
    //
    // It was withheld because vector-ta answered "what is vwap" two ways. The
    // cause was NOT scalar-vs-AVX: `vwap_avx2` and `vwap_avx512` delegate to
    // `vwap_scalar` verbatim, and so do all four `vwap_row_avx*`. It was
    // `vwap_row_scalar_pv`, a SECOND implementation reached only by the
    // `Kernel::Scalar` arm of `vwap_batch_inner` / `vwap_batch_inner_into`,
    // which consumed a precomputed `pv[i] = price * volume` and added it
    // (TWO roundings) where `vwap_scalar` writes
    // `vol_price_sum = p.mul_add(v, vol_price_sum)` (ONE).
    //
    // Fixed in the CPU, like `wilders` before it, and fixed by DELETION rather
    // than by reconciliation: both batch arms now call `vwap_row_scalar`, the
    // `pv` precompute is gone, and `vwap_row_scalar_pv` no longer exists. The
    // mul_add form wins on three counts -- fewer roundings, already used by
    // four of the five kernel arms, and identical to what the public
    // single-series `vwap()` has always returned.
    //
    // `neoethos_vwap_batch_f64` was already written against `vwap_scalar`, so
    // registering it is the entire device-side change; no kernel was touched.
    F64KernelSpec {
        indicator_id: "vwap",
        kernel: F64Kernel::Vwap,
        input: F64InputKind::TimestampCloseVolume,
        first_valid: F64FirstValidRule::Ignored,
    },

    // ------------------------------------------------------------ shard 2
    //
    // Kernels written in the indicator's OWN `.cu` file, beside the f32 entry
    // points the 180 f32 wrappers still call. Each row is a claim that the
    // kernel was transcribed from the CPU reference named in that file's
    // `S2 f64 LANE` header, that its warmup and first-valid rule match, and
    // that its accumulation order reproduces the CPU's rounding count.
    //
    // `tradjema` is the one that is NOT `AllInputsNonNan`: `tradjema_prepare`
    // (:260) scans CLOSE ALONE, the `adxr` rule, so it declares
    // `HlcCloseOnly`. Reading high/low as well would shift the whole series
    // on any frame where either starts later than close.
    F64KernelSpec {
        indicator_id: "minmax",
        kernel: F64Kernel::Minmax,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "chandelier_exit",
        kernel: F64Kernel::ChandelierExit,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::HlcCloseOnly,
    },
    F64KernelSpec {
        indicator_id: "devstop",
        kernel: F64Kernel::Devstop,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ---------------------------------------------------------------- shard 1
    //
    // Nineteen indicators whose f64 kernel was written INTO THE FILE THE
    // INDICATOR ALREADY SHIPS IN, beside the f32 entry points the f32 wrappers
    // still call. Each `.cu` carries an `S1 f64 LANE` header naming the exact
    // CPU function, line number, warmup and rounding count it was written
    // against; the claim an entry here makes is that that work was done.
    //
    // FIRST-VALID: every one of the nineteen is `AllInputsNonNan`, and that was
    // READ OFF THE CPU rather than assumed. The rule the brief warns about --
    // three of the high/low/close indicators using a different scan from the
    // other three -- does not recur here: `chop` (chop.rs:281-287) and `stochf`
    // (stochf.rs:387-389) both scan high, low and close SIMULTANEOUSLY, which
    // is the common rule, not `adx`'s max-of-independent-firsts and not
    // `adxr`'s close-only. `kvo` (kvo.rs:292-297) scans all four series
    // simultaneously and `emv` (emv.rs:219) scans high, low and volume
    // simultaneously -- close is not in its scan at all, which is why it
    // declares `HighLowVolume` and not `Hlc`.
    //
    // SOURCE SERIES: `kurtosis` and `alligator` are `hl2`-sourced
    // (cpu_batch.rs:3522, :13912), a third source this table had no name for
    // until now. `gatorosc` is close-sourced (:14908) even though it is the
    // same family as `alligator` -- taking the source from the family rather
    // than from the batch arm would have computed the wrong series for one of
    // the two.
    //
    // PERIOD-INVARIANT ids in this block (their CPU batch arm never reads the
    // swept `period`): apo, vidya, gatorosc, ppo, pma, alligator, nvi, stochf,
    // emv, kvo. They emit identical rows for every period, faithfully -- see
    // `F64Kernel::is_period_invariant`.
    //
    // MULTI-OUTPUT ids emit the series the CPU's `output_id: "value"` maps to,
    // and only that one: alligator -> jaw, gatorosc -> upper, stochf -> k,
    // fisher -> fisher, pma -> predict. Named here so a caller cannot mistake
    // a one-matrix result for the whole indicator.
    F64KernelSpec {
        indicator_id: "apo",
        kernel: F64Kernel::Apo,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "vidya",
        kernel: F64Kernel::Vidya,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "gatorosc",
        kernel: F64Kernel::Gatorosc,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "ppo",
        kernel: F64Kernel::Ppo,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "pma",
        kernel: F64Kernel::Pma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "kama",
        kernel: F64Kernel::Kama,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "linreg",
        kernel: F64Kernel::Linreg,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "edcf",
        kernel: F64Kernel::Edcf,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "alma",
        kernel: F64Kernel::Alma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "hma",
        kernel: F64Kernel::Hma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "kurtosis",
        kernel: F64Kernel::Kurtosis,
        input: F64InputKind::Hl2Slice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "alligator",
        kernel: F64Kernel::Alligator,
        input: F64InputKind::Hl2Slice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "nvi",
        kernel: F64Kernel::Nvi,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "fisher",
        kernel: F64Kernel::Fisher,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "safezonestop",
        kernel: F64Kernel::Safezonestop,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "chop",
        kernel: F64Kernel::Chop,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "stochf",
        kernel: F64Kernel::Stochf,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "emv",
        kernel: F64Kernel::Emv,
        input: F64InputKind::HighLowVolume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "kvo",
        kernel: F64Kernel::Kvo,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ------------------------------------------------------------- shard 4
    //
    // Twenty-four indicators whose kernel lives in the file the indicator
    // already ships in. Ten are PERIOD-INVARIANT because their CPU batch
    // function reads NAMED window parameters and never `period` -- that is
    // faithful, not a shortcut, and `F64Kernel::is_period_invariant` reports
    // it so telemetry can explain the identical rows instead of leaving them
    // to be discovered.
    // er.rs:218 scans the single series; er.rs:322 `er_scalar`.
    F64KernelSpec {
        indicator_id: "er",
        kernel: F64Kernel::Er,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // linearreg_angle.rs:218.
    F64KernelSpec {
        indicator_id: "linearreg_angle",
        kernel: F64Kernel::LinearregAngle,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // linearreg_intercept.rs:198.
    F64KernelSpec {
        indicator_id: "linearreg_intercept",
        kernel: F64Kernel::LinearregIntercept,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // moving_averages/highpass_2_pole.rs:352. Served by
    // `moving_averages/highpass2_kernel.cu` -- the FILE stem and the indicator
    // id differ, proven by `highpass2_wrapper.rs:160`.
    F64KernelSpec {
        indicator_id: "highpass_2_pole",
        kernel: F64Kernel::Highpass2Pole,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // moving_averages/supersmoother_3_pole.rs:185.
    F64KernelSpec {
        indicator_id: "supersmoother_3_pole",
        kernel: F64Kernel::Supersmoother3Pole,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // moving_averages/cwma.rs:230 (`cwma_prepare`).
    F64KernelSpec {
        indicator_id: "cwma",
        kernel: F64Kernel::Cwma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // moving_averages/ehma.rs:224 (`ehma_prepare`).
    F64KernelSpec {
        indicator_id: "ehma",
        kernel: F64Kernel::Ehma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // cmo.rs:188.
    F64KernelSpec {
        indicator_id: "cmo",
        kernel: F64Kernel::Cmo,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // stddev.rs:230. nbdev is pinned at the batch default 1.0, which is
    // the `stddev_scalar_nbdev1` path (stddev.rs:360).
    F64KernelSpec {
        indicator_id: "stddev",
        kernel: F64Kernel::Stddev,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // ui.rs:112.
    F64KernelSpec {
        indicator_id: "ui",
        kernel: F64Kernel::Ui,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // bollinger_bands.rs:285 (`bb_prepare`). Emits the UPPER band,
    // which is what cpu_batch.rs:5514 maps output_id "value" to.
    F64KernelSpec {
        indicator_id: "bollinger_bands",
        kernel: F64Kernel::BollingerBands,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // macd.rs:602. PERIOD-INVARIANT: cpu_batch.rs:5444-5447 reads
    // fast/slow/signal and never `period`. Emits the MACD line (:5463).
    F64KernelSpec {
        indicator_id: "macd",
        kernel: F64Kernel::Macd,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // ift_rsi.rs:242. PERIOD-INVARIANT: cpu_batch.rs:3142 reads
    // rsi_period/wma_period. The (5, 9) defaults take the SPECIALISED CPU path
    // `ift_rsi_scalar_default_5_9`, whose seed multiplies by 0.2 where the
    // generic path divides by 5.0 -- not the same number.
    F64KernelSpec {
        indicator_id: "ift_rsi",
        kernel: F64Kernel::IftRsi,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // damiani_volatmeter.rs:355 scans close. PERIOD-INVARIANT:
    // cpu_batch.rs:14380-14384 reads four window names. The batch passes ONE
    // slice, and prepare:323 expands it to (slice, slice, slice).
    F64KernelSpec {
        indicator_id: "damiani_volatmeter",
        kernel: F64Kernel::DamianiVolatmeter,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // wavetrend.rs:282. SOURCE IS hlc3 (cpu_batch.rs:6490), not close.
    // PERIOD-INVARIANT: channel/average/ma lengths. Emits wt1 (:6492).
    F64KernelSpec {
        indicator_id: "wavetrend",
        kernel: F64Kernel::Wavetrend,
        input: F64InputKind::Hlc3Slice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // dx.rs -- the triple is scanned SIMULTANEOUSLY, so the common rule.
    F64KernelSpec {
        indicator_id: "dx",
        kernel: F64Kernel::Dx,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // moving_averages/frama.rs:287 -- `find(|i| !h[i].is_nan() &&
    // !l[i].is_nan() && !c[i].is_nan())`, i.e. the triple simultaneously.
    F64KernelSpec {
        indicator_id: "frama",
        kernel: F64Kernel::Frama,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // cksp.rs:281 -- `close.iter().position(...)`. High and low are
    // never scanned. PERIOD-INVARIANT: cpu_batch.rs:14285-14287 reads p/x/q.
    // Emits the LONG stop (:14308).
    F64KernelSpec {
        indicator_id: "cksp",
        kernel: F64Kernel::Cksp,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::HlcCloseOnly,
    },
    // ttm_squeeze.rs:384 -- close alone. PERIOD-INVARIANT: the
    // `length` is pinned at 20 because ttm_squeeze.rs:402-408 only takes
    // `ttm_squeeze_scalar_classic` at exactly 20 with the default multipliers;
    // any other length is a DIFFERENT CPU function. Emits momentum (:5913).
    F64KernelSpec {
        indicator_id: "ttm_squeeze",
        kernel: F64Kernel::TtmSqueeze,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::HlcCloseOnly,
    },
    // mass.rs -- (high, low) with no close.
    F64KernelSpec {
        indicator_id: "mass",
        kernel: F64Kernel::Mass,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // aroon_scalar starts at absolute index `length` and never reads
    // a first-valid index -- see the header of aroon_kernel.cu.
    F64KernelSpec {
        indicator_id: "aroon",
        kernel: F64Kernel::Aroon,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::Ignored,
    },
    // acosc has no length parameter and no first-valid scan -- see the
    // header of oscillators/acosc_kernel.cu. PERIOD-INVARIANT by construction.
    F64KernelSpec {
        indicator_id: "acosc",
        kernel: F64Kernel::Acosc,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::Ignored,
    },
    // vpci.rs:389 `first_valid_both` -- close AND volume non-NaN at the
    // same index, which is exactly the common rule for a pair.
    // PERIOD-INVARIANT: cpu_batch.rs:5797-5798 reads short_range/long_range.
    // Emits the VPCI line (:5776).
    F64KernelSpec {
        indicator_id: "vpci",
        kernel: F64Kernel::Vpci,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // `ad_with_kernel` (ad.rs:209) calls `alloc_with_nan_prefix(size, 0)`
    // and `ad_scalar` (:298) starts at index 0 -- there is no first-valid scan
    // anywhere in this indicator. PERIOD-INVARIANT: it takes no parameters.
    F64KernelSpec {
        indicator_id: "ad",
        kernel: F64Kernel::Ad,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::Ignored,
    },
    // dvdiqqe.rs:385 scans CLOSE with `is_finite`, not `!is_nan`. PERIOD-SWEPT:
    // cpu_batch.rs:14490 reads a parameter literally named `period` (13), so
    // unlike the ten invariant rows above this one uses `periods[combo]`.
    // Emits the `dvdi` line; fast_tl / slow_tl / center_line are separate
    // outputs and nothing they compute feeds this matrix.
    F64KernelSpec {
        indicator_id: "dvdiqqe",
        kernel: F64Kernel::Dvdiqqe,
        input: F64InputKind::OpenCloseVolume,
        first_valid: F64FirstValidRule::CloseFinite,
    },
    // cci_cycle.rs:409 scans the single close series. PERIOD-INVARIANT:
    // cpu_batch.rs:3454 reads `length` (10) and `factor` (0.5). `length` is
    // pinned at 10 for a second reason as well -- `cci_cycle_compute_from_
    // parts:526` sends `length > 16` to `fused_pf_and_normalize_scalar`, a
    // different function from the `naive_` one this kernel mirrors.
    F64KernelSpec {
        indicator_id: "cci_cycle",
        kernel: F64Kernel::CciCycle,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ----------------------------------------------------------- shard 3
    //
    // Twenty-five indicators whose f64 kernel was written into the `.cu` file
    // the indicator already ships in, beside the f32 entry points the f32
    // wrappers still call. Each row is a claim that the kernel was written
    // against the CPU reference named in that file's `S3 f64 LANE` header and
    // that its warmup, seed window and accumulation order match it.
    //
    // THE FIRST-VALID RULES ARE NOT ALL THE SAME, AND TWO ARE NOT THE COMMON
    // ONE:
    //   * `aso` -> HlcCloseOnly. `aso_prepare` (aso.rs:405) scans CLOSE ALONE
    //     -- open, high and low are never looked at. Same rule adxr uses.
    //   * `wad` and `mama` -> Ignored. Neither CPU reference computes a
    //     first-non-NaN index at all: `wad_with_kernel` starts at bar 0 with
    //     out[0] = 0.0, and `mama` uses the literal warmup 10. Declaring
    //     AllInputsNonNan for them would be a claim the reference does not
    //     make.
    //   * everything else -> AllInputsNonNan, verified per indicator:
    //     chande's `first_valid3` (chande.rs:242), di's inline triple scan
    //     (di.rs:225), kdj's zipped scan (kdj.rs:202) and sar's zipped
    //     high/low scan (sar.rs:215) are all SIMULTANEOUS scans, NOT the
    //     max-of-independent-firsts rule adx and natr use, even though
    //     chande and di share a Wilder core with them.
    F64KernelSpec {
        indicator_id: "deviation",
        kernel: F64Kernel::Deviation,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "mean_ad",
        kernel: F64Kernel::MeanAd,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // PERIOD-INVARIANT: the CPU batch reads named parameters, never `period`.
    F64KernelSpec {
        indicator_id: "ao",
        kernel: F64Kernel::Ao,
        input: F64InputKind::Hl2Slice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "linearreg_slope",
        kernel: F64Kernel::LinearregSlope,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "tsf",
        kernel: F64Kernel::Tsf,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "highpass",
        kernel: F64Kernel::Highpass,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "decycler",
        kernel: F64Kernel::Decycler,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "supersmoother",
        kernel: F64Kernel::Supersmoother,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "tilson",
        kernel: F64Kernel::Tilson,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // PERIOD-INVARIANT: the CPU batch reads named parameters, never `period`.
    F64KernelSpec {
        indicator_id: "wad",
        kernel: F64Kernel::Wad,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    // PERIOD-INVARIANT: the CPU batch reads named parameters, never `period`.
    F64KernelSpec {
        indicator_id: "sar",
        kernel: F64Kernel::Sar,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // PERIOD-INVARIANT: the CPU batch reads named parameters, never `period`.
    F64KernelSpec {
        indicator_id: "dti",
        kernel: F64Kernel::Dti,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "zscore",
        kernel: F64Kernel::Zscore,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "pfe",
        kernel: F64Kernel::Pfe,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "chande",
        kernel: F64Kernel::Chande,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "di",
        kernel: F64Kernel::Di,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // PERIOD-INVARIANT: the CPU batch reads named parameters, never `period`.
    F64KernelSpec {
        indicator_id: "kdj",
        kernel: F64Kernel::Kdj,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "aso",
        kernel: F64Kernel::Aso,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::HlcCloseOnly,
    },
    // PERIOD-INVARIANT: the CPU batch reads named parameters, never `period`.
    F64KernelSpec {
        indicator_id: "wto",
        kernel: F64Kernel::Wto,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // PERIOD-INVARIANT: the CPU batch reads named parameters, never `period`.
    F64KernelSpec {
        indicator_id: "range_filter",
        kernel: F64Kernel::RangeFilter,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "correlation_cycle",
        kernel: F64Kernel::CorrelationCycle,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // PERIOD-INVARIANT: the CPU batch reads named parameters, never `period`.
    F64KernelSpec {
        indicator_id: "mama",
        kernel: F64Kernel::Mama,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "volume_adjusted_ma",
        kernel: F64Kernel::VolumeAdjustedMa,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "reverse_rsi",
        kernel: F64Kernel::ReverseRsi,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "ehlers_ecema",
        kernel: F64Kernel::EhlersEcema,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ---------------------------------------------------------------- closer 6
    //
    // Three multi-output indicators the earlier batches skipped for exactly
    // that reason -- `resolve_output_id` (cpu_batch.rs:2185) errors for a
    // multi-output id when `hpc_ta` calls with `output_id: None`, so the
    // comment on batch 2 above reads "an f64 kernel for any of them would have
    // nothing to be checked against".
    //
    // THAT IS TRUE OF `hpc_ta`'S CALL, NOT OF THE INDICATOR. Each of these
    // CPU batch functions answers `output_id == "value"` perfectly well and
    // names which column it means: emd -> upperband (cpu_batch.rs:14554),
    // keltner -> upper_band (:6232), stoch -> k (:5603). The oracle exists;
    // it is reached by asking for "value" explicitly rather than by passing
    // None. Each kernel emits that column and its `.cu` header says so.
    F64KernelSpec {
        indicator_id: "emd",
        kernel: F64Kernel::Emd,
        input: F64InputKind::HighLow,
        // `emd_prepare:333` scans high and low SIMULTANEOUSLY
        // (`!high[i].is_nan() && !low[i].is_nan()`), so this is the plain
        // conjunction rule -- NOT `MaxOfIndependentFirsts`, which is what
        // donchian's independent scans need. The two answers differ whenever
        // one series has a hole after the other has started.
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "keltner",
        kernel: F64Kernel::Keltner,
        input: F64InputKind::Hlc,
        // `keltner_with_kernel:293-296` scans CLOSE ALONE. Declaring
        // `AllInputsNonNan` over the triple here would adopt high's or low's
        // first bar and shift the entire series on any frame where close is
        // not the last series to start.
        first_valid: F64FirstValidRule::HlcCloseOnly,
    },
    F64KernelSpec {
        indicator_id: "stoch",
        kernel: F64Kernel::Stoch,
        input: F64InputKind::Hlc,
        // `stoch_with_kernel:297-301` scans high, low and close
        // simultaneously.
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ------------------------------------------------------------- closer 4
    //
    // Seven indicators whose f64 kernel was written INTO THE FILE THE
    // INDICATOR ALREADY SHIPS IN, beside the f32 entry points the f32 wrappers
    // still call. Each `.cu` carries an `f64 LANE  --  closer 4` header naming
    // the exact CPU function, line number, warmup and rounding structure it
    // was written against; a row here is a claim that that work was done.
    //
    // FIRST-VALID was read out of each indicator's own `*_prepare`, never
    // inferred from the input shape. Five are the common rule, and the two
    // that are not would each shift the entire series if they were:
    //
    // * `qstick` -> `OpenCloseNonNan`. `qstick.rs:235-243` scans OPEN AND
    //   CLOSE. It is registered `Ohlc4` so the kernel can be handed the four
    //   price pointers the resident upload already carries, but it reads only
    //   two of them, and the Ohlc rule would adopt high/low's start.
    // * `rolling_z_score_trend` -> `Ignored`. Both of its CPU paths iterate
    //   `for i in 0..data.len()` (:304, :441); there is no first-valid scan in
    //   this indicator at all, and its validity test is
    //   `longest_valid_run(data) >= lookback` (:262-272), which the kernel
    //   computes itself. Reporting 0 is the truth, not a shortcut.
    //
    // TWO OF THE SEVEN HAVE NO CPU COLUMN UNDER THE DEFAULT OUTPUT ID, and
    // that is stated here rather than left for a parity run to discover:
    // `compute_random_walk_index_batch` accepts only "high"/"low"
    // (cpu_batch.rs:10352-10359) and `compute_rolling_z_score_trend_batch`
    // only "zscore"/"momentum" (:8046-8053). Both REJECT "value". The kernels
    // emit `high` and `zscore` respectively -- the first-declared series of
    // each -- so a parity check must ask the CPU for that id explicitly.

    // --- single price series, CPU source `close`, common first-valid rule.
    // psychological_line.rs:222 scans the single series; :249/:395 for the
    // value; warmup `first + length` -- ONE BAR LATER than the usual
    // `first + length - 1`, because the value counts COMPARISONS.
    F64KernelSpec {
        indicator_id: "psychological_line",
        kernel: F64Kernel::PsychologicalLine,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // rank_correlation_index.rs:224 scans the single series; :251 for the
    // value. Warmup `first + length - 1`.
    F64KernelSpec {
        indicator_id: "rank_correlation_index",
        kernel: F64Kernel::RankCorrelationIndex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // moving_averages/sinwma.rs:307 scans the single series; :494 for the
    // value, :273 for the weights. Warmup `first + period - 1`.
    F64KernelSpec {
        indicator_id: "sinwma",
        kernel: F64Kernel::Sinwma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // moving_averages/srwma.rs:203 scans the single series; :287 for the
    // value. Warmup `first + period + 1` -- TWO BARS LATER than the crate's
    // other weighted moving averages, because `srwma_scalar` starts its emit
    // loop at `first_val + period + 1` (:306).
    F64KernelSpec {
        indicator_id: "srwma",
        kernel: F64Kernel::Srwma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // --- high/low/close, all three non-NaN at the SAME index.
    // random_walk_index.rs:247 uses `first_valid_hlc`, the simultaneous scan
    // -- NOT adx's max-of-independent-firsts and NOT adxr's close-only.
    // :359 for the value. Warmup `first + length - 1`.
    F64KernelSpec {
        indicator_id: "random_walk_index",
        kernel: F64Kernel::RandomWalkIndex,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // --- the two that do NOT use the common rule.
    F64KernelSpec {
        indicator_id: "qstick",
        kernel: F64Kernel::Qstick,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::OpenCloseNonNan,
    },
    F64KernelSpec {
        indicator_id: "rolling_z_score_trend",
        kernel: F64Kernel::RollingZScoreTrend,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "nadaraya_watson_envelope",
        kernel: F64Kernel::NadarayaWatsonEnvelope,
        input: F64InputKind::CloseSlice,
        // `nwe_prepare:397-400` scans the single source series for the first
        // non-NaN. `compute_nadaraya_watson_envelope_batch:15611` extracts
        // that series with `extract_slice_input(.., "close")`, so the source
        // is close and not hlc3.
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ------------------------------------------------------------- closer 3
    //
    // Six kernels written into the file each indicator already ships in --
    // search those files for "f64 LANE  --  closer C3". Each header names the
    // CPU function the kernel was written against and, where the file already
    // carried an f64 entry point, states the MEASURED reason that one could not
    // simply be registered instead.
    F64KernelSpec {
        indicator_id: "l1_ehlers_phasor",
        kernel: F64Kernel::L1EhlersPhasor,
        input: F64InputKind::CloseSlice,
        // `l1_ehlers_phasor.rs:229` scans with `is_finite`, so an INFINITE bar
        // is skipped. `AllInputsNonNan` would accept it and start the phasor a
        // bar early, shifting the whole series.
        first_valid: F64FirstValidRule::CloseFinite,
    },
    F64KernelSpec {
        indicator_id: "l2_ehlers_signal_to_noise",
        kernel: F64Kernel::L2EhlersSignalToNoise,
        // The CPU source is `hl2`, which `Candles::compute_hl2` defines as
        // `(h + l) / 2.0`, so the pair IS the whole input and the kernel forms
        // the source itself.
        input: F64InputKind::HighLow,
        // `first_valid_triple` (:263) requires source, high and low all
        // `is_finite` at the same index; with source == hl2 that is exactly
        // "high and low both finite".
        first_valid: F64FirstValidRule::HighLowFinite,
    },
    F64KernelSpec {
        indicator_id: "kairi_relative_index",
        kernel: F64Kernel::KairiRelativeIndex,
        input: F64InputKind::CloseSlice,
        // `compute_default_sma50_into` (:732) fills the output with NaN and
        // then walks from index 0. It never consults a first-valid index, so
        // declaring one here would imply a warmup prefix that does not exist.
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "linear_correlation_oscillator",
        kernel: F64Kernel::LinearCorrelationOscillator,
        input: F64InputKind::CloseSlice,
        // `linear_correlation_oscillator_prepare:253` -- `!v.is_nan()`.
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "medium_ad",
        kernel: F64Kernel::MediumAd,
        input: F64InputKind::CloseSlice,
        // `medium_ad_with_kernel:199` -- `!x.is_nan()`.
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "marketefi",
        kernel: F64Kernel::Marketefi,
        // (high, low, volume) with NO close: `marketefi_first_valid` (:206)
        // scans exactly those three and the value is `(high - low) / volume`.
        // An Hlc triple would both read the wrong series and adopt close's
        // first bar.
        input: F64InputKind::HighLowVolume,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // ------------------------------------------------------------- closer 5
    //
    // Eleven rows, each a claim that the kernel named by the variant was
    // written against the CPU reference cited in its `.cu` header and that
    // its warmup, seed window and accumulation order match.
    //
    // FOUR DECLARE `Ignored` FIRST-VALID, and that is the CPU behaviour
    // rather than an omission: velocity_acceleration_indicator (:657),
    // its convergence/divergence twin (:624), trend_direction_force_index
    // (:485) and trend_continuation_factor (:429) all walk from index 0
    // and RESET every accumulator on a non-finite bar, so there is no
    // warmup prefix to align and passing an index in would shift the
    // series. volume_weighted_rsi (:403) does the same.
    //
    // TWO USE RULES NOTHING ELSE USES: `ultosc` needs a CONSECUTIVE PAIR
    // because its true range reads close[i-1], and `volume_zone_oscillator`
    // scans VOLUME ALONE because a NaN close is a signed-negative bar
    // rather than a skipped one.
    F64KernelSpec {
        indicator_id: "velocity",
        kernel: F64Kernel::Velocity,
        input: F64InputKind::Hlcc4Slice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "velocity_acceleration_indicator",
        kernel: F64Kernel::VelocityAccelerationIndicator,
        input: F64InputKind::Hlcc4Slice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "velocity_acceleration_convergence_divergence_indicator",
        kernel: F64Kernel::VelocityAccelerationConvergenceDivergenceIndicator,
        input: F64InputKind::Hlcc4Slice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "trend_direction_force_index",
        kernel: F64Kernel::TrendDirectionForceIndex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "trend_continuation_factor",
        kernel: F64Kernel::TrendContinuationFactor,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "trima",
        kernel: F64Kernel::Trima,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "trend_trigger_factor",
        kernel: F64Kernel::TrendTriggerFactor,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::HighLowFinite,
    },
    F64KernelSpec {
        indicator_id: "volume_weighted_rsi",
        kernel: F64Kernel::VolumeWeightedRsi,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "volume_zone_oscillator",
        kernel: F64Kernel::VolumeZoneOscillator,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::VolumeFiniteOnly,
    },
    F64KernelSpec {
        indicator_id: "vosc",
        kernel: F64Kernel::Vosc,
        input: F64InputKind::VolumeSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "ultosc",
        kernel: F64Kernel::Ultosc,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::HlcConsecutivePairNonNan,
    },
    F64KernelSpec {
        indicator_id: "momentum_ratio_oscillator",
        kernel: F64Kernel::MomentumRatioOscillator,
        input: F64InputKind::CloseSlice,
        // `momentum_ratio_oscillator_compute_into` (:292) loops
        // `for i in 0..data.len()` and never consults a first-valid index, and
        // `with_kernel` (:406) allocates with NO NaN prefix. Declaring a warmup
        // would blank bars the CPU fills.
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "on_balance_volume_oscillator",
        kernel: F64Kernel::OnBalanceVolumeOscillator,
        // `extract_close_volume_input(.., "close")` (cpu_batch.rs:9740) -- the
        // price is CLOSE, not hlc3.
        input: F64InputKind::CloseVolume,
        // `with_kernel` (:667) is `alloc_with_nan_prefix(len, 0)` and the walk
        // starts at index 0, resetting its own state on any non-finite bar.
        first_valid: F64FirstValidRule::Ignored,
    },
    // ------------------------------------------------------------ closer 2
    //
    // Fourteen rows whose kernels were written into the .cu file each
    // indicator already ships in. Eleven are period-invariant and three
    // (epma, fosc, eri) read a CPU parameter literally named `period`.
    /// Source is hlcc4 = (h + l + 2c)/4 (data_loader.rs:171); the kernel builds it
    /// in-thread from the resident Hlc upload rather than asking for a fourth
    /// upload shape. The emitted column is `filt`, which the CPU stores in the
    /// field it calls `edf` (ehlers_detrending_filter.rs:400).
    F64KernelSpec {
        indicator_id: "ehlers_detrending_filter",
        kernel: F64Kernel::EhlersDetrendingFilter,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// CPU source is hl2 (:27), not close. Column `cycle` (cpu_batch.rs:10196).
    F64KernelSpec {
        indicator_id: "ehlers_simple_cycle_indicator",
        kernel: F64Kernel::EhlersSimpleCycleIndicator,
        input: F64InputKind::Hl2Slice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// CPU source is hl2 (:31), not close.
    F64KernelSpec {
        indicator_id: "ehlers_smoothed_adaptive_momentum",
        kernel: F64Kernel::EhlersSmoothedAdaptiveMomentum,
        input: F64InputKind::Hl2Slice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// The series cannot start before a RETURN can be formed: `valid_sq_return`
    /// (ewma_volatility.rs:274) needs two consecutive finite, strictly positive
    /// closes. `AllInputsNonNan` would name a bar at least one earlier and would
    /// accept a zero previous close the CPU skips. The kernel also derives its own
    /// seed index because the seed is the 32nd VALID return, not the 32nd bar.
    F64KernelSpec {
        indicator_id: "ewma_volatility",
        kernel: F64Kernel::EwmaVolatility,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::ConsecutiveValidReturnPair,
    },
    /// `compute_fdi_row` (:406) walks from index 0 and gates each window on a prefix
    /// count of non-finite bars, so the kernel reproduces the warmup rather than
    /// reading it.
    F64KernelSpec {
        indicator_id: "fractal_dimension_index",
        kernel: F64Kernel::FractalDimensionIndex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// `valid_high_low_bar` (:337) is `is_finite` on BOTH series -- the same rule
    /// aroonosc uses, and stricter than non-NaN: an infinite high is skipped by the
    /// CPU and would be accepted by `AllInputsNonNan`.
    F64KernelSpec {
        indicator_id: "gopalakrishnan_range_index",
        kernel: F64Kernel::GopalakrishnanRangeIndex,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::HighLowFinite,
    },
    /// `validity_summary` (:346) wants all FOUR prices finite AND strictly positive,
    /// which no rule here expresses, so the kernel derives its own start and this
    /// row declares the value unused. `open` is a real input: `gk_term` (:315)
    /// takes ln(close/open).
    F64KernelSpec {
        indicator_id: "garman_klass_volatility",
        kernel: F64Kernel::GarmanKlassVolatility,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    /// `impulse_macd_compute_into` (:527) walks from 0 and RESETS the whole cascade
    /// on an invalid bar, so the kernel reproduces the warmup. Column `impulse_macd`
    /// = `md` (cpu_batch.rs:13044).
    F64KernelSpec {
        indicator_id: "impulse_macd",
        kernel: F64Kernel::ImpulseMacd,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// CPU source default is close (cpu_batch.rs:12785). `valid_bar` (:281) also
    /// requires high >= low, and the row walk reproduces that gate itself. Column
    /// `average` (cpu_batch.rs:12765).
    F64KernelSpec {
        indicator_id: "hypertrend",
        kernel: F64Kernel::Hypertrend,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// The "value" column is `average` (cpu_batch.rs:14604), and `average` is
    /// `ma("sma", src, 28)` (emd_trend.rs:684-695) -- the SAME computation the `sma`
    /// row above performs, at a pinned length. The envelope and the direction state
    /// machine feed the other four outputs and never this one.
    F64KernelSpec {
        indicator_id: "emd_trend",
        kernel: F64Kernel::EmdTrend,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-SWEPT (`ma_batch.rs:1881` reads a parameter named `period`; `offset`
    /// defaults to 4). Warmup is `first + period + offset + 1` (epma.rs:1069), not
    /// `first + period - 1`.
    F64KernelSpec {
        indicator_id: "epma",
        kernel: F64Kernel::Epma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-SWEPT (cpu_batch.rs:3117, default 5). Warmup `first + period - 1`
    /// (fosc.rs:212).
    F64KernelSpec {
        indicator_id: "fosc",
        kernel: F64Kernel::Fosc,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-INVARIANT by construction of the dispatcher: `ma_batch.rs:1679`
    /// computes it once with default params and repeats `predict` for every row.
    /// Warmup `first + 13` (ehlers_pma.rs:320-321).
    F64KernelSpec {
        indicator_id: "ehlers_pma",
        kernel: F64Kernel::EhlersPma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-SWEPT (cpu_batch.rs:14727, default 13). The default ma_type is `ema`,
    /// so the CPU takes `eri_scalar_classic_ema` (:2176) and never the generic
    /// `eri_scalar`. Column `bull` = high - ema (cpu_batch.rs:14743); the CPU scans
    /// the high/low/source TRIPLE simultaneously for first-valid (:249).
    F64KernelSpec {
        indicator_id: "eri",
        kernel: F64Kernel::Eri,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ---------------------------------------------------------------- closer 1
    //
    // Twenty indicators that had a `.cu` file and NO lane entry point. Each one
    // now carries an `<id>_neo_batch_f64` written INTO THAT SAME FILE (search it
    // for "NEOETHOS f64 LANE"), beside the entry points the per-indicator
    // wrappers still call.
    //
    // "Just register the row" was NOT an option for any of them. The entry
    // points already in those files take several int arrays, emit several
    // outputs and never take `first_valid` -- for example
    // `absolute_strength_index_oscillator_batch_f64` takes `ema_lengths` and
    // `signal_lengths` and writes three matrices. This lane launches
    // (series..., n, periods, n_combos, first_valid, out). A row pointing at
    // one of the old symbols would have mismatched the ABI and read the stack.
    //
    // FIFTEEN OF THE TWENTY ARE PERIOD-INVARIANT, and that is faithful rather
    // than lazy: their CPU batch functions read NAMED parameters --
    // `ema_length`/`signal_length`, `rsi_length`/`alpha`, `short_length`/
    // `medium_length`/`long_length`, `atr_length`/`percentile_length`,
    // `short_roc_period`/`long_roc_period`/`ma_period` -- and never `period`.
    // A caller sweeping [7,21,50,100,200] gets five identical CPU columns, so
    // the kernel emits five identical rows and `is_period_invariant` says so.
    //
    // FIVE ARE GENUINELY PERIOD-SWEPT: bull_power_vs_bear_power, cg, dm,
    // donchian_channel_width and dpo each read a parameter literally named
    // `period`.
    // `..._row_field_from_slice` (:521) walks from index 0 with a fresh stream and
    // resets it on any non-finite bar; the prepare first index never reaches it.
    F64KernelSpec {
        indicator_id: "absolute_strength_index_oscillator",
        kernel: F64Kernel::AbsoluteStrengthIndexOscillator,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // `first_valid_ohlc` (:245) -- OPEN, high, low and close all `is_finite`.
    F64KernelSpec {
        indicator_id: "accumulation_swing_index",
        kernel: F64Kernel::AccumulationSwingIndex,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ohlc4AllFinite,
    },
    // Column is `in_phase`, the FIRST declared output: this indicator has no
    // "value" output (registry.rs:2523). The row walks from 0 and resets (:490).
    F64KernelSpec {
        indicator_id: "adaptive_bandpass_trigger_oscillator",
        kernel: F64Kernel::AdaptiveBandpassTriggerOscillator,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Column is `rsi` (cpu_batch.rs:12552). The row fills NaN and walks from 0 (:670).
    F64KernelSpec {
        indicator_id: "adaptive_bounds_rsi",
        kernel: F64Kernel::AdaptiveBoundsRsi,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Column is `macd` (cpu_batch.rs:4800). `compute_output_row` (:789) walks from 0.
    F64KernelSpec {
        indicator_id: "adaptive_macd",
        kernel: F64Kernel::AdaptiveMacd,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Column is `amo` (cpu_batch.rs:4206). `compute_output_into_slice` (:573) fills
    // NaN and walks from 0; nothing in the state machine is reset by a hole.
    F64KernelSpec {
        indicator_id: "adaptive_momentum_oscillator",
        kernel: F64Kernel::AdaptiveMomentumOscillator,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // `advance_decline_line_row` (:227) walks from 0 and RESTARTS the running sum
    // after every non-finite bar.
    F64KernelSpec {
        indicator_id: "advance_decline_line",
        kernel: F64Kernel::AdvanceDeclineLine,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Column is `bull`, the FIRST declared output (registry.rs:1457). The input is
    // (open, close), carried on the Ohlc4 shape; `first_valid_pair` (:244).
    F64KernelSpec {
        indicator_id: "andean_oscillator",
        kernel: F64Kernel::AndeanOscillator,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::OpenCloseFinite,
    },
    // `atr_percentile_into_slice` (:636) discards the prepare first index and the
    // row walks from 0 with per-bar validity flags rather than a warmup.
    F64KernelSpec {
        indicator_id: "atr_percentile",
        kernel: F64Kernel::AtrPercentile,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    // `bop_with_kernel` (:209) -- all four `!is_nan`, NOT `is_finite`.
    F64KernelSpec {
        indicator_id: "bop",
        kernel: F64Kernel::Bop,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ohlc4AllNonNan,
    },
    // PERIOD-SWEPT. `bbpower_row_from_ohlc` (:352) walks from 0 and resets both
    // `count` and `mean` on any invalid bar, where invalid also means close == 0.
    F64KernelSpec {
        indicator_id: "bull_power_vs_bear_power",
        kernel: F64Kernel::BullPowerVsBearPower,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    // PERIOD-SWEPT. The warmup is `first + period` (:231), one bar longer than the
    // `first + period - 1` most windowed indicators in this crate use.
    F64KernelSpec {
        indicator_id: "cg",
        kernel: F64Kernel::Cg,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // `coppock_with_kernel` (:322) -- `!is_nan` on the single source series. The
    // WMA stage then runs its OWN first-non-NaN scan over the ROC series
    // (wma.rs:259), which the kernel reproduces rather than assumes.
    F64KernelSpec {
        indicator_id: "coppock",
        kernel: F64Kernel::Coppock,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // Column is `value`, the range factor (cpu_batch.rs:10063). `first_valid_ohlc`
    // (:258) -- all four `is_finite`.
    F64KernelSpec {
        indicator_id: "daily_factor",
        kernel: F64Kernel::DailyFactor,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ohlc4AllFinite,
    },
    // The input is (advancing, declining) on the HighLow shape. The row (:329)
    // walks from 0 and resets the EMA seed and the 5-slot SMA ring on any
    // invalid pair.
    F64KernelSpec {
        indicator_id: "decisionpoint_breadth_swenlin_trading_oscillator",
        kernel: F64Kernel::DecisionpointBreadthSwenlinTradingOscillator,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Column is `short` (cpu_batch.rs:8481). `..._selected_row_from_slice` (:480)
    // walks from 0 and resets all three SMA windows on a hole.
    F64KernelSpec {
        indicator_id: "didi_index",
        kernel: F64Kernel::DidiIndex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // `compute_row` (:564) walks from 0; the entry points pass no first index at
    // all (`alloc_with_nan_prefix(len, 0)` then `fill(NAN)`, :603).
    F64KernelSpec {
        indicator_id: "disparity_index",
        kernel: F64Kernel::DisparityIndex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // PERIOD-SWEPT. Column is `plus` (cpu_batch.rs:6057). `dm_prepare` (:191) zips
    // high and low and takes the first index where BOTH are non-NaN.
    F64KernelSpec {
        indicator_id: "dm",
        kernel: F64Kernel::Dm,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // PERIOD-SWEPT. `compute_row` (:354) derives its own SEGMENT boundaries -- an
    // invalid pair restarts the window -- so one series-wide index cannot
    // express its warmup.
    F64KernelSpec {
        indicator_id: "donchian_channel_width",
        kernel: F64Kernel::DonchianChannelWidth,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::Ignored,
    },
    // PERIOD-SWEPT. The first written bar is `max(first + period - 1, period/2 + 1)`.
    F64KernelSpec {
        indicator_id: "dpo",
        kernel: F64Kernel::Dpo,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // ---------------------------------------------------------- closer 2b
    /// PERIOD-SWEPT (cpu_batch.rs:3811, default 30). Reads OPEN and CLOSE only --
    /// high and low are length-checked and discarded (cpu_batch.rs:3765-3777) -- and
    /// `batch_prepare` (:566) scans exactly that pair for first-valid, which is a
    /// different bar from the OHLC quadruple whenever high or low has the earlier
    /// hole. Served by Ohlc4 because the resident OHLCV upload already carries open.
    F64KernelSpec {
        indicator_id: "ehlers_fm_demodulator",
        kernel: F64Kernel::EhlersFmDemodulator,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::OpenCloseNonNan,
    },
    /// PERIOD-INVARIANT (cpu_batch.rs:15968/:15974 read `length` 20 and `smooth` 10).
    /// The EMA rate comes from SMOOTH, not from the window length. Column
    /// `forward_backward` (cpu_batch.rs:15993).
    F64KernelSpec {
        indicator_id: "forward_backward_exponential_oscillator",
        kernel: F64Kernel::ForwardBackwardExponentialOscillator,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-INVARIANT (cpu_batch.rs:7577-7580 read gmma_type / smooth_length /
    /// signal_length / anchor_minutes). Column `oscillator` (:7610). anchor_minutes
    /// defaults to 0, so `resolve_multiplier` (:404) returns 1 and the fan periods
    /// are the literal guppy tables.
    F64KernelSpec {
        indicator_id: "gmma_oscillator",
        kernel: F64Kernel::GmmaOscillator,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-INVARIANT (cpu_batch.rs:7229-7235 read atr_length / base_multiplier /
    /// noise_threshold / expansion_alpha). `open` never enters the arithmetic but it
    /// DOES gate validity (`is_valid_ohlc`, :354), so dropping it would carry trend
    /// state across a gap the CPU breaks. Column `band` (cpu_batch.rs:7254).
    F64KernelSpec {
        indicator_id: "evasive_supertrend",
        kernel: F64Kernel::EvasiveSupertrend,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ------------------------------------------------------ closer 6, round 2
    //
    // Eight indicators that had NO double-in/double-out entry point anywhere
    // in the crate -- their `.cu` files carried only f32 kernels. Each now
    // carries a "NEOETHOS f64 LANE  --  closer 6" section written against the
    // CPU reference named in that section's header, and each of those files is
    // listed in build.rs's fast-math opt-out.

    /// PERIOD-SWEPT (cpu_batch.rs:15582 reads a parameter literally named
    /// `period`, default 5). Column `sine` (:15594). The kernel reproduces
    /// BOTH CPU paths: `msw_period5_into` (msw.rs:764) forms its angles as
    /// exact multiples of `step` while `msw_scalar` (:261) ACCUMULATES them,
    /// and those are different doubles. TULIP_PI is 3.1415926, not M_PI
    /// (msw.rs:37).
    F64KernelSpec {
        indicator_id: "msw",
        kernel: F64Kernel::Msw,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-INVARIANT (cpu_batch.rs:8311-8313 read `lookback` 14,
    /// `k_override` false and `k` 0.34). Because `k_override` is FALSE the
    /// effective k is `k_default(lookback)` (yang_zhang_volatility.rs:403),
    /// NOT the 0.34 the parameter carries. Column `yz` (cpu_batch.rs:8331).
    ///
    /// `Ohlc4AllNonNan` rather than `Ohlc4AllFinite`: `first_valid_ohlc`
    /// (:411) tests `!is_nan`, so an INFINITE bar is accepted here and would
    /// be skipped under the stricter rule -- a different start index, and
    /// therefore a different seed window and a shifted series.
    F64KernelSpec {
        indicator_id: "yang_zhang_volatility",
        kernel: F64Kernel::YangZhangVolatility,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ohlc4AllNonNan,
    },
    /// PERIOD-INVARIANT (cpu_batch.rs:15880-15882 read `rsi_period` 14,
    /// `smoothing_factor` 5 and `fast_factor` 4.236). Column `fast` (:15896).
    /// Written against `qqe_scalar_classic` (qqe.rs:556); the crate's own
    /// length-dependent kernel selection (:505) routes series longer than
    /// 20_000 bars to a DIFFERENT accumulation, which is recorded as a crate
    /// defect in the .cu header rather than papered over.
    F64KernelSpec {
        indicator_id: "qqe",
        kernel: F64Kernel::Qqe,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-INVARIANT (cpu_batch.rs:6308-6311 read `rsi_period` 14,
    /// `stoch_period` 14, `k` 3 and `d` 3). Column `k` (:6329). The `FLT_MIN`
    /// guard the f32 lane carried is REMOVED, not re-sized: `srsi_scalar`
    /// (:511) has no epsilon at all, it tests `hi > lo`.
    F64KernelSpec {
        indicator_id: "srsi",
        kernel: F64Kernel::Srsi,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-SWEPT (cpu_batch.rs:16608 reads `period`, default 10). `ma_len`
    /// 14, `matype` 1 (EMA) and `devtype` 0 are pinned at the batch defaults.
    /// Single output (:16626). `rvi_scalar` walks from index 0, not from
    /// `first` -- `first` enters only through the warmup gate (rvi.rs:415).
    F64KernelSpec {
        indicator_id: "rvi",
        kernel: F64Kernel::Rvi,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-SWEPT (cpu_batch.rs:16704 reads `period`, default 14). Single
    /// output (:16717). The bar at `first + period - 1` is written as exactly
    /// 0.0, not NaN (net_myrsi.rs:312).
    F64KernelSpec {
        indicator_id: "net_myrsi",
        kernel: F64Kernel::NetMyrsi,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-INVARIANT (cpu_batch.rs:15734-15737 read `min_period` 5,
    /// `max_period` 50, `matype` "sma" and `devtype` 0). Column `value`
    /// (:15753). Written against `vlma_scalar_sma_stddev_into` (vlma.rs:451),
    /// which is the arm those defaults select -- NOT the generic
    /// `vlma_scalar_into` (:580), which calls out to `ma()`/`deviation()` and
    /// is unreachable at the defaults.
    F64KernelSpec {
        indicator_id: "vlma",
        kernel: F64Kernel::Vlma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    /// PERIOD-INVARIANT (cpu_batch.rs:16571-16574 read `fast_period` 23,
    /// `slow_period` 50, `k_period` 10 and `d_period` 3). Single output
    /// (:16591). `stc_scalar` picks between two implementations on a property
    /// of the DATA (stc.rs:491) -- all-finite or not -- and BOTH are
    /// transcribed, because they disagree on the MACD validity gate and on
    /// whether a hole blanks or carries the smoothing state.
    F64KernelSpec {
        indicator_id: "stc",
        kernel: F64Kernel::Stc,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // ------------------------------------------------------ closer 2, round 2
    //
    // Ten indicators that already shipped an f64 kernel in their own `.cu` file
    // and had NO row here, so `resolve_f64_kernel` answered
    // `CudaF64KernelMissing` for every one of them and the lane could not run
    // them at all. Each now has an `<id>_neo_batch_f64` entry point written into
    // that same file against the CPU reference the file's header names.
    //
    // Three declare a rule that is NOT `AllInputsNonNan`; each of those is
    // listed with its CPU file:line in the `DECLARED` table in
    // `first_valid_departures_are_declared`, which is what stops a rule from
    // being asserted here and never checked.

    /// `mwdx_scalar`, moving_averages/mwdx.rs:284. One fma per bar, seeded from
    /// the first non-NaN close (:308). PERIOD-INVARIANT: the only parameter is
    /// `factor`, default 0.2 (:119).
    F64KernelSpec {
        indicator_id: "mwdx",
        kernel: F64Kernel::Mwdx,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    /// `lrsi_scalar_hl`, lrsi.rs:397. Four carried Laguerre stages, each one
    /// `mul_add`. PERIOD-INVARIANT: the only parameter is `alpha`, default 0.2
    /// (cpu_batch.rs:3481).
    ///
    /// `HighLow` and not `Hlc`: the CPU reads high and low only and forms
    /// `(h + l) * 0.5` itself (:416), and its first-valid scan is over those
    /// two (:206-213). An Ohlc ref would adopt close's index.
    F64KernelSpec {
        indicator_id: "lrsi",
        kernel: F64Kernel::Lrsi,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    /// `pivot_scalar`, pivot.rs:536, mode-3 arm (:670-721). Emits `pp`, which
    /// is what the dispatcher returns for "value" (cpu_batch.rs:16743).
    /// PERIOD-INVARIANT: the only parameter is `mode`, default 3 (:16734) --
    /// an integer selecting WHICH formula runs, and a period list cannot stand
    /// in for it.
    ///
    /// `Hlc` and not `Ohlc4`: the batch extractor hands the CPU an `open`
    /// slice, but the mode-3 arm never reads it and the first-valid scan covers
    /// high, low and close only (:271-282).
    F64KernelSpec {
        indicator_id: "pivot",
        kernel: F64Kernel::Pivot,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    /// `kaufmanstop_scalar_classic_sma`, kaufmanstop.rs:2093 -- the NaN-aware
    /// form, which is bit-identical to the fast path (:2160) on NaN-free data
    /// and is the form the fast path RESTARTS INTO the moment it meets a NaN
    /// (:2179, :2208). PERIOD-SWEPT (cpu_batch.rs:15178); `mult` 2.0,
    /// `direction` "long", `ma_type` "sma" (:15179-15181).
    F64KernelSpec {
        indicator_id: "kaufmanstop",
        kernel: F64Kernel::Kaufmanstop,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    /// `sgf_compute_into`, moving_averages/sgf.rs:570, with `sgf_dot` (:479)
    /// and `build_endpoint_sgf_weights` (:331). PERIOD-SWEPT; `poly_order` is
    /// its default 2 (:87), so the kernel solves a 3x3 normal system per row
    /// rather than taking a host-solved weight vector.
    F64KernelSpec {
        indicator_id: "sgf",
        kernel: F64Kernel::Sgf,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    /// `polynomial_regression_extrapolation_scalar`,
    /// polynomial_regression_extrapolation.rs:542, with
    /// `build_forecast_weights_uncached` (:410). PERIOD-SWEPT via `length`
    /// (cpu_batch.rs:4764); `extrapolate` 10 and `degree` 3 (:4766, :4773), so
    /// the kernel solves a 4x4 normal system per row.
    F64KernelSpec {
        indicator_id: "polynomial_regression_extrapolation",
        kernel: F64Kernel::PolynomialRegressionExtrapolation,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    /// `compute_dual_ulcer_index_selected_row`, dual_ulcer_index.rs:566. Emits
    /// the LONG ULCER series, which is what the dispatcher returns for "value"
    /// (cpu_batch.rs:6700-6706). PERIOD-SWEPT (:6723).
    F64KernelSpec {
        indicator_id: "dual_ulcer_index",
        kernel: F64Kernel::DualUlcerIndex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `HullButterflyOscillatorStream::update`,
    /// hull_butterfly_oscillator.rs:404, driven from
    /// `..._selected_row_from_slice` (:517). Emits the OSCILLATOR series --
    /// this indicator's batch does NOT accept `output_id == "value"`
    /// (cpu_batch.rs:8751-8762), so a parity run must ask for "oscillator" by
    /// name. PERIOD-SWEPT via `length` (:8769); `mult` 2.0 (:8770).
    F64KernelSpec {
        indicator_id: "hull_butterfly_oscillator",
        kernel: F64Kernel::HullButterflyOscillator,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `compute_into_slices`, range_oscillator.rs:974 (general arm), with
    /// `AtrState::update` (:276) and `compute_weighted_ma` (:537). Emits the
    /// OSCILLATOR series, of which "value" is an alias
    /// (cpu_batch.rs:16044-16049). PERIOD-SWEPT via `length` (:16029); `mult`
    /// 2.0 (:16030).
    F64KernelSpec {
        indicator_id: "range_oscillator",
        kernel: F64Kernel::RangeOscillator,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `compute_run`, market_structure_trailing_stop.rs:486, driven by
    /// `compute_row` (:604). Emits the TRAILING STOP series, of which "value"
    /// is an alias (cpu_batch.rs:7197-7201). PERIOD-SWEPT via `length`
    /// (:7168); `increment_factor` 100.0 and `reset_on` "CHoCH"
    /// (:7169-7180).
    ///
    /// `Ohlc4` because the run segmentation is `is_valid_ohlc` (:279-281),
    /// which reads OPEN. The value loop itself reads high, low and close, but
    /// an Hlc ref would put the GPU and the CPU on different runs on any frame
    /// whose open has a hole the other three do not.
    F64KernelSpec {
        indicator_id: "market_structure_trailing_stop",
        kernel: F64Kernel::MarketStructureTrailingStop,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },


    // -------------------------------------------------- closer 4, round 2
    //
    // Fifteen indicators that had a `.cu` file and, for four of them, a
    // real all-double kernel in it -- but NO entry point with this lane's
    // ABI and no row here, so `resolve_f64_kernel` returned
    // CudaF64KernelMissing and the card never ran them. Each now has a
    // `<id>_neo_batch_f64` written into the file its indicator already
    // ships in, against the CPU reference named in that file's header.
    //
    // ELEVEN DECLARE `Ignored`, and that is a claim about the KERNEL, not
    // a shrug. Their CPU references either never compute a first-valid
    // index at all (rogers_satchell_volatility counts valid bars but
    // never locates the first; price_density_market_noise's
    // `with_kernel` is literally `let _ = first;`), or compute one with a
    // rule no variant here expresses -- `high >= low` as well as
    // finiteness for pretty_good_oscillator and
    // keltner_channel_width_oscillator, `finite AND strictly positive` on
    // three series for kase_peak_oscillator_with_divergences -- or reset
    // their whole accumulator on every non-finite bar and consult no
    // index. In each case the kernel derives what it needs itself and
    // starts the row all-NaN, exactly as `garman_klass_volatility`
    // already does in this lane. Declaring AllInputsNonNan instead would
    // be a claim the kernel does not honour.
    //
    // `kst` is the one AllInputsNonNan (kst.rs:359 scans !is_nan on the
    // single close series); `qqe_weighted_oscillator` and
    // `smooth_theil_sen` are CloseFinite (is_finite, not !is_nan), and
    // for both the index is LOAD-BEARING -- qqe seeds prev_src from
    // data[first] and starts at first + 1, smooth_theil_sen hangs its
    // whole warmup off first + length + offset - 1.
    //
    // `rogers_satchell_volatility` is registered Ohlc4 because its term
    // reads OPEN as well as high, low and close. `market_meanness_index`
    // is registered CloseSlice and NOT Ohlc4 for the opposite reason: its
    // CPU default source_mode is "Price", under which both is_valid_bar
    // and source_value read CLOSE ALONE and open is never dereferenced.
    F64KernelSpec {
        indicator_id: "kase_peak_oscillator_with_divergences",
        kernel: F64Kernel::KasePeakOscillatorWithDivergences,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "keltner_channel_width_oscillator",
        kernel: F64Kernel::KeltnerChannelWidthOscillator,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "kst",
        kernel: F64Kernel::Kst,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    F64KernelSpec {
        indicator_id: "leavitt_convolution_acceleration",
        kernel: F64Kernel::LeavittConvolutionAcceleration,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "market_meanness_index",
        kernel: F64Kernel::MarketMeannessIndex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "market_structure_confluence",
        kernel: F64Kernel::MarketStructureConfluence,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "monotonicity_index",
        kernel: F64Kernel::MonotonicityIndex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "premier_rsi_oscillator",
        kernel: F64Kernel::PremierRsiOscillator,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "pretty_good_oscillator",
        kernel: F64Kernel::PrettyGoodOscillator,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "price_density_market_noise",
        kernel: F64Kernel::PriceDensityMarketNoise,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "projection_oscillator",
        kernel: F64Kernel::ProjectionOscillator,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "qqe_weighted_oscillator",
        kernel: F64Kernel::QqeWeightedOscillator,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::CloseFinite,
    },
    F64KernelSpec {
        indicator_id: "rogers_satchell_volatility",
        kernel: F64Kernel::RogersSatchellVolatility,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "rolling_skewness_kurtosis",
        kernel: F64Kernel::RollingSkewnessKurtosis,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "smooth_theil_sen",
        kernel: F64Kernel::SmoothTheilSen,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::CloseFinite,
    },
    // ------------------------------------------------------ closer 3, round 2
    //
    // Twenty-five rows. Every kernel behind them was written INTO the .cu file
    // its indicator already ships in, against the CPU reference named in that
    // file's "NEOETHOS f64 LANE  --  closer 3" header, and each emits the
    // column the CPU batch produces for `output_id == "value"` -- except the
    // two named below, whose batch has no `value` alias at all.
    //
    // WHY EVERY ONE OF THEM DECLARES `Ignored`, AND WHY THAT IS A CONTRACT
    // RATHER THAN A SHRUG. `first_valid` is not a tolerance-sized detail: it
    // sets both the NaN prefix and the seed window, so a wrong rule SHIFTS the
    // whole series. These twenty-five CPU references fall into two groups, and
    // NEITHER can be served by a caller-supplied index:
    //
    //   * Most have NO warmup index at all. They emit from bar 0 and RESET
    //     their whole state -- ATR seed, ring positions, EMA, deque, trend
    //     state machine -- at every non-finite bar. A global first-valid is
    //     correct only until the first hole; after that the CPU is counting
    //     from the hole and the caller's index names a bar the CPU has already
    //     left behind. `exponential_trend`, `fvg_positioning_average`,
    //     `grover_llorens_cycle_oscillator` and
    //     `ehlers_autocorrelation_periodogram` are the clearest.
    //   * The rest scan with a predicate no declared rule expresses:
    //     `is_finite` over a high/low/close TRIPLE
    //     (adjustable_ma_alternating_extremities.rs:600 -- stricter than
    //     `AllInputsNonNan`, which accepts an INFINITE high the CPU skips),
    //     `is_finite` over a DERIVED midpoint series that is not any input
    //     series (ehlers_data_sampling_relative_strength_indicator), or
    //     "finite AND strictly positive" decided window by window because the
    //     return is `ln(curr/prev)` (both historical_volatility indicators).
    //
    // So each kernel does the scan itself, which keeps the two halves of one
    // rule in one place. This is the same contract
    // `garman_klass_volatility_neo_batch_f64` already carries, and the reason
    // `F64FirstValidRule::Ignored` exists.
    //
    // `vertical_horizontal_filter` is the one row that adds NO kernel: the
    // entry point already in `vertical_horizontal_filter_kernel.cu` carries
    // the lane ABI exactly, so this row is registration only.
    F64KernelSpec {
        indicator_id: "vertical_horizontal_filter",
        kernel: F64Kernel::VerticalHorizontalFilter,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::CloseFinite,
    },
    F64KernelSpec {
        indicator_id: "adjustable_ma_alternating_extremities",
        kernel: F64Kernel::AdjustableMaAlternatingExtremities,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "autocorrelation_indicator",
        kernel: F64Kernel::AutocorrelationIndicator,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "historical_volatility_rank",
        kernel: F64Kernel::HistoricalVolatilityRank,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Emits `hvp`. `compute_historical_volatility_percentile_batch`
    // (cpu_batch.rs:9681-9690) accepts only `hvp` and `hvp_sma` and returns
    // `UnknownOutput` for `value`, so a parity run must name the column.
    F64KernelSpec {
        indicator_id: "historical_volatility_percentile",
        kernel: F64Kernel::HistoricalVolatilityPercentile,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "directional_imbalance_index",
        kernel: F64Kernel::DirectionalImbalanceIndex,
        input: F64InputKind::HighLow,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "cycle_channel_oscillator",
        kernel: F64Kernel::CycleChannelOscillator,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "dynamic_momentum_index",
        kernel: F64Kernel::DynamicMomentumIndex,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // `hl2`, not close: `extract_slice_input("ehlers_adaptive_cg", req.data,
    // "hl2")` (cpu_batch.rs:15793). Handing either of these two kernels close
    // would compute a different indicator and pass every length check.
    F64KernelSpec {
        indicator_id: "ehlers_adaptive_cg",
        kernel: F64Kernel::EhlersAdaptiveCg,
        input: F64InputKind::Hl2Slice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "ehlers_adaptive_cyber_cycle",
        kernel: F64Kernel::EhlersAdaptiveCyberCycle,
        input: F64InputKind::Hl2Slice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Emits `ds_rsi` -- the RSI of the bar MIDPOINT, not of close. The CPU
    // batch (cpu_batch.rs:8132-8144) has no `value` alias. `Ohlc4` because
    // only OPEN and CLOSE are read and the four-pointer launch arm already
    // exists -- the same reason `qstick` is declared `Ohlc4`.
    F64KernelSpec {
        indicator_id: "ehlers_data_sampling_relative_strength_indicator",
        kernel: F64Kernel::EhlersDataSamplingRelativeStrengthIndicator,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "exponential_trend",
        kernel: F64Kernel::ExponentialTrend,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "geometric_bias_oscillator",
        kernel: F64Kernel::GeometricBiasOscillator,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Only OPEN and CLOSE are read -- `extract_ohlc_full_input` destructures
    // high and low as `_high` and `_low` (cpu_batch.rs:13393).
    F64KernelSpec {
        indicator_id: "intraday_momentum_index",
        kernel: F64Kernel::IntradayMomentumIndex,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "bulls_v_bears",
        kernel: F64Kernel::BullsVBears,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "candle_strength_oscillator",
        kernel: F64Kernel::CandleStrengthOscillator,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "cyberpunk_value_trend_analyzer",
        kernel: F64Kernel::CyberpunkValueTrendAnalyzer,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "fvg_positioning_average",
        kernel: F64Kernel::FvgPositioningAverage,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Ohlc4 rather than CloseSlice because the CPU's validity gate -- the
    // reset trigger for the whole EMA cascade -- reads all four series, so
    // open, high and low are inputs to fast_hema's warmup even though its
    // arithmetic reads close alone.
    F64KernelSpec {
        indicator_id: "hema_trend_levels",
        kernel: F64Kernel::HemaTrendLevels,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "fibonacci_trailing_stop",
        kernel: F64Kernel::FibonacciTrailingStop,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "grover_llorens_cycle_oscillator",
        kernel: F64Kernel::GroverLlorensCycleOscillator,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "demand_index",
        kernel: F64Kernel::DemandIndex,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "adaptive_schaff_trend_cycle",
        kernel: F64Kernel::AdaptiveSchaffTrendCycle,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "ehlers_linear_extrapolation_predictor",
        kernel: F64Kernel::EhlersLinearExtrapolationPredictor,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "ehlers_autocorrelation_periodogram",
        kernel: F64Kernel::EhlersAutocorrelationPeriodogram,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Emits `bullish_high`. The CPU batch (cpu_batch.rs:12880-12905) accepts
    // twelve output ids and has NO `value` alias, so a parity run must name
    // the column. `Ohlc4` because the state machine reads all four series and
    // its validity gate is `finite(o,h,l,c) && h >= l`.
    F64KernelSpec {
        indicator_id: "ict_propulsion_block",
        kernel: F64Kernel::IctPropulsionBlock,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },

    // ------------------------------------------------------ closer 6, round 3
    //
    // Six indicators that had NO CUDA presence at all before this round: no
    // `.cu`, no wrapper, no row here. Each now has a from-scratch f64 kernel in
    // its own translation unit under `kernels/cuda/moving_averages/`, written
    // against the CPU reference named in the kernel header.

    // (hlcc4, volume). `Hlcc4Volume` because the declared default source is
    // `hlcc4` (:113) and volume is a second input, not metadata.
    // `PriceVolumeFinite` because `find_first_valid` (:308-317) scans BOTH with
    // `is_finite`. The kernel takes the `use_volume_sum == true` branch
    // (:382), which is the branch the period-sweeping route selects --
    // ma.rs:1105-1113 and registry.rs:608 -- and it is the only branch that
    // reads `length` at all.
    F64KernelSpec {
        indicator_id: "elastic_volume_weighted_moving_average",
        kernel: F64Kernel::ElasticVolumeWeightedMovingAverage,
        input: F64InputKind::Hlcc4Volume,
        first_valid: F64FirstValidRule::PriceVolumeFinite,
    },
    // Emits the PRIMARY output, the corrected line (registry.rs:537).
    // `Ignored` because `compute_into_slices` (:353) walks from index 0 and
    // RESETS the whole cascade on every non-finite bar -- there is no single
    // warmup prefix for a first-valid index to name.
    F64KernelSpec {
        indicator_id: "ema_deviation_corrected_t3",
        kernel: F64Kernel::EmaDeviationCorrectedT3,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Emits the PRIMARY output, `lma` (:1165). `Ignored` for the same reason:
    // `compute_lma` (:775) walks from index 0 behind a `run` counter that a
    // non-finite bar resets, and `out_lma` is NaN-filled first (:1156).
    // Bounded by `LMA_MAX_PERIOD`; a larger period is refused BY NAME.
    F64KernelSpec {
        indicator_id: "logarithmic_moving_average",
        kernel: F64Kernel::LogarithmicMovingAverage,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // `Ignored`: `n_order_ema_compute_into` (:781) walks the whole series and
    // `IirCoreFilter::update` (:374) resets on a non-finite bar.
    F64KernelSpec {
        indicator_id: "n_order_ema",
        kernel: F64Kernel::NOrderEma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // `AllInputsNonNan` over one close series: `vama_prepare` (:716) is
    // `position(|x| !x.is_nan())`, the common rule exactly.
    F64KernelSpec {
        indicator_id: "volatility_adjusted_ma",
        kernel: F64Kernel::VolatilityAdjustedMa,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },
    // `CloseFinite`, NOT `AllInputsNonNan`: `prepare_input` (:242) scans with
    // `is_finite`, so an INFINITE bar is skipped by the CPU and would be
    // accepted by the non-NaN scan. Bounded by `WS_MAX_PERIOD`.
    F64KernelSpec {
        indicator_id: "wave_smoother",
        kernel: F64Kernel::WaveSmoother,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::CloseFinite,
    },
    // ---------------------------------------------------- closer 3, round 3
    //
    // Ten rows. Every one names a `*_neo_batch_f64` entry point written into
    // the indicator's OWN .cu file against the CPU reference quoted in that
    // file's header, and every one emits the column the CPU batch produces for
    // `output_id == "value"`.
    //
    // NINE DECLARE `Ignored` AND ONE DECLARES `HlcCloseOnly`, and the split is
    // read from the CPU rather than chosen. Eight of the nine walk every bar
    // from index 0 and RESET their whole state at an invalid bar, so a global
    // warmup index would name the wrong seed after the first hole. The ninth,
    // `avsl`, scans with a rule no variant here expresses: `first_valid_max3`
    // (avsl.rs:272) is the MAX of THREE INDEPENDENT first-non-NaN scans over
    // close, low and volume -- NOT "the first index at which all three are
    // non-NaN", which is later whenever one series has a hole after another
    // has started. Rather than add a variant one indicator would use, that
    // kernel derives the index and declares the caller's value unused.
    // `alphatrend` is the exception in the other direction: it scans
    // `close.iter().position(|x| !x.is_nan())` (alphatrend.rs:493) and never
    // looks at high, low or volume, which is exactly the rule `adxr` declares.
    F64KernelSpec {
        indicator_id: "reversal_signals",
        kernel: F64Kernel::ReversalSignals,
        input: F64InputKind::Ohlcv5,
        first_valid: F64FirstValidRule::Ignored,
    },
    // `Hlcv`, not `Ohlcv5`: `trend_follower_prepare` binds open nowhere and
    // `first_valid_bar` (:678) scans high, low and close only.
    F64KernelSpec {
        indicator_id: "trend_follower",
        kernel: F64Kernel::TrendFollower,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "vdubus_divergence_wave_pattern_generator",
        kernel: F64Kernel::VdubusDivergenceWavePatternGenerator,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "volatility_ratio_adaptive_rsx",
        kernel: F64Kernel::VolatilityRatioAdaptiveRsx,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "volume_energy_reservoirs",
        kernel: F64Kernel::VolumeEnergyReservoirs,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "volume_weighted_relative_strength_index",
        kernel: F64Kernel::VolumeWeightedRelativeStrengthIndex,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "volume_weighted_stochastic_rsi",
        kernel: F64Kernel::VolumeWeightedStochasticRsi,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::Ignored,
    },
    // `Ohlc4`, not `Hlc`: `is_valid_ohlc` (:240) tests OPEN, so a bar with a
    // non-finite open BREAKS the run and the two segments either side are
    // computed independently. A three-pointer shape would never see it.
    F64KernelSpec {
        indicator_id: "zig_zag_channels",
        kernel: F64Kernel::ZigZagChannels,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    // `HlcCloseOnly`: alphatrend_prepare (:493) scans CLOSE ALONE. Adopting
    // the Hlc triple's index would shift the whole series on any frame where
    // high or low starts later than close, and `first` sets BOTH the NaN
    // prefix and the true-range seed window.
    F64KernelSpec {
        indicator_id: "alphatrend",
        kernel: F64Kernel::Alphatrend,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::HlcCloseOnly,
    },
    // `Hlcv` with HIGH bound and unread: the CPU batch calls
    // `extract_hlcv_input` and discards high (cpu_batch.rs:14123), and
    // `avsl_scalar` reads close, low and volume only.
    F64KernelSpec {
        indicator_id: "avsl",
        kernel: F64Kernel::Avsl,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::Ignored,
    },

    // ------------------------------------------------------ closer 4, round 3
    //
    // Ten rows. Every kernel behind them was written INTO the `.cu` file its
    // indicator already ships in, beside the f32 entry points the f32 wrappers
    // still call, against the CPU reference named in that file's
    // "NEOETHOS f64 LANE  --  closer 4, round 3" header.
    //
    // ALL TEN FILES WERE PURE f32 BEFORE THIS CHANGE. `bandpass_kernel.cu` had
    // two `__global__`s and both took `const float*`; `dma_kernel.cu` had
    // seven; `buff_averages_kernel.cu` twelve; `halftrend_kernel.cu` did not
    // contain the token `double` at all. There was no f64 symbol for a row to
    // point at, so the lane could not reach these ten indicators and answered
    // `CudaF64KernelMissing` for every one of them.

    /// `bandpass.rs:303` -- the `bp` column, which is what `value` resolves to
    /// (cpu_batch.rs:14152). `CloseFinite` because `bandpass_prepare:255`
    /// scans with `is_finite`, not `!is_nan`: an infinite bar is SKIPPED by
    /// the CPU and would be accepted by `AllInputsNonNan`.
    F64KernelSpec {
        indicator_id: "bandpass",
        kernel: F64Kernel::Bandpass,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::CloseFinite,
    },

    /// `buff_averages.rs:599` -- the FAST buff, the `output` default in
    /// `ma_batch.rs:629`. `Ignored` and derived in the kernel: `buff_averages_
    /// prepare:470` scans PRICE ALONE, and under `AllInputsNonNan` a
    /// `CloseVolume` shape resolves to a scan over BOTH series, which names a
    /// later bar on any frame whose volume has a hole -- and first-valid sets
    /// the NaN prefix AND the seed window, so that is a shifted series, not an
    /// ULP.
    F64KernelSpec {
        indicator_id: "buff_averages",
        kernel: F64Kernel::BuffAverages,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `cora_wave.rs:246`. `AllInputsNonNan` is exact here: `cora_wave_
    /// prepare:325` is `position(|x| !x.is_nan())` over the single close
    /// series.
    F64KernelSpec {
        indicator_id: "cora_wave",
        kernel: F64Kernel::CoraWave,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    /// `dma.rs:296`. `dma_prepare:395` is `!is_nan` over close.
    F64KernelSpec {
        indicator_id: "dma",
        kernel: F64Kernel::Dma,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    /// `fvg_trailing_stop.rs:1035` -- the UPPER band (cpu_batch.rs:14884).
    /// `Ignored` because the batch takes `fvg_trailing_stop_with_kernel`
    /// (:1040), which allocates with `alloc_uninit_f64` and applies NO warmup
    /// prefix at all: the loop runs from bar 0 and writes every bar.
    F64KernelSpec {
        indicator_id: "fvg_trailing_stop",
        kernel: F64Kernel::FvgTrailingStop,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `halftrend.rs:298` -- the `halftrend` column (cpu_batch.rs:14979).
    /// `Ignored` and derived in the kernel: `first_valid_ohlc` (:291) takes
    /// the MIN of three INDEPENDENT scans, which no declared rule expresses --
    /// `HlcMaxOfIndependentFirsts` is the MAX and names a LATER bar, and the
    /// index sets both the NaN prefix and the ATR and SMA seed windows.
    F64KernelSpec {
        indicator_id: "halftrend",
        kernel: F64Kernel::Halftrend,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `mod_god_mode.rs:555` -- the WAVETREND column (cpu_batch.rs:15556).
    /// `Hlcv` because the batch default is `use_volume = true`
    /// (cpu_batch.rs:15521) and the money-flow term reads `volume[i]`.
    /// `Ignored` and derived in the kernel: `mod_god_mode_into_slices:693`
    /// scans CLOSE ALONE, while `AllInputsNonNan` over an `Hlcv` shape would
    /// scan all four and name a later bar.
    F64KernelSpec {
        indicator_id: "mod_god_mode",
        kernel: F64Kernel::ModGodMode,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `ott.rs:275`. `ott_prepare:349` is `!is_nan` over close, and the VAR
    /// moving average rescans the same series to the same index, so the NaN
    /// prefix is that index.
    F64KernelSpec {
        indicator_id: "ott",
        kernel: F64Kernel::Ott,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    /// `otto.rs:1599` -- the HOTT column (cpu_batch.rs:15680). `Ignored`
    /// because `otto_with_kernel:1605` allocates with
    /// `alloc_with_nan_prefix(len, 0)` -- there is NO warmup prefix, both
    /// passes walk from bar 0, and a first-valid index would name a bar the
    /// CPU never skips.
    F64KernelSpec {
        indicator_id: "otto",
        kernel: F64Kernel::Otto,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `prb.rs:938` -- the `values` column (cpu_batch.rs:15857).
    /// `prb_with_kernel:1385` is `!is_nan` over close.
    F64KernelSpec {
        indicator_id: "prb",
        kernel: F64Kernel::Prb,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::AllInputsNonNan,
    },

    // ------------------------------------------------------ closer 2, round 3
    //
    // Ten indicators whose `.cu` file ALREADY held a real double-in/double-out
    // kernel that this lane could not call, because every one of those entry
    // points is MULTI-OUTPUT with a bespoke parameter list -- 21, 14, 10, 16,
    // 28, 19, 22, 25, 11 and 15 parameters, writing between two and thirteen
    // output matrices each, three of them also demanding caller-allocated
    // scratch matrices and three more calling `new double[]` on the DEVICE.
    // The lane launches one shape and allocates ONE output matrix, so each file
    // now carries a lane-shaped `<id>_neo_batch_f64` twin written against the
    // CPU reference named in that file's
    // "NEOETHOS f64 LANE  --  closer 2, round 3" header.
    //
    // EVERY ONE DECLARES `Ignored`, for the same reason in all ten: the CPU row
    // function writes EVERY index of the emitted column -- NaN wherever its
    // state machine is not ready -- so whatever prefix `alloc_with_nan_prefix`
    // laid down is overwritten wholesale and there is no start index for the
    // two sides to disagree about. Each kernel derives its own readiness from
    // its own counters, exactly as the CPU does. Two of them go further and
    // derive a start index that NONE of the rules in `F64FirstValidRule` can
    // express: `possible_rsi` and `price_moving_average_ratio_percentile` reach
    // into `rsi.rs:284` / `sma.rs:274`, both of which scan with
    // `position(|x| !x.is_nan())` -- which ACCEPTS an infinity. Declaring
    // `AllInputsNonNan` for those two would be a claim the kernels do not
    // honour.
    //
    // THREE INPUT KINDS ARE NARROWER THAN THEY LOOK, each read out of the CPU
    // rather than assumed:
    //
    //  * `normalized_resonator` is `Hl2Slice`, not `CloseSlice`. Its
    //    DEFAULT_SOURCE is "hl2" (normalized_resonator.rs:37) and the batch's
    //    `get_enum_param` default is "hl2". Handing it close computes a
    //    different indicator and passes every length check on the way through.
    //  * `normalized_volume_true_range` and `range_breakout_signals` are
    //    `Ohlcv5`, not `Hlcv`: both read OPEN at every bar -- the first because
    //    its default Body style measures `close - open`
    //    (normalized_volume_true_range.rs:511), the second because both the
    //    bar's body and its signed-volume split compare `close` against `open`
    //    (range_breakout_signals.rs:895, :1000) -- so a four-pointer shape
    //    would drop the series they are built on.
    //  * `relative_strength_index_wave_indicator` is `Hlc` because it runs
    //    THREE independent Wilder RSIs -- on the source, on high and on low
    //    (:601-603) -- and the third pointer is the SOURCE, which defaults to
    //    close.
    //
    // TWO OF THE TEN HAVE NO `value` OUTPUT ON THE CPU AT ALL, so the column
    // each kernel emits is named here rather than left to be discovered:
    // `normalized_resonator`'s batch accepts only "oscillator" and "signal", so
    // the kernel emits oscillator; `range_filtered_trend_signals`'s batch
    // REJECTS "value" and accepts thirteen named columns, so the kernel emits
    // kalman -- the first arm of its own CPU match and the filtered price the
    // indicator is named after. The other eight emit what `output_id ==
    // "value"` resolves to: trailing_stop, value, normalized_volume, value,
    // plotline, range_top, value and rsi_ma1.

    /// `neighboring_trailing_stop.rs:858` -- the `trailing_stop` column
    /// (cpu_batch.rs:9043, `"trailing_stop" | "value"`).
    F64KernelSpec {
        indicator_id: "neighboring_trailing_stop",
        kernel: F64Kernel::NeighboringTrailingStop,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `nonlinear_regression_zero_lag_moving_average.rs:729` -- the `value`
    /// column (cpu_batch.rs:7666).
    F64KernelSpec {
        indicator_id: "nonlinear_regression_zero_lag_moving_average",
        kernel: F64Kernel::NonlinearRegressionZeroLagMovingAverage,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `normalized_resonator.rs:731` -- the `oscillator` column. Source hl2.
    F64KernelSpec {
        indicator_id: "normalized_resonator",
        kernel: F64Kernel::NormalizedResonator,
        input: F64InputKind::Hl2Slice,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `normalized_volume_true_range.rs:791` -- the `normalized_volume` column
    /// (`"normalized_volume" || "value"`). Open is an input.
    F64KernelSpec {
        indicator_id: "normalized_volume_true_range",
        kernel: F64Kernel::NormalizedVolumeTrueRange,
        input: F64InputKind::Ohlcv5,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `possible_rsi.rs:1359` -- the `value` column. The ONLY row of this batch
    /// that is period-SWEPT: its CPU batch reads a parameter literally named
    /// `period` (default 32) and it is the RSI length.
    F64KernelSpec {
        indicator_id: "possible_rsi",
        kernel: F64Kernel::PossibleRsi,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `price_moving_average_ratio_percentile.rs:715` -- the `plotline` column
    /// (`"plotline" || "value"`), which at the CPU default `line_mode = "pmar"`
    /// is `pmar` itself (:707-710). Volume is bound and unread at ma_type
    /// "sma", and named so the launch cannot pass one series where the kernel
    /// asked for two.
    F64KernelSpec {
        indicator_id: "price_moving_average_ratio_percentile",
        kernel: F64Kernel::PriceMovingAverageRatioPercentile,
        input: F64InputKind::CloseVolume,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `range_breakout_signals.rs:1381` -- the `range_top` column
    /// (`"range_top" || "value"`). Open and volume are inputs.
    F64KernelSpec {
        indicator_id: "range_breakout_signals",
        kernel: F64Kernel::RangeBreakoutSignals,
        input: F64InputKind::Ohlcv5,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `range_filtered_trend_signals.rs:744` -- the `kalman` column. Its CPU
    /// batch REJECTS "value".
    F64KernelSpec {
        indicator_id: "range_filtered_trend_signals",
        kernel: F64Kernel::RangeFilteredTrendSignals,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `regression_slope_oscillator.rs:564` -- the `value` column.
    F64KernelSpec {
        indicator_id: "regression_slope_oscillator",
        kernel: F64Kernel::RegressionSlopeOscillator,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },

    /// `relative_strength_index_wave_indicator.rs:708` -- the `rsi_ma1` column
    /// (`"rsi_ma1" || "value"`). The third pointer is the SOURCE, default close.
    F64KernelSpec {
        indicator_id: "relative_strength_index_wave_indicator",
        kernel: F64Kernel::RelativeStrengthIndexWaveIndicator,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    // ------------------------------------------------ closer 1, round 3
    //
    // Ten indicators whose `.cu` file already shipped a genuine
    // double-in/double-out kernel that this table could not name, because
    // every one of those entry points is MULTI-OUTPUT with a bespoke
    // parameter list rather than the lane ABI. Each file now carries a
    // `*_neo_batch_f64` entry point beside what it had, written against the
    // CPU reference named in that file's `NEOETHOS f64 LANE  --  closer 1,
    // round 3` header. All ten are PERIOD-INVARIANT, so `first_valid` is
    // `Ignored` and every swept period gives the same column.
    // Emits `basis`. The CPU batch (cpu_batch.rs:8850-8946) accepts eighteen
    // output ids and has NO `value` alias, so a parity run must name the column;
    // `basis` is aliased `middle` there and is the series every band is offset
    // from. `Ohlc4` because `valid_bar` reads all four and the source is hlc3.
    F64KernelSpec {
        indicator_id: "fibonacci_entry_bands",
        kernel: F64Kernel::FibonacciEntryBands,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    // The ONLY bar-parallel row of this batch. `compute_row`
    // (goertzel_cycle_composite_wave.rs:886-901) recomputes each 601-bar window
    // independently and carries NOTHING between bars, so the recurrence is
    // inside the window and the kernel is launched over (combo, bar).
    F64KernelSpec {
        indicator_id: "goertzel_cycle_composite_wave",
        kernel: F64Kernel::GoertzelCycleCompositeWave,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Emits `estimate`. `TimestampCloseVolume` because the CPU door that works
    // at default parameters is the Candles one: it INFERS `slots_per_day` from
    // the bar timestamps (half_causal_estimator.rs:1319) and takes VOLUME as the
    // source. The Slice door needs an explicit `slots_per_day` and the batch
    // passes None (cpu_batch.rs:9573), which is `MissingSlotsPerDay`.
    F64KernelSpec {
        indicator_id: "half_causal_estimator",
        kernel: F64Kernel::HalfCausalEstimator,
        input: F64InputKind::TimestampCloseVolume,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Emits `signal`, which cpu_batch.rs:10906 also aliases as `value`.
    F64KernelSpec {
        indicator_id: "ichimoku_oscillator",
        kernel: F64Kernel::IchimokuOscillator,
        input: F64InputKind::Hlc,
        first_valid: F64FirstValidRule::Ignored,
    },
    // `insync_index` has NO `compute_*_batch` arm in cpu_batch.rs at all; the
    // CPU reference is the scalar `insync_index_with_kernel`, one `values`
    // vector. `Hlcv` -- the validity gate reads volume and requires it > 0.
    F64KernelSpec {
        indicator_id: "insync_index",
        kernel: F64Kernel::InsyncIndex,
        input: F64InputKind::Hlcv,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "linear_regression_intensity",
        kernel: F64Kernel::LinearRegressionIntensity,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Emits `diff`, which cpu_batch.rs:13600 also aliases as `value`. `Ohlc4`
    // because the validity gate is four-way finite even though only `close`
    // reaches this column.
    F64KernelSpec {
        indicator_id: "macd_wave_signal_pro",
        kernel: F64Kernel::MacdWaveSignalPro,
        input: F64InputKind::Ohlc4,
        first_valid: F64FirstValidRule::Ignored,
    },
    // Emits `mesa_1`. cpu_batch.rs:10604-10629 accepts eight output ids --
    // mesa_1..4 and trigger_1..4 -- and has NO `value` alias, so a parity run
    // must name the column; mesa_1 is the longest line (length_1 = 48).
    F64KernelSpec {
        indicator_id: "mesa_stochastic_multi_length",
        kernel: F64Kernel::MesaStochasticMultiLength,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "moving_average_cross_probability",
        kernel: F64Kernel::MovingAverageCrossProbability,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
    F64KernelSpec {
        indicator_id: "multi_length_stochastic_average",
        kernel: F64Kernel::MultiLengthStochasticAverage,
        input: F64InputKind::CloseSlice,
        first_valid: F64FirstValidRule::Ignored,
    },
];


/// Kernels that are WRITTEN, COMPILED and deliberately NOT REGISTERED.
///
/// # Why a written kernel would be withheld
///
/// Every kernel in this lane is written against a named `*_scalar` CPU
/// reference. `hpc_ta` calls the CPU with `Kernel::Auto`, and `neoethos-data`
/// enables vector-ta's `nightly-avx`, so on x86_64 `Auto` resolves to `Avx2` or
/// `Avx512` for many indicators. For two of them the crate's OWN scalar and AVX
/// implementations do not agree, so there is no single CPU answer for a device
/// result to be in parity WITH — and shipping either one would mean the lane
/// silently picks a side.
///
/// `crates/neoethos-data/tests/f64_lane_cpu_reference.rs` measures this without
/// a card. On 3000 synthetic M5 bars it reports:
///
/// * `vwap` — differs at index 136 for every period, 1 ULP
///   (`0x3ff27b0011b24696` scalar vs `0x3ff27b0011b24695` auto). `vwap_scalar`
///   accumulates `volume_sum` and `vol_price_sum` bar by bar; `vwap_avx2`
///   reassociates them.
/// * `wilders` — differs at exactly the SEED bar (index 13 at period 14, index
///   99 at period 100), 1 ULP. `wilders_scalar` sums the seed window in groups
///   of four (`sum += *p0 + *p1 + *p2 + *p3`); the AVX path groups differently.
///   The seed then feeds the recurrence, so the whole series carries it.
///
/// The kernels for both are present in `neoethos_f64_kernels.cu`, correct
/// against their scalar references, and compiled. They are one line each from
/// being live: put the `F64KernelSpec` back. What must happen first is that
/// vector-ta's CPU becomes self-consistent for them — or that `hpc_ta` pins
/// `Kernel::Scalar` for its sweep, which would also make the CPU lane
/// reproducible across hosts with different AVX support. Both are behaviour
/// changes to the CPU reference and neither belongs in a kernel-writing change.
///
/// Until then `resolve_f64_kernel` returns `CudaF64KernelMissing` for these two
/// BY NAME, which is the whole contract: no f32 kernel, no CPU, no guess.
/// # `wilders` HAS LEFT THIS LIST
///
/// Shard 2 took the third option the paragraph above did not consider: fix the
/// CPU. `wilders_scalar`, `wilders_avx2`, `wilders_avx512_short` and
/// `wilders_avx512_long` now share one seed function, `wilders_seed_sum`, so
/// the crate agrees with itself and no host needs to pin `Kernel::Scalar` for
/// the answer to be reproducible. `vwap` is the same defect awaiting the same
/// remedy in `moving_averages/vwap.rs`, which belongs to another shard.
/// # THIS LIST IS NOW EMPTY
///
/// `wilders` left it when shard 2 gave the crate one seed function.
/// `vwap` left it when shard 6 deleted `vwap_row_scalar_pv`, the second vwap
/// implementation that only the `Kernel::Scalar` batch arm reached. Both were
/// fixed in the CPU rather than worked around on the device, so no host needs
/// to pin `Kernel::Scalar` for either answer to be reproducible.
///
/// Kept, empty, on purpose: it is the place a future "the crate disagrees with
/// itself about this indicator" belongs, and an empty list is a claim that
/// there is no such indicator today.
pub const WITHHELD_PENDING_CPU_SELF_CONSISTENCY: &[(&str, F64Kernel)] = &[];

/// Does this indicator have an f64 CUDA kernel?
pub fn f64_kernel_for(indicator_id: &str) -> Option<&'static F64KernelSpec> {
    F64_KERNELS
        .iter()
        .find(|spec| spec.indicator_id == indicator_id)

}

// ---------------------------------------------------------------------------
// The three MOVING-AVERAGE DISPATCHERS — closer 6, round 3
// ---------------------------------------------------------------------------

/// `ma`, `ma_batch` and `ma_stream` are not indicators and never will be.
///
/// `ma.rs:200` is `ma(ma_type: &str, data, period)`: a `match` over the name of
/// a family member that forwards to that member's own implementation. It owns
/// no arithmetic, which is why there is no `ma_kernel.cu` and no
/// `F64KernelSpec` row for it — a row would have to name an entry point, and
/// there is no entry point for "whichever moving average you meant".
/// `ma_batch.rs:122` and `ma_stream.rs:199` are the same dispatcher over the
/// batch and streaming shapes.
///
/// That is a STRUCTURAL obstruction, stated specifically: the id does not
/// determine the computation. It is not a difficulty claim, and the remedy is
/// not a kernel — it is ROUTING, which is what this section is.
pub const MA_DISPATCHER_IDS: &[&str] = &["ma", "ma_batch", "ma_stream"];

/// Is this id one of the three dispatchers rather than an indicator?
pub fn is_ma_dispatcher(indicator_id: &str) -> bool {
    MA_DISPATCHER_IDS.contains(&indicator_id)
}

/// The aliases `ma.rs` accepts that are NOT the family member's own id.
///
/// Read straight off the match arms rather than guessed: `"corrected_moving_
/// average" | "cma"` (ma.rs:263), `"highpass2" | "highpass_2_pole"` (:503),
/// `"volatility_adjusted_ma" | "vama"` (:1397). Every other arm's pattern is
/// the id itself, so no table entry is needed for it.
const MA_TYPE_ALIASES: &[(&str, &str)] = &[
    ("cma", "corrected_moving_average"),
    ("highpass2", "highpass_2_pole"),
    ("vama", "volatility_adjusted_ma"),
];

/// Normalise an `ma_type` the way `ma.rs:201` does — `trim().to_lowercase()` —
/// and then resolve the three aliases above.
fn canonical_ma_type(ma_type: &str) -> String {
    let lowered = ma_type.trim().to_lowercase();
    for (alias, canonical) in MA_TYPE_ALIASES {
        if lowered == *alias {
            return (*canonical).to_string();
        }
    }
    lowered
}

/// Route a dispatcher request to the f64 kernel of the family member named by
/// `ma_type`.
///
/// This is what `ma` / `ma_batch` / `ma_stream` need instead of a kernel. The
/// caller supplies the `ma_type` it would have handed `ma.rs:200`; the answer
/// is the [`F64KernelSpec`] of that member, with its own input kind and its own
/// first-valid rule — never a generic one, because the family members do not
/// agree on either.
///
/// # Failure is loud and names what was asked for
///
/// * an `ma_type` this crate does not know, or one whose family member has no
///   f64 kernel yet, produces an `Err` naming the requested type. There is no
///   arm that substitutes `sma` for an unrecognised name — `ma.rs:1118` does
///   exactly that on the CPU (`eprintln!` then "Defaulting to 'sma'"), and
///   silently computing a different moving average is precisely the class of
///   defect this lane exists to remove.
/// * a `dispatcher_id` that is not one of the three is an `Err` too, rather
///   than being quietly treated as one.
pub fn resolve_f64_kernel_for_ma_type(
    dispatcher_id: &str,
    ma_type: &str,
) -> Result<&'static F64KernelSpec, IndicatorDispatchError> {
    if !is_ma_dispatcher(dispatcher_id) {
        return Err(IndicatorDispatchError::InvalidParam {
            indicator: dispatcher_id.to_string(),
            key: "ma_type".to_string(),
            reason: format!(
                "'{dispatcher_id}' is not a moving-average dispatcher; the dispatchers are {}",
                MA_DISPATCHER_IDS.join(", ")
            ),
        });
    }

    let canonical = canonical_ma_type(ma_type);
    if canonical.is_empty() {
        return Err(IndicatorDispatchError::InvalidParam {
            indicator: dispatcher_id.to_string(),
            key: "ma_type".to_string(),
            reason: "empty; the dispatcher cannot pick a family member without one".to_string(),
        });
    }

    f64_kernel_for(&canonical).ok_or_else(|| IndicatorDispatchError::InvalidParam {
        indicator: dispatcher_id.to_string(),
        key: "ma_type".to_string(),
        reason: format!(
            "'{ma_type}' resolves to '{canonical}', which has no f64 CUDA kernel. The f64 lane \
             does not fall back to f32, does not fall back to the CPU, and does not substitute \
             another moving average. Indicators with an f64 kernel: {}",
            F64_KERNELS
                .iter()
                .map(|s| s.indicator_id)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    })
}

/// Resolve an indicator id to its `*_f64` entry point, or fail by name.
///
/// This is the whole of item 2 of the f64 contract: an f64 request selects an
/// f64 entry point, and an indicator without one produces a clear `Err` naming
/// it. There is NO branch here that returns an f32 symbol and none that
/// returns a host computation.
pub fn resolve_f64_kernel(
    indicator_id: &str,
) -> Result<&'static F64KernelSpec, IndicatorDispatchError> {
    // closer 6, round 3: the three dispatchers get a reason, not the generic
    // "no kernel" message. `CudaF64KernelMissing` reads as "this indicator is
    // still owed a kernel", and for these three that would be false forever --
    // they own no arithmetic. What they are owed is an `ma_type`, and
    // `resolve_f64_kernel_for_ma_type` is the door.
    if is_ma_dispatcher(indicator_id) {
        return Err(IndicatorDispatchError::InvalidParam {
            indicator: indicator_id.to_string(),
            key: "ma_type".to_string(),
            reason: format!(
                "'{indicator_id}' is a moving-average DISPATCHER, not an indicator: \
                 ma.rs:200 / ma_batch.rs:122 / ma_stream.rs:199 select a family member \
                 and own no arithmetic, so there is no kernel by construction. Call \
                 resolve_f64_kernel_for_ma_type with the ma_type you meant."
            ),
        });
    }

    f64_kernel_for(indicator_id).ok_or_else(|| IndicatorDispatchError::CudaF64KernelMissing {
        indicator: indicator_id.to_string(),
        available: F64_KERNELS
            .iter()
            .map(|s| s.indicator_id)
            .collect::<Vec<_>>()
            .join(", "),
    })
}

/// The `__global__` entry point an f64 request for `indicator_id` will launch.
pub fn resolve_f64_entry_point(
    indicator_id: &str,
) -> Result<&'static str, IndicatorDispatchError> {
    Ok(resolve_f64_kernel(indicator_id)?.kernel.entry_point())
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn inputs_for(
    spec: &F64KernelSpec,
    data: IndicatorCudaDeviceDataRefF64,
) -> Result<F64Inputs, IndicatorDispatchError> {
    let mismatch = |wanted: &str| IndicatorDispatchError::DataLengthMismatch {
        details: format!(
            "{}: the f64 device lane needs {wanted}",
            spec.indicator_id
        ),
    };

    match (spec.input, data) {
        // shard 1: `hl2`-sourced indicators take the same explicit `Slice`
        // shape as the `hlc3`-sourced ones -- the shape says "one price
        // series", the KIND says which series the caller must have put in it.
        (F64InputKind::Hl2Slice, IndicatorCudaDeviceDataRefF64::Slice { values }) => {
            Ok(F64Inputs::Prices(values))
        }
        // closer 5: `hlcc4` and VOLUME-only take the same explicit `Slice`
        // shape for the same reason `hl2` does -- the shape says "one
        // series", the KIND says which series the caller must have put in
        // it, and the disagreement check upstream refuses a mismatch.
        (F64InputKind::Hlcc4Slice, IndicatorCudaDeviceDataRefF64::Slice { values })
        | (F64InputKind::VolumeSlice, IndicatorCudaDeviceDataRefF64::Slice { values }) => {
            Ok(F64Inputs::Prices(values))
        }
        (F64InputKind::HighLowVolume, IndicatorCudaDeviceDataRefF64::Ohlcv(r)) => {
            Ok(F64Inputs::HighLowVolume {
                high: r.high(),
                low: r.low(),
                volume: r.volume(),
            })
        }
        // closer 1: `dvdiqqe` reads OPEN, close and volume and never touches
        // high or low. Ohlcv is the only device shape that carries open
        // alongside volume, so it is the only shape accepted -- and the three
        // pointers are named here rather than passed positionally so the
        // kernel cannot receive `high` where it asked for `open`.
        (F64InputKind::OpenCloseVolume, IndicatorCudaDeviceDataRefF64::Ohlcv(r)) => {
            Ok(F64Inputs::OpenCloseVolume {
                open: r.open(),
                close: r.close(),
                volume: r.volume(),
            })
        }
        (F64InputKind::Hlcv, IndicatorCudaDeviceDataRefF64::Ohlcv(r)) => Ok(F64Inputs::Hlcv {
            high: r.high(),
            low: r.low(),
            close: r.close(),
            volume: r.volume(),
        }),
        // closer 5, round 2: the FULL bar. Only the Ohlcv ref can serve it --
        // an Ohlc ref carries open but no volume, and the Hlcv shape carries
        // volume but no open.
        (F64InputKind::Ohlcv5, IndicatorCudaDeviceDataRefF64::Ohlcv(r)) => Ok(F64Inputs::Ohlcv5 {
            open: r.open(),
            high: r.high(),
            low: r.low(),
            close: r.close(),
            volume: r.volume(),
        }),
        // shard 3: `aso` needs open as well. Both Ohlc and Ohlcv carry it.
        (F64InputKind::Ohlc4, IndicatorCudaDeviceDataRefF64::Ohlc(r)) => Ok(F64Inputs::Ohlc4 {
            open: r.open(),
            high: r.high(),
            low: r.low(),
            close: r.close(),
        }),
        (F64InputKind::Ohlc4, IndicatorCudaDeviceDataRefF64::Ohlcv(r)) => Ok(F64Inputs::Ohlc4 {
            open: r.open(),
            high: r.high(),
            low: r.low(),
            close: r.close(),
        }),
        // `Slice` is the only shape that can carry an explicitly-chosen price
        // series, which is what `hlc3`-sourced indicators require.
        (F64InputKind::CloseSlice, IndicatorCudaDeviceDataRefF64::Slice { values })
        | (F64InputKind::Hlc3Slice, IndicatorCudaDeviceDataRefF64::Slice { values }) => {
            Ok(F64Inputs::Prices(values))
        }
        (F64InputKind::CloseSlice, IndicatorCudaDeviceDataRefF64::Ohlcv(r)) => {
            Ok(F64Inputs::Prices(r.prices()))
        }
        (F64InputKind::CloseSlice, IndicatorCudaDeviceDataRefF64::Ohlc(r)) => {
            Ok(F64Inputs::Prices(r.prices()))
        }
        (F64InputKind::Hlc, IndicatorCudaDeviceDataRefF64::Ohlcv(r)) => Ok(F64Inputs::Hlc {
            high: r.high(),
            low: r.low(),
            close: r.close(),
        }),
        (F64InputKind::Hlc, IndicatorCudaDeviceDataRefF64::Ohlc(r)) => Ok(F64Inputs::Hlc {
            high: r.high(),
            low: r.low(),
            close: r.close(),
        }),
        // `Hlc3Volume` and `CloseVolume` intentionally accept the SAME device
        // shape: a (price, volume) pair. They differ only in which price series
        // the caller must have put in it, and that is checked upstream by
        // `GpuIndicatorEngine::data_ref`, which builds the pair from hlc3 or
        // from close according to this same declaration. Two names for one
        // shape is the point — it makes the wrong pairing a compile-time
        // mismatch in the table rather than a plausible-looking number.
        // closer 6, round 3: `Hlcc4Volume` joins the same shape for the same
        // reason -- the pair carries hlcc4 rather than close or hlc3, and which
        // one it is has already been settled by `GpuIndicatorEngine::data_ref`
        // reading this very declaration.
        (F64InputKind::Hlc3Volume, IndicatorCudaDeviceDataRefF64::CloseVolume(r))
        | (F64InputKind::Hlcc4Volume, IndicatorCudaDeviceDataRefF64::CloseVolume(r))
        | (F64InputKind::CloseVolume, IndicatorCudaDeviceDataRefF64::CloseVolume(r)) => {
            Ok(F64Inputs::PriceVolume {
                price: r.close(),
                volume: r.volume(),
            })
        }
        (F64InputKind::HighLow, IndicatorCudaDeviceDataRefF64::HighLow(r)) => {
            Ok(F64Inputs::HighLow {
                high: r.high(),
                low: r.low(),
            })
        }
        (
            F64InputKind::TimestampCloseVolume,
            IndicatorCudaDeviceDataRefF64::TimestampCloseVolume {
                timestamps,
                close,
                volume,
            },
        ) => Ok(F64Inputs::TimestampPriceVolume {
            timestamps,
            price: close,
            volume,
        }),
        // Refuse rather than substitute. Feeding `close` to an hlc3-sourced
        // indicator would compute a different indicator and pass every
        // shape check on the way.
        (F64InputKind::Hlc3Slice, _) => Err(mismatch(
            "an explicit Slice over the hlc3 series (the CPU source for this indicator is hlc3, \
             not close)",
        )),
        (F64InputKind::Hlc3Volume, _) => Err(mismatch(
            "an explicit CloseVolume built from hlc3 and volume (the CPU source for this \
             indicator is hlc3, not close)",
        )),
        (F64InputKind::Hlc, _) => Err(mismatch("Ohlc or Ohlcv (high, low and close)")),
        (F64InputKind::CloseSlice, _) => Err(mismatch("Slice, Ohlc or Ohlcv")),
        (F64InputKind::CloseVolume, _) => Err(mismatch(
            "a CloseVolume built from CLOSE and volume (this indicator's CPU source is close, not hlc3)",
        )),
        // Deliberately NOT served by Ohlc/Ohlcv: those carry close, and
        // accepting them would silently adopt close's first-valid index for an
        // indicator whose CPU reference scans high and low only.
        (F64InputKind::HighLow, _) => Err(mismatch("HighLow (high and low, with no close)")),
        (F64InputKind::Hl2Slice, _) => Err(mismatch(
            "an explicit Slice over the hl2 series (the CPU source for this indicator is hl2, \
             not close)",
        )),
        (F64InputKind::HighLowVolume, _) => Err(mismatch(
            "Ohlcv (high, low and volume; this indicator never reads close)",
        )),
        (F64InputKind::Hlcv, _) => Err(mismatch("Ohlcv (high, low, close and volume)")),
        (F64InputKind::Ohlcv5, _) => Err(mismatch(
            "Ohlcv (open, high, low, close AND volume -- this indicator reads open in its \n             validity gate, so the four-pointer Hlcv shape would miss the bars that reset it)",
        )),
        (F64InputKind::OpenCloseVolume, _) => Err(mismatch(
            "Ohlcv (open, close and volume -- this indicator reads OPEN and never high or low)",
        )),
        (F64InputKind::Ohlc4, _) => Err(mismatch(
            "Ohlc or Ohlcv (open, high, low and close -- this indicator reads open)",
        )),
        // closer 6, round 3
        (F64InputKind::Hlcc4Volume, _) => Err(mismatch(
            "CloseVolume built from HLCC4 and volume (the CPU default source here is \
             hlcc4, not close and not hlc3)",
        )),
        (F64InputKind::TimestampCloseVolume, _) => Err(mismatch(
            "TimestampCloseVolume — vwap is anchored by calendar bucket, so the bar timestamps are an input, not metadata",
        )),
        // THE ONE BUILD, round 2 — these two kinds were added by closer 5
        // together with their `Slice` acceptance arm, but neither got the
        // trailing mismatch arm every other kind has, so the match was
        // non-exhaustive. rustc named only `Hlcc4Slice` (it reports one
        // witness set); `VolumeSlice` was the same omission one arm later and
        // would have failed the very next round. Both are closed here.
        (F64InputKind::Hlcc4Slice, _) => Err(mismatch(
            "an explicit Slice over the hlcc4 series (the CPU source for this indicator is \
             hlcc4, not close and not hlc3)",
        )),
        (F64InputKind::VolumeSlice, _) => Err(mismatch(
            "an explicit Slice over the VOLUME series (this indicator reads volume alone and \
             never a price series)",
        )),
    }
}

/// Run one indicator's f64 period sweep on the device.
///
/// `engine` is held by the caller so a whole frame pays ONE module load rather
/// than one per indicator — this crate has no module cache, and the f32 lane's
/// dispatcher constructs a fresh wrapper (and therefore a fresh JIT) on every
/// call.
pub fn compute_cuda_device_f64(
    engine: &CudaF64Indicators,
    req: IndicatorCudaDeviceRequestF64<'_>,
) -> Result<IndicatorCudaOutputF64, IndicatorDispatchError> {
    let spec = resolve_f64_kernel(req.indicator_id)?;

    if req.data.is_empty() {
        return Err(IndicatorDispatchError::DataLengthMismatch {
            details: format!("{}: empty device series", req.indicator_id),
        });
    }
    if req.periods.is_empty() {
        return Err(IndicatorDispatchError::InvalidParam {
            indicator: req.indicator_id.to_string(),
            key: "periods".to_string(),
            reason: "empty period list".to_string(),
        });
    }

    let cols = req.data.len();
    let inputs = inputs_for(spec, req.data)?;

    let result = engine
        .sweep(spec.kernel, inputs, req.periods, req.first_valid)
        .map_err(|e| IndicatorDispatchError::ComputeFailed {
            indicator: req.indicator_id.to_string(),
            details: format!("{e}"),
        })?;

    let rows = result.rows;
    let entry_point = spec.kernel.entry_point();

    let series = match req.target {
        CudaOutputTargetF64::Host => {
            let host = result
                .to_host()
                .map_err(|e| IndicatorDispatchError::ComputeFailed {
                    indicator: req.indicator_id.to_string(),
                    details: format!("device→host copy failed: {e}"),
                })?;
            IndicatorCudaSeriesF64::HostF64(host)
        }
        CudaOutputTargetF64::Device => {
            let matrix = CudaDeviceMatrixF64::from_buffer(
                result.buf,
                rows,
                cols,
                engine.context_arc(),
                engine.device_id(),
            )
            .map_err(|e| IndicatorDispatchError::ComputeFailed {
                indicator: req.indicator_id.to_string(),
                details: format!("device matrix view failed: {e}"),
            })?;
            IndicatorCudaSeriesF64::DeviceF64(matrix)
        }
    };

    Ok(IndicatorCudaOutputF64 {
        indicator_id: req.indicator_id.to_string(),
        series,
        rows,
        cols,
        entry_point,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Resolution must never produce an f32 symbol, and must fail by name for
    /// an indicator with no f64 kernel. Runs without a card.
    #[test]
    fn resolution_is_f64_only_and_fails_by_name() {
        for spec in F64_KERNELS {
            let name = resolve_f64_entry_point(spec.indicator_id).expect("registered");
            assert!(
                name.ends_with("_f64"),
                "{}: resolved to {name}, which is not an f64 entry point",
                spec.indicator_id
            );
        }

        // The probe used to be `stoch`. Closer 6 gave stoch an f64 kernel
        // (`stoch_batch_f64`, oscillators/stoch_kernel.cu), so it is no longer
        // an example of a missing one and asserting that it fails would now
        // assert the opposite of the truth.
        //
        // `rogers_satchell_volatility` replaces it, and it is the STRONGEST
        // remaining example rather than an arbitrary one: it has no `.cu` file
        // at all, and its wrapper
        // (`src/cuda/rogers_satchell_volatility_wrapper.rs`) computes on the
        // HOST and uploads the result with `DeviceBuffer::from_slice`
        // (:263, :309, :372) without a single `get_function` call -- the
        // disguise this lane exists to make impossible. The f64 lane must say
        // "no kernel" by name rather than quietly serving that.
        //
        // WHEN THAT KERNEL IS WRITTEN, THIS PROBE MOVES -- it does not get
        // deleted. An empty probe would let the "fails by name" guarantee rot
        // silently.
        let err = resolve_f64_entry_point("rogers_satchell_volatility").unwrap_err();
        let text = format!("{err}");
        assert!(
            text.contains("rogers_satchell_volatility"),
            "the error must name the indicator: {text}"
        );
        assert!(
            matches!(
                err,
                IndicatorDispatchError::CudaF64KernelMissing { .. }
            ),
            "wrong variant: {err:?}"
        );
    }

    #[test]
    fn table_has_no_duplicate_ids() {
        let mut ids: Vec<&str> = F64_KERNELS.iter().map(|s| s.indicator_id).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "duplicate indicator id in F64_KERNELS");
    }

    /// The two indicators whose CPU source is hlc3 must be declared as such,
    /// so the dispatcher refuses an OHLCV ref (whose price accessor is close)
    /// instead of computing a different indicator.
    #[test]
    fn hlc3_sourced_indicators_are_declared() {
        for id in ["cci", "mfi"] {
            let spec = f64_kernel_for(id).expect("registered");
            assert!(
                matches!(
                    spec.input,
                    F64InputKind::Hlc3Slice | F64InputKind::Hlc3Volume
                ),
                "{id}: CPU source is hlc3 but the spec would accept close"
            );
        }
    }

    /// The six high/low/close indicators do NOT share one first-valid rule,
    /// and the whole point of declaring the rule per indicator is that a caller
    /// cannot re-merge them. Pinned by name, with the CPU site each answer came
    /// from, so a future edit that "simplifies" this back to one rule fails
    /// here rather than on a gapped symbol six months later.
    #[test]
    fn the_hlc_indicators_do_not_share_one_first_valid_rule() {
        let rule = |id: &str| f64_kernel_for(id).expect("registered").first_valid;

        // atr.rs:197-206, willr.rs:300, wclprice.rs:176 — all three non-NaN
        // at the SAME index.
        for id in ["atr", "willr", "wclprice"] {
            assert_eq!(
                rule(id),
                F64FirstValidRule::AllInputsNonNan,
                "{id}: CPU scans the triple simultaneously"
            );
        }

        // adx.rs:201-219 `first_valid_triple_checked`, natr.rs:226-235 — the
        // MAX of three INDEPENDENT scans.
        for id in ["adx", "natr"] {
            assert_eq!(
                rule(id),
                F64FirstValidRule::HlcMaxOfIndependentFirsts,
                "{id}: CPU takes fh.max(fl).max(fc)"
            );
        }

        // adxr.rs:255-258 — close alone; high and low are never scanned.
        assert_eq!(
            rule("adxr"),
            F64FirstValidRule::HlcCloseOnly,
            "adxr: CPU scans close only"
        );

        // And the three rules really are distinct, so a caller that collapses
        // them is not merely tidying.
        assert_ne!(rule("atr"), rule("adx"));
        assert_ne!(rule("adx"), rule("adxr"));
        assert_ne!(rule("atr"), rule("adxr"));
    }

    /// Every departure from the common first-valid rule is PINNED BY NAME,
    /// with the CPU site it was read from.
    ///
    /// This test used to assert that only `F64InputKind::Hlc` entries could
    /// depart. That was true of the ten indicators in the table when it was
    /// written and it is not a law: four of shard 6's indicators depart while
    /// reading (high, low) or a single close series --
    ///
    ///   * `aroonosc`               `aroonosc.rs:16-20`   both `is_finite`
    ///   * `parkinson_volatility`   `parkinson_volatility.rs:214-223`
    ///                              both finite AND `> 0`
    ///   * `donchian`               `donchian.rs:183-188` MAX of two
    ///                              INDEPENDENT scans
    ///   * `historical_volatility`  `historical_volatility.rs:334-355`
    ///                              first index that can form a RETURN
    ///
    /// So the invariant that is actually worth holding is not "only Hlc may
    /// depart" but "a departure is deliberate and listed here". A new row with
    /// an unusual rule fails this test until someone writes down which CPU
    /// line it came from.
    #[test]
    fn departures_from_the_common_rule_are_declared_by_name() {
        const DECLARED: &[(&str, F64FirstValidRule)] = &[
            // adosc.rs:331 -- `first = 0` outright, cumulative from bar zero
            ("adosc", F64FirstValidRule::Ignored),
            // ------------------------------------------- closer 4, round 2
            // kase_peak_oscillator_with_divergences.rs -- the stream resets on any bar
            // whose high, low or close is non-finite OR non-positive; no variant
            // expresses `finite AND > 0` on three series
            ("kase_peak_oscillator_with_divergences", F64FirstValidRule::Ignored),
            // keltner_channel_width_oscillator.rs:400 is_valid_bar -- finite
            // h/l/c/source AND `high >= low`, an ORDERING condition no variant
            // expresses
            ("keltner_channel_width_oscillator", F64FirstValidRule::Ignored),
            // leavitt_convolution_acceleration.rs:776 -- `first` bounds the leading
            // NaN fill only; the stream resets on any non-finite bar
            ("leavitt_convolution_acceleration", F64FirstValidRule::Ignored),
            // market_meanness_index.rs:456 -- the stream walks from 0 and resets; the
            // CPU's dirty path leaves post-prefix misses uninitialised
            ("market_meanness_index", F64FirstValidRule::Ignored),
            // market_structure_confluence.rs -- the stream walks from index 0 and
            // computes no first-valid index at all
            ("market_structure_confluence", F64FirstValidRule::Ignored),
            // monotonicity_index.rs:499 -- resets on any non-finite bar and
            // row_from_slice writes NaN into every slot it does not fill (:664)
            ("monotonicity_index", F64FirstValidRule::Ignored),
            // premier_rsi_oscillator.rs:556 -- `first` sizes the NaN prefix only; the
            // stream resets on any non-finite bar (:288)
            ("premier_rsi_oscillator", F64FirstValidRule::Ignored),
            // pretty_good_oscillator.rs:244 is_valid_bar -- finite h/l/c/source AND
            // `high >= low`
            ("pretty_good_oscillator", F64FirstValidRule::Ignored),
            // price_density_market_noise.rs:565 is literally `let _ = first;` and the
            // prefix is alloc_with_nan_prefix(len, 0)
            ("price_density_market_noise", F64FirstValidRule::Ignored),
            // projection_oscillator.rs:713 -- resets on any invalid triple; the NaN
            // prefix is the fixed warmup, not a scan
            ("projection_oscillator", F64FirstValidRule::Ignored),
            // qqe_weighted_oscillator.rs:428 -- data.iter().position(|v|
            // v.is_finite()); LOAD-BEARING, the loop starts at first + 1
            ("qqe_weighted_oscillator", F64FirstValidRule::CloseFinite),
            // rogers_satchell_volatility.rs:442 -- prepare_input COUNTS valid bars but
            // never locates the first; both compute paths start at index 0
            ("rogers_satchell_volatility", F64FirstValidRule::Ignored),
            // rolling_skewness_kurtosis.rs:350 -- walks from 0 and resets the whole
            // accumulator on any non-finite bar
            ("rolling_skewness_kurtosis", F64FirstValidRule::Ignored),
            // smooth_theil_sen.rs:455 -- data.iter().position(|v| v.is_finite());
            // LOAD-BEARING, warmup is first + length + offset - 1
            ("smooth_theil_sen", F64FirstValidRule::CloseFinite),
            // ------------------------------------------- closer 5, round 2
            ("smoothed_gaussian_trend_filter", F64FirstValidRule::Ignored),
            ("spearman_correlation", F64FirstValidRule::Ignored),
            ("squeeze_index", F64FirstValidRule::Ignored),
            ("standardized_psar_oscillator", F64FirstValidRule::Ignored),
            ("statistical_trailing_stop", F64FirstValidRule::Ignored),
            ("stochastic_adaptive_d", F64FirstValidRule::Ignored),
            ("stochastic_connors_rsi", F64FirstValidRule::Ignored),
            ("stochastic_distance", F64FirstValidRule::Ignored),
            ("stochastic_money_flow_index", F64FirstValidRule::Ignored),
            ("supertrend_oscillator", F64FirstValidRule::Ignored),
            ("supertrend_recovery", F64FirstValidRule::Ignored),
            ("trend_flow_trail", F64FirstValidRule::Ignored),
            ("twiggs_money_flow", F64FirstValidRule::Ignored),
            ("volatility_quality_index", F64FirstValidRule::Ignored),
            ("vwap_deviation_oscillator", F64FirstValidRule::Ignored),
            ("vwap_zscore_with_signals", F64FirstValidRule::Ignored),
            // ------------------------------------------- closer 1
            // absolute_strength_index_oscillator.rs:521
            ("absolute_strength_index_oscillator", F64FirstValidRule::Ignored),
            // accumulation_swing_index.rs:245
            ("accumulation_swing_index", F64FirstValidRule::Ohlc4AllFinite),
            // adaptive_bandpass_trigger_oscillator.rs:490
            ("adaptive_bandpass_trigger_oscillator", F64FirstValidRule::Ignored),
            // adaptive_bounds_rsi.rs:670
            ("adaptive_bounds_rsi", F64FirstValidRule::Ignored),
            // adaptive_macd.rs:789
            ("adaptive_macd", F64FirstValidRule::Ignored),
            // adaptive_momentum_oscillator.rs:573
            ("adaptive_momentum_oscillator", F64FirstValidRule::Ignored),
            // advance_decline_line.rs:227
            ("advance_decline_line", F64FirstValidRule::Ignored),
            // andean_oscillator.rs:244
            ("andean_oscillator", F64FirstValidRule::OpenCloseFinite),
            // atr_percentile.rs:636
            ("atr_percentile", F64FirstValidRule::Ignored),
            // bop.rs:209
            ("bop", F64FirstValidRule::Ohlc4AllNonNan),
            // bull_power_vs_bear_power.rs:352
            ("bull_power_vs_bear_power", F64FirstValidRule::Ignored),
            // daily_factor.rs:258
            ("daily_factor", F64FirstValidRule::Ohlc4AllFinite),
            // decisionpoint_breadth_swenlin_trading_oscillator.rs:329
            ("decisionpoint_breadth_swenlin_trading_oscillator", F64FirstValidRule::Ignored),
            // didi_index.rs:480
            ("didi_index", F64FirstValidRule::Ignored),
            // disparity_index.rs:564
            ("disparity_index", F64FirstValidRule::Ignored),
            // donchian_channel_width.rs:354
            ("donchian_channel_width", F64FirstValidRule::Ignored),
            ("adx", F64FirstValidRule::HlcMaxOfIndependentFirsts),
            ("natr", F64FirstValidRule::HlcMaxOfIndependentFirsts),
            ("adxr", F64FirstValidRule::HlcCloseOnly),
            // shard 2 -- `tradjema_prepare` scans close alone; see the
            // rationale beside its row in F64_KERNELS.
            ("tradjema", F64FirstValidRule::HlcCloseOnly),
            ("vwap", F64FirstValidRule::Ignored),
            ("aroonosc", F64FirstValidRule::HighLowFinite),
            (
                "parkinson_volatility",
                F64FirstValidRule::HighLowFiniteAndPositive,
            ),
            ("donchian", F64FirstValidRule::MaxOfIndependentFirsts),
            (
                "historical_volatility",
                F64FirstValidRule::ConsecutiveValidReturnPair,
            ),
            // shard 4 -- three more departures, each read from the CPU:
            //   cksp.rs:281        `close.iter().position(..)`, close alone
            //   ttm_squeeze.rs:384 `close.iter().position(..)`, close alone
            //   aroon_scalar       starts at absolute index `length`; `first`
            //                      is never read
            //   acosc              no first-valid scan and no length parameter
            //   ad.rs:209          `alloc_with_nan_prefix(size, 0)`; ad_scalar
            //                      (:298) starts at index 0
            ("cksp", F64FirstValidRule::HlcCloseOnly),
            ("ttm_squeeze", F64FirstValidRule::HlcCloseOnly),
            ("aroon", F64FirstValidRule::Ignored),
            ("acosc", F64FirstValidRule::Ignored),
            ("ad", F64FirstValidRule::Ignored),
            //   dvdiqqe.rs:385   `position(|x| x.is_finite())` on close --
            //                    rejects an infinity that `!is_nan` accepts
            ("dvdiqqe", F64FirstValidRule::CloseFinite),
            // closer 6 -- keltner.rs:293-296
            //   `close.iter().position(|x| !x.is_nan())`, close alone, even
            //   though the indicator reads high and low at every bar for the
            //   true range. The ATR seed does not consult `first` at all
            //   (keltner.rs:707), so a late high cannot be repaired by a later
            //   first-valid and declaring the triple rule here would only
            //   shift the output relative to the CPU.
            ("keltner", F64FirstValidRule::HlcCloseOnly),
            // ------------------------------------------------------ closer 3
            // l1_ehlers_phasor.rs:229 -- `first_valid` scans with `is_finite`,
            // so an INFINITE bar is skipped where `!is_nan` would accept it.
            ("l1_ehlers_phasor", F64FirstValidRule::CloseFinite),
            // l2_ehlers_signal_to_noise.rs:263 -- `first_valid_triple` needs
            // source/high/low all `is_finite`, and the source is hl2, i.e.
            // exactly "high and low both finite".
            ("l2_ehlers_signal_to_noise", F64FirstValidRule::HighLowFinite),
            // kairi_relative_index.rs:732 -- `compute_default_sma50_into` fills
            // the output with NaN and walks from index 0, so there is no
            // first-valid index to declare and no warmup prefix to imply.
            ("kairi_relative_index", F64FirstValidRule::Ignored),
            // momentum_ratio_oscillator.rs:292 and
            // on_balance_volume_oscillator.rs:523 both walk from index 0 and
            // allocate without a NaN prefix, so there is no first-valid index
            // to declare.
            ("momentum_ratio_oscillator", F64FirstValidRule::Ignored),
            ("on_balance_volume_oscillator", F64FirstValidRule::Ignored),
            // ------------------------------------------------------ closer 2
            // ewma_volatility.rs:274 `valid_sq_return` -- the series cannot
            // start before a RETURN exists, and a return needs two consecutive
            // finite, strictly positive closes. `AllInputsNonNan` would name a
            // bar at least one earlier and would accept a zero previous close.
            ("ewma_volatility", F64FirstValidRule::ConsecutiveValidReturnPair),
            // gopalakrishnan_range_index.rs:337 `valid_high_low_bar` -- both
            // series `is_finite`, which rejects an infinite high that the
            // `!is_nan` scan would accept.
            ("gopalakrishnan_range_index", F64FirstValidRule::HighLowFinite),
            // garman_klass_volatility.rs:346 `validity_summary` -- all FOUR
            // prices finite AND strictly positive, because `gk_term` (:315)
            // takes ln(high/low) and ln(close/open). No rule here expresses
            // that, so the kernel derives its own start and this row declares
            // the caller value unused rather than declaring a rule it does not
            // honour.
            ("garman_klass_volatility", F64FirstValidRule::Ignored),
            // ----------------------------------------------------- closer 2b
            // ehlers_fm_demodulator.rs:566 `batch_prepare` -- the scan covers
            // OPEN and CLOSE only; high and low are length-checked and never
            // read, so the OHLC quadruple rule would start the series late on
            // any frame whose high or low has the earlier hole.
            ("ehlers_fm_demodulator", F64FirstValidRule::OpenCloseNonNan),
            // ---------------------------------------------- closer 2, round 2
            // Four `Ignored` rows, each read off the CPU rather than assumed.
            // In every one of them the row-producing function fills the whole
            // output with NaN and then walks from index 0, so there is no
            // first-valid index to declare and no warmup prefix to imply; the
            // kernel derives whatever start it needs itself.
            //
            // dual_ulcer_index.rs:574 `out.fill(f64::NAN)` then
            //   `for i in 0..len` (:589). The `warmup` at :701 belongs to
            //   `dual_ulcer_index_with_kernel`, a DIFFERENT entry point that
            //   allocates its own prefix and is not what the batch calls.
            ("dual_ulcer_index", F64FirstValidRule::Ignored),
            // hull_butterfly_oscillator.rs:521 `out.fill(f64::NAN)` then the
            //   stream is zipped over `data.iter()` from 0 (:524). Same
            //   two-entry-point split as above -- the `warmup` at :551 is
            //   `hull_butterfly_oscillator_with_kernel`'s.
            ("hull_butterfly_oscillator", F64FirstValidRule::Ignored),
            // range_oscillator.rs:335-343 allocates with
            //   `alloc_with_nan_prefix(len, 0)` -- prefix ZERO -- and
            //   `compute_into_slices` writes EVERY index. `prepared.first`
            //   (:498) is used only by the NotEnoughValidData check at
            //   :515-521, which the kernel reproduces itself.
            ("range_oscillator", F64FirstValidRule::Ignored),
            // market_structure_trailing_stop.rs:604 `compute_row` scans from
            //   index 0 and SEGMENTS the frame into maximal valid-OHLC runs
            //   itself (:615-624). The `clean_tail` branch (:670-706) is the
            //   same answer with one run, so there is no single first-valid
            //   index that describes the output.
            ("market_structure_trailing_stop", F64FirstValidRule::Ignored),
            // ------------------------------------------- closer 4, round 3
            // bandpass.rs:255 -- `position(|x| x.is_finite())`, not `!is_nan`.
            //   An INFINITE bar is skipped by the CPU and would be accepted by
            //   the common rule.
            ("bandpass", F64FirstValidRule::CloseFinite),
            // buff_averages.rs:470 -- PRICE ALONE, `!is_nan`. Volume is never
            //   scanned, so the common rule over the (close, volume) pair this
            //   row declares would name a later bar on any frame whose volume
            //   has a hole. Derived in the kernel.
            ("buff_averages", F64FirstValidRule::Ignored),
            // fvg_trailing_stop.rs:1040 -- the batch takes
            //   `fvg_trailing_stop_with_kernel`, which allocates with
            //   `alloc_uninit_f64` and applies NO warmup prefix; the loop runs
            //   from bar 0 and writes every bar.
            ("fvg_trailing_stop", F64FirstValidRule::Ignored),
            // halftrend.rs:291 `first_valid_ohlc` -- the MIN of three
            //   INDEPENDENT scans, `fh.min(fl).min(fc)`. No variant expresses
            //   it: HlcMaxOfIndependentFirsts is the MAX and names a LATER
            //   bar. Derived in the kernel.
            ("halftrend", F64FirstValidRule::Ignored),
            // mod_god_mode.rs:693 -- CLOSE ALONE, `!is_nan`, while the row is
            //   registered Hlcv; the common rule over that shape scans all
            //   four series. Derived in the kernel.
            ("mod_god_mode", F64FirstValidRule::Ignored),
            // otto.rs:1605 -- `alloc_with_nan_prefix(len, 0)`. There is no
            //   warmup prefix at all and both passes walk from bar 0, so a
            //   first-valid index would name a bar the CPU never skips.
            ("otto", F64FirstValidRule::Ignored),
        ];

        for spec in F64_KERNELS {
            if spec.first_valid == F64FirstValidRule::AllInputsNonNan {
                assert!(
                    !DECLARED.iter().any(|(id, _)| *id == spec.indicator_id),
                    "{}: listed as a departure but declares the common rule",
                    spec.indicator_id
                );
                continue;
            }
            let declared = DECLARED
                .iter()
                .find(|(id, _)| *id == spec.indicator_id)
                .unwrap_or_else(|| {
                    panic!(
                        "{}: departs from AllInputsNonNan with {:?} but is not listed here.                          Add it, with the CPU file:line the rule was read from.",
                        spec.indicator_id, spec.first_valid
                    )
                });
            assert_eq!(
                declared.1, spec.first_valid,
                "{}: table and this list disagree about the rule",
                spec.indicator_id
            );
        }
    }
}
