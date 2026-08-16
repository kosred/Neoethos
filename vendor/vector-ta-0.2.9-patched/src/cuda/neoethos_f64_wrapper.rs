#![cfg(feature = "cuda")]

//! Host side of the NeoEthos f64 indicator lane.
//!
//! One module, ten indicators, `f64` end to end.
//!
//! # What this is for
//!
//! Every other CUDA wrapper in this crate loads its own module (309
//! `load_cuda_embedded_module!` sites) and there is no module cache, so a
//! dispatcher call constructs a wrapper and pays a full JIT per indicator —
//! about fifty per frame in the NeoEthos feature build. This wrapper holds ONE
//! module covering all ten indicators, so a frame pays one load and then only
//! launches.
//!
//! # The three properties that matter more than speed
//!
//! * **f64 in, f64 out.** No narrowing anywhere. The device buffers are
//!   `DeviceBuffer<f64>`, the views are [`CudaDeviceSliceF64Ref`], and the
//!   kernels are the ones in `kernels/cuda/neoethos_f64_kernels.cu`, compiled
//!   without fast math by construction.
//! * **No silent fallback.** Every failure is a named `Err`. Nothing here
//!   computes a host value, and nothing here reaches for an f32 kernel when an
//!   f64 one is unavailable. That is the failure mode this lane exists to
//!   remove: nine `*_f64` kernels already in this crate are one-line empty
//!   stubs whose wrappers resolve the symbol, never launch it, and compute on
//!   the CPU instead (`possible_rsi_wrapper.rs:104-152` is the clearest).
//! * **Peak memory is a function of the hardware.** The output is
//!   `rows * cols * 8` bytes and `rows` is caller-supplied, so the sweep is
//!   CHUNKED over rows against `mem_get_info` — see [`rows_per_chunk`]. A
//!   larger period list makes the sweep slower, never fatter.

use crate::cuda::device_types::CudaDeviceSliceI64Ref;
use crate::cuda::device_types_f64::CudaDeviceSliceF64Ref;
use cust::context::Context;
use cust::device::{Device, DeviceAttribute};
use cust::function::{BlockSize, GridSize};
use cust::launch;
use cust::memory::{mem_get_info, CopyDestination, DeviceBuffer};
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::sync::Arc;
use thiserror::Error;

const BLOCK_X: u32 = 64;
/// Bars per block for the (combo, bar)-parallel kernels.
const BAR_BLOCK_X: u32 = 256;
/// VRAM left untouched so a concurrent allocation does not fail because this
/// sweep sized itself to the last free byte.
const DEFAULT_HEADROOM: usize = 128 * 1024 * 1024;
/// Must match `MFI_MAX_PERIOD` in `kernels/cuda/neoethos_f64_kernels.cu`.
/// The mfi kernel keeps its money-flow ring in a per-thread local array of
/// this fixed size, so the bound is a property of the kernel, not of the
/// caller's request — and a request beyond it is REFUSED, never truncated.
pub const MFI_MAX_PERIOD: usize = 512;
/// Must match `ADXR_MAX_PERIOD` in `kernels/cuda/neoethos_f64_kernels.cu`.
/// Same contract as [`MFI_MAX_PERIOD`]: adxr keeps `period` past ADX values in
/// a per-thread ring, so the bound belongs to the kernel and a larger period is
/// refused by name rather than silently truncating the lookback.
pub const ADXR_MAX_PERIOD: usize = 512;
/// Must match `NEO_EHMA_MAX_PERIOD` in
/// `kernels/cuda/moving_averages/ehma_kernel.cu`.
///
/// `ehma` is the only kernel in shard 4 that needs a per-thread array. Its CPU
/// reference builds the Hann weights by a forward (cos, sin) rotation and then
/// REVERSES them, and running that rotation backwards is not bit-equal to
/// running it forwards — so the row must materialise the weights before it can
/// consume them in reverse. `sweep` refuses a larger period BY NAME rather
/// than truncating the window (a different indicator) or moving it to the host
/// (the silent fallback this lane exists to remove).
pub const EHMA_MAX_PERIOD: usize = 512;
/// Must match `NEO_S1_CHOP_MAX_PERIOD` in
/// `kernels/cuda/oscillators/chop_kernel.cu`. The general path slides
/// `rolling_sum_atr` over a ring of `period` values with subtract-then-add, so
/// the ring cannot be recomputed from the window and its length is a property
/// of the compiled kernel.
pub const CHOP_MAX_PERIOD: usize = 1024;
/// Must match `NEO_S1_HMA_MAX_PERIOD` in
/// `kernels/cuda/moving_averages/hma_kernel.cu`. The third stage keeps the last
/// `floor(sqrt(period))` values of the `2*wma_half - wma_full` series; a
/// 64-entry ring bounds the period at 4095.
pub const HMA_MAX_PERIOD: usize = 4095;
/// Must match `NEO_S1_EDCF_MAX_PERIOD` in
/// `kernels/cuda/moving_averages/edcf_kernel.cu`. Two rings of `period`
/// doubles -- the prices and their weights -- are structural to the O(1)
/// reformulation the CPU uses.
pub const EDCF_MAX_PERIOD: usize = 512;
/// Must match `NEO_S1_ALMA_MAX_PERIOD` in
/// `kernels/cuda/moving_averages/alma_kernel.cu`. The Gaussian weights are
/// built per row into a per-thread array.
pub const ALMA_MAX_PERIOD: usize = 1024;


/// The per-thread ring bound shared by every shard-2 kernel that keeps a
/// window in local memory (`reflex`, `maaq`, `tradjema`, `pwma`, `nama`,
/// `sama`, `ehlers_itrend`, ...). Must match the `*_MAX_PERIOD` /
/// `*_MAX_LENGTH` `#define` in each of those `.cu` files. Stated once here so
/// the host refuses an oversized period BY NAME instead of the kernel
/// overrunning a local array.
pub const S2_RING_MAX_PERIOD: usize = 512;
/// Must match `NEO_TRIMA_MAX_PERIOD` in
/// `kernels/cuda/moving_averages/trima_kernel.cu`. `trima` is a double
/// moving average whose inner ring is `m2 = period - (period+1)/2 + 1`
/// deep, so 512 bounds the per-thread array at 257 slots. An oversized
/// period is REFUSED BY NAME rather than truncated.
pub const TRIMA_MAX_PERIOD: usize = 512;

/// The per-thread weight-array bound `nadaraya_watson_envelope` carries.
///
/// `nwe_prepare` (nadaraya_watson_envelope.rs:412-418) builds a `lookback`-long
/// Gaussian weight vector on the host; a kernel has no allocator, so the array
/// is sized at compile time and this is the number. It must match
/// `NWE_F64_LOOKBACK` in `kernels/cuda/nadaraya_watson_envelope_kernel.cu`.
///
/// 500 rather than 512 on purpose: it is the CPU DEFAULT
/// (`get_usize_param("nadaraya_watson_envelope", params, "lookback", 500)`,
/// cpu_batch.rs:15621) and the indicator is period-invariant, so 500 is not a
/// generous bound on a swept parameter -- it is the only value the CPU batch
/// ever uses. A larger lookback is refused BY NAME rather than truncated.
pub const NWE_MAX_LOOKBACK: usize = 500;
/// Must match `MEDIUM_AD_MAX_PERIOD` in `kernels/cuda/medium_ad_kernel.cu`.
/// `medium_ad_neo_batch_f64` copies the whole trailing window into a per-thread
/// local array before selecting the median, so the bound is a property of the
/// COMPILED KERNEL: a larger period is REFUSED BY NAME rather than computing a
/// median over a truncated window, which would be a different indicator that
/// still returned plausible numbers.
pub const MEDIUM_AD_MAX_PERIOD: usize = 512;
/// Must match `NEO_DII_MAX_PERIOD` in
/// `kernels/cuda/directional_imbalance_index_kernel.cu`.
///
/// `directional_imbalance_index` keeps its up/down HIT flags in a per-thread
/// ring `period` deep and maintains the two sums with subtract-then-add, so the
/// ring cannot be recomputed from the window and its length is a property of
/// the COMPILED kernel. A larger period is REFUSED BY NAME rather than
/// truncating the hit window, which would be a different indicator that still
/// returned plausible numbers.
pub const DII_MAX_PERIOD: usize = 512;
/// Must match `NEO_CSO_MAX_PERIOD` in
/// `kernels/cuda/candle_strength_oscillator_kernel.cu`.
///
/// `candle_strength_oscillator` is three nested weighted moving averages whose
/// windows are `period`, `period / 2` and `floor(sqrt(period))`; all three
/// rings live in per-thread arrays sized at compile time, so the bound belongs
/// to the kernel and not to the caller.
pub const CSO_MAX_PERIOD: usize = 512;

/// Must match `NEO_VRARSX_MAX_PERIOD` in
/// `kernels/cuda/volatility_ratio_adaptive_rsx_kernel.cu`.
///
/// `volatility_ratio_adaptive_rsx` keeps TWO per-thread rings of `period`
/// doubles -- one of prices, one of the rolling deviations built from them
/// (volatility_ratio_adaptive_rsx.rs:388, :394) -- and both are sized at
/// compile time, so the bound belongs to the kernel and not to the caller.
pub const VRARSX_MAX_PERIOD: usize = 512;

/// Must match `NEO_AT_MAX_PERIOD` in `kernels/cuda/alphatrend_kernel.cu`.
///
/// The swept period is BOTH the true-range window AND the MFI period
/// (alphatrend.rs:604-630), so the kernel carries a true-range ring of
/// `period + 1` and the MFI's two flow rings of `period` each. All three are
/// per-thread arrays sized at compile time.
pub const ALPHATREND_MAX_PERIOD: usize = 512;

/// `kernels/cuda/moving_averages/cora_wave_kernel.cu` and
/// `kernels/cuda/moving_averages/dma_kernel.cu`.
///
/// Both keep ONE per-thread ring whose depth is `round(sqrt(period))` --
/// cora_wave's smoothing WMA and dma's difference window. A 64-entry ring
/// admits every period up to 4160, because `round(sqrt(4160))` is 64 and
/// `round(sqrt(4161))` is 65. The bound therefore belongs to the compiled
/// kernel, and `CudaF64Indicators::sweep` refuses a larger period BY NAME
/// rather than truncating the window or moving the sweep to the host.
pub const CORA_WAVE_MAX_PERIOD: usize = 4160;
pub const DMA_MAX_PERIOD: usize = 4160;

// ---------------------------------------------------------- closer 6, round 3
/// Must match `LMA_MAX_PERIOD` in
/// `kernels/cuda/moving_averages/logarithmic_moving_average_kernel.cu`.
///
/// `compute_lma` (logarithmic_moving_average.rs:775) weights the window with
/// `1 / ln(max(i + steepness, 2))^2`, one logarithm per slot. Rebuilding that
/// vector inside the bar loop would run `period` logarithms PER BAR, so it is
/// built once into a per-thread array and its length is a property of the
/// COMPILED kernel. A larger period is REFUSED BY NAME.
pub const LMA_MAX_PERIOD: usize = 512;
/// Must match `WS_MAX_PERIOD` in
/// `kernels/cuda/moving_averages/wave_smoother_kernel.cu`; the array there is
/// one longer, because the wave window is `period + 1` (wave_smoother.rs:260).
///
/// Same reason as [`LMA_MAX_PERIOD`]: the weights are a sin/cos per slot
/// (:268) and are built once per thread rather than per bar.
pub const WS_MAX_PERIOD: usize = 512;

#[derive(Debug, Error)]
pub enum CudaF64IndicatorError {
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error("invalid input for {indicator}: {reason}")]
    InvalidInput {
        indicator: &'static str,
        reason: String,
    },
    #[error(
        "no f64 CUDA kernel for indicator '{indicator}'. This lane never falls back to f32 or to \
         the CPU: an f32 result would be a different number fed to the same threshold comparison, \
         and a CPU result would be reported as device work. Add an f64 kernel to \
         kernels/cuda/neoethos_f64_kernels.cu and register it in \
         indicators::dispatch::cuda_f64::F64_KERNELS, or route this indicator to the CPU \
         explicitly at the call site."
    )]
    NoF64Kernel { indicator: String },
    #[error("missing kernel symbol '{name}' in the neoethos f64 module")]
    MissingKernelSymbol { name: &'static str },
    #[error(
        "period {period} for '{indicator}' exceeds the f64 kernel's fixed bound of {max}. \
         Refusing to truncate the window or to move this sweep to the host."
    )]
    PeriodTooLarge {
        indicator: &'static str,
        period: usize,
        max: usize,
    },
    #[error(
        "out of memory for a SINGLE row of {indicator}: required={required} free={free} \
         headroom={headroom}. Peak memory here is a function of the hardware, so a sweep that \
         cannot fit one row cannot be chunked smaller."
    )]
    OutOfMemory {
        indicator: &'static str,
        required: usize,
        free: usize,
        headroom: usize,
    },
    #[error("launch config too large: grid=({gx},{gy},{gz}) block=({bx},{by},{bz})")]
    LaunchConfigTooLarge {
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
    },
}

/// Which of the ten f64 kernels to launch, and what host series it needs.
///
/// This enum IS the kernel-name resolution table: each variant maps to exactly
/// one `*_f64` entry point in the module. There is no string concatenation and
/// no `_f32`/`_f64` suffix switch, so an f64 request can never resolve to an
/// f32 symbol by accident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F64Kernel {
    /// Single price series. Sequential per column.
    Sma,
    Ema,
    Rsi,
    /// Single price series. Parallel over (combo, bar).
    Roc,
    Mom,
    /// high / low / close. Sequential per column.
    Atr,
    Adx,
    /// high / low / close. Parallel over (combo, bar).
    Willr,
    /// Single price series, which must be the CPU's SOURCE (hlc3 by default).
    /// Two-pass: sequential running mean, then parallel deviation.
    Cci,
    /// (typical price = hlc3, volume). Sequential per column.
    Mfi,

    // ---------------------------------------------------------------- batch 2
    //
    // The three that complete `hpc_ta::MULTI_PERIOD_IDS`. Ten of that table's
    // eighteen ids already had kernels; five (stoch, macd, bollinger_bands,
    // keltner, supertrend) are multi-output and emit no CPU column at all, so
    // these three were the entire remainder of the REACHABLE sweep.
    /// Single price series. Sequential. NeoEthos treats each requested period
    /// as the named `long_period` anchor and scales `short_period` with the
    /// documented/default 25:13 relation before comparing with the CPU row.
    Tsi,
    /// (close, volume). Sequential. PERIOD-INVARIANT — `compute_obv_batch`
    /// takes `|_params|`.
    Obv,
    /// (timestamps, close, volume). Sequential. PERIOD-INVARIANT — anchored by
    /// calendar bucket, never by a rolling window.
    Vwap,

    /// Single price series. Sequential per column.
    Wma,
    Wilders,
    Smma,
    Dema,
    Tema,
    Zlema,
    /// (close, volume). Sequential per column.
    Vwma,
    Efi,
    /// high / low / close. Sequential per column.
    Natr,
    Adxr,

    /// (high, low). Parallel over (combo, bar). PERIOD-INVARIANT.
    Medprice,
    /// high / low / close. Parallel over (combo, bar). PERIOD-INVARIANT.
    Wclprice,
    /// Single price series. Parallel over (combo, bar).
    Midpoint,
    Rocp,
    Rocr,
    /// (high, low). Parallel over (combo, bar).
    Midprice,

    // ---------------------------------------------------------------- shard 2
    //
    // From here down a variant's kernel may live in the indicator's OWN `.cu`
    // file rather than in `neoethos_f64_kernels.cu` — see [`F64Kernel::
    // module_stem`]. That is deliberate: the standing instruction is to fix the
    // kernels this crate already ships, in place, not to accumulate a parallel
    // file. Each such variant declares the module it belongs to and the engine
    // loads it.
    /// Single price series. Sequential per column. Kernel in
    /// `kernels/cuda/moving_averages/sqwma_kernel.cu`.
    Sqwma,
    Devstop,
    ChandelierExit,
    Minmax,
    Rsx,
    Trix,
    Vpt,
    Pvi,
    EhlersItrend,
    EhlersKama,
    Sama,
    Nama,
    Pwma,
    Tradjema,
    Maaq,
    Jma,
    Reflex,
    Gaussian,

    // ---------------------------------------------------------------- shard 6
    //
    // Every variant below launches an entry point that lives in the
    // INDICATOR'S OWN `.cu` file, beside the f32 entry points the f32 wrappers
    // still call -- see `module_stem`. That is the standing instruction: fix
    // the kernels this crate already ships, in place, rather than accumulate a
    // second implementation in a parallel file.
    //
    // Sequential unless noted. `Donchian` and `PercentileNearestRank` are the
    // two that are genuinely bar-parallel: donchian is a fresh max over its own
    // window with no carried state, and percentile-nearest-rank is an ORDER
    // STATISTIC, which selects an input value and therefore has no accumulation
    // order to preserve at all.
    Fwma,
    Hwma,
    Jsa,
    Nma,
    Swma,
    Trendflex,
    Vpwma,
    Cfo,
    Var,
    BollingerBandsWidth,
    DecOsc,
    Voss,
    PercentileNearestRank,
    TtmTrend,
    Vi,
    Cvi,
    CorrelHl,
    Aroonosc,
    ParkinsonVolatility,
    HistoricalVolatility,
    Donchian,

    // --------------------------------------------------------------- closer 4
    //
    // Seven more, every one written INTO THE FILE ITS INDICATOR ALREADY SHIPS
    // IN (see `module_stem`), against the CPU reference named in that file's
    // "f64 LANE  --  closer 4" header.
    //
    // All seven are SEQUENTIAL, for two different reasons that are worth
    // separating. `RandomWalkIndex`, `Qstick` and `RollingZScoreTrend`
    // genuinely carry state across bars -- a Wilder ATR recurrence, a sliding
    // sum updated as `(sum + new) - old`, and a rolling (sum, sumsq) pair plus
    // a carried `smoothed` -- so a bar-parallel form would change the
    // rounding. `PsychologicalLine`, `RankCorrelationIndex`, `Sinwma` and
    // `Srwma` rebuild their window at every bar and COULD be bar-parallel;
    // they are sequential here because each must also know how many
    // CONSECUTIVE FINITE values precede the bar -- the CPU's stream-reset
    // semantics -- or, for `Srwma`, because the eight-accumulator fold is
    // per-bar work whose order a split across threads would not preserve.
    //
    // NONE is period-invariant: every one of the seven CPU batch functions
    // reads the swept window parameter -- `length` for psychological_line
    // (cpu_batch.rs:11906), rank_correlation_index and random_walk_index
    // (:10337), `period` for qstick (:3738), sinwma and srwma, and
    // `lookback_period` for rolling_z_score_trend (:8033).
    //
    // NONE declares a `max_period`: all seven read their window straight out
    // of the resident input in global memory and keep no per-thread ring, so
    // there is no compile-time bound for an oversized period to be refused
    // against.
    //
    // TWO SERVE MULTI-OUTPUT INDICATORS whose CPU batch does NOT accept
    // `output_id == "value"`, which is why each is named here rather than left
    // to be discovered: `RandomWalkIndex` emits `high` (the batch accepts only
    // "high"/"low", cpu_batch.rs:10352-10359) and `RollingZScoreTrend` emits
    // `zscore` (only "zscore"/"momentum", :8046-8053). A parity run must ask
    // the CPU for that output id explicitly.
    /// Single price series, CPU source `close`. Rolling up-close count.
    PsychologicalLine,
    /// Single price series, CPU source `close`. ORDER STATISTIC -- ranks come
    /// from counting comparisons, exactly as `compute_window_rci` does.
    RankCorrelationIndex,
    /// (open, high, low, close), of which only OPEN and CLOSE are read.
    /// Declared `Ohlc4` because the resident upload already carries all four
    /// and the four-pointer launch arm already exists.
    Qstick,
    /// Single price series, CPU source `close`. Sine-weighted convolution.
    Sinwma,
    /// Single price series, CPU source `close`. Sqrt-weighted convolution with
    /// the CPU's eight independent accumulators.
    Srwma,
    /// Single price series, CPU source `close`. Emits the ZSCORE series.
    RollingZScoreTrend,
    /// high / low / close. Emits the HIGH series.
    RandomWalkIndex,

    // ---------------------------------------------------------------- shard 1
    //
    // Nineteen more, every one written INTO THE FILE ITS INDICATOR ALREADY
    // SHIPS IN (see `module_stem`) and against the `*_scalar` CPU reference
    // named in that file's `S1 f64 LANE` header.
    //
    // All nineteen are SEQUENTIAL: each carries state across bars -- an
    // EMA/SMMA recurrence, a sliding sum that is accumulation-order dependent,
    // or a monotonic deque. None can be made bar-parallel without changing the
    // rounding, which is the whole reason this lane exists.
    /// Single price series, CPU source `close`.
    Apo,
    Vidya,
    Gatorosc,
    Ppo,
    Pma,
    Kama,
    Linreg,
    Edcf,
    Alma,
    Hma,
    /// Single price series whose CPU source is `hl2`, NOT close.
    Kurtosis,
    Alligator,
    /// (close, volume).
    Nvi,
    /// (high, low), with no close.
    Fisher,
    Safezonestop,
    /// high / low / close.
    Chop,
    Stochf,
    /// (high, low, volume) -- close is never read. `emv` alone.
    Emv,
    /// (high, low, close, volume). `kvo` alone.
    Kvo,

    // ---------------------------------------------------------------- shard 4
    //
    // Twenty-four more, every one written INTO THE FILE ITS INDICATOR ALREADY
    // SHIPS IN (see `module_stem`), against the CPU reference named in that
    // file's `S4 f64 LANE` / `NEOETHOS f64 LANE` header.
    //
    // All twenty-four are SEQUENTIAL. Nothing in this shard is bar-parallel:
    // every one carries an EMA/Wilder recurrence, a rolling sum whose
    // accumulation order is load-bearing, a monotonic deque, or a prefix sum
    // accumulated from index 0.
    //
    // TEN OF THEM ARE PERIOD-INVARIANT, and that is FAITHFUL rather than lazy.
    // Their CPU batch functions read named parameters — `fast_period`/
    // `slow_period`/`signal_period` for macd, `p`/`x`/`q` for cksp,
    // `rsi_period`/`wma_period` for ift_rsi, `short_range`/`long_range` for
    // vpci, `channel_length`/`average_length`/`ma_length` for wavetrend,
    // `vis_atr`/`vis_std`/`sed_atr`/`sed_std` for damiani_volatmeter,
    // `length` plus four multipliers for ttm_squeeze — and NEVER `period`. A
    // caller sweeping `[7,21,50,100,200]` gets five identical CPU columns for
    // each of them, so the kernel emits five identical rows and
    // `is_period_invariant` says so. `ad` and `acosc` have no length parameter
    // at all. Inventing a mapping from the swept int onto one of several named
    // periods would compute something the CPU never computes.
    /// Single price series, CPU source `close`.
    Er,
    LinearregAngle,
    LinearregIntercept,
    Highpass2Pole,
    Supersmoother3Pole,
    Cwma,
    Cmo,
    Stddev,
    Ui,
    BollingerBands,
    /// Single price series. Builds the Hann weight recurrence into a per-thread
    /// array because the CPU REVERSES the weights and the rotation is not
    /// bit-reproducible backwards — hence the only `max_period` in this shard.
    Ehma,
    /// Single price series. PERIOD-INVARIANT — macd reads fast/slow/signal.
    Macd,
    /// Single price series. PERIOD-INVARIANT — rsi_period/wma_period.
    IftRsi,
    /// Single price series. PERIOD-INVARIANT — four window parameters, and the
    /// batch passes it as ONE slice, so high == low == close.
    DamianiVolatmeter,
    /// Single price series whose CPU source is `hlc3`, NOT close.
    /// PERIOD-INVARIANT — channel/average/ma lengths.
    Wavetrend,
    /// high / low / close.
    Dx,
    Frama,
    /// high / low / close, first-valid scanned on CLOSE ALONE.
    /// PERIOD-INVARIANT — p/x/q.
    Cksp,
    /// high / low / close, first-valid scanned on CLOSE ALONE.
    /// PERIOD-INVARIANT — length is pinned at 20 because that is the only
    /// value for which the CPU takes `ttm_squeeze_scalar_classic`.
    TtmSqueeze,
    /// (high, low), with no close.
    Mass,
    /// (high, low). PERIOD-INVARIANT and FIRST-VALID-IGNORED by construction.
    Aroon,
    Acosc,
    /// (close, volume). PERIOD-INVARIANT — short_range/long_range.
    Vpci,
    /// (high, low, close, volume). PERIOD-INVARIANT and FIRST-VALID-IGNORED —
    /// `ad_scalar` starts at index 0 and takes no parameters at all.
    Ad,
    /// (open, close, volume) -- `dvdiqqe` alone. PERIOD-SWEPT: its CPU batch
    /// reads a parameter literally named `period`, unlike the ten invariant
    /// variants above.
    Dvdiqqe,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT: cci_cycle's
    /// CPU batch reads `length` (10) and `factor` (0.5), and `length` is
    /// pinned at 10 because `cci_cycle_compute_from_parts:526` routes
    /// `length > 16` to a DIFFERENT function.
    CciCycle,

    // ---------------------------------------------------------------- shard 3
    //
    // Twenty-five more, every one written INTO THE FILE ITS INDICATOR ALREADY
    // SHIPS IN (see `module_stem`), against the CPU reference named in that
    // file's `S3 f64 LANE` header.
    //
    // All twenty-five are SEQUENTIAL. Every one carries state across bars: an
    // EMA/Wilder recurrence, an Ehlers IIR, a rolling sum whose accumulation
    // order is load-bearing, or a monotone-deque extreme. None can be made
    // bar-parallel without changing the rounding.
    //
    // EIGHT ARE PERIOD-INVARIANT and that is faithful, not lazy: their CPU
    // batch functions read named parameters -- `short_period`/`long_period`
    // for ao, `r`/`s`/`u` for dti, `fast_k_period`/`slow_k_period` for kdj,
    // `channel_length`/`average_length` for wto, `range_size`/`range_period`
    // for range_filter, `acceleration`/`maximum` for sar, `fast_limit`/
    // `slow_limit` for mama -- and NEVER `period`. wad has no parameter at
    // all. A sweep of five periods gets five identical CPU columns, so the
    // kernel emits five identical rows.
    //
    // SEVEN SERVE MULTI-OUTPUT INDICATORS, and each emits the column the CPU
    // batch produces for `output_id == "value"`: di -> plus, kdj -> k,
    // aso -> bulls, wto -> wavetrend1, range_filter -> filter,
    // correlation_cycle -> real, mama -> mama. Never a different one
    // silently.
    Deviation,
    MeanAd,
    Ao,
    LinearregSlope,
    Tsf,
    Highpass,
    Decycler,
    Supersmoother,
    Tilson,
    Wad,
    Sar,
    Dti,
    Zscore,
    Pfe,
    Chande,
    Di,
    Kdj,
    Aso,
    Wto,
    RangeFilter,
    CorrelationCycle,
    Mama,
    VolumeAdjustedMa,
    ReverseRsi,
    EhlersEcema,

    // --------------------------------------------------------------- closer 6
    //
    // Written INTO THE FILE EACH INDICATOR ALREADY SHIPS IN (see
    // `module_stem`), against the CPU reference named in that file's
    // "f64 LANE  --  closer 6" header. All four are SEQUENTIAL.
    //
    // `Emd` and `Keltner` are PERIOD-SWEPT: their CPU batch functions read a
    // parameter literally named `period` (cpu_batch.rs:14532, :6212). `Stoch`
    // is PERIOD-INVARIANT: its batch reads `fastk_period`/`slowk_period`/
    // `slowd_period` and never `period` (cpu_batch.rs:5580-5582), so a sweep
    // of five periods gets five identical CPU columns and the kernel emits
    // five identical rows.
    //
    // ALL THREE SERVE MULTI-OUTPUT INDICATORS, and each emits the column the
    // CPU batch produces for `output_id == "value"`: emd -> upperband
    // (cpu_batch.rs:14554), keltner -> upper_band (:6232), stoch -> k
    // (:5603). Never a different one silently.
    /// (high, low), with NO close. `emd_scalar_into` forms
    /// `price = (h + l) * 0.5` itself; the hl2 path is unreachable from the
    /// batch, which builds the input with `from_slices`.
    Emd,
    /// high / low / close, first-valid scanned on CLOSE ALONE.
    Keltner,
    /// high / low / close. PERIOD-INVARIANT -- fastk/slowk/slowd.
    Stoch,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT --
    /// `bandwidth`/`multiplier`/`lookback`, never `period`.
    ///
    /// SEQUENTIAL despite the kernel regression itself being bar-parallel:
    /// the band is `y +/- mae` where `mae` is a sliding sum of the last 499
    /// absolute residuals (`rbuf`/`rsum`,
    /// nadaraya_watson_envelope.rs:523-525), and that sum is carried across
    /// bars with `rsum -= old; rsum += resid` rather than recomputed.
    NadarayaWatsonEnvelope,

    // --------------------------------------------------------------- closer 3
    //
    // Six more, every one written INTO THE FILE ITS INDICATOR ALREADY SHIPS IN
    // (see `module_stem`), against the CPU reference named in that file's
    // "f64 LANE  --  closer C3" header.
    //
    // All six are registered SEQUENTIAL, and for five of them that is forced: a
    // phasor rotation, five interlocking Ehlers IIRs, a rolling sum with an
    // add-on-entry / subtract-on-exit accumulator, and a windowed order
    // statistic. `Marketefi` is the exception -- its value is POINTWISE and
    // would be bar-parallel -- but the lane's bar-parallel launch arm REFUSES
    // the `HighLowVolume` shape, so it is registered sequential rather than
    // widening shared launch code this change does not own.
    //
    // FOUR ARE PERIOD-INVARIANT, and that is faithful rather than lazy. Their
    // CPU batch functions read named parameters -- `domestic_cycle_length` for
    // l1_ehlers_phasor (cpu_batch.rs:10224), `source`/`smooth_period` for
    // l2_ehlers_signal_to_noise (:9894-9896), `length`/`ma_type` for
    // kairi_relative_index (:7037-7038) -- and NEVER `period`, while
    // `marketefi` has no length parameter at all and no `compute_*_batch` entry
    // in cpu_batch.rs. A caller sweeping `[7,21,50,100,200]` gets five
    // identical CPU columns for each of them, so the kernel emits five
    // identical rows and `is_period_invariant` says so.
    /// Single price series, CPU source `close`. PERIOD-INVARIANT.
    L1EhlersPhasor,
    /// (high, low), with NO close. The CPU source is `hl2`, and
    /// `Candles::compute_hl2` (data_loader.rs:168) defines it as `(h + l) / 2.0`
    /// -- so the kernel forms it from the pair rather than being handed a third
    /// series that would carry close's first-valid. PERIOD-INVARIANT.
    L2EhlersSignalToNoise,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT, and
    /// FIRST-VALID IGNORED: `compute_default_sma50_into`
    /// (kairi_relative_index.rs:732) fills the output with NaN and walks from
    /// index 0, so it never consults a first-valid index at all.
    KairiRelativeIndex,
    /// Single price series, CPU source `close`. PERIOD-SWEPT.
    LinearCorrelationOscillator,
    /// Single price series, CPU source `close`. PERIOD-SWEPT. An ORDER
    /// STATISTIC over a per-thread window copy, hence a `max_period`.
    MediumAd,
    /// (high, low, volume) with NO close -- the same shape `emv` takes.
    /// PERIOD-INVARIANT.
    Marketefi,
    /// Single price series, CPU source `close`. PERIOD-SWEPT. Emits the
    /// `line` column, which is what `output_id == "value"` resolves to
    /// (cpu_batch.rs:11713).
    MomentumRatioOscillator,
    /// (close, volume). PERIOD-INVARIANT -- `obv_length`/`ema_length`. Emits
    /// the `line` column (cpu_batch.rs:9768).
    OnBalanceVolumeOscillator,

    // ------------------------------------------------------------- closer 5
    //
    // Eleven more, every one written INTO THE FILE ITS INDICATOR ALREADY
    // SHIPS IN (see `module_stem`), against the CPU reference named in that
    // file's NEOETHOS f64 LANE header.
    //
    // All eleven are SEQUENTIAL. Every one carries state across bars: a lag
    // ring read in full at every bar (the velocity family), a mean-seeded
    // EMA cascade, a monotone deque, or an INCREMENTAL window sum whose
    // accumulation order is load-bearing.
    //
    // NINE ARE PERIOD-INVARIANT and that is faithful rather than lazy: their
    // CPU batch functions read NAMED parameters -- length/smooth_length for
    // the velocity family, length for trend_direction_force_index,
    // trend_continuation_factor, trend_trigger_factor and
    // volume_zone_oscillator, short_period/long_period for vosc, and
    // timeperiod1/timeperiod2/timeperiod3 for ultosc -- and NEVER period. A
    // sweep of five periods gets five identical CPU columns, so the kernel
    // emits five identical rows.
    //
    // THE TWO THAT ARE NOT: trima has no cpu_batch arm at all, so its oracle
    // is the single-series function, which reads `period`; and
    // volume_weighted_rsi calls combo_periods(.., "period", 14) outright at
    // cpu_batch.rs:6787.
    /// Single price series whose CPU default source is `hlcc4`, NOT close.
    Velocity,
    VelocityAccelerationIndicator,
    VelocityAccelerationConvergenceDivergenceIndicator,
    /// Single price series, CPU source `close`.
    TrendDirectionForceIndex,
    TrendContinuationFactor,
    Trima,
    /// (high, low), with no close.
    TrendTriggerFactor,
    /// (close, volume).
    VolumeWeightedRsi,
    VolumeZoneOscillator,
    /// VOLUME alone -- `vosc` reads no price series at all.
    Vosc,
    /// high / low / close.
    Ultosc,

    // ------------------------------------------------------------ closer 2
    //
    // Fourteen more, every entry point written INTO THE .cu FILE ITS
    // INDICATOR ALREADY SHIPS IN (see `module_stem`), beside the f32 entry
    // points the f32 wrappers still call, and against the CPU reference
    // named in that file's `NEOETHOS f64 LANE` header.
    //
    // All fourteen are SEQUENTIAL. Every one carries state across bars: an
    // Ehlers IIR, a Wilder or EMA recurrence, a rolling sum whose
    // accumulation order is load-bearing, a monotone-deque extreme, or a
    // prefix sum accumulated from index 0.
    //
    // ELEVEN ARE PERIOD-INVARIANT and that is faithful, not lazy: their CPU
    // batch functions read named parameters -- `length` for
    // ehlers_detrending_filter / fractal_dimension_index /
    // gopalakrishnan_range_index / emd_trend, `alpha` (+ `cutoff`) for the
    // two Ehlers cycle indicators, `lambda` for ewma_volatility, `lookback`
    // for garman_klass_volatility, `length_ma`/`length_signal` for
    // impulse_macd, `factor`/`slope`/`width_percent` for hypertrend -- and
    // NEVER `period`. `ehlers_pma` is invariant for a stronger reason still:
    // `ma_batch.rs:1679` computes it ONCE and repeats the row.
    // The three that ARE period-swept read a parameter literally named
    // `period`: epma (:1881), fosc (cpu_batch.rs:3117), eri (:14727).
    EhlersDetrendingFilter,
    EhlersSimpleCycleIndicator,
    EhlersSmoothedAdaptiveMomentum,
    EwmaVolatility,
    FractalDimensionIndex,
    GopalakrishnanRangeIndex,
    GarmanKlassVolatility,
    ImpulseMacd,
    Hypertrend,
    EmdTrend,
    Epma,
    Fosc,
    EhlersPma,
    Eri,

    // ---------------------------------------------------------------- closer 1
    //
    // Twenty more, every one launching an `<id>_neo_batch_f64` written INTO the
    // file its indicator already ships in (see `module_stem`).
    //
    // NONE OF THEM COULD REUSE THE ENTRY POINT ALREADY IN THAT FILE. Those take
    // several int arrays, emit several matrices and never take `first_valid` --
    // `absolute_strength_index_oscillator_batch_f64` takes `ema_lengths` and
    // `signal_lengths` and writes three outputs, for example. This lane
    // launches (series..., n, periods, n_combos, first_valid, out), so a
    // variant pointing at one of the old symbols would have read the stack.
    //
    // All twenty are SEQUENTIAL per column. Even the three whose per-bar value
    // carries no state -- `Bop`, and the window rescans in
    // `DonchianChannelWidth` and `Cg` -- are launched one thread per combo,
    // because that is the shape this lane launches; the loop over bars is the
    // thread body.
    //
    // FIFTEEN ARE PERIOD-INVARIANT: their CPU batch functions read NAMED
    // parameters and never `period`. The five that are not are
    // `BullPowerVsBearPower`, `Cg`, `Dm`, `DonchianChannelWidth` and `Dpo`.
    AbsoluteStrengthIndexOscillator,
    AccumulationSwingIndex,
    AdaptiveBandpassTriggerOscillator,
    AdaptiveBoundsRsi,
    AdaptiveMacd,
    AdaptiveMomentumOscillator,
    AdvanceDeclineLine,
    AndeanOscillator,
    AtrPercentile,
    Bop,
    BullPowerVsBearPower,
    Cg,
    Coppock,
    DailyFactor,
    DecisionpointBreadthSwenlinTradingOscillator,
    DidiIndex,
    DisparityIndex,
    Dm,
    DonchianChannelWidth,
    Dpo,

    // ---------------------------------------------------------- closer 2b
    //
    // Four more, same construction as the fourteen above.
    // `ehlers_fm_demodulator` is PERIOD-SWEPT (cpu_batch.rs:3811 reads a
    // parameter literally named `period`, default 30) and reads OPEN and
    // CLOSE only -- high and low are length-checked and discarded.
    EhlersFmDemodulator,
    ForwardBackwardExponentialOscillator,
    GmmaOscillator,
    EvasiveSupertrend,

    // ------------------------------------------------- closer 6, round 2
    //
    // Eight more, every entry point written INTO the `.cu` file its indicator
    // already ships in (see `module_stem`), beside the f32 entry points the
    // f32 wrappers still call, and against the CPU reference named in that
    // file's "NEOETHOS f64 LANE  --  closer 6" header.
    //
    // All eight are SEQUENTIAL. Every one carries state across bars: a Wilder
    // or EMA recurrence, a rolling (sum, sumsq) pair whose accumulation order
    // is load-bearing, an adaptive-period state machine, or a rank count
    // rolled rather than recomputed.
    //
    // FIVE ARE PERIOD-INVARIANT and that is faithful, not lazy: their CPU
    // batch functions read NAMED parameters and never `period` --
    // `lookback`/`k_override`/`k` for yang_zhang_volatility
    // (cpu_batch.rs:8311), `rsi_period`/`smoothing_factor`/`fast_factor` for
    // qqe (:15880), `rsi_period`/`stoch_period`/`k`/`d` for srsi (:6308),
    // `min_period`/`max_period`/`matype`/`devtype` for vlma (:15734), and
    // `fast_period`/`slow_period`/`k_period`/`d_period` for stc (:16571). A
    // sweep of five periods gets five identical CPU columns, so each kernel
    // writes five identical rows.
    //
    // THE THREE THAT ARE SWEPT read a parameter literally named `period`:
    // msw (cpu_batch.rs:15582), rvi (:16608) and net_myrsi (:16704).
    //
    // FIVE SERVE MULTI-OUTPUT INDICATORS, and each emits the column the CPU
    // batch produces for `output_id == "value"`: yang_zhang_volatility -> yz,
    // qqe -> fast, srsi -> k, msw -> sine, stc -> value. Never a different one
    // silently.
    /// Single price series, CPU source `close`. PERIOD-SWEPT.
    Msw,
    /// (open, high, low, close). PERIOD-INVARIANT. First-valid is
    /// `Ohlc4AllNonNan` -- `first_valid_ohlc`, yang_zhang_volatility.rs:411.
    YangZhangVolatility,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT.
    Qqe,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT.
    Srsi,
    /// Single price series, CPU source `close`. PERIOD-SWEPT.
    Rvi,
    /// Single price series, CPU source `close`. PERIOD-SWEPT. Keeps two
    /// `period`-wide per-thread rings, hence the only `max_period` in this
    /// round.
    NetMyrsi,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT.
    Vlma,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT.
    Stc,

    // ------------------------------------------------------ closer 2, round 2
    //
    // Ten indicators that already shipped an f64 kernel in their own `.cu`
    // file and had NO `F64Kernel` variant and NO `F64_KERNELS` row, so the lane
    // could not reach them at all. Each now carries an `<id>_neo_batch_f64`
    // entry point written INTO that same file (search it for
    // "f64 LANE  --  closer 2, round 2") against the CPU reference named in the
    // file's header.
    //
    // NOT ONE of them could reuse the entry point that was already there. Those
    // take host-solved weight matrices (`sgf`, `polynomial_regression_
    // extrapolation`, `hull_butterfly_oscillator`), per-row parameter arrays
    // (`lrsi`, `kaufmanstop`, `range_oscillator`), or write several output
    // matrices in two passes (`dual_ulcer_index`, `pivot`), and none of them
    // takes `first_valid`. This lane launches
    // (series..., n, periods, n_combos, first_valid, out) and allocates ONE
    // matrix, so a variant pointing at one of the old symbols would have read
    // the stack.
    //
    // ALL TEN ARE SEQUENTIAL. Six carry genuine state across bars -- a sliding
    // sum (`dual_ulcer_index`, `kaufmanstop`), two ATR recurrences plus a
    // sticky trend (`range_oscillator`), a four-stage Laguerre filter (`lrsi`),
    // a one-fma EMA (`mwdx`), a cumulative-absolute running sum and a signal
    // state machine (`hull_butterfly_oscillator`), and an eleven-scalar market
    // structure machine whose `ts` reads its own previous value
    // (`market_structure_trailing_stop`). The remaining three rebuild their
    // window at every bar and COULD be bar-parallel; they are sequential
    // because that is the shape this lane launches, and the bar loop is the
    // thread body.
    //
    // NONE declares a `max_period`. Every one of them reads its window straight
    // out of the resident input and recomputes rather than caches: the sgf and
    // polynomial-regression weight vectors are re-derived from the solved
    // coefficients inside the dot loop, the hull-butterfly coefficients from
    // their closed form, and the dual-ulcer sliding sum's leaving term from the
    // data. So there is no compile-time bound for an oversized period to be
    // refused against, and NEVER-OOM holds by construction.
    //
    // THREE ARE PERIOD-INVARIANT, and that is the indicator rather than a
    // shortcut: `mwdx` has only a `factor` (mwdx.rs:80), `lrsi` only an `alpha`
    // (cpu_batch.rs:3481), and `pivot` only a `mode` (:16734) -- an integer
    // selecting WHICH formula runs, which a period list cannot stand in for.
    //
    // THREE SERVE MULTI-OUTPUT INDICATORS whose CPU batch does not accept
    // `output_id == "value"`, or accepts it only as an alias, so a parity run
    // must ask for the right column by name:
    // `HullButterflyOscillator` emits "oscillator" (cpu_batch.rs:8751-8762
    // accepts nothing else), `RangeOscillator` emits "oscillator" ("value" is
    // an alias, :16044-16049) and `MarketStructureTrailingStop` emits
    // "trailing_stop" ("value" is an alias, :7197-7201). `DualUlcerIndex` emits
    // "long_ulcer", of which "value" is also an alias (:6700-6706), and `Pivot`
    // emits "pp" (:16743-16745).
    /// Single price series, CPU source `close`. PERIOD-INVARIANT -- `factor`,
    /// not a period.
    Mwdx,
    /// (high, low), no close. PERIOD-INVARIANT -- `alpha`, not a period.
    Lrsi,
    /// high / low / close. PERIOD-INVARIANT -- `mode`, not a period. Emits the
    /// `pp` series. Declared Hlc and not Ohlc4 because the mode-3 arm never
    /// reads open.
    Pivot,
    /// (high, low). PERIOD-SWEPT.
    Kaufmanstop,
    /// Single price series, CPU source `close`. PERIOD-SWEPT. Solves its own
    /// 3x3 Savitzky-Golay normal system per row.
    Sgf,
    /// Single price series, CPU source `close`. PERIOD-SWEPT. Solves its own
    /// 4x4 polynomial normal system per row.
    PolynomialRegressionExtrapolation,
    /// Single price series, CPU source `close`. PERIOD-SWEPT. Emits the
    /// LONG ULCER series.
    DualUlcerIndex,
    /// Single price series, CPU source `close`. PERIOD-SWEPT. Emits the
    /// OSCILLATOR series.
    HullButterflyOscillator,
    /// high / low / close. PERIOD-SWEPT. Emits the OSCILLATOR series.
    RangeOscillator,
    /// open / high / low / close, of which the value loop reads high, low and
    /// close -- open is an INPUT to the validity predicate that segments the
    /// series into runs. PERIOD-SWEPT. Emits the TRAILING STOP series.
    MarketStructureTrailingStop,

    // ------------------------------------------------ closer 3, round 2
    //
    // Twenty-five more, every entry point written INTO THE .cu FILE ITS
    // INDICATOR ALREADY SHIPS IN (see `module_stem`), beside the entry points
    // the existing wrappers still call, and against the CPU reference named in
    // that file's "NEOETHOS f64 LANE  --  closer 3" header.
    //
    // ALL TWENTY-FIVE ARE SEQUENTIAL. Every one carries state across bars: a
    // Wilder or EMA recurrence, an Ehlers IIR, a monotone deque, a sliding sum
    // maintained with subtract-then-add, a ratchet, or a state machine. None
    // can be made bar-parallel without changing the rounding, which is the
    // whole reason this lane exists.
    //
    // TWENTY-TWO ARE PERIOD-INVARIANT and that is FAITHFUL, not lazy: their
    // CPU batch functions read NAMED parameters -- `length`/`annual_length`
    // for historical_volatility_percentile, `hv_length`/`rank_length` for
    // historical_volatility_rank, `alpha` for the two Ehlers adaptive
    // indicators, `entry_level`/`exit_level` for
    // cyberpunk_value_trend_analyzer, `left_bars`/`right_bars`/`level` for
    // fibonacci_trailing_stop, and so on -- and NEVER `period`. A caller
    // sweeping `[7,21,50,100,200]` gets five identical CPU columns for each of
    // them, so the kernel emits five identical rows and `is_period_invariant`
    // says so. The THREE that are genuinely period-swept read a parameter
    // literally named `period`: `BullsVBears` (cpu_batch.rs:11153),
    // `CandleStrengthOscillator` (:7514) and `DirectionalImbalanceIndex`
    // (:7437).
    //
    // EVERY ONE DECLARES `F64FirstValidRule::Ignored`, and that is a contract
    // the kernels honour rather than a shrug. Each of these CPU references
    // either has NO warmup index at all (it emits from bar 0 and RESETS its
    // state at every invalid bar, so a global first-valid would be wrong after
    // the first hole), or scans with a predicate no declared rule expresses --
    // `is_finite` over a triple, over a DERIVED midpoint series, or "strictly
    // positive as well as finite". Deriving it inside the kernel keeps the two
    // halves of one rule in one place; the alternative is a rule that names a
    // different bar than the CPU and shifts the whole series.
    //
    // MOST SERVE MULTI-OUTPUT INDICATORS, and each emits the column the CPU
    // batch produces for `output_id == "value"`: adjustable_ma... -> ma,
    // autocorrelation_indicator -> filtered, cycle_channel_oscillator -> fast,
    // adaptive_schaff_trend_cycle -> stc, exponential_trend -> uptrend_base,
    // fvg_positioning_average -> bull_average, hema_trend_levels -> fast_hema,
    // fibonacci_trailing_stop -> trailing_stop, demand_index -> demand_index,
    // cyberpunk_value_trend_analyzer -> value_trend,
    // directional_imbalance_index -> up, intraday_momentum_index -> imi,
    // ehlers_adaptive_cg -> cg, ehlers_adaptive_cyber_cycle -> cycle,
    // ehlers_autocorrelation_periodogram -> dominant_cycle,
    // ehlers_linear_extrapolation_predictor -> prediction,
    // grover_llorens_cycle_oscillator -> value, historical_volatility_rank ->
    // hvr. Never a different one silently.
    //
    // TWO HAVE NO `value` ALIAS AT ALL and are named here for that reason: the
    // `historical_volatility_percentile` batch accepts only `hvp` / `hvp_sma`
    // (cpu_batch.rs:9681-9690) and the
    // `ehlers_data_sampling_relative_strength_indicator` batch only `ds_rsi` /
    // `original_rsi` / `signal` (:8132-8144). A parity run must ask the CPU
    // for `hvp` and `ds_rsi` explicitly; these kernels emit those columns.
    /// Single price series, CPU source `close`. PERIOD-SWEPT -- the kernel
    /// already carried the lane ABI, so this row is registration only.
    VerticalHorizontalFilter,
    /// high / low / close, of which only CLOSE is convolved -- high and low
    /// are inputs to the validity scan that sets the warmup.
    AdjustableMaAlternatingExtremities,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT.
    AutocorrelationIndicator,
    HistoricalVolatilityRank,
    /// Emits `hvp`; the CPU batch has no `value` alias.
    HistoricalVolatilityPercentile,
    /// (high, low), with no close. PERIOD-SWEPT.
    DirectionalImbalanceIndex,
    /// high / low / close, source pinned to `close` by the CPU default.
    CycleChannelOscillator,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT.
    DynamicMomentumIndex,
    /// Single price series whose CPU source is `hl2`, NOT close.
    EhlersAdaptiveCg,
    EhlersAdaptiveCyberCycle,
    /// (open, high, low, close), of which only OPEN and CLOSE are read.
    /// Emits `ds_rsi`; the CPU batch has no `value` alias.
    EhlersDataSamplingRelativeStrengthIndicator,
    /// high / low / close. PERIOD-INVARIANT.
    ExponentialTrend,
    GeometricBiasOscillator,
    /// (open, high, low, close), of which only OPEN and CLOSE are read.
    IntradayMomentumIndex,
    /// high / low / close. PERIOD-SWEPT.
    BullsVBears,
    /// open / high / low / close. PERIOD-SWEPT.
    CandleStrengthOscillator,
    /// open / high / low / close. PERIOD-INVARIANT.
    CyberpunkValueTrendAnalyzer,
    FvgPositioningAverage,
    HemaTrendLevels,
    /// high / low / close. PERIOD-INVARIANT.
    FibonacciTrailingStop,
    /// open / high / low / close; at the pinned source only high, low and
    /// close are read. PERIOD-INVARIANT.
    GroverLlorensCycleOscillator,
    /// (high, low, close, volume). PERIOD-INVARIANT.
    DemandIndex,
    /// high / low / close. PERIOD-INVARIANT.
    AdaptiveSchaffTrendCycle,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT.
    EhlersLinearExtrapolationPredictor,
    EhlersAutocorrelationPeriodogram,
    /// open / high / low / close. PERIOD-INVARIANT. A per-column STATE
    /// MACHINE; emits the `bullish_high` column, which is what the CPU batch
    /// produces -- it has no `value` alias at all (cpu_batch.rs:12880-12905).
    IctPropulsionBlock,

    // ------------------------------------------------ closer 4, round 2
    //
    // Fifteen more, every entry point written INTO THE .cu FILE ITS
    // INDICATOR ALREADY SHIPS IN (see `module_stem`), beside the entry
    // points that file already had, and against the CPU reference named in
    // that file's `NEOETHOS f64 LANE  --  closer 4` header.
    //
    // ALL FIFTEEN ARE SEQUENTIAL. Every one carries state across bars: a
    // Wilder or EMA recurrence, a rolling sum or (sum, sumsq) pair whose
    // accumulation order is load-bearing, a monotone deque, an
    // incrementally sorted window, or -- for `SmoothTheilSen` and
    // `MonotonicityIndex` -- a per-bar order statistic over a window the
    // previous bar built.
    //
    // ALL FIFTEEN ARE PERIOD-INVARIANT, and that is FAITHFUL rather than
    // lazy. Every one of the fifteen CPU batch functions reads NAMED
    // parameters and NEVER one called `period`: sma_period1..4 /
    // roc_period1..4 / signal_period for kst (cpu_batch.rs:15128),
    // length + smooth_length for rolling_skewness_kurtosis (:7990),
    // rsi_length / stoch_length / smooth_length for premier_rsi_oscillator
    // (:8641), length for pretty_good_oscillator (:11744), length +
    // eval_period for price_density_market_noise (:11846), length +
    // smooth_length for projection_oscillator (:7134), length / factor /
    // smooth / weight for qqe_weighted_oscillator (:15917), length / mode /
    // index_smooth for monotonicity_index (:9420), length + source_mode for
    // market_meanness_index (:11597), length / multiplier /
    // use_exponential / bands_style / atr_length for
    // keltner_channel_width_oscillator (:13077), length / norm_length /
    // use_norm_hyperbolic for leavitt_convolution_acceleration (:13196),
    // lookback + signal_length for rogers_satchell_volatility (:6608),
    // length / offset / multiplier / four styles for smooth_theil_sen
    // (:11213), deviations / short_cycle / long_cycle / sensitivity for
    // kase_peak_oscillator_with_divergences (:13695), and swing_size /
    // basis_length / atr_length / atr_smooth / vol_mult for
    // market_structure_confluence (:16101). A caller sweeping
    // [7,21,50,100,200] gets five IDENTICAL CPU columns for each of them,
    // so each kernel emits five identical rows and `is_period_invariant`
    // says so. Mapping the swept int onto one of the named windows would
    // compute something the CPU never computes.
    //
    // FOUR SERVE MULTI-OUTPUT INDICATORS WHOSE CPU BATCH HAS NO "value"
    // ARM, so a parity run must ask for the named column explicitly:
    // `RollingSkewnessKurtosis` emits `skewness` (the batch accepts only
    // "skewness"/"kurtosis", cpu_batch.rs:8007), `MonotonicityIndex`
    // emits `index` (:9450), and `MarketStructureConfluence` emits `basis`
    // (:16145). The rest emit the column "value" resolves to: kst -> line,
    // pretty_good_oscillator -> values, price_density_market_noise ->
    // price_density, projection_oscillator -> pbo, qqe_weighted_oscillator
    // -> rsi, market_meanness_index -> mmi,
    // keltner_channel_width_oscillator -> kbw,
    // leavitt_convolution_acceleration -> conv_acceleration,
    // rogers_satchell_volatility -> rs, smooth_theil_sen -> value,
    // kase_peak_oscillator_with_divergences -> oscillator,
    // premier_rsi_oscillator -> its single `values` series.
    KasePeakOscillatorWithDivergences,
    KeltnerChannelWidthOscillator,
    Kst,
    LeavittConvolutionAcceleration,
    MarketMeannessIndex,
    MarketStructureConfluence,
    MonotonicityIndex,
    PremierRsiOscillator,
    PrettyGoodOscillator,
    PriceDensityMarketNoise,
    ProjectionOscillator,
    QqeWeightedOscillator,
    RogersSatchellVolatility,
    RollingSkewnessKurtosis,
    SmoothTheilSen,

    // ------------------------------------------------ closer 2, round 3
    //
    // Ten indicators whose `.cu` file ALREADY held a genuine
    // double-in/double-out kernel that the lane could not call. Every one of
    // those ten entry points is MULTI-OUTPUT with a bespoke parameter list --
    // `range_filtered_trend_signals_batch_f64` declares 25 parameters and
    // THIRTEEN output matrices, `possible_rsi_batch_f64` 28 and seven plus two
    // scratch arenas, `neighboring_trailing_stop_batch_f64` 21 and six plus
    // four -- while the lane launches exactly one shape and allocates ONE
    // output matrix. So none could be reused, and each file now carries a
    // lane-shaped twin beside it (search the file for
    // `NEOETHOS f64 LANE  --  closer 2, round 3`).
    //
    // THREE of the ten also dropped a device-side `new double[]` on the way:
    // `normalized_volume_true_range`, `regression_slope_oscillator` and
    // `relative_strength_index_wave_indicator` allocated their rings inside the
    // kernel. The twins size every ring from a CPU default at compile time, so
    // the bound is a property of the compiled kernel rather than of the caller.
    //
    // WHICH COLUMN EACH EMITS -- eight are what `output_id == "value"` resolves
    // to on the CPU (neighboring_trailing_stop -> trailing_stop,
    // normalized_volume_true_range -> normalized_volume,
    // price_moving_average_ratio_percentile -> plotline, range_breakout_signals
    // -> range_top, relative_strength_index_wave_indicator -> rsi_ma1, and
    // nonlinear_regression_zero_lag_moving_average / possible_rsi /
    // regression_slope_oscillator name `value` outright). The other two have NO
    // `value` output at all: `normalized_resonator` accepts only "oscillator"
    // and "signal", and `range_filtered_trend_signals` REJECTS "value" and
    // accepts thirteen named columns, so each emits its primary series --
    // oscillator and kalman respectively, the first arm of its own CPU match.
    NeighboringTrailingStop,
    NonlinearRegressionZeroLagMovingAverage,
    NormalizedResonator,
    NormalizedVolumeTrueRange,
    PossibleRsi,
    PriceMovingAverageRatioPercentile,
    RangeBreakoutSignals,
    RangeFilteredTrendSignals,
    RegressionSlopeOscillator,
    RelativeStrengthIndexWaveIndicator,

    // ------------------------------------------------ closer 4, round 3
    //
    // Ten more, every entry point written INTO THE .cu FILE ITS INDICATOR
    // ALREADY SHIPS IN (see `module_stem`), beside the f32 entry points the
    // f32 wrappers still call, and against the CPU reference named in that
    // file's `NEOETHOS f64 LANE  --  closer 4, round 3` header.
    //
    // NOT ONE could reuse a symbol already in its file. Every one of those
    // ten files was PURE f32 before this change -- `bandpass_kernel.cu` had
    // two `__global__`s and both took `const float*`, `dma_kernel.cu` had
    // seven, `buff_averages_kernel.cu` twelve -- so there was no f64 entry
    // point to point a variant at, and the lane could not reach these
    // indicators at all.
    //
    // ALL TEN ARE SEQUENTIAL. Every one carries state across bars: a 2-pole
    // IIR (bandpass, prb's super-smoother), a variable-alpha EMA driven by a
    // CMO ring plus a band ratchet (ott, otto), a weighted sliding sum rolled
    // rather than rebuilt (buff_averages, cora_wave, dma), a Wilder ATR with a
    // trend state machine (halftrend), a gap ledger with a trailing-stop
    // ratchet (fvg_trailing_stop), or five interlocking recurrences
    // (mod_god_mode). None can be made bar-parallel without changing the
    // rounding, which is the whole reason this lane exists.
    //
    // FIVE ARE PERIOD-INVARIANT and that is FAITHFUL, not lazy. Their CPU
    // batch functions read NAMED parameters and NEVER `period`:
    // unmitigated_fvg_lookback / smoothing_length / reset_on_cross for
    // fvg_trailing_stop (cpu_batch.rs:14862), amplitude / channel_deviation /
    // atr_period for halftrend (:14960), n1 / n2 / n3 / mode / use_volume for
    // mod_god_mode (:15516), ott_period / ott_percent / fast_vidya_length /
    // slow_vidya_length / correcting_constant / ma_type for otto (:15657), and
    // smooth_data / smooth_period / regression_period / polynomial_order /
    // regression_offset / ndev / equ_from for prb (:15833). A caller sweeping
    // [7,21,50,100,200] gets five identical CPU columns for each, so the
    // kernel writes five identical rows.
    //
    // THE FIVE THAT ARE SWEPT each map the swept int onto the window its own
    // CPU entry point reads, and the mapping is NOT always `period`:
    // `bandpass` and `ott` and `cora_wave` read a parameter literally named
    // `period`, but `ma_batch.rs:593` sweeps buff_averages' SLOW period and
    // `:1868` sweeps dma's HULL length. Mapping onto the other named window
    // would compute a different indicator.
    //
    // FIVE SERVE MULTI-OUTPUT INDICATORS, and each emits the column the CPU
    // batch produces for `output_id == "value"`: bandpass -> bp,
    // buff_averages -> fast (the `output` default in ma_batch.rs:629),
    // fvg_trailing_stop -> upper, halftrend -> halftrend, mod_god_mode ->
    // wavetrend, otto -> hott, prb -> values. Never a different one silently.
    /// Single price series, CPU source `close`. PERIOD-SWEPT.
    Bandpass,
    /// (close, volume). PERIOD-SWEPT -- the swept int is the SLOW period.
    BuffAverages,
    /// Single price series, CPU source `close`. PERIOD-SWEPT. Keeps a
    /// smoothing ring of `round(sqrt(period))` entries, hence a `max_period`.
    CoraWave,
    /// Single price series, CPU source `close`. PERIOD-SWEPT -- the swept int
    /// is the HULL length. Keeps a difference ring of `round(sqrt(hull))`
    /// entries, hence a `max_period`.
    Dma,
    /// high / low / close. PERIOD-INVARIANT. Emits the UPPER band.
    FvgTrailingStop,
    /// high / low / close. PERIOD-INVARIANT. Emits the HALFTREND series.
    Halftrend,
    /// (high, low, close, volume). PERIOD-INVARIANT. Emits the WAVETREND
    /// series.
    ModGodMode,
    /// Single price series, CPU source `close`. PERIOD-SWEPT.
    Ott,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT. Emits the
    /// HOTT series.
    Otto,
    /// Single price series, CPU source `close`. PERIOD-INVARIANT. Emits the
    /// `values` series.
    Prb,

    // ------------------------------------------------- closer 5, round 2
    //
    // Seventeen more, every entry point written INTO the `.cu` file its
    // indicator already ships in (see `module_stem`), beside the bespoke f64
    // entry points that file already carried, and against the CPU reference
    // named in that file's "NEOETHOS f64 LANE" header.
    //
    // NONE could reuse the entry point already in that file: those take a
    // bespoke parameter list (`squeeze_index_batch_f64` takes conv and length
    // arrays and one output; `twiggs_money_flow_batch_f64` takes six) and none
    // of them takes `first_valid`. A variant pointing at one of those symbols
    // would have read the stack.
    //
    // All seventeen are SEQUENTIAL: every one carries state across bars -- an
    // EMA or Wilder recurrence, a monotone deque, an incremental window sum
    // whose accumulation order is load-bearing, or a latched trend state.
    //
    // SIXTEEN ARE PERIOD-INVARIANT, and that is faithful rather than lazy:
    // their CPU batch functions read NAMED parameters and never `period` --
    // `gaussian_length`/`poles`/`smoothing_length`/`linreg_offset`,
    // `lookback`/`smoothing_length`, `conv`/`length`, `start`/`increment`/
    // `maximum`/`standardization_length`, `data_length`/`normalization_length`,
    // `k_length`/`d_smoothing`/`pre_smooth`, `stoch_length`/`smooth_k`/
    // `smooth_d`/`rsi_length`/`updown_length`/`roc_length`,
    // `lookback_length`/`length1`/`length2`, `stoch_k_length`/`mfi_length`,
    // `length`/`mult`/`smooth`, `atr_length`/`multiplier`/`alpha_percent`,
    // `alpha_length`/`alpha_multiplier`/`mfi_length`,
    // `length`/`smoothing_length`/`ma_type`, `fast_length`/`slow_length`,
    // `session_mode`/`rolling_period`/`deviation_mode`, and
    // `length`/`upper_bottom`/`lower_bottom`.
    //
    // `Supertrend` is THE ONE THAT IS NOT: `compute_supertrend_batch` reads a
    // parameter literally named `period` (cpu_batch.rs:6348), so its rows
    // genuinely differ and its kernel reads `periods[combo]`.
    //
    // SIXTEEN ARE FIRST-VALID-IGNORED because their CPU row walks EVERY bar
    // from index 0 and RESETS its state mid-series on an invalid bar; adopting
    // a start index would skip bars the CPU processes. `Supertrend` is again
    // the exception: `supertrend_prepare` (supertrend.rs:239) scans for the
    // first index at which high, low and close are all non-NaN, which is
    // exactly `AllInputsNonNan`.
    //
    // EVERY ONE SERVES A MULTI-OUTPUT INDICATOR EXCEPT `SqueezeIndex` and
    // `VolatilityQualityIndex`, and each emits the column the CPU batch
    // produces for `output_id == "value"`: smoothed_gaussian_trend_filter ->
    // filter, spearman_correlation -> raw, standardized_psar_oscillator ->
    // oscillator, statistical_trailing_stop -> level, stochastic_adaptive_d ->
    // standard_d, stochastic_connors_rsi -> k, stochastic_money_flow_index ->
    // k, supertrend -> trend, supertrend_oscillator -> oscillator,
    // supertrend_recovery -> band, trend_flow_trail -> alpha_trail,
    // twiggs_money_flow -> tmf, vwap_deviation_oscillator -> osc,
    // vwap_zscore_with_signals -> zvwap. `StochasticDistance` is the one whose
    // CPU batch REJECTS "value" outright (cpu_batch.rs:13310, it accepts only
    // "oscillator"/"signal"); it emits `oscillator`, and a parity run must ask
    // the CPU for that output id explicitly.
    SmoothedGaussianTrendFilter,
    SpearmanCorrelation,
    SqueezeIndex,
    StandardizedPsarOscillator,
    StatisticalTrailingStop,
    StochasticAdaptiveD,
    StochasticConnorsRsi,
    StochasticDistance,
    StochasticMoneyFlowIndex,
    Supertrend,
    SupertrendOscillator,
    SupertrendRecovery,
    TrendFlowTrail,
    TwiggsMoneyFlow,
    VolatilityQualityIndex,
    VwapDeviationOscillator,
    VwapZscoreWithSignals,

    // ------------------------------------------- closer 5, round 2 (adosc)
    /// (high, low, close, volume). PERIOD-INVARIANT -- `compute_adosc_batch`
    /// reads `short_period` (3) and `long_period` (10) and never `period`
    /// (cpu_batch.rs:2670-2671), and those two defaults are exactly what
    /// selects `adosc_scalar_3_10` (adosc.rs:372). FIRST-VALID IGNORED:
    /// `adosc_prepare` returns `first = 0` outright (:331) because the
    /// accumulation-distribution line is a cumulative sum from bar zero.
    ///
    /// `kernels/cuda/oscillators/adosc_kernel.cu` previously contained ZERO
    /// double-pointer entry points -- the whole file was f32.
    Adosc,

    // -------------------------------------------------- closer 6, round 3
    //
    // Six indicators with ZERO prior CUDA presence: no `.cu`, no wrapper, no
    // row here, so `resolve_f64_kernel` answered `CudaF64KernelMissing` for
    // every one of them. Each now has a from-scratch f64 kernel in its own
    // translation unit under `kernels/cuda/moving_averages/`.
    /// (hlcc4, volume). Sequential per column. Its CPU default source is
    /// `hlcc4` (elastic_volume_weighted_moving_average.rs:113) and it takes
    /// the `use_volume_sum == true` branch, because that is the branch the
    /// period-sweeping route selects (ma.rs:1105-1113, registry.rs:608) and
    /// the other branch never reads `length` at all.
    ElasticVolumeWeightedMovingAverage,
    /// Single price series (close). Sequential per column: six T3 cascade
    /// stages, two deviation EMAs and the correction, nine carried scalars,
    /// all reset by a non-finite bar. Emits the PRIMARY output, the corrected
    /// line (registry.rs:537).
    EmaDeviationCorrectedT3,
    /// Single price series (close). Sequential per column -- a `run` counter
    /// carries across bars and the window sum is accumulated in 8-wide chunks
    /// (logarithmic_moving_average.rs:801), an association a bar-parallel
    /// kernel could not reproduce. Emits the PRIMARY output, `lma`.
    LogarithmicMovingAverage,
    /// Single price series (close). Sequential per column: a first-order IIR
    /// whose history is reset by a non-finite bar.
    NOrderEma,
    /// Single price series (close). Sequential per column: an EMA, two
    /// monotonic deques over a 51-bar volatility window, and a rolling WMA(5),
    /// three chained recurrences.
    VolatilityAdjustedMa,
    /// Single price series (close). Sequential per column: the pre-smoother
    /// reads bar i and bar i-1 and the weighted sum runs over the last
    /// `period + 1` smoothed values.
    WaveSmoother,
    // ------------------------------------------------- closer 5, round 3
    //
    // Nine indicators that had NO reachable f64 entry point before this round.
    // Seven lived in files whose kernels were f32-only or MIXED (a `double*`
    // accumulator feeding a `float*` consumer, which is an f64 ACCUMULATOR and
    // not an f64 API); two -- `corrected_moving_average` and
    // `ehlers_undersampled_double_moving_average` -- had no `.cu` file at all.
    //
    // Every one is SEQUENTIAL: each carries at least one scalar across bars.
    // `Rsmk` and `CorrectedMovingAverage` are the only two that are genuinely
    // PERIOD-SWEPT -- their CPU batches read a parameter literally named
    // `period` (cpu_batch.rs:16479, ma.rs:263). The other seven pin every
    // window at a CPU default and are declared period-invariant for that
    // reason, not for convenience.
    Rsmk,
    SqueezeMomentum,
    Uma,
    Lpc,
    Mab,
    Macz,
    Vwmacd,
    CorrectedMovingAverage,
    EhlersUndersampledDoubleMovingAverage,

    // ------------------------------------------------ closer 3, round 3
    //
    // Ten more, every entry point written INTO THE .cu FILE ITS INDICATOR
    // ALREADY SHIPS IN (see `module_stem`), beside the entry points the
    // existing f32 and multi-output wrappers still call, and against the CPU
    // reference named in that file's "f64 LANE  --  closer 3, round 3" header.
    //
    // ALL TEN ARE SEQUENTIAL. Every one carries state across bars: a Wilder or
    // EMA recurrence, a six-stage smoothing cascade, a ratchet, a monotone
    // deque, a pivot state machine, or a sliding sum maintained by
    // subtract-then-add. None can be made bar-parallel without changing the
    // rounding, which is the whole reason this lane exists.
    //
    // EIGHT ARE PERIOD-INVARIANT and that is FAITHFUL, not lazy: their CPU
    // batch functions read NAMED parameters and NEVER `period` --
    // `lookback_period`/`confirmation_period`/`trend_ma_period`/
    // `ma_step_period` for reversal_signals (cpu_batch.rs:7286-7295),
    // `trend_period`/`ma_period`/`channel_rate_percent`/
    // `linear_regression_period`/`matype` for trend_follower
    // (trend_follower.rs:147-169), sixteen `fast_depth`/`slow_depth`/
    // `show_*` names for vdubus (:5109-5209), `length`/`sensitivity` for
    // volume_energy_reservoirs (:8984), `rsi_length`/`range_length`/
    // `ma_length` for volume_weighted_relative_strength_index (:16340),
    // `rsi_length`/`stoch_length`/`k_length`/`d_length` for
    // volume_weighted_stochastic_rsi (:12413), `length`/`extend` for
    // zig_zag_channels (:7365) and `fast_period`/`slow_period`/`multiplier`
    // for avsl (:14126). A caller sweeping `[7,21,50,100,200]` gets five
    // identical CPU columns for each of them, so the kernel emits five
    // identical rows and `is_period_invariant` says so.
    //
    // THE TWO THAT ARE GENUINELY PERIOD-SWEPT read a parameter literally named
    // `period`: `VolatilityRatioAdaptiveRsx` (cpu_batch.rs:9707) and
    // `Alphatrend` (:13998). Both size a per-thread ring from it, so both
    // carry a `max_period` bound.
    //
    // EVERY ONE DECLARES `F64FirstValidRule::Ignored` EXCEPT `Alphatrend`.
    // That asymmetry is read from the CPU, not chosen: eight of these
    // references walk every bar from 0 and RESET their whole state at an
    // invalid one, so a global warmup index would be wrong after the first
    // hole; `avsl` scans with a rule no variant expresses (the MAX of three
    // INDEPENDENT first-non-NaN scans over close, low and volume,
    // avsl.rs:272) and derives it inside the kernel; and `alphatrend` scans
    // `close.iter().position(|x| !x.is_nan())` (alphatrend.rs:493), which is
    // exactly the existing `HlcCloseOnly` rule that `adxr` already declares.
    //
    // EVERY ONE SERVES A MULTI-OUTPUT INDICATOR, and each emits the column its
    // CPU batch produces for `output_id == "value"`: reversal_signals ->
    // stepped_ma, trend_follower -> values, volatility_ratio_adaptive_rsx ->
    // line, volume_energy_reservoirs -> momentum,
    // volume_weighted_relative_strength_index -> rsi,
    // volume_weighted_stochastic_rsi -> k, zig_zag_channels -> middle,
    // alphatrend -> k1, avsl -> values. Never a different one silently.
    //
    // ONE HAS NO REACHABLE `value` ALIAS AT ALL and is named here for that
    // reason: `compute_vdubus_divergence_wave_pattern_generator_batch`
    // (cpu_batch.rs:5095) calls `expect_value_output`, which admits ONLY the
    // literal "value", and then matches against twelve arms none of which is
    // "value" -- so every request falls through to `UnknownOutput`. Its kernel
    // emits `fast_standard`, output index 0 of the registry's list
    // (registry.rs:1240), and a parity run must ask the CPU for that column by
    // name.
    /// (open, high, low, close, volume) -- OPEN is in the validity gate.
    /// PERIOD-INVARIANT. Emits the STEPPED MA.
    ReversalSignals,
    /// (high, low, close, volume). PERIOD-INVARIANT. Emits `values`. Carries
    /// BOTH CPU branches -- the clean sliding-linreg path and the
    /// reset-on-NaN streaming path -- because they are different arithmetic.
    TrendFollower,
    /// high / low / close. PERIOD-INVARIANT. Emits `fast_standard`.
    VdubusDivergenceWavePatternGenerator,
    /// Single price series, CPU source `close`. PERIOD-SWEPT -- the swept int
    /// sets both ring depths, hence a `max_period`. Emits `line`.
    VolatilityRatioAdaptiveRsx,
    /// (high, low, close, volume). PERIOD-INVARIANT. Emits `momentum`.
    VolumeEnergyReservoirs,
    /// (close, volume). PERIOD-INVARIANT. Emits `rsi`.
    VolumeWeightedRelativeStrengthIndex,
    /// (close, volume). PERIOD-INVARIANT. Emits `k`.
    VolumeWeightedStochasticRsi,
    /// open / high / low / close -- OPEN is in the validity gate.
    /// PERIOD-INVARIANT. Emits `middle`.
    ZigZagChannels,
    /// (high, low, close, volume). PERIOD-SWEPT -- the swept int is BOTH the
    /// true-range window AND the MFI period, hence a `max_period`. Emits `k1`.
    Alphatrend,
    /// (high, low, close, volume) with HIGH bound and unread. PERIOD-INVARIANT.
    /// Emits `values`.
    Avsl,

    // ------------------------------------------------ closer 1, round 3
    //
    // Ten indicators whose `.cu` file ALREADY held a genuine
    // double-in/double-out kernel that the lane could not call, because every
    // one of those entry points is MULTI-OUTPUT with a bespoke parameter list
    // -- `ichimoku_oscillator_batch_f64` declares 36 parameters and 20
    // `double*`, `goertzel_cycle_composite_wave_batch_f64` declares 31 and
    // takes two host scratch pointers, `macd_wave_signal_pro_batch_f64` takes
    // no `periods` array at all. None of them is the lane ABI, so each file
    // received a `*_neo_batch_f64` entry point beside what it already had.
    // Every one of the ten is PERIOD-INVARIANT: its CPU batch reads named
    // windows and never `period`.
    /// open / high / low / close. Emits `basis` -- the CPU batch has no
    /// `value` alias and every other column is derived from basis.
    FibonacciEntryBands,
    /// Single price series, CPU source `close`. Emits `value`. The ONLY
    /// bar-parallel kernel of the ten: `compute_row` recomputes each 601-bar
    /// window from scratch and carries NOTHING across bars, so the Goertzel
    /// recurrence lives inside the window, not along the series.
    GoertzelCycleCompositeWave,
    /// (timestamps, close, volume). Emits `estimate`. Timestamps are an
    /// INPUT -- the working CPU door infers `slots_per_day` from them and
    /// takes VOLUME as the source; close is passed by the shape and unread.
    HalfCausalEstimator,
    /// high / low / close. Emits `signal`, which the CPU batch also aliases
    /// as `value`.
    IchimokuOscillator,
    /// (high, low, close, volume). Emits its single `value` series. Ten
    /// sub-indicators, all reset together on an invalid bar.
    InsyncIndex,
    /// Single price series, CPU source `close`. Emits `value`.
    LinearRegressionIntensity,
    /// open / high / low / close. Emits `diff`, which the CPU batch also
    /// aliases as `value`.
    MacdWaveSignalPro,
    /// Single price series, CPU source `close`. Emits `mesa_1` -- the CPU
    /// batch has no `value` alias and mesa_1 is the longest line.
    MesaStochasticMultiLength,
    /// Single price series, CPU source `close`. Emits `value`.
    MovingAverageCrossProbability,
    /// Single price series, CPU source `close`. Emits `value`.
    MultiLengthStochasticAverage,
}


impl F64Kernel {
    /// The exact `__global__` entry point this variant launches.
    pub fn entry_point(self) -> &'static str {
        match self {
            F64Kernel::Sma => "neoethos_sma_batch_f64",
            F64Kernel::Adosc => "adosc_neo_batch_f64",
            // ------------------------------------------ closer 5, round 3
            F64Kernel::Rsmk => "rsmk_neo_batch_f64",
            F64Kernel::SqueezeMomentum => "squeeze_momentum_neo_batch_f64",
            F64Kernel::Uma => "uma_neo_batch_f64",
            F64Kernel::Lpc => "lpc_neo_batch_f64",
            F64Kernel::Mab => "mab_neo_batch_f64",
            F64Kernel::Macz => "macz_neo_batch_f64",
            F64Kernel::Vwmacd => "vwmacd_neo_batch_f64",
            F64Kernel::CorrectedMovingAverage => "corrected_moving_average_neo_batch_f64",
            F64Kernel::EhlersUndersampledDoubleMovingAverage => "ehlers_undersampled_double_moving_average_neo_batch_f64",
            // ------------------------------------------- closer 5, round 2
            F64Kernel::SmoothedGaussianTrendFilter => "smoothed_gaussian_trend_filter_neo_batch_f64",
            F64Kernel::SpearmanCorrelation => "spearman_correlation_neo_batch_f64",
            F64Kernel::SqueezeIndex => "squeeze_index_neo_batch_f64",
            F64Kernel::StandardizedPsarOscillator => "standardized_psar_oscillator_neo_batch_f64",
            F64Kernel::StatisticalTrailingStop => "statistical_trailing_stop_neo_batch_f64",
            F64Kernel::StochasticAdaptiveD => "stochastic_adaptive_d_neo_batch_f64",
            F64Kernel::StochasticConnorsRsi => "stochastic_connors_rsi_neo_batch_f64",
            F64Kernel::StochasticDistance => "stochastic_distance_neo_batch_f64",
            F64Kernel::StochasticMoneyFlowIndex => "stochastic_money_flow_index_neo_batch_f64",
            F64Kernel::Supertrend => "supertrend_neo_batch_f64",
            F64Kernel::SupertrendOscillator => "supertrend_oscillator_neo_batch_f64",
            F64Kernel::SupertrendRecovery => "supertrend_recovery_neo_batch_f64",
            F64Kernel::TrendFlowTrail => "trend_flow_trail_neo_batch_f64",
            F64Kernel::TwiggsMoneyFlow => "twiggs_money_flow_neo_batch_f64",
            F64Kernel::VolatilityQualityIndex => "volatility_quality_index_neo_batch_f64",
            F64Kernel::VwapDeviationOscillator => "vwap_deviation_oscillator_neo_batch_f64",
            F64Kernel::VwapZscoreWithSignals => "vwap_zscore_with_signals_neo_batch_f64",
            F64Kernel::Ema => "neoethos_ema_batch_f64",
            F64Kernel::Rsi => "neoethos_rsi_batch_f64",
            F64Kernel::Roc => "neoethos_roc_batch_f64",
            F64Kernel::Mom => "neoethos_mom_batch_f64",
            F64Kernel::Atr => "neoethos_atr_batch_f64",
            F64Kernel::Adx => "neoethos_adx_batch_f64",
            F64Kernel::Willr => "neoethos_willr_batch_f64",
            F64Kernel::Cci => "neoethos_cci_from_sma_f64",
            F64Kernel::Mfi => "neoethos_mfi_batch_f64",
            F64Kernel::Tsi => "neoethos_tsi_batch_f64",
            F64Kernel::Obv => "neoethos_obv_batch_f64",
            F64Kernel::Vwap => "neoethos_vwap_batch_f64",
            F64Kernel::Wma => "neoethos_wma_batch_f64",
            F64Kernel::Wilders => "neoethos_wilders_batch_f64",
            F64Kernel::Smma => "neoethos_smma_batch_f64",
            F64Kernel::Dema => "neoethos_dema_batch_f64",
            F64Kernel::Tema => "neoethos_tema_batch_f64",
            F64Kernel::Zlema => "neoethos_zlema_batch_f64",
            F64Kernel::Vwma => "neoethos_vwma_batch_f64",
            F64Kernel::Efi => "neoethos_efi_batch_f64",
            F64Kernel::Natr => "neoethos_natr_batch_f64",
            F64Kernel::Adxr => "neoethos_adxr_batch_f64",
            F64Kernel::Medprice => "neoethos_medprice_batch_f64",
            F64Kernel::Wclprice => "neoethos_wclprice_batch_f64",
            F64Kernel::Midpoint => "neoethos_midpoint_batch_f64",
            F64Kernel::Midprice => "neoethos_midprice_batch_f64",
            F64Kernel::Rocp => "neoethos_rocp_batch_f64",
            F64Kernel::Rocr => "neoethos_rocr_batch_f64",
            F64Kernel::Sqwma => "neoethos_sqwma_batch_f64",
            F64Kernel::Deviation => "neoethos_deviation_batch_f64",
            F64Kernel::MeanAd => "neoethos_mean_ad_batch_f64",
            F64Kernel::Ao => "neoethos_ao_batch_f64",
            F64Kernel::LinearregSlope => "neoethos_linearreg_slope_batch_f64",
            F64Kernel::Tsf => "neoethos_tsf_batch_f64",
            F64Kernel::Highpass => "neoethos_highpass_batch_f64",
            F64Kernel::Decycler => "neoethos_decycler_batch_f64",
            F64Kernel::Supersmoother => "neoethos_supersmoother_batch_f64",
            F64Kernel::Tilson => "neoethos_tilson_batch_f64",
            F64Kernel::Wad => "neoethos_wad_batch_f64",
            F64Kernel::Sar => "neoethos_sar_batch_f64",
            F64Kernel::Dti => "neoethos_dti_batch_f64",
            F64Kernel::Zscore => "neoethos_zscore_batch_f64",
            F64Kernel::Pfe => "neoethos_pfe_batch_f64",
            F64Kernel::Chande => "neoethos_chande_batch_f64",
            F64Kernel::Di => "neoethos_di_batch_f64",
            F64Kernel::Kdj => "neoethos_kdj_batch_f64",
            F64Kernel::Aso => "neoethos_aso_batch_f64",
            F64Kernel::Wto => "neoethos_wto_batch_f64",
            F64Kernel::RangeFilter => "neoethos_range_filter_batch_f64",
            F64Kernel::CorrelationCycle => "neoethos_correlation_cycle_batch_f64",
            F64Kernel::Mama => "neoethos_mama_batch_f64",
            F64Kernel::VolumeAdjustedMa => "neoethos_volume_adjusted_ma_batch_f64",
            F64Kernel::ReverseRsi => "neoethos_reverse_rsi_batch_f64",
            F64Kernel::EhlersEcema => "neoethos_ehlers_ecema_batch_f64",
            F64Kernel::Devstop => "neoethos_devstop_batch_f64",
            F64Kernel::ChandelierExit => "neoethos_chandelier_exit_batch_f64",
            F64Kernel::Minmax => "neoethos_minmax_batch_f64",
            // ------------------------------------------------------------- shard 1 (S1)
            F64Kernel::Apo => "neoethos_apo_batch_f64",
            F64Kernel::Vidya => "neoethos_vidya_batch_f64",
            F64Kernel::Gatorosc => "neoethos_gatorosc_batch_f64",
            F64Kernel::Ppo => "neoethos_ppo_batch_f64",
            F64Kernel::Pma => "neoethos_pma_batch_f64",
            F64Kernel::Kama => "neoethos_kama_batch_f64",
            F64Kernel::Linreg => "neoethos_linreg_batch_f64",
            F64Kernel::Edcf => "neoethos_edcf_batch_f64",
            F64Kernel::Alma => "neoethos_alma_batch_f64",
            F64Kernel::Hma => "neoethos_hma_batch_f64",
            F64Kernel::Kurtosis => "neoethos_kurtosis_batch_f64",
            F64Kernel::Alligator => "neoethos_alligator_batch_f64",
            F64Kernel::Nvi => "neoethos_nvi_batch_f64",
            F64Kernel::Fisher => "neoethos_fisher_batch_f64",
            F64Kernel::Safezonestop => "neoethos_safezonestop_batch_f64",
            F64Kernel::Chop => "neoethos_chop_batch_f64",
            F64Kernel::Stochf => "neoethos_stochf_batch_f64",
            F64Kernel::Emv => "neoethos_emv_batch_f64",
            F64Kernel::Kvo => "neoethos_kvo_batch_f64",
            // ------------------------------------------------------- shard 4
            F64Kernel::Er => "er_neo_batch_f64",
            F64Kernel::LinearregAngle => "linearreg_angle_neo_batch_f64",
            F64Kernel::LinearregIntercept => "linearreg_intercept_neo_batch_f64",
            F64Kernel::Highpass2Pole => "highpass2_neo_batch_f64",
            F64Kernel::Supersmoother3Pole => "supersmoother_3_pole_neo_batch_f64",
            F64Kernel::Cwma => "cwma_neo_batch_f64",
            F64Kernel::Cmo => "cmo_neo_batch_f64",
            F64Kernel::Stddev => "stddev_neo_batch_f64",
            F64Kernel::Ui => "ui_neo_batch_f64",
            F64Kernel::BollingerBands => "bollinger_bands_neo_batch_f64",
            F64Kernel::Ehma => "ehma_neo_batch_f64",
            F64Kernel::Macd => "macd_neo_batch_f64",
            F64Kernel::IftRsi => "ift_rsi_neo_batch_f64",
            F64Kernel::DamianiVolatmeter => "damiani_volatmeter_neo_batch_f64",
            F64Kernel::Wavetrend => "wavetrend_neo_batch_f64",
            F64Kernel::Dx => "dx_neo_batch_f64",
            F64Kernel::Frama => "frama_neo_batch_f64",
            F64Kernel::Cksp => "cksp_neo_batch_f64",
            F64Kernel::TtmSqueeze => "ttm_squeeze_neo_batch_f64",
            F64Kernel::Mass => "mass_neo_batch_f64",
            F64Kernel::Aroon => "aroon_neo_batch_f64",
            F64Kernel::Acosc => "acosc_neo_batch_f64",
            F64Kernel::Vpci => "vpci_neo_batch_f64",
            F64Kernel::Ad => "ad_neo_batch_f64",
            F64Kernel::Dvdiqqe => "dvdiqqe_neo_batch_f64",
            F64Kernel::CciCycle => "cci_cycle_neo_batch_f64",
            F64Kernel::Rsx => "neoethos_rsx_batch_f64",
            F64Kernel::Trix => "neoethos_trix_batch_f64",
            F64Kernel::Vpt => "neoethos_vpt_batch_f64",
            F64Kernel::Pvi => "neoethos_pvi_batch_f64",
            F64Kernel::EhlersItrend => "neoethos_ehlers_itrend_batch_f64",
            F64Kernel::EhlersKama => "neoethos_ehlers_kama_batch_f64",
            F64Kernel::Sama => "neoethos_sama_batch_f64",
            F64Kernel::Nama => "neoethos_nama_batch_f64",
            F64Kernel::Pwma => "neoethos_pwma_batch_f64",
            F64Kernel::Tradjema => "neoethos_tradjema_batch_f64",
            F64Kernel::Maaq => "neoethos_maaq_batch_f64",
            F64Kernel::Jma => "neoethos_jma_batch_f64",
            F64Kernel::Reflex => "neoethos_reflex_batch_f64",
            F64Kernel::Gaussian => "neoethos_gaussian_batch_f64",
            F64Kernel::Fwma => "fwma_batch_f64",
            F64Kernel::Hwma => "hwma_batch_f64",
            F64Kernel::Jsa => "jsa_batch_f64",
            F64Kernel::Nma => "nma_batch_f64",
            F64Kernel::Swma => "swma_batch_f64",
            F64Kernel::Trendflex => "trendflex_batch_f64",
            F64Kernel::Vpwma => "vpwma_batch_f64",
            F64Kernel::Cfo => "cfo_batch_f64",
            F64Kernel::Var => "var_batch_f64",
            F64Kernel::BollingerBandsWidth => "bollinger_bands_width_batch_f64",
            F64Kernel::DecOsc => "dec_osc_batch_f64",
            F64Kernel::Voss => "voss_batch_f64",
            F64Kernel::PercentileNearestRank => "percentile_nearest_rank_batch_f64",
            F64Kernel::TtmTrend => "ttm_trend_batch_f64",
            F64Kernel::Vi => "vi_batch_f64",
            F64Kernel::Cvi => "cvi_batch_f64",
            F64Kernel::CorrelHl => "correl_hl_batch_f64",
            F64Kernel::Aroonosc => "aroonosc_batch_f64",
            F64Kernel::ParkinsonVolatility => "parkinson_volatility_batch_f64",
            F64Kernel::HistoricalVolatility => "historical_volatility_batch_f64",
            F64Kernel::Donchian => "donchian_batch_f64",
            // --------------------------------------------------- closer 5
            F64Kernel::Velocity => "velocity_neo_batch_f64",
            F64Kernel::VelocityAccelerationIndicator => "velocity_acceleration_indicator_neo_batch_f64",
            F64Kernel::VelocityAccelerationConvergenceDivergenceIndicator => "velocity_acceleration_convergence_divergence_indicator_neo_batch_f64",
            F64Kernel::TrendDirectionForceIndex => "trend_direction_force_index_neo_batch_f64",
            F64Kernel::TrendContinuationFactor => "trend_continuation_factor_neo_batch_f64",
            F64Kernel::Trima => "trima_neo_batch_f64",
            F64Kernel::TrendTriggerFactor => "trend_trigger_factor_neo_batch_f64",
            F64Kernel::VolumeWeightedRsi => "volume_weighted_rsi_neo_batch_f64",
            F64Kernel::VolumeZoneOscillator => "volume_zone_oscillator_neo_batch_f64",
            F64Kernel::Vosc => "vosc_neo_batch_f64",
            F64Kernel::Ultosc => "ultosc_neo_batch_f64",
            // ------------------------------------------------------ closer 4
            F64Kernel::PsychologicalLine => "neoethos_psychological_line_batch_f64",
            F64Kernel::RankCorrelationIndex => "neoethos_rank_correlation_index_batch_f64",
            F64Kernel::Qstick => "neoethos_qstick_batch_f64",
            F64Kernel::Sinwma => "neoethos_sinwma_batch_f64",
            F64Kernel::Srwma => "neoethos_srwma_batch_f64",
            F64Kernel::RollingZScoreTrend => "neoethos_rolling_z_score_trend_batch_f64",
            F64Kernel::RandomWalkIndex => "neoethos_random_walk_index_batch_f64",
            // --------------------------------------------------------- closer 3
            F64Kernel::L1EhlersPhasor => "l1_ehlers_phasor_neo_batch_f64",
            F64Kernel::L2EhlersSignalToNoise => "l2_ehlers_signal_to_noise_neo_batch_f64",
            F64Kernel::KairiRelativeIndex => "kairi_relative_index_neo_batch_f64",
            F64Kernel::LinearCorrelationOscillator => "linear_correlation_oscillator_neo_batch_f64",
            F64Kernel::MediumAd => "medium_ad_neo_batch_f64",
            F64Kernel::Marketefi => "marketefi_neo_batch_f64",
            F64Kernel::MomentumRatioOscillator => "momentum_ratio_oscillator_neo_batch_f64",
            F64Kernel::OnBalanceVolumeOscillator => {
                "on_balance_volume_oscillator_neo_batch_f64"
            }
            // ------------------------------------------------------ closer 6
            F64Kernel::Emd => "emd_batch_f64",
            F64Kernel::Keltner => "keltner_batch_f64",
            F64Kernel::Stoch => "stoch_batch_f64",
            F64Kernel::NadarayaWatsonEnvelope => "nadaraya_watson_envelope_batch_f64",
                    // ------------------------------------------------------------ closer 2
            F64Kernel::EhlersDetrendingFilter => "ehlers_detrending_filter_neo_batch_f64",
            F64Kernel::EhlersSimpleCycleIndicator => "ehlers_simple_cycle_indicator_neo_batch_f64",
            F64Kernel::EhlersSmoothedAdaptiveMomentum => "ehlers_smoothed_adaptive_momentum_neo_batch_f64",
            F64Kernel::EwmaVolatility => "ewma_volatility_neo_batch_f64",
            F64Kernel::FractalDimensionIndex => "fractal_dimension_index_neo_batch_f64",
            F64Kernel::GopalakrishnanRangeIndex => "gopalakrishnan_range_index_neo_batch_f64",
            F64Kernel::GarmanKlassVolatility => "garman_klass_volatility_neo_batch_f64",
            F64Kernel::ImpulseMacd => "impulse_macd_neo_batch_f64",
            F64Kernel::Hypertrend => "hypertrend_neo_batch_f64",
            F64Kernel::EmdTrend => "emd_trend_neo_batch_f64",
            F64Kernel::Epma => "epma_neo_batch_f64",
            F64Kernel::Fosc => "fosc_neo_batch_f64",
            F64Kernel::EhlersPma => "ehlers_pma_neo_batch_f64",
            F64Kernel::Eri => "eri_neo_batch_f64",
            // ---------------------------------------------------------- closer 2b
            F64Kernel::EhlersFmDemodulator => "ehlers_fm_demodulator_neo_batch_f64",
            F64Kernel::ForwardBackwardExponentialOscillator => "forward_backward_exponential_oscillator_neo_batch_f64",
            F64Kernel::GmmaOscillator => "gmma_oscillator_neo_batch_f64",
            F64Kernel::EvasiveSupertrend => "evasive_supertrend_neo_batch_f64",
            // ------------------------------------------- closer 6, round 2
            F64Kernel::Msw => "msw_neo_batch_f64",
            F64Kernel::YangZhangVolatility => "yang_zhang_volatility_neo_batch_f64",
            F64Kernel::Qqe => "qqe_neo_batch_f64",
            F64Kernel::Srsi => "srsi_neo_batch_f64",
            F64Kernel::Rvi => "rvi_neo_batch_f64",
            F64Kernel::NetMyrsi => "net_myrsi_neo_batch_f64",
            F64Kernel::Vlma => "vlma_neo_batch_f64",
            F64Kernel::Stc => "stc_neo_batch_f64",
            // ------------------------------------------------------ closer 1
            F64Kernel::AbsoluteStrengthIndexOscillator => {
                "absolute_strength_index_oscillator_neo_batch_f64"
            }
            F64Kernel::AccumulationSwingIndex => "accumulation_swing_index_neo_batch_f64",
            F64Kernel::AdaptiveBandpassTriggerOscillator => {
                "adaptive_bandpass_trigger_oscillator_neo_batch_f64"
            }
            F64Kernel::AdaptiveBoundsRsi => "adaptive_bounds_rsi_neo_batch_f64",
            F64Kernel::AdaptiveMacd => "adaptive_macd_neo_batch_f64",
            F64Kernel::AdaptiveMomentumOscillator => "adaptive_momentum_oscillator_neo_batch_f64",
            F64Kernel::AdvanceDeclineLine => "advance_decline_line_neo_batch_f64",
            F64Kernel::AndeanOscillator => "andean_oscillator_neo_batch_f64",
            F64Kernel::AtrPercentile => "atr_percentile_neo_batch_f64",
            F64Kernel::Bop => "bop_neo_batch_f64",
            F64Kernel::BullPowerVsBearPower => "bull_power_vs_bear_power_neo_batch_f64",
            F64Kernel::Cg => "cg_neo_batch_f64",
            F64Kernel::Coppock => "coppock_neo_batch_f64",
            F64Kernel::DailyFactor => "daily_factor_neo_batch_f64",
            F64Kernel::DecisionpointBreadthSwenlinTradingOscillator => {
                "decisionpoint_breadth_swenlin_trading_oscillator_neo_batch_f64"
            }
            F64Kernel::DidiIndex => "didi_index_neo_batch_f64",
            F64Kernel::DisparityIndex => "disparity_index_neo_batch_f64",
            F64Kernel::Dm => "dm_neo_batch_f64",
            F64Kernel::DonchianChannelWidth => "donchian_channel_width_neo_batch_f64",
            F64Kernel::Dpo => "dpo_neo_batch_f64",
            // ------------------------------------------------ closer 2, round 2
            F64Kernel::Mwdx => "mwdx_neo_batch_f64",
            F64Kernel::Lrsi => "lrsi_neo_batch_f64",
            F64Kernel::Pivot => "pivot_neo_batch_f64",
            F64Kernel::Kaufmanstop => "kaufmanstop_neo_batch_f64",
            F64Kernel::Sgf => "sgf_neo_batch_f64",
            F64Kernel::PolynomialRegressionExtrapolation => {
                "polynomial_regression_extrapolation_neo_batch_f64"
            }
            F64Kernel::DualUlcerIndex => "dual_ulcer_index_neo_batch_f64",
            F64Kernel::HullButterflyOscillator => "hull_butterfly_oscillator_neo_batch_f64",
            F64Kernel::RangeOscillator => "range_oscillator_neo_batch_f64",
            F64Kernel::MarketStructureTrailingStop => {
                "market_structure_trailing_stop_neo_batch_f64"
            }
            // ------------------------------------------------ closer 3, round 2
            //
            // `VerticalHorizontalFilter` is the one entry here that is NOT a
            // `*_neo_batch_f64`: the kernel already in
            // `vertical_horizontal_filter_kernel.cu` carries the lane ABI
            // exactly -- (data, len, lengths, n_combos, first_valid, out) --
            // so adding a second entry point beside it would have been a
            // duplicate, not a fix. Everything else in this block is a new
            // entry point written into the indicator's own file.
            F64Kernel::VerticalHorizontalFilter => "vertical_horizontal_filter_batch_f64",
            F64Kernel::AdjustableMaAlternatingExtremities => {
                "adjustable_ma_alternating_extremities_neo_batch_f64"
            }
            F64Kernel::AutocorrelationIndicator => "autocorrelation_indicator_neo_batch_f64",
            F64Kernel::HistoricalVolatilityRank => "historical_volatility_rank_neo_batch_f64",
            F64Kernel::HistoricalVolatilityPercentile => {
                "historical_volatility_percentile_neo_batch_f64"
            }
            F64Kernel::DirectionalImbalanceIndex => "directional_imbalance_index_neo_batch_f64",
            F64Kernel::CycleChannelOscillator => "cycle_channel_oscillator_neo_batch_f64",
            F64Kernel::DynamicMomentumIndex => "dynamic_momentum_index_neo_batch_f64",
            F64Kernel::EhlersAdaptiveCg => "ehlers_adaptive_cg_neo_batch_f64",
            F64Kernel::EhlersAdaptiveCyberCycle => "ehlers_adaptive_cyber_cycle_neo_batch_f64",
            F64Kernel::EhlersDataSamplingRelativeStrengthIndicator => {
                "ehlers_data_sampling_relative_strength_indicator_neo_batch_f64"
            }
            F64Kernel::ExponentialTrend => "exponential_trend_neo_batch_f64",
            F64Kernel::GeometricBiasOscillator => "geometric_bias_oscillator_neo_batch_f64",
            F64Kernel::IntradayMomentumIndex => "intraday_momentum_index_neo_batch_f64",
            F64Kernel::BullsVBears => "bulls_v_bears_neo_batch_f64",
            F64Kernel::CandleStrengthOscillator => "candle_strength_oscillator_neo_batch_f64",
            F64Kernel::CyberpunkValueTrendAnalyzer => {
                "cyberpunk_value_trend_analyzer_neo_batch_f64"
            }
            F64Kernel::FvgPositioningAverage => "fvg_positioning_average_neo_batch_f64",
            F64Kernel::HemaTrendLevels => "hema_trend_levels_neo_batch_f64",
            F64Kernel::FibonacciTrailingStop => "fibonacci_trailing_stop_neo_batch_f64",
            F64Kernel::GroverLlorensCycleOscillator => {
                "grover_llorens_cycle_oscillator_neo_batch_f64"
            }
            F64Kernel::DemandIndex => "demand_index_neo_batch_f64",
            F64Kernel::AdaptiveSchaffTrendCycle => "adaptive_schaff_trend_cycle_neo_batch_f64",
            F64Kernel::EhlersLinearExtrapolationPredictor => {
                "ehlers_linear_extrapolation_predictor_neo_batch_f64"
            }
            F64Kernel::EhlersAutocorrelationPeriodogram => {
                "ehlers_autocorrelation_periodogram_neo_batch_f64"
            }
            F64Kernel::IctPropulsionBlock => "ict_propulsion_block_neo_batch_f64",
            // ------------------------------------------ closer 4, round 2
            F64Kernel::KasePeakOscillatorWithDivergences => "kase_peak_oscillator_with_divergences_neo_batch_f64",
            F64Kernel::KeltnerChannelWidthOscillator => "keltner_channel_width_oscillator_neo_batch_f64",
            F64Kernel::Kst => "kst_neo_batch_f64",
            F64Kernel::LeavittConvolutionAcceleration => "leavitt_convolution_acceleration_neo_batch_f64",
            F64Kernel::MarketMeannessIndex => "market_meanness_index_neo_batch_f64",
            F64Kernel::MarketStructureConfluence => "market_structure_confluence_neo_batch_f64",
            F64Kernel::MonotonicityIndex => "monotonicity_index_neo_batch_f64",
            F64Kernel::PremierRsiOscillator => "premier_rsi_oscillator_neo_batch_f64",
            F64Kernel::PrettyGoodOscillator => "pretty_good_oscillator_neo_batch_f64",
            F64Kernel::PriceDensityMarketNoise => "price_density_market_noise_neo_batch_f64",
            F64Kernel::ProjectionOscillator => "projection_oscillator_neo_batch_f64",
            F64Kernel::QqeWeightedOscillator => "qqe_weighted_oscillator_neo_batch_f64",
            F64Kernel::RogersSatchellVolatility => "rogers_satchell_volatility_neo_batch_f64",
            F64Kernel::RollingSkewnessKurtosis => "rolling_skewness_kurtosis_neo_batch_f64",
            F64Kernel::SmoothTheilSen => "smooth_theil_sen_neo_batch_f64",
            // ------------------------------------------ closer 2, round 3
            F64Kernel::NeighboringTrailingStop => "neighboring_trailing_stop_neo_batch_f64",
            F64Kernel::NonlinearRegressionZeroLagMovingAverage => {
                "nonlinear_regression_zero_lag_moving_average_neo_batch_f64"
            }
            F64Kernel::NormalizedResonator => "normalized_resonator_neo_batch_f64",
            F64Kernel::NormalizedVolumeTrueRange => {
                "normalized_volume_true_range_neo_batch_f64"
            }
            F64Kernel::PossibleRsi => "possible_rsi_neo_batch_f64",
            F64Kernel::PriceMovingAverageRatioPercentile => {
                "price_moving_average_ratio_percentile_neo_batch_f64"
            }
            F64Kernel::RangeBreakoutSignals => "range_breakout_signals_neo_batch_f64",
            F64Kernel::RangeFilteredTrendSignals => {
                "range_filtered_trend_signals_neo_batch_f64"
            }
            F64Kernel::RegressionSlopeOscillator => "regression_slope_oscillator_neo_batch_f64",
            F64Kernel::RelativeStrengthIndexWaveIndicator => {
                "relative_strength_index_wave_indicator_neo_batch_f64"
            }
            // ------------------------------------------ closer 4, round 3
            F64Kernel::Bandpass => "bandpass_neo_batch_f64",
            F64Kernel::BuffAverages => "buff_averages_neo_batch_f64",
            F64Kernel::CoraWave => "cora_wave_neo_batch_f64",
            F64Kernel::Dma => "dma_neo_batch_f64",
            F64Kernel::FvgTrailingStop => "fvg_trailing_stop_neo_batch_f64",
            F64Kernel::Halftrend => "halftrend_neo_batch_f64",
            F64Kernel::ModGodMode => "mod_god_mode_neo_batch_f64",
            F64Kernel::Ott => "ott_neo_batch_f64",
            F64Kernel::Otto => "otto_neo_batch_f64",
            F64Kernel::Prb => "prb_neo_batch_f64",
            // ---------------------------------------------- closer 6, round 3
            F64Kernel::ElasticVolumeWeightedMovingAverage => {
                "elastic_volume_weighted_moving_average_neo_batch_f64"
            }
            F64Kernel::EmaDeviationCorrectedT3 => "ema_deviation_corrected_t3_neo_batch_f64",
            F64Kernel::LogarithmicMovingAverage => "logarithmic_moving_average_neo_batch_f64",
            F64Kernel::NOrderEma => "n_order_ema_neo_batch_f64",
            F64Kernel::VolatilityAdjustedMa => "volatility_adjusted_ma_neo_batch_f64",
            F64Kernel::WaveSmoother => "wave_smoother_neo_batch_f64",

            // ------------------------------------------------ closer 3, round 3
            F64Kernel::ReversalSignals => "reversal_signals_neo_batch_f64",
            F64Kernel::TrendFollower => "trend_follower_neo_batch_f64",
            F64Kernel::VdubusDivergenceWavePatternGenerator => {
                "vdubus_divergence_wave_pattern_generator_neo_batch_f64"
            }
            F64Kernel::VolatilityRatioAdaptiveRsx => "volatility_ratio_adaptive_rsx_neo_batch_f64",
            F64Kernel::VolumeEnergyReservoirs => "volume_energy_reservoirs_neo_batch_f64",
            F64Kernel::VolumeWeightedRelativeStrengthIndex => {
                "volume_weighted_relative_strength_index_neo_batch_f64"
            }
            F64Kernel::VolumeWeightedStochasticRsi => {
                "volume_weighted_stochastic_rsi_neo_batch_f64"
            }
            F64Kernel::ZigZagChannels => "zig_zag_channels_neo_batch_f64",
            F64Kernel::Alphatrend => "alphatrend_neo_batch_f64",
            F64Kernel::Avsl => "avsl_neo_batch_f64",

            // ------------------------------------------ closer 1, round 3
            F64Kernel::FibonacciEntryBands => "fibonacci_entry_bands_neo_batch_f64",
            F64Kernel::GoertzelCycleCompositeWave => "goertzel_cycle_composite_wave_neo_batch_f64",
            F64Kernel::HalfCausalEstimator => "half_causal_estimator_neo_batch_f64",
            F64Kernel::IchimokuOscillator => "ichimoku_oscillator_neo_batch_f64",
            F64Kernel::InsyncIndex => "insync_index_neo_batch_f64",
            F64Kernel::LinearRegressionIntensity => "linear_regression_intensity_neo_batch_f64",
            F64Kernel::MacdWaveSignalPro => "macd_wave_signal_pro_neo_batch_f64",
            F64Kernel::MesaStochasticMultiLength => "mesa_stochastic_multi_length_neo_batch_f64",
            F64Kernel::MovingAverageCrossProbability => "moving_average_cross_probability_neo_batch_f64",
            F64Kernel::MultiLengthStochasticAverage => "multi_length_stochastic_average_neo_batch_f64",
        }
    }

    /// The fixed per-thread window bound this kernel carries, if any.
    ///
    /// A kernel with a bound keeps a ring in a per-thread local array sized at
    /// compile time, so the bound is a property of the compiled kernel and NOT
    /// of the caller. [`CudaF64Indicators::sweep`] refuses a larger period by
    /// name rather than truncating the window or moving the sweep to the host.
    pub fn max_period(self) -> Option<usize> {
        match self {
            F64Kernel::Mfi => Some(MFI_MAX_PERIOD),
            F64Kernel::Adxr => Some(ADXR_MAX_PERIOD),
            F64Kernel::Ehma => Some(EHMA_MAX_PERIOD),
            F64Kernel::Devstop => Some(S2_RING_MAX_PERIOD),
            F64Kernel::ChandelierExit => Some(S2_RING_MAX_PERIOD),
            // ------------------------------------------------------------- shard 1 (S1)
            // Each of these keeps a fixed per-thread array whose
            // length is a function of `period`, so the bound belongs
            // to the compiled kernel and an oversized period is
            // REFUSED BY NAME rather than truncated or moved to the
            // host. The numbers match the `#define`s in the .cu files.
            F64Kernel::Chop => Some(CHOP_MAX_PERIOD),
            F64Kernel::Hma => Some(HMA_MAX_PERIOD),
            F64Kernel::Edcf => Some(EDCF_MAX_PERIOD),
            F64Kernel::Alma => Some(ALMA_MAX_PERIOD),
            F64Kernel::EhlersItrend => Some(S2_RING_MAX_PERIOD),
            F64Kernel::Sama => Some(S2_RING_MAX_PERIOD),
            F64Kernel::Nama => Some(S2_RING_MAX_PERIOD),
            F64Kernel::Pwma => Some(S2_RING_MAX_PERIOD),
            F64Kernel::Tradjema => Some(S2_RING_MAX_PERIOD),
            F64Kernel::Maaq => Some(S2_RING_MAX_PERIOD),
            F64Kernel::Reflex => Some(S2_RING_MAX_PERIOD),
            // ------------------------------------------------------- closer 3
            // `medium_ad_neo_batch_f64` copies the whole trailing window into
            // a per-thread local array before selecting the median, so the
            // bound belongs to the compiled kernel and an oversized period is
            // REFUSED BY NAME rather than truncating the window into a
            // different indicator. The number matches the `MEDIUM_AD_MAX_PERIOD`
            // `#define` in kernels/cuda/medium_ad_kernel.cu.
            F64Kernel::MediumAd => Some(MEDIUM_AD_MAX_PERIOD),
            // ------------------------------------------------------ closer 6
            // `emd_scalar_into` keeps a `2 * period` ring of bandpass values
            // (`per_mid`, emd.rs:536) plus two 50-long rings. The kernel sizes
            // the bandpass ring at `2 * EMD_F64_MAX_PERIOD` doubles at compile
            // time, so the bound belongs to the compiled kernel and an
            // oversized period is REFUSED BY NAME rather than truncated or
            // moved to the host.
            F64Kernel::Emd => Some(S2_RING_MAX_PERIOD),
            // `nwe_prepare:412` builds a `lookback`-long weight vector and
            // `nwe_compute_scalar_no_nan:523` a 499-long residual ring. The
            // kernel sizes both at compile time (NWE_F64_LOOKBACK = 500, the
            // CPU default), so the bound belongs to the compiled kernel.
            F64Kernel::NadarayaWatsonEnvelope => Some(NWE_MAX_LOOKBACK),
            // `stoch` keeps `slowk_period` and `slowd_period` rings, both
            // pinned at the CPU default 3, and the CPU itself caps its stack
            // buffers at 64 (stoch.rs:750-760). `keltner` carries no ring at
            // all -- its middle band is an EMA, not a sliding window -- so
            // neither declares a bound.
            // ----------------------------------------------- closer 5
            // `trima` keeps an `m2 = period - (period+1)/2 + 1` deep ring
            // in a per-thread array, so the bound belongs to the COMPILED
            // kernel and not to the caller. Matches NEO_TRIMA_MAX_PERIOD
            // in kernels/cuda/moving_averages/trima_kernel.cu.
            F64Kernel::Trima => Some(TRIMA_MAX_PERIOD),
            // ------------------------------------------------------ closer 1
            // None of the twenty declares a bound, and the distinction is the
            // SWEPT period, not the presence of a local array. Fifteen ARE
            // period-invariant (`is_period_invariant` below): their rings --
            // `AMACD_NEO_LENGTH`, `AMO_NEO_LENGTH`, `ATRP_NEO_ATR_LEN`,
            // `COPPOCK_NEO_MA`, `DIDI_NEO_LONG`, `DISP_NEO_LOOKBACK` -- are
            // sized from the CPU's NAMED parameters, which the sweep never
            // touches, so no caller-supplied number can overrun them.
            // The five that DO read `period` -- `BullPowerVsBearPower`, `Cg`,
            // `Dm`, `DonchianChannelWidth`, `Dpo` -- carry NO per-thread array
            // at all; each rescans its window straight out of global memory.
            // So there is nothing here for an oversized period to truncate.
            // ------------------------------------------- closer 6, round 2
            // `net_myrsi_neo_batch_f64` keeps TWO `period`-wide per-thread
            // rings -- the diff ring and the myrsi ring (net_myrsi.rs:300,
            // :304) -- and `period` IS the swept parameter, so an oversized
            // period would overrun them. Bound belongs to the compiled kernel
            // and matches NET_MYRSI_NEO_MAX_PERIOD in
            // kernels/cuda/net_myrsi_kernel.cu; an oversized period is
            // REFUSED BY NAME rather than truncated into a different
            // indicator.
            //
            // The other seven declare nothing, and for two different reasons.
            // `Msw` and `Rvi` are period-SWEPT but carry NO per-thread array:
            // msw regenerates its sine/cosine table by re-running the CPU's
            // `ang += step` accumulation inside the inner loop, and rvi reads
            // its variance window straight out of global memory. `Qqe`,
            // `Srsi`, `Vlma`, `Stc` and `YangZhangVolatility` are
            // period-INVARIANT, so their fixed rings are sized from the CPU's
            // NAMED parameters, which no caller-supplied number can reach.
            F64Kernel::NetMyrsi => Some(S2_RING_MAX_PERIOD),
            // ------------------------------------------------ closer 3, round 2
            //
            // Only TWO of this closer's twenty-five carry a bound, and that
            // asymmetry is the point: the other twenty-three pin every window
            // at a CPU DEFAULT, so no caller-supplied number reaches a
            // per-thread array at all and there is nothing for a bound to
            // refuse. These two are the ones whose ring depth is a function of
            // the SWEPT period.
            F64Kernel::DirectionalImbalanceIndex => Some(DII_MAX_PERIOD),
            F64Kernel::CandleStrengthOscillator => Some(CSO_MAX_PERIOD),
            // ---------------------------------------------- closer 6, round 3
            //
            // Two of this closer's six. The other four hold no per-thread array
            // whose length is a function of the SWEPT period:
            // `volatility_adjusted_ma`'s deques are `vol_period` = 51, a CPU
            // DEFAULT the sweep cannot move; `elastic_volume_weighted_moving_
            // average` reads its rolling window straight out of the volume
            // series instead of keeping a ring; and `n_order_ema` /
            // `ema_deviation_corrected_t3` carry scalars only.
            F64Kernel::LogarithmicMovingAverage => Some(LMA_MAX_PERIOD),
            F64Kernel::WaveSmoother => Some(WS_MAX_PERIOD),
            // ---------------------------------------------- closer 4, round 3
            //
            // Two of this round's ten. Both keep a per-thread ring whose depth
            // is `round(sqrt(swept period))` -- cora_wave's smoothing WMA
            // (cora_wave.rs:378) and dma's difference ring (dma.rs:420) -- so
            // the bound belongs to the compiled kernel and an oversized period
            // is REFUSED BY NAME rather than truncated or moved to the host.
            // The numbers match the `#define`s in the two `.cu` files.
            //
            // The other eight hold no per-thread array whose length a caller
            // can move: bandpass, ott and otto carry scalars and a NINE-wide
            // CMO ring that is a constant of the indicator; buff_averages
            // reads its window straight out of global memory; and
            // fvg_trailing_stop, halftrend, mod_god_mode and prb are
            // PERIOD-INVARIANT, so every window they keep is a CPU default the
            // sweep cannot move.
            F64Kernel::CoraWave => Some(CORA_WAVE_MAX_PERIOD),
            F64Kernel::Dma => Some(DMA_MAX_PERIOD),
            // ------------------------------------------------ closer 3, round 3
            // Only TWO of this closer's ten carry a bound, and that asymmetry
            // is the point: the other eight pin every window at a CPU DEFAULT,
            // so no caller-supplied number reaches a per-thread array at all
            // and there is nothing for a bound to refuse. These two size a ring
            // from the SWEPT period.
            F64Kernel::VolatilityRatioAdaptiveRsx => Some(VRARSX_MAX_PERIOD),
            F64Kernel::Alphatrend => Some(ALPHATREND_MAX_PERIOD),
            _ => None,
        }
    }

    /// `true` when the CPU reference this kernel mirrors does not read the
    /// swept `period` at all, so every row of the sweep is byte-identical.
    ///
    /// This is FAITHFUL, not a defect to be fixed here: `compute_obv_batch`
    /// (cpu_batch.rs:3897) takes `|_params|`, while
    /// `vwap`/`medprice`/`wclprice` have no period parameter. The CPU emits
    /// identical columns for those ids, so their kernels must emit identical
    /// rows. TSI is deliberately absent: NeoEthos maps its sweep to the named
    /// `long_period`/`short_period` pair before either lane runs.
    pub fn is_period_invariant(self) -> bool {
        matches!(
            self,
            F64Kernel::Adosc
                // ------------------------------------ closer 5, round 3
                // Seven of the nine. `Rsmk` reads a parameter literally named
                // `period` (cpu_batch.rs:16479) and `CorrectedMovingAverage`
                // reaches this crate through `ma(ma_type, period, ..)`
                // (ma.rs:263); both are deliberately absent.
                | F64Kernel::SqueezeMomentum
                | F64Kernel::Uma
                | F64Kernel::Lpc
                | F64Kernel::Mab
                | F64Kernel::Macz
                | F64Kernel::Vwmacd
                | F64Kernel::EhlersUndersampledDoubleMovingAverage
                // -------------------------- closer 5, round 2 (invariant)
                // Sixteen of the seventeen. `Supertrend` reads a parameter
                // literally named `period` (cpu_batch.rs:6348) and is
                // deliberately absent.
                | F64Kernel::SmoothedGaussianTrendFilter
                | F64Kernel::SpearmanCorrelation
                | F64Kernel::SqueezeIndex
                | F64Kernel::StandardizedPsarOscillator
                | F64Kernel::StatisticalTrailingStop
                | F64Kernel::StochasticAdaptiveD
                | F64Kernel::StochasticConnorsRsi
                | F64Kernel::StochasticDistance
                | F64Kernel::StochasticMoneyFlowIndex
                | F64Kernel::SupertrendOscillator
                | F64Kernel::SupertrendRecovery
                | F64Kernel::TrendFlowTrail
                | F64Kernel::TwiggsMoneyFlow
                | F64Kernel::VolatilityQualityIndex
                | F64Kernel::VwapDeviationOscillator
                | F64Kernel::VwapZscoreWithSignals
                | F64Kernel::Obv
                | F64Kernel::Vwap
                | F64Kernel::Medprice
                | F64Kernel::Wclprice
                // ------------------------------------------------------------- shard 1 (S1)
                | F64Kernel::Apo
                | F64Kernel::Vidya
                | F64Kernel::Gatorosc
                | F64Kernel::Ppo
                | F64Kernel::Pma
                | F64Kernel::Alligator
                | F64Kernel::Nvi
                | F64Kernel::Stochf
                | F64Kernel::Emv
                | F64Kernel::Kvo
                // ------------------------------------------------------------- shard 4 (S4)
                | F64Kernel::Macd
                | F64Kernel::Cksp
                | F64Kernel::IftRsi
                | F64Kernel::Vpci
                | F64Kernel::TtmSqueeze
                | F64Kernel::DamianiVolatmeter
                | F64Kernel::Wavetrend
                // THE ONE BUILD: `Aroon` was here and the claim is FALSE.
                // This predicate's own contract is "the CPU reference does not
                // read the swept period at all". `compute_aroon_batch`
                // (cpu_batch.rs:5959) reads
                // `get_usize_param("aroon", params, "length", 14)` ONCE PER
                // COMBO, and `aroon_neo_batch_f64` (kernels/cuda/aroon_kernel.cu:312)
                // correspondingly reads `periods[combo]` -- so the rows of an
                // aroon sweep genuinely differ. Nothing consumes this
                // predicate today, which is exactly why the wrong answer was
                // survivable; the moment anything uses it to collapse
                // duplicate launches, a real aroon sweep would silently become
                // one repeated row. Removed rather than left as a latent trap.
                | F64Kernel::Acosc
                | F64Kernel::Ad
                | F64Kernel::CciCycle
                // ---------------------------------------------------- closer 6
                // `compute_stoch_batch` (cpu_batch.rs:5580-5582) reads
                // `fastk_period`, `slowk_period` and `slowd_period` and NEVER
                // `period`, so five swept periods produce five identical CPU
                // columns. `emd` and `keltner` are NOT here: both batches read
                // a parameter literally named `period` (:14532, :6212).
                //
                // `compute_nadaraya_watson_envelope_batch`
                // (cpu_batch.rs:15619-15621) reads `bandwidth`, `multiplier`
                // and `lookback` -- also never `period`.
                | F64Kernel::Stoch
                | F64Kernel::NadarayaWatsonEnvelope
                // ---------------------------------------------------- closer 3
                // l1_ehlers_phasor reads `domestic_cycle_length`,
                // l2_ehlers_signal_to_noise reads `source`/`smooth_period`,
                // kairi_relative_index reads `length`/`ma_type`, and marketefi
                // has no length parameter and no `compute_*_batch` at all.
                // None of the four reads `period`; the other two closer-3
                // variants (linear_correlation_oscillator, medium_ad) DO, and
                // are deliberately absent.
                | F64Kernel::L1EhlersPhasor
                | F64Kernel::L2EhlersSignalToNoise
                | F64Kernel::KairiRelativeIndex
                | F64Kernel::Marketefi
                | F64Kernel::OnBalanceVolumeOscillator
                // --------------------------------------------- closer 5
                // Nine of the eleven. `Trima` and `VolumeWeightedRsi` are
                // genuinely period-swept and are deliberately absent.
                | F64Kernel::Velocity
                | F64Kernel::VelocityAccelerationIndicator
                | F64Kernel::VelocityAccelerationConvergenceDivergenceIndicator
                | F64Kernel::TrendDirectionForceIndex
                | F64Kernel::TrendContinuationFactor
                | F64Kernel::TrendTriggerFactor
                | F64Kernel::VolumeZoneOscillator
                | F64Kernel::Vosc
                | F64Kernel::Ultosc
        
                // ------------------------------------------------------------ closer 2
                | F64Kernel::EhlersDetrendingFilter
                | F64Kernel::EhlersSimpleCycleIndicator
                | F64Kernel::EhlersSmoothedAdaptiveMomentum
                | F64Kernel::EwmaVolatility
                | F64Kernel::FractalDimensionIndex
                | F64Kernel::GopalakrishnanRangeIndex
                | F64Kernel::GarmanKlassVolatility
                | F64Kernel::ImpulseMacd
                | F64Kernel::Hypertrend
                | F64Kernel::EmdTrend
                | F64Kernel::EhlersPma
                // -------------------------------------------- closer 1
                | F64Kernel::AbsoluteStrengthIndexOscillator
                | F64Kernel::AccumulationSwingIndex
                | F64Kernel::AdaptiveBandpassTriggerOscillator
                | F64Kernel::AdaptiveBoundsRsi
                | F64Kernel::AdaptiveMacd
                | F64Kernel::AdaptiveMomentumOscillator
                | F64Kernel::AdvanceDeclineLine
                | F64Kernel::AndeanOscillator
                | F64Kernel::AtrPercentile
                | F64Kernel::Bop
                | F64Kernel::Coppock
                | F64Kernel::DailyFactor
                | F64Kernel::DecisionpointBreadthSwenlinTradingOscillator
                | F64Kernel::DidiIndex
                | F64Kernel::DisparityIndex
        
                // ---------------------------------------------------------- closer 2b
                | F64Kernel::ForwardBackwardExponentialOscillator
                | F64Kernel::GmmaOscillator
                | F64Kernel::EvasiveSupertrend
                // --------------------------------------- closer 6, round 2
                // Five of the eight. `Msw`, `Rvi` and `NetMyrsi` are
                // genuinely period-swept and are deliberately absent.
                | F64Kernel::YangZhangVolatility
                | F64Kernel::Qqe
                | F64Kernel::Srsi
                | F64Kernel::Vlma
                | F64Kernel::Stc
                // --------------------------------------- closer 2, round 2
                // Three of the ten. The other seven read a genuine window
                // parameter -- `period` for kaufmanstop and dual_ulcer_index,
                // `length` for sgf, polynomial_regression_extrapolation,
                // hull_butterfly_oscillator, range_oscillator and
                // market_structure_trailing_stop -- and are deliberately
                // absent.
                | F64Kernel::Mwdx
                | F64Kernel::Lrsi
                | F64Kernel::Pivot
                // ------------------------------------------ closer 3, round 2
                //
                // Twenty-one of this closer's twenty-five. Their CPU batch
                // functions read NAMED parameters and never `period`, AND
                // their kernels pin every window at the CPU DEFAULT rather
                // than reading `periods[combo]` -- the two halves have to
                // agree or the claim is false.
                //
                // `vertical_horizontal_filter` is the one that is NOT here
                // despite reading `length` (cpu_batch.rs:13371): its kernel
                // was already written to take the swept int AS the length, and
                // this flag reports what the KERNEL does, not what would have
                // been tidy.
                //
                // The named parameters: `length`/`mult`/`alpha`/`beta`
                // for adjustable_ma_alternating_extremities (:6417),
                // `length`/`lag` for autocorrelation_indicator (:7798),
                // `hv_length`/`rank_length` for historical_volatility_rank
                // (:6664), `length`/`annual_length` for
                // historical_volatility_percentile (:9660),
                // `short_cycle_length`/`medium_cycle_length` for
                // cycle_channel_oscillator (:9969), `rsi_period`/
                // `volatility_period` for dynamic_momentum_index (:6874),
                // `alpha` for both Ehlers adaptive indicators (:15801,
                // :10126), `length` for
                // ehlers_data_sampling_relative_strength_indicator (:8114),
                // `exp_rate`/`initial_distance` for exponential_trend (:4404),
                // `length`/`multiplier`/`atr_length`/`smooth` for
                // geometric_bias_oscillator (:5041), `length`/`length_ma` for
                // intraday_momentum_index (:13401), `entry_level`/
                // `exit_level` for cyberpunk_value_trend_analyzer (:14658),
                // `lookback`/`atr_multiplier` for fvg_positioning_average
                // (:14802), `fast_length`/`slow_length` for hema_trend_levels
                // (:13557), `left_bars`/`right_bars`/`level` for
                // fibonacci_trailing_stop (:8807), `length`/`mult`/
                // `rsi_period` for grover_llorens_cycle_oscillator (:9073),
                // `len_bs`/`len_bs_ma`/`len_di_ma` for demand_index (:13647),
                // `adaptive_length`/`stc_length` for
                // adaptive_schaff_trend_cycle (:12645), `high_pass_length`/
                // `low_pass_length`/`gain` for
                // ehlers_linear_extrapolation_predictor (:9193), and
                // `min_period`/`max_period`/`avg_length` for
                // ehlers_autocorrelation_periodogram (:9125) -- note that
                // THOSE `min_period` / `max_period` bound the SPECTRUM the
                // indicator scans and are not the lane's swept period.
                //
                // The THREE that are genuinely period-swept -- BullsVBears,
                // CandleStrengthOscillator and DirectionalImbalanceIndex --
                // read a parameter literally named `period` and are
                // deliberately absent, as is VerticalHorizontalFilter for the
                // reason above.
                | F64Kernel::AdjustableMaAlternatingExtremities
                | F64Kernel::AutocorrelationIndicator
                | F64Kernel::HistoricalVolatilityRank
                | F64Kernel::HistoricalVolatilityPercentile
                | F64Kernel::CycleChannelOscillator
                | F64Kernel::DynamicMomentumIndex
                | F64Kernel::EhlersAdaptiveCg
                | F64Kernel::EhlersAdaptiveCyberCycle
                | F64Kernel::EhlersDataSamplingRelativeStrengthIndicator
                | F64Kernel::ExponentialTrend
                | F64Kernel::GeometricBiasOscillator
                | F64Kernel::IntradayMomentumIndex
                | F64Kernel::CyberpunkValueTrendAnalyzer
                | F64Kernel::FvgPositioningAverage
                | F64Kernel::HemaTrendLevels
                | F64Kernel::FibonacciTrailingStop
                | F64Kernel::GroverLlorensCycleOscillator
                | F64Kernel::DemandIndex
                | F64Kernel::AdaptiveSchaffTrendCycle
                | F64Kernel::EhlersLinearExtrapolationPredictor
                | F64Kernel::EhlersAutocorrelationPeriodogram
                | F64Kernel::IctPropulsionBlock
                // ------------------------------------ closer 4, round 2
                // All fifteen: every CPU batch reads NAMED windows and
                // never `period` -- see the enum note.
                | F64Kernel::KasePeakOscillatorWithDivergences
                | F64Kernel::KeltnerChannelWidthOscillator
                | F64Kernel::Kst
                | F64Kernel::LeavittConvolutionAcceleration
                | F64Kernel::MarketMeannessIndex
                | F64Kernel::MarketStructureConfluence
                | F64Kernel::MonotonicityIndex
                | F64Kernel::PremierRsiOscillator
                | F64Kernel::PrettyGoodOscillator
                | F64Kernel::PriceDensityMarketNoise
                | F64Kernel::ProjectionOscillator
                | F64Kernel::QqeWeightedOscillator
                | F64Kernel::RogersSatchellVolatility
                | F64Kernel::RollingSkewnessKurtosis
                | F64Kernel::SmoothTheilSen
                // ------------------------------------ closer 4, round 3
                //
                // Five of this round's ten. Each CPU batch reads NAMED
                // windows and never `period` -- see the enum note for the
                // parameter list of each. The other five (Bandpass,
                // BuffAverages, CoraWave, Dma, Ott) ARE swept and are
                // deliberately absent.
                | F64Kernel::FvgTrailingStop
                | F64Kernel::Halftrend
                | F64Kernel::ModGodMode
                | F64Kernel::Otto
                | F64Kernel::Prb
                // ------------------------------------------ closer 2, round 3
                //
                // NINE of the ten. Every one of those nine CPU batches reads
                // its own NAMED windows and never `period`:
                // neighboring_trailing_stop reads buffer_size/k/percentile/
                // smooth, nonlinear_regression_zero_lag_moving_average reads
                // zlma_period/regression_period, normalized_resonator reads
                // source/delta/lookback_mult/signal_length,
                // normalized_volume_true_range reads true_range_style/
                // outlier_range/atr_length/volume_length,
                // price_moving_average_ratio_percentile reads ma_length/
                // ma_type/pmarp_lookback/line_mode, range_breakout_signals
                // reads range_length/confirmation_length,
                // range_filtered_trend_signals reads the six kalman and
                // supertrend names, regression_slope_oscillator reads
                // min_range/max_range/step/signal_line, and
                // relative_strength_index_wave_indicator reads rsi_length and
                // length1..length4. Five swept periods therefore give five
                // identical CPU columns and five identical kernel rows.
                //
                // `PossibleRsi` is DELIBERATELY ABSENT: its CPU batch reads a
                // parameter literally named `period` (default 32) and that
                // parameter is the RSI length, so every row of the sweep is a
                // different column and the sweep does real work.
                | F64Kernel::NeighboringTrailingStop
                | F64Kernel::NonlinearRegressionZeroLagMovingAverage
                | F64Kernel::NormalizedResonator
                | F64Kernel::NormalizedVolumeTrueRange
                | F64Kernel::PriceMovingAverageRatioPercentile
                | F64Kernel::RangeBreakoutSignals
                | F64Kernel::RangeFilteredTrendSignals
                | F64Kernel::RegressionSlopeOscillator
                | F64Kernel::RelativeStrengthIndexWaveIndicator
                // ------------------------------- closer 3, round 3
                // Eight of this closer's ten. Their CPU batch functions read
                // NAMED parameters and never `period`, AND their kernels pin
                // every window at the CPU DEFAULT rather than reading
                // `periods[combo]` -- the two halves have to agree or the
                // claim is false. `VolatilityRatioAdaptiveRsx` and
                // `Alphatrend` are deliberately absent: both read a parameter
                // literally named `period` and both size a ring from it.
                | F64Kernel::ReversalSignals
                | F64Kernel::TrendFollower
                | F64Kernel::VdubusDivergenceWavePatternGenerator
                | F64Kernel::VolumeEnergyReservoirs
                | F64Kernel::VolumeWeightedRelativeStrengthIndex
                | F64Kernel::VolumeWeightedStochasticRsi
                | F64Kernel::ZigZagChannels
                | F64Kernel::Avsl

                // ------------------------------------ closer 1, round 3
                // All ten. Every CPU batch reads NAMED windows and never
                // `period` -- see the enum note.
                | F64Kernel::FibonacciEntryBands
                | F64Kernel::GoertzelCycleCompositeWave
                | F64Kernel::HalfCausalEstimator
                | F64Kernel::IchimokuOscillator
                | F64Kernel::InsyncIndex
                | F64Kernel::LinearRegressionIntensity
                | F64Kernel::MacdWaveSignalPro
                | F64Kernel::MesaStochasticMultiLength
                | F64Kernel::MovingAverageCrossProbability
                | F64Kernel::MultiLengthStochasticAverage
        )
    }

    pub fn indicator_id(self) -> &'static str {
        match self {
            F64Kernel::Sma => "sma",
            F64Kernel::Adosc => "adosc",
            // ------------------------------------------ closer 5, round 3
            F64Kernel::Rsmk => "rsmk",
            F64Kernel::SqueezeMomentum => "squeeze_momentum",
            F64Kernel::Uma => "uma",
            F64Kernel::Lpc => "lpc",
            F64Kernel::Mab => "mab",
            F64Kernel::Macz => "macz",
            F64Kernel::Vwmacd => "vwmacd",
            F64Kernel::CorrectedMovingAverage => "corrected_moving_average",
            F64Kernel::EhlersUndersampledDoubleMovingAverage => "ehlers_undersampled_double_moving_average",
            // ---------------------------------- closer 5, round 2 (ids)
            F64Kernel::SmoothedGaussianTrendFilter => "smoothed_gaussian_trend_filter",
            F64Kernel::SpearmanCorrelation => "spearman_correlation",
            F64Kernel::SqueezeIndex => "squeeze_index",
            F64Kernel::StandardizedPsarOscillator => "standardized_psar_oscillator",
            F64Kernel::StatisticalTrailingStop => "statistical_trailing_stop",
            F64Kernel::StochasticAdaptiveD => "stochastic_adaptive_d",
            F64Kernel::StochasticConnorsRsi => "stochastic_connors_rsi",
            F64Kernel::StochasticDistance => "stochastic_distance",
            F64Kernel::StochasticMoneyFlowIndex => "stochastic_money_flow_index",
            F64Kernel::Supertrend => "supertrend",
            F64Kernel::SupertrendOscillator => "supertrend_oscillator",
            F64Kernel::SupertrendRecovery => "supertrend_recovery",
            F64Kernel::TrendFlowTrail => "trend_flow_trail",
            F64Kernel::TwiggsMoneyFlow => "twiggs_money_flow",
            F64Kernel::VolatilityQualityIndex => "volatility_quality_index",
            F64Kernel::VwapDeviationOscillator => "vwap_deviation_oscillator",
            F64Kernel::VwapZscoreWithSignals => "vwap_zscore_with_signals",
            F64Kernel::Ema => "ema",
            F64Kernel::Rsi => "rsi",
            F64Kernel::Roc => "roc",
            F64Kernel::Mom => "mom",
            F64Kernel::Atr => "atr",
            F64Kernel::Adx => "adx",
            F64Kernel::Willr => "willr",
            F64Kernel::Cci => "cci",
            F64Kernel::Mfi => "mfi",
            F64Kernel::Tsi => "tsi",
            F64Kernel::Obv => "obv",
            F64Kernel::Vwap => "vwap",
            F64Kernel::Wma => "wma",
            F64Kernel::Wilders => "wilders",
            F64Kernel::Smma => "smma",
            F64Kernel::Dema => "dema",
            F64Kernel::Tema => "tema",
            F64Kernel::Zlema => "zlema",
            F64Kernel::Vwma => "vwma",
            F64Kernel::Efi => "efi",
            F64Kernel::Natr => "natr",
            F64Kernel::Adxr => "adxr",
            F64Kernel::Medprice => "medprice",
            F64Kernel::Wclprice => "wclprice",
            F64Kernel::Midpoint => "midpoint",
            F64Kernel::Midprice => "midprice",
            F64Kernel::Rocp => "rocp",
            F64Kernel::Rocr => "rocr",
            F64Kernel::Sqwma => "sqwma",
            F64Kernel::Deviation => "deviation",
            F64Kernel::MeanAd => "mean_ad",
            F64Kernel::Ao => "ao",
            F64Kernel::LinearregSlope => "linearreg_slope",
            F64Kernel::Tsf => "tsf",
            F64Kernel::Highpass => "highpass",
            F64Kernel::Decycler => "decycler",
            F64Kernel::Supersmoother => "supersmoother",
            F64Kernel::Tilson => "tilson",
            F64Kernel::Wad => "wad",
            F64Kernel::Sar => "sar",
            F64Kernel::Dti => "dti",
            F64Kernel::Zscore => "zscore",
            F64Kernel::Pfe => "pfe",
            F64Kernel::Chande => "chande",
            F64Kernel::Di => "di",
            F64Kernel::Kdj => "kdj",
            F64Kernel::Aso => "aso",
            F64Kernel::Wto => "wto",
            F64Kernel::RangeFilter => "range_filter",
            F64Kernel::CorrelationCycle => "correlation_cycle",
            F64Kernel::Mama => "mama",
            F64Kernel::VolumeAdjustedMa => "volume_adjusted_ma",
            F64Kernel::ReverseRsi => "reverse_rsi",
            F64Kernel::EhlersEcema => "ehlers_ecema",
            F64Kernel::Devstop => "devstop",
            F64Kernel::ChandelierExit => "chandelier_exit",
            F64Kernel::Minmax => "minmax",
            // ------------------------------------------------------------- shard 1 (S1)
            F64Kernel::Apo => "apo",
            F64Kernel::Vidya => "vidya",
            F64Kernel::Gatorosc => "gatorosc",
            F64Kernel::Ppo => "ppo",
            F64Kernel::Pma => "pma",
            F64Kernel::Kama => "kama",
            F64Kernel::Linreg => "linreg",
            F64Kernel::Edcf => "edcf",
            F64Kernel::Alma => "alma",
            F64Kernel::Hma => "hma",
            F64Kernel::Kurtosis => "kurtosis",
            F64Kernel::Alligator => "alligator",
            F64Kernel::Nvi => "nvi",
            F64Kernel::Fisher => "fisher",
            F64Kernel::Safezonestop => "safezonestop",
            F64Kernel::Chop => "chop",
            F64Kernel::Stochf => "stochf",
            F64Kernel::Emv => "emv",
            F64Kernel::Kvo => "kvo",
            // ------------------------------------------------------- shard 4
            F64Kernel::Er => "er",
            F64Kernel::LinearregAngle => "linearreg_angle",
            F64Kernel::LinearregIntercept => "linearreg_intercept",
            F64Kernel::Highpass2Pole => "highpass_2_pole",
            F64Kernel::Supersmoother3Pole => "supersmoother_3_pole",
            F64Kernel::Cwma => "cwma",
            F64Kernel::Cmo => "cmo",
            F64Kernel::Stddev => "stddev",
            F64Kernel::Ui => "ui",
            F64Kernel::BollingerBands => "bollinger_bands",
            F64Kernel::Ehma => "ehma",
            F64Kernel::Macd => "macd",
            F64Kernel::IftRsi => "ift_rsi",
            F64Kernel::DamianiVolatmeter => "damiani_volatmeter",
            F64Kernel::Wavetrend => "wavetrend",
            F64Kernel::Dx => "dx",
            F64Kernel::Frama => "frama",
            F64Kernel::Cksp => "cksp",
            F64Kernel::TtmSqueeze => "ttm_squeeze",
            F64Kernel::Mass => "mass",
            F64Kernel::Aroon => "aroon",
            F64Kernel::Acosc => "acosc",
            F64Kernel::Vpci => "vpci",
            F64Kernel::Ad => "ad",
            F64Kernel::Dvdiqqe => "dvdiqqe",
            F64Kernel::CciCycle => "cci_cycle",
            F64Kernel::Rsx => "rsx",
            F64Kernel::Trix => "trix",
            F64Kernel::Vpt => "vpt",
            F64Kernel::Pvi => "pvi",
            F64Kernel::EhlersItrend => "ehlers_itrend",
            F64Kernel::EhlersKama => "ehlers_kama",
            F64Kernel::Sama => "sama",
            F64Kernel::Nama => "nama",
            F64Kernel::Pwma => "pwma",
            F64Kernel::Tradjema => "tradjema",
            F64Kernel::Maaq => "maaq",
            F64Kernel::Jma => "jma",
            F64Kernel::Reflex => "reflex",
            F64Kernel::Gaussian => "gaussian",
            F64Kernel::Fwma => "fwma",
            F64Kernel::Hwma => "hwma",
            F64Kernel::Jsa => "jsa",
            F64Kernel::Nma => "nma",
            F64Kernel::Swma => "swma",
            F64Kernel::Trendflex => "trendflex",
            F64Kernel::Vpwma => "vpwma",
            F64Kernel::Cfo => "cfo",
            F64Kernel::Var => "var",
            F64Kernel::BollingerBandsWidth => "bollinger_bands_width",
            F64Kernel::DecOsc => "dec_osc",
            F64Kernel::Voss => "voss",
            F64Kernel::PercentileNearestRank => "percentile_nearest_rank",
            F64Kernel::TtmTrend => "ttm_trend",
            F64Kernel::Vi => "vi",
            F64Kernel::Cvi => "cvi",
            F64Kernel::CorrelHl => "correl_hl",
            F64Kernel::Aroonosc => "aroonosc",
            F64Kernel::ParkinsonVolatility => "parkinson_volatility",
            F64Kernel::HistoricalVolatility => "historical_volatility",
            F64Kernel::Donchian => "donchian",
            // --------------------------------------------------- closer 5
            F64Kernel::Velocity => "velocity",
            F64Kernel::VelocityAccelerationIndicator => "velocity_acceleration_indicator",
            F64Kernel::VelocityAccelerationConvergenceDivergenceIndicator => "velocity_acceleration_convergence_divergence_indicator",
            F64Kernel::TrendDirectionForceIndex => "trend_direction_force_index",
            F64Kernel::TrendContinuationFactor => "trend_continuation_factor",
            F64Kernel::Trima => "trima",
            F64Kernel::TrendTriggerFactor => "trend_trigger_factor",
            F64Kernel::VolumeWeightedRsi => "volume_weighted_rsi",
            F64Kernel::VolumeZoneOscillator => "volume_zone_oscillator",
            F64Kernel::Vosc => "vosc",
            F64Kernel::Ultosc => "ultosc",
            // ------------------------------------------------------ closer 4
            F64Kernel::PsychologicalLine => "psychological_line",
            F64Kernel::RankCorrelationIndex => "rank_correlation_index",
            F64Kernel::Qstick => "qstick",
            F64Kernel::Sinwma => "sinwma",
            F64Kernel::Srwma => "srwma",
            F64Kernel::RollingZScoreTrend => "rolling_z_score_trend",
            F64Kernel::RandomWalkIndex => "random_walk_index",
            // --------------------------------------------------------- closer 3
            F64Kernel::L1EhlersPhasor => "l1_ehlers_phasor",
            F64Kernel::L2EhlersSignalToNoise => "l2_ehlers_signal_to_noise",
            F64Kernel::KairiRelativeIndex => "kairi_relative_index",
            F64Kernel::LinearCorrelationOscillator => "linear_correlation_oscillator",
            F64Kernel::MediumAd => "medium_ad",
            F64Kernel::Marketefi => "marketefi",
            F64Kernel::MomentumRatioOscillator => "momentum_ratio_oscillator",
            F64Kernel::OnBalanceVolumeOscillator => "on_balance_volume_oscillator",
            // ------------------------------------------------------ closer 6
            F64Kernel::Emd => "emd",
            F64Kernel::Keltner => "keltner",
            F64Kernel::Stoch => "stoch",
            F64Kernel::NadarayaWatsonEnvelope => "nadaraya_watson_envelope",
                    // ------------------------------------------------------------ closer 2
            F64Kernel::EhlersDetrendingFilter => "ehlers_detrending_filter",
            F64Kernel::EhlersSimpleCycleIndicator => "ehlers_simple_cycle_indicator",
            F64Kernel::EhlersSmoothedAdaptiveMomentum => "ehlers_smoothed_adaptive_momentum",
            F64Kernel::EwmaVolatility => "ewma_volatility",
            F64Kernel::FractalDimensionIndex => "fractal_dimension_index",
            F64Kernel::GopalakrishnanRangeIndex => "gopalakrishnan_range_index",
            F64Kernel::GarmanKlassVolatility => "garman_klass_volatility",
            F64Kernel::ImpulseMacd => "impulse_macd",
            F64Kernel::Hypertrend => "hypertrend",
            F64Kernel::EmdTrend => "emd_trend",
            F64Kernel::Epma => "epma",
            F64Kernel::Fosc => "fosc",
            F64Kernel::EhlersPma => "ehlers_pma",
            F64Kernel::Eri => "eri",
            // ---------------------------------------------------------- closer 2b
            F64Kernel::EhlersFmDemodulator => "ehlers_fm_demodulator",
            F64Kernel::ForwardBackwardExponentialOscillator => "forward_backward_exponential_oscillator",
            F64Kernel::GmmaOscillator => "gmma_oscillator",
            F64Kernel::EvasiveSupertrend => "evasive_supertrend",
            // ------------------------------------------- closer 6, round 2
            F64Kernel::Msw => "msw",
            F64Kernel::YangZhangVolatility => "yang_zhang_volatility",
            F64Kernel::Qqe => "qqe",
            F64Kernel::Srsi => "srsi",
            F64Kernel::Rvi => "rvi",
            F64Kernel::NetMyrsi => "net_myrsi",
            F64Kernel::Vlma => "vlma",
            F64Kernel::Stc => "stc",
            // ------------------------------------------------------ closer 1
            F64Kernel::AbsoluteStrengthIndexOscillator => "absolute_strength_index_oscillator",
            F64Kernel::AccumulationSwingIndex => "accumulation_swing_index",
            F64Kernel::AdaptiveBandpassTriggerOscillator => "adaptive_bandpass_trigger_oscillator",
            F64Kernel::AdaptiveBoundsRsi => "adaptive_bounds_rsi",
            F64Kernel::AdaptiveMacd => "adaptive_macd",
            F64Kernel::AdaptiveMomentumOscillator => "adaptive_momentum_oscillator",
            F64Kernel::AdvanceDeclineLine => "advance_decline_line",
            F64Kernel::AndeanOscillator => "andean_oscillator",
            F64Kernel::AtrPercentile => "atr_percentile",
            F64Kernel::Bop => "bop",
            F64Kernel::BullPowerVsBearPower => "bull_power_vs_bear_power",
            F64Kernel::Cg => "cg",
            F64Kernel::Coppock => "coppock",
            F64Kernel::DailyFactor => "daily_factor",
            F64Kernel::DecisionpointBreadthSwenlinTradingOscillator => {
                "decisionpoint_breadth_swenlin_trading_oscillator"
            }
            F64Kernel::DidiIndex => "didi_index",
            F64Kernel::DisparityIndex => "disparity_index",
            F64Kernel::Dm => "dm",
            F64Kernel::DonchianChannelWidth => "donchian_channel_width",
            F64Kernel::Dpo => "dpo",
            // ------------------------------------------------ closer 2, round 2
            F64Kernel::Mwdx => "mwdx",
            F64Kernel::Lrsi => "lrsi",
            F64Kernel::Pivot => "pivot",
            F64Kernel::Kaufmanstop => "kaufmanstop",
            F64Kernel::Sgf => "sgf",
            F64Kernel::PolynomialRegressionExtrapolation => {
                "polynomial_regression_extrapolation"
            }
            F64Kernel::DualUlcerIndex => "dual_ulcer_index",
            F64Kernel::HullButterflyOscillator => "hull_butterfly_oscillator",
            F64Kernel::RangeOscillator => "range_oscillator",
            F64Kernel::MarketStructureTrailingStop => "market_structure_trailing_stop",
            // ------------------------------------------ closer 3, round 2
            F64Kernel::VerticalHorizontalFilter => "vertical_horizontal_filter",
            F64Kernel::AdjustableMaAlternatingExtremities => {
                "adjustable_ma_alternating_extremities"
            }
            F64Kernel::AutocorrelationIndicator => "autocorrelation_indicator",
            F64Kernel::HistoricalVolatilityRank => "historical_volatility_rank",
            F64Kernel::HistoricalVolatilityPercentile => "historical_volatility_percentile",
            F64Kernel::DirectionalImbalanceIndex => "directional_imbalance_index",
            F64Kernel::CycleChannelOscillator => "cycle_channel_oscillator",
            F64Kernel::DynamicMomentumIndex => "dynamic_momentum_index",
            F64Kernel::EhlersAdaptiveCg => "ehlers_adaptive_cg",
            F64Kernel::EhlersAdaptiveCyberCycle => "ehlers_adaptive_cyber_cycle",
            F64Kernel::EhlersDataSamplingRelativeStrengthIndicator => {
                "ehlers_data_sampling_relative_strength_indicator"
            }
            F64Kernel::ExponentialTrend => "exponential_trend",
            F64Kernel::GeometricBiasOscillator => "geometric_bias_oscillator",
            F64Kernel::IntradayMomentumIndex => "intraday_momentum_index",
            F64Kernel::BullsVBears => "bulls_v_bears",
            F64Kernel::CandleStrengthOscillator => "candle_strength_oscillator",
            F64Kernel::CyberpunkValueTrendAnalyzer => "cyberpunk_value_trend_analyzer",
            F64Kernel::FvgPositioningAverage => "fvg_positioning_average",
            F64Kernel::HemaTrendLevels => "hema_trend_levels",
            F64Kernel::FibonacciTrailingStop => "fibonacci_trailing_stop",
            F64Kernel::GroverLlorensCycleOscillator => "grover_llorens_cycle_oscillator",
            F64Kernel::DemandIndex => "demand_index",
            F64Kernel::AdaptiveSchaffTrendCycle => "adaptive_schaff_trend_cycle",
            F64Kernel::EhlersLinearExtrapolationPredictor => {
                "ehlers_linear_extrapolation_predictor"
            }
            F64Kernel::EhlersAutocorrelationPeriodogram => "ehlers_autocorrelation_periodogram",
            F64Kernel::IctPropulsionBlock => "ict_propulsion_block",

            // ------------------------------------------ closer 4, round 2
            F64Kernel::KasePeakOscillatorWithDivergences => "kase_peak_oscillator_with_divergences",
            F64Kernel::KeltnerChannelWidthOscillator => "keltner_channel_width_oscillator",
            F64Kernel::Kst => "kst",
            F64Kernel::LeavittConvolutionAcceleration => "leavitt_convolution_acceleration",
            F64Kernel::MarketMeannessIndex => "market_meanness_index",
            F64Kernel::MarketStructureConfluence => "market_structure_confluence",
            F64Kernel::MonotonicityIndex => "monotonicity_index",
            F64Kernel::PremierRsiOscillator => "premier_rsi_oscillator",
            F64Kernel::PrettyGoodOscillator => "pretty_good_oscillator",
            F64Kernel::PriceDensityMarketNoise => "price_density_market_noise",
            F64Kernel::ProjectionOscillator => "projection_oscillator",
            F64Kernel::QqeWeightedOscillator => "qqe_weighted_oscillator",
            F64Kernel::RogersSatchellVolatility => "rogers_satchell_volatility",
            F64Kernel::RollingSkewnessKurtosis => "rolling_skewness_kurtosis",
            F64Kernel::SmoothTheilSen => "smooth_theil_sen",
            // ------------------------------------------ closer 2, round 3
            F64Kernel::NeighboringTrailingStop => "neighboring_trailing_stop",
            F64Kernel::NonlinearRegressionZeroLagMovingAverage => {
                "nonlinear_regression_zero_lag_moving_average"
            }
            F64Kernel::NormalizedResonator => "normalized_resonator",
            F64Kernel::NormalizedVolumeTrueRange => "normalized_volume_true_range",
            F64Kernel::PossibleRsi => "possible_rsi",
            F64Kernel::PriceMovingAverageRatioPercentile => {
                "price_moving_average_ratio_percentile"
            }
            F64Kernel::RangeBreakoutSignals => "range_breakout_signals",
            F64Kernel::RangeFilteredTrendSignals => "range_filtered_trend_signals",
            F64Kernel::RegressionSlopeOscillator => "regression_slope_oscillator",
            F64Kernel::RelativeStrengthIndexWaveIndicator => {
                "relative_strength_index_wave_indicator"
            }
            // ------------------------------------------ closer 4, round 3
            F64Kernel::Bandpass => "bandpass",
            F64Kernel::BuffAverages => "buff_averages",
            F64Kernel::CoraWave => "cora_wave",
            F64Kernel::Dma => "dma",
            F64Kernel::FvgTrailingStop => "fvg_trailing_stop",
            F64Kernel::Halftrend => "halftrend",
            F64Kernel::ModGodMode => "mod_god_mode",
            F64Kernel::Ott => "ott",
            F64Kernel::Otto => "otto",
            F64Kernel::Prb => "prb",
            // ---------------------------------------------- closer 6, round 3
            F64Kernel::ElasticVolumeWeightedMovingAverage => {
                "elastic_volume_weighted_moving_average"
            }
            F64Kernel::EmaDeviationCorrectedT3 => "ema_deviation_corrected_t3",
            F64Kernel::LogarithmicMovingAverage => "logarithmic_moving_average",
            F64Kernel::NOrderEma => "n_order_ema",
            F64Kernel::VolatilityAdjustedMa => "volatility_adjusted_ma",
            F64Kernel::WaveSmoother => "wave_smoother",

            // ------------------------------------------------ closer 3, round 3
            F64Kernel::ReversalSignals => "reversal_signals",
            F64Kernel::TrendFollower => "trend_follower",
            F64Kernel::VdubusDivergenceWavePatternGenerator => {
                "vdubus_divergence_wave_pattern_generator"
            }
            F64Kernel::VolatilityRatioAdaptiveRsx => "volatility_ratio_adaptive_rsx",
            F64Kernel::VolumeEnergyReservoirs => "volume_energy_reservoirs",
            F64Kernel::VolumeWeightedRelativeStrengthIndex => {
                "volume_weighted_relative_strength_index"
            }
            F64Kernel::VolumeWeightedStochasticRsi => "volume_weighted_stochastic_rsi",
            F64Kernel::ZigZagChannels => "zig_zag_channels",
            F64Kernel::Alphatrend => "alphatrend",
            F64Kernel::Avsl => "avsl",

            // ------------------------------------------ closer 1, round 3
            F64Kernel::FibonacciEntryBands => "fibonacci_entry_bands",
            F64Kernel::GoertzelCycleCompositeWave => "goertzel_cycle_composite_wave",
            F64Kernel::HalfCausalEstimator => "half_causal_estimator",
            F64Kernel::IchimokuOscillator => "ichimoku_oscillator",
            F64Kernel::InsyncIndex => "insync_index",
            F64Kernel::LinearRegressionIntensity => "linear_regression_intensity",
            F64Kernel::MacdWaveSignalPro => "macd_wave_signal_pro",
            F64Kernel::MesaStochasticMultiLength => "mesa_stochastic_multi_length",
            F64Kernel::MovingAverageCrossProbability => "moving_average_cross_probability",
            F64Kernel::MultiLengthStochasticAverage => "multi_length_stochastic_average",
        }
    }

    /// Every variant, so tests and telemetry cannot silently miss one added
    /// later. Kept next to the match arms above for the same reason.
    pub const ALL: &'static [F64Kernel] = &[
        F64Kernel::Sma,
        F64Kernel::Adosc,
        // ------------------------------------------ closer 5, round 3
        F64Kernel::Rsmk,
        F64Kernel::SqueezeMomentum,
        F64Kernel::Uma,
        F64Kernel::Lpc,
        F64Kernel::Mab,
        F64Kernel::Macz,
        F64Kernel::Vwmacd,
        F64Kernel::CorrectedMovingAverage,
        F64Kernel::EhlersUndersampledDoubleMovingAverage,
        // ----------------------------------------------- closer 5, round 2
        F64Kernel::SmoothedGaussianTrendFilter,
        F64Kernel::SpearmanCorrelation,
        F64Kernel::SqueezeIndex,
        F64Kernel::StandardizedPsarOscillator,
        F64Kernel::StatisticalTrailingStop,
        F64Kernel::StochasticAdaptiveD,
        F64Kernel::StochasticConnorsRsi,
        F64Kernel::StochasticDistance,
        F64Kernel::StochasticMoneyFlowIndex,
        F64Kernel::Supertrend,
        F64Kernel::SupertrendOscillator,
        F64Kernel::SupertrendRecovery,
        F64Kernel::TrendFlowTrail,
        F64Kernel::TwiggsMoneyFlow,
        F64Kernel::VolatilityQualityIndex,
        F64Kernel::VwapDeviationOscillator,
        F64Kernel::VwapZscoreWithSignals,
        F64Kernel::Ema,
        F64Kernel::Rsi,
        F64Kernel::Roc,
        F64Kernel::Mom,
        F64Kernel::Atr,
        F64Kernel::Adx,
        F64Kernel::Willr,
        F64Kernel::Cci,
        F64Kernel::Mfi,
        F64Kernel::Tsi,
        F64Kernel::Obv,
        F64Kernel::Vwap,
        F64Kernel::Wma,
        F64Kernel::Wilders,
        F64Kernel::Smma,
        F64Kernel::Dema,
        F64Kernel::Tema,
        F64Kernel::Zlema,
        F64Kernel::Vwma,
        F64Kernel::Efi,
        F64Kernel::Natr,
        F64Kernel::Adxr,
        F64Kernel::Medprice,
        F64Kernel::Wclprice,
        F64Kernel::Midpoint,
        F64Kernel::Midprice,
        F64Kernel::Rocp,
        F64Kernel::Rocr,
        F64Kernel::Sqwma,
        F64Kernel::Deviation,
        F64Kernel::MeanAd,
        F64Kernel::Ao,
        F64Kernel::LinearregSlope,
        F64Kernel::Tsf,
        F64Kernel::Highpass,
        F64Kernel::Decycler,
        F64Kernel::Supersmoother,
        F64Kernel::Tilson,
        F64Kernel::Wad,
        F64Kernel::Sar,
        F64Kernel::Dti,
        F64Kernel::Zscore,
        F64Kernel::Pfe,
        F64Kernel::Chande,
        F64Kernel::Di,
        F64Kernel::Kdj,
        F64Kernel::Aso,
        F64Kernel::Wto,
        F64Kernel::RangeFilter,
        F64Kernel::CorrelationCycle,
        F64Kernel::Mama,
        F64Kernel::VolumeAdjustedMa,
        F64Kernel::ReverseRsi,
        F64Kernel::EhlersEcema,
        F64Kernel::Devstop,
        F64Kernel::ChandelierExit,
        F64Kernel::Minmax,
        // ------------------------------------------------------------- shard 1 (S1)
        F64Kernel::Apo,
        F64Kernel::Vidya,
        F64Kernel::Gatorosc,
        F64Kernel::Ppo,
        F64Kernel::Pma,
        F64Kernel::Kama,
        F64Kernel::Linreg,
        F64Kernel::Edcf,
        F64Kernel::Alma,
        F64Kernel::Hma,
        F64Kernel::Kurtosis,
        F64Kernel::Alligator,
        F64Kernel::Nvi,
        F64Kernel::Fisher,
        F64Kernel::Safezonestop,
        F64Kernel::Chop,
        F64Kernel::Stochf,
        F64Kernel::Emv,
        F64Kernel::Kvo,
        // ----------------------------------------------------------- shard 4
        F64Kernel::Er,
        F64Kernel::LinearregAngle,
        F64Kernel::LinearregIntercept,
        F64Kernel::Highpass2Pole,
        F64Kernel::Supersmoother3Pole,
        F64Kernel::Cwma,
        F64Kernel::Cmo,
        F64Kernel::Stddev,
        F64Kernel::Ui,
        F64Kernel::BollingerBands,
        F64Kernel::Ehma,
        F64Kernel::Macd,
        F64Kernel::IftRsi,
        F64Kernel::DamianiVolatmeter,
        F64Kernel::Wavetrend,
        F64Kernel::Dx,
        F64Kernel::Frama,
        F64Kernel::Cksp,
        F64Kernel::TtmSqueeze,
        F64Kernel::Mass,
        F64Kernel::Aroon,
        F64Kernel::Acosc,
        F64Kernel::Vpci,
        F64Kernel::Ad,
        F64Kernel::Dvdiqqe,
        F64Kernel::CciCycle,
        F64Kernel::Rsx,
        F64Kernel::Trix,
        F64Kernel::Vpt,
        F64Kernel::Pvi,
        F64Kernel::EhlersItrend,
        F64Kernel::EhlersKama,
        F64Kernel::Sama,
        F64Kernel::Nama,
        F64Kernel::Pwma,
        F64Kernel::Tradjema,
        F64Kernel::Maaq,
        F64Kernel::Jma,
        F64Kernel::Reflex,
        F64Kernel::Gaussian,
        F64Kernel::Fwma,
        F64Kernel::Hwma,
        F64Kernel::Jsa,
        F64Kernel::Nma,
        F64Kernel::Swma,
        F64Kernel::Trendflex,
        F64Kernel::Vpwma,
        F64Kernel::Cfo,
        F64Kernel::Var,
        F64Kernel::BollingerBandsWidth,
        F64Kernel::DecOsc,
        F64Kernel::Voss,
        F64Kernel::PercentileNearestRank,
        F64Kernel::TtmTrend,
        F64Kernel::Vi,
        F64Kernel::Cvi,
        F64Kernel::CorrelHl,
        F64Kernel::Aroonosc,
        F64Kernel::ParkinsonVolatility,
        F64Kernel::HistoricalVolatility,
        F64Kernel::Donchian,
        // ------------------------------------------------------- closer 5
        F64Kernel::Velocity,
        F64Kernel::VelocityAccelerationIndicator,
        F64Kernel::VelocityAccelerationConvergenceDivergenceIndicator,
        F64Kernel::TrendDirectionForceIndex,
        F64Kernel::TrendContinuationFactor,
        F64Kernel::Trima,
        F64Kernel::TrendTriggerFactor,
        F64Kernel::VolumeWeightedRsi,
        F64Kernel::VolumeZoneOscillator,
        F64Kernel::Vosc,
        F64Kernel::Ultosc,
        // ---------------------------------------------------------- closer 4
        F64Kernel::PsychologicalLine,
        F64Kernel::RankCorrelationIndex,
        F64Kernel::Qstick,
        F64Kernel::Sinwma,
        F64Kernel::Srwma,
        F64Kernel::RollingZScoreTrend,
        F64Kernel::RandomWalkIndex,
        // ------------------------------------------------------- closer 3
        F64Kernel::L1EhlersPhasor,
        F64Kernel::L2EhlersSignalToNoise,
        F64Kernel::KairiRelativeIndex,
        F64Kernel::LinearCorrelationOscillator,
        F64Kernel::MediumAd,
        F64Kernel::Marketefi,
        F64Kernel::MomentumRatioOscillator,
        F64Kernel::OnBalanceVolumeOscillator,
        // ------------------------------------------------------------ closer 2
        F64Kernel::EhlersDetrendingFilter,
        F64Kernel::EhlersSimpleCycleIndicator,
        F64Kernel::EhlersSmoothedAdaptiveMomentum,
        F64Kernel::EwmaVolatility,
        F64Kernel::FractalDimensionIndex,
        F64Kernel::GopalakrishnanRangeIndex,
        F64Kernel::GarmanKlassVolatility,
        F64Kernel::ImpulseMacd,
        F64Kernel::Hypertrend,
        F64Kernel::EmdTrend,
        F64Kernel::Epma,
        F64Kernel::Fosc,
        F64Kernel::EhlersPma,
        F64Kernel::Eri,
        // ---------------------------------------------------------- closer 2b
        F64Kernel::EhlersFmDemodulator,
        F64Kernel::ForwardBackwardExponentialOscillator,
        F64Kernel::GmmaOscillator,
        F64Kernel::EvasiveSupertrend,
        // ------------------------------------------------------- closer 6
        // These four have had an entry point, a module and an `F64_KERNELS`
        // row since closer 6, but were never listed here -- so every test and
        // every telemetry pass that walks `ALL` skipped them silently, which
        // is exactly the failure this list exists to prevent.
        F64Kernel::Emd,
        F64Kernel::Keltner,
        F64Kernel::Stoch,
        F64Kernel::NadarayaWatsonEnvelope,
        // ------------------------------------------------------- closer 1
        F64Kernel::AbsoluteStrengthIndexOscillator,
        F64Kernel::AccumulationSwingIndex,
        F64Kernel::AdaptiveBandpassTriggerOscillator,
        F64Kernel::AdaptiveBoundsRsi,
        F64Kernel::AdaptiveMacd,
        F64Kernel::AdaptiveMomentumOscillator,
        F64Kernel::AdvanceDeclineLine,
        F64Kernel::AndeanOscillator,
        F64Kernel::AtrPercentile,
        F64Kernel::Bop,
        F64Kernel::BullPowerVsBearPower,
        F64Kernel::Cg,
        F64Kernel::Coppock,
        F64Kernel::DailyFactor,
        F64Kernel::DecisionpointBreadthSwenlinTradingOscillator,
        F64Kernel::DidiIndex,
        F64Kernel::DisparityIndex,
        F64Kernel::Dm,
        F64Kernel::DonchianChannelWidth,
        F64Kernel::Dpo,
        // ------------------------------------------------- closer 6, round 2
        F64Kernel::Msw,
        F64Kernel::YangZhangVolatility,
        F64Kernel::Qqe,
        F64Kernel::Srsi,
        F64Kernel::Rvi,
        F64Kernel::NetMyrsi,
        F64Kernel::Vlma,
        F64Kernel::Stc,
        // ---------------------------------------------------- closer 2, round 2
        F64Kernel::Mwdx,
        F64Kernel::Lrsi,
        F64Kernel::Pivot,
        F64Kernel::Kaufmanstop,
        F64Kernel::Sgf,
        F64Kernel::PolynomialRegressionExtrapolation,
        F64Kernel::DualUlcerIndex,
        F64Kernel::HullButterflyOscillator,
        F64Kernel::RangeOscillator,
        F64Kernel::MarketStructureTrailingStop,
        // ---------------------------------------------- closer 3, round 2
        F64Kernel::VerticalHorizontalFilter,
        F64Kernel::AdjustableMaAlternatingExtremities,
        F64Kernel::AutocorrelationIndicator,
        F64Kernel::HistoricalVolatilityRank,
        F64Kernel::HistoricalVolatilityPercentile,
        F64Kernel::DirectionalImbalanceIndex,
        F64Kernel::CycleChannelOscillator,
        F64Kernel::DynamicMomentumIndex,
        F64Kernel::EhlersAdaptiveCg,
        F64Kernel::EhlersAdaptiveCyberCycle,
        F64Kernel::EhlersDataSamplingRelativeStrengthIndicator,
        F64Kernel::ExponentialTrend,
        F64Kernel::GeometricBiasOscillator,
        F64Kernel::IntradayMomentumIndex,
        F64Kernel::BullsVBears,
        F64Kernel::CandleStrengthOscillator,
        F64Kernel::CyberpunkValueTrendAnalyzer,
        F64Kernel::FvgPositioningAverage,
        F64Kernel::HemaTrendLevels,
        F64Kernel::FibonacciTrailingStop,
        F64Kernel::GroverLlorensCycleOscillator,
        F64Kernel::DemandIndex,
        F64Kernel::AdaptiveSchaffTrendCycle,
        F64Kernel::EhlersLinearExtrapolationPredictor,
        F64Kernel::EhlersAutocorrelationPeriodogram,
        F64Kernel::IctPropulsionBlock,
        // ---------------------------------------------- closer 4, round 2
        F64Kernel::KasePeakOscillatorWithDivergences,
        F64Kernel::KeltnerChannelWidthOscillator,
        F64Kernel::Kst,
        F64Kernel::LeavittConvolutionAcceleration,
        F64Kernel::MarketMeannessIndex,
        F64Kernel::MarketStructureConfluence,
        F64Kernel::MonotonicityIndex,
        F64Kernel::PremierRsiOscillator,
        F64Kernel::PrettyGoodOscillator,
        F64Kernel::PriceDensityMarketNoise,
        F64Kernel::ProjectionOscillator,
        F64Kernel::QqeWeightedOscillator,
        F64Kernel::RogersSatchellVolatility,
        F64Kernel::RollingSkewnessKurtosis,
        F64Kernel::SmoothTheilSen,
        // ------------------------------------------ closer 4, round 3
        F64Kernel::Bandpass,
        F64Kernel::BuffAverages,
        F64Kernel::CoraWave,
        F64Kernel::Dma,
        F64Kernel::FvgTrailingStop,
        F64Kernel::Halftrend,
        F64Kernel::ModGodMode,
        F64Kernel::Ott,
        F64Kernel::Otto,
        F64Kernel::Prb,
        // --------------------------------------------------- closer 6, round 3
        F64Kernel::ElasticVolumeWeightedMovingAverage,
        F64Kernel::EmaDeviationCorrectedT3,
        F64Kernel::LogarithmicMovingAverage,
        F64Kernel::NOrderEma,
        F64Kernel::VolatilityAdjustedMa,
        F64Kernel::WaveSmoother,
        // --------------------------------------------------- closer 2, round 3
        F64Kernel::NeighboringTrailingStop,
        F64Kernel::NonlinearRegressionZeroLagMovingAverage,
        F64Kernel::NormalizedResonator,
        F64Kernel::NormalizedVolumeTrueRange,
        F64Kernel::PossibleRsi,
        F64Kernel::PriceMovingAverageRatioPercentile,
        F64Kernel::RangeBreakoutSignals,
        F64Kernel::RangeFilteredTrendSignals,
        F64Kernel::RegressionSlopeOscillator,
        F64Kernel::RelativeStrengthIndexWaveIndicator,
        // ---------------------------------------------- closer 3, round 3
        F64Kernel::ReversalSignals,
        F64Kernel::TrendFollower,
        F64Kernel::VdubusDivergenceWavePatternGenerator,
        F64Kernel::VolatilityRatioAdaptiveRsx,
        F64Kernel::VolumeEnergyReservoirs,
        F64Kernel::VolumeWeightedRelativeStrengthIndex,
        F64Kernel::VolumeWeightedStochasticRsi,
        F64Kernel::ZigZagChannels,
        F64Kernel::Alphatrend,
        F64Kernel::Avsl,

        // ---------------------------------------------- closer 1, round 3
        F64Kernel::FibonacciEntryBands,
        F64Kernel::GoertzelCycleCompositeWave,
        F64Kernel::HalfCausalEstimator,
        F64Kernel::IchimokuOscillator,
        F64Kernel::InsyncIndex,
        F64Kernel::LinearRegressionIntensity,
        F64Kernel::MacdWaveSignalPro,
        F64Kernel::MesaStochasticMultiLength,
        F64Kernel::MovingAverageCrossProbability,
        F64Kernel::MultiLengthStochasticAverage,
    ];

    /// `true` when each output bar depends on the one before it, which is why
    /// the kernel is one thread per column. Reported so telemetry can explain
    /// an occupancy number instead of leaving it to be guessed at.
    pub fn is_sequential(self) -> bool {
        matches!(
            self,
            F64Kernel::Sma
                | F64Kernel::Adosc
                // ------------------------------------ closer 5, round 3
                | F64Kernel::Rsmk
                | F64Kernel::SqueezeMomentum
                | F64Kernel::Uma
                | F64Kernel::Lpc
                | F64Kernel::Mab
                | F64Kernel::Macz
                | F64Kernel::Vwmacd
                | F64Kernel::CorrectedMovingAverage
                | F64Kernel::EhlersUndersampledDoubleMovingAverage
                // --------------------------------------- closer 5, round 2
                | F64Kernel::SmoothedGaussianTrendFilter
                | F64Kernel::SpearmanCorrelation
                | F64Kernel::SqueezeIndex
                | F64Kernel::StandardizedPsarOscillator
                | F64Kernel::StatisticalTrailingStop
                | F64Kernel::StochasticAdaptiveD
                | F64Kernel::StochasticConnorsRsi
                | F64Kernel::StochasticDistance
                | F64Kernel::StochasticMoneyFlowIndex
                | F64Kernel::Supertrend
                | F64Kernel::SupertrendOscillator
                | F64Kernel::SupertrendRecovery
                | F64Kernel::TrendFlowTrail
                | F64Kernel::TwiggsMoneyFlow
                | F64Kernel::VolatilityQualityIndex
                | F64Kernel::VwapDeviationOscillator
                | F64Kernel::VwapZscoreWithSignals
                | F64Kernel::Ema
                | F64Kernel::Rsi
                | F64Kernel::Atr
                | F64Kernel::Adx
                | F64Kernel::Cci
                | F64Kernel::Mfi
                | F64Kernel::Tsi
                | F64Kernel::Obv
                | F64Kernel::Vwap
                | F64Kernel::Wma
                | F64Kernel::Wilders
                | F64Kernel::Smma
                | F64Kernel::Dema
                | F64Kernel::Tema
                | F64Kernel::Zlema
                | F64Kernel::Vwma
                | F64Kernel::Efi
                | F64Kernel::Natr
                | F64Kernel::Adxr
                | F64Kernel::Sqwma
                | F64Kernel::Deviation
                | F64Kernel::MeanAd
                | F64Kernel::Ao
                | F64Kernel::LinearregSlope
                | F64Kernel::Tsf
                | F64Kernel::Highpass
                | F64Kernel::Decycler
                | F64Kernel::Supersmoother
                | F64Kernel::Tilson
                | F64Kernel::Wad
                | F64Kernel::Sar
                | F64Kernel::Dti
                | F64Kernel::Zscore
                | F64Kernel::Pfe
                | F64Kernel::Chande
                | F64Kernel::Di
                | F64Kernel::Kdj
                | F64Kernel::Aso
                | F64Kernel::Wto
                | F64Kernel::RangeFilter
                | F64Kernel::CorrelationCycle
                | F64Kernel::Mama
                | F64Kernel::VolumeAdjustedMa
                | F64Kernel::ReverseRsi
                | F64Kernel::EhlersEcema
                | F64Kernel::Devstop
                | F64Kernel::ChandelierExit
                | F64Kernel::Minmax
                // ------------------------------------------------------------- shard 1 (S1)
                | F64Kernel::Apo
                | F64Kernel::Vidya
                | F64Kernel::Gatorosc
                | F64Kernel::Ppo
                | F64Kernel::Pma
                | F64Kernel::Kama
                | F64Kernel::Linreg
                | F64Kernel::Edcf
                | F64Kernel::Alma
                | F64Kernel::Hma
                | F64Kernel::Kurtosis
                | F64Kernel::Alligator
                | F64Kernel::Nvi
                | F64Kernel::Fisher
                | F64Kernel::Safezonestop
                | F64Kernel::Chop
                | F64Kernel::Stochf
                | F64Kernel::Emv
                | F64Kernel::Kvo
                | F64Kernel::Rsx
                | F64Kernel::Trix
                | F64Kernel::Vpt
                | F64Kernel::Pvi
                | F64Kernel::EhlersItrend
                | F64Kernel::EhlersKama
                | F64Kernel::Sama
                | F64Kernel::Nama
                | F64Kernel::Pwma
                | F64Kernel::Tradjema
                | F64Kernel::Maaq
                | F64Kernel::Jma
                | F64Kernel::Reflex
                | F64Kernel::Gaussian
                | F64Kernel::Fwma
                | F64Kernel::Hwma
                | F64Kernel::Jsa
                | F64Kernel::Nma
                | F64Kernel::Swma
                | F64Kernel::Trendflex
                | F64Kernel::Vpwma
                | F64Kernel::Cfo
                | F64Kernel::Var
                | F64Kernel::BollingerBandsWidth
                | F64Kernel::DecOsc
                | F64Kernel::Voss
                | F64Kernel::TtmTrend
                | F64Kernel::Vi
                | F64Kernel::Cvi
                | F64Kernel::CorrelHl
                | F64Kernel::Aroonosc
                | F64Kernel::ParkinsonVolatility
                | F64Kernel::HistoricalVolatility
                // ---------------------------------------------------- closer 4
                | F64Kernel::PsychologicalLine
                | F64Kernel::RankCorrelationIndex
                | F64Kernel::Qstick
                | F64Kernel::Sinwma
                | F64Kernel::Srwma
                | F64Kernel::RollingZScoreTrend
                | F64Kernel::RandomWalkIndex
                // --------------------------------------------------------- shard 4 (S4)
                | F64Kernel::Er
                | F64Kernel::LinearregAngle
                | F64Kernel::LinearregIntercept
                | F64Kernel::Highpass2Pole
                | F64Kernel::Supersmoother3Pole
                | F64Kernel::Cwma
                | F64Kernel::Cmo
                | F64Kernel::Stddev
                | F64Kernel::Ui
                | F64Kernel::BollingerBands
                | F64Kernel::Ehma
                | F64Kernel::Macd
                | F64Kernel::IftRsi
                | F64Kernel::DamianiVolatmeter
                | F64Kernel::Wavetrend
                | F64Kernel::Dx
                | F64Kernel::Frama
                | F64Kernel::Cksp
                | F64Kernel::TtmSqueeze
                | F64Kernel::Mass
                | F64Kernel::Aroon
                | F64Kernel::Acosc
                | F64Kernel::Vpci
                | F64Kernel::Ad
                | F64Kernel::Dvdiqqe
                | F64Kernel::CciCycle
                // ---------------------------------------------------- closer 6
                // `emd` carries a 2-pole bandpass IIR plus three sliding sums
                // over rings, `keltner` a Wilder ATR recurrence plus an
                // SMA-seeded EMA, and `stoch` a monotone-extreme tracker whose
                // rescan decision depends on the PREVIOUS bar's argmax. None
                // is bar-parallel without changing the rounding or the tie
                // handling.
                | F64Kernel::Emd
                | F64Kernel::Keltner
                | F64Kernel::Stoch
                | F64Kernel::NadarayaWatsonEnvelope
                // ---------------------------------------------------- closer 3
                // Five are forced sequential by a carried state: a phasor
                // rotation, five interlocking Ehlers IIRs, a rolling sum whose
                // accumulation order is load-bearing, and a windowed order
                // statistic. `Marketefi` is POINTWISE and would be
                // bar-parallel, but the bar-parallel launch arm refuses the
                // (high, low, volume) shape, so it is sequential here.
                | F64Kernel::L1EhlersPhasor
                | F64Kernel::L2EhlersSignalToNoise
                | F64Kernel::KairiRelativeIndex
                | F64Kernel::LinearCorrelationOscillator
                | F64Kernel::MediumAd
                | F64Kernel::Marketefi
                | F64Kernel::MomentumRatioOscillator
                | F64Kernel::OnBalanceVolumeOscillator
                // --------------------------------------------- closer 5
                | F64Kernel::Velocity
                | F64Kernel::VelocityAccelerationIndicator
                | F64Kernel::VelocityAccelerationConvergenceDivergenceIndicator
                | F64Kernel::TrendDirectionForceIndex
                | F64Kernel::TrendContinuationFactor
                | F64Kernel::Trima
                | F64Kernel::TrendTriggerFactor
                | F64Kernel::VolumeWeightedRsi
                | F64Kernel::VolumeZoneOscillator
                | F64Kernel::Vosc
                | F64Kernel::Ultosc
        
                // ------------------------------------------------------------ closer 2
                | F64Kernel::EhlersDetrendingFilter
                | F64Kernel::EhlersSimpleCycleIndicator
                | F64Kernel::EhlersSmoothedAdaptiveMomentum
                | F64Kernel::EwmaVolatility
                | F64Kernel::FractalDimensionIndex
                | F64Kernel::GopalakrishnanRangeIndex
                | F64Kernel::GarmanKlassVolatility
                | F64Kernel::ImpulseMacd
                | F64Kernel::Hypertrend
                | F64Kernel::EmdTrend
                | F64Kernel::Epma
                | F64Kernel::Fosc
                | F64Kernel::EhlersPma
                | F64Kernel::Eri
                // -------------------------------------------- closer 1
                | F64Kernel::AbsoluteStrengthIndexOscillator
                | F64Kernel::AccumulationSwingIndex
                | F64Kernel::AdaptiveBandpassTriggerOscillator
                | F64Kernel::AdaptiveBoundsRsi
                | F64Kernel::AdaptiveMacd
                | F64Kernel::AdaptiveMomentumOscillator
                | F64Kernel::AdvanceDeclineLine
                | F64Kernel::AndeanOscillator
                | F64Kernel::AtrPercentile
                | F64Kernel::Bop
                | F64Kernel::BullPowerVsBearPower
                | F64Kernel::Cg
                | F64Kernel::Coppock
                | F64Kernel::DailyFactor
                | F64Kernel::DecisionpointBreadthSwenlinTradingOscillator
                | F64Kernel::DidiIndex
                | F64Kernel::DisparityIndex
                | F64Kernel::Dm
                | F64Kernel::DonchianChannelWidth
                | F64Kernel::Dpo
        
                // ---------------------------------------------------------- closer 2b
                | F64Kernel::EhlersFmDemodulator
                | F64Kernel::ForwardBackwardExponentialOscillator
                | F64Kernel::GmmaOscillator
                | F64Kernel::EvasiveSupertrend
                // --------------------------------------- closer 6, round 2
                // All eight carry state across bars: a Wilder or EMA
                // recurrence (qqe, srsi, stc, rvi), a rolling (sum, sumsq)
                // pair (yang_zhang_volatility, vlma), an adaptive-period
                // state machine (vlma), or a rank count rolled rather than
                // recomputed (net_myrsi). `Msw` is the one whose per-bar value
                // carries NO state and could be bar-parallel; it is launched
                // sequential because that is the shape this lane launches, and
                // the bar loop is the thread body.
                | F64Kernel::Msw
                | F64Kernel::YangZhangVolatility
                | F64Kernel::Qqe
                | F64Kernel::Srsi
                | F64Kernel::Rvi
                | F64Kernel::NetMyrsi
                | F64Kernel::Vlma
                | F64Kernel::Stc
                // --------------------------------------- closer 2, round 2
                // All ten. Six carry genuine state across bars; the other four
                // rebuild their window every bar and are launched sequential
                // because that is the shape this lane launches -- the bar loop
                // is the thread body. See the enum block for the per-variant
                // reason.
                | F64Kernel::Mwdx
                | F64Kernel::Lrsi
                | F64Kernel::Pivot
                | F64Kernel::Kaufmanstop
                | F64Kernel::Sgf
                | F64Kernel::PolynomialRegressionExtrapolation
                | F64Kernel::DualUlcerIndex
                | F64Kernel::HullButterflyOscillator
                | F64Kernel::RangeOscillator
                | F64Kernel::MarketStructureTrailingStop
                // ------------------------------------ closer 3, round 2
                //
                // All twenty-five. Every one carries state across bars: a
                // Wilder or EMA recurrence, an Ehlers IIR, a monotone deque, a
                // sliding sum maintained with subtract-then-add, a ratchet, or
                // a state machine whose reset points depend on where the holes
                // in the series are. Even the three whose per-bar value would
                // be window-parallel -- vertical_horizontal_filter,
                // adjustable_ma_alternating_extremities and
                // autocorrelation_indicator's convolution -- are launched one
                // thread per combo, because that is the shape this lane
                // launches and the bar loop is the thread body.
                | F64Kernel::VerticalHorizontalFilter
                | F64Kernel::AdjustableMaAlternatingExtremities
                | F64Kernel::AutocorrelationIndicator
                | F64Kernel::HistoricalVolatilityRank
                | F64Kernel::HistoricalVolatilityPercentile
                | F64Kernel::DirectionalImbalanceIndex
                | F64Kernel::CycleChannelOscillator
                | F64Kernel::DynamicMomentumIndex
                | F64Kernel::EhlersAdaptiveCg
                | F64Kernel::EhlersAdaptiveCyberCycle
                | F64Kernel::EhlersDataSamplingRelativeStrengthIndicator
                | F64Kernel::ExponentialTrend
                | F64Kernel::GeometricBiasOscillator
                | F64Kernel::IntradayMomentumIndex
                | F64Kernel::BullsVBears
                | F64Kernel::CandleStrengthOscillator
                | F64Kernel::CyberpunkValueTrendAnalyzer
                | F64Kernel::FvgPositioningAverage
                | F64Kernel::HemaTrendLevels
                | F64Kernel::FibonacciTrailingStop
                | F64Kernel::GroverLlorensCycleOscillator
                | F64Kernel::DemandIndex
                | F64Kernel::AdaptiveSchaffTrendCycle
                | F64Kernel::EhlersLinearExtrapolationPredictor
                | F64Kernel::EhlersAutocorrelationPeriodogram
                | F64Kernel::IctPropulsionBlock
                // ------------------------------------ closer 4, round 2
                // All fifteen carry state across bars -- see the enum
                // note for which recurrence each one is.
                | F64Kernel::KasePeakOscillatorWithDivergences
                | F64Kernel::KeltnerChannelWidthOscillator
                | F64Kernel::Kst
                | F64Kernel::LeavittConvolutionAcceleration
                | F64Kernel::MarketMeannessIndex
                | F64Kernel::MarketStructureConfluence
                | F64Kernel::MonotonicityIndex
                | F64Kernel::PremierRsiOscillator
                | F64Kernel::PrettyGoodOscillator
                | F64Kernel::PriceDensityMarketNoise
                | F64Kernel::ProjectionOscillator
                | F64Kernel::QqeWeightedOscillator
                | F64Kernel::RogersSatchellVolatility
                | F64Kernel::RollingSkewnessKurtosis
                | F64Kernel::SmoothTheilSen
                // ------------------------------------------ closer 4, round 3
                //
                // All ten. Every one carries state across bars -- a 2-pole
                // IIR, a variable-alpha EMA plus a band ratchet, a weighted
                // sliding sum rolled rather than rebuilt, a Wilder ATR with a
                // trend state machine, a gap ledger with a trailing stop, or
                // five interlocking recurrences. See the enum note.
                | F64Kernel::Bandpass
                | F64Kernel::BuffAverages
                | F64Kernel::CoraWave
                | F64Kernel::Dma
                | F64Kernel::FvgTrailingStop
                | F64Kernel::Halftrend
                | F64Kernel::ModGodMode
                | F64Kernel::Ott
                | F64Kernel::Otto
                | F64Kernel::Prb
                // ------------------------------------------ closer 6, round 3
                //
                // All six. Every one of them carries at least one scalar from
                // bar to bar -- an IIR history, a T3 cascade, an EMA plus two
                // monotonic deques plus a rolling WMA, a rolling volume sum, a
                // finite-run counter, or a 2-bar pre-smoother -- so each is one
                // thread per column walking bars ascending.
                | F64Kernel::ElasticVolumeWeightedMovingAverage
                | F64Kernel::EmaDeviationCorrectedT3
                | F64Kernel::LogarithmicMovingAverage
                | F64Kernel::NOrderEma
                | F64Kernel::VolatilityAdjustedMa
                | F64Kernel::WaveSmoother
                // ------------------------------------------ closer 2, round 3
                //
                // ALL TEN. Every one carries state from bar to bar and none can
                // be made bar-parallel without changing the rounding:
                // a 200-deep price window plus a stop ratchet
                // (neighboring_trailing_stop); two cascaded rolling WMAs into a
                // quadratic-regression moment triple updated from itself
                // (nonlinear_regression_zero_lag_moving_average); a 2-pole
                // resonator IIR plus an EMA plus a monotonic peak deque
                // (normalized_resonator); an all-bars running mean plus two
                // never-evicted variance sums plus two filled smoothing rings
                // (normalized_volume_true_range); a Wilder RSI feeding a
                // rolling min/max feeding a fisher recursion feeding a 74-deep
                // nonlag ring (possible_rsi); a running SMA window sum plus an
                // incremental sorted percentile window
                // (price_moving_average_ratio_percentile); a breakout state
                // machine over a median window, a Wilder ATR and two
                // confirmation windows (range_breakout_signals); a Kalman
                // covariance updated from itself plus a Wilder ATR plus a
                // 200-deep WMA (range_filtered_trend_signals); two running
                // prefix sums of logarithms (regression_slope_oscillator); and
                // three Wilder recursions into a rolling WMA
                // (relative_strength_index_wave_indicator).
                | F64Kernel::NeighboringTrailingStop
                | F64Kernel::NonlinearRegressionZeroLagMovingAverage
                | F64Kernel::NormalizedResonator
                | F64Kernel::NormalizedVolumeTrueRange
                | F64Kernel::PossibleRsi
                | F64Kernel::PriceMovingAverageRatioPercentile
                | F64Kernel::RangeBreakoutSignals
                | F64Kernel::RangeFilteredTrendSignals
                | F64Kernel::RegressionSlopeOscillator
                | F64Kernel::RelativeStrengthIndexWaveIndicator
                // ------------------------------- closer 3, round 3
                // ALL TEN. Every one carries a scalar, a ratchet, a monotone
                // deque or a pivot ring across bars; none is bar-parallel.
                | F64Kernel::ReversalSignals
                | F64Kernel::TrendFollower
                | F64Kernel::VdubusDivergenceWavePatternGenerator
                | F64Kernel::VolatilityRatioAdaptiveRsx
                | F64Kernel::VolumeEnergyReservoirs
                | F64Kernel::VolumeWeightedRelativeStrengthIndex
                | F64Kernel::VolumeWeightedStochasticRsi
                | F64Kernel::ZigZagChannels
                | F64Kernel::Alphatrend
                | F64Kernel::Avsl

                // ------------------------------------ closer 1, round 3
                // Nine of the ten. Each carries state across bars: a double
                // EMA cascade, a Chebyshev/Gaussian smoothing chain, a
                // time-of-day store, a sliding Kendall counter, ten reset-
                // together sub-indicators, two IIR poles, or a truncated-EMA
                // pair with a drop-scale correction.
                // `GoertzelCycleCompositeWave` is DELIBERATELY ABSENT: its
                // CPU `compute_row` recomputes every window from scratch and
                // carries nothing, so it is bar-parallel.
                | F64Kernel::FibonacciEntryBands
                | F64Kernel::HalfCausalEstimator
                | F64Kernel::IchimokuOscillator
                | F64Kernel::InsyncIndex
                | F64Kernel::LinearRegressionIntensity
                | F64Kernel::MacdWaveSignalPro
                | F64Kernel::MesaStochasticMultiLength
                | F64Kernel::MovingAverageCrossProbability
                | F64Kernel::MultiLengthStochasticAverage
        )
    }

    /// Which compiled CUDA module carries this variant's entry point.
    ///
    /// # Why this is not a constant
    ///
    /// The lane started as one `.cu` file holding every f64 kernel, because
    /// one module meant one load per frame instead of the ~50 this crate's
    /// other 309 `load_cuda_embedded_module!` sites pay. That is still the
    /// right default and every variant above shard 2's marker still uses it.
    ///
    /// It stopped being the right RULE once the job became "fix the 712 f32
    /// kernels this crate already ships", because those kernels live in their
    /// own files next to the f32 entry points that 180 wrappers still call.
    /// Copying each converted kernel into `neoethos_f64_kernels.cu` would leave
    /// two implementations of one indicator — which is precisely the failure
    /// this lane exists to remove, just relocated.
    ///
    /// So a variant names its module. [`CudaF64Indicators::new`] loads the
    /// distinct set once at construction, so a frame still pays zero loads.
    pub fn module_stem(self) -> &'static str {
        match self {
            F64Kernel::Sqwma => "sqwma_kernel",
            F64Kernel::Adosc => "adosc_kernel",
            // ------------------------------------------ closer 5, round 3
            F64Kernel::Rsmk => "rsmk_kernel",
            F64Kernel::SqueezeMomentum => "squeeze_momentum_kernel",
            F64Kernel::Uma => "uma_kernel",
            F64Kernel::Lpc => "lpc_kernel",
            F64Kernel::Mab => "mab_kernel",
            F64Kernel::Macz => "macz_kernel",
            F64Kernel::Vwmacd => "vwmacd_kernel",
            F64Kernel::CorrectedMovingAverage => "corrected_moving_average_kernel",
            F64Kernel::EhlersUndersampledDoubleMovingAverage => "ehlers_undersampled_double_moving_average_kernel",
            // ------------------------------ closer 5, round 2 (modules)
            F64Kernel::SmoothedGaussianTrendFilter => "smoothed_gaussian_trend_filter_kernel",
            F64Kernel::SpearmanCorrelation => "spearman_correlation_kernel",
            F64Kernel::SqueezeIndex => "squeeze_index_kernel",
            F64Kernel::StandardizedPsarOscillator => "standardized_psar_oscillator_kernel",
            F64Kernel::StatisticalTrailingStop => "statistical_trailing_stop_kernel",
            F64Kernel::StochasticAdaptiveD => "stochastic_adaptive_d_kernel",
            F64Kernel::StochasticConnorsRsi => "stochastic_connors_rsi_kernel",
            F64Kernel::StochasticDistance => "stochastic_distance_kernel",
            F64Kernel::StochasticMoneyFlowIndex => "stochastic_money_flow_index_kernel",
            F64Kernel::Supertrend => "supertrend_kernel",
            F64Kernel::SupertrendOscillator => "supertrend_oscillator_kernel",
            F64Kernel::SupertrendRecovery => "supertrend_recovery_kernel",
            F64Kernel::TrendFlowTrail => "trend_flow_trail_kernel",
            F64Kernel::TwiggsMoneyFlow => "twiggs_money_flow_kernel",
            F64Kernel::VolatilityQualityIndex => "volatility_quality_index_kernel",
            F64Kernel::VwapDeviationOscillator => "vwap_deviation_oscillator_kernel",
            F64Kernel::VwapZscoreWithSignals => "vwap_zscore_with_signals_kernel",
            F64Kernel::Deviation => "deviation_kernel",
            F64Kernel::MeanAd => "mean_ad_kernel",
            F64Kernel::Ao => "ao_kernel",
            F64Kernel::LinearregSlope => "linearreg_slope_kernel",
            F64Kernel::Tsf => "tsf_kernel",
            F64Kernel::Highpass => "highpass_kernel",
            F64Kernel::Decycler => "decycler_kernel",
            F64Kernel::Supersmoother => "supersmoother_kernel",
            F64Kernel::Tilson => "tilson_kernel",
            F64Kernel::Wad => "wad_kernel",
            F64Kernel::Sar => "sar_kernel",
            F64Kernel::Dti => "dti_kernel",
            F64Kernel::Zscore => "zscore_kernel",
            F64Kernel::Pfe => "pfe_kernel",
            F64Kernel::Chande => "chande_kernel",
            F64Kernel::Di => "di_kernel",
            F64Kernel::Kdj => "kdj_kernel",
            F64Kernel::Aso => "aso_kernel",
            F64Kernel::Wto => "wto_kernel",
            F64Kernel::RangeFilter => "range_filter_kernel",
            F64Kernel::CorrelationCycle => "correlation_cycle_kernel",
            F64Kernel::Mama => "mama_kernel",
            F64Kernel::VolumeAdjustedMa => "volume_adjusted_ma_kernel",
            F64Kernel::ReverseRsi => "reverse_rsi_kernel",
            F64Kernel::EhlersEcema => "ehlers_ecema_kernel",
            F64Kernel::Devstop => "devstop_kernel",
            F64Kernel::ChandelierExit => "chandelier_exit_kernel",
            F64Kernel::Minmax => "minmax_kernel",
            // ------------------------------------------------------------- shard 1 (S1)
            F64Kernel::Apo => "apo_kernel",
            F64Kernel::Vidya => "vidya_kernel",
            F64Kernel::Gatorosc => "gatorosc_kernel",
            F64Kernel::Ppo => "ppo_kernel",
            F64Kernel::Pma => "pma_kernel",
            F64Kernel::Kama => "kama_kernel",
            F64Kernel::Linreg => "linreg_kernel",
            F64Kernel::Edcf => "edcf_kernel",
            F64Kernel::Alma => "alma_kernel",
            F64Kernel::Hma => "hma_kernel",
            F64Kernel::Kurtosis => "kurtosis_kernel",
            F64Kernel::Alligator => "alligator_kernel",
            F64Kernel::Nvi => "nvi_kernel",
            F64Kernel::Fisher => "fisher_kernel",
            F64Kernel::Safezonestop => "safezonestop_kernel",
            F64Kernel::Chop => "chop_kernel",
            F64Kernel::Stochf => "stochf_kernel",
            F64Kernel::Emv => "emv_kernel",
            F64Kernel::Kvo => "kvo_kernel",
            // ------------------------------------------------------- shard 4
            F64Kernel::Er => "er_kernel",
            F64Kernel::LinearregAngle => "linearreg_angle_kernel",
            F64Kernel::LinearregIntercept => "linearreg_intercept_kernel",
            F64Kernel::Highpass2Pole => "highpass2_kernel",
            F64Kernel::Supersmoother3Pole => "supersmoother_3_pole_kernel",
            F64Kernel::Cwma => "cwma_kernel",
            F64Kernel::Cmo => "cmo_kernel",
            F64Kernel::Stddev => "stddev_kernel",
            F64Kernel::Ui => "ui_kernel",
            F64Kernel::BollingerBands => "bollinger_bands_kernel",
            F64Kernel::Ehma => "ehma_kernel",
            F64Kernel::Macd => "macd_kernel",
            F64Kernel::IftRsi => "ift_rsi_kernel",
            F64Kernel::DamianiVolatmeter => "damiani_volatmeter_kernel",
            F64Kernel::Wavetrend => "wavetrend_kernel",
            F64Kernel::Dx => "dx_kernel",
            F64Kernel::Frama => "frama_kernel",
            F64Kernel::Cksp => "cksp_kernel",
            F64Kernel::TtmSqueeze => "ttm_squeeze_kernel",
            F64Kernel::Mass => "mass_kernel",
            F64Kernel::Aroon => "aroon_kernel",
            F64Kernel::Acosc => "acosc_kernel",
            F64Kernel::Vpci => "vpci_kernel",
            F64Kernel::Ad => "ad_kernel",
            F64Kernel::Dvdiqqe => "dvdiqqe_kernel",
            F64Kernel::CciCycle => "cci_cycle_kernel",
            F64Kernel::Rsx => "rsx_kernel",
            F64Kernel::Trix => "trix_kernel",
            F64Kernel::Vpt => "vpt_kernel",
            F64Kernel::Pvi => "pvi_kernel",
            F64Kernel::EhlersItrend => "ehlers_itrend_kernel",
            F64Kernel::EhlersKama => "ehlers_kama_kernel",
            F64Kernel::Sama => "sama_kernel",
            F64Kernel::Nama => "nama_kernel",
            F64Kernel::Pwma => "pwma_kernel",
            F64Kernel::Tradjema => "tradjema_kernel",
            F64Kernel::Maaq => "maaq_kernel",
            F64Kernel::Jma => "jma_kernel",
            F64Kernel::Reflex => "reflex_kernel",
            F64Kernel::Gaussian => "gaussian_kernel",
            F64Kernel::Fwma => "fwma_kernel",
            F64Kernel::Hwma => "hwma_kernel",
            F64Kernel::Jsa => "jsa_kernel",
            F64Kernel::Nma => "nma_kernel",
            F64Kernel::Swma => "swma_kernel",
            F64Kernel::Trendflex => "trendflex_kernel",
            F64Kernel::Vpwma => "vpwma_kernel",
            F64Kernel::Cfo => "cfo_kernel",
            F64Kernel::Var => "var_kernel",
            F64Kernel::BollingerBandsWidth => "bollinger_bands_width_kernel",
            F64Kernel::DecOsc => "dec_osc_kernel",
            F64Kernel::Voss => "voss_kernel",
            F64Kernel::PercentileNearestRank => "percentile_nearest_rank_kernel",
            F64Kernel::TtmTrend => "ttm_trend_kernel",
            F64Kernel::Vi => "vi_kernel",
            F64Kernel::Cvi => "cvi_kernel",
            F64Kernel::CorrelHl => "correl_hl_kernel",
            F64Kernel::Aroonosc => "aroonosc_kernel",
            F64Kernel::ParkinsonVolatility => "parkinson_volatility_kernel",
            F64Kernel::HistoricalVolatility => "historical_volatility_kernel",
            F64Kernel::Donchian => "donchian_kernel",
            // --------------------------------------------------- closer 5
            F64Kernel::Velocity => "velocity_kernel",
            F64Kernel::VelocityAccelerationIndicator => "velocity_acceleration_indicator_kernel",
            F64Kernel::VelocityAccelerationConvergenceDivergenceIndicator => "velocity_acceleration_convergence_divergence_indicator_kernel",
            F64Kernel::TrendDirectionForceIndex => "trend_direction_force_index_kernel",
            F64Kernel::TrendContinuationFactor => "trend_continuation_factor_kernel",
            F64Kernel::Trima => "trima_kernel",
            F64Kernel::TrendTriggerFactor => "trend_trigger_factor_kernel",
            F64Kernel::VolumeWeightedRsi => "volume_weighted_rsi_kernel",
            F64Kernel::VolumeZoneOscillator => "volume_zone_oscillator_kernel",
            F64Kernel::Vosc => "vosc_kernel",
            F64Kernel::Ultosc => "ultosc_kernel",
            // ------------------------------------------------------ closer 4
            F64Kernel::PsychologicalLine => "psychological_line_kernel",
            F64Kernel::RankCorrelationIndex => "rank_correlation_index_kernel",
            F64Kernel::Qstick => "qstick_kernel",
            F64Kernel::Sinwma => "sinwma_kernel",
            F64Kernel::Srwma => "srwma_kernel",
            F64Kernel::RollingZScoreTrend => "rolling_z_score_trend_kernel",
            F64Kernel::RandomWalkIndex => "random_walk_index_kernel",
            // --------------------------------------------------------- closer 3
            F64Kernel::L1EhlersPhasor => "l1_ehlers_phasor_kernel",
            F64Kernel::L2EhlersSignalToNoise => "l2_ehlers_signal_to_noise_kernel",
            F64Kernel::KairiRelativeIndex => "kairi_relative_index_kernel",
            F64Kernel::LinearCorrelationOscillator => "linear_correlation_oscillator_kernel",
            F64Kernel::MediumAd => "medium_ad_kernel",
            F64Kernel::Marketefi => "marketefi_kernel",
            F64Kernel::MomentumRatioOscillator => "momentum_ratio_oscillator_kernel",
            F64Kernel::OnBalanceVolumeOscillator => {
                "on_balance_volume_oscillator_kernel"
            }
            // ------------------------------------------------------ closer 6
            F64Kernel::Emd => "emd_kernel",
            F64Kernel::Keltner => "keltner_kernel",
            F64Kernel::Stoch => "stoch_kernel",
            F64Kernel::NadarayaWatsonEnvelope => "nadaraya_watson_envelope_kernel",
            // ------------------------------------------------------------ closer 2
            F64Kernel::EhlersDetrendingFilter => "ehlers_detrending_filter_kernel",
            F64Kernel::EhlersSimpleCycleIndicator => "ehlers_simple_cycle_indicator_kernel",
            F64Kernel::EhlersSmoothedAdaptiveMomentum => "ehlers_smoothed_adaptive_momentum_kernel",
            F64Kernel::EwmaVolatility => "ewma_volatility_kernel",
            F64Kernel::FractalDimensionIndex => "fractal_dimension_index_kernel",
            F64Kernel::GopalakrishnanRangeIndex => "gopalakrishnan_range_index_kernel",
            F64Kernel::GarmanKlassVolatility => "garman_klass_volatility_kernel",
            F64Kernel::ImpulseMacd => "impulse_macd_kernel",
            F64Kernel::Hypertrend => "hypertrend_kernel",
            F64Kernel::EmdTrend => "emd_trend_kernel",
            F64Kernel::Epma => "epma_kernel",
            F64Kernel::Fosc => "fosc_kernel",
            F64Kernel::EhlersPma => "ehlers_pma_kernel",
            F64Kernel::Eri => "eri_kernel",
            // ---------------------------------------------------------- closer 2b
            F64Kernel::EhlersFmDemodulator => "ehlers_fm_demodulator_kernel",
            F64Kernel::ForwardBackwardExponentialOscillator => "forward_backward_exponential_oscillator_kernel",
            F64Kernel::GmmaOscillator => "gmma_oscillator_kernel",
            F64Kernel::EvasiveSupertrend => "evasive_supertrend_kernel",
            // ------------------------------------------- closer 6, round 2
            //
            // Each of the eight is compiled into its OWN module, because its
            // entry point was written into the `.cu` file its indicator
            // already ships in. Six live under a subdirectory
            // (`oscillators/`, `moving_averages/`), but `compile_kernel` names
            // the PTX after the file STEM, so the module is `msw_kernel`, not
            // `oscillators_msw_kernel`.
            //
            // These arms MUST stay above the `_ => NEOETHOS_F64_MODULE`
            // catch-all below, for the reason spelled out under closer 1.
            F64Kernel::Msw => "msw_kernel",
            F64Kernel::YangZhangVolatility => "yang_zhang_volatility_kernel",
            F64Kernel::Qqe => "qqe_kernel",
            F64Kernel::Srsi => "srsi_kernel",
            F64Kernel::Rvi => "rvi_kernel",
            F64Kernel::NetMyrsi => "net_myrsi_kernel",
            F64Kernel::Vlma => "vlma_kernel",
            F64Kernel::Stc => "stc_kernel",
            // ------------------------------------------------------ closer 1
            //
            // Every one of the twenty is compiled into its OWN module, because
            // its entry point was written into the `.cu` file its indicator
            // already ships in. Three of them live under a subdirectory --
            // `oscillators/bop_kernel.cu`, `oscillators/cg_kernel.cu`,
            // `oscillators/coppock_kernel.cu`, `oscillators/dpo_kernel.cu` --
            // but `compile_kernel` names the PTX after the file STEM, so the
            // module is still `bop_kernel` and not `oscillators_bop_kernel`.
            //
            // These arms MUST stay above the `_ => NEOETHOS_F64_MODULE`
            // catch-all below. Behind it they are dead, `module_stem` answers
            // `neoethos_f64_kernels`, and `get_function` then hunts for
            // `bop_neo_batch_f64` in a module that never contained it.
            F64Kernel::AbsoluteStrengthIndexOscillator => "absolute_strength_index_oscillator_kernel",
            F64Kernel::AccumulationSwingIndex => "accumulation_swing_index_kernel",
            F64Kernel::AdaptiveBandpassTriggerOscillator => "adaptive_bandpass_trigger_oscillator_kernel",
            F64Kernel::AdaptiveBoundsRsi => "adaptive_bounds_rsi_kernel",
            F64Kernel::AdaptiveMacd => "adaptive_macd_kernel",
            F64Kernel::AdaptiveMomentumOscillator => "adaptive_momentum_oscillator_kernel",
            F64Kernel::AdvanceDeclineLine => "advance_decline_line_kernel",
            F64Kernel::AndeanOscillator => "andean_oscillator_kernel",
            F64Kernel::AtrPercentile => "atr_percentile_kernel",
            F64Kernel::Bop => "bop_kernel",
            F64Kernel::BullPowerVsBearPower => "bull_power_vs_bear_power_kernel",
            F64Kernel::Cg => "cg_kernel",
            F64Kernel::Coppock => "coppock_kernel",
            F64Kernel::DailyFactor => "daily_factor_kernel",
            F64Kernel::DecisionpointBreadthSwenlinTradingOscillator => "decisionpoint_breadth_swenlin_trading_oscillator_kernel",
            F64Kernel::DidiIndex => "didi_index_kernel",
            F64Kernel::DisparityIndex => "disparity_index_kernel",
            F64Kernel::Dm => "dm_kernel",
            F64Kernel::DonchianChannelWidth => "donchian_channel_width_kernel",
            F64Kernel::Dpo => "dpo_kernel",
            // ------------------------------------------------ closer 2, round 2
            F64Kernel::Mwdx => "mwdx_kernel",
            F64Kernel::Lrsi => "lrsi_kernel",
            F64Kernel::Pivot => "pivot_kernel",
            F64Kernel::Kaufmanstop => "kaufmanstop_kernel",
            F64Kernel::Sgf => "sgf_kernel",
            F64Kernel::PolynomialRegressionExtrapolation => {
                "polynomial_regression_extrapolation_kernel"
            }
            F64Kernel::DualUlcerIndex => "dual_ulcer_index_kernel",
            F64Kernel::HullButterflyOscillator => "hull_butterfly_oscillator_kernel",
            F64Kernel::RangeOscillator => "range_oscillator_kernel",
            F64Kernel::MarketStructureTrailingStop => {
                "market_structure_trailing_stop_kernel"
            }
            // ------------------------------------------ closer 3, round 2
            //
            // Every one of these lives in the .cu file its indicator already
            // ships in, beside the entry point the existing wrapper still
            // calls -- not in a parallel file.
            F64Kernel::VerticalHorizontalFilter => "vertical_horizontal_filter_kernel",
            F64Kernel::AdjustableMaAlternatingExtremities => {
                "adjustable_ma_alternating_extremities_kernel"
            }
            F64Kernel::AutocorrelationIndicator => "autocorrelation_indicator_kernel",
            F64Kernel::HistoricalVolatilityRank => "historical_volatility_rank_kernel",
            F64Kernel::HistoricalVolatilityPercentile => {
                "historical_volatility_percentile_kernel"
            }
            F64Kernel::DirectionalImbalanceIndex => "directional_imbalance_index_kernel",
            F64Kernel::CycleChannelOscillator => "cycle_channel_oscillator_kernel",
            F64Kernel::DynamicMomentumIndex => "dynamic_momentum_index_kernel",
            F64Kernel::EhlersAdaptiveCg => "ehlers_adaptive_cg_kernel",
            F64Kernel::EhlersAdaptiveCyberCycle => "ehlers_adaptive_cyber_cycle_kernel",
            F64Kernel::EhlersDataSamplingRelativeStrengthIndicator => {
                "ehlers_data_sampling_relative_strength_indicator_kernel"
            }
            F64Kernel::ExponentialTrend => "exponential_trend_kernel",
            F64Kernel::GeometricBiasOscillator => "geometric_bias_oscillator_kernel",
            F64Kernel::IntradayMomentumIndex => "intraday_momentum_index_kernel",
            F64Kernel::BullsVBears => "bulls_v_bears_kernel",
            F64Kernel::CandleStrengthOscillator => "candle_strength_oscillator_kernel",
            F64Kernel::CyberpunkValueTrendAnalyzer => "cyberpunk_value_trend_analyzer_kernel",
            F64Kernel::FvgPositioningAverage => "fvg_positioning_average_kernel",
            F64Kernel::HemaTrendLevels => "hema_trend_levels_kernel",
            F64Kernel::FibonacciTrailingStop => "fibonacci_trailing_stop_kernel",
            F64Kernel::GroverLlorensCycleOscillator => {
                "grover_llorens_cycle_oscillator_kernel"
            }
            F64Kernel::DemandIndex => "demand_index_kernel",
            F64Kernel::AdaptiveSchaffTrendCycle => "adaptive_schaff_trend_cycle_kernel",
            F64Kernel::EhlersLinearExtrapolationPredictor => {
                "ehlers_linear_extrapolation_predictor_kernel"
            }
            F64Kernel::EhlersAutocorrelationPeriodogram => {
                "ehlers_autocorrelation_periodogram_kernel"
            }
            F64Kernel::IctPropulsionBlock => "ict_propulsion_block_kernel",
            // ------------------------------------------ closer 4, round 2
            F64Kernel::KasePeakOscillatorWithDivergences => "kase_peak_oscillator_with_divergences_kernel",
            F64Kernel::KeltnerChannelWidthOscillator => "keltner_channel_width_oscillator_kernel",
            F64Kernel::Kst => "kst_kernel",
            F64Kernel::LeavittConvolutionAcceleration => "leavitt_convolution_acceleration_kernel",
            F64Kernel::MarketMeannessIndex => "market_meanness_index_kernel",
            F64Kernel::MarketStructureConfluence => "market_structure_confluence_kernel",
            F64Kernel::MonotonicityIndex => "monotonicity_index_kernel",
            F64Kernel::PremierRsiOscillator => "premier_rsi_oscillator_kernel",
            F64Kernel::PrettyGoodOscillator => "pretty_good_oscillator_kernel",
            F64Kernel::PriceDensityMarketNoise => "price_density_market_noise_kernel",
            F64Kernel::ProjectionOscillator => "projection_oscillator_kernel",
            F64Kernel::QqeWeightedOscillator => "qqe_weighted_oscillator_kernel",
            F64Kernel::RogersSatchellVolatility => "rogers_satchell_volatility_kernel",
            F64Kernel::RollingSkewnessKurtosis => "rolling_skewness_kurtosis_kernel",
            F64Kernel::SmoothTheilSen => "smooth_theil_sen_kernel",
            // ------------------------------------------ closer 4, round 3
            //
            // Each of the ten is compiled into its OWN module, because its
            // entry point was written into the `.cu` file its indicator
            // already ships in. Five live under `moving_averages/`, but
            // `compile_kernel` names the PTX after the file STEM, so the
            // module is `dma_kernel`, not `moving_averages_dma_kernel`.
            //
            // These arms MUST stay above the `_ => NEOETHOS_F64_MODULE`
            // catch-all below, for the reason spelled out under closer 1.
            F64Kernel::Bandpass => "bandpass_kernel",
            F64Kernel::BuffAverages => "buff_averages_kernel",
            F64Kernel::CoraWave => "cora_wave_kernel",
            F64Kernel::Dma => "dma_kernel",
            F64Kernel::FvgTrailingStop => "fvg_trailing_stop_kernel",
            F64Kernel::Halftrend => "halftrend_kernel",
            F64Kernel::ModGodMode => "mod_god_mode_kernel",
            F64Kernel::Ott => "ott_kernel",
            F64Kernel::Otto => "otto_kernel",
            F64Kernel::Prb => "prb_kernel",
            // ---------------------------------------------- closer 6, round 3
            // These arms MUST stay above the `_ => NEOETHOS_F64_MODULE`
            // catch-all: none of these six kernels lives in
            // `neoethos_f64_kernels.cu`, and falling through would look for
            // the symbol in a module that does not contain it.
            F64Kernel::ElasticVolumeWeightedMovingAverage => {
                "elastic_volume_weighted_moving_average_kernel"
            }
            F64Kernel::EmaDeviationCorrectedT3 => "ema_deviation_corrected_t3_kernel",
            F64Kernel::LogarithmicMovingAverage => "logarithmic_moving_average_kernel",
            F64Kernel::NOrderEma => "n_order_ema_kernel",
            F64Kernel::VolatilityAdjustedMa => "volatility_adjusted_ma_kernel",
            F64Kernel::WaveSmoother => "wave_smoother_kernel",
            // ------------------------------------------------ closer 3, round 3
            F64Kernel::ReversalSignals => "reversal_signals_kernel",
            F64Kernel::TrendFollower => "trend_follower_kernel",
            F64Kernel::VdubusDivergenceWavePatternGenerator => {
                "vdubus_divergence_wave_pattern_generator_kernel"
            }
            F64Kernel::VolatilityRatioAdaptiveRsx => "volatility_ratio_adaptive_rsx_kernel",
            F64Kernel::VolumeEnergyReservoirs => "volume_energy_reservoirs_kernel",
            F64Kernel::VolumeWeightedRelativeStrengthIndex => {
                "volume_weighted_relative_strength_index_kernel"
            }
            F64Kernel::VolumeWeightedStochasticRsi => "volume_weighted_stochastic_rsi_kernel",
            F64Kernel::ZigZagChannels => "zig_zag_channels_kernel",
            F64Kernel::Alphatrend => "alphatrend_kernel",
            F64Kernel::Avsl => "avsl_kernel",

            // ------------------------------------------ closer 1, round 3
            F64Kernel::FibonacciEntryBands => "fibonacci_entry_bands_kernel",
            F64Kernel::GoertzelCycleCompositeWave => "goertzel_cycle_composite_wave_kernel",
            F64Kernel::HalfCausalEstimator => "half_causal_estimator_kernel",
            F64Kernel::IchimokuOscillator => "ichimoku_oscillator_kernel",
            F64Kernel::InsyncIndex => "insync_index_kernel",
            F64Kernel::LinearRegressionIntensity => "linear_regression_intensity_kernel",
            F64Kernel::MacdWaveSignalPro => "macd_wave_signal_pro_kernel",
            F64Kernel::MesaStochasticMultiLength => "mesa_stochastic_multi_length_kernel",
            F64Kernel::MovingAverageCrossProbability => "moving_average_cross_probability_kernel",
            F64Kernel::MultiLengthStochasticAverage => "multi_length_stochastic_average_kernel",
            // ---------------------------------------------- closer 2, round 3
            // Same rule as every block above: these arms MUST stay above the
            // `_ => NEOETHOS_F64_MODULE` catch-all. Each of these ten entry
            // points was written INTO the indicator's own `.cu` file, beside
            // the bespoke-shaped f64 kernel its wrapper already loads, so
            // falling through would look for the symbol in a module that does
            // not contain it.
            F64Kernel::NeighboringTrailingStop => "neighboring_trailing_stop_kernel",
            F64Kernel::NonlinearRegressionZeroLagMovingAverage => {
                "nonlinear_regression_zero_lag_moving_average_kernel"
            }
            F64Kernel::NormalizedResonator => "normalized_resonator_kernel",
            F64Kernel::NormalizedVolumeTrueRange => "normalized_volume_true_range_kernel",
            F64Kernel::PossibleRsi => "possible_rsi_kernel",
            F64Kernel::PriceMovingAverageRatioPercentile => {
                "price_moving_average_ratio_percentile_kernel"
            }
            F64Kernel::RangeBreakoutSignals => "range_breakout_signals_kernel",
            F64Kernel::RangeFilteredTrendSignals => "range_filtered_trend_signals_kernel",
            F64Kernel::RegressionSlopeOscillator => "regression_slope_oscillator_kernel",
            F64Kernel::RelativeStrengthIndexWaveIndicator => {
                "relative_strength_index_wave_indicator_kernel"
            }
            _ => NEOETHOS_F64_MODULE,

        }
    }
}

/// The module every kernel written into `neoethos_f64_kernels.cu` lives in.
pub const NEOETHOS_F64_MODULE: &str = "neoethos_f64_kernels";

/// Load the fatbin for one per-indicator f64 module.
///
/// `load_cuda_embedded_module!` takes a string LITERAL — it embeds the fatbin
/// and the PTX at compile time — so this is a match on the stem and not a
/// lookup. Every arm must correspond to a `compile_kernel` call in `build.rs`
/// and to a `module_stem` arm above; a stem with no arm is a loud `Err` at
/// construction rather than a missing symbol at launch.
fn load_f64_module(stem: &str) -> Result<Module, CudaF64IndicatorError> {
    let m = match stem {
        NEOETHOS_F64_MODULE => crate::load_cuda_embedded_module!("neoethos_f64_kernels")?,
        "sqwma_kernel" => crate::load_cuda_embedded_module!("sqwma_kernel")?,
        "deviation_kernel" => crate::load_cuda_embedded_module!("deviation_kernel")?,
        "mean_ad_kernel" => crate::load_cuda_embedded_module!("mean_ad_kernel")?,
        "ao_kernel" => crate::load_cuda_embedded_module!("ao_kernel")?,
        "linearreg_slope_kernel" => crate::load_cuda_embedded_module!("linearreg_slope_kernel")?,
        "tsf_kernel" => crate::load_cuda_embedded_module!("tsf_kernel")?,
        "highpass_kernel" => crate::load_cuda_embedded_module!("highpass_kernel")?,
        "decycler_kernel" => crate::load_cuda_embedded_module!("decycler_kernel")?,
        "supersmoother_kernel" => crate::load_cuda_embedded_module!("supersmoother_kernel")?,
        "tilson_kernel" => crate::load_cuda_embedded_module!("tilson_kernel")?,
        "wad_kernel" => crate::load_cuda_embedded_module!("wad_kernel")?,
        "sar_kernel" => crate::load_cuda_embedded_module!("sar_kernel")?,
        "dti_kernel" => crate::load_cuda_embedded_module!("dti_kernel")?,
        "zscore_kernel" => crate::load_cuda_embedded_module!("zscore_kernel")?,
        "pfe_kernel" => crate::load_cuda_embedded_module!("pfe_kernel")?,
        "chande_kernel" => crate::load_cuda_embedded_module!("chande_kernel")?,
        "di_kernel" => crate::load_cuda_embedded_module!("di_kernel")?,
        "kdj_kernel" => crate::load_cuda_embedded_module!("kdj_kernel")?,
        "aso_kernel" => crate::load_cuda_embedded_module!("aso_kernel")?,
        "wto_kernel" => crate::load_cuda_embedded_module!("wto_kernel")?,
        "range_filter_kernel" => crate::load_cuda_embedded_module!("range_filter_kernel")?,
        "correlation_cycle_kernel" => crate::load_cuda_embedded_module!("correlation_cycle_kernel")?,
        "mama_kernel" => crate::load_cuda_embedded_module!("mama_kernel")?,
        "volume_adjusted_ma_kernel" => crate::load_cuda_embedded_module!("volume_adjusted_ma_kernel")?,
        "reverse_rsi_kernel" => crate::load_cuda_embedded_module!("reverse_rsi_kernel")?,
        "ehlers_ecema_kernel" => crate::load_cuda_embedded_module!("ehlers_ecema_kernel")?,
        "devstop_kernel" => crate::load_cuda_embedded_module!("devstop_kernel")?,
        "chandelier_exit_kernel" => crate::load_cuda_embedded_module!("chandelier_exit_kernel")?,
        "minmax_kernel" => crate::load_cuda_embedded_module!("minmax_kernel")?,
        // ------------------------------------------------------------- shard 4 (S4)
        "er_kernel" => crate::load_cuda_embedded_module!("er_kernel")?,
        "linearreg_angle_kernel" => crate::load_cuda_embedded_module!("linearreg_angle_kernel")?,
        "linearreg_intercept_kernel" => crate::load_cuda_embedded_module!("linearreg_intercept_kernel")?,
        "highpass2_kernel" => crate::load_cuda_embedded_module!("highpass2_kernel")?,
        "supersmoother_3_pole_kernel" => crate::load_cuda_embedded_module!("supersmoother_3_pole_kernel")?,
        "cwma_kernel" => crate::load_cuda_embedded_module!("cwma_kernel")?,
        "cmo_kernel" => crate::load_cuda_embedded_module!("cmo_kernel")?,
        "stddev_kernel" => crate::load_cuda_embedded_module!("stddev_kernel")?,
        "ui_kernel" => crate::load_cuda_embedded_module!("ui_kernel")?,
        "bollinger_bands_kernel" => crate::load_cuda_embedded_module!("bollinger_bands_kernel")?,
        "ehma_kernel" => crate::load_cuda_embedded_module!("ehma_kernel")?,
        "macd_kernel" => crate::load_cuda_embedded_module!("macd_kernel")?,
        "ift_rsi_kernel" => crate::load_cuda_embedded_module!("ift_rsi_kernel")?,
        "damiani_volatmeter_kernel" => crate::load_cuda_embedded_module!("damiani_volatmeter_kernel")?,
        "wavetrend_kernel" => crate::load_cuda_embedded_module!("wavetrend_kernel")?,
        "dx_kernel" => crate::load_cuda_embedded_module!("dx_kernel")?,
        "frama_kernel" => crate::load_cuda_embedded_module!("frama_kernel")?,
        "cksp_kernel" => crate::load_cuda_embedded_module!("cksp_kernel")?,
        "ttm_squeeze_kernel" => crate::load_cuda_embedded_module!("ttm_squeeze_kernel")?,
        "mass_kernel" => crate::load_cuda_embedded_module!("mass_kernel")?,
        "aroon_kernel" => crate::load_cuda_embedded_module!("aroon_kernel")?,
        "acosc_kernel" => crate::load_cuda_embedded_module!("acosc_kernel")?,
        "vpci_kernel" => crate::load_cuda_embedded_module!("vpci_kernel")?,
        "ad_kernel" => crate::load_cuda_embedded_module!("ad_kernel")?,
        "dvdiqqe_kernel" => crate::load_cuda_embedded_module!("dvdiqqe_kernel")?,
        "cci_cycle_kernel" => crate::load_cuda_embedded_module!("cci_cycle_kernel")?,
        // ------------------------------------------------------------- shard 1 (S1)
        "apo_kernel" => crate::load_cuda_embedded_module!("apo_kernel")?,
        "vidya_kernel" => crate::load_cuda_embedded_module!("vidya_kernel")?,
        "gatorosc_kernel" => crate::load_cuda_embedded_module!("gatorosc_kernel")?,
        "ppo_kernel" => crate::load_cuda_embedded_module!("ppo_kernel")?,
        "pma_kernel" => crate::load_cuda_embedded_module!("pma_kernel")?,
        "kama_kernel" => crate::load_cuda_embedded_module!("kama_kernel")?,
        "linreg_kernel" => crate::load_cuda_embedded_module!("linreg_kernel")?,
        "edcf_kernel" => crate::load_cuda_embedded_module!("edcf_kernel")?,
        "alma_kernel" => crate::load_cuda_embedded_module!("alma_kernel")?,
        "hma_kernel" => crate::load_cuda_embedded_module!("hma_kernel")?,
        "kurtosis_kernel" => crate::load_cuda_embedded_module!("kurtosis_kernel")?,
        "alligator_kernel" => crate::load_cuda_embedded_module!("alligator_kernel")?,
        "nvi_kernel" => crate::load_cuda_embedded_module!("nvi_kernel")?,
        "fisher_kernel" => crate::load_cuda_embedded_module!("fisher_kernel")?,
        "safezonestop_kernel" => crate::load_cuda_embedded_module!("safezonestop_kernel")?,
        "chop_kernel" => crate::load_cuda_embedded_module!("chop_kernel")?,
        "stochf_kernel" => crate::load_cuda_embedded_module!("stochf_kernel")?,
        "emv_kernel" => crate::load_cuda_embedded_module!("emv_kernel")?,
        "kvo_kernel" => crate::load_cuda_embedded_module!("kvo_kernel")?,
        "rsx_kernel" => crate::load_cuda_embedded_module!("rsx_kernel")?,
        "trix_kernel" => crate::load_cuda_embedded_module!("trix_kernel")?,
        "vpt_kernel" => crate::load_cuda_embedded_module!("vpt_kernel")?,
        "pvi_kernel" => crate::load_cuda_embedded_module!("pvi_kernel")?,
        "ehlers_itrend_kernel" => crate::load_cuda_embedded_module!("ehlers_itrend_kernel")?,
        "ehlers_kama_kernel" => crate::load_cuda_embedded_module!("ehlers_kama_kernel")?,
        "sama_kernel" => crate::load_cuda_embedded_module!("sama_kernel")?,
        "nama_kernel" => crate::load_cuda_embedded_module!("nama_kernel")?,
        "pwma_kernel" => crate::load_cuda_embedded_module!("pwma_kernel")?,
        "tradjema_kernel" => crate::load_cuda_embedded_module!("tradjema_kernel")?,
        "maaq_kernel" => crate::load_cuda_embedded_module!("maaq_kernel")?,
        "jma_kernel" => crate::load_cuda_embedded_module!("jma_kernel")?,
        "reflex_kernel" => crate::load_cuda_embedded_module!("reflex_kernel")?,
        "gaussian_kernel" => crate::load_cuda_embedded_module!("gaussian_kernel")?,
        "fwma_kernel" => crate::load_cuda_embedded_module!("fwma_kernel")?,
        "hwma_kernel" => crate::load_cuda_embedded_module!("hwma_kernel")?,
        "jsa_kernel" => crate::load_cuda_embedded_module!("jsa_kernel")?,
        "nma_kernel" => crate::load_cuda_embedded_module!("nma_kernel")?,
        "swma_kernel" => crate::load_cuda_embedded_module!("swma_kernel")?,
        "trendflex_kernel" => crate::load_cuda_embedded_module!("trendflex_kernel")?,
        "vpwma_kernel" => crate::load_cuda_embedded_module!("vpwma_kernel")?,
        "cfo_kernel" => crate::load_cuda_embedded_module!("cfo_kernel")?,
        "var_kernel" => crate::load_cuda_embedded_module!("var_kernel")?,
        "bollinger_bands_width_kernel" => crate::load_cuda_embedded_module!("bollinger_bands_width_kernel")?,
        "dec_osc_kernel" => crate::load_cuda_embedded_module!("dec_osc_kernel")?,
        "voss_kernel" => crate::load_cuda_embedded_module!("voss_kernel")?,
        "percentile_nearest_rank_kernel" => crate::load_cuda_embedded_module!("percentile_nearest_rank_kernel")?,
        "ttm_trend_kernel" => crate::load_cuda_embedded_module!("ttm_trend_kernel")?,
        "vi_kernel" => crate::load_cuda_embedded_module!("vi_kernel")?,
        "cvi_kernel" => crate::load_cuda_embedded_module!("cvi_kernel")?,
        "correl_hl_kernel" => crate::load_cuda_embedded_module!("correl_hl_kernel")?,
        "aroonosc_kernel" => crate::load_cuda_embedded_module!("aroonosc_kernel")?,
        "parkinson_volatility_kernel" => crate::load_cuda_embedded_module!("parkinson_volatility_kernel")?,
        "historical_volatility_kernel" => crate::load_cuda_embedded_module!("historical_volatility_kernel")?,
        "donchian_kernel" => crate::load_cuda_embedded_module!("donchian_kernel")?,
        // ------------------------------------------------------- closer 5
        "velocity_kernel" => crate::load_cuda_embedded_module!("velocity_kernel")?,
        "velocity_acceleration_indicator_kernel" => crate::load_cuda_embedded_module!("velocity_acceleration_indicator_kernel")?,
        "velocity_acceleration_convergence_divergence_indicator_kernel" => crate::load_cuda_embedded_module!("velocity_acceleration_convergence_divergence_indicator_kernel")?,
        "trend_direction_force_index_kernel" => crate::load_cuda_embedded_module!("trend_direction_force_index_kernel")?,
        "trend_continuation_factor_kernel" => crate::load_cuda_embedded_module!("trend_continuation_factor_kernel")?,
        "trima_kernel" => crate::load_cuda_embedded_module!("trima_kernel")?,
        "trend_trigger_factor_kernel" => crate::load_cuda_embedded_module!("trend_trigger_factor_kernel")?,
        "volume_weighted_rsi_kernel" => crate::load_cuda_embedded_module!("volume_weighted_rsi_kernel")?,
        "volume_zone_oscillator_kernel" => crate::load_cuda_embedded_module!("volume_zone_oscillator_kernel")?,
        "vosc_kernel" => crate::load_cuda_embedded_module!("vosc_kernel")?,
        "ultosc_kernel" => crate::load_cuda_embedded_module!("ultosc_kernel")?,
        // ---------------------------------------------------------- closer 4
        "psychological_line_kernel" => crate::load_cuda_embedded_module!("psychological_line_kernel")?,
        "rank_correlation_index_kernel" => crate::load_cuda_embedded_module!("rank_correlation_index_kernel")?,
        "qstick_kernel" => crate::load_cuda_embedded_module!("qstick_kernel")?,
        "sinwma_kernel" => crate::load_cuda_embedded_module!("sinwma_kernel")?,
        "srwma_kernel" => crate::load_cuda_embedded_module!("srwma_kernel")?,
        "rolling_z_score_trend_kernel" => crate::load_cuda_embedded_module!("rolling_z_score_trend_kernel")?,
        "random_walk_index_kernel" => crate::load_cuda_embedded_module!("random_walk_index_kernel")?,
        // ------------------------------------------------------------- closer 3
        "l1_ehlers_phasor_kernel" => crate::load_cuda_embedded_module!("l1_ehlers_phasor_kernel")?,
        "l2_ehlers_signal_to_noise_kernel" => {
            crate::load_cuda_embedded_module!("l2_ehlers_signal_to_noise_kernel")?
        }
        "kairi_relative_index_kernel" => {
            crate::load_cuda_embedded_module!("kairi_relative_index_kernel")?
        }
        "linear_correlation_oscillator_kernel" => {
            crate::load_cuda_embedded_module!("linear_correlation_oscillator_kernel")?
        }
        "medium_ad_kernel" => crate::load_cuda_embedded_module!("medium_ad_kernel")?,
        "marketefi_kernel" => crate::load_cuda_embedded_module!("marketefi_kernel")?,
        "momentum_ratio_oscillator_kernel" => {
            crate::load_cuda_embedded_module!("momentum_ratio_oscillator_kernel")?
        }
        "on_balance_volume_oscillator_kernel" => {
            crate::load_cuda_embedded_module!("on_balance_volume_oscillator_kernel")?
        }
        // ---------------------------------------------------------- closer 6
        "emd_kernel" => crate::load_cuda_embedded_module!("emd_kernel")?,
        "keltner_kernel" => crate::load_cuda_embedded_module!("keltner_kernel")?,
        "stoch_kernel" => crate::load_cuda_embedded_module!("stoch_kernel")?,
        "nadaraya_watson_envelope_kernel" => {
            crate::load_cuda_embedded_module!("nadaraya_watson_envelope_kernel")?
        }
        // ------------------------------------------------------------ closer 2
        "ehlers_detrending_filter_kernel" => crate::load_cuda_embedded_module!("ehlers_detrending_filter_kernel")?,
        "ehlers_simple_cycle_indicator_kernel" => crate::load_cuda_embedded_module!("ehlers_simple_cycle_indicator_kernel")?,
        "ehlers_smoothed_adaptive_momentum_kernel" => crate::load_cuda_embedded_module!("ehlers_smoothed_adaptive_momentum_kernel")?,
        "ewma_volatility_kernel" => crate::load_cuda_embedded_module!("ewma_volatility_kernel")?,
        "fractal_dimension_index_kernel" => crate::load_cuda_embedded_module!("fractal_dimension_index_kernel")?,
        "gopalakrishnan_range_index_kernel" => crate::load_cuda_embedded_module!("gopalakrishnan_range_index_kernel")?,
        "garman_klass_volatility_kernel" => crate::load_cuda_embedded_module!("garman_klass_volatility_kernel")?,
        "impulse_macd_kernel" => crate::load_cuda_embedded_module!("impulse_macd_kernel")?,
        "hypertrend_kernel" => crate::load_cuda_embedded_module!("hypertrend_kernel")?,
        "emd_trend_kernel" => crate::load_cuda_embedded_module!("emd_trend_kernel")?,
        "epma_kernel" => crate::load_cuda_embedded_module!("epma_kernel")?,
        "fosc_kernel" => crate::load_cuda_embedded_module!("fosc_kernel")?,
        "ehlers_pma_kernel" => crate::load_cuda_embedded_module!("ehlers_pma_kernel")?,
        "eri_kernel" => crate::load_cuda_embedded_module!("eri_kernel")?,
        // ---------------------------------------------------------- closer 1
        "absolute_strength_index_oscillator_kernel" => crate::load_cuda_embedded_module!("absolute_strength_index_oscillator_kernel")?,
        "accumulation_swing_index_kernel" => crate::load_cuda_embedded_module!("accumulation_swing_index_kernel")?,
        "adaptive_bandpass_trigger_oscillator_kernel" => crate::load_cuda_embedded_module!("adaptive_bandpass_trigger_oscillator_kernel")?,
        "adaptive_bounds_rsi_kernel" => crate::load_cuda_embedded_module!("adaptive_bounds_rsi_kernel")?,
        "adaptive_macd_kernel" => crate::load_cuda_embedded_module!("adaptive_macd_kernel")?,
        "adaptive_momentum_oscillator_kernel" => crate::load_cuda_embedded_module!("adaptive_momentum_oscillator_kernel")?,
        "advance_decline_line_kernel" => crate::load_cuda_embedded_module!("advance_decline_line_kernel")?,
        "andean_oscillator_kernel" => crate::load_cuda_embedded_module!("andean_oscillator_kernel")?,
        "atr_percentile_kernel" => crate::load_cuda_embedded_module!("atr_percentile_kernel")?,
        "bop_kernel" => crate::load_cuda_embedded_module!("bop_kernel")?,
        "bull_power_vs_bear_power_kernel" => crate::load_cuda_embedded_module!("bull_power_vs_bear_power_kernel")?,
        "cg_kernel" => crate::load_cuda_embedded_module!("cg_kernel")?,
        "coppock_kernel" => crate::load_cuda_embedded_module!("coppock_kernel")?,
        "daily_factor_kernel" => crate::load_cuda_embedded_module!("daily_factor_kernel")?,
        "decisionpoint_breadth_swenlin_trading_oscillator_kernel" => crate::load_cuda_embedded_module!("decisionpoint_breadth_swenlin_trading_oscillator_kernel")?,
        "didi_index_kernel" => crate::load_cuda_embedded_module!("didi_index_kernel")?,
        "disparity_index_kernel" => crate::load_cuda_embedded_module!("disparity_index_kernel")?,
        "dm_kernel" => crate::load_cuda_embedded_module!("dm_kernel")?,
        "donchian_channel_width_kernel" => crate::load_cuda_embedded_module!("donchian_channel_width_kernel")?,
        "dpo_kernel" => crate::load_cuda_embedded_module!("dpo_kernel")?,
        // ---------------------------------------------------------- closer 2b
        "ehlers_fm_demodulator_kernel" => crate::load_cuda_embedded_module!("ehlers_fm_demodulator_kernel")?,
        "forward_backward_exponential_oscillator_kernel" => crate::load_cuda_embedded_module!("forward_backward_exponential_oscillator_kernel")?,
        "gmma_oscillator_kernel" => crate::load_cuda_embedded_module!("gmma_oscillator_kernel")?,
        "evasive_supertrend_kernel" => crate::load_cuda_embedded_module!("evasive_supertrend_kernel")?,
        // ------------------------------------------------- closer 6, round 2
        "msw_kernel" => crate::load_cuda_embedded_module!("msw_kernel")?,
        "yang_zhang_volatility_kernel" => crate::load_cuda_embedded_module!("yang_zhang_volatility_kernel")?,
        "qqe_kernel" => crate::load_cuda_embedded_module!("qqe_kernel")?,
        "srsi_kernel" => crate::load_cuda_embedded_module!("srsi_kernel")?,
        "rvi_kernel" => crate::load_cuda_embedded_module!("rvi_kernel")?,
        "net_myrsi_kernel" => crate::load_cuda_embedded_module!("net_myrsi_kernel")?,
        "vlma_kernel" => crate::load_cuda_embedded_module!("vlma_kernel")?,
        "stc_kernel" => crate::load_cuda_embedded_module!("stc_kernel")?,
        // ---------------------------------------------------- closer 2, round 2
        "mwdx_kernel" => crate::load_cuda_embedded_module!("mwdx_kernel")?,
        "lrsi_kernel" => crate::load_cuda_embedded_module!("lrsi_kernel")?,
        "pivot_kernel" => crate::load_cuda_embedded_module!("pivot_kernel")?,
        "kaufmanstop_kernel" => crate::load_cuda_embedded_module!("kaufmanstop_kernel")?,
        "sgf_kernel" => crate::load_cuda_embedded_module!("sgf_kernel")?,
        "polynomial_regression_extrapolation_kernel" => {
            crate::load_cuda_embedded_module!("polynomial_regression_extrapolation_kernel")?
        }
        "dual_ulcer_index_kernel" => crate::load_cuda_embedded_module!("dual_ulcer_index_kernel")?,
        "hull_butterfly_oscillator_kernel" => {
            crate::load_cuda_embedded_module!("hull_butterfly_oscillator_kernel")?
        }
        "range_oscillator_kernel" => crate::load_cuda_embedded_module!("range_oscillator_kernel")?,
        "market_structure_trailing_stop_kernel" => {
            crate::load_cuda_embedded_module!("market_structure_trailing_stop_kernel")?
        }
        // ---------------------------------------------- closer 3, round 2
        "vertical_horizontal_filter_kernel" => {
            crate::load_cuda_embedded_module!("vertical_horizontal_filter_kernel")?
        }
        "adjustable_ma_alternating_extremities_kernel" => {
            crate::load_cuda_embedded_module!("adjustable_ma_alternating_extremities_kernel")?
        }
        "autocorrelation_indicator_kernel" => {
            crate::load_cuda_embedded_module!("autocorrelation_indicator_kernel")?
        }
        "historical_volatility_rank_kernel" => {
            crate::load_cuda_embedded_module!("historical_volatility_rank_kernel")?
        }
        "historical_volatility_percentile_kernel" => {
            crate::load_cuda_embedded_module!("historical_volatility_percentile_kernel")?
        }
        "directional_imbalance_index_kernel" => {
            crate::load_cuda_embedded_module!("directional_imbalance_index_kernel")?
        }
        "cycle_channel_oscillator_kernel" => {
            crate::load_cuda_embedded_module!("cycle_channel_oscillator_kernel")?
        }
        "dynamic_momentum_index_kernel" => {
            crate::load_cuda_embedded_module!("dynamic_momentum_index_kernel")?
        }
        "ehlers_adaptive_cg_kernel" => {
            crate::load_cuda_embedded_module!("ehlers_adaptive_cg_kernel")?
        }
        "ehlers_adaptive_cyber_cycle_kernel" => {
            crate::load_cuda_embedded_module!("ehlers_adaptive_cyber_cycle_kernel")?
        }
        "ehlers_data_sampling_relative_strength_indicator_kernel" => {
            crate::load_cuda_embedded_module!(
                "ehlers_data_sampling_relative_strength_indicator_kernel"
            )?
        }
        "exponential_trend_kernel" => {
            crate::load_cuda_embedded_module!("exponential_trend_kernel")?
        }
        "geometric_bias_oscillator_kernel" => {
            crate::load_cuda_embedded_module!("geometric_bias_oscillator_kernel")?
        }
        "intraday_momentum_index_kernel" => {
            crate::load_cuda_embedded_module!("intraday_momentum_index_kernel")?
        }
        "bulls_v_bears_kernel" => crate::load_cuda_embedded_module!("bulls_v_bears_kernel")?,
        "candle_strength_oscillator_kernel" => {
            crate::load_cuda_embedded_module!("candle_strength_oscillator_kernel")?
        }
        "cyberpunk_value_trend_analyzer_kernel" => {
            crate::load_cuda_embedded_module!("cyberpunk_value_trend_analyzer_kernel")?
        }
        "fvg_positioning_average_kernel" => {
            crate::load_cuda_embedded_module!("fvg_positioning_average_kernel")?
        }
        "hema_trend_levels_kernel" => {
            crate::load_cuda_embedded_module!("hema_trend_levels_kernel")?
        }
        "fibonacci_trailing_stop_kernel" => {
            crate::load_cuda_embedded_module!("fibonacci_trailing_stop_kernel")?
        }
        "grover_llorens_cycle_oscillator_kernel" => {
            crate::load_cuda_embedded_module!("grover_llorens_cycle_oscillator_kernel")?
        }
        "demand_index_kernel" => crate::load_cuda_embedded_module!("demand_index_kernel")?,
        "adaptive_schaff_trend_cycle_kernel" => {
            crate::load_cuda_embedded_module!("adaptive_schaff_trend_cycle_kernel")?
        }
        "ehlers_linear_extrapolation_predictor_kernel" => {
            crate::load_cuda_embedded_module!("ehlers_linear_extrapolation_predictor_kernel")?
        }
        "ehlers_autocorrelation_periodogram_kernel" => {
            crate::load_cuda_embedded_module!("ehlers_autocorrelation_periodogram_kernel")?
        }
        "ict_propulsion_block_kernel" => {
            crate::load_cuda_embedded_module!("ict_propulsion_block_kernel")?
        }
        // ---------------------------------------------- closer 4, round 2
        "kase_peak_oscillator_with_divergences_kernel" => crate::load_cuda_embedded_module!("kase_peak_oscillator_with_divergences_kernel")?,
        "keltner_channel_width_oscillator_kernel" => crate::load_cuda_embedded_module!("keltner_channel_width_oscillator_kernel")?,
        "kst_kernel" => crate::load_cuda_embedded_module!("kst_kernel")?,
        "leavitt_convolution_acceleration_kernel" => crate::load_cuda_embedded_module!("leavitt_convolution_acceleration_kernel")?,
        "market_meanness_index_kernel" => crate::load_cuda_embedded_module!("market_meanness_index_kernel")?,
        "market_structure_confluence_kernel" => crate::load_cuda_embedded_module!("market_structure_confluence_kernel")?,
        "monotonicity_index_kernel" => crate::load_cuda_embedded_module!("monotonicity_index_kernel")?,
        "premier_rsi_oscillator_kernel" => crate::load_cuda_embedded_module!("premier_rsi_oscillator_kernel")?,
        "pretty_good_oscillator_kernel" => crate::load_cuda_embedded_module!("pretty_good_oscillator_kernel")?,
        "price_density_market_noise_kernel" => crate::load_cuda_embedded_module!("price_density_market_noise_kernel")?,
        "projection_oscillator_kernel" => crate::load_cuda_embedded_module!("projection_oscillator_kernel")?,
        "qqe_weighted_oscillator_kernel" => crate::load_cuda_embedded_module!("qqe_weighted_oscillator_kernel")?,
        "rogers_satchell_volatility_kernel" => crate::load_cuda_embedded_module!("rogers_satchell_volatility_kernel")?,
        "rolling_skewness_kurtosis_kernel" => crate::load_cuda_embedded_module!("rolling_skewness_kurtosis_kernel")?,
        "smooth_theil_sen_kernel" => crate::load_cuda_embedded_module!("smooth_theil_sen_kernel")?,
        // ------------------------------------------ closer 4, round 3
        "bandpass_kernel" => crate::load_cuda_embedded_module!("bandpass_kernel")?,
        "buff_averages_kernel" => crate::load_cuda_embedded_module!("buff_averages_kernel")?,
        "cora_wave_kernel" => crate::load_cuda_embedded_module!("cora_wave_kernel")?,
        "dma_kernel" => crate::load_cuda_embedded_module!("dma_kernel")?,
        "fvg_trailing_stop_kernel" => crate::load_cuda_embedded_module!("fvg_trailing_stop_kernel")?,
        "halftrend_kernel" => crate::load_cuda_embedded_module!("halftrend_kernel")?,
        "mod_god_mode_kernel" => crate::load_cuda_embedded_module!("mod_god_mode_kernel")?,
        "ott_kernel" => crate::load_cuda_embedded_module!("ott_kernel")?,
        "otto_kernel" => crate::load_cuda_embedded_module!("otto_kernel")?,
        "prb_kernel" => crate::load_cuda_embedded_module!("prb_kernel")?,
        // ------------------------------------------------- closer 6, round 3
        "elastic_volume_weighted_moving_average_kernel" => {
            crate::load_cuda_embedded_module!("elastic_volume_weighted_moving_average_kernel")?
        }
        "ema_deviation_corrected_t3_kernel" => {
            crate::load_cuda_embedded_module!("ema_deviation_corrected_t3_kernel")?
        }
        "logarithmic_moving_average_kernel" => {
            crate::load_cuda_embedded_module!("logarithmic_moving_average_kernel")?
        }
        "n_order_ema_kernel" => crate::load_cuda_embedded_module!("n_order_ema_kernel")?,
        "volatility_adjusted_ma_kernel" => {
            crate::load_cuda_embedded_module!("volatility_adjusted_ma_kernel")?
        }
        "wave_smoother_kernel" => crate::load_cuda_embedded_module!("wave_smoother_kernel")?,
        // --------------------------------------------------- closer 5, round 2
        "smoothed_gaussian_trend_filter_kernel" => crate::load_cuda_embedded_module!("smoothed_gaussian_trend_filter_kernel")?,
        "spearman_correlation_kernel" => crate::load_cuda_embedded_module!("spearman_correlation_kernel")?,
        "squeeze_index_kernel" => crate::load_cuda_embedded_module!("squeeze_index_kernel")?,
        "standardized_psar_oscillator_kernel" => crate::load_cuda_embedded_module!("standardized_psar_oscillator_kernel")?,
        "statistical_trailing_stop_kernel" => crate::load_cuda_embedded_module!("statistical_trailing_stop_kernel")?,
        "stochastic_adaptive_d_kernel" => crate::load_cuda_embedded_module!("stochastic_adaptive_d_kernel")?,
        "stochastic_connors_rsi_kernel" => crate::load_cuda_embedded_module!("stochastic_connors_rsi_kernel")?,
        "stochastic_distance_kernel" => crate::load_cuda_embedded_module!("stochastic_distance_kernel")?,
        "stochastic_money_flow_index_kernel" => crate::load_cuda_embedded_module!("stochastic_money_flow_index_kernel")?,
        "supertrend_kernel" => crate::load_cuda_embedded_module!("supertrend_kernel")?,
        "supertrend_oscillator_kernel" => crate::load_cuda_embedded_module!("supertrend_oscillator_kernel")?,
        "supertrend_recovery_kernel" => crate::load_cuda_embedded_module!("supertrend_recovery_kernel")?,
        "trend_flow_trail_kernel" => crate::load_cuda_embedded_module!("trend_flow_trail_kernel")?,
        "twiggs_money_flow_kernel" => crate::load_cuda_embedded_module!("twiggs_money_flow_kernel")?,
        "volatility_quality_index_kernel" => crate::load_cuda_embedded_module!("volatility_quality_index_kernel")?,
        "vwap_deviation_oscillator_kernel" => crate::load_cuda_embedded_module!("vwap_deviation_oscillator_kernel")?,
        "vwap_zscore_with_signals_kernel" => crate::load_cuda_embedded_module!("vwap_zscore_with_signals_kernel")?,
        "adosc_kernel" => crate::load_cuda_embedded_module!("adosc_kernel")?,
        // ---------------------------------------------- closer 5, round 3
        "rsmk_kernel" => crate::load_cuda_embedded_module!("rsmk_kernel")?,
        "squeeze_momentum_kernel" => crate::load_cuda_embedded_module!("squeeze_momentum_kernel")?,
        "uma_kernel" => crate::load_cuda_embedded_module!("uma_kernel")?,
        "lpc_kernel" => crate::load_cuda_embedded_module!("lpc_kernel")?,
        "mab_kernel" => crate::load_cuda_embedded_module!("mab_kernel")?,
        "macz_kernel" => crate::load_cuda_embedded_module!("macz_kernel")?,
        "vwmacd_kernel" => crate::load_cuda_embedded_module!("vwmacd_kernel")?,
        "corrected_moving_average_kernel" => crate::load_cuda_embedded_module!("corrected_moving_average_kernel")?,
        "ehlers_undersampled_double_moving_average_kernel" => crate::load_cuda_embedded_module!("ehlers_undersampled_double_moving_average_kernel")?,
        // ---------------------------------------------- closer 3, round 3
        "reversal_signals_kernel" => {
            crate::load_cuda_embedded_module!("reversal_signals_kernel")?
        }
        "trend_follower_kernel" => {
            crate::load_cuda_embedded_module!("trend_follower_kernel")?
        }
        "vdubus_divergence_wave_pattern_generator_kernel" => {
            crate::load_cuda_embedded_module!("vdubus_divergence_wave_pattern_generator_kernel")?
        }
        "volatility_ratio_adaptive_rsx_kernel" => {
            crate::load_cuda_embedded_module!("volatility_ratio_adaptive_rsx_kernel")?
        }
        "volume_energy_reservoirs_kernel" => {
            crate::load_cuda_embedded_module!("volume_energy_reservoirs_kernel")?
        }
        "volume_weighted_relative_strength_index_kernel" => {
            crate::load_cuda_embedded_module!("volume_weighted_relative_strength_index_kernel")?
        }
        "volume_weighted_stochastic_rsi_kernel" => {
            crate::load_cuda_embedded_module!("volume_weighted_stochastic_rsi_kernel")?
        }
        "zig_zag_channels_kernel" => {
            crate::load_cuda_embedded_module!("zig_zag_channels_kernel")?
        }
        "alphatrend_kernel" => {
            crate::load_cuda_embedded_module!("alphatrend_kernel")?
        }
        "avsl_kernel" => {
            crate::load_cuda_embedded_module!("avsl_kernel")?
        }
        // ---------------------------------------------------- closer 2, round 3
        "neighboring_trailing_stop_kernel" => {
            crate::load_cuda_embedded_module!("neighboring_trailing_stop_kernel")?
        }
        "nonlinear_regression_zero_lag_moving_average_kernel" => {
            crate::load_cuda_embedded_module!("nonlinear_regression_zero_lag_moving_average_kernel")?
        }
        "normalized_resonator_kernel" => {
            crate::load_cuda_embedded_module!("normalized_resonator_kernel")?
        }
        "normalized_volume_true_range_kernel" => {
            crate::load_cuda_embedded_module!("normalized_volume_true_range_kernel")?
        }
        "possible_rsi_kernel" => {
            crate::load_cuda_embedded_module!("possible_rsi_kernel")?
        }
        "price_moving_average_ratio_percentile_kernel" => {
            crate::load_cuda_embedded_module!("price_moving_average_ratio_percentile_kernel")?
        }
        "range_breakout_signals_kernel" => {
            crate::load_cuda_embedded_module!("range_breakout_signals_kernel")?
        }
        "range_filtered_trend_signals_kernel" => {
            crate::load_cuda_embedded_module!("range_filtered_trend_signals_kernel")?
        }
        "regression_slope_oscillator_kernel" => {
            crate::load_cuda_embedded_module!("regression_slope_oscillator_kernel")?
        }
        "relative_strength_index_wave_indicator_kernel" => {
            crate::load_cuda_embedded_module!("relative_strength_index_wave_indicator_kernel")?
        }
        // ---------------------------------------------------- closer 1, round 3
        "fibonacci_entry_bands_kernel" => {
            crate::load_cuda_embedded_module!("fibonacci_entry_bands_kernel")?
        }
        "goertzel_cycle_composite_wave_kernel" => {
            crate::load_cuda_embedded_module!("goertzel_cycle_composite_wave_kernel")?
        }
        "half_causal_estimator_kernel" => {
            crate::load_cuda_embedded_module!("half_causal_estimator_kernel")?
        }
        "ichimoku_oscillator_kernel" => {
            crate::load_cuda_embedded_module!("ichimoku_oscillator_kernel")?
        }
        "insync_index_kernel" => {
            crate::load_cuda_embedded_module!("insync_index_kernel")?
        }
        "linear_regression_intensity_kernel" => {
            crate::load_cuda_embedded_module!("linear_regression_intensity_kernel")?
        }
        "macd_wave_signal_pro_kernel" => {
            crate::load_cuda_embedded_module!("macd_wave_signal_pro_kernel")?
        }
        "mesa_stochastic_multi_length_kernel" => {
            crate::load_cuda_embedded_module!("mesa_stochastic_multi_length_kernel")?
        }
        "moving_average_cross_probability_kernel" => {
            crate::load_cuda_embedded_module!("moving_average_cross_probability_kernel")?
        }
        "multi_length_stochastic_average_kernel" => {
            crate::load_cuda_embedded_module!("multi_length_stochastic_average_kernel")?
        }
        other => {
            return Err(CudaF64IndicatorError::InvalidInput {
                indicator: "<module>",
                reason: format!(
                    "no embedded fatbin arm for f64 module '{other}'. Add it to                      `load_f64_module` and to `build.rs`; this lane will not guess a module                      name or fall back to another one."
                ),
            })
        }
    };
    Ok(m)
}

/// Which device series a launch needs. Built by the caller from ONE resident
/// upload, so no indicator here forces a device→host→device round trip.
#[derive(Debug, Clone, Copy)]
pub enum F64Inputs {
    /// A single price series — for `cci`/`mfi` this must be the CPU's source
    /// (hlc3), not close.
    Prices(CudaDeviceSliceF64Ref),
    /// high, low, close.
    Hlc {
        high: CudaDeviceSliceF64Ref,
        low: CudaDeviceSliceF64Ref,
        close: CudaDeviceSliceF64Ref,
    },
    /// A price series and volume. The price is hlc3 for `mfi`, close for
    /// `obv` / `vwma` / `efi` — which series is correct is declared per
    /// indicator by `F64InputKind`, never guessed here.
    PriceVolume {
        price: CudaDeviceSliceF64Ref,
        volume: CudaDeviceSliceF64Ref,
    },
    /// high and low only. `medprice` and `midprice` never read close, and
    /// their CPU first-valid rule scans high and low only — handing them an
    /// Ohlc ref would silently adopt close's first-valid and shift the series
    /// on any frame where close is the late series.
    HighLow {
        high: CudaDeviceSliceF64Ref,
        low: CudaDeviceSliceF64Ref,
    },
    /// Timestamps, price and volume — `vwap`, whose anchor is a calendar
    /// bucket rather than a rolling window. The only non-f64 array in this
    /// lane; it is `i64` because that is what the bar timestamps are, and
    /// narrowing it would move session boundaries.
    TimestampPriceVolume {
        timestamps: CudaDeviceSliceI64Ref,
        price: CudaDeviceSliceF64Ref,
        volume: CudaDeviceSliceF64Ref,
    },
    /// high, low and VOLUME, with NO close -- `emv` alone. Its CPU reference
    /// destructures close as `_close` and its first-valid scan covers high, low
    /// and volume only (emv.rs:196, :219), so passing an Hlc triple would both
    /// compute a different series and adopt a different warmup. A separate
    /// shape rather than a convention, so the wrong pairing is a mismatch in
    /// the table and not a plausible-looking number.
    HighLowVolume {
        high: CudaDeviceSliceF64Ref,
        low: CudaDeviceSliceF64Ref,
        volume: CudaDeviceSliceF64Ref,
    },
    /// open, high, low and close. `aso` alone in shard 3: its per-bar value
    /// reads `open[i]` AND `open[window_start]`, so open is an INPUT, not
    /// metadata, and an Hlc ref cannot serve it.
    Ohlc4 {
        open: CudaDeviceSliceF64Ref,
        high: CudaDeviceSliceF64Ref,
        low: CudaDeviceSliceF64Ref,
        close: CudaDeviceSliceF64Ref,
    },
    /// (open, close, volume) -- `dvdiqqe` alone. Its CPU reference takes high
    /// and low as `_high` and `_low` and NEVER reads them (dvdiqqe.rs:447-448),
    /// while `open` is read at every bar for the tick-range term. A separate
    /// shape rather than reusing Ohlcv, so a kernel that needs `open` cannot be
    /// handed `high` by a launch arm that happens to have four pointers.
    OpenCloseVolume {
        open: CudaDeviceSliceF64Ref,
        close: CudaDeviceSliceF64Ref,
        volume: CudaDeviceSliceF64Ref,
    },
    /// high, low, close AND volume -- `kvo` alone, whose trend state machine
    /// reads `h + l + c` and whose volume force reads volume.
    Hlcv {
        high: CudaDeviceSliceF64Ref,
        low: CudaDeviceSliceF64Ref,
        close: CudaDeviceSliceF64Ref,
        volume: CudaDeviceSliceF64Ref,
    },
    /// open, high, low, close AND volume -- the FULL bar. `trend_flow_trail`
    /// (closer 5, round 2) and every other indicator whose CPU batch calls
    /// `extract_ohlcv_full_input`.
    ///
    /// Deliberately NOT folded into [`Self::Hlcv`]: trend_flow_trail reads
    /// OPEN only in its validity gate (trend_flow_trail.rs:506-516), and a
    /// bar with a non-finite open and finite high/low/close/volume RESETS the
    /// whole cascade on the CPU. A kernel handed the four-pointer shape would
    /// never see that bar and would carry straight through the reset -- a
    /// divergence that no length or shape check on the way through would
    /// catch.
    Ohlcv5 {
        open: CudaDeviceSliceF64Ref,
        high: CudaDeviceSliceF64Ref,
        low: CudaDeviceSliceF64Ref,
        close: CudaDeviceSliceF64Ref,
        volume: CudaDeviceSliceF64Ref,
    },
}

impl F64Inputs {
    fn len(&self) -> usize {
        match self {
            F64Inputs::Prices(p) => p.len(),
            F64Inputs::Hlc { close, .. } => close.len(),
            F64Inputs::PriceVolume { price, .. } => price.len(),
            F64Inputs::HighLow { high, .. } => high.len(),
            F64Inputs::TimestampPriceVolume { price, .. } => price.len(),
            F64Inputs::HighLowVolume { high, .. } => high.len(),
            F64Inputs::Ohlc4 { close, .. } => close.len(),
            F64Inputs::Hlcv { close, .. } => close.len(),
            F64Inputs::Ohlcv5 { close, .. } => close.len(),
            F64Inputs::OpenCloseVolume { close, .. } => close.len(),
        }
    }

    fn device_id(&self) -> u32 {
        match self {
            F64Inputs::Prices(p) => p.device_id(),
            F64Inputs::Hlc { close, .. } => close.device_id(),
            F64Inputs::PriceVolume { price, .. } => price.device_id(),
            F64Inputs::HighLow { high, .. } => high.device_id(),
            F64Inputs::TimestampPriceVolume { price, .. } => price.device_id(),
            F64Inputs::HighLowVolume { high, .. } => high.device_id(),
            F64Inputs::Ohlc4 { close, .. } => close.device_id(),
            F64Inputs::Hlcv { close, .. } => close.device_id(),
            F64Inputs::Ohlcv5 { close, .. } => close.device_id(),
            F64Inputs::OpenCloseVolume { close, .. } => close.device_id(),
        }
    }
}

/// A completed sweep: `rows` period-series of `cols` bars, row-major, resident
/// on the device.
pub struct F64SweepResult {
    pub buf: DeviceBuffer<f64>,
    pub rows: usize,
    pub cols: usize,
    pub device_id: u32,
}

impl F64SweepResult {
    #[inline]
    pub fn len(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Copy the whole matrix back. Only for the parity oracle and for callers
    /// that genuinely need host values — the point of the lane is to keep
    /// results device-resident between indicators.
    pub fn to_host(&self) -> Result<Vec<f64>, CudaF64IndicatorError> {
        let mut host = vec![0.0f64; self.len()];
        self.buf.copy_to(host.as_mut_slice())?;
        Ok(host)
    }
}

/// How many rows fit alongside the scratch this kernel needs.
///
/// NEVER-OOM: this is derived from `mem_get_info`, never from the caller's
/// period count. A 200-entry period list on a small card runs in more chunks;
/// it does not allocate more.
fn rows_per_chunk(
    indicator: &'static str,
    cols: usize,
    scratch_matrices: usize,
    headroom: usize,
) -> Result<usize, CudaF64IndicatorError> {
    let (free, _total) = mem_get_info()?;
    let budget = free.saturating_sub(headroom);
    // 1 output matrix + `scratch_matrices` extra of the same shape.
    let per_row = cols
        .checked_mul(std::mem::size_of::<f64>())
        .and_then(|b| b.checked_mul(1 + scratch_matrices))
        .ok_or_else(|| CudaF64IndicatorError::InvalidInput {
            indicator,
            reason: "row byte count overflow".into(),
        })?;
    if per_row == 0 {
        return Ok(1);
    }
    let rows = budget / per_row;
    if rows == 0 {
        return Err(CudaF64IndicatorError::OutOfMemory {
            indicator,
            required: per_row,
            free,
            headroom,
        });
    }
    Ok(rows)
}

/// The f64 indicator engine.
pub struct CudaF64Indicators {
    /// `neoethos_f64_kernels`, the module most variants live in.
    module: Module,
    /// Per-indicator modules, for the variants whose kernel was fixed in the
    /// file it already shipped in. Loaded once at construction from the
    /// distinct set of [`F64Kernel::module_stem`] values, so a frame still
    /// pays zero module loads. Never populated lazily: a missing module is a
    /// construction-time error, not a surprise on the hot path.
    extra_modules: Vec<(&'static str, Module)>,
    stream: Stream,
    context: Arc<Context>,
    device_id: u32,
}

impl CudaF64Indicators {
    pub fn new(device_id: usize) -> Result<Self, CudaF64IndicatorError> {
        cust::init(CudaFlags::empty())?;
        let device = Device::get_device(device_id as u32)?;
        let context = Arc::new(Context::new(device)?);
        // Architecture-agnostic: `load_cuda_embedded_module!` reaches for the
        // multi-arch fatbin first and the lowest-arch PTX second. No card is
        // named here or in the macro.
        let module = crate::load_cuda_embedded_module!("neoethos_f64_kernels")?;

        // Every OTHER module some variant names, loaded once, deduplicated.
        // Driven by `F64Kernel::ALL` so a variant added without a module arm
        // fails here — loudly, at construction — rather than at its first
        // launch on a machine that has a card.
        let mut extra_modules: Vec<(&'static str, Module)> = Vec::new();
        for k in F64Kernel::ALL {
            let stem = k.module_stem();
            if stem == NEOETHOS_F64_MODULE {
                continue;
            }
            if extra_modules.iter().any(|(s, _)| *s == stem) {
                continue;
            }
            extra_modules.push((stem, load_f64_module(stem)?));
        }

        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            module,
            extra_modules,
            stream,
            context,
            device_id: device_id as u32,
        })
    }

    #[inline]
    pub fn context_arc(&self) -> Arc<Context> {
        self.context.clone()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    pub fn synchronize(&self) -> Result<(), CudaF64IndicatorError> {
        self.stream.synchronize()?;
        Ok(())
    }

    /// The module holding `kernel`'s entry point.
    ///
    /// There is no fallback arm: a variant whose module was not loaded is an
    /// `Err` naming it. Reaching into `self.module` "just in case the symbol is
    /// also there" is how an f64 request would silently land on a kernel
    /// nobody checked.
    fn module_for(&self, kernel: F64Kernel) -> Result<&Module, CudaF64IndicatorError> {
        let stem = kernel.module_stem();
        if stem == NEOETHOS_F64_MODULE {
            return Ok(&self.module);
        }
        self.extra_modules
            .iter()
            .find(|(s, _)| *s == stem)
            .map(|(_, m)| m)
            .ok_or(CudaF64IndicatorError::MissingKernelSymbol {
                name: kernel.entry_point(),
            })
    }

    fn validate_launch(&self, grid: GridSize, block: BlockSize) -> Result<(), CudaF64IndicatorError> {
        let device = Device::get_device(self.device_id)?;
        let max_grid_x = device.get_attribute(DeviceAttribute::MaxGridDimX)? as u32;
        let max_grid_y = device.get_attribute(DeviceAttribute::MaxGridDimY)? as u32;
        let max_block_x = device.get_attribute(DeviceAttribute::MaxBlockDimX)? as u32;
        if grid.x > max_grid_x || grid.y > max_grid_y || block.x > max_block_x {
            return Err(CudaF64IndicatorError::LaunchConfigTooLarge {
                gx: grid.x,
                gy: grid.y,
                gz: grid.z,
                bx: block.x,
                by: block.y,
                bz: block.z,
            });
        }
        Ok(())
    }

    /// Sweep one indicator over an explicit period list.
    ///
    /// `first_valid` MUST be computed on the host with the same rule the
    /// corresponding CPU `*_prepare` uses — the first index whose inputs are
    /// all non-NaN — because the warmup prefix and the seed window both hang
    /// off it. Passing a different value produces a shifted series, which the
    /// parity test reads as "different windows", not as "rounding".
    pub fn sweep(
        &self,
        kernel: F64Kernel,
        inputs: F64Inputs,
        periods: &[i32],
        first_valid: usize,
    ) -> Result<F64SweepResult, CudaF64IndicatorError> {
        let indicator = kernel.indicator_id();
        let cols = inputs.len();
        let rows = periods.len();

        if cols == 0 {
            return Err(CudaF64IndicatorError::InvalidInput {
                indicator,
                reason: "empty input series".into(),
            });
        }
        if rows == 0 {
            return Err(CudaF64IndicatorError::InvalidInput {
                indicator,
                reason: "empty period list".into(),
            });
        }
        if inputs.device_id() != self.device_id {
            return Err(CudaF64IndicatorError::InvalidInput {
                indicator,
                reason: format!(
                    "inputs are resident on device {} but this engine is bound to device {}",
                    inputs.device_id(),
                    self.device_id
                ),
            });
        }
        if first_valid >= cols {
            return Err(CudaF64IndicatorError::InvalidInput {
                indicator,
                reason: format!("first_valid={first_valid} is not inside a {cols}-bar series"),
            });
        }
        if let Some(bad) = periods.iter().find(|p| **p <= 0) {
            return Err(CudaF64IndicatorError::InvalidInput {
                indicator,
                reason: format!("period {bad} is not >= 1"),
            });
        }
        // Kernels that keep a fixed per-thread ring refuse an oversized period
        // BY NAME. Truncating the window would compute a different indicator
        // and moving the sweep to the host would be the silent fallback this
        // lane exists to remove.
        if let Some(max) = kernel.max_period() {
            if let Some(too_big) = periods.iter().find(|p| **p as usize > max) {
                return Err(CudaF64IndicatorError::PeriodTooLarge {
                    indicator,
                    period: *too_big as usize,
                    max,
                });
            }
        }

        // cci needs one extra matrix of the same shape for the sequential
        // running-mean pass; everything else needs none.
        let scratch = usize::from(kernel == F64Kernel::Cci);

        let output_elems =
            rows.checked_mul(cols)
                .ok_or_else(|| CudaF64IndicatorError::InvalidInput {
                    indicator,
                    reason: "rows*cols overflow".into(),
                })?;
        let out = unsafe { DeviceBuffer::<f64>::uninitialized(output_elems)? };

        // Chunk over ROWS so peak transient memory tracks the card, not the
        // period list. The output itself is the caller's contract and is
        // allocated whole; the scratch is what gets chunked.
        let chunk = rows_per_chunk(indicator, cols, scratch, DEFAULT_HEADROOM)?.min(rows);

        let mut row0 = 0usize;
        while row0 < rows {
            let nrows = (rows - row0).min(chunk);
            let d_periods = DeviceBuffer::from_slice(&periods[row0..row0 + nrows])?;
            // Row-offset view into the whole output buffer. `as_device_ptr()`
            // plus an element offset avoids a second allocation and a copy.
            let out_ptr = unsafe { out.as_device_ptr().offset((row0 * cols) as isize) };

            self.launch_chunk(
                kernel,
                inputs,
                &d_periods,
                nrows,
                cols,
                first_valid,
                out_ptr,
                indicator,
            )?;

            // MUST synchronize before `d_periods` drops at the end of this
            // iteration. Launches are asynchronous on the stream, so freeing
            // the period buffer while a kernel may still be reading it is a
            // use-after-free on the device — the kind that corrupts a
            // neighbouring allocation and surfaces as a wrong number three
            // indicators later rather than as a crash here.
            self.stream.synchronize()?;

            row0 += nrows;
        }


        Ok(F64SweepResult {
            buf: out,
            rows,
            cols,
            device_id: self.device_id,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_chunk(
        &self,
        kernel: F64Kernel,
        inputs: F64Inputs,
        d_periods: &DeviceBuffer<i32>,
        rows: usize,
        cols: usize,
        first_valid: usize,
        out_ptr: cust::memory::DevicePointer<f64>,
        indicator: &'static str,
    ) -> Result<(), CudaF64IndicatorError> {
        let name = kernel.entry_point();
        // The module this VARIANT declares, not "the" module — see
        // `F64Kernel::module_stem`.
        let kmod = self.module_for(kernel)?;
        let func = kmod
            .get_function(name)
            .map_err(|_| CudaF64IndicatorError::MissingKernelSymbol { name })?;
        let stream = &self.stream;
        let n = cols as i32;
        let n_combos = rows as i32;
        let fv = first_valid as i32;

        if kernel.is_sequential() && kernel != F64Kernel::Cci {
            // One thread per column, walking bars in CPU order.
            let grid = GridSize::x(((rows as u32) + BLOCK_X - 1) / BLOCK_X);
            let block = BlockSize::x(BLOCK_X);
            self.validate_launch(grid, block)?;

            match inputs {
                F64Inputs::Prices(p) => unsafe {
                    let prices = cust::memory::DevicePointer::<f64>::from_raw(p.device_ptr());
                    launch!(func<<<grid, block, 0, stream>>>(
                        prices,
                        n,
                        d_periods.as_device_ptr(),
                        n_combos,
                        fv,
                        out_ptr
                    ))?;
                },
                F64Inputs::Hlc { high, low, close } => unsafe {
                    let h = cust::memory::DevicePointer::<f64>::from_raw(high.device_ptr());
                    let l = cust::memory::DevicePointer::<f64>::from_raw(low.device_ptr());
                    let c = cust::memory::DevicePointer::<f64>::from_raw(close.device_ptr());
                    launch!(func<<<grid, block, 0, stream>>>(
                        h,
                        l,
                        c,
                        n,
                        d_periods.as_device_ptr(),
                        n_combos,
                        fv,
                        out_ptr
                    ))?;
                },
                // `HighLow` shares this arm because the kernel signature is the
                // same shape — two `const double*` in declaration order. Which
                // two series they are is settled upstream by `F64InputKind`,
                // not here.
                F64Inputs::PriceVolume {
                    price: a,
                    volume: b,
                }
                | F64Inputs::HighLow { high: a, low: b } => unsafe {
                    let p = cust::memory::DevicePointer::<f64>::from_raw(a.device_ptr());
                    let v = cust::memory::DevicePointer::<f64>::from_raw(b.device_ptr());
                    launch!(func<<<grid, block, 0, stream>>>(
                        p,
                        v,
                        n,
                        d_periods.as_device_ptr(),
                        n_combos,
                        fv,
                        out_ptr
                    ))?;
                },
                F64Inputs::TimestampPriceVolume {
                    timestamps,
                    price,
                    volume,
                } => unsafe {
                    let t = cust::memory::DevicePointer::<i64>::from_raw(timestamps.device_ptr());
                    let p = cust::memory::DevicePointer::<f64>::from_raw(price.device_ptr());
                    let v = cust::memory::DevicePointer::<f64>::from_raw(volume.device_ptr());
                    launch!(func<<<grid, block, 0, stream>>>(
                        t,
                        p,
                        v,
                        n,
                        d_periods.as_device_ptr(),
                        n_combos,
                        fv,
                        out_ptr
                    ))?;
                },
                // shard 1: three `const double*` in declaration order (high,
                // low, volume). Deliberately NOT folded into the `PriceVolume |
                // HighLow` arm above: those pass two pointers, and a kernel
                // reading a third would read whatever followed on the stack.
                F64Inputs::HighLowVolume { high, low, volume } => unsafe {
                    let h = cust::memory::DevicePointer::<f64>::from_raw(high.device_ptr());
                    let l = cust::memory::DevicePointer::<f64>::from_raw(low.device_ptr());
                    let v = cust::memory::DevicePointer::<f64>::from_raw(volume.device_ptr());
                    launch!(func<<<grid, block, 0, stream>>>(
                        h,
                        l,
                        v,
                        n,
                        d_periods.as_device_ptr(),
                        n_combos,
                        fv,
                        out_ptr
                    ))?;
                },
                // shard 1: four `const double*` (high, low, close, volume).
                F64Inputs::Hlcv {
                    high,
                    low,
                    close,
                    volume,
                } => unsafe {
                    let h = cust::memory::DevicePointer::<f64>::from_raw(high.device_ptr());
                    let l = cust::memory::DevicePointer::<f64>::from_raw(low.device_ptr());
                    let c = cust::memory::DevicePointer::<f64>::from_raw(close.device_ptr());
                    let v = cust::memory::DevicePointer::<f64>::from_raw(volume.device_ptr());
                    launch!(func<<<grid, block, 0, stream>>>(
                        h,
                        l,
                        c,
                        v,
                        n,
                        d_periods.as_device_ptr(),
                        n_combos,
                        fv,
                        out_ptr
                    ))?;
                },
                // shard 4: three `const double*` in the order (open, close,
                // volume). NOT the Hlc arm's order -- `dvdiqqe` reads open
                // where that arm passes high.
                F64Inputs::OpenCloseVolume {
                    open,
                    close,
                    volume,
                } => unsafe {
                    let op = cust::memory::DevicePointer::<f64>::from_raw(open.device_ptr());
                    let c = cust::memory::DevicePointer::<f64>::from_raw(close.device_ptr());
                    let v = cust::memory::DevicePointer::<f64>::from_raw(volume.device_ptr());
                    launch!(func<<<grid, block, 0, stream>>>(
                        op,
                        c,
                        v,
                        n,
                        d_periods.as_device_ptr(),
                        n_combos,
                        fv,
                        out_ptr
                    ))?;
                },
                // shard 3: four `const double*` (open, high, low, close).
                // `aso` alone. Deliberately NOT folded into the `Hlc` arm: that
                // one passes three pointers, and a kernel declaring four would
                // read whatever followed them.
                // closer 5, round 2: five `const double*` in the order
                // (open, high, low, close, volume). Deliberately NOT folded
                // into the `Hlcv` arm above: that one passes four pointers,
                // and a kernel declaring five would read whatever followed
                // them.
                F64Inputs::Ohlcv5 {
                    open,
                    high,
                    low,
                    close,
                    volume,
                } => unsafe {
                    let o5 = cust::memory::DevicePointer::<f64>::from_raw(open.device_ptr());
                    let h = cust::memory::DevicePointer::<f64>::from_raw(high.device_ptr());
                    let l = cust::memory::DevicePointer::<f64>::from_raw(low.device_ptr());
                    let c = cust::memory::DevicePointer::<f64>::from_raw(close.device_ptr());
                    let v = cust::memory::DevicePointer::<f64>::from_raw(volume.device_ptr());
                    launch!(func<<<grid, block, 0, stream>>>(
                        o5,
                        h,
                        l,
                        c,
                        v,
                        n,
                        d_periods.as_device_ptr(),
                        n_combos,
                        fv,
                        out_ptr
                    ))?;
                },
                F64Inputs::Ohlc4 {
                    open,
                    high,
                    low,
                    close,
                } => unsafe {
                    let o = cust::memory::DevicePointer::<f64>::from_raw(open.device_ptr());
                    let h = cust::memory::DevicePointer::<f64>::from_raw(high.device_ptr());
                    let l = cust::memory::DevicePointer::<f64>::from_raw(low.device_ptr());
                    let c = cust::memory::DevicePointer::<f64>::from_raw(close.device_ptr());
                    launch!(func<<<grid, block, 0, stream>>>(
                        o,
                        h,
                        l,
                        c,
                        n,
                        d_periods.as_device_ptr(),
                        n_combos,
                        fv,
                        out_ptr
                    ))?;
                },
            }
            return Ok(());
        }

        if kernel == F64Kernel::Cci {
            // Pass 1: sequential running mean into scratch. Pass 2: parallel
            // mean absolute deviation. Splitting keeps the numerically fragile
            // accumulation in CPU order and the O(period) work wide.
            let price = match inputs {
                F64Inputs::Prices(p) => p,
                _ => {
                    return Err(CudaF64IndicatorError::InvalidInput {
                        indicator,
                        reason: "cci needs a single price series (the CPU source, hlc3 by default)"
                            .into(),
                    })
                }
            };
            let scratch = unsafe { DeviceBuffer::<f64>::uninitialized(rows * cols)? };

            let sma_name = "neoethos_cci_sma_f64";
            let sma_func = self
                .module
                .get_function(sma_name)
                .map_err(|_| CudaF64IndicatorError::MissingKernelSymbol { name: sma_name })?;
            let grid1 = GridSize::x(((rows as u32) + BLOCK_X - 1) / BLOCK_X);
            let block1 = BlockSize::x(BLOCK_X);
            self.validate_launch(grid1, block1)?;
            unsafe {
                let p = cust::memory::DevicePointer::<f64>::from_raw(price.device_ptr());
                launch!(sma_func<<<grid1, block1, 0, stream>>>(
                    p,
                    n,
                    d_periods.as_device_ptr(),
                    n_combos,
                    fv,
                    scratch.as_device_ptr()
                ))?;
            }

            let grid2 = GridSize::xy(
                ((cols as u32) + BAR_BLOCK_X - 1) / BAR_BLOCK_X,
                rows as u32,
            );
            let block2 = BlockSize::x(BAR_BLOCK_X);
            self.validate_launch(grid2, block2)?;
            unsafe {
                let p = cust::memory::DevicePointer::<f64>::from_raw(price.device_ptr());
                launch!(func<<<grid2, block2, 0, stream>>>(
                    p,
                    scratch.as_device_ptr(),
                    n,
                    d_periods.as_device_ptr(),
                    n_combos,
                    fv,
                    out_ptr
                ))?;
            }
            // The scratch must outlive the launch; synchronising here is the
            // simple, obviously-correct way to guarantee that.
            self.stream.synchronize()?;
            return Ok(());
        }

        // Parallel over (combo, bar): roc, mom, willr.
        let grid = GridSize::xy(
            ((cols as u32) + BAR_BLOCK_X - 1) / BAR_BLOCK_X,
            rows as u32,
        );
        let block = BlockSize::x(BAR_BLOCK_X);
        self.validate_launch(grid, block)?;

        match inputs {
            F64Inputs::Prices(p) => unsafe {
                let prices = cust::memory::DevicePointer::<f64>::from_raw(p.device_ptr());
                launch!(func<<<grid, block, 0, stream>>>(
                    prices,
                    n,
                    d_periods.as_device_ptr(),
                    n_combos,
                    fv,
                    out_ptr
                ))?;
            },
            F64Inputs::Hlc { high, low, close } => unsafe {
                let h = cust::memory::DevicePointer::<f64>::from_raw(high.device_ptr());
                let l = cust::memory::DevicePointer::<f64>::from_raw(low.device_ptr());
                let c = cust::memory::DevicePointer::<f64>::from_raw(close.device_ptr());
                launch!(func<<<grid, block, 0, stream>>>(
                    h,
                    l,
                    c,
                    n,
                    d_periods.as_device_ptr(),
                    n_combos,
                    fv,
                    out_ptr
                ))?;
            },
            F64Inputs::PriceVolume {
                price: a,
                volume: b,
            }
            | F64Inputs::HighLow { high: a, low: b } => unsafe {
                let p = cust::memory::DevicePointer::<f64>::from_raw(a.device_ptr());
                let v = cust::memory::DevicePointer::<f64>::from_raw(b.device_ptr());
                launch!(func<<<grid, block, 0, stream>>>(
                    p,
                    v,
                    n,
                    d_periods.as_device_ptr(),
                    n_combos,
                    fv,
                    out_ptr
                ))?;
            },
            // No parallel kernel takes timestamps: `vwap` is the only
            // timestamped indicator here and its bucket accumulators carry
            // across bars, so it is sequential. Refuse rather than launch a
            // kernel with a signature that does not match the arguments.
            F64Inputs::TimestampPriceVolume { .. } => {
                return Err(CudaF64IndicatorError::InvalidInput {
                    indicator,
                    reason: "timestamped inputs are only accepted by the sequential lane".into(),
                })
            }
            // shard 1: every indicator that asks for these two shapes carries
            // state across bars, so none of them has a bar-parallel kernel.
            // Refused by name rather than launched against a signature nobody
            // wrote -- a launch with the wrong argument count reads garbage and
            // returns plausible numbers.
            // shard 4: dvdiqqe carries PVI/NVI and four EMAs across bars.
            F64Inputs::OpenCloseVolume { .. } => {
                return Err(CudaF64IndicatorError::InvalidInput {
                    indicator,
                    reason: "(open, close, volume) inputs are only accepted by the sequential lane"
                        .into(),
                })
            }
            // closer 5, round 2: every indicator that asks for the full bar
            // carries state across bars, so none of them has a bar-parallel
            // kernel. Refused by name rather than launched against a signature
            // nobody wrote.
            F64Inputs::Ohlcv5 { .. } => {
                return Err(CudaF64IndicatorError::InvalidInput {
                    indicator,
                    reason: "(open, high, low, close, volume) inputs are only accepted by the \
                             sequential lane"
                        .into(),
                })
            }
            F64Inputs::HighLowVolume { .. } | F64Inputs::Hlcv { .. } => {
                return Err(CudaF64IndicatorError::InvalidInput {
                    indicator,
                    reason: "(high, low, volume) and (high, low, close, volume) inputs are only \
                             accepted by the sequential lane"
                        .into(),
                })
            }
            // shard 3: same reason. `aso` carries a running mean across bars.
            F64Inputs::Ohlc4 { .. } => {
                return Err(CudaF64IndicatorError::InvalidInput {
                    indicator,
                    reason: "(open, high, low, close) inputs are only accepted by the sequential lane"
                        .into(),
                })
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every variant resolves to a distinct `*_f64` entry point, and none of
    /// them can be mistaken for an f32 symbol. Runs without a card.
    #[test]
    fn entry_points_are_f64_and_unique() {
        // Drives off `F64Kernel::ALL` rather than a literal list. The literal
        // list this replaced covered ten variants and would have kept passing,
        // silently, while the enum grew to twenty-nine — a test that shrinks
        // relative to what it guards is worse than no test.
        let all = F64Kernel::ALL;
        let mut names: Vec<&str> = all.iter().map(|k| k.entry_point()).collect();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), before, "two variants share an entry point");
        for &k in all {
            assert!(
                k.entry_point().ends_with("_f64"),
                "{}: entry point {} is not an f64 symbol",
                k.indicator_id(),
                k.entry_point()
            );
            assert!(
                !k.entry_point().contains("_f32"),
                "{}: entry point names an f32 symbol",
                k.indicator_id()
            );
        }
    }

    /// Indicator ids must be unique too, and every variant must be in `ALL`.
    ///
    /// `ALL` is hand-written, so the risk it carries is a variant added to the
    /// enum and forgotten here — which would make `entry_points_are_f64_and_
    /// unique` quietly stop covering it. The id count is the tripwire: it must
    /// equal the number of `indicator_id` match arms, and that match is
    /// exhaustive by the compiler.
    #[test]
    fn all_is_complete_and_ids_are_unique() {
        let mut ids: Vec<&str> = F64Kernel::ALL.iter().map(|k| k.indicator_id()).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "two variants share an indicator id");
        assert_eq!(
            before, 338,
            "F64Kernel::ALL has {before} entries. If a variant was added to the enum, add it here \
             too — otherwise entry_points_are_f64_and_unique silently stops covering it."
        );
    }

    /// NeoEthos sweeps TSI by using the requested period as `long_period` and
    /// scaling `short_period` with the documented 25:13 default ratio.  Calling
    /// this row invariant makes the CUDA kernel discard that production
    /// contract and compute the default 25/13 series for every requested row.
    #[test]
    fn tsi_coupled_window_is_not_period_invariant() {
        assert!(
            !F64Kernel::Tsi.is_period_invariant(),
            "TSI must consume the requested long-period anchor; treating it as invariant makes \
             CUDA disagree with the named long_period/short_period CPU request"
        );
    }

    /// Each kernel that carries a fixed per-thread ring must state its bound,
    /// and the host bound must equal the `#define` in the .cu. A period between
    /// the two would overrun a per-thread array on the device.
    #[test]
    fn ring_bounds_are_stated_once() {
        assert_eq!(MFI_MAX_PERIOD, 512);
        assert_eq!(ADXR_MAX_PERIOD, 512);
        assert_eq!(F64Kernel::Mfi.max_period(), Some(MFI_MAX_PERIOD));
        assert_eq!(F64Kernel::Adxr.max_period(), Some(ADXR_MAX_PERIOD));
        assert_eq!(S2_RING_MAX_PERIOD, 512);
        // shard 1's four rings. Each is the length of a fixed per-thread array
        // in the indicator's own .cu, so the host constant and the `#define`
        // must agree exactly -- a period between the two would overrun.
        assert_eq!(CHOP_MAX_PERIOD, 1024);
        assert_eq!(HMA_MAX_PERIOD, 4095);
        assert_eq!(EDCF_MAX_PERIOD, 512);
        assert_eq!(ALMA_MAX_PERIOD, 1024);
        assert_eq!(F64Kernel::Chop.max_period(), Some(CHOP_MAX_PERIOD));
        assert_eq!(F64Kernel::Hma.max_period(), Some(HMA_MAX_PERIOD));
        assert_eq!(F64Kernel::Edcf.max_period(), Some(EDCF_MAX_PERIOD));
        assert_eq!(F64Kernel::Alma.max_period(), Some(ALMA_MAX_PERIOD));
        // shard 4's only ring: ehma must materialise the Hann weights because
        // the CPU reverses them and the rotation is not bit-reproducible
        // backwards.
        assert_eq!(EHMA_MAX_PERIOD, 512);
        assert_eq!(F64Kernel::Ehma.max_period(), Some(EHMA_MAX_PERIOD));
        // closer 6, round 3: two weight vectors built once per thread rather
        // than per bar -- `logarithmic_moving_average`'s logarithmic weights
        // and `wave_smoother`'s sin/cos wave weights. The .cu `#define` and the
        // constant here must agree exactly; `wave_smoother`'s array is
        // `WS_MAX_PERIOD + 1` long because its window is `period + 1`.
        assert_eq!(LMA_MAX_PERIOD, 512);
        assert_eq!(WS_MAX_PERIOD, 512);
        assert_eq!(
            F64Kernel::LogarithmicMovingAverage.max_period(),
            Some(LMA_MAX_PERIOD)
        );
        assert_eq!(F64Kernel::WaveSmoother.max_period(), Some(WS_MAX_PERIOD));
        // closer 4, round 3: cora_wave's smoothing ring and dma's difference
        // ring are both `round(sqrt(period))` deep, so ONE 64-entry array
        // admits every period up to 4160 -- `round(sqrt(4160))` is 64 and
        // `round(sqrt(4161))` is 65. The two share the number for that reason
        // and not by coincidence.
        assert_eq!(CORA_WAVE_MAX_PERIOD, 4160);
        assert_eq!(DMA_MAX_PERIOD, 4160);
        assert_eq!(F64Kernel::CoraWave.max_period(), Some(CORA_WAVE_MAX_PERIOD));
        assert_eq!(F64Kernel::Dma.max_period(), Some(DMA_MAX_PERIOD));
        // Every declared bound must be one of the three constants above. A
        // kernel that invents its own number would compile and then overrun a
        // local array for periods between the two values.
        for &k in F64Kernel::ALL {
            if let Some(m) = k.max_period() {
                assert!(
                    m == MFI_MAX_PERIOD
                        || m == ADXR_MAX_PERIOD
                        || m == S2_RING_MAX_PERIOD
                        || m == CHOP_MAX_PERIOD
                        || m == HMA_MAX_PERIOD
                        || m == EDCF_MAX_PERIOD
                        || m == ALMA_MAX_PERIOD
                        || m == EHMA_MAX_PERIOD
                        || m == LMA_MAX_PERIOD
                        || m == WS_MAX_PERIOD
                        || m == CORA_WAVE_MAX_PERIOD
                        || m == DMA_MAX_PERIOD,
                    "{}: period bound {m} is not one of the stated ring constants",
                    k.indicator_id()
                );
            }
        }
    }
}
