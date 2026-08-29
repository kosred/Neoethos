use crate::utilities::data_loader::Candles;
use crate::utilities::enums::Kernel;
use crate::utilities::helpers::{alloc_uninit_f64, detect_best_batch_kernel, detect_best_kernel};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use thiserror::Error;

const DEFAULT_DEVIATIONS: f64 = 2.0;
const DEFAULT_SHORT_CYCLE: usize = 8;
const DEFAULT_LONG_CYCLE: usize = 65;
const DEFAULT_SENSITIVITY: f64 = 40.0;
const DEFAULT_ALL_PEAKS_MODE: bool = true;
const DEFAULT_LB_R: usize = 5;
const DEFAULT_LB_L: usize = 5;
const DEFAULT_RANGE_UPPER: usize = 60;
const DEFAULT_RANGE_LOWER: usize = 5;
const DEFAULT_PLOT_BULL: bool = true;
const DEFAULT_PLOT_HIDDEN_BULL: bool = false;
const DEFAULT_PLOT_BEAR: bool = true;
const DEFAULT_PLOT_HIDDEN_BEAR: bool = false;

type KpoTuple = (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);

#[derive(Debug, Clone)]
pub enum KasePeakOscillatorWithDivergencesData<'a> {
    Candles {
        candles: &'a Candles,
    },
    Slices {
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
    },
}

#[derive(Debug, Clone)]
pub struct KasePeakOscillatorWithDivergencesOutput {
    pub oscillator: Vec<f64>,
    pub histogram: Vec<f64>,
    pub max_peak_value: Vec<f64>,
    pub min_peak_value: Vec<f64>,
    pub market_extreme: Vec<f64>,
    pub regular_bullish: Vec<f64>,
    pub hidden_bullish: Vec<f64>,
    pub regular_bearish: Vec<f64>,
    pub hidden_bearish: Vec<f64>,
    pub go_long: Vec<f64>,
    pub go_short: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KasePeakOscillatorWithDivergencesParams {
    pub deviations: Option<f64>,
    pub short_cycle: Option<usize>,
    pub long_cycle: Option<usize>,
    pub sensitivity: Option<f64>,
    pub all_peaks_mode: Option<bool>,
    pub lb_r: Option<usize>,
    pub lb_l: Option<usize>,
    pub range_upper: Option<usize>,
    pub range_lower: Option<usize>,
    pub plot_bull: Option<bool>,
    pub plot_hidden_bull: Option<bool>,
    pub plot_bear: Option<bool>,
    pub plot_hidden_bear: Option<bool>,
}

impl Default for KasePeakOscillatorWithDivergencesParams {
    fn default() -> Self {
        Self {
            deviations: Some(DEFAULT_DEVIATIONS),
            short_cycle: Some(DEFAULT_SHORT_CYCLE),
            long_cycle: Some(DEFAULT_LONG_CYCLE),
            sensitivity: Some(DEFAULT_SENSITIVITY),
            all_peaks_mode: Some(DEFAULT_ALL_PEAKS_MODE),
            lb_r: Some(DEFAULT_LB_R),
            lb_l: Some(DEFAULT_LB_L),
            range_upper: Some(DEFAULT_RANGE_UPPER),
            range_lower: Some(DEFAULT_RANGE_LOWER),
            plot_bull: Some(DEFAULT_PLOT_BULL),
            plot_hidden_bull: Some(DEFAULT_PLOT_HIDDEN_BULL),
            plot_bear: Some(DEFAULT_PLOT_BEAR),
            plot_hidden_bear: Some(DEFAULT_PLOT_HIDDEN_BEAR),
        }
    }
}

#[derive(Debug, Clone)]
pub struct KasePeakOscillatorWithDivergencesInput<'a> {
    pub data: KasePeakOscillatorWithDivergencesData<'a>,
    pub params: KasePeakOscillatorWithDivergencesParams,
}

impl<'a> KasePeakOscillatorWithDivergencesInput<'a> {
    #[inline]
    pub fn from_candles(
        candles: &'a Candles,
        params: KasePeakOscillatorWithDivergencesParams,
    ) -> Self {
        Self {
            data: KasePeakOscillatorWithDivergencesData::Candles { candles },
            params,
        }
    }

    #[inline]
    pub fn from_slices(
        high: &'a [f64],
        low: &'a [f64],
        close: &'a [f64],
        params: KasePeakOscillatorWithDivergencesParams,
    ) -> Self {
        Self {
            data: KasePeakOscillatorWithDivergencesData::Slices { high, low, close },
            params,
        }
    }

    #[inline]
    pub fn with_default_candles(candles: &'a Candles) -> Self {
        Self::from_candles(candles, KasePeakOscillatorWithDivergencesParams::default())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct KasePeakOscillatorWithDivergencesBuilder {
    deviations: Option<f64>,
    short_cycle: Option<usize>,
    long_cycle: Option<usize>,
    sensitivity: Option<f64>,
    all_peaks_mode: Option<bool>,
    lb_r: Option<usize>,
    lb_l: Option<usize>,
    range_upper: Option<usize>,
    range_lower: Option<usize>,
    plot_bull: Option<bool>,
    plot_hidden_bull: Option<bool>,
    plot_bear: Option<bool>,
    plot_hidden_bear: Option<bool>,
    kernel: Kernel,
}

impl Default for KasePeakOscillatorWithDivergencesBuilder {
    fn default() -> Self {
        Self {
            deviations: None,
            short_cycle: None,
            long_cycle: None,
            sensitivity: None,
            all_peaks_mode: None,
            lb_r: None,
            lb_l: None,
            range_upper: None,
            range_lower: None,
            plot_bull: None,
            plot_hidden_bull: None,
            plot_bear: None,
            plot_hidden_bear: None,
            kernel: Kernel::Auto,
        }
    }
}

impl KasePeakOscillatorWithDivergencesBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn deviations(mut self, deviations: f64) -> Self {
        self.deviations = Some(deviations);
        self
    }

    #[inline]
    pub fn short_cycle(mut self, short_cycle: usize) -> Self {
        self.short_cycle = Some(short_cycle);
        self
    }

    #[inline]
    pub fn long_cycle(mut self, long_cycle: usize) -> Self {
        self.long_cycle = Some(long_cycle);
        self
    }

    #[inline]
    pub fn sensitivity(mut self, sensitivity: f64) -> Self {
        self.sensitivity = Some(sensitivity);
        self
    }

    #[inline]
    pub fn all_peaks_mode(mut self, all_peaks_mode: bool) -> Self {
        self.all_peaks_mode = Some(all_peaks_mode);
        self
    }

    #[inline]
    pub fn lb_r(mut self, lb_r: usize) -> Self {
        self.lb_r = Some(lb_r);
        self
    }

    #[inline]
    pub fn lb_l(mut self, lb_l: usize) -> Self {
        self.lb_l = Some(lb_l);
        self
    }

    #[inline]
    pub fn range_upper(mut self, range_upper: usize) -> Self {
        self.range_upper = Some(range_upper);
        self
    }

    #[inline]
    pub fn range_lower(mut self, range_lower: usize) -> Self {
        self.range_lower = Some(range_lower);
        self
    }

    #[inline]
    pub fn plot_bull(mut self, plot_bull: bool) -> Self {
        self.plot_bull = Some(plot_bull);
        self
    }

    #[inline]
    pub fn plot_hidden_bull(mut self, plot_hidden_bull: bool) -> Self {
        self.plot_hidden_bull = Some(plot_hidden_bull);
        self
    }

    #[inline]
    pub fn plot_bear(mut self, plot_bear: bool) -> Self {
        self.plot_bear = Some(plot_bear);
        self
    }

    #[inline]
    pub fn plot_hidden_bear(mut self, plot_hidden_bear: bool) -> Self {
        self.plot_hidden_bear = Some(plot_hidden_bear);
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply(
        self,
        candles: &Candles,
    ) -> Result<KasePeakOscillatorWithDivergencesOutput, KasePeakOscillatorWithDivergencesError>
    {
        let input = KasePeakOscillatorWithDivergencesInput::from_candles(
            candles,
            KasePeakOscillatorWithDivergencesParams {
                deviations: self.deviations,
                short_cycle: self.short_cycle,
                long_cycle: self.long_cycle,
                sensitivity: self.sensitivity,
                all_peaks_mode: self.all_peaks_mode,
                lb_r: self.lb_r,
                lb_l: self.lb_l,
                range_upper: self.range_upper,
                range_lower: self.range_lower,
                plot_bull: self.plot_bull,
                plot_hidden_bull: self.plot_hidden_bull,
                plot_bear: self.plot_bear,
                plot_hidden_bear: self.plot_hidden_bear,
            },
        );
        kase_peak_oscillator_with_divergences_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<KasePeakOscillatorWithDivergencesOutput, KasePeakOscillatorWithDivergencesError>
    {
        let input = KasePeakOscillatorWithDivergencesInput::from_slices(
            high,
            low,
            close,
            KasePeakOscillatorWithDivergencesParams {
                deviations: self.deviations,
                short_cycle: self.short_cycle,
                long_cycle: self.long_cycle,
                sensitivity: self.sensitivity,
                all_peaks_mode: self.all_peaks_mode,
                lb_r: self.lb_r,
                lb_l: self.lb_l,
                range_upper: self.range_upper,
                range_lower: self.range_lower,
                plot_bull: self.plot_bull,
                plot_hidden_bull: self.plot_hidden_bull,
                plot_bear: self.plot_bear,
                plot_hidden_bear: self.plot_hidden_bear,
            },
        );
        kase_peak_oscillator_with_divergences_with_kernel(&input, self.kernel)
    }

    #[inline]
    pub fn into_stream(
        self,
    ) -> Result<KasePeakOscillatorWithDivergencesStream, KasePeakOscillatorWithDivergencesError>
    {
        KasePeakOscillatorWithDivergencesStream::try_new(KasePeakOscillatorWithDivergencesParams {
            deviations: self.deviations,
            short_cycle: self.short_cycle,
            long_cycle: self.long_cycle,
            sensitivity: self.sensitivity,
            all_peaks_mode: self.all_peaks_mode,
            lb_r: self.lb_r,
            lb_l: self.lb_l,
            range_upper: self.range_upper,
            range_lower: self.range_lower,
            plot_bull: self.plot_bull,
            plot_hidden_bull: self.plot_hidden_bull,
            plot_bear: self.plot_bear,
            plot_hidden_bear: self.plot_hidden_bear,
        })
    }
}

#[derive(Debug, Error)]
pub enum KasePeakOscillatorWithDivergencesError {
    #[error("kase_peak_oscillator_with_divergences: Input data slice is empty.")]
    EmptyInputData,
    #[error("kase_peak_oscillator_with_divergences: All values are NaN.")]
    AllValuesNaN,
    #[error("kase_peak_oscillator_with_divergences: Invalid deviations: {deviations}")]
    InvalidDeviations { deviations: f64 },
    #[error(
        "kase_peak_oscillator_with_divergences: Invalid short_cycle: short_cycle = {short_cycle}, data length = {data_len}"
    )]
    InvalidShortCycle { short_cycle: usize, data_len: usize },
    #[error(
        "kase_peak_oscillator_with_divergences: Invalid long_cycle: long_cycle = {long_cycle}, data length = {data_len}"
    )]
    InvalidLongCycle { long_cycle: usize, data_len: usize },
    #[error(
        "kase_peak_oscillator_with_divergences: Invalid cycle order: short_cycle = {short_cycle}, long_cycle = {long_cycle}"
    )]
    InvalidCycleOrder {
        short_cycle: usize,
        long_cycle: usize,
    },
    #[error("kase_peak_oscillator_with_divergences: Invalid sensitivity: {sensitivity}")]
    InvalidSensitivity { sensitivity: f64 },
    #[error("kase_peak_oscillator_with_divergences: Invalid lb_r: {lb_r}")]
    InvalidLbR { lb_r: usize },
    #[error("kase_peak_oscillator_with_divergences: Invalid lb_l: {lb_l}")]
    InvalidLbL { lb_l: usize },
    #[error("kase_peak_oscillator_with_divergences: Invalid range_upper: {range_upper}")]
    InvalidRangeUpper { range_upper: usize },
    #[error("kase_peak_oscillator_with_divergences: Invalid range_lower: {range_lower}")]
    InvalidRangeLower { range_lower: usize },
    #[error(
        "kase_peak_oscillator_with_divergences: Invalid divergence range: range_lower = {range_lower}, range_upper = {range_upper}"
    )]
    InvalidDivergenceRange {
        range_lower: usize,
        range_upper: usize,
    },
    #[error(
        "kase_peak_oscillator_with_divergences: Inconsistent slice lengths: high={high_len}, low={low_len}, close={close_len}"
    )]
    InconsistentSliceLengths {
        high_len: usize,
        low_len: usize,
        close_len: usize,
    },
    #[error(
        "kase_peak_oscillator_with_divergences: Not enough valid data: needed = {needed}, valid = {valid}"
    )]
    NotEnoughValidData { needed: usize, valid: usize },
    #[error(
        "kase_peak_oscillator_with_divergences: Output length mismatch: expected = {expected}, oscillator = {oscillator_got}, histogram = {histogram_got}, max_peak_value = {max_peak_got}, min_peak_value = {min_peak_got}, market_extreme = {market_extreme_got}, regular_bullish = {regular_bullish_got}, hidden_bullish = {hidden_bullish_got}, regular_bearish = {regular_bearish_got}, hidden_bearish = {hidden_bearish_got}, go_long = {go_long_got}, go_short = {go_short_got}"
    )]
    OutputLengthMismatch {
        expected: usize,
        oscillator_got: usize,
        histogram_got: usize,
        max_peak_got: usize,
        min_peak_got: usize,
        market_extreme_got: usize,
        regular_bullish_got: usize,
        hidden_bullish_got: usize,
        regular_bearish_got: usize,
        hidden_bearish_got: usize,
        go_long_got: usize,
        go_short_got: usize,
    },
    #[error(
        "kase_peak_oscillator_with_divergences: Invalid range: start={start}, end={end}, step={step}"
    )]
    InvalidRange {
        start: String,
        end: String,
        step: String,
    },
    #[error("kase_peak_oscillator_with_divergences: Invalid kernel for batch: {0:?}")]
    InvalidKernelForBatch(Kernel),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ResolvedParams {
    deviations: f64,
    short_cycle: usize,
    long_cycle: usize,
    sensitivity: f64,
    all_peaks_mode: bool,
    lb_r: usize,
    lb_l: usize,
    range_upper: usize,
    range_lower: usize,
    plot_bull: bool,
    plot_hidden_bull: bool,
    plot_bear: bool,
    plot_hidden_bear: bool,
}

#[derive(Debug, Clone)]
struct RollingSma {
    period: usize,
    values: Vec<f64>,
    idx: usize,
    count: usize,
    sum: f64,
}

impl RollingSma {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            values: vec![0.0; period.max(1)],
            idx: 0,
            count: 0,
            sum: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.idx = 0;
        self.count = 0;
        self.sum = 0.0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        if self.count < self.period {
            self.values[self.idx] = value;
            self.sum += value;
            self.count += 1;
        } else {
            let old = self.values[self.idx];
            self.values[self.idx] = value;
            self.sum += value - old;
        }
        self.idx += 1;
        if self.idx == self.period {
            self.idx = 0;
        }
        if self.count == self.period {
            Some(self.sum / self.period as f64)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
struct RollingStd {
    period: usize,
    values: Vec<f64>,
    idx: usize,
    count: usize,
    sum: f64,
    sumsq: f64,
}

impl RollingStd {
    #[inline]
    fn new(period: usize) -> Self {
        Self {
            period,
            values: vec![0.0; period.max(1)],
            idx: 0,
            count: 0,
            sum: 0.0,
            sumsq: 0.0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.idx = 0;
        self.count = 0;
        self.sum = 0.0;
        self.sumsq = 0.0;
    }

    #[inline]
    fn update(&mut self, value: f64) -> Option<f64> {
        if !value.is_finite() {
            self.reset();
            return None;
        }
        if self.count < self.period {
            self.values[self.idx] = value;
            self.sum += value;
            self.sumsq += value * value;
            self.count += 1;
        } else {
            let old = self.values[self.idx];
            self.values[self.idx] = value;
            self.sum += value - old;
            self.sumsq += value * value - old * old;
        }
        self.idx += 1;
        if self.idx == self.period {
            self.idx = 0;
        }
        if self.count == self.period {
            let n = self.period as f64;
            let mean = self.sum / n;
            let variance = (self.sumsq / n) - mean * mean;
            Some(variance.max(0.0).sqrt())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct KasePeakOscillatorWithDivergencesStream {
    params: ResolvedParams,
    roots: Vec<f64>,
    prev_close: Option<f64>,
    cc_dev: RollingStd,
    avg: RollingSma,
    x1_sma: RollingSma,
    xs_sma: RollingSma,
    xp_abs_sma: RollingSma,
    xp_abs_std: RollingStd,
    osc_history: Vec<f64>,
    high_history: Vec<f64>,
    low_history: Vec<f64>,
    prev_osc_1: Option<f64>,
    prev_osc_2: Option<f64>,
    last_pivot_low: Option<usize>,
    last_pivot_high: Option<usize>,
}

impl KasePeakOscillatorWithDivergencesStream {
    pub fn try_new(
        params: KasePeakOscillatorWithDivergencesParams,
    ) -> Result<Self, KasePeakOscillatorWithDivergencesError> {
        let params = resolve_params(&params, 0)?;
        Ok(Self::from_resolved(params))
    }

    fn from_resolved(params: ResolvedParams) -> Self {
        let mut roots = vec![1.0; params.long_cycle.max(1)];
        let mut k = params.short_cycle;
        while k < params.long_cycle {
            roots[k] = (k as f64).sqrt();
            k += 1;
        }
        Self {
            params,
            roots,
            prev_close: None,
            cc_dev: RollingStd::new(9),
            avg: RollingSma::new(30),
            x1_sma: RollingSma::new(3),
            xs_sma: RollingSma::new(3),
            xp_abs_sma: RollingSma::new(50),
            xp_abs_std: RollingStd::new(50),
            osc_history: Vec::with_capacity(params.long_cycle + params.lb_l + params.lb_r + 64),
            high_history: Vec::with_capacity(params.long_cycle + params.lb_l + params.lb_r + 64),
            low_history: Vec::with_capacity(params.long_cycle + params.lb_l + params.lb_r + 64),
            prev_osc_1: None,
            prev_osc_2: None,
            last_pivot_low: None,
            last_pivot_high: None,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.prev_close = None;
        self.cc_dev.reset();
        self.avg.reset();
        self.x1_sma.reset();
        self.xs_sma.reset();
        self.xp_abs_sma.reset();
        self.xp_abs_std.reset();
        self.osc_history.clear();
        self.high_history.clear();
        self.low_history.clear();
        self.prev_osc_1 = None;
        self.prev_osc_2 = None;
        self.last_pivot_low = None;
        self.last_pivot_high = None;
    }

    #[inline]
    pub fn get_warmup_period(&self) -> usize {
        main_warmup(self.params.short_cycle, self.params.long_cycle)
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<KpoTuple> {
        if !high.is_finite()
            || !low.is_finite()
            || !close.is_finite()
            || high <= 0.0
            || low <= 0.0
            || close <= 0.0
        {
            self.reset();
            return None;
        }

        self.high_history.push(high);
        self.low_history.push(low);

        let cc_log = match self.prev_close {
            Some(prev_close) if prev_close > 0.0 => (close / prev_close).ln(),
            _ => {
                self.prev_close = Some(close);
                self.osc_history.push(f64::NAN);
                return None;
            }
        };
        self.prev_close = Some(close);

        let cc_dev = match self.cc_dev.update(cc_log) {
            Some(value) if value.is_finite() => value,
            _ => {
                self.osc_history.push(f64::NAN);
                return None;
            }
        };

        let avg = match self.avg.update(cc_dev) {
            Some(value) if value.is_finite() && value > 0.0 => value,
            _ => {
                self.osc_history.push(f64::NAN);
                return None;
            }
        };

        if self.high_history.len() < self.params.long_cycle {
            self.osc_history.push(f64::NAN);
            return None;
        }

        let current_hist_len = self.high_history.len();
        let mut max1 = 0.0;
        let mut maxs = 0.0;
        for k in self.params.short_cycle..self.params.long_cycle {
            let past_low = self.low_history[current_hist_len - 1 - k];
            let past_high = self.high_history[current_hist_len - 1 - k];
            let root = self.roots[k];
            let v1 = (high / past_low).ln() / root;
            let vs = (past_high / low).ln() / root;
            if v1.is_finite() && v1 > max1 {
                max1 = v1;
            }
            if vs.is_finite() && vs > maxs {
                maxs = vs;
            }
        }

        // These are two branches of the same bar-level formula. Advance both
        // rolling windows before testing readiness; returning after only the
        // up branch delayed the down branch by two bars and contradicted the
        // declared Kase warmup.
        let x1_avg = self.x1_sma.update(max1 / avg);
        let xs_avg = self.xs_sma.update(maxs / avg);
        let (x1_avg, xs_avg) = match (x1_avg, xs_avg) {
            (Some(x1), Some(xs)) if x1.is_finite() && xs.is_finite() => (x1, xs),
            _ => {
                self.osc_history.push(f64::NAN);
                return None;
            }
        };

        let oscillator = self.params.sensitivity * (x1_avg - xs_avg);
        if !oscillator.is_finite() {
            self.reset();
            return None;
        }
        self.osc_history.push(oscillator);

        let xp_abs = oscillator.abs();
        let xp_abs_avg = self.xp_abs_sma.update(xp_abs);
        let xp_abs_std = self.xp_abs_std.update(xp_abs);

        let histogram = oscillator;
        let mut max_peak_value = f64::NAN;
        let mut min_peak_value = f64::NAN;
        let mut market_extreme = 0.0;
        let mut go_long = 0.0;
        let mut go_short = 0.0;

        if let (Some(abs_avg), Some(abs_std)) = (xp_abs_avg, xp_abs_std) {
            let tmp_val = abs_avg + self.params.deviations * abs_std;
            let max_val = tmp_val.max(90.0);
            let min_val = tmp_val.min(90.0);
            if oscillator > 0.0 {
                max_peak_value = max_val;
                min_peak_value = min_val;
            } else {
                max_peak_value = -max_val;
                min_peak_value = -min_val;
            }

            if let (Some(prev1), Some(prev2)) = (self.prev_osc_1, self.prev_osc_2) {
                if self.params.all_peaks_mode {
                    if prev1 > 0.0 && prev1 > oscillator && prev1 >= prev2 {
                        market_extreme = oscillator;
                    }
                    if prev1 < 0.0 && prev1 < oscillator && prev1 <= prev2 {
                        market_extreme = oscillator;
                    }
                } else {
                    if prev1 > 0.0 && prev1 > oscillator && prev1 >= prev2 && prev1 >= max_val {
                        market_extreme = oscillator;
                    }
                    if prev1 < 0.0 && prev1 < oscillator && prev1 <= prev2 && prev1 <= -max_val {
                        market_extreme = oscillator;
                    }
                }
            }
        }

        if market_extreme < 0.0 {
            go_long = 1.0;
        } else if market_extreme > 0.0 {
            go_short = 1.0;
        }

        self.prev_osc_2 = self.prev_osc_1;
        self.prev_osc_1 = Some(oscillator);

        let mut regular_bullish = f64::NAN;
        let mut hidden_bullish = f64::NAN;
        let mut regular_bearish = f64::NAN;
        let mut hidden_bearish = f64::NAN;

        let current_idx = self.osc_history.len() - 1;
        if current_idx >= self.params.lb_r {
            let pivot_idx = current_idx - self.params.lb_r;
            if pivot_idx >= self.params.lb_l {
                if is_pivot_low(
                    &self.osc_history,
                    pivot_idx,
                    self.params.lb_l,
                    self.params.lb_r,
                ) {
                    if let Some(prev_pivot_idx) = self.last_pivot_low {
                        let bars = pivot_idx - prev_pivot_idx;
                        if in_range(bars, self.params.range_lower, self.params.range_upper) {
                            let osc_now = self.osc_history[pivot_idx];
                            let osc_prev = self.osc_history[prev_pivot_idx];
                            let low_now = self.low_history[pivot_idx];
                            let low_prev = self.low_history[prev_pivot_idx];
                            if self.params.plot_bull && low_now < low_prev && osc_now > osc_prev {
                                regular_bullish = osc_now;
                            }
                            if self.params.plot_hidden_bull
                                && low_now > low_prev
                                && osc_now < osc_prev
                            {
                                hidden_bullish = osc_now;
                            }
                        }
                    }
                    self.last_pivot_low = Some(pivot_idx);
                }

                if is_pivot_high(
                    &self.osc_history,
                    pivot_idx,
                    self.params.lb_l,
                    self.params.lb_r,
                ) {
                    if let Some(prev_pivot_idx) = self.last_pivot_high {
                        let bars = pivot_idx - prev_pivot_idx;
                        if in_range(bars, self.params.range_lower, self.params.range_upper) {
                            let osc_now = self.osc_history[pivot_idx];
                            let osc_prev = self.osc_history[prev_pivot_idx];
                            let high_now = self.high_history[pivot_idx];
                            let high_prev = self.high_history[prev_pivot_idx];
                            if self.params.plot_bear && high_now > high_prev && osc_now < osc_prev {
                                regular_bearish = osc_now;
                            }
                            if self.params.plot_hidden_bear
                                && high_now < high_prev
                                && osc_now > osc_prev
                            {
                                hidden_bearish = osc_now;
                            }
                        }
                    }
                    self.last_pivot_high = Some(pivot_idx);
                }
            }
        }

        Some((
            oscillator,
            histogram,
            max_peak_value,
            min_peak_value,
            market_extreme,
            regular_bullish,
            hidden_bullish,
            regular_bearish,
            hidden_bearish,
            go_long,
            go_short,
        ))
    }
}

#[inline]
fn main_warmup(_short_cycle: usize, long_cycle: usize) -> usize {
    long_cycle.max(39) + 1
}

#[inline]
fn threshold_warmup(short_cycle: usize, long_cycle: usize) -> usize {
    let _ = short_cycle;
    main_warmup(short_cycle, long_cycle) + 49
}

#[inline]
fn divergence_warmup(short_cycle: usize, long_cycle: usize, lb_l: usize, lb_r: usize) -> usize {
    main_warmup(short_cycle, long_cycle) + lb_l + lb_r
}

#[inline]
fn in_range(value: usize, lower: usize, upper: usize) -> bool {
    lower <= value && value <= upper
}

#[inline]
fn is_pivot_low(values: &[f64], pivot_idx: usize, lb_l: usize, lb_r: usize) -> bool {
    let pivot = values[pivot_idx];
    if !pivot.is_finite() {
        return false;
    }
    let start = pivot_idx - lb_l;
    let end = pivot_idx + lb_r;
    let mut i = start;
    while i <= end {
        if i != pivot_idx {
            let v = values[i];
            if !v.is_finite() || v < pivot {
                return false;
            }
        }
        i += 1;
    }
    true
}

#[inline]
fn is_pivot_high(values: &[f64], pivot_idx: usize, lb_l: usize, lb_r: usize) -> bool {
    let pivot = values[pivot_idx];
    if !pivot.is_finite() {
        return false;
    }
    let start = pivot_idx - lb_l;
    let end = pivot_idx + lb_r;
    let mut i = start;
    while i <= end {
        if i != pivot_idx {
            let v = values[i];
            if !v.is_finite() || v > pivot {
                return false;
            }
        }
        i += 1;
    }
    true
}

#[inline]
fn first_valid_ohlc(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut i = 0usize;
    while i < high.len() {
        if high[i].is_finite()
            && low[i].is_finite()
            && close[i].is_finite()
            && high[i] > 0.0
            && low[i] > 0.0
            && close[i] > 0.0
        {
            break;
        }
        i += 1;
    }
    i.min(high.len())
}

#[inline]
fn count_valid_ohlc(high: &[f64], low: &[f64], close: &[f64]) -> usize {
    let mut count = 0usize;
    let mut i = 0usize;
    while i < high.len() {
        if high[i].is_finite()
            && low[i].is_finite()
            && close[i].is_finite()
            && high[i] > 0.0
            && low[i] > 0.0
            && close[i] > 0.0
        {
            count += 1;
        }
        i += 1;
    }
    count
}

#[inline]
fn resolve_params(
    params: &KasePeakOscillatorWithDivergencesParams,
    data_len: usize,
) -> Result<ResolvedParams, KasePeakOscillatorWithDivergencesError> {
    let deviations = params.deviations.unwrap_or(DEFAULT_DEVIATIONS);
    if !deviations.is_finite() || deviations < 0.0 {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidDeviations { deviations });
    }
    let short_cycle = params.short_cycle.unwrap_or(DEFAULT_SHORT_CYCLE);
    if short_cycle == 0 || (data_len != 0 && short_cycle >= data_len) {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidShortCycle {
            short_cycle,
            data_len,
        });
    }
    let long_cycle = params.long_cycle.unwrap_or(DEFAULT_LONG_CYCLE);
    if long_cycle == 0 || (data_len != 0 && long_cycle > data_len) {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidLongCycle {
            long_cycle,
            data_len,
        });
    }
    if short_cycle >= long_cycle {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidCycleOrder {
            short_cycle,
            long_cycle,
        });
    }
    let sensitivity = params.sensitivity.unwrap_or(DEFAULT_SENSITIVITY);
    if !sensitivity.is_finite() {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidSensitivity { sensitivity });
    }
    let lb_r = params.lb_r.unwrap_or(DEFAULT_LB_R);
    if lb_r == 0 {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidLbR { lb_r });
    }
    let lb_l = params.lb_l.unwrap_or(DEFAULT_LB_L);
    if lb_l == 0 {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidLbL { lb_l });
    }
    let range_upper = params.range_upper.unwrap_or(DEFAULT_RANGE_UPPER);
    if range_upper == 0 {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidRangeUpper { range_upper });
    }
    let range_lower = params.range_lower.unwrap_or(DEFAULT_RANGE_LOWER);
    if range_lower == 0 {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidRangeLower { range_lower });
    }
    if range_lower > range_upper {
        return Err(
            KasePeakOscillatorWithDivergencesError::InvalidDivergenceRange {
                range_lower,
                range_upper,
            },
        );
    }

    Ok(ResolvedParams {
        deviations,
        short_cycle,
        long_cycle,
        sensitivity,
        all_peaks_mode: params.all_peaks_mode.unwrap_or(DEFAULT_ALL_PEAKS_MODE),
        lb_r,
        lb_l,
        range_upper,
        range_lower,
        plot_bull: params.plot_bull.unwrap_or(DEFAULT_PLOT_BULL),
        plot_hidden_bull: params.plot_hidden_bull.unwrap_or(DEFAULT_PLOT_HIDDEN_BULL),
        plot_bear: params.plot_bear.unwrap_or(DEFAULT_PLOT_BEAR),
        plot_hidden_bear: params.plot_hidden_bear.unwrap_or(DEFAULT_PLOT_HIDDEN_BEAR),
    })
}

#[inline]
fn prepare_input<'a>(
    input: &'a KasePeakOscillatorWithDivergencesInput<'a>,
    kernel: Kernel,
) -> Result<
    (
        &'a [f64],
        &'a [f64],
        &'a [f64],
        usize,
        ResolvedParams,
        Kernel,
    ),
    KasePeakOscillatorWithDivergencesError,
> {
    let (high, low, close) = match &input.data {
        KasePeakOscillatorWithDivergencesData::Candles { candles } => {
            (&candles.high[..], &candles.low[..], &candles.close[..])
        }
        KasePeakOscillatorWithDivergencesData::Slices { high, low, close } => (*high, *low, *close),
    };
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(KasePeakOscillatorWithDivergencesError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(
            KasePeakOscillatorWithDivergencesError::InconsistentSliceLengths {
                high_len: high.len(),
                low_len: low.len(),
                close_len: close.len(),
            },
        );
    }
    let first = first_valid_ohlc(high, low, close);
    if first == high.len() {
        return Err(KasePeakOscillatorWithDivergencesError::AllValuesNaN);
    }
    let params = resolve_params(&input.params, high.len())?;
    let valid = count_valid_ohlc(high, low, close);
    let needed = main_warmup(params.short_cycle, params.long_cycle) + 1;
    if valid < needed {
        return Err(KasePeakOscillatorWithDivergencesError::NotEnoughValidData { needed, valid });
    }
    let chosen = match kernel {
        Kernel::Auto => Kernel::Scalar,
        other => other.to_non_batch(),
    };
    Ok((high, low, close, first, params, chosen))
}

#[inline]
pub fn kase_peak_oscillator_with_divergences(
    input: &KasePeakOscillatorWithDivergencesInput,
) -> Result<KasePeakOscillatorWithDivergencesOutput, KasePeakOscillatorWithDivergencesError> {
    kase_peak_oscillator_with_divergences_with_kernel(input, Kernel::Auto)
}

fn row_from_slices(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: &KasePeakOscillatorWithDivergencesParams,
    oscillator_out: &mut [f64],
    histogram_out: &mut [f64],
    max_peak_out: &mut [f64],
    min_peak_out: &mut [f64],
    market_extreme_out: &mut [f64],
    regular_bullish_out: &mut [f64],
    hidden_bullish_out: &mut [f64],
    regular_bearish_out: &mut [f64],
    hidden_bearish_out: &mut [f64],
    go_long_out: &mut [f64],
    go_short_out: &mut [f64],
) -> Result<(), KasePeakOscillatorWithDivergencesError> {
    let params = resolve_params(params, 0)?;
    row_from_slices_resolved(
        high,
        low,
        close,
        params,
        oscillator_out,
        histogram_out,
        max_peak_out,
        min_peak_out,
        market_extreme_out,
        regular_bullish_out,
        hidden_bullish_out,
        regular_bearish_out,
        hidden_bearish_out,
        go_long_out,
        go_short_out,
    )
}

fn row_from_slices_resolved(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    params: ResolvedParams,
    oscillator_out: &mut [f64],
    histogram_out: &mut [f64],
    max_peak_out: &mut [f64],
    min_peak_out: &mut [f64],
    market_extreme_out: &mut [f64],
    regular_bullish_out: &mut [f64],
    hidden_bullish_out: &mut [f64],
    regular_bearish_out: &mut [f64],
    hidden_bearish_out: &mut [f64],
    go_long_out: &mut [f64],
    go_short_out: &mut [f64],
) -> Result<(), KasePeakOscillatorWithDivergencesError> {
    let mut stream = KasePeakOscillatorWithDivergencesStream::from_resolved(params);
    let mut i = 0usize;
    while i < high.len() {
        if let Some((
            oscillator,
            histogram,
            max_peak_value,
            min_peak_value,
            market_extreme,
            regular_bullish,
            hidden_bullish,
            regular_bearish,
            hidden_bearish,
            go_long,
            go_short,
        )) = stream.update(high[i], low[i], close[i])
        {
            oscillator_out[i] = oscillator;
            histogram_out[i] = histogram;
            max_peak_out[i] = max_peak_value;
            min_peak_out[i] = min_peak_value;
            market_extreme_out[i] = market_extreme;
            regular_bullish_out[i] = regular_bullish;
            hidden_bullish_out[i] = hidden_bullish;
            regular_bearish_out[i] = regular_bearish;
            hidden_bearish_out[i] = hidden_bearish;
            go_long_out[i] = go_long;
            go_short_out[i] = go_short;
        } else {
            oscillator_out[i] = f64::NAN;
            histogram_out[i] = f64::NAN;
            max_peak_out[i] = f64::NAN;
            min_peak_out[i] = f64::NAN;
            market_extreme_out[i] = f64::NAN;
            regular_bullish_out[i] = f64::NAN;
            hidden_bullish_out[i] = f64::NAN;
            regular_bearish_out[i] = f64::NAN;
            hidden_bearish_out[i] = f64::NAN;
            go_long_out[i] = f64::NAN;
            go_short_out[i] = f64::NAN;
        }
        i += 1;
    }
    Ok(())
}

pub fn kase_peak_oscillator_with_divergences_with_kernel(
    input: &KasePeakOscillatorWithDivergencesInput,
    kernel: Kernel,
) -> Result<KasePeakOscillatorWithDivergencesOutput, KasePeakOscillatorWithDivergencesError> {
    let (high, low, close, _first, params, _chosen) = prepare_input(input, kernel)?;
    let len = high.len();

    let mut oscillator = alloc_uninit_f64(len);
    let mut histogram = alloc_uninit_f64(len);
    let mut max_peak_value = alloc_uninit_f64(len);
    let mut min_peak_value = alloc_uninit_f64(len);
    let mut market_extreme = alloc_uninit_f64(len);
    let mut regular_bullish = alloc_uninit_f64(len);
    let mut hidden_bullish = alloc_uninit_f64(len);
    let mut regular_bearish = alloc_uninit_f64(len);
    let mut hidden_bearish = alloc_uninit_f64(len);
    let mut go_long = alloc_uninit_f64(len);
    let mut go_short = alloc_uninit_f64(len);

    row_from_slices_resolved(
        high,
        low,
        close,
        params,
        &mut oscillator,
        &mut histogram,
        &mut max_peak_value,
        &mut min_peak_value,
        &mut market_extreme,
        &mut regular_bullish,
        &mut hidden_bullish,
        &mut regular_bearish,
        &mut hidden_bearish,
        &mut go_long,
        &mut go_short,
    )?;

    Ok(KasePeakOscillatorWithDivergencesOutput {
        oscillator,
        histogram,
        max_peak_value,
        min_peak_value,
        market_extreme,
        regular_bullish,
        hidden_bullish,
        regular_bearish,
        hidden_bearish,
        go_long,
        go_short,
    })
}

pub fn kase_peak_oscillator_with_divergences_into_slices(
    input: &KasePeakOscillatorWithDivergencesInput,
    kernel: Kernel,
    oscillator_out: &mut [f64],
    histogram_out: &mut [f64],
    max_peak_out: &mut [f64],
    min_peak_out: &mut [f64],
    market_extreme_out: &mut [f64],
    regular_bullish_out: &mut [f64],
    hidden_bullish_out: &mut [f64],
    regular_bearish_out: &mut [f64],
    hidden_bearish_out: &mut [f64],
    go_long_out: &mut [f64],
    go_short_out: &mut [f64],
) -> Result<(), KasePeakOscillatorWithDivergencesError> {
    let (high, low, close, _first, params, _chosen) = prepare_input(input, kernel)?;
    let len = high.len();
    if oscillator_out.len() != len
        || histogram_out.len() != len
        || max_peak_out.len() != len
        || min_peak_out.len() != len
        || market_extreme_out.len() != len
        || regular_bullish_out.len() != len
        || hidden_bullish_out.len() != len
        || regular_bearish_out.len() != len
        || hidden_bearish_out.len() != len
        || go_long_out.len() != len
        || go_short_out.len() != len
    {
        return Err(
            KasePeakOscillatorWithDivergencesError::OutputLengthMismatch {
                expected: len,
                oscillator_got: oscillator_out.len(),
                histogram_got: histogram_out.len(),
                max_peak_got: max_peak_out.len(),
                min_peak_got: min_peak_out.len(),
                market_extreme_got: market_extreme_out.len(),
                regular_bullish_got: regular_bullish_out.len(),
                hidden_bullish_got: hidden_bullish_out.len(),
                regular_bearish_got: regular_bearish_out.len(),
                hidden_bearish_got: hidden_bearish_out.len(),
                go_long_got: go_long_out.len(),
                go_short_got: go_short_out.len(),
            },
        );
    }

    row_from_slices_resolved(
        high,
        low,
        close,
        params,
        oscillator_out,
        histogram_out,
        max_peak_out,
        min_peak_out,
        market_extreme_out,
        regular_bullish_out,
        hidden_bullish_out,
        regular_bearish_out,
        hidden_bearish_out,
        go_long_out,
        go_short_out,
    )
}

#[cfg(not(target_arch = "wasm32"))]
pub fn kase_peak_oscillator_with_divergences_into(
    input: &KasePeakOscillatorWithDivergencesInput,
    oscillator_out: &mut [f64],
    histogram_out: &mut [f64],
    max_peak_out: &mut [f64],
    min_peak_out: &mut [f64],
    market_extreme_out: &mut [f64],
    regular_bullish_out: &mut [f64],
    hidden_bullish_out: &mut [f64],
    regular_bearish_out: &mut [f64],
    hidden_bearish_out: &mut [f64],
    go_long_out: &mut [f64],
    go_short_out: &mut [f64],
) -> Result<(), KasePeakOscillatorWithDivergencesError> {
    kase_peak_oscillator_with_divergences_into_slices(
        input,
        Kernel::Auto,
        oscillator_out,
        histogram_out,
        max_peak_out,
        min_peak_out,
        market_extreme_out,
        regular_bullish_out,
        hidden_bullish_out,
        regular_bearish_out,
        hidden_bearish_out,
        go_long_out,
        go_short_out,
    )
}

#[derive(Debug, Clone)]
pub struct KasePeakOscillatorWithDivergencesBatchRange {
    pub deviations: (f64, f64, f64),
    pub short_cycle: (usize, usize, usize),
    pub long_cycle: (usize, usize, usize),
    pub sensitivity: (f64, f64, f64),
    pub all_peaks_mode: bool,
    pub lb_r: usize,
    pub lb_l: usize,
    pub range_upper: usize,
    pub range_lower: usize,
    pub plot_bull: bool,
    pub plot_hidden_bull: bool,
    pub plot_bear: bool,
    pub plot_hidden_bear: bool,
}

impl Default for KasePeakOscillatorWithDivergencesBatchRange {
    fn default() -> Self {
        Self {
            deviations: (DEFAULT_DEVIATIONS, DEFAULT_DEVIATIONS, 0.0),
            short_cycle: (DEFAULT_SHORT_CYCLE, DEFAULT_SHORT_CYCLE, 0),
            long_cycle: (DEFAULT_LONG_CYCLE, DEFAULT_LONG_CYCLE, 0),
            sensitivity: (DEFAULT_SENSITIVITY, DEFAULT_SENSITIVITY, 0.0),
            all_peaks_mode: DEFAULT_ALL_PEAKS_MODE,
            lb_r: DEFAULT_LB_R,
            lb_l: DEFAULT_LB_L,
            range_upper: DEFAULT_RANGE_UPPER,
            range_lower: DEFAULT_RANGE_LOWER,
            plot_bull: DEFAULT_PLOT_BULL,
            plot_hidden_bull: DEFAULT_PLOT_HIDDEN_BULL,
            plot_bear: DEFAULT_PLOT_BEAR,
            plot_hidden_bear: DEFAULT_PLOT_HIDDEN_BEAR,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct KasePeakOscillatorWithDivergencesBatchBuilder {
    range: KasePeakOscillatorWithDivergencesBatchRange,
    kernel: Kernel,
}

#[derive(Debug, Clone)]
pub struct KasePeakOscillatorWithDivergencesBatchOutput {
    pub oscillator: Vec<f64>,
    pub histogram: Vec<f64>,
    pub max_peak_value: Vec<f64>,
    pub min_peak_value: Vec<f64>,
    pub market_extreme: Vec<f64>,
    pub regular_bullish: Vec<f64>,
    pub hidden_bullish: Vec<f64>,
    pub regular_bearish: Vec<f64>,
    pub hidden_bearish: Vec<f64>,
    pub go_long: Vec<f64>,
    pub go_short: Vec<f64>,
    pub deviations: Vec<f64>,
    pub short_cycles: Vec<usize>,
    pub long_cycles: Vec<usize>,
    pub sensitivities: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl KasePeakOscillatorWithDivergencesBatchBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn deviations_range(mut self, range: (f64, f64, f64)) -> Self {
        self.range.deviations = range;
        self
    }

    #[inline]
    pub fn short_cycle_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.short_cycle = range;
        self
    }

    #[inline]
    pub fn long_cycle_range(mut self, range: (usize, usize, usize)) -> Self {
        self.range.long_cycle = range;
        self
    }

    #[inline]
    pub fn sensitivity_range(mut self, range: (f64, f64, f64)) -> Self {
        self.range.sensitivity = range;
        self
    }

    #[inline]
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    #[inline]
    pub fn apply_slices(
        self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Result<KasePeakOscillatorWithDivergencesBatchOutput, KasePeakOscillatorWithDivergencesError>
    {
        kase_peak_oscillator_with_divergences_batch_with_kernel(
            high,
            low,
            close,
            &self.range,
            self.kernel,
        )
    }
}

fn axis_usize(
    start: usize,
    end: usize,
    step: usize,
) -> Result<Vec<usize>, KasePeakOscillatorWithDivergencesError> {
    if step == 0 {
        return Ok(vec![start]);
    }
    if start > end {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut value = start;
    while value <= end {
        out.push(value);
        match value.checked_add(step) {
            Some(next) => value = next,
            None => break,
        }
    }
    Ok(out)
}

fn axis_f64(
    start: f64,
    end: f64,
    step: f64,
) -> Result<Vec<f64>, KasePeakOscillatorWithDivergencesError> {
    if !start.is_finite() || !end.is_finite() || !step.is_finite() {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    if step == 0.0 {
        return Ok(vec![start]);
    }
    if start > end || step < 0.0 {
        return Err(KasePeakOscillatorWithDivergencesError::InvalidRange {
            start: start.to_string(),
            end: end.to_string(),
            step: step.to_string(),
        });
    }
    let mut out = Vec::new();
    let mut value = start;
    while value <= end + step * 1e-12 {
        out.push(value);
        value += step;
    }
    Ok(out)
}

/// Public so the CUDA wrapper can expand the SAME grid the CPU expands.
///
/// The wrapper must not re-derive the parameter list: a second implementation
/// of the cross product is a second place for it to drift, and a row-order
/// difference would silently pair every combo with the wrong output row.
pub fn expand_grid_kpo(
    sweep: &KasePeakOscillatorWithDivergencesBatchRange,
) -> Result<Vec<KasePeakOscillatorWithDivergencesParams>, KasePeakOscillatorWithDivergencesError> {
    let deviations = axis_f64(sweep.deviations.0, sweep.deviations.1, sweep.deviations.2)?;
    let short_cycles = axis_usize(
        sweep.short_cycle.0,
        sweep.short_cycle.1,
        sweep.short_cycle.2,
    )?;
    let long_cycles = axis_usize(sweep.long_cycle.0, sweep.long_cycle.1, sweep.long_cycle.2)?;
    let sensitivities = axis_f64(
        sweep.sensitivity.0,
        sweep.sensitivity.1,
        sweep.sensitivity.2,
    )?;
    let mut out = Vec::new();
    for &deviation in &deviations {
        for &short_cycle in &short_cycles {
            for &long_cycle in &long_cycles {
                for &sensitivity in &sensitivities {
                    out.push(KasePeakOscillatorWithDivergencesParams {
                        deviations: Some(deviation),
                        short_cycle: Some(short_cycle),
                        long_cycle: Some(long_cycle),
                        sensitivity: Some(sensitivity),
                        all_peaks_mode: Some(sweep.all_peaks_mode),
                        lb_r: Some(sweep.lb_r),
                        lb_l: Some(sweep.lb_l),
                        range_upper: Some(sweep.range_upper),
                        range_lower: Some(sweep.range_lower),
                        plot_bull: Some(sweep.plot_bull),
                        plot_hidden_bull: Some(sweep.plot_hidden_bull),
                        plot_bear: Some(sweep.plot_bear),
                        plot_hidden_bear: Some(sweep.plot_hidden_bear),
                    });
                }
            }
        }
    }
    Ok(out)
}

pub fn kase_peak_oscillator_with_divergences_batch_with_kernel(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KasePeakOscillatorWithDivergencesBatchRange,
    kernel: Kernel,
) -> Result<KasePeakOscillatorWithDivergencesBatchOutput, KasePeakOscillatorWithDivergencesError> {
    let batch_kernel = match kernel {
        Kernel::Auto => detect_best_batch_kernel(),
        other => other,
    };
    kase_peak_oscillator_with_divergences_batch_inner(
        high,
        low,
        close,
        sweep,
        batch_kernel.to_non_batch(),
        false,
    )
}

pub fn kase_peak_oscillator_with_divergences_batch_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KasePeakOscillatorWithDivergencesBatchRange,
) -> Result<KasePeakOscillatorWithDivergencesBatchOutput, KasePeakOscillatorWithDivergencesError> {
    kase_peak_oscillator_with_divergences_batch_with_kernel(high, low, close, sweep, Kernel::Auto)
}

pub fn kase_peak_oscillator_with_divergences_batch_par_slice(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KasePeakOscillatorWithDivergencesBatchRange,
) -> Result<KasePeakOscillatorWithDivergencesBatchOutput, KasePeakOscillatorWithDivergencesError> {
    let kernel = detect_best_kernel();
    #[cfg(not(target_arch = "wasm32"))]
    {
        return kase_peak_oscillator_with_divergences_batch_inner(
            high, low, close, sweep, kernel, true,
        );
    }
    #[cfg(target_arch = "wasm32")]
    {
        kase_peak_oscillator_with_divergences_batch_inner(high, low, close, sweep, kernel, false)
    }
}

pub fn kase_peak_oscillator_with_divergences_batch_inner(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KasePeakOscillatorWithDivergencesBatchRange,
    kernel: Kernel,
    parallel: bool,
) -> Result<KasePeakOscillatorWithDivergencesBatchOutput, KasePeakOscillatorWithDivergencesError> {
    if high.is_empty() || low.is_empty() || close.is_empty() {
        return Err(KasePeakOscillatorWithDivergencesError::EmptyInputData);
    }
    if high.len() != low.len() || high.len() != close.len() {
        return Err(
            KasePeakOscillatorWithDivergencesError::InconsistentSliceLengths {
                high_len: high.len(),
                low_len: low.len(),
                close_len: close.len(),
            },
        );
    }

    let combos = expand_grid_kpo(sweep)?;
    let rows = combos.len();
    let cols = high.len();
    let total = rows * cols;
    let mut oscillator_out = vec![f64::NAN; total];
    let mut histogram_out = vec![f64::NAN; total];
    let mut max_peak_out = vec![f64::NAN; total];
    let mut min_peak_out = vec![f64::NAN; total];
    let mut market_extreme_out = vec![f64::NAN; total];
    let mut regular_bullish_out = vec![f64::NAN; total];
    let mut hidden_bullish_out = vec![f64::NAN; total];
    let mut regular_bearish_out = vec![f64::NAN; total];
    let mut hidden_bearish_out = vec![f64::NAN; total];
    let mut go_long_out = vec![f64::NAN; total];
    let mut go_short_out = vec![f64::NAN; total];

    #[cfg(not(target_arch = "wasm32"))]
    if parallel && rows > 1 {
        oscillator_out
            .par_chunks_mut(cols)
            .zip(histogram_out.par_chunks_mut(cols))
            .zip(max_peak_out.par_chunks_mut(cols))
            .zip(min_peak_out.par_chunks_mut(cols))
            .zip(market_extreme_out.par_chunks_mut(cols))
            .zip(regular_bullish_out.par_chunks_mut(cols))
            .zip(hidden_bullish_out.par_chunks_mut(cols))
            .zip(regular_bearish_out.par_chunks_mut(cols))
            .zip(hidden_bearish_out.par_chunks_mut(cols))
            .zip(go_long_out.par_chunks_mut(cols))
            .zip(go_short_out.par_chunks_mut(cols))
            .zip(combos.par_iter())
            .try_for_each(
                |((((((((((((a, b), c), d), e), f), g), h), i), j), k), combo))| {
                    let _ = kernel;
                    row_from_slices(high, low, close, combo, a, b, c, d, e, f, g, h, i, j, k)
                },
            )?;
    } else {
        for (row, combo) in combos.iter().enumerate() {
            let start = row * cols;
            let end = start + cols;
            let _ = kernel;
            row_from_slices(
                high,
                low,
                close,
                combo,
                &mut oscillator_out[start..end],
                &mut histogram_out[start..end],
                &mut max_peak_out[start..end],
                &mut min_peak_out[start..end],
                &mut market_extreme_out[start..end],
                &mut regular_bullish_out[start..end],
                &mut hidden_bullish_out[start..end],
                &mut regular_bearish_out[start..end],
                &mut hidden_bearish_out[start..end],
                &mut go_long_out[start..end],
                &mut go_short_out[start..end],
            )?;
        }
    }

    Ok(KasePeakOscillatorWithDivergencesBatchOutput {
        oscillator: oscillator_out,
        histogram: histogram_out,
        max_peak_value: max_peak_out,
        min_peak_value: min_peak_out,
        market_extreme: market_extreme_out,
        regular_bullish: regular_bullish_out,
        hidden_bullish: hidden_bullish_out,
        regular_bearish: regular_bearish_out,
        hidden_bearish: hidden_bearish_out,
        go_long: go_long_out,
        go_short: go_short_out,
        deviations: combos
            .iter()
            .map(|p| p.deviations.unwrap_or(DEFAULT_DEVIATIONS))
            .collect(),
        short_cycles: combos
            .iter()
            .map(|p| p.short_cycle.unwrap_or(DEFAULT_SHORT_CYCLE))
            .collect(),
        long_cycles: combos
            .iter()
            .map(|p| p.long_cycle.unwrap_or(DEFAULT_LONG_CYCLE))
            .collect(),
        sensitivities: combos
            .iter()
            .map(|p| p.sensitivity.unwrap_or(DEFAULT_SENSITIVITY))
            .collect(),
        rows,
        cols,
    })
}

pub fn kase_peak_oscillator_with_divergences_batch_inner_into(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    sweep: &KasePeakOscillatorWithDivergencesBatchRange,
    kernel: Kernel,
    oscillator_out: &mut [f64],
    histogram_out: &mut [f64],
    max_peak_out: &mut [f64],
    min_peak_out: &mut [f64],
    market_extreme_out: &mut [f64],
    regular_bullish_out: &mut [f64],
    hidden_bullish_out: &mut [f64],
    regular_bearish_out: &mut [f64],
    hidden_bearish_out: &mut [f64],
    go_long_out: &mut [f64],
    go_short_out: &mut [f64],
) -> Result<Vec<KasePeakOscillatorWithDivergencesParams>, KasePeakOscillatorWithDivergencesError> {
    let out =
        kase_peak_oscillator_with_divergences_batch_inner(high, low, close, sweep, kernel, false)?;
    let total = out.rows * out.cols;
    if oscillator_out.len() != total
        || histogram_out.len() != total
        || max_peak_out.len() != total
        || min_peak_out.len() != total
        || market_extreme_out.len() != total
        || regular_bullish_out.len() != total
        || hidden_bullish_out.len() != total
        || regular_bearish_out.len() != total
        || hidden_bearish_out.len() != total
        || go_long_out.len() != total
        || go_short_out.len() != total
    {
        return Err(
            KasePeakOscillatorWithDivergencesError::OutputLengthMismatch {
                expected: total,
                oscillator_got: oscillator_out.len(),
                histogram_got: histogram_out.len(),
                max_peak_got: max_peak_out.len(),
                min_peak_got: min_peak_out.len(),
                market_extreme_got: market_extreme_out.len(),
                regular_bullish_got: regular_bullish_out.len(),
                hidden_bullish_got: hidden_bullish_out.len(),
                regular_bearish_got: regular_bearish_out.len(),
                hidden_bearish_got: hidden_bearish_out.len(),
                go_long_got: go_long_out.len(),
                go_short_got: go_short_out.len(),
            },
        );
    }
    oscillator_out.copy_from_slice(&out.oscillator);
    histogram_out.copy_from_slice(&out.histogram);
    max_peak_out.copy_from_slice(&out.max_peak_value);
    min_peak_out.copy_from_slice(&out.min_peak_value);
    market_extreme_out.copy_from_slice(&out.market_extreme);
    regular_bullish_out.copy_from_slice(&out.regular_bullish);
    hidden_bullish_out.copy_from_slice(&out.hidden_bullish);
    regular_bearish_out.copy_from_slice(&out.regular_bearish);
    hidden_bearish_out.copy_from_slice(&out.hidden_bearish);
    go_long_out.copy_from_slice(&out.go_long);
    go_short_out.copy_from_slice(&out.go_short);
    expand_grid_kpo(sweep)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ohlc(len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::with_capacity(len);
        let mut low = Vec::with_capacity(len);
        let mut close = Vec::with_capacity(len);
        let mut price = 100.0;
        for i in 0..len {
            let wave = ((i as f64) * 0.17).sin() * 1.8 + ((i as f64) * 0.07).cos() * 0.9;
            price += wave * 0.35 + 0.12;
            close.push(price);
            high.push(price + 0.8 + ((i % 5) as f64) * 0.05);
            low.push(price - 0.7 - ((i % 7) as f64) * 0.04);
        }
        (high, low, close)
    }

    fn approx_eq_or_nan(a: f64, b: f64) -> bool {
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-10
    }

    fn direct_population_std(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / values.len() as f64;
        variance.sqrt()
    }

    /// Independent, deliberately non-streaming reconstruction of the Kase
    /// PeakOscillator formula. This does not call the production rolling
    /// helpers, so parity catches a shared-state/indexing mistake rather than
    /// merely proving that two entry points reuse the same implementation.
    fn direct_peak_oscillator(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        short_cycle: usize,
        long_cycle: usize,
        sensitivity: f64,
    ) -> Vec<f64> {
        let len = close.len();
        let mut log_return = vec![f64::NAN; len];
        let mut return_deviation = vec![f64::NAN; len];
        let mut average_deviation = vec![f64::NAN; len];
        let mut normalized_up = vec![f64::NAN; len];
        let mut normalized_down = vec![f64::NAN; len];
        let mut oscillator = vec![f64::NAN; len];

        for index in 1..len {
            log_return[index] = (close[index] / close[index - 1]).ln();
            if index >= 9 {
                return_deviation[index] = direct_population_std(&log_return[index + 1 - 9..=index]);
            }
            if index >= 38 {
                average_deviation[index] =
                    return_deviation[index + 1 - 30..=index].iter().sum::<f64>() / 30.0;
            }
            if index + 1 < long_cycle || !average_deviation[index].is_finite() {
                continue;
            }

            let mut strongest_up = 0.0_f64;
            let mut strongest_down = 0.0_f64;
            for cycle in short_cycle..long_cycle {
                let root = (cycle as f64).sqrt();
                strongest_up = strongest_up.max((high[index] / low[index - cycle]).ln() / root);
                strongest_down = strongest_down.max((high[index - cycle] / low[index]).ln() / root);
            }
            normalized_up[index] = strongest_up / average_deviation[index];
            normalized_down[index] = strongest_down / average_deviation[index];

            // The first ratio exists at `long_cycle - 1`; the third sample
            // therefore completes both concurrent SMA(3) windows at
            // `long_cycle + 1`.
            if index >= long_cycle + 1 {
                let smoothed_up = normalized_up[index - 2..=index].iter().sum::<f64>() / 3.0;
                let smoothed_down = normalized_down[index - 2..=index].iter().sum::<f64>() / 3.0;
                oscillator[index] = sensitivity * (smoothed_up - smoothed_down);
            }
        }
        oscillator
    }

    #[test]
    fn kpo_scalar_matches_independent_window_formula() {
        let (high, low, close) = sample_ohlc(256);
        let params = KasePeakOscillatorWithDivergencesParams::default();
        let out = kase_peak_oscillator_with_divergences(
            &KasePeakOscillatorWithDivergencesInput::from_slices(
                &high,
                &low,
                &close,
                params.clone(),
            ),
        )
        .unwrap();
        let direct = direct_peak_oscillator(
            &high,
            &low,
            &close,
            params.short_cycle.unwrap_or(DEFAULT_SHORT_CYCLE),
            params.long_cycle.unwrap_or(DEFAULT_LONG_CYCLE),
            params.sensitivity.unwrap_or(DEFAULT_SENSITIVITY),
        );

        let production_first = out.oscillator.iter().position(|value| value.is_finite());
        let direct_first = direct.iter().position(|value| value.is_finite());
        assert_eq!(
            production_first, direct_first,
            "the independent formula and production path disagree on warmup semantics"
        );
        let first = direct_first.expect("the independent fixture must reach a valid oscillator");
        for index in [first, first + 1, first + 24, first + 49, first + 93, 255] {
            assert!(
                (out.oscillator[index] - direct[index]).abs() <= 1e-9,
                "independent Kase formula mismatch at {index}: production={} direct={}",
                out.oscillator[index],
                direct[index]
            );
        }

        let threshold_index = first + 49;
        let absolute_window = direct[threshold_index + 1 - 50..=threshold_index]
            .iter()
            .map(|value| value.abs())
            .collect::<Vec<_>>();
        let threshold = absolute_window.iter().sum::<f64>() / 50.0
            + DEFAULT_DEVIATIONS * direct_population_std(&absolute_window);
        let sign = direct[threshold_index].signum();
        assert!((out.max_peak_value[threshold_index] - sign * threshold.max(90.0)).abs() <= 1e-9);
        assert!((out.min_peak_value[threshold_index] - sign * threshold.min(90.0)).abs() <= 1e-9);
    }

    #[test]
    fn kpo_histogram_is_a_display_alias_not_a_distinct_feature() {
        let (high, low, close) = sample_ohlc(256);
        let out = kase_peak_oscillator_with_divergences(
            &KasePeakOscillatorWithDivergencesInput::from_slices(
                &high,
                &low,
                &close,
                Default::default(),
            ),
        )
        .unwrap();

        for (index, (&oscillator, &histogram)) in
            out.oscillator.iter().zip(&out.histogram).enumerate()
        {
            if oscillator.is_nan() {
                assert!(
                    histogram.is_nan(),
                    "histogram[{index}] lost undefined state"
                );
            } else {
                assert_eq!(
                    oscillator.to_bits(),
                    histogram.to_bits(),
                    "histogram[{index}] is not the exact PeakOscillator display alias"
                );
            }
        }
    }

    #[test]
    fn kpo_output_contract_and_signals_are_consistent() {
        let (high, low, close) = sample_ohlc(256);
        let out = kase_peak_oscillator_with_divergences(
            &KasePeakOscillatorWithDivergencesInput::from_slices(
                &high,
                &low,
                &close,
                Default::default(),
            ),
        )
        .unwrap();
        assert_eq!(out.oscillator.len(), close.len());
        assert!(
            out.oscillator[..main_warmup(DEFAULT_SHORT_CYCLE, DEFAULT_LONG_CYCLE)]
                .iter()
                .all(|v| v.is_nan())
        );
        for i in 0..close.len() {
            assert!(approx_eq_or_nan(out.oscillator[i], out.histogram[i]));
            if out.market_extreme[i].is_finite() {
                if out.market_extreme[i] < 0.0 {
                    assert_eq!(out.go_long[i], 1.0);
                }
                if out.market_extreme[i] > 0.0 {
                    assert_eq!(out.go_short[i], 1.0);
                }
            }
        }
    }

    #[test]
    fn kpo_stream_matches_batch_with_reset() {
        let (mut high, mut low, mut close) = sample_ohlc(220);
        high[110] = f64::NAN;
        low[110] = f64::NAN;
        close[110] = f64::NAN;
        let batch = kase_peak_oscillator_with_divergences(
            &KasePeakOscillatorWithDivergencesInput::from_slices(
                &high,
                &low,
                &close,
                Default::default(),
            ),
        )
        .unwrap();
        let mut stream =
            KasePeakOscillatorWithDivergencesStream::try_new(Default::default()).unwrap();
        let mut streamed = vec![Vec::new(); 11];
        for i in 0..high.len() {
            match stream.update(high[i], low[i], close[i]) {
                Some(v) => {
                    let arr = [v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7, v.8, v.9, v.10];
                    for (series, value) in streamed.iter_mut().zip(arr) {
                        series.push(value);
                    }
                }
                None => {
                    for series in &mut streamed {
                        series.push(f64::NAN);
                    }
                }
            }
        }
        let batch_series = [
            &batch.oscillator,
            &batch.histogram,
            &batch.max_peak_value,
            &batch.min_peak_value,
            &batch.market_extreme,
            &batch.regular_bullish,
            &batch.hidden_bullish,
            &batch.regular_bearish,
            &batch.hidden_bearish,
            &batch.go_long,
            &batch.go_short,
        ];
        for (series_idx, (lhs, rhs)) in batch_series.iter().zip(streamed.iter()).enumerate() {
            for i in 0..lhs.len() {
                assert!(
                    approx_eq_or_nan(lhs[i], rhs[i]),
                    "series {} mismatch at {}: batch={}, stream={}",
                    series_idx,
                    i,
                    lhs[i],
                    rhs[i]
                );
            }
        }
    }

    #[test]
    fn kpo_rejects_invalid_cycles() {
        let (high, low, close) = sample_ohlc(32);
        let err = kase_peak_oscillator_with_divergences(
            &KasePeakOscillatorWithDivergencesInput::from_slices(
                &high,
                &low,
                &close,
                KasePeakOscillatorWithDivergencesParams {
                    short_cycle: Some(10),
                    long_cycle: Some(10),
                    ..Default::default()
                },
            ),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            KasePeakOscillatorWithDivergencesError::InvalidCycleOrder { .. }
        ));
    }
}
