#include <cmath>
#include <cstddef>

namespace {

constexpr int MA_EMA = 0;
constexpr int MA_SMA = 1;
constexpr int MA_WMA = 2;
constexpr int MA_RMA = 3;

struct SmaState {
    double* values;
    double* valid;
    int period;
    int idx;
    int count;
    int valid_count;
    double sum;

    __device__ void init(double* values_storage, double* valid_storage, int period_value) {
        values = values_storage;
        valid = valid_storage;
        period = period_value;
        idx = 0;
        count = 0;
        valid_count = 0;
        sum = 0.0;
    }

    __device__ bool update(bool has_value, double value, double* out) {
        if (count >= period) {
            const int old_idx = idx;
            if (valid[old_idx] != 0.0) {
                valid_count -= 1;
                sum -= values[old_idx];
            }
        } else {
            count += 1;
        }

        if (has_value && isfinite(value)) {
            values[idx] = value;
            valid[idx] = 1.0;
            valid_count += 1;
            sum += value;
        } else {
            values[idx] = 0.0;
            valid[idx] = 0.0;
        }

        idx += 1;
        if (idx == period) {
            idx = 0;
        }

        if (count < period) {
            *out = NAN;
            return false;
        }
        if (valid_count == period) {
            *out = sum / static_cast<double>(period);
        } else {
            *out = NAN;
        }
        return true;
    }
};

struct WmaState {
    double* values;
    double* valid;
    int period;
    double denom;
    int idx;
    int count;
    int valid_count;

    __device__ void init(double* values_storage, double* valid_storage, int period_value) {
        values = values_storage;
        valid = valid_storage;
        period = period_value;
        denom = static_cast<double>(period_value * (period_value + 1) / 2);
        idx = 0;
        count = 0;
        valid_count = 0;
    }

    __device__ bool update(bool has_value, double value, double* out) {
        if (count >= period) {
            const int old_idx = idx;
            if (valid[old_idx] != 0.0) {
                valid_count -= 1;
            }
        } else {
            count += 1;
        }

        if (has_value && isfinite(value)) {
            values[idx] = value;
            valid[idx] = 1.0;
            valid_count += 1;
        } else {
            values[idx] = 0.0;
            valid[idx] = 0.0;
        }

        idx += 1;
        if (idx == period) {
            idx = 0;
        }

        if (count < period) {
            *out = NAN;
            return false;
        }
        if (valid_count != period) {
            *out = NAN;
            return true;
        }

        double weighted = 0.0;
        double weight = 1.0;
        int pos = idx;
        for (int i = 0; i < period; ++i) {
            weighted += values[pos] * weight;
            weight += 1.0;
            pos += 1;
            if (pos == period) {
                pos = 0;
            }
        }
        *out = weighted / denom;
        return true;
    }
};

struct ExpState {
    double alpha;
    double value;
    bool initialized;

    __device__ void init_ema(int period) {
        alpha = 2.0 / (static_cast<double>(period) + 1.0);
        value = NAN;
        initialized = false;
    }

    __device__ void init_rma(int period) {
        alpha = 1.0 / static_cast<double>(period);
        value = NAN;
        initialized = false;
    }

    __device__ bool update(bool has_value, double input, double* out) {
        if (has_value && isfinite(input)) {
            if (initialized) {
                value += alpha * (input - value);
            } else {
                value = input;
                initialized = true;
            }
            *out = value;
            return true;
        }
        value = NAN;
        initialized = false;
        *out = NAN;
        return true;
    }
};

struct MaState {
    int kind;
    SmaState sma;
    WmaState wma;
    ExpState exp;

    __device__ void init(
        int kind_value,
        int period,
        double* values_storage,
        double* valid_storage
    ) {
        kind = kind_value;
        if (kind == MA_SMA) {
            sma.init(values_storage, valid_storage, period);
        } else if (kind == MA_WMA) {
            wma.init(values_storage, valid_storage, period);
        } else if (kind == MA_EMA) {
            exp.init_ema(period);
        } else {
            exp.init_rma(period);
        }
    }

    __device__ bool update(bool has_value, double input, double* out) {
        if (kind == MA_SMA) {
            return sma.update(has_value, input, out);
        }
        if (kind == MA_WMA) {
            return wma.update(has_value, input, out);
        }
        return exp.update(has_value, input, out);
    }
};

__device__ inline bool valid_ohlcv(double high, double low, double close, double volume) {
    return isfinite(high) && isfinite(low) && isfinite(close) && isfinite(volume);
}

__device__ inline double normalize_h0_l0(double h0, double l0) {
    const double diff = fabs(h0 - l0);
    return diff > 1e-12 ? diff : 1e-12;
}

__device__ inline bool compute_bp_sp(
    bool has_volume_avg,
    double volume_avg,
    double volume,
    double p,
    double prev_p,
    double h0,
    double l0,
    double* bp,
    double* sp
) {
    if (!has_volume_avg || !isfinite(volume_avg) || !isfinite(p) || !isfinite(prev_p)) {
        return false;
    }

    const double v_ratio = volume_avg == 0.0 ? 1.0 : volume / volume_avg;
    if (!isfinite(v_ratio)) {
        return false;
    }

    const double denom = normalize_h0_l0(h0, l0);
    const double k = 0.375;

    if (p < prev_p) {
        if (p == 0.0) {
            return false;
        }
        const double exponent = (k * (p + prev_p) / denom) * ((prev_p - p) / p);
        *bp = v_ratio / exp(exponent);
        *sp = v_ratio;
        return true;
    }
    if (p > prev_p) {
        if (prev_p == 0.0) {
            return false;
        }
        const double exponent = (k * (p + prev_p) / denom) * ((p - prev_p) / prev_p);
        *bp = v_ratio;
        *sp = v_ratio / exp(exponent);
        return true;
    }

    *bp = v_ratio;
    *sp = v_ratio;
    return true;
}

__device__ inline bool finalize_di(bool has_bp, double bp, bool has_sp, double sp, double* out) {
    if (!has_bp || !has_sp) {
        *out = NAN;
        return false;
    }
    if (!isfinite(bp) || !isfinite(sp)) {
        *out = NAN;
        return false;
    }
    if (bp > sp) {
        *out = bp == 0.0 ? 100.0 : 100.0 * (1.0 - sp / bp);
        return true;
    }
    if (bp < sp) {
        *out = sp == 0.0 ? -100.0 : 100.0 * (bp / sp - 1.0);
        return true;
    }
    *out = 0.0;
    return true;
}

}

extern "C" __global__ void demand_index_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ len_bs_values,
    const int* __restrict__ len_bs_ma_values,
    const int* __restrict__ len_di_ma_values,
    const int* __restrict__ ma_codes,
    int rows,
    int scratch_cap,
    double* __restrict__ scratch_buf,
    double* __restrict__ out_demand_index,
    double* __restrict__ out_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int len_bs = len_bs_values[row];
    const int len_bs_ma = len_bs_ma_values[row];
    const int len_di_ma = len_di_ma_values[row];
    const int ma_code = ma_codes[row];

    double* row_di = out_demand_index + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_di[i] = NAN;
        row_signal[i] = NAN;
    }

    if (len_bs <= 0 || len_bs_ma <= 0 || len_di_ma <= 0) {
        return;
    }
    if (ma_code < MA_EMA || ma_code > MA_RMA) {
        return;
    }

    const int needed = len_bs * 2 + len_bs_ma * 4 + len_di_ma * 2;
    if (needed > scratch_cap) {
        return;
    }

    double* row_scratch = scratch_buf + static_cast<size_t>(row) * static_cast<size_t>(scratch_cap);
    int offset = 0;

    double* volume_values = row_scratch + offset;
    offset += len_bs;
    double* volume_valid = row_scratch + offset;
    offset += len_bs;

    double* bp_values = row_scratch + offset;
    offset += len_bs_ma;
    double* bp_valid = row_scratch + offset;
    offset += len_bs_ma;

    double* sp_values = row_scratch + offset;
    offset += len_bs_ma;
    double* sp_valid = row_scratch + offset;
    offset += len_bs_ma;

    double* signal_values = row_scratch + offset;
    offset += len_di_ma;
    double* signal_valid = row_scratch + offset;

    MaState volume_avg;
    MaState bp_avg;
    MaState sp_avg;
    SmaState signal_avg;
    volume_avg.init(ma_code, len_bs, volume_values, volume_valid);
    bp_avg.init(ma_code, len_bs_ma, bp_values, bp_valid);
    sp_avg.init(ma_code, len_bs_ma, sp_values, sp_valid);
    signal_avg.init(signal_values, signal_valid, len_di_ma);

    double h0 = NAN;
    double l0 = NAN;
    bool has_h0_l0 = false;
    double prev_p = NAN;
    bool has_prev_p = false;

    for (int i = 0; i < len; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double v = volume[i];

        if (!valid_ohlcv(h, l, c, v)) {
            double unused = NAN;
            volume_avg.update(false, 0.0, &unused);

            double bp_ma = NAN;
            double sp_ma = NAN;
            const bool has_bp_ma = bp_avg.update(false, 0.0, &bp_ma);
            const bool has_sp_ma = sp_avg.update(false, 0.0, &sp_ma);

            double di = NAN;
            const bool has_di = finalize_di(has_bp_ma, bp_ma, has_sp_ma, sp_ma, &di);
            double signal = NAN;
            const bool has_signal = signal_avg.update(has_di && isfinite(di), di, &signal);

            row_di[i] = has_di ? di : NAN;
            row_signal[i] = has_signal ? signal : NAN;
            has_prev_p = false;
            continue;
        }

        if (!has_h0_l0) {
            h0 = h;
            l0 = l;
            has_h0_l0 = true;
        }

        double volume_avg_now = NAN;
        const bool has_volume_avg = volume_avg.update(true, v, &volume_avg_now);
        const double p = h + l + 2.0 * c;

        if (!has_prev_p) {
            double bp_ma = NAN;
            double sp_ma = NAN;
            const bool has_bp_ma = bp_avg.update(false, 0.0, &bp_ma);
            const bool has_sp_ma = sp_avg.update(false, 0.0, &sp_ma);
            double di = NAN;
            const bool has_di = finalize_di(has_bp_ma, bp_ma, has_sp_ma, sp_ma, &di);
            double signal = NAN;
            const bool has_signal = signal_avg.update(has_di && isfinite(di), di, &signal);

            row_di[i] = has_di ? di : NAN;
            row_signal[i] = has_signal ? signal : NAN;
            prev_p = p;
            has_prev_p = true;
            continue;
        }

        double bp = NAN;
        double sp = NAN;
        const bool has_bp_sp =
            compute_bp_sp(has_volume_avg, volume_avg_now, v, p, prev_p, h0, l0, &bp, &sp);

        double bp_ma = NAN;
        double sp_ma = NAN;
        const bool has_bp_ma = bp_avg.update(has_bp_sp, bp, &bp_ma);
        const bool has_sp_ma = sp_avg.update(has_bp_sp, sp, &sp_ma);

        double di = NAN;
        const bool has_di = finalize_di(has_bp_ma, bp_ma, has_sp_ma, sp_ma, &di);
        double signal = NAN;
        const bool has_signal = signal_avg.update(has_di && isfinite(di), di, &signal);

        row_di[i] = has_di ? di : NAN;
        row_signal[i] = has_signal ? signal : NAN;
        prev_p = p;
        has_prev_p = true;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 3
//
// CPU reference: src/indicators/demand_index.rs:634
// (demand_index_with_kernel). The column this emits is demand_index, which is
// what output_id == "value" resolves to (dispatch/cpu_batch.rs:13669-13672).
//
// SHAPE: one thread per combo, bars ascending. FORCED sequential -- three
// moving averages carried across bars (the volume average, and the buying and
// selling pressure averages) plus the previous bar's price sum, and the CPU
// FEEDS AN INVALID BAR INTO EACH AVERAGE as an explicit "no value" step rather
// than skipping it, so the ring positions advance even on a hole. Skipping
// would desynchronise every later window.
//
// PERIOD-INVARIANT. compute_demand_index_batch (cpu_batch.rs:13647-13650)
// reads len_bs, len_bs_ma, len_di_ma and ma_type and NEVER period, so five
// swept periods give five identical CPU columns and this kernel emits five
// identical rows. All four CPU defaults are pinned below, ma_type "ema" among
// them.
//
// WHAT IS DELIBERATELY ABSENT: the signal average (len_di_ma). It consumes the
// demand index, it does not produce it. len_di_ma is still VALIDATED, because
// the CPU refuses the whole computation for a non-positive one.
//
// THE SCRATCH IS PER-THREAD, so its size is a property of THIS COMPILED
// KERNEL. The three averages need len_bs * 2 + len_bs_ma * 4 slots (a value
// and a validity flag each); at the pinned 19 that is 114, far under the bound
// below, which is checked rather than assumed.
//
// FIRST VALID IS NOT READ: the CPU emits from bar 0 and handles holes inside
// the loop rather than by skipping a prefix, so there is no warmup index. The
// lane row declares F64FirstValidRule::Ignored.
//
// f64 END TO END: double literals, double exp/fabs, no f32-suffixed math
// function, no fast-math intrinsic. normalize_h0_l0 floors the range at 1e-12,
// an f64-sized guard against dividing by a zero bar range -- not an f32
// epsilon carried forward (rule 2).
// ---------------------------------------------------------------------------

#define NEO_DEMAND_LEN_BS 19
#define NEO_DEMAND_LEN_BS_MA 19
#define NEO_DEMAND_LEN_DI_MA 19
#define NEO_DEMAND_MA_CODE MA_EMA
#define NEO_DEMAND_SCRATCH_CAP 1024

extern "C" __global__ void demand_index_neo_batch_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    const double* __restrict__ volume,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    const int row_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row_idx >= n_combos || n <= 0) {
        return;
    }
    (void)periods;
    (void)first_valid;

    double* row = out + static_cast<size_t>(row_idx) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = NAN;
    }

    const int len_bs = NEO_DEMAND_LEN_BS;
    const int len_bs_ma = NEO_DEMAND_LEN_BS_MA;
    const int len_di_ma = NEO_DEMAND_LEN_DI_MA;
    const int ma_code = NEO_DEMAND_MA_CODE;

    if (len_bs <= 0 || len_bs_ma <= 0 || len_di_ma <= 0) {
        return;
    }
    if (ma_code < MA_EMA || ma_code > MA_RMA) {
        return;
    }

    const int needed = len_bs * 2 + len_bs_ma * 4;
    if (needed > NEO_DEMAND_SCRATCH_CAP) {
        return;
    }

    double scratch[NEO_DEMAND_SCRATCH_CAP];
    int offset = 0;

    double* volume_values = scratch + offset;
    offset += len_bs;
    double* volume_valid = scratch + offset;
    offset += len_bs;

    double* bp_values = scratch + offset;
    offset += len_bs_ma;
    double* bp_valid = scratch + offset;
    offset += len_bs_ma;

    double* sp_values = scratch + offset;
    offset += len_bs_ma;
    double* sp_valid = scratch + offset;

    MaState volume_avg;
    MaState bp_avg;
    MaState sp_avg;
    volume_avg.init(ma_code, len_bs, volume_values, volume_valid);
    bp_avg.init(ma_code, len_bs_ma, bp_values, bp_valid);
    sp_avg.init(ma_code, len_bs_ma, sp_values, sp_valid);

    double h0 = NAN;
    double l0 = NAN;
    bool has_h0_l0 = false;
    double prev_p = NAN;
    bool has_prev_p = false;

    for (int i = 0; i < n; ++i) {
        const double h = high[i];
        const double l = low[i];
        const double c = close[i];
        const double v = volume[i];

        if (!valid_ohlcv(h, l, c, v)) {
            double unused = NAN;
            volume_avg.update(false, 0.0, &unused);

            double bp_ma = NAN;
            double sp_ma = NAN;
            const bool has_bp_ma = bp_avg.update(false, 0.0, &bp_ma);
            const bool has_sp_ma = sp_avg.update(false, 0.0, &sp_ma);

            double di = NAN;
            const bool has_di = finalize_di(has_bp_ma, bp_ma, has_sp_ma, sp_ma, &di);
            row[i] = has_di ? di : NAN;
            has_prev_p = false;
            continue;
        }

        if (!has_h0_l0) {
            h0 = h;
            l0 = l;
            has_h0_l0 = true;
        }

        double volume_avg_now = NAN;
        const bool has_volume_avg = volume_avg.update(true, v, &volume_avg_now);
        const double p = h + l + 2.0 * c;

        if (!has_prev_p) {
            double bp_ma = NAN;
            double sp_ma = NAN;
            const bool has_bp_ma = bp_avg.update(false, 0.0, &bp_ma);
            const bool has_sp_ma = sp_avg.update(false, 0.0, &sp_ma);
            double di = NAN;
            const bool has_di = finalize_di(has_bp_ma, bp_ma, has_sp_ma, sp_ma, &di);
            row[i] = has_di ? di : NAN;
            prev_p = p;
            has_prev_p = true;
            continue;
        }

        double bp = NAN;
        double sp = NAN;
        const bool has_bp_sp =
            compute_bp_sp(has_volume_avg, volume_avg_now, v, p, prev_p, h0, l0, &bp, &sp);

        double bp_ma = NAN;
        double sp_ma = NAN;
        const bool has_bp_ma = bp_avg.update(has_bp_sp, bp, &bp_ma);
        const bool has_sp_ma = sp_avg.update(has_bp_sp, sp, &sp_ma);

        double di = NAN;
        const bool has_di = finalize_di(has_bp_ma, bp_ma, has_sp_ma, sp_ma, &di);
        row[i] = has_di ? di : NAN;
        prev_p = p;
        has_prev_p = true;
    }
}
