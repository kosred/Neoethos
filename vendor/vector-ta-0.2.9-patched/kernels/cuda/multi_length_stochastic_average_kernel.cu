#include <cmath>
#include <cstddef>

namespace {
constexpr int METHOD_NONE = 0;
constexpr int METHOD_SMA = 1;
constexpr int METHOD_TMA = 2;
constexpr int METHOD_LSMA = 3;
constexpr int MIN_STOCH_LENGTH = 4;
constexpr double FLOAT_TOL = 1e-12;

struct SmaState {
    double* ring;
    int length;
    int head;
    int count;
    double sum;

    __device__ void init(double* storage, int len) {
        ring = storage;
        length = len;
        head = 0;
        count = 0;
        sum = 0.0;
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum = 0.0;
    }

    __device__ bool update(double value, double* out) {
        if (count == length) {
            sum -= ring[head];
        } else {
            count += 1;
        }
        ring[head] = value;
        sum += value;
        head += 1;
        if (head == length) {
            head = 0;
        }
        if (count == length) {
            *out = sum / static_cast<double>(length);
            return true;
        }
        *out = NAN;
        return false;
    }
};

struct LsmaState {
    double* ring;
    int length;
    int head;
    int count;
    double sum_y;
    double sum_xy;
    double x_sum;
    double denom;

    __device__ void init(double* storage, int len) {
        ring = storage;
        length = len;
        head = 0;
        count = 0;
        sum_y = 0.0;
        sum_xy = 0.0;
        const double n_f = static_cast<double>(len);
        x_sum = static_cast<double>((len * (len - 1)) / 2);
        const double x2_sum = static_cast<double>((len * (len - 1) * (2 * len - 1)) / 6);
        denom = n_f * x2_sum - x_sum * x_sum;
    }

    __device__ void reset() {
        head = 0;
        count = 0;
        sum_y = 0.0;
        sum_xy = 0.0;
    }

    __device__ double endpoint() const {
        const double n = static_cast<double>(length);
        const double slope = (n * sum_xy - x_sum * sum_y) / denom;
        const double intercept = (sum_y - slope * x_sum) / n;
        return intercept + slope * static_cast<double>(length - 1);
    }

    __device__ bool update(double value, double* out) {
        if (count < length) {
            const int idx = count;
            ring[head] = value;
            head += 1;
            if (head == length) {
                head = 0;
            }
            count += 1;
            sum_y += value;
            sum_xy += static_cast<double>(idx) * value;
            if (count == length) {
                *out = endpoint();
                return true;
            }
            *out = NAN;
            return false;
        }

        const double old = ring[head];
        const double old_sum_y = sum_y;
        ring[head] = value;
        head += 1;
        if (head == length) {
            head = 0;
        }
        sum_y = old_sum_y - old + value;
        sum_xy = sum_xy - (old_sum_y - old) + static_cast<double>(length - 1) * value;
        *out = endpoint();
        return true;
    }
};

struct SmoothingState {
    int method;
    SmaState sma_1;
    SmaState sma_2;
    LsmaState lsma;

    __device__ void init(int method_code, int length, double* storage) {
        method = method_code;
        if (method == METHOD_SMA) {
            sma_1.init(storage, length);
        } else if (method == METHOD_TMA) {
            sma_1.init(storage, length);
            sma_2.init(storage + length, length);
        } else if (method == METHOD_LSMA) {
            lsma.init(storage, length);
        }
    }

    __device__ void reset() {
        if (method == METHOD_SMA) {
            sma_1.reset();
        } else if (method == METHOD_TMA) {
            sma_1.reset();
            sma_2.reset();
        } else if (method == METHOD_LSMA) {
            lsma.reset();
        }
    }

    __device__ bool update(double value, double* out) {
        if (method == METHOD_NONE) {
            *out = value;
            return true;
        }
        if (method == METHOD_SMA) {
            return sma_1.update(value, out);
        }
        if (method == METHOD_TMA) {
            double inner = NAN;
            if (!sma_1.update(value, &inner)) {
                *out = NAN;
                return false;
            }
            return sma_2.update(inner, out);
        }
        return lsma.update(value, out);
    }
};
}

extern "C" __global__ void multi_length_stochastic_average_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ presmooths,
    const int* __restrict__ postsmooths,
    const int* __restrict__ premethods,
    const int* __restrict__ postmethods,
    int rows,
    int scratch_cap,
    double* __restrict__ scratch_buf,
    double* __restrict__ out_values
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length = lengths[row];
    const int presmooth = presmooths[row];
    const int postsmooth = postsmooths[row];
    const int premethod = premethods[row];
    const int postmethod = postmethods[row];

    double* row_out = out_values + static_cast<size_t>(row) * static_cast<size_t>(len);
    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    if (length < MIN_STOCH_LENGTH || presmooth <= 0 || postsmooth <= 0) {
        return;
    }
    if (premethod < METHOD_NONE || premethod > METHOD_LSMA || postmethod < METHOD_NONE ||
        postmethod > METHOD_LSMA) {
        return;
    }

    double* row_scratch = scratch_buf + static_cast<size_t>(row) * static_cast<size_t>(scratch_cap);
    int offset = 0;
    double* pre_storage = row_scratch + offset;
    if (premethod == METHOD_SMA || premethod == METHOD_LSMA) {
        offset += presmooth;
    } else if (premethod == METHOD_TMA) {
        offset += presmooth * 2;
    }
    double* post_storage = row_scratch + offset;
    if (postmethod == METHOD_SMA || postmethod == METHOD_LSMA) {
        offset += postsmooth;
    } else if (postmethod == METHOD_TMA) {
        offset += postsmooth * 2;
    }
    double* main_ring = row_scratch + offset;
    offset += length;
    if (offset > scratch_cap) {
        return;
    }

    SmoothingState pre_state;
    SmoothingState post_state;
    pre_state.init(premethod, presmooth, pre_storage);
    post_state.init(postmethod, postsmooth, post_storage);

    int main_head = 0;
    int main_count = 0;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            pre_state.reset();
            post_state.reset();
            main_head = 0;
            main_count = 0;
            continue;
        }

        double pre_value = NAN;
        if (!pre_state.update(value, &pre_value)) {
            continue;
        }

        main_ring[main_head] = pre_value;
        main_head += 1;
        if (main_head == length) {
            main_head = 0;
        }
        if (main_count < length) {
            main_count += 1;
        }
        if (main_count < length) {
            continue;
        }

        const int newest = (main_head + length - 1) % length;
        const double current = main_ring[newest];
        double min_value = current;
        double max_value = current;
        int idx = newest;
        double sum = 0.0;
        bool invalid = false;

        for (int window = 1; window <= length; ++window) {
            const double ring_value = main_ring[idx];
            if (ring_value < min_value) {
                min_value = ring_value;
            }
            if (ring_value > max_value) {
                max_value = ring_value;
            }
            if (window >= MIN_STOCH_LENGTH) {
                const double denom = max_value - min_value;
                if (fabs(denom) <= FLOAT_TOL) {
                    post_state.reset();
                    invalid = true;
                    break;
                }
                sum += (current - min_value) / denom;
            }
            idx = (idx == 0) ? (length - 1) : (idx - 1);
        }

        if (invalid) {
            continue;
        }

        const double norm =
            (sum / static_cast<double>(length - (MIN_STOCH_LENGTH - 1))) * 100.0;
        double post_value = NAN;
        if (!post_state.update(norm, &post_value)) {
            continue;
        }
        row_out[i] = post_value;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 1, round 3
//
// CPU REFERENCE: `multi_length_stochastic_average_with_kernel`
// (src/indicators/multi_length_stochastic_average.rs:876) ->
// `multi_length_stochastic_average_row_from_slice` (:743), which at the pinned
// defaults takes the `multi_length_stochastic_average_default_sma_finite`
// branch (:770).
//
// WHY A SECOND ENTRY POINT IN THIS FILE
//
// `multi_length_stochastic_average_batch_f64` (:174) is double-clean but
// declares eleven parameters -- five `const int*` per-row parameter arrays and
// a host-allocated `double* scratch_buf`. The f64 lane launches ONE shape:
//   (series..., int n, const int* periods, int n_combos, int first_valid,
//    double* out)
// and has no scratch pointer to give, so the lane gets its own entry point
// here. Every ring it needs is a fixed-size PER-THREAD array, sized by the
// pinned defaults below: 10 + 10 + 14 = 34 doubles, 272 bytes. Bounded at
// compile time, not allocated.
//
// WHICH COLUMN: the single `value` series. `compute_multi_length_stochastic_
// average_batch` calls `expect_value_output` (cpu_batch.rs:8669), so `value` is
// the only output id this indicator has.
//
// SHAPE: one thread per combo, bars ascending. Three carried states -- the
// presmooth SMA, the 14-slot stochastic ring and the postsmooth SMA -- and the
// CPU RESETS the postsmooth accumulator mid-bar when the stochastic denominator
// collapses (:838-843), so a bar-parallel form cannot know the postsmooth
// window's contents.
//
// PERIOD-INVARIANT: the CPU batch reads `source`, `length`, `presmooth`,
// `premethod`, `postsmooth` and `postmethod` and NEVER `period`
// (cpu_batch.rs:8689-8707), so every swept period gives the same CPU column and
// this kernel writes identical rows. The pinned values are the CPU defaults:
// length 14 (:31), presmooth 10 (:33), postsmooth 10 (:34), both methods `sma`
// (:35), source `close` (:32).
//
// ROUNDING: both SMAs are the CPU's RUNNING sums in the CPU's order --
// subtract the departing sample first, then store, then add the arriving one
// (:787-794, :851-856) -- so the accumulator carries the same rounding history.
// The stochastic sum accumulates `(current - min) / (max - min)` bar by bar
// from the newest sample backwards (:830-850), which is the order reproduced
// here. No `fma`: the CPU writes no `mul_add` anywhere in this indicator.
//
// EPSILON: `FLOAT_TOL` is 1e-12 (:37, and :10 in this file) and is ALREADY
// f64-sized -- it is the CPU's own constant, not an f32 epsilon carried over.
// Left exactly as the CPU writes it.
//
// f64 END TO END: no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. `fabs` is the double overload. The NaN is a DOUBLE quiet-NaN bit
// pattern.
//
// FIRST VALID IS NOT READ: the CPU stream restarts every accumulator at any
// non-finite bar (:761-765 via `stream.update`), so one global warmup index
// would be wrong after the first hole. The lane row declares
// `F64FirstValidRule::Ignored`.
// ---------------------------------------------------------------------------

#define NEO_MLSA_LENGTH 14
#define NEO_MLSA_PRESMOOTH 10
#define NEO_MLSA_POSTSMOOTH 10

__device__ inline double neo_mlsa_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void multi_length_stochastic_average_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    const int combo = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) {
        return;
    }
    (void)periods;
    (void)first_valid;

    const double nan_value = neo_mlsa_qnan();
    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);
    for (int i = 0; i < n; ++i) {
        row[i] = nan_value;
    }

    double pre_ring[NEO_MLSA_PRESMOOTH];
    double stoch_ring[NEO_MLSA_LENGTH];
    double post_ring[NEO_MLSA_POSTSMOOTH];

    int pre_head = 0;
    int pre_count = 0;
    double pre_sum = 0.0;
    int stoch_head = 0;
    int stoch_count = 0;
    int post_head = 0;
    int post_count = 0;
    double post_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            // `MultiLengthStochasticAverageStream::update` resets every
            // accumulator on a non-finite bar.
            pre_head = 0;
            pre_count = 0;
            pre_sum = 0.0;
            stoch_head = 0;
            stoch_count = 0;
            post_head = 0;
            post_count = 0;
            post_sum = 0.0;
            continue;
        }

        // Presmooth SMA -- :786-800.
        if (pre_count == NEO_MLSA_PRESMOOTH) {
            pre_sum -= pre_ring[pre_head];
        } else {
            pre_count += 1;
        }
        pre_ring[pre_head] = value;
        pre_sum += value;
        pre_head += 1;
        if (pre_head == NEO_MLSA_PRESMOOTH) {
            pre_head = 0;
        }
        if (pre_count < NEO_MLSA_PRESMOOTH) {
            continue;
        }
        const double pre = pre_sum / static_cast<double>(NEO_MLSA_PRESMOOTH);

        // Stochastic ring -- :803-813.
        stoch_ring[stoch_head] = pre;
        stoch_head += 1;
        if (stoch_head == NEO_MLSA_LENGTH) {
            stoch_head = 0;
        }
        if (stoch_count < NEO_MLSA_LENGTH) {
            stoch_count += 1;
            if (stoch_count < NEO_MLSA_LENGTH) {
                continue;
            }
        }

        // Multi-length stochastic average -- :815-853.
        const int newest = (stoch_head == 0) ? (NEO_MLSA_LENGTH - 1) : (stoch_head - 1);
        const double current = stoch_ring[newest];
        double min_value = current;
        double max_value = current;
        int idx = newest;
        double sum = 0.0;
        bool valid_norm = true;

        for (int window = 1; window <= NEO_MLSA_LENGTH; ++window) {
            const double sample = stoch_ring[idx];
            if (sample < min_value) {
                min_value = sample;
            }
            if (sample > max_value) {
                max_value = sample;
            }
            if (window >= MIN_STOCH_LENGTH) {
                const double denom = max_value - min_value;
                if (fabs(denom) <= FLOAT_TOL) {
                    post_head = 0;
                    post_count = 0;
                    post_sum = 0.0;
                    valid_norm = false;
                    break;
                }
                sum += (current - min_value) / denom;
            }
            idx = (idx == 0) ? (NEO_MLSA_LENGTH - 1) : (idx - 1);
        }
        if (!valid_norm) {
            continue;
        }

        // CPU: `sum / (LENGTH - (MIN_STOCH_LENGTH - 1)) as f64 * 100.0` (:857)
        // -- divide FIRST, then scale. Two roundings, in that order.
        const double norm =
            sum / static_cast<double>(NEO_MLSA_LENGTH - (MIN_STOCH_LENGTH - 1)) * 100.0;

        // Postsmooth SMA -- :858-871.
        if (post_count == NEO_MLSA_POSTSMOOTH) {
            post_sum -= post_ring[post_head];
        } else {
            post_count += 1;
        }
        post_ring[post_head] = norm;
        post_sum += norm;
        post_head += 1;
        if (post_head == NEO_MLSA_POSTSMOOTH) {
            post_head = 0;
        }
        if (post_count == NEO_MLSA_POSTSMOOTH) {
            row[i] = post_sum / static_cast<double>(NEO_MLSA_POSTSMOOTH);
        }
    }
}
