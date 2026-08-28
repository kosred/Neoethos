#include <cmath>
#include <cstddef>

namespace {
constexpr double PI = 3.14;

__device__ inline double nz(double value) {
    return isfinite(value) ? value : 0.0;
}

struct MesaLineState {
    double* ring;
    int length;
    int head;
    int count;
    double prev_1;
    double prev_2;

    __device__ void init(double* storage, int line_length) {
        ring = storage;
        length = line_length;
        head = 0;
        count = 0;
        prev_1 = NAN;
        prev_2 = NAN;
    }

    __device__ double update(double filt, double c1, double c2, double c3) {
        const double filt_nz = nz(filt);
        if (count < length) {
            ring[count] = filt_nz;
            count += 1;
        } else {
            ring[head] = filt_nz;
            head += 1;
            if (head == length) {
                head = 0;
            }
        }

        double out = NAN;
        if (isfinite(filt)) {
            double highest = filt;
            double lowest = filt;
            for (int i = 0; i < count; ++i) {
                const double value = ring[i];
                if (value > highest) {
                    highest = value;
                }
                if (value < lowest) {
                    lowest = value;
                }
            }
            if (count < length) {
                if (0.0 > highest) {
                    highest = 0.0;
                }
                if (0.0 < lowest) {
                    lowest = 0.0;
                }
            }
            const double denom = highest - lowest;
            if (denom != 0.0 && isfinite(denom)) {
                const double stoc = (filt - lowest) / denom;
                if (isfinite(stoc)) {
                    out = fma(c1, stoc, fma(c2, nz(prev_1), c3 * nz(prev_2)));
                }
            }
        }

        prev_2 = prev_1;
        prev_1 = out;
        return out;
    }
};

struct RollingSmaState {
    double* ring;
    int length;
    int head;
    int count;
    int finite_count;
    double finite_sum;

    __device__ void init(double* storage, int window_length) {
        ring = storage;
        length = window_length;
        head = 0;
        count = 0;
        finite_count = 0;
        finite_sum = 0.0;
    }

    __device__ double update(double value) {
        if (count < length) {
            ring[count] = value;
            if (isfinite(value)) {
                finite_sum += value;
                finite_count += 1;
            }
            count += 1;
        } else {
            const double old = ring[head];
            if (isfinite(old)) {
                finite_sum -= old;
                finite_count -= 1;
            }
            ring[head] = value;
            if (isfinite(value)) {
                finite_sum += value;
                finite_count += 1;
            }
            head += 1;
            if (head == length) {
                head = 0;
            }
        }

        if (count == length && finite_count == length) {
            return finite_sum / static_cast<double>(length);
        }
        return NAN;
    }
};
}

extern "C" __global__ void mesa_stochastic_multi_length_batch_f64(
    const double* __restrict__ source,
    int len,
    const int* __restrict__ length_1s,
    const int* __restrict__ length_2s,
    const int* __restrict__ length_3s,
    const int* __restrict__ length_4s,
    const int* __restrict__ trigger_lengths,
    int rows,
    int max_length,
    int max_trigger_length,
    double* __restrict__ mesa_ring_buf,
    double* __restrict__ trigger_ring_buf,
    double* __restrict__ out_mesa_1,
    double* __restrict__ out_mesa_2,
    double* __restrict__ out_mesa_3,
    double* __restrict__ out_mesa_4,
    double* __restrict__ out_trigger_1,
    double* __restrict__ out_trigger_2,
    double* __restrict__ out_trigger_3,
    double* __restrict__ out_trigger_4
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int length_1 = length_1s[row];
    const int length_2 = length_2s[row];
    const int length_3 = length_3s[row];
    const int length_4 = length_4s[row];
    const int trigger_length = trigger_lengths[row];

    double* row_out_mesa_1 = out_mesa_1 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_mesa_2 = out_mesa_2 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_mesa_3 = out_mesa_3 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_mesa_4 = out_mesa_4 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_trigger_1 =
        out_trigger_1 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_trigger_2 =
        out_trigger_2 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_trigger_3 =
        out_trigger_3 + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_trigger_4 =
        out_trigger_4 + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_mesa_1[i] = NAN;
        row_out_mesa_2[i] = NAN;
        row_out_mesa_3[i] = NAN;
        row_out_mesa_4[i] = NAN;
        row_out_trigger_1[i] = NAN;
        row_out_trigger_2[i] = NAN;
        row_out_trigger_3[i] = NAN;
        row_out_trigger_4[i] = NAN;
    }

    if (length_1 <= 0 || length_2 <= 0 || length_3 <= 0 || length_4 <= 0 ||
        trigger_length <= 0 || length_1 > max_length || length_2 > max_length ||
        length_3 > max_length || length_4 > max_length || trigger_length > max_trigger_length) {
        return;
    }

    double* mesa_base =
        mesa_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_length) * 4u;
    double* trigger_base =
        trigger_ring_buf + static_cast<size_t>(row) * static_cast<size_t>(max_trigger_length) * 4u;

    MesaLineState mesa_1_state;
    MesaLineState mesa_2_state;
    MesaLineState mesa_3_state;
    MesaLineState mesa_4_state;
    mesa_1_state.init(mesa_base, length_1);
    mesa_2_state.init(mesa_base + max_length, length_2);
    mesa_3_state.init(mesa_base + max_length * 2, length_3);
    mesa_4_state.init(mesa_base + max_length * 3, length_4);

    RollingSmaState trigger_1_state;
    RollingSmaState trigger_2_state;
    RollingSmaState trigger_3_state;
    RollingSmaState trigger_4_state;
    trigger_1_state.init(trigger_base, trigger_length);
    trigger_2_state.init(trigger_base + max_trigger_length, trigger_length);
    trigger_3_state.init(trigger_base + max_trigger_length * 2, trigger_length);
    trigger_4_state.init(trigger_base + max_trigger_length * 3, trigger_length);

    const double alpha1 =
        ((cos(0.707 * 2.0 * PI / 48.0) + sin(0.707 * 2.0 * PI / 48.0) - 1.0) /
         cos(0.707 * 2.0 * PI / 48.0));
    const double one_minus_alpha = 1.0 - alpha1;
    const double hp_coef = (1.0 - alpha1 * 0.5) * (1.0 - alpha1 * 0.5);
    const double a1 = exp(-1.414 * PI / 10.0);
    const double b1 = 2.0 * a1 * cos(1.414 * PI / 10.0);
    const double c2 = b1;
    const double c3 = -(a1 * a1);
    const double c1 = 1.0 - c2 - c3;
    const double hp_feedback_1 = 2.0 * one_minus_alpha;
    const double hp_feedback_2 = -(one_minus_alpha * one_minus_alpha);

    double prev_src_1 = NAN;
    double prev_src_2 = NAN;
    double prev_hp_1 = NAN;
    double prev_hp_2 = NAN;
    double prev_filt_1 = NAN;
    double prev_filt_2 = NAN;

    for (int i = 0; i < len; ++i) {
        const double value = source[i];
        const double hp = isfinite(value)
            ? fma(
                  hp_coef,
                  value - 2.0 * nz(prev_src_1) + nz(prev_src_2),
                  fma(hp_feedback_1, nz(prev_hp_1), hp_feedback_2 * nz(prev_hp_2)))
            : NAN;
        const double filt = isfinite(hp)
            ? fma(c1, hp, fma(c2, nz(prev_filt_1), c3 * nz(prev_filt_2)))
            : NAN;

        prev_src_2 = prev_src_1;
        prev_src_1 = value;
        prev_hp_2 = prev_hp_1;
        prev_hp_1 = hp;
        prev_filt_2 = prev_filt_1;
        prev_filt_1 = filt;

        const double mesa_1 = mesa_1_state.update(filt, c1, c2, c3);
        const double mesa_2 = mesa_2_state.update(filt, c1, c2, c3);
        const double mesa_3 = mesa_3_state.update(filt, c1, c2, c3);
        const double mesa_4 = mesa_4_state.update(filt, c1, c2, c3);
        const double trigger_1 = trigger_1_state.update(mesa_1);
        const double trigger_2 = trigger_2_state.update(mesa_2);
        const double trigger_3 = trigger_3_state.update(mesa_3);
        const double trigger_4 = trigger_4_state.update(mesa_4);

        row_out_mesa_1[i] = mesa_1;
        row_out_mesa_2[i] = mesa_2;
        row_out_mesa_3[i] = mesa_3;
        row_out_mesa_4[i] = mesa_4;
        row_out_trigger_1[i] = trigger_1;
        row_out_trigger_2[i] = trigger_2;
        row_out_trigger_3[i] = trigger_3;
        row_out_trigger_4[i] = trigger_4;
    }
}

// ---------------------------------------------------------------------------
// NEOETHOS f64 LANE  --  closer 1, round 3
//
// CPU REFERENCE: `mesa_stochastic_multi_length_with_kernel`
// (src/indicators/mesa_stochastic_multi_length.rs:663) ->
// `MesaStochasticMultiLengthStream::update` (:588), whose two stages are
// `SharedFilterState::update` (:522) and `MesaLineState::update` (:430).
//
// WHY A SECOND ENTRY POINT IN THIS FILE
//
// `mesa_stochastic_multi_length_batch_f64` (:127) is double-clean but declares
// twenty parameters -- five `const int*` per-row parameter arrays, two
// host-allocated ring scratch buffers and EIGHT output matrices. The f64 lane
// launches ONE shape:
//   (series..., int n, const int* periods, int n_combos, int first_valid,
//    double* out)
// and reads back ONE matrix, so the lane gets its own entry point here. The one
// ring it needs is a fixed-size PER-THREAD array of `length_1` = 48 doubles,
// 384 bytes, bounded at compile time.
//
// WHICH COLUMN: `mesa_1`. `compute_mesa_stochastic_multi_length_batch`
// (cpu_batch.rs:10604-10629) accepts eight output ids -- `mesa_1..4` and
// `trigger_1..4` -- and has NO `value` alias, so a parity run must name the
// column. `mesa_1` is the longest line (`length_1` = 48, the CPU default at
// cpu_batch.rs:10577) and the one the other three are read against; the four
// triggers are each an SMA OF a mesa line, so nothing is lost by not computing
// them here. Precedent for naming rather than inventing a `value`:
// `ict_propulsion_block_neo_batch_f64`, which emits `bullish_high`.
//
// SHAPE: one thread per combo, bars ascending. Six carried scalars in the
// shared two-pole high-pass / super-smoother cascade (`prev_src_1/2`,
// `prev_hp_1/2`, `prev_filt_1/2`, :488-493) plus the mesa line's own
// `prev_1`/`prev_2` and its 48-slot ring. Two IIRs in series -- there is no
// bar-parallel form.
//
// PERIOD-INVARIANT: the CPU batch reads `source`, `length_1..4` and
// `trigger_length` and NEVER `period` (cpu_batch.rs:10576-10582), so every
// swept period gives the same CPU column and this kernel writes identical rows.
// `length_1` is pinned at the CPU default 48.
//
// ROUNDING: every step is the CPU's `mul_add` nesting reproduced as `fma`,
// operand for operand:
//   hp   = fma(hp_coef, src - 2*nz(p1) + nz(p2),
//              fma(hp_fb1, nz(hp1), hp_fb2 * nz(hp2)))        -- CPU :524-529
//   filt = fma(c1, hp, fma(c2, nz(f1), c3 * nz(f2)))          -- CPU :532-537
//   out  = fma(c1, stoc, fma(c2, nz(prev_1), c3 * nz(prev_2)))-- CPU :464
// Writing these as separate multiplies and adds would be three roundings where
// the CPU has two, on a value that feeds its own recurrence.
//
// PI IS 3.14, NOT M_PI. `const PI: f64 = 3.14` (:30) is the CPU's own
// constant -- Ehlers' original script writes 3.14 and the crate reproduces it.
// Substituting `M_PI` here would compute a DIFFERENT indicator, so the file's
// `constexpr double PI = 3.14` (:5) is correct and is reused.
//
// f64 END TO END: no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic, no epsilon. `cos`, `sin`, `exp` are the double overloads. The NaN
// is a DOUBLE quiet-NaN bit pattern.
//
// FIRST VALID IS NOT READ: the CPU never resets -- a non-finite bar makes `hp`
// and `filt` NaN for that bar and `nz()` (:6 here, :410 there) folds the NaN
// history to 0.0 on the next one, so the series is defined from bar 0 with no
// warmup index. The lane row declares `F64FirstValidRule::Ignored`.
// ---------------------------------------------------------------------------

#define NEO_MESA_LENGTH_1 48

__device__ inline double neo_mesa_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void mesa_stochastic_multi_length_neo_batch_f64(
    const double* __restrict__ source,
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

    const double nan_value = neo_mesa_qnan();
    double* row = out + static_cast<size_t>(combo) * static_cast<size_t>(n);

    double mesa_ring[NEO_MESA_LENGTH_1];
    MesaLineState mesa_1_state;
    mesa_1_state.init(mesa_ring, NEO_MESA_LENGTH_1);

    // `SharedFilterState::new`, :498-506. Identical expressions, same order.
    const double alpha1 =
        ((cos(0.707 * 2.0 * PI / 48.0) + sin(0.707 * 2.0 * PI / 48.0) - 1.0) /
         cos(0.707 * 2.0 * PI / 48.0));
    const double one_minus_alpha = 1.0 - alpha1;
    const double hp_coef = (1.0 - alpha1 * 0.5) * (1.0 - alpha1 * 0.5);
    const double a1 = exp(-1.414 * PI / 10.0);
    const double b1 = 2.0 * a1 * cos(1.414 * PI / 10.0);
    const double c2 = b1;
    const double c3 = -(a1 * a1);
    const double c1 = 1.0 - c2 - c3;
    const double hp_feedback_1 = 2.0 * one_minus_alpha;
    const double hp_feedback_2 = -(one_minus_alpha * one_minus_alpha);

    double prev_src_1 = nan_value;
    double prev_src_2 = nan_value;
    double prev_hp_1 = nan_value;
    double prev_hp_2 = nan_value;
    double prev_filt_1 = nan_value;
    double prev_filt_2 = nan_value;

    for (int i = 0; i < n; ++i) {
        const double value = source[i];
        const double hp = isfinite(value)
            ? fma(
                  hp_coef,
                  value - 2.0 * nz(prev_src_1) + nz(prev_src_2),
                  fma(hp_feedback_1, nz(prev_hp_1), hp_feedback_2 * nz(prev_hp_2)))
            : nan_value;
        const double filt = isfinite(hp)
            ? fma(c1, hp, fma(c2, nz(prev_filt_1), c3 * nz(prev_filt_2)))
            : nan_value;

        prev_src_2 = prev_src_1;
        prev_src_1 = value;
        prev_hp_2 = prev_hp_1;
        prev_hp_1 = hp;
        prev_filt_2 = prev_filt_1;
        prev_filt_1 = filt;

        row[i] = mesa_1_state.update(filt, c1, c2, c3);
    }
}
