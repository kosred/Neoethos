#include <cmath>
#include <cstddef>

namespace {
__device__ inline bool is_valid_source(double value) {
    return isfinite(value);
}

__device__ inline double nz(double value) {
    return isfinite(value) ? value : 0.0;
}

__device__ inline double biased_std_from_sums(double sum, double sum_sq, int period) {
    const double n = static_cast<double>(period);
    const double centered = fmax(sum_sq - (sum * sum) / n, 0.0);
    return sqrt(centered / n);
}

__device__ inline void push_window_sum_sumsq(
    double* window,
    int window_len,
    int* head,
    int* count,
    int* valid,
    double* sum,
    double* sum_sq,
    double value
) {
    if (*count == window_len) {
        const double old = window[*head];
        if (isfinite(old)) {
            *valid -= 1;
            *sum -= old;
            *sum_sq -= old * old;
        }
    } else {
        *count += 1;
    }

    window[*head] = value;
    *head += 1;
    if (*head == window_len) {
        *head = 0;
    }

    if (isfinite(value)) {
        *valid += 1;
        *sum += value;
        *sum_sq += value * value;
    }
}

__device__ inline void push_window_sum(
    double* window,
    int window_len,
    int* head,
    int* count,
    int* valid,
    double* sum,
    double value
) {
    if (*count == window_len) {
        const double old = window[*head];
        if (isfinite(old)) {
            *valid -= 1;
            *sum -= old;
        }
    } else {
        *count += 1;
    }

    window[*head] = value;
    *head += 1;
    if (*head == window_len) {
        *head = 0;
    }

    if (isfinite(value)) {
        *valid += 1;
        *sum += value;
    }
}
}

extern "C" __global__ void volatility_ratio_adaptive_rsx_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ periods,
    const double* __restrict__ speeds,
    int rows,
    double* __restrict__ out_line,
    double* __restrict__ out_signal
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    const int period = periods[row];
    const double speed = speeds[row];

    double* row_line = out_line + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_line[i] = NAN;
        row_signal[i] = NAN;
    }

    if (period <= 0 || !isfinite(speed) || speed < 0.0 || speed > 1.0) {
        return;
    }

    double* price_window = new double[period];
    double* dev_window = new double[period];
    if (price_window == nullptr || dev_window == nullptr) {
        delete[] price_window;
        delete[] dev_window;
        return;
    }
    for (int i = 0; i < period; ++i) {
        price_window[i] = NAN;
        dev_window[i] = NAN;
    }

    double prev_src_out = NAN;
    double prev_line = NAN;
    int price_head = 0;
    int price_count = 0;
    int price_valid = 0;
    double price_sum = 0.0;
    double price_sum_sq = 0.0;
    int dev_head = 0;
    int dev_count = 0;
    int dev_valid = 0;
    double dev_sum = 0.0;
    double f28 = NAN;
    double f30 = NAN;
    double f38 = NAN;
    double f40 = NAN;
    double f48 = NAN;
    double f50 = NAN;
    double f58 = NAN;
    double f60 = NAN;
    double f68 = NAN;
    double f70 = NAN;
    double f78 = NAN;
    double f80 = NAN;

    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        const double src_out = is_valid_source(value) ? 100.0 * value : NAN;

        push_window_sum_sumsq(
            price_window,
            period,
            &price_head,
            &price_count,
            &price_valid,
            &price_sum,
            &price_sum_sq,
            value
        );

        const double dev = (price_count == period && price_valid == period)
            ? biased_std_from_sums(price_sum, price_sum_sq, period)
            : NAN;

        push_window_sum(
            dev_window,
            period,
            &dev_head,
            &dev_count,
            &dev_valid,
            &dev_sum,
            dev
        );

        const double devavg = (dev_count == period && dev_valid == period)
            ? dev_sum / static_cast<double>(period)
            : NAN;

        const double vol_ratio =
            isfinite(dev) && isfinite(devavg) && devavg != 0.0 ? dev / devavg : NAN;
        const double adaptive_len =
            isfinite(vol_ratio) && vol_ratio > 0.0
                ? trunc(static_cast<double>(period) / vol_ratio)
                : NAN;
        const double kg = isfinite(adaptive_len) ? 3.0 / (adaptive_len + 2.0) : NAN;
        const double hg = isfinite(kg) ? 1.0 - kg : NAN;

        const double mom0 =
            isfinite(src_out) && isfinite(prev_src_out) ? src_out - prev_src_out : NAN;
        const double moa0 = isfinite(mom0) ? fabs(mom0) : NAN;
        const double spdp1 = speed + 1.0;

        f28 = isfinite(kg) && isfinite(hg) && isfinite(mom0) ? kg * mom0 + hg * nz(f28) : NAN;
        f30 = isfinite(kg) && isfinite(hg) && isfinite(f28) ? hg * nz(f30) + kg * f28 : NAN;
        const double mom1 = isfinite(f28) && isfinite(f30) ? f28 * spdp1 - f30 * speed : NAN;

        f38 = isfinite(kg) && isfinite(hg) && isfinite(mom1) ? hg * nz(f38) + kg * mom1 : NAN;
        f40 = isfinite(kg) && isfinite(hg) && isfinite(f38) ? kg * f38 + hg * nz(f40) : NAN;
        const double mom2 = isfinite(f38) && isfinite(f40) ? f38 * spdp1 - f40 * speed : NAN;

        f48 = isfinite(kg) && isfinite(hg) && isfinite(mom2) ? hg * nz(f48) + kg * mom2 : NAN;
        f50 = isfinite(kg) && isfinite(hg) && isfinite(f48) ? kg * f48 + hg * nz(f50) : NAN;
        const double mom_out = isfinite(f48) && isfinite(f50) ? f48 * spdp1 - f50 * speed : NAN;

        f58 = isfinite(kg) && isfinite(hg) && isfinite(moa0) ? hg * nz(f58) + kg * moa0 : NAN;
        f60 = isfinite(kg) && isfinite(hg) && isfinite(f58) ? kg * f58 + hg * nz(f60) : NAN;
        const double moa1 = isfinite(f58) && isfinite(f60) ? f58 * spdp1 - f60 * speed : NAN;

        f68 = isfinite(kg) && isfinite(hg) && isfinite(moa1) ? hg * nz(f68) + kg * moa1 : NAN;
        f70 = isfinite(kg) && isfinite(hg) && isfinite(f68) ? kg * f68 + hg * nz(f70) : NAN;
        const double moa2 = isfinite(f68) && isfinite(f70) ? f68 * spdp1 - f70 * speed : NAN;

        f78 = isfinite(kg) && isfinite(hg) && isfinite(moa2) ? hg * nz(f78) + kg * moa2 : NAN;
        f80 = isfinite(kg) && isfinite(hg) && isfinite(f78) ? kg * f78 + hg * nz(f80) : NAN;
        const double moa_out = isfinite(f78) && isfinite(f80) ? f78 * spdp1 - f80 * speed : NAN;

        const double line = isfinite(mom_out) && isfinite(moa_out) && moa_out != 0.0
            ? fmin(fmax((mom_out / moa_out + 1.0) * 50.0, 0.0), 100.0)
            : NAN;
        const double signal = prev_line;

        row_line[i] = line;
        row_signal[i] = signal;

        prev_src_out = src_out;
        prev_line = line;
    }

    delete[] price_window;
    delete[] dev_window;
}

/* ===========================================================================
 * NEOETHOS f64 LANE  --  closer 3, round 3
 *
 * CPU REFERENCE: src/indicators/volatility_ratio_adaptive_rsx.rs
 *   VrarsxState::update (:415-600), vrarsx_compute_into (:607-621).
 *   Batch dispatcher: cpu_batch.rs:9695 -- output "value" is an ALIAS OF
 *   "line" (:9724), so this kernel emits `line`, never `signal`.
 *
 * WHY A SECOND ENTRY POINT: `volatility_ratio_adaptive_rsx_batch_f64` (:85)
 *   takes 7 parameters and emits TWO series. The lane launches
 *   (data, n, periods, n_combos, first_valid, out) and consumes ONE. Adding
 *   the lane-shaped entry beside it leaves the existing wrapper untouched.
 *
 * INPUT: one price series, CPU source `close` (cpu_batch.rs:9699) --
 *   F64InputKind::CloseSlice.
 *
 * FIRST-VALID IGNORED: `vrarsx_compute_into` walks EVERY bar from 0 and the
 *   per-bar validity is handled inside `update` (:416, the `is_valid_source`
 *   test) -- `first` is read only by `vrarsx_prepare` for the length check,
 *   never by the loop. There is no NaN prefix to place, so the caller's index
 *   is not read.
 *
 * PERIOD-SWEPT: `period` is the swept parameter (cpu_batch.rs:9707) and it
 *   sets BOTH ring depths and the adaptive length. `speed` is pinned at the
 *   CPU default 0.5 (cpu_batch.rs:9708).
 *
 * SHAPE: ONE THREAD PER COLUMN, bars ascending. Twelve interlocking scalars
 *   (f28..f80) carry across bars -- a six-stage smoothing cascade. No scan
 *   reformulation: the cascade is not associative.
 *
 * ARITHMETIC taken verbatim:
 *   * every stage is `kg * x + hg * nz(prev)` or `hg * nz(prev) + kg * x` --
 *     the CPU writes the two orders DIFFERENTLY per stage (:483, :489, :505,
 *     :509, ...) and the order is reproduced stage for stage. Two products
 *     and one add, NOT a fused prev + kg*(x - prev).
 *   * `nz` (:279) maps a non-finite carry to 0.0, and is applied to the
 *     PREVIOUS bar's value only.
 *   * `biased_std_from_sums` (:272) is
 *     `((sum_sq - sum*sum/n).max(0.0) / n).sqrt()` -- f64::max, hence fmax,
 *     which returns the non-NaN operand.
 *   * `adaptive_len` is `(period / vol_ratio).trunc()` (:460) -- trunc, not
 *     floor: the two differ on a negative quotient.
 *   * the clamp (:580) is Rust `f64::clamp`, which is an if-chain and lets a
 *     NaN through; fmin/fmax would map NaN to a bound. Written as the
 *     if-chain for that reason -- the ONE place in this file where fmax is
 *     deliberately NOT used.
 *
 * EPSILON: there is none. The CPU guards are the exact tests `devavg != 0.0`
 *   (:453) and `moa_out != 0.0` (:579). No f32-sized tolerance is imported.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Ring depth is a function of the SWEPT period, so it carries a bound and the
 * host reports it through `F64Kernel::max_period`. */
#define NEO_VRARSX_MAX_PERIOD 512
/* cpu_batch.rs:9708 */
#define NEO_VRARSX_SPEED 0.5

__device__ __forceinline__ double neo_vrarsx_nz(double v)
{
    return isfinite(v) ? v : 0.0;
}

extern "C" __global__
void volatility_ratio_adaptive_rsx_neo_batch_f64(const double* __restrict__ data,
                                                 int n,
                                                 const int* __restrict__ periods,
                                                 int n_combos,
                                                 int first_valid,
                                                 double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)first_valid; /* handled in place -- see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int period = periods[combo];
    /* vrarsx_prepare refuses period == 0 or period > len. */
    if (period <= 0 || period > n || period > NEO_VRARSX_MAX_PERIOD) return;

    double price_w[NEO_VRARSX_MAX_PERIOD];
    double dev_w[NEO_VRARSX_MAX_PERIOD];
    for (int k = 0; k < period; ++k) { price_w[k] = NEO_F64_NAN; dev_w[k] = NEO_F64_NAN; }

    int price_head = 0, price_count = 0, price_valid = 0;
    int dev_head   = 0, dev_count   = 0, dev_valid   = 0;
    double price_sum = 0.0, price_sum_sq = 0.0, dev_sum = 0.0;

    double prev_src_out = NEO_F64_NAN;
    double s28 = NEO_F64_NAN, s30 = NEO_F64_NAN;
    double s38 = NEO_F64_NAN, s40 = NEO_F64_NAN;
    double s48 = NEO_F64_NAN, s50 = NEO_F64_NAN;
    double s58 = NEO_F64_NAN, s60 = NEO_F64_NAN;
    double s68 = NEO_F64_NAN, s70 = NEO_F64_NAN;
    double s78 = NEO_F64_NAN, s80 = NEO_F64_NAN;

    const double speed = NEO_VRARSX_SPEED;
    const double spdp1 = speed + 1.0;
    const double pf    = (double)period;

    for (int i = 0; i < n; ++i) {
        const double value = data[i];
        /* is_valid_source (:253) is `is_finite`. */
        const double src_out = isfinite(value) ? (100.0 * value) : NEO_F64_NAN;

        /* push_window_sum_sumsq (:288) */
        if (price_count == period) {
            const double old = price_w[price_head];
            if (isfinite(old)) { price_valid -= 1; price_sum -= old; price_sum_sq -= old * old; }
        } else {
            price_count += 1;
        }
        price_w[price_head] = value;
        price_head += 1; if (price_head == period) price_head = 0;
        if (isfinite(value)) { price_valid += 1; price_sum += value; price_sum_sq += value * value; }

        double dev;
        if (price_count == period && price_valid == period) {
            /* biased_std_from_sums (:272) */
            const double centered = fmax(price_sum_sq - (price_sum * price_sum) / pf, 0.0);
            dev = sqrt(centered / pf);
        } else {
            dev = NEO_F64_NAN;
        }

        /* push_window_sum (:320) */
        if (dev_count == period) {
            const double old = dev_w[dev_head];
            if (isfinite(old)) { dev_valid -= 1; dev_sum -= old; }
        } else {
            dev_count += 1;
        }
        dev_w[dev_head] = dev;
        dev_head += 1; if (dev_head == period) dev_head = 0;
        if (isfinite(dev)) { dev_valid += 1; dev_sum += dev; }

        const double devavg = (dev_count == period && dev_valid == period)
                            ? (dev_sum / pf) : NEO_F64_NAN;

        const double vol_ratio = (isfinite(dev) && isfinite(devavg) && devavg != 0.0)
                               ? (dev / devavg) : NEO_F64_NAN;

        const double adaptive_len = (isfinite(vol_ratio) && vol_ratio > 0.0)
                                  ? trunc(pf / vol_ratio) : NEO_F64_NAN;

        const double kg = isfinite(adaptive_len) ? (3.0 / (adaptive_len + 2.0)) : NEO_F64_NAN;
        const double hg = isfinite(kg) ? (1.0 - kg) : NEO_F64_NAN;

        const double mom0 = (isfinite(src_out) && isfinite(prev_src_out))
                          ? (src_out - prev_src_out) : NEO_F64_NAN;
        const double moa0 = isfinite(mom0) ? fabs(mom0) : NEO_F64_NAN;

        const bool kh = isfinite(kg) && isfinite(hg);

        const double f28  = (kh && isfinite(mom0)) ? (kg * mom0 + hg * neo_vrarsx_nz(s28)) : NEO_F64_NAN;
        const double f30  = (kh && isfinite(f28))  ? (hg * neo_vrarsx_nz(s30) + kg * f28)  : NEO_F64_NAN;
        const double mom1 = (isfinite(f28) && isfinite(f30)) ? (f28 * spdp1 - f30 * speed) : NEO_F64_NAN;

        const double f38  = (kh && isfinite(mom1)) ? (hg * neo_vrarsx_nz(s38) + kg * mom1) : NEO_F64_NAN;
        const double f40  = (kh && isfinite(f38))  ? (kg * f38 + hg * neo_vrarsx_nz(s40))  : NEO_F64_NAN;
        const double mom2 = (isfinite(f38) && isfinite(f40)) ? (f38 * spdp1 - f40 * speed) : NEO_F64_NAN;

        const double f48 = (kh && isfinite(mom2)) ? (hg * neo_vrarsx_nz(s48) + kg * mom2) : NEO_F64_NAN;
        const double f50 = (kh && isfinite(f48))  ? (kg * f48 + hg * neo_vrarsx_nz(s50))  : NEO_F64_NAN;
        const double mom_out = (isfinite(f48) && isfinite(f50)) ? (f48 * spdp1 - f50 * speed) : NEO_F64_NAN;

        const double f58  = (kh && isfinite(moa0)) ? (hg * neo_vrarsx_nz(s58) + kg * moa0) : NEO_F64_NAN;
        const double f60  = (kh && isfinite(f58))  ? (kg * f58 + hg * neo_vrarsx_nz(s60))  : NEO_F64_NAN;
        const double moa1 = (isfinite(f58) && isfinite(f60)) ? (f58 * spdp1 - f60 * speed) : NEO_F64_NAN;

        const double f68  = (kh && isfinite(moa1)) ? (hg * neo_vrarsx_nz(s68) + kg * moa1) : NEO_F64_NAN;
        const double f70  = (kh && isfinite(f68))  ? (kg * f68 + hg * neo_vrarsx_nz(s70))  : NEO_F64_NAN;
        const double moa2 = (isfinite(f68) && isfinite(f70)) ? (f68 * spdp1 - f70 * speed) : NEO_F64_NAN;

        const double f78 = (kh && isfinite(moa2)) ? (hg * neo_vrarsx_nz(s78) + kg * moa2) : NEO_F64_NAN;
        const double f80 = (kh && isfinite(f78))  ? (kg * f78 + hg * neo_vrarsx_nz(s80))  : NEO_F64_NAN;
        const double moa_out = (isfinite(f78) && isfinite(f80)) ? (f78 * spdp1 - f80 * speed) : NEO_F64_NAN;

        double line = NEO_F64_NAN;
        if (isfinite(mom_out) && isfinite(moa_out) && moa_out != 0.0) {
            const double raw = (mom_out / moa_out + 1.0) * 50.0;
            /* Rust f64::clamp -- an if-chain, NOT fmin/fmax. See header. */
            line = (raw < 0.0) ? 0.0 : ((raw > 100.0) ? 100.0 : raw);
        }

        o[i] = line;

        prev_src_out = src_out;
        s28 = f28; s30 = f30; s38 = f38; s40 = f40; s48 = f48; s50 = f50;
        s58 = f58; s60 = f60; s68 = f68; s70 = f70; s78 = f78; s80 = f80;
    }
}
