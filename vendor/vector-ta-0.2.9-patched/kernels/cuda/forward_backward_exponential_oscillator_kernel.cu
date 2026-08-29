#include <cuda_runtime.h>
#include <math.h>

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

namespace {

// One exact f64 authority for the preserved primary ABI and the canonical
// three-output production ABI. The forward_backward matrix temporarily owns
// the complete EMA1 history. A descending second pass may then replace EMA1
// with the final output without destroying a value needed by a later row.
__device__ inline void forward_backward_exponential_oscillator_row_f64(
    const double* __restrict__ data,
    int len,
    int length,
    int smooth,
    double* __restrict__ diff_ring,
    int diff_stride,
    double* __restrict__ out_forward_backward,
    double* __restrict__ out_backward,
    double* __restrict__ out_histogram
) {
    if (len <= 0 || out_forward_backward == nullptr) {
        return;
    }

    for (int i = 0; i < len; ++i) {
        out_forward_backward[i] = NEO_F64_NAN;
        if (out_backward != nullptr) {
            out_backward[i] = NEO_F64_NAN;
        }
        if (out_histogram != nullptr) {
            out_histogram[i] = NEO_F64_NAN;
        }
    }

    const bool needs_backward = out_backward != nullptr || out_histogram != nullptr;
    if (length <= 0 || smooth <= 0 ||
        (needs_backward && (out_backward == nullptr || diff_ring == nullptr ||
                            diff_stride < length))) {
        return;
    }

    const double alpha = 2.0 / ((double)smooth + 1.0);

    bool have_ema1_state = false;
    bool have_ema2_state = false;
    bool have_prev_ema2 = false;
    double ema1_state = 0.0;
    double ema2_state = 0.0;
    double prev_ema2 = 0.0;

    int diff_count = 0;
    int diff_head = 0;
    double diff_sum = 0.0;
    double diff_abs_sum = 0.0;

    // CPU phase/order: ema_step(EMA1), ema_step(EMA2), then
    // RollingDiffWindow::update. The rolling sums add the new value before
    // removing the old front; preserving that order is exact-bit load-bearing.
    for (int i = 0; i < len; ++i) {
        const double value = data[i];
        if (!isfinite(value)) {
            have_ema1_state = false;
            have_ema2_state = false;
            have_prev_ema2 = false;
            ema1_state = 0.0;
            ema2_state = 0.0;
            prev_ema2 = 0.0;
            diff_count = 0;
            diff_head = 0;
            diff_sum = 0.0;
            diff_abs_sum = 0.0;
            continue;
        }

        const double ema1 = have_ema1_state
                                ? alpha * value + (1.0 - alpha) * ema1_state
                                : value;
        ema1_state = ema1;
        have_ema1_state = true;
        out_forward_backward[i] = ema1;

        if (!needs_backward) {
            continue;
        }

        const double ema2 = have_ema2_state
                                ? alpha * ema1 + (1.0 - alpha) * ema2_state
                                : ema1;
        ema2_state = ema2;
        have_ema2_state = true;

        if (have_prev_ema2) {
            const double diff = ema2 - prev_ema2;
            if (diff_count < length) {
                diff_ring[diff_count] = diff;
                ++diff_count;
                diff_sum += diff;
                diff_abs_sum += fabs(diff);
            } else {
                const double removed = diff_ring[diff_head];
                diff_sum += diff;
                diff_abs_sum += fabs(diff);
                diff_sum -= removed;
                diff_abs_sum -= fabs(removed);
                diff_ring[diff_head] = diff;
                ++diff_head;
                if (diff_head == length) {
                    diff_head = 0;
                }
            }

            if (diff_count == length && diff_abs_sum != 0.0) {
                out_backward[i] = diff_sum / diff_abs_sum * 50.0 + 50.0;
            }
        }
        prev_ema2 = ema2;
        have_prev_ema2 = true;
    }

    // CPU compute_forward_backward_value seeds EMA2 at the newest EMA1 and
    // walks the window in reverse. Descending bars keep every older EMA1 value
    // intact until its final consumer has read it.
    for (int i = len - 1; i >= 0; --i) {
        if (!isfinite(data[i]) || i + 1 < length) {
            out_forward_backward[i] = NEO_F64_NAN;
            continue;
        }
        bool contiguous = true;
        for (int offset = 0; offset < length; ++offset) {
            if (!isfinite(data[i - offset])) {
                contiguous = false;
                break;
            }
        }
        if (!contiguous) {
            out_forward_backward[i] = NEO_F64_NAN;
            continue;
        }

        double ema2 = out_forward_backward[i];
        double prev = ema2;
        double num = 0.0;
        double den = 0.0;
        for (int j = i - 1; j >= i - length + 1; --j) {
            const double window_value = out_forward_backward[j];
            ema2 += alpha * (window_value - ema2);
            const double dt = prev - ema2;
            num += dt;
            den += fabs(dt);
            prev = ema2;
        }
        if (den != 0.0) {
            const double value = num / den * 50.0 + 50.0;
            out_forward_backward[i] = isfinite(value) ? value : NEO_F64_NAN;
        } else {
            out_forward_backward[i] = NEO_F64_NAN;
        }
    }

    if (out_histogram != nullptr) {
        for (int i = 0; i < len; ++i) {
            const double forward_backward_value = out_forward_backward[i];
            const double backward_value = out_backward[i];
            if (isfinite(forward_backward_value)) {
                out_histogram[i] =
                    (forward_backward_value - backward_value) * 0.25 + 50.0;
            }
        }
    }
}

}  // namespace

// Preserved public/full ABI. Production calls this symbol through the one
// shared resident session; its input pointer already names the resident close
// frame and only the two bounded integer parameter arrays are uploaded.
extern "C" __global__ void forward_backward_exponential_oscillator_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    const int* __restrict__ smooths,
    int n_combos,
    int max_length,
    double* __restrict__ ema1_buffer,
    double* __restrict__ diff_buffer,
    double* __restrict__ out_forward_backward,
    double* __restrict__ out_backward,
    double* __restrict__ out_histogram
) {
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || len <= 0 || max_length <= 0) {
        return;
    }
    (void)ema1_buffer;
    forward_backward_exponential_oscillator_row_f64(
        data,
        len,
        lengths[combo],
        smooths[combo],
        diff_buffer + (size_t)combo * (size_t)max_length,
        max_length,
        out_forward_backward + (size_t)combo * (size_t)len,
        out_backward + (size_t)combo * (size_t)len,
        out_histogram + (size_t)combo * (size_t)len
    );
}

// Preserved generic primary ABI. The periods array is the canonical length
// dimension and smooth remains the exact registry default 10. The shared row
// needs no auxiliary ring when only forward_backward is requested.
extern "C" __global__
void forward_backward_exponential_oscillator_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out
) {
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) {
        return;
    }
    (void)first_valid;
    forward_backward_exponential_oscillator_row_f64(
        data,
        n,
        periods[combo],
        10,
        nullptr,
        0,
        out + (size_t)combo * (size_t)n,
        nullptr,
        nullptr
    );
}
