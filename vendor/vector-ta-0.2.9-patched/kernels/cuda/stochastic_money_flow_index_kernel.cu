#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

extern "C" __global__ void stochastic_money_flow_index_batch_f64(
    const double* __restrict__ source,
    const double* __restrict__ volume,
    int len,
    const int* __restrict__ stoch_k_lengths,
    const int* __restrict__ stoch_k_smooths,
    const int* __restrict__ stoch_d_smooths,
    const int* __restrict__ mfi_lengths,
    int n_combos,
    int max_flow_len,
    int max_stoch_k_length,
    int max_k_smooth,
    int max_d_smooth,
    double* __restrict__ pos_buf,
    double* __restrict__ neg_buf,
    int* __restrict__ maxdq_idx,
    double* __restrict__ maxdq_val,
    int* __restrict__ mindq_idx,
    double* __restrict__ mindq_val,
    double* __restrict__ k_buf,
    double* __restrict__ d_buf,
    double* __restrict__ out_k,
    double* __restrict__ out_d
) {
    int combo_idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo_idx >= n_combos || len <= 0) {
        return;
    }

    int stoch_k_length = stoch_k_lengths[combo_idx];
    int stoch_k_smooth = stoch_k_smooths[combo_idx];
    int stoch_d_smooth = stoch_d_smooths[combo_idx];
    int mfi_length = mfi_lengths[combo_idx];
    int flow_len = mfi_length - 1;

    double* row_k = out_k + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* row_d = out_d + static_cast<size_t>(combo_idx) * static_cast<size_t>(len);
    double* pos = pos_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_flow_len);
    double* neg = neg_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_flow_len);
    int* max_idx = maxdq_idx + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_stoch_k_length);
    double* max_val = maxdq_val + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_stoch_k_length);
    int* min_idx = mindq_idx + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_stoch_k_length);
    double* min_val = mindq_val + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_stoch_k_length);
    double* k_ring = k_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_k_smooth);
    double* d_ring = d_buf + static_cast<size_t>(combo_idx) * static_cast<size_t>(max_d_smooth);

    for (int i = 0; i < len; ++i) {
        row_k[i] = CUDART_NAN;
        row_d[i] = CUDART_NAN;
    }

    if (stoch_k_length <= 0 || stoch_k_smooth <= 0 || stoch_d_smooth <= 0 || mfi_length <= 0) {
        return;
    }
    if (flow_len > max_flow_len || stoch_k_length > max_stoch_k_length ||
        stoch_k_smooth > max_k_smooth || stoch_d_smooth > max_d_smooth) {
        return;
    }

    int flow_head = 0;
    int flow_count = 0;
    double pos_sum = 0.0;
    double neg_sum = 0.0;
    double prev_source = CUDART_NAN;
    bool has_prev = false;
    int mfi_index = 0;
    int max_head = 0;
    int max_size = 0;
    int min_head = 0;
    int min_size = 0;
    int k_head = 0;
    int k_len = 0;
    double k_sum = 0.0;
    int d_head = 0;
    int d_len = 0;
    double d_sum = 0.0;

    for (int i = 0; i < len; ++i) {
        double src = source[i];
        double vol = volume[i];
        if (!isfinite(src) || !isfinite(vol)) {
            flow_head = 0;
            flow_count = 0;
            pos_sum = 0.0;
            neg_sum = 0.0;
            prev_source = CUDART_NAN;
            has_prev = false;
            mfi_index = 0;
            max_head = 0;
            max_size = 0;
            min_head = 0;
            min_size = 0;
            k_head = 0;
            k_len = 0;
            k_sum = 0.0;
            d_head = 0;
            d_len = 0;
            d_sum = 0.0;
            continue;
        }

        bool have_mfi = false;
        double mfi = 0.0;
        if (!has_prev) {
            prev_source = src;
            has_prev = true;
            if (mfi_length == 1) {
                have_mfi = true;
                mfi = 0.0;
            }
        } else {
            double diff = src - prev_source;
            prev_source = src;

            if (flow_len == 0) {
                have_mfi = true;
                mfi = 0.0;
            } else {
                double flow = src * vol;
                double pos_new = diff > 0.0 ? flow : 0.0;
                double neg_new = diff < 0.0 ? flow : 0.0;

                if (flow_count == flow_len) {
                    pos_sum -= pos[flow_head];
                    neg_sum -= neg[flow_head];
                } else {
                    flow_count += 1;
                }

                pos[flow_head] = pos_new;
                neg[flow_head] = neg_new;
                pos_sum += pos_new;
                neg_sum += neg_new;
                flow_head += 1;
                if (flow_head == flow_len) {
                    flow_head = 0;
                }

                if (flow_count == flow_len) {
                    double total = pos_sum + neg_sum;
                    mfi = total <= 1e-14 ? 0.0 : 100.0 * pos_sum / total;
                    have_mfi = true;
                }
            }
        }

        if (!have_mfi) {
            continue;
        }

        int window_start = mfi_index + 1 - stoch_k_length;
        if (window_start < 0) {
            window_start = 0;
        }
        while (max_size > 0 && max_idx[max_head] < window_start) {
            max_head += 1;
            if (max_head == stoch_k_length) {
                max_head = 0;
            }
            max_size -= 1;
        }
        while (min_size > 0 && min_idx[min_head] < window_start) {
            min_head += 1;
            if (min_head == stoch_k_length) {
                min_head = 0;
            }
            min_size -= 1;
        }

        while (max_size > 0) {
            int back_pos = max_head + max_size - 1;
            if (back_pos >= stoch_k_length) {
                back_pos -= stoch_k_length;
            }
            if (max_val[back_pos] <= mfi) {
                max_size -= 1;
            } else {
                break;
            }
        }
        int max_insert = max_head + max_size;
        if (max_insert >= stoch_k_length) {
            max_insert -= stoch_k_length;
        }
        max_idx[max_insert] = mfi_index;
        max_val[max_insert] = mfi;
        max_size += 1;

        while (min_size > 0) {
            int back_pos = min_head + min_size - 1;
            if (back_pos >= stoch_k_length) {
                back_pos -= stoch_k_length;
            }
            if (min_val[back_pos] >= mfi) {
                min_size -= 1;
            } else {
                break;
            }
        }
        int min_insert = min_head + min_size;
        if (min_insert >= stoch_k_length) {
            min_insert -= stoch_k_length;
        }
        min_idx[min_insert] = mfi_index;
        min_val[min_insert] = mfi;
        min_size += 1;
        mfi_index += 1;

        if (mfi_index < stoch_k_length) {
            continue;
        }

        double highest = max_size > 0 ? max_val[max_head] : mfi;
        double lowest = min_size > 0 ? min_val[min_head] : mfi;
        double raw_k = (highest - lowest) > DBL_EPSILON
            ? 100.0 * (mfi - lowest) / (highest - lowest)
            : 0.0;

        if (stoch_k_smooth == 1) {
            row_k[i] = raw_k;
        } else {
            if (k_len == stoch_k_smooth) {
                k_sum -= k_ring[k_head];
            } else {
                k_len += 1;
            }
            k_ring[k_head] = raw_k;
            k_sum += raw_k;
            k_head += 1;
            if (k_head == stoch_k_smooth) {
                k_head = 0;
            }
            if (k_len < stoch_k_smooth) {
                continue;
            }
            row_k[i] = k_sum / static_cast<double>(stoch_k_smooth);
        }

        double k_value = row_k[i];
        if (stoch_d_smooth == 1) {
            row_d[i] = k_value;
            continue;
        }
        if (d_len == stoch_d_smooth) {
            d_sum -= d_ring[d_head];
        } else {
            d_len += 1;
        }
        d_ring[d_head] = k_value;
        d_sum += k_value;
        d_head += 1;
        if (d_head == stoch_d_smooth) {
            d_head = 0;
        }
        row_d[i] = d_len < stoch_d_smooth
            ? CUDART_NAN
            : d_sum / static_cast<double>(stoch_d_smooth);
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — stochastic_money_flow_index
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/stochastic_money_flow_index.rs:671
 *   compute_row_default_14_3_3_14::<true>. That specialisation — NOT the
 *   generic stream — is the oracle because compute_row (:876-885) dispatches
 *   to it whenever the four windows are 14/3/3/14, and those are exactly the
 *   defaults this period-invariant lane pins. CHECK_FINITE is true: it is the
 *   branch compute_row takes when the frame is not all-finite, and on an
 *   all-finite frame the two branches compute the same expressions.
 *
 * Column: output_id "value" resolves to out.k — cpu_batch.rs:5692 accepts
 *   "k"/"value". The D series is a separate output id; its SMA is still
 *   advanced here because the CPU advances it from K inside the same loop and
 *   a reset must clear both consistently.
 *
 * PERIOD-INVARIANT: compute_stochastic_money_flow_index_batch reads
 *   stoch_k_length (14), stoch_k_smooth (3), stoch_d_smooth (3) and
 *   mfi_length (14) and NEVER period (cpu_batch.rs:5668-5675).
 *
 * FIRST-VALID IGNORED: the row walks EVERY bar from index 0 and RESETS every
 *   accumulator on a non-finite source OR volume (:709-731), writing NaN for
 *   that bar. The caller's first-valid index is never read.
 *
 * Input: (close, volume) — extract_close_volume_input(.., "close")
 *   (cpu_batch.rs:5659) — F64InputKind::CloseVolume.
 *
 * Shape: ONE THREAD PER COLUMN, bars ascending. A 13-deep signed money-flow
 *   ring with running positive/negative sums, two MONOTONE DEQUES over the MFI
 *   series, and two sliding-sum SMAs all carry state.
 *
 * ARITHMETIC taken verbatim:
 *   * the flow sums SUBTRACT THE OUTGOING ENTRY FIRST and only then add the
 *     new one (:748-761) — pos_sum -= old; ...; pos_sum += new. Reversing
 *     those two roundings drifts.
 *   * the K SMA adds then subtracts (k_sum += raw_k; k_sum -= old, :839-840)
 *     while the ring is full, and accumulates plainly while warming.
 *   * the MFI guard is total <= 1e-14 (:767) — already an f64-sized
 *     tolerance, carried across unchanged.
 *   * the stochastic guard is highest - lowest > f64::EPSILON (:815), i.e.
 *     DBL_EPSILON = 2.220446049250313e-16. This is the f64 machine epsilon and
 *     is CORRECT here; it is not an f32 epsilon copied across.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Defaults from cpu_batch.rs:5668-5675, which are what selects the
 * 14/3/3/14 specialisation. Every ring below is sized by them, so the bounds
 * belong to the COMPILED kernel. */
#define NEO_SMFI_FLOW_LEN   13
#define NEO_SMFI_DEQUE_CAP  15
#define NEO_SMFI_STOCH_LEN  14
#define NEO_SMFI_SMOOTH     3
#define NEO_SMFI_DBL_EPS    2.2204460492503131e-16

extern "C" __global__
void stochastic_money_flow_index_neo_batch_f64(const double* __restrict__ price,
                                               const double* __restrict__ volume,
                                               int n,
                                               const int* __restrict__ periods,
                                               int n_combos,
                                               int first_valid,
                                               double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;     /* period-invariant — see header */
    (void)first_valid; /* the mid-series reset reproduces it — see header */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    double pos_buf[NEO_SMFI_FLOW_LEN], neg_buf[NEO_SMFI_FLOW_LEN];
    int    flow_head = 0, flow_count = 0;
    double pos_sum = 0.0, neg_sum = 0.0;
    double prev_source = NEO_F64_NAN;
    bool   has_prev = false;

    int    max_idx[NEO_SMFI_DEQUE_CAP]; double max_val[NEO_SMFI_DEQUE_CAP];
    int    min_idx[NEO_SMFI_DEQUE_CAP]; double min_val[NEO_SMFI_DEQUE_CAP];
    int    max_head = 0, max_len = 0, min_head = 0, min_len = 0;
    int    mfi_index = 0;

    double k_buf[NEO_SMFI_SMOOTH]; int k_head = 0, k_len = 0; double k_sum = 0.0;
    double d_buf[NEO_SMFI_SMOOTH]; int d_head = 0, d_len = 0; double d_sum = 0.0;

    for (int i = 0; i < n; ++i) {
        const double src = price[i];
        const double vol = volume[i];

        if (!isfinite(src) || !isfinite(vol)) {
            flow_head = 0; flow_count = 0; pos_sum = 0.0; neg_sum = 0.0;
            prev_source = NEO_F64_NAN; has_prev = false;
            max_head = 0; max_len = 0; min_head = 0; min_len = 0; mfi_index = 0;
            k_head = 0; k_len = 0; k_sum = 0.0;
            d_head = 0; d_len = 0; d_sum = 0.0;
            o[i] = NEO_F64_NAN;
            continue;
        }

        if (!has_prev) {
            prev_source = src; has_prev = true;
            o[i] = NEO_F64_NAN;
            continue;
        }

        const double diff = src - prev_source;
        prev_source = src;
        const double flow    = src * vol;
        const double pos_new = (diff > 0.0) ? flow : 0.0;
        const double neg_new = (diff < 0.0) ? flow : 0.0;

        if (flow_count == NEO_SMFI_FLOW_LEN) {
            pos_sum -= pos_buf[flow_head];
            neg_sum -= neg_buf[flow_head];
        } else {
            ++flow_count;
        }

        pos_buf[flow_head] = pos_new;
        neg_buf[flow_head] = neg_new;
        pos_sum += pos_new;
        neg_sum += neg_new;
        ++flow_head;
        if (flow_head == NEO_SMFI_FLOW_LEN) flow_head = 0;

        if (flow_count < NEO_SMFI_FLOW_LEN) { o[i] = NEO_F64_NAN; continue; }

        const double total = pos_sum + neg_sum;
        const double mfi   = (total <= 1e-14) ? 0.0 : (100.0 * pos_sum / total);

        while (max_len > 0) {
            const int back = (max_head + max_len - 1) % NEO_SMFI_DEQUE_CAP;
            if (max_val[back] > mfi) break;
            --max_len;
        }
        {
            const int tail = (max_head + max_len) % NEO_SMFI_DEQUE_CAP;
            max_idx[tail] = mfi_index; max_val[tail] = mfi; ++max_len;
        }

        while (min_len > 0) {
            const int back = (min_head + min_len - 1) % NEO_SMFI_DEQUE_CAP;
            if (min_val[back] < mfi) break;
            --min_len;
        }
        {
            const int tail = (min_head + min_len) % NEO_SMFI_DEQUE_CAP;
            min_idx[tail] = mfi_index; min_val[tail] = mfi; ++min_len;
        }

        const int window_start =
            (mfi_index + 1 >= NEO_SMFI_STOCH_LEN) ? (mfi_index + 1 - NEO_SMFI_STOCH_LEN) : 0;
        while (max_len > 0 && max_idx[max_head] < window_start) {
            max_head = (max_head + 1) % NEO_SMFI_DEQUE_CAP; --max_len;
        }
        while (min_len > 0 && min_idx[min_head] < window_start) {
            min_head = (min_head + 1) % NEO_SMFI_DEQUE_CAP; --min_len;
        }
        ++mfi_index;

        if (mfi_index < NEO_SMFI_STOCH_LEN) { o[i] = NEO_F64_NAN; continue; }

        const double highest = max_val[max_head];
        const double lowest  = min_val[min_head];
        const double raw_k   = (highest - lowest > NEO_SMFI_DBL_EPS)
            ? (100.0 * (mfi - lowest) / (highest - lowest))
            : 0.0;

        double k;
        if (k_len < NEO_SMFI_SMOOTH) {
            k_buf[k_head] = raw_k;
            ++k_head; if (k_head == NEO_SMFI_SMOOTH) k_head = 0;
            ++k_len;
            k_sum += raw_k;
            if (k_len < NEO_SMFI_SMOOTH) { o[i] = NEO_F64_NAN; continue; }
            k = k_sum / 3.0;
        } else {
            k_sum += raw_k;
            k_sum -= k_buf[k_head];
            k_buf[k_head] = raw_k;
            ++k_head; if (k_head == NEO_SMFI_SMOOTH) k_head = 0;
            k = k_sum / 3.0;
        }

        o[i] = k;

        /* The D SMA is advanced from K exactly as the CPU advances it, so a
         * later reset clears a consistent state. Its value is a different
         * output id and is not written here. */
        if (d_len < NEO_SMFI_SMOOTH) {
            d_buf[d_head] = k;
            ++d_head; if (d_head == NEO_SMFI_SMOOTH) d_head = 0;
            ++d_len;
            d_sum += k;
        } else {
            d_sum += k;
            d_sum -= d_buf[d_head];
            d_buf[d_head] = k;
            ++d_head; if (d_head == NEO_SMFI_SMOOTH) d_head = 0;
        }
    }
}
