#include <cmath>
#include <cstddef>

extern "C" __global__ void ehlers_detrending_filter_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int rows,
    double* __restrict__ out_edf,
    double* __restrict__ out_signal
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int length = lengths[row];
    double* row_out_edf = out_edf + static_cast<size_t>(row) * static_cast<size_t>(len);
    double* row_out_signal = out_signal + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out_edf[i] = NAN;
        row_out_signal[i] = NAN;
    }

    if (length <= 0 || length > len) {
        return;
    }

    double weight_sum = 0.0;
    double denom = static_cast<double>(length + 1);
    for (int i = 1; i <= length; ++i) {
        weight_sum += 1.0 - cos((2.0 * 3.14159265358979323846 * static_cast<double>(i)) / denom);
    }
    if (!(weight_sum > 0.0) || !isfinite(weight_sum)) {
        return;
    }

    bool initialized = false;
    double prev_src = 0.0;
    double prev_edf = 0.0;

    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            row_out_edf[i] = NAN;
            initialized = false;
            prev_src = 0.0;
            prev_edf = 0.0;
            continue;
        }

        double prev = initialized ? prev_src : 0.0;
        double edf_raw = (0.95 * value) - (0.95 * prev) + (0.9 * prev_edf);
        row_out_edf[i] = edf_raw;
        prev_src = value;
        prev_edf = edf_raw;
        initialized = true;
    }

    int run = 0;
    for (int i = 0; i < len; ++i) {
        double raw = row_out_edf[i];
        if (!isfinite(raw)) {
            row_out_signal[i] = NAN;
            run = 0;
            continue;
        }

        run += 1;
        double filt = 0.0;
        for (int offset = 0; offset < run && offset < length; ++offset) {
            double hist = row_out_edf[i - offset];
            double weight = 1.0 -
                cos((2.0 * 3.14159265358979323846 * static_cast<double>(offset + 1)) / denom);
            filt += weight * hist;
        }
        row_out_signal[i] = filt / weight_sum;
    }

    run = 0;
    double prev_filt = 0.0;
    double prev_slo = 0.0;

    for (int i = 0; i < len; ++i) {
        double filt = row_out_signal[i];
        if (!isfinite(filt)) {
            row_out_edf[i] = NAN;
            row_out_signal[i] = NAN;
            run = 0;
            prev_filt = 0.0;
            prev_slo = 0.0;
            continue;
        }

        run += 1;
        double slo = filt - prev_filt;
        double signal = 0.0;
        if (slo > 0.0) {
            signal = (slo > prev_slo) ? 2.0 : 1.0;
        } else if (slo < 0.0) {
            signal = (slo < prev_slo) ? -2.0 : -1.0;
        }
        prev_filt = filt;
        prev_slo = slo;

        if (run >= length) {
            row_out_edf[i] = filt;
            row_out_signal[i] = signal;
        } else {
            row_out_edf[i] = NAN;
            row_out_signal[i] = NAN;
        }
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — ehlers_detrending_filter
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/ehlers_detrending_filter.rs:287
 *   (`ehlers_detrending_filter_compute_into`) driving
 *   `EhlersDetrendingFilterStream::update` (:345). The driver walks EVERY bar
 *   from index 0 and the stream RESETS on a non-finite value, so `first_valid`
 *   is not read here — the reset reproduces it exactly.
 *
 * Column: the batch maps output_id "value" / "edf" to `out.edf`
 *   (cpu_batch.rs:12731), and `compute_into` writes `filt` — the cosine
 *   weighted filter — into that field (:400). The raw one-pole `edf` is the
 *   RING content, never the output. Emitting the one-pole value would be
 *   plausible-looking and wrong.
 *
 * PERIOD-INVARIANT: `compute_ehlers_detrending_filter_batch`
 *   (cpu_batch.rs:12705-12706) reads `source` and `length` and NEVER `period`,
 *   so a five-period sweep gets five identical CPU columns and this kernel
 *   emits five identical rows. `length` is pinned at the CPU default 10.
 *
 * Source: hlcc4 = (h + l + 2c) / 4 (data_loader.rs:171), computed in-thread
 *   from the resident Hlc upload. Asking for a fourth upload shape would move
 *   bytes that the card already holds.
 *
 * Arithmetic order, rounding for rounding:
 *   edf = (A*v) - (A*prev_src) + (F*prev_edf)  — three products, two adds,
 *   left-associative, NO fma. The CPU line (:350) has exactly that shape and
 *   a contraction here would change the last bit of a value that feeds a
 *   90%-feedback recursion.
 *   The weighted sum accumulates `offset` ASCENDING (weights[0] against the
 *   most recent ring entry) — the CPU's two-part walk over the ring (:322-336)
 *   in the same order.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* Must match DEFAULT_LENGTH in ehlers_detrending_filter.rs:26. The ring is a
 * per-thread array, so the bound belongs to the compiled kernel. */
#define NEO_EDF_LENGTH 10
#define NEO_EDF_ALPHA    0.95
#define NEO_EDF_FEEDBACK 0.9

extern "C" __global__
void ehlers_detrending_filter_neo_batch_f64(const double* __restrict__ high,
                                            const double* __restrict__ low,
                                            const double* __restrict__ close,
                                            int n,
                                            const int* __restrict__ periods,
                                            int n_combos,
                                            int first_valid,
                                            double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;      /* period-invariant — see the header */
    (void)first_valid;  /* the CPU driver starts at 0 and resets on NaN */

    double* __restrict__ o = out + (size_t)combo * (size_t)n;

    /* cosine_weights (ehlers_detrending_filter.rs:248): denom = length + 1,
     * weight_i = 1 - cos(2*pi*i/denom) for i in 1..=length, summed in i order. */
    const int    L     = NEO_EDF_LENGTH;
    const double denom = (double)(L + 1);
    double weights[NEO_EDF_LENGTH];
    double weight_sum = 0.0;
    for (int i = 1; i <= L; ++i) {
        const double w = 1.0 - cos(2.0 * 3.14159265358979323846 * (double)i / denom);
        weights[i - 1] = w;
        weight_sum += w;
    }

    double ring[NEO_EDF_LENGTH];
    for (int i = 0; i < L; ++i) ring[i] = 0.0;
    int  head = 0, count = 0;
    bool initialized = false;
    double prev_src = 0.0, prev_edf = 0.0, prev_filt = 0.0;

    for (int idx = 0; idx < n; ++idx) {
        const double h = high[idx], l = low[idx], c = close[idx];
        const double value = (h + l + 2.0 * c) * 0.25;

        if (!isfinite(value)) {
            /* stream.reset() (:301) — ring zeroed, counters cleared. */
            for (int i = 0; i < L; ++i) ring[i] = 0.0;
            head = 0; count = 0; initialized = false;
            prev_src = 0.0; prev_edf = 0.0; prev_filt = 0.0;
            o[idx] = NEO_F64_NAN;
            continue;
        }

        const double ps  = initialized ? prev_src : 0.0;
        const double edf = (NEO_EDF_ALPHA * value) - (NEO_EDF_ALPHA * ps)
                         + (NEO_EDF_FEEDBACK * prev_edf);

        ring[head] = edf;
        head = head + 1; if (head == L) head = 0;
        if (count < L) ++count;

        double sum = 0.0;
        int offset = 0;
        int ri = head;
        while (offset < count && ri > 0) { --ri; sum += weights[offset] * ring[ri]; ++offset; }
        ri = L;
        while (offset < count)          { --ri; sum += weights[offset] * ring[ri]; ++offset; }
        const double filt = (weight_sum != 0.0) ? (sum / weight_sum) : 0.0;

        prev_src  = value;
        prev_edf  = edf;
        prev_filt = filt;
        initialized = true;

        o[idx] = (count >= L) ? filt : NEO_F64_NAN;
    }
    (void)prev_filt;
}
