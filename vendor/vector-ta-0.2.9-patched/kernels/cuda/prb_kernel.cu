#include <cuda_runtime.h>
#include <math.h>

// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4, round 3
//
// CPU reference: prb_scalar (src/indicators/prb.rs:938-1108), reached through
// prb_with_kernel (:1378) -> prb_compute_into, with ssf_filter (:562) as the
// smoothing stage and lu_decomposition (:612) as the solver.
//
// OUTPUT: the `values` column -- compute_prb_batch (cpu_batch.rs:15857)
// resolves output_id == "value" to out.values.
//
// PERIOD-INVARIANT: that batch reads smooth_data (true), smooth_period (10),
// regression_period (100), polynomial_order (2), regression_offset (0), ndev
// (2.0) and equ_from (0) and NEVER `period` (cpu_batch.rs:15833-15839). A
// sweep of five periods gets five identical CPU columns, so this kernel
// writes five identical rows. Every one of those seven is a #define below,
// which is also why the design matrix is a fixed 3x3 and no allocation
// depends on a caller value -- NEVER-OOM by construction.
//
// SHAPE: one thread per combo walking bars ASCENDING. The super-smoother is a
// 2-pole IIR; the regression moments are ROLLED with the binomial shift at
// :1096-1103 rather than rebuilt, so their accumulation order is load-bearing.
// The ssf output is produced inside the same ascending walk and kept in a
// ring of REGRESSION_PERIOD + 1 entries -- the window plus the one bar ahead
// the roll consumes -- so it is never materialised for the whole series.
//
// EPSILON: the 1e-10 singular-matrix guard (:620, :645) is the CPU's own and
// is already f64-sized -- the normal-equation diagonal here is O(n^4) -- so it
// is carried across unchanged rather than rescaled from an f32 constant.
// ===========================================================================

#define PRB_NEO_SMOOTH_PERIOD 10
#define PRB_NEO_REG_PERIOD 100
#define PRB_NEO_ORDER 2
#define PRB_NEO_M 3           /* polynomial_order + 1 */
#define PRB_NEO_MAX_POW 4     /* 2 * polynomial_order */
#define PRB_NEO_OFFSET 0
#define PRB_NEO_EQU_FROM 0
#define PRB_NEO_RING (PRB_NEO_REG_PERIOD + 1)

static __forceinline__ __device__ double prb_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__
void prb_neo_batch_f64(const double* __restrict__ data,
                       int n,
                       const int* __restrict__ periods,
                       int n_combos,
                       int first_valid,
                       double* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;
    if (n <= 0) return;
    (void)periods;   // PERIOD-INVARIANT -- see the header.

    double* __restrict__ row = out + (size_t)combo * (size_t)n;
    const double nn = prb_neo_qnan();

    const int reg_n = PRB_NEO_REG_PERIOD;
    const int k = PRB_NEO_ORDER;
    const int m = PRB_NEO_M;
    const double ndev = 2.0;

    // prb_with_kernel, :1385-1388 -- `!is_nan` over the single close series,
    // which is F64FirstValidRule::AllInputsNonNan for CloseSlice.
    int first = first_valid;
    if (first < 0) first = 0;

    bool refused = false;
    if (first >= n) refused = true;
    if (reg_n <= 0 || reg_n > n) refused = true;            // :1410
    long long warm_ll = (long long)first + (long long)reg_n - 1 + PRB_NEO_EQU_FROM;
    if (!refused && warm_ll >= (long long)n) refused = true; // :1418

    if (refused) {
        for (int i = 0; i < n; ++i) row[i] = nn;
        return;
    }

    const int warmup = (int)warm_ll;
    for (int i = 0; i < n; ++i) row[i] = nn;

    // ------------------------------------------------------- the fixed design
    // :1013-1020 -- sx[p] = sum over j=1..n of j^p, built by repeated multiply
    // in exactly that order.
    double sx[PRB_NEO_MAX_POW + 1];
    for (int p = 0; p <= PRB_NEO_MAX_POW; ++p) sx[p] = 0.0;
    for (int j = 1; j <= reg_n; ++j) {
        const double jf = (double)j;
        double pwr = 1.0;
        sx[0] += 1.0;
        for (int p = 1; p <= PRB_NEO_MAX_POW; ++p) {
            pwr *= jf;
            sx[p] += pwr;
        }
    }

    double A[PRB_NEO_M * PRB_NEO_M];
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            A[i * m + j] = sx[i + j];

    // lu_decomposition, :612-654, including both singular refusals.
    double L[PRB_NEO_M * PRB_NEO_M];
    double U[PRB_NEO_M * PRB_NEO_M];
    for (int i = 0; i < m * m; ++i) { L[i] = 0.0; U[i] = 0.0; }
    for (int j = 0; j < m; ++j) U[j] = A[j];
    if (fabs(U[0]) < 1e-10) {
        return;   // SingularMatrix -- the CPU returns Err and no column at all
    }
    for (int i = 1; i < m; ++i) L[i * m] = A[i * m] / U[0];
    for (int i = 0; i < m; ++i) L[i * m + i] = 1.0;
    for (int i = 1; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            double sum = 0.0;
            for (int kk = 0; kk < i; ++kk) sum += L[i * m + kk] * U[kk * m + j];
            U[i * m + j] = A[i * m + j] - sum;

            if (j > i) {
                double sum2 = 0.0;
                for (int kk = 0; kk < i; ++kk) sum2 += L[j * m + kk] * U[kk * m + i];
                if (fabs(U[i * m + i]) < 1e-10) return;
                L[j * m + i] = (A[j * m + i] - sum2) / U[i * m + i];
            }
        }
    }

    // :1029-1043 -- Pascal's triangle and the powers of n used by the shift.
    double binom[PRB_NEO_M * PRB_NEO_M];
    for (int i = 0; i < m * m; ++i) binom[i] = 0.0;
    for (int r = 0; r <= k; ++r) {
        const int r_off = r * m;
        binom[r_off + 0] = 1.0;
        binom[r_off + r] = 1.0;
        for (int c = 1; c < r; ++c) {
            const int prev = (r - 1) * m;
            binom[r_off + c] = binom[prev + (c - 1)] + binom[prev + c];
        }
    }
    double n_pow[PRB_NEO_M];
    n_pow[0] = 1.0;
    const double n_f = (double)reg_n;
    for (int r = 1; r <= k; ++r) n_pow[r] = n_pow[r - 1] * n_f;

    const double x_pos = n_f - (double)PRB_NEO_OFFSET + (double)PRB_NEO_EQU_FROM;
    const double inv_n = 1.0 / n_f;

    // ------------------------------------------------- the super-smoother
    // ssf_filter, :570-576. `a`, `b`, `c1..c3` are the CPU's own spellings.
    const double sp = (double)PRB_NEO_SMOOTH_PERIOD;
    const double omega = 2.0 * M_PI / sp;
    const double sqrt2 = 1.4142135623730951;   // std::f64::consts::SQRT_2
    const double ssf_a = exp(-sqrt2 * M_PI / sp);
    const double ssf_b = 2.0 * ssf_a * cos((sqrt2 / 2.0) * omega);
    const double c3 = -ssf_a * ssf_a;
    const double c2 = ssf_b;
    const double c1 = 1.0 - c2 - c3;

    double ring[PRB_NEO_RING];
    double y1, y2;
    bool ssf_phase2 = false;

    {
        const double x0 = data[first];
        const double y0 = c1 * x0 + c2 * x0 + c3 * x0;   // :579
        ring[first % PRB_NEO_RING] = y0;
        y1 = y0;
        y2 = y0;
    }

    // One ssf step at absolute index `idx`, reproducing the two-phase loop at
    // :583-609: the first phase runs until the first non-finite bar and takes
    // it as well, after which every later bar takes the NaN-guarded form.
    #define PRB_NEO_SSF_STEP(idx)                                            \
        do {                                                                 \
            const double xi = data[(idx)];                                   \
            double yv;                                                       \
            if (!ssf_phase2) {                                               \
                yv = c1 * xi + c2 * y1 + c3 * y2;                            \
                if (!isfinite(xi)) ssf_phase2 = true;                        \
            } else {                                                         \
                const double prev1 = isnan(y1) ? xi : y1;                    \
                const double prev2 = isnan(y2) ? prev1 : y2;                 \
                yv = c1 * xi + c2 * prev1 + c3 * prev2;                      \
            }                                                                \
            ring[(idx) % PRB_NEO_RING] = yv;                                 \
            y2 = y1;                                                         \
            y1 = yv;                                                         \
        } while (0)

    for (int idx = first + 1; idx <= warmup; ++idx) PRB_NEO_SSF_STEP(idx);

    // -------------------------------------------------------- the seed window
    // :1076-1093 -- start = warmup + 1 - n - equ_from, which is `first`.
    int start = warmup + 1 - reg_n - PRB_NEO_EQU_FROM;
    double s_xy[PRB_NEO_M];
    for (int r = 0; r < m; ++r) s_xy[r] = 0.0;
    double sum = 0.0, sumsq = 0.0;
    for (int t = 0; t < reg_n; ++t) {
        const double y = ring[(start + t) % PRB_NEO_RING];
        sum += y;
        sumsq += y * y;

        const double jf = (double)t + 1.0;
        s_xy[0] += y;
        double w = jf;
        for (int p = 1; p <= k; ++p) {
            s_xy[p] = fma(y, w, s_xy[p]);
            w *= jf;
        }
    }

    double tmp_y[PRB_NEO_M];
    double coeffs[PRB_NEO_M];
    double s_prev[PRB_NEO_M];
    for (int r = 0; r < m; ++r) { tmp_y[r] = 0.0; coeffs[r] = 0.0; s_prev[r] = 0.0; }

    for (int i = warmup; i < n; ++i) {
        // :1043-1058 -- forward then back substitution, subtracting term by
        // term in index order.
        for (int r = 0; r < m; ++r) {
            double acc = s_xy[r];
            const int rowo = r * m;
            for (int c = 0; c < r; ++c) acc -= L[rowo + c] * tmp_y[c];
            tmp_y[r] = acc / L[rowo + r];
        }
        for (int r = m - 1; r >= 0; --r) {
            const int rowo = r * m;
            double acc = tmp_y[r];
            for (int c = r + 1; c < m; ++c) acc -= U[rowo + c] * coeffs[c];
            coeffs[r] = acc / U[rowo + r];
        }

        // :1060-1063 -- Horner, highest order first, one fma per term.
        double reg = 0.0;
        for (int p = m - 1; p >= 0; --p) reg = fma(reg, x_pos, coeffs[p]);

        const double mean = sum * inv_n;
        const double var = (sumsq * inv_n) - mean * mean;
        const double stdev = (var > 0.0) ? sqrt(var) : 0.0;

        row[i] = reg;
        (void)ndev; (void)stdev;   // the bands are not this lane's columns

        if (i + 1 == n) break;                    // :1078
        const int y_new_idx = start + reg_n;
        if (y_new_idx >= n) break;                // :1082

        PRB_NEO_SSF_STEP(y_new_idx);              // smoothed[i + 1]

        const double y_old = ring[start % PRB_NEO_RING];
        const double y_new = ring[y_new_idx % PRB_NEO_RING];

        for (int r = 0; r < m; ++r) s_prev[r] = s_xy[r];

        s_xy[0] = s_prev[0] - y_old + y_new;
        sum = sum - y_old + y_new;
        sumsq = sumsq - y_old * y_old + y_new * y_new;

        // :1096-1103 -- the binomial shift that re-centres the moments.
        for (int r = 1; r <= k; ++r) {
            const int rowo = r * m;
            double acc = 0.0;
            for (int m2 = 0; m2 <= r; ++m2) {
                const double sign = (((r - m2) & 1) == 1) ? -1.0 : 1.0;
                acc += sign * binom[rowo + m2] * s_prev[m2];
            }
            s_xy[r] = acc + n_pow[r] * y_new;
        }

        start += 1;
    }

    #undef PRB_NEO_SSF_STEP
}
