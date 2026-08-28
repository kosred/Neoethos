#include <cmath>
#include <cstddef>

extern "C" __global__ void polynomial_regression_extrapolation_batch_f64(
    const double* __restrict__ data,
    int len,
    const int* __restrict__ lengths,
    int rows,
    int max_length,
    const double* __restrict__ weights,
    double* __restrict__ out
) {
    int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || len <= 0) {
        return;
    }

    int length = lengths[row];
    const double* row_weights =
        weights + static_cast<size_t>(row) * static_cast<size_t>(max_length);
    double* row_out = out + static_cast<size_t>(row) * static_cast<size_t>(len);

    for (int i = 0; i < len; ++i) {
        row_out[i] = NAN;
    }

    if (length <= 0 || length > len || length > max_length) {
        return;
    }

    int valid_run = 0;
    for (int i = 0; i < len; ++i) {
        double value = data[i];
        if (!isfinite(value)) {
            valid_run = 0;
            continue;
        }

        valid_run += 1;
        if (valid_run < length) {
            continue;
        }

        double acc = 0.0;
        for (int offset = 0; offset < length; ++offset) {
            acc += row_weights[offset] * data[i - offset];
        }
        row_out[i] = acc;
    }
}

/* ===========================================================================
 * f64 LANE  --  closer 2, round 2      polynomial_regression_extrapolation
 * ---------------------------------------------------------------------------
 * CPU reference: `polynomial_regression_extrapolation_scalar`,
 * src/indicators/polynomial_regression_extrapolation.rs:542, with the weight
 * construction `build_forecast_weights_uncached` (:410) and the dense solve
 * `solve_dense_system_in_place` (:341), reached through
 * `polynomial_regression_extrapolation_prepare` (:463).
 *
 * `length` IS the swept parameter (cpu_batch.rs:4764, default 100).
 * `extrapolate` is 10 and `degree` is 3 (:4766, :4773), so the normal system is
 * 4x4 and the forecast abscissa is x_eval = -10.
 *
 * WHY THE WEIGHTS ARE BUILT IN THE KERNEL. The entry point already in this
 * file, `polynomial_regression_extrapolation_batch_f64`, takes a `weights`
 * matrix the HOST solved for. The lane launches
 * (series, n, periods, n_combos, first_valid, out): there is no weight buffer
 * in that signature, and each row sweeps a different `length` and therefore a
 * different weight vector. So the thread solves its own 4x4 system.
 *
 * NO PER-THREAD WEIGHT ARRAY, so no `max_period` bound and NEVER-OOM by
 * construction. The CPU materialises `length` weights (:451-458) purely as a
 * lookup table; every one of them is `Horner(rhs, x)` over the FOUR solved
 * coefficients, so the kernel keeps the four and recomputes the weight inside
 * the dot loop with the same `acc.mul_add(xf, rhs[power])` sequence, descending
 * in `power`, that the CPU uses (:455-457). Same operations in the same order
 * gives the same double.
 *
 * `powi` IS REPRODUCED, NOT APPROXIMATED. `(x as f64).powi(p)` (:439, :445)
 * with a RUNTIME exponent lowers to `llvm.powi`, i.e. compiler-rt `__powidf2`,
 * which is binary exponentiation: `if (b & 1) r *= a; b /= 2; a *= a`.
 * `neo_pre_powi` below is that loop verbatim. For the default degree the
 * exponents run 0..6 and the bases are small integers, so every intermediate is
 * an exact integer double and the association could not matter -- but it WOULD
 * matter for a long sweep, and a kernel that is only accidentally right is not
 * right.
 *
 * ACCUMULATION ORDER: `acc += weights[offset] * data[idx - offset]` for
 * `offset` ASCENDING (:569-571), a single running accumulator. The crate's
 * length-100 fast path (:513-539) is the same loop with the same ascending
 * offset and the same single accumulator, so there is one oracle rather than
 * two and no special case here.
 *
 * NaN resets `valid_run` and emits NaN (:556-560); a bar with fewer than
 * `length` consecutive finite predecessors emits NaN (:563-566). Both
 * reproduced. Note the CPU tests `is_nan`, not `is_finite`: an INFINITY is
 * accepted into the window and propagates through the dot, which is what this
 * kernel does too.
 *
 * `first_valid` is the lane AllInputsNonNan over the close slice, which is the
 * CPU `position(|value| !value.is_nan())` (:470-473).
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

/* cpu_batch.rs:4766 / :4773 -- `extrapolate` and `degree` defaults. */
#define NEO_PRE_EXTRAPOLATE 10
#define NEO_PRE_DEGREE      3
#define NEO_PRE_ORDER       (NEO_PRE_DEGREE + 1)

/* polynomial_regression_extrapolation.rs:67 -- SINGULAR_EPSILON. An f64 guard
 * in an f64 routine; not an f32 epsilon and not to be resized. */
#define NEO_PRE_SINGULAR_EPS 1e-12

/* compiler-rt __powidf2, which is what Rust `f64::powi` with a runtime exponent
 * lowers to. Reproduced so the Gram sums round exactly as the CPU rounds them. */
__device__ __forceinline__
static double neo_pre_powi(double a, int b)
{
    const bool recip = (b < 0);
    double r = 1.0;
    for (;;) {
        if (b & 1) r *= a;
        b /= 2;
        if (b == 0) break;
        a *= a;
    }
    return recip ? (1.0 / r) : r;
}

/* solve_dense_system_in_place, polynomial_regression_extrapolation.rs:341 --
 * Gaussian elimination with partial pivoting and BACK SUBSTITUTION. Note this
 * is NOT the Gauss-Jordan the sgf kernel needs: the elimination touches rows
 * BELOW the pivot only (:355-366), and the solution is then walked back up
 * (:369-378). The two produce different roundings, so each kernel transcribes
 * its own indicator's solver. */
__device__ __forceinline__
static bool neo_pre_solve(double m[NEO_PRE_ORDER * NEO_PRE_ORDER],
                          double rhs[NEO_PRE_ORDER])
{
    const int n = NEO_PRE_ORDER;
    for (int pivot_col = 0; pivot_col < n; ++pivot_col) {
        int    pivot_row = pivot_col;
        double pivot_abs = fabs(m[pivot_col * n + pivot_col]);
        for (int row = pivot_col + 1; row < n; ++row) {
            const double candidate = fabs(m[row * n + pivot_col]);
            if (candidate > pivot_abs) { pivot_abs = candidate; pivot_row = row; }
        }
        if (pivot_abs <= NEO_PRE_SINGULAR_EPS) return false;

        if (pivot_row != pivot_col) {
            for (int col = pivot_col; col < n; ++col) {
                const double t = m[pivot_col * n + col];
                m[pivot_col * n + col] = m[pivot_row * n + col];
                m[pivot_row * n + col] = t;
            }
            const double tr = rhs[pivot_col];
            rhs[pivot_col] = rhs[pivot_row];
            rhs[pivot_row] = tr;
        }

        const double pivot = m[pivot_col * n + pivot_col];
        for (int row = pivot_col + 1; row < n; ++row) {
            const double factor = m[row * n + pivot_col] / pivot;
            if (factor == 0.0) continue;
            m[row * n + pivot_col] = 0.0;
            for (int col = pivot_col + 1; col < n; ++col) {
                m[row * n + col] -= factor * m[pivot_col * n + col];
            }
            rhs[row] -= factor * rhs[pivot_col];
        }
    }

    for (int row = n - 1; row >= 0; --row) {
        double acc = rhs[row];
        for (int col = row + 1; col < n; ++col) {
            acc -= m[row * n + col] * rhs[col];
        }
        const double pivot = m[row * n + row];
        if (fabs(pivot) <= NEO_PRE_SINGULAR_EPS) return false;
        rhs[row] = acc / pivot;
    }
    return true;
}

extern "C" __global__
void polynomial_regression_extrapolation_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    const int length = periods[combo];
    if (first_valid < 0 || first_valid >= n) return;
    /* :476-481 -- length 0 or longer than the data errors. */
    if (length <= 0 || length > n) return;
    /* :419-428 -- degree + 1 > length errors. */
    if (NEO_PRE_ORDER > length) return;
    /* :483-489 -- NotEnoughValidData leaves the row NaN. */
    if (n - first_valid < length) return;

    /* build_forecast_weights_uncached, :430-440 -- the normal matrix. */
    double normal[NEO_PRE_ORDER * NEO_PRE_ORDER];
    for (int row = 0; row < NEO_PRE_ORDER; ++row) {
        for (int col = 0; col < NEO_PRE_ORDER; ++col) {
            const int power = row + col;
            double sum = 0.0;
            for (int x = 0; x < length; ++x) {
                sum += neo_pre_powi((double)x, power);
            }
            normal[row * NEO_PRE_ORDER + col] = sum;
        }
    }

    /* :442-447 -- the right-hand side is the forecast abscissa's powers. */
    const double x_eval = -(double)NEO_PRE_EXTRAPOLATE;
    double rhs[NEO_PRE_ORDER];
    for (int power = 0; power < NEO_PRE_ORDER; ++power) {
        rhs[power] = neo_pre_powi(x_eval, power);
    }

    if (!neo_pre_solve(normal, rhs)) return;   /* SingularFit -> NaN row */

    /* :551-573 -- ascending offset, one accumulator, NaN resets the run. */
    int valid_run = 0;
    for (int idx = first_valid; idx < n; ++idx) {
        const double value = data[idx];
        if (isnan(value)) {
            valid_run = 0;
            o[idx] = NEO_F64_NAN;
            continue;
        }

        valid_run += 1;
        if (valid_run < length) {
            o[idx] = NEO_F64_NAN;
            continue;
        }

        double acc = 0.0;
        for (int offset = 0; offset < length; ++offset) {
            /* weights[offset] = Horner over rhs, :453-458. */
            const double xf = (double)offset;
            double w = 0.0;
            for (int power = NEO_PRE_ORDER - 1; power >= 0; --power) {
                w = fma(w, xf, rhs[power]);
            }
            acc += w * data[idx - offset];
        }
        o[idx] = acc;
    }
}
