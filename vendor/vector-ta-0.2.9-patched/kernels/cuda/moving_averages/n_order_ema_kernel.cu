// n_order_ema — CUDA f64 kernel.
//
// Creator authority: https://www.tradingview.com/script/Hgvs8kZi-N-Order-EMA/
// Facade identity: PUB;0d0d8869215f4446b4c17e62c6080830
// Exact Pine source SHA-256:
// 539EEC25A8422DDE96705873212CD55302301BD6CE3284A411C2010536B843D3
//
// WHAT THIS REPLACES
// ------------------
// NOTHING. No `.cu`, no wrapper, no `F64_KERNELS` row: the lane answered
// `CudaF64KernelMissing`.
//
// CPU REFERENCE — src/indicators/moving_averages/n_order_ema.rs
// --------------------------------------------------------------
//   :340 IirCoefficients / :347 IirCoreFilter
//   :373 IirCoreFilter::update       — the recurrence, rounding for rounding
//   :581 binomial / :599 build_coefficients
//   :666 resolve_params / :704 validate_input
//   :739 n_order_ema                 — the entry the brief names
//   :781 n_order_ema_compute_into
//   :794 is_default_ema / :802 n_order_ema_default_ema_into
//
// THE PARAMETERS THAT ARE NOT IN THE LANE ABI, AND WHAT THAT SETTLES
// ------------------------------------------------------------------
// The sweep request carries `periods[]` and nothing else, so `order`,
// `ema_style` and `iir_style` are the CPU defaults: DEFAULT_ORDER = 1 (:29),
// "ema" (:30), "impulse_matched" (:31). That is the same rule the registered
// `tilson` lane follows for `v_factor`.
//
// ORDER 1 COLLAPSES THE COEFFICIENT BUILD TO TWO NUMBERS. From
// build_coefficients (:599), ImpulseMatched, order = 1:
//
//   fc  = 2 / (period + 1)
//   r   = 1 - fc
//   a   = [ binomial(1,1) * (-r)^1 ]  = [ 1.0 * (-r) ]      = [ -r ]
//   sum = binomial(0,0) * r^0         = 1.0  ->  s = 1/1    = 1.0
//   b   = [ fc^1 * binomial(0,0) * r^0 * s ] = [ fc * 1 * 1 * 1 ] = [ fc ]
//
// Every collapsed multiply is by an EXACT 1.0, so `-r` and `fc` are not an
// algebraic simplification that moves a bit — they are the same doubles the
// CPU builds. The brief's "cap N and fail loud above it" therefore does not
// arise: there is no N to cap, because the ABI cannot express one, and a local
// array of 64 histories would be dead weight in every launch.
//
// SO THE RECURRENCE IS (update, :373-386):
//
//   y = first (the first finite value of the run) when y_hist is empty
//   acc = b0 * x            <- one rounding
//   acc = acc - (a0 * y)    <- multiply then subtract, TWO more
//
// THREE roundings, NOT an fma. The CPU writes `acc -= self.coeffs.a[k] * y`
// (:381) and does not fuse; `-fmad=false` keeps nvcc from fusing either.
//
// THE `is_default_ema` FAST PATH IS THE SAME NUMBERS — PROVEN, NOT ASSUMED
// -----------------------------------------------------------------------
// When period == 9.0 the CPU takes `n_order_ema_default_ema_into` (:802)
// instead of the filter. That path computes
//
//   acc = 0.2 * value;  acc -= -0.8 * old
//
// With period = 9: fc = 2/10 = 0.2 exactly, r = 0.8 exactly, b0 = 0.2 and
// a0 = -0.8. The seed is identical too — the fast path's `else` arm uses the
// safe source for `old`, while the filter's empty `y_hist` is prefilled with
// that same safe source. Both branches emit immediately.
//
// SHAPE — ONE THREAD PER COLUMN, BARS ASCENDING
// ---------------------------------------------
// A first-order IIR is a serial recurrence. Leading NaNs remain unstarted;
// creator `nz` maps later NaNs to zero without resetting. Infinity is outside
// Pine's representable series domain and fails this lane closed by resetting.
//
// ARITHMETIC
// ----------
// f64 end to end; no f32 literal, no f32-suffixed math function, no fast-math
// intrinsic. Listed in `F64_LANE_SOURCES`, so never `--use_fast_math`.

#include <cmath>
#include <cstdint>

__device__ __forceinline__ double noe_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

extern "C" __global__ void n_order_ema_neo_batch_f64(
    const double* __restrict__ data,
    int n,
    const int* __restrict__ periods,
    int n_combos,
    int first_valid,
    double* __restrict__ out)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_combos) return;

    double* __restrict__ row = out + (size_t)r * (size_t)n;
    const int period_i = periods[r];

    // resolve_params: period must be finite and >= 1.0. The lane hands ints.
    const bool bad_params = (n <= 0) || (period_i < 1);
    if (bad_params) {
        for (int i = 0; i < n; ++i) row[i] = noe_qnan();
        return;
    }

    const double period = (double)period_i;
    const double fc = 2.0 / (period + 1.0);
    const double b0 = fc;
    const double a0 = -(1.0 - fc);

    // The CPU walks from index 0. `first_valid` remains shape metadata only;
    // creator `nz` semantics are applied per bar after the first finite source.
    (void)first_valid;

    double y = 0.0;
    bool has_y = false;

    for (int i = 0; i < n; ++i) {
        const double x = data[i];
        if (isinf(x)) {
            has_y = false;
            row[i] = noe_qnan();
            continue;
        }
        if (isnan(x) && !has_y) {
            row[i] = noe_qnan();
            continue;
        }
        const double safe_x = isnan(x) ? 0.0 : x;

        const double y_in = has_y ? y : safe_x;

        double acc = b0 * safe_x;
        acc = acc - (a0 * y_in);

        y = acc;
        has_y = true;
        row[i] = acc;
    }
}
