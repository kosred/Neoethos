// n_order_ema — CUDA f64 kernel.
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
//   :548 base_lookback / :553 warmup_len
//   :568 required_valid_len
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
// and gates on `count > 8`. With period = 9: fc = 2/10 = 0.2 exactly,
// r = 0.8 exactly, b0 = 0.2, a0 = -0.8, and warmup = 1 * (9 - 1) = 8. The seed
// is identical too — the fast path's `else` arm uses `value` for `old`, the
// filter's empty `y_hist` uses `first`, and at the first bar of a run those
// are the same number. So ONE implementation serves both branches; there is no
// period at which they disagree.
//
// SHAPE — ONE THREAD PER COLUMN, BARS ASCENDING
// ---------------------------------------------
// A first-order IIR is a serial recurrence and a non-finite bar RESETS it
// (:377, :806-810), which no scan reformulation survives. One thread walks the
// column.
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

    // resolve_params (:666): period must be finite, >= 1.0, and
    // `period.ceil() as usize <= len`. The lane hands whole numbers, so
    // ceil(period) == period.
    const bool bad_params = (n <= 0) || (period_i < 1) || (period_i > n);
    if (bad_params) {
        for (int i = 0; i < n; ++i) row[i] = noe_qnan();
        return;
    }

    // warmup_len (:553) for Ema = base_lookback (:548) = order * (ceil(period)
    // - 1), and order is 1.
    const int warmup = period_i - 1;

    // validate_input (:704): the series must contain a RUN of
    // required_valid_len (:568) = warmup + 1 = period consecutive finite
    // values, otherwise the CPU returns Err and produces no series at all.
    // AllValuesNaN (:730) is the same rejection by another name.
    {
        const int needed = warmup + 1;
        int cur = 0;
        bool ok = false;
        for (int i = 0; i < n; ++i) {
            if (isfinite(data[i])) {
                cur += 1;
                if (cur >= needed) { ok = true; break; }
            } else {
                cur = 0;
            }
        }
        if (!ok) {
            for (int i = 0; i < n; ++i) row[i] = noe_qnan();
            return;
        }
    }

    const double period = (double)period_i;
    const double fc = 2.0 / (period + 1.0);
    const double b0 = fc;
    const double a0 = -(1.0 - fc);

    // `first_valid` is deliberately UNUSED for the series start: the CPU walks
    // from index 0 and resets on every non-finite bar (:781-792, :802-832), so
    // there is no single warmup prefix to hang off it. The row is registered
    // F64FirstValidRule::Ignored for that reason.
    (void)first_valid;

    double y = 0.0;
    bool has_y = false;
    int count = 0;

    for (int i = 0; i < n; ++i) {
        const double x = data[i];
        if (!isfinite(x)) {
            // update (:374-378) resets the whole filter and returns None.
            has_y = false;
            count = 0;
            row[i] = noe_qnan();
            continue;
        }

        // `first_value.get_or_insert(value)` (:379): on the first bar of a run
        // the history slot reads `first`, which IS this bar's value.
        const double y_in = has_y ? y : x;

        double acc = b0 * x;
        acc = acc - (a0 * y_in);

        y = acc;
        has_y = true;
        count += 1;

        // NOrderEmaStream::update (:527-531) emits only past the warmup.
        row[i] = (count > warmup) ? acc : noe_qnan();
    }
}
