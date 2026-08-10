#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>

static __device__ __forceinline__ float nan32() {
    return nanf("");
}


static __device__ __forceinline__ void kahan_add(float y, float& s, float& c) {
    float t = __fadd_rn(s, __fsub_rn(y, c));
    c = __fsub_rn(__fsub_rn(t, s), __fsub_rn(y, c));
    s = t;
}

static __device__ __forceinline__ void kahan_add_prod(float a, float b, float& s, float& c) {
    float p = __fmul_rn(a, b);
    float r = __fmaf_rn(a, b, -p);
    kahan_add(p, s, c);
    kahan_add(r, s, c);
}

struct ff {
    float hi;
    float lo;
};

static __device__ __forceinline__ ff two_sum(float a, float b) {
    ff res;
    float s = __fadd_rn(a, b);
    float bb = __fsub_rn(s, a);
    float e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb));
    res.hi = s;
    res.lo = e;
    return res;
}


struct lwma7_f32 {
    float buf[7];
    int   head;
    int   count;
    int   ticks;
    float s1, c1;
    float s2, c2;

    __device__ __forceinline__ void init() {
#pragma unroll
        for (int i = 0; i < 7; ++i) buf[i] = 0.f;
        head = 0; count = 0; ticks = 0; s1 = c1 = 0.f; s2 = c2 = 0.f;
    }

    __device__ __forceinline__ void push(float x) {
        if (count < 7) {
            buf[head] = x;
            head++; if (head == 7) head = 0;
            count++;

            kahan_add(x, s1, c1);
            kahan_add(__fmul_rn(static_cast<float>(count), x), s2, c2);
        } else {
            const float old = buf[head];
            buf[head] = x;
            head++; if (head == 7) head = 0;

            const float s1_old = s1;
            kahan_add(__fmaf_rn(7.f, x, -s1_old), s2, c2);

            kahan_add(x, s1, c1);
            kahan_add(-old, s1, c1);


            ticks++;
            if ((ticks & 0x3FF) == 0) {
                float ns1 = 0.f, nc1 = 0.f;
                float ns2 = 0.f, nc2 = 0.f;
#pragma unroll
                for (int i = 0; i < 7; ++i) {
                    const int idx = (head + i) % 7;
                    const float v = buf[idx];
                    kahan_add(v, ns1, nc1);
                    kahan_add(__fmul_rn(static_cast<float>(i + 1), v), ns2, nc2);
                }
                s1 = ns1; c1 = nc1; s2 = ns2; c2 = nc2;
            }
        }
    }

    __device__ __forceinline__ bool full() const { return count >= 7; }
    __device__ __forceinline__ float value() const { return __fmul_rn(s2, 1.0f / 28.0f); }
    __device__ __forceinline__ float newest() const {
        int idx = head - 1; if (idx < 0) idx += 7; return buf[idx];
    }
};


struct lwma4_ff {
    ff    buf[4];
    int   head;
    int   count;
    int   ticks;

    float s1h, c1h, s2h, c2h;

    float s1l, c1l, s2l, c2l;

    __device__ __forceinline__ void init() {
#pragma unroll
        for (int i = 0; i < 4; ++i) { buf[i].hi = 0.f; buf[i].lo = 0.f; }
        head = 0; count = 0; ticks = 0;
        s1h = c1h = s2h = c2h = 0.f;
        s1l = c1l = s2l = c2l = 0.f;
    }

    __device__ __forceinline__ void push(const ff& p) {
        if (count < 4) {
            buf[head] = p;
            head++; if (head == 4) head = 0;
            count++;

            kahan_add(p.hi, s1h, c1h);
            kahan_add(__fmul_rn(static_cast<float>(count), p.hi), s2h, c2h);
            kahan_add(p.lo, s1l, c1l);
            kahan_add(__fmul_rn(static_cast<float>(count), p.lo), s2l, c2l);
        } else {
            const ff old = buf[head];
            buf[head] = p;
            head++; if (head == 4) head = 0;

            const float s1h_old = s1h, s1l_old = s1l;

            kahan_add(__fmaf_rn(4.f, p.hi, -s1h_old), s2h, c2h);
            kahan_add(p.hi, s1h, c1h);
            kahan_add(-old.hi, s1h, c1h);

            kahan_add(__fmaf_rn(4.f, p.lo, -s1l_old), s2l, c2l);
            kahan_add(p.lo, s1l, c1l);
            kahan_add(-old.lo, s1l, c1l);


            ticks++;
            if ((ticks & 0x3FF) == 0) {
                float ns1h = 0.f, nc1h = 0.f, ns2h = 0.f, nc2h = 0.f;
                float ns1l = 0.f, nc1l = 0.f, ns2l = 0.f, nc2l = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int idx = (head + i) % 4;
                    const ff v = buf[idx];
                    const float w = static_cast<float>(i + 1);

                    kahan_add(v.hi, ns1h, nc1h);
                    kahan_add(__fmul_rn(w, v.hi), ns2h, nc2h);

                    kahan_add(v.lo, ns1l, nc1l);
                    kahan_add(__fmul_rn(w, v.lo), ns2l, nc2l);
                }
                s1h = ns1h; c1h = nc1h; s2h = ns2h; c2h = nc2h;
                s1l = ns1l; c1l = nc1l; s2l = ns2l; c2l = nc2l;
            }
        }
    }

    __device__ __forceinline__ bool full() const { return count >= 4; }
    __device__ __forceinline__ float value() const {

        return __fmul_rn(__fadd_rn(s2h, s2l), 0.1f);
    }
};


static __device__ __forceinline__ float wma7_from_prices_f32(const float* __restrict__ prices,
                                                             int idx) {
    float s = 0.f, c = 0.f;
#pragma unroll
    for (int k = 1, w = 7; k <= 7; ++k, --w) {
        kahan_add_prod(static_cast<float>(w), prices[idx - k], s, c);
    }
    return __fmul_rn(s, 1.0f / 28.0f);
}


static __device__ __forceinline__ float wma7_from_ring_f32(const float ring[7], int head) {
    float s = 0.f, c = 0.f;
    const float v0 = ring[(head + 6) % 7];
    const float v1 = ring[(head + 5) % 7];
    const float v2 = ring[(head + 4) % 7];
    const float v3 = ring[(head + 3) % 7];
    const float v4 = ring[(head + 2) % 7];
    const float v5 = ring[(head + 1) % 7];
    const float v6 = ring[(head + 0) % 7];
    kahan_add_prod(7.f, v0, s, c);
    kahan_add_prod(6.f, v1, s, c);
    kahan_add_prod(5.f, v2, s, c);
    kahan_add_prod(4.f, v3, s, c);
    kahan_add_prod(3.f, v4, s, c);
    kahan_add_prod(2.f, v5, s, c);
    kahan_add_prod(1.f, v6, s, c);
    return __fmul_rn(s, 1.0f / 28.0f);
}


static __device__ __forceinline__ float wma7_from_prices_tm_f32(const float* __restrict__ prices_tm,
                                                                int idx, int stride) {
    float s = 0.f, c = 0.f;
    kahan_add_prod(7.f, prices_tm[idx - stride], s, c);
    kahan_add_prod(6.f, prices_tm[idx - 2 * stride], s, c);
    kahan_add_prod(5.f, prices_tm[idx - 3 * stride], s, c);
    kahan_add_prod(4.f, prices_tm[idx - 4 * stride], s, c);
    kahan_add_prod(3.f, prices_tm[idx - 5 * stride], s, c);
    kahan_add_prod(2.f, prices_tm[idx - 6 * stride], s, c);
    kahan_add_prod(1.f, prices_tm[idx - 7 * stride], s, c);
    return __fmul_rn(s, 1.0f / 28.0f);
}


static __device__ __forceinline__ float trigger4_from_ff_ring(const ff pr[4], int head) {
    float s = 0.f, c = 0.f;
    const ff p0 = pr[(head + 0) % 4];
    const ff p1 = pr[(head + 1) % 4];
    const ff p2 = pr[(head + 2) % 4];
    const ff p3 = pr[(head + 3) % 4];

    kahan_add_prod(1.f, p0.hi, s, c);
    kahan_add_prod(2.f, p1.hi, s, c);
    kahan_add_prod(3.f, p2.hi, s, c);
    kahan_add_prod(4.f, p3.hi, s, c);

    kahan_add_prod(1.f, p0.lo, s, c);
    kahan_add_prod(2.f, p1.lo, s, c);
    kahan_add_prod(3.f, p2.lo, s, c);
    kahan_add_prod(4.f, p3.lo, s, c);
    return __fmul_rn(s, 0.1f);
}

static __device__ __forceinline__ void ehlers_pma_batch_core(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (threadIdx.x != 0) return;

    const float nan_f = nan32();
    if (series_len <= 0) return;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;


    const int warm_wma1    = first_valid + 7;
    const int warm_wma2    = first_valid + 13;
    const int warm_trigger = warm_wma2 + 3;

    float* predict_row = out_predict + combo * series_len;
    float* trigger_row = out_trigger + combo * series_len;


    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int i = 0; i < stop; ++i) { predict_row[i] = nan_f; }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int i = 0; i < stop; ++i) { trigger_row[i] = nan_f; }
    }


    if (warm_wma1 >= series_len) return;


    lwma7_f32 price_w7;  price_w7.init();
    lwma7_f32 wma1_w7;   wma1_w7.init();
    lwma4_ff  trig_w4;   trig_w4.init();


    for (int idx = first_valid; idx < series_len; ++idx) {


        float wma1_val = nan_f;
        if (price_w7.full()) {
            wma1_val = price_w7.value();
        }


        if (idx >= warm_wma1) {
            wma1_w7.push(wma1_val);

            if (wma1_w7.full()) {
                const float wma2_val = wma1_w7.value();
                const float current_wma1 = wma1_w7.newest();
                const float two_m = __fadd_rn(current_wma1, current_wma1);
                const ff     pred  = two_sum(two_m, -wma2_val);
                predict_row[idx]   = __fadd_rn(pred.hi, pred.lo);


                trig_w4.push(pred);
                if (trig_w4.full() && idx >= warm_trigger) {
                    trigger_row[idx] = trig_w4.value();
                }
            }
        }


        const float p_new = prices[idx];
        price_w7.push(p_new);
    }
}

extern "C" __global__ void ehlers_pma_batch_f32(const float* __restrict__ prices,
                                                 int series_len,
                                                 int n_combos,
                                                 int first_valid,
                                                 float* __restrict__ out_predict,
                                                 float* __restrict__ out_trigger) {
    ehlers_pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}


extern "C" __global__ void ehlers_pma_batch_tiled_f32_tile128(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger) {
    ehlers_pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}

extern "C" __global__ void ehlers_pma_batch_tiled_f32_tile256(
    const float* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    float* __restrict__ out_predict,
    float* __restrict__ out_trigger) {
    ehlers_pma_batch_core(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}

extern "C" __global__ void ehlers_pma_many_series_one_param_f32(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    const int series = blockIdx.x;
    if (series >= num_series) { return; }
    if (threadIdx.x != 0) { return; }

    const int stride = num_series;
    const float nan_f = nan32();

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = warm_wma1 + 6;
    const int warm_trigger = warm_wma2 + 3;


    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_predict_tm[idx] = nan_f;
        }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_trigger_tm[idx] = nan_f;
        }
    }

    if (first_valid + 7 >= series_len) return;

    lwma7_f32 price_w7; price_w7.init();
    lwma7_f32 wma1_w7;  wma1_w7.init();
    lwma4_ff  trig_w4;  trig_w4.init();

    for (int row = first_valid; row < series_len; ++row) {
        float wma1_val = nan_f;
        if (price_w7.full()) { wma1_val = price_w7.value(); }

        if (row >= warm_wma1) {
            wma1_w7.push(wma1_val);

            if (wma1_w7.full()) {
                const float wma2_val = wma1_w7.value();
                const float current_wma1 = wma1_w7.newest();
                const float two_m = __fadd_rn(current_wma1, current_wma1);
                const ff pred = two_sum(two_m, -wma2_val);
                const int idx = row * stride + series;
                out_predict_tm[idx] = __fadd_rn(pred.hi, pred.lo);

                trig_w4.push(pred);
                if (trig_w4.full() && row >= first_valid + 16) {
                    out_trigger_tm[idx] = trig_w4.value();
                }
            }
        }


        const int pidx = row * stride + series;
        price_w7.push(prices_tm[pidx]);
    }
}


extern "C" __global__ void ehlers_pma_ms1p_tiled_f32_tx1_ty2(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    int series0 = static_cast<int>(blockIdx.x) * 2;
    int local = static_cast<int>(threadIdx.y);
    int series = series0 + local;
    if (series >= num_series) { return; }
    if (threadIdx.x != 0) { return; }

    const int stride = num_series;
    const float nan_f = nan32();

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) { first_valid = 0; }
    if (first_valid >= series_len) { return; }

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = warm_wma1 + 6;
    const int warm_trigger = warm_wma2 + 3;
    if (warm_wma1 >= series_len) { return; }


    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_predict_tm[idx] = nan_f;
        }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_trigger_tm[idx] = nan_f;
        }
    }

    lwma7_f32 price_w7; price_w7.init();
    lwma7_f32 wma1_w7;  wma1_w7.init();
    lwma4_ff  trig_w4;  trig_w4.init();

    for (int row = first_valid; row < series_len; ++row) {
        float wma1_val = nan_f;
        if (price_w7.full()) { wma1_val = price_w7.value(); }

        if (row >= warm_wma1) {
            wma1_w7.push(wma1_val);

            if (wma1_w7.full()) {
                const float wma2_val = wma1_w7.value();
                const float current_wma1 = wma1_w7.newest();
                const float two_m = __fadd_rn(current_wma1, current_wma1);
                const ff pred = two_sum(two_m, -wma2_val);
                const int idx = row * stride + series;
                out_predict_tm[idx] = __fadd_rn(pred.hi, pred.lo);

                trig_w4.push(pred);
                if (trig_w4.full() && row >= first_valid + 16) {
                    out_trigger_tm[idx] = trig_w4.value();
                }
            }
        }

        const int pidx = row * stride + series;
        price_w7.push(prices_tm[pidx]);
    }
}

extern "C" __global__ void ehlers_pma_ms1p_tiled_f32_tx1_ty4(
    const float* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    float* __restrict__ out_predict_tm,
    float* __restrict__ out_trigger_tm) {
    int series0 = static_cast<int>(blockIdx.x) * 4;
    int local = static_cast<int>(threadIdx.y);
    int series = series0 + local;
    if (series >= num_series) { return; }
    if (threadIdx.x != 0) { return; }

    const int stride = num_series;
    const float nan_f = nan32();

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) { first_valid = 0; }
    if (first_valid >= series_len) { return; }

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = warm_wma1 + 6;
    const int warm_trigger = warm_wma2 + 3;
    if (warm_wma1 >= series_len) { return; }


    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_predict_tm[idx] = nan_f;
        }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_trigger_tm[idx] = nan_f;
        }
    }

    lwma7_f32 price_w7; price_w7.init();
    lwma7_f32 wma1_w7;  wma1_w7.init();
    lwma4_ff  trig_w4;  trig_w4.init();

    for (int row = first_valid; row < series_len; ++row) {
        float wma1_val = nan_f;
        if (price_w7.full()) { wma1_val = price_w7.value(); }

        if (row >= warm_wma1) {
            wma1_w7.push(wma1_val);

            if (wma1_w7.full()) {
                const float wma2_val = wma1_w7.value();
                const float current_wma1 = wma1_w7.newest();
                const float two_m = __fadd_rn(current_wma1, current_wma1);
                const ff pred = two_sum(two_m, -wma2_val);
                const int idx = row * stride + series;
                out_predict_tm[idx] = __fadd_rn(pred.hi, pred.lo);

                trig_w4.push(pred);
                if (trig_w4.full() && row >= first_valid + 16) {
                    out_trigger_tm[idx] = trig_w4.value();
                }
            }
        }

        const int pidx = row * stride + series;
        price_w7.push(prices_tm[pidx]);
    }
}


// ===========================================================================
// f64 LANE  --  shard S5
// ===========================================================================
//
// The f32 entry points above are LEFT IN PLACE because the generated f32
// dispatcher and this indicator's own `*_wrapper.rs` still launch them by
// name. Everything below is the SAME algorithm at f64, in this same file, and
// it is what the NeoEthos f64 lane consumes. Nothing here narrows, and nothing
// here is fast-math:
//
//   * every `float` data pointer, local and shared array is `double`
//   * every f32 literal lost its `f` suffix
//   * expf/sqrtf/fmaxf/fminf/fabsf/powf/logf -> exp/sqrt/fmax/fmin/fabs/pow/log
//   * __fadd_rn/__fsub_rn/__fmul_rn -> __dadd_rn/__dsub_rn/__dmul_rn
//     __fmaf_rn -> __fma_rn  (ONE rounding, matching `f64::mul_add`)
//     __fdividef -> __ddiv_rn and __frcp_rn -> __drcp_rn: those two are the
//     FAST APPROXIMATE divide and reciprocal, and their f64 images here are
//     the correctly-rounded operations, not a wider approximation
//   * an f32 NaN bit pattern is NOT a NaN when reinterpreted as f64 --
//     `__longlong_as_double(0x7fc00000)` is 2.09e-314, a finite denormal that
//     compares ORDERED against everything, so a warmup prefix meant to read
//     NaN would read ~0.0 instead. Every such site became the f64 pattern
//     (0x7ff8000000000000 / 0x7fffffffffffffff).
//   * every epsilon was RE-DERIVED at f64 width from the CPU reference rather
//     than carried over; see the per-file note where one exists.
// ===========================================================================

static __device__ __forceinline__ double nan32_f64() {
    return nan("");
}
static __device__ __forceinline__ void kahan_add_f64(double y, double& s, double& c) {
    double t = __dadd_rn(s, __dsub_rn(y, c));
    c = __dsub_rn(__dsub_rn(t, s), __dsub_rn(y, c));
    s = t;
}
static __device__ __forceinline__ void kahan_add_prod_f64(double a, double b, double& s, double& c) {
    double p = __dmul_rn(a, b);
    double r = __fma_rn(a, b, -p);
    kahan_add_f64(p, s, c);
    kahan_add_f64(r, s, c);
}
struct ff_f64 {
    double hi;
    double lo;
};
static __device__ __forceinline__ ff_f64 two_sum_f64(double a, double b) {
    ff_f64 res;
    double s = __dadd_rn(a, b);
    double bb = __dsub_rn(s, a);
    double e = __dadd_rn(__dsub_rn(a, __dsub_rn(s, bb)), __dsub_rn(b, bb));
    res.hi = s;
    res.lo = e;
    return res;
}
struct lwma7_f64 {
    double buf[7];
    int   head;
    int   count;
    int   ticks;
    double s1, c1;
    double s2, c2;

    __device__ __forceinline__ void init_f64() {
#pragma unroll
        for (int i = 0; i < 7; ++i) buf[i] = 0.;
        head = 0; count = 0; ticks = 0; s1 = c1 = 0.; s2 = c2 = 0.;
    }

    __device__ __forceinline__ void push_f64(double x) {
        if (count < 7) {
            buf[head] = x;
            head++; if (head == 7) head = 0;
            count++;

            kahan_add_f64(x, s1, c1);
            kahan_add_f64(__dmul_rn(static_cast<double>(count), x), s2, c2);
        } else {
            const double old = buf[head];
            buf[head] = x;
            head++; if (head == 7) head = 0;

            const double s1_old = s1;
            kahan_add_f64(__fma_rn(7., x, -s1_old), s2, c2);

            kahan_add_f64(x, s1, c1);
            kahan_add_f64(-old, s1, c1);


            ticks++;
            if ((ticks & 0x3FF) == 0) {
                double ns1 = 0., nc1 = 0.;
                double ns2 = 0., nc2 = 0.;
#pragma unroll
                for (int i = 0; i < 7; ++i) {
                    const int idx = (head + i) % 7;
                    const double v = buf[idx];
                    kahan_add_f64(v, ns1, nc1);
                    kahan_add_f64(__dmul_rn(static_cast<double>(i + 1), v), ns2, nc2);
                }
                s1 = ns1; c1 = nc1; s2 = ns2; c2 = nc2;
            }
        }
    }

    __device__ __forceinline__ bool full_f64() const { return count >= 7; }
    __device__ __forceinline__ double value_f64() const { return __dmul_rn(s2, 1.0 / 28.0); }
    __device__ __forceinline__ double newest_f64() const {
        int idx = head - 1; if (idx < 0) idx += 7; return buf[idx];
    }
};
struct lwma4_ff_f64 {
    ff_f64    buf[4];
    int   head;
    int   count;
    int   ticks;

    double s1h, c1h, s2h, c2h;

    double s1l, c1l, s2l, c2l;

    __device__ __forceinline__ void init_f64() {
#pragma unroll
        for (int i = 0; i < 4; ++i) { buf[i].hi = 0.; buf[i].lo = 0.; }
        head = 0; count = 0; ticks = 0;
        s1h = c1h = s2h = c2h = 0.;
        s1l = c1l = s2l = c2l = 0.;
    }

    __device__ __forceinline__ void push_f64(const ff_f64& p) {
        if (count < 4) {
            buf[head] = p;
            head++; if (head == 4) head = 0;
            count++;

            kahan_add_f64(p.hi, s1h, c1h);
            kahan_add_f64(__dmul_rn(static_cast<double>(count), p.hi), s2h, c2h);
            kahan_add_f64(p.lo, s1l, c1l);
            kahan_add_f64(__dmul_rn(static_cast<double>(count), p.lo), s2l, c2l);
        } else {
            const ff_f64 old = buf[head];
            buf[head] = p;
            head++; if (head == 4) head = 0;

            const double s1h_old = s1h, s1l_old = s1l;

            kahan_add_f64(__fma_rn(4., p.hi, -s1h_old), s2h, c2h);
            kahan_add_f64(p.hi, s1h, c1h);
            kahan_add_f64(-old.hi, s1h, c1h);

            kahan_add_f64(__fma_rn(4., p.lo, -s1l_old), s2l, c2l);
            kahan_add_f64(p.lo, s1l, c1l);
            kahan_add_f64(-old.lo, s1l, c1l);


            ticks++;
            if ((ticks & 0x3FF) == 0) {
                double ns1h = 0., nc1h = 0., ns2h = 0., nc2h = 0.;
                double ns1l = 0., nc1l = 0., ns2l = 0., nc2l = 0.;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int idx = (head + i) % 4;
                    const ff_f64 v = buf[idx];
                    const double w = static_cast<double>(i + 1);

                    kahan_add_f64(v.hi, ns1h, nc1h);
                    kahan_add_f64(__dmul_rn(w, v.hi), ns2h, nc2h);

                    kahan_add_f64(v.lo, ns1l, nc1l);
                    kahan_add_f64(__dmul_rn(w, v.lo), ns2l, nc2l);
                }
                s1h = ns1h; c1h = nc1h; s2h = ns2h; c2h = nc2h;
                s1l = ns1l; c1l = nc1l; s2l = ns2l; c2l = nc2l;
            }
        }
    }

    __device__ __forceinline__ bool full_f64() const { return count >= 4; }
    __device__ __forceinline__ double value_f64() const {

        return __dmul_rn(__dadd_rn(s2h, s2l), 0.1);
    }
};
static __device__ __forceinline__ double wma7_from_prices_f64(const double* __restrict__ prices,
                                                             int idx) {
    double s = 0., c = 0.;
#pragma unroll
    for (int k = 1, w = 7; k <= 7; ++k, --w) {
        kahan_add_prod_f64(static_cast<double>(w), prices[idx - k], s, c);
    }
    return __dmul_rn(s, 1.0 / 28.0);
}
static __device__ __forceinline__ double wma7_from_ring_f64(const double ring[7], int head) {
    double s = 0., c = 0.;
    const double v0 = ring[(head + 6) % 7];
    const double v1 = ring[(head + 5) % 7];
    const double v2 = ring[(head + 4) % 7];
    const double v3 = ring[(head + 3) % 7];
    const double v4 = ring[(head + 2) % 7];
    const double v5 = ring[(head + 1) % 7];
    const double v6 = ring[(head + 0) % 7];
    kahan_add_prod_f64(7., v0, s, c);
    kahan_add_prod_f64(6., v1, s, c);
    kahan_add_prod_f64(5., v2, s, c);
    kahan_add_prod_f64(4., v3, s, c);
    kahan_add_prod_f64(3., v4, s, c);
    kahan_add_prod_f64(2., v5, s, c);
    kahan_add_prod_f64(1., v6, s, c);
    return __dmul_rn(s, 1.0 / 28.0);
}
static __device__ __forceinline__ double wma7_from_prices_tm_f64(const double* __restrict__ prices_tm,
                                                                int idx, int stride) {
    double s = 0., c = 0.;
    kahan_add_prod_f64(7., prices_tm[idx - stride], s, c);
    kahan_add_prod_f64(6., prices_tm[idx - 2 * stride], s, c);
    kahan_add_prod_f64(5., prices_tm[idx - 3 * stride], s, c);
    kahan_add_prod_f64(4., prices_tm[idx - 4 * stride], s, c);
    kahan_add_prod_f64(3., prices_tm[idx - 5 * stride], s, c);
    kahan_add_prod_f64(2., prices_tm[idx - 6 * stride], s, c);
    kahan_add_prod_f64(1., prices_tm[idx - 7 * stride], s, c);
    return __dmul_rn(s, 1.0 / 28.0);
}
static __device__ __forceinline__ double trigger4_from_ff_ring_f64(const ff_f64 pr[4], int head) {
    double s = 0., c = 0.;
    const ff_f64 p0 = pr[(head + 0) % 4];
    const ff_f64 p1 = pr[(head + 1) % 4];
    const ff_f64 p2 = pr[(head + 2) % 4];
    const ff_f64 p3 = pr[(head + 3) % 4];

    kahan_add_prod_f64(1., p0.hi, s, c);
    kahan_add_prod_f64(2., p1.hi, s, c);
    kahan_add_prod_f64(3., p2.hi, s, c);
    kahan_add_prod_f64(4., p3.hi, s, c);

    kahan_add_prod_f64(1., p0.lo, s, c);
    kahan_add_prod_f64(2., p1.lo, s, c);
    kahan_add_prod_f64(3., p2.lo, s, c);
    kahan_add_prod_f64(4., p3.lo, s, c);
    return __dmul_rn(s, 0.1);
}
static __device__ __forceinline__ void ehlers_pma_batch_core_f64(
    const double* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    double* __restrict__ out_predict,
    double* __restrict__ out_trigger)
{
    const int combo = blockIdx.x;
    if (combo >= n_combos) return;
    if (threadIdx.x != 0) return;

    const double nan_f = nan32_f64();
    if (series_len <= 0) return;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;


    const int warm_wma1    = first_valid + 7;
    const int warm_wma2    = first_valid + 13;
    const int warm_trigger = warm_wma2 + 3;

    double* predict_row = out_predict + combo * series_len;
    double* trigger_row = out_trigger + combo * series_len;


    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int i = 0; i < stop; ++i) { predict_row[i] = nan_f; }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int i = 0; i < stop; ++i) { trigger_row[i] = nan_f; }
    }


    if (warm_wma1 >= series_len) return;


    lwma7_f64 price_w7;  price_w7.init_f64();
    lwma7_f64 wma1_w7;   wma1_w7.init_f64();
    lwma4_ff_f64  trig_w4;   trig_w4.init_f64();


    for (int idx = first_valid; idx < series_len; ++idx) {


        double wma1_val = nan_f;
        if (price_w7.full_f64()) {
            wma1_val = price_w7.value_f64();
        }


        if (idx >= warm_wma1) {
            wma1_w7.push_f64(wma1_val);

            if (wma1_w7.full_f64()) {
                const double wma2_val = wma1_w7.value_f64();
                const double current_wma1 = wma1_w7.newest_f64();
                const double two_m = __dadd_rn(current_wma1, current_wma1);
                const ff_f64     pred  = two_sum_f64(two_m, -wma2_val);
                predict_row[idx]   = __dadd_rn(pred.hi, pred.lo);


                trig_w4.push_f64(pred);
                if (trig_w4.full_f64() && idx >= warm_trigger) {
                    trigger_row[idx] = trig_w4.value_f64();
                }
            }
        }


        const double p_new = prices[idx];
        price_w7.push_f64(p_new);
    }
}
extern "C" __global__ void ehlers_pma_batch_f64(const double* __restrict__ prices,
                                                 int series_len,
                                                 int n_combos,
                                                 int first_valid,
                                                 double* __restrict__ out_predict,
                                                 double* __restrict__ out_trigger) {
    ehlers_pma_batch_core_f64(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}
extern "C" __global__ void ehlers_pma_batch_tiled_f64_tile128(
    const double* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    double* __restrict__ out_predict,
    double* __restrict__ out_trigger) {
    ehlers_pma_batch_core_f64(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}
extern "C" __global__ void ehlers_pma_batch_tiled_f64_tile256(
    const double* __restrict__ prices,
    int series_len,
    int n_combos,
    int first_valid,
    double* __restrict__ out_predict,
    double* __restrict__ out_trigger) {
    ehlers_pma_batch_core_f64(prices, series_len, n_combos, first_valid, out_predict, out_trigger);
}
extern "C" __global__ void ehlers_pma_many_series_one_param_f64(
    const double* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    double* __restrict__ out_predict_tm,
    double* __restrict__ out_trigger_tm) {
    const int series = blockIdx.x;
    if (series >= num_series) { return; }
    if (threadIdx.x != 0) { return; }

    const int stride = num_series;
    const double nan_f = nan32_f64();

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) first_valid = 0;
    if (first_valid >= series_len) return;

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = warm_wma1 + 6;
    const int warm_trigger = warm_wma2 + 3;


    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_predict_tm[idx] = nan_f;
        }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_trigger_tm[idx] = nan_f;
        }
    }

    if (first_valid + 7 >= series_len) return;

    lwma7_f64 price_w7; price_w7.init_f64();
    lwma7_f64 wma1_w7;  wma1_w7.init_f64();
    lwma4_ff_f64  trig_w4;  trig_w4.init_f64();

    for (int row = first_valid; row < series_len; ++row) {
        double wma1_val = nan_f;
        if (price_w7.full_f64()) { wma1_val = price_w7.value_f64(); }

        if (row >= warm_wma1) {
            wma1_w7.push_f64(wma1_val);

            if (wma1_w7.full_f64()) {
                const double wma2_val = wma1_w7.value_f64();
                const double current_wma1 = wma1_w7.newest_f64();
                const double two_m = __dadd_rn(current_wma1, current_wma1);
                const ff_f64 pred = two_sum_f64(two_m, -wma2_val);
                const int idx = row * stride + series;
                out_predict_tm[idx] = __dadd_rn(pred.hi, pred.lo);

                trig_w4.push_f64(pred);
                if (trig_w4.full_f64() && row >= first_valid + 16) {
                    out_trigger_tm[idx] = trig_w4.value_f64();
                }
            }
        }


        const int pidx = row * stride + series;
        price_w7.push_f64(prices_tm[pidx]);
    }
}
extern "C" __global__ void ehlers_pma_ms1p_tiled_f64_tx1_ty2(
    const double* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    double* __restrict__ out_predict_tm,
    double* __restrict__ out_trigger_tm) {
    int series0 = static_cast<int>(blockIdx.x) * 2;
    int local = static_cast<int>(threadIdx.y);
    int series = series0 + local;
    if (series >= num_series) { return; }
    if (threadIdx.x != 0) { return; }

    const int stride = num_series;
    const double nan_f = nan32_f64();

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) { first_valid = 0; }
    if (first_valid >= series_len) { return; }

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = warm_wma1 + 6;
    const int warm_trigger = warm_wma2 + 3;
    if (warm_wma1 >= series_len) { return; }


    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_predict_tm[idx] = nan_f;
        }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_trigger_tm[idx] = nan_f;
        }
    }

    lwma7_f64 price_w7; price_w7.init_f64();
    lwma7_f64 wma1_w7;  wma1_w7.init_f64();
    lwma4_ff_f64  trig_w4;  trig_w4.init_f64();

    for (int row = first_valid; row < series_len; ++row) {
        double wma1_val = nan_f;
        if (price_w7.full_f64()) { wma1_val = price_w7.value_f64(); }

        if (row >= warm_wma1) {
            wma1_w7.push_f64(wma1_val);

            if (wma1_w7.full_f64()) {
                const double wma2_val = wma1_w7.value_f64();
                const double current_wma1 = wma1_w7.newest_f64();
                const double two_m = __dadd_rn(current_wma1, current_wma1);
                const ff_f64 pred = two_sum_f64(two_m, -wma2_val);
                const int idx = row * stride + series;
                out_predict_tm[idx] = __dadd_rn(pred.hi, pred.lo);

                trig_w4.push_f64(pred);
                if (trig_w4.full_f64() && row >= first_valid + 16) {
                    out_trigger_tm[idx] = trig_w4.value_f64();
                }
            }
        }

        const int pidx = row * stride + series;
        price_w7.push_f64(prices_tm[pidx]);
    }
}
extern "C" __global__ void ehlers_pma_ms1p_tiled_f64_tx1_ty4(
    const double* __restrict__ prices_tm,
    int num_series,
    int series_len,
    const int* __restrict__ first_valids,
    double* __restrict__ out_predict_tm,
    double* __restrict__ out_trigger_tm) {
    int series0 = static_cast<int>(blockIdx.x) * 4;
    int local = static_cast<int>(threadIdx.y);
    int series = series0 + local;
    if (series >= num_series) { return; }
    if (threadIdx.x != 0) { return; }

    const int stride = num_series;
    const double nan_f = nan32_f64();

    int first_valid = first_valids ? first_valids[series] : 0;
    if (first_valid < 0) { first_valid = 0; }
    if (first_valid >= series_len) { return; }

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = warm_wma1 + 6;
    const int warm_trigger = warm_wma2 + 3;
    if (warm_wma1 >= series_len) { return; }


    {
        int stop = (series_len < warm_wma2) ? series_len : warm_wma2;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_predict_tm[idx] = nan_f;
        }
    }
    {
        int stop = (series_len < warm_trigger) ? series_len : warm_trigger;
        for (int row = 0; row < stop; ++row) {
            const int idx = row * stride + series;
            out_trigger_tm[idx] = nan_f;
        }
    }

    lwma7_f64 price_w7; price_w7.init_f64();
    lwma7_f64 wma1_w7;  wma1_w7.init_f64();
    lwma4_ff_f64  trig_w4;  trig_w4.init_f64();

    for (int row = first_valid; row < series_len; ++row) {
        double wma1_val = nan_f;
        if (price_w7.full_f64()) { wma1_val = price_w7.value_f64(); }

        if (row >= warm_wma1) {
            wma1_w7.push_f64(wma1_val);

            if (wma1_w7.full_f64()) {
                const double wma2_val = wma1_w7.value_f64();
                const double current_wma1 = wma1_w7.newest_f64();
                const double two_m = __dadd_rn(current_wma1, current_wma1);
                const ff_f64 pred = two_sum_f64(two_m, -wma2_val);
                const int idx = row * stride + series;
                out_predict_tm[idx] = __dadd_rn(pred.hi, pred.lo);

                trig_w4.push_f64(pred);
                if (trig_w4.full_f64() && row >= first_valid + 16) {
                    out_trigger_tm[idx] = trig_w4.value_f64();
                }
            }
        }

        const int pidx = row * stride + series;
        price_w7.push_f64(prices_tm[pidx]);
    }
}

/* ===========================================================================
 * NEOETHOS f64 LANE — ehlers_pma
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/moving_averages/ehlers_pma.rs:451
 *   `ehlers_pma_scalar`, reached from `ehlers_pma_with_kernel` (:295).
 *
 * Column: `ma_batch.rs:1679` routes the id `ehlers_pma` by calling
 *   `ehlers_pma_with_kernel` ONCE with `EhlersPmaParams::default()` and then
 *   REPEATING `out.predict` for every period in the sweep. So this indicator is
 *   PERIOD-INVARIANT by construction of the dispatcher itself, not merely
 *   because a parameter is named differently — and the emitted column is
 *   `predict`, never `trigger`.
 *
 * Warmups, verbatim from :319-322:
 *   warm_wma1 = first + 7, warm_wma2 = first + 13, warm_predict = warm_wma2.
 *   `predict` is NaN below warm_predict.
 *
 * Arithmetic: both CPU forms (`ehlers_pma_scalar`, :451, and
 *   `ehlers_pma_scalar_direct`, :380) compute the same two seven-tap weighted
 *   sums in the same descending-weight order and scale by 1/28 at the end, so
 *   there is one number to match, not two. The weights are exact small
 *   integers, so `7.0 * x` is exact and no fma contraction is needed or wanted.
 *
 * Shape: ONE THREAD PER COLUMN. The second stage reads the first stage seven
 *   bars back, so the row keeps a seven-entry ring rather than a second buffer.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

extern "C" __global__
void ehlers_pma_neo_batch_f64(const double* __restrict__ data,
                              int n,
                              const int* __restrict__ periods,
                              int n_combos,
                              int first_valid,
                              double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos || n <= 0) return;
    (void)periods;

    double* __restrict__ o = out + (size_t)combo * (size_t)n;
    for (int i = 0; i < n; ++i) o[i] = NEO_F64_NAN;

    if (first_valid < 0 || first_valid >= n) return;
    if (n - first_valid < 14) return;      /* MIN_REQUIRED (:311) */

    const int warm_wma1 = first_valid + 7;
    const int warm_wma2 = first_valid + 13;
    if (warm_wma1 >= n) return;

    const double inv28 = 1.0 / 28.0;

    double w_ring[7];
    for (int i = 0; i < 7; ++i) w_ring[i] = 0.0;
    int w_head = 0;

    for (int i = warm_wma1; i < n; ++i) {
        const double w1 = (7.0 * data[i - 1]
                         + 6.0 * data[i - 2]
                         + 5.0 * data[i - 3]
                         + 4.0 * data[i - 4]
                         + 3.0 * data[i - 5]
                         + 2.0 * data[i - 6]
                         + 1.0 * data[i - 7]) * inv28;

        w_ring[w_head] = w1;
        w_head = w_head + 1; if (w_head == 7) w_head = 0;

        if (i < warm_wma2) continue;

        const int k0 = (w_head == 0) ? 6 : (w_head - 1);
        const int k1 = (k0 == 0) ? 6 : (k0 - 1);
        const int k2 = (k1 == 0) ? 6 : (k1 - 1);
        const int k3 = (k2 == 0) ? 6 : (k2 - 1);
        const int k4 = (k3 == 0) ? 6 : (k3 - 1);
        const int k5 = (k4 == 0) ? 6 : (k4 - 1);
        const int k6 = (k5 == 0) ? 6 : (k5 - 1);

        const double w2 = (7.0 * w_ring[k0]
                         + 6.0 * w_ring[k1]
                         + 5.0 * w_ring[k2]
                         + 4.0 * w_ring[k3]
                         + 3.0 * w_ring[k4]
                         + 2.0 * w_ring[k5]
                         + 1.0 * w_ring[k6]) * inv28;

        o[i] = 2.0 * w1 - w2;
    }
}
