extern "C" {

#if __CUDACC_VER_MAJOR__ >= 10
#define UO_LDG(x) __ldg(&(x))
#else
#define UO_LDG(x) (x)
#endif


__device__ __constant__ float UO_W1 = 100.0f * (4.0f / 7.0f);
__device__ __constant__ float UO_W2 = 100.0f * (2.0f / 7.0f);
__device__ __constant__ float UO_W3 = 100.0f * (1.0f / 7.0f);


__device__ __forceinline__ float2 ldg_float2(const float2* __restrict__ base, int idx)
{
    float2 v; v.x = __ldg(&base[idx].x); v.y = __ldg(&base[idx].y); return v;
}


__device__ __forceinline__ void d_to_ds(float& hi, float& lo, const double d)
{

    hi = (float)d;

    lo = (float)(d - (double)hi);
}


__device__ __forceinline__ float ds_diff_to_f(
    const float ah, const float al,
    const float bh, const float bl)
{

    float s  = ah - bh;
    float vb = s - ah;
    float e  = (ah - (s - vb)) - (bh + vb);


    float t   = (al - bl);
    float s2  = s + t;
    float vb2 = s2 - s;
    e += (t - vb2);


    float hi = s2 + e;
    float lo = (s2 - hi) + e;
    return hi + lo;
}


__device__ __forceinline__ float recip_nr1(float x)
{
    float r = __frcp_rn(x);
    r = r * (2.0f - x * r);
    r = r * (2.0f - x * r);
    return r;
}

__device__ __forceinline__ float uo_nan() { return __int_as_float(0x7fffffff); }

__global__ void ultosc_build_prefix_sums_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    int len,
    int first,
    float2* __restrict__ pcmtl,
    float2* __restrict__ ptr)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    pcmtl[0] = make_float2(0.0f, 0.0f);
    ptr[0] = make_float2(0.0f, 0.0f);

    double pcmtl_acc = 0.0;
    double ptr_acc = 0.0;
    for (int i = 0; i < len; ++i) {
        double add_c = 0.0;
        double add_t = 0.0;
        if (i >= first) {
            const double hi = (double)high[i];
            const double lo = (double)low[i];
            const double ci = (double)close[i];
            const double pc = (double)close[i - 1];
            const double tl = lo < pc ? lo : pc;
            double trv = hi - lo;
            const double d1 = fabs(hi - pc);
            if (d1 > trv) trv = d1;
            const double d2 = fabs(lo - pc);
            if (d2 > trv) trv = d2;
            add_c = ci - tl;
            add_t = trv;
        }

        pcmtl_acc += add_c;
        ptr_acc += add_t;
        d_to_ds(pcmtl[i + 1].x, pcmtl[i + 1].y, pcmtl_acc);
        d_to_ds(ptr[i + 1].x, ptr[i + 1].y, ptr_acc);
    }
}


__global__ void ultosc_batch_f32(
    const float2* __restrict__ pcmtl,
    const float2* __restrict__ ptr,
    int len,
    int first,
    const int3* __restrict__ periods,
    int nrows,
    float* __restrict__ out)
{
    const int row = blockIdx.y;
    if (row >= nrows) return;


    __shared__ int sp1, sp2, sp3, sstart;
    if (threadIdx.x == 0) {
        const int3 p = periods[row];
        const int p1 = p.x;
        const int p2 = p.y;
        const int p3 = p.z;
        sp1 = p1; sp2 = p2; sp3 = p3;
        const int maxp = max(p1, max(p2, p3));
        sstart = first + maxp - 1;
    }
    __syncthreads();

    float* __restrict__ row_out = out + (size_t)row * (size_t)len;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += stride)
    {
        if (i < sstart) { row_out[i] = uo_nan(); continue; }

        const int a  = i + 1;
        const int i1 = a - sp1;
        const int i2 = a - sp2;
        const int i3 = a - sp3;


        const float2 c_now  = ldg_float2(pcmtl, a);
        const float2 c_p1   = ldg_float2(pcmtl, i1);
        const float2 c_p2   = ldg_float2(pcmtl, i2);
        const float2 c_p3   = ldg_float2(pcmtl, i3);
        const float2 tr_now = ldg_float2(ptr, a);
        const float2 tr_p1  = ldg_float2(ptr, i1);
        const float2 tr_p2  = ldg_float2(ptr, i2);
        const float2 tr_p3  = ldg_float2(ptr, i3);


        const float s1a = ds_diff_to_f(c_now.x,  c_now.y,  c_p1.x,  c_p1.y);
        const float s1b = ds_diff_to_f(tr_now.x, tr_now.y, tr_p1.x, tr_p1.y);
        const float s2a = ds_diff_to_f(c_now.x,  c_now.y,  c_p2.x,  c_p2.y);
        const float s2b = ds_diff_to_f(tr_now.x, tr_now.y, tr_p2.x, tr_p2.y);
        const float s3a = ds_diff_to_f(c_now.x,  c_now.y,  c_p3.x,  c_p3.y);
        const float s3b = ds_diff_to_f(tr_now.x, tr_now.y, tr_p3.x, tr_p3.y);

        const float t1 = (s1b != 0.0f) ? (s1a * recip_nr1(s1b)) : 0.0f;
        const float t2 = (s2b != 0.0f) ? (s2a * recip_nr1(s2b)) : 0.0f;
        const float t3 = (s3b != 0.0f) ? (s3a * recip_nr1(s3b)) : 0.0f;

        row_out[i] = fmaf(UO_W1, t1, fmaf(UO_W2, t2, UO_W3 * t3));
    }
}


__global__ void ultosc_many_series_one_param_f32(
    const float2* __restrict__ pcmtl_tm,
    const float2* __restrict__ ptr_tm,
    int cols,
    int rows,
    int p1,
    int p2,
    int p3,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{

    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (s >= cols) return;

    const int maxp  = max(p1, max(p2, p3));
    const int first = UO_LDG(first_valids[s]);
    const int start = first + maxp - 1;

    const int t_stride = blockDim.x * gridDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < rows; t += t_stride)
    {
        float* out_row = out_tm + (size_t)t * (size_t)cols;
        if (t < start) { out_row[s] = uo_nan(); continue; }

        const int idx_now = (t + 1) * cols + s;
        const int idx_1   = idx_now - p1 * cols;
        const int idx_2   = idx_now - p2 * cols;
        const int idx_3   = idx_now - p3 * cols;

        const float2 c_now  = ldg_float2(pcmtl_tm, idx_now);
        const float2 c_1    = ldg_float2(pcmtl_tm, idx_1);
        const float2 c_2    = ldg_float2(pcmtl_tm, idx_2);
        const float2 c_3    = ldg_float2(pcmtl_tm, idx_3);
        const float2 tr_now = ldg_float2(ptr_tm, idx_now);
        const float2 tr_1   = ldg_float2(ptr_tm, idx_1);
        const float2 tr_2   = ldg_float2(ptr_tm, idx_2);
        const float2 tr_3   = ldg_float2(ptr_tm, idx_3);

        const float s1a = ds_diff_to_f(c_now.x,  c_now.y,  c_1.x,  c_1.y);
        const float s1b = ds_diff_to_f(tr_now.x, tr_now.y, tr_1.x, tr_1.y);
        const float s2a = ds_diff_to_f(c_now.x,  c_now.y,  c_2.x,  c_2.y);
        const float s2b = ds_diff_to_f(tr_now.x, tr_now.y, tr_2.x, tr_2.y);
        const float s3a = ds_diff_to_f(c_now.x,  c_now.y,  c_3.x,  c_3.y);
        const float s3b = ds_diff_to_f(tr_now.x, tr_now.y, tr_3.x, tr_3.y);

        const float t1 = (s1b != 0.0f) ? (s1a * recip_nr1(s1b)) : 0.0f;
        const float t2 = (s2b != 0.0f) ? (s2a * recip_nr1(s2b)) : 0.0f;
        const float t3 = (s3b != 0.0f) ? (s3a * recip_nr1(s3b)) : 0.0f;

        out_row[s] = fmaf(UO_W1, t1, fmaf(UO_W2, t2, UO_W3 * t3));
    }
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

extern "C" {

#if __CUDACC_VER_MAJOR__ >= 10
#define UO_LDG_F64(x) __ldg(&(x))
#else
#define UO_LDG_F64(x) (x)
#endif


__device__ __constant__ double UO_W1_F64 = 100.0 * (4.0 / 7.0);
__device__ __constant__ double UO_W2_F64 = 100.0 * (2.0 / 7.0);
__device__ __constant__ double UO_W3_F64 = 100.0 * (1.0 / 7.0);


__device__ __forceinline__ double2 ldg_float2_f64(const double2* __restrict__ base, int idx)
{
    double2 v; v.x = __ldg(&base[idx].x); v.y = __ldg(&base[idx].y); return v;
}


__device__ __forceinline__ void d_to_ds_f64(double& hi, double& lo, const double d)
{

    hi = (double)d;

    lo = (double)(d - (double)hi);
}


__device__ __forceinline__ double ds_diff_to_f_f64(
    const double ah, const double al,
    const double bh, const double bl)
{

    double s  = ah - bh;
    double vb = s - ah;
    double e  = (ah - (s - vb)) - (bh + vb);


    double t   = (al - bl);
    double s2  = s + t;
    double vb2 = s2 - s;
    e += (t - vb2);


    double hi = s2 + e;
    double lo = (s2 - hi) + e;
    return hi + lo;
}


// S5 CORRECTION -- an f32-sized refinement is WRONG in f64.
//
// The f32 original (`recip_nr1`, L52) seeds with `__frcp_rn`, which is the FAST
// APPROXIMATE reciprocal, and then runs two Newton-Raphson steps to claw the
// missing bits back. Carrying that shape into f64 keeps the two NR steps but
// seeds them with `__drcp_rn`, which is the CORRECTLY-ROUNDED reciprocal --
// bit-identical to `1.0 / x`. Refining an already-correctly-rounded value
// cannot improve it; each NR step is two more roundings that can only move the
// result AWAY from `1/x`.
//
// The CPU reference is `ultosc.rs:650` -- `sum1_a * sum1_b.recip()`, where
// `f64::recip()` is `1.0 / x`: ONE rounding for the reciprocal, then ONE for
// the multiply. So this is `1.0 / x` and nothing else, and the caller keeps the
// separate multiply so the rounding COUNT matches the CPU line exactly.
__device__ __forceinline__ double recip_nr1_f64(double x)
{
    return 1.0 / x;
}

__device__ __forceinline__ double uo_nan_f64() { return __longlong_as_double(0x7fffffffffffffffULL); }

__global__ void ultosc_build_prefix_sums_f64(
    const double* __restrict__ high,
    const double* __restrict__ low,
    const double* __restrict__ close,
    int len,
    int first,
    double2* __restrict__ pcmtl,
    double2* __restrict__ ptr)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len < 0) return;

    pcmtl[0] = make_double2(0.0, 0.0);
    ptr[0] = make_double2(0.0, 0.0);

    double pcmtl_acc = 0.0;
    double ptr_acc = 0.0;
    for (int i = 0; i < len; ++i) {
        double add_c = 0.0;
        double add_t = 0.0;
        if (i >= first) {
            const double hi = high[i];
            const double lo = low[i];
            const double ci = close[i];
            const double pc = close[i - 1];

            // S5 CORRECTION 1 -- THE NaN GATE THE ORIGINAL NEVER HAD.
            //
            // `ultosc.rs:598` computes `valid = !(hi|lo|ci|prev_c).is_nan()`
            // and contributes `(0.0, 0.0)` to BOTH running sums for an invalid
            // bar. The original gated on `i >= first` ALONE, so one NaN bar
            // anywhere after `first` entered a RUNNING PREFIX SUM -- and a
            // prefix sum does not forget: every bar from there to the end of
            // the series reads NaN where the CPU emits a number. This is the
            // poisoned-recursion mode in its worst form, because the damage is
            // unbounded in time rather than decaying.
            const bool valid = !(isnan(hi) || isnan(lo) || isnan(ci) || isnan(pc));
            if (valid) {
                // S5 CORRECTION 2 -- the CPU's TWO-TERM true range.
                //
                // `ultosc.rs:600-604` is `th - tl`, `th = max(hi, prev_c)`,
                // `tl = min(lo, prev_c)`: ONE subtraction of two SELECTED
                // operands. The original built Wilder's three-candidate
                // `max(hi-lo, |hi-pc|, |lo-pc|)`. The two agree in value on any
                // bar with `hi >= lo`, but only this form is what the reference
                // rounds, and only this form stays right on a malformed
                // `hi < lo` bar -- which this repository has measured in its own
                // store.
                //
                // `fmin`/`fmax`, not a comparison chain: they return the non-NaN
                // operand, which is `f64::min`/`f64::max` semantics. The gate
                // above already excludes NaN, so this is belt and braces -- but
                // a bare comparison chain is the wrong shape to leave behind.
                const double tl = fmin(lo, pc);
                const double th = fmax(hi, pc);
                add_c = ci - tl;
                add_t = th - tl;
            }
        }

        pcmtl_acc += add_c;
        ptr_acc += add_t;
        d_to_ds_f64(pcmtl[i + 1].x, pcmtl[i + 1].y, pcmtl_acc);
        d_to_ds_f64(ptr[i + 1].x, ptr[i + 1].y, ptr_acc);
    }
}


__global__ void ultosc_batch_f64(
    const double2* __restrict__ pcmtl,
    const double2* __restrict__ ptr,
    int len,
    int first,
    const int3* __restrict__ periods,
    int nrows,
    double* __restrict__ out)
{
    const int row = blockIdx.y;
    if (row >= nrows) return;


    __shared__ int sp1, sp2, sp3, sstart;
    if (threadIdx.x == 0) {
        const int3 p = periods[row];
        const int p1 = p.x;
        const int p2 = p.y;
        const int p3 = p.z;
        sp1 = p1; sp2 = p2; sp3 = p3;
        const int maxp = max(p1, max(p2, p3));
        sstart = first + maxp - 1;
    }
    __syncthreads();

    double* __restrict__ row_out = out + (size_t)row * (size_t)len;
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < len; i += stride)
    {
        if (i < sstart) { row_out[i] = uo_nan_f64(); continue; }

        const int a  = i + 1;
        const int i1 = a - sp1;
        const int i2 = a - sp2;
        const int i3 = a - sp3;


        const double2 c_now  = ldg_float2_f64(pcmtl, a);
        const double2 c_p1   = ldg_float2_f64(pcmtl, i1);
        const double2 c_p2   = ldg_float2_f64(pcmtl, i2);
        const double2 c_p3   = ldg_float2_f64(pcmtl, i3);
        const double2 tr_now = ldg_float2_f64(ptr, a);
        const double2 tr_p1  = ldg_float2_f64(ptr, i1);
        const double2 tr_p2  = ldg_float2_f64(ptr, i2);
        const double2 tr_p3  = ldg_float2_f64(ptr, i3);


        const double s1a = ds_diff_to_f_f64(c_now.x,  c_now.y,  c_p1.x,  c_p1.y);
        const double s1b = ds_diff_to_f_f64(tr_now.x, tr_now.y, tr_p1.x, tr_p1.y);
        const double s2a = ds_diff_to_f_f64(c_now.x,  c_now.y,  c_p2.x,  c_p2.y);
        const double s2b = ds_diff_to_f_f64(tr_now.x, tr_now.y, tr_p2.x, tr_p2.y);
        const double s3a = ds_diff_to_f_f64(c_now.x,  c_now.y,  c_p3.x,  c_p3.y);
        const double s3b = ds_diff_to_f_f64(tr_now.x, tr_now.y, tr_p3.x, tr_p3.y);

        const double t1 = (s1b != 0.0) ? (s1a * recip_nr1_f64(s1b)) : 0.0;
        const double t2 = (s2b != 0.0) ? (s2a * recip_nr1_f64(s2b)) : 0.0;
        const double t3 = (s3b != 0.0) ? (s3a * recip_nr1_f64(s3b)) : 0.0;

        row_out[i] = fma(UO_W1_F64, t1, fma(UO_W2_F64, t2, UO_W3_F64 * t3));
    }
}


__global__ void ultosc_many_series_one_param_f64(
    const double2* __restrict__ pcmtl_tm,
    const double2* __restrict__ ptr_tm,
    int cols,
    int rows,
    int p1,
    int p2,
    int p3,
    const int* __restrict__ first_valids,
    double* __restrict__ out_tm)
{

    const int s = blockIdx.y * blockDim.y + threadIdx.y;
    if (s >= cols) return;

    const int maxp  = max(p1, max(p2, p3));
    const int first = UO_LDG_F64(first_valids[s]);
    const int start = first + maxp - 1;

    const int t_stride = blockDim.x * gridDim.x;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < rows; t += t_stride)
    {
        double* out_row = out_tm + (size_t)t * (size_t)cols;
        if (t < start) { out_row[s] = uo_nan_f64(); continue; }

        const int idx_now = (t + 1) * cols + s;
        const int idx_1   = idx_now - p1 * cols;
        const int idx_2   = idx_now - p2 * cols;
        const int idx_3   = idx_now - p3 * cols;

        const double2 c_now  = ldg_float2_f64(pcmtl_tm, idx_now);
        const double2 c_1    = ldg_float2_f64(pcmtl_tm, idx_1);
        const double2 c_2    = ldg_float2_f64(pcmtl_tm, idx_2);
        const double2 c_3    = ldg_float2_f64(pcmtl_tm, idx_3);
        const double2 tr_now = ldg_float2_f64(ptr_tm, idx_now);
        const double2 tr_1   = ldg_float2_f64(ptr_tm, idx_1);
        const double2 tr_2   = ldg_float2_f64(ptr_tm, idx_2);
        const double2 tr_3   = ldg_float2_f64(ptr_tm, idx_3);

        const double s1a = ds_diff_to_f_f64(c_now.x,  c_now.y,  c_1.x,  c_1.y);
        const double s1b = ds_diff_to_f_f64(tr_now.x, tr_now.y, tr_1.x, tr_1.y);
        const double s2a = ds_diff_to_f_f64(c_now.x,  c_now.y,  c_2.x,  c_2.y);
        const double s2b = ds_diff_to_f_f64(tr_now.x, tr_now.y, tr_2.x, tr_2.y);
        const double s3a = ds_diff_to_f_f64(c_now.x,  c_now.y,  c_3.x,  c_3.y);
        const double s3b = ds_diff_to_f_f64(tr_now.x, tr_now.y, tr_3.x, tr_3.y);

        const double t1 = (s1b != 0.0) ? (s1a * recip_nr1_f64(s1b)) : 0.0;
        const double t2 = (s2b != 0.0) ? (s2a * recip_nr1_f64(s2b)) : 0.0;
        const double t3 = (s3b != 0.0) ? (s3a * recip_nr1_f64(s3b)) : 0.0;

        out_row[s] = fma(UO_W1_F64, t1, fma(UO_W2_F64, t2, UO_W3_F64 * t3));
    }
}

}

/* ===========================================================================
 * NEOETHOS f64 LANE — ultosc                                      (Closer 5)
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/ultosc.rs
 *   :554 ultosc_scalar_impl   <- the per-bar body reproduced here
 *   :326 ultosc_prepare       first_valid AND start_idx
 *   :433 ultosc_with_kernel   NaN prefix is start_idx, not first_valid
 *
 * PERIOD-INVARIANT. cpu_batch.rs:5330 reads "timeperiod1" (7), "timeperiod2"
 * (14) and "timeperiod3" (28) and NEVER "period".
 *
 * FIRST-VALID IS ITS OWN RULE. ultosc.rs:391 scans for the first i >= 1 at
 * which high/low/close are non-NaN at BOTH i-1 AND i, because the true range
 * reads close[i-1]. That names a later bar than "first index where all three
 * are non-NaN", so it is registered as HlcConsecutivePairNonNan rather than
 * folded into AllInputsNonNan.
 *
 * TWO ROUNDING DETAILS, both deliberate:
 *   1. sum_a * sum_b.recip() is a RECIPROCAL THEN A MULTIPLY -- two roundings.
 *      Writing sum_a / sum_b would be more accurate and WRONG here.
 *   2. The blend is mul_add(w1, t1, mul_add(w2, t2, w3 * t3)) -- two fmas over
 *      one pre-rounded product, reproduced with fma in the same nesting.
 *
 * SEQUENTIAL, one thread per column: six running sums with ring eviction.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

#define NEO_ULTOSC_P1 7
#define NEO_ULTOSC_P2 14
#define NEO_ULTOSC_P3 28
#define NEO_ULTOSC_MAXP 28

extern "C" __global__
void ultosc_neo_batch_f64(const double* __restrict__ high,
                          const double* __restrict__ low,
                          const double* __restrict__ close,
                          int series_len,
                          const int* __restrict__ periods,
                          int n_combos,
                          int first_valid,
                          double* __restrict__ out)
{
    const int combo = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (combo >= n_combos) return;

    const int len = series_len;
    double* __restrict__ o = out + (size_t)combo * (size_t)len;
    (void)periods;  // PERIOD-INVARIANT

    const int p1 = NEO_ULTOSC_P1;
    const int p2 = NEO_ULTOSC_P2;
    const int p3 = NEO_ULTOSC_P3;
    const int max_p = NEO_ULTOSC_MAXP;

    // ultosc_prepare: first_valid >= 1 always (the scan starts at 1), and
    // start_idx >= len is NotEnoughValidData, i.e. no CPU column at all.
    if (len <= 0 || first_valid < 1 || first_valid >= len || max_p > len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }
    const int start_idx = first_valid + (max_p - 1);
    if (start_idx >= len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    for (int i = 0; i < start_idx; ++i) o[i] = NEO_F64_NAN;

    const double inv7_100 = 100.0 / 7.0;
    const double w1 = inv7_100 * 4.0;
    const double w2 = inv7_100 * 2.0;
    const double w3 = inv7_100 * 1.0;

    double cmtl_buf[NEO_ULTOSC_MAXP];
    double tr_buf[NEO_ULTOSC_MAXP];

    double sum1_a = 0.0, sum1_b = 0.0;
    double sum2_a = 0.0, sum2_b = 0.0;
    double sum3_a = 0.0, sum3_b = 0.0;

    int buf_idx = 0;
    int count = 0;

    for (int i = first_valid; i < len; ++i) {
        const double hi = high[i];
        const double lo = low[i];
        const double ci = close[i];
        const double prev_c = close[i - 1];

        const bool valid = !(isnan(hi) || isnan(lo) || isnan(ci) || isnan(prev_c));

        double c_new = 0.0, t_new = 0.0;
        if (valid) {
            // The CPU writes these as explicit comparisons, NOT fmin/fmax --
            // and with `valid` already excluding NaN both forms agree, so the
            // literal form is kept.
            const double tl = (lo < prev_c) ? lo : prev_c;
            const double th = (hi > prev_c) ? hi : prev_c;
            t_new = th - tl;
            c_new = ci - tl;
        }

        if (count >= p1) {
            int old_idx1 = buf_idx + max_p - p1;
            if (old_idx1 >= max_p) old_idx1 -= max_p;
            sum1_a -= cmtl_buf[old_idx1];
            sum1_b -= tr_buf[old_idx1];
        }
        if (count >= p2) {
            int old_idx2 = buf_idx + max_p - p2;
            if (old_idx2 >= max_p) old_idx2 -= max_p;
            sum2_a -= cmtl_buf[old_idx2];
            sum2_b -= tr_buf[old_idx2];
        }
        if (count >= p3) {
            int old_idx3 = buf_idx + max_p - p3;
            if (old_idx3 >= max_p) old_idx3 -= max_p;
            sum3_a -= cmtl_buf[old_idx3];
            sum3_b -= tr_buf[old_idx3];
        }

        cmtl_buf[buf_idx] = c_new;
        tr_buf[buf_idx] = t_new;

        if (valid) {
            sum1_a += c_new; sum1_b += t_new;
            sum2_a += c_new; sum2_b += t_new;
            sum3_a += c_new; sum3_b += t_new;
        }

        count += 1;
        if (i >= start_idx) {
            const double t1 = (sum1_b != 0.0) ? (sum1_a * (1.0 / sum1_b)) : 0.0;
            const double t2 = (sum2_b != 0.0) ? (sum2_a * (1.0 / sum2_b)) : 0.0;
            const double t3 = (sum3_b != 0.0) ? (sum3_a * (1.0 / sum3_b)) : 0.0;
            const double acc = fma(w2, t2, w3 * t3);
            o[i] = fma(w1, t1, acc);
        }

        buf_idx += 1;
        if (buf_idx == max_p) buf_idx = 0;
    }
}
