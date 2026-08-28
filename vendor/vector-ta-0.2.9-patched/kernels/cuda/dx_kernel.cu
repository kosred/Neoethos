#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__
void dx_build_terms_f64(const float* __restrict__ high,
                        const float* __restrict__ low,
                        const float* __restrict__ close,
                        int len,
                        double* __restrict__ plus_dm,
                        double* __restrict__ minus_dm,
                        double* __restrict__ tr,
                        unsigned char* __restrict__ carry) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (len <= 0) return;

    plus_dm[0] = 0.0;
    minus_dm[0] = 0.0;
    tr[0] = 0.0;
    carry[0] = 0u;

    for (int i = 1; i < len; ++i) {
        const double h = (double)high[i];
        const double l = (double)low[i];
        const double c = (double)close[i];
        plus_dm[i] = 0.0;
        minus_dm[i] = 0.0;
        tr[i] = 0.0;
        carry[i] = 0u;

        if (isnan(h) || isnan(l) || isnan(c)) {
            carry[i] = 1u;
            continue;
        }

        const double prev_h = (double)high[i - 1];
        const double prev_l = (double)low[i - 1];
        const double prev_c = (double)close[i - 1];
        const double up = h - prev_h;
        const double dn = prev_l - l;
        plus_dm[i] = (up > 0.0 && up > dn) ? up : 0.0;
        minus_dm[i] = (dn > 0.0 && dn > up) ? dn : 0.0;
        const double tr1 = h - l;
        const double tr2 = fabs(h - prev_c);
        const double tr3 = fabs(l - prev_c);
        tr[i] = fmax(fmax(tr1, tr2), tr3);
    }
}


extern "C" __global__
void dx_batch_f32(const double* __restrict__ plus_dm,
                  const double* __restrict__ minus_dm,
                  const double* __restrict__ tr,
                  const unsigned char* __restrict__ carry,
                  const int* __restrict__ periods,
                  int series_len,
                  int n_combos,
                  int first_valid,
                  float* __restrict__ out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_combos) return;

    float* dst = out + row * series_len;

    const int p = periods[row];
    if (p <= 0) return;
    if (first_valid < 0 || first_valid + 1 >= series_len) return;

    const int i0 = first_valid;
    const int warm_needed = p - 1;
    const int warm = first_valid + p - 1;
    const float nanv = nanf("");


    for (int i = 0; i < min(warm, series_len); ++i) {
        dst[i] = nanv;
    }

    double s_plus = 0.0;
    double s_minus = 0.0;
    double s_tr = 0.0;
    int init_count = 0;
    float last_out = nanv;
    const double rp = 1.0 / (double)p;

    for (int i = i0 + 1; i < series_len; ++i) {
        if (carry[i] != 0) {

            dst[i] = last_out;
            continue;
        }

        const double pdm = plus_dm[i];
        const double mdm = minus_dm[i];
        const double t   = tr[i];

        if (init_count < warm_needed) {
            s_plus  += pdm;
            s_minus += mdm;
            s_tr    += t;
            init_count += 1;
            if (init_count == warm_needed) {
                const double plus_di  = (s_tr != 0.0) ? ((s_plus  / s_tr) * 100.0) : 0.0;
                const double minus_di = (s_tr != 0.0) ? ((s_minus / s_tr) * 100.0) : 0.0;
                const double sum_di = plus_di + minus_di;
                const double dx = (sum_di != 0.0) ? (fabs(plus_di - minus_di) / sum_di) * 100.0 : 0.0;
                last_out = (float)dx;
                dst[i] = last_out;
            } else if (i >= warm) {

                dst[i] = nanv;
            }
            continue;
        }


        s_plus  = s_plus  - (s_plus  * rp) + pdm;
        s_minus = s_minus - (s_minus * rp) + mdm;
        s_tr    = s_tr    - (s_tr    * rp) + t;

        const double plus_di  = (s_tr != 0.0) ? ((s_plus  / s_tr) * 100.0) : 0.0;
        const double minus_di = (s_tr != 0.0) ? ((s_minus / s_tr) * 100.0) : 0.0;
        const double sum_di = plus_di + minus_di;
        if (sum_di != 0.0) {
            const double dx = (fabs(plus_di - minus_di) / sum_di) * 100.0;
            last_out = (float)dx;
            dst[i] = last_out;
        } else {
            dst[i] = last_out;
        }
    }
}


extern "C" __global__
void dx_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols) return;

    if (period <= 0) return;

    const int fv = first_valids[s];
    if (fv < 0 || fv + 1 >= rows) return;

    auto at = [&](int t) { return t * cols + s; };

    const int warm_needed = period - 1;
    const int warm = fv + period - 1;
    const float nanv = nanf("");

    for (int t = 0; t < min(warm, rows); ++t) {
        out_tm[at(t)] = nanv;
    }
    double s_plus = 0.0, s_minus = 0.0, s_tr = 0.0;
    int init_count = 0;
    float last_out = nanv;
    const double rp = 1.0 / (double)period;

    double prev_h = (double)high_tm[at(fv)];
    double prev_l = (double)low_tm[at(fv)];
    double prev_c = (double)close_tm[at(fv)];

    for (int t = fv + 1; t < rows; ++t) {
        const double ch = (double)high_tm[at(t)];
        const double cl = (double)low_tm[at(t)];
        const double cc = (double)close_tm[at(t)];
        if (isnan(ch) || isnan(cl) || isnan(cc)) {
            out_tm[at(t)] = last_out;
            prev_h = ch; prev_l = cl; prev_c = cc;
            continue;
        }

        if (isnan(prev_h) || isnan(prev_l) || isnan(prev_c)) {
            prev_h = ch; prev_l = cl; prev_c = cc;
            out_tm[at(t)] = nanv;
            continue;
        }
        const double up = ch - prev_h;
        const double dn = prev_l - cl;
        const double pdm = (up > 0.0 && up > dn) ? up : 0.0;
        const double mdm = (dn > 0.0 && dn > up) ? dn : 0.0;
        const double tr1 = ch - cl;
        const double tr2 = fabs(ch - prev_c);
        const double tr3 = fabs(cl - prev_c);
        const double tmax = fmax(fmax(tr1, tr2), tr3);

        if (init_count < warm_needed) {
            s_plus  += pdm;
            s_minus += mdm;
            s_tr    += tmax;
            init_count += 1;
            if (init_count == warm_needed) {
                const double plus_di  = (s_tr != 0.0) ? ((s_plus  / s_tr) * 100.0) : 0.0;
                const double minus_di = (s_tr != 0.0) ? ((s_minus / s_tr) * 100.0) : 0.0;
                const double sum_di = plus_di + minus_di;
                const double dx = (sum_di != 0.0) ? (fabs(plus_di - minus_di) / sum_di) * 100.0 : 0.0;
                last_out = (float)dx;
                out_tm[at(t)] = last_out;
            } else if (t >= warm) {
                out_tm[at(t)] = nanv;
            }
        } else {
            s_plus  = s_plus  - (s_plus  * rp) + pdm;
            s_minus = s_minus - (s_minus * rp) + mdm;
            s_tr    = s_tr    - (s_tr    * rp) + tmax;
            const double plus_di  = (s_tr != 0.0) ? ((s_plus  / s_tr) * 100.0) : 0.0;
            const double minus_di = (s_tr != 0.0) ? ((s_minus / s_tr) * 100.0) : 0.0;
            const double sum_di = plus_di + minus_di;
            if (sum_di != 0.0) {
                const double dx = (fabs(plus_di - minus_di) / sum_di) * 100.0;
                last_out = (float)dx;
                out_tm[at(t)] = last_out;
            } else {
                out_tm[at(t)] = last_out;
            }
        }

        prev_h = ch; prev_l = cl; prev_c = cc;
    }
}


struct dsf32 { float hi, lo; };
__device__ __forceinline__ void two_sum_f(float a, float b, float& s, float& err) {
    s = a + b; float bb = s - a; err = (a - (s - bb)) + (b - bb);
}
__device__ __forceinline__ void renorm_f(float& hi, float& lo) {
    float t = hi + lo; lo = lo - (t - hi); hi = t;
}
__device__ __forceinline__ void ds_add_inplace_f(dsf32& s, float x) {
    float sum, e; two_sum_f(s.hi, x, sum, e); s.lo += e; renorm_f(s.hi, s.lo);
}
__device__ __forceinline__ void ds_scale_add_inplace_f(dsf32& s, float a, float x) {
    float p = s.hi * a; float pe = fmaf(a, s.hi, -p); pe += s.lo * a; float sum, e2; two_sum_f(p, x, sum, e2); s.hi = sum; s.lo = pe + e2; renorm_f(s.hi, s.lo);
}

extern "C" __global__
void dx_batch_f32_fast(const double* __restrict__ plus_dm,
                       const double* __restrict__ minus_dm,
                       const double* __restrict__ ,
                       const unsigned char* __restrict__ carry,
                       const int* __restrict__ periods,
                       int series_len,
                       int n_combos,
                       int first_valid,
                       float* __restrict__ out)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_combos) return;

    float* dst = out + row * series_len;

    const int p = periods[row];
    if (p <= 0 || first_valid < 0 || first_valid + 1 >= series_len) return;

    const int warm = first_valid + p - 1;
    const float nanv = nanf("");
    for (int i = 0; i < min(warm, series_len); ++i) {
        dst[i] = nanv;
    }

    const float rp = 1.0f / (float)p;
    const float ap = 1.0f - rp;
    int warm_left = p - 1;
    dsf32 s_plus{0.f, 0.f}, s_minus{0.f, 0.f};
    float last_out = nanv;

    for (int i = first_valid + 1; i < series_len; ++i) {
        if (carry[i]) { dst[i] = last_out; continue; }
        const float pdm = (float)plus_dm[i]; const float mdm = (float)minus_dm[i];
        if (warm_left > 0) {
            ds_add_inplace_f(s_plus,  pdm); ds_add_inplace_f(s_minus, mdm); --warm_left;
            if (warm_left == 0) {
                const float sp = s_plus.hi + s_plus.lo; const float sm = s_minus.hi + s_minus.lo;
                const float denom = sp + sm; const float dx = (denom > 0.f) ? (fabsf(sp - sm) / denom) * 100.f : 0.f;
                last_out = dx; dst[i] = dx;
            } else if (i >= warm) {
                dst[i] = nanv;
            }
            continue;
        }
        ds_scale_add_inplace_f(s_plus, ap, pdm); ds_scale_add_inplace_f(s_minus, ap, mdm);
        const float sp = s_plus.hi + s_plus.lo; const float sm = s_minus.hi + s_minus.lo; const float denom = sp + sm;
        if (denom > 0.f) { const float dx = (fabsf(sp - sm) / denom) * 100.f; last_out = dx; dst[i] = dx; } else { dst[i] = last_out; }
    }
}

extern "C" __global__
void dx_many_series_one_param_time_major_f32_fast(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    int cols,
    int rows,
    int period,
    const int* __restrict__ first_valids,
    float* __restrict__ out_tm)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x; if (s >= cols || period <= 0) return;
    const int fv = first_valids[s]; if (fv < 0 || fv + 1 >= rows) return;
    const int warm = fv + period - 1;
    const float nanv = nanf("");
    for (int t = 0; t < min(warm, rows); ++t) {
        out_tm[t * cols + s] = nanv;
    }
    auto idx = [&](int t){ return t*cols + s; };
    int warm_left = period - 1; dsf32 s_plus{0.f,0.f}, s_minus{0.f,0.f}; const float rp = 1.0f/(float)period, ap = 1.0f - rp; float last_out = nanv;
    float ph = high_tm[idx(fv)], pl = low_tm[idx(fv)], pc = close_tm[idx(fv)];
    for (int t = fv + 1; t < rows; ++t) {
        const int k = idx(t); const float ch = high_tm[k], cl = low_tm[k], cc = close_tm[k];
        if (isnan(ch) || isnan(cl) || isnan(cc)) { out_tm[k] = last_out; ph = ch; pl = cl; pc = cc; continue; }
        if (isnan(ph) || isnan(pl) || isnan(pc)) { ph = ch; pl = cl; pc = cc; out_tm[k] = nanv; continue; }
        const float up = ch - ph, dn = pl - cl; const float pdm = (up > 0.f && up > dn) ? up : 0.f; const float mdm = (dn > 0.f && dn > up) ? dn : 0.f;
        if (warm_left > 0) { ds_add_inplace_f(s_plus, pdm); ds_add_inplace_f(s_minus, mdm); --warm_left; if (warm_left == 0) { const float sp = s_plus.hi + s_plus.lo; const float sm = s_minus.hi + s_minus.lo; const float denom = sp + sm; const float dx = (denom > 0.f) ? (fabsf(sp - sm)/denom)*100.f : 0.f; last_out = dx; out_tm[k] = dx; } else if (t >= warm) { out_tm[k] = nanv; } }
        else { ds_scale_add_inplace_f(s_plus, ap, pdm); ds_scale_add_inplace_f(s_minus, ap, mdm); const float sp = s_plus.hi + s_plus.lo; const float sm = s_minus.hi + s_minus.lo; const float denom = sp + sm; if (denom > 0.f) { const float dx = (fabsf(sp - sm)/denom)*100.f; last_out = dx; out_tm[k] = dx; } else { out_tm[k] = last_out; } }
        ph = ch; pl = cl; pc = cc;
    }
}


/* ===========================================================================
 * NEOETHOS f64 LANE — dx
 * ---------------------------------------------------------------------------
 * CPU oracle: src/indicators/dx.rs:286 `dx_scalar`, with `dx_initial_value`
 * (dx.rs:507) and `dx_rolling_value` (dx.rs:528).
 *
 * TWO SEED FORMULAS, SELECTED BY SERIES LENGTH. dx.rs:298 routes series of
 * 1,000,000 bars or more to `dx_scalar_original` (dx.rs:398), whose seed
 * computes +DI and -DI through tr_sum FIRST and then combines them, where
 * `dx_scalar` cancels tr_sum analytically. The two agree in exact arithmetic
 * and DIFFER in floating point. Ten years of M1 is ~3.7M bars per symbol, so
 * this branch is the LIVE one for our data, not a corner case — it is
 * reproduced rather than normalised away.
 *
 * NaN: `tr1.max(tr2).max(tr3)` is f64::max (returns the non-NaN operand) ->
 * fmax. The explicit is_nan() bar-skip is kept exactly: the CPU carries the
 * PREVIOUS output forward and still advances prev_high/low/close to the NaN
 * values, so the next real bar's up_move/down_move are NaN too.
 * =========================================================================== */

#ifndef NEO_F64_NAN
#define NEO_F64_NAN (__longlong_as_double(0x7ff8000000000000ULL))
#endif

__device__ __forceinline__
double dx_neo_initial_value_f64(double plus_dm_sum, double minus_dm_sum, double tr_sum)
{
    const double hundred = 100.0;
    if (tr_sum != 0.0 && isfinite(tr_sum)) {
        const double dm_sum = plus_dm_sum + minus_dm_sum;
        if (dm_sum != 0.0) return hundred * (fabs(plus_dm_sum - minus_dm_sum) / dm_sum);
        return 0.0;
    }
    const double plus_di  = (plus_dm_sum / tr_sum) * hundred;
    const double minus_di = (minus_dm_sum / tr_sum) * hundred;
    const double sum_di   = plus_di + minus_di;
    return (sum_di != 0.0) ? hundred * (fabs(plus_di - minus_di) / sum_di) : 0.0;
}

__device__ __forceinline__
double dx_neo_rolling_value_f64(double plus_dm_sum, double minus_dm_sum,
                                double tr_sum, double fallback)
{
    const double hundred = 100.0;
    if (tr_sum != 0.0) {
        if (isfinite(tr_sum)) {
            const double dm_sum = plus_dm_sum + minus_dm_sum;
            if (dm_sum != 0.0) return hundred * (fabs(plus_dm_sum - minus_dm_sum) / dm_sum);
            return fallback;
        }
        const double plus_di  = (plus_dm_sum / tr_sum) * hundred;
        const double minus_di = (minus_dm_sum / tr_sum) * hundred;
        const double sum_di   = plus_di + minus_di;
        if (sum_di != 0.0) return hundred * (fabs(plus_di - minus_di) / sum_di);
    }
    return fallback;
}

extern "C" __global__
void dx_neo_batch_f64(const double* __restrict__ high,
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
    const int period = periods[combo];

    if (period <= 0 || period > len || first_valid < 0 || first_valid >= len) {
        for (int i = 0; i < len; ++i) o[i] = NEO_F64_NAN;
        return;
    }

    // dx_with_kernel: warm = first + period - 1 (dx.rs:258)
    const int warm = first_valid + period - 1;
    for (int i = 0; i < warm && i < len; ++i) o[i] = NEO_F64_NAN;
    for (int i = warm; i < len; ++i) o[i] = NEO_F64_NAN;   // every bar is written below or stays NaN

    const bool long_series = (len >= 1000000);              // dx.rs:298
    const double p_f64 = (double)period;
    const double hundred = 100.0;

    double prev_high  = high[first_valid];
    double prev_low   = low[first_valid];
    double prev_close = close[first_valid];

    double plus_dm_sum = 0.0, minus_dm_sum = 0.0, tr_sum = 0.0;
    int initial_count = 0;

    for (int i = first_valid + 1; i < len; ++i) {
        const double h  = high[i];
        const double l  = low[i];
        const double cl = close[i];

        if (isnan(h) || isnan(l) || isnan(cl)) {
            o[i] = (i > 0) ? o[i - 1] : NEO_F64_NAN;
            prev_high = h; prev_low = l; prev_close = cl;
            continue;
        }

        const double up_move   = h - prev_high;
        const double down_move = prev_low - l;
        double plus_dm = 0.0, minus_dm = 0.0;
        if (up_move > 0.0 && up_move > down_move)        plus_dm  = up_move;
        else if (down_move > 0.0 && down_move > up_move) minus_dm = down_move;

        const double tr = fmax(fmax(h - l, fabs(h - prev_close)), fabs(l - prev_close));

        if (initial_count < (period - 1)) {
            plus_dm_sum  += plus_dm;
            minus_dm_sum += minus_dm;
            tr_sum       += tr;
            initial_count += 1;
            if (initial_count == (period - 1)) {
                if (long_series) {
                    const double plus_di  = (plus_dm_sum / tr_sum) * hundred;
                    const double minus_di = (minus_dm_sum / tr_sum) * hundred;
                    const double sum_di   = plus_di + minus_di;
                    o[i] = (sum_di != 0.0) ? hundred * (fabs(plus_di - minus_di) / sum_di) : 0.0;
                } else {
                    o[i] = dx_neo_initial_value_f64(plus_dm_sum, minus_dm_sum, tr_sum);
                }
            }
        } else {
            plus_dm_sum  = plus_dm_sum  - (plus_dm_sum  / p_f64) + plus_dm;
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / p_f64) + minus_dm;
            tr_sum       = tr_sum       - (tr_sum       / p_f64) + tr;
            if (long_series) {
                const double plus_di  = (plus_dm_sum / tr_sum) * hundred;
                const double minus_di = (minus_dm_sum / tr_sum) * hundred;
                const double sum_di   = plus_di + minus_di;
                o[i] = (sum_di != 0.0) ? hundred * (fabs(plus_di - minus_di) / sum_di)
                                       : ((i > 0) ? o[i - 1] : NEO_F64_NAN);
            } else {
                o[i] = dx_neo_rolling_value_f64(plus_dm_sum, minus_dm_sum, tr_sum,
                                                (i > 0) ? o[i - 1] : NEO_F64_NAN);
            }
        }

        prev_high = h; prev_low = l; prev_close = cl;
    }
}
