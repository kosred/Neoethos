#ifndef _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#define _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>

namespace {
__device__ inline bool is_finite(float x) { return !isnan(x) && !isinf(x); }


__device__ inline float ema_step(float x, float prev, float alpha, float beta) {
    return fmaf(beta, prev, alpha * x);
}


struct LaguerreRSI {
    float l0, l1, l2, l3;
    __device__ LaguerreRSI() : l0(0.f), l1(0.f), l2(0.f), l3(0.f) {}
    __device__ float update(float x) {
        const float alpha = 0.7f;
        const float one_m = 1.0f - alpha;
        float prev_l0 = l0;
        l0 = alpha * x + one_m * prev_l0;
        float prev_l1 = l1;
        l1 = -one_m * l0 + prev_l0 + one_m * prev_l1;
        float prev_l2 = l2;
        l2 = -one_m * l1 + prev_l1 + one_m * prev_l2;
        l3 = -one_m * l2 + prev_l2 + one_m * l3;
        float cu = fmaxf(l0 - l1, 0.f) + fmaxf(l1 - l2, 0.f) + fmaxf(l2 - l3, 0.f);
        float cd = fmaxf(l1 - l0, 0.f) + fmaxf(l2 - l1, 0.f) + fmaxf(l3 - l2, 0.f);
        float denom = cu + cd;
        if (denom != 0.f && isfinite(denom)) {
            return 100.f * (cu / denom);
        }
        return CUDART_NAN_F;
    }
};


__device__ inline float willr_close_only(const float* __restrict__ close,
                                         int idx, int win) {
    if (win <= 0 || idx + 1 < win) return CUDART_NAN_F;
    int start = idx + 1 - win;
    float hi = -CUDART_INF_F;
    float lo = CUDART_INF_F;
    for (int j = start; j <= idx; ++j) {
        float v = close[j];
        if (v > hi) hi = v;
        if (v < lo) lo = v;
    }
    float rng = hi - lo;
    if (rng == 0.f) return CUDART_NAN_F;
    return 60.f * (close[idx] - hi) / rng + 80.f;
}
}


extern "C" __global__ void mod_god_mode_batch_f32(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    int n_rows,
    const int* __restrict__ n1s,
    const int* __restrict__ n2s,
    const int* __restrict__ n3s,
    const int* __restrict__ modes,
    const int use_volume_flag,
    float* __restrict__ wavetrend_out,
    float* __restrict__ signal_out,
    float* __restrict__ histogram_out) {
    const int row0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;


    const int MAX_RING = 2048;

    for (int row = row0; row < n_rows; row += stride) {
        const int n1 = n1s[row];
        const int n2 = n2s[row];
        const int n3 = n3s[row];
        const int mode = modes[row];

        float* wt_row = wavetrend_out + (size_t)row * len;
        float* sig_row = signal_out + (size_t)row * len;
        float* hist_row = histogram_out + (size_t)row * len;

        for (int i = 0; i < len; ++i) {
            wt_row[i] = CUDART_NAN_F;
            sig_row[i] = CUDART_NAN_F;
            hist_row[i] = CUDART_NAN_F;
        }

        const float a1 = 2.f / (float(n1) + 1.f);
        const float b1 = 1.f - a1;
        const float a2 = 2.f / (float(n2) + 1.f);
        const float b2 = 1.f - a2;
        const float a3 = 2.f / (float(n3) + 1.f);
        const float b3 = 1.f - a3;

        int warm = first_valid + max(n1, max(n2, n3)) - 1;
        if (warm < 0) warm = 0;
        if (warm > len) warm = len;
        const int sig_start = warm + 6 - 1;


        float ema1_c = 0.f, ema2_abs = 0.f, ema3_ci = 0.f;
        bool seed_e1 = false, seed_e2 = false, seed_e3 = false;


        float rs_avg_gain = 0.f, rs_avg_loss = 0.f; bool rsi_seeded = false; int rs_init = 0;
        float prev_close = (first_valid < len) ? close[first_valid] : 0.f;


        LaguerreRSI lrsi;


        const int rsi_mod = (n2 > 0 && n2 < MAX_RING) ? n2 : MAX_RING;
        float rsi_ring[MAX_RING];
        for (int i = 0; i < rsi_mod; ++i) rsi_ring[i] = CUDART_NAN_F;
        int rsi_head = 0;
        float rsi_ema = 0.f; bool rsi_ema_seed = false;


        const int mf_mod = (n3 > 0 && n3 < MAX_RING) ? n3 : MAX_RING;
        float mf_ring_mf[MAX_RING];
        signed char mf_ring_sgn[MAX_RING];
        for (int i = 0; i < mf_mod; ++i) { mf_ring_mf[i] = 0.f; mf_ring_sgn[i] = 0; }
        float mf_pos_sum = 0.f, mf_neg_sum = 0.f; int mf_head = 0; bool tp_has_prev = false; float tp_prev = 0.f;


        float tsi_m_s = 0.f, tsi_m_l = 0.f, tsi_a_s = 0.f, tsi_a_l = 0.f; bool tsi_seed_s = false, tsi_seed_l = false;
        float csi_num_e1 = 0.f, csi_num_e2 = 0.f, csi_den_e1 = 0.f, csi_den_e2 = 0.f; bool csi_seed_e1 = false, csi_seed_e2 = false;

        for (int i = first_valid; i < len; ++i) {
            const float c = close[i];


            if (!seed_e1) { ema1_c = c; seed_e1 = true; }
            else { ema1_c = ema_step(c, ema1_c, a1, b1); }
            float abs_dev = fabsf(c - ema1_c);
            if (!seed_e2) { ema2_abs = abs_dev; seed_e2 = true; }
            else { ema2_abs = ema_step(abs_dev, ema2_abs, a1, b1); }
            float tci_val = CUDART_NAN_F;
            if (ema2_abs != 0.f && isfinite(ema2_abs)) {
                float ci = (c - ema1_c) / (0.025f * ema2_abs);
                if (!seed_e3) { ema3_ci = ci; seed_e3 = true; }
                else { ema3_ci = ema_step(ci, ema3_ci, a2, b2); }
                tci_val = ema3_ci + 50.f;
            }


            float rsi_val = CUDART_NAN_F;
            if (i == first_valid) {
                rs_avg_gain = 0.f; rs_avg_loss = 0.f; rs_init = 0; rsi_seeded = false;
            } else {
                float ch = c - prev_close;
                float gain = fmaxf(ch, 0.f);
                float loss = fmaxf(-ch, 0.f);
                if (!rsi_seeded) {
                    rs_init += 1;
                    rs_avg_gain += gain; rs_avg_loss += loss;
                    if (rs_init >= n3) {
                        rs_avg_gain /= (float)n3; rs_avg_loss /= (float)n3;
                        rsi_seeded = true;
                        float rs = (rs_avg_loss == 0.f) ? CUDART_INF_F : (rs_avg_gain / rs_avg_loss);
                        rsi_val = 100.f - 100.f / (1.f + rs);
                    }
                } else {
                    rs_avg_gain = ((rs_avg_gain * (float)(n3 - 1)) + gain) / (float)n3;
                    rs_avg_loss = ((rs_avg_loss * (float)(n3 - 1)) + loss) / (float)n3;
                    float rs = (rs_avg_loss == 0.f) ? CUDART_INF_F : (rs_avg_gain / rs_avg_loss);
                    rsi_val = 100.f - 100.f / (1.f + rs);
                }
            }


            float lrsi_val = lrsi.update(c);


            float mf_val = CUDART_NAN_F;
            if (use_volume_flag && volume != nullptr) {
                float tp = (high[i] + low[i] + c) * (1.f / 3.f);
                if (tp_has_prev) {
                    signed char sign = (tp > tp_prev) ? 1 : ((tp < tp_prev) ? -1 : 0);
                    float mf_raw = tp * volume[i];
                    if (rsi_seeded) {
                        int old_idx = mf_head % mf_mod;
                        float old_mf = mf_ring_mf[old_idx];
                        signed char old_sign = mf_ring_sgn[old_idx];
                        if (old_sign > 0) mf_pos_sum -= old_mf;
                        else if (old_sign < 0) mf_neg_sum -= old_mf;
                    }
                    int idx = mf_head % mf_mod;
                    mf_ring_mf[idx] = mf_raw;
                    mf_ring_sgn[idx] = sign;
                    if (sign > 0) mf_pos_sum += mf_raw; else if (sign < 0) mf_neg_sum += mf_raw;
                    mf_head = (mf_head + 1);
                    if (rsi_seeded) {
                        mf_val = (mf_neg_sum == 0.f) ? 100.f : (100.f - 100.f / (1.f + (mf_pos_sum / mf_neg_sum)));
                    }
                }
                tp_prev = tp; tp_has_prev = true;
            } else {
                mf_val = rsi_val;
            }


            float cbci_val = CUDART_NAN_F;
            if (rsi_seeded) {
                int oldi = rsi_head % rsi_mod;
                float old_rsi = rsi_ring[oldi];
                rsi_ring[oldi] = rsi_val;
                rsi_head = (rsi_head + 1);
                float mom = (is_finite(old_rsi) && is_finite(rsi_val)) ? (rsi_val - old_rsi) : CUDART_NAN_F;
                if (!rsi_ema_seed && is_finite(rsi_val)) { rsi_ema = rsi_val; rsi_ema_seed = true; }
                else if (is_finite(rsi_val)) { rsi_ema = ema_step(rsi_val, rsi_ema, a3, b3); }
                if (is_finite(mom) && rsi_ema_seed) cbci_val = mom + rsi_ema;
            }


            float csi_val = CUDART_NAN_F;
            float csi_mg_val = CUDART_NAN_F;

            if (i > first_valid) {
                float mom = c - prev_close; float am = fabsf(mom);
                if (!tsi_seed_s) { tsi_m_s = mom; tsi_a_s = am; tsi_seed_s = true; }
                else { tsi_m_s = ema_step(mom, tsi_m_s, a1, b1); tsi_a_s = ema_step(am, tsi_a_s, a1, b1); }
                if (!tsi_seed_l && tsi_seed_s) { tsi_m_l = tsi_m_s; tsi_a_l = tsi_a_s; tsi_seed_l = true; }
                else if (tsi_seed_l) { tsi_m_l = ema_step(tsi_m_s, tsi_m_l, a2, b2); tsi_a_l = ema_step(tsi_a_s, tsi_a_l, a2, b2); }
                if (tsi_seed_l && tsi_a_l != 0.f && isfinite(tsi_a_l) && is_finite(rsi_val)) {
                    float tsi = 100.f * (tsi_m_l / tsi_a_l);
                    csi_val = 0.5f * (rsi_val + (tsi * 0.5f + 50.f));
                }
            }


            if (i > first_valid) {
                float pc = c - prev_close; float apc = fabsf(pc);
                if (!csi_seed_e1) { csi_num_e1 = pc; csi_den_e1 = apc; csi_seed_e1 = true; }
                else { csi_num_e1 = ema_step(pc, csi_num_e1, a1, b1); csi_den_e1 = ema_step(apc, csi_den_e1, a1, b1); }
                if (!csi_seed_e2 && csi_seed_e1) { csi_num_e2 = csi_num_e1; csi_den_e2 = csi_den_e1; csi_seed_e2 = true; }
                else if (csi_seed_e2) { csi_num_e2 = ema_step(csi_num_e1, csi_num_e2, a2, b2); csi_den_e2 = ema_step(csi_den_e1, csi_den_e2, a2, b2); }
                if (csi_seed_e2 && csi_den_e2 != 0.f && isfinite(csi_den_e2) && is_finite(rsi_val)) {
                    float pc_norm = 50.f * (csi_num_e2 / csi_den_e2) + 50.f;
                    csi_mg_val = 0.5f * (rsi_val + pc_norm);
                }
            }


            if (i >= warm) {
                float sum = 0.f; int cnt = 0;
                if (mode == 0) {
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(csi_val)) { sum += csi_val; cnt++; }
                    if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                    float wil = willr_close_only(close, i, n2);
                    if (is_finite(wil)) { sum += wil; cnt++; }
                } else if (mode == 1) {
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                    if (is_finite(rsi_val)) { sum += rsi_val; cnt++; }
                } else if (mode == 2) {
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(csi_mg_val)) { sum += csi_mg_val; cnt++; }
                    if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                    float wil = willr_close_only(close, i, n2);
                    if (is_finite(wil)) { sum += wil; cnt++; }
                    if (is_finite(cbci_val)) { sum += cbci_val; cnt++; }
                    if (is_finite(lrsi_val)) { sum += lrsi_val; cnt++; }
                } else {
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                    if (is_finite(rsi_val)) { sum += rsi_val; cnt++; }
                    if (is_finite(cbci_val)) { sum += cbci_val; cnt++; }
                    if (is_finite(lrsi_val)) { sum += lrsi_val; cnt++; }
                }
                if (cnt > 0) {
                    float wt = sum / (float)cnt;
                    wt_row[i] = wt;

                if (i >= sig_start) {
                    float s = 0.f; int ready = 1;
                    for (int k = 0; k < 6; ++k) {
                        float x = wt_row[i - (6 - 1) + k];
                        if (!is_finite(x)) { ready = 0; break; }
                        s += x;
                    }
                    if (ready) {
                        float sig = s / 6.f;
                        sig_row[i] = sig;

                        float d = (wt - sig) * 2.f + 50.f;
                        if (!is_finite(hist_row[i - 1])) {
                            hist_row[i] = d;
                        } else {
                            hist_row[i] = ema_step(d, hist_row[i - 1], a3, b3);
                        }
                    }
                }
            }
            prev_close = c;
        }

    }
}

}


extern "C" __global__ void mod_god_mode_many_series_one_param_time_major_f32(
    const float* __restrict__ high_tm,
    const float* __restrict__ low_tm,
    const float* __restrict__ close_tm,
    const float* __restrict__ volume_tm,
    int cols,
    int rows,
    int n1, int n2, int n3, int mode, int use_volume_flag,
    float* __restrict__ wt_tm,
    float* __restrict__ sig_tm,
    float* __restrict__ hist_tm) {
    int s = blockIdx.x;
    if (s >= cols) return;
    if (threadIdx.x != 0) return;

    const int MAX_RING = 2048;

    auto idx = [cols](int t, int s) { return t * cols + s; };


    int first_valid = 0; bool found = false;
    for (int t = 0; t < rows; ++t) {
        float v = close_tm[idx(t, s)];
        if (is_finite(v)) { first_valid = t; found = true; break; }
    }
    if (!found) return;

    const float a1 = 2.f / (float(n1) + 1.f);
    const float b1 = 1.f - a1;
    const float a2 = 2.f / (float(n2) + 1.f);
    const float b2 = 1.f - a2;
    const float a3 = 2.f / (float(n3) + 1.f);
    const float b3 = 1.f - a3;

    int warm = first_valid + max(n1, max(n2, n3)) - 1;
    if (warm < 0) warm = 0;
    if (warm > rows) warm = rows;
    const int sig_start = warm + 6 - 1;


    for (int t = 0; t < rows; ++t) {
        wt_tm[idx(t, s)] = CUDART_NAN_F;
        sig_tm[idx(t, s)] = CUDART_NAN_F;
        hist_tm[idx(t, s)] = CUDART_NAN_F;
    }

    float ema1_c = 0.f, ema2_abs = 0.f, ema3_ci = 0.f; bool seed_e1=false, seed_e2=false, seed_e3=false;
    float rs_avg_gain=0.f, rs_avg_loss=0.f; bool rsi_seeded=false; int rs_init=0;
    float prev_close = close_tm[idx(first_valid, s)];
    LaguerreRSI lrsi;

    const int rsi_mod = (n2 > 0 && n2 < MAX_RING) ? n2 : MAX_RING;
    float rsi_ring[MAX_RING]; for (int i=0;i<rsi_mod;++i) rsi_ring[i]=CUDART_NAN_F;
    int rsi_head=0; float rsi_ema=0.f; bool rsi_ema_seed=false;
    float mf_ring_mf[MAX_RING]; signed char mf_ring_sgn[MAX_RING];
    const int mf_mod = (n3 > 0 && n3 < MAX_RING) ? n3 : MAX_RING;
    for (int i=0;i<mf_mod;++i){ mf_ring_mf[i]=0.f; mf_ring_sgn[i]=0; }
    float mf_pos_sum=0.f, mf_neg_sum=0.f; int mf_head=0; bool tp_has_prev=false; float tp_prev=0.f;


    float tsi_m_s=0.f, tsi_m_l=0.f, tsi_a_s=0.f, tsi_a_l=0.f; bool tsi_seed_s=false, tsi_seed_l=false;
    float csi_num_e1=0.f, csi_num_e2=0.f, csi_den_e1=0.f, csi_den_e2=0.f; bool csi_seed_e1=false, csi_seed_e2=false;

    for (int t = first_valid; t < rows; ++t) {
        float c = close_tm[idx(t, s)];

        if (!seed_e1) { ema1_c=c; seed_e1=true; } else { ema1_c = ema_step(c, ema1_c, a1, b1); }
        float abs_dev = fabsf(c - ema1_c);
        if (!seed_e2) { ema2_abs=abs_dev; seed_e2=true; } else { ema2_abs = ema_step(abs_dev, ema2_abs, a1, b1); }
        float tci_val = CUDART_NAN_F;
        if (ema2_abs != 0.f && isfinite(ema2_abs)) {
            float ci = (c - ema1_c) / (0.025f * ema2_abs);
            if (!seed_e3) { ema3_ci=ci; seed_e3=true; } else { ema3_ci = ema_step(ci, ema3_ci, a2, b2); }
            tci_val = ema3_ci + 50.f;
        }

        float rsi_val = CUDART_NAN_F;
        if (t == first_valid) { rs_avg_gain=0.f; rs_avg_loss=0.f; rs_init=0; }
        else {
            float ch = c - prev_close; float gain=fmaxf(ch,0.f), loss=fmaxf(-ch,0.f);
            if (!rsi_seeded) {
                rs_init++; rs_avg_gain+=gain; rs_avg_loss+=loss;
                if (rs_init >= n3) { rs_avg_gain/=(float)n3; rs_avg_loss/=(float)n3; rsi_seeded=true; float rs=(rs_avg_loss==0.f)?CUDART_INF_F:(rs_avg_gain/rs_avg_loss); rsi_val=100.f - 100.f/(1.f+rs); }
            } else {
                rs_avg_gain=((rs_avg_gain*(float)(n3-1))+gain)/(float)n3;
                rs_avg_loss=((rs_avg_loss*(float)(n3-1))+loss)/(float)n3;
                float rs=(rs_avg_loss==0.f)?CUDART_INF_F:(rs_avg_gain/rs_avg_loss); rsi_val=100.f - 100.f/(1.f+rs);
            }
        }
        float lrsi_val = lrsi.update(c);

        float mf_val = CUDART_NAN_F;
        if (use_volume_flag && volume_tm != nullptr) {
            float tp = (high_tm[idx(t,s)] + low_tm[idx(t,s)] + c) * (1.f/3.f);
            if (tp_has_prev) {
                signed char sign = (tp > tp_prev) ? 1 : ((tp < tp_prev) ? -1 : 0);
                float mf_raw = tp * volume_tm[idx(t,s)];
                if (rsi_seeded) {
                    int old = mf_head % mf_mod; float old_mf = mf_ring_mf[old]; signed char old_s = mf_ring_sgn[old];
                    if (old_s > 0) mf_pos_sum -= old_mf; else if (old_s < 0) mf_neg_sum -= old_mf;
                }
                int cur = mf_head % mf_mod; mf_ring_mf[cur]=mf_raw; mf_ring_sgn[cur]=sign; if (sign>0) mf_pos_sum += mf_raw; else if (sign<0) mf_neg_sum += mf_raw; mf_head++;
                if (rsi_seeded) mf_val = (mf_neg_sum == 0.f) ? 100.f : (100.f - 100.f/(1.f + (mf_pos_sum/mf_neg_sum)));
            }
            tp_prev = tp; tp_has_prev = true;
        } else { mf_val = rsi_val; }


        float cbci_val = CUDART_NAN_F;
        if (rsi_seeded) {
            int old = rsi_head % rsi_mod; float old_r = rsi_ring[old]; rsi_ring[old] = rsi_val; rsi_head++;
            float mom = (is_finite(old_r) && is_finite(rsi_val)) ? (rsi_val - old_r) : CUDART_NAN_F;
            if (!rsi_ema_seed && is_finite(rsi_val)) { rsi_ema = rsi_val; rsi_ema_seed = true; }
            else if (is_finite(rsi_val)) { rsi_ema = ema_step(rsi_val, rsi_ema, a3, b3); }
            if (is_finite(mom) && rsi_ema_seed) cbci_val = mom + rsi_ema;
        }

        float csi_val = CUDART_NAN_F, csi_mg_val = CUDART_NAN_F;
        if (t > first_valid) {
            float mom = c - prev_close; float am = fabsf(mom);

            if (!tsi_seed_s) { tsi_m_s=mom; tsi_a_s=am; tsi_seed_s=true; }
            else { tsi_m_s = ema_step(mom, tsi_m_s, a1, b1); tsi_a_s = ema_step(am, tsi_a_s, a1, b1); }
            if (!tsi_seed_l && tsi_seed_s) { tsi_m_l=tsi_m_s; tsi_a_l=tsi_a_s; tsi_seed_l=true; }
            else if (tsi_seed_l) { tsi_m_l = ema_step(tsi_m_s, tsi_m_l, a2, b2); tsi_a_l = ema_step(tsi_a_s, tsi_a_l, a2, b2); }
            if (tsi_seed_l && tsi_a_l != 0.f && isfinite(tsi_a_l) && is_finite(rsi_val)) {
                float tsi = 100.f * (tsi_m_l / tsi_a_l); csi_val = 0.5f * (rsi_val + (tsi * 0.5f + 50.f));
            }

            if (!csi_seed_e1) { csi_num_e1=mom; csi_den_e1=am; csi_seed_e1=true; }
            else { csi_num_e1 = ema_step(mom, csi_num_e1, a1, b1); csi_den_e1 = ema_step(am, csi_den_e1, a1, b1); }
            if (!csi_seed_e2 && csi_seed_e1) { csi_num_e2=csi_num_e1; csi_den_e2=csi_den_e1; csi_seed_e2=true; }
            else if (csi_seed_e2) { csi_num_e2 = ema_step(csi_num_e1, csi_num_e2, a2, b2); csi_den_e2 = ema_step(csi_den_e1, csi_den_e2, a2, b2); }
            if (csi_seed_e2 && csi_den_e2 != 0.f && isfinite(csi_den_e2) && is_finite(rsi_val)) {
                float pc_norm = 50.f * (csi_num_e2 / csi_den_e2) + 50.f; csi_mg_val = 0.5f * (rsi_val + pc_norm);
            }
        }

        float sum = 0.f; int cnt = 0;
        if (t >= warm) {
            if (mode == 0) {
                if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                if (is_finite(csi_val)) { sum += csi_val; cnt++; }
                if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                float hi = -CUDART_INF_F, lo = CUDART_INF_F; bool ok = (t + 1 >= n2);
                if (ok) {
                    int start = t + 1 - n2;
                    for (int j = start; j <= t; ++j) { float v = close_tm[idx(j, s)]; if (v > hi) hi = v; if (v < lo) lo = v; }
                    float rng = hi - lo; if (rng != 0.f) { float wil = 60.f * (c - hi) / rng + 80.f; if (is_finite(wil)) { sum += wil; cnt++; } }
                }
            }
            else if (mode == 1) { if (is_finite(tci_val)) { sum += tci_val; cnt++; } if (is_finite(mf_val)) { sum += mf_val; cnt++; } if (is_finite(rsi_val)) { sum += rsi_val; cnt++; } }
            else if (mode == 2) {
                if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                if (is_finite(csi_mg_val)) { sum += csi_mg_val; cnt++; }
                if (is_finite(mf_val)) { sum += mf_val; cnt++; }
                float hi = -CUDART_INF_F, lo = CUDART_INF_F; bool ok = (t + 1 >= n2);
                if (ok) {
                    int start = t + 1 - n2; for (int j = start; j <= t; ++j) { float v = close_tm[idx(j, s)]; if (v > hi) hi = v; if (v < lo) lo = v; }
                    float rng = hi - lo; if (rng != 0.f) { float wil = 60.f * (c - hi) / rng + 80.f; if (is_finite(wil)) { sum += wil; cnt++; } }
                }
                if (is_finite(cbci_val)) { sum += cbci_val; cnt++; }
                if (is_finite(lrsi_val)) { sum += lrsi_val; cnt++; }
            }
            else { if (is_finite(tci_val)) { sum += tci_val; cnt++; } if (is_finite(mf_val)) { sum += mf_val; cnt++; } if (is_finite(rsi_val)) { sum += rsi_val; cnt++; } if (is_finite(cbci_val)) { sum += cbci_val; cnt++; } if (is_finite(lrsi_val)) { sum += lrsi_val; cnt++; } }
            if (cnt > 0) {
                float wt = sum / (float)cnt; wt_tm[idx(t, s)] = wt;
                if (t >= sig_start) {
                float ssum = 0.f; int ready = 1;
                for (int k=0;k<6;++k){ float x = wt_tm[idx(t - (6 - 1) + k, s)]; if (!is_finite(x)){ ready=0; break; } ssum += x; }
                if (ready) { float sig = ssum / 6.f; sig_tm[idx(t, s)] = sig; float d = (wt - sig)*2.f + 50.f; if (!is_finite(hist_tm[idx(t-1,s)])) hist_tm[idx(t,s)] = d; else hist_tm[idx(t,s)] = ema_step(d, hist_tm[idx(t-1,s)], a3, b3); }
            }
    }
    prev_close = c;
  }
}

}


#ifndef MGM_RING_KCAP


#define MGM_RING_KCAP 64
#endif

namespace {

__device__ __forceinline__ int pow2_cap() { return MGM_RING_KCAP; }
__device__ __forceinline__ int pow2_mask() { return MGM_RING_KCAP - 1; }


struct Kahan {
    float s, c;
    __device__ Kahan(): s(0.f), c(0.f) {}
    __device__ inline void add(float x){
        float y = x - c;
        float t = s + y;
        c = (t - s) - y;
        s = t;
    }
    __device__ inline void sub(float x){ add(-x); }
};


struct MonoDeque {
    int *buf; int head, tail, mask;
    __device__ MonoDeque(): buf(nullptr), head(0), tail(0), mask(0) {}
    __device__ inline void init(int* storage, int cap_mask) {
        buf = storage; head = tail = 0; mask = cap_mask;
    }
    __device__ inline bool empty() const { return head == tail; }
    __device__ inline int size()  const { return tail - head; }
    __device__ inline int& slot(int i)   { return buf[i & mask]; }
    __device__ inline int  front() const { return buf[head & mask]; }
    __device__ inline void pop_front()   { head++; }
    __device__ inline void push_back_max(const float* __restrict__ close, int idx) {
        while (head != tail) {
            int last = buf[(tail - 1) & mask];
            if (close[last] >= close[idx]) break;
            tail--;
        }
        slot(tail) = idx; tail++;
    }
    __device__ inline void push_back_min(const float* __restrict__ close, int idx) {
        while (head != tail) {
            int last = buf[(tail - 1) & mask];
            if (close[last] <= close[idx]) break;
            tail--;
        }
        slot(tail) = idx; tail++;
    }
    __device__ inline void expire(int oldest_idx_allowed) {
        while (head != tail && front() < oldest_idx_allowed) head++;
    }
};

}

extern "C" __global__ void mod_god_mode_batch_f32_shared_fast(
    const float* __restrict__ high,
    const float* __restrict__ low,
    const float* __restrict__ close,
    const float* __restrict__ volume,
    int len,
    int first_valid,
    int n_rows,
    const int* __restrict__ n1s,
    const int* __restrict__ n2s,
    const int* __restrict__ n3s,
    const int* __restrict__ modes,
    const int use_volume_flag,
    float* __restrict__ wavetrend_out,
    float* __restrict__ signal_out,
    float* __restrict__ histogram_out)
{

    const int tid    = threadIdx.x;
    const int row0   = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;


    extern __shared__ unsigned char smem_raw[];
    const int  cap   = pow2_cap();
    const int  mask  = pow2_mask();


    unsigned char* p = smem_raw;

    float* rsi_base = reinterpret_cast<float*>(p);
    p += sizeof(float) * cap * blockDim.x;

    float* mfi_mf_base = reinterpret_cast<float*>(p);
    p += sizeof(float) * cap * blockDim.x;


    size_t off = reinterpret_cast<size_t>(p);
    off = (off + 3u) & ~size_t(3u);
    p = reinterpret_cast<unsigned char*>(off);

    signed char* mfi_sgn_base = reinterpret_cast<signed char*>(p);
    p += sizeof(signed char) * cap * blockDim.x;

    off = reinterpret_cast<size_t>(p);
    off = (off + 3u) & ~size_t(3u);
    p = reinterpret_cast<unsigned char*>(off);

    int* dq_max_base = reinterpret_cast<int*>(p);
    p += sizeof(int) * cap * blockDim.x;

    int* dq_min_base = reinterpret_cast<int*>(p);


    float*       rsi_ring  = rsi_base     + tid * cap;
    float*       mfi_mf    = mfi_mf_base  + tid * cap;
    signed char* mfi_sgn   = mfi_sgn_base + tid * cap;
    int*         dq_max    = dq_max_base  + tid * cap;
    int*         dq_min    = dq_min_base  + tid * cap;


    for (int k = 0; k < cap; ++k) {
        rsi_ring[k] = CUDART_NAN_F;
        mfi_mf[k]   = 0.f;
        mfi_sgn[k]  = 0;
        dq_max[k]   = 0;
        dq_min[k]   = 0;
    }
    __syncthreads();

    for (int row = row0; row < n_rows; row += stride) {
        const int n1   = n1s[row];
        const int n2   = n2s[row];
        const int n3   = n3s[row];
        const int mode = modes[row];


        if (n2 > cap || n3 > cap) {

            continue;
        }

        float* wt_row   = wavetrend_out  + (size_t)row * len;
        float* sig_row  = signal_out     + (size_t)row * len;
        float* hist_row = histogram_out  + (size_t)row * len;


        for (int i = 0; i < len; ++i) {
            wt_row[i]   = CUDART_NAN_F;
            sig_row[i]  = CUDART_NAN_F;
            hist_row[i] = CUDART_NAN_F;
        }

        const float a1 = 2.f / (float(n1) + 1.f);
        const float b1 = 1.f - a1;
        const float a2 = 2.f / (float(n2) + 1.f);
        const float b2 = 1.f - a2;
        const float a3 = 2.f / (float(n3) + 1.f);
        const float b3 = 1.f - a3;
        const float inv_n3 = 1.f / float(n3);
        const float n3m1   = float(n3 - 1);

        int warm = first_valid + max(n1, max(n2, n3)) - 1;
        if (warm < 0) warm = 0;
        if (warm > len) warm = len;
        const int sig_start = warm + 6 - 1;


        float ema1_c=0.f, ema2_abs=0.f, ema3_ci=0.f; bool seed_e1=false, seed_e2=false, seed_e3=false;

        float rs_avg_gain=0.f, rs_avg_loss=0.f; bool rsi_seeded=false; int rs_init=0;
        float prev_close = (first_valid < len) ? close[first_valid] : 0.f;

        LaguerreRSI lrsi;


        int   rsi_head = 0, rsi_count = 0;
        float rsi_ema  = 0.f; bool rsi_ema_seed = false;


        Kahan mf_pos_sum, mf_neg_sum;
        int   mf_head = 0, mf_count = 0; bool tp_has_prev=false; float tp_prev=0.f;


        float tsi_m_s=0.f, tsi_m_l=0.f, tsi_a_s=0.f, tsi_a_l=0.f; bool tsi_seed_s=false, tsi_seed_l=false;
        float csi_num_e1=0.f, csi_num_e2=0.f, csi_den_e1=0.f, csi_den_e2=0.f; bool csi_seed_e1=false, csi_seed_e2=false;


        MonoDeque dqHi, dqLo;
        dqHi.init(dq_max, mask);
        dqLo.init(dq_min, mask);


        float sig_sum6 = 0.f; bool sig_seeded=false;


        for (int i = first_valid; i < len; ++i) {
            const float c = close[i];


            if (!seed_e1) { ema1_c = c; seed_e1 = true; }
            else          { ema1_c = fmaf(b1, ema1_c, a1 * c); }
            float abs_dev = fabsf(c - ema1_c);
            if (!seed_e2) { ema2_abs = abs_dev; seed_e2 = true; }
            else          { ema2_abs = fmaf(b1, ema2_abs, a1 * abs_dev); }
            float tci_val = CUDART_NAN_F;
            if (ema2_abs != 0.f && isfinite(ema2_abs)) {
                float ci = (c - ema1_c) / (0.025f * ema2_abs);
                if (!seed_e3) { ema3_ci = ci; seed_e3 = true; }
                else          { ema3_ci = fmaf(b2, ema3_ci, a2 * ci); }
                tci_val = ema3_ci + 50.f;
            }


            float rsi_val = CUDART_NAN_F;
            if (i == first_valid) {
                rs_avg_gain=0.f; rs_avg_loss=0.f; rs_init=0; rsi_seeded=false;
            } else {
                float ch   = c - prev_close;
                float gain = fmaxf(ch, 0.f);
                float loss = fmaxf(-ch, 0.f);
                if (!rsi_seeded) {
                    rs_init += 1;
                    rs_avg_gain += gain; rs_avg_loss += loss;
                    if (rs_init >= n3) {
                        rs_avg_gain *= inv_n3; rs_avg_loss *= inv_n3;
                        rsi_seeded = true;
                        float rs = (rs_avg_loss == 0.f) ? CUDART_INF_F : (rs_avg_gain / rs_avg_loss);
                        rsi_val = 100.f - 100.f / (1.f + rs);
                    }
                } else {
                    rs_avg_gain = (rs_avg_gain * n3m1 + gain) * inv_n3;
                    rs_avg_loss = (rs_avg_loss * n3m1 + loss) * inv_n3;
                    float rs = (rs_avg_loss == 0.f) ? CUDART_INF_F : (rs_avg_gain / rs_avg_loss);
                    rsi_val = 100.f - 100.f / (1.f + rs);
                }
            }


            float lrsi_val = lrsi.update(c);


            float mf_val = CUDART_NAN_F;
            if (use_volume_flag && volume != nullptr) {
                float tp = (high[i] + low[i] + c) * (1.f/3.f);
                if (tp_has_prev) {
                    signed char sign = (tp > tp_prev) ? 1 : ((tp < tp_prev) ? -1 : 0);
                    float mf_raw = tp * volume[i];


                    if (rsi_seeded && mf_count >= n3) {
                        int ev = mf_head & mask;
                        float old_mf = mfi_mf[ev];
                        signed char old_s = mfi_sgn[ev];
                        if (old_s > 0) mf_pos_sum.sub(old_mf);
                        else if (old_s < 0) mf_neg_sum.sub(old_mf);
                    }

                    int idx = mf_head & mask;
                    mfi_mf[idx]  = mf_raw;
                    mfi_sgn[idx] = sign;
                    if (sign > 0)      mf_pos_sum.add(mf_raw);
                    else if (sign < 0) mf_neg_sum.add(mf_raw);
                    mf_head++; mf_count++;

                    if (rsi_seeded && mf_count >= n3) {
                        float pos = mf_pos_sum.s;
                        float neg = mf_neg_sum.s;
                        mf_val = (neg == 0.f) ? 100.f : (100.f - 100.f / (1.f + (pos / neg)));
                    }
                }
                tp_prev = tp; tp_has_prev = true;
            } else {
                mf_val = rsi_val;
            }


            float cbci_val = CUDART_NAN_F;
            if (rsi_seeded) {

                rsi_ring[rsi_head & mask] = rsi_val;
                rsi_head++; rsi_count++;

                float mom = CUDART_NAN_F;
                if (rsi_count > n2) {
                    float old_rsi = rsi_ring[(rsi_head - 1 - n2) & mask];
                    if (is_finite(old_rsi) && is_finite(rsi_val)) mom = rsi_val - old_rsi;
                }

                if (!rsi_ema_seed && is_finite(rsi_val)) { rsi_ema = rsi_val; rsi_ema_seed = true; }
                else if (is_finite(rsi_val))             { rsi_ema = fmaf(b3, rsi_ema, a3 * rsi_val); }

                if (is_finite(mom) && rsi_ema_seed) cbci_val = mom + rsi_ema;
            }


            float csi_val = CUDART_NAN_F;
            if (i > first_valid) {
                float mom = c - prev_close; float am = fabsf(mom);
                if (!tsi_seed_s) { tsi_m_s=mom; tsi_a_s=am; tsi_seed_s=true; }
                else             { tsi_m_s = fmaf(b1, tsi_m_s, a1 * mom);
                                   tsi_a_s = fmaf(b1, tsi_a_s, a1 * am); }
                if (!tsi_seed_l) { tsi_m_l = tsi_m_s; tsi_a_l = tsi_a_s; tsi_seed_l = true; }
                else             { tsi_m_l = fmaf(b2, tsi_m_l, a2 * tsi_m_s);
                                   tsi_a_l = fmaf(b2, tsi_a_l, a2 * tsi_a_s); }
                if (tsi_seed_l && tsi_a_l != 0.f && isfinite(tsi_a_l) && is_finite(rsi_val)) {
                    float tsi = 100.f * (tsi_m_l / tsi_a_l);
                    csi_val = 0.5f * (rsi_val + (tsi * 0.5f + 50.f));
                }
            }


            float csi_mg_val = CUDART_NAN_F;
            if (i > first_valid) {
                float pc  = c - prev_close; float apc = fabsf(pc);
                if (!csi_seed_e1) { csi_num_e1=pc; csi_den_e1=apc; csi_seed_e1=true; }
                else              { csi_num_e1 = fmaf(b1, csi_num_e1, a1 * pc);
                                    csi_den_e1 = fmaf(b1, csi_den_e1, a1 * apc); }
                if (!csi_seed_e2) { csi_num_e2=csi_num_e1; csi_den_e2=csi_den_e1; csi_seed_e2=true; }
                else              { csi_num_e2 = fmaf(b2, csi_num_e2, a2 * csi_num_e1);
                                    csi_den_e2 = fmaf(b2, csi_den_e2, a2 * csi_den_e1); }
                if (csi_seed_e2 && csi_den_e2 != 0.f && isfinite(csi_den_e2) && is_finite(rsi_val)) {
                    float pc_norm = 50.f * (csi_num_e2 / csi_den_e2) + 50.f;
                    csi_mg_val = 0.5f * (rsi_val + pc_norm);
                }
            }


            float wil = CUDART_NAN_F;

            dqHi.push_back_max(close, i);
            dqLo.push_back_min(close, i);

            int oldest = i - n2 + 1;
            if (oldest < 0) oldest = 0;
            dqHi.expire(oldest);
            dqLo.expire(oldest);
            if (i + 1 >= n2) {
                float hi = close[dqHi.front()];
                float lo = close[dqLo.front()];
                float rng = hi - lo;
                if (rng != 0.f) wil = 60.f * (c - hi) / rng + 80.f;
            }


            if (i >= warm) {
                float sum = 0.f; int cnt = 0;
                if (mode == 0) {
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(csi_val)) { sum += csi_val; cnt++; }
                    if (is_finite(mf_val))  { sum += mf_val;  cnt++; }
                    if (is_finite(wil))     { sum += wil;     cnt++; }
                } else if (mode == 1) {
                    if (is_finite(tci_val)) { sum += tci_val; cnt++; }
                    if (is_finite(mf_val))  { sum += mf_val;  cnt++; }
                    if (is_finite(rsi_val)) { sum += rsi_val; cnt++; }
                } else if (mode == 2) {
                    if (is_finite(tci_val))     { sum += tci_val;     cnt++; }
                    if (is_finite(csi_mg_val))  { sum += csi_mg_val;  cnt++; }
                    if (is_finite(mf_val))      { sum += mf_val;      cnt++; }
                    if (is_finite(wil))         { sum += wil;         cnt++; }
                    if (is_finite(cbci_val))    { sum += cbci_val;    cnt++; }
                    if (is_finite(lrsi_val))    { sum += lrsi_val;    cnt++; }
                } else {
                    if (is_finite(tci_val))  { sum += tci_val;  cnt++; }
                    if (is_finite(mf_val))   { sum += mf_val;   cnt++; }
                    if (is_finite(rsi_val))  { sum += rsi_val;  cnt++; }
                    if (is_finite(cbci_val)) { sum += cbci_val; cnt++; }
                    if (is_finite(lrsi_val)) { sum += lrsi_val; cnt++; }
                }
                if (cnt > 0) {
                    float wt = sum / (float)cnt;
                    wt_row[i] = wt;


                if (i >= sig_start) {
                    if (!sig_seeded) {
                        float s = 0.f; bool ok = true;
                        for (int k = 0; k < 6; ++k) {
                            float x = wt_row[i - (6 - 1) + k];
                            if (!is_finite(x)) { ok = false; break; }
                            s += x;
                        }
                        if (ok) { sig_sum6 = s; sig_seeded = true; }
                    } else {
                        float old_w = wt_row[i - 6];
                        if (is_finite(old_w)) sig_sum6 += wt - old_w;
                        else {

                            float s = 0.f; bool ok = true;
                            for (int k = 0; k < 6; ++k) {
                                float x = wt_row[i - (6 - 1) + k];
                                if (!is_finite(x)) { ok = false; break; }
                                s += x;
                            }
                            if (ok) sig_sum6 = s; else sig_seeded = false;
                        }
                    }
                    if (sig_seeded) {
                        float sig = sig_sum6 / 6.f;
                        sig_row[i] = sig;
                        float d = (wt - sig) * 2.f + 50.f;
                        if (!is_finite(hist_row[i - 1])) hist_row[i] = d;
                        else                             hist_row[i] = fmaf(b3, hist_row[i - 1], a3 * d);
                    }
                }
            }
            prev_close = c;
        }
    }
}

}


// ===========================================================================
// NEOETHOS f64 LANE  --  closer 4, round 3
//
// CPU reference: mod_god_mode_scalar_fused_into_slices
// (src/indicators/mod_god_mode.rs:1118-1600). mod_god_mode_with_kernel (:555)
// maps Kernel::Auto to Kernel::Scalar (:611) and mod_god_mode_into_slices
// (:717) then takes the fused scalar, so it is the only path this lane can
// reach; the component-wise arm at :745 is unreachable from it.
//
// OUTPUT: the WAVETREND column -- compute_mod_god_mode_batch
// (cpu_batch.rs:15556) resolves output_id == "value" to out.wavetrend.
//
// PERIOD-INVARIANT: that batch reads n1 (17), n2 (6), n3 (4), mode
// ("tradition_mg") and use_volume (TRUE -- cpu_batch.rs:15521, not the input
// struct default of false) and NEVER `period`. A sweep of five periods gets
// five identical CPU columns, so this kernel writes five identical rows.
//
// MODE: TraditionMg alone is reproduced, because that is the only mode the
// batch default selects. Its component set is tci + mf + rsi + cbci + lrsi
// (:1541-1560); the Godmode arms compute csi and willr, which that set never
// reads.
//
// FIRST-VALID: Ignored, and derived here. mod_god_mode_into_slices (:693)
// scans CLOSE ALONE with !is_nan while this row is registered Hlcv, and under
// AllInputsNonNan that shape resolves to a scan over high/low/close/volume --
// a later bar on any frame whose volume or high has a hole, which would shift
// both the NaN prefix and the seed.
//
// SHAPE: one thread per combo walking bars ASCENDING. Three EMAs, a Wilder
// RSI, a four-stage Laguerre filter, a money-flow ring, an RSI ring feeding a
// momentum term, a six-wide signal SMA and an EMA over the histogram are all
// carried across bars.
//
// NaN SEMANTICS: the Laguerre cu/cd terms use f64::max against 0.0
// (:1332-1333), which returns the non-NaN operand; fmax is used so a NaN
// close cannot survive into the accumulation as it would through an if-chain.
// ===========================================================================

#define MGM_NEO_N1 17
#define MGM_NEO_N2 6
#define MGM_NEO_N3 4
#define MGM_NEO_SIGP 6
#define MGM_NEO_MAX_RING 64

static __forceinline__ __device__ double mgm_neo_qnan() {
    return __longlong_as_double(0x7ff8000000000000ULL);
}

static __forceinline__ __device__ double mgm_neo_inf() {
    return __longlong_as_double(0x7ff0000000000000ULL);
}

static __forceinline__ __device__ double mgm_neo_ema_step(double x, double prev,
                                                          double alpha, double beta) {
    return fma(beta, prev, alpha * x);   // :1157
}

static __forceinline__ __device__ bool mgm_neo_nonzero(double v) {
    return v != 0.0 && isfinite(v);      // :1161
}

extern "C" __global__
void mod_god_mode_neo_batch_f64(const double* __restrict__ high,
                                const double* __restrict__ low,
                                const double* __restrict__ close,
                                const double* __restrict__ volume,
                                int n,
                                const int* __restrict__ periods,
                                int n_combos,
                                int first_valid,
                                double* __restrict__ out) {
    const int combo = blockIdx.x * blockDim.x + threadIdx.x;
    if (combo >= n_combos) return;
    if (n <= 0) return;
    (void)periods;      // PERIOD-INVARIANT -- see the header.
    (void)first_valid;  // Ignored -- see the header.

    double* __restrict__ row = out + (size_t)combo * (size_t)n;
    const double nn = mgm_neo_qnan();

    const int n1 = MGM_NEO_N1, n2 = MGM_NEO_N2, n3 = MGM_NEO_N3;

    // :693-696 -- close alone, !is_nan.
    int first = -1;
    for (int i = 0; i < n; ++i) {
        if (!isnan(close[i])) { first = i; break; }
    }
    int need = n1;
    if (n2 > need) need = n2;
    if (n3 > need) need = n3;

    if (first < 0 || (n - first) < need) {
        for (int i = 0; i < n; ++i) row[i] = nn;
        return;
    }

    const int warm = first + need - 1;
    for (int i = 0; i < n; ++i) row[i] = nn;

    // :1265 -- if the first close is not finite the CPU returns with nothing
    // written, so the whole column is the NaN it was allocated as.
    if (!isfinite(close[first])) return;

    const double alpha1 = 2.0 / ((double)n1 + 1.0);
    const double beta1 = 1.0 - alpha1;
    const double alpha2 = 2.0 / ((double)n2 + 1.0);
    const double beta2 = 1.0 - alpha2;
    const double alpha3 = 2.0 / ((double)n3 + 1.0);
    const double beta3 = 1.0 - alpha3;

    double ema1_c = 0.0, ema2_abs = 0.0, ema3_ci = 0.0;
    bool seed_ema1 = false, seed_ema2 = false, seed_ema3 = false;

    double rs_avg_gain = 0.0, rs_avg_loss = 0.0;
    bool rsi_seeded = false;
    int rs_init_cnt = 0;
    double prev_close = close[first];

    const int rsi_len = n2 > 1 ? n2 : 1;
    double rsi_ring[MGM_NEO_MAX_RING];
    for (int k = 0; k < rsi_len; ++k) rsi_ring[k] = nn;
    int rsi_ring_head = 0;
    double rsi_ema = 0.0;
    bool rsi_ema_seed = false;

    const double alpha_l = 0.7;
    const double one_m_l = 1.0 - alpha_l;
    double l0 = 0.0, l1 = 0.0, l2 = 0.0, l3 = 0.0;

    const int mf_len = n3 > 1 ? n3 : 1;
    double mf_ring_mf[MGM_NEO_MAX_RING];
    int mf_ring_sgn[MGM_NEO_MAX_RING];
    for (int k = 0; k < mf_len; ++k) { mf_ring_mf[k] = 0.0; mf_ring_sgn[k] = 0; }
    double mf_pos_sum = 0.0, mf_neg_sum = 0.0;
    int mf_head = 0;
    double tp_prev = 0.0;
    bool tp_has_prev = false;

    double sig_ring[MGM_NEO_SIGP];
    for (int k = 0; k < MGM_NEO_SIGP; ++k) sig_ring[k] = 0.0;
    int sig_head = 0;
    double sig_sum = 0.0;
    const int sig_start = warm + MGM_NEO_SIGP - 1;
    bool have_sig = false;

    for (int i = first; i < n; ++i) {
        const double c_i = close[i];

        // :1272-1289 -- tci
        if (!seed_ema1) { ema1_c = c_i; seed_ema1 = true; }
        else { ema1_c = mgm_neo_ema_step(c_i, ema1_c, alpha1, beta1); }
        const double abs_dev = fabs(c_i - ema1_c);
        if (!seed_ema2) { ema2_abs = abs_dev; seed_ema2 = true; }
        else { ema2_abs = mgm_neo_ema_step(abs_dev, ema2_abs, alpha1, beta1); }
        double tci_val = nn;
        if (mgm_neo_nonzero(ema2_abs)) {
            const double ci = (c_i - ema1_c) / (0.025 * ema2_abs);
            if (!seed_ema3) { ema3_ci = ci; seed_ema3 = true; }
            else { ema3_ci = mgm_neo_ema_step(ci, ema3_ci, alpha2, beta2); }
            tci_val = ema3_ci + 50.0;
        }

        // :1291-1325 -- the Wilder RSI over n3.
        double rsi_val = nn;
        if (i == first) {
            rs_avg_gain = 0.0;
            rs_avg_loss = 0.0;
            rs_init_cnt = 0;
        } else {
            const double ch = c_i - prev_close;
            const double gain = (ch > 0.0) ? ch : 0.0;
            const double loss = (ch < 0.0) ? -ch : 0.0;
            if (!rsi_seeded) {
                rs_init_cnt += 1;
                rs_avg_gain += gain;
                rs_avg_loss += loss;
                if (rs_init_cnt >= n3) {
                    rs_avg_gain /= (double)n3;
                    rs_avg_loss /= (double)n3;
                    rsi_seeded = true;
                    const double rs = (rs_avg_loss == 0.0)
                        ? mgm_neo_inf() : (rs_avg_gain / rs_avg_loss);
                    rsi_val = 100.0 - 100.0 / (1.0 + rs);
                }
            } else {
                rs_avg_gain = ((rs_avg_gain * (double)(n3 - 1)) + gain) / (double)n3;
                rs_avg_loss = ((rs_avg_loss * (double)(n3 - 1)) + loss) / (double)n3;
                const double rs = (rs_avg_loss == 0.0)
                    ? mgm_neo_inf() : (rs_avg_gain / rs_avg_loss);
                rsi_val = 100.0 - 100.0 / (1.0 + rs);
            }
        }

        // :1327-1340 -- the four-stage Laguerre filter.
        {
            const double prev_l0 = l0;
            l0 = alpha_l * c_i + one_m_l * prev_l0;
            const double prev_l1 = l1;
            l1 = -one_m_l * l0 + prev_l0 + one_m_l * prev_l1;
            const double prev_l2 = l2;
            l2 = -one_m_l * l1 + prev_l1 + one_m_l * prev_l2;
            l3 = -one_m_l * l2 + prev_l2 + one_m_l * l3;
        }
        const double cu = fmax(l0 - l1, 0.0) + fmax(l1 - l2, 0.0) + fmax(l2 - l3, 0.0);
        const double cd = fmax(l1 - l0, 0.0) + fmax(l2 - l1, 0.0) + fmax(l3 - l2, 0.0);
        const double lrsi_val = mgm_neo_nonzero(cu + cd) ? (100.0 * cu / (cu + cd)) : nn;

        // :1342-1385 -- money flow. use_volume is TRUE at the batch default.
        double mf_val = nn;
        {
            const double tp = (high[i] + low[i] + c_i) / 3.0;
            if (tp_has_prev) {
                int sign;
                if (tp > tp_prev) sign = 1;
                else if (tp < tp_prev) sign = -1;
                else sign = 0;
                const double mf_raw = tp * volume[i];
                if (rsi_seeded) {
                    const double old_mf = mf_ring_mf[mf_head];
                    const int old_sign = mf_ring_sgn[mf_head];
                    if (old_sign > 0) mf_pos_sum -= old_mf;
                    else if (old_sign < 0) mf_neg_sum -= old_mf;
                }
                mf_ring_mf[mf_head] = mf_raw;
                mf_ring_sgn[mf_head] = sign;
                if (sign > 0) mf_pos_sum += mf_raw;
                else if (sign < 0) mf_neg_sum += mf_raw;
                mf_head += 1;
                if (mf_head == mf_len) mf_head = 0;
                if (rsi_seeded) {
                    mf_val = (mf_neg_sum == 0.0)
                        ? 100.0
                        : (100.0 - 100.0 / (1.0 + (mf_pos_sum / mf_neg_sum)));
                }
            }
            tp_prev = tp;
            tp_has_prev = true;
        }

        // :1387-1421 -- cbci = momentum of the RSI plus an EMA of the RSI.
        double cbci_val = nn;
        if (rsi_seeded) {
            const double old = rsi_ring[rsi_ring_head];
            rsi_ring[rsi_ring_head] = rsi_val;
            rsi_ring_head += 1;
            if (rsi_ring_head == rsi_len) rsi_ring_head = 0;
            const double mom = (isfinite(old) && isfinite(rsi_val)) ? (rsi_val - old) : nn;
            if (!rsi_ema_seed && isfinite(rsi_val)) {
                rsi_ema = rsi_val;
                rsi_ema_seed = true;
            } else if (isfinite(rsi_val)) {
                rsi_ema = mgm_neo_ema_step(rsi_val, rsi_ema, alpha3, beta3);
            }
            if (isfinite(mom) && rsi_ema_seed) cbci_val = mom + rsi_ema;
        }

        // :1541-1560 -- TraditionMg: tci + mf + rsi + cbci + lrsi, in that
        // order. The order sets the rounding of `sum`.
        if (i >= warm) {
            double sum = 0.0;
            int cnt = 0;
            if (isfinite(tci_val))  { sum += tci_val;  cnt += 1; }
            if (isfinite(mf_val))   { sum += mf_val;   cnt += 1; }
            if (isfinite(rsi_val))  { sum += rsi_val;  cnt += 1; }
            if (isfinite(cbci_val)) { sum += cbci_val; cnt += 1; }
            if (isfinite(lrsi_val)) { sum += lrsi_val; cnt += 1; }

            if (cnt > 0) {
                const double wt = sum / (double)cnt;
                row[i] = wt;

                // The signal and histogram columns are not this lane's output,
                // but the six-wide ring must still advance because the CPU
                // advances it only on bars where cnt > 0.
                if (i >= sig_start) {
                    if (!have_sig) {
                        double s = 0.0;
                        for (int k = 0; k < MGM_NEO_SIGP; ++k) {
                            const double x = row[i + 1 - MGM_NEO_SIGP + k];
                            sig_ring[k] = x;
                            s += x;
                        }
                        sig_sum = s;
                        have_sig = true;
                        sig_head = 0;
                    } else {
                        const double old = sig_ring[sig_head];
                        sig_ring[sig_head] = wt;
                        sig_head += 1;
                        if (sig_head == MGM_NEO_SIGP) sig_head = 0;
                        sig_sum += wt - old;
                    }
                }
            }
        }
        prev_close = c_i;
    }
}
