pub fn detect_divergence(price: &[f64], indicator: &[f64], window: usize) -> Vec<f64> {
    let n = price.len();
    let mut out = vec![0.0; n];
    if n < window || indicator.len() != n {
        return out;
    }
    for i in window..n {
        let p_curr = price[i];
        let ind_curr = indicator[i];
        let mut p_min = f64::INFINITY;
        let mut ind_min = f64::INFINITY;
        let mut p_max = f64::NEG_INFINITY;
        let mut ind_max = f64::NEG_INFINITY;
        for j in (i - window)..i {
            p_min = p_min.min(price[j]);
            ind_min = ind_min.min(indicator[j]);
            p_max = p_max.max(price[j]);
            ind_max = ind_max.max(indicator[j]);
        }
        if p_curr < p_min && ind_curr > ind_min {
            out[i] = 1.0;
        } else if p_curr > p_max && ind_curr < ind_max {
            out[i] = -1.0;
        }
    }
    out
}

pub fn vortex_indicator(high: &[f64], low: &[f64], close: &[f64], period: usize) -> (Vec<f64>, Vec<f64>) {
    let n = close.len();
    let mut vi_plus = vec![1.0; n];
    let mut vi_minus = vec![1.0; n];
    if n < period + 1 || high.len() != n || low.len() != n {
        return (vi_plus, vi_minus);
    }
    let mut vm_plus = vec![0.0; n];
    let mut vm_minus = vec![0.0; n];
    let mut tr = vec![0.0; n];
    for i in 1..n {
        vm_plus[i] = (high[i] - low[i - 1]).abs();
        vm_minus[i] = (low[i] - high[i - 1]).abs();
        tr[i] = (high[i] - low[i]).max((high[i] - close[i - 1]).abs()).max((low[i] - close[i - 1]).abs());
    }
    let mut s_vmp = 0.0;
    let mut s_vmm = 0.0;
    let mut s_tr = 0.0;
    for i in 0..n {
        s_vmp += vm_plus[i];
        s_vmm += vm_minus[i];
        s_tr += tr[i];
        if i >= period {
            s_vmp -= vm_plus[i - period];
            s_vmm -= vm_minus[i - period];
            s_tr -= tr[i - period];
        }
        if i >= period - 1 && s_tr > 0.0 {
            vi_plus[i] = s_vmp / s_tr;
            vi_minus[i] = s_vmm / s_tr;
        }
    }
    (vi_plus, vi_minus)
}

pub fn fisher_transform(price: &[f64], period: usize) -> Vec<f64> {
    let n = price.len();
    let mut fisher = vec![0.0; n];
    if n < period {
        return fisher;
    }
    let mut value = vec![0.0; n];
    for i in period..n {
        let mut p_min = f64::INFINITY;
        let mut p_max = f64::NEG_INFINITY;
        for &sample in price.iter().take(i + 1).skip(i - period + 1) {
            p_min = p_min.min(sample);
            p_max = p_max.max(sample);
        }
        let mut val = if p_max != p_min {
            0.66 * ((price[i] - p_min) / (p_max - p_min) - 0.5) + 0.67 * value[i - 1]
        } else {
            0.0
        };
        val = val.clamp(-0.999, 0.999);
        value[i] = val;
        fisher[i] = 0.5 * ((1.0 + val) / (1.0 - val)).ln() + 0.5 * fisher[i - 1];
    }
    fisher
}
