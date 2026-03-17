use super::super::Ohlcv;

pub fn compute_smc_feature_columns(ohlcv: &Ohlcv) -> Vec<(String, Vec<f64>)> {
    let n = ohlcv.len();
    let mut ob = vec![0.0_f64; n];
    let mut fvg = vec![0.0_f64; n];
    let mut liq = vec![0.0_f64; n];
    let mut trend = vec![0.0_f64; n];
    let mut premium = vec![0.0_f64; n];
    let mut inducement = vec![0.0_f64; n];
    let mut bos = vec![0.0_f64; n];
    let mut choch = vec![0.0_f64; n];
    let mut eqh = vec![0.0_f64; n];
    let mut eql = vec![0.0_f64; n];
    let mut displacement = vec![0.0_f64; n];

    if n == 0 {
        return vec![
            ("smc_ob".to_string(), ob),
            ("smc_fvg".to_string(), fvg),
            ("smc_liq".to_string(), liq),
            ("smc_trend".to_string(), trend),
            ("smc_premium".to_string(), premium),
            ("smc_inducement".to_string(), inducement),
            ("smc_bos".to_string(), bos),
            ("smc_choch".to_string(), choch),
            ("smc_eqh".to_string(), eqh),
            ("smc_eql".to_string(), eql),
            ("smc_displacement".to_string(), displacement),
        ];
    }

    const TREND_LOOKBACK: usize = 12;
    const EQUAL_LOOKBACK: usize = 20;
    const DISPLACEMENT_LOOKBACK: usize = 20;
    const DISPLACEMENT_MULT: f64 = 1.8;

    for i in 0..n {
        let high_i = ohlcv.high[i];
        let low_i = ohlcv.low[i];
        let open_i = ohlcv.open[i];
        let close_i = ohlcv.close[i];

        if i >= TREND_LOOKBACK {
            let d = close_i - ohlcv.close[i - TREND_LOOKBACK];
            trend[i] = if d > 0.0 { 1.0 } else if d < 0.0 { -1.0 } else { 0.0 };
        } else if i >= 1 {
            let d = close_i - ohlcv.close[i - 1];
            trend[i] = if d > 0.0 { 1.0 } else if d < 0.0 { -1.0 } else { 0.0 };
        }

        let range_i = (high_i - low_i).abs();
        if range_i > 1e-12 {
            let rel = (close_i - low_i) / range_i;
            premium[i] = if rel <= 0.5 { 1.0 } else { -1.0 };
        }

        if i >= 1 {
            let prev_open = ohlcv.open[i - 1];
            let prev_close = ohlcv.close[i - 1];
            let prev_high = ohlcv.high[i - 1];
            let prev_low = ohlcv.low[i - 1];

            let bull_ob = close_i > open_i && prev_close < prev_open && close_i >= prev_high;
            let bear_ob = close_i < open_i && prev_close > prev_open && close_i <= prev_low;
            ob[i] = if bull_ob { 1.0 } else if bear_ob { -1.0 } else { 0.0 };

            let body = (close_i - open_i).abs();
            let upper_wick = high_i - open_i.max(close_i);
            let lower_wick = open_i.min(close_i) - low_i;
            if body > 1e-12 && ((upper_wick / body) > 2.0 || (lower_wick / body) > 2.0) {
                inducement[i] = 1.0;
            }
        }

        if i >= 2 {
            if ohlcv.low[i] > ohlcv.high[i - 2] { fvg[i] = 1.0; }
            else if ohlcv.high[i] < ohlcv.low[i - 2] { fvg[i] = -1.0; }
        }

        if i >= 3 {
            let prev_low = ohlcv.low[(i - 3)..i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let prev_high = ohlcv.high[(i - 3)..i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if ohlcv.low[i] < prev_low && ohlcv.close[i] > prev_low { liq[i] = 1.0; }
            else if ohlcv.high[i] > prev_high && ohlcv.close[i] < prev_high { liq[i] = -1.0; }
        }

        if i >= TREND_LOOKBACK {
            let prev_low = ohlcv.low[(i - TREND_LOOKBACK)..i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let prev_high = ohlcv.high[(i - TREND_LOOKBACK)..i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if ohlcv.close[i] > prev_high { bos[i] = 1.0; }
            else if ohlcv.close[i] < prev_low { bos[i] = -1.0; }
        }

        if i >= 1 && trend[i] != 0.0 && trend[i - 1] != 0.0 && trend[i] != trend[i - 1] {
            choch[i] = trend[i];
        }

        if i >= EQUAL_LOOKBACK {
            let lb = i - EQUAL_LOOKBACK;
            let mut range_sum = 0.0;
            for j in lb..=i { range_sum += (ohlcv.high[j] - ohlcv.low[j]).abs(); }
            let avg_range = range_sum / ((EQUAL_LOOKBACK as f64) + 1.0);
            let tol = (avg_range * 0.1).max(1e-9);
            for j in lb..i { if (ohlcv.high[i] - ohlcv.high[j]).abs() <= tol { eqh[i] = -1.0; break; } }
            for j in lb..i { if (ohlcv.low[i] - ohlcv.low[j]).abs() <= tol { eql[i] = 1.0; break; } }
        }

        if i >= DISPLACEMENT_LOOKBACK {
            let body = (ohlcv.close[i] - ohlcv.open[i]).abs();
            let mut avg_body = 0.0;
            for j in (i - DISPLACEMENT_LOOKBACK)..i { avg_body += (ohlcv.close[j] - ohlcv.open[j]).abs(); }
            avg_body /= DISPLACEMENT_LOOKBACK as f64;
            if avg_body > 1e-12 && body >= (DISPLACEMENT_MULT * avg_body) {
                displacement[i] = if ohlcv.close[i] > ohlcv.open[i] { 1.0 } else if ohlcv.close[i] < ohlcv.open[i] { -1.0 } else { 0.0 };
            }
        }
    }

    vec![
        ("smc_ob".to_string(), ob),
        ("smc_fvg".to_string(), fvg),
        ("smc_liq".to_string(), liq),
        ("smc_trend".to_string(), trend),
        ("smc_premium".to_string(), premium),
        ("smc_inducement".to_string(), inducement),
        ("smc_bos".to_string(), bos),
        ("smc_choch".to_string(), choch),
        ("smc_eqh".to_string(), eqh),
        ("smc_eql".to_string(), eql),
        ("smc_displacement".to_string(), displacement),
    ]
}
