use super::super::Ohlcv;
use chrono::{Datelike, TimeZone, Timelike, Utc};

pub fn compute_smc_feature_columns(ohlcv: &Ohlcv) -> Vec<(String, Vec<f64>)> {
    let n = ohlcv.len();
    // Matrix initialization - Deep Institutional IPDA Variables
    let mut ob = vec![0.0_f64; n];
    let mut fvg = vec![0.0_f64; n];
    let mut ifvg = vec![0.0_f64; n];
    let mut liq_sweep = vec![0.0_f64; n];
    let mut pd_array = vec![0.0_f64; n];
    let mut macro_active = vec![0.0_f64; n];
    let mut displacement = vec![0.0_f64; n];

    // Advanced SMC features
    let mut breaker_block = vec![0.0_f64; n];
    let mut mitigation_block = vec![0.0_f64; n];
    let mut mss = vec![0.0_f64; n]; // Market Structure Shift / CHOCH
    let mut volume_imbalance = vec![0.0_f64; n];

    // Deep ICT Concepts
    let mut bos = vec![0.0_f64; n]; // Break of Structure
    let mut eqh = vec![0.0_f64; n]; // Equal Highs (liquidity trap)
    let mut eql = vec![0.0_f64; n]; // Equal Lows (liquidity trap)
    let mut inducement = vec![0.0_f64; n]; // Minor swing bait before real move
    let mut asian_range = vec![0.0_f64; n]; // Asian session range position
    let mut silver_bullet = vec![0.0_f64; n]; // ICT Silver Bullet windows
    let mut judas_swing = vec![0.0_f64; n]; // Fake move at session open
    let mut nwog = vec![0.0_f64; n]; // New Week Opening Gap
    let mut ndog = vec![0.0_f64; n]; // New Day Opening Gap
    let mut ict_macro = vec![0.0_f64; n]; // ICT Macro time windows
    let mut fvg_strength = vec![0.0_f64; n]; // FVG gap size normalized by ATR
    let mut dealing_range_width = vec![0.0_f64; n]; // Dealing range as % of price
    let mut swing_range_pct = vec![0.0_f64; n]; // Current swing point range %
    let mut ob_strength = vec![0.0_f64; n]; // OB candle body vs range ratio
    let mut trend_bias = vec![0.0_f64; n]; // Multi-TF trend alignment
    let mut unicorn_model = vec![0.0_f64; n]; // Breaker + FVG + OB confluence
    let mut rejection_block = vec![0.0_f64; n]; // Long wick rejection zones
    let mut propulsion_block = vec![0.0_f64; n]; // Strong displacement after consolidation
    let mut fib_time_ratio = vec![0.0_f64; n]; // Fibonacci time-based clustering

    // Extended Fibonacci Matrix
    // Internal ranges (retracements)
    let mut fib_236 = vec![0.0_f64; n];
    let mut fib_382 = vec![0.0_f64; n];
    let mut fib_500 = vec![0.0_f64; n]; // Equilibrium exactly
    let mut fib_618 = vec![0.0_f64; n];
    let mut fib_705 = vec![0.0_f64; n];
    let mut fib_786 = vec![0.0_f64; n];
    let mut fib_886 = vec![0.0_f64; n];

    // External ranges (extensions / standard deviations)
    let mut fib_1272 = vec![0.0_f64; n];
    let mut fib_1414 = vec![0.0_f64; n];
    let mut fib_1618 = vec![0.0_f64; n];
    let mut fib_2000 = vec![0.0_f64; n];
    let mut fib_2618 = vec![0.0_f64; n];

    if n == 0 {
        return build_smc_return_vec(
            ob,
            fvg,
            ifvg,
            liq_sweep,
            pd_array,
            macro_active,
            displacement,
            breaker_block,
            mitigation_block,
            mss,
            volume_imbalance,
            bos,
            eqh,
            eql,
            inducement,
            asian_range,
            silver_bullet,
            judas_swing,
            nwog,
            ndog,
            ict_macro,
            fvg_strength,
            dealing_range_width,
            swing_range_pct,
            ob_strength,
            trend_bias,
            unicorn_model,
            rejection_block,
            propulsion_block,
            fib_time_ratio,
            fib_236,
            fib_382,
            fib_500,
            fib_618,
            fib_705,
            fib_786,
            fib_886,
            fib_1272,
            fib_1414,
            fib_1618,
            fib_2000,
            fib_2618,
        );
    }

    // IPDA Configuration
    const IPDA_LOOKBACK: usize = 40; // 40 period dealing range
    const SWING_FRACTAL: usize = 5; // 5-bar swing high/low required for old highs
    const DISPLACEMENT_LOOKBACK: usize = 20;
    const DISPLACEMENT_MULT: f64 = 1.8;

    // Active IPDA memory for IFVG computations
    let mut active_buy_fvgs: Vec<(f64, f64)> = Vec::new(); // (top, bottom) boundaries
    let mut active_sell_fvgs: Vec<(f64, f64)> = Vec::new(); // (top, bottom) boundaries
    let mut swing_highs: Vec<f64> = Vec::new();
    let mut swing_lows: Vec<f64> = Vec::new();

    // For MSS tracking
    let mut latest_bull_sweep_idx = 0;
    let mut latest_bear_sweep_idx = 0;
    let mut active_sell_ob: Vec<(usize, f64, f64)> = Vec::new(); // idx, top, bot
    let mut active_buy_ob: Vec<(usize, f64, f64)> = Vec::new(); // idx, top, bot

    // For BOS tracking
    let mut last_confirmed_high = f64::NEG_INFINITY;
    let mut last_confirmed_low = f64::INFINITY;

    // For Asian Range
    let mut asian_high = f64::NEG_INFINITY;
    let mut asian_low = f64::INFINITY;
    let mut asian_range_set = false;

    // For NWOG/NDOG
    let mut prev_day_close = f64::NAN;
    let mut prev_week_close = f64::NAN;
    let mut last_day: i64 = -1;
    let mut last_week: i64 = -1;

    // Running ATR for normalization
    let mut atr_sum = 0.0_f64;
    let mut atr_count = 0_usize;

    // Consolidation tracking for propulsion blocks
    let mut consol_count = 0_usize;
    let mut _consol_range_avg = 0.0_f64;

    for i in 0..n {
        let open = ohlcv.open[i];
        let high = ohlcv.high[i];
        let low = ohlcv.low[i];
        let close = ohlcv.close[i];
        let ts = ohlcv.timestamp.as_ref().map(|t| t[i]);

        // Running ATR
        if i > 0 {
            let tr = (high - low)
                .max((high - ohlcv.close[i - 1]).abs())
                .max((low - ohlcv.close[i - 1]).abs());
            atr_sum += tr;
            atr_count += 1;
        }
        let running_atr = if atr_count > 0 {
            atr_sum / atr_count as f64
        } else {
            (high - low).max(1e-10)
        };

        // 1. Killzones (Macro Time Delivery Algorithm)
        // London Killzone 07:00-11:00 UTC, NY Killzone 13:00-17:00 UTC
        macro_active[i] = 1.0;
        if let Some(t_ms) = ts {
            if let chrono::LocalResult::Single(dt) = Utc.timestamp_millis_opt(t_ms) {
                let hour = dt.hour();
                let minute = dt.minute();
                if (7..11).contains(&hour) || (13..17).contains(&hour) {
                    macro_active[i] = 1.0;
                } else {
                    macro_active[i] = 0.0;
                }

                // ICT Macro Time Windows (precise 15-90min windows of institutional activity)
                // 9:50-10:10 UTC, 10:50-11:10, 13:10-13:40, 14:50-15:10, 15:15-15:45
                let hm = hour * 60 + minute;
                if (590..=610).contains(&hm)
                    || (650..=670).contains(&hm)
                    || (790..=820).contains(&hm)
                    || (890..=910).contains(&hm)
                    || (915..=945).contains(&hm)
                {
                    ict_macro[i] = 1.0;
                }

                // Silver Bullet Windows: 10:00-11:00 UTC (London), 14:00-15:00 UTC (NY AM), 18:00-19:00 UTC (NY PM)
                if (hour == 10) || (hour == 14) || (hour == 18) {
                    silver_bullet[i] = 1.0;
                }

                // Asian Range: 00:00-08:00 UTC (track high/low for the session)
                if hour < 8 {
                    if high > asian_high {
                        asian_high = high;
                    }
                    if low < asian_low {
                        asian_low = low;
                    }
                    asian_range_set = true;
                } else if hour == 8 && minute == 0 {
                    // Reset at end of Asian session
                    asian_range_set = true;
                }

                // Asian Range Position (where is price relative to the Asian range?)
                if asian_range_set && asian_high > asian_low {
                    let ar = asian_high - asian_low;
                    asian_range[i] = (close - asian_low) / ar; // 0=at low, 1=at high, <0 or >1 = outside
                }

                // Reset Asian range at midnight UTC
                if hour == 0 && minute == 0 {
                    asian_high = f64::NEG_INFINITY;
                    asian_low = f64::INFINITY;
                    asian_range_set = false;
                }

                // NWOG/NDOG: New Day/Week Opening Gaps
                let day_key = dt.ordinal() as i64 + dt.year() as i64 * 400;
                let week_key = dt.iso_week().week() as i64 + dt.year() as i64 * 60;

                if day_key != last_day {
                    if prev_day_close.is_finite() {
                        // Gap between previous day close and current open
                        let gap = open - prev_day_close;
                        ndog[i] = gap / running_atr.max(1e-10);
                    }
                    prev_day_close = if i > 0 { ohlcv.close[i - 1] } else { close };
                    last_day = day_key;
                }

                if week_key != last_week {
                    if prev_week_close.is_finite() {
                        let gap = open - prev_week_close;
                        nwog[i] = gap / running_atr.max(1e-10);
                    }
                    prev_week_close = if i > 0 { ohlcv.close[i - 1] } else { close };
                    last_week = week_key;
                }

                // Judas Swing: Fake move in first 30 minutes of London/NY session
                if (hour == 7 || hour == 13) && minute < 30 && i >= 1 {
                    let prev_c = ohlcv.close[i - 1];
                    // If price moves strongly one way then reverses
                    if (high - prev_c).abs() > running_atr * 0.5 && close < prev_c {
                        judas_swing[i] = -1.0; // Bearish Judas (faked up)
                    } else if (prev_c - low).abs() > running_atr * 0.5 && close > prev_c {
                        judas_swing[i] = 1.0; // Bullish Judas (faked down)
                    }
                }
            }
        }

        // Volume Imbalance (Gap between bodies but wicks overlap)
        if i >= 1 {
            let prev_open = ohlcv.open[i - 1];
            let prev_close = ohlcv.close[i - 1];
            let prev_body_top = prev_open.max(prev_close);
            let prev_body_bot = prev_open.min(prev_close);
            let curr_body_top = open.max(close);
            let curr_body_bot = open.min(close);

            // Bullish Volume Imbalance
            if curr_body_bot > prev_body_top && low <= ohlcv.high[i - 1] {
                volume_imbalance[i] = 1.0;
            }
            // Bearish Volume Imbalance
            else if curr_body_top < prev_body_bot && high >= ohlcv.low[i - 1] {
                volume_imbalance[i] = -1.0;
            }
        }

        // Deep SMC Range Evaluation and Fibonacci Profiling
        if i >= IPDA_LOOKBACK {
            let mut highest = f64::NEG_INFINITY;
            let mut lowest = f64::INFINITY;
            for j in (i - IPDA_LOOKBACK)..i {
                if ohlcv.high[j] > highest {
                    highest = ohlcv.high[j];
                }
                if ohlcv.low[j] < lowest {
                    lowest = ohlcv.low[j];
                }
            }

            let equilibrium = (highest + lowest) / 2.0;
            pd_array[i] = if close > equilibrium { 1.0 } else { -1.0 };

            // Fibonacci Optimal Trade Entry (OTE) Distances + Extensions
            let range = highest - lowest;
            if range > 1e-9 {
                let dist_236 = ((close - (lowest + range * 0.236)).abs() / range).min(1.0);
                let dist_382 = ((close - (lowest + range * 0.382)).abs() / range).min(1.0);
                let dist_500 = ((close - equilibrium).abs() / range).min(1.0);
                let dist_618 = ((close - (lowest + range * 0.618)).abs() / range).min(1.0);
                let dist_705 = ((close - (lowest + range * 0.705)).abs() / range).min(1.0);
                let dist_786 = ((close - (lowest + range * 0.786)).abs() / range).min(1.0);
                let dist_886 = ((close - (lowest + range * 0.886)).abs() / range).min(1.0);

                // Extensions
                let dist_1272 = ((close - (lowest + range * 1.272)).abs() / range).min(1.0);
                let dist_1414 = ((close - (lowest + range * 1.414)).abs() / range).min(1.0);
                let dist_1618 = ((close - (lowest + range * 1.618)).abs() / range).min(1.0);
                let dist_2000 = ((close - (lowest + range * 2.0)).abs() / range).min(1.0);
                let dist_2618 = ((close - (lowest + range * 2.618)).abs() / range).min(1.0);

                // Convert to proximity score (peaks directly on the line)
                fib_236[i] = (1.0 - (15.0 * dist_236)).max(0.0);
                fib_382[i] = (1.0 - (15.0 * dist_382)).max(0.0);
                fib_500[i] = (1.0 - (15.0 * dist_500)).max(0.0);
                fib_618[i] = (1.0 - (15.0 * dist_618)).max(0.0);
                fib_705[i] = (1.0 - (15.0 * dist_705)).max(0.0);
                fib_786[i] = (1.0 - (15.0 * dist_786)).max(0.0);
                fib_886[i] = (1.0 - (15.0 * dist_886)).max(0.0);
                fib_1272[i] = (1.0 - (10.0 * dist_1272)).max(0.0);
                fib_1414[i] = (1.0 - (10.0 * dist_1414)).max(0.0);
                fib_1618[i] = (1.0 - (10.0 * dist_1618)).max(0.0);
                fib_2000[i] = (1.0 - (5.0 * dist_2000)).max(0.0);
                fib_2618[i] = (1.0 - (5.0 * dist_2618)).max(0.0);
            }
        }

        // Displacement Detection (Body size relative to historical moving average)
        if i >= DISPLACEMENT_LOOKBACK {
            let body = (close - open).abs();
            let mut avg_body = 0.0;
            for j in (i - DISPLACEMENT_LOOKBACK)..i {
                avg_body += (ohlcv.close[j] - ohlcv.open[j]).abs();
            }
            avg_body /= DISPLACEMENT_LOOKBACK as f64;
            if avg_body > 1e-12 && body >= (DISPLACEMENT_MULT * avg_body) {
                displacement[i] = if close > open { 1.0 } else { -1.0 };
            }
        }

        // Fractal Swing High/Low Tracking (For Engineered Liquidity pools)
        if i >= SWING_FRACTAL * 2 {
            let center = i - SWING_FRACTAL;
            let center_h = ohlcv.high[center];
            let center_l = ohlcv.low[center];
            let mut is_swing_high = true;
            let mut is_swing_low = true;

            for j in (center - SWING_FRACTAL)..=(center + SWING_FRACTAL) {
                if j == center {
                    continue;
                }
                if ohlcv.high[j] >= center_h {
                    is_swing_high = false;
                }
                if ohlcv.low[j] <= center_l {
                    is_swing_low = false;
                }
            }
            if is_swing_high {
                swing_highs.push(center_h);
            }
            if is_swing_low {
                swing_lows.push(center_l);
            }

            // Keep memory vector bounded to avoid bloat (Track last 15 pools)
            if swing_highs.len() > 15 {
                swing_highs.remove(0);
            }
            if swing_lows.len() > 15 {
                swing_lows.remove(0);
            }
        }

        // Purge and Revert (True Liquidity Sweeps - BISI/SIBI Mitigation)
        for &sh in &swing_highs {
            if high > sh && close < sh {
                liq_sweep[i] = -1.0;
                latest_bear_sweep_idx = i;
            }
        }
        for &sl in &swing_lows {
            if low < sl && close > sl {
                liq_sweep[i] = 1.0;
                latest_bull_sweep_idx = i;
            }
        }

        // Market Structure Shift (MSS) logic
        // Displacement past opposite swing high/low AFTER a sweep
        if displacement[i] == 1.0 && latest_bull_sweep_idx > 0 && i - latest_bull_sweep_idx <= 15 {
            // Swept lows, now displaced up past a recent swing high?
            if let Some(&recent_sh) = swing_highs.last() {
                if close > recent_sh && ohlcv.close[i - 1] <= recent_sh {
                    mss[i] = 1.0; // Bullish CHOCH
                }
            }
        }
        if displacement[i] == -1.0 && latest_bear_sweep_idx > 0 && i - latest_bear_sweep_idx <= 15 {
            // Swept highs, now displaced down past a recent swing low?
            if let Some(&recent_sl) = swing_lows.last() {
                if close < recent_sl && ohlcv.close[i - 1] >= recent_sl {
                    mss[i] = -1.0; // Bearish CHOCH
                } // Bearish CHOCH
            }
        }

        // Order Block / Mitigation / Breaker Block Validation
        if i >= 3 {
            let was_bull_sweep =
                liq_sweep[i - 1] == 1.0 || liq_sweep[i - 2] == 1.0 || liq_sweep[i - 3] == 1.0;
            let was_bear_sweep =
                liq_sweep[i - 1] == -1.0 || liq_sweep[i - 2] == -1.0 || liq_sweep[i - 3] == -1.0;

            let prev_open = ohlcv.open[i - 1];
            let prev_close = ohlcv.close[i - 1];
            let prev_high = ohlcv.high[i - 1];
            let prev_low = ohlcv.low[i - 1];

            // Institutional Bullish OB: Swept sell-side lows, displacing up past the down candle high
            if was_bull_sweep && close > open && prev_close < prev_open && close >= prev_high {
                ob[i] = 1.0;
                active_buy_ob.push((i, prev_high, prev_low));
            }
            // Institutional Bearish OB: Swept buy-side highs, displacing down past the up candle low
            if was_bear_sweep && close < open && prev_close > prev_open && close <= prev_low {
                ob[i] = -1.0;
                active_sell_ob.push((i, prev_high, prev_low));
            }

            // Mitigation block: Same profile as OB but failed to sweep liquidity first
            if !was_bull_sweep
                && close > open
                && prev_close < prev_open
                && close >= prev_high
                && displacement[i] == 1.0
            {
                mitigation_block[i] = 1.0;
            }
            if !was_bear_sweep
                && close < open
                && prev_close > prev_open
                && close <= prev_low
                && displacement[i] == -1.0
            {
                mitigation_block[i] = -1.0;
            }
        }

        // Breaker Block: Failed Order Block
        // Active Sell OB gets broken strongly up -> Becomes Bullish Breaker
        active_sell_ob.retain(|&(idx, top, _bot)| {
            if i - idx < 50 && displacement[i] == 1.0 && close > top {
                breaker_block[i] = 1.0; // Bullish Breaker Signal
                false // Re-evaluated
            } else {
                i - idx < 50
            }
        });

        // Active Buy OB gets broken strongly down -> Becomes Bearish Breaker
        active_buy_ob.retain(|&(idx, _top, bot)| {
            if i - idx < 50 && displacement[i] == -1.0 && close < bot {
                breaker_block[i] = -1.0; // Bearish Breaker Signal
                false // Re-evaluated
            } else {
                i - idx < 50
            }
        });

        // Standard FVG Generation
        if i >= 2 {
            let bot_fvg_sell = ohlcv.high[i];
            let top_fvg_sell = ohlcv.low[i - 2];
            // Bearish FVG: Low of candle 1 is higher than High of candle 3
            if bot_fvg_sell < top_fvg_sell {
                fvg[i] = -1.0;
                active_sell_fvgs.push((top_fvg_sell, bot_fvg_sell));
            }

            let top_fvg_buy = ohlcv.low[i];
            let bot_fvg_buy = ohlcv.high[i - 2];
            // Bullish FVG: High of candle 1 is lower than Low of candle 3
            if top_fvg_buy > bot_fvg_buy {
                fvg[i] = 1.0;
                active_buy_fvgs.push((top_fvg_buy, bot_fvg_buy));
            }
        }

        // Inversion FVG (IFVG) Detection
        // Retain active selling gaps; if price CLOSES above gap top, it's an IFVG flip to bullish
        active_sell_fvgs.retain(|&(top, _bot)| {
            if close > top {
                ifvg[i] = 1.0;
                false
            } else {
                true
            }
        });

        // Retain active buying gaps; if price CLOSES below gap bottom, it's an IFVG flip to bearish
        active_buy_fvgs.retain(|&(_top, bot)| {
            if close < bot {
                ifvg[i] = -1.0;
                false
            } else {
                true
            }
        });

        if active_buy_fvgs.len() > 10 {
            active_buy_fvgs.remove(0);
        }
        if active_sell_fvgs.len() > 10 {
            active_sell_fvgs.remove(0);
        }

        // FVG Strength: Gap size as a fraction of ATR (normalized significance)
        if fvg[i] != 0.0 && i >= 2 {
            let gap_size = if fvg[i] > 0.0 {
                ohlcv.low[i] - ohlcv.high[i - 2]
            } else {
                ohlcv.low[i - 2] - ohlcv.high[i]
            }
            .abs();
            fvg_strength[i] = gap_size / running_atr.max(1e-10);
        }

        // BOS: Break of Structure (simpler than MSS, no sweep required)
        if !swing_highs.is_empty() {
            let recent_high = *swing_highs.last().unwrap();
            if close > recent_high && recent_high > last_confirmed_high {
                bos[i] = 1.0;
                last_confirmed_high = recent_high;
            }
        }
        if !swing_lows.is_empty() {
            let recent_low = *swing_lows.last().unwrap();
            if close < recent_low && recent_low < last_confirmed_low {
                bos[i] = -1.0;
                last_confirmed_low = recent_low;
            }
        }

        // EQH/EQL: Equal Highs/Lows (engineered liquidity pools)
        // Two swing highs within 0.05% of each other = liquidity magnet
        if swing_highs.len() >= 2 {
            let h1 = swing_highs[swing_highs.len() - 1];
            let h2 = swing_highs[swing_highs.len() - 2];
            if h1 > 0.0 && ((h1 - h2).abs() / h1) < 0.0005 {
                eqh[i] = 1.0;
            }
        }
        if swing_lows.len() >= 2 {
            let l1 = swing_lows[swing_lows.len() - 1];
            let l2 = swing_lows[swing_lows.len() - 2];
            if l1 > 0.0 && ((l1 - l2).abs() / l1) < 0.0005 {
                eql[i] = 1.0;
            }
        }

        // Inducement: Minor swing broken without displacement (bait before real move)
        if i >= 3 {
            if swing_highs.len() >= 2 {
                let minor_high = swing_highs[swing_highs.len() - 1];
                if high > minor_high && displacement[i] == 0.0 {
                    inducement[i] = 1.0;
                }
            }
            if swing_lows.len() >= 2 {
                let minor_low = swing_lows[swing_lows.len() - 1];
                if low < minor_low && displacement[i] == 0.0 {
                    inducement[i] = -1.0;
                }
            }
        }

        // OB Strength: Body-to-range ratio of the OB candle
        if ob[i] != 0.0 && i >= 1 {
            let prev_range = ohlcv.high[i - 1] - ohlcv.low[i - 1];
            let prev_body = (ohlcv.close[i - 1] - ohlcv.open[i - 1]).abs();
            ob_strength[i] = if prev_range > 1e-10 {
                prev_body / prev_range
            } else {
                0.0
            };
        }

        // Rejection Block: Long wick with small body (> 60% wick of total range)
        let total_range = high - low;
        if total_range > 1e-10 {
            let body = (close - open).abs();
            let wick_ratio = 1.0 - (body / total_range);
            if wick_ratio > 0.6 {
                let upper_wick = high - close.max(open);
                let lower_wick = close.min(open) - low;
                if upper_wick > lower_wick {
                    rejection_block[i] = -1.0; // Bearish rejection (long upper wick)
                } else {
                    rejection_block[i] = 1.0; // Bullish rejection (long lower wick)
                }
            }
        }

        // Dealing Range Width (as % of price for cross-pair normalization)
        if i >= IPDA_LOOKBACK {
            let mut highest = f64::NEG_INFINITY;
            let mut lowest = f64::INFINITY;
            for j in (i - IPDA_LOOKBACK)..i {
                if ohlcv.high[j] > highest {
                    highest = ohlcv.high[j];
                }
                if ohlcv.low[j] < lowest {
                    lowest = ohlcv.low[j];
                }
            }
            dealing_range_width[i] = if close > 1e-10 {
                (highest - lowest) / close
            } else {
                0.0
            };
        }

        // Swing Range %
        if !swing_highs.is_empty() && !swing_lows.is_empty() {
            let sh = *swing_highs.last().unwrap();
            let sl = *swing_lows.last().unwrap();
            swing_range_pct[i] = if close > 1e-10 {
                (sh - sl).abs() / close
            } else {
                0.0
            };
        }

        // Propulsion Block: Strong displacement following consolidation (4+ bars narrow range)
        let bar_range = (high - low) / running_atr.max(1e-10);
        if bar_range < 0.5 {
            consol_count += 1;
            _consol_range_avg += bar_range;
        } else {
            if consol_count >= 4 && displacement[i] != 0.0 {
                propulsion_block[i] = displacement[i]; // Same direction as displacement
            }
            consol_count = 0;
            _consol_range_avg = 0.0;
        }

        // Trend Bias: Simple multi-period EMA alignment proxy (using closing price only)
        // Fast vs slow moving average via exponential smoothing
        if i >= 50 {
            let fast_period = 8;
            let slow_period = 50;
            let mut fast_sum = 0.0;
            let mut slow_sum = 0.0;
            for j in (i - fast_period)..=i {
                fast_sum += ohlcv.close[j];
            }
            for j in (i - slow_period)..=i {
                slow_sum += ohlcv.close[j];
            }
            let fast_ma = fast_sum / (fast_period + 1) as f64;
            let slow_ma = slow_sum / (slow_period + 1) as f64;
            trend_bias[i] = (fast_ma - slow_ma) / running_atr.max(1e-10);
        }

        // Unicorn Model: Confluence of Breaker + FVG + OB within last 5 bars
        if i >= 5 {
            let has_breaker = (i.saturating_sub(5)..=i).any(|j| breaker_block[j] != 0.0);
            let has_fvg = (i.saturating_sub(5)..=i).any(|j| fvg[j] != 0.0);
            let has_ob = (i.saturating_sub(5)..=i).any(|j| ob[j] != 0.0);
            if has_breaker && has_fvg && has_ob {
                unicorn_model[i] = if close > open { 1.0 } else { -1.0 };
            }
        }

        // Fibonacci Time Ratio: Distance from swing points in bars as Fib ratios
        // How many bars since last swing high/low, normalized by Fib numbers
        if i >= SWING_FRACTAL * 2 {
            // Use distance to pattern Fibonacci time clusters (8, 13, 21, 34, 55)
            let fib_times = [8, 13, 21, 34, 55];
            let bars_since_event = if liq_sweep[i] != 0.0 || mss[i] != 0.0 || displacement[i] != 0.0
            {
                0
            } else {
                // Look back to find last significant event
                let mut dist = 0_usize;
                for j in (0..i).rev() {
                    dist += 1;
                    if liq_sweep[j] != 0.0 || mss[j] != 0.0 || displacement[j] != 0.0 {
                        break;
                    }
                    if dist > 60 {
                        break;
                    }
                }
                dist
            };
            // Score peaks when bars_since_event matches a Fibonacci time number
            for &ft in &fib_times {
                if bars_since_event == ft {
                    fib_time_ratio[i] = 1.0;
                    break;
                } else if (bars_since_event as i64 - ft as i64).unsigned_abs() <= 1 {
                    fib_time_ratio[i] = 0.5;
                }
            }
        }
    }

    build_smc_return_vec(
        ob,
        fvg,
        ifvg,
        liq_sweep,
        pd_array,
        macro_active,
        displacement,
        breaker_block,
        mitigation_block,
        mss,
        volume_imbalance,
        bos,
        eqh,
        eql,
        inducement,
        asian_range,
        silver_bullet,
        judas_swing,
        nwog,
        ndog,
        ict_macro,
        fvg_strength,
        dealing_range_width,
        swing_range_pct,
        ob_strength,
        trend_bias,
        unicorn_model,
        rejection_block,
        propulsion_block,
        fib_time_ratio,
        fib_236,
        fib_382,
        fib_500,
        fib_618,
        fib_705,
        fib_786,
        fib_886,
        fib_1272,
        fib_1414,
        fib_1618,
        fib_2000,
        fib_2618,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_smc_return_vec(
    ob: Vec<f64>,
    fvg: Vec<f64>,
    ifvg: Vec<f64>,
    liq_sweep: Vec<f64>,
    pd_array: Vec<f64>,
    macro_active: Vec<f64>,
    displacement: Vec<f64>,
    breaker_block: Vec<f64>,
    mitigation_block: Vec<f64>,
    mss: Vec<f64>,
    volume_imbalance: Vec<f64>,
    bos: Vec<f64>,
    eqh: Vec<f64>,
    eql: Vec<f64>,
    inducement: Vec<f64>,
    asian_range: Vec<f64>,
    silver_bullet: Vec<f64>,
    judas_swing: Vec<f64>,
    nwog: Vec<f64>,
    ndog: Vec<f64>,
    ict_macro: Vec<f64>,
    fvg_strength: Vec<f64>,
    dealing_range_width: Vec<f64>,
    swing_range_pct: Vec<f64>,
    ob_strength: Vec<f64>,
    trend_bias: Vec<f64>,
    unicorn_model: Vec<f64>,
    rejection_block: Vec<f64>,
    propulsion_block: Vec<f64>,
    fib_time_ratio: Vec<f64>,
    fib_236: Vec<f64>,
    fib_382: Vec<f64>,
    fib_500: Vec<f64>,
    fib_618: Vec<f64>,
    fib_705: Vec<f64>,
    fib_786: Vec<f64>,
    fib_886: Vec<f64>,
    fib_1272: Vec<f64>,
    fib_1414: Vec<f64>,
    fib_1618: Vec<f64>,
    fib_2000: Vec<f64>,
    fib_2618: Vec<f64>,
) -> Vec<(String, Vec<f64>)> {
    vec![
        ("smc_ob".to_string(), ob),
        ("smc_fvg".to_string(), fvg),
        ("smc_ifvg".to_string(), ifvg),
        ("smc_liq_sweep".to_string(), liq_sweep),
        ("smc_pd_array".to_string(), pd_array),
        ("smc_killzone".to_string(), macro_active),
        ("smc_displacement".to_string(), displacement),
        ("smc_breaker_block".to_string(), breaker_block),
        ("smc_mitigation_block".to_string(), mitigation_block),
        ("smc_mss".to_string(), mss),
        ("smc_volume_imbalance".to_string(), volume_imbalance),
        ("smc_bos".to_string(), bos),
        ("smc_eqh".to_string(), eqh),
        ("smc_eql".to_string(), eql),
        ("smc_inducement".to_string(), inducement),
        ("smc_asian_range".to_string(), asian_range),
        ("smc_silver_bullet".to_string(), silver_bullet),
        ("smc_judas_swing".to_string(), judas_swing),
        ("smc_nwog".to_string(), nwog),
        ("smc_ndog".to_string(), ndog),
        ("smc_ict_macro".to_string(), ict_macro),
        ("smc_fvg_strength".to_string(), fvg_strength),
        ("smc_dealing_range_width".to_string(), dealing_range_width),
        ("smc_swing_range_pct".to_string(), swing_range_pct),
        ("smc_ob_strength".to_string(), ob_strength),
        ("smc_trend_bias".to_string(), trend_bias),
        ("smc_unicorn_model".to_string(), unicorn_model),
        ("smc_rejection_block".to_string(), rejection_block),
        ("smc_propulsion_block".to_string(), propulsion_block),
        ("smc_fib_time_ratio".to_string(), fib_time_ratio),
        ("smc_fib_236".to_string(), fib_236),
        ("smc_fib_382".to_string(), fib_382),
        ("smc_fib_500".to_string(), fib_500),
        ("smc_fib_618".to_string(), fib_618),
        ("smc_fib_705".to_string(), fib_705),
        ("smc_fib_786".to_string(), fib_786),
        ("smc_fib_886".to_string(), fib_886),
        ("smc_fib_1272".to_string(), fib_1272),
        ("smc_fib_1414".to_string(), fib_1414),
        ("smc_fib_1618".to_string(), fib_1618),
        ("smc_fib_2000".to_string(), fib_2000),
        ("smc_fib_2618".to_string(), fib_2618),
    ]
}
