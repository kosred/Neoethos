use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderExecutorConfig {
    pub symbol: String,
    pub partial_take_profit_enabled: bool,
    pub partial_tp_min_total_lot: f64,
    pub partial_tp_r_levels: Vec<f64>,
    pub partial_tp_size_fracs: Vec<f64>,
    pub min_risk_reward: f64,
    pub entry_patience_enabled: bool,
    pub entry_patience_bars: usize,
    pub entry_patience_pullback_atr: f64,
    pub min_edge_cost_multiple: f64,
    pub commission_per_lot: f64,
}

impl Default for OrderExecutorConfig {
    fn default() -> Self {
        Self {
            symbol: "EURUSD".to_string(),
            partial_take_profit_enabled: true,
            partial_tp_min_total_lot: 0.03,
            partial_tp_r_levels: vec![1.0, 2.0, 3.0],
            partial_tp_size_fracs: vec![0.5, 0.25, 0.25],
            min_risk_reward: 1.5,
            entry_patience_enabled: true,
            entry_patience_bars: 3,
            entry_patience_pullback_atr: 0.2,
            min_edge_cost_multiple: 3.0,
            commission_per_lot: 7.0,
        }
    }
}

pub struct OrderExecutor {
    pub config: OrderExecutorConfig,
}

impl OrderExecutor {
    pub fn new(config: OrderExecutorConfig) -> Self {
        Self { config }
    }

    pub fn build_order_legs(
        &self,
        total_size: f64,
        signal: i8,
        entry_price: f64,
        _sl: f64,
        sl_dist: f64,
        default_tp: f64,
    ) -> Vec<(f64, f64)> {
        if !self.config.partial_take_profit_enabled
            || total_size < self.config.partial_tp_min_total_lot
        {
            return vec![(Self::round_2(total_size), default_tp)];
        }

        let levels = &self.config.partial_tp_r_levels;
        let mut fracs = self.config.partial_tp_size_fracs.clone();
        let n = levels.len().min(fracs.len());

        if n == 0 {
            return vec![(Self::round_2(total_size), default_tp)];
        }

        let frac_sum: f64 = fracs[..n].iter().sum();
        if frac_sum <= 0.0 {
            return vec![(Self::round_2(total_size), default_tp)];
        }

        for f in &mut fracs[..n] {
            *f /= frac_sum;
        }

        let mut vols: Vec<f64> = fracs[..n]
            .iter()
            .map(|f| Self::round_2_down(total_size * f))
            .collect();

        let sum_vols: f64 = vols.iter().sum();
        let rem = Self::round_2(total_size - sum_vols);

        if rem > 0.0 && !vols.is_empty() {
            let mut max_idx = 0;
            let mut max_val = vols[0];
            for (i, &v) in vols.iter().enumerate().skip(1) {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }
            vols[max_idx] = Self::round_2(vols[max_idx] + rem);
        }

        let mut legs = Vec::new();
        for (vol, r) in vols.iter().zip(levels.iter()) {
            if *vol < 0.01 {
                continue;
            }
            let tp = if signal == 1 {
                entry_price + (r * sl_dist)
            } else {
                entry_price - (r * sl_dist)
            };
            legs.push((*vol, tp));
        }

        if legs.is_empty() {
            return vec![(Self::round_2(total_size), default_tp)];
        }
        legs
    }

    fn round_2(val: f64) -> f64 {
        (val * 100.0).round() / 100.0
    }

    fn round_2_down(val: f64) -> f64 {
        (val * 100.0).trunc() / 100.0
    }

    pub fn compute_order_prices(
        &self,
        entry_price: f64,
        signal: i8,
        sl_pips: f64,
        rr: f64,
        pip_size: f64,
    ) -> Result<(f64, f64, f64), String> {
        if !entry_price.is_finite()
            || !sl_pips.is_finite()
            || !rr.is_finite()
            || !pip_size.is_finite()
            || entry_price <= 0.0
            || sl_pips <= 0.0
            || rr <= 0.0
            || pip_size <= 0.0
        {
            return Err("entry_price/sl_pips/rr/pip_size must be finite positive values".into());
        }

        let sl_dist = sl_pips * pip_size;
        if signal > 0 {
            let sl = entry_price - sl_dist;
            let tp = entry_price + (rr * sl_dist);
            Ok((sl, tp, sl_dist))
        } else if signal < 0 {
            let sl = entry_price + sl_dist;
            let tp = entry_price - (rr * sl_dist);
            Ok((sl, tp, sl_dist))
        } else {
            Err("signal must be +1 or -1".into())
        }
    }

    pub fn evaluate_trade_edge(
        &self,
        sl_pips: f64,
        rr: f64,
        spread_pips: f64,
        slippage_pips: f64,
        pip_value_per_lot: f64,
    ) -> Result<(bool, f64, f64), String> {
        if !sl_pips.is_finite()
            || !rr.is_finite()
            || !spread_pips.is_finite()
            || !slippage_pips.is_finite()
            || !pip_value_per_lot.is_finite()
        {
            return Err("all inputs must be finite".into());
        }

        let commission_pips = self.config.commission_per_lot / pip_value_per_lot.max(1e-9);
        let total_cost_pips =
            (spread_pips.max(0.0) + slippage_pips.max(0.0) + commission_pips.max(0.0)).max(0.0);
        let expected_profit_pips = (sl_pips.max(0.0) * rr.max(0.0)).max(0.0);
        let passed = if self.config.min_edge_cost_multiple <= 0.0 {
            true
        } else {
            expected_profit_pips >= (self.config.min_edge_cost_multiple * total_cost_pips)
        };
        Ok((passed, expected_profit_pips, total_cost_pips))
    }
}
