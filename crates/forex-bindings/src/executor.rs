use pyo3::prelude::*;
use pyo3::types::PyAny;
use forex_core::domain::risk::{PropFirmRules, RiskManager as CoreRiskManager};
use crate::utils::{split_symbol, pip_size_from_parts, contract_size_from_parts, norm_symbol, quote_to_account_rate};
use std::collections::HashMap;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (
    equity,
    risk_pct,
    stop_loss_pips,
    pip_value,
    max_lot_size=10.0,
    lot_step=0.01,
    min_lot=0.0
))]
pub fn compute_position_size_lots(
    equity: f64,
    risk_pct: f64,
    stop_loss_pips: f64,
    pip_value: f64,
    max_lot_size: f64,
    lot_step: f64,
    min_lot: f64,
) -> PyResult<f64> {
    if !equity.is_finite()
        || !risk_pct.is_finite()
        || !stop_loss_pips.is_finite()
        || !pip_value.is_finite()
        || equity <= 0.0
        || risk_pct <= 0.0
        || stop_loss_pips <= 0.0
        || pip_value <= 0.0
    {
        return Ok(0.0);
    }

    let risk_amount = equity * risk_pct.max(0.0);
    let denom = (stop_loss_pips * pip_value).max(1e-9);
    let mut lot_size = risk_amount / denom;
    if !lot_size.is_finite() || lot_size <= 0.0 {
        return Ok(0.0);
    }

    let step = if lot_step.is_finite() && lot_step > 0.0 {
        lot_step
    } else {
        0.01
    };
    lot_size = (lot_size / step).floor() * step;

    let cap = if max_lot_size.is_finite() && max_lot_size > 0.0 {
        max_lot_size
    } else {
        lot_size
    };
    let floor = if min_lot.is_finite() {
        min_lot.max(0.0)
    } else {
        0.0
    };
    lot_size = lot_size.max(floor).min(cap);
    if !lot_size.is_finite() || lot_size < floor {
        return Ok(0.0);
    }
    Ok(lot_size.max(0.0))
}

#[pyfunction]
#[pyo3(signature = (symbol, point=None, digits=None))]
pub fn pip_size_from_symbol(symbol: &str, point: Option<f64>, digits: Option<i64>) -> PyResult<f64> {
    let sym = symbol.to_ascii_uppercase();
    let pip_size = if let (Some(pt), Some(dig)) = (point, digits) {
        let ptv = if pt.is_finite() && pt > 0.0 {
            pt
        } else {
            0.0001
        };
        let d = dig.max(0) as i32;
        if sym.ends_with("JPY") || sym.starts_with("JPY") {
            ptv * if d >= 3 { 10.0 } else { 1.0 }
        } else if sym.starts_with("XAU") || sym.starts_with("XAG") {
            0.01
        } else if sym.contains("BTC") || sym.contains("ETH") || sym.contains("LTC") {
            1.0
        } else {
            ptv * if d >= 4 { 10.0 } else { 1.0 }
        }
    } else if sym.ends_with("JPY") || sym.starts_with("JPY") {
        0.01
    } else if sym.starts_with("XAU") || sym.starts_with("XAG") {
        0.01
    } else if sym.contains("BTC") || sym.contains("ETH") || sym.contains("LTC") {
        1.0
    } else {
        0.0001
    };
    Ok(pip_size.max(1e-9))
}

#[pyfunction]
#[pyo3(signature = (symbol, price=None, account_currency="USD", reference_prices=None))]
pub fn infer_pip_metrics(
    symbol: &str,
    price: Option<f64>,
    account_currency: &str,
    reference_prices: Option<&Bound<'_, PyAny>>,
) -> PyResult<(f64, f64)> {
    let parts = split_symbol(symbol);
    let pip_size = pip_size_from_parts(symbol, parts.as_ref());
    let contract_size = contract_size_from_parts(symbol, parts.as_ref());
    let pip_value_quote = pip_size * contract_size;

    let mut refs: HashMap<String, f64> = HashMap::new();
    if let Some(raw) = reference_prices {
        if let Ok(dict) = raw.cast::<PyDict>() {
            for (k, v) in dict.iter() {
                let key = match k.extract::<String>() {
                    Ok(s) => norm_symbol(&s),
                    Err(_) => continue,
                };
                if key.len() != 6 {
                    continue;
                }
                let val = match v.extract::<f64>() {
                    Ok(x) if x.is_finite() && x > 0.0 => x,
                    _ => continue,
                };
                refs.insert(key, val);
            }
        }
    }

    let mut pip_value = pip_value_quote;
    if let Some((base, quote)) = parts {
        let rate = quote_to_account_rate(&base, &quote, account_currency, price, &refs);
        if let Some(r) = rate {
            if r.is_finite() && r > 0.0 {
                pip_value = pip_value_quote * r;
            }
        }
    }

    if !pip_value.is_finite() || pip_value <= 0.0 {
        pip_value = pip_value_quote.max(1e-6);
    }
    Ok((pip_size, pip_value))
}

#[pyclass(name = "RiskManager")]
pub struct RiskManager {
    pub inner: CoreRiskManager,
}

#[pymethods]
impl RiskManager {
    #[new]
    #[pyo3(signature = (prop_max_daily_loss_pct=0.045, prop_max_total_loss_pct=0.10, prop_max_trades_per_day=3, challenge_mode=false, initial_balance=10000.0))]
    pub fn new(
        prop_max_daily_loss_pct: f64,
        prop_max_total_loss_pct: f64,
        prop_max_trades_per_day: usize,
        challenge_mode: bool,
        initial_balance: f64,
    ) -> Self {
        let rules = PropFirmRules {
            max_daily_loss_pct: prop_max_daily_loss_pct,
            max_total_loss_pct: prop_max_total_loss_pct,
            profit_target_pct: 0.10,
            min_trading_days: 5,
            max_trading_days: 60,
            max_lot_size: 10.0,
            news_trading_allowed: false,
            weekend_holding: false,
            scaling_enabled: true,
            daily_dd_warning_pct: prop_max_daily_loss_pct * 0.8,
            daily_dd_stop_trading_pct: prop_max_daily_loss_pct,
            daily_profit_lock_pct: 0.03,
            max_trades_per_day: prop_max_trades_per_day,
        };
        Self {
            inner: CoreRiskManager::new(rules, challenge_mode, initial_balance),
        }
    }

    pub fn calculate_position_size(
        &mut self,
        equity: f64,
        base_risk_pct: f64,
        max_risk_cap: f64,
        confidence: f64,
        uncertainty: f64,
        market_volatility: f64,
        target_volatility: f64,
        is_volatile_regime: bool,
    ) -> f64 {
        self.inner.calculate_position_size(
            equity,
            base_risk_pct,
            max_risk_cap,
            confidence,
            uncertainty,
            market_volatility,
            target_volatility,
            is_volatile_regime,
        )
    }

    pub fn update_recovery_state(&mut self, equity: f64) {
        self.inner.update_recovery_state(equity);
    }
}

#[pyclass(name = "OrderExecutor")]
pub struct OrderExecutor {
    pub inner: forex_core::domain::order_execution::OrderExecutor,
}

#[pymethods]
impl OrderExecutor {
    #[new]
    #[pyo3(signature = (
        symbol="EURUSD".to_string(),
        partial_take_profit_enabled=true,
        partial_tp_min_total_lot=0.03,
        partial_tp_r_levels=vec![1.0, 2.0, 3.0],
        partial_tp_size_fracs=vec![0.5, 0.25, 0.25],
        min_risk_reward=1.5,
        entry_patience_enabled=true,
        entry_patience_bars=3,
        entry_patience_pullback_atr=0.2,
        min_edge_cost_multiple=3.0,
        commission_per_lot=7.0
    ))]
    pub fn new(
        symbol: String,
        partial_take_profit_enabled: bool,
        partial_tp_min_total_lot: f64,
        partial_tp_r_levels: Vec<f64>,
        partial_tp_size_fracs: Vec<f64>,
        min_risk_reward: f64,
        entry_patience_enabled: bool,
        entry_patience_bars: usize,
        entry_patience_pullback_atr: f64,
        min_edge_cost_multiple: f64,
        commission_per_lot: f64,
    ) -> Self {
        let config = forex_core::domain::order_execution::OrderExecutorConfig {
            symbol,
            partial_take_profit_enabled,
            partial_tp_min_total_lot,
            partial_tp_r_levels,
            partial_tp_size_fracs,
            min_risk_reward,
            entry_patience_enabled,
            entry_patience_bars,
            entry_patience_pullback_atr,
            min_edge_cost_multiple,
            commission_per_lot,
        };
        Self {
            inner: forex_core::domain::order_execution::OrderExecutor::new(config),
        }
    }

    pub fn build_order_legs(
        &self,
        total_size: f64,
        signal: i8,
        entry_price: f64,
        sl: f64,
        sl_dist: f64,
        default_tp: f64,
    ) -> Vec<(f64, f64)> {
        self.inner.build_order_legs(total_size, signal, entry_price, sl, sl_dist, default_tp)
    }

    pub fn compute_order_prices(
        &self,
        entry_price: f64,
        signal: i8,
        sl_pips: f64,
        rr: f64,
        pip_size: f64,
    ) -> PyResult<(f64, f64, f64)> {
        self.inner
            .compute_order_prices(entry_price, signal, sl_pips, rr, pip_size)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    pub fn evaluate_trade_edge(
        &self,
        sl_pips: f64,
        rr: f64,
        spread_pips: f64,
        slippage_pips: f64,
        pip_value_per_lot: f64,
    ) -> PyResult<(bool, f64, f64)> {
        self.inner
            .evaluate_trade_edge(sl_pips, rr, spread_pips, slippage_pips, pip_value_per_lot)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }
}
