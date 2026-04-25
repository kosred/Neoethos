use forex_core::domain::consistency::{ConsistencyTracker as CoreConsistencyTracker, TradeEvent};
use forex_core::domain::drift_monitor::ConceptDriftMonitor as CoreDriftMonitor;
use forex_core::domain::meta_controller::{MetaController as CoreMetaController, PropMetaState};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

#[pyclass(name = "ConsistencyTracker")]
pub struct ConsistencyTracker {
    pub inner: CoreConsistencyTracker,
}

#[pymethods]
impl ConsistencyTracker {
    #[new]
    #[pyo3(signature = (cache_dir, lookback_days=30))]
    pub fn new(cache_dir: &Bound<'_, PyAny>, lookback_days: i64) -> Self {
        let _ = cache_dir;
        Self {
            inner: CoreConsistencyTracker::new(lookback_days),
        }
    }

    pub fn update(&mut self, trade_event: &Bound<'_, PyDict>) -> PyResult<()> {
        let entry_time: String = trade_event.get_item("entry_time")?.unwrap().extract()?;
        let pnl: f64 = trade_event
            .get_item("pnl")?
            .map(|x| x.extract().unwrap_or(0.0))
            .unwrap_or(0.0);
        let risk_pct: f64 = trade_event
            .get_item("risk_pct")?
            .map(|x| x.extract().unwrap_or(0.0))
            .unwrap_or(0.0);
        let size: f64 = trade_event
            .get_item("size")?
            .map(|x| x.extract().unwrap_or(0.0))
            .unwrap_or(0.0);
        let hold_minutes: f64 = trade_event
            .get_item("hold_minutes")?
            .map(|x| x.extract().unwrap_or(0.0))
            .unwrap_or(0.0);

        let win: Option<i32> = match trade_event.get_item("win")? {
            Some(v) => {
                if let Ok(b) = v.extract::<bool>() {
                    Some(if b { 1 } else { 0 })
                } else {
                    v.extract::<i32>().ok()
                }
            }
            None => None,
        };

        let event = TradeEvent {
            entry_time,
            pnl,
            risk_pct,
            size,
            hold_minutes,
            win,
        };
        self.inner.update(&event);
        Ok(())
    }

    pub fn get_metrics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let metrics = self.inner.get_metrics();
        let dict = PyDict::new(py);
        dict.set_item("score", metrics.score)?;
        dict.set_item("daily_profit_consistency", metrics.daily_profit_consistency)?;
        dict.set_item("daily_trade_consistency", metrics.daily_trade_consistency)?;
        dict.set_item("daily_risk_consistency", metrics.daily_risk_consistency)?;
        dict.set_item(
            "weekly_profit_consistency",
            metrics.weekly_profit_consistency,
        )?;
        dict.set_item(
            "weekly_drawdown_consistency",
            metrics.weekly_drawdown_consistency,
        )?;
        dict.set_item("trade_size_consistency", metrics.trade_size_consistency)?;
        dict.set_item("hold_time_consistency", metrics.hold_time_consistency)?;
        dict.set_item("win_rate_rolling", metrics.win_rate_rolling)?;
        dict.set_item("grade", metrics.grade)?;

        let dataclass_module = PyModule::import(py, "forex_bot.execution.consistency")?;
        let class = dataclass_module.getattr("ConsistencyMetrics")?;
        let inst = class.call((), Some(&dict))?;
        Ok(inst)
    }
}

#[pyclass(name = "MetaController")]
pub struct MetaController {
    pub inner: CoreMetaController,
}

#[pymethods]
impl MetaController {
    #[new]
    #[pyo3(signature = (max_daily_dd=None, safety_buffer=None, base_risk_per_trade=None, base_confidence=None, settings=None, silent=None))]
    pub fn new(
        max_daily_dd: Option<f64>,
        safety_buffer: Option<f64>,
        base_risk_per_trade: Option<f64>,
        base_confidence: Option<f64>,
        settings: Option<&Bound<'_, PyAny>>,
        silent: Option<bool>,
    ) -> PyResult<Self> {
        let mut k_steepness = 200.0;
        let mut final_base_confidence = base_confidence.unwrap_or(0.55);

        if let Some(s) = settings {
            if let Ok(dyn_cfg) = s.getattr("dynamic") {
                if let Ok(risk_params) = dyn_cfg.call_method0("get") {
                    if let Ok(k) = risk_params.call_method1("get", ("risk_curve_steepness", 200.0))
                    {
                        k_steepness = k.extract().unwrap_or(200.0);
                    }
                    if let Ok(c) = risk_params.call_method1("get", ("confidence_threshold",)) {
                        if let Ok(c_val) = c.extract::<f64>() {
                            final_base_confidence = c_val;
                        }
                    }
                }
            }
        }

        Ok(Self {
            inner: CoreMetaController::new(
                max_daily_dd,
                safety_buffer,
                base_risk_per_trade,
                Some(final_base_confidence),
                silent,
                Some(k_steepness),
            ),
        })
    }

    pub fn get_risk_parameters(&mut self, state: &Bound<'_, PyAny>) -> PyResult<(f64, f64, bool)> {
        let mut m_regime = "Normal".to_string();
        if let Ok(regime) = state.getattr("market_regime") {
            if let Ok(r) = regime.extract::<String>() {
                m_regime = r;
            }
        }

        let s = PropMetaState {
            daily_dd_pct: state.getattr("daily_dd_pct")?.extract()?,
            daily_profit_pct: 0.0,
            volatility_regime: state.getattr("volatility_regime")?.extract()?,
            recent_win_rate: state.getattr("recent_win_rate")?.extract()?,
            consecutive_losses: state.getattr("consecutive_losses")?.extract()?,
            model_confidence: state.getattr("model_confidence")?.extract()?,
            hour_of_day: state.getattr("hour_of_day")?.extract()?,
            market_regime: m_regime,
        };
        Ok(self.inner.get_risk_parameters(&s))
    }
}

#[pyclass(name = "ConceptDriftMonitor")]
pub struct ConceptDriftMonitor {
    pub inner: CoreDriftMonitor,
}

#[pymethods]
impl ConceptDriftMonitor {
    #[new]
    #[pyo3(signature = (window_size=100, threshold=0.05))]
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            inner: CoreDriftMonitor::new(Some(window_size), Some(threshold)),
        }
    }

    pub fn update(&mut self, y_true: i32, y_pred_prob: Vec<f64>) -> bool {
        self.inner.update(y_true, &y_pred_prob)
    }

    pub fn should_retrain(&self) -> bool {
        self.inner.should_retrain()
    }

    pub fn reset_after_retrain(&mut self) {
        self.inner.reset_after_retrain();
    }

    pub fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("drift_detected", self.inner.drift_detected)?;
        dict.set_item("drift_magnitude", self.inner.drift_magnitude)?;
        dict.set_item("drift_method", &self.inner.drift_method_used)?;
        dict.set_item("errors_tracked", self.inner.error_stream.len())?;
        dict.set_item("last_drift_at", self.inner.last_drift_at)?;
        dict.set_item("ks_statistic", self.inner.ks_statistic)?;
        dict.set_item("psi_score", self.inner.psi_score)?;
        dict.set_item("kl_divergence", self.inner.kl_divergence)?;
        Ok(dict)
    }
}
