use anyhow::{bail, Result};
use forex_core::logging::write_subsystem_record;
use forex_core::sectioned_log::{SectionedRunRecord, SubsystemSection};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};

fn format_last_error(err: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok((code, description)) = err.extract::<(i32, String)>() {
        return Ok(format!("code={} description={}", code, description));
    }

    if let Ok(description) = err.extract::<String>() {
        return Ok(description);
    }

    Ok(err.str()?.to_string_lossy().into_owned())
}

fn mt5_record(operation: &str, status: &str, message: impl Into<String>) -> SectionedRunRecord {
    let now = system_time_string();
    let operation = operation.to_string();
    SectionedRunRecord {
        run_id: format!("mt5-{}-{}", operation, now.replace(':', "-")),
        parent_run_id: None,
        started_at: now.clone(),
        finished_at: now,
        subsystem: SubsystemSection::Mt5,
        operation,
        status: status.to_string(),
        symbol: None,
        timeframe: None,
        error_code: None,
        message: message.into(),
        body: String::new(),
    }
}

fn system_time_string() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch");
    format!("{}.{:09}Z", now.as_secs(), now.subsec_nanos())
}

fn record_mt5_event(operation: &str, status: &str, message: impl Into<String>) {
    let message = message.into();
    if let Err(err) = write_subsystem_record(
        SubsystemSection::Mt5,
        mt5_record(operation, status, message.clone()),
    ) {
        error!(
            "Failed to update canonical MT5 log for operation={} status={}: {}",
            operation, status, err
        );
        eprintln!(
            "Failed to update canonical MT5 log for operation={} status={}: {}",
            operation, status, err
        );
    }
}

pub struct MT5Engine {
    connected: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PositionInfo {
    pub ticket: i64,
    pub symbol: String,
    pub order_side: String,
    pub volume: f64,
    pub price_open: f64,
    pub price_current: f64,
    pub profit: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub comment: String,
    pub opened_at: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PendingOrderInfo {
    pub ticket: i64,
    pub symbol: String,
    pub order_kind: String,
    pub volume_initial: f64,
    pub price_open: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub comment: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DealInfo {
    pub ticket: i64,
    pub order_ticket: i64,
    pub position_id: i64,
    pub symbol: String,
    pub entry_kind: String,
    pub order_side: String,
    pub volume: f64,
    pub price: f64,
    pub profit: f64,
    pub fee: f64,
    pub comment: String,
    pub executed_at: i64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HistoricalBarInfo {
    pub timestamp_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub tick_volume: f64,
    pub spread: i64,
    pub real_volume: f64,
}

impl MT5Engine {
    pub fn new() -> Result<Self> {
        // Python::initialize() is the new way in 0.26+
        Python::initialize();
        Ok(Self { connected: false })
    }

    pub fn initialize(&mut self) -> Result<bool> {
        record_mt5_event("initialize", "STARTED", "starting MT5 initialization");
        let res = Python::attach(|py| -> PyResult<bool> {
            let mt5 = match py.import("MetaTrader5") {
                Ok(m) => m,
                Err(e) => {
                    warn!("MetaTrader5 module not found: {}", e);
                    record_mt5_event(
                        "initialize",
                        "DEGRADED",
                        format!("MetaTrader5 module not found: {e}"),
                    );
                    return Ok(false);
                }
            };
            
            let init_result: bool = mt5.getattr("initialize")?.call0()?.extract()?;
            
            if !init_result {
                let err_obj = mt5.getattr("last_error")?.call0()?;
                let err = format_last_error(err_obj.as_any())?;
                error!("MT5 Initialization failed. Last error: {}", err);
                record_mt5_event("initialize", "FAILED", format!("MT5 initialize returned false: {err}"));
                return Ok(false);
            }

            info!("MT5 successfully initialized from Pure Rust.");
            record_mt5_event("initialize", "SUCCESS", "MT5 successfully initialized from Pure Rust");
            Ok(true)
        }).map_err(|e| anyhow::anyhow!("PyError: {}", e))?;
        
        self.connected = res;
        Ok(res)
    }

    pub fn terminal_info(&self) -> Result<String> {
        if !self.connected { anyhow::bail!("MT5 is not connected."); }

        Python::attach(|py| -> PyResult<String> {
            let mt5 = py.import("MetaTrader5")?;
            let info = mt5.getattr("terminal_info")?.call0()?;
            if info.is_none() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err("Failed to get terminal info."));
            }
            Ok(info.to_string())
        }).map_err(|e| anyhow::anyhow!("PyError: {}", e))
    }

    pub fn positions(&self, symbol: Option<&str>) -> Result<Vec<PositionInfo>> {
        if !self.connected {
            bail!("MT5 is not connected.");
        }

        Python::attach(|py| -> PyResult<Vec<PositionInfo>> {
            let mt5 = py.import("MetaTrader5")?;
            let positions = call_mt5_collection(py, &mt5, "positions_get", symbol)?;
            positions
                .try_iter()?
                .map(|item| parse_position_info(&item?))
                .collect()
        })
        .map_err(|e| anyhow::anyhow!("PyError: {}", e))
    }

    pub fn orders(&self, symbol: Option<&str>) -> Result<Vec<PendingOrderInfo>> {
        if !self.connected {
            bail!("MT5 is not connected.");
        }

        Python::attach(|py| -> PyResult<Vec<PendingOrderInfo>> {
            let mt5 = py.import("MetaTrader5")?;
            let orders = call_mt5_collection(py, &mt5, "orders_get", symbol)?;
            orders
                .try_iter()?
                .map(|item| parse_pending_order_info(&item?))
                .collect()
        })
        .map_err(|e| anyhow::anyhow!("PyError: {}", e))
    }

    pub fn recent_deals(&self, symbol: Option<&str>, lookback_hours: i64) -> Result<Vec<DealInfo>> {
        if !self.connected {
            bail!("MT5 is not connected.");
        }

        Python::attach(|py| -> PyResult<Vec<DealInfo>> {
            let mt5 = py.import("MetaTrader5")?;
            let kwargs = PyDict::new(py);
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system time should be after unix epoch")
                .as_secs() as i64;
            let from_ts = now.saturating_sub(lookback_hours.max(1) * 3600);
            kwargs.set_item("date_from", from_ts)?;
            kwargs.set_item("date_to", now)?;
            if let Some(symbol) = symbol {
                kwargs.set_item("group", symbol)?;
            }
            let deals = mt5.getattr("history_deals_get")?.call((), Some(&kwargs))?;
            let deals = ensure_mt5_collection(&mt5, "history_deals_get", deals)?;
            deals
                .try_iter()?
                .map(|item| parse_deal_info(&item?))
                .collect()
        })
        .map_err(|e| anyhow::anyhow!("PyError: {}", e))
    }

    pub fn historical_bars(
        &self,
        symbol: &str,
        timeframe: &str,
        from_timestamp_s: i64,
        to_timestamp_s: i64,
    ) -> Result<Vec<HistoricalBarInfo>> {
        if !self.connected {
            bail!("MT5 is not connected.");
        }

        Python::attach(|py| -> PyResult<Vec<HistoricalBarInfo>> {
            let mt5 = py.import("MetaTrader5")?;
            let timeframe_value = mt5_timeframe_value(&mt5, timeframe)?;
            let rates = mt5
                .getattr("copy_rates_range")?
                .call1((symbol, timeframe_value, from_timestamp_s, to_timestamp_s))?;
            let rates = ensure_mt5_collection(&mt5, "copy_rates_range", rates)?;
            let rows = if rates.hasattr("tolist")? {
                rates.getattr("tolist")?.call0()?
            } else {
                rates
            };

            rows.try_iter()?
                .map(|item| parse_historical_bar_info(&item?))
                .collect()
        })
        .map_err(|e| anyhow::anyhow!("PyError: {}", e))
    }

    pub fn shutdown(&mut self) {
        if self.connected {
            let _ = Python::attach(|py| -> PyResult<()> {
                if let Ok(mt5) = py.import("MetaTrader5") {
                    let _ = mt5.getattr("shutdown")?.call0();
                    info!("MT5 Connection Shutdown.");
                }
                Ok(())
            });
            self.connected = false;
            record_mt5_event("shutdown", "SUCCESS", "MT5 connection shutdown completed");
        }
    }
}

impl Drop for MT5Engine {
    fn drop(&mut self) { self.shutdown(); }
}

fn call_mt5_collection<'py>(
    py: Python<'py>,
    mt5: &Bound<'py, PyModule>,
    function_name: &str,
    symbol: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let function = mt5.getattr(function_name)?;
    let result = if let Some(symbol) = symbol {
        let kwargs = PyDict::new(py);
        kwargs.set_item("symbol", symbol)?;
        function.call((), Some(&kwargs))?
    } else {
        function.call0()?
    };

    ensure_mt5_collection(mt5, function_name, result)
}

fn ensure_mt5_collection<'py>(
    mt5: &Bound<'py, PyModule>,
    function_name: &str,
    result: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if result.is_none() {
        let err_obj = mt5.getattr("last_error")?.call0()?;
        let err = format_last_error(err_obj.as_any())?;
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{function_name} returned None: {err}"
        )));
    }
    Ok(result)
}

fn parse_position_info(item: &Bound<'_, PyAny>) -> PyResult<PositionInfo> {
    Ok(PositionInfo {
        ticket: attr_i64(item, "ticket")?,
        symbol: attr_string(item, "symbol")?,
        order_side: position_type_name(attr_i64(item, "type")?),
        volume: attr_f64(item, "volume")?,
        price_open: attr_f64(item, "price_open")?,
        price_current: attr_f64(item, "price_current")?,
        profit: attr_f64(item, "profit")?,
        stop_loss: attr_f64(item, "sl")?,
        take_profit: attr_f64(item, "tp")?,
        comment: attr_string(item, "comment")?,
        opened_at: attr_i64(item, "time")?,
    })
}

fn parse_pending_order_info(item: &Bound<'_, PyAny>) -> PyResult<PendingOrderInfo> {
    Ok(PendingOrderInfo {
        ticket: attr_i64(item, "ticket")?,
        symbol: attr_string(item, "symbol")?,
        order_kind: order_type_name(attr_i64(item, "type")?),
        volume_initial: attr_f64(item, "volume_initial")?,
        price_open: attr_f64(item, "price_open")?,
        stop_loss: attr_f64(item, "sl")?,
        take_profit: attr_f64(item, "tp")?,
        comment: attr_string(item, "comment")?,
        created_at: attr_i64(item, "time_setup")?,
    })
}

fn parse_deal_info(item: &Bound<'_, PyAny>) -> PyResult<DealInfo> {
    Ok(DealInfo {
        ticket: attr_i64(item, "ticket")?,
        order_ticket: attr_i64(item, "order")?,
        position_id: attr_i64(item, "position_id")?,
        symbol: attr_string(item, "symbol")?,
        entry_kind: deal_entry_name(attr_i64(item, "entry")?),
        order_side: order_type_name(attr_i64(item, "type")?),
        volume: attr_f64(item, "volume")?,
        price: attr_f64(item, "price")?,
        profit: attr_f64(item, "profit")?,
        fee: attr_f64(item, "fee")?,
        comment: attr_string(item, "comment")?,
        executed_at: attr_i64(item, "time")?,
    })
}

fn parse_historical_bar_info(item: &Bound<'_, PyAny>) -> PyResult<HistoricalBarInfo> {
    if let Ok((time_s, open, high, low, close, tick_volume, spread, real_volume)) =
        item.extract::<(i64, f64, f64, f64, f64, f64, i64, f64)>()
    {
        return Ok(HistoricalBarInfo {
            timestamp_ms: time_s.saturating_mul(1000),
            open,
            high,
            low,
            close,
            tick_volume,
            spread,
            real_volume,
        });
    }

    Ok(HistoricalBarInfo {
        timestamp_ms: attr_i64(item, "time")?.saturating_mul(1000),
        open: attr_f64(item, "open")?,
        high: attr_f64(item, "high")?,
        low: attr_f64(item, "low")?,
        close: attr_f64(item, "close")?,
        tick_volume: attr_f64(item, "tick_volume")?,
        spread: attr_i64(item, "spread")?,
        real_volume: attr_f64(item, "real_volume")?,
    })
}

fn mt5_timeframe_value(mt5: &Bound<'_, PyModule>, timeframe: &str) -> PyResult<Py<PyAny>> {
    let attr = match timeframe.trim().to_ascii_uppercase().as_str() {
        "M1" => "TIMEFRAME_M1",
        "M5" => "TIMEFRAME_M5",
        "M15" => "TIMEFRAME_M15",
        "H1" => "TIMEFRAME_H1",
        "H4" => "TIMEFRAME_H4",
        "D1" => "TIMEFRAME_D1",
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported MT5 timeframe: {other}"
            )))
        }
    };

    Ok(mt5.getattr(attr)?.unbind())
}

fn attr_i64(item: &Bound<'_, PyAny>, attr: &str) -> PyResult<i64> {
    item.getattr(attr)?.extract()
}

fn attr_f64(item: &Bound<'_, PyAny>, attr: &str) -> PyResult<f64> {
    item.getattr(attr)?.extract()
}

fn attr_string(item: &Bound<'_, PyAny>, attr: &str) -> PyResult<String> {
    item.getattr(attr)?.extract()
}

fn position_type_name(type_id: i64) -> String {
    match type_id {
        0 => "BUY".to_string(),
        1 => "SELL".to_string(),
        other => format!("POSITION_TYPE_{other}"),
    }
}

fn order_type_name(type_id: i64) -> String {
    match type_id {
        0 => "BUY".to_string(),
        1 => "SELL".to_string(),
        2 => "BUY_LIMIT".to_string(),
        3 => "SELL_LIMIT".to_string(),
        4 => "BUY_STOP".to_string(),
        5 => "SELL_STOP".to_string(),
        6 => "BUY_STOP_LIMIT".to_string(),
        7 => "SELL_STOP_LIMIT".to_string(),
        8 => "CLOSE_BY".to_string(),
        other => format!("ORDER_TYPE_{other}"),
    }
}

fn deal_entry_name(entry_id: i64) -> String {
    match entry_id {
        0 => "IN".to_string(),
        1 => "OUT".to_string(),
        2 => "INOUT".to_string(),
        3 => "OUT_BY".to_string(),
        other => format!("DEAL_ENTRY_{other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    use std::ffi::CString;

    fn run_py(py: Python<'_>, code: &str, locals: &Bound<'_, PyDict>) -> PyResult<()> {
        let code = CString::new(code).expect("python code should not contain embedded nulls");
        py.run(code.as_c_str(), Some(locals), Some(locals))
    }

    #[test]
    fn test_format_last_error_handles_mt5_tuple_payload() -> Result<()> {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            run_py(py, r#"err = (-6, "Terminal: Authorization failed")"#, &locals)?;
            let err = locals
                .get_item("err")?
                .expect("err should be defined");

            let formatted = format_last_error(err.as_any())?;

            assert_eq!(
                formatted,
                "code=-6 description=Terminal: Authorization failed"
            );
            Ok(())
        })
    }

    #[test]
    fn test_format_last_error_falls_back_to_string_payload() -> Result<()> {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            run_py(py, r#"err = "module unavailable""#, &locals)?;
            let err = locals
                .get_item("err")?
                .expect("err should be defined");

            let formatted = format_last_error(err.as_any())?;

            assert_eq!(formatted, "module unavailable");
            Ok(())
        })
    }

    #[test]
    fn test_mt5_record_targets_mt5_section() {
        let record = mt5_record("initialize", "FAILED", "authorization failed");

        assert_eq!(record.subsystem, forex_core::sectioned_log::SubsystemSection::Mt5);
        assert_eq!(record.operation, "initialize");
        assert_eq!(record.status, "FAILED");
        assert_eq!(record.message, "authorization failed");
    }

    #[test]
    fn test_positions_orders_and_recent_deals_parse_namedtuple_payloads() -> Result<()> {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            run_py(
                py,
                r#"
import sys
import types
from collections import namedtuple

Position = namedtuple("Position", "ticket time type volume price_open price_current sl tp profit symbol comment")
Order = namedtuple("Order", "ticket time_setup type volume_initial price_open sl tp symbol comment")
Deal = namedtuple("Deal", "ticket order position_id time type entry volume price profit fee symbol comment")

mt5 = types.ModuleType("MetaTrader5")
mt5.positions_get = lambda **kwargs: (
    Position(1001, 1710001000, 0, 0.20, 1.1000, 1.1025, 1.0950, 1.1100, 50.0, "EURUSD", "trend"),
)
mt5.orders_get = lambda **kwargs: (
    Order(2001, 1710002000, 2, 0.15, 1.0985, 1.0940, 1.1070, "EURUSD", "breakout"),
)
mt5.history_deals_get = lambda *args, **kwargs: (
    Deal(3001, 2001, 4001, 1710003000, 0, 0, 0.15, 1.0990, 12.5, -0.4, "EURUSD", "filled"),
)
sys.modules["MetaTrader5"] = mt5
"#,
                &locals,
            )?;

            let engine = MT5Engine { connected: true };

            let positions = engine.positions(Some("EURUSD"))?;
            let orders = engine.orders(Some("EURUSD"))?;
            let deals = engine.recent_deals(Some("EURUSD"), 24)?;

            assert_eq!(positions.len(), 1);
            assert_eq!(positions[0].ticket, 1001);
            assert_eq!(positions[0].order_side, "BUY");
            assert_eq!(positions[0].symbol, "EURUSD");
            assert_eq!(positions[0].profit, 50.0);

            assert_eq!(orders.len(), 1);
            assert_eq!(orders[0].ticket, 2001);
            assert_eq!(orders[0].order_kind, "BUY_LIMIT");
            assert_eq!(orders[0].volume_initial, 0.15);

            assert_eq!(deals.len(), 1);
            assert_eq!(deals[0].ticket, 3001);
            assert_eq!(deals[0].order_ticket, 2001);
            assert_eq!(deals[0].position_id, 4001);
            assert_eq!(deals[0].order_side, "BUY");
            assert_eq!(deals[0].entry_kind, "IN");
            assert_eq!(deals[0].profit, 12.5);

            Ok(())
        })
    }

    #[test]
    fn test_historical_bars_parse_copy_rates_range_payload() -> Result<()> {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            run_py(
                py,
                r#"
import sys
import types

class RatesResult(list):
    def tolist(self):
        return list(self)

mt5 = types.ModuleType("MetaTrader5")
mt5.TIMEFRAME_M15 = 15
mt5.copy_rates_range = lambda symbol, timeframe, date_from, date_to: RatesResult([
    (1710000000, 1.1000, 1.1010, 1.0990, 1.1005, 100.0, 2, 50.0),
    (1710000900, 1.1005, 1.1020, 1.1000, 1.1015, 120.0, 3, 55.0),
])
sys.modules["MetaTrader5"] = mt5
"#,
                &locals,
            )?;

            let engine = MT5Engine { connected: true };
            let bars = engine.historical_bars("EURUSD", "M15", 1710000000, 1710001800)?;

            assert_eq!(bars.len(), 2);
            assert_eq!(bars[0].timestamp_ms, 1_710_000_000_000);
            assert_eq!(bars[0].spread, 2);
            assert_eq!(bars[1].tick_volume, 120.0);
            assert_eq!(bars[1].real_volume, 55.0);

            Ok(())
        })
    }

    #[test]
    fn test_historical_bars_require_connected_engine() {
        let engine = MT5Engine { connected: false };

        let err = engine
            .historical_bars("EURUSD", "M1", 1710000000, 1710000600)
            .expect_err("disconnected MT5 engine must fail");

        assert!(err.to_string().contains("not connected"));
    }
}
