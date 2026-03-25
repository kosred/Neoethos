use std::collections::{HashMap, VecDeque};
use chrono::{NaiveDate, DateTime, Utc, Duration};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyMetrics {
    pub score: f64,
    pub daily_profit_consistency: f64,
    pub daily_trade_consistency: f64,
    pub daily_risk_consistency: f64,
    pub weekly_profit_consistency: f64,
    pub weekly_drawdown_consistency: f64,
    pub trade_size_consistency: f64,
    pub hold_time_consistency: f64,
    pub win_rate_rolling: f64,
    pub grade: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeEvent {
    pub entry_time: String, // ISO format
    pub pnl: f64,
    pub risk_pct: f64,
    pub size: f64,
    pub hold_minutes: f64,
    pub win: Option<i32>,
}

pub struct ConsistencyTracker {
    lookback_days: i64,
    daily_pnl: HashMap<NaiveDate, f64>,
    daily_trades: HashMap<NaiveDate, i64>,
    daily_risk: HashMap<NaiveDate, f64>,
    max_hist: usize,
    trade_sizes: VecDeque<f64>,
    hold_times: VecDeque<f64>,
    trade_outcomes: VecDeque<i32>, // 1 win, 0 loss
}

impl ConsistencyTracker {
    pub fn new(lookback_days: i64) -> Self {
        Self {
            lookback_days,
            daily_pnl: HashMap::new(),
            daily_trades: HashMap::new(),
            daily_risk: HashMap::new(),
            max_hist: 500,
            trade_sizes: VecDeque::with_capacity(500),
            hold_times: VecDeque::with_capacity(500),
            trade_outcomes: VecDeque::with_capacity(500),
        }
    }

    pub fn update(&mut self, event: &TradeEvent) {
        let ts = match DateTime::parse_from_rfc3339(&event.entry_time) {
            Ok(dt) => dt.with_timezone(&Utc),
            Err(_) => {
                match event.entry_time.parse::<DateTime<Utc>>() {
                    Ok(dt) => dt,
                    Err(_) => {
                        tracing::warn!("ConsistencyTracker dropped trade with invalid entry_time: {}", event.entry_time);
                        return;
                    }
                }
            }
        };
        
        let d = ts.date_naive();
        let pnl = event.pnl;
        let risk_pct = event.risk_pct;
        let size = event.size;
        let hold_minutes = event.hold_minutes;
        let win_flag = event.win.unwrap_or(if pnl > 0.0 { 1 } else { 0 });

        *self.daily_pnl.entry(d).or_insert(0.0) += pnl;
        *self.daily_trades.entry(d).or_insert(0) += 1;
        *self.daily_risk.entry(d).or_insert(0.0) += risk_pct;

        if self.trade_sizes.len() >= self.max_hist { self.trade_sizes.pop_front(); }
        self.trade_sizes.push_back(size);

        if self.hold_times.len() >= self.max_hist { self.hold_times.pop_front(); }
        self.hold_times.push_back(hold_minutes);

        if self.trade_outcomes.len() >= self.max_hist { self.trade_outcomes.pop_front(); }
        self.trade_outcomes.push_back(win_flag);

        let cutoff_date = d - Duration::days(self.lookback_days);

        self.daily_pnl.retain(|&k, _| k >= cutoff_date);
        self.daily_trades.retain(|&k, _| k >= cutoff_date);
        self.daily_risk.retain(|&k, _| k >= cutoff_date);
    }

    pub fn get_metrics(&self) -> ConsistencyMetrics {
        if self.daily_pnl.is_empty() {
            return ConsistencyMetrics {
                score: 0.0,
                daily_profit_consistency: 0.0,
                daily_trade_consistency: 0.0,
                daily_risk_consistency: 0.0,
                weekly_profit_consistency: 0.0,
                weekly_drawdown_consistency: 0.0,
                trade_size_consistency: 0.0,
                hold_time_consistency: 0.0,
                win_rate_rolling: 0.0,
                grade: "F".to_string(),
            };
        }

        let mut days: Vec<NaiveDate> = self.daily_pnl.keys().cloned().collect();
        days.sort();
        let start_idx = days.len().saturating_sub(self.lookback_days as usize);
        let recent_days = &days[start_idx..];

        let pnls: Vec<f64> = recent_days.iter().map(|d| *self.daily_pnl.get(d).unwrap_or(&0.0)).collect();
        let daily_profit_consistency = if !pnls.is_empty() {
            pnls.iter().filter(|&&p| p > 0.0).count() as f64 / pnls.len().max(1) as f64
        } else {
            0.0
        };

        let trades: Vec<f64> = recent_days.iter().map(|d| *self.daily_trades.get(d).unwrap_or(&0) as f64).collect();
        let trade_var = variance(&trades);
        let daily_trade_consistency = 1.0 / (1.0 + trade_var);

        let risks: Vec<f64> = recent_days.iter().map(|d| *self.daily_risk.get(d).unwrap_or(&0.0)).collect();
        let risk_var = variance(&risks);
        let daily_risk_consistency = 1.0 / (1.0 + risk_var);

        // Weekly chunks (pseudo-weekly based on trading days)
        let mut weekly = Vec::new();
        for chunk in pnls.chunks(5) {
            weekly.push(chunk.iter().sum::<f64>());
        }
        let weekly_profit_consistency = if !weekly.is_empty() {
            weekly.iter().filter(|&&w| w > 0.0).count() as f64 / weekly.len().max(1) as f64
        } else {
            0.0
        };
        let weekly_dd_consistency = if weekly.len() > 1 {
            1.0 / (1.0 + std_dev(&weekly))
        } else {
            1.0
        };

        let size_var = if self.trade_sizes.len() > 1 { variance(&self.trade_sizes.iter().copied().collect::<Vec<f64>>()) } else { 0.0 };
        let trade_size_consistency = 1.0 / (1.0 + size_var);

        let hold_var = if self.hold_times.len() > 1 { variance(&self.hold_times.iter().copied().collect::<Vec<f64>>()) } else { 0.0 };
        let hold_time_consistency = 1.0 / (1.0 + hold_var);

        let win_rate = if !self.trade_outcomes.is_empty() {
            let n = self.trade_outcomes.len().min(30);
            let recent: f64 = self.trade_outcomes.iter().skip(self.trade_outcomes.len() - n).map(|&v| v as f64).sum();
            recent / (n as f64)
        } else {
            0.0
        };

        let score = (
            0.25 * daily_profit_consistency
            + 0.2 * daily_trade_consistency
            + 0.15 * daily_risk_consistency
            + 0.1 * weekly_profit_consistency
            + 0.1 * weekly_dd_consistency
            + 0.1 * trade_size_consistency
            + 0.05 * hold_time_consistency
            + 0.05 * win_rate
        ) * 100.0;

        let grade = if score >= 90.0 {
            "A+"
        } else if score >= 80.0 {
            "A"
        } else if score >= 70.0 {
            "B"
        } else if score >= 60.0 {
            "C"
        } else if score >= 50.0 {
            "D"
        } else {
            "F"
        };


        ConsistencyMetrics {
            score,
            daily_profit_consistency,
            daily_trade_consistency,
            daily_risk_consistency,
            weekly_profit_consistency,
            weekly_drawdown_consistency: weekly_dd_consistency,
            trade_size_consistency,
            hold_time_consistency,
            win_rate_rolling: win_rate,
            grade: grade.to_string(),
        }
    }
}

fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|&value| {
        let diff = mean - value;
        diff * diff
    }).sum::<f64>() / (data.len() as f64 - 1.0)
}

fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}
