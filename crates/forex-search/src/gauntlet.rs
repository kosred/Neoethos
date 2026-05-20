use crate::eval::BacktestSettings;
use crate::genetic::{Gene, month_day_indices, signals_for_gene};
use forex_core::domain::prop_firm::PropFirmConstraints;
use forex_data::{FeatureFrame, Ohlcv};

/// Default strategy quality floor: minimum win rate over the backtest
/// to clear the gauntlet. NOT a prop firm constraint — this is an
/// internal-tunable quality threshold per operator directive
/// 2026-05-14 (only prop firm DD and 4% monthly target are sacred).
/// Production training plans MAY override via `GauntletConfig`.
pub const DEFAULT_MIN_WIN_RATE: f64 = 0.55;

/// Default strategy quality floor: minimum profit factor (gross win /
/// gross loss) to clear the gauntlet. Internal tunable.
pub const DEFAULT_MIN_PROFIT_FACTOR: f64 = 1.2;

/// Default internal trailing total-drawdown cap. Deliberately BELOW
/// the FTMO ceiling
/// (`PropFirmConstraints::FTMO_STANDARD.max_overall_drawdown_pct = 0.10`)
/// so live trading exits the strategy well before the prop firm
/// boundary is touched. Internal tunable.
pub const DEFAULT_MAX_DRAWDOWN_PCT: f64 = 0.07;

/// Default internal daily-drawdown cap. Deliberately BELOW the FTMO
/// daily ceiling
/// (`PropFirmConstraints::FTMO_STANDARD.max_daily_loss_pct = 0.05`)
/// for the same reason as `DEFAULT_MAX_DRAWDOWN_PCT`. Internal tunable.
pub const DEFAULT_MAX_DAILY_DD: f64 = 0.04;

#[derive(Debug, Clone)]
pub struct GauntletConfig {
    pub min_win_rate: f64,
    pub min_profit_factor: f64,
    pub max_drawdown_pct: f64,
    pub max_daily_dd: f64,
    pub warn_only: bool,
    pub backtest: BacktestSettings,
}

impl Default for GauntletConfig {
    fn default() -> Self {
        // Sanity-check that operator-tier defaults stay BELOW the
        // prop firm ceilings; if FTMO_STANDARD ever changes upward,
        // this debug_assert catches the inversion before live trading.
        let ftmo = PropFirmConstraints::FTMO_STANDARD;
        debug_assert!(
            DEFAULT_MAX_DRAWDOWN_PCT < ftmo.max_overall_drawdown_pct as f64,
            "gauntlet total-DD cap must stay below prop firm ceiling"
        );
        debug_assert!(
            DEFAULT_MAX_DAILY_DD < ftmo.max_daily_loss_pct as f64,
            "gauntlet daily-DD cap must stay below prop firm ceiling"
        );
        Self {
            min_win_rate: DEFAULT_MIN_WIN_RATE,
            min_profit_factor: DEFAULT_MIN_PROFIT_FACTOR,
            max_drawdown_pct: DEFAULT_MAX_DRAWDOWN_PCT,
            max_daily_dd: DEFAULT_MAX_DAILY_DD,
            warn_only: false,
            backtest: BacktestSettings::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StrategyGauntlet {
    pub config: GauntletConfig,
}

impl StrategyGauntlet {
    pub fn new(config: Option<GauntletConfig>) -> Self {
        Self {
            config: config.unwrap_or_default(),
        }
    }

    pub fn run(&self, features: &FeatureFrame, ohlcv: &Ohlcv, gene: &Gene) -> bool {
        if features.data.nrows() == 0 {
            return false;
        }
        if ohlcv.close.len() != features.data.nrows() {
            return false;
        }

        let signals = signals_for_gene(features, gene);
        if signals.is_empty() || signals.len() != ohlcv.close.len() {
            return false;
        }
        let (months, days) = month_day_indices(&features.timestamps);

        let mut settings = self.config.backtest.clone();
        settings.sl_pips = gene.sl_pips;
        settings.tp_pips = gene.tp_pips;

        let metrics = crate::eval::fast_evaluate_strategy_core(
            &ohlcv.close,
            &ohlcv.high,
            &ohlcv.low,
            &signals,
            &months,
            &days,
            &features.timestamps,
            &settings,
        );

        let net_profit = metrics[0];
        let max_dd = metrics[3];
        let win_rate = metrics[4];
        let profit_factor = metrics[5];
        let max_daily_dd = metrics[10];

        let mut failures: Vec<String> = Vec::new();
        if win_rate < self.config.min_win_rate {
            failures.push(format!(
                "win_rate {:.3} < {:.3}",
                win_rate, self.config.min_win_rate
            ));
        }
        if max_dd > self.config.max_drawdown_pct {
            failures.push(format!(
                "max_dd {:.3} > {:.3}",
                max_dd, self.config.max_drawdown_pct
            ));
        }
        if max_daily_dd > self.config.max_daily_dd {
            failures.push(format!(
                "max_daily_dd {:.3} > {:.3}",
                max_daily_dd, self.config.max_daily_dd
            ));
        }
        if profit_factor <= self.config.min_profit_factor {
            failures.push(format!(
                "profit_factor {:.3} <= {:.3}",
                profit_factor, self.config.min_profit_factor
            ));
        }
        if net_profit <= 0.0 {
            failures.push(format!("net_profit {:.2} <= 0.0", net_profit));
        }
        if failures.is_empty() {
            return true;
        }
        // Previously the function silently returned `warn_only` here without
        // surfacing WHICH metric failed, hiding bad strategies in warn-only
        // mode. Always emit a structured warn so operators can audit.
        tracing::warn!(
            target: "forex_search::gauntlet",
            warn_only = self.config.warn_only,
            failures = failures.join("; "),
            "strategy gauntlet failed"
        );
        self.config.warn_only
    }
}
