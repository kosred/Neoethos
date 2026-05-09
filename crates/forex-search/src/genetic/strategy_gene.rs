use super::runtime_overrides::current_strategy_evaluation_runtime_overrides;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Gene {
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
    pub long_threshold: f32,
    pub short_threshold: f32,
    pub fitness: f64,
    pub sharpe_ratio: f64,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub profit_factor: f64,
    pub expectancy: f64,
    pub trades_count: usize,
    pub generation: usize,
    pub strategy_id: String,
    pub use_ob: bool,
    pub use_fvg: bool,
    pub use_liq_sweep: bool,
    pub mtf_confirmation: bool,
    pub use_premium_discount: bool,
    pub use_inducement: bool,
    #[serde(default)]
    pub use_bos: bool,
    #[serde(default)]
    pub use_choch: bool,
    #[serde(default)]
    pub use_eqh: bool,
    #[serde(default)]
    pub use_eql: bool,
    #[serde(default)]
    pub use_displacement: bool,
    pub tp_pips: f64,
    pub sl_pips: f64,
    pub slice_pass_rate: f64,
    pub consistency: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FilteringConfig {
    pub max_dd: f64,
    pub min_profit: f64,
    pub min_trades: f64,
    pub min_sharpe: f64,
    pub min_win_rate: f64,
    pub min_profit_factor: f64,
    pub min_positive_months: usize,
    pub min_trades_per_month: f64,
    pub min_monthly_return_pct: f64,
    pub log_trades: bool,
    pub trade_log_max: usize,
    pub opportunistic_enabled: bool,
    pub use_opportunistic_candidates: bool,
    pub opportunistic_min_positive_months: usize,
    pub opportunistic_min_trades_per_month: f64,
    pub opportunistic_min_trade_return_pct: f64,
    pub opportunistic_max_dd: f64,
    pub anomaly_guard: bool,
    pub elite_mode: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MarketCostProfile {
    pub symbol: String,
    pub account_currency: String,
    pub pip_value: f64,
    pub pip_value_per_lot: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
}

impl Default for FilteringConfig {
    fn default() -> Self {
        Self {
            max_dd: 0.15,
            min_profit: 10.0,
            min_trades: 10.0,
            min_sharpe: 0.3,
            min_win_rate: 0.50,
            min_profit_factor: 1.05,
            min_positive_months: 0,
            min_trades_per_month: 0.0,
            min_monthly_return_pct: 0.0,
            log_trades: false,
            trade_log_max: 20,
            opportunistic_enabled: false,
            use_opportunistic_candidates: false,
            opportunistic_min_positive_months: 0,
            opportunistic_min_trades_per_month: 0.0,
            opportunistic_min_trade_return_pct: 0.0,
            opportunistic_max_dd: 1.0,
            anomaly_guard: true,
            elite_mode: false,
        }
    }
}

fn normalized_symbol(symbol: &str) -> String {
    symbol
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
}

fn split_symbol_parts(symbol: &str) -> Option<(String, String)> {
    let normalized = normalized_symbol(symbol);
    if normalized.len() >= 6 {
        Some((normalized[..3].to_string(), normalized[3..6].to_string()))
    } else {
        None
    }
}

fn symbol_kind(symbol: &str) -> &'static str {
    let normalized = normalized_symbol(symbol);
    if normalized.starts_with("XAU") || normalized.starts_with("XAG") {
        return "metal";
    }
    if normalized.contains("BTC") || normalized.contains("ETH") || normalized.contains("LTC") {
        return "crypto";
    }
    if split_symbol_parts(&normalized).is_some() {
        return "fx";
    }
    "other"
}

fn default_pip_size(symbol: &str) -> f64 {
    match symbol_kind(symbol) {
        "metal" => 0.01,
        "crypto" => 1.0,
        "fx" => match split_symbol_parts(symbol) {
            Some((_base, quote)) if quote == "JPY" => 0.01,
            _ => 0.0001,
        },
        _ => 0.0001,
    }
}

fn default_contract_size(symbol: &str) -> f64 {
    match symbol_kind(symbol) {
        "metal" => match split_symbol_parts(symbol) {
            Some((base, _quote)) if base == "XAG" => 5_000.0,
            Some((base, _quote)) if base == "XAU" => 100.0,
            _ => 100.0,
        },
        "crypto" => 1.0,
        "fx" => 100_000.0,
        _ => 1.0,
    }
}

/// Convert one pip on `symbol` into account-currency units per standard lot.
///
/// `quote_to_account_rate` is the live `quote_currency → account_currency` FX
/// rate (e.g. for EURGBP on a USD account this is the GBP→USD rate ≈ 1.27).
/// When `None`, we fall back to a price-hint-only model that is correct for
/// pairs where account currency is base or quote, and approximate (but
/// flagged via `tracing::warn`) for cross pairs. Set the env override
/// `FOREX_BOT_PROP_PIP_VALUE_PER_LOT` to bypass this entirely when the
/// caller already knows the value (e.g. straight from cTrader metadata).
fn estimate_pip_value_per_lot(
    symbol: &str,
    account_currency: &str,
    price_hint: Option<f64>,
    quote_to_account_rate: Option<f64>,
) -> f64 {
    let pip_size = default_pip_size(symbol);
    let contract_size = default_contract_size(symbol);
    let pip_value_quote = pip_size * contract_size;
    let account_currency = account_currency.trim().to_ascii_uppercase();
    let normalized = normalized_symbol(symbol);
    let price = price_hint.filter(|value| value.is_finite() && *value > 0.0);
    let conv_rate = quote_to_account_rate.filter(|v| v.is_finite() && *v > 0.0);

    if let Some((base, quote)) = split_symbol_parts(&normalized) {
        if quote == account_currency {
            // pip is already in account currency
            return pip_value_quote.max(1e-6);
        }
        if base == account_currency {
            // For e.g. USDJPY on a USD account: pip_value_USD = pip_value_JPY / USDJPY
            return price
                .map(|value| pip_value_quote / value.max(1e-9))
                .unwrap_or_else(|| pip_value_quote.max(1e-6));
        }
        // Cross pair (e.g. EURGBP on USD account). Correct formula needs the
        // quote→account FX rate. If supplied, use it. If not, fall back to
        // pip_value_quote (i.e. the pip in QUOTE currency) and warn — this
        // matches the previous (silently wrong) behaviour for the symbol's
        // own price, but flags the gap so callers can plug in a real rate.
        if let Some(rate) = conv_rate {
            return (pip_value_quote * rate).max(1e-6);
        }
        tracing::warn!(
            target: "forex_search::cost_model",
            symbol,
            account_currency = %account_currency,
            "cross-pair pip_value_per_lot estimated without quote→account FX rate; \
             set FOREX_BOT_PROP_PIP_VALUE_PER_LOT to override"
        );
        return pip_value_quote.max(1e-6);
    }

    pip_value_quote.max(1e-6)
}

pub fn infer_market_cost_profile(
    symbol: &str,
    account_currency: &str,
    price_hint: Option<f64>,
    spread_pips_override: Option<f64>,
    commission_override: Option<f64>,
) -> MarketCostProfile {
    // Cost-profile fallbacks resolve through the typed
    // `CostProfileRuntimeOverrides` boundary so the legacy
    // `FOREX_BOT_PROP_*` env vars are read at most once per process.
    let runtime_overrides = current_strategy_evaluation_runtime_overrides();
    let cost = runtime_overrides.cost_profile;

    let symbol = if symbol.trim().is_empty() {
        cost.symbol.clone().unwrap_or_else(|| "EURUSD".to_string())
    } else {
        symbol.trim().to_string()
    };
    let account_currency = if account_currency.trim().is_empty() {
        cost.account_currency
            .clone()
            .unwrap_or_else(|| "USD".to_string())
    } else {
        account_currency.trim().to_string()
    };

    let pip_value = cost.pip_value.unwrap_or_else(|| default_pip_size(&symbol));
    let pip_value_per_lot = cost.pip_value_per_lot.unwrap_or_else(|| {
        estimate_pip_value_per_lot(
            &symbol,
            &account_currency,
            price_hint,
            cost.quote_to_account_rate,
        )
    });

    let spread_pips = spread_pips_override
        .filter(|value| value.is_finite() && *value >= 0.0)
        .or(cost.spread_pips)
        .unwrap_or_else(|| match symbol_kind(&symbol) {
            "metal" => 2.5,
            "crypto" => 8.0,
            "fx" => 1.5,
            _ => 1.0,
        });
    let commission_per_trade = commission_override
        .filter(|value| value.is_finite() && *value >= 0.0)
        .or(cost.commission_per_trade)
        .unwrap_or(7.0);

    MarketCostProfile {
        symbol,
        account_currency,
        pip_value,
        pip_value_per_lot,
        spread_pips,
        commission_per_trade,
    }
}

impl Gene {
    pub fn is_anomalous(&self) -> bool {
        let trades = self.trades_count as f64;
        let win_rate = self.win_rate;
        let profit_factor = self.profit_factor;
        let profit = self.fitness; // Using fitness as profit proxy
        let dd = self.max_drawdown;
        let ppt = if trades > 0.0 { profit / trades } else { 0.0 };

        // Thresholds from evo_prop.py
        let min_trades = 120.0;
        let max_dd = 0.0025;
        let min_win_rate = 0.92;
        let min_pf = 12.0;
        let min_profit = 200_000.0;
        let max_ppt = 2_000.0;

        let suspicious_combo = trades >= min_trades
            && dd <= max_dd
            && win_rate >= min_win_rate
            && profit_factor >= min_pf
            && profit >= min_profit;

        let suspicious_ppt = trades >= 40.0 && dd <= 0.01 && ppt >= max_ppt;

        let suspicious_ultra =
            trades >= 50.0 && dd <= 0.001 && profit >= 150_000.0 && ppt >= 1_000.0;

        let suspicious_low_dd = trades >= 80.0 && dd <= 0.001 && profit >= 50_000.0;

        suspicious_combo || suspicious_ppt || suspicious_ultra || suspicious_low_dd
    }

    pub fn passes_filter(&self, cfg: &FilteringConfig) -> bool {
        if self.fitness <= cfg.min_profit {
            return false;
        }
        if self.max_drawdown > cfg.max_dd {
            return false;
        }
        if (self.trades_count as f64) < cfg.min_trades {
            return false;
        }
        if self.sharpe_ratio < cfg.min_sharpe {
            return false;
        }
        if self.win_rate < cfg.min_win_rate {
            return false;
        }
        if self.profit_factor < cfg.min_profit_factor {
            return false;
        }
        if cfg.anomaly_guard && self.is_anomalous() {
            return false;
        }
        true
    }

    pub fn requires_quality_screen(cfg: &FilteringConfig) -> bool {
        cfg.min_positive_months > 0
            || cfg.min_trades_per_month > 0.0
            || cfg.min_monthly_return_pct > 0.0
            || cfg.log_trades
            || (cfg.opportunistic_enabled && cfg.use_opportunistic_candidates)
    }

    pub fn normalize(&mut self, n_indicators: usize, min_indicators: usize) {
        let n_indicators = n_indicators.max(1);
        let min_indicators = min_indicators.clamp(1, n_indicators);

        if self.indices.is_empty() {
            self.indices.push(0);
        }
        if self.weights.len() != self.indices.len() {
            self.weights = vec![1.0; self.indices.len()];
        }

        let mut terms: Vec<(usize, f32)> = self
            .indices
            .iter()
            .copied()
            .zip(self.weights.iter().copied())
            .map(|(idx, weight)| {
                let idx = idx % n_indicators;
                let weight = if weight.is_finite() { weight } else { 0.0 };
                (idx, weight)
            })
            .collect();
        terms.sort_by_key(|(idx, _)| *idx);

        let mut merged: Vec<(usize, f32)> = Vec::with_capacity(terms.len());
        for (idx, weight) in terms {
            if let Some((last_idx, last_weight)) = merged.last_mut() {
                if *last_idx == idx {
                    *last_weight += weight;
                    continue;
                }
            }
            merged.push((idx, weight));
        }

        if merged.len() > min_indicators {
            merged.retain(|(_, weight)| weight.abs() > 1e-6);
        }
        if merged.is_empty() {
            merged.push((0, 1.0));
        }

        let mut rng = rand::rng();
        let mut seen: HashSet<usize> = merged.iter().map(|(idx, _)| *idx).collect();
        while merged.len() < min_indicators {
            let idx = rng.random_range(0..n_indicators);
            if seen.insert(idx) {
                merged.push((idx, rng.random_range(0.1..1.0)));
            }
        }
        merged.sort_by_key(|(idx, _)| *idx);

        self.indices = merged.iter().map(|(idx, _)| *idx).collect();
        self.weights = merged
            .iter()
            .map(|(_, weight)| {
                if weight.is_finite() && weight.abs() > 1e-6 {
                    weight.clamp(-5.0, 5.0)
                } else {
                    1.0
                }
            })
            .collect();

        if !self.long_threshold.is_finite() {
            self.long_threshold = 0.25;
        }
        if !self.short_threshold.is_finite() {
            self.short_threshold = -0.25;
        }
        if self.long_threshold <= self.short_threshold {
            let mid = (self.long_threshold + self.short_threshold) * 0.5;
            self.long_threshold = mid + 0.05;
            self.short_threshold = mid - 0.05;
        }
        if !self.tp_pips.is_finite() || self.tp_pips <= 0.0 {
            self.tp_pips = 40.0;
        }
        if !self.sl_pips.is_finite() || self.sl_pips <= 0.0 {
            self.sl_pips = 20.0;
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub genes: Vec<Gene>,
    pub metrics: Vec<[f64; 11]>,
}

#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub symbol: String,
    pub account_currency: String,
    pub max_hold_bars: usize,
    pub trailing_enabled: bool,
    pub trailing_atr_multiplier: f64,
    pub trailing_be_trigger_r: f64,
    pub pip_value: f64,
    pub spread_pips: f64,
    pub commission_per_trade: f64,
    pub pip_value_per_lot: f64,
    pub smc_gate_threshold: f32,
    pub smc_weight_ob: f32,
    pub smc_weight_fvg: f32,
    pub smc_weight_liq: f32,
    pub smc_weight_mtf: f32,
    pub smc_weight_premium: f32,
    pub smc_weight_inducement: f32,
    pub smc_weight_bos: f32,
    pub smc_weight_choch: f32,
    pub smc_weight_eqh: f32,
    pub smc_weight_eql: f32,
    pub smc_weight_displacement: f32,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        // The cost profile already consults the typed
        // `StrategyEvaluationRuntimeOverrides` boundary, so the four cost
        // fields below come straight from the resolved profile rather
        // than re-reading the env. SMC weights come from the typed SMC
        // weight overrides on the same boundary.
        let profile = infer_market_cost_profile("", "", None, None, None);
        let smc = current_strategy_evaluation_runtime_overrides().smc_weights;

        Self {
            symbol: profile.symbol,
            account_currency: profile.account_currency,
            max_hold_bars: 0,
            trailing_enabled: false,
            trailing_atr_multiplier: 1.0,
            trailing_be_trigger_r: 1.0,
            pip_value: profile.pip_value,
            spread_pips: profile.spread_pips,
            commission_per_trade: profile.commission_per_trade,
            pip_value_per_lot: profile.pip_value_per_lot,
            smc_gate_threshold: smc.gate_threshold,
            smc_weight_ob: smc.w_ob,
            smc_weight_fvg: smc.w_fvg,
            smc_weight_liq: smc.w_liq,
            smc_weight_mtf: smc.w_mtf,
            smc_weight_premium: smc.w_premium,
            smc_weight_inducement: smc.w_inducement,
            smc_weight_bos: smc.w_bos,
            smc_weight_choch: smc.w_choch,
            smc_weight_eqh: smc.w_eqh,
            smc_weight_eql: smc.w_eql,
            smc_weight_displacement: smc.w_displacement,
        }
    }
}

impl EvaluationConfig {
    pub fn for_symbol(
        symbol: &str,
        account_currency: &str,
        price_hint: Option<f64>,
        spread_pips_override: Option<f64>,
        commission_override: Option<f64>,
    ) -> Self {
        let profile = infer_market_cost_profile(
            symbol,
            account_currency,
            price_hint,
            spread_pips_override,
            commission_override,
        );
        Self {
            symbol: profile.symbol,
            account_currency: profile.account_currency,
            pip_value: profile.pip_value,
            pip_value_per_lot: profile.pip_value_per_lot,
            spread_pips: profile.spread_pips,
            commission_per_trade: profile.commission_per_trade,
            ..Self::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_canonicalizes_duplicate_indicator_terms() {
        let mut gene = Gene {
            indices: vec![4, 1, 4, 9],
            weights: vec![0.25, 1.0, 0.75, 0.5],
            long_threshold: 0.4,
            short_threshold: -0.4,
            tp_pips: 40.0,
            sl_pips: 20.0,
            ..Default::default()
        };

        gene.normalize(5, 1);

        assert_eq!(gene.indices, vec![1, 4]);
        assert_eq!(gene.weights, vec![1.0, 1.5]);
    }

    #[test]
    fn normalize_repairs_invalid_numeric_fields() {
        let mut gene = Gene {
            indices: vec![0],
            weights: vec![f32::NAN],
            long_threshold: f32::NAN,
            short_threshold: f32::NAN,
            tp_pips: f64::NAN,
            sl_pips: -1.0,
            ..Default::default()
        };

        gene.normalize(3, 1);

        assert_eq!(gene.weights, vec![1.0]);
        assert!(gene.long_threshold > gene.short_threshold);
        assert_eq!(gene.tp_pips, 40.0);
        assert_eq!(gene.sl_pips, 20.0);
    }
}
