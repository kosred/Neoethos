//! Symbol metadata — the broker-authoritative source for pip size,
//! contract size, lot constraints and quote-conversion rates.
//!
//! Trading systems are notoriously unforgiving about hardcoded
//! per-symbol constants. JPY pairs use a 0.01 pip; metals use 0.01 too
//! but with different lot sizes; indices and crypto each have their
//! own contract conventions. The previous heuristic (`split_symbol_parts`
//! + `if quote == "JPY"`) was right for the common cases but couldn't
//! tell EURJPY's *quote-conversion rate to USD* — that rate is what
//! turns "JPY pips per lot" into "USD per lot", and for cross pairs
//! you need real broker data to get it right.
//!
//! This module is the typed boundary:
//!
//! - A `SymbolMetadata` struct carries everything pip-math needs.
//! - A `SymbolMetadataTable` loads from disk (`data/symbol_metadata.json`)
//!   so operators can override anything, and so the cTrader connector
//!   has a write-target after fetching `ProtoOASymbol` records.
//! - `lookup` falls back to baked-in defaults for the major symbols
//!   (FX 7-majors, JPY pairs, XAU/XAG) when disk has nothing — so the
//!   system is usable from a fresh checkout without a broker.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Authoritative per-symbol trading constants. All fields use the
/// broker convention (cTrader Open API field names where possible).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SymbolMetadata {
    /// Canonical symbol — uppercase, no separators. e.g. "EURUSD".
    pub symbol: String,
    /// Base currency / instrument code (3 chars usually).
    pub base: String,
    /// Quote currency (3 chars usually).
    pub quote: String,
    /// One pip in price terms. EURUSD = 0.0001, USDJPY = 0.01,
    /// XAUUSD = 0.01, XAGUSD = 0.001.
    pub pip_size: f64,
    /// Units of base currency per standard lot. FX = 100_000,
    /// XAU = 100, XAG = 5_000, BTC = 1.
    pub contract_size: f64,
    /// Pip value in QUOTE currency per standard lot. Pre-computed
    /// (= pip_size * contract_size) so callers don't recompute.
    pub pip_value_quote: f64,
    /// Number of decimal digits the broker quotes to. cTrader's
    /// `digits` field. Used for tick rounding.
    pub digits: u32,
    /// Minimum lot size (broker rule). 0.01 typical for FX.
    pub min_lot: f64,
    /// Maximum lot size.
    pub max_lot: f64,
    /// Lot step (granularity). 0.01 typical.
    pub lot_step: f64,
    /// Approximate spot price hint for quick conversion math
    /// (especially for JPY pairs and metals where pip math depends
    /// on price). Not authoritative — use a live quote when you have
    /// one. None for symbols with no obvious "typical" value.
    pub typical_price: Option<f64>,
}

impl SymbolMetadata {
    /// Convert `pip_value_quote` into the account currency. For
    /// pairs where `quote == account_currency` the conversion is
    /// the identity. For `base == account_currency` the pip value
    /// in account-ccy is `pip_value_quote / spot_price`. For cross
    /// pairs you need a live quote→account FX rate from the broker
    /// (passed in as `quote_to_account_rate`); the heuristic
    /// fallback returns NaN so downstream PnL collapses visibly
    /// rather than silently shipping wrong sizing.
    pub fn pip_value_in_account(
        &self,
        account_currency: &str,
        quote_to_account_rate: Option<f64>,
        live_price: Option<f64>,
    ) -> f64 {
        let acct = account_currency.trim().to_ascii_uppercase();
        if self.quote == acct {
            return self.pip_value_quote;
        }
        if self.base == acct {
            let price = live_price
                .or(self.typical_price)
                .filter(|v| v.is_finite() && *v > 0.0);
            return price
                .map(|v| self.pip_value_quote / v)
                .unwrap_or(self.pip_value_quote);
        }
        if let Some(rate) = quote_to_account_rate.filter(|v| v.is_finite() && *v > 0.0) {
            return self.pip_value_quote * rate;
        }
        f64::NAN
    }
}

/// Disk-backed table. Loaded once per process; subsequent lookups are
/// in-memory. Reload with `reload()` after the cTrader connector
/// writes new entries.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SymbolMetadataTable {
    /// Map keyed by canonical (uppercase, no-separator) symbol.
    pub entries: HashMap<String, SymbolMetadata>,
}

impl SymbolMetadataTable {
    pub fn load_from_disk(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read symbol metadata at {}", path.display()))?;
        let table: SymbolMetadataTable = serde_json::from_str(&text)
            .with_context(|| format!("parse symbol metadata at {}", path.display()))?;
        Ok(table)
    }

    pub fn save_to_disk(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        let text = serde_json::to_string_pretty(self)
            .context("serialize symbol metadata")?;
        std::fs::write(path, text)
            .with_context(|| format!("write symbol metadata to {}", path.display()))
    }

    /// Look up by symbol — case- and separator-insensitive.
    pub fn lookup(&self, symbol: &str) -> Option<&SymbolMetadata> {
        self.entries.get(&canonical_symbol(symbol))
    }

    /// Insert / replace. Used by the cTrader connector after fetching
    /// `ProtoOASymbol` records.
    pub fn upsert(&mut self, meta: SymbolMetadata) {
        let key = canonical_symbol(&meta.symbol);
        self.entries.insert(key, meta);
    }
}

pub fn canonical_symbol(symbol: &str) -> String {
    symbol
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
}

/// Process-wide cache of the on-disk metadata. Layered on top of the
/// baked-in defaults: `lookup` checks disk first, then baked-in.
static GLOBAL_TABLE: OnceLock<SymbolMetadataTable> = OnceLock::new();

/// Resolve the canonical metadata file path. Operators can override
/// via `FOREX_BOT_SYMBOL_METADATA` env. Default: `data/symbol_metadata.json`
/// relative to CWD.
pub fn metadata_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("FOREX_BOT_SYMBOL_METADATA") {
        return std::path::PathBuf::from(p);
    }
    std::path::PathBuf::from("data").join("symbol_metadata.json")
}

/// Load (and cache) the on-disk metadata table. Returns the empty
/// table if no file exists — callers should fall back to
/// [`baked_in_default`] when `lookup` returns None.
pub fn global_table() -> &'static SymbolMetadataTable {
    GLOBAL_TABLE.get_or_init(|| {
        let path = metadata_path();
        if path.exists() {
            match SymbolMetadataTable::load_from_disk(&path) {
                Ok(t) => {
                    tracing::info!(
                        target: "forex_core::symbol_metadata",
                        path = %path.display(),
                        entries = t.entries.len(),
                        "loaded on-disk symbol metadata"
                    );
                    t
                }
                Err(err) => {
                    tracing::warn!(
                        target: "forex_core::symbol_metadata",
                        path = %path.display(),
                        error = %err,
                        "failed to load symbol metadata; using baked-in defaults only"
                    );
                    SymbolMetadataTable::default()
                }
            }
        } else {
            SymbolMetadataTable::default()
        }
    })
}

/// One-shot resolver: disk first (operator/cTrader-supplied), then
/// baked-in default for the majors. Returns `None` only for symbols
/// the system has no information about at all.
pub fn resolve(symbol: &str) -> Option<SymbolMetadata> {
    if let Some(m) = global_table().lookup(symbol) {
        return Some(m.clone());
    }
    baked_in_default(symbol)
}

/// Hand-curated metadata for the symbols every prop-firm operator
/// trades. Numbers verified against cTrader / TradingView spec sheets.
/// This is the floor — disk overrides win.
pub fn baked_in_default(symbol: &str) -> Option<SymbolMetadata> {
    let canon = canonical_symbol(symbol);
    let m = match canon.as_str() {
        // ── FX majors, USD-quoted (5-digit) ──
        "EURUSD" => fx(canon, "EUR", "USD", 5, Some(1.10)),
        "GBPUSD" => fx(canon, "GBP", "USD", 5, Some(1.27)),
        "AUDUSD" => fx(canon, "AUD", "USD", 5, Some(0.66)),
        "NZDUSD" => fx(canon, "NZD", "USD", 5, Some(0.60)),
        // ── FX majors, USD-base ──
        "USDCAD" => fx(canon, "USD", "CAD", 5, Some(1.36)),
        "USDCHF" => fx(canon, "USD", "CHF", 5, Some(0.90)),
        "USDJPY" => fx_jpy(canon, "USD", "JPY", Some(150.0)),
        // ── JPY crosses ──
        "EURJPY" => fx_jpy(canon, "EUR", "JPY", Some(165.0)),
        "GBPJPY" => fx_jpy(canon, "GBP", "JPY", Some(190.0)),
        "AUDJPY" => fx_jpy(canon, "AUD", "JPY", Some(99.0)),
        "NZDJPY" => fx_jpy(canon, "NZD", "JPY", Some(90.0)),
        "CADJPY" => fx_jpy(canon, "CAD", "JPY", Some(110.0)),
        "CHFJPY" => fx_jpy(canon, "CHF", "JPY", Some(167.0)),
        // ── EUR crosses ──
        "EURGBP" => fx(canon, "EUR", "GBP", 5, Some(0.86)),
        "EURAUD" => fx(canon, "EUR", "AUD", 5, Some(1.66)),
        "EURNZD" => fx(canon, "EUR", "NZD", 5, Some(1.83)),
        "EURCHF" => fx(canon, "EUR", "CHF", 5, Some(0.99)),
        "EURCAD" => fx(canon, "EUR", "CAD", 5, Some(1.50)),
        // ── GBP crosses ──
        "GBPAUD" => fx(canon, "GBP", "AUD", 5, Some(1.93)),
        "GBPNZD" => fx(canon, "GBP", "NZD", 5, Some(2.12)),
        "GBPCHF" => fx(canon, "GBP", "CHF", 5, Some(1.15)),
        "GBPCAD" => fx(canon, "GBP", "CAD", 5, Some(1.74)),
        // ── Metals ──
        "XAUUSD" => SymbolMetadata {
            symbol: canon,
            base: "XAU".into(),
            quote: "USD".into(),
            pip_size: 0.01,
            contract_size: 100.0,
            pip_value_quote: 1.0,
            digits: 2,
            min_lot: 0.01,
            max_lot: 100.0,
            lot_step: 0.01,
            typical_price: Some(2_400.0),
        },
        "XAGUSD" => SymbolMetadata {
            symbol: canon,
            base: "XAG".into(),
            quote: "USD".into(),
            pip_size: 0.001,
            contract_size: 5_000.0,
            pip_value_quote: 5.0,
            digits: 3,
            min_lot: 0.01,
            max_lot: 100.0,
            lot_step: 0.01,
            typical_price: Some(28.0),
        },
        // ── Crypto ──
        "BTCUSD" => SymbolMetadata {
            symbol: canon,
            base: "BTC".into(),
            quote: "USD".into(),
            pip_size: 1.0,
            contract_size: 1.0,
            pip_value_quote: 1.0,
            digits: 1,
            min_lot: 0.01,
            max_lot: 100.0,
            lot_step: 0.01,
            typical_price: Some(70_000.0),
        },
        "ETHUSD" => SymbolMetadata {
            symbol: canon,
            base: "ETH".into(),
            quote: "USD".into(),
            pip_size: 0.1,
            contract_size: 1.0,
            pip_value_quote: 0.1,
            digits: 2,
            min_lot: 0.01,
            max_lot: 100.0,
            lot_step: 0.01,
            typical_price: Some(3_500.0),
        },
        _ => return None,
    };
    Some(m)
}

fn fx(symbol: String, base: &str, quote: &str, digits: u32, typical: Option<f64>) -> SymbolMetadata {
    let pip_size = 0.0001;
    let contract_size = 100_000.0;
    SymbolMetadata {
        symbol,
        base: base.into(),
        quote: quote.into(),
        pip_size,
        contract_size,
        pip_value_quote: pip_size * contract_size,
        digits,
        min_lot: 0.01,
        max_lot: 100.0,
        lot_step: 0.01,
        typical_price: typical,
    }
}

fn fx_jpy(symbol: String, base: &str, quote: &str, typical: Option<f64>) -> SymbolMetadata {
    // JPY pairs use 0.01 pip and quote in 3 digits (cTrader). Pip
    // value in quote (JPY) per standard lot = 0.01 * 100_000 = 1000 JPY.
    let pip_size = 0.01;
    let contract_size = 100_000.0;
    SymbolMetadata {
        symbol,
        base: base.into(),
        quote: quote.into(),
        pip_size,
        contract_size,
        pip_value_quote: pip_size * contract_size,
        digits: 3,
        min_lot: 0.01,
        max_lot: 100.0,
        lot_step: 0.01,
        typical_price: typical,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn baked_in_defaults_cover_majors_and_jpy_and_metals() {
        for sym in ["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "XAUUSD", "XAGUSD"] {
            let m = baked_in_default(sym).expect(sym);
            assert_eq!(m.symbol, sym);
            assert!(m.pip_size > 0.0);
            assert!(m.contract_size > 0.0);
        }
    }

    #[test]
    fn jpy_pairs_use_two_digit_pip() {
        let usdjpy = baked_in_default("USDJPY").unwrap();
        assert_eq!(usdjpy.pip_size, 0.01);
        let eurjpy = baked_in_default("EURJPY").unwrap();
        assert_eq!(eurjpy.pip_size, 0.01);
    }

    #[test]
    fn xauusd_uses_metal_constants() {
        let xau = baked_in_default("XAUUSD").unwrap();
        assert_eq!(xau.pip_size, 0.01);
        assert_eq!(xau.contract_size, 100.0);
        assert_eq!(xau.pip_value_quote, 1.0);
    }

    #[test]
    fn pip_value_in_account_handles_quote_match_directly() {
        let eurusd = baked_in_default("EURUSD").unwrap();
        // EURUSD on a USD account — quote IS account, no conversion.
        assert!((eurusd.pip_value_in_account("USD", None, None) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn pip_value_in_account_handles_base_match_via_price() {
        // USDJPY on a USD account — base IS account. pip_value_USD
        // = pip_value_quote (JPY) / spot.
        let usdjpy = baked_in_default("USDJPY").unwrap();
        let v = usdjpy.pip_value_in_account("USD", None, Some(150.0));
        // 1000 JPY / 150 ≈ 6.67 USD per pip per lot.
        assert!((v - 6.6667).abs() < 0.01);
    }

    #[test]
    fn pip_value_in_account_returns_nan_on_cross_without_conv_rate() {
        // EURJPY on a USD account, no rate supplied — should refuse
        // rather than silently lie.
        let eurjpy = baked_in_default("EURJPY").unwrap();
        let v = eurjpy.pip_value_in_account("USD", None, None);
        assert!(v.is_nan());
    }

    #[test]
    fn pip_value_in_account_uses_supplied_conv_rate_for_cross() {
        // EURJPY on USD account with JPY→USD rate ≈ 0.0067.
        let eurjpy = baked_in_default("EURJPY").unwrap();
        let v = eurjpy.pip_value_in_account("USD", Some(0.0067), None);
        // 1000 JPY * 0.0067 ≈ 6.7 USD.
        assert!((v - 6.7).abs() < 0.01);
    }

    #[test]
    fn canonical_strips_separators_and_uppercases() {
        assert_eq!(canonical_symbol("eur/usd"), "EURUSD");
        assert_eq!(canonical_symbol("xau-usd"), "XAUUSD");
        assert_eq!(canonical_symbol("EUR_USD.cTr"), "EURUSDCTR");
    }
}
