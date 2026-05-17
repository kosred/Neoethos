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
//! - `resolve` consults only the on-disk table. There is NO in-source
//!   default table any more: synthetic per-symbol constants are a
//!   risk-gate hazard (a stale "typical price" for a JPY pair changes
//!   pip-value-in-account by 30% silently). The legacy
//!   `baked_in_default` function below is retained for the unit-test
//!   gate (`#[cfg(test)]`) only, behind the `allow_baked_defaults`
//!   feature so production code paths can never reach it.

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
            return match price {
                Some(v) => self.pip_value_quote / v,
                None => {
                    // Previously silently divided by 1.0 (i.e. returned
                    // pip_value_quote) when no live or typical price was
                    // available. That masks a missing-metadata bug for the
                    // base==account case; return NaN so downstream PnL
                    // collapses visibly and the order is refused.
                    tracing::warn!(
                        target: "forex_core::symbol_metadata",
                        symbol = %self.symbol,
                        account = %acct,
                        "pip_value_in_account: base==account but no live or typical price; returning NaN"
                    );
                    f64::NAN
                }
            };
        }
        if let Some(rate) = quote_to_account_rate.filter(|v| v.is_finite() && *v > 0.0) {
            return self.pip_value_quote * rate;
        }
        f64::NAN
    }
}

/// Disk-backed table. Loaded once per process; subsequent lookups are
/// Current schema version of `symbol_metadata.json`. Per D4
/// versioning policy.
pub const SYMBOL_METADATA_SCHEMA_VERSION: crate::schema_version::SchemaVersion =
    crate::schema_version::SchemaVersion::new(1);

/// in-memory. Reload with `reload()` after the cTrader connector
/// writes new entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolMetadataTable {
    /// On-disk schema version. Defaults to v1 (the pre-versioning
    /// shape) for files written by older builds.
    #[serde(default = "crate::schema_version::default_v1")]
    pub schema_version: crate::schema_version::SchemaVersion,
    /// Map keyed by canonical (uppercase, no-separator) symbol.
    pub entries: HashMap<String, SymbolMetadata>,
}

impl Default for SymbolMetadataTable {
    fn default() -> Self {
        Self {
            schema_version: SYMBOL_METADATA_SCHEMA_VERSION,
            entries: HashMap::new(),
        }
    }
}

impl crate::schema_version::HasSchemaVersion for SymbolMetadataTable {
    const CURRENT: crate::schema_version::SchemaVersion = SYMBOL_METADATA_SCHEMA_VERSION;
    fn schema_version(&self) -> crate::schema_version::SchemaVersion {
        self.schema_version
    }
}

impl SymbolMetadataTable {
    pub fn load_from_disk(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read symbol metadata at {}", path.display()))?;
        let table: SymbolMetadataTable = serde_json::from_str(&text)
            .with_context(|| format!("parse symbol metadata at {}", path.display()))?;
        // Reject too-new schema versions loud — caller falls back
        // to defaults (per the D4 contract).
        crate::schema_version::ensure_schema_version_readable(
            &table,
            "symbol_metadata.json",
        )?;
        Ok(table)
    }

    pub fn save_to_disk(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        // Stamp current schema version on every save — defends
        // against in-memory construction paths that forgot to set
        // it. Cloning is cheap since `entries` is references-by-value
        // and the version field is a u32 newtype.
        let mut to_write = self.clone();
        to_write.schema_version = SYMBOL_METADATA_SCHEMA_VERSION;
        let text = serde_json::to_string_pretty(&to_write)
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
/// relative to CWD, falling back to the packaged
/// `assets/symbol_metadata/defaults.json` for fresh checkouts.
///
/// The asset file is a *snapshot* of the broker symbol table — it is
/// NOT a synthetic table. The cTrader connector overwrites
/// `data/symbol_metadata.json` with live ProtoOASymbol records on
/// every reconcile, and the asset version is only used when no
/// operator file exists yet.
pub fn metadata_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("FOREX_BOT_SYMBOL_METADATA") {
        return std::path::PathBuf::from(p);
    }
    let cwd_path = std::path::PathBuf::from("data").join("symbol_metadata.json");
    if cwd_path.exists() {
        return cwd_path;
    }
    // Packaged asset, walked up from CARGO_MANIFEST_DIR at build time.
    let asset = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("assets")
        .join("symbol_metadata")
        .join("defaults.json");
    if asset.exists() {
        return asset;
    }
    cwd_path
}

/// Load (and cache) the on-disk metadata table. Returns the empty
/// table if no file exists — callers MUST treat a `None` from
/// `resolve` as an unrecoverable configuration error rather than
/// silently synthesising defaults. The previous lenient behaviour
/// (fall back to `baked_in_default`) made it impossible to detect
/// when the cTrader bootstrap forgot to write the per-broker
/// metadata, so JPY-pair pip-value math could be silently wrong.
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
                    tracing::error!(
                        target: "forex_core::symbol_metadata",
                        path = %path.display(),
                        error = %err,
                        "failed to load symbol metadata; lookups will return None \
                         (no synthetic fallback). Populate the JSON from cTrader \
                         ProtoOASymbol records."
                    );
                    SymbolMetadataTable::default()
                }
            }
        } else {
            tracing::error!(
                target: "forex_core::symbol_metadata",
                path = %path.display(),
                "no symbol metadata file found; lookups will return None \
                 (no synthetic fallback). Run the cTrader bootstrap or \
                 supply assets/symbol_metadata/defaults.json."
            );
            SymbolMetadataTable::default()
        }
    })
}

/// One-shot resolver: disk-table only. Returns `None` for any symbol
/// the operator has not supplied metadata for. Production callers
/// must treat `None` as a hard error (refuse to size the order,
/// refuse to backtest) — they are NOT permitted to compute a
/// synthetic default. The legacy `baked_in_default` is a
/// `#[cfg(test)]`-only helper for the unit-test gate that exercises
/// `pip_value_in_account`.
pub fn resolve(symbol: &str) -> Option<SymbolMetadata> {
    global_table().lookup(symbol).cloned()
}

/// Hand-curated metadata for the symbols every prop-firm operator
/// trades. Numbers verified against cTrader / TradingView spec sheets.
///
/// TEST-ONLY. Production code must never call this — it is the
/// synthetic fallback the rest of this module was rewritten to
/// remove. The function is retained exclusively so the unit tests
/// below (`pip_value_in_account_*`, `canonical_*`) can exercise the
/// math without spinning up a JSON loader. The disk-backed table
/// (populated by the cTrader connector / shipped in
/// `assets/symbol_metadata/defaults.json`) is the only legitimate
/// source for runtime callers.
#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
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
