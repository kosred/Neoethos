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
    /// **F-126 fix (2026-05-25)** — broker-authoritative typical spread
    /// in pips for this symbol. Replaces the per-asset-class synthetic
    /// defaults in `genetic::strategy_gene::infer_market_cost_profile`
    /// (metal 2.5 / crypto 8 / fx 1.5 — F-029 root cause). When `Some`,
    /// the backtest cost-model MUST use this value instead of the
    /// asset-class heuristic. When `None`, the caller is responsible
    /// for supplying an override or rejecting the backtest.
    ///
    /// Sourced from cTrader's `ProtoOASymbol::spread` field (in pips)
    /// or an operator override in `data/symbol_metadata.json`. Use
    /// `serde(default)` so existing on-disk tables without this field
    /// load with `None`.
    #[serde(default)]
    pub typical_spread_pips: Option<f64>,
    /// **F-126 fix (2026-05-25)** — broker-authoritative commission per
    /// standard lot, denominated in `quote` currency. Replaces the
    /// hardcoded `$7 per lot` synthetic default in
    /// `infer_market_cost_profile`. When `Some`, the backtest cost-model
    /// MUST use this value instead of the synthetic default. When
    /// `None`, the caller is responsible for supplying an override or
    /// rejecting the backtest.
    ///
    /// Sourced from cTrader's commission schedule (per-symbol via
    /// `ProtoOASymbolCommissionType` + `commission` fields) or an
    /// operator override in `data/symbol_metadata.json`.
    #[serde(default)]
    pub commission_per_lot: Option<f64>,

    /// **Phase C (2026-05-28) — broker swap & conversion-fee fields.**
    ///
    /// The cTrader broker hands us these values for every symbol it
    /// streams via `ProtoOASymbol`. Backtest cost models that ignore
    /// them ship strategies whose live PnL silently lags by the swap
    /// + fee delta. The audit-finding F-CY3-2 + F-CY3-3 chain
    /// motivated lifting these onto the canonical metadata table so
    /// the GA fitness function has the same cost view the live
    /// trader does.
    ///
    /// Daily SWAP charge for a long position, in **pips per day**.
    /// Positive value = COST (debit per overnight roll). Negative =
    /// credit. Only used when the broker's
    /// `swap_calculation_type` is `PIPS` — the other proto variants
    /// (`PERCENTAGE`, `POINTS`) require a different conversion path
    /// and currently fall back to `None` (Phase C follow-up).
    #[serde(default)]
    pub daily_swap_long_pips: Option<f64>,
    /// Daily SWAP charge for a short position. Same semantics as
    /// `daily_swap_long_pips`.
    #[serde(default)]
    pub daily_swap_short_pips: Option<f64>,
    /// Per-trade conversion fee when symbol-quote ≠ deposit-currency.
    /// Stored as a **fraction**, e.g. `0.0005` = 0.05 %. cTrader's
    /// proto value is `i32` with the convention `1 = 0.01 %`, so the
    /// conversion is `fraction = proto_value / 10_000.0`. Applied
    /// once at trade exit: `pnl_net = pnl_gross × (1 − fee_rate)`.
    /// Typical broker values: 0.005 – 0.01 (0.5 – 1 %) for cross-
    /// currency accounts; `None` for same-quote-and-deposit accounts
    /// (no conversion needed).
    #[serde(default)]
    pub pnl_conversion_fee_rate: Option<f64>,
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
                        target: "neoethos_core::symbol_metadata",
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

    // ─── Unit conversion helpers (2026-05-27 cycle-3) ──────────────────
    //
    // The user-pointed-out fundamental: **PnL is in MONEY, trades are in
    // LOT SIZES**. Every operation that translates between those two
    // dimensions must consult per-symbol economic data (pip_size,
    // contract_size, quote currency) and per-account economic data
    // (deposit currency, live FX rate). The helpers below are the
    // *only* sanctioned bridge between the two dimensions — call-sites
    // that compute pips/lots/money math by hand are bugs waiting to
    // surface.
    //
    // Worked example (referenced in tests below):
    //   - Account: GBP deposit
    //   - Symbol:  EURUSD (base=EUR, quote=USD, pip_size=0.0001,
    //              contract_size=100_000)
    //   - Live USD→GBP rate: ~0.79 (i.e. 1 USD = 0.79 GBP)
    //   - pip_value_quote = 0.0001 × 100_000 = $10 / lot
    //   - pip_value_in_account = $10 × 0.79 = £7.90 / lot
    //   - Risk £100 with 20-pip SL → max_lots =
    //       £100 / (20 pips × £7.90/lot) ≈ 0.633 lot
    //   - Actual loss on the 20-pip stop = 0.633 × 20 × £7.90 = £100.01 ✓

    /// Compute the lot size that risks AT MOST `max_loss_account_ccy`
    /// over a `sl_pips` stop-loss distance. Returns `None` when the
    /// inputs make the answer ill-defined: zero SL distance, missing
    /// FX rate for a cross pair, etc.
    ///
    /// Inverse of `account_pnl_to_pips`. Used by the risk-gate when
    /// converting a percent-of-equity risk budget into a concrete
    /// order size.
    pub fn risk_money_to_lots(
        &self,
        max_loss_account_ccy: f64,
        sl_pips: f64,
        account_currency: &str,
        quote_to_account_rate: Option<f64>,
        live_price: Option<f64>,
    ) -> Option<f64> {
        if !sl_pips.is_finite() || sl_pips <= 0.0 {
            return None;
        }
        if !max_loss_account_ccy.is_finite() || max_loss_account_ccy <= 0.0 {
            return None;
        }
        let pip_value_account = self.pip_value_in_account(
            account_currency,
            quote_to_account_rate,
            live_price,
        );
        if !pip_value_account.is_finite() || pip_value_account <= 0.0 {
            return None;
        }
        let lots = max_loss_account_ccy / (sl_pips * pip_value_account);
        if !lots.is_finite() || lots <= 0.0 {
            return None;
        }
        Some(lots)
    }

    /// Compute the pip-distance corresponding to a given PnL in account
    /// currency. Used by `bridge.rs` to render the "PnL in pips" column.
    ///
    /// This is the **correct** formula:
    ///   pips = pnl_account / (lots × pip_value_in_account)
    /// where `pip_value_in_account` already includes the
    /// quote-currency → deposit-currency FX conversion.
    ///
    /// **A.3 fix** (batch-1 audit): the previous implementation used
    /// `pnl_account / (lots × pip_value_quote)` — wrong for any
    /// non-quote-currency account because it skipped the FX step. On a
    /// GBP account trading EURUSD that bug under-reported pips by the
    /// USD/GBP factor (~25%).
    pub fn account_pnl_to_pips(
        &self,
        pnl_account_ccy: f64,
        lots: f64,
        account_currency: &str,
        quote_to_account_rate: Option<f64>,
        live_price: Option<f64>,
    ) -> Option<f64> {
        if !lots.is_finite() || lots == 0.0 {
            return None;
        }
        let pip_value_account = self.pip_value_in_account(
            account_currency,
            quote_to_account_rate,
            live_price,
        );
        if !pip_value_account.is_finite() || pip_value_account <= 0.0 {
            return None;
        }
        let pips = pnl_account_ccy / (lots * pip_value_account);
        if !pips.is_finite() {
            return None;
        }
        Some(pips)
    }

    /// Translate a user-facing lot count into the cTrader broker-wire
    /// `volume` field that `ProtoOANewOrderReq` and
    /// `ProtoOAClosePositionReq` accept.
    ///
    /// cTrader's wire `volume` is in centi-units of the base currency
    /// (the `lotSize` proto field, in cents). For EURUSD `lotSize` is
    /// `10_000_000` (= 100,000 EUR × 100 cents), so 0.01 lot →
    /// 100,000 wire.
    ///
    /// The historical bug here was a redundant `× 100` outside the
    /// `lotSize` factor (see commit 6cd24a78 batch B). This helper
    /// removes that footgun by making the conversion a single call.
    ///
    /// `lot_size_cents` is the per-symbol `CTraderSymbolInfo.lot_size`
    /// value. Pass `None` and the helper returns `None` so callers
    /// don't silently underflow with a default.
    pub fn lots_to_wire_volume(
        &self,
        lots: f64,
        lot_size_cents: Option<i64>,
    ) -> Option<i64> {
        if !lots.is_finite() || lots <= 0.0 {
            return None;
        }
        let lot_size = lot_size_cents.filter(|&v| v > 0)?;
        let wire = (lots * lot_size as f64).round() as i64;
        if wire <= 0 { return None; }
        Some(wire)
    }

    /// Reverse direction — translate a broker-reported wire volume
    /// back into lots for display in the Open Positions table.
    ///
    /// For EURUSD wire `10_000_000` → `1.0` lot.
    pub fn wire_volume_to_lots(
        &self,
        wire_volume: i64,
        lot_size_cents: Option<i64>,
    ) -> Option<f64> {
        if wire_volume <= 0 {
            return None;
        }
        let lot_size = lot_size_cents.filter(|&v| v > 0)?;
        let lots = wire_volume as f64 / lot_size as f64;
        if !lots.is_finite() { return None; }
        Some(lots)
    }

    /// Compute the gross PnL in **account currency** for an open
    /// position. Currently unused (the broker is authoritative via
    /// `ProtoOAPositionUnrealizedPnL.netUnrealizedPnL`), but exposed
    /// so the backtest can reuse the same formula and the live
    /// drift-detection circuit-breaker (currently dead per E.2 audit
    /// finding) can be re-armed.
    ///
    /// `entry_price` and `exit_price` are in the symbol's quote
    /// currency. For a BUY: PnL = (exit - entry) × contract × lots ×
    /// quote_to_account. For a SELL: invert sign.
    pub fn position_pnl_account(
        &self,
        entry_price: f64,
        exit_price: f64,
        lots: f64,
        is_buy: bool,
        account_currency: &str,
        quote_to_account_rate: Option<f64>,
        live_price_for_base_account: Option<f64>,
    ) -> Option<f64> {
        if !entry_price.is_finite() || !exit_price.is_finite() {
            return None;
        }
        if !lots.is_finite() || lots <= 0.0 {
            return None;
        }
        let price_delta_pips = (exit_price - entry_price) / self.pip_size;
        let signed_pips = if is_buy { price_delta_pips } else { -price_delta_pips };
        let pip_value_account = self.pip_value_in_account(
            account_currency,
            quote_to_account_rate,
            live_price_for_base_account,
        );
        if !pip_value_account.is_finite() {
            return None;
        }
        Some(signed_pips * lots * pip_value_account)
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
        crate::schema_version::ensure_schema_version_readable(&table, "symbol_metadata.json")?;
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
        let text = serde_json::to_string_pretty(&to_write).context("serialize symbol metadata")?;
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
                        target: "neoethos_core::symbol_metadata",
                        path = %path.display(),
                        entries = t.entries.len(),
                        "loaded on-disk symbol metadata"
                    );
                    t
                }
                Err(err) => {
                    tracing::error!(
                        target: "neoethos_core::symbol_metadata",
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
                target: "neoethos_core::symbol_metadata",
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
            // F-126 fix: None forces caller to supply real broker spread
            // / commission — no per-asset-class synthetic default.
            typical_spread_pips: None,
            commission_per_lot: None,
            // Phase C — broker-supplied; None forces caller to fetch.
            daily_swap_long_pips: None,
            daily_swap_short_pips: None,
            pnl_conversion_fee_rate: None,
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
            typical_spread_pips: None,
            commission_per_lot: None,
            daily_swap_long_pips: None,
            daily_swap_short_pips: None,
            pnl_conversion_fee_rate: None,
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
            typical_spread_pips: None,
            commission_per_lot: None,
            daily_swap_long_pips: None,
            daily_swap_short_pips: None,
            pnl_conversion_fee_rate: None,
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
            typical_spread_pips: None,
            commission_per_lot: None,
            daily_swap_long_pips: None,
            daily_swap_short_pips: None,
            pnl_conversion_fee_rate: None,
        },
        _ => return None,
    };
    Some(m)
}

#[cfg(test)]
fn fx(
    symbol: String,
    base: &str,
    quote: &str,
    digits: u32,
    typical: Option<f64>,
) -> SymbolMetadata {
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
        typical_spread_pips: None,
        commission_per_lot: None,
        daily_swap_long_pips: None,
        daily_swap_short_pips: None,
        pnl_conversion_fee_rate: None,
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
        typical_spread_pips: None,
        commission_per_lot: None,
        daily_swap_long_pips: None,
        daily_swap_short_pips: None,
        pnl_conversion_fee_rate: None,
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

    // ─── Phase C broker-cost fields (2026-05-28) ──────────────────────
    //
    // Schema additions for `daily_swap_long_pips`, `daily_swap_short_pips`,
    // and `pnl_conversion_fee_rate`. These are populated by the cTrader
    // bootstrap path from `SymbolFinancials` and consumed by the
    // backtest cost model. Tests verify:
    //   1. Baked-in defaults set them to `None` (so caller-supplied
    //      broker data is the ONLY source of truth — no synthetic
    //      "1.5 pips/day swap" silent default).
    //   2. Serde round-trip preserves them.
    //   3. Loading a `SymbolMetadata` JSON written before Phase C
    //      (i.e. without these fields) still works — they default to
    //      `None` thanks to `#[serde(default)]`.

    #[test]
    fn phase_c_fields_default_to_none_in_baked_defaults() {
        for sym in ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "XAGUSD"] {
            let m = baked_in_default(sym).unwrap_or_else(|| panic!("missing baked: {sym}"));
            assert_eq!(m.daily_swap_long_pips, None, "{sym}: swap_long default");
            assert_eq!(m.daily_swap_short_pips, None, "{sym}: swap_short default");
            assert_eq!(m.pnl_conversion_fee_rate, None, "{sym}: fee_rate default");
        }
    }

    #[test]
    fn phase_c_fields_serde_round_trip() {
        let mut m = baked_in_default("EURUSD").unwrap();
        m.daily_swap_long_pips = Some(-0.7);
        m.daily_swap_short_pips = Some(0.3);
        m.pnl_conversion_fee_rate = Some(0.005);
        let json = serde_json::to_string(&m).expect("serialize");
        let back: SymbolMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.daily_swap_long_pips, Some(-0.7));
        assert_eq!(back.daily_swap_short_pips, Some(0.3));
        assert_eq!(back.pnl_conversion_fee_rate, Some(0.005));
    }

    #[test]
    fn phase_c_fields_default_when_missing_in_legacy_json() {
        // Pre-Phase-C SymbolMetadata JSON — no swap/fee fields. Must
        // still deserialize, with the new fields landing as `None`.
        let legacy_json = r#"{
            "symbol": "EURUSD",
            "base": "EUR",
            "quote": "USD",
            "pip_size": 0.0001,
            "contract_size": 100000.0,
            "pip_value_quote": 10.0,
            "digits": 5,
            "min_lot": 0.01,
            "max_lot": 100.0,
            "lot_step": 0.01,
            "typical_price": 1.08
        }"#;
        let m: SymbolMetadata = serde_json::from_str(legacy_json).expect("legacy deserialize");
        assert_eq!(m.daily_swap_long_pips, None);
        assert_eq!(m.daily_swap_short_pips, None);
        assert_eq!(m.pnl_conversion_fee_rate, None);
        // Sanity: existing fields landed correctly.
        assert_eq!(m.symbol, "EURUSD");
        assert_eq!(m.pip_size, 0.0001);
    }

    // ─── Money ↔ Lots conversion (cycle-3 helpers, 2026-05-27) ────────

    #[test]
    fn risk_money_to_lots_eurusd_usd_account() {
        // Account = USD (quote == account), so pip_value_account = $10/lot.
        // Risk $100 with 20-pip SL → 100 / (20 × 10) = 0.5 lot.
        let eurusd = baked_in_default("EURUSD").unwrap();
        let lots = eurusd
            .risk_money_to_lots(100.0, 20.0, "USD", None, None)
            .expect("ok");
        assert!((lots - 0.5).abs() < 1e-9, "expected 0.5, got {lots}");
    }

    #[test]
    fn risk_money_to_lots_eurusd_gbp_account_uses_fx() {
        // Account = GBP (cross), pip_value_quote = $10/lot, USD→GBP ≈ 0.79.
        // pip_value_account = 10 × 0.79 = £7.90/lot.
        // Risk £100 with 20-pip SL → 100 / (20 × 7.90) ≈ 0.6329 lot.
        let eurusd = baked_in_default("EURUSD").unwrap();
        let lots = eurusd
            .risk_money_to_lots(100.0, 20.0, "GBP", Some(0.79), None)
            .expect("ok");
        assert!(
            (lots - 0.6329).abs() < 1e-3,
            "expected ≈0.633, got {lots}"
        );
        // And the realized loss at 20 pips × 0.6329 lots × £7.90/pip/lot
        // should land within ~1p of the £100 budget — the round-trip
        // sanity check.
        let realized = 20.0 * lots * (10.0 * 0.79);
        assert!((realized - 100.0).abs() < 0.1, "realized={realized}");
    }

    #[test]
    fn risk_money_to_lots_refuses_when_fx_missing_for_cross() {
        let eurusd = baked_in_default("EURUSD").unwrap();
        // Cross pair with no quote_to_account rate → cannot size.
        let lots = eurusd.risk_money_to_lots(100.0, 20.0, "GBP", None, None);
        assert!(lots.is_none(), "expected None, got {:?}", lots);
    }

    #[test]
    fn risk_money_to_lots_refuses_zero_sl() {
        let eurusd = baked_in_default("EURUSD").unwrap();
        assert!(eurusd
            .risk_money_to_lots(100.0, 0.0, "USD", None, None)
            .is_none());
        assert!(eurusd
            .risk_money_to_lots(100.0, -5.0, "USD", None, None)
            .is_none());
    }

    #[test]
    fn account_pnl_to_pips_is_inverse_of_risk_sizing() {
        // Round-trip: size 0.5 lot, lose £100 over 20 pips on GBP account.
        // Then ask the helper how many pips that £100 loss represents.
        // Should be ~20 pips back out.
        let eurusd = baked_in_default("EURUSD").unwrap();
        let pips = eurusd
            .account_pnl_to_pips(-100.0, 0.5, "USD", None, None)
            .expect("ok");
        // pip_value_account = $10/lot. PnL/(lots × pip_value) = -100 / (0.5 × 10) = -20.
        assert!((pips - -20.0).abs() < 1e-9, "expected -20, got {pips}");
    }

    #[test]
    fn account_pnl_to_pips_fixes_a3_under_cross_currency_account() {
        // Documented A.3 bug from batch-1 audit: on a GBP account
        // trading EURUSD, the OLD formula (`pnl / (lots × pip_value_quote)`)
        // ignored the USD→GBP FX conversion → reported pips ~25% off
        // (USD vs GBP rate). This test pins the CORRECT formula.
        //
        // Scenario: GBP account, EURUSD 0.5 lot, broker-reported PnL
        // = +£79 (which corresponds to +20 pips: 20 × 0.5 × $10 × 0.79).
        let eurusd = baked_in_default("EURUSD").unwrap();
        let pips = eurusd
            .account_pnl_to_pips(79.0, 0.5, "GBP", Some(0.79), None)
            .expect("ok");
        assert!((pips - 20.0).abs() < 1e-2, "expected ≈20, got {pips}");
    }

    #[test]
    fn lots_to_wire_volume_eurusd() {
        // EURUSD: lot_size_cents = 10_000_000.
        // 0.01 lot → 100_000 wire.
        // 1.0  lot → 10_000_000 wire.
        let eurusd = baked_in_default("EURUSD").unwrap();
        assert_eq!(
            eurusd.lots_to_wire_volume(0.01, Some(10_000_000)),
            Some(100_000)
        );
        assert_eq!(
            eurusd.lots_to_wire_volume(1.0, Some(10_000_000)),
            Some(10_000_000)
        );
    }

    #[test]
    fn lots_to_wire_volume_refuses_when_lot_size_unknown() {
        let eurusd = baked_in_default("EURUSD").unwrap();
        assert!(eurusd.lots_to_wire_volume(0.01, None).is_none());
        assert!(eurusd.lots_to_wire_volume(0.01, Some(0)).is_none());
        assert!(eurusd.lots_to_wire_volume(0.01, Some(-100)).is_none());
    }

    #[test]
    fn wire_volume_to_lots_round_trips() {
        let eurusd = baked_in_default("EURUSD").unwrap();
        let lots = 0.01;
        let wire = eurusd.lots_to_wire_volume(lots, Some(10_000_000)).unwrap();
        let back = eurusd.wire_volume_to_lots(wire, Some(10_000_000)).unwrap();
        assert!((back - lots).abs() < 1e-9, "round-trip lost: {back} ≠ {lots}");
    }

    #[test]
    fn position_pnl_account_buy_positive_move_gbp_account() {
        // Buy EURUSD at 1.0800, exits at 1.0820 → +20 pips.
        // 0.1 lot × 20 pips × $10/pip × 0.79 (USD→GBP) = +£15.80.
        let eurusd = baked_in_default("EURUSD").unwrap();
        let pnl = eurusd
            .position_pnl_account(
                1.0800,
                1.0820,
                0.1,
                /* is_buy */ true,
                "GBP",
                Some(0.79),
                None,
            )
            .expect("ok");
        assert!((pnl - 15.80).abs() < 0.01, "expected ≈£15.80, got £{pnl}");
    }

    #[test]
    fn position_pnl_account_sell_negative_move_is_profit() {
        // Sell EURUSD at 1.0820, exits at 1.0800 → +20 pips for a short.
        let eurusd = baked_in_default("EURUSD").unwrap();
        let pnl = eurusd
            .position_pnl_account(
                1.0820,
                1.0800,
                0.1,
                /* is_buy */ false,
                "USD",
                None,
                None,
            )
            .expect("ok");
        // pip_value $10/lot, 0.1 lot, 20 pips → $20.
        assert!((pnl - 20.0).abs() < 1e-6, "expected $20, got ${pnl}");
    }

    #[test]
    fn position_pnl_account_usdjpy_uses_pip_size_0_01() {
        // USDJPY pip_size = 0.01. Buy at 149.00, exits at 149.20 → +20 pips.
        // pip_value_quote = 0.01 × 100_000 = 1000 JPY/lot.
        // 0.1 lot × 20 × 1000 = 2000 JPY. On USD account
        // with live price 149.0 (using base==account fallback), the
        // pip_value_in_account = 1000 / 149 ≈ 6.71 USD/lot → 0.1 × 20 × 6.71 ≈ $13.42.
        let usdjpy = baked_in_default("USDJPY").unwrap();
        let pnl = usdjpy
            .position_pnl_account(
                149.00,
                149.20,
                0.1,
                /* is_buy */ true,
                "USD",
                None,
                Some(149.0),
            )
            .expect("ok");
        assert!((pnl - 13.42).abs() < 0.05, "expected ≈$13.42, got ${pnl}");
    }
}
