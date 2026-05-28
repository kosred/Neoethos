//! cTrader `ProtoOASymbolByIdRes` raw-payload capture tool.
//!
//! **Why this exists**: the Phase A.1 audit (2026-05-28) confirmed
//! that ZERO recorded broker payloads exist in the repo. Every
//! `SymbolFinancials` parser test uses synthetic JSON with only 4
//! fields. The 30+ other proto fields (swap, commission, distance,
//! schedule, holidays, rollover, ...) are parsed against assumptions
//! drawn from the proto comments — never against what cTrader
//! actually sends.
//!
//! This module connects to the configured cTrader account, runs the
//! 3-step sequence
//!   `ProtoOAApplicationAuthReq → ProtoOAAccountAuthReq →
//!    ProtoOASymbolByIdReq`
//! for each requested symbol, and dumps the raw envelope JSON to
//! `<output_dir>/ctrader_symbol_<SYMBOL>.json`. The fixtures then
//! drive a parser test under
//! `crates/neoethos-app/tests/fixtures/ctrader_symbol_*.json` so
//! every Phase C cost-model assumption is grounded in real bytes.
//!
//! **Operator usage** (from a developer machine with a valid cTrader
//! OAuth bundle in the OS keyring):
//!   neoethos-app --capture-symbols EURUSD,USDJPY,XAUUSD,BTCUSD \
//!                --capture-output crates/neoethos-app/tests/fixtures
//!
//! The binary exits after dumping; no HTTP server is started. Safe
//! to run multiple times — each invocation overwrites the named
//! fixture files.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use serde_json::{Map, Value, json, to_value};

use crate::app_services::broker_persistence::load_broker_settings;
use crate::app_services::ctrader_data::{
    parse_asset_class_list_response, parse_asset_list_response, parse_symbol_by_id_response,
    parse_symbol_category_list_response, parse_symbols_list_response, CTraderSymbolInfo,
};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    build_account_auth_request, build_application_auth_request, build_asset_class_list_request,
    build_asset_list_request, build_symbol_by_id_request, build_symbol_category_list_request,
    build_symbols_list_request, CTraderOpenApiTransport, ProductionCTraderOpenApiTransport,
};
use crate::app_services::secure_store::production_ctrader_token_store;

/// Asset classes the forex-ai catalog keeps. Match is
/// case-insensitive substring. Anything else (Stocks, ETFs,
/// Cryptocurrencies, ...) is dropped at bootstrap time.
///
/// User directive (2026-05-28): "FX majors/minors/exotics + Metals
/// + Indices + Commodities (oil/gas) — ΑΥΤΑ ΚΑΙ ΜΟΝΟ ΑΥΤΑ".
pub const FOREX_AI_ASSET_CLASS_KEYWORDS: &[&str] =
    &["forex", "fx", "metal", "indice", "index", "commodit", "energ", "oil", "gas"];

/// True iff the broker's asset class name matches one of the
/// forex-ai keep keywords. Case-insensitive substring match — broker
/// naming varies ("Forex" vs "FX" vs "Currencies", "Metals" vs
/// "Spot Metals", "Indices" vs "Stock Indices", "Commodities" vs
/// "Energies"), so a keyword list with permissive matching is the
/// most-portable filter.
pub fn is_forex_ai_asset_class(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    FOREX_AI_ASSET_CLASS_KEYWORDS
        .iter()
        .any(|kw| lower.contains(kw))
}

/// Resolve creds, hit cTrader, write one fixture per symbol.
///
/// The output is a pair of files per symbol:
///   - `ctrader_symbol_<SYMBOL>.raw.json` — the verbatim envelope
///     returned by `ProtoOASymbolByIdReq`. This is the ground-truth
///     payload that parser tests assert against.
///   - `ctrader_symbol_<SYMBOL>.decoded.json` — the same payload run
///     through `parse_symbol_by_id_response`. Useful for humans
///     reviewing the capture; tests should ignore this file (it's
///     a function of the parser, not the broker).
///
/// Both files are written even on per-symbol failure, so a partial
/// run still leaves usable evidence. Per-symbol errors are logged
/// and counted; the function returns `Ok(())` if at least one
/// symbol succeeded.
pub fn run_capture(symbols: &[String], output_dir: &Path) -> Result<()> {
    if symbols.is_empty() {
        return Err(anyhow!("--capture-symbols requires at least one symbol"));
    }
    std::fs::create_dir_all(output_dir)
        .map_err(|e| anyhow!("could not create {}: {e}", output_dir.display()))?;

    // ── Step 1: load credentials + token bundle (same path as the
    // production bridge) ─────────────────────────────────────────
    let settings = load_broker_settings();
    let ctrader = &settings.ctrader;
    if ctrader.client_id.trim().is_empty() || ctrader.client_secret.trim().is_empty() {
        return Err(anyhow!(
            "broker_credentials.toml is missing cTrader client_id / client_secret"
        ));
    }
    let account_target = ctrader
        .accounts
        .first()
        .ok_or_else(|| anyhow!("broker_credentials.toml has no cTrader account picked"))?;
    let account_id: i64 = account_target
        .account_id
        .parse()
        .map_err(|e| anyhow!("account_id must be numeric: {e}"))?;

    let token_bundle = production_ctrader_token_store()
        .load_token_bundle_with_legacy_fallback()
        .map_err(|e| anyhow!("token bundle load failed: {e}"))?
        .ok_or_else(|| {
            anyhow!(
                "no saved cTrader OAuth token bundle — run `--reauth` first \
                 to seed the keyring before capturing fixtures"
            )
        })?;

    let environment = match ctrader.environment {
        crate::app_services::broker_config::CTraderBrokerEnvironment::Demo => {
            CTraderEnvironment::Demo
        }
        crate::app_services::broker_config::CTraderBrokerEnvironment::Live => {
            CTraderEnvironment::Live
        }
    };
    let transport = ProductionCTraderOpenApiTransport::new(environment.endpoint_host());

    eprintln!(
        "[capture] environment = {:?}, account_id = {}",
        environment, account_id
    );

    // ── Step 2: fetch the symbols list once to map names → IDs ───
    let auth_responses = transport.send_sequence(&[
        build_application_auth_request(
            &ctrader.client_id,
            &ctrader.client_secret,
            "capture-app-auth",
        ),
        build_account_auth_request(account_id, &token_bundle.access_token, "capture-account-auth"),
        build_symbols_list_request(account_id, false, "capture-symbols-list"),
    ])?;
    if auth_responses.len() < 3 {
        return Err(anyhow!(
            "expected 3 cTrader auth/list responses, got {}",
            auth_responses.len()
        ));
    }
    let symbols_list = parse_symbols_list_response(&auth_responses[2])?;
    eprintln!(
        "[capture] symbols catalog loaded: {} entries",
        symbols_list.symbols.len()
    );

    // ── Step 3: per requested symbol, run symbol-by-id and dump ──
    let mut succeeded = 0usize;
    let mut failed: Vec<String> = Vec::new();
    for requested in symbols {
        let needle = requested.trim().to_ascii_uppercase();
        if needle.is_empty() {
            continue;
        }
        let light = match symbols_list
            .symbols
            .iter()
            .find(|s| s.symbol_name.eq_ignore_ascii_case(&needle))
        {
            Some(s) => s.clone(),
            None => {
                eprintln!(
                    "[capture] {needle}: not in the broker's catalog (skipping). \
                     Available symbols: {} entries.",
                    symbols_list.symbols.len()
                );
                failed.push(needle);
                continue;
            }
        };

        // Per cTrader's session model, every WSS round-trip via
        // ProductionCTraderOpenApiTransport opens a fresh connection.
        // We must re-auth at the head of each batch — same pattern as
        // `resolve_symbol_with_transport` in ctrader_data.rs.
        let detail = match transport.send_sequence(&[
            build_application_auth_request(
                &ctrader.client_id,
                &ctrader.client_secret,
                "capture-app-auth-2",
            ),
            build_account_auth_request(
                account_id,
                &token_bundle.access_token,
                "capture-account-auth-2",
            ),
            build_symbol_by_id_request(account_id, &[light.symbol_id], "capture-symbol-by-id"),
        ]) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[capture] {needle}: WSS sequence failed: {e}");
                failed.push(needle);
                continue;
            }
        };
        if detail.len() < 3 {
            eprintln!(
                "[capture] {needle}: expected 3 responses, got {} — \
                 partial responses follow:\n{:#?}",
                detail.len(),
                detail
            );
            failed.push(needle);
            continue;
        }

        let raw_payload = &detail[2];
        let raw_pretty = match serde_json::from_str::<Value>(raw_payload) {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| raw_payload.clone()),
            Err(_) => raw_payload.clone(),
        };

        let raw_path = output_dir.join(format!("ctrader_symbol_{needle}.raw.json"));
        if let Err(e) = std::fs::write(&raw_path, &raw_pretty) {
            eprintln!("[capture] {needle}: failed to write {}: {e}", raw_path.display());
            failed.push(needle);
            continue;
        }
        eprintln!(
            "[capture] {needle}: wrote raw response ({} bytes) → {}",
            raw_pretty.len(),
            raw_path.display()
        );

        // ── Side artefact: parsed projection for human review ────
        match parse_symbol_by_id_response(raw_payload) {
            Ok(symbols) => {
                if let Some(sym) = symbols
                    .iter()
                    .find(|s| s.symbol_id == light.symbol_id)
                    .cloned()
                {
                    let decoded_path =
                        output_dir.join(format!("ctrader_symbol_{needle}.decoded.json"));
                    let decoded = make_decoded_summary(&sym, &light.symbol_name);
                    if let Ok(text) = serde_json::to_string_pretty(&decoded) {
                        if let Err(e) = std::fs::write(&decoded_path, &text) {
                            eprintln!(
                                "[capture] {needle}: failed to write decoded summary: {e}"
                            );
                        } else {
                            eprintln!(
                                "[capture] {needle}: wrote decoded summary → {}",
                                decoded_path.display()
                            );
                        }
                    }
                } else {
                    eprintln!(
                        "[capture] {needle}: parser returned {} entries but none matched \
                         symbol_id {} — fixture saved, decoded skipped",
                        symbols.len(),
                        light.symbol_id
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "[capture] {needle}: raw saved but parse_symbol_by_id_response \
                     errored: {e} — this IS the audit signal (Phase C). \
                     Inspect the .raw.json to see what the broker actually returned."
                );
            }
        }

        succeeded += 1;
    }

    eprintln!(
        "[capture] done: {succeeded}/{} symbol(s) captured to {}",
        symbols.len(),
        output_dir.display()
    );
    if !failed.is_empty() {
        eprintln!("[capture] failed: {failed:?}");
    }
    if succeeded == 0 {
        return Err(anyhow!(
            "no symbol payloads captured — see log above for per-symbol errors"
        ));
    }
    Ok(())
}

/// Build a JSON-friendly summary of the parsed `CTraderSymbolInfo`,
/// flattening the most operationally-relevant `SymbolFinancials`
/// fields so an operator can eyeball the broker's actual swap,
/// commission, distance, and pnl-conversion-fee values without
/// digging through the raw envelope.
fn make_decoded_summary(symbol: &CTraderSymbolInfo, broker_name: &str) -> Value {
    let financials = symbol.financials.as_ref();

    let financials_value: Value = match financials {
        None => Value::Null,
        Some(f) => {
            // The `json!` macro hits the default `recursion_limit = 128`
            // when the literal exceeds ~20 nested keys, so we build the
            // ~30-field financials projection via a `Map` instead.
            let mut m = Map::new();
            // Commission
            m.insert("commission_type".into(), to_value(f.commission_type).unwrap_or(Value::Null));
            m.insert(
                "precise_trading_commission_rate".into(),
                to_value(f.precise_trading_commission_rate).unwrap_or(Value::Null),
            );
            m.insert(
                "commission_rate_decimal_resolved".into(),
                to_value(f.commission_rate_decimal()).unwrap_or(Value::Null),
            );
            m.insert(
                "precise_min_commission".into(),
                to_value(f.precise_min_commission).unwrap_or(Value::Null),
            );
            m.insert(
                "min_commission_type".into(),
                to_value(f.min_commission_type).unwrap_or(Value::Null),
            );
            m.insert(
                "min_commission_asset".into(),
                to_value(&f.min_commission_asset).unwrap_or(Value::Null),
            );
            m.insert(
                "pnl_conversion_fee_rate".into(),
                to_value(f.pnl_conversion_fee_rate).unwrap_or(Value::Null),
            );
            // Swap
            m.insert("swap_long".into(), to_value(f.swap_long).unwrap_or(Value::Null));
            m.insert("swap_short".into(), to_value(f.swap_short).unwrap_or(Value::Null));
            m.insert(
                "swap_calculation_type".into(),
                to_value(f.swap_calculation_type).unwrap_or(Value::Null),
            );
            m.insert(
                "swap_period_hours".into(),
                to_value(f.swap_period_hours).unwrap_or(Value::Null),
            );
            m.insert(
                "swap_time_minutes_from_utc_midnight".into(),
                to_value(f.swap_time_minutes_from_utc_midnight).unwrap_or(Value::Null),
            );
            m.insert(
                "swap_rollover_3_days".into(),
                to_value(f.swap_rollover_3_days).unwrap_or(Value::Null),
            );
            m.insert(
                "charge_swap_at_weekends".into(),
                to_value(f.charge_swap_at_weekends).unwrap_or(Value::Null),
            );
            m.insert(
                "skip_swap_periods".into(),
                to_value(f.skip_swap_periods).unwrap_or(Value::Null),
            );
            m.insert(
                "daily_swap_long_resolved".into(),
                to_value(f.daily_swap_long()).unwrap_or(Value::Null),
            );
            m.insert(
                "daily_swap_short_resolved".into(),
                to_value(f.daily_swap_short()).unwrap_or(Value::Null),
            );
            // Rollover (Shariah)
            m.insert(
                "rollover_commission".into(),
                to_value(f.rollover_commission).unwrap_or(Value::Null),
            );
            m.insert(
                "rollover_commission_3_days".into(),
                to_value(f.rollover_commission_3_days).unwrap_or(Value::Null),
            );
            m.insert(
                "skip_rollover_days".into(),
                to_value(f.skip_rollover_days).unwrap_or(Value::Null),
            );
            // Distance
            m.insert(
                "sl_distance_points".into(),
                to_value(f.sl_distance_points).unwrap_or(Value::Null),
            );
            m.insert(
                "tp_distance_points".into(),
                to_value(f.tp_distance_points).unwrap_or(Value::Null),
            );
            m.insert(
                "gsl_distance_points".into(),
                to_value(f.gsl_distance_points).unwrap_or(Value::Null),
            );
            m.insert("gsl_charge".into(), to_value(f.gsl_charge).unwrap_or(Value::Null));
            m.insert(
                "distance_set_in".into(),
                to_value(f.distance_set_in).unwrap_or(Value::Null),
            );
            m.insert(
                "guaranteed_stop_loss_available".into(),
                to_value(f.guaranteed_stop_loss_available).unwrap_or(Value::Null),
            );
            // Trading-mode + short-selling
            m.insert("trading_mode".into(), to_value(f.trading_mode).unwrap_or(Value::Null));
            m.insert(
                "enable_short_selling".into(),
                to_value(f.enable_short_selling).unwrap_or(Value::Null),
            );
            // Schedule + holidays
            m.insert(
                "schedule_time_zone".into(),
                to_value(&f.schedule_time_zone).unwrap_or(Value::Null),
            );
            m.insert(
                "trading_intervals_count".into(),
                Value::from(f.trading_intervals.len()),
            );
            m.insert("holidays_count".into(), Value::from(f.holidays.len()));
            // Misc
            m.insert("max_exposure".into(), to_value(f.max_exposure).unwrap_or(Value::Null));
            m.insert("leverage_id".into(), to_value(f.leverage_id).unwrap_or(Value::Null));
            m.insert(
                "measurement_units".into(),
                to_value(&f.measurement_units).unwrap_or(Value::Null),
            );
            Value::Object(m)
        }
    };

    json!({
        "symbol_name_from_list": broker_name,
        "symbol_id": symbol.symbol_id,
        "digits": symbol.digits,
        "pip_position": symbol.pip_position,
        "is_trading_enabled": symbol.is_trading_enabled,
        "lot_size": symbol.lot_size,
        "min_volume": symbol.min_volume,
        "max_volume": symbol.max_volume,
        "step_volume": symbol.step_volume,
        "pnl_conversion_fee_rate_parent": symbol.pnl_conversion_fee_rate,
        "financials_present": financials.is_some(),
        "financials": financials_value,
    })
}

/// Parse the comma-separated CLI argument into a clean list of
/// uppercase symbol names. Empty entries and whitespace are
/// trimmed. Returns the original input if no valid symbols
/// survived so the caller's error message can quote it.
pub fn parse_symbol_list(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim().to_ascii_uppercase())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Default output directory for fixtures captured via the
/// `--capture-symbols` CLI flag.
pub fn default_output_dir() -> PathBuf {
    PathBuf::from("crates/neoethos-app/tests/fixtures")
}

/// Default output directory for the full broker catalog bootstrap
/// (`--bootstrap-broker-catalog`).
///
/// One sub-folder per environment so demo and live catalogs don't
/// collide. The live trader catalog is what production reads at
/// app-startup — the demo catalog is for testing the GA against
/// the broker's actual cost model without touching live.
pub fn default_bootstrap_root() -> PathBuf {
    PathBuf::from("data/broker_symbols")
}

/// **Phase D.1 (2026-05-28)** — full broker catalog bootstrap.
///
/// Fetches ALL symbols from the configured cTrader account
/// (typically 800-900 entries on a typical retail demo) plus their
/// full `ProtoOASymbolByIdRes` payloads, batched 50 IDs per
/// request. Writes:
///
///   - `<output_root>/<env>/raw_batches/batch_NNN.json` — verbatim
///     broker envelopes (audit trail; one file per batch).
///   - `<output_root>/<env>/symbol_index.json` — light index
///     `[{symbol_id, symbol_name, batch_file}, ...]` so callers
///     can locate the batch carrying a given symbol without
///     reading every file.
///   - `<output_root>/<env>/bootstrap_meta.json` — refresh
///     timestamp, env, account_id, schema version. The bridge's
///     24 h refresh consults this to decide whether to re-bootstrap.
///
/// Phase D.2 will add a converter that walks these files and
/// populates the canonical `SymbolMetadataTable` consumed by the
/// backtest cost model, replacing the synthetic fallbacks for
/// commission / spread / swap.
///
/// Runtime: ~30-90 s for 830 symbols (17 batches × ~3-5 s per WSS
/// round-trip). Network-bound; not parallelised because cTrader's
/// 50 req/sec rate limit is shared across the entire account and
/// we want headroom for the live trader.
///
/// **Idempotent.** Re-running overwrites every output file. Safe
/// to call from the 24 h bridge refresh once Phase D.3 lands.
pub fn run_bootstrap(env_label: &str, output_root: &Path) -> Result<()> {
    use crate::app_services::ctrader_messages::{
        CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE, parse_open_api_envelope,
    };
    const SYMBOLS_PER_BATCH: usize = 50;

    let env_dir = output_root.join(env_label);
    let raw_dir = env_dir.join("raw_batches");
    std::fs::create_dir_all(&raw_dir)
        .map_err(|e| anyhow!("could not create {}: {e}", raw_dir.display()))?;

    // ── Load credentials (same as run_capture) ───────────────────
    let settings = load_broker_settings();
    let ctrader = &settings.ctrader;
    if ctrader.client_id.trim().is_empty() || ctrader.client_secret.trim().is_empty() {
        return Err(anyhow!(
            "broker_credentials.toml is missing cTrader client_id / client_secret"
        ));
    }
    let account_target = ctrader
        .accounts
        .first()
        .ok_or_else(|| anyhow!("broker_credentials.toml has no cTrader account picked"))?;
    let account_id: i64 = account_target
        .account_id
        .parse()
        .map_err(|e| anyhow!("account_id must be numeric: {e}"))?;
    let token_bundle = production_ctrader_token_store()
        .load_token_bundle_with_legacy_fallback()
        .map_err(|e| anyhow!("token bundle load failed: {e}"))?
        .ok_or_else(|| {
            anyhow!("no saved cTrader OAuth token bundle — run --reauth first")
        })?;

    let environment = match ctrader.environment {
        crate::app_services::broker_config::CTraderBrokerEnvironment::Demo => {
            CTraderEnvironment::Demo
        }
        crate::app_services::broker_config::CTraderBrokerEnvironment::Live => {
            CTraderEnvironment::Live
        }
    };
    let transport = ProductionCTraderOpenApiTransport::new(environment.endpoint_host());
    eprintln!(
        "[bootstrap] environment = {:?}, account_id = {}, output = {}",
        environment,
        account_id,
        env_dir.display()
    );

    // ── Step 1: symbols list (one round-trip) ────────────────────
    let auth_responses = transport.send_sequence(&[
        build_application_auth_request(
            &ctrader.client_id,
            &ctrader.client_secret,
            "bootstrap-app-auth",
        ),
        build_account_auth_request(account_id, &token_bundle.access_token, "bootstrap-acct-auth"),
        build_symbols_list_request(account_id, false, "bootstrap-symbols-list"),
    ])?;
    if auth_responses.len() < 3 {
        return Err(anyhow!(
            "bootstrap: expected 3 cTrader auth/list responses, got {}",
            auth_responses.len()
        ));
    }
    let symbols_list = parse_symbols_list_response(&auth_responses[2])?;
    let total_raw = symbols_list.symbols.len();
    eprintln!("[bootstrap] symbols catalog: {} entries (unfiltered)", total_raw);
    if total_raw == 0 {
        return Err(anyhow!(
            "broker returned an empty symbol catalog — refusing to write a useless bootstrap"
        ));
    }

    // ── Step 1b: asset-class + symbol-category filter ────────────
    //
    // Forex-ai trades FX / Metals / Indices / Commodities only — the
    // 700+ equity symbols this broker carries are useless cost-model
    // ballast. We fetch the broker's own classification tables and
    // keep only the LightSymbols whose category belongs to a class
    // matching `FOREX_AI_ASSET_CLASS_KEYWORDS`. No name-pattern hacks.
    let class_responses = transport.send_sequence(&[
        build_application_auth_request(
            &ctrader.client_id,
            &ctrader.client_secret,
            "bootstrap-class-auth",
        ),
        build_account_auth_request(
            account_id,
            &token_bundle.access_token,
            "bootstrap-class-acct",
        ),
        build_asset_class_list_request(account_id, "bootstrap-asset-classes"),
    ])?;
    if class_responses.len() < 3 {
        return Err(anyhow!(
            "asset-class list: expected 3 responses, got {}",
            class_responses.len()
        ));
    }
    let asset_classes = parse_asset_class_list_response(&class_responses[2])?;
    let kept_class_ids: std::collections::HashSet<i64> = asset_classes
        .iter()
        .filter(|c| is_forex_ai_asset_class(&c.name))
        .map(|c| c.id)
        .collect();
    eprintln!(
        "[bootstrap] asset classes total={} kept={} ({:?})",
        asset_classes.len(),
        kept_class_ids.len(),
        asset_classes
            .iter()
            .filter(|c| kept_class_ids.contains(&c.id))
            .map(|c| &c.name)
            .collect::<Vec<_>>()
    );
    if kept_class_ids.is_empty() {
        return Err(anyhow!(
            "no asset classes matched the forex-ai keep-list — broker named them: {:?}",
            asset_classes.iter().map(|c| &c.name).collect::<Vec<_>>()
        ));
    }

    let cat_responses = transport.send_sequence(&[
        build_application_auth_request(
            &ctrader.client_id,
            &ctrader.client_secret,
            "bootstrap-cat-auth",
        ),
        build_account_auth_request(account_id, &token_bundle.access_token, "bootstrap-cat-acct"),
        build_symbol_category_list_request(account_id, "bootstrap-symbol-categories"),
    ])?;
    if cat_responses.len() < 3 {
        return Err(anyhow!(
            "symbol-category list: expected 3 responses, got {}",
            cat_responses.len()
        ));
    }
    let categories = parse_symbol_category_list_response(&cat_responses[2])?;
    let kept_category_ids: std::collections::HashSet<i64> = categories
        .iter()
        .filter(|c| kept_class_ids.contains(&c.asset_class_id))
        .map(|c| c.id)
        .collect();
    eprintln!(
        "[bootstrap] symbol categories total={} kept={}",
        categories.len(),
        kept_category_ids.len()
    );

    // Filter the LightSymbols list now.
    let filtered_symbols: Vec<_> = symbols_list
        .symbols
        .into_iter()
        .filter(|ls| {
            // None category_id → broker classification unknown,
            // safest to DROP rather than potentially keep an
            // equity. Phase D.2 can later widen this if any
            // legit forex symbol is found to lack a category.
            ls.symbol_category_id
                .map(|id| kept_category_ids.contains(&id))
                .unwrap_or(false)
        })
        .collect();
    let total = filtered_symbols.len();
    eprintln!(
        "[bootstrap] filtered: {} kept of {} (forex/metals/indices/commodities only)",
        total, total_raw
    );
    if total == 0 {
        return Err(anyhow!(
            "after filtering by forex-ai asset classes, 0 symbols remain — \
             check the broker's asset class names: {:?}",
            asset_classes.iter().map(|c| &c.name).collect::<Vec<_>>()
        ));
    }

    // Rebind the loop's source to the filtered list. We keep the
    // original variable name `symbols_list.symbols` semantics so
    // the downstream `chunk` loop below doesn't need re-plumbing.
    let symbols_list = crate::app_services::ctrader_data::CTraderSymbolsListResult {
        account_id: symbols_list.account_id,
        symbols: filtered_symbols,
        archived_symbols: symbols_list.archived_symbols,
    };

    // ── Step 1c: persist the FILTERED light-symbols list ─────────
    //
    // Saved alongside the raw_batches/ folder so the Phase D.2
    // loader knows which base/quote asset ids each symbol points to
    // (the symbol_by_id response doesn't echo these — they live only
    // on the LightSymbol).
    let light_symbols_json: Vec<Value> = symbols_list
        .symbols
        .iter()
        .map(|s| {
            json!({
                "symbol_id": s.symbol_id,
                "symbol_name": s.symbol_name,
                "enabled": s.enabled,
                "description": s.description,
                "symbol_category_id": s.symbol_category_id,
                "base_asset_id": s.base_asset_id,
                "quote_asset_id": s.quote_asset_id,
            })
        })
        .collect();
    let light_path = env_dir.join("light_symbols.json");
    std::fs::write(
        &light_path,
        serde_json::to_string_pretty(&json!({ "symbols": light_symbols_json }))
            .map_err(|e| anyhow!("serialize light_symbols: {e}"))?,
    )
    .map_err(|e| anyhow!("write {}: {e}", light_path.display()))?;
    eprintln!(
        "[bootstrap] wrote {} filtered light symbols → {}",
        symbols_list.symbols.len(),
        light_path.display()
    );

    // ── Step 1d: fetch and persist the asset list ────────────────
    //
    // ProtoOAAssetListReq (2112). Joins LightSymbol.{base,quote}AssetId
    // to a 3-letter currency code (EUR/USD/XAU/...). The D.2 loader
    // needs this to populate `SymbolMetadata.base` and `.quote` from
    // broker-canonical strings — NO hand-rolled asset_id_to_currency
    // table.
    let asset_responses = transport.send_sequence(&[
        build_application_auth_request(
            &ctrader.client_id,
            &ctrader.client_secret,
            "bootstrap-asset-auth",
        ),
        build_account_auth_request(
            account_id,
            &token_bundle.access_token,
            "bootstrap-asset-acct",
        ),
        build_asset_list_request(account_id, "bootstrap-asset-list"),
    ])?;
    if asset_responses.len() < 3 {
        return Err(anyhow!(
            "asset list: expected 3 responses, got {}",
            asset_responses.len()
        ));
    }
    let assets = parse_asset_list_response(&asset_responses[2])?;
    let assets_json: Vec<Value> = assets
        .iter()
        .map(|a| {
            json!({
                "asset_id": a.asset_id,
                "name": a.name,
                "display_name": a.display_name,
                "digits": a.digits,
            })
        })
        .collect();
    let asset_path = env_dir.join("asset_list.json");
    std::fs::write(
        &asset_path,
        serde_json::to_string_pretty(&json!({ "assets": assets_json }))
            .map_err(|e| anyhow!("serialize asset_list: {e}"))?,
    )
    .map_err(|e| anyhow!("write {}: {e}", asset_path.display()))?;
    eprintln!(
        "[bootstrap] wrote {} broker assets → {}",
        assets.len(),
        asset_path.display()
    );

    // ── Step 2: chunked symbol_by_id round-trips ─────────────────
    // We batch IDs to amortise the per-WSS-connection auth overhead.
    // cTrader accepts up to ~100 IDs per request in practice, but we
    // stay conservative at 50 to leave headroom for unusually large
    // responses (e.g. symbols with long holiday lists).
    let mut index_entries: Vec<Value> = Vec::with_capacity(total);
    let mut batch_idx: usize = 0;
    for chunk in symbols_list.symbols.chunks(SYMBOLS_PER_BATCH) {
        let symbol_ids: Vec<i64> = chunk.iter().map(|s| s.symbol_id).collect();
        let batch_label = format!("batch-{batch_idx:03}");
        eprintln!(
            "[bootstrap] {} : requesting {} symbols (ids: {:?}...{:?})",
            batch_label,
            symbol_ids.len(),
            symbol_ids.first(),
            symbol_ids.last()
        );

        // Each batch opens a fresh WSS connection (same constraint as
        // resolve_symbol_with_transport) so we re-auth at the head.
        let responses = transport.send_sequence(&[
            build_application_auth_request(
                &ctrader.client_id,
                &ctrader.client_secret,
                format!("{batch_label}-app-auth"),
            ),
            build_account_auth_request(
                account_id,
                &token_bundle.access_token,
                format!("{batch_label}-acct-auth"),
            ),
            build_symbol_by_id_request(account_id, &symbol_ids, batch_label.clone()),
        ])?;
        if responses.len() < 3 {
            // Walk the partial set to surface the broker's actual error
            // code (CH_ACCESS_TOKEN_INVALID etc.) instead of an opaque
            // count mismatch.
            for r in &responses {
                if let Ok(env) = parse_open_api_envelope(r) {
                    if env.payload_type
                        != CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE
                        && env.payload_type != CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE
                    {
                        return Err(anyhow!(
                            "{batch_label}: cTrader returned non-success payload \
                             type {} — likely auth or quota error. Raw: {}",
                            env.payload_type,
                            r
                        ));
                    }
                }
            }
            return Err(anyhow!(
                "{batch_label}: expected 3 cTrader responses, got {}",
                responses.len()
            ));
        }

        // Persist the verbatim envelope for audit + future re-parsing.
        let raw_payload = &responses[2];
        let raw_pretty = match serde_json::from_str::<Value>(raw_payload) {
            Ok(v) => serde_json::to_string_pretty(&v).unwrap_or_else(|_| raw_payload.clone()),
            Err(_) => raw_payload.clone(),
        };
        let raw_path = raw_dir.join(format!("batch_{batch_idx:03}.json"));
        std::fs::write(&raw_path, &raw_pretty)
            .map_err(|e| anyhow!("write {}: {e}", raw_path.display()))?;

        // Verify the parser accepts this envelope before we commit it to
        // the index (catches enum-encoding drift early — same regression
        // guard that motivated commit e60972ad).
        let parsed = parse_symbol_by_id_response(raw_payload)
            .map_err(|e| anyhow!("{batch_label}: parser rejected broker payload: {e}"))?;
        for s in &parsed {
            // Map symbol_id → name via the light-symbols list (the
            // detail response doesn't echo the name).
            let name = chunk
                .iter()
                .find(|ls| ls.symbol_id == s.symbol_id)
                .map(|ls| ls.symbol_name.clone())
                .unwrap_or_else(|| format!("sym#{}", s.symbol_id));
            index_entries.push(json!({
                "symbol_id": s.symbol_id,
                "symbol_name": name,
                "batch_file": format!("raw_batches/batch_{batch_idx:03}.json"),
            }));
        }
        eprintln!(
            "[bootstrap] {} : parsed {} symbols, wrote {}",
            batch_label,
            parsed.len(),
            raw_path.display()
        );
        batch_idx += 1;
    }

    // ── Step 3: top-level index + metadata ───────────────────────
    let index_path = env_dir.join("symbol_index.json");
    let index_json = serde_json::to_string_pretty(&json!({
        "symbols": index_entries,
    }))
    .map_err(|e| anyhow!("serialize symbol_index: {e}"))?;
    std::fs::write(&index_path, &index_json)
        .map_err(|e| anyhow!("write {}: {e}", index_path.display()))?;

    let meta_path = env_dir.join("bootstrap_meta.json");
    let meta = json!({
        "schema_version": 1u32,
        "captured_at_unix_ms": chrono::Utc::now().timestamp_millis(),
        "environment": format!("{:?}", environment),
        "account_id": account_id,
        "symbol_count": index_entries.len(),
        "batch_count": batch_idx,
        "symbols_per_batch": SYMBOLS_PER_BATCH,
        "proto_source": "https://github.com/spotware/openapi-proto-messages",
    });
    std::fs::write(
        &meta_path,
        serde_json::to_string_pretty(&meta)
            .map_err(|e| anyhow!("serialize bootstrap_meta: {e}"))?,
    )
    .map_err(|e| anyhow!("write {}: {e}", meta_path.display()))?;

    eprintln!(
        "[bootstrap] done: {} symbols across {} batches → {}",
        index_entries.len(),
        batch_idx,
        env_dir.display()
    );

    // ── Phase D.2b — emit the canonical SymbolMetadataTable ──────
    //
    // Convert the raw broker artefacts into the
    // `neoethos_core::symbol_metadata::SymbolMetadataTable` shape
    // that `resolve(symbol)` consumes everywhere in the cost model
    // + risk gates. Write to `data/symbol_metadata.json` which is
    // the canonical path the existing `global_table()` loader picks
    // up automatically (no operator action needed).
    //
    // After this point the cost model has real broker commission +
    // swap + fee data for all 92 filtered FX/Metals/Indices/Energies
    // symbols, with NO synthetic fallbacks. Phase D.2c/D.2d delete
    // the now-unreachable fallback code paths.
    let metadata_table = build_symbol_metadata_table_from_catalog(&env_dir)
        .map_err(|e| anyhow!("D.2b converter failed: {e}"))?;
    let metadata_path = std::path::PathBuf::from("data").join("symbol_metadata.json");
    metadata_table
        .save_to_disk(&metadata_path)
        .map_err(|e| anyhow!("write {}: {e}", metadata_path.display()))?;
    eprintln!(
        "[bootstrap] wrote SymbolMetadataTable ({} entries) → {}",
        metadata_table.entries.len(),
        metadata_path.display()
    );

    Ok(())
}

/// Derive the on-disk environment label (`"demo"` / `"live"`) from
/// the configured broker environment. Used by both
/// `--bootstrap-broker-catalog` and the future bridge auto-refresh.
pub fn env_label_from_settings() -> &'static str {
    use crate::app_services::broker_config::CTraderBrokerEnvironment;
    match load_broker_settings().ctrader.environment {
        CTraderBrokerEnvironment::Demo => "demo",
        CTraderBrokerEnvironment::Live => "live",
    }
}

/// **Phase D.2b (2026-05-28)** — convert the broker catalog produced
/// by `run_bootstrap` into a `SymbolMetadataTable` consumable by the
/// `neoethos_core::symbol_metadata::resolve` path used throughout
/// the cost model + risk gates.
///
/// Inputs (read from `<env_dir>/`):
///   - `light_symbols.json`  — 92 filtered symbols with base/quote
///                             asset ids + category id
///   - `asset_list.json`     — 2400 entries mapping asset_id → name
///                             ("EUR", "USD", "XAU", "WTI", ...)
///   - `raw_batches/batch_NNN.json` — full `ProtoOASymbolByIdRes`
///                             envelopes; re-parsed via
///                             `parse_symbol_by_id_response`
///
/// Output: a `SymbolMetadataTable` with one entry per symbol, fully
/// populated from broker data. Fields:
///   - `symbol`            : broker's symbol name (e.g. "EURUSD")
///   - `base` / `quote`    : looked up via asset_list join
///   - `pip_size`          : `10^-pip_position`
///   - `contract_size`     : `lot_size / 100` (base units per lot)
///   - `pip_value_quote`   : pre-computed product
///   - `digits`            : broker's `digits` field
///   - `min_lot` / `max_lot` / `lot_step`: derived from
///                             `{min,max,step}_volume / lot_size`
///   - `typical_price`     : `None` — operator must use the live
///                             tick stream, NOT a stale baked value
///   - `typical_spread_pips`: `None` — live tick is the source
///   - `commission_per_lot`: `None` — cost model derives at runtime
///                             via `SymbolFinancials::commission_rate_decimal()`
///                             + live spot (see Phase C cost model)
///   - `daily_swap_long_pips` / `daily_swap_short_pips`: from
///                             `SymbolFinancials::daily_swap_*()`
///                             when `swap_calculation_type == Pips`
///                             (proto default; verified against 92/92
///                             of our filtered catalog)
///   - `pnl_conversion_fee_rate`: proto i32 (`1 = 0.01%`) converted
///                             to fraction (`/ 10_000.0`)
///
/// Symbols where the broker omits required fields (no `lot_size`,
/// no `digits`) are silently SKIPPED — the broker has authoritative
/// data for the 92 filtered symbols, so a skip indicates we filtered
/// something we shouldn't have, not legitimately-missing data. The
/// total kept/skipped count is logged for the operator.
pub fn build_symbol_metadata_table_from_catalog(
    env_dir: &Path,
) -> Result<neoethos_core::symbol_metadata::SymbolMetadataTable> {
    use neoethos_core::symbol_metadata::{SymbolMetadata, SymbolMetadataTable};
    use std::collections::HashMap;

    // ── Load the three on-disk artefacts ─────────────────────────
    let assets_raw = std::fs::read_to_string(env_dir.join("asset_list.json"))
        .map_err(|e| anyhow!("read asset_list.json: {e}"))?;
    let assets_doc: Value = serde_json::from_str(&assets_raw)
        .map_err(|e| anyhow!("parse asset_list.json: {e}"))?;
    let mut asset_lookup: HashMap<i64, String> = HashMap::new();
    if let Some(arr) = assets_doc.get("assets").and_then(|v| v.as_array()) {
        for entry in arr {
            let id = entry.get("asset_id").and_then(|v| v.as_i64());
            let name = entry.get("name").and_then(|v| v.as_str());
            if let (Some(id), Some(name)) = (id, name) {
                if !name.is_empty() {
                    asset_lookup.insert(id, name.to_string());
                }
            }
        }
    }
    eprintln!("[convert] asset_list: {} entries", asset_lookup.len());

    let light_raw = std::fs::read_to_string(env_dir.join("light_symbols.json"))
        .map_err(|e| anyhow!("read light_symbols.json: {e}"))?;
    let light_doc: Value = serde_json::from_str(&light_raw)
        .map_err(|e| anyhow!("parse light_symbols.json: {e}"))?;
    let mut light_lookup: HashMap<i64, (String, Option<i64>, Option<i64>)> = HashMap::new();
    if let Some(arr) = light_doc.get("symbols").and_then(|v| v.as_array()) {
        for entry in arr {
            let sid = entry.get("symbol_id").and_then(|v| v.as_i64());
            let name = entry
                .get("symbol_name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());
            let base = entry.get("base_asset_id").and_then(|v| v.as_i64());
            let quote = entry.get("quote_asset_id").and_then(|v| v.as_i64());
            if let (Some(sid), Some(name)) = (sid, name) {
                light_lookup.insert(sid, (name, base, quote));
            }
        }
    }
    eprintln!("[convert] light_symbols: {} entries", light_lookup.len());

    // ── Iterate the raw batches and convert each entry ───────────
    let raw_dir = env_dir.join("raw_batches");
    let mut batch_files: Vec<_> = std::fs::read_dir(&raw_dir)
        .map_err(|e| anyhow!("read raw_batches dir: {e}"))?
        .filter_map(|d| d.ok())
        .map(|d| d.path())
        .filter(|p| {
            p.extension().and_then(|s| s.to_str()) == Some("json")
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|n| n.starts_with("batch_"))
                    .unwrap_or(false)
        })
        .collect();
    batch_files.sort();

    let mut table = SymbolMetadataTable::default();
    let mut kept = 0usize;
    let mut skipped: Vec<String> = Vec::new();

    for path in &batch_files {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("read {}: {e}", path.display()))?;
        let parsed = parse_symbol_by_id_response(&raw)
            .map_err(|e| anyhow!("parse {}: {e}", path.display()))?;
        for sym in parsed {
            let Some((name, base_id, quote_id)) = light_lookup.get(&sym.symbol_id) else {
                skipped.push(format!("sym#{} (no light entry)", sym.symbol_id));
                continue;
            };
            // pip_size from broker's pip_position. Proto convention:
            //   pip_position == 4 → pip = 1e-4 = 0.0001 (5-digit FX)
            //   pip_position == 2 → pip = 1e-2 = 0.01   (JPY pairs, indices)
            //   pip_position == 1 → pip = 1e-1 = 0.1    (BTC)
            // Defensive: clamp pip_position to a sane range (the
            // Phase B audit already showed garbage in the synthetic
            // path can poison `10f64.powi`).
            if !(-10..=10).contains(&sym.pip_position) {
                skipped.push(format!("{name} (bad pip_position {})", sym.pip_position));
                continue;
            }
            let pip_size = 10f64.powi(-sym.pip_position);
            // contract_size: lot_size proto field is in centi-units
            // of the base currency. For EURUSD lot_size = 10_000_000
            // = 100,000 EUR × 100 cents/EUR. So:
            //   contract_size (base units per lot) = lot_size / 100.0
            let Some(lot_size_cents) = sym.lot_size.filter(|&v| v > 0) else {
                skipped.push(format!("{name} (no lot_size)"));
                continue;
            };
            let contract_size = lot_size_cents as f64 / 100.0;
            let pip_value_quote = pip_size * contract_size;

            // base/quote currency names from the asset registry.
            // For indices (where base is the index's synthetic asset)
            // we keep whatever the broker recorded — cost model
            // doesn't need a real ccy code for `base` when the
            // commission_type is `PercentOfValue` (indices use
            // notional × percent, not base/quote arithmetic).
            let base_name = base_id
                .and_then(|id| asset_lookup.get(&id).cloned())
                .unwrap_or_else(|| String::new());
            let quote_name = quote_id
                .and_then(|id| asset_lookup.get(&id).cloned())
                .unwrap_or_else(|| String::new());

            // Lot constraints. The broker reports min_volume,
            // max_volume, step_volume in the SAME cents-of-base units
            // as lot_size. So lot count = volume / lot_size.
            let min_lot = sym
                .min_volume
                .map(|v| v as f64 / lot_size_cents as f64)
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(0.01);
            let max_lot = sym
                .max_volume
                .map(|v| v as f64 / lot_size_cents as f64)
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(100.0);
            let lot_step = sym
                .step_volume
                .map(|v| v as f64 / lot_size_cents as f64)
                .filter(|v| v.is_finite() && *v > 0.0)
                .unwrap_or(0.01);

            // Swap conversion: only honour PIPS-typed swap. Other
            // calc types (PERCENTAGE/POINTS) would need additional
            // per-symbol math; bootstrap leaves them None so the
            // cost model can fail loud rather than apply the wrong
            // formula. The 92/92 filtered catalog uses PIPS so this
            // is safe for forex-ai today.
            let (daily_swap_long, daily_swap_short) =
                if let Some(fin) = sym.financials.as_ref() {
                    use crate::app_services::ctrader_data::SwapCalculationType;
                    let is_pips = fin
                        .swap_calculation_type
                        .map(|k| matches!(k, SwapCalculationType::Pips))
                        .unwrap_or(true); // None defaults to PIPS per proto
                    if is_pips {
                        (fin.daily_swap_long(), fin.daily_swap_short())
                    } else {
                        (None, None)
                    }
                } else {
                    (None, None)
                };

            // pnl_conversion_fee_rate: proto stores `1 = 0.01%`.
            // We carry a fraction (`0.0001`) so cost_model multiplies
            // directly. Filter NaN-shaped values from the conversion.
            let pnl_conv_fee = sym
                .financials
                .as_ref()
                .and_then(|f| f.pnl_conversion_fee_rate)
                .map(|raw| raw as f64 / 10_000.0)
                .filter(|v| v.is_finite() && *v >= 0.0 && *v < 1.0);

            let meta = SymbolMetadata {
                symbol: name.clone(),
                base: base_name,
                quote: quote_name,
                pip_size,
                contract_size,
                pip_value_quote,
                digits: sym.digits.max(0) as u32,
                min_lot,
                max_lot,
                lot_step,
                typical_price: None,           // live tick is source of truth
                typical_spread_pips: None,     // live tick is source of truth
                commission_per_lot: None,      // cost model derives at runtime
                daily_swap_long_pips: daily_swap_long,
                daily_swap_short_pips: daily_swap_short,
                pnl_conversion_fee_rate: pnl_conv_fee,
            };
            table.upsert(meta);
            kept += 1;
        }
    }

    eprintln!(
        "[convert] kept {} symbols, skipped {} ({})",
        kept,
        skipped.len(),
        if skipped.is_empty() {
            "none".to_string()
        } else {
            skipped.iter().take(5).cloned().collect::<Vec<_>>().join("; ")
        }
    );

    if kept == 0 {
        return Err(anyhow!(
            "converter produced 0 SymbolMetadata entries — broker catalog at \
             {} likely missing or empty",
            env_dir.display()
        ));
    }

    Ok(table)
}
