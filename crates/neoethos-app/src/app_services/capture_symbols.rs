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
    parse_symbol_by_id_response, parse_symbols_list_response, CTraderSymbolInfo,
};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;
use crate::app_services::ctrader_messages::{
    CTraderOpenApiTransport, ProductionCTraderOpenApiTransport, build_account_auth_request,
    build_application_auth_request, build_symbol_by_id_request, build_symbols_list_request,
};
use crate::app_services::secure_store::production_ctrader_token_store;

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
