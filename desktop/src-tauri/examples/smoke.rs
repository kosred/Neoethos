//! Headless smoke test of every NON-destructive Tauri command path.
//! Calls the EXACT functions the UI buttons invoke, so we exercise the real
//! data + broker logic without the GUI. Read-only only: it never places or
//! closes an order (financial actions need explicit per-action approval).

use std::path::PathBuf;

use neoethos_app::app_services::broker_api;
use neoethos_app::app_services::broker_persistence::load_broker_settings;
use neoethos_app::app_services::secure_store::production_ctrader_token_store;

fn data_root() -> PathBuf {
    neoethos_core::Settings::load()
        .ok()
        .map(|s| s.system.data_dir)
        .filter(|d| d.exists())
        .unwrap_or_else(|| PathBuf::from(r"C:\Users\konst\development\forex-ai\data"))
}

fn main() {
    println!("================ NeoEthos Tauri smoke test ================\n");

    // ── 1. app_info / data root ──────────────────────────────────────────
    let root = data_root();
    println!("[app_info]  version=0.5.0  data_root={}  exists={}", root.display(), root.exists());

    // ── 2. list_symbols (local vortex) ───────────────────────────────────
    match neoethos_data::discover_symbols(&root) {
        Ok(syms) => println!("[list_symbols]  OK  {} symbols: {:?}", syms.len(), &syms.iter().take(8).collect::<Vec<_>>()),
        Err(e) => println!("[list_symbols]  ERR  {e}"),
    }

    // ── 3. list_timeframes(EURUSD) + 4. chart(EURUSD,H1) ─────────────────
    let probe_sym = "EURUSD";
    match neoethos_data::discover_timeframes(&root, probe_sym) {
        Ok(tfs) => {
            println!("[list_timeframes {probe_sym}]  OK  {tfs:?}");
            let tf = if tfs.iter().any(|t| t == "H1") { "H1" } else { tfs.first().map(|s| s.as_str()).unwrap_or("H1") };
            match neoethos_data::load_symbol_timeframe(&root, probe_sym, tf) {
                Ok(o) => {
                    let n = o.close.len();
                    let last = o.close.last().copied().unwrap_or(f64::NAN);
                    let first_ts = o.timestamp.as_ref().and_then(|t| t.first()).copied().unwrap_or(0);
                    let last_ts = o.timestamp.as_ref().and_then(|t| t.last()).copied().unwrap_or(0);
                    println!("[chart {probe_sym} {tf}]  OK  {n} candles  last_close={last:.5}  span_ms=[{first_ts}..{last_ts}]");
                }
                Err(e) => println!("[chart {probe_sym} {tf}]  ERR  {e}"),
            }
        }
        Err(e) => println!("[list_timeframes {probe_sym}]  ERR  {e}"),
    }

    // ── 5. broker_status (cheap, no network) ─────────────────────────────
    let s = load_broker_settings();
    let ct = &s.ctrader;
    let configured = !ct.client_id.is_empty() && !ct.client_secret.is_empty();
    let token = production_ctrader_token_store()
        .load_token_bundle_with_legacy_fallback()
        .ok()
        .flatten();
    let has_token = token.as_ref().map(|b| !b.access_token.is_empty()).unwrap_or(false);
    let refresh_present = token.as_ref().map(|b| !b.refresh_token.is_empty()).unwrap_or(false);
    println!(
        "[broker_status]  configured={configured}  has_token={has_token}  refresh_token={refresh_present}  env={:?}  account={:?}",
        ct.environment,
        ct.accounts.first().map(|a| a.account_id.clone())
    );

    if !configured {
        println!("\n[broker]  SKIP live calls — broker not configured (no client_id/secret). Local data path verified above.");
        println!("\n================ smoke test done ================");
        return;
    }

    // ── 6. broker_symbols (live — triggers AUTO token refresh) ───────────
    println!("\n[broker_symbols]  calling (this exercises ensure_fresh_token_bundle → silent refresh)…");
    match broker_api::fetch_broker_symbols_blocking() {
        Ok(b) => println!("[broker_symbols]  OK  env={}  {} symbols (account {})", b.environment, b.symbols.len(), b.account_id),
        Err(e) => println!("[broker_symbols]  ERR  {e}"),
    }

    // ── 7. account_snapshot (live — balance/equity/positions) ────────────
    println!("\n[account_snapshot]  calling fetch_account_runtime_blocking…");
    match broker_api::fetch_account_runtime_blocking() {
        Ok(snap) => {
            println!(
                "[account_snapshot]  OK  account={}  balance={:.2}  unrealized={:.2}  equity={:.2}  open_positions={}",
                snap.trader.account_id,
                snap.trader.balance,
                snap.trader.unrealized_pnl,
                snap.trader.balance + snap.trader.unrealized_pnl,
                snap.reconcile.positions.len()
            );
            for p in snap.reconcile.positions.iter().take(5) {
                println!("    position #{} sym#{} {} vol={} @ {:?}", p.position_id, p.symbol_id, p.trade_side, p.volume, p.price);
            }
        }
        Err(e) => println!("[account_snapshot]  ERR (expected if refresh_token revoked → one-time OAuth needed)  {e}"),
    }

    // ── 8. broker_accounts (live — accounts granted by token) ────────────
    println!("\n[broker_accounts]  calling…");
    match broker_api::fetch_broker_accounts_blocking() {
        Ok(b) => {
            println!("[broker_accounts]  OK  {} accounts (scope: {})", b.accounts.len(), b.permission_scope);
            for a in b.accounts.iter().take(5) {
                println!("    {} · {} · {}", a.account_id, a.broker_title, if a.is_live == Some(true) { "LIVE" } else { "DEMO" });
            }
        }
        Err(e) => println!("[broker_accounts]  ERR  {e}"),
    }

    // ── 9. broker_chart (live trendbars from the server) ─────────────────
    println!("\n[broker_chart EURUSD H1]  calling fetch_recent_chart_bars_blocking…");
    match broker_api::fetch_recent_chart_bars_blocking("EURUSD", "H1", 50) {
        Ok(bars) => println!("[broker_chart]  OK  {} broker bars  last_close={:?}", bars.len(), bars.last().map(|b| b.close)),
        Err(e) => println!("[broker_chart]  ERR  {e}"),
    }

    // NOTE: place_order / close_position are NOT exercised — they execute real
    // trades and require explicit per-action approval. Their wiring compiles
    // and is registered; firing them is a deliberate user action.

    println!("\n================ smoke test done ================");
}
