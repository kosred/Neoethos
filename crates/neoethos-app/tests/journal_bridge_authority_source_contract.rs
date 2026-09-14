//! Source-level guardrails around the Journal Money V3 integration seam.
//!
//! The account bridge currently has only a bounded recent-deal window. That
//! window may continue feeding the legacy UI journal while the global broker
//! money capability is sealed, but it must never be presented as complete V3
//! authority. V3 finalization itself requires exact persisted fills plus the
//! current account reconcile snapshot.

use std::fs;
use std::path::PathBuf;

fn crate_root() -> PathBuf {
    option_env!("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/neoethos-app"))
}

fn read(relative: &str) -> String {
    let path = crate_root().join(relative);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read app source {}: {error}", path.display()))
}

fn item_body<'a>(source: &'a str, signature: &str) -> &'a str {
    let start = source
        .find(signature)
        .unwrap_or_else(|| panic!("missing production item `{signature}`"));
    let open = source[start..]
        .find('{')
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing body for production item `{signature}`"));
    let mut depth = 0_u32;
    for (offset, byte) in source.as_bytes()[open..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=open + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated production item `{signature}`")
}

#[test]
fn bounded_recent_deals_are_not_minted_into_v3_authority() {
    let bridge = read("src/server/bridge.rs");
    let reconcile = read("src/app_services/journal_reconcile.rs");
    assert!(bridge.contains("journal_reconcile::reconcile_best_effort"));
    assert!(
        !bridge.contains("journal_money_v3"),
        "the 24h/100-row bridge cannot claim complete Journal V3 authority"
    );
    assert!(reconcile.contains("runtime.recent_deals"));
    assert!(
        !reconcile.contains("ClosedPositionJournalReceiptV3"),
        "legacy recent-deal reconciliation cannot emit finalized V3 receipts"
    );
}

#[test]
fn journal_consumers_keep_their_distinct_runtime_authority_boundaries() {
    let reconcile = read("src/app_services/journal_reconcile.rs");
    let reconcile_body = item_body(&reconcile, "pub fn reconcile_best_effort(");
    let reconcile_gate = reconcile_body
        .find("current_broker_financial_truth_capability_v1")
        .expect("legacy journal reconcile must start behind the global capability");
    let local_state = reconcile_body
        .find("let Some(dir) = data_dir()")
        .expect("legacy journal local-state access");
    assert!(reconcile_gate < local_state);

    let live_gate = read("src/app_services/live_gate.rs");
    let wrapper = item_body(&live_gate, "pub fn evaluate_for_portfolio(");
    assert!(wrapper.contains("evaluate_for_portfolio_with_settings(portfolio_path, &settings)"));
    let promotion_body = item_body(&live_gate, "pub fn evaluate_for_portfolio_with_settings(");
    let disabled = promotion_body.find("if !gate_config.enabled").unwrap();
    let load = promotion_body
        .find("load_live_portfolio_json(portfolio_path)")
        .unwrap();
    let benchmark = promotion_body
        .find("refuse_unavailable_final_benchmark(&artifact)")
        .expect("enabled check must require source-scoped final benchmark evidence");
    assert!(disabled < load && load < benchmark);
    let production = live_gate.split("#[cfg(test)]").next().unwrap();
    for forbidden in [
        ".oos_metrics()",
        "fetch_broker_symbols_blocking",
        "query_closed_trades",
        "query_equity",
        "scope_to_demo_environment",
    ] {
        assert!(
            !production.contains(forbidden),
            "unavailable final benchmark must not retain a legacy substitute: {forbidden}"
        );
    }
    assert!(!promotion_body.contains("ClosedPositionJournalReceiptV3"));

    let live_trading = read("src/app_services/live_trading.rs");
    let capability = live_trading
        .find("current_broker_financial_truth_capability_v1")
        .expect("live risk must require the sealed broker capability");
    let legacy_query = live_trading
        .find("journal_store::query_closed_trades")
        .expect("current account-loss query remains explicit legacy code");
    assert!(capability < legacy_query);
}

#[test]
fn v3_finalization_uses_exact_persisted_fills_and_runtime_flat_evidence() {
    let source = read("src/app_services/journal_money_v3.rs");
    let finalize = item_body(&source, "pub fn finalize_from_account_runtime(");
    for token in [
        "load_durable_fills",
        "BrokerFlatReconcileEvidenceV3::from_account_runtime",
        "expected_entry_filled_volume_raw_centi_units",
        "persist_immutable_json",
    ] {
        assert!(
            finalize.contains(token),
            "V3 finalization is missing `{token}`"
        );
    }
    for forbidden in ["recent_deals", "position_still_open: bool", "ClosedTrade"] {
        assert!(
            !finalize.contains(forbidden),
            "V3 finalization cannot trust `{forbidden}`"
        );
    }
}

#[test]
fn no_scalar_legacy_adapter_exists_at_the_v3_boundary() {
    let source = read("src/app_services/journal_money_v3.rs");
    for forbidden in [
        "journal_store::ClosedTrade",
        "query_closed_trades",
        "net_profit",
        "gross_profit: f64",
        "commission: f64",
        "swap: f64",
    ] {
        assert!(
            !source.contains(forbidden),
            "legacy scalar journal data leaked into V3 through `{forbidden}`"
        );
    }
}
