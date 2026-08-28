#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_auth.rs"]
pub(crate) mod ctrader_auth;

// This leaf binary intentionally path-compiles only the production cTrader
// slice. Items used by the rest of neoethos-app are unreachable here, so
// suppress only topology-induced dead-code diagnostics inside those modules.
#[allow(dead_code)]
#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_data.rs"]
pub(crate) mod ctrader_data;

#[allow(dead_code)]
#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_historical_admission.rs"]
pub(crate) mod ctrader_historical_admission;

#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_historical_page.rs"]
pub(crate) mod ctrader_historical_page;

#[allow(dead_code)]
#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_live_auth.rs"]
pub(crate) mod ctrader_live_auth;

#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_messages.rs"]
pub(crate) mod ctrader_messages;

#[allow(dead_code)]
#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_money.rs"]
pub(crate) mod ctrader_money;

#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_tick_delta.rs"]
pub(crate) mod ctrader_tick_delta;

#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_tls.rs"]
pub(crate) mod ctrader_tls;

#[allow(dead_code)]
#[path = "../../../../crates/neoethos-app/src/app_services/secure_store.rs"]
pub(crate) mod secure_store;
