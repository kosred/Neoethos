#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_historical_admission.rs"]
pub(crate) mod ctrader_historical_admission;

#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_money.rs"]
pub(crate) mod ctrader_money;

#[path = "../../../../crates/neoethos-app/src/app_services/ctrader_messages.rs"]
pub mod ctrader_messages;

/// The application owns process-wide rustls provider installation. The leaf
/// harness deliberately substitutes only that side effect; it compiles the
/// real DNS/TCP/TLS/WebSocket transport and never opens a network connection.
pub(crate) mod ctrader_tls {
    pub(crate) fn ensure_ctrader_rustls_provider() {}
}
