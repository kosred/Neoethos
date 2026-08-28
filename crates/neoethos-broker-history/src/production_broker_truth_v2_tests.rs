use std::sync::{Arc, Mutex};

use anyhow::Result;
use neoethos_broker_truth::{BrokerFinancialTruthBindingV1, EvidenceWindowV1};
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalTimeframe,
};
use serde_json::json;

use crate::broker_truth_capture::{
    BrokerFinancialTruthCaptureRequestV2, ExactConversionRouteCaptureRequestV2,
    ExactQuoteInstrumentV2,
};
use crate::broker_truth_ctrader::CTraderBrokerTruthSameSessionV2;
use crate::ctrader_messages::CTraderOpenApiJsonMessage;
use crate::production_broker_truth_v2::{
    CTraderBrokerTruthAuthenticationWireV2, OpaqueAuthenticationFailureV2,
    PRODUCTION_BROKER_TRUTH_DEAL_MAX_ROWS_V2, PRODUCTION_BROKER_TRUTH_RETURN_PROTECTION_ORDERS_V2,
    ProductionBrokerTruthCancelResultV2, ProductionBrokerTruthCancellationV2,
    ProductionBrokerTruthCaptureErrorCodeV2, ProductionBrokerTruthCaptureStageV2,
    establish_exact_authenticated_session_v2, validate_exact_production_request_binding_v2,
};
use crate::service::BrokerEnvironment;

const ACCOUNT_ID: i64 = 7;
const SYMBOL_ID: i64 = 42;
const FROM_MS: i64 = 1_700_000_000_000;
const TO_MS: i64 = FROM_MS + 60_000;
const SECRET_SENTINEL: &str = "must-never-cross-the-private-wire";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailAt {
    Connect,
    ApplicationAuth,
    AccountAuth,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AuthenticationEvent {
    Connect {
        connection_id: u64,
        endpoint_host: String,
    },
    ApplicationAuth {
        connection_id: u64,
    },
    AccountAuth {
        connection_id: u64,
        account_id: i64,
    },
    IntoSession {
        connection_id: u64,
    },
    AdapterRpc {
        connection_id: u64,
        client_msg_id: String,
    },
}

struct SpyAuthenticationWire {
    connection_id: u64,
    events: Arc<Mutex<Vec<AuthenticationEvent>>>,
    fail_at: Option<FailAt>,
    secret_sentinel: String,
    connected: bool,
    application_authenticated: bool,
    account_authenticated: bool,
}

impl SpyAuthenticationWire {
    fn new(
        connection_id: u64,
        events: Arc<Mutex<Vec<AuthenticationEvent>>>,
        fail_at: Option<FailAt>,
    ) -> Self {
        Self {
            connection_id,
            events,
            fail_at,
            secret_sentinel: SECRET_SENTINEL.to_owned(),
            connected: false,
            application_authenticated: false,
            account_authenticated: false,
        }
    }

    fn record(&self, event: AuthenticationEvent) {
        self.events
            .lock()
            .expect("authentication event lock")
            .push(event);
    }
}

#[derive(Debug)]
struct SpyAuthenticatedSession {
    connection_id: u64,
    events: Arc<Mutex<Vec<AuthenticationEvent>>>,
}

impl CTraderBrokerTruthSameSessionV2 for SpyAuthenticatedSession {
    fn exchange_same_session(&mut self, message: &CTraderOpenApiJsonMessage) -> Result<String> {
        self.events
            .lock()
            .expect("session event lock")
            .push(AuthenticationEvent::AdapterRpc {
                connection_id: self.connection_id,
                client_msg_id: message.client_msg_id.clone(),
            });
        Ok(json!({
            "clientMsgId": message.client_msg_id,
            "payloadType": message.payload_type,
            "payload": {}
        })
        .to_string())
    }
}

impl CTraderBrokerTruthAuthenticationWireV2 for SpyAuthenticationWire {
    type Session = SpyAuthenticatedSession;

    fn connect(&mut self, endpoint_host: &str) -> Result<(), OpaqueAuthenticationFailureV2> {
        assert!(!self.connected, "wire connected more than once");
        assert!(!self.secret_sentinel.is_empty());
        self.record(AuthenticationEvent::Connect {
            connection_id: self.connection_id,
            endpoint_host: endpoint_host.to_owned(),
        });
        if self.fail_at == Some(FailAt::Connect) {
            return Err(OpaqueAuthenticationFailureV2);
        }
        self.connected = true;
        Ok(())
    }

    fn application_auth(&mut self) -> Result<(), OpaqueAuthenticationFailureV2> {
        assert!(self.connected, "application auth ran before connect");
        assert!(!self.application_authenticated);
        self.record(AuthenticationEvent::ApplicationAuth {
            connection_id: self.connection_id,
        });
        if self.fail_at == Some(FailAt::ApplicationAuth) {
            return Err(OpaqueAuthenticationFailureV2);
        }
        self.application_authenticated = true;
        Ok(())
    }

    fn exact_account_auth(
        &mut self,
        expected_account_id: i64,
    ) -> Result<(), OpaqueAuthenticationFailureV2> {
        assert!(
            self.application_authenticated,
            "account auth ran before application auth"
        );
        assert!(!self.account_authenticated);
        self.record(AuthenticationEvent::AccountAuth {
            connection_id: self.connection_id,
            account_id: expected_account_id,
        });
        if self.fail_at == Some(FailAt::AccountAuth) {
            return Err(OpaqueAuthenticationFailureV2);
        }
        self.account_authenticated = true;
        Ok(())
    }

    fn into_authenticated_session(self) -> Result<Self::Session, OpaqueAuthenticationFailureV2> {
        assert!(
            self.account_authenticated,
            "wire yielded a session before auth"
        );
        self.record(AuthenticationEvent::IntoSession {
            connection_id: self.connection_id,
        });
        Ok(SpyAuthenticatedSession {
            connection_id: self.connection_id,
            events: self.events,
        })
    }
}

fn capture_request(
    environment: CTraderEnvironment,
    server: &str,
    identity_account_id: i64,
    capture_account_id: i64,
) -> BrokerFinancialTruthCaptureRequestV2 {
    let window = EvidenceWindowV1::new(FROM_MS, TO_MS).expect("evidence window");
    let identity = CanonicalDatasetIdentity::ctrader(
        environment,
        server,
        identity_account_id,
        SYMBOL_ID,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("canonical identity");
    let binding = BrokerFinancialTruthBindingV1::new(
        &identity,
        "11".repeat(32),
        window,
        1,
        "EUR",
        2,
        "USD",
        2,
        "USD",
    )
    .expect("exact binding");
    let primary = ExactQuoteInstrumentV2::new(SYMBOL_ID, "EURUSD", 1, "EUR", 2, "USD")
        .expect("primary instrument");
    let settlement = ExactConversionRouteCaptureRequestV2::new(
        "primary_pnl_settlement",
        2,
        "USD",
        2,
        "USD",
        Vec::new(),
    )
    .expect("identity settlement route");
    BrokerFinancialTruthCaptureRequestV2::new(
        capture_account_id,
        binding,
        primary,
        vec![settlement],
    )
    .expect("capture request shape")
}

#[test]
fn one_private_wire_connects_and_authenticates_in_order_then_yields_that_session() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let cancellation = ProductionBrokerTruthCancellationV2::new();
    let mut session = establish_exact_authenticated_session_v2(
        SpyAuthenticationWire::new(91, Arc::clone(&events), None),
        "demo.ctraderapi.com",
        ACCOUNT_ID,
        &cancellation,
    )
    .expect("exact private authentication state machine");
    session
        .exchange_same_session(&CTraderOpenApiJsonMessage {
            client_msg_id: "first-adapter-rpc".to_owned(),
            payload_type: 2_101,
            payload: json!({}),
        })
        .expect("same returned session handles adapter RPC");

    assert_eq!(
        *events.lock().expect("authentication event lock"),
        vec![
            AuthenticationEvent::Connect {
                connection_id: 91,
                endpoint_host: "demo.ctraderapi.com".to_owned(),
            },
            AuthenticationEvent::ApplicationAuth { connection_id: 91 },
            AuthenticationEvent::AccountAuth {
                connection_id: 91,
                account_id: ACCOUNT_ID,
            },
            AuthenticationEvent::IntoSession { connection_id: 91 },
            AuthenticationEvent::AdapterRpc {
                connection_id: 91,
                client_msg_id: "first-adapter-rpc".to_owned(),
            },
        ]
    );
}

#[test]
fn authentication_failures_stop_at_the_exact_stage_and_never_expose_secrets() {
    for (fail_at, expected_stage, expected_code, expected_events) in [
        (
            FailAt::Connect,
            ProductionBrokerTruthCaptureStageV2::Connect,
            ProductionBrokerTruthCaptureErrorCodeV2::TransportFailed,
            1,
        ),
        (
            FailAt::ApplicationAuth,
            ProductionBrokerTruthCaptureStageV2::ApplicationAuth,
            ProductionBrokerTruthCaptureErrorCodeV2::AuthenticationFailed,
            2,
        ),
        (
            FailAt::AccountAuth,
            ProductionBrokerTruthCaptureStageV2::AccountAuth,
            ProductionBrokerTruthCaptureErrorCodeV2::AuthenticationFailed,
            3,
        ),
    ] {
        let events = Arc::new(Mutex::new(Vec::new()));
        let cancellation = ProductionBrokerTruthCancellationV2::new();
        let error = establish_exact_authenticated_session_v2(
            SpyAuthenticationWire::new(92, Arc::clone(&events), Some(fail_at)),
            "demo.ctraderapi.com",
            ACCOUNT_ID,
            &cancellation,
        )
        .expect_err("the selected authentication stage must fail closed");
        assert_eq!(error.stage(), expected_stage);
        assert_eq!(error.code(), expected_code);
        assert_eq!(
            events.lock().expect("authentication event lock").len(),
            expected_events
        );
        let rendered = format!("{error:?} {error} {}", error.detail());
        assert!(!rendered.contains(SECRET_SENTINEL));
    }
}

#[test]
fn exact_environment_server_and_account_must_match_before_connect() {
    let exact = capture_request(
        CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        ACCOUNT_ID,
        ACCOUNT_ID,
    );
    validate_exact_production_request_binding_v2(BrokerEnvironment::Demo, ACCOUNT_ID, &exact)
        .expect("exact request binding");

    for (environment, account_id, request) in [
        (BrokerEnvironment::Live, ACCOUNT_ID, exact.clone()),
        (BrokerEnvironment::Demo, ACCOUNT_ID + 1, exact),
        (
            BrokerEnvironment::Demo,
            ACCOUNT_ID,
            capture_request(
                CTraderEnvironment::Demo,
                "different.demo.endpoint",
                ACCOUNT_ID,
                ACCOUNT_ID,
            ),
        ),
        (
            BrokerEnvironment::Demo,
            ACCOUNT_ID,
            capture_request(
                CTraderEnvironment::Demo,
                "demo.ctraderapi.com",
                ACCOUNT_ID,
                ACCOUNT_ID + 1,
            ),
        ),
    ] {
        let error = validate_exact_production_request_binding_v2(environment, account_id, &request)
            .expect_err("environment/server/account mismatch must fail before connection");
        assert_eq!(
            error.stage(),
            ProductionBrokerTruthCaptureStageV2::Admission
        );
        assert_eq!(
            error.code(),
            ProductionBrokerTruthCaptureErrorCodeV2::ConfigurationMismatch
        );
    }
}

#[test]
fn cancellation_is_run_scoped_and_publication_transition_is_one_way() {
    let cancellation = ProductionBrokerTruthCancellationV2::new();
    assert_eq!(
        cancellation.cancel(),
        ProductionBrokerTruthCancelResultV2::Cancelled
    );
    let events = Arc::new(Mutex::new(Vec::new()));
    let error = establish_exact_authenticated_session_v2(
        SpyAuthenticationWire::new(93, Arc::clone(&events), None),
        "demo.ctraderapi.com",
        ACCOUNT_ID,
        &cancellation,
    )
    .expect_err("pre-connect cancellation must fail closed");
    assert_eq!(
        error.stage(),
        ProductionBrokerTruthCaptureStageV2::Admission
    );
    assert_eq!(
        error.code(),
        ProductionBrokerTruthCaptureErrorCodeV2::Cancelled
    );
    assert!(events.lock().expect("authentication event lock").is_empty());

    let publication = ProductionBrokerTruthCancellationV2::new();
    let guard = publication
        .begin_publication()
        .expect("one run-scoped publication transition");
    assert_eq!(
        publication.cancel(),
        ProductionBrokerTruthCancelResultV2::PublicationInProgress
    );
    drop(guard);
    assert_eq!(
        publication.cancel(),
        ProductionBrokerTruthCancelResultV2::PublicationInProgress
    );
}

#[test]
fn production_surface_has_fixed_capture_policy_and_no_fallback_or_global_registry() {
    assert_eq!(PRODUCTION_BROKER_TRUTH_DEAL_MAX_ROWS_V2, 100);
    assert!(PRODUCTION_BROKER_TRUTH_RETURN_PROTECTION_ORDERS_V2);

    let source = include_str!("production_broker_truth_v2.rs");
    for required in [
        "load_exact_production_broker_truth_credentials_v2",
        "ProductionCTraderOpenApiSession",
        "CTraderBrokerTruthAdapterV2::new",
        "capture_and_publish_broker_financial_truth_v2",
        "authority_receipt.manifest_sha256()",
    ] {
        assert!(
            source.contains(required),
            "missing production route {required}"
        );
    }
    for forbidden in [
        "enabled_for_execution",
        ".first()",
        ".or_else(",
        "HistoricalFetchRegistry",
        "begin_process_historical_capture",
        "OnceLock",
        "LazyLock",
        "static mut",
        "BrokerFinancialTruthPermitV1",
        "BrokerFinancialTruthCapabilityV1",
        "current.json",
        "default()",
        "println!",
        "dbg!",
    ] {
        assert!(
            !source.contains(forbidden),
            "production acquisition session contains forbidden fallback/authority token {forbidden}"
        );
    }

    let library = include_str!("lib.rs");
    for public_type in [
        "ProductionBrokerTruthCaptureRequestV2",
        "ProductionBrokerTruthCaptureOutcomeV2",
        "ProductionBrokerTruthCaptureStageV2",
        "ProductionBrokerTruthCaptureErrorCodeV2",
        "ProductionBrokerTruthCaptureErrorV2",
        "ProductionBrokerTruthCancellationV2",
        "capture_production_broker_financial_truth_v2",
    ] {
        assert!(
            library.contains(public_type),
            "missing public export {public_type}"
        );
    }
}
