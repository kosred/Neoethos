//! Production acquisition runner for one exact cTrader broker-truth bundle.
//!
//! This boundary only captures and content-addresses evidence. It does not
//! validate review signatures, create semantic authority, or authorize an
//! evaluator.

use std::cell::Cell;
use std::error::Error;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use neoethos_broker_truth::{
    BrokerFinancialTruthBundleReceiptV2, BrokerFinancialTruthBundleStoreV1,
    BrokerTruthAcquisitionAuthorityReceiptV1,
};
use neoethos_data::{CTraderEnvironment, CanonicalDatasetIdentity};

use crate::broker_truth_capture::{
    BrokerFinancialTruthCaptureErrorCodeV2 as EvidenceCaptureErrorCodeV2,
    BrokerFinancialTruthCaptureErrorV2 as EvidenceCaptureErrorV2,
    BrokerFinancialTruthCaptureRequestV2, capture_and_publish_broker_financial_truth_v2,
};
use crate::broker_truth_ctrader::{
    CTraderBrokerTruthAdapterV2, CTraderBrokerTruthSameSessionV2,
    ReviewedCTraderQuoteSynchronizationV2,
};
use crate::ctrader_messages::{
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE, CTraderOpenApiSessionResponse,
    ProductionCTraderOpenApiSession, ProductionCTraderOpenApiTransport, build_account_auth_request,
    build_application_auth_request, parse_open_api_envelope,
};
use crate::service::{
    BrokerEnvironment, ExactProductionBrokerTruthCredentialsV2,
    load_exact_production_broker_truth_credentials_v2,
};

pub const PRODUCTION_BROKER_TRUTH_DEAL_MAX_ROWS_V2: u32 = 100;
pub const PRODUCTION_BROKER_TRUTH_RETURN_PROTECTION_ORDERS_V2: bool = true;

const CANCELLATION_ACTIVE_V2: u8 = 0;
const CANCELLATION_CANCELLED_V2: u8 = 1;
const CANCELLATION_PUBLICATION_V2: u8 = 2;
const CANCELLATION_RUNNING_V2: u8 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductionBrokerTruthCaptureStageV2 {
    Admission,
    Credentials,
    Connect,
    ApplicationAuth,
    AccountAuth,
    Adapter,
    Publication,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductionBrokerTruthCaptureErrorCodeV2 {
    ConfigurationMismatch,
    CredentialsUnavailable,
    TransportFailed,
    AuthenticationFailed,
    CaptureFailed,
    PublicationFailed,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductionBrokerTruthCaptureErrorV2 {
    stage: ProductionBrokerTruthCaptureStageV2,
    code: ProductionBrokerTruthCaptureErrorCodeV2,
    detail: &'static str,
}

impl ProductionBrokerTruthCaptureErrorV2 {
    pub const fn stage(&self) -> ProductionBrokerTruthCaptureStageV2 {
        self.stage
    }

    pub const fn code(&self) -> ProductionBrokerTruthCaptureErrorCodeV2 {
        self.code
    }

    pub const fn detail(&self) -> &'static str {
        self.detail
    }
}

impl fmt::Display for ProductionBrokerTruthCaptureErrorV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.detail)
    }
}

impl Error for ProductionBrokerTruthCaptureErrorV2 {}

fn production_error(
    stage: ProductionBrokerTruthCaptureStageV2,
    code: ProductionBrokerTruthCaptureErrorCodeV2,
    detail: &'static str,
) -> ProductionBrokerTruthCaptureErrorV2 {
    ProductionBrokerTruthCaptureErrorV2 {
        stage,
        code,
        detail,
    }
}

/// Non-secret, exact inputs for one production evidence acquisition.
///
/// Fields remain private so callers cannot mutate the environment/account
/// binding after construction or choose transport and pagination policy.
pub struct ProductionBrokerTruthCaptureRequestV2 {
    environment: BrokerEnvironment,
    account_id: i64,
    authority_receipt: BrokerTruthAcquisitionAuthorityReceiptV1,
    capture_request: BrokerFinancialTruthCaptureRequestV2,
    reviewed_synchronizations: Vec<ReviewedCTraderQuoteSynchronizationV2>,
    capture_work_parent: PathBuf,
    store_root: PathBuf,
}

impl ProductionBrokerTruthCaptureRequestV2 {
    pub fn new(
        environment: BrokerEnvironment,
        account_id: i64,
        authority_receipt: BrokerTruthAcquisitionAuthorityReceiptV1,
        capture_request: BrokerFinancialTruthCaptureRequestV2,
        reviewed_synchronizations: Vec<ReviewedCTraderQuoteSynchronizationV2>,
        capture_work_parent: impl Into<PathBuf>,
        store_root: impl Into<PathBuf>,
    ) -> Result<Self, ProductionBrokerTruthCaptureErrorV2> {
        validate_exact_production_request_binding_v2(environment, account_id, &capture_request)?;
        authority_receipt.canonical_json_bytes().map_err(|_| {
            production_error(
                ProductionBrokerTruthCaptureStageV2::Admission,
                ProductionBrokerTruthCaptureErrorCodeV2::ConfigurationMismatch,
                "broker-truth authority receipt is invalid",
            )
        })?;
        if reviewed_synchronizations.is_empty() {
            return Err(production_error(
                ProductionBrokerTruthCaptureStageV2::Admission,
                ProductionBrokerTruthCaptureErrorCodeV2::ConfigurationMismatch,
                "reviewed quote synchronization evidence is missing",
            ));
        }
        let capture_work_parent = capture_work_parent.into();
        let store_root = store_root.into();
        validate_explicit_acquisition_paths(&capture_work_parent, &store_root)?;
        Ok(Self {
            environment,
            account_id,
            authority_receipt,
            capture_request,
            reviewed_synchronizations,
            capture_work_parent,
            store_root,
        })
    }
}

pub struct ProductionBrokerTruthCaptureOutcomeV2 {
    receipt: BrokerFinancialTruthBundleReceiptV2,
}

impl ProductionBrokerTruthCaptureOutcomeV2 {
    pub const fn receipt(&self) -> &BrokerFinancialTruthBundleReceiptV2 {
        &self.receipt
    }
}

fn validate_explicit_acquisition_paths(
    capture_work_parent: &Path,
    store_root: &Path,
) -> Result<(), ProductionBrokerTruthCaptureErrorV2> {
    if !capture_work_parent.is_absolute()
        || !store_root.is_absolute()
        || capture_work_parent == store_root
        || capture_work_parent.starts_with(store_root)
        || store_root.starts_with(capture_work_parent)
    {
        return Err(production_error(
            ProductionBrokerTruthCaptureStageV2::Admission,
            ProductionBrokerTruthCaptureErrorCodeV2::ConfigurationMismatch,
            "capture work parent and immutable store root must be explicit disjoint absolute paths",
        ));
    }
    Ok(())
}

pub(crate) fn validate_exact_production_request_binding_v2(
    environment: BrokerEnvironment,
    account_id: i64,
    request: &BrokerFinancialTruthCaptureRequestV2,
) -> Result<(), ProductionBrokerTruthCaptureErrorV2> {
    if account_id <= 0 || request.account_id() != account_id {
        return Err(configuration_mismatch());
    }
    let canonical_environment = match environment {
        BrokerEnvironment::Demo => CTraderEnvironment::Demo,
        BrokerEnvironment::Live => CTraderEnvironment::Live,
    };
    let actual_identity = request.binding().canonical_dataset_identity();
    let expected_identity = CanonicalDatasetIdentity::ctrader(
        canonical_environment,
        environment.endpoint_host(),
        account_id,
        request.primary_instrument().symbol_id(),
        request.primary_instrument().symbol_name(),
        actual_identity.timeframe(),
        actual_identity.bar_timestamp_convention(),
    )
    .map_err(|_| configuration_mismatch())?;
    if actual_identity != &expected_identity {
        return Err(configuration_mismatch());
    }
    Ok(())
}

fn configuration_mismatch() -> ProductionBrokerTruthCaptureErrorV2 {
    production_error(
        ProductionBrokerTruthCaptureStageV2::Admission,
        ProductionBrokerTruthCaptureErrorCodeV2::ConfigurationMismatch,
        "production request does not match its exact cTrader dataset binding",
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductionBrokerTruthCancelResultV2 {
    Cancelled,
    PublicationInProgress,
}

/// One lexical cancellation state for one acquisition call.
pub struct ProductionBrokerTruthCancellationV2 {
    state: Arc<AtomicU8>,
}

impl ProductionBrokerTruthCancellationV2 {
    pub fn new() -> Self {
        Self {
            state: Arc::new(AtomicU8::new(CANCELLATION_ACTIVE_V2)),
        }
    }

    pub fn cancel(&self) -> ProductionBrokerTruthCancelResultV2 {
        loop {
            let state = self.state.load(Ordering::Acquire);
            match state {
                CANCELLATION_CANCELLED_V2 => {
                    return ProductionBrokerTruthCancelResultV2::Cancelled;
                }
                CANCELLATION_PUBLICATION_V2 => {
                    return ProductionBrokerTruthCancelResultV2::PublicationInProgress;
                }
                CANCELLATION_ACTIVE_V2 | CANCELLATION_RUNNING_V2 => {
                    if self
                        .state
                        .compare_exchange(
                            state,
                            CANCELLATION_CANCELLED_V2,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return ProductionBrokerTruthCancelResultV2::Cancelled;
                    }
                }
                _ => return ProductionBrokerTruthCancelResultV2::PublicationInProgress,
            }
        }
    }

    fn is_cancelled(&self) -> bool {
        self.state.load(Ordering::Acquire) == CANCELLATION_CANCELLED_V2
    }

    fn begin_run(&self) -> Result<(), ProductionBrokerTruthCaptureErrorV2> {
        self.state
            .compare_exchange(
                CANCELLATION_ACTIVE_V2,
                CANCELLATION_RUNNING_V2,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|state| {
                if state == CANCELLATION_CANCELLED_V2 {
                    production_error(
                        ProductionBrokerTruthCaptureStageV2::Admission,
                        ProductionBrokerTruthCaptureErrorCodeV2::Cancelled,
                        "broker-truth acquisition was cancelled",
                    )
                } else {
                    production_error(
                        ProductionBrokerTruthCaptureStageV2::Admission,
                        ProductionBrokerTruthCaptureErrorCodeV2::ConfigurationMismatch,
                        "broker-truth cancellation state cannot be reused across runs",
                    )
                }
            })
    }

    pub(crate) fn begin_publication(
        &self,
    ) -> Result<ProductionBrokerTruthPublicationGuardV2, ProductionBrokerTruthCaptureErrorV2> {
        loop {
            let state = self.state.load(Ordering::Acquire);
            match state {
                CANCELLATION_ACTIVE_V2 | CANCELLATION_RUNNING_V2 => {
                    if self
                        .state
                        .compare_exchange(
                            state,
                            CANCELLATION_PUBLICATION_V2,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return Ok(ProductionBrokerTruthPublicationGuardV2 {
                            state: Arc::clone(&self.state),
                        });
                    }
                }
                CANCELLATION_CANCELLED_V2 => {
                    return Err(production_error(
                        ProductionBrokerTruthCaptureStageV2::Publication,
                        ProductionBrokerTruthCaptureErrorCodeV2::Cancelled,
                        "broker-truth acquisition was cancelled before publication",
                    ));
                }
                _ => {
                    return Err(production_error(
                        ProductionBrokerTruthCaptureStageV2::Publication,
                        ProductionBrokerTruthCaptureErrorCodeV2::PublicationFailed,
                        "broker-truth publication transition was already consumed",
                    ));
                }
            }
        }
    }
}

#[must_use = "the publication guard keeps the acquisition in its irreversible publication state"]
pub(crate) struct ProductionBrokerTruthPublicationGuardV2 {
    state: Arc<AtomicU8>,
}

impl Drop for ProductionBrokerTruthPublicationGuardV2 {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.state.load(Ordering::Acquire),
            CANCELLATION_PUBLICATION_V2
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OpaqueAuthenticationFailureV2;

pub(crate) trait CTraderBrokerTruthAuthenticationWireV2 {
    type Session: CTraderBrokerTruthSameSessionV2;

    fn connect(&mut self, endpoint_host: &str) -> Result<(), OpaqueAuthenticationFailureV2>;

    fn application_auth(&mut self) -> Result<(), OpaqueAuthenticationFailureV2>;

    fn exact_account_auth(
        &mut self,
        expected_account_id: i64,
    ) -> Result<(), OpaqueAuthenticationFailureV2>;

    fn into_authenticated_session(self) -> Result<Self::Session, OpaqueAuthenticationFailureV2>;
}

#[cfg(test)]
pub(crate) fn establish_exact_authenticated_session_v2<W>(
    wire: W,
    endpoint_host: &str,
    expected_account_id: i64,
    cancellation: &ProductionBrokerTruthCancellationV2,
) -> Result<W::Session, ProductionBrokerTruthCaptureErrorV2>
where
    W: CTraderBrokerTruthAuthenticationWireV2,
{
    cancellation.begin_run()?;
    establish_exact_authenticated_session_after_run_claimed_v2(
        wire,
        endpoint_host,
        expected_account_id,
        cancellation,
    )
}

fn establish_exact_authenticated_session_after_run_claimed_v2<W>(
    mut wire: W,
    endpoint_host: &str,
    expected_account_id: i64,
    cancellation: &ProductionBrokerTruthCancellationV2,
) -> Result<W::Session, ProductionBrokerTruthCaptureErrorV2>
where
    W: CTraderBrokerTruthAuthenticationWireV2,
{
    ensure_not_cancelled_at(cancellation, ProductionBrokerTruthCaptureStageV2::Admission)?;
    wire.connect(endpoint_host).map_err(|_| {
        production_error(
            ProductionBrokerTruthCaptureStageV2::Connect,
            ProductionBrokerTruthCaptureErrorCodeV2::TransportFailed,
            "cTrader transport connection failed",
        )
    })?;
    ensure_not_cancelled_at(
        cancellation,
        ProductionBrokerTruthCaptureStageV2::ApplicationAuth,
    )?;
    wire.application_auth().map_err(|_| {
        production_error(
            ProductionBrokerTruthCaptureStageV2::ApplicationAuth,
            ProductionBrokerTruthCaptureErrorCodeV2::AuthenticationFailed,
            "cTrader application authentication failed",
        )
    })?;
    ensure_not_cancelled_at(
        cancellation,
        ProductionBrokerTruthCaptureStageV2::AccountAuth,
    )?;
    wire.exact_account_auth(expected_account_id).map_err(|_| {
        production_error(
            ProductionBrokerTruthCaptureStageV2::AccountAuth,
            ProductionBrokerTruthCaptureErrorCodeV2::AuthenticationFailed,
            "cTrader exact account authentication failed",
        )
    })?;
    ensure_not_cancelled_at(cancellation, ProductionBrokerTruthCaptureStageV2::Adapter)?;
    wire.into_authenticated_session().map_err(|_| {
        production_error(
            ProductionBrokerTruthCaptureStageV2::AccountAuth,
            ProductionBrokerTruthCaptureErrorCodeV2::AuthenticationFailed,
            "cTrader authenticated session was unavailable",
        )
    })
}

fn ensure_not_cancelled_at(
    cancellation: &ProductionBrokerTruthCancellationV2,
    stage: ProductionBrokerTruthCaptureStageV2,
) -> Result<(), ProductionBrokerTruthCaptureErrorV2> {
    if cancellation.is_cancelled() {
        return Err(production_error(
            stage,
            ProductionBrokerTruthCaptureErrorCodeV2::Cancelled,
            "broker-truth acquisition was cancelled",
        ));
    }
    Ok(())
}

struct ProductionCTraderBrokerTruthAuthenticationWireV2 {
    credentials: ExactProductionBrokerTruthCredentialsV2,
    client_message_namespace: String,
    session: Option<ProductionCTraderOpenApiSession>,
    application_authenticated: bool,
    account_authenticated: bool,
}

impl ProductionCTraderBrokerTruthAuthenticationWireV2 {
    fn new(
        credentials: ExactProductionBrokerTruthCredentialsV2,
        client_message_namespace: String,
    ) -> Self {
        Self {
            credentials,
            client_message_namespace,
            session: None,
            application_authenticated: false,
            account_authenticated: false,
        }
    }

    fn session_mut(
        &mut self,
    ) -> Result<&mut ProductionCTraderOpenApiSession, OpaqueAuthenticationFailureV2> {
        self.session.as_mut().ok_or(OpaqueAuthenticationFailureV2)
    }

    fn exchange_auth_message(
        &mut self,
        message: &crate::ctrader_messages::CTraderOpenApiJsonMessage,
    ) -> Result<String, OpaqueAuthenticationFailureV2> {
        match self
            .session_mut()?
            .send_one(message, None)
            .map_err(|_| OpaqueAuthenticationFailureV2)?
        {
            CTraderOpenApiSessionResponse::Expected(response) => Ok(response),
            CTraderOpenApiSessionResponse::BrokerError(_) => Err(OpaqueAuthenticationFailureV2),
        }
    }
}

impl CTraderBrokerTruthAuthenticationWireV2 for ProductionCTraderBrokerTruthAuthenticationWireV2 {
    type Session = ProductionCTraderOpenApiSession;

    fn connect(&mut self, endpoint_host: &str) -> Result<(), OpaqueAuthenticationFailureV2> {
        if self.session.is_some() || endpoint_host != self.credentials.environment().endpoint_host()
        {
            return Err(OpaqueAuthenticationFailureV2);
        }
        let transport = ProductionCTraderOpenApiTransport::new(endpoint_host);
        let session = transport
            .connect_session(None)
            .map_err(|_| OpaqueAuthenticationFailureV2)?;
        self.session = Some(session);
        Ok(())
    }

    fn application_auth(&mut self) -> Result<(), OpaqueAuthenticationFailureV2> {
        if self.application_authenticated || self.account_authenticated {
            return Err(OpaqueAuthenticationFailureV2);
        }
        let message = build_application_auth_request(
            self.credentials.client_id(),
            self.credentials.client_secret(),
            format!("{}-application-auth", self.client_message_namespace),
        );
        let response = self.exchange_auth_message(&message)?;
        let envelope =
            parse_open_api_envelope(&response).map_err(|_| OpaqueAuthenticationFailureV2)?;
        if envelope.payload_type != CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE {
            return Err(OpaqueAuthenticationFailureV2);
        }
        self.application_authenticated = true;
        Ok(())
    }

    fn exact_account_auth(
        &mut self,
        expected_account_id: i64,
    ) -> Result<(), OpaqueAuthenticationFailureV2> {
        if !self.application_authenticated
            || self.account_authenticated
            || expected_account_id != self.credentials.account_id()
        {
            return Err(OpaqueAuthenticationFailureV2);
        }
        let message = build_account_auth_request(
            expected_account_id,
            self.credentials.access_token(),
            format!("{}-account-auth", self.client_message_namespace),
        );
        let response = self.exchange_auth_message(&message)?;
        let envelope =
            parse_open_api_envelope(&response).map_err(|_| OpaqueAuthenticationFailureV2)?;
        let response_account_id = envelope
            .payload
            .get("ctidTraderAccountId")
            .and_then(serde_json::Value::as_i64)
            .ok_or(OpaqueAuthenticationFailureV2)?;
        if envelope.payload_type != CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE
            || response_account_id != expected_account_id
        {
            return Err(OpaqueAuthenticationFailureV2);
        }
        self.account_authenticated = true;
        Ok(())
    }

    fn into_authenticated_session(
        mut self,
    ) -> Result<Self::Session, OpaqueAuthenticationFailureV2> {
        if !self.application_authenticated || !self.account_authenticated {
            return Err(OpaqueAuthenticationFailureV2);
        }
        self.session.take().ok_or(OpaqueAuthenticationFailureV2)
    }
}

pub fn capture_production_broker_financial_truth_v2(
    request: ProductionBrokerTruthCaptureRequestV2,
    cancellation: &ProductionBrokerTruthCancellationV2,
) -> Result<ProductionBrokerTruthCaptureOutcomeV2, ProductionBrokerTruthCaptureErrorV2> {
    validate_exact_production_request_binding_v2(
        request.environment,
        request.account_id,
        &request.capture_request,
    )?;
    cancellation.begin_run()?;

    let ProductionBrokerTruthCaptureRequestV2 {
        environment,
        account_id,
        authority_receipt,
        capture_request,
        reviewed_synchronizations,
        capture_work_parent,
        store_root,
    } = request;
    let credentials = load_exact_production_broker_truth_credentials_v2(environment, account_id)
        .map_err(|_| {
            production_error(
                ProductionBrokerTruthCaptureStageV2::Credentials,
                ProductionBrokerTruthCaptureErrorCodeV2::CredentialsUnavailable,
                "exact cTrader credentials are unavailable",
            )
        })?;
    ensure_not_cancelled_at(cancellation, ProductionBrokerTruthCaptureStageV2::Connect)?;

    let client_message_namespace = format!("bft2-{}", authority_receipt.manifest_sha256());
    let wire = ProductionCTraderBrokerTruthAuthenticationWireV2::new(
        credentials,
        client_message_namespace.clone(),
    );
    let mut session = establish_exact_authenticated_session_after_run_claimed_v2(
        wire,
        environment.endpoint_host(),
        account_id,
        cancellation,
    )?;
    ensure_not_cancelled_at(cancellation, ProductionBrokerTruthCaptureStageV2::Adapter)?;
    let mut adapter = CTraderBrokerTruthAdapterV2::new(
        &mut session,
        &capture_request,
        client_message_namespace,
        PRODUCTION_BROKER_TRUTH_DEAL_MAX_ROWS_V2,
        PRODUCTION_BROKER_TRUTH_RETURN_PROTECTION_ORDERS_V2,
        reviewed_synchronizations,
    )
    .map_err(|_| {
        production_error(
            ProductionBrokerTruthCaptureStageV2::Adapter,
            ProductionBrokerTruthCaptureErrorCodeV2::CaptureFailed,
            "exact cTrader capture adapter rejected the acquisition request",
        )
    })?;
    let store = BrokerFinancialTruthBundleStoreV1::new(store_root);
    let publication_attempted = Cell::new(false);
    let receipt = capture_and_publish_broker_financial_truth_v2(
        &mut adapter,
        &capture_request,
        capture_work_parent,
        &store,
        || cancellation.is_cancelled(),
        || {
            publication_attempted.set(true);
            cancellation.begin_publication().map_err(|error| {
                let code = match error.code() {
                    ProductionBrokerTruthCaptureErrorCodeV2::Cancelled => {
                        EvidenceCaptureErrorCodeV2::Cancelled
                    }
                    _ => EvidenceCaptureErrorCodeV2::PublicationFailed,
                };
                EvidenceCaptureErrorV2::new(code, "broker-truth publication boundary unavailable")
            })
        },
    )
    .map_err(|error| map_evidence_capture_error(error.code(), publication_attempted.get()))?;
    Ok(ProductionBrokerTruthCaptureOutcomeV2 { receipt })
}

fn map_evidence_capture_error(
    code: EvidenceCaptureErrorCodeV2,
    publication_attempted: bool,
) -> ProductionBrokerTruthCaptureErrorV2 {
    match code {
        EvidenceCaptureErrorCodeV2::Cancelled if publication_attempted => production_error(
            ProductionBrokerTruthCaptureStageV2::Publication,
            ProductionBrokerTruthCaptureErrorCodeV2::Cancelled,
            "broker-truth acquisition was cancelled before publication",
        ),
        EvidenceCaptureErrorCodeV2::Cancelled => production_error(
            ProductionBrokerTruthCaptureStageV2::Adapter,
            ProductionBrokerTruthCaptureErrorCodeV2::Cancelled,
            "broker-truth acquisition was cancelled during capture",
        ),
        EvidenceCaptureErrorCodeV2::PublicationFailed => production_error(
            ProductionBrokerTruthCaptureStageV2::Publication,
            ProductionBrokerTruthCaptureErrorCodeV2::PublicationFailed,
            "immutable broker-truth publication failed",
        ),
        _ => production_error(
            ProductionBrokerTruthCaptureStageV2::Adapter,
            ProductionBrokerTruthCaptureErrorCodeV2::CaptureFailed,
            "exact cTrader evidence capture failed",
        ),
    }
}
