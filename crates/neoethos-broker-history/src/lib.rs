//! Lean, model-free cTrader historical capture boundary.
//!
//! The history-specific production transport is temporarily compiled from the
//! same authoritative source files used by `neoethos-app`. Once the shared
//! service is proven, the app delegates here and its superseded inline capture
//! loop is removed.

#![forbid(unsafe_code)]

#[path = "../../neoethos-app/src/app_services/bootstrap_writer.rs"]
pub mod bootstrap_writer;
#[path = "../../neoethos-app/src/app_services/ctrader_auth.rs"]
pub mod ctrader_auth;
#[path = "../../neoethos-app/src/app_services/ctrader_data.rs"]
pub mod ctrader_data;
#[path = "../../neoethos-app/src/app_services/ctrader_historical_admission.rs"]
pub(crate) mod ctrader_historical_admission;
#[path = "../../neoethos-app/src/app_services/ctrader_historical_page.rs"]
pub(crate) mod ctrader_historical_page;
#[path = "../../neoethos-app/src/app_services/ctrader_live_auth.rs"]
pub mod ctrader_live_auth;
#[path = "../../neoethos-app/src/app_services/ctrader_messages.rs"]
pub mod ctrader_messages;
#[path = "../../neoethos-app/src/app_services/ctrader_money.rs"]
pub mod ctrader_money;
#[path = "../../neoethos-app/src/app_services/ctrader_tick_delta.rs"]
pub(crate) mod ctrader_tick_delta;
#[path = "../../neoethos-app/src/app_services/ctrader_tls.rs"]
pub mod ctrader_tls;
#[path = "../../neoethos-app/src/app_services/secure_store.rs"]
pub mod secure_store;

/// Compatibility namespace required by the shared cTrader source files while
/// their physical move out of `neoethos-app` remains pending GREEN proof.
pub mod app_services {
    pub use crate::ctrader_auth;
    pub use crate::ctrader_data;
    pub(crate) use crate::ctrader_historical_admission;
    pub(crate) use crate::ctrader_historical_page;
    pub use crate::ctrader_live_auth;
    pub use crate::ctrader_messages;
    pub use crate::ctrader_money;
    pub(crate) use crate::ctrader_tick_delta;
    pub use crate::ctrader_tls;
}

pub mod broker_truth_capture;
pub mod broker_truth_ctrader;
mod broker_truth_vortex;
pub mod bulk_cli;
pub mod cli;
mod historical_series_acquisition_v1;
mod historical_series_runner_v1;
mod production_broker_truth_v2;
mod reviewed_sync_ingress_v2;
mod service;
pub mod symbol_contract_cli;

pub use production_broker_truth_v2::{
    PRODUCTION_BROKER_TRUTH_DEAL_MAX_ROWS_V2, PRODUCTION_BROKER_TRUTH_RETURN_PROTECTION_ORDERS_V2,
    ProductionBrokerTruthCancelResultV2, ProductionBrokerTruthCancellationV2,
    ProductionBrokerTruthCaptureErrorCodeV2, ProductionBrokerTruthCaptureErrorV2,
    ProductionBrokerTruthCaptureOutcomeV2, ProductionBrokerTruthCaptureRequestV2,
    ProductionBrokerTruthCaptureStageV2, capture_production_broker_financial_truth_v2,
};

pub use historical_series_acquisition_v1::{
    CANONICAL_TRENDBAR_PAGING_POLICY_V1, CANONICAL_TRENDBAR_SERIES_FROM_MS_V1,
    CanonicalTrendbarAcquisitionCellV1, CanonicalTrendbarAcquisitionCheckpointV1,
    CanonicalTrendbarAcquisitionPlanV1, CanonicalTrendbarAcquisitionStoreV1,
    CanonicalTrendbarCheckpointReceiptV1, CanonicalTrendbarMatrixReceiptV1,
    CanonicalTrendbarMatrixV1, CanonicalTrendbarPlanReceiptV1, CanonicalTrendbarSymbolV1,
};

pub use historical_series_runner_v1::{
    CanonicalTrendbarAcquisitionRunFailureV1, CanonicalTrendbarAcquisitionRunOutcomeV1,
    CanonicalTrendbarAcquisitionRunStageV1, resume_canonical_trendbar_acquisition_v1_with,
    run_production_canonical_trendbar_acquisition_v1,
};

pub use reviewed_sync_ingress_v2::{
    LoadedReviewedCTraderQuoteSynchronizationV2,
    ReviewedCTraderQuoteSynchronizationIngressErrorCodeV2,
    ReviewedCTraderQuoteSynchronizationIngressErrorV2, ReviewedCTraderQuoteSynchronizationSourceV2,
    load_reviewed_ctrader_quote_synchronizations_v2,
};

pub use service::{
    BrokerEnvironment, BrokerHistoryConflict, HistoricalCaptureCancellationHandle,
    HistoricalCaptureRequest, HistoricalCaptureStatus, HistoricalCaptureTarget,
    HistoricalCredentials, HistoricalDownloadOutcome, HistoricalFetchCancelResult,
    HistoricalFetchStartFailure, ProcessHistoricalCapture, begin_process_historical_capture,
    cancel_process_historical_capture, capture_historical_generation,
    is_historical_capture_cancelled, load_exact_production_historical_credentials,
    load_production_historical_credentials, process_historical_capture_status,
};

pub use symbol_contract_cli::{
    ExactBrokerSymbolContractBindingV1, ExactBrokerSymbolContractReceiptV1,
};

#[cfg(test)]
pub(crate) use ctrader_historical_admission::{
    HistoricalFetchCancelOutcome, HistoricalFetchRegistry,
};

#[cfg(test)]
mod service_tests;

#[cfg(test)]
mod production_broker_truth_v2_tests;
