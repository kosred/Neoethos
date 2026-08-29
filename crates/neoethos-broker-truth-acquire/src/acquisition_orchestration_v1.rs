use std::error::Error;
use std::fmt;
#[cfg(test)]
use std::path::Path;
use std::path::PathBuf;

use neoethos_broker_history::broker_truth_capture::ExactQuoteInstrumentV2;
use neoethos_broker_history::{
    BrokerEnvironment, ProductionBrokerTruthCancellationV2, ProductionBrokerTruthCaptureRequestV2,
    ReviewedCTraderQuoteSynchronizationSourceV2, capture_production_broker_financial_truth_v2,
    load_reviewed_ctrader_quote_synchronizations_v2,
};
#[cfg(test)]
use neoethos_broker_truth::{BrokerFinancialTruthBindingV1, EvidenceWindowV1};
use neoethos_broker_truth::{
    BrokerFinancialTruthBundleReceiptV2, BrokerTruthAcquisitionArtifactRoleV1,
    BrokerTruthAcquisitionAuthorityReceiptV1, BrokerTruthAcquisitionLinkReceiptV1,
    BrokerTruthAcquisitionPromotionEligibilityV1, BrokerTruthAcquisitionSemanticStatusV1,
    BrokerTruthAcquisitionStoreV1,
};
use neoethos_data::CTraderEnvironment;

use crate::PreparedBrokerTruthAcquisitionV1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrokerTruthAcquisitionOrchestrationErrorCodeV1 {
    ReviewedSynchronizationInvalid,
    AuthorityPublicationFailed,
    CaptureRequestInvalid,
    CaptureFailed,
    LinkPublicationFailed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerTruthAcquisitionOrchestrationErrorV1 {
    code: BrokerTruthAcquisitionOrchestrationErrorCodeV1,
    detail: &'static str,
}

impl BrokerTruthAcquisitionOrchestrationErrorV1 {
    pub const fn code(&self) -> BrokerTruthAcquisitionOrchestrationErrorCodeV1 {
        self.code
    }

    pub const fn detail(&self) -> &'static str {
        self.detail
    }
}

impl fmt::Display for BrokerTruthAcquisitionOrchestrationErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.detail)
    }
}

impl Error for BrokerTruthAcquisitionOrchestrationErrorV1 {}

fn orchestration_error(
    code: BrokerTruthAcquisitionOrchestrationErrorCodeV1,
    detail: &'static str,
) -> BrokerTruthAcquisitionOrchestrationErrorV1 {
    BrokerTruthAcquisitionOrchestrationErrorV1 { code, detail }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrokerTruthAcquisitionOutcomeV1 {
    authority_receipt: BrokerTruthAcquisitionAuthorityReceiptV1,
    broker_truth_receipt: BrokerFinancialTruthBundleReceiptV2,
    link_receipt: BrokerTruthAcquisitionLinkReceiptV1,
    semantic_status: BrokerTruthAcquisitionSemanticStatusV1,
    promotion_eligibility: BrokerTruthAcquisitionPromotionEligibilityV1,
}

impl BrokerTruthAcquisitionOutcomeV1 {
    pub const fn authority_receipt(&self) -> &BrokerTruthAcquisitionAuthorityReceiptV1 {
        &self.authority_receipt
    }

    pub const fn broker_truth_receipt(&self) -> &BrokerFinancialTruthBundleReceiptV2 {
        &self.broker_truth_receipt
    }

    pub const fn link_receipt(&self) -> &BrokerTruthAcquisitionLinkReceiptV1 {
        &self.link_receipt
    }

    pub const fn semantic_status(&self) -> BrokerTruthAcquisitionSemanticStatusV1 {
        self.semantic_status
    }

    pub const fn promotion_eligibility(&self) -> BrokerTruthAcquisitionPromotionEligibilityV1 {
        self.promotion_eligibility
    }
}

pub(super) struct BrokerTruthCaptureRunnerFailureV1;

impl BrokerTruthCaptureRunnerFailureV1 {
    pub(super) const fn opaque() -> Self {
        Self
    }
}

pub(super) struct BrokerTruthCaptureInvocationV1 {
    request: ProductionBrokerTruthCaptureRequestV2,
    #[cfg(test)]
    environment: BrokerEnvironment,
    #[cfg(test)]
    account_id: i64,
    #[cfg(test)]
    window: EvidenceWindowV1,
    #[cfg(test)]
    binding: BrokerFinancialTruthBindingV1,
    #[cfg(test)]
    authority_receipt: BrokerTruthAcquisitionAuthorityReceiptV1,
    #[cfg(test)]
    reviewed_synchronization_count: usize,
    #[cfg(test)]
    store_root: PathBuf,
}

impl BrokerTruthCaptureInvocationV1 {
    fn into_request(self) -> ProductionBrokerTruthCaptureRequestV2 {
        self.request
    }

    #[cfg(test)]
    pub(super) const fn environment(&self) -> BrokerEnvironment {
        self.environment
    }

    #[cfg(test)]
    pub(super) const fn account_id(&self) -> i64 {
        self.account_id
    }

    #[cfg(test)]
    pub(super) const fn window(&self) -> EvidenceWindowV1 {
        self.window
    }

    #[cfg(test)]
    pub(super) const fn binding(&self) -> &BrokerFinancialTruthBindingV1 {
        &self.binding
    }

    #[cfg(test)]
    pub(super) const fn authority_receipt(&self) -> &BrokerTruthAcquisitionAuthorityReceiptV1 {
        &self.authority_receipt
    }

    #[cfg(test)]
    pub(super) const fn reviewed_synchronization_count(&self) -> usize {
        self.reviewed_synchronization_count
    }

    #[cfg(test)]
    pub(super) fn store_root(&self) -> &Path {
        &self.store_root
    }
}

#[cfg(test)]
pub(super) trait BrokerTruthCaptureRunnerV1 {
    fn capture(
        &mut self,
        invocation: BrokerTruthCaptureInvocationV1,
        cancellation: &ProductionBrokerTruthCancellationV2,
    ) -> Result<BrokerFinancialTruthBundleReceiptV2, BrokerTruthCaptureRunnerFailureV1>;
}

fn broker_environment(environment: CTraderEnvironment) -> BrokerEnvironment {
    match environment {
        CTraderEnvironment::Demo => BrokerEnvironment::Demo,
        CTraderEnvironment::Live => BrokerEnvironment::Live,
    }
}

fn reviewed_synchronization_error() -> BrokerTruthAcquisitionOrchestrationErrorV1 {
    orchestration_error(
        BrokerTruthAcquisitionOrchestrationErrorCodeV1::ReviewedSynchronizationInvalid,
        "reviewed quote synchronization evidence is invalid",
    )
}

fn validate_artifact_source_order(
    prepared: &PreparedBrokerTruthAcquisitionV1,
) -> Result<(), BrokerTruthAcquisitionOrchestrationErrorV1> {
    let artifacts = prepared.authority_manifest().artifacts();
    let sources = prepared.artifact_sources();
    if artifacts.len() != sources.len() {
        return Err(reviewed_synchronization_error());
    }
    for (artifact, source) in artifacts.iter().zip(sources) {
        if artifact.relative_path() != source.relative_path() {
            return Err(reviewed_synchronization_error());
        }
    }
    Ok(())
}

fn source_path_for_role(
    prepared: &PreparedBrokerTruthAcquisitionV1,
    expected_role: BrokerTruthAcquisitionArtifactRoleV1,
) -> Result<PathBuf, BrokerTruthAcquisitionOrchestrationErrorV1> {
    let mut found = None;
    for (artifact, source) in prepared
        .authority_manifest()
        .artifacts()
        .iter()
        .zip(prepared.artifact_sources())
    {
        if artifact.role() == expected_role {
            if found.is_some() {
                return Err(reviewed_synchronization_error());
            }
            found = Some(source.source_path().to_path_buf());
        }
    }
    match found {
        Some(path) => Ok(path),
        None => Err(reviewed_synchronization_error()),
    }
}

fn planned_instruments(prepared: &PreparedBrokerTruthAcquisitionV1) -> Vec<ExactQuoteInstrumentV2> {
    std::iter::once(prepared.capture_request().primary_instrument().clone())
        .chain(
            prepared
                .capture_request()
                .conversion_routes()
                .iter()
                .flat_map(|route| route.legs().iter().map(|leg| leg.instrument().clone())),
        )
        .collect()
}

fn load_exact_reviewed_synchronizations(
    prepared: &PreparedBrokerTruthAcquisitionV1,
) -> Result<
    Vec<neoethos_broker_history::broker_truth_ctrader::ReviewedCTraderQuoteSynchronizationV2>,
    BrokerTruthAcquisitionOrchestrationErrorV1,
> {
    validate_artifact_source_order(prepared)?;
    let bindings = prepared.authority_manifest().reviewed_synchronizations();
    let instruments = planned_instruments(prepared);
    if bindings.len() != prepared.reviewed_synchronization_count()
        || instruments.len() != bindings.len()
    {
        return Err(reviewed_synchronization_error());
    }

    let mut sources = Vec::with_capacity(bindings.len());
    for (index, (binding, instrument)) in bindings.iter().zip(&instruments).enumerate() {
        let expected_ordinal =
            u32::try_from(index).map_err(|_| reviewed_synchronization_error())?;
        if binding.ordinal() != expected_ordinal
            || binding.symbol_id() != instrument.symbol_id()
            || binding.window() != prepared.evidence_window()
        {
            return Err(reviewed_synchronization_error());
        }
        let observations_path = source_path_for_role(
            prepared,
            BrokerTruthAcquisitionArtifactRoleV1::QuoteSessionObservations {
                ordinal: expected_ordinal,
            },
        )?;
        let replay_rules_path = source_path_for_role(
            prepared,
            BrokerTruthAcquisitionArtifactRoleV1::ReviewedQuoteReplayRules {
                ordinal: expected_ordinal,
            },
        )?;
        sources.push(
            ReviewedCTraderQuoteSynchronizationSourceV2::new(
                binding.clone(),
                instrument.clone(),
                observations_path,
                replay_rules_path,
            )
            .map_err(|_| reviewed_synchronization_error())?,
        );
    }

    let loaded = load_reviewed_ctrader_quote_synchronizations_v2(sources)
        .map_err(|_| reviewed_synchronization_error())?;
    if loaded.len() != bindings.len() {
        return Err(reviewed_synchronization_error());
    }
    let mut reviewed = Vec::with_capacity(loaded.len());
    for ((loaded, binding), instrument) in loaded.into_iter().zip(bindings).zip(&instruments) {
        if loaded.ordinal() != binding.ordinal()
            || loaded.account_id() != binding.account_id()
            || loaded.symbol_id() != instrument.symbol_id()
            || loaded.window() != binding.window()
            || loaded.review_identity_sha256() != binding.review_identity().identity_sha256()
            || loaded.raw_observation_count() != 2
            || loaded.decoded_replay_rule_count() != 1
        {
            return Err(reviewed_synchronization_error());
        }
        reviewed.push(loaded.into_synchronization());
    }
    Ok(reviewed)
}

fn execute_prepared_acquisition_with_capture_v1<Capture>(
    prepared: PreparedBrokerTruthAcquisitionV1,
    cancellation: &ProductionBrokerTruthCancellationV2,
    capture: Capture,
) -> Result<BrokerTruthAcquisitionOutcomeV1, BrokerTruthAcquisitionOrchestrationErrorV1>
where
    Capture:
        FnOnce(
            BrokerTruthCaptureInvocationV1,
            &ProductionBrokerTruthCancellationV2,
        )
            -> Result<BrokerFinancialTruthBundleReceiptV2, BrokerTruthCaptureRunnerFailureV1>,
{
    let opened_generation_count = prepared.opened_generation_count();
    if opened_generation_count == 0 {
        return Err(orchestration_error(
            BrokerTruthAcquisitionOrchestrationErrorCodeV1::CaptureRequestInvalid,
            "exact canonical generation lease is missing",
        ));
    }
    let reviewed_synchronizations = load_exact_reviewed_synchronizations(&prepared)?;
    let store_root = prepared.store_root().to_path_buf();
    let store = BrokerTruthAcquisitionStoreV1::new(&store_root);
    let authority_receipt = store
        .publish_authority(prepared.authority_manifest(), prepared.artifact_sources())
        .map_err(|_| {
            orchestration_error(
                BrokerTruthAcquisitionOrchestrationErrorCodeV1::AuthorityPublicationFailed,
                "immutable acquisition authority publication failed",
            )
        })?;

    let environment = broker_environment(prepared.environment());
    let account_id = prepared.account_id();
    let binding = prepared.capture_request().binding().clone();
    let request = ProductionBrokerTruthCaptureRequestV2::new(
        environment,
        account_id,
        authority_receipt.clone(),
        prepared.capture_request().clone(),
        reviewed_synchronizations,
        prepared.work_parent().to_path_buf(),
        store_root.clone(),
    )
    .map_err(|_| {
        orchestration_error(
            BrokerTruthAcquisitionOrchestrationErrorCodeV1::CaptureRequestInvalid,
            "exact production capture request is invalid",
        )
    })?;
    let invocation = BrokerTruthCaptureInvocationV1 {
        request,
        #[cfg(test)]
        environment,
        #[cfg(test)]
        account_id,
        #[cfg(test)]
        window: prepared.evidence_window(),
        #[cfg(test)]
        binding: binding.clone(),
        #[cfg(test)]
        authority_receipt: authority_receipt.clone(),
        #[cfg(test)]
        reviewed_synchronization_count: prepared.reviewed_synchronization_count(),
        #[cfg(test)]
        store_root: store_root.clone(),
    };
    let broker_truth_receipt = capture(invocation, cancellation).map_err(|_| {
        orchestration_error(
            BrokerTruthAcquisitionOrchestrationErrorCodeV1::CaptureFailed,
            "exact broker-truth capture failed",
        )
    })?;
    let link_receipt = store
        .publish_link(&authority_receipt, &broker_truth_receipt, &binding)
        .map_err(|_| {
            orchestration_error(
                BrokerTruthAcquisitionOrchestrationErrorCodeV1::LinkPublicationFailed,
                "immutable acquisition link publication failed",
            )
        })?;
    let reopened = store.open_link(&link_receipt).map_err(|_| {
        orchestration_error(
            BrokerTruthAcquisitionOrchestrationErrorCodeV1::LinkPublicationFailed,
            "immutable acquisition link verification failed",
        )
    })?;
    let manifest = reopened.manifest();
    if reopened.receipt() != &link_receipt
        || manifest.authority_receipt() != &authority_receipt
        || manifest.broker_truth_receipt() != &broker_truth_receipt
        || manifest.binding() != &binding
        || prepared.opened_generation_count() != opened_generation_count
        || manifest.semantic_status()
            != BrokerTruthAcquisitionSemanticStatusV1::UnvalidatedEvidenceOnly
        || manifest.promotion_eligibility()
            != BrokerTruthAcquisitionPromotionEligibilityV1::NotPromotionEligible
    {
        return Err(orchestration_error(
            BrokerTruthAcquisitionOrchestrationErrorCodeV1::LinkPublicationFailed,
            "immutable acquisition link verification failed",
        ));
    }

    Ok(BrokerTruthAcquisitionOutcomeV1 {
        authority_receipt,
        broker_truth_receipt,
        link_receipt,
        semantic_status: manifest.semantic_status(),
        promotion_eligibility: manifest.promotion_eligibility(),
    })
}

pub fn execute_prepared_acquisition_v1(
    prepared: PreparedBrokerTruthAcquisitionV1,
    cancellation: &ProductionBrokerTruthCancellationV2,
) -> Result<BrokerTruthAcquisitionOutcomeV1, BrokerTruthAcquisitionOrchestrationErrorV1> {
    execute_prepared_acquisition_with_capture_v1(
        prepared,
        cancellation,
        |invocation, cancellation| {
            let outcome = capture_production_broker_financial_truth_v2(
                invocation.into_request(),
                cancellation,
            )
            .map_err(|_| BrokerTruthCaptureRunnerFailureV1::opaque())?;
            Ok(outcome.receipt().clone())
        },
    )
}

#[cfg(test)]
pub(super) fn execute_prepared_acquisition_with_runner_v1<Runner>(
    prepared: PreparedBrokerTruthAcquisitionV1,
    cancellation: &ProductionBrokerTruthCancellationV2,
    runner: &mut Runner,
) -> Result<BrokerTruthAcquisitionOutcomeV1, BrokerTruthAcquisitionOrchestrationErrorV1>
where
    Runner: BrokerTruthCaptureRunnerV1,
{
    execute_prepared_acquisition_with_capture_v1(
        prepared,
        cancellation,
        |invocation, cancellation| runner.capture(invocation, cancellation),
    )
}
