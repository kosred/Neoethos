//! Data-owned resident higher-timeframe causal-alignment producer.
//!
//! Each selected direct timeframe moves an opaque runtime parent carrier into
//! this owner. The carrier, its generation leases, resident buffers, clock
//! schedule, ready-event context/stream identities, and source-route receipts
//! therefore outlive every aligned batch. No parent is reconstructed from a
//! hash and no aligned feature or validity cell is copied to the host.

use std::collections::BTreeSet;

use anyhow::{Context as _, Result, bail, ensure};
use neoethos_dataset_contracts::CanonicalTimeframe;
use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentFeatureStageV3, ResidentProducerCapabilityV3,
};
use neoethos_gpu_cuda::resident_classic_ta_v3::{
    ResidentClassicTaPreDeviceMemoryReceiptV4, ResidentClassicTaRecipeV3,
};
use neoethos_gpu_cuda::resident_feature_store_v3::{
    GpuOnlyRunDeviceAdmissionV3, ResidentFeatureColumnBindingV3, ResidentFeatureStoreAssemblerV3,
    ResidentFeatureStoreCudaErrorV3, resident_canonical_content_sha256_capability_v3,
    resident_feature_major_to_bar_major_capability_v3,
};
use neoethos_gpu_cuda::resident_footprint_v2::ResidentFootprintPreDeviceMemoryReceiptV4;
use neoethos_gpu_cuda::resident_higher_timeframe_alignment_v3::{
    PendingResidentHigherTimeframeDirectParentCaptureV3, ResidentHigherTimeframeAvailabilityRuleV3,
    ResidentHigherTimeframeDirectParentLaunchPlanV3, ResidentHigherTimeframeDirectParentV3,
    ResidentHigherTimeframeLaunchAuthorityV3, ResidentHigherTimeframeParentAuthorityV3,
    ResidentHigherTimeframeRouteAuthorityV3, ResidentHigherTimeframeRuntimeReceiptV3,
    capture_resident_higher_timeframe_direct_parent_v3, resident_higher_timeframe_capability_v3,
    seal_resident_higher_timeframe_source_closure_v3,
};
use neoethos_gpu_cuda::resident_regime_v3::ResidentRegimePreDeviceMemoryReceiptV4;
use neoethos_gpu_cuda::resident_robust_normalization_v2::resident_robust_normalization_capability_v2;
use neoethos_gpu_cuda::resident_smc_v3::{
    ResidentSmcMaterializationV3, ResidentSmcPreDeviceMemoryReceiptV4,
};
use sha2::{Digest, Sha256};

use super::gpu_resident_feature_recipe_v4::{
    MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V4, ResidentCanonicalParameterV4,
    ResidentCanonicalParameterValueV4, ResidentColumnSchemaAssemblerV4,
    ResidentProducerBatchDraftV4, ResidentProducerDraftV4, ResidentRouteDraftV4,
    ResidentTransformCapabilityDraftV4,
};
use super::gpu_resident_quant_v3::{
    PendingResidentQuantHigherTimeframeParentV3, PreparedResidentQuantRuntimeV3,
    ResidentQuantHigherTimeframeBatchMemoryV3,
};
use super::gpu_resident_regime_v3::PreparedResidentRegimeInputV3;
use super::gpu_resident_session_v2::{
    PendingResidentSessionHigherTimeframeParentV2, PreparedResidentSessionRuntimeV2,
    ResidentSessionHigherTimeframeBatchMemoryV2,
};
use super::pinned_canonical_series_v1::MaterializedPinnedResidentCanonicalSourceV1;
use super::timestamps::validate_canonical_millisecond_timestamps;

pub(crate) const HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION_V3: u32 = 3;
pub(crate) const RESIDENT_HTF_IMPLEMENTATION_ID_V3: &str =
    "neoethos.cuda.resident-higher-timeframe-alignment.semantic-v3";
pub(crate) const RESIDENT_HTF_EXACT_MATH_AUTHORITY_V3: &str = "neoethos.higher-timeframe-alignment.cpu-oracle.semantic-v3;direct-source-only;selected-parent-order;cpu-producer-order;fixed-open-plus-period-v1;calendar-next-direct-bar-open-v1;forward-fill=true;fixed-max-age=2x-period;logical-validity-preserved;zero-feature-d2h";
const RESIDENT_HTF_ROUTE_DOMAIN_V3: &str =
    "neoethos.data.resident-higher-timeframe-route.semantic-v3";
const RESIDENT_HTF_INDICATOR_ID_V3: &str = "neoethos_higher_timeframe_alignment_semantic_v3";

const CANONICAL_CPU_PRODUCER_ORDER_V3: [ResidentFeatureProducerV3; 6] = [
    ResidentFeatureProducerV3::Smc,
    ResidentFeatureProducerV3::ClassicTa,
    ResidentFeatureProducerV3::Quant,
    ResidentFeatureProducerV3::Session,
    ResidentFeatureProducerV3::Regime,
    ResidentFeatureProducerV3::Footprint,
];

pub(crate) trait ResidentHigherTimeframeNativeCarrierV3: Sized {
    fn into_native_parent(
        self,
        authority: ResidentHigherTimeframeParentAuthorityV3,
    ) -> Result<ResidentHigherTimeframeDirectParentV3>;
}

#[derive(Debug)]
pub(crate) struct ResidentHigherTimeframeParentRouteV3 {
    feature_name: String,
    producer: ResidentFeatureProducerV3,
    source_route_receipt_sha256: [u8; 32],
}

impl ResidentHigherTimeframeParentRouteV3 {
    /// Intended only for the direct-timeframe resident parent sealer. The
    /// complete parent-order validator below remains the authority boundary.
    pub(crate) fn from_sealed_parent_route(
        feature_name: impl Into<String>,
        producer: ResidentFeatureProducerV3,
        source_route_receipt_sha256: [u8; 32],
    ) -> Result<Self> {
        let feature_name = feature_name.into();
        ensure!(
            !feature_name.trim().is_empty(),
            "HTF parent route name is empty"
        );
        ensure!(
            CANONICAL_CPU_PRODUCER_ORDER_V3.contains(&producer),
            "HTF parent route is not a canonical CPU column producer"
        );
        ensure!(
            source_route_receipt_sha256 != [0; 32],
            "HTF parent source route receipt is zero"
        );
        Ok(Self {
            feature_name,
            producer,
            source_route_receipt_sha256,
        })
    }
}

/// Exact resident extent expected after all six direct-parent producer
/// families have launched and moved into the opaque capture. This formula is
/// constructed only from their owner preflight receipts; it contains no
/// device/context/stream observation and cannot be caller-minted.
#[derive(Debug)]
pub(crate) struct ResidentHigherTimeframeParentBatchMemoryFormulaV3 {
    row_count: u64,
    source_route_count: u64,
    expected_retained_parent_device_bytes: u64,
    quant_feature_column_count: u64,
    quant_retained_feature_device_bytes: u64,
    session_feature_column_count: u64,
    session_retained_feature_device_bytes: u64,
    regime_feature_column_count: u64,
    regime_retained_feature_device_bytes: u64,
    regime_scratch_device_bytes: u64,
    footprint_feature_column_count: u64,
    footprint_retained_feature_device_bytes: u64,
    footprint_scratch_device_bytes: u64,
}

impl ResidentHigherTimeframeParentBatchMemoryFormulaV3 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_owner_preflight_receipts_v3(
        smc: &ResidentSmcPreDeviceMemoryReceiptV4,
        classic: &ResidentClassicTaPreDeviceMemoryReceiptV4,
        quant: &ResidentQuantHigherTimeframeBatchMemoryV3,
        session: &ResidentSessionHigherTimeframeBatchMemoryV2,
        regime: &ResidentRegimePreDeviceMemoryReceiptV4,
        footprint: &ResidentFootprintPreDeviceMemoryReceiptV4,
    ) -> Result<Self> {
        let row_count = u64::try_from(smc.row_count()).context("HTF SMC row extent overflow")?;
        ensure!(row_count > 0, "HTF parent memory formula has zero rows");
        for (producer, rows) in [
            (
                "Classic",
                u64::try_from(classic.rows()).context("HTF Classic rows overflow")?,
            ),
            ("Quant", quant.row_count()),
            ("Session", session.row_count()),
            (
                "Regime",
                u64::try_from(regime.row_count()).context("HTF Regime rows overflow")?,
            ),
            (
                "Footprint",
                u64::try_from(footprint.row_count()).context("HTF Footprint rows overflow")?,
            ),
        ] {
            ensure!(
                rows == row_count,
                "HTF {producer} memory rows differ from SMC"
            );
        }

        let mut classic_feature_column_count = 0_u64;
        let mut classic_retained_scratch_bytes = 0_u64;
        for launch in classic.launch_plans() {
            let validity_bytes = u64::try_from(launch.validity_bytes())
                .context("HTF Classic validity bytes overflow")?;
            ensure!(
                validity_bytes % row_count == 0,
                "HTF Classic validity extent is not row-major exact"
            );
            let columns = validity_bytes / row_count;
            let expected_value_bytes = row_count
                .checked_mul(columns)
                .and_then(|cells| cells.checked_mul(8))
                .context("HTF Classic value bytes overflow")?;
            ensure!(
                columns > 0
                    && u64::try_from(launch.selected_value_bytes())
                        .context("HTF Classic selected bytes overflow")?
                        == expected_value_bytes
                    && u64::try_from(launch.all_output_retained_bytes())
                        .context("HTF Classic retained output bytes overflow")?
                        == expected_value_bytes
                    && launch.additional_retained_bytes() == classic.derived_input_bytes(),
                "HTF Classic owner memory formula drifted"
            );
            classic_feature_column_count = classic_feature_column_count
                .checked_add(columns)
                .context("HTF Classic column census overflow")?;
            classic_retained_scratch_bytes = classic_retained_scratch_bytes
                .checked_add(
                    u64::try_from(launch.retained_scratch_bytes())
                        .context("HTF Classic retained scratch overflow")?,
                )
                .context("HTF Classic retained scratch sum overflow")?;
        }
        ensure!(
            classic_feature_column_count > 0,
            "HTF Classic memory formula has no launch routes"
        );

        let smc_feature_column_count =
            u64::try_from(smc.feature_column_count()).context("HTF SMC columns overflow")?;
        let regime_feature_column_count =
            u64::try_from(regime.feature_column_count()).context("HTF Regime columns overflow")?;
        let footprint_feature_column_count = u64::try_from(footprint.feature_column_count())
            .context("HTF Footprint columns overflow")?;
        let quant_feature_column_count = quant.feature_column_count();
        let session_feature_column_count = session.feature_column_count();
        let source_route_count = smc_feature_column_count
            .checked_add(classic_feature_column_count)
            .and_then(|count| count.checked_add(quant_feature_column_count))
            .and_then(|count| count.checked_add(session_feature_column_count))
            .and_then(|count| count.checked_add(regime_feature_column_count))
            .and_then(|count| count.checked_add(footprint_feature_column_count))
            .context("HTF direct-parent route census overflow")?;
        let expected_quant_bytes = row_count
            .checked_mul(quant_feature_column_count)
            .and_then(|cells| cells.checked_mul(9))
            .context("HTF Quant retained bytes overflow")?;
        let expected_session_bytes = row_count
            .checked_mul(session_feature_column_count)
            .and_then(|cells| cells.checked_mul(9))
            .context("HTF Session retained bytes overflow")?;
        ensure!(
            quant.retained_feature_device_bytes() == expected_quant_bytes
                && quant.additional_retained_device_bytes() == 0
                && quant.scratch_device_bytes() == 0
                && session.retained_feature_device_bytes() == expected_session_bytes
                && session.additional_retained_device_bytes() == 0
                && session.scratch_device_bytes() == 0
                && regime.additional_retained_bytes() == 0
                && footprint.additional_retained_bytes() == 0,
            "HTF Quant/Session/Regime/Footprint owner memory formula drifted"
        );

        let regime_retained_feature_device_bytes = row_count
            .checked_mul(regime_feature_column_count)
            .and_then(|cells| cells.checked_mul(9))
            .context("HTF Regime retained bytes overflow")?;
        let footprint_retained_feature_device_bytes = row_count
            .checked_mul(footprint_feature_column_count)
            .and_then(|cells| cells.checked_mul(9))
            .context("HTF Footprint retained bytes overflow")?;
        let regime_scratch_device_bytes =
            u64::try_from(regime.scratch_bytes()).context("HTF Regime scratch bytes overflow")?;
        let footprint_scratch_device_bytes = u64::try_from(footprint.scratch_bytes())
            .context("HTF Footprint scratch bytes overflow")?;
        let retained_feature_bytes = row_count
            .checked_mul(source_route_count)
            .and_then(|cells| cells.checked_mul(9))
            .context("HTF direct-parent feature bytes overflow")?;
        let expected_retained_parent_device_bytes = u64::try_from(smc.additional_retained_bytes())
            .context("HTF retained SMC parent bytes overflow")?
            .checked_add(retained_feature_bytes)
            .and_then(|bytes| bytes.checked_add(classic_retained_scratch_bytes))
            .and_then(|bytes| bytes.checked_add(regime_scratch_device_bytes))
            .and_then(|bytes| bytes.checked_add(footprint_scratch_device_bytes))
            .context("HTF retained direct-parent bytes overflow")?;

        Ok(Self {
            row_count,
            source_route_count,
            expected_retained_parent_device_bytes,
            quant_feature_column_count,
            quant_retained_feature_device_bytes: expected_quant_bytes,
            session_feature_column_count,
            session_retained_feature_device_bytes: expected_session_bytes,
            regime_feature_column_count,
            regime_retained_feature_device_bytes,
            regime_scratch_device_bytes,
            footprint_feature_column_count,
            footprint_retained_feature_device_bytes,
            footprint_scratch_device_bytes,
        })
    }

    fn validate_captured_receipts_v3(
        &self,
        capture: &PendingResidentHigherTimeframeDirectParentCaptureV3,
    ) -> Result<()> {
        let quant = capture.quant_runtime_receipt();
        let session = capture.session_runtime_receipt();
        let regime = capture.regime_runtime_receipt();
        let footprint = capture.footprint_runtime_receipt();
        ensure!(
            u64::try_from(quant.row_count()).context("HTF captured Quant rows overflow")?
                == self.row_count
                && u64::try_from(quant.feature_column_count())
                    .context("HTF captured Quant columns overflow")?
                    == self.quant_feature_column_count
                && u64::try_from(quant.retained_feature_device_bytes())
                    .context("HTF captured Quant bytes overflow")?
                    == self.quant_retained_feature_device_bytes
                && quant.additional_retained_device_bytes() == 0
                && quant.scratch_device_bytes() == 0,
            "HTF captured Quant receipt differs from owner preflight"
        );
        ensure!(
            u64::try_from(session.row_count()).context("HTF captured Session rows overflow")?
                == self.row_count
                && u64::try_from(session.feature_column_count())
                    .context("HTF captured Session columns overflow")?
                    == self.session_feature_column_count
                && u64::try_from(session.retained_feature_device_bytes())
                    .context("HTF captured Session bytes overflow")?
                    == self.session_retained_feature_device_bytes
                && session.additional_retained_device_bytes() == 0
                && session.scratch_device_bytes() == 0,
            "HTF captured Session receipt differs from owner preflight"
        );
        ensure!(
            u64::try_from(regime.row_count()).context("HTF captured Regime rows overflow")?
                == self.row_count
                && u64::try_from(regime.feature_column_count())
                    .context("HTF captured Regime columns overflow")?
                    == self.regime_feature_column_count
                && u64::try_from(regime.retained_feature_device_bytes())
                    .context("HTF captured Regime bytes overflow")?
                    == self.regime_retained_feature_device_bytes
                && u64::try_from(regime.scratch_device_bytes())
                    .context("HTF captured Regime scratch overflow")?
                    == self.regime_scratch_device_bytes,
            "HTF captured Regime receipt differs from owner preflight"
        );
        ensure!(
            u64::try_from(footprint.row_count()).context("HTF captured Footprint rows overflow")?
                == self.row_count
                && u64::try_from(footprint.feature_column_count())
                    .context("HTF captured Footprint columns overflow")?
                    == self.footprint_feature_column_count
                && u64::try_from(footprint.retained_feature_device_bytes())
                    .context("HTF captured Footprint bytes overflow")?
                    == self.footprint_retained_feature_device_bytes
                && u64::try_from(footprint.prefix_scratch_device_bytes())
                    .context("HTF captured Footprint scratch overflow")?
                    == self.footprint_scratch_device_bytes,
            "HTF captured Footprint receipt differs from owner preflight"
        );
        Ok(())
    }
}

/// Move-only host recipe for one selected direct parent. It is complete enough
/// to mint the HTF recipe-v4 draft before CUDA acquisition, but deliberately
/// contains no native carrier, device byte observation, context, or stream.
#[must_use = "the HTF host parent recipe must move into preflight"]
#[derive(Debug)]
pub(crate) struct ResidentHigherTimeframeHostParentRecipeV3 {
    timeframe: CanonicalTimeframe,
    row_count: u64,
    routes: Vec<ResidentHigherTimeframeParentRouteV3>,
    parent_open_ms: Vec<i64>,
    parent_available_at_ms: Vec<Option<i64>>,
    availability_rule: &'static str,
    availability_lag_ms: Option<i64>,
    max_age_ms: Option<i64>,
    source_binding_sha256: [u8; 32],
    parent_store_identity_sha256: [u8; 32],
    memory: ResidentHigherTimeframeParentBatchMemoryFormulaV3,
}

pub(crate) fn prepare_resident_higher_timeframe_host_parent_v3(
    timeframe: CanonicalTimeframe,
    parent_open_ms: Vec<i64>,
    source_binding_sha256: [u8; 32],
    routes: Vec<ResidentHigherTimeframeParentRouteV3>,
    memory: ResidentHigherTimeframeParentBatchMemoryFormulaV3,
) -> Result<ResidentHigherTimeframeHostParentRecipeV3> {
    let row_count = u64::try_from(parent_open_ms.len()).context("HTF parent rows overflow")?;
    ensure!(
        row_count > 0 && row_count == memory.row_count,
        "HTF host parent clock differs from its owner memory formula"
    );
    ensure!(
        u64::try_from(routes.len()).context("HTF parent routes overflow")?
            == memory.source_route_count,
        "HTF host parent route census differs from its owner memory formula"
    );
    validate_canonical_millisecond_timestamps(&parent_open_ms)?;
    validate_canonical_cpu_route_order_v3(&routes)?;
    ensure!(
        source_binding_sha256 != [0; 32] && memory.expected_retained_parent_device_bytes > 0,
        "HTF host parent carries a zero source or retained extent"
    );
    let (parent_available_at_ms, availability_rule, availability_lag_ms, max_age_ms) =
        parent_availability_v3(timeframe, &parent_open_ms)?;
    let parent_store_identity_sha256 = host_parent_store_identity_sha256_v3(
        timeframe,
        &parent_open_ms,
        source_binding_sha256,
        &routes,
        memory.expected_retained_parent_device_bytes,
    )?;
    Ok(ResidentHigherTimeframeHostParentRecipeV3 {
        timeframe,
        row_count,
        routes,
        parent_open_ms,
        parent_available_at_ms,
        availability_rule,
        availability_lag_ms,
        max_age_ms,
        source_binding_sha256,
        parent_store_identity_sha256,
        memory,
    })
}

/// Host-only result of sealing one direct parent's six producer drafts. SMC
/// bindings move out first for the sole parent upload; every remaining launch
/// authority stays sealed inside the continuation.
#[must_use = "the direct-parent capture template must prepare SMC and launch once"]
#[derive(Debug)]
pub(crate) struct PreparedResidentHigherTimeframeDirectParentCaptureTemplateV3 {
    smc_bindings: Vec<ResidentFeatureColumnBindingV3>,
    pending: PendingResidentHigherTimeframeDirectParentCaptureLaunchV3,
}

impl PreparedResidentHigherTimeframeDirectParentCaptureTemplateV3 {
    pub(crate) fn into_smc_preparation_parts_v3(
        self,
    ) -> (
        Vec<ResidentFeatureColumnBindingV3>,
        PendingResidentHigherTimeframeDirectParentCaptureLaunchV3,
    ) {
        (self.smc_bindings, self.pending)
    }
}

#[must_use = "the direct-parent SMC owner and launch continuation must capture once"]
#[derive(Debug)]
pub(crate) struct PendingResidentHigherTimeframeDirectParentCaptureLaunchV3 {
    plan: ResidentHigherTimeframeDirectParentLaunchPlanV3,
    quant_parent: PendingResidentQuantHigherTimeframeParentV3,
    session_parent: PendingResidentSessionHigherTimeframeParentV2,
}

impl PendingResidentHigherTimeframeDirectParentCaptureLaunchV3 {
    pub(crate) fn capture_direct_parent_v3(
        self,
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        smc_materialization: ResidentSmcMaterializationV3,
    ) -> Result<ValidatedResidentHigherTimeframeDirectParentCaptureV3> {
        let Self {
            plan,
            quant_parent,
            session_parent,
        } = self;
        let capture = capture_resident_higher_timeframe_direct_parent_v3(
            run_device,
            smc_materialization,
            plan,
        )?;
        quant_parent.validate_captured_parent_receipt_v3(capture.quant_runtime_receipt())?;
        session_parent.validate_captured_parent_receipt_v2(capture.session_runtime_receipt())?;
        Ok(ValidatedResidentHigherTimeframeDirectParentCaptureV3 { capture })
    }
}

/// Opaque capture paired with the exact Quant/Session admissions that moved
/// their launch authorities into it. Receipt validation happens before this
/// value exists, and all evidence remains move-owned through native binding.
#[must_use = "the validated direct-parent capture must bind into HTF"]
#[derive(Debug)]
pub(crate) struct ValidatedResidentHigherTimeframeDirectParentCaptureV3 {
    capture: PendingResidentHigherTimeframeDirectParentCaptureV3,
}

impl ResidentHigherTimeframeNativeCarrierV3
    for ValidatedResidentHigherTimeframeDirectParentCaptureV3
{
    fn into_native_parent(
        self,
        authority: ResidentHigherTimeframeParentAuthorityV3,
    ) -> Result<ResidentHigherTimeframeDirectParentV3> {
        let Self { capture } = self;
        capture.into_direct_parent(authority).map_err(Into::into)
    }
}

/// Consume the exact six producer drafts for one pinned direct parent. The
/// canonical recipe-v4 assembler derives every local ordinal, parameter hash
/// and route receipt. The internal sentinel merely lets that already-frozen
/// seven-producer assembler seal; it is discarded before source routes or
/// launch bindings are returned.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_resident_higher_timeframe_direct_parent_owner_v3(
    source: &MaterializedPinnedResidentCanonicalSourceV1,
    smc_draft: ResidentProducerDraftV4,
    smc_memory: ResidentSmcPreDeviceMemoryReceiptV4,
    classic_draft: ResidentProducerDraftV4,
    classic_recipe: ResidentClassicTaRecipeV3,
    classic_memory: ResidentClassicTaPreDeviceMemoryReceiptV4,
    quant_draft: ResidentProducerDraftV4,
    quant_runtime: PreparedResidentQuantRuntimeV3,
    session_draft: ResidentProducerDraftV4,
    session_runtime: PreparedResidentSessionRuntimeV2,
    regime_draft: ResidentProducerDraftV4,
    regime_input: PreparedResidentRegimeInputV3,
    regime_memory: ResidentRegimePreDeviceMemoryReceiptV4,
    footprint_draft: ResidentProducerDraftV4,
    footprint_memory: ResidentFootprintPreDeviceMemoryReceiptV4,
) -> Result<(
    ResidentHigherTimeframeHostParentRecipeV3,
    PreparedResidentHigherTimeframeDirectParentCaptureTemplateV3,
)> {
    let mut quant_parent = quant_runtime.into_higher_timeframe_parent_v3();
    let quant_memory = quant_parent.higher_timeframe_batch_memory_v3()?;
    let mut session_parent = session_runtime.into_higher_timeframe_parent_v2();
    let session_memory = session_parent.higher_timeframe_batch_memory_v2()?;
    let memory =
        ResidentHigherTimeframeParentBatchMemoryFormulaV3::from_owner_preflight_receipts_v3(
            &smc_memory,
            &classic_memory,
            &quant_memory,
            &session_memory,
            &regime_memory,
            &footprint_memory,
        )?;

    let mut schema = ResidentColumnSchemaAssemblerV4::default();
    for draft in [
        smc_draft,
        classic_draft,
        quant_draft,
        session_draft,
        regime_draft,
        footprint_draft,
        direct_parent_schema_sentinel_draft_v3()?,
    ] {
        schema.append(draft)?;
    }
    let transforms = ResidentTransformCapabilityDraftV4::from_owner_capabilities(
        resident_robust_normalization_capability_v2()?,
        resident_canonical_content_sha256_capability_v3()?,
        resident_feature_major_to_bar_major_capability_v3()?,
    )?;
    let sealed = schema.seal(transforms)?;
    let mut routes = Vec::new();
    let mut smc_bindings = Vec::new();
    let mut classic_bindings = Vec::new();
    let mut quant_bindings = Vec::new();
    let mut session_bindings = Vec::new();
    let mut regime_bindings = Vec::new();
    let mut footprint_bindings = Vec::new();
    let mut sentinel_count = 0_usize;
    for route in sealed.routes() {
        let producer = route.producer();
        if producer == ResidentFeatureProducerV3::HigherTimeframeAlignment {
            sentinel_count = sentinel_count
                .checked_add(1)
                .context("HTF direct-parent sentinel count overflow")?;
            continue;
        }
        let binding = ResidentFeatureColumnBindingV3::from_admitted_route(route)?;
        routes.push(
            ResidentHigherTimeframeParentRouteV3::from_sealed_parent_route(
                route.feature_name(),
                producer,
                route.route_receipt_sha256(),
            )?,
        );
        match producer {
            ResidentFeatureProducerV3::Smc => smc_bindings.push(binding),
            ResidentFeatureProducerV3::ClassicTa => classic_bindings.push(binding),
            ResidentFeatureProducerV3::Quant => quant_bindings.push(binding),
            ResidentFeatureProducerV3::Session => session_bindings.push(binding),
            ResidentFeatureProducerV3::Regime => regime_bindings.push(binding),
            ResidentFeatureProducerV3::Footprint => footprint_bindings.push(binding),
            _ => bail!("HTF direct-parent schema admitted a non-column producer"),
        }
    }
    ensure!(
        sentinel_count == 1
            && [
                smc_bindings.as_slice(),
                classic_bindings.as_slice(),
                quant_bindings.as_slice(),
                session_bindings.as_slice(),
                regime_bindings.as_slice(),
                footprint_bindings.as_slice(),
            ]
            .iter()
            .all(|bindings| !bindings.is_empty()),
        "HTF direct-parent local schema omitted a producer or sentinel"
    );

    let parent_open_ms = source
        .frame()
        .ohlcv()
        .timestamp
        .as_ref()
        .context("HTF pinned direct parent has no canonical timestamp_ms")?
        .clone();
    let host_parent = prepare_resident_higher_timeframe_host_parent_v3(
        source.timeframe(),
        parent_open_ms,
        source.source_binding_sha256(),
        routes,
        memory,
    )?;
    let quant_launch_authority = quant_parent.take_launch_authority_v3()?;
    let session_launch_authority = session_parent.take_launch_authority_v2()?;
    let (regime_rows, regime_scale_anchor, regime_input_identity_sha256) = regime_input.consume();
    ensure!(
        regime_rows == source.frame().len() && regime_input_identity_sha256 != [0; 32],
        "HTF direct-parent Regime admission differs from pinned frame"
    );
    let plan = ResidentHigherTimeframeDirectParentLaunchPlanV3::seal(
        classic_recipe,
        classic_bindings,
        classic_memory,
        quant_bindings,
        quant_launch_authority,
        session_bindings,
        session_launch_authority,
        regime_bindings,
        regime_scale_anchor,
        footprint_bindings,
    )?;
    Ok((
        host_parent,
        PreparedResidentHigherTimeframeDirectParentCaptureTemplateV3 {
            smc_bindings,
            pending: PendingResidentHigherTimeframeDirectParentCaptureLaunchV3 {
                plan,
                quant_parent,
                session_parent,
            },
        },
    ))
}

fn direct_parent_schema_sentinel_draft_v3() -> Result<ResidentProducerDraftV4> {
    let route = ResidentRouteDraftV4::from_typed_parts(
        "__neoethos_htf_direct_parent_schema_sentinel_v3",
        Some(RESIDENT_HTF_INDICATOR_ID_V3),
        Some("direct_parent_schema_sentinel_v3"),
        ResidentFeatureStageV3::HigherTimeframeAligned,
        None,
        vec![parameter(
            "direct_parent_schema_sentinel",
            ResidentCanonicalParameterValueV4::Bool(true),
        )?],
        RESIDENT_HTF_ROUTE_DOMAIN_V3,
    )?;
    ResidentProducerDraftV4::from_owner_preflight(
        ResidentFeatureProducerV3::HigherTimeframeAlignment,
        HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION_V3,
        vec![route],
        vec![ResidentProducerBatchDraftV4::from_owner_preflight(
            0, 1, 0, 0,
        )],
        resident_higher_timeframe_capability_v3()?,
    )
    .map_err(Into::into)
}

/// Opaque direct-timeframe owner. `P` is the actual native parent carrier; it
/// is stored, never cloned, and returned only by consuming the complete HTF
/// producer owner after downstream pack-ready retirement.
#[must_use = "the retained direct-timeframe parent must move into HTF assembly"]
#[derive(Debug)]
pub(crate) struct RetainedResidentHigherTimeframeParentV3<P> {
    parent: P,
    timeframe: CanonicalTimeframe,
    row_count: u64,
    routes: Vec<ResidentHigherTimeframeParentRouteV3>,
    availability_rule: &'static str,
    availability_lag_ms: Option<i64>,
    max_age_ms: Option<i64>,
    source_binding_sha256: [u8; 32],
    parent_store_identity_sha256: [u8; 32],
    retained_parent_device_bytes: u64,
    parent_context_process_token: [u8; 32],
    parent_stream_process_token: [u8; 32],
}

/// Bind the post-launch opaque CUDA capture to the exact host recipe sealed
/// before run-device acquisition. Every route, owner-derived byte count,
/// producer receipt, device, context and stream must match; nothing is filled
/// with a synthetic execution identity.
fn bind_captured_parent_v3(
    run_device: &GpuOnlyRunDeviceAdmissionV3,
    recipe: ResidentHigherTimeframeHostParentRecipeV3,
    capture: ValidatedResidentHigherTimeframeDirectParentCaptureV3,
) -> Result<
    RetainedResidentHigherTimeframeParentV3<ValidatedResidentHigherTimeframeDirectParentCaptureV3>,
> {
    let native_capture = &capture.capture;
    ensure!(
        u64::try_from(native_capture.rows()).context("HTF captured rows overflow")?
            == recipe.row_count
            && native_capture.device_ordinal() == run_device.device_identity().ordinal()
            && native_capture.context_process_token()
                == run_device.device_identity().primary_context_process_token()
            && native_capture.stream_process_token() == run_device.run_stream_process_token_v3(),
        "HTF opaque capture shape/device/context/stream differs from its canonical frame"
    );
    ensure!(
        native_capture.route_descriptors().len() == recipe.routes.len(),
        "HTF opaque capture route census differs from host recipe"
    );
    for (expected_ordinal, (descriptor, source_route)) in native_capture
        .route_descriptors()
        .iter()
        .zip(&recipe.routes)
        .enumerate()
    {
        let binding = descriptor.binding();
        ensure!(
            binding.ordinal == expected_ordinal
                && binding.feature_name == source_route.feature_name
                && descriptor.producer() == source_route.producer
                && binding.canonical_parameter_tuple_sha256 != [0; 32]
                && binding.route_receipt_sha256 == source_route.source_route_receipt_sha256,
            "HTF opaque capture route differs from its sealed host recipe"
        );
    }
    ensure!(
        u64::try_from(native_capture.retained_device_bytes())
            .context("HTF opaque captured bytes overflow")?
            == recipe.memory.expected_retained_parent_device_bytes,
        "HTF opaque capture retained bytes differ from owner preflight"
    );
    recipe
        .memory
        .validate_captured_receipts_v3(native_capture)?;
    let ResidentHigherTimeframeHostParentRecipeV3 {
        timeframe,
        row_count,
        routes,
        parent_open_ms: _,
        parent_available_at_ms: _,
        availability_rule,
        availability_lag_ms,
        max_age_ms,
        source_binding_sha256,
        parent_store_identity_sha256,
        memory,
    } = recipe;
    Ok(RetainedResidentHigherTimeframeParentV3 {
        parent: capture,
        timeframe,
        row_count,
        routes,
        availability_rule,
        availability_lag_ms,
        max_age_ms,
        source_binding_sha256,
        parent_store_identity_sha256,
        retained_parent_device_bytes: memory.expected_retained_parent_device_bytes,
        parent_context_process_token: run_device.device_identity().primary_context_process_token(),
        parent_stream_process_token: run_device.run_stream_process_token_v3(),
    })
}

fn host_parent_store_identity_sha256_v3(
    timeframe: CanonicalTimeframe,
    parent_open_ms: &[i64],
    source_binding_sha256: [u8; 32],
    routes: &[ResidentHigherTimeframeParentRouteV3],
    expected_retained_parent_device_bytes: u64,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"neoethos.data.resident-htf-host-parent-recipe.semantic-v3\0");
    hash.update([timeframe.identity_tag()]);
    hash.update(
        u64::try_from(parent_open_ms.len())
            .context("HTF host parent clock extent overflow")?
            .to_le_bytes(),
    );
    for timestamp in parent_open_ms {
        hash.update(timestamp.to_le_bytes());
    }
    hash.update(source_binding_sha256);
    hash.update(expected_retained_parent_device_bytes.to_le_bytes());
    for route in routes {
        hash.update([route.producer as u8]);
        hash.update(route.feature_name.as_bytes());
        hash.update([0]);
        hash.update(route.source_route_receipt_sha256);
    }
    Ok(hash.finalize().into())
}

fn parent_availability_v3(
    timeframe: CanonicalTimeframe,
    parent_open_ms: &[i64],
) -> Result<(Vec<Option<i64>>, &'static str, Option<i64>, Option<i64>)> {
    if let Some(period_ms) = timeframe.fixed_duration_ms() {
        validate_fixed_parent_grid_v3(parent_open_ms, period_ms)?;
        let available = parent_open_ms
            .iter()
            .enumerate()
            .map(|(row, timestamp)| {
                timestamp.checked_add(period_ms).map(Some).with_context(|| {
                    format!("HTF fixed availability timestamp overflow at row {row}")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((
            available,
            "fixed_open_plus_period_v1",
            Some(period_ms),
            Some(period_ms.saturating_mul(2)),
        ))
    } else {
        let mut available = parent_open_ms
            .iter()
            .skip(1)
            .copied()
            .map(Some)
            .collect::<Vec<_>>();
        available.push(None);
        Ok((available, "next_direct_bar_open_v1", None, None))
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentHigherTimeframeGlobalParentSegmentV3 {
    parent_index: usize,
    first_column: usize,
    column_count: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ResidentHigherTimeframeGlobalBatchV3 {
    local_first_column: usize,
    column_count: usize,
    parent_segments: Vec<ResidentHigherTimeframeGlobalParentSegmentV3>,
}

fn build_global_batches_v3(
    parents: &[ResidentHigherTimeframeHostParentRecipeV3],
) -> Result<Vec<ResidentHigherTimeframeGlobalBatchV3>> {
    let total_columns = parents.iter().try_fold(0_usize, |sum, parent| {
        sum.checked_add(parent.routes.len())
            .context("HTF global live route census overflow")
    })?;
    ensure!(total_columns > 0, "HTF global live route census is empty");
    let mut batches = Vec::new();
    for local_first_column in (0..total_columns).step_by(MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V4) {
        let next_column = local_first_column
            .checked_add(MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V4)
            .map(|end| end.min(total_columns))
            .context("HTF global batch end overflow")?;
        let column_count = next_column
            .checked_sub(local_first_column)
            .context("HTF global batch width underflow")?;
        let mut parent_segments = Vec::new();
        let mut parent_first_column = 0_usize;
        for (parent_index, parent) in parents.iter().enumerate() {
            let parent_next_column = parent_first_column
                .checked_add(parent.routes.len())
                .context("HTF global parent span overflow")?;
            let overlap_first = local_first_column.max(parent_first_column);
            let overlap_next = next_column.min(parent_next_column);
            if overlap_first < overlap_next {
                parent_segments.push(ResidentHigherTimeframeGlobalParentSegmentV3 {
                    parent_index,
                    first_column: overlap_first
                        .checked_sub(local_first_column)
                        .context("HTF parent-segment first column underflow")?,
                    column_count: overlap_next
                        .checked_sub(overlap_first)
                        .context("HTF parent-segment width underflow")?,
                });
            }
            parent_first_column = parent_next_column;
        }
        let mut expected_first = 0_usize;
        for segment in &parent_segments {
            ensure!(
                segment.first_column == expected_first && segment.column_count > 0,
                "HTF global parent segments are not exact and contiguous"
            );
            expected_first = expected_first
                .checked_add(segment.column_count)
                .context("HTF global parent-segment coverage overflow")?;
        }
        ensure!(
            expected_first == column_count && !parent_segments.is_empty(),
            "HTF global parent segments do not cover their recipe-v4 batch"
        );
        batches.push(ResidentHigherTimeframeGlobalBatchV3 {
            local_first_column,
            column_count,
            parent_segments,
        });
    }
    Ok(batches)
}

#[derive(Debug)]
pub(crate) struct ResidentHigherTimeframeAllocationReceiptV3 {
    base_row_count: u64,
    parent_count: u64,
    feature_column_count: u64,
    retained_feature_device_bytes: u64,
    retained_parent_device_bytes: u64,
    scratch_device_bytes: u64,
    pointer_table_device_bytes: u64,
    pointer_table_h2d_bytes: u64,
    isolated_pointer_schema_metadata_bytes: u64,
    parent_input_h2d_bytes: u64,
    feature_value_d2h_bytes: u64,
    feature_validity_d2h_bytes: u64,
    native_launch_count: u64,
    native_kernel_launch_count: u64,
    producer_ready_event_count: u64,
    producer_ready_event_synchronize_count: u64,
    host_synchronize_count: u64,
}

/// Host-only runtime admission template. Device execution identities are
/// intentionally absent until exact opaque captures bind after CUDA launch.
#[derive(Debug)]
pub(crate) struct ResidentHigherTimeframeRuntimeAdmissionTemplateV3 {
    input_identity_sha256: [u8; 32],
    semantic_source_sha256: [u8; 32],
    implementation_sha256: [u8; 32],
    selected_parent_order: String,
    canonical_cpu_producer_order: String,
    allocation: ResidentHigherTimeframeAllocationReceiptV3,
}

#[derive(Debug)]
pub(crate) struct ResidentHigherTimeframeRuntimeAdmissionV3 {
    input_identity_sha256: [u8; 32],
    semantic_source_sha256: [u8; 32],
    implementation_sha256: [u8; 32],
    selected_parent_order: String,
    canonical_cpu_producer_order: String,
    base_context_process_token: [u8; 32],
    base_stream_process_token: [u8; 32],
    allocation: ResidentHigherTimeframeAllocationReceiptV3,
}

impl ResidentHigherTimeframeRuntimeAdmissionV3 {
    fn validate_native_receipt(
        &self,
        receipt: &ResidentHigherTimeframeRuntimeReceiptV3,
    ) -> Result<()> {
        ensure!(
            receipt.semantic_version() == HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION_V3,
            "resident HTF native semantic version drifted"
        );
        ensure!(
            u64::try_from(receipt.base_row_count())
                .context("HTF native base-row receipt overflow")?
                == self.allocation.base_row_count
                && u64::try_from(receipt.parent_count())
                    .context("HTF native parent-count receipt overflow")?
                    == self.allocation.parent_count
                && u64::try_from(receipt.parent_feature_column_count())
                    .context("HTF native feature-column receipt overflow")?
                    == self.allocation.feature_column_count,
            "resident HTF native shape drifted"
        );
        ensure!(
            u64::try_from(receipt.retained_feature_device_bytes())
                .context("HTF native retained-feature receipt overflow")?
                == self.allocation.retained_feature_device_bytes
                && u64::try_from(receipt.retained_parent_device_bytes())
                    .context("HTF native retained-parent receipt overflow")?
                    == self.allocation.retained_parent_device_bytes
                && u64::try_from(receipt.scratch_device_bytes())
                    .context("HTF native scratch receipt overflow")?
                    == self.allocation.scratch_device_bytes
                && u64::try_from(receipt.pointer_table_device_bytes())
                    .context("HTF native pointer-table receipt overflow")?
                    == self.allocation.pointer_table_device_bytes
                && u64::try_from(receipt.pointer_table_h2d_bytes())
                    .context("HTF native pointer H2D receipt overflow")?
                    == self.allocation.pointer_table_h2d_bytes
                && u64::try_from(receipt.isolated_pointer_schema_metadata_bytes())
                    .context("HTF native isolated-schema receipt overflow")?
                    == self.allocation.isolated_pointer_schema_metadata_bytes,
            "resident HTF native allocation receipt drifted"
        );
        ensure!(
            u64::try_from(receipt.parent_feature_h2d_bytes())
                .context("HTF native parent-feature H2D receipt overflow")?
                == self.allocation.parent_input_h2d_bytes
                && u64::try_from(receipt.feature_value_d2h_bytes())
                    .context("HTF native feature-value D2H receipt overflow")?
                    == self.allocation.feature_value_d2h_bytes
                && u64::try_from(receipt.feature_validity_d2h_bytes())
                    .context("HTF native feature-validity D2H receipt overflow")?
                    == self.allocation.feature_validity_d2h_bytes
                && u64::try_from(receipt.native_launch_count())
                    .context("HTF native ABI-launch receipt overflow")?
                    == self.allocation.native_launch_count
                && u64::try_from(receipt.native_kernel_launch_count())
                    .context("HTF native kernel-launch receipt overflow")?
                    == self.allocation.native_kernel_launch_count
                && u64::try_from(receipt.producer_ready_event_count())
                    .context("HTF native event receipt overflow")?
                    == self.allocation.producer_ready_event_count
                && u64::try_from(receipt.producer_ready_event_synchronize_count())
                    .context("HTF native event-sync receipt overflow")?
                    == self.allocation.producer_ready_event_synchronize_count
                && u64::try_from(receipt.host_synchronize_count())
                    .context("HTF native host-sync receipt overflow")?
                    == self.allocation.host_synchronize_count,
            "resident HTF native transfer/launch/event receipt drifted"
        );
        ensure!(
            receipt.logical_validity_codes() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                && receipt.logical_validity_schema()
                    == "neoethos.feature-cell-validity.logical-u8.codes-0-through-9.v3"
                && receipt.canonical_qnan_bits() == 0x7ff8_0000_0000_0000,
            "resident HTF native validity authority drifted"
        );
        ensure!(
            receipt.input_identity_sha256() == self.input_identity_sha256
                && receipt.semantic_source_sha256() == self.semantic_source_sha256
                && receipt.implementation_sha256() == self.implementation_sha256
                && receipt.selected_parent_order() == self.selected_parent_order.as_str()
                && receipt.canonical_cpu_producer_order()
                    == self.canonical_cpu_producer_order.as_str()
                && receipt.base_context_process_token() == self.base_context_process_token
                && receipt.base_stream_process_token() == self.base_stream_process_token,
            "resident HTF native source/order/context identity drifted"
        );
        Ok(())
    }
}

#[must_use = "resident HTF host preflight must move into recipe and later capture binding"]
#[derive(Debug)]
pub(crate) struct PreparedResidentHigherTimeframeProducerV3 {
    draft: ResidentProducerDraftV4,
    runtime_template: ResidentHigherTimeframeRuntimeAdmissionTemplateV3,
    parent_recipes: Vec<ResidentHigherTimeframeHostParentRecipeV3>,
    global_batches: Vec<ResidentHigherTimeframeGlobalBatchV3>,
}

impl PreparedResidentHigherTimeframeProducerV3 {
    pub(crate) fn into_recipe_and_runtime(
        self,
    ) -> (
        ResidentProducerDraftV4,
        PendingResidentHigherTimeframeRuntimeV3,
    ) {
        (
            self.draft,
            PendingResidentHigherTimeframeRuntimeV3 {
                runtime_template: self.runtime_template,
                parent_recipes: self.parent_recipes,
                global_batches: self.global_batches,
            },
        )
    }
}

#[must_use = "the HTF host runtime must bind exact opaque captures after device launch"]
#[derive(Debug)]
pub(crate) struct PendingResidentHigherTimeframeRuntimeV3 {
    runtime_template: ResidentHigherTimeframeRuntimeAdmissionTemplateV3,
    parent_recipes: Vec<ResidentHigherTimeframeHostParentRecipeV3>,
    global_batches: Vec<ResidentHigherTimeframeGlobalBatchV3>,
}

#[must_use = "captured HTF parents must bind resolved recipe-v4 routes and append once"]
#[derive(Debug)]
pub(crate) struct PendingResidentHigherTimeframeCapturedRuntimeV3 {
    runtime_admission: ResidentHigherTimeframeRuntimeAdmissionV3,
    retained_parents: Vec<
        RetainedResidentHigherTimeframeParentV3<
            ValidatedResidentHigherTimeframeDirectParentCaptureV3,
        >,
    >,
    global_batches: Vec<ResidentHigherTimeframeGlobalBatchV3>,
}

#[must_use = "the bound native HTF owner must append to the resident store once"]
#[derive(Debug)]
pub(crate) struct PreparedResidentHigherTimeframeAppendV3 {
    runtime_admission: ResidentHigherTimeframeRuntimeAdmissionV3,
    parents: Vec<ResidentHigherTimeframeDirectParentV3>,
    admitted_global_bindings: Vec<ResidentFeatureColumnBindingV3>,
    launch_authority: ResidentHigherTimeframeLaunchAuthorityV3,
}

impl PendingResidentHigherTimeframeRuntimeV3 {
    pub(crate) fn bind_captured_parents_v3(
        self,
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        captures: Vec<ValidatedResidentHigherTimeframeDirectParentCaptureV3>,
    ) -> Result<PendingResidentHigherTimeframeCapturedRuntimeV3> {
        let Self {
            runtime_template,
            parent_recipes,
            global_batches,
        } = self;
        ensure!(
            captures.len() == parent_recipes.len() && !captures.is_empty(),
            "resident HTF capture count differs from host parent recipes"
        );
        let mut retained_parents = Vec::with_capacity(parent_recipes.len());
        for (recipe, capture) in parent_recipes.into_iter().zip(captures) {
            retained_parents.push(bind_captured_parent_v3(run_device, recipe, capture)?);
        }
        let base_context_process_token =
            run_device.device_identity().primary_context_process_token();
        let base_stream_process_token = run_device.run_stream_process_token_v3();
        ensure!(
            base_context_process_token != [0; 32]
                && base_stream_process_token != [0; 32]
                && base_context_process_token != base_stream_process_token,
            "resident HTF captured runtime has invalid execution identities"
        );
        let ResidentHigherTimeframeRuntimeAdmissionTemplateV3 {
            input_identity_sha256,
            semantic_source_sha256,
            implementation_sha256,
            selected_parent_order,
            canonical_cpu_producer_order,
            allocation,
        } = runtime_template;
        Ok(PendingResidentHigherTimeframeCapturedRuntimeV3 {
            runtime_admission: ResidentHigherTimeframeRuntimeAdmissionV3 {
                input_identity_sha256,
                semantic_source_sha256,
                implementation_sha256,
                selected_parent_order,
                canonical_cpu_producer_order,
                base_context_process_token,
                base_stream_process_token,
                allocation,
            },
            retained_parents,
            global_batches,
        })
    }
}

impl PendingResidentHigherTimeframeCapturedRuntimeV3 {
    pub(crate) fn bind_current_native_v3(
        self,
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        admitted_global_bindings: Vec<ResidentFeatureColumnBindingV3>,
    ) -> Result<PreparedResidentHigherTimeframeAppendV3> {
        let Self {
            runtime_admission,
            retained_parents,
            global_batches,
        } = self;
        ensure!(
            run_device.device_identity().primary_context_process_token()
                == runtime_admission.base_context_process_token
                && run_device.run_stream_process_token_v3()
                    == runtime_admission.base_stream_process_token,
            "resident HTF runtime was rebound to a different base context/stream"
        );
        validate_resolved_global_bindings_v3(
            &retained_parents,
            &global_batches,
            &admitted_global_bindings,
        )?;
        let mut binding_cursor = 0_usize;
        let mut parents = Vec::with_capacity(retained_parents.len());
        for retained_parent in retained_parents {
            let binding_end = binding_cursor
                .checked_add(retained_parent.routes.len())
                .context("HTF resolved parent binding end overflow")?;
            let parent_bindings = admitted_global_bindings
                .get(binding_cursor..binding_end)
                .context("HTF resolved parent binding span is incomplete")?;
            parents.push(seal_native_parent_v3(retained_parent, parent_bindings)?);
            binding_cursor = binding_end;
        }
        ensure!(
            binding_cursor == admitted_global_bindings.len(),
            "HTF resolved bindings contain an unowned tail"
        );
        let source_closure = seal_resident_higher_timeframe_source_closure_v3();
        ensure!(
            source_closure.implementation_sha256() == runtime_admission.implementation_sha256,
            "resident HTF native implementation/source closure drifted"
        );
        let launch_authority = ResidentHigherTimeframeLaunchAuthorityV3::seal(
            usize::try_from(runtime_admission.allocation.base_row_count)
                .context("HTF launch base-row extent overflow")?,
            runtime_admission.input_identity_sha256,
            runtime_admission.semantic_source_sha256,
            runtime_admission.selected_parent_order.clone(),
            runtime_admission.canonical_cpu_producer_order.clone(),
            runtime_admission.base_context_process_token,
            runtime_admission.base_stream_process_token,
            source_closure,
        )?;
        Ok(PreparedResidentHigherTimeframeAppendV3 {
            runtime_admission,
            parents,
            admitted_global_bindings,
            launch_authority,
        })
    }
}

impl PreparedResidentHigherTimeframeAppendV3 {
    pub(crate) fn append_to(
        self,
        assembler: &mut ResidentFeatureStoreAssemblerV3,
    ) -> std::result::Result<
        (
            ResidentHigherTimeframeRuntimeAdmissionV3,
            ResidentHigherTimeframeRuntimeReceiptV3,
        ),
        ResidentFeatureStoreCudaErrorV3,
    > {
        let Self {
            runtime_admission,
            parents,
            admitted_global_bindings,
            launch_authority,
        } = self;
        let receipt = assembler.append_resident_higher_timeframe_alignment_v3(
            parents,
            admitted_global_bindings,
            launch_authority,
        )?;
        runtime_admission
            .validate_native_receipt(&receipt)
            .map_err(|error| ResidentFeatureStoreCudaErrorV3::InvalidInput(error.to_string()))?;
        Ok((runtime_admission, receipt))
    }
}

fn validate_resolved_global_bindings_v3<P>(
    parents: &[RetainedResidentHigherTimeframeParentV3<P>],
    global_batches: &[ResidentHigherTimeframeGlobalBatchV3],
    admitted_global_bindings: &[ResidentFeatureColumnBindingV3],
) -> Result<()> {
    let expected_columns = parents.iter().try_fold(0_usize, |sum, parent| {
        sum.checked_add(parent.routes.len())
            .context("HTF resolved route census overflow")
    })?;
    ensure!(
        expected_columns == admitted_global_bindings.len() && !global_batches.is_empty(),
        "HTF resolved route census differs from its live direct parents"
    );
    let first_ordinal = admitted_global_bindings
        .first()
        .context("HTF resolved bindings are empty")?
        .ordinal;
    let mut binding_cursor = 0_usize;
    for parent in parents {
        for source_route in &parent.routes {
            let binding = admitted_global_bindings
                .get(binding_cursor)
                .context("HTF resolved binding span ended early")?;
            ensure!(
                binding.ordinal
                    == first_ordinal
                        .checked_add(binding_cursor)
                        .context("HTF resolved global ordinal overflow")?
                    && binding.feature_name
                        == format!(
                            "{}_{}",
                            parent.timeframe.as_str(),
                            source_route.feature_name
                        )
                    && binding.canonical_parameter_tuple_sha256 != [0; 32]
                    && binding.route_receipt_sha256 != [0; 32],
                "HTF resolved recipe-v4 binding differs from its live parent route"
            );
            binding_cursor = binding_cursor
                .checked_add(1)
                .context("HTF resolved binding cursor overflow")?;
        }
    }
    let mut expected_first_column = 0_usize;
    for batch in global_batches {
        ensure!(
            batch.local_first_column == expected_first_column
                && batch.column_count > 0
                && batch.column_count <= MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V4
                && !batch.parent_segments.is_empty(),
            "HTF recipe-v4 global batch ledger drifted"
        );
        expected_first_column = expected_first_column
            .checked_add(batch.column_count)
            .context("HTF recipe-v4 batch coverage overflow")?;
    }
    ensure!(
        binding_cursor == expected_columns && expected_first_column == expected_columns,
        "HTF recipe-v4 batches do not cover every resolved live route"
    );
    Ok(())
}

fn seal_native_parent_v3<P>(
    retained_parent: RetainedResidentHigherTimeframeParentV3<P>,
    output_bindings: &[ResidentFeatureColumnBindingV3],
) -> Result<ResidentHigherTimeframeDirectParentV3>
where
    P: ResidentHigherTimeframeNativeCarrierV3,
{
    let RetainedResidentHigherTimeframeParentV3 {
        parent,
        timeframe,
        row_count,
        routes,
        availability_rule,
        availability_lag_ms,
        max_age_ms,
        source_binding_sha256,
        parent_store_identity_sha256,
        retained_parent_device_bytes,
        parent_context_process_token,
        parent_stream_process_token,
    } = retained_parent;
    ensure!(
        routes.len() == output_bindings.len(),
        "HTF native parent route/output span differs"
    );
    let mut native_routes = Vec::with_capacity(routes.len());
    for (source_route, output_binding) in routes.into_iter().zip(output_bindings) {
        native_routes.push(ResidentHigherTimeframeRouteAuthorityV3::seal(
            source_route.feature_name,
            source_route.producer,
            source_route.source_route_receipt_sha256,
            output_binding.route_receipt_sha256,
        )?);
    }
    let (native_availability_rule, fixed_period_ms, native_max_age_ms) = match availability_rule {
        "fixed_open_plus_period_v1" => (
            ResidentHigherTimeframeAvailabilityRuleV3::FixedOpenPlusPeriod,
            availability_lag_ms.context("HTF fixed parent lost its availability lag")?,
            max_age_ms.context("HTF fixed parent lost its max age")?,
        ),
        "next_direct_bar_open_v1" => (
            ResidentHigherTimeframeAvailabilityRuleV3::NextDirectBarOpen,
            0,
            -1,
        ),
        _ => bail!("HTF retained parent has an unsupported availability rule"),
    };
    let authority = ResidentHigherTimeframeParentAuthorityV3::seal(
        timeframe.as_str(),
        usize::try_from(row_count).context("HTF native parent row extent overflow")?,
        native_availability_rule,
        fixed_period_ms,
        native_max_age_ms,
        source_binding_sha256,
        parent_store_identity_sha256,
        usize::try_from(retained_parent_device_bytes)
            .context("HTF native retained-parent bytes overflow")?,
        parent_context_process_token,
        parent_stream_process_token,
        native_routes,
    )?;
    parent.into_native_parent(authority)
}

pub(crate) fn preflight_resident_higher_timeframe_alignment_v3(
    base_timeframe: CanonicalTimeframe,
    base_open_ms: &[i64],
    parent_recipes: Vec<ResidentHigherTimeframeHostParentRecipeV3>,
    native_capability: ResidentProducerCapabilityV3,
) -> Result<PreparedResidentHigherTimeframeProducerV3> {
    ensure!(
        native_capability.producer() == ResidentFeatureProducerV3::HigherTimeframeAlignment,
        "resident HTF native capability has the wrong producer"
    );
    ensure!(
        native_capability.implementation_id() == RESIDENT_HTF_IMPLEMENTATION_ID_V3,
        "resident HTF native capability has the wrong implementation id"
    );
    ensure!(
        native_capability.exact_math_authority() == RESIDENT_HTF_EXACT_MATH_AUTHORITY_V3,
        "resident HTF native capability has the wrong exact-math authority"
    );
    let implementation_sha256 = native_capability.implementation_sha256();
    let semantic_source_sha256 = htf_semantic_source_sha256_v3();
    validate_canonical_millisecond_timestamps(base_open_ms)?;
    ensure!(
        !parent_recipes.is_empty(),
        "resident HTF has no selected direct parents"
    );

    let mut selected = BTreeSet::new();
    for parent in &parent_recipes {
        ensure!(
            parent.timeframe > base_timeframe,
            "resident HTF parent {} is not above base {base_timeframe}",
            parent.timeframe
        );
        ensure!(
            selected.insert(parent.timeframe),
            "resident HTF repeats a selected parent"
        );
    }
    let selected_parent_order = parent_recipes
        .iter()
        .map(|parent| parent.timeframe.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let canonical_cpu_producer_order = CANONICAL_CPU_PRODUCER_ORDER_V3
        .iter()
        .map(|producer| producer.as_str())
        .collect::<Vec<_>>()
        .join(",");
    let input_identity_sha256 = htf_input_identity_v3(
        base_timeframe,
        base_open_ms,
        &parent_recipes,
        &selected_parent_order,
        &canonical_cpu_producer_order,
    );
    ensure!(
        input_identity_sha256 != [0; 32],
        "resident HTF input identity is zero"
    );

    let retained_parent_device_bytes = parent_recipes.iter().try_fold(0_u64, |sum, parent| {
        sum.checked_add(parent.memory.expected_retained_parent_device_bytes)
            .context("resident HTF retained parent bytes overflow")
    })?;
    let route_count = parent_recipes.iter().try_fold(0_usize, |sum, parent| {
        sum.checked_add(parent.routes.len())
            .context("resident HTF route count overflow")
    })?;
    ensure!(
        route_count > 0,
        "resident HTF parents expose no source routes"
    );
    let aligned_route_names = parent_recipes
        .iter()
        .flat_map(|parent| {
            parent
                .routes
                .iter()
                .map(|route| format!("{}_{}", parent.timeframe.as_str(), route.feature_name))
        })
        .collect::<Vec<_>>();
    let (pointer_table_device_bytes, isolated_pointer_schema_metadata_bytes) =
        htf_pointer_schema_bytes_v3(&aligned_route_names)?;
    let mut routes = Vec::with_capacity(route_count);
    for parent in &parent_recipes {
        for source_route in &parent.routes {
            routes.push(htf_route_v3(
                parent,
                source_route,
                input_identity_sha256,
                &selected_parent_order,
                &canonical_cpu_producer_order,
            )?);
        }
    }
    let global_batches = build_global_batches_v3(&parent_recipes)?;
    let batches = global_batches
        .iter()
        .map(|batch| {
            ResidentProducerBatchDraftV4::from_owner_preflight(
                batch.local_first_column,
                batch.column_count,
                retained_parent_device_bytes,
                0,
            )
        })
        .collect::<Vec<_>>();
    let native_launch_count = u64::try_from(batches.len()).context("HTF launch count overflow")?;
    let native_kernel_launch_count = global_batches.iter().try_fold(0_u64, |sum, batch| {
        sum.checked_add(
            u64::try_from(batch.parent_segments.len())
                .context("HTF parent-segment launch count overflow")?,
        )
        .context("HTF native kernel launch count overflow")
    })?;
    let draft = ResidentProducerDraftV4::from_owner_preflight(
        ResidentFeatureProducerV3::HigherTimeframeAlignment,
        HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION_V3,
        routes,
        batches,
        native_capability,
    )?;
    let base_row_count = u64::try_from(base_open_ms.len()).context("HTF base rows overflow")?;
    let feature_column_count = u64::try_from(route_count).context("HTF columns overflow")?;
    let parent_count = u64::try_from(parent_recipes.len()).context("HTF parent count overflow")?;
    let retained_feature_device_bytes = base_row_count
        .checked_mul(feature_column_count)
        .and_then(|cells| cells.checked_mul(9))
        .context("HTF retained aligned output bytes overflow")?;
    let pointer_table_h2d_bytes = feature_column_count
        .checked_mul(4)
        .and_then(|entries| entries.checked_mul(u64::BITS as u64 / 8))
        .context("HTF pointer-table H2D bytes overflow")?;
    Ok(PreparedResidentHigherTimeframeProducerV3 {
        draft,
        runtime_template: ResidentHigherTimeframeRuntimeAdmissionTemplateV3 {
            input_identity_sha256,
            semantic_source_sha256,
            implementation_sha256,
            selected_parent_order,
            canonical_cpu_producer_order,
            allocation: ResidentHigherTimeframeAllocationReceiptV3 {
                base_row_count,
                parent_count,
                feature_column_count,
                retained_feature_device_bytes,
                retained_parent_device_bytes,
                scratch_device_bytes: 0,
                pointer_table_device_bytes,
                pointer_table_h2d_bytes,
                isolated_pointer_schema_metadata_bytes,
                parent_input_h2d_bytes: 0,
                feature_value_d2h_bytes: 0,
                feature_validity_d2h_bytes: 0,
                native_launch_count,
                native_kernel_launch_count,
                producer_ready_event_count: native_launch_count,
                producer_ready_event_synchronize_count: 0,
                host_synchronize_count: 0,
            },
        },
        parent_recipes,
        global_batches,
    })
}

fn validate_canonical_cpu_route_order_v3(
    routes: &[ResidentHigherTimeframeParentRouteV3],
) -> Result<()> {
    ensure!(!routes.is_empty(), "HTF parent has no routes");
    let mut cursor = 0_usize;
    let mut names = BTreeSet::new();
    for expected in CANONICAL_CPU_PRODUCER_ORDER_V3 {
        let start = cursor;
        while cursor < routes.len() && routes[cursor].producer == expected {
            ensure!(
                names.insert(routes[cursor].feature_name.as_str()),
                "HTF parent repeats feature name {}",
                routes[cursor].feature_name
            );
            cursor += 1;
        }
        ensure!(
            cursor > start,
            "HTF parent omits canonical producer {}",
            expected.as_str()
        );
    }
    ensure!(
        cursor == routes.len(),
        "HTF parent route order has a noncanonical tail"
    );
    Ok(())
}

fn validate_fixed_parent_grid_v3(timestamps: &[i64], period_ms: i64) -> Result<()> {
    ensure!(period_ms > 0, "HTF fixed period is not positive");
    for (row, timestamp) in timestamps.iter().copied().enumerate() {
        ensure!(
            timestamp.rem_euclid(period_ms) == 0,
            "HTF fixed parent row {row} is off its declared epoch grid"
        );
    }
    for (row, pair) in timestamps.windows(2).enumerate() {
        let gap = pair[1]
            .checked_sub(pair[0])
            .context("HTF fixed parent timestamp gap overflow")?;
        ensure!(
            gap > 0 && gap.rem_euclid(period_ms) == 0,
            "HTF fixed parent gap ending at row {} is not a positive period multiple",
            row + 1
        );
    }
    Ok(())
}

fn htf_route_v3(
    parent: &ResidentHigherTimeframeHostParentRecipeV3,
    source_route: &ResidentHigherTimeframeParentRouteV3,
    input_identity_sha256: [u8; 32],
    selected_parent_order: &str,
    canonical_cpu_producer_order: &str,
) -> Result<ResidentRouteDraftV4> {
    let aligned_name = format!(
        "{}_{}",
        parent.timeframe.as_str(),
        source_route.feature_name
    );
    let parameters = vec![
        parameter(
            "input_identity_sha256",
            ResidentCanonicalParameterValueV4::Hash(input_identity_sha256),
        )?,
        parameter(
            "source_binding_sha256",
            ResidentCanonicalParameterValueV4::Hash(parent.source_binding_sha256),
        )?,
        parameter(
            "parent_store_identity_sha256",
            ResidentCanonicalParameterValueV4::Hash(parent.parent_store_identity_sha256),
        )?,
        parameter(
            "source_route_receipt_sha256",
            ResidentCanonicalParameterValueV4::Hash(source_route.source_route_receipt_sha256),
        )?,
        parameter(
            "timeframe",
            ResidentCanonicalParameterValueV4::Text(parent.timeframe.as_str().to_owned()),
        )?,
        parameter(
            "source_feature_name",
            ResidentCanonicalParameterValueV4::Text(source_route.feature_name.clone()),
        )?,
        parameter(
            "source_producer",
            ResidentCanonicalParameterValueV4::Text(source_route.producer.as_str().to_owned()),
        )?,
        parameter(
            "selected_parent_order",
            ResidentCanonicalParameterValueV4::Text(selected_parent_order.to_owned()),
        )?,
        parameter(
            "canonical_cpu_producer_order",
            ResidentCanonicalParameterValueV4::Text(canonical_cpu_producer_order.to_owned()),
        )?,
        parameter(
            "availability_rule",
            ResidentCanonicalParameterValueV4::Text(parent.availability_rule.to_owned()),
        )?,
        parameter(
            "availability_lag_ms",
            ResidentCanonicalParameterValueV4::I64(parent.availability_lag_ms.unwrap_or(-1)),
        )?,
        parameter(
            "max_age_ms",
            ResidentCanonicalParameterValueV4::I64(parent.max_age_ms.unwrap_or(-1)),
        )?,
        parameter(
            "forward_fill",
            ResidentCanonicalParameterValueV4::Bool(true),
        )?,
        parameter(
            "parent_row_count",
            ResidentCanonicalParameterValueV4::U64(parent.row_count),
        )?,
    ];
    ResidentRouteDraftV4::from_typed_parts(
        aligned_name.clone(),
        Some(RESIDENT_HTF_INDICATOR_ID_V3),
        Some(aligned_name),
        ResidentFeatureStageV3::HigherTimeframeAligned,
        None,
        parameters,
        RESIDENT_HTF_ROUTE_DOMAIN_V3,
    )
    .map_err(Into::into)
}

fn htf_pointer_schema_bytes_v3(names: &[String]) -> Result<(u64, u64)> {
    let mut max_pointer = 0_u64;
    let mut max_isolated = 0_u64;
    for batch in names.chunks(MAX_RESIDENT_PRODUCER_BATCH_COLUMNS_V4) {
        let count =
            u64::try_from(batch.len()).context("HTF pointer-table column count overflow")?;
        let pointer = count
            .checked_mul(4 * u64::BITS as u64 / 8)
            .context("HTF pointer-table bytes overflow")?;
        let name_offsets = count
            .checked_add(1)
            .and_then(|extent| extent.checked_mul(u64::BITS as u64 / 8))
            .context("HTF name-offset bytes overflow")?;
        let name_bytes = batch.iter().try_fold(0_u64, |sum, name| {
            sum.checked_add(name.len() as u64)
                .context("HTF route-name bytes overflow")
        })?;
        let isolated = pointer
            .checked_add(name_offsets)
            .and_then(|bytes| bytes.checked_add(name_bytes))
            .context("HTF isolated pointer/schema bytes overflow")?;
        max_pointer = max_pointer.max(pointer);
        max_isolated = max_isolated.max(isolated);
    }
    ensure!(
        max_pointer > 0 && max_isolated > 0,
        "HTF pointer/schema extent is empty"
    );
    Ok((max_pointer, max_isolated))
}

fn htf_semantic_source_sha256_v3() -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"neoethos.data.resident-higher-timeframe-semantic-source.v3\0");
    hash.update(include_bytes!("features.rs"));
    hash.update(include_bytes!("../lib.rs"));
    hash.update(RESIDENT_HTF_EXACT_MATH_AUTHORITY_V3.as_bytes());
    hash.finalize().into()
}

fn parameter(
    name: &'static str,
    value: ResidentCanonicalParameterValueV4,
) -> Result<ResidentCanonicalParameterV4> {
    ResidentCanonicalParameterV4::from_typed_value(name, value).map_err(Into::into)
}

fn htf_input_identity_v3(
    base_timeframe: CanonicalTimeframe,
    base_open_ms: &[i64],
    parents: &[ResidentHigherTimeframeHostParentRecipeV3],
    selected_parent_order: &str,
    canonical_cpu_producer_order: &str,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"neoethos.data.resident-higher-timeframe-input.semantic-v3\0");
    hash.update(HIGHER_TIMEFRAME_ALIGNMENT_SEMANTIC_VERSION_V3.to_le_bytes());
    hash.update([base_timeframe.identity_tag()]);
    hash.update((base_open_ms.len() as u64).to_le_bytes());
    for timestamp in base_open_ms {
        hash.update(timestamp.to_le_bytes());
    }
    hash.update(selected_parent_order.as_bytes());
    hash.update([0]);
    hash.update(canonical_cpu_producer_order.as_bytes());
    for parent in parents {
        hash.update([parent.timeframe.identity_tag()]);
        hash.update(parent.row_count.to_le_bytes());
        hash.update(parent.source_binding_sha256);
        hash.update(parent.parent_store_identity_sha256);
        hash.update(
            parent
                .memory
                .expected_retained_parent_device_bytes
                .to_le_bytes(),
        );
        hash.update(parent.availability_rule.as_bytes());
        hash.update(parent.availability_lag_ms.unwrap_or(-1).to_le_bytes());
        hash.update(parent.max_age_ms.unwrap_or(-1).to_le_bytes());
        for (&open, available) in parent
            .parent_open_ms
            .iter()
            .zip(&parent.parent_available_at_ms)
        {
            hash.update(open.to_le_bytes());
            match available {
                Some(timestamp) => {
                    hash.update([1]);
                    hash.update(timestamp.to_le_bytes());
                }
                None => hash.update([0]),
            }
        }
        for route in &parent.routes {
            hash.update(route.feature_name.as_bytes());
            hash.update([0]);
            hash.update([route.producer as u8]);
            hash.update(route.source_route_receipt_sha256);
        }
    }
    hash.finalize().into()
}
