//! One-shot Search binding for a sealed Data V3 resident feature store.
//!
//! This module accepts only the opaque sealed store and the canonical Search
//! scope. Device handles and route selection remain owned by the admitted
//! store and the gpu-cuda session wrapper.

use crate::data_selection::CanonicalGpuResidentSearchArtifactScopeV3;
use anyhow::{Context, Result, bail};
use neoethos_data::SealedGpuResidentFeatureStoreV3;
use neoethos_gpu_cuda::SealedDataPopulationExecutionLimitsV1;
use neoethos_gpu_cuda::resident_feature_store_v3::{
    ResidentFeatureStoreConsumerLeaseV3, ResidentFeatureStoreImportV3, ResidentPopulationSessionV3,
};

const SEALED_STORE_AUTHORITY_V3: &str = "neoethos.data.sealed-gpu-resident-feature-store.v3";

/// One move-only native Search run whose parent features remain owned by the
/// sealed Data V3 store. This type is intentionally separate from the host
/// population run: constructing it never materializes host feature or base-bar
/// arrays and never creates a V1 population parent.
pub struct StrictResidentPopulationExecutionRunV3 {
    scope: CanonicalGpuResidentSearchArtifactScopeV3,
    session: Option<ResidentPopulationSessionV3>,
    row_count: usize,
    column_count: usize,
}

impl StrictResidentPopulationExecutionRunV3 {
    pub const fn scope(&self) -> &CanonicalGpuResidentSearchArtifactScopeV3 {
        &self.scope
    }

    pub const fn row_count(&self) -> usize {
        self.row_count
    }

    pub const fn column_count(&self) -> usize {
        self.column_count
    }

    fn resident_session(&self) -> Result<&ResidentPopulationSessionV3> {
        self.session
            .as_ref()
            .context("resident V3 Search run has no bound population session")
    }

    /// The selected ordinal from the already-admitted resident session. This
    /// is metadata-only and never performs another CUDA inventory probe.
    pub fn selected_device_ordinal(&self) -> Result<u32> {
        Ok(self.resident_session()?.device_identity().ordinal())
    }

    /// Identity of the exact CUDA run admission that owns both the resident
    /// feature parent and this population session.
    pub fn cuda_admission_identity_sha256(&self) -> Result<[u8; 32]> {
        Ok(self.resident_session()?.admission_identity_sha256())
    }

    /// Same-context free-memory snapshot captured before Data materialized the
    /// resident parent. Population sizing must consume this sealed value and
    /// must not query a later, already-depleted free-memory state.
    pub fn pre_parent_free_memory_bytes(&self) -> Result<u64> {
        Ok(self
            .resident_session()?
            .pre_materialization_free_bytes_snapshot())
    }

    /// Canonical semantic-v3 Merkle identity of the resident parent values,
    /// validity and timestamp domain used by this exact Search run.
    pub fn parent_content_identity_sha256(&self) -> Result<[u8; 32]> {
        Ok(self.resident_session()?.canonical_content_merkle())
    }

    /// Process-local token minted only after the post-free producer-stream
    /// synchronization completed and before the population parent allocated.
    pub fn data_transient_retirement_process_token(&self) -> Result<[u8; 32]> {
        let token = self
            .resident_session()?
            .data_transient_retirement_process_token();
        if token == [0; 32] {
            bail!("resident Search run lacks Data transient-retirement proof");
        }
        Ok(token)
    }

    pub fn data_population_limits(&self) -> Result<&SealedDataPopulationExecutionLimitsV1> {
        self.resident_session()?
            .data_population_limits()
            .context("resident Search run lacks a sealed Data+population workspace authority")
    }

    /// Versioned Search scope identity bound to the same resident parent.
    pub fn scope_identity_sha256(&self) -> Result<String> {
        self.scope
            .identity_sha256()
            .context("seal canonical GPU-resident Search scope identity")
    }

    /// Purpose-bound access to the resident population evaluator. The raw V1
    /// `PopulationSession` is never exposed: the callback can only use the V3
    /// session already bound to this store, context, stream and ready event.
    pub fn with_resident_population_session_v3<Output, Consumer>(
        &mut self,
        consumer: Consumer,
    ) -> Result<Output>
    where
        Consumer: FnOnce(&mut ResidentPopulationSessionV3) -> Result<Output>,
    {
        self.scope
            .validate()
            .context("validate resident V3 Search scope before population evaluation")?;
        let row_count = self.row_count;
        let column_count = self.column_count;
        let expected_admission = self.cuda_admission_identity_sha256()?;
        let expected_content = self.parent_content_identity_sha256()?;
        let expected_transient_retirement = self.data_transient_retirement_process_token()?;
        let expected_workspace = self
            .data_population_limits()?
            .workspace_plan_identity_sha256();
        let session = self
            .session
            .as_mut()
            .context("resident V3 Search run has no session to evaluate")?;
        if session.rows() != row_count
            || session.columns() != column_count
            || session.admission_identity_sha256() != expected_admission
            || session.canonical_content_merkle() != expected_content
            || session.data_transient_retirement_process_token() != expected_transient_retirement
            || session
                .data_population_limits()
                .map(SealedDataPopulationExecutionLimitsV1::workspace_plan_identity_sha256)
                != Some(expected_workspace)
        {
            bail!("resident V3 population session drifted before evaluation");
        }
        let outcome = consumer(session);
        if session.rows() != row_count
            || session.columns() != column_count
            || session.admission_identity_sha256() != expected_admission
            || session.canonical_content_merkle() != expected_content
            || session.data_transient_retirement_process_token() != expected_transient_retirement
            || session
                .data_population_limits()
                .map(SealedDataPopulationExecutionLimitsV1::workspace_plan_identity_sha256)
                != Some(expected_workspace)
        {
            bail!("resident V3 population session drifted during evaluation");
        }
        outcome
    }
}

fn hex_lower(bytes: [u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn validate_strict_resident_feature_store_v3(
    sealed_store: &SealedGpuResidentFeatureStoreV3,
    scope: &CanonicalGpuResidentSearchArtifactScopeV3,
) -> Result<()> {
    scope
        .validate()
        .context("validate canonical GPU-resident Search V3 scope")?;
    let scope_rows = scope
        .evaluated_window()
        .row_end()
        .checked_sub(scope.evaluated_window().row_start())
        .and_then(|rows| usize::try_from(rows).ok())
        .context("canonical Search window row extent does not fit this process")?;
    let admission_identity = sealed_store.admission_identity_sha256();
    let feature_plan = sealed_store.final_feature_plan_v3_sha256();
    let normalization = sealed_store.normalization_fit_sha256();
    let provenance = sealed_store.source_provenance_sha256();
    let content = sealed_store
        .contract()
        .canonical_feature_content_merkle_sha256();
    let ordered_feature_count = sealed_store.ordered_feature_names().len();
    let device_identity = sealed_store.device_identity();
    let build_identity = device_identity.native_sass_target();
    let ordinal = sealed_store.device_ordinal();
    let context_identity = device_identity.primary_context_process_token();
    let stream_identity = sealed_store.ready_event();
    let resident_rows = usize::try_from(sealed_store.contract().layout().row_count())
        .context("resident V3 row extent does not fit this process")?;
    let resident_columns = usize::try_from(sealed_store.contract().layout().column_count())
        .context("resident V3 column extent does not fit this process")?;

    if sealed_store.authority_id() != SEALED_STORE_AUTHORITY_V3
        || admission_identity == [0; 32]
        || feature_plan == [0; 32]
        || normalization == [0; 32]
        || provenance == [0; 32]
        || content == [0; 32]
        || hex_lower(feature_plan) != scope.receipt().feature_plan_identity()
        || hex_lower(provenance) != scope.receipt().feature_provenance_identity()
        || hex_lower(content) != scope.receipt().feature_content_merkle_sha256()
        || hex_lower(normalization) != scope.receipt().normalization_fit_sha256()
        || u64::try_from(resident_rows).ok() != Some(scope.receipt().row_count())
        || u64::try_from(resident_columns).ok() != Some(scope.receipt().column_count())
        || scope_rows != resident_rows
        || ordered_feature_count != resident_columns
        || ordinal != device_identity.ordinal()
        || context_identity == [0; 32]
        || build_identity.trim().is_empty()
        || stream_identity.host_synchronize_count() != 0
        || !stream_identity.consumer_must_wait_before_first_read()
        || !stream_identity.retains_store_until_consumer_completion()
    {
        bail!(
            "sealed resident V3 store does not match the exact Search scope, shape or admitted route"
        );
    }
    Ok(())
}

pub(crate) fn bind_strict_resident_feature_store_v3_run_input(
    sealed_store: SealedGpuResidentFeatureStoreV3,
    scope: &CanonicalGpuResidentSearchArtifactScopeV3,
) -> Result<StrictResidentPopulationExecutionRunV3> {
    validate_strict_resident_feature_store_v3(&sealed_store, scope)?;
    let resident_import = sealed_store
        .into_resident_feature_store_import_v3()
        .context("consume validated sealed Data V3 store into its admitted-stream import")?;
    bind_resident_feature_store_v3(resident_import, scope)
}

pub(crate) fn bind_resident_feature_store_v3(
    resident_import: ResidentFeatureStoreImportV3,
    scope: &CanonicalGpuResidentSearchArtifactScopeV3,
) -> Result<StrictResidentPopulationExecutionRunV3> {
    scope
        .validate()
        .context("validate canonical GPU-resident Search V3 scope")?;
    let scope_rows = scope
        .evaluated_window()
        .row_end()
        .checked_sub(scope.evaluated_window().row_start())
        .and_then(|rows| usize::try_from(rows).ok())
        .context("canonical Search window row extent does not fit this process")?;
    let selected_ordinal = resident_import
        .device_ordinal()
        .context("read sealed resident V3 import ordinal")?;
    let admission_identity = resident_import
        .admission_identity_sha256()
        .context("read sealed resident V3 import identity")?;
    let resident_columns = resident_import.columns();
    if admission_identity == [0; 32]
        || resident_import.rows() != scope_rows
        || resident_columns == 0
    {
        bail!(
            "resident V3 import does not match the exact Search scope, shape or admitted ordinal"
        );
    }
    let resident_feature_store_session_v3: ResidentPopulationSessionV3 = resident_import
        .consume_into_population_session_v3()
        .context("consume resident V3 import into its admitted population session")?;
    if resident_feature_store_session_v3.rows() != scope_rows
        || resident_feature_store_session_v3.columns() != resident_columns
        || resident_feature_store_session_v3.admission_identity_sha256() != admission_identity
        || resident_feature_store_session_v3
            .device_identity()
            .ordinal()
            != selected_ordinal
        || resident_feature_store_session_v3.data_transient_retirement_process_token() == [0; 32]
    {
        bail!("resident V3 population session drifted from its validated import");
    }
    Ok(StrictResidentPopulationExecutionRunV3 {
        scope: scope.clone(),
        session: Some(resident_feature_store_session_v3),
        row_count: scope_rows,
        column_count: resident_columns,
    })
}

pub(crate) fn record_resident_feature_store_consumer_completion_v3(
    mut run: StrictResidentPopulationExecutionRunV3,
) -> Result<ResidentFeatureStoreConsumerLeaseV3> {
    run.scope
        .validate()
        .context("validate resident V3 Search scope at consumer completion")?;
    let session = run
        .session
        .take()
        .context("resident V3 Search run has no session to complete")?;
    if session.rows() != run.row_count || session.columns() != run.column_count {
        bail!("resident V3 Search session shape drifted before consumer completion");
    }
    session
        .record_consumer_completion()
        .context("record resident V3 consumer completion event")
}

/// Consume one strict native run through a single purpose-bound Search
/// callback and record the completion event on every normal/error return from
/// that callback. The nested result preserves the Search error while returning
/// the lease which must outlive all queued resident reads.
pub fn consume_strict_resident_population_execution_run_v3<Output, Consumer>(
    mut run: StrictResidentPopulationExecutionRunV3,
    consumer: Consumer,
) -> Result<(Result<Output>, ResidentFeatureStoreConsumerLeaseV3)>
where
    Consumer: FnOnce(&mut StrictResidentPopulationExecutionRunV3) -> Result<Output>,
{
    let expected_shape = (run.row_count(), run.column_count());
    let outcome = consumer(&mut run);
    let consumer_completion_lease = record_resident_feature_store_consumer_completion_v3(run)
        .context("record strict resident population consumer completion")?;
    if consumer_completion_lease.rows() != expected_shape.0
        || consumer_completion_lease.columns() != expected_shape.1
    {
        bail!("resident Search completion lease shape drifted from its consumed native run");
    }
    Ok((outcome, consumer_completion_lease))
}
