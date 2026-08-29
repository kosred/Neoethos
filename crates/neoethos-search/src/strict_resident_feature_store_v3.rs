//! One-shot Search binding for a sealed Data V3 resident feature store.
//!
//! This module accepts only the opaque sealed store and the canonical Search
//! scope. Device handles and route selection remain owned by the admitted
//! store and the gpu-cuda session wrapper.

use crate::data_selection::CanonicalSearchArtifactScopeV2;
use anyhow::{Context, Result, bail};
use neoethos_data::SealedGpuResidentFeatureStoreV3;
use neoethos_gpu_cuda::resident_feature_store_v3::{
    ResidentFeatureStoreConsumerLeaseV3, ResidentFeatureStoreImportV3, ResidentPopulationSessionV3,
};

const SEALED_STORE_AUTHORITY_V3: &str = "neoethos.data.sealed-gpu-resident-feature-store.v3";

/// One move-only native Search run whose parent features remain owned by the
/// sealed Data V3 store. This type is intentionally separate from the host
/// population run: constructing it never materializes host feature or base-bar
/// arrays and never creates a V1 population parent.
pub(crate) struct StrictResidentPopulationExecutionRunV3 {
    scope: CanonicalSearchArtifactScopeV2,
    session: Option<ResidentPopulationSessionV3>,
    row_count: usize,
    column_count: usize,
}

impl StrictResidentPopulationExecutionRunV3 {
    pub(crate) const fn scope(&self) -> &CanonicalSearchArtifactScopeV2 {
        &self.scope
    }

    pub(crate) const fn row_count(&self) -> usize {
        self.row_count
    }

    pub(crate) const fn column_count(&self) -> usize {
        self.column_count
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
    scope: &CanonicalSearchArtifactScopeV2,
) -> Result<()> {
    scope
        .validate()
        .context("validate canonical Search V2 scope")?;
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
        || hex_lower(content) != scope.receipt().feature_content_sha256()
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
    scope: &CanonicalSearchArtifactScopeV2,
) -> Result<StrictResidentPopulationExecutionRunV3> {
    validate_strict_resident_feature_store_v3(&sealed_store, scope)?;
    let resident_import = sealed_store
        .into_resident_feature_store_import_v3()
        .context("consume validated sealed Data V3 store into its admitted-stream import")?;
    bind_resident_feature_store_v3(resident_import, scope)
}

pub(crate) fn bind_resident_feature_store_v3(
    resident_import: ResidentFeatureStoreImportV3,
    scope: &CanonicalSearchArtifactScopeV2,
) -> Result<StrictResidentPopulationExecutionRunV3> {
    scope
        .validate()
        .context("validate canonical Search V2 scope")?;
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
