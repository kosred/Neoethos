use super::{
    ExactPopulationEvaluationV1, ExactPopulationTimestampModeV1, UnsplittablePopulationAllocationV1,
};
use crate::eval::BacktestSettings;
use crate::exact_resident_dataset_authority_v1::{
    ExactResidentDatasetAuthorityV1, ExactResidentDatasetViewV1, SealedExactResidentDatasetParentV1,
};
use crate::native_population_residency_receipt_v1::{
    NativePopulationResidencyReceiptV1, seal_native_population_residency_receipt_v1,
};
use anyhow::{Context, Result, bail};
use neoethos_gpu_cuda::{
    CudaPopulationDeviceIdentityV1, PopulationEvaluationViewV1, PopulationParentDatasetV1,
    PopulationResidencyCountersV1, PopulationSession, PopulationTimestampModeV1,
};
use std::sync::{Arc, Mutex};

const VESTIGIAL_SESSION_MAX_EVENTS_V1: usize = 1;

pub(super) fn exact_native_device_for_evidence_v1(
    evidence: &ExactPopulationEvaluationV1<'_>,
) -> Result<i32> {
    let exact_ordinal = evidence.require_exact_cuda_device_ordinal_v1()?;
    i32::try_from(exact_ordinal.selected_ordinal())
        .context("sealed CUDA ordinal does not fit the native session ABI")
}

struct NativePopulationSessionV1 {
    device: i32,
    device_identity: CudaPopulationDeviceIdentityV1,
    current_view_identity_sha256: Option<String>,
    session: PopulationSession,
}

/// The raw session is moved only while protected by the run-owned mutex. Every
/// native entry point binds `session->device` before touching CUDA state, and no
/// operation can overlap another operation on this same session.
struct SendNativePopulationSessionV1(NativePopulationSessionV1);
unsafe impl Send for SendNativePopulationSessionV1 {}

#[derive(Debug, Default)]
struct NativePopulationResidencyEvidenceStateV1 {
    successful_native_population_count: u64,
    latest_counters: Option<PopulationResidencyCountersV1>,
    closed: bool,
}

#[derive(Clone)]
pub(super) struct NativePopulationResidencyRunV1 {
    parent_dataset_identity_sha256: Arc<str>,
    parent: PopulationParentDatasetV1,
    session: Arc<Mutex<Option<SendNativePopulationSessionV1>>>,
    evidence: Arc<Mutex<NativePopulationResidencyEvidenceStateV1>>,
}

pub(super) fn begin_native_population_residency_v1(
    sealed_parent: &SealedExactResidentDatasetParentV1,
    parent: PopulationParentDatasetV1,
) -> NativePopulationResidencyRunV1 {
    NativePopulationResidencyRunV1 {
        parent_dataset_identity_sha256: Arc::from(sealed_parent.parent_dataset_identity_sha256()),
        parent,
        session: Arc::new(Mutex::new(None)),
        evidence: Arc::new(Mutex::new(
            NativePopulationResidencyEvidenceStateV1::default(),
        )),
    }
}

fn exact_native_view(
    authority: &ExactResidentDatasetAuthorityV1,
    timestamp_mode: ExactPopulationTimestampModeV1,
    settings: &BacktestSettings,
) -> Result<PopulationEvaluationViewV1> {
    let timestamp_mode = match timestamp_mode {
        ExactPopulationTimestampModeV1::Canonical => PopulationTimestampModeV1::Canonical,
        ExactPopulationTimestampModeV1::DisabledIndexDelta => {
            PopulationTimestampModeV1::DisabledIndexDelta
        }
    };
    let adaptive = settings
        .adaptive_base_pips
        .as_ref()
        .map(|values| Arc::<[f64]>::from(values.clone()));
    let parent_rows = authority.parent_row_count();
    match authority.view() {
        ExactResidentDatasetViewV1::Full { .. } => {
            PopulationEvaluationViewV1::full(parent_rows, timestamp_mode, adaptive)
                .map_err(anyhow::Error::new)
        }
        ExactResidentDatasetViewV1::ContiguousRange { start, end } => {
            PopulationEvaluationViewV1::contiguous_range(
                parent_rows,
                *start,
                *end,
                timestamp_mode,
                adaptive,
            )
            .map_err(anyhow::Error::new)
        }
        ExactResidentDatasetViewV1::OrderedIndices { indices } => {
            let indices = indices
                .iter()
                .copied()
                .map(u64::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            PopulationEvaluationViewV1::ordered_indices(
                parent_rows,
                Arc::from(indices),
                timestamp_mode,
                adaptive,
            )
            .map_err(anyhow::Error::new)
        }
    }
}

impl NativePopulationResidencyRunV1 {
    pub(super) fn parent_dataset_identity_sha256(&self) -> &str {
        self.parent_dataset_identity_sha256.as_ref()
    }

    pub(super) fn bind_exact_native_population_view_v1<T>(
        &self,
        authority: &ExactResidentDatasetAuthorityV1,
        timestamp_mode: ExactPopulationTimestampModeV1,
        settings: &BacktestSettings,
        resident_execution_identity_sha256: &str,
        device: i32,
        execute: impl FnOnce(&mut PopulationSession) -> Result<T>,
    ) -> Result<(T, PopulationResidencyCountersV1)> {
        if authority.parent_dataset_identity_sha256()
            != self.parent_dataset_identity_sha256.as_ref()
        {
            bail!("native population authority names a different sealed parent");
        }
        let view = exact_native_view(authority, timestamp_mode, settings)?;
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if session.is_none() {
            let mut native = PopulationSession::create(device, VESTIGIAL_SESSION_MAX_EVENTS_V1)
                .map_err(anyhow::Error::new)
                .context("create run-scoped native population session")?;
            let device_identity = native
                .read_device_identity_v1()
                .map_err(anyhow::Error::new)
                .context("read exact native population device identity")?;
            let requested_ordinal = u32::try_from(device)
                .context("native population device ordinal does not fit u32")?;
            if device_identity.selected_device_ordinal() != requested_ordinal {
                bail!(
                    "native population session selected CUDA ordinal {}; requested {requested_ordinal}",
                    device_identity.selected_device_ordinal()
                );
            }
            native
                .upload_parent_dataset_v1(self.parent.clone())
                .map_err(anyhow::Error::new)
                .context(UnsplittablePopulationAllocationV1(
                    "the immutable native population parent upload",
                ))
                .context("upload immutable native population parent")?;
            *session = Some(SendNativePopulationSessionV1(NativePopulationSessionV1 {
                device,
                device_identity,
                current_view_identity_sha256: None,
                session: native,
            }));
        }
        let native = &mut session
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("native population session installation failed"))?
            .0;
        if native.device != device {
            bail!(
                "one exact population run cannot span CUDA devices {} and {device}",
                native.device
            );
        }
        if native.current_view_identity_sha256.as_deref()
            != Some(resident_execution_identity_sha256)
        {
            native
                .session
                .bind_evaluation_view_v1(view)
                .map_err(anyhow::Error::new)
                .context("bind exact native population view")?;
            native.current_view_identity_sha256 =
                Some(resident_execution_identity_sha256.to_owned());
        }
        let output = execute(&mut native.session)?;
        let counters = native
            .session
            .read_residency_counters_v1()
            .map_err(anyhow::Error::new)
            .context("read native population residency counters")?;
        Ok((output, counters))
    }

    pub(super) fn record_successful_native_population_v1(
        &self,
        expected_output_rows: usize,
        actual_output_rows: usize,
        counters: PopulationResidencyCountersV1,
    ) -> Result<()> {
        if expected_output_rows == 0 || expected_output_rows != actual_output_rows {
            bail!(
                "native residency output cardinality {actual_output_rows} does not match non-empty expected {expected_output_rows}"
            );
        }
        let mut evidence = self
            .evidence
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if evidence.closed {
            bail!("native population residency run is already closed");
        }
        let previous_metric_readback_count = evidence
            .latest_counters
            .map(PopulationResidencyCountersV1::metric_rows_readback_count)
            .unwrap_or(0);
        let metric_readback_delta = counters
            .metric_rows_readback_count()
            .checked_sub(previous_metric_readback_count)
            .filter(|delta| *delta > 0)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "successful native population has no new metric-row readback boundary"
                )
            })?;
        evidence.successful_native_population_count = evidence
            .successful_native_population_count
            .checked_add(metric_readback_delta)
            .ok_or_else(|| anyhow::anyhow!("successful native population count overflow"))?;
        evidence.latest_counters = Some(counters);
        Ok(())
    }

    pub(super) fn finish(&self) -> Result<Option<NativePopulationResidencyReceiptV1>> {
        let mut evidence = self
            .evidence
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if evidence.closed {
            bail!("native population residency run is already closed");
        }
        evidence.closed = true;
        if evidence.successful_native_population_count == 0 {
            return Ok(None);
        }
        let counters = evidence
            .latest_counters
            .ok_or_else(|| anyhow::anyhow!("successful native population has no counters"))?;
        let session = self
            .session
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let native = &session
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("successful native population has no session"))?
            .0;
        let cuda_build_manifest_v1 =
            neoethos_gpu_cuda::cuda_build_manifest_v1().ok_or_else(|| {
                anyhow::anyhow!("successful native population has no CUDA build manifest")
            })?;
        seal_native_population_residency_receipt_v1(
            &self.parent_dataset_identity_sha256,
            self.parent_dataset_identity_sha256(),
            evidence.successful_native_population_count,
            counters,
            native.device_identity,
            neoethos_gpu_cuda::native_abi_version(),
            cuda_build_manifest_v1,
        )
        .map(Some)
        .map_err(anyhow::Error::new)
    }
}
