use crate::data_selection::CanonicalSearchArtifactScopeV2;
use crate::engine_identity::PopulationEvalEngine;
use crate::eval::BacktestSettings;
use crate::exact_resident_dataset_authority_v1::{
    ExactResidentDatasetAuthorityDeriveRequestV1, ExactResidentDatasetAuthorityV1,
    ExactResidentDatasetParentSealRequestV1, ExactResidentDatasetViewRequestV1,
    ExactResidentDatasetViewV1, SealedExactResidentDatasetParentV1,
    derive_exact_resident_dataset_authority_v1, seal_exact_resident_dataset_parent_v1,
};
use crate::population_engine_run_receipt_v1::{
    PopulationEngineRunScopeV1, begin_population_engine_run_v1,
};
use crate::population_execution_run_receipt_v2::{
    ExactPopulationExecutionRunReceiptV2, seal_exact_population_execution_run_receipt_v2,
};
use crate::strict_discovery_device_route_v1::{
    ExactCudaDeviceOrdinalV1, SealedCpuDiscoveryRouteReceiptV2,
    SealedStrictDiscoveryDeviceAdmissionV1, SealedStrictDiscoveryDeviceRouteV1,
};
use neoethos_data::{FeatureFrame, Ohlcv};
#[cfg(feature = "gpu-b-adapter")]
use neoethos_gpu_cuda::{
    PopulationParentDatasetInputV1, PopulationParentDatasetV1, PopulationResidencyCountersV1,
    PopulationSession,
};
use sha2::{Digest, Sha256};
use std::fmt;
use std::marker::PhantomData;
#[cfg(feature = "gpu-b-adapter")]
use std::sync::Arc;

#[cfg(feature = "gpu-b-adapter")]
mod native_cuda_resident_v1;
#[cfg(feature = "gpu-b-adapter")]
use native_cuda_resident_v1::{
    NativePopulationResidencyRunV1, begin_native_population_residency_v1,
    exact_native_device_for_evidence_v1,
};

const RESIDENT_EXECUTION_HASH_DOMAIN_V1: &[u8] = b"neoethos.search.resident-execution.v1\0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExactPopulationExecutionErrorCodeV1 {
    InvalidParent,
    Authority,
    ViewLayoutMismatch,
    EngineReceipt,
    DeviceRoute,
    #[cfg(feature = "gpu-b-adapter")]
    NativeResidency,
    RunReceipt,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExactPopulationExecutionErrorV1 {
    code: ExactPopulationExecutionErrorCodeV1,
    message: String,
}

impl ExactPopulationExecutionErrorV1 {
    #[cfg(test)]
    pub(crate) const fn code(&self) -> ExactPopulationExecutionErrorCodeV1 {
        self.code
    }
}

impl fmt::Display for ExactPopulationExecutionErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ExactPopulationExecutionErrorV1 {}

/// A device allocation whose byte extent is invariant under scenario-list
/// splitting. Parent and gene-store uploads use this marker so an allocation
/// failure cannot recurse through thousands of leaves while retrying the same
/// immutable allocation.
#[cfg(feature = "gpu-b-adapter")]
#[derive(Debug)]
pub(crate) struct UnsplittablePopulationAllocationV1(pub(crate) &'static str);

#[cfg(feature = "gpu-b-adapter")]
impl fmt::Display for UnsplittablePopulationAllocationV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} does not depend on the work list size — splitting it cannot help",
            self.0
        )
    }
}

#[cfg(feature = "gpu-b-adapter")]
impl std::error::Error for UnsplittablePopulationAllocationV1 {}

fn error(
    code: ExactPopulationExecutionErrorCodeV1,
    message: impl Into<String>,
) -> ExactPopulationExecutionErrorV1 {
    ExactPopulationExecutionErrorV1 {
        code,
        message: message.into(),
    }
}

/// Timestamp arithmetic is part of the resident computation, not a detachable
/// caller convention. Ordered CPCV views that intentionally use index deltas
/// must name that distinct mode; they cannot masquerade as canonical-time runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExactPopulationTimestampModeV1 {
    Canonical,
    DisabledIndexDelta,
}

/// One explicitly owned discovery-run parent. The exact source arrays are
/// validated and hashed once, then converted once into the buffers consumed by
/// population evaluation. Later view seals receive only the opaque parent seal.
pub(crate) struct ExactPopulationExecutionRunV1<'a> {
    parent: SealedExactResidentDatasetParentV1,
    strict_device_route: SealedStrictDiscoveryDeviceRouteV1,
    #[cfg(feature = "gpu-b-adapter")]
    native_residency: NativePopulationResidencyRunV1,
    engine_run: PopulationEngineRunScopeV1,
    source_lifetime: PhantomData<(&'a FeatureFrame, &'a Ohlcv)>,
}

/// Immutable sizing primitives borrowed from the already-created exact run.
/// It deliberately excludes month capacity and the Stage-1 view: those become
/// known only after the caller resolves the actual evaluation configuration and
/// range. Reading this value performs no device operation.
pub(crate) struct ExactPopulationAutoSizingPrimitivesV1 {
    pub(crate) parent_canonical_scope_identity_sha256: String,
    pub(crate) parent_dataset_identity_sha256: String,
    pub(crate) resident_parent_rows: usize,
    pub(crate) feature_count: usize,
    pub(crate) route: crate::PopulationAutoSizingRouteV1,
}

/// One sealed evaluation view plus the exact buffers/settings it is allowed to
/// execute. Prototype B receives this object rather than separately supplied
/// same-shaped arrays, so the resident cache key and uploaded bytes cannot be
/// detached from one another.
pub(crate) struct ExactPopulationEvaluationV1<'a> {
    authority: ExactResidentDatasetAuthorityV1,
    resident_identity_sha256: String,
    strict_device_route: SealedStrictDiscoveryDeviceRouteV1,
    #[cfg(feature = "gpu-b-adapter")]
    timestamp_mode: ExactPopulationTimestampModeV1,
    settings: BacktestSettings,
    engine_run: PopulationEngineRunScopeV1,
    #[cfg(feature = "gpu-b-adapter")]
    native_residency: NativePopulationResidencyRunV1,
    source_lifetime: PhantomData<&'a ()>,
}

fn hex_lower(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn resident_execution_identity(
    authority: &ExactResidentDatasetAuthorityV1,
    timestamp_mode: ExactPopulationTimestampModeV1,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(RESIDENT_EXECUTION_HASH_DOMAIN_V1);
    hasher.update((authority.identity_sha256().len() as u64).to_le_bytes());
    hasher.update(authority.identity_sha256().as_bytes());
    hasher.update([match timestamp_mode {
        ExactPopulationTimestampModeV1::Canonical => 0,
        ExactPopulationTimestampModeV1::DisabledIndexDelta => 1,
    }]);
    hex_lower(&hasher.finalize())
}

pub(crate) fn begin_exact_population_execution_run_v1<'a>(
    admission: SealedStrictDiscoveryDeviceAdmissionV1,
    scope: &'a CanonicalSearchArtifactScopeV2,
    features: &'a FeatureFrame,
    ohlcv: &'a Ohlcv,
) -> Result<ExactPopulationExecutionRunV1<'a>, ExactPopulationExecutionErrorV1> {
    scope.validate().map_err(|source| {
        error(
            ExactPopulationExecutionErrorCodeV1::InvalidParent,
            format!("invalid canonical population scope: {source}"),
        )
    })?;
    let strict_device_route = admission.into_route_v1();
    let rows = features.n_samples();
    if rows == 0
        || features.n_features() == 0
        || ohlcv.open.len() != rows
        || ohlcv.high.len() != rows
        || ohlcv.low.len() != rows
        || ohlcv.close.len() != rows
        || ohlcv
            .volume
            .as_ref()
            .is_some_and(|volume| volume.len() != rows)
        || ohlcv.timestamp.as_deref() != Some(features.timestamps.as_slice())
    {
        return Err(error(
            ExactPopulationExecutionErrorCodeV1::InvalidParent,
            "exact population parent OHLCV, feature rows, or timestamps disagree",
        ));
    }
    let window = scope.evaluated_window();
    let scope_rows = window
        .row_end()
        .checked_sub(window.row_start())
        .and_then(|value| usize::try_from(value).ok());
    if scope_rows != Some(rows)
        || window.timestamp_start_ms() != features.timestamps[0]
        || window.timestamp_end_ms() != features.timestamps[rows - 1]
    {
        return Err(error(
            ExactPopulationExecutionErrorCodeV1::InvalidParent,
            "canonical population scope does not name the exact parent row/timestamp window",
        ));
    }

    let (ob, fvg, liq, trend, premium, inducement, bos, choch, eqh, eql, displacement) =
        crate::genetic::build_smc_arrays(features, ohlcv).map_err(|source| {
            error(
                ExactPopulationExecutionErrorCodeV1::InvalidParent,
                format!("derive exact population SMC parent: {source}"),
            )
        })?;
    let smc_data = (0..rows)
        .map(|row| {
            [
                ob[row],
                fvg[row],
                liq[row],
                trend[row],
                premium[row],
                inducement[row],
                bos[row],
                choch[row],
                eqh[row],
                eql[row],
                displacement[row],
            ]
        })
        .collect::<Vec<_>>();

    let parent = seal_exact_resident_dataset_parent_v1(ExactResidentDatasetParentSealRequestV1 {
        scope,
        features,
        ohlcv,
        smc_data: &smc_data,
    })
    .map_err(|source| {
        error(
            ExactPopulationExecutionErrorCodeV1::Authority,
            format!("seal exact population parent once: {source}"),
        )
    })?;
    #[cfg(feature = "gpu-b-adapter")]
    let native_residency = {
        // Build the one native feature-major parent one column at a time. This
        // keeps temporary materialization bounded to one column instead of
        // allocating a full samples-major duplicate and then a second full
        // feature-major matrix solely to change layout.
        let feature_values = rows.checked_mul(features.n_features()).ok_or_else(|| {
            error(
                ExactPopulationExecutionErrorCodeV1::InvalidParent,
                "exact native population feature extent overflows usize",
            )
        })?;
        let mut indicators_feature_major = Vec::with_capacity(feature_values);
        for feature in 0..features.n_features() {
            let column = features.feature_column(feature).map_err(|source| {
                error(
                    ExactPopulationExecutionErrorCodeV1::InvalidParent,
                    format!(
                        "materialize exact native population feature column {feature}: {source}"
                    ),
                )
            })?;
            if column.values.len() != rows {
                return Err(error(
                    ExactPopulationExecutionErrorCodeV1::InvalidParent,
                    format!(
                        "exact native population feature column {feature} has {} rows; expected {rows}",
                        column.values.len()
                    ),
                ));
            }
            indicators_feature_major.extend_from_slice(&column.values);
        }
        let (months, days) = crate::genetic::month_day_indices(&features.timestamps);
        let smc_rows = smc_data
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        let native_parent = PopulationParentDatasetV1::new(PopulationParentDatasetInputV1 {
            close: Arc::from(ohlcv.close.clone()),
            high: Arc::from(ohlcv.high.clone()),
            low: Arc::from(ohlcv.low.clone()),
            indicators_feature_major: Arc::from(indicators_feature_major),
            feature_count: features.n_features(),
            months: Arc::from(months),
            days: Arc::from(days),
            timestamps: Arc::from(features.timestamps.clone()),
            smc_rows: Arc::from(smc_rows),
        })
        .map_err(|source| {
            error(
                ExactPopulationExecutionErrorCodeV1::InvalidParent,
                format!("construct exact native population parent: {source}"),
            )
        })?;
        begin_native_population_residency_v1(&parent, native_parent)
    };
    let engine_run = begin_population_engine_run_v1(scope).map_err(|source| {
        error(
            ExactPopulationExecutionErrorCodeV1::EngineReceipt,
            format!("begin exact population engine run: {source}"),
        )
    })?;

    Ok(ExactPopulationExecutionRunV1 {
        parent,
        strict_device_route,
        #[cfg(feature = "gpu-b-adapter")]
        native_residency,
        engine_run,
        source_lifetime: PhantomData,
    })
}

impl ExactPopulationExecutionRunV1<'_> {
    pub(crate) fn population_auto_sizing_primitives_v1(
        &self,
    ) -> Result<ExactPopulationAutoSizingPrimitivesV1, ExactPopulationExecutionErrorV1> {
        let route = self
            .strict_device_route
            .population_auto_sizing_route_v1()
            .map_err(|source| {
                error(
                    ExactPopulationExecutionErrorCodeV1::DeviceRoute,
                    format!("read run-owned population-auto route facts: {source}"),
                )
            })?;
        Ok(ExactPopulationAutoSizingPrimitivesV1 {
            parent_canonical_scope_identity_sha256: self
                .parent
                .canonical_scope_identity_sha256()
                .to_owned(),
            parent_dataset_identity_sha256: self.parent.parent_dataset_identity_sha256().to_owned(),
            resident_parent_rows: self.parent.parent_row_count(),
            feature_count: self.parent.feature_count(),
            route,
        })
    }

    pub(crate) fn seal_evaluation(
        &self,
        settings: &BacktestSettings,
        view: ExactResidentDatasetViewRequestV1<'_>,
    ) -> Result<ExactPopulationEvaluationV1<'_>, ExactPopulationExecutionErrorV1> {
        self.seal_evaluation_with_timestamp_mode(
            settings,
            view,
            ExactPopulationTimestampModeV1::Canonical,
        )
    }

    pub(crate) fn seal_evaluation_with_timestamp_mode(
        &self,
        settings: &BacktestSettings,
        view: ExactResidentDatasetViewRequestV1<'_>,
        timestamp_mode: ExactPopulationTimestampModeV1,
    ) -> Result<ExactPopulationEvaluationV1<'_>, ExactPopulationExecutionErrorV1> {
        let authority = derive_exact_resident_dataset_authority_v1(
            ExactResidentDatasetAuthorityDeriveRequestV1 {
                parent: &self.parent,
                settings,
                view,
            },
        )
        .map_err(|source| {
            error(
                ExactPopulationExecutionErrorCodeV1::Authority,
                format!("derive exact population evaluation: {source}"),
            )
        })?;
        let resident_identity_sha256 = resident_execution_identity(&authority, timestamp_mode);

        match authority.view() {
            ExactResidentDatasetViewV1::Full { .. }
            | ExactResidentDatasetViewV1::ContiguousRange { .. } => {}
            ExactResidentDatasetViewV1::OrderedIndices { indices } => {
                if indices.is_empty() {
                    return Err(error(
                        ExactPopulationExecutionErrorCodeV1::ViewLayoutMismatch,
                        "sealed ordered population view is empty",
                    ));
                }
            }
        }
        let timestamps = match timestamp_mode {
            ExactPopulationTimestampModeV1::Canonical => authority.view().row_count(),
            ExactPopulationTimestampModeV1::DisabledIndexDelta => 0,
        };

        let evaluation = ExactPopulationEvaluationV1 {
            authority,
            resident_identity_sha256,
            strict_device_route: self.strict_device_route.clone(),
            #[cfg(feature = "gpu-b-adapter")]
            timestamp_mode,
            settings: settings.clone(),
            engine_run: self.engine_run.clone(),
            #[cfg(feature = "gpu-b-adapter")]
            native_residency: self.native_residency.clone(),
            source_lifetime: PhantomData,
        };
        let expected_timestamp_rows = match timestamp_mode {
            ExactPopulationTimestampModeV1::Canonical => evaluation.authority.view().row_count(),
            ExactPopulationTimestampModeV1::DisabledIndexDelta => 0,
        };
        if timestamps != expected_timestamp_rows {
            return Err(error(
                ExactPopulationExecutionErrorCodeV1::ViewLayoutMismatch,
                "population timestamp mode disagrees with the sealed view",
            ));
        }
        evaluation.validate_population_layout(
            evaluation.authority.view().row_count(),
            evaluation.authority.feature_count(),
        )?;
        Ok(evaluation)
    }

    pub(crate) fn finish(
        &self,
    ) -> Result<ExactPopulationExecutionRunReceiptV2, ExactPopulationExecutionErrorV1> {
        let engine_receipt_v1 = self.engine_run.finish().map_err(|source| {
            error(
                ExactPopulationExecutionErrorCodeV1::EngineReceipt,
                format!("finish exact population engine run: {source}"),
            )
        })?;
        #[cfg(feature = "gpu-b-adapter")]
        let native_residency_receipt_v1 = self.native_residency.finish().map_err(|source| {
            error(
                ExactPopulationExecutionErrorCodeV1::NativeResidency,
                format!("finish exact native population residency: {source}"),
            )
        })?;
        #[cfg(not(feature = "gpu-b-adapter"))]
        let native_residency_receipt_v1 = None;
        seal_exact_population_execution_run_receipt_v2(
            engine_receipt_v1,
            native_residency_receipt_v1,
        )
        .map_err(|source| {
            error(
                ExactPopulationExecutionErrorCodeV1::RunReceipt,
                format!("seal exact population execution V2 receipt: {source}"),
            )
        })
    }
}

impl ExactPopulationEvaluationV1<'_> {
    pub(crate) fn require_cpu_route_receipt_v1(
        &self,
    ) -> Result<&SealedCpuDiscoveryRouteReceiptV2, ExactPopulationExecutionErrorV1> {
        self.strict_device_route
            .require_cpu_route_receipt_v1()
            .map_err(|source| {
                error(
                    ExactPopulationExecutionErrorCodeV1::DeviceRoute,
                    format!("require sealed no-compatible-GPU route: {source}"),
                )
            })
    }

    pub(crate) fn require_exact_cuda_device_ordinal_v1(
        &self,
    ) -> Result<&ExactCudaDeviceOrdinalV1, ExactPopulationExecutionErrorV1> {
        self.strict_device_route
            .require_exact_cuda_device_ordinal_v1()
            .map_err(|source| {
                error(
                    ExactPopulationExecutionErrorCodeV1::DeviceRoute,
                    format!("require sealed exact CUDA ordinal: {source}"),
                )
            })
    }

    #[cfg(test)]
    pub(crate) const fn authority(&self) -> &ExactResidentDatasetAuthorityV1 {
        &self.authority
    }

    #[cfg(test)]
    pub(crate) fn resident_identity_sha256(&self) -> &str {
        &self.resident_identity_sha256
    }

    pub(crate) fn validate_population_layout(
        &self,
        row_count: usize,
        feature_count: usize,
    ) -> Result<(), ExactPopulationExecutionErrorV1> {
        let sealed_rows = self.authority.view().row_count();
        let adaptive_matches = self
            .settings
            .adaptive_base_pips
            .as_ref()
            .is_none_or(|values| values.len() == sealed_rows);
        let resident_identity_matches = self.resident_identity_sha256.len() == 64
            && self
                .resident_identity_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
        #[cfg(feature = "gpu-b-adapter")]
        let native_parent_matches = self.authority.parent_dataset_identity_sha256()
            == self.native_residency.parent_dataset_identity_sha256();
        #[cfg(not(feature = "gpu-b-adapter"))]
        let native_parent_matches = true;
        if row_count != sealed_rows
            || feature_count != self.authority.feature_count()
            || !adaptive_matches
            || !resident_identity_matches
            || !native_parent_matches
        {
            return Err(error(
                ExactPopulationExecutionErrorCodeV1::ViewLayoutMismatch,
                format!(
                    "population layout {row_count}x{feature_count} does not match sealed view {}x{}",
                    sealed_rows,
                    self.authority.feature_count()
                ),
            ));
        }
        Ok(())
    }

    #[cfg(feature = "gpu-b-adapter")]
    pub(crate) const fn settings(&self) -> &BacktestSettings {
        &self.settings
    }

    #[cfg(feature = "gpu-b-adapter")]
    pub(crate) fn row_count(&self) -> usize {
        self.authority.view().row_count()
    }

    #[cfg(feature = "gpu-b-adapter")]
    pub(crate) fn parent_row_count(&self) -> usize {
        self.authority.parent_row_count()
    }

    #[cfg(feature = "gpu-b-adapter")]
    pub(crate) fn parent_dataset_identity_sha256(&self) -> &str {
        self.authority.parent_dataset_identity_sha256()
    }

    #[cfg(feature = "gpu-b-adapter")]
    pub(crate) fn feature_count(&self) -> usize {
        self.authority.feature_count()
    }

    #[cfg(feature = "gpu-b-adapter")]
    pub(crate) fn ordered_index_capacity_v1(&self) -> usize {
        self.authority
            .view()
            .ordered_indices()
            .map_or(0, <[usize]>::len)
    }

    #[cfg(feature = "gpu-b-adapter")]
    pub(crate) fn adaptive_row_capacity_v1(&self) -> usize {
        self.settings
            .adaptive_base_pips
            .as_deref()
            .map_or(0, <[f64]>::len)
    }

    #[cfg(feature = "gpu-b-adapter")]
    pub(crate) fn bind_exact_native_population_view_v1<T>(
        &self,
        device: i32,
        execute: impl FnOnce(&mut PopulationSession) -> anyhow::Result<T>,
    ) -> anyhow::Result<(T, PopulationResidencyCountersV1)> {
        let sealed_device = exact_native_device_for_evidence_v1(self)?;
        if device != sealed_device {
            anyhow::bail!(
                "native population caller requested CUDA ordinal {device}, but the run-bound probe sealed ordinal {sealed_device}"
            );
        }
        self.native_residency.bind_exact_native_population_view_v1(
            &self.authority,
            self.timestamp_mode,
            &self.settings,
            &self.resident_identity_sha256,
            device,
            execute,
        )
    }

    #[cfg(feature = "gpu-b-adapter")]
    pub(crate) fn record_successful_native_population_v1(
        &self,
        expected_output_rows: usize,
        actual_output_rows: usize,
        counters: PopulationResidencyCountersV1,
    ) -> anyhow::Result<()> {
        self.native_residency
            .record_successful_native_population_v1(
                expected_output_rows,
                actual_output_rows,
                counters,
            )
    }

    pub(crate) fn record_successful_population(
        &self,
        engine: PopulationEvalEngine,
        expected_output_rows: usize,
        actual_output_rows: usize,
    ) -> Result<(), ExactPopulationExecutionErrorV1> {
        self.engine_run
            .record_successful_population(engine, expected_output_rows, actual_output_rows)
            .map_err(|source| {
                error(
                    ExactPopulationExecutionErrorCodeV1::EngineReceipt,
                    format!("record exact population output: {source}"),
                )
            })
    }
}

/// Strict V3 execution over the actual Data owner retained by repeated Search.
/// This is metrics evidence only: it cannot be converted to the legacy
/// WindowEvaluation, whose absent ledger would trigger a CPU replay.
#[cfg(all(feature = "gpu-cuda", any(test, target_os = "linux")))]
pub(crate) mod retained_compact_v3 {
    use super::*;
    use crate::exact_resident_dataset_authority_v1::SealedExactResidentCompactParentV3;
    use crate::genetic::search_engine::{
        ExactSearchSizingPolicyV1, ResidentGenerationZeroRuntimeSnapshotV1,
        evaluation_backtest_settings, pack_resident_generation_genes_v1,
        resident_generation_population_settings_v1,
    };
    use crate::genetic::{EvaluationConfig, Gene};
    use crate::resident_population_auto_sizing_receipt_v2::ResidentPopulationAutoSizingReceiptV2;
    use anyhow::{Context, Result, ensure};
    use neoethos_gpu_cuda::resident_feature_store_v3::{
        ResidentFeatureStoreCudaErrorV3, ResidentPopulationSessionV3,
    };
    use neoethos_gpu_cuda::{
        HostPopulationMetricsReceiptV1, PopulationEvaluationViewV1, PopulationTimestampModeV1,
        ResidentAdaptiveBaseRequestV1,
    };
    use std::ops::Range;

    pub struct ExactResidentCompactMetricsV3 {
        authority: ExactResidentDatasetAuthorityV1,
        execution_identity_sha256: String,
        metrics: Vec<[f64; 11]>,
        batch_receipts: Vec<HostPopulationMetricsReceiptV1>,
        counters_before: neoethos_gpu_cuda::PopulationResidencyCountersV1,
        counters_after: neoethos_gpu_cuda::PopulationResidencyCountersV1,
        adaptive_token_identity_sha256: Option<[u8; 32]>,
    }

    impl ExactResidentCompactMetricsV3 {
        pub fn authority(&self) -> &ExactResidentDatasetAuthorityV1 {
            &self.authority
        }
        pub fn execution_identity_sha256(&self) -> &str {
            &self.execution_identity_sha256
        }
        pub fn metrics(&self) -> &[[f64; 11]] {
            &self.metrics
        }
        pub fn batch_receipts(&self) -> &[HostPopulationMetricsReceiptV1] {
            &self.batch_receipts
        }
        pub fn counters_before(&self) -> neoethos_gpu_cuda::PopulationResidencyCountersV1 {
            self.counters_before
        }
        pub fn counters_after(&self) -> neoethos_gpu_cuda::PopulationResidencyCountersV1 {
            self.counters_after
        }
        pub fn adaptive_token_identity_sha256(&self) -> Option<[u8; 32]> {
            self.adaptive_token_identity_sha256
        }
    }

    fn selected_view(
        parent: &SealedExactResidentDatasetParentV1,
        settings: &BacktestSettings,
        request: ExactResidentDatasetViewRequestV1<'_>,
        selection: &Range<usize>,
    ) -> Result<ExactResidentDatasetAuthorityV1> {
        ensure!(
            selection.start < selection.end && selection.end <= parent.parent_row_count(),
            "invalid immutable resident selection range"
        );
        let authority = derive_exact_resident_dataset_authority_v1(
            ExactResidentDatasetAuthorityDeriveRequestV1 {
                parent,
                settings,
                view: request,
            },
        )?;
        validate_selected_view(authority.view(), selection)?;
        Ok(authority)
    }

    fn validate_selected_view(
        view: &ExactResidentDatasetViewV1,
        selection: &Range<usize>,
    ) -> Result<()> {
        let in_selection = match view {
            ExactResidentDatasetViewV1::Full { row_count } => {
                selection.start == 0 && *row_count <= selection.end
            }
            ExactResidentDatasetViewV1::ContiguousRange { start, end } => {
                *start >= selection.start && *end <= selection.end
            }
            ExactResidentDatasetViewV1::OrderedIndices { indices } => {
                indices.iter().all(|index| selection.contains(index))
            }
        };
        ensure!(
            in_selection,
            "resident validation view reaches outside the sealed selection range; holdout is not authorized"
        );
        Ok(())
    }

    fn gene_chunks(
        genes: &[Gene],
        candidate_cap: usize,
        term_cap: usize,
    ) -> Result<Vec<Range<usize>>> {
        ensure!(
            candidate_cap > 0 && term_cap > 0,
            "resident validation has zero admitted gene capacity"
        );
        let mut chunks = Vec::new();
        chunks.try_reserve(genes.len().div_ceil(candidate_cap))?;
        let mut start = 0;
        while start < genes.len() {
            let mut end = start;
            let mut terms = 0usize;
            while end < genes.len() && end - start < candidate_cap {
                let count = genes[end].indices.len();
                ensure!(
                    count > 0 && count <= term_cap && count == genes[end].weights.len(),
                    "resident validation gene has invalid or individually unadmitted term extent"
                );
                let next = terms
                    .checked_add(count)
                    .context("resident validation term sum overflow")?;
                if next > term_cap {
                    break;
                }
                terms = next;
                end += 1;
            }
            ensure!(
                end > start,
                "resident validation cannot admit one complete gene"
            );
            if chunks.len() == chunks.capacity() {
                chunks.try_reserve(1)?;
            }
            chunks.push(start..end);
            start = end;
        }
        Ok(chunks)
    }

    fn validate_rows(
        scenarios: &[neoethos_gpu_contracts::device::ScenarioDescriptor],
        rows: &[neoethos_gpu_contracts::device::NeoPopulationMetricRow],
    ) -> Result<()> {
        ensure!(
            rows.len() == scenarios.len(),
            "strict resident validation metric cardinality mismatch"
        );
        for (scenario, row) in scenarios.iter().zip(rows) {
            ensure!(
                row.candidate_id == scenario.base_candidate_id
                    && row.scenario_id == scenario.scenario_id,
                "strict resident validation metric identity/order mismatch"
            );
            neoethos_gpu_contracts::resident_search_scoring_v2::classify_resident_metrics_v2(
                &row.values,
            )
            .map_err(|error| anyhow::anyhow!("resident validation arithmetic fault: {error:?}"))?;
            ensure!(
                row.values[8] >= 0.0 && row.values[8].fract() == 0.0,
                "resident validation trade count is negative or fractional"
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_genes_v3(
        parent: &SealedExactResidentCompactParentV3,
        session: &mut ResidentPopulationSessionV3,
        selection: Range<usize>,
        genes: &[Gene],
        config: &EvaluationConfig,
        sizing: &ResidentPopulationAutoSizingReceiptV2,
        runtime: &ResidentGenerationZeroRuntimeSnapshotV1,
        request: ExactResidentDatasetViewRequestV1<'_>,
        timestamp_mode: ExactPopulationTimestampModeV1,
        cancelled: impl Fn() -> bool,
    ) -> Result<ExactResidentCompactMetricsV3> {
        ensure!(
            !cancelled(),
            "__DISCOVERY_CANCELLED__ before resident validation"
        );
        parent.validate_session_v3(session)?;
        runtime.validate_current("before retained compact validation")?;
        runtime.validate_against_receipt_v2(sizing)?;
        ensure!(!genes.is_empty(), "resident validation gene batch is empty");
        let limits = *session
            .data_population_limits()
            .context("missing resident validation workspace")?;
        sizing.validate_against_execution_limits_v2(
            session.device_identity().ordinal(),
            session.pre_materialization_free_bytes_snapshot(),
            session.rows(),
            session.columns(),
            &limits,
        )?;
        ensure!(
            config.pip_value.to_bits() == sizing.adaptive_pip_size().to_bits()
                && config.pip_value_per_lot.to_bits() == sizing.pip_value_per_lot().to_bits(),
            "retained validation financial geometry differs from the sealed sizing authority"
        );
        let mut settings = evaluation_backtest_settings(config)?;
        ensure!(
            settings.adaptive_base_pips.is_none(),
            "resident validation refuses a host adaptive series"
        );
        settings.adaptive_rr = sizing.adaptive_rr();
        let mut authority = selected_view(parent.parent_v3(), &settings, request, &selection)?;
        let mode = match timestamp_mode {
            ExactPopulationTimestampModeV1::Canonical => PopulationTimestampModeV1::Canonical,
            ExactPopulationTimestampModeV1::DisabledIndexDelta => {
                PopulationTimestampModeV1::DisabledIndexDelta
            }
        };
        let view = match authority.view() {
            ExactResidentDatasetViewV1::Full { .. } => {
                PopulationEvaluationViewV1::full(session.rows(), mode, None)?
            }
            ExactResidentDatasetViewV1::ContiguousRange { start, end } => {
                PopulationEvaluationViewV1::contiguous_range(
                    session.rows(),
                    *start,
                    *end,
                    mode,
                    None,
                )?
            }
            ExactResidentDatasetViewV1::OrderedIndices { indices } => {
                ensure!(
                    u64::try_from(indices.len())? <= limits.max_ordered_index_count(),
                    "ordered validation view exceeds the pre-materialization admission"
                );
                let mut ordered = Vec::new();
                ordered.try_reserve_exact(indices.len())?;
                for index in indices {
                    ordered.push(u64::try_from(*index)?);
                }
                PopulationEvaluationViewV1::ordered_indices(
                    session.rows(),
                    ordered.into(),
                    mode,
                    None,
                )?
            }
        };
        let view_rows = view.row_count();
        u32::try_from(view_rows).context("resident validation rows exceed scenario ABI")?;
        let adaptive = sizing.adaptive_stops_requested_for_run()
            && genes.iter().any(|gene| gene.stop_vol_mult > 0.0)
            && view_rows >= ResidentAdaptiveBaseRequestV1::MIN_VIEW_ROWS_V1;
        let adaptive_request = if adaptive {
            ensure!(
                u64::try_from(view_rows)? <= limits.max_adaptive_row_count(),
                "adaptive validation view exceeds the pre-materialization admission"
            );
            // The checked producer explicitly refuses ordered adaptive views.
            Some(ResidentAdaptiveBaseRequestV1::checked_canonical_v1(
                &view,
                sizing.adaptive_pip_size(),
                sizing.adaptive_tail_step(),
                sizing.adaptive_tail_max_bars(),
            )?)
        } else {
            None
        };
        let chunks = gene_chunks(
            genes,
            usize::try_from(limits.max_candidate_count())?,
            usize::try_from(limits.max_gene_term_count())?,
        )?;
        let scenario_cap = usize::try_from(limits.max_concurrent_scenario_count())?;
        ensure!(
            scenario_cap > 0,
            "resident validation has zero admitted scenario capacity"
        );
        let sizing_policy = ExactSearchSizingPolicyV1::from_resident_population_receipt_v2(sizing)?;
        let native_settings = resident_generation_population_settings_v1(&settings)?;
        ensure!(
            u64::from(native_settings.month_capacity) == limits.month_capacity(),
            "resident validation month capacity differs from the admission"
        );
        let mut metrics = Vec::new();
        metrics.try_reserve_exact(genes.len())?;
        let batch_count = chunks.iter().try_fold(0usize, |n, chunk| {
            n.checked_add(chunk.len().div_ceil(scenario_cap))
                .context("validation batch count overflow")
        })?;
        let mut batch_receipts = Vec::new();
        batch_receipts.try_reserve_exact(batch_count)?;
        let counters_before = session.read_residency_counters_v1()?;
        let adaptive_token_identity_sha256 = if let Some(recipe) = adaptive_request {
            let expected = recipe.identity_sha256();
            let mut adaptive_authority = None;
            let facts = session.bind_evaluation_view_with_resident_adaptive_base_checked_v1(view, recipe, |token| {
                if token.request_identity_sha256() != expected
                    || token.resident_session_identity_sha256() == [0; 32]
                    || token.view_identity_sha256() == [0; 32]
                    || token.token_identity_sha256() == [0; 32] {
                    return Err(ResidentFeatureStoreCudaErrorV3::InvalidInput(
                        "retained validation adaptive token differs from the exact checked recipe".into()));
                }
                adaptive_authority = Some(crate::exact_resident_dataset_authority_v1::bind_resident_adaptive_evaluation_v3(
                    authority.clone(), &recipe, token,
                ).map_err(|error| ResidentFeatureStoreCudaErrorV3::InvalidInput(error.to_string()))?);
                Ok(())
            })?;
            authority = adaptive_authority.context("resident adaptive authority was not sealed")?;
            Some(facts.token_identity_sha256())
        } else {
            session.bind_evaluation_view_v1(view)?;
            None
        };
        for chunk in chunks {
            ensure!(
                !cancelled(),
                "__DISCOVERY_CANCELLED__ during resident validation"
            );
            let packed = pack_resident_generation_genes_v1(
                &genes[chunk.clone()],
                session.columns(),
                config,
                &sizing_policy,
                runtime.smc_gate_disabled(),
            )?;
            session.upload_genes(packed.view())?;
            for scenario_start in (0..chunk.len()).step_by(scenario_cap) {
                ensure!(
                    !cancelled(),
                    "__DISCOVERY_CANCELLED__ during resident validation"
                );
                let scenario_end = scenario_start.saturating_add(scenario_cap).min(chunk.len());
                let mut scenarios = Vec::new();
                scenarios.try_reserve_exact(scenario_end - scenario_start)?;
                for local in scenario_start..scenario_end {
                    scenarios.push(crate::gpu_native::scenario::base_scenario(
                        u64::try_from(local)?,
                        u64::try_from(chunk.start + local)?,
                        view_rows,
                    ));
                }
                session.upload_scenarios(&scenarios)?;
                let readback = session
                    .enqueue_metrics_only_v1(&native_settings)?
                    .consume_host_metrics_v1()?;
                validate_rows(&scenarios, readback.metric_rows())?;
                ensure!(
                    readback.terminal_readback_count() == 1
                        && readback.terminal_synchronization_count() == 1
                        && readback.terminal_readback_rows() == u64::try_from(scenarios.len())?
                        && readback.terminal_readback_bytes()
                            == u64::try_from(scenarios.len())?
                                .checked_mul(104)
                                .context("validation readback bytes overflow")?,
                    "strict validation receipt does not describe its exact bounded readback"
                );
                metrics.extend(readback.metric_rows().iter().map(|row| row.values));
                batch_receipts.push(readback);
            }
        }
        ensure!(
            !cancelled(),
            "__DISCOVERY_CANCELLED__ after resident validation"
        );
        ensure!(
            metrics.len() == genes.len() && batch_receipts.len() == batch_count,
            "resident validation returned a partial candidate cohort"
        );
        let counters_after = session.read_residency_counters_v1()?;
        ensure!(
            counters_after.parent_upload_count() == counters_before.parent_upload_count()
                && counters_after.parent_upload_bytes() == counters_before.parent_upload_bytes()
                && counters_after.stream_creation_count()
                    == counters_before.stream_creation_count()
                && counters_after.adaptive_upload_bytes()
                    == counters_before.adaptive_upload_bytes()
                && counters_after.diagnostic_readback_count()
                    == counters_before.diagnostic_readback_count()
                && counters_after
                    .metric_rows_readback_count()
                    .checked_sub(counters_before.metric_rows_readback_count())
                    == Some(u64::try_from(batch_count)?)
                && counters_after
                    .metric_rows_readback_rows()
                    .checked_sub(counters_before.metric_rows_readback_rows())
                    == Some(u64::try_from(genes.len())?),
            "retained validation performed an unplanned transfer or returned incomplete metric evidence"
        );
        parent.validate_session_v3(session)?;
        runtime.validate_current("after retained compact validation")?;
        let mut hash = Sha256::new();
        hash.update(b"neoethos.search.retained-compact-metrics.v3\0");
        hash.update(resident_execution_identity(&authority, timestamp_mode).as_bytes());
        if let Some(token) = adaptive_token_identity_sha256 {
            hash.update(token);
        }
        for receipt in &batch_receipts {
            hash.update(receipt.receipt_identity_sha256());
        }
        Ok(ExactResidentCompactMetricsV3 {
            authority,
            execution_identity_sha256: hex_lower(&hash.finalize()),
            metrics,
            batch_receipts,
            counters_before,
            counters_after,
            adaptive_token_identity_sha256,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn gene(terms: usize) -> Gene {
            Gene {
                indices: (0..terms).collect(),
                weights: vec![1.0; terms],
                ..Default::default()
            }
        }

        #[test]
        fn complete_cohort_batches_respect_both_gene_and_term_admission_without_truncation() {
            let genes = vec![
                gene(3),
                gene(2),
                gene(4),
                gene(1),
                gene(3),
                gene(2),
                gene(4),
            ];
            let chunks = gene_chunks(&genes, 3, 5).unwrap();
            assert_eq!(chunks, vec![0..2, 2..4, 4..6, 6..7]);
            assert_eq!(
                chunks
                    .iter()
                    .flat_map(|range| range.clone())
                    .collect::<Vec<_>>(),
                (0..7).collect::<Vec<_>>()
            );
            assert!(gene_chunks(&[gene(6)], 3, 5).is_err());
            assert!(gene_chunks(&[gene(0)], 3, 5).is_err());
            assert!(gene_chunks(&genes, 0, 5).is_err());
        }

        #[test]
        fn immutable_selected_views_exclude_suffix_trimmed_rows_and_all_holdout() {
            let selection = 20..80;
            assert!(
                validate_selected_view(
                    &ExactResidentDatasetViewV1::ContiguousRange { start: 20, end: 80 },
                    &selection
                )
                .is_ok()
            );
            assert!(
                validate_selected_view(
                    &ExactResidentDatasetViewV1::OrderedIndices {
                        indices: vec![20, 29, 79]
                    },
                    &selection
                )
                .is_ok()
            );
            for view in [
                ExactResidentDatasetViewV1::Full { row_count: 100 },
                ExactResidentDatasetViewV1::ContiguousRange { start: 19, end: 40 },
                ExactResidentDatasetViewV1::ContiguousRange { start: 60, end: 81 },
                ExactResidentDatasetViewV1::OrderedIndices {
                    indices: vec![20, 80],
                },
            ] {
                assert!(validate_selected_view(&view, &selection).is_err());
            }
        }

        #[test]
        fn strict_rows_preserve_economic_rejection_but_reject_fault_identity_and_missing_rows() {
            let scenarios = [crate::gpu_native::scenario::base_scenario(0, 81, 200)];
            let mut row = neoethos_gpu_contracts::device::NeoPopulationMetricRow {
                candidate_id: 0,
                scenario_id: 81,
                values: [0.0; 11],
            };
            row.values[0] = -1000.0;
            row.values[1] = f64::NEG_INFINITY;
            row.values[3] = 1.0;
            row.values[8] = 12.0;
            validate_rows(&scenarios, &[row]).unwrap();
            assert!(validate_rows(&scenarios, &[]).is_err());
            let mut wrong = row;
            wrong.scenario_id = 0;
            assert!(validate_rows(&scenarios, &[wrong]).is_err());
            wrong = row;
            wrong.values[2] = f64::NAN;
            assert!(validate_rows(&scenarios, &[wrong]).is_err());
            wrong = row;
            wrong.values[8] = 12.5;
            assert!(validate_rows(&scenarios, &[wrong]).is_err());
        }

        #[test]
        fn gene_chunk_mapping_keeps_global_scenario_and_local_gene_id_distinct() {
            let chunks = gene_chunks(&[gene(2), gene(2), gene(2), gene(2)], 2, 4).unwrap();
            let second = &chunks[1];
            let scenario =
                crate::gpu_native::scenario::base_scenario(1, (second.start + 1) as u64, 200);
            assert_eq!(scenario.base_candidate_id, 1);
            assert_eq!(scenario.scenario_id, 3);
            assert_eq!(scenario.window_len, 200);
        }
    }
}
