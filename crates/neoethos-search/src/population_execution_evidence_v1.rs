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
