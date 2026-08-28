//! Opaque, non-authoritative launch recipe for the strict resident Classic TA lane.
//!
//! Data owns registry, accounting and column-order resolution. This module
//! deliberately accepts only the immutable result of that resolution; the
//! recipe cannot confer device authority. The executable authority remains
//! the one-shot
//! [`GpuOnlyRunDeviceAdmissionV3`](crate::resident_feature_store_v3::GpuOnlyRunDeviceAdmissionV3)
//! retained by the resident-store
//! assembler and every launch is revalidated against the gpu-cuda/vector-ta
//! implementation before a device output can be appended.

use std::collections::BTreeSet;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use cust::context::{Context, CurrentContext};
use cust::memory::{AsyncCopyDestination, DeviceBuffer, DeviceCopy, GpuBuffer, LockedBuffer};
use cust::stream::Stream;
use cust::sys::CUstream;
use sha2::{Digest, Sha256};
use thiserror::Error;
use vector_ta::cuda::{
    CudaDeviceHighLowF64Ref, CudaDeviceOhlcvF64Ref, CudaDeviceSliceF64Ref, CudaDeviceSliceI64Ref,
    CudaF64IndicatorError, CudaF64Indicators, CudaSession, F64_EXACT_MATH_AUTHORITY_V3, F64Inputs,
    F64ResidentNamedPartsV3, F64ResidentObservedRouteManifestV3,
    F64ResidentSingleSweepAllocationPlanV4, F64ResidentSweepResultV3,
    preflight_resident_single_sweep_allocation_v4,
};
use vector_ta::indicators::dispatch::{F64InputKind, f64_kernel_for};

use neoethos_gpu_contracts::resident_feature_store_v3::{
    ResidentFeatureProducerV3, ResidentProducerCapabilityV3,
};

use crate::resident_feature_store_v3::{
    GpuOnlyRunDeviceAdmissionV3, ResidentF64FeatureBatchV3, ResidentFeatureColumnBindingV3,
    ResidentFeatureStoreCudaErrorV3, ResidentParentDatasetSourceV3, ResidentProducerReadyEventV3,
};

pub const RESIDENT_CLASSIC_TA_RECIPE_AUTHORITY_V3: &str =
    "neoethos.cuda.resident-classic-ta-recipe.v3";
pub const MAX_RESIDENT_CLASSIC_TA_BATCH_COLUMNS_V3: usize = 64;

const SHA256_BYTES: usize = 32;

#[cfg(feature = "cuda-device-fixtures")]
#[path = "resident_classic_ta_v3_device_fixture.rs"]
mod resident_classic_ta_v3_device_fixture;
#[cfg(feature = "cuda-device-fixtures")]
pub use resident_classic_ta_v3_device_fixture::{
    ResidentClassicTaDeviceFixtureReceiptV3, ResidentClassicTaDeviceFixtureRequestV3,
    ResidentClassicTaExpectedColumnV3, run_resident_classic_ta_v3_device_fixture,
};

pub fn resident_classic_ta_capability_v3()
-> Result<ResidentProducerCapabilityV3, ResidentFeatureStoreCudaErrorV3> {
    let mut implementation = Sha256::new();
    implementation.update(b"neoethos.gpu-cuda.resident-classic-ta.f64.semantic-v3");
    implementation.update(include_bytes!("resident_classic_ta_v3.rs"));
    implementation.update(include_bytes!("../native/resident_classic_ta_v3.cu"));
    implementation.update(F64_EXACT_MATH_AUTHORITY_V3.as_bytes());
    let implementation_sha256: [u8; SHA256_BYTES] = implementation.finalize().into();
    ResidentProducerCapabilityV3::new(
        ResidentFeatureProducerV3::ClassicTa,
        "neoethos.gpu-cuda.resident-classic-ta.f64.semantic-v3",
        implementation_sha256,
        F64_EXACT_MATH_AUTHORITY_V3,
    )
    .map_err(Into::into)
}

unsafe extern "C" {
    fn neoethos_resident_classic_derived_inputs_f64_v3(
        high: *const f64,
        low: *const f64,
        close: *const f64,
        rows: usize,
        hlc3: *mut f64,
        hl2: *mut f64,
        hlcc4: *mut f64,
        stream: CUstream,
    ) -> i32;
    fn neoethos_resident_classic_fill_nan_f64_v3(
        values: *mut f64,
        cells: usize,
        stream: CUstream,
    ) -> i32;
    fn neoethos_resident_classic_validity_u8_v3(
        value_addresses: *const u64,
        value_offsets: *const u64,
        all_nan_validity_codes: *const u8,
        rows: usize,
        columns: usize,
        first_finite_rows: *mut u64,
        validity_u8: *mut u8,
        device_error: *mut u32,
        stream: CUstream,
    ) -> i32;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentClassicTaStageV3 {
    Base,
    Historical,
    Extended,
}

impl ResidentClassicTaStageV3 {
    const fn tag(self) -> u8 {
        match self {
            Self::Base => 0,
            Self::Historical => 1,
            Self::Extended => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentClassicTaInputV3 {
    Close,
    Ohlc,
    Hlc3,
    Hlc3Volume,
    CloseVolume,
    HighLow,
    TimestampCloseVolume,
    Hl2,
    HighLowVolume,
    Hlcv,
    Ohlcv,
    OpenCloseVolume,
    Hlcc4,
    Volume,
    Hlcc4Volume,
    Hlc,
}

impl ResidentClassicTaInputV3 {
    const fn tag(self) -> u8 {
        match self {
            Self::Close => 0,
            Self::Ohlc => 1,
            Self::Hlc3 => 2,
            Self::Hlc3Volume => 3,
            Self::CloseVolume => 4,
            Self::HighLow => 5,
            Self::TimestampCloseVolume => 6,
            Self::Hl2 => 7,
            Self::HighLowVolume => 8,
            Self::Hlcv => 9,
            Self::Ohlcv => 10,
            Self::OpenCloseVolume => 11,
            Self::Hlcc4 => 12,
            Self::Volume => 13,
            Self::Hlcc4Volume => 14,
            Self::Hlc => 15,
        }
    }
}

/// Exact first-valid semantics selected by the canonical vector-ta dispatcher.
/// Named full-output routes use `NamedRouteOwned`: their method/entry-point
/// validates the more specific finite-count or consecutive-run authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentClassicTaFirstValidRuleV3 {
    AllInputsNonNan,
    AllInputsFinite,
    PriceVolumeFinite,
    HighLowFinitePositive,
    CloseReturnPair,
    NamedRouteOwned,
    NotApplicable,
}

impl ResidentClassicTaFirstValidRuleV3 {
    const fn tag(self) -> u8 {
        match self {
            Self::AllInputsNonNan => 0,
            Self::AllInputsFinite => 1,
            Self::PriceVolumeFinite => 2,
            Self::HighLowFinitePositive => 3,
            Self::CloseReturnPair => 4,
            Self::NamedRouteOwned => 5,
            Self::NotApplicable => 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResidentClassicTaParameterValueV3 {
    Usize(u64),
    I64(i64),
    I32(i32),
    Bool(bool),
    F64Bits(u64),
    Text(String),
}

impl ResidentClassicTaParameterValueV3 {
    fn update_hash(&self, hasher: &mut Sha256) -> Result<(), ResidentClassicTaRecipeErrorV3> {
        match self {
            Self::Usize(value) => {
                hasher.update([0]);
                hasher.update(value.to_le_bytes());
            }
            Self::I64(value) => {
                hasher.update([1]);
                hasher.update(value.to_le_bytes());
            }
            Self::I32(value) => {
                hasher.update([2]);
                hasher.update(value.to_le_bytes());
            }
            Self::Bool(value) => {
                hasher.update([3, u8::from(*value)]);
            }
            Self::F64Bits(value) => {
                hasher.update([4]);
                hasher.update(value.to_le_bytes());
            }
            Self::Text(value) => {
                hasher.update([5]);
                update_text(hasher, value)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResidentClassicTaParameterV3 {
    key: String,
    value: ResidentClassicTaParameterValueV3,
}

impl ResidentClassicTaParameterV3 {
    pub fn new(
        key: impl Into<String>,
        value: ResidentClassicTaParameterValueV3,
    ) -> Result<Self, ResidentClassicTaRecipeErrorV3> {
        let key = key.into();
        require_text("Classic TA parameter key", &key)?;
        if let ResidentClassicTaParameterValueV3::Text(text) = &value {
            require_text("Classic TA text parameter", text)?;
        }
        Ok(Self { key, value })
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn value(&self) -> &ResidentClassicTaParameterValueV3 {
        &self.value
    }

    fn update_hash(&self, hasher: &mut Sha256) -> Result<(), ResidentClassicTaRecipeErrorV3> {
        update_text(hasher, &self.key)?;
        self.value.update_hash(hasher)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentClassicTaOutputRouteV3 {
    destination_column: usize,
    feature_name: String,
    output_id: String,
    stage: ResidentClassicTaStageV3,
    swept_period: Option<u64>,
    canonical_parameter_tuple_sha256: [u8; SHA256_BYTES],
    route_receipt_sha256: [u8; SHA256_BYTES],
}

impl ResidentClassicTaOutputRouteV3 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        destination_column: usize,
        feature_name: impl Into<String>,
        output_id: impl Into<String>,
        stage: ResidentClassicTaStageV3,
        swept_period: Option<u64>,
        canonical_parameter_tuple_sha256: [u8; SHA256_BYTES],
        route_receipt_sha256: [u8; SHA256_BYTES],
    ) -> Result<Self, ResidentClassicTaRecipeErrorV3> {
        let feature_name = feature_name.into();
        let output_id = output_id.into();
        require_text("Classic TA feature name", &feature_name)?;
        require_text("Classic TA output id", &output_id)?;
        require_hash(
            "Classic TA canonical parameter tuple",
            &canonical_parameter_tuple_sha256,
        )?;
        require_hash("Classic TA route receipt", &route_receipt_sha256)?;
        let period_is_valid = match stage {
            ResidentClassicTaStageV3::Base => swept_period.is_none(),
            ResidentClassicTaStageV3::Historical | ResidentClassicTaStageV3::Extended => {
                swept_period.is_some_and(|period| period > 0)
            }
        };
        if !period_is_valid {
            return Err(ResidentClassicTaRecipeErrorV3::InvalidStagePeriod { feature_name });
        }
        Ok(Self {
            destination_column,
            feature_name,
            output_id,
            stage,
            swept_period,
            canonical_parameter_tuple_sha256,
            route_receipt_sha256,
        })
    }

    pub const fn destination_column(&self) -> usize {
        self.destination_column
    }

    pub fn feature_name(&self) -> &str {
        &self.feature_name
    }

    pub fn output_id(&self) -> &str {
        &self.output_id
    }

    pub const fn stage(&self) -> ResidentClassicTaStageV3 {
        self.stage
    }

    pub const fn swept_period(&self) -> Option<u64> {
        self.swept_period
    }

    pub const fn canonical_parameter_tuple_sha256(&self) -> [u8; SHA256_BYTES] {
        self.canonical_parameter_tuple_sha256
    }

    pub const fn route_receipt_sha256(&self) -> [u8; SHA256_BYTES] {
        self.route_receipt_sha256
    }

    fn update_hash(&self, hasher: &mut Sha256) -> Result<(), ResidentClassicTaRecipeErrorV3> {
        update_usize(hasher, self.destination_column)?;
        update_text(hasher, &self.feature_name)?;
        update_text(hasher, &self.output_id)?;
        hasher.update([self.stage.tag()]);
        match self.swept_period {
            Some(period) => {
                hasher.update([1]);
                hasher.update(period.to_le_bytes());
            }
            None => hasher.update([0]),
        }
        hasher.update(self.canonical_parameter_tuple_sha256);
        hasher.update(self.route_receipt_sha256);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentClassicTaLaunchRecipeV3 {
    indicator_id: String,
    entry_point: String,
    input: ResidentClassicTaInputV3,
    first_valid_rule: ResidentClassicTaFirstValidRuleV3,
    parameters: Vec<ResidentClassicTaParameterV3>,
    outputs: Vec<ResidentClassicTaOutputRouteV3>,
    all_nan_validity_code: u8,
    next_destination_column: usize,
}

impl ResidentClassicTaLaunchRecipeV3 {
    pub fn new(
        indicator_id: impl Into<String>,
        entry_point: impl Into<String>,
        input: ResidentClassicTaInputV3,
        first_valid_rule: ResidentClassicTaFirstValidRuleV3,
        parameters: Vec<ResidentClassicTaParameterV3>,
        outputs: Vec<ResidentClassicTaOutputRouteV3>,
        all_nan_validity_code: u8,
    ) -> Result<Self, ResidentClassicTaRecipeErrorV3> {
        let indicator_id = indicator_id.into();
        let entry_point = entry_point.into();
        require_text("Classic TA indicator id", &indicator_id)?;
        require_text("Classic TA CUDA entry point", &entry_point)?;
        if outputs.is_empty() || outputs.len() > MAX_RESIDENT_CLASSIC_TA_BATCH_COLUMNS_V3 {
            return Err(ResidentClassicTaRecipeErrorV3::InvalidLaunchWidth {
                columns: outputs.len(),
            });
        }
        if all_nan_validity_code > 9 {
            return Err(ResidentClassicTaRecipeErrorV3::InvalidValidityCode(
                all_nan_validity_code,
            ));
        }
        let mut parameter_keys = BTreeSet::new();
        for parameter in &parameters {
            if !parameter_keys.insert(parameter.key()) {
                return Err(ResidentClassicTaRecipeErrorV3::DuplicateParameterKey(
                    parameter.key().to_owned(),
                ));
            }
        }
        let first_destination_column = outputs[0].destination_column();
        for (offset, output) in outputs.iter().enumerate() {
            let expected = first_destination_column.checked_add(offset).ok_or(
                ResidentClassicTaRecipeErrorV3::ArithmeticOverflow(
                    "Classic TA launch output range",
                ),
            )?;
            if output.destination_column() != expected {
                return Err(ResidentClassicTaRecipeErrorV3::NonContiguousOutputRange {
                    expected,
                    actual: output.destination_column(),
                });
            }
        }
        let next_destination_column = first_destination_column.checked_add(outputs.len()).ok_or(
            ResidentClassicTaRecipeErrorV3::ArithmeticOverflow("Classic TA launch destination end"),
        )?;
        Ok(Self {
            indicator_id,
            entry_point,
            input,
            first_valid_rule,
            parameters,
            outputs,
            all_nan_validity_code,
            next_destination_column,
        })
    }

    pub fn indicator_id(&self) -> &str {
        &self.indicator_id
    }

    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    pub const fn input(&self) -> ResidentClassicTaInputV3 {
        self.input
    }

    pub const fn first_valid_rule(&self) -> ResidentClassicTaFirstValidRuleV3 {
        self.first_valid_rule
    }

    pub fn parameters(&self) -> &[ResidentClassicTaParameterV3] {
        &self.parameters
    }

    pub fn outputs(&self) -> &[ResidentClassicTaOutputRouteV3] {
        &self.outputs
    }

    pub const fn all_nan_validity_code(&self) -> u8 {
        self.all_nan_validity_code
    }

    pub fn first_destination_column(&self) -> usize {
        self.outputs[0].destination_column()
    }

    pub const fn next_destination_column(&self) -> usize {
        self.next_destination_column
    }

    fn update_hash(&self, hasher: &mut Sha256) -> Result<(), ResidentClassicTaRecipeErrorV3> {
        update_text(hasher, &self.indicator_id)?;
        update_text(hasher, &self.entry_point)?;
        hasher.update([self.input.tag(), self.first_valid_rule.tag()]);
        update_usize(hasher, self.parameters.len())?;
        for parameter in &self.parameters {
            parameter.update_hash(hasher)?;
        }
        update_usize(hasher, self.outputs.len())?;
        for output in &self.outputs {
            output.update_hash(hasher)?;
        }
        hasher.update([self.all_nan_validity_code]);
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentClassicTaRecipeV3 {
    rows: usize,
    budget_rows: usize,
    available_bytes_at_admission: u64,
    admitted_working_set_sha256: [u8; SHA256_BYTES],
    launches: Vec<ResidentClassicTaLaunchRecipeV3>,
    route_plan_sha256: [u8; SHA256_BYTES],
}

impl ResidentClassicTaRecipeV3 {
    pub fn seal(
        rows: usize,
        budget_rows: usize,
        available_bytes_at_admission: u64,
        admitted_working_set_sha256: [u8; SHA256_BYTES],
        launches: Vec<ResidentClassicTaLaunchRecipeV3>,
    ) -> Result<Self, ResidentClassicTaRecipeErrorV3> {
        if rows == 0 || budget_rows < rows || available_bytes_at_admission == 0 {
            return Err(ResidentClassicTaRecipeErrorV3::InvalidExtent {
                rows,
                budget_rows,
                available_bytes_at_admission,
            });
        }
        require_hash(
            "Classic TA admitted working-set identity",
            &admitted_working_set_sha256,
        )?;
        if launches.is_empty() {
            return Err(ResidentClassicTaRecipeErrorV3::EmptyLaunchPlan);
        }
        let mut next_destination_column = 0_usize;
        let mut feature_names = BTreeSet::new();
        for launch in &launches {
            if launch.first_destination_column() != next_destination_column {
                return Err(ResidentClassicTaRecipeErrorV3::NonContiguousLaunchRange {
                    expected: next_destination_column,
                    actual: launch.first_destination_column(),
                });
            }
            for output in launch.outputs() {
                if !feature_names.insert(output.feature_name()) {
                    return Err(ResidentClassicTaRecipeErrorV3::DuplicateFeatureName(
                        output.feature_name().to_owned(),
                    ));
                }
            }
            next_destination_column = launch.next_destination_column();
        }
        let route_plan_sha256 = hash_recipe(
            rows,
            budget_rows,
            available_bytes_at_admission,
            admitted_working_set_sha256,
            &launches,
        )?;
        Ok(Self {
            rows,
            budget_rows,
            available_bytes_at_admission,
            admitted_working_set_sha256,
            launches,
            route_plan_sha256,
        })
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn budget_rows(&self) -> usize {
        self.budget_rows
    }

    pub const fn available_bytes_at_admission(&self) -> u64 {
        self.available_bytes_at_admission
    }

    pub const fn admitted_working_set_sha256(&self) -> [u8; SHA256_BYTES] {
        self.admitted_working_set_sha256
    }

    pub fn launches(&self) -> &[ResidentClassicTaLaunchRecipeV3] {
        &self.launches
    }

    pub const fn route_plan_sha256(&self) -> [u8; SHA256_BYTES] {
        self.route_plan_sha256
    }

    pub fn output_count(&self) -> usize {
        self.launches
            .iter()
            .map(|launch| launch.outputs().len())
            .sum()
    }
}

/// Structural preflight is intentionally separate from runtime authority.
/// The executor consumes this recipe only after the resident-store assembler
/// proves it belongs to the moved
/// [`GpuOnlyRunDeviceAdmissionV3`](crate::resident_feature_store_v3::GpuOnlyRunDeviceAdmissionV3).
pub fn preflight_resident_classic_ta_recipe_v3(
    recipe: ResidentClassicTaRecipeV3,
) -> Result<ResidentClassicTaRecipeV3, ResidentClassicTaRecipeErrorV3> {
    if recipe.route_plan_sha256
        != hash_recipe(
            recipe.rows,
            recipe.budget_rows,
            recipe.available_bytes_at_admission,
            recipe.admitted_working_set_sha256,
            &recipe.launches,
        )?
    {
        return Err(ResidentClassicTaRecipeErrorV3::RoutePlanHashMismatch);
    }
    Ok(recipe)
}

/// Exact owner-derived memory for one Classic launch. The primary variant
/// retains VectorTA's move-only single-sweep plan; named all-output variants
/// remain inadmissible until their real allocation owners expose the same
/// pre-device contract.
#[derive(Debug, PartialEq, Eq)]
pub enum ResidentClassicTaLaunchMemoryPlanV4 {
    Primary {
        vector_plan: F64ResidentSingleSweepAllocationPlanV4,
        selected_value_bytes: usize,
        all_output_retained_bytes: usize,
        additional_retained_bytes: usize,
        validity_bytes: usize,
        validity_scratch_bytes: usize,
        retained_scratch_bytes: usize,
        ready_event_count: usize,
    },
    Warmup {
        selected_value_bytes: usize,
        all_output_retained_bytes: usize,
        additional_retained_bytes: usize,
        validity_bytes: usize,
        validity_scratch_bytes: usize,
        retained_scratch_bytes: usize,
        ready_event_count: usize,
    },
}

impl ResidentClassicTaLaunchMemoryPlanV4 {
    fn vector_plan(&self) -> Option<&F64ResidentSingleSweepAllocationPlanV4> {
        match self {
            Self::Primary { vector_plan, .. } => Some(vector_plan),
            Self::Warmup { .. } => None,
        }
    }

    pub const fn selected_value_bytes(&self) -> usize {
        match self {
            Self::Primary {
                selected_value_bytes,
                ..
            }
            | Self::Warmup {
                selected_value_bytes,
                ..
            } => *selected_value_bytes,
        }
    }

    pub const fn all_output_retained_bytes(&self) -> usize {
        match self {
            Self::Primary {
                all_output_retained_bytes,
                ..
            }
            | Self::Warmup {
                all_output_retained_bytes,
                ..
            } => *all_output_retained_bytes,
        }
    }

    pub const fn additional_retained_bytes(&self) -> usize {
        match self {
            Self::Primary {
                additional_retained_bytes,
                ..
            }
            | Self::Warmup {
                additional_retained_bytes,
                ..
            } => *additional_retained_bytes,
        }
    }

    pub const fn validity_bytes(&self) -> usize {
        match self {
            Self::Primary { validity_bytes, .. } | Self::Warmup { validity_bytes, .. } => {
                *validity_bytes
            }
        }
    }

    pub const fn validity_scratch_bytes(&self) -> usize {
        match self {
            Self::Primary {
                validity_scratch_bytes,
                ..
            }
            | Self::Warmup {
                validity_scratch_bytes,
                ..
            } => *validity_scratch_bytes,
        }
    }

    pub const fn retained_scratch_bytes(&self) -> usize {
        match self {
            Self::Primary {
                retained_scratch_bytes,
                ..
            }
            | Self::Warmup {
                retained_scratch_bytes,
                ..
            } => *retained_scratch_bytes,
        }
    }

    pub const fn ready_event_count(&self) -> usize {
        match self {
            Self::Primary {
                ready_event_count, ..
            }
            | Self::Warmup {
                ready_event_count, ..
            } => *ready_event_count,
        }
    }
}

/// Move-only Classic memory authority produced without a CUDA context. It is
/// bound to the immutable recipe and must compare equal to runtime allocation
/// evidence before a batch may enter the resident store.
#[derive(Debug, PartialEq, Eq)]
pub struct ResidentClassicTaPreDeviceMemoryReceiptV4 {
    recipe_sha256: [u8; SHA256_BYTES],
    rows: usize,
    derived_input_bytes: usize,
    derived_ready_event_count: usize,
    launch_plans: Vec<ResidentClassicTaLaunchMemoryPlanV4>,
}

impl ResidentClassicTaPreDeviceMemoryReceiptV4 {
    pub const fn recipe_sha256(&self) -> [u8; SHA256_BYTES] {
        self.recipe_sha256
    }

    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn derived_input_bytes(&self) -> usize {
        self.derived_input_bytes
    }

    pub const fn derived_ready_event_count(&self) -> usize {
        self.derived_ready_event_count
    }

    pub fn launch_plans(&self) -> &[ResidentClassicTaLaunchMemoryPlanV4] {
        &self.launch_plans
    }
}

/// Plan every currently exact Classic primary/warmup launch before the run
/// carrier is consumed. A named multi-output launch fails closed: its actual
/// VectorTA owner must expose all-output/parameter/scratch sizing first.
pub fn preflight_resident_classic_ta_memory_v4(
    recipe: &ResidentClassicTaRecipeV3,
) -> Result<ResidentClassicTaPreDeviceMemoryReceiptV4, ResidentClassicTaExecutorErrorV3> {
    if recipe.route_plan_sha256()
        != hash_recipe(
            recipe.rows(),
            recipe.budget_rows(),
            recipe.available_bytes_at_admission(),
            recipe.admitted_working_set_sha256(),
            recipe.launches(),
        )?
    {
        return Err(ResidentClassicTaRecipeErrorV3::RoutePlanHashMismatch.into());
    }
    let rows = recipe.rows();
    let derived_input_bytes = rows
        .checked_mul(3)
        .and_then(|elements| elements.checked_mul(std::mem::size_of::<f64>()))
        .ok_or(ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
            "resident Classic TA derived-input preflight bytes",
        ))?;
    let mut launch_plans = Vec::with_capacity(recipe.launches().len());
    for launch in recipe.launches() {
        let columns = launch.outputs().len();
        let selected_value_bytes = rows
            .checked_mul(columns)
            .and_then(|cells| cells.checked_mul(std::mem::size_of::<f64>()))
            .ok_or(ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA selected value preflight bytes",
            ))?;
        let validity_bytes = rows.checked_mul(columns).ok_or(
            ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA validity preflight bytes",
            ),
        )?;
        let validity_scratch_bytes = columns
            .checked_mul(25)
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u32>()))
            .ok_or(ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA validity scratch preflight bytes",
            ))?;
        let warmup_only = launch
            .outputs()
            .first()
            .and_then(ResidentClassicTaOutputRouteV3::swept_period)
            .is_some_and(|period| (period as f64) * 1.25 >= rows as f64);
        if warmup_only {
            launch_plans.push(ResidentClassicTaLaunchMemoryPlanV4::Warmup {
                selected_value_bytes,
                all_output_retained_bytes: selected_value_bytes,
                additional_retained_bytes: derived_input_bytes,
                validity_bytes,
                validity_scratch_bytes,
                retained_scratch_bytes: validity_scratch_bytes,
                ready_event_count: 1,
            });
            continue;
        }
        let primary = f64_kernel_for(launch.indicator_id()).filter(|spec| {
            launch.first_valid_rule() != ResidentClassicTaFirstValidRuleV3::NamedRouteOwned
                && spec.kernel.entry_point() == launch.entry_point()
                && columns == 1
                && spec.primary_output_id() == Some(launch.outputs()[0].output_id())
        });
        let Some(spec) = primary else {
            return Err(
                ResidentClassicTaExecutorErrorV3::UnsupportedNamedAllocationPlan {
                    indicator_id: launch.indicator_id().to_owned(),
                },
            );
        };
        let period = require_usize_parameter_v3(launch, "cuda_period")?;
        let period = i32::try_from(period).map_err(|_| {
            ResidentClassicTaExecutorErrorV3::ParameterWidth {
                indicator_id: launch.indicator_id().to_owned(),
                key: "cuda_period",
            }
        })?;
        let vector_plan =
            preflight_resident_single_sweep_allocation_v4(spec.kernel, &[period], rows)?;
        if vector_plan.output_bytes() != selected_value_bytes {
            return Err(
                ResidentClassicTaExecutorErrorV3::PreDeviceMemoryReceiptMismatch {
                    indicator_id: launch.indicator_id().to_owned(),
                },
            );
        }
        let vector_scratch_bytes = vector_plan
            .retained_parameter_bytes()?
            .checked_add(vector_plan.retained_scratch_bytes()?)
            .ok_or(ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA VectorTA preflight scratch bytes",
            ))?;
        let retained_scratch_bytes = vector_scratch_bytes
            .checked_add(validity_scratch_bytes)
            .ok_or(ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA total preflight scratch bytes",
            ))?;
        launch_plans.push(ResidentClassicTaLaunchMemoryPlanV4::Primary {
            vector_plan,
            selected_value_bytes,
            all_output_retained_bytes: selected_value_bytes,
            additional_retained_bytes: derived_input_bytes,
            validity_bytes,
            validity_scratch_bytes,
            retained_scratch_bytes,
            ready_event_count: 1,
        });
    }
    Ok(ResidentClassicTaPreDeviceMemoryReceiptV4 {
        recipe_sha256: recipe.route_plan_sha256(),
        rows,
        derived_input_bytes,
        derived_ready_event_count: 1,
        launch_plans,
    })
}

/// Bind Data's already-global schema span to the immutable producer-local
/// Classic recipe. The first Classic ordinal must follow at least one prior
/// producer column (SMC in schema-v4); the resident-store assembler later
/// proves the exact preceding span by matching these bindings against its
/// sealed global route ledger.
pub fn validate_admitted_global_bindings_v4(
    recipe: &ResidentClassicTaRecipeV3,
    admitted_global_bindings: &[ResidentFeatureColumnBindingV3],
) -> Result<(), ResidentClassicTaExecutorErrorV3> {
    if admitted_global_bindings.len() != recipe.output_count() {
        return Err(ResidentClassicTaExecutorErrorV3::AdmittedGlobalBindingsMismatch);
    }
    let Some(first_binding) = admitted_global_bindings.first() else {
        return Err(ResidentClassicTaExecutorErrorV3::AdmittedGlobalBindingsMismatch);
    };
    let classic_global_start = first_binding.ordinal;
    if classic_global_start == 0 {
        return Err(ResidentClassicTaExecutorErrorV3::AdmittedGlobalBindingsMismatch);
    }
    let mut local_column = 0_usize;
    for launch in recipe.launches() {
        for route in launch.outputs() {
            let expected_global_ordinal = classic_global_start.checked_add(local_column).ok_or(
                ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                    "resident Classic TA global column ordinal",
                ),
            )?;
            let binding = admitted_global_bindings
                .get(local_column)
                .ok_or(ResidentClassicTaExecutorErrorV3::AdmittedGlobalBindingsMismatch)?;
            if route.destination_column() != local_column
                || binding.ordinal != expected_global_ordinal
                || binding.feature_name != route.feature_name()
                || binding.canonical_parameter_tuple_sha256
                    != route.canonical_parameter_tuple_sha256()
                || binding.route_receipt_sha256.iter().all(|byte| *byte == 0)
            {
                return Err(ResidentClassicTaExecutorErrorV3::AdmittedGlobalBindingsMismatch);
            }
            local_column = local_column.checked_add(1).ok_or(
                ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                    "resident Classic TA local column cursor",
                ),
            )?;
        }
    }
    if local_column != admitted_global_bindings.len() {
        return Err(ResidentClassicTaExecutorErrorV3::AdmittedGlobalBindingsMismatch);
    }
    Ok(())
}

/// One allocation that is always retired on the carried producer stream.
///
/// The ordinary `DeviceBuffer` destructor is never the successful-path
/// authority for a Classic TA producer allocation. If the admitted primary
/// context cannot be restored, leaking is safer than invoking a legacy free
/// while queued work may still refer to the allocation.
#[derive(Debug)]
struct ResidentClassicTaDeviceBufferV3<T: DeviceCopy> {
    buffer: Option<DeviceBuffer<T>>,
    context: Arc<Context>,
    stream: Arc<Stream>,
}

impl<T: DeviceCopy> ResidentClassicTaDeviceBufferV3<T> {
    fn uninitialized_async(
        len: usize,
        context: Arc<Context>,
        stream: Arc<Stream>,
    ) -> Result<Self, ResidentClassicTaExecutorErrorV3> {
        // SAFETY: this owner retains the exact context and stream through its
        // stream-ordered destruction path. Every caller initializes the full
        // allocation before exposing it to a downstream batch.
        let buffer = unsafe { DeviceBuffer::<T>::uninitialized_async(len, stream.as_ref())? };
        Ok(Self {
            buffer: Some(buffer),
            context,
            stream,
        })
    }

    fn is_owned_by_stream(&self, stream: &Stream) -> bool {
        !stream.as_inner().is_null() && self.stream.as_inner() == stream.as_inner()
    }
}

impl<T: DeviceCopy> Deref for ResidentClassicTaDeviceBufferV3<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("live Classic TA allocation retains its device buffer")
    }
}

impl<T: DeviceCopy> DerefMut for ResidentClassicTaDeviceBufferV3<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer
            .as_mut()
            .expect("live Classic TA allocation retains its device buffer")
    }
}

impl<T: DeviceCopy> Drop for ResidentClassicTaDeviceBufferV3<T> {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        if CurrentContext::set_current(self.context.as_ref()).is_ok() {
            let _ = buffer.drop_async(self.stream.as_ref());
        } else {
            std::mem::forget(buffer);
        }
    }
}

#[derive(Debug)]
struct ResidentClassicTaDerivedInputsV3 {
    hlc3: ResidentClassicTaDeviceBufferV3<f64>,
    hl2: ResidentClassicTaDeviceBufferV3<f64>,
    hlcc4: ResidentClassicTaDeviceBufferV3<f64>,
    producer_ready_event: ResidentProducerReadyEventV3,
    retained_device_bytes: usize,
}

impl ResidentClassicTaDerivedInputsV3 {
    fn launch(
        parent: &dyn ResidentParentDatasetSourceV3,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        device_ordinal: u32,
    ) -> Result<Self, ResidentClassicTaExecutorErrorV3> {
        let rows = parent.rows();
        if rows == 0 {
            return Err(ResidentClassicTaExecutorErrorV3::InvalidInput(
                "resident Classic TA parent is empty".into(),
            ));
        }
        parent.producer_ready_event().wait_before_read(
            context.as_ref(),
            stream.as_ref(),
            device_ordinal,
        )?;
        let hlc3 = ResidentClassicTaDeviceBufferV3::<f64>::uninitialized_async(
            rows,
            Arc::clone(context),
            Arc::clone(stream),
        )?;
        let hl2 = ResidentClassicTaDeviceBufferV3::<f64>::uninitialized_async(
            rows,
            Arc::clone(context),
            Arc::clone(stream),
        )?;
        let hlcc4 = ResidentClassicTaDeviceBufferV3::<f64>::uninitialized_async(
            rows,
            Arc::clone(context),
            Arc::clone(stream),
        )?;
        let status = unsafe {
            // SAFETY: the parent trait guarantees exact row extents and the
            // three destinations were allocated for exactly `rows` f64 cells.
            // The recorded parent event is ordered before this launch on the
            // same admitted non-default stream.
            neoethos_resident_classic_derived_inputs_f64_v3(
                parent.high().as_device_ptr().as_ptr(),
                parent.low().as_device_ptr().as_ptr(),
                parent.close().as_device_ptr().as_ptr(),
                rows,
                hlc3.as_device_ptr().as_mut_ptr(),
                hl2.as_device_ptr().as_mut_ptr(),
                hlcc4.as_device_ptr().as_mut_ptr(),
                stream.as_inner(),
            )
        };
        if status != 0 {
            return Err(ResidentClassicTaExecutorErrorV3::Native {
                operation: "neoethos_resident_classic_derived_inputs_f64_v3",
                status,
            });
        }
        let producer_ready_event =
            ResidentProducerReadyEventV3::record(context, stream, device_ordinal)?;
        let retained_device_bytes = rows.checked_mul(3 * std::mem::size_of::<f64>()).ok_or(
            ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA derived-input bytes",
            ),
        )?;
        Ok(Self {
            hlc3,
            hl2,
            hlcc4,
            producer_ready_event,
            retained_device_bytes,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct ResidentClassicTaParentViewsV3 {
    open: CudaDeviceSliceF64Ref,
    high: CudaDeviceSliceF64Ref,
    low: CudaDeviceSliceF64Ref,
    close: CudaDeviceSliceF64Ref,
    volume: CudaDeviceSliceF64Ref,
    timestamps: CudaDeviceSliceI64Ref,
    hlc3: CudaDeviceSliceF64Ref,
    hl2: CudaDeviceSliceF64Ref,
    hlcc4: CudaDeviceSliceF64Ref,
}

impl ResidentClassicTaParentViewsV3 {
    fn new(
        parent: &dyn ResidentParentDatasetSourceV3,
        derived: &ResidentClassicTaDerivedInputsV3,
    ) -> Result<Self, ResidentClassicTaExecutorErrorV3> {
        let rows = parent.rows();
        let device_ordinal = parent.device_ordinal();
        let f64_view = |buffer: &DeviceBuffer<f64>, field: &'static str| {
            // SAFETY: the sealed parent/derived owners retain each allocation
            // for the complete executor lifetime and expose its exact extent.
            unsafe {
                CudaDeviceSliceF64Ref::from_raw_parts(
                    buffer.as_device_ptr().as_raw(),
                    rows,
                    device_ordinal,
                )
            }
            .map_err(|error| ResidentClassicTaExecutorErrorV3::DeviceView { field, error })
        };
        let timestamps = unsafe {
            // SAFETY: same sealed parent ownership and exact `rows` extent.
            CudaDeviceSliceI64Ref::from_raw_parts(
                parent.timestamps().as_device_ptr().as_raw(),
                rows,
                device_ordinal,
            )
        }
        .map_err(|error| ResidentClassicTaExecutorErrorV3::DeviceView {
            field: "timestamps",
            error,
        })?;
        Ok(Self {
            open: f64_view(parent.open(), "open")?,
            high: f64_view(parent.high(), "high")?,
            low: f64_view(parent.low(), "low")?,
            close: f64_view(parent.close(), "close")?,
            volume: f64_view(parent.volume(), "volume")?,
            timestamps,
            hlc3: f64_view(&derived.hlc3, "hlc3")?,
            hl2: f64_view(&derived.hl2, "hl2")?,
            hlcc4: f64_view(&derived.hlcc4, "hlcc4")?,
        })
    }

    fn inputs(self, input: ResidentClassicTaInputV3) -> F64Inputs {
        match input {
            ResidentClassicTaInputV3::Close => F64Inputs::Prices(self.close),
            ResidentClassicTaInputV3::Ohlc => F64Inputs::Ohlc4 {
                open: self.open,
                high: self.high,
                low: self.low,
                close: self.close,
            },
            ResidentClassicTaInputV3::Hlc3 => F64Inputs::Prices(self.hlc3),
            ResidentClassicTaInputV3::Hlc3Volume => F64Inputs::PriceVolume {
                price: self.hlc3,
                volume: self.volume,
            },
            ResidentClassicTaInputV3::CloseVolume => F64Inputs::PriceVolume {
                price: self.close,
                volume: self.volume,
            },
            ResidentClassicTaInputV3::HighLow => F64Inputs::HighLow {
                high: self.high,
                low: self.low,
            },
            ResidentClassicTaInputV3::TimestampCloseVolume => F64Inputs::TimestampPriceVolume {
                timestamps: self.timestamps,
                price: self.close,
                volume: self.volume,
            },
            ResidentClassicTaInputV3::Hl2 => F64Inputs::Prices(self.hl2),
            ResidentClassicTaInputV3::HighLowVolume => F64Inputs::HighLowVolume {
                high: self.high,
                low: self.low,
                volume: self.volume,
            },
            ResidentClassicTaInputV3::Hlcv => F64Inputs::Hlcv {
                high: self.high,
                low: self.low,
                close: self.close,
                volume: self.volume,
            },
            ResidentClassicTaInputV3::Ohlcv => F64Inputs::Ohlcv5 {
                open: self.open,
                high: self.high,
                low: self.low,
                close: self.close,
                volume: self.volume,
            },
            ResidentClassicTaInputV3::OpenCloseVolume => F64Inputs::OpenCloseVolume {
                open: self.open,
                close: self.close,
                volume: self.volume,
            },
            ResidentClassicTaInputV3::Hlcc4 => F64Inputs::Prices(self.hlcc4),
            ResidentClassicTaInputV3::Volume => F64Inputs::Prices(self.volume),
            ResidentClassicTaInputV3::Hlcc4Volume => F64Inputs::PriceVolume {
                price: self.hlcc4,
                volume: self.volume,
            },
            ResidentClassicTaInputV3::Hlc => F64Inputs::Hlc {
                high: self.high,
                low: self.low,
                close: self.close,
            },
        }
    }

    fn ohlcv(self) -> Result<CudaDeviceOhlcvF64Ref, ResidentClassicTaExecutorErrorV3> {
        CudaDeviceOhlcvF64Ref::new(
            self.open,
            self.high,
            self.low,
            self.close,
            self.volume,
            None,
        )
        .map_err(|error| ResidentClassicTaExecutorErrorV3::DeviceView {
            field: "ohlcv",
            error,
        })
    }

    fn high_low(self) -> Result<CudaDeviceHighLowF64Ref, ResidentClassicTaExecutorErrorV3> {
        CudaDeviceHighLowF64Ref::new(self.high, self.low).map_err(|error| {
            ResidentClassicTaExecutorErrorV3::DeviceView {
                field: "high_low",
                error,
            }
        })
    }
}

#[derive(Debug)]
struct ResidentClassicTaPinnedCopyV3<T: DeviceCopy> {
    host: Option<LockedBuffer<T>>,
    device: Option<ResidentClassicTaDeviceBufferV3<T>>,
}

impl<T: DeviceCopy> ResidentClassicTaPinnedCopyV3<T> {
    fn copy_async(
        source: &[T],
        context: &Arc<Context>,
        stream: &Arc<Stream>,
    ) -> Result<Self, ResidentClassicTaExecutorErrorV3> {
        let host = LockedBuffer::from_slice(source)?;
        let mut device = ResidentClassicTaDeviceBufferV3::<T>::uninitialized_async(
            source.len(),
            Arc::clone(context),
            Arc::clone(stream),
        )?;
        if let Err(error) = unsafe { device.async_copy_from(&host, stream.as_ref()) } {
            std::mem::forget(host);
            std::mem::forget(device);
            return Err(error.into());
        }
        Ok(Self {
            host: Some(host),
            device: Some(device),
        })
    }

    fn device(&self) -> &DeviceBuffer<T> {
        self.device
            .as_ref()
            .expect("live Classic TA compact copy retains its device buffer")
    }

    fn retained_device_bytes(&self) -> usize {
        self.device().len().saturating_mul(std::mem::size_of::<T>())
    }

    fn enqueue_release(
        mut self,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if !self
            .device
            .as_ref()
            .is_some_and(|device| device.is_owned_by_stream(release_stream))
        {
            std::mem::forget(self);
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        drop(self.host.take());
        drop(self.device.take());
        Ok(())
    }
}

impl<T: DeviceCopy> Drop for ResidentClassicTaPinnedCopyV3<T> {
    fn drop(&mut self) {
        // If construction/launch fails after an async compact copy, CUDA may
        // still retain the page-locked host pointer. Leak that tiny control
        // plane input; the device owner itself retires stream-ordered.
        if let Some(host) = self.host.take() {
            std::mem::forget(host);
        }
    }
}

#[derive(Debug)]
enum ResidentClassicTaOutputOwnerV3 {
    Primary(F64ResidentSweepResultV3),
    Named {
        parts: F64ResidentNamedPartsV3,
        output_indices: Vec<usize>,
    },
    Warmup(Vec<ResidentClassicTaDeviceBufferV3<f64>>),
}

impl ResidentClassicTaOutputOwnerV3 {
    fn value_buffer(&self, column: usize) -> &DeviceBuffer<f64> {
        match self {
            Self::Primary(primary) => {
                assert_eq!(column, 0, "primary Classic TA batch has one output");
                primary.output_buffer()
            }
            Self::Named {
                parts,
                output_indices,
            } => parts.outputs()[output_indices[column]].matrix.buffer(),
            Self::Warmup(buffers) => &buffers[column],
        }
    }

    fn retained_output_bytes(&self) -> usize {
        match self {
            Self::Primary(primary) => primary
                .rows()
                .saturating_mul(primary.cols())
                .saturating_mul(std::mem::size_of::<f64>()),
            Self::Named { parts, .. } => parts
                .outputs()
                .iter()
                .map(|output| {
                    output
                        .matrix
                        .len()
                        .saturating_mul(std::mem::size_of::<f64>())
                })
                .sum(),
            Self::Warmup(buffers) => buffers
                .iter()
                .map(|buffer| buffer.len().saturating_mul(std::mem::size_of::<f64>()))
                .sum(),
        }
    }

    fn retained_launch_scratch_bytes(&self) -> usize {
        match self {
            Self::Primary(primary) => primary
                .retained_parameter_bytes()
                .saturating_add(primary.retained_scratch_bytes()),
            Self::Named { parts, .. } => parts
                .retained_parameter_bytes()
                .saturating_add(parts.retained_scratch_bytes()),
            Self::Warmup(_) => 0,
        }
    }

    fn enqueue_release(
        self,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        match self {
            Self::Primary(primary) => primary
                .enqueue_release_v3(release_stream)
                .map_err(vector_ta_error_v3),
            Self::Named { parts, .. } => parts
                .enqueue_release_v3(release_stream)
                .map_err(vector_ta_error_v3),
            Self::Warmup(buffers) => {
                if buffers
                    .iter()
                    .any(|buffer| !buffer.is_owned_by_stream(release_stream))
                {
                    std::mem::forget(buffers);
                    return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
                }
                drop(buffers);
                Ok(())
            }
        }
    }
}

fn vector_ta_error_v3(error: CudaF64IndicatorError) -> ResidentFeatureStoreCudaErrorV3 {
    ResidentFeatureStoreCudaErrorV3::InvalidInput(format!(
        "resident Classic TA stream retirement failed: {error}"
    ))
}

#[derive(Debug)]
pub struct PendingResidentClassicTaBatchV3 {
    bindings: Vec<ResidentFeatureColumnBindingV3>,
    output_owner: Option<ResidentClassicTaOutputOwnerV3>,
    value_addresses: Option<ResidentClassicTaPinnedCopyV3<u64>>,
    value_offsets: Option<ResidentClassicTaPinnedCopyV3<u64>>,
    all_nan_validity_codes: Option<ResidentClassicTaPinnedCopyV3<u8>>,
    first_finite_rows: Option<ResidentClassicTaDeviceBufferV3<u64>>,
    validity_u8: Option<ResidentClassicTaDeviceBufferV3<u8>>,
    validity_device_error: Option<ResidentClassicTaDeviceBufferV3<u32>>,
    producer_ready_event: ResidentProducerReadyEventV3,
    context: Arc<Context>,
    stream: Arc<Stream>,
    rows: usize,
    device_ordinal: u32,
    retained_device_bytes: usize,
    retained_scratch_bytes: usize,
}

impl PendingResidentClassicTaBatchV3 {
    /// Remove the executor-owned derived-input charge before an HTF capture
    /// retains this batch beyond the executor lifetime. The ordinary store
    /// path leaves the charge intact because the executor is live while that
    /// one batch is packed and retired.
    pub(crate) fn detach_shared_derived_input_charge_v3(
        &mut self,
        shared_derived_input_bytes: usize,
    ) -> Result<(), ResidentClassicTaExecutorErrorV3> {
        self.retained_device_bytes = self
            .retained_device_bytes
            .checked_sub(shared_derived_input_bytes)
            .ok_or(ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA detached HTF retained bytes",
            ))?;
        Ok(())
    }

    fn launch_validity(
        launch: &ResidentClassicTaLaunchRecipeV3,
        bindings: Vec<ResidentFeatureColumnBindingV3>,
        output_owner: ResidentClassicTaOutputOwnerV3,
        context: &Arc<Context>,
        stream: &Arc<Stream>,
        device_ordinal: u32,
        rows: usize,
    ) -> Result<Self, ResidentClassicTaExecutorErrorV3> {
        if bindings.len() != launch.outputs().len() {
            return Err(ResidentClassicTaExecutorErrorV3::AdmittedGlobalBindingsMismatch);
        }
        let columns = bindings.len();
        let value_addresses_host = (0..columns)
            .map(|column| output_owner.value_buffer(column).as_device_ptr().as_raw())
            .collect::<Vec<_>>();
        let value_offsets_host = vec![0_u64; columns];
        let all_nan_validity_codes_host = vec![launch.all_nan_validity_code(); columns];
        let value_addresses =
            ResidentClassicTaPinnedCopyV3::copy_async(&value_addresses_host, context, stream)?;
        let value_offsets =
            ResidentClassicTaPinnedCopyV3::copy_async(&value_offsets_host, context, stream)?;
        let all_nan_validity_codes = ResidentClassicTaPinnedCopyV3::copy_async(
            &all_nan_validity_codes_host,
            context,
            stream,
        )?;
        let first_finite_rows = ResidentClassicTaDeviceBufferV3::<u64>::uninitialized_async(
            columns,
            Arc::clone(context),
            Arc::clone(stream),
        )?;
        let validity_cells = rows.checked_mul(columns).ok_or(
            ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA validity cells",
            ),
        )?;
        let validity_u8 = ResidentClassicTaDeviceBufferV3::<u8>::uninitialized_async(
            validity_cells,
            Arc::clone(context),
            Arc::clone(stream),
        )?;
        let validity_device_error = ResidentClassicTaDeviceBufferV3::<u32>::uninitialized_async(
            1,
            Arc::clone(context),
            Arc::clone(stream),
        )?;
        let status = unsafe {
            // SAFETY: all compact tables and destinations have the checked
            // exact extents, remain owned below, and the one carried stream
            // orders their async copies before this launch.
            neoethos_resident_classic_validity_u8_v3(
                value_addresses.device().as_device_ptr().as_ptr(),
                value_offsets.device().as_device_ptr().as_ptr(),
                all_nan_validity_codes.device().as_device_ptr().as_ptr(),
                rows,
                columns,
                first_finite_rows.as_device_ptr().as_mut_ptr(),
                validity_u8.as_device_ptr().as_mut_ptr(),
                validity_device_error.as_device_ptr().as_mut_ptr(),
                stream.as_inner(),
            )
        };
        if status != 0 {
            return Err(ResidentClassicTaExecutorErrorV3::Native {
                operation: "neoethos_resident_classic_validity_u8_v3",
                status,
            });
        }
        let producer_ready_event =
            ResidentProducerReadyEventV3::record(context, stream, device_ordinal)?;
        let retained_device_bytes = output_owner
            .retained_output_bytes()
            .checked_add(validity_cells)
            .ok_or(ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA output and validity bytes",
            ))?;
        let retained_scratch_bytes = output_owner
            .retained_launch_scratch_bytes()
            .checked_add(value_addresses.retained_device_bytes())
            .and_then(|bytes| bytes.checked_add(value_offsets.retained_device_bytes()))
            .and_then(|bytes| bytes.checked_add(all_nan_validity_codes.retained_device_bytes()))
            .and_then(|bytes| bytes.checked_add(columns.saturating_mul(std::mem::size_of::<u64>())))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u32>()))
            .ok_or(ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA launch scratch bytes",
            ))?;
        Ok(Self {
            bindings,
            output_owner: Some(output_owner),
            value_addresses: Some(value_addresses),
            value_offsets: Some(value_offsets),
            all_nan_validity_codes: Some(all_nan_validity_codes),
            first_finite_rows: Some(first_finite_rows),
            validity_u8: Some(validity_u8),
            validity_device_error: Some(validity_device_error),
            producer_ready_event,
            context: Arc::clone(context),
            stream: Arc::clone(stream),
            rows,
            device_ordinal,
            retained_device_bytes,
            retained_scratch_bytes,
        })
    }
}

unsafe impl ResidentF64FeatureBatchV3 for PendingResidentClassicTaBatchV3 {
    fn column_bindings(&self) -> &[ResidentFeatureColumnBindingV3] {
        &self.bindings
    }

    fn value_buffer(&self, column: usize) -> &DeviceBuffer<f64> {
        self.output_owner
            .as_ref()
            .expect("live Classic TA batch retains its output owner")
            .value_buffer(column)
    }

    fn validity_buffer(&self, _column: usize) -> &DeviceBuffer<u8> {
        self.validity_u8
            .as_ref()
            .expect("live Classic TA batch retains exact validity")
    }

    fn value_offset(&self, _column: usize) -> usize {
        0
    }

    fn validity_offset(&self, column: usize) -> usize {
        column * self.rows
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn device_ordinal(&self) -> u32 {
        self.device_ordinal
    }

    fn producer_context(&self) -> &Context {
        self.context.as_ref()
    }

    fn producer_stream(&self) -> &Stream {
        self.stream.as_ref()
    }

    fn producer_ready_event(&self) -> &ResidentProducerReadyEventV3 {
        &self.producer_ready_event
    }

    fn retained_device_bytes(&self) -> usize {
        self.retained_device_bytes
    }

    fn retained_scratch_bytes(&self) -> usize {
        self.retained_scratch_bytes
    }

    fn enqueue_nonblocking_release(
        mut self: Box<Self>,
        release_stream: &Stream,
    ) -> Result<(), ResidentFeatureStoreCudaErrorV3> {
        if release_stream.as_inner().is_null()
            || release_stream.as_inner() != self.stream.as_inner()
            || !self
                .first_finite_rows
                .as_ref()
                .is_some_and(|buffer| buffer.is_owned_by_stream(release_stream))
            || !self
                .validity_u8
                .as_ref()
                .is_some_and(|buffer| buffer.is_owned_by_stream(release_stream))
            || !self
                .validity_device_error
                .as_ref()
                .is_some_and(|buffer| buffer.is_owned_by_stream(release_stream))
        {
            std::mem::forget(self);
            return Err(ResidentFeatureStoreCudaErrorV3::ProducerStreamMismatch);
        }
        if let Some(output_owner) = self.output_owner.take() {
            output_owner.enqueue_release(release_stream)?;
        }
        if let Some(copy) = self.value_addresses.take() {
            copy.enqueue_release(release_stream)?;
        }
        if let Some(copy) = self.value_offsets.take() {
            copy.enqueue_release(release_stream)?;
        }
        if let Some(copy) = self.all_nan_validity_codes.take() {
            copy.enqueue_release(release_stream)?;
        }
        drop(self.first_finite_rows.take());
        drop(self.validity_u8.take());
        drop(self.validity_device_error.take());
        Ok(())
    }
}

/// gpu-cuda-owned Classic TA execution authority over the exact carried
/// primary context, non-default run stream and parent allocations.
///
/// This type is crate-private on purpose. Data can pass a validated recipe to
/// the assembler, but cannot obtain the carrier, parent pointers or vector-ta
/// session. The eventual assembler entrypoint consumes one recipe launch at a
/// time into `PendingResidentClassicTaBatchV3` and retires it before the next
/// producer batch is admitted.
pub(crate) struct ResidentClassicTaExecutorV3 {
    recipe: ResidentClassicTaRecipeV3,
    admitted_global_bindings: Vec<Option<ResidentFeatureColumnBindingV3>>,
    pre_device_memory_receipt_v4: Option<ResidentClassicTaPreDeviceMemoryReceiptV4>,
    parent_views: ResidentClassicTaParentViewsV3,
    derived: ResidentClassicTaDerivedInputsV3,
    engine: CudaF64Indicators,
    context: Arc<Context>,
    stream: Arc<Stream>,
    device_identity:
        neoethos_gpu_contracts::resident_feature_store_v3::CudaPrimaryContextBuildIdentityV3,
    device_ordinal: u32,
    next_launch_index: usize,
}

impl ResidentClassicTaExecutorV3 {
    /// Device-fixture constructor retained for the isolated VectorTA parity
    /// harness. Production materialization must use [`Self::new_v4`] because a
    /// local zero-based recipe is not global schema authority.
    #[cfg(feature = "cuda-device-fixtures")]
    pub(crate) fn new(
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        parent: &dyn ResidentParentDatasetSourceV3,
        recipe: ResidentClassicTaRecipeV3,
    ) -> Result<Self, ResidentClassicTaExecutorErrorV3> {
        let recipe = preflight_resident_classic_ta_recipe_v3(recipe)?;
        let mut fixture_local_bindings = Vec::with_capacity(recipe.output_count());
        for launch in recipe.launches() {
            for route in launch.outputs() {
                fixture_local_bindings.push(ResidentFeatureColumnBindingV3 {
                    ordinal: route.destination_column(),
                    feature_name: route.feature_name().to_owned(),
                    canonical_parameter_tuple_sha256: route.canonical_parameter_tuple_sha256(),
                    route_receipt_sha256: route.route_receipt_sha256(),
                });
            }
        }
        Self::new_with_bound_preflight_v4(run_device, parent, recipe, fixture_local_bindings, None)
    }

    pub(crate) fn new_v4(
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        parent: &dyn ResidentParentDatasetSourceV3,
        recipe: ResidentClassicTaRecipeV3,
        admitted_global_bindings: Vec<ResidentFeatureColumnBindingV3>,
        pre_device_memory_receipt_v4: ResidentClassicTaPreDeviceMemoryReceiptV4,
    ) -> Result<Self, ResidentClassicTaExecutorErrorV3> {
        let recipe = preflight_resident_classic_ta_recipe_v3(recipe)?;
        validate_admitted_global_bindings_v4(&recipe, &admitted_global_bindings)?;
        let runtime_memory_receipt_v4 = preflight_resident_classic_ta_memory_v4(&recipe)?;
        if runtime_memory_receipt_v4 != pre_device_memory_receipt_v4 {
            return Err(
                ResidentClassicTaExecutorErrorV3::PreDeviceMemoryReceiptMismatch {
                    indicator_id: "complete Classic primary/warmup recipe".to_owned(),
                },
            );
        }
        Self::new_with_bound_preflight_v4(
            run_device,
            parent,
            recipe,
            admitted_global_bindings,
            Some(pre_device_memory_receipt_v4),
        )
    }

    fn new_with_bound_preflight_v4(
        run_device: &GpuOnlyRunDeviceAdmissionV3,
        parent: &dyn ResidentParentDatasetSourceV3,
        recipe: ResidentClassicTaRecipeV3,
        admitted_global_bindings: Vec<ResidentFeatureColumnBindingV3>,
        pre_device_memory_receipt_v4: Option<ResidentClassicTaPreDeviceMemoryReceiptV4>,
    ) -> Result<Self, ResidentClassicTaExecutorErrorV3> {
        let context = Arc::clone(run_device.primary_context_for_resident_producer_v3());
        let stream = Arc::clone(run_device.run_stream_for_resident_producer_v3());
        let device_ordinal = run_device.device_identity().ordinal();
        CurrentContext::set_current(context.as_ref())?;
        if stream.as_inner().is_null()
            || parent.rows() != recipe.rows()
            || parent.device_ordinal() != device_ordinal
            || parent.producer_context().as_raw() != context.as_raw()
            || parent.producer_stream().as_inner() != stream.as_inner()
        {
            return Err(ResidentClassicTaExecutorErrorV3::ParentAuthorityMismatch);
        }
        let derived =
            ResidentClassicTaDerivedInputsV3::launch(parent, &context, &stream, device_ordinal)?;
        if pre_device_memory_receipt_v4
            .as_ref()
            .is_some_and(|receipt| {
                receipt.derived_input_bytes() != derived.retained_device_bytes
                    || receipt.derived_ready_event_count() != 1
            })
        {
            return Err(
                ResidentClassicTaExecutorErrorV3::PreDeviceMemoryReceiptMismatch {
                    indicator_id: "retained Classic derived inputs".to_owned(),
                },
            );
        }
        derived.producer_ready_event.wait_before_read(
            context.as_ref(),
            stream.as_ref(),
            device_ordinal,
        )?;
        let parent_views = ResidentClassicTaParentViewsV3::new(parent, &derived)?;
        let session = Arc::new(CudaSession::from_parts(
            Arc::clone(&context),
            Arc::clone(&stream),
            device_ordinal,
        ));
        let engine = CudaF64Indicators::from_session(session)?;
        Ok(Self {
            recipe,
            admitted_global_bindings: admitted_global_bindings.into_iter().map(Some).collect(),
            pre_device_memory_receipt_v4,
            parent_views,
            derived,
            engine,
            context,
            stream,
            device_identity: run_device.device_identity().clone(),
            device_ordinal,
            next_launch_index: 0,
        })
    }

    fn prepared_launch(
        &self,
    ) -> Result<Option<ResidentClassicTaPreparedLaunchV3<'_>>, ResidentClassicTaExecutorErrorV3>
    {
        let Some(recipe) = self.recipe.launches().get(self.next_launch_index) else {
            return Ok(None);
        };
        let first_valid = match recipe.first_valid_rule() {
            ResidentClassicTaFirstValidRuleV3::CloseReturnPair => 1,
            // The sealed resident parent rejects every non-finite OHLCV cell
            // before this executor is created. Every named first-index scan is
            // therefore zero; methods requiring a finite-count/run receive
            // the exact full row count in the typed dispatcher.
            ResidentClassicTaFirstValidRuleV3::NamedRouteOwned => 0,
            ResidentClassicTaFirstValidRuleV3::AllInputsNonNan
            | ResidentClassicTaFirstValidRuleV3::AllInputsFinite
            | ResidentClassicTaFirstValidRuleV3::PriceVolumeFinite
            | ResidentClassicTaFirstValidRuleV3::HighLowFinitePositive
            | ResidentClassicTaFirstValidRuleV3::NotApplicable => 0,
        };
        if first_valid >= self.recipe.rows() {
            return Err(ResidentClassicTaExecutorErrorV3::InvalidInput(format!(
                "Classic TA first_valid={first_valid} is outside {} rows",
                self.recipe.rows()
            )));
        }
        Ok(Some(ResidentClassicTaPreparedLaunchV3 {
            recipe,
            inputs: self.parent_views.inputs(recipe.input()),
            first_valid,
        }))
    }

    pub(crate) fn retained_derived_input_bytes(&self) -> usize {
        self.derived.retained_device_bytes
    }

    pub(crate) fn next_pending_batch_v3(
        &mut self,
    ) -> Result<Option<PendingResidentClassicTaBatchV3>, ResidentClassicTaExecutorErrorV3> {
        let Some(launch_for_bindings) = self.recipe.launches().get(self.next_launch_index) else {
            return Ok(None);
        };
        let binding_start = launch_for_bindings.first_destination_column();
        let binding_end = launch_for_bindings.next_destination_column();
        let mut bindings = Vec::with_capacity(binding_end.saturating_sub(binding_start));
        for binding in self
            .admitted_global_bindings
            .get_mut(binding_start..binding_end)
            .ok_or(ResidentClassicTaExecutorErrorV3::AdmittedGlobalBindingsMismatch)?
        {
            bindings.push(
                binding
                    .take()
                    .ok_or(ResidentClassicTaExecutorErrorV3::AdmittedGlobalBindingsMismatch)?,
            );
        }
        let Some(prepared) = self.prepared_launch()? else {
            return Ok(None);
        };
        let launch = prepared.recipe;
        let launch_memory_plan_v4 = self
            .pre_device_memory_receipt_v4
            .as_ref()
            .and_then(|receipt| receipt.launch_plans().get(self.next_launch_index));
        let output_owner = if self.launch_is_warmup_only(launch) {
            if launch_memory_plan_v4.is_some_and(|plan| {
                !matches!(plan, ResidentClassicTaLaunchMemoryPlanV4::Warmup { .. })
            }) {
                return Err(
                    ResidentClassicTaExecutorErrorV3::PreDeviceMemoryReceiptMismatch {
                        indicator_id: launch.indicator_id().to_owned(),
                    },
                );
            }
            self.launch_warmup_outputs(launch)?
        } else {
            let primary = f64_kernel_for(launch.indicator_id()).filter(|spec| {
                launch.first_valid_rule() != ResidentClassicTaFirstValidRuleV3::NamedRouteOwned
                    && spec.kernel.entry_point() == launch.entry_point()
                    && launch.outputs().len() == 1
                    && spec.primary_output_id() == Some(launch.outputs()[0].output_id())
            });
            match primary {
                Some(spec) => self.launch_primary(
                    prepared,
                    spec,
                    launch_memory_plan_v4
                        .and_then(ResidentClassicTaLaunchMemoryPlanV4::vector_plan),
                )?,
                None if launch_memory_plan_v4.is_none() => self.launch_named(prepared)?,
                None => {
                    return Err(
                        ResidentClassicTaExecutorErrorV3::PreDeviceMemoryReceiptMismatch {
                            indicator_id: launch.indicator_id().to_owned(),
                        },
                    );
                }
            }
        };
        if launch_memory_plan_v4.is_some_and(|plan| {
            output_owner.retained_output_bytes() != plan.all_output_retained_bytes()
                || plan.selected_value_bytes() != plan.all_output_retained_bytes()
                || output_owner
                    .retained_launch_scratch_bytes()
                    .checked_add(plan.validity_scratch_bytes())
                    != Some(plan.retained_scratch_bytes())
                || plan.ready_event_count() != 1
        }) {
            return Err(
                ResidentClassicTaExecutorErrorV3::PreDeviceMemoryReceiptMismatch {
                    indicator_id: launch.indicator_id().to_owned(),
                },
            );
        }
        let mut batch = PendingResidentClassicTaBatchV3::launch_validity(
            launch,
            bindings,
            output_owner,
            &self.context,
            &self.stream,
            self.device_ordinal,
            self.recipe.rows(),
        )?;
        batch.retained_device_bytes = batch
            .retained_device_bytes
            .checked_add(self.retained_derived_input_bytes())
            .ok_or(ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA batch plus retained derived inputs",
            ))?;
        if launch_memory_plan_v4.is_some_and(|plan| {
            let expected_retained_device_bytes = plan
                .all_output_retained_bytes()
                .checked_add(plan.validity_bytes())
                .and_then(|bytes| bytes.checked_add(plan.additional_retained_bytes()));
            expected_retained_device_bytes != Some(batch.retained_device_bytes)
                || batch.retained_scratch_bytes != plan.retained_scratch_bytes()
        }) {
            return Err(
                ResidentClassicTaExecutorErrorV3::PreDeviceMemoryReceiptMismatch {
                    indicator_id: launch.indicator_id().to_owned(),
                },
            );
        }
        self.next_launch_index = self.next_launch_index.checked_add(1).ok_or(
            ResidentClassicTaExecutorErrorV3::ArithmeticOverflow(
                "resident Classic TA launch cursor",
            ),
        )?;
        Ok(Some(batch))
    }

    fn launch_is_warmup_only(&self, launch: &ResidentClassicTaLaunchRecipeV3) -> bool {
        launch
            .outputs()
            .first()
            .and_then(ResidentClassicTaOutputRouteV3::swept_period)
            .is_some_and(|period| (period as f64) * 1.25 >= self.recipe.rows() as f64)
    }

    fn launch_warmup_outputs(
        &self,
        launch: &ResidentClassicTaLaunchRecipeV3,
    ) -> Result<ResidentClassicTaOutputOwnerV3, ResidentClassicTaExecutorErrorV3> {
        let mut outputs = Vec::with_capacity(launch.outputs().len());
        for _ in launch.outputs() {
            let output = ResidentClassicTaDeviceBufferV3::<f64>::uninitialized_async(
                self.recipe.rows(),
                Arc::clone(&self.context),
                Arc::clone(&self.stream),
            )?;
            let status = unsafe {
                // SAFETY: the destination owns exactly recipe.rows() cells and
                // the one carried non-default stream orders initialization
                // before validity classification and packing.
                neoethos_resident_classic_fill_nan_f64_v3(
                    output.as_device_ptr().as_mut_ptr(),
                    self.recipe.rows(),
                    self.stream.as_inner(),
                )
            };
            if status != 0 {
                return Err(ResidentClassicTaExecutorErrorV3::Native {
                    operation: "neoethos_resident_classic_fill_nan_f64_v3",
                    status,
                });
            }
            outputs.push(output);
        }
        Ok(ResidentClassicTaOutputOwnerV3::Warmup(outputs))
    }

    fn launch_primary(
        &self,
        prepared: ResidentClassicTaPreparedLaunchV3<'_>,
        spec: &vector_ta::indicators::dispatch::F64KernelSpec,
        preallocation_plan_v4: Option<&F64ResidentSingleSweepAllocationPlanV4>,
    ) -> Result<ResidentClassicTaOutputOwnerV3, ResidentClassicTaExecutorErrorV3> {
        require_input_kind_v3(prepared.recipe.input(), spec.input)?;
        require_parameter_keys_v3(prepared.recipe, &["cuda_period"])?;
        let period = require_usize_parameter_v3(prepared.recipe, "cuda_period")?;
        let period = i32::try_from(period).map_err(|_| {
            ResidentClassicTaExecutorErrorV3::InvalidInput(format!(
                "{}.cuda_period exceeds the CUDA i32 ABI",
                prepared.recipe.indicator_id()
            ))
        })?;
        let output = match preallocation_plan_v4 {
            Some(preallocation_plan_v4) => self.engine.sweep_resident_preplanned_v4(
                spec.kernel,
                prepared.inputs,
                &[period],
                prepared.first_valid,
                preallocation_plan_v4,
            )?,
            None => self.engine.sweep_resident_v3(
                spec.kernel,
                prepared.inputs,
                &[period],
                prepared.first_valid,
            )?,
        };
        let output_indices =
            self.validate_observed_route_v3(prepared.recipe, &output.route_manifest_v3())?;
        if output_indices != [0] {
            return Err(ResidentClassicTaExecutorErrorV3::ObservedRouteMismatch {
                indicator_id: prepared.recipe.indicator_id().to_owned(),
            });
        }
        Ok(ResidentClassicTaOutputOwnerV3::Primary(output))
    }

    fn launch_named(
        &self,
        prepared: ResidentClassicTaPreparedLaunchV3<'_>,
    ) -> Result<ResidentClassicTaOutputOwnerV3, ResidentClassicTaExecutorErrorV3> {
        let output = self.launch_named_result_v3(prepared)?;
        let parts = output.into_resident_parts_v3();
        let output_indices =
            self.validate_observed_route_v3(prepared.recipe, &parts.route_manifest_v3())?;
        Ok(ResidentClassicTaOutputOwnerV3::Named {
            parts,
            output_indices,
        })
    }

    fn launch_named_result_v3(
        &self,
        prepared: ResidentClassicTaPreparedLaunchV3<'_>,
    ) -> Result<vector_ta::cuda::F64NamedOutputsResult, ResidentClassicTaExecutorErrorV3> {
        let launch = prepared.recipe;
        let views = self.parent_views;
        let rows = self.recipe.rows();
        let result = match launch.indicator_id() {
            "absolute_strength_index_oscillator" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["ema_length", "signal_length"])?;
                self.engine.absolute_strength_index_oscillator_all_outputs(
                    views.close,
                    &[(
                        (require_usize_parameter_v3(launch, "ema_length")?),
                        require_usize_parameter_v3(launch, "signal_length")?,
                    )],
                )?
            }
            "adaptive_bandpass_trigger_oscillator" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["delta", "alpha"])?;
                self.engine
                    .adaptive_bandpass_trigger_oscillator_all_outputs(
                        views.close,
                        rows,
                        &[(
                            (require_f64_parameter_v3(launch, "delta")?),
                            require_f64_parameter_v3(launch, "alpha")?,
                        )],
                    )?
            }
            "adaptive_bounds_rsi" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["rsi_length", "alpha"])?;
                let parameters =
                    vector_ta::indicators::adaptive_bounds_rsi::AdaptiveBoundsRsiParams {
                        rsi_length: Some(require_usize_parameter_v3(launch, "rsi_length")?),
                        alpha: Some(require_f64_parameter_v3(launch, "alpha")?),
                    };
                self.engine
                    .adaptive_bounds_rsi_all_outputs(views.close, rows, &[parameters])?
            }
            "adaptive_macd" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(
                    launch,
                    &["length", "fast_period", "slow_period", "signal_period"],
                )?;
                self.engine.adaptive_macd_all_outputs(
                    views.close,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "length")?),
                        require_usize_parameter_v3(launch, "fast_period")?,
                        require_usize_parameter_v3(launch, "slow_period")?,
                        require_usize_parameter_v3(launch, "signal_period")?,
                    )],
                )?
            }
            "adaptive_momentum_oscillator" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["length", "smoothing_length"])?;
                self.engine.adaptive_momentum_oscillator_all_outputs(
                    views.close,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "length")?),
                        require_usize_parameter_v3(launch, "smoothing_length")?,
                    )],
                )?
            }
            "adaptive_schaff_trend_cycle" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Hlc)?;
                require_parameter_keys_v3(
                    launch,
                    &[
                        "adaptive_length",
                        "stc_length",
                        "smoothing_factor",
                        "fast_length",
                        "slow_length",
                    ],
                )?;
                let parameters = vector_ta::indicators::adaptive_schaff_trend_cycle::AdaptiveSchaffTrendCycleParams {
                    adaptive_length: Some(require_usize_parameter_v3(launch, "adaptive_length")?),
                    stc_length: Some(require_usize_parameter_v3(launch, "stc_length")?),
                    smoothing_factor: Some(require_f64_parameter_v3(launch, "smoothing_factor")?),
                    fast_length: Some(require_usize_parameter_v3(launch, "fast_length")?),
                    slow_length: Some(require_usize_parameter_v3(launch, "slow_length")?),
                };
                self.engine.adaptive_schaff_trend_cycle_all_outputs(
                    views.ohlcv()?,
                    rows,
                    &[parameters],
                )?
            }
            "adjustable_ma_alternating_extremities" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["length", "mult", "alpha", "beta"])?;
                let parameters = vector_ta::indicators::adjustable_ma_alternating_extremities::AdjustableMaAlternatingExtremitiesParams {
                    length: Some(require_usize_parameter_v3(launch, "length")?),
                    mult: Some(require_f64_parameter_v3(launch, "mult")?),
                    alpha: Some(require_f64_parameter_v3(launch, "alpha")?),
                    beta: Some(require_f64_parameter_v3(launch, "beta")?),
                };
                self.engine
                    .adjustable_ma_alternating_extremities_all_outputs(
                        views.ohlcv()?,
                        rows,
                        &[parameters],
                    )?
            }
            "alligator" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Hl2)?;
                require_parameter_keys_v3(
                    launch,
                    &[
                        "jaw_period",
                        "jaw_offset",
                        "teeth_period",
                        "teeth_offset",
                        "lips_period",
                        "lips_offset",
                    ],
                )?;
                let parameters = vector_ta::indicators::alligator::AlligatorParams {
                    jaw_period: Some(require_usize_parameter_v3(launch, "jaw_period")?),
                    jaw_offset: Some(require_usize_parameter_v3(launch, "jaw_offset")?),
                    teeth_period: Some(require_usize_parameter_v3(launch, "teeth_period")?),
                    teeth_offset: Some(require_usize_parameter_v3(launch, "teeth_offset")?),
                    lips_period: Some(require_usize_parameter_v3(launch, "lips_period")?),
                    lips_offset: Some(require_usize_parameter_v3(launch, "lips_offset")?),
                };
                self.engine
                    .alligator_all_outputs(views.hl2, prepared.first_valid, &[parameters])?
            }
            "alphatrend" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["coeff", "period", "no_volume"])?;
                let parameters = vector_ta::indicators::alphatrend::AlphaTrendParams {
                    coeff: Some(require_f64_parameter_v3(launch, "coeff")?),
                    period: Some(require_usize_parameter_v3(launch, "period")?),
                    no_volume: Some(require_bool_parameter_v3(launch, "no_volume")?),
                };
                self.engine.alphatrend_all_outputs(
                    views.ohlcv()?,
                    prepared.first_valid,
                    &[parameters],
                )?
            }
            "acosc" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &[])?;
                self.engine.acosc_all_outputs(views.ohlcv()?)?
            }
            "andean_oscillator" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["length", "signal_length"])?;
                self.engine.andean_oscillator_all_outputs(
                    views.ohlcv()?,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "length")?),
                        require_usize_parameter_v3(launch, "signal_length")?,
                    )],
                )?
            }
            "aroon" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["length"])?;
                self.engine.aroon_all_outputs(
                    views.ohlcv()?,
                    &[require_usize_parameter_v3(launch, "length")?],
                )?
            }
            "aso" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["period", "mode"])?;
                self.engine.aso_all_outputs(
                    views.ohlcv()?,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "period")?),
                        require_usize_parameter_v3(launch, "mode")?,
                    )],
                )?
            }
            "autocorrelation_indicator" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["length", "lag", "use_test_signal"])?;
                self.engine.autocorrelation_indicator_all_outputs(
                    views.close,
                    rows,
                    &[(
                        (require_usize_parameter_v3(launch, "length")?),
                        require_usize_parameter_v3(launch, "lag")?,
                        require_bool_parameter_v3(launch, "use_test_signal")?,
                    )],
                )?
            }
            "avsl" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["fast_period", "slow_period", "multiplier"])?;
                self.engine.avsl_production_output(
                    views.close,
                    views.low,
                    views.volume,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "fast_period")?),
                        require_usize_parameter_v3(launch, "slow_period")?,
                        require_f64_parameter_v3(launch, "multiplier")?,
                    )],
                )?
            }
            "bandpass" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["period", "bandwidth"])?;
                self.engine.bandpass_all_outputs(
                    views.close,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "period")?),
                        require_f64_parameter_v3(launch, "bandwidth")?,
                    )],
                )?
            }
            "bollinger_bands" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["period", "devup", "devdn"])?;
                self.engine.bollinger_bands_all_outputs(
                    views.close,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "period")?),
                        require_f64_parameter_v3(launch, "devup")?,
                        require_f64_parameter_v3(launch, "devdn")?,
                    )],
                )?
            }
            "buff_averages" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::CloseVolume)?;
                require_parameter_keys_v3(launch, &["fast_period", "slow_period"])?;
                self.engine.buff_averages_all_outputs(
                    views.close,
                    views.volume,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "fast_period")?),
                        require_usize_parameter_v3(launch, "slow_period")?,
                    )],
                )?
            }
            "candle_strength_oscillator" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(
                    launch,
                    &["period", "atr_enabled", "atr_length", "mode"],
                )?;
                let parameters = vector_ta::indicators::candle_strength_oscillator::CandleStrengthOscillatorParams {
                    period: Some(require_usize_parameter_v3(launch, "period")?),
                    atr_enabled: Some(require_bool_parameter_v3(launch, "atr_enabled")?),
                    atr_length: Some(require_usize_parameter_v3(launch, "atr_length")?),
                    mode: Some(require_text_parameter_v3(launch, "mode")?.to_owned()),
                };
                self.engine.candle_strength_oscillator_all_outputs(
                    views.ohlcv()?,
                    rows,
                    &[parameters],
                )?
            }
            "chandelier_exit" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["period", "mult", "use_close"])?;
                let parameters = vector_ta::indicators::chandelier_exit::ChandelierExitParams {
                    period: Some(require_usize_parameter_v3(launch, "period")?),
                    mult: Some(require_f64_parameter_v3(launch, "mult")?),
                    use_close: Some(require_bool_parameter_v3(launch, "use_close")?),
                };
                self.engine.chandelier_exit_all_outputs(
                    views.ohlcv()?,
                    prepared.first_valid,
                    prepared.first_valid,
                    &[parameters],
                )?
            }
            "cksp" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["p", "x", "q"])?;
                let parameters = vector_ta::indicators::cksp::CkspParams {
                    p: Some(require_usize_parameter_v3(launch, "p")?),
                    x: Some(require_f64_parameter_v3(launch, "x")?),
                    q: Some(require_usize_parameter_v3(launch, "q")?),
                };
                self.engine
                    .cksp_all_outputs(views.ohlcv()?, prepared.first_valid, &[parameters])?
            }
            "coppock" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(
                    launch,
                    &["short_roc_period", "long_roc_period", "ma_period"],
                )?;
                let parameters = vector_ta::indicators::coppock::CoppockParams {
                    short_roc_period: Some(require_usize_parameter_v3(launch, "short_roc_period")?),
                    long_roc_period: Some(require_usize_parameter_v3(launch, "long_roc_period")?),
                    ma_period: Some(require_usize_parameter_v3(launch, "ma_period")?),
                    ma_type: Some("wma".to_owned()),
                };
                self.engine.coppock_production_output(
                    views.close,
                    prepared.first_valid,
                    &[parameters],
                )?
            }
            "correlation_cycle" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["period", "threshold"])?;
                self.engine.correlation_cycle_all_outputs(
                    views.close,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "period")?),
                        require_f64_parameter_v3(launch, "threshold")?,
                    )],
                )?
            }
            "cvi" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::HighLow)?;
                require_parameter_keys_v3(launch, &["period"])?;
                self.engine.cvi_production_output(
                    views.high_low()?,
                    prepared.first_valid,
                    &[require_usize_parameter_v3(launch, "period")?],
                )?
            }
            "cyberpunk_value_trend_analyzer" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["entry_level", "exit_level"])?;
                self.engine.cyberpunk_value_trend_analyzer_all_outputs(
                    views.ohlcv()?,
                    rows,
                    &[(
                        (require_usize_parameter_v3(launch, "entry_level")?),
                        require_usize_parameter_v3(launch, "exit_level")?,
                    )],
                )?
            }
            "cycle_channel_oscillator" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(
                    launch,
                    &[
                        "short_cycle_length",
                        "medium_cycle_length",
                        "short_multiplier",
                        "medium_multiplier",
                    ],
                )?;
                self.engine.cycle_channel_oscillator_all_outputs(
                    views.ohlcv()?,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "short_cycle_length")?),
                        require_usize_parameter_v3(launch, "medium_cycle_length")?,
                        require_f64_parameter_v3(launch, "short_multiplier")?,
                        require_f64_parameter_v3(launch, "medium_multiplier")?,
                    )],
                )?
            }
            "daily_factor" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["threshold_level"])?;
                self.engine.daily_factor_all_outputs(
                    views.ohlcv()?,
                    prepared.first_valid,
                    &[require_f64_parameter_v3(launch, "threshold_level")?],
                )?
            }
            "damiani_volatmeter" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(
                    launch,
                    &["vis_atr", "vis_std", "sed_atr", "sed_std", "threshold"],
                )?;
                self.engine.damiani_volatmeter_all_outputs(
                    views.close,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "vis_atr")?),
                        require_usize_parameter_v3(launch, "vis_std")?,
                        require_usize_parameter_v3(launch, "sed_atr")?,
                        require_usize_parameter_v3(launch, "sed_std")?,
                        require_f64_parameter_v3(launch, "threshold")?,
                    )],
                )?
            }
            "di" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["period"])?;
                self.engine.di_all_outputs(
                    views.ohlcv()?,
                    prepared.first_valid,
                    &[require_usize_parameter_v3(launch, "period")?],
                )?
            }
            "didi_index" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(
                    launch,
                    &["short_length", "medium_length", "long_length"],
                )?;
                self.engine.didi_index_all_outputs(
                    views.close,
                    rows,
                    &[(
                        (require_usize_parameter_v3(launch, "short_length")?),
                        require_usize_parameter_v3(launch, "medium_length")?,
                        require_usize_parameter_v3(launch, "long_length")?,
                    )],
                )?
            }
            "directional_imbalance_index" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["length", "period"])?;
                self.engine.directional_imbalance_index_all_outputs(
                    views.ohlcv()?,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "length")?),
                        require_usize_parameter_v3(launch, "period")?,
                    )],
                )?
            }
            "disparity_index" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(
                    launch,
                    &[
                        "ema_period",
                        "lookback_period",
                        "smoothing_period",
                        "smoothing_is_sma",
                    ],
                )?;
                self.engine.disparity_index_production_output(
                    views.close,
                    rows,
                    &[(
                        (require_usize_parameter_v3(launch, "ema_period")?),
                        require_usize_parameter_v3(launch, "lookback_period")?,
                        require_usize_parameter_v3(launch, "smoothing_period")?,
                        require_bool_parameter_v3(launch, "smoothing_is_sma")?,
                    )],
                )?
            }
            "dm" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::HighLow)?;
                require_parameter_keys_v3(launch, &["period"])?;
                self.engine.dm_all_outputs(
                    views.high_low()?,
                    prepared.first_valid,
                    &[require_usize_parameter_v3(launch, "period")?],
                )?
            }
            "donchian" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::HighLow)?;
                require_parameter_keys_v3(launch, &["period"])?;
                self.engine.donchian_all_outputs(
                    views.high_low()?,
                    prepared.first_valid,
                    &[require_usize_parameter_v3(launch, "period")?],
                )?
            }
            "dual_ulcer_index" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["period", "auto_threshold", "threshold"])?;
                self.engine.dual_ulcer_index_all_outputs(
                    views.close,
                    rows,
                    &[(
                        (require_usize_parameter_v3(launch, "period")?),
                        require_bool_parameter_v3(launch, "auto_threshold")?,
                        require_f64_parameter_v3(launch, "threshold")?,
                    )],
                )?
            }
            "dvdiqqe" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(
                    launch,
                    &[
                        "period",
                        "smoothing_period",
                        "fast_multiplier",
                        "slow_multiplier",
                        "use_tick_only",
                        "dynamic_center",
                        "tick_size",
                    ],
                )?;
                self.engine.dvdiqqe_all_outputs(
                    views.ohlcv()?,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "period")?),
                        require_usize_parameter_v3(launch, "smoothing_period")?,
                        require_f64_parameter_v3(launch, "fast_multiplier")?,
                        require_f64_parameter_v3(launch, "slow_multiplier")?,
                        require_bool_parameter_v3(launch, "use_tick_only")?,
                        require_bool_parameter_v3(launch, "dynamic_center")?,
                        require_f64_parameter_v3(launch, "tick_size")?,
                    )],
                )?
            }
            "ehlers_autocorrelation_periodogram" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(
                    launch,
                    &["min_period", "max_period", "avg_length", "enhance"],
                )?;
                self.engine.ehlers_autocorrelation_periodogram_all_outputs(
                    views.close,
                    &[(
                        (require_usize_parameter_v3(launch, "min_period")?),
                        require_usize_parameter_v3(launch, "max_period")?,
                        require_usize_parameter_v3(launch, "avg_length")?,
                        require_bool_parameter_v3(launch, "enhance")?,
                    )],
                )?
            }
            "ehlers_linear_extrapolation_predictor" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(
                    launch,
                    &[
                        "high_pass_length",
                        "low_pass_length",
                        "gain",
                        "bars_forward",
                        "signal_mode",
                    ],
                )?;
                self.engine
                    .ehlers_linear_extrapolation_predictor_all_outputs(
                        views.close,
                        rows,
                        &[(
                            (require_usize_parameter_v3(launch, "high_pass_length")?),
                            require_usize_parameter_v3(launch, "low_pass_length")?,
                            require_f64_parameter_v3(launch, "gain")?,
                            require_usize_parameter_v3(launch, "bars_forward")?,
                            require_i32_parameter_v3(launch, "signal_mode")?,
                        )],
                    )?
            }
            "ehlers_undersampled_double_moving_average" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(
                    launch,
                    &["fast_length", "slow_length", "sample_length"],
                )?;
                self.engine
                    .ehlers_undersampled_double_moving_average_all_outputs(
                        views.close,
                        prepared.first_valid,
                        &[(
                            (require_usize_parameter_v3(launch, "fast_length")?),
                            require_usize_parameter_v3(launch, "slow_length")?,
                            require_usize_parameter_v3(launch, "sample_length")?,
                        )],
                    )?
            }
            "ema_deviation_corrected_t3" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["period", "hot", "t3_mode"])?;
                self.engine.ema_deviation_corrected_t3_all_outputs(
                    views.close,
                    &[(
                        (require_usize_parameter_v3(launch, "period")?),
                        require_f64_parameter_v3(launch, "hot")?,
                        require_usize_parameter_v3(launch, "t3_mode")?,
                    )],
                )?
            }
            "emd" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::HighLow)?;
                require_parameter_keys_v3(launch, &["period", "delta", "fraction"])?;
                self.engine.emd_all_outputs(
                    views.high_low()?,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "period")?),
                        require_f64_parameter_v3(launch, "delta")?,
                        require_f64_parameter_v3(launch, "fraction")?,
                    )],
                )?
            }
            "emd_trend" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["length", "mult"])?;
                self.engine.emd_trend_all_outputs(
                    views.close,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "length")?),
                        require_f64_parameter_v3(launch, "mult")?,
                    )],
                )?
            }
            "eri" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["period"])?;
                self.engine.eri_all_outputs(
                    views.ohlcv()?,
                    prepared.first_valid,
                    &[require_usize_parameter_v3(launch, "period")?],
                )?
            }
            "evasive_supertrend" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(
                    launch,
                    &[
                        "atr_length",
                        "base_multiplier",
                        "noise_threshold",
                        "expansion_alpha",
                    ],
                )?;
                self.engine.evasive_supertrend_all_outputs(
                    views.ohlcv()?,
                    rows,
                    &[(
                        (require_usize_parameter_v3(launch, "atr_length")?),
                        require_f64_parameter_v3(launch, "base_multiplier")?,
                        require_f64_parameter_v3(launch, "noise_threshold")?,
                        require_f64_parameter_v3(launch, "expansion_alpha")?,
                    )],
                )?
            }
            "fibonacci_trailing_stop" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(
                    launch,
                    &["left_bars", "right_bars", "level", "trigger_mode"],
                )?;
                self.engine.fibonacci_trailing_stop_all_outputs(
                    views.ohlcv()?,
                    rows,
                    &[(
                        (require_usize_parameter_v3(launch, "left_bars")?),
                        require_usize_parameter_v3(launch, "right_bars")?,
                        require_f64_parameter_v3(launch, "level")?,
                        require_i32_parameter_v3(launch, "trigger_mode")?,
                    )],
                )?
            }
            "fisher" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::HighLow)?;
                require_parameter_keys_v3(launch, &["period"])?;
                self.engine.fisher_all_outputs(
                    views.high_low()?,
                    prepared.first_valid,
                    &[require_usize_parameter_v3(launch, "period")?],
                )?
            }
            "forward_backward_exponential_oscillator" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(launch, &["length", "smooth"])?;
                self.engine
                    .forward_backward_exponential_oscillator_all_outputs(
                        views.close,
                        rows,
                        &[(
                            (require_usize_parameter_v3(launch, "length")?),
                            require_usize_parameter_v3(launch, "smooth")?,
                        )],
                    )?
            }
            "fvg_trailing_stop" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(
                    launch,
                    &[
                        "unmitigated_fvg_lookback",
                        "smoothing_length",
                        "reset_on_cross",
                    ],
                )?;
                self.engine.fvg_trailing_stop_all_outputs(
                    views.ohlcv()?,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "unmitigated_fvg_lookback")?),
                        require_usize_parameter_v3(launch, "smoothing_length")?,
                        require_bool_parameter_v3(launch, "reset_on_cross")?,
                    )],
                )?
            }
            "gatorosc" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Close)?;
                require_parameter_keys_v3(
                    launch,
                    &[
                        "jaws_length",
                        "jaws_shift",
                        "teeth_length",
                        "teeth_shift",
                        "lips_length",
                        "lips_shift",
                    ],
                )?;
                self.engine.gatorosc_all_outputs(
                    views.close,
                    prepared.first_valid,
                    &[(
                        (require_usize_parameter_v3(launch, "jaws_length")?),
                        require_usize_parameter_v3(launch, "jaws_shift")?,
                        require_usize_parameter_v3(launch, "teeth_length")?,
                        require_usize_parameter_v3(launch, "teeth_shift")?,
                        require_usize_parameter_v3(launch, "lips_length")?,
                        require_usize_parameter_v3(launch, "lips_shift")?,
                    )],
                )?
            }
            "halftrend" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(
                    launch,
                    &["amplitude", "channel_deviation", "atr_period"],
                )?;
                self.engine.halftrend_all_outputs(
                    views.high,
                    views.low,
                    views.close,
                    &[(
                        (require_usize_parameter_v3(launch, "amplitude")?),
                        require_f64_parameter_v3(launch, "channel_deviation")?,
                        require_usize_parameter_v3(launch, "atr_period")?,
                    )],
                )?
            }
            "fibonacci_entry_bands" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["length"])?;
                let length = require_usize_parameter_v3(launch, "length")?;
                let sweep =
                    vector_ta::indicators::fibonacci_entry_bands::FibonacciEntryBandsBatchRange {
                        length: (length, length, 0),
                        atr_length: (14, 14, 0),
                        source: "hlc3".to_owned(),
                        use_atr: true,
                        tp_aggressiveness: "low".to_owned(),
                    };
                self.engine
                    .fibonacci_entry_bands_all_outputs(views.ohlcv()?, rows, rows, &sweep)?
            }
            "ehlers_data_sampling_relative_strength_indicator" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(launch, &["length"])?;
                self.engine
                    .ehlers_data_sampling_relative_strength_indicator_all_outputs(
                        views.ohlcv()?,
                        &[require_usize_parameter_v3(launch, "length")?],
                    )?
            }
            "bulls_v_bears" => {
                require_named_input_v3(launch, ResidentClassicTaInputV3::Ohlcv)?;
                require_parameter_keys_v3(
                    launch,
                    &[
                        "period",
                        "ma_type",
                        "calculation_method",
                        "normalized_bars_back",
                        "raw_rolling_period",
                        "raw_threshold_percentile",
                        "threshold_level",
                    ],
                )?;
                let ma_type = match require_text_parameter_v3(launch, "ma_type")? {
                    "ema" => vector_ta::indicators::bulls_v_bears::BullsVBearsMaType::Ema,
                    "sma" => vector_ta::indicators::bulls_v_bears::BullsVBearsMaType::Sma,
                    "wma" => vector_ta::indicators::bulls_v_bears::BullsVBearsMaType::Wma,
                    value => {
                        return Err(ResidentClassicTaExecutorErrorV3::InvalidTextParameter {
                            indicator_id: launch.indicator_id().to_owned(),
                            key: "ma_type",
                            value: value.to_owned(),
                        });
                    }
                };
                let calculation_method =
                    match require_text_parameter_v3(launch, "calculation_method")? {
                        "normalized" => vector_ta::indicators::bulls_v_bears::BullsVBearsCalculationMethod::Normalized,
                        "raw" => vector_ta::indicators::bulls_v_bears::BullsVBearsCalculationMethod::Raw,
                        value => {
                            return Err(ResidentClassicTaExecutorErrorV3::InvalidTextParameter {
                                indicator_id: launch.indicator_id().to_owned(),
                                key: "calculation_method",
                                value: value.to_owned(),
                            });
                        }
                    };
                let parameters = vector_ta::indicators::bulls_v_bears::BullsVBearsParams {
                    period: Some(require_usize_parameter_v3(launch, "period")?),
                    ma_type: Some(ma_type),
                    calculation_method: Some(calculation_method),
                    normalized_bars_back: Some(require_usize_parameter_v3(
                        launch,
                        "normalized_bars_back",
                    )?),
                    raw_rolling_period: Some(require_usize_parameter_v3(
                        launch,
                        "raw_rolling_period",
                    )?),
                    raw_threshold_percentile: Some(require_f64_parameter_v3(
                        launch,
                        "raw_threshold_percentile",
                    )?),
                    threshold_level: Some(require_f64_parameter_v3(launch, "threshold_level")?),
                };
                self.engine
                    .bulls_v_bears_all_outputs(views.ohlcv()?, rows, &[parameters])?
            }
            _ => {
                return Err(ResidentClassicTaExecutorErrorV3::UnsupportedNamedRoute {
                    indicator_id: launch.indicator_id().to_owned(),
                });
            }
        };
        Ok(result)
    }

    fn validate_observed_route_v3(
        &self,
        launch: &ResidentClassicTaLaunchRecipeV3,
        observed: &F64ResidentObservedRouteManifestV3,
    ) -> Result<Vec<usize>, ResidentClassicTaExecutorErrorV3> {
        let expected_outputs = launch
            .outputs()
            .iter()
            .map(ResidentClassicTaOutputRouteV3::output_id)
            .collect::<Vec<_>>();
        let device = self.run_device_identity_v3();
        let sass_arch = parse_native_sass_arch_v3(device.native_sass_target())?;
        let output_indices = expected_outputs
            .iter()
            .map(|expected| {
                observed
                    .output_ids
                    .iter()
                    .position(|actual| actual == expected)
            })
            .collect::<Option<Vec<_>>>();
        let ordered_unique_subset = output_indices
            .as_ref()
            .is_some_and(|indices| indices.windows(2).all(|window| window[0] < window[1]));
        let identity_matches = observed.indicator_id == launch.indicator_id()
            && observed.entry_point == launch.entry_point()
            && ordered_unique_subset
            && observed.rows == 1
            && observed.cols == self.recipe.rows()
            && observed.device_id == self.device_ordinal
            && observed.compiled_architectures.contains(&sass_arch)
            && !observed.compiled_arch_source.trim().is_empty()
            && observed.compiled_nvcc_version == device.nvcc_version()
            && observed.exact_math_authority == F64_EXACT_MATH_AUTHORITY_V3
            && device.exact_math_authority() == F64_EXACT_MATH_AUTHORITY_V3
            && device.vector_ta_build_sha256() != [0; 32];
        if !identity_matches {
            return Err(ResidentClassicTaExecutorErrorV3::ObservedRouteMismatch {
                indicator_id: launch.indicator_id().to_owned(),
            });
        }
        Ok(output_indices.expect("ordered subset was proved above"))
    }

    fn run_device_identity_v3(
        &self,
    ) -> &neoethos_gpu_contracts::resident_feature_store_v3::CudaPrimaryContextBuildIdentityV3 {
        &self.device_identity
    }
}

#[derive(Debug, Clone, Copy)]
struct ResidentClassicTaPreparedLaunchV3<'recipe> {
    recipe: &'recipe ResidentClassicTaLaunchRecipeV3,
    inputs: F64Inputs,
    first_valid: usize,
}

fn require_input_kind_v3(
    input: ResidentClassicTaInputV3,
    expected: F64InputKind,
) -> Result<(), ResidentClassicTaExecutorErrorV3> {
    let matches = matches!(
        (input, expected),
        (ResidentClassicTaInputV3::Close, F64InputKind::CloseSlice)
            | (ResidentClassicTaInputV3::Ohlc, F64InputKind::Ohlc4)
            | (ResidentClassicTaInputV3::Hlc3, F64InputKind::Hlc3Slice)
            | (
                ResidentClassicTaInputV3::Hlc3Volume,
                F64InputKind::Hlc3Volume
            )
            | (
                ResidentClassicTaInputV3::CloseVolume,
                F64InputKind::CloseVolume
            )
            | (ResidentClassicTaInputV3::HighLow, F64InputKind::HighLow)
            | (
                ResidentClassicTaInputV3::TimestampCloseVolume,
                F64InputKind::TimestampCloseVolume
            )
            | (ResidentClassicTaInputV3::Hl2, F64InputKind::Hl2Slice)
            | (
                ResidentClassicTaInputV3::HighLowVolume,
                F64InputKind::HighLowVolume
            )
            | (ResidentClassicTaInputV3::Hlcv, F64InputKind::Hlcv)
            | (ResidentClassicTaInputV3::Ohlcv, F64InputKind::Ohlcv5)
            | (
                ResidentClassicTaInputV3::OpenCloseVolume,
                F64InputKind::OpenCloseVolume
            )
            // These canonical source selections share an ABI shape with other
            // registry kinds, but their semantic kinds must still match: the
            // recipe identity and parent view select the exact source series.
            | (ResidentClassicTaInputV3::Hlcc4, F64InputKind::Hlcc4Slice)
            | (ResidentClassicTaInputV3::Volume, F64InputKind::VolumeSlice)
            | (
                ResidentClassicTaInputV3::Hlcc4Volume,
                F64InputKind::Hlcc4Volume
            )
            | (ResidentClassicTaInputV3::Hlc, F64InputKind::Hlc)
    );
    if !matches {
        return Err(ResidentClassicTaExecutorErrorV3::InputKindMismatch { input, expected });
    }
    Ok(())
}

fn require_named_input_v3(
    launch: &ResidentClassicTaLaunchRecipeV3,
    expected: ResidentClassicTaInputV3,
) -> Result<(), ResidentClassicTaExecutorErrorV3> {
    if launch.input() != expected {
        return Err(ResidentClassicTaExecutorErrorV3::NamedInputMismatch {
            indicator_id: launch.indicator_id().to_owned(),
            expected,
            actual: launch.input(),
        });
    }
    Ok(())
}

fn require_parameter_v3<'recipe>(
    launch: &'recipe ResidentClassicTaLaunchRecipeV3,
    key: &'static str,
) -> Result<&'recipe ResidentClassicTaParameterValueV3, ResidentClassicTaExecutorErrorV3> {
    launch
        .parameters()
        .iter()
        .find(|parameter| parameter.key() == key)
        .map(ResidentClassicTaParameterV3::value)
        .ok_or_else(|| ResidentClassicTaExecutorErrorV3::MissingParameter {
            indicator_id: launch.indicator_id().to_owned(),
            key,
        })
}

fn require_parameter_keys_v3(
    launch: &ResidentClassicTaLaunchRecipeV3,
    expected: &[&'static str],
) -> Result<(), ResidentClassicTaExecutorErrorV3> {
    if launch.parameters().len() != expected.len()
        || launch
            .parameters()
            .iter()
            .zip(expected)
            .any(|(parameter, expected)| parameter.key() != *expected)
    {
        return Err(ResidentClassicTaExecutorErrorV3::ParameterSchemaMismatch {
            indicator_id: launch.indicator_id().to_owned(),
        });
    }
    Ok(())
}

fn require_usize_parameter_v3(
    launch: &ResidentClassicTaLaunchRecipeV3,
    key: &'static str,
) -> Result<usize, ResidentClassicTaExecutorErrorV3> {
    let ResidentClassicTaParameterValueV3::Usize(value) = require_parameter_v3(launch, key)? else {
        return Err(ResidentClassicTaExecutorErrorV3::ParameterTypeMismatch {
            indicator_id: launch.indicator_id().to_owned(),
            key,
            expected: "usize",
        });
    };
    usize::try_from(*value).map_err(|_| ResidentClassicTaExecutorErrorV3::ParameterWidth {
        indicator_id: launch.indicator_id().to_owned(),
        key,
    })
}

fn require_i32_parameter_v3(
    launch: &ResidentClassicTaLaunchRecipeV3,
    key: &'static str,
) -> Result<i32, ResidentClassicTaExecutorErrorV3> {
    let ResidentClassicTaParameterValueV3::I32(value) = require_parameter_v3(launch, key)? else {
        return Err(ResidentClassicTaExecutorErrorV3::ParameterTypeMismatch {
            indicator_id: launch.indicator_id().to_owned(),
            key,
            expected: "i32",
        });
    };
    Ok(*value)
}

fn require_bool_parameter_v3(
    launch: &ResidentClassicTaLaunchRecipeV3,
    key: &'static str,
) -> Result<bool, ResidentClassicTaExecutorErrorV3> {
    let ResidentClassicTaParameterValueV3::Bool(value) = require_parameter_v3(launch, key)? else {
        return Err(ResidentClassicTaExecutorErrorV3::ParameterTypeMismatch {
            indicator_id: launch.indicator_id().to_owned(),
            key,
            expected: "bool",
        });
    };
    Ok(*value)
}

fn require_f64_parameter_v3(
    launch: &ResidentClassicTaLaunchRecipeV3,
    key: &'static str,
) -> Result<f64, ResidentClassicTaExecutorErrorV3> {
    let ResidentClassicTaParameterValueV3::F64Bits(value) = require_parameter_v3(launch, key)?
    else {
        return Err(ResidentClassicTaExecutorErrorV3::ParameterTypeMismatch {
            indicator_id: launch.indicator_id().to_owned(),
            key,
            expected: "f64 bits",
        });
    };
    Ok(f64::from_bits(*value))
}

fn require_text_parameter_v3<'recipe>(
    launch: &'recipe ResidentClassicTaLaunchRecipeV3,
    key: &'static str,
) -> Result<&'recipe str, ResidentClassicTaExecutorErrorV3> {
    let ResidentClassicTaParameterValueV3::Text(value) = require_parameter_v3(launch, key)? else {
        return Err(ResidentClassicTaExecutorErrorV3::ParameterTypeMismatch {
            indicator_id: launch.indicator_id().to_owned(),
            key,
            expected: "text",
        });
    };
    Ok(value)
}

fn parse_native_sass_arch_v3(
    native_sass_target: &str,
) -> Result<u32, ResidentClassicTaExecutorErrorV3> {
    native_sass_target
        .strip_prefix("sm_")
        .and_then(|digits| digits.parse::<u32>().ok())
        .filter(|architecture| *architecture > 0)
        .ok_or_else(
            || ResidentClassicTaExecutorErrorV3::InvalidNativeSassTarget {
                actual: native_sass_target.to_owned(),
            },
        )
}

#[derive(Debug, Error)]
pub enum ResidentClassicTaExecutorErrorV3 {
    #[error(transparent)]
    Recipe(#[from] ResidentClassicTaRecipeErrorV3),
    #[error(transparent)]
    ResidentStore(#[from] ResidentFeatureStoreCudaErrorV3),
    #[error(transparent)]
    Cuda(#[from] cust::error::CudaError),
    #[error(transparent)]
    VectorTa(#[from] CudaF64IndicatorError),
    #[error("resident Classic TA parent/context/stream/ordinal authority mismatch")]
    ParentAuthorityMismatch,
    #[error("resident Classic TA device view `{field}` is invalid: {error}")]
    DeviceView {
        field: &'static str,
        error: vector_ta::cuda::CudaDeviceViewError,
    },
    #[error("resident Classic TA native `{operation}` failed with status {status}")]
    Native {
        operation: &'static str,
        status: i32,
    },
    #[error("resident Classic TA input kind {input:?} does not match vector-ta {expected:?}")]
    InputKindMismatch {
        input: ResidentClassicTaInputV3,
        expected: F64InputKind,
    },
    #[error(
        "resident Classic TA named route `{indicator_id}` expected {expected:?}, found {actual:?}"
    )]
    NamedInputMismatch {
        indicator_id: String,
        expected: ResidentClassicTaInputV3,
        actual: ResidentClassicTaInputV3,
    },
    #[error("resident Classic TA `{indicator_id}` omitted exact parameter `{key}`")]
    MissingParameter {
        indicator_id: String,
        key: &'static str,
    },
    #[error("resident Classic TA `{indicator_id}.{key}` is not encoded as {expected}")]
    ParameterTypeMismatch {
        indicator_id: String,
        key: &'static str,
        expected: &'static str,
    },
    #[error("resident Classic TA `{indicator_id}.{key}` exceeds the host ABI width")]
    ParameterWidth {
        indicator_id: String,
        key: &'static str,
    },
    #[error("resident Classic TA `{indicator_id}.{key}` has unsupported text `{value}`")]
    InvalidTextParameter {
        indicator_id: String,
        key: &'static str,
        value: String,
    },
    #[error("resident Classic TA `{indicator_id}` parameter schema/order is not canonical")]
    ParameterSchemaMismatch { indicator_id: String },
    #[error("resident Classic TA observed route/build identity mismatched `{indicator_id}`")]
    ObservedRouteMismatch { indicator_id: String },
    #[error("resident Classic TA native SASS target `{actual}` is not canonical sm_NN")]
    InvalidNativeSassTarget { actual: String },
    #[error("resident Classic TA named route `{indicator_id}` has no exact typed dispatcher")]
    UnsupportedNamedRoute { indicator_id: String },
    #[error(
        "resident Classic TA named route `{indicator_id}` lacks an exact pre-device allocation plan"
    )]
    UnsupportedNamedAllocationPlan { indicator_id: String },
    #[error("resident Classic TA pre-device memory receipt mismatched `{indicator_id}`")]
    PreDeviceMemoryReceiptMismatch { indicator_id: String },
    #[error("resident Classic TA admitted global column bindings do not match the local draft")]
    AdmittedGlobalBindingsMismatch,
    #[error("invalid resident Classic TA input: {0}")]
    InvalidInput(String),
    #[error("arithmetic overflow while deriving {0}")]
    ArithmeticOverflow(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResidentClassicTaRecipeErrorV3 {
    #[error("{0} must be nonempty")]
    EmptyField(&'static str),
    #[error("{0} must be a nonzero SHA-256 digest")]
    ZeroHash(&'static str),
    #[error(
        "invalid Classic TA extents rows={rows} budget_rows={budget_rows} available_bytes={available_bytes_at_admission}"
    )]
    InvalidExtent {
        rows: usize,
        budget_rows: usize,
        available_bytes_at_admission: u64,
    },
    #[error("Classic TA launch plan is empty")]
    EmptyLaunchPlan,
    #[error("Classic TA launch width must be in 1..=64, got {columns}")]
    InvalidLaunchWidth { columns: usize },
    #[error("Classic TA validity code {0} is outside the sealed 0..=9 domain")]
    InvalidValidityCode(u8),
    #[error("duplicate Classic TA parameter key `{0}`")]
    DuplicateParameterKey(String),
    #[error("duplicate Classic TA feature name `{0}`")]
    DuplicateFeatureName(String),
    #[error("Classic TA output range is not contiguous: expected {expected}, got {actual}")]
    NonContiguousOutputRange { expected: usize, actual: usize },
    #[error("Classic TA launch range is not contiguous: expected {expected}, got {actual}")]
    NonContiguousLaunchRange { expected: usize, actual: usize },
    #[error("Classic TA stage/period identity is invalid for `{feature_name}`")]
    InvalidStagePeriod { feature_name: String },
    #[error("Classic TA route plan SHA-256 does not match its canonical bytes")]
    RoutePlanHashMismatch,
    #[error("arithmetic overflow while deriving {0}")]
    ArithmeticOverflow(&'static str),
}

fn require_text(field: &'static str, value: &str) -> Result<(), ResidentClassicTaRecipeErrorV3> {
    if value.trim().is_empty() {
        Err(ResidentClassicTaRecipeErrorV3::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_hash(
    field: &'static str,
    value: &[u8; SHA256_BYTES],
) -> Result<(), ResidentClassicTaRecipeErrorV3> {
    if value.iter().all(|byte| *byte == 0) {
        Err(ResidentClassicTaRecipeErrorV3::ZeroHash(field))
    } else {
        Ok(())
    }
}

fn update_usize(hasher: &mut Sha256, value: usize) -> Result<(), ResidentClassicTaRecipeErrorV3> {
    let value = u64::try_from(value)
        .map_err(|_| ResidentClassicTaRecipeErrorV3::ArithmeticOverflow("usize wire width"))?;
    hasher.update(value.to_le_bytes());
    Ok(())
}

fn update_text(hasher: &mut Sha256, value: &str) -> Result<(), ResidentClassicTaRecipeErrorV3> {
    update_usize(hasher, value.len())?;
    hasher.update(value.as_bytes());
    Ok(())
}

fn hash_recipe(
    rows: usize,
    budget_rows: usize,
    available_bytes_at_admission: u64,
    admitted_working_set_sha256: [u8; SHA256_BYTES],
    launches: &[ResidentClassicTaLaunchRecipeV3],
) -> Result<[u8; SHA256_BYTES], ResidentClassicTaRecipeErrorV3> {
    let mut hasher = Sha256::new();
    update_text(&mut hasher, RESIDENT_CLASSIC_TA_RECIPE_AUTHORITY_V3)?;
    update_usize(&mut hasher, rows)?;
    update_usize(&mut hasher, budget_rows)?;
    hasher.update(available_bytes_at_admission.to_le_bytes());
    hasher.update(admitted_working_set_sha256);
    update_usize(&mut hasher, launches.len())?;
    for launch in launches {
        launch.update_hash(&mut hasher)?;
    }
    Ok(hasher.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(byte: u8) -> [u8; SHA256_BYTES] {
        [byte; SHA256_BYTES]
    }

    fn output(destination_column: usize, feature_name: &str) -> ResidentClassicTaOutputRouteV3 {
        ResidentClassicTaOutputRouteV3::new(
            destination_column,
            feature_name,
            "value",
            ResidentClassicTaStageV3::Base,
            None,
            hash(1),
            hash(2),
        )
        .expect("test route must be exact")
    }

    fn launch(destination_column: usize, feature_name: &str) -> ResidentClassicTaLaunchRecipeV3 {
        ResidentClassicTaLaunchRecipeV3::new(
            "sma",
            "neoethos_sma_batch_f64",
            ResidentClassicTaInputV3::Close,
            ResidentClassicTaFirstValidRuleV3::AllInputsNonNan,
            vec![
                ResidentClassicTaParameterV3::new(
                    "period",
                    ResidentClassicTaParameterValueV3::Usize(14),
                )
                .expect("test period must be exact"),
            ],
            vec![output(destination_column, feature_name)],
            8,
        )
        .expect("test launch must be exact")
    }

    #[test]
    fn recipe_hash_binds_order_names_bits_and_admission_evidence() {
        let left =
            ResidentClassicTaRecipeV3::seal(128, 256, 8 << 30, hash(3), vec![launch(0, "sma")])
                .expect("test recipe must seal");
        let right =
            ResidentClassicTaRecipeV3::seal(128, 256, 8 << 30, hash(3), vec![launch(0, "ema")])
                .expect("test recipe must seal");
        assert_ne!(left.route_plan_sha256(), right.route_plan_sha256());
        assert_eq!(
            preflight_resident_classic_ta_recipe_v3(left.clone())
                .expect("unchanged recipe must pass"),
            left
        );
    }

    #[test]
    fn recipe_refuses_stage_period_and_destination_drift() {
        assert!(matches!(
            ResidentClassicTaOutputRouteV3::new(
                0,
                "sma_7",
                "value",
                ResidentClassicTaStageV3::Historical,
                None,
                hash(1),
                hash(2),
            ),
            Err(ResidentClassicTaRecipeErrorV3::InvalidStagePeriod { .. })
        ));
        assert!(matches!(
            ResidentClassicTaRecipeV3::seal(128, 256, 8 << 30, hash(3), vec![launch(1, "sma")],),
            Err(ResidentClassicTaRecipeErrorV3::NonContiguousLaunchRange {
                expected: 0,
                actual: 1
            })
        ));
    }
}
