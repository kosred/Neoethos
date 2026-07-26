//! Safe Rust ownership wrapper around the persistent Prototype B CUDA session.
//!
//! Every argument is validated on the host before it crosses the C ABI, so an
//! invalid shape is a typed Rust error rather than undefined behaviour on the
//! device. A session owns exactly one native session; `Drop` destroys it on
//! every path, including error paths and unwinds.

use super::{
    DatasetHeader, GeneDescriptor, NeoPopulationCounters, NeoPopulationEvent,
    NeoPopulationMetricRow, NeoPopulationOutcome, NeoPopulationSettings, SMC_SLOTS,
    ScenarioDescriptor,
};
use neoethos_gpu_contracts::ABI_VERSION;
use std::ffi::c_void;
use thiserror::Error;

pub const STATUS_OK: i32 = 0;
pub const STATUS_UNSUPPORTED: i32 = -1;
pub const STATUS_NULL_SESSION: i32 = -30;
pub const STATUS_ABI_MISMATCH: i32 = -31;
pub const STATUS_INVALID_ARGUMENT: i32 = -32;
pub const STATUS_DEVICE_UNAVAILABLE: i32 = -33;
pub const STATUS_ALLOCATION_FAILED: i32 = -34;
pub const STATUS_TRANSFER_FAILED: i32 = -35;
pub const STATUS_LAUNCH_FAILED: i32 = -36;
pub const STATUS_EVENT_CAPACITY: i32 = -37;
pub const STATUS_MISSING_UPLOAD: i32 = -38;
pub const STATUS_READBACK_CAPACITY: i32 = -39;
pub const STATUS_SYNC_FAILED: i32 = -40;
pub const STATUS_UNKNOWN_EVENT: i32 = -41;
pub const STATUS_DATASET_REUPLOAD: i32 = -42;

pub fn population_status_message(status: i32) -> &'static str {
    match status {
        STATUS_OK => "ok",
        STATUS_UNSUPPORTED => "this build has no CUDA runtime",
        STATUS_NULL_SESSION => "null native session",
        STATUS_ABI_MISMATCH => "native ABI version mismatch",
        STATUS_INVALID_ARGUMENT => "invalid native argument",
        STATUS_DEVICE_UNAVAILABLE => "CUDA device is unavailable",
        STATUS_ALLOCATION_FAILED => "device allocation failed",
        STATUS_TRANSFER_FAILED => "host/device transfer failed",
        STATUS_LAUNCH_FAILED => "kernel launch failed",
        STATUS_EVENT_CAPACITY => "emitted events exceeded the session capacity",
        STATUS_MISSING_UPLOAD => "a required upload or evaluation is missing",
        STATUS_READBACK_CAPACITY => "readback buffer is too small",
        STATUS_SYNC_FAILED => "stream or event synchronization failed",
        STATUS_UNKNOWN_EVENT => "event id does not belong to this session",
        STATUS_DATASET_REUPLOAD => "a session accepts exactly one logical dataset upload",
        _ => "unknown native status",
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CudaPopulationError {
    #[error("native CUDA population {operation} failed with status {status} ({message})")]
    Native {
        operation: &'static str,
        status: i32,
        message: &'static str,
    },
    #[error("native CUDA population runtime is unavailable in this build")]
    RuntimeUnavailable,
    #[error("invalid Prototype B population input: {0}")]
    InvalidInput(String),
}

impl CudaPopulationError {
    fn native(operation: &'static str, status: i32) -> Self {
        if status == STATUS_UNSUPPORTED {
            return Self::RuntimeUnavailable;
        }
        Self::Native {
            operation,
            status,
            message: population_status_message(status),
        }
    }

    pub fn is_runtime_unavailable(&self) -> bool {
        matches!(self, Self::RuntimeUnavailable)
    }

    pub fn is_capacity_exhausted(&self) -> bool {
        matches!(
            self,
            Self::Native {
                status: STATUS_EVENT_CAPACITY,
                ..
            }
        )
    }
}

fn invalid(detail: impl Into<String>) -> CudaPopulationError {
    CudaPopulationError::InvalidInput(detail.into())
}

/// Borrowed host dataset for exactly one logical upload.
#[derive(Debug, Clone, Copy)]
pub struct PopulationDatasetView<'a> {
    pub close: &'a [f64],
    pub high: &'a [f64],
    pub low: &'a [f64],
    /// Feature-major `[feature][bar]` values.
    pub indicators: &'a [f32],
    pub feature_count: usize,
    pub months: &'a [i64],
    pub days: &'a [i64],
    pub timestamps: &'a [i64],
    /// Row-major `[bar][slot]` SMC contract, `SMC_SLOTS` per bar.
    pub smc_rows: &'a [i8],
    /// f64 on purpose: the canonical adaptive-at-entry stop distance is f64,
    /// and narrowing it would silently break exact parity.
    pub adaptive_base_pips: Option<&'a [f64]>,
}

impl PopulationDatasetView<'_> {
    fn validate(&self) -> Result<usize, CudaPopulationError> {
        let bars = self.close.len();
        if bars == 0 {
            return Err(invalid("dataset has no bars"));
        }
        if self.feature_count == 0 {
            return Err(invalid("dataset has no features"));
        }
        for (field, actual) in [
            ("high", self.high.len()),
            ("low", self.low.len()),
            ("months", self.months.len()),
            ("days", self.days.len()),
            ("timestamps", self.timestamps.len()),
        ] {
            if actual != bars {
                return Err(invalid(format!(
                    "dataset {field} length {actual} does not match {bars} bars"
                )));
            }
        }
        if self.smc_rows.len() != bars * SMC_SLOTS {
            return Err(invalid(format!(
                "dataset smc_rows length {} does not match {bars} bars x {SMC_SLOTS} slots",
                self.smc_rows.len()
            )));
        }
        let expected = self
            .feature_count
            .checked_mul(bars)
            .ok_or_else(|| invalid("dataset indicator extent overflows usize"))?;
        if self.indicators.len() != expected {
            return Err(invalid(format!(
                "dataset indicators length {} does not match {expected}",
                self.indicators.len()
            )));
        }
        if let Some(base) = self.adaptive_base_pips {
            if base.len() != bars {
                return Err(invalid(format!(
                    "adaptive base length {} does not match {bars} bars",
                    base.len()
                )));
            }
        }
        for (field, values) in [
            ("close", self.close),
            ("high", self.high),
            ("low", self.low),
        ] {
            if let Some(index) = values.iter().position(|value| !value.is_finite()) {
                return Err(invalid(format!("dataset {field}[{index}] is not finite")));
            }
        }
        Ok(bars)
    }
}

/// Borrowed canonical gene batch.
#[derive(Debug, Clone, Copy)]
pub struct PopulationGeneView<'a> {
    pub descriptors: &'a [GeneDescriptor],
    pub offsets: &'a [i32],
    pub indices: &'a [i32],
    pub weights: &'a [f32],
    pub stop_pips: &'a [f64],
    pub target_pips: &'a [f64],
    pub stop_vol_multipliers: &'a [f64],
    /// Row-major `[candidate][slot]` gene SMC flags.
    pub smc_flags: &'a [i8],
    pub smc_weights: &'a [f32; SMC_SLOTS],
    pub gate_threshold: f32,
    pub smc_gate_disabled: bool,
}

impl PopulationGeneView<'_> {
    fn validate(&self, feature_count: usize) -> Result<usize, CudaPopulationError> {
        let population = self.descriptors.len();
        if population == 0 {
            return Err(invalid("gene batch is empty"));
        }
        for (field, actual) in [
            ("stop_pips", self.stop_pips.len()),
            ("target_pips", self.target_pips.len()),
            ("stop_vol_multipliers", self.stop_vol_multipliers.len()),
        ] {
            if actual != population {
                return Err(invalid(format!(
                    "gene {field} length {actual} does not match population {population}"
                )));
            }
        }
        if self.smc_flags.len() != population * SMC_SLOTS {
            return Err(invalid(format!(
                "gene smc_flags length {} does not match {population} x {SMC_SLOTS}",
                self.smc_flags.len()
            )));
        }
        if self.offsets.len() != population + 1 {
            return Err(invalid(format!(
                "gene offsets length {} does not match population {population} + 1",
                self.offsets.len()
            )));
        }
        if self.indices.len() != self.weights.len() {
            return Err(invalid("gene indices and weights lengths differ"));
        }
        if self.offsets.first().copied() != Some(0)
            || self.offsets.last().copied() != Some(self.indices.len() as i32)
            || self
                .offsets
                .windows(2)
                .any(|window| window[0] < 0 || window[0] > window[1])
        {
            return Err(invalid("gene CSR offsets are not monotonic and complete"));
        }
        if let Some((position, index)) = self
            .indices
            .iter()
            .copied()
            .enumerate()
            .find(|(_, index)| *index < 0 || *index as usize >= feature_count)
        {
            return Err(invalid(format!(
                "gene term {position} references feature {index}, outside 0..{feature_count}"
            )));
        }
        Ok(population)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PopulationDiagnostics {
    pub events: Vec<NeoPopulationEvent>,
    pub outcomes: Vec<NeoPopulationOutcome>,
}

#[repr(C)]
struct RawDatasetView {
    header: DatasetHeader,
    close: *const f64,
    high: *const f64,
    low: *const f64,
    indicators: *const f32,
    months: *const i64,
    days: *const i64,
    timestamps: *const i64,
    smc_rows: *const i8,
    adaptive_base_pips: *const f64,
    adaptive_base_pips_len: usize,
}

#[repr(C)]
struct RawGeneView {
    descriptors: *const GeneDescriptor,
    count: usize,
    offsets: *const i32,
    indices: *const i32,
    weights: *const f32,
    term_count: usize,
    stop_pips: *const f64,
    target_pips: *const f64,
    stop_vol_multipliers: *const f64,
    smc_flags: *const i8,
    smc_weights: *const f32,
    gate_threshold: f32,
    smc_gate_disabled: u32,
}

#[repr(C)]
struct RawScenarioView {
    descriptors: *const ScenarioDescriptor,
    count: usize,
}

#[repr(C)]
struct RawReadback {
    rows: *mut NeoPopulationMetricRow,
    capacity: usize,
    written: *mut usize,
}

#[repr(C)]
struct RawDiagnosticReadback {
    events: *mut NeoPopulationEvent,
    outcomes: *mut NeoPopulationOutcome,
    capacity: usize,
    written: *mut usize,
}

unsafe extern "C" {
    fn neoethos_gpu_cuda_population_create(
        abi_version: u32,
        device: i32,
        max_events: usize,
        status: *mut i32,
    ) -> *mut c_void;
    fn neoethos_gpu_cuda_population_upload_dataset(
        session: *mut c_void,
        dataset: *const RawDatasetView,
    ) -> i32;
    fn neoethos_gpu_cuda_population_upload_genes(
        session: *mut c_void,
        genes: *const RawGeneView,
    ) -> i32;
    fn neoethos_gpu_cuda_population_upload_scenarios(
        session: *mut c_void,
        scenarios: *const RawScenarioView,
    ) -> i32;
    fn neoethos_gpu_cuda_population_b_evaluate(
        session: *mut c_void,
        settings: *const NeoPopulationSettings,
        event_id: *mut u64,
        counters: *mut NeoPopulationCounters,
    ) -> i32;
    fn neoethos_gpu_cuda_population_wait(session: *mut c_void, event_id: u64) -> i32;
    fn neoethos_gpu_cuda_population_read_metrics(
        session: *mut c_void,
        readback: *mut RawReadback,
    ) -> i32;
    fn neoethos_gpu_cuda_population_read_diagnostics(
        session: *mut c_void,
        readback: *mut RawDiagnosticReadback,
    ) -> i32;
    fn neoethos_gpu_cuda_population_destroy(session: *mut c_void);
}

/// Owns exactly one native CUDA population session.
#[derive(Debug)]
pub struct PopulationSession {
    handle: *mut c_void,
    device: i32,
    max_events: usize,
    bars: usize,
    feature_count: usize,
    population: usize,
    emitted_events: usize,
    dataset_uploaded: bool,
    genes_uploaded: bool,
    scenarios_uploaded: bool,
    pending_event: Option<u64>,
    metrics_ready: bool,
}

impl PopulationSession {
    pub fn create(device: i32, max_events: usize) -> Result<Self, CudaPopulationError> {
        if max_events == 0 {
            return Err(invalid("max_events must be non-zero"));
        }
        if device < 0 {
            return Err(invalid("device index must be non-negative"));
        }
        let mut status = STATUS_OK;
        // SAFETY: `status` is a valid out-parameter; the native side either
        // returns a live session or a null pointer plus a typed status.
        let handle = unsafe {
            neoethos_gpu_cuda_population_create(ABI_VERSION, device, max_events, &mut status)
        };
        if handle.is_null() {
            return Err(CudaPopulationError::native("create", status));
        }
        Ok(Self {
            handle,
            device,
            max_events,
            bars: 0,
            feature_count: 0,
            population: 0,
            emitted_events: 0,
            dataset_uploaded: false,
            genes_uploaded: false,
            scenarios_uploaded: false,
            pending_event: None,
            metrics_ready: false,
        })
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn max_events(&self) -> usize {
        self.max_events
    }

    pub fn population(&self) -> usize {
        self.population
    }

    pub fn bars(&self) -> usize {
        self.bars
    }

    pub fn emitted_events(&self) -> usize {
        self.emitted_events
    }

    pub fn upload_dataset(
        &mut self,
        dataset: PopulationDatasetView<'_>,
    ) -> Result<(), CudaPopulationError> {
        if self.dataset_uploaded {
            return Err(CudaPopulationError::native(
                "upload_dataset",
                STATUS_DATASET_REUPLOAD,
            ));
        }
        let bars = dataset.validate()?;
        let header = DatasetHeader {
            abi_version: ABI_VERSION,
            row_count: bars as u64,
            feature_count: dataset.feature_count as u32,
            ..DatasetHeader::default()
        };
        let raw = RawDatasetView {
            header,
            close: dataset.close.as_ptr(),
            high: dataset.high.as_ptr(),
            low: dataset.low.as_ptr(),
            indicators: dataset.indicators.as_ptr(),
            months: dataset.months.as_ptr(),
            days: dataset.days.as_ptr(),
            timestamps: dataset.timestamps.as_ptr(),
            smc_rows: dataset.smc_rows.as_ptr(),
            adaptive_base_pips: dataset
                .adaptive_base_pips
                .map_or(std::ptr::null(), <[f64]>::as_ptr),
            adaptive_base_pips_len: dataset.adaptive_base_pips.map_or(0, <[f64]>::len),
        };
        // SAFETY: every pointer borrows a host slice that outlives this call and
        // whose length was validated above.
        let status = unsafe { neoethos_gpu_cuda_population_upload_dataset(self.handle, &raw) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("upload_dataset", status));
        }
        self.bars = bars;
        self.feature_count = dataset.feature_count;
        self.dataset_uploaded = true;
        Ok(())
    }

    pub fn upload_genes(
        &mut self,
        genes: PopulationGeneView<'_>,
    ) -> Result<(), CudaPopulationError> {
        if !self.dataset_uploaded {
            return Err(CudaPopulationError::native(
                "upload_genes",
                STATUS_MISSING_UPLOAD,
            ));
        }
        let population = genes.validate(self.feature_count)?;
        let raw = RawGeneView {
            descriptors: genes.descriptors.as_ptr(),
            count: population,
            offsets: genes.offsets.as_ptr(),
            indices: genes.indices.as_ptr(),
            weights: genes.weights.as_ptr(),
            term_count: genes.indices.len(),
            stop_pips: genes.stop_pips.as_ptr(),
            target_pips: genes.target_pips.as_ptr(),
            stop_vol_multipliers: genes.stop_vol_multipliers.as_ptr(),
            smc_flags: genes.smc_flags.as_ptr(),
            smc_weights: genes.smc_weights.as_ptr(),
            gate_threshold: genes.gate_threshold,
            smc_gate_disabled: u32::from(genes.smc_gate_disabled),
        };
        // SAFETY: as above; the native side copies before returning.
        let status = unsafe { neoethos_gpu_cuda_population_upload_genes(self.handle, &raw) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("upload_genes", status));
        }
        self.population = population;
        self.genes_uploaded = true;
        self.scenarios_uploaded = false;
        self.metrics_ready = false;
        self.pending_event = None;
        Ok(())
    }

    pub fn upload_scenarios(
        &mut self,
        scenarios: &[ScenarioDescriptor],
    ) -> Result<(), CudaPopulationError> {
        if !self.genes_uploaded {
            return Err(CudaPopulationError::native(
                "upload_scenarios",
                STATUS_MISSING_UPLOAD,
            ));
        }
        if scenarios.len() != self.population {
            return Err(invalid(format!(
                "scenario count {} does not match population {}",
                scenarios.len(),
                self.population
            )));
        }
        let raw = RawScenarioView {
            descriptors: scenarios.as_ptr(),
            count: scenarios.len(),
        };
        // SAFETY: as above.
        let status = unsafe { neoethos_gpu_cuda_population_upload_scenarios(self.handle, &raw) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("upload_scenarios", status));
        }
        self.scenarios_uploaded = true;
        self.metrics_ready = false;
        self.pending_event = None;
        Ok(())
    }

    pub fn evaluate(
        &mut self,
        settings: &NeoPopulationSettings,
    ) -> Result<(u64, NeoPopulationCounters), CudaPopulationError> {
        if !self.scenarios_uploaded {
            return Err(CudaPopulationError::native(
                "evaluate",
                STATUS_MISSING_UPLOAD,
            ));
        }
        if settings.month_capacity == 0 {
            return Err(invalid("month_capacity must be non-zero"));
        }
        let mut event_id = 0_u64;
        let mut counters = NeoPopulationCounters::default();
        // SAFETY: settings is a live POD reference; both out-parameters are valid.
        let status = unsafe {
            neoethos_gpu_cuda_population_b_evaluate(
                self.handle,
                settings,
                &mut event_id,
                &mut counters,
            )
        };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("evaluate", status));
        }
        self.emitted_events = counters.event_count as usize;
        self.pending_event = Some(event_id);
        self.metrics_ready = false;
        Ok((event_id, counters))
    }

    pub fn wait(&mut self, event_id: u64) -> Result<(), CudaPopulationError> {
        if self.pending_event != Some(event_id) {
            return Err(CudaPopulationError::native("wait", STATUS_UNKNOWN_EVENT));
        }
        // SAFETY: the handle is live for the lifetime of `self`.
        let status = unsafe { neoethos_gpu_cuda_population_wait(self.handle, event_id) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("wait", status));
        }
        self.metrics_ready = true;
        Ok(())
    }

    pub fn read_metrics(&mut self) -> Result<Vec<NeoPopulationMetricRow>, CudaPopulationError> {
        if !self.metrics_ready {
            return Err(CudaPopulationError::native(
                "read_metrics",
                STATUS_MISSING_UPLOAD,
            ));
        }
        let mut rows = vec![NeoPopulationMetricRow::default(); self.population];
        let mut written = 0_usize;
        let mut raw = RawReadback {
            rows: rows.as_mut_ptr(),
            capacity: rows.len(),
            written: &mut written,
        };
        // SAFETY: `rows` covers `capacity` elements and does not alias inputs.
        let status = unsafe { neoethos_gpu_cuda_population_read_metrics(self.handle, &mut raw) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("read_metrics", status));
        }
        if written != self.population {
            return Err(invalid(format!(
                "native metric readback wrote {written} rows, expected {}",
                self.population
            )));
        }
        Ok(rows)
    }

    /// Diagnostic-only event/outcome readback. Never call this inside a timed
    /// benchmark repetition: it is a full device-to-host copy that exists to
    /// localize a parity failure.
    pub fn read_diagnostics(&mut self) -> Result<PopulationDiagnostics, CudaPopulationError> {
        if !self.metrics_ready {
            return Err(CudaPopulationError::native(
                "read_diagnostics",
                STATUS_MISSING_UPLOAD,
            ));
        }
        let mut events = vec![NeoPopulationEvent::default(); self.emitted_events];
        let mut outcomes = vec![NeoPopulationOutcome::default(); self.emitted_events];
        let mut written = 0_usize;
        let mut raw = RawDiagnosticReadback {
            events: events.as_mut_ptr(),
            outcomes: outcomes.as_mut_ptr(),
            capacity: self.emitted_events,
            written: &mut written,
        };
        // SAFETY: both buffers cover `capacity` elements and do not alias.
        let status =
            unsafe { neoethos_gpu_cuda_population_read_diagnostics(self.handle, &mut raw) };
        if status != STATUS_OK {
            return Err(CudaPopulationError::native("read_diagnostics", status));
        }
        if written != self.emitted_events {
            return Err(invalid(format!(
                "native diagnostic readback wrote {written} events, expected {}",
                self.emitted_events
            )));
        }
        Ok(PopulationDiagnostics { events, outcomes })
    }
}

impl Drop for PopulationSession {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: the handle was produced by `create` and is destroyed once.
            unsafe { neoethos_gpu_cuda_population_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset_slices() -> (Vec<f64>, Vec<f32>, Vec<i64>, Vec<i8>) {
        (
            vec![1.0; 4],
            vec![0.5; 8],
            vec![0; 4],
            vec![0; 4 * SMC_SLOTS],
        )
    }

    #[test]
    fn zero_capacity_and_negative_device_are_rejected_before_ffi() {
        assert!(matches!(
            PopulationSession::create(0, 0),
            Err(CudaPopulationError::InvalidInput(_))
        ));
        assert!(matches!(
            PopulationSession::create(-1, 8),
            Err(CudaPopulationError::InvalidInput(_))
        ));
    }

    #[test]
    fn status_messages_cover_every_documented_code() {
        for status in [
            STATUS_UNSUPPORTED,
            STATUS_NULL_SESSION,
            STATUS_ABI_MISMATCH,
            STATUS_INVALID_ARGUMENT,
            STATUS_DEVICE_UNAVAILABLE,
            STATUS_ALLOCATION_FAILED,
            STATUS_TRANSFER_FAILED,
            STATUS_LAUNCH_FAILED,
            STATUS_EVENT_CAPACITY,
            STATUS_MISSING_UPLOAD,
            STATUS_READBACK_CAPACITY,
            STATUS_SYNC_FAILED,
            STATUS_UNKNOWN_EVENT,
            STATUS_DATASET_REUPLOAD,
        ] {
            assert_ne!(population_status_message(status), "unknown native status");
        }
        assert_eq!(population_status_message(-999), "unknown native status");
    }

    #[test]
    fn dataset_shape_errors_are_typed_not_undefined_behaviour() {
        let (prices, indicators, calendar, smc) = dataset_slices();
        let view = PopulationDatasetView {
            close: &prices,
            high: &prices,
            low: &prices[..3],
            indicators: &indicators,
            feature_count: 2,
            months: &calendar,
            days: &calendar,
            timestamps: &calendar,
            smc_rows: &smc,
            adaptive_base_pips: None,
        };
        let error = view.validate().unwrap_err();
        assert!(
            matches!(error, CudaPopulationError::InvalidInput(ref detail) if detail.contains("low"))
        );
    }

    #[test]
    fn non_finite_prices_are_rejected_before_ffi() {
        let (mut prices, indicators, calendar, smc) = dataset_slices();
        prices[2] = f64::NAN;
        let view = PopulationDatasetView {
            close: &prices,
            high: &prices,
            low: &prices,
            indicators: &indicators,
            feature_count: 2,
            months: &calendar,
            days: &calendar,
            timestamps: &calendar,
            smc_rows: &smc,
            adaptive_base_pips: None,
        };
        assert!(
            matches!(view.validate(), Err(CudaPopulationError::InvalidInput(detail)) if detail.contains("close[2]"))
        );
    }

    #[test]
    fn gene_csr_and_feature_bounds_are_validated_before_ffi() {
        let descriptors = vec![GeneDescriptor::default(); 2];
        let smc_flags = vec![0_i8; 2 * SMC_SLOTS];
        let smc_weights = [0.0_f32; SMC_SLOTS];
        let genes = PopulationGeneView {
            descriptors: &descriptors,
            offsets: &[0, 1, 2],
            indices: &[0, 7],
            weights: &[1.0, 1.0],
            stop_pips: &[10.0, 10.0],
            target_pips: &[20.0, 20.0],
            stop_vol_multipliers: &[0.0, 0.0],
            smc_flags: &smc_flags,
            smc_weights: &smc_weights,
            gate_threshold: 0.0,
            smc_gate_disabled: false,
        };
        assert!(
            matches!(genes.validate(4), Err(CudaPopulationError::InvalidInput(detail)) if detail.contains("feature 7"))
        );

        let broken = PopulationGeneView {
            offsets: &[0, 2, 1],
            ..genes
        };
        assert!(
            matches!(broken.validate(8), Err(CudaPopulationError::InvalidInput(detail)) if detail.contains("CSR"))
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn default_build_reports_runtime_unavailable_for_population_sessions() {
        match PopulationSession::create(0, 1024) {
            Err(error) => assert!(
                error.is_runtime_unavailable(),
                "a CUDA-free build must report an unavailable runtime, got {error}"
            ),
            Ok(_) => panic!("a CUDA-free build must not create a native population session"),
        }
    }
}
