//! Device self-report for the INDICATOR lane.
//!
//! `neoethos-search::eval_telemetry::device_summary` answers "did the card do
//! the population evaluation, or did something quietly fall back?" for the GA
//! lane. This module answers the same question for the feature-build lane —
//! the vector-ta indicator sweep — and it is deliberately shaped the same way:
//!
//!   * it records per-indicator wall time under a NAMED lane,
//!   * it prints EVEN WHEN EMPTY, so "0% GPU" is never left as an inference,
//!   * it records the REASON a frame did not use the card, by name, so a
//!     declared fallback can never be mistaken for a silent one.
//!
//! `neoethos-data` sits BELOW `neoethos-search` in the dependency graph, so
//! this cannot reuse that registry; it is an independent, self-contained one.
//!
//! Everything here compiles in the card-less build too. That is on purpose:
//! the summary must be able to say "this binary has no CUDA indicator lane
//! compiled in" rather than printing nothing at all.

use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Duration;

/// Which lane actually computed a frame's indicator columns, and — when it was
/// not the card — exactly why not.
///
/// Every variant other than `Gpu` is a REASON, not a shrug. There is no
/// "fallback" variant without a cause attached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndicatorLane {
    /// Ran on the card. `arch` is the compute capability the PTX was built
    /// for, as recorded by `build.rs`.
    Gpu { arch: String },
    /// This binary has no CUDA indicator lane compiled in (`gpu-cuda` off).
    /// The normal, expected state for a card-less build.
    CpuNoFeature,
    /// The feature is compiled in but the caller explicitly asked for CPU.
    CpuByPolicy,
    /// The feature is compiled in and the caller allowed the card, but the
    /// card could not serve this frame. `why` is the propagated device error.
    CpuDeviceUnavailable { why: String },
    /// The indicator has no device dispatch in vector-ta at all, so it stays
    /// on the CPU by construction. Enumerated, not discovered at run time.
    CpuIndicatorNotPortable,
}

impl IndicatorLane {
    pub fn is_gpu(&self) -> bool {
        matches!(self, IndicatorLane::Gpu { .. })
    }

    /// Short stable label for logs and for the summary table.
    pub fn label(&self) -> &'static str {
        match self {
            IndicatorLane::Gpu { .. } => "gpu",
            IndicatorLane::CpuNoFeature => "cpu:no-feature",
            IndicatorLane::CpuByPolicy => "cpu:by-policy",
            IndicatorLane::CpuDeviceUnavailable { .. } => "cpu:device-unavailable",
            IndicatorLane::CpuIndicatorNotPortable => "cpu:indicator-not-portable",
        }
    }
}

/// One frame's worth of indicator-lane accounting.
#[derive(Debug, Clone, Default)]
pub struct IndicatorRunSummary {
    /// Indicator ids that were swept on the device, in emission order.
    pub gpu_indicators: Vec<&'static str>,
    /// Indicator ids that stayed on the CPU, each with the reason.
    pub cpu_indicators: Vec<(&'static str, IndicatorLane)>,
    pub gpu_time: Duration,
    pub cpu_time: Duration,
    /// Number of `compute_cuda_device` calls issued. One per indicator per
    /// frame — the whole period sweep is a single launch.
    pub device_calls: u64,
    /// Host→device uploads. Should be ONE per frame: the OHLCV stays resident
    /// across every indicator. More than one means residency broke.
    pub uploads: u64,
    /// Set when the card ran: the numeric class of the device result relative
    /// to the CPU reference. vector-ta's device layer is f32-only while every
    /// CPU indicator is f64, so this is NEVER "identical" — see
    /// `hpc_ta::gpu_cpu_sma_sweep_parity` for the measured tolerance.
    pub precision: Option<&'static str>,
}

impl IndicatorRunSummary {
    pub fn gpu_fraction(&self) -> f64 {
        let denom = self.gpu_time.as_nanos() + self.cpu_time.as_nanos();
        if denom == 0 {
            return 0.0;
        }
        self.gpu_time.as_nanos() as f64 / denom as f64
    }
}

fn registry() -> &'static Mutex<Vec<IndicatorRunSummary>> {
    static REG: OnceLock<Mutex<Vec<IndicatorRunSummary>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(Vec::new()))
}

/// Record one frame. Called by `hpc_ta::compute_classic_ta_columns_with_policy`
/// on EVERY frame, GPU or not — a frame that never reaches the card still
/// records why.
pub fn record(summary: IndicatorRunSummary) {
    // A poisoned telemetry mutex must never take down a discovery run: the
    // measurement is not the product. Recover the guard and carry on.
    let mut guard = match registry().lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.push(summary);
}

/// True when this binary has a CUDA indicator lane compiled in at all.
/// Distinguishes "no card" from "card present, nothing used it".
pub const fn indicator_gpu_lane_compiled() -> bool {
    cfg!(feature = "gpu-cuda")
}

/// The arch of the PTX embedded in the vector-ta fatbin — the one the driver
/// JITs on a card newer than anything we compiled SASS for.
///
/// RE-EXPORTED, not recomputed. This used to be `env!("NEOETHOS_VECTOR_TA_PTX_ARCH")`,
/// resolved by `neoethos-data/build.rs` from `CUDA_ARCHS`/`CUDA_ARCH`/`nvidia-smi`
/// — i.e. a fact about the BUILD HOST, printed in the diagnostic that exists to
/// explain a mismatch on the RUN host. A portable binary built on a 4090
/// reported `sm_89` in the message meant to explain why an sm_86 card failed.
/// vector-ta records what it actually compiled; there is now one answer.
#[cfg(feature = "gpu-cuda")]
pub const VECTOR_TA_PTX_ARCH: &str = vector_ta::cuda::module_loader::COMPILED_PTX_ARCH;
/// `"none"` in a card-less build: there is no fatbin and no PTX.
#[cfg(not(feature = "gpu-cuda"))]
pub const VECTOR_TA_PTX_ARCH: &str = "none";

/// The SASS architectures the vector-ta fatbin actually contains, e.g.
/// `"sm_80,sm_86,sm_89,sm_90"`. Same single-source rule as
/// [`VECTOR_TA_PTX_ARCH`].
#[cfg(feature = "gpu-cuda")]
pub const VECTOR_TA_ARCHS: &str = vector_ta::cuda::module_loader::COMPILED_ARCHS;
/// `"none"` in a card-less build.
#[cfg(not(feature = "gpu-cuda"))]
pub const VECTOR_TA_ARCHS: &str = "none";

/// Print the indicator-lane device split. Prints EVEN WHEN EMPTY, exactly like
/// `eval_telemetry::device_summary`: a run that recorded no indicator frames
/// still states whether a CUDA lane exists in this binary, so a 0% reading is
/// always attributable.
pub fn indicator_device_summary() {
    let rows: Vec<IndicatorRunSummary> = match registry().lock() {
        Ok(g) => g.clone(),
        Err(poisoned) => poisoned.into_inner().clone(),
    };

    let lane_compiled = indicator_gpu_lane_compiled();

    if rows.is_empty() {
        tracing::info!(
            target: "neoethos_data::indicator_telemetry",
            gpu_lane_compiled = lane_compiled,
            ptx_arch = VECTOR_TA_PTX_ARCH,
            frames = 0,
            "indicator-lane device split: no frames recorded"
        );
        return;
    }

    let frames = rows.len() as u64;
    let gpu_frames = rows.iter().filter(|r| r.device_calls > 0).count() as u64;
    let gpu_nanos: u128 = rows.iter().map(|r| r.gpu_time.as_nanos()).sum();
    let cpu_nanos: u128 = rows.iter().map(|r| r.cpu_time.as_nanos()).sum();
    let device_calls: u64 = rows.iter().map(|r| r.device_calls).sum();
    let uploads: u64 = rows.iter().map(|r| r.uploads).sum();
    let denom = (gpu_nanos + cpu_nanos) as f64;
    let gpu_pct = if denom > 0.0 {
        100.0 * gpu_nanos as f64 / denom
    } else {
        0.0
    };

    tracing::info!(
        target: "neoethos_data::indicator_telemetry",
        gpu_lane_compiled = lane_compiled,
        ptx_arch = VECTOR_TA_PTX_ARCH,
        frames,
        gpu_frames,
        gpu_pct = format!("{gpu_pct:.1}"),
        gpu_s = format!("{:.1}", gpu_nanos as f64 / 1e9),
        cpu_s = format!("{:.1}", cpu_nanos as f64 / 1e9),
        device_calls,
        uploads,
        precision = rows.iter().find_map(|r| r.precision).unwrap_or("f64 (cpu reference)"),
        "indicator-lane device split"
    );

    // Second pass: name every CPU reason that occurred, with a count. This is
    // the line that makes a declared fallback un-mistakable for a silent one —
    // if the card was supposed to run and did not, the reason is printed with
    // the indicator that hit it.
    let mut reasons: Vec<(String, u64)> = Vec::new();
    for row in &rows {
        for (_id, lane) in &row.cpu_indicators {
            let key = match lane {
                IndicatorLane::CpuDeviceUnavailable { why } => {
                    format!("{}: {why}", lane.label())
                }
                other => other.label().to_string(),
            };
            match reasons.iter_mut().find(|(k, _)| *k == key) {
                Some((_, n)) => *n += 1,
                None => reasons.push((key, 1)),
            }
        }
    }
    for (reason, count) in reasons {
        tracing::info!(
            target: "neoethos_data::indicator_telemetry",
            reason = %reason,
            indicator_columns = count,
            "indicator-lane CPU reason"
        );
    }
}

/// Drop everything recorded so far. Test-support only — a long-lived process
/// that summarises per run needs a clean slate between runs.
pub fn reset() {
    let mut guard = match registry().lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    };
    guard.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_labels_are_distinct_and_stable() {
        // These strings end up in operator-facing logs and in the reason
        // aggregation above; a collision would silently merge two different
        // causes into one row.
        let labels = [
            IndicatorLane::Gpu {
                arch: "sm_86".into(),
            }
            .label(),
            IndicatorLane::CpuNoFeature.label(),
            IndicatorLane::CpuByPolicy.label(),
            IndicatorLane::CpuDeviceUnavailable { why: "x".into() }.label(),
            IndicatorLane::CpuIndicatorNotPortable.label(),
        ];
        let mut sorted = labels.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), labels.len(), "lane labels must be distinct");
    }

    #[test]
    fn only_gpu_variant_reports_gpu() {
        assert!(
            IndicatorLane::Gpu {
                arch: "sm_86".into()
            }
            .is_gpu()
        );
        assert!(!IndicatorLane::CpuNoFeature.is_gpu());
        assert!(!IndicatorLane::CpuByPolicy.is_gpu());
        assert!(!IndicatorLane::CpuDeviceUnavailable { why: "x".into() }.is_gpu());
        assert!(!IndicatorLane::CpuIndicatorNotPortable.is_gpu());
    }

    #[test]
    fn ptx_arch_is_none_without_the_gpu_feature() {
        // A card-less build must still define the constant (so the cfg-gated
        // module compiles) and must define it as the sentinel, never as a
        // plausible-looking arch.
        if !indicator_gpu_lane_compiled() {
            assert_eq!(VECTOR_TA_PTX_ARCH, "none");
            assert_eq!(VECTOR_TA_ARCHS, "none");
        }
    }

    /// The arch strings must describe the KERNELS, not the build host. The
    /// only way to keep that true is for them to have exactly one source, and
    /// the only way to pin THAT card-lessly is to assert the sentinel is not
    /// something a host probe could have produced.
    #[test]
    fn arch_strings_are_never_a_host_probe_result() {
        if !indicator_gpu_lane_compiled() {
            for s in [VECTOR_TA_PTX_ARCH, VECTOR_TA_ARCHS] {
                assert!(
                    !s.contains("sm_") && !s.contains("compute_"),
                    "a card-less build reported {s:?}, which reads like a resolved arch — \
                     this constant must be re-exported from vector-ta, never recomputed \
                     from CUDA_ARCH / nvidia-smi"
                );
            }
        }
    }

    #[test]
    fn gpu_fraction_is_zero_when_nothing_ran() {
        let s = IndicatorRunSummary::default();
        assert_eq!(s.gpu_fraction(), 0.0);
    }
}
