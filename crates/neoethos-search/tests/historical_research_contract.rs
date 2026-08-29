use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use neoethos_core::execution::BudgetedCpuExecutor;
use neoethos_core::execution_budget::{CpuPermitBroker, CpuPermitRequest, WorkerLimit};
use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvPublishRequest,
    CanonicalTimeframe, CanonicalVolumeRef, Ohlcv, publish_canonical_ohlcv_generation,
};
use neoethos_search::{
    ExactCanonicalSeries, Gene, HistoricalCandidateDistanceSourceV1,
    HistoricalCandidateFailurePolicyV1, HistoricalCandidateResultStatusV1,
    HistoricalCandidateScanError, HistoricalCandidateScanKindV1, HistoricalCandidateScanRequestV2,
    HistoricalCandidateScanResultV2, HistoricalResearchAccountingV1,
    HistoricalResearchArtifactClassV1, HistoricalResearchBackendV1,
    HistoricalResearchEntryReferenceV1, HistoricalResearchError, HistoricalResearchGeometryV1,
    HistoricalResearchIntrabarAmbiguityV1, HistoricalResearchPriceBasisV1,
    HistoricalResearchPromotionEligibilityV1, HistoricalResearchRequestV2,
    HistoricalResearchSignalTimingV1, HistoricalResearchSignalV1, run_historical_research_v2,
    scan_historical_candidates_v2, signals_for_gene,
};

static NEXT_TEMP_ROOT: AtomicU64 = AtomicU64::new(0);

fn scan_with_width(
    request: HistoricalCandidateScanRequestV2<'_, '_>,
    width: usize,
) -> Result<HistoricalCandidateScanResultV2, HistoricalCandidateScanError> {
    let limit = WorkerLimit::new(width).expect("positive test worker width");
    let broker = CpuPermitBroker::new(limit);
    let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), limit);
    let lease = broker
        .acquire(CpuPermitRequest::local(limit))
        .expect("acquire isolated historical scan lease");
    scan_historical_candidates_v2(request, &executor, lease.into_transfer())
}

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        let unique = NEXT_TEMP_ROOT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "neoethos-historical-research-{label}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create isolated canonical store");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.0) {
            eprintln!(
                "ERROR historical-research test cleanup failed for {}: {error}",
                self.0.display()
            );
        }
    }
}

fn publish_bars(
    root: &Path,
    namespace: &str,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
) -> CanonicalDatasetIdentity {
    let identity = CanonicalDatasetIdentity::external(
        namespace,
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid external identity");
    let rows = close.len();
    let timestamp = (0..rows)
        .map(|row| 1_704_067_200_000_i64 + row as i64 * 60_000)
        .collect::<Vec<_>>();
    let ohlcv = Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume: None,
    };
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.historical-research-contract.v1",
        identity.canonical_bytes(),
    )
    .expect("valid test provenance");
    publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity: &identity,
        expected_generation: None,
        provenance: &provenance,
        ohlcv: &ohlcv,
        volume: CanonicalVolumeRef::Absent,
        rows_per_chunk: 128,
    })
    .expect("publish canonical test generation");
    identity
}

fn publish_contract_bars(root: &Path, namespace: &str, scale: f64) -> CanonicalDatasetIdentity {
    let rows = 512_usize;
    let mut open = vec![100.0 * scale; rows];
    let mut high = vec![100.5 * scale; rows];
    let mut low = vec![99.5 * scale; rows];
    let mut close = vec![100.0 * scale; rows];

    // Signal on bar 0 enters at bar 1 open. Both barriers occur on bar 1,
    // therefore the explicit conservative ambiguity rule realizes -1R.
    open[1] = 100.0 * scale;
    high[1] = 102.5 * scale;
    low[1] = 98.5 * scale;
    close[1] = 100.5 * scale;

    // Signal on the now-closed bar 1 enters at bar 2 open and reaches +2R.
    open[2] = 100.0 * scale;
    high[2] = 102.5 * scale;
    low[2] = 99.5 * scale;
    close[2] = 102.0 * scale;

    publish_bars(root, namespace, open, high, low, close)
}

fn publish_no_lookahead_bars(root: &Path) -> CanonicalDatasetIdentity {
    let rows = 512_usize;
    let open = vec![100.0; rows];
    let mut high = vec![100.5; rows];
    let mut low = vec![99.5; rows];
    let mut close = vec![100.0; rows];

    // The signal bar itself breaches the prospective stop. A same-bar engine
    // would record -1R, but that bar was already closed when the signal arose.
    low[0] = 98.0;

    // The only admissible entry is the next bar's open; that bar reaches +2R.
    high[1] = 102.5;
    low[1] = 99.5;
    close[1] = 102.0;

    publish_bars(
        root,
        "historical-research-no-lookahead",
        open,
        high,
        low,
        close,
    )
}

fn single_signal_geometry_result(
    label: &str,
    signal: HistoricalResearchSignalV1,
    distance: f64,
    stop_multiple: f64,
    target_multiple: f64,
) -> Result<(), HistoricalResearchError> {
    let root = TempRoot::new(label);
    let identity = publish_contract_bars(root.path(), label, 1.0);
    let selected = ExactCanonicalSeries::open(root.path(), identity)
        .expect("select exact barrier fixture")
        .load_search_input(&[])
        .expect("load exact barrier fixture");
    let input = selected
        .as_run_input()
        .expect("bind exact barrier fixture receipt");
    let rows = input.ohlcv().len();
    let mut signals = vec![HistoricalResearchSignalV1::Flat; rows];
    signals[0] = signal;
    let distances = vec![distance; rows];

    run_historical_research_v2(HistoricalResearchRequestV2 {
        input: &input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        signal_semantic_id: "contract.collapsed-barrier-direction.v1",
        signals: &signals,
        geometry: HistoricalResearchGeometryV1::PriceNativeVolatilityDistance {
            distance_semantic_id: "contract.collapsed-barrier-distance.v1",
            distance_by_signal_bar: &distances,
            stop_multiple,
            target_multiple,
        },
    })
    .map(|_| ())
}

#[test]
fn cpu_research_is_receipt_bound_gross_r_with_causal_entries() {
    let root = TempRoot::new("gross-r");
    let identity = publish_contract_bars(root.path(), "historical-research-contract", 1.0);
    let selected = ExactCanonicalSeries::open(root.path(), identity)
        .expect("select exact canonical generation")
        .load_search_input(&[])
        .expect("load exact canonical search input");
    let input = selected.as_run_input().expect("bind receipt to values");
    let rows = input.ohlcv().len();
    let mut signals = vec![HistoricalResearchSignalV1::Flat; rows];
    signals[0] = HistoricalResearchSignalV1::Long;
    signals[1] = HistoricalResearchSignalV1::Long;
    let distances = vec![1.0; rows];

    let artifact = run_historical_research_v2(HistoricalResearchRequestV2 {
        input: &input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        signal_semantic_id: "contract.direction.v1",
        signals: &signals,
        geometry: HistoricalResearchGeometryV1::PriceNativeVolatilityDistance {
            distance_semantic_id: "contract.closed-bar-range.v1",
            distance_by_signal_bar: &distances,
            stop_multiple: 1.0,
            target_multiple: 2.0,
        },
    })
    .expect("gross-R research must not require broker financial truth");

    assert_eq!(artifact.schema_version(), 2);
    assert_eq!(
        artifact.artifact_class(),
        HistoricalResearchArtifactClassV1::ResearchOnly
    );
    assert_eq!(
        artifact.promotion_eligibility(),
        HistoricalResearchPromotionEligibilityV1::NotPromotionEligible
    );
    assert_eq!(artifact.backend(), HistoricalResearchBackendV1::CpuOnly);
    assert_eq!(
        artifact.accounting(),
        HistoricalResearchAccountingV1::GrossReferenceR
    );
    assert_eq!(
        artifact.intrabar_ambiguity(),
        HistoricalResearchIntrabarAmbiguityV1::StopBeforeTarget
    );
    assert_eq!(artifact.scope().receipt(), input.receipt());
    let contract = artifact.execution_contract();
    assert_eq!(contract.schema_version(), 1);
    assert_eq!(
        contract.signal_timing(),
        HistoricalResearchSignalTimingV1::PriorClosedBar
    );
    assert_eq!(
        contract.entry_reference(),
        HistoricalResearchEntryReferenceV1::NextBarOpen
    );
    assert_eq!(
        contract.price_basis(),
        HistoricalResearchPriceBasisV1::CanonicalReferenceOhlc
    );
    assert_eq!(
        contract.signal_source().semantic_id(),
        "contract.direction.v1"
    );
    assert_eq!(
        contract.geometry().distance_source().semantic_id(),
        "contract.closed-bar-range.v1"
    );
    assert_eq!(contract.geometry().stop_multiple(), 1.0);
    assert_eq!(contract.geometry().target_multiple(), 2.0);
    assert_eq!(artifact.evidence_identity_sha256().len(), 64);
    artifact
        .validate()
        .expect("artifact contract and identity validate");

    let serialized = serde_json::to_value(&artifact).expect("serialize research artifact");
    assert_eq!(
        serialized["execution_contract"]["signal_timing"],
        "prior_closed_bar"
    );
    assert_eq!(
        serialized["execution_contract"]["entry_reference"],
        "next_bar_open"
    );
    assert_eq!(
        serialized["execution_contract"]["price_basis"],
        "canonical_reference_ohlc"
    );

    let metrics = artifact.metrics();
    assert_eq!(metrics.trade_count(), 2);
    assert_eq!(metrics.gross_r_expectancy(), Some(0.5));
    assert_eq!(metrics.gross_r_max_drawdown(), 1.0);
    assert_eq!(metrics.gross_r_win_rate(), Some(0.5));
    assert_eq!(metrics.gross_r_payoff(), Some(2.0));

    let changed_multiplier = run_historical_research_v2(HistoricalResearchRequestV2 {
        input: &input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        signal_semantic_id: "contract.direction.v1",
        signals: &signals,
        geometry: HistoricalResearchGeometryV1::PriceNativeVolatilityDistance {
            distance_semantic_id: "contract.closed-bar-range.v1",
            distance_by_signal_bar: &distances,
            stop_multiple: 1.0,
            target_multiple: 3.0,
        },
    })
    .expect("changed multiplier remains valid research");
    assert_ne!(
        artifact.evidence_identity_sha256(),
        changed_multiplier.evidence_identity_sha256(),
        "a value-affecting multiplier must change evidence identity"
    );

    let mut changed_distances = distances.clone();
    changed_distances[7] = 1.25;
    let changed_distance = run_historical_research_v2(HistoricalResearchRequestV2 {
        input: &input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        signal_semantic_id: "contract.direction.v1",
        signals: &signals,
        geometry: HistoricalResearchGeometryV1::PriceNativeVolatilityDistance {
            distance_semantic_id: "contract.closed-bar-range.v1",
            distance_by_signal_bar: &changed_distances,
            stop_multiple: 1.0,
            target_multiple: 2.0,
        },
    })
    .expect("changed finite distance remains valid research");
    assert_ne!(
        artifact.evidence_identity_sha256(),
        changed_distance.evidence_identity_sha256(),
        "distance content must be evidence-bound even on a flat row"
    );

    let mut changed_signals = signals.clone();
    changed_signals[rows - 1] = HistoricalResearchSignalV1::Short;
    let changed_signal = run_historical_research_v2(HistoricalResearchRequestV2 {
        input: &input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        signal_semantic_id: "contract.direction.v1",
        signals: &changed_signals,
        geometry: HistoricalResearchGeometryV1::PriceNativeVolatilityDistance {
            distance_semantic_id: "contract.closed-bar-range.v1",
            distance_by_signal_bar: &distances,
            stop_multiple: 1.0,
            target_multiple: 2.0,
        },
    })
    .expect("changed terminal signal remains valid research");
    assert_ne!(
        artifact.evidence_identity_sha256(),
        changed_signal.evidence_identity_sha256(),
        "signal content must be evidence-bound even when no later bar can execute it"
    );
}

#[test]
fn unsupported_lanes_and_nonfinite_geometry_fail_before_evaluation() {
    let root = TempRoot::new("fail-closed");
    let identity = publish_contract_bars(root.path(), "historical-research-fail-closed", 1.0);
    let selected = ExactCanonicalSeries::open(root.path(), identity)
        .expect("select exact canonical generation")
        .load_search_input(&[])
        .expect("load exact canonical search input");
    let input = selected.as_run_input().expect("bind receipt to values");
    let rows = input.ohlcv().len();
    let signals = vec![HistoricalResearchSignalV1::Flat; rows];
    let mut invalid_distances = vec![1.0; rows];
    invalid_distances[7] = f64::NAN;

    for backend in [
        HistoricalResearchBackendV1::Auto,
        HistoricalResearchBackendV1::GpuOnly,
    ] {
        let error = run_historical_research_v2(HistoricalResearchRequestV2 {
            input: &input,
            backend,
            signal_semantic_id: "contract.direction.v1",
            signals: &signals,
            geometry: HistoricalResearchGeometryV1::PriceNativeVolatilityDistance {
                distance_semantic_id: "contract.closed-bar-range.v1",
                distance_by_signal_bar: &invalid_distances,
                stop_multiple: 1.0,
                target_multiple: 2.0,
            },
        })
        .expect_err("non-CPU research must be rejected before inspecting values");
        assert_eq!(
            error,
            HistoricalResearchError::UnsupportedBackend { requested: backend }
        );
    }

    let fixed_pip_error = run_historical_research_v2(HistoricalResearchRequestV2 {
        input: &input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        signal_semantic_id: "contract.direction.v1",
        signals: &[],
        geometry: HistoricalResearchGeometryV1::FixedPips {
            stop_pips: 10.0,
            target_pips: 20.0,
        },
    })
    .expect_err("fixed-pip geometry must be rejected before inspecting signal shape");
    assert_eq!(
        fixed_pip_error,
        HistoricalResearchError::UnsupportedGeometry
    );

    let nonfinite_error = run_historical_research_v2(HistoricalResearchRequestV2 {
        input: &input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        signal_semantic_id: "contract.direction.v1",
        signals: &signals,
        geometry: HistoricalResearchGeometryV1::PriceNativeVolatilityDistance {
            distance_semantic_id: "contract.closed-bar-range.v1",
            distance_by_signal_bar: &invalid_distances,
            stop_multiple: 1.0,
            target_multiple: 2.0,
        },
    })
    .expect_err("non-finite price-native geometry must fail closed");
    assert_eq!(
        nonfinite_error,
        HistoricalResearchError::NonFinite {
            field: "distance_by_signal_bar",
            index: Some(7),
        }
    );
}

#[test]
fn positive_source_distance_that_underflows_after_multiplication_is_rejected() {
    let error = single_signal_geometry_result(
        "resolved-distance-underflow",
        HistoricalResearchSignalV1::Long,
        f64::from_bits(1),
        0.5,
        1.0,
    )
    .expect_err("a positive source distance must not resolve to zero risk");

    assert_eq!(
        error,
        HistoricalResearchError::NonPositive {
            field: "resolved_stop_distance",
            index: Some(0),
        }
    );
}

#[test]
fn long_stop_that_rounds_back_to_the_entry_price_is_rejected() {
    let entry = 100.0_f64;
    let one_ulp = f64::from_bits(entry.to_bits() + 1) - entry;
    let error = single_signal_geometry_result(
        "collapsed-long-stop",
        HistoricalResearchSignalV1::Long,
        one_ulp,
        0.25,
        1.0,
    )
    .expect_err("a long stop must be strictly below its entry after rounding");

    assert_eq!(
        error,
        HistoricalResearchError::NonPositive {
            field: "resolved_stop_barrier_distance",
            index: Some(0),
        }
    );
}

#[test]
fn short_target_that_rounds_back_to_the_entry_price_is_rejected() {
    let entry = 100.0_f64;
    let one_ulp = f64::from_bits(entry.to_bits() + 1) - entry;
    let error = single_signal_geometry_result(
        "collapsed-short-target",
        HistoricalResearchSignalV1::Short,
        one_ulp,
        1.0,
        0.25,
    )
    .expect_err("a short target must be strictly below its entry after rounding");

    assert_eq!(
        error,
        HistoricalResearchError::NonPositive {
            field: "resolved_target_barrier_distance",
            index: Some(0),
        }
    );
}

#[test]
fn gross_r_metrics_are_invariant_to_consistent_price_scaling() {
    let unscaled_root = TempRoot::new("scale-one");
    let scaled_root = TempRoot::new("scale-ten");
    let unscaled_identity =
        publish_contract_bars(unscaled_root.path(), "historical-research-scale-one", 1.0);
    let scaled_identity =
        publish_contract_bars(scaled_root.path(), "historical-research-scale-ten", 10.0);
    let unscaled_selected = ExactCanonicalSeries::open(unscaled_root.path(), unscaled_identity)
        .expect("select unscaled generation")
        .load_search_input(&[])
        .expect("load unscaled input");
    let scaled_selected = ExactCanonicalSeries::open(scaled_root.path(), scaled_identity)
        .expect("select scaled generation")
        .load_search_input(&[])
        .expect("load scaled input");
    let unscaled = unscaled_selected
        .as_run_input()
        .expect("bind unscaled input");
    let scaled = scaled_selected.as_run_input().expect("bind scaled input");
    let rows = unscaled.ohlcv().len();
    let mut signals = vec![HistoricalResearchSignalV1::Flat; rows];
    signals[0] = HistoricalResearchSignalV1::Long;
    signals[1] = HistoricalResearchSignalV1::Long;
    let unscaled_distances = vec![1.0; rows];
    let scaled_distances = vec![10.0; rows];

    let run = |input, distances: &[f64]| {
        run_historical_research_v2(HistoricalResearchRequestV2 {
            input,
            backend: HistoricalResearchBackendV1::CpuOnly,
            signal_semantic_id: "contract.direction.v1",
            signals: &signals,
            geometry: HistoricalResearchGeometryV1::PriceNativeVolatilityDistance {
                distance_semantic_id: "contract.closed-bar-range.v1",
                distance_by_signal_bar: distances,
                stop_multiple: 1.0,
                target_multiple: 2.0,
            },
        })
        .expect("scaled research input is valid")
    };
    let unscaled_artifact = run(&unscaled, &unscaled_distances);
    let scaled_artifact = run(&scaled, &scaled_distances);

    assert_eq!(unscaled_artifact.metrics(), scaled_artifact.metrics());
}

#[test]
fn signal_bar_cannot_trigger_an_exit_before_next_bar_open_entry() {
    let root = TempRoot::new("no-lookahead");
    let identity = publish_no_lookahead_bars(root.path());
    let selected = ExactCanonicalSeries::open(root.path(), identity)
        .expect("select exact no-lookahead generation")
        .load_search_input(&[])
        .expect("load exact no-lookahead input");
    let input = selected.as_run_input().expect("bind receipt to values");
    let rows = input.ohlcv().len();
    let mut signals = vec![HistoricalResearchSignalV1::Flat; rows];
    signals[0] = HistoricalResearchSignalV1::Long;
    let distances = vec![1.0; rows];

    let artifact = run_historical_research_v2(HistoricalResearchRequestV2 {
        input: &input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        signal_semantic_id: "contract.direction.v1",
        signals: &signals,
        geometry: HistoricalResearchGeometryV1::PriceNativeVolatilityDistance {
            distance_semantic_id: "contract.closed-bar-range.v1",
            distance_by_signal_bar: &distances,
            stop_multiple: 1.0,
            target_multiple: 2.0,
        },
    })
    .expect("causal research run");

    assert_eq!(artifact.metrics().trade_count(), 1);
    assert_eq!(artifact.metrics().gross_r_expectancy(), Some(2.0));
}

fn fully_valid_varying_feature(input: &neoethos_search::CanonicalSearchRunInputV2<'_>) -> usize {
    for feature in 0..input.features().n_features() {
        let column = input
            .features()
            .feature_column(feature)
            .expect("read candidate feature column");
        if !column.validity.iter().all(|validity| validity.is_valid())
            || !column.values.iter().all(|value| value.is_finite())
        {
            continue;
        }
        let min = column.values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = column
            .values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if min < max {
            return feature;
        }
    }
    panic!("canonical test input has no fully-valid varying feature column");
}

fn distinct_signal_genes(input: &neoethos_search::CanonicalSearchRunInputV2<'_>) -> [Gene; 2] {
    let feature = fully_valid_varying_feature(input);
    let column = input
        .features()
        .feature_column(feature)
        .expect("read selected feature");
    let min = column.values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = column
        .values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let midpoint = min + (max - min) * 0.5;
    let margin = (max - min).max(f64::EPSILON);
    [
        Gene {
            indices: vec![feature],
            weights: vec![1.0],
            long_threshold: midpoint,
            short_threshold: min - margin,
            ..Default::default()
        },
        Gene {
            indices: vec![feature],
            weights: vec![-1.0],
            long_threshold: -min + margin,
            short_threshold: -midpoint,
            ..Default::default()
        },
    ]
}

fn canonical_range_distances(input: &neoethos_search::CanonicalSearchRunInputV2<'_>) -> Vec<f64> {
    input
        .ohlcv()
        .high
        .iter()
        .zip(&input.ohlcv().low)
        .map(|(high, low)| high - low)
        .collect()
}

#[test]
fn candidate_scan_preserves_typed_warmup_and_gap_cells_as_ineligible_rows() {
    let root = TempRoot::new("candidate-validity");
    let identity = publish_contract_bars(root.path(), "historical-candidate-validity", 1.0);
    let selected = ExactCanonicalSeries::open(root.path(), identity)
        .expect("select exact validity generation")
        .load_search_input(&[])
        .expect("load exact validity input");
    let input = selected.as_run_input().expect("bind validity input");

    let (feature, invalid_rows, min, max) = (0..input.features().n_features())
        .find_map(|feature| {
            let column = input
                .features()
                .feature_column(feature)
                .expect("read validity candidate column");
            let invalid_rows = column
                .validity
                .iter()
                .enumerate()
                .filter_map(|(row, validity)| (!validity.is_valid()).then_some(row))
                .collect::<Vec<_>>();
            let valid_values = column
                .values
                .iter()
                .zip(&column.validity)
                .filter_map(|(value, validity)| validity.is_valid().then_some(*value))
                .collect::<Vec<_>>();
            let min = valid_values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = valid_values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            (!invalid_rows.is_empty()
                && valid_values.len() >= 32
                && min.is_finite()
                && max.is_finite()
                && min < max)
                .then_some((feature, invalid_rows, min, max))
        })
        .expect("fixture has a varying feature with typed warmup/gap rows");
    let margin = (max - min).max(f64::EPSILON);
    let candidate = Gene {
        indices: vec![feature],
        weights: vec![1.0],
        long_threshold: min + (max - min) * 0.5,
        short_threshold: min - margin,
        ..Default::default()
    };

    let raw_signals = signals_for_gene(input.features(), &candidate)
        .expect("typed validity is supported by the real signal generator");
    assert!(
        invalid_rows.iter().all(|row| raw_signals[*row] == 0),
        "warmup/gap cells must be ineligible rather than numeric zero inputs"
    );
    assert!(
        raw_signals.iter().any(|signal| *signal != 0),
        "valid causal segment must still produce real signals"
    );

    let distances = canonical_range_distances(&input);
    let receipt_sha256 = input.receipt().identity_sha256().expect("receipt identity");
    let result = scan_with_width(
        HistoricalCandidateScanRequestV2 {
            input: &input,
            backend: HistoricalResearchBackendV1::CpuOnly,
            candidates: &[candidate],
            failure_policy: HistoricalCandidateFailurePolicyV1::FailEntireScan,
            distance_source: HistoricalCandidateDistanceSourceV1 {
                receipt_sha256: &receipt_sha256,
                semantic_id: "canonical.reference-ohlc.range.v1",
                values: &distances,
            },
            stop_multiple: 1.0,
            target_multiple: 2.0,
        },
        2,
    )
    .expect("scan must preserve typed invalid cells instead of rejecting the candidate");

    assert_eq!(result.results().len(), 1);
    assert_eq!(
        result.results()[0].status(),
        HistoricalCandidateResultStatusV1::Evaluated
    );
}

#[test]
fn explicit_gene_candidates_are_signal_generated_evaluated_and_ranked_stably() {
    let root = TempRoot::new("candidate-scan");
    let identity = publish_contract_bars(root.path(), "historical-candidate-scan", 1.0);
    let selected = ExactCanonicalSeries::open(root.path(), identity)
        .expect("select exact candidate-scan generation")
        .load_search_input(&[])
        .expect("load exact candidate-scan input");
    let input = selected.as_run_input().expect("bind receipt to values");
    let candidates = distinct_signal_genes(&input);
    let distances = canonical_range_distances(&input);
    let receipt_sha256 = input.receipt().identity_sha256().expect("receipt identity");

    let run = || {
        scan_with_width(
            HistoricalCandidateScanRequestV2 {
                input: &input,
                backend: HistoricalResearchBackendV1::CpuOnly,
                candidates: &candidates,
                failure_policy: HistoricalCandidateFailurePolicyV1::FailEntireScan,
                distance_source: HistoricalCandidateDistanceSourceV1 {
                    receipt_sha256: &receipt_sha256,
                    semantic_id: "canonical.reference-ohlc.range.v1",
                    values: &distances,
                },
                stop_multiple: 1.0,
                target_multiple: 2.0,
            },
            2,
        )
        .expect("scan explicit real-gene candidates")
    };
    let first = run();
    let second = run();

    assert_eq!(
        first.search_kind(),
        HistoricalCandidateScanKindV1::ExplicitOrderedCandidateScan
    );
    assert_eq!(first.contract().schema_version(), 2);
    assert_eq!(
        first.contract().ranking_policy_id(),
        "gross-r-expectancy-desc_drawdown-asc_win-rate-desc_payoff-desc_trade-count-desc_candidate-identity-asc.v1"
    );
    assert_eq!(
        first.artifact_class(),
        HistoricalResearchArtifactClassV1::ResearchOnly
    );
    assert_eq!(
        first.promotion_eligibility(),
        HistoricalResearchPromotionEligibilityV1::NotPromotionEligible
    );
    assert_eq!(first.backend(), HistoricalResearchBackendV1::CpuOnly);
    assert_eq!(first.results().len(), 2);
    assert_eq!(first.ranking().len(), 2);
    assert_eq!(
        first.search_identity_sha256(),
        second.search_identity_sha256()
    );
    assert_eq!(first.ranking(), second.ranking());
    assert_eq!(
        first.best_candidate_identity_sha256(),
        first
            .ranking()
            .first()
            .map(|entry| entry.candidate_identity_sha256())
    );

    let first_result = &first.results()[0];
    let second_result = &first.results()[1];
    assert_ne!(
        first_result.candidate_identity_sha256(),
        second_result.candidate_identity_sha256()
    );
    assert_ne!(
        first_result
            .signal_identity_sha256()
            .expect("first candidate signal identity"),
        second_result
            .signal_identity_sha256()
            .expect("second candidate signal identity")
    );
    assert_ne!(
        first_result
            .artifact()
            .expect("first candidate artifact")
            .evidence_identity_sha256(),
        second_result
            .artifact()
            .expect("second candidate artifact")
            .evidence_identity_sha256()
    );
    assert!(first.results().iter().all(|result| matches!(
        result.status(),
        HistoricalCandidateResultStatusV1::Evaluated
    )));
}

#[test]
fn candidate_scan_artifact_is_byte_identical_across_budgeted_worker_widths() {
    let root = TempRoot::new("candidate-width-parity");
    let identity = publish_contract_bars(root.path(), "historical-candidate-width", 1.0);
    let selected = ExactCanonicalSeries::open(root.path(), identity)
        .expect("select exact width-parity generation")
        .load_search_input(&[])
        .expect("load exact width-parity input");
    let input = selected.as_run_input().expect("bind width-parity receipt");
    let candidates = distinct_signal_genes(&input);
    let distances = canonical_range_distances(&input);
    let receipt_sha256 = input.receipt().identity_sha256().expect("receipt identity");
    let request = || HistoricalCandidateScanRequestV2 {
        input: &input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        candidates: &candidates,
        failure_policy: HistoricalCandidateFailurePolicyV1::FailEntireScan,
        distance_source: HistoricalCandidateDistanceSourceV1 {
            receipt_sha256: &receipt_sha256,
            semantic_id: "canonical.reference-ohlc.range.v1",
            values: &distances,
        },
        stop_multiple: 1.0,
        target_multiple: 2.0,
    };

    let width_one = scan_with_width(request(), 1).expect("one-worker scan");
    let width_four = scan_with_width(request(), 4).expect("four-worker scan");

    assert_eq!(
        serde_json::to_vec(&width_one).expect("serialize one-worker evidence"),
        serde_json::to_vec(&width_four).expect("serialize four-worker evidence"),
        "worker width must not enter or perturb canonical search evidence"
    );
}

#[test]
fn candidate_scan_rejects_foreign_receipt_and_duplicate_or_invalid_gene_identity() {
    let selected_root = TempRoot::new("candidate-selected");
    let foreign_root = TempRoot::new("candidate-foreign");
    let selected_identity =
        publish_contract_bars(selected_root.path(), "historical-candidate-selected", 1.0);
    let foreign_identity =
        publish_contract_bars(foreign_root.path(), "historical-candidate-foreign", 1.0);
    let selected = ExactCanonicalSeries::open(selected_root.path(), selected_identity)
        .expect("select chosen generation")
        .load_search_input(&[])
        .expect("load chosen generation");
    let foreign = ExactCanonicalSeries::open(foreign_root.path(), foreign_identity)
        .expect("select foreign generation")
        .load_search_input(&[])
        .expect("load foreign generation");
    let input = selected.as_run_input().expect("bind chosen input");
    let foreign_input = foreign.as_run_input().expect("bind foreign input");
    let candidates = distinct_signal_genes(&input);
    let distances = canonical_range_distances(&input);
    let selected_receipt = input.receipt().identity_sha256().expect("chosen receipt");
    let foreign_receipt = foreign_input
        .receipt()
        .identity_sha256()
        .expect("foreign receipt");
    let request = |candidates, receipt_sha256| HistoricalCandidateScanRequestV2 {
        input: &input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        candidates,
        failure_policy: HistoricalCandidateFailurePolicyV1::FailEntireScan,
        distance_source: HistoricalCandidateDistanceSourceV1 {
            receipt_sha256,
            semantic_id: "canonical.reference-ohlc.range.v1",
            values: &distances,
        },
        stop_multiple: 1.0,
        target_multiple: 2.0,
    };

    let foreign_error = scan_with_width(request(&candidates, &foreign_receipt), 2)
        .expect_err("distance source from another generation must fail");
    assert!(matches!(
        foreign_error,
        HistoricalCandidateScanError::DistanceReceiptMismatch { .. }
    ));

    let duplicates = [candidates[0].clone(), candidates[0].clone()];
    let duplicate_error = scan_with_width(request(&duplicates, &selected_receipt), 2)
        .expect_err("duplicate exact candidate identities must fail");
    assert!(matches!(
        duplicate_error,
        HistoricalCandidateScanError::DuplicateCandidateIdentity {
            first_index: 0,
            duplicate_index: 1,
            ..
        }
    ));

    let mut invalid = candidates[0].clone();
    invalid.weights[0] = f64::NAN;
    let invalid_error = scan_with_width(request(&[invalid], &selected_receipt), 2)
        .expect_err("non-finite gene identity must fail before signal generation");
    assert!(matches!(
        invalid_error,
        HistoricalCandidateScanError::InvalidCandidateIdentity {
            candidate_index: 0,
            ..
        }
    ));
}

#[test]
fn candidate_failure_policy_is_typed_and_never_turns_failure_into_zero_signals() {
    let root = TempRoot::new("candidate-failure-policy");
    let identity = publish_contract_bars(root.path(), "historical-candidate-failure", 1.0);
    let selected = ExactCanonicalSeries::open(root.path(), identity)
        .expect("select exact failure-policy generation")
        .load_search_input(&[])
        .expect("load exact failure-policy input");
    let input = selected.as_run_input().expect("bind receipt to values");
    let mut invalid_candidate = distinct_signal_genes(&input)[0].clone();
    invalid_candidate.indices[0] = input.features().n_features();
    let candidates = [invalid_candidate];
    let distances = canonical_range_distances(&input);
    let receipt_sha256 = input.receipt().identity_sha256().expect("receipt identity");
    let request = |failure_policy| HistoricalCandidateScanRequestV2 {
        input: &input,
        backend: HistoricalResearchBackendV1::CpuOnly,
        candidates: &candidates,
        failure_policy,
        distance_source: HistoricalCandidateDistanceSourceV1 {
            receipt_sha256: &receipt_sha256,
            semantic_id: "canonical.reference-ohlc.range.v1",
            values: &distances,
        },
        stop_multiple: 1.0,
        target_multiple: 2.0,
    };

    let fail_entire = scan_with_width(
        request(HistoricalCandidateFailurePolicyV1::FailEntireScan),
        2,
    )
    .expect_err("FailEntireScan must surface the candidate error");
    assert!(matches!(
        fail_entire,
        HistoricalCandidateScanError::CandidateFailed {
            candidate_index: 0,
            ..
        }
    ));

    let retained = scan_with_width(
        request(HistoricalCandidateFailurePolicyV1::RetainFailedCandidate),
        2,
    )
    .expect("RetainFailedCandidate returns an explicit failed result");
    assert_eq!(retained.results().len(), 1);
    assert!(matches!(
        retained.results()[0].status(),
        HistoricalCandidateResultStatusV1::Failed
    ));
    assert!(retained.results()[0].artifact().is_none());
    assert!(retained.ranking().is_empty());
    assert_eq!(retained.best_candidate_identity_sha256(), None);
}

#[test]
fn research_source_has_no_legacy_financial_or_live_api_dependency() {
    let source = include_str!("../src/historical_research.rs");
    for forbidden in [
        "BacktestSettings",
        "BrokerFinancialTruthCapability",
        "current_broker_financial_truth_capability_v1",
        "simulate_trades",
        "DiscoveryResult",
        "LivePortfolioArtifact",
        "default_pip_size",
        "infer_market_cost_profile",
        "spread_pips",
        "commission_per_trade",
        "swap_",
        "net_profit",
        "account_currency",
    ] {
        assert!(
            !source.contains(forbidden),
            "historical research reached forbidden legacy API/token {forbidden}"
        );
    }
}
