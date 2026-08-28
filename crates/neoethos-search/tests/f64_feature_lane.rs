use neoethos_data::core::features::{FeatureCellValidity, FeatureColumnF64, FeatureFrame};
use neoethos_search::genetic::search_engine::signals_and_confidence_for_gene_with_config;
use neoethos_search::genetic::strategy_gene::{EvaluationConfig, Gene};

fn precision_frame() -> FeatureFrame {
    let base = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
    let rows = base.n_samples();
    let mut precise = vec![0.0; rows];
    precise[0] = 1.000_000_059_604_644_8_f64;
    let mut precise_validity = vec![FeatureCellValidity::Valid; rows];
    precise_validity[1] = FeatureCellValidity::Warmup;
    let guard = vec![0.0; rows];

    FeatureFrame::from_columns(
        base.timestamps.clone(),
        vec![
            FeatureColumnF64::new("close_minus_open", precise, precise_validity)
                .expect("precision feature"),
            FeatureColumnF64::new("range_pips", guard, vec![FeatureCellValidity::Valid; rows])
                .expect("guard feature"),
        ],
        base.plan().clone(),
        base.provenance().clone(),
    )
    .expect("precision frame")
}

#[test]
fn cpu_search_preserves_f64_threshold_bits_and_never_signals_invalid_rows() {
    let frame = precision_frame();
    let gene = Gene {
        indices: vec![0],
        weights: vec![1.0_f64],
        long_threshold: 1.000_000_03_f64,
        short_threshold: -1.0_f64,
        ..Gene::default()
    };

    fn require_f64(_: f64) {}
    require_f64(gene.weights[0]);
    require_f64(gene.long_threshold);
    require_f64(gene.short_threshold);

    let (signals, confidence): (Vec<i8>, Vec<f64>) =
        signals_and_confidence_for_gene_with_config(&frame, &gene, &EvaluationConfig::default())
            .expect("f64 feature projection and signal synthesis");

    assert_eq!(
        signals[0], 1,
        "the f64-only threshold crossing must survive"
    );
    assert!(confidence[0] > 0.0);
    assert_eq!(signals[1], 0, "warmup input must be ineligible");
    assert_eq!(confidence[1].to_bits(), 0.0_f64.to_bits());
}
