use neoethos_app::server::knob_catalog::{KnobKind, build_catalog};

#[test]
fn catalog_exposes_one_canonical_cpu_width_knob_with_truthful_auto_semantics() {
    let catalog = build_catalog();
    let cpu_knobs: Vec<_> = catalog
        .iter()
        .filter(|entry| {
            let id = entry.id.to_ascii_lowercase();
            id.contains("cpu") || id.contains("rayon") || id.contains("thread")
        })
        .collect();

    assert!(
        catalog
            .iter()
            .all(|entry| entry.id != "backtest.rayon_threads"),
        "the retired search-local Rayon knob must not remain exposed"
    );

    let canonical: Vec<_> = cpu_knobs
        .iter()
        .copied()
        .filter(|entry| entry.id == "system.hardware.cpu_budget")
        .collect();
    assert_eq!(
        canonical.len(),
        1,
        "there must be exactly one persistent CPU-width knob"
    );

    let entry = canonical[0];
    assert!(
        matches!(
            entry.kind,
            KnobKind::Int {
                min: Some(1),
                max: None
            }
        ),
        "the UI must not introduce a second hard upper bound"
    );
    let guidance =
        format!("{} {} {}", entry.default, entry.help_short, entry.help_long).to_ascii_lowercase();
    assert!(guidance.contains("effective logical"));
    assert!(guidance.contains("minus two") || guidance.contains("- 2"));
    assert!(guidance.contains("reserve"));
    assert!(!guidance.contains("all logical cores"));
    assert!(!guidance.contains("physical-core"));
    assert!(!guidance.contains("physical core"));
}
