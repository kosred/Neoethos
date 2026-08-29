use std::fs;
use std::path::PathBuf;

#[test]
fn resolved_config_reports_the_manifest_backed_canonical_layout() {
    let source = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/resolved_config.rs"),
    )
    .expect("read resolved-config source");

    assert!(source.contains("d1-<canonical-dataset-identity>"));
    assert!(source.contains("data.vortex.complete"));
    assert!(source.contains("g1-<sha256>.vortex"));
    assert!(!source.contains("symbol={SYM}/timeframe={TF}"));
}
