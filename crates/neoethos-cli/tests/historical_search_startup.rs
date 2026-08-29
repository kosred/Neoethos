use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use neoethos_core::Settings;
use neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use neoethos_data::{
    BarTimestampConvention, CanonicalDatasetIdentity, CanonicalOhlcvPublishRequest,
    CanonicalTimeframe, CanonicalVolumeRef, Ohlcv, publish_canonical_ohlcv_generation,
};
use neoethos_search::ExactCanonicalSeries;

static NEXT_TEMP_ROOT: AtomicU64 = AtomicU64::new(0);

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        let unique = NEXT_TEMP_ROOT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "neoethos-cli-historical-search-startup-{label}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create isolated historical-search startup root");
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
                "ERROR historical-search startup test cleanup failed for {}: {error}",
                self.0.display()
            );
        }
    }
}

fn publish_fixture(root: &Path) -> CanonicalDatasetIdentity {
    let identity = CanonicalDatasetIdentity::external(
        "legacy-cli-config-independent-search",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid direct M1 identity");
    let rows = 192_usize;
    let timestamp = (0..rows)
        .map(|row| 1_704_067_200_000_i64 + row as i64 * 60_000)
        .collect::<Vec<_>>();
    let close = (0..rows)
        .map(|row| 1.10 + row as f64 * 0.00001 + (row as f64 / 9.0).sin() * 0.0004)
        .collect::<Vec<_>>();
    let open = (0..rows)
        .map(|row| {
            row.checked_sub(1)
                .map_or(close[0], |previous| close[previous])
        })
        .collect::<Vec<_>>();
    let high = open
        .iter()
        .zip(&close)
        .map(|(open, close)| open.max(*close) + 0.0003)
        .collect::<Vec<_>>();
    let low = open
        .iter()
        .zip(&close)
        .map(|(open, close)| open.min(*close) - 0.0003)
        .collect::<Vec<_>>();
    let ohlcv = Ohlcv {
        timestamp: Some(timestamp),
        open,
        high,
        low,
        close,
        volume: Some((0..rows).map(|row| 100.0 + row as f64).collect()),
    };
    let provenance = ProducerProvenanceEnvelopeV1::new(
        "neoethos.cli-historical-search-startup-test.v1",
        identity.canonical_bytes(),
    )
    .expect("fixture provenance");
    publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
        configured_root: root,
        identity: &identity,
        expected_generation: None,
        provenance: &provenance,
        ohlcv: &ohlcv,
        volume: CanonicalVolumeRef::Float64(ohlcv.volume.as_deref().expect("fixture volume")),
        rows_per_chunk: 64,
    })
    .expect("publish direct canonical fixture");
    identity
}

fn write_settings(path: &Path, normalize_features: bool, cpu_budget: usize) {
    let mut settings = Settings::default();
    settings.models.data_runtime.normalize_features = normalize_features;
    settings.system.hardware.cpu_budget = Some(cpu_budget);
    settings.save(path).expect("write isolated config override");
}

#[test]
fn legacy_search_artifact_is_independent_of_missing_poisoned_and_semantic_settings() {
    let temp = TempRoot::new("config-independent");
    let identity = publish_fixture(temp.path());
    let selected = ExactCanonicalSeries::open(temp.path(), identity)
        .expect("select exact canonical fixture")
        .load_search_input(&[])
        .expect("build unnormalized exact fixture receipt");
    let receipt_path = temp.path().join("expected-receipt.json");
    fs::write(
        &receipt_path,
        selected
            .receipt()
            .expect("canonical input receipt")
            .to_json_bytes()
            .expect("serialize canonical input receipt"),
    )
    .expect("write exact receipt");

    let missing_config = temp.path().join("missing-config.yaml");
    let poisoned_config = temp.path().join("poisoned-config.yaml");
    fs::write(&poisoned_config, b"models: [this is not valid settings")
        .expect("write malformed config");
    let normalized_cap_one = temp.path().join("normalized-cap-one.yaml");
    write_settings(&normalized_cap_one, true, 1);
    let raw_cap_two = temp.path().join("raw-cap-two.yaml");
    write_settings(&raw_cap_two, false, 2);

    let configurations = [
        ("missing", &missing_config),
        ("poisoned", &poisoned_config),
        ("normalized-cap-one", &normalized_cap_one),
        ("raw-cap-two", &raw_cap_two),
    ];
    let mut artifacts = Vec::new();
    for (label, config) in configurations {
        let output_path = temp.path().join(format!("{label}.json"));
        let output = Command::new(env!("CARGO_BIN_EXE_neoethos-cli"))
            .current_dir(temp.path())
            .env("CONFIG_FILE", config)
            .args([
                "search",
                "--expected-input-receipt",
                receipt_path.to_str().expect("UTF-8 receipt path"),
                "--seed",
                "42",
                "--candidates",
                "1",
                "--max-indicators",
                "1",
                "--stop-multiple",
                "1.0",
                "--target-multiple",
                "2.0",
                "--out",
                output_path.to_str().expect("UTF-8 output path"),
                "--root",
                temp.path().to_str().expect("UTF-8 data root"),
            ])
            .output()
            .expect("run legacy strict historical search");
        assert!(
            output.status.success(),
            "search must ignore {label} config {}\nstdout:\n{}\nstderr:\n{}",
            config.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("CONFIG NOT LOADED"),
            "strict search inspected {label} config:\n{stderr}"
        );
        artifacts.push(fs::read(output_path).expect("read strict historical-search artifact"));
    }

    for received in &artifacts[1..] {
        assert_eq!(
            received, &artifacts[0],
            "config normalization and persistent CPU caps must not change search evidence"
        );
    }
}
