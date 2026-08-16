use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use neoethos_data::core::feature_store::{FeatureStore, FeatureStoreWriter};
use neoethos_data::core::features::{FeatureData, FeatureFrame};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct FeatureStoreContract {
    schema: String,
    timestamps_ms: Vec<i64>,
    feature_names: Vec<String>,
    f32_exact_values: Vec<f32>,
    f32_exact_bits: Vec<u32>,
    high_precision_f64_values: Vec<f64>,
    high_precision_f64_bits: Vec<u64>,
    current_narrowed_f32_bits: Vec<u32>,
    current_widened_f64_bits: Vec<u64>,
    labels: Vec<i8>,
}

fn load_contract() -> Result<FeatureStoreContract> {
    serde_json::from_str(include_str!("fixtures/feature_store_contract_v1.json"))
        .context("parse feature-store contract fixture")
}

fn unique_store_path() -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock predates Unix epoch")?
        .as_nanos();
    Ok(std::env::temp_dir().join(format!(
        "neoethos-feature-store-contract-{}-{nonce}.fstore",
        std::process::id()
    )))
}

fn bits_f32(values: impl IntoIterator<Item = f32>) -> Vec<u32> {
    values.into_iter().map(f32::to_bits).collect()
}

fn bits_f64(values: impl IntoIterator<Item = f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

fn measured_samples_ns<F>(mut operation: F) -> (Vec<u128>, f64)
where
    F: FnMut() -> f64,
{
    let mut checksum = 0.0;
    for _ in 0..3 {
        checksum += operation();
    }
    let mut samples = Vec::with_capacity(10);
    for _ in 0..10 {
        let started = Instant::now();
        checksum += operation();
        samples.push(started.elapsed().as_nanos());
    }
    (samples, checksum)
}

#[cfg(target_os = "linux")]
fn peak_rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}

#[cfg(not(target_os = "linux"))]
fn peak_rss_kib() -> Option<u64> {
    None
}

#[test]
fn current_feature_store_preserves_layout_but_exposes_f64_narrowing() -> Result<()> {
    let fixture = load_contract()?;
    ensure!(
        fixture.schema == "neoethos.feature_store_contract.v1",
        "unexpected fixture schema {}",
        fixture.schema
    );
    let rows = fixture.timestamps_ms.len();
    ensure!(rows == fixture.labels.len(), "timestamp/label mismatch");
    ensure!(
        fixture.feature_names.len() == 2,
        "fixture must contain exactly two feature columns"
    );

    assert_eq!(
        bits_f32(fixture.f32_exact_values.iter().copied()),
        fixture.f32_exact_bits,
        "the f32-exact oracle changed"
    );
    assert_eq!(
        bits_f64(fixture.high_precision_f64_values.iter().copied()),
        fixture.high_precision_f64_bits,
        "the pre-narrowing f64 oracle changed"
    );

    let narrowed: Vec<f32> = fixture
        .high_precision_f64_values
        .iter()
        .copied()
        .map(|value| value as f32)
        .collect();
    assert_eq!(
        bits_f32(narrowed.iter().copied()),
        fixture.current_narrowed_f32_bits,
        "the recorded current f64-to-f32 conversion changed"
    );
    assert_eq!(
        bits_f64(narrowed.iter().copied().map(f64::from)),
        fixture.current_widened_f64_bits,
        "the recorded current f32-to-f64 widening changed"
    );
    for ((source, source_bits), widened_bits) in fixture
        .high_precision_f64_values
        .iter()
        .zip(&fixture.high_precision_f64_bits)
        .zip(&fixture.current_widened_f64_bits)
    {
        assert_ne!(
            source_bits, widened_bits,
            "high-precision fixture value {source:?} unexpectedly survived f32 narrowing"
        );
    }

    let path = unique_store_path()?;
    let mut writer = FeatureStoreWriter::create(&path, rows)?;
    writer.append_feature(&fixture.f32_exact_values)?;
    writer.append_feature(&narrowed)?;
    assert_eq!(writer.finish()?, fixture.feature_names.len());

    let store = Arc::new(FeatureStore::open(
        &path,
        fixture.feature_names.len(),
        rows,
        true,
    )?);
    let frame = FeatureFrame {
        timestamps: fixture.timestamps_ms.clone(),
        names: fixture.feature_names.clone(),
        data: FeatureData::Mmap(store),
    };

    assert_eq!(frame.n_samples(), rows);
    assert_eq!(frame.n_features(), 2);
    assert_eq!(
        bits_f32(frame.feature_column(0).iter().copied()),
        fixture.f32_exact_bits,
        "selected f32-exact mmap column changed"
    );
    assert_eq!(
        bits_f32(frame.feature_column(1).iter().copied()),
        fixture.current_narrowed_f32_bits,
        "selected high-precision mmap column no longer matches the recorded legacy result"
    );
    for sample in 0..rows {
        assert_eq!(
            frame.feature_at(sample, 0).to_bits(),
            fixture.f32_exact_bits[sample]
        );
        assert_eq!(
            frame.feature_at(sample, 1).to_bits(),
            fixture.current_narrowed_f32_bits[sample]
        );
    }

    let selected = frame.sample_window(1, 5);
    assert_eq!(selected.dim(), (4, 2));
    for local_row in 0..4 {
        let source_row = local_row + 1;
        assert_eq!(
            selected[(local_row, 0)].to_bits(),
            fixture.f32_exact_bits[source_row]
        );
        assert_eq!(
            selected[(local_row, 1)].to_bits(),
            fixture.current_narrowed_f32_bits[source_row]
        );
    }

    let outer = frame.row_window(1, 6);
    assert_eq!(outer.timestamps, fixture.timestamps_ms[1..6]);
    assert_eq!(outer.n_samples(), 5);
    let inner = outer.row_window(1, 4);
    assert_eq!(inner.timestamps, fixture.timestamps_ms[2..5]);
    assert_eq!(inner.names, fixture.feature_names);
    assert_eq!(inner.n_samples(), 3);
    assert_eq!(
        bits_f32(inner.feature_column(0).iter().copied()),
        fixture.f32_exact_bits[2..5]
    );
    assert_eq!(
        bits_f32(inner.feature_column(1).iter().copied()),
        fixture.current_narrowed_f32_bits[2..5]
    );

    let aligned_labels = &fixture.labels[2..5];
    assert_eq!(aligned_labels, [1, -1, 0]);
    assert_eq!(inner.timestamps.len(), aligned_labels.len());

    drop(inner);
    drop(outer);
    drop(frame);
    assert!(
        !path.exists(),
        "delete-on-drop feature store was not reclaimed: {}",
        path.display()
    );
    Ok(())
}

#[test]
#[ignore = "explicit Task-1 performance baseline; three warmups plus ten measured runs"]
fn baseline_vortex_and_fstore_access_metrics() -> Result<()> {
    const ROWS: usize = 250_000;
    const FEATURES: usize = 64;
    const WINDOW_START: usize = 50_000;
    const WINDOW_END: usize = 150_000;
    const SELECTED: [usize; 8] = [0, 3, 7, 13, 21, 31, 47, 63];

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock predates Unix epoch")?
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "neoethos-task1-data-baseline-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&root)?;
    let vortex_path = root.join("data.vortex");
    let fstore_path = root.join("features.fstore");

    let timestamps: Vec<i64> = (0..ROWS)
        .map(|row| 1_704_067_200_000_i64 + row as i64 * 60_000)
        .collect();
    let open: Vec<f64> = (0..ROWS)
        .map(|row| 1.05 + (row % 10_000) as f64 * 1e-7)
        .collect();
    let high: Vec<f64> = open.iter().map(|value| value + 0.0002).collect();
    let low: Vec<f64> = open.iter().map(|value| value - 0.0002).collect();
    let close: Vec<f64> = open
        .iter()
        .enumerate()
        .map(|(row, value)| value + ((row % 7) as f64 - 3.0) * 1e-6)
        .collect();
    let volume: Vec<f64> = (0..ROWS).map(|row| 100.0 + (row % 1_000) as f64).collect();
    let ohlcv = neoethos_data::Ohlcv {
        timestamp: Some(timestamps),
        open,
        high,
        low,
        close,
        volume: Some(volume),
    };
    neoethos_data::write_ohlcv_vortex(&vortex_path, &ohlcv)?;

    let mut writer = FeatureStoreWriter::create(&fstore_path, ROWS)?;
    for feature in 0..FEATURES {
        let values: Vec<f32> = (0..ROWS)
            .map(|row| feature as f32 * 0.01 + row as f32 * 1e-6)
            .collect();
        writer.append_feature(&values)?;
    }
    assert_eq!(writer.finish()?, FEATURES);
    let store = Arc::new(FeatureStore::open(&fstore_path, FEATURES, ROWS, false)?);
    let frame = FeatureFrame {
        timestamps: (0..ROWS).map(|row| row as i64).collect(),
        names: (0..FEATURES)
            .map(|feature| format!("feature_{feature}"))
            .collect(),
        data: FeatureData::Mmap(store.clone()),
    };

    let (vortex_scan_ns, vortex_checksum) = measured_samples_ns(|| {
        let loaded = neoethos_data::load_vortex(&vortex_path).expect("baseline Vortex scan");
        loaded.open[0]
            + loaded.close[loaded.len() - 1]
            + loaded.volume.as_ref().expect("volume")[loaded.len() - 1]
    });
    let (selected_column_ns, selected_checksum) = measured_samples_ns(|| {
        SELECTED
            .iter()
            .flat_map(|feature| store.feature_row(*feature).iter())
            .map(|value| f64::from(*value))
            .sum()
    });
    let (row_window_ns, window_checksum) = measured_samples_ns(|| {
        let window = frame.sample_window(WINDOW_START, WINDOW_END);
        f64::from(window[(0, 0)]) + f64::from(window[(window.nrows() - 1, window.ncols() - 1)])
    });
    let (ga_selected_access_ns, ga_checksum) = measured_samples_ns(|| {
        let mut checksum = 0.0_f64;
        for round in 0..8 {
            for feature in SELECTED {
                checksum += store
                    .feature_row((feature + round) % FEATURES)
                    .iter()
                    .map(|value| f64::from(*value))
                    .sum::<f64>();
            }
        }
        checksum
    });

    let report = serde_json::json!({
        "schema": "neoethos.task1_data_baseline.v1",
        "rows": ROWS,
        "features": FEATURES,
        "warmups": 3,
        "measured_runs": 10,
        "page_cache": "warm_after_three_explicit_warmups",
        "vortex_bytes": std::fs::metadata(&vortex_path)?.len(),
        "fstore_bytes": std::fs::metadata(&fstore_path)?.len(),
        "peak_rss_kib": peak_rss_kib(),
        "vortex_scan_ns": vortex_scan_ns,
        "fstore_selected_column_ns": selected_column_ns,
        "fstore_row_window_ns": row_window_ns,
        "ga_repeated_selected_access_ns": ga_selected_access_ns,
        "checksums": {
            "vortex": vortex_checksum.to_bits(),
            "selected": selected_checksum.to_bits(),
            "window": window_checksum.to_bits(),
            "ga": ga_checksum.to_bits()
        }
    });
    println!("NEOETHOS_BASELINE_JSON={report}");

    drop(frame);
    drop(store);
    std::fs::remove_dir_all(&root)
        .with_context(|| format!("remove baseline root {}", root.display()))?;
    Ok(())
}
