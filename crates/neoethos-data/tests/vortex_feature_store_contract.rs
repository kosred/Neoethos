use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use neoethos_data::core::feature_run_lease::FeatureRunLease;
use neoethos_data::core::features::{FeatureCellValidity, FeatureColumnF64};
use neoethos_data::core::vortex_feature_store::{
    VortexFeatureStore, VortexFeatureStoreOptions, VortexFeatureStoreSet,
};

fn unique_root(label: &str) -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock predates Unix epoch")?
        .as_nanos();
    Ok(std::env::temp_dir().join(format!(
        "neoethos-vortex-feature-{label}-{}-{nonce}",
        std::process::id()
    )))
}

fn fixture_columns() -> Result<Vec<FeatureColumnF64>> {
    Ok(vec![
        FeatureColumnF64::new(
            "exact",
            vec![0.0, 1.5, f64::NAN, -2.25, f64::NAN, 65_536.0],
            vec![
                FeatureCellValidity::Valid,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Warmup,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Gap,
                FeatureCellValidity::Valid,
            ],
        )?,
        FeatureColumnF64::new(
            "high_precision",
            vec![
                1.000_000_059_604_644_8,
                f64::NAN,
                std::f64::consts::PI,
                f64::NAN,
                123_456.789_012_345_67,
                f64::NAN,
            ],
            vec![
                FeatureCellValidity::Valid,
                FeatureCellValidity::ZeroDenominator,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Stale,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Degenerate,
            ],
        )?,
    ])
}

fn timestamps() -> Vec<i64> {
    (0..6)
        .map(|row| 1_704_067_200_000_i64 + row * 60_000)
        .collect()
}

#[test]
fn vortex_store_preserves_f64_bits_validity_projection_and_nested_windows() -> Result<()> {
    let root = unique_root("contract")?;
    let lease = Arc::new(FeatureRunLease::create(&root, "contract")?);
    let columns = fixture_columns()?;
    let store = VortexFeatureStore::create(
        Arc::clone(&lease),
        &timestamps(),
        &columns,
        VortexFeatureStoreOptions {
            chunk_rows: 2,
            decoded_cache_bytes: 64 * 1024,
        },
    )?;

    assert_eq!(store.n_samples(), 6);
    assert_eq!(store.names(), ["exact", "high_precision"]);
    let full = store.project(&[0, 1], 0..6)?;
    assert_eq!(full.timestamps, timestamps());
    assert_eq!(full.row_ids, [0, 1, 2, 3, 4, 5]);
    for (actual, expected) in full.columns.iter().zip(&columns) {
        assert_eq!(actual.name, expected.name);
        assert_eq!(actual.validity, expected.validity);
        assert_eq!(actual.values.len(), expected.values.len());
        for (actual, expected) in actual.values.iter().zip(&expected.values) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
    assert_eq!(full.columns[0].values[0].to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        full.columns[1].values[0].to_bits(),
        1.000_000_059_604_644_8_f64.to_bits()
    );

    let selected = store.project(&[1], 1..5)?;
    assert_eq!(selected.timestamps, timestamps()[1..5]);
    assert_eq!(selected.row_ids, [1, 2, 3, 4]);
    assert_eq!(selected.columns.len(), 1);
    assert_eq!(selected.columns[0].name, "high_precision");
    assert_eq!(selected.columns[0].validity, columns[1].validity[1..5]);
    for row in 0..4 {
        assert_eq!(
            selected.columns[0].values[row].to_bits(),
            columns[1].values[row + 1].to_bits()
        );
    }

    let outer = store.window(1..6)?;
    let inner = outer.window(1..4)?;
    let nested = inner.project(&[0, 1])?;
    assert_eq!(nested.timestamps, timestamps()[2..5]);
    assert_eq!(nested.row_ids, [2, 3, 4]);
    assert_eq!(nested.columns[0].validity, columns[0].validity[2..5]);
    assert_eq!(nested.columns[1].validity, columns[1].validity[2..5]);

    let stats_before = store.cache_stats();
    let _ = store.project(&[1], 1..5)?;
    let stats_after = store.cache_stats();
    assert!(stats_after.hits > stats_before.hits);
    assert!(stats_after.resident_bytes <= 64 * 1024);

    let labels = [-1_i8, 0, 1, -1, 0, 1];
    let aligned_labels: Vec<i8> = nested
        .row_ids
        .iter()
        .map(|row| labels[*row as usize])
        .collect();
    assert_eq!(aligned_labels, [1, -1, 0]);

    let run_dir = lease.run_dir().to_path_buf();
    drop(inner);
    drop(outer);
    drop(store);
    drop(lease);
    assert!(!run_dir.exists(), "normal RAII did not remove the run");
    Ok(())
}

#[test]
fn vortex_store_preserves_values_that_an_f32_narrowing_would_destroy() -> Result<()> {
    let root = unique_root("precision")?;
    std::fs::create_dir_all(&root)?;
    let exact = [0.0_f32, 1.5, -2.25, 65_536.0];
    let high_precision = [
        1.000_000_059_604_644_8_f64,
        std::f64::consts::PI,
        -std::f64::consts::E,
        123_456.789_012_345_67,
    ];
    let lease = Arc::new(FeatureRunLease::create(&root, "precision")?);
    let columns = vec![
        FeatureColumnF64::new(
            "exact",
            exact.iter().copied().map(f64::from).collect(),
            vec![FeatureCellValidity::Valid; exact.len()],
        )?,
        FeatureColumnF64::new(
            "high_precision",
            high_precision.to_vec(),
            vec![FeatureCellValidity::Valid; high_precision.len()],
        )?,
    ];
    let store = VortexFeatureStore::create(
        lease,
        &timestamps()[..4],
        &columns,
        VortexFeatureStoreOptions::default(),
    )?;
    let round_trip = store.project(&[0, 1], 0..4)?;

    for row in 0..4 {
        assert_eq!(
            round_trip.columns[0].values[row].to_bits(),
            f64::from(exact[row]).to_bits()
        );
        assert_ne!(
            round_trip.columns[1].values[row].to_bits(),
            f64::from(high_precision[row] as f32).to_bits(),
            "high-precision value unexpectedly equals its f32-narrowed representation"
        );
    }
    drop(store);
    std::fs::remove_dir_all(&root)?;
    Ok(())
}

#[test]
fn vortex_store_set_preserves_arbitrary_global_projection_order_and_cell_contract() -> Result<()> {
    let root = unique_root("set-contract")?;
    let first_lease = Arc::new(FeatureRunLease::create(&root, "set-first")?);
    let second_lease = Arc::new(FeatureRunLease::create(&root, "set-second")?);
    let first_columns = fixture_columns()?;
    let second_columns = vec![
        FeatureColumnF64::new(
            "second_exact",
            vec![9.0, f64::NAN, 7.0, 6.0, f64::NAN, 4.0],
            vec![
                FeatureCellValidity::Valid,
                FeatureCellValidity::AlignmentMissing,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Valid,
                FeatureCellValidity::ComputeFailure,
                FeatureCellValidity::Valid,
            ],
        )?,
        FeatureColumnF64::new(
            "second_precise",
            vec![
                std::f64::consts::E,
                1.000_000_000_000_000_2,
                f64::NAN,
                -std::f64::consts::PI,
                0.0,
                42.000_000_000_000_01,
            ],
            vec![
                FeatureCellValidity::Valid,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Gap,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Valid,
                FeatureCellValidity::Valid,
            ],
        )?,
    ];
    let options = VortexFeatureStoreOptions {
        chunk_rows: 2,
        decoded_cache_bytes: 64 * 1024,
    };
    let first = VortexFeatureStore::create(
        Arc::clone(&first_lease),
        &timestamps(),
        &first_columns,
        options,
    )?;
    let second = VortexFeatureStore::create(
        Arc::clone(&second_lease),
        &timestamps(),
        &second_columns,
        options,
    )?;
    let stores = VortexFeatureStoreSet::new(vec![Arc::clone(&first), Arc::clone(&second)])?;

    assert_eq!(stores.shard_count(), 2);
    assert_eq!(
        stores.names(),
        ["exact", "high_precision", "second_exact", "second_precise"]
    );
    assert_eq!(stores.n_samples(), timestamps().len());

    let projection = stores.project(&[2, 0, 3, 1], 1..5)?;
    assert_eq!(projection.timestamps, timestamps()[1..5]);
    assert_eq!(projection.row_ids, [1, 2, 3, 4]);
    assert_eq!(
        projection
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        ["second_exact", "exact", "second_precise", "high_precision"]
    );
    for (actual, expected) in projection.columns.iter().zip([
        &second_columns[0],
        &first_columns[0],
        &second_columns[1],
        &first_columns[1],
    ]) {
        assert_eq!(actual.validity, expected.validity[1..5]);
        for (actual, expected) in actual.values.iter().zip(&expected.values[1..5]) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    let first_run = first_lease.run_dir().to_path_buf();
    let second_run = second_lease.run_dir().to_path_buf();
    drop(stores);
    drop(first);
    drop(second);
    drop(first_lease);
    drop(second_lease);
    assert!(!first_run.exists());
    assert!(!second_run.exists());
    Ok(())
}

#[test]
fn vortex_store_set_rejects_same_length_shards_with_different_timestamp_identity() -> Result<()> {
    let root = unique_root("set-identity-mismatch")?;
    let first_lease = Arc::new(FeatureRunLease::create(&root, "identity-first")?);
    let second_lease = Arc::new(FeatureRunLease::create(&root, "identity-second")?);
    let validity = vec![FeatureCellValidity::Valid; timestamps().len()];
    let first_columns = vec![FeatureColumnF64::new(
        "first_identity_feature",
        vec![1.0; timestamps().len()],
        validity.clone(),
    )?];
    let second_columns = vec![FeatureColumnF64::new(
        "second_identity_feature",
        vec![2.0; timestamps().len()],
        validity,
    )?];
    let first = VortexFeatureStore::create(
        first_lease,
        &timestamps(),
        &first_columns,
        VortexFeatureStoreOptions::default(),
    )?;
    let shifted_timestamps = timestamps()
        .into_iter()
        .map(|timestamp| timestamp + 1_000)
        .collect::<Vec<_>>();
    let second = VortexFeatureStore::create(
        second_lease,
        &shifted_timestamps,
        &second_columns,
        VortexFeatureStoreOptions::default(),
    )?;

    let error = VortexFeatureStoreSet::new(vec![first, second])
        .expect_err("different timestamp identities must fail when the set is constructed");
    assert!(
        error
            .to_string()
            .contains("Vortex feature shard identity mismatch"),
        "unexpected error: {error:#}"
    );
    Ok(())
}
