use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use neoethos_data::core::feature_run_lease::FeatureRunLease;
use neoethos_data::core::features::{FeatureCellValidity, FeatureColumnF64};
use neoethos_data::core::vortex_feature_store::{VortexFeatureStore, VortexFeatureStoreOptions};

fn unique_root(label: &str) -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock predates Unix epoch")?
        .as_nanos();
    Ok(std::env::temp_dir().join(format!(
        "neoethos-vortex-feature-crash-{label}-{}-{nonce}",
        std::process::id()
    )))
}

#[test]
fn unfinished_candidate_is_never_opened_as_a_feature_store() -> Result<()> {
    let root = unique_root("unfinished")?;
    let lease = Arc::new(FeatureRunLease::create(&root, "unfinished")?);
    let unfinished = lease.run_dir().join("features.vortex.tmp-crashed-writer");
    std::fs::write(&unfinished, b"partial Vortex bytes")?;

    let error = VortexFeatureStore::open(Arc::clone(&lease), vec!["feature".to_owned()], 1024)
        .expect_err("an unfinished candidate must not be discovered as complete");
    assert!(
        error
            .to_string()
            .contains("completed Vortex feature store is missing")
    );
    assert!(unfinished.is_file());

    let run_dir = lease.run_dir().to_path_buf();
    drop(lease);
    assert!(!run_dir.exists());
    Ok(())
}

#[test]
fn corrupt_completed_name_fails_closed_instead_of_opening_partial_data() -> Result<()> {
    let root = unique_root("corrupt")?;
    let lease = Arc::new(FeatureRunLease::create(&root, "corrupt")?);
    std::fs::write(
        lease.run_dir().join("features.vortex"),
        b"not a Vortex file",
    )?;

    let error = VortexFeatureStore::open(Arc::clone(&lease), vec!["feature".to_owned()], 1024)
        .expect_err("corrupt final file must fail closed");
    assert!(
        error.to_string().contains("Vortex") || error.to_string().contains("vortex"),
        "unexpected error: {error:#}"
    );
    Ok(())
}

#[test]
fn invalid_schema_fails_before_any_feature_file_is_written() -> Result<()> {
    let root = unique_root("schema")?;
    let lease = Arc::new(FeatureRunLease::create(&root, "schema")?);
    let timestamps = [1_704_067_200_000_i64, 1_704_067_260_000_i64];
    let column = FeatureColumnF64::new(
        "duplicate",
        vec![1.0, 2.0],
        vec![FeatureCellValidity::Valid; 2],
    )?;
    let error = VortexFeatureStore::create(
        Arc::clone(&lease),
        &timestamps,
        &[column.clone(), column],
        VortexFeatureStoreOptions::default(),
    )
    .expect_err("duplicate names must fail before writing");
    assert!(error.to_string().contains("duplicate"));
    assert!(!lease.run_dir().join("features.vortex").exists());
    assert!(
        std::fs::read_dir(lease.run_dir())?.next().is_none(),
        "schema rejection left a candidate artifact"
    );
    Ok(())
}
