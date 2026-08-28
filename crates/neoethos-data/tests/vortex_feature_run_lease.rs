use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use neoethos_data::core::feature_run_lease::{FeatureRunLease, sweep_orphan_feature_runs};
use neoethos_data::core::features::{FeatureCellValidity, FeatureColumnF64};
use neoethos_data::core::vortex_feature_store::{VortexFeatureStore, VortexFeatureStoreOptions};

fn unique_root(label: &str) -> Result<PathBuf> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock predates Unix epoch")?
        .as_nanos();
    Ok(std::env::temp_dir().join(format!(
        "neoethos-feature-lease-{label}-{}-{nonce}",
        std::process::id()
    )))
}

fn wait_for(path: &Path, timeout: Duration) -> Result<()> {
    let started = Instant::now();
    while !path.exists() {
        if started.elapsed() > timeout {
            anyhow::bail!("timed out waiting for {}", path.display());
        }
        thread::sleep(Duration::from_millis(20));
    }
    Ok(())
}

#[test]
fn active_run_is_never_swept_and_process_death_makes_it_collectible() -> Result<()> {
    let root = unique_root("cross-process")?;
    std::fs::create_dir_all(&root)?;
    let ready = root.join("ready");
    let mut child = Command::new(std::env::current_exe()?)
        .arg("--exact")
        .arg("feature_run_lease_child_holder")
        .arg("--nocapture")
        .env("NEOETHOS_FEATURE_LEASE_CHILD", "1")
        .env("NEOETHOS_FEATURE_LEASE_ROOT", &root)
        .spawn()
        .context("spawn feature lease holder")?;

    wait_for(&ready, Duration::from_secs(10))?;
    let run_dir = root.join("run-held");
    assert!(run_dir.is_dir());
    let canonical_run_dir = run_dir.canonicalize()?;

    // Misleading old/PID-like diagnostics are not liveness authority.
    std::fs::write(
        run_dir.join("diagnostic-owner.txt"),
        "pid=1\ncreated=1970-01-01",
    )?;
    let removed = sweep_orphan_feature_runs(&root)?;
    assert!(removed.is_empty());
    assert!(run_dir.is_dir(), "sweeper removed an OS-locked live run");

    child.kill().context("kill feature lease holder")?;
    child.wait().context("join feature lease holder")?;
    let removed = sweep_orphan_feature_runs(&root)?;
    assert_eq!(removed, vec![canonical_run_dir]);
    assert!(!run_dir.exists());
    std::fs::remove_dir_all(&root)?;
    Ok(())
}

#[test]
fn feature_run_lease_child_holder() -> Result<()> {
    if std::env::var_os("NEOETHOS_FEATURE_LEASE_CHILD").is_none() {
        return Ok(());
    }
    let root = PathBuf::from(
        std::env::var_os("NEOETHOS_FEATURE_LEASE_ROOT")
            .context("missing child feature lease root")?,
    );
    let lease = Arc::new(FeatureRunLease::create(&root, "held")?);
    let timestamps = [1_704_067_200_000_i64, 1_704_067_260_000];
    let columns = [FeatureColumnF64::new(
        "held_feature",
        vec![1.0, 2.0],
        vec![FeatureCellValidity::Valid; 2],
    )?];
    let store = VortexFeatureStore::create(
        lease,
        &timestamps,
        &columns,
        VortexFeatureStoreOptions {
            chunk_rows: 1,
            decoded_cache_bytes: 4 * 1024,
        },
    )?;
    let first = store.project(&[0], 0..2)?;
    anyhow::ensure!(
        first.row_ids == [0, 1],
        "child projection identity mismatch"
    );
    std::fs::write(root.join("ready"), b"ready")?;
    loop {
        let projected = store.project(&[0], 0..2)?;
        anyhow::ensure!(
            projected.columns[0].values == [1.0, 2.0],
            "live child projection changed while sweeper was active"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn unsafe_run_identity_cannot_escape_the_scratch_root() -> Result<()> {
    let root = unique_root("containment")?;
    for unsafe_id in ["", ".", "..", "../escape", "a/b", "a\\b", "C:escape"] {
        let error = FeatureRunLease::create(&root, unsafe_id)
            .expect_err("unsafe run id must fail before creating a path");
        assert!(error.to_string().contains("run id"));
    }
    assert!(!root.parent().expect("parent").join("escape").exists());
    Ok(())
}
