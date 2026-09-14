//! Local, write-once accounting of reserved final-window use. This records
//! exposure since this journal was introduced, not never-ever-seen market data.

use anyhow::{Context, Result, ensure};
use neoethos_search::data_selection::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchWindowRoleV1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const JOURNAL_SCHEMA: &str = "neoethos.final-holdout-local-use.v1";
static ATTEMPT_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[path = "final_holdout_reader.rs"]
mod reader;
pub(crate) use reader::{
    CombinedResearchReportsDto, load_saved_final_research_context,
    read_saved_combined_research_reports, read_saved_final_research_reports_with_context,
};
#[cfg(test)]
pub(crate) use reader::{
    install_saved_research_test_fixture, install_saved_strategy_research_test_fixture,
    read_saved_final_research_reports,
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FirstUse {
    schema: String,
    raw_scope_sha256: String,
    locked_inputs_sha256: String,
}

pub(super) struct FinalHoldoutUse {
    directory: PathBuf,
    attempt: String,
    pub(super) raw_scope_sha256: String,
    pub(super) locked_inputs_sha256: String,
    pub(super) first_recorded_use: bool,
    pub(super) first_locked_inputs_sha256: Option<String>,
}

fn hash_material(domain: &[u8], value: &serde_json::Value) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(serde_json::to_vec(value)?);
    Ok(format!("{:x}", hash.finalize()))
}

/// Bind the raw anchor and evaluated interval, deliberately NOT a candidate,
/// feature plan, normalization fit or higher-TF projection. Changing those must
/// not make this exact raw market window look newly unobserved.
fn raw_scope_material(scope: &CanonicalSearchArtifactScopeV2) -> Result<serde_json::Value> {
    scope.validate_against_receipt(scope.receipt())?;
    let window = scope.evaluated_window();
    ensure!(
        window.role() == CanonicalSearchWindowRoleV1::Holdout,
        "final use requires the reserved Holdout role"
    );
    let receipt = scope.receipt();
    let mut anchors = receipt
        .source_bindings()
        .iter()
        .filter(|binding| binding.dataset_identity() == receipt.anchor_dataset_identity());
    let anchor = anchors.next().context("final scope has no raw anchor")?;
    ensure!(
        anchors.next().is_none(),
        "final scope has ambiguous raw anchor bindings"
    );
    Ok(serde_json::json!({
        "dataset_identity": anchor.dataset_identity(),
        "generation_id": anchor.generation_id(),
        "manifest_sha256": anchor.manifest_sha256(),
        "vortex_sha256": anchor.vortex_sha256(),
        "bar_timestamp_convention": anchor.bar_timestamp_convention(),
        "row_start": window.row_start(), "row_end": window.row_end(),
        "timestamp_start_ms": window.timestamp_start_ms(),
        "timestamp_end_ms": window.timestamp_end_ms(),
    }))
}

fn write_new(path: &Path, value: &impl Serialize) -> Result<()> {
    // Serialize BEFORE creating the marker; a failed write still leaves its
    // reserved name consumed and must never turn a retry into a fresh use.
    let bytes = serde_json::to_vec(value)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

pub(super) fn begin(
    data_root: &Path,
    scope: &CanonicalSearchArtifactScopeV2,
    locked_inputs: serde_json::Value,
) -> Result<FinalHoldoutUse> {
    begin_material(data_root, raw_scope_material(scope)?, locked_inputs)
}

fn begin_material(
    data_root: &Path,
    raw_scope: serde_json::Value,
    locked_inputs: serde_json::Value,
) -> Result<FinalHoldoutUse> {
    let raw_scope_sha256 = hash_material(b"neoethos.final-raw-window.v1\0", &raw_scope)?;
    let locked_inputs_sha256 = hash_material(b"neoethos.final-locked-inputs.v1\0", &locked_inputs)?;
    let directory = data_root.join("final_holdout_uses").join(&raw_scope_sha256);
    std::fs::create_dir_all(&directory)?;
    let first_path = directory.join("first-start.json");
    let first = FirstUse {
        schema: JOURNAL_SCHEMA.to_owned(),
        raw_scope_sha256: raw_scope_sha256.clone(),
        locked_inputs_sha256: locked_inputs_sha256.clone(),
    };
    let first_recorded_use = match write_new(&first_path, &first) {
        Ok(()) => true,
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == ErrorKind::AlreadyExists) =>
        {
            false
        }
        Err(error) => return Err(error.context("persist reserved final-window first-start marker")),
    };
    // A concurrent writer or interrupted prior process may leave an incomplete
    // first marker. Its EXISTENCE still proves reuse; never overwrite it or
    // invent a fresh claim because its first locked identity is unavailable.
    let first_locked_inputs_sha256 = if first_recorded_use {
        Some(locked_inputs_sha256.clone())
    } else {
        let mut bytes = Vec::new();
        File::open(&first_path)?
            .take(65_537)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 65_536 {
            None
        } else {
            serde_json::from_slice::<FirstUse>(&bytes)
                .ok()
                .filter(|record| {
                    record.schema == JOURNAL_SCHEMA && record.raw_scope_sha256 == raw_scope_sha256
                })
                .map(|record| record.locked_inputs_sha256)
        }
    };
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let attempt = format!(
        "{}-{nanos}-{}",
        std::process::id(),
        ATTEMPT_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    );
    let value = AttemptStart {
        schema: JOURNAL_SCHEMA,
        raw_scope_sha256: &raw_scope_sha256,
        locked_inputs_sha256: &locked_inputs_sha256,
        first_recorded_use,
        raw_scope: &raw_scope,
        locked_inputs: &locked_inputs,
    };
    write_new(&directory.join(format!("{attempt}.start.json")), &value)
        .context("persist locked final-evaluation inputs before reading final data")?;
    Ok(FinalHoldoutUse {
        directory,
        attempt,
        raw_scope_sha256,
        locked_inputs_sha256,
        first_recorded_use,
        first_locked_inputs_sha256,
    })
}

#[derive(Serialize)]
struct AttemptStart<'a> {
    schema: &'a str,
    raw_scope_sha256: &'a str,
    locked_inputs_sha256: &'a str,
    first_recorded_use: bool,
    raw_scope: &'a serde_json::Value,
    locked_inputs: &'a serde_json::Value,
}

impl FinalHoldoutUse {
    pub(super) fn report_path(&self, candidate_root: &Path, suffix: &str) -> PathBuf {
        candidate_root.join(format!(
            "{}.{}.{suffix}",
            self.locked_inputs_sha256, self.attempt
        ))
    }

    pub(super) fn complete(&self, report_path: &Path, report: &serde_json::Value) -> Result<()> {
        write_new(report_path, report).context("persist write-once final research report")?;
        write_new(
            &self
                .directory
                .join(format!("{}.completed.json", self.attempt)),
            &serde_json::json!({
                "schema": JOURNAL_SCHEMA,
                "raw_scope_sha256": self.raw_scope_sha256,
                "locked_inputs_sha256": self.locked_inputs_sha256,
                "report_path": report_path,
                "report_sha256": hash_material(b"neoethos.final-report.v1\0", report)?,
                "first_recorded_use": self.first_recorded_use,
            }),
        )
        .context("persist write-once final-evaluation completion")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) struct TestDirectory(PathBuf);
    impl TestDirectory {
        pub(super) fn new() -> Self {
            let stamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "neoethos-final-use-test-{}-{stamp}-{}",
                std::process::id(),
                ATTEMPT_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path.canonicalize().unwrap())
        }
        pub(super) fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TestDirectory {
        fn drop(&mut self) {
            // This exact absolute directory was freshly and exclusively created
            // by this test, never a caller-provided data/journal directory.
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn public_scope_journal_tracks_raw_data_across_different_feature_fits() {
        use neoethos_search::data_selection::{
            CanonicalSearchEvaluatedWindowV1, CanonicalSearchInputReceiptV2,
        };
        let root = TestDirectory::new();
        let raw = neoethos_data::test_fixtures::ctrader_sample_feature_frame();
        let fitted = neoethos_data::test_fixtures::ctrader_test_feature_frame_with_normalization(
            &raw,
            0..80,
            None,
        )
        .unwrap();
        let make_scope = |frame: &neoethos_data::FeatureFrame, role| {
            let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(
                frame.provenance().bindings()[0].dataset_identity(),
                frame,
            )
            .unwrap();
            CanonicalSearchArtifactScopeV2::new(
                receipt,
                CanonicalSearchEvaluatedWindowV1::new(
                    role,
                    90,
                    100,
                    frame.timestamps[90],
                    frame.timestamps[99],
                )
                .unwrap(),
            )
            .unwrap()
        };
        let raw_scope = make_scope(&raw, CanonicalSearchWindowRoleV1::Holdout);
        let fitted_scope = make_scope(&fitted, CanonicalSearchWindowRoleV1::Holdout);
        assert_ne!(raw_scope.receipt(), fitted_scope.receipt());
        let first = begin(
            root.path(),
            &raw_scope,
            serde_json::json!({"feature_fit": "raw"}),
        )
        .unwrap();
        let changed_fit = begin(
            root.path(),
            &fitted_scope,
            serde_json::json!({"feature_fit": "normalized"}),
        )
        .unwrap();
        assert!(first.first_recorded_use);
        assert!(!changed_fit.first_recorded_use);
        assert_eq!(first.raw_scope_sha256, changed_fit.raw_scope_sha256);
        assert_ne!(first.locked_inputs_sha256, changed_fit.locked_inputs_sha256);
        assert!(
            begin(
                root.path(),
                &make_scope(&raw, CanonicalSearchWindowRoleV1::SelectionValidation),
                serde_json::json!({})
            )
            .is_err()
        );
    }

    #[test]
    fn retries_and_changed_candidates_consume_the_same_raw_window() {
        let root = TestDirectory::new();
        let raw = serde_json::json!({"raw_generation": "one", "rows": [90, 100]});
        let first = begin_material(
            root.path(),
            raw.clone(),
            serde_json::json!({"candidate": "a", "blend": 0.34}),
        )
        .unwrap();
        assert!(first.first_recorded_use);
        for lock in [
            serde_json::json!({"candidate": "a", "blend": 0.34}),
            serde_json::json!({"candidate": "a", "blend": 0.5}),
            serde_json::json!({"candidate": "b", "blend": 0.34}),
        ] {
            let retry = begin_material(root.path(), raw.clone(), lock).unwrap();
            assert!(!retry.first_recorded_use);
            assert_eq!(retry.raw_scope_sha256, first.raw_scope_sha256);
            assert_eq!(
                retry.first_locked_inputs_sha256.as_deref(),
                Some(first.locked_inputs_sha256.as_str())
            );
        }
        let changed = begin_material(
            root.path(),
            serde_json::json!({"raw_generation": "two", "rows": [90, 100]}),
            serde_json::json!({"candidate": "a"}),
        )
        .unwrap();
        assert!(changed.first_recorded_use);
        assert_ne!(changed.raw_scope_sha256, first.raw_scope_sha256);
    }

    #[test]
    fn changed_blend_changes_locked_identity_and_completion_cannot_be_overwritten() {
        let root = TestDirectory::new();
        let raw = serde_json::json!({"raw_generation": "one"});
        let first =
            begin_material(root.path(), raw.clone(), serde_json::json!({"blend": 0.34})).unwrap();
        let second = begin_material(root.path(), raw, serde_json::json!({"blend": 0.5})).unwrap();
        assert_ne!(first.locked_inputs_sha256, second.locked_inputs_sha256);
        let path = first.report_path(root.path(), "report.json");
        let report = serde_json::json!({"profit": 10});
        first.complete(&path, &report).unwrap();
        assert!(
            first
                .complete(&path, &serde_json::json!({"profit": 99}))
                .is_err()
        );
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&std::fs::read(path).unwrap()).unwrap(),
            report
        );
        assert!(
            first
                .directory
                .join(format!("{}.completed.json", first.attempt))
                .is_file()
        );
    }

    #[test]
    fn simultaneous_starts_have_only_one_first_use_and_partial_marker_never_becomes_fresh() {
        let root = TestDirectory::new();
        let raw = serde_json::json!({"raw_generation": "one"});
        let starts = std::thread::scope(|scope| {
            let handles = (0..4)
                .map(|candidate| {
                    let root = root.path();
                    let raw = &raw;
                    scope.spawn(move || {
                        begin_material(
                            root,
                            raw.clone(),
                            serde_json::json!({"candidate": candidate}),
                        )
                        .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(
            starts
                .iter()
                .filter(|value| value.first_recorded_use)
                .count(),
            1
        );
        let other = serde_json::json!({"raw_generation": "interrupted"});
        let key = hash_material(b"neoethos.final-raw-window.v1\0", &other).unwrap();
        let directory = root.path().join("final_holdout_uses").join(key);
        std::fs::create_dir_all(&directory).unwrap();
        File::create(directory.join("first-start.json")).unwrap();
        let retry =
            begin_material(root.path(), other, serde_json::json!({"candidate": "next"})).unwrap();
        assert!(!retry.first_recorded_use);
        assert!(retry.first_locked_inputs_sha256.is_none());
    }
}
