//! Executable, model-free authorization/copy tests.
//!
//! This test includes the small production boundary directly and runs with
//! `rustc --test`; it therefore exercises real temp-directory reads/writes
//! without compiling the unrelated model graph.

// The app consumes every error variant; this isolated harness intentionally
// exercises only the authorization/copy subset and has no app-wide callers.
#[allow(dead_code)]
#[path = "../src/server/promotion_authorization.rs"]
mod promotion_authorization;

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use promotion_authorization::{
    CompositeAuthorityChecksV3, PromotionAuthorizationError,
    REQUIRED_COMPOSITE_PROMOTION_AUTHORITY_KIND_V3, authorize_exact_composite_promotion_v3,
    copy_model_tree_if_authorized, validate_promotion_path_leafs,
};

struct TempRoot(PathBuf);

impl TempRoot {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "neoethos-promotion-auth-{label}-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path).expect("create unique temp root");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempRoot {
    fn drop(&mut self) {
        if self
            .0
            .file_name()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|name| name.starts_with("neoethos-promotion-auth-"))
        {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn fully_verified_checks() -> CompositeAuthorityChecksV3 {
    CompositeAuthorityChecksV3 {
        exact_receipt_config_sidecar: true,
        exact_composite_scope: true,
        required_evidence_complete: true,
        required_evidence_passed: true,
    }
}

fn seed_models(root: &Path) {
    let source = root.join("models").join("EURUSD").join("M1");
    std::fs::create_dir_all(source.join("nested")).expect("create model fixture");
    std::fs::write(source.join("model.bin"), b"model").expect("write model fixture");
    std::fs::write(source.join("nested").join("state.bin"), b"state")
        .expect("write nested fixture");
}

fn assert_zero_copy(
    root: &TempRoot,
    authorization: Result<
        promotion_authorization::PromotionCopyPermit,
        PromotionAuthorizationError,
    >,
) {
    seed_models(root.path());
    let live_root = root.path().join("live_models");
    let result =
        copy_model_tree_if_authorized(authorization, &root.path().join("models"), &live_root);
    assert!(result.is_err(), "rejected authority must not copy");
    assert!(
        !live_root.exists(),
        "rejected authority created a live destination"
    );
}

#[test]
fn traversal_absolute_prefix_and_noncanonical_tf_are_zero_copy() {
    let unsafe_symbols = [
        "",
        ".",
        "..",
        "../EURUSD",
        "EUR/USD",
        "EUR\\USD",
        "/tmp/EURUSD",
        r"C:\escape",
        r"\\server\share",
        "CON",
        "LPT1.txt",
        "EURUSD.",
        "EUR?USD",
        "EUR\u{1f}USD",
    ];
    for (index, symbol) in unsafe_symbols.into_iter().enumerate() {
        let root = TempRoot::new(&format!("unsafe-symbol-{index}"));
        let error = validate_promotion_path_leafs(symbol, "M1")
            .expect_err("unsafe symbol must fail before path construction");
        assert_zero_copy(&root, Err(error));
    }

    let root = TempRoot::new("noncanonical-tf");
    let error = validate_promotion_path_leafs("EURUSD", "H2")
        .expect_err("synthetic/noncanonical timeframe must fail");
    assert_zero_copy(&root, Err(error));
}

#[test]
fn current_v1_summary_is_typed_unsupported_and_zero_copy() {
    let root = TempRoot::new("v1");
    let path = validate_promotion_path_leafs("EURUSD", "M1").expect("safe leaves");
    let authorization = authorize_exact_composite_promotion_v3(
        path,
        "neoethos.search-promotion-summary.v1",
        fully_verified_checks(),
    );
    assert!(matches!(
        &authorization,
        Err(PromotionAuthorizationError::UnsupportedEvidenceSchema { .. })
    ));
    assert_zero_copy(&root, authorization);
}

#[test]
fn malformed_and_swapped_authorities_are_zero_copy() {
    let malformed_root = TempRoot::new("malformed");
    assert_zero_copy(
        &malformed_root,
        Err(PromotionAuthorizationError::MalformedModelTargets {
            reason: "fixture malformed".to_owned(),
        }),
    );

    let swapped_root = TempRoot::new("swapped");
    let path = validate_promotion_path_leafs("EURUSD", "M1").expect("safe leaves");
    let mut checks = fully_verified_checks();
    checks.exact_receipt_config_sidecar = false;
    let authorization = authorize_exact_composite_promotion_v3(
        path,
        REQUIRED_COMPOSITE_PROMOTION_AUTHORITY_KIND_V3,
        checks,
    );
    assert!(matches!(
        &authorization,
        Err(PromotionAuthorizationError::PromotionSummaryMismatch)
    ));
    assert_zero_copy(&swapped_root, authorization);
}

#[test]
fn missing_or_failed_exact_evidence_is_zero_copy() {
    let missing_root = TempRoot::new("missing-evidence");
    let path = validate_promotion_path_leafs("EURUSD", "M1").expect("safe leaves");
    let mut checks = fully_verified_checks();
    checks.required_evidence_complete = false;
    let authorization = authorize_exact_composite_promotion_v3(
        path,
        REQUIRED_COMPOSITE_PROMOTION_AUTHORITY_KIND_V3,
        checks,
    );
    assert!(matches!(
        &authorization,
        Err(PromotionAuthorizationError::MissingHeldOutEvidence { .. })
    ));
    assert_zero_copy(&missing_root, authorization);

    let failed_root = TempRoot::new("failed-evidence");
    let path = validate_promotion_path_leafs("EURUSD", "M1").expect("safe leaves");
    let mut checks = fully_verified_checks();
    checks.required_evidence_passed = false;
    let authorization = authorize_exact_composite_promotion_v3(
        path,
        REQUIRED_COMPOSITE_PROMOTION_AUTHORITY_KIND_V3,
        checks,
    );
    assert!(matches!(
        &authorization,
        Err(PromotionAuthorizationError::FailedHeldOutEvidence { .. })
    ));
    assert_zero_copy(&failed_root, authorization);
}

#[test]
fn only_a_fully_verified_v3_permit_can_copy() {
    let root = TempRoot::new("v3");
    seed_models(root.path());
    let path = validate_promotion_path_leafs("EURUSD", "M1").expect("safe leaves");
    let authorization = authorize_exact_composite_promotion_v3(
        path,
        REQUIRED_COMPOSITE_PROMOTION_AUTHORITY_KIND_V3,
        fully_verified_checks(),
    );
    let live_root = root.path().join("live_models");
    let copied =
        copy_model_tree_if_authorized(authorization, &root.path().join("models"), &live_root)
            .expect("fully verified synthetic v3 permit copies");
    assert_eq!(copied.files_copied, 2);
    assert_eq!(
        std::fs::read(copied.destination.join("model.bin")).unwrap(),
        b"model"
    );
    assert_eq!(
        std::fs::read(copied.destination.join("nested").join("state.bin")).unwrap(),
        b"state"
    );
}
