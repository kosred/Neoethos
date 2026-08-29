use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashSet};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};

const FIXTURE_PACKAGE: &str = "neoethos-resident-search-slice2-ui 0.0.0";
const REPO_CONFIG_SHA256: &str = "de69922d58cddec2c0383536b40ea2491a23c56a34510c5bde7488d13964fdb8";
const ENCLOSING_CONFIG_SHA256: &str =
    "5ee03587848a82cc5b50a2d41ae2cd7a56c6da1a3e320c865a54262442757b38";
const R7_RED_ONLY_DEAD_CODE_MARKER: &str =
    "// R7 RED-only: first production GREEN must consume real state and remove this allow.";
const AUTHORITY_COMMIT: &str = "512f2f68ef68b63ebf6469f67e0b749e77666309";
const AUTHORITY_PARENT: &str = "110b4c1aa1c1700655fee5ad4ea2276e52e733dc";
const AUTHORITY_TREE: &str = "1274591f1318ee7be90bc634a490b2434041d4f1";
const ROOT_LOCK_SHA256: &str = "725cc6fb8645a0d7d9cd11f32bab01dcc8cc3de0497a9df5472886e20eb2167f";
const VECTOR_TA_FILE_COUNT: usize = 1_077;
const VECTOR_TA_TREE_SHA256: &str =
    "def4551c993af6e9149c6a93fee1733a43c77629d132d28eee1c1fc16bd224b5";
const CARGO_IDENTITY: &str = concat!(
    "cargo 1.96.0-nightly (888f67534 2026-03-30)\n",
    "release: 1.96.0-nightly\n",
    "commit-hash: 888f675344eb1cf2308fd53183e667bdd2c58e51\n",
    "commit-date: 2026-03-30\n",
    "host: x86_64-pc-windows-msvc\n",
    "libgit2: 1.9.2 (sys:0.20.4 vendored)\n",
    "libcurl: 8.19.0-DEV (sys:0.4.87+curl-8.19.0 vendored ssl:Schannel)\n",
    "os: Windows 10.0.26200 (Windows 11 Professional) [64-bit]\n",
);
const CARGO_IDENTITY_SHA256: &str =
    "7d4a0723c4202c639b08fdf5a12b01f4cd6eaad342126018e401c6c01ce794a3";
const RUSTC_IDENTITY: &str = concat!(
    "rustc 1.96.0-nightly (bcded3316 2026-04-06)\n",
    "binary: rustc\n",
    "commit-hash: bcded331651b60a0383b3ff51db4f24c4495ac53\n",
    "commit-date: 2026-04-06\n",
    "host: x86_64-pc-windows-msvc\n",
    "release: 1.96.0-nightly\n",
    "LLVM version: 22.1.2\n",
);
const RUSTC_IDENTITY_SHA256: &str =
    "465eee234c23db98d3a15572455693d7e8e2284b2bf80b8db0b9691adf7ef643";
const API_SURFACE_V3: &str = concat!(
    "gpu|enum|ResidentSearchTryCompleteV3\n",
    "gpu|method|ResidentSearchArchiveStagedV3|enqueue_evolve_and_publish_v3|fn(self)->Result<ResidentSearchGenerationChainV3,ResidentSearchRejectedAuthorityV3<Self>>\n",
    "gpu|method|ResidentSearchGenerationChainV3|enqueue_score_and_rank_v3|fn(self)->Result<ResidentSearchRankEnqueuedV3,ResidentSearchRejectedAuthorityV3<Self>>\n",
    "gpu|method|ResidentSearchGenerationChainV3|enqueue_terminal_seal_v3|fn(self)->Result<ResidentSearchTerminalPendingV3,ResidentSearchRejectedAuthorityV3<Self>>\n",
    "gpu|method|ResidentSearchRankEnqueuedV3|enqueue_stage_archive_from_rank_v3|fn(self)->Result<ResidentSearchArchiveStagedV3,ResidentSearchRejectedAuthorityV3<Self>>\n",
    "gpu|method|ResidentSearchRejectedAuthorityV3<A>|into_parts_v3|fn(self)->(ResidentSearchTransitionErrorV3,A)\n",
    "gpu|method|ResidentSearchTerminalPendingV3|try_complete_v3|fn(self)->Result<ResidentSearchTryCompleteV3,ResidentSearchTransitionErrorV3>\n",
    "gpu|module|resident_search_slice2_v3\n",
    "gpu|struct|ResidentArchiveKnnCalibrationReceiptV2\n",
    "gpu|struct|ResidentSearchArchiveStagedV3\n",
    "gpu|struct|ResidentSearchGenerationChainV3\n",
    "gpu|struct|ResidentSearchRankEnqueuedV3\n",
    "gpu|struct|ResidentSearchRejectedAuthorityV3<A>\n",
    "gpu|struct|ResidentSearchTerminalPendingV3\n",
    "gpu|struct|ResidentSearchTerminalReceiptV3\n",
    "gpu|struct|ResidentSearchTransitionErrorV3\n",
    "gpu|variant|ResidentSearchTryCompleteV3::Complete(ResidentSearchTerminalReceiptV3)\n",
    "gpu|variant|ResidentSearchTryCompleteV3::NotReady(ResidentSearchTerminalPendingV3)\n",
    "search|module|resident_search_slice2_v3\n",
    "search|reexport|FullResidentDiscoveryDeadlineReceiptV1|crate::gpu_resident_current_config_plan_v1::FullResidentDiscoveryDeadlineReceiptV1\n",
    "search|reexport|ResidentArchiveKnnCalibrationReceiptV2|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentArchiveKnnCalibrationReceiptV2\n",
    "search|reexport|ResidentSearchArchiveStagedV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchArchiveStagedV3\n",
    "search|reexport|ResidentSearchGenerationChainV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchGenerationChainV3\n",
    "search|reexport|ResidentSearchRankEnqueuedV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchRankEnqueuedV3\n",
    "search|reexport|ResidentSearchRejectedAuthorityV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchRejectedAuthorityV3\n",
    "search|reexport|ResidentSearchTerminalPendingV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchTerminalPendingV3\n",
    "search|reexport|ResidentSearchTerminalReceiptV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchTerminalReceiptV3\n",
    "search|reexport|ResidentSearchTransitionErrorV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchTransitionErrorV3\n",
    "search|reexport|ResidentSearchTryCompleteV3|neoethos_gpu_cuda::resident_search_slice2_v3::ResidentSearchTryCompleteV3\n",
);

const HOST_ENV_ALLOWLIST: &[&str] = &[
    "SystemRoot",
    "WINDIR",
    "ComSpec",
    "PATH",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "LOCALAPPDATA",
    "APPDATA",
    "PROGRAMDATA",
    "ProgramFiles",
    "ProgramFiles(x86)",
    "ProgramW6432",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "PROCESSOR_IDENTIFIER",
    "RUSTUP_HOME",
    "CARGO_HOME",
    "VSINSTALLDIR",
    "VCINSTALLDIR",
    "VCToolsInstallDir",
    "WindowsSdkDir",
    "WindowsSDKVersion",
    "UCRTVersion",
    "UniversalCRTSdkDir",
    "INCLUDE",
    "LIB",
    "LIBPATH",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FeatureMode {
    On,
    Off,
}

#[derive(Clone, Copy, Debug)]
struct UiCase {
    bin: &'static str,
    source: &'static str,
    feature: FeatureMode,
    expected_code: Option<&'static str>,
    line: usize,
    column_start: usize,
    column_end: usize,
    stderr: Option<&'static str>,
}

const CASES: &[UiCase] = &[
    UiCase {
        bin: "pass_typed_surface",
        source: "pass/typed_surface.rs",
        feature: FeatureMode::On,
        expected_code: None,
        line: 0,
        column_start: 0,
        column_end: 0,
        stderr: None,
    },
    UiCase {
        bin: "fail_clone_owner_e0599",
        source: "fail/clone_owner_e0599.rs",
        feature: FeatureMode::On,
        expected_code: Some("E0599"),
        line: 8,
        column_start: 13,
        column_end: 18,
        stderr: Some("fail/clone_owner_e0599.stderr"),
    },
    UiCase {
        bin: "fail_copy_owner_e0277",
        source: "fail/copy_owner_e0277.rs",
        feature: FeatureMode::On,
        expected_code: Some("E0277"),
        line: 10,
        column_start: 18,
        column_end: 25,
        stderr: Some("fail/copy_owner_e0277.stderr"),
    },
    UiCase {
        bin: "fail_read_chain_inner_e0616",
        source: "fail/read_chain_inner_e0616.rs",
        feature: FeatureMode::On,
        expected_code: Some("E0616"),
        line: 8,
        column_start: 21,
        column_end: 26,
        stderr: Some("fail/read_chain_inner_e0616.stderr"),
    },
    UiCase {
        bin: "fail_read_ranked_inner_e0616",
        source: "fail/read_ranked_inner_e0616.rs",
        feature: FeatureMode::On,
        expected_code: Some("E0616"),
        line: 8,
        column_start: 21,
        column_end: 26,
        stderr: Some("fail/read_ranked_inner_e0616.stderr"),
    },
    UiCase {
        bin: "fail_read_staged_inner_e0616",
        source: "fail/read_staged_inner_e0616.rs",
        feature: FeatureMode::On,
        expected_code: Some("E0616"),
        line: 8,
        column_start: 21,
        column_end: 26,
        stderr: Some("fail/read_staged_inner_e0616.stderr"),
    },
    UiCase {
        bin: "fail_read_pending_inner_e0616",
        source: "fail/read_pending_inner_e0616.rs",
        feature: FeatureMode::On,
        expected_code: Some("E0616"),
        line: 8,
        column_start: 21,
        column_end: 26,
        stderr: Some("fail/read_pending_inner_e0616.stderr"),
    },
    UiCase {
        bin: "fail_call_staged_constructor_e0624",
        source: "fail/call_staged_constructor_e0624.rs",
        feature: FeatureMode::On,
        expected_code: Some("E0624"),
        line: 8,
        column_start: 44,
        column_end: 58,
        stderr: Some("fail/call_staged_constructor_e0624.stderr"),
    },
    UiCase {
        bin: "fail_construct_ranked_state_e0451",
        source: "fail/construct_ranked_state_e0451.rs",
        feature: FeatureMode::On,
        expected_code: Some("E0451"),
        line: 8,
        column_start: 46,
        column_end: 54,
        stderr: Some("fail/construct_ranked_state_e0451.stderr"),
    },
    UiCase {
        bin: "fail_novelty_receipt_as_full_deadline_e0308",
        source: "fail/novelty_receipt_as_full_deadline_e0308.rs",
        feature: FeatureMode::On,
        expected_code: Some("E0308"),
        line: 11,
        column_start: 27,
        column_end: 40,
        stderr: Some("fail/novelty_receipt_as_full_deadline_e0308.stderr"),
    },
    UiCase {
        bin: "fail_feature_gate_off_e0432",
        source: "fail/feature_gate_off_e0432.rs",
        feature: FeatureMode::Off,
        expected_code: Some("E0432"),
        line: 1,
        column_start: 22,
        column_end: 47,
        stderr: Some("fail/feature_gate_off_e0432.stderr"),
    },
];

struct RunContext {
    repo: PathBuf,
    fixture: PathBuf,
    manifest: PathBuf,
    cargo: PathBuf,
    rustc: PathBuf,
    host_environment: Vec<(OsString, OsString)>,
    outer_target: PathBuf,
    target_on: PathBuf,
    target_off: PathBuf,
    target_doc: PathBuf,
    evidence: PathBuf,
}

struct CapturedProcess {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl CapturedProcess {
    fn stdout_text(&self, name: &str) -> Result<&str, String> {
        std::str::from_utf8(&self.stdout)
            .map_err(|error| format!("{name} stdout is not valid UTF-8: {error}"))
    }

    fn stderr_text(&self, name: &str) -> Result<&str, String> {
        std::str::from_utf8(&self.stderr)
            .map_err(|error| format!("{name} process stderr is not valid UTF-8: {error}"))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct EventCounts {
    info: usize,
    warning: usize,
    error: usize,
    other: usize,
}

#[derive(Debug)]
struct AuthoredError {
    code: Option<String>,
    line: usize,
    column_start: usize,
    column_end: usize,
    rendered: String,
}

struct Observation {
    status: ExitStatus,
    build_finished_success: bool,
    selected_warning_events: usize,
    selected_error_events: usize,
    dependency_error_events: usize,
    wrong_primary_spans: usize,
    authored_errors: Vec<AuthoredError>,
    process_stderr: EventCounts,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn normalized_path_text(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    text.strip_prefix("//?/").unwrap_or(&text).to_owned()
}

fn ordinary_absolute_path(path: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from(normalized_path_text(path).replace('/', "\\"))
    }
    #[cfg(not(windows))]
    {
        path.to_path_buf()
    }
}

fn canonical(path: &Path, label: &str) -> Result<PathBuf, String> {
    fs::canonicalize(path).map_err(|error| format!("cannot canonicalize {label}: {error}"))
}

fn same_path(left: &Path, right: &Path) -> bool {
    normalized_path_text(left).eq_ignore_ascii_case(&normalized_path_text(right))
}

fn path_is_within(path: &Path, root: &Path) -> bool {
    let path = normalized_path_text(path).to_ascii_lowercase();
    let root = normalized_path_text(root).to_ascii_lowercase();
    path == root || path.starts_with(&(root + "/"))
}

fn verify_fixture_tree_does_not_escape(fixture: &Path, repo: &Path) -> Result<(), String> {
    let mut pending = vec![fixture.to_path_buf()];
    let mut seen = HashSet::new();
    while let Some(directory) = pending.pop() {
        let resolved = canonical(&directory, "fixture entry")?;
        if !path_is_within(&resolved, repo) {
            return Err(format!(
                "fixture symlink escapes repository: {}",
                normalized_path_text(&directory)
            ));
        }
        if !seen.insert(normalized_path_text(&resolved).to_ascii_lowercase()) {
            continue;
        }
        for entry in fs::read_dir(&directory)
            .map_err(|error| format!("cannot enumerate fixture tree: {error}"))?
        {
            let entry = entry.map_err(|error| format!("cannot read fixture entry: {error}"))?;
            let path = entry.path();
            let resolved_entry = canonical(&path, "fixture child")?;
            if !path_is_within(&resolved_entry, repo) {
                return Err(format!(
                    "fixture symlink escapes repository: {}",
                    normalized_path_text(&path)
                ));
            }
            if resolved_entry.is_dir() {
                pending.push(path);
            }
        }
    }
    Ok(())
}

fn validate_case_ledger(fixture: &Path) -> Result<(), String> {
    if CASES.len() != 11 {
        return Err(format!(
            "case ledger has {} entries, expected 11",
            CASES.len()
        ));
    }

    let mut bins = HashSet::new();
    let mut sources = HashSet::new();
    let mut stderr_paths = HashSet::new();
    for case in CASES {
        if !bins.insert(case.bin) {
            return Err(format!("duplicate bin in ledger: {}", case.bin));
        }
        if !sources.insert(case.source) {
            return Err(format!("duplicate source in ledger: {}", case.source));
        }
        if !fixture.join(case.source).is_file() {
            return Err(format!("missing ledger source: {}", case.source));
        }
        match (case.expected_code, case.stderr) {
            (None, None) => {
                if case.line != 0 || case.column_start != 0 || case.column_end != 0 {
                    return Err("positive case unexpectedly declares a span".to_owned());
                }
            }
            (Some(_), Some(path)) => {
                if !stderr_paths.insert(path) {
                    return Err(format!("duplicate stderr path in ledger: {path}"));
                }
                if case.line == 0 || case.column_start == 0 || case.column_end <= case.column_start
                {
                    return Err(format!("invalid negative span for {}", case.bin));
                }
            }
            _ => return Err(format!("incomplete expectation for {}", case.bin)),
        }
    }

    let positives = CASES
        .iter()
        .filter(|case| case.expected_code.is_none())
        .count();
    let feature_on_negatives = CASES
        .iter()
        .filter(|case| case.feature == FeatureMode::On && case.expected_code.is_some())
        .count();
    let feature_off_negatives = CASES
        .iter()
        .filter(|case| case.feature == FeatureMode::Off && case.expected_code.is_some())
        .count();
    if positives != 1 || feature_on_negatives != 9 || feature_off_negatives != 1 {
        return Err(format!(
            "invalid case partition: positives={positives}, on-negatives={feature_on_negatives}, off-negatives={feature_off_negatives}"
        ));
    }
    if CASES[0].bin != "pass_typed_surface" || CASES[0].feature != FeatureMode::On {
        return Err("positive case is not first and feature-on".to_owned());
    }
    Ok(())
}

fn verify_red_only_dead_code_allows(repo: &Path) -> Result<(), String> {
    let source_path = repo
        .join("crates")
        .join("neoethos-gpu-cuda")
        .join("src")
        .join("resident_search_slice2_v3.rs");
    let source = fs::read_to_string(&source_path)
        .map_err(|error| format!("cannot read R7 canonical GPU source: {error}"))?
        .replace("\r\n", "\n")
        .replace('\r', "\n");
    let allow = "#[allow(dead_code)]";
    let allow_count = source.matches(allow).count();
    let marker_count = source.matches(R7_RED_ONLY_DEAD_CODE_MARKER).count();
    if allow_count != 4 || marker_count != 4 || source.matches("dead_code").count() != 4 {
        return Err(format!(
            "R7 RED-only dead-code suppression drift: allows={allow_count}, markers={marker_count}, tokens={}",
            source.matches("dead_code").count()
        ));
    }

    let field_owners = [
        "ResidentArchiveKnnCalibrationReceiptV2",
        "ResidentSearchTerminalReceiptV3",
        "ResidentSearchTransitionErrorV3",
    ];
    for owner in field_owners {
        let expected = format!(
            "pub struct {owner} {{\n    {R7_RED_ONLY_DEAD_CODE_MARKER}\n    {allow}\n    inner: core::convert::Infallible,\n}}"
        );
        if source.matches(&expected).count() != 1 {
            return Err(format!("R7 RED-only field suppression drift for {owner}"));
        }
    }

    let constructor = format!(
        "    {R7_RED_ONLY_DEAD_CODE_MARKER}\n    {allow}\n    pub(crate) fn from_ranked_v3("
    );
    if source.matches(&constructor).count() != 1 {
        return Err("R7 RED-only staged-constructor suppression drift".to_owned());
    }
    Ok(())
}

fn config_candidates(repo: &Path) -> Result<Vec<PathBuf>, String> {
    let mut candidates = Vec::new();
    for directory in repo.ancestors() {
        candidates.push(directory.join(".cargo").join("config.toml"));
        candidates.push(directory.join(".cargo").join("config"));
    }

    let cargo_home = match env::var_os("CARGO_HOME") {
        Some(value) => PathBuf::from(value),
        None => PathBuf::from(
            env::var_os("USERPROFILE")
                .ok_or_else(|| "USERPROFILE is absent while CARGO_HOME is unset".to_owned())?,
        )
        .join(".cargo"),
    };
    candidates.push(cargo_home.join("config.toml"));
    candidates.push(cargo_home.join("config"));

    let mut unique = HashSet::new();
    let mut found = Vec::new();
    for candidate in candidates {
        if candidate.exists() {
            let resolved = canonical(&candidate, "Cargo config")?;
            let key = normalized_path_text(&resolved).to_ascii_lowercase();
            if unique.insert(key) {
                found.push(resolved);
            }
        }
    }
    Ok(found)
}

fn verify_cargo_config_inventory(repo: &Path) -> Result<(), String> {
    let configs = config_candidates(repo)?;
    if configs.len() != 2 {
        return Err(format!(
            "unexpected Cargo config inventory ({} files): {:?}",
            configs.len(),
            configs
        ));
    }

    let repo_config = canonical(&repo.join(".cargo").join("config.toml"), "repo config")?;
    let mut saw_repo = false;
    let mut enclosing_count = 0;
    for config in configs {
        let bytes = fs::read(&config)
            .map_err(|error| format!("cannot read Cargo config {}: {error}", config.display()))?;
        let digest = sha256(&bytes);
        if same_path(&config, &repo_config) {
            saw_repo = true;
            if digest != REPO_CONFIG_SHA256 {
                return Err(format!("repository Cargo config drift: {digest}"));
            }
        } else {
            enclosing_count += 1;
            if digest != ENCLOSING_CONFIG_SHA256 {
                return Err(format!("enclosing Cargo config drift: {digest}"));
            }
        }
    }
    if !saw_repo || enclosing_count != 1 {
        return Err(
            "Cargo config inventory did not bind repo plus one enclosing config".to_owned(),
        );
    }
    Ok(())
}

fn canonical_tool(leaf: &str, expected: &Path, label: &str) -> Result<PathBuf, String> {
    let path = env::var_os("PATH").ok_or_else(|| "allowlisted PATH is absent".to_owned())?;
    let discovered = env::split_paths(&path)
        .map(|directory| directory.join(leaf))
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| format!("{leaf} is absent from allowlisted PATH"))?;
    let parent = canonical(
        discovered
            .parent()
            .ok_or_else(|| format!("discovered {leaf} has no parent"))?,
        &format!("{label} executable parent"),
    )?;
    let tool = ordinary_absolute_path(&parent).join(leaf);
    if !same_path(&tool, expected) {
        return Err(format!(
            "canonical {label} path mismatch: {}",
            normalized_path_text(&tool)
        ));
    }
    Ok(tool)
}

fn canonical_cargo() -> Result<PathBuf, String> {
    canonical_tool(
        "cargo.exe",
        Path::new(r"C:\Users\konst\.cargo\bin\cargo.exe"),
        "Cargo",
    )
}

fn canonical_rustc() -> Result<PathBuf, String> {
    canonical_tool(
        "rustc.exe",
        Path::new(r"C:\Users\konst\.cargo\bin\rustc.exe"),
        "rustc",
    )
}

fn git_output(repo: &Path, arguments: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .current_dir(repo)
        .args(arguments)
        .output()
        .map_err(|error| format!("cannot run git {arguments:?}: {error}"))?;
    if !output.status.success() || !output.stderr.is_empty() {
        return Err(format!(
            "git {arguments:?} failed: exit={:?}, stderr={:?}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| format!("git {arguments:?} stdout is not UTF-8: {error}"))
}

fn normalized_lf_digest(path: &Path, label: &str) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {label}: {error}"))?;
    if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        return Err(format!("{label} has a UTF-8 BOM"));
    }
    let text =
        std::str::from_utf8(&bytes).map_err(|error| format!("{label} is not UTF-8: {error}"))?;
    Ok(sha256(normalized_lf(text).as_bytes()))
}

fn verify_authority_and_protected(repo: &Path) -> Result<(), String> {
    let authority = normalized_lf(&git_output(
        repo,
        &["show", "-s", "--format=%P%n%T", AUTHORITY_COMMIT],
    )?);
    let expected = format!("{AUTHORITY_PARENT}\n{AUTHORITY_TREE}\n");
    if authority != expected {
        return Err(format!("authority commit parent/tree drift: {authority:?}"));
    }
    let ancestor = Command::new("git")
        .current_dir(repo)
        .args(["merge-base", "--is-ancestor", AUTHORITY_COMMIT, "HEAD"])
        .status()
        .map_err(|error| format!("cannot test authority ancestry: {error}"))?;
    if !ancestor.success() {
        return Err("current HEAD does not descend from the R7 authority commit".to_owned());
    }

    const PROTECTED: &[(&str, &str)] = &[
        (
            "crates/neoethos-gpu-cuda/src/resident_search_v2.rs",
            "661ae2a779b534f725a372360fd407635bc7cf7e864691bb842aa8241a7b97f4",
        ),
        (
            "crates/neoethos-gpu-cuda/src/resident_search_slice2_admission_v2.rs",
            "c05cd1066a4dea9681b535c76bbbd5fb09ef431cef9a01ffc390bfe326b48232",
        ),
        (
            "crates/neoethos-gpu-cuda/build.rs",
            "4b74a0e84e9b42a455be8809ee0b151167217d6c8b14aec364ba62128060008a",
        ),
        (
            "Cargo.toml",
            "84007747a2ad69f2a0ccab4a1391284403afd611b832dc934fac9c7ff3b51410",
        ),
        ("Cargo.lock", ROOT_LOCK_SHA256),
    ];
    for (relative, expected_digest) in PROTECTED {
        let bytes = fs::read(repo.join(relative))
            .map_err(|error| format!("cannot read protected {relative}: {error}"))?;
        let observed = sha256(&bytes);
        if observed != *expected_digest {
            return Err(format!("protected file drift: {relative} {observed}"));
        }
    }
    for (relative, expected_digest) in [
        (
            "docs/superpowers/specs/2026-08-28-resident-search-slice2-archive-knn-design.md",
            "52b166cc52a09358e47e9da3ce1daad5a692783fea820027fb4db491d2b1431a",
        ),
        (
            "docs/superpowers/plans/2026-08-29-resident-search-slice2-r6-combined-preallocation.md",
            "e3e876948d98bee1f6f8bdf351011ad3480a40d46505f378da759d0223e25a27",
        ),
        (
            "audit/resident-search-slice2-design-v5.sha256",
            "3f370ffe7561dc26e99b1834b482d0399a188befa2bd68da2b661e771b7de144",
        ),
        (
            "audit/resident-search-slice2-design-v8.sha256",
            "413263ceaa403e486ed626571407a1af3f00ce6ee1fe5ebe02504eef1454e443",
        ),
    ] {
        let observed = normalized_lf_digest(&repo.join(relative), relative)?;
        if observed != expected_digest {
            return Err(format!(
                "protected normalized-LF drift: {relative} {observed}"
            ));
        }
    }
    Ok(())
}

fn collect_regular_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut pending = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = pending.pop() {
        verify_plain_directory(&directory, "tree directory")?;
        for entry in fs::read_dir(&directory)
            .map_err(|error| format!("cannot enumerate {}: {error}", directory.display()))?
        {
            let path = entry
                .map_err(|error| format!("cannot read tree entry: {error}"))?
                .path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
            if metadata.file_type().is_symlink() {
                return Err(format!("tree contains a symlink: {}", path.display()));
            }
            #[cfg(windows)]
            {
                use std::os::windows::fs::MetadataExt;
                if metadata.file_attributes() & 0x0400 != 0 {
                    return Err(format!("tree contains a reparse point: {}", path.display()));
                }
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() {
                files.push(path);
            } else {
                return Err(format!(
                    "tree contains a non-file entry: {}",
                    path.display()
                ));
            }
        }
    }
    Ok(files)
}

fn verify_vendor_closure(repo: &Path) -> Result<(), String> {
    let vendor = repo.join("vendor");
    let vector = vendor.join("vector-ta-0.2.9-patched");
    let mut rows = Vec::new();
    for path in collect_regular_files(&vector)? {
        let relative = path
            .strip_prefix(&vector)
            .map_err(|_| "VectorTA file escaped its root".to_owned())?;
        let relative = relative.to_string_lossy().replace('\\', "/");
        let bytes = fs::read(&path)
            .map_err(|error| format!("cannot read VectorTA file {relative}: {error}"))?;
        rows.push(format!("{}  {relative}", sha256(&bytes)));
    }
    rows.sort();
    if rows.len() != VECTOR_TA_FILE_COUNT {
        return Err(format!("VectorTA file-count drift: {}", rows.len()));
    }
    let manifest = format!("{}\n", rows.join("\n"));
    let digest = sha256(manifest.as_bytes());
    if digest != VECTOR_TA_TREE_SHA256 {
        return Err(format!("VectorTA tree-manifest drift: {digest}"));
    }
    for relative in [
        "lightgbm3",
        "lightgbm3-sys",
        "xgboost_lib-sys",
        "sklears-core",
        "rlkit",
        "cubecl-runtime-0.10.0-patched",
        "cubecl-cuda-0.10.0-patched",
        "cubek-matmul-0.2.0-patched",
        "cubek-convolution-0.2.0-patched",
        "catboost-rust-0.3.8-patched",
    ] {
        verify_plain_directory(&vendor.join(relative), "inactive vendor patch")?;
        fs::File::open(vendor.join(relative).join("Cargo.toml"))
            .map_err(|error| format!("cannot read vendor/{relative}/Cargo.toml: {error}"))?;
    }
    if !git_output(repo, &["ls-files", "vendor"])?.is_empty() {
        return Err("prepared vendor closure unexpectedly contains tracked files".to_owned());
    }
    Ok(())
}

fn verify_source_shape(repo: &Path) -> Result<(), String> {
    let gpu_root = repo.join("crates/neoethos-gpu-cuda/src");
    let canonical_path = gpu_root.join("resident_search_slice2_v3.rs");
    let canonical_source = fs::read_to_string(&canonical_path)
        .map_err(|error| format!("cannot read canonical V3 module: {error}"))?;
    for (kind, name) in [
        ("struct", "ResidentArchiveKnnCalibrationReceiptV2"),
        ("struct", "ResidentSearchGenerationChainV3"),
        ("struct", "ResidentSearchRankEnqueuedV3"),
        ("struct", "ResidentSearchArchiveStagedV3"),
        ("struct", "ResidentSearchTerminalPendingV3"),
        ("struct", "ResidentSearchTerminalReceiptV3"),
        ("enum", "ResidentSearchTryCompleteV3"),
        ("struct", "ResidentSearchTransitionErrorV3"),
        ("struct", "ResidentSearchRejectedAuthorityV3"),
    ] {
        let needle = format!("pub {kind} {name}");
        let count = collect_regular_files(&gpu_root)?
            .into_iter()
            .filter(|path| path.extension() == Some(OsStr::new("rs")))
            .map(|path| {
                fs::read_to_string(path)
                    .unwrap_or_default()
                    .matches(&needle)
                    .count()
            })
            .sum::<usize>();
        if count != 1 || !canonical_source.contains(&needle) {
            return Err(format!(
                "canonical definition uniqueness drift for {needle}: {count}"
            ));
        }
    }
    for exact in [
        "inner: core::convert::Infallible,",
        "error: ResidentSearchTransitionErrorV3,",
        "authority: A,",
        "pub(crate) fn from_ranked_v3(",
    ] {
        if canonical_source.matches(exact).count() != 1
            && exact != "inner: core::convert::Infallible,"
        {
            return Err(format!("canonical private shape drift for {exact:?}"));
        }
    }
    if canonical_source
        .matches("inner: core::convert::Infallible,")
        .count()
        != 7
        || canonical_source.matches("pub fn ").count() != 6
        || [
            "pub fn new(",
            "pub fn inner(",
            "pub fn into_inner(",
            "pub fn error(",
            "pub fn authority(",
            "pub fn from_ranked_v3(",
        ]
        .iter()
        .any(|token| canonical_source.contains(token))
    {
        return Err("canonical V3 field/method visibility drift".to_owned());
    }
    let gpu_lib = fs::read_to_string(gpu_root.join("lib.rs"))
        .map_err(|error| format!("cannot read GPU lib.rs: {error}"))?;
    let gpu_declaration = "#[cfg(any(feature = \"cuda\", feature = \"resident-search-slice2-compile-contract\"))]\npub mod resident_search_slice2_v3;";
    if gpu_lib.matches(gpu_declaration).count() != 1 {
        return Err("GPU V3 module declaration drift".to_owned());
    }
    let search_lib = normalized_lf(
        &fs::read_to_string(repo.join("crates/neoethos-search/src/lib.rs"))
            .map_err(|error| format!("cannot read Search lib.rs: {error}"))?,
    );
    let search_gate = concat!(
        "#[cfg(any(\n",
        "    feature = \"gpu-b-native\",\n",
        "    feature = \"resident-search-slice2-compile-contract\"\n",
        "))]\n",
        "pub mod resident_search_slice2_v3 {",
    );
    if search_lib.matches(search_gate).count() != 1 {
        return Err("Search V3 feature gate drift".to_owned());
    }
    let facade_start = search_lib
        .find("pub mod resident_search_slice2_v3 {")
        .ok_or_else(|| "Search V3 facade is absent".to_owned())?;
    if search_lib[facade_start + 1..].contains("pub mod resident_search_slice2_v3 {") {
        return Err("duplicate Search V3 facade".to_owned());
    }
    let facade_end = search_lib[facade_start..]
        .find("\n}\n")
        .map(|index| facade_start + index + 3)
        .ok_or_else(|| "Search V3 facade body is unterminated".to_owned())?;
    let facade = &search_lib[facade_start..facade_end];
    let expected_facade = concat!(
        "pub mod resident_search_slice2_v3 {\n",
        "    pub use crate::gpu_resident_current_config_plan_v1::FullResidentDiscoveryDeadlineReceiptV1;\n",
        "    pub use neoethos_gpu_cuda::resident_search_slice2_v3::{\n",
        "        ResidentArchiveKnnCalibrationReceiptV2, ResidentSearchArchiveStagedV3,\n",
        "        ResidentSearchGenerationChainV3, ResidentSearchRankEnqueuedV3,\n",
        "        ResidentSearchRejectedAuthorityV3, ResidentSearchTerminalPendingV3,\n",
        "        ResidentSearchTerminalReceiptV3, ResidentSearchTransitionErrorV3,\n",
        "        ResidentSearchTryCompleteV3,\n",
        "    };\n",
        "}\n",
    );
    if facade != expected_facade {
        return Err("Search facade is not declaration-only exact re-export shape".to_owned());
    }
    Ok(())
}

fn expected_receipt_rows(repo: &Path, fixture: &Path) -> Result<Vec<String>, String> {
    let mut paths = vec![fixture.join("Cargo.toml"), fixture.join("Cargo.lock")];
    for case in CASES {
        paths.push(fixture.join(case.source));
        if let Some(stderr) = case.stderr {
            paths.push(fixture.join(stderr));
        }
    }
    paths.extend([
        fixture.join("api-surface-v3.txt"),
        repo.join("crates/neoethos-gpu-cuda/src/resident_search_slice2_v3.rs"),
        repo.join("crates/neoethos-search/src/lib.rs"),
    ]);
    let mut rows = Vec::new();
    for path in paths {
        let relative = path
            .strip_prefix(repo)
            .map_err(|_| format!("receipt path escapes repo: {}", path.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        let bytes = fs::read(&path)
            .map_err(|error| format!("cannot read receipt input {relative}: {error}"))?;
        rows.push(format!("{}  {relative}", sha256(&bytes)));
    }
    rows.sort();
    Ok(rows)
}

fn verify_hash_receipt(repo: &Path, fixture: &Path) -> Result<(), String> {
    let path = fixture.join("r7-v9-receipt.sha256");
    let bytes = fs::read(&path).map_err(|error| format!("cannot read R7 receipt: {error}"))?;
    let expected = format!("{}\n", expected_receipt_rows(repo, fixture)?.join("\n"));
    if bytes != expected.as_bytes() {
        return Err(
            "r7-v9-receipt.sha256 is missing, extra, duplicate, malformed, or stale".to_owned(),
        );
    }
    Ok(())
}

fn verify_plain_directory(path: &Path, label: &str) -> Result<PathBuf, String> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("cannot inspect {label}: {error}"))?;
    if !metadata.is_dir() {
        return Err(format!("{label} is not a directory: {}", path.display()));
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;

        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0400;
        if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            return Err(format!(
                "{label} is a symlink or junction: {}",
                path.display()
            ));
        }
    }
    #[cfg(not(windows))]
    if metadata.file_type().is_symlink() {
        return Err(format!("{label} is a symlink: {}", path.display()));
    }
    canonical(path, label)
}

fn fresh_directory(path: &Path, label: &str) -> Result<PathBuf, String> {
    if path.exists() {
        return Err(format!(
            "{label} must be absent before the run: {}",
            path.display()
        ));
    }
    fs::create_dir(path).map_err(|error| format!("cannot create {label}: {error}"))?;
    verify_plain_directory(path, label)?;
    Ok(path.to_path_buf())
}

fn build_context() -> Result<RunContext, String> {
    let search_manifest = canonical(Path::new(env!("CARGO_MANIFEST_DIR")), "Search manifest")?;
    let repo = canonical(
        search_manifest
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| "Search manifest is not under <repo>/crates".to_owned())?,
        "repository root",
    )?;
    let fixture = canonical(
        &search_manifest
            .join("tests")
            .join("ui")
            .join("resident_search_slice2"),
        "fixture root",
    )?;
    if !path_is_within(&fixture, &repo) {
        return Err("fixture root escapes the repository".to_owned());
    }
    verify_authority_and_protected(&repo)?;
    verify_vendor_closure(&repo)?;
    verify_source_shape(&repo)?;
    verify_hash_receipt(&repo, &fixture)?;
    verify_red_only_dead_code_allows(&repo)?;
    verify_fixture_tree_does_not_escape(&fixture, &repo)?;
    let manifest = canonical(&fixture.join("Cargo.toml"), "fixture manifest")?;
    validate_case_ledger(&fixture)?;
    verify_evidence_names()?;
    verify_cargo_config_inventory(&repo)?;

    let cargo = canonical_cargo()?;
    let rustc = canonical_rustc()?;
    let outer_target = PathBuf::from(
        env::var_os("CARGO_TARGET_DIR")
            .ok_or_else(|| "controlled CARGO_TARGET_DIR is absent".to_owned())?,
    );
    let resolved_outer_target = canonical(&outer_target, "outer target")?;
    if !path_is_within(&resolved_outer_target, &repo) {
        return Err("outer target is not under the exact repository".to_owned());
    }
    let outer_target = ordinary_absolute_path(&resolved_outer_target);
    if outer_target.file_name() != Some(OsStr::new("o")) {
        return Err(format!(
            "outer target leaf is not exact `o`: {}",
            outer_target.display()
        ));
    }
    verify_plain_directory(&outer_target, "outer target")?;
    let run_root = outer_target
        .parent()
        .ok_or_else(|| "outer target has no run root".to_owned())?;
    let run_id = run_root
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| "R7 run ID is not UTF-8".to_owned())?;
    if run_id.len() != 16
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("R7 run ID is not 16 lowercase hex: {run_id}"));
    }
    let r7_root = run_root
        .parent()
        .ok_or_else(|| "R7 run root has no parent".to_owned())?;
    if r7_root.file_name() != Some(OsStr::new("r7")) {
        return Err(format!(
            "R7 run parent is not exact `r7`: {}",
            r7_root.display()
        ));
    }
    let target_root = r7_root
        .parent()
        .ok_or_else(|| "R7 root has no target parent".to_owned())?;
    if !same_path(
        target_root,
        &canonical(&repo.join("target"), "repo target root")?,
    ) {
        return Err("short R7 root is not under $REPO/target".to_owned());
    }
    verify_plain_directory(r7_root, "short R7 root")?;
    verify_plain_directory(run_root, "short R7 run root")?;
    let run_root = run_root.to_path_buf();

    let target_on_path = run_root.join("on");
    let target_off_path = run_root.join("off");
    let target_doc_path = run_root.join("doc");
    let target_on = fresh_directory(&target_on_path, "feature-on UI target")?;
    let target_off = fresh_directory(&target_off_path, "feature-off UI target")?;
    let target_doc = fresh_directory(&target_doc_path, "rustdoc API target")?;
    let evidence = fresh_directory(&run_root.join("e"), "R7 v9 evidence directory")?;

    let host_environment = HOST_ENV_ALLOWLIST
        .iter()
        .filter_map(|name| env::var_os(name).map(|value| (OsString::from(name), value)))
        .collect();

    Ok(RunContext {
        repo,
        fixture,
        manifest,
        cargo,
        rustc,
        host_environment,
        outer_target,
        target_on,
        target_off,
        target_doc,
        evidence,
    })
}

fn directory_file_bytes(root: &Path) -> Result<u64, String> {
    let mut pending = vec![root.to_path_buf()];
    let mut total = 0_u64;
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .map_err(|error| format!("cannot enumerate outer target for disk gate: {error}"))?
        {
            let entry =
                entry.map_err(|error| format!("cannot read outer target entry: {error}"))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("cannot inspect outer target entry: {error}"))?;
            if metadata.file_type().is_symlink() {
                return Err(format!(
                    "outer target contains a symlink at disk gate: {}",
                    path.display()
                ));
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() {
                total = total
                    .checked_add(metadata.len())
                    .ok_or_else(|| "outer target byte count overflow".to_owned())?;
            }
        }
    }
    Ok(total)
}

fn disk_mount_contains(path: &Path, mount: &Path) -> bool {
    let path = normalized_path_text(path).to_ascii_lowercase();
    let mount = normalized_path_text(mount).to_ascii_lowercase();
    if mount.ends_with('/') {
        path.starts_with(&mount)
    } else {
        path == mount || path.starts_with(&(mount + "/"))
    }
}

fn verify_child_disk_budget(context: &RunContext, phase: &str) -> Result<(), String> {
    const SAFETY_HEADROOM_BYTES: u64 = 1_073_741_824;

    let outer_bytes = directory_file_bytes(&context.outer_target)?;
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let disk = disks
        .list()
        .iter()
        .filter(|disk| disk_mount_contains(&context.outer_target, disk.mount_point()))
        .max_by_key(|disk| normalized_path_text(disk.mount_point()).len())
        .ok_or_else(|| "cannot bind outer target to a refreshed disk".to_owned())?;
    let free_bytes = disk.available_space();
    let required_free = outer_bytes
        .checked_add(SAFETY_HEADROOM_BYTES)
        .ok_or_else(|| "disk-gate required byte count overflow".to_owned())?;
    let report = format!(
        "outer_target_bytes={outer_bytes}\ncurrent_free_bytes={free_bytes}\nrequired_free_bytes={required_free}\nsafety_headroom_bytes={SAFETY_HEADROOM_BYTES}\n"
    );
    fs::write(
        context.evidence.join(format!("disk-gate-{phase}.txt")),
        &report,
    )
    .map_err(|error| format!("cannot write {phase} disk-gate evidence: {error}"))?;
    if free_bytes < required_free {
        return Err(format!(
            "disk gate refused child Cargo: current free {free_bytes} < outer-size-plus-1GiB {required_free}; outer target is {outer_bytes} bytes"
        ));
    }
    Ok(())
}

fn clean_child_target(context: &RunContext, target: &Path, name: &str) -> Result<(), String> {
    let leaf = target.file_name().and_then(OsStr::to_str);
    if !matches!(leaf, Some("doc" | "on" | "off")) {
        return Err(format!(
            "refusing non-canonical child target cleanup: {}",
            target.display()
        ));
    }
    let repository_target = verify_plain_directory(
        &context.repo.join("target"),
        "repository target cleanup root",
    )?;
    let resolved_run_root = verify_plain_directory(&context.run_root, "R7 cleanup run root")?;
    let resolved_target = verify_plain_directory(target, "R7 cleanup child target")?;
    let expected_target = context
        .run_root
        .join(leaf.expect("validated cleanup leaf is present"));
    if !path_is_within(&resolved_run_root, &repository_target)
        || !path_is_within(&resolved_target, &resolved_run_root)
        || !same_path(target, &expected_target)
        || !same_path(&resolved_target, &expected_target)
    {
        return Err(format!(
            "refusing resolved or redirected child target cleanup: {} -> {}",
            target.display(),
            resolved_target.display()
        ));
    }
    let capture = capture_cargo_process(
        context,
        target,
        name,
        &[
            OsString::from("+nightly-2026-04-07"),
            OsString::from("clean"),
            OsString::from("--target-dir"),
            target.as_os_str().to_owned(),
        ],
    )?;
    let stdout = capture.stdout_text(name)?;
    let stderr = capture.stderr_text(name)?;
    let mut events = Vec::new();
    let counts = classify_process_stderr(stderr, &mut events);
    if !capture.status.success()
        || !stdout.is_empty()
        || counts.warning != 0
        || counts.error != 0
        || counts.other != 0
    {
        return Err(format!(
            "{name} cleanup mismatch: exit={:?}, stdout-bytes={}, stderr={counts:?}",
            capture.status.code(),
            stdout.len()
        ));
    }
    finish_capture_events(context, name, &events)?;
    if target.exists() {
        return Err(format!(
            "{name} cleanup left target present: {}",
            target.display()
        ));
    }
    Ok(())
}

fn configure_child_environment(
    command: &mut Command,
    context: &RunContext,
    target: &Path,
) -> Result<(), String> {
    command.env_clear();
    for (name, value) in &context.host_environment {
        command.env(name, value);
    }
    command.env("CARGO", &context.cargo);
    command.env("CARGO_TARGET_DIR", target);
    command.env("CARGO_INCREMENTAL", "0");
    command.env("RUSTFLAGS", "-Dwarnings");
    command.env("CARGO_NET_OFFLINE", "true");
    command.env("CARGO_TERM_COLOR", "never");
    command.env("RUST_BACKTRACE", "0");

    let sentinels = [
        ("CUDA_PATH", target.join("missing-cuda-path")),
        ("CUDA_HOME", target.join("missing-cuda-home")),
        ("CUDACXX", target.join("missing-nvcc.exe")),
        ("CUDAOBJDUMP", target.join("missing-cuobjdump.exe")),
    ];
    for (name, path) in sentinels {
        if path.exists() {
            return Err(format!(
                "CUDA sentinel unexpectedly exists: {}",
                path.display()
            ));
        }
        command.env(name, path);
    }
    Ok(())
}

fn valid_evidence_name(name: &str) -> bool {
    !name.is_empty()
        && name.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
}

fn verify_evidence_names() -> Result<(), String> {
    for name in CASES
        .iter()
        .map(|case| case.bin)
        .chain(RUSTDOC_SPECS.iter().map(|spec| spec.evidence_name))
        .chain([
            "cargo-id",
            "rustc-id",
            "metadata-feature-on",
            "metadata-feature-off",
            "metadata-outer",
            "clean-doc",
            "clean-on",
            "clean-off",
        ])
    {
        if !valid_evidence_name(name) {
            return Err(format!("authorized evidence name is rejected: {name}"));
        }
    }
    for malicious in ["", ".", "..", "a/b", "a\\b", "a.b", "a:b", "A", "_/"] {
        if valid_evidence_name(malicious) {
            return Err(format!(
                "malicious evidence name is unexpectedly accepted: {malicious:?}"
            ));
        }
    }
    Ok(())
}

fn write_capture_hashes(
    evidence: &Path,
    name: &str,
    files: &[(&str, &[u8])],
) -> Result<(), String> {
    let mut rows = Vec::with_capacity(files.len());
    for (file_name, bytes) in files {
        rows.push(format!("{} {} {file_name}", sha256(bytes), bytes.len()));
    }
    rows.sort();
    fs::write(
        evidence.join(format!("{name}.capture.sha256")),
        format!("{}\n", rows.join("\n")),
    )
    .map_err(|error| format!("cannot write {name} capture hash ledger: {error}"))
}

fn capture_process(
    context: &RunContext,
    target: &Path,
    name: &str,
    executable: &Path,
    arguments: &[OsString],
) -> Result<CapturedProcess, String> {
    if !valid_evidence_name(name) {
        return Err(format!("invalid evidence name: {name:?}"));
    }
    let mut command = Command::new(executable);
    command.current_dir(&context.repo);
    command.args(arguments);
    configure_child_environment(&mut command, context, target)?;
    let output = command
        .output()
        .map_err(|error| format!("cannot start exact captured process {name}: {error}"))?;

    let stdout_name = format!("{name}.stdout.raw");
    let stderr_name = format!("{name}.process-stderr.raw");
    let status_name = format!("{name}.status.txt");
    let status_bytes = format!(
        "success={}\ncode={:?}\n",
        output.status.success(),
        output.status.code()
    )
    .into_bytes();
    fs::write(context.evidence.join(&stdout_name), &output.stdout)
        .map_err(|error| format!("cannot write {name} stdout evidence: {error}"))?;
    fs::write(context.evidence.join(&stderr_name), &output.stderr)
        .map_err(|error| format!("cannot write {name} process-stderr evidence: {error}"))?;
    fs::write(context.evidence.join(&status_name), &status_bytes)
        .map_err(|error| format!("cannot write {name} status evidence: {error}"))?;
    write_capture_hashes(
        &context.evidence,
        name,
        &[
            (&stdout_name, &output.stdout),
            (&stderr_name, &output.stderr),
            (&status_name, &status_bytes),
        ],
    )?;

    std::str::from_utf8(&output.stdout)
        .map_err(|error| format!("{name} stdout is not valid UTF-8: {error}"))?;
    std::str::from_utf8(&output.stderr)
        .map_err(|error| format!("{name} process stderr is not valid UTF-8: {error}"))?;
    io::stdout()
        .write_all(&output.stdout)
        .map_err(|error| format!("cannot re-emit complete {name} stdout: {error}"))?;
    io::stderr()
        .write_all(&output.stderr)
        .map_err(|error| format!("cannot re-emit complete {name} process stderr: {error}"))?;

    let stdout_text =
        std::str::from_utf8(&output.stdout).expect("stdout UTF-8 was validated immediately above");
    let stderr_text = std::str::from_utf8(&output.stderr)
        .expect("process-stderr UTF-8 was validated immediately above");
    let mut fallback_events = vec![format!(
        "capture|INFO|exit={:?}|success={}|stdout-bytes={}|stdout-sha256={}|process-stderr-bytes={}|process-stderr-sha256={}",
        output.status.code(),
        output.status.success(),
        output.stdout.len(),
        sha256(&output.stdout),
        output.stderr.len(),
        sha256(&output.stderr)
    )];
    for (index, line) in stdout_text.lines().enumerate() {
        let severity = match serde_json::from_str::<Value>(line) {
            Ok(message)
                if message.get("reason").and_then(Value::as_str) == Some("compiler-message") =>
            {
                match message
                    .get("message")
                    .and_then(|diagnostic| diagnostic.get("level"))
                    .and_then(Value::as_str)
                {
                    Some("warning") => "WARNING",
                    Some("error") => "ERROR",
                    _ => "INFO",
                }
            }
            Ok(_) => "INFO",
            Err(_) => "OTHER",
        };
        fallback_events.push(format!(
            "raw-stdout|{severity}|line={}|text={}",
            index + 1,
            serde_json::to_string(line).expect("serializing a string cannot fail")
        ));
    }
    let _ = classify_process_stderr(stderr_text, &mut fallback_events);
    finish_capture_events(context, name, &fallback_events)?;

    Ok(CapturedProcess {
        status: output.status,
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

fn capture_cargo_process(
    context: &RunContext,
    target: &Path,
    name: &str,
    arguments: &[OsString],
) -> Result<CapturedProcess, String> {
    capture_process(context, target, name, &context.cargo, arguments)
}

fn normalized_lf(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn verify_tool_identities(context: &RunContext) -> Result<(), String> {
    let cargo = capture_cargo_process(
        context,
        &context.outer_target,
        "cargo-id",
        &[OsString::from("+nightly-2026-04-07"), OsString::from("-Vv")],
    )?;
    let cargo_stdout = normalized_lf(cargo.stdout_text("cargo-id")?);
    let cargo_stderr = cargo.stderr_text("cargo-id")?;
    if !cargo.status.success()
        || !cargo_stderr.is_empty()
        || cargo_stdout.as_bytes() != CARGO_IDENTITY.as_bytes()
        || cargo_stdout.len() != 337
        || sha256(cargo_stdout.as_bytes()) != CARGO_IDENTITY_SHA256
    {
        return Err(format!(
            "pinned Cargo identity mismatch: exit={:?}, stderr-bytes={}, stdout-bytes={}, stdout-sha256={}",
            cargo.status.code(),
            cargo_stderr.len(),
            cargo_stdout.len(),
            sha256(cargo_stdout.as_bytes())
        ));
    }
    finish_capture_events(
        context,
        "cargo-id",
        &[format!(
            "identity|INFO|tool=cargo|bytes=337|sha256={CARGO_IDENTITY_SHA256}"
        )],
    )?;

    let rustc = capture_process(
        context,
        &context.outer_target,
        "rustc-id",
        &context.rustc,
        &[OsString::from("+nightly-2026-04-07"), OsString::from("-vV")],
    )?;
    let rustc_stdout = normalized_lf(rustc.stdout_text("rustc-id")?);
    let rustc_stderr = rustc.stderr_text("rustc-id")?;
    if !rustc.status.success()
        || !rustc_stderr.is_empty()
        || rustc_stdout.as_bytes() != RUSTC_IDENTITY.as_bytes()
        || rustc_stdout.len() != 210
        || sha256(rustc_stdout.as_bytes()) != RUSTC_IDENTITY_SHA256
    {
        return Err(format!(
            "pinned rustc identity mismatch: exit={:?}, stderr-bytes={}, stdout-bytes={}, stdout-sha256={}",
            rustc.status.code(),
            rustc_stderr.len(),
            rustc_stdout.len(),
            sha256(rustc_stdout.as_bytes())
        ));
    }
    finish_capture_events(
        context,
        "rustc-id",
        &[format!(
            "identity|INFO|tool=rustc|bytes={}|sha256={}",
            RUSTC_IDENTITY.len(),
            RUSTC_IDENTITY_SHA256
        )],
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MetadataMode {
    FeatureOn,
    FeatureOff,
    Outer,
}

fn metadata_arguments(context: &RunContext, mode: MetadataMode) -> Vec<OsString> {
    let manifest = if mode == MetadataMode::Outer {
        context.repo.join("crates/neoethos-search/Cargo.toml")
    } else {
        context.manifest.clone()
    };
    let mut arguments = vec![
        OsString::from("+nightly-2026-04-07"),
        OsString::from("metadata"),
        OsString::from("--manifest-path"),
        manifest.into_os_string(),
        OsString::from("--locked"),
        OsString::from("--offline"),
        OsString::from("--format-version"),
        OsString::from("1"),
    ];
    if mode == MetadataMode::Outer {
        arguments.push(OsString::from("--no-deps"));
    } else {
        arguments.push(OsString::from("--no-default-features"));
        if mode == MetadataMode::FeatureOn {
            arguments.extend([
                OsString::from("--features"),
                OsString::from("resident-search-slice2-compile-contract"),
            ]);
        }
    }
    arguments
}

fn metadata_package<'a>(
    document: &'a Value,
    name: &str,
    manifest: &Path,
) -> Result<&'a Value, String> {
    let expected = canonical(manifest, "metadata package manifest")?;
    let matches = document
        .get("packages")
        .and_then(Value::as_array)
        .ok_or_else(|| "metadata has no packages array".to_owned())?
        .iter()
        .filter(|package| package.get("name").and_then(Value::as_str) == Some(name))
        .filter(|package| {
            package
                .get("manifest_path")
                .and_then(Value::as_str)
                .and_then(|path| canonical(Path::new(path), "metadata observed manifest").ok())
                .is_some_and(|path| same_path(&path, &expected))
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!(
            "metadata package binding drift for {name}: {}",
            matches.len()
        ));
    }
    Ok(matches[0])
}

fn metadata_node<'a>(document: &'a Value, package: &Value) -> Result<&'a Value, String> {
    let id = package
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| "metadata package has no id".to_owned())?;
    let matches = document
        .get("resolve")
        .and_then(|resolve| resolve.get("nodes"))
        .and_then(Value::as_array)
        .ok_or_else(|| "metadata has no resolve nodes".to_owned())?
        .iter()
        .filter(|node| node.get("id").and_then(Value::as_str) == Some(id))
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!(
            "metadata resolve-node drift for {id}: {}",
            matches.len()
        ));
    }
    Ok(matches[0])
}

fn metadata_features(value: &Value) -> Result<Vec<String>, String> {
    let mut features = value_string_array(value, "features")?;
    features.sort();
    Ok(features)
}

fn verify_fixture_metadata(
    context: &RunContext,
    document: &Value,
    mode: MetadataMode,
) -> Result<(), String> {
    let fixture = metadata_package(
        document,
        "neoethos-resident-search-slice2-ui",
        &context.manifest,
    )?;
    let targets = fixture
        .get("targets")
        .and_then(Value::as_array)
        .ok_or_else(|| "fixture metadata has no targets".to_owned())?;
    if targets.len() != CASES.len() {
        return Err(format!("fixture target count drift: {}", targets.len()));
    }
    let mut observed_targets = Vec::new();
    for target in targets {
        let name = target
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| "fixture target has no name".to_owned())?;
        let kind = value_string_array(target, "kind")?;
        let crate_types = value_string_array(target, "crate_types")?;
        let source = target
            .get("src_path")
            .and_then(Value::as_str)
            .ok_or_else(|| "fixture target has no src_path".to_owned())?;
        if kind != ["bin"] || crate_types != ["bin"] {
            return Err(format!("fixture target kind drift for {name}"));
        }
        let case = CASES
            .iter()
            .find(|case| case.bin == name)
            .ok_or_else(|| format!("unexpected fixture target {name}"))?;
        if !same_path(
            &canonical(Path::new(source), "fixture metadata source")?,
            &canonical(&context.fixture.join(case.source), "fixture ledger source")?,
        ) {
            return Err(format!("fixture target source drift for {name}"));
        }
        observed_targets.push(name);
    }
    observed_targets.sort();
    let mut expected_targets = CASES.iter().map(|case| case.bin).collect::<Vec<_>>();
    expected_targets.sort();
    if observed_targets != expected_targets {
        return Err("fixture target inventory drift".to_owned());
    }
    let feature_map = fixture
        .get("features")
        .and_then(Value::as_object)
        .ok_or_else(|| "fixture metadata has no feature map".to_owned())?;
    if feature_map
        .get("default")
        .and_then(Value::as_array)
        .is_none_or(|v| !v.is_empty())
        || feature_map
            .get("resident-search-slice2-compile-contract")
            .and_then(Value::as_array)
            .is_none_or(|values| {
                values.len() != 1
                    || values[0].as_str()
                        != Some("neoethos-search/resident-search-slice2-compile-contract")
            })
    {
        return Err("fixture feature forwarding map drift".to_owned());
    }
    let fixture_dependencies = fixture
        .get("dependencies")
        .and_then(Value::as_array)
        .ok_or_else(|| "fixture metadata has no dependencies".to_owned())?;
    if fixture_dependencies.len() != 2 {
        return Err(format!(
            "fixture direct dependency count drift: {}",
            fixture_dependencies.len()
        ));
    }
    for (name, manifest) in [
        (
            "neoethos-search",
            context.repo.join("crates/neoethos-search/Cargo.toml"),
        ),
        (
            "neoethos-gpu-cuda",
            context.repo.join("crates/neoethos-gpu-cuda/Cargo.toml"),
        ),
    ] {
        let matches = fixture_dependencies
            .iter()
            .filter(|dependency| dependency.get("name").and_then(Value::as_str) == Some(name))
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(format!("fixture dependency multiplicity drift for {name}"));
        }
        let dependency = matches[0];
        let path = dependency
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("fixture dependency has no path: {name}"))?;
        if dependency.get("optional").and_then(Value::as_bool) != Some(false)
            || dependency
                .get("uses_default_features")
                .and_then(Value::as_bool)
                != Some(false)
            || dependency
                .get("features")
                .and_then(Value::as_array)
                .is_none_or(|values| !values.is_empty())
            || !same_path(
                &canonical(Path::new(path), "fixture dependency path")?,
                &canonical(
                    manifest.parent().expect("manifest path has a parent"),
                    "fixture dependency expected path",
                )?,
            )
        {
            return Err("fixture direct dependency feature/default drift".to_owned());
        }
    }

    let search = metadata_package(
        document,
        "neoethos-search",
        &context.repo.join("crates/neoethos-search/Cargo.toml"),
    )?;
    let gpu = metadata_package(
        document,
        "neoethos-gpu-cuda",
        &context.repo.join("crates/neoethos-gpu-cuda/Cargo.toml"),
    )?;
    let expected = if mode == MetadataMode::FeatureOn {
        vec!["resident-search-slice2-compile-contract".to_owned()]
    } else {
        Vec::new()
    };
    if metadata_features(metadata_node(document, search)?)? != expected
        || metadata_features(metadata_node(document, gpu)?)? != expected
    {
        return Err(format!("Search/GPU active feature drift in {mode:?}"));
    }
    let search_feature = search
        .get("features")
        .and_then(Value::as_object)
        .and_then(|features| features.get("resident-search-slice2-compile-contract"))
        .and_then(Value::as_array)
        .ok_or_else(|| "Search compile-contract feature is absent".to_owned())?;
    let mut forwarding = search_feature
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    forwarding.sort();
    if forwarding
        != [
            "dep:neoethos-gpu-cuda",
            "neoethos-gpu-cuda/resident-search-slice2-compile-contract",
        ]
        || gpu
            .get("features")
            .and_then(Value::as_object)
            .and_then(|features| features.get("resident-search-slice2-compile-contract"))
            .and_then(Value::as_array)
            .is_none_or(|values| !values.is_empty())
    {
        return Err("Search-to-GPU forwarding edge drift".to_owned());
    }
    let search_gpu_dependencies = search
        .get("dependencies")
        .and_then(Value::as_array)
        .ok_or_else(|| "Search metadata has no dependencies".to_owned())?
        .iter()
        .filter(|dependency| {
            dependency.get("name").and_then(Value::as_str) == Some("neoethos-gpu-cuda")
        })
        .collect::<Vec<_>>();
    if search_gpu_dependencies.len() != 1 {
        return Err("Search-to-GPU dependency multiplicity drift".to_owned());
    }
    let search_gpu_dependency = search_gpu_dependencies[0];
    let search_gpu_path = search_gpu_dependency
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "Search-to-GPU dependency has no path".to_owned())?;
    if search_gpu_dependency
        .get("optional")
        .and_then(Value::as_bool)
        != Some(true)
        || search_gpu_dependency
            .get("uses_default_features")
            .and_then(Value::as_bool)
            != Some(false)
        || search_gpu_dependency
            .get("features")
            .and_then(Value::as_array)
            .is_none_or(|values| !values.is_empty())
        || !same_path(
            &canonical(Path::new(search_gpu_path), "Search-to-GPU dependency path")?,
            &canonical(
                &context.repo.join("crates/neoethos-gpu-cuda"),
                "Search-to-GPU expected path",
            )?,
        )
    {
        return Err("Search-to-GPU dependency metadata drift".to_owned());
    }

    let resolve_nodes = document
        .get("resolve")
        .and_then(|resolve| resolve.get("nodes"))
        .and_then(Value::as_array)
        .ok_or_else(|| "metadata resolve nodes are absent".to_owned())?;
    let packages = document
        .get("packages")
        .and_then(Value::as_array)
        .ok_or_else(|| "metadata packages are absent".to_owned())?;
    for node in resolve_nodes {
        let id = node.get("id").and_then(Value::as_str).unwrap_or_default();
        let package = packages
            .iter()
            .find(|package| package.get("id").and_then(Value::as_str) == Some(id))
            .ok_or_else(|| format!("resolve node has no package: {id}"))?;
        let name = package
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let features = metadata_features(node)?;
        if matches!(name, "cust" | "cust_raw" | "find_cuda_helper")
            || (name == "neoethos-gpu-cuda"
                && features
                    .iter()
                    .any(|feature| matches!(feature.as_str(), "cuda" | "cuda-device-fixtures")))
            || (name == "neoethos-data" && features.iter().any(|feature| feature == "gpu-cuda"))
        {
            return Err(format!(
                "forbidden metadata closure node: {name} {features:?}"
            ));
        }
    }
    let vector = metadata_package(
        document,
        "vector-ta",
        &context
            .repo
            .join("vendor/vector-ta-0.2.9-patched/Cargo.toml"),
    )?;
    if !vector.get("source").is_some_and(Value::is_null) {
        return Err("VectorTA metadata source is not null".to_owned());
    }
    let vector_features = metadata_features(metadata_node(document, vector)?)?;
    if !vector_features
        .iter()
        .any(|feature| feature == "nightly-avx")
        || vector_features
            .iter()
            .any(|feature| feature == "cuda-build-native")
    {
        return Err(format!("VectorTA feature drift: {vector_features:?}"));
    }
    Ok(())
}

fn verify_outer_metadata(context: &RunContext, document: &Value) -> Result<(), String> {
    let search = metadata_package(
        document,
        "neoethos-search",
        &context.repo.join("crates/neoethos-search/Cargo.toml"),
    )?;
    let expected_source = canonical(
        &context
            .repo
            .join("crates/neoethos-search/tests/resident_search_slice2_compile_contract.rs"),
        "outer test source",
    )?;
    let matches = search
        .get("targets")
        .and_then(Value::as_array)
        .ok_or_else(|| "Search metadata has no targets".to_owned())?
        .iter()
        .filter(|target| {
            target.get("name").and_then(Value::as_str)
                == Some("resident_search_slice2_compile_contract")
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(format!("outer test target count drift: {}", matches.len()));
    }
    let target = matches[0];
    if value_string_array(target, "kind")? != ["test"]
        || value_string_array(target, "crate_types")? != ["bin"]
        || value_string_array(target, "required-features")?
            != ["resident-search-slice2-compile-contract"]
        || !same_path(
            &canonical(
                Path::new(
                    target
                        .get("src_path")
                        .and_then(Value::as_str)
                        .ok_or_else(|| "outer test target has no src_path".to_owned())?,
                ),
                "outer metadata test source",
            )?,
            &expected_source,
        )
    {
        return Err("outer integration target metadata drift".to_owned());
    }
    Ok(())
}

fn run_metadata(context: &RunContext, mode: MetadataMode) -> Result<(), String> {
    let name = match mode {
        MetadataMode::FeatureOn => "metadata-feature-on",
        MetadataMode::FeatureOff => "metadata-feature-off",
        MetadataMode::Outer => "metadata-outer",
    };
    let capture = capture_cargo_process(
        context,
        &context.outer_target,
        name,
        &metadata_arguments(context, mode),
    )?;
    let stdout = capture.stdout_text(name)?;
    let stderr = capture.stderr_text(name)?;
    let lines = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    if !capture.status.success() || !stderr.is_empty() || lines.len() != 1 || has_ansi_sgr(stdout) {
        return Err(format!(
            "{name} capture mismatch: exit={:?}, stdout-lines={}, stderr-bytes={}",
            capture.status.code(),
            lines.len(),
            stderr.len()
        ));
    }
    let document: Value = serde_json::from_str(lines[0])
        .map_err(|error| format!("{name} stdout is not one metadata JSON document: {error}"))?;
    if mode == MetadataMode::Outer {
        verify_outer_metadata(context, &document)?;
    } else {
        verify_fixture_metadata(context, &document, mode)?;
    }
    finish_capture_events(
        context,
        name,
        &[format!(
            "metadata|INFO|mode={mode:?}|bytes={}|sha256={}",
            capture.stdout.len(),
            sha256(&capture.stdout)
        )],
    )
}

fn finish_capture_events(
    context: &RunContext,
    name: &str,
    events: &[String],
) -> Result<(), String> {
    let stdout_name = format!("{name}.stdout.raw");
    let stderr_name = format!("{name}.process-stderr.raw");
    let status_name = format!("{name}.status.txt");
    let events_name = format!("{name}.events.txt");
    let events_bytes = format!("{}\n", events.join("\n")).into_bytes();
    fs::write(context.evidence.join(&events_name), &events_bytes)
        .map_err(|error| format!("cannot write {name} classified events: {error}"))?;
    let stdout = fs::read(context.evidence.join(&stdout_name))
        .map_err(|error| format!("cannot reread {name} stdout evidence: {error}"))?;
    let stderr = fs::read(context.evidence.join(&stderr_name))
        .map_err(|error| format!("cannot reread {name} process-stderr evidence: {error}"))?;
    let status = fs::read(context.evidence.join(&status_name))
        .map_err(|error| format!("cannot reread {name} status evidence: {error}"))?;
    write_capture_hashes(
        &context.evidence,
        name,
        &[
            (&stdout_name, &stdout),
            (&stderr_name, &stderr),
            (&status_name, &status),
            (&events_name, &events_bytes),
        ],
    )
}

fn percent_decode_path(encoded: &str) -> Result<String, String> {
    let bytes = encoded.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if index + 2 >= bytes.len() {
                return Err("truncated percent escape in Cargo package ID".to_owned());
            }
            let high = (bytes[index + 1] as char)
                .to_digit(16)
                .ok_or_else(|| "invalid percent escape in Cargo package ID".to_owned())?;
            let low = (bytes[index + 2] as char)
                .to_digit(16)
                .ok_or_else(|| "invalid percent escape in Cargo package ID".to_owned())?;
            decoded.push(((high << 4) | low) as u8);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded)
        .map_err(|error| format!("Cargo package ID path is not valid UTF-8: {error}"))
}

fn fixture_package_identity(package_id: &str, fixture: &Path) -> Result<bool, String> {
    let source = if let Some((source, fragment)) = package_id.rsplit_once('#') {
        if fragment != "neoethos-resident-search-slice2-ui@0.0.0" && fragment != "0.0.0" {
            return Ok(false);
        }
        source
    } else if let Some(source) = package_id
        .strip_prefix("neoethos-resident-search-slice2-ui 0.0.0 (")
        .and_then(|value| value.strip_suffix(')'))
    {
        source
    } else {
        return Ok(false);
    };
    let encoded_path = source
        .strip_prefix("path+file:///")
        .ok_or_else(|| format!("fixture package has a non-path source: {package_id}"))?;
    let decoded_path = percent_decode_path(encoded_path)?;
    let package_root = canonical(
        &PathBuf::from(decoded_path.replace('/', "\\")),
        "fixture package ID root",
    )?;
    if !same_path(&package_root, fixture) {
        return Err(format!(
            "fixture package ID root drift: {}",
            normalized_path_text(&package_root)
        ));
    }
    Ok(true)
}

fn path_package_identity(
    package_id: &str,
    name: &str,
    version: &str,
    expected_root: &Path,
) -> Result<bool, String> {
    let source = if let Some((source, fragment)) = package_id.rsplit_once('#') {
        if fragment != format!("{name}@{version}") && fragment != version {
            return Ok(false);
        }
        source
    } else if let Some(source) = package_id
        .strip_prefix(&format!("{name} {version} ("))
        .and_then(|value| value.strip_suffix(')'))
    {
        source
    } else {
        return Ok(false);
    };
    let Some(encoded_path) = source.strip_prefix("path+file:///") else {
        return Ok(false);
    };
    let decoded_path = percent_decode_path(encoded_path)?;
    let package_root = canonical(
        &PathBuf::from(decoded_path.replace('/', "\\")),
        &format!("{name} package ID root"),
    )?;
    if !same_path(&package_root, expected_root) {
        return Ok(false);
    }
    Ok(true)
}

fn package_name(package_id: &str) -> Option<&str> {
    if let Some(fragment) = package_id.rsplit('#').next()
        && fragment != package_id
    {
        return fragment.split('@').next();
    }
    package_id.split_ascii_whitespace().next()
}

fn resolve_span_path(file_name: &str, context: &RunContext) -> Result<PathBuf, String> {
    let path = PathBuf::from(file_name);
    let candidates = if path.is_absolute() {
        vec![path]
    } else {
        vec![context.fixture.join(&path), context.repo.join(&path)]
    };
    for candidate in candidates {
        if candidate.exists() {
            return canonical(&candidate, "diagnostic span source");
        }
    }
    Err(format!(
        "cannot resolve diagnostic source path: {file_name}"
    ))
}

fn value_string_array(value: &Value, field: &str) -> Result<Vec<String>, String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("Cargo JSON target has no {field} array"))?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("Cargo JSON target {field} contains a non-string"))
        })
        .collect()
}

fn has_ansi_sgr(text: &str) -> bool {
    let bytes = text.as_bytes();
    let mut index = 0;
    while index + 2 < bytes.len() {
        if bytes[index] == 0x1b && bytes[index + 1] == b'[' {
            let mut cursor = index + 2;
            while cursor < bytes.len() && (bytes[cursor].is_ascii_digit() || bytes[cursor] == b';')
            {
                cursor += 1;
            }
            if cursor < bytes.len() && bytes[cursor] == b'm' {
                return true;
            }
        }
        index += 1;
    }
    false
}

fn normalize_rendered(
    rendered: &str,
    target: &Path,
    fixture: &Path,
    repo: &Path,
) -> Result<String, String> {
    if has_ansi_sgr(rendered) {
        return Err("ANSI SGR appeared despite --color never".to_owned());
    }
    let mut normalized = rendered.replace("\r\n", "\n").replace('\r', "\n");
    normalized = normalized.replace('\\', "/");
    let mut roots = vec![
        (normalized_path_text(target), "$TARGET"),
        (normalized_path_text(fixture), "$FIXTURE"),
        (normalized_path_text(repo), "$REPO"),
    ];
    roots.sort_by(|left, right| right.0.len().cmp(&left.0.len()));
    for (prefix, replacement) in roots {
        normalized = normalized.replace(&prefix, replacement);
    }
    Ok(normalized)
}

fn activation_from_json(message: &Value) -> Option<String> {
    let package_id = message.get("package_id").and_then(Value::as_str);
    if let Some(name) = package_id.and_then(package_name)
        && matches!(name, "cust" | "cust_raw" | "find_cuda_helper")
    {
        return Some(format!("active CUDA dependency package: {name}"));
    }
    if let Some(target_name) = message
        .get("target")
        .and_then(|target| target.get("name"))
        .and_then(Value::as_str)
        && matches!(
            target_name,
            "cust" | "cust_raw" | "find_cuda_helper" | "cuda-build-native"
        )
    {
        return Some(format!("forbidden Cargo target: {target_name}"));
    }
    if message.get("reason").and_then(Value::as_str) == Some("build-script-executed") {
        for field in ["linked_libs", "linked_paths"] {
            if let Some(values) = message.get(field).and_then(Value::as_array) {
                for value in values.iter().filter_map(Value::as_str) {
                    let lower = value.to_ascii_lowercase();
                    if lower.contains("cudart")
                        || lower == "cuda"
                        || lower.ends_with("/cuda")
                        || lower.contains("\\cuda")
                    {
                        return Some(format!("CUDA build-script {field}: {value}"));
                    }
                }
            }
        }
    }
    None
}

fn activation_from_process_stderr(stderr: &str) -> Option<&'static str> {
    let lower = stderr.to_ascii_lowercase();
    for (pattern, label) in [
        ("cuda-build-native", "cuda-build-native feature"),
        ("find_cuda_helper", "find_cuda_helper dependency"),
        ("cust_raw", "cust_raw dependency"),
        (" cuobjdump", "cuobjdump invocation"),
        (" nvcc", "nvcc invocation"),
        ("-lcudart", "cudart link"),
        ("-lcuda", "CUDA driver link"),
    ] {
        if lower.contains(pattern) {
            return Some(label);
        }
    }
    None
}

fn classify_process_stderr(stderr: &str, events: &mut Vec<String>) -> EventCounts {
    let mut counts = EventCounts::default();
    for (index, line) in stderr.lines().enumerate() {
        let trimmed = line.trim_start();
        let severity = if trimmed.starts_with("Compiling ")
            || trimmed.starts_with("Checking ")
            || trimmed.starts_with("Documenting ")
            || trimmed.starts_with("Finished ")
            || trimmed.starts_with("Locking ")
            || trimmed.starts_with("Running ")
            || trimmed.starts_with("Removed ")
        {
            "INFO"
        } else if trimmed.starts_with("warning:") {
            "WARNING"
        } else if trimmed.starts_with("error:")
            || trimmed.starts_with("fatal error:")
            || trimmed.contains("panicked at")
        {
            "ERROR"
        } else {
            "OTHER"
        };
        match severity {
            "INFO" => counts.info += 1,
            "WARNING" => counts.warning += 1,
            "ERROR" => counts.error += 1,
            _ => counts.other += 1,
        }
        let encoded = serde_json::to_string(line).expect("serializing a string cannot fail");
        events.push(format!(
            "process-stderr|{severity}|line={}|text={encoded}",
            index + 1
        ));
    }
    counts
}

#[derive(Clone, Copy)]
struct RustdocSpec {
    evidence_name: &'static str,
    package_name: &'static str,
    package_version: &'static str,
    package_directory: &'static str,
    manifest: &'static str,
    target_name: &'static str,
    source: &'static str,
}

const RUSTDOC_SPECS: &[RustdocSpec] = &[
    RustdocSpec {
        evidence_name: "gpu-rustdoc",
        package_name: "neoethos-gpu-cuda",
        package_version: "0.1.0",
        package_directory: "crates/neoethos-gpu-cuda",
        manifest: "crates/neoethos-gpu-cuda/Cargo.toml",
        target_name: "neoethos_gpu_cuda",
        source: "crates/neoethos-gpu-cuda/src/lib.rs",
    },
    RustdocSpec {
        evidence_name: "search-rustdoc",
        package_name: "neoethos-search",
        package_version: "0.5.6",
        package_directory: "crates/neoethos-search",
        manifest: "crates/neoethos-search/Cargo.toml",
        target_name: "neoethos_search",
        source: "crates/neoethos-search/src/lib.rs",
    },
];

fn parse_rustdoc_cargo_stream(
    context: &RunContext,
    spec: RustdocSpec,
    capture: &CapturedProcess,
) -> Result<Vec<String>, String> {
    let stdout = capture.stdout_text(spec.evidence_name)?;
    let stderr = capture.stderr_text(spec.evidence_name)?;
    if let Some(activation) = activation_from_process_stderr(stderr) {
        return Err(format!(
            "{} observed CUDA/native activation in process stderr: {activation}",
            spec.evidence_name
        ));
    }

    let package_root = canonical(
        &context.repo.join(spec.package_directory),
        "rustdoc package",
    )?;
    let expected_manifest = canonical(&context.repo.join(spec.manifest), "rustdoc manifest")?;
    let expected_source = canonical(&context.repo.join(spec.source), "rustdoc lib source")?;
    let mut events = Vec::new();
    let mut build_finished = None;
    let mut selected_lib_artifacts = 0;
    let mut compiler_messages = 0;
    let mut last_reason = None;

    for (line_index, line) in stdout.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = serde_json::from_str(line).map_err(|error| {
            format!(
                "{} emitted non-JSON Cargo stdout at line {}: {error}; text={line:?}",
                spec.evidence_name,
                line_index + 1
            )
        })?;
        if let Some(activation) = activation_from_json(&message) {
            return Err(format!(
                "{} observed CUDA/native activation in Cargo JSON: {activation}",
                spec.evidence_name
            ));
        }
        let reason = message
            .get("reason")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!(
                    "{} Cargo JSON line {} has no reason",
                    spec.evidence_name,
                    line_index + 1
                )
            })?;
        last_reason = Some(reason.to_owned());

        if let Some(features) = message.get("features").and_then(Value::as_array) {
            for feature in features.iter().filter_map(Value::as_str) {
                if matches!(
                    feature,
                    "cuda"
                        | "cuda-device-fixtures"
                        | "cuda-build-native"
                        | "gpu-b-native"
                        | "gpu-cuda"
                ) {
                    return Err(format!(
                        "{} Cargo JSON activated forbidden feature {feature}",
                        spec.evidence_name
                    ));
                }
            }
        }

        if reason == "build-finished" {
            if build_finished.is_some() {
                return Err(format!(
                    "{} emitted duplicate build-finished",
                    spec.evidence_name
                ));
            }
            build_finished = Some(message.get("success").and_then(Value::as_bool).ok_or_else(
                || format!("{} build-finished has no success bit", spec.evidence_name),
            )?);
            events.push(format!(
                "cargo-json|INFO|line={}|reason=build-finished|success={}",
                line_index + 1,
                build_finished.expect("just assigned")
            ));
            continue;
        }

        if reason == "compiler-message" {
            compiler_messages += 1;
            let level = message
                .get("message")
                .and_then(|value| value.get("level"))
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    format!(
                        "{} compiler-message has no diagnostic level",
                        spec.evidence_name
                    )
                })?;
            events.push(format!(
                "compiler-message|{}|line={}|message={}",
                if level == "warning" {
                    "WARNING"
                } else if level == "error" {
                    "ERROR"
                } else {
                    "INFO"
                },
                line_index + 1,
                serde_json::to_string(
                    message
                        .get("message")
                        .ok_or_else(|| "compiler-message payload disappeared".to_owned())?
                )
                .map_err(|error| format!("cannot serialize rustdoc diagnostic: {error}"))?
            ));
            continue;
        }

        if reason == "compiler-artifact" {
            let package_id = message
                .get("package_id")
                .and_then(Value::as_str)
                .ok_or_else(|| "rustdoc compiler-artifact has no package_id".to_owned())?;
            let is_selected = path_package_identity(
                package_id,
                spec.package_name,
                spec.package_version,
                &package_root,
            )?;
            let target = message
                .get("target")
                .ok_or_else(|| "rustdoc compiler-artifact has no target".to_owned())?;
            let target_name = target
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "rustdoc compiler-artifact target has no name".to_owned())?;
            let kinds = value_string_array(target, "kind")?;
            let crate_types = value_string_array(target, "crate_types")?;
            if is_selected
                && target_name == spec.target_name
                && kinds == ["lib"]
                && crate_types == ["lib"]
            {
                selected_lib_artifacts += 1;
                let manifest = message
                    .get("manifest_path")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "selected rustdoc artifact has no manifest_path".to_owned())?;
                let source = target
                    .get("src_path")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "selected rustdoc artifact has no src_path".to_owned())?;
                if !same_path(
                    &canonical(Path::new(manifest), "selected rustdoc manifest")?,
                    &expected_manifest,
                ) || !same_path(
                    &canonical(Path::new(source), "selected rustdoc source")?,
                    &expected_source,
                ) {
                    return Err(format!(
                        "{} selected manifest/source binding drift",
                        spec.evidence_name
                    ));
                }
                let features = value_string_array(&message, "features")?;
                if features != ["resident-search-slice2-compile-contract"] {
                    return Err(format!(
                        "{} selected feature drift: {features:?}",
                        spec.evidence_name
                    ));
                }
                let filenames = value_string_array(&message, "filenames")?;
                if !filenames.is_empty() {
                    return Err(format!(
                        "{} selected rustdoc artifact unexpectedly names Cargo artifacts: {filenames:?}",
                        spec.evidence_name
                    ));
                }
                events.push(format!(
                    "selected-rustdoc-artifact|INFO|line={}|package={}|manifest={}|target={}|kind=[\"lib\"]|crate-types=[\"lib\"]|source={}|features=[\"resident-search-slice2-compile-contract\"]|filenames=[]",
                    line_index + 1,
                    serde_json::to_string(package_id)
                        .expect("serializing a string cannot fail"),
                    serde_json::to_string(&normalized_path_text(&expected_manifest))
                        .expect("serializing a string cannot fail"),
                    spec.target_name,
                    serde_json::to_string(&normalized_path_text(&expected_source))
                        .expect("serializing a string cannot fail")
                ));
            }
        }

        events.push(format!(
            "cargo-json|INFO|line={}|reason={reason}",
            line_index + 1
        ));
    }

    let stderr_counts = classify_process_stderr(stderr, &mut events);
    if !capture.status.success()
        || build_finished != Some(true)
        || last_reason.as_deref() != Some("build-finished")
        || selected_lib_artifacts != 1
        || compiler_messages != 0
        || stderr_counts.warning != 0
        || stderr_counts.error != 0
        || stderr_counts.other != 0
    {
        return Err(format!(
            "{} observation mismatch: exit={:?}, build-finished={build_finished:?}, last-reason={last_reason:?}, selected-libs={selected_lib_artifacts}, compiler-messages={compiler_messages}, stderr-info={}, stderr-warning={}, stderr-error={}, stderr-other={}",
            spec.evidence_name,
            capture.status.code(),
            stderr_counts.info,
            stderr_counts.warning,
            stderr_counts.error,
            stderr_counts.other
        ));
    }
    Ok(events)
}

fn rustdoc_arguments(package: &str) -> Vec<OsString> {
    [
        "+nightly-2026-04-07",
        "rustdoc",
        "--locked",
        "--offline",
        "-j",
        "7",
        "-p",
        package,
        "--lib",
        "--no-default-features",
        "--features",
        "resident-search-slice2-compile-contract",
        "--message-format=json",
        "--color",
        "never",
        "--",
        "-Dwarnings",
        "-Z",
        "unstable-options",
        "--output-format",
        "json",
    ]
    .into_iter()
    .map(OsString::from)
    .collect()
}

fn rustdoc_id_key(id: &Value, label: &str) -> Result<String, String> {
    id.as_u64()
        .map(|value| value.to_string())
        .ok_or_else(|| format!("{label} is not a numeric rustdoc ID"))
}

fn rustdoc_index(document: &Value) -> Result<&serde_json::Map<String, Value>, String> {
    document
        .get("index")
        .and_then(Value::as_object)
        .ok_or_else(|| "rustdoc JSON has no index object".to_owned())
}

fn rustdoc_item<'a>(document: &'a Value, id: &Value, label: &str) -> Result<&'a Value, String> {
    let key = rustdoc_id_key(id, label)?;
    let item = rustdoc_index(document)?
        .get(&key)
        .ok_or_else(|| format!("{label} ID {key} is absent from rustdoc index"))?;
    let embedded = item
        .get("id")
        .ok_or_else(|| format!("{label} item has no embedded ID"))?;
    if rustdoc_id_key(embedded, label)? != key {
        return Err(format!("{label} rustdoc index/embedded ID mismatch"));
    }
    Ok(item)
}

fn rustdoc_item_inner<'a>(item: &'a Value, kind: &str, label: &str) -> Result<&'a Value, String> {
    item.get("inner")
        .and_then(|value| value.get(kind))
        .ok_or_else(|| format!("{label} is not rustdoc item kind {kind}"))
}

fn rustdoc_canonical_segments(
    document: &Value,
    id: &Value,
    label: &str,
) -> Result<Vec<String>, String> {
    let key = rustdoc_id_key(id, label)?;
    document
        .get("paths")
        .and_then(Value::as_object)
        .and_then(|paths| paths.get(&key))
        .and_then(|summary| summary.get("path"))
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} ID {key} has no canonical rustdoc path"))?
        .iter()
        .map(|segment| {
            segment
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{label} canonical path contains a non-string"))
        })
        .collect()
}

fn rustdoc_item_name<'a>(item: &'a Value, label: &str) -> Result<&'a str, String> {
    item.get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} has no name"))
}

fn require_rustdoc_public(item: &Value, label: &str) -> Result<(), String> {
    if item.get("visibility").and_then(Value::as_str) != Some("public") {
        return Err(format!("{label} is not public in rustdoc JSON"));
    }
    Ok(())
}

fn rustdoc_module_id(
    document: &Value,
    crate_name: &str,
    module_name: &str,
) -> Result<Value, String> {
    if document.get("format_version").and_then(Value::as_u64) != Some(57) {
        return Err(format!(
            "{crate_name} rustdoc format drift: {:?}",
            document.get("format_version")
        ));
    }
    if document.get("includes_private").and_then(Value::as_bool) != Some(false) {
        return Err(format!(
            "{crate_name} rustdoc unexpectedly includes private items"
        ));
    }
    let root_id = document
        .get("root")
        .ok_or_else(|| format!("{crate_name} rustdoc has no root ID"))?;
    let root = rustdoc_item(document, root_id, "crate root")?;
    let root_module = rustdoc_item_inner(root, "module", "crate root")?;
    if root_module.get("is_crate").and_then(Value::as_bool) != Some(true) {
        return Err(format!("{crate_name} rustdoc root is not a crate module"));
    }
    let children = root_module
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{crate_name} root module has no item list"))?;
    let mut matches = Vec::new();
    for id in children {
        let item = rustdoc_item(document, id, "crate-root child")?;
        if item.get("name").and_then(Value::as_str) == Some(module_name)
            && item
                .get("inner")
                .and_then(|inner| inner.get("module"))
                .is_some()
        {
            require_rustdoc_public(item, module_name)?;
            let path = rustdoc_canonical_segments(document, id, module_name)?;
            if path != [crate_name, module_name] {
                return Err(format!(
                    "{crate_name} module canonical path drift: {path:?}"
                ));
            }
            matches.push(id.clone());
        }
    }
    if matches.len() != 1 {
        return Err(format!(
            "{crate_name} has {} public {module_name} modules in rustdoc JSON",
            matches.len()
        ));
    }
    Ok(matches.remove(0))
}

fn rustdoc_generics<'a>(data: &'a Value, label: &str) -> Result<&'a Value, String> {
    data.get("generics")
        .ok_or_else(|| format!("{label} has no generics object"))
}

fn rustdoc_generic_names(data: &Value, label: &str) -> Result<Vec<String>, String> {
    let generics = rustdoc_generics(data, label)?;
    let where_predicates = generics
        .get("where_predicates")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} generics has no where_predicates"))?;
    if !where_predicates.is_empty() {
        return Err(format!("{label} has unexpected where predicates"));
    }
    generics
        .get("params")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} generics has no params"))?
        .iter()
        .map(|param| {
            let name = param
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("{label} generic param has no name"))?;
            let type_data = param
                .get("kind")
                .and_then(|kind| kind.get("type"))
                .ok_or_else(|| format!("{label} generic {name} is not a type parameter"))?;
            if type_data
                .get("bounds")
                .and_then(Value::as_array)
                .is_none_or(|bounds| !bounds.is_empty())
                || !type_data.get("default").is_some_and(Value::is_null)
                || type_data.get("is_synthetic").and_then(Value::as_bool) != Some(false)
            {
                return Err(format!("{label} generic {name} shape drift"));
            }
            Ok(name.to_owned())
        })
        .collect()
}

fn rustdoc_impl_ids<'a>(data: &'a Value, label: &str) -> Result<&'a [Value], String> {
    data.get("impls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label} has no impl ID list"))
}

fn render_rustdoc_type(document: &Value, value: &Value, label: &str) -> Result<String, String> {
    if let Some(generic) = value.get("generic").and_then(Value::as_str) {
        if matches!(generic, "Self" | "A") {
            return Ok(generic.to_owned());
        }
        return Err(format!("{label} has unexpected generic type {generic}"));
    }
    if let Some(elements) = value.get("tuple").and_then(Value::as_array) {
        let rendered = elements
            .iter()
            .map(|element| render_rustdoc_type(document, element, label))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(format!("({})", rendered.join(",")));
    }
    let path = value
        .get("resolved_path")
        .ok_or_else(|| format!("{label} uses an unsupported rustdoc type: {value}"))?;
    let id = path
        .get("id")
        .ok_or_else(|| format!("{label} resolved path has no ID"))?;
    let canonical = rustdoc_canonical_segments(document, id, label)?;
    let base = if canonical == ["core", "result", "Result"] {
        "Result".to_owned()
    } else if canonical.len() == 3
        && canonical[0] == "neoethos_gpu_cuda"
        && canonical[1] == "resident_search_slice2_v3"
    {
        canonical[2].clone()
    } else {
        return Err(format!(
            "{label} resolved to an unauthorized type origin: {canonical:?}"
        ));
    };
    if path.get("path").and_then(Value::as_str).is_none() {
        return Err(format!("{label} resolved path has no displayed path"));
    }
    let Some(args) = path.get("args") else {
        return Ok(base);
    };
    if args.is_null() {
        return Ok(base);
    }
    let angle = args
        .get("angle_bracketed")
        .ok_or_else(|| format!("{label} has non-angle generic arguments"))?;
    let constraints = angle
        .get("constraints")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} angle args have no constraints list"))?;
    if !constraints.is_empty() {
        return Err(format!("{label} has associated-type constraints"));
    }
    let rendered = angle
        .get("args")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} angle args have no arg list"))?
        .iter()
        .map(|argument| {
            render_rustdoc_type(
                document,
                argument
                    .get("type")
                    .ok_or_else(|| format!("{label} has a non-type generic argument"))?,
                label,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(format!("{base}<{}>", rendered.join(",")))
}

fn render_rustdoc_function(
    document: &Value,
    function: &Value,
    label: &str,
) -> Result<String, String> {
    let header = function
        .get("header")
        .ok_or_else(|| format!("{label} has no function header"))?;
    if header.get("is_const").and_then(Value::as_bool) != Some(false)
        || header.get("is_unsafe").and_then(Value::as_bool) != Some(false)
        || header.get("is_async").and_then(Value::as_bool) != Some(false)
        || header.get("abi").and_then(Value::as_str) != Some("Rust")
    {
        return Err(format!("{label} function qualifier/ABI drift: {header}"));
    }
    if !rustdoc_generic_names(function, label)?.is_empty() {
        return Err(format!("{label} has method-level generic parameters"));
    }
    let signature = function
        .get("sig")
        .ok_or_else(|| format!("{label} has no signature"))?;
    if signature.get("is_c_variadic").and_then(Value::as_bool) != Some(false) {
        return Err(format!("{label} is variadic"));
    }
    let inputs = signature
        .get("inputs")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} has no inputs"))?;
    if inputs.len() != 1
        || inputs[0].as_array().is_none_or(|input| {
            input.len() != 2
                || input[0].as_str() != Some("self")
                || input[1].get("generic").and_then(Value::as_str) != Some("Self")
        })
    {
        return Err(format!(
            "{label} does not have exactly one by-value self receiver"
        ));
    }
    let output = signature
        .get("output")
        .filter(|value| !value.is_null())
        .ok_or_else(|| format!("{label} has no return type"))?;
    Ok(format!(
        "fn(self)->{}",
        render_rustdoc_type(document, output, label)?
    ))
}

fn rustdoc_local_crate_item(item: &Value) -> bool {
    item.get("crate_id").and_then(Value::as_u64) == Some(0)
}

fn gpu_api_rows(document: &Value) -> Result<BTreeSet<String>, String> {
    const STRUCTS: &[&str] = &[
        "ResidentArchiveKnnCalibrationReceiptV2",
        "ResidentSearchArchiveStagedV3",
        "ResidentSearchGenerationChainV3",
        "ResidentSearchRankEnqueuedV3",
        "ResidentSearchRejectedAuthorityV3",
        "ResidentSearchTerminalPendingV3",
        "ResidentSearchTerminalReceiptV3",
        "ResidentSearchTransitionErrorV3",
    ];
    const BANNED_TRAITS: &[&str] = &[
        "Clone", "Copy", "Default", "Deref", "AsRef", "Borrow", "From", "Into",
    ];

    let module_id = rustdoc_module_id(document, "neoethos_gpu_cuda", "resident_search_slice2_v3")?;
    let module = rustdoc_item(document, &module_id, "GPU API module")?;
    let module_data = rustdoc_item_inner(module, "module", "GPU API module")?;
    let children = module_data
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| "GPU API module has no child list".to_owned())?;
    if children.len() != 9 {
        return Err(format!(
            "GPU API module has {} children, expected 9",
            children.len()
        ));
    }

    let mut rows = BTreeSet::new();
    rows.insert("gpu|module|resident_search_slice2_v3".to_owned());
    let mut named_items = Vec::new();
    let mut names = BTreeSet::new();
    for id in children {
        let item = rustdoc_item(document, id, "GPU API child")?;
        if !rustdoc_local_crate_item(item) {
            return Err("GPU API child is not defined in the local crate".to_owned());
        }
        require_rustdoc_public(item, "GPU API child")?;
        let name = rustdoc_item_name(item, "GPU API child")?.to_owned();
        if !names.insert(name.clone()) {
            return Err(format!("duplicate GPU API child {name}"));
        }
        let path = rustdoc_canonical_segments(document, id, &name)?;
        if path != ["neoethos_gpu_cuda", "resident_search_slice2_v3", &name] {
            return Err(format!("GPU API child {name} path drift: {path:?}"));
        }
        named_items.push((name, id.clone()));
    }
    let mut expected_names = STRUCTS
        .iter()
        .map(|name| (*name).to_owned())
        .collect::<BTreeSet<_>>();
    expected_names.insert("ResidentSearchTryCompleteV3".to_owned());
    if names != expected_names {
        return Err(format!("GPU API child inventory drift: {names:?}"));
    }

    for (name, id) in &named_items {
        let item = rustdoc_item(document, id, name)?;
        let (data, generics, impls) = if STRUCTS.contains(&name.as_str()) {
            let data = rustdoc_item_inner(item, "struct", name)?;
            let plain = data
                .get("kind")
                .and_then(|kind| kind.get("plain"))
                .ok_or_else(|| format!("{name} is not a plain struct"))?;
            let fields = plain
                .get("fields")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{name} has no public field list"))?;
            if !fields.is_empty()
                || plain.get("has_stripped_fields").and_then(Value::as_bool) != Some(true)
            {
                return Err(format!("{name} is not an opaque public struct"));
            }
            let generics = rustdoc_generic_names(data, name)?;
            if name == "ResidentSearchRejectedAuthorityV3" {
                if generics != ["A"] {
                    return Err(format!("{name} generic parameter drift: {generics:?}"));
                }
                rows.insert("gpu|struct|ResidentSearchRejectedAuthorityV3<A>".to_owned());
            } else {
                if !generics.is_empty() {
                    return Err(format!("{name} unexpectedly has generics: {generics:?}"));
                }
                rows.insert(format!("gpu|struct|{name}"));
            }
            (data, generics, rustdoc_impl_ids(data, name)?)
        } else {
            let data = rustdoc_item_inner(item, "enum", name)?;
            let generics = rustdoc_generic_names(data, name)?;
            if !generics.is_empty()
                || data.get("has_stripped_variants").and_then(Value::as_bool) != Some(false)
            {
                return Err(format!("{name} enum shape drift"));
            }
            rows.insert(format!("gpu|enum|{name}"));
            let variants = data
                .get("variants")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{name} has no variant list"))?;
            if variants.len() != 2 {
                return Err(format!(
                    "{name} has {} variants, expected 2",
                    variants.len()
                ));
            }
            let mut variant_names = BTreeSet::new();
            for variant_id in variants {
                let variant = rustdoc_item(document, variant_id, "try-complete variant")?;
                if !rustdoc_local_crate_item(variant)
                    || variant.get("visibility").and_then(Value::as_str) != Some("default")
                {
                    return Err("try-complete variant origin/visibility drift".to_owned());
                }
                let variant_name = rustdoc_item_name(variant, "try-complete variant")?;
                if !variant_names.insert(variant_name.to_owned()) {
                    return Err(format!("duplicate try-complete variant {variant_name}"));
                }
                let tuple = rustdoc_item_inner(variant, "variant", variant_name)?
                    .get("kind")
                    .and_then(|kind| kind.get("tuple"))
                    .and_then(Value::as_array)
                    .ok_or_else(|| format!("variant {variant_name} is not tuple-shaped"))?;
                if tuple.len() != 1 || tuple[0].is_null() {
                    return Err(format!("variant {variant_name} payload cardinality drift"));
                }
                let field = rustdoc_item(document, &tuple[0], "variant tuple field")?;
                let payload = rustdoc_item_inner(field, "struct_field", "variant tuple field")?;
                rows.insert(format!(
                    "gpu|variant|{name}::{variant_name}({})",
                    render_rustdoc_type(document, payload, variant_name)?
                ));
            }
            if variant_names != BTreeSet::from(["Complete".to_owned(), "NotReady".to_owned()]) {
                return Err(format!(
                    "try-complete variant inventory drift: {variant_names:?}"
                ));
            }
            (data, generics, rustdoc_impl_ids(data, name)?)
        };
        let _ = data;

        for impl_id in impls {
            let impl_item = rustdoc_item(document, impl_id, &format!("{name} impl"))?;
            let impl_data = rustdoc_item_inner(impl_item, "impl", &format!("{name} impl"))?;
            let is_local = rustdoc_local_crate_item(impl_item);
            let is_synthetic = impl_data
                .get("is_synthetic")
                .and_then(Value::as_bool)
                .ok_or_else(|| format!("{name} impl has no is_synthetic bit"))?;
            let blanket = impl_data
                .get("blanket_impl")
                .ok_or_else(|| format!("{name} impl has no blanket_impl field"))?;
            let trait_path = impl_data
                .get("trait")
                .ok_or_else(|| format!("{name} impl has no trait field"))?;
            if !trait_path.is_null() {
                let trait_id = trait_path
                    .get("id")
                    .ok_or_else(|| format!("{name} trait impl has no trait ID"))?;
                let origin = rustdoc_canonical_segments(document, trait_id, "trait impl")?;
                let trait_name = origin.last().map(String::as_str).unwrap_or("");
                if is_local
                    && !is_synthetic
                    && blanket.is_null()
                    && BANNED_TRAITS.contains(&trait_name)
                {
                    return Err(format!(
                        "{name} has banned explicit local impl of {}",
                        origin.join("::")
                    ));
                }
                continue;
            }
            if !is_local || is_synthetic || !blanket.is_null() {
                return Err(format!("{name} has a nonlocal/synthetic inherent impl"));
            }
            let impl_generics = rustdoc_generic_names(impl_data, &format!("{name} impl"))?;
            if impl_generics != generics {
                return Err(format!(
                    "{name} inherent impl generic drift: {impl_generics:?}"
                ));
            }
            let owner = render_rustdoc_type(
                document,
                impl_data
                    .get("for")
                    .ok_or_else(|| format!("{name} inherent impl has no owner type"))?,
                &format!("{name} impl owner"),
            )?;
            let expected_owner = if generics.is_empty() {
                name.clone()
            } else {
                format!("{name}<{}>", generics.join(","))
            };
            if owner != expected_owner {
                return Err(format!("{name} inherent impl owner drift: {owner}"));
            }
            let method_owner = expected_owner;
            for method_id in impl_data
                .get("items")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{name} inherent impl has no item list"))?
            {
                let method = rustdoc_item(document, method_id, &format!("{name} impl item"))?;
                if method.get("visibility").and_then(Value::as_str) != Some("public") {
                    continue;
                }
                if !rustdoc_local_crate_item(method) {
                    return Err(format!("{name} public inherent item is nonlocal"));
                }
                let method_name = rustdoc_item_name(method, &format!("{name} method"))?;
                let function = rustdoc_item_inner(method, "function", method_name)?;
                rows.insert(format!(
                    "gpu|method|{method_owner}|{method_name}|{}",
                    render_rustdoc_function(document, function, method_name)?
                ));
            }
        }
    }
    Ok(rows)
}

fn search_api_rows(document: &Value) -> Result<BTreeSet<String>, String> {
    let module_id = rustdoc_module_id(document, "neoethos_search", "resident_search_slice2_v3")?;
    let module = rustdoc_item(document, &module_id, "Search API module")?;
    let children = rustdoc_item_inner(module, "module", "Search API module")?
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| "Search API module has no child list".to_owned())?;
    if children.len() != 10 {
        return Err(format!(
            "Search API module has {} children, expected 10",
            children.len()
        ));
    }
    let mut rows = BTreeSet::new();
    rows.insert("search|module|resident_search_slice2_v3".to_owned());
    let mut names = BTreeSet::new();
    for id in children {
        let item = rustdoc_item(document, id, "Search API child")?;
        require_rustdoc_public(item, "Search API child")?;
        let use_data = rustdoc_item_inner(item, "use", "Search API child")?;
        if use_data.get("is_glob").and_then(Value::as_bool) != Some(false) {
            return Err("Search API contains a glob re-export".to_owned());
        }
        if use_data.get("source").and_then(Value::as_str).is_none() {
            return Err("Search API re-export has no source text".to_owned());
        }
        let name = use_data
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| "Search API re-export has no name".to_owned())?;
        if !names.insert(name.to_owned()) {
            return Err(format!("duplicate Search API re-export {name}"));
        }
        let origin_id = use_data
            .get("id")
            .filter(|value| !value.is_null())
            .ok_or_else(|| format!("Search API re-export {name} has no origin ID"))?;
        let mut origin = rustdoc_canonical_segments(document, origin_id, name)?;
        if origin.first().map(String::as_str) == Some("neoethos_search") {
            origin[0] = "crate".to_owned();
        }
        rows.insert(format!("search|reexport|{name}|{}", origin.join("::")));
    }
    Ok(rows)
}

fn read_rustdoc_document(path: &Path, label: &str) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {label}: {error}"))?;
    if bytes.starts_with(&[0xef, 0xbb, 0xbf]) {
        return Err(format!("{label} has a UTF-8 BOM"));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("{label} is not valid UTF-8: {error}"))?;
    serde_json::from_str(text).map_err(|error| format!("{label} is not one JSON document: {error}"))
}

#[derive(Clone)]
struct RustdocArtifact {
    path: PathBuf,
    bytes: u64,
    digest: String,
}

fn verify_rustdoc_inventory(
    context: &RunContext,
    expected_names: &[&str],
) -> Result<Vec<RustdocArtifact>, String> {
    let doc = context.target_doc.join("doc");
    let mut json_files = Vec::new();
    let mut pending = vec![doc.clone()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)
            .map_err(|error| format!("cannot enumerate rustdoc output: {error}"))?
        {
            let entry =
                entry.map_err(|error| format!("cannot read rustdoc output entry: {error}"))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| format!("cannot inspect rustdoc output entry: {error}"))?;
            #[cfg(windows)]
            let is_reparse = {
                use std::os::windows::fs::MetadataExt;
                metadata.file_attributes() & 0x0400 != 0
            };
            #[cfg(not(windows))]
            let is_reparse = metadata.file_type().is_symlink();
            if is_reparse {
                return Err(format!(
                    "rustdoc output contains a symlink/junction/reparse point: {}",
                    path.display()
                ));
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() && path.extension() == Some(OsStr::new("json")) {
                json_files.push(canonical(&path, "rustdoc JSON output")?);
            }
        }
    }
    json_files.sort_by_key(|path| normalized_path_text(path).to_ascii_lowercase());
    let mut expected = expected_names
        .iter()
        .map(|name| canonical(&doc.join(name), "expected rustdoc JSON"))
        .collect::<Result<Vec<_>, _>>()?;
    expected.sort_by_key(|path| normalized_path_text(path).to_ascii_lowercase());
    if json_files.len() != expected.len()
        || json_files
            .iter()
            .zip(&expected)
            .any(|(observed, expected)| !same_path(observed, expected))
    {
        return Err(format!("rustdoc JSON inventory drift: {json_files:?}"));
    }
    json_files
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path)
                .map_err(|error| format!("cannot hash rustdoc JSON {}: {error}", path.display()))?;
            Ok(RustdocArtifact {
                path,
                bytes: bytes.len() as u64,
                digest: sha256(&bytes),
            })
        })
        .collect()
}

fn write_rustdoc_artifact_ledger(
    context: &RunContext,
    name: &str,
    artifacts: &[RustdocArtifact],
) -> Result<(), String> {
    let mut rows = artifacts
        .iter()
        .map(|artifact| {
            let file_name = artifact
                .path
                .file_name()
                .and_then(OsStr::to_str)
                .ok_or_else(|| "rustdoc JSON filename is not UTF-8".to_owned())?;
            Ok(format!(
                "{} {} doc/{file_name}",
                artifact.digest, artifact.bytes
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    rows.sort();
    fs::write(
        context.evidence.join(format!("{name}.artifacts.sha256")),
        format!("{}\n", rows.join("\n")),
    )
    .map_err(|error| format!("cannot write {name} rustdoc artifact ledger: {error}"))
}

fn run_rustdoc_api(context: &RunContext) -> Result<(), String> {
    let mut gpu_before_search = None;
    let mut final_artifacts = Vec::new();
    for (index, spec) in RUSTDOC_SPECS.iter().enumerate() {
        let capture = capture_cargo_process(
            context,
            &context.target_doc,
            spec.evidence_name,
            &rustdoc_arguments(spec.package_name),
        )?;
        let events = parse_rustdoc_cargo_stream(context, *spec, &capture)?;
        finish_capture_events(context, spec.evidence_name, &events)?;
        if index == 0 {
            let artifacts = verify_rustdoc_inventory(context, &["neoethos_gpu_cuda.json"])?;
            write_rustdoc_artifact_ledger(context, "gpu-rustdoc", &artifacts)?;
            gpu_before_search = artifacts.into_iter().next();
        } else {
            final_artifacts = verify_rustdoc_inventory(
                context,
                &["neoethos_gpu_cuda.json", "neoethos_search.json"],
            )?;
            let gpu_after = final_artifacts
                .iter()
                .find(|artifact| {
                    artifact.path.file_name() == Some(OsStr::new("neoethos_gpu_cuda.json"))
                })
                .ok_or_else(|| "GPU rustdoc JSON disappeared after Search rustdoc".to_owned())?;
            let gpu_before = gpu_before_search
                .as_ref()
                .ok_or_else(|| "GPU rustdoc pre-Search hash is absent".to_owned())?;
            if gpu_before.bytes != gpu_after.bytes || gpu_before.digest != gpu_after.digest {
                return Err("GPU rustdoc JSON changed during Search rustdoc".to_owned());
            }
            write_rustdoc_artifact_ledger(context, "rustdoc-final", &final_artifacts)?;
        }
    }
    let gpu_path = final_artifacts
        .iter()
        .find(|artifact| artifact.path.file_name() == Some(OsStr::new("neoethos_gpu_cuda.json")))
        .map(|artifact| artifact.path.clone())
        .ok_or_else(|| "final GPU rustdoc JSON is absent".to_owned())?;
    let search_path = final_artifacts
        .iter()
        .find(|artifact| artifact.path.file_name() == Some(OsStr::new("neoethos_search.json")))
        .map(|artifact| artifact.path.clone())
        .ok_or_else(|| "final Search rustdoc JSON is absent".to_owned())?;
    let gpu = read_rustdoc_document(&gpu_path, "GPU rustdoc JSON")?;
    let search = read_rustdoc_document(&search_path, "Search rustdoc JSON")?;
    let mut rows = gpu_api_rows(&gpu)?;
    for row in search_api_rows(&search)? {
        if !rows.insert(row.clone()) {
            return Err(format!("duplicate cross-crate API receipt row: {row}"));
        }
    }
    let observed = format!("{}\n", rows.into_iter().collect::<Vec<_>>().join("\n"));
    if observed.as_bytes() != API_SURFACE_V3.as_bytes() {
        return Err(format!(
            "rustdoc API allowlist mismatch\n--- expected ---\n{API_SURFACE_V3}--- observed ---\n{observed}"
        ));
    }
    let receipt = context.fixture.join("api-surface-v3.txt");
    let tracked =
        fs::read(&receipt).map_err(|error| format!("cannot read tracked API receipt: {error}"))?;
    if tracked != API_SURFACE_V3.as_bytes() {
        return Err("tracked api-surface-v3.txt drift".to_owned());
    }
    Ok(())
}

fn parse_cargo_json(
    stdout: &str,
    stderr: &str,
    status: ExitStatus,
    case: &UiCase,
    target_root: &Path,
    context: &RunContext,
) -> Result<(Observation, Vec<String>), String> {
    if let Some(activation) = activation_from_process_stderr(stderr) {
        return Err(format!(
            "CUDA/native activation in process stderr: {activation}"
        ));
    }

    let expected_source = canonical(&context.fixture.join(case.source), "expected case source")?;
    let mut events = Vec::new();
    let mut build_finished = None;
    let mut selected_warning_events = 0;
    let mut selected_error_events = 0;
    let mut dependency_error_events = 0;
    let mut wrong_primary_spans = 0;
    let mut authored_errors = Vec::new();

    for (line_index, line) in stdout.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let message: Value = serde_json::from_str(line).map_err(|error| {
            format!(
                "non-JSON Cargo stdout at line {}: {error}; text={line:?}",
                line_index + 1
            )
        })?;
        if let Some(activation) = activation_from_json(&message) {
            return Err(format!(
                "CUDA/native activation in Cargo JSON: {activation}"
            ));
        }
        let reason = message
            .get("reason")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("Cargo JSON line {} has no reason", line_index + 1))?;

        if reason == "build-finished" {
            if build_finished.is_some() {
                return Err("duplicate build-finished message".to_owned());
            }
            build_finished = Some(
                message
                    .get("success")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| "build-finished has no boolean success".to_owned())?,
            );
            events.push(format!(
                "cargo-json|INFO|line={}|reason=build-finished|success={}",
                line_index + 1,
                build_finished.expect("just assigned")
            ));
            continue;
        }

        if reason != "compiler-message" {
            events.push(format!(
                "cargo-json|INFO|line={}|reason={reason}",
                line_index + 1
            ));
            continue;
        }

        let package_id = message
            .get("package_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "compiler-message has no package_id".to_owned())?;
        let is_fixture_package = fixture_package_identity(package_id, &context.fixture)?;
        let normalized_package = if is_fixture_package {
            FIXTURE_PACKAGE.to_owned()
        } else {
            package_id.to_owned()
        };
        let target = message
            .get("target")
            .ok_or_else(|| "compiler-message has no target".to_owned())?;
        let target_name = target
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| "compiler-message target has no name".to_owned())?;
        let target_kind = value_string_array(target, "kind")?;
        let crate_types = value_string_array(target, "crate_types")?;
        let src_path = target
            .get("src_path")
            .and_then(Value::as_str)
            .ok_or_else(|| "compiler-message target has no src_path".to_owned())?;
        let diagnostic = message
            .get("message")
            .ok_or_else(|| "compiler-message has no diagnostic message".to_owned())?;
        let level = diagnostic
            .get("level")
            .and_then(Value::as_str)
            .ok_or_else(|| "compiler diagnostic has no level".to_owned())?;
        let code = diagnostic
            .get("code")
            .and_then(|value| value.get("code"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let spans = diagnostic
            .get("spans")
            .and_then(Value::as_array)
            .ok_or_else(|| "compiler diagnostic has no spans array".to_owned())?;
        let children = diagnostic
            .get("children")
            .and_then(Value::as_array)
            .ok_or_else(|| "compiler diagnostic has no children array".to_owned())?;
        let rendered = diagnostic
            .get("rendered")
            .and_then(Value::as_str)
            .unwrap_or("");
        let encoded_spans = serde_json::to_string(spans)
            .map_err(|error| format!("cannot serialize diagnostic spans: {error}"))?;
        events.push(format!(
            "compiler-message|{}|line={}|package={}|target={}|kind={:?}|crate-types={:?}|code={:?}|spans={encoded_spans}",
            if level == "error" {
                "ERROR"
            } else if level == "warning" {
                "WARNING"
            } else {
                "INFO"
            },
            line_index + 1,
            serde_json::to_string(&normalized_package)
                .expect("serializing a string cannot fail"),
            serde_json::to_string(target_name).expect("serializing a string cannot fail"),
            target_kind,
            crate_types,
            code
        ));
        for child in children {
            events.push(format!(
                "compiler-child|INFO|parent-line={}|child={}",
                line_index + 1,
                serde_json::to_string(child)
                    .map_err(|error| format!("cannot serialize child diagnostic: {error}"))?
            ));
        }

        let is_selected_target = is_fixture_package && target_name == case.bin;
        if !is_fixture_package && target_name == case.bin {
            return Err(format!(
                "selected bin name came from a foreign package: {package_id}"
            ));
        }
        if is_fixture_package && !is_selected_target {
            return Err(format!(
                "fixture compiler-message targeted {target_name}, expected {}",
                case.bin
            ));
        }
        if !is_selected_target {
            if level == "error" {
                dependency_error_events += 1;
            }
            continue;
        }

        if target_kind.len() != 1
            || target_kind[0] != "bin"
            || crate_types.len() != 1
            || crate_types[0] != "bin"
        {
            return Err(format!(
                "selected target kind/crate_types drift: {target_kind:?}/{crate_types:?}"
            ));
        }
        let resolved_src = resolve_span_path(src_path, context)?;
        if !same_path(&resolved_src, &expected_source) {
            return Err(format!(
                "selected target source drift: {}",
                normalized_path_text(&resolved_src)
            ));
        }

        if level == "warning" {
            selected_warning_events += 1;
        }
        if level != "error" {
            continue;
        }
        selected_error_events += 1;
        let normalized_rendered =
            normalize_rendered(rendered, target_root, &context.fixture, &context.repo)?;
        for span in spans
            .iter()
            .filter(|span| span.get("is_primary").and_then(Value::as_bool) == Some(true))
        {
            let file_name = span
                .get("file_name")
                .and_then(Value::as_str)
                .ok_or_else(|| "primary span has no file_name".to_owned())?;
            let span_path = resolve_span_path(file_name, context)?;
            if !same_path(&span_path, &expected_source) {
                wrong_primary_spans += 1;
                continue;
            }
            authored_errors.push(AuthoredError {
                code: code.clone(),
                line: span
                    .get("line_start")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| "primary span has no line_start".to_owned())?
                    as usize,
                column_start: span
                    .get("column_start")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| "primary span has no column_start".to_owned())?
                    as usize,
                column_end: span
                    .get("column_end")
                    .and_then(Value::as_u64)
                    .ok_or_else(|| "primary span has no column_end".to_owned())?
                    as usize,
                rendered: normalized_rendered.clone(),
            });
        }
    }

    let process_stderr = classify_process_stderr(stderr, &mut events);
    let build_finished_success =
        build_finished.ok_or_else(|| "missing build-finished Cargo JSON message".to_owned())?;
    Ok((
        Observation {
            status,
            build_finished_success,
            selected_warning_events,
            selected_error_events,
            dependency_error_events,
            wrong_primary_spans,
            authored_errors,
            process_stderr,
        },
        events,
    ))
}

fn validate_observation(observation: &Observation, case: &UiCase) -> Result<(), String> {
    let expected_terminal_errors = usize::from(case.expected_code.is_some());
    if observation.process_stderr.warning != 0
        || observation.process_stderr.error != expected_terminal_errors
        || observation.process_stderr.other != 0
    {
        return Err(format!(
            "{} process-stderr classification mismatch: info={}, warning={}, error={}, other={}, expected terminal errors={expected_terminal_errors}",
            case.bin,
            observation.process_stderr.info,
            observation.process_stderr.warning,
            observation.process_stderr.error,
            observation.process_stderr.other
        ));
    }
    if observation.dependency_error_events != 0 || observation.wrong_primary_spans != 0 {
        return Err(format!(
            "dependency/wrong-primary errors: dependency={}, wrong-primary={}",
            observation.dependency_error_events, observation.wrong_primary_spans
        ));
    }

    match case.expected_code {
        None => {
            if !observation.status.success()
                || !observation.build_finished_success
                || observation.selected_warning_events != 0
                || observation.selected_error_events != 0
                || !observation.authored_errors.is_empty()
            {
                let exact_missing_api_red = !observation.status.success()
                    && !observation.build_finished_success
                    && observation.selected_warning_events == 0
                    && observation.authored_errors.len() == 2
                    && observation.authored_errors.iter().any(|error| {
                        error.code.as_deref() == Some("E0432")
                            && error.line == 1
                            && error.column_start == 5
                            && error.column_end == 56
                    })
                    && observation.authored_errors.iter().any(|error| {
                        error.code.as_deref() == Some("E0432")
                            && error.line == 2
                            && error.column_start == 5
                            && error.column_end == 57
                    });
                if exact_missing_api_red {
                    return Err(format!(
                        "positive {} reached the compiler and observed the exact two missing canonical-module E0432 diagnostics; this is the intended tests-first mismatch: {:#?}",
                        case.bin, observation.authored_errors
                    ));
                }
                return Err(format!(
                    "positive {} expected success but observed exit={:?}, build-finished.success={}, selected-warnings={}, selected-errors={}, authored={:#?}",
                    case.bin,
                    observation.status.code(),
                    observation.build_finished_success,
                    observation.selected_warning_events,
                    observation.selected_error_events,
                    observation.authored_errors
                ));
            }
        }
        Some(expected_code) => {
            if observation.status.success() || observation.build_finished_success {
                return Err(format!("negative {} unexpectedly succeeded", case.bin));
            }
            if observation.selected_warning_events != 0
                || observation.selected_error_events == 0
                || observation.authored_errors.len() != 1
            {
                return Err(format!(
                    "negative {} diagnostic cardinality mismatch: warnings={}, errors={}, authored={:#?}",
                    case.bin,
                    observation.selected_warning_events,
                    observation.selected_error_events,
                    observation.authored_errors
                ));
            }
            let authored = &observation.authored_errors[0];
            if authored.code.as_deref() != Some(expected_code)
                || authored.line != case.line
                || authored.column_start != case.column_start
                || authored.column_end != case.column_end
                || authored.rendered.is_empty()
            {
                return Err(format!(
                    "negative {} expectation mismatch: expected {} {}:{}-{}, observed {authored:#?}",
                    case.bin, expected_code, case.line, case.column_start, case.column_end
                ));
            }
        }
    }
    Ok(())
}

fn run_case(context: &RunContext, case: &UiCase) -> Result<(), String> {
    let target = match case.feature {
        FeatureMode::On => &context.target_on,
        FeatureMode::Off => &context.target_off,
    };
    let mut arguments = vec![
        OsString::from("+nightly-2026-04-07"),
        OsString::from("check"),
        OsString::from("--manifest-path"),
        context.manifest.as_os_str().to_owned(),
        OsString::from("--locked"),
        OsString::from("--offline"),
        OsString::from("-j"),
        OsString::from("7"),
        OsString::from("--no-default-features"),
    ];
    if case.feature == FeatureMode::On {
        arguments.extend([
            OsString::from("--features"),
            OsString::from("resident-search-slice2-compile-contract"),
        ]);
    }
    arguments.extend([
        OsString::from("--bin"),
        OsString::from(case.bin),
        OsString::from("--message-format=json"),
        OsString::from("--color"),
        OsString::from("never"),
    ]);
    let capture = capture_cargo_process(context, target, case.bin, &arguments)?;
    let stdout = capture.stdout_text(case.bin)?;
    let stderr = capture.stderr_text(case.bin)?;

    let (observation, events) =
        parse_cargo_json(stdout, stderr, capture.status, case, target, context)?;
    finish_capture_events(context, case.bin, &events)?;
    validate_observation(&observation, case)
}

#[test]
fn resident_search_slice2_compile_contract_v9() {
    let context = build_context().unwrap_or_else(|error| panic!("R7 v9 preflight failed: {error}"));
    verify_tool_identities(&context)
        .unwrap_or_else(|error| panic!("R7 v9 identity preflight failed: {error}"));
    for mode in [
        MetadataMode::FeatureOn,
        MetadataMode::FeatureOff,
        MetadataMode::Outer,
    ] {
        run_metadata(&context, mode)
            .unwrap_or_else(|error| panic!("R7 v9 metadata preflight failed: {error}"));
    }
    verify_child_disk_budget(&context, "rustdoc")
        .unwrap_or_else(|error| panic!("R7 v9 preflight failed: {error}"));
    run_rustdoc_api(&context).unwrap_or_else(|error| panic!("R7 v9 rustdoc/API mismatch: {error}"));
    clean_child_target(&context, &context.target_doc, "clean-doc")
        .unwrap_or_else(|error| panic!("R7 v9 doc cleanup failed: {error}"));
    verify_child_disk_budget(&context, "feature-on")
        .unwrap_or_else(|error| panic!("R7 v9 preflight failed: {error}"));
    for case in CASES.iter().filter(|case| case.feature == FeatureMode::On) {
        run_case(&context, case).unwrap_or_else(|error| {
            panic!(
                "R7 v9 provisional compiler contract mismatch for {}: {error}",
                case.bin
            )
        });
    }
    clean_child_target(&context, &context.target_on, "clean-on")
        .unwrap_or_else(|error| panic!("R7 v9 feature-on cleanup failed: {error}"));
    verify_child_disk_budget(&context, "feature-off")
        .unwrap_or_else(|error| panic!("R7 v9 preflight failed: {error}"));
    for case in CASES.iter().filter(|case| case.feature == FeatureMode::Off) {
        run_case(&context, case).unwrap_or_else(|error| {
            panic!(
                "R7 v9 provisional compiler contract mismatch for {}: {error}",
                case.bin
            )
        });
    }
    clean_child_target(&context, &context.target_off, "clean-off")
        .unwrap_or_else(|error| panic!("R7 v9 feature-off cleanup failed: {error}"));
}
