//! Portable → installed migration detector.
//!
//! Spec: `installer_wizard_ux_spec.md` §6 "Migration from portable".
//! forex-ai pre-0.5 was portable (all state under `~/.forex-ai/`); the
//! installer-aware wizard sniffs for that directory at Step 2 entry
//! and surfaces a migration prompt.
//!
//! Scope of THIS module: detection only. The migration UI itself
//! lives in `path.rs`; the actual file-copy machinery is
//! `// TODO(wizard-migration-copy)` because spec §6 wants
//! per-file SHA-256 + atomic-move semantics that need a careful
//! review pass against `broker_persistence.rs`.

use std::path::{Path, PathBuf};

/// Files that, if present in a candidate directory, qualify it as a
/// legacy portable install. Spec §6 enumerates these verbatim.
pub const WIZARD_PORTABLE_SENTINEL_FILES: &[&str] = &[
    "config.yaml",
    "broker_credentials.toml",
];

/// Directories that count as legacy payloads.
pub const WIZARD_PORTABLE_SENTINEL_DIRS: &[&str] = &["checkpoints", "data", "history"];

/// Where to look for the legacy portable install. Spec §6 lists three
/// canonical roots; we probe all of them and stop at the first that
/// matches.
pub fn portable_candidate_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(home) = dirs::home_dir() {
        roots.push(home.join(".forex-ai"));
        roots.push(home.join("forex-ai"));
    }
    // Windows-style `%USERPROFILE%/.forex-ai` is covered by `home_dir`
    // on Windows.
    roots
}

#[derive(Debug, Clone, Default)]
pub struct PortableMigrationReport {
    pub root: PathBuf,
    pub has_config_yaml: bool,
    pub has_broker_credentials: bool,
    pub has_checkpoints: bool,
    pub has_data: bool,
    pub has_history: bool,
}

impl PortableMigrationReport {
    pub fn summary_lines(&self) -> Vec<String> {
        let mut out = vec![format!("Source: {}", self.root.display())];
        if self.has_config_yaml {
            out.push("  • config.yaml".to_string());
        }
        if self.has_broker_credentials {
            out.push("  • broker_credentials.toml".to_string());
        }
        if self.has_checkpoints {
            out.push("  • checkpoints/".to_string());
        }
        if self.has_data {
            out.push("  • data/".to_string());
        }
        if self.has_history {
            out.push("  • history/".to_string());
        }
        out
    }

    /// Has anything migrate-worthy been detected at all?
    pub fn is_actionable(&self) -> bool {
        self.has_config_yaml
            || self.has_broker_credentials
            || self.has_checkpoints
            || self.has_data
            || self.has_history
    }
}

/// Walk the candidate roots and return the first that contains at
/// least one sentinel file or dir. Returns `None` if no legacy
/// install is detected.
pub fn detect_portable_install() -> Option<PortableMigrationReport> {
    for root in portable_candidate_roots() {
        let report = describe_root(&root);
        if report.is_actionable() {
            return Some(report);
        }
    }
    None
}

/// Describe a single candidate root. Public so tests can inject a
/// scratch directory rather than touching the real `$HOME`.
pub fn describe_root(root: &Path) -> PortableMigrationReport {
    PortableMigrationReport {
        root: root.to_path_buf(),
        has_config_yaml: root.join("config.yaml").is_file(),
        has_broker_credentials: root.join("broker_credentials.toml").is_file(),
        has_checkpoints: root.join("checkpoints").is_dir(),
        has_data: root.join("data").is_dir(),
        has_history: root.join("history").is_dir(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_dir(label: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("forex-ai-wizard-migration-{}-{}", label, nanos));
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn describe_empty_root_is_not_actionable() {
        let root = tmp_dir("empty");
        let report = describe_root(&root);
        assert!(!report.is_actionable());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn describe_root_detects_config_yaml_only() {
        let root = tmp_dir("config-only");
        fs::write(root.join("config.yaml"), b"placeholder").unwrap();
        let report = describe_root(&root);
        assert!(report.has_config_yaml);
        assert!(report.is_actionable());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn describe_root_detects_dirs() {
        let root = tmp_dir("dirs");
        fs::create_dir_all(root.join("history")).unwrap();
        fs::create_dir_all(root.join("checkpoints")).unwrap();
        let report = describe_root(&root);
        assert!(report.has_history);
        assert!(report.has_checkpoints);
        assert!(!report.has_data);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn portable_sentinel_files_constants_are_non_empty() {
        assert!(!WIZARD_PORTABLE_SENTINEL_FILES.is_empty());
        assert!(!WIZARD_PORTABLE_SENTINEL_DIRS.is_empty());
    }
}
