//! Inference runtime trait surface + bundled-model path resolver.
//!
//! Phase G0 — fail-loud `StubGemmaRuntime`. The real
//! `mistral.rs`-backed implementation lands in G1 behind the
//! `runtime-mistralrs` cargo feature. The trait is fixed now
//! so downstream layers (G2 gate, G3 tools, G4 bridge) can be
//! built + tested against the stub.
//!
//! ## Bundled-model path resolution
//!
//! Per the operator's 2026-05-18 directive, the Gemma 4 E4B
//! Uncensored Aggressive GGUF ships **bundled** with the
//! installer (no first-run download — the binary's enough on
//! its own). At runtime we look for the model file in this
//! order:
//!
//! 1. `FOREX_AI_GEMMA_MODEL_PATH` env override — dev convenience.
//! 2. `<exe_dir>/resources/models/*.gguf` — installed bundle.
//! 3. `<project_root>/resources/models/*.gguf` — dev tree fallback
//!    (when running `cargo run` from the repo).
//! 4. `<dirs::data_dir>/forex-ai/models/*.gguf` — XDG-style
//!    user-data cache so the user can drop in a manual swap
//!    even after the app is installed.
//!
//! First hit wins. If none exists, `resolve_bundled_model_path`
//! returns `GemmaError::ConfigInvalid` with a message naming
//! every path it tried — operators get an actionable error,
//! not a silent "model missing".
//!
//! The bundled filename convention is
//! `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`
//! (the operator approved Q4_K_M as default; Q5_K_M is OK as a
//! drop-in if the user replaces the file).

use crate::error::GemmaError;
use anyhow::Result;
use std::path::{Path, PathBuf};

/// Synchronous, request-shaped inference API. Async streaming
/// arrives in G1 alongside mistral.rs integration; the sync
/// shape is what we exercise in G0 tests without dragging tokio
/// into the trait.
pub trait GemmaRuntime: Send + Sync {
    /// Run a single prompt through the model and return the
    /// full response text.
    fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String>;
    /// Human-readable identifier for the loaded model.
    fn model_id(&self) -> &str;
}

/// Token-by-token streaming primitive — what G1 will actually
/// drive. The shape is `Iterator<Item = Result<String>>` so the
/// SSE bridge (G8) can pipe tokens straight from this iterator
/// into the wire stream without an async runtime in between.
pub trait GemmaTokenStream: Send {
    fn next_token(&mut self) -> Option<Result<String>>;
}

/// G0 stub. Returns `GemmaError::pending("G1 inference")` on
/// every call. The constructor is `pub` so other layers can
/// wire trait-object tests without pulling in mistral.rs.
pub struct StubGemmaRuntime {
    model_id: String,
}

impl StubGemmaRuntime {
    pub fn new() -> Self {
        Self {
            model_id: "stub-no-model-loaded".to_string(),
        }
    }
    pub fn with_model_id(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
        }
    }
}

impl Default for StubGemmaRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl GemmaRuntime for StubGemmaRuntime {
    fn generate(&self, _prompt: &str, _max_tokens: u32) -> Result<String> {
        Err(GemmaError::pending("G1 mistral.rs inference runtime").into())
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }
}

// ---------------------------------------------------------------------------
// Bundled-model path resolution
// ---------------------------------------------------------------------------

/// Default bundled filename — what the installer copies into
/// `resources/models/` and what the resolver looks for first
/// inside each candidate directory. Operator can drop in a
/// different file with this name to swap quantization.
pub const BUNDLED_MODEL_FILENAME: &str = "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf";

/// Env-var name the dev / operator can set to override the
/// resolver entirely.
pub const MODEL_PATH_ENV_VAR: &str = "FOREX_AI_GEMMA_MODEL_PATH";

/// HuggingFace download URL for the default bundled quant —
/// used by the installer-prep script (`scripts/fetch-gemma-model.ps1`)
/// and documented in the crate README so operators can re-fetch
/// the file if it ever goes missing.
pub const BUNDLED_MODEL_DOWNLOAD_URL: &str = "https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/resolve/main/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf";

/// Approximate on-disk size of the bundled quant, in bytes.
/// Updated when the operator rebases the bundled quant. Used by
/// the disk-safety pre-check in the installer-prep script.
pub const BUNDLED_MODEL_APPROX_BYTES: u64 = 5_000_000_000;

/// Result of a successful path-resolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModelPath {
    pub path: PathBuf,
    /// Which candidate matched (`"env"`, `"exe_dir"`, etc.).
    pub source: &'static str,
}

/// Pluggable filesystem probe. Tests inject a `FakeFsProbe`
/// that pretends a file exists at a specific path so the
/// resolver's precedence rules are tested in isolation without
/// touching real disk.
pub trait FsProbe: Send + Sync {
    fn file_exists(&self, path: &Path) -> bool;
    fn env_var(&self, name: &str) -> Option<String>;
    fn current_exe_dir(&self) -> Option<PathBuf>;
    fn manifest_dir(&self) -> Option<PathBuf>;
    fn user_data_dir(&self) -> Option<PathBuf>;
}

/// Real filesystem probe — uses `std::env` and `std::env::current_exe`.
pub struct RealFsProbe;

impl FsProbe for RealFsProbe {
    fn file_exists(&self, path: &Path) -> bool {
        path.is_file()
    }
    fn env_var(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
    fn current_exe_dir(&self) -> Option<PathBuf> {
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf))
    }
    fn manifest_dir(&self) -> Option<PathBuf> {
        // CARGO_MANIFEST_DIR is set at build time for the crate
        // being compiled. At runtime we just fall back to the
        // current working directory — `cargo run` from the
        // repo root will resolve through here.
        std::env::current_dir().ok()
    }
    fn user_data_dir(&self) -> Option<PathBuf> {
        // We don't take a hard dep on the `dirs` crate at this
        // layer; the real path lookup is something the chrome
        // can provide once we're wired in. For now we fall
        // back to `$HOME/.forex-ai/models` (POSIX) /
        // `%LOCALAPPDATA%\forex-ai\models` (Windows).
        if cfg!(windows) {
            std::env::var("LOCALAPPDATA")
                .ok()
                .map(|s| PathBuf::from(s).join("forex-ai"))
        } else {
            std::env::var("HOME")
                .ok()
                .map(|s| PathBuf::from(s).join(".forex-ai"))
        }
    }
}

/// Resolve the bundled-model path via the chain documented in
/// the module-level docs. Returns the first hit, or
/// `GemmaError::ConfigInvalid` listing every path tried.
pub fn resolve_bundled_model_path(
    probe: &dyn FsProbe,
) -> std::result::Result<ResolvedModelPath, GemmaError> {
    let mut tried: Vec<String> = Vec::new();

    // 1. env var override
    if let Some(raw) = probe.env_var(MODEL_PATH_ENV_VAR) {
        let p = PathBuf::from(&raw);
        tried.push(format!("env({MODEL_PATH_ENV_VAR})={raw}"));
        if probe.file_exists(&p) {
            return Ok(ResolvedModelPath {
                path: p,
                source: "env",
            });
        }
    }

    // 2. exe_dir / resources / models / <filename>
    if let Some(exe_dir) = probe.current_exe_dir() {
        let p = exe_dir
            .join("resources")
            .join("models")
            .join(BUNDLED_MODEL_FILENAME);
        tried.push(format!("exe_dir({})", p.display()));
        if probe.file_exists(&p) {
            return Ok(ResolvedModelPath {
                path: p,
                source: "exe_dir",
            });
        }
    }

    // 3. manifest_dir / resources / models / <filename>
    if let Some(mfd) = probe.manifest_dir() {
        let p = mfd
            .join("resources")
            .join("models")
            .join(BUNDLED_MODEL_FILENAME);
        tried.push(format!("manifest_dir({})", p.display()));
        if probe.file_exists(&p) {
            return Ok(ResolvedModelPath {
                path: p,
                source: "manifest_dir",
            });
        }
    }

    // 4. user_data_dir / models / <filename>
    if let Some(udd) = probe.user_data_dir() {
        let p = udd.join("models").join(BUNDLED_MODEL_FILENAME);
        tried.push(format!("user_data_dir({})", p.display()));
        if probe.file_exists(&p) {
            return Ok(ResolvedModelPath {
                path: p,
                source: "user_data_dir",
            });
        }
    }

    Err(GemmaError::ConfigInvalid {
        reason: format!(
            "Gemma model file '{BUNDLED_MODEL_FILENAME}' not found. \
             Set {MODEL_PATH_ENV_VAR} or drop the .gguf in one of: \
             {}. Download URL: {BUNDLED_MODEL_DOWNLOAD_URL}",
            tried.join("; ")
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    #[test]
    fn stub_generate_fails_loud_with_phase_tag() {
        let rt = StubGemmaRuntime::new();
        let err = rt.generate("hello", 64).expect_err("must bail");
        let msg = err.to_string();
        assert!(msg.contains("G1"));
        assert!(msg.contains("not yet implemented"));
    }

    #[test]
    fn stub_model_id_carries_constructor_input() {
        let rt = StubGemmaRuntime::with_model_id("my-gemma");
        assert_eq!(rt.model_id(), "my-gemma");
    }

    #[test]
    fn trait_object_compiles_for_downstream_wiring() {
        let rt: Box<dyn GemmaRuntime> = Box::new(StubGemmaRuntime::new());
        assert_eq!(rt.model_id(), "stub-no-model-loaded");
    }

    // -------- FsProbe-driven resolver tests --------

    struct FakeFsProbe {
        files: Mutex<Vec<PathBuf>>,
        env: Mutex<HashMap<String, String>>,
        exe_dir: Option<PathBuf>,
        manifest_dir: Option<PathBuf>,
        user_data_dir: Option<PathBuf>,
    }

    impl FakeFsProbe {
        fn new() -> Self {
            Self {
                files: Mutex::new(vec![]),
                env: Mutex::new(HashMap::new()),
                exe_dir: None,
                manifest_dir: None,
                user_data_dir: None,
            }
        }
        fn with_exe(mut self, p: &str) -> Self {
            self.exe_dir = Some(PathBuf::from(p));
            self
        }
        fn with_manifest(mut self, p: &str) -> Self {
            self.manifest_dir = Some(PathBuf::from(p));
            self
        }
        fn with_user(mut self, p: &str) -> Self {
            self.user_data_dir = Some(PathBuf::from(p));
            self
        }
        fn add_file(self, p: PathBuf) -> Self {
            self.files.lock().unwrap().push(p);
            self
        }
        fn set_env(self, k: &str, v: &str) -> Self {
            self.env
                .lock()
                .unwrap()
                .insert(k.to_string(), v.to_string());
            self
        }
    }

    impl FsProbe for FakeFsProbe {
        fn file_exists(&self, path: &Path) -> bool {
            self.files.lock().unwrap().iter().any(|p| p == path)
        }
        fn env_var(&self, name: &str) -> Option<String> {
            self.env.lock().unwrap().get(name).cloned()
        }
        fn current_exe_dir(&self) -> Option<PathBuf> {
            self.exe_dir.clone()
        }
        fn manifest_dir(&self) -> Option<PathBuf> {
            self.manifest_dir.clone()
        }
        fn user_data_dir(&self) -> Option<PathBuf> {
            self.user_data_dir.clone()
        }
    }

    fn bundled(dir: &str) -> PathBuf {
        PathBuf::from(dir)
            .join("resources")
            .join("models")
            .join(BUNDLED_MODEL_FILENAME)
    }

    #[test]
    fn resolver_returns_env_var_path_when_file_exists() {
        let env_path = PathBuf::from("/custom/path/my-model.gguf");
        let probe = FakeFsProbe::new()
            .set_env(MODEL_PATH_ENV_VAR, "/custom/path/my-model.gguf")
            .add_file(env_path.clone());
        let r = resolve_bundled_model_path(&probe).expect("ok");
        assert_eq!(r.source, "env");
        assert_eq!(r.path, env_path);
    }

    #[test]
    fn resolver_falls_through_env_when_file_missing() {
        let probe = FakeFsProbe::new()
            .set_env(MODEL_PATH_ENV_VAR, "/missing.gguf")
            .with_exe("/app")
            .add_file(bundled("/app"));
        let r = resolve_bundled_model_path(&probe).expect("ok");
        assert_eq!(r.source, "exe_dir");
        assert_eq!(r.path, bundled("/app"));
    }

    #[test]
    fn resolver_picks_exe_dir_over_manifest_and_user() {
        let probe = FakeFsProbe::new()
            .with_exe("/app")
            .with_manifest("/repo")
            .with_user("/home")
            .add_file(bundled("/app"))
            .add_file(bundled("/repo"))
            .add_file(PathBuf::from("/home/models").join(BUNDLED_MODEL_FILENAME));
        let r = resolve_bundled_model_path(&probe).unwrap();
        assert_eq!(r.source, "exe_dir");
    }

    #[test]
    fn resolver_falls_through_to_manifest_when_exe_dir_misses() {
        let probe = FakeFsProbe::new()
            .with_exe("/app")
            .with_manifest("/repo")
            .add_file(bundled("/repo"));
        let r = resolve_bundled_model_path(&probe).unwrap();
        assert_eq!(r.source, "manifest_dir");
    }

    #[test]
    fn resolver_falls_through_to_user_data_dir() {
        let probe = FakeFsProbe::new()
            .with_exe("/app")
            .with_manifest("/repo")
            .with_user("/home/forex-ai")
            .add_file(PathBuf::from("/home/forex-ai/models").join(BUNDLED_MODEL_FILENAME));
        let r = resolve_bundled_model_path(&probe).unwrap();
        assert_eq!(r.source, "user_data_dir");
    }

    #[test]
    fn resolver_returns_actionable_error_when_no_path_matches() {
        let probe = FakeFsProbe::new()
            .with_exe("/app")
            .with_manifest("/repo")
            .with_user("/home/forex-ai");
        let err = resolve_bundled_model_path(&probe).expect_err("must bail");
        let msg = err.to_string();
        assert!(msg.contains(BUNDLED_MODEL_FILENAME));
        assert!(msg.contains(MODEL_PATH_ENV_VAR));
        assert!(msg.contains("HauhauCS")); // download URL hint
        assert!(msg.contains("/app/resources/models"));
    }

    #[test]
    fn bundled_model_constants_pin_to_q4_k_m_default() {
        assert!(BUNDLED_MODEL_FILENAME.contains("Q4_K_M"));
        assert!(BUNDLED_MODEL_FILENAME.ends_with(".gguf"));
        assert!(BUNDLED_MODEL_DOWNLOAD_URL.starts_with("https://huggingface.co/"));
        assert!(BUNDLED_MODEL_DOWNLOAD_URL.contains("HauhauCS"));
    }
}
