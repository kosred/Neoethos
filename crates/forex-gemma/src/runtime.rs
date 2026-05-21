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

// ---------------------------------------------------------------------------
// G1 — real llama.cpp-backed runtime (feature = "mistralrs-runtime")
// ---------------------------------------------------------------------------
//
// `LlamaCppGemmaRuntime` loads the GGUF from disk once and processes every
// `generate()` call on a dedicated background thread so the egui render loop
// is never blocked. All llama.cpp objects (`LlamaBackend`, `LlamaModel`,
// `LlamaContext`) live on that thread, which avoids the self-referential
// lifetime problem that arises when you try to store `LlamaModel<'backend>`
// inside the same struct that owns `LlamaBackend`.
//
// Architecture:
//   ┌─ UI / render thread ─────────────────────┐
//   │  LlamaCppGemmaRuntime::generate(prompt)  │
//   │    → send(InferRequest) to worker         │
//   │    → recv(Result<String>) — BLOCKS HERE   │  ← max_tokens caps latency
//   └──────────────────────────────────────────┘
//            │ SyncChannel(4)
//   ┌─ "forex-gemma-llama" thread ─────────────┐
//   │  LlamaBackend (one per process via init)  │
//   │  LlamaModel   (loaded once from GGUF)     │
//   │  loop { new_context + decode + sample }   │
//   └──────────────────────────────────────────┘
//
// The UI should wrap `generate()` in `std::thread::spawn` (see ai_helper.rs)
// to keep egui frames responsive while inference runs.

#[cfg(feature = "mistralrs-runtime")]
mod llama_impl {
    use super::GemmaRuntime;
    use anyhow::{Context, Result, bail};
    use llama_cpp_2::{
        context::params::LlamaContextParams,
        llama_backend::LlamaBackend,
        llama_batch::LlamaBatch,
        model::{AddBos, LlamaModel, params::LlamaModelParams},
        sampling::LlamaSampler,
    };
    use std::{
        num::NonZeroU32,
        path::PathBuf,
        sync::mpsc::{self, SyncSender},
    };

    /// Per-request message from the UI thread to the inference worker.
    struct InferRequest {
        prompt: String,
        max_tokens: u32,
        reply_tx: mpsc::SyncSender<Result<String>>,
    }

    /// G1 real inference runtime. Construct once per model file; the
    /// background worker thread is spawned in `load()` and lives until
    /// the `LlamaCppGemmaRuntime` is dropped (at which point the channel
    /// is closed and the thread exits gracefully).
    pub struct LlamaCppGemmaRuntime {
        tx: SyncSender<InferRequest>,
        model_id: String,
    }

    impl LlamaCppGemmaRuntime {
        /// Load the model at `model_path` and start the inference worker
        /// thread. Returns an error if the GGUF cannot be read or if the
        /// background thread fails to spawn.
        ///
        /// Loading a 4B Q4_K_M model takes ~5–30 s depending on NVMe vs.
        /// spinning-disk and the amount of CPU the OS dedicates to mmaping
        /// the file. Call this off the egui render thread.
        pub fn load(model_path: PathBuf) -> Result<Self> {
            let model_id = model_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("gemma-gguf")
                .to_string();

            // Bounded queue: 4 pending requests max. A second chat message
            // while inference is running will block the sender (UI thread)
            // but with the async dispatch in ai_helper.rs the sender IS a
            // background thread, so no deadlock.
            let (tx, rx) = mpsc::sync_channel::<InferRequest>(4);

            std::thread::Builder::new()
                .name("forex-gemma-llama".to_string())
                .spawn(move || inference_worker(model_path, rx))
                .context("failed to spawn forex-gemma inference thread")?;

            Ok(Self { tx, model_id })
        }
    }

    impl GemmaRuntime for LlamaCppGemmaRuntime {
        fn generate(&self, prompt: &str, max_tokens: u32) -> Result<String> {
            let (reply_tx, reply_rx) = mpsc::sync_channel(1);
            self.tx
                .send(InferRequest {
                    prompt: prompt.to_string(),
                    max_tokens,
                    reply_tx,
                })
                .context("inference worker thread has stopped")?;
            reply_rx
                .recv()
                .context("inference worker did not reply (thread crashed?)")?
        }

        fn model_id(&self) -> &str {
            &self.model_id
        }
    }

    // ── Inference worker ────────────────────────────────────────────────────

    /// Owns all llama.cpp objects. Runs until the `Receiver` is disconnected
    /// (i.e. until `LlamaCppGemmaRuntime` is dropped).
    fn inference_worker(model_path: PathBuf, rx: mpsc::Receiver<InferRequest>) {
        // One LlamaBackend per process is the intended usage. Using
        // `init()` instead of `init_numa` — numa pinning is optional.
        let backend = match LlamaBackend::init() {
            Ok(b) => b,
            Err(e) => {
                tracing::error!(
                    target: "forex_gemma::runtime",
                    error = %e,
                    "LlamaBackend::init failed; inference worker exiting"
                );
                return;
            }
        };

        let model_params = LlamaModelParams::default();
        let model = match LlamaModel::load_from_file(&backend, &model_path, &model_params) {
            Ok(m) => {
                tracing::info!(
                    target: "forex_gemma::runtime",
                    path = %model_path.display(),
                    "Gemma GGUF loaded"
                );
                m
            }
            Err(e) => {
                tracing::error!(
                    target: "forex_gemma::runtime",
                    path = %model_path.display(),
                    error = %e,
                    "LlamaModel::load_from_file failed; inference worker exiting"
                );
                return;
            }
        };

        // Process requests sequentially. One request at a time — this
        // keeps memory usage bounded (one KV-cache worth of RAM).
        for req in rx {
            let result = run_single_inference(&backend, &model, &req.prompt, req.max_tokens);
            // Ignore send errors — the UI may have timed out and dropped the receiver.
            let _ = req.reply_tx.send(result);
        }

        tracing::debug!(
            target: "forex_gemma::runtime",
            "forex-gemma inference worker: channel closed, exiting"
        );
    }

    /// Run one inference request end-to-end.
    fn run_single_inference(
        backend: &LlamaBackend,
        model: &LlamaModel,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<String> {
        // A new context per request gives a clean KV-cache. In G2 we will
        // keep the context alive and call `llama_kv_cache_clear()` between
        // turns for lower per-request latency.
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(4096));

        let mut ctx = model
            .new_context(backend, ctx_params)
            .context("failed to create LlamaContext")?;

        // Tokenize the full prompt.
        let tokens = model
            .str_to_token(prompt, AddBos::Always)
            .context("failed to tokenize prompt")?;

        if tokens.is_empty() {
            bail!("tokenizer produced zero tokens for the given prompt");
        }

        // Load the prompt into the context in a single batch.
        // Only the LAST token needs logits for the first sampling step.
        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch
                .add(token, i as i32, &[0], is_last)
                .context("batch.add (prompt) failed")?;
        }
        ctx.decode(&mut batch).context("initial prompt decode failed")?;

        // Sampler chain: temperature → top-k → nucleus → random dist.
        // Greedy (`LlamaSampler::greedy()`) is a valid alternative for
        // deterministic responses but tends to repeat itself on longer
        // outputs.
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.7),
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::dist(42),
        ]);

        // Stateful UTF-8 decoder handles tokens that span byte boundaries
        // (e.g. multi-byte CJK/emoji split across two llama.cpp pieces).
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut n_cur = batch.n_tokens();
        let mut output = String::new();

        for _ in 0..max_tokens {
            // Sample the next token from the last position in the context.
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            // End-of-generation: <eos> / <eot> / any model-specific stop token.
            if model.is_eog_token(token) {
                break;
            }

            sampler.accept(token);

            // Convert the token id back to its text fragment.
            let piece = model
                .token_to_piece(token, &mut decoder, true, None)
                .context("token_to_piece failed")?;
            output.push_str(&piece);

            // Roll the context one step forward.
            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .context("batch.add (generation) failed")?;
            n_cur += 1;
            ctx.decode(&mut batch)
                .context("generation step decode failed")?;
        }

        Ok(output)
    }
}

#[cfg(feature = "mistralrs-runtime")]
pub use llama_impl::LlamaCppGemmaRuntime;

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

    // G1 compile-gate test — verifies that `LlamaCppGemmaRuntime` satisfies
    // `GemmaRuntime: Send + Sync` (a boxed trait object in Arc) without
    // requiring an actual model file on disk. The `load` call will fail
    // (no real GGUF) but the type must _compile_ and the channel + thread
    // setup is validated by the type checker. Run with:
    //   cargo test -p forex-gemma --features mistralrs-runtime
    #[cfg(feature = "mistralrs-runtime")]
    #[test]
    fn llama_cpp_runtime_satisfies_gemma_runtime_trait_bounds() {
        use std::sync::Arc;
        // Attempting to construct with a non-existent path — we expect
        // this to return an error (model not found) OR spawn a thread
        // that fails gracefully. Either way the TYPE is what we're testing.
        let result = LlamaCppGemmaRuntime::load(PathBuf::from("/nonexistent/model.gguf"));
        match result {
            Ok(rt) => {
                // Runtime constructed — it's on an inference thread;
                // type-check that it fits behind `Arc<dyn GemmaRuntime>`.
                let _: Arc<dyn GemmaRuntime> = Arc::new(rt);
            }
            Err(_) => {
                // Expected when cmake or llama-cpp build step hasn't run yet
                // (CI without C++ toolchain). The compile-gate test still
                // passes — the type system was checked by rustc.
            }
        }
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
        // PathBuf display uses the platform's native separator.
        // On Windows the rendered path is `/app\resources\models\…`,
        // on POSIX it is `/app/resources/models/…`. Build the expected
        // fragment with the same separator the platform actually uses
        // so this test passes on every CI matrix slot.
        let sep = std::path::MAIN_SEPARATOR_STR;
        let expected = format!("/app{sep}resources{sep}models");
        assert!(
            msg.contains(&expected),
            "error message did not mention the exe_dir path. expected to contain `{expected}`, got: {msg}"
        );
    }

    #[test]
    fn bundled_model_constants_pin_to_q4_k_m_default() {
        assert!(BUNDLED_MODEL_FILENAME.contains("Q4_K_M"));
        assert!(BUNDLED_MODEL_FILENAME.ends_with(".gguf"));
        assert!(BUNDLED_MODEL_DOWNLOAD_URL.starts_with("https://huggingface.co/"));
        assert!(BUNDLED_MODEL_DOWNLOAD_URL.contains("HauhauCS"));
    }
}
