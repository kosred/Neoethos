//! Local Gemma-4 inference endpoints — backs the AI Helper chat
//! screen and the News-summary feature.
//!
//! Design decisions:
//!   - Lazy load. The first /gemma/chat call after process start
//!     spawns the inference worker thread (LlamaCppGemmaRuntime
//!     handles that). Cold startup eats 5–30 s reading the 3 GB
//!     GGUF off NVMe — we do it on demand so the binary boots
//!     instantly when the operator isn't using the LLM.
//!   - Single-flight. One physical model in memory, all chat
//!     requests serialise on a tokio Mutex. A second concurrent
//!     /chat call waits its turn — no crashes from concurrent
//!     llama_decode on the same KV cache.
//!   - spawn_blocking. The underlying `runtime.generate()` is
//!     synchronous and blocks the calling thread for the full
//!     length of the response. Wrapping in `spawn_blocking` keeps
//!     the tokio reactor free for the other axum routes.
//!   - Defensive. If the binary wasn't built with the
//!     `gemma-backend` feature, or the GGUF isn't on disk, every
//!     endpoint returns a 503 with an actionable instruction
//!     instead of pretending everything is fine.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use super::state::AppApiState;

// ─── GET /gemma/status ────────────────────────────────────────────────────

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GemmaStatusDto {
    /// Was the binary built with `--features gemma-backend` (i.e.
    /// llama-cpp-2 is compiled in)? If false, no inference is
    /// possible regardless of what's on disk.
    pub runtime_compiled_in: bool,
    /// Did the binary find a usable GGUF file on disk? The lookup
    /// honours `$NEOETHOS_GEMMA_MODEL_PATH` env override and falls
    /// back to a small list of canonical install dirs.
    pub model_file_present: bool,
    /// Whichever path the resolver matched first (or the empty
    /// string when it found nothing).
    pub resolved_path: String,
    /// Filename the resolver looks for by default — operators
    /// dropping in a different quant must use this name.
    pub expected_filename: String,
    /// HuggingFace URL the bundled installer fetches the default
    /// quant from. Surfaced in the UI so the user can re-download
    /// without leaving the app.
    pub download_url: String,
    /// On-disk size in bytes, or 0 if the file is missing.
    pub size_bytes: u64,
    /// Approximate expected size from the model card. Lets the UI
    /// detect "partial download" (e.g. a .tmp file left behind
    /// from a failed wget).
    pub expected_size_bytes: u64,
    /// Context-window cap currently configured. Operator can
    /// override via the NEOETHOS_GEMMA_N_CTX env var; default 32k.
    pub n_ctx: u32,
    /// Plain-language hint for the user when something is wrong.
    /// Empty when ready.
    pub message: String,
}

pub async fn status(State(_state): State<AppApiState>) -> Json<GemmaStatusDto> {
    let probe = neoethos_gemma::runtime::RealFsProbe;
    let resolved = neoethos_gemma::runtime::resolve_bundled_model_path(&probe);

    let (path_str, size_bytes, present) = match &resolved {
        Ok(r) => {
            let size = std::fs::metadata(&r.path).map(|m| m.len()).unwrap_or(0);
            (r.path.display().to_string(), size, true)
        }
        Err(_) => (String::new(), 0, false),
    };

    let runtime_compiled_in = cfg!(feature = "gemma-backend");
    let n_ctx = std::env::var("NEOETHOS_GEMMA_N_CTX")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(32_768);

    let message = if !runtime_compiled_in {
        "This build was compiled without the gemma-backend feature. \
         Rebuild with `cargo build -p neoethos-app --release --features gemma-backend`."
            .to_string()
    } else if !present {
        format!(
            "Gemma GGUF not found on disk. Download the default quant \
             from HuggingFace ({}) and place it at \
             resources/models/{} (or set NEOETHOS_GEMMA_MODEL_PATH to \
             a custom path).",
            neoethos_gemma::runtime::BUNDLED_MODEL_DOWNLOAD_URL,
            neoethos_gemma::runtime::BUNDLED_MODEL_FILENAME,
        )
    } else if size_bytes < neoethos_gemma::BUNDLED_MODEL_MIN_BYTES {
        // #196: stricter threshold (99% of expected) — the previous
        // 50% threshold accepted obviously-corrupted files. GGUF tail
        // bytes hold critical tensor data; losing the last 50 MB
        // still corrupts the model even if the first 5.3 GB are fine.
        format!(
            "GGUF file at {} is only {} bytes — looks like a partial \
             download. Expected ≥{} bytes. Delete the file and click \
             Download again.",
            path_str,
            size_bytes,
            neoethos_gemma::BUNDLED_MODEL_MIN_BYTES,
        )
    } else {
        String::new()
    };

    Json(GemmaStatusDto {
        runtime_compiled_in,
        // Treat "Ready" as: file exists AND not truncated. Anything
        // less and we surface the message so the UI shows "Download
        // Gemma" instead of a fake-Ready badge that leads to a chat
        // crash (#197).
        model_file_present: present && message.is_empty(),
        resolved_path: path_str,
        expected_filename: neoethos_gemma::runtime::BUNDLED_MODEL_FILENAME.to_string(),
        download_url: neoethos_gemma::runtime::BUNDLED_MODEL_DOWNLOAD_URL.to_string(),
        size_bytes,
        expected_size_bytes: neoethos_gemma::runtime::BUNDLED_MODEL_APPROX_BYTES,
        n_ctx,
        message,
    })
}

// ─── POST /gemma/chat ─────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
pub struct ChatBody {
    pub prompt: String,
    #[serde(rename = "maxTokens")]
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

// Only constructed inside the `#[cfg(feature = "gemma-backend")]`
// inference path. Without the feature, the chat/news handlers return
// 503 before they ever reach the point of building a DTO — the
// struct is unreachable code in that build, hence the gate.
#[cfg(feature = "gemma-backend")]
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatResponseDto {
    pub model_id: String,
    pub response: String,
    pub elapsed_ms: u64,
}

pub async fn chat(State(state): State<AppApiState>, Json(body): Json<ChatBody>) -> Response {
    if body.prompt.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "prompt must not be empty"})),
        )
            .into_response();
    }
    let max_tokens = body.max_tokens.unwrap_or(512).clamp(1, 4096);

    #[cfg(feature = "gemma-backend")]
    {
        chat_impl(state, body.prompt, max_tokens).await
    }
    #[cfg(not(feature = "gemma-backend"))]
    {
        // `state`, `body.prompt`, `max_tokens` are only consumed by
        // the feature-gated path above; suppress the unused-warning
        // here without touching the public API.
        let _ = (state, body.prompt, max_tokens);
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Gemma runtime not compiled in. Rebuild with \
                          `cargo build -p neoethos-app --release \
                          --features gemma-backend` and restart the server.",
            })),
        )
            .into_response()
    }
}

// ─── /gemma/news ──────────────────────────────────────────────────────────

#[derive(Debug, serde::Deserialize)]
pub struct NewsBody {
    pub symbol: String,
}

pub async fn news(State(state): State<AppApiState>, Json(body): Json<NewsBody>) -> Response {
    let symbol = body.symbol.trim().to_uppercase();
    if symbol.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "symbol must not be empty"})),
        )
            .into_response();
    }

    // We wrap the news request as a chat prompt so we can re-use the
    // same single-flight runtime. The system prompt nudges Gemma to
    // emit a short, factual summary instead of free-form opinions.
    let prompt = format!(
        "You are an expert forex news analyst. In 4–6 bullet points, \
         summarise the most recent and most impactful drivers for the \
         {symbol} currency pair. Stay factual — no recommendations, \
         no opinions. If you have no real recent data, say so explicitly."
    );

    #[cfg(feature = "gemma-backend")]
    {
        chat_impl(state, prompt, 600).await
    }
    #[cfg(not(feature = "gemma-backend"))]
    {
        let _ = (state, prompt);
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Gemma runtime not compiled in. News uses the \
                          same local inference path as Chat. Rebuild with \
                          `--features gemma-backend`.",
            })),
        )
            .into_response()
    }
}

// ─── inference path (only compiled with feature flag) ─────────────────────

/// Outcome of one full `run_chat_with_tools` round-trip — what the
/// HTTP handler turns into a `ChatResponseDto` and what the
/// background news-watcher (#128) consumes directly.
#[cfg(feature = "gemma-backend")]
#[derive(Debug, Clone)]
pub struct ChatOutcome {
    pub model_id: String,
    pub response: String,
    pub elapsed_ms: u64,
}

/// Inference + ReAct tool-loop entry point. Extracted from
/// `chat_impl` so the news-watcher task (#128) can drive the same
/// pipeline without going through HTTP. The HTTP handler is now a
/// thin wrapper around this.
///
/// Errors surface as `Err` rather than `Response` so callers
/// (HTTP / scheduler / future API) can map them however they want.
/// Specifically: 503 when the GGUF isn't on disk, 500 on inference
/// failure or task panic. The function loads the model on first
/// call (5-30s mmap) and reuses the loaded handle thereafter.
#[cfg(feature = "gemma-backend")]
pub async fn run_chat_with_tools(
    state: AppApiState,
    prompt: String,
    max_tokens: u32,
) -> anyhow::Result<ChatOutcome> {
    use crate::app_services::gemma_tools::{ToolRegistry, register_default_tools, run_tool_loop};
    use neoethos_gemma::runtime::{GemmaRuntime, LlamaCppGemmaRuntime};
    use std::sync::Arc;
    use std::time::Instant;
    use tokio::sync::Mutex;

    // Module-local handle to the loaded runtime. `OnceLock<Mutex<...>>`
    // would be cleaner but requires a stable model_id, so we use a
    // plain `OnceLock<Arc<...>>` plus a single-flight Mutex inside.
    use std::sync::OnceLock;
    static RUNTIME: OnceLock<Arc<Mutex<LlamaCppGemmaRuntime>>> = OnceLock::new();

    let runtime = if let Some(r) = RUNTIME.get() {
        r.clone()
    } else {
        let probe = neoethos_gemma::runtime::RealFsProbe;
        let resolved = neoethos_gemma::runtime::resolve_bundled_model_path(&probe)
            .map_err(|e| anyhow::anyhow!("Gemma model file not found: {e}"))?;
        let path = resolved.path;
        // #197: verify file integrity BEFORE handing the path to
        // llama-cpp. A truncated GGUF (interrupted download) triggers
        // C-side abort() inside llama-cpp's mmap path, which kills
        // the entire backend process — tokio's panic catcher can't
        // recover from a libc abort. Pre-validating the file size
        // and magic bytes here keeps the process alive and lets us
        // return a clean 503 with an actionable message.
        if let Err(reason) = neoethos_gemma::verify_gguf_file(
            &path,
            neoethos_gemma::BUNDLED_MODEL_MIN_BYTES,
        ) {
            tracing::warn!(
                target: "neoethos_app::server::gemma",
                path = %path.display(),
                error = %reason,
                "GGUF file failed integrity check — refusing to load"
            );
            return Err(anyhow::anyhow!(
                "Gemma model file is corrupted or incomplete: {reason}"
            ));
        }
        let loaded = tokio::task::spawn_blocking(move || LlamaCppGemmaRuntime::load(path))
            .await
            .map_err(|e| anyhow::anyhow!("Gemma load task panicked: {e}"))?
            .map_err(|e| anyhow::anyhow!("LlamaCppGemmaRuntime::load failed: {e}"))?;
        let arc = Arc::new(Mutex::new(loaded));
        let _ = RUNTIME.set(arc.clone());
        arc
    };

    let started = Instant::now();
    let runtime_clone = runtime.clone();
    let prompt_clone = prompt.clone();
    let state_clone = state.clone();
    let join = tokio::task::spawn_blocking(move || {
        let mut registry = ToolRegistry::new();
        register_default_tools(&mut registry);
        let rt = runtime_clone.blocking_lock();
        let model_id = rt.model_id().to_string();
        let final_text = run_tool_loop(&state_clone, &registry, &prompt_clone, |conv| {
            rt.generate(conv, max_tokens)
                .map_err(|e| anyhow::anyhow!("llama-cpp inference failed: {e}"))
        })?;
        Ok::<(String, String), anyhow::Error>((model_id, final_text))
    })
    .await
    .map_err(|e| anyhow::anyhow!("Gemma inference task panicked: {e}"))?;
    let (model_id, text) = join?;
    Ok(ChatOutcome {
        model_id,
        response: text,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

/// HTTP handler — thin wrapper around `run_chat_with_tools` that
/// maps the structured Result into an axum Response. Kept as a
/// separate function so unit tests can hit the underlying pipeline
/// without spinning up axum.
#[cfg(feature = "gemma-backend")]
async fn chat_impl(state: AppApiState, prompt: String, max_tokens: u32) -> Response {
    use neoethos_gemma::runtime::LlamaCppGemmaRuntime;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    // Same static — referenced here only so the `cfg` doesn't trip
    // an unused-import lint from the line below; the actual cache
    // sits inside `run_chat_with_tools`.
    use std::sync::OnceLock;
    static _RUNTIME_TYPE_ANCHOR: OnceLock<Arc<Mutex<LlamaCppGemmaRuntime>>> = OnceLock::new();
    let _ = &_RUNTIME_TYPE_ANCHOR;

    match run_chat_with_tools(state, prompt, max_tokens).await {
        Ok(outcome) => Json(ChatResponseDto {
            model_id: outcome.model_id,
            response: outcome.response,
            elapsed_ms: outcome.elapsed_ms,
        })
        .into_response(),
        Err(err) => {
            let message = err.to_string();
            let status = if message.contains("model file not found") {
                StatusCode::SERVICE_UNAVAILABLE
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
                Json(serde_json::json!({"error": message})),
            )
                .into_response()
        }
    }
}

