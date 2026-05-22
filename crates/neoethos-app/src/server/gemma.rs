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
            let size = std::fs::metadata(&r.path)
                .map(|m| m.len())
                .unwrap_or(0);
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
    } else if size_bytes < neoethos_gemma::runtime::BUNDLED_MODEL_APPROX_BYTES / 2 {
        format!(
            "GGUF file at {} is only {} bytes — looks like a partial \
             download. Expected ≥{} bytes.",
            path_str,
            size_bytes,
            neoethos_gemma::runtime::BUNDLED_MODEL_APPROX_BYTES / 2,
        )
    } else {
        String::new()
    };

    Json(GemmaStatusDto {
        runtime_compiled_in,
        model_file_present: present && !message.contains("partial"),
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

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatResponseDto {
    pub model_id: String,
    pub response: String,
    pub elapsed_ms: u64,
}

pub async fn chat(
    State(state): State<AppApiState>,
    Json(body): Json<ChatBody>,
) -> Response {
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
        let _ = state;
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

pub async fn news(
    State(state): State<AppApiState>,
    Json(body): Json<NewsBody>,
) -> Response {
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
        let _ = state;
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

#[cfg(feature = "gemma-backend")]
async fn chat_impl(state: AppApiState, prompt: String, max_tokens: u32) -> Response {
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
        // Resolve + load on first call. spawn_blocking because the
        // mmap + LlamaBackend init takes seconds.
        let _state = state;
        let probe = neoethos_gemma::runtime::RealFsProbe;
        let resolved =
            match neoethos_gemma::runtime::resolve_bundled_model_path(&probe) {
                Ok(r) => r,
                Err(err) => {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(serde_json::json!({
                            "error": format!("Gemma model file not found: {err}"),
                        })),
                    )
                        .into_response();
                }
            };
        let path = resolved.path;
        let loaded = tokio::task::spawn_blocking(move || {
            LlamaCppGemmaRuntime::load(path)
        })
        .await;
        match loaded {
            Ok(Ok(rt)) => {
                let arc = Arc::new(Mutex::new(rt));
                let _ = RUNTIME.set(arc.clone());
                arc
            }
            Ok(Err(err)) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("LlamaCppGemmaRuntime::load failed: {err}"),
                    })),
                )
                    .into_response();
            }
            Err(join_err) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": format!("Gemma load task panicked: {join_err}"),
                    })),
                )
                    .into_response();
            }
        }
    };

    let started = Instant::now();
    let runtime_clone = runtime.clone();
    let prompt_clone = prompt.clone();
    let inference_result = tokio::task::spawn_blocking(move || {
        // Lock the runtime so only one inference at a time hits the
        // worker thread. We use blocking_lock here because this whole
        // closure already runs on a blocking thread, never the
        // reactor.
        let rt = runtime_clone.blocking_lock();
        rt.generate(&prompt_clone, max_tokens)
            .map(|text| (rt.model_id().to_string(), text))
    })
    .await;

    match inference_result {
        Ok(Ok((model_id, text))) => Json(ChatResponseDto {
            model_id,
            response: text,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
        .into_response(),
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Gemma inference failed: {err}"),
            })),
        )
            .into_response(),
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Gemma inference task panicked: {join_err}"),
            })),
        )
            .into_response(),
    }
}
