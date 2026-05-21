use anyhow::{Context, Result};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde_json::json;
use tracing::warn;

pub struct PerplexitySearcher {
    client: Client,
    api_key: SecretString,
}

impl PerplexitySearcher {
    /// Note — explicit `SecretString` constructor.
    ///
    /// The legacy `new()` (preserved below for back-compat with code that
    /// hasn't migrated) read `PERPLEXITY_API_KEY` from `std::env`, which
    /// silently activated the news searcher on any dev machine that
    /// happened to have the env var preset — bypassing the wizard's
    /// explicit opt-in for paid API keys. Mirror the
    /// `OpenAIScorer::new(SecretString, ...)` pattern: callers must pass
    /// the key explicitly. The wizard's NewsApi step is the canonical
    /// source.
    pub fn with_api_key(api_key: SecretString) -> Self {
        Self {
            client: Client::new(),
            api_key,
        }
    }

    /// **Deprecated** — reads `PERPLEXITY_API_KEY` from the environment.
    /// Prefer [`Self::with_api_key`] which requires the operator to
    /// supply the key explicitly (closes the silent-activation hole
    /// flagged by Note). Retained as a compatibility
    /// shim for code that hasn't been migrated yet; callers must
    /// understand that an env-preset key activates the searcher
    /// without UI confirmation.
    #[deprecated(
        since = "0.4.19",
        note = "prefer with_api_key(SecretString) — env reads silently activate paid APIs"
    )]
    pub fn new() -> Result<Self> {
        let api_key = SecretString::from(std::env::var("PERPLEXITY_API_KEY").unwrap_or_default());
        Ok(Self {
            client: Client::new(),
            api_key,
        })
    }

    pub async fn search_news(&self, symbol: &str) -> Result<String> {
        if self.api_key.expose_secret().is_empty() {
            warn!("Perplexity API key not configured (wizard NewsApi step), skipping search.");
            return Ok(String::new());
        }

        let prompt = format!(
            "Search for the latest, most impactful financial news regarding the forex pair {}. Provide a concise summary of the key drivers.",
            symbol
        );

        let body = json!({
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "You are an expert forex news aggregator."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        });

        let resp = self
            .client
            .post("https://api.perplexity.ai/chat/completions")
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json: serde_json::Value = resp.json().await?;
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .context("Invalid Perplexity response")?;

        Ok(content.to_string())
    }
}
