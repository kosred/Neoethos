//! OpenAI-backed news sentiment scorer.
//!
//! ## Operator directive 2026-05-17 — user-provided API key only
//!
//! Earlier builds of this scorer read `OPENAI_API_KEY` directly from
//! the process environment in `OpenAIScorer::new`. That was a UX bug
//! and a quiet-credential-leak hazard: on a developer machine whose
//! shell already exports the dev's personal key, the bot would
//! silently bill that key with zero operator awareness, and the
//! distributed binary had no in-app surface for an end user to
//! supply their OWN key. The wizard's `news_api.rs` module held a
//! `NewsApiKeyHolder` for exactly this purpose but it wasn't wired
//! through.
//!
//! The fix in this module is intentionally narrow: the constructor
//! takes the key as an explicit argument. The caller (settings UI,
//! wizard, or news-pipeline bootstrap) is responsible for sourcing
//! the key from the operator's wizard input / settings panel and
//! handing it in. An empty/None key disables the scorer; the
//! sentiment call then short-circuits with a warn and returns `0.0`.
//!
//! There is intentionally NO `env::var` fallback. The wider news
//! pipeline composes the scorer and is the place to decide what to
//! show the operator when no key is configured ("LLM news scoring
//! DISABLED — provide a key in Settings → News").
//!
//! Test backends in other crates should construct the scorer with an
//! empty key when they want the disable-and-return-zero behaviour.

use anyhow::{Context, Result};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde_json::json;
use tracing::warn;

/// Sentinel value returned by [`OpenAIScorer::analyze_sentiment`]
/// when the scorer is disabled (empty key) or the model response is
/// not parseable as a float.
pub const NEUTRAL_SENTIMENT_SCORE: f64 = 0.0;

/// OpenAI Chat Completions endpoint. Centralised so swapping it for
/// the Azure-OpenAI variant in v0.5 is a single-line change.
pub const OPENAI_CHAT_COMPLETIONS_ENDPOINT: &str = "https://api.openai.com/v1/chat/completions";

/// Default model. The operator can override via the
/// `openai_model` config field; this default is the cost-optimised
/// shorter-context variant used in `gpt-4o-mini`-class workloads.
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";

/// News sentiment scorer backed by OpenAI Chat Completions.
///
/// Constructed with an explicit `SecretString` so the wrapping type
/// guarantees the key never lands in a `Debug` impl, structured-log
/// payload, or panic message. The `secrecy::SecretString` destructor
/// zeroes the buffer on drop.
pub struct OpenAIScorer {
    client: Client,
    api_key: Option<SecretString>,
    model: String,
}

impl OpenAIScorer {
    /// Build a scorer with an explicit (operator-provided) API key
    /// and model. Pass `None` for `api_key` to construct a disabled
    /// scorer — the sentiment call will warn and return
    /// [`NEUTRAL_SENTIMENT_SCORE`].
    ///
    /// The `model` argument is taken from the operator's config
    /// (`config.yaml::news::openai_model` or the wizard's News step)
    /// so an upgrade to a new model is a config flip rather than a
    /// code change. Empty string defaults to [`DEFAULT_OPENAI_MODEL`].
    pub fn new(api_key: Option<SecretString>, model: impl Into<String>) -> Result<Self> {
        let mut model = model.into();
        if model.trim().is_empty() {
            model = DEFAULT_OPENAI_MODEL.to_string();
        }
        Ok(Self {
            client: Client::new(),
            api_key,
            model,
        })
    }

    /// Convenience constructor: disabled scorer, default model.
    /// Used by the news pipeline when the operator has not yet
    /// provided a key. The scorer still functions — every
    /// `analyze_sentiment` call short-circuits with a warn and
    /// returns the neutral score.
    pub fn disabled() -> Self {
        Self {
            client: Client::new(),
            api_key: None,
            model: DEFAULT_OPENAI_MODEL.to_string(),
        }
    }

    /// `true` iff a non-empty API key was supplied at construction.
    /// The settings UI and the wizard's News step consult this to
    /// render the "LLM news scoring DISABLED — provide a key in
    /// Settings" banner without having to expose the secret.
    pub fn is_enabled(&self) -> bool {
        self.api_key
            .as_ref()
            .map(|s| !s.expose_secret().is_empty())
            .unwrap_or(false)
    }

    /// Score the supplied text on `[-1.0, 1.0]` (very negative → very
    /// positive). Returns [`NEUTRAL_SENTIMENT_SCORE`] when the scorer
    /// is disabled or the model response cannot be parsed as a float.
    pub async fn analyze_sentiment(&self, text: &str) -> Result<f64> {
        let Some(api_key) = self.api_key.as_ref() else {
            warn!(
                target: "forex_news::openai",
                "OpenAI scorer disabled (no API key supplied); returning neutral score"
            );
            return Ok(NEUTRAL_SENTIMENT_SCORE);
        };
        let api_key = api_key.expose_secret();
        if api_key.is_empty() {
            warn!(
                target: "forex_news::openai",
                "OpenAI scorer constructed with empty key; returning neutral score"
            );
            return Ok(NEUTRAL_SENTIMENT_SCORE);
        }

        let prompt = format!(
            "Analyze the sentiment of the following financial news. Return strictly a single float between -1.0 (very negative) and 1.0 (very positive).\n\nText: {}",
            text
        );

        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a financial news sentiment analyzer."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        });

        let resp = self
            .client
            .post(OPENAI_CHAT_COMPLETIONS_ENDPOINT)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json: serde_json::Value = resp.json().await?;
        let content = json["choices"][0]["message"]["content"]
            .as_str()
            .context("Invalid OpenAI response")?;

        let trimmed = content.trim();
        let score: f64 = match trimmed.parse::<f64>() {
            Ok(v) => v,
            Err(err) => {
                // Surface model-response drift: a non-numeric reply means the
                // prompt contract slipped (model upgrade, content filter, etc.)
                // and the caller deserves to know rather than getting silent 0.0.
                warn!(
                    target: "forex_news::openai",
                    response = %trimmed,
                    error = %err,
                    "OpenAI sentiment scorer: response was not parseable as f64; returning neutral score"
                );
                NEUTRAL_SENTIMENT_SCORE
            }
        };
        Ok(score.clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_constructor_yields_disabled_scorer() {
        let s = OpenAIScorer::disabled();
        assert!(!s.is_enabled());
    }

    #[test]
    fn new_with_none_key_yields_disabled_scorer() {
        let s = OpenAIScorer::new(None, "").expect("constructor must accept None");
        assert!(!s.is_enabled());
    }

    #[test]
    fn new_with_empty_key_yields_disabled_scorer() {
        let s = OpenAIScorer::new(Some(SecretString::from("")), "")
            .expect("constructor must accept empty key");
        assert!(!s.is_enabled());
    }

    #[test]
    fn new_with_non_empty_key_yields_enabled_scorer() {
        let s = OpenAIScorer::new(Some(SecretString::from("sk-test")), "")
            .expect("constructor must accept a key");
        assert!(s.is_enabled());
    }

    #[test]
    fn empty_model_falls_back_to_default() {
        let s = OpenAIScorer::new(Some(SecretString::from("sk-test")), "").expect("constructor");
        assert_eq!(s.model, DEFAULT_OPENAI_MODEL);
    }

    #[test]
    fn custom_model_is_preserved() {
        let s =
            OpenAIScorer::new(Some(SecretString::from("sk-test")), "gpt-4o").expect("constructor");
        assert_eq!(s.model, "gpt-4o");
    }

    #[tokio::test]
    async fn disabled_scorer_returns_neutral_score_without_network() {
        let s = OpenAIScorer::disabled();
        let score = s
            .analyze_sentiment("anything")
            .await
            .expect("must not error");
        assert!((score - NEUTRAL_SENTIMENT_SCORE).abs() < f64::EPSILON);
    }
}
