use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;
use std::env;
use tracing::warn;

pub struct OpenAIScorer {
    client: Client,
    api_key: String,
}

impl OpenAIScorer {
    pub fn new() -> Result<Self> {
        // DOCUMENTED-DEFAULT: an empty key disables this scorer (see
        // `analyze_sentiment`, which short-circuits with a warn). We
        // deliberately do not fail construction so the rest of the news
        // pipeline can run without an OPENAI_API_KEY.
        let api_key = env::var("OPENAI_API_KEY").unwrap_or_default();
        Ok(Self {
            client: Client::new(),
            api_key,
        })
    }

    pub async fn analyze_sentiment(&self, text: &str) -> Result<f64> {
        if self.api_key.is_empty() {
            warn!("OPENAI_API_KEY not set, skipping sentiment analysis.");
            return Ok(0.0);
        }

        let prompt = format!(
            "Analyze the sentiment of the following financial news. Return strictly a single float between -1.0 (very negative) and 1.0 (very positive).\n\nText: {}",
            text
        );

        let body = json!({
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a financial news sentiment analyzer."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        });

        let resp = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
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
                    "OpenAI sentiment scorer: response was not parseable as f64; returning 0.0"
                );
                0.0
            }
        };
        Ok(score.clamp(-1.0, 1.0))
    }
}
