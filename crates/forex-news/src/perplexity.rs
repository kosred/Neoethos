use anyhow::{Context, Result};
use reqwest::Client;
use serde_json::json;
use std::env;
use tracing::warn;

pub struct PerplexitySearcher {
    client: Client,
    api_key: String,
}

impl PerplexitySearcher {
    pub fn new() -> Result<Self> {
        let api_key = env::var("PERPLEXITY_API_KEY").unwrap_or_default();
        Ok(Self {
            client: Client::new(),
            api_key,
        })
    }

    pub async fn search_news(&self, symbol: &str) -> Result<String> {
        if self.api_key.is_empty() {
            warn!("PERPLEXITY_API_KEY not set, skipping search.");
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

        let resp = self.client
            .post("https://api.perplexity.ai/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
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
