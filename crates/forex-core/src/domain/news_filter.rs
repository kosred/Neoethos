use secrecy::{ExposeSecret, SecretString};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct NewsEvent {
    pub currency: String,
    pub impact: String,
    pub timestamp_ms: i64,
}

#[derive(Debug, Clone)]
pub struct NewsFilter {
    pub enabled: bool,
    /// SECURITY: the LLM API key is wrapped in [`secrecy::SecretString`]
    /// (a `SecretBox<str>` — a `Box<str>` that cannot reallocate, so the
    /// secret bytes have a single fixed location in memory which is
    /// zeroized on drop). The `Debug` impl masks the value as `[REDACTED]`
    /// and `serde::Serialize` is opt-in, so accidental logging or
    /// serialization cannot exfiltrate the operator's OpenAI/Perplexity
    /// bearer token. Access the underlying string via [`ExposeSecret`].
    ///
    /// The zeroize upstream docs explicitly recommend `secrecy` over
    /// `Zeroizing<String>` for string-shaped secrets because `String`'s
    /// realloc-on-push can leave un-zeroed copies of the secret on the
    /// heap.
    pub api_key: Option<SecretString>,
    pub llm_provider: String, // "openai" or "perplexity"
    pub blackout_minutes_before: i64,
    pub blackout_minutes_after: i64,
    pub current_status: String, // "SAFE" or "BLACKOUT"
    pub recent_events: Vec<NewsEvent>,
}

impl NewsFilter {
    pub fn new(enabled: bool, before: i64, after: i64) -> Self {
        Self {
            enabled,
            api_key: None,
            llm_provider: "perplexity".to_string(),
            blackout_minutes_before: before,
            blackout_minutes_after: after,
            current_status: "SAFE".to_string(),
            recent_events: Vec::new(),
        }
    }

    pub fn set_credentials(&mut self, provider: String, api_key: String) {
        self.llm_provider = provider;
        // Wrap into a `SecretString` (`SecretBox<str>`): the previous
        // value (if any) is dropped, which zeroes the underlying buffer.
        self.api_key = Some(SecretString::from(api_key));
    }

    /// Run synchronously (should be spawned in a dedicated blocking thread by the app)
    pub fn poll_llm_news_sentiment(
        &mut self,
        currency_pair: &str,
    ) -> Result<String, anyhow::Error> {
        if !self.enabled {
            return Ok("SAFE".to_string());
        }

        // `expose_secret()` is the deliberate API that surfaces the
        // secret only at the point of use; every call-site is grep-able.
        let api_key: &str = match self.api_key.as_ref() {
            Some(s) => {
                let revealed: &str = s.expose_secret();
                if revealed.trim().is_empty() {
                    return Ok("SAFE".to_string());
                }
                revealed
            }
            None => return Ok("SAFE".to_string()),
        };

        let prompt = format!(
            "You are an expert forex macroeconomic evaluator. Analyze real-time breaking news for {}. If there is a massive macroeconomic blackout event (e.g. NFP, CPI, Central Bank Rate Decision) happening Right Now or within the next 15 minutes, output EXACTLY the word \"BLACKOUT\" and nothing else. Otherwise, output EXACTLY \"SAFE\".",
            currency_pair
        );

        let client = reqwest::blocking::Client::new();

        let (endpoint, model) = if self.llm_provider == "openai" {
            ("https://api.openai.com/v1/chat/completions", "gpt-4o-mini")
        } else {
            ("https://api.perplexity.ai/chat/completions", "sonar-pro")
        };

        let body = serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": "Obey instructions strictly."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        });

        let res = client
            .post(endpoint)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .map_err(|e| anyhow::anyhow!("LLM HTTP Request Failed: {}", e))?;

        if res.status().is_success() {
            let json: Value = res.json()?;
            if let Some(content) = json["choices"][0]["message"]["content"].as_str() {
                let status = if content.to_uppercase().contains("BLACKOUT") {
                    "BLACKOUT"
                } else {
                    "SAFE"
                };
                self.current_status = status.to_string();
                return Ok(status.to_string());
            }
        } else {
            let status = res.status();
            let text = res.text().unwrap_or_default();
            return Err(anyhow::anyhow!("LLM API returned {}: {}", status, text));
        }

        Ok("SAFE".to_string())
    }

    pub fn is_blackout_active(&self, _currency_pair: &str, _current_timestamp_ms: i64) -> bool {
        if !self.enabled {
            return false;
        }
        self.current_status == "BLACKOUT"
    }
}
