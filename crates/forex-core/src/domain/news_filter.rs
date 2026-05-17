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

    /// `true` iff the filter is currently in BLACKOUT and orders must
    /// be rejected at the pre-trade gate. The check is intentionally
    /// strict — case-insensitive equality against the literal string
    /// `"BLACKOUT"` — because the upstream LLM prompt instructs the
    /// model to output exactly the word `BLACKOUT` and nothing else
    /// (see `poll_llm_news_sentiment` prompt body). A `SAFE` /
    /// `UNKNOWN` / `<empty>` value falls through and trading is
    /// allowed.
    ///
    /// Operator semantics: when the filter is `enabled=false`, this
    /// returns `false` regardless of `current_status` so a stale
    /// `BLACKOUT` leftover from a prior session cannot wedge the
    /// trader off the market after they disabled the feature.
    /// Audit gap #2 / roadmap §5.2 news-blackout pre-trade requirement.
    pub fn is_blackout(&self) -> bool {
        self.enabled && self.current_status.trim().eq_ignore_ascii_case("BLACKOUT")
    }

    /// Single-line status string for journalling and UI tooltips when
    /// the blackout gate rejects an order. Mirrors `is_blackout` so
    /// callers don't have to reach into the struct manually.
    pub fn blackout_reason(&self) -> String {
        format!(
            "news-blackout gate · status='{}' · provider={} · window=-{}/+{}min",
            self.current_status, self.llm_provider, self.blackout_minutes_before, self.blackout_minutes_after
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled_safe() -> NewsFilter {
        NewsFilter::new(true, 15, 10)
    }

    fn enabled_blackout() -> NewsFilter {
        let mut f = NewsFilter::new(true, 15, 10);
        f.current_status = "BLACKOUT".to_string();
        f
    }

    #[test]
    fn is_blackout_false_when_status_is_safe() {
        let f = enabled_safe();
        assert!(!f.is_blackout());
    }

    #[test]
    fn is_blackout_true_when_status_is_blackout_uppercase() {
        let f = enabled_blackout();
        assert!(f.is_blackout());
    }

    #[test]
    fn is_blackout_case_insensitive_on_status() {
        // Tolerate lowercase / mixed-case LLM responses — the prompt
        // asks for exactly "BLACKOUT" but we should not silently
        // green-light a "blackout" response.
        let mut f = enabled_safe();
        f.current_status = "blackout".to_string();
        assert!(f.is_blackout());
        f.current_status = "BlackOut".to_string();
        assert!(f.is_blackout());
    }

    #[test]
    fn is_blackout_trims_whitespace() {
        let mut f = enabled_safe();
        f.current_status = "  BLACKOUT  ".to_string();
        assert!(f.is_blackout());
        f.current_status = "\nBLACKOUT\n".to_string();
        assert!(f.is_blackout());
    }

    #[test]
    fn is_blackout_returns_false_when_filter_disabled_even_if_status_says_blackout() {
        // The disable kill-switch must beat a stale status — otherwise
        // a leftover BLACKOUT from a prior session keeps the operator
        // locked out forever.
        let mut f = enabled_blackout();
        f.enabled = false;
        assert!(!f.is_blackout(), "disabled filter must never block");
    }

    #[test]
    fn is_blackout_handles_unknown_and_empty_status_as_safe() {
        let mut f = enabled_safe();
        f.current_status = "UNKNOWN".to_string();
        assert!(!f.is_blackout());
        f.current_status = "".to_string();
        assert!(!f.is_blackout());
        f.current_status = "   ".to_string();
        assert!(!f.is_blackout());
    }

    #[test]
    fn blackout_reason_includes_status_provider_and_window() {
        let mut f = enabled_blackout();
        f.llm_provider = "openai".to_string();
        f.blackout_minutes_before = 30;
        f.blackout_minutes_after = 15;
        let reason = f.blackout_reason();
        assert!(reason.contains("BLACKOUT"), "reason must include status: {reason}");
        assert!(reason.contains("openai"), "reason must include provider: {reason}");
        assert!(reason.contains("-30"), "reason must include before-window: {reason}");
        assert!(reason.contains("+15"), "reason must include after-window: {reason}");
    }
}
