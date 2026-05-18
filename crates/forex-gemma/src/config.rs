//! Configuration for the forex-gemma helper.
//!
//! Phase G0. Carries `schema_version` from day one; same pattern
//! as `broker_credentials.toml` etc.

use forex_core::{HasSchemaVersion, SchemaVersion, default_v1};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const GEMMA_CONFIG_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1);

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GemmaQuantization {
    Q4_K_M,
    #[default]
    Q5_K_M,
    Q6_K,
    Q8_0,
    F16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchProvider {
    #[default]
    Tavily,
    Brave,
    Disabled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopicGateConfig {
    #[serde(default = "default_true")]
    pub jailbreak_regex_enabled: bool,
    #[serde(default = "default_true")]
    pub embedding_gate_enabled: bool,
    #[serde(default = "default_in_scope_threshold")]
    pub in_scope_threshold: f64,
    #[serde(default = "default_soft_warning_threshold")]
    pub soft_warning_threshold: f64,
    #[serde(default = "default_true")]
    pub post_filter_enabled: bool,
    #[serde(default = "default_true")]
    pub session_watchdog_enabled: bool,
}

fn default_true() -> bool {
    true
}
fn default_in_scope_threshold() -> f64 {
    -0.05
}
fn default_soft_warning_threshold() -> f64 {
    0.15
}

impl Default for TopicGateConfig {
    fn default() -> Self {
        Self {
            jailbreak_regex_enabled: true,
            embedding_gate_enabled: true,
            in_scope_threshold: default_in_scope_threshold(),
            soft_warning_threshold: default_soft_warning_threshold(),
            post_filter_enabled: true,
            session_watchdog_enabled: true,
        }
    }
}

/// Trade-suggestion (Role 2) gating. The tool layer ALWAYS routes
/// through Approve/Reject; these knobs control suggestion rate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TradingToolsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_suggestion_rate")]
    pub suggestion_rate_per_minute: u32,
    #[serde(default = "default_true")]
    pub per_trade_approval: bool,
    #[serde(default = "default_suggestion_timeout")]
    pub suggestion_timeout_seconds: u32,
}

fn default_suggestion_rate() -> u32 {
    1
}
fn default_suggestion_timeout() -> u32 {
    60
}

impl Default for TradingToolsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            suggestion_rate_per_minute: default_suggestion_rate(),
            per_trade_approval: true,
            suggestion_timeout_seconds: default_suggestion_timeout(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditLogConfig {
    #[serde(default)]
    pub store_full_text: bool,
    #[serde(default = "default_audit_max_mb")]
    pub max_size_mb: u32,
}

fn default_audit_max_mb() -> u32 {
    64
}

impl Default for AuditLogConfig {
    fn default() -> Self {
        Self {
            store_full_text: false,
            max_size_mb: default_audit_max_mb(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GemmaConfig {
    #[serde(default = "default_v1")]
    pub schema_version: SchemaVersion,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    #[serde(default)]
    pub quantization: GemmaQuantization,
    #[serde(default)]
    pub topic_gate: TopicGateConfig,
    #[serde(default)]
    pub trading_tools: TradingToolsConfig,
    #[serde(default)]
    pub search_provider: SearchProvider,
    #[serde(default)]
    pub search_api_key_persisted: bool,
    #[serde(default)]
    pub audit: AuditLogConfig,
}

impl Default for GemmaConfig {
    fn default() -> Self {
        Self {
            schema_version: GEMMA_CONFIG_SCHEMA_VERSION,
            enabled: false,
            model_path: None,
            quantization: GemmaQuantization::default(),
            topic_gate: TopicGateConfig::default(),
            trading_tools: TradingToolsConfig::default(),
            search_provider: SearchProvider::default(),
            search_api_key_persisted: false,
            audit: AuditLogConfig::default(),
        }
    }
}

impl HasSchemaVersion for GemmaConfig {
    const CURRENT: SchemaVersion = GEMMA_CONFIG_SCHEMA_VERSION;
    fn schema_version(&self) -> SchemaVersion {
        self.schema_version
    }
}

impl GemmaConfig {
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        let TopicGateConfig {
            in_scope_threshold,
            soft_warning_threshold,
            ..
        } = self.topic_gate;
        if in_scope_threshold > soft_warning_threshold {
            warnings.push(format!(
                "topic_gate.in_scope_threshold ({in_scope_threshold}) > soft_warning_threshold ({soft_warning_threshold})"
            ));
        }
        if self.trading_tools.enabled && !self.enabled {
            warnings.push("trading_tools.enabled but master switch off".to_string());
        }
        if self.trading_tools.enabled && !self.trading_tools.per_trade_approval {
            warnings.push(
                "trading_tools.per_trade_approval is false; mandatory by operator directive"
                    .to_string(),
            );
        }
        if self.trading_tools.enabled && self.trading_tools.suggestion_rate_per_minute == 0 {
            warnings.push("trading_tools.suggestion_rate_per_minute is 0".to_string());
        }
        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_safe_off() {
        let c = GemmaConfig::default();
        assert!(!c.enabled);
        assert!(!c.trading_tools.enabled);
        assert!(c.trading_tools.per_trade_approval);
        assert_eq!(c.trading_tools.suggestion_rate_per_minute, 1);
        assert_eq!(c.trading_tools.suggestion_timeout_seconds, 60);
        assert!(!c.audit.store_full_text);
    }

    #[test]
    fn schema_version_defaults_to_v1_for_pre_versioning_files() {
        let json = r#"{ "enabled": false }"#;
        let parsed: GemmaConfig = serde_json::from_str(json).expect("parse");
        assert_eq!(parsed.schema_version, SchemaVersion::new(1));
    }

    #[test]
    fn schema_version_round_trips_through_serde() {
        let c = GemmaConfig::default();
        let s = serde_json::to_string(&c).unwrap();
        let back: GemmaConfig = serde_json::from_str(&s).unwrap();
        assert_eq!(back.schema_version, GEMMA_CONFIG_SCHEMA_VERSION);
        assert_eq!(back, c);
    }

    #[test]
    fn validate_flags_inverted_topic_gate_thresholds() {
        let mut c = GemmaConfig::default();
        c.topic_gate.in_scope_threshold = 0.5;
        c.topic_gate.soft_warning_threshold = 0.0;
        assert!(
            c.validate()
                .iter()
                .any(|w| w.contains("in_scope_threshold"))
        );
    }

    #[test]
    fn validate_flags_trading_enabled_without_master_switch() {
        let mut c = GemmaConfig::default();
        c.trading_tools.enabled = true;
        assert!(c.validate().iter().any(|w| w.contains("master switch")));
    }

    #[test]
    fn validate_flags_per_trade_approval_disabled() {
        let mut c = GemmaConfig::default();
        c.enabled = true;
        c.trading_tools.enabled = true;
        c.trading_tools.per_trade_approval = false;
        assert!(
            c.validate()
                .iter()
                .any(|w| w.contains("per_trade_approval"))
        );
    }

    #[test]
    fn validate_flags_zero_suggestion_rate() {
        let mut c = GemmaConfig::default();
        c.enabled = true;
        c.trading_tools.enabled = true;
        c.trading_tools.suggestion_rate_per_minute = 0;
        assert!(
            c.validate()
                .iter()
                .any(|w| w.contains("suggestion_rate_per_minute"))
        );
    }

    #[test]
    fn validate_passes_on_clean_default() {
        assert!(GemmaConfig::default().validate().is_empty());
    }

    #[test]
    fn has_schema_version_trait_returns_struct_field() {
        assert_eq!(
            GemmaConfig::default().schema_version(),
            GEMMA_CONFIG_SCHEMA_VERSION
        );
    }

    #[test]
    fn quantization_default_is_q5_km() {
        assert_eq!(GemmaQuantization::default(), GemmaQuantization::Q5_K_M);
    }

    #[test]
    fn search_provider_default_is_tavily() {
        assert_eq!(SearchProvider::default(), SearchProvider::Tavily);
    }
}
