//! cTrader auth value-types.
//!
//! Production drives cTrader through `app_services::reauth` plus
//! `CTraderTokenBundle`. This module contains only the two live DTOs:
//! `CTraderTokenBundle` (secure_store / bridge) and
//! `CTraderDiscoveredAccount` (broker_api / live auth).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderDiscoveredAccount {
    pub account_id: String,
    pub broker_title: String,
    pub account_name: String,
    pub trader_login: Option<i64>,
    pub is_live: Option<bool>,
    pub enabled_for_execution: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CTraderTokenBundle {
    pub access_token: String,
    pub refresh_token: String,
    pub token_type: String,
    pub expires_in: i64,
    pub scope: String,
    pub created_at_unix: i64,
}

impl CTraderTokenBundle {
    pub fn expires_at_unix(&self) -> i64 {
        self.created_at_unix.saturating_add(self.expires_in.max(0))
    }

    pub fn is_expired_at(&self, now_unix: i64) -> bool {
        now_unix >= self.expires_at_unix()
    }

    pub fn needs_refresh_at(&self, now_unix: i64, refresh_window_secs: i64) -> bool {
        let refresh_window_secs = refresh_window_secs.max(0);
        self.is_expired_at(now_unix)
            || self.expires_at_unix().saturating_sub(now_unix) <= refresh_window_secs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_bundle_detects_expired_access_tokens() {
        let bundle = CTraderTokenBundle {
            access_token: "access".to_string(),
            refresh_token: "refresh".to_string(),
            token_type: "bearer".to_string(),
            expires_in: 60,
            scope: "trading".to_string(),
            created_at_unix: 1_000,
        };

        assert!(bundle.is_expired_at(1_061));
    }

    #[test]
    fn token_bundle_requests_refresh_when_inside_safety_window() {
        let bundle = CTraderTokenBundle {
            access_token: "access".to_string(),
            refresh_token: "refresh".to_string(),
            token_type: "bearer".to_string(),
            expires_in: 600,
            scope: "trading".to_string(),
            created_at_unix: 2_000,
        };

        assert!(bundle.needs_refresh_at(2_301, 300));
        assert!(!bundle.needs_refresh_at(2_200, 300));
    }
}
