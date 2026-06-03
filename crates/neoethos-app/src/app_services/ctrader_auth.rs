//! cTrader auth session — minimal state held by `TradingSession` between
//! restore (load_token_bundle), account discovery, and proactive token
//! refresh.
//!
//! The legacy interactive OAuth state machine (AwaitingAuthorizationCode →
//! ListeningForCallback → ExchangingToken → AccessTokenReady) was removed
//! when the egui wizard came down — the production loopback OAuth flow now
//! lives in `app_services::reauth` and only deals with a final
//! `CTraderTokenBundle`. What remains here are the three states the
//! account / bridge pipeline still inspects via `snapshot().state`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CTraderAuthState {
    NotConfigured,
    ReadyToAuthorize,
    RestoredFromStorage,
    AccountsAvailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderAccountSummary {
    pub account_id: String,
    pub broker_title: String,
    pub enabled_for_execution: bool,
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderAuthSnapshot {
    pub state: CTraderAuthState,
    pub status_line: String,
    pub token_persisted: bool,
    pub persistence_status: String,
    pub account_count: usize,
    pub enabled_target_count: usize,
    pub discovered_accounts: Vec<CTraderDiscoveredAccount>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderAuthSession {
    state: CTraderAuthState,
    token_bundle: Option<CTraderTokenBundle>,
    accounts: Vec<CTraderAccountSummary>,
    discovered_accounts: Vec<CTraderDiscoveredAccount>,
}

impl CTraderAuthSession {
    pub fn new(client_id: impl Into<String>, redirect_uri: impl Into<String>) -> Self {
        let client_id = client_id.into();
        let redirect_uri = redirect_uri.into();
        let configured = !client_id.trim().is_empty() && !redirect_uri.trim().is_empty();

        Self {
            state: if configured {
                CTraderAuthState::ReadyToAuthorize
            } else {
                CTraderAuthState::NotConfigured
            },
            token_bundle: None,
            accounts: Vec::new(),
            discovered_accounts: Vec::new(),
        }
    }

    pub fn restore_from_storage(&mut self, token_bundle: CTraderTokenBundle) {
        self.token_bundle = Some(token_bundle);
        self.accounts.clear();
        self.discovered_accounts.clear();
        self.state = CTraderAuthState::RestoredFromStorage;
    }

    pub fn set_discovered_accounts(&mut self, accounts: Vec<CTraderDiscoveredAccount>) {
        self.discovered_accounts = accounts.clone();
        self.accounts = accounts
            .into_iter()
            .map(|account| CTraderAccountSummary {
                account_id: account.account_id,
                broker_title: account.broker_title,
                enabled_for_execution: account.enabled_for_execution,
            })
            .collect();
        self.state = CTraderAuthState::AccountsAvailable;
    }

    pub fn snapshot(&self) -> CTraderAuthSnapshot {
        CTraderAuthSnapshot {
            state: self.state.clone(),
            status_line: match self.state {
                CTraderAuthState::NotConfigured => "cTrader auth is not configured.".to_string(),
                CTraderAuthState::ReadyToAuthorize => "cTrader is ready to authorize.".to_string(),
                CTraderAuthState::RestoredFromStorage => {
                    "cTrader session restored from secure storage.".to_string()
                }
                CTraderAuthState::AccountsAvailable => {
                    format!(
                        "{} cTrader accounts are available.",
                        self.discovered_accounts.len()
                    )
                }
            },
            token_persisted: self.token_bundle.is_some(),
            persistence_status: if self.token_bundle.is_some() {
                "Stored securely".to_string()
            } else {
                "Not stored".to_string()
            },
            account_count: self.discovered_accounts.len(),
            enabled_target_count: self
                .discovered_accounts
                .iter()
                .filter(|account| account.enabled_for_execution)
                .count(),
            discovered_accounts: self.discovered_accounts.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_session_starts_in_ready_to_authorize() {
        let auth = CTraderAuthSession::new("client-id", "http://localhost:43001/callback");
        assert_eq!(auth.snapshot().state, CTraderAuthState::ReadyToAuthorize);
    }

    #[test]
    fn unconfigured_session_starts_in_not_configured() {
        let auth = CTraderAuthSession::new("", "");
        assert_eq!(auth.snapshot().state, CTraderAuthState::NotConfigured);
    }

    #[test]
    fn restored_session_snapshot_reports_persisted_token_bundle() {
        let mut auth = CTraderAuthSession::new("client-id", "http://localhost:43001/callback");
        auth.restore_from_storage(CTraderTokenBundle {
            access_token: "access".to_string(),
            refresh_token: "refresh".to_string(),
            token_type: "bearer".to_string(),
            expires_in: 3600,
            scope: "trading".to_string(),
            created_at_unix: 1_774_147_200,
        });

        let snapshot = auth.snapshot();
        assert_eq!(snapshot.state, CTraderAuthState::RestoredFromStorage);
        assert!(snapshot.token_persisted);
        assert_eq!(snapshot.persistence_status, "Stored securely");
    }

    #[test]
    fn discovered_accounts_are_retained_in_snapshot() {
        let mut auth = CTraderAuthSession::new("client-id", "http://localhost:43001/callback");
        auth.set_discovered_accounts(vec![
            CTraderDiscoveredAccount {
                account_id: "1001".to_string(),
                broker_title: "Broker A".to_string(),
                account_name: "Primary".to_string(),
                trader_login: Some(9001),
                is_live: Some(true),
                enabled_for_execution: true,
            },
            CTraderDiscoveredAccount {
                account_id: "1002".to_string(),
                broker_title: "Broker A".to_string(),
                account_name: "Demo".to_string(),
                trader_login: Some(9002),
                is_live: Some(false),
                enabled_for_execution: false,
            },
        ]);

        let snapshot = auth.snapshot();
        assert_eq!(snapshot.state, CTraderAuthState::AccountsAvailable);
        assert_eq!(snapshot.account_count, 2);
        assert_eq!(snapshot.enabled_target_count, 1);
        assert_eq!(snapshot.discovered_accounts.len(), 2);
        assert_eq!(snapshot.discovered_accounts[0].account_name, "Primary");
    }

    #[test]
    fn restoring_session_clears_stale_discovered_accounts() {
        let mut auth = CTraderAuthSession::new("client-id", "http://localhost:43001/callback");
        auth.set_discovered_accounts(vec![CTraderDiscoveredAccount {
            account_id: "1001".to_string(),
            broker_title: "Broker A".to_string(),
            account_name: "Primary".to_string(),
            trader_login: Some(9001),
            is_live: Some(true),
            enabled_for_execution: true,
        }]);

        auth.restore_from_storage(CTraderTokenBundle {
            access_token: "access".to_string(),
            refresh_token: "refresh".to_string(),
            token_type: "bearer".to_string(),
            expires_in: 3600,
            scope: "trading".to_string(),
            created_at_unix: 1_774_147_200,
        });

        let snapshot = auth.snapshot();
        assert_eq!(snapshot.state, CTraderAuthState::RestoredFromStorage);
        assert_eq!(snapshot.account_count, 0);
        assert!(snapshot.discovered_accounts.is_empty());
    }

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
