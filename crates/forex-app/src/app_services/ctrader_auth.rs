use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CTraderAuthState {
    NotConfigured,
    ReadyToAuthorize,
    AwaitingAuthorizationCode,
    ListeningForCallback,
    AuthorizationCodeReceived,
    ExchangingToken,
    AccessTokenReady,
    RestoredFromStorage,
    AccountsAvailable,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderTokenExchangeRequest {
    pub grant_type: String,
    pub code: String,
    pub client_id: String,
    pub client_secret: String,
    pub redirect_uri: String,
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
    pub authorize_url: Option<String>,
    pub callback_port: Option<u16>,
    pub authorization_code_present: bool,
    pub token_request_ready: bool,
    pub token_persisted: bool,
    pub persistence_status: String,
    pub account_count: usize,
    pub enabled_target_count: usize,
    pub discovered_accounts: Vec<CTraderDiscoveredAccount>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CTraderAuthSession {
    client_id: String,
    redirect_uri: String,
    state: CTraderAuthState,
    authorize_url: Option<String>,
    callback_port: Option<u16>,
    authorization_code: Option<String>,
    token_bundle: Option<CTraderTokenBundle>,
    accounts: Vec<CTraderAccountSummary>,
    discovered_accounts: Vec<CTraderDiscoveredAccount>,
    failure_message: Option<String>,
}

impl CTraderAuthSession {
    pub fn new(client_id: impl Into<String>, redirect_uri: impl Into<String>) -> Self {
        let client_id = client_id.into();
        let redirect_uri = redirect_uri.into();
        let configured = !client_id.trim().is_empty() && !redirect_uri.trim().is_empty();

        Self {
            client_id,
            redirect_uri,
            state: if configured {
                CTraderAuthState::ReadyToAuthorize
            } else {
                CTraderAuthState::NotConfigured
            },
            authorize_url: None,
            callback_port: None,
            authorization_code: None,
            token_bundle: None,
            accounts: Vec::new(),
            discovered_accounts: Vec::new(),
            failure_message: None,
        }
    }

    pub fn start_authorization(&mut self, scope: &str) -> String {
        self.failure_message = None;
        let url = format!(
            "https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id={}&redirect_uri={}&scope={}&product=web",
            percent_encode(&self.client_id),
            percent_encode(&self.redirect_uri),
            percent_encode(scope),
        );
        self.authorize_url = Some(url.clone());
        self.state = CTraderAuthState::AwaitingAuthorizationCode;
        url
    }

    pub fn mark_listening_for_callback(&mut self, callback_port: u16) {
        self.failure_message = None;
        self.callback_port = Some(callback_port);
        self.state = CTraderAuthState::ListeningForCallback;
    }

    pub fn receive_authorization_code(&mut self, code: impl Into<String>) {
        self.failure_message = None;
        self.authorization_code = Some(code.into());
        self.state = CTraderAuthState::AuthorizationCodeReceived;
    }

    pub fn build_token_exchange_request(
        &mut self,
        client_secret: impl Into<String>,
    ) -> CTraderTokenExchangeRequest {
        self.failure_message = None;
        self.state = CTraderAuthState::ExchangingToken;
        let request = CTraderTokenExchangeRequest {
            grant_type: "authorization_code".to_string(),
            code: self.authorization_code.clone().unwrap_or_default(),
            client_id: self.client_id.clone(),
            client_secret: client_secret.into(),
            redirect_uri: self.redirect_uri.clone(),
        };
        self.state = CTraderAuthState::AccessTokenReady;
        request
    }

    pub fn restore_from_storage(&mut self, token_bundle: CTraderTokenBundle) {
        self.authorize_url = None;
        self.callback_port = None;
        self.authorization_code = None;
        self.token_bundle = Some(token_bundle);
        self.accounts.clear();
        self.discovered_accounts.clear();
        self.failure_message = None;
        self.state = CTraderAuthState::RestoredFromStorage;
    }

    pub fn replace_persisted_token_bundle(&mut self, token_bundle: CTraderTokenBundle) {
        self.authorize_url = None;
        self.callback_port = None;
        self.authorization_code = None;
        self.token_bundle = Some(token_bundle);
        self.failure_message = None;
        if !matches!(self.state, CTraderAuthState::AccountsAvailable) {
            self.state = CTraderAuthState::RestoredFromStorage;
        }
    }

    pub fn set_accounts(&mut self, accounts: Vec<CTraderAccountSummary>) {
        self.accounts = accounts;
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
        self.failure_message = None;
        self.state = CTraderAuthState::AccountsAvailable;
    }

    pub fn mark_failed(&mut self, message: impl Into<String>) {
        self.failure_message = Some(message.into());
        self.state = CTraderAuthState::Failed;
    }

    pub fn snapshot(&self) -> CTraderAuthSnapshot {
        CTraderAuthSnapshot {
            state: self.state.clone(),
            status_line: match self.state {
                CTraderAuthState::NotConfigured => "cTrader auth is not configured.".to_string(),
                CTraderAuthState::ReadyToAuthorize => "cTrader is ready to authorize.".to_string(),
                CTraderAuthState::AwaitingAuthorizationCode => {
                    "Waiting for cTrader authorization code.".to_string()
                }
                CTraderAuthState::ListeningForCallback => {
                    "Listening for cTrader callback.".to_string()
                }
                CTraderAuthState::AuthorizationCodeReceived => {
                    "Authorization code received.".to_string()
                }
                CTraderAuthState::ExchangingToken => {
                    "Exchanging cTrader authorization code for tokens.".to_string()
                }
                CTraderAuthState::AccessTokenReady => {
                    "Token exchange request is ready.".to_string()
                }
                CTraderAuthState::RestoredFromStorage => {
                    "cTrader session restored from secure storage.".to_string()
                }
                CTraderAuthState::AccountsAvailable => {
                    format!(
                        "{} cTrader accounts are available.",
                        self.discovered_accounts.len()
                    )
                }
                CTraderAuthState::Failed => self
                    .failure_message
                    .as_ref()
                    .filter(|message| !message.trim().is_empty())
                    .map(|message| format!("cTrader auth failed: {message}"))
                    .unwrap_or_else(|| "cTrader auth failed.".to_string()),
            },
            authorize_url: self.authorize_url.clone(),
            callback_port: self.callback_port,
            authorization_code_present: self.authorization_code.is_some(),
            token_request_ready: matches!(
                self.state,
                CTraderAuthState::AccessTokenReady
                    | CTraderAuthState::RestoredFromStorage
                    | CTraderAuthState::AccountsAvailable
            ),
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

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .flat_map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                vec![char::from(byte).to_string()]
            }
            _ => vec![format!("%{byte:02X}")],
        })
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_ctrader_auth_builds_authorize_url() {
        let mut auth = CTraderAuthSession::new("client-id", "http://localhost:3000/callback");

        let url = auth.start_authorization("trading");

        assert!(url.contains("client_id=client-id"));
        assert!(url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A3000%2Fcallback"));
        assert!(url.contains("scope=trading"));
        assert_eq!(
            auth.snapshot().state,
            CTraderAuthState::AwaitingAuthorizationCode
        );
    }

    #[test]
    fn receiving_authorization_code_advances_state_and_builds_token_request() {
        let mut auth = CTraderAuthSession::new("client-id", "http://localhost:3000/callback");
        auth.start_authorization("trading");

        auth.receive_authorization_code("auth-code-123");
        let token_request = auth.build_token_exchange_request("secret-456");

        assert_eq!(auth.snapshot().state, CTraderAuthState::AccessTokenReady);
        assert_eq!(token_request.grant_type, "authorization_code");
        assert_eq!(token_request.code, "auth-code-123");
        assert_eq!(token_request.client_id, "client-id");
        assert_eq!(token_request.client_secret, "secret-456");
        assert_eq!(token_request.redirect_uri, "http://localhost:3000/callback");
    }

    #[test]
    fn configured_account_targets_do_not_fabricate_discovered_accounts() {
        let mut auth = CTraderAuthSession::new("client-id", "http://localhost:3000/callback");
        auth.set_accounts(vec![
            CTraderAccountSummary {
                account_id: "1001".to_string(),
                broker_title: "Broker A".to_string(),
                enabled_for_execution: true,
            },
            CTraderAccountSummary {
                account_id: "1002".to_string(),
                broker_title: "Broker B".to_string(),
                enabled_for_execution: false,
            },
        ]);

        let snapshot = auth.snapshot();
        assert_eq!(snapshot.state, CTraderAuthState::ReadyToAuthorize);
        assert_eq!(snapshot.account_count, 0);
        assert_eq!(snapshot.enabled_target_count, 0);
        assert!(snapshot.discovered_accounts.is_empty());
    }

    #[test]
    fn listener_state_tracks_callback_port_and_persistence_status() {
        let mut auth = CTraderAuthSession::new("client-id", "http://localhost:43001/callback");
        auth.start_authorization("trading");

        auth.mark_listening_for_callback(43001);

        let snapshot = auth.snapshot();
        assert_eq!(snapshot.state, CTraderAuthState::ListeningForCallback);
        assert_eq!(snapshot.callback_port, Some(43001));
        assert_eq!(snapshot.persistence_status, "Not stored");
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
        assert_eq!(snapshot.discovered_accounts[0].trader_login, Some(9001));
        assert_eq!(snapshot.discovered_accounts[0].is_live, Some(true));
        assert_eq!(snapshot.discovered_accounts[1].is_live, Some(false));
    }

    #[test]
    fn restored_session_remains_distinct_from_accounts_available() {
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
        assert_eq!(snapshot.account_count, 0);
        assert_eq!(snapshot.enabled_target_count, 0);
        assert!(snapshot.discovered_accounts.is_empty());
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
        assert_eq!(snapshot.enabled_target_count, 0);
        assert!(snapshot.discovered_accounts.is_empty());
    }

    #[test]
    fn enabled_target_count_derives_from_discovered_accounts() {
        let mut auth = CTraderAuthSession::new("client-id", "http://localhost:43001/callback");
        auth.set_discovered_accounts(vec![
            CTraderDiscoveredAccount {
                account_id: "1001".to_string(),
                broker_title: "Broker A".to_string(),
                account_name: "Live".to_string(),
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
            CTraderDiscoveredAccount {
                account_id: "1003".to_string(),
                broker_title: "Broker A".to_string(),
                account_name: "Backup".to_string(),
                trader_login: Some(9003),
                is_live: Some(true),
                enabled_for_execution: true,
            },
        ]);

        let snapshot = auth.snapshot();
        assert_eq!(snapshot.enabled_target_count, 2);
    }

    #[test]
    fn discovered_accounts_include_identity_needed_for_sync() {
        let mut auth = CTraderAuthSession::new("client-id", "http://localhost:43001/callback");
        auth.set_discovered_accounts(vec![CTraderDiscoveredAccount {
            account_id: "1001".to_string(),
            broker_title: "Broker A".to_string(),
            account_name: "Primary Live".to_string(),
            trader_login: Some(9901),
            is_live: Some(true),
            enabled_for_execution: true,
        }]);

        let snapshot = auth.snapshot();
        assert_eq!(snapshot.discovered_accounts[0].account_id, "1001");
        assert_eq!(snapshot.discovered_accounts[0].broker_title, "Broker A");
        assert_eq!(snapshot.discovered_accounts[0].account_name, "Primary Live");
        assert_eq!(snapshot.discovered_accounts[0].trader_login, Some(9901));
        assert_eq!(snapshot.discovered_accounts[0].is_live, Some(true));
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
