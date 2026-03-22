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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CTraderTokenBundle {
    pub access_token: String,
    pub refresh_token: String,
    pub token_type: String,
    pub expires_in: i64,
    pub scope: String,
    pub created_at_unix: i64,
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
        }
    }

    pub fn start_authorization(&mut self, scope: &str) -> String {
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
        self.callback_port = Some(callback_port);
        self.state = CTraderAuthState::ListeningForCallback;
    }

    pub fn receive_authorization_code(&mut self, code: impl Into<String>) {
        self.authorization_code = Some(code.into());
        self.state = CTraderAuthState::AuthorizationCodeReceived;
    }

    pub fn build_token_exchange_request(
        &mut self,
        client_secret: impl Into<String>,
    ) -> CTraderTokenExchangeRequest {
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
        self.token_bundle = Some(token_bundle);
        self.state = CTraderAuthState::RestoredFromStorage;
    }

    pub fn set_accounts(&mut self, accounts: Vec<CTraderAccountSummary>) {
        self.accounts = accounts;
        self.state = CTraderAuthState::AccountsAvailable;
    }

    pub fn mark_failed(&mut self) {
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
                    format!("{} cTrader accounts are available.", self.accounts.len())
                }
                CTraderAuthState::Failed => "cTrader auth failed.".to_string(),
            },
            authorize_url: self.authorize_url.clone(),
            callback_port: self.callback_port,
            authorization_code_present: self.authorization_code.is_some(),
            token_request_ready: matches!(
                self.state,
                CTraderAuthState::AccessTokenReady | CTraderAuthState::RestoredFromStorage | CTraderAuthState::AccountsAvailable
            ),
            token_persisted: self.token_bundle.is_some(),
            persistence_status: if self.token_bundle.is_some() {
                "Stored securely".to_string()
            } else {
                "Not stored".to_string()
            },
            account_count: self.accounts.len(),
            enabled_target_count: self
                .accounts
                .iter()
                .filter(|account| account.enabled_for_execution)
                .count(),
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
        assert_eq!(auth.snapshot().state, CTraderAuthState::AwaitingAuthorizationCode);
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
    fn auth_session_retains_account_summaries() {
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
        assert_eq!(snapshot.state, CTraderAuthState::AccountsAvailable);
        assert_eq!(snapshot.account_count, 2);
        assert_eq!(snapshot.enabled_target_count, 1);
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
}
