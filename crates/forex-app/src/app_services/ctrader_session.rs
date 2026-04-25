use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use futures::prelude::*;
use anyhow::{Result, Context, anyhow};
use tracing::{info, error, warn};
use tokio::time::{interval, Duration};

use crate::app_services::ctrader_proto_messages::{build_heartbeat, build_app_auth_req, build_account_auth_req};
use crate::app_services::ctrader_live_auth::CTraderEnvironment;

pub struct CTraderSession {
    tx: mpsc::Sender<Vec<u8>>,
    rx: Arc<Mutex<mpsc::Receiver<Vec<u8>>>>,
}

impl CTraderSession {
    pub async fn connect(
        environment: CTraderEnvironment,
        client_id: String,
        client_secret: String,
        account_id: i64,
        access_token: String,
    ) -> Result<Self> {
        let url = format!("wss://{}:5036", environment.endpoint_host());
        let (ws_stream, _) = connect_async(url).await.context("failed to connect to cTrader")?;
        let (mut write, mut read) = ws_stream.split();

        let (tx_out, mut rx_out) = mpsc::channel::<Vec<u8>>(100);
        let (tx_in, rx_in) = mpsc::channel::<Vec<u8>>(100);

        // Background write task
        tokio::spawn(async move {
            while let Some(msg) = rx_out.recv().await {
                if let Err(e) = write.send(Message::Binary(msg.into())).await {
                    error!("cTrader session write error: {}", e);
                    break;
                }
            }
        });

        // Background read task
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                match msg {
                    Ok(Message::Binary(data)) => {
                        if let Err(e) = tx_in.send(data.to_vec()).await {
                            error!("cTrader session read error: {}", e);
                            break;
                        }
                    }
                    Ok(Message::Text(text)) => {
                        // Some environments might send JSON even over binary socket if requested, 
                        // but we are moving to binary.
                        warn!("received unexpected text message: {}", text);
                    }
                    Err(e) => {
                        error!("cTrader session error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        });

        // Initial Auth sequence
        let app_auth = build_app_auth_req(&client_id, &client_secret, Some("app-auth".to_string()))?;
        tx_out.send(app_auth).await?;

        let account_auth = build_account_auth_req(account_id, &access_token, Some("acc-auth".to_string()))?;
        tx_out.send(account_auth).await?;

        // Heartbeat task
        let tx_heartbeat = tx_out.clone();
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_secs(10));
            loop {
                ticker.tick().await;
                if let Ok(hb) = build_heartbeat() {
                    if let Err(e) = tx_heartbeat.send(hb).await {
                        error!("failed to send heartbeat: {}", e);
                        break;
                    }
                }
            }
        });

        Ok(Self {
            tx: tx_out,
            rx: Arc::new(Mutex::new(rx_in)),
        })
    }

    pub async fn send(&self, data: Vec<u8>) -> Result<()> {
        self.tx.send(data).await.context("session closed")
    }

    pub async fn next_message(&self) -> Option<Vec<u8>> {
        self.rx.lock().await.recv().await
    }
}
