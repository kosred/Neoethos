use super::*;
use crate::app_state::AppRuntimeConfig;
use forex_core::config::RiskConfig;
use forex_data::Ohlcv;
use std::path::PathBuf;

fn sample_state(source: DataSource, status_msg: &str) -> AppState {
    let runtime = AppRuntimeConfig {
        config_path: "config.yaml".to_string(),
        data_dir: PathBuf::from("data"),
        start_local: matches!(source, DataSource::Local),
        auto_discovery: false,
        auto_training: false,
    };
    let mut state = AppState::new(
        runtime,
        &forex_core::Settings::default(),
        vec!["EURUSD".to_string()],
    );
    state.data_source = source;
    state.status_msg = status_msg.to_string();
    state
}

fn fresh_ctrader_token_bundle(
    access_token: &str,
    refresh_token: &str,
) -> crate::app_services::ctrader_auth::CTraderTokenBundle {
    crate::app_services::ctrader_auth::CTraderTokenBundle {
        access_token: access_token.to_string(),
        refresh_token: refresh_token.to_string(),
        token_type: "bearer".to_string(),
        expires_in: 3600,
        scope: "trading".to_string(),
        created_at_unix: current_unix_seconds().expect("current unix time"),
    }
}

#[test]
fn connection_snapshot_reports_local_mode_without_live_runtime() {
    let state = sample_state(DataSource::Local, "Local Mode");
    let session = TradingSession::new();

    let snapshot = session.snapshot(&state);

    assert_eq!(snapshot.mode, TradingPanelMode::LocalOnly);
    assert!(!snapshot.connected);
    assert_eq!(snapshot.status_text, "Local Mode");
    assert_eq!(snapshot.terminal_info, "");
}

#[test]
fn panel_mode_uses_data_source_and_connection_state() {
    assert_eq!(
        panel_mode(DataSource::Local, false),
        TradingPanelMode::LocalOnly
    );
    assert_eq!(
        panel_mode(DataSource::CTrader, false),
        TradingPanelMode::Disconnected
    );
    assert_eq!(
        panel_mode(DataSource::CTrader, true),
        TradingPanelMode::Connected
    );
}

#[test]
fn connection_snapshot_reports_remote_api_metadata_for_stubbed_ctrader() {
    let state = sample_state(DataSource::CTrader, "Offline");
    let session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

    let snapshot = session.snapshot(&state);

    assert_eq!(snapshot.adapter_name, "cTrader");
    assert_eq!(snapshot.integration_mode, "Remote Open API");
    assert!(!snapshot.requires_local_terminal);
    assert!(snapshot.supports_market_data);
    assert!(snapshot.supports_live_orders);
    assert_eq!(snapshot.mode, TradingPanelMode::Disconnected);
}

#[test]
fn connection_snapshot_reports_remote_api_metadata_for_stubbed_dxtrade() {
    let state = sample_state(DataSource::CTrader, "Offline");
    let session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::DxTrade);

    let snapshot = session.snapshot(&state);

    assert_eq!(snapshot.adapter_name, "DXtrade");
    assert_eq!(snapshot.integration_mode, "Remote broker API");
    assert!(!snapshot.requires_local_terminal);
    assert!(snapshot.supports_market_data);
    assert!(snapshot.supports_live_orders);
    assert_eq!(snapshot.mode, TradingPanelMode::Disconnected);
}

#[test]
fn market_chart_snapshot_uses_recent_real_candles_and_preserves_price_bounds() {
    let ohlcv = Ohlcv {
        timestamp: Some((1_i64..=140).collect()),
        open: (0..140).map(|idx| 1.1000 + idx as f64 * 0.0001).collect(),
        high: (0..140).map(|idx| 1.1010 + idx as f64 * 0.0001).collect(),
        low: (0..140).map(|idx| 1.0990 + idx as f64 * 0.0001).collect(),
        close: (0..140).map(|idx| 1.1005 + idx as f64 * 0.0001).collect(),
        volume: None,
    };

    let snapshot = build_market_chart_snapshot_from_ohlcv(
        "EURUSD",
        "M5",
        vec!["M1".to_string(), "M5".to_string(), "H1".to_string()],
        &ohlcv,
        Vec::new(),
        Vec::new(),
    );

    assert_eq!(snapshot.symbol, "EURUSD");
    assert_eq!(snapshot.timeframe, "M5");
    assert_eq!(snapshot.available_timeframes, vec!["M1", "M5", "H1"]);
    assert_eq!(snapshot.candles.len(), 96);
    assert_eq!(snapshot.candles.first().and_then(|c| c.timestamp), Some(45));
    assert_eq!(snapshot.candles.last().and_then(|c| c.timestamp), Some(140));
    assert!(snapshot.price_min < snapshot.price_max);
    assert!(snapshot.headline.contains("96 candles"));
}

#[test]
fn build_market_chart_snapshot_from_historical_bars_preserves_recent_candles() {
    let bars: Vec<HistoricalBar> = (0..140)
        .map(|idx| HistoricalBar {
            timestamp_ms: 1_700_000_000_000 + idx as i64 * 60_000,
            open: 1.1000 + idx as f64 * 0.0001,
            high: 1.1010 + idx as f64 * 0.0001,
            low: 1.0990 + idx as f64 * 0.0001,
            close: 1.1005 + idx as f64 * 0.0001,
            volume: Some(10 + idx as i64),
        })
        .collect();

    let snapshot = build_market_chart_snapshot_from_historical_bars(
        "EURUSD",
        "M5",
        vec!["M1".to_string(), "M5".to_string()],
        &bars,
        Vec::new(),
        Vec::new(),
    );

    assert_eq!(snapshot.candles.len(), 96);
    assert_eq!(
        snapshot.candles.first().and_then(|candle| candle.timestamp),
        Some(1_700_002_640_000)
    );
    assert_eq!(snapshot.available_timeframes, vec!["M1", "M5"]);
}

#[test]
fn ctrader_chart_history_request_uses_enabled_target_and_selected_environment() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:8877/callback".to_string();
    session.broker_settings_mut().ctrader.environment = CTraderBrokerEnvironment::Demo;
    session.broker_settings_mut().ctrader.accounts = vec![
        BrokerAccountTarget {
            account_id: "1001".to_string(),
            label: "standby".to_string(),
            enabled_for_execution: false,
        },
        BrokerAccountTarget {
            account_id: "2002".to_string(),
            label: "primary".to_string(),
            enabled_for_execution: true,
        },
    ];
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session
        .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
        .expect("seed token bundle");

    let request = session
        .build_ctrader_chart_history_request("EURUSD", "M15")
        .expect("request should build");

    assert_eq!(request.environment, CTraderEnvironment::Demo);
    assert_eq!(request.account_id, "2002");
    assert_eq!(request.symbol_name, "EURUSD");
    assert_eq!(request.timeframe, "M15");
    assert_eq!(request.count, Some((MAX_CHART_CANDLES + 24) as u32));
    assert!(request.to_timestamp_ms >= request.from_timestamp_ms);
}

#[test]
fn market_chart_snapshot_reports_ctrader_requirements_instead_of_fake_fallback() {
    let state = sample_state(DataSource::CTrader, "Offline");
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();

    let snapshot = session.market_chart_snapshot(&state);

    assert!(snapshot.candles.is_empty());
    assert_eq!(snapshot.timeframe, "M1");
    assert_eq!(
        snapshot.available_timeframes.first().map(String::as_str),
        Some("M1")
    );
    assert!(
        snapshot
            .warnings
            .iter()
            .any(|warning| warning.contains("stored token bundle"))
    );
    assert!(snapshot.headline.contains("No cTrader market data loaded"));
}

#[test]
fn execution_surface_snapshot_disables_live_actions_and_surfaces_unwired_gaps() {
    let state = sample_state(DataSource::Local, "Local Mode");
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

    let snapshot = session.execution_surface_snapshot(&state);

    assert_eq!(snapshot.symbol, "EURUSD");
    assert_eq!(snapshot.adapter_name, "cTrader");
    assert_eq!(snapshot.primary_actions.len(), 2);
    assert!(
        snapshot
            .primary_actions
            .iter()
            .all(|action| !action.enabled)
    );
    assert!(
        snapshot
            .warnings
            .iter()
            .any(|warning| warning.contains("Local mode"))
    );
    assert!(
        snapshot
            .diagnostics
            .iter()
            .any(|line| line.contains("central broker background loop"))
    );
}

#[test]
fn execution_surface_snapshot_surfaces_adapter_specific_unwired_feed_reason() {
    let state = sample_state(DataSource::CTrader, "Offline");
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

    let snapshot = session.execution_surface_snapshot(&state);

    assert!(snapshot.warnings.iter().any(|warning| warning.contains(
        "cTrader execution feed is unavailable until the remote account session connects"
    )));
}

#[test]
fn selecting_adapter_updates_configured_runtime_and_status_message() {
    let mut state = sample_state(DataSource::CTrader, "Offline");
    let mut session = TradingSession::new();

    session.select_adapter(&mut state, TradingAdapterKind::CTrader);
    let snapshot = session.snapshot(&state);

    assert_eq!(snapshot.adapter_name, "cTrader");
    assert_eq!(snapshot.integration_mode, "Remote Open API");
    assert!(!session.is_connected());
    assert_eq!(state.status_msg, "cTrader selected · disconnected");
}

#[test]
fn connect_sets_missing_credentials_status_for_unready_remote_adapter() {
    let mut state = sample_state(DataSource::CTrader, "Offline");
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);

    session.connect(&mut state);

    assert_eq!(
        state.status_msg,
        "cTrader configuration incomplete: missing client_id, client_secret, redirect_uri"
    );
    assert!(!session.is_connected());
}

#[test]
fn connect_requires_restored_ctrader_session_before_runtime_probe() {
    let mut state = sample_state(DataSource::CTrader, "Offline");
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://localhost:3000/callback".to_string();

    session.connect(&mut state);

    assert_eq!(
        state.status_msg,
        "cTrader login required · restore or start auth first"
    );
    assert!(!session.is_connected());
}

#[test]
fn connect_uses_ctrader_account_runtime_probe_when_session_is_restored() {
    let mut state = sample_state(DataSource::CTrader, "Offline");
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://localhost:3000/callback".to_string();
    session.broker_settings_mut().ctrader.accounts = vec![BrokerAccountTarget {
        account_id: "712345".to_string(),
        label: "primary".to_string(),
        enabled_for_execution: true,
    }];
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session
        .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
        .expect("seed token bundle");
    session.set_ctrader_account_runtime_backend_for_test(
        crate::app_services::ctrader_account::StubCTraderAccountRuntimeBackend::success(
            crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
                trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                    account_id: 712345,
                    balance: 1000.0,
                    leverage: Some(50.0),
                    trader_login: Some(998877),
                    account_type: Some("NETTED".to_string()),
                    broker_name: Some("Demo Broker".to_string()),
                    money_digits: 2,
                    unrealized_pnl: 0.0,
                },
                reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                    account_id: 712345,
                    positions: Vec::new(),
                    pending_orders: Vec::new(),
                },
                recent_deals: Vec::new(),
            },
        ),
    );

    session.connect(&mut state);

    assert!(session.is_connected());
    assert_eq!(state.status_msg, "cTrader connected");
}

#[test]
fn ctrader_live_spot_cache_reuses_backend_update_within_ttl() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    let backend = crate::app_services::ctrader_streaming::StubCTraderLiveStreamingBackend::success(
        crate::app_services::ctrader_streaming::CTraderLiveChartUpdate {
            symbol_id: 14,
            bid: Some(1.09995),
            ask: Some(1.10015),
            timestamp_ms: Some(1_710_000_200_000),
            latest_trendbar: None,
        },
    );
    session.set_ctrader_live_streaming_backend_for_test(backend.clone());

    let request = crate::app_services::ctrader_streaming::CTraderLiveChartUpdateRequest {
        client_id: "client".to_string(),
        client_secret: "secret".to_string(),
        access_token: "access".to_string(),
        environment: crate::app_services::ctrader_live_auth::CTraderEnvironment::Demo,
        account_id: "712345".to_string(),
        symbol_id: 14,
        digits: 5,
        timeframe: "M1".to_string(),
        subscribe_to_spot_timestamp: true,
    };

    let first = session
        .load_ctrader_live_chart_update_cached(&request)
        .expect("first live update");
    let second = session
        .load_ctrader_live_chart_update_cached(&request)
        .expect("second live update");

    assert_eq!(first, second);
    assert_eq!(backend.call_count(), 1);
}

#[test]
fn execution_surface_snapshot_uses_ctrader_reconcile_runtime_when_connected() {
    let mut state = sample_state(DataSource::CTrader, "Connected");
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://localhost:3000/callback".to_string();
    session.broker_settings_mut().ctrader.accounts = vec![BrokerAccountTarget {
        account_id: "712345".to_string(),
        label: "primary".to_string(),
        enabled_for_execution: true,
    }];
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session
        .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
        .expect("seed token bundle");
    session.set_ctrader_account_runtime_backend_for_test(
        crate::app_services::ctrader_account::StubCTraderAccountRuntimeBackend::success(
            crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
                trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                    account_id: 712345,
                    balance: 1000.0,
                    leverage: Some(50.0),
                    trader_login: Some(998877),
                    account_type: Some("NETTED".to_string()),
                    broker_name: Some("Demo Broker".to_string()),
                    money_digits: 2,
                    unrealized_pnl: 0.0,
                },
                reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                    account_id: 712345,
                    positions: vec![
                        crate::app_services::ctrader_account::CTraderPositionSnapshot {
                            position_id: 9001,
                            symbol_id: 14,
                            trade_side: "BUY".to_string(),
                            volume: 25.0,
                            open_timestamp_ms: Some(1710000000000),
                            price: Some(1.10123),
                            stop_loss: Some(1.095),
                            take_profit: Some(1.11),
                            swap: None,
                            commission: None,
                            mirroring_commission: None,
                            used_margin: None,
                            label: Some("trend".to_string()),
                            comment: Some("bot".to_string()),
                            client_order_id: None,
                        },
                    ],
                    pending_orders: vec![
                        crate::app_services::ctrader_account::CTraderPendingOrderSnapshot {
                            order_id: 8001,
                            symbol_id: 14,
                            trade_side: "SELL".to_string(),
                            order_type: "LIMIT".to_string(),
                            volume: 15.0,
                            open_timestamp_ms: Some(1710000100000),
                            limit_price: Some(1.099),
                            stop_price: None,
                            stop_loss: Some(1.105),
                            take_profit: Some(1.09),
                            label: Some("breakout".to_string()),
                            comment: Some("pending".to_string()),
                            client_order_id: None,
                        },
                    ],
                },
                recent_deals: vec![crate::app_services::ctrader_account::CTraderDealSnapshot {
                    deal_id: 3001,
                    order_id: 8001,
                    position_id: 9001,
                    symbol_id: 14,
                    trade_side: "BUY".to_string(),
                    deal_status: "FILLED".to_string(),
                    volume: 15.0,
                    filled_volume: 15.0,
                    execution_timestamp_ms: 1710000201000,
                    execution_price: Some(1.0990),
                    entry_price: Some(1.0980),
                    gross_profit: Some(12.5),
                    fee: Some(-0.4),
                    swap: Some(0.0),
                    pnl_conversion_fee: Some(0.0),
                    net_profit: Some(12.1),
                }],
            },
        ),
    );

    session.connect(&mut state);
    let snapshot = session.execution_surface_snapshot(&state);

    assert!(snapshot.positions.iter().any(|line| line.contains("#9001")));
    assert!(
        snapshot
            .pending_orders
            .iter()
            .any(|line| line.contains("#8001"))
    );
    assert!(
        snapshot
            .bot_timeline
            .iter()
            .any(|line| line.contains("#3001"))
    );
    assert!(
        snapshot
            .diagnostics
            .iter()
            .any(|line| line.contains("Recent fills: 1"))
    );
    assert!(
        snapshot
            .diagnostics
            .iter()
            .any(|line| line.contains("Trader balance"))
    );
    assert!(snapshot.warnings.is_empty());
}

#[test]
fn cancel_selected_order_records_ctrader_journal_and_updates_status() {
    let mut state = sample_state(DataSource::CTrader, "Connected");
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://localhost:3000/callback".to_string();
    session.broker_settings_mut().ctrader.accounts = vec![BrokerAccountTarget {
        account_id: "712345".to_string(),
        label: "primary".to_string(),
        enabled_for_execution: true,
    }];
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session
        .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
        .expect("seed token bundle");
    session.set_ctrader_account_runtime_backend_for_test(
        crate::app_services::ctrader_account::StubCTraderAccountRuntimeBackend::success(
            crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
                trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                    account_id: 712345,
                    balance: 1000.0,
                    leverage: Some(50.0),
                    trader_login: Some(998877),
                    account_type: Some("NETTED".to_string()),
                    broker_name: Some("Demo Broker".to_string()),
                    money_digits: 2,
                    unrealized_pnl: 0.0,
                },
                reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                    account_id: 712345,
                    positions: vec![
                        crate::app_services::ctrader_account::CTraderPositionSnapshot {
                            position_id: 9001,
                            symbol_id: 14,
                            trade_side: "BUY".to_string(),
                            volume: 25.0,
                            open_timestamp_ms: Some(1710000000000),
                            price: Some(1.10123),
                            stop_loss: Some(1.095),
                            take_profit: Some(1.11),
                            swap: None,
                            commission: None,
                            mirroring_commission: None,
                            used_margin: None,
                            label: Some("trend".to_string()),
                            comment: Some("bot".to_string()),
                            client_order_id: None,
                        },
                    ],
                    pending_orders: vec![
                        crate::app_services::ctrader_account::CTraderPendingOrderSnapshot {
                            order_id: 8001,
                            symbol_id: 14,
                            trade_side: "SELL".to_string(),
                            order_type: "LIMIT".to_string(),
                            volume: 15.0,
                            open_timestamp_ms: Some(1710000100000),
                            limit_price: Some(1.099),
                            stop_price: None,
                            stop_loss: Some(1.105),
                            take_profit: Some(1.09),
                            label: Some("breakout".to_string()),
                            comment: Some("pending".to_string()),
                            client_order_id: None,
                        },
                    ],
                },
                recent_deals: Vec::new(),
            },
        ),
    );
    session.set_ctrader_position_order_history_backend_for_test(
        crate::app_services::ctrader_history::StubCTraderPositionOrderHistoryBackend::success(
            Vec::new(),
        ),
    );
    session.connect(&mut state);
    state.order_ticket.selected_order_id = Some(8001);
    session.set_ctrader_execution_backend_for_test(
        crate::app_services::ctrader_execution::StubCTraderExecutionBackend::succeed(
            crate::app_services::ctrader_execution::CTraderExecutionOutcome {
                status: crate::app_services::ctrader_execution::CTraderExecutionStatus::Cancelled,
                account_id: 712345,
                symbol_id: Some(14),
                order_id: Some(8001),
                position_id: None,
                deal_id: None,
                trade_side: Some("SELL".to_string()),
                order_type: Some("LIMIT".to_string()),
                lot_size: Some(15.0),
                requested_lot_size: Some(15.0),
                filled_lot_size: None,
                execution_price: None,
                gross_profit: None,
                fee: None,
                swap: None,
                net_profit: None,
                timestamp_ms: Some(1710000300000),
                error_code: None,
                description: None,
            },
        ),
    );

    session.cancel_selected_order(&mut state);
    let snapshot = session.execution_surface_snapshot(&state);

    assert!(state.status_msg.contains("Cancelled order"));
    assert!(
        snapshot
            .journal_rows
            .iter()
            .any(|line| line.contains("Cancel order #8001"))
    );
}

#[test]
fn close_selected_position_surfaces_ctrader_execution_failure() {
    let mut state = sample_state(DataSource::CTrader, "Connected");
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://localhost:3000/callback".to_string();
    session.broker_settings_mut().ctrader.accounts = vec![BrokerAccountTarget {
        account_id: "712345".to_string(),
        label: "primary".to_string(),
        enabled_for_execution: true,
    }];
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session
        .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
        .expect("seed token bundle");
    session.set_ctrader_account_runtime_backend_for_test(
        crate::app_services::ctrader_account::StubCTraderAccountRuntimeBackend::success(
            crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
                trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                    account_id: 712345,
                    balance: 1000.0,
                    leverage: Some(50.0),
                    trader_login: Some(998877),
                    account_type: Some("NETTED".to_string()),
                    broker_name: Some("Demo Broker".to_string()),
                    money_digits: 2,
                    unrealized_pnl: 0.0,
                },
                reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                    account_id: 712345,
                    positions: vec![
                        crate::app_services::ctrader_account::CTraderPositionSnapshot {
                            position_id: 9001,
                            symbol_id: 14,
                            trade_side: "BUY".to_string(),
                            volume: 25.0,
                            open_timestamp_ms: Some(1710000000000),
                            price: Some(1.10123),
                            stop_loss: Some(1.095),
                            take_profit: Some(1.11),
                            swap: None,
                            commission: None,
                            mirroring_commission: None,
                            used_margin: None,
                            label: Some("trend".to_string()),
                            comment: Some("bot".to_string()),
                            client_order_id: None,
                        },
                    ],
                    pending_orders: Vec::new(),
                },
                recent_deals: Vec::new(),
            },
        ),
    );
    session.connect(&mut state);
    state.order_ticket.selected_position_id = Some(9001);
    session.set_ctrader_execution_backend_for_test(
        crate::app_services::ctrader_execution::StubCTraderExecutionBackend::fail(
            "BROKER_REJECTED",
        ),
    );

    session.close_selected_position(&mut state);

    assert!(state.status_msg.contains("failed"));
    assert!(
        session
            .trade_journal
            .iter()
            .any(|line| line.contains("BROKER_REJECTED"))
    );
}

#[test]
fn start_ctrader_auth_exposes_authorize_url_when_ready() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://localhost:3000/callback".to_string();

    let snapshot = session.start_ctrader_auth().expect("auth snapshot");

    assert_eq!(
        snapshot.state,
        crate::app_services::ctrader_auth::CTraderAuthState::AwaitingAuthorizationCode
    );
    assert!(
        snapshot
            .authorize_url
            .as_deref()
            .unwrap_or_default()
            .contains("client_id=client")
    );
}

#[test]
fn receive_ctrader_authorization_code_updates_auth_snapshot() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://localhost:3000/callback".to_string();
    session.start_ctrader_auth().expect("auth snapshot");

    let snapshot = session.receive_ctrader_authorization_code("code-123");

    assert_eq!(
        snapshot.state,
        crate::app_services::ctrader_auth::CTraderAuthState::AuthorizationCodeReceived
    );
    assert!(snapshot.authorization_code_present);
}

#[test]
fn build_ctrader_token_exchange_request_uses_configured_secret() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://localhost:3000/callback".to_string();
    session.start_ctrader_auth().expect("auth snapshot");
    session.receive_ctrader_authorization_code("code-123");

    let request = session
        .build_ctrader_token_exchange_request()
        .expect("token request");

    assert_eq!(request.code, "code-123");
    assert_eq!(request.client_secret, "secret");
    assert_eq!(request.redirect_uri, "http://localhost:3000/callback");
}

#[test]
fn build_ctrader_token_exchange_request_keeps_staged_targets_pending_until_discovery() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://localhost:3000/callback".to_string();
    session.broker_settings_mut().ctrader.accounts.push(
        crate::app_services::broker_config::BrokerAccountTarget {
            account_id: "acct-1".to_string(),
            label: "Primary".to_string(),
            enabled_for_execution: true,
        },
    );
    session.start_ctrader_auth().expect("auth snapshot");
    session.receive_ctrader_authorization_code("code-123");

    let _ = session
        .build_ctrader_token_exchange_request()
        .expect("token request");
    let auth = session.ctrader_auth_snapshot().expect("auth snapshot");

    assert_eq!(
        auth.state,
        crate::app_services::ctrader_auth::CTraderAuthState::AccessTokenReady
    );
    assert_eq!(auth.account_count, 0);
    assert_eq!(auth.enabled_target_count, 0);
}

#[test]
fn restore_ctrader_session_loads_saved_bundle_into_auth_snapshot() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session
        .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
        .expect("seed should succeed");

    let snapshot = session
        .restore_ctrader_session()
        .expect("restore should succeed")
        .expect("snapshot should exist");

    assert_eq!(
        snapshot.state,
        crate::app_services::ctrader_auth::CTraderAuthState::RestoredFromStorage
    );
    assert!(snapshot.token_persisted);
}

#[test]
fn start_ctrader_live_auth_persists_tokens_and_updates_snapshot() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session.set_ctrader_live_auth_backend_for_test(
        crate::app_services::ctrader_live_auth::StubCTraderLiveAuthBackend::success(
            crate::app_services::ctrader_live_auth::CTraderLiveAuthResult {
                callback_port: 43001,
                authorization_code: "code-123".to_string(),
                token_bundle: crate::app_services::ctrader_auth::CTraderTokenBundle {
                    access_token: "access".to_string(),
                    refresh_token: "refresh".to_string(),
                    token_type: "bearer".to_string(),
                    expires_in: 3600,
                    scope: "trading".to_string(),
                    created_at_unix: current_unix_seconds().expect("current unix time"),
                },
            },
        ),
    );
    session.set_ctrader_account_discovery_backend_for_test(
        crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::success(
            crate::app_services::ctrader_live_auth::CTraderAccountDiscoveryResult {
                access_token: "access".to_string(),
                permission_scope: "SCOPE_TRADE".to_string(),
                accounts: vec![
                    crate::app_services::ctrader_auth::CTraderDiscoveredAccount {
                        account_id: "101".to_string(),
                        broker_title: "Broker A".to_string(),
                        account_name: "Primary Demo".to_string(),
                        trader_login: Some(500101),
                        is_live: Some(false),
                        enabled_for_execution: false,
                    },
                ],
            },
        ),
    );

    let (tx, _rx) = tokio::sync::mpsc::channel(8);
    let snapshot = session
        .start_ctrader_live_auth(tx)
        .expect("live auth should start");
    assert_eq!(
        snapshot.state,
        crate::app_services::ctrader_auth::CTraderAuthState::ListeningForCallback
    );

    let mut completed = None;
    for _ in 0..20 {
        if let Some(snapshot) = session.poll_ctrader_live_auth() {
            completed = Some(snapshot);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let completed = completed.expect("poll should return completion");

    assert_eq!(
        completed.state,
        crate::app_services::ctrader_auth::CTraderAuthState::AccountsAvailable
    );
    assert!(completed.token_persisted);
    assert_eq!(completed.account_count, 1);
    assert_eq!(
        session.broker_settings_mut().ctrader.accounts[0].account_id,
        "101"
    );
}

#[test]
fn failed_ctrader_live_auth_reports_backend_error() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session.set_ctrader_live_auth_backend_for_test(
        crate::app_services::ctrader_live_auth::StubCTraderLiveAuthBackend::failure(
            "INVALID_CLIENT: cTrader rejected the OAuth application",
        ),
    );

    let (tx, _rx) = tokio::sync::mpsc::channel(8);
    session
        .start_ctrader_live_auth(tx)
        .expect("live auth should start");

    let mut completed = None;
    for _ in 0..20 {
        if let Some(snapshot) = session.poll_ctrader_live_auth() {
            completed = Some(snapshot);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let completed = completed.expect("poll should return failure");

    assert_eq!(
        completed.state,
        crate::app_services::ctrader_auth::CTraderAuthState::Failed
    );
    assert!(completed.status_line.contains("INVALID_CLIENT"));
}

#[test]
fn live_auth_completion_reports_account_discovery_failure() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session.set_ctrader_live_auth_backend_for_test(
        crate::app_services::ctrader_live_auth::StubCTraderLiveAuthBackend::success(
            crate::app_services::ctrader_live_auth::CTraderLiveAuthResult {
                callback_port: 43001,
                authorization_code: "code-123".to_string(),
                token_bundle: fresh_ctrader_token_bundle("access", "refresh"),
            },
        ),
    );
    session.set_ctrader_account_discovery_backend_for_test(
        crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::failure(
            "INVALID_REQUEST: account list failed",
        ),
    );

    let (tx, _rx) = tokio::sync::mpsc::channel(8);
    session
        .start_ctrader_live_auth(tx)
        .expect("live auth should start");

    let mut completed = None;
    for _ in 0..20 {
        if let Some(snapshot) = session.poll_ctrader_live_auth() {
            completed = Some(snapshot);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let completed = completed.expect("poll should return discovery failure");

    assert_eq!(
        completed.state,
        crate::app_services::ctrader_auth::CTraderAuthState::Failed
    );
    assert!(completed.token_persisted);
    assert!(
        completed
            .status_line
            .contains("INVALID_REQUEST: account list failed")
    );
}

#[test]
fn clear_ctrader_saved_session_clears_restored_state() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session
        .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
        .expect("seed should succeed");
    session
        .restore_ctrader_session()
        .expect("restore should succeed");

    session
        .clear_ctrader_saved_session()
        .expect("clear should succeed");

    let restored = session
        .restore_ctrader_session()
        .expect("restore should succeed");
    assert!(restored.is_none());
}

#[test]
fn discover_ctrader_accounts_syncs_discovered_catalog_into_targets() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();
    session.broker_settings_mut().ctrader.accounts.push(
        crate::app_services::broker_config::BrokerAccountTarget {
            account_id: "101".to_string(),
            label: "Operator Primary".to_string(),
            enabled_for_execution: true,
        },
    );
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session
        .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
        .expect("seed should succeed");
    session
        .restore_ctrader_session()
        .expect("restore should succeed")
        .expect("snapshot should exist");
    session.set_ctrader_account_discovery_backend_for_test(
        crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::success(
            crate::app_services::ctrader_live_auth::CTraderAccountDiscoveryResult {
                access_token: "access".to_string(),
                permission_scope: "SCOPE_TRADE".to_string(),
                accounts: vec![
                    crate::app_services::ctrader_auth::CTraderDiscoveredAccount {
                        account_id: "101".to_string(),
                        broker_title: "Broker A".to_string(),
                        account_name: "Primary Live".to_string(),
                        trader_login: Some(500101),
                        is_live: Some(true),
                        enabled_for_execution: false,
                    },
                    crate::app_services::ctrader_auth::CTraderDiscoveredAccount {
                        account_id: "202".to_string(),
                        broker_title: "Broker B".to_string(),
                        account_name: "Secondary Demo".to_string(),
                        trader_login: Some(500202),
                        is_live: Some(false),
                        enabled_for_execution: false,
                    },
                ],
            },
        ),
    );

    let snapshot = session
        .discover_ctrader_accounts()
        .expect("discovery should succeed")
        .expect("snapshot should exist");

    assert_eq!(
        snapshot.state,
        crate::app_services::ctrader_auth::CTraderAuthState::AccountsAvailable
    );
    assert_eq!(snapshot.account_count, 2);
    assert_eq!(snapshot.enabled_target_count, 1);
    assert_eq!(session.broker_settings_mut().ctrader.accounts.len(), 2);
    assert!(session.broker_settings_mut().ctrader.accounts.iter().any(
        |account| account.account_id == "101"
            && account.label == "Operator Primary"
            && account.enabled_for_execution
    ));
    assert!(
        session
            .broker_settings_mut()
            .ctrader
            .accounts
            .iter()
            .any(|account| account.account_id == "202" && !account.enabled_for_execution)
    );
}

#[test]
fn discover_ctrader_accounts_uses_configured_demo_environment() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();
    session.broker_settings_mut().ctrader.environment = CTraderBrokerEnvironment::Demo;
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session
        .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
        .expect("seed should succeed");
    session
        .restore_ctrader_session()
        .expect("restore should succeed")
        .expect("snapshot should exist");

    let backend =
        crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::success(
            crate::app_services::ctrader_live_auth::CTraderAccountDiscoveryResult {
                access_token: "access".to_string(),
                permission_scope: "SCOPE_TRADE".to_string(),
                accounts: Vec::new(),
            },
        );
    session.set_ctrader_account_discovery_backend_for_test(backend.clone());

    session
        .discover_ctrader_accounts()
        .expect("discovery should succeed");

    let request = backend
        .last_request()
        .expect("stub backend should capture discovery request");
    assert_eq!(request.environment, CTraderEnvironment::Demo);
}

#[test]
fn discover_ctrader_accounts_requires_restored_token_session() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();

    let err = session
        .discover_ctrader_accounts()
        .expect_err("discovery should fail without a restored token");

    assert!(
        err.to_string()
            .contains("cTrader account discovery requires a restored token session")
    );
}

#[test]
fn discover_ctrader_accounts_requires_persisted_token_bundle() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session.set_ctrader_account_discovery_backend_for_test(
        crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::success(
            crate::app_services::ctrader_live_auth::CTraderAccountDiscoveryResult {
                access_token: "access".to_string(),
                permission_scope: "SCOPE_TRADE".to_string(),
                accounts: Vec::new(),
            },
        ),
    );
    let mut auth = crate::app_services::ctrader_auth::CTraderAuthSession::new(
        "client",
        "http://127.0.0.1:43001/callback",
    );
    auth.restore_from_storage(crate::app_services::ctrader_auth::CTraderTokenBundle {
        access_token: "access".to_string(),
        refresh_token: "refresh".to_string(),
        token_type: "bearer".to_string(),
        expires_in: 3600,
        scope: "trading".to_string(),
        created_at_unix: current_unix_seconds().expect("current unix time"),
    });
    session.ctrader_auth = Some(auth);

    let err = session
        .discover_ctrader_accounts()
        .expect_err("discovery should fail without persisted token bundle");

    assert!(
        err.to_string()
            .contains("cTrader account discovery requires a stored token bundle")
    );
}

#[test]
fn discover_ctrader_accounts_uses_selected_demo_environment() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();
    session.broker_settings_mut().ctrader.environment =
        crate::app_services::broker_config::CTraderBrokerEnvironment::Demo;
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session
        .seed_ctrader_token_bundle_for_test(fresh_ctrader_token_bundle("access", "refresh"))
        .expect("seed should succeed");
    session
        .restore_ctrader_session()
        .expect("restore should succeed")
        .expect("snapshot should exist");
    let backend =
        crate::app_services::ctrader_live_auth::StubCTraderAccountDiscoveryBackend::success(
            crate::app_services::ctrader_live_auth::CTraderAccountDiscoveryResult {
                access_token: "access".to_string(),
                permission_scope: "SCOPE_TRADE".to_string(),
                accounts: Vec::new(),
            },
        );
    session.set_ctrader_account_discovery_backend_for_test(backend.clone());

    session
        .discover_ctrader_accounts()
        .expect("discovery should succeed")
        .expect("snapshot should exist");

    let request = backend
        .last_request()
        .expect("stub should capture the discovery request");
    assert_eq!(request.environment, CTraderEnvironment::Demo);
}

#[test]
fn ctrader_runtime_request_refreshes_expired_token_bundle_before_use() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();
    session.broker_settings_mut().ctrader.accounts.push(
        crate::app_services::broker_config::BrokerAccountTarget {
            account_id: "101".to_string(),
            label: "Primary".to_string(),
            enabled_for_execution: true,
        },
    );
    let store = crate::app_services::secure_store::MemorySecretStoreBackend::default();
    session.set_ctrader_store_for_test(store.clone());
    session
        .seed_ctrader_token_bundle_for_test(crate::app_services::ctrader_auth::CTraderTokenBundle {
            access_token: "expired-access".to_string(),
            refresh_token: "expired-refresh".to_string(),
            token_type: "bearer".to_string(),
            expires_in: 60,
            scope: "trading".to_string(),
            created_at_unix: 1,
        })
        .expect("seed should succeed");
    session
        .restore_ctrader_session()
        .expect("restore should succeed")
        .expect("snapshot should exist");
    session.set_ctrader_live_auth_backend_for_test(
        crate::app_services::ctrader_live_auth::StubCTraderLiveAuthBackend::with_refresh_success(
            crate::app_services::ctrader_auth::CTraderTokenBundle {
                access_token: "fresh-access".to_string(),
                refresh_token: "fresh-refresh".to_string(),
                token_type: "bearer".to_string(),
                expires_in: 2628000,
                scope: "trading".to_string(),
                created_at_unix: 1_900_000_000,
            },
        ),
    );

    let request = session
        .build_ctrader_account_runtime_request()
        .expect("runtime request should refresh");

    assert_eq!(request.access_token, "fresh-access");
    let stored = session
        .ctrader_token_store
        .load_token_bundle()
        .expect("load should succeed")
        .expect("bundle should exist");
    assert_eq!(stored.access_token, "fresh-access");
    assert_eq!(stored.refresh_token, "fresh-refresh");
}

#[test]
fn ctrader_runtime_request_fails_closed_when_refresh_fails() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.broker_settings_mut().ctrader.client_id = "client".to_string();
    session.broker_settings_mut().ctrader.client_secret = "secret".to_string();
    session.broker_settings_mut().ctrader.redirect_uri =
        "http://127.0.0.1:43001/callback".to_string();
    session.broker_settings_mut().ctrader.accounts.push(
        crate::app_services::broker_config::BrokerAccountTarget {
            account_id: "101".to_string(),
            label: "Primary".to_string(),
            enabled_for_execution: true,
        },
    );
    session.set_ctrader_store_for_test(
        crate::app_services::secure_store::MemorySecretStoreBackend::default(),
    );
    session
        .seed_ctrader_token_bundle_for_test(crate::app_services::ctrader_auth::CTraderTokenBundle {
            access_token: "expired-access".to_string(),
            refresh_token: "expired-refresh".to_string(),
            token_type: "bearer".to_string(),
            expires_in: 60,
            scope: "trading".to_string(),
            created_at_unix: 1,
        })
        .expect("seed should succeed");
    session
        .restore_ctrader_session()
        .expect("restore should succeed")
        .expect("snapshot should exist");
    session.set_ctrader_live_auth_backend_for_test(
        crate::app_services::ctrader_live_auth::StubCTraderLiveAuthBackend::with_refresh_failure(
            "token refresh failed",
        ),
    );

    let err = session
        .build_ctrader_account_runtime_request()
        .expect_err("runtime request should fail when refresh fails");

    assert!(err.to_string().contains("token refresh failed"));
}

#[test]
fn refresh_runtime_skips_ctrader_probe_within_refresh_window() {
    let mut state = sample_state(DataSource::CTrader, "Connected");
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    let backend = crate::app_services::ctrader_account::StubCTraderAccountRuntimeBackend::success(
        crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
            trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                account_id: 712345,
                balance: 1000.0,
                leverage: Some(50.0),
                trader_login: Some(998877),
                account_type: Some("NETTED".to_string()),
                broker_name: Some("Demo Broker".to_string()),
                money_digits: 2,
                unrealized_pnl: 0.0,
            },
            reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                account_id: 712345,
                positions: Vec::new(),
                pending_orders: Vec::new(),
            },
            recent_deals: Vec::new(),
        },
    );
    session.set_ctrader_account_runtime_backend_for_test(backend.clone());
    session.handle_ctrader_connect_result(
        &mut state,
        crate::app_services::ctrader_account::CTraderAccountRuntimeSnapshot {
            trader: crate::app_services::ctrader_account::CTraderTraderSnapshot {
                account_id: 712345,
                balance: 1000.0,
                leverage: Some(50.0),
                trader_login: Some(998877),
                account_type: Some("NETTED".to_string()),
                broker_name: Some("Demo Broker".to_string()),
                money_digits: 2,
                unrealized_pnl: 0.0,
            },
            reconcile: crate::app_services::ctrader_account::CTraderReconcileSnapshot {
                account_id: 712345,
                positions: Vec::new(),
                pending_orders: Vec::new(),
            },
            recent_deals: Vec::new(),
        },
    );

    session
        .refresh_runtime(&mut state)
        .expect("refresh within throttle window should succeed");

    assert!(backend.last_request().is_none());
}

#[test]
fn start_ctrader_bootstrap_batch_rejects_concurrent_request() {
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session.bootstrap_handle = Some(std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }));
    let (tx, _rx) = tokio::sync::mpsc::channel(4);

    let err = session
        .start_ctrader_bootstrap_batch(
            std::path::PathBuf::from("data"),
            vec!["EURUSD".to_string()],
            vec!["M15".to_string()],
            1,
            tx,
        )
        .expect_err("concurrent bootstrap should be rejected");

    assert!(err.to_string().contains("already running"));
    if let Some(handle) = session.bootstrap_handle.take() {
        let _ = handle.join();
    }
}

fn sample_prop_firm_order() -> CTraderNewOrderRequest {
    CTraderNewOrderRequest {
        account_id: 101,
        symbol_id: 1,
        order_type: CTraderOrderType::Market,
        trade_side: CTraderTradeSide::Buy,
        volume: 100000,
        limit_price: None,
        stop_price: None,
        time_in_force: Some(CTraderTimeInForce::ImmediateOrCancel),
        expiration_timestamp_ms: None,
        stop_loss: Some(1.05000),
        take_profit: None,
        comment: None,
        base_slippage_price: None,
        slippage_in_points: None,
        label: None,
        position_id: None,
        client_order_id: None,
        relative_stop_loss: None,
        relative_take_profit: None,
        guaranteed_stop_loss: None,
        trailing_stop_loss: None,
        stop_trigger_method: None,
    }
}

// TODO(real-data): the risk-gate tests below currently rely on the
// `baked_in_default` symbol-metadata fallback (EURUSD / USDJPY) inside
// forex_core::symbol_metadata to resolve pip values without hitting a
// live cTrader connection. Once the cTrader bootstrap writes the
// symbol-metadata JSON to disk in CI, replace these fixtures with a
// loader that reads the real broker payload for the symbol/timeframe.

// Tests that mutate `FOREX_BOT_PROP_ACCOUNT_CURRENCY` /
// `FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE` MUST run in series — cargo
// runs tests in a single process with a multi-threaded default pool,
// and parallel env mutation is racy. The same `OnceLock<Mutex<()>>`
// pattern that gates `FOREX_AI_LICENSE_PATH` in
// `ui/wizard/welcome.rs::tests::env_lock` keeps these serial without
// requiring the operator to set RUST_TEST_THREADS=1 or pull in a new
// dev-dependency.
fn prop_firm_env_lock() -> &'static std::sync::Mutex<()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

/// RAII guard returned by `install_prop_firm_test_env`. Holds the
/// suite-wide env-mutation mutex AND captures the prior values of the
/// env vars it sets, restoring them on drop. This makes each test's
/// env mutation invisible to other tests, even when the cargo runner
/// re-orders or interleaves them.
struct PropFirmEnvGuard {
    // Keeps the suite-wide env mutex held for the duration of the test.
    // `'static` lifetime is sound because the mutex is owned by a
    // `OnceLock`-backed singleton that lives for the whole process.
    _lock: std::sync::MutexGuard<'static, ()>,
    prior_account_currency: Option<String>,
    prior_quote_rate: Option<String>,
}

impl Drop for PropFirmEnvGuard {
    fn drop(&mut self) {
        // SAFETY: env mutation is gated by the mutex held in `_lock`,
        // so no other test mutates these vars in parallel. Required
        // because `set_var` / `remove_var` are marked unsafe in
        // edition 2024.
        unsafe {
            match self.prior_account_currency.take() {
                Some(v) => std::env::set_var("FOREX_BOT_PROP_ACCOUNT_CURRENCY", v),
                None => std::env::remove_var("FOREX_BOT_PROP_ACCOUNT_CURRENCY"),
            }
            match self.prior_quote_rate.take() {
                Some(v) => std::env::set_var("FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE", v),
                None => std::env::remove_var("FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE"),
            }
        }
    }
}

/// Acquire the env mutex, snapshot existing values, and apply the
/// prop-firm risk-gate test fixture. The returned guard must be bound
/// to a local (e.g. `let _guard = install_prop_firm_test_env();`) so
/// it lives for the whole test body — dropping it eagerly would
/// release the lock and let another thread clobber the env mid-test.
#[must_use = "bind the returned guard to a local; dropping it eagerly releases the env mutex"]
fn install_prop_firm_test_env() -> PropFirmEnvGuard {
    // `.unwrap_or_else(|e| e.into_inner())` so a poisoned mutex (from
    // a panicking sibling test) doesn't cascade-fail the rest of the
    // suite — we still get serialized access.
    let lock = prop_firm_env_lock()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let prior_account_currency = std::env::var("FOREX_BOT_PROP_ACCOUNT_CURRENCY").ok();
    let prior_quote_rate = std::env::var("FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE").ok();
    // SAFETY: tests in this binary share env; risk-gate tests must set
    // FOREX_BOT_PROP_ACCOUNT_CURRENCY explicitly because production
    // refuses to synthesize a default account currency. Mutation is
    // gated by the mutex held in `lock` above.
    unsafe {
        std::env::set_var("FOREX_BOT_PROP_ACCOUNT_CURRENCY", "USD");
        // EURJPY-style cross test below requires this; USDJPY only uses
        // base==account, so it's safe to leave the rate present.
        std::env::set_var("FOREX_BOT_PROP_QUOTE_TO_ACCOUNT_RATE", "0.0067");
    }
    PropFirmEnvGuard {
        _lock: lock,
        prior_account_currency,
        prior_quote_rate,
    }
}

#[test]
fn prop_firm_gate_blocks_order_without_stop_loss() {
    let _guard = install_prop_firm_test_env();
    let mut order = sample_prop_firm_order();
    order.stop_loss = None;
    let risk = RiskConfig::default();
    let err = prop_firm_pre_trade_check(&risk, &order, 10000.0, 10000.0, 10000.0, 4, "EURUSD")
        .unwrap_err();
    assert!(err.to_string().contains("missing stop_loss"));
}

#[test]
fn prop_firm_gate_blocks_when_daily_drawdown_breached() {
    let _guard = install_prop_firm_test_env();
    let order = sample_prop_firm_order();
    let risk = RiskConfig::default();
    let err = prop_firm_pre_trade_check(&risk, &order, 9500.0, 10000.0, 10000.0, 4, "EURUSD")
        .unwrap_err();
    assert!(err.to_string().contains("Daily drawdown limit reached"));
    assert!(err.to_string().contains("current 5.00% >= max 4.00%"));
}

#[test]
fn prop_firm_gate_respects_jpy_pip_precision() {
    let _guard = install_prop_firm_test_env();
    let mut order = sample_prop_firm_order();
    order.limit_price = Some(150.00);
    order.stop_loss = Some(149.50); // 50 pips in JPY (2 digits)
    // The default fixture's `volume = 100000` represents 1000 standard lots
    // which would (correctly) trip the new hard risk-per-trade gate even
    // for a tight 50-pip stop. This test exists ONLY to verify that
    // pip-position 2 (JPY) is interpreted as 50 pips, not 5000, so widen
    // the risk gate so it can't reject for a separate reason. Without
    // this widening, the test would assert the wrong invariant if
    // someone later regressed pip_position to 4 (5000 pips × 1 std lot
    // would still fail, but for the wrong reason).
    order.volume = 100; // 0.01 std lot = 1 micro lot
    let risk = RiskConfig {
        risk_per_trade: 1.0, // 100% — disable the per-trade size gate
        ..RiskConfig::default()
    };
    // Should pass under 2-digit precision (50 pips).
    assert!(
        prop_firm_pre_trade_check(&risk, &order, 10000.0, 10000.0, 10000.0, 2, "USDJPY").is_ok()
    );
    // 4-digit precision would amplify pip_distance by 100×; with the same
    // lot size and pip value that's $50,000 of risk on a $10,000 account,
    // still > 100% so it must reject.
    assert!(
        prop_firm_pre_trade_check(&risk, &order, 10000.0, 10000.0, 10000.0, 4, "USDJPY").is_err()
    );
}

#[test]
fn prop_firm_gate_blocks_when_total_drawdown_breached() {
    let _guard = install_prop_firm_test_env();
    let order = sample_prop_firm_order();
    let risk = RiskConfig::default();
    // Set day_start_equity equal to account_equity so daily DD is 0%, forcing it to hit total DD rule
    let err = prop_firm_pre_trade_check(&risk, &order, 8900.0, 10000.0, 8900.0, 4, "EURUSD")
        .unwrap_err();
    assert!(err.to_string().contains("Total drawdown limit reached"));
    assert!(err.to_string().contains("current 11.00% >= max 7.00%"));
}

#[test]
fn prop_firm_gate_passes_valid_order_within_limits() {
    let _guard = install_prop_firm_test_env();
    let mut order = sample_prop_firm_order();
    // Keep the size sane: the default fixture's 1000-lot volume would
    // (correctly) be rejected by the real-pip risk-per-trade gate.
    order.volume = 1; // 0.01 micro-lot
    order.limit_price = Some(1.10000);
    let risk = RiskConfig::default();
    assert!(
        prop_firm_pre_trade_check(&risk, &order, 10100.0, 10000.0, 10000.0, 4, "EURUSD").is_ok()
    );
}

#[test]
fn prop_firm_gate_respects_disabled_stop_loss_requirement() {
    let _guard = install_prop_firm_test_env();
    let mut order = sample_prop_firm_order();
    order.stop_loss = None;
    let risk = RiskConfig {
        require_stop_loss: false,
        ..RiskConfig::default()
    };
    assert!(
        prop_firm_pre_trade_check(&risk, &order, 10000.0, 10000.0, 10000.0, 4, "EURUSD").is_ok()
    );
}

#[test]
fn prop_firm_gate_rejects_unknown_symbol_without_synthetic_fallback() {
    let _guard = install_prop_firm_test_env();
    let order = sample_prop_firm_order();
    let risk = RiskConfig::default();
    // Empty symbol must be rejected — the old code silently used
    // `infer_market_cost_profile("", "", …)` and the EURUSD default,
    // producing a synthetic pip value. That is not allowed any more.
    let err = prop_firm_pre_trade_check(&risk, &order, 10000.0, 10000.0, 10000.0, 4, "")
        .unwrap_err();
    assert!(
        err.to_string().contains("symbol name was not supplied")
            || err.to_string().contains("Daily drawdown")
            || err.to_string().contains("Total drawdown"),
        "unexpected error: {err}"
    );
}

#[test]
fn prop_firm_gate_rejects_when_account_currency_unset() {
    // Deliberately clear the env to verify the gate refuses to size
    // an order rather than falling back to "USD". Reuse the same
    // `install_prop_firm_test_env` guard so this test takes the
    // suite-wide env mutex — otherwise a sibling test running in
    // parallel can re-set `FOREX_BOT_PROP_ACCOUNT_CURRENCY` to "USD"
    // between our `remove_var` below and the gate's `env::var` read,
    // which is the original flake. The guard's `Drop` restores
    // whatever value (if any) was in the env before this test ran.
    let _guard = install_prop_firm_test_env();
    // SAFETY: env mutation is gated by the mutex held in `_guard`.
    unsafe {
        std::env::remove_var("FOREX_BOT_PROP_ACCOUNT_CURRENCY");
    }
    let order = sample_prop_firm_order();
    let risk = RiskConfig::default();
    let err = prop_firm_pre_trade_check(&risk, &order, 10000.0, 10000.0, 10000.0, 4, "EURUSD")
        .unwrap_err();
    assert!(
        err.to_string().contains("FOREX_BOT_PROP_ACCOUNT_CURRENCY")
            || err.to_string().contains("symbol metadata"),
        "unexpected error: {err}"
    );
}

// ── Risky Mode integration (research §4–§5 + operator directive
// 2026-05-15). These tests cover the `TradingSession` ⇄
// `RiskyModeManager` wiring; the manager's own per-stage maths is
// covered exhaustively by `forex_core::domain::risky_mode::tests`.

/// Build a default Risky Mode config with the autonomous-only contract
/// explicitly accepted (the test-harness analogue of the operator
/// ticking the wizard's §7.1 acknowledgement). `RiskyModeManager::new`
/// rejects construction without this flag, so every Risky-Mode
/// integration test in this file routes through this helper.
fn signed_risky_mode_config() -> forex_core::RiskyModeConfig {
    let mut cfg = forex_core::RiskyModeConfig::default();
    cfg.autonomous_only_contract_accepted = true;
    cfg
}

#[test]
fn risky_mode_gate_rejects_order_when_kill_switch_tripped() {
    // Step-by-step recreation of the integration path:
    //   1. Enable Risky Mode on the session.
    //   2. Trip the per-day kill switch via record_trade_outcome
    //      (exceeds the stage daily-loss cap).
    //   3. Assert that `risky_mode_manager().check_trade_allowed`
    //      now returns Err — the same call the production
    //      `execute_ctrader_order` path makes before the prop-firm
    //      gate.
    let mut session = TradingSession::new();
    session
        .enable_risky_mode(signed_risky_mode_config(), 100.0)
        .expect("enable_risky_mode");
    assert!(session.risky_mode_active());

    // Drive the per-day kill switch by accumulating a loss larger
    // than the stage's `daily_loss_cap_fraction * bankroll`.
    {
        let rm = session.risky_mode_manager_mut().expect("manager");
        let stage = *rm.current_stage();
        let cap_usd = stage.daily_loss_cap_fraction * rm.current_bankroll_usd();
        rm.record_trade_outcome(-(cap_usd + 1.0));
    }

    // Now the same gate that production calls inside
    // execute_ctrader_order must reject every new order.
    let rm = session.risky_mode_manager().expect("manager still set");
    let result = rm.check_trade_allowed(0.5_f64, 10.0_f64, 30.0_f64);
    assert_eq!(result, Err(forex_core::KillSwitchTier::PerDay));
}

#[test]
fn halt_button_also_trips_risky_mode_kill_switch() {
    // The operator hits the red HALT button. T-Manual must trip
    // BOTH the session.halt_state AND (when Risky Mode is armed)
    // the Risky Mode sticky manual halt — research §5.5.
    let mut state = sample_state(DataSource::CTrader, "Connected");
    let mut session = TradingSession::with_configured_adapter_for_test(TradingAdapterKind::CTrader);
    session
        .enable_risky_mode(signed_risky_mode_config(), 100.0)
        .expect("enable_risky_mode");
    assert!(
        session
            .risky_mode_manager()
            .expect("manager")
            .last_kill_switch_trip()
            .is_none(),
        "fresh manager must not have any kill-switch trips"
    );

    let summary = session.trip_manual_halt(&mut state);
    // Sanity — halt machinery itself worked (no broker connected
    // means zero positions/orders closed; the in-memory flag is
    // authoritative).
    assert!(session.is_halted());
    assert_eq!(summary.positions_closed, 0);
    assert_eq!(summary.orders_cancelled, 0);

    let trip = session
        .risky_mode_manager()
        .expect("manager")
        .last_kill_switch_trip();
    let (tier, _ts) = trip.expect("HALT must propagate to Risky Mode manager");
    assert_eq!(tier, forex_core::KillSwitchTier::Manual);

    // Defence-in-depth: Risky Mode's check_trade_allowed must now
    // reject every order, regardless of size/SL/TP — research §5.5
    // says the manual halt is sticky until clear_halt fires.
    let result = session
        .risky_mode_manager()
        .expect("manager")
        .check_trade_allowed(0.1_f64, 10.0_f64, 30.0_f64);
    assert_eq!(result, Err(forex_core::KillSwitchTier::Manual));
}

#[test]
fn closed_trade_advances_risky_mode_bankroll() {
    // The close-path realises a profit large enough to cross a
    // stage boundary. record_trade_outcome must (a) update the
    // bankroll and (b) re-locate the stage cursor. This is the
    // mechanism the production `execute_ctrader_request` close-
    // outcome hook invokes on every ClosePosition fill.
    let mut session = TradingSession::new();
    let cfg = signed_risky_mode_config();
    // Start at the lower bound of stage 0 so we can promote after
    // a single profitable close.
    let stage1_lower = cfg.stages[1].bankroll_lower_usd;
    let starting_bankroll = cfg.stages[0].bankroll_lower_usd + 0.01;
    session
        .enable_risky_mode(cfg.clone(), starting_bankroll)
        .expect("enable_risky_mode");
    let snapshot_before = session.risky_mode_state().expect("state");
    assert_eq!(snapshot_before.current_stage_idx, 0);

    // Realised PnL that lands the bankroll inside stage 1's range.
    let pnl_to_stage1 = (stage1_lower - starting_bankroll) + 0.5;
    {
        let rm = session.risky_mode_manager_mut().expect("manager");
        rm.record_trade_outcome(pnl_to_stage1);
    }
    let snapshot_after = session.risky_mode_state().expect("state");
    assert!(
        snapshot_after.current_stage_idx >= 1,
        "expected stage advancement past S1, got {}",
        snapshot_after.current_stage_idx
    );
    assert!(
        snapshot_after.current_bankroll_usd >= stage1_lower,
        "bankroll did not advance: {} < {}",
        snapshot_after.current_bankroll_usd,
        stage1_lower
    );

    // disable_risky_mode tears down cleanly.
    session.disable_risky_mode();
    assert!(!session.risky_mode_active());
    assert!(session.risky_mode_state().is_none());
}

// ── Bot decision overlays (audit gap #11) ─────────────────────────────

#[test]
fn record_bot_decision_appends_to_buffer() {
    let mut session = TradingSession::new();
    assert_eq!(session.bot_decision_buffer_len(), 0);
    session.record_bot_decision(BotDecisionEntry {
        symbol: "EURUSD".to_string(),
        side: BotDecisionSide::Buy,
        price: 1.0843,
        timestamp_ms: 1_700_000_000_000,
        label: "BUY".to_string(),
        source: BotDecisionSource::Manual,
        confidence: None,
    });
    assert_eq!(session.bot_decision_buffer_len(), 1);
}

#[test]
fn bot_decisions_for_filters_by_symbol() {
    let mut session = TradingSession::new();
    for sym in ["EURUSD", "GBPUSD", "EURUSD"] {
        session.record_bot_decision(BotDecisionEntry {
            symbol: sym.to_string(),
            side: BotDecisionSide::Buy,
            price: 1.0,
            timestamp_ms: 1_700_000_000_000,
            label: "x".to_string(),
            source: BotDecisionSource::Manual,
            confidence: None,
        });
    }
    assert_eq!(session.bot_decisions_for("EURUSD").len(), 2);
    assert_eq!(session.bot_decisions_for("GBPUSD").len(), 1);
    assert_eq!(session.bot_decisions_for("USDJPY").len(), 0);
}

#[test]
fn bot_decision_buffer_caps_at_capacity_fifo() {
    let mut session = TradingSession::new();
    let cap = crate::app_services::trading::BOT_DECISION_BUFFER_CAPACITY;
    for i in 0..(cap + 5) {
        session.record_bot_decision(BotDecisionEntry {
            symbol: "EURUSD".to_string(),
            side: BotDecisionSide::Buy,
            price: 1.0 + (i as f64) * 0.0001,
            timestamp_ms: 1_700_000_000_000 + i as i64,
            label: format!("entry-{i}"),
            source: BotDecisionSource::Manual,
            confidence: None,
        });
    }
    // The buffer must hold exactly the capacity and the oldest 5
    // entries must be the ones dropped (FIFO).
    assert_eq!(session.bot_decision_buffer_len(), cap);
    let earliest = session.bot_decisions_for("EURUSD")[0].label.clone();
    assert_eq!(earliest, "entry-5", "oldest 5 entries must be dropped FIFO");
}

#[test]
fn bot_decisions_to_overlays_maps_timestamps_to_nearest_candle() {
    use crate::app_services::trading::ChartCandle;
    let mut session = TradingSession::new();

    // 4 candles spaced 1 minute apart.
    let candles = vec![
        ChartCandle { timestamp: Some(1_700_000_000_000), open: 1.0, high: 1.0, low: 1.0, close: 1.0, volume: 0.0 },
        ChartCandle { timestamp: Some(1_700_000_060_000), open: 1.0, high: 1.0, low: 1.0, close: 1.0, volume: 0.0 },
        ChartCandle { timestamp: Some(1_700_000_120_000), open: 1.0, high: 1.0, low: 1.0, close: 1.0, volume: 0.0 },
        ChartCandle { timestamp: Some(1_700_000_180_000), open: 1.0, high: 1.0, low: 1.0, close: 1.0, volume: 0.0 },
    ];

    // Decision at exactly candle 2's timestamp → maps to index 2.
    session.record_bot_decision(BotDecisionEntry {
        symbol: "EURUSD".to_string(),
        side: BotDecisionSide::Buy,
        price: 1.0843,
        timestamp_ms: 1_700_000_120_000,
        label: "BUY".to_string(),
        source: BotDecisionSource::Manual,
        confidence: None,
    });
    // Decision 30s past candle 1 → still maps to candle 1 (largest
    // <= target; we never paint on a future candle).
    session.record_bot_decision(BotDecisionEntry {
        symbol: "EURUSD".to_string(),
        side: BotDecisionSide::Sell,
        price: 1.0855,
        timestamp_ms: 1_700_000_090_000,
        label: "SELL".to_string(),
        source: BotDecisionSource::Ai,
        confidence: Some(0.74),
    });
    // Decision before the first candle — must be dropped.
    session.record_bot_decision(BotDecisionEntry {
        symbol: "EURUSD".to_string(),
        side: BotDecisionSide::Flat,
        price: 1.07,
        timestamp_ms: 1_699_999_990_000,
        label: "OLD".to_string(),
        source: BotDecisionSource::Manual,
        confidence: None,
    });
    // Different-symbol decision — must not appear in EURUSD overlays.
    session.record_bot_decision(BotDecisionEntry {
        symbol: "GBPUSD".to_string(),
        side: BotDecisionSide::Buy,
        price: 1.2654,
        timestamp_ms: 1_700_000_120_000,
        label: "GBP".to_string(),
        source: BotDecisionSource::Manual,
        confidence: None,
    });

    let overlays = session.bot_decisions_to_overlays("EURUSD", &candles);
    assert_eq!(
        overlays.len(),
        2,
        "expected 2 EURUSD overlays (the pre-window OLD entry and the GBPUSD entry must drop)"
    );

    // Order is insertion-order (== chronological by record_bot_decision call).
    assert_eq!(overlays[0].label, "BUY");
    assert_eq!(overlays[0].candle_index, 2);
    assert_eq!(overlays[1].label, "SELL");
    assert_eq!(overlays[1].candle_index, 1); // 30s past candle 1, before candle 2
}

#[test]
fn bot_decisions_to_overlays_returns_empty_when_no_candles() {
    let mut session = TradingSession::new();
    session.record_bot_decision(BotDecisionEntry {
        symbol: "EURUSD".to_string(),
        side: BotDecisionSide::Buy,
        price: 1.0,
        timestamp_ms: 0,
        label: "x".to_string(),
        source: BotDecisionSource::Manual,
        confidence: None,
    });
    assert!(session.bot_decisions_to_overlays("EURUSD", &[]).is_empty());
}
