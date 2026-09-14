//! Synthetic protocol sequences exercise the production collector without a
//! socket, credentials, sleep, account runtime, or broker side effect.
use super::*;
use crate::app_services::ctrader_historical_admission::CTraderMonotonicClock;
use std::cell::Cell;
use std::collections::VecDeque;
use std::rc::Rc;

#[derive(Clone)]
struct Clock(Rc<Cell<Instant>>);

impl Clock {
    fn new() -> Self {
        Self(Rc::new(Cell::new(Instant::now())))
    }
    fn advance(&self, duration: Duration) {
        self.0.set(self.0.get() + duration);
    }
}

impl CTraderMonotonicClock for Clock {
    fn now(&self) -> Instant {
        self.0.get()
    }
}

struct SequenceIo {
    clock: Clock,
    frames: VecDeque<(Duration, std::result::Result<Message, tungstenite::Error>)>,
    sent: Vec<Message>,
    reads: usize,
    send_duration: Duration,
    fail_send: bool,
}

impl SequenceIo {
    fn new(clock: Clock, frames: Vec<Message>) -> Self {
        Self {
            clock,
            frames: frames
                .into_iter()
                .map(|frame| (Duration::ZERO, Ok(frame)))
                .collect(),
            sent: Vec::new(),
            reads: 0,
            send_duration: Duration::ZERO,
            fail_send: false,
        }
    }

    fn execution_sends(&self) -> usize {
        self.sent.iter().filter(|frame| match frame {
            Message::Text(text) => parse_open_api_envelope(text).is_ok_and(|envelope|
                envelope.payload_type == crate::app_services::ctrader_messages::CTRADER_OA_NEW_ORDER_REQUEST_PAYLOAD_TYPE),
            _ => false,
        }).count()
    }
}

impl ImmediateExecutionIo for SequenceIo {
    fn send_frame(&mut self, frame: Message) -> std::result::Result<(), tungstenite::Error> {
        self.sent.push(frame);
        self.clock.advance(self.send_duration);
        if self.fail_send {
            return Err(tungstenite::Error::Io(std::io::Error::other(
                "possibly written",
            )));
        }
        Ok(())
    }

    fn read_frame(&mut self) -> std::result::Result<Message, tungstenite::Error> {
        self.reads += 1;
        let (elapsed, result) = self.frames.pop_front().expect("unexpected extra read");
        self.clock.advance(elapsed);
        result
    }
}

fn request(order_type: CTraderOrderType) -> CTraderExecutionRuntimeRequest {
    sample_runtime_request(CTraderExecutionRequest::NewOrder(Box::new(
        CTraderNewOrderRequest {
            account_id: 712345,
            symbol_id: 14,
            order_type,
            trade_side: crate::app_services::ctrader_messages::CTraderTradeSide::Buy,
            volume: 10000000,
            limit_price: None,
            stop_price: None,
            time_in_force: Some(CTraderTimeInForce::ImmediateOrCancel),
            expiration_timestamp_ms: None,
            stop_loss: None,
            take_profit: None,
            comment: None,
            base_slippage_price: None,
            slippage_in_points: None,
            label: None,
            position_id: None,
            client_order_id: Some("entry-terminal-1".to_owned()),
            relative_stop_loss: None,
            relative_take_profit: None,
            guaranteed_stop_loss: None,
            trailing_stop_loss: None,
            stop_trigger_method: None,
        },
    )))
}

fn wire(order_type: CTraderOrderType, accepted: bool) -> Value {
    let mut wire = single_opening_execution_wire();
    wire["clientMsgId"] = serde_json::json!("terminal-1");
    wire["payload"]["order"]["clientOrderId"] = serde_json::json!("entry-terminal-1");
    wire["payload"]["order"]["orderType"] = serde_json::json!(match order_type {
        CTraderOrderType::Market => 1,
        CTraderOrderType::MarketRange => 5,
        CTraderOrderType::Limit => 2,
        CTraderOrderType::Stop => 3,
        _ => unreachable!(),
    });
    if accepted {
        wire["payload"]["executionType"] = serde_json::json!(2);
        wire["payload"]["order"]["orderStatus"] = serde_json::json!(1);
        wire["payload"]["order"]["executedVolume"] = serde_json::json!(0);
        wire["payload"]["order"]["positionId"] = Value::Null;
        wire["payload"]["position"] = Value::Null;
        wire["payload"]["deal"] = Value::Null;
    }
    wire
}

fn text(wire: Value) -> Message {
    Message::Text(wire.to_string().into())
}

fn execute(
    request: &CTraderExecutionRuntimeRequest,
    clock: Clock,
    io: &mut SequenceIo,
    timeout: Duration,
) -> Result<CTraderExecutionOutcome> {
    let budget = CTraderOperationBudget::new(clock, timeout, None)?;
    let message = request.request.to_message("terminal-1");
    ProductionCTraderExecutionBackend::execute_authenticated_once(request, || {
        collect_immediate_execution(request, &message, &budget, io)
    })
}

#[test]
fn market_and_market_range_wait_for_owned_fill_after_acceptance() {
    for kind in [CTraderOrderType::Market, CTraderOrderType::MarketRange] {
        let request = request(kind);
        let mut unrelated = wire(kind, false);
        unrelated["clientMsgId"] = serde_json::json!("manual-other-order");
        unrelated["payload"]["order"]["orderId"] = serde_json::json!(9999);
        unrelated["payload"]["deal"]["orderId"] = serde_json::json!(9999);
        unrelated["payload"]["order"]["clientOrderId"] = serde_json::json!("manual-unrelated");
        let mut filled = wire(kind, false);
        // Optional correlation on follow-up: the previously bound broker order
        // and account, not a wildcard message id, establishes ownership.
        filled.as_object_mut().unwrap().remove("clientMsgId");
        let clock = Clock::new();
        let mut io = SequenceIo::new(
            clock.clone(),
            vec![
                text(wire(kind, true)),
                text(serde_json::json!({"payloadType":51,"payload":{}})),
                text(unrelated),
                Message::Ping(vec![1, 2].into()),
                Message::Binary(filled.to_string().into_bytes().into()),
            ],
        );
        let result = execute(&request, clock, &mut io, Duration::from_secs(30)).unwrap();
        assert_eq!(result.status, CTraderExecutionStatus::Filled);
        assert!(result.opening_fill_evidence.is_some());
        assert_eq!(io.execution_sends(), 1);
        assert_eq!(io.reads, 5);
        assert!(matches!(&io.sent[1], Message::Pong(payload) if payload.as_ref() == [1, 2]));
    }
}

#[test]
fn direct_filled_response_keeps_existing_single_fill_proof() {
    let clock = Clock::new();
    let mut io = SequenceIo::new(
        clock.clone(),
        vec![text(wire(CTraderOrderType::Market, false))],
    );
    let result = execute(
        &request(CTraderOrderType::Market),
        clock,
        &mut io,
        Duration::from_secs(30),
    )
    .unwrap();
    assert!(result.opening_fill_evidence.is_some());
    assert_eq!(io.execution_sends(), 1);
}

#[test]
fn durable_uncertainty_clears_only_with_the_exact_single_opening_and_keeps_linkage() {
    use crate::app_services::account_risk::{AccountRiskIdentity, AccountRiskRegistry};
    let root = std::env::temp_dir().join(format!(
        "neoethos-proven-entry-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let identity = AccountRiskIdentity::new("demo", 712345, "USD").unwrap();
    let registry = AccountRiskRegistry::new();
    let shared = registry.acquire_entry(identity.clone(), &root).unwrap();
    let request = request(CTraderOrderType::Market);
    {
        let mut state = shared.lock().unwrap();
        state.prepare_day(20_260_908).unwrap();
        state.try_reserve_entry(20_260_908, None).unwrap();
        state
            .begin_submission(
                20_260_908,
                &request,
                &std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            )
            .unwrap();
    }
    let clock = Clock::new();
    let mut io = SequenceIo::new(
        clock.clone(),
        vec![text(wire(CTraderOrderType::Market, false))],
    );
    let outcome = execute(&request, clock, &mut io, Duration::from_secs(30)).unwrap();
    let opening = outcome
        .opening_fill_evidence
        .as_ref()
        .expect("actual parser single-opening evidence");
    let mut state = shared.lock().unwrap();
    assert!(
        state
            .confirm_verified_opening("different-client", opening)
            .is_err()
    );
    assert_eq!(state.unresolved_client_order_id(), Some("entry-terminal-1"));
    state
        .confirm_verified_opening("entry-terminal-1", opening)
        .unwrap();
    assert_eq!(state.unresolved_client_order_id(), None);
    assert_eq!(state.entries_today(), 1);
    drop(state);
    let restored = AccountRiskRegistry::new()
        .acquire_entry(identity, &root)
        .unwrap();
    assert_eq!(restored.lock().unwrap().unresolved_client_order_id(), None);
    let saved: Value = serde_json::from_slice(
        &std::fs::read(
            root.join("runtime")
                .join("account-entry")
                .join("demo-712345.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        saved["lastVerifiedEntry"]["positionId"],
        opening.position_id()
    );
    assert_eq!(saved["lastVerifiedEntry"]["dealId"], opening.deal_id());
    assert_eq!(
        saved["lastVerifiedEntry"]["intent"]["clientOrderId"],
        "entry-terminal-1"
    );
    assert_eq!(io.execution_sends(), 1);
    // This tests checkpoint/collector behavior, not full live admission, restart
    // restoration of position ownership, or real broker execution.
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn netted_filled_order_reference_survives_restart_without_claiming_opening_or_resend() {
    use crate::app_services::account_risk::{AccountRiskIdentity, AccountRiskRegistry};
    let root = std::env::temp_dir().join(format!(
        "neoethos-netted-intent-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let identity = AccountRiskIdentity::new("demo", 712345, "USD").unwrap();
    let shared = AccountRiskRegistry::new()
        .acquire_entry(identity.clone(), &root)
        .unwrap();
    let request = request(CTraderOrderType::Market);
    {
        let mut state = shared.lock().unwrap();
        state.prepare_day(20_260_908).unwrap();
        state.try_reserve_entry(20_260_908, None).unwrap();
        state
            .begin_submission(
                20_260_908,
                &request,
                &std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            )
            .unwrap();
    }
    let mut filled = wire(CTraderOrderType::Market, false);
    filled["payload"]["order"]["closingOrder"] = serde_json::json!(true);
    filled["payload"]["position"]["tradeData"]["tradeSide"] = serde_json::json!(2);
    filled["payload"]["deal"]["closePositionDetail"] = serde_json::json!({
        "entryPrice":1.09876,"grossProfit":-1000,"swap":-20,
        "commission":-40,"balance":9999940,"moneyDigits":2
    });
    let clock = Clock::new();
    let mut io = SequenceIo::new(clock.clone(), vec![text(filled)]);
    let outcome = execute(&request, clock, &mut io, Duration::from_secs(30)).unwrap();
    assert_eq!(outcome.status, CTraderExecutionStatus::Filled);
    assert_eq!(outcome.deal_closes_position, Some(true));
    assert!(outcome.opening_fill_evidence.is_none());
    assert_eq!(io.execution_sends(), 1);
    {
        let mut state = shared.lock().unwrap();
        for case in ["account", "environment", "client", "symbol", "side"] {
            let mut foreign = outcome.clone();
            let mut environment = request.environment;
            let mut client_id = "entry-terminal-1";
            match case {
                "account" => foreign.account_id = 99,
                "environment" => environment = CTraderEnvironment::Live,
                "client" => client_id = "another-intent",
                "symbol" => foreign.symbol_id = Some(99),
                _ => foreign.trade_side = Some("SELL".into()),
            }
            assert!(
                state
                    .record_unresolved_outcome(client_id, environment, &foreign)
                    .is_err(),
                "{case}"
            );
        }
        state
            .record_unresolved_outcome("entry-terminal-1", request.environment, &outcome)
            .unwrap();
        assert_eq!(state.unresolved_client_order_id(), Some("entry-terminal-1"));
    }
    let restored = AccountRiskRegistry::new()
        .acquire_entry(identity, &root)
        .unwrap();
    let mut state = restored.lock().unwrap();
    state.prepare_day(20_260_909).unwrap();
    assert_eq!(
        state.try_reserve_entry(20_260_909, None).unwrap_err().rule,
        "risk.unresolved_entry"
    );
    let saved: Value = serde_json::from_slice(
        &std::fs::read(
            root.join("runtime")
                .join("account-entry")
                .join("demo-712345.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(saved["unresolvedIntent"]["acceptedOrderId"], 8001);
    assert_eq!(
        saved["unresolvedIntent"]["clientOrderId"],
        "entry-terminal-1"
    );
    assert!(
        saved["lastVerifiedEntry"].is_null(),
        "NETTED is not an owned opening"
    );
    drop(state);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn direct_fill_financial_parse_error_keeps_request_bound_order_id_and_never_resends() {
    let request = request(CTraderOrderType::Market);
    let mut filled = wire(CTraderOrderType::Market, false);
    filled["payload"]["deal"]["commission"] = serde_json::json!(-40);
    filled["payload"]["deal"]
        .as_object_mut()
        .unwrap()
        .remove("moneyDigits");
    let clock = Clock::new();
    let mut io = SequenceIo::new(clock.clone(), vec![text(filled)]);
    let error = execute(&request, clock, &mut io, Duration::from_secs(30)).unwrap_err();
    assert!(format!("{error:#}").contains("execution.deal.money_digits"));
    let recovery = error
        .downcast_ref::<CTraderUnresolvedExecution>()
        .expect("typed known direct-fill order");
    assert_eq!(recovery.environment, request.environment);
    assert_eq!(recovery.account_id, 712345);
    assert_eq!(
        recovery.client_order_id.as_deref(),
        Some("entry-terminal-1")
    );
    assert_eq!(recovery.accepted_order_id, Some(8001));
    assert_eq!(io.execution_sends(), 1);
    assert_eq!(io.reads, 1);
}

#[test]
fn acceptance_cannot_be_rebound_to_foreign_fill_facts() {
    for case in [
        "account", "order", "symbol", "side", "type", "volume", "client", "position",
    ] {
        let mut accepted = wire(CTraderOrderType::Market, true);
        let mut filled = wire(CTraderOrderType::Market, false);
        match case {
            "account" => filled["payload"]["ctidTraderAccountId"] = serde_json::json!(99),
            "order" => {
                filled["payload"]["order"]["orderId"] = serde_json::json!(8002);
                filled["payload"]["deal"]["orderId"] = serde_json::json!(8002);
            }
            "symbol" => {
                filled["payload"]["order"]["tradeData"]["symbolId"] = serde_json::json!(15);
                filled["payload"]["position"]["tradeData"]["symbolId"] = serde_json::json!(15);
                filled["payload"]["deal"]["symbolId"] = serde_json::json!(15);
            }
            "side" => {
                filled["payload"]["order"]["tradeData"]["tradeSide"] = serde_json::json!(2);
                filled["payload"]["position"]["tradeData"]["tradeSide"] = serde_json::json!(2);
                filled["payload"]["deal"]["tradeSide"] = serde_json::json!(2);
            }
            "type" => filled["payload"]["order"]["orderType"] = serde_json::json!(2),
            "volume" => {
                filled["payload"]["order"]["tradeData"]["volume"] = serde_json::json!(11000000)
            }
            "client" => {
                filled["payload"]["order"]["clientOrderId"] = serde_json::json!("foreign-intent");
                filled["payload"]["order"]["tradeData"]["clientOrderId"] =
                    serde_json::json!("entry-terminal-1");
            }
            _ => accepted["payload"]["order"]["positionId"] = serde_json::json!(9002),
        }
        let clock = Clock::new();
        let mut io = SequenceIo::new(clock.clone(), vec![text(accepted), text(filled)]);
        assert!(
            execute(
                &request(CTraderOrderType::Market),
                clock,
                &mut io,
                Duration::from_secs(30)
            )
            .is_err(),
            "{case}"
        );
        assert_eq!(io.execution_sends(), 1, "{case}");
    }
}

#[test]
fn known_order_with_foreign_message_id_is_not_accepted_as_our_completion() {
    let mut filled = wire(CTraderOrderType::Market, false);
    filled["clientMsgId"] = serde_json::json!("different-intent");
    let clock = Clock::new();
    let mut io = SequenceIo::new(
        clock.clone(),
        vec![text(wire(CTraderOrderType::Market, true)), text(filled)],
    );
    let error = execute(
        &request(CTraderOrderType::Market),
        clock,
        &mut io,
        Duration::from_secs(30),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("conflicting clientMsgId"));
    assert_eq!(io.execution_sends(), 1);
}

#[test]
fn partial_cancelled_expired_and_rejected_never_become_full_openings() {
    for kind in [11, 5, 6, 7] {
        let mut terminal = wire(CTraderOrderType::Market, false);
        terminal["payload"]["executionType"] = serde_json::json!(kind);
        terminal["payload"]["errorCode"] = serde_json::json!("BROKER_TEST_ERROR");
        if kind == 11 {
            terminal["payload"]["order"]["orderStatus"] = serde_json::json!(1);
            terminal["payload"]["order"]["executedVolume"] = serde_json::json!(4000000);
            terminal["payload"]["position"]["tradeData"]["volume"] = serde_json::json!(4000000);
            terminal["payload"]["deal"]["filledVolume"] = serde_json::json!(4000000);
        } else {
            terminal["payload"]["order"]["orderStatus"] = serde_json::json!(match kind {
                5 => 5,
                6 => 4,
                _ => 3,
            });
            terminal["payload"]["order"]["executedVolume"] = serde_json::json!(0);
            terminal["payload"]["deal"] = Value::Null;
            terminal["payload"]["position"] = Value::Null;
        }
        let clock = Clock::new();
        let mut io = SequenceIo::new(
            clock.clone(),
            vec![
                text(wire(CTraderOrderType::Market, true)),
                text(terminal),
                text(wire(CTraderOrderType::Market, false)),
            ],
        );
        let error = execute(
            &request(CTraderOrderType::Market),
            clock,
            &mut io,
            Duration::from_secs(30),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains(if kind == 7 {
            "BROKER_TEST_ERROR"
        } else {
            "reconciliation"
        }));
        assert_eq!(io.execution_sends(), 1);
        assert_eq!(
            io.frames.len(),
            1,
            "not consuming a later fill as a guessed single opening"
        );
    }
}

#[test]
fn heartbeat_traffic_and_late_buffered_fill_cannot_renew_absolute_budget() {
    let clock = Clock::new();
    let mut io = SequenceIo::new(
        clock.clone(),
        vec![
            text(wire(CTraderOrderType::Market, true)),
            text(serde_json::json!({"payloadType":51,"payload":{}})),
            text(serde_json::json!({"payloadType":51,"payload":{}})),
            text(wire(CTraderOrderType::Market, false)),
        ],
    );
    for (duration, _) in &mut io.frames {
        *duration = Duration::from_secs(1);
    }
    let error = execute(
        &request(CTraderOrderType::Market),
        clock,
        &mut io,
        Duration::from_secs(3),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("absolute"));
    assert!(format!("{error:#}").contains("accepted_order_id=Some(8001)"));
    let recovery = error
        .downcast_ref::<CTraderUnresolvedExecution>()
        .expect("accepted order must survive as typed recovery context");
    assert_eq!(recovery.environment, CTraderEnvironment::Demo);
    assert_eq!(recovery.account_id, 712345);
    assert_eq!(
        recovery.client_order_id.as_deref(),
        Some("entry-terminal-1")
    );
    assert_eq!(recovery.accepted_order_id, Some(8001));
    assert_eq!(io.frames.len(), 1);
    assert_eq!(io.execution_sends(), 1);

    let clock = Clock::new();
    let mut io = SequenceIo::new(
        clock.clone(),
        vec![text(wire(CTraderOrderType::Market, false))],
    );
    io.frames[0].0 = Duration::from_secs(3);
    assert!(
        execute(
            &request(CTraderOrderType::Market),
            clock,
            &mut io,
            Duration::from_secs(3)
        )
        .is_err()
    );
    assert_eq!(io.execution_sends(), 1);
}

#[test]
fn write_time_and_write_failure_do_not_grant_new_budget_or_resend() {
    for fail in [false, true] {
        let clock = Clock::new();
        let mut io = SequenceIo::new(clock.clone(), Vec::new());
        io.send_duration = Duration::from_secs(3);
        io.fail_send = fail;
        assert!(
            execute(
                &request(CTraderOrderType::Market),
                clock,
                &mut io,
                Duration::from_secs(3)
            )
            .is_err()
        );
        assert_eq!(io.execution_sends(), 1);
        assert_eq!(io.reads, 0);
    }
}

#[test]
fn poll_timeouts_resume_reading_not_submission() {
    let clock = Clock::new();
    let mut io = SequenceIo::new(
        clock.clone(),
        vec![text(wire(CTraderOrderType::Market, false))],
    );
    io.frames.push_front((
        Duration::from_millis(100),
        Err(tungstenite::Error::Io(std::io::Error::from(
            std::io::ErrorKind::WouldBlock,
        ))),
    ));
    assert!(
        execute(
            &request(CTraderOrderType::Market),
            clock,
            &mut io,
            Duration::from_secs(3)
        )
        .is_ok()
    );
    assert_eq!(io.execution_sends(), 1);
    assert_eq!(io.reads, 2);
}

#[test]
fn pending_acceptance_stays_valid_but_immediate_acceptance_is_not_terminal_cache() {
    for kind in [
        CTraderOrderType::Market,
        CTraderOrderType::MarketRange,
        CTraderOrderType::Limit,
        CTraderOrderType::Stop,
    ] {
        let request = request(kind);
        let accepted = parse_execution_outcome(&wire(kind, true).to_string()).unwrap();
        validate_execution_outcome(&request, &accepted).unwrap();
        assert_eq!(
            cacheable_execution_outcome(&request, &accepted),
            immediate_order(&request).is_none()
        );
        let filled = parse_execution_outcome(&wire(kind, false).to_string()).unwrap();
        assert!(cacheable_execution_outcome(&request, &filled));
    }
    assert_eq!(immediate_execution_timeout(0), Duration::from_secs(30));
    assert_eq!(immediate_execution_timeout(2), Duration::from_secs(2));
    assert_eq!(immediate_execution_timeout(3600), Duration::from_secs(30));
}

#[test]
fn correlated_broker_error_retains_original_diagnostic_without_resend() {
    let clock = Clock::new();
    let mut io = SequenceIo::new(
        clock.clone(),
        vec![text(serde_json::json!({
            "clientMsgId":"terminal-1", "payloadType":2142,
            "payload":{"errorCode":"NOT_ENOUGH_MONEY","description":"fixture rejected"}
        }))],
    );
    let error = execute(
        &request(CTraderOrderType::Market),
        clock,
        &mut io,
        Duration::from_secs(30),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("NOT_ENOUGH_MONEY"));
    assert!(format!("{error:#}").contains("fixture rejected"));
    assert_eq!(io.execution_sends(), 1);
}

#[test]
fn missing_initial_correlation_cannot_bind_a_foreign_unsolicited_order() {
    let mut filled = wire(CTraderOrderType::Market, false);
    filled.as_object_mut().unwrap().remove("clientMsgId");
    let clock = Clock::new();
    let mut io = SequenceIo::new(clock.clone(), vec![text(filled), Message::Close(None)]);
    let error = execute(
        &request(CTraderOrderType::Market),
        clock,
        &mut io,
        Duration::from_secs(30),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("accepted_order_id=None"));
    assert_eq!(
        error
            .downcast_ref::<CTraderUnresolvedExecution>()
            .expect("unresolved context")
            .accepted_order_id,
        None
    );
    assert_eq!(io.execution_sends(), 1);
}

#[test]
fn acceptance_with_wrong_root_client_order_id_is_rejected_before_waiting() {
    let mut accepted = wire(CTraderOrderType::Market, true);
    accepted["payload"]["order"]["clientOrderId"] = serde_json::json!("other-intent");
    accepted["payload"]["order"]["tradeData"]["clientOrderId"] =
        serde_json::json!("entry-terminal-1");
    let clock = Clock::new();
    let mut io = SequenceIo::new(clock.clone(), vec![text(accepted)]);
    let error = execute(
        &request(CTraderOrderType::Market),
        clock,
        &mut io,
        Duration::from_secs(30),
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("root clientOrderId"));
    assert_eq!(io.reads, 1);
    assert_eq!(io.execution_sends(), 1);
}

#[test]
fn reused_deadline_io_checks_each_underlying_fragment_not_only_complete_messages() {
    struct Fragments(Clock);
    impl std::io::Read for Fragments {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            self.0.advance(Duration::from_secs(1));
            bytes[0] = b'x';
            Ok(1)
        }
    }
    let clock = Clock::new();
    let budget = CTraderOperationBudget::new(clock.clone(), Duration::from_secs(2), None).unwrap();
    let mut io = crate::app_services::ctrader_historical_admission::DeadlineIo::new(
        Fragments(clock),
        budget,
        CTraderIoPhase::ResponseRead,
        Duration::from_millis(100),
    );
    let mut byte = [0u8];
    assert_eq!(std::io::Read::read(&mut io, &mut byte).unwrap(), 1);
    let error = std::io::Read::read(&mut io, &mut byte).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
    assert!(error.to_string().contains("absolute"));
}
