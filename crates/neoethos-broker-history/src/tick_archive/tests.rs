use super::*;
use crate::ctrader_data::HistoricalTick;
use crate::ctrader_messages::CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE;

fn plan(end: i64) -> Plan {
    Plan {
        schema: SCHEMA.into(),
        page_boundary_policy: PAGE_BOUNDARY_POLICY.into(),
        environment: Environment::Demo,
        endpoint: BrokerEnvironment::Demo.endpoint_host().into(),
        account_id: 42,
        symbol_id: 1,
        symbol: "EURUSD".into(),
        from_ms: 1_000,
        to_ms_exclusive: end,
    }
}

fn symbol() -> CTraderSymbolInfo {
    CTraderSymbolInfo {
        symbol_id: 1,
        symbol_name: "EURUSD".into(),
        display_name: "EURUSD".into(),
        digits: 5,
        pip_position: 4,
        is_archived: false,
        is_trading_enabled: true,
        min_volume: None,
        max_volume: None,
        step_volume: None,
        lot_size: None,
        pnl_conversion_fee_rate: None,
        financials: None,
    }
}

fn page(times: &[i64], has_more: bool) -> HistoricalTicksResult {
    HistoricalTicksResult {
        symbol_id: 1,
        ticks: times
            .iter()
            .map(|&timestamp_ms| HistoricalTick {
                timestamp_ms,
                price: 1.1,
            })
            .collect(),
        has_more,
    }
}

fn record(plan: &Plan, cursor: &Cursor, sequence: u64, has_more: bool) -> PageRecord {
    let raw = serde_json::json!({
        "payloadType": CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE,
        "clientMsgId": "test-page",
        "payload": { "ctidTraderAccountId": 42, "hasMore": has_more,
            "tickData": [
                {"timestamp": cursor.page_to_ms_exclusive - 10, "tick": 110002},
                {"timestamp": -10, "tick": -1}
            ] }
    })
    .to_string();
    PageRecord {
        schema: SCHEMA.into(),
        plan_sha256: sha256(&serde_json::to_vec(plan).unwrap()),
        sequence,
        cursor: cursor.clone(),
        client_msg_id: "test-page".into(),
        captured_at_ms: 9_000,
        raw_response_sha256: sha256(raw.as_bytes()),
        raw_response_json: raw,
    }
}

fn persist(root: &Path, record: &PageRecord) {
    let raw = serde_json::to_string(record).unwrap();
    write_vortex_chunks_fallible_limited(
        page_path(root, record.sequence),
        [Ok(VarBinArray::from(vec![raw.as_str()]).into_array())],
        MAX_PAGE_BYTES,
    )
    .unwrap();
}

#[test]
fn visits_both_sides_and_every_week_including_short_final_window() {
    let plan = plan(1_000 + WEEK_MS * 2 + 17);
    let mut cursor = Some(Cursor::first(&plan));
    let mut windows = Vec::new();
    while let Some(current) = cursor {
        windows.push((
            current.chunk_from_ms,
            current.chunk_to_ms_exclusive,
            current.side,
        ));
        cursor = current.after(&plan, &page(&[], false)).unwrap();
    }
    assert_eq!(windows.len(), 6);
    for pair in windows.chunks_exact(2) {
        assert_eq!(pair[0].0, pair[1].0);
        assert_eq!(pair[0].1, pair[1].1);
        assert_eq!((pair[0].2, pair[1].2), (1, 2));
        assert!(pair[0].1 - pair[0].0 <= WEEK_MS);
    }
    assert_eq!(windows[2].0, windows[0].1);
    assert_eq!(windows[4].0, windows[2].1);
    assert_eq!(windows[5].1, plan.to_ms_exclusive);
}

#[test]
fn has_more_keeps_same_side_and_chunk_until_exhausted() {
    let plan = plan(2_000);
    let first = Cursor::first(&plan);
    let next = first
        .after(&plan, &page(&[1_700, 1_999], true))
        .unwrap()
        .unwrap();
    assert_eq!(next.page_to_ms_exclusive, 1_701);
    assert_eq!(next.side, 1);
    let ask = next
        .after(&plan, &page(&[1_001, 1_700], false))
        .unwrap()
        .unwrap();
    assert_eq!(ask.side, 2);
    assert_eq!(ask.page_to_ms_exclusive, 2_000);
}

#[test]
fn rejects_stalled_empty_or_inconsistent_pagination() {
    let plan = plan(2_000);
    let cursor = Cursor::first(&plan);
    for times in [&[][..], &[2_000][..], &[1_999][..]] {
        assert!(cursor.after(&plan, &page(times, true)).is_err());
    }
    assert!(cursor.after(&plan, &page(&[], false)).is_ok());
}

#[test]
fn rejects_bad_math_order_bounds_and_symbol_without_sorting() {
    let plan = plan(2_000);
    let cursor = Cursor::first(&plan);
    for times in [&[999][..], &[2_001][..], &[1_800, 1_700][..]] {
        assert!(cursor.after(&plan, &page(times, false)).is_err());
    }
    for price in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        let mut p = page(&[1_500], false);
        p.ticks[0].price = price;
        assert!(cursor.after(&plan, &p).is_err());
    }
    let mut p = page(&[1_500], false);
    p.symbol_id = 2;
    assert!(cursor.after(&plan, &p).is_err());
}

#[test]
fn inclusive_upper_boundary_is_preserved_raw_but_not_double_counted() {
    let plan = plan(2_000);
    let cursor = Cursor::first(&plan);
    let mut summary = TickArchiveSummary::new(&plan);
    summary
        .include(&plan, &cursor, &page(&[1_700, 1_999, 2_000], true), 10, "a")
        .unwrap();
    let next = summary.next.clone().unwrap();
    summary
        .include(&plan, &next, &page(&[1_001, 1_700], false), 12, "b")
        .unwrap();
    assert_eq!(summary.bid_ticks, 3);
    assert_eq!(summary.excluded_upper_boundary_ticks, 1);
    assert_eq!(summary.deferred_oldest_boundary_ticks, 1);
    assert_eq!(summary.archive_bytes, 22);
    assert_eq!(summary.oldest_tick_ms, Some(1_001));
    assert_eq!(summary.newest_tick_ms, Some(1_999));
    assert!(!summary.all_requested_windows_visited);
}

#[test]
fn real_signed_delta_decoder_and_exact_identity_are_used() {
    let plan = plan(2_000);
    let cursor = Cursor::first(&plan);
    let mut record = record(&plan, &cursor, 0, false);
    let hash = record.plan_sha256.clone();
    let p = decode_checked(&record, &plan, &hash, &cursor, 0, &symbol()).unwrap();
    assert_eq!(p.ticks[0].timestamp_ms, 1_980);
    assert_eq!(p.ticks[1].timestamp_ms, 1_990);
    assert_eq!(p.ticks[0].price, 1.10001);
    assert!(decode_checked(&record, &plan, &hash, &cursor, 1, &symbol()).is_err());
    record.client_msg_id = "wrong".into();
    assert!(decode_checked(&record, &plan, &hash, &cursor, 0, &symbol()).is_err());
}

#[test]
fn vortex_raw_roundtrip_and_resume_revalidate_actual_pages_not_progress_claim() {
    let root = tempfile::tempdir().unwrap();
    let plan = plan(2_000);
    let first = Cursor::first(&plan);
    let a = record(&plan, &first, 0, false);
    let plan_hash = a.plan_sha256.clone();
    persist(root.path(), &a);
    assert_eq!(
        decode_record(&page_path(root.path(), 0))
            .unwrap()
            .raw_response_json,
        a.raw_response_json
    );
    write_json_atomic(
        root.path().join("progress.json"),
        &serde_json::json!({"all_requested_windows_visited":true}),
    )
    .unwrap();
    let cancel = HistoricalRequestCancellation::new();
    let partial = resume_archive(root.path(), &plan, &plan_hash, &symbol(), &cancel, None).unwrap();
    assert_eq!(partial.pages, 1);
    assert_eq!(partial.bid_ticks, 2);
    assert!(!partial.all_requested_windows_visited);
    let b = record(&plan, partial.next.as_ref().unwrap(), 1, false);
    persist(root.path(), &b);
    let complete =
        resume_archive(root.path(), &plan, &plan_hash, &symbol(), &cancel, None).unwrap();
    assert!(complete.all_requested_windows_visited);
    assert_eq!(complete.ask_ticks, 2);
    assert_eq!(complete.authority, "unreviewed-raw-market-data-only");
    fs::remove_file(page_path(root.path(), 0)).unwrap();
    assert!(resume_archive(root.path(), &plan, &plan_hash, &symbol(), &cancel, None).is_err());
}

#[test]
fn resume_rejects_corruption_wrong_scope_and_cancellation() {
    let root = tempfile::tempdir().unwrap();
    let plan = plan(2_000);
    let mut a = record(&plan, &Cursor::first(&plan), 0, false);
    let hash = a.plan_sha256.clone();
    a.raw_response_json.push(' ');
    persist(root.path(), &a);
    let cancel = HistoricalRequestCancellation::new();
    assert!(resume_archive(root.path(), &plan, &hash, &symbol(), &cancel, None).is_err());
    a.raw_response_sha256 = sha256(a.raw_response_json.as_bytes());
    persist(root.path(), &a);
    assert!(resume_archive(root.path(), &plan, "wrong", &symbol(), &cancel, None).is_err());
    cancel.cancel();
    assert!(resume_archive(root.path(), &plan, &hash, &symbol(), &cancel, None).is_err());
}

#[test]
fn disk_admission_keeps_reserve_and_bounds_next_page_even_near_integer_limit() {
    let page = MAX_PAGE_BYTES;
    assert!(has_disk_budget(0, page, page * 3, page));
    assert!(!has_disk_budget(1, page, page * 3, page));
    assert!(!has_disk_budget(0, page, page * 3 - 1, page));
    assert!(!has_disk_budget(u64::MAX, page, u64::MAX, page));
    assert!(!has_disk_budget(0, page, u64::MAX - 1, u64::MAX));
}

#[test]
fn absent_repeated_ticks_advances_the_window_but_never_fabricates_quotes() {
    let plan = plan(2_000);
    let cursor = Cursor::first(&plan);
    let mut record = record(&plan, &cursor, 0, false);
    let mut raw: serde_json::Value = serde_json::from_str(&record.raw_response_json).unwrap();
    raw["payload"].as_object_mut().unwrap().remove("tickData");
    record.raw_response_json = raw.to_string();
    record.raw_response_sha256 = sha256(record.raw_response_json.as_bytes());
    let page = decode_checked(&record, &plan, &record.plan_sha256, &cursor, 0, &symbol()).unwrap();
    let mut summary = TickArchiveSummary::new(&plan);
    summary
        .include(&plan, &cursor, &page, 100, "empty")
        .unwrap();
    assert_eq!(summary.bid_ticks, 0);
    assert_eq!(summary.empty_responses, 1);
    assert_eq!(summary.oldest_tick_ms, None);
    assert_eq!(summary.next.unwrap().side, 2);
}

#[test]
fn fragmented_vortex_record_preserves_large_unicode_raw_response_and_compresses() {
    let root = tempfile::tempdir().unwrap();
    let plan = plan(2_000);
    let mut record = record(&plan, &Cursor::first(&plan), 0, false);
    record.raw_response_json =
        "{\"timestamp\":-187,\"tick\":1,\"note\":\"Δεδομένα\"}".repeat(10_000);
    record.raw_response_sha256 = sha256(record.raw_response_json.as_bytes());
    let raw = serde_json::to_string(&record).unwrap();
    let path = page_path(root.path(), 0);
    let stats =
        write_vortex_chunks_fallible_limited(&path, [Ok(raw_fragments(&raw))], MAX_PAGE_BYTES)
            .unwrap();
    let reread = decode_record(&path).unwrap();
    assert_eq!(reread.raw_response_json, record.raw_response_json);
    assert_eq!(reread.raw_response_sha256, record.raw_response_sha256);
    assert!(
        stats.file_size < raw.len() as u64 / 2,
        "compression did not reduce repeated raw JSON: {} / {}",
        stats.file_size,
        raw.len()
    );
}

#[test]
fn oldest_millisecond_split_across_pages_is_retrieved_without_loss_or_double_count() {
    let plan = plan(2_000);
    let first = Cursor::first(&plan);
    let mut summary = TickArchiveSummary::new(&plan);
    // Two rows in a capped boundary group are NOT evidence the group is complete.
    summary
        .include(&plan, &first, &page(&[1_700, 1_700, 1_800], true), 10, "a")
        .unwrap();
    assert_eq!(summary.bid_ticks, 1);
    let next = summary.next.clone().unwrap();
    assert_eq!(next.page_to_ms_exclusive, 1_701);
    summary
        .include(
            &plan,
            &next,
            &page(&[1_100, 1_700, 1_700, 1_700], false),
            10,
            "b",
        )
        .unwrap();
    assert_eq!(summary.bid_ticks, 5);
    assert_eq!(summary.deferred_oldest_boundary_ticks, 2);
    // A whole broker page occupied by one millisecond cannot be paged safely.
    assert!(next.after(&plan, &page(&[1_700, 1_700], true)).is_err());
}

#[test]
fn final_oldest_millisecond_can_be_fetched_before_switching_quote_side() {
    let plan = plan(2_000);
    let first = Cursor::first(&plan);
    let final_ms = first
        .after(&plan, &page(&[1_000, 1_100], true))
        .unwrap()
        .unwrap();
    assert_eq!(final_ms.page_to_ms_exclusive, 1_001);
    assert_eq!(
        final_ms
            .after(&plan, &page(&[1_000, 1_000], false))
            .unwrap()
            .unwrap()
            .side,
        2
    );
}

fn archive_cli(root: &Path) -> TickArchiveCli {
    TickArchiveCli {
        environment: Environment::Demo,
        account_id: 42,
        symbol_id: 1,
        symbol: "EURUSD".into(),
        from_ms: 1_000,
        to_ms: 2_000,
        output: root.to_owned(),
        max_archive_bytes: 8 * 1024 * 1024 * 1024,
        reserve_disk_bytes: 4 * 1024 * 1024 * 1024,
        max_new_pages: None,
        stop_file: None,
        cpu_threads: None,
    }
}

fn symbol_observation() -> serde_json::Value {
    serde_json::json!({
        "payloadType": crate::ctrader_messages::CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE,
        "clientMsgId": "test-symbol",
        "payload": {
            "ctidTraderAccountId": 42,
            "symbol": [{"symbolId": 1, "digits": 5, "pipPosition": 4}]
        }
    })
}

#[test]
fn resume_reuses_verified_pages_and_preserves_first_symbol_observation() {
    let root = tempfile::tempdir().unwrap();
    let cli = archive_cli(root.path());
    let plan = plan(2_000);
    let record = record(&plan, &Cursor::first(&plan), 0, false);
    persist(root.path(), &record);
    let observation_path = root.path().join("symbol-observation.json");
    write_json_atomic(&observation_path, &symbol_observation()).unwrap();
    let original_observation = fs::read(&observation_path).unwrap();
    let original_page = fs::read(page_path(root.path(), 0)).unwrap();
    write_json_atomic(
        root.path().join("progress.json"),
        &serde_json::json!({"pages":9999, "all_requested_windows_visited":true}),
    )
    .unwrap();
    let mut opened = 0;
    let (_, current_symbol, summary) = prepare_archive_session(
        &cli,
        &plan,
        &record.plan_sha256,
        &HistoricalRequestCancellation::new(),
        || {
            opened += 1;
            Ok((
                (),
                symbol(),
                "not used to replace the first observation".into(),
            ))
        },
    )
    .unwrap();
    assert_eq!(opened, 1);
    assert_eq!(current_symbol.digits, 5);
    assert_eq!(summary.pages, 1);
    assert_eq!(summary.bid_ticks, 2);
    assert_eq!(summary.next.as_ref().unwrap().side, 2);
    assert!(!summary.all_requested_windows_visited);
    assert_eq!(summary.authority, "unreviewed-raw-market-data-only");
    assert_eq!(fs::read(observation_path).unwrap(), original_observation);
    assert_eq!(fs::read(page_path(root.path(), 0)).unwrap(), original_page);
}

#[test]
fn invalid_archived_pages_are_rejected_before_any_session_is_opened() {
    for defect in ["raw-hash", "scope", "page-gap"] {
        let root = tempfile::tempdir().unwrap();
        let cli = archive_cli(root.path());
        let plan = plan(2_000);
        let mut record = record(&plan, &Cursor::first(&plan), 0, false);
        let plan_hash = record.plan_sha256.clone();
        match defect {
            "raw-hash" => record.raw_response_json.push(' '),
            "scope" => record.plan_sha256 = "another-plan".into(),
            "page-gap" => record.sequence = 1,
            _ => unreachable!(),
        }
        persist(root.path(), &record);
        write_json_atomic(
            root.path().join("symbol-observation.json"),
            &symbol_observation(),
        )
        .unwrap();
        let mut opened = 0;
        let result = prepare_archive_session(
            &cli,
            &plan,
            &plan_hash,
            &HistoricalRequestCancellation::new(),
            || {
                opened += 1;
                Ok(((), symbol(), symbol_observation().to_string()))
            },
        );
        assert!(result.is_err(), "{defect}");
        assert_eq!(
            opened, 0,
            "local {defect} was checked only after connecting"
        );
    }
}

#[test]
fn missing_or_wrong_saved_symbol_identity_refuses_connection() {
    for defect in ["missing", "account", "symbol", "multiple"] {
        let root = tempfile::tempdir().unwrap();
        let cli = archive_cli(root.path());
        let plan = plan(2_000);
        let record = record(&plan, &Cursor::first(&plan), 0, false);
        persist(root.path(), &record);
        let mut raw = symbol_observation();
        match defect {
            "missing" => {}
            "account" => raw["payload"]["ctidTraderAccountId"] = 43.into(),
            "symbol" => raw["payload"]["symbol"][0]["symbolId"] = 2.into(),
            "multiple" => {
                let duplicate = raw["payload"]["symbol"][0].clone();
                raw["payload"]["symbol"]
                    .as_array_mut()
                    .unwrap()
                    .push(duplicate);
            }
            _ => unreachable!(),
        }
        if defect != "missing" {
            write_json_atomic(root.path().join("symbol-observation.json"), &raw).unwrap();
        }
        let mut opened = 0;
        let result = prepare_archive_session(
            &cli,
            &plan,
            &record.plan_sha256,
            &HistoricalRequestCancellation::new(),
            || {
                opened += 1;
                Ok(((), symbol(), symbol_observation().to_string()))
            },
        );
        assert!(result.is_err(), "{defect}");
        assert_eq!(opened, 0, "{defect}");
    }
}

#[test]
fn fresh_connected_symbol_must_still_match_after_local_validation() {
    for defect in ["digits", "symbol", "name"] {
        let root = tempfile::tempdir().unwrap();
        let cli = archive_cli(root.path());
        let plan = plan(2_000);
        let record = record(&plan, &Cursor::first(&plan), 0, false);
        persist(root.path(), &record);
        let observation_path = root.path().join("symbol-observation.json");
        write_json_atomic(&observation_path, &symbol_observation()).unwrap();
        let old = fs::read(&observation_path).unwrap();
        let result = prepare_archive_session(
            &cli,
            &plan,
            &record.plan_sha256,
            &HistoricalRequestCancellation::new(),
            || {
                let mut live = symbol();
                match defect {
                    "digits" => live.digits = 4,
                    "symbol" => live.symbol_id = 2,
                    "name" => live.symbol_name = "GBPUSD".into(),
                    _ => unreachable!(),
                }
                Ok(((), live, symbol_observation().to_string()))
            },
        );
        assert!(result.is_err(), "{defect}");
        assert_eq!(fs::read(observation_path).unwrap(), old);
        assert_eq!(archive_page_count(root.path()).unwrap(), 1);
    }
}

#[test]
fn empty_new_archive_opens_once_and_saves_the_first_observation() {
    let root = tempfile::tempdir().unwrap();
    let cli = archive_cli(root.path());
    let plan = plan(2_000);
    let mut opened = 0;
    let (_, _, summary) = prepare_archive_session(
        &cli,
        &plan,
        &sha256(&serde_json::to_vec(&plan).unwrap()),
        &HistoricalRequestCancellation::new(),
        || {
            opened += 1;
            Ok(((), symbol(), symbol_observation().to_string()))
        },
    )
    .unwrap();
    assert_eq!(opened, 1);
    assert_eq!(summary.pages, 0);
    assert_eq!(summary.next, Some(Cursor::first(&plan)));
    let saved: serde_json::Value =
        serde_json::from_slice(&fs::read(root.path().join("symbol-observation.json")).unwrap())
            .unwrap();
    assert_eq!(saved, symbol_observation());
}

#[test]
fn user_stop_or_cancellation_prevents_opening_the_archive_session() {
    for stop_file in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut cli = archive_cli(root.path());
        let cancel = HistoricalRequestCancellation::new();
        if stop_file {
            let path = root.path().join("stop.request");
            fs::write(&path, b"stop").unwrap();
            cli.stop_file = Some(path);
        } else {
            cancel.cancel();
        }
        let mut opened = 0;
        let result = prepare_archive_session(&cli, &plan(2_000), "unused", &cancel, || {
            opened += 1;
            Ok(((), symbol(), symbol_observation().to_string()))
        });
        assert!(result.unwrap_err().to_string().contains("cancelled"));
        assert_eq!(opened, 0);
        assert!(!root.path().join("symbol-observation.json").exists());
    }
}

#[test]
fn local_page_validation_itself_honors_the_user_stop_file() {
    let root = tempfile::tempdir().unwrap();
    let plan = plan(2_000);
    let record = record(&plan, &Cursor::first(&plan), 0, false);
    persist(root.path(), &record);
    let stop = root.path().join("stop.request");
    fs::write(&stop, b"stop").unwrap();
    let result = resume_archive(
        root.path(),
        &plan,
        &record.plan_sha256,
        &symbol(),
        &HistoricalRequestCancellation::new(),
        Some(&stop),
    );
    assert!(result.unwrap_err().to_string().contains("cancelled"));
    assert_eq!(archive_page_count(root.path()).unwrap(), 1);
}
