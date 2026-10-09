use super::*;
use crate::ctrader_messages::{
    CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE,
};
use neoethos_core::storage::json::write_json_atomic;
use neoethos_data::core::vortex_io::write_vortex_chunks_fallible_limited;
use vortex_array::IntoArray;
use vortex_array::arrays::VarBinArray;

fn event(timestamp_ms: i64, side: i32, price_units: i64) -> Event {
    Event {
        timestamp_ms,
        side,
        price_units,
    }
}

fn assert_partition(d: &SpreadDiagnostics) {
    assert_eq!(
        d.valid_book_ms + d.unavailable_or_stale_ms + d.ambiguous_book_ms + d.crossed_book_ms,
        d.interval_ms
    );
}

#[test]
fn spread_quantiles_weight_elapsed_time_instead_of_number_of_updates() {
    let d = spread_diagnostics(
        vec![
            event(90, 2, 200),
            event(0, 1, 100),
            event(90, 1, 100),
            event(0, 2, 110),
        ],
        0,
        100,
        1_000,
        10.0,
    );
    assert_eq!(d.valid_book_ms, 100);
    assert_eq!(d.time_weighted_mean_pips, Some(1.9));
    assert_eq!(d.time_weighted_p50_pips, Some(1.0));
    assert_eq!(d.time_weighted_p95_pips, Some(10.0));
    assert_eq!(d.time_weighted_p99_pips, Some(10.0));
    assert_partition(&d);
}

#[test]
fn stale_and_unseeded_intervals_never_borrow_future_quotes() {
    let d = spread_diagnostics(vec![event(10, 1, 100), event(20, 2, 110)], 0, 100, 30, 10.0);
    assert_eq!(d.valid_book_ms, 20);
    assert_eq!(d.unavailable_or_stale_ms, 80);
    assert_partition(&d);
    let seeded = spread_diagnostics(vec![event(-10, 1, 100), event(-5, 2, 110)], 0, 30, 20, 10.0);
    assert_eq!(seeded.valid_book_ms, 10);
    assert_eq!(seeded.unavailable_or_stale_ms, 20);
    assert_partition(&seeded);
}

#[test]
fn tied_distinct_prices_remain_ambiguous_until_a_new_unique_update() {
    let events = vec![
        event(0, 1, 105),
        event(0, 1, 100),
        event(0, 2, 110),
        event(50, 1, 100),
    ];
    let d = spread_diagnostics(events.clone(), 0, 100, 1_000, 10.0);
    assert_eq!(d.ambiguous_book_ms, 50);
    assert_eq!(d.valid_book_ms, 50);
    assert_eq!(d.ambiguous_side_timestamp_groups, 1);
    assert_eq!(d.time_weighted_mean_pips, Some(1.0));
    assert_partition(&d);
    // Sorting tied events in a different order must not pick a different price.
    let reversed = spread_diagnostics(events.into_iter().rev().collect(), 0, 100, 1_000, 10.0);
    assert_eq!(
        serde_json::to_value(&d).unwrap(),
        serde_json::to_value(&reversed).unwrap()
    );
    let identical = spread_diagnostics(
        vec![event(0, 1, 100), event(0, 1, 100), event(0, 2, 110)],
        0,
        100,
        1_000,
        10.0,
    );
    assert_eq!(identical.ambiguous_book_ms, 0);
    assert_eq!(identical.valid_book_ms, 100);
}

#[test]
fn crossed_and_fully_stale_books_cannot_pollute_spread_extrema() {
    let d = spread_diagnostics(
        vec![
            event(0, 1, 110),
            event(0, 2, 100),
            event(10, 1, 90),
            event(10, 2, 100),
            event(50, 1, 200),
        ],
        0,
        100,
        20,
        10.0,
    );
    assert_eq!(d.crossed_book_ms, 10);
    assert_eq!(d.valid_book_ms, 20);
    assert_eq!(d.unavailable_or_stale_ms, 70);
    assert_eq!(d.min_pips, Some(1.0));
    assert_eq!(d.max_pips, Some(1.0));
    assert_partition(&d);
    let empty = spread_diagnostics(Vec::new(), 0, 100, 20, 10.0);
    assert_eq!(empty.time_weighted_mean_pips, None);
    assert_eq!(empty.valid_book_fraction, 0.0);
    assert_partition(&empty);
}

fn fixture() -> (tempfile::TempDir, TickInspectCli) {
    let root = tempfile::tempdir().unwrap();
    File::create(root.path().join("archive.lock")).unwrap();
    let plan = Plan {
        schema: super::super::SCHEMA.into(),
        page_boundary_policy: super::super::PAGE_BOUNDARY_POLICY.into(),
        environment: super::super::Environment::Demo,
        endpoint: "demo.ctraderapi.com".into(),
        account_id: 42,
        symbol_id: 1,
        symbol: "EURUSD".into(),
        from_ms: 1_000,
        to_ms_exclusive: 2_000,
    };
    write_json_atomic(root.path().join("request.json"), &plan).unwrap();
    write_json_atomic(
        root.path().join("symbol-observation.json"),
        &serde_json::json!({
            "payloadType": CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE,
            "clientMsgId": "observation", "payload": {"ctidTraderAccountId":42,
                "symbol":[{"symbolId":1, "digits":5, "pipPosition":4}]}
        }),
    )
    .unwrap();
    let mut chain = String::new();
    for sequence in 0..2 {
        let raw = serde_json::json!({
            "payloadType": CTRADER_OA_GET_TICK_DATA_RESPONSE_PAYLOAD_TYPE,
            "clientMsgId": "page", "payload": {"ctidTraderAccountId":42, "hasMore":false,
                "tickData":[{"timestamp":1900,"tick":110010 + sequence * 10},
                    {"timestamp":-800,"tick":-10}]}
        })
        .to_string();
        let record = super::super::PageRecord {
            schema: super::super::SCHEMA.into(),
            plan_sha256: sha256(&serde_json::to_vec(&plan).unwrap()),
            sequence,
            cursor: super::super::Cursor {
                chunk_from_ms: 1_000,
                chunk_to_ms_exclusive: 2_000,
                page_to_ms_exclusive: 2_000,
                side: sequence as i32 + 1,
            },
            client_msg_id: "page".into(),
            captured_at_ms: 3_000,
            raw_response_sha256: sha256(raw.as_bytes()),
            raw_response_json: raw,
        };
        persist(root.path(), &record);
        chain = sha256(
            format!(
                "{chain}:{}",
                file_hash(&page_path(root.path(), sequence)).unwrap()
            )
            .as_bytes(),
        );
    }
    // Deliberately false progress: only the actual pages establish completion.
    write_json_atomic(
        root.path().join("progress.json"),
        &serde_json::json!({"pages":0}),
    )
    .unwrap();
    let cli = TickInspectCli {
        archive: root.path().to_owned(),
        expected_page_hash_chain: chain,
        from_ms: 1_000,
        to_ms: 2_000,
        max_quote_age_ms: 1_000,
        max_events: 100,
        cpu_threads: None,
    };
    (root, cli)
}

fn persist(root: &Path, record: &super::super::PageRecord) {
    let raw = serde_json::to_string(record).unwrap();
    write_vortex_chunks_fallible_limited(
        page_path(root, record.sequence),
        [Ok(VarBinArray::from(vec![raw.as_str()]).into_array())],
        super::super::MAX_PAGE_BYTES,
    )
    .unwrap();
}

#[test]
fn offline_inspection_verifies_real_decoder_pages_and_preserves_all_inputs() {
    let (root, cli) = fixture();
    let before = fs::read_dir(root.path())
        .unwrap()
        .map(|entry| {
            let path = entry.unwrap().path();
            (path.clone(), fs::read(path).unwrap())
        })
        .collect::<Vec<_>>();
    let report = inspect_archive(&cli, &HistoricalRequestCancellation::new()).unwrap();
    assert_eq!(report.verified_archive.pages, 2);
    assert_eq!(report.verified_archive.bid_ticks, 2);
    assert_eq!(report.verified_archive.ask_ticks, 2);
    assert_eq!(report.spreads.valid_book_ms, 900);
    assert_eq!(report.spreads.time_weighted_mean_pips, Some(1.0));
    assert_eq!(report.promotion_eligibility, "NotPromotionEligible");
    assert_eq!(report.account_id, 42);
    let again = inspect_archive(&cli, &HistoricalRequestCancellation::new()).unwrap();
    assert_eq!(report.binding_sha256, again.binding_sha256);
    for (path, bytes) in before {
        assert_eq!(fs::read(path).unwrap(), bytes);
    }
}

#[test]
fn incomplete_corrupt_gapped_or_digest_mismatched_archives_are_refused() {
    for defect in [
        "incomplete",
        "gap",
        "raw-hash",
        "account",
        "digest",
        "budget",
        "cancel",
        "lock",
    ] {
        let (root, mut cli) = fixture();
        let cancel = HistoricalRequestCancellation::new();
        let mut writer = None;
        match defect {
            "incomplete" => fs::remove_file(page_path(root.path(), 1)).unwrap(),
            "gap" => fs::remove_file(page_path(root.path(), 0)).unwrap(),
            "raw-hash" | "account" => {
                let mut record = decode_record(&page_path(root.path(), 0)).unwrap();
                if defect == "raw-hash" {
                    record.raw_response_json.push(' ');
                } else {
                    record.raw_response_json = record
                        .raw_response_json
                        .replace("\"ctidTraderAccountId\":42", "\"ctidTraderAccountId\":43");
                    record.raw_response_sha256 = sha256(record.raw_response_json.as_bytes());
                }
                persist(root.path(), &record);
            }
            "digest" => cli.expected_page_hash_chain = "a".repeat(64),
            "budget" => cli.max_events = 1,
            "cancel" => cancel.cancel(),
            "lock" => {
                let file = fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .open(root.path().join("archive.lock"))
                    .unwrap();
                fs2::FileExt::try_lock_exclusive(&file).unwrap();
                writer = Some(file);
            }
            _ => unreachable!(),
        }
        assert!(
            inspect_archive(&cli, &cancel).is_err(),
            "{defect} was accepted"
        );
        drop(writer);
    }
}

#[test]
fn diagnostic_scope_and_resource_limits_are_validated_before_scanning() {
    for defect in ["too-wide", "outside", "age", "memory", "metadata", "policy"] {
        let (root, mut cli) = fixture();
        match defect {
            "too-wide" => cli.to_ms = cli.from_ms + WEEK_MS + 1,
            "outside" => cli.from_ms = 999,
            "age" => cli.max_quote_age_ms = 0,
            "memory" => cli.max_events = MAX_EVENTS + 1,
            "metadata" => fs::write(
                root.path().join("request.json"),
                vec![b' '; MAX_METADATA_BYTES as usize + 1],
            )
            .unwrap(),
            "policy" => {
                let path = root.path().join("request.json");
                let bytes = fs::read_to_string(&path).unwrap().replace(
                    super::super::PAGE_BOUNDARY_POLICY,
                    "legacy-lossy-pagination",
                );
                fs::write(path, bytes).unwrap();
            }
            _ => unreachable!(),
        }
        assert!(
            inspect_archive(&cli, &HistoricalRequestCancellation::new()).is_err(),
            "{defect}"
        );
    }
}
