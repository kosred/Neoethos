use anyhow::Result;
use neoethos_broker_history::bootstrap_writer::{
    BrokerTrendbarStreamRequest, publish_broker_trendbar_chunks,
};
use neoethos_broker_history::{
    CANONICAL_TRENDBAR_PAGING_POLICY_V1, CANONICAL_TRENDBAR_SERIES_FROM_MS_V1,
    CanonicalTrendbarAcquisitionCellV1, CanonicalTrendbarAcquisitionPlanV1,
    CanonicalTrendbarAcquisitionStoreV1, CanonicalTrendbarPlanReceiptV1, CanonicalTrendbarSymbolV1,
};
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalOhlcvChunk,
    CanonicalTimeframe, CanonicalVolumeChunk, SelectedDatasetGenerationV1,
};
use serde_json::Value;
use std::fs;

const SERVER: &str = "demo.ctraderapi.com";
const ACCOUNT_ID: i64 = 42;
const TO_MS_EXCLUSIVE: i64 = 1_767_225_600_000;

fn symbols() -> Vec<CanonicalTrendbarSymbolV1> {
    vec![
        CanonicalTrendbarSymbolV1::new(1, "EURUSD").expect("EURUSD symbol"),
        CanonicalTrendbarSymbolV1::new(2, "USDJPY").expect("USDJPY symbol"),
    ]
}

fn timeframes() -> Vec<CanonicalTimeframe> {
    vec![CanonicalTimeframe::M1, CanonicalTimeframe::H1]
}

fn plan() -> CanonicalTrendbarAcquisitionPlanV1 {
    CanonicalTrendbarAcquisitionPlanV1::new(
        CTraderEnvironment::Demo,
        SERVER,
        ACCOUNT_ID,
        CANONICAL_TRENDBAR_SERIES_FROM_MS_V1,
        TO_MS_EXCLUSIVE,
        symbols(),
        timeframes(),
    )
    .expect("exact canonical trendbar plan")
}

fn publish_cell(
    data_root: &std::path::Path,
    symbol: &CanonicalTrendbarSymbolV1,
    timeframe: CanonicalTimeframe,
    account_id: i64,
) -> SelectedDatasetGenerationV1 {
    publish_cell_with_window(
        data_root,
        symbol,
        timeframe,
        account_id,
        CANONICAL_TRENDBAR_SERIES_FROM_MS_V1,
        TO_MS_EXCLUSIVE,
    )
}

fn publish_cell_with_window(
    data_root: &std::path::Path,
    symbol: &CanonicalTrendbarSymbolV1,
    timeframe: CanonicalTimeframe,
    account_id: i64,
    requested_from_ms: i64,
    requested_to_ms: i64,
) -> SelectedDatasetGenerationV1 {
    let identity = CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        SERVER,
        account_id,
        symbol.symbol_id(),
        symbol.symbol_name(),
        timeframe,
        BarTimestampConvention::BarOpen,
    )
    .expect("cell identity");
    let step = timeframe
        .fixed_duration_ms()
        .expect("test timeframes are fixed");
    let first = CANONICAL_TRENDBAR_SERIES_FROM_MS_V1 + step;
    let second = first + step;
    let chunk = CanonicalOhlcvChunk {
        timestamp_ms: vec![first, second],
        open: vec![1.10, 1.11],
        high: vec![1.12, 1.13],
        low: vec![1.09, 1.10],
        close: vec![1.11, 1.12],
        volume: CanonicalVolumeChunk::Int64(vec![10, 11]),
    };
    let published = publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
        configured_root: data_root,
        identity: &identity,
        expected_generation: None,
        requested_from_ms,
        requested_to_ms,
        retrieved_unix_ms: 1_767_225_600_000,
        returned_from_ms: first,
        returned_to_ms: second,
        row_count: 2,
        chunks: vec![Ok::<_, anyhow::Error>(chunk)],
    })
    .expect("publish exact test generation");
    SelectedDatasetGenerationV1::from_manifest(published.manifest()).expect("selected cell receipt")
}

fn publish_all_cells(data_root: &std::path::Path) -> Vec<CanonicalTrendbarAcquisitionCellV1> {
    symbols()
        .into_iter()
        .flat_map(|symbol| {
            timeframes().into_iter().map(move |timeframe| {
                CanonicalTrendbarAcquisitionCellV1::new(publish_cell(
                    data_root, &symbol, timeframe, ACCOUNT_ID,
                ))
                .expect("completed cell")
            })
        })
        .collect()
}

#[test]
fn plan_is_fixed_2016_explicit_exact_account_and_canonically_ordered() -> Result<()> {
    assert_eq!(
        CanonicalTrendbarSymbolV1::new(99, "EUR/USD")?.symbol_name(),
        "EUR/USD",
        "the exact broker symbol name is identity data, not a path component"
    );
    assert!(
        CanonicalTrendbarSymbolV1::new(99, "EUR/USD\n").is_err(),
        "control characters must still fail closed"
    );

    let plan = plan();
    assert_eq!(plan.environment(), CTraderEnvironment::Demo);
    assert_eq!(plan.server(), SERVER);
    assert_eq!(plan.account_id(), ACCOUNT_ID);
    assert_eq!(plan.from_ms(), CANONICAL_TRENDBAR_SERIES_FROM_MS_V1);
    assert_eq!(plan.to_ms_exclusive(), TO_MS_EXCLUSIVE);
    assert_eq!(plan.paging_policy_id(), CANONICAL_TRENDBAR_PAGING_POLICY_V1);
    assert_eq!(plan.symbols(), symbols());
    assert_eq!(plan.timeframes(), timeframes());
    assert_eq!(plan.cell_count(), 4);

    assert!(
        CanonicalTrendbarAcquisitionPlanV1::new(
            CTraderEnvironment::Demo,
            SERVER,
            ACCOUNT_ID,
            CANONICAL_TRENDBAR_SERIES_FROM_MS_V1 + 1,
            TO_MS_EXCLUSIVE,
            symbols(),
            timeframes(),
        )
        .is_err(),
        "the lower bound is a versioned fixed authority, not a caller default"
    );
    assert!(
        CanonicalTrendbarAcquisitionPlanV1::new(
            CTraderEnvironment::Demo,
            SERVER,
            ACCOUNT_ID,
            CANONICAL_TRENDBAR_SERIES_FROM_MS_V1,
            TO_MS_EXCLUSIVE,
            vec![symbols()[0].clone(), symbols()[0].clone()],
            timeframes(),
        )
        .is_err(),
        "duplicate symbols must fail closed"
    );
    assert!(
        CanonicalTrendbarAcquisitionPlanV1::new(
            CTraderEnvironment::Demo,
            SERVER,
            ACCOUNT_ID,
            CANONICAL_TRENDBAR_SERIES_FROM_MS_V1,
            TO_MS_EXCLUSIVE,
            symbols(),
            vec![CanonicalTimeframe::M1, CanonicalTimeframe::M1],
        )
        .is_err(),
        "duplicate timeframes must fail closed"
    );

    let mut unknown: Value = serde_json::from_slice(&plan.to_json_bytes()?)?;
    unknown
        .as_object_mut()
        .expect("plan object")
        .insert("current".to_owned(), Value::Bool(true));
    assert!(
        CanonicalTrendbarAcquisitionPlanV1::from_json_bytes(&serde_json::to_vec(&unknown)?)
            .is_err()
    );
    Ok(())
}

#[test]
fn content_addressed_plan_and_checkpoint_reopen_exact_verified_cells() -> Result<()> {
    let data = tempfile::tempdir()?;
    let authority = tempfile::tempdir()?;
    let store = CanonicalTrendbarAcquisitionStoreV1::new(authority.path());
    let plan_receipt = store.publish_plan(&plan())?;
    assert_eq!(store.open_plan(&plan_receipt)?, plan());

    let cells = publish_all_cells(data.path());
    let first = store.publish_checkpoint(data.path(), &plan_receipt, None, cells[..2].to_vec())?;
    let reopened_first = store.open_checkpoint(data.path(), &plan_receipt, &first)?;
    assert_eq!(reopened_first.completed_cells(), &cells[..2]);
    assert_eq!(
        reopened_first.next_cell(&plan())?,
        Some((symbols()[1].clone(), CanonicalTimeframe::M1))
    );

    let second = store.publish_checkpoint(
        data.path(),
        &plan_receipt,
        Some(&first),
        cells[..3].to_vec(),
    )?;
    let reopened_second = store.open_checkpoint(data.path(), &plan_receipt, &second)?;
    assert_eq!(
        reopened_second.previous_checkpoint_sha256(),
        Some(first.sha256())
    );
    assert_eq!(reopened_second.completed_cells(), &cells[..3]);

    assert!(
        store
            .publish_checkpoint(
                data.path(),
                &plan_receipt,
                Some(&first),
                vec![cells[1].clone(), cells[0].clone()],
            )
            .is_err(),
        "out-of-order or non-prefix completion cannot be resumed"
    );
    assert!(
        store
            .publish_checkpoint(
                data.path(),
                &plan_receipt,
                Some(&first),
                vec![cells[0].clone(), cells[0].clone()],
            )
            .is_err(),
        "duplicate completion cannot be resumed"
    );
    Ok(())
}

#[test]
fn checkpoint_refuses_wrong_account_window_and_tampered_content() -> Result<()> {
    let data = tempfile::tempdir()?;
    let authority = tempfile::tempdir()?;
    let store = CanonicalTrendbarAcquisitionStoreV1::new(authority.path());
    let plan_receipt = store.publish_plan(&plan())?;

    let wrong_account = CanonicalTrendbarAcquisitionCellV1::new(publish_cell(
        data.path(),
        &symbols()[0],
        CanonicalTimeframe::M1,
        ACCOUNT_ID + 1,
    ))?;
    assert!(
        store
            .publish_checkpoint(data.path(), &plan_receipt, None, vec![wrong_account])
            .is_err(),
        "a different broker account must not enter the checkpoint"
    );

    let wrong_window_data = tempfile::tempdir()?;
    let wrong_window = CanonicalTrendbarAcquisitionCellV1::new(publish_cell_with_window(
        wrong_window_data.path(),
        &symbols()[0],
        CanonicalTimeframe::M1,
        ACCOUNT_ID,
        CANONICAL_TRENDBAR_SERIES_FROM_MS_V1 + 1,
        TO_MS_EXCLUSIVE,
    ))?;
    assert!(
        store
            .publish_checkpoint(
                wrong_window_data.path(),
                &plan_receipt,
                None,
                vec![wrong_window],
            )
            .is_err(),
        "a generation requested for a different window must not enter the checkpoint"
    );

    let cells = publish_all_cells(data.path());
    let checkpoint =
        store.publish_checkpoint(data.path(), &plan_receipt, None, vec![cells[0].clone()])?;
    let path = store.checkpoint_path(&checkpoint);
    let original = fs::read(&path)?;
    let mut tampered: Value = serde_json::from_slice(&original)?;
    tampered
        .as_object_mut()
        .expect("checkpoint object")
        .insert("plan_sha256".to_owned(), Value::String("0".repeat(64)));
    fs::write(&path, serde_json::to_vec(&tampered)?)?;
    assert!(
        store
            .open_checkpoint(data.path(), &plan_receipt, &checkpoint)
            .is_err(),
        "same-path checkpoint tampering must be detected by its receipt"
    );
    Ok(())
}

#[test]
fn complete_checkpoint_publishes_exact_series_and_matrix_authority_only() -> Result<()> {
    let data = tempfile::tempdir()?;
    let authority = tempfile::tempdir()?;
    let store = CanonicalTrendbarAcquisitionStoreV1::new(authority.path());
    let plan_receipt = store.publish_plan(&plan())?;
    let cells = publish_all_cells(data.path());

    let incomplete =
        store.publish_checkpoint(data.path(), &plan_receipt, None, cells[..3].to_vec())?;
    assert!(
        store
            .publish_matrix(data.path(), &plan_receipt, &incomplete)
            .is_err(),
        "an incomplete matrix cannot become search/training authority"
    );

    let complete =
        store.publish_checkpoint(data.path(), &plan_receipt, Some(&incomplete), cells.clone())?;
    let matrix_receipt = store.publish_matrix(data.path(), &plan_receipt, &complete)?;
    let matrix = store.open_matrix(data.path(), &plan_receipt, &matrix_receipt)?;
    assert_eq!(matrix.plan_sha256(), plan_receipt.sha256());
    assert_eq!(matrix.checkpoint_sha256(), complete.sha256());
    assert_eq!(matrix.series().len(), 2);
    for (series, expected_symbol) in matrix.series().iter().zip(symbols()) {
        assert_eq!(
            series.anchor().identity().symbol_name(),
            expected_symbol.symbol_name()
        );
        assert_eq!(series.direct_timeframes().len(), 2);
        assert_eq!(
            series
                .direct_timeframes()
                .iter()
                .map(|receipt| receipt.identity().timeframe())
                .collect::<Vec<_>>(),
            timeframes()
        );
    }
    assert!(
        !authority.path().join("current").exists(),
        "the content-addressed authority must have no mutable current pointer"
    );
    Ok(())
}

#[test]
fn exact_open_never_creates_a_missing_authority_root() -> Result<()> {
    let parent = tempfile::tempdir()?;
    let missing_root = parent.path().join("missing-authority");
    let store = CanonicalTrendbarAcquisitionStoreV1::new(&missing_root);
    let receipt = CanonicalTrendbarPlanReceiptV1::from_sha256("a".repeat(64))?;
    assert!(store.open_plan(&receipt).is_err());
    assert!(
        !missing_root.exists(),
        "an exact open is read-only and must not create its missing root"
    );
    Ok(())
}
