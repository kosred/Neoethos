//! Integration proof for the shared screening producer, not strategy performance.
//! All sources are test-owned; the two D1 rows are deliberately synthetic.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use neoethos_broker_history::bootstrap_writer::{
    BrokerTrendbarStreamRequest, publish_broker_trendbar_chunks,
};
use neoethos_broker_history::canonical_research_costs::{
    PipValueConversionOperationV1, SCREENING_COST_SCHEMA_V2, ScreeningCostEnvelopeWireV2,
    build_screening_cost_envelope_v2,
};
use neoethos_broker_history::{
    CANONICAL_TRENDBAR_SERIES_FROM_MS_V1, CanonicalTrendbarAcquisitionCellV1,
    CanonicalTrendbarAcquisitionPlanV1, CanonicalTrendbarAcquisitionStoreV1,
    CanonicalTrendbarMatrixReceiptV1, CanonicalTrendbarPlanReceiptV1, CanonicalTrendbarSymbolV1,
};
use neoethos_core::Settings;
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalOhlcvChunk,
    CanonicalTimeframe, CanonicalVolumeChunk, SelectedDatasetGenerationV1,
};
use sha2::{Digest, Sha256};

const SERVER: &str = "demo.ctraderapi.com";
const ACCOUNT_ID: i64 = 42;
const TO_MS_EXCLUSIVE: i64 = 1_767_225_600_000;
const SETTINGS_BYTES: &[u8] = b"system:\n  symbol: EURUSD\n  account_currency: USD\nrisk:\n  backtest_spread_pips: 1.25\n  slippage_pips: 0.5\n  commission_per_lot: 5.625\n  commission_per_lot_is_per_side: true\n";

struct Fixture {
    _directory: tempfile::TempDir,
    data_root: PathBuf,
    store: CanonicalTrendbarAcquisitionStoreV1,
    plan: CanonicalTrendbarPlanReceiptV1,
    matrix: CanonicalTrendbarMatrixReceiptV1,
    settings_path: PathBuf,
    broker_path: PathBuf,
    settings: Settings,
    generation_sha256: String,
    last_timestamp_ms: i64,
}

fn plan(account_id: i64) -> Result<CanonicalTrendbarAcquisitionPlanV1> {
    CanonicalTrendbarAcquisitionPlanV1::new(
        CTraderEnvironment::Demo,
        SERVER,
        account_id,
        CANONICAL_TRENDBAR_SERIES_FROM_MS_V1,
        TO_MS_EXCLUSIVE,
        vec![CanonicalTrendbarSymbolV1::new(1, "EURUSD")?],
        vec![CanonicalTimeframe::D1],
    )
}

impl Fixture {
    fn quantity_contract(
        &mut self,
    ) -> Result<neoethos_core::config::RiskyReplayLotConstraintsConfigV1> {
        let mut document: serde_json::Value =
            serde_json::from_slice(&fs::read(&self.broker_path)?)?;
        let symbol = &mut document["payload"]["symbol"][0];
        symbol["minVolume"] = 100_000.into();
        symbol["maxVolume"] = 1_000_000_000_i64.into();
        symbol["stepVolume"] = 100_000.into();
        let bytes = serde_json::to_vec(&document)?;
        fs::write(&self.broker_path, &bytes)?;
        Ok(neoethos_core::config::RiskyReplayLotConstraintsConfigV1 {
            min_lots: 0.01,
            max_lots: 100.0,
            lot_step: 0.01,
            source_artifact_sha256: format!("{:x}", Sha256::digest(&bytes)),
            fixed_leverage: None,
        })
    }

    fn write_quantity_settings(
        &mut self,
        grid: &neoethos_core::config::RiskyReplayLotConstraintsConfigV1,
    ) -> Result<()> {
        let base = String::from_utf8(SETTINGS_BYTES.to_vec())?
            .replace(
                "account_currency: USD\n",
                "account_currency: USD\n  trading_mode: risky\n",
            )
            .replace("risk:\n", "risk:\n  initial_balance: 100\n");
        let source = format!(
            "{base}models:\n  discovery_runtime:\n    risky_survival_only: true\n    risky_account_experiment_v3: true\n    risky_lot_constraints:\n      min_lots: {}\n      max_lots: {}\n      lot_step: {}\n      source_artifact_sha256: '{}'\n",
            grid.min_lots, grid.max_lots, grid.lot_step, grid.source_artifact_sha256
        );
        fs::write(&self.settings_path, source)?;
        self.settings = Settings::from_yaml(&self.settings_path)?;
        Ok(())
    }

    fn new() -> Result<Self> {
        let directory = tempfile::tempdir()?;
        let data_root = directory.path().join("data");
        let store = CanonicalTrendbarAcquisitionStoreV1::new(directory.path().join("authority"));
        let plan = store.publish_plan(&plan(ACCOUNT_ID)?)?;
        let identity = CanonicalDatasetIdentity::ctrader(
            CTraderEnvironment::Demo,
            SERVER,
            ACCOUNT_ID,
            1,
            "EURUSD",
            CanonicalTimeframe::D1,
            BarTimestampConvention::BarOpen,
        )?;
        // Chosen UTC opens for this January fixture, not an assumption that
        // every broker D1 session has a fixed duration across DST changes.
        let first = CANONICAL_TRENDBAR_SERIES_FROM_MS_V1 + 3 * 86_400_000;
        let last_timestamp_ms = CANONICAL_TRENDBAR_SERIES_FROM_MS_V1 + 4 * 86_400_000;
        let published = publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
            configured_root: &data_root,
            identity: &identity,
            expected_generation: None,
            requested_from_ms: CANONICAL_TRENDBAR_SERIES_FROM_MS_V1,
            requested_to_ms: TO_MS_EXCLUSIVE,
            retrieved_unix_ms: TO_MS_EXCLUSIVE.try_into()?,
            returned_from_ms: first,
            returned_to_ms: last_timestamp_ms,
            row_count: 2,
            chunks: vec![Ok::<_, anyhow::Error>(CanonicalOhlcvChunk {
                timestamp_ms: vec![first, last_timestamp_ms],
                open: vec![1.19, 1.20],
                high: vec![1.21, 1.26],
                low: vec![1.18, 1.19],
                close: vec![1.20, 1.25],
                volume: CanonicalVolumeChunk::Int64(vec![10, 11]),
            })],
        })?;
        let selected = SelectedDatasetGenerationV1::from_manifest(published.manifest())?;
        let generation_sha256 = selected
            .generation_id()
            .strip_prefix("g1-")
            .unwrap()
            .strip_suffix(".vortex")
            .unwrap()
            .to_owned();
        let checkpoint = store.publish_checkpoint(
            &data_root,
            &plan,
            None,
            vec![CanonicalTrendbarAcquisitionCellV1::new(selected)?],
        )?;
        let matrix = store.publish_matrix(&data_root, &plan, &checkpoint)?;
        let settings_path = directory.path().join("settings.yaml");
        fs::write(&settings_path, SETTINGS_BYTES)?;
        let settings = Settings::from_yaml(&settings_path)?;
        let broker_path = directory.path().join("broker-symbol.json");
        fs::write(
            &broker_path,
            serde_json::to_vec(&serde_json::json!({
                "payloadType": 2117,
                "payload": {
                    "ctidTraderAccountId": ACCOUNT_ID,
                    "symbol": [{
                        "symbolId": 1,
                        "pipPosition": 4,
                        "lotSize": 10_000_000,
                        "commissionType": 1,
                        "preciseTradingCommissionRate": 4_500_000_000_i64,
                        "preciseMinCommission": 0,
                        "minCommission": 0,
                        "swapCalculationType": 0,
                        "swapLong": -2.5,
                        "swapShort": 0.75,
                        "pnlConversionFeeRate": 13
                    }]
                }
            }))?,
        )?;
        Ok(Self {
            _directory: directory,
            data_root,
            store,
            plan,
            matrix,
            settings_path,
            broker_path,
            settings,
            generation_sha256,
            last_timestamp_ms,
        })
    }

    fn build(&self, basis: CanonicalTimeframe) -> Result<ScreeningCostEnvelopeWireV2> {
        build_screening_cost_envelope_v2(
            &self.settings,
            &self.data_root,
            &self.store,
            &self.plan,
            &self.matrix,
            "EURUSD",
            basis,
            &self.broker_path,
            &self.settings_path,
        )
    }
}

#[test]
fn risky_lot_limits_must_match_the_exact_broker_contract() -> Result<()> {
    let mut fixture = Fixture::new()?;
    let captured = fixture.quantity_contract()?;
    for (min, max, step) in [(0.02, 100.0, 0.01), (0.01, 99.0, 0.01), (0.01, 100.0, 0.02)] {
        let mut changed = captured.clone();
        changed.min_lots = min;
        changed.max_lots = max;
        changed.lot_step = step;
        fixture.write_quantity_settings(&changed)?;
        let error = fixture.build(CanonicalTimeframe::D1).expect_err(
            "a valid source hash cannot authorize quantity values absent from that source",
        );
        assert!(error.to_string().contains("lot constraints"), "{error:#}");
    }
    Ok(())
}

#[test]
fn risky_quantity_source_must_be_actual_and_uses_the_reported_lot_size() -> Result<()> {
    let mut fixture = Fixture::new()?;
    let mut grid = fixture.quantity_contract()?;
    fixture.write_quantity_settings(&grid)?;
    let costs = fixture.build(CanonicalTimeframe::D1)?;
    assert_eq!(
        costs
            .source_components
            .iter()
            .find(|part| part.role == "broker_symbol_contract")
            .unwrap()
            .sha256,
        grid.source_artifact_sha256
    );
    grid.source_artifact_sha256 = "0".repeat(64);
    fixture.write_quantity_settings(&grid)?;
    let error = fixture.build(CanonicalTimeframe::D1).unwrap_err();
    assert!(error.to_string().contains("lot constraints source"));

    // A broker-defined 10,000-unit lot must not become a conventional 100,000.
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&fixture.broker_path)?)?;
    document["payload"]["symbol"][0]["lotSize"] = 1_000_000.into();
    let bytes = serde_json::to_vec(&document)?;
    fs::write(&fixture.broker_path, &bytes)?;
    grid.min_lots = 0.1;
    grid.max_lots = 1000.0;
    grid.lot_step = 0.1;
    grid.source_artifact_sha256 = format!("{:x}", Sha256::digest(&bytes));
    fixture.write_quantity_settings(&grid)?;
    assert_eq!(
        fixture
            .build(CanonicalTimeframe::D1)?
            .pip_value_quote_per_lot,
        1.0
    );
    Ok(())
}

#[test]
fn risky_quantity_missing_invalid_or_inexact_wire_fields_are_refused_even_after_rehash()
-> Result<()> {
    let mut fixture = Fixture::new()?;
    let mut grid = fixture.quantity_contract()?;
    let base: serde_json::Value = serde_json::from_slice(&fs::read(&fixture.broker_path)?)?;
    for (field, value) in [
        ("minVolume", serde_json::Value::Null),
        ("stepVolume", serde_json::json!(0)),
        ("maxVolume", serde_json::json!(-1)),
        ("minVolume", serde_json::json!(1.5)),
        ("maxVolume", serde_json::json!(9_000_000_000_000_001_i64)),
        ("stepVolume", serde_json::json!(2_000_000_000_i64)),
        ("minVolume", serde_json::json!(2_000_000_000_i64)),
    ] {
        let mut document = base.clone();
        document["payload"]["symbol"][0][field] = value;
        let bytes = serde_json::to_vec(&document)?;
        fs::write(&fixture.broker_path, &bytes)?;
        grid.source_artifact_sha256 = format!("{:x}", Sha256::digest(&bytes));
        fixture.write_quantity_settings(&grid)?;
        let error = fixture.build(CanonicalTimeframe::D1).unwrap_err();
        assert!(
            error.to_string().contains("lot constraints"),
            "{field}: {error:#}"
        );
    }
    Ok(())
}

#[test]
fn risky_quantity_consumers_cannot_mix_facts_from_another_cost_envelope() -> Result<()> {
    use neoethos_broker_history::canonical_research_costs::{
        validate_broker_symbol_contract, validate_costs,
    };
    let mut fixture = Fixture::new()?;
    let mut grid = fixture.quantity_contract()?;
    fixture.write_quantity_settings(&grid)?;
    let original = fixture.build(CanonicalTimeframe::D1)?;
    let mut document: serde_json::Value = serde_json::from_slice(&fs::read(&fixture.broker_path)?)?;
    document["payload"]["symbol"][0]["digits"] = 5.into();
    let changed = serde_json::to_vec(&document)?;
    fs::write(&fixture.broker_path, &changed)?;
    grid.source_artifact_sha256 = format!("{:x}", Sha256::digest(&changed));
    fixture.write_quantity_settings(&grid)?;
    let replacement = fixture.build(CanonicalTimeframe::D1)?;
    let plan = fixture.store.open_plan(&fixture.plan)?;
    let matrix = fixture
        .store
        .open_matrix(&fixture.data_root, &fixture.plan, &fixture.matrix)?;
    let replacement_facts =
        validate_broker_symbol_contract(&changed, &replacement, "EURUSD", &plan)?;
    let error = validate_costs(
        &original,
        "EURUSD",
        &fixture.settings,
        &plan,
        &matrix,
        &fixture.data_root,
        replacement_facts,
    )
    .unwrap_err();
    assert!(error.to_string().contains("lot constraints source"));
    Ok(())
}

#[test]
fn shared_producer_accepts_desktop_settings_and_derives_bound_d1_costs() -> Result<()> {
    let fixture = Fixture::new()?;
    let costs = fixture.build(CanonicalTimeframe::D1)?;
    assert_eq!(costs.schema, SCREENING_COST_SCHEMA_V2);
    assert_eq!(costs.version, 2);
    assert_eq!(costs.source_environment, "demo");
    assert_eq!(costs.source_server, SERVER);
    assert_eq!(costs.source_account_id, ACCOUNT_ID);
    assert_eq!(costs.symbol, "EURUSD");
    assert_eq!(costs.account_currency, "USD");
    assert_eq!(costs.pip_size, 0.0001);
    assert_eq!(costs.pip_value_quote_per_lot, 10.0);
    assert_eq!(
        costs.pip_value_conversion.operation,
        PipValueConversionOperationV1::Identity
    );
    assert_eq!(
        costs.pip_value_conversion.timestamp_ms,
        fixture.last_timestamp_ms
    );
    assert_eq!(
        costs.commission_symbol_price_basis.timestamp_ms,
        fixture.last_timestamp_ms
    );
    assert_eq!(costs.commission_symbol_price_basis.close, 1.25);
    assert_eq!(costs.full_spread_pips_assumption, 1.25);
    assert_eq!(costs.slippage_pips_per_fill_assumption, 0.5);
    // Independent exact oracle: USD 45 per million * USD 125,000 notional.
    assert_eq!(costs.commission_account_per_lot_per_fill_assumption, 5.625);
    assert_eq!(costs.swap_long_pips_per_day, -2.5);
    assert_eq!(costs.swap_short_pips_per_day, 0.75);
    // The captured symbol advertises 13 (=0.13%), but quote==deposit USD.
    // No conversion occurs; it must not levy that symbol rate on USD P&L.
    assert_eq!(costs.pnl_conversion_fee_rate, 0.0);
    assert_eq!(costs.pnl_conversion_fee_policy,
        neoethos_core::research_conversion_fee::ResearchPnlConversionFeePolicyV1::AbsoluteRealizedPriceGrossDebitV1);
    let source = |role: &str| {
        &costs
            .source_components
            .iter()
            .find(|component| component.role == role)
            .unwrap()
            .sha256
    };
    assert_eq!(costs.source_components.len(), 4);
    assert_eq!(
        source("settings"),
        &format!("{:x}", Sha256::digest(SETTINGS_BYTES))
    );
    assert_eq!(
        source("broker_symbol_contract"),
        &format!("{:x}", Sha256::digest(fs::read(&fixture.broker_path)?))
    );
    assert_eq!(source("pip_value_basis"), &fixture.generation_sha256);
    assert_eq!(
        source("commission_symbol_price_basis"),
        &fixture.generation_sha256
    );
    let wire = serde_json::to_vec(&costs)?;
    let mut legacy = serde_json::to_value(&costs)?;
    legacy
        .as_object_mut()
        .unwrap()
        .remove("pnl_conversion_fee_policy");
    assert!(serde_json::from_value::<ScreeningCostEnvelopeWireV2>(legacy).is_err());
    assert_eq!(
        serde_json::from_slice::<ScreeningCostEnvelopeWireV2>(&wire)?,
        costs
    );
    assert_eq!(fixture.build(CanonicalTimeframe::D1)?, costs);
    Ok(())
}

#[test]
fn shared_producer_keeps_the_rate_when_quote_currency_really_requires_conversion() -> Result<()> {
    let mut fixture = Fixture::new()?;
    let source = String::from_utf8(SETTINGS_BYTES.to_vec())?
        .replace("account_currency: USD", "account_currency: EUR");
    fs::write(&fixture.settings_path, source.as_bytes())?;
    fixture.settings = Settings::from_yaml(&fixture.settings_path)?;
    let costs = fixture.build(CanonicalTimeframe::D1)?;
    assert_eq!(costs.account_currency, "EUR");
    assert_eq!(costs.pnl_conversion_fee_rate, 0.0013);
    assert_eq!(
        costs.pip_value_conversion.operation,
        PipValueConversionOperationV1::Divide
    );
    assert_eq!(costs.pip_value_conversion.symbol, "EURUSD");
    assert_eq!(costs.commission_account_per_lot_per_fill_assumption, 4.5);
    Ok(())
}

#[test]
fn explicit_matrix_symbol_is_independent_of_the_saved_default_symbol() -> Result<()> {
    let mut fixture = Fixture::new()?;
    let source =
        String::from_utf8(SETTINGS_BYTES.to_vec())?.replace("symbol: EURUSD", "symbol: GBPUSD");
    fs::write(&fixture.settings_path, source.as_bytes())?;
    fixture.settings = Settings::from_yaml(&fixture.settings_path)?;

    let costs = fixture.build(CanonicalTimeframe::D1)?;
    assert_eq!(fixture.settings.system.symbol, "GBPUSD");
    assert_eq!(costs.symbol, "EURUSD");
    assert_eq!(costs.source_account_id, ACCOUNT_ID);
    assert_eq!(costs.commission_account_per_lot_per_fill_assumption, 5.625);
    assert_eq!(
        costs
            .source_components
            .iter()
            .find(|part| part.role == "settings")
            .unwrap()
            .sha256,
        format!("{:x}", Sha256::digest(source.as_bytes()))
    );

    fixture.settings.system.symbol = "EURUSD".to_owned();
    let error = fixture.build(CanonicalTimeframe::D1).unwrap_err();
    assert!(format!("{error:#}").contains("exact Settings used by this process"));
    Ok(())
}

#[test]
fn shared_producer_rejects_a_matrix_from_a_different_plan() -> Result<()> {
    let mut fixture = Fixture::new()?;
    fixture.plan = fixture.store.publish_plan(&plan(ACCOUNT_ID + 1)?)?;
    let error = fixture.build(CanonicalTimeframe::D1).unwrap_err();
    assert!(format!("{error:#}").contains("plan"));
    Ok(())
}

#[test]
fn shared_producer_requires_direct_d1_and_unchanged_resolved_settings() -> Result<()> {
    let mut fixture = Fixture::new()?;
    let error = fixture.build(CanonicalTimeframe::H1).unwrap_err();
    assert!(error.to_string().contains("explicit direct D1"));
    fixture.settings.risk.slippage_pips = 3.0;
    let error = fixture.build(CanonicalTimeframe::D1).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("exact Settings used by this process")
    );
    Ok(())
}

#[test]
fn shared_producer_rejects_the_wrong_broker_symbol_before_costs() -> Result<()> {
    let fixture = Fixture::new()?;
    let mut broker: serde_json::Value = serde_json::from_slice(&fs::read(&fixture.broker_path)?)?;
    broker["payload"]["symbol"][0]["symbolId"] = 2.into();
    fs::write(&fixture.broker_path, serde_json::to_vec(&broker)?)?;
    let error = fixture.build(CanonicalTimeframe::D1).unwrap_err();
    assert!(error.to_string().contains("symbol id does not match"));
    Ok(())
}
