//! Read-only headless entry to the same broker money parser as the desktop.
//! Retains broker responses, actual currency and deal truncation; never creates
//! historical cost policy, quote authority or a trading permit.

use std::cell::RefCell;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::ctrader_account::{
    CTraderAccountRuntimeRequest, CTraderAccountRuntimeSnapshot,
    load_account_runtime_with_transport, parse_deal_list_bundle_response,
};
use crate::ctrader_live_auth::CTraderEnvironment;
use crate::ctrader_messages::{
    CTraderOpenApiJsonMessage, CTraderOpenApiTransport, ProductionCTraderOpenApiTransport,
};
use crate::{BrokerEnvironment, load_exact_production_historical_credentials};

const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;
const RESPONSE_NAMES: [&str; 5] = [
    "trader.json",
    "reconcile.json",
    "deals.json",
    "unrealized-pnl.json",
    "assets.json",
];

/// Validate saved source integrity and the observed account currency. This
/// current-account evidence does not authorize historical pricing or trading.
pub fn validate_saved_account_currency(
    path: &Path,
    environment: BrokerEnvironment,
    account_id: i64,
    configured_currency: &str,
) -> Result<String> {
    use crate::canonical_research_costs::read_regular_file_with_limit;
    use crate::ctrader_account::parse_trader_response;
    use crate::ctrader_data::parse_asset_list_response;
    use crate::ctrader_messages::parse_open_api_envelope;
    let bytes = read_regular_file_with_limit(path, MAX_RESPONSE_BYTES as u64)?;
    let document: Value = serde_json::from_slice(&bytes)?;
    let name = match environment {
        BrokerEnvironment::Demo => "demo",
        BrokerEnvironment::Live => "live",
    };
    ensure!(
        document["schema"] == "neoethos.broker-account-snapshot.v1"
            && document["environment"] == name
            && document["server"] == environment.endpoint_host()
            && document["account_id"].as_i64() == Some(account_id),
        "account snapshot does not match the exact broker acquisition account"
    );
    let started = document["capture_started_unix_ms"]
        .as_i64()
        .context("missing capture start")?;
    let completed = document["capture_completed_unix_ms"]
        .as_i64()
        .context("missing capture end")?;
    ensure!(
        started > 0 && completed >= started,
        "invalid account capture timestamps"
    );
    let sources = document["source_responses"]
        .as_array()
        .context("missing account source responses")?;
    ensure!(
        sources.len() == RESPONSE_NAMES.len(),
        "incomplete account response inventory"
    );
    let parent = path.parent().context("account snapshot has no parent")?;
    let mut responses = Vec::with_capacity(RESPONSE_NAMES.len());
    let mut total_bytes = 0_usize;
    for ((source, name), payload_type) in sources
        .iter()
        .zip(RESPONSE_NAMES)
        .zip([2122, 2125, 2134, 2188, 2113])
    {
        ensure!(
            source["path"] == name,
            "account response inventory differs from fixed capture layout"
        );
        let response = read_regular_file_with_limit(&parent.join(name), MAX_RESPONSE_BYTES as u64)?;
        total_bytes += response.len();
        ensure!(
            total_bytes <= MAX_RESPONSE_BYTES,
            "account source responses exceed capture bound"
        );
        ensure!(
            source["bytes"].as_u64() == Some(response.len() as u64)
                && source["sha256"] == format!("{:x}", Sha256::digest(&response)),
            "saved broker account response differs from its captured hash"
        );
        let text = String::from_utf8(response)?;
        let envelope = parse_open_api_envelope(&text)?;
        ensure!(
            envelope.payload_type == payload_type
                && envelope.payload["ctidTraderAccountId"].as_i64() == Some(account_id),
            "account source response identity mismatch"
        );
        responses.push(text);
    }
    let trader = parse_trader_response(&responses[0])?;
    let asset_id = trader
        .deposit_asset_id
        .context("broker trader omitted depositAssetId")?;
    let assets = parse_asset_list_response(&responses[4])?;
    let matches = assets
        .iter()
        .filter(|asset| asset.asset_id == asset_id)
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1 && !matches[0].name.trim().is_empty(),
        "account deposit asset has no unique broker name"
    );
    let currency = &matches[0].name;
    ensure!(
        document["financial"]["account_currency"] == *currency,
        "saved account currency differs from exact broker responses"
    );
    ensure!(
        configured_currency == currency,
        "configured account currency {configured_currency} differs from actual broker account currency {currency}"
    );
    Ok(format!("{:x}", Sha256::digest(&bytes)))
}

pub fn freeze_account_currency_evidence(
    source: &Path,
    destination: &Path,
    environment: BrokerEnvironment,
    account_id: i64,
    configured_currency: &str,
) -> Result<()> {
    use crate::canonical_research_costs::read_regular_file_with_limit;
    let before =
        validate_saved_account_currency(source, environment, account_id, configured_currency)?;
    fs::create_dir(destination).context("create new frozen account evidence directory")?;
    let parent = source.parent().context("account snapshot has no parent")?;
    for name in RESPONSE_NAMES {
        fs::write(
            destination.join(name),
            read_regular_file_with_limit(&parent.join(name), MAX_RESPONSE_BYTES as u64)?,
        )?;
    }
    let frozen = destination.join("account-snapshot.json");
    fs::write(
        &frozen,
        read_regular_file_with_limit(source, MAX_RESPONSE_BYTES as u64)?,
    )?;
    let after =
        validate_saved_account_currency(&frozen, environment, account_id, configured_currency)?;
    ensure!(
        before == after,
        "account evidence changed while freezing research inputs"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum SnapshotEnvironment {
    Demo,
    Live,
}

impl SnapshotEnvironment {
    fn broker(self) -> BrokerEnvironment {
        match self {
            Self::Demo => BrokerEnvironment::Demo,
            Self::Live => BrokerEnvironment::Live,
        }
    }
    fn transport(self) -> CTraderEnvironment {
        match self {
            Self::Demo => CTraderEnvironment::Demo,
            Self::Live => CTraderEnvironment::Live,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Demo => "demo",
            Self::Live => "live",
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Capture actual broker account money without sending orders")]
pub struct AccountSnapshotCli {
    #[arg(long, value_enum)]
    pub environment: SnapshotEnvironment,
    #[arg(long)]
    pub account_id: i64,
    #[arg(long)]
    pub out_dir: PathBuf,
}

// Only the last successful response sequence is retained when the shared
// transport retries. Requests/authentication responses are never published.
struct RecordedAccountSequence {
    responses: Vec<String>,
    deal_request: Value,
}

struct RecordingTransport<T> {
    inner: T,
    recorded: RefCell<Option<RecordedAccountSequence>>,
}

impl<T: CTraderOpenApiTransport> CTraderOpenApiTransport for RecordingTransport<T> {
    fn send_sequence(&self, messages: &[CTraderOpenApiJsonMessage]) -> Result<Vec<String>> {
        let responses = self.inner.send_sequence(messages)?;
        ensure!(
            responses.iter().map(String::len).sum::<usize>() <= MAX_RESPONSE_BYTES,
            "broker account response exceeds capture bound"
        );
        ensure!(messages.len() == 7, "unexpected account request sequence");
        *self.recorded.borrow_mut() = Some(RecordedAccountSequence {
            responses: responses.clone(),
            deal_request: messages[4].payload.clone(),
        });
        Ok(responses)
    }
}

fn now_ms() -> Result<i64> {
    i64::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis())
        .context("capture timestamp exceeds i64")
}

fn financial_report(
    snapshot: &CTraderAccountRuntimeSnapshot,
    recorded: &RecordedAccountSequence,
) -> Result<Value> {
    ensure!(recorded.responses.len() == 7, "incomplete account capture");
    // The desktop's legacy Vec view drops hasMore. Keep the existing bundle
    // parser's completeness result in this evidence artifact instead.
    let deals = parse_deal_list_bundle_response(&recorded.responses[4])?;
    ensure!(
        deals.account_id == snapshot.trader.account_id && deals.deals == snapshot.recent_deals,
        "deal evidence differs from the validated account snapshot"
    );
    let mut rows = Vec::with_capacity(deals.deals.len());
    for deal in &deals.deals {
        rows.push(json!({
            "deal_id": deal.deal_id,
            "order_id": deal.order_id,
            "position_id": deal.position_id,
            "symbol_id": deal.symbol_id,
            "trade_side": deal.trade_side,
            "deal_status": deal.deal_status,
            "filled_volume_raw_centi_units": deal.filled_volume_raw_centi_units,
            "execution_timestamp_ms": deal.execution_timestamp_ms,
            "execution_price": deal.execution_price,
            "entry_price": deal.entry_price,
            "money_digits": deal.money_digits,
            "gross_profit": deal.gross_profit,
            "commission_signed": deal.fee,
            "swap_signed": deal.swap,
            "pnl_conversion_fee": deal.pnl_conversion_fee,
            "gross_profit_raw_scaled": deal.gross_profit_raw_scaled,
            "commission_raw_scaled_signed": deal.commission_raw_scaled_signed,
            "swap_raw_scaled_signed": deal.swap_raw_scaled_signed,
            "component_sum_account_currency": deal.component_sum_account_currency,
        }));
    }
    let pnl = snapshot
        .unrealized_pnl_by_position
        .values()
        .map(|p| {
            json!({
                "position_id": p.position_id,
                "gross_unrealized_pnl": p.gross_unrealized_pnl,
                "net_unrealized_pnl": p.net_unrealized_pnl,
            })
        })
        .collect::<Vec<_>>();
    let equity = snapshot.trader.balance + snapshot.unrealized_pnl;
    ensure!(equity.is_finite(), "broker account equity is not finite");
    Ok(json!({
        "account_currency": snapshot.deposit_asset_name,
        "deposit_asset_id": snapshot.trader.deposit_asset_id,
        "money_digits": snapshot.trader.money_digits,
        "balance": snapshot.trader.balance,
        "broker_net_unrealized_pnl": snapshot.unrealized_pnl,
        "equity": equity,
        "open_positions": snapshot.reconcile.positions.len(),
        "pending_orders": snapshot.reconcile.pending_orders.len(),
        "unrealized_pnl_by_position": pnl,
        "recent_deal_request": recorded.deal_request,
        "recent_deals_has_more": deals.has_more,
        "recent_deals_complete_for_requested_window": !deals.has_more,
        "recent_deals": rows,
    }))
}

pub fn capture(cli: AccountSnapshotCli) -> Result<Value> {
    ensure!(cli.account_id > 0, "broker account id must be positive");
    ensure!(cli.out_dir.is_absolute(), "out-dir must be absolute");
    // create_dir is deliberately exclusive. A failed run is retained and a
    // later invocation must select a new directory, never overwrite evidence.
    fs::create_dir(&cli.out_dir).context("create new broker account evidence directory")?;
    let started = now_ms()?;
    let credentials =
        load_exact_production_historical_credentials(cli.environment.broker(), cli.account_id)?;
    let transport = RecordingTransport {
        inner: ProductionCTraderOpenApiTransport::new(cli.environment.broker().endpoint_host()),
        recorded: RefCell::new(None),
    };
    let snapshot = load_account_runtime_with_transport(
        &transport,
        &CTraderAccountRuntimeRequest {
            client_id: credentials.client_id,
            client_secret: credentials.client_secret,
            access_token: credentials.access_token,
            environment: cli.environment.transport(),
            account_id: cli.account_id.to_string(),
            return_protection_orders: true,
        },
    )?;
    let completed = now_ms()?;
    let recorded = transport
        .recorded
        .into_inner()
        .context("missing account responses")?;
    let financial = financial_report(&snapshot, &recorded)?;
    let mut files = Vec::new();
    for (index, name) in [
        (2, "trader.json"),
        (3, "reconcile.json"),
        (4, "deals.json"),
        (5, "unrealized-pnl.json"),
        (6, "assets.json"),
    ] {
        let bytes = recorded.responses[index].as_bytes();
        fs::write(cli.out_dir.join(name), bytes).context("preserve broker response bytes")?;
        files.push(json!({"path": name, "bytes": bytes.len(), "sha256": format!("{:x}", Sha256::digest(bytes))}));
    }
    let report = json!({
        "schema": "neoethos.broker-account-snapshot.v1",
        "environment": cli.environment.name(),
        "server": cli.environment.broker().endpoint_host(),
        "account_id": cli.account_id,
        "capture_started_unix_ms": started,
        "capture_completed_unix_ms": completed,
        "financial": financial,
        "source_responses": files,
        "historical_cost_policy_evaluated": false,
        "authorization_issued": false,
    });
    fs::write(
        cli.out_dir.join("account-snapshot.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ctrader_account::{CTraderReconcileSnapshot, CTraderTraderSnapshot};
    use std::collections::BTreeMap;

    fn saved_currency_fixture(dir: &Path) -> PathBuf {
        use crate::ctrader_messages::CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE;
        let mut sources = Vec::new();
        for name in RESPONSE_NAMES {
            let value = match name {
                "trader.json" => {
                    json!({"payloadType":2122,"payload":{"ctidTraderAccountId":91,"trader":{"balance":1234,"moneyDigits":2,"depositAssetId":7}}})
                }
                "assets.json" => {
                    json!({"payloadType":CTRADER_OA_ASSET_LIST_RESPONSE_PAYLOAD_TYPE,"payload":{"ctidTraderAccountId":91,"asset":[{"assetId":7,"name":"EUR"}]}})
                }
                "reconcile.json" => {
                    json!({"payloadType":2125,"payload":{"ctidTraderAccountId":91}})
                }
                "unrealized-pnl.json" => {
                    json!({"payloadType":2188,"payload":{"ctidTraderAccountId":91}})
                }
                _ => json!({"payloadType":2134,"payload":{"ctidTraderAccountId":91}}),
            };
            let bytes = serde_json::to_vec(&value).unwrap();
            fs::write(dir.join(name), &bytes).unwrap();
            sources.push(json!({"path":name,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
        }
        let path = dir.join("account-snapshot.json");
        fs::write(
            &path,
            serde_json::to_vec(&json!({
                "schema":"neoethos.broker-account-snapshot.v1", "environment":"demo",
                "server":"demo.ctraderapi.com", "account_id":91,
                "capture_started_unix_ms":1, "capture_completed_unix_ms":2,
                "financial":{"account_currency":"EUR"}, "source_responses":sources,
            }))
            .unwrap(),
        )
        .unwrap();
        path
    }

    #[test]
    fn saved_currency_refuses_config_mismatch_wrong_account_and_source_tampering() {
        let dir = tempfile::tempdir().unwrap();
        let path = saved_currency_fixture(dir.path());
        validate_saved_account_currency(&path, BrokerEnvironment::Demo, 91, "EUR").unwrap();
        let error =
            validate_saved_account_currency(&path, BrokerEnvironment::Demo, 91, "GBP").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("GBP differs from actual broker account currency EUR")
        );
        assert!(
            validate_saved_account_currency(&path, BrokerEnvironment::Demo, 92, "EUR").is_err()
        );
        assert!(
            validate_saved_account_currency(&path, BrokerEnvironment::Live, 91, "EUR").is_err()
        );
        let frozen = dir.path().join("frozen");
        freeze_account_currency_evidence(&path, &frozen, BrokerEnvironment::Demo, 91, "EUR")
            .unwrap();
        assert_eq!(
            fs::read(&path).unwrap(),
            fs::read(frozen.join("account-snapshot.json")).unwrap()
        );
        fs::write(dir.path().join("assets.json"), b"{}").unwrap();
        assert!(
            validate_saved_account_currency(&path, BrokerEnvironment::Demo, 91, "EUR").is_err()
        );
        validate_saved_account_currency(
            &frozen.join("account-snapshot.json"),
            BrokerEnvironment::Demo,
            91,
            "EUR",
        )
        .unwrap();
    }

    #[test]
    fn actual_currency_and_truncated_deals_are_preserved() {
        let snapshot = CTraderAccountRuntimeSnapshot {
            environment: CTraderEnvironment::Demo,
            trader: CTraderTraderSnapshot {
                account_id: 91,
                balance: 12.34,
                leverage: None,
                trader_login: None,
                account_type: None,
                broker_name: None,
                money_digits: 2,
                deposit_asset_id: Some(7),
            },
            reconcile: CTraderReconcileSnapshot {
                account_id: 91,
                positions: vec![],
                pending_orders: vec![],
            },
            recent_deals: vec![],
            unrealized_pnl: 0.0,
            unrealized_pnl_by_position: BTreeMap::new(),
            deposit_asset_name: "EUR".into(),
        };
        let mut responses = vec![String::new(); 7];
        responses[4] = json!({"payloadType": 2134, "payload": {"ctidTraderAccountId":91,"deal":[],"hasMore":true}}).to_string();
        let mut recorded = RecordedAccountSequence {
            responses,
            deal_request: json!({"fromTimestamp":1,"toTimestamp":2,"maxRows":100}),
        };
        let report = financial_report(&snapshot, &recorded).unwrap();
        assert_eq!(report["account_currency"], "EUR");
        assert_eq!(report["balance"], 12.34);
        assert_eq!(report["recent_deals_has_more"], true);
        assert_eq!(report["recent_deals_complete_for_requested_window"], false);
        recorded.responses[4] = json!({"payloadType": 2134, "payload": {"ctidTraderAccountId":92,"deal":[],"hasMore":false}}).to_string();
        assert!(financial_report(&snapshot, &recorded).is_err());
    }

    #[test]
    fn refuses_existing_output_before_loading_credentials() {
        let dir = tempfile::tempdir().unwrap();
        let sentinel = dir.path().join("account-snapshot.json");
        fs::write(&sentinel, b"original").unwrap();
        assert!(
            capture(AccountSnapshotCli {
                environment: SnapshotEnvironment::Demo,
                account_id: 91,
                out_dir: dir.path().to_owned()
            })
            .is_err()
        );
        assert_eq!(fs::read(sentinel).unwrap(), b"original");
    }
}
