//! Exact-account cTrader symbol-contract evidence capture.

use crate::ctrader_messages::{
    CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
    CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE, CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
    CTraderOpenApiJsonMessage, CTraderOpenApiSessionResponse, ProductionCTraderOpenApiSession,
    ProductionCTraderOpenApiTransport, build_account_auth_request, build_application_auth_request,
    build_symbol_by_id_request, build_symbols_list_request,
    ctrader_historical_session_error_from_response, parse_open_api_envelope,
};
use crate::{
    BrokerEnvironment, HistoricalCredentials, load_exact_production_historical_credentials,
};
use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const LIGHT_SYMBOLS_CLIENT_MESSAGE_ID_V1: &str = "symbol-contract-light-symbols";
const FULL_SYMBOL_CLIENT_MESSAGE_ID_V1: &str = "symbol-contract-full-symbol";
// A symbols-list response contains the broker's entire catalog, not one
// instrument's cost contract (the latter has the separate 64 KiB bound).
const MAX_CACHED_SYMBOL_CATALOG_BYTES_V1: u64 = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ExactBrokerSymbolEnvironmentArgV1 {
    Demo,
    Live,
}

impl ExactBrokerSymbolEnvironmentArgV1 {
    const fn broker(self) -> BrokerEnvironment {
        match self {
            Self::Demo => BrokerEnvironment::Demo,
            Self::Live => BrokerEnvironment::Live,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactBrokerSymbolContractBindingV1 {
    environment: BrokerEnvironment,
    server: String,
    account_id: i64,
    symbol_id: i64,
    symbol_name: String,
}

impl ExactBrokerSymbolContractBindingV1 {
    pub fn new(
        environment: BrokerEnvironment,
        account_id: i64,
        symbol_id: i64,
        symbol_name: impl Into<String>,
    ) -> Result<Self> {
        ensure!(
            account_id > 0,
            "exact broker symbol account id must be positive"
        );
        ensure!(symbol_id > 0, "exact broker symbol id must be positive");
        let symbol_name = symbol_name.into();
        ensure!(
            !symbol_name.is_empty()
                && symbol_name.len() <= 64
                && symbol_name.trim() == symbol_name
                && symbol_name.bytes().all(|byte| byte.is_ascii_graphic()),
            "exact broker symbol name is invalid"
        );
        Ok(Self {
            environment,
            server: environment.endpoint_host().to_owned(),
            account_id,
            symbol_id,
            symbol_name,
        })
    }

    pub const fn environment(&self) -> BrokerEnvironment {
        self.environment
    }

    pub fn server(&self) -> &str {
        &self.server
    }

    pub const fn account_id(&self) -> i64 {
        self.account_id
    }

    pub const fn symbol_id(&self) -> i64 {
        self.symbol_id
    }

    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "neoethos-broker-symbol-contract",
    about = "Capture one exact-account cTrader full-symbol contract"
)]
pub struct ExactBrokerSymbolContractCaptureCliV1 {
    #[arg(long, value_enum)]
    environment: ExactBrokerSymbolEnvironmentArgV1,
    #[arg(long)]
    account_id: i64,
    #[arg(long)]
    symbol_id: i64,
    #[arg(long)]
    symbol_name: String,
    #[arg(long)]
    output_root: PathBuf,
}

impl ExactBrokerSymbolContractCaptureCliV1 {
    pub fn try_parse_from<I, T>(args: I) -> std::result::Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        <Self as Parser>::try_parse_from(args)
    }

    pub fn prepare(self) -> Result<PreparedExactBrokerSymbolContractCaptureV1> {
        PreparedExactBrokerSymbolContractCaptureV1::new(
            ExactBrokerSymbolContractBindingV1::new(
                self.environment.broker(),
                self.account_id,
                self.symbol_id,
                self.symbol_name,
            )?,
            self.output_root,
        )
    }
}

pub struct PreparedExactBrokerSymbolContractCaptureV1 {
    binding: ExactBrokerSymbolContractBindingV1,
    output_root: PathBuf,
}

impl PreparedExactBrokerSymbolContractCaptureV1 {
    /// Typed entry shared by the desktop and CLI. Preparation performs no
    /// broker request; capture remains an explicit later operation.
    pub fn new(binding: ExactBrokerSymbolContractBindingV1, output_root: PathBuf) -> Result<Self> {
        ensure!(
            output_root.is_absolute(),
            "symbol-contract output-root must be an explicit absolute path"
        );
        fs::create_dir_all(&output_root).with_context(|| {
            format!(
                "create exact broker symbol output root {}",
                output_root.display()
            )
        })?;
        let metadata = fs::symlink_metadata(&output_root).with_context(|| {
            format!(
                "inspect exact broker symbol output root {}",
                output_root.display()
            )
        })?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "symbol-contract output-root must be one real directory"
        );
        Ok(Self {
            binding,
            output_root,
        })
    }
    pub const fn binding(&self) -> &ExactBrokerSymbolContractBindingV1 {
        &self.binding
    }

    pub fn output_root(&self) -> &Path {
        &self.output_root
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactBrokerSymbolContractReceiptV1 {
    binding: ExactBrokerSymbolContractBindingV1,
    light_symbols_sha256: String,
    full_symbol_sha256: String,
    light_symbols_path: PathBuf,
    full_symbol_path: PathBuf,
}

impl ExactBrokerSymbolContractReceiptV1 {
    pub const fn binding(&self) -> &ExactBrokerSymbolContractBindingV1 {
        &self.binding
    }

    pub fn light_symbols_sha256(&self) -> &str {
        &self.light_symbols_sha256
    }

    pub fn full_symbol_sha256(&self) -> &str {
        &self.full_symbol_sha256
    }

    pub fn light_symbols_path(&self) -> &Path {
        &self.light_symbols_path
    }

    pub fn full_symbol_path(&self) -> &Path {
        &self.full_symbol_path
    }
}

fn exact_document(bytes: &[u8], label: &str) -> Result<Value> {
    ensure!(!bytes.is_empty(), "{label} response is empty");
    serde_json::from_slice(bytes).with_context(|| format!("decode exact {label} response"))
}

fn exact_payload<'a>(
    document: &'a Value,
    expected_payload_type: i64,
    expected_client_message_id: &str,
    label: &str,
) -> Result<&'a serde_json::Map<String, Value>> {
    ensure!(
        document.get("payloadType").and_then(Value::as_i64) == Some(expected_payload_type),
        "{label} response payload type differs from the exact request"
    );
    ensure!(
        document.get("clientMsgId").and_then(Value::as_str) == Some(expected_client_message_id),
        "{label} response client message id differs from the exact request"
    );
    document
        .get("payload")
        .and_then(Value::as_object)
        .with_context(|| format!("exact {label} response omits its payload"))
}

fn validate_light_symbols_response(
    binding: &ExactBrokerSymbolContractBindingV1,
    bytes: &[u8],
) -> Result<()> {
    let document = exact_document(bytes, "light-symbol")?;
    let payload = exact_payload(
        &document,
        i64::from(CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE),
        LIGHT_SYMBOLS_CLIENT_MESSAGE_ID_V1,
        "light-symbol",
    )?;
    ensure!(
        payload.get("ctidTraderAccountId").and_then(Value::as_i64) == Some(binding.account_id()),
        "light-symbol response account differs from the exact binding"
    );
    let symbols = payload
        .get("symbol")
        .and_then(Value::as_array)
        .context("light-symbol response omits its symbol array")?;
    let matching = symbols
        .iter()
        .filter(|symbol| {
            symbol.get("symbolId").and_then(Value::as_i64) == Some(binding.symbol_id())
        })
        .collect::<Vec<_>>();
    ensure!(
        matching.len() == 1
            && matching[0].get("symbolName").and_then(Value::as_str) == Some(binding.symbol_name()),
        "light-symbol response does not bind exactly one requested id/name"
    );
    Ok(())
}

fn validate_full_symbol_response(
    binding: &ExactBrokerSymbolContractBindingV1,
    bytes: &[u8],
) -> Result<()> {
    let document = exact_document(bytes, "full-symbol")?;
    let payload = exact_payload(
        &document,
        i64::from(CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE),
        FULL_SYMBOL_CLIENT_MESSAGE_ID_V1,
        "full-symbol",
    )?;
    ensure!(
        payload.get("ctidTraderAccountId").and_then(Value::as_i64) == Some(binding.account_id()),
        "full-symbol response account differs from the exact binding"
    );
    let symbols = payload
        .get("symbol")
        .and_then(Value::as_array)
        .context("full-symbol response omits its symbol array")?;
    ensure!(
        symbols.len() == 1
            && symbols[0].get("symbolId").and_then(Value::as_i64) == Some(binding.symbol_id()),
        "full-symbol response does not contain exactly the requested symbol id"
    );
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn publish_content_addressed_json(
    root: &Path,
    prefix: &str,
    bytes: &[u8],
) -> Result<(String, PathBuf)> {
    let sha256 = sha256_hex(bytes);
    let path = root.join(format!("{prefix}-{sha256}.json"));
    match OpenOptions::new().write(true).create_new(true).open(&path) {
        Ok(mut file) => {
            file.write_all(bytes)
                .with_context(|| format!("write exact artifact {}", path.display()))?;
            file.sync_all()
                .with_context(|| format!("sync exact artifact {}", path.display()))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = fs::read(&path)
                .with_context(|| format!("reopen exact artifact {}", path.display()))?;
            ensure!(
                existing == bytes,
                "content-addressed broker symbol artifact differs from its digest"
            );
        }
        Err(error) => {
            return Err(error).with_context(|| format!("create exact artifact {}", path.display()));
        }
    }
    let reopened =
        fs::read(&path).with_context(|| format!("verify exact artifact {}", path.display()))?;
    ensure!(
        reopened == bytes && sha256_hex(&reopened) == sha256,
        "reopened broker symbol artifact differs from exact broker bytes"
    );
    Ok((sha256, path))
}

pub fn publish_validated_broker_symbol_contract_response_v1(
    output_root: &Path,
    binding: &ExactBrokerSymbolContractBindingV1,
    light_symbols_response: &[u8],
    full_symbol_response: &[u8],
) -> Result<ExactBrokerSymbolContractReceiptV1> {
    ensure!(
        output_root.is_absolute(),
        "broker symbol artifact root must be absolute"
    );
    let metadata = fs::symlink_metadata(output_root).with_context(|| {
        format!(
            "inspect broker symbol artifact root {}",
            output_root.display()
        )
    })?;
    ensure!(
        metadata.is_dir() && !metadata.file_type().is_symlink(),
        "broker symbol artifact root must be one real directory"
    );
    validate_light_symbols_response(binding, light_symbols_response)?;
    validate_full_symbol_response(binding, full_symbol_response)?;
    let (light_symbols_sha256, light_symbols_path) =
        publish_content_addressed_json(output_root, "bsl1", light_symbols_response)?;
    let (full_symbol_sha256, full_symbol_path) =
        publish_content_addressed_json(output_root, "bsc1", full_symbol_response)?;
    Ok(ExactBrokerSymbolContractReceiptV1 {
        binding: binding.clone(),
        light_symbols_sha256,
        full_symbol_sha256,
        light_symbols_path,
        full_symbol_path,
    })
}

fn exchange_expected(
    session: &mut ProductionCTraderOpenApiSession,
    message: &CTraderOpenApiJsonMessage,
    expected_payload_type: u32,
    stage: &'static str,
) -> Result<Vec<u8>> {
    let response = match session.send_one(message, None)? {
        CTraderOpenApiSessionResponse::Expected(response) => response,
        CTraderOpenApiSessionResponse::BrokerError(response) => {
            return Err(broker_rejection_for_stage(stage, &response)?);
        }
    };
    let envelope = parse_open_api_envelope(&response)
        .context("decode exact broker symbol contract exchange")?;
    ensure!(
        envelope.payload_type == expected_payload_type
            && envelope.client_msg_id == message.client_msg_id,
        "broker symbol contract response differs from its exact request"
    );
    Ok(response.into_bytes())
}

fn broker_rejection_for_stage(stage: &'static str, response: &str) -> Result<anyhow::Error> {
    let error = ctrader_historical_session_error_from_response(response)?;
    Ok(error.context(format!(
        "cTrader rejected exact broker symbol contract {stage} request"
    )))
}

fn authenticate_exact_session(
    binding: &ExactBrokerSymbolContractBindingV1,
    credentials: &HistoricalCredentials,
) -> Result<ProductionCTraderOpenApiSession> {
    ensure!(
        credentials.environment == binding.environment()
            && credentials.account_id == binding.account_id(),
        "loaded broker credentials differ from the exact symbol binding"
    );
    let transport = ProductionCTraderOpenApiTransport::new(binding.server());
    let mut session = transport
        .connect_session(None)
        .context("connect exact broker symbol contract session")?;
    exchange_expected(
        &mut session,
        &build_application_auth_request(
            &credentials.client_id,
            &credentials.client_secret,
            "symbol-contract-application-auth",
        ),
        CTRADER_OA_APPLICATION_AUTH_RESPONSE_PAYLOAD_TYPE,
        "application-auth",
    )?;
    let account_response = exchange_expected(
        &mut session,
        &build_account_auth_request(
            binding.account_id(),
            &credentials.access_token,
            "symbol-contract-account-auth",
        ),
        CTRADER_OA_ACCOUNT_AUTH_RESPONSE_PAYLOAD_TYPE,
        "account-auth",
    )?;
    let account = exact_document(&account_response, "account-auth")?;
    ensure!(
        account
            .pointer("/payload/ctidTraderAccountId")
            .and_then(Value::as_i64)
            == Some(binding.account_id()),
        "authenticated broker account differs from exact symbol binding"
    );
    Ok(session)
}

pub fn capture_exact_production_broker_symbol_contract_v1(
    prepared: &PreparedExactBrokerSymbolContractCaptureV1,
) -> Result<ExactBrokerSymbolContractReceiptV1> {
    let credentials = load_exact_production_historical_credentials(
        prepared.binding.environment(),
        prepared.binding.account_id(),
    )?;
    let mut session = authenticate_exact_session(&prepared.binding, &credentials)?;
    let light_symbols_response = exchange_expected(
        &mut session,
        &build_symbols_list_request(
            prepared.binding.account_id(),
            false,
            LIGHT_SYMBOLS_CLIENT_MESSAGE_ID_V1,
        ),
        CTRADER_OA_SYMBOLS_LIST_RESPONSE_PAYLOAD_TYPE,
        "light-symbols",
    )?;
    validate_light_symbols_response(&prepared.binding, &light_symbols_response)?;
    let full_symbol_response = exchange_expected(
        &mut session,
        &build_symbol_by_id_request(
            prepared.binding.account_id(),
            &[prepared.binding.symbol_id()],
            FULL_SYMBOL_CLIENT_MESSAGE_ID_V1,
        ),
        CTRADER_OA_SYMBOL_BY_ID_RESPONSE_PAYLOAD_TYPE,
        "full-symbol",
    )?;
    publish_validated_broker_symbol_contract_response_v1(
        &prepared.output_root,
        &prepared.binding,
        &light_symbols_response,
        &full_symbol_response,
    )
}

/// Reopen a unique, previously captured pair of exact broker responses for
/// offline *research*. This does not refresh metadata or authorize live orders.
/// The caller owns the source/account-specific directory; every payload and
/// content-addressed filename is verified again. Ambiguity is never resolved
/// by silently choosing a filesystem entry.
pub fn reopen_cached_research_symbol_contract_v1(
    prepared: &PreparedExactBrokerSymbolContractCaptureV1,
) -> Result<Option<ExactBrokerSymbolContractReceiptV1>> {
    let mut light = Vec::new();
    let mut full = Vec::new();
    for entry in fs::read_dir(prepared.output_root())? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let (prefix, destination) = if name.starts_with("bsl1-") {
            ("bsl1-", &mut light)
        } else if name.starts_with("bsc1-") {
            ("bsc1-", &mut full)
        } else {
            continue;
        };
        let hash = name
            .strip_prefix(prefix)
            .and_then(|name| name.strip_suffix(".json"))
            .context("cached broker response has a malformed content-addressed name")?;
        let bytes = if prefix == "bsl1-" {
            crate::canonical_research_costs::read_regular_file_with_limit(
                &entry.path(),
                MAX_CACHED_SYMBOL_CATALOG_BYTES_V1,
            )
        } else {
            crate::canonical_research_costs::read_bounded_regular_file(&entry.path())
        }
        .with_context(|| format!("reopen cached broker metadata {}", entry.path().display()))?;
        ensure!(
            sha256_hex(&bytes) == hash,
            "cached broker response bytes differ from their filename digest"
        );
        destination.push((hash.to_owned(), entry.path(), bytes));
    }
    if light.is_empty() && full.is_empty() {
        return Ok(None);
    }
    ensure!(
        light.len() == 1 && full.len() == 1,
        "offline research needs exactly one captured light/full-symbol pair; found {}/{}; explicitly refresh the source selection",
        light.len(),
        full.len()
    );
    let (light_symbols_sha256, light_symbols_path, light_bytes) = light.remove(0);
    let (full_symbol_sha256, full_symbol_path, full_bytes) = full.remove(0);
    validate_light_symbols_response(prepared.binding(), &light_bytes)?;
    validate_full_symbol_response(prepared.binding(), &full_bytes)?;
    Ok(Some(ExactBrokerSymbolContractReceiptV1 {
        binding: prepared.binding().clone(),
        light_symbols_sha256,
        full_symbol_sha256,
        light_symbols_path,
        full_symbol_path,
    }))
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ExactBrokerSymbolContractReceiptWireV1<'a> {
    schema: &'static str,
    version: u16,
    environment: &'static str,
    server: &'a str,
    account_id: i64,
    symbol_id: i64,
    symbol_name: &'a str,
    light_symbols_sha256: &'a str,
    full_symbol_sha256: &'a str,
    light_symbols_path: &'a Path,
    full_symbol_path: &'a Path,
}

pub fn render_exact_broker_symbol_contract_receipt_v1(
    receipt: &ExactBrokerSymbolContractReceiptV1,
) -> Result<Vec<u8>> {
    let binding = receipt.binding();
    let environment = match binding.environment() {
        BrokerEnvironment::Demo => "demo",
        BrokerEnvironment::Live => "live",
    };
    let mut bytes = serde_json::to_vec(&ExactBrokerSymbolContractReceiptWireV1 {
        schema: "neoethos.exact_broker_symbol_contract_receipt.v1",
        version: 1,
        environment,
        server: binding.server(),
        account_id: binding.account_id(),
        symbol_id: binding.symbol_id(),
        symbol_name: binding.symbol_name(),
        light_symbols_sha256: receipt.light_symbols_sha256(),
        full_symbol_sha256: receipt.full_symbol_sha256(),
        light_symbols_path: receipt.light_symbols_path(),
        full_symbol_path: receipt.full_symbol_path(),
    })?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod broker_error_tests {
    use super::*;

    #[test]
    fn broker_rejection_keeps_the_exact_stage_and_typed_safe_error() {
        let response = serde_json::json!({
            "clientMsgId": "symbol-contract-light-symbols",
            "payloadType": 2142,
            "payload": {
                "errorCode": "CANT_ROUTE_REQUEST",
                "description": "Cannot route request"
            }
        })
        .to_string();

        let error =
            broker_rejection_for_stage("light-symbols", &response).expect("typed broker rejection");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("light-symbols"));
        assert!(rendered.contains("CANT_ROUTE_REQUEST"));
        assert!(!rendered.contains("clientSecret"));
        assert!(!rendered.contains("accessToken"));
    }

    #[test]
    fn research_cache_reopens_exact_bytes_and_refuses_corruption_or_another_account() {
        let root = tempfile::tempdir().unwrap();
        let binding =
            ExactBrokerSymbolContractBindingV1::new(BrokerEnvironment::Demo, 42, 1, "EURUSD")
                .unwrap();
        let prepared = PreparedExactBrokerSymbolContractCaptureV1::new(
            binding.clone(),
            root.path().to_path_buf(),
        )
        .unwrap();
        assert!(
            reopen_cached_research_symbol_contract_v1(&prepared)
                .unwrap()
                .is_none()
        );
        let mut catalog = vec![serde_json::json!({"symbolId": 1, "symbolName": "EURUSD"})];
        catalog.extend((2..=2500).map(|id| {
            serde_json::json!({
                "symbolId": id, "symbolName": format!("SYMBOL_{id}"),
                "baseAssetId": 1, "quoteAssetId": 2, "symbolCategoryId": 10,
            })
        }));
        let light = serde_json::to_vec(&serde_json::json!({"payloadType": 2115, "clientMsgId": LIGHT_SYMBOLS_CLIENT_MESSAGE_ID_V1,
            "payload": {"ctidTraderAccountId": 42, "symbol": catalog}})).unwrap();
        assert!(
            light.len() > 64 * 1024,
            "real broker catalogs exceed a single contract's bound"
        );
        let full = serde_json::to_vec(&serde_json::json!({"payloadType": 2117, "clientMsgId": FULL_SYMBOL_CLIENT_MESSAGE_ID_V1,
            "payload": {"ctidTraderAccountId": 42, "symbol": [{"symbolId": 1}]}})).unwrap();
        let original = publish_validated_broker_symbol_contract_response_v1(
            root.path(),
            &binding,
            &light,
            &full,
        )
        .unwrap();
        assert_eq!(
            reopen_cached_research_symbol_contract_v1(&prepared)
                .unwrap()
                .unwrap(),
            original
        );
        let wrong = PreparedExactBrokerSymbolContractCaptureV1::new(
            ExactBrokerSymbolContractBindingV1::new(BrokerEnvironment::Demo, 43, 1, "EURUSD")
                .unwrap(),
            root.path().to_path_buf(),
        )
        .unwrap();
        assert!(reopen_cached_research_symbol_contract_v1(&wrong).is_err());
        fs::write(original.full_symbol_path(), b"{}").unwrap();
        assert!(reopen_cached_research_symbol_contract_v1(&prepared).is_err());
    }
}
