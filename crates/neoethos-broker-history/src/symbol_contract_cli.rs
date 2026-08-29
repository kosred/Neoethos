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
        ensure!(
            self.output_root.is_absolute(),
            "symbol-contract output-root must be an explicit absolute path"
        );
        fs::create_dir_all(&self.output_root).with_context(|| {
            format!(
                "create exact broker symbol output root {}",
                self.output_root.display()
            )
        })?;
        let metadata = fs::symlink_metadata(&self.output_root).with_context(|| {
            format!(
                "inspect exact broker symbol output root {}",
                self.output_root.display()
            )
        })?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "symbol-contract output-root must be one real directory"
        );
        Ok(PreparedExactBrokerSymbolContractCaptureV1 {
            binding: ExactBrokerSymbolContractBindingV1::new(
                self.environment.broker(),
                self.account_id,
                self.symbol_id,
                self.symbol_name,
            )?,
            output_root: self.output_root,
        })
    }
}

pub struct PreparedExactBrokerSymbolContractCaptureV1 {
    binding: ExactBrokerSymbolContractBindingV1,
    output_root: PathBuf,
}

impl PreparedExactBrokerSymbolContractCaptureV1 {
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
}
