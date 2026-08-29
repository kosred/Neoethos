//! Thin operator client for the app-owned canonical Native Research lane.
//!
//! The app process owns the move-only job handle. CLI invocations only call
//! its loopback HTTP control surface, so start, status, and exact-token cancel
//! all address that same handle across separate invocations.

use std::io::Write;
use std::net::IpAddr;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const NATIVE_RESEARCH_START_ROUTE: &str = "/engines/native-research/start";
const NATIVE_RESEARCH_CANCEL_ROUTE: &str = "/engines/native-research/cancel";
const ENGINES_STATUS_ROUTE: &str = "/engines/status";
const DEFAULT_API_PORT: u16 = 7423;
const API_TIMEOUT: Duration = Duration::from_secs(8);
const MAX_FAILURE_DETAIL_CHARS: usize = 320;
const MAX_PATH_CHARS: usize = 180;
const MAX_PUBLISHED_SUMMARY_CHARS: usize = 800;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ContractArtifactBodyV1 {
    relative_path: String,
    expected_sha256: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StartBodyV1 {
    contract_artifact: ContractArtifactBodyV1,
    population: Option<usize>,
    population_auto: Option<bool>,
    max_indicators: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EnginesStatusEnvelopeV1 {
    canonical_native_research: CanonicalNativeResearchStatusV1,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CanonicalNativeResearchStatusV1 {
    state: String,
    stage: String,
    percent: f64,
    lease_token: Option<String>,
    cancellation_requested: bool,
    failure_stage: Option<String>,
    failure_code: Option<String>,
    failure_detail: Option<String>,
    published: Option<PublishedCanonicalNativeResearchV1>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PublishedCanonicalNativeResearchV1 {
    relative_path: String,
    byte_count: u64,
    file_sha256: String,
    evidence_identity_sha256: String,
    configured_population: usize,
    resolved_population: usize,
    population_cap: usize,
    hard_growth_cap: usize,
    term_cap: usize,
    selected_device_ordinal: u32,
    engine: String,
    parent_h2d_bytes: u64,
    adaptive_h2d_bytes: u64,
    metric_rows: u64,
    metric_bytes: u64,
    consumer_completion_confirmed: bool,
    replay_identity_sealed: bool,
}

#[derive(Debug)]
enum CommandV1 {
    Start(StartBodyV1),
    Status,
    Cancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HttpMethodV1 {
    Get,
    Post,
}

trait NativeResearchTransportV1 {
    fn request(
        &self,
        method: HttpMethodV1,
        path: &'static str,
        body: Option<&Value>,
    ) -> Result<Value>;
}

struct ReqwestNativeResearchTransportV1 {
    base: String,
    client: reqwest::blocking::Client,
}

impl ReqwestNativeResearchTransportV1 {
    fn new(base: String) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(API_TIMEOUT)
            .build()
            .context("build Native Research loopback HTTP client")?;
        Ok(Self { base, client })
    }
}

impl NativeResearchTransportV1 for ReqwestNativeResearchTransportV1 {
    fn request(
        &self,
        method: HttpMethodV1,
        path: &'static str,
        body: Option<&Value>,
    ) -> Result<Value> {
        let url = format!("{}{}", self.base, path);
        let request = match method {
            HttpMethodV1::Get => self.client.get(&url),
            HttpMethodV1::Post => self.client.post(&url).json(body.unwrap_or(&Value::Null)),
        };
        let response = request
            .send()
            .with_context(|| format!("Native Research API request failed: {path}"))?;
        let status = response.status();
        let text = response
            .text()
            .with_context(|| format!("read Native Research API response: {path}"))?;
        if !status.is_success() {
            let detail = response_error_detail_v1(&text);
            bail!("Native Research API rejected {path} ({status}): {detail}");
        }
        serde_json::from_str(&text)
            .with_context(|| format!("decode Native Research API response: {path}"))
    }
}

pub fn run(args: &[String]) -> Result<()> {
    let (base, command) = parse_command_v1(args)?;
    let transport = ReqwestNativeResearchTransportV1::new(base)?;
    let stdout = std::io::stdout();
    let mut output = stdout.lock();
    run_command_v1(&transport, command, &mut output)
}

fn run_command_v1(
    transport: &dyn NativeResearchTransportV1,
    command: CommandV1,
    output: &mut dyn Write,
) -> Result<()> {
    match command {
        CommandV1::Start(body) => {
            let request = serde_json::to_value(body).context("encode native start request")?;
            let accepted = transport.request(
                HttpMethodV1::Post,
                NATIVE_RESEARCH_START_ROUTE,
                Some(&request),
            )?;
            let lease_token = opaque_lease_token_from_value_v1(&accepted)?;
            writeln!(output, "accepted native research lease={lease_token}")?;
            print_current_status_v1(transport, output)
        }
        CommandV1::Status => print_current_status_v1(transport, output),
        CommandV1::Cancel => {
            let current = fetch_status_v1(transport)?;
            let lease_token = current_lease_token_v1(&current)?;
            let body = json!({ "leaseToken": lease_token });
            let response = transport.request(
                HttpMethodV1::Post,
                NATIVE_RESEARCH_CANCEL_ROUTE,
                Some(&body),
            )?;
            ensure!(
                response
                    .get("cancellationRequested")
                    .and_then(Value::as_bool)
                    == Some(true),
                "Native Research cancel response did not confirm cancellation"
            );
            writeln!(output, "cancellation requested for lease={lease_token}")?;
            print_current_status_v1(transport, output)
        }
    }
}

fn print_current_status_v1(
    transport: &dyn NativeResearchTransportV1,
    output: &mut dyn Write,
) -> Result<()> {
    let status = fetch_status_v1(transport)?;
    writeln!(output, "{}", render_status_v1(&status))?;
    Ok(())
}

fn fetch_status_v1(
    transport: &dyn NativeResearchTransportV1,
) -> Result<CanonicalNativeResearchStatusV1> {
    let value = transport.request(HttpMethodV1::Get, ENGINES_STATUS_ROUTE, None)?;
    let envelope: EnginesStatusEnvelopeV1 =
        serde_json::from_value(value).context("decode canonicalNativeResearch status")?;
    Ok(envelope.canonical_native_research)
}

fn current_lease_token_v1(status: &CanonicalNativeResearchStatusV1) -> Result<&str> {
    let token = status
        .lease_token
        .as_deref()
        .context("no queued/running canonical Native Research lease to cancel")?;
    validate_opaque_lease_token_v1(token)?;
    Ok(token)
}

fn opaque_lease_token_from_value_v1(value: &Value) -> Result<&str> {
    let token = value
        .get("leaseToken")
        .and_then(Value::as_str)
        .context("native start response omitted opaque leaseToken")?;
    validate_opaque_lease_token_v1(token)?;
    Ok(token)
}

fn validate_opaque_lease_token_v1(token: &str) -> Result<()> {
    ensure!(
        !token.is_empty() && token.bytes().all(|byte| byte.is_ascii_digit()),
        "native lease token is not an opaque decimal string"
    );
    ensure!(token != "0", "native lease token must be nonzero");
    Ok(())
}

fn parse_command_v1(args: &[String]) -> Result<(String, CommandV1)> {
    let (base, args) = extract_api_base_v1(args)?;
    let Some(subcommand) = args.first().map(String::as_str) else {
        bail!("native-research requires start, status, or cancel");
    };
    let command = match subcommand {
        "status" => {
            ensure!(
                args.len() == 1,
                "native-research status takes no other arguments"
            );
            CommandV1::Status
        }
        "cancel" => {
            ensure!(
                args.len() == 1,
                "native-research cancel takes no other arguments"
            );
            CommandV1::Cancel
        }
        "start" => CommandV1::Start(parse_start_body_v1(&args[1..])?),
        other => {
            bail!("unknown native-research action `{other}`; expected start, status, or cancel")
        }
    };
    Ok((base, command))
}

fn extract_api_base_v1(args: &[String]) -> Result<(String, Vec<String>)> {
    let mut remaining = Vec::with_capacity(args.len());
    let mut explicit = None;
    let mut index = 0;
    while index < args.len() {
        if args[index] == "--api-base" {
            ensure!(explicit.is_none(), "--api-base may be specified only once");
            let raw = args.get(index + 1).context("--api-base requires a value")?;
            explicit = Some(normalize_loopback_base_v1(raw)?);
            index += 2;
        } else {
            remaining.push(args[index].clone());
            index += 1;
        }
    }
    let base = match explicit {
        Some(base) => base,
        None => default_api_base_v1()?,
    };
    Ok((base, remaining))
}

fn default_api_base_v1() -> Result<String> {
    let port_path = std::env::temp_dir().join("neoethos_api_port");
    let port = match std::fs::read_to_string(&port_path) {
        Ok(raw) => raw
            .trim()
            .parse::<u16>()
            .with_context(|| format!("invalid API port in {}", port_path.display()))?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => DEFAULT_API_PORT,
        Err(error) => {
            return Err(error)
                .with_context(|| format!("read app API port from {}", port_path.display()));
        }
    };
    ensure!(port != 0, "app API port must be nonzero");
    Ok(format!("http://127.0.0.1:{port}"))
}

fn normalize_loopback_base_v1(raw: &str) -> Result<String> {
    let parsed = reqwest::Url::parse(raw).context("parse --api-base URL")?;
    ensure!(
        parsed.scheme() == "http",
        "--api-base must use http on loopback"
    );
    ensure!(
        parsed.username().is_empty() && parsed.password().is_none(),
        "--api-base must not contain credentials"
    );
    let host = parsed.host_str().context("--api-base is missing a host")?;
    let loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .map(|address| address.is_loopback())
            .unwrap_or(false);
    ensure!(
        loopback,
        "--api-base must resolve explicitly to a loopback host"
    );
    ensure!(
        matches!(parsed.path(), "" | "/")
            && parsed.query().is_none()
            && parsed.fragment().is_none(),
        "--api-base must not contain a path, query, or fragment"
    );
    ensure!(
        parsed.port().is_some(),
        "--api-base must include an explicit port"
    );
    Ok(raw.trim_end_matches('/').to_owned())
}

fn parse_start_body_v1(args: &[String]) -> Result<StartBodyV1> {
    let mut relative_path = None;
    let mut expected_sha256 = None;
    let mut population = None;
    let mut population_auto = None;
    let mut max_indicators = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args
            .get(index + 1)
            .with_context(|| format!("{flag} requires a value"))?;
        match flag {
            "--contract-relative-path" => set_once_v1(&mut relative_path, value, flag)?,
            "--expected-sha256" => set_once_v1(&mut expected_sha256, value, flag)?,
            "--population" => {
                ensure!(
                    population.is_none(),
                    "--population may be specified only once"
                );
                population = Some(parse_positive_usize_v1(value, flag)?);
            }
            "--population-auto" => {
                ensure!(
                    population_auto.is_none(),
                    "--population-auto may be specified only once"
                );
                population_auto = Some(match value.as_str() {
                    "true" => true,
                    "false" => false,
                    _ => bail!("--population-auto must be true or false"),
                });
            }
            "--max-indicators" => {
                ensure!(
                    max_indicators.is_none(),
                    "--max-indicators may be specified only once"
                );
                max_indicators = Some(parse_positive_usize_v1(value, flag)?);
            }
            _ => bail!("unknown native-research start flag `{flag}`"),
        }
        index += 2;
    }
    let relative_path = relative_path.context("--contract-relative-path is required")?;
    let expected_sha256 = expected_sha256.context("--expected-sha256 is required")?;
    ensure!(
        expected_sha256.len() == 64
            && expected_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "--expected-sha256 must be exactly 64 lowercase hexadecimal characters"
    );
    Ok(StartBodyV1 {
        contract_artifact: ContractArtifactBodyV1 {
            relative_path,
            expected_sha256,
        },
        population,
        population_auto,
        max_indicators,
    })
}

fn set_once_v1(slot: &mut Option<String>, value: &str, flag: &str) -> Result<()> {
    ensure!(slot.is_none(), "{flag} may be specified only once");
    ensure!(!value.trim().is_empty(), "{flag} must not be empty");
    *slot = Some(value.to_owned());
    Ok(())
}

fn parse_positive_usize_v1(value: &str, flag: &str) -> Result<usize> {
    let parsed = value
        .parse::<usize>()
        .with_context(|| format!("{flag} must be a positive integer"))?;
    ensure!(parsed > 0, "{flag} must be greater than zero");
    Ok(parsed)
}

fn render_status_v1(status: &CanonicalNativeResearchStatusV1) -> String {
    let mut lines = vec![format!(
        "state={} stage={} percent={:.2}% cancellation_requested={} lease={}",
        bounded_text(&status.state, 48),
        bounded_text(&status.stage, 96),
        status.percent.clamp(0.0, 100.0),
        status.cancellation_requested,
        status.lease_token.as_deref().unwrap_or("none")
    )];
    if status.failure_stage.is_some()
        || status.failure_code.is_some()
        || status.failure_detail.is_some()
    {
        lines.push(format!(
            "failure stage={} code={} detail={}",
            bounded_text(status.failure_stage.as_deref().unwrap_or("unknown"), 96),
            bounded_text(status.failure_code.as_deref().unwrap_or("unknown"), 96),
            bounded_text(
                status.failure_detail.as_deref().unwrap_or(""),
                MAX_FAILURE_DETAIL_CHARS,
            )
        ));
    }
    if let Some(published) = &status.published {
        lines.push(bounded_text(
            &format!(
                "published path={} bytes={} sha256={} evidence={} configured_P={} resolved_P={} population_cap={} hard_growth_cap={} T={} device={} engine={} H2D={}+{} metrics={}/{} consumer={} replay={}",
                bounded_text(&published.relative_path, MAX_PATH_CHARS),
                published.byte_count,
                bounded_text(&published.file_sha256, 64),
                bounded_text(&published.evidence_identity_sha256, 64),
                published.configured_population,
                published.resolved_population,
                published.population_cap,
                published.hard_growth_cap,
                published.term_cap,
                published.selected_device_ordinal,
                bounded_text(&published.engine, 48),
                published.parent_h2d_bytes,
                published.adaptive_h2d_bytes,
                published.metric_rows,
                published.metric_bytes,
                published.consumer_completion_confirmed,
                published.replay_identity_sealed,
            ),
            MAX_PUBLISHED_SUMMARY_CHARS,
        ));
    }
    lines.join("\n")
}

fn response_error_detail_v1(text: &str) -> String {
    let parsed = serde_json::from_str::<Value>(text).ok();
    let detail = parsed
        .as_ref()
        .and_then(|value| value.get("detail"))
        .and_then(Value::as_str)
        .or_else(|| {
            parsed
                .as_ref()
                .and_then(|value| value.get("errorCode"))
                .and_then(Value::as_str)
        })
        .unwrap_or(text);
    bounded_text(detail, MAX_FAILURE_DETAIL_CHARS)
}

fn bounded_text(value: &str, max_chars: usize) -> String {
    let normalized: String = value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect();
    let normalized = normalized.trim();
    if normalized.chars().count() <= max_chars {
        return normalized.to_owned();
    }
    if max_chars == 0 {
        return String::new();
    }
    let mut bounded: String = normalized
        .chars()
        .take(max_chars.saturating_sub(1))
        .collect();
    bounded.push('…');
    bounded
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use super::*;

    #[derive(Debug)]
    struct FakeTransportV1 {
        responses: RefCell<VecDeque<Value>>,
        calls: RefCell<Vec<(HttpMethodV1, &'static str, Option<Value>)>>,
    }

    impl FakeTransportV1 {
        fn new(responses: Vec<Value>) -> Self {
            Self {
                responses: RefCell::new(responses.into()),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl NativeResearchTransportV1 for FakeTransportV1 {
        fn request(
            &self,
            method: HttpMethodV1,
            path: &'static str,
            body: Option<&Value>,
        ) -> Result<Value> {
            self.calls.borrow_mut().push((method, path, body.cloned()));
            self.responses
                .borrow_mut()
                .pop_front()
                .context("missing fake response")
        }
    }

    fn running_status(token: &str) -> Value {
        json!({
            "canonicalNativeResearch": {
                "state": "Running",
                "stage": "generation_zero_evaluation",
                "percent": 25.0,
                "leaseToken": token,
                "cancellationRequested": false,
                "failureStage": null,
                "failureCode": null,
                "failureDetail": null,
                "published": null
            }
        })
    }

    #[test]
    fn start_uses_only_the_native_route_and_exact_nested_contract_body() {
        let token = "18446744073709551615";
        let transport = FakeTransportV1::new(vec![
            json!({"started": true, "leaseToken": token}),
            running_status(token),
        ]);
        let command = CommandV1::Start(StartBodyV1 {
            contract_artifact: ContractArtifactBodyV1 {
                relative_path: "contracts/EURUSD-M1.json".to_owned(),
                expected_sha256: "ab".repeat(32),
            },
            population: Some(4096),
            population_auto: Some(false),
            max_indicators: Some(17),
        });
        let mut output = Vec::new();

        run_command_v1(&transport, command, &mut output).expect("native start");

        let calls = transport.calls.borrow();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, HttpMethodV1::Post);
        assert_eq!(calls[0].1, NATIVE_RESEARCH_START_ROUTE);
        assert_eq!(calls[1].0, HttpMethodV1::Get);
        assert_eq!(calls[1].1, ENGINES_STATUS_ROUTE);
        assert_eq!(
            calls[0].2,
            Some(json!({
                "contractArtifact": {
                    "relativePath": "contracts/EURUSD-M1.json",
                    "expectedSha256": "ab".repeat(32)
                },
                "population": 4096,
                "populationAuto": false,
                "maxIndicators": 17
            }))
        );
    }

    #[test]
    fn cancel_reads_and_reuses_the_exact_opaque_live_token() {
        let token = "18446744073709551615";
        let mut cancelling = running_status(token);
        cancelling["canonicalNativeResearch"]["cancellationRequested"] = json!(true);
        let transport = FakeTransportV1::new(vec![
            running_status(token),
            json!({"cancellationRequested": true, "leaseToken": token}),
            cancelling,
        ]);
        let mut output = Vec::new();

        run_command_v1(&transport, CommandV1::Cancel, &mut output).expect("native cancel");

        let calls = transport.calls.borrow();
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].1, ENGINES_STATUS_ROUTE);
        assert_eq!(calls[1].1, NATIVE_RESEARCH_CANCEL_ROUTE);
        assert_eq!(calls[1].2, Some(json!({"leaseToken": token})));
        assert_eq!(calls[2].1, ENGINES_STATUS_ROUTE);
    }

    #[test]
    fn status_bounds_failure_detail_and_renders_published_evidence() {
        let status: CanonicalNativeResearchStatusV1 = serde_json::from_value(json!({
            "state": "Failed",
            "stage": "consumer_completion",
            "percent": 99.0,
            "leaseToken": null,
            "cancellationRequested": false,
            "failureStage": "ConsumerCompletion",
            "failureCode": "completion_failed",
            "failureDetail": "x".repeat(2_000),
            "published": {
                "relativePath": "generation-zero/result.json",
                "byteCount": 1234,
                "fileSha256": "cd".repeat(32),
                "evidenceIdentitySha256": "ef".repeat(32),
                "configuredPopulation": 2048,
                "resolvedPopulation": 1024,
                "populationCap": 1024,
                "hardGrowthCap": 1024,
                "termCap": 96,
                "selectedDeviceOrdinal": 0,
                "engine": "cuda",
                "parentH2dBytes": 40,
                "adaptiveH2dBytes": 2,
                "metricRows": 18,
                "metricBytes": 144,
                "consumerCompletionConfirmed": true,
                "replayIdentitySealed": true
            }
        }))
        .expect("fixture");

        let rendered = render_status_v1(&status);
        assert!(rendered.contains("failure stage=ConsumerCompletion"));
        assert!(rendered.contains("published path=generation-zero/result.json"));
        assert!(rendered.contains("resolved_P=1024"));
        assert!(rendered.contains("hard_growth_cap=1024"));
        assert!(rendered.contains("consumer=true"));
        assert!(rendered.lines().all(|line| line.chars().count() <= 800));
    }
}
