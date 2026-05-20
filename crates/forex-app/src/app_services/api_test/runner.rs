//! Test-runner orchestration. Builds a `TradingSession` from the
//! current persisted broker settings, walks each flow in order, and
//! writes the JSON report.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};

use super::flows;
use super::report::{ApiTestReport, FailureKind, FlowResult, FlowStatus, HostSummary, ReportTotals};
use crate::app_services::trading::TradingSession;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiTestEnvironment {
    Demo,
    Live,
}

impl ApiTestEnvironment {
    pub fn as_str(self) -> &'static str {
        match self {
            ApiTestEnvironment::Demo => "demo",
            ApiTestEnvironment::Live => "live",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApiTestConfig {
    pub environment: ApiTestEnvironment,
    pub output_path: PathBuf,
    pub slow: bool,
    /// When `Some`, only flow names matching this glob are executed.
    /// Cleanup still runs at the end. Examples: `"orders.*"`,
    /// `"streaming.spot.*"`.
    pub only_filter: Option<String>,
}

/// Convenience entry point used by `forex-app --api-test`. The async
/// runtime is already established by `#[tokio::main]` in `main.rs`.
pub async fn run_api_test_suite(config: ApiTestConfig) -> Result<()> {
    let mut session = TradingSession::new_with_persisted_credentials();
    let started_unix_ms = unix_now_ms();
    let started_instant = Instant::now();

    let mut flows = Vec::new();
    let mut state = flows::SuiteState::default();

    let blueprints = flows::all_flow_blueprints();
    for blueprint in blueprints {
        if !flow_matches_filter(blueprint.name, config.only_filter.as_deref()) {
            continue;
        }
        // Skip subsequent flows that depend on a previously-failed
        // dependency (e.g. `orders.modify_sltp` needs the buy from
        // `orders.market_buy_001`). Per-flow `requires_state_keys`
        // controls this.
        if let Some(missing) = blueprint.first_missing_dependency(&state) {
            let reason = format!(
                "skipped: depends on prior flow output `{}` which is not in suite state",
                missing
            );
            flows.push(FlowResult::skip(blueprint.name, reason));
            continue;
        }
        let start = Instant::now();
        let outcome = (blueprint.run)(&mut session, &mut state).await;
        let result = match outcome {
            Ok(mut r) => {
                if matches!(r.status, FlowStatus::Pass) {
                    r.duration_ms = start.elapsed().as_millis();
                }
                r
            }
            Err(err) => FlowResult::fail(
                blueprint.name,
                start.elapsed(),
                err.to_string(),
                FailureKind::Other,
            ),
        };
        flows.push(result);
        if config.slow {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
    }

    // Cleanup pass — always runs, even when --api-test-only filtered
    // out everything, because a previous interrupted run may have left
    // a position open.
    let cleanup_start = Instant::now();
    let cleanup_outcome = flows::cleanup_flatten_all(&mut session, &state).await;
    let cleanup_result = match cleanup_outcome {
        Ok(r) => r,
        Err(err) => FlowResult::fail(
            "cleanup.flatten_all",
            cleanup_start.elapsed(),
            err.to_string(),
            FailureKind::CleanupFailure,
        ),
    };
    flows.push(cleanup_result);

    let totals = ReportTotals::recompute(&flows);
    let finished_unix_ms = unix_now_ms();
    let _ = started_instant;

    let report = ApiTestReport {
        schema_version: ApiTestReport::SCHEMA_VERSION,
        started_at_unix_ms: started_unix_ms,
        finished_at_unix_ms: finished_unix_ms,
        environment: config.environment.as_str().to_string(),
        forex_app_version: env!("CARGO_PKG_VERSION").to_string(),
        host_summary: detect_host_summary(),
        flows,
        totals,
    };

    write_report(&report, &config.output_path)
        .with_context(|| format!("write report to {}", config.output_path.display()))?;

    print_terminal_summary(&report);
    Ok(())
}

fn write_report(report: &ApiTestReport, path: &std::path::Path) -> Result<()> {
    let body =
        serde_json::to_string_pretty(report).map_err(|e| anyhow!("serialise report: {e}"))?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(path, body).map_err(|e| anyhow!("write report file: {e}"))?;
    Ok(())
}

fn print_terminal_summary(report: &ApiTestReport) {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "\n=== forex-app api-test report — {} ===",
        report.environment
    );
    for flow in &report.flows {
        let badge = match flow.status {
            FlowStatus::Pass => "PASS",
            FlowStatus::Fail => "FAIL",
            FlowStatus::Skip => "SKIP",
        };
        let dur = format!("{:>5} ms", flow.duration_ms);
        let _ = writeln!(out, "  [{}] {}  {}", badge, dur, flow.name);
        if let Some(err) = &flow.error
            && !matches!(flow.status, FlowStatus::Pass)
        {
            let _ = writeln!(out, "         └─ {}", err);
        }
    }
    let _ = writeln!(
        out,
        "--- totals: {} passed / {} failed / {} skipped (+ {} cleanup-fails) in {} ms",
        report.totals.flows_passed,
        report.totals.flows_failed,
        report.totals.flows_skipped,
        report.totals.cleanup_failures,
        report.totals.total_duration_ms,
    );
    eprintln!("{}", out);
}

fn flow_matches_filter(flow: &str, filter: Option<&str>) -> bool {
    let Some(pat) = filter else {
        return true;
    };
    // Minimal glob: support `*` only. Sufficient for the audit filters
    // we expect (`orders.*`, `streaming.spot.*`, `errors.*`).
    if let Some(prefix) = pat.strip_suffix(".*") {
        return flow.starts_with(&format!("{prefix}."));
    }
    if let Some(rest) = pat.strip_prefix("*.") {
        return flow.ends_with(&format!(".{rest}"));
    }
    flow == pat
}

fn unix_now_ms() -> i64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

fn detect_host_summary() -> HostSummary {
    HostSummary {
        os: std::env::consts::OS.to_string(),
        cpu_brand: detect_cpu_brand(),
        logical_cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        total_memory_bytes: detect_total_memory_bytes(),
    }
}

fn detect_cpu_brand() -> String {
    // Avoid pulling a sysinfo dep just for one string — the OS-specific
    // helpers are short and the report does not need to be exhaustive.
    #[cfg(target_os = "windows")]
    {
        std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".to_string())
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "unknown".to_string())
    }
    #[cfg(target_os = "macos")]
    {
        "macos-cpu".to_string()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        "unknown".to_string()
    }
}

fn detect_total_memory_bytes() -> u64 {
    // Best-effort, returns 0 if we can't tell. Linux only for now —
    // Windows would need GlobalMemoryStatusEx via winapi, which is more
    // ceremony than this report needs.
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|kb| kb.parse::<u64>().ok())
                    .map(|kb| kb * 1024)
            })
            .unwrap_or(0)
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}
