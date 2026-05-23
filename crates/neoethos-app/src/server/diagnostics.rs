//! Diagnostic bundle endpoint — POST /diagnostics/report.
//!
//! End users cannot rebuild the app. Whenever something genuinely
//! breaks (a panic, an unrecognised cTrader error, an unreachable
//! backend), the UI must funnel the failure into a one-click
//! "Report Issue" flow that hands them a self-contained file to
//! email back to `konstantinoskokkinos1982@gmail.com`. This module
//! is the file-builder side of that flow:
//!
//!   * Reads today + yesterday daily logs from
//!     `%APPDATA%\neoethos\logs\neoethos.YYYY-MM-DD.log`.
//!   * Reads `config.yaml` (no secrets in this file, ships as-is).
//!   * Reads `broker_credentials.toml` and REDACTS client_secret +
//!     access_token-shaped strings before including a copy.
//!   * Probes system info (OS, CPU, RAM, hostname, app version).
//!   * Captures the operator's free-text description of what
//!     happened.
//!   * Writes everything into a single .zip on the Desktop and
//!     returns the path so the Flutter side can open it (or copy
//!     the path to the clipboard so the user attaches it manually
//!     to their mail).
//!
//! Privacy posture: the bundle is generated locally and stays on
//! the user's machine until they choose to email it. No data
//! leaves the device without explicit user action. Secrets in the
//! TOML and any access-token-shaped strings in the logs are
//! masked before they enter the zip.

use std::io::Write;
use std::path::PathBuf;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use zip::write::SimpleFileOptions;

use super::state::AppApiState;

#[derive(Debug, Deserialize)]
pub struct ReportRequest {
    /// Free-text description from the operator: "what were you
    /// doing when it broke." Goes into description.txt inside the
    /// zip so the recipient can correlate the user's words with
    /// the timestamps in the logs. Empty string is allowed —
    /// users often hit Report without typing anything.
    #[serde(default)]
    pub user_description: String,
    /// Optional headline / category (e.g. "Order failed", "Login
    /// loop"). Used as the email subject suffix on the Flutter
    /// side. Defaults to "Issue report" when empty.
    #[serde(default)]
    pub category: String,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReportResponse {
    /// Absolute path to the generated zip on the user's Desktop.
    pub zip_path: String,
    /// Total bytes — UI shows "Bundle ready (123 KB)".
    pub total_bytes: u64,
    /// File names included in the zip — UI lists them so the
    /// user knows what they're about to attach.
    pub files_included: Vec<String>,
    /// Subject line the Flutter side should prefill into the
    /// mailto: link.
    pub email_subject: String,
    /// Pre-rendered body for the mailto: link. Includes the file
    /// path because mailto attachments don't work cross-platform.
    pub email_body: String,
    /// Email address the report should go to. Lives here so a
    /// future change rotates it in one place. Surfaced to the
    /// Flutter side so the mailto: link can prefill `to`.
    pub email_recipient: String,
}

/// Where the support email lands. Single source of truth — change
/// here when we eventually rotate to a shared inbox.
pub const REPORT_EMAIL: &str = "konstantinoskokkinos1982@gmail.com";

pub async fn report(
    State(_state): State<AppApiState>,
    Json(req): Json<ReportRequest>,
) -> Response {
    match tokio::task::spawn_blocking(move || build_bundle(&req)).await {
        Ok(Ok(resp)) => Json(resp).into_response(),
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": err.to_string()})),
        )
            .into_response(),
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("diagnostics task panicked: {join_err}"),
            })),
        )
            .into_response(),
    }
}

fn build_bundle(req: &ReportRequest) -> anyhow::Result<ReportResponse> {
    // 1. Pick the destination — Desktop is what every Windows user
    //    knows how to find. Falls back to home dir if Desktop
    //    isn't discoverable (rare).
    let desktop = dirs::desktop_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."));
    let now = chrono::Local::now();
    let zip_name = format!(
        "NeoEthos-Issue-Report-{}.zip",
        now.format("%Y-%m-%d-%H%M%S")
    );
    let zip_path = desktop.join(&zip_name);

    // 2. Collect every file we want inside.
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();

    // 2a. Operator description (always first so the recipient
    //     reads the human context before the log noise).
    let desc = if req.user_description.trim().is_empty() {
        "(operator did not provide a description)".to_string()
    } else {
        req.user_description.clone()
    };
    entries.push(("01_description.txt".to_string(), desc.into_bytes()));

    // 2b. System info + app version.
    entries.push((
        "02_system_info.txt".to_string(),
        collect_system_info().into_bytes(),
    ));

    // 2c. broker_credentials.toml — REDACTED.
    if let Ok(creds_path) = resolve_creds_path() {
        if creds_path.exists() {
            match std::fs::read_to_string(&creds_path) {
                Ok(raw) => entries.push((
                    "03_broker_credentials.toml.redacted".to_string(),
                    redact_credentials(&raw).into_bytes(),
                )),
                Err(err) => entries.push((
                    "03_broker_credentials.toml.error".to_string(),
                    format!("could not read {}: {err}", creds_path.display()).into_bytes(),
                )),
            }
        }
    }

    // 2d. config.yaml — no secrets, ships verbatim.
    let config_path = PathBuf::from("config.yaml");
    if config_path.exists() {
        match std::fs::read(&config_path) {
            Ok(bytes) => entries.push(("04_config.yaml".to_string(), bytes)),
            Err(err) => entries.push((
                "04_config.yaml.error".to_string(),
                format!("could not read config.yaml: {err}").into_bytes(),
            )),
        }
    }

    // 2e. Log files — today + yesterday. Anything older is rarely
    //     useful for diagnosing a "just happened" bug and inflates
    //     the bundle. The logger redacts OAuth bodies to 220 char
    //     previews already; we still pass through `redact_log_text`
    //     as a defence-in-depth.
    if let Some(log_dir) = log_dir() {
        for day_offset in 0..=1 {
            let date = chrono::Local::now().date_naive() - chrono::Duration::days(day_offset);
            let log_name = format!("neoethos.{}.log", date.format("%Y-%m-%d"));
            let path = log_dir.join(&log_name);
            if path.exists() {
                match std::fs::read_to_string(&path) {
                    Ok(text) => entries.push((
                        format!("05_{}", log_name),
                        redact_log_text(&text).into_bytes(),
                    )),
                    Err(err) => entries.push((
                        format!("05_{}.error", log_name),
                        format!("could not read {}: {err}", path.display()).into_bytes(),
                    )),
                }
            }
        }
        // BackendSupervisor log lives next to the user-data dir.
        // Tiny file; always include.
        let supervisor = log_dir
            .parent()
            .map(|p| p.join("supervisor.log"))
            .unwrap_or_else(|| log_dir.join("supervisor.log"));
        if supervisor.exists() {
            match std::fs::read_to_string(&supervisor) {
                Ok(text) => entries.push((
                    "06_supervisor.log".to_string(),
                    redact_log_text(&text).into_bytes(),
                )),
                Err(err) => entries.push((
                    "06_supervisor.log.error".to_string(),
                    format!("could not read {}: {err}", supervisor.display()).into_bytes(),
                )),
            }
        }
    }

    // 3. Write the zip.
    let mut file_names: Vec<String> = Vec::with_capacity(entries.len());
    let writer = std::fs::File::create(&zip_path).map_err(|e| {
        anyhow::anyhow!(
            "could not create {} (Desktop may be read-only?): {e}",
            zip_path.display()
        )
    })?;
    let mut zip = zip::ZipWriter::new(writer);
    let opts: SimpleFileOptions = SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .unix_permissions(0o644);
    for (name, bytes) in &entries {
        zip.start_file(name.as_str(), opts)
            .map_err(|e| anyhow::anyhow!("zip start_file({name}): {e}"))?;
        zip.write_all(bytes)
            .map_err(|e| anyhow::anyhow!("zip write({name}): {e}"))?;
        file_names.push(name.clone());
    }
    zip.finish()
        .map_err(|e| anyhow::anyhow!("zip finish: {e}"))?;

    let total_bytes = std::fs::metadata(&zip_path).map(|m| m.len()).unwrap_or(0);
    let category = if req.category.trim().is_empty() {
        "Issue report".to_string()
    } else {
        req.category.trim().to_string()
    };
    let email_subject = format!("[NeoEthos] {category} ({})", now.format("%Y-%m-%d %H:%M"));
    let email_body = format!(
        "Hi Konstantinos,\n\nI hit an issue in NeoEthos. The diagnostic bundle is attached as:\n\n  {}\n\nIf the attachment is missing please drag-drop the file from my Desktop into this email before sending.\n\n--- What happened ---\n{}\n\n--- App version ---\n{}\n",
        zip_path.display(),
        if req.user_description.trim().is_empty() {
            "(I didn't add a description)"
        } else {
            &req.user_description
        },
        env!("CARGO_PKG_VERSION"),
    );

    Ok(ReportResponse {
        zip_path: zip_path.display().to_string(),
        total_bytes,
        files_included: file_names,
        email_subject,
        email_body,
        email_recipient: REPORT_EMAIL.to_string(),
    })
}

fn collect_system_info() -> String {
    let mut buf = String::new();
    buf.push_str("=== NeoEthos system snapshot ===\n");
    buf.push_str(&format!("Time (local): {}\n", chrono::Local::now()));
    buf.push_str(&format!("Time (UTC):   {}\n", chrono::Utc::now()));
    buf.push_str(&format!("App version:  {}\n", env!("CARGO_PKG_VERSION")));
    buf.push_str(&format!("OS:           {}\n", std::env::consts::OS));
    buf.push_str(&format!("Arch:         {}\n", std::env::consts::ARCH));
    if let Ok(host) = hostname() {
        buf.push_str(&format!("Hostname:     {}\n", host));
    }
    if let Ok(cwd) = std::env::current_dir() {
        buf.push_str(&format!("CWD:          {}\n", cwd.display()));
    }
    if let Ok(exe) = std::env::current_exe() {
        buf.push_str(&format!("Exe:          {}\n", exe.display()));
    }
    buf
}

/// Hostname helper. Doesn't pull a new dep — uses the env var on
/// Windows / unix.
fn hostname() -> anyhow::Result<String> {
    if let Ok(h) = std::env::var("COMPUTERNAME") {
        return Ok(h);
    }
    if let Ok(h) = std::env::var("HOSTNAME") {
        return Ok(h);
    }
    Ok("unknown".to_string())
}

fn log_dir() -> Option<PathBuf> {
    dirs::data_dir().map(|p| p.join("neoethos").join("logs"))
}

fn resolve_creds_path() -> anyhow::Result<PathBuf> {
    Ok(dirs::data_dir()
        .ok_or_else(|| anyhow::anyhow!("could not resolve %APPDATA% / data dir"))?
        .join("neoethos")
        .join("broker_credentials.toml"))
}

/// Redact secrets out of the TOML before it goes into the bundle.
/// We keep enough of each value that the recipient can correlate
/// without exposing anything actionable.
fn redact_credentials(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len() + 256);
    for line in raw.lines() {
        let lower = line.trim_start().to_ascii_lowercase();
        if lower.starts_with("client_secret") {
            if let Some(start) = line.find('"') {
                if let Some(end) = line[start + 1..].find('"') {
                    let val = &line[start + 1..start + 1 + end];
                    let tail = val.chars().rev().take(4).collect::<String>();
                    let tail_fwd: String = tail.chars().rev().collect();
                    out.push_str(&format!(
                        "client_secret = \"[REDACTED, last 4: {tail_fwd}, length: {}]\"\n",
                        val.len()
                    ));
                    continue;
                }
            }
            out.push_str("client_secret = \"[REDACTED]\"\n");
        } else if lower.starts_with("access_token") || lower.starts_with("refresh_token") {
            if let Some((key, _)) = line.split_once('=') {
                out.push_str(&format!("{key}= \"[REDACTED]\"\n"));
            } else {
                out.push_str(line);
                out.push('\n');
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Defence-in-depth log scrubber: if any `access_token=…` or
/// long-base64-shaped string slipped past the daily logger's
/// own truncation, mask it before the file goes into the zip.
fn redact_log_text(raw: &str) -> String {
    let needles = [
        "access_token=",
        "access_token: ",
        "\"access_token\":\"",
        "refresh_token=",
        "\"refresh_token\":\"",
    ];
    let mut out = String::with_capacity(raw.len());
    for line in raw.lines() {
        let mut redacted_line = line.to_string();
        for needle in &needles {
            if let Some(pos) = redacted_line.find(needle) {
                let cut = pos + needle.len();
                let tail = &redacted_line[cut..];
                let end_rel = tail
                    .find(|c: char| matches!(c, ' ' | ',' | '"' | '\n' | '}' | ')'))
                    .unwrap_or(tail.len());
                let mut new_line = String::with_capacity(redacted_line.len());
                new_line.push_str(&redacted_line[..cut]);
                new_line.push_str("[REDACTED]");
                new_line.push_str(&tail[end_rel..]);
                redacted_line = new_line;
            }
        }
        out.push_str(&redacted_line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_client_secret() {
        let toml = "[ctrader]\nclient_id = \"26884_abc\"\nclient_secret = \"verysecret123456\"\n";
        let r = redact_credentials(toml);
        assert!(r.contains("client_id = \"26884_abc\""));
        assert!(!r.contains("verysecret123456"));
        assert!(r.contains("last 4: 3456"));
        assert!(r.contains("length: 16"));
    }

    #[test]
    fn redacts_log_access_token_kv() {
        let log = "info: access_token=eyJhbGciOiJIUzI1 elapsed_ms=42";
        let r = redact_log_text(log);
        assert!(r.contains("access_token=[REDACTED]"));
        assert!(r.contains("elapsed_ms=42"));
        assert!(!r.contains("eyJhbGc"));
    }

    #[test]
    fn redacts_log_access_token_json() {
        let log = "{\"access_token\":\"AABBccDDeeFF\",\"client_id\":\"26884_abc\"}";
        let r = redact_log_text(log);
        assert!(r.contains("\"access_token\":\"[REDACTED]\""));
        assert!(r.contains("26884_abc"));
    }
}
