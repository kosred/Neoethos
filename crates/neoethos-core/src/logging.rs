// Structured logging facade.
//
// Two writers run side-by-side:
//   1. Console (stdout) — colored, current process only.
//   2. Daily-rotating file in <user-data-dir>/neoethos/logs/. The file is
//      named `neoethos.YYYY-MM-DD.log` and a new file is opened each calendar
//      day. On startup, files older than `LOG_RETENTION_DAYS` are deleted so
//      the log directory stays focused on the current week.
//
// The user explicitly requested "logs of today, not garbage of days/months",
// so the default retention is intentionally short. Override the dir with
// the `LOG_DIR` environment variable.

use crate::sectioned_log::{SectionedRunRecord, SubsystemSection};
use chrono::Utc;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{Duration, SystemTime};
use tracing::Level;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

static TRACING_INITIALIZED: OnceLock<()> = OnceLock::new();

/// Number of days of historical log files to keep on disk. Anything older
/// is deleted at startup. Set deliberately low so the operator sees only
/// recent activity by default.
const LOG_RETENTION_DAYS: u64 = 7;

/// Filename prefix used by the daily file rotator. The rotator appends
/// the calendar date and the `.log` suffix automatically, producing
/// e.g. `neoethos.2026-05-21.log`.
const LOG_FILE_PREFIX: &str = "neoethos";

/// Setup structured logging with tracing.
///
/// Single unified log layout — **one file per day**, with both raw tracing
/// events and visually-sectioned subsystem records inside the same file:
///
/// ```text
/// <user-data-dir>/neoethos/logs/neoethos.YYYY-MM-DD.log
/// ```
///
/// - **Console (stdout)** — colored, INFO/DEBUG depending on `verbose`.
/// - **Daily-rotating file** — same path returned by `canonical_log_path()`.
///   A new file opens each calendar day. Files older than 7 days are deleted
///   at startup so the operator sees only the current week.
/// - **Subsystem records** (`write_subsystem_record` callers) emit a
///   formatted multi-line block into the same file with visual dividers,
///   so a human tail/Notepad scroll surfaces each subsystem checkpoint
///   without searching across multiple log files.
///
/// Override the log directory with the `LOG_DIR` environment variable.
pub fn setup_logging(verbose: bool) -> anyhow::Result<()> {
    // Switch the console to UTF-8 BEFORE any tracing macro fires.
    // No-op on Linux/macOS (they default to UTF-8 already); on
    // Windows this flips the active code page from CP-1252/437 to
    // CP_UTF8 so Greek characters, the box-drawing ▶ in
    // `format_section_block`, the ULP-tick em-dashes in error
    // copy, etc. don't render as `?` or mojibake. Failure is
    // non-fatal — the operator just loses Unicode in the console
    // (the file layer is unaffected; that path writes UTF-8 bytes
    // verbatim regardless of console code page).
    if let Err(err) = configure_console_for_utf8() {
        // Use eprintln rather than tracing — tracing isn't up yet.
        eprintln!(
            "[neoethos] non-fatal: could not configure console for UTF-8 ({err}); \
             non-ASCII characters may render as `?` in this terminal"
        );
    }

    initialize_console_and_file_tracing(verbose)?;
    write_subsystem_record(
        SubsystemSection::System,
        system_record(
            "setup_logging",
            "SUCCESS",
            format!("logging initialized (verbose={verbose})"),
        ),
    )?;

    tracing::info!("Logging initialized (verbose={})", verbose);
    tracing::info!("Unified log file: {}", canonical_log_path().display());
    tracing::info!("Daily log directory: {}", default_log_dir().display());

    Ok(())
}

/// Setup minimal logging (console only, no files)
pub fn setup_minimal_logging(verbose: bool) -> anyhow::Result<()> {
    if let Err(err) = configure_console_for_utf8() {
        eprintln!("[neoethos] non-fatal: could not configure console for UTF-8 ({err})");
    }
    initialize_console_tracing(verbose)?;

    tracing::info!("Minimal logging initialized");
    Ok(())
}

/// Show a one-shot info dialog on Windows when the binary was
/// double-clicked directly (no Flutter shell parent).
///
/// Context (task #101): when an end-user double-clicks
/// `neoethos-app.exe` from a file manager, the binary is built with
/// `windows_subsystem = "windows"` so NO console window appears.
/// The HTTP server starts, binds 127.0.0.1:7423, but there is no
/// visible feedback — the user assumes it crashed silently. This
/// helper pops a Win32 MessageBox telling the user where to find the
/// actual NeoEthos UI.
///
/// **2026-08-09 (dead-code purge D2)**: the `NEOETHOS_LAUNCHED_BY_FLUTTER`
/// escape hatch was deleted with the rest of the Flutter surface. Nothing set
/// it after the 2026-06-22 Tauri migration, so the dialog was already
/// unconditional here.
///
/// Failure modes:
/// - Non-Windows: no-op (CLI/terminal users see logs directly).
/// - Debug builds: silent (developers run from terminal; popups annoy).
/// - MessageBoxW fails: silent (no console fallback either; the only
///   user impact is the missing dialog).
///
/// Returns immediately — the dialog is shown synchronously but the
/// HTTP server hasn't started yet, so this brief block is fine.
pub fn show_double_click_help_dialog_if_orphaned(server_url: &str) {
    // Skip in debug — devs run from terminal and don't need the popup.
    if cfg!(debug_assertions) {
        return;
    }
    #[cfg(windows)]
    {
        use windows::Win32::UI::WindowsAndMessaging::{
            MB_ICONINFORMATION, MB_OK, MB_SETFOREGROUND, MB_TOPMOST, MessageBoxW,
        };
        use windows::core::PCWSTR;

        // Build the body — keep it short, give the user a clear next
        // step. The server_url ends up in `body` so power users can
        // confirm the port matches their expectation.
        let body = format!(
            "NeoEthos backend is running on {server_url}.\n\n\
             This is the BACKEND server — it has no window of its own.\n\n\
             To use NeoEthos:\n\
             1. Close this dialog (the backend keeps running).\n\
             2. Launch the NeoEthos shortcut from the Start menu \
                or Desktop. The UI will connect to this backend \
                automatically.\n\n\
             If you don't have a NeoEthos shortcut, reinstall NeoEthos \
             — the installer creates one. You can stop this backend \
             by closing it from Task Manager (neoethos-app.exe)."
        );
        let title = "NeoEthos backend";
        let title_w: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
        let body_w: Vec<u16> = body.encode_utf16().chain(std::iter::once(0)).collect();
        // SAFETY: Pointers come from `Vec<u16>`s that outlive the call;
        // MessageBoxW takes wide-string pointers and an HWND (null is
        // valid = no owner window). Returns the user's choice; we
        // discard it since the dialog has a single OK button.
        unsafe {
            MessageBoxW(
                None,
                PCWSTR(body_w.as_ptr()),
                PCWSTR(title_w.as_ptr()),
                MB_OK | MB_ICONINFORMATION | MB_TOPMOST | MB_SETFOREGROUND,
            );
        }
    }
    #[cfg(not(windows))]
    let _ = server_url;
}

/// Switch the active console to UTF-8 on Windows, no-op elsewhere.
///
/// Why: Windows consoles default to a legacy code page (1252 on
/// Western installs, 437 on US-English fresh installs, 1253 on Greek
/// locales, etc.) which mangles any UTF-8 bytes we write — Greek
/// labels in error messages, the ▶ box-drawing chars in
/// `format_section_block`, the em-dash separators in CLI help text.
/// `SetConsoleOutputCP(CP_UTF8)` flips just the active console
/// without touching system locale.
///
/// The fix is the same trick Python's `PYTHONUTF8=1`, Node's
/// `chcp 65001`, and Rust's `colored` crate use under the hood. We
/// do it once, at logging-init time, before any non-ASCII char hits
/// stdout.
///
/// Failure is non-fatal: if the call fails (running headless without
/// a real console, or under a different OS, or with stdin/stdout
/// already redirected), we leave the console alone. The file log
/// layer is unaffected — that writes UTF-8 bytes regardless of
/// console code page.
pub fn configure_console_for_utf8() -> anyhow::Result<()> {
    #[cfg(windows)]
    {
        // SAFETY: `SetConsoleOutputCP` is a thread-safe Win32 call
        // that touches only the calling process's console. No
        // invariants to uphold. Returns BOOL via windows-rs's
        // `Result<()>` wrapper; an Err here means the call failed.
        use windows::Win32::System::Console::SetConsoleOutputCP;
        const CP_UTF8: u32 = 65001;
        unsafe {
            SetConsoleOutputCP(CP_UTF8).map_err(|e| {
                anyhow::anyhow!(
                    "SetConsoleOutputCP(CP_UTF8) failed: {e} \
                     (no attached console, or insufficient permissions)"
                )
            })?;
        }
    }
    // Non-Windows: every modern terminal we'd run under (xterm,
    // gnome-terminal, kitty, alacritty, iTerm, macOS Terminal) is
    // UTF-8 by default. Nothing to do.
    #[cfg(not(windows))]
    let _ = ();
    Ok(())
}

/// Path of the unified log file for the *current* calendar day.
///
/// Always evaluated fresh — if the process runs across midnight the next
/// call returns tomorrow's filename, matching what `tracing-appender`'s
/// daily rotator writes to. UI buttons like "Open log" call this every
/// time so they always open today's file.
pub fn canonical_log_path() -> PathBuf {
    canonical_log_path_from_dir(default_log_dir())
}

/// Emit a subsystem checkpoint into the unified log file.
///
/// The record is formatted as a multi-line block and routed through tracing,
/// so subsystem checkpoints and the live event stream share one daily file.
pub fn write_subsystem_record(
    section: SubsystemSection,
    record: SectionedRunRecord,
) -> anyhow::Result<()> {
    let block = format_section_block(section, &record);
    // The target prefix `subsystem.*` is what makes these blocks scannable
    // in the file (e.g. `grep target=subsystem.training`). tracing's
    // `target:` macro arg must be a string literal — the macro stashes it
    // into a `static __CALLSITE` at compile time. So we match on the enum
    // and hand each arm its own literal. This keeps the per-subsystem
    // grep affordance without leaving a runtime-formatted target lying
    // around (which would silently fall back to the module path).
    match section {
        SubsystemSection::System => {
            tracing::info!(target: "subsystem.system", "{block}");
        }
        SubsystemSection::App => {
            tracing::info!(target: "subsystem.app", "{block}");
        }
        SubsystemSection::Cli => {
            tracing::info!(target: "subsystem.cli", "{block}");
        }
        SubsystemSection::Discovery => {
            tracing::info!(target: "subsystem.discovery", "{block}");
        }
        SubsystemSection::Training => {
            tracing::info!(target: "subsystem.training", "{block}");
        }
        SubsystemSection::Bindings => {
            tracing::info!(target: "subsystem.bindings", "{block}");
        }
    }
    Ok(())
}

/// Render a `SectionedRunRecord` as a multi-line visual block. The horizontal
/// rule plus the right-arrow header make subsystem checkpoints trivially
/// greppable (`grep '> \['`) and visually obvious in a tail.
///
/// Pure ASCII — no box-drawing Unicode chars. When the block travels through
/// a pipe (TUI jobs.rs BufReader) on a Windows Greek locale (CP1253 default),
/// multi-byte UTF-8 codepoints like ═ (E2 95 90) and ▶ (E2 96 B6) render as
/// mojibake (`âÃÃÃ`). ASCII is always safe regardless of code page.
fn format_section_block(section: SubsystemSection, record: &SectionedRunRecord) -> String {
    use std::fmt::Write as _;
    let rule = "=".repeat(78);
    let mut s = String::with_capacity(512);
    let _ = writeln!(s, "{rule}");

    // Header line: > [SECTION] STATUS operation  |  symbol/timeframe  |  run_id
    let _ = write!(
        s,
        "> [{}] {} {}",
        section.as_str(),
        record.status,
        record.operation
    );
    if let (Some(sym), Some(tf)) = (record.symbol.as_deref(), record.timeframe.as_deref()) {
        let _ = write!(s, "  |  {sym} {tf}");
    } else if let Some(sym) = record.symbol.as_deref() {
        let _ = write!(s, "  |  {sym}");
    }
    let _ = writeln!(s, "  |  run_id={}", record.run_id);

    if let Some(parent) = record.parent_run_id.as_deref() {
        let _ = writeln!(s, "  parent_run_id: {parent}");
    }
    let _ = writeln!(s, "  started:  {}", record.started_at);
    let _ = writeln!(s, "  finished: {}", record.finished_at);
    if let Some(code) = record.error_code.as_deref() {
        let _ = writeln!(s, "  error_code: {code}");
    }
    if !record.message.is_empty() {
        let _ = writeln!(s, "  message: {}", record.message);
    }
    if !record.body.is_empty() {
        let _ = writeln!(s, "  body:");
        for line in record.body.lines() {
            let _ = writeln!(s, "    {line}");
        }
    }
    let _ = writeln!(s, "{rule}");
    s
}

fn initialize_console_tracing(verbose: bool) -> anyhow::Result<()> {
    if TRACING_INITIALIZED.get().is_some() {
        return Ok(());
    }

    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let env_filter = build_env_filter(level);
    let console_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_ansi(true)
        .with_writer(std::io::stdout);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(console_layer)
        .try_init()
        .map_err(|err| anyhow::anyhow!("failed to initialize tracing subscriber: {err}"))?;
    // DOCUMENTED-DEFAULT: TRACING_INITIALIZED is a OnceLock idempotency
    // guard; `set` returning Err just means we initialised earlier.
    let _ = TRACING_INITIALIZED.set(());
    Ok(())
}

/// Console + daily-rotating file tracing. Called by `setup_logging`.
///
/// File layout: `<default_log_dir()>/neoethos.YYYY-MM-DD.log`
///
/// On the first call per process, files older than `LOG_RETENTION_DAYS` in
/// the log dir are deleted. The cleanup is best-effort — if it fails (perms,
/// missing dir, etc.) we log the failure to console and proceed.
///
/// `tracing-appender 0.2.4`'s `RollingFileAppender` is a blocking writer. We
/// deliberately avoid `non_blocking()` here because that returns a
/// `WorkerGuard` that the caller must hold for the lifetime of the program;
/// changing `setup_logging`'s signature to return that guard would break
/// every downstream caller. Blocking I/O is acceptable for a desktop trading
/// app's log volume (tens of records per second at worst).
fn initialize_console_and_file_tracing(verbose: bool) -> anyhow::Result<()> {
    if TRACING_INITIALIZED.get().is_some() {
        return Ok(());
    }

    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let env_filter = build_env_filter(level);

    let log_dir = default_log_dir();
    // Ensure dir exists before either cleanup or the appender try to use it.
    // Best-effort: if create fails we'll still get console output below.
    let _ = fs::create_dir_all(&log_dir);

    // Best-effort cleanup of old daily files. Don't fail startup on perms etc.
    if let Err(err) = cleanup_old_logs(&log_dir, LOG_RETENTION_DAYS) {
        eprintln!(
            "[neoethos-core::logging] could not clean up old logs in {}: {err}",
            log_dir.display()
        );
    }

    let console_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_ansi(true)
        .with_writer(std::io::stdout);

    let file_appender = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix(LOG_FILE_PREFIX)
        .filename_suffix("log")
        .build(&log_dir)
        .map_err(|err| {
            anyhow::anyhow!(
                "failed to build rolling file appender at {}: {err}",
                log_dir.display()
            )
        })?;

    // No ANSI in the file — colour escape sequences become noise in a tail.
    let file_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_ansi(false)
        .with_writer(file_appender);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(console_layer)
        .with(file_layer)
        .try_init()
        .map_err(|err| anyhow::anyhow!("failed to initialize tracing subscriber: {err}"))?;
    let _ = TRACING_INITIALIZED.set(());
    Ok(())
}

/// Delete only exact daily log filenames older than `retain_days`.
///
/// Accepted names are `neoethos.YYYY-MM-DD.log[.gz]` with a valid date.
/// Other files in a shared LOG_DIR are never retention targets.
fn cleanup_old_logs(dir: &Path, retain_days: u64) -> std::io::Result<()> {
    let now = SystemTime::now();
    let max_age = Duration::from_secs(retain_days.saturating_mul(86_400));

    let read_dir = match fs::read_dir(dir) {
        Ok(it) => it,
        // No dir yet (first run) → nothing to clean. Not an error.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err),
    };

    for entry in read_dir.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !is_daily_log_filename(name) {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        if let Ok(age) = now.duration_since(modified)
            && age > max_age
        {
            let _ = fs::remove_file(&path);
        }
    }
    Ok(())
}

fn build_env_filter(level: Level) -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(format!("{level}"))
            .add_directive("httpcore=warn".parse().expect("valid directive"))
            .add_directive("httpx=warn".parse().expect("valid directive"))
            .add_directive("hyper=warn".parse().expect("valid directive"))
            .add_directive("reqwest=warn".parse().expect("valid directive"))
            .add_directive("h2=warn".parse().expect("valid directive"))
            .add_directive("tokio=info".parse().expect("valid directive"))
            .add_directive("runtime=info".parse().expect("valid directive"))
    })
}

/// Resolve the log directory.
///
/// Priority:
/// 1. `LOG_DIR` env var (escape hatch for tests, CI, sandboxed environments)
/// 2. Platform user-data dir: `<dirs::data_dir>/neoethos/logs`
///    - Windows: `%APPDATA%\neoethos\logs`
///    - macOS:   `~/Library/Application Support/neoethos/logs`
///    - Linux:   `$XDG_DATA_HOME/neoethos/logs` (or `~/.local/share/neoethos/logs`)
/// 3. Fallback: relative `./logs` (only if `dirs::data_dir()` returns None,
///    which is rare — typically only on exotic configurations with no HOME).
pub fn default_log_dir() -> PathBuf {
    // **F-CORE3 closure (2026-05-25)**: routed through the canonical
    // `env_overrides::log_dir_override` typed getter so the env-var
    // name lives in one grep-able place.
    if let Some(custom) = crate::env_overrides::log_dir_override() {
        return PathBuf::from(custom);
    }
    dirs::data_dir()
        .map(|d| d.join("neoethos").join("logs"))
        .unwrap_or_else(|| PathBuf::from("logs"))
}

/// Build today's UTC path, matching tracing-appender's daily rotation.
fn canonical_log_path_from_dir(log_dir: impl AsRef<Path>) -> PathBuf {
    canonical_log_path_at(log_dir.as_ref(), Utc::now())
}

fn canonical_log_path_at(log_dir: &Path, now: chrono::DateTime<Utc>) -> PathBuf {
    log_dir.join(format!("{LOG_FILE_PREFIX}.{}.log", now.format("%Y-%m-%d")))
}

fn is_daily_log_filename(name: &str) -> bool {
    let Some(date) = name
        .strip_prefix(LOG_FILE_PREFIX)
        .and_then(|tail| tail.strip_prefix('.'))
        .and_then(|tail| {
            tail.strip_suffix(".log.gz")
                .or_else(|| tail.strip_suffix(".log"))
        })
    else {
        return false;
    };
    date.len() == 10
        && date.bytes().enumerate().all(|(index, byte)| {
            if index == 4 || index == 7 {
                byte == b'-'
            } else {
                byte.is_ascii_digit()
            }
        })
        && chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").is_ok()
}

fn system_record(operation: &str, status: &str, message: String) -> SectionedRunRecord {
    let now = Utc::now().to_rfc3339();
    SectionedRunRecord {
        run_id: format!("system-{}-{}", operation, now.replace(':', "-")),
        parent_run_id: None,
        started_at: now.clone(),
        finished_at: now,
        subsystem: SubsystemSection::System,
        operation: operation.to_string(),
        status: status.to_string(),
        symbol: None,
        timeframe: None,
        error_code: None,
        message,
        body: String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sectioned_log::SubsystemSection;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(test_name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "neoethos_core_logging_{}_{}_{}",
            test_name,
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn canonical_log_path_uses_utc_across_local_midnight_and_year_boundaries() {
        for (timestamp, expected) in [
            ("2026-09-08T00:30:00+02:00", "neoethos.2026-09-07.log"),
            ("2026-12-31T23:30:00-02:00", "neoethos.2027-01-01.log"),
            ("2024-03-01T00:30:00+02:00", "neoethos.2024-02-29.log"),
        ] {
            let instant = chrono::DateTime::parse_from_rfc3339(timestamp)
                .unwrap()
                .with_timezone(&Utc);
            assert_eq!(
                canonical_log_path_at(Path::new("logs"), instant),
                Path::new("logs").join(expected)
            );
        }
    }

    #[test]
    fn canonical_path_matches_the_actual_daily_appender_file() {
        use std::io::Write;
        let dir = unique_temp_dir("actual_daily_appender");
        let before = Utc::now();
        let mut writer = RollingFileAppender::builder()
            .rotation(Rotation::DAILY)
            .filename_prefix(LOG_FILE_PREFIX)
            .filename_suffix("log")
            .build(&dir)
            .unwrap();
        writer.write_all(b"actual daily writer\n").unwrap();
        writer.flush().unwrap();
        drop(writer);
        let after = Utc::now();
        let files: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert!(!files.is_empty());
        for path in &files {
            assert!(
                path == &canonical_log_path_at(&dir, before)
                    || path == &canonical_log_path_at(&dir, after)
            );
            assert!(is_daily_log_filename(
                path.file_name().unwrap().to_str().unwrap()
            ));
        }
        assert!(
            files
                .iter()
                .any(|path| fs::read(path).unwrap() == b"actual daily writer\n")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn format_section_block_includes_visual_dividers_and_key_fields() {
        let record = SectionedRunRecord {
            run_id: "training-42".to_string(),
            parent_run_id: Some("discovery-7".to_string()),
            started_at: "2026-05-21T10:00:00Z".to_string(),
            finished_at: "2026-05-21T10:01:23Z".to_string(),
            subsystem: SubsystemSection::Training,
            operation: "train".to_string(),
            status: "SUCCESS".to_string(),
            symbol: Some("EURUSD".to_string()),
            timeframe: Some("M1".to_string()),
            error_code: None,
            message: "training completed".to_string(),
            body: "loss: 0.002\naccuracy: 0.94".to_string(),
        };
        let block = format_section_block(SubsystemSection::Training, &record);

        // Visual dividers wrap the block (two horizontal rules, one prefix line).
        assert!(block.contains("=="), "expected ASCII divider");
        assert!(
            block.contains("> [TRAINING] SUCCESS train"),
            "expected greppable header"
        );
        assert!(
            block.contains("EURUSD M1"),
            "expected symbol/timeframe in header"
        );
        assert!(block.contains("run_id=training-42"));
        assert!(block.contains("parent_run_id: discovery-7"));
        assert!(block.contains("message: training completed"));
        assert!(block.contains("loss: 0.002"));
        assert!(block.contains("accuracy: 0.94"));
        // The dividers must appear top and bottom.
        let rule_count = block.matches("==").count();
        assert!(
            rule_count >= 2,
            "expected at least two divider rules, got {rule_count} occurrences of =="
        );
    }

    #[test]
    fn test_minimal_logging() {
        // This test just ensures the function doesn't panic
        let _ = setup_minimal_logging(false);
    }

    #[test]
    fn cleanup_old_logs_deletes_only_stale_exact_daily_filenames() {
        let dir = unique_temp_dir("cleanup_stale");
        fs::create_dir_all(&dir).unwrap();
        let stale_names = ["neoethos.2020-01-01.log", "neoethos.2020-02-29.log.gz"];
        let unrelated_names = [
            "neoethos-notes.log",
            "neoethos.log",
            "neoethos.2020-02-30.log",
            "neoethos.2020-1-01.log",
            "neoethos.2020-01-01.log.backup",
            "neoethos.2020-01-01-notes.log.gz",
            "other.2020-01-01.log",
        ];
        let old = SystemTime::now() - Duration::from_secs(30 * 86_400);
        for name in stale_names.into_iter().chain(unrelated_names) {
            let path = dir.join(name);
            fs::write(&path, b"old\n").unwrap();
            fs::File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_modified(old)
                .unwrap();
        }
        let recent = dir.join("neoethos.2099-01-01.log");
        fs::write(&recent, b"fresh\n").unwrap();
        cleanup_old_logs(&dir, 7).unwrap();
        for name in stale_names {
            assert!(!dir.join(name).exists(), "stale exact log retained: {name}");
        }
        for name in unrelated_names {
            assert!(dir.join(name).exists(), "unrelated file removed: {name}");
        }
        assert!(recent.exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn cleanup_old_logs_succeeds_on_missing_dir() {
        // Cleanup pointed at a non-existent dir must succeed (NotFound is benign).
        let missing = unique_temp_dir("cleanup_missing").join("does-not-exist");
        let result = cleanup_old_logs(&missing, 7);
        assert!(
            result.is_ok(),
            "cleanup should treat missing dir as no-op, got {result:?}"
        );
    }

    #[test]
    fn default_log_dir_honours_log_dir_env_override() {
        // A child owns its environment. No parent mutation or private mutex
        // can race with the other logging tests or erase an existing LOG_DIR.
        const CHILD: &str = "NEOETHOS_LOG_DIR_TEST_CHILD";
        if let Some(expected) = std::env::var_os(CHILD) {
            assert_eq!(default_log_dir(), PathBuf::from(expected));
            return;
        }
        let sentinel = unique_temp_dir("log_dir_override");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "logging::tests::default_log_dir_honours_log_dir_env_override",
                "--test-threads=1",
                "--nocapture",
            ])
            .env("LOG_DIR", &sentinel)
            .env(CHILD, &sentinel)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            if child.try_wait().unwrap().is_some() {
                break;
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let output = child.wait_with_output().unwrap();
                panic!(
                    "LOG_DIR child timed out: stdout={} stderr={}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        let output = child.wait_with_output().unwrap();
        println!(
            "=== BEGIN LOG_DIR CHILD STDOUT ===\n{}\n=== END LOG_DIR CHILD STDOUT ===",
            String::from_utf8_lossy(&output.stdout)
        );
        eprintln!(
            "=== BEGIN LOG_DIR CHILD STDERR ===\n{}\n=== END LOG_DIR CHILD STDERR ===",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.status.success(),
            "LOG_DIR child failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "child must actually run its one exact test"
        );
    }
}
