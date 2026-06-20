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

use crate::sectioned_log::{
    CanonicalSectionedLog, SectionedRunRecord, SubsystemSection, update_section_file,
};
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
/// helper detects "I was launched directly, not by Flutter" via the
/// `NEOETHOS_LAUNCHED_BY_FLUTTER` env var that the Flutter shell
/// sets when it spawns the backend, and pops a Win32 MessageBox
/// telling the user where to find the actual NeoEthos UI.
///
/// Failure modes:
/// - Non-Windows: no-op (CLI/terminal users see logs directly).
/// - Env var present: silent (Flutter shell spawn).
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
    // Skip when the Flutter shell launched us.
    // **F-CORE3 closure (2026-05-25)**: routed through the canonical
    // `env_overrides::launched_by_flutter` typed getter.
    if crate::env_overrides::launched_by_flutter() {
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
/// Previously this wrote to a parallel `neoethos.log` JSON file via
/// `update_section_file`. That created a *second* place operators had to
/// hunt for clues. Now the record is formatted as a multi-line block with
/// visual dividers and routed through tracing, so it lands in the **same**
/// daily file as the live event stream.
///
/// The return type is preserved for backward compatibility with callers
/// that ignore the result (every production caller does); we return an
/// empty `CanonicalSectionedLog` marker.
///
/// For tests that need the structured JSON round-trip, call
/// `write_subsystem_record_to_path` with an explicit path — it still
/// writes the legacy JSON snapshot file.
pub fn write_subsystem_record(
    section: SubsystemSection,
    record: SectionedRunRecord,
) -> anyhow::Result<CanonicalSectionedLog> {
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
    Ok(CanonicalSectionedLog::new())
}

/// Test-only escape hatch: write the legacy JSON sectioned snapshot to
/// an explicit path. Production code path is `write_subsystem_record`.
pub fn write_subsystem_record_to_path(
    path: impl AsRef<Path>,
    section: SubsystemSection,
    record: SectionedRunRecord,
) -> anyhow::Result<CanonicalSectionedLog> {
    update_section_file(path, section, record)
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

/// Delete `neoethos.*.log` files older than `retain_days` in `dir`.
///
/// Scoped narrowly to our own filename prefix so we never touch other files
/// even if the operator pointed `LOG_DIR` at a shared directory. The mtime
/// check uses the OS-reported modification time; we ignore files whose
/// metadata we can't read.
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
        // Only touch our own files: neoethos.YYYY-MM-DD.log (or .log alone).
        let prefix_matches = name.starts_with(LOG_FILE_PREFIX);
        let ext_matches = name.ends_with(".log") || name.ends_with(".log.gz");
        if !prefix_matches || !ext_matches {
            continue;
        }
        let Ok(metadata) = entry.metadata() else {
            continue;
        };
        let Ok(modified) = metadata.modified() else {
            continue;
        };
        if let Ok(age) = now.duration_since(modified) {
            if age > max_age {
                let _ = fs::remove_file(&path);
            }
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

/// Build today's unified-log path inside `log_dir`.
///
/// Format: `neoethos.YYYY-MM-DD.log`. The date component is evaluated every
/// call so the path follows the calendar — important when the app runs
/// across midnight. We use `chrono::Local` (not `Utc`) so the file name
/// matches what the operator's clock shows; tracing-appender's daily
/// rotator also uses local-time rotation by default.
fn canonical_log_path_from_dir(log_dir: impl AsRef<Path>) -> PathBuf {
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    log_dir
        .as_ref()
        .join(format!("{LOG_FILE_PREFIX}.{today}.log"))
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
    fn canonical_log_path_carries_todays_date_and_prefix() {
        // The unified file is `neoethos.YYYY-MM-DD.log` inside the chosen
        // dir, where YYYY-MM-DD is today's local date. We test the *shape*
        // (parent + prefix + suffix) rather than the exact date so the test
        // is stable across the midnight boundary.
        let path = canonical_log_path_from_dir("logs");
        let parent = path.parent().expect("path must have a parent");
        assert_eq!(parent, Path::new("logs"));
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .expect("path must have a filename");
        assert!(
            name.starts_with("neoethos."),
            "expected neoethos.* prefix, got {name}"
        );
        assert!(name.ends_with(".log"), "expected .log suffix, got {name}");
        // Sanity: the date segment is the chrono-formatted local date.
        let expected_date = chrono::Local::now().format("%Y-%m-%d").to_string();
        assert!(
            name.contains(&expected_date),
            "expected {expected_date} in filename, got {name}"
        );
    }

    #[test]
    fn test_write_subsystem_record_to_path_updates_expected_section() {
        // This exercises the *test-only* JSON snapshot path. Production now
        // routes through `write_subsystem_record` → tracing, which lands in
        // the unified daily file — that path is covered by the formatter
        // test below. We use a custom filename here so the test doesn't
        // collide with the daily-rotator filename pattern.
        let dir = unique_temp_dir("write_subsystem_record");
        fs::create_dir_all(&dir).expect("create temp log dir");
        let path = dir.join("legacy-snapshot.log");
        let record = SectionedRunRecord {
            run_id: "training-1".to_string(),
            parent_run_id: None,
            started_at: "2026-03-21T12:00:00Z".to_string(),
            finished_at: "2026-03-21T12:00:01Z".to_string(),
            subsystem: SubsystemSection::Training,
            operation: "train".to_string(),
            status: "SUCCESS".to_string(),
            symbol: Some("EURUSD".to_string()),
            timeframe: Some("M1".to_string()),
            error_code: None,
            message: "training ok".to_string(),
            body: "body".to_string(),
        };

        let log = write_subsystem_record_to_path(&path, SubsystemSection::Training, record.clone())
            .expect("section write should succeed");

        let training = log
            .section(SubsystemSection::Training)
            .expect("training section should exist");
        assert_eq!(training.current.as_ref(), Some(&record));
        assert!(path.exists(), "canonical log file should be created");

        let _ = fs::remove_file(path);
        let _ = fs::remove_dir_all(dir);
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
    fn cleanup_old_logs_deletes_only_stale_neoethos_files() {
        use std::fs::File;
        use std::time::Duration;

        let dir = unique_temp_dir("cleanup_stale");
        fs::create_dir_all(&dir).expect("create temp log dir");

        // 1. A stale neoethos log — should be deleted. Back-date its mtime
        //    to 30 days ago using std::fs::File::set_modified (Rust 1.75+).
        let stale = dir.join("neoethos.2020-01-01.log");
        fs::write(&stale, b"old\n").expect("write stale file");
        let thirty_days_ago = SystemTime::now() - Duration::from_secs(30 * 86_400);
        let stale_handle = File::options()
            .write(true)
            .open(&stale)
            .expect("open stale for mtime");
        stale_handle
            .set_modified(thirty_days_ago)
            .expect("backdate stale mtime");
        drop(stale_handle);

        // 2. A recent neoethos log (mtime = now) — must survive.
        let recent = dir.join("neoethos.2099-01-01.log");
        fs::write(&recent, b"fresh\n").expect("write recent file");

        // 3. An unrelated file with our prefix substring but wrong shape — must survive.
        let unrelated = dir.join("not-our-logs.txt");
        fs::write(&unrelated, b"unrelated\n").expect("write unrelated file");

        // 4. A file matching prefix but not the .log extension — must survive.
        let prefix_only = dir.join("neoethos-keepme.txt");
        fs::write(&prefix_only, b"keep\n").expect("write prefix-only file");

        // Run cleanup with 7-day retention. Stale (30 days old) must go;
        // the rest must stay.
        cleanup_old_logs(&dir, 7).expect("cleanup ran");

        assert!(!stale.exists(), "stale neoethos log was not deleted");
        assert!(
            recent.exists(),
            "recent neoethos log was incorrectly deleted"
        );
        assert!(unrelated.exists(), "unrelated file was incorrectly deleted");
        assert!(
            prefix_only.exists(),
            "prefix-matching non-.log file was incorrectly deleted"
        );

        let _ = fs::remove_dir_all(&dir);
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
        // SAFETY-NOTE: this test mutates a process-global env var. If run in
        // parallel with another test that reads LOG_DIR it could flake; the
        // workspace runs `cargo test` single-process per-crate so we accept
        // this trade-off rather than wiring serial_test in.
        let sentinel = unique_temp_dir("log_dir_override");
        unsafe {
            std::env::set_var("LOG_DIR", &sentinel);
        }
        let resolved = default_log_dir();
        assert_eq!(resolved, sentinel, "LOG_DIR override was not honoured");
        unsafe {
            std::env::remove_var("LOG_DIR");
        }
    }
}
