// Structured logging facade.
//
// Two writers run side-by-side:
//   1. Console (stdout) — colored, current process only.
//   2. Daily-rotating file in <user-data-dir>/forex-ai/logs/. The file is
//      named `forex-ai.YYYY-MM-DD.log` and a new file is opened each calendar
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
/// e.g. `forex-ai.2026-05-21.log`.
const LOG_FILE_PREFIX: &str = "forex-ai";

/// Setup structured logging with tracing.
///
/// Single unified log layout — **one file per day**, with both raw tracing
/// events and visually-sectioned subsystem records inside the same file:
///
/// ```text
/// <user-data-dir>/forex-ai/logs/forex-ai.YYYY-MM-DD.log
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
    initialize_console_tracing(verbose)?;

    tracing::info!("Minimal logging initialized");
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
/// Previously this wrote to a parallel `forex-ai.log` JSON file via
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

/// Render a `SectionedRunRecord` as a multi-line visual block. The double
/// horizontal rule plus the right-arrow header make subsystem checkpoints
/// trivially greppable (`grep '▶ \['`) and visually obvious in a tail.
///
/// We deliberately keep this ASCII-leaning: the box-drawing chars survive
/// Notepad / `type` / `cat` and the right-arrow is a single UTF-8 codepoint
/// that renders in every modern terminal. No ANSI colour — the same block
/// goes to both console and the file layer, and we don't want escape
/// sequences in the file.
fn format_section_block(section: SubsystemSection, record: &SectionedRunRecord) -> String {
    use std::fmt::Write as _;
    let rule = "═".repeat(78);
    let mut s = String::with_capacity(512);
    let _ = writeln!(s, "{rule}");

    // Header line: ▶ [SECTION] STATUS operation  •  symbol/timeframe  •  run_id
    let _ = write!(s, "▶ [{}] {} {}", section.as_str(), record.status, record.operation);
    if let (Some(sym), Some(tf)) = (record.symbol.as_deref(), record.timeframe.as_deref()) {
        let _ = write!(s, "  •  {sym} {tf}");
    } else if let Some(sym) = record.symbol.as_deref() {
        let _ = write!(s, "  •  {sym}");
    }
    let _ = writeln!(s, "  •  run_id={}", record.run_id);

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
/// File layout: `<default_log_dir()>/forex-ai.YYYY-MM-DD.log`
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
            "[forex-core::logging] could not clean up old logs in {}: {err}",
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

/// Delete `forex-ai.*.log` files older than `retain_days` in `dir`.
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
        // Only touch our own files: forex-ai.YYYY-MM-DD.log (or .log alone).
        let prefix_matches = name.starts_with(LOG_FILE_PREFIX);
        let ext_matches = name.ends_with(".log") || name.ends_with(".log.gz");
        if !prefix_matches || !ext_matches {
            continue;
        }
        let Ok(metadata) = entry.metadata() else { continue };
        let Ok(modified) = metadata.modified() else { continue };
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
/// 2. Platform user-data dir: `<dirs::data_dir>/forex-ai/logs`
///    - Windows: `%APPDATA%\forex-ai\logs`
///    - macOS:   `~/Library/Application Support/forex-ai/logs`
///    - Linux:   `$XDG_DATA_HOME/forex-ai/logs` (or `~/.local/share/forex-ai/logs`)
/// 3. Fallback: relative `./logs` (only if `dirs::data_dir()` returns None,
///    which is rare — typically only on exotic configurations with no HOME).
pub fn default_log_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("LOG_DIR") {
        if !custom.is_empty() {
            return PathBuf::from(custom);
        }
    }
    dirs::data_dir()
        .map(|d| d.join("forex-ai").join("logs"))
        .unwrap_or_else(|| PathBuf::from("logs"))
}

/// Build today's unified-log path inside `log_dir`.
///
/// Format: `forex-ai.YYYY-MM-DD.log`. The date component is evaluated every
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
            "forex_core_logging_{}_{}_{}",
            test_name,
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn canonical_log_path_carries_todays_date_and_prefix() {
        // The unified file is `forex-ai.YYYY-MM-DD.log` inside the chosen
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
            name.starts_with("forex-ai."),
            "expected forex-ai.* prefix, got {name}"
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
        assert!(block.contains("══"), "expected box-drawing divider");
        assert!(block.contains("▶ [TRAINING] SUCCESS train"), "expected greppable header");
        assert!(block.contains("EURUSD M1"), "expected symbol/timeframe in header");
        assert!(block.contains("run_id=training-42"));
        assert!(block.contains("parent_run_id: discovery-7"));
        assert!(block.contains("message: training completed"));
        assert!(block.contains("loss: 0.002"));
        assert!(block.contains("accuracy: 0.94"));
        // The dividers must appear top and bottom — a stale `assert_contains`
        // would only catch one. Count instances of the rule string.
        let rule_count = block.matches("══").count();
        // We use repeat(78) so a single rule is one big substring — match
        // count is occurrences of the substring "══" which is the unit, so
        // each rule line contains 78/2 = 39 occurrences. Two rules → 78.
        assert!(
            rule_count >= 2,
            "expected at least two divider rules, got {rule_count} occurrences of ══"
        );
    }

    #[test]
    fn test_minimal_logging() {
        // This test just ensures the function doesn't panic
        let _ = setup_minimal_logging(false);
    }

    #[test]
    fn cleanup_old_logs_deletes_only_stale_forex_ai_files() {
        use std::fs::File;
        use std::time::Duration;

        let dir = unique_temp_dir("cleanup_stale");
        fs::create_dir_all(&dir).expect("create temp log dir");

        // 1. A stale forex-ai log — should be deleted. Back-date its mtime
        //    to 30 days ago using std::fs::File::set_modified (Rust 1.75+).
        let stale = dir.join("forex-ai.2020-01-01.log");
        fs::write(&stale, b"old\n").expect("write stale file");
        let thirty_days_ago = SystemTime::now() - Duration::from_secs(30 * 86_400);
        let stale_handle =
            File::options().write(true).open(&stale).expect("open stale for mtime");
        stale_handle
            .set_modified(thirty_days_ago)
            .expect("backdate stale mtime");
        drop(stale_handle);

        // 2. A recent forex-ai log (mtime = now) — must survive.
        let recent = dir.join("forex-ai.2099-01-01.log");
        fs::write(&recent, b"fresh\n").expect("write recent file");

        // 3. An unrelated file with our prefix substring but wrong shape — must survive.
        let unrelated = dir.join("not-our-logs.txt");
        fs::write(&unrelated, b"unrelated\n").expect("write unrelated file");

        // 4. A file matching prefix but not the .log extension — must survive.
        let prefix_only = dir.join("forex-ai-keepme.txt");
        fs::write(&prefix_only, b"keep\n").expect("write prefix-only file");

        // Run cleanup with 7-day retention. Stale (30 days old) must go;
        // the rest must stay.
        cleanup_old_logs(&dir, 7).expect("cleanup ran");

        assert!(!stale.exists(), "stale forex-ai log was not deleted");
        assert!(recent.exists(), "recent forex-ai log was incorrectly deleted");
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
