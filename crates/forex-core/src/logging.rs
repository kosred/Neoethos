// Structured logging facade.

use crate::sectioned_log::{
    CanonicalSectionedLog, SectionedRunRecord, SubsystemSection, update_section_file,
};
use chrono::Utc;
use std::path::Path;
use std::path::PathBuf;
use std::sync::OnceLock;
use tracing::Level;
use tracing_subscriber::{EnvFilter, fmt, layer::SubscriberExt, util::SubscriberInitExt};

static TRACING_INITIALIZED: OnceLock<()> = OnceLock::new();

/// Setup structured logging with tracing
///
/// This configures:
/// - Console output with color and timestamps
/// - Canonical sectioned file updates via the shared log writer
/// - Environment variable filtering
/// - Silencing of noisy libraries
pub fn setup_logging(verbose: bool) -> anyhow::Result<()> {
    initialize_console_tracing(verbose)?;
    write_subsystem_record(
        SubsystemSection::System,
        system_record(
            "setup_logging",
            "SUCCESS",
            format!("logging initialized (verbose={verbose})"),
        ),
    )?;

    tracing::info!("Logging initialized (verbose={})", verbose);
    tracing::info!("Canonical log file: {:?}", canonical_log_path());

    Ok(())
}

/// Setup minimal logging (console only, no files)
pub fn setup_minimal_logging(verbose: bool) -> anyhow::Result<()> {
    initialize_console_tracing(verbose)?;

    tracing::info!("Minimal logging initialized");
    Ok(())
}

pub fn canonical_log_path() -> PathBuf {
    canonical_log_path_from_dir(default_log_dir())
}

pub fn write_subsystem_record(
    section: SubsystemSection,
    record: SectionedRunRecord,
) -> anyhow::Result<CanonicalSectionedLog> {
    write_subsystem_record_to_path(canonical_log_path(), section, record)
}

pub fn write_subsystem_record_to_path(
    path: impl AsRef<Path>,
    section: SubsystemSection,
    record: SectionedRunRecord,
) -> anyhow::Result<CanonicalSectionedLog> {
    update_section_file(path, section, record)
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

fn default_log_dir() -> PathBuf {
    std::env::var("LOG_DIR")
        .unwrap_or_else(|_| "logs".to_string())
        .into()
}

fn canonical_log_path_from_dir(log_dir: impl AsRef<Path>) -> PathBuf {
    log_dir.as_ref().join("forex-ai.log")
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
    fn test_canonical_log_path_uses_stable_filename() {
        let path = canonical_log_path_from_dir("logs");
        assert_eq!(path, PathBuf::from("logs").join("forex-ai.log"));
    }

    #[test]
    fn test_write_subsystem_record_to_path_updates_expected_section() {
        let dir = unique_temp_dir("write_subsystem_record");
        let path = canonical_log_path_from_dir(&dir);
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
    fn test_minimal_logging() {
        // This test just ensures the function doesn't panic
        let _ = setup_minimal_logging(false);
    }
}
