//! Headless multi-timeframe Discovery sweep used to validate that the
//! genetic search produces meaningful strategies (rather than
//! noise-fit overfit candidates) on a single symbol's locally cached
//! dataset. Triggered by `--validation-mode` on the CLI; not reachable
//! from the HTTP server or the existing `--auto-discovery` path.
//!
//! Design: sequential per-TF runs that share `DiscoveryConfig::from_settings`
//! so the operator's `config.yaml` drives population / generations /
//! candidate-count — the in-code defaults are deliberately small for
//! unit tests and would not give a meaningful pass/fail signal here.
//! Each TF gets a hard timeout (`--validation-tf-timeout-secs`) so a
//! single bad TF cannot stall the whole sweep. The CSV is flushed after
//! every row so a crash mid-sweep still leaves partial evidence on disk.

use crate::app_services::{
    ServiceEvent,
    discovery::{DiscoveryRequest, start_discovery_job},
    jobs::JobState,
};
use crate::app_state::AppRuntimeConfig;
use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use neoethos_core::Settings;
use neoethos_search::{DiscoveryConfig, PropFirmRiskRules};
use std::fs::{File, OpenOptions, create_dir_all};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{info, warn};

/// TF priority order used to derive the higher-TFs vector for a given
/// `base_tf`. Drives the discovery search's multi-timeframe feature
/// stack — three steps above the base, clamped to the list end.
///
/// M1 is intentionally excluded from validation sweeps (noise-dominated,
/// pure cost for GA discovery). Operators who want it can pass
/// `--validation-tfs M1,...`.
const TF_PRIORITY: &[&str] = &["M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"];

/// Number of higher timeframes to include above the base TF.
const HIGHER_TF_LEVELS: usize = 3;

/// One row in the CSV — captured per TF after the Discovery job
/// reaches a terminal state (or times out).
#[derive(Debug, Clone)]
struct TfOutcome {
    tf: String,
    status: String,
    duration_secs: f64,
    candidate_count: u64,
    portfolio_count: u64,
    /// In-sample (stage-1) top Sharpe — the metric the GA optimized
    /// against. Inflated relative to OOS because the GA fit it directly.
    top_sharpe_is: Option<f64>,
    /// Out-of-sample top Sharpe — computed on the forward-test tail the
    /// GA never saw. The IS/OOS gap is itself diagnostic (#211); a small
    /// gap is a strong edge signal, a 3× gap means overfit.
    top_sharpe_oos: Option<f64>,
    top_max_dd_pct: Option<f64>,
    error_message: String,
}

impl TfOutcome {
    fn to_csv_row(&self) -> String {
        // No CSV crate dependency — the columns are simple scalars and
        // the only field that needs escaping is `error_message`. We
        // double-quote it and escape internal quotes per RFC 4180.
        let sharpe_is = self
            .top_sharpe_is
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default();
        let sharpe_oos = self
            .top_sharpe_oos
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default();
        let dd = self
            .top_max_dd_pct
            .map(|v| format!("{:.6}", v))
            .unwrap_or_default();
        let err_escaped = self.error_message.replace('"', "\"\"");
        format!(
            "{},{},{:.3},{},{},{},{},{},\"{}\"\n",
            self.tf,
            self.status,
            self.duration_secs,
            self.candidate_count,
            self.portfolio_count,
            sharpe_is,
            sharpe_oos,
            dd,
            err_escaped
        )
    }
}

/// CSV header — kept in sync with `TfOutcome::to_csv_row`. Stable
/// columns so downstream notebook analysis can rely on the order.
///
/// Column change vs prior schema: `top_sharpe` is now split into
/// `top_sharpe_is` (in-sample, stage-1) and `top_sharpe_oos`
/// (forward-test on the strictly held-out tail). A big IS-OOS gap is
/// itself diagnostic — see #211.
const CSV_HEADER: &str =
    "tf,status,duration_secs,candidate_count,portfolio_count,top_sharpe_is,top_sharpe_oos,top_max_dd_pct,error_message\n";

/// Run a multi-TF Discovery sweep on the first locally-discoverable
/// symbol (falls back to AUDUSD if none). Returns the exit code the
/// caller should propagate: 0 if at least one TF succeeded, 1 otherwise.
///
/// `min_generations` is the GA generation floor applied per TF —
/// `DiscoveryConfig.generations` is bumped up to this value when the
/// operator's `config.yaml` set it lower. `0` honors `config.yaml` as-is.
pub async fn run_validation_sweep(
    runtime: &AppRuntimeConfig,
    settings: &Settings,
    tfs_csv: &str,
    tf_timeout_secs: u64,
    min_generations: usize,
) -> Result<i32> {
    let symbol = resolve_symbol(runtime);
    let tfs = parse_tfs(tfs_csv);
    if tfs.is_empty() {
        anyhow::bail!("validation-mode received an empty --validation-tfs list");
    }

    let run_dir = build_run_dir()?;
    let csv_path = run_dir.join("sweep.csv");
    let summary_path = run_dir.join("summary.txt");

    info!(
        target: "neoethos_app::validation",
        symbol = %symbol,
        data_dir = %runtime.data_dir.display(),
        tfs = ?tfs,
        tf_timeout_secs,
        min_generations,
        csv = %csv_path.display(),
        "validation-mode sweep starting"
    );

    let mut csv_file = open_csv_writer(&csv_path)?;
    let mut outcomes: Vec<TfOutcome> = Vec::with_capacity(tfs.len());
    let sweep_start = Instant::now();

    for tf in &tfs {
        info!(
            target: "neoethos_app::validation",
            tf = %tf,
            "validation-mode: starting TF run"
        );
        let outcome = run_one_tf(
            runtime,
            settings,
            &symbol,
            tf,
            Duration::from_secs(tf_timeout_secs),
            min_generations,
        )
        .await;
        info!(
            target: "neoethos_app::validation",
            tf = %tf,
            status = %outcome.status,
            duration_secs = outcome.duration_secs,
            candidate_count = outcome.candidate_count,
            portfolio_count = outcome.portfolio_count,
            "validation-mode: finished TF run"
        );
        let row = outcome.to_csv_row();
        if let Err(err) = csv_file.write_all(row.as_bytes()) {
            warn!(
                target: "neoethos_app::validation",
                error = %err,
                "validation-mode: CSV write failed; continuing with in-memory outcomes"
            );
        } else if let Err(err) = csv_file.flush() {
            // Flush after every row so a crash leaves partial data on
            // disk. A flush failure is logged but does not abort the
            // sweep — the next row's flush will surface the same issue
            // if the disk is truly gone.
            warn!(
                target: "neoethos_app::validation",
                error = %err,
                "validation-mode: CSV flush failed; partial sweep may not be on disk"
            );
        }
        outcomes.push(outcome);
    }

    let total_elapsed = sweep_start.elapsed();
    let summary = build_summary(&symbol, &outcomes, total_elapsed);
    write_summary_file(&summary_path, &summary)?;
    println!("{}", summary);

    let any_success = outcomes
        .iter()
        .any(|o| o.status == "Succeeded" || o.status == "Degraded");
    Ok(if any_success { 0 } else { 1 })
}

/// Build the higher-TF vector for a given base. Walks
/// [`TF_PRIORITY`] and takes up to [`HIGHER_TF_LEVELS`] entries above
/// the base index. Returns an empty vector if the base is the last
/// (or not in the priority list) — Discovery accepts an empty
/// `higher_tfs` slice and runs single-timeframe.
fn higher_tfs_for(base_tf: &str) -> Vec<String> {
    let Some(idx) = TF_PRIORITY.iter().position(|tf| *tf == base_tf) else {
        return Vec::new();
    };
    TF_PRIORITY
        .iter()
        .skip(idx + 1)
        .take(HIGHER_TF_LEVELS)
        .map(|s| s.to_string())
        .collect()
}

fn parse_tfs(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn resolve_symbol(runtime: &AppRuntimeConfig) -> String {
    match neoethos_data::discover_symbols(&runtime.data_dir) {
        Ok(symbols) => symbols
            .first()
            .cloned()
            .unwrap_or_else(|| "AUDUSD".to_string()),
        Err(err) => {
            warn!(
                target: "neoethos_app::validation",
                data_dir = %runtime.data_dir.display(),
                error = %err,
                "validation-mode: discover_symbols failed; falling back to AUDUSD"
            );
            "AUDUSD".to_string()
        }
    }
}

fn build_run_dir() -> Result<PathBuf> {
    // Windows-safe ISO timestamp: drop `:` from `2026-05-24T12:34:56Z`
    // so `validation-runs\2026-05-24T12-34-56Z` is a legal path on every
    // OS in the support matrix.
    let stamp = Utc::now().format("%Y-%m-%dT%H-%M-%SZ").to_string();
    let dir = PathBuf::from("validation-runs").join(stamp);
    create_dir_all(&dir).with_context(|| format!("create validation run dir {}", dir.display()))?;
    Ok(dir)
}

fn open_csv_writer(path: &Path) -> Result<File> {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open validation CSV at {}", path.display()))?;
    file.write_all(CSV_HEADER.as_bytes())
        .with_context(|| format!("write CSV header to {}", path.display()))?;
    file.flush()
        .with_context(|| format!("flush CSV header to {}", path.display()))?;
    Ok(file)
}

/// Drive one Discovery job to a terminal state (or timeout) and
/// distil the result into a single [`TfOutcome`] row.
async fn run_one_tf(
    runtime: &AppRuntimeConfig,
    settings: &Settings,
    symbol: &str,
    base_tf: &str,
    tf_timeout: Duration,
    min_generations: usize,
) -> TfOutcome {
    let started = Instant::now();
    // Honor the operator's config.yaml — defaults in code are smaller
    // than what we need for a meaningful validation signal.
    let mut config = DiscoveryConfig::from_settings(settings);
    // #214 + F-304: bind the *actual* sweep symbol into the discovery
    // config so the cost-model lookup sees a real cTrader symbol
    // instead of the empty-string fallback. The settings
    // `system.symbol` may differ from the sweep symbol — `discover_symbols`
    // chose the latter from on-disk Parquet, and that's the symbol whose
    // data the GA actually backtests.
    //
    // F-304 (2026-05-28): `SystemConfig.account_currency` now exists
    // as a typed channel. `from_settings` above already populates
    // `evaluation_account_currency`; we no longer need the hardcoded
    // "USD" patch (which would override operator's GBP/EUR account
    // setting). If the operator hasn't configured a currency, the
    // value stays empty and the cost-model NaN guard fires loud.
    config.evaluation_symbol = symbol.to_string();
    // #215: floor the GA generation count so short-data TFs (D1/H4) can't
    // smoke-test through the sweep with a 0.2s run that produces a tiny
    // archive. The floor is applied per-TF — `min_generations = 0` skips
    // the override entirely.
    if min_generations > 0 && config.generations < min_generations {
        info!(
            target: "neoethos_app::validation",
            tf = %base_tf,
            configured_generations = config.generations,
            floor = min_generations,
            "applying --validation-min-generations floor"
        );
        config.generations = min_generations;
    }
    let higher_tfs = higher_tfs_for(base_tf);
    let request = DiscoveryRequest {
        data_root: runtime.data_dir.clone(),
        symbol: symbol.to_string(),
        base_tf: base_tf.to_string(),
        higher_tfs,
        config,
        prop_firm_rules: PropFirmRiskRules::default(),
    };

    // One-shot channel large enough to absorb the discovery progress
    // burst without dropping the terminal snapshot. 4096 mirrors the
    // pattern in `start_discovery_job` callers in the server path.
    let (tx, mut rx) = mpsc::channel::<ServiceEvent>(4096);

    let handle = match start_discovery_job(request, tx.clone()) {
        Ok(h) => h,
        Err(err) => {
            return TfOutcome {
                tf: base_tf.to_string(),
                status: "FailedToStart".to_string(),
                duration_secs: started.elapsed().as_secs_f64(),
                candidate_count: 0,
                portfolio_count: 0,
                top_sharpe_is: None,
                top_sharpe_oos: None,
                top_max_dd_pct: None,
                error_message: err.to_string(),
            };
        }
    };

    // Drop the local sender so when the discovery task finishes its
    // job (and drops its own sender clones held inside the spawned
    // task) the receiver's `recv` returns None. We don't actually
    // rely on close-to-detect-completion — we watch for terminal
    // JobState transitions in the snapshot itself — but holding the
    // sender open would prevent rx from closing if the discovery
    // task panics mid-flight, which would otherwise mask the bug.
    drop(tx);

    // Wait for a terminal snapshot OR the per-TF timeout. The
    // discovery job itself can't be cancelled cooperatively from
    // here without changing the public API, so on timeout we flag
    // it as Timeout and move on — the job's background task will
    // keep running until natural completion but its events will
    // hit a closed channel (drop above + recv-loop exit) and be
    // silently dropped.
    let mut last_snapshot = handle.snapshot.clone();
    let drain = async {
        while let Some(event) = rx.recv().await {
            if let ServiceEvent::DiscoveryUpdated(snap) = event {
                let state = snap.state;
                last_snapshot = snap;
                if matches!(
                    state,
                    JobState::Succeeded
                        | JobState::Degraded
                        | JobState::Failed
                        | JobState::Cancelled
                ) {
                    return;
                }
            }
        }
        // rx closed before reaching terminal — the spawned task
        // dropped its sender (panic) or completed without emitting
        // a final terminal snapshot. Either way we treat the last
        // snapshot we saw as authoritative.
    };

    let timed_out = timeout(tf_timeout, drain).await.is_err();
    let duration_secs = started.elapsed().as_secs_f64();

    if timed_out {
        // Best-effort: request cancellation so the background task
        // exits its tight loops at the next checkpoint. The CSV row
        // still shows Timeout because we did not get a terminal
        // snapshot within the cap.
        handle.cancel.request();
        return TfOutcome {
            tf: base_tf.to_string(),
            status: "Timeout".to_string(),
            duration_secs,
            candidate_count: 0,
            portfolio_count: 0,
            top_sharpe_is: None,
            top_sharpe_oos: None,
            top_max_dd_pct: None,
            error_message: format!("hit per-TF timeout {}s", tf_timeout.as_secs()),
        };
    }

    snapshot_to_outcome(base_tf, &last_snapshot, duration_secs)
}

fn snapshot_to_outcome(
    base_tf: &str,
    snapshot: &crate::app_services::jobs::JobSnapshot,
    duration_secs: f64,
) -> TfOutcome {
    let status = match snapshot.state {
        JobState::Queued => "Queued",
        JobState::Running => "Running",
        JobState::Succeeded => "Succeeded",
        JobState::Degraded => "Degraded",
        JobState::Failed => "Failed",
        JobState::Cancelled => "Cancelled",
    }
    .to_string();

    // `completed_snapshot` populates these counters on Succeeded /
    // Degraded; on Failed we get whatever the last progress event
    // wrote, which can be zero. That's fine — the CSV column will
    // reflect "we did not get there", which IS the signal validation
    // mode is meant to surface.
    let candidate_count = counter_value(snapshot, "candidates").unwrap_or(0);
    let portfolio_count = counter_value(snapshot, "portfolio").unwrap_or(0);

    let top_sharpe_is = highlight_f64(snapshot, "best_sharpe");
    let top_sharpe_oos = highlight_f64(snapshot, "best_oos_sharpe");
    let top_max_dd_pct = highlight_f64(snapshot, "best_max_dd");

    // #213 diagnostic: when the funnel produced a non-zero candidate
    // pool but no portfolio, surface the funnel counters so an operator
    // can tell whether genes were rejected on `min_trades`, `passes_filter`,
    // or `nonzero_signal`. The data is already in `snapshot.report.counters`
    // — we just log it at sweep-end so the validation harness produces a
    // single line of attribution per TF instead of forcing the operator
    // to read the full discovery log.
    if candidate_count > 0 && portfolio_count == 0 {
        let post_passes_filter = counter_value(snapshot, "filtered_candidates").unwrap_or(0);
        let post_min_trades = counter_value(snapshot, "quality_screened").unwrap_or(0);
        let min_trades_required = counter_value(snapshot, "min_trades_required").unwrap_or(0);
        warn!(
            target: "neoethos_app::validation",
            tf = %base_tf,
            candidate_count,
            post_passes_filter,
            post_min_trades,
            min_trades_required,
            "validation-mode: TF produced candidates but zero portfolio — \
             funnel rejected every candidate (see counters)"
        );
    }

    let error_message = if matches!(snapshot.state, JobState::Failed | JobState::Cancelled) {
        // The summary string carries the most operator-friendly
        // failure text — prefer it; fall back to the first error
        // entry if summary is empty.
        if !snapshot.report.summary.is_empty() {
            snapshot.report.summary.clone()
        } else {
            snapshot.report.errors.first().cloned().unwrap_or_default()
        }
    } else {
        String::new()
    };

    TfOutcome {
        tf: base_tf.to_string(),
        status,
        duration_secs,
        candidate_count,
        portfolio_count,
        top_sharpe_is,
        top_sharpe_oos,
        top_max_dd_pct,
        error_message,
    }
}

fn counter_value(snapshot: &crate::app_services::jobs::JobSnapshot, key: &str) -> Option<u64> {
    snapshot
        .report
        .counters
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| *v)
}

fn highlight_f64(snapshot: &crate::app_services::jobs::JobSnapshot, key: &str) -> Option<f64> {
    snapshot
        .report
        .highlights
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| v.parse::<f64>().ok())
}

fn build_summary(symbol: &str, outcomes: &[TfOutcome], total_elapsed: Duration) -> String {
    let mut buf = String::new();
    buf.push_str("=== NeoEthos validation sweep summary ===\n");
    buf.push_str(&format!("symbol: {}\n", symbol));
    buf.push_str(&format!(
        "total_elapsed_secs: {:.3}\n",
        total_elapsed.as_secs_f64()
    ));
    buf.push_str(&format!("tf_count: {}\n", outcomes.len()));

    let succeeded = outcomes
        .iter()
        .filter(|o| o.status == "Succeeded" || o.status == "Degraded")
        .count();
    let failed = outcomes.len() - succeeded;
    buf.push_str(&format!("succeeded: {}\n", succeeded));
    buf.push_str(&format!("failed: {}\n", failed));

    buf.push_str("\nper-TF runtime:\n");
    for outcome in outcomes {
        buf.push_str(&format!(
            "  {:<5} status={:<12} duration={:>8.2}s candidates={:>5} portfolio={:>5} \
             top_sharpe_is={} top_sharpe_oos={}\n",
            outcome.tf,
            outcome.status,
            outcome.duration_secs,
            outcome.candidate_count,
            outcome.portfolio_count,
            outcome
                .top_sharpe_is
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".to_string()),
            outcome
                .top_sharpe_oos
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".to_string()),
        ));
    }

    // BEST TF by OOS sharpe (preferred when present) falling back to IS
    // — only consider successful runs so a Timeout/Failed row with a
    // zero sharpe can't masquerade as best. OOS is what an operator
    // actually cares about: the GA always inflates IS, so picking on IS
    // would crown the most overfit TF.
    let best = outcomes
        .iter()
        .filter(|o| {
            (o.status == "Succeeded" || o.status == "Degraded")
                && (o.top_sharpe_oos.is_some() || o.top_sharpe_is.is_some())
        })
        .max_by(|a, b| {
            let a_key = a.top_sharpe_oos.or(a.top_sharpe_is).unwrap_or(f64::MIN);
            let b_key = b.top_sharpe_oos.or(b.top_sharpe_is).unwrap_or(f64::MIN);
            a_key
                .partial_cmp(&b_key)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    match best {
        Some(o) => buf.push_str(&format!(
            "\nbest_tf: {} (top_sharpe_is={:.4} top_sharpe_oos={})\n",
            o.tf,
            o.top_sharpe_is.unwrap_or(0.0),
            o.top_sharpe_oos
                .map(|v| format!("{:.4}", v))
                .unwrap_or_else(|| "-".to_string()),
        )),
        None => buf.push_str("\nbest_tf: <none — every TF failed or had no portfolio>\n"),
    }
    buf
}

fn write_summary_file(path: &Path, body: &str) -> Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .map_err(|err| anyhow!("open summary {}: {err}", path.display()))?;
    file.write_all(body.as_bytes())
        .map_err(|err| anyhow!("write summary {}: {err}", path.display()))?;
    file.flush()
        .map_err(|err| anyhow!("flush summary {}: {err}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_tfs_for_h1_picks_three_above() {
        assert_eq!(
            higher_tfs_for("H1"),
            vec!["H4".to_string(), "D1".to_string(), "W1".to_string()]
        );
    }

    #[test]
    fn higher_tfs_for_h4_clamps_at_list_end() {
        assert_eq!(
            higher_tfs_for("H4"),
            vec!["D1".to_string(), "W1".to_string(), "MN1".to_string()]
        );
    }

    #[test]
    fn higher_tfs_for_m5_takes_first_three() {
        assert_eq!(
            higher_tfs_for("M5"),
            vec!["M15".to_string(), "M30".to_string(), "H1".to_string()]
        );
    }

    #[test]
    fn higher_tfs_for_mn1_returns_empty_at_top_of_priority_list() {
        assert!(higher_tfs_for("MN1").is_empty());
    }

    #[test]
    fn higher_tfs_for_unknown_tf_returns_empty() {
        assert!(higher_tfs_for("XYZ").is_empty());
    }

    #[test]
    fn parse_tfs_trims_and_drops_blanks() {
        let parsed = parse_tfs(" M5, M15 ,, H1,");
        assert_eq!(parsed, vec!["M5", "M15", "H1"]);
    }

    #[test]
    fn parse_tfs_default_csv_yields_six_entries() {
        // Mirrors the clap `default_value` literal on `--validation-tfs`
        // (see main.rs Args struct). If the default ever drifts, update
        // both sides.
        let parsed = parse_tfs("M5,M15,M30,H1,H4,D1");
        assert_eq!(parsed, vec!["M5", "M15", "M30", "H1", "H4", "D1"]);
    }

    #[test]
    fn csv_row_escapes_embedded_quotes() {
        let outcome = TfOutcome {
            tf: "H1".to_string(),
            status: "Failed".to_string(),
            duration_secs: 1.5,
            candidate_count: 0,
            portfolio_count: 0,
            top_sharpe_is: None,
            top_sharpe_oos: None,
            top_max_dd_pct: None,
            error_message: "load failed: missing \"frame\"".to_string(),
        };
        let row = outcome.to_csv_row();
        assert!(row.contains("\"load failed: missing \"\"frame\"\"\""));
        assert!(row.ends_with('\n'));
    }

    #[test]
    fn csv_row_emits_optional_sharpe_when_present() {
        let outcome = TfOutcome {
            tf: "H4".to_string(),
            status: "Succeeded".to_string(),
            duration_secs: 12.345,
            candidate_count: 42,
            portfolio_count: 7,
            top_sharpe_is: Some(1.8261),
            top_sharpe_oos: Some(0.9123),
            top_max_dd_pct: Some(0.0921),
            error_message: String::new(),
        };
        let row = outcome.to_csv_row();
        // tf,status,duration_secs(12.345),candidate(42),portfolio(7),
        // sharpe_is(1.826100),sharpe_oos(0.912300),dd(0.092100),""
        assert!(row.starts_with(
            "H4,Succeeded,12.345,42,7,1.826100,0.912300,0.092100,\"\""
        ));
    }

    // #211: header carries both IS and OOS sharpe columns. Downstream
    // notebook analysis relies on the column order being stable — this
    // test traps drift between header and row.
    #[test]
    fn csv_header_includes_both_is_and_oos_sharpe_columns() {
        assert!(CSV_HEADER.contains("top_sharpe_is"));
        assert!(CSV_HEADER.contains("top_sharpe_oos"));
        // Column order: IS comes before OOS (matches `to_csv_row`).
        let is_pos = CSV_HEADER.find("top_sharpe_is").unwrap();
        let oos_pos = CSV_HEADER.find("top_sharpe_oos").unwrap();
        assert!(is_pos < oos_pos);
        assert!(CSV_HEADER.ends_with('\n'));
    }

    #[test]
    fn csv_row_field_count_matches_header_field_count() {
        // Trap regressions where someone adds a CSV column to the row
        // but forgets the header (or vice versa). Compare comma counts
        // — the error_message field can contain embedded quotes but no
        // commas thanks to the RFC 4180 double-quoting.
        let outcome = TfOutcome {
            tf: "M5".to_string(),
            status: "Succeeded".to_string(),
            duration_secs: 1.0,
            candidate_count: 1,
            portfolio_count: 1,
            top_sharpe_is: Some(1.0),
            top_sharpe_oos: Some(0.5),
            top_max_dd_pct: Some(0.01),
            error_message: String::new(),
        };
        let row = outcome.to_csv_row();
        let header_commas = CSV_HEADER.matches(',').count();
        let row_commas = row.matches(',').count();
        assert_eq!(header_commas, row_commas);
    }

    #[test]
    fn build_summary_picks_highest_oos_sharpe_amongst_successful_tfs() {
        let outcomes = vec![
            TfOutcome {
                tf: "M5".to_string(),
                status: "Succeeded".to_string(),
                duration_secs: 1.0,
                candidate_count: 1,
                portfolio_count: 1,
                top_sharpe_is: Some(5.0),
                // IS is highest here but OOS is low — overfit. The
                // summary should NOT crown this one.
                top_sharpe_oos: Some(0.5),
                top_max_dd_pct: Some(0.01),
                error_message: String::new(),
            },
            TfOutcome {
                tf: "H4".to_string(),
                status: "Succeeded".to_string(),
                duration_secs: 2.0,
                candidate_count: 2,
                portfolio_count: 2,
                top_sharpe_is: Some(2.1),
                top_sharpe_oos: Some(2.0),
                top_max_dd_pct: Some(0.04),
                error_message: String::new(),
            },
            TfOutcome {
                tf: "D1".to_string(),
                status: "Failed".to_string(),
                duration_secs: 0.5,
                candidate_count: 0,
                portfolio_count: 0,
                // Even with a huge sharpe a Failed row must never be
                // crowned best — only Succeeded/Degraded count.
                top_sharpe_is: Some(9.9),
                top_sharpe_oos: Some(9.9),
                top_max_dd_pct: None,
                error_message: "boom".to_string(),
            },
        ];
        let summary = build_summary("AUDUSD", &outcomes, Duration::from_secs(3));
        assert!(summary.contains("best_tf: H4"));
        assert!(summary.contains("succeeded: 2"));
        assert!(summary.contains("failed: 1"));
    }

    #[test]
    fn build_summary_falls_back_to_is_sharpe_when_oos_missing() {
        // Older runs (or TFs that didn't produce a forward-test
        // artifact) may emit `best_sharpe` but not `best_oos_sharpe`.
        // The picker must fall back to IS in that case.
        let outcomes = vec![
            TfOutcome {
                tf: "M5".to_string(),
                status: "Succeeded".to_string(),
                duration_secs: 1.0,
                candidate_count: 1,
                portfolio_count: 1,
                top_sharpe_is: Some(0.5),
                top_sharpe_oos: None,
                top_max_dd_pct: Some(0.01),
                error_message: String::new(),
            },
            TfOutcome {
                tf: "H4".to_string(),
                status: "Succeeded".to_string(),
                duration_secs: 2.0,
                candidate_count: 2,
                portfolio_count: 2,
                top_sharpe_is: Some(2.1),
                top_sharpe_oos: None,
                top_max_dd_pct: Some(0.04),
                error_message: String::new(),
            },
        ];
        let summary = build_summary("AUDUSD", &outcomes, Duration::from_secs(3));
        assert!(summary.contains("best_tf: H4"));
    }

    #[test]
    fn build_summary_reports_no_best_when_every_tf_failed() {
        let outcomes = vec![TfOutcome {
            tf: "M5".to_string(),
            status: "Failed".to_string(),
            duration_secs: 1.0,
            candidate_count: 0,
            portfolio_count: 0,
            top_sharpe_is: None,
            top_sharpe_oos: None,
            top_max_dd_pct: None,
            error_message: "boom".to_string(),
        }];
        let summary = build_summary("AUDUSD", &outcomes, Duration::from_secs(1));
        assert!(summary.contains("best_tf: <none"));
        assert!(summary.contains("succeeded: 0"));
        assert!(summary.contains("failed: 1"));
    }
}
