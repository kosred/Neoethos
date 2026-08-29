use anyhow::{Context, Result};
use neoethos_core::execution_budget::{
    StartupEvent, StartupRuntimeKind, StartupTrace, format_startup_diagnostics,
    parse_parent_cpu_assignment, startup_diagnostics_requested,
};
use neoethos_core::logging::{setup_logging, write_subsystem_record};
use neoethos_core::sectioned_log::{SectionedRunRecord, SubsystemSection};
use std::time::{SystemTime, UNIX_EPOCH};

mod canonical_full_run;
mod gpu_bench;
mod gpu_bench_population;
mod gpu_bench_prepare;
mod gpu_bench_snapshot;
mod native_research;
mod tui;

fn main() -> Result<()> {
    let mut startup_trace = StartupTrace::default();
    // Must precede every possible thread/runtime/global-pool initialization;
    // child threads inherit the blocked SourceSeal signal mask on Linux.
    neoethos_data::initialize_source_seal_before_runtime()?;
    startup_trace.record(StartupEvent::ImportSignalPreflightCompleted)?;
    // Config-consolidation: ordinary discovery/model runtime overrides come
    // from the canonical config, not the environment. Strict receipt-bound
    // historical search branches below before that config surface exists.
    //
    // ⚠ AUDIT #125/#289, CLI HALF — this line read
    // `Settings::load().unwrap_or_default()` and its own comment admitted
    // "Falls back to defaults if it can't be loaded". `Settings::default()` is
    // NOT the shipped configuration: `ModelsConfig::default()` still encodes
    // the pre-2026-06-06 posture and re-arms `require_walkforward_for_export`
    // and `prop_firm_min_pass_rate`, so a config that failed to load silently
    // changed WHICH STRATEGIES MAY REACH LIVE — on the binary that actually
    // runs discovery. The same defect was recorded for the desktop shell
    // (#125, `desktop/src-tauri/src/lib.rs:65`) and closed there; nothing in
    // the 323 items noticed this one.
    //
    // Now: a failed load is fatal for every subcommand that decides money or
    // spends hours, and survivable ONLY for the diagnostic commands you would
    // reach for to fix it (`config`, `credentials`, `setup`, `wizard`,
    // `--help`, `--version`). Those run on built-in defaults with an
    // unmissable banner, never quietly.
    let raw_args: Vec<String> = std::env::args().collect();
    // Owned, so the `raw_args` vector can be moved into `args` further down.
    let subcommand: String = raw_args.get(1).cloned().unwrap_or_default();
    if subcommand == "search" {
        neoethos_search::historical_search_cli::install_historical_search_process_budget(
            &raw_args,
        )?;
        setup_logging(false)?;
        return neoethos_search::historical_search_cli::run(&raw_args[2..]);
    }
    let startup_settings = match neoethos_core::Settings::load() {
        Ok(s) => s,
        Err(err) => {
            let path = neoethos_core::config::user_config_path();
            let diagnostic = matches!(
                subcommand.as_str(),
                // NOT the empty subcommand: a bare `neoethos-cli` opens the
                // TUI, which can START ENGINES. It takes over the terminal, so
                // the banner above scrolls away unread. `config` is the way in.
                "config"
                    | "credentials"
                    | "setup"
                    | "wizard"
                    | "--help"
                    | "-h"
                    | "help"
                    | "--version"
                    | "-V"
                    | "version"
                    | "native-research"
            );
            eprintln!("──────────────────────────────────────────────────────────────");
            eprintln!("CONFIG NOT LOADED");
            eprintln!(
                "  tried: $CONFIG_FILE, then {}, then ./config.yaml",
                path.display()
            );
            eprintln!("  {err:#}");
            eprintln!("  Built-in defaults are NOT your settings: they re-arm the export");
            eprintln!("  gates and risk limits you set deliberately, and they decide which");
            eprintln!("  strategies may reach live money.");
            eprintln!("──────────────────────────────────────────────────────────────");
            tracing::error!(
                target: "neoethos_cli",
                config = %path.display(),
                error = %format!("{err:#}"),
                subcommand = %subcommand,
                diagnostic_command = diagnostic,
                "config.yaml could not be loaded"
            );
            if !diagnostic {
                anyhow::bail!(
                    "refusing to run `{subcommand}` without a readable config.\n\
                     Put a valid config.yaml at {} (or in the working directory, or point \
                     $CONFIG_FILE at one) and re-run. `neoethos-cli config` still works \
                     without it.",
                    path.display()
                );
            }
            eprintln!(
                "Continuing on built-in defaults because `{subcommand}` is a diagnostic command."
            );
            neoethos_core::Settings::default()
        }
    };
    startup_trace.record(StartupEvent::ConfigurationLoaded)?;
    // CPU budget for THIS process — an internal parent→child handoff, not an
    // operator knob.
    //
    // It used to travel as the environment variable `NEOETHOS_BOT_CPU_BUDGET`,
    // set by `spawn_discover_combo` on each `discover` child the schedule
    // orchestrator starts. That was the config-chaos pattern in miniature: it
    // appeared in no config file and in no knob catalog, yet a stale export
    // left in a shell silently re-partitioned the cores of every later run,
    // and the old `.ok().and_then(parse.ok())` chain swallowed a typo without
    // a word — `NEOETHOS_BOT_CPU_BUDGET=eight` meant "no assignment".
    //
    // It is now `--cpu-threads`: visible in the process list, scoped to the
    // one invocation it was written for, and FATAL when malformed.
    let process_cpu_assignment = parse_parent_cpu_assignment(&raw_args)?;
    startup_trace.record(StartupEvent::ParentCpuCapParsed)?;
    warn_retired_env_vars();
    let coordination_scope = if process_cpu_assignment.is_some() {
        neoethos_core::execution_budget::CoordinationScope::ManagedProcessTree
    } else {
        neoethos_core::execution_budget::CoordinationScope::ProcessLocal
    };
    let execution_budget_inputs = neoethos_core::ExecutionBudgetInputs::from_settings_and_parent(
        &startup_settings,
        process_cpu_assignment.map(|limit| limit.get()),
        coordination_scope,
    )?;
    execution_budget_inputs.clone().resolve()?;
    startup_trace.record(StartupEvent::CpuBudgetResolved)?;
    let installed = neoethos_core::execution_budget::install_process_budget(
        execution_budget_inputs.request().clone(),
    )?;
    startup_trace.record(StartupEvent::CpuBudgetInstalled)?;
    // Logging and environment diagnostics may initialize process-global
    // state, so they run only after the immutable CPU budget is installed.
    setup_logging(false)?;
    neoethos_core::env_overrides::log_active_overrides_at_startup();
    neoethos_search::install_search_runtime_overrides_from_settings(&startup_settings);
    neoethos_models::tree_models::config::install_tree_runtime_from_settings(&startup_settings);
    neoethos_models::statistical::common::install_statistical_runtime_from_settings(
        &startup_settings,
    );
    neoethos_core::system::install_hardware_runtime_overrides_from_settings(&startup_settings);
    neoethos_data::install_data_runtime_overrides(
        startup_settings.models.data_runtime.normalize_features,
    );
    startup_trace.record(StartupEvent::RuntimeSettingsInstalled)?;
    if startup_diagnostics_requested(&raw_args) {
        eprintln!(
            "{}",
            format_startup_diagnostics(
                "neoethos-cli",
                installed,
                StartupRuntimeKind::Synchronous,
                None,
                &startup_trace,
            )
        );
        return Ok(());
    }
    // Same vector the config-load guard above already collected.
    let args: Vec<String> = raw_args;
    if args.len() < 2 {
        // No subcommand → launch interactive TUI. Use `--help` for
        // legacy bare help; explicit subcommands keep working
        // unchanged for scripting.
        if let Err(err) = write_subsystem_record(
            SubsystemSection::Cli,
            cli_record("tui", "STARTED", "launching interactive TUI"),
        ) {
            tracing::warn!(
                target: "neoethos_cli",
                error = %err,
                "failed to write CLI 'tui STARTED' subsystem record"
            );
        }
        let res = tui::run_tui(None);
        if let Err(err) = write_subsystem_record(
            SubsystemSection::Cli,
            cli_record(
                "tui",
                if res.is_ok() { "SUCCESS" } else { "FAILED" },
                match &res {
                    Ok(_) => "TUI session ended cleanly".to_string(),
                    Err(err) => format!("TUI session ended with error: {err}"),
                },
            ),
        ) {
            tracing::warn!(
                target: "neoethos_cli",
                error = %err,
                "failed to write CLI 'tui {}' subsystem record",
                if res.is_ok() { "SUCCESS" } else { "FAILED" }
            );
        }
        return res;
    }
    if matches!(args[1].as_str(), "--help" | "-h" | "help") {
        print_help();
        return Ok(());
    }
    let command = args[1].clone();
    write_subsystem_record(
        SubsystemSection::Cli,
        cli_record(
            &command,
            "STARTED",
            format!("starting CLI command {}", command),
        ),
    )?;

    let tail = &args[2..];
    let settings = &startup_settings;
    let result = match args[1].as_str() {
        "symbols" => cmd_symbols(&args[2..]),
        "timeframes" => cmd_timeframes(&args[2..]),
        "load" => cmd_load(&args[2..]),
        "features" => cmd_features(&args[2..]),
        "prepare" => cmd_prepare(&args[2..]),
        "canonical-cost-build" => canonical_full_run::build_cost_assumptions(tail, settings),
        "canonical-train" => canonical_full_run::train_receipt_bound(tail, settings),
        "canonical-full-run" => canonical_full_run::run(&args[2..], &startup_settings),
        "native-research" => native_research::run(tail),
        "train" => cmd_train(&args[2..]),
        "discover" => cmd_discover(&args[2..]),
        "discovery-promote-weekly" => cmd_discovery_promote_weekly(&args[2..]),
        "trader-replay" => cmd_trader_replay(&args[2..]),
        "forward-test" => cmd_forward_test(&args[2..]),
        "blend-test" => cmd_blend_test(&args[2..]),
        "batch-discover" => cmd_batch_discover(&args[2..]),
        "bench" => gpu_bench::run(&args[2..]),
        "bench-prepare" => gpu_bench_prepare::run_prepare(&args[2..]),
        "bench-matrix" => gpu_bench_prepare::run_matrix(&args[2..]),
        "bench-collate" => gpu_bench_prepare::run_collate(&args[2..]),
        "bench-preflight-report" => gpu_bench_prepare::run_preflight_report(&args[2..]),
        "slice-dataset" => cmd_slice_dataset(&args[2..]),
        "import" => cmd_import(&args[2..]),
        "config" => cmd_config(&args[2..]),
        "auto-loop" => cmd_auto_loop(&args[2..]),
        "autoresearch" => cmd_autoresearch(&args[2..]),
        "schedule" => cmd_schedule(&args[2..]),
        "stop-target" => cmd_stop_target(&args[2..]),
        "wizard" => cmd_wizard(&args[2..]),
        "setup" => cmd_setup(&args[2..]),
        "credentials" => cmd_credentials(&args[2..]),
        _ => {
            print_help();
            Ok(())
        }
    };

    match &result {
        Ok(_) => {
            write_subsystem_record(
                SubsystemSection::Cli,
                cli_record(
                    &command,
                    "SUCCESS",
                    format!("CLI command {} completed", command),
                ),
            )?;
        }
        Err(err) => {
            write_subsystem_record(
                SubsystemSection::Cli,
                cli_record(
                    &command,
                    "FAILED",
                    format!("CLI command {} failed: {}", command, err),
                ),
            )?;
        }
    }

    result
}

fn cmd_load(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let mut root = parse_root(args, settings.as_ref());
    let mut symbol = None;
    let mut timeframe = None;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--root" => {
                if let Some(val) = iter.next() {
                    root = val.to_string();
                }
            }
            "--symbol" => {
                if let Some(val) = iter.next() {
                    symbol = Some(val.to_string());
                }
            }
            "--timeframe" => {
                if let Some(val) = iter.next() {
                    timeframe = Some(val.to_string());
                }
            }
            _ => {}
        }
    }

    let symbol = symbol.unwrap_or_else(|| default_symbol(settings.as_ref()));
    let timeframe = timeframe.unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let identities = inventory_canonical_identities(&root, &symbol)?;
    let identity = select_exact_runtime_identity(&identities, args, &symbol, &timeframe)?;
    let ohlcv = load_exact_runtime_timeframe(&root, &identity)?;
    println!(
        "Loaded {} {} identity={} rows: {}",
        symbol,
        timeframe,
        identity.to_path_component(),
        ohlcv.len()
    );
    Ok(())
}

/// `slice-dataset --symbol EURUSD --base M1 --root <SRC> --out-root <DST>
///                --from-date 2018-01-01 --to-date 2021-01-01`
///
/// Additive, NON-destructive: resolves exactly one manifest-backed source
/// identity for `(symbol, base)`, fully verifies and pins that immutable Vortex
/// generation, keeps only bars in `[from-date, to-date)` (UTC), and atomically
/// publishes the result under the same exact identity in a NEW configured
/// root. The output manifest carries typed derivation provenance binding the
/// source identity, manifest, generation, Vortex hash and selected row/range.
///
/// Purpose: OOM-safe walk-forward. A multi-year M1 dataset that overflows
/// RAM on a weak machine can be chopped into e.g. 3-year windows that each
/// fit, discovered independently, and stitched by the operator.
///
/// Fails closed when source identity selection is missing/ambiguous, the
/// source generation cannot be fully verified, an output generation already
/// exists without an explicit CAS base, or the range yields zero rows.
const SLICE_PROVENANCE_DOMAIN: &[u8] = b"neoethos.cli.slice-dataset-provenance.v1\0";
const SLICE_PROVENANCE_VERSION: u16 = 1;
const SLICE_SELECTION_HALF_OPEN: u8 = 1;
const SLICE_VOLUME_ABSENT: u8 = 1;
const SLICE_VOLUME_FLOAT64: u8 = 2;
const SLICE_ROWS_PER_VORTEX_CHUNK: usize = 65_536;

#[derive(Clone, Debug, PartialEq, Eq)]
struct SliceDatasetProvenanceV1 {
    source_identity: neoethos_data::CanonicalDatasetIdentity,
    source_manifest_schema_id: String,
    source_manifest_hash: [u8; 32],
    source_generation: String,
    source_vortex_hash: [u8; 32],
    source_row_count: u64,
    source_timestamp_start_ms: i64,
    source_timestamp_end_ms: i64,
    selected_row_start: u64,
    selected_row_end: u64,
    requested_from_ms: i64,
    requested_to_ms: i64,
    selected_timestamp_start_ms: i64,
    selected_timestamp_end_ms: i64,
    volume_encoding: u8,
}

impl SliceDatasetProvenanceV1 {
    const SCHEMA_ID: &'static str = "neoethos.cli.slice-dataset-provenance.v1";

    #[allow(clippy::too_many_arguments)]
    fn new(
        source_identity: neoethos_data::CanonicalDatasetIdentity,
        source_manifest_schema_id: impl Into<String>,
        source_manifest_hash: [u8; 32],
        source_generation: impl Into<String>,
        source_vortex_hash: [u8; 32],
        source_row_count: u64,
        source_timestamp_start_ms: i64,
        source_timestamp_end_ms: i64,
        selected_row_start: u64,
        selected_row_end: u64,
        requested_from_ms: i64,
        requested_to_ms: i64,
        selected_timestamp_start_ms: i64,
        selected_timestamp_end_ms: i64,
        volume_encoding: u8,
    ) -> Result<Self> {
        let value = Self {
            source_identity,
            source_manifest_schema_id: source_manifest_schema_id.into(),
            source_manifest_hash,
            source_generation: source_generation.into(),
            source_vortex_hash,
            source_row_count,
            source_timestamp_start_ms,
            source_timestamp_end_ms,
            selected_row_start,
            selected_row_end,
            requested_from_ms,
            requested_to_ms,
            selected_timestamp_start_ms,
            selected_timestamp_end_ms,
            volume_encoding,
        };
        value.validate()?;
        Ok(value)
    }

    fn source_identity(&self) -> &neoethos_data::CanonicalDatasetIdentity {
        &self.source_identity
    }

    #[cfg(test)]
    const fn selected_row_range(&self) -> std::ops::Range<u64> {
        self.selected_row_start..self.selected_row_end
    }

    #[cfg(test)]
    const fn requested_range_ms(&self) -> (i64, i64) {
        (self.requested_from_ms, self.requested_to_ms)
    }

    #[cfg(test)]
    const fn selected_timestamp_range_ms(&self) -> (i64, i64) {
        (
            self.selected_timestamp_start_ms,
            self.selected_timestamp_end_ms,
        )
    }

    const fn output_row_count(&self) -> u64 {
        self.selected_row_end - self.selected_row_start
    }

    fn validate(&self) -> Result<()> {
        validate_slice_text("source manifest schema", &self.source_manifest_schema_id)?;
        validate_slice_opaque_component("source generation", &self.source_generation)?;
        if self.source_row_count == 0 {
            anyhow::bail!("slice provenance cannot bind an empty source generation");
        }
        if self.source_timestamp_start_ms > self.source_timestamp_end_ms {
            anyhow::bail!("slice provenance source timestamp range is descending");
        }
        if self.selected_row_start >= self.selected_row_end
            || self.selected_row_end > self.source_row_count
        {
            anyhow::bail!("slice provenance selected row range is empty or outside the source");
        }
        if self.requested_from_ms >= self.requested_to_ms {
            anyhow::bail!("slice provenance requested range is empty or descending");
        }
        if self.selected_timestamp_start_ms > self.selected_timestamp_end_ms
            || self.selected_timestamp_start_ms < self.source_timestamp_start_ms
            || self.selected_timestamp_end_ms > self.source_timestamp_end_ms
            || self.selected_timestamp_start_ms < self.requested_from_ms
            || self.selected_timestamp_end_ms >= self.requested_to_ms
        {
            anyhow::bail!(
                "slice provenance selected timestamp range is outside the source or half-open request"
            );
        }
        if !matches!(
            self.volume_encoding,
            SLICE_VOLUME_ABSENT | SLICE_VOLUME_FLOAT64
        ) {
            anyhow::bail!("unsupported slice provenance volume encoding");
        }
        Ok(())
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(384);
        bytes.extend_from_slice(SLICE_PROVENANCE_DOMAIN);
        bytes.extend_from_slice(&SLICE_PROVENANCE_VERSION.to_be_bytes());
        push_slice_bytes(&mut bytes, &self.source_identity.canonical_bytes());
        push_slice_bytes(&mut bytes, self.source_manifest_schema_id.as_bytes());
        bytes.extend_from_slice(&self.source_manifest_hash);
        push_slice_bytes(&mut bytes, self.source_generation.as_bytes());
        bytes.extend_from_slice(&self.source_vortex_hash);
        bytes.extend_from_slice(&self.source_row_count.to_be_bytes());
        bytes.extend_from_slice(&self.source_timestamp_start_ms.to_be_bytes());
        bytes.extend_from_slice(&self.source_timestamp_end_ms.to_be_bytes());
        bytes.extend_from_slice(&self.selected_row_start.to_be_bytes());
        bytes.extend_from_slice(&self.selected_row_end.to_be_bytes());
        bytes.extend_from_slice(&self.requested_from_ms.to_be_bytes());
        bytes.extend_from_slice(&self.requested_to_ms.to_be_bytes());
        bytes.extend_from_slice(&self.selected_timestamp_start_ms.to_be_bytes());
        bytes.extend_from_slice(&self.selected_timestamp_end_ms.to_be_bytes());
        bytes.push(SLICE_SELECTION_HALF_OPEN);
        bytes.push(self.volume_encoding);
        bytes
    }

    fn to_envelope(
        &self,
    ) -> Result<neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1> {
        self.validate()?;
        neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1::new(
            Self::SCHEMA_ID,
            self.canonical_bytes(),
        )
    }

    fn from_envelope(
        envelope: &neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1,
    ) -> Result<Self> {
        envelope.validate()?;
        if envelope.schema_id() != Self::SCHEMA_ID {
            anyhow::bail!(
                "slice provenance schema mismatch: expected {}, got {}",
                Self::SCHEMA_ID,
                envelope.schema_id()
            );
        }
        let mut cursor = SliceProvenanceCursor::new(envelope.canonical_payload());
        cursor.require_exact(SLICE_PROVENANCE_DOMAIN, "domain")?;
        if cursor.read_u16("version")? != SLICE_PROVENANCE_VERSION {
            anyhow::bail!("unsupported slice provenance version");
        }
        let source_identity = neoethos_data::CanonicalDatasetIdentity::from_canonical_bytes(
            cursor.read_bytes("source identity")?,
        )
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        let value = Self {
            source_identity,
            source_manifest_schema_id: cursor.read_text("source manifest schema")?,
            source_manifest_hash: cursor.read_array("source manifest hash")?,
            source_generation: cursor.read_text("source generation")?,
            source_vortex_hash: cursor.read_array("source Vortex hash")?,
            source_row_count: cursor.read_u64("source row count")?,
            source_timestamp_start_ms: cursor.read_i64("source timestamp start")?,
            source_timestamp_end_ms: cursor.read_i64("source timestamp end")?,
            selected_row_start: cursor.read_u64("selected row start")?,
            selected_row_end: cursor.read_u64("selected row end")?,
            requested_from_ms: cursor.read_i64("requested from")?,
            requested_to_ms: cursor.read_i64("requested to")?,
            selected_timestamp_start_ms: cursor.read_i64("selected timestamp start")?,
            selected_timestamp_end_ms: cursor.read_i64("selected timestamp end")?,
            volume_encoding: 0,
        };
        cursor.require_tag(SLICE_SELECTION_HALF_OPEN, "selection semantics")?;
        let volume_encoding = cursor.read_u8("volume encoding")?;
        let value = Self {
            volume_encoding,
            ..value
        };
        if !cursor.is_empty() {
            anyhow::bail!("slice provenance has trailing bytes");
        }
        value.validate()?;
        if value.canonical_bytes() != envelope.canonical_payload() {
            anyhow::bail!("slice provenance is not canonically encoded");
        }
        Ok(value)
    }
}

#[derive(Debug)]
struct CanonicalDatasetSliceOutcome {
    publication: neoethos_data::core::dataset_manifest::PublishResult,
    source_rows: usize,
    kept_rows: usize,
    first_ms: i64,
    last_ms: i64,
}

fn publish_canonical_dataset_slice(
    source_root: &std::path::Path,
    output_root: &std::path::Path,
    identity: &neoethos_data::CanonicalDatasetIdentity,
    from_ms: i64,
    to_ms: i64,
) -> Result<CanonicalDatasetSliceOutcome> {
    if from_ms >= to_ms {
        anyhow::bail!("slice-dataset range is empty or descending");
    }
    let source =
        neoethos_data::core::canonical_ohlcv::load_canonical_timeframe(source_root, identity)
            .with_context(|| {
                format!(
                    "fully verify exact slice source {}",
                    identity.to_path_component()
                )
            })?;
    let source_rows = source.len();
    let timestamps = source
        .ohlcv()
        .timestamp
        .as_deref()
        .context("verified canonical slice source has no timestamp_ms")?;
    let selected_row_start = timestamps.partition_point(|timestamp| *timestamp < from_ms);
    let selected_row_end = timestamps.partition_point(|timestamp| *timestamp < to_ms);
    let (slice, span) = neoethos_data::slice_ohlcv_by_date_range_ms(source.ohlcv(), from_ms, to_ms)
        .map_err(|error| anyhow::anyhow!(error))?;
    let kept_rows = slice.len();
    let (first_ms, last_ms) = span.context("slice-dataset selected zero source rows")?;
    anyhow::ensure!(
        kept_rows == selected_row_end.saturating_sub(selected_row_start),
        "slice row selection disagrees with the shared date-range slicer"
    );

    let binding = source
        .artifact()
        .source_binding("slice-dataset-source")
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let volume_encoding = if slice.volume.is_some() {
        SLICE_VOLUME_FLOAT64
    } else {
        SLICE_VOLUME_ABSENT
    };
    let provenance = SliceDatasetProvenanceV1::new(
        binding.dataset_identity().clone(),
        binding.manifest_schema_id(),
        *binding.manifest_hash(),
        binding.generation_id(),
        *binding.vortex_hash(),
        source.artifact().row_count(),
        source.artifact().timestamp_start_ms(),
        source.artifact().timestamp_end_ms(),
        u64::try_from(selected_row_start).context("slice row start exceeds u64")?,
        u64::try_from(selected_row_end).context("slice row end exceeds u64")?,
        from_ms,
        to_ms,
        first_ms,
        last_ms,
        volume_encoding,
    )?;
    let envelope = provenance.to_envelope()?;
    let volume = slice
        .volume
        .as_deref()
        .map_or(neoethos_data::CanonicalVolumeRef::Absent, |values| {
            neoethos_data::CanonicalVolumeRef::Float64(values)
        });
    let publication = neoethos_data::publish_canonical_ohlcv_generation(
        neoethos_data::CanonicalOhlcvPublishRequest {
            configured_root: output_root,
            identity,
            expected_generation: None,
            provenance: &envelope,
            ohlcv: &slice,
            volume,
            rows_per_chunk: SLICE_ROWS_PER_VORTEX_CHUNK,
        },
    )
    .with_context(|| {
        format!(
            "atomically publish canonical slice {} into {}",
            identity.to_path_component(),
            output_root.display()
        )
    })?;
    let reopened = SliceDatasetProvenanceV1::from_envelope(publication.manifest().provenance())?;
    anyhow::ensure!(
        publication.manifest().identity() == identity
            && reopened.source_identity() == identity
            && reopened == provenance
            && publication.row_count() == provenance.output_row_count(),
        "reopened slice manifest/provenance disagrees with the exact publication request"
    );

    Ok(CanonicalDatasetSliceOutcome {
        publication,
        source_rows,
        kept_rows,
        first_ms,
        last_ms,
    })
}

fn cmd_slice_dataset(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());

    let out_root = parse_flag(args, "--out-root")
        .or_else(|| parse_flag(args, "--out"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "slice-dataset requires --out-root <DST> (the new root dir to write the slice into)"
            )
        })?;

    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    if symbol.is_empty() {
        anyhow::bail!(
            "slice-dataset: no --symbol supplied and config.yaml could not provide one — \
             pass --symbol explicitly (e.g. --symbol EURUSD)"
        );
    }
    // `--base` is the primary name (matches discover/prepare); `--timeframe`
    // is accepted as an alias for parity with `load`.
    let base = parse_flag(args, "--base")
        .or_else(|| parse_flag(args, "--timeframe"))
        .unwrap_or_else(|| default_base_tf(settings.as_ref()));
    if base.is_empty() {
        anyhow::bail!(
            "slice-dataset: no --base supplied and config.yaml could not provide one — \
             pass --base explicitly (e.g. --base M1)"
        );
    }

    let from_date = parse_flag(args, "--from-date").ok_or_else(|| {
        anyhow::anyhow!(
            "slice-dataset requires --from-date YYYY-MM-DD (inclusive lower bound, UTC)"
        )
    })?;
    let to_date = parse_flag(args, "--to-date").ok_or_else(|| {
        anyhow::anyhow!("slice-dataset requires --to-date YYYY-MM-DD (exclusive upper bound, UTC)")
    })?;

    let from_ms = parse_ymd_to_epoch_ms(&from_date, "--from-date")?;
    let to_ms = parse_ymd_to_epoch_ms(&to_date, "--to-date")?;
    if to_ms <= from_ms {
        anyhow::bail!(
            "slice-dataset: --to-date ({to_date}) must be strictly after --from-date ({from_date}) \
             (the range is half-open [from, to))"
        );
    }

    let identities = inventory_canonical_identities(&root, &symbol)?;
    let identity = select_exact_runtime_identity(&identities, args, &symbol, &base)
        .context("slice-dataset exact source selection")?;
    let outcome = publish_canonical_dataset_slice(
        std::path::Path::new(&root),
        std::path::Path::new(&out_root),
        &identity,
        from_ms,
        to_ms,
    )
    .map_err(|error| anyhow::anyhow!("slice-dataset: {error:#}"))?;

    println!(
        "slice-dataset {symbol} {base}: [{from_date}, {to_date})  source rows={}  kept rows={}",
        outcome.source_rows, outcome.kept_rows
    );
    println!(
        "  kept span: {} .. {}",
        format_epoch_ms_date(outcome.first_ms),
        format_epoch_ms_date(outcome.last_ms)
    );
    println!("  dataset identity: {}", identity.to_path_component());
    println!("  generation:       {}", outcome.publication.generation());
    println!(
        "  durable commit:   {}",
        outcome.publication.durable_commit_id()
    );
    println!(
        "  canonical Vortex: {}",
        outcome.publication.manifest().generation_path().display()
    );
    Ok(())
}

fn validate_slice_text(field: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() || value.len() > 4 * 1024 || value.chars().any(char::is_control) {
        anyhow::bail!("slice provenance {field} is empty, too long, or contains control data");
    }
    Ok(())
}

fn validate_slice_opaque_component(field: &str, value: &str) -> Result<()> {
    validate_slice_text(field, value)?;
    if matches!(value, "." | "..")
        || value.contains('/')
        || value.contains('\\')
        || value.contains(':')
    {
        anyhow::bail!("slice provenance {field} is not one opaque path component");
    }
    Ok(())
}

fn push_slice_bytes(target: &mut Vec<u8>, value: &[u8]) {
    target.extend_from_slice(
        &u32::try_from(value.len())
            .expect("validated slice provenance field length fits u32")
            .to_be_bytes(),
    );
    target.extend_from_slice(value);
}

struct SliceProvenanceCursor<'a> {
    remaining: &'a [u8],
}

impl<'a> SliceProvenanceCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }

    const fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }

    fn take(&mut self, length: usize, field: &str) -> Result<&'a [u8]> {
        if self.remaining.len() < length {
            anyhow::bail!("slice provenance is truncated at {field}");
        }
        let (value, rest) = self.remaining.split_at(length);
        self.remaining = rest;
        Ok(value)
    }

    fn require_exact(&mut self, expected: &[u8], field: &str) -> Result<()> {
        if self.take(expected.len(), field)? != expected {
            anyhow::bail!("invalid slice provenance {field}");
        }
        Ok(())
    }

    fn require_tag(&mut self, expected: u8, field: &str) -> Result<()> {
        let actual = self.read_u8(field)?;
        if actual != expected {
            anyhow::bail!("unsupported slice provenance {field} {actual}");
        }
        Ok(())
    }

    fn read_u8(&mut self, field: &str) -> Result<u8> {
        Ok(self.take(1, field)?[0])
    }

    fn read_u16(&mut self, field: &str) -> Result<u16> {
        Ok(u16::from_be_bytes(self.read_array(field)?))
    }

    fn read_u64(&mut self, field: &str) -> Result<u64> {
        Ok(u64::from_be_bytes(self.read_array(field)?))
    }

    fn read_i64(&mut self, field: &str) -> Result<i64> {
        Ok(i64::from_be_bytes(self.read_array(field)?))
    }

    fn read_bytes(&mut self, field: &str) -> Result<&'a [u8]> {
        let length = usize::try_from(u32::from_be_bytes(self.read_array(field)?))
            .context("slice provenance field length does not fit usize")?;
        self.take(length, field)
    }

    fn read_text(&mut self, field: &str) -> Result<String> {
        let bytes = self.read_bytes(field)?;
        let value = std::str::from_utf8(bytes)
            .with_context(|| format!("slice provenance {field} is not UTF-8"))?;
        validate_slice_text(field, value)?;
        Ok(value.to_owned())
    }

    fn read_array<const N: usize>(&mut self, field: &str) -> Result<[u8; N]> {
        self.take(N, field)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid slice provenance {field}"))
    }
}

fn inventory_canonical_identities(
    root: impl AsRef<std::path::Path>,
    symbol: &str,
) -> Result<Vec<neoethos_data::CanonicalDatasetIdentity>> {
    let inventory =
        neoethos_data::DatasetDiscovery::scan_metadata(root.as_ref()).with_context(|| {
            format!(
                "inventory canonical manifests under {}",
                root.as_ref().display()
            )
        })?;
    for entry in inventory.entries.iter().filter(|entry| {
        entry
            .symbol
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case(symbol))
    }) {
        print_dataset_inventory_entry(entry)?;
    }
    print_dataset_inventory_rejections(&inventory);
    let mut identities = Vec::new();
    for entry in inventory.entries.iter().filter(|entry| {
        entry
            .symbol
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case(symbol))
    }) {
        anyhow::ensure!(
            entry.verification == neoethos_data::DataVerificationStatus::ManifestOnly,
            "metadata identity inventory unexpectedly authorized a generation as fully verified"
        );
        let identity =
            neoethos_data::CanonicalDatasetIdentity::from_path_component(&entry.dataset_identity)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        anyhow::ensure!(
            identity.symbol_name().eq_ignore_ascii_case(symbol)
                && entry.timeframe.as_deref() == Some(identity.timeframe().as_str()),
            "canonical identity inventory metadata disagrees with its reversible identity"
        );
        identities.push(identity);
    }
    identities.sort_by(|left, right| {
        left.timeframe()
            .ctrader_protocol_code()
            .cmp(&right.timeframe().ctrader_protocol_code())
            .then_with(|| left.to_path_component().cmp(&right.to_path_component()))
    });
    Ok(identities)
}

fn verified_canonical_identities(
    inventory: &neoethos_data::DatasetDiscovery,
    symbol: &str,
) -> Result<Vec<neoethos_data::CanonicalDatasetIdentity>> {
    let mut identities = Vec::new();
    for entry in inventory.entries.iter().filter(|entry| {
        entry
            .symbol
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case(symbol))
    }) {
        anyhow::ensure!(
            entry.verification == neoethos_data::DataVerificationStatus::GenerationVerified,
            "schedule identity={} generation={} is not fully verified",
            entry.dataset_identity,
            entry.generation
        );
        let identity =
            neoethos_data::CanonicalDatasetIdentity::from_path_component(&entry.dataset_identity)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        anyhow::ensure!(
            identity.symbol_name().eq_ignore_ascii_case(symbol)
                && entry.timeframe.as_deref() == Some(identity.timeframe().as_str()),
            "verified schedule inventory metadata disagrees with its reversible identity"
        );
        identities.push(identity);
    }
    identities.sort_by(|left, right| {
        left.timeframe()
            .ctrader_protocol_code()
            .cmp(&right.timeframe().ctrader_protocol_code())
            .then_with(|| left.to_path_component().cmp(&right.to_path_component()))
    });
    Ok(identities)
}

fn unique_canonical_identity(
    identities: &[neoethos_data::CanonicalDatasetIdentity],
    symbol: &str,
    timeframe: neoethos_data::CanonicalTimeframe,
) -> Result<Option<neoethos_data::CanonicalDatasetIdentity>> {
    let matching = identities
        .iter()
        .filter(|identity| {
            identity.symbol_name().eq_ignore_ascii_case(symbol) && identity.timeframe() == timeframe
        })
        .collect::<Vec<_>>();
    anyhow::ensure!(
        matching.len() <= 1,
        "expected exactly one canonical dataset identity for {symbol} {timeframe}, found {}; select an exact source/account identity",
        matching.len()
    );
    Ok(matching.first().map(|identity| (*identity).clone()))
}

fn select_exact_runtime_identity(
    identities: &[neoethos_data::CanonicalDatasetIdentity],
    args: &[String],
    symbol: &str,
    timeframe_label: &str,
) -> Result<neoethos_data::CanonicalDatasetIdentity> {
    let timeframe = timeframe_label
        .parse::<neoethos_data::CanonicalTimeframe>()
        .with_context(|| format!("unsupported canonical timeframe {timeframe_label}"))?;
    if let Some(encoded) = parse_flag(args, "--dataset-identity") {
        let identity = neoethos_data::CanonicalDatasetIdentity::from_path_component(&encoded)
            .map_err(|error| anyhow::anyhow!(error.to_string()))
            .with_context(|| format!("decode --dataset-identity {encoded}"))?;
        anyhow::ensure!(
            identity.symbol_name().eq_ignore_ascii_case(symbol),
            "--dataset-identity belongs to {}, but --symbol/config selected {symbol}",
            identity.symbol_name()
        );
        anyhow::ensure!(
            identity.timeframe() == timeframe,
            "--dataset-identity is {}, but --timeframe/--base selected {timeframe}",
            identity.timeframe()
        );
        anyhow::ensure!(
            identities.contains(&identity),
            "--dataset-identity {encoded} is not a current canonical manifest under the selected data root"
        );
        return Ok(identity);
    }

    unique_canonical_identity(identities, symbol, timeframe)?.with_context(|| {
        format!(
            "expected exactly one canonical dataset identity for {symbol} {timeframe}, found 0; pass --dataset-identity <d1-...>"
        )
    })
}

#[derive(Debug)]
struct DirectTimeframeSelection {
    base_identity: neoethos_data::CanonicalDatasetIdentity,
    required: Vec<neoethos_data::CanonicalTimeframe>,
    identities: Vec<neoethos_data::CanonicalDatasetIdentity>,
}

#[cfg(feature = "gpu-nvidia")]
#[derive(Debug)]
struct PinnedDirectTimeframeSelection {
    series: neoethos_data::CanonicalDatasetSeriesReceiptV1,
    base_row_count: usize,
    pinned_series: Option<neoethos_data::PinnedCanonicalSeriesV1>,
}

#[cfg(feature = "gpu-nvidia")]
impl PinnedDirectTimeframeSelection {
    fn take_or_repin(
        &mut self,
        root: &std::path::Path,
    ) -> Result<neoethos_data::PinnedCanonicalSeriesV1> {
        if let Some(pinned) = self.pinned_series.take() {
            return Ok(pinned);
        }
        neoethos_data::pin_exact_canonical_series_v1(root, self.series.clone())
    }
}

#[cfg(feature = "gpu-nvidia")]
fn pin_direct_timeframe_selection(
    root: impl AsRef<std::path::Path>,
    selection: &DirectTimeframeSelection,
) -> Result<PinnedDirectTimeframeSelection> {
    let root = root.as_ref();
    let mut selected = Vec::with_capacity(selection.identities.len());
    let mut base_row_count = None;
    for identity in &selection.identities {
        let manifest =
            neoethos_data::core::dataset_manifest::read_current_manifest_metadata(root, identity)
                .with_context(|| {
                format!(
                    "pin exact canonical generation metadata for {}",
                    identity.to_path_component()
                )
            })?;
        if identity == &selection.base_identity {
            base_row_count = Some(
                usize::try_from(manifest.row_count())
                    .context("canonical base row count does not fit this process")?,
            );
        }
        selected.push(neoethos_data::SelectedDatasetGenerationV1::from_manifest(
            &manifest,
        )?);
    }
    let anchor = selected
        .iter()
        .find(|receipt| receipt.identity() == &selection.base_identity)
        .cloned()
        .context("pinned direct timeframe selection lost its exact base generation")?;
    let series = neoethos_data::CanonicalDatasetSeriesReceiptV1::new(anchor, selected)?;
    let pinned_series = neoethos_data::pin_exact_canonical_series_v1(root, series.clone())?;
    neoethos_search::fx_rates::set_store_selection(
        root.to_path_buf(),
        selection.base_identity.clone(),
    )
    .map_err(anyhow::Error::new)
    .context("install exact FX source/account selection")?;
    Ok(PinnedDirectTimeframeSelection {
        series,
        base_row_count: base_row_count.context("pinned selection has no base row count")?,
        pinned_series: Some(pinned_series),
    })
}

fn select_runtime_timeframe_identities(
    identities: &[neoethos_data::CanonicalDatasetIdentity],
    symbol: &str,
    base: &str,
    requested_higher: &[String],
) -> Result<DirectTimeframeSelection> {
    let base_timeframe = base
        .parse::<neoethos_data::CanonicalTimeframe>()
        .with_context(|| format!("unsupported canonical base timeframe {base}"))?;
    let base_identity = unique_canonical_identity(identities, symbol, base_timeframe)?
        .with_context(|| {
            format!("expected exactly one canonical dataset identity for {symbol} {base}, found 0")
        })?;
    select_runtime_timeframe_identities_for_base(identities, &base_identity, requested_higher)
}

fn select_runtime_timeframe_identities_for_base(
    identities: &[neoethos_data::CanonicalDatasetIdentity],
    base_identity: &neoethos_data::CanonicalDatasetIdentity,
    requested_higher: &[String],
) -> Result<DirectTimeframeSelection> {
    let symbol = base_identity.symbol_name();
    let base_timeframe = base_identity.timeframe();
    anyhow::ensure!(
        identities.contains(base_identity),
        "selected base identity {} is not present in the current canonical inventory",
        base_identity.to_path_component()
    );
    let mut required = Vec::with_capacity(1 + requested_higher.len());
    required.push(base_timeframe);
    for label in requested_higher {
        let label = label.trim();
        if label.is_empty() {
            continue;
        }
        let timeframe = label
            .parse::<neoethos_data::CanonicalTimeframe>()
            .with_context(|| format!("unsupported canonical higher timeframe {label}"))?;
        if !required.contains(&timeframe) {
            required.push(timeframe);
        }
    }
    required.sort_by_key(|timeframe| timeframe.ctrader_protocol_code());

    let mut selected = vec![base_identity.clone()];
    for timeframe in &required {
        if *timeframe == base_timeframe {
            continue;
        }
        let matching = identities
            .iter()
            .filter(|identity| {
                identity.symbol_name().eq_ignore_ascii_case(symbol)
                    && identity.timeframe() == *timeframe
                    && identity.scope() == base_identity.scope()
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(
            matching.len() <= 1,
            "multiple canonical {symbol} {timeframe} identities match the exact base source/account scope"
        );
        let identity = matching.first().with_context(|| {
            format!(
                "missing direct canonical timeframe {symbol} {timeframe} in the exact base source/account scope; import/download required"
            )
        })?;
        selected.push((**identity).clone());
    }
    Ok(DirectTimeframeSelection {
        base_identity: base_identity.clone(),
        required,
        identities: selected,
    })
}

fn load_exact_runtime_timeframe(
    root: impl AsRef<std::path::Path>,
    identity: &neoethos_data::CanonicalDatasetIdentity,
) -> Result<neoethos_data::CanonicalOhlcvFrame> {
    let loaded =
        neoethos_data::core::canonical_ohlcv::load_canonical_timeframe(root.as_ref(), identity)
            .with_context(|| {
                format!(
                    "fully verify exact canonical dataset {}",
                    identity.to_path_component()
                )
            })?;
    Ok(loaded)
}

fn load_exact_symbol_dataset(
    root: impl AsRef<std::path::Path>,
    symbol: &str,
    identities: &[neoethos_data::CanonicalDatasetIdentity],
) -> Result<neoethos_data::SymbolDataset> {
    anyhow::ensure!(
        !identities.is_empty(),
        "no exact canonical dataset identities selected for {symbol}"
    );
    let mut frames = std::collections::HashMap::new();
    let mut source_artifacts = std::collections::HashMap::new();
    for identity in identities {
        anyhow::ensure!(
            identity.symbol_name().eq_ignore_ascii_case(symbol),
            "selected canonical dataset identity belongs to {}, not {symbol}",
            identity.symbol_name()
        );
        let timeframe = identity.timeframe().as_str().to_owned();
        anyhow::ensure!(
            !frames.contains_key(&timeframe),
            "selected canonical runtime identities duplicate {symbol} {timeframe}"
        );
        let loaded =
            neoethos_data::core::canonical_ohlcv::load_canonical_timeframe(root.as_ref(), identity)
                .with_context(|| {
                    format!(
                        "fully verify selected canonical dataset {}",
                        identity.to_path_component()
                    )
                })?;
        frames.insert(timeframe.clone(), loaded.ohlcv().clone());
        source_artifacts.insert(timeframe, loaded.artifact().clone());
    }
    Ok(neoethos_data::SymbolDataset {
        symbol: symbol.to_owned(),
        frames,
        source_artifacts,
    })
}

fn load_required_direct_symbol_dataset(
    root: impl AsRef<std::path::Path>,
    symbol: &str,
    selection: &DirectTimeframeSelection,
) -> Result<neoethos_data::SymbolDataset> {
    let root = root.as_ref();
    let dataset = load_exact_symbol_dataset(root, symbol, &selection.identities)?;
    neoethos_data::require_direct_timeframes(
        &dataset,
        &selection.base_identity,
        &selection.required,
    )
    .with_context(|| {
        format!(
            "direct canonical timeframe verification failed for {symbol} {}; import/download required",
            selection.base_identity.timeframe()
        )
    })?;
    neoethos_search::fx_rates::set_store_selection(
        root.to_path_buf(),
        selection.base_identity.clone(),
    )
    .map_err(anyhow::Error::new)
    .with_context(|| {
        format!(
            "install exact FX source/account selection {}",
            selection.base_identity.to_path_component()
        )
    })?;
    Ok(dataset)
}

/// Parse a `YYYY-MM-DD` date as midnight UTC and return epoch milliseconds.
/// Fails loud with the offending flag name when the string isn't a valid date.
fn parse_ymd_to_epoch_ms(date: &str, flag: &str) -> Result<i64> {
    let naive = chrono::NaiveDate::parse_from_str(date.trim(), "%Y-%m-%d").map_err(|err| {
        anyhow::anyhow!("slice-dataset: {flag} '{date}' is not a valid YYYY-MM-DD date: {err}")
    })?;
    let dt = naive.and_hms_opt(0, 0, 0).ok_or_else(|| {
        anyhow::anyhow!("slice-dataset: {flag} '{date}' could not be set to midnight")
    })?;
    Ok(dt.and_utc().timestamp_millis())
}

/// Render an epoch-ms timestamp as a `YYYY-MM-DD` UTC date for the kept-span
/// summary line.
fn format_epoch_ms_date(ms: i64) -> String {
    chrono::DateTime::from_timestamp_millis(ms)
        .map(|dt| dt.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| format!("ms:{ms}"))
}

fn cmd_symbols(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let report = neoethos_data::DatasetDiscovery::scan_metadata(&root)?;
    println!("Canonical dataset identities ({}):", report.entries.len());
    for entry in &report.entries {
        print_dataset_inventory_entry(entry)?;
    }
    print_dataset_inventory_rejections(&report);
    if report.entries.is_empty() {
        println!("  NO CANONICAL DATASET IDENTITIES FOUND");
    }
    Ok(())
}

fn cmd_timeframes(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let report = neoethos_data::DatasetDiscovery::scan_metadata(&root)?;
    let entries = report
        .entries
        .iter()
        .filter(|entry| {
            entry
                .symbol
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case(&symbol))
        })
        .collect::<Vec<_>>();
    println!(
        "Canonical dataset identities for {symbol} ({}):",
        entries.len()
    );
    for entry in entries {
        print_dataset_inventory_entry(entry)?;
    }
    print_dataset_inventory_rejections(&report);
    if !report.entries.iter().any(|entry| {
        entry
            .symbol
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case(&symbol))
    }) {
        println!("  NO CANONICAL DATASET IDENTITIES FOUND FOR {symbol}");
    }
    Ok(())
}

fn print_dataset_inventory_entry(entry: &neoethos_data::DataFileEntry) -> Result<()> {
    let symbol = entry
        .symbol
        .as_deref()
        .context("canonical inventory entry has no symbol")?;
    let timeframe = entry
        .timeframe
        .as_deref()
        .context("canonical inventory entry has no timeframe")?;
    println!(
        "  {symbol} {timeframe} identity={} generation={} manifest_binding_sha256={} verification={:?} bytes={} path={}",
        entry.dataset_identity,
        entry.generation,
        entry.manifest_binding_sha256,
        entry.verification,
        entry.size_bytes,
        entry.path.display()
    );
    Ok(())
}

fn print_dataset_inventory_rejections(report: &neoethos_data::DatasetDiscovery) {
    for skipped in &report.skipped {
        println!(
            "  REJECTED path={} category={} detail={:?}",
            skipped.path.display(),
            skipped.reason.category(),
            skipped.reason
        );
    }
}

fn metadata_inventory_symbols(root: impl AsRef<std::path::Path>) -> Result<Vec<String>> {
    let report =
        neoethos_data::DatasetDiscovery::scan_metadata(root.as_ref()).with_context(|| {
            format!(
                "inventory canonical dataset identities under {}",
                root.as_ref().display()
            )
        })?;
    for entry in &report.entries {
        print_dataset_inventory_entry(entry)?;
    }
    print_dataset_inventory_rejections(&report);
    let symbols = report.symbols();
    anyhow::ensure!(
        !symbols.is_empty(),
        "metadata inventory found no canonical dataset identities under {}",
        root.as_ref().display()
    );
    Ok(symbols)
}

fn cmd_features(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let timeframe =
        parse_flag(args, "--timeframe").unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let identities = inventory_canonical_identities(&root, &symbol)?;
    let identity = select_exact_runtime_identity(&identities, args, &symbol, &timeframe)?;
    let canonical = load_exact_runtime_timeframe(&root, &identity)?;
    let features = neoethos_data::compute_hpc_features(&canonical)?;
    println!(
        "Features {} {} -> rows={}, cols={}",
        symbol,
        timeframe,
        features.n_samples(),
        features.n_features()
    );
    Ok(())
}

fn cmd_prepare(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let higher = parse_flag(args, "--higher")
        .unwrap_or_else(|| default_higher_tfs_csv(settings.as_ref(), &base));
    let higher_list: Vec<String> = higher
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().to_string())
        .collect();
    let higher_refs: Vec<&str> = higher_list.iter().map(|s| s.as_str()).collect();
    let identities = inventory_canonical_identities(&root, &symbol)?;
    let base_identity = select_exact_runtime_identity(&identities, args, &symbol, &base)?;
    let selection =
        select_runtime_timeframe_identities_for_base(&identities, &base_identity, &higher_list)?;
    let dataset = load_required_direct_symbol_dataset(&root, &symbol, &selection)?;
    let features = neoethos_data::prepare_multitimeframe_features(&dataset, &base, &higher_refs)?;
    println!(
        "Prepared {} base={} rows={} cols={}",
        symbol,
        base,
        features.n_samples(),
        features.n_features()
    );
    Ok(())
}

/// `discovery-promote-weekly --portfolio PATH [--cache-dir ...]`
/// — the weekly-refresh promotion step of the search-memory feature.
///
/// The selected strict v3 live portfolio is the sole authority for receipt,
/// search-config identity, symbol and timeframe. Loads that exact ledger and
/// compares its recorded genes with the portfolio under the **additive** policy.
///
/// SCOPE NOTE (deferred — see report): the ledger records gene *signatures*
/// (hash + indicator names + SMC flags + fitness), not the full `Gene`
/// (indices/weights). The live portfolio stores full genes. So we can detect
/// which ledger genes are NEW relative to the live portfolio and carry the
/// existing full genes forward, but we cannot synthesize full genes for ledger-
/// only records here — adding them as live genes would require the discovery
/// run to also persist full genes per ledger entry (a future enhancement). We
/// therefore write a merged **growth-summary JSON** next to the portfolio and
/// report the new-vs-carried breakdown.
fn cmd_discovery_promote_weekly(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let ledger_cfg = settings
        .as_ref()
        .map(|s| s.models.discovery_ledger.clone())
        .unwrap_or_default();

    let portfolio_path = parse_flag(args, "--portfolio").ok_or_else(|| {
        anyhow::anyhow!(
            "discovery-promote-weekly requires --portfolio PATH naming a strict v3 live portfolio"
        )
    })?;
    let artifact = neoethos_search::load_live_portfolio_json(&portfolio_path)
        .with_context(|| format!("load strict v3 live portfolio {portfolio_path}"))?;
    let search_receipt = artifact.search_scope.receipt().clone();
    let config_hash = artifact.search_config_hash.clone();
    let anchor = search_receipt.validate()?;
    let symbol = anchor.symbol_name().to_owned();
    let tf = anchor.timeframe().as_str().to_owned();
    let cache_dir = parse_flag(args, "--cache-dir").unwrap_or(ledger_cfg.cache_dir);

    let exact_ledger_path =
        neoethos_search::ledger_path(&cache_dir, &search_receipt, &config_hash)?;
    let ledger = neoethos_search::load_prior_ledger(
        &cache_dir,
        &symbol,
        &tf,
        &search_receipt,
        &config_hash,
    )?
    .ok_or_else(|| {
        anyhow::anyhow!(
            "no receipt-bound discovery ledger found at {} — run this exact dataset/feature/config search first",
            exact_ledger_path.display()
        )
    })?;

    let mut existing_hashes: std::collections::HashSet<String> = std::collections::HashSet::new();
    for gene in &artifact.genes {
        existing_hashes.insert(neoethos_search::genetic::gene_signature_hash(gene).to_string());
    }
    let existing_count = artifact.genes.len();

    let new_genes: Vec<&neoethos_search::GeneRecord> = ledger
        .portfolio
        .iter()
        .chain(ledger.archive.iter())
        .filter(|rec| !existing_hashes.contains(&rec.hash))
        .collect();
    // Distinct new hashes (a hash could appear in both portfolio + archive).
    let new_hashes: std::collections::HashSet<&String> =
        new_genes.iter().map(|r| &r.hash).collect();
    let added = new_hashes.len();
    let carried = existing_count;
    let total = carried + added;

    // Write a merged growth-summary JSON next to the portfolio so the weekly run
    // leaves an auditable record of what grew.
    #[derive(serde::Serialize)]
    struct PromotionSummary<'a> {
        symbol: &'a str,
        tf: &'a str,
        config_hash: &'a str,
        policy: &'a str,
        carried: usize,
        added: usize,
        total: usize,
        new_genes: Vec<&'a neoethos_search::GeneRecord>,
    }
    let summary_path = exact_ledger_path
        .parent()
        .expect("receipt-bound ledger path has a parent")
        .join("weekly_promotion.v2.json");
    let summary = PromotionSummary {
        symbol: &symbol,
        tf: &tf,
        config_hash: &config_hash,
        policy: &ledger_cfg.promotion_policy,
        carried,
        added,
        total,
        new_genes: new_genes.clone(),
    };
    let summary = neoethos_search::CanonicalSearchArtifactEnvelopeV2::new(
        "neoethos.search-weekly-promotion.v2",
        artifact.search_scope.clone(),
        config_hash.clone(),
        summary,
    )?;
    neoethos_core::storage::json::write_json_atomic(&summary_path, &summary)
        .with_context(|| format!("write weekly-promotion summary {}", summary_path.display()))?;

    println!(
        "discovery-promote-weekly {} {} (policy={}): added {} new, carried {}, total {}",
        symbol, tf, ledger_cfg.promotion_policy, added, carried, total
    );
    println!("  ledger: {}", exact_ledger_path.display());
    println!("  summary written: {}", summary_path.display());
    if added > 0 {
        println!("  new strategies this run:");
        for rec in new_genes.iter().take(20) {
            let flags = if rec.smc_flags.is_empty() {
                "-".to_string()
            } else {
                rec.smc_flags.clone()
            };
            println!(
                "    fitness={:.4} sharpe={:.3} trades={:.0} smc=[{}] indicators={:?}",
                rec.fitness, rec.sharpe, rec.trades, flags, rec.indicator_names
            );
        }
        if new_genes.len() > 20 {
            println!("    ... and {} more", new_genes.len() - 20);
        }
    }
    Ok(())
}

/// `trader-replay --symbol EURUSD --base M1 [--root data]` — offline dry-run of
/// the autonomous-trader engine over real on-disk history. Drives the SAME
/// `neoethos_trader` engine the app's `/autonomous/replay` endpoint does (UI↔CLI
/// parity) with ZERO broker calls, and prints the resulting EngineStats. Symbol
/// and base resolve through the shared `SystemConfig` resolvers, same as
/// `discover`.
fn cmd_trader_replay(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    // With --portfolio <live_portfolio.json>, run the REAL discovered genes
    // (symbol/base come from the artifact). Without it, run the momentum stub on
    // --symbol/--base. Both drive the SAME engine (parity with /autonomous/replay).
    let stats = if let Some(portfolio) = parse_flag(args, "--portfolio") {
        // `--blend off|confirm|scale` (default off). With confirm/scale the
        // discovered genes' size is gated by the per-(symbol,base_tf)
        // SoftVotingEnsemble loaded from `--models-root` (default `models`) —
        // gene-dominant meta-labeling; ML never flips direction. `off` is
        // byte-identical to the gene-only path.
        let blend_arg = parse_flag(args, "--blend").unwrap_or_else(|| "off".to_string());
        let mode = match blend_arg.trim().to_ascii_lowercase().as_str() {
            "off" | "genes" | "genes_only" | "genesonly" => neoethos_trader::BlendMode::GenesOnly,
            "confirm" | "mlconfirm" => neoethos_trader::BlendMode::MlConfirm,
            "scale" | "mlscale" => neoethos_trader::BlendMode::MlScale,
            other => anyhow::bail!("--blend must be off|confirm|scale (got '{other}')"),
        };
        if matches!(mode, neoethos_trader::BlendMode::GenesOnly) {
            neoethos_trader::replay_portfolio_from_dir(
                &root,
                &portfolio,
                neoethos_trader::EngineConfig::try_for_replay_from_settings(
                    settings.as_ref(),
                    &default_symbol(settings.as_ref()),
                )?,
            )?
        } else {
            let models_root =
                parse_flag(args, "--models-root").unwrap_or_else(|| "models".to_string());
            // Both multipliers go through `BlendConfig::from_config_values` —
            // the ONE constructor that validates them. Writing the fields
            // directly (what this did until 2026-08-10) bypassed the inversion
            // refusal, so the CLI could hand the replay a blend the constructor
            // would have rejected: `--veto-below 0.80 --gate-floor 0.10` used to
            // be accepted silently and vetoes every bar the floor exists to keep
            // tradeable. The old `.clamp(0.0, 1.0)` was the other half of the
            // problem — it turned `--gate-floor 1.5` into 1.0 with no message.
            let requested_gate_floor = parse_blend_knob(args, "--gate-floor")?;
            let requested_veto_below = parse_blend_knob(args, "--veto-below")?;
            let blend = neoethos_trader::BlendConfig::from_config_values(
                mode,
                requested_gate_floor,
                requested_veto_below,
            );
            // Print the EFFECTIVE numbers next to what was asked for, so a
            // refusal (also logged at warn by the constructor, with both values)
            // is visible on stdout too — this replay's every position size
            // depends on which of the two won.
            println!(
                "  blend mode={blend_arg} models_root={models_root} gate_floor={:.2} veto_below={:.2}",
                blend.gate_floor, blend.veto_below
            );
            for (name, requested, used) in [
                ("gate-floor", requested_gate_floor, blend.gate_floor),
                ("veto-below", requested_veto_below, blend.veto_below),
            ] {
                if let Some(req) = requested {
                    if req != used {
                        println!(
                            "  WARNING: --{name} {req} REFUSED (out of [0,1] or inverted pair) — using {used}"
                        );
                    }
                }
            }
            neoethos_trader::replay_blend_from_dir(
                &root,
                &portfolio,
                &models_root,
                neoethos_trader::EngineConfig::try_for_replay_from_settings(
                    settings.as_ref(),
                    &default_symbol(settings.as_ref()),
                )?,
                blend,
            )?
        }
    } else {
        let symbol =
            parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
        let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
        if symbol.trim().is_empty() || base.trim().is_empty() {
            anyhow::bail!(
                "trader-replay needs --symbol and --base (or a reachable config.yaml with \
                 system.symbol / system.base_timeframe), or pass \
                 --portfolio <live_portfolio.json> to run the discovered genes"
            );
        }
        neoethos_trader::replay_symbol_from_dir(
            &root,
            &symbol,
            &base,
            neoethos_trader::EngineConfig::try_for_replay_from_settings(
                settings.as_ref(),
                &symbol,
            )?,
        )?
    };
    println!("trader-replay (offline dry-run, zero broker calls):");
    println!(
        "  bars={} signals={} intents={} executed={} blocked={}",
        stats.bars_processed,
        stats.signals_evaluated,
        stats.intents_emitted,
        stats.intents_executed,
        stats.intents_blocked
    );
    println!(
        "  positions: opened={} closed={} open_now={}",
        stats.positions_opened, stats.positions_closed, stats.open_positions
    );
    println!(
        "  realized_pnl={:.5} equity={:.2}",
        stats.realized_pnl, stats.equity
    );
    Ok(())
}

/// `forward-test --portfolio <live_portfolio.json> [--root data] [--oos-from 2023-01-01]`
/// — FAITHFUL out-of-sample test: runs each gene's REAL strategy (its own SL/TP +
/// risk-based confidence-scaled sizing + full costs) via the discovery backtest
/// engine on the holdout window, features computed warm over the FULL series then
/// sliced to [oos-from, end). Reports per-gene IS-vs-OOS + Walk-Forward Efficiency.
fn cmd_forward_test(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let config = settings
        .as_ref()
        .map(neoethos_search::DiscoveryConfig::try_from_settings)
        .transpose()?
        .unwrap_or_default();
    let portfolio = parse_flag(args, "--portfolio").ok_or_else(|| {
        anyhow::anyhow!("forward-test requires --portfolio <live_portfolio.json>")
    })?;
    let oos_from = parse_flag(args, "--oos-from").unwrap_or_else(|| "2023-01-01".to_string());
    let oos_ms = parse_ymd_to_epoch_ms(&oos_from, "--oos-from")?;

    let results = neoethos_search::faithful_oos_eval(
        &config,
        std::path::Path::new(&root),
        std::path::Path::new(&portfolio),
        oos_ms,
    )?;

    println!("FAITHFUL OOS forward-test (gene real SL/TP + risk sizing; holdout from {oos_from}):");
    println!(
        "{:<16}{:>5}{:>5}{:>8}{:>8}{:>8}{:>8}{:>10}{:>7}{:>8}  verdict",
        "gene",
        "#ind",
        "#smc",
        "IS_PF",
        "IS_DD%",
        "OOS_PF",
        "OOS_DD%",
        "OOS_net",
        "OOS_tr",
        "WFE_shp"
    );
    let mut survives = 0usize;
    for r in &results {
        let net = r.oos.net_profit;
        let oos_dd = r.oos.max_drawdown * 100.0;
        let is_survivor = r.oos.trade_count >= 30
            && r.wfe_sharpe >= 0.5
            && r.oos.profit_factor >= 1.3
            && oos_dd <= 10.0
            && net > 0.0;
        let verdict = if r.oos.trade_count < 30 {
            "DEAD (<30 tr)"
        } else if is_survivor {
            "SURVIVES"
        } else if net > 0.0 && r.oos.profit_factor >= 1.0 {
            "weak+"
        } else {
            "FAILS-OOS"
        };
        if is_survivor {
            survives += 1;
        }
        println!(
            "{:<16}{:>5}{:>5}{:>8.2}{:>8.1}{:>8.2}{:>8.1}{:>10.0}{:>7}{:>8.2}  {}",
            r.strategy_id,
            r.n_indicators,
            r.n_smc,
            r.is_profit_factor,
            r.is_max_drawdown * 100.0,
            r.oos.profit_factor,
            oos_dd,
            net,
            r.oos.trade_count,
            r.wfe_sharpe,
            verdict
        );
    }
    println!(
        "SURVIVES={}/{} (WFE_sharpe>=0.5, OOS PF>=1.3, OOS DD<=10%, >=30 trades, net>0)",
        survives,
        results.len()
    );
    Ok(())
}

/// `blend-test --portfolio <live_portfolio.json> --models-root <models_oos_locked>
/// [--root data] [--gate-floor 0.34] [--veto-below 0.15]`
///
/// Stage 4 — re-validate the gene↔ML blend on the NETTED portfolio the live
/// engine actually trades (verdict #1: the trader nets the genes via
/// combine_gene_signals, so we compare on the netted signal, not per-gene). Runs
/// the SAME trader engine three ways — GenesOnly (baseline) vs MlConfirm vs
/// MlScale — over identical bars, and prints a paired EngineStats table + a
/// non-degradation verdict. The blend ships ON only if it does NOT degrade
/// genes-only.
///
/// IMPORTANT: point `--models-root` at a LEAK-FREE root (`cli train --oos-from
/// <date> --models-dir models_oos_locked`), else the ensemble has seen the
/// evaluation window and the comparison is contaminated. Note the trader engine
/// uses a uniform SL/TP model (decision.rs), so the absolute P&L is a simplified
/// figure — but a GenesOnly-vs-blend comparison on the SAME engine is a valid
/// apples-to-apples accept/reject (only the gated signal differs). The rigorous
/// per-gene faithful number remains `forward-test`.
fn cmd_blend_test(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let portfolio = parse_flag(args, "--portfolio")
        .ok_or_else(|| anyhow::anyhow!("blend-test requires --portfolio <live_portfolio.json>"))?;
    let models_root =
        parse_flag(args, "--models-root").unwrap_or_else(|| "models_oos_locked".to_string());
    // Same rule as `trader-replay`: the flags are handed to
    // `BlendConfig::from_config_values`, never written into the fields. This
    // site used to carry its OWN copies of the shipped defaults (0.34 / 0.15)
    // and its own `.clamp(0.0, 1.0)` — three ways to drift from
    // `DEFAULT_BLEND_GATE_FLOOR` / `DEFAULT_BLEND_VETO_BELOW` and one way
    // (an inverted pair) to run all three arms of the comparison with a blend
    // that vetoes every bar, which reads as "the blend is safe, it opened
    // nothing".
    let requested_gate_floor = parse_blend_knob(args, "--gate-floor")?;
    let requested_veto_below = parse_blend_knob(args, "--veto-below")?;
    let validated = neoethos_trader::BlendConfig::from_config_values(
        neoethos_trader::BlendMode::GenesOnly,
        requested_gate_floor,
        requested_veto_below,
    );
    let (gate_floor, veto_below) = (validated.gate_floor, validated.veto_below);
    for (name, requested, used) in [
        ("gate-floor", requested_gate_floor, gate_floor),
        ("veto-below", requested_veto_below, veto_below),
    ] {
        if let Some(req) = requested {
            if req != used {
                println!(
                    "  WARNING: --{name} {req} REFUSED (out of [0,1] or inverted pair) — using {used}"
                );
            }
        }
    }

    // All three arms get the SAME validated multipliers — only `mode` differs,
    // which is the entire point of the comparison.
    let run = |mode| -> Result<neoethos_trader::EngineStats> {
        neoethos_trader::replay_blend_from_dir(
            &root,
            &portfolio,
            &models_root,
            neoethos_trader::EngineConfig::try_for_replay_from_settings(
                settings.as_ref(),
                &default_symbol(settings.as_ref()),
            )?,
            neoethos_trader::BlendConfig::from_config_values(
                mode,
                Some(gate_floor),
                Some(veto_below),
            ),
        )
    };
    let genes = run(neoethos_trader::BlendMode::GenesOnly)?;
    let confirm = run(neoethos_trader::BlendMode::MlConfirm)?;
    let scale = run(neoethos_trader::BlendMode::MlScale)?;

    println!(
        "blend-test (NETTED trader engine, models_root={models_root}, gate_floor={gate_floor:.2}, veto_below={veto_below:.2}):"
    );
    let row = |name: &str, s: &neoethos_trader::EngineStats| {
        println!(
            "  {name:<8} pnl={:>12.5} equity={:>12.2} opened={:>5} closed={:>5} blocked={:>5} signals={:>6}",
            s.realized_pnl,
            s.equity,
            s.positions_opened,
            s.positions_closed,
            s.intents_blocked,
            s.signals_evaluated
        );
    };
    row("genes", &genes);
    row("confirm", &confirm);
    row("scale", &scale);

    // Non-degradation accept gate vs the genes-only baseline (same engine/bars).
    let verdict = |name: &str, s: &neoethos_trader::EngineStats| {
        let pnl_ok = s.realized_pnl >= genes.realized_pnl - 1e-9;
        let eq_ok = s.equity >= genes.equity - 1e-9;
        let traded = s.positions_opened >= 1;
        if pnl_ok && eq_ok && traded {
            println!(
                "  -> {name}: ACCEPT (>= genes-only on realized_pnl AND equity, still trades)"
            );
        } else if !traded {
            println!("  -> {name}: REJECT (blend vetoed every trade)");
        } else {
            println!("  -> {name}: REJECT (degrades vs genes-only)");
        }
    };
    verdict("confirm", &confirm);
    verdict("scale", &scale);
    println!(
        "NOTE: relative comparison on the trader's uniform-SL/TP engine; ensure --models-root is \
         leak-free (train --oos-from). Per-gene faithful numbers: `forward-test`."
    );
    Ok(())
}

fn cmd_train(args: &[String]) -> Result<()> {
    let result = (|| -> Result<(String, String)> {
        let settings_opt = resolve_cli_settings(args)?;
        // Folder-browse support (2026-05-14): `--data-path <folder>`
        // scans the folder, prints a discovery summary, and (if
        // `--dry-run` is also set) exits before training kicks off.
        if has_flag(args, "--data-path") || has_flag(args, "--dry-run") {
            let root = parse_root(args, settings_opt.as_ref());
            let _ = print_dataset_discovery_summary(&root)?;
            if has_flag(args, "--dry-run") {
                let dry_symbol = parse_flag(args, "--symbol")
                    .unwrap_or_else(|| default_symbol(settings_opt.as_ref()));
                let dry_base = parse_flag(args, "--base")
                    .unwrap_or_else(|| default_base_tf(settings_opt.as_ref()));
                return Ok((dry_symbol, dry_base));
            }
        }
        let settings = settings_opt.unwrap_or_else(neoethos_core::Settings::default);
        let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| settings.system.symbol.clone());
        let base =
            parse_flag(args, "--base").unwrap_or_else(|| settings.system.base_timeframe.clone());
        let data_root = parse_root(args, Some(&settings));
        // Stage 4 leak-free OOS-locked retrain: `--oos-from YYYY-MM-DD` truncates
        // each symbol's training to rows strictly before the cutoff (minus the
        // triple-barrier purge), so the experts can be used in an OOS blend
        // validation on [cutoff, end) without look-ahead. The locked experts MUST
        // go to a SEPARATE root so production `models/` is never overwritten.
        let oos_ms = match parse_flag(args, "--oos-from") {
            Some(d) => Some(parse_ymd_to_epoch_ms(&d, "--oos-from")?),
            None => None,
        };
        let default_models_dir = if oos_ms.is_some() {
            "models_oos_locked"
        } else {
            "models"
        };
        let models_dir =
            parse_flag(args, "--models-dir").unwrap_or_else(|| default_models_dir.to_string());
        if oos_ms.is_some() {
            let norm = models_dir.replace('\\', "/");
            if models_dir == "models" || norm.ends_with("/models") {
                anyhow::bail!(
                    "--oos-from trains LEAK-LOCKED experts; refusing to write them to the \
                     production '{models_dir}' root. Use a distinct --models-dir \
                     (e.g. models_oos_locked)."
                );
            }
        }
        let mut orchestrator = neoethos_models::TrainingOrchestrator::new(
            settings,
            std::path::PathBuf::from(models_dir),
        )
        .with_data_root(data_root);
        if let Some(ms) = oos_ms {
            orchestrator = orchestrator.with_oos_lock_from_ms(ms);
        }

        let installed = neoethos_core::execution_budget::installed_process_budget()
            .context("training unavailable before the process CPU budget is installed")?;
        let lease = installed.broker().acquire(
            neoethos_core::execution_budget::CpuPermitRequest::local(
                installed.resolved().effective_worker_limit,
            ),
        )?;
        orchestrator.train_symbol(&symbol, &base, &lease)?;

        println!("Pure Rust training complete for {}", symbol);
        Ok((symbol, base))
    })();

    match &result {
        Ok((symbol, base)) => {
            write_subsystem_record(
                SubsystemSection::Training,
                section_record(
                    SubsystemSection::Training,
                    "train",
                    "SUCCESS",
                    format!("training completed for {} {}", symbol, base),
                ),
            )?;
        }
        Err(err) => {
            write_subsystem_record(
                SubsystemSection::Training,
                section_record(
                    SubsystemSection::Training,
                    "train",
                    "FAILED",
                    format!("training failed: {}", err),
                ),
            )?;
        }
    }

    result.map(|_| ())
}

/// What a streaming sweep leaves behind, beyond the per-batch portfolios: the
/// run-level canonical feature list, the survivors remapped onto it, and the
/// census of every batch the sweep abandoned.
///
/// Carried as a struct so the artifact-writing block below reads the same
/// whether the sweep ran one batch or forty.
struct StreamingArtifactBundle {
    canonical: neoethos_search::orchestration::CanonicalFeatureIndex,
    survivors: Vec<neoethos_search::orchestration::CanonicalSurvivor>,
    ledger: neoethos_search::orchestration::StreamingRunLedger,
    primary_cursor: usize,
    next_cursor: usize,
    space_len: usize,
    batch_columns: usize,
    streamed: bool,
}

fn cmd_discover(args: &[String]) -> Result<()> {
    let result = (|| -> Result<(String, String, usize, usize)> {
        let settings = resolve_cli_settings(args)?;
        let defaults = settings
            .as_ref()
            .map(neoethos_search::DiscoveryConfig::try_from_settings)
            .transpose()?
            .unwrap_or_default();
        let root = parse_root(args, settings.as_ref());
        let symbol =
            parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
        let base = parse_flag(args, "--base").unwrap_or_else(|| default_base_tf(settings.as_ref()));
        let higher = parse_flag(args, "--higher")
            .unwrap_or_else(|| default_higher_tfs_csv(settings.as_ref(), &base));
        let higher_list: Vec<String> = higher
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect();
        // Folder-browse support (2026-05-14): when `--data-path` or
        // `--dry-run` are supplied, scan the folder and emit a
        // canonical manifest summary before the GA pipeline starts.
        if has_flag(args, "--data-path") || has_flag(args, "--dry-run") {
            let _ = print_dataset_discovery_summary(&root)?;
        }
        let identities = inventory_canonical_identities(&root, &symbol)?;
        let base_identity = select_exact_runtime_identity(&identities, args, &symbol, &base)?;
        let selection = select_runtime_timeframe_identities_for_base(
            &identities,
            &base_identity,
            &higher_list,
        )?;
        if has_flag(args, "--dry-run") {
            return Ok((symbol, base, 0, 0));
        }
        // F-304 fix (2026-05-28): bind the account currency for the
        // cost model. Resolution order:
        //   1. `--account-currency` CLI flag (operator-explicit)
        //   2. `Settings.system.account_currency` (from config.yaml or
        //      cTrader trader profile written back by the bridge)
        // (The legacy `NEOETHOS_BOT_PROP_ACCOUNT_CURRENCY` env override was
        // removed in v0.4.36 — config / CLI is the source.)
        // Empty propagates downstream — the cost-model NaN guard will
        // reject the run with a clear error message rather than
        // silently producing NaN spread/pip values that the sanitizer
        // scrubs to 0.0 (= GA sees zero-trade candidates).
        let account_currency = parse_flag(args, "--account-currency")
            .or_else(|| {
                settings
                    .as_ref()
                    .map(|s| s.system.account_currency.clone())
                    .filter(|c| !c.trim().is_empty())
            })
            .unwrap_or_default();
        let population: usize = parse_flag(args, "--population")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.population);
        let generations: usize = parse_flag(args, "--generations")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.generations);
        let max_indicators: usize = parse_flag(args, "--max-indicators")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.max_indicators);
        let candidate_count: usize = parse_flag(args, "--candidates")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.candidate_count);
        let portfolio_size: usize = parse_flag(args, "--portfolio-size")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.portfolio_size);
        let corr_threshold: f64 = parse_flag(args, "--corr")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.corr_threshold);
        let min_trades_per_day: f64 = parse_flag(args, "--min-trades")
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.min_trades_per_day);
        let out = parse_flag(args, "--out")
            .unwrap_or_else(|| "cache/vector_ta_knowledge.json".to_string());

        #[cfg(not(feature = "gpu-nvidia"))]
        let higher_refs: Vec<&str> = higher_list.iter().map(|s| s.as_str()).collect();

        // Snapshot immutable generation metadata before any run. On a CUDA
        // build, values are decoded only inside the selected prepared CPU
        // factory, after one cross-vendor physical-device admission. The
        // featureless CPU build retains its compatibility path below.
        #[cfg(feature = "gpu-nvidia")]
        let mut pinned_selection = pin_direct_timeframe_selection(&root, &selection)?;
        #[cfg(not(feature = "gpu-nvidia"))]
        let dataset = load_required_direct_symbol_dataset(&root, &symbol, &selection)?;
        #[cfg(not(feature = "gpu-nvidia"))]
        let base_frame = dataset.canonical_frame(&base)?;
        #[cfg(not(feature = "gpu-nvidia"))]
        let base_ohlcv = base_frame.ohlcv();

        let config = neoethos_search::DiscoveryConfig {
            timeframe_label: base.clone(),
            // F-304 fix (2026-05-28): bind the CLI-resolved symbol +
            // account currency BEFORE `..defaults.clone()` so the
            // cost-model receives the operator's chosen values, not
            // the (potentially stale or empty) settings copy. Empty
            // values still propagate and trip the run-loud guard.
            evaluation_symbol: symbol.clone(),
            evaluation_account_currency: account_currency.clone(),
            population,
            generations,
            max_indicators,
            candidate_count,
            portfolio_size,
            corr_threshold,
            min_trades_per_day,
            filtering: defaults.filtering,
            ..defaults.clone()
        }
        .apply_mode_overrides();
        // ── THE STREAMING WORKING-SET SWEEP (opt-in) ────────────────────────
        //
        // `--stream-sweep` advances the working set through the
        // (indicator, period) space in batches instead of building ONE cube and
        // searching it. `--stream-max-batches N` spends at most N batches
        // (default: until the space is exhausted).
        //
        // The batch WIDTH is deliberately not a flag: it comes from free RAM
        // and the widest frame via `hpc_ta::streaming_batch_columns`, so peak
        // memory is a function of the hardware and never of what the operator
        // typed (the never-OOM invariant).
        //
        // WITHOUT the flag this is byte-for-byte the previous code: one
        // `prepare_multitimeframe_features`, one holdout cycle, the same
        // artifacts. That is the parity case, and it is the default.
        let stream_sweep = has_flag(args, "--stream-sweep");
        let stream_max_batches: usize = parse_flag(args, "--stream-max-batches")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        // Audit B02/B03 (2026-07-13): the CLI used to run discovery on the
        // FULL series — no held-out tail, so every "validation" window had
        // already been seen during selection. The holdout wrapper withholds
        // the last 20% and attaches honest forward-test/prop-firm evidence.
        #[cfg(feature = "gpu-nvidia")]
        let (result, streaming) = if !stream_sweep {
            let feature_options = neoethos_data::FeatureBuildOptions {
                higher_tfs: higher_list.clone(),
                ..neoethos_data::FeatureBuildOptions::default()
            };
            let pinned_series = std::cell::RefCell::new(Some(
                pinned_selection.take_or_repin(std::path::Path::new(root.as_str()))?,
            ));
            let prepared = neoethos_search::prepare_canonical_discovery_run_input_v3(
                |no_physical_gpu_admission| {
                    let pinned_series = pinned_series
                        .borrow_mut()
                        .take()
                        .context("CLI Discovery pin was already consumed")?;
                    let dataset = pinned_series
                        .into_cpu_dataset_after_no_physical_gpu_v1(&no_physical_gpu_admission)?;
                    let base_frame =
                        dataset.canonical_frame(selection.base_identity.timeframe().as_str())?;
                    let features = neoethos_data::prepare_multitimeframe_features_with_options(
                        &dataset,
                        selection.base_identity.timeframe().as_str(),
                        &feature_options,
                    )?;
                    let input = neoethos_search::data_selection::CanonicalSearchInput::from_prepared_canonical_frame(
                        selection.base_identity.clone(),
                        base_frame,
                        features,
                    )?;
                    Ok((input, no_physical_gpu_admission))
                },
                || {
                    anyhow::bail!(
                        "CLI Discovery cannot seal the complete native workspace yet; refusing host feature materialization on a physical GPU"
                    )
                },
                |_admitted_native_run| {
                    let _pinned_series = pinned_series
                        .borrow_mut()
                        .take()
                        .context("CLI Discovery pin was already consumed")?;
                    anyhow::bail!(
                        "CLI native Data materialization is unreachable before workspace sealing"
                    )
                },
            )?;
            let result =
                neoethos_search::run_prepared_canonical_discovery_with_holdout_and_progress_v3(
                    prepared,
                    &config,
                    neoethos_search::PropFirmRiskRules::default(),
                    |_| {},
                )?;
            (result, None)
        } else {
            let feature_options = neoethos_data::FeatureBuildOptions {
                higher_tfs: higher_list.clone(),
                ..neoethos_data::FeatureBuildOptions::default()
            };
            let mut outcome =
                neoethos_search::orchestration::run_prepared_streaming_working_set_v3(
                    &neoethos_search::orchestration::StreamingPlan::streaming(stream_max_batches),
                    pinned_selection.base_row_count,
                    |_batch| pinned_selection.take_or_repin(std::path::Path::new(root.as_str())),
                    |batch, pinned_series, no_physical_gpu_admission| {
                        let dataset = pinned_series.into_cpu_dataset_after_no_physical_gpu_v1(
                            &no_physical_gpu_admission,
                        )?;
                        let base_frame = dataset
                            .canonical_frame(selection.base_identity.timeframe().as_str())?;
                        let input = neoethos_data::with_extended_sweep_working_set(batch, || {
                            let features =
                                neoethos_data::prepare_multitimeframe_features_with_options(
                                    &dataset,
                                    selection.base_identity.timeframe().as_str(),
                                    &feature_options,
                                )?;
                            neoethos_search::data_selection::CanonicalSearchInput::from_prepared_canonical_frame(
                                selection.base_identity.clone(),
                                base_frame,
                                features,
                            )
                            .map_err(anyhow::Error::new)
                        })?;
                        Ok((input, no_physical_gpu_admission))
                    },
                    |_batch| {
                        anyhow::bail!(
                            "CLI streaming Discovery cannot seal the complete native workspace yet; refusing host feature materialization on a physical GPU"
                        )
                    },
                    |_batch, _pinned_series, _admitted_native_run| {
                        anyhow::bail!(
                            "CLI streaming native Data materialization is unreachable before workspace sealing"
                        )
                    },
                    |prepared| {
                        neoethos_search::run_prepared_canonical_discovery_with_holdout_and_progress_v3(
                        prepared,
                        &config,
                        neoethos_search::PropFirmRiskRules::default(),
                        |_| {},
                    )
                    },
                )?;
            if outcome.batches.is_empty() {
                let rejected: Vec<(usize, &'static str)> = outcome
                    .ledger
                    .rejected_rows()
                    .iter()
                    .map(|row| (row.cursor, row.outcome.as_str()))
                    .collect();
                anyhow::bail!(
                    "streaming sweep produced no portfolio: {} batches attempted, {} abandoned \
                     (counts: {:?}; abandoned cursors: {:?}). The sweep covered pairs \
                     [0, {}) of {} at {} columns per batch. Nothing was lost silently — every \
                     cursor above has a reason in the run log \
                     (target=neoethos_search::batch_ledger).",
                    outcome.ledger.batches_seen(),
                    outcome.ledger.batches_rejected(),
                    outcome.ledger.counts_by_outcome(),
                    rejected,
                    outcome.next_cursor,
                    outcome.space_len,
                    outcome.batch_columns
                );
            }
            let survivors = outcome.survivors();
            let primary = outcome.batches.remove(0);
            let bundle = StreamingArtifactBundle {
                canonical: outcome.canonical.clone(),
                survivors,
                ledger: outcome.ledger.clone(),
                primary_cursor: primary.cursor,
                next_cursor: outcome.next_cursor,
                space_len: outcome.space_len,
                batch_columns: outcome.batch_columns,
                streamed: outcome.streamed,
            };
            let extra: Vec<(usize, neoethos_search::DiscoveryResult)> = outcome
                .batches
                .into_iter()
                .map(|batch| (batch.cursor, batch.result))
                .collect();
            (primary.result, Some((bundle, extra)))
        };

        #[cfg(not(feature = "gpu-nvidia"))]
        let (result, streaming) = if !stream_sweep {
            let features =
                neoethos_data::prepare_multitimeframe_features(&dataset, &base, &higher_refs)?;
            let receipt =
                neoethos_search::data_selection::CanonicalSearchInputReceiptV2::from_feature_frame(
                    &selection.base_identity,
                    &features,
                )?;
            let run_input = neoethos_search::data_selection::CanonicalSearchRunInputV2::new(
                receipt,
                &features,
                &base_frame,
            )?;
            let result = neoethos_search::run_discovery_cycle_with_holdout(
                &run_input,
                &config,
                neoethos_search::PropFirmRiskRules::default(),
            )?;
            (result, None)
        } else {
            let mut outcome = neoethos_search::orchestration::run_streaming_working_set(
                &neoethos_search::orchestration::StreamingPlan::streaming(stream_max_batches),
                base_ohlcv.close.len(),
                // The ONLY sanctioned build entry point: it installs the batch
                // as the working set and restores the previous one afterwards,
                // even on panic. `None` is documented as byte-identical to
                // `prepare_multitimeframe_features`.
                |batch| {
                    neoethos_data::prepare_multitimeframe_features_batch(
                        &dataset,
                        &base,
                        &higher_refs,
                        batch,
                    )
                },
                |features| {
                    let receipt = neoethos_search::data_selection::CanonicalSearchInputReceiptV2::from_feature_frame(
                        &selection.base_identity,
                        features,
                    )?;
                    let run_input =
                        neoethos_search::data_selection::CanonicalSearchRunInputV2::new(
                            receipt,
                            features,
                            &base_frame,
                        )?;
                    neoethos_search::run_discovery_cycle_with_holdout(
                        &run_input,
                        &config,
                        neoethos_search::PropFirmRiskRules::default(),
                    )
                },
            )?;
            if outcome.batches.is_empty() {
                // Loud, and it NAMES every abandoned cursor. An empty sweep
                // that says only "no portfolio" is the silent drop with extra
                // steps.
                let rejected: Vec<(usize, &'static str)> = outcome
                    .ledger
                    .rejected_rows()
                    .iter()
                    .map(|row| (row.cursor, row.outcome.as_str()))
                    .collect();
                anyhow::bail!(
                    "streaming sweep produced no portfolio: {} batches attempted, {} abandoned \
                     (counts: {:?}; abandoned cursors: {:?}). The sweep covered pairs \
                     [0, {}) of {} at {} columns per batch. Nothing was lost silently — every \
                     cursor above has a reason in the run log \
                     (target=neoethos_search::batch_ledger).",
                    outcome.ledger.batches_seen(),
                    outcome.ledger.batches_rejected(),
                    outcome.ledger.counts_by_outcome(),
                    rejected,
                    outcome.next_cursor,
                    outcome.space_len,
                    outcome.batch_columns
                );
            }
            // Provenance for the run-level artifact is collected BEFORE the
            // primary batch is moved out for the per-run artifacts below.
            let survivors = outcome.survivors();
            // The FIRST surviving batch takes today's artifact paths, so a
            // single-batch sweep writes exactly the file set a non-streaming
            // run writes. Later batches are written beside it, keyed by cursor.
            let primary = outcome.batches.remove(0);
            let bundle = StreamingArtifactBundle {
                canonical: outcome.canonical.clone(),
                survivors,
                ledger: outcome.ledger.clone(),
                primary_cursor: primary.cursor,
                next_cursor: outcome.next_cursor,
                space_len: outcome.space_len,
                batch_columns: outcome.batch_columns,
                streamed: outcome.streamed,
            };
            let extra: Vec<(usize, neoethos_search::DiscoveryResult)> = outcome
                .batches
                .into_iter()
                .map(|batch| (batch.cursor, batch.result))
                .collect();
            (primary.result, Some((bundle, extra)))
        };
        if let Some(parent) = std::path::Path::new(&out).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        // F-306 fix (2026-05-28): save funnel + quality + trade log
        // BEFORE the empty-portfolio guard. The previous order made
        // every empty run leave ZERO artifacts on disk, blocking
        // post-mortem diagnosis. Mirrors the app server pattern at
        // `crates/neoethos-app/src/app_services/discovery.rs:988-1007`.
        let funnel_path = format!("{out}.funnel.json");
        if let Err(err) = neoethos_search::save_funnel_json(&funnel_path, &result) {
            tracing::warn!(
                target: "neoethos_cli::discover",
                error = %err,
                path = %funnel_path,
                "save_funnel_json failed (non-fatal — continuing to other artifacts)"
            );
        }
        if !result.quality_metrics.is_empty() {
            let quality_path = format!("{out}.quality.json");
            if let Err(err) = neoethos_search::save_quality_report_json(&quality_path, &result) {
                tracing::warn!(
                    target: "neoethos_cli::discover",
                    error = %err,
                    path = %quality_path,
                    "save_quality_report_json failed (non-fatal)"
                );
            }
        }
        if !result.logged_trades.is_empty() {
            let trade_log_path = format!("{out}.trades.json");
            if let Err(err) = neoethos_search::save_trade_log_json(&trade_log_path, &result) {
                tracing::warn!(
                    target: "neoethos_cli::discover",
                    error = %err,
                    path = %trade_log_path,
                    "save_trade_log_json failed (non-fatal)"
                );
            }
        }
        // Now the empty-portfolio guard. Diagnostics are already on
        // disk, so the operator can post-mortem even when this fires.
        neoethos_search::ensure_non_empty_portfolio(&result, &format!("{} {}", symbol, base))?;
        neoethos_search::save_portfolio_json(&out, &result)?;
        // Phase 4 (2026-06-04): also emit the self-describing live portfolio
        // artifact (full genes + effective_feature_names + base/higher TFs +
        // normalize flag) the autonomous trader loads to evaluate the discovered
        // strategies with backtest parity. Additive + non-fatal.
        {
            let live_path = format!("{out}.live_portfolio.json");
            if let Err(err) = neoethos_search::save_live_portfolio_json(&live_path, &result) {
                tracing::warn!(
                    target: "neoethos_cli::discover",
                    error = %err,
                    path = %live_path,
                    "save_live_portfolio_json failed (non-fatal)"
                );
            }
        }
        let profile_path = format!("{out}.profile.json");
        neoethos_search::save_discovery_profile_json(&profile_path, &config, &result)?;
        let primary_snapshot_root = format!("{out}.validation_snapshot");
        let primary_snapshot =
            neoethos_search::save_discovery_validation_snapshot(&primary_snapshot_root, &result)?;
        // ── The streaming run artifact ──────────────────────────────────────
        //
        // Written ONLY on a streaming run, and never instead of the per-batch
        // artifacts: each surviving batch keeps its own portfolio JSON, whose
        // genes address that batch's own `effective_feature_names` and are
        // therefore still internally consistent. THIS file is the run-level
        // view — Option C's canonical name list, the survivors remapped onto
        // it with the cursor that produced each one, and the batch census.
        if let Some((bundle, extra)) = streaming {
            let mut batch_snapshots = Vec::with_capacity(extra.len() + 1);
            batch_snapshots.push(
                neoethos_search::orchestration::StreamingBatchValidationSnapshotRefV1 {
                    source_cursor: bundle.primary_cursor,
                    snapshot_root: primary_snapshot_root,
                    pointer: primary_snapshot,
                },
            );
            for (cursor, batch_result) in &extra {
                let batch_out = format!("{out}.batch{cursor}.json");
                // Non-fatal, deliberately: `save_portfolio_json` runs the
                // export-readiness gate, and a later batch failing it must not
                // discard the batches already written. Nothing is lost — the
                // genes themselves are in the run-level artifact below, with
                // this cursor on them — but the failure is named, not swallowed.
                if let Err(err) = neoethos_search::save_portfolio_json(&batch_out, batch_result) {
                    tracing::warn!(
                        target: "neoethos_cli::discover",
                        error = %err,
                        cursor = *cursor,
                        path = %batch_out,
                        "per-batch portfolio export failed (non-fatal — this batch's survivors \
                        are still in the streaming run artifact)"
                    );
                }
                let batch_snapshot_root = format!("{out}.batch{cursor}.validation_snapshot");
                let pointer = neoethos_search::save_discovery_validation_snapshot(
                    &batch_snapshot_root,
                    batch_result,
                )?;
                batch_snapshots.push(
                    neoethos_search::orchestration::StreamingBatchValidationSnapshotRefV1 {
                        source_cursor: *cursor,
                        snapshot_root: batch_snapshot_root,
                        pointer,
                    },
                );
            }
            batch_snapshots.sort_by_key(|snapshot| snapshot.source_cursor);
            anyhow::ensure!(
                batch_snapshots
                    .windows(2)
                    .all(|pair| pair[0].source_cursor < pair[1].source_cursor),
                "streaming validation snapshots contain duplicate source cursors"
            );
            let genes: Vec<neoethos_search::Gene> = bundle
                .survivors
                .iter()
                .map(|survivor| survivor.gene.clone())
                .collect();
            // INVARIANT 4 again, at the artifact boundary this time: nothing
            // leaves the process addressing a name that does not exist.
            bundle
                .canonical
                .assert_indices_in_range(&genes, "cli streaming run portfolio")?;
            let artifact = neoethos_search::orchestration::StreamingRunPortfolio {
                schema_version:
                    neoethos_search::orchestration::STREAMING_RUN_PORTFOLIO_SCHEMA_VERSION,
                symbol: symbol.clone(),
                base_timeframe: base.clone(),
                higher_timeframes: config.higher_timeframes.clone(),
                canonical_feature_names: bundle.canonical.names().to_vec(),
                survivors: bundle.survivors,
                promotion_authority:
                    neoethos_search::orchestration::StreamingPromotionAuthorityV1::PerBatchLocalOnly {
                        batch_snapshots,
                    },
                next_cursor: bundle.next_cursor,
                space_len: bundle.space_len,
                batch_columns: bundle.batch_columns,
                ledger: bundle.ledger,
            };
            let streaming_path = format!("{out}.streaming.json");
            neoethos_core::storage::json::write_json_atomic(&streaming_path, &artifact)?;
            println!(
                "Streaming sweep streamed={} batches={} kept={} abandoned={} survivors={} \
                 canonical_features={} cursor={}/{} batch_columns={} out={}",
                bundle.streamed,
                artifact.ledger.batches_seen(),
                artifact.ledger.batches_kept(),
                artifact.ledger.batches_rejected(),
                artifact.survivors.len(),
                artifact.canonical_feature_names.len(),
                artifact.next_cursor,
                artifact.space_len,
                artifact.batch_columns,
                streaming_path
            );
        }
        println!(
            "Discovery {} portfolio={} candidates={} out={}",
            symbol,
            result.portfolio.len(),
            result.candidates.len(),
            out
        );
        Ok((
            symbol,
            base,
            result.portfolio.len(),
            result.candidates.len(),
        ))
    })();

    match &result {
        Ok((symbol, base, portfolio, candidates)) => {
            write_subsystem_record(
                SubsystemSection::Discovery,
                section_record(
                    SubsystemSection::Discovery,
                    "discover",
                    "SUCCESS",
                    format!(
                        "discovery completed for {} {} portfolio={} candidates={}",
                        symbol, base, portfolio, candidates
                    ),
                ),
            )?;
        }
        Err(err) => {
            write_subsystem_record(
                SubsystemSection::Discovery,
                section_record(
                    SubsystemSection::Discovery,
                    "discover",
                    "FAILED",
                    format!("discovery failed: {}", err),
                ),
            )?;
        }
    }

    result.map(|_| ())
}

fn apply_batch_discover_cli_overrides(
    args: &[String],
    config: &mut neoethos_search::DiscoveryConfig,
) -> Result<()> {
    if let Some(p) = parse_flag(args, "--population")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
    {
        config.population = p;
    }
    if has_flag(args, "--population-auto") {
        let raw = parse_flag(args, "--population-auto")
            .context("--population-auto requires an explicit true or false value")?;
        config.population_auto = raw.parse::<bool>().with_context(|| {
            format!("invalid --population-auto value `{raw}`; expected true or false")
        })?;
    }
    if let Some(g) = parse_flag(args, "--generations")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
    {
        config.generations = g;
    }
    if let Some(ps) = parse_flag(args, "--portfolio-size")
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
    {
        config.portfolio_size = ps;
    }
    Ok(())
}

fn cmd_batch_discover(args: &[String]) -> Result<()> {
    let result = (|| -> Result<(String, usize, usize)> {
        let settings = resolve_cli_settings(args)?;
        let root = parse_root(args, settings.as_ref());
        let symbols_raw = parse_flag(args, "--symbols").unwrap_or_default();
        let tfs_raw = parse_flag(args, "--timeframes")
            .unwrap_or_else(|| default_batch_timeframes_csv(settings.as_ref()));
        let out_dir =
            parse_flag(args, "--out-dir").unwrap_or_else(|| "cache/discovery".to_string());

        let symbols: Vec<String> = if symbols_raw.is_empty() {
            metadata_inventory_symbols(&root)?
        } else {
            symbols_raw
                .split(',')
                .map(|s| s.trim().to_uppercase())
                .collect()
        };

        let tfs: Vec<String> = tfs_raw
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .collect();

        let mut config = settings
            .as_ref()
            .map(neoethos_search::DiscoveryConfig::try_from_settings)
            .transpose()?
            .unwrap_or_default();
        // Explicit overrides win over the config-derived values (same
        // precedence as env > config elsewhere). These let the TUI Discover
        // form's Population/Generations/Portfolio-size fields actually take
        // effect instead of being silently dropped (parity fix 2026-06-08).
        apply_batch_discover_cli_overrides(args, &mut config)?;
        let inventory = neoethos_data::DatasetDiscovery::scan(&root)
            .with_context(|| format!("fully verify canonical batch inventory under {root}"))?;
        for entry in &inventory.entries {
            print_dataset_inventory_entry(entry)?;
        }
        print_dataset_inventory_rejections(&inventory);
        let mut anchors = Vec::with_capacity(symbols.len().saturating_mul(tfs.len()));
        for symbol in &symbols {
            let identities = verified_canonical_identities(&inventory, symbol)?;
            for timeframe in &tfs {
                let selection = select_runtime_timeframe_identities(
                    &identities,
                    symbol,
                    timeframe,
                    &config.higher_timeframes,
                )
                .with_context(|| {
                    format!(
                        "batch-discover direct-timeframe preflight failed for {symbol}/{timeframe}; import/download required"
                    )
                })?;
                anchors.push(selection.base_identity);
            }
        }
        anchors.sort_by_key(|identity| identity.to_path_component());
        anchors.dedup();
        anyhow::ensure!(
            !anchors.is_empty(),
            "batch-discover resolved zero exact canonical anchors"
        );
        let orchestrator = neoethos_search::DiscoveryOrchestrator::new(&root, &out_dir, config);

        let summary = orchestrator.run_batch(&anchors)?;

        println!(
            "Batch discovery complete. Results in {} (saved={} work_units={} skipped_symbols={} skipped_timeframes={} feature_failures={} empty_portfolios={})",
            out_dir,
            summary.portfolios_saved,
            summary.work_units_seen,
            summary.skipped_symbols,
            summary.skipped_timeframes,
            summary.feature_failures,
            summary.empty_portfolios
        );
        Ok((out_dir, summary.portfolios_saved, summary.work_units_seen))
    })();

    match &result {
        Ok((out_dir, saved, work_units)) => {
            write_subsystem_record(
                SubsystemSection::Discovery,
                section_record(
                    SubsystemSection::Discovery,
                    "batch-discover",
                    "SUCCESS",
                    format!(
                        "batch discovery completed out_dir={} saved={} work_units={}",
                        out_dir, saved, work_units
                    ),
                ),
            )?;
        }
        Err(err) => {
            write_subsystem_record(
                SubsystemSection::Discovery,
                section_record(
                    SubsystemSection::Discovery,
                    "batch-discover",
                    "FAILED",
                    format!("batch discovery failed: {}", err),
                ),
            )?;
        }
    }

    result.map(|_| ())
}

/// Print the resolved config: every setting with raw value, resolved
/// value, source (config / sentinel-expanded / env / default), and
/// notes. The TUI's Config page renders the same data.
fn cmd_config(args: &[String]) -> Result<()> {
    if args.first().map(String::as_str) == Some("normalize") {
        return cmd_config_normalize(&args[1..]);
    }
    let settings = resolve_cli_settings(args)?.unwrap_or_else(neoethos_core::Settings::default);
    let resolved = neoethos_core::resolved_config::ResolvedConfig::from_settings(&settings);

    if has_flag(args, "--json") {
        let text = serde_json::to_string_pretty(&resolved)
            .map_err(|e| anyhow::anyhow!("serialize resolved config: {e}"))?;
        println!("{}", text);
        return Ok(());
    }

    println!("Resolved configuration");
    println!("======================");
    println!(
        "{:<10} {:<28} {:<28} {:<28} {:<8}",
        "section", "field", "raw", "resolved", "source"
    );
    println!("{}", "-".repeat(110));
    for row in resolved.display_table() {
        println!(
            "{:<10} {:<28} {:<28} {:<28} {:<8}",
            row[0], row[1], row[2], row[3], row[4]
        );
    }
    println!();
    println!("Notes:");
    for f in &resolved.display_fields {
        if let Some(note) = &f.note {
            println!("  {} / {}: {}", f.section, f.field, note);
        }
    }
    Ok(())
}

/// `config normalize` — show, and optionally rewrite, the operator's store as
/// an OVERRIDE DOCUMENT instead of a full snapshot.
///
/// Why this command exists. Older builds saved settings by dumping every field,
/// so a store written then is a photograph of that build's defaults. From that
/// moment it shadows every default the codebase improves — not by disagreeing
/// with them, but by repeating older ones that were never chosen. The operator
/// cannot see this by reading his file: a deliberate limit and a fossilised
/// default are the same line of YAML. The measured case on this machine was a
/// 509-line store in which seven gates sat at values no one had selected,
/// including a payoff floor of 0.0 against a default of 2.0 and an export gate
/// disabled against a default that requires walk-forward.
///
/// The fix is not to patch values — that is the same mistake with fresher
/// numbers. It is to make the file carry ONLY what diverges, so that every
/// future default arrives on its own. `Settings::save` already writes that
/// shape; nothing called it on an existing store, so no store was ever
/// converted.
///
///   config normalize            Print each divergence beside the default it
///                               shadows. Reads nothing else, writes nothing.
///   config normalize --write    Back the store up, rewrite it as overrides,
///                               reload it, and REFUSE (restoring the backup)
///                               unless the reloaded settings are identical.
fn cmd_config_normalize(args: &[String]) -> Result<()> {
    let write = has_flag(args, "--write");
    let settings = resolve_cli_settings(args)?.unwrap_or_else(neoethos_core::Settings::default);
    let provenance = settings.provenance().describe();
    let path = settings
        .provenance()
        .path()
        .map(std::path::Path::to_path_buf);
    let before = settings.overrides_against_defaults()?;

    println!("Config store: {provenance}");
    if let Some(p) = &path {
        let lines = std::fs::read_to_string(p)
            .map(|t| t.lines().count())
            .unwrap_or(0);
        println!("On disk     : {lines} lines");
    }
    println!(
        "Overrides   : {} ({} diverge from default, {} money keys carried by rule)",
        before.len(),
        before.iter().filter(|o| o.diverges).count(),
        before.iter().filter(|o| o.money_key).count()
    );
    println!();
    println!("{:<46} {:<32} {:<32} {}", "key", "your file", "default", "");
    println!("{}", "-".repeat(120));
    for o in &before {
        let default = o.default.as_deref().unwrap_or("<not in schema>");
        let mark = match (o.money_key, o.diverges, o.default.is_none()) {
            (_, _, true) => "LEFTOVER",
            (true, false, _) => "money (= default)",
            (true, true, _) => "MONEY",
            (false, true, _) => "",
            (false, false, _) => "",
        };
        println!(
            "{:<46} {:<32} {:<32} {}",
            truncate(&o.path, 46),
            truncate(&o.live, 32),
            truncate(default, 32),
            mark
        );
    }
    println!();

    if !write {
        println!(
            "Nothing written. Re-run with --write to convert the store to this \
             shape; every key not listed above then follows the compiled default \
             instead of a frozen copy of it."
        );
        return Ok(());
    }

    let Some(path) = path else {
        anyhow::bail!(
            "no config file to normalize — this process resolved to the compiled \
             defaults ({provenance}). There is nothing on disk to convert."
        );
    };

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| anyhow::anyhow!("read the clock: {e}"))?
        .as_secs();
    let backup = path.with_extension(format!("yaml.pre-normalize-{stamp}"));
    std::fs::copy(&path, &backup)
        .map_err(|e| anyhow::anyhow!("back up {} to {}: {e}", path.display(), backup.display()))?;
    println!("Backup      : {}", backup.display());

    settings.save(&path)?;

    // Refuse to leave a rewritten store in place unless it reloads to the same
    // settings. A conversion that changes behaviour is the exact failure this
    // command is meant to end, so it is checked rather than assumed — and on
    // any mismatch the operator's original file is put back before we return.
    let restore = |why: String| -> anyhow::Error {
        let _ = std::fs::copy(&backup, &path);
        anyhow::anyhow!(
            "{why}\nThe original store has been restored from {}.",
            backup.display()
        )
    };
    let reloaded = neoethos_core::Settings::from_yaml(&path)
        .map_err(|e| restore(format!("the normalized store failed to load: {e}")))?;
    let after = reloaded
        .overrides_against_defaults()
        .map_err(|e| restore(format!("the normalized store could not be re-read: {e}")))?;
    if after != before {
        let mut diff = Vec::new();
        for o in &before {
            if !after.contains(o) {
                diff.push(format!("  lost:    {} = {}", o.path, o.live));
            }
        }
        for o in &after {
            if !before.contains(o) {
                diff.push(format!("  gained:  {} = {}", o.path, o.live));
            }
        }
        return Err(restore(format!(
            "the normalized store does not round-trip — {} change(s):\n{}",
            diff.len(),
            diff.join("\n")
        )));
    }

    let lines = std::fs::read_to_string(&path)
        .map(|t| t.lines().count())
        .unwrap_or(0);
    println!(
        "Written     : {} ({lines} lines, was a full snapshot)",
        path.display()
    );
    println!(
        "Verified    : reloads to identical settings; every other key now follows the default."
    );
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n.saturating_sub(1)).collect();
        format!("{head}…")
    }
}

/// Auto search-train loop (P9). Forward-only:
///   import → discover → train → export → next (symbol, timeframe)
///
/// Controls:
///   --symbols X,Y,Z         (default: auto-detect from data root)
///   --timeframes M3,M5,...  (default: ResolvedConfig.timeframes.canonical_default)
///   --skip-training         (run discover + export only)
///   --max-jobs N            (stop after N work-units, 0 = no limit)
///   --resume                (continue from cache/auto_loop_checkpoint.json)
///   --stop-flag PATH        (file whose existence stops the loop after current job)
///
/// Persists checkpoint to cache/auto_loop_checkpoint.json — on crash,
/// re-run with --resume to continue.
/// Multi-GPU / hybrid scheduler driver.
///
/// Enumerates symbol×TF combos, asks `scheduler::plan_combo` how each should be
/// admitted, then runs single-card / CPU combos across all cards concurrently —
/// each as a subprocess `discover` pinned via
/// `NEOETHOS_BOT_SEARCH_EVAL_{WGPU,CUDA}_DEVICE`. Oversized populations are
/// reported as requiring single-device chunking; they are never described as
/// cross-card shards that the worker does not execute.
///
/// `--dry-run` prints the full admission plan WITHOUT spawning — the way to
/// validate the decisions against a real hardware probe + real on-disk data
/// sizes (works on any machine, no GPU needed).
fn cmd_schedule(args: &[String]) -> Result<()> {
    use neoethos_core::scheduler::{
        AdmissionPolicy, ComboItem, ComboShape, WorkScheduler, plan_combo,
    };

    let settings = resolve_cli_settings(args)?.unwrap_or_else(neoethos_core::Settings::default);
    let resolved = neoethos_core::resolved_config::ResolvedConfig::from_settings(&settings);
    let root = parse_root(args, Some(&settings));
    let dry_run = has_flag(args, "--dry-run");
    // Stage 1 schedules DISCOVERY only; training co-scheduling on separate
    // cards is Stage 3, so there is no `--skip-training` knob here yet.
    let resume = has_flag(args, "--resume");
    let stop_flag =
        parse_flag(args, "--stop-flag").unwrap_or_else(|| "cache/schedule_stop.flag".to_string());
    // feature-cube column count is hard to know without building the cube; a
    // conservative estimate is fine because the planner's safety margins
    // (0.75 RAM / 0.80 VRAM / 2× overhead) absorb the error. Overridable.
    let feature_count: usize = parse_flag(args, "--features")
        .and_then(|v| v.parse().ok())
        .unwrap_or(1500);
    let population = resolved.search.population;

    // Set the data root for any child orchestrator BEFORE any thread spawns.
    unsafe {
        std::env::set_var("NEOETHOS_BOT_DATA_ROOT", &root);
    }

    // One full verification pass for the entire scheduling snapshot. The
    // scheduler never materializes OHLCV, but it also never authorizes a row
    // shape from compressed byte size or manifest-only inventory.
    let inventory = neoethos_data::DatasetDiscovery::scan(&root)
        .with_context(|| format!("fully verify canonical schedule inventory under {root}"))?;
    for entry in &inventory.entries {
        print_dataset_inventory_entry(entry)?;
    }
    print_dataset_inventory_rejections(&inventory);
    anyhow::ensure!(
        !inventory.entries.is_empty(),
        "schedule found no fully verified canonical dataset identities under {root}"
    );
    let symbols: Vec<String> = match parse_flag(args, "--symbols") {
        Some(s) if !s.trim().is_empty() => s.split(',').map(|x| x.trim().to_uppercase()).collect(),
        _ => inventory.symbols(),
    };
    let tfs: Vec<String> = parse_flag(args, "--timeframes")
        .unwrap_or_else(|| resolved.timeframes.canonical_default.join(","))
        .split(',')
        .map(|x| x.trim().to_uppercase())
        .collect();

    let mut probe = neoethos_core::system::HardwareProbe::new();
    let hw = probe.detect();
    let policy = AdmissionPolicy::default();
    println!(
        "Hardware: {} cores, {:.0}GB RAM avail, {} detected GPU(s) names={:?} VRAM={:?}GB",
        hw.cpu_cores, hw.available_ram_gb, hw.num_gpus, hw.gpu_names, hw.gpu_mem_gb
    );

    // Build admission plans for the runtime's actual one-device-per-worker
    // contract. Oversized populations remain schedulable only because the
    // evaluator chunks them on that same device.
    let mut id_combo: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();
    let mut schedulable: Vec<ComboItem> = Vec::new();
    for sym in &symbols {
        let identities = verified_canonical_identities(&inventory, sym)?;
        for tf in &tfs {
            let id = format!("{sym}/{tf}");
            let requested_direct = default_higher_tfs_csv(Some(&settings), tf)
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let selection = select_runtime_timeframe_identities(
                &identities,
                sym,
                tf,
                &requested_direct,
            )
            .with_context(|| {
                format!(
                    "schedule direct-timeframe preflight failed for {id}; import/download required"
                )
            })?;
            let rows = schedule_series_rows(&inventory, &root, &selection.base_identity)?;
            let shape = ComboShape::new(rows, population, feature_count);
            let plan = plan_combo(shape, &hw, &policy);
            id_combo.insert(id.clone(), (sym.clone(), tf.clone()));
            schedulable.push(ComboItem::new(id, shape, plan));
        }
    }

    println!(
        "\n=== Admission plan: {} schedulable ===",
        schedulable.len()
    );
    for it in &schedulable {
        let tag = if it.plan.cards_per_combo == 0 {
            "  [CPU lane]"
        } else if !it.plan.fits_on_gpu {
            "  [does NOT fit on one card — chunk]"
        } else {
            ""
        };
        println!(
            "  [{:?}] {:<14} rows={:>9}  assigned_cards={} population/device={} cpu={} RAM≈{:.1}GB VRAM/device≈{:.1}GB{}",
            it.plan.class,
            it.id,
            it.shape.series_rows,
            it.plan.cards_per_combo,
            it.plan.genes_per_card,
            it.plan.cpu_threads_per_worker,
            it.plan.est_ram_per_combo_gb,
            it.plan.est_vram_per_card_gb,
            tag
        );
    }
    if dry_run {
        println!("\n--dry-run: plan only, no processes spawned.");
        return Ok(());
    }

    let checkpoint_path = std::path::PathBuf::from("cache").join("schedule_checkpoint.json");
    let mut completed: Vec<String> = Vec::new();
    if resume && checkpoint_path.exists() {
        if let Ok(text) = std::fs::read_to_string(&checkpoint_path) {
            if let Ok(prev) = serde_json::from_str::<ScheduleCheckpoint>(&text) {
                completed = prev.completed.clone();
                schedulable.retain(|it| !completed.contains(&it.id));
                println!(
                    "Resuming: {} already done; {} remaining",
                    completed.len(),
                    schedulable.len()
                );
            }
        }
    }

    let mut sched = WorkScheduler::new(schedulable, &hw, &policy);
    let mut children: std::collections::HashMap<String, std::process::Child> =
        std::collections::HashMap::new();
    let mut failed: Vec<String> = Vec::new();
    let total = sched.pending_len();
    println!(
        "\nScheduling {} combos across {} card(s)...",
        total,
        sched.total_cards()
    );

    loop {
        let stopping = std::path::Path::new(&stop_flag).exists();
        if stopping && children.is_empty() {
            println!("Stop-flag found and no in-flight work — exiting.");
            break;
        }
        if !stopping {
            for a in sched.poll() {
                let (sym, tf) = id_combo.get(&a.id).cloned().unwrap_or_default();
                match spawn_discover_combo(&a, &sym, &tf, &root, &resolved, &hw) {
                    Ok(child) => {
                        println!(
                            "  ▶ {} on cards {:?} (population/device {}, {} cpu threads)",
                            a.id, a.card_ids, a.genes_per_card, a.cpu_threads
                        );
                        children.insert(a.id.clone(), child);
                    }
                    Err(e) => {
                        eprintln!("  spawn {} failed: {e:#}", a.id);
                        sched.complete(&a.id);
                        failed.push(a.id.clone());
                    }
                }
            }
        }
        if children.is_empty() {
            if sched.is_done() || stopping {
                break;
            }
            // Nothing running and nothing dispatchable — avoid a spin loop.
            eprintln!(
                "scheduler stalled with {} pending and no free capacity — stopping",
                sched.pending_len()
            );
            break;
        }

        let ids: Vec<String> = children.keys().cloned().collect();
        let mut any_done = false;
        for id in ids {
            let finished = match children.get_mut(&id).map(|c| c.try_wait()) {
                Some(Ok(Some(status))) => Some(status),
                Some(Err(e)) => {
                    eprintln!("  wait {id}: {e}");
                    None
                }
                _ => None,
            };
            if let Some(status) = finished {
                any_done = true;
                children.remove(&id);
                sched.complete(&id);
                if status.success() {
                    completed.push(id.clone());
                    println!("  ✔ {id} done ({}/{})", completed.len(), total);
                    write_schedule_checkpoint(&checkpoint_path, &completed);
                } else {
                    // Stage 1 has no CPU lane yet (that is Stage 3), so a GPU
                    // failure is logged and resources freed — NOT retried in a
                    // loop. The OOM->CPU requeue path is exercised once Stage 3
                    // lands a real fast-CPU fallback.
                    failed.push(id.clone());
                    eprintln!(
                        "  ✗ {id} FAILED (exit {:?}) — freed; not retried (CPU lane is Stage 3)",
                        status.code()
                    );
                }
            }
        }
        if !any_done {
            std::thread::sleep(std::time::Duration::from_millis(750));
        }
    }

    println!(
        "\nSchedule done: {} completed, {} failed.",
        completed.len(),
        failed.len()
    );
    if !failed.is_empty() {
        println!("  failed: {failed:?}");
    }
    Ok(())
}

/// Exact, load-free row count for one entry from the already fully verified
/// scheduling snapshot. Re-reading the small manifest is safe only if its
/// generation and binding still equal that snapshot; publication races fail
/// closed and the caller can rerun schedule against a fresh inventory.
fn schedule_series_rows(
    inventory: &neoethos_data::DatasetDiscovery,
    root: impl AsRef<std::path::Path>,
    identity: &neoethos_data::CanonicalDatasetIdentity,
) -> Result<usize> {
    let symbol = identity.symbol_name();
    let timeframe = identity.timeframe();
    let identity_path = identity.to_path_component();
    let matching = inventory
        .entries
        .iter()
        .filter(|entry| entry.dataset_identity == identity_path)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        matching.len() == 1,
        "schedule requires exactly one fully verified canonical dataset identity for {symbol} {timeframe}, found {}; import/download required",
        matching.len()
    );
    let entry = matching[0];
    anyhow::ensure!(
        entry.verification == neoethos_data::DataVerificationStatus::GenerationVerified,
        "schedule inventory entry for {symbol} {timeframe} is not generation-verified"
    );
    let manifest = neoethos_data::core::dataset_manifest::read_current_manifest_metadata(
        root.as_ref(),
        identity,
    )?;
    anyhow::ensure!(
        manifest.generation_id() == entry.generation.as_str()
            && manifest.manifest_binding_sha256() == entry.manifest_binding_sha256.as_str(),
        "canonical generation changed after schedule verification for {symbol} {timeframe}; rerun schedule"
    );
    let rows = usize::try_from(manifest.row_count())
        .with_context(|| format!("manifest row count for {symbol} {timeframe} exceeds usize"))?;
    Ok(rows)
}

/// Spawn one `discover` subprocess pinned to its assigned card. The planner and
/// worker both use the single-device contract (`card_ids[0]`).
fn spawn_discover_combo(
    a: &neoethos_core::scheduler::Assignment,
    symbol: &str,
    timeframe: &str,
    root: &str,
    resolved: &neoethos_core::resolved_config::ResolvedConfig,
    hardware: &neoethos_core::system::HardwareProfile,
) -> Result<std::process::Child> {
    let exe = std::env::current_exe().context("locating current executable")?;
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("discover")
        .arg("--symbol")
        .arg(symbol)
        .arg("--base")
        .arg(timeframe)
        .arg("--root")
        .arg(root)
        .arg("--population")
        .arg(resolved.search.population.to_string())
        .arg("--generations")
        .arg(resolved.search.generations.to_string())
        .arg("--portfolio-size")
        .arg(resolved.search.portfolio_size.to_string())
        .arg("--out")
        .arg(format!("cache/schedule/{symbol}_{timeframe}.json"))
        // The child's CPU budget travels as an ARGUMENT, not as
        // `NEOETHOS_BOT_CPU_BUDGET`. An env var here was indistinguishable
        // from an operator knob and outlived the invocation it was meant for;
        // an argument is scoped to this child and visible in the process list.
        // Read back in `main()` via `parse_flag(&raw_args, "--cpu-threads")`.
        .arg("--cpu-threads")
        .arg(a.cpu_threads.to_string());
    // NOTE: intra-combo GPU sharding (the plural *_DEVICES env) is DISABLED at
    // the dispatch layer — the cubecl wgpu multi-device path panics
    // at runtime ("Memory page 0 doesn't exist" in cubecl-runtime client.rs) and
    // falls back to slow CPU recompute. Validated 2026-06-07 on the 2×A6000 VPS:
    // M1 ran 58 min on CPU (gen 305/20000, 0 results) before we caught it. Until
    // that cubecl multi-device issue is fixed, every combo runs on the PROVEN
    // single-device path, pinned to its first assigned card (H4 validated: card
    // at 120 MiB, clean run). Combo-level throughput (one combo per card,
    // concurrent) is preserved; the multi-device code in eval.rs stays in place,
    // gated behind the plural env which we simply no longer set.
    for (key, value) in gpu_assignment_env(a, hardware) {
        cmd.env(key, value);
    }
    // `NEOETHOS_BOT_DATA_ROOT` still travels as an env var because its reader
    // is `neoethos-models/src/training_orchestrator.rs:329`, in a crate this
    // change does not own. The `--root` argument above already carries the same
    // value to the search; the env copy exists only for the in-process trainer.
    // Collapsing it is a neoethos-models edit, routed, not done here.
    cmd.env("NEOETHOS_BOT_DATA_ROOT", root);
    cmd.spawn()
        .with_context(|| format!("spawning discover for {symbol}/{timeframe}"))
}

fn gpu_assignment_env(
    assignment: &neoethos_core::scheduler::Assignment,
    hardware: &neoethos_core::system::HardwareProfile,
) -> std::collections::BTreeMap<&'static str, String> {
    use neoethos_core::system::AcceleratorBackend;

    let mut envs = std::collections::BTreeMap::new();
    let Some(&slot) = assignment.card_ids.first() else {
        return envs;
    };
    if let Some(device) = hardware.accelerator_devices.get(slot) {
        match device.backend {
            AcceleratorBackend::Cuda => {
                envs.insert(
                    "NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE",
                    device.backend_index.to_string(),
                );
            }
            backend if backend.is_wgpu_family() => {
                if let Some(selector) = device.cubecl_wgpu_selector() {
                    envs.insert("NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE", selector);
                }
            }
            AcceleratorBackend::Cpu | AcceleratorBackend::Rocm => {}
            _ => {}
        }
        return envs;
    }

    // Backwards compatibility for synthetic/legacy profiles that contain only
    // `gpu_mem_gb` and therefore cannot describe a backend or adapter class.
    envs.insert("NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE", slot.to_string());
    envs.insert("NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE", slot.to_string());
    envs
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct ScheduleCheckpoint {
    updated_at: String,
    completed: Vec<String>,
}

fn write_schedule_checkpoint(path: &std::path::Path, completed: &[String]) {
    let ck = ScheduleCheckpoint {
        updated_at: chrono::Utc::now().to_rfc3339(),
        completed: completed.to_vec(),
    };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(text) = serde_json::to_string_pretty(&ck) {
        let _ = std::fs::write(path, text);
    }
}

fn cmd_auto_loop(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?.unwrap_or_else(neoethos_core::Settings::default);
    let resolved = neoethos_core::resolved_config::ResolvedConfig::from_settings(&settings);
    let root = parse_root(args, Some(&settings));

    let symbols_raw = parse_flag(args, "--symbols").unwrap_or_default();
    let tfs_raw = parse_flag(args, "--timeframes")
        .unwrap_or_else(|| resolved.timeframes.canonical_default.join(","));
    let skip_training = has_flag(args, "--skip-training");
    let max_jobs: usize = parse_flag(args, "--max-jobs")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let resume = has_flag(args, "--resume");
    let stop_flag =
        parse_flag(args, "--stop-flag").unwrap_or_else(|| "cache/auto_loop_stop.flag".to_string());
    let checkpoint_path = std::path::PathBuf::from("cache").join("auto_loop_checkpoint.json");

    let symbols: Vec<String> = if symbols_raw.is_empty() {
        metadata_inventory_symbols(&root)?
    } else {
        symbols_raw
            .split(',')
            .map(|s| s.trim().to_uppercase())
            .collect()
    };
    let tfs: Vec<String> = tfs_raw
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .collect();

    // Build the (symbol, timeframe) work queue.
    let mut work_queue: Vec<(String, String)> = symbols
        .iter()
        .flat_map(|s| tfs.iter().map(move |t| (s.clone(), t.clone())))
        .collect();
    let total_units = work_queue.len();

    // Resume support: read checkpoint and skip already-completed pairs.
    let mut completed: Vec<(String, String)> = Vec::new();
    if resume && checkpoint_path.exists() {
        if let Ok(text) = std::fs::read_to_string(&checkpoint_path) {
            if let Ok(prev) = serde_json::from_str::<AutoLoopCheckpoint>(&text) {
                completed = prev.completed.clone();
                work_queue.retain(|w| !completed.contains(w));
                println!(
                    "Resuming from checkpoint: {} already completed; {} remaining",
                    completed.len(),
                    work_queue.len()
                );
            }
        }
    }

    let mut jobs_run = 0usize;
    println!(
        "Auto-loop start: {} work units ({} symbols × {} timeframes); skip_training={}; stop-flag={}",
        work_queue.len(),
        symbols.len(),
        tfs.len(),
        skip_training,
        stop_flag
    );

    for (sym, tf) in work_queue.into_iter() {
        if std::path::Path::new(&stop_flag).exists() {
            println!("Stop-flag found at {} — exiting loop", stop_flag);
            break;
        }
        if max_jobs > 0 && jobs_run >= max_jobs {
            println!("Reached --max-jobs={}; exiting", max_jobs);
            break;
        }

        println!(
            "[{}/{}] discovering {} {}",
            jobs_run + 1,
            total_units,
            sym,
            tf
        );
        let discover_args: Vec<String> = vec![
            "discover".to_string(),
            "--symbol".to_string(),
            sym.clone(),
            "--base".to_string(),
            tf.clone(),
            // 2026-06-04 parity: no hardcoded `--higher H4`. Omitting `--higher`
            // lets cmd_discover resolve the higher-TF ladder from config relative
            // to THIS base (`tf`) via the shared `resolve_higher_timeframes`, so
            // every base in the sweep gets its correct top-down context — same as
            // a standalone `discover` run (H4-as-base no longer self-references).
            "--root".to_string(),
            root.clone(),
            "--population".to_string(),
            resolved.search.population.to_string(),
            "--generations".to_string(),
            resolved.search.generations.to_string(),
            "--portfolio-size".to_string(),
            resolved.search.portfolio_size.to_string(),
            "--out".to_string(),
            format!("cache/auto_loop/{}_{}.json", sym, tf),
        ];
        let discover_ok = match cmd_discover(&discover_args) {
            Ok(()) => {
                println!("  discover OK");
                true
            }
            Err(err) => {
                eprintln!("  discover FAILED: {err:#}");
                // Continue to next; don't bail the whole loop.
                false
            }
        };

        let mut train_ok = true;
        if !skip_training {
            // NEOETHOS_BOT_DATA_ROOT was set at the top of `cmd_auto_loop`
            // before any thread spawned; cmd_train reads it via
            // training_orchestrator::train_symbol.
            let train_args: Vec<String> = vec![
                "train".to_string(),
                "--symbol".to_string(),
                sym.clone(),
                "--base".to_string(),
                tf.clone(),
                "--models-dir".to_string(),
                "cache/auto_loop_models".to_string(),
                "--root".to_string(),
                root.clone(),
            ];
            match cmd_train(&train_args) {
                Ok(()) => println!("  train OK"),
                Err(err) => {
                    eprintln!("  train FAILED: {err:#}");
                    train_ok = false;
                }
            }
        }

        // Audit B14 (2026-07-13): mark this combo COMPLETE only when its
        // stages actually succeeded, so `--resume` RETRIES failed work rather
        // than silently skipping it. Previously `completed.push` ran
        // unconditionally, so a transient discovery/training failure (OOM, bad
        // data, a crash mid-run) was checkpointed as "done" and the user
        // permanently lost those strategies on resume.
        if discover_ok && train_ok {
            completed.push((sym.clone(), tf.clone()));
        } else {
            eprintln!(
                "  [{sym} {tf}] NOT marked complete (discover_ok={discover_ok}, \
                 train_ok={train_ok}) — will retry on --resume"
            );
        }
        let checkpoint = AutoLoopCheckpoint {
            started_at: completed
                .first()
                .map(|_| chrono::Utc::now().to_rfc3339())
                .unwrap_or_default(),
            updated_at: chrono::Utc::now().to_rfc3339(),
            completed: completed.clone(),
            remaining: total_units.saturating_sub(completed.len()),
        };
        if let Some(dir) = checkpoint_path.parent() {
            if let Err(err) = std::fs::create_dir_all(dir) {
                tracing::warn!(
                    target: "neoethos_cli",
                    dir = %dir.display(),
                    error = %err,
                    "auto_loop: failed to create checkpoint directory"
                );
            }
        }
        match serde_json::to_string_pretty(&checkpoint) {
            Ok(text) => {
                if let Err(err) = std::fs::write(&checkpoint_path, text) {
                    tracing::warn!(
                        target: "neoethos_cli",
                        path = %checkpoint_path.display(),
                        error = %err,
                        "auto_loop: failed to write checkpoint"
                    );
                }
            }
            Err(err) => {
                tracing::warn!(
                    target: "neoethos_cli",
                    error = %err,
                    "auto_loop: failed to serialize checkpoint"
                );
            }
        }

        jobs_run += 1;
    }

    println!(
        "Auto-loop done: {}/{} work units processed; checkpoint at {}",
        completed.len(),
        total_units,
        checkpoint_path.display()
    );
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct AutoLoopCheckpoint {
    started_at: String,
    updated_at: String,
    completed: Vec<(String, String)>,
    remaining: usize,
}

fn cmd_import(args: &[String]) -> Result<()> {
    validate_import_arguments(args)?;
    let source = required_import_flag(args, "--source")?;
    let source = std::path::PathBuf::from(source);
    let source = if source.is_absolute() {
        source
    } else {
        std::env::current_dir()
            .context("resolve current directory for relative import source")?
            .join(source)
    };

    // Discovery is metadata-only. It may suggest labels from paths/extensions,
    // but it never publishes, converts, or supplies implicit values to a real
    // import request.
    if has_flag(args, "--dry-run") {
        return print_import_discovery_summary(&source);
    }

    let source_format = required_import_flag(args, "--format")?
        .parse::<neoethos_data::core::import_provenance::ImportSourceFormat>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let source_namespace = required_import_flag(args, "--source-namespace")?;
    let symbol = required_import_flag(args, "--symbol")?;
    let timeframe = required_import_flag(args, "--timeframe")?
        .parse::<neoethos_data::CanonicalTimeframe>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let bar_timestamp_convention = required_import_flag(args, "--bar-timestamps")?
        .parse::<neoethos_data::BarTimestampConvention>()
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if !bar_timestamp_convention.is_canonical_bar_open() {
        anyhow::bail!(
            "--bar-timestamps={} cannot become canonical; only explicitly evidenced bar_open timestamps are accepted",
            bar_timestamp_convention
        );
    }
    let expected_generation = unique_import_flag(args, "--expected-generation")?;

    let settings = resolve_cli_settings(args)?;
    let root = unique_import_flag(args, "--root")?
        .map(std::path::PathBuf::from)
        .or_else(|| {
            settings
                .as_ref()
                .map(|settings| settings.system.data_dir.clone())
        })
        .unwrap_or_else(|| std::path::PathBuf::from("data"));
    let identity = neoethos_data::CanonicalDatasetIdentity::external(
        source_namespace.trim(),
        symbol.trim(),
        timeframe,
        bar_timestamp_convention,
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;

    let installed = neoethos_core::execution_budget::installed_process_budget()
        .context("import unavailable before the process CPU budget is installed")?;
    let auxiliary_limit = neoethos_core::execution_budget::AuxiliarySlotLimit::new(
        neoethos_data::source_seal_slot_limit(),
    )?;
    let admission = neoethos_core::execution_budget::CompositeAdmissionAuthority::new(
        installed.broker().clone(),
        auxiliary_limit,
    );
    let grant = admission.acquire(
        neoethos_core::execution_budget::CompositeAdmissionRequest::new(
            neoethos_core::execution_budget::CpuPermitRequest::local(
                installed.resolved().effective_worker_limit,
            ),
            neoethos_core::execution_budget::AuxiliarySlotRequest::One,
        ),
    )?;
    let (cpu_lease, auxiliary_slot) = grant.into_parts();
    let auxiliary_slot = auxiliary_slot
        .context("composite import admission returned no SourceSeal auxiliary slot")?;
    let limits = neoethos_data::core::import_limits::ImportLimits::default();
    let imported = cpu_lease.scope(|| -> Result<_> {
        let imported = neoethos_data::core::import_service::import_path_to_vortex(
            neoethos_data::core::import_service::ImportRequest {
                source_path: &source,
                configured_root: &root,
                identity: &identity,
                declared_format: source_format,
                expected_generation: expected_generation.as_deref(),
                limits: &limits,
                auxiliary_slot: &auxiliary_slot,
            },
        )?;
        let provenance = imported.provenance();
        if provenance.dataset_identity() != &identity
            || provenance.selected_format() != source_format
            || provenance.detected_format() != source_format
        {
            anyhow::bail!("reopened canonical import provenance disagrees with the request");
        }
        Ok(imported)
    })?;
    let manifest = imported.manifest();
    let provenance = imported.provenance();

    println!("Canonical Vortex import committed and reopened successfully");
    println!("  source:             {}", source.display());
    println!("  declared format:    {}", source_format);
    println!("  dataset identity:   {}", identity.to_path_component());
    println!("  generation:         {}", imported.generation());
    println!("  durable commit:     {}", imported.durable_commit_id());
    println!("  rows:               {}", imported.row_count());
    let source_sha256 = provenance
        .source_sha256()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    println!("  source sha256:      {source_sha256}");
    println!(
        "  canonical Vortex:   {}",
        manifest.generation_path().display()
    );
    println!("  vortex sha256:      {}", manifest.vortex_sha256());
    Ok(())
}

fn validate_import_arguments(args: &[String]) -> Result<()> {
    const VALUE_FLAGS: &[&str] = &[
        "--source",
        "--format",
        "--source-namespace",
        "--symbol",
        "--timeframe",
        "--bar-timestamps",
        "--expected-generation",
        "--root",
        "--config",
    ];
    const BOOLEAN_FLAGS: &[&str] = &["--dry-run"];
    let mut index = 0;
    while index < args.len() {
        let argument = args[index].as_str();
        if BOOLEAN_FLAGS.contains(&argument) {
            index += 1;
            continue;
        }
        if VALUE_FLAGS.contains(&argument) {
            let value = args
                .get(index + 1)
                .ok_or_else(|| anyhow::anyhow!("import {argument} requires a non-empty value"))?;
            if value.trim().is_empty() || value.starts_with("--") {
                anyhow::bail!("import {argument} requires a non-empty value");
            }
            index += 2;
            continue;
        }
        anyhow::bail!("unknown import argument `{argument}`; refusing to infer its meaning");
    }
    for flag in VALUE_FLAGS {
        let _ = unique_import_flag(args, flag)?;
    }
    Ok(())
}

fn unique_import_flag(args: &[String], name: &str) -> Result<Option<String>> {
    let mut values = Vec::new();
    let mut index = 0;
    while index < args.len() {
        if args[index] == name {
            let value = args
                .get(index + 1)
                .ok_or_else(|| anyhow::anyhow!("import {name} requires a value"))?;
            values.push(value.trim().to_owned());
            index += 2;
        } else {
            index += 1;
        }
    }
    match values.len() {
        0 => Ok(None),
        1 => Ok(values.pop()),
        count => anyhow::bail!("import {name} was supplied {count} times"),
    }
}

fn required_import_flag(args: &[String], name: &str) -> Result<String> {
    unique_import_flag(args, name)?.ok_or_else(|| {
        anyhow::anyhow!(
            "import requires {name}; source identity and schema are never inferred for publication"
        )
    })
}

fn print_import_discovery_summary(source: &std::path::Path) -> Result<()> {
    let report = neoethos_data::ImportDiscovery::scan(source)?;
    println!("Import discovery only — no bytes were converted or published");
    println!("  scanned: {}", report.root.display());
    println!("  candidates: {}", report.entries.len());
    for entry in &report.entries {
        println!(
            "  {}  suggested-format={}  suggested-symbol={}  suggested-timeframe={}  bytes={}",
            entry.path.display(),
            entry.format,
            entry.symbol.as_deref().unwrap_or("<declare --symbol>"),
            entry
                .timeframe
                .as_deref()
                .unwrap_or("<declare --timeframe>"),
            entry.size_bytes
        );
    }
    for skipped in &report.skipped {
        println!(
            "  SKIPPED {} ({:?})",
            skipped.path.display(),
            skipped.reason
        );
    }
    println!(
        "Run a real import with one file and explicit --format, --source-namespace, --symbol, --timeframe, and --bar-timestamps bar_open."
    );
    Ok(())
}

fn cmd_stop_target(args: &[String]) -> Result<()> {
    let settings = resolve_cli_settings(args)?;
    let root = parse_root(args, settings.as_ref());
    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(settings.as_ref()));
    let timeframe =
        parse_flag(args, "--timeframe").unwrap_or_else(|| default_base_tf(settings.as_ref()));
    let pip_size: f64 = parse_flag(args, "--pip")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0001);
    let signal: i8 = parse_flag(args, "--signal")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let identities = inventory_canonical_identities(&root, &symbol)?;
    let identity = select_exact_runtime_identity(&identities, args, &symbol, &timeframe)?;
    let canonical = load_exact_runtime_timeframe(&root, &identity)?;
    let ohlcv = canonical.ohlcv();
    let settings = neoethos_search::StopTargetSettings::default();
    let result = neoethos_search::infer_stop_target_pips(
        &ohlcv.open,
        &ohlcv.high,
        &ohlcv.low,
        &ohlcv.close,
        &settings,
        pip_size,
        signal,
    );
    if let Some((sl, tp, rr)) = result {
        println!(
            "Stop/Target {} {}: SL={:.2} pips TP={:.2} pips RR={:.2}",
            symbol, timeframe, sl, tp, rr
        );
    } else {
        println!("Stop/Target {} {}: insufficient data", symbol, timeframe);
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// `autoresearch` — the goal-driven loop's operator entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Every flag `autoresearch` accepts. Anything else is REFUSED by name.
///
/// The allow-list is the point, not a convenience. `RunArgs` deliberately has no
/// field naming a goal constant, a cost field or a judge threshold — *"if there
/// were, the loop's goals would be writable from the command line, which is the
/// same defect wearing a different hat"*. A CLI that silently ignored
/// `--target-balance 999999` would give an operator every reason to believe he
/// had set one, which is worse than not offering it: the loop would run toward
/// the config's goal while its author believed it was running toward his.
const AUTORESEARCH_FLAGS: &[&str] = &[
    "--symbol",
    "--max-sweeps",
    "--max-hours",
    "--session",
    "--scenario",
    "--dry-run",
    // Set by the schedule orchestrator on the children it spawns; it is parsed
    // in `main` from the full argv and would otherwise look unknown here.
    "--cpu-threads",
    "--help",
    "-h",
];

/// `neoethos-cli autoresearch` — start or resume the goal-driven research loop.
///
/// **THE ENTRY POINT THAT DID NOT EXIST.** `neoethos_autoresearch::runner::run`
/// was invoked from nowhere in the workspace — no CLI subcommand, no UI route,
/// no caller outside its own crate — so the loop the operator asked for ("time
/// for the karpathy loop, so that you are not needed at all; it matters that the
/// bot runs on the GOAL and not on a human") could not be started by a human
/// either.
///
/// ## One config, no env, no second config file
///
/// The settings come from `Settings::load()` and from nothing else. `--config`
/// is REFUSED rather than honoured: the loop freezes the goal set, the judge and
/// the cost model into a `session_id` at S0 and verifies all three on every
/// resume, so a session started against one file and resumed against another
/// would be refused halfway through a multi-day run — or, worse, resumed under a
/// goal the earlier sweeps were never optimised toward.
///
/// ## What it will not do
///
/// It never places an order, never contacts a broker and never writes
/// `live_portfolio.json`. On success it writes a PROPOSAL beside the verdict and
/// stops. The operator promotes.
fn cmd_autoresearch(args: &[String]) -> Result<()> {
    if has_flag(args, "--help") || has_flag(args, "-h") {
        autoresearch_help();
        return Ok(());
    }

    // Checked BEFORE the allow-list, so the operator who reaches for the flag
    // every other subcommand has gets the reason rather than a list.
    if has_flag(args, "--config") {
        anyhow::bail!(
            "`autoresearch` reads the ONE config — the same file every other reader in this \
             process resolves — and has no --config override. A session freezes its goal_hash, \
             judge_hash and cost_hash at open and refuses to resume when any of them moves, so a \
             run started against a second config file would be refused partway through or, worse, \
             resumed under a goal its earlier sweeps were never optimised toward."
        );
    }

    // Refuse an unknown flag rather than ignore it.
    for arg in args {
        if arg.starts_with("--")
            && !AUTORESEARCH_FLAGS.contains(&arg.as_str())
            && arg != "--dataset-identity"
        {
            anyhow::bail!(
                "`autoresearch` does not accept {arg}. Accepted: {}.\n\n\
                 There is deliberately NO flag naming a goal constant, a cost field or a judge \
                 threshold. The loop optimises TOWARD the goals in your config and can never \
                 rewrite them, and a flag that let the command line move a goal would be that \
                 rule broken from the outside. Change the value in your config.yaml and start a \
                 NEW session — a session's goal_hash, judge_hash and cost_hash are frozen into \
                 its id at open and verified on every resume.",
                format!("{} --dataset-identity", AUTORESEARCH_FLAGS.join(" "))
            );
        }
    }
    let settings = neoethos_core::Settings::load().context(
        "loading the operator's config for the autoresearch loop. The loop's GOALS live in it \
         (system.risky_* or risk.monthly_profit_target_pct + models.prop_firm_min_pass_rate), so \
         there is nothing to optimise toward without it — and built-in defaults are not your \
         settings.",
    )?;

    let symbol = parse_flag(args, "--symbol").unwrap_or_else(|| default_symbol(Some(&settings)));
    let base_timeframe = default_base_tf(Some(&settings));
    let identities = inventory_canonical_identities(&settings.system.data_dir, &symbol)?;
    let dataset_identity =
        select_exact_runtime_identity(&identities, args, &symbol, &base_timeframe)
            .context("selecting the exact canonical dataset for the autoresearch loop")?;
    let mut run_args = neoethos_autoresearch::RunArgs::new(dataset_identity);
    run_args.dry_run = has_flag(args, "--dry-run");
    run_args.session = parse_flag(args, "--session");
    run_args.scenario = parse_flag(args, "--scenario");
    if let Some(raw) = parse_flag(args, "--max-sweeps") {
        run_args.max_sweeps = match raw.trim().parse::<usize>() {
            Ok(v) if v > 0 => v,
            _ => anyhow::bail!(
                "--max-sweeps expects a positive integer, got `{raw}`. Refusing to guess a budget \
                 for a run that can last days."
            ),
        };
    }
    if let Some(raw) = parse_flag(args, "--max-hours") {
        run_args.max_hours = match raw.trim().parse::<f64>() {
            Ok(v) if v.is_finite() && v > 0.0 => v,
            _ => anyhow::bail!(
                "--max-hours expects a positive number of hours, got `{raw}`. Refusing to guess."
            ),
        };
    }

    println!(
        "autoresearch: identity {} | symbol {} | base {} | max_sweeps {} | max_hours {} | session {} | scenario {}{}",
        run_args.dataset_identity.to_path_component(),
        run_args.dataset_identity.symbol_name(),
        run_args.dataset_identity.timeframe(),
        run_args.max_sweeps,
        if run_args.max_hours.is_finite() {
            format!("{:.2}", run_args.max_hours)
        } else {
            "unbounded".to_string()
        },
        run_args.session.as_deref().unwrap_or("<new>"),
        run_args
            .scenario
            .as_deref()
            .unwrap_or("<the goal set's primary>"),
        if run_args.dry_run { " | DRY RUN" } else { "" }
    );
    println!(
        "The loop PROPOSES. It never places an order, never contacts the broker and never writes \
         live_portfolio.json — its outputs are a verdict and, on success, a proposal, both under \
         the autoresearch store."
    );

    let verdict = neoethos_autoresearch::runner::run(run_args, &settings)
        .context("running the autoresearch loop")?;

    println!("{}", verdict.render());
    match neoethos_autoresearch::SessionStore::root() {
        Ok(root) => println!("artifacts: {}", root.join(&verdict.session_id).display()),
        Err(err) => println!("artifacts: <store root unresolved: {err:#}>"),
    }
    Ok(())
}

fn autoresearch_help() {
    println!("neoethos-cli autoresearch — the goal-driven research loop");
    println!();
    println!("USAGE:");
    println!("    neoethos-cli autoresearch [--symbol EURUSD] [--max-sweeps N] [--max-hours H]");
    println!("                              [--session <id>] [--scenario <label>] [--dry-run]");
    println!();
    println!("    --symbol      Instrument to search. Defaults to the config's symbol.");
    println!("    --max-sweeps  Sweep budget; one sweep is 100 searches. Default 200.");
    println!("    --max-hours   Wall-clock budget. Unbounded when omitted.");
    println!("    --session     Resume an existing session id. A new id when omitted.");
    println!("    --scenario    Optimise toward this scenario label instead of the primary.");
    println!("                  An unknown label is REFUSED — it never falls back.");
    println!("    --dry-run     Draw and stamp one sweep's proposals; run no search.");
    println!();
    println!("    There is NO flag for a goal, a cost or a judge threshold. The loop optimises");
    println!("    toward the constants in your config.yaml and can never rewrite them; a session");
    println!("    freezes their hashes into its id and refuses to resume when any of them moves.");
    println!();
    println!("    The loop PROPOSES. It writes a verdict and, on success, a proposal, under the");
    println!("    autoresearch store. It never places an order and never touches the broker.");
}

/// `neoethos-cli wizard` — TUI counterpart of the desktop first-run
/// wizard. Spec §8 (`installer_wizard_ux_spec.md`).
fn cmd_wizard(_args: &[String]) -> Result<()> {
    tui::run_wizard_tui()
}

/// `neoethos-cli setup` — Task #61 headless setup helper. Closes the
/// CLI parity gap: prints canonical credentials paths, shows which
/// config files exist on disk, and emits ready-to-paste TOML / JSON
/// templates for the operator to scp into place on a headless host.
///
/// Sub-modes:
///   `neoethos-cli setup`             — same as `setup show`
///   `neoethos-cli setup show`        — list expected paths + existence
///   `neoethos-cli setup ctrader`     — print broker_credentials.toml template
///   `neoethos-cli setup paths`       — print just the canonical directories
///
/// We intentionally do NOT write binary state here — the on-disk
/// schemas live in `neoethos-app::app_services` which the CLI crate
/// can't depend on (creates a cycle). Operators paste the template
/// into the canonical path manually OR drive the egui wizard once
/// on a desktop and `scp` the resulting `broker_credentials.toml`
/// to the headless host.
fn cmd_setup(args: &[String]) -> Result<()> {
    let mode = args.first().map(String::as_str).unwrap_or("show");
    match mode {
        "show" => setup_show(),
        "ctrader" => setup_ctrader_template(),
        "paths" => setup_paths(),
        "--help" | "-h" | "help" => {
            setup_help();
            Ok(())
        }
        other => {
            eprintln!(
                "neoethos-cli setup: unknown sub-mode '{other}'. \
                 Try 'neoethos-cli setup --help'."
            );
            setup_help();
            Ok(())
        }
    }
}

fn setup_help() {
    println!("neoethos-cli setup — headless credentials helper");
    println!();
    println!("USAGE:");
    println!("    neoethos-cli setup [SUBCOMMAND]");
    println!();
    println!("SUBCOMMANDS:");
    println!("    show       List expected config paths + which already exist (default)");
    println!("    paths      Print just the canonical directories, one per line (scripting)");
    println!("    ctrader    Emit a broker_credentials.toml template for the cTrader broker");
    println!();
    println!("    The CLI does NOT write binary state — paste the template into the canonical");
    println!("    path printed by `setup paths`. Drive the egui wizard once on a desktop if you");
    println!("    prefer a graphical flow, then `scp` the resulting `broker_credentials.toml`.");
}

/// Canonical user-config directory — matches the resolution in
/// `neoethos-app::broker_persistence::credentials_file_path` exactly so
/// `neoethos-cli setup` prints the same paths the GUI writes to.
/// Order: env override → `dirs::config_dir()/neoethos` → `.local/neoethos`.
fn canonical_user_config_dir() -> std::path::PathBuf {
    // Test-seam env var: matches `BROKER_CREDENTIALS_PATH_ENV_VAR` in
    // neoethos-app so an operator running a sandboxed CLI session sees
    // the same override path the GUI does.
    // **F-CORE3 closure (2026-05-25)**: routed through the canonical
    // `neoethos_core::env_overrides::broker_credentials_path_override`
    // typed getter — single grep-able source for the env-var name.
    if let Some(custom) = neoethos_core::env_overrides::broker_credentials_path_override()
        && let Some(parent) = std::path::Path::new(&custom).parent()
    {
        return parent.to_path_buf();
    }
    if let Some(config_dir) = dirs::config_dir() {
        return config_dir.join("neoethos");
    }
    // Last-resort dev-machine fallback — mirrors the third candidate
    // in `neoethos-app::broker_persistence::candidate_paths`.
    std::path::PathBuf::from(".local/neoethos")
}

fn setup_show() -> Result<()> {
    let config_dir = canonical_user_config_dir();
    println!("NeoEthos headless setup status");
    println!("==============================");
    println!();
    println!("Canonical config directory:");
    println!("  {}", config_dir.display());
    if !config_dir.exists() {
        println!("    ! directory does not exist yet — `mkdir -p` it before pasting templates");
    }
    println!();
    let entries: &[(&str, &str)] = &[
        (
            "broker_credentials.toml",
            "cTrader OAuth credentials (client_id, redirect_uri, accounts, environment)",
        ),
        (
            "risky_mode_state.json",
            "Risky Mode arm + ack ledger (written by the desktop wizard's Apply step)",
        ),
        (
            "wizard_state.json",
            "Wizard completion sentinel + per-step status (resume-from-disk hint)",
        ),
        (
            "risk_acknowledgement.json",
            "Append-only ledger of the 5-question risk-quiz acknowledgements (Task #68)",
        ),
    ];
    println!("Expected files:");
    for (name, description) in entries {
        let path = config_dir.join(name);
        let mark = if path.exists() { "✓" } else { "·" };
        println!("  [{}] {}", mark, path.display());
        println!("      {}", description);
    }
    println!();
    println!("Run `neoethos-cli setup ctrader` for a paste-ready cTrader template.");
    Ok(())
}

fn setup_paths() -> Result<()> {
    let dir = canonical_user_config_dir();
    println!("{}", dir.display());
    Ok(())
}

fn setup_ctrader_template() -> Result<()> {
    let dir = canonical_user_config_dir();
    let path = dir.join("broker_credentials.toml");
    println!("# Paste this into:");
    println!("#   {}", path.display());
    println!("# Replace the placeholder values with the credentials from the cTrader Open API");
    println!("# Developer Portal (https://openapi.ctrader.com). For accounts with");
    println!("# `enabled_for_execution = true`, the bot will route orders. Leaving the");
    println!("# array empty is fine — the GUI's account-discovery step populates it.");
    println!();
    println!("schema_version = 1");
    println!();
    println!("[ctrader]");
    println!("environment = \"Demo\"  # or \"Live\" — match the cTrader account's tier");
    println!("client_id = \"<your cTrader app client_id>\"");
    println!("client_secret = \"<your cTrader app client_secret>\"");
    println!("redirect_uri = \"http://127.0.0.1:43001/callback\"");
    println!("accounts = []");
    Ok(())
}

// REMOVED 2026-08-09: `setup news` printed a `news_api.toml` template and told
// the operator that "the news API key drives ... [the] blackout filter".
// Nothing in this repo has ever read `news_api.toml`, and batch D3 deleted
// `neoethos_core::domain::news_filter` — the only type that ever held a news
// API key — together with the `news.perplexity_enabled`,
// `news.news_lookahead_minutes` and `news.news_kill_window_min` knobs.
//
// The LIVE news gate is `neoethos_app::app_services::news_calendar`, whose
// blackout window is HARDCODED (15 min before / 10 min after) and whose only
// operator knob is `news.news_trading_mode`. If that window should become
// settable, add ONE knob and give it a reader in `news_calendar` — do not
// restore a command that advertises a capability with no implementation.

fn parse_root(args: &[String], settings: Option<&neoethos_core::Settings>) -> String {
    resolve_root(args, settings)
}

fn resolve_root(args: &[String], settings: Option<&neoethos_core::Settings>) -> String {
    // `--data-path` is the operator-facing canonical manifest-root flag;
    // `--root` remains for existing scripts. `--data-path` wins when both are
    // supplied because it is the explicit inventory root.
    if let Some(p) = parse_flag(args, "--data-path") {
        return p;
    }
    parse_flag(args, "--root").unwrap_or_else(|| {
        settings
            .map(|settings| settings.system.data_dir.to_string_lossy().to_string())
            .unwrap_or_else(|| "data".to_string())
    })
}

/// Run bounded manifest-only inventory on the supplied root and print every
/// exact identity/generation plus every rejected path. This is status output,
/// never authorization to consume a generation; runtime readers still perform
/// full hash/footer/timestamp verification.
///
/// Shell-completion hint: when this codebase migrates to clap-derive,
/// the `--data-path` argument should be annotated with
/// `value_hint = clap::ValueHint::DirPath` so shells that respect the
/// hint can complete directory paths. Today the CLI uses manual arg
/// parsing, so the hint is documented here as a future-work marker.
fn print_dataset_discovery_summary(root: &str) -> Result<neoethos_data::DatasetDiscovery> {
    let report = neoethos_data::DatasetDiscovery::scan_metadata(root)?;
    println!("Scanned: {}", report.root.display());
    if report.is_empty() && report.skipped.is_empty() {
        // Real-data only: never silently fall back to a packaged demo
        // dataset. Surface the empty result so the operator can pick
        // a different folder.
        println!("  (no canonical manifest identities or rejected input paths found)");
        return Ok(report);
    }

    let total = report.entries.len();
    let format_breakdown: Vec<String> = report
        .format_counts()
        .into_iter()
        .map(|(fmt, n)| format!("{}: {}", fmt.as_str(), n))
        .collect();
    println!(
        "Canonical identities: {} ({})",
        total,
        format_breakdown.join(", ")
    );

    let symbols = report.symbols();
    let symbols_preview: String = if symbols.len() > 6 {
        format!(
            "{}, ...",
            symbols
                .iter()
                .take(6)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        symbols.join(", ")
    };
    println!(
        "Exact symbols:      {}  ({})",
        symbols.len(),
        symbols_preview
    );

    let tfs = report.timeframes();
    println!("Timeframes:         {}", tfs.join(", "));

    for entry in &report.entries {
        print_dataset_inventory_entry(entry)?;
    }

    if !report.skipped.is_empty() {
        let buckets = report.skip_counts_by_category();
        // Per-bucket detail: e.g. "import_required: explicit import ... (x4)".
        let mut detail_parts: Vec<String> = Vec::new();
        for (cat, count) in &buckets {
            let example_labels: Vec<String> = report
                .skipped
                .iter()
                .filter(|s| s.reason.category() == cat)
                .filter_map(|s| match &s.reason {
                    neoethos_data::SkipReason::ImportRequired(detail)
                    | neoethos_data::SkipReason::RetiredLayout(detail)
                    | neoethos_data::SkipReason::InvalidCanonicalIdentity(detail)
                    | neoethos_data::SkipReason::UnverifiedGeneration(detail)
                    | neoethos_data::SkipReason::Unreadable(detail) => Some(detail.clone()),
                })
                .collect();
            let mut uniq: Vec<String> = example_labels;
            uniq.sort();
            uniq.dedup();
            let labels = if uniq.is_empty() {
                "".to_string()
            } else {
                format!(": {}", uniq.join(", "))
            };
            detail_parts.push(format!("{count} {cat}{labels}"));
        }
        println!(
            "Skipped:            {}   ({})",
            report.skipped.len(),
            detail_parts.join("; ")
        );
        print_dataset_inventory_rejections(&report);
    }

    Ok(report)
}

fn resolve_cli_settings(args: &[String]) -> Result<Option<neoethos_core::Settings>> {
    if let Some(config_path) = parse_flag(args, "--config") {
        return neoethos_core::Settings::from_yaml(&config_path).map(Some);
    }

    // Without --config, resolve the way the rest of the process does.
    //
    // This checked the working directory for `config.yaml` and stopped there,
    // while `Settings::load()` — which installs the runtime overrides at
    // startup, main.rs:18 — prefers the user config under %LOCALAPPDATA% and
    // only falls back to the relative path. Run from the repo and the two
    // disagreed inside one process.
    //
    // They are not variants of each other. Measured on this machine
    // (2026-08-01): the user config says trading_mode prop_firm, preset ftmo,
    // prop_search_device auto; the repo template, last touched two weeks
    // earlier, says risky, none, and `cpu` — the line recorded as the cause of
    // eight months of discovery never reaching the card. Which one a run got
    // depended on the directory it was started from.
    //
    // `load()` still ends at the relative path, so a workspace checkout with no
    // user config behaves exactly as before.
    neoethos_core::Settings::load().map(Some)
}

// `replay_engine_config` MOVED 2026-08-10 (#229) to
// `neoethos_trader::EngineConfig::try_for_replay_from_settings`, and DELETED here.
//
// It was private to this binary, so `neoethos-app`'s `POST /autonomous/replay`
// could not call it and passed `EngineConfig::default()` instead: zero spread,
// zero slippage, zero commission, on the synthetic 10 000 balance. One
// front-end charged the operator's real broker costs and the other charged
// nothing, while `data_replay`'s header claimed both produce byte-identical
// `EngineStats`. The rules it enforces (unusable balance -> synthetic default
// and say so; unknown pip size -> charge NOTHING rather than a wrong cost;
// commission halved from the round trip because the adapter bills entry AND
// exit) are unchanged and documented on the moved function.

fn default_symbol(settings: Option<&neoethos_core::Settings>) -> String {
    // **F-648 / F-CORE2 closure (2026-05-25)**: previously fell back to
    // `"EURUSD"` when settings was None — a synthetic default that the
    // no-synthetic-data directive forbids. Now returns the configured
    // symbol when settings loaded, empty string otherwise. Downstream
    // code rejects empty symbols (see `default_pip_size` returning NaN
    // for empty input → fitness guard rejects) so the operator gets a
    // clear "symbol required" error instead of silent EURUSD execution.
    // SHARED resolution (2026-06-04 parity unification): the Some branch now
    // delegates to `SystemConfig::resolve_symbol` in neoethos-core — the SAME
    // function the app server calls — so UI and CLI can never diverge. Only the
    // None-path F-CORE2 error logging stays CLI-specific.
    match settings {
        Some(settings) => settings.system.resolve_symbol(),
        None => {
            tracing::error!(
                target: "neoethos_cli::defaults",
                "No --symbol supplied and config.yaml could not be loaded; \
                 cannot synthesise a default per F-CORE2 doctrine — supply \
                 --symbol explicitly or ensure config.yaml is reachable."
            );
            String::new()
        }
    }
}

fn default_base_tf(settings: Option<&neoethos_core::Settings>) -> String {
    // **F-648 / F-CORE2 closure (2026-05-25)**: previously fell back to
    // `"M1"` when settings was None. Same fix as `default_symbol`.
    // 2026-06-04 parity: Some branch delegates to the shared core resolver.
    match settings {
        Some(settings) => settings.system.resolve_base_timeframe(),
        None => {
            tracing::error!(
                target: "neoethos_cli::defaults",
                "No --timeframe supplied and config.yaml could not be loaded; \
                 cannot synthesise a default per F-CORE2 doctrine — supply \
                 --timeframe explicitly or ensure config.yaml is reachable."
            );
            String::new()
        }
    }
}

/// Resolve the higher-TF CSV for the **effective** `base` (which may be a
/// `--base` override, not the config base). Delegates the actual selection to
/// the shared `SystemConfig::resolve_higher_timeframes` so the CLI and the app
/// server always pick the same ladder.
fn default_higher_tfs_csv(settings: Option<&neoethos_core::Settings>, base: &str) -> String {
    settings
        .map(|settings| settings.system.resolve_higher_timeframes(base).join(","))
        .unwrap_or_default()
}

fn default_batch_timeframes_csv(settings: Option<&neoethos_core::Settings>) -> String {
    // **F-648 / F-CORE2 closure (2026-05-25)**: previously fell back to
    // `"M1,M5,M15,H1,H4"` when settings was None — a synthetic default
    // that the no-synthetic-data directive forbids. Now returns empty
    // when settings can't load; downstream sweep code surfaces a clear
    // "no timeframes specified" error.
    if let Some(settings) = settings {
        let mut timeframes = vec![settings.system.base_timeframe.clone()];
        let higher_timeframes = if settings.system.multi_resolution_enabled
            && !settings.system.multi_resolution_timeframes.is_empty()
        {
            &settings.system.multi_resolution_timeframes
        } else {
            &settings.system.higher_timeframes
        };
        for timeframe in higher_timeframes {
            if !timeframes.contains(timeframe) {
                timeframes.push(timeframe.clone());
            }
        }
        return timeframes.join(",");
    }

    tracing::error!(
        target: "neoethos_cli::defaults",
        "No --timeframes supplied and config.yaml could not be loaded; \
         cannot synthesise a default per F-CORE2 doctrine — supply \
         --timeframes explicitly or ensure config.yaml is reachable."
    );
    String::new()
}

/// Environment variables this binary used to obey and no longer does, each
/// paired with what replaced it.
///
/// NOTHING branches on this list — it is a report, not a decision. It exists
/// because the failure mode this whole consolidation is closing is the SILENT
/// one: an operator who exported a name that used to work, and a binary that
/// ignored it without saying so. A deleted lever that is still being pulled
/// must announce that it is deleted.
const RETIRED_ENV_VARS: &[(&str, &str)] = &[(
    "NEOETHOS_BOT_CPU_BUDGET",
    "--cpu-threads (set by the schedule orchestrator on the children it spawns)",
)];

/// Name every retired variable that is still set in this process's
/// environment, state its replacement, and say plainly that the value was
/// ignored. Called once from `main()`.
fn warn_retired_env_vars() {
    for (name, replacement) in RETIRED_ENV_VARS {
        let Some(value) = std::env::var_os(name) else {
            continue;
        };
        let value = value.to_string_lossy().to_string();
        eprintln!(
            "IGNORED ENV VAR: {name}={value} — this variable was deleted. Nothing read it, \
             and NOTHING in this run was changed by it. Its replacement is: {replacement}."
        );
        tracing::warn!(
            target: "neoethos_cli",
            env_var = %name,
            value = %value,
            replacement = %replacement,
            "a deleted environment variable is still set; its value was ignored"
        );
    }
}

/// Parse an optional blend-multiplier flag (`--gate-floor` / `--veto-below`).
///
/// An unparseable value is a HARD ERROR, never a silent `None`. The previous
/// `.and_then(|v| v.parse().ok())` turned `--gate-floor 0,34` (comma) or a typo
/// into "operator said nothing" and ran the default — on a knob that scales
/// every position in the replay. Absent still means absent; RANGE and inversion
/// validation belong to `BlendConfig::from_config_values`, which logs both the
/// configured and the used number when it refuses.
fn parse_blend_knob(args: &[String], name: &str) -> Result<Option<f64>> {
    match parse_flag(args, name) {
        None => Ok(None),
        Some(raw) => match raw.trim().parse::<f64>() {
            Ok(v) => Ok(Some(v)),
            Err(err) => anyhow::bail!(
                "{name} expects a number in [0,1] (got '{raw}'): {err}. This value \
                 scales every position's size — refusing to guess."
            ),
        },
    }
}

fn parse_flag(args: &[String], name: &str) -> Option<String> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            return iter.next().cloned();
        }
    }
    None
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

/// `credentials` subcommand — write `broker_credentials.toml` headlessly.
///
/// This is the CLI parity for the Flutter Settings → cTrader credentials
/// form. Same file, same schema, same path-resolution rules — the
/// shared writer lives in `neoethos_core::broker_config` so the two
/// frontends can never drift.
///
/// Subcommands:
///   credentials show
///     Read the current `broker_credentials.toml` and print a redacted
///     summary (client_secret is shown as `••••<last4>` only). Useful
///     for verifying which set the binary is picking up via the
///     `NEOETHOS_BROKER_CREDENTIALS_PATH` env override.
///
///   credentials set --client-id <id> [--client-secret <secret>]
///                   [--redirect-uri <uri>] [--environment Demo|Live]
///                   [--account-id <cTID>]
///     Merge-update the on-disk file. Unspecified fields keep their
///     current value (merge semantics match `POST /broker/credentials`).
///     If --client-secret is provided but blank, the existing secret
///     is preserved (same rule as the UI's "Leave blank to keep" form).
fn cmd_credentials(args: &[String]) -> Result<()> {
    if args.is_empty() {
        anyhow::bail!(
            "credentials requires a subcommand: `show` or `set`. \
             Run `neoethos-cli credentials show` to read the current \
             on-disk values."
        );
    }
    match args[0].as_str() {
        "show" => cmd_credentials_show(),
        "set" => cmd_credentials_set(&args[1..]),
        other => {
            anyhow::bail!("unknown credentials subcommand `{other}`. Expected `show` or `set`.")
        }
    }
}

fn cmd_credentials_show() -> Result<()> {
    let path = neoethos_core::broker_config::credentials_file_path()?;
    let loaded = neoethos_core::broker_config::load_from_disk(&path)?;
    println!("Path: {}", path.display());
    match loaded {
        None => {
            println!("(no file at that path — defaults will be used)");
        }
        Some(state) => {
            println!("Schema version: {}", state.schema_version);
            println!("\n[ctrader]");
            println!("  client_id    : {}", maybe_blank(&state.ctrader.client_id));
            println!(
                "  client_secret: {}",
                redact_secret(&state.ctrader.client_secret)
            );
            println!(
                "  redirect_uri : {}",
                maybe_blank(&state.ctrader.redirect_uri)
            );
            println!("  environment  : {}", state.ctrader.environment.as_str());
            println!("  accounts     : {} entries", state.ctrader.accounts.len());
            for (i, a) in state.ctrader.accounts.iter().enumerate() {
                println!(
                    "    [{i}] id={} label={} enabled={}",
                    a.account_id, a.label, a.enabled_for_execution
                );
            }
        }
    }
    Ok(())
}

fn cmd_credentials_set(args: &[String]) -> Result<()> {
    let mut client_id: Option<String> = None;
    let mut client_secret: Option<String> = None;
    let mut redirect_uri: Option<String> = None;
    let mut environment: Option<String> = None;
    let mut account_id: Option<String> = None;

    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--client-id" => client_id = iter.next().cloned(),
            "--client-secret" => client_secret = iter.next().cloned(),
            "--redirect-uri" => redirect_uri = iter.next().cloned(),
            "--environment" => environment = iter.next().cloned(),
            "--account-id" => account_id = iter.next().cloned(),
            other => anyhow::bail!(
                "unknown flag `{other}` for `credentials set`. \
                 Supported flags: --client-id, --client-secret, \
                 --redirect-uri, --environment, --account-id."
            ),
        }
    }

    let path = neoethos_core::broker_config::credentials_file_path()?;
    let mut state = neoethos_core::broker_config::load_from_disk(&path)?.unwrap_or_default();

    if let Some(v) = client_id {
        state.ctrader.client_id = v.trim().to_string();
    }
    // Empty-secret semantics match the UI: blank means "keep current".
    if let Some(v) = client_secret {
        if !v.is_empty() {
            state.ctrader.client_secret = v;
        }
    }
    if let Some(v) = redirect_uri {
        state.ctrader.redirect_uri = v.trim().to_string();
    }
    if let Some(v) = environment {
        let parsed =
            neoethos_core::broker_config::CTraderBrokerEnvironment::parse(&v).ok_or_else(|| {
                anyhow::anyhow!("invalid --environment value `{v}`. Expected `Demo` or `Live`.")
            })?;
        state.ctrader.environment = parsed;
    }
    if let Some(v) = account_id {
        let trimmed = v.trim().to_string();
        if !trimmed.is_empty() {
            // Replace the entire account list with the single target —
            // matches the UI's behaviour (the dropdown sends one
            // accountId, the server overwrites the targets vec).
            state.ctrader.accounts = vec![neoethos_core::broker_config::BrokerAccountTarget {
                account_id: trimmed,
                label: String::new(),
                enabled_for_execution: true,
            }];
        }
    }

    if state.ctrader.client_id.trim().is_empty() {
        anyhow::bail!(
            "client_id is required (it is currently blank on disk and no \
             --client-id was supplied). Provide --client-id at least once \
             before saving."
        );
    }
    if state.ctrader.client_secret.is_empty() {
        anyhow::bail!(
            "client_secret is required (it is currently blank on disk and \
             no --client-secret was supplied). Provide --client-secret at \
             least once before saving."
        );
    }
    if state.ctrader.redirect_uri.trim().is_empty() {
        // Sourced from neoethos-core so the listener, the CLI
        // default, and the embedded fallback can't drift apart
        // (#150).
        state.ctrader.redirect_uri =
            neoethos_core::broker_config::CTRADER_OAUTH_REDIRECT_URI.to_string();
    }

    neoethos_core::broker_config::save_to_disk(&path, &state)?;
    println!(
        "Wrote {} ({} ctrader.accounts)",
        path.display(),
        state.ctrader.accounts.len()
    );
    println!(
        "Next step: open the GUI and run Broker Setup → Re-authenticate to fetch an OAuth token."
    );
    Ok(())
}

fn redact_secret(s: &str) -> String {
    if s.is_empty() {
        return "(blank)".to_string();
    }
    let last4: String = s
        .chars()
        .rev()
        .take(4)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("••••{last4} (len={})", s.len())
}

fn maybe_blank(s: &str) -> &str {
    if s.is_empty() { "(blank)" } else { s }
}

fn print_help() {
    println!("neoethos-cli");
    println!("  [--cpu-threads N] [--startup-diagnostics]");
    println!(
        "  native-research start --contract-relative-path <path> --expected-sha256 <64-lowerhex> [--population N] [--population-auto true|false] [--max-indicators N] [--api-base http://127.0.0.1:PORT]"
    );
    println!("  native-research status [--api-base http://127.0.0.1:PORT]");
    println!("  native-research cancel [--api-base http://127.0.0.1:PORT]");
    println!("  symbols --root data");
    println!("  timeframes --symbol EURUSD --root data");
    println!("  load --symbol EURUSD --timeframe M1 [--dataset-identity d1-...] --root data");
    println!("  features --symbol EURUSD --timeframe M1 [--dataset-identity d1-...] --root data");
    println!(
        "  prepare --symbol EURUSD --base M1 --higher H1,H4 [--dataset-identity d1-...] --root data"
    );
    println!(
        "  bench --dry-run --fixture tiny --prototype a --backend cuda --out cache/gpu-bench/plan.json"
    );
    println!("  train --symbol EURUSD --base M1 --higher H1,H4 --horizon 1 --root data");
    println!(
        "  canonical-cost-build --authority-root <dir> --data-root <dir> --plan-sha256 <sha> --matrix-sha256 <sha> --symbol EURUSD --basis-timeframe D1 --broker-symbol-contract <json> --settings-source <yaml> --out <json>"
    );
    println!(
        "  canonical-full-run --authority-root <dir> --data-root <dir> --plan-sha256 <sha> \\\n         --matrix-sha256 <sha> --symbol EURUSD --base-timeframe M1 \\\n         --cost-assumptions <json> --broker-symbol-contract <json> \\\n         --settings-source <yaml> --models-dir <dir> --out <json> \\\n         --receipt-out <json>"
    );
    println!(
        "  canonical-train --authority-root <dir> --data-root <dir> --plan-sha256 <sha> --matrix-sha256 <sha> --symbol EURUSD --base-timeframe H4 --input-receipt <json> --cost-assumptions <json> --broker-symbol-contract <json> --settings-source <yaml> --models-dir <dir> --oos-from-ms <unix-ms> --out <json> --receipt-out <json>"
    );
    println!(
        "  search --expected-input-receipt <receipt.json> --seed 42 --candidates 100 --max-indicators 12 --stop-multiple 1.0 --target-multiple 2.0 --out <artifact.json> --root data"
    );
    println!(
        "                               CpuOnly gross-R research from exact direct generations; artifacts are ResearchOnly and NotPromotionEligible."
    );
    println!(
        "  discover --symbol EURUSD --base M1 --higher H1,H4 [--dataset-identity d1-...] --population 100 --generations 5 --max-indicators 12 --portfolio-size 100 --candidates 200 --corr 0.7 --min-trades 1 --out cache/vector_ta_knowledge.json --root data"
    );
    println!(
        "                               Historical search requires every exact generation named by its receipt. Discovery/schedule require the direct base plus explicitly requested higher TFs. No timeframe is manufactured; missing data requires import/download."
    );
    println!(
        "  discover ... --stream-sweep [--stream-max-batches N]  Sweep the (indicator, period) space in BATCHES instead of building one cube: a batch is sized from FREE RAM (never a flag), a discovery cycle runs per batch, and a batch whose candidates cannot clear the CONFIGURED expectancy floor is abandoned before the quality screen. Survivors from different batches are remapped onto one run-level feature list; every abandoned batch is named by cursor in <out>.streaming.json."
    );
    println!(
        "  discovery-promote-weekly --portfolio <live_portfolio.json> [--cache-dir cache/search]  Weekly-refresh using the strict v3 portfolio's embedded exact receipt/config authority; print 'added N new, carried M, total K'."
    );
    println!(
        "  trader-replay [--symbol EURUSD --base M1 | --portfolio <live_portfolio.json>] [--root data] [--blend off|confirm|scale] [--models-root models]  Offline dry-run of the autonomous trader (zero broker calls; same engine as /autonomous/replay). With --portfolio runs the REAL genes; --blend gates their size by the ML ensemble (gene-dominant)."
    );
    println!(
        "  blend-test --portfolio <live_portfolio.json> --models-root models_oos_locked [--root data] [--gate-floor 0.34] [--veto-below 0.15]  Re-validate the gene<->ML blend on the NETTED portfolio: GenesOnly vs MlConfirm vs MlScale on the same engine + non-degradation verdict. Point --models-root at a LEAK-FREE root (train --oos-from)."
    );
    println!(
        "  train --symbol EURUSD --base H1 [--models-dir models] [--oos-from 2023-01-01]  Train the ML ensemble. --oos-from trains LEAK-LOCKED experts (rows < cutoff, purged) to a SEPARATE root for OOS blend validation."
    );
    println!(
        "  import --source <FILE> --format <csv|tsv|json-array|json-lines|parquet|arrow-ipc-file|arrow-ipc-stream|vortex> --source-namespace <ID> --symbol <EXACT> --timeframe <TF> --bar-timestamps bar_open [--expected-generation <GEN>] [--root data]"
    );
    println!(
        "                               Import exactly one declared source, atomically publish it as canonical Vortex, then reopen and verify the acknowledged generation."
    );
    println!(
        "  import --source <FILE-OR-DIR> --dry-run   Metadata-only candidate discovery; never converts or publishes."
    );
    println!(
        "  slice-dataset --symbol EURUSD --base M1 [--dataset-identity d1-...] --root <SRC> --out-root <DST> --from-date 2018-01-01 --to-date 2021-01-01"
    );
    println!(
        "                               Resolve one exact source identity, fully verify it, and atomically publish"
    );
    println!(
        "                               the [from,to) subset to a NEW root with typed source-generation provenance."
    );
    println!(
        "                               Missing/ambiguous identities and an existing output generation fail closed."
    );
    println!(
        "  bench-prepare --data-root data --symbol EURUSD --timeframe M1 --dataset-identity d1-... --out snapshots/M1.json"
    );
    println!(
        "  stop-target --symbol EURUSD --timeframe M1 [--dataset-identity d1-...] --pip 0.0001 --signal 1 --root data"
    );
    println!(
        "  autoresearch [--symbol EURUSD] [--max-sweeps 200] [--max-hours H] [--session <id>] [--scenario <label>] [--dry-run]"
    );
    println!(
        "                               The goal-driven loop: it proposes (search config, objective) pairs, runs them as"
    );
    println!(
        "                               sweeps of 100 searches, judges each against a FROZEN judge, journals every one, and"
    );
    println!(
        "                               stops with one of three verdicts. It optimises TOWARD the goals in your config and"
    );
    println!(
        "                               can never rewrite them — there is no flag for a goal, a cost or a judge threshold."
    );
    println!(
        "                               It never places an order and never writes live_portfolio.json. The operator promotes."
    );
    println!("  wizard                       Launch the interactive first-run wizard (TUI).");
    println!("  setup [show|paths|ctrader]  Headless credentials helper (Task #61).");
    println!("                               Prints canonical paths + ready-to-paste templates.");
    println!("  credentials show             Show on-disk broker_credentials.toml (redacted).");
    println!("  credentials set --client-id X --client-secret Y [--redirect-uri Z]");
    println!("                  [--environment Demo|Live] [--account-id N]");
    println!("                               Merge-update broker_credentials.toml. Same writer");
    println!("                               as the GUI Settings screen — never drifts.");
    println!();
    println!("  --data-path <root>     Select a canonical manifest-backed data root.");
    println!("                         Inventory support: train, discover.");
    println!(
        "  --dry-run              Print exact canonical identities/rejections and exit before execution."
    );
}

fn cli_record(operation: &str, status: &str, message: impl Into<String>) -> SectionedRunRecord {
    section_record(SubsystemSection::Cli, operation, status, message)
}

fn section_record(
    section: SubsystemSection,
    operation: &str,
    status: &str,
    message: impl Into<String>,
) -> SectionedRunRecord {
    let now = system_time_string();
    SectionedRunRecord {
        run_id: format!(
            "{}-{}-{}",
            section.as_str().to_lowercase(),
            operation,
            now.replace(':', "-")
        ),
        parent_run_id: None,
        started_at: now.clone(),
        finished_at: now,
        subsystem: section,
        operation: operation.to_string(),
        status: status.to_string(),
        symbol: None,
        timeframe: None,
        error_code: None,
        message: message.into(),
        body: String::new(),
    }
}

fn system_time_string() -> String {
    // F-282 + F-656 fix (2026-05-25): match the neoethos-app pattern —
    // never panic on pre-1970 clock; emit a sentinel + structured warn
    // so the operator sees the clock skew without losing the whole CLI.
    match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(d) => format!("unix:{}", d.as_secs()),
        Err(err) => {
            tracing::warn!(
                target: "neoethos_cli::main",
                error = %err,
                "system clock is before UNIX epoch; falling back to sentinel"
            );
            "unix:pre-1970".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SliceDatasetProvenanceV1, apply_batch_discover_cli_overrides, cli_record,
        gpu_assignment_env, publish_canonical_dataset_slice, schedule_series_rows, section_record,
        select_runtime_timeframe_identities, select_runtime_timeframe_identities_for_base,
    };
    use neoethos_core::sectioned_log::SubsystemSection;

    fn external_identity(
        namespace: &str,
        timeframe: neoethos_data::CanonicalTimeframe,
    ) -> neoethos_data::CanonicalDatasetIdentity {
        neoethos_data::CanonicalDatasetIdentity::external(
            namespace,
            "EURUSD",
            timeframe,
            neoethos_data::BarTimestampConvention::BarOpen,
        )
        .expect("test identity")
    }

    fn unique_test_root(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("test clock")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "neoethos_cli_{label}_{}_{}",
            std::process::id(),
            nonce
        ))
    }

    #[test]
    fn batch_population_auto_override_is_typed_and_inherits_when_absent() {
        let mut inherited = neoethos_search::DiscoveryConfig::default();
        inherited.population_auto = true;
        apply_batch_discover_cli_overrides(&[], &mut inherited)
            .expect("missing override inherits Settings-derived value");
        assert!(inherited.population_auto);

        let mut enabled = neoethos_search::DiscoveryConfig::default();
        apply_batch_discover_cli_overrides(
            &["--population-auto".to_owned(), "true".to_owned()],
            &mut enabled,
        )
        .expect("explicit true override");
        assert!(enabled.population_auto);

        let mut disabled = neoethos_search::DiscoveryConfig::default();
        disabled.population_auto = true;
        apply_batch_discover_cli_overrides(
            &["--population-auto".to_owned(), "false".to_owned()],
            &mut disabled,
        )
        .expect("explicit false override");
        assert!(!disabled.population_auto);

        let mut malformed = neoethos_search::DiscoveryConfig::default();
        let error = apply_batch_discover_cli_overrides(
            &["--population-auto".to_owned(), "maybe".to_owned()],
            &mut malformed,
        )
        .expect_err("malformed boolean must fail loudly");
        assert!(error.to_string().contains("expected true or false"));
    }

    fn publish_fixture(root: &std::path::Path, identity: &neoethos_data::CanonicalDatasetIdentity) {
        let ohlcv = neoethos_data::Ohlcv {
            timestamp: Some(vec![
                1_577_836_800_000,
                1_577_923_200_000,
                1_578_009_600_000,
            ]),
            open: vec![1.0, 2.0, 3.0],
            high: vec![1.1, 2.1, 3.1],
            low: vec![0.9, 1.9, 2.9],
            close: vec![1.05, 2.05, 3.05],
            volume: Some(vec![10.0, 20.0, 30.0]),
        };
        let provenance = neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1::new(
            "neoethos.cli-test-fixture.v1",
            b"fixture".to_vec(),
        )
        .expect("fixture provenance");
        neoethos_data::publish_canonical_ohlcv_generation(
            neoethos_data::CanonicalOhlcvPublishRequest {
                configured_root: root,
                identity,
                expected_generation: None,
                provenance: &provenance,
                ohlcv: &ohlcv,
                volume: neoethos_data::CanonicalVolumeRef::Float64(
                    ohlcv.volume.as_deref().expect("fixture volume"),
                ),
                rows_per_chunk: 2,
            },
        )
        .expect("publish fixture");
    }

    #[test]
    fn runtime_timeframes_stay_on_the_unique_base_source_scope() {
        let source_a = [
            neoethos_data::CanonicalTimeframe::M1,
            neoethos_data::CanonicalTimeframe::H1,
            neoethos_data::CanonicalTimeframe::H4,
        ]
        .into_iter()
        .map(|timeframe| external_identity("source-a", timeframe))
        .collect::<Vec<_>>();
        let source_a_m1 = source_a
            .iter()
            .find(|identity| identity.timeframe() == neoethos_data::CanonicalTimeframe::M1)
            .expect("M1 direct identity")
            .clone();
        let source_b_h1 = external_identity("source-b", neoethos_data::CanonicalTimeframe::H1);
        let mut available = source_a.clone();
        available.push(source_b_h1);
        let selected = select_runtime_timeframe_identities(
            &available,
            "EURUSD",
            "M1",
            &["H1".to_owned(), "H4".to_owned()],
        )
        .expect("select exact runtime identities");

        assert_eq!(selected.base_identity, source_a_m1);
        assert_eq!(
            selected.required.as_slice(),
            &[
                neoethos_data::CanonicalTimeframe::M1,
                neoethos_data::CanonicalTimeframe::H1,
                neoethos_data::CanonicalTimeframe::H4,
            ]
        );
        assert_eq!(selected.identities.len(), source_a.len());
        assert!(
            selected
                .identities
                .iter()
                .all(|identity| identity.scope() == selected.base_identity.scope())
        );
    }

    #[test]
    fn runtime_timeframes_require_every_direct_generation() {
        let error = select_runtime_timeframe_identities(
            &[external_identity(
                "source-a",
                neoethos_data::CanonicalTimeframe::M1,
            )],
            "EURUSD",
            "M1",
            &["H1".to_owned()],
        )
        .expect_err("missing direct generations must fail closed");

        assert!(
            error.to_string().contains("import/download required"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn runtime_timeframes_do_not_require_unconsumed_defaults() {
        let h4 = external_identity("source-a", neoethos_data::CanonicalTimeframe::H4);
        let selected = select_runtime_timeframe_identities_for_base(&[h4.clone()], &h4, &[])
            .expect("base-only search consumes one direct timeframe");

        assert_eq!(selected.base_identity, h4);
        assert_eq!(
            selected.required,
            vec![neoethos_data::CanonicalTimeframe::H4]
        );
        assert_eq!(selected.identities, vec![selected.base_identity.clone()]);
    }

    #[test]
    fn runtime_timeframes_reject_an_ambiguous_base_identity() {
        let error = select_runtime_timeframe_identities(
            &[
                external_identity("source-a", neoethos_data::CanonicalTimeframe::M1),
                external_identity("source-b", neoethos_data::CanonicalTimeframe::M1),
            ],
            "EURUSD",
            "M1",
            &[],
        )
        .expect_err("ambiguous base must fail closed");

        assert!(
            error.to_string().contains("exactly one"),
            "unexpected error: {error:#}"
        );
    }

    #[test]
    fn schedule_reads_the_verified_manifest_row_count() {
        let root = unique_test_root("schedule_rows");
        let identity = external_identity("source-a", neoethos_data::CanonicalTimeframe::D1);
        publish_fixture(&root, &identity);
        let inventory = neoethos_data::DatasetDiscovery::scan(&root).expect("verified inventory");

        let rows = schedule_series_rows(&inventory, &root, &identity).expect("schedule row count");

        assert_eq!(rows, 3);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn slice_publication_reopens_with_typed_source_provenance() {
        let source_root = unique_test_root("slice_source");
        let output_root = unique_test_root("slice_output");
        let identity = external_identity("source-a", neoethos_data::CanonicalTimeframe::D1);
        publish_fixture(&source_root, &identity);

        let outcome = publish_canonical_dataset_slice(
            &source_root,
            &output_root,
            &identity,
            1_577_923_200_000,
            1_578_096_000_000,
        )
        .expect("publish canonical slice");
        let decoded =
            SliceDatasetProvenanceV1::from_envelope(outcome.publication.manifest().provenance())
                .expect("typed slice provenance");

        assert_eq!(outcome.source_rows, 3);
        assert_eq!(outcome.kept_rows, 2);
        assert_eq!(decoded.source_identity(), &identity);
        assert_eq!(decoded.selected_row_range(), 1..3);
        assert_eq!(
            decoded.requested_range_ms(),
            (1_577_923_200_000, 1_578_096_000_000)
        );
        assert_eq!(
            decoded.selected_timestamp_range_ms(),
            (1_577_923_200_000, 1_578_009_600_000)
        );

        std::fs::remove_dir_all(source_root).ok();
        std::fs::remove_dir_all(output_root).ok();
    }

    #[test]
    fn integrated_wgpu_assignment_uses_typed_selector_without_cuda_pin() {
        use neoethos_core::scheduler::Assignment;
        use neoethos_core::system::{
            AcceleratorBackend, AcceleratorDevice, AcceleratorDeviceClass, HardwareProfile,
            TrainingPrecision,
        };

        let assignment = Assignment {
            id: "EURUSD/M1".to_string(),
            card_ids: vec![0],
            genes_per_card: 128,
            cpu_threads: 3,
            class: neoethos_core::scheduler::ComboClass::Light,
        };
        let profile = HardwareProfile {
            schema_version: neoethos_core::system::HARDWARE_PROFILE_SCHEMA_VERSION,
            cpu_cores: 12,
            total_ram_gb: 32.0,
            available_ram_gb: 24.0,
            gpu_names: vec!["AMD Radeon Graphics".to_string()],
            num_gpus: 1,
            gpu_mem_gb: vec![0.0],
            accelerator_devices: vec![AcceleratorDevice {
                id: 0,
                name: "AMD Radeon Graphics".to_string(),
                backend: AcceleratorBackend::Vulkan,
                device_class: AcceleratorDeviceClass::IntegratedGpu,
                backend_index: 0,
                memory_gb: 0.0,
                supported_precisions: vec![TrainingPrecision::Fp32],
                compute_capability: None,
                source: "test".to_string(),
            }],
            timestamp: "test".to_string(),
            platform_label: "test".to_string(),
        };

        let envs = gpu_assignment_env(&assignment, &profile);

        assert_eq!(
            envs.get("NEOETHOS_BOT_SEARCH_EVAL_WGPU_DEVICE"),
            Some(&"integrated:0".to_string())
        );
        assert!(!envs.contains_key("NEOETHOS_BOT_SEARCH_EVAL_CUDA_DEVICE"));
    }

    #[test]
    fn cli_record_targets_cli_section() {
        let record = cli_record("load", "SUCCESS", "load completed");
        assert_eq!(record.subsystem, SubsystemSection::Cli);
        assert_eq!(record.operation, "load");
        assert_eq!(record.status, "SUCCESS");
    }

    #[test]
    fn section_record_targets_requested_subsystem() {
        let record = section_record(
            SubsystemSection::Discovery,
            "discover",
            "FAILED",
            "discovery failed",
        );
        assert_eq!(record.subsystem, SubsystemSection::Discovery);
        assert_eq!(record.operation, "discover");
        assert_eq!(record.status, "FAILED");
        assert_eq!(record.message, "discovery failed");
    }
}
