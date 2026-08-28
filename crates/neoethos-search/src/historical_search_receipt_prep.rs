//! Strict selected-generation preparation for model-free historical search.
//!
//! Both receipt preparation and historical replay enter feature computation
//! through [`build_exact_selected_feature_input`]. The builder accepts only
//! validated, manifest-bound selections and opens them only through the exact
//! loader. It never inventories a root, follows a current-generation fallback,
//! derives an identity, or synthesizes a missing timeframe.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail, ensure};
use neoethos_core::execution::BudgetedCpuExecutor;
use neoethos_core::execution_budget::{
    CpuPermitRequest, InstalledExecutionBudget, WorkerLimit, installed_process_budget,
};
use neoethos_data::{
    CanonicalDatasetSeriesReceiptV1, CanonicalOhlcvFrame, FeatureFrame,
    SelectedDatasetGenerationV1, SymbolDataset, load_exact_canonical_timeframe,
    prepare_multitimeframe_features,
};

use crate::data_selection::CanonicalSearchInputReceiptV2;

static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct HistoricalSearchReceiptPrepArgs {
    root: PathBuf,
    anchor_selected_generation: PathBuf,
    direct_selected_generations: Vec<PathBuf>,
    output: PathBuf,
    cpu_threads: Option<WorkerLimit>,
}

/// Exact frames and their one canonical feature recipe.
///
/// The loaded frames retain immutable generation leases for the base OHLCV and
/// every direct higher timeframe while callers bind or replay the receipt.
#[derive(Debug)]
pub struct ExactSelectedFeatureInput {
    direct_frames: Vec<CanonicalOhlcvFrame>,
    base_index: usize,
    features: FeatureFrame,
}

impl ExactSelectedFeatureInput {
    pub fn base_frame(&self) -> &CanonicalOhlcvFrame {
        &self.direct_frames[self.base_index]
    }

    pub const fn features(&self) -> &FeatureFrame {
        &self.features
    }
}

/// Load a manifest-bound anchor and its direct higher timeframes, then execute
/// the sole canonical historical feature recipe.
pub fn build_exact_selected_feature_input(
    root: &Path,
    selected: &CanonicalDatasetSeriesReceiptV1,
) -> Result<ExactSelectedFeatureInput> {
    selected
        .validate()
        .context("validate exact selected dataset series before any dataset open")?;
    let anchor = selected.anchor().identity();
    let mut direct_frames = Vec::with_capacity(selected.direct_timeframes().len());
    let mut base_index = None;
    let mut dataset = SymbolDataset {
        symbol: anchor.symbol_name().to_owned(),
        frames: Default::default(),
        source_artifacts: Default::default(),
    };
    let mut higher_names = Vec::with_capacity(selected.direct_timeframes().len().saturating_sub(1));

    for direct in selected.direct_timeframes() {
        let timeframe = direct.identity().timeframe();
        let frame = load_exact_canonical_timeframe(root, direct).with_context(|| {
            format!(
                "load exact selected direct generation {} {}",
                direct.identity().to_path_component(),
                direct.generation_id()
            )
        })?;
        let timeframe_name = timeframe.as_str().to_owned();
        ensure!(
            dataset
                .frames
                .insert(timeframe_name.clone(), frame.ohlcv().clone())
                .is_none(),
            "duplicate exact selected direct timeframe {timeframe}"
        );
        ensure!(
            dataset
                .source_artifacts
                .insert(timeframe_name.clone(), frame.artifact().clone())
                .is_none(),
            "duplicate exact selected direct artifact {timeframe}"
        );
        if direct == selected.anchor() {
            base_index = Some(direct_frames.len());
        } else {
            ensure!(
                timeframe > anchor.timeframe(),
                "selected direct timeframe {timeframe} is not higher than anchor timeframe {}",
                anchor.timeframe()
            );
            higher_names.push(timeframe_name);
        }
        direct_frames.push(frame);
    }

    let base_index = base_index.context("exact selected frame set lost its anchor")?;
    let higher_refs = higher_names.iter().map(String::as_str).collect::<Vec<_>>();
    let features =
        prepare_multitimeframe_features(&dataset, anchor.timeframe().as_str(), &higher_refs)
            .context("compute features from only the exact selected direct generations")?;

    Ok(ExactSelectedFeatureInput {
        direct_frames,
        base_index,
        features,
    })
}

/// Prepare one create-new canonical search-input receipt from strict selected
/// generation JSON files.
pub fn run(args: &[String]) -> Result<()> {
    let args = HistoricalSearchReceiptPrepArgs::parse(args)?;
    preflight_create_new_output(&args.output)?;

    // Strictly decode every persisted selection before opening a dataset or
    // computing features. Unknown fields and invalid bindings fail here.
    let anchor = read_selected_generation(
        &args.anchor_selected_generation,
        "--anchor-selected-generation",
    )?;
    let mut direct = Vec::with_capacity(args.direct_selected_generations.len() + 1);
    direct.push(anchor.clone());
    for path in &args.direct_selected_generations {
        direct.push(read_selected_generation(
            path,
            "--direct-selected-generation",
        )?);
    }
    let selected = CanonicalDatasetSeriesReceiptV1::new(anchor, direct)
        .context("validate exact anchor and unique direct timeframe selections")?;

    let installed = installed_process_budget().context(
        "historical search receipt preparation requires the immutable process CPU budget to be installed",
    )?;
    args.validate_cpu_assignment(installed)?;
    let width = installed.resolved().effective_worker_limit;
    let broker = installed.broker().clone();
    let executor = BudgetedCpuExecutor::new_for_broker(broker.clone(), width);
    let feature_lease = broker
        .acquire(CpuPermitRequest::local(width))
        .context("acquire the process CPU budget for exact selected feature preparation")?;
    let loaded = executor
        .execute(feature_lease.into_transfer(), || {
            build_exact_selected_feature_input(&args.root, &selected)
        })
        .map_err(|error| anyhow::anyhow!("budgeted exact feature preparation failed: {error}"))??;

    let receipt = CanonicalSearchInputReceiptV2::from_feature_frame(
        selected.anchor().identity(),
        loaded.features(),
    )
    .context("bind exact selected generations and feature recipe into search receipt")?;
    let receipt_sha256 = receipt
        .identity_sha256()
        .context("hash prepared canonical search input receipt")?;
    let bytes = receipt
        .to_json_bytes()
        .context("serialize prepared canonical search input receipt")?;
    atomic_write_create_new(&args.output, &bytes)?;

    println!("receipt_sha256={receipt_sha256}");
    println!("output={}", args.output.display());
    Ok(())
}

impl HistoricalSearchReceiptPrepArgs {
    fn parse(args: &[String]) -> Result<Self> {
        let mut root = None;
        let mut anchor = None;
        let mut direct = Vec::new();
        let mut output = None;
        let mut cpu_threads = None;
        let mut index = 0;
        while index < args.len() {
            let flag = args[index].as_str();
            if let Some(raw) = flag.strip_prefix("--cpu-threads=") {
                ensure!(
                    cpu_threads.is_none(),
                    "--cpu-threads may be supplied only once"
                );
                cpu_threads = Some(parse_positive_worker_limit(raw, "--cpu-threads")?);
                index += 1;
                continue;
            }
            let value = args.get(index + 1).with_context(|| {
                format!("historical search receipt-preparation flag {flag} requires one value")
            })?;
            match flag {
                "--root" => set_once_path(&mut root, flag, value)?,
                "--anchor-selected-generation" => set_once_path(&mut anchor, flag, value)?,
                "--direct-selected-generation" => {
                    ensure!(!value.trim().is_empty(), "{flag} path is empty");
                    direct.push(PathBuf::from(value));
                }
                "--out" => set_once_path(&mut output, flag, value)?,
                "--cpu-threads" => {
                    ensure!(
                        cpu_threads.is_none(),
                        "--cpu-threads may be supplied only once"
                    );
                    cpu_threads = Some(parse_positive_worker_limit(value, flag)?);
                }
                _ => bail!("unknown historical search receipt-preparation flag `{flag}`"),
            }
            index += 2;
        }

        Ok(Self {
            root: required(root, "--root")?,
            anchor_selected_generation: required(anchor, "--anchor-selected-generation")?,
            direct_selected_generations: direct,
            output: required(output, "--out")?,
            cpu_threads,
        })
    }

    fn validate_cpu_assignment(&self, installed: &InstalledExecutionBudget) -> Result<()> {
        match (self.cpu_threads, installed.resolved().parent_limit) {
            (Some(received), Some(installed_parent)) => ensure!(
                received == installed_parent.limit,
                "--cpu-threads={} does not match installed parent assignment {}",
                received.get(),
                installed_parent.limit.get()
            ),
            (Some(received), None) => bail!(
                "--cpu-threads={} was not installed at process startup",
                received.get()
            ),
            (None, _) => {}
        }
        Ok(())
    }
}

fn read_selected_generation(path: &Path, flag: &str) -> Result<SelectedDatasetGenerationV1> {
    let bytes = fs::read(path).with_context(|| format!("read {flag} {}", path.display()))?;
    SelectedDatasetGenerationV1::from_json_bytes(&bytes)
        .with_context(|| format!("strictly decode {flag} {}", path.display()))
}

fn preflight_create_new_output(path: &Path) -> Result<()> {
    let parent = output_parent(path);
    ensure!(
        parent.is_dir(),
        "output parent {} does not exist",
        parent.display()
    );
    ensure!(
        !path.exists(),
        "refusing to overwrite existing output {}",
        path.display()
    );
    Ok(())
}

fn atomic_write_create_new(path: &Path, bytes: &[u8]) -> Result<()> {
    preflight_create_new_output(path)?;
    let parent = output_parent(path);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .context("output path has no UTF-8 file name")?;
    let sequence = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        ".{file_name}.neoethos-receipt-prep-{}-{sequence}.tmp",
        std::process::id()
    ));
    let mut cleanup = TemporaryOutput::new(temporary.clone());
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .with_context(|| format!("create temporary output {}", temporary.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write temporary output {}", temporary.display()))?;
    file.sync_all()
        .with_context(|| format!("sync temporary output {}", temporary.display()))?;
    drop(file);
    fs::hard_link(&temporary, path).with_context(|| {
        format!(
            "atomically install create-new output {} from {}",
            path.display(),
            temporary.display()
        )
    })?;
    fs::remove_file(&temporary)
        .with_context(|| format!("remove linked temporary output {}", temporary.display()))?;
    cleanup.disarm();
    Ok(())
}

fn output_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

struct TemporaryOutput {
    path: PathBuf,
    armed: bool,
}

impl TemporaryOutput {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        if self.armed
            && let Err(error) = fs::remove_file(&self.path)
        {
            eprintln!(
                "ERROR failed to clean historical-search receipt temporary output {}: {error}",
                self.path.display()
            );
        }
    }
}

fn set_once<T>(slot: &mut Option<T>, flag: &str, value: T) -> Result<()> {
    ensure!(slot.is_none(), "{flag} may be supplied only once");
    *slot = Some(value);
    Ok(())
}

fn set_once_path(slot: &mut Option<PathBuf>, flag: &str, value: &str) -> Result<()> {
    ensure!(!value.trim().is_empty(), "{flag} path is empty");
    set_once(slot, flag, PathBuf::from(value))
}

fn required<T>(value: Option<T>, flag: &str) -> Result<T> {
    value.with_context(|| format!("historical search receipt preparation requires {flag}"))
}

fn parse_positive_worker_limit(value: &str, flag: &str) -> Result<WorkerLimit> {
    let parsed = value
        .parse::<usize>()
        .with_context(|| format!("{flag} expects a positive integer, got `{value}`"))?;
    ensure!(parsed > 0, "{flag} must be greater than zero");
    WorkerLimit::new(parsed).map_err(|error| anyhow::anyhow!("{flag}: {error}"))
}
