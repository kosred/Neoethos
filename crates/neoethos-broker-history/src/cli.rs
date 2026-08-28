use crate::{
    HistoricalCaptureRequest, HistoricalCaptureTarget, begin_process_historical_capture,
    capture_historical_generation, load_production_historical_credentials,
};
use anyhow::{Context, Result, bail};
use clap::Parser;
use neoethos_core::CanonicalTimeframe;
use neoethos_data::SelectedDatasetGenerationV1;
use neoethos_execution_budget::{
    CancellationToken, CpuPermitRequest, InstalledExecutionBudget, WorkerLimit,
};
use std::ffi::OsString;
use std::path::PathBuf;

pub const HISTORICAL_START_2016_UNIX_MS: i64 = 1_451_606_400_000;

#[derive(Debug, Parser)]
#[command(
    name = "neoethos-historical-fetch",
    about = "Capture one direct cTrader timeframe and print its exact immutable receipt"
)]
pub struct HistoricalFetchCli {
    #[arg(long)]
    symbol: String,
    #[arg(long)]
    timeframe: String,
    #[arg(long)]
    data_root: PathBuf,
    /// Exact strict receipt JSON required for an update. Omit only for CREATE.
    #[arg(long)]
    selected_generation_file: Option<PathBuf>,
    /// Exclusive Unix-millisecond upper bound. Omit to capture through now.
    #[arg(long)]
    to_ms: Option<i64>,
    /// Parsed by the immutable process-budget installer before capture.
    #[arg(long, hide = true)]
    cpu_threads: Option<usize>,
}

impl HistoricalFetchCli {
    pub fn try_parse_from<I, T>(args: I) -> std::result::Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        <Self as Parser>::try_parse_from(args)
    }

    pub fn into_requests(self) -> Result<Vec<HistoricalCaptureRequest>> {
        let symbol = self.symbol.trim().to_ascii_uppercase();
        if symbol.is_empty() {
            bail!("symbol must be non-empty");
        }
        let timeframe = self
            .timeframe
            .trim()
            .to_ascii_uppercase()
            .parse::<CanonicalTimeframe>()
            .map_err(|_| {
                anyhow::anyhow!("unsupported direct cTrader timeframe {:?}", self.timeframe)
            })?;
        let to_ms = self
            .to_ms
            .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
        if to_ms <= HISTORICAL_START_2016_UNIX_MS {
            bail!("to-ms must be later than 2016-01-01T00:00:00Z");
        }
        let target = match self.selected_generation_file {
            Some(path) => {
                let bytes = std::fs::read(&path)
                    .with_context(|| format!("read exact receipt {}", path.display()))?;
                HistoricalCaptureTarget::SelectedGeneration(
                    SelectedDatasetGenerationV1::from_json_bytes(&bytes)
                        .with_context(|| format!("decode exact receipt {}", path.display()))?,
                )
            }
            None => HistoricalCaptureTarget::NewIdentity,
        };
        let _ = self.cpu_threads;
        Ok(vec![HistoricalCaptureRequest {
            symbol,
            timeframe,
            from_ms: HISTORICAL_START_2016_UNIX_MS,
            to_ms,
            data_root: self.data_root,
            target,
        }])
    }
}

pub fn render_receipt_stdout(receipt: &SelectedDatasetGenerationV1) -> Result<Vec<u8>> {
    let mut bytes = receipt.to_json_bytes()?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn execute(
    cli: HistoricalFetchCli,
    budget: &'static InstalledExecutionBudget,
) -> Result<SelectedDatasetGenerationV1> {
    let mut requests = cli.into_requests()?;
    let request = requests.pop().context("one historical request")?;
    if !requests.is_empty() {
        bail!("one process invocation accepts exactly one symbol/timeframe transaction");
    }

    // Register the process-wide exact run before joining the CPU queue. A
    // concurrent invocation in this process therefore conflicts immediately.
    let active = begin_process_historical_capture()?;
    let capture_cancel = active.cancellation_handle();
    let cpu_cancel = CancellationToken::new();
    let signal_cpu_cancel = cpu_cancel.clone();
    ctrlc::set_handler(move || {
        signal_cpu_cancel.cancel();
        let _ = capture_cancel.cancel();
    })
    .context("install historical capture Ctrl-C handler")?;

    let width = WorkerLimit::new(1).context("one-worker historical CPU demand")?;
    let lease = budget
        .broker()
        .acquire_cancellable(CpuPermitRequest::local(width), &cpu_cancel)
        .context("historical capture CPU admission")?;
    let outcome = lease.scope(|| {
        let credentials = load_production_historical_credentials()?;
        capture_historical_generation(request, credentials, &active)
    })?;
    Ok(outcome.selected_generation)
}
