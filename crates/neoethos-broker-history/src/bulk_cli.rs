use crate::{
    BrokerEnvironment, CANONICAL_TRENDBAR_SERIES_FROM_MS_V1, CanonicalTrendbarAcquisitionPlanV1,
    CanonicalTrendbarAcquisitionRunFailureV1, CanonicalTrendbarAcquisitionRunOutcomeV1,
    CanonicalTrendbarAcquisitionRunStageV1, CanonicalTrendbarAcquisitionStoreV1,
    CanonicalTrendbarCheckpointReceiptV1, CanonicalTrendbarPlanReceiptV1,
    CanonicalTrendbarSymbolV1, begin_process_historical_capture,
    run_production_canonical_trendbar_acquisition_v1,
};
use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use neoethos_data::{CTraderEnvironment, CanonicalTimeframe};
use neoethos_execution_budget::{
    CancellationToken, CpuPermitRequest, InstalledExecutionBudget, WorkerLimit,
};
use serde::Serialize;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

const MAX_CAPTURE_ATTEMPTS_PER_CHECKPOINT_V1: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum CanonicalTrendbarEnvironmentArgV1 {
    Demo,
    Live,
}

impl CanonicalTrendbarEnvironmentArgV1 {
    const fn broker(self) -> BrokerEnvironment {
        match self {
            Self::Demo => BrokerEnvironment::Demo,
            Self::Live => BrokerEnvironment::Live,
        }
    }

    const fn canonical(self) -> CTraderEnvironment {
        match self {
            Self::Demo => CTraderEnvironment::Demo,
            Self::Live => CTraderEnvironment::Live,
        }
    }
}

fn parse_checkpoint_sha256(value: &str) -> std::result::Result<String, String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        Ok(value.to_owned())
    } else {
        Err("checkpoint SHA-256 must be exactly 64 lowercase hexadecimal characters".to_owned())
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "neoethos-canonical-trendbar-bulk",
    about = "Capture one exact account/symbol matrix in every direct canonical cTrader timeframe"
)]
pub struct CanonicalTrendbarBulkCli {
    #[arg(long, value_enum)]
    environment: CanonicalTrendbarEnvironmentArgV1,
    #[arg(long)]
    account_id: i64,
    #[arg(long, required = true)]
    symbol: Vec<String>,
    #[arg(long)]
    to_ms_exclusive: i64,
    #[arg(long)]
    data_root: PathBuf,
    #[arg(long)]
    authority_root: PathBuf,
    #[arg(long, value_parser = parse_checkpoint_sha256)]
    checkpoint_sha256: Option<String>,
    #[arg(long, hide = true)]
    cpu_threads: Option<usize>,
}

impl CanonicalTrendbarBulkCli {
    pub fn try_parse_from<I, T>(args: I) -> std::result::Result<Self, clap::Error>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        <Self as Parser>::try_parse_from(args)
    }

    pub fn prepare(self) -> Result<PreparedCanonicalTrendbarBulkV1> {
        if self.account_id <= 0 {
            bail!("account-id must be positive");
        }
        if self.to_ms_exclusive <= CANONICAL_TRENDBAR_SERIES_FROM_MS_V1 {
            bail!("to-ms-exclusive must be later than 2016-01-01T00:00:00Z");
        }
        require_absolute_distinct_roots(&self.data_root, &self.authority_root)?;
        let symbols = self
            .symbol
            .into_iter()
            .map(parse_symbol_binding)
            .collect::<Result<Vec<_>>>()?;
        let environment = self.environment.broker();
        let plan = CanonicalTrendbarAcquisitionPlanV1::new(
            self.environment.canonical(),
            environment.endpoint_host(),
            self.account_id,
            CANONICAL_TRENDBAR_SERIES_FROM_MS_V1,
            self.to_ms_exclusive,
            symbols,
            CanonicalTimeframe::ALL.to_vec(),
        )?;
        let store = CanonicalTrendbarAcquisitionStoreV1::new(&self.authority_root);
        let plan_receipt = store.publish_plan(&plan)?;
        let checkpoint_receipt = self
            .checkpoint_sha256
            .map(CanonicalTrendbarCheckpointReceiptV1::from_sha256)
            .transpose()?;
        if let Some(receipt) = checkpoint_receipt.as_ref() {
            store
                .open_checkpoint(&self.data_root, &plan_receipt, receipt)
                .context("open exact requested canonical trendbar checkpoint")?;
        }
        let _ = self.cpu_threads;
        Ok(PreparedCanonicalTrendbarBulkV1 {
            data_root: self.data_root,
            store,
            plan,
            plan_receipt,
            checkpoint_receipt,
        })
    }
}

fn parse_symbol_binding(binding: String) -> Result<CanonicalTrendbarSymbolV1> {
    let (symbol_id, symbol_name) = binding
        .split_once('=')
        .context("symbol must use the exact <positive-id>=<broker-name> form")?;
    if symbol_name.contains('=') {
        bail!("symbol binding contains more than one '=' delimiter");
    }
    let symbol_id = symbol_id
        .parse::<i64>()
        .context("symbol id is not one exact integer")?;
    CanonicalTrendbarSymbolV1::new(symbol_id, symbol_name)
}

fn require_absolute_distinct_roots(data_root: &Path, authority_root: &Path) -> Result<()> {
    if !data_root.is_absolute() || !authority_root.is_absolute() {
        bail!("data-root and authority-root must both be explicit absolute paths");
    }
    if data_root == authority_root {
        bail!("data-root and authority-root must be distinct");
    }
    Ok(())
}

pub struct PreparedCanonicalTrendbarBulkV1 {
    data_root: PathBuf,
    store: CanonicalTrendbarAcquisitionStoreV1,
    plan: CanonicalTrendbarAcquisitionPlanV1,
    plan_receipt: CanonicalTrendbarPlanReceiptV1,
    checkpoint_receipt: Option<CanonicalTrendbarCheckpointReceiptV1>,
}

fn run_canonical_trendbar_capture_with_bounded_resume_v1<T, R, C, W>(
    initial_checkpoint: Option<CanonicalTrendbarCheckpointReceiptV1>,
    mut run: R,
    mut is_cancelled: C,
    mut wait: W,
) -> std::result::Result<T, CanonicalTrendbarAcquisitionRunFailureV1>
where
    R: FnMut(
        Option<&CanonicalTrendbarCheckpointReceiptV1>,
    ) -> std::result::Result<T, CanonicalTrendbarAcquisitionRunFailureV1>,
    C: FnMut() -> bool,
    W: FnMut(Duration),
{
    let mut checkpoint = initial_checkpoint;
    let mut attempts_at_checkpoint = 0_usize;
    loop {
        let attempted_checkpoint = checkpoint
            .as_ref()
            .map(|receipt| receipt.sha256().to_owned());
        match run(checkpoint.as_ref()) {
            Ok(outcome) => return Ok(outcome),
            Err(failure) => {
                if failure.stage() != CanonicalTrendbarAcquisitionRunStageV1::Capture
                    || is_cancelled()
                {
                    return Err(failure);
                }
                let next_checkpoint = failure.last_checkpoint_receipt().cloned();
                if checkpoint.is_some() && next_checkpoint.is_none() {
                    return Err(failure);
                }
                let next_checkpoint_sha = next_checkpoint
                    .as_ref()
                    .map(|receipt| receipt.sha256().to_owned());
                attempts_at_checkpoint = if attempted_checkpoint == next_checkpoint_sha {
                    attempts_at_checkpoint.saturating_add(1)
                } else {
                    1
                };
                if attempts_at_checkpoint >= MAX_CAPTURE_ATTEMPTS_PER_CHECKPOINT_V1 {
                    return Err(failure);
                }

                checkpoint = next_checkpoint;
                let backoff_exponent = u32::try_from(attempts_at_checkpoint - 1)
                    .expect("bounded capture attempt exponent");
                let delay_seconds = 1_u64 << backoff_exponent.min(3);
                wait(Duration::from_secs(delay_seconds));
                if is_cancelled() {
                    return Err(failure);
                }
            }
        }
    }
}

impl PreparedCanonicalTrendbarBulkV1 {
    pub fn data_root(&self) -> &Path {
        &self.data_root
    }

    pub const fn store(&self) -> &CanonicalTrendbarAcquisitionStoreV1 {
        &self.store
    }

    pub const fn plan(&self) -> &CanonicalTrendbarAcquisitionPlanV1 {
        &self.plan
    }

    pub const fn plan_receipt(&self) -> &CanonicalTrendbarPlanReceiptV1 {
        &self.plan_receipt
    }

    pub const fn checkpoint_receipt(&self) -> Option<&CanonicalTrendbarCheckpointReceiptV1> {
        self.checkpoint_receipt.as_ref()
    }
}

pub fn execute_canonical_trendbar_bulk_v1(
    prepared: PreparedCanonicalTrendbarBulkV1,
    budget: &'static InstalledExecutionBudget,
) -> Result<CanonicalTrendbarAcquisitionRunOutcomeV1> {
    let active = begin_process_historical_capture()?;
    let capture_cancel = active.cancellation_handle();
    let cpu_cancel = CancellationToken::new();
    let signal_cpu_cancel = cpu_cancel.clone();
    ctrlc::set_handler(move || {
        signal_cpu_cancel.cancel();
        let _ = capture_cancel.cancel();
    })
    .context("install canonical trendbar bulk Ctrl-C handler")?;
    let width = WorkerLimit::new(1).context("one-worker canonical trendbar bulk CPU demand")?;
    let lease = budget
        .broker()
        .acquire_cancellable(CpuPermitRequest::local(width), &cpu_cancel)
        .context("canonical trendbar bulk CPU admission")?;
    lease.scope(|| {
        run_canonical_trendbar_capture_with_bounded_resume_v1(
            prepared.checkpoint_receipt.clone(),
            |checkpoint| {
                run_production_canonical_trendbar_acquisition_v1(
                    &prepared.data_root,
                    &prepared.store,
                    &prepared.plan_receipt,
                    checkpoint,
                    &active,
                )
            },
            || active.is_cancelled(),
            std::thread::sleep,
        )
        .map_err(anyhow::Error::new)
    })
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct CanonicalTrendbarBulkOutcomeWireV1<'a> {
    schema: &'static str,
    version: u16,
    plan_receipt: &'a CanonicalTrendbarPlanReceiptV1,
    checkpoint_receipt: &'a CanonicalTrendbarCheckpointReceiptV1,
    matrix_receipt: &'a crate::CanonicalTrendbarMatrixReceiptV1,
    completed_cells: usize,
    total_cells: usize,
}

pub fn render_canonical_trendbar_bulk_stdout_v1(
    outcome: &CanonicalTrendbarAcquisitionRunOutcomeV1,
) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(&CanonicalTrendbarBulkOutcomeWireV1 {
        schema: "neoethos.canonical_trendbar_bulk_outcome.v1",
        version: 1,
        plan_receipt: outcome.plan_receipt(),
        checkpoint_receipt: outcome.checkpoint_receipt(),
        matrix_receipt: outcome.matrix_receipt(),
        completed_cells: outcome.completed_cells(),
        total_cells: outcome.total_cells(),
    })?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CanonicalTrendbarAcquisitionRunFailureV1, CanonicalTrendbarAcquisitionRunStageV1};
    use anyhow::anyhow;
    use std::cell::{Cell, RefCell};
    use std::time::Duration;

    fn checkpoint(byte: char) -> CanonicalTrendbarCheckpointReceiptV1 {
        CanonicalTrendbarCheckpointReceiptV1::from_sha256(byte.to_string().repeat(64))
            .expect("checkpoint receipt")
    }

    fn failure(
        stage: CanonicalTrendbarAcquisitionRunStageV1,
        checkpoint: Option<CanonicalTrendbarCheckpointReceiptV1>,
    ) -> CanonicalTrendbarAcquisitionRunFailureV1 {
        CanonicalTrendbarAcquisitionRunFailureV1::new(
            stage,
            checkpoint,
            anyhow!("synthetic sanitized acquisition failure"),
        )
    }

    #[test]
    fn transient_capture_failure_reopens_from_the_exact_latest_checkpoint() {
        let latest = checkpoint('a');
        let calls = Cell::new(0_usize);
        let seen = RefCell::new(Vec::new());
        let waits = RefCell::new(Vec::new());

        let result = run_canonical_trendbar_capture_with_bounded_resume_v1(
            None,
            |resume| {
                seen.borrow_mut()
                    .push(resume.map(|receipt| receipt.sha256().to_owned()));
                let call = calls.get();
                calls.set(call + 1);
                if call == 0 {
                    Err(failure(
                        CanonicalTrendbarAcquisitionRunStageV1::Capture,
                        Some(latest.clone()),
                    ))
                } else {
                    Ok(168_usize)
                }
            },
            || false,
            |duration| waits.borrow_mut().push(duration),
        )
        .expect("transient capture restart");

        assert_eq!(result, 168);
        assert_eq!(calls.get(), 2);
        assert_eq!(
            seen.into_inner(),
            vec![None, Some(latest.sha256().to_owned())]
        );
        assert_eq!(waits.into_inner(), vec![Duration::from_secs(1)]);
    }

    #[test]
    fn capture_restart_is_bounded_at_one_unchanged_checkpoint() {
        let latest = checkpoint('b');
        let calls = Cell::new(0_usize);
        let waits = Cell::new(0_usize);

        let result = run_canonical_trendbar_capture_with_bounded_resume_v1(
            Some(latest.clone()),
            |_| {
                calls.set(calls.get() + 1);
                Err::<(), _>(failure(
                    CanonicalTrendbarAcquisitionRunStageV1::Capture,
                    Some(latest.clone()),
                ))
            },
            || false,
            |_| waits.set(waits.get() + 1),
        );

        assert!(result.is_err());
        assert_eq!(calls.get(), MAX_CAPTURE_ATTEMPTS_PER_CHECKPOINT_V1);
        assert_eq!(waits.get(), MAX_CAPTURE_ATTEMPTS_PER_CHECKPOINT_V1 - 1);
    }

    #[test]
    fn non_capture_failure_and_cancellation_never_restart() {
        for (stage, cancelled) in [
            (CanonicalTrendbarAcquisitionRunStageV1::Matrix, false),
            (CanonicalTrendbarAcquisitionRunStageV1::Capture, true),
        ] {
            let calls = Cell::new(0_usize);
            let waits = Cell::new(0_usize);
            let result = run_canonical_trendbar_capture_with_bounded_resume_v1(
                None,
                |_| {
                    calls.set(calls.get() + 1);
                    Err::<(), _>(failure(stage, None))
                },
                || cancelled,
                |_| waits.set(waits.get() + 1),
            );
            assert!(result.is_err());
            assert_eq!(calls.get(), 1);
            assert_eq!(waits.get(), 0);
        }
    }
}
