use crate::service::{HistoricalSeriesCapture, ProductionHistoricalSessionConnector};
use crate::{
    BrokerEnvironment, CanonicalTrendbarAcquisitionCellV1, CanonicalTrendbarAcquisitionStoreV1,
    CanonicalTrendbarCheckpointReceiptV1, CanonicalTrendbarMatrixReceiptV1,
    CanonicalTrendbarPlanReceiptV1, HistoricalCaptureRequest, HistoricalCaptureTarget,
    ProcessHistoricalCapture, load_exact_production_historical_credentials,
};
use anyhow::Error;
use std::fmt;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CanonicalTrendbarAcquisitionRunStageV1 {
    Plan,
    Credentials,
    Capture,
    Checkpoint,
    Matrix,
}

#[derive(Debug)]
pub struct CanonicalTrendbarAcquisitionRunFailureV1 {
    stage: CanonicalTrendbarAcquisitionRunStageV1,
    last_checkpoint_receipt: Option<CanonicalTrendbarCheckpointReceiptV1>,
    source: Error,
}

impl CanonicalTrendbarAcquisitionRunFailureV1 {
    pub const fn stage(&self) -> CanonicalTrendbarAcquisitionRunStageV1 {
        self.stage
    }

    pub const fn last_checkpoint_receipt(&self) -> Option<&CanonicalTrendbarCheckpointReceiptV1> {
        self.last_checkpoint_receipt.as_ref()
    }

    pub(crate) fn new(
        stage: CanonicalTrendbarAcquisitionRunStageV1,
        last_checkpoint_receipt: Option<CanonicalTrendbarCheckpointReceiptV1>,
        source: Error,
    ) -> Self {
        Self {
            stage,
            last_checkpoint_receipt,
            source,
        }
    }
}

impl fmt::Display for CanonicalTrendbarAcquisitionRunFailureV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "canonical trendbar acquisition failed at {:?}: {}",
            self.stage, self.source
        )?;
        if let Some(receipt) = &self.last_checkpoint_receipt {
            write!(formatter, "; resume_checkpoint_sha256={}", receipt.sha256())?;
        }
        Ok(())
    }
}

impl std::error::Error for CanonicalTrendbarAcquisitionRunFailureV1 {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.source.as_ref())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalTrendbarAcquisitionRunOutcomeV1 {
    plan_receipt: CanonicalTrendbarPlanReceiptV1,
    checkpoint_receipt: CanonicalTrendbarCheckpointReceiptV1,
    matrix_receipt: CanonicalTrendbarMatrixReceiptV1,
    completed_cells: usize,
    total_cells: usize,
}

impl CanonicalTrendbarAcquisitionRunOutcomeV1 {
    pub const fn plan_receipt(&self) -> &CanonicalTrendbarPlanReceiptV1 {
        &self.plan_receipt
    }

    pub const fn checkpoint_receipt(&self) -> &CanonicalTrendbarCheckpointReceiptV1 {
        &self.checkpoint_receipt
    }

    pub const fn matrix_receipt(&self) -> &CanonicalTrendbarMatrixReceiptV1 {
        &self.matrix_receipt
    }

    pub const fn completed_cells(&self) -> usize {
        self.completed_cells
    }

    pub const fn total_cells(&self) -> usize {
        self.total_cells
    }
}

pub fn resume_canonical_trendbar_acquisition_v1_with<F>(
    data_root: &Path,
    store: &CanonicalTrendbarAcquisitionStoreV1,
    plan_receipt: &CanonicalTrendbarPlanReceiptV1,
    checkpoint_receipt: Option<&CanonicalTrendbarCheckpointReceiptV1>,
    mut capture: F,
) -> Result<CanonicalTrendbarAcquisitionRunOutcomeV1, CanonicalTrendbarAcquisitionRunFailureV1>
where
    F: FnMut(
        HistoricalCaptureRequest,
    ) -> anyhow::Result<neoethos_data::SelectedDatasetGenerationV1>,
{
    let plan = store.open_plan(plan_receipt).map_err(|source| {
        CanonicalTrendbarAcquisitionRunFailureV1::new(
            CanonicalTrendbarAcquisitionRunStageV1::Plan,
            checkpoint_receipt.cloned(),
            source,
        )
    })?;
    let mut last_checkpoint_receipt = checkpoint_receipt.cloned();
    let mut completed_cells = match checkpoint_receipt {
        Some(receipt) => store
            .open_checkpoint(data_root, plan_receipt, receipt)
            .map_err(|source| {
                CanonicalTrendbarAcquisitionRunFailureV1::new(
                    CanonicalTrendbarAcquisitionRunStageV1::Checkpoint,
                    last_checkpoint_receipt.clone(),
                    source,
                )
            })?
            .completed_cells()
            .to_vec(),
        None => Vec::new(),
    };

    while completed_cells.len() < plan.cell_count() {
        let index = completed_cells.len();
        let timeframe_count = plan.timeframes().len();
        let symbol = plan
            .symbols()
            .get(index / timeframe_count)
            .expect("validated plan cell index has a symbol");
        let timeframe = *plan
            .timeframes()
            .get(index % timeframe_count)
            .expect("validated plan cell index has a timeframe");
        let request = HistoricalCaptureRequest {
            symbol: symbol.symbol_name().to_owned(),
            timeframe,
            from_ms: plan.from_ms(),
            to_ms: plan.to_ms_exclusive(),
            data_root: data_root.to_path_buf(),
            target: HistoricalCaptureTarget::NewIdentity,
        };
        let selected_generation = capture(request).map_err(|source| {
            CanonicalTrendbarAcquisitionRunFailureV1::new(
                CanonicalTrendbarAcquisitionRunStageV1::Capture,
                last_checkpoint_receipt.clone(),
                source,
            )
        })?;
        let cell =
            CanonicalTrendbarAcquisitionCellV1::new(selected_generation).map_err(|source| {
                CanonicalTrendbarAcquisitionRunFailureV1::new(
                    CanonicalTrendbarAcquisitionRunStageV1::Checkpoint,
                    last_checkpoint_receipt.clone(),
                    source,
                )
            })?;
        let mut next_completed_cells = completed_cells.clone();
        next_completed_cells.push(cell);
        let next_checkpoint = store
            .publish_checkpoint(
                data_root,
                plan_receipt,
                last_checkpoint_receipt.as_ref(),
                next_completed_cells.clone(),
            )
            .map_err(|source| {
                CanonicalTrendbarAcquisitionRunFailureV1::new(
                    CanonicalTrendbarAcquisitionRunStageV1::Checkpoint,
                    last_checkpoint_receipt.clone(),
                    source,
                )
            })?;
        completed_cells = next_completed_cells;
        last_checkpoint_receipt = Some(next_checkpoint);
    }

    let checkpoint_receipt = last_checkpoint_receipt.expect("validated plan contains cells");
    let matrix_receipt = store
        .publish_matrix(data_root, plan_receipt, &checkpoint_receipt)
        .map_err(|source| {
            CanonicalTrendbarAcquisitionRunFailureV1::new(
                CanonicalTrendbarAcquisitionRunStageV1::Matrix,
                Some(checkpoint_receipt.clone()),
                source,
            )
        })?;
    Ok(CanonicalTrendbarAcquisitionRunOutcomeV1 {
        plan_receipt: plan_receipt.clone(),
        checkpoint_receipt,
        matrix_receipt,
        completed_cells: completed_cells.len(),
        total_cells: plan.cell_count(),
    })
}

pub fn run_production_canonical_trendbar_acquisition_v1(
    data_root: &Path,
    store: &CanonicalTrendbarAcquisitionStoreV1,
    plan_receipt: &CanonicalTrendbarPlanReceiptV1,
    checkpoint_receipt: Option<&CanonicalTrendbarCheckpointReceiptV1>,
    active_fetch: &ProcessHistoricalCapture,
) -> Result<CanonicalTrendbarAcquisitionRunOutcomeV1, CanonicalTrendbarAcquisitionRunFailureV1> {
    let plan = store.open_plan(plan_receipt).map_err(|source| {
        CanonicalTrendbarAcquisitionRunFailureV1::new(
            CanonicalTrendbarAcquisitionRunStageV1::Plan,
            checkpoint_receipt.cloned(),
            source,
        )
    })?;
    let environment = BrokerEnvironment::from_canonical(plan.environment());
    if plan.server() != environment.endpoint_host() {
        return Err(CanonicalTrendbarAcquisitionRunFailureV1::new(
            CanonicalTrendbarAcquisitionRunStageV1::Plan,
            checkpoint_receipt.cloned(),
            anyhow::anyhow!("canonical trendbar plan server differs from its exact environment"),
        ));
    }
    let credentials = load_exact_production_historical_credentials(environment, plan.account_id())
        .map_err(|source| {
            CanonicalTrendbarAcquisitionRunFailureV1::new(
                CanonicalTrendbarAcquisitionRunStageV1::Credentials,
                checkpoint_receipt.cloned(),
                source,
            )
        })?;
    let mut series_capture =
        HistoricalSeriesCapture::new(credentials, ProductionHistoricalSessionConnector);
    resume_canonical_trendbar_acquisition_v1_with(
        data_root,
        store,
        plan_receipt,
        checkpoint_receipt,
        |request| {
            series_capture
                .capture_historical_series_generation(request, active_fetch)
                .map(|outcome| outcome.selected_generation)
        },
    )
}
