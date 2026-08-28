//! Narrow library adapters used by the `neoethos-app` binary entrypoint.
//!
//! The binary is a separate crate target, so it cannot consume the
//! crate-private lease/job types directly. This module keeps those authorities
//! private while returning one move-only handle that the headless process must
//! retain until shutdown.

use std::fmt;

use neoethos_data::CanonicalTimeframe;

use crate::server::engines_control::{
    TypedDiscoveryDatasetPolicyV1, TypedDiscoveryExecutionIntentV1, TypedDiscoveryOverridesV1,
    TypedDiscoverySettingsGateV1, TypedHigherTimeframePolicyV1, TypedLegacyExecutionJobHandleV1,
    TypedLegacyExecutionStartErrorV1, TypedLegacyExecutionTerminalV1,
    TypedTrainingExecutionIntentV1, TypedTrainingSelectionPolicyV1,
    start_typed_discovery_execution_v1, start_typed_training_execution_v1,
};
use crate::server::state::AppApiState;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadlessExecutionPipelineIntentV1 {
    symbol: String,
    base_timeframe: CanonicalTimeframe,
    auto_discovery: bool,
    auto_training: bool,
}

impl HeadlessExecutionPipelineIntentV1 {
    pub fn checked_new(
        symbol: String,
        base_timeframe: String,
        auto_discovery: bool,
        auto_training: bool,
    ) -> Result<Self, HeadlessExecutionStartErrorV1> {
        let symbol = symbol.trim().to_uppercase();
        if symbol.is_empty() {
            return Err(HeadlessExecutionStartErrorV1::InvalidIntent(
                "headless execution requires a configured symbol".to_owned(),
            ));
        }
        let base_label = base_timeframe.trim().to_uppercase();
        let base_timeframe = base_label.parse::<CanonicalTimeframe>().map_err(|error| {
            HeadlessExecutionStartErrorV1::InvalidIntent(format!(
                "invalid configured base timeframe {base_label:?}: {error}"
            ))
        })?;
        Ok(Self {
            symbol,
            base_timeframe,
            auto_discovery,
            auto_training,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadlessExecutionStartErrorV1 {
    InvalidIntent(String),
    Busy { requested: String, active: String },
    RuntimeUnavailable(String),
}

impl fmt::Display for HeadlessExecutionStartErrorV1 {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIntent(detail) => output.write_str(detail),
            Self::Busy { requested, active } => write!(
                output,
                "headless execution is busy: requested {requested} while {active} is active"
            ),
            Self::RuntimeUnavailable(detail) => {
                write!(
                    output,
                    "headless execution runtime is unavailable: {detail}"
                )
            }
        }
    }
}

impl std::error::Error for HeadlessExecutionStartErrorV1 {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadlessExecutionTerminalStateV1 {
    Succeeded,
    Failed,
    Cancelled,
    WorkerPanicked,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadlessExecutionTerminalV1 {
    state: HeadlessExecutionTerminalStateV1,
    lease_token: u64,
    completed_kind: Option<String>,
    summary: String,
}

impl HeadlessExecutionTerminalV1 {
    pub const fn state(&self) -> HeadlessExecutionTerminalStateV1 {
        self.state
    }

    pub const fn lease_token(&self) -> u64 {
        self.lease_token
    }

    pub fn completed_kind(&self) -> Option<&str> {
        self.completed_kind.as_deref()
    }

    pub fn summary(&self) -> &str {
        &self.summary
    }
}

pub struct HeadlessExecutionHandleV1 {
    inner: TypedLegacyExecutionJobHandleV1,
}

impl HeadlessExecutionHandleV1 {
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    pub async fn await_terminal(self) -> HeadlessExecutionTerminalV1 {
        map_terminal_v1(self.inner.await_terminal().await)
    }
}

pub fn run_headless_execution_pipeline_v1(
    state: AppApiState,
    intent: HeadlessExecutionPipelineIntentV1,
) -> Result<Option<HeadlessExecutionHandleV1>, HeadlessExecutionStartErrorV1> {
    if !intent.auto_discovery && !intent.auto_training {
        return Ok(None);
    }

    let inner = if intent.auto_discovery {
        start_typed_discovery_execution_v1(
            state,
            TypedDiscoveryExecutionIntentV1 {
                training_after_success: intent.auto_training,
                symbol: intent.symbol,
                base_timeframe: intent.base_timeframe,
                higher_timeframes: TypedHigherTimeframePolicyV1::Configured,
                overrides: TypedDiscoveryOverridesV1::default(),
                settings_gate: TypedDiscoverySettingsGateV1::None,
                dataset_policy: TypedDiscoveryDatasetPolicyV1::Current,
            },
        )
    } else {
        start_typed_training_execution_v1(
            state,
            TypedTrainingExecutionIntentV1 {
                selection: TypedTrainingSelectionPolicyV1::Exact {
                    symbol: intent.symbol,
                    base_timeframe: intent.base_timeframe,
                },
            },
        )
    }
    .map_err(map_start_error_v1)?;

    Ok(Some(HeadlessExecutionHandleV1 { inner }))
}

fn map_start_error_v1(error: TypedLegacyExecutionStartErrorV1) -> HeadlessExecutionStartErrorV1 {
    match error {
        TypedLegacyExecutionStartErrorV1::Busy(busy) => HeadlessExecutionStartErrorV1::Busy {
            requested: busy.requested().to_string(),
            active: busy.active().to_string(),
        },
        TypedLegacyExecutionStartErrorV1::RuntimeUnavailable(detail) => {
            HeadlessExecutionStartErrorV1::RuntimeUnavailable(detail)
        }
    }
}

fn map_terminal_v1(terminal: TypedLegacyExecutionTerminalV1) -> HeadlessExecutionTerminalV1 {
    match terminal {
        TypedLegacyExecutionTerminalV1::Succeeded {
            final_snapshot,
            lease_token,
            completed_kind,
        } => HeadlessExecutionTerminalV1 {
            state: HeadlessExecutionTerminalStateV1::Succeeded,
            lease_token,
            completed_kind: Some(format!("{completed_kind:?}")),
            summary: final_snapshot.report.summary,
        },
        TypedLegacyExecutionTerminalV1::Failed {
            final_snapshot,
            lease_token,
            detail,
        } => HeadlessExecutionTerminalV1 {
            state: HeadlessExecutionTerminalStateV1::Failed,
            lease_token,
            completed_kind: Some(format!("{:?}", final_snapshot.kind)),
            summary: detail,
        },
        TypedLegacyExecutionTerminalV1::Cancelled {
            final_snapshot,
            lease_token,
        } => HeadlessExecutionTerminalV1 {
            state: HeadlessExecutionTerminalStateV1::Cancelled,
            lease_token,
            completed_kind: Some(format!("{:?}", final_snapshot.kind)),
            summary: final_snapshot.report.summary,
        },
        TypedLegacyExecutionTerminalV1::WorkerPanicked {
            lease_token,
            detail,
        } => HeadlessExecutionTerminalV1 {
            state: HeadlessExecutionTerminalStateV1::WorkerPanicked,
            lease_token,
            completed_kind: None,
            summary: detail,
        },
    }
}
