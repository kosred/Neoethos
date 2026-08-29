use std::fmt;
use std::sync::{Mutex, MutexGuard};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessExecutionKindV1 {
    Discovery,
    Training,
    NativeResearch,
    Migration,
}

impl fmt::Display for ProcessExecutionKindV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Discovery => "Discovery",
            Self::Training => "Training",
            Self::NativeResearch => "NativeResearch",
            Self::Migration => "Migration",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcessExecutionBusyV1 {
    requested: ProcessExecutionKindV1,
    active: ProcessExecutionKindV1,
}

impl ProcessExecutionBusyV1 {
    pub const fn requested(&self) -> ProcessExecutionKindV1 {
        self.requested
    }

    pub const fn active(&self) -> ProcessExecutionKindV1 {
        self.active
    }
}

impl fmt::Display for ProcessExecutionBusyV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "process execution is busy: requested {} while {} is active",
            self.requested, self.active
        )
    }
}

impl std::error::Error for ProcessExecutionBusyV1 {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessExecutionLeaseTransitionErrorV1 {
    InvalidSource { current: ProcessExecutionKindV1 },
    AlreadyTransitioned,
    AuthorityLost,
}

impl fmt::Display for ProcessExecutionLeaseTransitionErrorV1 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSource { current } => write!(
                formatter,
                "Discovery-to-Training transition requires Discovery authority, not {current}"
            ),
            Self::AlreadyTransitioned => {
                formatter.write_str("Discovery-to-Training transition was already consumed")
            }
            Self::AuthorityLost => formatter
                .write_str("process execution lease no longer matches the process-wide authority"),
        }
    }
}

impl std::error::Error for ProcessExecutionLeaseTransitionErrorV1 {}

#[derive(Clone, Copy, Debug)]
struct ActiveProcessExecutionV1 {
    token: u64,
    kind: ProcessExecutionKindV1,
    discovery_to_training_transitioned: bool,
}

#[derive(Debug)]
struct ProcessExecutionCoordinatorStateV1 {
    active: Option<ActiveProcessExecutionV1>,
    next_token: u64,
}

static PROCESS_EXECUTION_COORDINATOR_V1: Mutex<ProcessExecutionCoordinatorStateV1> =
    Mutex::new(ProcessExecutionCoordinatorStateV1 {
        active: None,
        next_token: 1,
    });

fn lock_process_execution_coordinator_v1() -> MutexGuard<'static, ProcessExecutionCoordinatorStateV1>
{
    PROCESS_EXECUTION_COORDINATOR_V1
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn take_next_nonzero_token_v1(state: &mut ProcessExecutionCoordinatorStateV1) -> u64 {
    let token = state.next_token.max(1);
    state.next_token = token.wrapping_add(1).max(1);
    token
}

#[must_use = "dropping the lease releases the process-wide execution authority"]
#[derive(Debug)]
pub struct ProcessExecutionLeaseV1 {
    token: u64,
    kind: ProcessExecutionKindV1,
    discovery_to_training_transitioned: bool,
}

impl ProcessExecutionLeaseV1 {
    pub const fn token(&self) -> u64 {
        self.token
    }

    pub const fn kind(&self) -> ProcessExecutionKindV1 {
        self.kind
    }

    pub fn transition_discovery_to_training_v1(
        &mut self,
    ) -> Result<(), ProcessExecutionLeaseTransitionErrorV1> {
        if self.discovery_to_training_transitioned {
            return Err(ProcessExecutionLeaseTransitionErrorV1::AlreadyTransitioned);
        }
        if self.kind != ProcessExecutionKindV1::Discovery {
            return Err(ProcessExecutionLeaseTransitionErrorV1::InvalidSource {
                current: self.kind,
            });
        }

        let mut state = lock_process_execution_coordinator_v1();
        let Some(active) = state.active.as_mut() else {
            return Err(ProcessExecutionLeaseTransitionErrorV1::AuthorityLost);
        };
        if active.token != self.token
            || active.kind != ProcessExecutionKindV1::Discovery
            || active.discovery_to_training_transitioned
        {
            return Err(ProcessExecutionLeaseTransitionErrorV1::AuthorityLost);
        }

        active.kind = ProcessExecutionKindV1::Training;
        active.discovery_to_training_transitioned = true;
        self.kind = ProcessExecutionKindV1::Training;
        self.discovery_to_training_transitioned = true;
        Ok(())
    }
}

impl Drop for ProcessExecutionLeaseV1 {
    fn drop(&mut self) {
        let mut state = lock_process_execution_coordinator_v1();
        if state.active.as_ref().map(|active| active.token) == Some(self.token) {
            state.active = None;
        }
    }
}

pub fn try_acquire_process_execution_lease_v1(
    requested: ProcessExecutionKindV1,
) -> Result<ProcessExecutionLeaseV1, ProcessExecutionBusyV1> {
    let mut state = lock_process_execution_coordinator_v1();
    if let Some(active) = state.active {
        return Err(ProcessExecutionBusyV1 {
            requested,
            active: active.kind,
        });
    }

    let token = take_next_nonzero_token_v1(&mut state);
    state.active = Some(ActiveProcessExecutionV1 {
        token,
        kind: requested,
        discovery_to_training_transitioned: false,
    });
    Ok(ProcessExecutionLeaseV1 {
        token,
        kind: requested,
        discovery_to_training_transitioned: false,
    })
}
