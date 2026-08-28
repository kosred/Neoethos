//! Control endpoints for the Discovery and Training engines.
//!
//! POST /engines/discovery/start  — kick off a discovery job
//! POST /engines/discovery/stop   — request cancellation
//! POST /engines/training/start   — kick off a training job
//! POST /engines/training/stop    — request cancellation
//!
//! Each engine has at most one in-flight job at a time. Starting while
//! one is already running returns 409 Conflict. Stopping when nothing is
//! running returns 200 with `{"running": false}` — idempotent.
//!
//! Engine state ("Idle" / "Running" / "Failed: …" / "Succeeded") is
//! tracked through a `EngineSlot` held inside `AppApiState`. The
//! background task that drives each job drains the `ServiceEvent`
//! channel and writes the latest `JobState` back into the slot, which
//! `/engines/status` then reads.

#[expect(
    dead_code,
    reason = "Chunk4C entrypoint adapters consume this crate-private typed boundary"
)]
mod typed_execution_v1;
#[allow(
    unused_imports,
    reason = "Chunk4C entrypoint adapters consume this crate-private typed boundary"
)]
pub(crate) use typed_execution_v1::{
    TypedDiscoveryDatasetPolicyV1, TypedDiscoveryExecutionIntentV1,
    TypedDiscoveryGenerationOverrideV1, TypedDiscoveryOverridesV1, TypedDiscoverySettingsGateV1,
    TypedHigherTimeframePolicyV1, TypedLegacyExecutionAdmissionErrorV1,
    TypedLegacyExecutionAdmissionV1, TypedLegacyExecutionJobHandleV1,
    TypedLegacyExecutionSnapshotV1, TypedLegacyExecutionStartErrorV1,
    TypedLegacyExecutionTerminalV1, TypedTrainingExecutionIntentV1, TypedTrainingSelectionPolicyV1,
    detach_typed_legacy_execution_observer_v1, start_typed_discovery_execution_v1,
    start_typed_training_execution_v1,
};

use anyhow::Result;
use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_data::{CanonicalTimeframe, SelectedDatasetGenerationV1};
use neoethos_search::{
    CanonicalNativeGenerationZeroOverridesV1, CanonicalNativeRuntimeInstallReceiptV1,
    CanonicalResearchContractArtifactRefV1, ProcessExecutionKindV1,
};
use std::sync::Arc;

use crate::app_services::ServiceEvent;
use crate::app_services::canonical_native_discovery::{
    CanonicalNativeResearchEventV1, CanonicalNativeResearchIntentV1,
    CanonicalNativeResearchJobHandleV1, CanonicalNativeResearchSnapshotV1,
    CanonicalNativeResearchStartErrorV1, start_canonical_native_research_lane_v1,
};
use crate::app_services::jobs::{JobKind, JobState};

use super::errors::actionable_error;
use super::state::{AppApiState, CanonicalNativeResearchCancellationOutcomeV1};

/// Shared request body for `start` endpoints. Discovery requires an exact
/// canonical `dataset_selection`; the legacy symbol/base fields may only
/// assert consistency with it. Training still resolves symbol/base through
/// its existing configuration path because it shares this wire type.
///
/// `higher_tfs` is the MTF context discovery considers alongside
/// `base_tf`. When omitted, the server resolves the operator's configured
/// ladder via `SystemConfig::resolve_higher_timeframes` (honouring
/// `multi_resolution_timeframes` / `higher_timeframes`) — IDENTICAL to a
/// CLI `discover` with no `--higher`. The wire form is a JSON array of
/// canonical timeframe labels (`["M5", "M15", "H1"]`).
#[derive(Debug, Default, serde::Deserialize)]
#[serde(default)]
pub struct StartJobBody {
    /// Exact identity + immutable generation + manifest binding selected from
    /// the data inventory. Optional here only because Training shares this
    /// request body; Discovery rejects `None` before any data lookup.
    pub dataset_selection: Option<SelectedDatasetGenerationV1>,
    pub symbol: Option<String>,
    pub base_tf: Option<String>,
    pub higher_tfs: Option<Vec<String>>,
    /// #194: optional GA hyperparameter overrides. When `None` the
    /// engine uses the defaults baked into
    /// `neoethos_search::DiscoveryConfig::default()`; sending any field
    /// here replaces only that knob. The UI's "Advanced" expander
    /// builds this struct from the operator's sliders.
    pub population: Option<usize>,
    pub generations: Option<usize>,
    pub max_indicators: Option<usize>,
    pub target_candidates: Option<usize>,
    pub portfolio_size: Option<usize>,
}

fn resolve_discovery_selection(body: &StartJobBody) -> Result<SelectedDatasetGenerationV1> {
    let selected = body
        .dataset_selection
        .clone()
        .ok_or_else(|| anyhow::anyhow!("dataset_selection is required for Discovery"))?;
    selected.validate()?;
    let identity = selected.identity();
    if let Some(asserted) = body
        .symbol
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        anyhow::ensure!(
            asserted.eq_ignore_ascii_case(identity.symbol_name()),
            "legacy symbol assertion {asserted:?} disagrees with exact dataset identity symbol {:?}",
            identity.symbol_name()
        );
    }
    if let Some(asserted) = body
        .base_tf
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        anyhow::ensure!(
            asserted.eq_ignore_ascii_case(identity.timeframe().as_str()),
            "legacy base timeframe assertion {asserted:?} disagrees with exact dataset identity timeframe {:?}",
            identity.timeframe().as_str()
        );
    }
    Ok(selected)
}

#[derive(Debug, serde::Serialize)]
pub struct StartResponse {
    pub started: bool,
    pub kind: &'static str,
    pub symbol: String,
    pub base_tf: String,
    pub dataset_identity: Option<String>,
    pub dataset_generation: Option<String>,
    pub manifest_binding_sha256: Option<String>,
}

#[derive(Debug, serde::Serialize)]
pub struct StopResponse {
    pub running: bool,
    pub kind: &'static str,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CanonicalNativeResearchContractArtifactBodyV1 {
    pub relative_path: String,
    pub expected_sha256: String,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CanonicalNativeResearchStartBodyV1 {
    pub contract_artifact: CanonicalNativeResearchContractArtifactBodyV1,
    #[serde(default)]
    pub population: Option<usize>,
    #[serde(default)]
    pub population_auto: Option<bool>,
    #[serde(default)]
    pub max_indicators: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalNativeResearchStartResponseV1 {
    pub started: bool,
    pub kind: &'static str,
    pub lease_token: String,
    pub state: &'static str,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalNativeResearchStartErrorResponseV1 {
    pub started: bool,
    pub kind: &'static str,
    pub error_code: &'static str,
    pub detail: String,
    pub requested_kind: Option<&'static str>,
    pub active_kind: Option<&'static str>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CanonicalNativeResearchCancelBodyV1 {
    pub lease_token: String,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalNativeResearchCancelResponseV1 {
    pub cancellation_requested: bool,
    pub kind: &'static str,
    pub lease_token: String,
    pub state: &'static str,
    pub error_code: Option<&'static str>,
}

pub async fn canonical_native_research_start(
    State(state): State<AppApiState>,
    Json(body): Json<CanonicalNativeResearchStartBodyV1>,
) -> Response {
    let contract_ref = match CanonicalResearchContractArtifactRefV1::checked_new(
        body.contract_artifact.relative_path,
        body.contract_artifact.expected_sha256,
    ) {
        Ok(reference) => reference,
        Err(error) => {
            return native_start_error_response_v1(
                StatusCode::BAD_REQUEST,
                "invalid_contract_artifact_reference",
                error.to_string(),
                None,
                None,
            );
        }
    };
    let overrides = match CanonicalNativeGenerationZeroOverridesV1::checked_new(
        body.population,
        body.population_auto,
        body.max_indicators,
    ) {
        Ok(overrides) => overrides,
        Err(error) => {
            return native_start_error_response_v1(
                StatusCode::BAD_REQUEST,
                "invalid_generation_zero_overrides",
                error.to_string(),
                None,
                None,
            );
        }
    };
    let Some(authority) = state.canonical_native_startup_authority_v1() else {
        return native_start_error_response_v1(
            StatusCode::SERVICE_UNAVAILABLE,
            "native_runtime_authority_unavailable",
            "canonical native startup authority is not installed".to_owned(),
            None,
            None,
        );
    };
    let intent = CanonicalNativeResearchIntentV1::new(contract_ref, overrides);
    match start_and_observe_canonical_native_research_v1(
        state,
        authority.settings(),
        authority.runtime_install_receipt(),
        intent,
    )
    .await
    {
        Ok(lease_token) => (
            StatusCode::ACCEPTED,
            Json(CanonicalNativeResearchStartResponseV1 {
                started: true,
                kind: "canonical_native_research",
                lease_token: lease_token.to_string(),
                state: "Queued",
            }),
        )
            .into_response(),
        Err(CanonicalNativeResearchStartErrorV1::Busy(error)) => native_start_error_response_v1(
            StatusCode::CONFLICT,
            "process_execution_busy",
            error.to_string(),
            Some(process_execution_kind_wire_v1(error.requested())),
            Some(process_execution_kind_wire_v1(error.active())),
        ),
        Err(CanonicalNativeResearchStartErrorV1::RuntimeUnavailable(detail)) => {
            native_start_error_response_v1(
                StatusCode::SERVICE_UNAVAILABLE,
                "native_runtime_unavailable",
                detail,
                None,
                None,
            )
        }
    }
}

pub async fn canonical_native_research_cancel(
    State(state): State<AppApiState>,
    Json(body): Json<CanonicalNativeResearchCancelBodyV1>,
) -> Response {
    let lease_token = match body
        .lease_token
        .parse::<u64>()
        .ok()
        .filter(|token| *token != 0)
    {
        Some(token) => token,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(CanonicalNativeResearchCancelResponseV1 {
                    cancellation_requested: false,
                    kind: "canonical_native_research",
                    lease_token: body.lease_token,
                    state: "Invalid",
                    error_code: Some("invalid_native_research_lease_token"),
                }),
            )
                .into_response();
        }
    };
    let (status, cancellation_requested, state_name, error_code) = match state
        .cancel_canonical_native_research_exact_v1(lease_token)
        .await
    {
        CanonicalNativeResearchCancellationOutcomeV1::Requested => {
            (StatusCode::ACCEPTED, true, "Running", None)
        }
        CanonicalNativeResearchCancellationOutcomeV1::AlreadyRequested => {
            (StatusCode::OK, true, "Running", None)
        }
        CanonicalNativeResearchCancellationOutcomeV1::NotRunning => (
            StatusCode::CONFLICT,
            false,
            "Idle",
            Some("native_research_not_running"),
        ),
        CanonicalNativeResearchCancellationOutcomeV1::TokenMismatch => (
            StatusCode::CONFLICT,
            false,
            "Running",
            Some("native_research_token_mismatch"),
        ),
    };
    (
        status,
        Json(CanonicalNativeResearchCancelResponseV1 {
            cancellation_requested,
            kind: "canonical_native_research",
            lease_token: body.lease_token,
            state: state_name,
            error_code,
        }),
    )
        .into_response()
}

fn native_start_error_response_v1(
    status: StatusCode,
    error_code: &'static str,
    detail: String,
    requested_kind: Option<&'static str>,
    active_kind: Option<&'static str>,
) -> Response {
    (
        status,
        Json(CanonicalNativeResearchStartErrorResponseV1 {
            started: false,
            kind: "canonical_native_research",
            error_code,
            detail,
            requested_kind,
            active_kind,
        }),
    )
        .into_response()
}

const fn process_execution_kind_wire_v1(kind: ProcessExecutionKindV1) -> &'static str {
    match kind {
        ProcessExecutionKindV1::Discovery => "discovery",
        ProcessExecutionKindV1::Training => "training",
        ProcessExecutionKindV1::NativeResearch => "canonical_native_research",
        ProcessExecutionKindV1::Migration => "migration",
    }
}

// ─── Discovery ────────────────────────────────────────────────────────────

pub async fn discovery_start(
    State(state): State<AppApiState>,
    body: Option<Json<StartJobBody>>,
) -> Response {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let dataset_selection = match resolve_discovery_selection(&body) {
        Ok(selection) => selection,
        Err(error) => {
            return actionable_error(
                StatusCode::BAD_REQUEST,
                "Discovery requires the exact canonical dataset generation selected from Data. Symbol/timeframe alone are not a safe selector.",
                &error,
            );
        }
    };
    let dataset_identity = dataset_selection.identity().clone();
    let symbol = dataset_identity.symbol_name().to_owned();
    let base_timeframe = dataset_identity.timeframe();
    let higher_timeframes = match body.higher_tfs {
        Some(labels) => {
            let parsed = labels
                .into_iter()
                .map(|label| {
                    label
                        .trim()
                        .parse::<CanonicalTimeframe>()
                        .map_err(|error| anyhow::anyhow!(error.to_string()))
                })
                .collect::<Result<Vec<_>>>();
            match parsed {
                Ok(timeframes) => TypedHigherTimeframePolicyV1::Exact(timeframes),
                Err(error) => {
                    return actionable_error(
                        StatusCode::BAD_REQUEST,
                        "Discovery higher_tfs contains a non-canonical timeframe.",
                        &error,
                    );
                }
            }
        }
        None => TypedHigherTimeframePolicyV1::Configured,
    };
    let overrides = match TypedDiscoveryOverridesV1::checked_new(
        body.population.filter(|value| *value > 0),
        body.generations
            .filter(|value| *value > 0)
            .map(TypedDiscoveryGenerationOverrideV1::Exact),
        body.max_indicators.filter(|value| *value > 0),
        None,
        body.target_candidates.filter(|value| *value > 0),
        body.portfolio_size.filter(|value| *value > 0),
    ) {
        Ok(overrides) => overrides,
        Err(detail) => {
            return actionable_error(
                StatusCode::BAD_REQUEST,
                "Discovery overrides are invalid.",
                &anyhow::anyhow!(detail),
            );
        }
    };
    let intent = TypedDiscoveryExecutionIntentV1 {
        symbol: symbol.clone(),
        base_timeframe,
        higher_timeframes,
        overrides,
        settings_gate: TypedDiscoverySettingsGateV1::None,
        dataset_policy: TypedDiscoveryDatasetPolicyV1::Exact(dataset_selection.clone()),
        training_after_success: true,
    };
    let mut handle = match start_typed_discovery_execution_v1(state.clone(), intent) {
        Ok(handle) => handle,
        Err(error) => return typed_legacy_start_error_response_v1("Discovery", error),
    };
    let admitted = match handle.await_admission_v1().await {
        Ok(TypedLegacyExecutionAdmissionV1::Discovery {
            selected_generation,
        }) => selected_generation,
        Ok(TypedLegacyExecutionAdmissionV1::Training { .. }) => {
            handle.cancel();
            let _ = handle.await_terminal().await;
            return actionable_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Discovery worker returned the wrong admission evidence.",
                &anyhow::anyhow!("typed Discovery admission was Training"),
            );
        }
        Err(error) => {
            let response = typed_legacy_admission_error_response_v1("Discovery", &error);
            let _ = handle.await_terminal().await;
            return response;
        }
    };
    if admitted != dataset_selection {
        handle.cancel();
        let _ = handle.await_terminal().await;
        return actionable_error(
            StatusCode::CONFLICT,
            "Discovery pinned a different dataset generation. Refresh Data and explicitly select the current generation.",
            &anyhow::anyhow!("typed Discovery admission differs from the requested generation"),
        );
    }
    detach_typed_legacy_execution_observer_v1(state, handle);
    Json(StartResponse {
        started: true,
        kind: "discovery",
        symbol,
        base_tf: base_timeframe.as_str().to_owned(),
        dataset_identity: Some(dataset_identity.to_path_component()),
        dataset_generation: Some(admitted.generation_id().to_owned()),
        manifest_binding_sha256: Some(admitted.manifest_binding_sha256().to_owned()),
    })
    .into_response()
}

pub async fn discovery_stop(State(state): State<AppApiState>) -> Json<StopResponse> {
    let running = state.cancel_engine(JobKind::Discovery).await;
    Json(StopResponse {
        running,
        kind: "discovery",
    })
}

// ─── Training ─────────────────────────────────────────────────────────────

pub async fn training_start(
    State(state): State<AppApiState>,
    body: Option<Json<StartJobBody>>,
) -> Response {
    let body = body.map(|Json(b)| b).unwrap_or_default();
    let selection = match (body.symbol, body.base_tf) {
        (None, None) => TypedTrainingSelectionPolicyV1::Configured,
        (Some(symbol), Some(base_tf)) => {
            let symbol = symbol.trim().to_uppercase();
            if symbol.is_empty() {
                return actionable_error(
                    StatusCode::BAD_REQUEST,
                    "Training symbol must not be empty.",
                    &anyhow::anyhow!("empty Training symbol"),
                );
            }
            let base_timeframe = match base_tf.trim().parse::<CanonicalTimeframe>() {
                Ok(timeframe) => timeframe,
                Err(error) => {
                    return actionable_error(
                        StatusCode::BAD_REQUEST,
                        "Training base_tf must be a canonical timeframe.",
                        &anyhow::anyhow!(error.to_string()),
                    );
                }
            };
            TypedTrainingSelectionPolicyV1::Exact {
                symbol,
                base_timeframe,
            }
        }
        _ => {
            return actionable_error(
                StatusCode::BAD_REQUEST,
                "Training requires both symbol and base_tf together, or neither to use Settings.",
                &anyhow::anyhow!("partial Training symbol/base_tf selection"),
            );
        }
    };
    let mut handle = match start_typed_training_execution_v1(
        state.clone(),
        TypedTrainingExecutionIntentV1 { selection },
    ) {
        Ok(handle) => handle,
        Err(error) => return typed_legacy_start_error_response_v1("Training", error),
    };
    let (symbol, base_timeframe) = match handle.await_admission_v1().await {
        Ok(TypedLegacyExecutionAdmissionV1::Training {
            symbol,
            base_timeframe,
        }) => (symbol, base_timeframe),
        Ok(TypedLegacyExecutionAdmissionV1::Discovery { .. }) => {
            handle.cancel();
            let _ = handle.await_terminal().await;
            return actionable_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Training worker returned the wrong admission evidence.",
                &anyhow::anyhow!("typed Training admission was Discovery"),
            );
        }
        Err(error) => {
            let response = typed_legacy_admission_error_response_v1("Training", &error);
            let _ = handle.await_terminal().await;
            return response;
        }
    };
    detach_typed_legacy_execution_observer_v1(state, handle);
    Json(StartResponse {
        started: true,
        kind: "training",
        symbol,
        base_tf: base_timeframe.as_str().to_owned(),
        dataset_identity: None,
        dataset_generation: None,
        manifest_binding_sha256: None,
    })
    .into_response()
}

pub async fn training_stop(State(state): State<AppApiState>) -> Json<StopResponse> {
    let running = state.cancel_engine(JobKind::Training).await;
    Json(StopResponse {
        running,
        kind: "training",
    })
}

/// Typed in-process native entry used by the forthcoming frontend adapters.
/// There is intentionally no JSON self-call and no legacy Discovery result.
#[allow(
    dead_code,
    reason = "Chunk4C entrypoint adapters consume this crate-private native boundary"
)]
pub(crate) async fn start_and_observe_canonical_native_research_v1(
    state: AppApiState,
    startup_settings: Arc<Settings>,
    runtime_install_receipt: Arc<CanonicalNativeRuntimeInstallReceiptV1>,
    intent: CanonicalNativeResearchIntentV1,
) -> Result<u64, CanonicalNativeResearchStartErrorV1> {
    let mut handle =
        start_canonical_native_research_lane_v1(startup_settings, runtime_install_receipt, intent)?;
    let lease_token = handle.snapshot_receiver_mut().borrow().lease_token();
    observe_canonical_native_research_v1(state, handle).await;
    Ok(lease_token)
}

#[allow(
    dead_code,
    reason = "Chunk4C entrypoint adapters consume this crate-private native boundary"
)]
pub(crate) async fn observe_canonical_native_research_v1(
    state: AppApiState,
    mut handle: CanonicalNativeResearchJobHandleV1,
) {
    let initial = handle.snapshot_receiver_mut().borrow().clone();
    let lease_token = initial.lease_token();
    state
        .install_canonical_native_research_v1(handle.cancellation_token().clone(), initial)
        .await;
    tokio::spawn(async move {
        loop {
            let changed = handle.snapshot_receiver_mut().changed().await;
            if changed.is_err() {
                break;
            }
            let snapshot = handle.snapshot_receiver_mut().borrow().clone();
            let terminal = snapshot.state().is_terminal();
            reduce_canonical_native_research_event_v1(
                &state,
                ServiceEvent::CanonicalNativeResearchUpdated(CanonicalNativeResearchEventV1::new(
                    snapshot,
                )),
            )
            .await;
            if terminal {
                break;
            }
        }
        let terminal = handle.await_terminal().await;
        state
            .update_canonical_native_research_v1(CanonicalNativeResearchSnapshotV1::from_terminal(
                lease_token,
                terminal.clone(),
            ))
            .await;
        tracing::info!(
            target: "neoethos_app::server::engines_control",
            terminal_state = terminal.state().as_str(),
            "canonical native research worker reached terminal and released its lease"
        );
    });
}

#[allow(
    dead_code,
    reason = "Chunk4C entrypoint adapters consume this crate-private native boundary"
)]
async fn reduce_canonical_native_research_event_v1(state: &AppApiState, event: ServiceEvent) {
    if let ServiceEvent::CanonicalNativeResearchUpdated(event) = event {
        state
            .update_canonical_native_research_v1(event.snapshot().clone())
            .await;
    }
}

// ─── shared helpers ───────────────────────────────────────────────────────

fn typed_legacy_start_error_response_v1(
    kind: &'static str,
    error: TypedLegacyExecutionStartErrorV1,
) -> Response {
    match error {
        TypedLegacyExecutionStartErrorV1::Busy(error) => (
            StatusCode::CONFLICT,
            Json(serde_json::json!({ "error": error.to_string() })),
        )
            .into_response(),
        TypedLegacyExecutionStartErrorV1::RuntimeUnavailable(detail) => actionable_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("{kind} runtime is unavailable. Restart the application and try again."),
            &anyhow::anyhow!(detail),
        ),
    }
}

fn typed_legacy_admission_error_response_v1(
    kind: &'static str,
    error: &TypedLegacyExecutionAdmissionErrorV1,
) -> Response {
    let status = match error {
        TypedLegacyExecutionAdmissionErrorV1::BadRequest(_) => StatusCode::BAD_REQUEST,
        TypedLegacyExecutionAdmissionErrorV1::Conflict(_)
        | TypedLegacyExecutionAdmissionErrorV1::Cancelled(_) => StatusCode::CONFLICT,
        TypedLegacyExecutionAdmissionErrorV1::UnprocessableEntity(_) => {
            StatusCode::UNPROCESSABLE_ENTITY
        }
        TypedLegacyExecutionAdmissionErrorV1::ServiceUnavailable(_) => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        TypedLegacyExecutionAdmissionErrorV1::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
    };
    actionable_error(
        status,
        format!(
            "{kind} could not be admitted. Check Settings and the selected inputs, then try again."
        ),
        &anyhow::anyhow!(error.detail().to_owned()),
    )
}

#[cfg(test)]
mod exact_dataset_selection_tests {
    use super::*;
    use neoethos_data::{BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe};

    fn selected_identity() -> CanonicalDatasetIdentity {
        CanonicalDatasetIdentity::external(
            "operator-upload",
            "EURUSD",
            CanonicalTimeframe::M5,
            BarTimestampConvention::BarOpen,
        )
        .expect("valid exact dataset identity")
    }

    fn selected_generation() -> SelectedDatasetGenerationV1 {
        SelectedDatasetGenerationV1::new(
            selected_identity(),
            format!("g1-{}.vortex", "1".repeat(64)),
            "2".repeat(64),
        )
        .expect("valid selected generation")
    }

    fn body(selection: Option<SelectedDatasetGenerationV1>) -> StartJobBody {
        StartJobBody {
            dataset_selection: selection,
            symbol: Some("EURUSD".to_owned()),
            base_tf: Some("M5".to_owned()),
            ..StartJobBody::default()
        }
    }

    #[test]
    fn discovery_requires_an_exact_dataset_generation() {
        let error = resolve_discovery_selection(&body(None))
            .expect_err("symbol and timeframe alone must never select discovery data");

        assert!(error.to_string().contains("dataset_selection"));
    }

    #[test]
    fn discovery_derives_symbol_and_timeframe_from_the_exact_identity() {
        let selection = selected_generation();
        let resolved = resolve_discovery_selection(&body(Some(selection.clone())))
            .expect("consistent legacy assertions must be accepted");

        assert_eq!(resolved, selection);
        assert_eq!(resolved.identity().symbol_name(), "EURUSD");
        assert_eq!(resolved.identity().timeframe().as_str(), "M5");
    }

    #[test]
    fn discovery_rejects_a_legacy_symbol_that_disagrees_with_the_identity() {
        let mut request = body(Some(selected_generation()));
        request.symbol = Some("GBPUSD".to_owned());

        let error = resolve_discovery_selection(&request)
            .expect_err("legacy symbol may assert consistency but cannot select another series");
        assert!(error.to_string().contains("GBPUSD"));
        assert!(error.to_string().contains("EURUSD"));
    }

    #[test]
    fn discovery_rejects_a_legacy_timeframe_that_disagrees_with_the_identity() {
        let mut request = body(Some(selected_generation()));
        request.base_tf = Some("H1".to_owned());

        let error = resolve_discovery_selection(&request).expect_err(
            "legacy timeframe may assert consistency but cannot select another timeframe",
        );
        assert!(error.to_string().contains("H1"));
        assert!(error.to_string().contains("M5"));
    }
}

// ─── EngineRunState (wire-friendly subset of JobState) ────────────────────

/// Compact engine state for `/engines/status`. We collapse Queued and
/// Running into the same "Running" label (the UI only cares whether
/// it should show a green dot + a "Stop" button), and Degraded into
/// Succeeded (still a terminal-OK outcome).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineRunState {
    Idle,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl EngineRunState {
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineRunState::Idle => "Idle",
            EngineRunState::Running => "Running",
            EngineRunState::Succeeded => "Succeeded",
            EngineRunState::Failed => "Failed",
            EngineRunState::Cancelled => "Cancelled",
        }
    }
}

impl From<JobState> for EngineRunState {
    fn from(value: JobState) -> Self {
        match value {
            JobState::Queued | JobState::Running => EngineRunState::Running,
            JobState::Succeeded | JobState::Degraded => EngineRunState::Succeeded,
            JobState::Failed => EngineRunState::Failed,
            JobState::Cancelled => EngineRunState::Cancelled,
        }
    }
}
