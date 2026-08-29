//! `/strategy_lab/*` — Promotion Gate status + promote-to-live (F-330).
//!
//! The Strategy Lab pipeline is Discovery → Training → Validation →
//! **Promotion Gate**. Promotion is intentionally fail-closed today: search
//! emits `neoethos.search-promotion-summary.v3`, whose selection-envelope plus
//! typed holdout payload still does
//! not prove the exact in-sample/holdout/forward/prop windows. Both endpoints
//! return 412 and copy zero files until search-core emits an exact composite
//! v3 authority and this loader verifies that new schema.
//!
//! Endpoints:
//!   - `GET  /strategy_lab/promotion?symbol=EURUSD&base_tf=M5`
//!       Evaluate the latest portfolio for that symbol/timeframe
//!       against the promotion gate and return the decision +
//!       per-criterion breakdown. Read-only.
//!   - `POST /strategy_lab/promote`  (body: {symbol, baseTf})
//!       If the gate passes, copy the trained artifacts from
//!       `models/<symbol>/<tf>/` to `live_models/<symbol>/<tf>/` so the
//!       auto-trade producer (which prefers `live_models/`) starts
//!       using them. Refuses with 412 when the gate fails.

use std::path::Path;

use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use neoethos_core::Settings;
use neoethos_core::domain::promotion_gate::{
    PromotionDecision, PromotionGateConfig, PromotionMetrics, aggregate_portfolio,
    evaluate_promotion,
};
use neoethos_data::CanonicalTimeframe;
use neoethos_search::{
    CanonicalSearchArtifactEnvelopeV2, CanonicalSearchArtifactScopeV2, CanonicalSearchWindowRoleV1,
    PROMOTION_SUMMARY_ARTIFACT_KIND_V3, PromotionSummaryAuthorityPayloadV3,
};
use serde::Deserialize;

use crate::app_services::discovery::{
    MODEL_TARGETS_SCHEMA_VERSION, ModelTargetsFile, model_targets_path_for,
    promotion_summary_path_for,
};

use super::errors::{actionable_error, internal_panic};
use super::state::AppApiState;

#[path = "promotion_authorization.rs"]
mod promotion_authorization;
use promotion_authorization::{
    CompositeAuthorityChecksV3, PromotionAuthorizationError, PromotionCopyPermit,
    REQUIRED_COMPOSITE_PROMOTION_AUTHORITY_KIND_V3, authorize_exact_composite_promotion_v3,
    copy_model_tree_if_authorized, validate_promotion_path_leafs,
};

/// Root dir for trained models (what Training writes).
const MODELS_DIR: &str = "models";
/// Root dir for promoted models (what live inference prefers).
pub const LIVE_MODELS_DIR: &str = "live_models";

#[derive(Debug, Deserialize)]
pub struct PromotionQuery {
    pub symbol: Option<String>,
    pub base_tf: Option<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromotionResponseDto {
    pub symbol: String,
    pub base_tf: String,
    pub portfolio_size: usize,
    /// Portfolio-aggregate metrics the gate evaluated (None when the
    /// portfolio is empty after an exact composite-v3 authority is available).
    pub aggregate: Option<PromotionMetrics>,
    pub decision: PromotionDecision,
    /// The thresholds in effect, echoed so the UI can render
    /// "Sharpe 1.4 ≥ 1.0 ✓" without a second round-trip.
    pub config: PromotionGateConfig,
}

// ─── GET /strategy_lab/promotion ───────────────────────────────────────────

pub async fn promotion_status(
    State(_state): State<AppApiState>,
    Query(q): Query<PromotionQuery>,
) -> Response {
    // 2026-06-04 PARITY: empty → resolved from config.yaml inside
    // evaluate_promotion_for (shared SystemConfig resolvers), not a hardcoded
    // "EURUSD"/"M5" that ignored the operator's configured symbol/base.
    let symbol = q.symbol.unwrap_or_default();
    let base_tf = q.base_tf.unwrap_or_default();

    let result =
        tokio::task::spawn_blocking(move || evaluate_promotion_for(&symbol, &base_tf)).await;
    match result {
        Ok(Ok(dto)) => Json(dto).into_response(),
        Ok(Err(err)) => promotion_error_response(
            err,
            "Could not evaluate the promotion gate. Run Discovery first to produce a \
             receipt-bound portfolio for this exact dataset, then retry.",
        ),
        Err(join_err) => internal_panic("Evaluating the promotion gate", join_err),
    }
}

fn promotion_error_response(error: anyhow::Error, unexpected_action: &str) -> Response {
    if error
        .downcast_ref::<PromotionAuthorizationError>()
        .is_some()
    {
        return actionable_error(
            StatusCode::PRECONDITION_FAILED,
            "Promotion authorization failed closed. Run a new Discovery cycle and keep its \
             exact model_targets v3 and promotion-summary authority together.",
            &error,
        );
    }
    actionable_error(StatusCode::INTERNAL_SERVER_ERROR, unexpected_action, &error)
}

/// Load the gate config the operator actually configured.
///
/// **2026-08-04 fix — the gate ran, but with thresholds nobody could
/// reach.** This function used to be
/// `fn load_gate_config(_settings: &Settings) -> PromotionGateConfig {
/// PromotionGateConfig::default() }`: it took the operator's `Settings`,
/// discarded it, and returned the hardcoded moderate defaults. Both the
/// read-only `GET /strategy_lab/promotion` and the authoritative
/// `POST /strategy_lab/promote` route through here, so no portfolio had
/// ever been judged against an operator-set bar — the endpoint even
/// echoed the hardcoded config back to the UI as "the thresholds in
/// effect", which made the wrong thresholds look confirmed.
///
/// The old doc comment said "a future `ModelsConfig.promotion_gate`
/// field can override them here without touching the endpoint". That
/// field now exists and this is the read. `models.promotion_gate` is
/// `#[serde(default)]`, so a `config.yaml` without the key still yields
/// exactly the previous thresholds — see
/// `a_settings_without_the_key_reproduces_the_previous_thresholds`.
fn load_gate_config(settings: &Settings) -> PromotionGateConfig {
    settings.models.promotion_gate.clone()
}

struct AuthorizedPromotionEvaluation {
    response: PromotionResponseDto,
    copy_permit: PromotionCopyPermit,
}

fn evaluate_promotion_for(symbol: &str, base_tf: &str) -> anyhow::Result<PromotionResponseDto> {
    Ok(evaluate_authorized_promotion_for(symbol, base_tf)?.response)
}

fn evaluate_authorized_promotion_for(
    symbol: &str,
    base_tf: &str,
) -> anyhow::Result<AuthorizedPromotionEvaluation> {
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::Promotion)
        .map_err(anyhow::Error::new)?;

    let config_path = super::state::current_config_path();
    let settings = Settings::from_yaml(&config_path)
        .map_err(|e| anyhow::anyhow!("{} not loadable: {e}", config_path.display()))?;
    let gate_config = load_gate_config(&settings);
    // 2026-06-04 PARITY: an empty symbol/base (request omitted it) resolves to
    // the configured default via the SAME shared SystemConfig resolvers the CLI
    // and the discovery/training endpoints use — never a hardcoded EURUSD/M5.
    let symbol = if symbol.trim().is_empty() {
        settings.system.resolve_symbol()
    } else {
        symbol.trim().to_uppercase()
    };
    let base_tf = if base_tf.trim().is_empty() {
        settings.system.resolve_base_timeframe()
    } else {
        base_tf.trim().to_uppercase()
    };
    let data_root = settings.system.data_dir;

    let (file, copy_permit) = authorize_model_targets_for_promotion(&data_root, &symbol, &base_tf)
        .map_err(anyhow::Error::new)?;

    let metrics: Vec<PromotionMetrics> = file
        .portfolio
        .iter()
        .map(|e| PromotionMetrics {
            sharpe: e.sharpe_ratio,
            win_rate: e.win_rate,
            profit_factor: e.profit_factor,
            max_drawdown_pct: e.max_drawdown_pct,
            trades: e.trades_count,
        })
        .collect();

    let aggregate = aggregate_portfolio(&metrics);
    let decision = match &aggregate {
        Some(agg) => evaluate_promotion(agg, &gate_config),
        None => PromotionDecision {
            promoted: false,
            criteria: Vec::new(),
            summary: "Portfolio is empty — nothing to promote.".to_string(),
        },
    };

    Ok(AuthorizedPromotionEvaluation {
        response: PromotionResponseDto {
            symbol: symbol.to_string(),
            base_tf: base_tf.to_string(),
            portfolio_size: file.portfolio.len(),
            aggregate,
            decision,
            config: gate_config,
        },
        copy_permit,
    })
}

fn authorize_model_targets_for_promotion(
    data_root: &Path,
    symbol: &str,
    base_tf: &str,
) -> Result<(ModelTargetsFile, PromotionCopyPermit), PromotionAuthorizationError> {
    let validated_path = validate_promotion_path_leafs(symbol, base_tf)?;
    let canonical_timeframe = base_tf.parse::<CanonicalTimeframe>().map_err(|_| {
        PromotionAuthorizationError::UnsafePathLeaf {
            field: "base timeframe",
            value: base_tf.to_owned(),
        }
    })?;
    if canonical_timeframe.as_str() != base_tf {
        return Err(PromotionAuthorizationError::UnsafePathLeaf {
            field: "base timeframe",
            value: base_tf.to_owned(),
        });
    }
    let path = model_targets_path_for(data_root, symbol, base_tf);
    let bytes = std::fs::read(&path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PromotionAuthorizationError::MissingModelTargets { path: path.clone() }
        } else {
            PromotionAuthorizationError::MalformedModelTargets {
                reason: format!("read {}: {error}", path.display()),
            }
        }
    })?;
    let wire: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        PromotionAuthorizationError::MalformedModelTargets {
            reason: format!("parse {}: {error}", path.display()),
        }
    })?;
    let found_schema_version = wire
        .get("schema_version")
        .and_then(serde_json::Value::as_u64);
    let schema_version = found_schema_version
        .and_then(|value| u32::try_from(value).ok())
        .unwrap_or_default();
    if schema_version != MODEL_TARGETS_SCHEMA_VERSION {
        return Err(PromotionAuthorizationError::UnsupportedSchema {
            found: found_schema_version,
        });
    }
    let file: ModelTargetsFile = serde_json::from_value(wire).map_err(|error| {
        PromotionAuthorizationError::MalformedModelTargets {
            reason: format!("decode strict v3 schema at {}: {error}", path.display()),
        }
    })?;

    let anchor = file.search_input_receipt.validate().map_err(|error| {
        PromotionAuthorizationError::MalformedModelTargets {
            reason: format!("invalid embedded search receipt: {error}"),
        }
    })?;
    let recomputed_receipt_sha256 =
        file.search_input_receipt
            .identity_sha256()
            .map_err(|error| PromotionAuthorizationError::MalformedModelTargets {
                reason: format!("cannot identify embedded search receipt: {error}"),
            })?;
    if file.search_input_receipt_sha256 != recomputed_receipt_sha256 {
        return Err(PromotionAuthorizationError::ReceiptDigestMismatch {
            expected: recomputed_receipt_sha256,
            found: file.search_input_receipt_sha256.clone(),
        });
    }
    if file.symbol != symbol
        || file.base_tf != base_tf
        || anchor.symbol_name() != symbol
        || anchor.timeframe().as_str() != base_tf
    {
        return Err(PromotionAuthorizationError::RequestedIdentityMismatch {
            reason: format!(
                "requested {symbol}/{base_tf}, targets name {}/{}, receipt anchor {}/{}",
                file.symbol,
                file.base_tf,
                anchor.symbol_name(),
                anchor.timeframe()
            ),
        });
    }

    let summary_path = promotion_summary_path_for(data_root, symbol, base_tf);
    let expected_file_name = summary_path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .ok_or_else(|| PromotionAuthorizationError::InvalidPromotionSummary {
            reason: "canonical promotion-summary filename is not UTF-8".to_owned(),
        })?;
    if file.promotion_summary_authority.canonical_file_name != expected_file_name {
        return Err(PromotionAuthorizationError::RequestedIdentityMismatch {
            reason: format!(
                "promotion-summary binding names `{}` instead of canonical `{expected_file_name}`",
                file.promotion_summary_authority.canonical_file_name
            ),
        });
    }
    let summary_bytes = std::fs::read(&summary_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PromotionAuthorizationError::MissingPromotionSummary {
                path: summary_path.clone(),
            }
        } else {
            PromotionAuthorizationError::InvalidPromotionSummary {
                reason: format!("read {}: {error}", summary_path.display()),
            }
        }
    })?;
    let mut embedded_bytes = serde_json::to_vec_pretty(&file.promotion_summary_authority.envelope)
        .map_err(
            |error| PromotionAuthorizationError::InvalidPromotionSummary {
                reason: format!("serialize embedded promotion authority: {error}"),
            },
        )?;
    embedded_bytes.push(b'\n');
    if summary_bytes != embedded_bytes {
        return Err(PromotionAuthorizationError::PromotionSummaryMismatch);
    }
    let actual_authority =
        CanonicalSearchArtifactEnvelopeV2::<PromotionSummaryAuthorityPayloadV3>::from_json_bytes(
            &summary_bytes,
        )
        .map_err(
            |error| PromotionAuthorizationError::InvalidPromotionSummary {
                reason: error.to_string(),
            },
        )?;
    if actual_authority != file.promotion_summary_authority.envelope {
        return Err(PromotionAuthorizationError::PromotionSummaryMismatch);
    }

    let expected_scope = CanonicalSearchArtifactScopeV2::for_entire_receipt(
        CanonicalSearchWindowRoleV1::DiscoveryInput,
        file.search_input_receipt.clone(),
    )
    .map_err(
        |error| PromotionAuthorizationError::InvalidPromotionSummary {
            reason: format!("derive expected promotion scope: {error}"),
        },
    )?;
    actual_authority
        .validate_against(
            PROMOTION_SUMMARY_ARTIFACT_KIND_V3,
            &file.search_config_hash,
            &file.search_input_receipt,
            expected_scope.evaluated_window(),
        )
        .map_err(
            |error| PromotionAuthorizationError::InvalidPromotionSummary {
                reason: error.to_string(),
            },
        )?;
    if actual_authority.scope().receipt_sha256() != file.search_input_receipt_sha256 {
        return Err(PromotionAuthorizationError::ReceiptDigestMismatch {
            expected: file.search_input_receipt_sha256.clone(),
            found: actual_authority.scope().receipt_sha256().to_owned(),
        });
    }

    // The current search-core writer emits a V2 whole-DiscoveryInput scope.
    // Its OOS booleans and per-kind hashes are diagnostic only: they cannot
    // prove the exact 80/20 in-sample + held-out/forward/prop composite. Keep
    // the v3 target/sidecar equality diagnostics above, but deliberately mint
    // no permit until search-core ships the exact composite v3 authority.
    let copy_permit = authorize_exact_composite_promotion_v3(
        validated_path,
        actual_authority.artifact_kind(),
        CompositeAuthorityChecksV3 {
            exact_receipt_config_sidecar: true,
            exact_composite_scope: false,
            required_evidence_complete: false,
            required_evidence_passed: false,
        },
    )?;
    debug_assert_eq!(
        actual_authority.artifact_kind(),
        REQUIRED_COMPOSITE_PROMOTION_AUTHORITY_KIND_V3,
        "only search-core composite v3 may mint a promotion copy permit"
    );
    Ok((file, copy_permit))
}

// ─── POST /strategy_lab/promote ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct PromoteBody {
    pub symbol: Option<String>,
    pub base_tf: Option<String>,
}

#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromoteResponseDto {
    pub promoted: bool,
    pub symbol: String,
    pub base_tf: String,
    /// Where the artifacts were copied to when promoted.
    pub live_models_path: Option<String>,
    pub files_copied: usize,
    pub message: String,
}

pub async fn promote(State(_state): State<AppApiState>, Json(body): Json<PromoteBody>) -> Response {
    // 2026-06-04 PARITY: empty → resolved from config.yaml inside
    // evaluate_promotion_for, matching the discovery/training defaults.
    let symbol = body.symbol.unwrap_or_default();
    let base_tf = body.base_tf.unwrap_or_default();

    let result = tokio::task::spawn_blocking(move || promote_if_gated(&symbol, &base_tf)).await;
    match result {
        Ok(Ok(dto)) if dto.promoted => Json(dto).into_response(),
        Ok(Ok(dto)) => {
            // Gate failed — 412 Precondition Failed is the honest code:
            // the request was well-formed but the resource state
            // (portfolio quality) didn't meet the precondition.
            (StatusCode::PRECONDITION_FAILED, Json(dto)).into_response()
        }
        Ok(Err(err)) => promotion_error_response(
            err,
            "Promotion failed after authorization. Make sure the models folder has valid \
             artifacts for this exact symbol/timeframe.",
        ),
        Err(join_err) => internal_panic("Promoting the strategy", join_err),
    }
}

fn promote_if_gated(symbol: &str, base_tf: &str) -> anyhow::Result<PromoteResponseDto> {
    // Re-evaluate the gate server-side — never trust a client claim
    // that the portfolio passed. This is the authoritative check.
    let evaluation = evaluate_authorized_promotion_for(symbol, base_tf)?;
    let status = evaluation.response;
    // 2026-07-19 deep-audit fix: use the RESOLVED symbol/base_tf from the
    // gate evaluation everywhere below. The raw args may be EMPTY (the body
    // omitted them and the gate resolved config defaults internally) — the
    // old code then built `models/""/""` as the copy source, which is the
    // models ROOT: a passing gate would have copied the ENTIRE model store
    // flat into live_models/, a layout live inference cannot read.
    let symbol = status.symbol.clone();
    let base_tf = status.base_tf.clone();
    if !status.decision.promoted {
        return Ok(PromoteResponseDto {
            promoted: false,
            symbol,
            base_tf,
            live_models_path: None,
            files_copied: 0,
            message: format!("Promotion blocked: {}", status.decision.summary),
        });
    }

    // The model/live destination leaves are constructed only inside the
    // model-free copy boundary, after its opaque composite-v3 permit exists.
    let copied = copy_model_tree_if_authorized(
        Ok(evaluation.copy_permit),
        Path::new(MODELS_DIR),
        Path::new(LIVE_MODELS_DIR),
    )
    .map_err(anyhow::Error::new)?;
    let files_copied = copied.files_copied;
    let dst = copied.destination;

    tracing::info!(
        target: "neoethos_app::strategy_lab::promote",
        %symbol, %base_tf,
        files = files_copied,
        dst = %dst.display(),
        "promoted portfolio to live_models"
    );

    let message = format!("Promoted {symbol} {base_tf} to live trading ({files_copied} files).");
    Ok(PromoteResponseDto {
        promoted: true,
        symbol,
        base_tf,
        live_models_path: Some(dst.display().to_string()),
        files_copied,
        message,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression for the 2026-08-04 "no final recipient" fix. Before it,
    /// `load_gate_config` was
    /// `fn load_gate_config(_settings: &Settings) -> PromotionGateConfig {
    /// PromotionGateConfig::default() }` and this test could not fail no
    /// matter what the operator configured.
    #[test]
    fn load_gate_config_honours_the_operator_threshold() {
        let mut settings = Settings::default();
        settings.models.promotion_gate.min_sharpe = 2.5;
        settings.models.promotion_gate.min_trades = 500;
        settings.models.promotion_gate.max_drawdown_pct = 8.0;

        let cfg = load_gate_config(&settings);
        assert_eq!(
            cfg.min_sharpe, 2.5,
            "the gate must read the operator's file"
        );
        assert_eq!(cfg.min_trades, 500);
        assert_eq!(cfg.max_drawdown_pct, 8.0);
    }

    /// A portfolio that clears the shipped default bar must be REJECTED
    /// once the operator tightens it — the decision, not just the echoed
    /// config, has to change. `POST /strategy_lab/promote` re-runs this
    /// same path server-side, so this is the copy-to-`live_models/` guard.
    #[test]
    fn a_tightened_gate_changes_the_promotion_decision() {
        let portfolio = PromotionMetrics {
            sharpe: 1.8,
            win_rate: 0.56,
            profit_factor: 1.6,
            max_drawdown_pct: 12.0,
            trades: 240,
        };

        let shipped = load_gate_config(&Settings::default());
        assert!(
            evaluate_promotion(&portfolio, &shipped).promoted,
            "this portfolio clears the shipped default bar"
        );

        let mut settings = Settings::default();
        settings.models.promotion_gate.min_sharpe = 2.5;
        let decision = evaluate_promotion(&portfolio, &load_gate_config(&settings));
        assert!(
            !decision.promoted,
            "an operator min_sharpe of 2.5 must block a 1.8-Sharpe portfolio: {}",
            decision.summary
        );
    }

    /// A `config.yaml` written before the knob existed must promote
    /// exactly what it promoted yesterday.
    #[test]
    fn a_settings_without_the_key_reproduces_the_previous_thresholds() {
        assert_eq!(
            load_gate_config(&Settings::default()),
            PromotionGateConfig::default(),
            "adding the knob must not move the bar for existing installs"
        );
    }
}
