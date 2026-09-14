//! Candidate-owned strategy/model bar research on the explicitly reserved final
//! scope. A local first-use journal records reuse; neither reservation nor the
//! report is broker/live admission or proof of no historical exposure.

use crate::app_services::jobs::CancellationFlag;
use anyhow::{Context, Result, ensure};
use neoethos_core::Settings;
use neoethos_core::execution::BudgetedCpuExecutor;
use neoethos_core::execution_budget::{CpuLease, CpuPermitRequest, WorkerLimit};
use neoethos_models::ensemble_inference::SoftVotingEnsemble;
use neoethos_models::{PromotionCandidateTrainingHandoffV1, PromotionCandidateTrainingManifestV1};
use neoethos_search::data_selection::CanonicalSearchInput;
use neoethos_search::eval::{NettedBarDecisionTapeV1, PreparedNettedCanonicalBarResearchV1};
use neoethos_trader::{BlendConfig, BlendMode, Direction, MlDecision, blend_decision};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub(super) const REPORT_SUFFIX: &str = "combined-bar-research.json";

/// The caller has completed training and released its CPU lease. Each feature
/// phase borrows the entire installed reservation in its exact private pool;
/// the two large cubes never coexist. Inference then splits that reservation
/// across independent causal windows, never across one account's ledger.
pub(super) fn evaluate_candidate(
    settings: &Settings,
    candidate_root: &Path,
    manifest: &PromotionCandidateTrainingManifestV1,
    ensemble: &SoftVotingEnsemble,
    cancel_flag: &CancellationFlag,
    progress: &(impl Fn(&str, usize, usize) + Sync),
) -> Result<PathBuf> {
    let budget_cancellation = cancel_flag.cpu_budget_token();
    let cancel_arc = cancel_flag.cancel_arc();
    let cancel = &cancel_arc;
    check_cancel(cancel)?;
    let handoff = manifest.reopen_handoff(candidate_root)?;
    handoff.validate_against_settings_v1(settings)?;
    let (portfolio, identity) = handoff.validated_live_portfolio_and_identity_sha256()?;
    let evaluation = portfolio.live_trading_policy.sealed_evaluation_config()?;
    handoff
        .screening_contract()
        .validate_evaluation_costs(&evaluation)?;
    ensure!(
        !portfolio.genes.is_empty(),
        "final research requires a nonempty locked portfolio"
    );
    let scope = &portfolio.final_holdout_scope;
    let input_contract = ensemble
        .model_feature_input()
        .context("candidate ensemble has no model-owned preprocessing contract")?;
    input_contract.validate_for_handoff(&handoff)?;
    let input_bytes = input_contract.to_json_bytes()?;
    let model_input_sha256 = {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(&input_bytes))
    };
    // Resolve and freeze the ACTUAL inference/blend policy before any final
    // dataset, feature or prediction reads. Ambient changes on a retry produce
    // another locked identity, but never another first use of this raw window.
    let blend = BlendConfig::from_config_values(
        BlendMode::MlScale,
        Some(settings.models.blend_gate_floor),
        Some(settings.models.blend_veto_below),
    );
    check_cancel(cancel)?;
    let final_use = super::final_holdout::begin(
        &settings.system.data_dir,
        scope,
        serde_json::json!({
            "protocol": "neoethos.candidate-combined-bar-research.v2",
            "training_handoff": identity,
            "locked_portfolio_identity_sha256": handoff.locked_portfolio().identity_sha256(),
            "candidate_tree_sha256": manifest.candidate_tree_sha256(),
            "model_input_sha256": model_input_sha256,
            "model_inference_settings": settings.models,
            "blend_mode": "ml_scale",
            "blend_gate_floor": blend.gate_floor,
            "blend_veto_below": blend.veto_below,
            "model_history_rows": 256,
            "account_policy": "archived_search_risk_fractional_lots_v1",
        }),
    )?;
    let installed = neoethos_core::execution_budget::installed_process_budget()
        .context("combined screening requires the installed CPU budget")?;
    let width = installed.resolved().effective_worker_limit;
    let executor = BudgetedCpuExecutor::new_for_broker(installed.broker().clone(), width);
    let feature_control = neoethos_data::FeatureBuildControl::new(Arc::clone(cancel));
    check_cancel(cancel)?;
    let PreparedStrategyInput {
        prepared,
        directions,
        confidences,
        stops,
        targets,
    } = prepare_strategy_input(
        settings,
        &handoff,
        &portfolio,
        &evaluation,
        cancel_flag,
        progress,
    )?;
    check_cancel(cancel)?;
    progress("model_features", 0, 1);
    let lease = installed
        .broker()
        .acquire_cancellable(CpuPermitRequest::local(width), budget_cancellation)?;
    let model_frame = executor.execute(lease.into_transfer(), || -> Result<_> {
        let dataset = neoethos_data::load_exact_dataset_series_receipt(
            &settings.system.data_dir,
            handoff.canonical_series(),
        )?;
        check_cancel(cancel)?;
        let frame =
            Arc::new(ensemble.prepare_model_features_with_control(&dataset, &feature_control)?);
        check_cancel(cancel)?;
        Ok(frame)
    })??;
    progress("model_features", 1, 1);
    let model_rows = exact_rows(&model_frame.timestamps, prepared.timestamps())?;
    let bound = ensemble.bind_model_features(&model_frame)?;
    let lease = installed
        .broker()
        .acquire_cancellable(CpuPermitRequest::local(width), budget_cancellation)?;
    progress("causal_model_inference", 0, model_rows.len());
    let decisions = parallel_rows(
        model_rows.len(),
        lease,
        cancel,
        |row, worker| {
            if directions[row] == Direction::Flat {
                return Ok((0.0, false));
            }
            let decision = bound.last_row(model_rows[row] + 1, 256, worker)?;
            if !decision.validity.is_valid() {
                return Ok((0.0, true));
            }
            let (_, multiplier) = blend_decision(
                directions[row],
                &MlDecision {
                    dir_probs: decision.dir_probs,
                    regime_gate: decision.regime_gate,
                    anomaly_scale: decision.anomaly_scale,
                },
                &blend,
            );
            // A model veto prevents a NEW entry; it must not replace the genes'
            // signal used to manage an already-open position.
            Ok((multiplier, false))
        },
        |done| progress("causal_model_inference", done, model_rows.len()),
    )?;
    drop(bound);
    drop(model_frame);
    check_cancel(cancel)?;
    let multipliers = decisions.iter().map(|value| value.0).collect::<Vec<_>>();
    let invalid_model_rows = decisions.iter().filter(|value| value.1).count();
    let signals = directions
        .iter()
        .map(|direction| match direction {
            Direction::Long => 1,
            Direction::Short => -1,
            Direction::Flat => 0,
        })
        .collect::<Vec<i8>>();
    let gene_only_multipliers = vec![1.0; signals.len()];
    progress("account_replay", 0, 2);
    // The two accounts are independent and may run concurrently; each account's
    // ledger remains chronological and owns one process-budget reservation.
    let account_width = WorkerLimit::new(width.get().min(2))?;
    let lease = installed
        .broker()
        .acquire_cancellable(CpuPermitRequest::local(account_width), budget_cancellation)?;
    let mut accounts = parallel_rows(
        2,
        lease,
        cancel,
        |account, _worker| {
            prepared.evaluate(
                NettedBarDecisionTapeV1 {
                    signals: &signals,
                    confidences: &confidences,
                    sl_pips: &stops,
                    tp_pips: &targets,
                    ml_multipliers: if account == 0 {
                        &gene_only_multipliers
                    } else {
                        &multipliers
                    },
                },
                None,
            )
        },
        |done| progress("account_replay", done, 2),
    )?
    .into_iter();
    let gene_only = accounts.next().context("gene-only account result absent")?;
    let combined = accounts.next().context("combined account result absent")?;
    check_cancel(cancel)?;
    // Beside, never inside, the content-addressed model tree: adding a report
    // must not mutate the manifest that was just verified by the real loader.
    let path = final_use.report_path(candidate_root, REPORT_SUFFIX);
    let report = serde_json::json!({
        "schema": "neoethos.candidate-combined-bar-research.v2",
        "training_handoff": identity,
        "locked_portfolio_identity_sha256": handoff.locked_portfolio().identity_sha256(),
        "candidate_tree_sha256": manifest.candidate_tree_sha256(),
        "model_input_sha256": model_input_sha256,
        "symbol": portfolio.symbol,
        "base_timeframe": portfolio.base_tf,
        "rows": prepared.timestamps().len(),
        "timestamp_start_ms": prepared.timestamps().first(),
        "timestamp_end_ms": prepared.timestamps().last(),
        "model_history_rows": 256,
        "inference_workers": width.get(),
        "invalid_model_signal_rows": invalid_model_rows,
        "blend_mode": "ml_scale",
        "blend_gate_floor": blend.gate_floor,
        "blend_veto_below": blend.veto_below,
        "configured_live_ml_gate": settings.models.live_ml_gate,
        "holdout_use": if final_use.first_recorded_use { "first_recorded_local_use_of_reserved_final_scope" } else { "reused_reserved_final_scope_research_only" },
        "historical_exposure": "unknown_before_this_local_journal_not_never_ever_seen_evidence",
        "raw_final_scope_sha256": final_use.raw_scope_sha256,
        "locked_final_inputs_sha256": final_use.locked_inputs_sha256,
        "first_locked_final_inputs_sha256": final_use.first_locked_inputs_sha256,
        "final_scope_window": scope.evaluated_window(),
        "training_cutoff_ms": handoff.oos_cutoff_ms(),
        "sizing_basis": "archived_search_confidence_risk_band_not_live_Risky_or_PropFirm_account_simulation",
        "exit_basis": "canonical_bar_brackets_trailing_time_and_session_policy_not_live_reversal_or_supervisor_actions",
        "volume_basis": "fractional_research_lots_broker_grid_not_attested",
        "promotion_eligible": false,
        "gene_only": gene_only,
        "combined": combined,
    });
    final_use.complete(&path, &report)?;
    Ok(path)
}

struct PreparedStrategyInput {
    prepared: PreparedNettedCanonicalBarResearchV1,
    directions: Vec<Direction>,
    confidences: Vec<f64>,
    stops: Vec<f64>,
    targets: Vec<f64>,
}

/// Both modes reconstruct the same saved Search recipe and archived gene policy.
/// The caller must reserve final-window use before entering this data-reading phase.
fn prepare_strategy_input(
    settings: &Settings,
    handoff: &PromotionCandidateTrainingHandoffV1,
    portfolio: &neoethos_search::live_portfolio::LivePortfolioArtifact,
    evaluation: &neoethos_search::EvaluationConfig,
    cancel: &CancellationFlag,
    progress: &(impl Fn(&str, usize, usize) + Sync),
) -> Result<PreparedStrategyInput> {
    let cancellation = cancel.cancel_arc();
    check_cancel(&cancellation)?;
    let installed = neoethos_core::execution_budget::installed_process_budget()
        .context("final research requires the installed CPU budget")?;
    let width = installed.resolved().effective_worker_limit;
    let executor = BudgetedCpuExecutor::new_for_broker(installed.broker().clone(), width);
    let control = neoethos_data::FeatureBuildControl::new(Arc::clone(&cancellation));
    progress("search_features", 0, 1);
    let lease = installed
        .broker()
        .acquire_cancellable(CpuPermitRequest::local(width), cancel.cpu_budget_token())?;
    let result = executor.execute(lease.into_transfer(), || -> Result<_> {
        let options = handoff
            .search_input_receipt()
            .feature_build_options()
            .context("final research needs the saved Search recipe")?;
        let input = CanonicalSearchInput::from_recorded_receipt_with_control(
            &settings.system.data_dir,
            handoff.search_input_receipt().clone(),
            options,
            &control,
        )?;
        check_cancel(&cancellation)?;
        let run_input = input.as_run_input_with_control(&control)?;
        let prepared = PreparedNettedCanonicalBarResearchV1::from_input(
            &run_input,
            &portfolio.final_holdout_scope,
            handoff.screening_contract(),
            evaluation,
        )?;
        let aligned = portfolio.project_live_features(input.features())?;
        let netted = neoethos_trader::combine_gene_signals_with_archived_policy(
            &portfolio.genes,
            &aligned,
            input.base_frame().ohlcv(),
            &portfolio.live_trading_policy,
        )?;
        let selected = exact_rows(&aligned.timestamps, prepared.timestamps())?;
        let take = |values: &[f64]| selected.iter().map(|row| values[*row]).collect::<Vec<_>>();
        Ok(PreparedStrategyInput {
            prepared,
            directions: selected.iter().map(|row| netted.directions[*row]).collect(),
            confidences: take(&netted.confidences),
            stops: take(&netted.sl_pips),
            targets: take(&netted.tp_pips),
        })
    })??;
    // The full Search cube is released before model preprocessing in combined mode.
    progress("search_features", 1, 1);
    Ok(result)
}

/// Explicit final-window strategy research. No model planning, training, loading
/// or inference is required; all financial assumptions belong to the locked portfolio.
pub(super) fn evaluate_strategies(
    settings: &Settings,
    candidate_root: &Path,
    handoff: &PromotionCandidateTrainingHandoffV1,
    cancel: &CancellationFlag,
    progress: &(impl Fn(&str, usize, usize) + Sync),
) -> Result<PathBuf> {
    let cancellation = cancel.cancel_arc();
    check_cancel(&cancellation)?;
    let (portfolio, identity) = handoff.validated_live_portfolio_and_identity_sha256()?;
    let evaluation = portfolio.live_trading_policy.sealed_evaluation_config()?;
    handoff
        .screening_contract()
        .validate_evaluation_costs(&evaluation)?;
    ensure!(
        !portfolio.genes.is_empty(),
        "final research requires a nonempty locked portfolio"
    );
    let scope = &portfolio.final_holdout_scope;
    check_cancel(&cancellation)?;
    let final_use = super::final_holdout::begin(
        &settings.system.data_dir,
        scope,
        serde_json::json!({
            "protocol": "neoethos.candidate-strategy-bar-research.v1",
            "training_handoff": identity,
            "locked_portfolio_identity_sha256": handoff.locked_portfolio().identity_sha256(),
            "account_policy": "archived_search_risk_fractional_lots_v1",
        }),
    )?;
    let PreparedStrategyInput {
        prepared,
        directions,
        confidences,
        stops,
        targets,
    } = prepare_strategy_input(settings, handoff, &portfolio, &evaluation, cancel, progress)?;
    check_cancel(&cancellation)?;
    let signals = directions
        .iter()
        .map(|direction| match direction {
            Direction::Long => 1,
            Direction::Short => -1,
            Direction::Flat => 0,
        })
        .collect::<Vec<i8>>();
    let multipliers = vec![1.0; signals.len()];
    let installed = neoethos_core::execution_budget::installed_process_budget()
        .context("final research requires the installed CPU budget")?;
    let width = WorkerLimit::new(1)?;
    let executor = BudgetedCpuExecutor::new_for_broker(installed.broker().clone(), width);
    let lease = installed
        .broker()
        .acquire_cancellable(CpuPermitRequest::local(width), cancel.cpu_budget_token())?;
    progress("account_replay", 0, 1);
    // One chronological account ledger; the preceding feature phase uses the
    // installed full-width CPU reservation. No model queue is created here.
    let gene_only = executor.execute(lease.into_transfer(), || {
        prepared.evaluate(
            NettedBarDecisionTapeV1 {
                signals: &signals,
                confidences: &confidences,
                sl_pips: &stops,
                tp_pips: &targets,
                ml_multipliers: &multipliers,
            },
            None,
        )
    })??;
    progress("account_replay", 1, 1);
    check_cancel(&cancellation)?;
    std::fs::create_dir_all(candidate_root)?;
    let path = final_use.report_path(candidate_root, "strategy-bar-research.json");
    let report = serde_json::json!({
        "schema": "neoethos.candidate-strategy-bar-research.v1",
        "training_handoff": identity,
        "locked_portfolio_identity_sha256": handoff.locked_portfolio().identity_sha256(),
        "symbol": portfolio.symbol,
        "base_timeframe": portfolio.base_tf,
        "rows": prepared.timestamps().len(),
        "timestamp_start_ms": prepared.timestamps().first(),
        "timestamp_end_ms": prepared.timestamps().last(),
        "holdout_use": if final_use.first_recorded_use { "first_recorded_local_use_of_reserved_final_scope" } else { "reused_reserved_final_scope_research_only" },
        "historical_exposure": "unknown_before_this_local_journal_not_never_ever_seen_evidence",
        "raw_final_scope_sha256": final_use.raw_scope_sha256,
        "locked_final_inputs_sha256": final_use.locked_inputs_sha256,
        "first_locked_final_inputs_sha256": final_use.first_locked_inputs_sha256,
        "final_scope_window": scope.evaluated_window(),
        "training_cutoff_ms": handoff.oos_cutoff_ms(),
        "sizing_basis": "archived_search_confidence_risk_band_not_live_Risky_or_PropFirm_account_simulation",
        "exit_basis": "canonical_bar_brackets_trailing_time_and_session_policy_not_live_reversal_or_supervisor_actions",
        "volume_basis": "fractional_research_lots_broker_grid_not_attested",
        "promotion_eligible": false,
        "gene_only": gene_only,
    });
    final_use.complete(&path, &report)?;
    Ok(path)
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    ensure!(
        !cancel.load(Ordering::Acquire),
        "operator cancelled combined candidate screening"
    );
    Ok(())
}

fn exact_rows(available: &[i64], requested: &[i64]) -> Result<Vec<usize>> {
    ensure!(
        available.windows(2).all(|pair| pair[0] < pair[1])
            && requested.windows(2).all(|pair| pair[0] < pair[1]),
        "combined inputs must have strictly increasing exact timestamps"
    );
    requested.iter().map(|timestamp| available.binary_search(timestamp)
        .map_err(|_| anyhow::anyhow!("combined model/Search timestamp {timestamp} is absent; no tail offset or synthesized row")))
        .collect()
}

/// Dynamic queue over a fixed number of owned one-worker leases. Each result
/// retains its original row index; scheduling cannot change temporal order.
fn parallel_rows<T: Send>(
    rows: usize,
    mut lease: CpuLease,
    cancel: &AtomicBool,
    compute: impl Fn(usize, &CpuLease) -> Result<T> + Sync,
    progress: impl Fn(usize) + Sync,
) -> Result<Vec<T>> {
    check_cancel(cancel)?;
    let next = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let one = WorkerLimit::new(1)?;
    let mut children = Vec::new();
    while lease.width().get() > 1 {
        children.push(lease.split(one)?);
    }
    let worker = |lease: &CpuLease| -> Result<Vec<(usize, T)>> {
        lease.scope(|| {
            let mut values = Vec::new();
            loop {
                check_cancel(cancel)?;
                if failed.load(Ordering::Acquire) {
                    break;
                }
                let row = next.fetch_add(1, Ordering::Relaxed);
                if row >= rows {
                    break;
                }
                let value = match compute(row, lease) {
                    Ok(value) => value,
                    Err(error) => {
                        failed.store(true, Ordering::Release);
                        return Err(error);
                    }
                };
                values.push((row, value));
                let done = completed.fetch_add(1, Ordering::AcqRel) + 1;
                if done % 256 == 0 || done == rows {
                    progress(done);
                }
            }
            Ok(values)
        })
    };
    let mut indexed = std::thread::scope(|scope| -> Result<Vec<(usize, T)>> {
        let handles = children
            .into_iter()
            .map(|child| {
                let worker = &worker;
                scope.spawn(move || worker(&child))
            })
            .collect::<Vec<_>>();
        let caller_result = worker(&lease);
        let mut parts = Vec::with_capacity(handles.len() + 1);
        parts.push(caller_result);
        for handle in handles {
            parts.push(
                handle
                    .join()
                    .map_err(|_| anyhow::anyhow!("causal inference worker panicked"))?,
            );
        }
        let mut values = Vec::with_capacity(rows);
        for part in parts {
            values.extend(part?);
        }
        Ok(values)
    })?;
    check_cancel(cancel)?;
    ensure!(
        indexed.len() == rows,
        "combined inference returned an incomplete row set"
    );
    indexed.sort_unstable_by_key(|(row, _)| *row);
    ensure!(
        indexed
            .iter()
            .enumerate()
            .all(|(expected, (actual, _))| expected == *actual),
        "combined inference returned duplicate/missing row indices"
    );
    Ok(indexed.into_iter().map(|(_, value)| value).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use neoethos_core::execution_budget::CpuPermitBroker;

    fn runtime_fixture_handoff() -> Result<PromotionCandidateTrainingHandoffV1> {
        // Match the saved-report tests: regenerated typed fixture bytes are
        // runtime inputs, not an obsolete copy embedded in the test binary.
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/combined-research-candidate-v1/training-handoff.json");
        Ok(serde_json::from_slice(&std::fs::read(path)?)?)
    }

    // The saved bundle supplies only its explicitly synthetic policy/gene and
    // calibration summary. Every data/feature/scope/handoff binding below is
    // rebuilt by the real public constructors; no fixture receipt is trusted
    // as though its imaginary embedded generations existed on disk.
    fn published_strategy_fixture(
        root: &Path,
    ) -> Result<(Settings, PromotionCandidateTrainingHandoffV1)> {
        use neoethos_data::{
            BarTimestampConvention, CanonicalDatasetIdentity, CanonicalDatasetSeriesReceiptV1,
            CanonicalOhlcvPublishRequest, CanonicalTimeframe, CanonicalVolumeRef,
            FeatureBuildOptions, FeatureCellValidity, SelectedDatasetGenerationV1,
        };
        use neoethos_search::{
            CanonicalSearchArtifactScopeV2, CanonicalSearchWindowRoleV1,
            CanonicalTrendbarResearchCostAssumptionsV2,
            CanonicalTrendbarResearchExecutionContractV3,
        };
        let original = runtime_fixture_handoff()?;
        let mut portfolio = original.locked_portfolio().deserialize_live_portfolio()?;
        original.screening_contract().validate_evaluation_costs(
            &portfolio.live_trading_policy.sealed_evaluation_config()?,
        )?;
        let data = root.join("data");
        let bars = neoethos_data::test_fixtures::ctrader_sample_ohlcv_first(100);
        ensure!(bars.len() == 100, "bounded fixture needs 100 captured bars");
        let anchor = CanonicalDatasetIdentity::external(
            "strategy-final-runtime-fixture-unverified",
            "EURUSD",
            CanonicalTimeframe::M1,
            BarTimestampConvention::BarOpen,
        )?;
        let provenance = neoethos_data::core::dataset_manifest::ProducerProvenanceEnvelopeV1::new(
            "neoethos.strategy-final-runtime-fixture.v1",
            b"synthetic strategy and costs; captured fixture bars".to_vec(),
        )?;
        eprintln!("strategy fixture: canonical publication start");
        let published =
            neoethos_data::publish_canonical_ohlcv_generation(CanonicalOhlcvPublishRequest {
                configured_root: &data,
                identity: &anchor,
                expected_generation: None,
                provenance: &provenance,
                ohlcv: &bars,
                volume: CanonicalVolumeRef::Float64(bars.volume.as_deref().unwrap()),
                rows_per_chunk: 128,
            })?;
        eprintln!("strategy fixture: canonical publication complete");
        let selected = SelectedDatasetGenerationV1::from_manifest(published.manifest())?;
        let series = CanonicalDatasetSeriesReceiptV1::new(selected.clone(), vec![selected])?;
        let installed = neoethos_core::execution_budget::installed_process_budget().unwrap();
        let width = installed.resolved().effective_worker_limit;
        let executor = BudgetedCpuExecutor::new_for_broker(installed.broker().clone(), width);
        eprintln!("strategy fixture: acquire {} CPU workers", width.get());
        let lease = installed.broker().acquire(CpuPermitRequest::local(width))?;
        let input = executor.execute(lease.into_transfer(), || -> Result<_> {
            eprintln!("strategy fixture: exact generation load start");
            let dataset = neoethos_data::load_exact_dataset_series_receipt(&data, &series)?;
            eprintln!("strategy fixture: exact generation load complete; feature producer start");
            let started = std::time::Instant::now();
            let control =
                neoethos_data::FeatureBuildControl::default().with_observer(move |event| {
                    eprintln!(
                        "strategy fixture feature +{}ms: {event:?}",
                        started.elapsed().as_millis()
                    );
                });
            let features =
                neoethos_data::prepare_multitimeframe_features_raw_with_options_and_control(
                    &dataset,
                    "M1",
                    &FeatureBuildOptions::default(),
                    &control,
                )?;
            eprintln!("strategy fixture: feature producer complete; canonical input sealing start");
            Ok(CanonicalSearchInput::from_prepared_canonical_frame(
                anchor,
                dataset.canonical_frame("M1")?,
                features,
            )?)
        })??;
        eprintln!("strategy fixture: canonical input complete; scope/portfolio sealing start");
        let run = input.as_run_input()?;
        let scope =
            |role, rows| CanonicalSearchArtifactScopeV2::from_run_input_range(role, &run, rows);
        portfolio.search_scope = scope(CanonicalSearchWindowRoleV1::InSample, 0..80)?;
        portfolio.final_holdout_scope = scope(CanonicalSearchWindowRoleV1::Holdout, 90..100)?;
        let calibration = scope(CanonicalSearchWindowRoleV1::SelectionValidation, 80..90)?;
        let column = (0..input.features().n_features())
            .find(|column| {
                (90..100).all(|row| {
                    input.features().cell(row, *column).is_ok_and(|cell| {
                        cell.validity == FeatureCellValidity::Valid && cell.value.is_finite()
                    })
                })
            })
            .context("real feature producer has no ready final-window column")?;
        portfolio.effective_feature_names = vec![input.features().names[column].clone()];
        portfolio.normalize_features = false;
        portfolio.higher_tfs.clear();
        // Deliberately fixed, directional test gene, not a searched strategy or
        // profitability claim. Small brackets ensure the real ledger closes fills.
        let gene = &mut portfolio.genes[0];
        gene.indices = vec![0];
        gene.weights = vec![0.0];
        gene.long_threshold = -1.0;
        gene.short_threshold = -2.0;
        gene.sl_pips = 1.0;
        gene.tp_pips = 1.0;
        gene.stop_vol_mult = 0.0;
        portfolio.sizing_evidence[0].forward_test =
            neoethos_search::validation::ForwardTestValidationArtifactFile::new(
                calibration,
                &portfolio.search_config_hash,
                gene,
                portfolio.sizing_evidence[0].forward_test.summary().clone(),
            )?;
        portfolio.validate()?;
        eprintln!("strategy fixture: scope/portfolio complete; research cost contract start");
        let evaluation = portfolio.live_trading_policy.sealed_evaluation_config()?;
        let contract = CanonicalTrendbarResearchExecutionContractV3::new(
            input.receipt()?,
            CanonicalTrendbarResearchCostAssumptionsV2 {
                symbol: &portfolio.symbol,
                account_currency: &evaluation.account_currency,
                assumption_source_id: "synthetic-strategy-runtime-regression-not-broker-truth",
                assumption_source_sha256: &"c".repeat(64),
                pip_size: evaluation.pip_value,
                pip_value_per_lot: evaluation.pip_value_per_lot,
                full_spread_pips_assumption: evaluation.spread_pips,
                slippage_pips_per_fill_assumption: 0.0,
                commission_account_per_lot_per_fill_assumption: evaluation.commission_per_trade
                    / 2.0,
                swap_long_pips_per_day: evaluation.swap_long_pips_per_day,
                swap_short_pips_per_day: evaluation.swap_short_pips_per_day,
                pnl_conversion_fee_rate: evaluation.pnl_conversion_fee_rate,
            },
        )?;
        let mut settings = Settings::default();
        settings.system.data_dir = data;
        settings.system.cache_dir = root.join("cache");
        settings.system.symbol = "EURUSD".to_owned();
        settings.system.base_timeframe = "M1".to_owned();
        settings.system.account_currency = "USD".to_owned();
        settings.system.higher_timeframes.clear();
        settings.system.enable_gpu_preference = "cpu".to_owned();
        settings.models.label_horizon_bars = 3;
        settings.models.ml_models = vec!["bayes_logit".to_owned()];
        settings.models.phase5_core_models.clear();
        settings.models.regime_router_enabled = false;
        settings.models.phase5_filter_meta_blender = false;
        settings.models.calibration_enabled = false;
        settings.models.use_sac_agent = false;
        settings.models.use_rl_agent = false;
        settings.models.use_neuroevolution = false;
        eprintln!("strategy fixture: research contract complete; handoff planning/sealing start");
        let handoff = PromotionCandidateTrainingHandoffV1::from_discovery_portfolio(
            series, contract, &portfolio, &settings,
        )?;
        eprintln!("strategy fixture: handoff complete");
        Ok((settings, handoff))
    }

    #[test]
    fn strategy_only_reconstructs_final_ledger_without_models_and_records_reuse() -> Result<()> {
        const CHILD: &str = "NEOETHOS_TEST_STRATEGY_FINAL_CHILD";
        if let Some(root) = std::env::var_os(CHILD) {
            use neoethos_core::execution_budget::{
                CapacityDetection, CoordinationScope, ExecutionBudgetRequest, LogicalThreadCount,
                install_process_budget,
            };
            let installed = install_process_budget(ExecutionBudgetRequest {
                host_logical_threads: None,
                detection: CapacityDetection::supplied(LogicalThreadCount::new(4)?),
                persistent_limit: None,
                legacy_persistent_limit: None,
                parent_limit: None,
                coordination_scope: CoordinationScope::ProcessLocal,
            })?;
            neoethos_data::core::hpc_ta::set_indicator_compute_policy(
                neoethos_data::core::hpc_ta::IndicatorComputePolicy::CpuOnly,
            )
            .unwrap();
            let root = PathBuf::from(root);
            let fixture_started = std::time::Instant::now();
            let (mut settings, handoff) = published_strategy_fixture(&root)?;
            eprintln!(
                "strategy fixture publish start: {:?}",
                fixture_started.elapsed()
            );
            let identity = super::super::handoff::publish(&settings.system.data_dir, &handoff)?;
            eprintln!(
                "strategy fixture publish complete: {:?}",
                fixture_started.elapsed()
            );
            let handoff = super::super::handoff::load(&settings.system.data_dir, &identity)?;
            eprintln!(
                "strategy fixture load complete: {:?}",
                fixture_started.elapsed()
            );
            // Evaluation must ignore today's model plan and need no installed tree.
            settings.models.ml_models = vec!["not-a-supported-model-runtime-sentinel".to_owned()];
            let candidates = root.join("models/candidates");
            let cancelled = CancellationFlag::new();
            cancelled.request();
            let mut no_data = settings.clone();
            no_data.system.data_dir = root.join("absent-data");
            let error =
                evaluate_strategies(&no_data, &candidates, &handoff, &cancelled, &|_, _, _| {
                    panic!("pre-cancelled evaluation must not begin a phase")
                })
                .unwrap_err();
            ensure!(format!("{error:#}").contains("operator cancelled"));
            ensure!(
                !no_data.system.data_dir.exists() && !candidates.exists(),
                "pre-cancel created data, journals or model/report directories"
            );
            eprintln!(
                "strategy fixture pre-cancel complete: {:?}",
                fixture_started.elapsed()
            );
            let mut reports = Vec::new();
            for attempt in 0..2 {
                let phases = std::sync::Mutex::new(Vec::new());
                let path = evaluate_strategies(
                    &settings,
                    &candidates,
                    &handoff,
                    &CancellationFlag::new(),
                    &|phase, done, total| {
                        eprintln!("strategy fixture attempt {attempt}: {phase} {done}/{total}");
                        phases.lock().unwrap().push((phase.to_owned(), done, total));
                        if phase == "search_features" && done == 0 {
                            assert!(settings.system.data_dir.join("final_holdout_uses").is_dir());
                        }
                    },
                )?;
                let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
                ensure!(value["schema"] == "neoethos.candidate-strategy-bar-research.v1");
                ensure!(value["rows"] == 10 && value["promotion_eligible"] == false);
                ensure!(
                    value.get("combined").is_none() && value.get("candidate_tree_sha256").is_none()
                );
                let trades = value["gene_only"]["closed_trades"]
                    .as_array()
                    .context("actual closed ledger absent")?;
                ensure!(
                    !trades.is_empty(),
                    "test did not exercise any real ledger closes"
                );
                let mut last_exit = i64::MIN;
                for trade in trades {
                    let entry = trade["entry_time"].as_i64().context("missing entry time")?;
                    let exit = trade["exit_time"].as_i64().context("missing close time")?;
                    ensure!(entry >= last_exit && exit >= entry);
                    last_exit = exit;
                    ensure!(trade["pnl"].as_f64().is_some_and(f64::is_finite));
                }
                ensure!(
                    value["holdout_use"]
                        == if attempt == 0 {
                            "first_recorded_local_use_of_reserved_final_scope"
                        } else {
                            "reused_reserved_final_scope_research_only"
                        }
                );
                assert_eq!(
                    *phases.lock().unwrap(),
                    vec![
                        ("search_features".to_owned(), 0, 1),
                        ("search_features".to_owned(), 1, 1),
                        ("account_replay".to_owned(), 0, 1),
                        ("account_replay".to_owned(), 1, 1)
                    ]
                );
                reports.push(value);
                ensure!(installed.broker().snapshot().live_reserved_sum == 0);
            }
            assert_eq!(reports[0]["gene_only"], reports[1]["gene_only"]);
            assert_eq!(
                reports[0]["raw_final_scope_sha256"],
                reports[1]["raw_final_scope_sha256"]
            );
            assert_eq!(
                reports[0]["locked_final_inputs_sha256"],
                reports[1]["locked_final_inputs_sha256"]
            );
            let reopened = super::super::read_saved_final_research_reports(
                &settings.system.data_dir,
                &candidates,
                &identity,
            )?;
            ensure!(
                reopened.training_handoff == identity && reopened.reports.len() == 2,
                "saved reader did not reopen both actual attempts: {reopened:?}"
            );
            ensure!(reopened.unavailable.is_empty());
            ensure!(reopened.reports.iter().all(|report| report.evaluation_mode
                == "strategy_only"
                && report.combined.is_none()
                && !report.promotion_eligible
                && report.gene_only.trade_count > 0));
            ensure!(
                std::fs::read_dir(&candidates)?.all(|entry| entry.is_ok_and(|entry| entry
                    .file_type()
                    .is_ok_and(|kind| kind.is_file())
                    && entry
                        .file_name()
                        .to_string_lossy()
                        .ends_with(".strategy-bar-research.json"))),
                "strategy-only evaluation created a model tree or manifest"
            );
            println!(
                "Actual synthetic strategy-only regression: 100 published bars, 10 final rows, two real ledger/report passes; no model tree or profitability claim."
            );
            // Written only after every actual evaluator/reader assertion. An
            // accidentally unmatched --exact filter must never pass the parent.
            std::fs::write(root.join("completed"), b"actual-strategy-final-pass")?;
            return Ok(());
        }
        // Process-local budgets/policies cannot be reset; isolate this runtime
        // test from every other App unit test, without mutating parent globals.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "neoethos-strategy-final-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir(&root)?;
        struct OwnedDirectory(PathBuf);
        impl Drop for OwnedDirectory {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let owned = OwnedDirectory(root.canonicalize()?);
        let stdout = owned.0.join("child.stdout");
        let stderr = owned.0.join("child.stderr");
        let mut command = std::process::Command::new(std::env::current_exe()?);
        command.args(["--exact", "app_services::training::combined::tests::strategy_only_reconstructs_final_ledger_without_models_and_records_reuse",
            "--nocapture", "--test-threads=1"]).env(CHILD, &owned.0).current_dir(&owned.0)
            .stdout(std::fs::File::create(&stdout)?).stderr(std::fs::File::create(&stderr)?);
        struct OwnedChild(std::process::Child);
        impl Drop for OwnedChild {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let mut child = OwnedChild(command.spawn()?);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        let status = loop {
            match child.0.try_wait() {
                Ok(Some(status)) => break Ok(Some(status)),
                Err(error) => break Err(error),
                Ok(None) => {}
            }
            if std::time::Instant::now() >= deadline {
                break Ok(None);
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        };
        // Kill/join on timeout and observation error BEFORE reading redirected
        // output or allowing the owned temporary directory to be removed.
        drop(child);
        println!(
            "BEGIN STRATEGY FINAL CHILD STDOUT\n{}\nEND STRATEGY FINAL CHILD STDOUT",
            std::fs::read_to_string(stdout)?
        );
        println!(
            "BEGIN STRATEGY FINAL CHILD STDERR\n{}\nEND STRATEGY FINAL CHILD STDERR",
            std::fs::read_to_string(stderr)?
        );
        let status = status.context("observe strategy final child")?;
        ensure!(
            status.is_some_and(|status| status.success()),
            "strategy final child failed or exceeded 120 seconds: {status:?}"
        );
        ensure!(
            std::fs::read(owned.0.join("completed"))? == b"actual-strategy-final-pass",
            "strategy final child did not complete its evaluator and reader assertions"
        );
        Ok(())
    }

    #[test]
    fn mismatched_costs_do_not_consume_final_scope_or_start_features() -> Result<()> {
        use neoethos_search::canonical_trendbar_research::{
            CanonicalTrendbarResearchExecutionContractRefV1,
            CanonicalTrendbarResearchExecutionContractV3,
        };
        let original = runtime_fixture_handoff()?;
        // Deliberately create an individually valid but detached cost contract.
        // Its exact identity is rebuilt by the real typed reference writer.
        let mut contract_wire = serde_json::to_value(original.screening_contract())?;
        contract_wire["screening_costs"]["full_spread_pips_assumption"] = serde_json::json!(
            original
                .screening_contract()
                .screening_costs()
                .full_spread_pips_assumption()
                + 0.25
        );
        let contract: CanonicalTrendbarResearchExecutionContractV3 =
            serde_json::from_value(contract_wire)?;
        contract.validate()?;
        let mut wire = serde_json::to_value(&original)?;
        wire["screening_contract"] = serde_json::to_value(
            CanonicalTrendbarResearchExecutionContractRefV1::from_contract(&contract)?,
        )?;
        let detached: PromotionCandidateTrainingHandoffV1 = serde_json::from_value(wire)?;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "neoethos-refused-final-costs-{}-{stamp}",
            std::process::id()
        ));
        ensure!(!root.exists(), "refusal-test path must be new");
        let mut settings = Settings::default();
        settings.system.data_dir = root.join("data");
        let error = evaluate_strategies(
            &settings,
            &root.join("candidates"),
            &detached,
            &CancellationFlag::new(),
            &|_, _, _| panic!("detached costs must fail before any feature phase"),
        )
        .unwrap_err();
        ensure!(format!("{error:#}").contains(": spread_pips (evaluation "));
        ensure!(
            !root.exists(),
            "detached costs created a final-use journal or report"
        );
        Ok(())
    }

    #[test]
    fn exact_alignment_never_guesses_tail_rows() {
        assert_eq!(exact_rows(&[10, 20, 30, 40], &[20, 40]).unwrap(), [1, 3]);
        assert!(exact_rows(&[10, 20, 30], &[25]).is_err());
        assert!(exact_rows(&[10, 10, 30], &[10]).is_err());
        assert!(exact_rows(&[10, 20, 30], &[20, 10]).is_err());
    }

    #[test]
    fn causal_queue_uses_parallel_single_worker_reservations_and_restores_row_order() {
        let width = WorkerLimit::new(4).unwrap();
        let broker = CpuPermitBroker::new(width);
        let lease = broker.acquire(CpuPermitRequest::local(width)).unwrap();
        let barrier = std::sync::Barrier::new(4);
        let calls = AtomicUsize::new(0);
        let values = parallel_rows(
            100,
            lease,
            &AtomicBool::new(false),
            |row, worker| {
                assert_eq!(worker.width().get(), 1);
                if calls.fetch_add(1, Ordering::SeqCst) < 4 {
                    barrier.wait();
                }
                Ok(row * row)
            },
            |_| {},
        )
        .unwrap();
        assert_eq!(values, (0..100).map(|row| row * row).collect::<Vec<_>>());
        assert!(
            broker
                .try_acquire(CpuPermitRequest::local(width))
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn causal_queue_does_not_publish_partial_or_cancelled_predictions() {
        let width = WorkerLimit::new(2).unwrap();
        let broker = CpuPermitBroker::new(width);
        for cancelled in [false, true] {
            let lease = broker.acquire(CpuPermitRequest::local(width)).unwrap();
            let result = parallel_rows(
                12,
                lease,
                &AtomicBool::new(cancelled),
                |row, _| {
                    ensure!(row != 5, "deliberate model failure");
                    Ok(row)
                },
                |_| {},
            );
            assert!(result.is_err());
            assert!(
                broker
                    .try_acquire(CpuPermitRequest::local(width))
                    .unwrap()
                    .is_some()
            );
        }
    }
}
