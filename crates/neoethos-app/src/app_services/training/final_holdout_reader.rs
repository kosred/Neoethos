//! Read-only, explicitly requested projection of saved final-window research.
//! A local journal is not broker admission or proof of historical non-exposure.

use super::{FirstUse, JOURNAL_SCHEMA, hash_material, raw_scope_material};
use anyhow::{Context, Result, ensure};
use neoethos_models::{PromotionCandidateTrainingHandoffV1, PromotionCandidateTrainingManifestV1};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Component, Path, PathBuf};

// Display-reader I/O bounds, never Search/candidate evaluation limits. Refuse
// explicitly instead of silently showing a profit-selected/truncated subset.
const MAX_REPORT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_REQUEST_BYTES: u64 = 256 * 1024 * 1024;
const MAX_JOURNAL_ENTRIES: usize = 4096;
const MAX_MATCHING_ATTEMPTS: usize = 64;
const PROTOCOL: &str = "neoethos.candidate-combined-bar-research.v2";
const STRATEGY_PROTOCOL: &str = "neoethos.candidate-strategy-bar-research.v1";
const STRATEGY_SUFFIX: &str = "strategy-bar-research.json";
const FIRST_USE: &str = "first_recorded_local_use_of_reserved_final_scope";
const REUSED: &str = "reused_reserved_final_scope_research_only";
const EXPOSURE: &str = "unknown_before_this_local_journal_not_never_ever_seen_evidence";

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CombinedResearchReportsDto {
    pub training_handoff: String,
    pub status: &'static str,
    pub reports: Vec<CombinedResearchReportDto>,
    pub unavailable: Vec<UnavailableCombinedResearchReportDto>,
}

impl CombinedResearchReportsDto {
    pub(crate) fn candidate_not_ready(identity: &str) -> Self {
        Self {
            training_handoff: identity.to_owned(),
            status: "candidate_not_ready",
            reports: Vec::new(),
            unavailable: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct UnavailableCombinedResearchReportDto {
    pub report_id: String,
    pub reason: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CombinedResearchReportDto {
    pub evaluation_mode: &'static str,
    /// Internal reuse by the candidate loader; never a new UI trust flag.
    #[serde(skip)]
    pub(crate) model_inference_settings: Value,
    pub report_id: String,
    pub report_sha256: String,
    pub raw_final_scope_sha256: String,
    pub locked_final_inputs_sha256: String,
    pub first_locked_final_inputs_sha256: Option<String>,
    pub holdout_use: String,
    pub historical_exposure: String,
    pub symbol: String,
    pub base_timeframe: String,
    pub account_currency: String,
    pub row_start: u64,
    pub row_end: u64,
    pub rows: usize,
    pub timestamp_start_ms: i64,
    pub timestamp_end_ms: i64,
    pub training_cutoff_ms: i64,
    pub blend_mode: Option<String>,
    pub blend_gate_floor: Option<f64>,
    pub blend_veto_below: Option<f64>,
    pub model_history_rows: Option<usize>,
    pub configured_live_ml_gate: Option<bool>,
    pub invalid_model_signal_rows: Option<usize>,
    pub promotion_eligible: bool,
    pub gene_only: CombinedResearchAccountDto,
    pub combined: Option<CombinedResearchAccountDto>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct CombinedResearchAccountDto {
    pub net_profit: Option<f64>,
    pub sharpe: Option<f64>,
    pub win_rate: Option<f64>,
    pub profit_factor: Option<f64>,
    pub expectancy: Option<f64>,
    pub trade_count: usize,
    pub max_drawdown_fraction: Option<f64>,
    pub ending_realized_balance: f64,
    pub terminal_open: bool,
    pub gross_unrealized_account: Option<f64>,
    pub pending_round_trip_commission_account: Option<f64>,
    pub below_min_entries: usize,
}

/// Only metadata, journal and saved reports are read. The installed manifest and
/// model-input contract are verified ONCE for this request, never per report.
/// No Settings reconstruction, final feature/data read, inference or evaluation.
pub(crate) fn read_saved_combined_research_reports(
    data_root: &Path,
    candidate_root: &Path,
    requested_handoff: &str,
    manifest: &PromotionCandidateTrainingManifestV1,
) -> Result<CombinedResearchReportsDto> {
    let context = verified_context(candidate_root, requested_handoff, manifest)?;
    read_verified_reports(data_root, candidate_root, &context)
}

/// Read both final-research modes from the exact saved handoff. Model evidence
/// is loaded at most once, and only if a matching combined attempt needs it.
/// A failed model verification cannot hide an independently bound strategy report.
#[cfg(test)]
pub(crate) fn read_saved_final_research_reports(
    data_root: &Path,
    candidate_root: &Path,
    requested_handoff: &str,
) -> Result<CombinedResearchReportsDto> {
    let context = load_saved_final_research_context(data_root, requested_handoff)?;
    read_saved_final_research_reports_with_context(data_root, candidate_root, context)
}

/// Load and validate the selected handoff once. The HTTP caller keeps its
/// existing selection-error classification before consuming this local value.
pub(crate) fn load_saved_final_research_context(
    data_root: &Path,
    requested_handoff: &str,
) -> Result<VerifiedContext> {
    let (handoff, portfolio) =
        super::super::handoff::load_with_live_portfolio(data_root, requested_handoff)?;
    context_from_validated_portfolio(&handoff, &portfolio, requested_handoff.to_owned())
}

/// Consume one checked request-local projection, never a persisted/cached proof.
/// Journal, reports and any installed-model evidence remain freshly verified.
pub(crate) fn read_saved_final_research_reports_with_context(
    data_root: &Path,
    candidate_root: &Path,
    context: VerifiedContext,
) -> Result<CombinedResearchReportsDto> {
    let requested_handoff = &context.handoff;
    read_reports(data_root, candidate_root, &context, true, || {
        let bytes =
            neoethos_broker_history::canonical_research_costs::read_regular_file_with_limit(
                &candidate_root.join(format!("{requested_handoff}.manifest.json")),
                neoethos_models::MAX_PROMOTION_CANDIDATE_HANDOFF_BYTES_V1 as u64,
            )?;
        let manifest: PromotionCandidateTrainingManifestV1 = serde_json::from_slice(&bytes)?;
        verified_context(candidate_root, requested_handoff, &manifest)
    })
}

fn verified_context(
    candidate_root: &Path,
    requested_handoff: &str,
    manifest: &PromotionCandidateTrainingManifestV1,
) -> Result<VerifiedContext> {
    ensure_sha(requested_handoff)?;
    let handoff = manifest.reopen_handoff(candidate_root)?;
    let mut context = verified_handoff_context(&handoff)?;
    ensure!(
        context.handoff == requested_handoff,
        "candidate manifest belongs to a different training handoff"
    );
    let input = neoethos_models::runtime::feature_input::load_model_feature_input_for_handoff_v1(
        &candidate_root.join(manifest.candidate_relative_dir()),
        &handoff,
    )?;
    context.model = Some(VerifiedModelContext {
        candidate_tree: manifest.candidate_tree_sha256().to_owned(),
        model_input: format!("{:x}", Sha256::digest(input.to_json_bytes()?)),
    });
    Ok(context)
}

fn verified_handoff_context(
    handoff: &PromotionCandidateTrainingHandoffV1,
) -> Result<VerifiedContext> {
    // Reuse one complete validation and bounded encoding for this immutable call.
    let (portfolio, handoff_identity) = handoff.validated_live_portfolio_and_identity_sha256()?;
    context_from_validated_portfolio(handoff, &portfolio, handoff_identity)
}

fn context_from_validated_portfolio(
    handoff: &PromotionCandidateTrainingHandoffV1,
    portfolio: &neoethos_search::live_portfolio::LivePortfolioArtifact,
    handoff_identity: String,
) -> Result<VerifiedContext> {
    let evaluation = portfolio.live_trading_policy.sealed_evaluation_config()?;
    let scope = &portfolio.final_holdout_scope;
    let window = scope.evaluated_window();
    Ok(VerifiedContext {
        handoff: handoff_identity,
        portfolio: handoff.locked_portfolio().identity_sha256().to_owned(),
        model: None,
        raw_scope: raw_scope_material(scope)?,
        final_window: serde_json::to_value(window)?,
        final_scope_identity: scope.identity_sha256()?,
        cost_identity: handoff.screening_contract().identity_sha256()?,
        symbol: portfolio.symbol.clone(),
        base_timeframe: portfolio.base_tf.clone(),
        account_currency: evaluation.account_currency.clone(),
        row_start: window.row_start() as u64,
        row_end: window.row_end() as u64,
        timestamp_start_ms: window.timestamp_start_ms(),
        timestamp_end_ms: window.timestamp_end_ms(),
        training_cutoff_ms: handoff.oos_cutoff_ms(),
    })
}

// Private request-local proof projection. Tests exercise the journal/reader seam
// with explicitly synthetic scope metadata, not a fake installed-model API.
pub(crate) struct VerifiedContext {
    handoff: String,
    portfolio: String,
    model: Option<VerifiedModelContext>,
    raw_scope: Value,
    final_window: Value,
    final_scope_identity: String,
    cost_identity: String,
    symbol: String,
    base_timeframe: String,
    account_currency: String,
    row_start: u64,
    row_end: u64,
    timestamp_start_ms: i64,
    timestamp_end_ms: i64,
    training_cutoff_ms: i64,
}

struct VerifiedModelContext {
    candidate_tree: String,
    model_input: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Start {
    schema: String,
    raw_scope_sha256: String,
    locked_inputs_sha256: String,
    first_recorded_use: bool,
    raw_scope: Value,
    locked_inputs: Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Completion {
    schema: String,
    raw_scope_sha256: String,
    locked_inputs_sha256: String,
    report_path: PathBuf,
    report_sha256: String,
    first_recorded_use: bool,
}

#[derive(Default)]
struct ReadBudget {
    consumed: u64,
}
impl ReadBudget {
    fn read(&mut self, path: &Path, limit: u64) -> Result<Vec<u8>> {
        ensure_physical(path, false)?;
        let length = std::fs::metadata(path)?.len();
        ensure!(
            length <= limit,
            "saved research input exceeds per-file display byte limit"
        );
        ensure!(
            length <= MAX_REQUEST_BYTES.saturating_sub(self.consumed),
            "saved research request exceeds total display byte limit"
        );
        let remaining = MAX_REQUEST_BYTES.saturating_sub(self.consumed);
        let bytes =
            neoethos_broker_history::canonical_research_costs::read_regular_file_with_limit(
                path,
                limit.min(remaining),
            )?;
        self.consumed += bytes.len() as u64;
        Ok(bytes)
    }
}

fn ensure_physical(path: &Path, directory: bool) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    ensure!(
        !metadata.file_type().is_symlink(),
        "saved research path must not be a symlink"
    );
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        ensure!(
            metadata.file_attributes() & 0x400 == 0,
            "saved research path must not be a reparse point"
        );
    }
    ensure!(
        if directory {
            metadata.is_dir()
        } else {
            metadata.is_file()
        },
        "saved research path has the wrong file type"
    );
    Ok(())
}

fn read_verified_reports(
    data_root: &Path,
    candidate_root: &Path,
    context: &VerifiedContext,
) -> Result<CombinedResearchReportsDto> {
    read_reports(data_root, candidate_root, context, false, || {
        anyhow::bail!("combined reader requires its already verified model context")
    })
}

fn read_reports(
    data_root: &Path,
    candidate_root: &Path,
    context: &VerifiedContext,
    include_strategy: bool,
    mut load_model_context: impl FnMut() -> Result<VerifiedContext>,
) -> Result<CombinedResearchReportsDto> {
    let mut result = CombinedResearchReportsDto::candidate_not_ready(&context.handoff);
    if context.model.is_some() {
        result.status = "no_completed_result";
    }
    let raw_hash = hash_material(b"neoethos.final-raw-window.v1\0", &context.raw_scope)?;
    let journal_root = data_root.join("final_holdout_uses");
    if !journal_root.try_exists()? {
        return Ok(result);
    }
    ensure_physical(&journal_root, true)?;
    let directory = journal_root.join(&raw_hash);
    if !directory.try_exists()? {
        return Ok(result);
    }
    ensure_physical(&directory, true)?;
    let mut attempts = Vec::new();
    for (index, entry) in std::fs::read_dir(&directory)?.enumerate() {
        ensure!(
            index < MAX_JOURNAL_ENTRIES,
            "saved research journal exceeds display entry limit"
        );
        let entry = entry?;
        let name = entry.file_name();
        let name = name
            .to_str()
            .context("saved journal filename is not UTF-8")?;
        if let Some(attempt) = name.strip_suffix(".completed.json") {
            ensure!(
                !attempt.is_empty() && attempt.bytes().all(|b| b.is_ascii_digit() || b == b'-'),
                "invalid saved final attempt filename"
            );
            attempts.push(attempt.to_owned());
        }
    }
    // Stable enumeration, deliberately no ranking by profit or latest attempt.
    attempts.sort();
    let mut budget = ReadBudget::default();
    let mut matched = 0;
    let mut completed_first_claims = 0;
    let mut loaded_model: Option<std::result::Result<VerifiedContext, String>> = None;
    for attempt in attempts {
        let start: Start = serde_json::from_slice(&budget.read(
            &directory.join(format!("{attempt}.start.json")),
            1024 * 1024,
        )?)
        .context("cannot attribute completed attempt to its start record")?;
        ensure!(
            start.schema == JOURNAL_SCHEMA
                && start.raw_scope == context.raw_scope
                && start.raw_scope_sha256 == raw_hash,
            "start record raw final scope changed"
        );
        ensure_sha(&start.locked_inputs_sha256)?;
        ensure!(
            hash_material(b"neoethos.final-locked-inputs.v1\0", &start.locked_inputs)?
                == start.locked_inputs_sha256,
            "locked final inputs hash mismatch before attempt attribution"
        );
        completed_first_claims += usize::from(start.first_recorded_use);
        ensure!(
            completed_first_claims <= 1,
            "multiple completed attempts claim the same first local final-window use"
        );
        let selected = text_field(&start.locked_inputs, "training_handoff")?;
        ensure_sha(selected)?;
        if selected != context.handoff {
            continue;
        }
        let protocol = text_field(&start.locked_inputs, "protocol")?;
        let strategy_only = protocol == STRATEGY_PROTOCOL;
        if strategy_only && !include_strategy {
            continue;
        }
        matched += 1;
        ensure!(
            matched <= MAX_MATCHING_ATTEMPTS,
            "saved research exceeds matching-attempt display limit"
        );
        let report_id = format!(
            "{}.{}.{}",
            start.locked_inputs_sha256,
            attempt,
            if strategy_only {
                STRATEGY_SUFFIX
            } else {
                super::super::combined::REPORT_SUFFIX
            }
        );
        let read = (|| {
            ensure!(
                strategy_only || protocol == PROTOCOL,
                "unsupported saved final research protocol"
            );
            // An in-progress strategy attempt may not have created its output
            // directory yet. Require that physical directory only for a saved completion.
            ensure_physical(candidate_root, true)?;
            let context = if strategy_only || context.model.is_some() {
                context
            } else {
                match loaded_model
                    .get_or_insert_with(|| load_model_context().map_err(|e| format!("{e:#}")))
                {
                    Ok(context) => context,
                    Err(error) => anyhow::bail!("combined model evidence unavailable: {error}"),
                }
            };
            read_attempt(
                candidate_root,
                &directory,
                context,
                &raw_hash,
                &attempt,
                &start,
                &mut budget,
                strategy_only,
            )
        })();
        match read {
            Ok(report) => result.reports.push(report),
            Err(error) => result
                .unavailable
                .push(UnavailableCombinedResearchReportDto {
                    report_id,
                    reason: format!("{error:#}"),
                }),
        }
    }
    if matched != 0 {
        result.status = "completed_results";
    }
    Ok(result)
}

fn read_attempt(
    candidate_root: &Path,
    directory: &Path,
    context: &VerifiedContext,
    raw_hash: &str,
    attempt: &str,
    start: &Start,
    budget: &mut ReadBudget,
    strategy_only: bool,
) -> Result<CombinedResearchReportDto> {
    let locked = &start.locked_inputs;
    if strategy_only {
        exact_keys(
            locked,
            &[
                "protocol",
                "training_handoff",
                "locked_portfolio_identity_sha256",
                "account_policy",
            ],
        )?;
        matches_text(locked, "protocol", STRATEGY_PROTOCOL)?;
    } else {
        exact_keys(
            locked,
            &[
                "protocol",
                "training_handoff",
                "locked_portfolio_identity_sha256",
                "candidate_tree_sha256",
                "model_input_sha256",
                "model_inference_settings",
                "blend_mode",
                "blend_gate_floor",
                "blend_veto_below",
                "model_history_rows",
                "account_policy",
            ],
        )?;
        matches_text(locked, "protocol", PROTOCOL)?;
    }
    matches_text(
        locked,
        "account_policy",
        "archived_search_risk_fractional_lots_v1",
    )?;
    for (key, expected) in [
        ("training_handoff", &context.handoff),
        ("locked_portfolio_identity_sha256", &context.portfolio),
    ] {
        matches_text(locked, key, expected)?;
    }
    if !strategy_only {
        let model = context
            .model
            .as_ref()
            .context("combined report has no verified model context")?;
        matches_text(locked, "candidate_tree_sha256", &model.candidate_tree)?;
        matches_text(locked, "model_input_sha256", &model.model_input)?;
        matches_text(locked, "blend_mode", "ml_scale")?;
        ensure!(
            count_field(locked, "model_history_rows")? == 256,
            "saved model history policy changed"
        );
        let settings = field(locked, "model_inference_settings")?;
        ensure!(
            settings.is_object(),
            "saved model inference settings are unavailable"
        );
        let blend = neoethos_trader::BlendConfig::from_config_values(
            neoethos_trader::BlendMode::MlScale,
            Some(number_field(settings, "blend_gate_floor")?),
            Some(number_field(settings, "blend_veto_below")?),
        );
        ensure!(
            number_field(locked, "blend_gate_floor")? == blend.gate_floor
                && number_field(locked, "blend_veto_below")? == blend.veto_below,
            "saved effective blend differs from locked inference settings"
        );
    }
    let completion: Completion = serde_json::from_slice(
        &budget.read(&directory.join(format!("{attempt}.completed.json")), 65_536)?,
    )?;
    ensure!(
        completion.schema == JOURNAL_SCHEMA
            && completion.raw_scope_sha256 == raw_hash
            && completion.locked_inputs_sha256 == start.locked_inputs_sha256
            && completion.first_recorded_use == start.first_recorded_use,
        "completion does not match its start record"
    );
    ensure_sha(&completion.report_sha256)?;
    let first_bytes = budget.read(&directory.join("first-start.json"), 65_536);
    // An interrupted/oversize first marker must remain consumed. It can only
    // support reused/unknown-first diagnostics, never an upgraded fresh claim.
    let first = match first_bytes {
        Ok(bytes) => serde_json::from_slice::<FirstUse>(&bytes).ok(),
        Err(error) => {
            let metadata = std::fs::symlink_metadata(directory.join("first-start.json"))?;
            ensure_physical(&directory.join("first-start.json"), false)?;
            if metadata.len() <= 65_536 {
                return Err(error);
            }
            None
        }
    };
    let first_locked = if let Some(first) = first {
        ensure!(
            first.schema == JOURNAL_SCHEMA && first.raw_scope_sha256 == raw_hash,
            "first marker belongs to a different final scope"
        );
        ensure_sha(&first.locked_inputs_sha256)?;
        Some(first.locked_inputs_sha256)
    } else {
        None
    };
    if start.first_recorded_use {
        ensure!(
            first_locked.as_deref() == Some(start.locked_inputs_sha256.as_str()),
            "first-use claim has no matching complete first marker"
        );
    }
    let report_id = format!(
        "{}.{}.{}",
        start.locked_inputs_sha256,
        attempt,
        if strategy_only {
            STRATEGY_SUFFIX
        } else {
            super::super::combined::REPORT_SUFFIX
        }
    );
    let path = candidate_root.join(&report_id);
    ensure!(
        absolute_lexical(&completion.report_path)? == absolute_lexical(&path)?,
        "completion report path is not its exact candidate-owned report"
    );
    let report: Value = serde_json::from_slice(&budget.read(&path, MAX_REPORT_BYTES)?)?;
    ensure!(
        hash_material(b"neoethos.final-report.v1\0", &report)? == completion.report_sha256,
        "saved report semantic SHA-256 mismatch"
    );
    let saved_first = field(&report, "first_locked_final_inputs_sha256")?;
    let recorded_first = if saved_first.is_null() {
        ensure!(
            !start.first_recorded_use,
            "first-use report cannot omit the first locked identity"
        );
        // A concurrent first writer may have completed its marker since this
        // reused attempt observed an incomplete one. Keep the saved unknown.
        None
    } else {
        let saved_first = saved_first
            .as_str()
            .context("saved first locked identity is not text")?;
        ensure!(
            first_locked.as_deref() == Some(saved_first),
            "saved first-use identity differs from local journal"
        );
        Some(saved_first.to_owned())
    };
    project_report(
        &report,
        context,
        raw_hash,
        start,
        recorded_first,
        report_id,
        completion.report_sha256,
        strategy_only,
    )
}

fn absolute_lexical(path: &Path) -> Result<PathBuf> {
    ensure!(
        !path
            .components()
            .any(|part| matches!(part, Component::ParentDir)),
        "saved report path contains parent traversal"
    );
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    #[cfg(windows)]
    if let Some(Component::Prefix(prefix)) = path.components().next() {
        if let std::path::Prefix::VerbatimDisk(drive) = prefix.kind() {
            // Only ordinary, generated disk paths have equivalent semantics.
            // Verbatim '/' is literal, and trailing dots/spaces or DOS device
            // names can change meaning when the prefix is removed.
            let spelling = path
                .to_str()
                .context("verbatim report path is not Unicode")?;
            ensure!(
                !spelling.contains('/'),
                "verbatim report path contains a literal slash"
            );
            let mut ordinary = PathBuf::from(format!("{}:\\", char::from(drive)));
            for component in path.components().skip(1) {
                match component {
                    Component::RootDir => {}
                    Component::Normal(name) => {
                        let name = name
                            .to_str()
                            .context("verbatim report component is not Unicode")?;
                        let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
                        let device = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
                            || (stem.len() == 4
                                && (stem.starts_with("COM") || stem.starts_with("LPT"))
                                && stem.as_bytes()[3].is_ascii_digit());
                        ensure!(
                            name != "."
                                && name != ".."
                                && !name.ends_with('.')
                                && !name.ends_with(' ')
                                && !name.contains(':')
                                && !device,
                            "verbatim report component is not an ordinary disk filename"
                        );
                        ordinary.push(name);
                    }
                    _ => anyhow::bail!("verbatim report path has nonordinary components"),
                }
            }
            return Ok(ordinary);
        }
    }
    Ok(path
        .components()
        .filter(|part| !matches!(part, Component::CurDir))
        .collect())
}

fn project_report(
    report: &Value,
    context: &VerifiedContext,
    raw_hash: &str,
    start: &Start,
    first_locked: Option<String>,
    report_id: String,
    report_sha256: String,
    strategy_only: bool,
) -> Result<CombinedResearchReportDto> {
    let all_keys = [
        "schema",
        "training_handoff",
        "locked_portfolio_identity_sha256",
        "candidate_tree_sha256",
        "model_input_sha256",
        "symbol",
        "base_timeframe",
        "rows",
        "timestamp_start_ms",
        "timestamp_end_ms",
        "model_history_rows",
        "inference_workers",
        "invalid_model_signal_rows",
        "blend_mode",
        "blend_gate_floor",
        "blend_veto_below",
        "configured_live_ml_gate",
        "holdout_use",
        "historical_exposure",
        "raw_final_scope_sha256",
        "locked_final_inputs_sha256",
        "first_locked_final_inputs_sha256",
        "final_scope_window",
        "training_cutoff_ms",
        "sizing_basis",
        "exit_basis",
        "volume_basis",
        "promotion_eligible",
        "gene_only",
        "combined",
    ];
    let model_keys = [
        "candidate_tree_sha256",
        "model_input_sha256",
        "model_history_rows",
        "inference_workers",
        "invalid_model_signal_rows",
        "blend_mode",
        "blend_gate_floor",
        "blend_veto_below",
        "configured_live_ml_gate",
        "combined",
    ];
    let keys: Vec<_> = all_keys
        .into_iter()
        .filter(|key| !strategy_only || !model_keys.contains(key))
        .collect();
    exact_keys(report, &keys)?;
    matches_text(
        report,
        "schema",
        if strategy_only {
            STRATEGY_PROTOCOL
        } else {
            PROTOCOL
        },
    )?;
    for (key, expected) in [
        ("training_handoff", &context.handoff),
        ("locked_portfolio_identity_sha256", &context.portfolio),
        ("symbol", &context.symbol),
        ("base_timeframe", &context.base_timeframe),
    ] {
        matches_text(report, key, expected)?;
    }
    if !strategy_only {
        let model = context
            .model
            .as_ref()
            .context("combined report has no verified model context")?;
        matches_text(report, "candidate_tree_sha256", &model.candidate_tree)?;
        matches_text(report, "model_input_sha256", &model.model_input)?;
    }
    matches_text(report, "raw_final_scope_sha256", raw_hash)?;
    matches_text(
        report,
        "locked_final_inputs_sha256",
        &start.locked_inputs_sha256,
    )?;
    ensure!(
        field(report, "first_locked_final_inputs_sha256")? == &serde_json::to_value(&first_locked)?,
        "saved first-use identity differs from local journal"
    );
    let use_label = if start.first_recorded_use {
        FIRST_USE
    } else {
        REUSED
    };
    matches_text(report, "holdout_use", use_label)?;
    matches_text(report, "historical_exposure", EXPOSURE)?;
    ensure!(
        field(report, "final_scope_window")? == &context.final_window,
        "saved report final window/role changed"
    );
    ensure!(
        integer_field(report, "training_cutoff_ms")? == context.training_cutoff_ms
            && context.timestamp_start_ms > context.training_cutoff_ms,
        "saved report training/final boundary changed"
    );
    let rows = usize::try_from(
        context
            .row_end
            .checked_sub(context.row_start)
            .context("invalid final row range")?,
    )?;
    ensure!(
        rows > 0
            && count_field(report, "rows")? == rows
            && integer_field(report, "timestamp_start_ms")? == context.timestamp_start_ms
            && integer_field(report, "timestamp_end_ms")? == context.timestamp_end_ms,
        "saved report final geometry changed"
    );
    let (
        blend_mode,
        blend_gate_floor,
        blend_veto_below,
        model_history_rows,
        configured_live_ml_gate,
        invalid,
    ) = if strategy_only {
        (None, None, None, None, None, None)
    } else {
        for key in [
            "blend_mode",
            "blend_gate_floor",
            "blend_veto_below",
            "model_history_rows",
        ] {
            ensure!(
                field(report, key)? == field(&start.locked_inputs, key)?,
                "report {key} differs from locked policy"
            );
        }
        let configured_live_ml_gate = bool_field(report, "configured_live_ml_gate")?;
        ensure!(
            configured_live_ml_gate
                == bool_field(
                    field(&start.locked_inputs, "model_inference_settings")?,
                    "live_ml_gate"
                )?,
            "report ML gate differs from locked inference settings"
        );
        ensure!(
            count_field(report, "inference_workers")? > 0,
            "saved inference worker count is invalid"
        );
        let invalid = count_field(report, "invalid_model_signal_rows")?;
        ensure!(
            invalid <= rows,
            "saved invalid prediction count exceeds final rows"
        );
        (
            Some("ml_scale".to_owned()),
            Some(number_field(report, "blend_gate_floor")?),
            Some(number_field(report, "blend_veto_below")?),
            Some(256),
            Some(configured_live_ml_gate),
            Some(invalid),
        )
    };
    ensure!(
        !bool_field(report, "promotion_eligible")?,
        "saved bar research must not claim promotion authority"
    );
    matches_text(
        report,
        "sizing_basis",
        "archived_search_confidence_risk_band_not_live_Risky_or_PropFirm_account_simulation",
    )?;
    matches_text(
        report,
        "exit_basis",
        "canonical_bar_brackets_trailing_time_and_session_policy_not_live_reversal_or_supervisor_actions",
    )?;
    matches_text(
        report,
        "volume_basis",
        "fractional_research_lots_broker_grid_not_attested",
    )?;
    Ok(CombinedResearchReportDto {
        evaluation_mode: if strategy_only {
            "strategy_only"
        } else {
            "train_models"
        },
        model_inference_settings: if strategy_only {
            Value::Null
        } else {
            field(&start.locked_inputs, "model_inference_settings")?.clone()
        },
        report_id,
        report_sha256,
        raw_final_scope_sha256: raw_hash.to_owned(),
        locked_final_inputs_sha256: start.locked_inputs_sha256.clone(),
        first_locked_final_inputs_sha256: first_locked,
        holdout_use: use_label.to_owned(),
        historical_exposure: EXPOSURE.to_owned(),
        symbol: context.symbol.clone(),
        base_timeframe: context.base_timeframe.clone(),
        account_currency: context.account_currency.clone(),
        row_start: context.row_start,
        row_end: context.row_end,
        rows,
        timestamp_start_ms: context.timestamp_start_ms,
        timestamp_end_ms: context.timestamp_end_ms,
        training_cutoff_ms: context.training_cutoff_ms,
        blend_mode,
        blend_gate_floor,
        blend_veto_below,
        model_history_rows,
        configured_live_ml_gate,
        invalid_model_signal_rows: invalid,
        promotion_eligible: false,
        gene_only: project_account(field(report, "gene_only")?, context)?,
        combined: if strategy_only {
            None
        } else {
            Some(project_account(field(report, "combined")?, context)?)
        },
    })
}

fn project_account(
    account: &Value,
    context: &VerifiedContext,
) -> Result<CombinedResearchAccountDto> {
    exact_keys(
        account,
        &[
            "execution_basis",
            "promotion_eligible",
            "scope_identity_sha256",
            "cost_contract_identity_sha256",
            "account_currency",
            "metrics",
            "closed_trades",
            "ending_realized_balance",
            "terminal_open",
            "below_min_entries",
            "lot_grid",
        ],
    )?;
    matches_text(
        account,
        "execution_basis",
        "canonical_cpu_ohlc_screening_prior_bar_signal_next_close_fill; scalar_account_pip_and_cost_assumptions; closed_trade_realized_pnl; open_gross_mark_before_pending_costs",
    )?;
    ensure!(
        !bool_field(account, "promotion_eligible")?,
        "bar account claims promotion authority"
    );
    matches_text(
        account,
        "scope_identity_sha256",
        &context.final_scope_identity,
    )?;
    matches_text(
        account,
        "cost_contract_identity_sha256",
        &context.cost_identity,
    )?;
    matches_text(account, "account_currency", &context.account_currency)?;
    ensure!(
        field(account, "lot_grid")?.is_null(),
        "saved account is not the declared fractional research volume policy"
    );
    let metrics = field(account, "metrics")?;
    exact_keys(
        metrics,
        &[
            "net_profit",
            "sharpe",
            "peak_equity",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "expectancy",
            "monthly_target_hit_rate",
            "trade_count",
            "consistency",
            "max_daily_drawdown",
        ],
    )?;
    for key in [
        "net_profit",
        "sharpe",
        "peak_equity",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "expectancy",
        "monthly_target_hit_rate",
        "consistency",
        "max_daily_drawdown",
    ] {
        nullable_number(metrics, key)?;
    }
    let trades = field(account, "closed_trades")?
        .as_array()
        .context("saved closed trades are not an array")?;
    let trade_count = count_field(metrics, "trade_count")?;
    ensure!(
        trade_count == trades.len(),
        "saved trade count differs from closed trade evidence"
    );
    for trade in trades {
        let entry = integer_field(trade, "entry_time")?;
        let exit = integer_field(trade, "exit_time")?;
        ensure!(
            entry >= context.timestamp_start_ms
                && exit >= entry
                && exit <= context.timestamp_end_ms,
            "saved closed trade lies outside the final scope"
        );
        for key in ["pnl", "mfe", "mae", "r_multiple"] {
            number_field(trade, key)?;
        }
        for key in ["pnl_pct", "duration_hours"] {
            nullable_number(trade, key)?;
        }
    }
    let terminal = field(account, "terminal_open")?;
    let (gross, pending) = if terminal.is_null() {
        (None, None)
    } else {
        ensure!(
            matches!(integer_field(terminal, "direction")?, -1 | 1),
            "saved open position direction is invalid"
        );
        ensure!(
            integer_field(terminal, "entry_timestamp_ms")? >= context.timestamp_start_ms
                && integer_field(terminal, "entry_timestamp_ms")? <= context.timestamp_end_ms
                && integer_field(terminal, "mark_timestamp_ms")? == context.timestamp_end_ms,
            "saved open position is outside final scope"
        );
        ensure!(
            count_field(terminal, "entry_bar_index")?
                < usize::try_from(context.row_end - context.row_start)?,
            "saved open entry index is outside final scope"
        );
        for key in [
            "modeled_entry_price",
            "lots",
            "stop_pips",
            "target_pips",
            "mark_close_price",
            "marked_equity_before_pending_costs",
        ] {
            number_field(terminal, key)?;
        }
        nullable_number(terminal, "active_trailing_stop_price")?;
        (
            Some(number_field(terminal, "gross_unrealized_account")?),
            Some(number_field(
                terminal,
                "pending_round_trip_commission_account",
            )?),
        )
    };
    Ok(CombinedResearchAccountDto {
        net_profit: nullable_number(metrics, "net_profit")?,
        sharpe: nullable_number(metrics, "sharpe")?,
        win_rate: nullable_number(metrics, "win_rate")?,
        profit_factor: nullable_number(metrics, "profit_factor")?,
        expectancy: nullable_number(metrics, "expectancy")?,
        trade_count,
        max_drawdown_fraction: nullable_number(metrics, "max_drawdown")?,
        ending_realized_balance: number_field(account, "ending_realized_balance")?,
        terminal_open: !terminal.is_null(),
        gross_unrealized_account: gross,
        pending_round_trip_commission_account: pending,
        below_min_entries: count_field(account, "below_min_entries")?,
    })
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value
        .get(key)
        .with_context(|| format!("saved research field {key} is missing"))
}
fn text_field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    field(value, key)?
        .as_str()
        .with_context(|| format!("saved research field {key} is not text"))
}
fn matches_text(value: &Value, key: &str, expected: &str) -> Result<()> {
    ensure!(
        text_field(value, key)? == expected,
        "saved research field {key} does not match its verified context"
    );
    Ok(())
}
fn integer_field(value: &Value, key: &str) -> Result<i64> {
    field(value, key)?
        .as_i64()
        .with_context(|| format!("saved research field {key} is not an integer"))
}
fn count_field(value: &Value, key: &str) -> Result<usize> {
    usize::try_from(
        field(value, key)?
            .as_u64()
            .with_context(|| format!("saved research field {key} is not a nonnegative count"))?,
    )
    .map_err(Into::into)
}
fn bool_field(value: &Value, key: &str) -> Result<bool> {
    field(value, key)?
        .as_bool()
        .with_context(|| format!("saved research field {key} is not boolean"))
}
fn number_field(value: &Value, key: &str) -> Result<f64> {
    let number = field(value, key)?
        .as_f64()
        .with_context(|| format!("saved research field {key} is not a finite number"))?;
    ensure!(
        number.is_finite(),
        "saved research field {key} is not finite"
    );
    Ok(number)
}
fn nullable_number(value: &Value, key: &str) -> Result<Option<f64>> {
    if field(value, key)?.is_null() {
        Ok(None)
    } else {
        number_field(value, key).map(Some)
    }
}
fn ensure_sha(value: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "saved research identity is not canonical lowercase SHA-256"
    );
    Ok(())
}
fn exact_keys(value: &Value, keys: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .context("saved research object is not an object")?;
    ensure!(
        object.len() == keys.len() && keys.iter().all(|key| object.contains_key(*key)),
        "saved research object fields differ from the supported schema"
    );
    Ok(())
}

#[cfg(test)]
#[path = "final_holdout_reader_tests.rs"]
mod tests;
#[cfg(test)]
pub(crate) use tests::{
    install_saved_research_test_fixture, install_saved_strategy_research_test_fixture,
};
