//! Per-strategy IS diagnostics. Membership comes from a validated portfolio;
//! trade/quality JSON is unsealed and cannot establish OOS or promotion authority.

use std::collections::HashSet;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use axum::Json;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use neoethos_core::Settings;
use neoethos_search::LivePortfolioArtifact;
use neoethos_search::data_selection::CanonicalSearchArtifactScopeV2;
use neoethos_search::genetic::Gene;
use neoethos_search::live_portfolio::LiveTradingPolicyV1;
use neoethos_search::validation::ValidationStrategyIdentityV2;
use serde::{Deserialize, Serialize};

use super::state::AppApiState;

const SEED: f64 = 1000.0;
const RESEARCH_DIR: &str = "discovery/research";
const MAX_SCAN_ENTRIES: usize = 10_000;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MonthRow {
    pub month: String,
    pub balance: f64,
    pub return_pct: f64,
    pub trades: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyEntry {
    pub mode: String,
    pub recorded_evaluation: Option<RecordedEvaluation>,
    /// Denominator from the selected unsealed diagnostic rows, not a policy fallback.
    pub diagnostic_initial_capital: f64,
    pub dir: String,
    pub symbol: String,
    pub timeframe: String,
    pub base: String,
    pub strategy_id: String,
    pub exact_gene_hash: String,
    pub trades: usize,
    pub win_rate: Option<f64>,
    pub profit_factor: Option<f64>,
    pub sharpe: Option<f64>,
    pub cpcv_passed: Option<bool>,
    pub walkforward_passed: Option<bool>,
    pub validation_complete: Option<bool>,
    pub span_start: Option<String>,
    pub span_end: Option<String>,
    pub years: f64,
    pub cagr_pct: Option<f64>,
    pub final_from_1000: f64,
    pub max_dd_pct: f64,
    pub flags: Vec<String>,
    pub discovered_at_ms: Option<i64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordedEvaluation {
    pub policy_identity_hash: String,
    pub account_currency: String,
    pub initial_capital: f64,
    pub risk_per_trade_min: f64,
    pub risk_per_trade_max: f64,
    pub high_quality_confidence: f64,
    pub confidence_basis: &'static str,
    pub growth_goal: Option<RecordedGrowthGoal>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordedGrowthGoal {
    pub reference_start_balance: f64,
    pub target_balance: f64,
    pub horizon_days: f64,
}

impl From<neoethos_search::scoring::RiskyGrowthGoal> for RecordedGrowthGoal {
    fn from(goal: neoethos_search::scoring::RiskyGrowthGoal) -> Self {
        Self {
            reference_start_balance: goal.start_balance,
            target_balance: goal.target_balance,
            horizon_days: goal.horizon_days,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyListDto {
    pub count: usize,
    pub strategies: Vec<StrategyEntry>,
    pub unavailable: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StrategyReportDto {
    #[serde(flatten)]
    pub head: StrategyEntry,
    pub monthly: Vec<MonthRow>,
    pub yearly: Vec<MonthRow>,
}

// Private projections of existing fields, NOT a new persisted schema.
// Unused feature/evidence/curve arrays are skipped instead of building a Value tree.
#[derive(Debug, Deserialize)]
struct ReportTrade {
    entry_time: i64,
    exit_time: Option<i64>,
    pnl: f64,
    pnl_pct: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ReportTrades {
    strategy_id: String,
    trades: Vec<ReportTrade>,
}

#[derive(Debug, Deserialize)]
struct ReportQuality {
    strategy_id: String,
    total_trades: usize,
    initial_capital: f64,
    net_profit: f64,
    total_return_pct: f64,
    win_rate: Option<f64>,
    profit_factor: Option<f64>,
    sharpe_ratio: Option<f64>,
}

#[derive(Deserialize)]
struct RecordedMode {
    mode: String,
}

#[derive(Deserialize)]
struct ResearchSelection {
    selection_scope: CanonicalSearchArtifactScopeV2,
    search_config_hash: String,
    portfolio: Vec<Gene>,
    effective_feature_names: Vec<String>,
    logged_trades: Vec<ReportTrades>,
    quality_metrics: Vec<ReportQuality>,
    funnel_profile: Option<RecordedMode>,
}

#[derive(Deserialize)]
struct ResearchProjection {
    schema_version: u16,
    artifact_class: neoethos_search::historical_research::HistoricalResearchArtifactClassV1,
    promotion_eligibility:
        neoethos_search::historical_research::HistoricalResearchPromotionEligibilityV1,
    execution_contract: neoethos_search::CanonicalTrendbarResearchExecutionContractV3,
    discovery_result: ResearchSelection,
    evidence_identity_sha256: String,
}

fn cache_dir(config_path: &Path) -> Result<PathBuf> {
    Ok(Settings::from_yaml(config_path)?.system.cache_dir)
}

#[derive(Debug)]
struct DiagnosticUnavailable(String);

impl std::fmt::Display for DiagnosticUnavailable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for DiagnosticUnavailable {}

fn diagnostic_unavailable(reason: impl std::fmt::Display) -> anyhow::Error {
    DiagnosticUnavailable(reason.to_string()).into()
}

fn diagnostic_input_error(source: anyhow::Error, reason: &str) -> anyhow::Error {
    if source.chain().any(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .is_some_and(|error| error.kind() != std::io::ErrorKind::NotFound)
    }) {
        source
    } else {
        diagnostic_unavailable(reason)
    }
}

fn file_modified_ms(p: &Path) -> Option<i64> {
    let ms = std::fs::metadata(p)
        .ok()?
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_millis();
    i64::try_from(ms).ok()
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let metadata = std::fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "diagnostic input is not a regular file"
    );
    serde_json::from_reader(BufReader::new(std::fs::File::open(path)?))
        .with_context(|| format!("read diagnostic {}", path.display()))
}

fn load_report_portfolio(path: &Path) -> Result<LivePortfolioArtifact> {
    // Preserve typed path errors before the strict loader formats its I/O error.
    let metadata = std::fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "selected diagnostic portfolio is not a regular file"
    );
    neoethos_search::load_live_portfolio_json(path)
}

fn plain_segment(s: &str) -> bool {
    !s.is_empty()
        && !s.contains(['/', '\\', ':'])
        && !s.contains("..")
        && !s.chars().any(|c| c.is_control())
}

fn research_hash(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn report_directory(cache: &Path, dir: &str) -> Result<PathBuf> {
    let path = if dir == RESEARCH_DIR {
        cache.join("discovery").join("research")
    } else {
        ensure!(
            dir.starts_with("auto_loop") && plain_segment(dir),
            "invalid report directory"
        );
        cache.join(dir)
    };
    let canonical_cache = std::fs::canonicalize(cache)?;
    let canonical_path = std::fs::canonicalize(&path)?;
    ensure!(
        canonical_path.starts_with(&canonical_cache),
        "report directory escaped configured cache"
    );
    ensure!(canonical_path.is_dir(), "report path is not a directory");
    Ok(canonical_path)
}

fn validate_selected_genes(selected: &[Gene], recorded: &[Gene]) -> Result<()> {
    ensure!(!selected.is_empty(), "selected portfolio is empty");
    let mut ids = HashSet::new();
    for gene in selected {
        ensure!(
            !gene.strategy_id.trim().is_empty() && ids.insert(gene.strategy_id.as_str()),
            "selected strategy ID is empty or ambiguous"
        );
        let mut matching = recorded
            .iter()
            .filter(|row| row.strategy_id == gene.strategy_id);
        let observed = matching
            .next()
            .ok_or_else(|| anyhow!("selected strategy is absent from research result"))?;
        ensure!(
            matching.next().is_none(),
            "selected strategy ID is ambiguous in research result"
        );
        ValidationStrategyIdentityV2::from_gene(gene)?.validate_against(observed)?;
    }
    Ok(())
}

struct BoundDiagnostic<'a> {
    gene: &'a Gene,
    trades: &'a [ReportTrade],
    quality: &'a ReportQuality,
}

fn bind_diagnostics<'a>(
    selected: &'a [Gene],
    logs: &'a [ReportTrades],
    quality: &'a [ReportQuality],
) -> Result<BoundDiagnostic<'a>> {
    ensure!(!selected.is_empty(), "selected portfolio is empty");
    let mut ids = HashSet::new();
    let mut representative: Option<BoundDiagnostic<'a>> = None;
    for gene in selected {
        ensure!(
            !gene.strategy_id.trim().is_empty() && ids.insert(gene.strategy_id.as_str()),
            "selected strategy ID is empty or ambiguous"
        );
        let mut rows = logs
            .iter()
            .filter(|row| row.strategy_id == gene.strategy_id);
        let row = rows.next().ok_or_else(|| {
            anyhow!(
                "selected strategy '{}' has no diagnostic trades",
                gene.strategy_id
            )
        })?;
        ensure!(
            rows.next().is_none(),
            "selected strategy '{}' has ambiguous diagnostic trade rows",
            gene.strategy_id
        );
        let mut metrics = quality
            .iter()
            .filter(|row| row.strategy_id == gene.strategy_id);
        let metric = metrics.next().ok_or_else(|| {
            anyhow!(
                "selected strategy '{}' has no diagnostic quality row",
                gene.strategy_id
            )
        })?;
        ensure!(
            metrics.next().is_none(),
            "selected strategy '{}' has ambiguous quality rows",
            gene.strategy_id
        );
        validate_trade_units(&row.trades, metric)?;
        // One representative, never a synthetic portfolio sum. Ties retain
        // actual selected-portfolio order, not filesystem/sidecar order.
        if representative
            .as_ref()
            .is_none_or(|best| row.trades.len() > best.trades.len())
        {
            representative = Some(BoundDiagnostic {
                gene,
                trades: &row.trades,
                quality: metric,
            });
        }
    }
    representative.ok_or_else(|| anyhow!("no selected diagnostic strategy"))
}

fn approximately_equal(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

fn validate_trade_units(trades: &[ReportTrade], quality: &ReportQuality) -> Result<()> {
    ensure!(
        !trades.is_empty() && trades.len() == quality.total_trades,
        "diagnostic trade count is missing or differs from the same strategy's quality row"
    );
    ensure!(
        quality.initial_capital.is_finite() && quality.initial_capital > 0.0,
        "diagnostic initial capital is unavailable"
    );
    let mut pnl = 0.0;
    for trade in trades {
        let exit = trade
            .exit_time
            .ok_or_else(|| anyhow!("diagnostic trade has no closed exit timestamp"))?;
        ensure!(
            trade.entry_time > 0 && exit >= trade.entry_time,
            "diagnostic trade timestamps are invalid"
        );
        let pct = trade
            .pnl_pct
            .ok_or_else(|| anyhow!("diagnostic trade has no initial-capital return fraction"))?;
        ensure!(
            approximately_equal(pct, trade.pnl / quality.initial_capital),
            "diagnostic pnl_pct does not equal pnl / initial_capital"
        );
        pnl += trade.pnl;
        ensure!(pnl.is_finite(), "diagnostic PnL overflow");
    }
    ensure!(
        approximately_equal(pnl, quality.net_profit)
            && approximately_equal(pnl / quality.initial_capital, quality.total_return_pct),
        "diagnostic trade PnL disagrees with the same strategy's quality row"
    );
    Ok(())
}

fn month_of(ms: i64) -> String {
    let z = ms.div_euclid(86_400_000) + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}")
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

struct DiagnosticCurve {
    monthly: Vec<MonthRow>,
    yearly: Vec<MonthRow>,
    start_ms: i64,
    end_ms: i64,
    equity: f64,
    max_dd: f64,
}

fn diagnostic_curve(trades: &[ReportTrade]) -> Result<DiagnosticCurve> {
    ensure!(!trades.is_empty(), "diagnostic trade log is empty");
    let mut rows = trades.iter().collect::<Vec<_>>();
    rows.sort_by_key(|row| (row.exit_time, row.entry_time));
    let start_ms = rows.iter().map(|row| row.entry_time).min().unwrap();
    let end_ms = rows
        .last()
        .unwrap()
        .exit_time
        .ok_or_else(|| anyhow!("missing exit timestamp"))?;
    let mut equity = SEED;
    let mut peak = SEED;
    let mut max_dd = 0.0f64;
    let mut monthly: Vec<MonthRow> = Vec::new();
    let mut current = String::new();
    let mut month_open = SEED;
    let mut count = 0;
    for row in rows {
        let exit = row
            .exit_time
            .ok_or_else(|| anyhow!("missing exit timestamp"))?;
        ensure!(
            row.entry_time > 0 && exit >= row.entry_time,
            "invalid trade timestamps"
        );
        let month = month_of(exit);
        if month != current {
            if !current.is_empty() {
                monthly.push(MonthRow {
                    month: current,
                    balance: equity,
                    return_pct: (equity / month_open - 1.0) * 100.0,
                    trades: count,
                });
            }
            current = month;
            month_open = equity;
            ensure!(
                month_open > 0.0,
                "monthly return after account exhaustion is undefined"
            );
            count = 0;
        }
        // record_closed_cpu_trade stores pnl/INITIAL balance. Summing those
        // normalized cash amounts preserves equity; multiplying invents reinvestment.
        let fraction = row
            .pnl_pct
            .ok_or_else(|| anyhow!("missing initial-capital return fraction"))?;
        ensure!(fraction.is_finite(), "non-finite diagnostic return");
        equity += SEED * fraction;
        ensure!(
            equity.is_finite() && equity >= 0.0,
            "diagnostic equity is invalid"
        );
        peak = peak.max(equity);
        max_dd = max_dd.max((peak - equity) / peak);
        count += 1;
    }
    monthly.push(MonthRow {
        month: current,
        balance: equity,
        return_pct: (equity / month_open - 1.0) * 100.0,
        trades: count,
    });
    let mut yearly: Vec<MonthRow> = Vec::new();
    let mut year_open = SEED;
    let mut prior_balance = SEED;
    for month in &monthly {
        let year = month
            .month
            .split_once('-')
            .ok_or_else(|| anyhow!("invalid month"))?
            .0;
        if yearly.last().is_none_or(|row| row.month != year) {
            year_open = prior_balance;
            yearly.push(MonthRow {
                month: year.to_owned(),
                balance: month.balance,
                return_pct: 0.0,
                trades: 0,
            });
        }
        ensure!(
            year_open > 0.0,
            "yearly return after account exhaustion is undefined"
        );
        let annual = yearly.last_mut().unwrap();
        annual.balance = month.balance;
        annual.return_pct = (month.balance / year_open - 1.0) * 100.0;
        annual.trades += month.trades;
        prior_balance = month.balance;
    }
    for row in monthly.iter_mut().chain(&mut yearly) {
        ensure!(
            row.return_pct.is_finite() && (row.balance * 100.0).is_finite(),
            "diagnostic display overflow"
        );
        row.balance = round2(row.balance);
        row.return_pct = round2(row.return_pct);
    }
    Ok(DiagnosticCurve {
        monthly,
        yearly,
        start_ms,
        end_ms,
        equity,
        max_dd,
    })
}

fn canonical_recorded_mode(mode: &str) -> &'static str {
    match mode {
        "Risky" | "risky" => "risky",
        "PropFirm" | "prop_firm" => "prop_firm",
        "Strict" | "strict" => "strict",
        _ => "unknown",
    }
}

fn recorded_evaluation(policy: &LiveTradingPolicyV1) -> Result<Option<RecordedEvaluation>> {
    if policy.schema_version == 1 {
        policy.validate()?;
        return Ok(None);
    }
    // This accessor validates and reopens the archived evaluator. No current
    // Settings, mode defaults, or diagnostic return denominator can fill gaps.
    let evaluation = policy.sealed_evaluation_config()?;
    Ok(Some(RecordedEvaluation {
        policy_identity_hash: policy.identity_hash.clone(),
        account_currency: evaluation.account_currency,
        initial_capital: evaluation.initial_equity,
        risk_per_trade_min: evaluation.risk_per_trade_min,
        risk_per_trade_max: evaluation.risk_per_trade_max,
        high_quality_confidence: evaluation.high_quality_confidence,
        // Search sizes from the threshold margin / long-short threshold gap,
        // clamped to [0, 1]. This is not a calibrated probability of winning.
        confidence_basis: "signal_threshold_margin",
        growth_goal: evaluation.growth_goal.map(RecordedGrowthGoal::from),
    }))
}

fn load_inputs(
    dir: &Path,
    base: &str,
) -> Result<(
    LivePortfolioArtifact,
    Vec<ReportTrades>,
    Vec<ReportQuality>,
    String,
)> {
    if let Some(hash) = base.strip_suffix(".research") {
        ensure!(research_hash(hash), "invalid research artifact name");
        let artifact = load_report_portfolio(&dir.join(format!("{base}.live_portfolio.json")))?;
        let research: ResearchProjection =
            read_json(&dir.join(format!("{base}.json"))).map_err(|error| {
                diagnostic_input_error(
                    error,
                    "saved research diagnostic data are missing or invalid",
                )
            })?;
        ensure!(
            research.schema_version == 3 && research.evidence_identity_sha256 == hash,
            "research diagnostic version/name mismatch"
        );
        ensure!(research.artifact_class == neoethos_search::historical_research::HistoricalResearchArtifactClassV1::ResearchOnly
            && research.promotion_eligibility == neoethos_search::historical_research::HistoricalResearchPromotionEligibilityV1::NotPromotionEligible,
            "expected research-only diagnostics");
        research
            .execution_contract
            .validate_against_receipt(artifact.search_scope.receipt())?;
        let result = research.discovery_result;
        ensure!(
            result.selection_scope == artifact.search_scope
                && result.search_config_hash == artifact.search_config_hash
                && result.effective_feature_names == artifact.effective_feature_names,
            "research diagnostic scope/config/features differ from selected portfolio"
        );
        validate_selected_genes(&artifact.genes, &result.portfolio)?;
        let mode = result
            .funnel_profile
            .map(|profile| canonical_recorded_mode(&profile.mode))
            .unwrap_or("unknown")
            .to_owned();
        return Ok((artifact, result.logged_trades, result.quality_metrics, mode));
    }
    // Legacy sidecars require exactly one validated adjacent portfolio.
    let paths = [
        dir.join(format!("{base}.live_portfolio.json")),
        dir.join(format!("{base}.json.live_portfolio.json")),
    ];
    let present = paths
        .iter()
        .filter(|path| path.exists())
        .collect::<Vec<_>>();
    if present.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "legacy diagnostic portfolio file is missing",
        )
        .into());
    }
    ensure!(
        present.len() == 1,
        "legacy diagnostic portfolio is ambiguous"
    );
    let artifact = load_report_portfolio(present[0])?;
    let trades = read_json(&dir.join(format!("{base}.json.trades.json"))).map_err(|error| {
        diagnostic_input_error(error, "saved diagnostic trade data are missing or invalid")
    })?;
    let quality = read_json(&dir.join(format!("{base}.json.quality.json"))).map_err(|error| {
        diagnostic_input_error(
            error,
            "saved diagnostic quality data are missing or invalid",
        )
    })?;
    Ok((artifact, trades, quality, "unknown".to_owned()))
}

fn build(cache: &Path, dir_key: &str, base: &str, with_monthly: bool) -> Result<StrategyReportDto> {
    ensure!(plain_segment(base), "invalid report basename");
    let dir = report_directory(cache, dir_key)?;
    let (artifact, trades, quality, mode) = load_inputs(&dir, base)?;
    let bound =
        bind_diagnostics(&artifact.genes, &trades, &quality).map_err(diagnostic_unavailable)?;
    let window = artifact.search_scope.evaluated_window();
    ensure!(
        bound
            .trades
            .iter()
            .all(|trade| trade.entry_time >= window.timestamp_start_ms()
                && trade
                    .exit_time
                    .is_some_and(|exit| exit <= window.timestamp_end_ms())),
        DiagnosticUnavailable("diagnostic trades fall outside the selected IS scope".to_owned())
    );
    let curve = diagnostic_curve(bound.trades).map_err(diagnostic_unavailable)?;
    let years = (curve.end_ms - curve.start_ms) as f64 / 86_400_000.0 / 365.25;
    let cagr = (years > 0.0)
        .then(|| ((curve.equity / SEED).powf(1.0 / years) - 1.0) * 100.0)
        .filter(|value| value.is_finite() && (value * 100.0).is_finite());
    let mut flags = vec!["Unsealed IS diagnostics for ONE selected strategy, not portfolio PnL, independent OOS evidence, or promotion authority".to_owned()];
    if bound.trades.len() < 100 {
        flags.push(format!("low sample ({} trades)", bound.trades.len()));
    }
    if cagr.is_none_or(|value| value.abs() > 1000.0) {
        flags.push("annualized diagnostic return is unavailable or extreme; inspect the sample span and units".to_owned());
    }
    let finite = |value: Option<f64>| value.filter(|number| number.is_finite());
    let head = StrategyEntry {
        mode,
        recorded_evaluation: recorded_evaluation(&artifact.live_trading_policy)?,
        diagnostic_initial_capital: bound.quality.initial_capital,
        dir: dir_key.to_owned(),
        symbol: artifact.symbol.clone(),
        timeframe: artifact.base_tf.clone(),
        base: base.to_owned(),
        strategy_id: bound.gene.strategy_id.clone(),
        exact_gene_hash: ValidationStrategyIdentityV2::from_gene(bound.gene)?
            .exact_gene_hash()
            .to_owned(),
        trades: bound.trades.len(),
        win_rate: finite(bound.quality.win_rate),
        profit_factor: finite(bound.quality.profit_factor),
        sharpe: finite(bound.quality.sharpe_ratio),
        cpcv_passed: None,
        walkforward_passed: None,
        validation_complete: None,
        span_start: Some(month_of(curve.start_ms)),
        span_end: Some(month_of(curve.end_ms)),
        years: round2(years),
        cagr_pct: cagr.map(round2),
        final_from_1000: round2(curve.equity),
        max_dd_pct: round2(curve.max_dd * 100.0),
        flags,
        discovered_at_ms: file_modified_ms(&dir.join(if base.ends_with(".research") {
            format!("{base}.json")
        } else {
            format!("{base}.json.trades.json")
        })),
    };
    Ok(StrategyReportDto {
        head,
        monthly: if with_monthly {
            curve.monthly
        } else {
            Vec::new()
        },
        yearly: if with_monthly {
            curve.yearly
        } else {
            Vec::new()
        },
    })
}

fn inventory(cache: &Path) -> (Vec<(String, String)>, Vec<String>) {
    let mut entries = Vec::new();
    let mut unavailable = Vec::new();
    let mut directories = vec![RESEARCH_DIR.to_owned()];
    if let Ok(list) = std::fs::read_dir(cache) {
        for (index, entry) in list.enumerate() {
            if index >= MAX_SCAN_ENTRIES {
                unavailable
                    .push("report root inventory truncated at its bounded scan limit".to_owned());
                break;
            }
            let Ok(entry) = entry else {
                continue;
            };
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("auto_loop")
                && plain_segment(&name)
                && entry.file_type().is_ok_and(|kind| kind.is_dir())
            {
                directories.push(name);
            }
        }
    }
    let mut scanned = 0;
    for key in directories {
        let Ok(dir) = report_directory(cache, &key) else {
            continue;
        };
        let Ok(files) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in files.flatten() {
            scanned += 1;
            if scanned > MAX_SCAN_ENTRIES {
                unavailable.push("report inventory truncated at its bounded scan limit".to_owned());
                return (entries, unavailable);
            }
            if !entry.file_type().is_ok_and(|kind| kind.is_file()) {
                continue;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            let base = if key == RESEARCH_DIR {
                name.strip_suffix(".research.json")
                    .map(|hash| format!("{hash}.research"))
            } else {
                name.strip_suffix(".json.trades.json").map(str::to_owned)
            };
            if let Some(base) = base {
                entries.push((key.clone(), base));
            }
        }
    }
    entries.sort();
    (entries, unavailable)
}

pub async fn list(State(state): State<AppApiState>) -> Json<StrategyListDto> {
    let cache = match cache_dir(state.config_path()) {
        Ok(path) => path,
        Err(error) => {
            return Json(StrategyListDto {
                count: 0,
                strategies: Vec::new(),
                unavailable: vec![format!("configured report storage unavailable: {error}")],
            });
        }
    };
    match tokio::task::spawn_blocking(move || scan_reports(&cache)).await {
        Ok(report) => Json(report),
        Err(error) => Json(StrategyListDto {
            count: 0,
            strategies: Vec::new(),
            unavailable: vec![format!("report reader failed: {error}")],
        }),
    }
}

fn scan_reports(cache: &Path) -> StrategyListDto {
    let (entries, mut unavailable) = inventory(cache);
    let mut strategies = Vec::new();
    for (dir, base) in entries {
        match build(cache, &dir, &base, false) {
            Ok(report) => strategies.push(report.head),
            Err(error) => unavailable.push(format!("{dir}/{base}: {error:#}")),
        }
    }
    strategies.sort_by(|a, b| b.discovered_at_ms.cmp(&a.discovered_at_ms));
    StrategyListDto {
        count: strategies.len(),
        strategies,
        unavailable,
    }
}

#[derive(Debug, Deserialize)]
pub struct ReportQuery {
    pub dir: String,
    pub base: String,
    pub strategy_id: Option<String>,
    pub exact_gene_hash: Option<String>,
}

type ReportHttpError = (StatusCode, Json<serde_json::Value>);

fn report_error(status: StatusCode, message: &str, detail: &str) -> ReportHttpError {
    (
        status,
        Json(serde_json::json!({ "error": message, "detail": detail })),
    )
}

fn report_build_error(error: anyhow::Error) -> ReportHttpError {
    if let Some(reason) = error.downcast_ref::<DiagnosticUnavailable>() {
        return report_error(
            StatusCode::CONFLICT,
            "Saved strategy diagnostics are unavailable.",
            &reason.0,
        );
    }
    if let Some(kind) = error.chain().find_map(|cause| {
        cause
            .downcast_ref::<std::io::Error>()
            .map(std::io::Error::kind)
    }) {
        return if kind == std::io::ErrorKind::NotFound {
            report_error(
                StatusCode::NOT_FOUND,
                "Saved strategy report was not found.",
                "The requested report or its selected portfolio file is missing.",
            )
        } else {
            report_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Saved strategy report could not be read.",
                "Report storage is unavailable.",
            )
        };
    }
    // Do not expose filesystem paths, parser input, or internal validation data.
    report_error(
        StatusCode::CONFLICT,
        "Saved strategy report is invalid.",
        "The saved archive failed strict identity or data validation; no report was generated.",
    )
}

pub async fn report(
    State(state): State<AppApiState>,
    Query(q): Query<ReportQuery>,
) -> Result<Json<StrategyReportDto>, ReportHttpError> {
    report_from_config(state.config_path(), q).await
}

async fn report_from_config(
    config_path: &Path,
    q: ReportQuery,
) -> Result<Json<StrategyReportDto>, ReportHttpError> {
    if !plain_segment(&q.base)
        || !(q.dir == RESEARCH_DIR || (q.dir.starts_with("auto_loop") && plain_segment(&q.dir)))
        || q.base
            .strip_suffix(".research")
            .is_some_and(|hash| !research_hash(hash))
    {
        return Err(report_error(
            StatusCode::BAD_REQUEST,
            "Invalid strategy report selector.",
            "Select a report basename and a supported report directory, not a path.",
        ));
    }
    let cache = cache_dir(config_path).map_err(|_| {
        report_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Strategy report storage is unavailable.",
            "The configured report storage could not be loaded.",
        )
    })?;
    let report = tokio::task::spawn_blocking(move || build(&cache, &q.dir, &q.base, true))
        .await
        .map_err(|_| {
            report_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Strategy report reader failed.",
                "No diagnostic report was generated.",
            )
        })?
        .map_err(report_build_error)?;
    if q.strategy_id
        .as_ref()
        .is_some_and(|id| id != &report.head.strategy_id)
        || q.exact_gene_hash
            .as_ref()
            .is_some_and(|hash| hash != &report.head.exact_gene_hash)
    {
        return Err(report_error(
            StatusCode::CONFLICT,
            "Selected strategy identity differs from the saved report.",
            "Refresh the report selection; no substitute strategy was returned.",
        ));
    }
    Ok(Json(report))
}

#[cfg(test)]
#[path = "strategy_report_tests.rs"]
mod tests;
