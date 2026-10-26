//! One operator entrypoint for the existing receipt-bound CPU research lane.
//! Acquisition receipts own source selection; this command issues no trading permit.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use neoethos_broker_history::canonical_research_costs::{
    ensure_unique_selected_timeframe, ensure_unique_series, read_bounded_regular_file,
    read_regular_file_with_limit, validate_exact_file_settings,
};
use neoethos_broker_history::{
    CanonicalTrendbarAcquisitionStoreV1, CanonicalTrendbarMatrixReceiptV1,
    CanonicalTrendbarPlanReceiptV1,
};
use neoethos_core::Settings;
use neoethos_data::CanonicalTimeframe;

const REQUIRED: [&str; 10] = [
    "--authority-root",
    "--plan-sha256",
    "--matrix-sha256",
    "--data-root",
    "--broker-symbol-contract",
    "--broker-account-snapshot",
    "--settings-source",
    "--symbol",
    "--base-timeframe",
    "--out-dir",
];
const OPTIONAL: [&str; 22] = [
    "--higher",
    "--population",
    "--generations",
    "--max-indicators",
    "--candidates",
    "--max-batches",
    "--cpu-threads",
    "--quote-replay-bundle",
    "--quote-replay-sha256",
    "--quote-replay-store",
    "--resume-research",
    "--resume-research-sha256",
    "--research-oos-eval",
    "--diagnose-candidate-costs-out",
    "--diagnose-cost-variants",
    "--diagnose-candidate-signals-out",
    "--diagnose-candidate-weekdays-out",
    "--diagnose-candidate-selectors",
    "--diagnose-exit-policy-out",
    "--new-search-research",
    "--new-search-research-sha256",
    "--prefilter-ranking-audit-out",
];

const RESUME_FLAGS: [&str; 2] = ["--resume-research", "--resume-research-sha256"];
const NEW_SEARCH_FLAGS: [&str; 2] = ["--new-search-research", "--new-search-research-sha256"];

/// The B9 opt-in fitness selector (docs/autoresearch-loop.md section 14.7)
/// exists only when the same named cargo feature is compiled in. Without it,
/// --fitness-table stays an unknown argument, so the shipped CLI's fitness
/// dispatch is byte-identical to the pre-selector build.
#[cfg(not(feature = "search-monthly-upside-objective"))]
fn is_optional_flag(flag: &str) -> bool {
    OPTIONAL.contains(&flag)
}

#[cfg(feature = "search-monthly-upside-objective")]
fn is_optional_flag(flag: &str) -> bool {
    OPTIONAL.contains(&flag) || flag == "--fitness-table"
}

/// Read optional all-or-none inputs on both the strict operator command and
/// the lower-level discover entrypoint. Neither accepts a duplicate binding.
fn optional_inputs<'a>(
    args: &'a [String],
    flags: &[&'a str],
) -> Result<BTreeMap<&'a str, &'a str>> {
    let mut values = BTreeMap::new();
    for &flag in flags {
        for (i, _) in args
            .iter()
            .enumerate()
            .filter(|(_, arg)| arg.as_str() == flag)
        {
            let value = args
                .get(i + 1)
                .filter(|v| !v.trim().is_empty() && !v.starts_with("--"))
                .with_context(|| format!("{flag} requires a value"))?;
            ensure!(
                values.insert(flag, value.as_str()).is_none(),
                "{flag} must occur exactly once"
            );
        }
    }
    ensure!(
        values.is_empty() || values.len() == flags.len(),
        "{} must be supplied together",
        flags.join(", ")
    );
    Ok(values)
}

fn read_pinned_input(
    source: &Path,
    expected: &str,
    label: &str,
    max_bytes: u64,
) -> Result<(Vec<u8>, String)> {
    // Complete envelopes are larger than the 64 KiB scalar cost file. The
    // caller supplies its bounded format limit; no unbounded read is allowed.
    let bytes = read_regular_file_with_limit(source, max_bytes)?;
    use sha2::{Digest, Sha256};
    let sha256 = format!("{:x}", Sha256::digest(&bytes));
    ensure!(sha256 == expected, "{label} SHA256 mismatch");
    Ok((bytes, sha256))
}

fn read_recorded_research(
    args: &[String],
    flags: &[&str; 2],
    label: &str,
) -> Result<
    Option<(
        neoethos_search::CanonicalTrendbarResearchDiscoveryResultV3,
        String,
        PathBuf,
    )>,
> {
    let values = optional_inputs(args, flags)?;
    if values.is_empty() {
        return Ok(None);
    }
    let source = PathBuf::from(values[flags[0]]);
    let (bytes, sha256) = read_pinned_input(&source, values[flags[1]], label, 512 * 1024 * 1024)?;
    let research =
        neoethos_search::CanonicalTrendbarResearchDiscoveryResultV3::from_json_bytes(&bytes)?;
    Ok(Some((research, sha256, source)))
}

pub(crate) struct SavedResearchInputs {
    pub research: Option<neoethos_search::CanonicalTrendbarResearchDiscoveryResultV3>,
    pub sha256: String,
    pub source: PathBuf,
}

impl SavedResearchInputs {
    pub fn load(args: &[String]) -> Result<Option<Self>> {
        let Some((research, sha256, source)) =
            read_recorded_research(args, &RESUME_FLAGS, "saved research")?
        else {
            return Ok(None);
        };
        // The research-only saved-candidate cost diagnostic intentionally reads
        // zero-portfolio archives: it never resumes a frozen finalist and never
        // needs a portfolio to exist. Every other continuation still requires
        // one, because it would otherwise restart a finalist replay over nothing.
        let research_only_diagnostic = args.iter().any(|arg| {
            arg == "--diagnose-candidate-costs-out"
                || arg == "--diagnose-candidate-signals-out"
                || arg == "--diagnose-candidate-weekdays-out"
                || arg == "--diagnose-exit-policy-out"
        });
        if !research_only_diagnostic {
            neoethos_search::ensure_non_empty_portfolio(
                research.discovery_result(),
                "resume frozen research",
            )?;
        }
        Ok(Some(Self {
            research: Some(research),
            sha256,
            source,
        }))
    }
}

/// A recorded research envelope used ONLY as upstream INPUT provenance for a
/// fresh, hash-pinned NEW search. It may be zero-portfolio; its genomes,
/// fitness and portfolio are never returned or replayed.
pub(crate) struct NewSearchInputs {
    pub research: Option<neoethos_search::CanonicalTrendbarResearchDiscoveryResultV3>,
    pub sha256: String,
    pub source: PathBuf,
}

impl NewSearchInputs {
    pub fn load(args: &[String]) -> Result<Option<Self>> {
        let Some((research, sha256, source)) =
            read_recorded_research(args, &NEW_SEARCH_FLAGS, "new-search recorded research")?
        else {
            return Ok(None);
        };
        Ok(Some(Self {
            research: Some(research),
            sha256,
            source,
        }))
    }
}

const QUOTE_FLAGS: [&str; 3] = [
    "--quote-replay-bundle",
    "--quote-replay-sha256",
    "--quote-replay-store",
];

/// An explicit, hash-pinned continuation of one locked research batch.
/// The stored input contains no permission to trade and supplies no defaults.
pub(crate) struct QuoteReplayInputs {
    input: Option<neoethos_broker_truth_acquire::StoredFinalistQuoteReplayV1>,
    store: neoethos_broker_truth::BrokerTruthAcquisitionStoreV1,
    pub sha256: String,
    pub source: PathBuf,
}

impl QuoteReplayInputs {
    pub fn load(args: &[String]) -> Result<Option<Self>> {
        let values = optional_inputs(args, &QUOTE_FLAGS)?;
        if values.is_empty() {
            return Ok(None);
        }
        ensure!(
            values.len() == QUOTE_FLAGS.len(),
            "quote replay requires bundle, SHA256 and store together"
        );
        let (bytes, sha256) = read_pinned_input(
            Path::new(values[QUOTE_FLAGS[0]]),
            values[QUOTE_FLAGS[1]],
            "quote replay bundle",
            64 * 1024 * 1024,
        )?;
        let input =
            neoethos_broker_truth_acquire::StoredFinalistQuoteReplayV1::from_json_bytes(&bytes)?;
        let store =
            neoethos_broker_truth::BrokerTruthAcquisitionStoreV1::new(values[QUOTE_FLAGS[2]]);
        input
            .capture
            .validate_stored_receipts(&store)
            .context("reopen the saved quote replay acquisition link before feature work")?;
        Ok(Some(Self {
            input: Some(input),
            store,
            sha256,
            source: PathBuf::from(values[QUOTE_FLAGS[0]]),
        }))
    }

    pub fn replay(
        &mut self,
        cpu: &neoethos_core::execution::BudgetedCpuScope<'_>,
        locked: &neoethos_search::LockedCanonicalSignalPlanV3<'_>,
    ) -> Result<neoethos_search::LockedPortfolioOuterHoldoutReplaySetV3> {
        self.input
            .take()
            .context("quote replay bundle was already consumed")?
            .replay(cpu, &self.store, locked)
            .context("replay the exact stored quote and financial evidence")
    }
}

struct Arguments(BTreeMap<String, String>);

impl Arguments {
    fn parse(args: &[String]) -> Result<Self> {
        ensure!(
            args.len() % 2 == 0,
            "canonical-research requires flag/value pairs"
        );
        let mut values = BTreeMap::new();
        for pair in args.chunks_exact(2) {
            let flag = pair[0].as_str();
            ensure!(
                REQUIRED.contains(&flag) || is_optional_flag(flag),
                "unknown canonical-research argument {flag}"
            );
            ensure!(
                !pair[1].trim().is_empty() && !pair[1].starts_with("--"),
                "{flag} requires a value"
            );
            ensure!(
                values.insert(pair[0].clone(), pair[1].clone()).is_none(),
                "{flag} must be supplied exactly once"
            );
        }
        for flag in REQUIRED {
            ensure!(
                values.contains_key(flag),
                "canonical-research requires {flag}"
            );
        }
        let parsed = Self(values);
        let quote_flags = QUOTE_FLAGS
            .iter()
            .filter(|flag| parsed.0.contains_key(**flag))
            .count();
        ensure!(
            quote_flags == 0 || quote_flags == QUOTE_FLAGS.len(),
            "quote replay requires bundle, SHA256 and store together"
        );
        let resume_flags = RESUME_FLAGS
            .iter()
            .filter(|flag| parsed.0.contains_key(**flag))
            .count();
        ensure!(
            resume_flags == 0 || resume_flags == RESUME_FLAGS.len(),
            "resume requires the saved research and its SHA256 together"
        );
        let new_search_flags = NEW_SEARCH_FLAGS
            .iter()
            .filter(|flag| parsed.0.contains_key(**flag))
            .count();
        ensure!(
            new_search_flags == 0 || new_search_flags == NEW_SEARCH_FLAGS.len(),
            "new-search recorded input requires the research path and its SHA256 together"
        );
        ensure!(
            new_search_flags == 0 || resume_flags == 0,
            "--new-search-research and --resume-research are mutually exclusive modes"
        );
        if new_search_flags != 0 {
            ensure!(
                quote_flags == 0,
                "new-search recorded inputs cannot be combined with quote replay"
            );
            ensure!(
                parsed.0.get("--diagnose-candidate-costs-out").is_none()
                    && parsed.0.get("--diagnose-cost-variants").is_none()
                    && parsed.0.get("--diagnose-candidate-signals-out").is_none()
                    && parsed.0.get("--diagnose-candidate-weekdays-out").is_none()
                    && parsed.0.get("--diagnose-exit-policy-out").is_none()
                    && parsed.0.get("--diagnose-candidate-selectors").is_none(),
                "new-search recorded inputs cannot be combined with candidate diagnostics"
            );
        }
        #[cfg(feature = "search-monthly-upside-objective")]
        if let Some(name) = parsed.0.get("--fitness-table") {
            let normalized = name.trim().to_ascii_lowercase();
            ensure!(
                normalized == "ga_fitness"
                    || normalized == "monthly_upside_v1"
                    || normalized == "monthly_upside_dd_averse_v1"
                    || normalized == "monthly_upside_dd_budget_v1",
                "--fitness-table must be ga_fitness, monthly_upside_v1, monthly_upside_dd_averse_v1 or monthly_upside_dd_budget_v1"
            );
            ensure!(
                resume_flags == 0 && quote_flags == 0,
                "--fitness-table is a fresh-search selector and cannot be combined with a saved continuation or quote replay"
            );
        }
        let diagnose_flags = ["--diagnose-candidate-costs-out", "--diagnose-cost-variants"]
            .iter()
            .filter(|flag| parsed.0.contains_key(**flag))
            .count();
        ensure!(
            diagnose_flags == 0 || diagnose_flags == 2,
            "candidate cost diagnosis requires --diagnose-candidate-costs-out and --diagnose-cost-variants together"
        );
        if diagnose_flags != 0 {
            ensure!(
                resume_flags != 0,
                "candidate cost diagnosis requires an exact saved research (--resume-research + --resume-research-sha256)"
            );
            parse_diagnostic_cost_variants(parsed.value("--diagnose-cost-variants"))?;
        }
        let signal_behavior_out = parsed.0.get("--diagnose-candidate-signals-out");
        if signal_behavior_out.is_some() {
            ensure!(
                resume_flags != 0,
                "candidate signal behavior census requires an exact saved research (--resume-research + --resume-research-sha256)"
            );
            ensure!(
                diagnose_flags == 0,
                "candidate signal behavior census cannot be combined with candidate cost diagnosis"
            );
            ensure!(
                new_search_flags == 0 && quote_flags == 0,
                "candidate signal behavior census cannot be combined with fresh-search or quote replay"
            );
            ensure!(
                parsed.0.get("--diagnose-candidate-weekdays-out").is_none()
                    && parsed.0.get("--diagnose-exit-policy-out").is_none()
                    && parsed.0.get("--diagnose-candidate-selectors").is_none(),
                "candidate signal behavior census cannot be combined with candidate selector diagnostics"
            );
        }
        let weekday_out = parsed.0.get("--diagnose-candidate-weekdays-out");
        let exit_out = parsed.0.get("--diagnose-exit-policy-out");
        let selector_out = weekday_out.or(exit_out);
        let weekday_selectors = parsed.0.get("--diagnose-candidate-selectors");
        ensure!(
            selector_out.is_some() == weekday_selectors.is_some(),
            "candidate selector diagnostics require an output flag and --diagnose-candidate-selectors together"
        );
        ensure!(
            !(weekday_out.is_some() && exit_out.is_some()),
            "candidate weekday and exit-policy diagnostics cannot be combined"
        );
        if selector_out.is_some() {
            ensure!(
                resume_flags != 0,
                "candidate selector diagnostics require an exact saved research (--resume-research + --resume-research-sha256)"
            );
            ensure!(
                diagnose_flags == 0,
                "candidate selector diagnostics cannot be combined with candidate cost diagnosis"
            );
            ensure!(
                parsed.0.get("--diagnose-candidate-signals-out").is_none(),
                "candidate selector diagnostics cannot be combined with signal behavior census"
            );
            ensure!(
                new_search_flags == 0 && quote_flags == 0,
                "candidate selector diagnostics cannot be combined with fresh-search or quote replay"
            );
            parse_saved_candidate_selections(parsed.value("--diagnose-candidate-selectors"))?;
        }
        if resume_flags != 0 {
            for flag in [
                "--population",
                "--generations",
                "--max-indicators",
                "--candidates",
            ] {
                ensure!(
                    !parsed.0.contains_key(flag),
                    "{flag} cannot change an already selected research result"
                );
            }
        }
        if quote_flags != 0 || resume_flags != 0 || new_search_flags != 0 {
            ensure!(
                parsed.positive("--max-batches", 2)? == 1,
                "a saved/new-search continuation binds one exact batch; supply --max-batches 1"
            );
        }
        for flag in [
            "--population",
            "--generations",
            "--max-indicators",
            "--candidates",
            "--max-batches",
            "--cpu-threads",
        ] {
            if parsed.0.contains_key(flag) {
                parsed.positive(flag, 1)?;
            }
        }
        parsed
            .value("--base-timeframe")
            .parse::<CanonicalTimeframe>()
            .map_err(anyhow::Error::msg)?;
        Ok(parsed)
    }

    fn value(&self, flag: &str) -> &str {
        &self.0[flag]
    }

    fn positive(&self, flag: &str, default: usize) -> Result<usize> {
        let Some(value) = self.0.get(flag) else {
            return Ok(default);
        };
        let value = value
            .parse::<usize>()
            .with_context(|| format!("{flag} requires a positive integer"))?;
        ensure!(value > 0, "{flag} requires a positive integer");
        Ok(value)
    }
}

/// One explicit cost envelope for the research-only saved-candidate cost
/// diagnostic CLI mode: `label=round_trip_pips:round_trip_commission_per_lot`.
/// The list is comma separated and explicit on purpose: no cost number is
/// invented by the CLI, and the caller must state every envelope it compares.
pub fn parse_diagnostic_cost_variants(spec: &str) -> Result<Vec<(String, f64, f64)>> {
    let mut variants = Vec::new();
    for item in spec.split(',') {
        let item = item.trim();
        ensure!(
            !item.is_empty(),
            "--diagnose-cost-variants contains an empty item"
        );
        let (label, numbers) = item.split_once('=').with_context(|| {
            format!("diagnostic cost variant '{item}' must be label=pips:commission")
        })?;
        let label = label.trim();
        ensure!(
            !label.is_empty()
                && label
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
            "diagnostic cost variant label '{label}' must be non-empty ascii alphanumeric, '-' or '_'"
        );
        let (pips, commission) = numbers.split_once(':').with_context(|| {
            format!("diagnostic cost variant '{item}' must be label=pips:commission")
        })?;
        let pips = pips.trim().parse::<f64>().with_context(|| {
            format!("diagnostic cost variant '{label}' round-trip pips is not a number")
        })?;
        let commission = commission.trim().parse::<f64>().with_context(|| {
            format!("diagnostic cost variant '{label}' round-trip commission is not a number")
        })?;
        ensure!(
            pips.is_finite() && pips >= 0.0,
            "diagnostic cost variant '{label}' round-trip pips must be finite and nonnegative"
        );
        ensure!(
            commission.is_finite() && commission >= 0.0,
            "diagnostic cost variant '{label}' round-trip commission must be finite and nonnegative"
        );
        ensure!(
            !variants.iter().any(|(existing, _, _)| existing == label),
            "diagnostic cost variant label '{label}' is duplicated"
        );
        variants.push((label.to_owned(), pips, commission));
    }
    ensure!(
        !variants.is_empty(),
        "--diagnose-cost-variants requires at least one variant"
    );
    Ok(variants)
}

pub fn parse_saved_candidate_selections(
    spec: &str,
) -> Result<Vec<neoethos_search::discovery::SavedCandidateSelectionV1>> {
    neoethos_search::discovery::parse_saved_candidate_selections_v1(spec)
}

/// Resolve the same explicit settings file before installing the process budget.
pub fn settings_path(args: &[String]) -> Result<PathBuf> {
    Ok(Arguments::parse(args)?.value("--settings-source").into())
}

fn forward_optional_mode_flags(args: &Arguments, discovery_args: &mut Vec<String>) {
    for flag in QUOTE_FLAGS
        .into_iter()
        .chain(RESUME_FLAGS)
        .chain(NEW_SEARCH_FLAGS)
    {
        if let Some(value) = args.0.get(flag) {
            push(discovery_args, flag, value);
        }
    }
    #[cfg(feature = "search-monthly-upside-objective")]
    if let Some(value) = args.0.get("--fitness-table") {
        push(discovery_args, "--fitness-table", value);
    }
}

fn forward_optional_diagnostic_flags(args: &Arguments, discovery_args: &mut Vec<String>) {
    for flag in [
        "--diagnose-candidate-costs-out",
        "--diagnose-cost-variants",
        "--diagnose-candidate-signals-out",
        "--diagnose-candidate-weekdays-out",
        "--diagnose-candidate-selectors",
        "--diagnose-exit-policy-out",
        "--prefilter-ranking-audit-out",
    ] {
        if let Some(value) = args.0.get(flag) {
            push(discovery_args, flag, value);
        }
    }
}

pub fn run(args: &[String], settings: &Settings) -> Result<()> {
    ensure!(
        !cfg!(feature = "gpu-nvidia"),
        "canonical-research is an explicit CPU research command; use a CPU build or the sealed native-research GPU route"
    );
    let mut args = Arguments::parse(args)?;
    // Reject incomplete/tampered replay packages before creating output files
    // or starting features. Discovery rechecks the pinned bytes when consumed.
    let input_args = args
        .0
        .iter()
        .flat_map(|(k, v)| [k.clone(), v.clone()])
        .collect::<Vec<_>>();
    SavedResearchInputs::load(&input_args)?;
    QuoteReplayInputs::load(&input_args)?;
    let source = Path::new(args.value("--settings-source"));
    let settings_bytes = read_bounded_regular_file(source)?;
    validate_exact_file_settings(settings, source, &settings_bytes)?;
    let data_root = Path::new(args.value("--data-root"));
    let store = CanonicalTrendbarAcquisitionStoreV1::new(args.value("--authority-root"));
    let plan = CanonicalTrendbarPlanReceiptV1::from_sha256(args.value("--plan-sha256").to_owned())?;
    let matrix_receipt =
        CanonicalTrendbarMatrixReceiptV1::from_sha256(args.value("--matrix-sha256").to_owned())?;
    let acquisition_plan = store.open_plan(&plan)?;
    let broker_environment =
        neoethos_broker_history::BrokerEnvironment::from_canonical(acquisition_plan.environment());
    neoethos_broker_history::account_snapshot_cli::validate_saved_account_currency(
        Path::new(args.value("--broker-account-snapshot")),
        broker_environment,
        acquisition_plan.account_id(),
        &settings.system.account_currency,
    )
    .context("verify actual broker account currency before research")?;
    let matrix = store.open_matrix(data_root, &plan, &matrix_receipt)?;
    let series = ensure_unique_series(&matrix, args.value("--symbol"))?;
    let base = args
        .value("--base-timeframe")
        .parse::<CanonicalTimeframe>()
        .map_err(anyhow::Error::msg)?;
    let selected = ensure_unique_selected_timeframe(series, base)?;
    let higher = args.0.get("--higher").map_or_else(
        || settings.system.resolve_higher_timeframes(&base.to_string()),
        |value| value.split(',').map(str::trim).map(str::to_owned).collect(),
    );
    for label in &higher {
        let timeframe = label
            .parse::<CanonicalTimeframe>()
            .map_err(anyhow::Error::msg)?;
        ensure!(
            timeframe > base,
            "higher timeframe {timeframe} must be above {base}"
        );
        ensure_unique_selected_timeframe(series, timeframe)?;
    }

    // Each run gets a new directory. Existing research, data and model trees
    // are never overwritten, including when a previous invocation failed.
    let output = PathBuf::from(args.value("--out-dir"));
    fs::create_dir(&output)
        .with_context(|| format!("create new research output directory {}", output.display()))?;
    let frozen_source = output.join("settings.yaml");
    fs::write(&frozen_source, &settings_bytes).context("preserve exact research settings bytes")?;
    let frozen_settings = Settings::from_yaml(&frozen_source)?;
    let frozen_account = output.join("broker-account");
    neoethos_broker_history::account_snapshot_cli::freeze_account_currency_evidence(
        Path::new(args.value("--broker-account-snapshot")),
        &frozen_account,
        broker_environment,
        acquisition_plan.account_id(),
        &frozen_settings.system.account_currency,
    )?;
    args.0.insert(
        "--broker-account-snapshot".to_owned(),
        frozen_account
            .join("account-snapshot.json")
            .to_string_lossy()
            .into_owned(),
    );
    ensure!(
        serde_json::to_value(&frozen_settings)? == serde_json::to_value(settings)?,
        "frozen settings must resolve to the same startup settings"
    );
    let broker_bytes =
        read_bounded_regular_file(Path::new(args.value("--broker-symbol-contract")))?;
    let frozen_broker = output.join("broker-symbol-contract.json");
    fs::write(&frozen_broker, broker_bytes).context("preserve exact broker symbol snapshot")?;
    args.0.insert(
        "--settings-source".to_owned(),
        frozen_source.to_string_lossy().into_owned(),
    );
    args.0.insert(
        "--broker-symbol-contract".to_owned(),
        frozen_broker.to_string_lossy().into_owned(),
    );
    neoethos_core::storage::json::write_json_atomic(output.join("run-arguments.json"), &args.0)?;
    let costs = output.join("screening-costs.json");
    let evidence = output.join("research.json");
    let mut cost_args = Vec::new();
    for flag in [
        "--authority-root",
        "--data-root",
        "--plan-sha256",
        "--matrix-sha256",
        "--symbol",
        "--broker-symbol-contract",
        "--settings-source",
    ] {
        push(&mut cost_args, flag, args.value(flag));
    }
    push(&mut cost_args, "--basis-timeframe", "D1");
    push(&mut cost_args, "--out", &costs.to_string_lossy());
    crate::canonical_full_run::build_cost_assumptions(&cost_args, &frozen_settings)
        .context("preflight exact broker, D1 costs and settings before feature work")?;

    let mut discovery_args = Vec::new();
    for (flag, value) in [
        ("--symbol", args.value("--symbol").to_owned()),
        ("--base", base.to_string()),
        ("--higher", higher.join(",")),
        (
            "--dataset-identity",
            selected.identity().to_path_component(),
        ),
        ("--root", args.value("--data-root").to_owned()),
        ("--config", args.value("--settings-source").to_owned()),
        ("--out", evidence.to_string_lossy().into_owned()),
        ("--population-auto", "false".to_owned()),
        (
            "--research-authority-root",
            args.value("--authority-root").to_owned(),
        ),
        (
            "--research-plan-sha256",
            args.value("--plan-sha256").to_owned(),
        ),
        (
            "--research-matrix-sha256",
            args.value("--matrix-sha256").to_owned(),
        ),
        (
            "--research-broker-symbol-contract",
            args.value("--broker-symbol-contract").to_owned(),
        ),
        (
            "--research-settings-source",
            args.value("--settings-source").to_owned(),
        ),
        (
            "--research-cost-assumptions",
            costs.to_string_lossy().into_owned(),
        ),
    ] {
        push(&mut discovery_args, flag, &value);
    }
    for (source_flag, target_flag, default) in [
        ("--population", "--population", 64),
        ("--generations", "--generations", 16),
        ("--max-indicators", "--max-indicators", 5),
        ("--candidates", "--candidates", 16),
        ("--max-batches", "--stream-max-batches", 2),
    ] {
        push(
            &mut discovery_args,
            target_flag,
            &args.positive(source_flag, default)?.to_string(),
        );
    }
    forward_optional_mode_flags(&args, &mut discovery_args);
    // Research-only outer-holdout evaluation: the CLI keeps the operator-facing
    // spelling, the discovery layer receives an explicit output path. This is
    // NOT a promotion artifact and never feeds back into Search.
    if let Some(path) = args.0.get("--research-oos-eval") {
        push(&mut discovery_args, "--research-oos-eval-out", path);
    }
    // Research-only saved-candidate cost diagnosis. It reuses the saved
    // research and its receipt-bound input; it never runs a new Search.
    forward_optional_diagnostic_flags(&args, &mut discovery_args);
    neoethos_core::storage::json::write_json_atomic(
        output.join("discovery-arguments.json"),
        &discovery_args,
    )?;
    println!("research_backend=CpuOnly");
    println!("research_scope=BoundedExploratory");
    println!("research_output={}", output.display());
    let result = crate::cmd_discover(&discovery_args);
    if let Err(error) = &result {
        let failure = serde_json::json!({
            "schema": "neoethos.canonical-cpu-research-failure.v1",
            "artifact_class": "ResearchOnly",
            "promotion_eligibility": "NotPromotionEligible",
            "authorization_issued": false,
            "status": "Failed",
            "error": format!("{error:#}"),
        });
        if let Err(save_error) =
            neoethos_core::storage::json::write_json_atomic(output.join("failure.json"), &failure)
        {
            return result.context(format!(
                "also failed to preserve research failure: {save_error:#}"
            ));
        }
    }
    result
}

fn push(args: &mut Vec<String>, flag: &str, value: &str) {
    args.extend([flag.to_owned(), value.to_owned()]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_arguments() -> Vec<String> {
        REQUIRED
            .into_iter()
            .flat_map(|flag| {
                [
                    flag.to_owned(),
                    if flag == "--base-timeframe" {
                        "H4"
                    } else {
                        "explicit"
                    }
                    .to_owned(),
                ]
            })
            .collect()
    }

    #[test]
    fn explicit_source_and_bounded_defaults_are_preserved() {
        let args = Arguments::parse(&valid_arguments()).unwrap();
        assert_eq!(args.positive("--population", 64).unwrap(), 64);
        assert_eq!(
            settings_path(&valid_arguments()).unwrap(),
            PathBuf::from("explicit")
        );
    }

    #[test]
    fn missing_duplicate_and_unknown_authority_arguments_are_refused() {
        let mut missing = valid_arguments();
        missing.drain(0..2);
        assert!(Arguments::parse(&missing).is_err());
        let mut duplicate = valid_arguments();
        push(&mut duplicate, "--symbol", "other");
        assert!(Arguments::parse(&duplicate).is_err());
        let mut unknown = valid_arguments();
        push(&mut unknown, "--promote", "true");
        assert!(Arguments::parse(&unknown).is_err());
    }

    #[test]
    fn malformed_budgets_cannot_silently_enable_unbounded_work() {
        for flag in [
            "--population",
            "--generations",
            "--max-batches",
            "--cpu-threads",
        ] {
            for value in ["0", "invalid", "-1"] {
                let mut args = valid_arguments();
                push(&mut args, flag, value);
                assert!(Arguments::parse(&args).is_err());
            }
        }
    }

    #[test]
    fn saved_continuations_require_one_batch_and_all_unique_inputs() {
        for flags in [QUOTE_FLAGS.as_slice(), RESUME_FLAGS.as_slice()] {
            let mut args = valid_arguments();
            push(&mut args, flags[0], "saved.json");
            assert!(Arguments::parse(&args).is_err());
            assert!(optional_inputs(&args, flags).is_err());
            for flag in &flags[1..] {
                push(&mut args, flag, "explicit");
            }
            assert!(Arguments::parse(&args).is_err());
            push(&mut args, "--max-batches", "1");
            assert!(Arguments::parse(&args).is_ok());
            if flags == RESUME_FLAGS {
                let mut retuned = args.clone();
                push(&mut retuned, "--population", "256");
                assert!(Arguments::parse(&retuned).is_err());
            }
            push(&mut args, flags[0], "duplicate.json");
            assert!(Arguments::parse(&args).is_err());
            assert!(optional_inputs(&args, flags).is_err());
        }
    }

    #[test]
    fn new_search_mode_requires_hash_pair_batch_one_and_exclusive_modes() {
        let mut args = valid_arguments();
        push(&mut args, "--new-search-research", "recorded.research.json");
        assert!(Arguments::parse(&args).is_err(), "missing hash must fail");
        push(&mut args, "--new-search-research-sha256", "explicit");
        assert!(
            Arguments::parse(&args).is_err(),
            "missing max-batches must fail"
        );
        push(&mut args, "--max-batches", "1");
        assert!(Arguments::parse(&args).is_ok());
        let mut with_population = args.clone();
        push(&mut with_population, "--population", "256");
        assert!(
            Arguments::parse(&with_population).is_ok(),
            "new search may use its own budget"
        );
        for (flag, value) in [
            ("--resume-research", "saved.json"),
            ("--resume-research-sha256", "explicit"),
            ("--quote-replay-bundle", "quote.json"),
            ("--quote-replay-sha256", "explicit"),
            ("--quote-replay-store", "store"),
            ("--diagnose-candidate-costs-out", "diag.json"),
            ("--diagnose-cost-variants", "old=2.5:9.0"),
        ] {
            let mut combined = args.clone();
            push(&mut combined, flag, value);
            assert!(
                Arguments::parse(&combined).is_err(),
                "{flag} must not combine with new-search mode"
            );
        }
    }

    #[test]
    fn diagnostic_cost_variants_are_explicit_and_require_a_saved_research() {
        let parsed = parse_diagnostic_cost_variants("old=2.5:9.0,broker=0.9:9.0").unwrap();
        assert_eq!(
            parsed,
            vec![
                ("old".to_owned(), 2.5, 9.0),
                ("broker".to_owned(), 0.9, 9.0)
            ]
        );
        for bad in [
            "",
            "old",
            "old=2.5",
            "old=-1:9",
            "old=2.5:nan",
            "a=1:1,a=2:2",
        ] {
            assert!(
                parse_diagnostic_cost_variants(bad).is_err(),
                "{bad} must be refused"
            );
        }
        let mut args = valid_arguments();
        push(&mut args, "--diagnose-candidate-costs-out", "out.json");
        push(&mut args, "--diagnose-cost-variants", "old=2.5:9.0");
        assert!(
            Arguments::parse(&args).is_err(),
            "cost diagnosis without an exact saved research must be refused"
        );
        push(&mut args, "--resume-research", "saved.json");
        push(&mut args, "--resume-research-sha256", "explicit");
        push(&mut args, "--max-batches", "1");
        assert!(Arguments::parse(&args).is_ok());
    }
    #[test]
    fn recorded_new_search_flags_survive_the_actual_forwarding_boundary_exactly_once() {
        let mut raw = valid_arguments();
        let path = "C:/recorded/v5.research.json";
        let sha = "abc123".repeat(10) + "abcd";
        push(&mut raw, "--new-search-research", path);
        push(&mut raw, "--new-search-research-sha256", &sha);
        push(&mut raw, "--max-batches", "1");
        let args = Arguments::parse(&raw).expect("new-search arguments");
        let mut forwarded = Vec::new();
        forward_optional_mode_flags(&args, &mut forwarded);
        for (flag, value) in [
            ("--new-search-research", path),
            ("--new-search-research-sha256", sha.as_str()),
        ] {
            assert_eq!(
                forwarded.iter().filter(|arg| arg.as_str() == flag).count(),
                1,
                "{flag} must be forwarded exactly once"
            );
            let index = forwarded
                .iter()
                .position(|arg| arg.as_str() == flag)
                .expect("forwarded flag");
            assert_eq!(forwarded[index + 1], value, "{flag} value changed");
        }
    }

    #[test]
    fn saved_candidate_signal_behavior_flag_survives_actual_forwarding_and_mode_checks() {
        let path = "C:/recorded/v7.research.json";
        let sha = "b".repeat(64);
        let mut raw = valid_arguments();
        push(&mut raw, "--resume-research", path);
        push(&mut raw, "--resume-research-sha256", &sha);
        push(&mut raw, "--max-batches", "1");
        push(
            &mut raw,
            "--diagnose-candidate-signals-out",
            "C:/out/signals.json",
        );
        let args = Arguments::parse(&raw).expect("signal census arguments");
        let mut forwarded = Vec::new();
        forward_optional_diagnostic_flags(&args, &mut forwarded);
        assert_eq!(
            forwarded
                .iter()
                .filter(|arg| arg.as_str() == "--diagnose-candidate-signals-out")
                .count(),
            1
        );
        assert!(
            Arguments::parse(&valid_arguments_with_flag(
                "--diagnose-candidate-signals-out",
                "C:/out/signals.json"
            ))
            .is_err()
        );
        let mut with_variant_only = raw.clone();
        push(
            &mut with_variant_only,
            "--diagnose-cost-variants",
            "base=1.0:2.0",
        );
        assert!(Arguments::parse(&with_variant_only).is_err());
        let mut with_cost = raw.clone();
        push(
            &mut with_cost,
            "--diagnose-candidate-costs-out",
            "C:/out/cost.json",
        );
        push(&mut with_cost, "--diagnose-cost-variants", "base=1.0:2.0");
        assert!(Arguments::parse(&with_cost).is_err());
        let mut with_new_search = raw.clone();
        push(&mut with_new_search, "--new-search-research", "C:/v7.json");
        push(
            &mut with_new_search,
            "--new-search-research-sha256",
            &"c".repeat(64),
        );
        assert!(Arguments::parse(&with_new_search).is_err());
    }

    fn valid_arguments_with_flag(flag: &str, value: &str) -> Vec<String> {
        let mut args = valid_arguments();
        push(&mut args, flag, value);
        args
    }

    fn selector_diagnostic_arguments(out_flag: &str, out_value: &str) -> Vec<String> {
        let path = "C:/recorded/v7.research.json";
        let sha = "b".repeat(64);
        let selectors =
            "0:gene_953632_18:fnv64:c7f02e00c7b758c5,83:gene_782139_19:fnv64:88f81216a43578eb";
        let mut raw = valid_arguments();
        push(&mut raw, "--resume-research", path);
        push(&mut raw, "--resume-research-sha256", &sha);
        push(&mut raw, "--max-batches", "1");
        push(&mut raw, "--diagnose-candidate-selectors", selectors);
        push(&mut raw, out_flag, out_value);
        raw
    }

    fn assert_selector_forwarded(raw: &[String], out_flag: &str) {
        let args = Arguments::parse(raw).expect("selector diagnostic arguments");
        let mut forwarded = Vec::new();
        forward_optional_diagnostic_flags(&args, &mut forwarded);
        for flag in ["--diagnose-candidate-selectors", out_flag] {
            assert_eq!(
                forwarded.iter().filter(|arg| arg.as_str() == flag).count(),
                1,
                "{flag} must be forwarded exactly once"
            );
        }
    }

    fn without_selector_pair(raw: &[String]) -> Vec<String> {
        let mut missing = raw.to_vec();
        let position = missing
            .iter()
            .position(|arg| arg == "--diagnose-candidate-selectors")
            .unwrap();
        missing.drain(position..position + 2);
        missing
    }

    #[test]
    fn saved_candidate_selector_mode_flags_parse_forward_and_exclude_other_diagnostics() {
        let weekday = selector_diagnostic_arguments(
            "--diagnose-candidate-weekdays-out",
            "C:/out/weekdays.json",
        );
        assert_selector_forwarded(&weekday, "--diagnose-candidate-weekdays-out");
        assert!(Arguments::parse(&without_selector_pair(&weekday)).is_err());
        assert!(
            Arguments::parse(&valid_arguments_with_flag(
                "--diagnose-candidate-weekdays-out",
                "C:/out/weekdays.json"
            ))
            .is_err()
        );
        let mut bad_selector = weekday.clone();
        let position = bad_selector
            .iter()
            .position(|arg| arg == "--diagnose-candidate-selectors")
            .unwrap();
        bad_selector[position + 1] = "0:gene_953632_18:fnv64:nothex".to_string();
        assert!(Arguments::parse(&bad_selector).is_err());

        let exit =
            selector_diagnostic_arguments("--diagnose-exit-policy-out", "C:/out/exit-policy.json");
        assert_selector_forwarded(&exit, "--diagnose-exit-policy-out");
        assert!(Arguments::parse(&without_selector_pair(&exit)).is_err());
        assert!(
            Arguments::parse(&valid_arguments_with_flag(
                "--diagnose-exit-policy-out",
                "C:/out/exit-policy.json"
            ))
            .is_err()
        );

        let mut both = exit.clone();
        push(
            &mut both,
            "--diagnose-candidate-weekdays-out",
            "C:/out/weekdays.json",
        );
        assert!(Arguments::parse(&both).is_err());
        for raw in [weekday, exit] {
            let mut with_signal = raw.clone();
            push(
                &mut with_signal,
                "--diagnose-candidate-signals-out",
                "C:/out/signals.json",
            );
            assert!(Arguments::parse(&with_signal).is_err());
            let mut with_cost = raw;
            push(
                &mut with_cost,
                "--diagnose-candidate-costs-out",
                "C:/out/cost.json",
            );
            push(&mut with_cost, "--diagnose-cost-variants", "base=1.0:2.0");
            assert!(Arguments::parse(&with_cost).is_err());
        }
    }

    #[test]
    fn saved_input_hash_is_checked_above_the_scalar_cost_file_limit() {
        use sha2::{Digest, Sha256};
        use std::io::Write;
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "neoethos-frozen-{}-{nonce}.json",
            std::process::id()
        ));
        let mut file = fs::File::create_new(&path).unwrap();
        let contents = vec![b' '; 70 * 1024];
        file.write_all(&contents).unwrap();
        let hash = format!("{:x}", Sha256::digest(&contents));
        assert_eq!(
            read_pinned_input(&path, &hash, "saved input", 1024 * 1024)
                .unwrap()
                .0,
            contents
        );
        assert!(
            read_pinned_input(&path, &"0".repeat(64), "saved input", 1024 * 1024)
                .unwrap_err()
                .to_string()
                .contains("SHA256 mismatch")
        );
        // Correct bytes/hash still cannot masquerade as decoded research.
        assert!(read_pinned_input(&path, &hash, "saved input", 64 * 1024).is_err());
        let args = vec![
            RESUME_FLAGS[0].to_owned(),
            path.to_string_lossy().into_owned(),
            RESUME_FLAGS[1].to_owned(),
            hash,
        ];
        assert!(SavedResearchInputs::load(&args).is_err());
        drop(file);
        fs::remove_file(path).unwrap();
    }

    #[cfg(feature = "search-monthly-upside-objective")]
    mod monthly_upside_selector_tests {
        use super::*;

        #[test]
        fn fitness_table_selector_is_accepted_for_fresh_search_and_forwarded_once() {
            for name in ["ga_fitness", "monthly_upside_v1", "monthly_upside_dd_averse_v1"] {
                let mut raw = valid_arguments();
                push(&mut raw, "--fitness-table", name);
                let args = Arguments::parse(&raw)
                    .unwrap_or_else(|error| panic!("fresh-search selector {name}: {error}"));
                let mut forwarded = Vec::new();
                forward_optional_mode_flags(&args, &mut forwarded);
                assert_eq!(forwarded, vec!["--fitness-table".to_owned(), name.to_owned()]);
            }

            // The drawdown-averse table is the third accepted fresh-search
            // selector (neo-671 2026-10-26) and forwards unchanged.
            let mut dd_averse = valid_arguments();
            push(&mut dd_averse, "--fitness-table", "monthly_upside_dd_averse_v1");
            let args = Arguments::parse(&dd_averse).expect("dd-averse selector is a fresh-search knob");
            let mut forwarded = Vec::new();
            forward_optional_mode_flags(&args, &mut forwarded);
            assert_eq!(
                forwarded,
                vec![
                    "--fitness-table".to_owned(),
                    "monthly_upside_dd_averse_v1".to_owned()
                ]
            );
            let mut dd_budget = valid_arguments();
            push(&mut dd_budget, "--fitness-table", "monthly_upside_dd_budget_v1");
            let args = Arguments::parse(&dd_budget).expect("dd-budget selector is a fresh-search knob");
            let mut forwarded = Vec::new();
            forward_optional_mode_flags(&args, &mut forwarded);
            assert_eq!(
                forwarded,
                vec![
                    "--fitness-table".to_owned(),
                    "monthly_upside_dd_budget_v1".to_owned()
                ]
            );
        }

        #[test]
        fn unknown_fitness_table_names_and_continuations_are_refused() {
            for bad in ["bogus", "archive_score", "expectancy_t_stat", ""] {
                let mut raw = valid_arguments();
                push(&mut raw, "--fitness-table", bad);
                assert!(
                    Arguments::parse(&raw).is_err(),
                    "fitness-table value {bad:?} must be refused"
                );
            }
            let mut resumed = valid_arguments();
            push(&mut resumed, "--resume-research", "saved.json");
            push(&mut resumed, "--resume-research-sha256", "explicit");
            push(&mut resumed, "--max-batches", "1");
            push(&mut resumed, "--fitness-table", "monthly_upside_v1");
            assert!(
                Arguments::parse(&resumed).is_err(),
                "saved continuation must refuse the fresh-search selector"
            );
            let mut replay = valid_arguments();
            for (flag, value) in [
                ("--quote-replay-bundle", "bundle.json"),
                ("--quote-replay-sha256", "explicit"),
                ("--quote-replay-store", "store"),
            ] {
                push(&mut replay, flag, value);
            }
            push(&mut replay, "--max-batches", "1");
            push(&mut replay, "--fitness-table", "monthly_upside_v1");
            assert!(
                Arguments::parse(&replay).is_err(),
                "quote replay must refuse the fresh-search selector"
            );
        }
    }
}
