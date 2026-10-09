//! One operator entrypoint for the existing receipt-bound CPU research lane.
//! Acquisition receipts own source selection; this command issues no trading permit.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use neoethos_broker_history::canonical_research_costs::{
    ensure_unique_selected_timeframe, ensure_unique_series, read_bounded_regular_file,
    validate_exact_file_settings,
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
const OPTIONAL: [&str; 7] = [
    "--higher",
    "--population",
    "--generations",
    "--max-indicators",
    "--candidates",
    "--max-batches",
    "--cpu-threads",
];

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
                REQUIRED.contains(&flag) || OPTIONAL.contains(&flag),
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

/// Resolve the same explicit settings file before installing the process budget.
pub fn settings_path(args: &[String]) -> Result<PathBuf> {
    Ok(Arguments::parse(args)?.value("--settings-source").into())
}

pub fn run(args: &[String], settings: &Settings) -> Result<()> {
    ensure!(
        !cfg!(feature = "gpu-nvidia"),
        "canonical-research is an explicit CPU research command; use a CPU build or the sealed native-research GPU route"
    );
    let mut args = Arguments::parse(args)?;
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
}
