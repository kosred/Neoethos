//! **DRIVE IT AND COUNT IT.**
//!
//! `loop_end_to_end.rs` asserts that the loop reaches a verdict. This file drives
//! the same loop and writes down the NUMBERS: how many searches ran, how many
//! screens passed, how many champions were recorded, how many times `best_ever`
//! advanced, whether a verdict was written, and — the question round 2 exists to
//! answer — whether the GOAL UNREACHABLE branch is a path that can actually fire
//! or only a variant that can be constructed by hand in a unit test.
//!
//! It writes a census JSON beside the store so the numbers can be read without
//! `--nocapture`, and asserts nothing that `loop_end_to_end.rs` does not already
//! assert. A census that fails is a census that reports nothing.

mod support;

use support::{ScriptedExecutor, tag_counts};

use neoethos_autoresearch::runner::{RunArgs, run_with_executor};

const MAX_SWEEPS: usize = 24;

fn census_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("autoresearch-census-{name}.json"))
}

#[test]
fn count_everything_the_loop_did() {
    let root = support::fresh_root("census");
    let settings = support::settings_in(&root);
    let base = support::base_config(&settings);

    let started = std::time::Instant::now();
    let mut executor = ScriptedExecutor::new();
    let outcome = run_with_executor(
        RunArgs {
            max_sweeps: MAX_SWEEPS,
            ..RunArgs::new("EURUSD")
        },
        &settings,
        base,
        &mut executor,
    );
    let elapsed_s = started.elapsed().as_secs_f64();

    let dir = support::only_session_dir();
    let lines = support::journal_lines(&dir);
    let tags = tag_counts(&lines);

    // Screen outcomes, counted by the conjunct that refused them. This is the
    // number that was ZERO for every slot before round 2: `passed`.
    let mut screen_outcomes: std::collections::BTreeMap<String, usize> = Default::default();
    for record in support::records_tagged(&lines, "Screened") {
        let key = record
            .get("screen_result")
            .and_then(|r| {
                let outcome = r.get("outcome").and_then(|o| o.as_str())?;
                Some(match r.get("conjunct").and_then(|c| c.as_str()) {
                    Some(conjunct) => format!("{outcome}:{conjunct}"),
                    None => outcome.to_string(),
                })
            })
            .unwrap_or_else(|| "<unparsed>".to_string());
        *screen_outcomes.entry(key).or_insert(0) += 1;
    }

    let (verdict_json, error) = match &outcome {
        Ok(v) => (
            serde_json::to_value(v).expect("the verdict serialises"),
            serde_json::Value::Null,
        ),
        Err(e) => (
            serde_json::Value::Null,
            serde_json::Value::String(format!("{e:#}")),
        ),
    };

    let census = serde_json::json!({
        "elapsed_s": elapsed_s,
        "run_error": error,
        "searches_executed_by_the_fixture": executor.searches_run(),
        "controls_executed_by_the_fixture":
            executor.executed.iter().filter(|e| e.control).count(),
        "oos_calls_into_the_fixture": executor.oos_calls.len(),
        "journal_tag_counts": tags,
        "screen_outcomes": screen_outcomes,
        "verdict_json_exists": dir.join("verdict.json").exists(),
        "champions_file_exists": dir.join("session_champions.json").exists(),
        "distinct_screen_failures": support::distinct_screen_failures(&lines),
        "matrices_left_on_disk": support::matrices_on_disk(&dir),
        "live_portfolio_written_anywhere": support::files_named(&root, "live_portfolio.json")
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>(),
        "verdict": verdict_json,
        "rendered": outcome.as_ref().map(|v| v.render()).unwrap_or_default(),
    });
    let path = census_path("scripted");
    std::fs::write(&path, serde_json::to_vec_pretty(&census).expect("census"))
        .unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    eprintln!("CENSUS WRITTEN TO {}", path.display());

    outcome.expect("the loop must reach a verdict");
}
