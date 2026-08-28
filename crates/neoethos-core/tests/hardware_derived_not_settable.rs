//! A hardware-derived value must never be writable into the operator's config,
//! and an answer-changing value must never be derived from hardware.
//!
//! Both halves of that sentence are defects this repo actually shipped.
//!
//! **Half one — derived values pickled as input.** `Settings::save` serialises
//! the entire settings struct on every write, and both `POST /settings` and
//! `POST /risk/preset` load-mutate-save. So one click on any unrelated control
//! froze every hardware-derived `Default` into the YAML as a literal. The
//! fingerprint is still visible in the repo config: `n_jobs: 11` is exactly
//! `available_parallelism() - 1` on a twelve-core box, written by a machine that
//! is not the machine the file is now read on. From then on the config *claims*
//! a thread budget it never chose, and a bigger box inherits the smaller box's
//! answer.
//!
//! **Half two — the same reflex applied to the wrong knobs.** The deleted
//! `AutoTuner` would have written `hpo_trials = if gpu { 50 } else { 20 }`.
//! That is not a capacity decision. A different trial count returns a different
//! answer, not the same answer in a different footprint, so letting the card
//! decide it means the strategy that reaches live money depends on which box
//! happened to be rented that week.
//!
//! The line between the two: **a knob is capacity when a machine with twice the
//! RAM would want a different number, and only because of the hardware.**
//!
//! These tests pin both directions. They are deliberately written against the
//! SERIALISED config and the retired-key table rather than against field names,
//! so they keep working whether a derived field was deleted outright or marked
//! `#[serde(skip)]` — either remedy satisfies "not settable", and a test that
//! named the fields would have to be rewritten by whoever chose.

use neoethos_core::config::Settings;
use neoethos_core::execution_budget::{CapacityDetection, CoordinationScope, LogicalThreadCount};
use neoethos_core::system::{
    ExecutionBudgetInputs, RETIRED_DERIVED_KEYS, WorkloadDemand, WorkloadKind, retired_derived_key,
};

/// Knobs that decide **what the search finds**, not how it fits in memory.
///
/// Deriving any of these from the hardware is the mistake this file exists to
/// prevent, and it is a tempting one because they all *look* like resource
/// parameters. If a future pass wants to make one hardware-aware, the pattern is
/// `population_auto`: opt-in, floored at the configured value, and warn-logged
/// as answer-changing — not a silent probe-driven assignment.
const ANSWER_CHANGING_KNOBS: &[&str] = &[
    "hpo_trials",
    "prop_search_population",
    "prop_search_generations",
    "prop_search_max_hours",
    "prop_search_max_rows",
    "cpcv_max_rows",
    "l1_feature_selection_sample_limit",
    "global_max_rows",
];

/// Resolve a dotted path such as `system.n_jobs` inside a YAML document.
fn lookup<'a>(doc: &'a serde_yaml_ng::Value, dotted: &str) -> Option<&'a serde_yaml_ng::Value> {
    let mut cursor = doc;
    for segment in dotted.split('.') {
        cursor = cursor.get(segment)?;
    }
    Some(cursor)
}

fn serialized_default_settings() -> serde_yaml_ng::Value {
    let text = serde_yaml_ng::to_string(&Settings::default())
        .expect("Settings must serialise — `Settings::save` depends on it");
    serde_yaml_ng::from_str(&text).expect("a serialised Settings must be valid YAML")
}

/// The headline. Whatever `Settings::save` writes is what the operator's file
/// ends up containing, so a derived key that still serialises is a derived key
/// that will be pickled as input on the next click.
#[test]
fn derived_keys_never_reach_the_operators_file() {
    let doc = serialized_default_settings();

    let leaked: Vec<&str> = RETIRED_DERIVED_KEYS
        .iter()
        .filter(|entry| lookup(&doc, entry.key).is_some())
        .map(|entry| entry.key)
        .collect();

    assert!(
        leaked.is_empty(),
        "these keys are derived from the machine but still serialise into the operator's \
         config, so the next `Settings::save` will pickle this box's probe result into his \
         file as if he had typed it: {leaked:?}\n\
         Fix by deleting the field (preferred — the value has no reader) or by marking it \
         `#[serde(skip)]`. Both satisfy this test."
    );
}

/// The other direction, and the one that costs money if it is ever crossed.
#[test]
fn answer_changing_knobs_are_never_retired_as_derived() {
    for knob in ANSWER_CHANGING_KNOBS {
        assert!(
            retired_derived_key(knob).is_none(),
            "`{knob}` was added to RETIRED_DERIVED_KEYS, which means the hardware now decides \
             it. It changes the ANSWER, not the footprint: a different value returns a \
             different set of surviving strategies on the same data. Capacity knobs may be \
             derived; this one may not."
        );
    }
}

/// A retired key must be findable both ways, because a serde error hands the
/// loader a bare leaf name while a document walk hands it the full path. A
/// lookup that only worked one way would report the operator's stale key as a
/// nameless unknown field.
#[test]
fn retired_keys_resolve_by_full_path_and_by_leaf() {
    for entry in RETIRED_DERIVED_KEYS {
        let by_path = retired_derived_key(entry.key)
            .unwrap_or_else(|| panic!("`{}` must resolve by its full path", entry.key));
        assert_eq!(by_path.key, entry.key);

        let leaf = entry
            .key
            .rsplit('.')
            .next()
            .expect("a dotted key has a last segment");
        let by_leaf = retired_derived_key(leaf)
            .unwrap_or_else(|| panic!("`{leaf}` must resolve by its leaf name"));
        assert_eq!(by_leaf.key, entry.key);
    }
}

/// "Ignored" on its own is a shrug. He set the key on purpose once, so the
/// message owes him the key, the derivation, and — where it can be had without
/// a full accelerator probe — the number this machine actually computes.
#[test]
fn the_ignored_message_names_the_key_the_reason_and_the_number() {
    let settings = Settings::default();

    for entry in RETIRED_DERIVED_KEYS {
        let message = entry.ignored_message(&settings);
        assert!(
            message.contains(entry.key),
            "the message for `{}` must name the key: {message}",
            entry.key
        );
        assert!(
            message.contains(entry.derived_from),
            "the message for `{}` must say what it is derived from: {message}",
            entry.key
        );
        if let Some(knob) = entry.set_instead {
            assert!(
                message.contains(knob),
                "the message for `{}` must point at `{knob}`, the knob that is still his: \
                 {message}",
                entry.key
            );
        }
    }

    // `system.n_jobs` is the one whose value is cheap enough to state outright,
    // and it is also the one whose stale copy is provably in the shipped config.
    let n_jobs = retired_derived_key("system.n_jobs").expect("system.n_jobs is retired");
    let expected = ExecutionBudgetInputs::from_settings_and_parent(
        &settings,
        None,
        CoordinationScope::ProcessLocal,
    )
    .expect("default settings are valid")
    .resolve()
    .expect("typed cap provenance is valid")
    .effective_worker_limit
    .get()
    .to_string();
    assert_eq!(
        n_jobs.computed_value(&settings).as_deref(),
        Some(expected.as_str()),
        "the retired-key report must quote the budget this machine computes, not a guess"
    );
    assert!(
        n_jobs.ignored_message(&settings).contains(&expected),
        "the computed value must appear in the message the operator reads"
    );
}

/// `system.hardware.cpu_budget` is the shape every hardware knob should have:
/// the probe sets the ceiling, the operator may only ask for less. A budget
/// that could exceed the cores that exist would be a user parameter deciding
/// peak resource use — the never-OOM invariant inverted.
#[test]
fn the_operator_may_narrow_the_cpu_budget_but_never_raise_it() {
    let mut settings = Settings::default();

    let resolve = |settings: &Settings| {
        ExecutionBudgetInputs::from_settings_parent_and_detection(
            settings,
            None,
            supplied_detection(12),
            CoordinationScope::ProcessLocal,
        )
        .expect("positive setting")
        .resolve()
        .expect("typed cap provenance")
        .effective_worker_limit
        .get()
    };

    assert_eq!(
        resolve(&settings),
        10,
        "auto = effective logical threads - 2"
    );

    settings.system.hardware.cpu_budget = Some(4);
    assert_eq!(resolve(&settings), 4, "asking for less works");

    settings.system.hardware.cpu_budget = Some(9_999);
    assert_eq!(
        resolve(&settings),
        10,
        "an oversized cap must not consume the fixed two-thread stability reserve"
    );
}

fn supplied_detection(effective_logical_threads: usize) -> CapacityDetection {
    CapacityDetection::supplied(
        LogicalThreadCount::new(effective_logical_threads).expect("test capacity must be positive"),
    )
}

#[test]
fn typed_inputs_apply_every_cap_by_minimum_without_mutating_settings() {
    let mut settings = Settings::default();
    settings.system.hardware.cpu_budget = Some(9);
    settings.models.backtest_runtime.rayon_threads = Some(7);

    let resolved = ExecutionBudgetInputs::from_settings_parent_and_detection(
        &settings,
        Some(5),
        supplied_detection(12),
        CoordinationScope::ManagedProcessTree,
    )
    .expect("positive caps are valid")
    .resolve()
    .expect("cap provenance is valid");

    assert_eq!(resolved.automatic_worker_limit.get(), 10);
    assert_eq!(resolved.effective_worker_limit.get(), 5);
    assert_eq!(settings.system.hardware.cpu_budget, Some(9));
    assert_eq!(settings.models.backtest_runtime.rayon_threads, Some(7));
}

#[test]
fn zero_cpu_caps_are_rejected_with_the_exact_config_key() {
    let mut canonical = Settings::default();
    canonical.system.hardware.cpu_budget = Some(0);
    let error = ExecutionBudgetInputs::from_settings_parent_and_detection(
        &canonical,
        None,
        supplied_detection(12),
        CoordinationScope::ProcessLocal,
    )
    .expect_err("zero canonical cap must fail");
    assert!(error.to_string().contains("system.hardware.cpu_budget"));

    let mut legacy = Settings::default();
    legacy.models.backtest_runtime.rayon_threads = Some(0);
    let error = ExecutionBudgetInputs::from_settings_parent_and_detection(
        &legacy,
        None,
        supplied_detection(12),
        CoordinationScope::ProcessLocal,
    )
    .expect_err("zero legacy cap must fail");
    assert!(
        error
            .to_string()
            .contains("models.backtest_runtime.rayon_threads")
    );
}

#[test]
fn workload_demand_uses_the_installed_capacity_without_hidden_8_16_or_2_clamps() {
    let settings = Settings::default();
    let resolved = ExecutionBudgetInputs::from_settings_parent_and_detection(
        &settings,
        None,
        supplied_detection(64),
        CoordinationScope::ProcessLocal,
    )
    .expect("default settings are valid")
    .resolve()
    .expect("resolution succeeds");
    assert_eq!(resolved.effective_worker_limit.get(), 62);

    for workload in [WorkloadKind::DataIngestion, WorkloadKind::Inference] {
        let demand = WorkloadDemand::for_parallel_units(workload, 64);
        assert_eq!(
            demand.granted_workers(&resolved),
            62,
            "{workload:?} must be able to use every admitted worker when useful work exists"
        );
    }

    let small = WorkloadDemand::for_parallel_units(WorkloadKind::DataIngestion, 3);
    assert_eq!(small.granted_workers(&resolved), 3);

    let ui = WorkloadDemand::lightweight_control(WorkloadKind::Ui);
    assert_eq!(ui.granted_workers(&resolved), 0);
}

/// Close the loop from the source side: `system.rs` decides the hardware, and it
/// must decide it from the probe and `Settings` alone. An `env::var` here is a
/// second resolution path, and a second resolution path is how two subsystems in
/// one process ended up reading different files.
///
/// This also guards the deletion that removed the last one: the OMP/MKL/OpenBLAS
/// clamp lived in `AutoTuner`, which had no callers, so those variables were
/// never actually set. Re-introducing the write here — rather than as its own
/// measured change — would clamp BLAS thread pools for the first time inside a
/// commit that claims to be about config.
#[test]
fn system_rs_resolves_no_environment_variable() {
    let source = include_str!("../src/system.rs");

    // Strip `//` comments; this file's own tombstones discuss `env::var` in
    // prose and must not trip the scan. `system.rs` contains no block comments
    // and no string literal containing `//`.
    let code: String = source
        .lines()
        .map(|line| match line.find("//") {
            Some(idx) => &line[..idx],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n");

    for forbidden in [
        "env::var",
        "env::set_var",
        "env::remove_var",
        "use std::env",
    ] {
        assert!(
            !code.contains(forbidden),
            "`{forbidden}` reappeared in crates/neoethos-core/src/system.rs. The hardware \
             decision has exactly two inputs — the probe and `Settings` — and adding a third \
             re-opens the second resolution path. Add a field to `system.hardware` instead."
        );
    }
}
