# training_orchestrator.rs split — draft (pending verification)

## Τι έγινε

`crates/forex-models/src/training_orchestrator.rs` (4137 γρ. on
Windows / 4051 γρ. καταγραμμένο πριν την fix) — **το μεγαλύτερο
god-file στο codebase** — τεμαχίστηκε σε **3 sibling modules** ως
draft, ακολουθώντας το ίδιο pattern με τα dxtrade + burn_models splits.

Active source παραμένει το ορίτζιναλ `training_orchestrator.rs` — το
split βρίσκεται στο `training_orchestrator_split_draft/` ώστε ο build
να μη σπάσει μέχρι να επιβεβαιωθεί από τον operator με ζωντανή `cargo
check`.

```
crates/forex-models/src/
├── training_orchestrator.rs                   4137 γρ. — ACTIVE source (μένει αμετάβλητο)
└── training_orchestrator_split_draft/         DRAFT — 3 αρχεία ≤ 1946 γρ.
    ├── mod.rs            ~1946    doc + module decls + types (TrainingRunSummary, TrainingOrchestrator) + impl block (51 methods)
    ├── helpers.rs        ~1809    51 top-level pure-fn helpers (param parsing, profile, artifacts, HPO dispatch)
    └── tests.rs          ~464     19 #[test] functions + 3 test fixture helpers
                       ─────
                        4219    +82 γραμμές vs original (overhead = module docs + re-exports)
```

**Max file:** 1946 γραμμές (mod.rs) — από 4137 → **53% reduction στο
largest file**.

## Cross-module wiring

- **`mod.rs`** δηλώνει `mod helpers;` + `#[cfg(test)] mod tests;` και
  κάνει `pub use helpers::*;` ώστε το `crate::training_orchestrator::*`
  surface να μείνει unchanged για κάθε consumer.
- **`helpers.rs`** εκθέτει 51 `pub(super) fn` (από `fn` που ήταν
  module-private pre-split). Όλα τα helpers είναι pure functions
  (κανένα `&self`). Το `use super::*;` στο top ανεβάζει όλα τα types
  + imports από το parent.
- **`tests.rs`** έχει inner `use super::*;` που pull-άρει και τα
  re-exported helpers + τα κανονικά types του parent module. Καμία
  αλλαγή σε test fixtures.

## Visibility μετατροπές που εφαρμόστηκαν

Όλα τα top-level helpers στο `helpers.rs` ήταν `fn` (module-private)
και έγιναν `pub(super) fn` ώστε ο parent `mod.rs` (που έχει τις
impl methods που τα καλούν) να μπορεί να φτάσει τα symbols από το
sibling module:

```
51 helpers promoted: compute_true_ranges, trailing_average,
labels_to_series, canonical_model_name, transformer_replica_index,
configured_contains_model, parse_tree_params, parse_f32_param,
parse_f64_param, parse_usize_param, parse_u64_param, parse_bool_param,
parse_string_param, model_params_only, hpo_backend_from_params,
hpo_trials_from_params, hpo_max_rows_from_params,
embargo_minutes_from_params, holdout_pct_from_params,
confidence_threshold_from_params, metric_weight_from_params,
accuracy_weight_from_params, export_onnx_requested,
parse_parent_selection_policy, parse_survivor_selection_policy,
training_runtime_profile, training_profile_higher_timeframes,
write_training_profile_sidecar, timeframe_to_minutes,
embargo_rows_for_timeframe, halton, sample_choice, sample_f64,
sample_usize, calibration_method_from_params, model_artifact_dir,
staged_training_artifact_dir, backup_training_artifact_dir,
cleanup_training_artifact_dir, replace_training_artifact_dir,
with_staged_training_artifact_dir, persist_training_artifacts,
supports_hpo, uses_shared_expert_dispatch, inject_tree_seed,
build_expert_model, generate_hpo_candidate_params, select_hpo_dataset,
optimize_model_config, write_onnx_status_sidecar, train_model_dispatch
```

## Stale-mount-cache discovery

Κατά τη διάρκεια αυτού του round εντόπισα ότι ο Linux sandbox mount
**τεμαχίζει** μεγάλα recently-modified αρχεία:

| File | Bash sees | Windows sees | Bash-truncated lines |
|---|---|---|---|
| `burn_models.rs` | 2629 | 2634 | 5 |
| `training_orchestrator.rs` | 4051 | 4137 | 86 |
| `dxtrade.rs` | 2787 | 2787 | 0 (intact) |

Implication: **Παλιά (μέρες πριν) splits που έγιναν μέσω bash sed
μπορεί να μην έχουν το test-tail του ορίτζιναλ.** Το `burn_models`
split το διόρθωσα μόλις τώρα προσθέτοντας τα 5 χαμένα closing-brace
lines στο `train_loop.rs`. Το `training_orchestrator` split tests.rs
γράφτηκε direct μέσω Windows-path Write tool ώστε να πάρει τα πλήρη
4137 lines.

## Πώς να ολοκληρώσεις το split (5-min job σε Windows)

```powershell
cd C:\Users\konst\development\forex-ai\crates\forex-models\src

# 1. Activate the split. ΑΣΦΑΛΕΣ — original δεν χάνεται μέχρι να
#    δεις cargo check OK.
Move-Item training_orchestrator.rs training_orchestrator.rs.bak
Move-Item training_orchestrator_split_draft training_orchestrator

# 2. Try compile (cd up two levels first).
cd ..\..\..
cargo check -p forex-models

# 3. Run the orchestrator tests
cargo test -p forex-models training_orchestrator
```

### Αν cargo βρει errors (πιο πιθανά)

| Πιθανό error | Αιτία | Fix |
|---|---|---|
| `error[E0432]: unresolved import` σε helpers.rs `use super::*;` | Κάποιο type/import που χρειάζονται οι helpers δεν εξάγεται από τον parent module | Πρόσθεσε το συγκεκριμένο import (π.χ. `use anyhow::Context;`) στο header του helpers.rs |
| `error[E0603]: function 'X' is private` σε mod.rs | Έχω ξεχάσει να promotedν σε `pub(super)` έναν helper που καλείται από impl method | Άλλαξε σε `pub(super) fn X` |
| `error[E0428]: 'helpers' defined multiple times` | Συγκρούεται με κάποιο άλλο `helpers` symbol | Μετονόμασε στο mod.rs |
| `error[E0252]: name 'X' is defined multiple times` | Re-export συγκρούεται | Αφαιρώ το διπλό `pub use` |
| Unused-import warnings | Το `use super::*;` του helpers.rs φέρνει πολλά που δεν χρησιμοποιεί | `cargo fix --bin forex-models --allow-dirty` τα διαγράφει |

### Rollback σε ένα βήμα

```powershell
cd C:\Users\konst\development\forex-ai\crates\forex-models\src
Move-Item training_orchestrator       training_orchestrator_split_draft
Move-Item training_orchestrator.rs.bak training_orchestrator.rs
```

## Cleanup artifacts στο draft directory

Στο `training_orchestrator_split_draft/` υπάρχουν 3 zero-byte
`_section_*.body` files από το sed pipeline. Operator μπορεί να
τα σβήσει πριν το activate ή να αφήσει το Move-Item να τα μετακινήσει
μαζί με τα ενεργά αρχεία (.body δεν είναι .rs οπότε δεν τα βλέπει
ο compiler).

## Disk + γραμμές

- Free disk: 102 GB stable
- 3 νέα `*.rs` αρχεία (~76 KB total) + 3 zero-byte staging artifacts
- Δεν τράβηξα τίποτα από δίσκο σε αυτό το round
