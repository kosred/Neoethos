# dqn_impl.rs split — draft (pending verification)

## Τι έγινε

`crates/forex-models/src/rl/dqn_impl.rs` (2658 γρ.) — DQN
reinforcement learner — τεμαχίστηκε σε **2 sibling modules** ως
draft. Το test block ήταν ήδη εξωτερικό μέσω
`#[path = "dqn_impl_tests.rs"] mod tests;`, οπότε δεν χρειάστηκε
να μετακινηθεί.

Active source παραμένει το ορίτζιναλ `dqn_impl.rs` — το split
βρίσκεται στο `dqn_impl_split_draft/` ώστε ο build να μη σπάσει
μέχρι να επιβεβαιωθεί από τον operator με ζωντανή `cargo check`.

```
crates/forex-models/src/rl/
├── dqn_impl.rs                         2658 γρ. — ACTIVE source (μένει αμετάβλητο)
├── dqn_impl_tests.rs                   1429 γρ. — UNTOUCHED (test block, ήδη external)
└── dqn_impl_split_draft/               DRAFT — 2 αρχεία
    ├── mod.rs           ~1941   doc + module decl + types + encoded {Transition, Episode, Env} + TradingReinforcementLearner impl (1553 γρ. inner!) + Default + test ref
    └── helpers.rs        ~763   36 pure helpers + FeatureBounds struct + impl (fit/discretize/normalize)
                       ─────
                        2704    +46 γραμμές vs original (overhead = imports + module docs)
```

**Max file:** 1941 γρ. (mod.rs) — από 2658 → **27% reduction**
στο largest file. Λιγότερο δραστική μείωση γιατί το
`impl TradingReinforcementLearner` είναι 1553 γρ. και δεν σπάει
ασφαλώς χωρίς cargo check (ο impl block έχει tight coupling με
self state + feature-gated paths).

## Cross-module wiring

- **`mod.rs`** δηλώνει `mod helpers;` + `pub(super) use helpers::*;`
  ώστε όλα τα symbols να είναι ορατά στο parent module (`rl`) και
  τους siblings (`dqn_impl_tests.rs` που reference-άρει συγκεκριμένα
  helpers μέσω `#[path]`).
- **`helpers.rs`** εκθέτει 36 `pub(super) fn` + `pub(super) struct
  FeatureBounds` με `pub(super)` fields (`mins`, `maxs`) και
  `pub(super) fn fit / discretize / normalize` methods. Όλα τα
  helpers ήταν `fn` (module-private) pre-split.

## Visibility μετατροπές

```
36 helpers promoted: validate_q_values, softmax_q_values,
default_rl_feature_columns, expand_fallback_basis, fallback_backend_name,
resolve_rl_training_precision_with_capability,
is_known_rl_requested_backend, is_known_rl_effective_backend,
is_known_rl_requested_device_policy, is_known_rl_effective_device_policy,
artifact_requested_device_policy, artifact_effective_backend,
artifact_effective_device_policy, staged_rl_file, backup_rl_file,
rl_runtime_metadata, requested_gpu_device_policy,
normalize_rl_network_precision, dtype_to_rl_network_precision,
artifact_network_precision, probe_runtime_rl_bf16_support,
validate_rl_metadata, resolve_rl_runtime_metadata,
cleanup_rl_temp_file, stage_rl_target, restore_rl_backup,
rollback_rl_target, build_reward_triplet, build_training_episodes,
normalize_rl_device_policy, requested_cuda_ordinal,
resolve_rl_training_device, resolve_rl_inference_device,
q_value_for_action, sync_linear_q_target

FeatureBounds promoted: struct + fields + 3 inherent methods (fit,
discretize, normalize)
```

## Stale-mount-cache status

`dqn_impl.rs` εμφανιζόταν ως **intact** σε bash side (2658 lines τότε
και τώρα). Το Windows side είχε 2659 lines (1 trailing newline). Άρα
δεν χρειάστηκε Windows-side write για το split — όλο το extraction
έγινε via bash + sed pipeline όπως πριν.

## Πώς να ολοκληρώσεις το split (5-min job σε Windows)

```powershell
cd C:\Users\konst\development\forex-ai\crates\forex-models\src\rl

# 1. Activate the split. ΑΣΦΑΛΕΣ — original δεν χάνεται.
Move-Item dqn_impl.rs dqn_impl.rs.bak
Move-Item dqn_impl_split_draft dqn_impl

# 2. Try compile
cd ..\..\..\..
cargo check -p forex-models

# 3. Run the DQN tests (the external test block still hooks up)
cargo test -p forex-models dqn_impl
```

**Σημαντικό:** το `#[path = "dqn_impl_tests.rs"]` στο νέο `mod.rs`
δείχνει σε σχετικό path. Όταν το directory γίνει `dqn_impl/`, το
`mod tests;` ψάχνει για `dqn_impl_tests.rs` δίπλα στο `mod.rs`
(δηλαδή μέσα στο `dqn_impl/` directory). Πρέπει να **μετακινήσεις**
το test file:

```powershell
Move-Item C:\Users\konst\development\forex-ai\crates\forex-models\src\rl\dqn_impl_tests.rs `
          C:\Users\konst\development\forex-ai\crates\forex-models\src\rl\dqn_impl\dqn_impl_tests.rs
```

ΑΛΛΑ μπορεί κάποιος consumer του `crate::rl::*` να reference-άρει
τα tests σαν `crate::rl::dqn_impl_tests::*`. Αν πέσει «file not
found», άλλαξε το path attribute στο mod.rs σε
`#[path = "../dqn_impl_tests.rs"]` και κράτα το test file στο
παλιό location (parent dir of the new `dqn_impl/` directory).

### Rollback σε ένα βήμα

```powershell
cd C:\Users\konst\development\forex-ai\crates\forex-models\src\rl
Move-Item dqn_impl         dqn_impl_split_draft
Move-Item dqn_impl.rs.bak  dqn_impl.rs
# (αν μετακίνησες το test file, επανέφερέ το πίσω)
```

### Πιθανά errors κατά cargo check

| Error | Αιτία | Fix |
|---|---|---|
| `error[E0603]: function 'X' is private` | Helper που ξέχασα να promotedν | `pub(super) fn X` στο helpers.rs |
| `error[E0432]: unresolved import super::*` | Κάποιο type που χρειάζεται helpers.rs δεν εξάγεται από parent | Add explicit `use crate::path::Type` στο helpers.rs |
| Test file not found | `#[path]` δεν είναι σωστό | Δες παραπάνω instructions για το test file |

## Disk

- Free disk: 102 GB stable
- 2 νέα `*.rs` αρχεία (~80 KB total) + 3 zero-byte staging artifacts (`_section_*.body`)
