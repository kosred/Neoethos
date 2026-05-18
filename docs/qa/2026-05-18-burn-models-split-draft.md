# burn_models.rs split — draft (pending verification)

## Τι έγινε

`crates/forex-models/src/burn_models.rs` (2629 γραμμές) τεμαχίστηκε
σε **5 sibling modules** ως draft, ακολουθώντας το ίδιο pattern
με το dxtrade split (round 1, doc στο `2026-05-18-dxtrade-split-draft.md`).

Active source παραμένει το ορίτζιναλ `burn_models.rs` — το split
βρίσκεται στο `burn_models_split_draft/` ώστε ο build να μη σπάσει
μέχρι να επιβεβαιωθεί από τον operator με ζωντανή `cargo check`.

```
crates/forex-models/src/
├── burn_models.rs                    2629 γρ. — ACTIVE source (μένει αμετάβλητο)
└── burn_models_split_draft/          DRAFT — 5 αρχεία ≤ 965 γρ.
    ├── mod.rs            136    doc + module decls + re-exports + BurnForward trait + 10 impls
    ├── device.rs         359    TrainBackend/InferBackend type aliases + Wgpu init + policy → device resolution + ManagedBurnBackend trait
    ├── shared.rs         187    map_labels + compute_class_weights + EarlyStopper + scalar_loss_value + tensor conversion helpers
    ├── arch.rs           965    Οι 10 architectures (MLP/NBeats/NBeatsx/TiDE/TiDENf/TabNet/KAN/Transformer/PatchTST/TimesNet) + shared inner blocks (NBeatsBlock, ResidualBlock, KANLayer, SequenceTransformerBlock)
    ├── train_config.rs   201    TrainConfig + BurnTrainingReport + BurnExecutionPrecision
    └── train_loop.rs     926    Cross-entropy loss + helpers + generic training loop + tests
                       ─────
                        2774    +145 γραμμές vs original (overhead = imports + module docs)
```

**Maximum single file:** 965 γραμμές (arch.rs) — από 2629 → **37% reduction στο largest file**.
**Όλα τα impl/utility αρχεία ≤ 360 γραμμές** εκτός των δύο "carriers":
arch.rs (965, οι 10 architectures μαζί λόγω shared helper blocks) και
train_loop.rs (926, η γενική training pipeline + tests).

## Cross-module wiring που εφαρμόστηκε

- **`shared.rs`** εκθέτει `pub(super)` τις εξής helpers ώστε το
  `train_loop.rs` να μπορεί να τα χρησιμοποιήσει χωρίς να τα
  εκθέσει στο `forex-models::*` API:
  - `map_labels`, `compute_class_weights`, `scalar_loss_value`,
    `float_dtype`, `cast_tensor_to_dtype`, `time_series_split`,
    `EarlyStopper` (struct + `new` + `check`),
    `array2_to_tensor_with_dtype`, `array2_to_tensor`,
    `labels_to_tensor`.
  - `cast_module_float_tensors` διατηρεί το pre-split `pub(crate)` —
    χρησιμοποιείται και από άλλα modules του `forex-models` crate.
- **`arch.rs`** κρατά κάθε architecture με τα δικά της `pub struct`
  + `pub struct {Name}Config` + `impl {Name}Config { pub fn init... }`
  + `impl<B: Backend> Burn{Name}<B> { pub fn forward... }`. Οι shared
  inner blocks (`NBeatsBlock` για NBeats/NBeatsx, `ResidualBlock` για
  TiDE/TiDENf, `KANLayer`, `SequenceTransformerBlock`) μένουν `pub`
  στο ίδιο αρχείο γιατί μοιράζονται από adjacent architectures.
- **`mod.rs`** ορίζει το `BurnForward` trait + τις 10 impls. Οι impls
  είναι local-trait + local-type (orphan-rule clean) γιατί `BurnForward`
  δηλώνεται εδώ και κάθε `BurnXxxx<B>` έρχεται από `arch.rs` που είναι
  child module.
- **`train_loop.rs`** καλεί `use super::arch::*` + targeted imports
  από `super::shared::*` + `super::device::*` + `super::train_config::*`.

## Γιατί draft (και όχι swap-in αμέσως)

Πιο σύνθετο σενάριο vs dxtrade γιατί:

1. **Generic-heavy Burn framework code** — `<B: Backend>` threading
   παντού, `#[derive(Module, Config, Debug)]` macros που απαιτούν
   `burn::prelude::*` per file, type aliases `TrainBackend`/`InferBackend`
   που χρειάζονται re-export από τα consumer modules.
2. **WGPU feature-gate** — `#[cfg(feature = "burn-wgpu-backend")]` paths
   χρειάζονται προσοχή ώστε όλα τα WGPU-only imports + helpers να μείνουν
   στο `device.rs` και να μη "ξεφύγουν" στα siblings.
3. **Test block 340 γραμμές** στο `train_loop.rs` που τρέχει κάθε
   architecture — γρήγορο επικυρωτικό σε ένα `cargo test`.

Χωρίς ζωντανή `cargo check` δεν εγγυώμαι ότι:
- Όλα τα `pub(super)` markers είναι σωστά scope-arισμένα.
- Δεν υπάρχουν unused imports (warnings, όχι errors, αλλά clean code matters).
- Τα macros του `#[derive(Module)]` λύνουν σωστά το `burn::prelude::*`
  σε όλα τα architecture sites.

## Πώς να ολοκληρώσεις το split (5-min job σε Windows)

```powershell
cd C:\Users\konst\development\forex-ai\crates\forex-models\src

# 1. Activate the split. ΑΣΦΑΛΕΣ — original δεν χάνεται μέχρι να
#    δεις cargo check OK.
Move-Item burn_models.rs burn_models.rs.bak
Move-Item burn_models_split_draft burn_models

# 2. Try compile (cd up two levels first).
cd ..\..\..
cargo check -p forex-models

# 3. Run the architecture + training tests
cargo test -p forex-models burn_models
```

### Αν το build περάσει + τα tests περάσουν

```powershell
Remove-Item C:\Users\konst\development\forex-ai\crates\forex-models\src\burn_models.rs.bak
Remove-Item C:\Users\konst\development\forex-ai\crates\forex-models\src\burn_models\_section_*.body
git add crates/forex-models/src/burn_models/
git rm  crates/forex-models/src/burn_models.rs
git commit -m "refactor · burn_models.rs (2629 γρ.) → burn_models/{mod,device,shared,arch,train_config,train_loop}.rs

Max file 965 γρ. (arch.rs). Σπασμένο σε device/shared/arch/train_config/
train_loop. BurnForward trait μένει στο mod.rs (orphan-rule clean);
οι 10 impls του ακολουθούν. Shared helpers στο shared.rs με pub(super)
visibility ώστε να μη ξεφύγουν στο forex-models::* API."
```

### Αν cargo βρει errors (πιο πιθανά)

| Πιθανό error | Αιτία | Fix |
|---|---|---|
| `error[E0432]: unresolved import super::BurnForward` σε train_loop.rs | Το trait δηλώθηκε στο mod.rs αλλά το child module δεν το βλέπει | Αλλάζω σε `use crate::burn_models::BurnForward;` ή `use super::BurnForward;` (το super θα δουλέψει αν το mod.rs το τοποθετεί ΠΡΙΝ τα `mod arch;` declarations) |
| `error[E0603]: function 'X' is private` σε train_loop.rs | Ένα ακόμα helper στο shared.rs χρειάζεται `pub(super)` | Σταυρώνω το όνομα στο shared.rs και αλλάζω visibility |
| `error[E0428]: type alias 'TrainBackend' defined multiple times` | Το device.rs εξάγει + το train_loop.rs τα ξανα-imports | Αφαιρώ το redundant import — ή κρατώ ένα μόνο μέσω `super::device::TrainBackend` |
| Unused-import warnings | Δεν χρησιμοποιώ κάθε `use` που πρόσθεσα στα headers | `cargo fix --bin forex-models --allow-dirty` τα διαγράφει αυτόματα |

### Rollback σε ένα βήμα αν θες reset

```powershell
cd C:\Users\konst\development\forex-ai\crates\forex-models\src
Move-Item burn_models       burn_models_split_draft
Move-Item burn_models.rs.bak burn_models.rs
# Original δεν αλλοιώθηκε, τίποτα δεν χάνεται.
```

## Αναμενόμενος χρόνος εκτέλεσης

| Step | ETA |
|---|---|
| `Move-Item × 2` | <5 sec |
| `cargo check -p forex-models` (incremental) | 30-60 sec |
| Fix οποιοδήποτε visibility error | 5-15 min ανά error (αν βγει) |
| `cargo test -p forex-models burn_models` | 2-5 min (έχει 10 architecture forward tests + 1-2 training cycles) |
| Σύνολο | 5-30 min ανάλογα με errors |

## Disk + γραμμές

- Free disk: 102 GB stable
- Σύνολο νέων αρχείων: 6 (mod, device, shared, arch, train_config, train_loop) + 6 zero-byte `_section_*.body` artifacts του sed pipeline που ο operator μπορεί να σβήσει
- Δεν τράβηξα τίποτα από δίσκο σε αυτό το round
