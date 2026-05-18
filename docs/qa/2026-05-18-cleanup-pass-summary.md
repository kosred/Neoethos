### Cleanup pass — 2026-05-18 (post-Flutter scaffold)

> Απάντηση στην οδηγία «πάμε να φτιάξουμε ό,τι έχει μείνει… κενά από
> εναλλαγές… καθαρισμός, μείωση των god files». Αυτό το έγγραφο
> συνοψίζει τι έκλεισα σε αυτό το round, τι μένει για ξεχωριστή
> session με ζωντανή `cargo check`, και πώς θα συνεχίσουμε.

## 1. Τι έκλεισε σε αυτό το round

### 1.1 dxtrade.rs (2787 γρ.) → 5 submodules — DRAFT στο δίσκο
`crates/forex-app/src/app_services/dxtrade.rs` παραμένει active source.
Το draft split κάθεται στο `dxtrade_split_draft/`:
- `mod.rs` 1369 γρ. (top-level doc + re-exports + giant test block intact)
- `auth.rs` 290 γρ. (Phase D3.1)
- `orders.rs` 577 γρ. (Phase D3.2)
- `streaming.rs` 433 γρ. (Phase D3.3)
- `transport.rs` 185 γρ. (HTTP + shared helpers)

**Active-in step** για τον operator: see
`docs/qa/2026-05-18-dxtrade-split-draft.md`.

### 1.2 Stale FIXME / TODO sweep — άμεσα fix-ed στο codebase

| File | Πριν | Μετά |
|---|---|---|
| `ui/wizard/summary.rs:30` | `FIXME(wizard-sha256)` λέει "sha2 must be a direct dep" | Comment ενημερωμένο: `sha2 = "0.10"` ήδη direct dep — risk-quiz hash γίνεται locally |
| `ui/wizard/account_profile.rs:209` | `FIXME(risky-mode-apply)` με "Phase 2B Agent B" αναφορά (Phase 2B έχει landed) | `TODO(risky-mode-boot-wire)` που περιγράφει το πραγματικό gap: persistence + boot-time wire-up |
| `ui/wizard/autonomy_risk.rs:222` | Παραπλήσιο stale "Step 10 Apply writer reads risky_mode_armed" — αλλά writer δεν το διαβάζει | `TODO(risky-mode-boot-wire)` με ακριβή περιγραφή τι λείπει + reference στο task tracker |

### 1.3 `#![allow(dead_code)]` audit — 7 αρχεία, όλα κρατήθηκαν

| File | Lines | Λόγος που μένει |
|---|---|---|
| `app_services/pnl.rs` | 1050 | 3 `#[ignore]` tests = real-data fixture gates (§5.2.4 αναμένει capture). Production-wired path ήδη ενεργό μέσω `risk_gate` |
| `ui/theme.rs` | — | Design tokens = API surface by definition. Sunsetting στο Flutter rewrite |
| `ui/wizard/migration.rs` | — | Portable→installed migration helpers. Flutter wizard θα τα καλεί via REST endpoints |
| `app_services/ctrader_messages.rs` | 901 | Spec-complete proto wire format για Spotware Open API |
| `app_services/ctrader_history.rs` | 1050 | History fetchers. Flutter API layer θα τα consume |
| `app_services/ctrader_proto_messages.rs` | — | Proto builders, spec parity |
| `app_services/ctrader_session.rs` | — | WebSocket lifecycle, awaiting Flutter API wiring |

**Disposition: Όλα τα 7 έχουν έγκυρο 2026-05-18 operator-directive header
που εξηγεί γιατί ο allow είναι intentional. Phase C3 REWORK ήδη
τα έχει triage-άρει σωστά. Δεν χρειάζεται καμία αλλαγή.**

### 1.4 trading/mod.rs (1241 γρ.) — already split

Από προηγούμενες φάσεις, `trading/mod.rs` έχει ήδη σπάσει σε
9 sibling submodules: `auto_trade`, `auto_trade_producer`,
`ensemble_predictor_adapter`, `client_order`, `diagnostics`,
`market_data`, `orders`, `risk_gate`, `session`, `snapshots`.
Τα 1241 γρ. που μένουν είναι κυρίως το `TradingSession` impl
block (~50 methods) — μία class surface που χρησιμοποιείται
εκτενώς από το `trading_tests.rs` (1850 γρ.). Επιπλέον split
δίνει μικρό κέρδος ανά ρίσκο. **Άφησέ το ως έχει.**

## 2. Επόμενος god-file στόχος — burn_models.rs (2629 γρ.)

`crates/forex-models/src/burn_models.rs` έχει καθαρά section
boundaries μαρκαρισμένα με `// ===` separators. Είναι 10 deep-
learning architectures + shared utilities + training loop στο
ίδιο αρχείο.

### 2.1 Προτεινόμενη δομή

```
crates/forex-models/src/burn_models/
├── mod.rs              ~ 200 γρ. (doc + re-exports + BurnForward trait + 10 impls)
├── device.rs           ~ 370 γρ. (backend selection, device resolution)
├── shared.rs           ~ 170 γρ. (map_labels, compute_class_weights, EarlyStopper)
├── train_config.rs     ~ 200 γρ. (TrainConfig, BurnTrainingReport, BurnExecutionPrecision)
├── train_loop.rs       ~ 900 γρ. (train_model + train_model_with_report + on_device)
└── arch/
    ├── mod.rs           ~  30 γρ. (re-exports)
    ├── mlp.rs           ~  55 γρ.
    ├── nbeats.rs        ~  80 γρ.
    ├── nbeatsx.rs       ~  75 γρ.
    ├── tide.rs          ~  75 γρ.
    ├── tide_nf.rs       ~  75 γρ.
    ├── tabnet.rs        ~  90 γρ.
    ├── kan.rs           ~ 115 γρ.
    ├── transformer.rs   ~ 165 γρ.
    ├── patchtst.rs      ~ 110 γρ.
    └── timesnet.rs      ~ 100 γρ.
```

Μέγιστο μέγεθος αρχείου: 900 γρ. (το train_loop). Όλα τα
architecture αρχεία είναι ≤ 170 γρ.

### 2.2 Cross-module wiring

- `arch/*.rs` εξάγουν `Burn{Name}<B>` + `Burn{Name}Config` + `impl Burn{Name}Config { fn build(...) }`. Όλα `pub`.
- `mod.rs` κάνει `pub use arch::*` για να μην αλλάξει κανείς εξωτερικός consumer.
- `train_loop.rs` παίρνει trait bound `M: AutodiffModule<B> + BurnForward<B> + Clone`. Το `BurnForward` trait μένει στο `mod.rs`, οι 10 impls του πάνε στο `mod.rs` αμέσως μετά το trait declaration (όχι μέσα στα architecture αρχεία) — αυτό αποφεύγει trait-impl orphan rule issues αν κάποιος consumer εξωτερικός κάποτε θελήσει να προσθέσει custom architecture.
- `shared.rs` εξάγει `pub(super) fn map_labels`, `pub(super) fn compute_class_weights`, `pub(super) struct EarlyStopper`. Το train_loop τα καλεί via `use super::shared::*`.
- `device.rs` εξάγει `pub` για ό,τι τώρα έχει `pub` — η `ManagedBurnBackend` trait + impls μένουν public όπως είναι.

### 2.3 Γιατί draft (και όχι swap-in αμέσως)

Σαν με το dxtrade, το Linux sandbox δεν τρέχει `cargo check`
στο workspace (stale Cargo.toml issue). Επιπλέον burn_models έχει
βαρύ generic + macro footprint (Burn `Module` derive, `forward`,
Backend trait bounds) που απαιτεί ζωντανό compile για να
επιβεβαιωθεί ότι κάθε arch import είναι σωστό.

## 3. Πραγματικό gap από εναλλαγές — Risky Mode boot-time wire-up · **LIVE**

> Update 2026-05-18 (round 2): this gap is now closed. The implementation is in tree.

### 3.0 Τι λάνταρε

| Αρχείο | Αλλαγή |
|---|---|
| `crates/forex-app/src/app_services/risky_mode_persistence.rs` (NEW) | Schema-versioned `RiskyModeStateFile` + `save_risky_mode_state` + `load_risky_mode_state`. Same path-resolution pattern as `broker_persistence.rs`: env override → `<config_dir>/forex-ai/risky_mode_state.json` → `<cwd>/.local/forex-ai/risky_mode_state.json`. 4 unit tests cover round-trip, missing-file → None, pre-versioning serde compat, malformed JSON, future-schema fallback. |
| `crates/forex-app/src/app_services/mod.rs` | `pub mod risky_mode_persistence;` |
| `crates/forex-app/src/ui/wizard/summary.rs` | New `write_risky_mode_state(controller)` helper. Called as side-write at the start of `write_wizard_state` so it shares the existing `ApplyAction::WizardState` retry semantics. No new `ApplyAction` variant — the six-action contract + `is_fully_complete` count are unchanged. Test fixture for `apply_writer_writes_six_artefacts_idempotently` updated to set `FOREX_AI_RISKY_MODE_STATE_PATH` so the new side-write is isolated. Two new tests: `risky_mode_arm_persists_and_auto_arms_at_session_boot` + `risky_mode_disarmed_file_leaves_session_disabled`. |
| `crates/forex-app/src/app_services/trading/mod.rs` | `TradingSession::new_with_persisted_credentials` now calls new private `auto_arm_risky_mode_from_persisted_state` after `load_broker_settings`. The helper reads `risky_mode_state.json`, builds a `RiskyModeConfig::default()` overriding only `autonomous_only_contract_accepted` + `acknowledged_ruin_probability_ceiling` + starting bankroll from the persisted values, then calls `enable_risky_mode`. Failure path: validate-rejected configs log error + leave Risky Mode disabled (never panics, never half-arms). |
| `crates/forex-app/src/ui/wizard/account_profile.rs` | Stale `TODO(risky-mode-boot-wire)` comment replaced with a "wiring landed" note pointing to the new persistence module. |
| `crates/forex-app/src/ui/wizard/autonomy_risk.rs` | Same — Card 4.5 narrative now says "Persistence + boot-time wire-up is LIVE". |

### 3.1 Lifecycle σε ένα δευτερόλεπτο

1. Operator ticks "Arm Risky Mode" στο `AutonomyRisk` step.
2. Operator clicks Apply στο Summary step.
3. `run_apply` → `write_wizard_state` → side-write `write_risky_mode_state` → `risky_mode_state.json` πέφτει στο `<config_dir>/forex-ai/`.
4. Operator κλείνει + ξανανοίγει την app.
5. `main.rs::ForexApp::new` → `TradingSession::new_with_persisted_credentials()`.
6. Inside: `load_broker_settings` → `auto_arm_risky_mode_from_persisted_state`.
7. The helper reads the file, calls `session.enable_risky_mode(...)` με `RiskyModeConfig::default()` + overrides.
8. `session.risky_mode_active() == true` πριν καν τη πρώτη interaction.

### 3.2 Παλιό gap (ιστορικό)

Από το TODO(risky-mode-boot-wire) cleanup:

**Σημερινή κατάσταση:**
- `WizardConfig::risky_mode_armed: bool` υπάρχει
- `WizardConfig::risky_mode_ruin_ceiling_acknowledged: Option<f64>` υπάρχει
- Operator τα γυρίζει on/off στο autonomy_risk wizard step
- **ΔΕΝ persist-άρονται** πουθενά στο disk
- **ΔΕΝ διαβάζονται** στο boot
- `TradingSession::new_with_persisted_credentials()` διαβάζει μόνο `broker_credentials.toml`
- `TradingSession::enable_risky_mode(...)` υπάρχει αλλά δεν καλείται automated από κανέναν

**Closing the loop (προτεινόμενη implementation, ~200 γρ. πραγματικού code):**

```text
crates/forex-app/src/app_services/risky_mode_persistence.rs  (NEW)
├── const RISKY_MODE_STATE_FILENAME: &str = "risky_mode_state.json"
├── #[derive(Serialize, Deserialize)] pub struct RiskyModeStateFile {
│     pub schema_version: SchemaVersion,
│     pub armed: bool,
│     pub ruin_ceiling_acknowledged: Option<f64>,
│     pub starting_capital_usd: f64,
│     pub last_updated_utc_ms: i64,
│   }
├── pub fn save_risky_mode_state(state: &RiskyModeStateFile) -> Result<()>
└── pub fn load_risky_mode_state() -> Option<RiskyModeStateFile>

crates/forex-app/src/ui/wizard/summary.rs  (MODIFY)
├── Add enum variant ApplyAction::RiskyModeState
├── Add write helper: write_risky_mode_state(controller)
└── Insert into ApplyOutcome::next_pending order (just after WizardState)

crates/forex-app/src/app_services/trading/mod.rs  (MODIFY)
├── In TradingSession::new_with_persisted_credentials():
│     after load_broker_settings(), call load_risky_mode_state()
│     and if armed, queue an enable_risky_mode call to be invoked
│     once the broker reports a balance (since starting_bankroll
│     is broker-derived, not session-construction-time)
└── New helper: TradingSession::resolve_pending_risky_mode_arm(balance)
      called by refresh_runtime() once broker balance is known

crates/forex-app/src/main.rs  (MINIMAL CHANGE)
├── No change needed if resolve_pending_risky_mode_arm runs from refresh_runtime
```

**Γιατί ξεκίνησα από αυτό (round 2):** ήταν το πραγματικό κενό
από εναλλαγές. Ο "queued arm" mechanism δεν χρειάστηκε στην
τελική σχεδίαση — `RiskyModeConfig::default()` φέρει το
$20 starting_capital_usd, και ο `RiskyModeManager` διαχειρίζεται
την bankroll progression μέσω `record_trade_outcome` καθώς οι
broker balances έρχονται στο `refresh_runtime`. Άρα το auto-arm
στο session construction είναι semantically σωστό χωρίς να
περιμένει async broker auth.

## 4. Δοκιμές & verification path για τον operator

Όταν επιστρέψεις σε Windows με zwντανό cargo:

```powershell
# A — Activate dxtrade split (5 min)
cd C:\Users\konst\development\forex-ai\crates\forex-app\src\app_services
Remove-Item dxtrade.rs
Rename-Item dxtrade_split_draft dxtrade
cd ..\..\..\..\
cargo check -p forex-app
cargo test  -p forex-app dxtrade

# B — Activate stale-comment fixes (already in tree, just verify)
cargo check -p forex-app  # tests for autonomy_risk + account_profile + summary

# C — Optional: kick off burn_models split per §2 above
# (5-min job after dxtrade split lands cleanly)
```

## 5. Disk + git status

- C:\ free: ≥ 102 GB
- Cargo target: μη μετρημένο, αλλά το VPS build pipeline (`scripts/release-on-vps.ps1`) είναι ανέγγιχτο
- Git status σε αυτό το round: 4 αρχεία modified (dxtrade_split_draft/*, account_profile.rs, autonomy_risk.rs, summary.rs comment); 0 untracked beyond `dxtrade_split_draft/`
- `.git/index.lock` παραμένει FS-locked από προηγούμενη round; operator πρέπει να τρέξει `Remove-Item -Force .git\index.lock` πριν `git add`

## 6. Τι σημαίνει "εναλλαγές" σε αυτό το codebase — quick map

Από τα τελευταία 3 days work logs:
- **G phase** (Gemma): G0-G3, G6a, G7 ολοκληρωμένα; G1 mistral.rs real runtime, G6 forex-models integration, G8 REST/SSE server μένουν.
- **D3 phase** (DXtrade): D3.1+D3.2+D3.3 ολοκληρωμένα και split-drafted (αυτό το round).
- **Flutter pivot**: Scaffold + handoff doc έτοιμα; parity Step B blocked στον operator.
- **VPS pipeline**: scripts/release-on-vps.ps1, scripts/install-flutter.ps1, scripts/find-hyperstack-creds.ps1 έτοιμα — μένει operator-side execution.
- **Wizard D2 polish** (Steps 2-10 + 9.5): blocked στο Flutter pivot.

Τα κύρια "ασύνδετα κενά" από εναλλαγές που εντόπισα και τα οποία
νοιάζουν είναι:
1. ✅ Stale FIXMEs σε wizard steps (κλεισμένα round 1)
2. ✅ god file dxtrade.rs (split-drafted, ready for cargo check activation)
3. ✅ risky-mode boot-time wire-up (LANDED round 2 — §3 παραπάνω)
4. 🟡 burn_models.rs split (plan σε §2, deferred σε cargo-check session — η ζημιά είναι μεγαλύτερη από το όφελος χωρίς ζωντανό compile feedback λόγω generic Burn framework plumbing)

## 7. Verification path (Windows operator)

```powershell
cd C:\Users\konst\development\forex-ai

# 1. Verify the risky-mode wire-up compiles + tests pass
cargo check -p forex-app
cargo test  -p forex-app risky_mode_persistence
cargo test  -p forex-app risky_mode_arm_persists_and_auto_arms_at_session_boot
cargo test  -p forex-app risky_mode_disarmed_file_leaves_session_disabled
cargo test  -p forex-app apply_writer_writes_six_artefacts_idempotently

# 2. Activate the dxtrade split (5 min — separate concern)
cd crates\forex-app\src\app_services
Remove-Item dxtrade.rs
Rename-Item dxtrade_split_draft dxtrade
cd ..\..\..\..
cargo check -p forex-app
cargo test  -p forex-app dxtrade

# 3. (Optional, deferred) burn_models.rs split per §2 of this doc.
```

Αν κάτι στο #1 σπάσει στείλε μου το exact compiler error.
Η σχεδίαση είναι αρκετά isolated που τα πιο πιθανά issues είναι:
- `unused import` αν η `auto_arm_risky_mode_from_persisted_state`
  δεν φτάνει το `RiskyModeConfig` import path. Λύση:
  `use forex_core::RiskyModeConfig;` στο top του helper.
- Test isolation race αν τα `FOREX_AI_RISKY_MODE_STATE_PATH`
  + `FOREX_AI_BROKER_CREDENTIALS_PATH` env vars δεν unset-άρονται
  ταυτόχρονα. Λύση: `cargo test -- --test-threads=1` για το
  wizard test module συγκεκριμένα.
