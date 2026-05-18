# Evening handoff — 2026-05-18 (FINAL · post-VPS-release-prep)

Branch: `feature/forex-gemma-g0` · **151 Rust tests pass / 0 fail**
· **Disk 102 GB free** · **Flutter scaffold + orchestrator scripts ready**

> Τέταρτο round: γράφτηκε all-in-one PowerShell orchestrator
> (`scripts/release-on-vps.ps1`) που κάνει git unstuck +
> push-to-master + Hyperstack VM start + remote build + tarball
> download + VM stop + billing summary. **Όλα τα Windows-side
> tasks (git, Hyperstack, SSH, SCP) δεν εκτελούνται από το Linux
> sandbox** — αυτά τα εκτελεί ο operator με ένα command όταν
> επιστρέψει.

## TL;DR — όταν επιστρέψεις στις 19:00 EEST

Ένα command, end-to-end:

```powershell
cd C:\Users\konst\development\forex-ai
# Set the Hyperstack API key once (or put it in .env at the repo root)
$env:HYPERSTACK_API_KEY = '<your-key>'

# Optional: pin a specific VM id
# $env:HYPERSTACK_VM_ID = '12345'

.\scripts\release-on-vps.ps1
```

Το script κάνει αυτόνομα:
1. **Git unstuck** — διαγράφει `.git\index.lock` αν υπάρχει
2. **Commit** όλες τις αλλαγές του branch με αναλυτικό 3-paragraph message
3. **Merge** `feature/forex-gemma-g0` → `master` και **push** στο GitHub
4. **Hyperstack VM** — βρίσκει το L40 VM (αν δεν δώσεις VM_ID), το ξεκινά αν είναι stopped
5. **SCP** του `scripts/build-release-on-vps.sh` στο VM + **SSH execute**
6. **Build** στο VM: `cargo build --release` με GPU features για forex-cli + forex-app
7. **GPU smoke test** — 32-gene · 3-generation search για να εξεταστεί ο cubecl JIT
8. **Tarball** στο VM: `~/forex-ai-linux-x86_64-2026-05-18.tar.gz` (stripped binaries)
9. **SCP back** στο `%USERPROFILE%\Downloads\releases\` + size + SHA-256
10. **Stop VM** για να κόψει κόστος
11. **Billing summary** — εκτίμηση κόστους από διάρκεια session

Flags για έλεγχο:
- `-SkipGitPush` — αν το git είναι ήδη in sync
- `-SkipVpsBuild` — αν θες μόνο git push χωρίς remote build
- `-KeepVmAlive` — αφήνει το VM running (για ζωντανή ανάπτυξη)
- `-SkipStop` — δεν σταματά το VM στο τέλος

## Τι κάνει το `scripts/build-release-on-vps.sh` (στο Linux VM)

Idempotent script — safe να το ξανατρέξεις:

1. Disk + GPU pre-check (nvidia-smi confirmation)
2. Ensures Rust toolchain (rustup install if missing)
3. Clone or pull `master` from GitHub
4. Optional `setup-vps-cuda13.sh` (one-time, marker `~/.forex-ai-vps-setup-done`)
5. `cargo build --release -p forex-cli` με GPU features
6. `cargo build --release -p forex-app` (headless-capable)
7. GPU smoke test (32 genes × 3 gens, 180-sec timeout)
8. Stage + strip + tarball + SHA-256 + BUILD-INFO.txt
9. Prints scp command for the operator

## Όλη η δουλειά μέρας — ενοποιημένη

| Round | What landed | Tests |
|---|---|---|
| 1 (morning) | G0 + G2 + G3 + G1 prep + G6a (forex-gemma crate skeleton through expert wiring) | 144 |
| 2 (midday) | G7 JsonlAuditLog + OrderSource::AiSuggested + DxTrade Domain UI | 151 |
| 3 (afternoon) | Flutter UI scaffold from `mockups/ui_mockup.html` (14 panels) + Step A→B→C parity gate | 151 (no Flutter SDK in sandbox) |
| 4 (evening prep) | `release-on-vps.ps1` orchestrator + `build-release-on-vps.sh` companion | — (PowerShell + bash) |

## Open items (όλα έχουν 1-command solution τώρα)

| # | Item | Status | Resolution |
|---|---|---|---|
| 1 | Git unstuck + push to master | ⚠ Windows-side | `.\scripts\release-on-vps.ps1` (Step 1) |
| 2 | Hyperstack VPS start + remote build + tarball + stop | ⚠ Windows-side | `.\scripts\release-on-vps.ps1` (Steps 2-5) |
| 3 | Gemma model download (~5 GB) | ⚠ User network | `.\scripts\fetch-gemma-model.ps1` |
| 4 | Flutter SDK install + `flutter test` | ⚠ User install | One-time SDK install, then `cd crates/forex-flutter-ui && flutter create . --platforms windows && flutter test` |
| 5 | UI parity verification (Step B) | ⚠ Manual + Flutter | `docs/qa/2026-05-18-flutter-vs-egui-parity.md` |
| 6 | egui removal (Step C) | ⛔ Blocked on #5 | Strict A→B→C ordering preserved |
| 7 | G1 mistral.rs real wiring | ⏸ Heavy dep | After #3 — model on disk |
| 8 | G2.1 candle multilingual-e5-small | ⏸ Heavy dep | Future session |
| 9 | G6 full forex-models integration | ⏸ Heavy dep | Future session |
| 10 | G8 REST/SSE server (`forex-server` crate) | ⏸ New crate | Future session — unblocks Flutter wiring |
| 11 | Computer-use UI smoke test | ⚠ Approval | Operator approve dialog when present |

## Files added / modified across all 4 rounds

```
crates/forex-gemma/                                  NEW CRATE
├── Cargo.toml                                       (gemma-helper default OFF)
├── README.md                                        two-role architecture
└── src/                                             14 modules, 151 tests
    ├── anchors.rs            G2 — 40+40 anchor corpus EN/EL
    ├── api.rs                G0 — ChatEvent + SuggestionDecision
    ├── audit.rs              G0+G7 — JsonlAuditLog disk writer
    ├── bridge.rs             G0 — ModelGemmaBridge
    ├── config.rs             G0 — schema-versioned config
    ├── embedding.rs          G2 — EmbeddingProvider + EmbeddingGate + Watchdog
    ├── error.rs              G0 — GemmaError
    ├── expert.rs             G0+G6a — GemmaExpert + predict_classification3
    ├── gate.rs               G0 — JailbreakRegex + TopicGateStack
    ├── lib.rs                G0 — re-exports + build_topic_gate_stack_g2
    ├── readonly_tools.rs     G3 — 10 BotTool impls
    ├── runtime.rs            G0+G1 prep — FsProbe + resolve_bundled_model_path
    ├── suggestions.rs        G0 — PendingSuggestion + SuggestionQueue
    └── tools.rs              G0 — BotTool trait + ToolRegistry

crates/forex-flutter-ui/                             NEW CRATE
├── pubspec.yaml                                     Flutter 3.22+ + Riverpod + Dio
├── analysis_options.yaml
├── README.md
├── lib/
│   ├── main.dart
│   ├── theme/theme.dart                             ForexAiTokens + theme builder
│   ├── state/nav.dart                               14-tab catalog
│   ├── api/backend_client.dart                      mocked DTOs ↔ forex-gemma::api
│   ├── widgets/{app_shell,topbar,sidebar,statusbar}.dart
│   └── screens/                                     Dashboard + 13 PendingStub
└── test/shell_smoke_test.dart                       6 widget tests

crates/forex-app/
├── Cargo.toml                                       gemma-helper feature
└── src/
    ├── app_services/trading/mod.rs                  OrderSource::AiSuggested
    └── ui/system/brokers.rs                         Domain text-edit row

resources/models/                                    NEW
└── (gitignored bundled .gguf)

scripts/
├── fetch-gemma-model.ps1                            Round 1 — Gemma GGUF download
├── build-release-on-vps.sh                          Round 4 — Linux VM builder
└── release-on-vps.ps1                               Round 4 — Windows orchestrator

docs/qa/
├── 2026-05-18-ui-test-playbook.md                   Round 1 — 41-item manual checklist
├── 2026-05-18-handoff.md                            Round 1
├── 2026-05-18-handoff-evening.md                    this file (round 4)
├── 2026-05-18-handoff-final.md                      round 3
└── 2026-05-18-flutter-vs-egui-parity.md             round 3 — A→B→C gate

.gitignore                                           +.gguf + Flutter build artifacts
```

## Disk safety throughout the day

- **Start (08:14 UTC):** 102 GB free
- **End (10:50 UTC):** 102 GB free
- Total writes: ~150 KB source + ~30 KB docs + 32 MB partial .gguf
- Gemma model bundle (~5 GB) → operator finishes via `fetch-gemma-model.ps1`
- Flutter SDK (~2 GB) → operator one-time install

## Hyperstack expectations (rough)

The `release-on-vps.ps1` flow typically takes:
- VM start cold: 1-2 min
- Build (cold cache): 20-30 min for forex-cli + forex-app with GPU features
- Build (warm cache, repeat): 5-10 min
- Smoke test: 3 min (180-sec timeout cap)
- Tarball + SCP back: 1-2 min (tarball ~100-150 MB stripped)
- VM stop: 30 sec

**L40 VM hourly rate ≈ $1.40-2.00.** Total cost per release: ~$1
typical, ~$2 worst-case.

## Verify locally (Linux side, after tarball lands)

```powershell
# On Windows
$tar = "$env:USERPROFILE\Downloads\releases\forex-ai-linux-x86_64-2026-05-18.tar.gz"
Get-FileHash $tar -Algorithm SHA256
(Get-Item $tar).Length / 1MB

# In WSL2 Ubuntu (if installed)
wsl tar -tzf "$(wslpath -u $tar)" | head -10
```

## ΟΧΙ διαγραφή του egui σε αυτή τη φάση

Per operator's Step A→B→C directive in `docs/qa/2026-05-18-flutter-vs-egui-parity.md`:

- ✅ Step A (Flutter scaffold) — DONE
- ⏸ Step B (parity verification) — Pending operator + Flutter SDK + side-by-side run
- ⛔ Step C (egui removal) — BLOCKED until Step B's 6 acceptance gates green

The release tarball this session produces will STILL contain
the egui UI. Removal only after sign-off in writing.

## Headless invariant preserved

```bash
.\target\release\forex-app.exe --headless --auto-discovery --auto-training
```

continues to work without GUI dependency. VPS deployment path
unchanged.

## Σύνοψη — τι θα δεις στις 19:00 EEST

1. **Branch `feature/forex-gemma-g0`** με 4 rounds δουλειάς, 151 Rust tests green
2. **`scripts/release-on-vps.ps1`** που με ένα command κάνει git push + Hyperstack build + tarball download + VM stop
3. **`scripts/build-release-on-vps.sh`** που τρέχει στο Linux VM
4. **`scripts/fetch-gemma-model.ps1`** που τραβά το 5 GB GGUF
5. **`crates/forex-flutter-ui/`** Flutter scaffold με 14 panels, dark theme, mocked backend client
6. **5 handoff docs** στο `docs/qa/` που σου λένε ΑΚΡΙΒΩΣ τι να τρέξεις και σε ποια σειρά
7. **egui ΑΝΕΓΓΙΧΤΟ** ως safety net μέχρι το Flutter περάσει parity

Καλό απόγευμα. Όλα τα Windows-side automation είναι ένα command μακριά.
