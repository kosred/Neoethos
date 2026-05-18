# Final handoff — 2026-05-18 (round 3 · Flutter scaffold)

Branch: `feature/forex-gemma-g0` · **151 Rust tests pass / 0 fail**
· **Disk 102 GB free** · **Flutter shell ready**

> Τρίτο round: Flutter UI scaffold από το `mockups/ui_mockup.html`.
> egui ΜΕΝΕΙ ανέπαφο — η αντικατάσταση γίνεται μόνο αφού περάσει
> Step B parity (manual user-side verification).

## Δομή που παραδόθηκε σήμερα

```
forex-ai/
├── crates/
│   ├── forex-gemma/                       NEW CRATE (round 1+2)
│   │   ├── Cargo.toml                     gemma-helper feature OFF
│   │   ├── README.md                      two-role architecture + bundled-model
│   │   └── src/                           14 modules, 151 tests
│   │
│   ├── forex-flutter-ui/                  NEW CRATE (round 3)
│   │   ├── pubspec.yaml                   Flutter 3.22+, Riverpod, Dio, fl_chart
│   │   ├── analysis_options.yaml          flutter_lints
│   │   ├── README.md                      setup + per-phase plan
│   │   ├── lib/
│   │   │   ├── main.dart
│   │   │   ├── theme/theme.dart           ForexAiTokens + buildForexAiTheme
│   │   │   ├── state/nav.dart             14-tab catalog
│   │   │   ├── api/backend_client.dart    mocked DTOs, ready for G8 wiring
│   │   │   ├── widgets/                   app_shell, topbar, sidebar, statusbar
│   │   │   └── screens/                   _placeholder + 14 panels
│   │   └── test/shell_smoke_test.dart     6 widget tests
│   │
│   ├── forex-app/
│   │   ├── Cargo.toml                     gemma-helper feature (optional dep)
│   │   ├── src/app_services/
│   │   │   ├── trading/mod.rs             OrderSource::AiSuggested (NEW round 2)
│   │   │   └── dxtrade.rs                 D3.1/D3.2/D3.3 full impls
│   │   └── src/ui/system/brokers.rs       Domain text-edit row (NEW round 2)
│   │
│   └── forex-app/src/ui/ ANEFFECTED        egui UI stays untouched
│
├── resources/models/                       NEW DIR (round 1)
│   └── *.gguf                              gitignored; bundled by installer
│
├── scripts/
│   └── fetch-gemma-model.ps1               NEW (round 1) — 5GB download script
│
└── docs/qa/                                NEW DIR
    ├── 2026-05-18-ui-test-playbook.md             round 1
    ├── 2026-05-18-handoff.md                      round 1
    ├── 2026-05-18-handoff-evening.md              round 2
    ├── 2026-05-18-flutter-vs-egui-parity.md       round 3 — A→B→C
    └── 2026-05-18-handoff-final.md                this file
```

## Σύνοψη όλων των rounds

| Round | What landed | Tests |
|---|---|---|
| 1 (morning) | G0 scaffolding, G2 topic gate, G3 read-only tools, G1 prep (model path resolver + installer bundle), G6a `GemmaExpert` wiring | 144 pass |
| 2 (midday) | G7 JSONL audit writer, `OrderSource::AiSuggested`, DxTrade Domain UI row | 151 pass |
| 3 (afternoon) | Flutter UI scaffold (shell + 14 screens + 6 widget tests + theme + Riverpod nav + mocked API client) + parity checklist + .gitignore | 151 pass |

## Operator action queue (όταν επιστρέψεις 19:00 EEST)

### Πρώτο, πιο γρήγορα:

```powershell
cd C:\Users\konst\development\forex-ai
Remove-Item -Force .git\index.lock
cargo test -p forex-gemma | Select-String "test result"
# Expect: "test result: ok. 151 passed; 0 failed; ..."
```

### Δεύτερο, finish the Gemma model download:

```powershell
.\scripts\fetch-gemma-model.ps1
# ~5 GB. Resumes from existing .gguf.tmp.
```

### Τρίτο, install Flutter SDK + verify scaffold:

```powershell
# One-time: install Flutter from https://docs.flutter.dev/get-started/install/windows
flutter config --enable-windows-desktop
cd crates\forex-flutter-ui
flutter create . --platforms windows,macos,linux --org com.forexai
flutter pub get
flutter test
# Expect: 6 widget tests pass.
flutter run -d windows
# Should open the Flutter shell with TradingView dark theme.
```

### Τέταρτο, side-by-side parity test:

`docs/qa/2026-05-18-flutter-vs-egui-parity.md` έχει ολοκληρωμένο
Step B checklist:
- 14 panels (each: egui works? / Flutter renders? / Flutter real
  data?)
- 7 critical-path integration tests
- 6 acceptance gates για Step C unlock

Άνοιξε σε δύο παράθυρα:
- `target/release/forex-app.exe` (egui)
- `flutter run -d windows` (Flutter)

Σύγκρινε panel-by-panel. Συμπλήρωσε τα ☐ checkboxes.

### Πέμπτο (ΜΟΝΟ αν Step B perfect):

```bash
git checkout -b chore/remove-egui
git commit --allow-empty -m "checkpoint — Flutter ready, egui still present"
# Remove egui (see Step C section in parity doc)
rm -r crates/forex-app/src/ui/
# Edit crates/forex-app/Cargo.toml to drop eframe/egui/egui_dock
# Edit crates/forex-app/src/main.rs to drop GUI entry
cargo build --release && cargo test --workspace
git add -A && git commit -m "remove egui — Flutter is sole UI"
```

**ΠΡΟΣΟΧΗ:** Step C ΔΕΝ εκτελείται μέχρι Step B verified.
Πιθανότατα θα χρειαστεί κι ένα G8 round για να υπάρχει REST
backend που το Flutter να καλέσει.

## Headless invariant

Σε όλα τα rounds έχει διατηρηθεί:

```powershell
.\target\release\forex-app.exe --headless --auto-discovery --auto-training
```

Τρέχει χωρίς GUI dependency. Όταν αργότερα φύγει το egui, ο
operator πρέπει να επιβεβαιώσει ότι το `--headless` flag δεν
ξαναβρίσκει eframe dep στα code paths.

## Commit boundaries (όλα τα rounds)

```bash
git checkout feature/forex-gemma-g0
Remove-Item -Force .git\index.lock  # if locked

# Round 1
git add crates/forex-gemma/ Cargo.toml crates/forex-app/Cargo.toml \
        .gitignore scripts/fetch-gemma-model.ps1 resources/
git commit -m "Phase G · forex-gemma helper crate (G0+G2+G3+G6a, model bundle prep)"

# Round 2
git add crates/forex-app/src/app_services/trading/mod.rs \
        crates/forex-app/src/ui/system/brokers.rs \
        crates/forex-gemma/src/audit.rs crates/forex-gemma/src/lib.rs
git commit -m "Phase G + D3.1 follow-ups · AiSuggested variant + DxTrade Domain UI + G7 JSONL audit"

# Round 3
git add crates/forex-flutter-ui/ .gitignore
git commit -m "Flutter UI · scaffold from mockups/ui_mockup.html (14 panels, dark theme, Riverpod)"

git add docs/qa/
git commit -m "QA · 2026-05-18 handoffs + Flutter-vs-egui parity checklist"
```

## Disk safety throughout

- Session start: 102 GB free
- Session end: 102 GB free (net change minimal — only test artifacts in /tmp/gemma-check)
- Gemma model: 31 MB partial (`resources/models/*.gguf.tmp`) — operator finishes via PowerShell script
- Flutter SDK: 0 MB (sandbox doesn't have SDK; operator installs)

## Πραγματικά "open"

Αυτά είναι τα **μόνα** items που χρειάζονται έξω από τα steps
παραπάνω:

1. **G1 mistral.rs runtime real wiring** — μετά το model download
2. **G2.1 candle multilingual-e5-small** — όταν παρθεί απόφαση για real embedder
3. **G6 full forex-models integration** — `ensemble-integration` feature
4. **G8 REST/SSE server (`forex-server` crate)** — το missing link για Flutter ↔ Rust
5. **Flutter screens real data wiring** — χωρίς G8 stays at PendingStub
6. **Flutter SSE chat for Gemma** — depends on G8 + G1
7. **Real UI tests via computer-use** — χρειάζεται user-approved request_access
8. **Flutter pivot acceptance gate → egui removal (Step C)** — depends on Step B

## Στα Ελληνικά — τι θα δεις όταν επιστρέψεις

Δουλειά σήμερα (3 αυτόνομα rounds):

- **Forex-gemma crate πλήρες** στο G0/G2/G3/G6a/G7 — 151 tests
  πάνω από 12 module files, schema-versioned, look-ahead-bias
  guarded, default-off feature flag.
- **DxTrade Domain UI row** + **`OrderSource::AiSuggested` variant**
  στο forex-app — fixes 2 από τα 8 follow-ups του πρώτου handoff
  σε pure Rust code.
- **Flutter UI scaffold** από το mockup σου — 14 panels, full
  dark TradingView theme, sidebar nav με 3 groups, Dashboard
  με dummy data + positions table, 13 placeholder screens που
  λένε εξήγηση τι έρχεται, backend client με DTOs ταιριαστά με
  το Gemma API. **egui ΑΝΕΓΓΙΧΤΟ.**
- **6 widget tests** στο Flutter (smoke + nav catalog +
  τα 14 screens load χωρίς panic) και **4 handoff docs** που
  σου λένε ΑΚΡΙΒΩΣ τι να τρέξεις σε ποια σειρά.

**Τι ΔΕΝ έγινε γιατί χρειάζεται εσένα:**
- Git commits (FS-level index.lock σε Linux sandbox)
- Gemma model download finalization (curl session lifecycle)
- Computer-use UI tests (request_access timeout χωρίς approval)
- `flutter create` + `flutter test` (no SDK στο sandbox)
- Step C egui removal (waits for Step B verification)

Όλα τα παραπάνω έχουν σαφή instructions στα handoff docs.
Branch `feature/forex-gemma-g0` σταθερό, 151 Rust tests
green, Flutter scaffold ready for `flutter pub get`. Καλό
απόγευμα.
