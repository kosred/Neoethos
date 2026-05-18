# Flutter ↔ egui Parity Checklist — 2026-05-18

**Status:** ⏸ Step A (scaffolding) DONE · Step B (verification)
PENDING-USER · Step C (egui removal) ⛔ BLOCKED until Step B passes.

> Αυστηρή σειρά: A → B → C. Καμία διαγραφή του egui προτού το
> Flutter περάσει όλα τα critical paths από αυτό το checklist
> στο πραγματικό hardware του operator. Το sandbox δεν έχει
> Flutter SDK εγκατεστημένο, οπότε το Step B δεν μπορεί να
> τρέξει αυτόματα — απαιτεί manual run μετά από `flutter pub
> get` + `flutter run`.

## Step A — Scaffold (DONE σε αυτό το session)

| Item | Status |
|---|---|
| `crates/forex-flutter-ui/` directory δημιουργήθηκε | ✅ |
| `pubspec.yaml` με Flutter 3.22+ + Riverpod + Dio + fl_chart | ✅ |
| `analysis_options.yaml` με flutter_lints | ✅ |
| Design tokens (`lib/theme/theme.dart`) — TradingView dark scheme από mockup | ✅ |
| Nav catalog (`lib/state/nav.dart`) — 14 panels grouped Trading/AI/System | ✅ |
| App shell grid (TopBar + Sidebar + Dock + StatusBar) | ✅ |
| 14 screen widgets (Dashboard fleshed-out, 13 placeholder) | ✅ |
| Backend client skeleton με mocked DTOs ταιριαστά με `forex-gemma::api` | ✅ |
| Widget tests (6 smoke tests) | ✅ |
| `README.md` με setup instructions για operator | ✅ |
| `egui` UI σε `crates/forex-app/src/ui/` ΑΝΕΓΓΙΧΤΟ | ✅ |

## Step B — Verification (PENDING-USER)

### B.1 — Setup (one-time, ~15 min)

```powershell
# Install Flutter SDK if not present
flutter --version  # if "not recognized": install from https://docs.flutter.dev/get-started/install/windows
flutter config --enable-windows-desktop

cd C:\Users\konst\development\forex-ai\crates\forex-flutter-ui
flutter create . --platforms windows,macos,linux --org com.forexai
flutter pub get

# Disk safety check
Get-PSDrive C | Select-Object @{N='FreeGB';E={[math]::Round($_.Free/1GB,2)}}
# Expect > 30 GB after Flutter SDK install.

flutter test
# Expect: 6 widget tests pass.

flutter run -d windows
# Should open the Flutter shell with topbar + sidebar + dashboard dummy data.
```

### B.2 — Per-panel parity (manual side-by-side)

Με το Flutter app ανοιχτό σε ένα παράθυρο και το `target/release/forex-app.exe` σε άλλο, σύγκρινε panel-by-panel:

| Panel | egui works | Flutter renders | Flutter real data | Notes |
|---|---|---|---|---|
| **Dashboard** | ☐ | ☐ | ☐ | Flutter has dummy data (balance 10K, 2 positions). Real data needs `/account/snapshot` REST endpoint |
| **Chart** | ☐ | ☐ | ☐ | Flutter shows PendingStub. Needs custom candlestick widget + bar feed |
| **Markets** | ☐ | ☐ | ☐ | Live quote subscription needs cTrader/DxTrade Push API bridge in Rust |
| **Order Ticket** | ☐ | ☐ | ☐ | Form validation works statically; submit needs `OrderSource::Manual` REST endpoint |
| **News** | ☐ | ☐ | ☐ | News feed needs OpenAI key (user-supplied, Phase 0b) |
| **Trade Watch** | ☐ | ☐ | ☐ | Same data source as Dashboard positions |
| **Discovery** | ☐ | ☐ | ☐ | Status feed from `ServiceEvent::DiscoveryUpdated` channel |
| **Training** | ☐ | ☐ | ☐ | Status feed from `ServiceEvent::TrainingUpdated` channel |
| **Intelligence** | ☐ | ☐ | ☐ | Per-expert vote breakdown — same data Gemma's `explain_last_decision` tool returns |
| **Broker Setup → cTrader** | ☐ | ☐ | ☐ | Pre-filled embedded credentials + OAuth flow |
| **Broker Setup → DxTrade** | ☐ | ☐ | ☐ | 4 rows: Platform URL / Username / Domain (NEW) / Password |
| **Data Bootstrap** | ☐ | ☐ | ☐ | Folder picker + historical download |
| **Hardware** | ☐ | ☐ | ☐ | CPU/GPU/RAM detection from `forex_core::HardwareProfile` |
| **Risk Settings** | ☐ | ☐ | ☐ | Risky Mode tier display + autonomous-only contract toggle |
| **Settings** | ☐ | ☐ | ☐ | App-wide preferences (theme override, log levels, etc.) |

### B.3 — Critical-path integration tests

| Test | Pass criteria | Status |
|---|---|---|
| cTrader OAuth round-trip | Click "Start cTrader Login (Automatic)" → browser opens → user-side OAuth → token saved to disk | ☐ |
| DxTrade login (D3.1 path) | Enter URL/Username/Domain/Password → "Save Credentials" → file written + readable on next launch | ☐ |
| Headless `--auto-discovery` | `target/release/forex-app.exe --headless --auto-discovery` runs without GUI launch | ☐ |
| Headless `--auto-training` | Same as above, training side | ☐ |
| Headless both | `--headless --auto-discovery --auto-training` runs simultaneously, no deadlock | ☐ |
| Gemma chat box | (Future G8 wiring) — Flutter SSE client receives chat token stream | ☐ |
| Theme fidelity | Side-by-side, the dark scheme + accent + sidebar styles match the mockup ≥ 95% | ☐ |

### B.4 — Acceptance gate for Step C

**Step C unlocks only when ALL of these hold:**

1. Every row in B.2 has Flutter renders = ✓.
2. Every "Flutter real data" column is either ✓ OR explicitly
   marked as "deferred until G8 REST wiring lands".
3. Every row in B.3 is ✓ or ⚠ (⚠ = known-deferred, not unknown).
4. `flutter test` passes (6 widget tests + any new ones the
   operator adds for critical paths).
5. `cargo test --workspace` still passes — the Flutter scaffold
   doesn't break the Rust side.
6. Operator gives explicit sign-off in writing
   (handoff doc entry).

## Step C — egui removal (BLOCKED)

⛔ **DO NOT EXECUTE** πριν περάσουν τα Step B gates παραπάνω.

Όταν Step B passes, ο operator τρέχει τα εξής σε ξεχωριστά
commits πάνω σε branch `chore/remove-egui`:

```bash
# Safety checkpoint commit BEFORE removal
git checkout -b chore/remove-egui
git commit --allow-empty -m "chore: checkpoint — Flutter ready, egui still present

Flutter UI passes B.2 parity for all 14 panels and B.3 critical
paths. Recording this state before egui removal in case rollback
is needed."

# Removal commit
rm -r crates/forex-app/src/ui/
# Remove egui deps from crates/forex-app/Cargo.toml:
#   eframe, egui, egui_dock
# Replace the GUI launcher in crates/forex-app/src/main.rs with
# either a Flutter sidecar launcher OR a no-op stub (headless
# mode must continue to work via the existing --headless flags).
# Remove any egui-specific tests (most live in
# crates/forex-app/src/app_services/trading_tests.rs and don't
# touch egui directly, but spot-check).

cargo build --release  # disk: needs ~20 GB free
cargo test --workspace # expect green

git add -A
git commit -m "chore: remove egui — Flutter is now the sole UI

Per operator directive 2026-05-18, with all Step B parity
gates green. Headless mode (--headless --auto-discovery /
--auto-training) still works."
```

### Headless mode invariant (MUST hold post-removal)

The Rust binary MUST still launch headless for VPS deployment:

```powershell
.\target\release\forex-app.exe --headless --auto-discovery --auto-training
# Should run forever (until Ctrl+C) WITHOUT any GUI window
# WITHOUT Flutter dep WITHOUT eframe dep.
```

Verify this works after the egui removal commit by:
- Running on a Windows server without Flutter installed
- Running inside a Linux container without X11
- Confirming `--headless` flag-handling code path doesn't
  reference eframe

## Παρατηρήσεις από αυτό το session

1. **Flutter SDK δεν εγκαταστάθηκε στο sandbox** — Πραγματικός
   verification απαιτεί operator να εγκαταστήσει Flutter SDK
   και να τρέξει manual από Windows.
2. **Real backend wiring (G8) χρειάζεται για τα περισσότερα
   panels.** Σήμερα ο Rust binary δεν εκθέτει REST surface —
   κάθε panel θα δείχνει mocked data μέχρι να φύγει το G8.
   Αυτό σημαίνει ότι Step B θα έχει πολλά "deferred"
   χαρακτηρισμένα ως ⚠ — αποδεκτό για acceptance gate B.4.2.
3. **OrderSource::AiSuggested + DxTrade Domain UI row** ήδη
   προστέθηκαν στο Rust side. Όταν Flutter wire-up γίνει,
   τα ίδια fields θα εμφανιστούν χωρίς extra change.
4. **gemma-helper feature OFF by default.** Η Flutter chat-box
   panel δεν θα χτυπήσει το Gemma μέχρι ο operator να ενεργοποιήσει
   τη feature στο build flags + να ολοκληρωθεί το download
   του Gemma model (`scripts/fetch-gemma-model.ps1`).

## Πιο σημαντικό

**Egui ΔΕΝ διαγράφεται σήμερα.** Παραμένει 100% λειτουργικό
στο `crates/forex-app/src/ui/` ώστε ο operator να έχει working
GUI ενώ το Flutter περνά Step B από τα production runs.
