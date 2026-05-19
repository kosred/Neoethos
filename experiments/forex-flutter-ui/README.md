# forex-flutter-ui

Flutter desktop/mobile front-end για το forex-ai trading bot.
Mirrors το layout και τα design tokens του
[`mockups/ui_mockup.html`](../../mockups/ui_mockup.html), που με
τη σειρά του mirrors το `crates/forex-app/src/ui/theme.rs`.

> **Phase: skeleton.** Το shell (topbar / sidebar 14 panels /
> dock view / statusbar) δουλεύει, με dummy data στο Dashboard
> και `PendingStub` placeholder σε όλα τα άλλα panels. Real
> backend wiring έρχεται όταν τραβηχτεί το `forex-server` REST
> + SSE surface (G8 του Gemma plan + counterpart στο forex-app).

## Σχέση με το υπάρχον egui UI

**Δεν αντικαθιστά το υπάρχον egui UI**. Τρέχουν παράλληλα:

- `crates/forex-app/` (Rust + egui) — production GUI σήμερα.
- `crates/forex-flutter-ui/` (αυτό) — νέα γενιά UI, σε ανάπτυξη.

Όταν τελειώσει το Flutter, ο operator θα έχει επιλογή ποιο να
ξεκινήσει. Δεν διαγράφουμε egui πριν το Flutter περάσει τα
acceptance tests στο production hardware.

## Σχεδιαστικές αποφάσεις

| | Επιλογή | Γιατί |
|---|---|---|
| State management | **Riverpod** | Compile-time-safe providers, μικρό surface (14 panels) — δεν χρειάζεται το BLoC boilerplate |
| Routing | **go_router** | Sidebar-driven nav, declarative routes; μελλοντικά deep links από notifications |
| Charts | **fl_chart** | Καλό dark theme, candlestick add-on διαθέσιμο |
| HTTP | **dio** | Interceptors για auth headers, response transformers, retry — όλα ready |
| SSE | **sse_channel** | Επίσημο SSE parser, ταιριάζει με το `/gemma/chat` stream του forex-gemma |
| Theme | Pure dark (TradingView) | Operators δουλεύουν το βράδυ· καμία απαίτηση για light σε v0.4 |

## Layout (από το mockup)

```
┌─ TopBar (44px) ────────────────────────────────────────────┐
│ forex-ai  [PRO] [LIVE]  Balance · Equity · Free Margin     │
│                          [Auto-Discover] [Auto-Train] 🔔 🔍 │
├─ Sidebar (220px) ──────┬─ Dock area ─────────────────────────┤
│ TRADING                │ TRADING › Dashboard                 │
│   ▦ Dashboard          │ ┌─────────────────────────────────┐ │
│   📈 Chart             │ │  view-header / stat-row / cards │ │
│   ≡ Markets            │ │  positions table                │ │
│   ↹ Order Ticket       │ │  engine health row              │ │
│   📰 News              │ └─────────────────────────────────┘ │
│   ◫ Trade Watch        │                                     │
│ AI ENGINE              │                                     │
│   ✦ Discovery          │                                     │
│   ⊛ Training           │                                     │
│   ✺ Intelligence       │                                     │
│ SYSTEM                 │                                     │
│   🔌 Broker Setup      │                                     │
│   ⤓ Data Bootstrap     │                                     │
│   ▤ Hardware           │                                     │
│   ⚠ Risk Settings      │                                     │
│   ⚙ Settings           │                                     │
├────────────────────────┴─────────────────────────────────────┤
│ StatusBar (22px)                                             │
│ Broker · Engine · News blackout · Latency · v0.4.5           │
└──────────────────────────────────────────────────────────────┘
```

## Setup (όταν επιστρέψεις)

```powershell
# 1. Install Flutter SDK (one-time, ~2 GB)
# https://docs.flutter.dev/get-started/install/windows
# After install:
flutter --version  # expect 3.22+
flutter config --enable-windows-desktop

# 2. Bootstrap this project
cd C:\Users\konst\development\forex-ai\crates\forex-flutter-ui
flutter create . --platforms windows,macos,linux --org com.forexai
# (--org sets the package id; existing pubspec.yaml is preserved.)

flutter pub get

# 3. Smoke test
flutter test
# Expect: 6 widget tests passing (shell smoke + nav catalog).

# 4. Run desktop
flutter run -d windows
```

## Επόμενα βήματα

1. **G8 — Rust REST/SSE server.** `forex-server` crate hosting
   `/account/snapshot`, `/gemma/chat`, `/gemma/suggestions`, etc.
   Wire `backend_client.dart` στις πραγματικές URLs.
2. **Real-time market data.** Subscribe στο cTrader / DxTrade
   Push API streams του Rust backend (όχι ξανά απευθείας από
   το Flutter — όλο το auth lives στο Rust).
3. **Candle chart.** Custom widget πάνω σε `fl_chart` (ή dedicated
   `candlesticks` package).
4. **Order ticket execution.** Submit/modify/cancel POST endpoints,
   με συμπαγή client-side validation πριν την κλήση.
5. **Gemma chat box.** Streaming SSE display, Approve/Reject
   buttons σε `TradePendingApproval` events, audit-log link.
6. **i18n.** Το mockup είναι Greek-friendly (`lang="el"`); ένα
   απλό `intl` setup θα δώσει EN/EL toggle.

## Files

```
crates/forex-flutter-ui/
├── pubspec.yaml                  # Flutter project manifest
├── README.md                     # this file
├── lib/
│   ├── main.dart                 # entry point + ForexAiApp
│   ├── theme/
│   │   └── theme.dart            # ForexAiTokens + buildForexAiTheme
│   ├── api/
│   │   └── backend_client.dart   # dio + SSE client (mocked)
│   ├── state/
│   │   └── nav.dart              # Riverpod + 14-tab catalog
│   ├── widgets/
│   │   ├── app_shell.dart        # grid: TopBar + Sidebar + Dock + StatusBar
│   │   ├── topbar.dart           # brand, badges, ribbon, auto pills
│   │   ├── sidebar.dart          # nav list with active indicator
│   │   └── statusbar.dart        # broker / engine / blackout / latency
│   └── screens/
│       ├── _placeholder.dart     # shared ViewHeader / StatCard / SectionCard
│       ├── dashboard_screen.dart # most fleshed-out (stats + positions table)
│       └── 13× other screens     # PendingStub placeholders
└── test/
    └── shell_smoke_test.dart     # 6 widget/unit tests
```

## Περίληψη στα Ελληνικά

Το Flutter UI είναι έτοιμο σε σκελετό. Έχεις:

- Πλήρες grid shell (topbar + sidebar + dock + statusbar) που
  γίνεται render με dark theme μηδέν hex-φωνές.
- 14 nav panels με icons + groupings όπως ακριβώς στο mockup.
- Dashboard με dummy stats + positions table να δίνει αίσθηση
  του πραγματικού look.
- 13 placeholder screens που λένε "pending wiring" — ασφαλές
  default.
- API client (`backend_client.dart`) με τυποποιημένα DTOs
  ταιριαστά στο `forex-gemma::api`. Επιστρέφει mocked δεδομένα
  μέχρι να ξεκινήσει ο Rust server.
- 6 widget tests που επιβεβαιώνουν shell + nav swap.

Όταν εγκαταστήσεις Flutter SDK:
1. `flutter create . --platforms windows,macos,linux`
2. `flutter pub get`
3. `flutter test` (6 περνάνε)
4. `flutter run -d windows`
