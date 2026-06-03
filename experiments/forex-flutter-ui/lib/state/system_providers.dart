// Riverpod providers for the system-level endpoints: hardware, risk
// caps, and app settings. These three are read-only Phase 1 — the
// data sources are config.yaml + a CPU/RAM probe, both cheap and
// slow-moving, so a `FutureProvider.autoDispose` is enough. No need
// for the `AsyncNotifier` + polling-timer machinery the
// `accountSnapshotProvider` uses.
//
// `enginesProvider` is the exception: once the user clicks Start on
// Discovery/Training, the status row needs to track progress in real
// time, so it polls every 2 seconds while mounted.

import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import 'account_provider.dart';

/// `/hardware` snapshot. autoDispose so navigating away frees the
/// Future; coming back re-fetches.
final hardwareProvider = FutureProvider.autoDispose<HardwareSnapshot>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchHardware();
});

/// `/risk` snapshot. Same autoDispose pattern.
final riskProvider = FutureProvider.autoDispose<RiskSnapshot>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchRisk();
});

/// `/settings` snapshot.
final settingsProvider = FutureProvider.autoDispose<SettingsSnapshot>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchSettings();
});

/// `/engines/status` — Discovery / Training / Auto-Trader.
///
/// Polls every 2s so the status row + progress summary track the
/// engine in real time after the user hits Start. We schedule the
/// next refetch from inside `onDispose` so the timer dies with the
/// provider (autoDispose still applies — leaving the screen halts
/// polling). The first fetch happens synchronously below.
final enginesProvider = FutureProvider.autoDispose<EnginesSnapshot>((ref) async {
  // **2026-05-26 fix (Κωνσταντίνος)**: schedule the next-tick timer
  // BEFORE the await. Old order (timer after await) silently broke
  // polling when the backend was mid-restart: fetchEngines threw,
  // control left the function, timer never got created → provider
  // wedged in AsyncError forever, UI permanently showed "—".
  final timer = Timer(const Duration(seconds: 2), () {
    ref.invalidateSelf();
  });
  ref.onDispose(timer.cancel);

  final client = ref.read(backendClientProvider);
  return await client.fetchEngines();
});

/// `/broker/status` — current broker session state.
///
/// **2026-05-26 — task #267 fix**: was previously a one-shot
/// `FutureProvider.autoDispose` — cached the COLD-START response forever
/// while the StatusBar was mounted (which is always). If the broker hadn't
/// finished its first cTrader handshake when the UI first read the
/// provider, `connected:false` got cached and the status bar showed
/// "Broker · offline" indefinitely, even though /broker/status was
/// reporting connected:true within a few seconds.
///
/// Fix: schedule a self-invalidation every 3 seconds, mirroring the
/// `enginesProvider` polling pattern. autoDispose still applies — the
/// timer dies with the provider when the StatusBar tree unmounts.
/// 3 s strikes a balance between responsiveness (cold-connect transitions
/// surface within one window) and load (a tiny GET that's already
/// in-process).
final brokerStatusProvider = FutureProvider.autoDispose<BrokerStatus>((ref) async {
  // **2026-05-26 fix #2 (Κωνσταντίνος)**: same trap as enginesProvider —
  // timer must be scheduled BEFORE the await. If the backend restarts
  // (watchdog respawn, OAuth re-auth), the in-flight fetchBrokerStatus
  // throws → control leaves function before timer creation → polling
  // stops forever. Status bar then stuck on "connecting" indefinitely
  // even though backend reports connected:true on subsequent fetches.
  final timer = Timer(const Duration(seconds: 3), () {
    ref.invalidateSelf();
  });
  ref.onDispose(timer.cancel);

  final client = ref.read(backendClientProvider);
  return await client.fetchBrokerStatus();
});

/// `/intelligence` — model artifacts + discovery targets + walkforward.
final intelligenceProvider =
    FutureProvider.autoDispose<IntelligenceSnapshot>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchIntelligence();
});

/// `/news/feed` — AI news desk: public-RSS headlines + a Codex market
/// briefing. autoDispose with a slow 5-minute self-refresh so a long
/// session surfaces fresh headlines; the backend coalesces the actual
/// RSS + Codex fetches behind a ~10-min cache, so the poll is cheap.
/// The panel's refresh button forces a fresh pull (NewsPanel._refresh).
final newsFeedProvider = FutureProvider.autoDispose<NewsFeed>((ref) async {
  final timer = Timer(const Duration(minutes: 5), () => ref.invalidateSelf());
  ref.onDispose(timer.cancel);
  final client = ref.read(backendClientProvider);
  return client.fetchNewsFeed();
});

/// `/risky/scenarios` — Risky/Growth Mode time-to-target projection,
/// computed by the engine. Family-keyed by (starting, target, fraction)
/// — a record, so structural equality keys the cache — so the Growth
/// card re-fetches only when the operator edits those inputs.
final riskyScenariosProvider = FutureProvider.autoDispose.family<RiskyScenario,
    ({double startingUsd, double targetUsd, double riskFraction})>((ref, q) {
  final client = ref.read(backendClientProvider);
  return client.fetchRiskyScenarios(
    startingUsd: q.startingUsd,
    targetUsd: q.targetUsd,
    riskFraction: q.riskFraction,
  );
});

/// `/broker/symbols` — broker-offered catalog. Heavy call (830+ symbols
/// on a typical cTrader account) so kept autoDispose; the UI caches it
/// in a ConsumerStatefulWidget while needed.
final brokerSymbolsProvider =
    FutureProvider.autoDispose<BrokerSymbolsSnapshot>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchBrokerSymbols();
});

/// `/broker/accounts` — the cTIDs the OAuth token grants access to.
/// Drives the Settings → Account dropdown. autoDispose because it's
/// only consumed by the Settings screen which the operator visits
/// rarely; refetched whenever the user reopens the screen.
final brokerAccountsProvider =
    FutureProvider.autoDispose<BrokerAccountsSnapshot>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchBrokerAccounts();
});

/// `/auth/codex/status` — ChatGPT subscription link state. Cheap
/// probe so the News + AI Helper screens know whether to render the
/// chat UI or the "Connect ChatGPT" CTA.
final codexStatusProvider =
    FutureProvider.autoDispose<CodexStatusSnapshot>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchCodexStatus();
});

/// `/broker/timeframes` — canonical cTrader-supported timeframe list,
/// sourced from neoethos_core::CANONICAL_TIMEFRAMES on the server.
/// Tiny (11 strings) and immutable for the life of the binary, so we
/// can use a regular FutureProvider (not autoDispose); first read
/// caches forever within the session.
final brokerTimeframesProvider = FutureProvider<List<String>>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchBrokerTimeframes();
});

/// Selected symbol + timeframe for the Chart screen — primary slot
/// (panel A). Bound to chips the operator clicks; survives screen
/// navigation since these are StateProviders (not autoDispose).
final chartSymbolProvider = StateProvider<String>((ref) => 'EURUSD');
final chartTimeframeProvider = StateProvider<String>((ref) => 'M1');

/// `/chart` snapshot for panel A. Refetches whenever the slot-A
/// symbol or timeframe changes.
final chartProvider =
    FutureProvider.autoDispose<ChartSnapshot>((ref) async {
  final symbol = ref.watch(chartSymbolProvider);
  final tf = ref.watch(chartTimeframeProvider);
  final client = ref.read(backendClientProvider);
  // F-344: 500-bar initial window (was 200). Panning left past the
  // oldest bar pages in MORE from the broker on demand (ProChart
  // onLoadMore → /chart/history), so this is just the starting view,
  // never a cap.
  return client.fetchChart(symbol: symbol, timeframe: tf, limit: 500);
});

// ─── Chart panel B (comparison mode) ────────────────────────────────
//
// Task #116: NeoEthos supports a maximum of TWO charts side-by-side.
// This is a deliberate UX constraint — multi-chart grids (MT5's 16
// tile layout, cTrader's 4-pane "Multi-Chart") are exhausting to
// monitor and bury the trade thesis. Two panels let you compare a
// pair against its correlated index (EURUSD vs DXY, gold vs DXY,
// majors vs their inverse, etc.) without spreading attention thin.
//
// Each panel has its own symbol, timeframe, and active-indicator
// state. We deliberately do NOT auto-sync these — the whole point
// of a second panel is comparison, not duplication.

/// Whether panel B is visible. False on first launch (single-chart
/// mode mirrors the pre-#116 default).
final multiChartEnabledProvider = StateProvider<bool>((ref) => false);

/// Panel-B symbol + timeframe. Default to GBPUSD so the very first
/// time the user toggles comparison on they see two different pairs
/// instead of EURUSD twice (no-op comparison would confuse).
final chartSymbolProviderB = StateProvider<String>((ref) => 'GBPUSD');
final chartTimeframeProviderB = StateProvider<String>((ref) => 'M1');

/// `/chart` snapshot for panel B. Independent of panel A.
final chartProviderB =
    FutureProvider.autoDispose<ChartSnapshot>((ref) async {
  final symbol = ref.watch(chartSymbolProviderB);
  final tf = ref.watch(chartTimeframeProviderB);
  final client = ref.read(backendClientProvider);
  // F-344: 500-bar initial window (was 200). Panning left past the
  // oldest bar pages in MORE from the broker on demand (ProChart
  // onLoadMore → /chart/history), so this is just the starting view,
  // never a cap.
  return client.fetchChart(symbol: symbol, timeframe: tf, limit: 500);
});

/// `/data/bootstrap` — local data-dir inventory.
final dataBootstrapProvider =
    FutureProvider.autoDispose<DataBootstrapSnapshot>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchDataBootstrap();
});

// ─── Chart indicator overlays ────────────────────────────────────────────
//
// The Chart screen lets the user toggle on/off a small set of
// technical indicators that overlay the candlestick canvas.
// Server-side compute (vector_ta) so the math stays in one place
// and the chart pan/zoom only re-fetches when symbol/timeframe
// change — toggling an indicator is a single round-trip.

/// Panel A currently-enabled indicator ids (e.g. `{"sma", "ema"}`).
final activeIndicatorsProvider = StateProvider<Set<String>>((ref) => {});

/// Panel B currently-enabled indicator ids. Independent of A — the
/// whole point of panel B is to compare with different settings.
final activeIndicatorsProviderB = StateProvider<Set<String>>((ref) => {});

/// Per-indicator fetch for panel A. Family-keyed by indicator id;
/// refetches whenever the slot-A symbol or timeframe changes.
final indicatorProvider = FutureProvider.autoDispose
    .family<IndicatorSnapshot, String>((ref, indicatorName) {
  final symbol = ref.watch(chartSymbolProvider);
  final tf = ref.watch(chartTimeframeProvider);
  final client = ref.read(backendClientProvider);
  return client.fetchIndicator(
    symbol: symbol,
    timeframe: tf,
    indicator: indicatorName,
    limit: 200,
  );
});

/// Per-indicator fetch for panel B. Same shape, different state.
final indicatorProviderB = FutureProvider.autoDispose
    .family<IndicatorSnapshot, String>((ref, indicatorName) {
  final symbol = ref.watch(chartSymbolProviderB);
  final tf = ref.watch(chartTimeframeProviderB);
  final client = ref.read(backendClientProvider);
  return client.fetchIndicator(
    symbol: symbol,
    timeframe: tf,
    indicator: indicatorName,
    limit: 200,
  );
});
