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
  final client = ref.read(backendClientProvider);
  final snapshot = await client.fetchEngines();

  // Schedule next tick. A one-shot Timer + invalidateSelf gives us
  // "poll, await, repeat" — each invalidation re-runs this whole
  // body, including scheduling the *next* timer. `onDispose` cancels
  // the pending timer so navigating away halts polling cleanly.
  final timer = Timer(const Duration(seconds: 2), () {
    ref.invalidateSelf();
  });
  ref.onDispose(timer.cancel);

  return snapshot;
});

/// `/broker/status` — current broker session state.
final brokerStatusProvider = FutureProvider.autoDispose<BrokerStatus>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchBrokerStatus();
});

/// `/intelligence` — model artifacts + discovery targets + walkforward.
final intelligenceProvider =
    FutureProvider.autoDispose<IntelligenceSnapshot>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchIntelligence();
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

/// `/gemma/status` — local LLM availability. Cheap probe so the
/// News + AI Helper screens know whether to render the chat UI or
/// the "install model" instructions.
final gemmaStatusProvider =
    FutureProvider.autoDispose<GemmaStatusSnapshot>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchGemmaStatus();
});

/// `/gemma/download/status` — drives the AI Helper screen's
/// progress bar while the GGUF is being fetched. Self-invalidates
/// every second while state == "downloading" so the bar advances
/// in real time; goes quiet once the download terminates
/// (completed / failed / cancelled / idle) so we don't keep
/// hammering the endpoint pointlessly.
final gemmaDownloadStatusProvider =
    FutureProvider.autoDispose<GemmaDownloadStatus>((ref) async {
  final client = ref.read(backendClientProvider);
  final snapshot = await client.fetchGemmaDownloadStatus();
  if (snapshot.isDownloading) {
    final timer = Timer(const Duration(seconds: 1), () {
      ref.invalidateSelf();
    });
    ref.onDispose(timer.cancel);
  }
  return snapshot;
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
  return client.fetchChart(symbol: symbol, timeframe: tf, limit: 200);
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
  return client.fetchChart(symbol: symbol, timeframe: tf, limit: 200);
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
