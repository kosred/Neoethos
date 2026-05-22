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

/// Selected symbol + timeframe for the Chart screen. Bound to chips
/// the operator clicks; survives screen navigation since these are
/// NotifierProviders (not autoDispose).
final chartSymbolProvider = StateProvider<String>((ref) => 'EURUSD');
final chartTimeframeProvider = StateProvider<String>((ref) => 'M1');

/// `/chart` snapshot. Family over (symbol, tf) so the user can flip
/// chips and the cache holds.
final chartProvider =
    FutureProvider.autoDispose<ChartSnapshot>((ref) async {
  final symbol = ref.watch(chartSymbolProvider);
  final tf = ref.watch(chartTimeframeProvider);
  final client = ref.read(backendClientProvider);
  return client.fetchChart(symbol: symbol, timeframe: tf, limit: 200);
});

/// `/data/bootstrap` — local data-dir inventory.
final dataBootstrapProvider =
    FutureProvider.autoDispose<DataBootstrapSnapshot>((ref) {
  final client = ref.read(backendClientProvider);
  return client.fetchDataBootstrap();
});
