// Riverpod providers for the system-level endpoints: hardware, risk
// caps, and app settings. These three are read-only Phase 1 — the
// data sources are config.yaml + a CPU/RAM probe, both cheap and
// slow-moving, so a `FutureProvider.autoDispose` is enough. No need
// for the `AsyncNotifier` + polling-timer machinery the
// `accountSnapshotProvider` uses.

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
