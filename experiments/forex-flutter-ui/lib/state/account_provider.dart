// Riverpod state for the account snapshot.
//
// One `AsyncNotifierProvider` that owns:
//   - A long-lived `BackendClient` (dio under the hood — reuses the
//     connection across calls).
//   - A 5-second polling timer that refreshes the snapshot in the
//     background. 5s matches the Rust bridge's cTrader refresh cadence
//     (see `crates/neoethos-app/src/server/bridge.rs::REFRESH_INTERVAL`);
//     polling faster than the bridge would burn CPU for no new data.
//
// The `AsyncValue<AccountSnapshot>` state machine handles three UI
// surfaces automatically:
//   - `loading` (first ever fetch in flight) → skeletons
//   - `data` (any successful response since last refresh) → live numbers
//   - `error` (network blip or BrokerNotReadyException) → banner
//
// Critically, when a periodic refresh fails we DO NOT clobber the last
// good data — `AsyncValue.guard` produces a new error state but the
// previous data remains accessible via `state.valueOrNull` so the UI
// can keep showing the last-known numbers instead of a flicker.

import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';

/// Singleton-per-provider-scope dio client. Exposed as a Provider so
/// tests can override it with a mock without going through the whole
/// `AccountSnapshotNotifier.build()` reconstruction.
final backendClientProvider = Provider<BackendClient>((ref) {
  return BackendClient();
});

class AccountSnapshotNotifier extends AsyncNotifier<AccountSnapshot> {
  Timer? _timer;
  bool _disposed = false;

  @override
  Future<AccountSnapshot> build() async {
    // Cancel the timer + flip the dispose flag when the provider is
    // disposed (route change, app shutdown). Without this we'd leak a
    // Timer per provider rebuild and the dio client would keep
    // hammering the server. We avoid `ref.mounted` because that API
    // landed in Riverpod 2.6 and pubspec.yaml floors at 2.5.1.
    ref.onDispose(() {
      _disposed = true;
      _timer?.cancel();
      _timer = null;
    });

    // Schedule the periodic refresh as soon as the first fetch starts.
    // The closure uses `ref.read` not `ref.watch` because we don't want
    // the timer callback to re-subscribe the notifier to itself.
    _timer ??= Timer.periodic(const Duration(seconds: 5), (_) => _refresh());

    return _fetchOnce();
  }

  Future<AccountSnapshot> _fetchOnce() async {
    final client = ref.read(backendClientProvider);
    return client.fetchAccountSnapshot();
  }

  Future<void> _refresh() async {
    // `AsyncValue.guard` automatically wraps in loading/data/error and
    // catches every exception (including BrokerNotReadyException) into
    // the error variant. We never need a try/catch around it.
    final next = await AsyncValue.guard(_fetchOnce);
    // If the provider was disposed mid-fetch, drop the result silently.
    if (_disposed) return;
    state = next;
  }

  /// Manual refresh trigger for a "pull to refresh" gesture or for
  /// tests that want to skip the 5-second wait.
  Future<void> refreshNow() async {
    state = const AsyncValue.loading();
    final next = await AsyncValue.guard(_fetchOnce);
    if (_disposed) return;
    state = next;
  }
}

final accountSnapshotProvider =
    AsyncNotifierProvider<AccountSnapshotNotifier, AccountSnapshot>(
  AccountSnapshotNotifier.new,
);
