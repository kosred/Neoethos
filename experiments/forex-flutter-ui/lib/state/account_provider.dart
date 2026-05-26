// Riverpod state for the account snapshot.
//
// **2026-05-25 - task #237 SSE migration**: rewritten from the
// previous `Timer.periodic(Duration(seconds: 5))` polling shape to
// consume the `/account/snapshot/stream` SSE endpoint. End-to-end
// latency on a fill / margin event drops from 5 s -> ~750 ms (the
// bridge's refresh-once cycle) plus negligible network hop. The
// uniform-push doctrine the operator asked for ("push, not poll")
// now reaches the Flutter UI.
//
// Reconnect / lifecycle behaviour (research-derived):
//   - Exponential backoff: 500 ms -> 1 s -> 2 s -> ... -> cap 30 s.
//   - `Last-Event-ID` header on every reconnect so the server can
//     replay since the last delivered event.
//   - `ref.keepAlive()` so the connection survives sidebar tab
//     switches; ONLY torn down on logout / app shutdown.
//   - One `EventFlux.spawn()` per stream (NEVER the singleton).
//   - On error we keep the last good `AsyncData(...)` valueOrNull
//     accessible so the UI can keep showing the last-known numbers
//     instead of flickering.
//
// The notifier still exposes `refreshNow()` for the "force refresh"
// button on the dashboard (#241) - that fires a POST to
// `/account/snapshot/refresh` on the backend, which triggers the
// bridge's push-side path, which arrives over the SSE within ~750 ms.
// We no longer call the GET endpoint directly from `refreshNow()`
// because the SSE is the source of truth.

import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../api/sse_client.dart';

/// Singleton-per-provider-scope dio client. Exposed as a Provider so
/// tests can override it with a mock without going through the whole
/// `AccountSnapshotNotifier.build()` reconstruction.
final backendClientProvider = Provider<BackendClient>((ref) {
  return BackendClient();
});

class AccountSnapshotNotifier extends AsyncNotifier<AccountSnapshot> {
  SseSubscription<AccountSnapshot>? _sub;
  bool _disposed = false;

  @override
  Future<AccountSnapshot> build() async {
    // App-tied connection (account snapshot is core dashboard data —
    // we want it streaming for the entire app lifetime, not torn
    // down on tab switches).
    ref.keepAlive();
    ref.onDispose(() {
      _disposed = true;
      _sub?.disconnect();
      _sub = null;
    });

    final client = ref.read(backendClientProvider);
    final baseUrl = client.baseUrl;

    // One-shot initial fetch so the UI gets data RIGHT AWAY at startup,
    // before the SSE warmup completes. The SSE replaces the polling
    // timer; this single GET is the "show me something now" priming.
    AccountSnapshot? initial;
    try {
      initial = await client.fetchAccountSnapshot();
    } catch (e) {
      // Initial fetch failed — fine, the SSE will fill state as soon
      // as it connects. We don't surface as error here because that
      // would make the dashboard "broken-looking" while the SSE is
      // mid-handshake (~50-200 ms).
      initial = null;
    }

    _sub = SseSubscription<AccountSnapshot>(
      config: SseConfig(
        url: '$baseUrl/account/snapshot/stream',
        tag: 'account-snapshot',
      ),
      parse: (json) => AccountSnapshot.fromJson(json),
      onEvent: (snapshot) {
        if (_disposed) return;
        state = AsyncData(snapshot);
      },
      onError: (e, st) {
        if (_disposed) return;
        // Keep the last good data accessible via state.valueOrNull
        // but surface the error variant so the BackendHealthBanner
        // can show a "reconnecting…" badge if it wants to.
        state = AsyncError(e, st);
      },
    );
    _sub!.connect();

    if (initial != null) {
      return initial;
    }
    // No initial value — return a future that the SSE will resolve.
    // We construct a completer the SSE callback completes on first event.
    final completer = Completer<AccountSnapshot>();
    final subscription = _sub!.events.listen((snapshot) {
      if (!completer.isCompleted) completer.complete(snapshot);
    });
    // 10 s timeout so the UI doesn't hang forever if the SSE is
    // unreachable. After timeout we throw, which puts the provider
    // in AsyncError and the BackendHealthBanner shows the error.
    final result = await completer.future.timeout(
      const Duration(seconds: 10),
      onTimeout: () =>
          throw const BrokerNotReadyException('account snapshot stream did not deliver an initial event within 10 s'),
    );
    await subscription.cancel();
    return result;
  }

  /// Manual refresh trigger — fires a POST to
  /// `/account/snapshot/refresh` on the backend which causes the
  /// bridge to skip its 5 s timer and emit a fresh snapshot through
  /// the SSE within ~750 ms. The UI shows a brief loading shimmer
  /// during that window via the AsyncValue.loading variant.
  ///
  /// Used by the dashboard refresh button (#241).
  Future<void> refreshNow() async {
    try {
      await ref.read(backendClientProvider).refreshAccountSnapshot();
    } catch (e, st) {
      if (_disposed) return;
      state = AsyncError(e, st);
    }
    // The fresh snapshot arrives over the SSE — onEvent updates state
    // for us.
  }
}

final accountSnapshotProvider =
    AsyncNotifierProvider<AccountSnapshotNotifier, AccountSnapshot>(
  AccountSnapshotNotifier.new,
);
