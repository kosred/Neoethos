// Riverpod state for the LLM-proposed pending actions queue (#136).
//
// Polls `/actions/pending` every 2 s so a freshly-proposed action
// (banner widget downstream) is visible to the operator within
// ~2 s of the LLM calling `propose_close_position`. The Rust queue
// is bounded at 16 entries with a 60 s TTL + 24 h history prune,
// so the response is always small.
//
// One AsyncNotifier so the banner can call `refreshNow()` after
// the operator clicks Confirm / Reject and see the new state
// without waiting for the next 2 s tick.

import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import 'account_provider.dart';

class PendingActionsNotifier extends AsyncNotifier<List<PendingAction>> {
  Timer? _timer;
  bool _disposed = false;

  @override
  Future<List<PendingAction>> build() async {
    ref.onDispose(() {
      _disposed = true;
      _timer?.cancel();
      _timer = null;
    });

    // 2 s cadence — fast enough that operator sees a brand-new
    // proposal within one heartbeat, slow enough that the backend
    // isn't getting hammered for the (usually empty) list.
    _timer ??= Timer.periodic(
      const Duration(seconds: 2),
      (_) => _refresh(),
    );

    return _fetchOnce();
  }

  Future<List<PendingAction>> _fetchOnce() async {
    final client = ref.read(backendClientProvider);
    return client.fetchPendingActions();
  }

  Future<void> _refresh() async {
    final next = await AsyncValue.guard(_fetchOnce);
    if (_disposed) return;
    state = next;
  }

  /// Force-refresh after a Confirm / Reject click so the banner
  /// reflects the post-click state immediately instead of waiting
  /// for the next 2 s tick.
  Future<void> refreshNow() async {
    final next = await AsyncValue.guard(_fetchOnce);
    if (_disposed) return;
    state = next;
  }
}

final pendingActionsProvider =
    AsyncNotifierProvider<PendingActionsNotifier, List<PendingAction>>(
  PendingActionsNotifier.new,
);
