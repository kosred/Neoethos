// Riverpod state for the cTrader live spot stream (#137).
//
// Polls `/live/spots` every 1 s so any consumer (chart
// current-candle overlay, trade-watch PnL, live-price ticker)
// gets sub-2 s freshness via a simple `.watch` on this provider.
//
// The backend streamer pushes ticks into its own cache at
// whatever cadence cTrader sends them (50–200 ms typical for
// active majors). 1 s polling here is fast enough to feel
// "live" in the UI without burning CPU on every Flutter frame.
//
// The provider is `autoDispose` so navigating off any consuming
// screen halts the timer. Multiple screens watching at the same
// time share the same provider instance via Riverpod's caching,
// so we don't end up with 3 parallel poll loops if e.g. Chart +
// Markets + TradeWatch all subscribe simultaneously.

import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import 'account_provider.dart';

class LiveSpotsNotifier extends AsyncNotifier<LiveSpotsSnapshot> {
  Timer? _timer;
  bool _disposed = false;

  @override
  Future<LiveSpotsSnapshot> build() async {
    ref.onDispose(() {
      _disposed = true;
      _timer?.cancel();
      _timer = null;
    });

    _timer ??= Timer.periodic(
      const Duration(seconds: 1),
      (_) => _refresh(),
    );

    return _fetchOnce();
  }

  Future<LiveSpotsSnapshot> _fetchOnce() async {
    final client = ref.read(backendClientProvider);
    return client.fetchLiveSpots();
  }

  Future<void> _refresh() async {
    final next = await AsyncValue.guard(_fetchOnce);
    if (_disposed) return;
    state = next;
  }
}

final liveSpotsProvider =
    AsyncNotifierProvider<LiveSpotsNotifier, LiveSpotsSnapshot>(
  LiveSpotsNotifier.new,
);
