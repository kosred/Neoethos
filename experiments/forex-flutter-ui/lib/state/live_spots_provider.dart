// Riverpod state for the cTrader live spot stream (#137 / #237).
//
// **2026-05-25 - task #237 SSE migration**: rewritten from the
// previous `Timer.periodic(Duration(seconds: 1))` polling shape to
// consume the `/live/spots/stream` SSE endpoint. End-to-end latency
// drops from ~1 s -> ~5 ms (the backend broadcast channel hop).
// Chart current-candle overlay, position PnL, and the live-price
// ticker now update at near tick-rate (50-200 ms typical for active
// majors) without burning CPU on every Flutter frame.
//
// Reconnect / lifecycle behaviour (research-derived):
//   - Exponential backoff: 500 ms -> 1 s -> 2 s -> ... -> cap 30 s.
//   - `Last-Event-ID` header on every reconnect.
//   - `ref.keepAlive()` so screen switches don't kill the stream.
//   - One `EventFlux.spawn()` per stream (NEVER the singleton).
//
// Multi-consumer note: chart + markets + trade_watch all watch this
// provider. Riverpod's caching ensures they share ONE SSE
// connection, not three.

import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../api/sse_client.dart';
import 'account_provider.dart';

class LiveSpotsNotifier extends AsyncNotifier<LiveSpotsSnapshot> {
  SseSubscription<LiveSpotTick>? _sub;
  bool _disposed = false;

  @override
  Future<LiveSpotsSnapshot> build() async {
    // App-tied connection so screen-to-screen navigation doesn't
    // tear down the stream + open a new one (the old polling
    // notifier was autoDispose precisely to avoid that, but SSE
    // is cheap to keep open continuously and reconnect cost dwarfs
    // the screen-switch frequency anyway).
    ref.keepAlive();
    ref.onDispose(() {
      _disposed = true;
      _sub?.disconnect();
      _sub = null;
    });

    final client = ref.read(backendClientProvider);
    final baseUrl = client.baseUrl;

    // Priming GET so the UI gets ticks RIGHT AWAY (the SSE warmup
    // takes ~50-200 ms; for the chart screen that's noticeable).
    LiveSpotsSnapshot? initial;
    try {
      initial = await client.fetchLiveSpots();
    } catch (_) {
      // SSE will fill in shortly.
      initial = null;
    }

    _sub = SseSubscription<LiveSpotTick>(
      config: SseConfig(
        url: '$baseUrl/live/spots/stream',
        tag: 'live-spots',
      ),
      parse: (json) => LiveSpotTick.fromJson(json),
      onEvent: (tick) {
        if (_disposed) return;
        final current = state.valueOrNull ?? LiveSpotsSnapshot.empty();
        state = AsyncData(current.mergeTick(tick));
      },
      onError: (e, st) {
        if (_disposed) return;
        state = AsyncError(e, st);
      },
    );
    _sub!.connect();

    if (initial != null) return initial;

    final completer = Completer<LiveSpotsSnapshot>();
    final subscription = _sub!.events.listen((tick) {
      if (!completer.isCompleted) {
        completer.complete(LiveSpotsSnapshot.empty().mergeTick(tick));
      }
    });
    // 10 s timeout so the UI doesn't hang forever if the SSE is
    // unreachable. After timeout we THROW (mirrors account_provider) —
    // this puts the provider into AsyncError so the consuming screens
    // render an explicit "prices unavailable / reconnecting" state.
    //
    // Returning LiveSpotsSnapshot.empty() here (the old behaviour) was
    // a silent-blank-prices bug: a stalled backend looked identical to
    // a closed market — Chart / Market Watch / PnL showed ZERO prices
    // with no error surfaced. Every consumer is safe with the throw:
    // _WatchlistPanel has an error: branch (marketWatchPricesUnavailable),
    // and the .valueOrNull readers (chart overlay, inline buy/sell,
    // summary strip) already treat the absence of a value as "no tick
    // yet" and fall back / render nothing rather than crash.
    final result = await completer.future.timeout(
      const Duration(seconds: 10),
      onTimeout: () => throw const BrokerNotReadyException(
          'live spots stream did not deliver an initial event within 10 s'),
    );
    await subscription.cancel();
    return result;
  }
}

final liveSpotsProvider =
    AsyncNotifierProvider<LiveSpotsNotifier, LiveSpotsSnapshot>(
  LiveSpotsNotifier.new,
);
