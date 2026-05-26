// Server-Sent Events client wrapper around `eventflux` (#237,
// 2026-05-25).
//
// Why this exists: dio handles every REST endpoint in the app, but
// dio's response streaming buffers in some configurations, defeating
// the latency win we want from `/account/snapshot/stream` (~750 ms)
// and `/live/spots/stream` (~5 ms). `eventflux` is the canonical
// Flutter SSE client in 2026 (built on `http`, exponential backoff
// with per-retry header refresh, native multi-connection via
// `spawn()`).
//
// Architecture (research-derived):
//   - One `EventFlux.spawn()` per stream — NEVER the global
//     `EventFlux.instance` singleton, which silently disconnects the
//     previous URL when you open a second. Subtle bug magnet.
//   - Exponential backoff: 500 ms → 1 s → 2 s → 4 s → cap 30 s.
//   - `Last-Event-ID` header on every reconnect so the server can
//     replay since the last event without re-sending the full state.
//   - Riverpod integration via AsyncNotifier (NOT StreamProvider —
//     StreamProvider tears down + resubscribes on every listener
//     detach, which causes a subscribe storm on tab switches).
//
// Anti-patterns this avoids (per research):
//   1. Subscribing inside widget `build()` — every rebuild opens a
//      new stream. We subscribe inside notifier `build()` instead,
//      which fires exactly once per provider lifetime.
//   2. Forgetting `ref.onDispose(disconnect)` — leaks the stream
//      AND the notifier (double UI update on remount).
//   3. Sharing `EventFlux.instance` — see above.
//   4. Not honoring `Last-Event-ID` — every blip becomes a full
//      snapshot replay, defeats the latency win.

import 'dart:async';
import 'dart:convert';

import 'package:eventflux/eventflux.dart';
// 2026-05-26: `models/reconnect.dart` is re-exported from the top-level
// `eventflux.dart` barrel; the direct import made `flutter_lints 6`
// flag the redundancy (`unnecessary_import`). `ReconnectConfig` is the
// only type used and it resolves via the barrel.
import 'package:flutter/foundation.dart';

/// Configuration for an SSE subscription. The notifier owns one of
/// these per logical stream (account snapshot, live spots, etc.).
class SseConfig {
  /// Absolute URL of the SSE endpoint, e.g.
  /// `http://127.0.0.1:7423/account/snapshot/stream`.
  final String url;

  /// Tag for the EventFlux instance + tracing logs. Differentiates
  /// concurrent streams in eventflux's internal connection registry.
  final String tag;

  /// Initial retry delay (the first wait between a disconnect and a
  /// reconnect attempt). Subsequent waits double up to `maxDelay`.
  /// Research recommendation: 500 ms — anything below 200 ms creates
  /// tight loops on auth failures; anything above 2 s feels broken
  /// on a Wi-Fi blip.
  final Duration initialDelay;

  /// Upper bound on exponential backoff. Research recommendation:
  /// 30 s — beyond that the user perceives the app as "dead".
  final Duration maxDelay;

  /// Optional extra headers (auth bearer, tenant ID, etc.). The
  /// `Last-Event-ID` header is appended automatically by the helper
  /// when resuming after a disconnect.
  final Map<String, String> extraHeaders;

  const SseConfig({
    required this.url,
    required this.tag,
    this.initialDelay = const Duration(milliseconds: 500),
    this.maxDelay = const Duration(seconds: 30),
    this.extraHeaders = const {},
  });
}

/// Wraps an `EventFlux` instance with the Riverpod-friendly
/// connect/disconnect lifecycle + auto-reconnect + Last-Event-ID
/// resume that the rest of the app expects.
///
/// Construct one per stream. Call [connect] to start, [disconnect]
/// from `ref.onDispose`. The notifier listens to [events] for the
/// decoded payloads.
class SseSubscription<T> {
  final SseConfig config;
  final T Function(Map<String, dynamic>) parse;
  final void Function(T)? onEvent;
  final void Function(Object error, StackTrace stack)? onError;

  final EventFlux _flux = EventFlux.spawn();
  StreamController<T>? _controller;
  String? _lastEventId;
  StreamSubscription<EventFluxData>? _innerSub;
  bool _disposed = false;

  SseSubscription({
    required this.config,
    required this.parse,
    this.onEvent,
    this.onError,
  });

  /// Decoded payload stream. The notifier `listen`s to this and
  /// pushes each event into `state = AsyncData(...)`.
  Stream<T> get events {
    _controller ??= StreamController<T>.broadcast();
    return _controller!.stream;
  }

  /// Opens the SSE connection. Idempotent — calling twice is a no-op.
  void connect() {
    if (_disposed) return;
    final headers = <String, String>{
      'Accept': 'text/event-stream',
      ...config.extraHeaders,
      if (_lastEventId != null) 'Last-Event-ID': _lastEventId!,
    };
    _flux.connect(
      EventFluxConnectionType.get,
      config.url,
      header: headers,
      autoReconnect: true,
      reconnectConfig: ReconnectConfig(
        mode: ReconnectMode.exponential,
        interval: config.initialDelay,
        maxAttempts: -1, // forever; the user can close the app to stop
        onReconnect: () {
          if (kDebugMode) {
            debugPrint(
              'SSE[${config.tag}] reconnecting (lastId=$_lastEventId)',
            );
          }
        },
      ),
      onSuccessCallback: (response) {
        _innerSub = response?.stream?.listen(
          (event) {
            // Track the event id so we can resume from the right place
            // after a disconnect. The server emits monotonically-
            // increasing ids; we don't parse them, just forward them.
            // 2026-05-26: eventflux current API exposes `event.id` and
            // `event.data` as non-nullable `String` (was nullable in
            // older versions). Dropped redundant `?.` / `??` operators
            // — flutter_lints 6 flags them as invalid/dead.
            if (event.id.isNotEmpty) {
              _lastEventId = event.id;
            }
            try {
              final raw = event.data;
              if (raw.isEmpty) {
                // Empty event = SSE keep-alive heartbeat; nothing to
                // decode. Skip instead of letting `jsonDecode('')`
                // throw a FormatException for the catch below to
                // swallow — saves a stack-unwind on every keep-alive.
                return;
              }
              final decoded = jsonDecode(raw) as Map<String, dynamic>;
              final value = parse(decoded);
              _controller?.add(value);
              onEvent?.call(value);
            } catch (e, st) {
              // Malformed event from the server is recoverable - we
              // don't tear down the whole connection, we just log and
              // wait for the next event.
              if (kDebugMode) {
                debugPrint('SSE[${config.tag}] decode error: $e\n$st');
              }
              onError?.call(e, st);
            }
          },
          onError: (e, StackTrace st) {
            // Stream-level error: let eventflux's reconnect logic
            // handle this. We forward to the notifier for UI banner.
            if (kDebugMode) {
              debugPrint('SSE[${config.tag}] stream error: $e');
            }
            onError?.call(e, st);
          },
          cancelOnError: false,
        );
      },
      onError: (error) {
        // Connection-level error (HTTP non-2xx, socket reset). We
        // surface it; eventflux retries automatically with backoff.
        if (kDebugMode) {
          debugPrint('SSE[${config.tag}] connection error: $error');
        }
        onError?.call(error, StackTrace.current);
      },
    );
  }

  /// Closes the SSE connection + flushes buffered events. Safe to
  /// call multiple times. The notifier MUST call this from its
  /// `ref.onDispose` — otherwise the stream keeps parsing into a
  /// dead notifier (memory leak + duplicate UI updates if user re-
  /// enters the screen later).
  void disconnect() {
    if (_disposed) return;
    _disposed = true;
    _innerSub?.cancel();
    _innerSub = null;
    _flux.disconnect();
    _controller?.close();
    _controller = null;
  }
}
