// BackendWatchdog — periodic /healthz poll + auto-respawn.
//
// Why this exists:
//   The Rust backend (`neoethos-app --server`) is a long-running
//   process spawned ONCE by `BackendSupervisor.ensureRunning()` at
//   Flutter startup. When it dies mid-session (OOM, panic, kill -9,
//   Windows update reboots the box, etc.) Flutter has zero feedback —
//   every subsequent dio call just returns
//   `DioException [unknown]: null` / `HttpException: Connection
//   closed before full header was received`. The operator sees a
//   blank Dashboard and has to manually relaunch the app.
//
// Behaviour:
//   1. Poll `GET /healthz` every 3 seconds with a 2 s per-request
//      timeout (matches the supervisor's existing probe contract).
//   2. After 3 consecutive failures call
//      `BackendSupervisor.instance.restartBackend()`.
//   3. Throttle respawns: minimum 15 seconds between consecutive
//      respawn attempts, even if `/healthz` keeps failing. This
//      prevents thrashing if the backend immediately dies on every
//      spawn (e.g. corrupted user-data dir, port 7423 held by a
//      stale process the supervisor can't see).
//   4. While a respawn is in flight, suspend the polling Timer.
//      Probing during spawn is a race — the freshly-killed process
//      reliably trips the failure counter again and we'd never
//      escape "reconnecting".
//
// State exposed:
//   `BackendHealthState` — online | reconnecting | offline + counters
//   for the UI. Consumed by:
//     - the red banner in `app_shell.dart` (top-of-screen)
//     - the Diagnostics dialog (last seen / attempts)
//
// Logging discipline:
//   Successful polls in the steady state are silent (we'd be writing
//   one line every 3 s otherwise — 1200 lines/hour). We log every
//   STATE TRANSITION via `dart:developer.log()` so the supervisor.log
//   is the canonical "why did the backend bounce" record without
//   drowning in noise.

import 'dart:async';
import 'dart:developer' as developer;

import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'backend_supervisor.dart';

/// Coarse connectivity state for the UI.
enum BackendHealthStatus {
  /// `/healthz` answered 200 on the most recent probe.
  online,

  /// At least one probe has failed and we're either counting toward
  /// the respawn threshold OR a respawn is currently in flight.
  /// Banner colour: red. The UI does NOT distinguish "1 failure" from
  /// "respawn underway" — both look the same to the operator and
  /// both warrant the same "your backend is wobbly" signal.
  reconnecting,
}

/// Immutable snapshot Riverpod publishes to listeners.
class BackendHealthState {
  final BackendHealthStatus status;

  /// Wall-clock time of the most recent successful `/healthz`. Null
  /// if we've never seen the backend (shouldn't happen given
  /// `ensureRunning()` runs first, but defensive).
  final DateTime? lastSeenAt;

  /// How many times the watchdog has triggered a respawn so far in
  /// this session. Shown in Diagnostics so the operator can tell
  /// "this happens once every couple of hours" from "this is
  /// thrashing every 20 seconds".
  final int respawnAttempts;

  /// Count of consecutive `/healthz` failures since the last
  /// success. Resets to 0 the moment a probe returns 200. Bounded
  /// by the respawn threshold — once we trigger a respawn the
  /// counter resets so the banner can show "1 failure" again
  /// post-respawn if the new process is also sick.
  final int consecutiveFailures;

  const BackendHealthState({
    required this.status,
    required this.lastSeenAt,
    required this.respawnAttempts,
    required this.consecutiveFailures,
  });

  /// Initial state used by `build()`. We assume the supervisor's
  /// `ensureRunning()` already brought the backend up (main.dart
  /// awaits it before runApp), so `online` is the truthful default;
  /// the first watchdog tick 3 s later confirms.
  factory BackendHealthState.initial() => BackendHealthState(
        status: BackendHealthStatus.online,
        lastSeenAt: DateTime.now(),
        respawnAttempts: 0,
        consecutiveFailures: 0,
      );

  BackendHealthState copyWith({
    BackendHealthStatus? status,
    DateTime? lastSeenAt,
    int? respawnAttempts,
    int? consecutiveFailures,
  }) {
    return BackendHealthState(
      status: status ?? this.status,
      lastSeenAt: lastSeenAt ?? this.lastSeenAt,
      respawnAttempts: respawnAttempts ?? this.respawnAttempts,
      consecutiveFailures: consecutiveFailures ?? this.consecutiveFailures,
    );
  }

  /// `true` when the banner should be rendered. Single source of
  /// truth for "is the UI in degraded mode" — keeps the banner
  /// widget's `build()` a one-liner.
  bool get isDegraded => status != BackendHealthStatus.online;

  @override
  String toString() => 'BackendHealthState(status: $status, lastSeenAt: '
      '$lastSeenAt, respawnAttempts: $respawnAttempts, '
      'consecutiveFailures: $consecutiveFailures)';
}

/// Riverpod notifier that drives the polling Timer and publishes
/// state changes. Held by `backendHealthProvider`.
///
/// Tunables:
/// - `_pollInterval`: 3 s between probes. The endpoint costs <50 ms
///   server-side and the operator wants <10 s detection of a crash.
///   3 s × 3 failures = ~9 s worst-case detection, ~6 s typical.
/// - `_perRequestTimeout`: 2 s — Loopback latency is microseconds,
///   so anything past 2 s is "the kernel is queueing because the
///   server isn't accepting".
/// - `_failureThreshold`: 3 consecutive failures before respawn.
///   Tolerates a single GC pause / IO stall blip without
///   nuke-from-orbit; still detects a real crash within ~9 s.
/// - `_respawnMinInterval`: 15 s between consecutive respawns. If
///   the backend immediately dies on every spawn (corrupt state,
///   port held by zombie) we'd otherwise hammer the OS at 9 s
///   intervals. 15 s is long enough that the operator notices the
///   banner and short enough that a transient OS issue resolves
///   within ~2 attempts.
class BackendWatchdog extends Notifier<BackendHealthState> {
  static const Duration _pollInterval = Duration(seconds: 3);
  static const Duration _perRequestTimeout = Duration(seconds: 2);
  static const int _failureThreshold = 3;
  static const Duration _respawnMinInterval = Duration(seconds: 15);

  Timer? _timer;

  /// Reentrancy guard. `Timer.periodic` ticks on the wall clock —
  /// if a probe runs long (say, the kernel queues a SYN for 1.9 s
  /// before timing out) the next tick can fire before the previous
  /// completes. We drop overlapping probes to keep the failure
  /// counter monotonic.
  bool _probing = false;

  /// Set to `true` while `restartBackend()` is in flight so the
  /// timer's `_tick` callback short-circuits. Without this the
  /// polling Timer would fight the spawn — probing port 7423 while
  /// the supervisor is killing the old process and starting the
  /// new one produces guaranteed failures that just inflate
  /// `consecutiveFailures` and trigger a second respawn before the
  /// first finished.
  bool _respawning = false;

  /// Wall-clock of the last respawn. Used to enforce
  /// `_respawnMinInterval`. Null until we've respawned at least
  /// once this session.
  DateTime? _lastRespawnAt;

  @override
  BackendHealthState build() {
    // ref.onDispose handles hot-reload + tear-down. Riverpod's
    // ProviderScope is recreated on hot restart; without this the
    // old Timer keeps polling against the new app's backend.
    ref.onDispose(() {
      developer.log(
        'BackendWatchdog disposed — cancelling poll timer.',
        name: 'backend_watchdog',
      );
      _timer?.cancel();
      _timer = null;
    });

    // Start polling immediately. The first probe runs 3 s from now
    // (Timer.periodic fires AFTER the interval, not at t=0) which
    // is intentional — `ensureRunning()` just confirmed health a
    // moment ago in main.dart, no need to re-probe instantly.
    _timer ??= Timer.periodic(_pollInterval, (_) => _tick());

    developer.log(
      'BackendWatchdog started. interval=${_pollInterval.inSeconds}s '
      'threshold=$_failureThreshold backoff=${_respawnMinInterval.inSeconds}s',
      name: 'backend_watchdog',
    );

    return BackendHealthState.initial();
  }

  /// One poll cycle. Guarded against overlapping ticks AND against
  /// running while a respawn is in flight (see `_respawning`
  /// rationale on the field).
  Future<void> _tick() async {
    if (_respawning) return;
    if (_probing) return;
    _probing = true;
    try {
      final ok = await BackendSupervisor.instance.probeHealth(
        timeout: _perRequestTimeout,
      );
      if (ok) {
        _onProbeSuccess();
      } else {
        _onProbeFailure();
      }
    } finally {
      _probing = false;
    }
  }

  void _onProbeSuccess() {
    final prev = state;
    // Quiet-when-fine: only log a transition from reconnecting →
    // online, not every healthy tick. Steady-state silence is the
    // contract.
    if (prev.status == BackendHealthStatus.reconnecting) {
      developer.log(
        'Backend health recovered. Was reconnecting (failures='
        '${prev.consecutiveFailures}); now online.',
        name: 'backend_watchdog',
      );
    }
    state = prev.copyWith(
      status: BackendHealthStatus.online,
      lastSeenAt: DateTime.now(),
      consecutiveFailures: 0,
    );
  }

  void _onProbeFailure() {
    final prev = state;
    final nextFailures = prev.consecutiveFailures + 1;
    // Log every failure tick — these are the events someone reading
    // supervisor.log post-incident actually wants to see. Three
    // lines max before we respawn so it's bounded.
    developer.log(
      '/healthz failed ($nextFailures/$_failureThreshold consecutive).',
      name: 'backend_watchdog',
    );
    state = prev.copyWith(
      status: BackendHealthStatus.reconnecting,
      consecutiveFailures: nextFailures,
    );

    if (nextFailures >= _failureThreshold) {
      // Don't `await` — keep `_tick` non-blocking. The respawn
      // sets `_respawning = true` itself so subsequent ticks
      // short-circuit while it works.
      unawaited(_triggerRespawn());
    }
  }

  /// Kick off a respawn if the backoff allows. Otherwise log and
  /// reset the failure counter so the banner doesn't get stuck on
  /// "3 failures" while we wait out the cooldown — counter resets,
  /// keeps counting toward the next 3-strike, fires again when the
  /// cooldown expires.
  Future<void> _triggerRespawn() async {
    if (_respawning) return; // belt-and-braces — _tick already gated

    final now = DateTime.now();
    if (_lastRespawnAt != null &&
        now.difference(_lastRespawnAt!) < _respawnMinInterval) {
      final remaining =
          _respawnMinInterval - now.difference(_lastRespawnAt!);
      developer.log(
        'Respawn requested but throttled — last respawn '
        '${now.difference(_lastRespawnAt!).inSeconds}s ago, '
        'wait ${remaining.inSeconds}s more.',
        name: 'backend_watchdog',
      );
      // Reset failure counter so the next 3-strike fires post-cooldown.
      // We stay in reconnecting (status sticky) — the operator sees
      // the banner the whole time we're throttled.
      state = state.copyWith(consecutiveFailures: 0);
      return;
    }

    _respawning = true;
    _lastRespawnAt = now;
    final attempt = state.respawnAttempts + 1;
    developer.log(
      'Triggering backend respawn — attempt #$attempt after '
      '${state.consecutiveFailures} consecutive failures.',
      name: 'backend_watchdog',
    );
    state = state.copyWith(
      respawnAttempts: attempt,
      consecutiveFailures: 0,
    );

    try {
      await BackendSupervisor.instance.restartBackend();
      developer.log(
        'restartBackend() returned (attempt #$attempt). Next poll '
        'will confirm.',
        name: 'backend_watchdog',
      );
    } catch (e, st) {
      // restartBackend doesn't throw under normal conditions, but
      // defensive: a missing binary or permissions issue could
      // throw. Banner stays red, watchdog keeps polling — the next
      // 3-strike will eventually re-trigger past the backoff.
      developer.log(
        'restartBackend() threw on attempt #$attempt: $e',
        name: 'backend_watchdog',
        error: e,
        stackTrace: st,
      );
    } finally {
      _respawning = false;
    }
  }

  /// Public hook used by the Diagnostics "Restart backend" button.
  /// Skips the backoff gate (the operator explicitly asked) and
  /// runs through the same machinery so state transitions log
  /// consistently.
  Future<void> manualRestart() async {
    developer.log(
      'Manual restart requested via Diagnostics dialog.',
      name: 'backend_watchdog',
    );
    // Bypass the cooldown — the operator clicked the button, they
    // know they want it now.
    _lastRespawnAt = null;
    await _triggerRespawn();
  }
}

/// The provider the rest of the app watches. We use `NotifierProvider`
/// (synchronous) rather than `AsyncNotifierProvider` because the
/// initial state is known without any IO — the watchdog assumes the
/// app started healthy (main.dart's `ensureRunning()` proved it) and
/// downgrades from there.
final backendHealthProvider =
    NotifierProvider<BackendWatchdog, BackendHealthState>(
  BackendWatchdog.new,
);
