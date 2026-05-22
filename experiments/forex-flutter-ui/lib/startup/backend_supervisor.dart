// BackendSupervisor — spawns the Rust backend on app launch so the
// user gets the double-click experience (one .exe, both halves come up).
//
// Behaviour:
//   1. Probe http://127.0.0.1:7423/healthz with a short timeout.
//   2. If it answers, we're piggybacking on an already-running server
//      (developer running `neoethos-app --server` in a terminal) —
//      don't spawn a duplicate. Return.
//   3. If it doesn't answer, locate the `neoethos-app` binary next to
//      the Flutter executable (production install) or fall back to
//      `target/debug/neoethos-app[.exe]` walking up from CWD (dev
//      mode). Spawn it detached with `--server`.
//   4. Wait (up to ~10s) for /healthz to come back. If it never does,
//      we still let the UI render — the existing AsyncValue.error
//      states will surface the failure on every screen, which beats
//      blocking the splash forever.
//
// We deliberately keep this lightweight (no IPC, no port negotiation):
// the backend always binds 7423, the UI always probes 7423, and the
// child stays alive only as long as our process needs it.

import 'dart:async';
import 'dart:io';

import 'package:dio/dio.dart';

class BackendSupervisor {
  BackendSupervisor._();
  static final BackendSupervisor instance = BackendSupervisor._();

  static const _baseUrl = 'http://127.0.0.1:7423';
  static const _maxWaitMs = 10000;
  static const _pollIntervalMs = 250;

  Process? _child;

  /// Spawn the backend if it isn't already responding. Returns once the
  /// backend is reachable, or after `_maxWaitMs` if it never came up
  /// (we still let the UI render so the user sees an actionable error
  /// instead of a hung splash).
  Future<void> ensureRunning() async {
    if (await _probeHealth()) {
      // Someone else owns the port (dev workflow). Don't double-spawn.
      return;
    }

    final binary = _locateBackendBinary();
    if (binary == null) {
      stderr.writeln('[BackendSupervisor] neoethos-app binary not found; '
          'backend must be started manually.');
      return;
    }

    // neoethos-app loads `config.yaml` from its CWD by default, so we
    // pin the CWD to whichever dir contains it. In a production install
    // that's the binary's own dir; in dev that's the repo root. We
    // resolve it by walking up from the binary.
    final workDir = _locateConfigDir(binary) ?? binary.parent;
    try {
      // `detached` (NOT `detachedWithStdio`) — the backend has very
      // chatty tracing output. If we keep stdio pipes open without
      // consuming them, the child eventually blocks on stdout writes
      // and the whole server hangs before binding port 7423. The
      // backend writes logs to its own rotating file via
      // tracing-appender, so dropping the pipes loses nothing.
      _child = await Process.start(
        binary.path,
        const ['--server'],
        workingDirectory: workDir.path,
        mode: ProcessStartMode.detached,
      );
      stderr.writeln('[BackendSupervisor] spawned ${binary.path} '
          '(cwd=${workDir.path}, pid=${_child?.pid})');
    } on ProcessException catch (err) {
      stderr.writeln('[BackendSupervisor] failed to spawn '
          '${binary.path}: $err');
      return;
    }

    await _waitForHealth();
  }

  Future<bool> _probeHealth() async {
    final dio = Dio(BaseOptions(
      baseUrl: _baseUrl,
      connectTimeout: const Duration(milliseconds: 500),
      receiveTimeout: const Duration(milliseconds: 500),
      sendTimeout: const Duration(milliseconds: 500),
    ));
    try {
      final r = await dio.get<dynamic>('/healthz');
      return r.statusCode == 200;
    } catch (_) {
      return false;
    } finally {
      dio.close(force: true);
    }
  }

  Future<void> _waitForHealth() async {
    final deadline = DateTime.now().add(
      const Duration(milliseconds: _maxWaitMs),
    );
    while (DateTime.now().isBefore(deadline)) {
      if (await _probeHealth()) return;
      await Future<void>.delayed(
        const Duration(milliseconds: _pollIntervalMs),
      );
    }
    stderr.writeln('[BackendSupervisor] backend did not respond within '
        '${_maxWaitMs}ms — UI will render with error states.');
  }

  /// Find the neoethos-app[.exe] binary.
  ///
  /// Search order:
  ///   1. Same directory as the Flutter executable (production install).
  ///   2. `target/debug/` and `target/release/` walking up from the
  ///      current working directory (developer running
  ///      `flutter run` from inside the repo).
  File? _locateBackendBinary() {
    final exeName = Platform.isWindows ? 'neoethos-app.exe' : 'neoethos-app';

    // 1. Production install — next to the Flutter executable.
    final flutterExeDir = File(Platform.resolvedExecutable).parent;
    final coLocated = File('${flutterExeDir.path}${Platform.pathSeparator}$exeName');
    if (coLocated.existsSync()) return coLocated;

    // 2. Dev mode — walk up from CWD looking for target/{debug,release}/.
    Directory dir = Directory.current;
    for (var i = 0; i < 8; i++) {
      for (final profile in const ['debug', 'release']) {
        final candidate = File(
          '${dir.path}${Platform.pathSeparator}target'
          '${Platform.pathSeparator}$profile'
          '${Platform.pathSeparator}$exeName',
        );
        if (candidate.existsSync()) return candidate;
      }
      final parent = dir.parent;
      if (parent.path == dir.path) break;
      dir = parent;
    }
    return null;
  }

  /// Walk up from the binary's directory looking for the first
  /// ancestor that contains `config.yaml`. That's the directory the
  /// backend needs as its CWD so `Settings::from_yaml("config.yaml")`
  /// resolves to the real file.
  Directory? _locateConfigDir(File binary) {
    Directory dir = binary.parent;
    for (var i = 0; i < 8; i++) {
      if (File('${dir.path}${Platform.pathSeparator}config.yaml')
          .existsSync()) {
        return dir;
      }
      final parent = dir.parent;
      if (parent.path == dir.path) break;
      dir = parent;
    }
    return null;
  }
}
