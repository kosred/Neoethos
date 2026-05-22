// BackendSupervisor — spawns the Rust backend on app launch so the
// user gets the double-click experience (one .exe, both halves come up).
//
// Behaviour:
//   1. Probe http://127.0.0.1:7423/healthz with a short timeout.
//   2. If it answers, we're piggybacking on an already-running server
//      (developer running `neoethos-app --server` in a terminal) —
//      don't spawn a duplicate. Return.
//   3. If it doesn't answer, locate the `neoethos-app` binary by
//      searching, in order:
//        a) Side-by-side with the Flutter executable (production
//           install — installer copies both .exe files into one dir).
//        b) Ancestors of the Flutter executable, looking for
//           `target/{debug,release}/` (dev workflow — running the
//           freshly-built Flutter .exe from `build/...`).
//        c) Ancestors of the OS current working directory (legacy /
//           fallback path).
//      Spawn the binary detached with `--server`.
//   4. Wait (up to ~10s) for /healthz to come back. If it never does,
//      we still let the UI render — the existing AsyncValue.error
//      states will surface the failure on every screen, which beats
//      blocking the splash forever.
//
// Diagnostics: every spawn attempt + outcome is logged to
//   <user-data>/neoethos/supervisor.log
// because the Flutter Windows app is built as a GUI subsystem
// process — stderr writes are invisible to the user. The log file
// is the only way to see why the backend failed to come up.

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
  late final File _logFile = _openLogFile();

  /// Spawn the backend if it isn't already responding. Returns once the
  /// backend is reachable, or after `_maxWaitMs` if it never came up
  /// (we still let the UI render so the user sees an actionable error
  /// instead of a hung splash).
  Future<void> ensureRunning() async {
    _log('ensureRunning() start. Flutter exe: ${Platform.resolvedExecutable}');
    _log('CWD: ${Directory.current.path}');

    if (await _probeHealth()) {
      _log('Backend already responding on $_baseUrl — not spawning.');
      return;
    }

    final binary = _locateBackendBinary();
    if (binary == null) {
      _log('FATAL: neoethos-app binary not found. Searched:\n'
          '  1. Side-by-side with Flutter exe '
          '(${File(Platform.resolvedExecutable).parent.path})\n'
          '  2. Ancestors of Flutter exe → target/{debug,release}/\n'
          '  3. Ancestors of CWD → target/{debug,release}/\n'
          'Drop the neoethos-app.exe next to the Flutter exe and re-launch.');
      return;
    }
    _log('Located backend binary: ${binary.path}');

    // neoethos-app loads `config.yaml` from its CWD by default, so we
    // pin the CWD to whichever dir contains it. In a production install
    // that's the binary's own dir; in dev that's the repo root. We
    // resolve it by walking up from the binary.
    final workDir = _locateConfigDir(binary) ?? binary.parent;
    _log('Spawn CWD (config.yaml dir): ${workDir.path}');

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
      _log('Spawned ${binary.path} (pid=${_child?.pid})');
    } on ProcessException catch (err) {
      _log('FATAL: Process.start failed: $err');
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
      if (await _probeHealth()) {
        _log('Backend reachable on /healthz.');
        return;
      }
      await Future<void>.delayed(
        const Duration(milliseconds: _pollIntervalMs),
      );
    }
    _log('Backend did not respond within ${_maxWaitMs}ms — UI will render '
        'with error states. Check the backend\'s own daily-rotating log '
        'under <user-data>/neoethos/logs/ for the real failure reason.');
  }

  /// Find the neoethos-app[.exe] binary.
  ///
  /// Search order: side-by-side with Flutter exe → ancestors of Flutter
  /// exe → ancestors of CWD. The first hit wins.
  File? _locateBackendBinary() {
    final exeName = Platform.isWindows ? 'neoethos-app.exe' : 'neoethos-app';

    // 1. Production install — next to the Flutter executable.
    final flutterExeDir = File(Platform.resolvedExecutable).parent;
    final coLocated =
        File('${flutterExeDir.path}${Platform.pathSeparator}$exeName');
    if (coLocated.existsSync()) return coLocated;

    // 2. Ancestors of Flutter exe → target/{debug,release}/.
    //    Covers `flutter build windows --debug` from inside the repo,
    //    where the Flutter exe lives 6 levels under the repo root and
    //    the backend lives at <root>/target/{debug,release}/.
    final viaFlutter = _searchAncestors(flutterExeDir, exeName);
    if (viaFlutter != null) return viaFlutter;

    // 3. Ancestors of OS current working directory. Last-resort fallback
    //    for unusual launch contexts (debugger, scheduled task, etc.).
    final viaCwd = _searchAncestors(Directory.current, exeName);
    if (viaCwd != null) return viaCwd;

    return null;
  }

  File? _searchAncestors(Directory start, String exeName) {
    Directory dir = start;
    for (var i = 0; i < 10; i++) {
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
    for (var i = 0; i < 10; i++) {
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

  // ─── Diagnostic file logging ─────────────────────────────────────────

  File _openLogFile() {
    try {
      final dir = _userDataDir();
      if (!dir.existsSync()) dir.createSync(recursive: true);
      final f = File('${dir.path}${Platform.pathSeparator}supervisor.log');
      // Truncate on each app launch — keep the latest run only.
      f.writeAsStringSync(
        '=== BackendSupervisor log — '
        '${DateTime.now().toIso8601String()} ===\n',
      );
      return f;
    } catch (_) {
      // Last resort: log to the OS temp dir.
      return File(
        '${Directory.systemTemp.path}${Platform.pathSeparator}'
        'neoethos-supervisor.log',
      );
    }
  }

  Directory _userDataDir() {
    if (Platform.isWindows) {
      final root = Platform.environment['LOCALAPPDATA'] ??
          Platform.environment['APPDATA'] ??
          Directory.systemTemp.path;
      return Directory('$root${Platform.pathSeparator}neoethos');
    }
    final home = Platform.environment['HOME'] ?? Directory.systemTemp.path;
    return Directory('$home${Platform.pathSeparator}.neoethos');
  }

  void _log(String msg) {
    final line = '[${DateTime.now().toIso8601String()}] $msg\n';
    try {
      _logFile.writeAsStringSync(line, mode: FileMode.append);
    } catch (_) {/* swallow — never crash on logging */}
    // Also write to stderr; harmless when the parent is GUI subsystem.
    stderr.write(line);
  }
}
