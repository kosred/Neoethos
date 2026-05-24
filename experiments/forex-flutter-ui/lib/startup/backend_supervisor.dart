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

  /// Absolute path to the supervisor log file. The Diagnostics dialog
  /// tails this so the operator can see why a respawn happened without
  /// having to dig through `%LOCALAPPDATA%` manually.
  String get logFilePath => _logFile.path;

  /// PID of the most recent spawn, or null if we never spawned (the
  /// initial `ensureRunning()` found an already-running backend) or
  /// the spawn failed. Surfaced so the "Restart backend" button can
  /// hard-kill the existing process before respawning.
  int? get childPid => _child?.pid;

  /// Cheap public probe — the watchdog hits this every 3 s. Same dio
  /// instance every call so we're not paying connection-setup cost on
  /// the polling path. Returns false on ANY error (timeout, refused,
  /// 5xx) so the watchdog treats "anything other than a clean 200" as
  /// a failure tick.
  ///
  /// 2 s timeout per request to match the watchdog's task contract —
  /// healthy responses come back in <50 ms; anything past 2 s is
  /// effectively dead.
  Future<bool> probeHealth({
    Duration timeout = const Duration(seconds: 2),
  }) async {
    final dio = Dio(BaseOptions(
      baseUrl: _baseUrl,
      connectTimeout: timeout,
      receiveTimeout: timeout,
      sendTimeout: timeout,
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

  /// Force a respawn: kill the tracked child (best-effort — on
  /// Windows `Process.killPid` with SIGKILL is mapped to
  /// TerminateProcess) and rerun `ensureRunning()`. The watchdog
  /// calls this after 3 consecutive `/healthz` failures; the
  /// Diagnostics dialog calls it from the "Restart backend" button.
  ///
  /// Returns the boolean from `ensureRunning()` — `false` only when
  /// the probe found ANOTHER live instance during the relaunch
  /// (extremely unlikely mid-session but kept for symmetry).
  Future<bool> restartBackend() async {
    final pid = _child?.pid;
    _log('restartBackend() requested. tracked child pid=${pid ?? "none"}');
    if (pid != null) {
      try {
        // Best-effort kill. `detached` mode means we don't own the
        // pipes, so killPid is the only handle we have. On Windows
        // SIGKILL → TerminateProcess; on Unix → SIGKILL. The child
        // is GUI-subsystem-less so no orphaned window to clean up.
        final killed = Process.killPid(pid, ProcessSignal.sigkill);
        _log('killPid($pid, SIGKILL) = $killed');
      } catch (e) {
        _log('killPid($pid) threw: $e — proceeding with respawn anyway.');
      }
      _child = null;
    } else {
      _log('No tracked child PID — respawning without a kill.');
    }
    // Brief settle so the OS releases the port; ensureRunning's
    // probeHealth would otherwise race the dying socket.
    await Future<void>.delayed(const Duration(milliseconds: 250));
    return ensureRunning();
  }

  /// Tail the supervisor log for the Diagnostics dialog.
  ///
  /// Reads the whole file (it's truncated on each app launch so
  /// "whole file" is bounded by one session — typically <30 KB
  /// even after a few respawns) and returns the last [maxLines]
  /// lines. Returns an empty string if the file doesn't exist yet
  /// (extreme cold start).
  String tailLog({int maxLines = 200}) {
    try {
      if (!_logFile.existsSync()) return '';
      final content = _logFile.readAsStringSync();
      final lines = content.split('\n');
      if (lines.length <= maxLines) return content;
      return lines.sublist(lines.length - maxLines).join('\n');
    } catch (e) {
      return '<log read failed: $e>';
    }
  }

  /// Spawn the backend if it isn't already responding. Returns once the
  /// backend is reachable, or after `_maxWaitMs` if it never came up
  /// (we still let the UI render so the user sees an actionable error
  /// instead of a hung splash).
  /// Returns `true` when the caller should continue running the
  /// Flutter UI. Returns `false` when another NeoEthos instance is
  /// already alive and the caller should `exit(0)` immediately to
  /// avoid two competing UI windows (#176).
  Future<bool> ensureRunning() async {
    _log('ensureRunning() start. Flutter exe: ${Platform.resolvedExecutable}');
    _log('CWD: ${Directory.current.path}');

    if (await _probeHealth()) {
      // #176: an already-running backend almost certainly means an
      // already-running Flutter shell owns the UI. Verified live
      // today — the user ended up with TWO NeoEthos windows + a
      // stale dev-bundle backend confusing every click. Refusing
      // the second launch is the cleanest UX: the existing window
      // stays in focus and the new shell never opens a competing
      // viewport.
      //
      // We could add a "bring existing to foreground" IPC here
      // (Win32 SetForegroundWindow on the existing Flutter HWND)
      // but that's polish — exiting clean already solves the
      // confusion. Logs say why.
      _log('Backend already responding on $_baseUrl — '
          'another NeoEthos instance is running. This shell will exit.');
      return false;
    }

    final binary = _locateBackendBinary();
    if (binary == null) {
      _log('FATAL: neoethos-app binary not found. Searched:\n'
          '  1. <flutter-exe-dir>/bin/ (production bundle)\n'
          '  2. Side-by-side with Flutter exe (legacy bundle)\n'
          '  3. Ancestors of Flutter exe → target/{debug,release}/\n'
          '  4. Ancestors of CWD → target/{debug,release}/\n'
          'Drop the neoethos-app.exe under <neoethos.exe-dir>/bin/ and re-launch.');
      return true; // continue showing the UI; "Backend unreachable" will explain
    }
    _log('Located backend binary: ${binary.path}');

    // Resolve a writable working directory for the backend. Bundle
    // layout: the bundle root itself has a `data/` that conflicts
    // with the Flutter engine's runtime asset folder; using it as
    // CWD makes neoethos-app misidentify `flutter_assets/` etc. as
    // symbol dirs. So in bundle mode we pin the CWD to a per-user
    // data dir (e.g. `%LOCALAPPDATA%\neoethos\`) and seed it on
    // first launch from the bundle's template config.
    final workDir = _resolveSpawnCwd(binary);
    _log('Spawn CWD: ${workDir.path}');

    try {
      // `detached` (NOT `detachedWithStdio`) — the backend has very
      // chatty tracing output. If we keep stdio pipes open without
      // consuming them, the child eventually blocks on stdout writes
      // and the whole server hangs before binding port 7423. The
      // backend writes logs to its own rotating file via
      // tracing-appender, so dropping the pipes loses nothing.
      _child = await Process.start(
        binary.path,
        // #179: tag the spawn via CLI flag, NOT env var. Verified
        // live that `Process.start(mode: detached)` on Windows does
        // NOT propagate the `environment` map to the child — the
        // backend then thought it was orphaned, showed a Win32
        // MessageBox, and blocked port 7423 forever. CLI flags
        // survive the detached spawn cleanly.
        const ['--server', '--launched-by-flutter'],
        workingDirectory: workDir.path,
        includeParentEnvironment: true,
        mode: ProcessStartMode.detached,
      );
      _log('Spawned ${binary.path} (pid=${_child?.pid})');
    } on ProcessException catch (err) {
      _log('FATAL: Process.start failed: $err');
      return true; // continue showing the UI; error states will surface
    }

    await _waitForHealth();
    return true;
  }

  Future<bool> _probeHealth() =>
      probeHealth(timeout: const Duration(milliseconds: 500));

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
  /// Search order:
  ///   1. `<flutter-exe-dir>/bin/` — the production bundle layout where
  ///      the backend is tucked into a `bin/` subfolder so the operator
  ///      sees one executable (`neoethos.exe`) at the bundle's top
  ///      level and the backend stays out of their way.
  ///   2. Side-by-side with the Flutter exe — older bundle layout, kept
  ///      working so existing installs don't break after the rename.
  ///   3. Ancestors of Flutter exe → `target/{debug,release}/` — dev
  ///      workflow where the Flutter exe is freshly-built inside the
  ///      repo and the Rust target dir is several levels up.
  ///   4. Ancestors of CWD → `target/{debug,release}/` — last-resort
  ///      fallback for unusual launch contexts.
  File? _locateBackendBinary() {
    final exeName = Platform.isWindows ? 'neoethos-app.exe' : 'neoethos-app';
    final flutterExeDir = File(Platform.resolvedExecutable).parent;

    // 1. <flutter-exe-dir>/bin/neoethos-app[.exe] — new bundle layout.
    final binSubdir = File(
      '${flutterExeDir.path}${Platform.pathSeparator}bin'
      '${Platform.pathSeparator}$exeName',
    );
    if (binSubdir.existsSync()) return binSubdir;

    // 2. Side-by-side with the Flutter exe — legacy bundle layout.
    final coLocated =
        File('${flutterExeDir.path}${Platform.pathSeparator}$exeName');
    if (coLocated.existsSync()) return coLocated;

    // 3. Ancestors of Flutter exe → target/{debug,release}/.
    final viaFlutter = _searchAncestors(flutterExeDir, exeName);
    if (viaFlutter != null) return viaFlutter;

    // 4. Ancestors of OS current working directory.
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

  /// Pick the backend's working directory.
  ///
  /// Two layouts to support:
  ///
  /// 1. **Production bundle** (`<bundle>/neoethos.exe` +
  ///    `<bundle>/bin/neoethos-app.exe`). The bundle ships a
  ///    `config.yaml` at its root, but the bundle also contains
  ///    a Flutter runtime `data/` directory that conflicts with
  ///    neoethos's `system.data_dir: data`. Solution: pin the CWD
  ///    to a per-user data dir under `%LOCALAPPDATA%\neoethos\`
  ///    (Linux/macOS equivalents below) and seed that dir on
  ///    first launch from the bundle's template `config.yaml`.
  ///    All relative paths in `config.yaml` now resolve under
  ///    the user-data dir, which the user owns and the bundle
  ///    can't accidentally collide with.
  ///
  /// 2. **Dev / cargo-built** (`target/{debug,release}/neoethos-app.exe`).
  ///    Walk up from the binary looking for an existing
  ///    `config.yaml` — that's the repo root, exactly what
  ///    cargo run uses.
  Directory _resolveSpawnCwd(File binary) {
    // Bundle detection: the production layout puts the backend
    // inside a `bin/` subdir of the Flutter exe dir. Anything
    // else is the dev tree.
    final flutterExeDir = File(Platform.resolvedExecutable).parent;
    final inBin = binary.parent.path.toLowerCase() ==
        '${flutterExeDir.path.toLowerCase()}${Platform.pathSeparator}bin';

    if (inBin) {
      final userDataDir = _userDataDir();
      _seedUserDataDir(userDataDir, flutterExeDir);
      return userDataDir;
    }

    // Dev path — walk up from the binary looking for config.yaml.
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
    return binary.parent;
  }

  /// First-launch seed: copy the bundle's `config.yaml` into
  /// the user-data dir if it isn't there, and pre-create the
  /// `data/`, `models/`, `logs/` subdirs the backend expects to
  /// be writable. Subsequent launches are a no-op.
  void _seedUserDataDir(Directory userDataDir, Directory bundleDir) {
    if (!userDataDir.existsSync()) {
      userDataDir.createSync(recursive: true);
      _log('Created user-data dir: ${userDataDir.path}');
    }
    for (final sub in const ['data', 'models', 'logs', 'resources']) {
      final d = Directory(
          '${userDataDir.path}${Platform.pathSeparator}$sub');
      if (!d.existsSync()) d.createSync(recursive: true);
    }
    final userConfig = File(
        '${userDataDir.path}${Platform.pathSeparator}config.yaml');
    final bundleConfig = File(
        '${bundleDir.path}${Platform.pathSeparator}config.yaml');
    if (!userConfig.existsSync() && bundleConfig.existsSync()) {
      userConfig.writeAsStringSync(bundleConfig.readAsStringSync());
      _log('Seeded ${userConfig.path} from bundle template.');
    }
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
