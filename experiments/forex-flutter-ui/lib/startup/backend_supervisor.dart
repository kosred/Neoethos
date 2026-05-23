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
          '  1. <flutter-exe-dir>/bin/ (production bundle)\n'
          '  2. Side-by-side with Flutter exe (legacy bundle)\n'
          '  3. Ancestors of Flutter exe → target/{debug,release}/\n'
          '  4. Ancestors of CWD → target/{debug,release}/\n'
          'Drop the neoethos-app.exe under <neoethos.exe-dir>/bin/ and re-launch.');
      return;
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
        const ['--server'],
        workingDirectory: workDir.path,
        // Tag this spawn so the Rust binary knows it was launched by
        // the Flutter shell (and skips the "you double-clicked me by
        // accident" help dialog from #101). Inherit the parent env
        // so user overrides like NEOETHOS_BROKER_CREDENTIALS_PATH
        // still flow through.
        environment: const {'NEOETHOS_LAUNCHED_BY_FLUTTER': '1'},
        includeParentEnvironment: true,
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
