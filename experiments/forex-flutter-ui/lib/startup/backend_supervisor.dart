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
import 'package:flutter/foundation.dart' show visibleForTesting;

/// F-270 (2026-05-28): rich `/healthz` probe result. The boolean
/// `probeHealth()` API used elsewhere collapses this to just `alive`;
/// the supervisor's start-of-run logic uses `launchedByFlutter` to
/// avoid the second-shell-exits-on-stale-backend false-positive.
class _HealthProbeResult {
  final bool alive;
  final bool launchedByFlutter;
  const _HealthProbeResult({
    required this.alive,
    required this.launchedByFlutter,
  });
}

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
    final result = await _probeHealthDetail(timeout: timeout);
    return result.alive;
  }

  /// F-270 (2026-05-28): rich /healthz probe that returns the
  /// backend's `launched_by_flutter` flag alongside liveness. The
  /// supervisor uses this to distinguish:
  ///   - sibling Flutter UI's backend → refuse second launch
  ///   - stale backend (api-test, manual run, zombie) → attach
  /// where the previous probeHealth() boolean conflated both.
  Future<_HealthProbeResult> _probeHealthDetail({
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
      if (r.statusCode != 200) {
        return const _HealthProbeResult(alive: false, launchedByFlutter: false);
      }
      // Old backends (pre-F-270) won't have `launched_by_flutter` in
      // the response. Treat missing field as `false` — the supervisor
      // then attaches (the safer default for ambiguous probes).
      final body = r.data;
      bool launched = false;
      if (body is Map && body.containsKey('launched_by_flutter')) {
        launched = body['launched_by_flutter'] == true;
      }
      return _HealthProbeResult(alive: true, launchedByFlutter: launched);
    } catch (_) {
      return const _HealthProbeResult(alive: false, launchedByFlutter: false);
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

    final initialProbe = await _probeHealthDetail(
      timeout: const Duration(milliseconds: 500),
    );
    if (initialProbe.alive) {
      // F-270 (2026-05-28): differentiate sibling-UI vs stale-backend.
      // The previous #176 logic refused EVERY second launch, but in
      // practice an api-test orphan, a manually-started --server, or
      // a zombie process from a hard-killed Flutter shell would hold
      // port 7423 with NO active UI — and the new Flutter shell would
      // exit before opening a window. Operator saw the splash, then
      // nothing.
      //
      // The backend now tags its /healthz response with
      // `launched_by_flutter`. We refuse the second launch ONLY when
      // that flag is true (= a sibling Flutter supervisor spawned
      // this backend, so a sibling UI is alive). When false, we
      // attach to the existing backend and proceed — the UI opens.
      if (initialProbe.launchedByFlutter) {
        _log('Backend on $_baseUrl was spawned by another Flutter '
            'supervisor — sibling UI is running. This shell will exit.');
        return false;
      }
      _log('Backend on $_baseUrl is alive but NOT spawned by Flutter '
          '(api-test orphan, manual --server, or zombie). '
          'Attaching to it instead of exiting.');
      return true;
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
        // 2026-06-01 settings-persistence defense-in-depth: pass an
        // ABSOLUTE --config so the engine load, install_config_path, and
        // the /settings GET/POST handlers all resolve the SAME user-data
        // config.yaml regardless of CWD (see state.rs current_config_path
        // fix). Without this, saved settings could land in a different
        // file than the one the engine reads on next launch.
        [
          '--server',
          '--launched-by-flutter',
          // Tie the backend's lifetime to this GUI: it self-terminates within
          // ~2s of us exiting, so it never orphans port 7423 and blocks the
          // next launch. `pid` (dart:io) is THIS Flutter process's PID.
          '--parent-pid',
          '$pid',
          '--config',
          '${_userDataDir().path}${Platform.pathSeparator}config.yaml',
        ],
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

  /// First-launch seed AND per-launch upgrade.
  ///
  /// Phase A (first launch): copy the bundle's `config.yaml` into
  /// the user-data dir if it isn't there, and pre-create the
  /// `data/`, `models/`, `logs/` subdirs the backend expects to
  /// be writable.
  ///
  /// Phase B (every subsequent launch): additive-merge any
  /// **new scalar fields** from the bundle template into the
  /// user's config. This is the "schema upgrade" path that closes
  /// task #310 — without it, a bundle that ships a new field
  /// (e.g. F-304's `system.account_currency`) NEVER reaches the
  /// running backend because the original seed-once logic kept
  /// the stale user copy forever, and the backend bailed at
  /// `discovery.rs:1666` with `evaluation_account_currency is
  /// empty` even though the bundle had the field. We preserve
  /// every value the user has customised; the merge only ADDS
  /// keys the user doesn't already have.
  ///
  /// Limited to **scalar fields under top-level sections**
  /// (lines matching `^  name: value`). Nested objects, lists
  /// re-ordered between bundle / user, and structurally-changed
  /// keys are out of scope — those require a manual migration.
  /// In practice every field added since v0.4.20 has been a
  /// scalar, so the heuristic covers the realistic upgrade path.
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
    // Phase A: first-launch seed.
    if (!userConfig.existsSync() && bundleConfig.existsSync()) {
      userConfig.writeAsStringSync(bundleConfig.readAsStringSync());
      _log('Seeded ${userConfig.path} from bundle template.');
      return;
    }
    // Phase B: per-launch schema upgrade.
    if (!userConfig.existsSync() || !bundleConfig.existsSync()) return;
    try {
      final userYaml = userConfig.readAsStringSync();
      final bundleYaml = bundleConfig.readAsStringSync();
      final upgraded = mergeMissingScalarFields(
        userYaml: userYaml,
        bundleYaml: bundleYaml,
      );
      // Always log the merger decision so we can tell the difference
      // between "nothing to add" and "merger never ran" in supervisor.log.
      // Costs one line per launch — cheap compared to the upgrade flush.
      _log('Schema-upgrade check: user=${userYaml.length}b bundle='
          '${bundleYaml.length}b → ${upgraded == null
              ? "no missing fields"
              : "${upgraded.added.length} missing"}');
      if (upgraded != null) {
        // Write a one-time backup before mutating — paranoia about
        // a parser bug clobbering user customisations. Timestamped
        // so multiple upgrade runs don't overwrite each other.
        final stamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final backup = File('${userConfig.path}.bak.$stamp');
        backup.writeAsStringSync(userConfig.readAsStringSync());
        userConfig.writeAsStringSync(upgraded.yaml);
        _log('Schema-upgraded ${userConfig.path} '
            '(added ${upgraded.added.length} field${upgraded.added.length == 1 ? '' : 's'}: '
            '${upgraded.added.join(', ')}). '
            'Backup at ${backup.path}.');
      }
    } catch (e, st) {
      // Merger blew up — log but don't block startup. The user's
      // existing config is intact and the backend still loads it;
      // a future fix to the merger can re-run safely.
      _log('Schema-upgrade merge failed (continuing with existing '
          'user config): $e\n$st');
    }
  }

  /// Additive merge: walks the bundle YAML, finds scalar fields under
  /// top-level sections that the user YAML doesn't already declare,
  /// appends them at the end of the matching section in the user file.
  /// Returns null if no fields are missing (no write needed).
  ///
  /// `@visibleForTesting` + public so `test/backend_supervisor_test.dart`
  /// can exercise it directly — pure-function semantics, no I/O.
  @visibleForTesting
  static MergeResult? mergeMissingScalarFields({
    required String userYaml,
    required String bundleYaml,
  }) {
    final userSections = _scalarFieldsBySection(userYaml);
    final bundleSections = _scalarFieldsBySection(bundleYaml);

    // Collect (section, key, value-block) triples for fields the
    // user is missing. A value-block is the bundle's full line(s)
    // for that key including any preceding contiguous comment lines —
    // that way doc comments travel with the field.
    final missing = <String, List<_MissingField>>{};
    bundleSections.forEach((section, bundleKeys) {
      final userKeys = userSections[section] ?? const <String, _Field>{};
      bundleKeys.forEach((key, bundleField) {
        // Only fill missing INLINE-scalar fields. Block-format keys
        // (a `name:` header with nested content on following lines)
        // cannot be safely spliced because the parser doesn't capture
        // their indented body — emitting just the header would corrupt
        // YAML. New block-format subsystems require a manual migration.
        if (!userKeys.containsKey(key) && bundleField.hasInlineValue) {
          missing.putIfAbsent(section, () => []).add(
                _MissingField(
                  key: key,
                  block: bundleField.lines.join('\n'),
                ),
              );
        }
      });
    });

    if (missing.isEmpty) return null;

    // Walk the user YAML, find the end of each section in `missing`,
    // and splice in the new fields. End of section = the line BEFORE
    // the next top-level line (or EOF).
    final lines = userYaml.split('\n');
    final patched = StringBuffer();
    String? currentSection;
    var i = 0;
    final addedKeys = <String>[];
    while (i < lines.length) {
      final line = lines[i];
      final nextTopLevel = _topLevelKey(line);
      if (nextTopLevel != null && nextTopLevel != currentSection) {
        // Before transitioning, flush missing fields for the section
        // we're leaving (if any).
        if (currentSection != null && missing.containsKey(currentSection)) {
          for (final f in missing[currentSection]!) {
            patched.write(f.block);
            patched.write('\n');
            addedKeys.add('$currentSection.${f.key}');
          }
          missing.remove(currentSection);
        }
        currentSection = nextTopLevel;
      }
      patched.write(line);
      patched.write('\n');
      i++;
    }
    // EOF flush for the last section.
    if (currentSection != null && missing.containsKey(currentSection)) {
      for (final f in missing[currentSection]!) {
        patched.write(f.block);
        patched.write('\n');
        addedKeys.add('$currentSection.${f.key}');
      }
      missing.remove(currentSection);
    }
    // Top-level sections that exist in bundle but NOT in user are
    // appended verbatim at EOF. (Unusual — usually means a brand-new
    // subsystem was added in this release.)
    if (missing.isNotEmpty) {
      missing.forEach((section, fields) {
        patched.write('\n$section:\n');
        for (final f in fields) {
          patched.write(f.block);
          patched.write('\n');
          addedKeys.add('$section.${f.key}');
        }
      });
    }

    // Drop the trailing newline we added at the very end if the
    // original file didn't end with one.
    var result = patched.toString();
    if (!userYaml.endsWith('\n') && result.endsWith('\n')) {
      result = result.substring(0, result.length - 1);
    }
    return MergeResult(yaml: result, added: addedKeys);
  }

  /// Parse `yaml` into a map of `section name -> {scalar key -> field}`.
  /// "Scalar" = a `name: value` line where `value` is non-empty and is
  /// NOT the start of a nested block (i.e. doesn't end with `:` alone,
  /// doesn't begin with a list/object literal continuation). Nested
  /// objects, lists, and multi-line strings are intentionally skipped
  /// — see the heuristic notes on `mergeMissingScalarFields`.
  static Map<String, Map<String, _Field>> _scalarFieldsBySection(
    String yaml,
  ) {
    final out = <String, Map<String, _Field>>{};
    final lines = yaml.split('\n');
    String? section;
    var pendingComments = <String>[];
    // Heuristic: `^(\s+)(name):` matches ANY indented sub-key including
    // block-format keys with no inline value (e.g. `  symbols:` on its
    // own line followed by `  - EURUSD` list items). We track the
    // existence of the key under the section regardless of format —
    // the merger only cares "does the user have this key", not the
    // exact shape of its value. The inline-vs-block distinction is
    // recorded separately via [_Field.hasInlineValue] so the splice
    // logic re-emits the bundle's verbatim block (comments + value)
    // when ADDING a missing key, but never duplicates a block-format
    // user key with an inline-format bundle key (root cause of the
    // first F-310 attempt's "duplicate `symbols:` lines crashed YAML
    // parse" regression).
    final keyRe = RegExp(r'^(\s+)([A-Za-z_][\w]*):(\s+\S.*)?$');
    for (var i = 0; i < lines.length; i++) {
      // Strip trailing \r so a CRLF file split by \n doesn't carry the
      // \r into the line content (where it breaks `$` in our regex).
      final raw = lines[i];
      final line = raw.endsWith('\r') ? raw.substring(0, raw.length - 1) : raw;
      final topLevel = _topLevelKey(line);
      if (topLevel != null) {
        section = topLevel;
        out.putIfAbsent(section, () => <String, _Field>{});
        pendingComments = <String>[];
        continue;
      }
      if (section == null) {
        pendingComments = <String>[];
        continue;
      }
      // Comment line: queue it; might prefix a scalar.
      if (line.trimLeft().startsWith('#')) {
        pendingComments.add(line);
        continue;
      }
      // Blank line breaks a comment block.
      if (line.trim().isEmpty) {
        pendingComments = <String>[];
        continue;
      }
      final m = keyRe.firstMatch(line);
      if (m == null) {
        // Deeper-indent continuation (e.g. `  - M1` list item under a
        // block-format key) or unrecognised structure — skip and
        // reset comments.
        pendingComments = <String>[];
        continue;
      }
      final key = m.group(2)!;
      final hasInlineValue = m.group(3) != null;
      final fullBlock = <String>[...pendingComments, line];
      // Only RECORD the inline-form block as the field's value — if the
      // user has a block-form version, we still want to know the key
      // exists so the merger doesn't add a duplicate. The lines payload
      // is only used when EMITTING a missing key (which comes from the
      // bundle side); for "present" detection we just need the key map.
      out[section]![key] = _Field(
        lines: hasInlineValue ? fullBlock : <String>[line],
        hasInlineValue: hasInlineValue,
      );
      pendingComments = <String>[];
    }
    return out;
  }

  /// Returns the section name if [line] is a top-level YAML key
  /// (no leading whitespace, ends with `:` and either nothing else
  /// or a `# comment`). Null otherwise.
  static String? _topLevelKey(String line) {
    final m = RegExp(r'^([A-Za-z_][\w]*):\s*(?:#.*)?$').firstMatch(line);
    return m?.group(1);
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

/// A scalar field captured by `_scalarFieldsBySection` — the
/// rendered lines (key + any preceding contiguous comment block)
/// that get re-emitted verbatim when merging into the user file.
///
/// [hasInlineValue] distinguishes `name: value` (inline scalar) from
/// `name:` (block-format header for a nested object/list). Both shapes
/// register the key under the section so the merger doesn't ADD a
/// duplicate, but only the inline shape's `lines` are emittable — the
/// merger uses bundle-side `lines` when filling a missing key, never
/// the user's.
class _Field {
  final List<String> lines;
  final bool hasInlineValue;
  const _Field({required this.lines, required this.hasInlineValue});
}

/// One missing-from-user field the merger plans to splice in.
class _MissingField {
  final String key;
  final String block;
  const _MissingField({required this.key, required this.block});
}

/// Result of [BackendSupervisor.mergeMissingScalarFields] — the patched
/// YAML plus the list of `section.key` identifiers actually added (for
/// logging + test assertions).
class MergeResult {
  final String yaml;
  final List<String> added;
  const MergeResult({required this.yaml, required this.added});
}
