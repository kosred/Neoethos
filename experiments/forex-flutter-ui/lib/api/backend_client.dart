// Backend API client — talks to the NeoEthos Rust HTTP server.
//
// The Rust side runs as `neoethos-app --server` and listens on
// `127.0.0.1:7423` (override via `NEOETHOS_SERVER_BIND` env var on the
// server). This client is a thin dio wrapper — every business decision
// lives on the Rust side (broker auth, polling cadence, prop-firm
// guards), and Flutter just renders what the API returns.
//
// Wire shapes mirror the Rust DTOs in `crates/neoethos-app/src/server/
// account.rs`. Keep them in lock-step: a missing `serde(rename_all =
// "camelCase")` on the Rust side surfaces here as "null balance" /
// "null positions" — extremely confusing because the HTTP call still
// returns 200. If you change either side, change both.

import 'package:dio/dio.dart';

class BackendConfig {
  final String baseUrl;
  const BackendConfig({this.baseUrl = 'http://127.0.0.1:7423'});
}

class Position {
  final String symbol;
  final String side;
  final double volume;
  final double pnlPips;
  final double pnlUsd;
  const Position({
    required this.symbol,
    required this.side,
    required this.volume,
    required this.pnlPips,
    required this.pnlUsd,
  });

  factory Position.fromJson(Map<String, dynamic> j) => Position(
        symbol: j['symbol'] as String,
        side: j['side'] as String,
        volume: (j['volume'] as num).toDouble(),
        pnlPips: (j['pnlPips'] as num).toDouble(),
        pnlUsd: (j['pnlUsd'] as num).toDouble(),
      );
}

class AccountSnapshot {
  final double balance;
  final double equity;
  final double freeMargin;
  final double usedMargin;
  final String currency;
  final List<Position> positions;
  const AccountSnapshot({
    required this.balance,
    required this.equity,
    required this.freeMargin,
    required this.usedMargin,
    required this.currency,
    required this.positions,
  });

  factory AccountSnapshot.fromJson(Map<String, dynamic> j) => AccountSnapshot(
        balance: (j['balance'] as num).toDouble(),
        equity: (j['equity'] as num).toDouble(),
        freeMargin: (j['freeMargin'] as num).toDouble(),
        usedMargin: (j['usedMargin'] as num).toDouble(),
        currency: j['currency'] as String,
        positions: ((j['positions'] as List?) ?? const [])
            .map((p) => Position.fromJson(p as Map<String, dynamic>))
            .toList(growable: false),
      );
}

/// Sentinel that distinguishes a "real" backend error (network down,
/// JSON malformed) from a "broker not ready yet" 503. The UI renders
/// the second as a friendly "connecting…" placeholder, not a red banner.
class BrokerNotReadyException implements Exception {
  final String message;
  const BrokerNotReadyException(this.message);
  @override
  String toString() => 'BrokerNotReadyException: $message';
}

class BackendClient {
  final Dio _dio;
  final BackendConfig config;

  /// Factory so the dio `baseUrl` actually picks up a caller-supplied
  /// [BackendConfig] (the older shorthand constructor hard-coded the
  /// default baseUrl because Dart initializer-list scoping made it
  /// awkward to reach the field). Pass `dio:` for tests that want to
  /// inject a `MockAdapter`.
  factory BackendClient({Dio? dio, BackendConfig? config}) {
    final cfg = config ?? const BackendConfig();
    final client = dio ?? _buildDefaultDio(cfg);
    return BackendClient._(dio: client, config: cfg);
  }

  BackendClient._({required Dio dio, required this.config}) : _dio = dio;

  static Dio _buildDefaultDio(BackendConfig cfg) {
    return Dio(BaseOptions(
      baseUrl: cfg.baseUrl,
      connectTimeout: const Duration(seconds: 3),
      receiveTimeout: const Duration(seconds: 10),
      // 503 is a legitimate "broker not ready yet" response, not a
      // transport error — let it flow through to the catch block so
      // we can translate it into BrokerNotReadyException.
      validateStatus: (code) =>
          code != null && ((code >= 200 && code < 300) || code == 503),
    ));
  }

  /// Pull the current account snapshot from `/account/snapshot`.
  ///
  /// Returns the parsed [AccountSnapshot] on 200.
  /// Throws [BrokerNotReadyException] on 503 (server is up but bridge
  /// hasn't completed its first cTrader fetch yet).
  /// Throws [DioException] on any other transport error.
  Future<AccountSnapshot> fetchAccountSnapshot() async {
    final response = await _dio.get<Map<String, dynamic>>('/account/snapshot');
    if (response.statusCode == 503) {
      final body = response.data ?? {};
      final code = body['code']?.toString() ?? 'unknown';
      final error = body['error']?.toString() ?? 'broker session not ready';
      throw BrokerNotReadyException('$code: $error');
    }
    final data = response.data;
    if (data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/account/snapshot returned 200 with empty body',
      );
    }
    return AccountSnapshot.fromJson(data);
  }

  /// Liveness ping. Returns the server's compile-time version string
  /// so the UI can flag bundle-version mismatch.
  Future<String> fetchServerVersion() async {
    final response = await _dio.get<Map<String, dynamic>>('/healthz');
    return response.data?['version']?.toString() ?? 'unknown';
  }

  /// `/hardware` — CPU/RAM/GPU snapshot. Refreshes on every call;
  /// the Flutter side polls at a slower cadence (e.g. 5s) than the
  /// account snapshot because hardware metrics don't move fast.
  Future<HardwareSnapshot> fetchHardware() async {
    final response = await _dio.get<Map<String, dynamic>>('/hardware');
    final data = response.data;
    if (data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/hardware returned 200 with empty body',
      );
    }
    return HardwareSnapshot.fromJson(data);
  }

  /// `/risk` — currently-loaded prop-firm risk caps from config.yaml.
  Future<RiskSnapshot> fetchRisk() async {
    final response = await _dio.get<Map<String, dynamic>>('/risk');
    if (response.statusCode != 200 || response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/risk failed: ${response.statusCode}',
      );
    }
    return RiskSnapshot.fromJson(response.data!);
  }

  /// `/settings` — non-risk app-wide settings (data dir, news, LLM).
  Future<SettingsSnapshot> fetchSettings() async {
    final response = await _dio.get<Map<String, dynamic>>('/settings');
    if (response.statusCode != 200 || response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/settings failed: ${response.statusCode}',
      );
    }
    return SettingsSnapshot.fromJson(response.data!);
  }

  /// `/engines/status` — Discovery / Training / Auto-Trader state.
  Future<EnginesSnapshot> fetchEngines() async {
    final response = await _dio.get<Map<String, dynamic>>('/engines/status');
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/engines/status returned empty body',
      );
    }
    return EnginesSnapshot.fromJson(response.data!);
  }

  /// `/broker/status` — current broker connection state.
  Future<BrokerStatus> fetchBrokerStatus() async {
    final response = await _dio.get<Map<String, dynamic>>('/broker/status');
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/broker/status returned empty body',
      );
    }
    return BrokerStatus.fromJson(response.data!);
  }

  /// POST `/engines/discovery/{start,stop}`. Returns the response body
  /// shape: `{started:true, kind:"discovery", symbol, base_tf}` for
  /// start, `{running, kind}` for stop. Throws DioException on 4xx/5xx.
  Future<Map<String, dynamic>> startDiscovery({
    String? symbol,
    String? baseTf,
  }) async {
    return _postEngine('/engines/discovery/start', symbol, baseTf);
  }

  Future<Map<String, dynamic>> stopDiscovery() async {
    return _postEngine('/engines/discovery/stop', null, null);
  }

  Future<Map<String, dynamic>> startTraining({
    String? symbol,
    String? baseTf,
  }) async {
    return _postEngine('/engines/training/start', symbol, baseTf);
  }

  Future<Map<String, dynamic>> stopTraining() async {
    return _postEngine('/engines/training/stop', null, null);
  }

  /// POST `/broker/reauth` — runs the full cTrader OAuth flow on the
  /// server side. Blocks until the operator approves in the browser
  /// (10–30 s typical). Returns `{callbackPort, accessTokenLen,
  /// refreshTokenPresent, message}` on success.
  Future<Map<String, dynamic>> reauthBroker() async {
    // 90 s receive timeout — gives the user time to click through the
    // Spotware consent screen without the request timing out under them.
    final response = await _dio.post<Map<String, dynamic>>(
      '/broker/reauth',
      options: Options(
        receiveTimeout: const Duration(seconds: 90),
        sendTimeout: const Duration(seconds: 10),
      ),
    );
    return response.data ?? const <String, dynamic>{};
  }

  Future<Map<String, dynamic>> _postEngine(
    String path,
    String? symbol,
    String? baseTf,
  ) async {
    final body = <String, dynamic>{};
    if (symbol != null && symbol.trim().isNotEmpty) body['symbol'] = symbol;
    if (baseTf != null && baseTf.trim().isNotEmpty) body['base_tf'] = baseTf;
    final response = await _dio.post<Map<String, dynamic>>(
      path,
      data: body.isEmpty ? null : body,
    );
    return response.data ?? const <String, dynamic>{};
  }

  /// `/data/bootstrap` — local data-dir inventory.
  Future<DataBootstrapSnapshot> fetchDataBootstrap() async {
    final response = await _dio.get<Map<String, dynamic>>('/data/bootstrap');
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/data/bootstrap returned empty body',
      );
    }
    return DataBootstrapSnapshot.fromJson(response.data!);
  }
}

class HardwareSnapshot {
  final String cpuModel;
  final int cpuCoresLogical;
  final int cpuCoresPhysical;
  final double cpuLoadAvg;
  final int ramTotalMb;
  final int ramUsedMb;
  final int ramAvailableMb;
  final String gpuName;
  final bool gpuAvailable;
  const HardwareSnapshot({
    required this.cpuModel,
    required this.cpuCoresLogical,
    required this.cpuCoresPhysical,
    required this.cpuLoadAvg,
    required this.ramTotalMb,
    required this.ramUsedMb,
    required this.ramAvailableMb,
    required this.gpuName,
    required this.gpuAvailable,
  });

  factory HardwareSnapshot.fromJson(Map<String, dynamic> j) {
    final cpu = j['cpu'] as Map<String, dynamic>;
    final ram = j['ram'] as Map<String, dynamic>;
    final gpu = j['gpu'] as Map<String, dynamic>;
    return HardwareSnapshot(
      cpuModel: cpu['model'] as String,
      cpuCoresLogical: cpu['coresLogical'] as int,
      cpuCoresPhysical: cpu['coresPhysical'] as int,
      cpuLoadAvg: (cpu['loadAvg'] as num).toDouble(),
      ramTotalMb: ram['totalMb'] as int,
      ramUsedMb: ram['usedMb'] as int,
      ramAvailableMb: ram['availableMb'] as int,
      gpuName: gpu['name'] as String,
      gpuAvailable: gpu['available'] as bool,
    );
  }
}

class RiskSnapshot {
  final double riskPerTrade;
  final double minRiskPerTrade;
  final double maxRiskPerTrade;
  final double dailyDrawdownLimit;
  final double totalDrawdownLimit;
  final double maxLotSize;
  final bool requireStopLoss;
  const RiskSnapshot({
    required this.riskPerTrade,
    required this.minRiskPerTrade,
    required this.maxRiskPerTrade,
    required this.dailyDrawdownLimit,
    required this.totalDrawdownLimit,
    required this.maxLotSize,
    required this.requireStopLoss,
  });

  factory RiskSnapshot.fromJson(Map<String, dynamic> j) => RiskSnapshot(
        riskPerTrade: (j['riskPerTrade'] as num).toDouble(),
        minRiskPerTrade: (j['minRiskPerTrade'] as num).toDouble(),
        maxRiskPerTrade: (j['maxRiskPerTrade'] as num).toDouble(),
        dailyDrawdownLimit: (j['dailyDrawdownLimit'] as num).toDouble(),
        totalDrawdownLimit: (j['totalDrawdownLimit'] as num).toDouble(),
        maxLotSize: (j['maxLotSize'] as num).toDouble(),
        requireStopLoss: j['requireStopLoss'] as bool,
      );
}

class SettingsSnapshot {
  final String dataDir;
  final bool newsCalendarEnabled;
  final String newsCalendarSource;
  final String openaiModel;
  const SettingsSnapshot({
    required this.dataDir,
    required this.newsCalendarEnabled,
    required this.newsCalendarSource,
    required this.openaiModel,
  });

  factory SettingsSnapshot.fromJson(Map<String, dynamic> j) => SettingsSnapshot(
        dataDir: j['dataDir'] as String,
        newsCalendarEnabled: j['newsCalendarEnabled'] as bool,
        newsCalendarSource: j['newsCalendarSource'] as String,
        openaiModel: j['openaiModel'] as String,
      );
}

class EnginesSnapshot {
  final String discovery;
  final String training;
  final String autoTrader;
  final String discoverySummary;
  final String trainingSummary;
  const EnginesSnapshot({
    required this.discovery,
    required this.training,
    required this.autoTrader,
    required this.discoverySummary,
    required this.trainingSummary,
  });
  factory EnginesSnapshot.fromJson(Map<String, dynamic> j) => EnginesSnapshot(
        discovery: j['discovery'] as String,
        training: j['training'] as String,
        autoTrader: j['autoTrader'] as String,
        discoverySummary: (j['discoverySummary'] as String?) ?? '',
        trainingSummary: (j['trainingSummary'] as String?) ?? '',
      );

  bool get discoveryRunning => discovery.toLowerCase() == 'running';
  bool get trainingRunning => training.toLowerCase() == 'running';
}

class BrokerStatus {
  final String adapter;
  final String environment;
  final String accountId;
  final bool connected;
  final String clientIdPrefix;
  const BrokerStatus({
    required this.adapter,
    required this.environment,
    required this.accountId,
    required this.connected,
    required this.clientIdPrefix,
  });
  factory BrokerStatus.fromJson(Map<String, dynamic> j) => BrokerStatus(
        adapter: j['adapter'] as String,
        environment: j['environment'] as String,
        accountId: j['accountId'] as String,
        connected: j['connected'] as bool,
        clientIdPrefix: j['clientIdPrefix'] as String,
      );
}

class DataBootstrapSnapshot {
  final String dataDir;
  final bool dataDirExists;
  final List<String> symbols;
  final int fileCount;
  final int? lastTouchedUnixMs;
  const DataBootstrapSnapshot({
    required this.dataDir,
    required this.dataDirExists,
    required this.symbols,
    required this.fileCount,
    required this.lastTouchedUnixMs,
  });
  factory DataBootstrapSnapshot.fromJson(Map<String, dynamic> j) => DataBootstrapSnapshot(
        dataDir: j['dataDir'] as String,
        dataDirExists: j['dataDirExists'] as bool,
        symbols: ((j['symbols'] as List?) ?? const [])
            .map((s) => s as String)
            .toList(growable: false),
        fileCount: j['fileCount'] as int,
        lastTouchedUnixMs: j['lastTouchedUnixMs'] as int?,
      );
}
