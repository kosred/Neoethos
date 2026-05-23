// Backend API client - talks to the neoethos Rust HTTP server.
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
  final int positionId;
  final int volumeUnits;
  final String symbol;
  final String side;
  final double volume;
  /// Position open time as Unix milliseconds (UTC). Convert with
  /// `DateTime.fromMillisecondsSinceEpoch(openTimestampMs!)` for
  /// the local-time "since HH:MM" badge in the position row.
  /// Null when the broker didn't stamp the fill (rare race).
  final int? openTimestampMs;
  final double pnlPips;
  final double pnlUsd;
  const Position({
    required this.positionId,
    required this.volumeUnits,
    required this.symbol,
    required this.side,
    required this.volume,
    required this.openTimestampMs,
    required this.pnlPips,
    required this.pnlUsd,
  });

  factory Position.fromJson(Map<String, dynamic> j) => Position(
        positionId: (j['positionId'] as num?)?.toInt() ?? 0,
        volumeUnits: (j['volumeUnits'] as num?)?.toInt() ?? 0,
        symbol: j['symbol'] as String,
        side: j['side'] as String,
        volume: (j['volume'] as num).toDouble(),
        openTimestampMs: (j['openTimestampMs'] as num?)?.toInt(),
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
  /// Unix-ms (UTC) when the snapshot was assembled server-side.
  /// Use `DateTime.fromMillisecondsSinceEpoch(fetchedAtUnixMs!)`
  /// for the local-time freshness badge on the Dashboard. Null
  /// only on older servers that predate the field.
  final int? fetchedAtUnixMs;
  final List<Position> positions;
  const AccountSnapshot({
    required this.balance,
    required this.equity,
    required this.freeMargin,
    required this.usedMargin,
    required this.currency,
    required this.fetchedAtUnixMs,
    required this.positions,
  });

  factory AccountSnapshot.fromJson(Map<String, dynamic> j) => AccountSnapshot(
        balance: (j['balance'] as num).toDouble(),
        equity: (j['equity'] as num).toDouble(),
        freeMargin: (j['freeMargin'] as num).toDouble(),
        usedMargin: (j['usedMargin'] as num).toDouble(),
        currency: j['currency'] as String,
        fetchedAtUnixMs: (j['fetchedAtUnixMs'] as num?)?.toInt(),
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

  /// `POST /risk/preset` — switch the active prop-firm preset.
  ///
  /// Pass the snake_case preset id (`ftmo`, `myforexfunds`,
  /// `fundednext`, `the5ers`, `none`). Returns the post-switch
  /// RiskSnapshot so the UI can refresh without a follow-up GET.
  Future<RiskSnapshot> savePropFirmPreset(String presetId) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '/risk/preset',
      data: {'preset': presetId},
    );
    if (response.statusCode != 200 || response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: 'POST /risk/preset failed: ${response.statusCode}',
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

  /// `POST /settings` — partial-update + persist to config.yaml.
  ///
  /// Only the non-null fields are sent; the backend merges them into
  /// the existing on-disk config and rewrites the YAML, leaving the
  /// 200+ unexposed fields untouched. Returns the post-merge snapshot
  /// so the UI can refresh its local copy without a follow-up GET.
  ///
  /// Throws `DioException` (with the backend's structured `error` body
  /// when validation fails — e.g. blank data_dir). Callers should
  /// surface that via `describeError()` in `error_translation.dart`.
  Future<SettingsSnapshot> saveSettings({
    String? dataDir,
    bool? newsCalendarEnabled,
    String? newsCalendarSource,
    String? openaiModel,
  }) async {
    final body = <String, dynamic>{};
    if (dataDir != null) body['dataDir'] = dataDir;
    if (newsCalendarEnabled != null) {
      body['newsCalendarEnabled'] = newsCalendarEnabled;
    }
    if (newsCalendarSource != null) {
      body['newsCalendarSource'] = newsCalendarSource;
    }
    if (openaiModel != null) body['openaiModel'] = openaiModel;

    final response = await _dio.post<Map<String, dynamic>>(
      '/settings',
      data: body,
    );
    if (response.statusCode != 200 || response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: 'POST /settings failed: ${response.statusCode}',
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

  /// `/broker/credentials` — fetch current cTrader credential state.
  /// `clientSecretMask` is the masked tail + length so the UI can show
  /// "yes, a secret is saved" without us echoing the secret in full.
  Future<Map<String, dynamic>> fetchBrokerCredentials() async {
    final r = await _dio.get<Map<String, dynamic>>('/broker/credentials');
    return r.data ?? const <String, dynamic>{};
  }

  /// POST `/broker/credentials` — persist new credentials.
  Future<Map<String, dynamic>> saveBrokerCredentials({
    required String clientId,
    required String clientSecret,
    String redirectUri = '',
    String environment = 'Demo',
    String accountId = '',
  }) async {
    final r = await _dio.post<Map<String, dynamic>>(
      '/broker/credentials',
      data: {
        'clientId': clientId,
        'clientSecret': clientSecret,
        'redirectUri': redirectUri,
        'environment': environment,
        'accountId': accountId,
      },
    );
    return r.data ?? const <String, dynamic>{};
  }

  /// `/gemma/status` — local LLM availability check. Returns whether
  /// the binary was built with --features gemma-backend AND whether
  /// the GGUF is on disk.
  Future<GemmaStatusSnapshot> fetchGemmaStatus() async {
    final response = await _dio.get<Map<String, dynamic>>('/gemma/status');
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/gemma/status returned empty body',
      );
    }
    return GemmaStatusSnapshot.fromJson(response.data!);
  }

  /// POST `/gemma/download` — start the background fetch of the
  /// bundled GGUF from HuggingFace into the user-data-dir. Used by
  /// the AI Helper screen's first-launch downloader as a fallback
  /// for when the NSIS install-time download was skipped or failed.
  /// 409 if a download is already in flight.
  Future<Map<String, dynamic>> startGemmaDownload() async {
    final r = await _dio.post<Map<String, dynamic>>(
      '/gemma/download',
      options: Options(
        // 409 (Conflict) is "already running" — let the UI treat it
        // as a no-op success and just poll status.
        validateStatus: (code) =>
            code != null && ((code >= 200 && code < 300) || code == 409),
      ),
    );
    return r.data ?? const <String, dynamic>{};
  }

  /// GET `/gemma/download/status` — polled by the AI Helper screen
  /// while the download is in flight so the progress bar can update.
  Future<GemmaDownloadStatus> fetchGemmaDownloadStatus() async {
    final r = await _dio.get<Map<String, dynamic>>('/gemma/download/status');
    return GemmaDownloadStatus.fromJson(r.data ?? const {});
  }

  /// POST `/gemma/download/cancel` — abort the in-flight download.
  Future<Map<String, dynamic>> cancelGemmaDownload() async {
    final r = await _dio.post<Map<String, dynamic>>('/gemma/download/cancel');
    return r.data ?? const <String, dynamic>{};
  }

  /// POST `/gemma/chat` — local Gemma-4 inference. 503 if the runtime
  /// or the model file is missing — the response body explains how to
  /// fix it.
  Future<GemmaChatResponse> gemmaChat({
    required String prompt,
    int? maxTokens,
  }) async {
    final body = <String, dynamic>{'prompt': prompt};
    if (maxTokens != null) body['maxTokens'] = maxTokens;
    final response = await _dio.post<Map<String, dynamic>>(
      '/gemma/chat',
      data: body,
      // 5 minutes — first call loads the GGUF (5-30s), and a
      // 600-token response on CPU can take a couple of minutes.
      options: Options(receiveTimeout: const Duration(minutes: 5)),
    );
    return GemmaChatResponse.fromJson(response.data ?? const {});
  }

  /// POST `/gemma/news` — symbol-specific news summary via local LLM.
  Future<GemmaChatResponse> gemmaNews({required String symbol}) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '/gemma/news',
      data: {'symbol': symbol},
      options: Options(receiveTimeout: const Duration(minutes: 5)),
    );
    return GemmaChatResponse.fromJson(response.data ?? const {});
  }

  /// POST `/positions/close` — close (or partially close) an open
  /// position. `volume` is in cTrader centi-lot units (use the
  /// Position.volumeUnits field straight through for a full close).
  Future<Map<String, dynamic>> closePosition({
    required int positionId,
    required int volume,
  }) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '/positions/close',
      data: {'positionId': positionId, 'volume': volume},
      options: Options(receiveTimeout: const Duration(seconds: 30)),
    );
    return response.data ?? const <String, dynamic>{};
  }

  /// POST `/orders/cancel` — cancel a pending (unfilled) order.
  Future<Map<String, dynamic>> cancelOrder({required int orderId}) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '/orders/cancel',
      data: {'orderId': orderId},
      options: Options(receiveTimeout: const Duration(seconds: 30)),
    );
    return response.data ?? const <String, dynamic>{};
  }

  /// POST `/orders` — submit a Market order. SL/TP are in pips
  /// (cTrader rejects absolute prices on market orders). The server
  /// requires at least one of stopLossPips / takeProfitPips unless
  /// `risky:true` is set.
  Future<Map<String, dynamic>> placeMarketOrder({
    required String symbol,
    required String side, // "buy" / "sell"
    required double volumeLots,
    double? stopLossPips,
    double? takeProfitPips,
    String? comment,
    bool risky = false,
  }) async {
    final body = <String, dynamic>{
      'symbol': symbol,
      'side': side,
      'volumeLots': volumeLots,
      if (stopLossPips != null) 'stopLossPips': stopLossPips,
      if (takeProfitPips != null) 'takeProfitPips': takeProfitPips,
      if (comment != null && comment.isNotEmpty) 'comment': comment,
      if (risky) 'risky': true,
    };
    final response = await _dio.post<Map<String, dynamic>>(
      '/orders',
      data: body,
      options: Options(receiveTimeout: const Duration(seconds: 30)),
    );
    return response.data ?? const <String, dynamic>{};
  }

  /// `/broker/timeframes` — canonical timeframe list from the
  /// neoethos_core contract (= what cTrader Open API's trendbar
  /// period mapper actually accepts: M1, M3, M5, M15, M30, H1, H4,
  /// H12, D1, W1, MN1). Used by Chart + Data Bootstrap chips.
  Future<List<String>> fetchBrokerTimeframes() async {
    final response = await _dio.get<Map<String, dynamic>>(
      '/broker/timeframes',
    );
    final raw = response.data?['timeframes'] as List?;
    return (raw ?? const [])
        .map((e) => e as String)
        .toList(growable: false);
  }

  /// `/broker/symbols` — full broker symbol catalog (not hardcoded).
  Future<BrokerSymbolsSnapshot> fetchBrokerSymbols() async {
    final response = await _dio.get<Map<String, dynamic>>(
      '/broker/symbols',
      options: Options(receiveTimeout: const Duration(seconds: 20)),
    );
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/broker/symbols returned empty body',
      );
    }
    return BrokerSymbolsSnapshot.fromJson(response.data!);
  }

  /// `/broker/accounts` — list every cTID the OAuth token grants
  /// access to. Drives the Settings-screen Account dropdown so the
  /// operator picks from real options instead of typing a numeric
  /// cTID by hand (and accidentally pinning a stale ID, which is
  /// what produced the `CH_ACCESS_TOKEN_INVALID` loop on the
  /// pre-fix builds).
  Future<BrokerAccountsSnapshot> fetchBrokerAccounts() async {
    final response = await _dio.get<Map<String, dynamic>>(
      '/broker/accounts',
      options: Options(receiveTimeout: const Duration(seconds: 25)),
    );
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/broker/accounts returned empty body',
      );
    }
    return BrokerAccountsSnapshot.fromJson(response.data!);
  }

  /// POST `/data/fetch` — download historical bars for [fromMs, toMs]
  /// and persist them under `data/symbol=<sym>/timeframe=<tf>/`.
  Future<Map<String, dynamic>> fetchHistoricalData({
    required String symbol,
    required String timeframe,
    required int fromMs,
    int? toMs,
  }) async {
    final body = <String, dynamic>{
      'symbol': symbol,
      'timeframe': timeframe,
      'fromMs': fromMs,
    };
    if (toMs != null) body['toMs'] = toMs;
    final response = await _dio.post<Map<String, dynamic>>(
      '/data/fetch',
      data: body,
      options: Options(receiveTimeout: const Duration(seconds: 60)),
    );
    return response.data ?? const <String, dynamic>{};
  }

  /// `/chart?symbol=&timeframe=&limit=` — OHLC candles for charting.
  Future<ChartSnapshot> fetchChart({
    String symbol = 'EURUSD',
    String timeframe = 'M1',
    int limit = 200,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '/chart',
      queryParameters: {
        'symbol': symbol,
        'timeframe': timeframe,
        'limit': limit,
      },
    );
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/chart returned empty body',
      );
    }
    return ChartSnapshot.fromJson(response.data!);
  }

  /// `/indicators?symbol=…&timeframe=…&indicator=…&period=…&limit=…`
  /// — compute a single indicator overlay for the chart. Backed by
  /// vector_ta. Returns a list of named lines; single-output
  /// indicators have 1 line, multi-output (Bollinger / MACD /
  /// Stochastic) decompose to 3-ish lines.
  Future<IndicatorSnapshot> fetchIndicator({
    required String symbol,
    required String timeframe,
    required String indicator,
    double? period,
    double? stdDev,
    double? fast,
    double? slow,
    double? signal,
    double? kPeriod,
    double? kSlow,
    double? dPeriod,
    int limit = 200,
  }) async {
    final query = <String, dynamic>{
      'symbol': symbol,
      'timeframe': timeframe,
      'indicator': indicator,
      'limit': limit,
    };
    if (period != null) query['period'] = period;
    if (stdDev != null) query['std_dev'] = stdDev;
    if (fast != null) query['fast'] = fast;
    if (slow != null) query['slow'] = slow;
    if (signal != null) query['signal'] = signal;
    if (kPeriod != null) query['k_period'] = kPeriod;
    if (kSlow != null) query['k_slow'] = kSlow;
    if (dPeriod != null) query['d_period'] = dPeriod;
    final r = await _dio.get<Map<String, dynamic>>(
      '/indicators',
      queryParameters: query,
    );
    return IndicatorSnapshot.fromJson(r.data ?? const {});
  }

  /// `/intelligence` — model artifacts + discovery targets + walkforward.
  Future<IntelligenceSnapshot> fetchIntelligence() async {
    final response = await _dio.get<Map<String, dynamic>>('/intelligence');
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/intelligence returned empty body',
      );
    }
    return IntelligenceSnapshot.fromJson(response.data!);
  }

  /// POST `/diagnostics/report` — bundle the day's logs + redacted
  /// config + system info into a .zip on the user's Desktop and
  /// return the pre-rendered email subject + body so the Flutter
  /// side can open the user's default mail client via mailto:.
  /// End users can't rebuild the app; this is the support channel
  /// that replaces every "rebuild with cargo build" hint in the UI.
  Future<DiagnosticReport> requestDiagnosticReport({
    String userDescription = '',
    String category = '',
  }) async {
    final r = await _dio.post<Map<String, dynamic>>(
      '/diagnostics/report',
      data: {
        'userDescription': userDescription,
        'category': category,
      },
      options: Options(receiveTimeout: const Duration(seconds: 30)),
    );
    return DiagnosticReport.fromJson(r.data ?? const {});
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
  /// Active prop-firm preset (snake_case id, e.g. `ftmo`, `none`).
  /// Empty string when the backend pre-dates the preset registry —
  /// the UI treats that as "ftmo" for back-compat display.
  final String preset;
  final String presetDisplayName;
  final List<PropFirmPresetSummary> availablePresets;
  /// Whether the prop-firm gate is armed (false when preset == none).
  final bool propFirmRulesEnabled;
  const RiskSnapshot({
    required this.riskPerTrade,
    required this.minRiskPerTrade,
    required this.maxRiskPerTrade,
    required this.dailyDrawdownLimit,
    required this.totalDrawdownLimit,
    required this.maxLotSize,
    required this.requireStopLoss,
    required this.preset,
    required this.presetDisplayName,
    required this.availablePresets,
    required this.propFirmRulesEnabled,
  });

  factory RiskSnapshot.fromJson(Map<String, dynamic> j) => RiskSnapshot(
        riskPerTrade: (j['riskPerTrade'] as num).toDouble(),
        minRiskPerTrade: (j['minRiskPerTrade'] as num).toDouble(),
        maxRiskPerTrade: (j['maxRiskPerTrade'] as num).toDouble(),
        dailyDrawdownLimit: (j['dailyDrawdownLimit'] as num).toDouble(),
        totalDrawdownLimit: (j['totalDrawdownLimit'] as num).toDouble(),
        maxLotSize: (j['maxLotSize'] as num).toDouble(),
        requireStopLoss: j['requireStopLoss'] as bool,
        preset: (j['preset'] as String?) ?? '',
        presetDisplayName: (j['presetDisplayName'] as String?) ?? '',
        availablePresets: ((j['availablePresets'] as List?) ?? const [])
            .map((e) =>
                PropFirmPresetSummary.fromJson(e as Map<String, dynamic>))
            .toList(growable: false),
        propFirmRulesEnabled: (j['propFirmRulesEnabled'] as bool?) ?? true,
      );
}

/// One row in the available-presets list returned by `/risk`. The UI
/// dropdown renders these so users can see each firm's hard ceilings
/// inline before they commit to switching.
class PropFirmPresetSummary {
  final String id;
  final String displayName;
  final double maxDailyLossPct;
  final double maxOverallDrawdownPct;
  final double challengeProfitTargetPct;
  final int minTradingDays;
  const PropFirmPresetSummary({
    required this.id,
    required this.displayName,
    required this.maxDailyLossPct,
    required this.maxOverallDrawdownPct,
    required this.challengeProfitTargetPct,
    required this.minTradingDays,
  });

  factory PropFirmPresetSummary.fromJson(Map<String, dynamic> j) =>
      PropFirmPresetSummary(
        id: j['id'] as String,
        displayName: j['displayName'] as String,
        maxDailyLossPct: (j['maxDailyLossPct'] as num).toDouble(),
        maxOverallDrawdownPct: (j['maxOverallDrawdownPct'] as num).toDouble(),
        challengeProfitTargetPct:
            (j['challengeProfitTargetPct'] as num).toDouble(),
        minTradingDays: (j['minTradingDays'] as num).toInt(),
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

class BrokerSymbol {
  final int symbolId;
  final String symbolName;
  final bool enabled;
  final String? description;
  const BrokerSymbol({
    required this.symbolId,
    required this.symbolName,
    required this.enabled,
    required this.description,
  });
  factory BrokerSymbol.fromJson(Map<String, dynamic> j) => BrokerSymbol(
        symbolId: (j['symbolId'] as num).toInt(),
        symbolName: (j['symbolName'] as String?) ?? '',
        enabled: (j['enabled'] as bool?) ?? false,
        description: j['description'] as String?,
      );
}

class BrokerSymbolsSnapshot {
  final int accountId;
  final String environment;
  final int symbolCount;
  final List<BrokerSymbol> symbols;
  final List<String> archivedSymbols;
  const BrokerSymbolsSnapshot({
    required this.accountId,
    required this.environment,
    required this.symbolCount,
    required this.symbols,
    required this.archivedSymbols,
  });
  factory BrokerSymbolsSnapshot.fromJson(Map<String, dynamic> j) =>
      BrokerSymbolsSnapshot(
        accountId: (j['accountId'] as num).toInt(),
        environment: (j['environment'] as String?) ?? '',
        symbolCount: (j['symbolCount'] as num?)?.toInt() ?? 0,
        symbols: ((j['symbols'] as List?) ?? const [])
            .map((e) => BrokerSymbol.fromJson(e as Map<String, dynamic>))
            .toList(growable: false),
        archivedSymbols: ((j['archivedSymbols'] as List?) ?? const [])
            .map((s) => s as String)
            .toList(growable: false),
      );

  /// Filter helper for the UI — return enabled-and-likely-forex symbols.
  /// cTrader catalogs mix forex pairs with stocks/indices; the chart
  /// screen wants just tradeable pairs by default.
  List<BrokerSymbol> get forexLikeEnabled => symbols
      .where((s) =>
          s.enabled &&
          s.symbolName.length == 6 &&
          RegExp(r'^[A-Z]{6}$').hasMatch(s.symbolName))
      .toList(growable: false);
}

/// One row in the `/broker/accounts` list — the user's view of a
/// single cTID granted during OAuth.
class BrokerAccount {
  /// Numeric cTID, kept as a string because cTrader IDs can exceed
  /// i32 range and we don't want JS-style number truncation.
  final String accountId;
  final String brokerTitle;
  final String accountName;
  final int? traderLogin;
  final bool? isLive;
  final bool enabledForExecution;
  const BrokerAccount({
    required this.accountId,
    required this.brokerTitle,
    required this.accountName,
    required this.traderLogin,
    required this.isLive,
    required this.enabledForExecution,
  });
  factory BrokerAccount.fromJson(Map<String, dynamic> j) => BrokerAccount(
        accountId: (j['accountId'] as String?) ?? '',
        brokerTitle: (j['brokerTitle'] as String?) ?? '',
        accountName: (j['accountName'] as String?) ?? '',
        traderLogin: (j['traderLogin'] as num?)?.toInt(),
        isLive: j['isLive'] as bool?,
        enabledForExecution: (j['enabledForExecution'] as bool?) ?? false,
      );

  /// One-line label for dropdown rows. Picks the most-useful bits so
  /// the operator can tell two demos apart at a glance.
  String get dropdownLabel {
    final liveTag = isLive == null ? '?' : (isLive! ? 'Live' : 'Demo');
    final broker = brokerTitle.isEmpty ? 'Spotware' : brokerTitle;
    return '$broker · $liveTag · $accountId';
  }
}

class BrokerAccountsSnapshot {
  final String environment;
  final String permissionScope;
  final int accountCount;
  final List<BrokerAccount> accounts;
  const BrokerAccountsSnapshot({
    required this.environment,
    required this.permissionScope,
    required this.accountCount,
    required this.accounts,
  });
  factory BrokerAccountsSnapshot.fromJson(Map<String, dynamic> j) =>
      BrokerAccountsSnapshot(
        environment: (j['environment'] as String?) ?? '',
        permissionScope: (j['permissionScope'] as String?) ?? '',
        accountCount: (j['accountCount'] as num?)?.toInt() ?? 0,
        accounts: ((j['accounts'] as List?) ?? const [])
            .map((e) => BrokerAccount.fromJson(e as Map<String, dynamic>))
            .toList(growable: false),
      );
}

class GemmaStatusSnapshot {
  final bool runtimeCompiledIn;
  final bool modelFilePresent;
  final String resolvedPath;
  final String expectedFilename;
  final String downloadUrl;
  final int sizeBytes;
  final int expectedSizeBytes;
  final int nCtx;
  final String message;
  const GemmaStatusSnapshot({
    required this.runtimeCompiledIn,
    required this.modelFilePresent,
    required this.resolvedPath,
    required this.expectedFilename,
    required this.downloadUrl,
    required this.sizeBytes,
    required this.expectedSizeBytes,
    required this.nCtx,
    required this.message,
  });
  factory GemmaStatusSnapshot.fromJson(Map<String, dynamic> j) =>
      GemmaStatusSnapshot(
        runtimeCompiledIn: (j['runtimeCompiledIn'] as bool?) ?? false,
        modelFilePresent: (j['modelFilePresent'] as bool?) ?? false,
        resolvedPath: (j['resolvedPath'] as String?) ?? '',
        expectedFilename: (j['expectedFilename'] as String?) ?? '',
        downloadUrl: (j['downloadUrl'] as String?) ?? '',
        sizeBytes: (j['sizeBytes'] as num?)?.toInt() ?? 0,
        expectedSizeBytes: (j['expectedSizeBytes'] as num?)?.toInt() ?? 0,
        nCtx: (j['nCtx'] as num?)?.toInt() ?? 0,
        message: (j['message'] as String?) ?? '',
      );
  bool get ready => runtimeCompiledIn && modelFilePresent;
}

/// Wire shape for the /gemma/download/status poll. Five state values:
///   - `idle`         — nothing has been started this session
///   - `downloading`  — bytes are flowing; UI shows progress bar
///   - `completed`    — file is on disk, AI Helper will flip to chat
///   - `failed`       — network/server error; UI surfaces `error`
///   - `cancelled`    — user clicked Cancel mid-flight
class GemmaDownloadStatus {
  final String state;
  final int bytesDone;
  final int bytesTotal;
  final int elapsedSeconds;
  final String? writtenPath;
  final String? error;
  const GemmaDownloadStatus({
    required this.state,
    required this.bytesDone,
    required this.bytesTotal,
    required this.elapsedSeconds,
    required this.writtenPath,
    required this.error,
  });
  factory GemmaDownloadStatus.fromJson(Map<String, dynamic> j) =>
      GemmaDownloadStatus(
        state: (j['state'] as String?) ?? 'idle',
        bytesDone: (j['bytesDone'] as num?)?.toInt() ?? 0,
        bytesTotal: (j['bytesTotal'] as num?)?.toInt() ?? 0,
        elapsedSeconds: (j['elapsedSeconds'] as num?)?.toInt() ?? 0,
        writtenPath: j['writtenPath'] as String?,
        error: j['error'] as String?,
      );

  bool get isDownloading => state == 'downloading';
  bool get isCompleted => state == 'completed';
  bool get isFailed => state == 'failed';
  bool get isCancelled => state == 'cancelled';

  /// 0.0–1.0 normalised fraction for the progress bar. Returns null
  /// when total isn't known yet (server hasn't replied with content-
  /// length) so the UI can render an indeterminate spinner.
  double? get fraction {
    if (bytesTotal <= 0) return null;
    return (bytesDone / bytesTotal).clamp(0.0, 1.0);
  }
}

class GemmaChatResponse {
  final String modelId;
  final String response;
  final int elapsedMs;
  const GemmaChatResponse({
    required this.modelId,
    required this.response,
    required this.elapsedMs,
  });
  factory GemmaChatResponse.fromJson(Map<String, dynamic> j) =>
      GemmaChatResponse(
        modelId: (j['modelId'] as String?) ?? '',
        response: (j['response'] as String?) ?? '',
        elapsedMs: (j['elapsedMs'] as num?)?.toInt() ?? 0,
      );
}

class ChartCandle {
  final int? tsMs;
  final double open;
  final double high;
  final double low;
  final double close;
  final double volume;
  const ChartCandle({
    required this.tsMs,
    required this.open,
    required this.high,
    required this.low,
    required this.close,
    required this.volume,
  });
  factory ChartCandle.fromJson(Map<String, dynamic> j) => ChartCandle(
        tsMs: j['tsMs'] as int?,
        open: (j['open'] as num).toDouble(),
        high: (j['high'] as num).toDouble(),
        low: (j['low'] as num).toDouble(),
        close: (j['close'] as num).toDouble(),
        volume: (j['volume'] as num).toDouble(),
      );
}

class ChartSnapshot {
  final String symbol;
  final String timeframe;
  final List<String> availableTimeframes;
  final int candleCount;
  final List<ChartCandle> candles;
  final double priceMin;
  final double priceMax;
  final double latestClose;
  final double priceChangePct;
  final String headline;
  const ChartSnapshot({
    required this.symbol,
    required this.timeframe,
    required this.availableTimeframes,
    required this.candleCount,
    required this.candles,
    required this.priceMin,
    required this.priceMax,
    required this.latestClose,
    required this.priceChangePct,
    required this.headline,
  });
  factory ChartSnapshot.fromJson(Map<String, dynamic> j) => ChartSnapshot(
        symbol: j['symbol'] as String,
        timeframe: j['timeframe'] as String,
        availableTimeframes: ((j['availableTimeframes'] as List?) ?? const [])
            .map((s) => s as String)
            .toList(growable: false),
        candleCount: (j['candleCount'] as int?) ?? 0,
        candles: ((j['candles'] as List?) ?? const [])
            .map((e) => ChartCandle.fromJson(e as Map<String, dynamic>))
            .toList(growable: false),
        priceMin: (j['priceMin'] as num).toDouble(),
        priceMax: (j['priceMax'] as num).toDouble(),
        latestClose: (j['latestClose'] as num).toDouble(),
        priceChangePct: (j['priceChangePct'] as num).toDouble(),
        headline: (j['headline'] as String?) ?? '',
      );
}

/// One series produced by `/indicators`. Multi-output indicators
/// (Bollinger Bands → lower/middle/upper) come back as multiple
/// of these in the same snapshot.
class IndicatorLine {
  final String name;
  final List<double> values;
  const IndicatorLine({required this.name, required this.values});
  factory IndicatorLine.fromJson(Map<String, dynamic> j) => IndicatorLine(
        name: (j['name'] as String?) ?? '',
        values: ((j['values'] as List?) ?? const [])
            .map((e) => (e as num).toDouble())
            .toList(growable: false),
      );
}

class IndicatorSnapshot {
  final String symbol;
  final String timeframe;
  final String indicator;
  final int candleCount;
  final List<IndicatorLine> lines;
  const IndicatorSnapshot({
    required this.symbol,
    required this.timeframe,
    required this.indicator,
    required this.candleCount,
    required this.lines,
  });
  factory IndicatorSnapshot.fromJson(Map<String, dynamic> j) =>
      IndicatorSnapshot(
        symbol: (j['symbol'] as String?) ?? '',
        timeframe: (j['timeframe'] as String?) ?? '',
        indicator: (j['indicator'] as String?) ?? '',
        candleCount: (j['candleCount'] as num?)?.toInt() ?? 0,
        lines: ((j['lines'] as List?) ?? const [])
            .map((e) => IndicatorLine.fromJson(e as Map<String, dynamic>))
            .toList(growable: false),
      );
}

class DiscoveryTarget {
  final String symbol;
  final String baseTf;
  final String strategyId;
  final double? sharpe;
  final double? winRate;
  const DiscoveryTarget({
    required this.symbol,
    required this.baseTf,
    required this.strategyId,
    required this.sharpe,
    required this.winRate,
  });
  factory DiscoveryTarget.fromJson(Map<String, dynamic> j) => DiscoveryTarget(
        symbol: (j['symbol'] as String?) ?? '',
        baseTf: (j['baseTf'] as String?) ?? '',
        strategyId: (j['strategyId'] as String?) ?? '',
        sharpe: (j['sharpe'] as num?)?.toDouble(),
        winRate: (j['winRate'] as num?)?.toDouble(),
      );
}

class IntelligenceSnapshot {
  final String modelsDir;
  final bool modelsDirExists;
  final int artifactCount;
  final List<String> artifacts;
  final int? lastTouchedUnixMs;
  final List<DiscoveryTarget> discoveryTargets;
  final int? walkforwardSplits;
  final double? walkforwardAvgAccuracy;
  const IntelligenceSnapshot({
    required this.modelsDir,
    required this.modelsDirExists,
    required this.artifactCount,
    required this.artifacts,
    required this.lastTouchedUnixMs,
    required this.discoveryTargets,
    required this.walkforwardSplits,
    required this.walkforwardAvgAccuracy,
  });
  factory IntelligenceSnapshot.fromJson(Map<String, dynamic> j) =>
      IntelligenceSnapshot(
        modelsDir: (j['modelsDir'] as String?) ?? '',
        modelsDirExists: (j['modelsDirExists'] as bool?) ?? false,
        artifactCount: (j['artifactCount'] as int?) ?? 0,
        artifacts: ((j['artifacts'] as List?) ?? const [])
            .map((s) => s as String)
            .toList(growable: false),
        lastTouchedUnixMs: j['lastTouchedUnixMs'] as int?,
        discoveryTargets: ((j['discoveryTargets'] as List?) ?? const [])
            .map((e) =>
                DiscoveryTarget.fromJson(e as Map<String, dynamic>))
            .toList(growable: false),
        walkforwardSplits: j['walkforwardSplits'] as int?,
        walkforwardAvgAccuracy:
            (j['walkforwardAvgAccuracy'] as num?)?.toDouble(),
      );
}

/// What `/diagnostics/report` returns: where the zip landed on
/// disk, what it contains, and a pre-rendered mailto: payload the
/// UI can pass to `Uri(scheme: 'mailto', ...).launch()`.
class DiagnosticReport {
  final String zipPath;
  final int totalBytes;
  final List<String> filesIncluded;
  final String emailSubject;
  final String emailBody;
  final String emailRecipient;
  const DiagnosticReport({
    required this.zipPath,
    required this.totalBytes,
    required this.filesIncluded,
    required this.emailSubject,
    required this.emailBody,
    required this.emailRecipient,
  });
  factory DiagnosticReport.fromJson(Map<String, dynamic> j) => DiagnosticReport(
        zipPath: (j['zipPath'] as String?) ?? '',
        totalBytes: (j['totalBytes'] as num?)?.toInt() ?? 0,
        filesIncluded: ((j['filesIncluded'] as List?) ?? const [])
            .map((e) => e as String)
            .toList(growable: false),
        emailSubject: (j['emailSubject'] as String?) ?? '',
        emailBody: (j['emailBody'] as String?) ?? '',
        emailRecipient: (j['emailRecipient'] as String?) ?? '',
      );

  String get sizeLabel {
    if (totalBytes < 1024) return '$totalBytes B';
    if (totalBytes < 1024 * 1024) return '${(totalBytes / 1024).toStringAsFixed(1)} KB';
    return '${(totalBytes / (1024 * 1024)).toStringAsFixed(1)} MB';
  }
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
