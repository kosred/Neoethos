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

  /// Convenience getter for SSE consumers (#237). The Riverpod SSE
  /// providers construct stream URLs like `$baseUrl/account/snapshot/stream`
  /// without needing to know about the internal `config` field.
  String get baseUrl => config.baseUrl;

  /// POST `/account/snapshot/refresh` — force-refresh trigger
  /// (#241, 2026-05-25). Used by the Dashboard "refresh" button to
  /// skip the bridge's 5 s safety timer; the resulting fresh
  /// snapshot arrives over the `/account/snapshot/stream` SSE
  /// within ~750 ms.
  Future<void> refreshAccountSnapshot() async {
    await _dio.post<void>('/account/snapshot/refresh');
  }

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

  /// `GET /settings/knob-catalog` (#238, 2026-05-25) — fetch the
  /// machine-readable inventory of ~42 runtime knobs. Each entry has
  /// label, help text, current vs default value, kind (Int/Float/
  /// Bool/Text/Enum/Path), and per-preset values
  /// (Conservative/Balanced/Aggressive). The AdvancedSettings UI
  /// renders one form widget per knob keyed by `kind`.
  Future<KnobCatalog> fetchKnobCatalog() async {
    final response =
        await _dio.get<Map<String, dynamic>>('/settings/knob-catalog');
    if (response.statusCode != 200 || response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/settings/knob-catalog failed: ${response.statusCode}',
      );
    }
    return KnobCatalog.fromJson(response.data!);
  }

  /// `GET /journal/stats` — computed trade-journal performance stats.
  Future<JournalStats> fetchJournalStats({int? fromMs, int? toMs}) async {
    final qp = <String, dynamic>{};
    if (fromMs != null) qp['fromMs'] = fromMs;
    if (toMs != null) qp['toMs'] = toMs;
    final response = await _dio.get<Map<String, dynamic>>(
      '/journal/stats',
      queryParameters: qp,
    );
    if (response.statusCode != 200 || response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/journal/stats failed: ${response.statusCode}',
      );
    }
    return JournalStats.fromJson(response.data!);
  }

  /// `GET /journal/trades` — closed trades, most-recent first.
  Future<List<ClosedTrade>> fetchJournalTrades({
    int? fromMs,
    int? toMs,
    int? limit,
  }) async {
    final qp = <String, dynamic>{};
    if (fromMs != null) qp['fromMs'] = fromMs;
    if (toMs != null) qp['toMs'] = toMs;
    if (limit != null) qp['limit'] = limit;
    final response = await _dio.get<List<dynamic>>(
      '/journal/trades',
      queryParameters: qp,
    );
    if (response.statusCode != 200 || response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/journal/trades failed: ${response.statusCode}',
      );
    }
    return response.data!
        .whereType<Map<String, dynamic>>()
        .map(ClosedTrade.fromJson)
        .toList();
  }

  /// `GET /news/feed` — market headlines (public RSS, fetched
  /// server-side) plus a Codex market briefing. `force` bypasses the
  /// backend's fetch-coalescing cache (wired to the panel's refresh
  /// button).
  Future<NewsFeed> fetchNewsFeed({bool force = false}) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '/news/feed',
      queryParameters: force ? <String, dynamic>{'force': true} : null,
    );
    if (response.statusCode != 200 || response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/news/feed failed: ${response.statusCode}',
      );
    }
    return NewsFeed.fromJson(response.data!);
  }

  /// `GET /risky/scenarios` — Risky/Growth time-to-target projection
  /// computed by the engine's own Brownian math (no hardcoded growth
  /// rates). `riskFraction` is clamped server-side to the signed Risky
  /// band [0.30, 0.50].
  Future<RiskyScenario> fetchRiskyScenarios({
    double? startingUsd,
    double? targetUsd,
    double? riskFraction,
    double? winRate,
    double? rewardToRisk,
    double? tradesPerDay,
  }) async {
    final qp = <String, dynamic>{};
    if (startingUsd != null) qp['startingUsd'] = startingUsd;
    if (targetUsd != null) qp['targetUsd'] = targetUsd;
    if (riskFraction != null) qp['riskFraction'] = riskFraction;
    if (winRate != null) qp['winRate'] = winRate;
    if (rewardToRisk != null) qp['rewardToRisk'] = rewardToRisk;
    if (tradesPerDay != null) qp['tradesPerDay'] = tradesPerDay;
    final response = await _dio.get<Map<String, dynamic>>(
      '/risky/scenarios',
      queryParameters: qp,
    );
    if (response.statusCode != 200 || response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/risky/scenarios failed: ${response.statusCode}',
      );
    }
    return RiskyScenario.fromJson(response.data!);
  }

  /// `GET /settings/presets` — fetch the named preset bundles
  /// (Conservative / Balanced / Aggressive). Each preset is a
  /// mapping of knob id → value that the operator can apply with
  /// one click.
  Future<KnobPresetCatalog> fetchKnobPresets() async {
    final response = await _dio.get<Map<String, dynamic>>('/settings/presets');
    if (response.statusCode != 200 || response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/settings/presets failed: ${response.statusCode}',
      );
    }
    return KnobPresetCatalog.fromJson(response.data!);
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
    String? uiLocale,
    bool? newsCalendarEnabled,
    String? newsCalendarSource,
    String? newsTradingMode,
    int? searchPopulation,
    int? searchGenerations,
    double? searchMaxHours,
    int? searchMaxIndicators,
    int? searchPortfolioSize,
    double? searchCorrThreshold,
    int? searchMaxRows,
  }) async {
    final body = <String, dynamic>{};
    if (dataDir != null) body['dataDir'] = dataDir;
    if (uiLocale != null) body['uiLocale'] = uiLocale;
    if (newsCalendarEnabled != null) {
      body['newsCalendarEnabled'] = newsCalendarEnabled;
    }
    if (newsCalendarSource != null) {
      body['newsCalendarSource'] = newsCalendarSource;
    }
    if (newsTradingMode != null) body['newsTradingMode'] = newsTradingMode;
    if (searchPopulation != null) body['searchPopulation'] = searchPopulation;
    if (searchGenerations != null) {
      body['searchGenerations'] = searchGenerations;
    }
    if (searchMaxHours != null) body['searchMaxHours'] = searchMaxHours;
    if (searchMaxIndicators != null) {
      body['searchMaxIndicators'] = searchMaxIndicators;
    }
    if (searchPortfolioSize != null) {
      body['searchPortfolioSize'] = searchPortfolioSize;
    }
    if (searchCorrThreshold != null) {
      body['searchCorrThreshold'] = searchCorrThreshold;
    }
    if (searchMaxRows != null) body['searchMaxRows'] = searchMaxRows;

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

  /// `GET /watchlist` (F-12) — the symbols the spot streamer is
  /// currently subscribed to. Returns the bare `symbols` list so the
  /// Market Watch "Edit watchlist" picker can round-trip the current
  /// selection.
  Future<List<String>> fetchWatchlist() async {
    final response = await _dio.get<Map<String, dynamic>>('/watchlist');
    final raw = response.data?['symbols'] as List?;
    return (raw ?? const []).map((e) => e as String).toList(growable: false);
  }

  /// `POST /watchlist` (F-12) — replace the streamer's subscription
  /// list. The backend persists the new set, re-subscribes the live
  /// spot stream, and reports back how many it `saved`, the canonical
  /// `symbols` it landed on, and whether it had to `restart` the
  /// streamer to apply. Returns a [WatchlistSaveResult].
  Future<WatchlistSaveResult> saveWatchlist(List<String> symbols) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '/watchlist',
      data: {'symbols': symbols},
    );
    if (response.statusCode != 200 || response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: 'POST /watchlist failed: ${response.statusCode}',
      );
    }
    return WatchlistSaveResult.fromJson(response.data!);
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
  ///
  /// #194: optional GA hyperparameter overrides. When `null`, the
  /// engine uses defaults from `config.yaml`. Passing any non-null
  /// value here lets the operator tune that knob from the Advanced
  /// expander without editing YAML.
  ///
  /// #264 (2026-05-29): [higherTfs] threads the multi-TF UI chip
  /// selection through to the backend's `StartJobBody.higher_tfs`
  /// (`engines_control.rs:48`). When `null` or empty, the server
  /// uses `DEFAULT_HIGHER_TFS = ["M5", "M15", "H1"]`. Passing
  /// e.g. `["M15", "H1", "H4"]` overrides only the higher-TF context
  /// for this run; everything else (symbol/base_tf/GA knobs) is
  /// unaffected. The list is sent as a JSON array of canonical
  /// timeframe labels; the server uppercases + trims defensively.
  Future<Map<String, dynamic>> startDiscovery({
    String? symbol,
    String? baseTf,
    List<String>? higherTfs,
    int? population,
    int? generations,
    int? maxIndicators,
    int? targetCandidates,
    int? portfolioSize,
  }) async {
    final body = <String, dynamic>{};
    if (symbol != null && symbol.trim().isNotEmpty) body['symbol'] = symbol;
    if (baseTf != null && baseTf.trim().isNotEmpty) body['base_tf'] = baseTf;
    if (higherTfs != null && higherTfs.isNotEmpty) {
      // De-dup + trim before sending so the server doesn't have to.
      // Skip the body field entirely when the cleaned set is empty so
      // the backend falls back to DEFAULT_HIGHER_TFS instead of
      // receiving an empty array (which it would also fall back on,
      // but the cleaner wire is easier to reason about in logs).
      final cleaned = higherTfs
          .map((tf) => tf.trim())
          .where((tf) => tf.isNotEmpty)
          .toSet()
          .toList(growable: false);
      if (cleaned.isNotEmpty) body['higher_tfs'] = cleaned;
    }
    if (population != null) body['population'] = population;
    if (generations != null) body['generations'] = generations;
    if (maxIndicators != null) body['max_indicators'] = maxIndicators;
    if (targetCandidates != null) body['target_candidates'] = targetCandidates;
    if (portfolioSize != null) body['portfolio_size'] = portfolioSize;
    final response = await _dio.post<Map<String, dynamic>>(
      '/engines/discovery/start',
      data: body.isEmpty ? null : body,
    );
    return response.data ?? const <String, dynamic>{};
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

  /// `GET /auth/codex/status` — does the operator have a ChatGPT
  /// subscription linked? Returns null fields when not logged in.
  Future<CodexStatusSnapshot> fetchCodexStatus() async {
    final r = await _dio.get<Map<String, dynamic>>('/auth/codex/status');
    return CodexStatusSnapshot.fromJson(r.data ?? const {});
  }

  /// `POST /auth/codex/start` — kicks off the PKCE OAuth flow.
  /// Returns the authorize URL the front-end opens in the browser.
  Future<CodexStartResponse> startCodexLogin() async {
    final r = await _dio.post<Map<String, dynamic>>('/auth/codex/start');
    return CodexStartResponse.fromJson(r.data ?? const {});
  }

  /// `POST /auth/codex/logout` — wipes `~/.codex/auth.json`.
  Future<void> logoutCodex() async {
    await _dio.post<void>('/auth/codex/logout');
  }

  /// `POST /codex/chat` — proxy a chat completion through the
  /// operator's ChatGPT subscription. 401 = re-auth needed.
  Future<CodexChatResponse> codexChat({
    required String prompt,
    String? model,
    int? maxTokens,
  }) async {
    final body = <String, dynamic>{'prompt': prompt};
    if (model != null) body['model'] = model;
    if (maxTokens != null) body['maxTokens'] = maxTokens;
    final r = await _dio.post<Map<String, dynamic>>(
      '/codex/chat',
      data: body,
    );
    return CodexChatResponse.fromJson(r.data ?? const {});
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
    return (raw ?? const []).map((e) => e as String).toList(growable: false);
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

  /// POST `/broker/account/select` (F-333) — set the *active* cTrader
  /// account. The backend reorders `broker_credentials.toml` so the
  /// picked cTID becomes first in the accounts list (the runtime always
  /// trades `accounts.first()`), adding it from the live OAuth grant if
  /// it wasn't persisted yet.
  ///
  /// Returns the backend payload:
  ///   `{ok: true, selectedAccountId: "...", requiresRestart: true}`.
  /// Runtime hot-swap isn't in scope yet, so `requiresRestart` is the
  /// honest signal that the operator must restart NeoEthos to apply.
  ///
  /// Throws `DioException` on 4xx/5xx — notably 404 when the id is in
  /// neither the on-disk list nor the current OAuth grant (stale UI /
  /// revoked access).
  Future<Map<String, dynamic>> selectBrokerAccount({
    required String accountId,
  }) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '/broker/account/select',
      data: {'accountId': accountId},
      // Selecting an account that isn't persisted yet triggers a live
      // /broker/accounts grant lookup server-side (WSS round-trip), so
      // give it headroom over the default 10 s receive timeout.
      options: Options(receiveTimeout: const Duration(seconds: 25)),
    );
    return response.data ?? const <String, dynamic>{};
  }

  /// GET `/settings/raw` (#193) — full `config.yaml` contents as a
  /// single string, plus the absolute on-disk path. Lets the Settings
  /// screen surface the 200+ knobs the typed `/settings` DTO can't.
  Future<Map<String, dynamic>> fetchRawConfigYaml() async {
    final response = await _dio.get<Map<String, dynamic>>('/settings/raw');
    return response.data ?? const <String, dynamic>{};
  }

  /// POST `/settings/raw` (F-312, 2026-05-29) — write the entire
  /// `config.yaml` verbatim. Closes the silent-drop hole where the
  /// typed `POST /settings` DTO only allowed 5 of the 200+ fields
  /// through. The backend validates the YAML against the `Settings`
  /// struct before writing, so a typo'd field surfaces here as a 400
  /// instead of waiting until the next discovery start.
  ///
  /// Returns the backend's structured success payload:
  ///   `{ok: true, path: "<abs>", backupPath: "<abs>", bytesWritten: N}`.
  /// On 400 (YAML/schema error), the response body carries
  /// `{error: "...", code: "yaml_parse_failed" | "yaml_schema_failed",
  ///  hint: "..."}` — surface verbatim to the operator so they can
  /// fix the typo without a guessing game.
  Future<Map<String, dynamic>> saveRawConfigYaml(String yaml) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '/settings/raw',
      data: {'yaml': yaml},
      // Bigger config files (~12 KB today, but room to grow) can take a
      // moment to schema-validate. Give the backend headroom over the
      // default 10 s receive timeout — 30 s covers worst-case fully-
      // populated configs without making a snappy save feel slow.
      options: Options(
        receiveTimeout: const Duration(seconds: 30),
        // Surface 4xx (validation errors) as Response, not exception,
        // so the UI can render the structured error body inline.
        validateStatus: (code) => code != null && code < 500,
      ),
    );
    return response.data ?? const <String, dynamic>{};
  }

  /// POST `/data/import` (#192) — convert a local CSV/Parquet/Arrow/
  /// JSON/JSONL/TSV file into the canonical Vortex layout under
  /// `data/symbol=<sym>/timeframe=<tf>/`. The source format is auto-
  /// detected from the file extension server-side.
  ///
  /// Returns the path the converted Vortex file landed at (so the UI
  /// can show "Imported to: `<path>`") plus the detected format string.
  Future<Map<String, dynamic>> importLocalFile({
    required String sourcePath,
    required String symbol,
    required String timeframe,
  }) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '/data/import',
      data: {
        'sourcePath': sourcePath,
        'symbol': symbol,
        'timeframe': timeframe,
      },
      options: Options(receiveTimeout: const Duration(seconds: 120)),
    );
    return response.data ?? const <String, dynamic>{};
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

  /// `/chart/history?symbol=&timeframe=&beforeMs=&limit=` — the next page
  /// of OLDER candles (strictly before `beforeMs`) for TradingView-style
  /// scroll-back. Broker-only, never persisted to disk. `hasMore == false`
  /// means the broker has no older bars, so the caller stops paginating.
  Future<ChartHistoryPage> fetchChartHistory({
    required String symbol,
    required String timeframe,
    required int beforeMs,
    int limit = 500,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '/chart/history',
      queryParameters: {
        'symbol': symbol,
        'timeframe': timeframe,
        'beforeMs': beforeMs,
        'limit': limit,
      },
    );
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/chart/history returned empty body',
      );
    }
    return ChartHistoryPage.fromJson(response.data!);
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

  /// `/live/spots` — current bid/ask cache populated by the
  /// long-running spot streamer (#137). Sub-2 s freshness for
  /// the symbols in the streamer's subscription list (forex
  /// majors today). UI components that need a live price
  /// (chart current-candle close, trade-watch PnL) poll this
  /// every 1 s.
  Future<LiveSpotsSnapshot> fetchLiveSpots() async {
    final response = await _dio.get<Map<String, dynamic>>('/live/spots');
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/live/spots returned 200 with empty body',
      );
    }
    return LiveSpotsSnapshot.fromJson(response.data!);
  }

  /// `/actions/pending` — list of LLM-proposed trade-management
  /// actions waiting for the operator (or recently-finalised ones
  /// kept around for audit / UI history). The banner polls this
  /// every 2 s while mounted; #136 backend caps the queue at 16
  /// entries and prunes >24 h old, so the response is small.
  Future<List<PendingAction>> fetchPendingActions() async {
    final response = await _dio.get<Map<String, dynamic>>('/actions/pending');
    final raw = response.data?['actions'] as List?;
    return (raw ?? const [])
        .map((e) => PendingAction.fromJson(e as Map<String, dynamic>))
        .toList(growable: false);
  }

  /// `POST /actions/{id}/confirm` — user clicked Confirm. Flips
  /// Pending→Confirmed server-side and dispatches the underlying
  /// broker call. The response carries `{ok:true, broker_outcome:
  /// {...}}` on a clean fill, or a 4xx/5xx with `{error, code}`
  /// when the broker rejects.
  ///
  /// [volumeUnitsOverride] lets the operator pick a partial-close
  /// volume even when the LLM proposed "close entire". Pass null
  /// to honour the LLM's proposal. The backend rejects volume == 0
  /// with `code:missing_volume`, so for "close entire" cases the
  /// UI must look up the position's actual volume and pass that.
  Future<Map<String, dynamic>> confirmPendingAction(
    String id, {
    int? volumeUnitsOverride,
  }) async {
    final body = <String, dynamic>{};
    if (volumeUnitsOverride != null) {
      body['volumeUnitsOverride'] = volumeUnitsOverride;
    }
    final response = await _dio.post<Map<String, dynamic>>(
      '/actions/$id/confirm',
      data: body.isEmpty ? null : body,
      // Confirm can take a few seconds (broker round-trip), give it
      // headroom over the default 10 s receive timeout.
      options: Options(
        receiveTimeout: const Duration(seconds: 30),
        // 4xx / 409 / 502 are all "real" responses from the backend
        // (expired, broker rejected, etc.) — let them flow back to
        // the caller so the UI can show the structured `code`.
        validateStatus: (code) => code != null && code < 600,
      ),
    );
    return response.data ?? const <String, dynamic>{};
  }

  /// `POST /actions/{id}/reject` — user clicked Reject. Flips
  /// Pending→Rejected server-side; no broker side effects. The
  /// optional [reason] is journalled to the audit JSONL so the LLM
  /// can later read "operator said X" and adjust.
  Future<Map<String, dynamic>> rejectPendingAction(
    String id, {
    String? reason,
  }) async {
    final body = <String, dynamic>{};
    if (reason != null && reason.trim().isNotEmpty) {
      body['reason'] = reason.trim();
    }
    final response = await _dio.post<Map<String, dynamic>>(
      '/actions/$id/reject',
      data: body.isEmpty ? null : body,
      options: Options(
        validateStatus: (code) => code != null && code < 600,
      ),
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

  /// `GET /strategy_lab/promotion?symbol=&base_tf=` (F-330) — the
  /// Promotion Gate verdict for a symbol/timeframe portfolio. Returns
  /// the aggregate backtest metrics, the per-criterion pass/fail
  /// breakdown, and the gate config thresholds so the UI can render
  /// "actual vs threshold" without hard-coding numbers. `aggregate`
  /// is null when there's no portfolio yet; `decision.criteria` is
  /// empty in that case with an actionable `summary`.
  Future<PromotionStatus> fetchPromotionStatus({
    required String symbol,
    required String baseTf,
  }) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '/strategy_lab/promotion',
      queryParameters: {'symbol': symbol, 'base_tf': baseTf},
    );
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: '/strategy_lab/promotion returned 200 with empty body',
      );
    }
    return PromotionStatus.fromJson(response.data!);
  }

  /// `POST /strategy_lab/promote` (F-330) — copy the staged model
  /// bundle to `live_models/<symbol>/<tf>/` if (and only if) the
  /// promotion gate passes.
  ///
  /// A blocked promotion comes back as **412 Precondition Failed**,
  /// which is a *valid* business response (the gate did its job), not
  /// a transport error. We widen `validateStatus` so the 412 body
  /// flows back as a parsed [PromoteResult] with `promoted:false` +
  /// the actionable `message`, instead of throwing. Real transport /
  /// 5xx failures still raise `DioException` for the caller's
  /// retry/error UI.
  Future<PromoteResult> promoteToLive({
    required String symbol,
    required String baseTf,
  }) async {
    final response = await _dio.post<Map<String, dynamic>>(
      '/strategy_lab/promote',
      data: {'symbol': symbol, 'baseTf': baseTf},
      // Copying the bundle is a filesystem round-trip (dozens of
      // files); give it headroom over the default 10 s.
      options: Options(
        receiveTimeout: const Duration(seconds: 30),
        // 412 = gate blocked (expected). Let everything below 500
        // through as a Response so we can parse `promoted:false`; only
        // 5xx + transport errors throw.
        validateStatus: (code) => code != null && code < 500,
      ),
    );
    if (response.data == null) {
      throw DioException(
        requestOptions: response.requestOptions,
        response: response,
        message: 'POST /strategy_lab/promote returned empty body '
            '(${response.statusCode})',
      );
    }
    return PromoteResult.fromJson(response.data!);
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

  /// **2026-05-25 — task #239**: Risky-Mode 24h re-arm cooldown
  /// remaining (seconds). `null` when no cooldown is active (the
  /// kill-switch has not tripped, or the 24 h window already elapsed).
  /// When non-null, the UI shows a countdown chip + blocks any
  /// "Arm Risky Mode" interaction with a modal explaining the
  /// cooldown is enforced and cannot be overridden.
  final int? riskyModeCooldownRemainingSecs;

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
    this.riskyModeCooldownRemainingSecs,
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
        riskyModeCooldownRemainingSecs:
            (j['riskyModeCooldownRemainingSecs'] as num?)?.toInt(),
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

  /// UI language code (`'en'` | `'el'`) from `SystemConfig.ui_locale`.
  /// Defaults to `'en'` for back-compat with a backend that predates the
  /// field. Drives the Settings language picker + the startup locale.
  final String localeCode;
  final bool newsCalendarEnabled;
  final String newsCalendarSource;

  /// Snake_case news-trading mode id from
  /// `crate::config::NewsTradingMode`. Empty when the backend
  /// predates the field — the UI treats that as `block_on_news`
  /// for safe back-compat.
  final String newsTradingMode;
  final String newsTradingModeDisplayName;

  // Discovery search knobs (models.prop_search_*) — 2026-06-01. Let the
  // operator tune search depth/budget from the UI (L40 VPS vs local).
  final int searchPopulation;
  final int searchGenerations;
  final double searchMaxHours;
  final int searchMaxIndicators;
  final int searchPortfolioSize;
  final double searchCorrThreshold;
  final int searchMaxRows;

  const SettingsSnapshot({
    required this.dataDir,
    this.localeCode = 'en',
    required this.newsCalendarEnabled,
    required this.newsCalendarSource,
    required this.newsTradingMode,
    required this.newsTradingModeDisplayName,
    this.searchPopulation = 100,
    this.searchGenerations = 50,
    this.searchMaxHours = 24,
    this.searchMaxIndicators = 0,
    this.searchPortfolioSize = 4,
    this.searchCorrThreshold = 0.85,
    this.searchMaxRows = 0,
  });

  factory SettingsSnapshot.fromJson(Map<String, dynamic> j) => SettingsSnapshot(
        dataDir: j['dataDir'] as String,
        localeCode: (j['uiLocale'] as String?) ?? 'en',
        newsCalendarEnabled: j['newsCalendarEnabled'] as bool,
        newsCalendarSource: j['newsCalendarSource'] as String,
        newsTradingMode: (j['newsTradingMode'] as String?) ?? '',
        newsTradingModeDisplayName:
            (j['newsTradingModeDisplayName'] as String?) ?? '',
        searchPopulation: (j['searchPopulation'] as num?)?.toInt() ?? 100,
        searchGenerations: (j['searchGenerations'] as num?)?.toInt() ?? 50,
        searchMaxHours: (j['searchMaxHours'] as num?)?.toDouble() ?? 24,
        searchMaxIndicators: (j['searchMaxIndicators'] as num?)?.toInt() ?? 0,
        searchPortfolioSize: (j['searchPortfolioSize'] as num?)?.toInt() ?? 4,
        searchCorrThreshold:
            (j['searchCorrThreshold'] as num?)?.toDouble() ?? 0.85,
        searchMaxRows: (j['searchMaxRows'] as num?)?.toInt() ?? 0,
      );
}

/// Computed trade-journal performance stats (mirrors the Rust
/// `JournalStats` wire DTO). Defensive parsing: missing/null fields fall
/// back to safe defaults so a partial payload never crashes the UI.
class JournalStats {
  final int totalTrades;
  final int wins;
  final int losses;
  final int breakeven;
  final double winRatePct;
  final double netProfit;
  final double grossProfit;
  final double grossLoss;
  final double? profitFactor;
  final double avgWin;
  final double avgLoss;
  final double? payoffRatio;
  final double expectancy;
  final double largestWin;
  final double largestLoss;
  final int maxConsecutiveWins;
  final int maxConsecutiveLosses;
  final double maxDrawdownAbs;
  final double maxDrawdownPct;
  final double? recoveryFactor;
  final double? sharpe;
  const JournalStats({
    this.totalTrades = 0,
    this.wins = 0,
    this.losses = 0,
    this.breakeven = 0,
    this.winRatePct = 0,
    this.netProfit = 0,
    this.grossProfit = 0,
    this.grossLoss = 0,
    this.profitFactor,
    this.avgWin = 0,
    this.avgLoss = 0,
    this.payoffRatio,
    this.expectancy = 0,
    this.largestWin = 0,
    this.largestLoss = 0,
    this.maxConsecutiveWins = 0,
    this.maxConsecutiveLosses = 0,
    this.maxDrawdownAbs = 0,
    this.maxDrawdownPct = 0,
    this.recoveryFactor,
    this.sharpe,
  });

  factory JournalStats.fromJson(Map<String, dynamic> j) => JournalStats(
        totalTrades: (j['totalTrades'] as num?)?.toInt() ?? 0,
        wins: (j['wins'] as num?)?.toInt() ?? 0,
        losses: (j['losses'] as num?)?.toInt() ?? 0,
        breakeven: (j['breakeven'] as num?)?.toInt() ?? 0,
        winRatePct: (j['winRatePct'] as num?)?.toDouble() ?? 0,
        netProfit: (j['netProfit'] as num?)?.toDouble() ?? 0,
        grossProfit: (j['grossProfit'] as num?)?.toDouble() ?? 0,
        grossLoss: (j['grossLoss'] as num?)?.toDouble() ?? 0,
        profitFactor: (j['profitFactor'] as num?)?.toDouble(),
        avgWin: (j['avgWin'] as num?)?.toDouble() ?? 0,
        avgLoss: (j['avgLoss'] as num?)?.toDouble() ?? 0,
        payoffRatio: (j['payoffRatio'] as num?)?.toDouble(),
        expectancy: (j['expectancy'] as num?)?.toDouble() ?? 0,
        largestWin: (j['largestWin'] as num?)?.toDouble() ?? 0,
        largestLoss: (j['largestLoss'] as num?)?.toDouble() ?? 0,
        maxConsecutiveWins: (j['maxConsecutiveWins'] as num?)?.toInt() ?? 0,
        maxConsecutiveLosses: (j['maxConsecutiveLosses'] as num?)?.toInt() ?? 0,
        maxDrawdownAbs: (j['maxDrawdownAbs'] as num?)?.toDouble() ?? 0,
        maxDrawdownPct: (j['maxDrawdownPct'] as num?)?.toDouble() ?? 0,
        recoveryFactor: (j['recoveryFactor'] as num?)?.toDouble(),
        sharpe: (j['sharpe'] as num?)?.toDouble(),
      );
}

/// One closed round-trip trade (mirrors the Rust `ClosedTrade` wire DTO).
class ClosedTrade {
  final int positionId;
  final String symbol;
  final String side;
  final double lots;
  final int? entryTsMs;
  final double? entryPrice;
  final int? exitTsMs;
  final double? exitPrice;
  final double grossProfit;
  final double commission;
  final double swap;
  final double netProfit;
  final double? balanceAfter;
  const ClosedTrade({
    required this.positionId,
    required this.symbol,
    required this.side,
    required this.lots,
    this.entryTsMs,
    this.entryPrice,
    this.exitTsMs,
    this.exitPrice,
    required this.grossProfit,
    required this.commission,
    required this.swap,
    required this.netProfit,
    this.balanceAfter,
  });

  factory ClosedTrade.fromJson(Map<String, dynamic> j) => ClosedTrade(
        positionId: (j['positionId'] as num?)?.toInt() ?? 0,
        symbol: (j['symbol'] as String?) ?? '',
        side: (j['side'] as String?) ?? '',
        lots: (j['lots'] as num?)?.toDouble() ?? 0,
        entryTsMs: (j['entryTsMs'] as num?)?.toInt(),
        entryPrice: (j['entryPrice'] as num?)?.toDouble(),
        exitTsMs: (j['exitTsMs'] as num?)?.toInt(),
        exitPrice: (j['exitPrice'] as num?)?.toDouble(),
        grossProfit: (j['grossProfit'] as num?)?.toDouble() ?? 0,
        commission: (j['commission'] as num?)?.toDouble() ?? 0,
        swap: (j['swap'] as num?)?.toDouble() ?? 0,
        netProfit: (j['netProfit'] as num?)?.toDouble() ?? 0,
        balanceAfter: (j['balanceAfter'] as num?)?.toDouble(),
      );
}

/// One headline from the AI news desk (`GET /news/feed`).
class NewsItem {
  final String title;
  final String link;
  final String source;
  final int? publishedMs;
  final String blurb;
  const NewsItem({
    required this.title,
    required this.link,
    required this.source,
    this.publishedMs,
    required this.blurb,
  });

  factory NewsItem.fromJson(Map<String, dynamic> j) => NewsItem(
        title: (j['title'] as String?) ?? '',
        link: (j['link'] as String?) ?? '',
        source: (j['source'] as String?) ?? '',
        publishedMs: (j['publishedMs'] as num?)?.toInt(),
        blurb: (j['blurb'] as String?) ?? '',
      );
}

/// `GET /news/feed` payload: public-RSS headlines + a Codex market
/// briefing. `aiAvailable` is false when the operator hasn't connected
/// their ChatGPT subscription — the panel then shows headlines only.
class NewsFeed {
  final List<NewsItem> items;
  final String aiSummary;
  final bool aiAvailable;
  final int generatedAtMs;
  final String notice;
  const NewsFeed({
    required this.items,
    required this.aiSummary,
    required this.aiAvailable,
    required this.generatedAtMs,
    required this.notice,
  });

  factory NewsFeed.fromJson(Map<String, dynamic> j) => NewsFeed(
        items: ((j['items'] as List<dynamic>?) ?? const [])
            .whereType<Map<String, dynamic>>()
            .map(NewsItem.fromJson)
            .toList(),
        aiSummary: (j['aiSummary'] as String?) ?? '',
        aiAvailable: (j['aiAvailable'] as bool?) ?? false,
        generatedAtMs: (j['generatedAtMs'] as num?)?.toInt() ?? 0,
        notice: (j['notice'] as String?) ?? '',
      );
}

/// `GET /risky/scenarios` payload — Risky/Growth Mode time-to-target
/// projection, computed by the live engine (`risky_mode.rs`
/// `time_to_target_scenarios`). The `_days` fields are null when the
/// configured edge has non-positive expected log-growth (target not
/// reachable on average). `ruinProbability` is the engine's
/// Brownian-barrier estimate — NOT a UI heuristic.
class RiskyScenario {
  final double startingUsd;
  final double targetUsd;
  final double riskFraction;
  final double winRate;
  final double rewardToRisk;
  final double tradesPerDay;
  final int? bestCaseDays;
  final int? expectedDays;
  final int? conservativeDays;
  final double ruinProbability;
  final double riskFractionMin;
  final double riskFractionMax;
  const RiskyScenario({
    required this.startingUsd,
    required this.targetUsd,
    required this.riskFraction,
    required this.winRate,
    required this.rewardToRisk,
    required this.tradesPerDay,
    this.bestCaseDays,
    this.expectedDays,
    this.conservativeDays,
    required this.ruinProbability,
    required this.riskFractionMin,
    required this.riskFractionMax,
  });

  factory RiskyScenario.fromJson(Map<String, dynamic> j) => RiskyScenario(
        startingUsd: (j['startingUsd'] as num?)?.toDouble() ?? 0,
        targetUsd: (j['targetUsd'] as num?)?.toDouble() ?? 0,
        riskFraction: (j['riskFraction'] as num?)?.toDouble() ?? 0,
        winRate: (j['winRate'] as num?)?.toDouble() ?? 0,
        rewardToRisk: (j['rewardToRisk'] as num?)?.toDouble() ?? 0,
        tradesPerDay: (j['tradesPerDay'] as num?)?.toDouble() ?? 0,
        bestCaseDays: (j['bestCaseDays'] as num?)?.toInt(),
        expectedDays: (j['expectedDays'] as num?)?.toInt(),
        conservativeDays: (j['conservativeDays'] as num?)?.toInt(),
        ruinProbability: (j['ruinProbability'] as num?)?.toDouble() ?? 0,
        riskFractionMin: (j['riskFractionMin'] as num?)?.toDouble() ?? 0.30,
        riskFractionMax: (j['riskFractionMax'] as num?)?.toDouble() ?? 0.50,
      );
}

/// One named live-discovery counter row from `/engines/status`
/// (F-14). The backend emits raw snake_case `name`s (`generation`,
/// `population`, `candidates`, `filtered_candidates`, `portfolio`,
/// `archived_profitable`, …); the Discovery screen maps the meaningful
/// ones to friendly labels.
class EngineCounter {
  final String name;
  final int value;
  const EngineCounter({required this.name, required this.value});

  factory EngineCounter.fromJson(Map<String, dynamic> j) => EngineCounter(
        name: (j['name'] as String?) ?? '',
        value: (j['value'] as num?)?.toInt() ?? 0,
      );
}

class EnginesSnapshot {
  final String discovery;
  final String training;
  final String autoTrader;
  final String discoverySummary;
  final String trainingSummary;

  /// F-14 live discovery telemetry. `discoveryStage` is the current
  /// pipeline phase (`""` when idle), `discoveryPercent` is overall
  /// progress in 0.0..1.0, and `discoveryCounters` are the named
  /// generation / population / candidate tallies for the live stats
  /// panel. All default to empty/zero on servers that predate F-14.
  final String discoveryStage;
  final double discoveryPercent;
  final List<EngineCounter> discoveryCounters;
  const EnginesSnapshot({
    required this.discovery,
    required this.training,
    required this.autoTrader,
    required this.discoverySummary,
    required this.trainingSummary,
    this.discoveryStage = '',
    this.discoveryPercent = 0.0,
    this.discoveryCounters = const [],
  });
  factory EnginesSnapshot.fromJson(Map<String, dynamic> j) => EnginesSnapshot(
        discovery: j['discovery'] as String,
        training: j['training'] as String,
        autoTrader: j['autoTrader'] as String,
        discoverySummary: (j['discoverySummary'] as String?) ?? '',
        trainingSummary: (j['trainingSummary'] as String?) ?? '',
        discoveryStage: (j['discoveryStage'] as String?) ?? '',
        discoveryPercent: (j['discoveryPercent'] as num?)?.toDouble() ?? 0.0,
        discoveryCounters: ((j['discoveryCounters'] as List?) ?? const [])
            .map((e) => EngineCounter.fromJson(e as Map<String, dynamic>))
            .toList(growable: false),
      );

  bool get discoveryRunning => discovery.toLowerCase() == 'running';
  bool get trainingRunning => training.toLowerCase() == 'running';
}

/// Result of `POST /watchlist` (F-12). `saved` is the count the backend
/// persisted, `symbols` is the canonical list it landed on (may differ
/// from what was sent — de-duped / normalised), and `restarted` flags
/// whether the spot streamer had to be torn down + re-subscribed.
class WatchlistSaveResult {
  final int saved;
  final bool restarted;
  final List<String> symbols;
  const WatchlistSaveResult({
    required this.saved,
    required this.restarted,
    required this.symbols,
  });

  factory WatchlistSaveResult.fromJson(Map<String, dynamic> j) =>
      WatchlistSaveResult(
        saved: (j['saved'] as num?)?.toInt() ?? 0,
        restarted: (j['restarted'] as bool?) ?? false,
        symbols: ((j['symbols'] as List?) ?? const [])
            .map((e) => e as String)
            .toList(growable: false),
      );
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

  /// #184: enabled metals — XAU/XAG/XPT/XPD heads, 6-letter codes.
  /// Surfaces gold, silver, platinum, palladium for the category
  /// chips that want a non-forex slice.
  List<BrokerSymbol> get metalsEnabled => symbols.where((s) {
        if (!s.enabled || s.symbolName.length != 6) return false;
        const heads = ['XAU', 'XAG', 'XPT', 'XPD'];
        return heads.contains(s.symbolName.substring(0, 3).toUpperCase());
      }).toList(growable: false);

  /// #184: enabled-and-not-(forex|metals) — the residual that is most
  /// likely indices, equities, or other CFDs on a retail broker catalog.
  /// Defined by exclusion so we don't have to enumerate every ticker.
  List<BrokerSymbol> get equitiesAndIndicesEnabled {
    final forexNames = forexLikeEnabled.map((x) => x.symbolName).toSet();
    final metalNames = metalsEnabled.map((x) => x.symbolName).toSet();
    return symbols.where((s) {
      if (!s.enabled) return false;
      if (forexNames.contains(s.symbolName)) return false;
      if (metalNames.contains(s.symbolName)) return false;
      return true;
    }).toList(growable: false);
  }
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

class CodexStatusSnapshot {
  final bool authenticated;
  final String? email;
  final bool loginInProgress;
  final String? lastError;
  final String authPath;

  CodexStatusSnapshot({
    required this.authenticated,
    required this.email,
    required this.loginInProgress,
    required this.lastError,
    required this.authPath,
  });

  factory CodexStatusSnapshot.fromJson(Map<String, dynamic> json) =>
      CodexStatusSnapshot(
        authenticated: json['authenticated'] as bool? ?? false,
        email: json['email'] as String?,
        loginInProgress: json['loginInProgress'] as bool? ?? false,
        lastError: json['lastError'] as String?,
        authPath: json['authPath'] as String? ?? '',
      );
}

class CodexStartResponse {
  final String authorizeUrl;
  final int callbackPort;

  CodexStartResponse({required this.authorizeUrl, required this.callbackPort});

  factory CodexStartResponse.fromJson(Map<String, dynamic> json) =>
      CodexStartResponse(
        authorizeUrl: json['authorizeUrl'] as String? ?? '',
        callbackPort: (json['callbackPort'] as num?)?.toInt() ?? 1455,
      );
}

class CodexChatResponse {
  final String model;
  final String response;
  final int promptTokens;
  final int completionTokens;
  final int totalTokens;

  CodexChatResponse({
    required this.model,
    required this.response,
    required this.promptTokens,
    required this.completionTokens,
    required this.totalTokens,
  });

  factory CodexChatResponse.fromJson(Map<String, dynamic> json) =>
      CodexChatResponse(
        model: json['model'] as String? ?? '',
        response: json['response'] as String? ?? '',
        promptTokens: (json['promptTokens'] as num?)?.toInt() ?? 0,
        completionTokens: (json['completionTokens'] as num?)?.toInt() ?? 0,
        totalTokens: (json['totalTokens'] as num?)?.toInt() ?? 0,
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
  final String source;
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
    required this.source,
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
        source: (j['source'] as String?) ?? 'unknown',
      );

  bool get isDiskCache => source == 'disk-cache';
  bool get isBrokerSource => source == 'broker';
}

/// One page of older candles from `/chart/history` (scroll-back). The
/// candles are oldest→newest, all strictly before the requested cursor,
/// so the caller can splice them onto the FRONT of its list.
class ChartHistoryPage {
  final List<ChartCandle> candles;

  /// `false` once the broker returns nothing older — stop paginating.
  final bool hasMore;
  final String source;
  const ChartHistoryPage({
    required this.candles,
    required this.hasMore,
    required this.source,
  });
  factory ChartHistoryPage.fromJson(Map<String, dynamic> j) => ChartHistoryPage(
        candles: ((j['candles'] as List?) ?? const [])
            .map((e) => ChartCandle.fromJson(e as Map<String, dynamic>))
            .toList(growable: false),
        hasMore: (j['hasMore'] as bool?) ?? false,
        source: (j['source'] as String?) ?? 'unknown',
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
            .map((e) => DiscoveryTarget.fromJson(e as Map<String, dynamic>))
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
    if (totalBytes < 1024) {
      return '$totalBytes B';
    }
    if (totalBytes < 1024 * 1024) {
      return '${(totalBytes / 1024).toStringAsFixed(1)} KB';
    }
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
  factory DataBootstrapSnapshot.fromJson(Map<String, dynamic> j) =>
      DataBootstrapSnapshot(
        dataDir: j['dataDir'] as String,
        dataDirExists: j['dataDirExists'] as bool,
        symbols: ((j['symbols'] as List?) ?? const [])
            .map((s) => s as String)
            .toList(growable: false),
        fileCount: j['fileCount'] as int,
        lastTouchedUnixMs: j['lastTouchedUnixMs'] as int?,
      );
}

/// Mirror of the `SpotTickDto` wire shape from `/live/spots`.
/// Already in camelCase on the wire (Rust side uses
/// `#[serde(rename = "...")]` per field), so the parsing is
/// straightforward.
class LiveSpotTick {
  final int symbolId;
  final String symbolName;
  final double? bid;
  final double? ask;
  final double? midPrice;
  final int receivedAtUnixMs;
  final int? brokerTimestampMs;

  /// Seconds since this tick was received server-side. The UI
  /// uses this for a "stale tick" warning when freshness > 5 s.
  final double freshnessSeconds;
  const LiveSpotTick({
    required this.symbolId,
    required this.symbolName,
    required this.bid,
    required this.ask,
    required this.midPrice,
    required this.receivedAtUnixMs,
    required this.brokerTimestampMs,
    required this.freshnessSeconds,
  });

  factory LiveSpotTick.fromJson(Map<String, dynamic> j) => LiveSpotTick(
        symbolId: (j['symbolId'] as num?)?.toInt() ?? 0,
        symbolName: (j['symbolName'] as String?) ?? '',
        bid: (j['bid'] as num?)?.toDouble(),
        ask: (j['ask'] as num?)?.toDouble(),
        midPrice: (j['midPrice'] as num?)?.toDouble(),
        receivedAtUnixMs: (j['receivedAtUnixMs'] as num?)?.toInt() ?? 0,
        brokerTimestampMs: (j['brokerTimestampMs'] as num?)?.toInt(),
        freshnessSeconds: (j['freshnessSeconds'] as num?)?.toDouble() ?? 0.0,
      );

  bool get isStale => freshnessSeconds > 5.0;
}

class LiveSpotsSnapshot {
  final List<LiveSpotTick> spots;
  final int snapshotAtUnixMs;
  final int symbolCount;
  const LiveSpotsSnapshot({
    required this.spots,
    required this.snapshotAtUnixMs,
    required this.symbolCount,
  });

  factory LiveSpotsSnapshot.fromJson(Map<String, dynamic> j) =>
      LiveSpotsSnapshot(
        spots: ((j['spots'] as List?) ?? const [])
            .map((e) => LiveSpotTick.fromJson(e as Map<String, dynamic>))
            .toList(growable: false),
        snapshotAtUnixMs: (j['snapshotAtUnixMs'] as num?)?.toInt() ?? 0,
        symbolCount: (j['symbolCount'] as num?)?.toInt() ?? 0,
      );

  /// Empty placeholder used by the SSE provider (#237) when the
  /// stream times out on initial connect. UI renders this as "no
  /// data yet" without throwing — operator sees a clean loading
  /// state until the SSE delivers the first event.
  factory LiveSpotsSnapshot.empty() => const LiveSpotsSnapshot(
        spots: [],
        snapshotAtUnixMs: 0,
        symbolCount: 0,
      );

  LiveSpotsSnapshot mergeTick(LiveSpotTick tick) {
    var replaced = false;
    final merged = <LiveSpotTick>[];
    for (final existing in spots) {
      if (_sameSpotIdentity(existing, tick)) {
        merged.add(_mergeSpotTick(existing, tick));
        replaced = true;
      } else {
        merged.add(existing);
      }
    }
    if (!replaced) merged.add(tick);
    return LiveSpotsSnapshot(
      spots: merged,
      snapshotAtUnixMs: tick.receivedAtUnixMs,
      symbolCount: merged.length,
    );
  }

  /// O(n) lookup by symbol name. The snapshot is small (≤ 8 majors
  /// in v1) so the linear scan is cheap; promoting to a map only
  /// pays off if the subscription set grows beyond ~50.
  LiveSpotTick? lookup(String symbol) {
    for (final s in spots) {
      if (s.symbolName.toUpperCase() == symbol.toUpperCase()) return s;
    }
    return null;
  }
}

bool _sameSpotIdentity(LiveSpotTick a, LiveSpotTick b) {
  if (a.symbolId != 0 && b.symbolId != 0) return a.symbolId == b.symbolId;
  return a.symbolName.toUpperCase() == b.symbolName.toUpperCase();
}

LiveSpotTick _mergeSpotTick(LiveSpotTick existing, LiveSpotTick update) {
  final bid = update.bid ?? existing.bid;
  final ask = update.ask ?? existing.ask;
  final mid = (bid != null && ask != null)
      ? (bid + ask) / 2.0
      : update.midPrice ?? existing.midPrice;
  return LiveSpotTick(
    symbolId: update.symbolId != 0 ? update.symbolId : existing.symbolId,
    symbolName:
        update.symbolName.isNotEmpty ? update.symbolName : existing.symbolName,
    bid: bid,
    ask: ask,
    midPrice: mid,
    receivedAtUnixMs: update.receivedAtUnixMs,
    brokerTimestampMs: update.brokerTimestampMs ?? existing.brokerTimestampMs,
    freshnessSeconds: update.freshnessSeconds,
  );
}

/// Mirror of `crate::app_services::pending_actions::PendingAction`.
///
/// The Rust struct uses serde defaults (no `rename_all = "camelCase"`),
/// so wire fields stay snake_case: `proposed_at_unix_ms`,
/// `expires_at_unix_ms`, `result_note`, `status`. The `kind` field is
/// a tagged enum — serde emits a nested `{"kind": "close_position",
/// "position_id": ..., ...}` object, so we keep the discriminant in
/// `kindTag` and surface the close-position fields as nullable here
/// (today only one variant exists; new ones go behind explicit code
/// changes per #136's whitelist guarantee).
class PendingAction {
  final String id;
  final String kindTag;
  final int? positionId;
  final int? volumeUnits;
  final String? symbolHint;
  final String reason;
  final int proposedAtUnixMs;
  final int expiresAtUnixMs;

  /// Snake-case state: `pending`, `confirmed`, `rejected`, `expired`,
  /// `executed`, `failed`. The widget switches on these directly.
  final String status;
  final String resultNote;

  const PendingAction({
    required this.id,
    required this.kindTag,
    required this.positionId,
    required this.volumeUnits,
    required this.symbolHint,
    required this.reason,
    required this.proposedAtUnixMs,
    required this.expiresAtUnixMs,
    required this.status,
    required this.resultNote,
  });

  factory PendingAction.fromJson(Map<String, dynamic> j) {
    final kindRaw = j['kind'];
    String tag = '';
    int? posId;
    int? volUnits;
    String? symHint;
    if (kindRaw is Map<String, dynamic>) {
      tag = (kindRaw['kind'] as String?) ?? '';
      posId = (kindRaw['position_id'] as num?)?.toInt();
      volUnits = (kindRaw['volume_units'] as num?)?.toInt();
      symHint = kindRaw['symbol_hint'] as String?;
    }
    return PendingAction(
      id: (j['id'] as String?) ?? '',
      kindTag: tag,
      positionId: posId,
      volumeUnits: volUnits,
      symbolHint: symHint,
      reason: (j['reason'] as String?) ?? '',
      proposedAtUnixMs: (j['proposed_at_unix_ms'] as num?)?.toInt() ?? 0,
      expiresAtUnixMs: (j['expires_at_unix_ms'] as num?)?.toInt() ?? 0,
      status: (j['status'] as String?) ?? 'pending',
      resultNote: (j['result_note'] as String?) ?? '',
    );
  }

  bool get isPending => status == 'pending';
  bool get isTerminal =>
      status == 'executed' ||
      status == 'failed' ||
      status == 'rejected' ||
      status == 'expired';

  /// Human summary mirroring the Rust `ActionKind::summary()`. Kept
  /// in Dart so we don't have to wait for a server round-trip after
  /// the user clicks Confirm — the banner re-renders instantly.
  String get summary {
    if (kindTag == 'close_position') {
      final vol = (volumeUnits ?? 0) == 0 ? 'entire' : '${volumeUnits!} units';
      final sym =
          (symbolHint == null || symbolHint!.isEmpty) ? '?' : symbolHint!;
      return 'Close $vol of position #${positionId ?? 0} ($sym)';
    }
    return kindTag.isEmpty ? 'Unknown action' : 'Action: $kindTag';
  }

  /// Seconds remaining until `expires_at_unix_ms`. Negative when
  /// already past expiry (sweep_expired hasn't run yet on the
  /// server). Used by the banner's countdown badge.
  int secondsUntilExpiry({DateTime? now}) {
    final nowMs = (now ?? DateTime.now().toUtc()).millisecondsSinceEpoch;
    return ((expiresAtUnixMs - nowMs) / 1000).round();
  }
}

// ============================================================================
// Task #238 — AdvancedSettings knob catalog DTOs.

/// One configurable knob. Mirrors the backend `KnobEntry` struct
/// (see crates/neoethos-app/src/server/knob_catalog.rs).
class KnobEntry {
  final String id;
  final String section;
  final String label;
  final String envVar;
  final String kind; // "Int" | "Float" | "Bool" | "Text" | "Enum" | "Path"
  final dynamic defaultValue;
  final dynamic currentValue;
  final String helpShort;
  final String helpLong;
  final dynamic presetConservative;
  final dynamic presetBalanced;
  final dynamic presetAggressive;

  /// Enum: list of valid choices. `null` for non-Enum kinds.
  final List<String>? enumChoices;

  /// Numeric ranges (Int/Float). `null` for non-numeric kinds.
  final double? minValue;
  final double? maxValue;

  const KnobEntry({
    required this.id,
    required this.section,
    required this.label,
    required this.envVar,
    required this.kind,
    required this.defaultValue,
    required this.currentValue,
    required this.helpShort,
    required this.helpLong,
    required this.presetConservative,
    required this.presetBalanced,
    required this.presetAggressive,
    this.enumChoices,
    this.minValue,
    this.maxValue,
  });

  factory KnobEntry.fromJson(Map<String, dynamic> j) => KnobEntry(
        id: j['id'] as String,
        section: j['section'] as String? ?? 'Other',
        label: j['label'] as String? ?? (j['id'] as String),
        envVar: j['envVar'] as String? ?? '',
        kind: j['kind'] as String? ?? 'Text',
        defaultValue: j['default'],
        currentValue: j['current'],
        helpShort: j['helpShort'] as String? ?? '',
        helpLong: j['helpLong'] as String? ?? '',
        presetConservative: j['presetConservative'],
        presetBalanced: j['presetBalanced'],
        presetAggressive: j['presetAggressive'],
        enumChoices: (j['enumChoices'] as List?)?.cast<String>(),
        minValue: (j['min'] as num?)?.toDouble(),
        maxValue: (j['max'] as num?)?.toDouble(),
      );

  /// Whether the current value differs from the active preset.
  /// Used to render the "dirty dot" indicator next to the knob
  /// label so the operator can see at a glance which knobs they
  /// have customised away from the preset baseline.
  bool isDirtyVs(String activePreset) {
    final preset = activePreset == 'conservative'
        ? presetConservative
        : activePreset == 'aggressive'
            ? presetAggressive
            : presetBalanced;
    return '$currentValue' != '$preset';
  }
}

class KnobCatalog {
  final int schemaVersion;
  final List<KnobEntry> knobs;
  const KnobCatalog({required this.schemaVersion, required this.knobs});

  factory KnobCatalog.fromJson(Map<String, dynamic> j) => KnobCatalog(
        schemaVersion: (j['schemaVersion'] as num?)?.toInt() ?? 1,
        knobs: ((j['knobs'] as List?) ?? const [])
            .map((e) => KnobEntry.fromJson(e as Map<String, dynamic>))
            .toList(growable: false),
      );

  /// Sections in canonical display order — useful for the left-pane
  /// section list in the AdvancedSettings screen.
  List<String> get sections {
    final seen = <String>{};
    final out = <String>[];
    for (final k in knobs) {
      if (seen.add(k.section)) out.add(k.section);
    }
    return out;
  }

  List<KnobEntry> knobsInSection(String section) =>
      knobs.where((k) => k.section == section).toList(growable: false);
}

class KnobPresetCatalog {
  /// Map preset id ("conservative" / "balanced" / "aggressive") to
  /// the human-readable display label ("Conservative", ...).
  final Map<String, String> presets;
  const KnobPresetCatalog({required this.presets});

  factory KnobPresetCatalog.fromJson(Map<String, dynamic> j) {
    final map = <String, String>{};
    final raw = j['presets'];
    if (raw is List) {
      // **Bug-fix (2026-05-31)**: the backend's actual shape is a LIST
      // of objects — `{"presets":[{"id":"conservative","label":
      // "Conservative","description":"..."}]}` — NOT a Map. The old
      // `as Map<String,dynamic>?` cast threw "List<dynamic> is not a
      // subtype of Map<String,dynamic>?" which crashed the entire
      // Advanced Settings tab. Map each entry's id → label.
      for (final e in raw) {
        if (e is Map<String, dynamic>) {
          final id = e['id'] as String?;
          final label = e['label'] as String?;
          if (id != null && id.isNotEmpty) {
            map[id] = (label != null && label.isNotEmpty) ? label : id;
          }
        }
      }
    } else if (raw is Map<String, dynamic>) {
      // Legacy flat shape: {id: label}.
      raw.forEach((k, v) {
        if (v is String) map[k] = v;
      });
    } else {
      // Last-resort: top-level flat {id: label}.
      j.forEach((k, v) {
        if (v is String) map[k] = v;
      });
    }
    return KnobPresetCatalog(presets: map);
  }
}

// ============================================================================
// F-330 — Promotion Gate DTOs.
//
// Mirror of the camelCase wire shape from `GET /strategy_lab/promotion`
// (see crates/neoethos-app/src/server/strategy_lab.rs). `aggregate` is
// null for an empty/absent portfolio; `decision.criteria` is empty in
// that case and `decision.promoted` is false with an actionable summary.

/// Aggregate backtest metrics across the whole portfolio. Null when
/// there is no portfolio on disk yet (nothing discovered/trained).
class PromotionAggregate {
  final double sharpe;
  final double winRate;
  final double profitFactor;
  final double maxDrawdownPct;
  final int trades;
  const PromotionAggregate({
    required this.sharpe,
    required this.winRate,
    required this.profitFactor,
    required this.maxDrawdownPct,
    required this.trades,
  });

  factory PromotionAggregate.fromJson(Map<String, dynamic> j) =>
      PromotionAggregate(
        sharpe: (j['sharpe'] as num?)?.toDouble() ?? 0.0,
        winRate: (j['winRate'] as num?)?.toDouble() ?? 0.0,
        profitFactor: (j['profitFactor'] as num?)?.toDouble() ?? 0.0,
        maxDrawdownPct: (j['maxDrawdownPct'] as num?)?.toDouble() ?? 0.0,
        trades: (j['trades'] as num?)?.toInt() ?? 0,
      );
}

/// One gate criterion (e.g. "Sharpe ratio") with its measured value,
/// the configured threshold, and the comparison operator (">=" etc.)
/// so the UI can render "1.40 >= 1.00 ✓" without hard-coding logic.
class PromotionCriterion {
  final String name;
  final bool passed;
  final double actual;
  final double threshold;
  final String comparison;
  const PromotionCriterion({
    required this.name,
    required this.passed,
    required this.actual,
    required this.threshold,
    required this.comparison,
  });

  factory PromotionCriterion.fromJson(Map<String, dynamic> j) =>
      PromotionCriterion(
        name: (j['name'] as String?) ?? '',
        passed: (j['passed'] as bool?) ?? false,
        actual: (j['actual'] as num?)?.toDouble() ?? 0.0,
        threshold: (j['threshold'] as num?)?.toDouble() ?? 0.0,
        comparison: (j['comparison'] as String?) ?? '>=',
      );
}

/// The gate verdict: overall `promoted` flag, the per-criterion
/// breakdown, and a human-readable `summary` line.
class PromotionDecision {
  final bool promoted;
  final List<PromotionCriterion> criteria;
  final String summary;
  const PromotionDecision({
    required this.promoted,
    required this.criteria,
    required this.summary,
  });

  factory PromotionDecision.fromJson(Map<String, dynamic> j) =>
      PromotionDecision(
        promoted: (j['promoted'] as bool?) ?? false,
        criteria: ((j['criteria'] as List?) ?? const [])
            .map((e) => PromotionCriterion.fromJson(e as Map<String, dynamic>))
            .toList(growable: false),
        summary: (j['summary'] as String?) ?? '',
      );
}

/// The configured gate thresholds (from config.yaml). Surfaced so the
/// UI can show "gate enabled?" + the limits even before a portfolio
/// exists.
class PromotionConfig {
  final bool enabled;
  final double minSharpe;
  final double minWinRate;
  final double minProfitFactor;
  final double maxDrawdownPct;
  final int minTrades;
  const PromotionConfig({
    required this.enabled,
    required this.minSharpe,
    required this.minWinRate,
    required this.minProfitFactor,
    required this.maxDrawdownPct,
    required this.minTrades,
  });

  factory PromotionConfig.fromJson(Map<String, dynamic> j) => PromotionConfig(
        enabled: (j['enabled'] as bool?) ?? false,
        minSharpe: (j['minSharpe'] as num?)?.toDouble() ?? 0.0,
        minWinRate: (j['minWinRate'] as num?)?.toDouble() ?? 0.0,
        minProfitFactor: (j['minProfitFactor'] as num?)?.toDouble() ?? 0.0,
        maxDrawdownPct: (j['maxDrawdownPct'] as num?)?.toDouble() ?? 0.0,
        minTrades: (j['minTrades'] as num?)?.toInt() ?? 0,
      );
}

/// Top-level `GET /strategy_lab/promotion` response.
class PromotionStatus {
  final String symbol;
  final String baseTf;
  final int portfolioSize;

  /// Null when there is no portfolio yet (empty staging dir).
  final PromotionAggregate? aggregate;
  final PromotionDecision decision;
  final PromotionConfig config;
  const PromotionStatus({
    required this.symbol,
    required this.baseTf,
    required this.portfolioSize,
    required this.aggregate,
    required this.decision,
    required this.config,
  });

  factory PromotionStatus.fromJson(Map<String, dynamic> j) => PromotionStatus(
        symbol: (j['symbol'] as String?) ?? '',
        baseTf: (j['baseTf'] as String?) ?? '',
        portfolioSize: (j['portfolioSize'] as num?)?.toInt() ?? 0,
        aggregate: j['aggregate'] == null
            ? null
            : PromotionAggregate.fromJson(
                j['aggregate'] as Map<String, dynamic>),
        decision: PromotionDecision.fromJson(
          (j['decision'] as Map<String, dynamic>?) ?? const {},
        ),
        config: PromotionConfig.fromJson(
          (j['config'] as Map<String, dynamic>?) ?? const {},
        ),
      );
}

/// `POST /strategy_lab/promote` response. `promoted` is false on a 412
/// gate-block (a valid response, not an error) — the `message` carries
/// the actionable reason. On a 200, `filesCopied` + `liveModelsPath`
/// describe what landed where.
class PromoteResult {
  final bool promoted;
  final String symbol;
  final String baseTf;
  final String liveModelsPath;
  final int filesCopied;
  final String message;
  const PromoteResult({
    required this.promoted,
    required this.symbol,
    required this.baseTf,
    required this.liveModelsPath,
    required this.filesCopied,
    required this.message,
  });

  factory PromoteResult.fromJson(Map<String, dynamic> j) => PromoteResult(
        promoted: (j['promoted'] as bool?) ?? false,
        symbol: (j['symbol'] as String?) ?? '',
        baseTf: (j['baseTf'] as String?) ?? '',
        liveModelsPath: (j['liveModelsPath'] as String?) ?? '',
        filesCopied: (j['filesCopied'] as num?)?.toInt() ?? 0,
        message: (j['message'] as String?) ?? '',
      );
}
