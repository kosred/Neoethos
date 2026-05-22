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
}
