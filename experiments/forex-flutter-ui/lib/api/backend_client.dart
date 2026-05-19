// Backend API client — talks to forex-app (existing CLI/IPC) +
// forex-gemma (G8 REST/SSE surface).
//
// Phase: skeleton with mocked responses. Real wiring lands when
// the Rust side ships `forex-server` (a binary that hosts the
// REST surface). Until then this client returns canned data
// matching the wire shapes defined in
// `crates/forex-gemma/src/api.rs`.

import 'package:dio/dio.dart';

class BackendConfig {
  final String baseUrl;
  const BackendConfig({this.baseUrl = 'http://127.0.0.1:7423'});
}

/// Stub data models — mirror the shapes from
/// `crates/forex-gemma/src/api.rs` (ChatRequest, ChatEvent,
/// SuggestionDecision, FeatureFlag) + forex-app trading state.
class Position {
  final String symbol;
  final String side;
  final double volume;
  final double pnlPips;
  final double pnlUsd;
  const Position(this.symbol, this.side, this.volume, this.pnlPips, this.pnlUsd);
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
}

class BackendClient {
  final Dio _dio;
  final BackendConfig config;

  BackendClient({Dio? dio, this.config = const BackendConfig()})
      : _dio = dio ??
            Dio(BaseOptions(
              baseUrl: const BackendConfig().baseUrl,
              connectTimeout: const Duration(seconds: 5),
              receiveTimeout: const Duration(seconds: 30),
            ));

  /// Fetch the current account snapshot. Returns canned data
  /// until the Rust server is live.
  Future<AccountSnapshot> fetchAccountSnapshot() async {
    // TODO(G8): GET /account/snapshot
    return const AccountSnapshot(
      balance: 10000.00,
      equity: 10243.55,
      freeMargin: 9762.40,
      usedMargin: 250.00,
      currency: 'EUR',
      positions: [
        Position('EURUSD', 'LONG', 0.10, 24.5, 23.65),
        Position('XAUUSD', 'SHORT', 0.02, -3.2, -6.40),
      ],
    );
  }

  /// Open a chat turn — returns a Stream of typed events. Once
  /// the Rust SSE endpoint is live, this wraps the SSE response
  /// in a Stream<ChatEventDto>.
  Stream<ChatEventDto> openChat({
    required String sessionId,
    required String prompt,
  }) async* {
    // TODO(G8): POST /gemma/chat → SSE stream
    yield ChatEventDto.refusedByGate(
      reason: 'Backend not yet running',
      cannedResponse:
          'Ο rust backend δεν τρέχει ακόμα. Όταν το forex-server '
          'εκκινήσει στο port ${config.baseUrl.split(":").last}, '
          'οι αληθινές απαντήσεις του Gemma θα ροή εδώ.',
    );
  }
}

/// Slim Dart twin of ChatEvent from
/// `crates/forex-gemma/src/api.rs`.
sealed class ChatEventDto {
  const ChatEventDto();
  factory ChatEventDto.refusedByGate({
    required String reason,
    required String cannedResponse,
  }) = ChatEventRefused;
  factory ChatEventDto.tokenDelta(String text) = ChatEventTokenDelta;
  factory ChatEventDto.toolResult({
    required String toolName,
    required String outcome,
  }) = ChatEventToolResult;
  factory ChatEventDto.tradePendingApproval({
    required String suggestionId,
    required String symbol,
    required String side,
    required int volume,
    required String reasoning,
  }) = ChatEventTradePending;
  factory ChatEventDto.turnFinished(int latencyMs) = ChatEventTurnFinished;
}

class ChatEventRefused extends ChatEventDto {
  final String reason;
  final String cannedResponse;
  const ChatEventRefused({
    required this.reason,
    required this.cannedResponse,
  });
}

class ChatEventTokenDelta extends ChatEventDto {
  final String text;
  const ChatEventTokenDelta(this.text);
}

class ChatEventToolResult extends ChatEventDto {
  final String toolName;
  final String outcome;
  const ChatEventToolResult({
    required this.toolName,
    required this.outcome,
  });
}

class ChatEventTradePending extends ChatEventDto {
  final String suggestionId;
  final String symbol;
  final String side;
  final int volume;
  final String reasoning;
  const ChatEventTradePending({
    required this.suggestionId,
    required this.symbol,
    required this.side,
    required this.volume,
    required this.reasoning,
  });
}

class ChatEventTurnFinished extends ChatEventDto {
  final int latencyMs;
  const ChatEventTurnFinished(this.latencyMs);
}
