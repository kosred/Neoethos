import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../api/currency_format.dart';
import '../state/account_provider.dart';
import '../state/live_spots_provider.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

/// Trade Watch — open-positions strip with per-row Close button.
/// Closing fires `POST /positions/close` and refreshes the
/// `accountSnapshotProvider` so the row disappears once the broker
/// acks.

class TradeWatchScreen extends ConsumerStatefulWidget {
  const TradeWatchScreen({super.key});

  @override
  ConsumerState<TradeWatchScreen> createState() => _TradeWatchScreenState();
}

class _TradeWatchScreenState extends ConsumerState<TradeWatchScreen> {
  final Set<int> _closingPositionIds = {};

  Future<void> _closePosition(Position p) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Close ${p.symbol}?'),
        content: Text(
          'Close ${p.side} position #${p.positionId} '
          '(${p.volume.toStringAsFixed(2)} lots, '
          'PnL ${p.pnlUsd.toStringAsFixed(2)})?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(backgroundColor: ForexAiTokens.sell),
            onPressed: () => Navigator.pop(ctx, true),
            child: const Text('Close position'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;

    setState(() => _closingPositionIds.add(p.positionId));
    try {
      final r = await ref.read(backendClientProvider).closePosition(
            positionId: p.positionId,
            volume: p.volumeUnits,
          );
      ref.invalidate(accountSnapshotProvider);
      if (!mounted) return;
      final status = (r['status'] as String?) ?? '?';
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: status.toLowerCase().contains('accept') ||
                  status.toLowerCase().contains('fill')
              ? ForexAiTokens.buy
              : ForexAiTokens.warning,
          content: Text(
            'Close ${p.symbol} #${p.positionId}: $status',
          ),
        ),
      );
    } on DioException catch (e) {
      final body = e.response?.data;
      final msg = (body is Map && body['error'] is String)
          ? body['error'] as String
          : e.message ?? e.toString();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: ForexAiTokens.sell,
          content: Text('Close failed: $msg'),
          duration: const Duration(seconds: 6),
        ),
      );
    } finally {
      if (mounted) {
        setState(() => _closingPositionIds.remove(p.positionId));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final snapshot = ref.watch(accountSnapshotProvider);
    final positions = snapshot.valueOrNull?.positions ?? const <Position>[];
    final currency = snapshot.valueOrNull?.currency ?? 'EUR';
    final money = NumberFormat.currency(
      symbol: currencyGlyph(currency),
      decimalDigits: 2,
    );

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Trade Watch',
            subtitle: 'Live PnL strip · all open positions · per-row close',
          ),
          SectionCard(
            title: 'Positions',
            child: positions.isEmpty
                ? const Padding(
                    padding: EdgeInsets.symmetric(vertical: 8),
                    child: Text(
                      'No open positions.',
                      style: TextStyle(
                        color: ForexAiTokens.textMuted,
                        fontSize: 12,
                      ),
                    ),
                  )
                : Column(
                    children: [
                      for (final p in positions)
                        _PositionRow(
                          position: p,
                          money: money,
                          busy: _closingPositionIds.contains(p.positionId),
                          onClose: () => _closePosition(p),
                        ),
                    ],
                  ),
          ),
        ],
      ),
    );
  }
}

class _PositionRow extends ConsumerWidget {
  final Position position;
  final NumberFormat money;
  final bool busy;
  final VoidCallback onClose;
  const _PositionRow({
    required this.position,
    required this.money,
    required this.busy,
    required this.onClose,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final p = position;
    final sideColor =
        p.side.toUpperCase() == 'LONG' || p.side.toUpperCase() == 'BUY'
            ? ForexAiTokens.buy
            : ForexAiTokens.sell;
    // #142: detect whether this position's PnL pips number is being
    // driven by the live-spot stream (sub-2 s) or the broker's 5 s
    // poll. The bridge silently overrides pnl_pips on the wire; the
    // UI surfaces a ⚡ icon so the operator knows which positions
    // have a live overlay and which don't (e.g. exotic pairs not
    // in DEFAULT_STREAMED_SYMBOLS).
    final liveSnap = ref.watch(liveSpotsProvider).valueOrNull;
    final hasFreshTick = liveSnap?.lookup(p.symbol)?.isStale == false;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 90,
            child: Text(
              p.symbol,
              style: const TextStyle(fontWeight: FontWeight.w700),
            ),
          ),
          SizedBox(
            width: 60,
            child: Text(
              p.side,
              style: TextStyle(fontWeight: FontWeight.w700, color: sideColor),
            ),
          ),
          SizedBox(
            width: 80,
            child: Text(
              '${p.volume.toStringAsFixed(2)} lots',
              style: const TextStyle(color: ForexAiTokens.textMuted),
            ),
          ),
          SizedBox(
            width: 70,
            child: Text(
              '#${p.positionId}',
              style: const TextStyle(
                color: ForexAiTokens.textFaint,
                fontSize: 11,
              ),
            ),
          ),
          const Spacer(),
          // Pips column with live indicator when fresh tick is
          // driving the calculation. Two-line text so the row
          // height stays consistent with the broker-only case.
          SizedBox(
            width: 90,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Align(
                  alignment: Alignment.centerRight,
                  child: FittedBox(
                    fit: BoxFit.scaleDown,
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        if (hasFreshTick) ...[
                          const Tooltip(
                            message: 'Live tick (sub-2 s)',
                            child: Icon(
                              Icons.bolt,
                              size: 12,
                              color: ForexAiTokens.buy,
                            ),
                          ),
                          const SizedBox(width: 2),
                        ],
                        Text(
                          '${p.pnlPips >= 0 ? '+' : ''}${p.pnlPips.toStringAsFixed(1)} p',
                          style: TextStyle(
                            fontWeight: FontWeight.w700,
                            fontSize: 12,
                            color: p.pnlPips >= 0
                                ? ForexAiTokens.buy
                                : ForexAiTokens.sell,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                Text(
                  money.format(p.pnlUsd),
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 11,
                    color:
                        p.pnlUsd >= 0 ? ForexAiTokens.buy : ForexAiTokens.sell,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          OutlinedButton.icon(
            onPressed: busy || p.positionId <= 0 ? null : onClose,
            icon: busy
                ? const SizedBox(
                    width: 12,
                    height: 12,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.close, size: 14),
            label: const Text('Close'),
            style: OutlinedButton.styleFrom(
              foregroundColor: ForexAiTokens.sell,
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              minimumSize: const Size(0, 28),
              textStyle: const TextStyle(fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }
}
