import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
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
      symbol: currency == 'EUR' ? '€' : r'$',
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

class _PositionRow extends StatelessWidget {
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
  Widget build(BuildContext context) {
    final p = position;
    final sideColor = p.side.toUpperCase() == 'LONG' ||
            p.side.toUpperCase() == 'BUY'
        ? ForexAiTokens.buy
        : ForexAiTokens.sell;
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
          Text(
            money.format(p.pnlUsd),
            style: TextStyle(
              fontWeight: FontWeight.w700,
              color: p.pnlUsd >= 0 ? ForexAiTokens.buy : ForexAiTokens.sell,
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
              padding:
                  const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              minimumSize: const Size(0, 28),
              textStyle: const TextStyle(fontSize: 11),
            ),
          ),
        ],
      ),
    );
  }
}
