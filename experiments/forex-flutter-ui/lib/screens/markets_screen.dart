import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

/// Markets — Phase 1 wiring shows the operator's open positions from
/// `/account/snapshot`. Live spot quotes per symbol (the second half
/// of what this screen will eventually own) need the `/quotes` SSE
/// endpoint, which is the next batch of Rust-side work.

class MarketsScreen extends ConsumerWidget {
  const MarketsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final snapshot = ref.watch(accountSnapshotProvider);
    final positions = snapshot.valueOrNull?.positions ?? const [];
    final usdFmt = NumberFormat.currency(symbol: r'$', decimalDigits: 2);
    final pipFmt = NumberFormat('+#,##0.0;-#,##0.0', 'en_US');

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Markets',
            subtitle: 'Open positions · symbol watchlist',
          ),
          SectionCard(
            title: 'Open Positions',
            child: positions.isEmpty
                ? Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    child: Text(
                      snapshot.hasError
                          ? 'Connection issue — positions unavailable.'
                          : 'No open positions on the connected account.',
                      style: const TextStyle(
                        color: ForexAiTokens.textMuted,
                        fontSize: 12,
                      ),
                    ),
                  )
                : Table(
                    defaultVerticalAlignment: TableCellVerticalAlignment.middle,
                    columnWidths: const {
                      0: FlexColumnWidth(2),
                      1: FlexColumnWidth(2),
                      2: FlexColumnWidth(2),
                      3: FlexColumnWidth(2),
                      4: FlexColumnWidth(2),
                    },
                    children: [
                      const TableRow(children: [
                        _Th('Symbol'),
                        _Th('Side'),
                        _Th('Volume'),
                        _Th('Pips'),
                        _Th('PnL'),
                      ]),
                      for (final p in positions)
                        TableRow(children: [
                          _Td(p.symbol),
                          _Td(
                            p.side,
                            color: p.side.toUpperCase() == 'LONG' ||
                                    p.side.toUpperCase() == 'BUY'
                                ? ForexAiTokens.buy
                                : ForexAiTokens.sell,
                          ),
                          _Td(p.volume.toStringAsFixed(2)),
                          _Td('${pipFmt.format(p.pnlPips)} pips'),
                          _Td(
                            usdFmt.format(p.pnlUsd),
                            color: p.pnlUsd >= 0
                                ? ForexAiTokens.buy
                                : ForexAiTokens.sell,
                          ),
                        ]),
                    ],
                  ),
          ),
          const SectionCard(
            title: 'Symbol Watchlist',
            child: _PendingNotice(
              line:
                  'Live tick stream lands when /quotes SSE endpoint ships. '
                  'For now, prices are intentionally absent rather than '
                  'fake — see the open-positions table above for the '
                  'broker-confirmed state.',
            ),
          ),
        ],
      ),
    );
  }
}

class _PendingNotice extends StatelessWidget {
  final String line;
  const _PendingNotice({required this.line});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Text(
          line,
          style: const TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
        ),
      );
}

class _Th extends StatelessWidget {
  final String text;
  const _Th(this.text);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Text(
          text.toUpperCase(),
          style: const TextStyle(
            fontSize: 10,
            letterSpacing: 0.4,
            color: ForexAiTokens.textMuted,
            fontWeight: FontWeight.w700,
          ),
        ),
      );
}

class _Td extends StatelessWidget {
  final String text;
  final Color? color;
  const _Td(this.text, {this.color});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Text(
          text,
          style: TextStyle(
            fontSize: 12,
            color: color ?? ForexAiTokens.textPrimary,
          ),
        ),
      );
}
