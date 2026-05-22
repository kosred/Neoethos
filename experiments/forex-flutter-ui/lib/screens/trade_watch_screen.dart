import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../state/account_provider.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

/// Trade Watch — compact one-line-per-position strip aimed at the
/// "I want a glance over my running positions while another panel
/// has my attention" use-case. Same data source as Markets but a
/// denser layout.

class TradeWatchScreen extends ConsumerWidget {
  const TradeWatchScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final snapshot = ref.watch(accountSnapshotProvider);
    final positions = snapshot.valueOrNull?.positions ?? const [];
    final usdFmt = NumberFormat.currency(symbol: r'$', decimalDigits: 2);

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Trade Watch',
            subtitle: 'Live PnL strip · all open positions',
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
                        Padding(
                          padding: const EdgeInsets.symmetric(vertical: 4),
                          child: Row(
                            children: [
                              SizedBox(
                                width: 90,
                                child: Text(
                                  p.symbol,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w700,
                                  ),
                                ),
                              ),
                              SizedBox(
                                width: 60,
                                child: Text(
                                  p.side,
                                  style: TextStyle(
                                    fontWeight: FontWeight.w700,
                                    color: p.side.toUpperCase() == 'LONG' ||
                                            p.side.toUpperCase() == 'BUY'
                                        ? ForexAiTokens.buy
                                        : ForexAiTokens.sell,
                                  ),
                                ),
                              ),
                              SizedBox(
                                width: 80,
                                child: Text(
                                  '${p.volume.toStringAsFixed(2)} lots',
                                  style: const TextStyle(
                                    color: ForexAiTokens.textMuted,
                                  ),
                                ),
                              ),
                              const Spacer(),
                              Text(
                                usdFmt.format(p.pnlUsd),
                                style: TextStyle(
                                  fontWeight: FontWeight.w700,
                                  color: p.pnlUsd >= 0
                                      ? ForexAiTokens.buy
                                      : ForexAiTokens.sell,
                                ),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
          ),
        ],
      ),
    );
  }
}
