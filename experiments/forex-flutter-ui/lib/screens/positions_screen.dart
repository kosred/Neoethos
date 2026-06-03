// Positions — focused trade-monitoring hub (F-326 final).
//
// **F-321** lifted the old TradeWatch view into a top-level sidebar
// tab. **F-326 (2026-05-29 rebuild)** replaces that thin wrapper with
// the Bloomberg-style positions hub the Codex mockup specified:
//
//   ┌─────────────────────────────────────────────────────────────┐
//   │ 4 metric cards  │  Open · Used Margin · Floating · Today PnL│
//   ├─────────────────────────────────────────────────────────────┤
//   │ Open Positions table (full width)                            │
//   │  Side · Symbol · Volume · Open · Current · PnL pips · PnL$  │
//   │  · Since · Position ID · [Close]                             │
//   ├─────────────────────────────────────────────────────────────┤
//   │ Pending Orders table (placeholder until backend lands them)  │
//   └─────────────────────────────────────────────────────────────┘
//
// Every number is broker-sourced (matches operator's invariant:
// "ola ta noumera apo ton server"). Per-row Close button hits
// `/positions/close` and force-refreshes the snapshot so the row
// disappears within ~500 ms.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../api/currency_format.dart';
import '../api/error_translation.dart';
import '../state/account_provider.dart';
import '../state/live_spots_provider.dart';
import '../theme/theme.dart';

/// Trade-journal stats (computed server-side from closed trades + equity).
final journalStatsProvider = FutureProvider.autoDispose<JournalStats>((ref) {
  return ref.read(backendClientProvider).fetchJournalStats();
});

/// Closed-trade log, most-recent first (capped).
final journalTradesProvider =
    FutureProvider.autoDispose<List<ClosedTrade>>((ref) {
  return ref.read(backendClientProvider).fetchJournalTrades(limit: 200);
});

class PositionsScreen extends ConsumerWidget {
  const PositionsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final accountAsync = ref.watch(accountSnapshotProvider);
    final spotsAsync = ref.watch(liveSpotsProvider);

    return DefaultTabController(
      length: 2,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const TabBar(
            isScrollable: true,
            tabAlignment: TabAlignment.start,
            tabs: [Tab(text: 'Live'), Tab(text: 'Journal')],
          ),
          const SizedBox(height: NeoethosTokens.spSm),
          Expanded(
            child: TabBarView(
              children: [
                // ── Live tab — the existing positions hub ──
                Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _MetricsStrip(accountAsync: accountAsync),
                    const SizedBox(height: NeoethosTokens.spSm),
                    Expanded(
                      flex: 3,
                      child: _OpenPositionsTable(
                        accountAsync: accountAsync,
                        spotsAsync: spotsAsync,
                      ),
                    ),
                    const SizedBox(height: NeoethosTokens.spSm),
                    const Expanded(flex: 2, child: _PendingOrdersTable()),
                  ],
                ),
                // ── Journal tab — closed trades + computed stats ──
                const _JournalTab(),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Journal tab (#flagship) — closed-trade log + professional stats
// ---------------------------------------------------------------------------

class _JournalTab extends ConsumerWidget {
  const _JournalTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final statsAsync = ref.watch(journalStatsProvider);
    final tradesAsync = ref.watch(journalTradesProvider);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        statsAsync.when(
          loading: () => const SizedBox(
            height: 56,
            child: Center(child: CircularProgressIndicator()),
          ),
          error: (e, _) => Padding(
            padding: const EdgeInsets.all(8),
            child: Text(
              'Stats unavailable: ${describeError(e)}',
              style: const TextStyle(fontSize: 11, color: NeoethosTokens.sell),
            ),
          ),
          data: (s) => _JournalStatCards(stats: s),
        ),
        const SizedBox(height: NeoethosTokens.spSm),
        Expanded(
          child: tradesAsync.when(
            loading: () => const Center(child: CircularProgressIndicator()),
            error: (e, _) => Center(
              child: Text(
                'Could not load trades: ${describeError(e)}',
                style: const TextStyle(color: NeoethosTokens.sell),
              ),
            ),
            data: (trades) => trades.isEmpty
                ? const Center(
                    child: Padding(
                      padding: EdgeInsets.all(24),
                      child: Text(
                        'No closed trades yet.\nThey appear here as the bot closes positions.',
                        textAlign: TextAlign.center,
                        style: TextStyle(color: NeoethosTokens.textMuted),
                      ),
                    ),
                  )
                : _ClosedTradesTable(trades: trades),
          ),
        ),
      ],
    );
  }
}

class _JournalStatCards extends StatelessWidget {
  final JournalStats stats;
  const _JournalStatCards({required this.stats});

  Widget _chip(String label, String value, {Color? accent}) => SizedBox(
        width: 132,
        child: Card(
          child: Padding(
            padding: const EdgeInsets.all(8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  label,
                  style: const TextStyle(
                    fontSize: 10,
                    color: NeoethosTokens.textMuted,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  value,
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                    color: accent,
                  ),
                ),
              ],
            ),
          ),
        ),
      );

  @override
  Widget build(BuildContext context) {
    String f2(double v) => v.toStringAsFixed(2);
    String optF(double? v) => v == null ? '—' : v.toStringAsFixed(2);
    final netAccent = stats.netProfit > 0
        ? NeoethosTokens.buy
        : stats.netProfit < 0
            ? NeoethosTokens.sell
            : null;
    return Wrap(
      spacing: NeoethosTokens.spSm,
      runSpacing: NeoethosTokens.spSm,
      children: [
        _chip('Net P/L', f2(stats.netProfit), accent: netAccent),
        _chip('Trades', '${stats.totalTrades}'),
        _chip('Win rate', '${stats.winRatePct.toStringAsFixed(1)}%'),
        _chip('Profit factor', optF(stats.profitFactor)),
        _chip('Expectancy', f2(stats.expectancy)),
        _chip('Max DD', '${stats.maxDrawdownPct.toStringAsFixed(1)}%'),
        _chip('Sharpe', optF(stats.sharpe)),
      ],
    );
  }
}

class _ClosedTradesTable extends StatelessWidget {
  final List<ClosedTrade> trades;
  const _ClosedTradesTable({required this.trades});

  @override
  Widget build(BuildContext context) {
    final df = DateFormat('yyyy-MM-dd HH:mm');
    return ListView.separated(
      itemCount: trades.length,
      separatorBuilder: (_, __) => const Divider(height: 1),
      itemBuilder: (context, i) {
        final t = trades[i];
        final when = t.exitTsMs != null
            ? df.format(DateTime.fromMillisecondsSinceEpoch(t.exitTsMs!))
            : '—';
        final pnlColor = t.netProfit > 0
            ? NeoethosTokens.buy
            : t.netProfit < 0
                ? NeoethosTokens.sell
                : null;
        return ListTile(
          dense: true,
          title: Text('${t.symbol}   ${t.side}   ${t.lots} lots'),
          subtitle: Text(
            when,
            style: const TextStyle(fontSize: 10, color: NeoethosTokens.textMuted),
          ),
          trailing: Text(
            t.netProfit.toStringAsFixed(2),
            style: TextStyle(fontWeight: FontWeight.w600, color: pnlColor),
          ),
        );
      },
    );
  }
}

// ---------------------------------------------------------------------------
// Top metrics strip
// ---------------------------------------------------------------------------

class _MetricsStrip extends StatelessWidget {
  final AsyncValue<AccountSnapshot> accountAsync;
  const _MetricsStrip({required this.accountAsync});

  @override
  Widget build(BuildContext context) {
    final acc = accountAsync.valueOrNull;
    final positions = acc?.positions ?? const <Position>[];
    final openCount = positions.length;
    final usedMargin = acc?.usedMargin;
    final currency = acc?.currency ?? 'USD';
    final floating = positions.fold<double>(0, (sum, p) => sum + p.pnlUsd);
    final fmt = NumberFormat.currency(
      symbol: currencyGlyph(currency),
      decimalDigits: 2,
    );
    return Row(
      children: [
        _MetricCard(
          label: 'Open positions',
          value: openCount == 0 ? '—' : '$openCount',
          accent: openCount > 0 ? NeoethosTokens.accent : null,
        ),
        const SizedBox(width: NeoethosTokens.spSm),
        _MetricCard(
          label: 'Used margin',
          value: usedMargin == null ? '—' : fmt.format(usedMargin),
        ),
        const SizedBox(width: NeoethosTokens.spSm),
        _MetricCard(
          label: 'Floating PnL',
          value: openCount == 0 ? '—' : fmt.format(floating),
          accent: openCount == 0
              ? null
              : floating > 0
                  ? NeoethosTokens.buy
                  : floating < 0
                      ? NeoethosTokens.sell
                      : null,
        ),
        const SizedBox(width: NeoethosTokens.spSm),
        const _MetricCard(
          label: "Today's realised",
          // Realised PnL endpoint hasn't shipped yet — explicit `—`
          // beats a misleading "$0" placeholder.
          value: '—',
        ),
      ],
    );
  }
}

class _MetricCard extends StatelessWidget {
  final String label;
  final String value;
  final Color? accent;
  const _MetricCard({
    required this.label,
    required this.value,
    this.accent,
  });

  @override
  Widget build(BuildContext context) {
    final color = accent ?? NeoethosTokens.textPrimary;
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: NeoethosTokens.spMd,
          vertical: 10,
        ),
        decoration: BoxDecoration(
          color: NeoethosTokens.panelBg,
          border: Border.all(color: NeoethosTokens.border),
          borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              label.toUpperCase(),
              style: const TextStyle(
                fontSize: NeoethosTokens.fsCaption - 1,
                fontWeight: FontWeight.w800,
                letterSpacing: 0.8,
                color: NeoethosTokens.textMuted,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              value,
              style: TextStyle(
                fontSize: NeoethosTokens.fsSubtitle + 2,
                fontWeight: FontWeight.w800,
                color: color,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Open Positions table (main focus)
// ---------------------------------------------------------------------------

class _OpenPositionsTable extends ConsumerWidget {
  final AsyncValue<AccountSnapshot> accountAsync;
  final AsyncValue<LiveSpotsSnapshot> spotsAsync;
  const _OpenPositionsTable({
    required this.accountAsync,
    required this.spotsAsync,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      decoration: BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _PanelTitle(
            title: 'Open Positions',
            count: accountAsync.valueOrNull?.positions.length ?? 0,
            trailing: IconButton(
              tooltip: 'Refresh snapshot',
              iconSize: 16,
              padding: EdgeInsets.zero,
              constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
              onPressed: () => ref
                  .read(accountSnapshotProvider.notifier)
                  .refreshNow(),
              icon: const Icon(Icons.refresh,
                  color: NeoethosTokens.textMuted),
            ),
          ),
          _PositionsHeader(),
          Expanded(
            child: accountAsync.when(
              loading: () => const _LoadingLine(),
              error: (err, _) => _ErrorBlock(error: err),
              data: (acc) {
                if (acc.positions.isEmpty) {
                  return const _EmptyLine(
                    message: 'No open positions.\n'
                        'Trades placed via Market Watch → Order Ticket '
                        'will appear here within ~2 s.',
                  );
                }
                final spotByName = <String, LiveSpotTick>{};
                for (final s in spotsAsync.valueOrNull?.spots ??
                    const <LiveSpotTick>[]) {
                  spotByName[s.symbolName] = s;
                }
                final currencySymbol =
                    currencyGlyph(acc.currency);
                return Scrollbar(
                  child: ListView.builder(
                    itemCount: acc.positions.length,
                    itemBuilder: (context, i) => _PositionDetailRow(
                      position: acc.positions[i],
                      currencySymbol: currencySymbol,
                      currentSpot: spotByName[acc.positions[i].symbol],
                      stripe: i.isOdd,
                      onClose: () => _confirmClose(
                        context,
                        ref,
                        acc.positions[i],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _confirmClose(
    BuildContext context,
    WidgetRef ref,
    Position position,
  ) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: NeoethosTokens.panelBg,
        title: Text(
          'Close ${_prettySymbol(position.symbol)} ${position.side.toUpperCase()} '
              '${position.volume.toStringAsFixed(2)} lots?',
          style: const TextStyle(
            color: NeoethosTokens.textPrimary,
            fontSize: 14,
          ),
        ),
        content: Text(
          'Position #${position.positionId} · current PnL '
              '${position.pnlPips.toStringAsFixed(1)} pips. '
              'This sends a market close to the broker — irreversible.',
          style: const TextStyle(
            color: NeoethosTokens.textMuted,
            fontSize: 13,
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: NeoethosTokens.sell,
            ),
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Close position'),
          ),
        ],
      ),
    );
    if (ok != true) return;
    final client = ref.read(backendClientProvider);
    try {
      // /positions/close needs both positionId AND the volume_units
       // (broker requires the volume so the close exactly matches the
       // open quantity — partial closes use a smaller volume).
      await client.closePosition(
        positionId: position.positionId,
        volume: position.volumeUnits,
      );
      await ref
          .read(accountSnapshotProvider.notifier)
          .refreshNow();
    } catch (e) {
      if (!context.mounted) return;
      showTranslatedErrorSnackbar(
        context,
        e,
        prefix: 'Position close was not confirmed — '
            'check the Positions list; if it still shows open, try again',
      );
    }
  }
}

class _PositionsHeader extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: const BoxDecoration(
        color: NeoethosTokens.appBg,
        border: Border(
          bottom: BorderSide(color: NeoethosTokens.border),
        ),
      ),
      child: const Row(
        children: [
          SizedBox(width: 56, child: _HeaderCell('Side')),
          SizedBox(width: 100, child: _HeaderCell('Symbol')),
          SizedBox(width: 70, child: _HeaderCell('Volume', right: true)),
          SizedBox(width: 80, child: _HeaderCell('Current', right: true)),
          SizedBox(width: 80, child: _HeaderCell('PnL pips', right: true)),
          SizedBox(width: 100, child: _HeaderCell('PnL', right: true)),
          Expanded(child: _HeaderCell('Since · ID')),
          SizedBox(width: 80, child: _HeaderCell('Action', right: true)),
        ],
      ),
    );
  }
}

class _HeaderCell extends StatelessWidget {
  final String text;
  final bool right;
  const _HeaderCell(this.text, {this.right = false});
  @override
  Widget build(BuildContext context) => Text(
        text.toUpperCase(),
        textAlign: right ? TextAlign.right : TextAlign.left,
        style: const TextStyle(
          fontSize: NeoethosTokens.fsCaption - 1,
          fontWeight: FontWeight.w800,
          letterSpacing: 0.8,
          color: NeoethosTokens.textFaint,
        ),
      );
}

class _PositionDetailRow extends StatelessWidget {
  final Position position;
  final String currencySymbol;
  final LiveSpotTick? currentSpot;
  final bool stripe;
  final VoidCallback onClose;
  const _PositionDetailRow({
    required this.position,
    required this.currencySymbol,
    required this.currentSpot,
    required this.stripe,
    required this.onClose,
  });

  @override
  Widget build(BuildContext context) {
    final isBuy = position.side.toLowerCase() == 'buy';
    final sideColor = isBuy ? NeoethosTokens.buy : NeoethosTokens.sell;
    final pnlColor = position.pnlUsd > 0
        ? NeoethosTokens.buy
        : position.pnlUsd < 0
            ? NeoethosTokens.sell
            : NeoethosTokens.textMuted;
    final since = position.openTimestampMs == null
        ? '—'
        : DateFormat('HH:mm:ss').format(
            DateTime.fromMillisecondsSinceEpoch(
              position.openTimestampMs!,
            ),
          );
    final currentPrice = isBuy ? currentSpot?.bid : currentSpot?.ask;
    final currentText = currentPrice == null
        ? '—'
        : currentPrice.toStringAsFixed(_isJpy(position.symbol) ? 3 : 5);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: stripe
            ? NeoethosTokens.appBg.withValues(alpha: 0.4)
            : Colors.transparent,
        border: const Border(
          bottom: BorderSide(color: NeoethosTokens.border, width: 0.4),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          SizedBox(
            width: 56,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: sideColor.withValues(alpha: 0.18),
                border: Border.all(color: sideColor.withValues(alpha: 0.6)),
                borderRadius: BorderRadius.circular(3),
              ),
              child: Text(
                isBuy ? 'BUY' : 'SELL',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: NeoethosTokens.fsCaption - 1,
                  fontWeight: FontWeight.w800,
                  color: sideColor,
                ),
              ),
            ),
          ),
          SizedBox(
            width: 100,
            child: Text(
              _prettySymbol(position.symbol),
              style: const TextStyle(
                fontSize: NeoethosTokens.fsBody,
                fontWeight: FontWeight.w700,
                color: NeoethosTokens.textPrimary,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ),
          SizedBox(
            width: 70,
            child: Text(
              position.volume.toStringAsFixed(2),
              textAlign: TextAlign.right,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsBody,
                fontWeight: FontWeight.w700,
                color: NeoethosTokens.textPrimary,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ),
          SizedBox(
            width: 80,
            child: Text(
              currentText,
              textAlign: TextAlign.right,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsBody,
                color: NeoethosTokens.textMuted,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ),
          SizedBox(
            width: 80,
            child: Text(
              position.pnlPips.toStringAsFixed(1),
              textAlign: TextAlign.right,
              style: TextStyle(
                fontSize: NeoethosTokens.fsBody,
                fontWeight: FontWeight.w800,
                color: pnlColor,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ),
          SizedBox(
            width: 100,
            child: Text(
              '$currencySymbol${position.pnlUsd.toStringAsFixed(2)}',
              textAlign: TextAlign.right,
              style: TextStyle(
                fontSize: NeoethosTokens.fsBody,
                fontWeight: FontWeight.w800,
                color: pnlColor,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 8),
              child: Text(
                '$since · #${position.positionId}',
                overflow: TextOverflow.ellipsis,
                style: const TextStyle(
                  fontSize: NeoethosTokens.fsCaption,
                  color: NeoethosTokens.textFaint,
                  fontFeatures: [FontFeature.tabularFigures()],
                ),
              ),
            ),
          ),
          SizedBox(
            width: 80,
            child: OutlinedButton(
              onPressed: onClose,
              style: OutlinedButton.styleFrom(
                foregroundColor: NeoethosTokens.sell,
                side: BorderSide(
                  color: NeoethosTokens.sell.withValues(alpha: 0.55),
                ),
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                minimumSize: const Size(0, 26),
                tapTargetSize: MaterialTapTargetSize.shrinkWrap,
              ),
              child: const Text(
                'Close',
                style: TextStyle(
                  fontSize: NeoethosTokens.fsCaption,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Pending Orders panel
// ---------------------------------------------------------------------------

class _PendingOrdersTable extends StatelessWidget {
  const _PendingOrdersTable();

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _PanelTitle(title: 'Pending Orders', count: 0),
          Expanded(
            child: _EmptyLine(
              message:
                  'No pending orders.\n\n'
                  'Limit / Stop entries placed via Order Ticket appear '
                  'here. The dedicated /orders/pending broker endpoint '
                  'is deferred until the live-pending-fill flow ships.',
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Shared primitives
// ---------------------------------------------------------------------------

class _PanelTitle extends StatelessWidget {
  final String title;
  final int count;
  final Widget? trailing;
  const _PanelTitle({
    required this.title,
    required this.count,
    this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: const BoxDecoration(
        color: NeoethosTokens.appBg,
        border: Border(
          bottom: BorderSide(color: NeoethosTokens.border),
        ),
      ),
      child: Row(
        children: [
          Text(
            title.toUpperCase(),
            style: const TextStyle(
              fontSize: NeoethosTokens.fsCaption,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.8,
              color: NeoethosTokens.textMuted,
            ),
          ),
          const SizedBox(width: 6),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
            decoration: BoxDecoration(
              color: NeoethosTokens.accentMuted,
              borderRadius: BorderRadius.circular(3),
            ),
            child: Text(
              '$count',
              style: const TextStyle(
                fontSize: NeoethosTokens.fsCaption - 1,
                fontWeight: FontWeight.w800,
                color: NeoethosTokens.accent,
              ),
            ),
          ),
          const Spacer(),
          if (trailing != null) trailing!,
        ],
      ),
    );
  }
}

class _LoadingLine extends StatelessWidget {
  const _LoadingLine();
  @override
  Widget build(BuildContext context) => const Center(
        child: Padding(
          padding: EdgeInsets.all(NeoethosTokens.spMd),
          child: Text(
            'Loading…',
            style: TextStyle(
              fontSize: NeoethosTokens.fsBody,
              color: NeoethosTokens.textMuted,
            ),
          ),
        ),
      );
}

class _EmptyLine extends StatelessWidget {
  final String message;
  const _EmptyLine({required this.message});
  @override
  Widget build(BuildContext context) => Center(
        child: Padding(
          padding: const EdgeInsets.all(NeoethosTokens.spMd),
          child: Text(
            message,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: NeoethosTokens.fsCaption,
              height: 1.4,
              color: NeoethosTokens.textFaint,
            ),
          ),
        ),
      );
}

class _ErrorBlock extends StatelessWidget {
  final Object error;
  const _ErrorBlock({required this.error});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.all(NeoethosTokens.spMd),
        child: Text(
          describeError(error),
          style: const TextStyle(
            fontSize: NeoethosTokens.fsCaption,
            color: NeoethosTokens.sell,
            height: 1.4,
          ),
        ),
      );
}

bool _isJpy(String symbol) {
  final upper = symbol.toUpperCase();
  return upper.endsWith('JPY') || upper.endsWith('JPY.SPB');
}

String _prettySymbol(String s) {
  if (s.length == 6) {
    return '${s.substring(0, 3)}/${s.substring(3)}';
  }
  return s;
}
