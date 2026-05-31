// Market Watch — unified trading hub (F-325 final).
//
// **F-321 (sidebar consolidation)** moved Markets + Chart + Execution +
// News under this single tab. **F-325 (the proper rebuild, 2026-05-29)**
// replaces the transitional TabBar wrapper with the Bloomberg / cTrader
// Pro layout the Codex mockup specified:
//
//   ┌──────────────────────────────────────────────────────────────┐
//   │  Watchlist · 24 symbols · 3 open · 2 pending · ⟳ 12s ago     │  ← summary strip
//   ├──────────────────────────────────────────────────────────────┤
//   │ Symbol Bid Ask Spread Δ% Trades Strategy Conf. Status Auto  │  ← dense table
//   │  EUR/USD 1.0823 1.0825 0.2 +0.34% 1 xgb_v44 58% Live  Off ▸  │
//   │  GBP/USD 1.2643 1.2646 0.3 −0.12% 0 — — Live  Off  ▸         │
//   │  …                                                            │
//   ├──────────────────────────────────────────────────────────────┤
//   │ Open Positions (3)                                            │
//   │ EUR/USD BUY 0.10 @ 1.0820 → +12.3 pips → +$12.30  · since 09:42│
//   │ …                                                              │
//   ├──────────────────────────────────────────────────────────────┤
//   │ Pending Orders (2)                                            │
//   │ EUR/USD SELL Limit 1.0850 expires 16:00                       │
//   │ …                                                              │
//   └──────────────────────────────────────────────────────────────┘
//
// Every number flows from the broker via the backend SSE / REST APIs.
// Columns that the broker hasn't surfaced yet (Volume, ATR%, Last
// Signal) render `—` instead of a fake value, per the operator's
// 2026-05-26 directive ("ola ta noumera apo ton server").

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../api/currency_format.dart';
import '../state/account_provider.dart';
import '../state/live_spots_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';

class MarketWatchScreen extends ConsumerWidget {
  const MarketWatchScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final spotsAsync = ref.watch(liveSpotsProvider);
    final accountAsync = ref.watch(accountSnapshotProvider);
    final intelAsync = ref.watch(intelligenceProvider);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _SummaryStrip(
          spotsAsync: spotsAsync,
          accountAsync: accountAsync,
        ),
        const SizedBox(height: ForexAiTokens.spSm),
        Expanded(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Left 2/3: the watchlist table (the main focus).
              Expanded(
                flex: 3,
                child: _WatchlistPanel(
                  spotsAsync: spotsAsync,
                  accountAsync: accountAsync,
                  intelAsync: intelAsync,
                ),
              ),
              const SizedBox(width: ForexAiTokens.spSm),
              // Right 1/3: stacked Open Positions + Pending Orders.
              Expanded(
                flex: 2,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Expanded(
                      flex: 3,
                      child: _OpenPositionsPanel(
                        accountAsync: accountAsync,
                        spotsAsync: spotsAsync,
                      ),
                    ),
                    const SizedBox(height: ForexAiTokens.spSm),
                    const Expanded(
                      flex: 2,
                      child: _PendingOrdersPanel(),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// Summary strip (top)
// ---------------------------------------------------------------------------

class _SummaryStrip extends StatelessWidget {
  final AsyncValue<LiveSpotsSnapshot> spotsAsync;
  final AsyncValue<AccountSnapshot> accountAsync;

  const _SummaryStrip({
    required this.spotsAsync,
    required this.accountAsync,
  });

  @override
  Widget build(BuildContext context) {
    final spots = spotsAsync.valueOrNull;
    final acc = accountAsync.valueOrNull;
    final symbolCount = spots?.symbolCount ?? 0;
    final visibleCount = spots?.spots.length ?? 0;
    final openCount = acc?.positions.length ?? 0;
    final updatedAt = spots?.snapshotAtUnixMs ?? 0;
    final ageSeconds = updatedAt == 0
        ? null
        : ((DateTime.now().millisecondsSinceEpoch - updatedAt) / 1000)
            .clamp(0, 1e6)
            .toInt();
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: ForexAiTokens.spMd,
        vertical: 8,
      ),
      decoration: BoxDecoration(
        color: ForexAiTokens.panelBg,
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Row(
        children: [
          const Text(
            'Watchlist',
            style: TextStyle(
              fontSize: ForexAiTokens.fsBody + 1,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.4,
              color: ForexAiTokens.textPrimary,
            ),
          ),
          const SizedBox(width: 12),
          _Pill(
            label: '$visibleCount of $symbolCount symbols',
            color: ForexAiTokens.accent,
          ),
          const SizedBox(width: 8),
          _Pill(
            label: '$openCount open',
            color: openCount > 0 ? ForexAiTokens.buy : ForexAiTokens.textFaint,
          ),
          const SizedBox(width: 8),
          const _Pill(
            // Pending orders endpoint hasn't shipped yet; mark explicitly
            // as 0 rather than hiding the pill — operator should know
            // the table is honestly empty, not just hidden.
            label: '0 pending',
            color: ForexAiTokens.textFaint,
          ),
          const Spacer(),
          if (ageSeconds != null)
            Text(
              'Updated ${ageSeconds}s ago',
              style: const TextStyle(
                fontSize: ForexAiTokens.fsCaption,
                color: ForexAiTokens.textMuted,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            )
          else
            const Text(
              'Streaming …',
              style: TextStyle(
                fontSize: ForexAiTokens.fsCaption,
                color: ForexAiTokens.textFaint,
                fontStyle: FontStyle.italic,
              ),
            ),
        ],
      ),
    );
  }
}

class _Pill extends StatelessWidget {
  final String label;
  final Color color;
  const _Pill({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.14),
        border: Border.all(color: color.withValues(alpha: 0.55)),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: ForexAiTokens.fsCaption,
          fontWeight: FontWeight.w700,
          color: color,
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Watchlist table (left, main focus)
// ---------------------------------------------------------------------------

class _WatchlistPanel extends StatelessWidget {
  final AsyncValue<LiveSpotsSnapshot> spotsAsync;
  final AsyncValue<AccountSnapshot> accountAsync;
  final AsyncValue<IntelligenceSnapshot> intelAsync;

  const _WatchlistPanel({
    required this.spotsAsync,
    required this.accountAsync,
    required this.intelAsync,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: ForexAiTokens.panelBg,
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const _WatchlistHeader(),
          Expanded(
            child: spotsAsync.when(
              loading: () => const Center(
                child: Padding(
                  padding: EdgeInsets.all(ForexAiTokens.spLg),
                  child: Text(
                    'Waiting for first SSE tick…',
                    style: TextStyle(
                      fontSize: ForexAiTokens.fsBody,
                      color: ForexAiTokens.textMuted,
                    ),
                  ),
                ),
              ),
              error: (err, _) => _ErrorBlock(message: err.toString()),
              data: (snap) {
                if (snap.spots.isEmpty) {
                  return const Center(
                    child: Padding(
                      padding: EdgeInsets.all(ForexAiTokens.spLg),
                      child: Text(
                        'No live spots yet — broker may still be warming '
                            'up or no symbols subscribed.',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: ForexAiTokens.fsBody,
                          color: ForexAiTokens.textMuted,
                        ),
                      ),
                    ),
                  );
                }
                // Build per-symbol strategy + position-count maps for the
                // join columns. Done once per build instead of inside the
                // ListView.builder so each row is O(1).
                final positions = accountAsync.valueOrNull?.positions ?? const <Position>[];
                final positionsBySymbol = <String, int>{};
                for (final p in positions) {
                  positionsBySymbol[p.symbol] =
                      (positionsBySymbol[p.symbol] ?? 0) + 1;
                }
                final intel = intelAsync.valueOrNull;
                final strategyBySymbol = <String, DiscoveryTarget>{};
                for (final t in intel?.discoveryTargets ?? const <DiscoveryTarget>[]) {
                  strategyBySymbol[t.symbol] = t;
                }
                final acc = intel?.walkforwardAvgAccuracy;

                final sorted = [...snap.spots]
                  ..sort((a, b) => a.symbolName.compareTo(b.symbolName));

                return Scrollbar(
                  child: ListView.builder(
                    itemCount: sorted.length,
                    itemBuilder: (context, i) => _SymbolRow(
                      spot: sorted[i],
                      positionsOnSymbol:
                          positionsBySymbol[sorted[i].symbolName] ?? 0,
                      strategy: strategyBySymbol[sorted[i].symbolName],
                      ensembleAcc: acc,
                      stripe: i.isOdd,
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
}

class _WatchlistHeader extends StatelessWidget {
  const _WatchlistHeader();

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        color: ForexAiTokens.appBg,
        border: Border(
          bottom: BorderSide(color: ForexAiTokens.border),
        ),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      child: const Row(
        children: [
          _HeaderCell(text: 'Symbol', width: 96, align: Alignment.centerLeft),
          _HeaderCell(text: 'Bid', width: 72, align: Alignment.centerRight),
          _HeaderCell(text: 'Ask', width: 72, align: Alignment.centerRight),
          _HeaderCell(text: 'Spread', width: 52, align: Alignment.centerRight),
          _HeaderCell(text: 'Trades', width: 48, align: Alignment.centerRight),
          Expanded(
            child: _HeaderCell(
              text: 'Strategy',
              width: null,
              align: Alignment.centerLeft,
            ),
          ),
          _HeaderCell(text: 'Conf.', width: 48, align: Alignment.centerRight),
          _HeaderCell(text: 'Status', width: 60, align: Alignment.center),
        ],
      ),
    );
  }
}

class _HeaderCell extends StatelessWidget {
  final String text;
  final double? width;
  final Alignment align;
  const _HeaderCell({
    required this.text,
    required this.width,
    required this.align,
  });
  @override
  Widget build(BuildContext context) {
    final child = Text(
      text.toUpperCase(),
      textAlign: align == Alignment.centerRight
          ? TextAlign.right
          : align == Alignment.center
              ? TextAlign.center
              : TextAlign.left,
      style: const TextStyle(
        fontSize: ForexAiTokens.fsCaption - 1,
        fontWeight: FontWeight.w800,
        letterSpacing: 0.8,
        color: ForexAiTokens.textFaint,
      ),
    );
    if (width == null) return Align(alignment: align, child: child);
    return SizedBox(width: width, child: Align(alignment: align, child: child));
  }
}

class _SymbolRow extends StatelessWidget {
  final LiveSpotTick spot;
  final int positionsOnSymbol;
  final DiscoveryTarget? strategy;
  final double? ensembleAcc;
  final bool stripe;
  const _SymbolRow({
    required this.spot,
    required this.positionsOnSymbol,
    required this.strategy,
    required this.ensembleAcc,
    required this.stripe,
  });

  @override
  Widget build(BuildContext context) {
    final bid = spot.bid;
    final ask = spot.ask;
    final spread = (bid != null && ask != null) ? (ask - bid) : null;
    final spreadPips = spread == null
        ? null
        : _isJpy(spot.symbolName) ? spread * 100 : spread * 10000;

    final stale = spot.freshnessSeconds > 5;
    final (statusLabel, statusColor) = stale
        ? ('Stale', ForexAiTokens.warning)
        : (bid != null && ask != null)
            ? ('Live', ForexAiTokens.buy)
            : ('Off', ForexAiTokens.textFaint);

    return Container(
      decoration: BoxDecoration(
        color: stripe
            ? ForexAiTokens.appBg.withValues(alpha: 0.4)
            : Colors.transparent,
        border: const Border(
          bottom: BorderSide(
            color: ForexAiTokens.border,
            width: 0.4,
          ),
        ),
      ),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 96,
            child: Text(
              _prettySymbol(spot.symbolName),
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                fontSize: ForexAiTokens.fsBody,
                fontWeight: FontWeight.w700,
                color: ForexAiTokens.textPrimary,
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ),
          _NumberCell(
            value: bid,
            width: 72,
            digits: _isJpy(spot.symbolName) ? 3 : 5,
          ),
          _NumberCell(
            value: ask,
            width: 72,
            digits: _isJpy(spot.symbolName) ? 3 : 5,
          ),
          _NumberCell(
            value: spreadPips,
            width: 52,
            digits: 1,
            faded: true,
          ),
          SizedBox(
            width: 48,
            child: Text(
              positionsOnSymbol == 0 ? '—' : '$positionsOnSymbol',
              textAlign: TextAlign.right,
              style: TextStyle(
                fontSize: ForexAiTokens.fsBody,
                fontWeight: FontWeight.w700,
                color: positionsOnSymbol == 0
                    ? ForexAiTokens.textFaint
                    : ForexAiTokens.accent,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: strategy == null
                  ? const Text(
                      '—',
                      style: TextStyle(
                        fontSize: ForexAiTokens.fsBody,
                        color: ForexAiTokens.textFaint,
                      ),
                    )
                  : Text(
                      '${strategy!.strategyId} · ${strategy!.baseTf}',
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: ForexAiTokens.fsBody,
                        color: ForexAiTokens.textPrimary,
                      ),
                    ),
            ),
          ),
          SizedBox(
            width: 48,
            child: Text(
              ensembleAcc == null
                  ? '—'
                  : '${(ensembleAcc! * 100).toStringAsFixed(0)}%',
              textAlign: TextAlign.right,
              style: TextStyle(
                fontSize: ForexAiTokens.fsBody,
                fontWeight: FontWeight.w700,
                color: ensembleAcc == null
                    ? ForexAiTokens.textFaint
                    : ensembleAcc! >= 0.55
                        ? ForexAiTokens.buy
                        : ensembleAcc! >= 0.50
                            ? ForexAiTokens.warning
                            : ForexAiTokens.sell,
                fontFeatures: const [FontFeature.tabularFigures()],
              ),
            ),
          ),
          SizedBox(
            width: 60,
            child: Align(
              alignment: Alignment.center,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                decoration: BoxDecoration(
                  color: statusColor.withValues(alpha: 0.14),
                  border: Border.all(color: statusColor.withValues(alpha: 0.55)),
                  borderRadius: BorderRadius.circular(3),
                ),
                child: Text(
                  statusLabel,
                  style: TextStyle(
                    fontSize: ForexAiTokens.fsCaption - 1,
                    fontWeight: FontWeight.w800,
                    color: statusColor,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _NumberCell extends StatelessWidget {
  final double? value;
  final double width;
  final int digits;
  final bool faded;
  const _NumberCell({
    required this.value,
    required this.width,
    required this.digits,
    this.faded = false,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: width,
      child: Text(
        value == null ? '—' : value!.toStringAsFixed(digits),
        textAlign: TextAlign.right,
        style: TextStyle(
          fontSize: ForexAiTokens.fsBody,
          fontWeight: faded ? FontWeight.w500 : FontWeight.w700,
          color: value == null
              ? ForexAiTokens.textFaint
              : faded
                  ? ForexAiTokens.textMuted
                  : ForexAiTokens.textPrimary,
          fontFeatures: const [FontFeature.tabularFigures()],
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Open Positions panel (right top)
// ---------------------------------------------------------------------------

class _OpenPositionsPanel extends StatelessWidget {
  final AsyncValue<AccountSnapshot> accountAsync;
  final AsyncValue<LiveSpotsSnapshot> spotsAsync;
  const _OpenPositionsPanel({
    required this.accountAsync,
    required this.spotsAsync,
  });

  @override
  Widget build(BuildContext context) {
    final acc = accountAsync.valueOrNull;
    final positions = acc?.positions ?? const <Position>[];
    final currencySymbol = currencyGlyph(acc?.currency ?? 'USD');
    return Container(
      decoration: BoxDecoration(
        color: ForexAiTokens.panelBg,
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _PanelTitle(
            title: 'Open Positions',
            count: positions.length,
          ),
          Expanded(
            child: accountAsync.when(
              loading: () => const _LoadingLine(),
              error: (err, _) => _ErrorBlock(message: err.toString()),
              data: (_) {
                if (positions.isEmpty) {
                  return const _EmptyLine(message: 'No open positions.');
                }
                return Scrollbar(
                  child: ListView.builder(
                    itemCount: positions.length,
                    itemBuilder: (context, i) => _PositionRow(
                      position: positions[i],
                      currencySymbol: currencySymbol,
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
}

class _PositionRow extends StatelessWidget {
  final Position position;
  final String currencySymbol;
  const _PositionRow({
    required this.position,
    required this.currencySymbol,
  });

  @override
  Widget build(BuildContext context) {
    final isBuy = position.side.toLowerCase() == 'buy';
    final pnlColor = position.pnlUsd > 0
        ? ForexAiTokens.buy
        : position.pnlUsd < 0
            ? ForexAiTokens.sell
            : ForexAiTokens.textMuted;
    final since = position.openTimestampMs == null
        ? '—'
        : DateFormat.Hm().format(
            DateTime.fromMillisecondsSinceEpoch(
              position.openTimestampMs!,
            ),
          );
    final volume = position.volume == 0 ? '—' : position.volume.toStringAsFixed(2);
    final pnlPips = position.pnlPips == 0
        ? '0.0'
        : position.pnlPips.toStringAsFixed(1);
    final pnlUsd = position.pnlUsd == 0
        ? '0.00'
        : position.pnlUsd.toStringAsFixed(2);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: const BoxDecoration(
        border: Border(
          bottom: BorderSide(color: ForexAiTokens.border, width: 0.4),
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding:
                const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(
              color: (isBuy ? ForexAiTokens.buy : ForexAiTokens.sell)
                  .withValues(alpha: 0.18),
              border: Border.all(
                color: (isBuy ? ForexAiTokens.buy : ForexAiTokens.sell)
                    .withValues(alpha: 0.6),
              ),
              borderRadius: BorderRadius.circular(3),
            ),
            child: Text(
              isBuy ? 'BUY' : 'SELL',
              style: TextStyle(
                fontSize: ForexAiTokens.fsCaption - 1,
                fontWeight: FontWeight.w800,
                color: isBuy ? ForexAiTokens.buy : ForexAiTokens.sell,
              ),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${_prettySymbol(position.symbol)} · $volume',
                  style: const TextStyle(
                    fontSize: ForexAiTokens.fsBody,
                    fontWeight: FontWeight.w700,
                    color: ForexAiTokens.textPrimary,
                    fontFeatures: [FontFeature.tabularFigures()],
                  ),
                ),
                Text(
                  'since $since · #${position.positionId}',
                  style: const TextStyle(
                    fontSize: ForexAiTokens.fsCaption,
                    color: ForexAiTokens.textFaint,
                    fontFeatures: [FontFeature.tabularFigures()],
                  ),
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                '$pnlPips pips',
                style: TextStyle(
                  fontSize: ForexAiTokens.fsBody,
                  fontWeight: FontWeight.w800,
                  color: pnlColor,
                  fontFeatures: const [FontFeature.tabularFigures()],
                ),
              ),
              Text(
                '$currencySymbol$pnlUsd',
                style: TextStyle(
                  fontSize: ForexAiTokens.fsCaption,
                  color: pnlColor.withValues(alpha: 0.8),
                  fontFeatures: const [FontFeature.tabularFigures()],
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Pending Orders panel (right bottom)
// ---------------------------------------------------------------------------

class _PendingOrdersPanel extends StatelessWidget {
  const _PendingOrdersPanel();

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: ForexAiTokens.panelBg,
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _PanelTitle(title: 'Pending Orders', count: 0),
          Expanded(
            child: _EmptyLine(
              message:
                  'No pending orders.\n'
                  'Limit/Stop entries will surface here once the '
                  '/orders/pending endpoint ships (deferred to a '
                  'broker-side change).',
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
  const _PanelTitle({required this.title, required this.count});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: const BoxDecoration(
        color: ForexAiTokens.appBg,
        border: Border(
          bottom: BorderSide(color: ForexAiTokens.border),
        ),
      ),
      child: Row(
        children: [
          Text(
            title.toUpperCase(),
            style: const TextStyle(
              fontSize: ForexAiTokens.fsCaption,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.8,
              color: ForexAiTokens.textMuted,
            ),
          ),
          const SizedBox(width: 6),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
            decoration: BoxDecoration(
              color: ForexAiTokens.accentMuted,
              borderRadius: BorderRadius.circular(3),
            ),
            child: Text(
              '$count',
              style: const TextStyle(
                fontSize: ForexAiTokens.fsCaption - 1,
                fontWeight: FontWeight.w800,
                color: ForexAiTokens.accent,
              ),
            ),
          ),
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
          padding: EdgeInsets.all(ForexAiTokens.spMd),
          child: Text(
            'Loading…',
            style: TextStyle(
              fontSize: ForexAiTokens.fsBody,
              color: ForexAiTokens.textMuted,
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
          padding: const EdgeInsets.all(ForexAiTokens.spMd),
          child: Text(
            message,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: ForexAiTokens.fsCaption,
              height: 1.4,
              color: ForexAiTokens.textFaint,
            ),
          ),
        ),
      );
}

class _ErrorBlock extends StatelessWidget {
  final String message;
  const _ErrorBlock({required this.message});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.all(ForexAiTokens.spMd),
        child: Text(
          'Error: $message',
          style: const TextStyle(
            fontSize: ForexAiTokens.fsCaption,
            color: ForexAiTokens.sell,
            height: 1.4,
          ),
        ),
      );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Heuristic: JPY pairs use 3 decimals on bid/ask + the pip is at
/// the 2nd decimal (multiply by 100), everything else 5 decimals
/// with pip at the 4th (multiply by 10000). Matches the cTrader
/// convention used elsewhere in the codebase (#222).
bool _isJpy(String symbol) {
  final upper = symbol.toUpperCase();
  return upper.endsWith('JPY') || upper.endsWith('JPY.SPB');
}

/// Visual tweak: turn "EURUSD" into "EUR/USD" for the watchlist and
/// the position rows. Keeps the underlying symbol IDs unchanged.
String _prettySymbol(String s) {
  if (s.length == 6) {
    return '${s.substring(0, 3)}/${s.substring(3)}';
  }
  return s;
}
