// Chart screen — symbol + timeframe chips, candlestick canvas painted
// from `/chart` OHLC data, plus optional indicator overlays from
// `/indicators` (server-side vector_ta). Read-only (the local data
// dir is the source); switching chips refetches via Riverpod.
//
// Symbol + timeframe lists come EXCLUSIVELY from the broker. No
// hardcoded fallbacks — when the broker isn't reachable, the screen
// shows an explicit "connect broker first" message instead of
// faking choices the broker may or may not actually offer.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'package:dio/dio.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../l10n/app_localizations.dart';
import '../widgets/backend_error_widget.dart';
import '../state/account_provider.dart';
import '../state/live_spots_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/inline_buy_sell.dart';
import '../widgets/pro_chart.dart';
import '../widgets/symbol_picker.dart';
import '_placeholder.dart';

/// Top-10 indicators surfaced as toggle chips above the chart.
/// Mirrors the server's ALLOWED_INDICATORS list. Order = display order.
const _indicatorChips = <String>[
  'sma',
  'ema',
  'rsi',
  'macd',
  'bollinger_bands',
  'atr',
  'stoch',
  'adx',
  'vwap',
];

/// F-361: the right-click multi-select menu groups the same 9 indicators
/// into "Price overlays" (drawn on/over the candles) and "Oscillators"
/// (sub-panels / strips). These are display groupings only — every id
/// still lives in the one `activeIndicators` set the chip row uses, so
/// the two controls stay in lock-step.
const _priceOverlayIndicators = <String>[
  'sma',
  'ema',
  'bollinger_bands',
  'vwap',
];
const _oscillatorIndicators = <String>[
  'rsi',
  'macd',
  'stoch',
  'atr',
  'adx',
];

String _indicatorLabel(String id) {
  switch (id) {
    case 'sma':
      return 'SMA';
    case 'ema':
      return 'EMA';
    case 'rsi':
      return 'RSI';
    case 'macd':
      return 'MACD';
    case 'bollinger_bands':
      return 'BBands';
    case 'atr':
      return 'ATR';
    case 'stoch':
      return 'Stoch';
    case 'adx':
      return 'ADX';
    case 'vwap':
      return 'VWAP';
    default:
      return id.toUpperCase();
  }
}

/// Slot descriptor — bundles the providers a chart panel reads so the
/// same `_ChartPanel` widget can serve both panel A and panel B. Two
/// instances live as `_slotA` / `_slotB` constants; the screen picks
/// which to render based on `multiChartEnabledProvider`.
class _ChartSlot {
  final String label;
  final StateProvider<String> symbol;
  final StateProvider<String> timeframe;
  final AutoDisposeFutureProvider<ChartSnapshot> chart;
  final StateProvider<Set<String>> activeIndicators;
  final AutoDisposeFutureProviderFamily<IndicatorSnapshot, String> indicator;
  const _ChartSlot({
    required this.label,
    required this.symbol,
    required this.timeframe,
    required this.chart,
    required this.activeIndicators,
    required this.indicator,
  });
}

final _slotA = _ChartSlot(
  label: 'A',
  symbol: chartSymbolProvider,
  timeframe: chartTimeframeProvider,
  chart: chartProvider,
  activeIndicators: activeIndicatorsProvider,
  indicator: indicatorProvider,
);

final _slotB = _ChartSlot(
  label: 'B',
  symbol: chartSymbolProviderB,
  timeframe: chartTimeframeProviderB,
  chart: chartProviderB,
  activeIndicators: activeIndicatorsProviderB,
  indicator: indicatorProviderB,
);

class ChartScreen extends ConsumerWidget {
  const ChartScreen({super.key});

  /// #198: open a TradingView-Copilot-style modal sheet seeded with
  /// the active chart's symbol + timeframe so the user can ask the
  /// AI a contextual question without leaving the screen. The full
  /// AI Helper screen still exists (for long-form chat); this is the
  /// "quick question" entry point that lives next to what the user
  /// is looking at.
  void _openContextualAi(
    BuildContext context,
    WidgetRef ref,
    String symbol,
    String timeframe,
  ) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.3,
        maxChildSize: 0.95,
        expand: false,
        builder: (sheetCtx, scrollCtrl) => _ContextualAiSheet(
          symbol: symbol,
          timeframe: timeframe,
          scrollController: scrollCtrl,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final multi = ref.watch(multiChartEnabledProvider);
    final activeSymbol = ref.watch(_slotA.symbol);
    final activeTimeframe = ref.watch(_slotA.timeframe);

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: ViewHeader(
                  title: multi ? l10n.chartTitleMulti : l10n.chartTitle,
                  subtitle: multi
                      ? l10n.chartSubtitleMulti
                      : l10n.chartSubtitleSingle,
                ),
              ),
              const SizedBox(width: 12),
              FloatingActionButton.extended(
                // #198: TradingView Copilot is a corner button on the chart
                // page that pops a context-aware chat. We keep it inside
                // the shell content so ChartScreen is not a nested Scaffold.
                onPressed: () => _openContextualAi(
                  context,
                  ref,
                  activeSymbol,
                  activeTimeframe,
                ),
                icon: const Icon(Icons.psychology_alt_outlined),
                label: Text(l10n.chartAskAi),
              ),
            ],
          ),
          // Compare-mode toggle. Tapping it flips panel B on/off.
          // Limit: TWO panels only. There is no UI affordance to add
          // a third — multi-chart grids fragment attention and bury
          // the trade thesis. The "vs 16" in the task title is the
          // anti-goal we're avoiding.
          SectionCard(
            title: l10n.chartLayout,
            child: Row(
              children: [
                Switch(
                  value: multi,
                  onChanged: (v) =>
                      ref.read(multiChartEnabledProvider.notifier).state = v,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    multi
                        ? l10n.chartLayoutComparisonOn
                        : l10n.chartLayoutSingle,
                    style: const TextStyle(
                      fontSize: 11,
                      color: NeoethosTokens.textMuted,
                    ),
                  ),
                ),
              ],
            ),
          ),
          if (multi)
            // Side-by-side when the window is wide enough; stack when
            // it isn't (e.g. window pinned to half-screen). 720px is
            // the empirical break where each panel becomes too narrow
            // to read price tick labels.
            LayoutBuilder(
              builder: (context, constraints) {
                final wide = constraints.maxWidth >= 720;
                if (wide) {
                  return Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Expanded(child: _ChartPanel(slot: _slotA)),
                      const SizedBox(width: 12),
                      Expanded(child: _ChartPanel(slot: _slotB)),
                    ],
                  );
                }
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _ChartPanel(slot: _slotA),
                    const SizedBox(height: 12),
                    _ChartPanel(slot: _slotB),
                  ],
                );
              },
            )
          else
            _ChartPanel(slot: _slotA),
        ],
      ),
    );
  }
}

/// #198: contextual AI bottom sheet — wraps a Codex chat with a
/// fixed system prefix that names the active symbol/timeframe so the
/// model can answer questions like "what's driving EURUSD today?"
/// without the user having to type the symbol every time.
class _ContextualAiSheet extends ConsumerStatefulWidget {
  final String symbol;
  final String timeframe;
  final ScrollController scrollController;
  const _ContextualAiSheet({
    required this.symbol,
    required this.timeframe,
    required this.scrollController,
  });

  @override
  ConsumerState<_ContextualAiSheet> createState() => _ContextualAiSheetState();
}

class _ContextualAiSheetState extends ConsumerState<_ContextualAiSheet> {
  final _input = TextEditingController();
  final List<({bool user, String text})> _msgs = [];
  bool _busy = false;

  @override
  void dispose() {
    _input.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final q = _input.text.trim();
    if (q.isEmpty || _busy) return;
    // Capture the localized error prefix before the await — the
    // BuildContext may be gone by the time the future resolves.
    final l10n = AppLocalizations.of(context)!;
    // Inject the chart context as a soft prefix the model can use
    // without echoing back. Keep terse.
    final prompt =
        'Active chart: ${widget.symbol} ${widget.timeframe}. User asks: $q';
    setState(() {
      _msgs.add((user: true, text: q));
      _busy = true;
      _input.clear();
    });
    try {
      final r = await ref.read(backendClientProvider).codexChat(
            prompt: prompt,
            maxTokens: 512,
          );
      if (!mounted) return;
      setState(() => _msgs.add((user: false, text: r.response.trim())));
    } on DioException catch (e) {
      if (!mounted) return;
      setState(() => _msgs.add(
          (user: false, text: '${l10n.commonError}: ${describeError(e)}')));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: const BoxDecoration(
        color: NeoethosTokens.panelBg,
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(16),
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              const Icon(Icons.psychology_alt_outlined,
                  color: NeoethosTokens.textPrimary),
              const SizedBox(width: 8),
              Text(
                l10n.chartAskAiContext(widget.symbol, widget.timeframe),
                style: const TextStyle(
                  fontWeight: FontWeight.w700,
                  fontSize: 14,
                ),
              ),
              const Spacer(),
              IconButton(
                onPressed: () => Navigator.of(context).pop(),
                icon: const Icon(Icons.close),
              ),
            ],
          ),
          const Divider(),
          Expanded(
            child: ListView.builder(
              controller: widget.scrollController,
              itemCount: _msgs.length,
              itemBuilder: (ctx, i) {
                final m = _msgs[i];
                return Container(
                  alignment:
                      m.user ? Alignment.centerRight : Alignment.centerLeft,
                  margin: const EdgeInsets.symmetric(vertical: 4),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: m.user
                          ? NeoethosTokens.buy.withValues(alpha: 0.15)
                          : NeoethosTokens.surfaceBg,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      m.text,
                      style: const TextStyle(fontSize: 12),
                    ),
                  ),
                );
              },
            ),
          ),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _input,
                  enabled: !_busy,
                  decoration: InputDecoration(
                    hintText: l10n.chartAskHint,
                    isDense: true,
                    border: const OutlineInputBorder(),
                  ),
                  onSubmitted: (_) => _send(),
                ),
              ),
              const SizedBox(width: 8),
              FilledButton.icon(
                onPressed: _busy ? null : _send,
                icon: _busy
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.send, size: 16),
                label: Text(l10n.chartSend),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// One chart panel — symbol picker + timeframe picker + indicator
/// chips + the candlestick canvas. Parameterised over a `_ChartSlot`
/// so the same widget renders panel A or panel B.
class _ChartPanel extends ConsumerWidget {
  final _ChartSlot slot;
  const _ChartPanel({required this.slot});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final symbol = ref.watch(slot.symbol);
    final timeframe = ref.watch(slot.timeframe);
    final async = ref.watch(slot.chart);
    final brokerTimeframes = ref.watch(brokerTimeframesProvider);

    final timeframeChoices = brokerTimeframes.maybeWhen(
      data: (list) => list,
      orElse: () => const <String>[],
    );

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          // #183: previously rendered as a wall of FilterChips (104+
          // entries) which scrolled the screen down and made the user
          // hunt visually for "EURUSD" among "XAUUSD, GBPJPY, …". The
          // SymbolPicker widget is type-ahead — operator types "eu" and
          // gets EURUSD / EURJPY / EURGBP / EURCAD as suggestions. The
          // catalog count is appended to the label so users still know
          // how many symbols the broker exposed.
          title: l10n.chartPanelSymbol(slot.label),
          child: SymbolPicker(
            value: symbol,
            label: l10n.thSymbol,
            onChanged: (s) => ref.read(slot.symbol.notifier).state = s,
          ),
        ),
        SectionCard(
          title: l10n.chartPanelTimeframe(slot.label) +
              (brokerTimeframes.hasValue
                  ? l10n.chartTimeframeCanonical(timeframeChoices.length)
                  : ''),
          child: timeframeChoices.isEmpty
              ? Text(
                  brokerTimeframes.hasError
                      ? l10n.chartTimeframeUnavailable(
                          brokerTimeframes.error.toString())
                      : l10n.chartTimeframeLoading,
                  style: const TextStyle(
                    fontSize: 11,
                    color: NeoethosTokens.warning,
                  ),
                )
              : Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: [
                    for (final t in timeframeChoices)
                      _Chip(
                        label: t,
                        selected: t == timeframe,
                        onTap: () =>
                            ref.read(slot.timeframe.notifier).state = t,
                      ),
                  ],
                ),
        ),
        // F-361: right-click anywhere on the Indicators card opens a
        // multi-select menu (price overlays + oscillators, each with a
        // checkbox) anchored at the cursor. It's an additional, richer
        // add path that shares the SAME `slot.activeIndicators` provider
        // as the chip row below — both stay in sync. We put the
        // onSecondaryTapDown handler HERE on the panel header card, NOT
        // on the KChartWidget, so k_chart_plus keeps full ownership of
        // its pan/zoom/long-press gestures (scroll-back pagination,
        // crosshair) untouched. HitTestBehavior.opaque so the gap around
        // the chips also catches the right-click.
        GestureDetector(
          behavior: HitTestBehavior.opaque,
          onSecondaryTapDown: (details) => _showIndicatorMenu(
            context,
            slot,
            details.globalPosition,
          ),
          child: SectionCard(
            title: l10n.chartPanelIndicators(slot.label),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: [
                    for (final ind in _indicatorChips)
                      Consumer(
                        builder: (ctx, indRef, _) {
                          final active = indRef
                              .watch(slot.activeIndicators)
                              .contains(ind);
                          // F-360: the chip row is the at-a-glance active
                          // view + quick toggle. Every chip actually draws
                          // on the live k_chart_plus chart below. Two
                          // render paths:
                          //   • client-side (k_chart_plus computes it over
                          //     the candles, pan/zoom-aware) — SMA/EMA/
                          //     BBands/RSI/MACD/Stoch.
                          //   • server-fed (ATR/ADX/VWAP have no
                          //     k_chart_plus equivalent) — pulled from
                          //     /indicators and drawn in a strip under the
                          //     k-line.
                          // We mark the server-fed ones with a leading "• "
                          // and a tooltip so the operator knows why those
                          // three look slightly different (separate strip
                          // vs on the candles), but BOTH paths render for
                          // real.
                          final serverFed =
                              kChartServerIndicators.contains(ind);
                          final chip = _Chip(
                            label: serverFed
                                ? '• ${_indicatorLabel(ind)}'
                                : _indicatorLabel(ind),
                            selected: active,
                            onTap: () {
                              final notifier =
                                  indRef.read(slot.activeIndicators.notifier);
                              final next = {...notifier.state};
                              if (next.contains(ind)) {
                                next.remove(ind);
                              } else {
                                next.add(ind);
                              }
                              notifier.state = next;
                            },
                          );
                          return serverFed
                              ? Tooltip(
                                  message: l10n.chartServerFedTooltip(
                                      _indicatorLabel(ind)),
                                  child: chip,
                                )
                              : chip;
                        },
                      ),
                  ],
                ),
                const SizedBox(height: 6),
                // Discoverability hint for the right-click add path.
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.ads_click,
                      size: 11,
                      color: NeoethosTokens.textMuted,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      l10n.chartRightClickHint,
                      style: const TextStyle(
                        fontSize: 10,
                        fontStyle: FontStyle.italic,
                        color: NeoethosTokens.textMuted,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
        async.when(
          data: (c) => _ChartBody(snapshot: c, slot: slot),
          loading: () => const _Loading(),
          error: (err, _) =>
              BackendErrorWidget(error: err, title: l10n.chartDataUnavailable),
        ),
      ],
    );
  }

  /// F-361: open the multi-select indicator menu anchored at the cursor.
  ///
  /// Implemented as a translucent-barrier dialog (NOT a `PopupMenuItem`
  /// list) precisely so it STAYS OPEN while the operator ticks several
  /// boxes — a `showMenu`/`PopupMenuButton` dismisses on every item tap,
  /// which is the opposite of what the operator asked for. Each checkbox
  /// writes straight through to `slot.activeIndicators` so the chart +
  /// the chip row update live on every tick; the menu closes only on the
  /// "Done" button or a tap on the barrier outside it.
  void _showIndicatorMenu(
    BuildContext context,
    _ChartSlot slot,
    Offset globalPosition,
  ) {
    final l10n = AppLocalizations.of(context)!;
    final overlaySize = MediaQuery.of(context).size;
    showDialog<void>(
      context: context,
      // Translucent (not transparent) so it reads as a modal layer but
      // the chart stays visible behind it.
      barrierColor: Colors.black.withValues(alpha: 0.15),
      barrierLabel: l10n.chartIndicatorMenuLabel,
      builder: (dialogCtx) {
        // Clamp the anchor so the ~300px-wide / up-to-440px-tall card
        // never spills off-screen when the operator right-clicks near an
        // edge.
        const menuW = 300.0;
        const menuH = 440.0;
        final left =
            globalPosition.dx.clamp(8.0, (overlaySize.width - menuW - 8.0).clamp(8.0, double.infinity));
        final top =
            globalPosition.dy.clamp(8.0, (overlaySize.height - menuH - 8.0).clamp(8.0, double.infinity));
        return Stack(
          children: [
            Positioned(
              left: left,
              top: top,
              width: menuW,
              child: _IndicatorMultiSelectMenu(slot: slot),
            ),
          ],
        );
      },
    );
  }
}

class _ChartBody extends ConsumerWidget {
  final ChartSnapshot snapshot;
  final _ChartSlot slot;
  const _ChartBody({required this.snapshot, required this.slot});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final changePos = snapshot.priceChangePct >= 0;
    // Live tick overlay (#137): if the streamer has a fresh tick
    // for this symbol, prefer it over the historical-bar latestClose
    // — the bar's close is up to 1 minute stale on an M1 chart,
    // whereas the tick is sub-2 s old. Falls back to historical
    // when no tick is available (e.g. the streamer didn't spawn,
    // or this symbol isn't in the default-streamed-symbols list).
    //
    // **2026-05-26 fix (Κωνσταντίνος)**: cTrader spot events arrive
    // separately for bid and ask. The cache can have bid=null when
    // only an ask tick has arrived yet (or vice versa), so the prior
    // `midPrice ?? bid` chain returned null and the chart silently
    // fell back to the stale-by-minutes bar close — user reported
    // "Chart not Live". Extend fallback through ask so any of the
    // three populated values keeps the display fresh.
    final liveSnap = ref.watch(liveSpotsProvider).valueOrNull;
    final tick = liveSnap?.lookup(snapshot.symbol);
    final livePrice = (tick != null && !tick.isStale)
        ? (tick.midPrice ?? tick.bid ?? tick.ask)
        : null;
    final displayedPrice = livePrice ?? snapshot.latestClose;
    // #182: when the backend returns an empty-data placeholder we get
    // `latestClose == 0.0` and `priceChangePct == 0.0`. Rendering that
    // as "0.00000  +0.00%" looks like a real instrument at zero —
    // operators reasonably ask "is EURUSD really priced at 0.00000?".
    // Treat the no-data case explicitly and show an em-dash so it's
    // unambiguous that no candles are loaded for this pair yet.
    final hasData =
        snapshot.candleCount > 0 && (displayedPrice > 0.0 || livePrice != null);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: l10n.chartPanelHeader(
              slot.label, snapshot.symbol, snapshot.timeframe),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    hasData ? displayedPrice.toStringAsFixed(5) : '—',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.w800,
                      color: livePrice != null
                          ? NeoethosTokens.buy
                          : NeoethosTokens.textPrimary,
                    ),
                  ),
                  if (livePrice != null) ...[
                    const SizedBox(width: 6),
                    Tooltip(
                      message: l10n.chartLiveTick,
                      child: const Icon(
                        Icons.bolt,
                        size: 16,
                        color: NeoethosTokens.buy,
                      ),
                    ),
                  ],
                  const SizedBox(width: 12),
                  Text(
                    hasData
                        ? '${changePos ? '+' : ''}${snapshot.priceChangePct.toStringAsFixed(2)} %'
                        : '—',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: hasData
                          ? (changePos ? NeoethosTokens.buy : NeoethosTokens.sell)
                          : NeoethosTokens.textMuted,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    hasData
                        ? l10n.chartRange(
                            snapshot.priceMin.toStringAsFixed(5),
                            snapshot.priceMax.toStringAsFixed(5))
                        : l10n.chartNoDataBootstrap,
                    style: const TextStyle(
                      fontSize: 11,
                      color: NeoethosTokens.textMuted,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                snapshot.headline,
                style: const TextStyle(
                  fontSize: 11,
                  color: NeoethosTokens.textMuted,
                ),
              ),
              if (!snapshot.isBrokerSource) ...[
                const SizedBox(height: 8),
                _ChartSourceBanner(snapshot: snapshot),
              ],
              const SizedBox(height: 12),
              if (snapshot.candles.isEmpty)
                _AutoFetchPrompt(
                  symbol: snapshot.symbol,
                  timeframe: snapshot.timeframe,
                  slot: slot,
                )
              else ...[
                // F-334: one-click buy/sell strip, right-aligned just
                // above the chart.
                Align(
                  alignment: Alignment.centerRight,
                  child: Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: InlineBuySell(symbol: snapshot.symbol),
                  ),
                ),
                // F-336/F-360: professional k-line chart (k_chart_plus).
                // The chip row above (slot.activeIndicators) is the ONE
                // unified indicator control — ProChart rebuilds its
                // k_chart_plus overlay/sub-panel lists from that set and
                // pulls ATR/ADX/VWAP from `slot.indicator` (/indicators)
                // for the strip below the k-line. Pan/zoom + live forming
                // candle + scroll-back all preserved.
                ProChart(
                  snapshot: snapshot,
                  activeIndicators: slot.activeIndicators,
                  indicatorFamily: slot.indicator,
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }
}

class _ChartSourceBanner extends StatelessWidget {
  final ChartSnapshot snapshot;
  const _ChartSourceBanner({required this.snapshot});

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final text = snapshot.isDiskCache
        ? l10n.chartCachedCandles
        : l10n.chartDataSource(snapshot.source);
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: NeoethosTokens.warning.withValues(alpha: 0.08),
        border:
            Border.all(color: NeoethosTokens.warning.withValues(alpha: 0.35)),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Text(
        text,
        style: const TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          color: NeoethosTokens.warning,
        ),
      ),
    );
  }
}

/// #190: "no candles loaded" prompt with a one-click broker fetch.
/// Replaces the silent placeholder text. The button asks the backend
/// for the last 30 days of bars on the active symbol/timeframe — the
/// same call Data Bootstrap makes, but pre-filled with what the chart
/// is already trying to show. After the fetch completes the chart
/// provider invalidates and re-loads.
class _AutoFetchPrompt extends ConsumerStatefulWidget {
  final String symbol;
  final String timeframe;
  final _ChartSlot slot;
  const _AutoFetchPrompt({
    required this.symbol,
    required this.timeframe,
    required this.slot,
  });

  @override
  ConsumerState<_AutoFetchPrompt> createState() => _AutoFetchPromptState();
}

class _AutoFetchPromptState extends ConsumerState<_AutoFetchPrompt> {
  bool _busy = false;
  String? _error;
  // 2026-05-26: guard so we only auto-trigger once per widget mount.
  // Re-mounts (e.g. symbol/timeframe changes) get a fresh auto-fetch
  // because the whole widget tree under `snapshot.candles.isEmpty` is
  // rebuilt — that's the right behaviour.
  bool _autoStarted = false;

  @override
  void initState() {
    super.initState();
    // 2026-05-26: operator directive — "θα πρέπει να βλέπεις κεριά
    // χωρίς να κάνεις κάτι". Auto-kick the broker fetch the first time
    // this widget mounts so the chart isn't blank-waiting-for-click.
    // The visible UI is unchanged (existing build() still shows the
    // spinner while `_busy`, then either the chart re-renders or the
    // manual retry button appears on error) — that retry path is the
    // safety net for transient network blips so we don't strand the
    // user with a permanently blank screen.
    //
    // Scheduled in a post-frame callback so we don't trigger setState
    // during the initial build.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted || _autoStarted || _busy) return;
      _autoStarted = true;
      _fetch();
    });
  }

  Future<void> _fetch() async {
    // Capture the localizations before the await so we can build the
    // error message even if the element is disposed mid-fetch.
    final l10n = AppLocalizations.of(context)!;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final api = ref.read(backendClientProvider);
      final now = DateTime.now().toUtc().millisecondsSinceEpoch;
      // 30 days is enough to render the default 200-candle window on
      // every supported timeframe (M1..D1) without rate-limiting the
      // broker — they pace bar fetches at roughly 1 per second.
      final from = now - const Duration(days: 30).inMilliseconds;
      await api.fetchHistoricalData(
        symbol: widget.symbol,
        timeframe: widget.timeframe,
        fromMs: from,
        toMs: now,
      );
      // Re-pull the chart now that bars exist on disk.
      ref.invalidate(widget.slot.chart);
    } catch (err) {
      if (mounted) {
        setState(() => _error =
            l10n.chartFetchFailed(widget.symbol, describeError(err)));
      }
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24),
      child: Column(
        children: [
          Text(
            l10n.chartNoCandles(widget.symbol, widget.timeframe),
            style: const TextStyle(
              color: NeoethosTokens.textMuted,
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 8),
          ElevatedButton.icon(
            onPressed: _busy ? null : _fetch,
            icon: _busy
                ? const SizedBox(
                    width: 14,
                    height: 14,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.cloud_download_outlined),
            label: Text(_busy ? l10n.chartFetchingBroker : l10n.chartFetchBroker),
          ),
          if (_error != null) ...[
            const SizedBox(height: 8),
            Text(
              _error!,
              style: const TextStyle(
                color: NeoethosTokens.sell,
                fontSize: 11,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

/// F-361: the multi-select indicator menu body shown by the chart
/// panel's right-click. A scrollable card with two labelled groups —
/// "Price overlays" and "Oscillators" — each entry a CheckboxListTile
/// bound to the SAME `slot.activeIndicators` provider the chip row uses.
///
/// It watches that provider (via the enclosing Consumer), so every tick
/// rebuilds the checkboxes from the canonical set and the chip row +
/// ProChart update simultaneously. Because this is a dialog body (not a
/// PopupMenuItem), ticking a box does NOT dismiss it — the operator can
/// toggle several in one session, then close with "Done" or by tapping
/// the barrier.
class _IndicatorMultiSelectMenu extends ConsumerWidget {
  final _ChartSlot slot;
  const _IndicatorMultiSelectMenu({required this.slot});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final active = ref.watch(slot.activeIndicators);

    void toggle(String id) {
      final notifier = ref.read(slot.activeIndicators.notifier);
      final next = {...notifier.state};
      if (next.contains(id)) {
        next.remove(id);
      } else {
        next.add(id);
      }
      notifier.state = next;
    }

    Widget groupHeader(String text) => Padding(
          padding: const EdgeInsets.fromLTRB(12, 10, 12, 4),
          child: Text(
            text.toUpperCase(),
            style: const TextStyle(
              fontSize: 10,
              letterSpacing: 0.5,
              fontWeight: FontWeight.w800,
              color: NeoethosTokens.textMuted,
            ),
          ),
        );

    Widget row(String id) {
      final serverFed = kChartServerIndicators.contains(id);
      return CheckboxListTile(
        dense: true,
        contentPadding: const EdgeInsets.symmetric(horizontal: 8),
        controlAffinity: ListTileControlAffinity.leading,
        value: active.contains(id),
        activeColor: NeoethosTokens.accent,
        // The dialog stays open: this only flips the provider; the menu
        // rebuilds from the watch above. No Navigator.pop here.
        onChanged: (_) => toggle(id),
        title: Text(
          _indicatorLabel(id),
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: NeoethosTokens.textPrimary,
          ),
        ),
        subtitle: serverFed
            ? Text(
                l10n.chartServerStrip,
                style: const TextStyle(
                    fontSize: 9, color: NeoethosTokens.textMuted),
              )
            : null,
      );
    }

    return Material(
      color: Colors.transparent,
      child: Container(
        constraints: const BoxConstraints(maxHeight: 440),
        decoration: BoxDecoration(
          color: NeoethosTokens.panelBg,
          border: Border.all(color: NeoethosTokens.border),
          borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
          boxShadow: const [
            BoxShadow(color: Colors.black54, blurRadius: 12, offset: Offset(0, 4)),
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 10, 8, 2),
              child: Row(
                children: [
                  const Icon(Icons.tune, size: 14, color: NeoethosTokens.accent),
                  const SizedBox(width: 6),
                  Text(
                    l10n.chartIndicatorsPanel(slot.label),
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w800,
                      color: NeoethosTokens.textPrimary,
                    ),
                  ),
                ],
              ),
            ),
            const Divider(height: 8),
            Flexible(
              child: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    groupHeader(l10n.chartGroupPriceOverlays),
                    for (final id in _priceOverlayIndicators) row(id),
                    groupHeader(l10n.chartGroupOscillators),
                    for (final id in _oscillatorIndicators) row(id),
                  ],
                ),
              ),
            ),
            const Divider(height: 8),
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 2, 8, 8),
              child: Row(
                children: [
                  Text(
                    l10n.chartActiveCount(active.length),
                    style: const TextStyle(
                      fontSize: 10,
                      color: NeoethosTokens.textMuted,
                    ),
                  ),
                  const Spacer(),
                  TextButton(
                    onPressed: () => Navigator.of(context).pop(),
                    child: Text(l10n.chartDone),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;
  const _Chip({
    required this.label,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: 10,
          vertical: 5,
        ),
        decoration: BoxDecoration(
          color: selected
              ? NeoethosTokens.accent.withValues(alpha: 0.18)
              : NeoethosTokens.surfaceBg,
          border: Border.all(
            color: selected ? NeoethosTokens.accent : NeoethosTokens.border,
          ),
          borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            color: selected ? NeoethosTokens.accent : NeoethosTokens.textPrimary,
          ),
        ),
      ),
    );
  }
}

class _Loading extends StatelessWidget {
  const _Loading();
  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16),
      child: Text(
        l10n.chartLoadingCandles,
        style: const TextStyle(color: NeoethosTokens.textMuted, fontSize: 12),
      ),
    );
  }
}

