// Chart screen — symbol + timeframe chips, candlestick canvas painted
// from `/chart` OHLC data, plus optional indicator overlays from
// `/indicators` (server-side vector_ta). Read-only (the local data
// dir is the source); switching chips refetches via Riverpod.
//
// Symbol + timeframe lists come EXCLUSIVELY from the broker. No
// hardcoded fallbacks — when the broker isn't reachable, the screen
// shows an explicit "connect broker first" message instead of
// faking choices the broker may or may not actually offer.

import 'dart:ui' as ui;

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import 'package:dio/dio.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../charts/chart_viewport.dart';
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

/// Which indicators render directly on the price chart (share the
/// candle's Y-axis range). The others would need a separate sub-panel
/// with its own Y-axis — that lands in a follow-up; for now toggling
/// them on shows the chip as active but doesn't draw anything.
const _priceBandOverlays = <String>{
  'sma',
  'ema',
  'bollinger_bands',
  'vwap',
};

/// One color per indicator-line index — round-robin through this when
/// painting multi-line overlays (e.g. Bollinger Bands has 3 lines).
const _overlayPalette = <Color>[
  Color(0xFF60A5FA), // blue
  Color(0xFFFBBF24), // amber
  Color(0xFF34D399), // emerald
  Color(0xFFF472B6), // pink
  Color(0xFFA78BFA), // violet
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
                  title: multi ? 'Charts (A + B)' : 'Chart',
                  subtitle: multi
                      ? 'Compare two symbols · max 2 panels (deliberate UX constraint)'
                      : 'Local OHLC · symbol / timeframe / 200 candles',
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
                label: const Text('Ask AI'),
              ),
            ],
          ),
          // Compare-mode toggle. Tapping it flips panel B on/off.
          // Limit: TWO panels only. There is no UI affordance to add
          // a third — multi-chart grids fragment attention and bury
          // the trade thesis. The "vs 16" in the task title is the
          // anti-goal we're avoiding.
          SectionCard(
            title: 'Layout',
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
                        ? 'Comparison mode ON — panel A + panel B side-by-side. '
                            'Each panel has its own symbol, timeframe, and '
                            'indicators.'
                        : 'Single chart. Toggle ON to compare two symbols '
                            'side-by-side (max 2 — no 4/8/16 grid by design).',
                    style: const TextStyle(
                      fontSize: 11,
                      color: ForexAiTokens.textMuted,
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
      setState(
          () => _msgs.add((user: false, text: 'Error: ${describeError(e)}')));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: const BoxDecoration(
        color: ForexAiTokens.panelBg,
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(16),
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              const Icon(Icons.psychology_alt_outlined,
                  color: ForexAiTokens.textPrimary),
              const SizedBox(width: 8),
              Text(
                'Ask AI · ${widget.symbol} ${widget.timeframe}',
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
                          ? ForexAiTokens.buy.withValues(alpha: 0.15)
                          : ForexAiTokens.surfaceBg,
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
                  decoration: const InputDecoration(
                    hintText: 'Ask about this chart…',
                    isDense: true,
                    border: OutlineInputBorder(),
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
                label: const Text('Send'),
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
          title: 'Panel ${slot.label} · Symbol',
          child: SymbolPicker(
            value: symbol,
            label: 'Symbol',
            onChanged: (s) => ref.read(slot.symbol.notifier).state = s,
          ),
        ),
        SectionCard(
          title: 'Panel ${slot.label} · Timeframe'
              '${brokerTimeframes.hasValue ? ' · ${timeframeChoices.length} canonical' : ''}',
          child: timeframeChoices.isEmpty
              ? Text(
                  brokerTimeframes.hasError
                      ? 'Timeframe list unavailable: '
                          '${brokerTimeframes.error}'
                      : 'Loading timeframes…',
                  style: const TextStyle(
                    fontSize: 11,
                    color: ForexAiTokens.warning,
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
        SectionCard(
          title: 'Panel ${slot.label} · Indicators',
          child: Wrap(
            spacing: 6,
            runSpacing: 6,
            children: [
              for (final ind in _indicatorChips)
                Consumer(
                  builder: (ctx, indRef, _) {
                    final active =
                        indRef.watch(slot.activeIndicators).contains(ind);
                    // F-268 (2026-05-28): oscillator-pending hint.
                    // Indicators NOT in `_priceBandOverlays` don't
                    // render any line on the price canvas yet — the
                    // user reported "chip activates but no overlay
                    // drawn" as a bug. The Wrap below already shows
                    // a small text note, but the chips themselves
                    // looked identical to working overlays. Mark
                    // oscillators with a leading "• " bullet so the
                    // distinction is visible at the click site, and
                    // wrap in a Tooltip explaining the pending state.
                    final isOscillator = !_priceBandOverlays.contains(ind);
                    final chip = _Chip(
                      label: isOscillator
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
                    return isOscillator
                        ? Tooltip(
                            message:
                                '${_indicatorLabel(ind)} is an oscillator '
                                '— renders in a sub-panel (not the price '
                                'canvas). Sub-panel rendering is parked; '
                                'toggling on/off still affects the '
                                'oscillator-status text below the chart.',
                            child: chip,
                          )
                        : chip;
                  },
                ),
            ],
          ),
        ),
        async.when(
          data: (c) => _ChartBody(snapshot: c, slot: slot),
          loading: () => const _Loading(),
          error: (err, _) => _Error(error: err.toString()),
        ),
      ],
    );
  }
}

class _ChartBody extends ConsumerWidget {
  final ChartSnapshot snapshot;
  final _ChartSlot slot;
  const _ChartBody({required this.snapshot, required this.slot});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
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
          title:
              'Panel ${slot.label} · ${snapshot.symbol} · ${snapshot.timeframe}',
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
                          ? ForexAiTokens.buy
                          : ForexAiTokens.textPrimary,
                    ),
                  ),
                  if (livePrice != null) ...[
                    const SizedBox(width: 6),
                    const Tooltip(
                      message: 'Live tick (sub-2 s)',
                      child: Icon(
                        Icons.bolt,
                        size: 16,
                        color: ForexAiTokens.buy,
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
                          ? (changePos ? ForexAiTokens.buy : ForexAiTokens.sell)
                          : ForexAiTokens.textMuted,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    hasData
                        ? 'range ${snapshot.priceMin.toStringAsFixed(5)} – '
                            '${snapshot.priceMax.toStringAsFixed(5)}'
                        : 'no data — run Data Bootstrap',
                    style: const TextStyle(
                      fontSize: 11,
                      color: ForexAiTokens.textMuted,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 4),
              Text(
                snapshot.headline,
                style: const TextStyle(
                  fontSize: 11,
                  color: ForexAiTokens.textMuted,
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
                // F-336: professional k-line chart (k_chart_plus) —
                // candlesticks + MA/BOLL overlays + MACD/KDJ/RSI/WR
                // sub-panels + pan/zoom + live forming candle. Replaces
                // the custom CustomPaint canvas.
                ProChart(snapshot: snapshot),
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
    final text = snapshot.isDiskCache
        ? 'Cached candles — broker unreachable, showing last saved data'
        : 'Chart data source: ${snapshot.source}';
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: ForexAiTokens.warning.withValues(alpha: 0.08),
        border:
            Border.all(color: ForexAiTokens.warning.withValues(alpha: 0.35)),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Text(
        text,
        style: const TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          color: ForexAiTokens.warning,
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
      if (mounted) setState(() => _error = err.toString());
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24),
      child: Column(
        children: [
          Text(
            'No candles on disk for ${widget.symbol} ${widget.timeframe}.',
            style: const TextStyle(
              color: ForexAiTokens.textMuted,
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
            label: Text(_busy
                ? 'Fetching last 30 days from broker…'
                : 'Fetch last 30 days from broker'),
          ),
          if (_error != null) ...[
            const SizedBox(height: 8),
            Text(
              _error!,
              style: const TextStyle(
                color: ForexAiTokens.sell,
                fontSize: 11,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

/// Glue widget — watches the active-indicator set, fetches each
/// price-band overlay, and hands a flattened list of lines to the
/// candlestick painter. Oscillators (RSI/MACD/Stoch/ADX/ATR) still
/// toggle in the chip row but don't draw here — they need their own
/// sub-panel with an independent Y-axis (next iteration).
class _ChartCanvasWithOverlays extends ConsumerStatefulWidget {
  final ChartSnapshot snapshot;
  final _ChartSlot slot;
  const _ChartCanvasWithOverlays({
    required this.snapshot,
    required this.slot,
  });

  @override
  ConsumerState<_ChartCanvasWithOverlays> createState() =>
      _ChartCanvasWithOverlaysState();
}

class _ChartCanvasWithOverlaysState
    extends ConsumerState<_ChartCanvasWithOverlays> {
  late ChartViewport _viewport;

  @override
  void initState() {
    super.initState();
    _viewport = ChartViewport.live(totalCount: widget.snapshot.candles.length);
  }

  @override
  void didUpdateWidget(covariant _ChartCanvasWithOverlays oldWidget) {
    super.didUpdateWidget(oldWidget);
    final switchedMarket =
        oldWidget.snapshot.symbol != widget.snapshot.symbol ||
            oldWidget.snapshot.timeframe != widget.snapshot.timeframe;
    if (switchedMarket) {
      _viewport =
          ChartViewport.live(totalCount: widget.snapshot.candles.length);
    } else {
      _viewport = _viewport.withTotalCount(widget.snapshot.candles.length);
    }
  }

  void _pan(int deltaBars) {
    if (deltaBars == 0) return;
    final next = _viewport.pan(deltaBars);
    if (next.firstIndex == _viewport.firstIndex &&
        next.visibleCount == _viewport.visibleCount) {
      return;
    }
    setState(() => _viewport = next);
  }

  void _zoom(double factor) {
    final next = _viewport.zoom(factor);
    if (next.firstIndex == _viewport.firstIndex &&
        next.visibleCount == _viewport.visibleCount) {
      return;
    }
    setState(() => _viewport = next);
  }

  void _goLive() {
    final next = _viewport.goLive();
    if (next.firstIndex == _viewport.firstIndex) {
      return;
    }
    setState(() => _viewport = next);
  }

  @override
  Widget build(BuildContext context) {
    final active = ref.watch(widget.slot.activeIndicators);
    final overlayLines = <_PaintedLine>[];
    var colorIdx = 0;
    for (final ind in active) {
      if (!_priceBandOverlays.contains(ind)) continue;
      final snap = ref.watch(widget.slot.indicator(ind));
      snap.whenData((s) {
        for (final line in s.lines) {
          overlayLines.add(_PaintedLine(
            label:
                '${_indicatorLabel(ind)}${s.lines.length > 1 ? " · ${line.name.split("_").last}" : ""}',
            values: line.values,
            color: _overlayPalette[colorIdx % _overlayPalette.length],
          ));
          colorIdx++;
        }
      });
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: Wrap(
            spacing: 6,
            runSpacing: 6,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              _ChartToolButton(
                tooltip: 'Older candles',
                icon: Icons.keyboard_arrow_left,
                onPressed: _viewport.firstIndex <= 0 ? null : () => _pan(-20),
              ),
              _ChartToolButton(
                tooltip: 'Newer candles',
                icon: Icons.keyboard_arrow_right,
                onPressed: _viewport.isAtLiveEnd ? null : () => _pan(20),
              ),
              _ChartToolButton(
                tooltip: 'Zoom in',
                icon: Icons.zoom_in,
                onPressed: () => _zoom(1.25),
              ),
              _ChartToolButton(
                tooltip: 'Zoom out',
                icon: Icons.zoom_out,
                onPressed: () => _zoom(0.8),
              ),
              _ChartToolButton(
                tooltip: 'Go live',
                icon: Icons.my_location,
                onPressed: _viewport.isAtLiveEnd ? null : _goLive,
              ),
              _ViewportBadge(viewport: _viewport),
            ],
          ),
        ),
        // F-334: one-click buy/sell strip, right-aligned just above the
        // candles. Placed in the normal column flow (NOT a Stack overlay
        // on top of the Size.infinite CustomPaint, which never laid the
        // overlay out) so it is always visible. Renders a compact
        // "awaiting price" stub until a live tick exists, an amber
        // "stale" marker when the tick ages, else live SELL/BUY.
        Align(
          alignment: Alignment.centerRight,
          child: Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: InlineBuySell(symbol: widget.snapshot.symbol),
          ),
        ),
        SizedBox(
          height: 320,
          child: Stack(
            children: [
              Listener(
                onPointerSignal: (event) {
                  if (event is PointerScrollEvent) {
                    _zoom(event.scrollDelta.dy < 0 ? 1.15 : 0.85);
                  }
                },
                child: GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onHorizontalDragUpdate: (details) =>
                      _pan((-details.delta.dx / 8).round()),
                  child: CustomPaint(
                    painter: _CandlestickPainter(
                      snapshot: widget.snapshot,
                      overlays: overlayLines,
                      viewport: _viewport,
                    ),
                    size: Size.infinite,
                  ),
                ),
              ),
            ],
          ),
        ),
        if (overlayLines.isNotEmpty) ...[
          const SizedBox(height: 6),
          Wrap(
            spacing: 10,
            runSpacing: 4,
            children: [
              for (final line in overlayLines)
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(width: 10, height: 2, color: line.color),
                    const SizedBox(width: 4),
                    Text(
                      line.label,
                      style: const TextStyle(
                        fontSize: 10,
                        color: ForexAiTokens.textMuted,
                      ),
                    ),
                  ],
                ),
            ],
          ),
        ],
        // F-268 (2026-05-28): oscillators that the user toggled on
        // but we can't render on the price canvas yet — surface an
        // ATTENTIVE hint with the warning icon so the user knows the
        // click registered but the line isn't drawn. Previous
        // fontSize=10 + faint color was easy to miss → operator
        // reported "chip activates but overlay not drawn" as a bug.
        if (active.any((i) => !_priceBandOverlays.contains(i))) ...[
          const SizedBox(height: 6),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(
                Icons.info_outline,
                size: 14,
                color: ForexAiTokens.warning,
              ),
              const SizedBox(width: 4),
              Expanded(
                child: Text(
                  '${active.where((i) => !_priceBandOverlays.contains(i)).map(_indicatorLabel).join(", ")}'
                  ' — oscillator(s) active but sub-panel rendering not yet '
                  'implemented. The chip click registered; no line is drawn '
                  'on the price canvas (oscillators have a different Y-axis).',
                  style: const TextStyle(
                    fontSize: 11,
                    color: ForexAiTokens.warning,
                  ),
                ),
              ),
            ],
          ),
        ],
      ],
    );
  }
}

class _ChartToolButton extends StatelessWidget {
  final String tooltip;
  final IconData icon;
  final VoidCallback? onPressed;
  const _ChartToolButton({
    required this.tooltip,
    required this.icon,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) => SizedBox(
        width: 30,
        height: 30,
        child: IconButton(
          tooltip: tooltip,
          onPressed: onPressed,
          icon: Icon(icon),
          iconSize: 18,
          padding: EdgeInsets.zero,
          splashRadius: 18,
          color: ForexAiTokens.textPrimary,
          disabledColor: ForexAiTokens.textFaint,
        ),
      );
}

class _ViewportBadge extends StatelessWidget {
  final ChartViewport viewport;
  const _ViewportBadge({required this.viewport});

  @override
  Widget build(BuildContext context) {
    final first = viewport.totalCount == 0 ? 0 : viewport.firstIndex + 1;
    final label = '$first-${viewport.visibleEndExclusive} / '
        '${viewport.totalCount}${viewport.isAtLiveEnd ? " · Live" : ""}';
    return Container(
      height: 30,
      padding: const EdgeInsets.symmetric(horizontal: 10),
      alignment: Alignment.center,
      decoration: BoxDecoration(
        color: viewport.isAtLiveEnd
            ? ForexAiTokens.buy.withValues(alpha: 0.10)
            : ForexAiTokens.surfaceBg,
        border: Border.all(
          color: viewport.isAtLiveEnd
              ? ForexAiTokens.buy.withValues(alpha: 0.45)
              : ForexAiTokens.border,
        ),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 10,
          fontWeight: FontWeight.w700,
          color: viewport.isAtLiveEnd
              ? ForexAiTokens.buy
              : ForexAiTokens.textMuted,
        ),
      ),
    );
  }
}

/// Cheap value-type for the painter: one line on the canvas.
class _PaintedLine {
  final String label;
  final List<double> values;
  final Color color;
  const _PaintedLine({
    required this.label,
    required this.values,
    required this.color,
  });
}

class _CandlestickPainter extends CustomPainter {
  final ChartSnapshot snapshot;
  final List<_PaintedLine> overlays;
  final ChartViewport viewport;
  _CandlestickPainter({
    required this.snapshot,
    required this.viewport,
    this.overlays = const [],
  });

  @override
  void paint(Canvas canvas, Size size) {
    final allCandles = snapshot.candles;
    if (allCandles.isEmpty) return;
    final start = viewport.firstIndex.clamp(0, allCandles.length - 1).toInt();
    final end = viewport.visibleEndExclusive
        .clamp(start + 1, allCandles.length)
        .toInt();
    final candles = allCandles.sublist(start, end);

    // Padding so wicks/tops don't clip the edges.
    const padTop = 8.0;
    const padBot = 24.0;
    const padLeft = 56.0;
    const padRight = 8.0;
    final plotW = size.width - padLeft - padRight;
    final plotH = size.height - padTop - padBot;
    if (plotW <= 0 || plotH <= 0) return;

    // Span a touch wider than min/max so the most extreme wicks don't
    // touch the frame.
    var priceMin = double.infinity;
    var priceMax = double.negativeInfinity;
    for (final candle in candles) {
      if (candle.low < priceMin) priceMin = candle.low;
      if (candle.high > priceMax) priceMax = candle.high;
    }
    if (!priceMin.isFinite || !priceMax.isFinite) return;
    final span = (priceMax - priceMin).abs();
    final pad = span == 0 ? 1e-5 : span * 0.04;
    final ymin = priceMin - pad;
    final ymax = priceMax + pad;
    final yspan = ymax - ymin;

    final gridPaint = Paint()
      ..color = ForexAiTokens.border
      ..strokeWidth = 0.5;
    const labelStyle = TextStyle(
      fontSize: 9,
      color: ForexAiTokens.textMuted,
    );

    // Horizontal gridlines + price labels (5 ticks).
    for (var i = 0; i <= 4; i++) {
      final t = i / 4;
      final y = padTop + (1 - t) * plotH;
      canvas.drawLine(
        Offset(padLeft, y),
        Offset(size.width - padRight, y),
        gridPaint,
      );
      final price = ymin + t * yspan;
      _drawText(
        canvas,
        price.toStringAsFixed(5),
        Offset(2, y - 6),
        labelStyle,
      );
    }

    // Candles: each is `slot` wide, with `bar` body width.
    final slot = plotW / candles.length;
    final bar = (slot * 0.7).clamp(1.0, 16.0);
    final upPaint = Paint()..color = ForexAiTokens.buy;
    final downPaint = Paint()..color = ForexAiTokens.sell;
    final wickPaint = Paint()
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;

    double yOf(double price) => padTop + (1 - (price - ymin) / yspan) * plotH;

    for (var i = 0; i < candles.length; i++) {
      final c = candles[i];
      final cx = padLeft + slot * (i + 0.5);
      final up = c.close >= c.open;
      final body = up ? upPaint : downPaint;
      wickPaint.color = up ? ForexAiTokens.buy : ForexAiTokens.sell;

      // Wick (high–low line).
      canvas.drawLine(
        Offset(cx, yOf(c.high)),
        Offset(cx, yOf(c.low)),
        wickPaint,
      );
      // Body (open–close rect).
      final yo = yOf(c.open);
      final yc = yOf(c.close);
      final top = yo < yc ? yo : yc;
      final h = (yc - yo).abs().clamp(1.0, double.infinity);
      canvas.drawRect(
        Rect.fromLTWH(cx - bar / 2, top, bar, h),
        body,
      );
    }

    // Indicator overlays: one polyline per series. Skips leading
    // NaN warm-up bars (most indicators emit NaN until they have
    // enough history) so the line begins where the indicator
    // actually has a value. Indicator series are right-aligned with
    // the candle slice. Indicator series are treated as right-aligned
    // with the chart candles so shorter warm-up series still line up
    // with the latest visible bars.
    for (final overlay in overlays) {
      final overlayPaint = Paint()
        ..color = overlay.color
        ..strokeWidth = 1.5
        ..style = PaintingStyle.stroke;
      Offset? prev;
      final valueOffset = overlay.values.length >= allCandles.length
          ? 0
          : allCandles.length - overlay.values.length;
      for (var i = 0; i < candles.length; i++) {
        final sourceIndex = start + i;
        final valueIndex = sourceIndex - valueOffset;
        if (valueIndex < 0 || valueIndex >= overlay.values.length) {
          prev = null;
          continue;
        }
        final v = overlay.values[valueIndex];
        if (v.isNaN || v.isInfinite) {
          prev = null;
          continue;
        }
        final cx = padLeft + slot * (i + 0.5);
        final cy = yOf(v);
        if (prev != null) {
          canvas.drawLine(prev, Offset(cx, cy), overlayPaint);
        }
        prev = Offset(cx, cy);
      }
    }

    // X-axis: first + middle + last timestamp label.
    final tsFmt = DateFormat('MM-dd HH:mm');
    String tsLabel(int idx) {
      final ts = candles[idx].tsMs;
      if (ts == null) return '#${start + idx}';
      return tsFmt.format(DateTime.fromMillisecondsSinceEpoch(ts));
    }

    _drawText(
      canvas,
      tsLabel(0),
      Offset(padLeft, size.height - padBot + 4),
      labelStyle,
    );
    _drawText(
      canvas,
      tsLabel(candles.length ~/ 2),
      Offset(
        padLeft + plotW / 2 - 30,
        size.height - padBot + 4,
      ),
      labelStyle,
    );
    _drawText(
      canvas,
      tsLabel(candles.length - 1),
      Offset(size.width - padRight - 70, size.height - padBot + 4),
      labelStyle,
    );
  }

  void _drawText(Canvas canvas, String text, Offset where, TextStyle style) {
    final tp = TextPainter(
      text: TextSpan(text: text, style: style),
      textDirection: ui.TextDirection.ltr,
    )..layout();
    tp.paint(canvas, where);
  }

  @override
  bool shouldRepaint(covariant _CandlestickPainter old) =>
      old.snapshot != snapshot ||
      old.viewport.firstIndex != viewport.firstIndex ||
      old.viewport.visibleCount != viewport.visibleCount ||
      old.viewport.totalCount != viewport.totalCount ||
      old.overlays.length != overlays.length ||
      // Compare colors/lengths as a cheap proxy — full value-by-value
      // comparison would dominate paint time for long histories.
      !_overlaysShallowEqual(old.overlays, overlays);

  static bool _overlaysShallowEqual(
      List<_PaintedLine> a, List<_PaintedLine> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i].color != b[i].color ||
          a[i].values.length != b[i].values.length) {
        return false;
      }
    }
    return true;
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
      borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      child: Container(
        padding: const EdgeInsets.symmetric(
          horizontal: 10,
          vertical: 5,
        ),
        decoration: BoxDecoration(
          color: selected
              ? ForexAiTokens.accent.withValues(alpha: 0.18)
              : ForexAiTokens.surfaceBg,
          border: Border.all(
            color: selected ? ForexAiTokens.accent : ForexAiTokens.border,
          ),
          borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            color: selected ? ForexAiTokens.accent : ForexAiTokens.textPrimary,
          ),
        ),
      ),
    );
  }
}

class _Loading extends StatelessWidget {
  const _Loading();
  @override
  Widget build(BuildContext context) => const Padding(
        padding: EdgeInsets.symmetric(vertical: 16),
        child: Text(
          'Loading candles…',
          style: TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
        ),
      );
}

class _Error extends StatelessWidget {
  final String error;
  const _Error({required this.error});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Text(
          'Chart failed: $error',
          style: const TextStyle(color: ForexAiTokens.sell, fontSize: 12),
        ),
      );
}
