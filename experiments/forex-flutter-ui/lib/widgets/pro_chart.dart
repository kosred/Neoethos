// Professional candlestick chart (F-336) — k_chart_plus.
//
// Replaces the custom CustomPaint chart with a TradingView/cTrader-grade
// k-line: pinch/scroll pan + zoom, client-side price overlays (SMA/EMA/
// BOLL) + sub-panels (RSI/MACD/Stoch) computed by k_chart_plus over the
// candle data (so they "just work" regardless of the /indicators
// endpoint and stay pan/zoom aware), a live "now price" line, and a
// long-press OHLC crosshair. Fed by the backend's broker-passthrough
// candles (source=broker) plus the live spot stream for the forming
// candle. `onLoadMore` pages older history in from /chart/history.
//
// F-360: the chart's indicator set is driven ENTIRELY by the chip row
// on the Chart screen (one unified control — no second toolbar here).
// The chips map to k_chart_plus indicators where one exists; the three
// indicators k_chart_plus can't compute (ATR, ADX, VWAP) are taken from
// the server's /indicators endpoint and drawn in a thin strip below the
// k-line (see `_ServerIndicatorStrip`).

import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:k_chart_plus/k_chart_plus.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
import '../state/live_spots_provider.dart';
import '../theme/theme.dart';

/// Chip ids that k_chart_plus computes client-side. Anything NOT in here
/// (atr/adx/vwap) has no k_chart_plus equivalent and is server-fed.
const kChartClientIndicators = <String>{
  'sma',
  'ema',
  'bollinger_bands',
  'rsi',
  'macd',
  'stoch',
};

/// Chip ids that have no k_chart_plus equivalent — computed server-side
/// (vector_ta via /indicators) and drawn in the strip below the k-line.
/// VWAP is price-valued; ATR/ADX are oscillators. All three share the
/// strip, each auto-scaled to its own min/max.
const kChartServerIndicators = <String>['vwap', 'atr', 'adx'];

class ProChart extends ConsumerStatefulWidget {
  final ChartSnapshot snapshot;

  /// The chip-row state for this panel (slot A or B). ProChart rebuilds
  /// its k_chart_plus indicator lists from this set on every build, so
  /// toggling a chip immediately changes what the chart draws.
  final StateProvider<Set<String>> activeIndicators;

  /// The per-indicator server fetch for this panel — family-keyed by
  /// indicator id. Used ONLY for atr/adx/vwap (the ones k_chart_plus
  /// can't compute). Same provider the chip row's `_ChartSlot.indicator`
  /// field points at, so panel A and panel B stay independent.
  final AutoDisposeFutureProviderFamily<IndicatorSnapshot, String>
      indicatorFamily;

  const ProChart({
    super.key,
    required this.snapshot,
    required this.activeIndicators,
    required this.indicatorFamily,
  });

  @override
  ConsumerState<ProChart> createState() => _ProChartState();
}

class _ProChartState extends ConsumerState<ProChart> {
  // ── Scroll-back pagination (F-344) ─────────────────────────────────
  // Older candles fetched on-demand from /chart/history as the operator
  // pans left past the oldest loaded bar — TradingView model: held only
  // in memory, never persisted. `_older` is oldest→newest, all strictly
  // before the snapshot's first candle. Reset when the symbol/TF changes.
  final List<ChartCandle> _older = [];
  bool _loadingMore = false;
  bool _exhausted = false;
  String _seriesKey = '';

  /// Soft memory bound. Covers ≥2 years on M15+ and a large window on M1
  /// without letting an endless M1 scroll-back grow the list unbounded
  /// (k_chart_plus keeps every candle live in memory). At the cap we stop
  /// paginating; switch to a higher timeframe to see further back.
  static const int _maxOlder = 50000;

  @override
  Widget build(BuildContext context) {
    final snap = widget.snapshot;
    final digits = _digits(snap.symbol);
    final active = ref.watch(widget.activeIndicators);

    // Reset the scroll-back buffer when the symbol or timeframe changes —
    // older bars from EURUSD M1 must never bleed into GBPUSD H1. Safe to
    // mutate directly here (no setState): we're already inside build.
    final key = '${snap.symbol}|${snap.timeframe}';
    if (key != _seriesKey) {
      _seriesKey = key;
      _older.clear();
      _loadingMore = false;
      _exhausted = false;
    }

    // k-line series = paged-in older candles + the snapshot's window.
    final data = <KLineEntity>[
      for (final c in _older)
        KLineEntity.fromCustom(
          time: c.tsMs ?? 0,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
          vol: c.volume,
          amount: null,
        ),
      for (final c in snap.candles)
        KLineEntity.fromCustom(
          time: c.tsMs ?? 0,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
          vol: c.volume,
          amount: null,
        ),
    ];

    // Live forming candle: fold the freshest tick into the last bar so the
    // rightmost candle moves in real time.
    final tick = ref.watch(liveSpotsProvider).valueOrNull?.lookup(snap.symbol);
    final bid = tick?.bid, ask = tick?.ask;
    if (data.isNotEmpty && bid != null && ask != null) {
      final mid = (bid + ask) / 2.0;
      final last = data.last;
      last.close = mid;
      if (mid > last.high) last.high = mid;
      if (mid < last.low) last.low = mid;
    }

    // ── Build the k_chart_plus indicator lists FROM THE CHIP ROW. ──────
    // Rebuilt on every build from `active`, then re-run through
    // DataUtil.calculateAll below — so toggling a chip immediately
    // changes what the chart renders. SMA/EMA use a single 20-period
    // line (the chip is a coarse on/off, not a per-period editor); BOLL
    // uses k_chart_plus's standard 20/2. Stoch maps to KDJ (KDJ IS the
    // stochastic oscillator — see the package's kdj_indicator.dart,
    // `name: 'stoch'`).
    final mainIndicators = <MainIndicator>[
      if (active.contains('sma')) MAIndicator(calcParams: const [20]),
      if (active.contains('ema')) EMAIndicator(calcParams: const [20]),
      if (active.contains('bollinger_bands')) BOLLIndicator(),
    ];
    final secondaryIndicators = <SecondaryIndicator>[
      if (active.contains('macd')) MACDIndicator(),
      if (active.contains('rsi')) RSIIndicator(),
      if (active.contains('stoch')) KDJIndicator(),
    ];
    if (data.isNotEmpty) {
      DataUtil.calculateAll(data, mainIndicators, secondaryIndicators);
    }

    // The server-fed indicators the operator has switched on. Only these
    // trigger a /indicators round-trip + the bottom strip.
    final serverActive =
        kChartServerIndicators.where(active.contains).toList(growable: false);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          height: 440,
          child: data.isEmpty
              ? const Center(
                  child: Text(
                    'No candles',
                    style: TextStyle(color: ForexAiTokens.textMuted),
                  ),
                )
              : KChartWidget(
                  data,
                  const KChartStyle(),
                  _colors(),
                  isTrendLine: false,
                  mainIndicators: mainIndicators,
                  secondaryIndicators: secondaryIndicators,
                  fixedLength: digits,
                  showNowPrice: true,
                  volHidden: false,
                  mBaseHeight: 300,
                  timeFormat: TimeFormat.YEAR_MONTH_DAY,
                  detailBuilder: (KLineEntity e) => _detail(e, digits),
                  onLoadMore: (bool isRightEdge) {
                    // k_chart_plus fires onLoadMore(false) when the user
                    // pans to the OLDEST loaded bar (left edge) and
                    // onLoadMore(true) at the newest (right). We only
                    // page older history; the live spot stream keeps the
                    // right edge current.
                    if (!isRightEdge) _fetchOlder(snap);
                  },
                ),
        ),
        // ── Server-fed strip (ATR/ADX/VWAP) ────────────────────────────
        // k_chart_plus has no equivalent for these and exposes no
        // custom-line hook, so we draw them ourselves from /indicators.
        // Right-aligned to the trailing window (the server returns the
        // last N candles, same tail the chart shows at the right edge).
        for (final id in serverActive)
          _ServerIndicatorStrip(
            indicatorId: id,
            digits: digits,
            family: widget.indicatorFamily,
          ),
      ],
    );
  }

  /// Fetch the next page of OLDER candles from the broker and splice them
  /// onto the front of `_older`. Guarded against re-entrancy and against
  /// running once the broker has no more history (or the soft cap is hit).
  Future<void> _fetchOlder(ChartSnapshot snap) async {
    if (_loadingMore || _exhausted) return;
    if (_older.length >= _maxOlder) {
      _exhausted = true;
      return;
    }
    // Cursor = the oldest candle we currently hold.
    final ChartCandle? oldest = _older.isNotEmpty
        ? _older.first
        : (snap.candles.isNotEmpty ? snap.candles.first : null);
    final cursor = oldest?.tsMs;
    if (cursor == null) return; // no timestamps → can't page

    _loadingMore = true;
    try {
      final page = await ref.read(backendClientProvider).fetchChartHistory(
            symbol: snap.symbol,
            timeframe: snap.timeframe,
            beforeMs: cursor,
            limit: 500,
          );
      if (!mounted) return;
      setState(() {
        // Backend guarantees every bar is strictly before the cursor, so
        // there's no overlap — prepend in order (oldest→newest).
        _older.insertAll(0, page.candles);
        if (!page.hasMore || page.candles.isEmpty) _exhausted = true;
      });
    } catch (_) {
      // Broker hiccup — don't hammer it; the in-flight flag is cleared
      // below so the next pan retries.
    } finally {
      _loadingMore = false;
    }
  }

  Widget _detail(KLineEntity e, int digits) {
    String f(double? v) => v == null ? '—' : v.toStringAsFixed(digits);
    Widget row(String k, String v) => Padding(
          padding: const EdgeInsets.symmetric(vertical: 1),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 14,
                child: Text(
                  k,
                  style: const TextStyle(
                    fontSize: 10,
                    color: ForexAiTokens.textMuted,
                  ),
                ),
              ),
              Text(
                v,
                style: const TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w700,
                  color: ForexAiTokens.textPrimary,
                ),
              ),
            ],
          ),
        );
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: ForexAiTokens.panelBg.withValues(alpha: 0.96),
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          row('O', f(e.open)),
          row('H', f(e.high)),
          row('L', f(e.low)),
          row('C', f(e.close)),
        ],
      ),
    );
  }

  int _digits(String symbol) {
    final s = symbol.toUpperCase();
    if (s.contains('JPY')) return 3;
    if (s.contains('XAU') || s.contains('XAG')) return 2;
    return 5;
  }

  KChartColors _colors() => const KChartColors(
        bgColor: ForexAiTokens.appBg,
        upColor: ForexAiTokens.buy,
        dnColor: ForexAiTokens.sell,
        volColor: ForexAiTokens.accent,
        gridColor: ForexAiTokens.border,
        defaultTextColor: ForexAiTokens.textMuted,
        nowPriceUpColor: ForexAiTokens.buy,
        nowPriceDnColor: ForexAiTokens.sell,
        maxColor: ForexAiTokens.textPrimary,
        minColor: ForexAiTokens.textPrimary,
        selectFillColor: ForexAiTokens.panelBg,
        selectBorderColor: ForexAiTokens.border,
      );
}

/// A thin sub-panel for ONE server-computed indicator (ATR / ADX / VWAP).
/// k_chart_plus can't compute these and exposes no external-line hook, so
/// we fetch the series from /indicators (already pan-window-correct: the
/// server returns the trailing N candles) and paint it ourselves. Each
/// strip auto-scales to its own min/max so ATR's tiny values and VWAP's
/// price-level values both fill the band.
class _ServerIndicatorStrip extends ConsumerWidget {
  final String indicatorId;
  final int digits;
  final AutoDisposeFutureProviderFamily<IndicatorSnapshot, String> family;
  const _ServerIndicatorStrip({
    required this.indicatorId,
    required this.digits,
    required this.family,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(family(indicatorId));
    return Container(
      margin: const EdgeInsets.only(top: 6),
      padding: const EdgeInsets.fromLTRB(8, 6, 8, 6),
      decoration: BoxDecoration(
        color: ForexAiTokens.panelBg.withValues(alpha: 0.5),
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: async.when(
        loading: () => _label('${_title()} · loading…'),
        error: (e, _) => _label('${_title()} · unavailable'),
        data: (snap) {
          // Pick the first finite line. atr/adx/vwap are single-output,
          // so there is exactly one line; guard for an empty/all-NaN
          // payload (too-short history) so we degrade to a hint instead
          // of painting nothing.
          final values = snap.lines.isEmpty
              ? const <double>[]
              : snap.lines.first.values;
          final finite = values.where((v) => v.isFinite).toList(growable: false);
          if (finite.isEmpty) {
            return _label('${_title()} · no data (window too short)');
          }
          final last = values.lastWhere((v) => v.isFinite,
              orElse: () => double.nan);
          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _label(
                '${_title()} · ${last.isFinite ? last.toStringAsFixed(_strDigits()) : '—'}',
              ),
              const SizedBox(height: 2),
              SizedBox(
                height: 40,
                width: double.infinity,
                child: CustomPaint(
                  painter: _LinePainter(
                    values: values,
                    color: _color(),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  String _title() {
    switch (indicatorId) {
      case 'atr':
        return 'ATR';
      case 'adx':
        return 'ADX';
      case 'vwap':
        return 'VWAP';
      default:
        return indicatorId.toUpperCase();
    }
  }

  // VWAP is a price level → show full pip precision. ATR/ADX read better
  // with fewer decimals.
  int _strDigits() => indicatorId == 'vwap' ? digits : 2;

  Color _color() {
    switch (indicatorId) {
      case 'atr':
        return ForexAiTokens.warning;
      case 'adx':
        return ForexAiTokens.accent;
      case 'vwap':
        return ForexAiTokens.buy;
      default:
        return ForexAiTokens.textPrimary;
    }
  }

  Widget _label(String text) => Text(
        text,
        style: const TextStyle(
          fontSize: 10,
          fontWeight: FontWeight.w700,
          color: ForexAiTokens.textMuted,
        ),
      );
}

/// Auto-scaling polyline over a value series, painted left→right with the
/// freshest value at the right edge. NaN warm-up entries break the line
/// (no segment drawn across a gap) so the leading flat-zero artefact the
/// old custom painter had can't reappear.
class _LinePainter extends CustomPainter {
  final List<double> values;
  final Color color;
  const _LinePainter({required this.values, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    if (values.length < 2) return;
    double lo = double.infinity, hi = double.negativeInfinity;
    for (final v in values) {
      if (!v.isFinite) continue;
      lo = math.min(lo, v);
      hi = math.max(hi, v);
    }
    if (!lo.isFinite || !hi.isFinite) return;
    final span = (hi - lo).abs() < 1e-12 ? 1.0 : (hi - lo);

    final dx = size.width / (values.length - 1);
    final paint = Paint()
      ..color = color
      ..strokeWidth = 1.2
      ..style = PaintingStyle.stroke
      ..isAntiAlias = true;

    final path = Path();
    var pen = false; // whether the path currently has a live point
    for (int i = 0; i < values.length; i++) {
      final v = values[i];
      if (!v.isFinite) {
        pen = false; // break the line across the gap
        continue;
      }
      final x = dx * i;
      // Invert Y: high values near the top. Pad 3px top/bottom.
      final y = 3 + (size.height - 6) * (1 - (v - lo) / span);
      if (!pen) {
        path.moveTo(x, y);
        pen = true;
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _LinePainter old) =>
      old.values != values || old.color != color;
}
