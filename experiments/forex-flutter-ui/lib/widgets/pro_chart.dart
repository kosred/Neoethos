// Professional candlestick chart (F-336) — k_chart_plus.
//
// Replaces the custom CustomPaint chart with a TradingView/cTrader-grade
// k-line: pinch/scroll pan + zoom, MA/BOLL price overlays + MACD/KDJ/RSI/
// WR sub-panels (computed client-side by k_chart_plus, so they "just
// work" regardless of the /indicators endpoint), a live "now price"
// line, and a long-press OHLC crosshair. Fed by the backend's
// broker-passthrough candles (source=broker) plus the live spot stream
// for the forming candle. `onLoadMore` is wired for future scroll-back
// history (needs a `before`-timestamp chart endpoint — follow-up).

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:k_chart_plus/k_chart_plus.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
import '../state/live_spots_provider.dart';
import '../theme/theme.dart';

class ProChart extends ConsumerStatefulWidget {
  final ChartSnapshot snapshot;
  const ProChart({super.key, required this.snapshot});

  @override
  ConsumerState<ProChart> createState() => _ProChartState();
}

class _ProChartState extends ConsumerState<ProChart> {
  bool _ma = true;
  bool _boll = false;
  String? _secondary = 'MACD'; // MACD | KDJ | RSI | WR | null

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

    final mainIndicators = <MainIndicator>[
      if (_ma) MAIndicator(calcParams: const [5, 10, 30]),
      if (_boll) BOLLIndicator(),
    ];
    final secondaryIndicators = <SecondaryIndicator>[
      if (_secondary == 'MACD') MACDIndicator(),
      if (_secondary == 'KDJ') KDJIndicator(),
      if (_secondary == 'RSI') RSIIndicator(),
      if (_secondary == 'WR') WRIndicator(),
    ];
    if (data.isNotEmpty) {
      DataUtil.calculateAll(data, mainIndicators, secondaryIndicators);
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _toolbar(),
        const SizedBox(height: 6),
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

  Widget _toolbar() {
    Widget chip(String label, bool on, VoidCallback onTap) => GestureDetector(
          onTap: onTap,
          child: Container(
            margin: const EdgeInsets.only(right: 6, bottom: 4),
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: on
                  ? ForexAiTokens.accent.withValues(alpha: 0.18)
                  : ForexAiTokens.appBg,
              border: Border.all(
                color: on ? ForexAiTokens.accent : ForexAiTokens.border,
              ),
              borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
            ),
            child: Text(
              label,
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w700,
                color: on ? ForexAiTokens.accent : ForexAiTokens.textMuted,
              ),
            ),
          ),
        );
    return Wrap(
      crossAxisAlignment: WrapCrossAlignment.center,
      children: [
        const Padding(
          padding: EdgeInsets.only(right: 8, bottom: 4),
          child: Text(
            'OVERLAY',
            style: TextStyle(
              fontSize: 9,
              letterSpacing: 0.5,
              color: ForexAiTokens.textMuted,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
        chip('MA', _ma, () => setState(() => _ma = !_ma)),
        chip('BOLL', _boll, () => setState(() => _boll = !_boll)),
        const SizedBox(width: 12),
        const Padding(
          padding: EdgeInsets.only(right: 8, bottom: 4),
          child: Text(
            'SUB-PANEL',
            style: TextStyle(
              fontSize: 9,
              letterSpacing: 0.5,
              color: ForexAiTokens.textMuted,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
        for (final s in const ['MACD', 'KDJ', 'RSI', 'WR'])
          chip(s, _secondary == s,
              () => setState(() => _secondary = _secondary == s ? null : s)),
      ],
    );
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
