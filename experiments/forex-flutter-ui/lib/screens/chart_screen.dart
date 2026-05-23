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

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
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

class ChartScreen extends ConsumerWidget {
  const ChartScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final symbol = ref.watch(chartSymbolProvider);
    final timeframe = ref.watch(chartTimeframeProvider);
    final async = ref.watch(chartProvider);
    final brokerSymbols = ref.watch(brokerSymbolsProvider);
    final brokerTimeframes = ref.watch(brokerTimeframesProvider);

    // Symbol list — broker catalog, filtered to forex-like pairs.
    // Empty list when the broker hasn't responded yet or is offline;
    // the screen surfaces an explicit "connect broker" message in
    // that case rather than faking a chip list.
    final symbolChoices = brokerSymbols.maybeWhen(
      data: (snap) =>
          snap.forexLikeEnabled.map((s) => s.symbolName).toList(growable: false),
      orElse: () => const <String>[],
    );

    // Timeframe list — canonical-timeframes endpoint
    // (`neoethos_core::CANONICAL_TIMEFRAMES` over the wire). Empty
    // list when the endpoint hasn't returned yet; no Dart-side
    // fallback because the contract lives on the server.
    final timeframeChoices = brokerTimeframes.maybeWhen(
      data: (list) => list,
      orElse: () => const <String>[],
    );

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Chart',
            subtitle: 'Local OHLC · symbol / timeframe / 200 candles',
          ),
          SectionCard(
            title: 'Symbol'
                '${brokerSymbols.hasValue ? ' · ${symbolChoices.length} from broker' : ''}',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (symbolChoices.isEmpty)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 6),
                    child: Text(
                      brokerSymbols.hasError
                          ? 'Broker symbol catalog unavailable: '
                              '${brokerSymbols.error}. Open Settings → '
                              'save cTrader credentials → Broker Setup '
                              '→ Re-authenticate. The chip list is '
                              'populated from /broker/symbols, never '
                              'hardcoded in the UI.'
                          : 'Loading broker symbol catalog…',
                      style: const TextStyle(
                        fontSize: 11,
                        color: ForexAiTokens.warning,
                      ),
                    ),
                  )
                else
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: [
                      for (final s in symbolChoices)
                        _Chip(
                          label: s,
                          selected: s == symbol,
                          onTap: () => ref
                              .read(chartSymbolProvider.notifier)
                              .state = s,
                        ),
                    ],
                  ),
              ],
            ),
          ),
          SectionCard(
            title: 'Timeframe'
                '${brokerTimeframes.hasValue ? ' · ${timeframeChoices.length} from canonical contract' : ''}',
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
                          onTap: () => ref
                              .read(chartTimeframeProvider.notifier)
                              .state = t,
                        ),
                    ],
                  ),
          ),
          SectionCard(
            title: 'Indicators · vector_ta',
            child: Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final ind in _indicatorChips)
                  Consumer(
                    builder: (ctx, indRef, _) {
                      final active = indRef
                          .watch(activeIndicatorsProvider)
                          .contains(ind);
                      return _Chip(
                        label: _indicatorLabel(ind),
                        selected: active,
                        onTap: () {
                          final notifier = indRef
                              .read(activeIndicatorsProvider.notifier);
                          final next = {...notifier.state};
                          if (next.contains(ind)) {
                            next.remove(ind);
                          } else {
                            next.add(ind);
                          }
                          notifier.state = next;
                        },
                      );
                    },
                  ),
              ],
            ),
          ),
          async.when(
            data: (c) => _ChartBody(snapshot: c),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
          ),
        ],
      ),
    );
  }
}

class _ChartBody extends ConsumerWidget {
  final ChartSnapshot snapshot;
  const _ChartBody({required this.snapshot});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final changePos = snapshot.priceChangePct >= 0;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: '${snapshot.symbol} · ${snapshot.timeframe}',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  Text(
                    snapshot.latestClose.toStringAsFixed(5),
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.w800,
                      color: ForexAiTokens.textPrimary,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    '${changePos ? '+' : ''}${snapshot.priceChangePct.toStringAsFixed(2)} %',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: changePos
                          ? ForexAiTokens.buy
                          : ForexAiTokens.sell,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    'range ${snapshot.priceMin.toStringAsFixed(5)} – '
                    '${snapshot.priceMax.toStringAsFixed(5)}',
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
              const SizedBox(height: 12),
              if (snapshot.candles.isEmpty)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 24),
                  child: Text(
                    'No candles in window.',
                    style: TextStyle(
                      color: ForexAiTokens.textMuted,
                      fontSize: 12,
                    ),
                  ),
                )
              else
                _ChartCanvasWithOverlays(snapshot: snapshot),
            ],
          ),
        ),
      ],
    );
  }
}

/// Glue widget — watches the active-indicator set, fetches each
/// price-band overlay, and hands a flattened list of lines to the
/// candlestick painter. Oscillators (RSI/MACD/Stoch/ADX/ATR) still
/// toggle in the chip row but don't draw here — they need their own
/// sub-panel with an independent Y-axis (next iteration).
class _ChartCanvasWithOverlays extends ConsumerWidget {
  final ChartSnapshot snapshot;
  const _ChartCanvasWithOverlays({required this.snapshot});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final active = ref.watch(activeIndicatorsProvider);
    final overlayLines = <_PaintedLine>[];
    var colorIdx = 0;
    for (final ind in active) {
      if (!_priceBandOverlays.contains(ind)) continue;
      final snap = ref.watch(indicatorProvider(ind));
      snap.whenData((s) {
        for (final line in s.lines) {
          overlayLines.add(_PaintedLine(
            label: '${_indicatorLabel(ind)}${s.lines.length > 1 ? " · ${line.name.split("_").last}" : ""}',
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
        SizedBox(
          height: 320,
          child: CustomPaint(
            painter: _CandlestickPainter(
              snapshot: snapshot,
              overlays: overlayLines,
            ),
            size: Size.infinite,
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
        // Oscillators that the user toggled on but we can't render
        // on the price canvas yet — surface a hint so they know it
        // worked but is parked.
        if (active.any((i) => !_priceBandOverlays.contains(i))) ...[
          const SizedBox(height: 4),
          Text(
            'Oscillators (${active.where((i) => !_priceBandOverlays.contains(i)).map(_indicatorLabel).join(", ")}) — sub-panel coming soon.',
            style: const TextStyle(
              fontSize: 10,
              color: ForexAiTokens.textFaint,
            ),
          ),
        ],
      ],
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
  _CandlestickPainter({required this.snapshot, this.overlays = const []});

  @override
  void paint(Canvas canvas, Size size) {
    final candles = snapshot.candles;
    if (candles.isEmpty) return;

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
    final span = (snapshot.priceMax - snapshot.priceMin).abs();
    final pad = span == 0 ? 1e-5 : span * 0.04;
    final ymin = snapshot.priceMin - pad;
    final ymax = snapshot.priceMax + pad;
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

    double yOf(double price) =>
        padTop + (1 - (price - ymin) / yspan) * plotH;

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
    // the candle slice: index 0 of the overlay maps to index 0 of
    // the visible candles (server already trimmed to limit).
    for (final overlay in overlays) {
      final overlayPaint = Paint()
        ..color = overlay.color
        ..strokeWidth = 1.5
        ..style = PaintingStyle.stroke;
      Offset? prev;
      final maxLen =
          overlay.values.length < candles.length ? overlay.values.length : candles.length;
      for (var i = 0; i < maxLen; i++) {
        final v = overlay.values[i];
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
      if (ts == null) return '#$idx';
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
      old.overlays.length != overlays.length ||
      // Compare colors/lengths as a cheap proxy — full value-by-value
      // comparison would dominate paint time for long histories.
      !_overlaysShallowEqual(old.overlays, overlays);

  static bool _overlaysShallowEqual(List<_PaintedLine> a, List<_PaintedLine> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i].color != b[i].color || a[i].values.length != b[i].values.length) {
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
            color: selected
                ? ForexAiTokens.accent
                : ForexAiTokens.border,
          ),
          borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            color: selected
                ? ForexAiTokens.accent
                : ForexAiTokens.textPrimary,
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
