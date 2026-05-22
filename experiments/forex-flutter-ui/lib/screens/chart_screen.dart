// Chart screen — symbol + timeframe chips, candlestick canvas painted
// from `/chart` OHLC data. Read-only (the local data dir is the
// source); switching chips refetches via Riverpod.

import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

/// Fallback symbol chips shown when /broker/symbols is unreachable
/// (offline / not authed yet). Real symbols come from the broker.
const _fallbackSymbolChoices = <String>[
  'EURUSD',
  'GBPUSD',
  'USDJPY',
  'XAUUSD',
];

/// Fallback timeframe chips shown ONLY while /broker/timeframes is
/// loading or unreachable. The real list comes from
/// `brokerTimeframesProvider` which reads
/// `neoethos_core::CANONICAL_TIMEFRAMES` over the wire — single source
/// of truth so a workspace-wide contract change propagates to the UI
/// automatically.
const _fallbackTimeframes = <String>['M1', 'H1', 'D1'];

class ChartScreen extends ConsumerWidget {
  const ChartScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final symbol = ref.watch(chartSymbolProvider);
    final timeframe = ref.watch(chartTimeframeProvider);
    final async = ref.watch(chartProvider);
    final brokerSymbols = ref.watch(brokerSymbolsProvider);
    final brokerTimeframes = ref.watch(brokerTimeframesProvider);

    // Symbol list: prefer the broker catalog (filtered to forex-like
    // pairs by default), fall back to a tiny hardcoded set so the
    // chart still renders when the broker is offline.
    final symbolChoices = brokerSymbols.maybeWhen(
      data: (snap) {
        final forex = snap.forexLikeEnabled;
        if (forex.isEmpty) return _fallbackSymbolChoices;
        return forex.map((s) => s.symbolName).toList(growable: false);
      },
      orElse: () => _fallbackSymbolChoices,
    );

    // Timeframe list: from the canonical-timeframes endpoint. We
    // never hardcode this in Dart — the source of truth is
    // `neoethos_core::CANONICAL_TIMEFRAMES`.
    final timeframeChoices = brokerTimeframes.maybeWhen(
      data: (list) => list.isEmpty ? _fallbackTimeframes : list,
      orElse: () => _fallbackTimeframes,
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
                if (brokerSymbols.hasError)
                  const Padding(
                    padding: EdgeInsets.only(bottom: 6),
                    child: Text(
                      'Broker symbol catalog unavailable — showing fallback list. '
                      'Re-authenticate in Broker Setup to populate.',
                      style: TextStyle(
                        fontSize: 11,
                        color: ForexAiTokens.warning,
                      ),
                    ),
                  ),
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
            child: Wrap(
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

class _ChartBody extends StatelessWidget {
  final ChartSnapshot snapshot;
  const _ChartBody({required this.snapshot});

  @override
  Widget build(BuildContext context) {
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
                SizedBox(
                  height: 320,
                  child: CustomPaint(
                    painter: _CandlestickPainter(snapshot: snapshot),
                    size: Size.infinite,
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }
}

class _CandlestickPainter extends CustomPainter {
  final ChartSnapshot snapshot;
  _CandlestickPainter({required this.snapshot});

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
      old.snapshot != snapshot;
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
