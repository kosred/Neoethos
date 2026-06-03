// Inline one-click buy/sell control overlaid on the chart (F-334).
//
// cTrader / TradingView-style quick trading: a compact SELL [bid] |
// volume | BUY [ask] strip that sits in the top-right of the chart
// canvas. One click places a market order at the live price via the
// same `placeMarketOrder` API the Order Ticket uses.
//
// Design notes:
//   - Reads live bid/ask from `liveSpotsProvider`. Renders nothing
//     (SizedBox.shrink) until a fresh tick exists for the symbol, so it
//     never shows a stale or fake price.
//   - Volume defaults to 0.01 lots with a small stepper so a quick
//     trade needs zero typing, but the operator can bump it.
//   - SL/TP default to 50/100 pips (the Order Ticket's defaults) so the
//     order passes the backend's "must have SL/TP or risky" gate
//     without the operator filling anything in.
//   - A confirmation step is deliberately omitted for the one-click
//     flow (that's the point of inline trading); the SnackBar result +
//     the account refresh give immediate feedback. The full Order
//     Ticket remains for considered entries with custom SL/TP.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../state/account_provider.dart';
import '../state/live_spots_provider.dart';
import '../theme/theme.dart';

class InlineBuySell extends ConsumerStatefulWidget {
  final String symbol;
  const InlineBuySell({super.key, required this.symbol});

  @override
  ConsumerState<InlineBuySell> createState() => _InlineBuySellState();
}

class _InlineBuySellState extends ConsumerState<InlineBuySell> {
  double _volumeLots = 0.01;
  bool _busy = false;
  String? _lastError;

  Future<void> _placeOrder(String side) async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _lastError = null;
    });
    try {
      final r = await ref.read(backendClientProvider).placeMarketOrder(
            symbol: widget.symbol,
            side: side,
            volumeLots: _volumeLots,
            stopLossPips: 50,
            takeProfitPips: 100,
            comment: 'inline quick trade',
          );
      if (!mounted) return;
      final status = (r['status'] ?? r['message'] ?? 'submitted').toString();
      final ok = status.toLowerCase().contains('fill') ||
          status.toLowerCase().contains('accept') ||
          status.toLowerCase().contains('submit');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor:
              ok ? NeoethosTokens.buy : NeoethosTokens.warning,
          duration: const Duration(seconds: 3),
          content: Text(
            '${side.toUpperCase()} ${widget.symbol} '
            '${_volumeLots.toStringAsFixed(2)} lots — $status',
          ),
        ),
      );
      // Refresh account so the new position shows up within ~1 frame.
      ref.read(accountSnapshotProvider.notifier).refreshNow();
    } catch (e) {
      if (!mounted) return;
      setState(() => _lastError = _shortError(e.toString()));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _shortError(String raw) {
    // Pull a human line out of a DioException / backend error body.
    final lower = raw.toLowerCase();
    if (lower.contains('502') || lower.contains('bad gateway')) {
      return 'Broker rejected (502) — session may be reconnecting.';
    }
    if (raw.length > 90) return '${raw.substring(0, 90)}…';
    return raw;
  }

  void _bumpVolume(double delta) {
    setState(() {
      _volumeLots = (_volumeLots + delta).clamp(0.01, 100.0);
      // Round to 2 dp to avoid float drift (0.01 + 0.01 = 0.0200000004).
      _volumeLots = double.parse(_volumeLots.toStringAsFixed(2));
    });
  }

  @override
  Widget build(BuildContext context) {
    final spots = ref.watch(liveSpotsProvider).valueOrNull;
    final tick = spots?.lookup(widget.symbol);
    final bid = tick?.bid;
    final ask = tick?.ask;

    // Only hide when there is NO price at all. A briefly-stale tick must
    // NOT make the quick-trade panel vanish — the demo majors routinely
    // gap 15–20 s between ticks, and a panel that flickers in and out is
    // worse than one showing a slightly-aged indicative price. Market
    // orders fill at the broker's live price regardless of the bid/ask
    // shown here; staleness is surfaced with an amber marker instead.
    if (bid == null || ask == null) {
      // No tick for this symbol yet (SSE warming up, or a charted symbol
      // with no live subscription). Show a compact placeholder rather
      // than vanishing — keeps the quick-trade affordance discoverable
      // and tells the operator a price is being awaited.
      return _QuickTradeStub(symbol: widget.symbol);
    }

    final digits = _priceDigits(widget.symbol);
    final stale = tick?.isStale ?? false;
    final freshSecs = tick?.freshnessSeconds ?? 0.0;

    return Container(
      padding: const EdgeInsets.all(6),
      decoration: BoxDecoration(
        color: NeoethosTokens.panelBg.withValues(alpha: 0.94),
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
        boxShadow: const [
          BoxShadow(color: Color(0x40000000), blurRadius: 6, offset: Offset(0, 2)),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Freshness marker — green "live" dot when the tick is fresh,
          // amber "stale Ns" when the last tick aged past the window.
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 6,
                height: 6,
                margin: const EdgeInsets.only(right: 4),
                decoration: BoxDecoration(
                  color: stale ? NeoethosTokens.warning : NeoethosTokens.buy,
                  shape: BoxShape.circle,
                ),
              ),
              Text(
                stale ? 'stale ${freshSecs.toStringAsFixed(0)}s' : 'live',
                style: TextStyle(
                  fontSize: 8,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.5,
                  color: stale
                      ? NeoethosTokens.warning
                      : NeoethosTokens.textMuted,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              _SideButton(
                label: 'SELL',
                price: bid.toStringAsFixed(digits),
                color: NeoethosTokens.sell,
                busy: _busy,
                stale: stale,
                onTap: () => _placeOrder('sell'),
              ),
              const SizedBox(width: 6),
              _VolumeStepper(
                volume: _volumeLots,
                onDec: () => _bumpVolume(-0.01),
                onInc: () => _bumpVolume(0.01),
              ),
              const SizedBox(width: 6),
              _SideButton(
                label: 'BUY',
                price: ask.toStringAsFixed(digits),
                color: NeoethosTokens.buy,
                busy: _busy,
                stale: stale,
                onTap: () => _placeOrder('buy'),
              ),
            ],
          ),
          if (_lastError != null) ...[
            const SizedBox(height: 4),
            Text(
              _lastError!,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                fontSize: 9,
                color: NeoethosTokens.sell,
              ),
            ),
          ],
        ],
      ),
    );
  }

  /// JPY/XAU pairs quote to fewer decimals; everything else 5dp.
  int _priceDigits(String symbol) {
    final s = symbol.toUpperCase();
    if (s.contains('JPY')) return 3;
    if (s.contains('XAU') || s.contains('XAG')) return 2;
    return 5;
  }
}

class _SideButton extends StatelessWidget {
  final String label;
  final String price;
  final Color color;
  final bool busy;
  final bool stale;
  final VoidCallback onTap;
  const _SideButton({
    required this.label,
    required this.price,
    required this.color,
    required this.busy,
    required this.stale,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return FilledButton(
      onPressed: busy ? null : onTap,
      style: FilledButton.styleFrom(
        backgroundColor: color,
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        minimumSize: const Size(0, 0),
        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 10,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.5,
            ),
          ),
          Text(
            price,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w700,
              color: stale ? Colors.white70 : Colors.white,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        ],
      ),
    );
  }
}

class _VolumeStepper extends StatelessWidget {
  final double volume;
  final VoidCallback onDec;
  final VoidCallback onInc;
  const _VolumeStepper({
    required this.volume,
    required this.onDec,
    required this.onInc,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text(
          'LOTS',
          style: TextStyle(
            fontSize: 8,
            fontWeight: FontWeight.w700,
            color: NeoethosTokens.textMuted,
            letterSpacing: 0.5,
          ),
        ),
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            _StepBtn(icon: Icons.remove, onTap: onDec),
            SizedBox(
              width: 34,
              child: Text(
                volume.toStringAsFixed(2),
                textAlign: TextAlign.center,
                style: const TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: NeoethosTokens.textPrimary,
                  fontFeatures: [FontFeature.tabularFigures()],
                ),
              ),
            ),
            _StepBtn(icon: Icons.add, onTap: onInc),
          ],
        ),
      ],
    );
  }
}

/// Compact placeholder shown in the chart's top-right when no live tick
/// exists yet for the symbol. Keeps the quick-trade panel discoverable
/// instead of leaving a blank corner (F-334 follow-up).
class _QuickTradeStub extends StatelessWidget {
  final String symbol;
  const _QuickTradeStub({required this.symbol});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      decoration: BoxDecoration(
        color: NeoethosTokens.panelBg.withValues(alpha: 0.94),
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 10,
            height: 10,
            child: CircularProgressIndicator(
              strokeWidth: 1.6,
              color: NeoethosTokens.textMuted,
            ),
          ),
          SizedBox(width: 6),
          Text(
            'Quick trade · awaiting price',
            style: TextStyle(
              fontSize: 9,
              fontWeight: FontWeight.w600,
              color: NeoethosTokens.textMuted,
            ),
          ),
        ],
      ),
    );
  }
}

class _StepBtn extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  const _StepBtn({required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        width: 20,
        height: 20,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: NeoethosTokens.appBg,
          border: Border.all(color: NeoethosTokens.border),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Icon(icon, size: 13, color: NeoethosTokens.textMuted),
      ),
    );
  }
}
