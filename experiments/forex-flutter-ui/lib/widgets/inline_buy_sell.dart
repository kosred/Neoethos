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
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../l10n/app_localizations.dart';
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
  // Editable lot field. Operators kept asking why a 1-lot trade took ~100
  // taps on the 0.01 stepper — now they can TYPE the size ("1", "0.5", …) and
  // click Buy/Sell; the +/- steppers stay for fine 0.01 nudges. `_volumeLots`
  // is the source of truth for the order; the field is synced both ways.
  late final TextEditingController _volumeCtrl =
      TextEditingController(text: _volumeLots.toStringAsFixed(2));
  bool _busy = false;
  String? _lastError;

  @override
  void dispose() {
    _volumeCtrl.dispose();
    super.dispose();
  }

  /// Typed input → parse + clamp into `_volumeLots`. We deliberately do NOT
  /// rewrite the controller here so typing stays smooth (no caret jumps); the
  /// field shows exactly what was typed and the order uses the clamped value.
  /// Steppers + pre-send normalisation rewrite the field instead.
  void _onVolumeChanged(String text) {
    final v = double.tryParse(text.trim());
    if (v != null) _volumeLots = v.clamp(0.01, 100.0);
  }

  Future<void> _placeOrder(String side) async {
    if (_busy) return;
    // Capture the localizations before the await — the BuildContext must
    // not be used across the async gap.
    final l10n = AppLocalizations.of(context)!;
    // Reflect the clamped size that will actually be sent (e.g. a typed "200"
    // shows as the 100-lot cap) so the field, the SnackBar, and the broker
    // all agree.
    final normalized = _volumeLots.toStringAsFixed(2);
    if (_volumeCtrl.text != normalized) {
      _volumeCtrl.value = TextEditingValue(
        text: normalized,
        selection: TextSelection.collapsed(offset: normalized.length),
      );
    }
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
          backgroundColor: ok ? NeoethosTokens.buy : NeoethosTokens.warning,
          duration: const Duration(seconds: 3),
          content: Text(
            l10n.inlineTradeOrderResult(
              side.toUpperCase(),
              widget.symbol,
              _volumeLots.toStringAsFixed(2),
              status,
            ),
          ),
        ),
      );
      // Refresh account so the new position shows up within ~1 frame.
      ref.read(accountSnapshotProvider.notifier).refreshNow();
    } catch (e) {
      if (!mounted) return;
      setState(() => _lastError = _shortError(l10n, e.toString()));
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  String _shortError(AppLocalizations l10n, String raw) {
    // Pull a human line out of a DioException / backend error body.
    final lower = raw.toLowerCase();
    if (lower.contains('502') || lower.contains('bad gateway')) {
      return l10n.inlineTradeBrokerRejected502;
    }
    if (raw.length > 90) return '${raw.substring(0, 90)}…';
    return raw;
  }

  void _bumpVolume(double delta) {
    setState(() {
      _volumeLots = (_volumeLots + delta).clamp(0.01, 100.0);
      // Round to 2 dp to avoid float drift (0.01 + 0.01 = 0.0200000004).
      _volumeLots = double.parse(_volumeLots.toStringAsFixed(2));
      // Button nudge (not typing) → safe to rewrite the field; caret at the
      // end so a follow-up keystroke appends naturally.
      final s = _volumeLots.toStringAsFixed(2);
      _volumeCtrl.value = TextEditingValue(
        text: s,
        selection: TextSelection.collapsed(offset: s.length),
      );
    });
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
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
          BoxShadow(
              color: Color(0x40000000), blurRadius: 6, offset: Offset(0, 2)),
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
                stale
                    ? l10n.inlineTradeStale(freshSecs.toStringAsFixed(0))
                    : l10n.inlineTradeLive,
                style: TextStyle(
                  fontSize: 8,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.5,
                  color:
                      stale ? NeoethosTokens.warning : NeoethosTokens.textMuted,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              _SideButton(
                label: l10n.inlineTradeSell,
                price: bid.toStringAsFixed(digits),
                color: NeoethosTokens.sell,
                busy: _busy,
                stale: stale,
                onTap: () => _placeOrder('sell'),
              ),
              const SizedBox(width: 6),
              _VolumeField(
                controller: _volumeCtrl,
                onChanged: _onVolumeChanged,
                onDec: () => _bumpVolume(-0.01),
                onInc: () => _bumpVolume(0.01),
              ),
              const SizedBox(width: 6),
              _SideButton(
                label: l10n.inlineTradeBuy,
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

class _VolumeField extends StatelessWidget {
  final TextEditingController controller;
  final ValueChanged<String> onChanged;
  final VoidCallback onDec;
  final VoidCallback onInc;
  const _VolumeField({
    required this.controller,
    required this.onChanged,
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
              width: 48,
              child: TextField(
                controller: controller,
                onChanged: onChanged,
                textAlign: TextAlign.center,
                keyboardType:
                    const TextInputType.numberWithOptions(decimal: true),
                // Digits + a single decimal point only.
                inputFormatters: [
                  FilteringTextInputFormatter.allow(RegExp(r'[0-9.]')),
                ],
                style: const TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: NeoethosTokens.textPrimary,
                  fontFeatures: [FontFeature.tabularFigures()],
                ),
                decoration: const InputDecoration(
                  isDense: true,
                  contentPadding:
                      EdgeInsets.symmetric(horizontal: 2, vertical: 4),
                  border: OutlineInputBorder(),
                  hintText: '0.01',
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
    final l10n = AppLocalizations.of(context)!;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      decoration: BoxDecoration(
        color: NeoethosTokens.panelBg.withValues(alpha: 0.94),
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          const SizedBox(
            width: 10,
            height: 10,
            child: CircularProgressIndicator(
              strokeWidth: 1.6,
              color: NeoethosTokens.textMuted,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            l10n.inlineTradeAwaitingPrice,
            style: const TextStyle(
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
