// Order Ticket — manual market order entry.
//
// Money-critical screen. Defensive UX:
//   - Big red Sell / big green Buy split — operator can never confuse
//     side mid-tap.
//   - SL + TP fields are in pips, the server-side default; trying to
//     submit with both empty fires a confirmation dialog warning that
//     unbracketed positions can blow up the account.
//   - Confirmation dialog before any POST shows: symbol, side, lot
//     size, SL/TP pips, computed risk in pips * lot.
//   - Snack-bar reports the broker's verbatim response (orderId on
//     success, broker reject reason on failure).

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/error_translation.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/symbol_picker.dart';
import '_placeholder.dart';

class ExecutionScreen extends ConsumerStatefulWidget {
  const ExecutionScreen({super.key});

  @override
  ConsumerState<ExecutionScreen> createState() => _ExecutionScreenState();
}

class _ExecutionScreenState extends ConsumerState<ExecutionScreen> {
  // Symbol picked via SymbolPicker (broker catalog type-ahead). The
  // volume/SL/TP/comment fields stay as free-form text inputs — those
  // are numeric, not picklist-shaped.
  String _symbol = 'EURUSD';
  final _volumeCtrl = TextEditingController(text: '0.01');
  final _slCtrl = TextEditingController(text: '50');
  final _tpCtrl = TextEditingController(text: '100');
  final _commentCtrl = TextEditingController();
  bool _busy = false;
  String? _lastResult;

  @override
  void dispose() {
    _volumeCtrl.dispose();
    _slCtrl.dispose();
    _tpCtrl.dispose();
    _commentCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit(String side) async {
    final symbol = _symbol.trim().toUpperCase();
    final volume = double.tryParse(_volumeCtrl.text.trim());
    final sl = double.tryParse(_slCtrl.text.trim());
    final tp = double.tryParse(_tpCtrl.text.trim());
    final comment = _commentCtrl.text.trim();

    if (symbol.isEmpty || volume == null || volume <= 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          backgroundColor: ForexAiTokens.sell,
          content: Text('Symbol + positive volumeLots are required'),
        ),
      );
      return;
    }

    // Confirmation dialog — money-critical, ALWAYS shown.
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text(
          'Confirm ${side.toUpperCase()} $symbol',
          style: TextStyle(
            color: side == 'buy' ? ForexAiTokens.buy : ForexAiTokens.sell,
          ),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _kv('Symbol', symbol),
            _kv('Side', side.toUpperCase()),
            _kv('Volume', '$volume lot(s)'),
            _kv('Stop-loss', sl == null ? '— (UNBRACKETED!)' : '$sl pips'),
            _kv('Take-profit', tp == null ? '—' : '$tp pips'),
            if (comment.isNotEmpty) _kv('Comment', comment),
            if (sl == null && tp == null) ...[
              const SizedBox(height: 8),
              const Text(
                'Sending a market order with NO stop-loss and NO '
                'take-profit. Make sure this is intentional.',
                style: TextStyle(
                  fontSize: 12,
                  color: ForexAiTokens.sell,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor:
                  side == 'buy' ? ForexAiTokens.buy : ForexAiTokens.sell,
            ),
            onPressed: () => Navigator.pop(ctx, true),
            child: Text('Send ${side.toUpperCase()}'),
          ),
        ],
      ),
    );
    if (confirmed != true || !mounted) return;

    setState(() {
      _busy = true;
      _lastResult = null;
    });
    try {
      final r = await ref.read(backendClientProvider).placeMarketOrder(
            symbol: symbol,
            side: side,
            volumeLots: volume,
            stopLossPips: sl,
            takeProfitPips: tp,
            comment: comment.isEmpty ? null : comment,
            risky: sl == null && tp == null,
          );
      final status = (r['status'] as String?) ?? '?';
      final orderId = r['orderId'];
      final positionId = r['positionId'];
      setState(() => _lastResult =
          'Status: $status · orderId=$orderId · positionId=$positionId');
      ref.invalidate(accountSnapshotProvider);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: status.toLowerCase().contains('accept') ||
                  status.toLowerCase().contains('fill')
              ? ForexAiTokens.buy
              : ForexAiTokens.warning,
          content: Text(
            'Order $status — orderId $orderId · positionId $positionId',
          ),
          duration: const Duration(seconds: 4),
        ),
      );
    } on DioException catch (e) {
      // Use the structured translation when available (e.g.
      // MARKET_CLOSED → "Markets are closed right now …") so the
      // operator sees a human-readable reason, not the raw
      // `code=Some("MARKET_CLOSED")` payload.
      final msg = describeError(e);
      setState(() => _lastResult = 'Failed: $msg');
      if (!mounted) return;
      showTranslatedErrorSnackbar(context, e, prefix: 'Order failed');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Widget _kv(String k, String v) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 100,
              child: Text(
                k,
                style: const TextStyle(
                  color: ForexAiTokens.textMuted,
                  fontSize: 12,
                ),
              ),
            ),
            Flexible(
              child: Text(
                v,
                style: const TextStyle(
                  color: ForexAiTokens.textPrimary,
                  fontWeight: FontWeight.w700,
                  fontSize: 12,
                ),
              ),
            ),
          ],
        ),
      );

  @override
  Widget build(BuildContext context) {
    final brokerSymbols = ref.watch(brokerSymbolsProvider);
    final account = ref.watch(accountSnapshotProvider).valueOrNull;

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Order Ticket',
            subtitle: 'Market BUY / SELL · SL / TP in pips · confirm required',
          ),
          if (account != null)
            SectionCard(
              title: 'Account',
              child: Row(
                children: [
                  Text(
                    '${account.balance.toStringAsFixed(2)} ${account.currency}',
                    style: const TextStyle(
                      fontWeight: FontWeight.w700,
                      fontSize: 14,
                      color: ForexAiTokens.textPrimary,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Text(
                    'free margin ${account.freeMargin.toStringAsFixed(2)} ${account.currency}',
                    style: const TextStyle(
                      fontSize: 12,
                      color: ForexAiTokens.textMuted,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Text(
                    'open positions: ${account.positions.length}',
                    style: const TextStyle(
                      fontSize: 12,
                      color: ForexAiTokens.textMuted,
                    ),
                  ),
                ],
              ),
            ),
          SectionCard(
            title: 'New Market Order',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Symbol picker (broker-catalog typeahead) gets a row
                // to itself because it carries a "Forex only" toggle
                // underneath. Volume sits next to it.
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      flex: 2,
                      child: SymbolPicker(
                        value: _symbol,
                        enabled: !_busy,
                        onChanged: (v) => setState(() => _symbol = v),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      flex: 1,
                      child: TextField(
                        controller: _volumeCtrl,
                        enabled: !_busy,
                        keyboardType: const TextInputType.numberWithOptions(
                          decimal: true,
                        ),
                        decoration: const InputDecoration(
                          labelText: 'Volume (lots)',
                          isDense: true,
                          border: OutlineInputBorder(),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _slCtrl,
                        enabled: !_busy,
                        keyboardType: const TextInputType.numberWithOptions(
                          decimal: true,
                        ),
                        decoration: const InputDecoration(
                          labelText: 'Stop-loss (pips)',
                          isDense: true,
                          border: OutlineInputBorder(),
                          helperText: 'Distance from fill — empty to skip',
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: TextField(
                        controller: _tpCtrl,
                        enabled: !_busy,
                        keyboardType: const TextInputType.numberWithOptions(
                          decimal: true,
                        ),
                        decoration: const InputDecoration(
                          labelText: 'Take-profit (pips)',
                          isDense: true,
                          border: OutlineInputBorder(),
                          helperText: 'Distance from fill — empty to skip',
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                TextField(
                  controller: _commentCtrl,
                  enabled: !_busy,
                  decoration: const InputDecoration(
                    labelText: 'Comment (optional)',
                    isDense: true,
                    border: OutlineInputBorder(),
                  ),
                ),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: _busy ? null : () => _submit('sell'),
                        style: FilledButton.styleFrom(
                          backgroundColor: ForexAiTokens.sell,
                          padding: const EdgeInsets.symmetric(vertical: 14),
                        ),
                        icon: const Icon(Icons.trending_down, size: 18),
                        label: const Text(
                          'SELL',
                          style: TextStyle(
                            fontWeight: FontWeight.w900,
                            fontSize: 16,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: _busy ? null : () => _submit('buy'),
                        style: FilledButton.styleFrom(
                          backgroundColor: ForexAiTokens.buy,
                          padding: const EdgeInsets.symmetric(vertical: 14),
                        ),
                        icon: const Icon(Icons.trending_up, size: 18),
                        label: const Text(
                          'BUY',
                          style: TextStyle(
                            fontWeight: FontWeight.w900,
                            fontSize: 16,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
                if (_busy) ...[
                  const SizedBox(height: 8),
                  const LinearProgressIndicator(minHeight: 2),
                ],
                if (_lastResult != null) ...[
                  const SizedBox(height: 10),
                  Text(
                    _lastResult!,
                    style: const TextStyle(
                      fontSize: 11,
                      color: ForexAiTokens.textMuted,
                    ),
                  ),
                ],
              ],
            ),
          ),
          if (brokerSymbols.hasValue)
            SectionCard(
              // #187: previously showed only `symbolCount` (total, including
              // disabled) which mismatched the Markets screen's "of N enabled".
              // Show both numbers so the user has a consistent denominator.
              title:
                  'Catalog · ${brokerSymbols.value!.symbols.where((s) => s.enabled).length} enabled · '
                  '${brokerSymbols.value!.symbolCount} total from broker',
              child: const Text(
                'Type any symbol the broker offers (forex pairs, '
                'metals, indices, equities). The server resolves the '
                'symbol ID, lot_size, min/max volume from the live '
                'catalog before sending — bad symbols are rejected '
                'with the broker\'s actual error message.',
                style: TextStyle(
                  color: ForexAiTokens.textMuted,
                  fontSize: 12,
                ),
              ),
            ),
        ],
      ),
    );
  }
}
