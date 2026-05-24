// Markets — open positions + broker symbol catalog.
//
// The symbol catalog is what /broker/symbols returns: the full
// 800-ish list the broker offers, including stocks and crypto on
// cTrader accounts. The "Forex only" toggle (on by default) filters
// to 6-letter A-Z names so the operator isn't drowning in equities
// they don't want to see.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

/// #185: ISO 4217 currency codes for the majors + commonly-traded
/// crosses available on cTrader retail. The previous "6-letter uppercase
/// ASCII" filter matched precious-metal codes (XAU/XAG/XPT/XPD) and oil
/// CFD codes alongside actual currency pairs, so the toggle looked
/// broken — operators saw XAUUSD listed under "Forex only" and (rightly)
/// asked why. Strict forex = BOTH halves are recognised ISO currencies.
const _isoCurrencyCodes = <String>{
  // G10 + most commonly cross-traded on retail brokers.
  'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD',
  // Scandinavian / European
  'SEK', 'NOK', 'DKK', 'PLN', 'CZK', 'HUF', 'RON',
  // Asia
  'SGD', 'HKD', 'CNH', 'CNY', 'KRW', 'INR', 'THB',
  // EMs commonly available
  'MXN', 'ZAR', 'TRY', 'BRL', 'RUB', 'ILS', 'ARS',
};

bool _isStrictForexPair(String name) {
  if (name.length != 6) return false;
  final base = name.substring(0, 3).toUpperCase();
  final quote = name.substring(3, 6).toUpperCase();
  return _isoCurrencyCodes.contains(base) &&
      _isoCurrencyCodes.contains(quote);
}

class MarketsScreen extends ConsumerStatefulWidget {
  const MarketsScreen({super.key});

  @override
  ConsumerState<MarketsScreen> createState() => _MarketsScreenState();
}

class _MarketsScreenState extends ConsumerState<MarketsScreen> {
  bool _forexOnly = true;
  String _search = '';
  final _searchCtrl = TextEditingController();

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final snapshot = ref.watch(accountSnapshotProvider);
    final positions = snapshot.valueOrNull?.positions ?? const [];
    final usdFmt = NumberFormat.currency(symbol: r'$', decimalDigits: 2);
    final pipFmt = NumberFormat('+#,##0.0;-#,##0.0', 'en_US');
    final brokerSymbols = ref.watch(brokerSymbolsProvider);

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Markets',
            subtitle: 'Open positions · broker symbol catalog',
          ),
          SectionCard(
            title: 'Open Positions',
            child: positions.isEmpty
                ? Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    child: Text(
                      snapshot.hasError
                          ? 'Connection issue — positions unavailable.'
                          : 'No open positions on the connected account.',
                      style: const TextStyle(
                        color: ForexAiTokens.textMuted,
                        fontSize: 12,
                      ),
                    ),
                  )
                : Table(
                    defaultVerticalAlignment: TableCellVerticalAlignment.middle,
                    columnWidths: const {
                      0: FlexColumnWidth(2),
                      1: FlexColumnWidth(2),
                      2: FlexColumnWidth(2),
                      3: FlexColumnWidth(2),
                      4: FlexColumnWidth(2),
                    },
                    children: [
                      const TableRow(children: [
                        _Th('Symbol'),
                        _Th('Side'),
                        _Th('Volume'),
                        _Th('Pips'),
                        _Th('PnL'),
                      ]),
                      for (final p in positions)
                        TableRow(children: [
                          _Td(p.symbol),
                          _Td(
                            p.side,
                            color: p.side.toUpperCase() == 'LONG' ||
                                    p.side.toUpperCase() == 'BUY'
                                ? ForexAiTokens.buy
                                : ForexAiTokens.sell,
                          ),
                          _Td(p.volume.toStringAsFixed(2)),
                          _Td('${pipFmt.format(p.pnlPips)} pips'),
                          _Td(
                            usdFmt.format(p.pnlUsd),
                            color: p.pnlUsd >= 0
                                ? ForexAiTokens.buy
                                : ForexAiTokens.sell,
                          ),
                        ]),
                    ],
                  ),
          ),
          brokerSymbols.when(
            data: (snap) => _symbolsCard(snap),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
          ),
        ],
      ),
    );
  }

  Widget _symbolsCard(BrokerSymbolsSnapshot snap) {
    final all = snap.symbols.where((s) => s.enabled).toList();
    final filtered = all.where((s) {
      if (_forexOnly && !_isStrictForexPair(s.symbolName)) {
        return false;
      }
      if (_search.isNotEmpty &&
          !s.symbolName.toUpperCase().contains(_search.toUpperCase())) {
        return false;
      }
      return true;
    }).toList();

    // #187: unify the denominators across Markets / Order Ticket /
    // SymbolPicker. There are TWO real numbers — total catalog
    // (`symbolCount`, includes disabled) and currently-tradable
    // (`enabled.length`). Show both so the operator can see at a glance
    // why "Forex only · 104" differs from "all enabled · 443" differs
    // from "catalog · 830".
    return SectionCard(
      title:
          'Broker Symbol Catalog · ${filtered.length} shown · '
          '${all.length} enabled · ${snap.symbolCount} total',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              FilterChip(
                label: const Text('Forex only'),
                selected: _forexOnly,
                onSelected: (v) => setState(() => _forexOnly = v),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: TextField(
                  controller: _searchCtrl,
                  decoration: const InputDecoration(
                    labelText: 'Search',
                    isDense: true,
                    border: OutlineInputBorder(),
                    prefixIcon: Icon(Icons.search, size: 18),
                  ),
                  onChanged: (v) => setState(() => _search = v.trim()),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          if (filtered.isEmpty)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 8),
              child: Text(
                'No symbols match the current filter.',
                style: TextStyle(
                  color: ForexAiTokens.textMuted,
                  fontSize: 12,
                ),
              ),
            )
          else
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final s in filtered.take(120))
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 3,
                    ),
                    decoration: BoxDecoration(
                      color: ForexAiTokens.surfaceBg,
                      border: Border.all(color: ForexAiTokens.border),
                      borderRadius:
                          BorderRadius.circular(ForexAiTokens.rSm),
                    ),
                    child: Text(
                      s.symbolName,
                      style: const TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: ForexAiTokens.textPrimary,
                      ),
                    ),
                  ),
              ],
            ),
          if (filtered.length > 120) ...[
            const SizedBox(height: 6),
            Text(
              'Showing first 120 of ${filtered.length} matches. '
              'Type in the search box to narrow further.',
              style: const TextStyle(
                fontSize: 10,
                color: ForexAiTokens.textFaint,
              ),
            ),
          ],
        ],
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
          'Loading broker symbol catalog…',
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
          'Broker symbol catalog unavailable: $error\n'
          'Configure credentials in Settings, then Re-authenticate in '
          'Broker Setup.',
          style: const TextStyle(color: ForexAiTokens.sell, fontSize: 12),
        ),
      );
}

class _Th extends StatelessWidget {
  final String text;
  const _Th(this.text);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Text(
          text.toUpperCase(),
          style: const TextStyle(
            fontSize: 10,
            letterSpacing: 0.4,
            color: ForexAiTokens.textMuted,
            fontWeight: FontWeight.w700,
          ),
        ),
      );
}

class _Td extends StatelessWidget {
  final String text;
  final Color? color;
  const _Td(this.text, {this.color});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Text(
          text,
          style: TextStyle(
            fontSize: 12,
            color: color ?? ForexAiTokens.textPrimary,
          ),
        ),
      );
}
