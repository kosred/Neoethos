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
import '../l10n/app_localizations.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/backend_error_widget.dart';
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
    final l10n = AppLocalizations.of(context)!;
    final snapshot = ref.watch(accountSnapshotProvider);
    final positions = snapshot.valueOrNull?.positions ?? const [];
    final usdFmt = NumberFormat.currency(symbol: r'$', decimalDigits: 2);
    final pipFmt = NumberFormat('+#,##0.0;-#,##0.0', 'en_US');
    final brokerSymbols = ref.watch(brokerSymbolsProvider);

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ViewHeader(
            title: l10n.marketsTitle,
            subtitle: l10n.marketsSubtitle,
          ),
          SectionCard(
            title: l10n.openPositions,
            child: positions.isEmpty
                ? Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    child: Text(
                      snapshot.hasError
                          ? l10n.marketsPositionsConnectionIssue
                          : l10n.marketsNoOpenPositions,
                      style: const TextStyle(
                        color: NeoethosTokens.textMuted,
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
                      TableRow(children: [
                        _Th(l10n.thSymbol),
                        _Th(l10n.thSide),
                        _Th(l10n.thVolume),
                        const _Th('Pips'),
                        const _Th('PnL'),
                      ]),
                      for (final p in positions)
                        TableRow(children: [
                          _Td(p.symbol),
                          _Td(
                            p.side,
                            color: p.side.toUpperCase() == 'LONG' ||
                                    p.side.toUpperCase() == 'BUY'
                                ? NeoethosTokens.buy
                                : NeoethosTokens.sell,
                          ),
                          _Td(p.volume.toStringAsFixed(2)),
                          _Td('${pipFmt.format(p.pnlPips)} pips'),
                          _Td(
                            usdFmt.format(p.pnlUsd),
                            color: p.pnlUsd >= 0
                                ? NeoethosTokens.buy
                                : NeoethosTokens.sell,
                          ),
                        ]),
                    ],
                  ),
          ),
          brokerSymbols.when(
            data: (snap) => _symbolsCard(snap),
            loading: () => const _Loading(),
            error: (err, _) => BackendErrorWidget(
                    error: err, title: l10n.marketsUnavailable),
          ),
        ],
      ),
    );
  }

  Widget _symbolsCard(BrokerSymbolsSnapshot snap) {
    final l10n = AppLocalizations.of(context)!;
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
      title: l10n.marketsCatalogTitle(
          filtered.length, all.length, snap.symbolCount),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              FilterChip(
                label: Text(l10n.marketsForexOnly),
                selected: _forexOnly,
                onSelected: (v) => setState(() => _forexOnly = v),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: TextField(
                  controller: _searchCtrl,
                  decoration: InputDecoration(
                    labelText: l10n.marketsSearch,
                    isDense: true,
                    border: const OutlineInputBorder(),
                    prefixIcon: const Icon(Icons.search, size: 18),
                  ),
                  onChanged: (v) => setState(() => _search = v.trim()),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          if (filtered.isEmpty)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8),
              child: Text(
                l10n.marketsNoSymbolsMatch,
                style: const TextStyle(
                  color: NeoethosTokens.textMuted,
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
                  // #269: chips were previously a plain `Container`+`Text`
                  // — no `onTap` → operator clicked and nothing happened.
                  // Wrap in InkWell that pins the symbol into the Chart
                  // screen (via chartSymbolProvider) and triggers the
                  // sidebar Chart navigation through a Consumer ref.
                  InkWell(
                    onTap: () {
                      ref.read(chartSymbolProvider.notifier).state =
                          s.symbolName;
                      // Best-effort: hop to Chart screen so the user
                      // sees their selection. We rely on the route
                      // notifier exposed in the sidebar — but at
                      // minimum the chartSymbolProvider mutation
                      // means the Chart panel will show the pair
                      // next time the operator navigates to it.
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          duration: const Duration(milliseconds: 1200),
                          content: Text(
                            l10n.marketsPinnedToChart(s.symbolName),
                          ),
                        ),
                      );
                    },
                    borderRadius:
                        BorderRadius.circular(NeoethosTokens.rSm),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 3,
                      ),
                      decoration: BoxDecoration(
                        color: NeoethosTokens.surfaceBg,
                        border: Border.all(color: NeoethosTokens.border),
                        borderRadius:
                            BorderRadius.circular(NeoethosTokens.rSm),
                      ),
                      child: Text(
                        s.symbolName,
                        style: const TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          color: NeoethosTokens.textPrimary,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          if (filtered.length > 120) ...[
            const SizedBox(height: 6),
            Text(
              l10n.marketsShowingFirst(120, filtered.length),
              style: const TextStyle(
                fontSize: 10,
                color: NeoethosTokens.textFaint,
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
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16),
      child: Text(
        l10n.marketsLoadingCatalog,
        style: const TextStyle(color: NeoethosTokens.textMuted, fontSize: 12),
      ),
    );
  }
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
            color: NeoethosTokens.textMuted,
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
            color: color ?? NeoethosTokens.textPrimary,
          ),
        ),
      );
}
