// Symbol + timeframe pickers backed by the live broker catalog.
//
// The broker offers ~830 symbols (cTrader Demo on this account). A
// 830-entry dropdown is unusable, and a plain TextField lets the user
// type garbage that the server only catches at submit time. The
// pickers here are the middle ground:
//
//   * `SymbolPicker` — type-ahead Autocomplete sourced from
//     `brokerSymbolsProvider` (= /broker/symbols). Filters to the
//     enabled-and-forex-shaped subset by default (the same filter
//     the Markets screen uses); toggle off to see the full 830-symbol
//     catalog including stocks, indices, crypto.
//
//   * `TimeframePicker` — DropdownButton sourced from
//     `brokerTimeframesProvider` (= /broker/timeframes, which mirrors
//     `neoethos_core::CANONICAL_TIMEFRAMES`). Eleven entries today
//     (M1..MN1) so the dropdown stays compact.
//
// Both widgets refuse to render their interactive surface until the
// upstream provider has data — no hardcoded fallbacks. Loading
// surfaces a small spinner; errors surface the actual error message
// + a remediation hint ("Settings → save credentials → Broker Setup
// → Re-authenticate"). This is the contract the user explicitly
// asked for ("τα tf,pairs,charts etc καρφωνουμε συνεχεια πραγματα
// στον κωδικα αντι να τα βρουμε απο τον broker").

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';

/// Type-ahead symbol picker backed by `brokerSymbolsProvider`. The
/// parent receives the current value via [onChanged] and provides
/// the initial value via [value]; this widget owns its internal
/// TextEditingController so the parent doesn't need one.
class SymbolPicker extends ConsumerStatefulWidget {
  /// Currently-selected symbol (e.g. `"EURUSD"`). Drives the field
  /// label when the user hasn't typed anything yet.
  final String value;

  /// Called whenever the user picks a suggestion OR submits the text
  /// directly. The string is already uppercased + trimmed.
  final ValueChanged<String> onChanged;

  /// Optional disabling — used while a button-driven request is
  /// in-flight so the user can't switch mid-submit.
  final bool enabled;

  /// Label rendered above the field (e.g. `"Symbol"`, `"Pair"`). The
  /// catalog count is appended automatically when the provider has data.
  final String label;

  /// When true (default), only enabled-and-forex-shaped symbols are
  /// shown. Equities and crypto are filtered out — that's what the
  /// operator wants 95% of the time. Toggle off via the "Show all"
  /// checkbox the widget renders inline.
  final bool forexOnlyDefault;

  const SymbolPicker({
    super.key,
    required this.value,
    required this.onChanged,
    this.enabled = true,
    this.label = 'Symbol',
    this.forexOnlyDefault = true,
  });

  @override
  ConsumerState<SymbolPicker> createState() => _SymbolPickerState();
}

/// #184: symbol category filters surfaced as chips above the
/// Autocomplete. Defaults to `forex` (the same behavior the picker
/// shipped with originally — operators want pairs 95% of the time)
/// but lets the user flip to metals, equities/indices, or "all"
/// without leaving the picker.
enum _SymbolCategory { forex, metals, equities, all }

class _SymbolPickerState extends ConsumerState<SymbolPicker> {
  late _SymbolCategory _category =
      widget.forexOnlyDefault ? _SymbolCategory.forex : _SymbolCategory.all;

  @override
  Widget build(BuildContext context) {
    final brokerSymbols = ref.watch(brokerSymbolsProvider);

    return brokerSymbols.when(
      data: (snap) => _buildPicker(snap),
      loading: () => _buildLoading(),
      error: (err, _) => _buildError(err),
    );
  }

  Widget _buildPicker(BrokerSymbolsSnapshot snap) {
    // #184: filter universe by selected category. `all` is intentionally
    // unfiltered (other than `enabled`) so the user can fall back to the
    // raw broker list if our heuristics misclassify something.
    final source = switch (_category) {
      _SymbolCategory.forex => snap.forexLikeEnabled,
      _SymbolCategory.metals => snap.metalsEnabled,
      _SymbolCategory.equities => snap.equitiesAndIndicesEnabled,
      _SymbolCategory.all => snap.symbols.where((s) => s.enabled).toList(),
    };

    final names = source.map((s) => s.symbolName).toList(growable: false);
    final categoryLabel = switch (_category) {
      _SymbolCategory.forex => 'forex',
      _SymbolCategory.metals => 'metals',
      _SymbolCategory.equities => 'equities/indices',
      _SymbolCategory.all => 'all',
    };
    final activeLabel =
        '${widget.label} · ${names.length} $categoryLabel from broker';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Autocomplete<String>(
          initialValue: TextEditingValue(text: widget.value),
          // Suggest top-20 case-insensitive substring matches. Returning
          // *every* match for an empty query would spam the dropdown
          // with 830 entries; we only suggest once the user has typed
          // at least one character.
          optionsBuilder: (TextEditingValue v) {
            final query = v.text.trim().toUpperCase();
            if (query.isEmpty) {
              return names.take(15);
            }
            return names
                .where((n) => n.toUpperCase().contains(query))
                .take(20);
          },
          onSelected: (sel) => widget.onChanged(sel.toUpperCase()),
          fieldViewBuilder: (
            ctx,
            controller,
            focusNode,
            onSubmitted,
          ) {
            return TextField(
              controller: controller,
              focusNode: focusNode,
              enabled: widget.enabled,
              textCapitalization: TextCapitalization.characters,
              inputFormatters: [
                FilteringTextInputFormatter.allow(RegExp(r'[A-Za-z0-9.]')),
              ],
              decoration: InputDecoration(
                labelText: activeLabel,
                isDense: true,
                border: const OutlineInputBorder(),
                helperText:
                    'Type to filter the broker catalog. Pick from suggestions.',
              ),
              onChanged: (v) {
                // Mirror to the parent on every keystroke too — the
                // user might submit without ever opening the
                // suggestions panel (typing the full name).
                widget.onChanged(v.trim().toUpperCase());
              },
              onSubmitted: (_) => onSubmitted(),
            );
          },
          optionsViewBuilder: (ctx, onSelected, options) {
            // Compact suggestion list with broker description on the
            // right so the user knows *what* each ticker is — useful
            // because "AAA" could be a stock, a metal, or whatever.
            return Align(
              alignment: Alignment.topLeft,
              child: Material(
                elevation: 4,
                child: ConstrainedBox(
                  constraints: const BoxConstraints(
                    maxHeight: 280,
                    maxWidth: 360,
                  ),
                  child: ListView.builder(
                    padding: EdgeInsets.zero,
                    shrinkWrap: true,
                    itemCount: options.length,
                    itemBuilder: (ctx, i) {
                      final name = options.elementAt(i);
                      final descr = source
                          .firstWhere(
                            (s) => s.symbolName == name,
                            orElse: () => const BrokerSymbol(
                              symbolId: 0,
                              symbolName: '',
                              enabled: false,
                              description: null,
                            ),
                          )
                          .description;
                      return InkWell(
                        onTap: () => onSelected(name),
                        child: Padding(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 10,
                            vertical: 6,
                          ),
                          child: Row(
                            children: [
                              SizedBox(
                                width: 80,
                                child: Text(
                                  name,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w700,
                                    fontSize: 12,
                                  ),
                                ),
                              ),
                              Expanded(
                                child: Text(
                                  descr ?? '',
                                  style: const TextStyle(
                                    fontSize: 11,
                                    color: ForexAiTokens.textMuted,
                                  ),
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ),
            );
          },
        ),
        const SizedBox(height: 4),
        // #184: category chips. Selecting one narrows the autocomplete
        // pool to that asset class; "all" gives the unfiltered broker
        // catalog. The picker remembers the choice for the lifetime of
        // the widget instance (state is local on purpose — different
        // screens may want different defaults).
        Row(
          children: [
            for (final entry in [
              (_SymbolCategory.forex, 'Forex (${snap.forexLikeEnabled.length})'),
              (_SymbolCategory.metals, 'Metals (${snap.metalsEnabled.length})'),
              (
                _SymbolCategory.equities,
                'Equities/Indices (${snap.equitiesAndIndicesEnabled.length})'
              ),
              (_SymbolCategory.all, 'All (${snap.symbols.where((s) => s.enabled).length})'),
            ])
              Padding(
                padding: const EdgeInsets.only(right: 6),
                child: FilterChip(
                  label: Text(
                    entry.$2,
                    style: const TextStyle(fontSize: 11),
                  ),
                  selected: _category == entry.$1,
                  visualDensity: VisualDensity.compact,
                  materialTapTargetSize:
                      MaterialTapTargetSize.shrinkWrap,
                  onSelected: widget.enabled
                      ? (_) => setState(() => _category = entry.$1)
                      : null,
                ),
              ),
          ],
        ),
      ],
    );
  }

  Widget _buildLoading() {
    return _StatusPlaceholder(
      label: widget.label,
      message: 'Loading broker symbol catalog from /broker/symbols…',
      tone: ForexAiTokens.textMuted,
    );
  }

  Widget _buildError(Object err) {
    return _StatusPlaceholder(
      label: widget.label,
      message:
          'Broker symbol catalog unavailable: $err\nOpen Settings → save cTrader credentials → Broker Setup → Re-authenticate.',
      tone: ForexAiTokens.warning,
    );
  }
}

/// Dropdown timeframe picker backed by `brokerTimeframesProvider`.
class TimeframePicker extends ConsumerWidget {
  final String value;
  final ValueChanged<String> onChanged;
  final bool enabled;
  final String label;

  const TimeframePicker({
    super.key,
    required this.value,
    required this.onChanged,
    this.enabled = true,
    this.label = 'Timeframe',
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final tfAsync = ref.watch(brokerTimeframesProvider);
    return tfAsync.when(
      data: (list) {
        if (list.isEmpty) {
          return _StatusPlaceholder(
            label: label,
            message: '/broker/timeframes returned an empty list — the '
                'server contract is broken; check neoethos_core::'
                'CANONICAL_TIMEFRAMES.',
            tone: ForexAiTokens.sell,
          );
        }
        // If the live list doesn't contain the saved pick (canonical
        // contract changed), snap to the first available so the
        // DropdownButton doesn't assert.
        final current = list.contains(value) ? value : list.first;
        if (current != value) {
          // One-shot post-frame nudge — parent should update next frame.
          WidgetsBinding.instance.addPostFrameCallback((_) => onChanged(current));
        }
        return InputDecorator(
          decoration: InputDecoration(
            labelText: '$label · ${list.length} canonical TFs',
            isDense: true,
            border: const OutlineInputBorder(),
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: current,
              isDense: true,
              isExpanded: true,
              items: [
                for (final tf in list)
                  DropdownMenuItem(value: tf, child: Text(tf)),
              ],
              onChanged: enabled ? (v) { if (v != null) onChanged(v); } : null,
            ),
          ),
        );
      },
      loading: () => _StatusPlaceholder(
        label: label,
        message: 'Loading timeframes from /broker/timeframes…',
        tone: ForexAiTokens.textMuted,
      ),
      error: (err, _) => _StatusPlaceholder(
        label: label,
        message: 'Timeframe list unavailable: $err',
        tone: ForexAiTokens.warning,
      ),
    );
  }
}

class _StatusPlaceholder extends StatelessWidget {
  final String label;
  final String message;
  final Color tone;
  const _StatusPlaceholder({
    required this.label,
    required this.message,
    required this.tone,
  });

  @override
  Widget build(BuildContext context) {
    return InputDecorator(
      decoration: InputDecoration(
        labelText: label,
        isDense: true,
        border: const OutlineInputBorder(),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Text(
          message,
          style: TextStyle(fontSize: 11, color: tone),
        ),
      ),
    );
  }
}
