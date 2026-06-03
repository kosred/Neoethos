// Multi-select symbol picker (F-337) — pick several broker pairs at once
// with checkboxes + search + category filter, for the Discovery multi-pair
// queue (operator asked for "list/dropdown with multi-choice checkbox OR
// search" instead of a free-text box). Backed by the live broker catalog
// (brokerSymbolsProvider).

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../l10n/app_localizations.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';

/// Opens the multi-select picker. Returns the chosen symbol names
/// (uppercase, sorted), or null if cancelled. `preselected` seeds the
/// checked set so it round-trips an existing queue.
Future<List<String>?> showMultiSymbolPicker(
  BuildContext context, {
  Set<String> preselected = const {},
}) {
  return showDialog<List<String>>(
    context: context,
    builder: (_) => _MultiSymbolPickerDialog(preselected: preselected),
  );
}

enum _Cat { forex, metals, equities, all }

class _MultiSymbolPickerDialog extends ConsumerStatefulWidget {
  final Set<String> preselected;
  const _MultiSymbolPickerDialog({required this.preselected});

  @override
  ConsumerState<_MultiSymbolPickerDialog> createState() =>
      _MultiSymbolPickerDialogState();
}

class _MultiSymbolPickerDialogState
    extends ConsumerState<_MultiSymbolPickerDialog> {
  _Cat _cat = _Cat.forex;
  String _query = '';
  late final Set<String> _selected = {...widget.preselected};

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final async = ref.watch(brokerSymbolsProvider);
    return Dialog(
      backgroundColor: NeoethosTokens.panelBg,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 520, maxHeight: 640),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                children: [
                  Text(
                    l10n.multiSymbolPickerTitle,
                    style: const TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: NeoethosTokens.textPrimary,
                    ),
                  ),
                  const Spacer(),
                  Text(
                    l10n.multiSymbolPickerSelectedCount(_selected.length),
                    style: const TextStyle(
                      fontSize: 12,
                      color: NeoethosTokens.accent,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              TextField(
                autofocus: true,
                onChanged: (v) =>
                    setState(() => _query = v.trim().toUpperCase()),
                textCapitalization: TextCapitalization.characters,
                decoration: InputDecoration(
                  isDense: true,
                  prefixIcon: const Icon(Icons.search, size: 18),
                  hintText: l10n.multiSymbolPickerSearchHint,
                  border: const OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 6,
                children: [for (final c in _Cat.values) _catChip(c)],
              ),
              const SizedBox(height: 8),
              Expanded(
                child: async.when(
                  data: _list,
                  loading: () =>
                      const Center(child: CircularProgressIndicator()),
                  error: (e, _) => Center(
                    child: Text(
                      l10n.multiSymbolPickerLoadError(describeError(e)),
                      style: const TextStyle(color: NeoethosTokens.sell),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 10),
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: Text(l10n.commonCancel),
                  ),
                  const SizedBox(width: 8),
                  FilledButton(
                    onPressed: _selected.isEmpty
                        ? null
                        : () => Navigator.pop(
                            context, _selected.toList()..sort()),
                    child: Text(l10n.multiSymbolPickerAddCount(_selected.length)),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _catChip(_Cat c) {
    final l10n = AppLocalizations.of(context)!;
    final label = switch (c) {
      _Cat.forex => l10n.symbolCatForex,
      _Cat.metals => l10n.symbolCatMetals,
      _Cat.equities => l10n.symbolCatEquities,
      _Cat.all => l10n.symbolCatAll,
    };
    final on = _cat == c;
    return GestureDetector(
      onTap: () => setState(() => _cat = c),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: on
              ? NeoethosTokens.accent.withValues(alpha: 0.18)
              : NeoethosTokens.appBg,
          border: Border.all(
              color: on ? NeoethosTokens.accent : NeoethosTokens.border),
          borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            color: on ? NeoethosTokens.accent : NeoethosTokens.textMuted,
          ),
        ),
      ),
    );
  }

  Widget _list(BrokerSymbolsSnapshot snap) {
    final source = switch (_cat) {
      _Cat.forex => snap.forexLikeEnabled,
      _Cat.metals => snap.metalsEnabled,
      _Cat.equities => snap.equitiesAndIndicesEnabled,
      _Cat.all => snap.symbols.where((s) => s.enabled).toList(),
    };
    final filtered = _query.isEmpty
        ? source
        : source
            .where((s) => s.symbolName.toUpperCase().contains(_query))
            .toList();
    if (filtered.isEmpty) {
      return Center(
        child: Text(AppLocalizations.of(context)!.multiSymbolPickerNoMatches,
            style: const TextStyle(color: NeoethosTokens.textMuted)),
      );
    }
    return ListView.builder(
      itemCount: filtered.length,
      itemBuilder: (ctx, i) {
        final s = filtered[i];
        return CheckboxListTile(
          dense: true,
          value: _selected.contains(s.symbolName),
          onChanged: (v) => setState(() {
            if (v == true) {
              _selected.add(s.symbolName);
            } else {
              _selected.remove(s.symbolName);
            }
          }),
          controlAffinity: ListTileControlAffinity.leading,
          activeColor: NeoethosTokens.accent,
          title: Text(
            s.symbolName,
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: NeoethosTokens.textPrimary,
            ),
          ),
          subtitle: s.description == null
              ? null
              : Text(
                  s.description!,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                      fontSize: 10, color: NeoethosTokens.textMuted),
                ),
        );
      },
    );
  }
}
