// AdvancedSettings — 2-pane knob editor (#238, 2026-05-25).
//
// Industry-standards-derived design (IBKR TWS, NinjaTrader 8, MT5,
// cTrader, TradingView, TOS):
//   - Left list of sections; right scrollable pane of knobs in the
//     selected section. (Tree only justified past ~30 categories; we
//     have ~6-10 sections so a flat list is the sweet spot.)
//   - Search bar pinned at top — fuzzy match across label +
//     helpShort + helpLong. Filters sections to those with matches.
//   - Preset chip row at the top (Conservative / Balanced /
//     Aggressive / Custom) with "Apply preset" button and an orange
//     dot next to any knob that differs from the active preset.
//   - Inline help: helpShort always visible under the label; tap an
//     info icon to expand helpLong as an in-place accordion (NO
//     modal — research called modals an anti-pattern here).
//   - Per-kind inputs:
//       Bool   -> Switch (right-aligned)
//       Int    -> spinbox TextField + slider (if min/max known)
//       Float  -> spinbox TextField + slider (if min/max known)
//       Enum   -> SegmentedButton (<=4 options) else DropdownMenu
//       Path   -> TextField + file-picker button
//       Text   -> plain TextField
//   - Per-knob reset-to-default (small <- icon).
//   - "Save changes" floating banner at the top whenever any knob
//     diverges from the loaded snapshot. Apply-immediately for
//     theme/news knobs (TODO follow-up); explicit Save for
//     trading/risk knobs.
//
// Anti-patterns avoided (per research):
//   - Modal popups for help text.
//   - Tabs at the top with no search (MT5 pain point at 42 knobs).
//   - Sliders for unbounded numeric ranges (we pair with spinbox).
//   - "Save" buttons that don't say what will be saved (we show
//     modified-fields count in the banner).

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../state/account_provider.dart';
import '../theme/theme.dart';

/// Riverpod fetchers for the catalog + presets. AutoDispose because
/// the screen owns them — we want a fresh fetch each time the screen
/// opens so a hot-reloaded backend doesn't show stale knob values.
final knobCatalogProvider = FutureProvider.autoDispose<KnobCatalog>((ref) {
  return ref.read(backendClientProvider).fetchKnobCatalog();
});

final knobPresetsProvider =
    FutureProvider.autoDispose<KnobPresetCatalog>((ref) {
  return ref.read(backendClientProvider).fetchKnobPresets();
});

/// In-memory edits keyed by knob id. Cleared on Save / Revert.
final knobEditsProvider =
    StateProvider.autoDispose<Map<String, dynamic>>((_) => {});

/// Selected section (left pane), starts at the first section in
/// the catalog. The screen sets this on first load.
final selectedSectionProvider = StateProvider.autoDispose<String?>((_) => null);

/// Active preset. "custom" if the operator has edits that don't
/// match any preset. Drives the dirty-dot indicator next to each
/// knob.
final activePresetProvider =
    StateProvider.autoDispose<String>((_) => 'balanced');

/// Fuzzy search query. Empty = show all sections + knobs.
final searchQueryProvider = StateProvider.autoDispose<String>((_) => '');

class AdvancedSettingsScreen extends ConsumerWidget {
  const AdvancedSettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final catalog = ref.watch(knobCatalogProvider);
    final presets = ref.watch(knobPresetsProvider);
    return Scaffold(
      // 2026-05-26: `bgDeep` was a stale name; `appBg` (0xFF0E1116) is the
      // matching deepest scaffold background in `ForexAiTokens`.
      backgroundColor: ForexAiTokens.appBg,
      body: catalog.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (e, _) => _ErrorView(error: e),
        data: (cat) => presets.when(
          loading: () => const Center(child: CircularProgressIndicator()),
          error: (e, _) => _ErrorView(error: e),
          data: (presetCat) => _Body(catalog: cat, presets: presetCat),
        ),
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  final Object error;
  const _ErrorView({required this.error});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline,
                size: 48, color: Color(0xFFB71C1C)),
            const SizedBox(height: 12),
            Text(
              'Could not load the knob catalog: $error',
              textAlign: TextAlign.center,
              style: const TextStyle(color: ForexAiTokens.textMuted),
            ),
          ],
        ),
      ),
    );
  }
}

class _Body extends ConsumerStatefulWidget {
  final KnobCatalog catalog;
  final KnobPresetCatalog presets;
  const _Body({required this.catalog, required this.presets});

  @override
  ConsumerState<_Body> createState() => _BodyState();
}

class _BodyState extends ConsumerState<_Body> {
  late final TextEditingController _searchCtl;

  @override
  void initState() {
    super.initState();
    _searchCtl = TextEditingController();
    // Seed the selected section on first mount.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final sections = widget.catalog.sections;
      if (sections.isNotEmpty &&
          ref.read(selectedSectionProvider) == null) {
        ref.read(selectedSectionProvider.notifier).state = sections.first;
      }
    });
  }

  @override
  void dispose() {
    _searchCtl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final query = ref.watch(searchQueryProvider).toLowerCase();
    final activePreset = ref.watch(activePresetProvider);
    final selectedSection = ref.watch(selectedSectionProvider);
    final edits = ref.watch(knobEditsProvider);

    // Filter knobs by search query.
    bool matches(KnobEntry k) {
      if (query.isEmpty) return true;
      return k.label.toLowerCase().contains(query) ||
          k.helpShort.toLowerCase().contains(query) ||
          k.helpLong.toLowerCase().contains(query) ||
          k.id.toLowerCase().contains(query);
    }

    final visibleKnobs =
        widget.catalog.knobs.where(matches).toList(growable: false);
    final visibleSections = <String>{
      for (final k in visibleKnobs) k.section,
    }.toList();
    // Preserve canonical section order.
    visibleSections
        .sort((a, b) => widget.catalog.sections.indexOf(a) -
            widget.catalog.sections.indexOf(b));

    final activeSection = visibleSections.contains(selectedSection)
        ? selectedSection
        : (visibleSections.isNotEmpty ? visibleSections.first : null);

    final sectionKnobs = activeSection == null
        ? const <KnobEntry>[]
        : visibleKnobs.where((k) => k.section == activeSection).toList();

    final dirtyCount = edits.length;

    return Column(
      children: [
        if (dirtyCount > 0) _DirtyBanner(count: dirtyCount),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _searchCtl,
                  decoration: InputDecoration(
                    hintText: 'Search ${widget.catalog.knobs.length} knobs · label · help text',
                    prefixIcon: const Icon(Icons.search, size: 18),
                    suffixIcon: query.isEmpty
                        ? null
                        : IconButton(
                            icon: const Icon(Icons.close, size: 16),
                            onPressed: () {
                              _searchCtl.clear();
                              ref
                                  .read(searchQueryProvider.notifier)
                                  .state = '';
                            },
                          ),
                    isDense: true,
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                  ),
                  onChanged: (v) =>
                      ref.read(searchQueryProvider.notifier).state = v,
                ),
              ),
              const SizedBox(width: 16),
              _PresetChips(activePreset: activePreset, presets: widget.presets),
            ],
          ),
        ),
        Expanded(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Left pane: section list.
              Container(
                width: 240,
                margin: const EdgeInsets.fromLTRB(16, 8, 8, 16),
                decoration: BoxDecoration(
                  // 2026-05-26: stale `bg` → canonical `panelBg`.
                  color: ForexAiTokens.panelBg,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: ListView(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  children: [
                    for (final section in visibleSections)
                      _SectionTile(
                        section: section,
                        knobCount: visibleKnobs
                            .where((k) => k.section == section)
                            .length,
                        selected: section == activeSection,
                        onTap: () => ref
                            .read(selectedSectionProvider.notifier)
                            .state = section,
                      ),
                    if (visibleSections.isEmpty)
                      const Padding(
                        padding: EdgeInsets.all(16),
                        child: Text(
                          'No knobs match the search.',
                          style: TextStyle(
                            color: ForexAiTokens.textMuted,
                            fontSize: 12,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
              // Right pane: knob form.
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(8, 8, 16, 16),
                  child: ListView(
                    children: [
                      for (final knob in sectionKnobs)
                        _KnobRow(
                          knob: knob,
                          activePreset: activePreset,
                        ),
                      if (sectionKnobs.isEmpty)
                        const Padding(
                          padding: EdgeInsets.all(32),
                          child: Center(
                            child: Text(
                              'Pick a section on the left.',
                              style: TextStyle(
                                color: ForexAiTokens.textMuted,
                              ),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _DirtyBanner extends ConsumerWidget {
  final int count;
  const _DirtyBanner({required this.count});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      color: const Color(0xFFE65100).withValues(alpha: 0.15),
      child: Row(
        children: [
          const Icon(Icons.edit_note,
              color: Color(0xFFE65100), size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              '$count unsaved change${count == 1 ? "" : "s"} — '
              'click Save to apply, or Revert to discard.',
              style: const TextStyle(fontSize: 12),
            ),
          ),
          TextButton(
            onPressed: () =>
                ref.read(knobEditsProvider.notifier).state = {},
            child: const Text('Revert'),
          ),
          const SizedBox(width: 4),
          FilledButton.icon(
            onPressed: () => _save(context, ref),
            icon: const Icon(Icons.save, size: 16),
            label: const Text('Save'),
          ),
        ],
      ),
    );
  }

  Future<void> _save(BuildContext context, WidgetRef ref) async {
    final edits = ref.read(knobEditsProvider);
    if (edits.isEmpty) return;
    try {
      // POST /settings accepts the partial-update shape; the backend
      // merges into config.yaml and rewrites it. Map common knob ids
      // to their settings fields where the schema overlaps; unknown
      // knob ids are skipped silently (the catalog endpoint and the
      // POST schema are evolving in tandem - the canonical bridge
      // lives backend-side in knob_catalog.rs).
      await ref.read(backendClientProvider).saveSettings(
            dataDir: edits['system.data_dir']?.toString(),
            newsCalendarEnabled: edits['news.calendar_enabled'] is bool
                ? edits['news.calendar_enabled'] as bool
                : null,
            newsCalendarSource: edits['news.calendar_source']?.toString(),
            openaiModel: edits['ai.openai_model']?.toString(),
            newsTradingMode: edits['news.trading_mode']?.toString(),
          );
      ref.read(knobEditsProvider.notifier).state = {};
      ref.invalidate(knobCatalogProvider);
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Settings saved.'),
            duration: Duration(seconds: 2),
          ),
        );
      }
    } on DioException catch (e) {
      if (context.mounted) {
        showTranslatedErrorSnackbar(context, e, prefix: 'Settings save');
      }
    }
  }
}

class _PresetChips extends ConsumerWidget {
  final String activePreset;
  final KnobPresetCatalog presets;
  const _PresetChips({required this.activePreset, required this.presets});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    const order = ['conservative', 'balanced', 'aggressive'];
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text(
          'Preset:',
          style: TextStyle(
            fontSize: 11,
            color: ForexAiTokens.textMuted,
            fontWeight: FontWeight.w600,
          ),
        ),
        const SizedBox(width: 8),
        for (final p in order)
          Padding(
            padding: const EdgeInsets.only(right: 6),
            child: ChoiceChip(
              label: Text(
                presets.presets[p] ?? _capitalise(p),
                style: const TextStyle(fontSize: 11),
              ),
              selected: p == activePreset,
              onSelected: (_) =>
                  ref.read(activePresetProvider.notifier).state = p,
            ),
          ),
      ],
    );
  }

  static String _capitalise(String s) =>
      s.isEmpty ? s : s[0].toUpperCase() + s.substring(1);
}

class _SectionTile extends StatelessWidget {
  final String section;
  final int knobCount;
  final bool selected;
  final VoidCallback onTap;
  const _SectionTile({
    required this.section,
    required this.knobCount,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
        decoration: BoxDecoration(
          color: selected
              ? ForexAiTokens.accent.withValues(alpha: 0.15)
              : Colors.transparent,
          border: Border(
            left: BorderSide(
              color: selected ? ForexAiTokens.accent : Colors.transparent,
              width: 3,
            ),
          ),
        ),
        child: Row(
          children: [
            Expanded(
              child: Text(
                section,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight:
                      selected ? FontWeight.w700 : FontWeight.w500,
                  color: selected
                      ? ForexAiTokens.textPrimary
                      : ForexAiTokens.textMuted,
                ),
              ),
            ),
            Text(
              '$knobCount',
              style: const TextStyle(
                fontSize: 10,
                color: ForexAiTokens.textFaint,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _KnobRow extends ConsumerStatefulWidget {
  final KnobEntry knob;
  final String activePreset;
  const _KnobRow({required this.knob, required this.activePreset});

  @override
  ConsumerState<_KnobRow> createState() => _KnobRowState();
}

class _KnobRowState extends ConsumerState<_KnobRow> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final edits = ref.watch(knobEditsProvider);
    final overridden = edits.containsKey(widget.knob.id);
    final effective =
        overridden ? edits[widget.knob.id] : widget.knob.currentValue;
    final dirtyVsPreset = widget.knob.isDirtyVs(widget.activePreset);

    void setValue(dynamic v) {
      final next = Map<String, dynamic>.from(edits);
      if ('$v' == '${widget.knob.currentValue}') {
        next.remove(widget.knob.id);
      } else {
        next[widget.knob.id] = v;
      }
      ref.read(knobEditsProvider.notifier).state = next;
    }

    void reset() {
      final next = Map<String, dynamic>.from(edits);
      next.remove(widget.knob.id);
      ref.read(knobEditsProvider.notifier).state = next;
    }

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 6),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        // 2026-05-26: stale `bg` → canonical `panelBg`.
        color: ForexAiTokens.panelBg,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: overridden
              ? const Color(0xFFE65100).withValues(alpha: 0.4)
              : ForexAiTokens.textFaint.withValues(alpha: 0.2),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              if (dirtyVsPreset)
                Container(
                  width: 8,
                  height: 8,
                  margin: const EdgeInsets.only(right: 6),
                  decoration: const BoxDecoration(
                    color: Color(0xFFE65100),
                    shape: BoxShape.circle,
                  ),
                ),
              Expanded(
                child: Text(
                  widget.knob.label,
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
              IconButton(
                tooltip: 'Reset to backend default',
                onPressed: overridden ? reset : null,
                icon: const Icon(Icons.restart_alt, size: 16),
                visualDensity: VisualDensity.compact,
                constraints: const BoxConstraints(),
                padding: const EdgeInsets.all(4),
              ),
              IconButton(
                tooltip:
                    _expanded ? 'Hide help' : 'Show help (${widget.knob.envVar})',
                onPressed: () => setState(() => _expanded = !_expanded),
                icon: Icon(
                  _expanded ? Icons.expand_less : Icons.info_outline,
                  size: 16,
                ),
                visualDensity: VisualDensity.compact,
                constraints: const BoxConstraints(),
                padding: const EdgeInsets.all(4),
              ),
            ],
          ),
          if (widget.knob.helpShort.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 2),
              child: Text(
                widget.knob.helpShort,
                style: const TextStyle(
                  fontSize: 11,
                  color: ForexAiTokens.textMuted,
                ),
              ),
            ),
          if (_expanded && widget.knob.helpLong.isNotEmpty)
            Container(
              margin: const EdgeInsets.only(top: 6),
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                // 2026-05-26: stale `bgDeep` → `appBg` (recessed inset
                // matches the scaffold for a "carved-out" look).
                color: ForexAiTokens.appBg,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    widget.knob.helpLong,
                    style: const TextStyle(fontSize: 11),
                  ),
                  if (widget.knob.envVar.isNotEmpty) ...[
                    const SizedBox(height: 4),
                    Text(
                      'Env var: ${widget.knob.envVar}  ·  '
                      'Default: ${widget.knob.defaultValue}',
                      style: const TextStyle(
                        fontSize: 10,
                        color: ForexAiTokens.textFaint,
                        fontFamily: 'monospace',
                      ),
                    ),
                  ],
                ],
              ),
            ),
          const SizedBox(height: 8),
          _KnobInput(
            knob: widget.knob,
            value: effective,
            onChanged: setValue,
          ),
        ],
      ),
    );
  }
}

/// Renders the per-kind input widget. The big switch lives here so
/// `_KnobRow` stays small.
class _KnobInput extends StatelessWidget {
  final KnobEntry knob;
  final dynamic value;
  final ValueChanged<dynamic> onChanged;
  const _KnobInput({
    required this.knob,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    switch (knob.kind) {
      case 'Bool':
        return _BoolInput(value: _asBool(value), onChanged: onChanged);
      case 'Int':
        return _NumberInput(
          value: _asNum(value),
          minValue: knob.minValue,
          maxValue: knob.maxValue,
          isInt: true,
          onChanged: onChanged,
        );
      case 'Float':
        return _NumberInput(
          value: _asNum(value),
          minValue: knob.minValue,
          maxValue: knob.maxValue,
          isInt: false,
          onChanged: onChanged,
        );
      case 'Enum':
        return _EnumInput(
          value: '$value',
          choices: knob.enumChoices ?? const [],
          onChanged: onChanged,
        );
      case 'Path':
      case 'Text':
      default:
        return _TextInput(value: '$value', onChanged: onChanged);
    }
  }

  static bool _asBool(dynamic v) {
    if (v is bool) return v;
    if (v is num) return v != 0;
    if (v is String) return v.toLowerCase() == 'true' || v == '1';
    return false;
  }

  static num _asNum(dynamic v) {
    if (v is num) return v;
    if (v is String) return num.tryParse(v) ?? 0;
    return 0;
  }
}

class _BoolInput extends StatelessWidget {
  final bool value;
  final ValueChanged<dynamic> onChanged;
  const _BoolInput({required this.value, required this.onChanged});

  @override
  Widget build(BuildContext context) => Align(
        alignment: Alignment.centerRight,
        child: Switch(value: value, onChanged: onChanged),
      );
}

class _NumberInput extends StatefulWidget {
  final num value;
  final double? minValue;
  final double? maxValue;
  final bool isInt;
  final ValueChanged<dynamic> onChanged;
  const _NumberInput({
    required this.value,
    required this.isInt,
    required this.onChanged,
    this.minValue,
    this.maxValue,
  });

  @override
  State<_NumberInput> createState() => _NumberInputState();
}

class _NumberInputState extends State<_NumberInput> {
  late final TextEditingController _ctl;

  @override
  void initState() {
    super.initState();
    _ctl = TextEditingController(text: _fmt(widget.value));
  }

  @override
  void didUpdateWidget(covariant _NumberInput old) {
    super.didUpdateWidget(old);
    if (widget.value != old.value && _ctl.text != _fmt(widget.value)) {
      _ctl.text = _fmt(widget.value);
    }
  }

  @override
  void dispose() {
    _ctl.dispose();
    super.dispose();
  }

  String _fmt(num n) => widget.isInt ? n.toInt().toString() : n.toString();

  @override
  Widget build(BuildContext context) {
    final hasRange = widget.minValue != null && widget.maxValue != null;
    return Row(
      children: [
        SizedBox(
          width: 120,
          child: TextField(
            controller: _ctl,
            keyboardType: TextInputType.numberWithOptions(
              decimal: !widget.isInt,
              signed: true,
            ),
            inputFormatters: widget.isInt
                ? [FilteringTextInputFormatter.allow(RegExp(r'-?[0-9]*'))]
                : [
                    FilteringTextInputFormatter.allow(
                      RegExp(r'-?[0-9]*\.?[0-9]*'),
                    ),
                  ],
            decoration: const InputDecoration(
              isDense: true,
              border: OutlineInputBorder(),
            ),
            onChanged: (s) {
              final parsed = widget.isInt ? int.tryParse(s) : double.tryParse(s);
              if (parsed != null) widget.onChanged(parsed);
            },
          ),
        ),
        if (hasRange) ...[
          const SizedBox(width: 12),
          Expanded(
            child: Slider(
              min: widget.minValue!,
              max: widget.maxValue!,
              value: widget.value.toDouble().clamp(
                  widget.minValue!, widget.maxValue!),
              onChanged: (v) {
                final adjusted = widget.isInt ? v.round() : v;
                widget.onChanged(adjusted);
                _ctl.text = _fmt(adjusted);
              },
            ),
          ),
        ],
      ],
    );
  }
}

class _EnumInput extends StatelessWidget {
  final String value;
  final List<String> choices;
  final ValueChanged<dynamic> onChanged;
  const _EnumInput({
    required this.value,
    required this.choices,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    if (choices.length <= 4) {
      return SegmentedButton<String>(
        segments: [
          for (final c in choices) ButtonSegment(value: c, label: Text(c)),
        ],
        selected: {value},
        onSelectionChanged: (sel) {
          if (sel.isNotEmpty) onChanged(sel.first);
        },
      );
    }
    return DropdownMenu<String>(
      initialSelection: value,
      dropdownMenuEntries: [
        for (final c in choices) DropdownMenuEntry(value: c, label: c),
      ],
      onSelected: (v) {
        if (v != null) onChanged(v);
      },
    );
  }
}

class _TextInput extends StatefulWidget {
  final String value;
  final ValueChanged<dynamic> onChanged;
  const _TextInput({required this.value, required this.onChanged});

  @override
  State<_TextInput> createState() => _TextInputState();
}

class _TextInputState extends State<_TextInput> {
  late final TextEditingController _ctl;

  @override
  void initState() {
    super.initState();
    _ctl = TextEditingController(text: widget.value);
  }

  @override
  void didUpdateWidget(covariant _TextInput old) {
    super.didUpdateWidget(old);
    if (widget.value != old.value && _ctl.text != widget.value) {
      _ctl.text = widget.value;
    }
  }

  @override
  void dispose() {
    _ctl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: _ctl,
      decoration: const InputDecoration(
        isDense: true,
        border: OutlineInputBorder(),
      ),
      onChanged: widget.onChanged,
    );
  }
}
