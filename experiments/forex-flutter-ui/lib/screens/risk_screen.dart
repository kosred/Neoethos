// Risk Settings screen — drawdown caps + the prop-firm preset
// selector that drives them.
//
// The preset dropdown calls `POST /risk/preset` which rewrites
// `config.yaml`'s `risk.preset` field. The numeric thresholds shown
// here are seeded from the active preset at backend startup; switching
// presets does NOT auto-overwrite caps the operator already tuned —
// the dropdown surfaces each preset's hard ceilings inline so the
// user can decide whether to also adjust the numeric fields.

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../l10n/app_localizations.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/backend_error_widget.dart';
import '_placeholder.dart';

class RiskScreen extends ConsumerWidget {
  const RiskScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final async = ref.watch(riskProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ViewHeader(
            title: l10n.riskTitle,
            subtitle: l10n.riskSubtitle,
          ),
          async.when(
            data: (r) => _Body(snapshot: r),
            loading: () => const _Loading(),
            error: (err, _) => BackendErrorWidget(
                    error: err, title: l10n.riskSettingsUnavailable),
          ),
        ],
      ),
    );
  }
}

class _Body extends ConsumerStatefulWidget {
  final RiskSnapshot snapshot;
  const _Body({required this.snapshot});

  @override
  ConsumerState<_Body> createState() => _BodyState();
}

class _BodyState extends ConsumerState<_Body> {
  bool _switching = false;

  Future<void> _switchPreset(String presetId) async {
    if (presetId == widget.snapshot.preset) return;
    final l10n = AppLocalizations.of(context)!;
    setState(() => _switching = true);
    try {
      await ref.read(backendClientProvider).savePropFirmPreset(presetId);
      if (!mounted) return;
      ref.invalidate(riskProvider);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: NeoethosTokens.buy,
          content: Text(l10n.riskPresetSwitched(presetId)),
          duration: const Duration(seconds: 3),
        ),
      );
    } on DioException catch (e) {
      if (!mounted) return;
      showTranslatedErrorSnackbar(context, e,
          prefix: l10n.riskPresetSwitchFailed);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: NeoethosTokens.sell,
          content: Text(l10n.riskPresetSwitchFailedDetail(e.toString())),
        ),
      );
    } finally {
      if (mounted) setState(() => _switching = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final pctFmt = NumberFormat.percentPattern('en_US')
      ..maximumFractionDigits = 2
      ..minimumFractionDigits = 2;
    final snap = widget.snapshot;
    // Back-compat: pre-preset backends return empty `preset`. Treat as
    // FTMO for display purposes so older servers don't render a blank
    // dropdown.
    final activePresetId =
        snap.preset.isEmpty ? 'ftmo' : snap.preset;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: l10n.riskSectionPropFirmPreset,
          child: _PresetPicker(
            activePresetId: activePresetId,
            activePresetDisplay: snap.presetDisplayName.isEmpty
                ? 'FTMO'
                : snap.presetDisplayName,
            available: snap.availablePresets,
            propFirmRulesEnabled: snap.propFirmRulesEnabled,
            switching: _switching,
            onPick: _switchPreset,
          ),
        ),
        SectionCard(
          title: l10n.riskSectionDrawdownLimits,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row(l10n.riskRowDailyDrawdownLimit,
                  pctFmt.format(snap.dailyDrawdownLimit)),
              _Row(l10n.riskRowTotalDrawdownLimit,
                  pctFmt.format(snap.totalDrawdownLimit)),
            ],
          ),
        ),
        SectionCard(
          title: l10n.riskSectionPerTradeRisk,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row(l10n.riskRowCurrentPerTradeRisk,
                  pctFmt.format(snap.riskPerTrade)),
              _Row(l10n.riskRowMinAllowed, pctFmt.format(snap.minRiskPerTrade)),
              _Row(l10n.riskRowMaxAllowed, pctFmt.format(snap.maxRiskPerTrade)),
              _Row(l10n.riskRowMaxLotSize,
                  l10n.riskLotsValue(snap.maxLotSize.toStringAsFixed(2))),
            ],
          ),
        ),
        SectionCard(
          title: l10n.riskSectionSafetyRails,
          child: _Row(
            l10n.riskRowStopLossRequired,
            snap.requireStopLoss
                ? l10n.riskStopLossEnforced
                : l10n.riskStopLossRelaxed,
            accent: snap.requireStopLoss
                ? NeoethosTokens.buy
                : NeoethosTokens.warning,
          ),
        ),
        SectionCard(
          title: l10n.riskSectionEditingCaps,
          child: Text(
            l10n.riskEditingCapsBody,
            style: const TextStyle(
                color: NeoethosTokens.textMuted, fontSize: 12),
          ),
        ),
      ],
    );
  }
}

class _PresetPicker extends StatelessWidget {
  final String activePresetId;
  final String activePresetDisplay;
  final List<PropFirmPresetSummary> available;
  final bool propFirmRulesEnabled;
  final bool switching;
  final ValueChanged<String> onPick;
  const _PresetPicker({
    required this.activePresetId,
    required this.activePresetDisplay,
    required this.available,
    required this.propFirmRulesEnabled,
    required this.switching,
    required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final pctFmt = NumberFormat.percentPattern('en_US')
      ..maximumFractionDigits = 1
      ..minimumFractionDigits = 1;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              l10n.riskActivePreset,
              style: const TextStyle(
                fontSize: 12,
                color: NeoethosTokens.textMuted,
              ),
            ),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: propFirmRulesEnabled
                    ? NeoethosTokens.buy.withValues(alpha: 0.18)
                    : NeoethosTokens.textFaint.withValues(alpha: 0.18),
                border: Border.all(
                  color: propFirmRulesEnabled
                      ? NeoethosTokens.buy
                      : NeoethosTokens.textFaint,
                ),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                activePresetDisplay,
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: propFirmRulesEnabled
                      ? NeoethosTokens.buy
                      : NeoethosTokens.textPrimary,
                ),
              ),
            ),
            if (!propFirmRulesEnabled) ...[
              const SizedBox(width: 8),
              Text(
                l10n.riskPropFirmGateDisabled,
                style: const TextStyle(
                  fontSize: 10,
                  color: NeoethosTokens.textFaint,
                ),
              ),
            ],
          ],
        ),
        const SizedBox(height: 12),
        Text(
          l10n.riskSwitchFirmExplainer,
          style: const TextStyle(
              fontSize: 11, color: NeoethosTokens.textMuted),
        ),
        const SizedBox(height: 10),
        ...available.map(
          (p) => InkWell(
            onTap: switching ? null : () => onPick(p.id),
            child: Container(
              margin: const EdgeInsets.symmetric(vertical: 3),
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: p.id == activePresetId
                    ? NeoethosTokens.accent.withValues(alpha: 0.12)
                    : NeoethosTokens.surfaceBg,
                border: Border.all(
                  color: p.id == activePresetId
                      ? NeoethosTokens.accent
                      : NeoethosTokens.border,
                ),
                borderRadius: BorderRadius.circular(4),
              ),
              child: Row(
                children: [
                  // Left-edge selection dot mirrors a radio without
                  // pulling in `Radio` (its `groupValue`/`onChanged`
                  // API is deprecated in current Flutter).
                  Container(
                    width: 14,
                    height: 14,
                    margin: const EdgeInsets.only(right: 10),
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: p.id == activePresetId
                            ? NeoethosTokens.accent
                            : NeoethosTokens.border,
                        width: 2,
                      ),
                      color: p.id == activePresetId
                          ? NeoethosTokens.accent
                          : Colors.transparent,
                    ),
                  ),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          p.displayName,
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                            color: p.id == activePresetId
                                ? NeoethosTokens.accent
                                : NeoethosTokens.textPrimary,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          l10n.riskPresetDetail(
                            pctFmt.format(p.maxDailyLossPct),
                            pctFmt.format(p.maxOverallDrawdownPct),
                            pctFmt.format(p.challengeProfitTargetPct),
                            p.minTradingDays,
                          ),
                          style: const TextStyle(
                            fontSize: 10,
                            color: NeoethosTokens.textMuted,
                          ),
                        ),
                      ],
                    ),
                  ),
                  if (switching && p.id == activePresetId)
                    const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _Row extends StatelessWidget {
  final String label;
  final String value;
  final Color? accent;
  const _Row(this.label, this.value, {this.accent});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(
          children: [
            SizedBox(
              width: 200,
              child: Text(
                label,
                style: const TextStyle(
                  fontSize: 12,
                  color: NeoethosTokens.textMuted,
                ),
              ),
            ),
            Text(
              value,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: accent ?? NeoethosTokens.textPrimary,
              ),
            ),
          ],
        ),
      );
}

class _Loading extends StatelessWidget {
  const _Loading();
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 16),
        child: Text(
          AppLocalizations.of(context)!.riskLoading,
          style: const TextStyle(
              color: NeoethosTokens.textMuted, fontSize: 12),
        ),
      );
}

