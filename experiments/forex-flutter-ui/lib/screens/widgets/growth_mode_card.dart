// Growth Mode — the "small-account multiplier" pitch panel.
//
// The user explicitly named this as the differentiator they want
// to lead with: ML Discovery + Training + Auto-Trade pipeline
// turns a €100 starter into something materially bigger over time.
// This card surfaces the math live: starting balance × multiplier
// = current equity, and a forward projection at the current pace
// to a user-set target.
//
// All inputs are local state for now (StateProvider). Persistence
// to config.yaml lands with the broader UI ↔ CLI parity work
// (#118); until then the values reset on app restart, which is OK
// — the panel's job is communication, not analytics.

import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../state/account_provider.dart';
import '../../theme/theme.dart';
import '../_placeholder.dart';

/// User-set "where I started" balance — anchors the multiplier.
/// Defaults to €100 because that's the user's stated starter
/// scenario ("from 100 euros to thousands").
final growthStartingBalanceProvider = StateProvider<double>((_) => 100.0);

/// User-set target balance for the projection line.
final growthTargetBalanceProvider = StateProvider<double>((_) => 10000.0);

/// Risk profile selector. Drives an assumed daily-growth rate for
/// the ETA projection. Real per-day growth varies enormously with
/// the strategy mix Discovery picks — these are conservative
/// anchors so the projection looks credible rather than fantastical.
enum GrowthRiskProfile {
  conservative, // ~0.3% daily ≈ 8% / month
  standard, // ~0.7% daily ≈ 24% / month
  aggressive, // ~1.5% daily ≈ 65% / month
}

extension GrowthRiskProfileExt on GrowthRiskProfile {
  double get dailyRate {
    switch (this) {
      case GrowthRiskProfile.conservative:
        return 0.003;
      case GrowthRiskProfile.standard:
        return 0.007;
      case GrowthRiskProfile.aggressive:
        return 0.015;
    }
  }

  String get label {
    switch (this) {
      case GrowthRiskProfile.conservative:
        return 'Conservative';
      case GrowthRiskProfile.standard:
        return 'Standard';
      case GrowthRiskProfile.aggressive:
        return 'Aggressive';
    }
  }

  /// Hint copy under the chip — sets expectations honestly.
  String get tagline {
    switch (this) {
      case GrowthRiskProfile.conservative:
        return '~0.3 %/day · 8 %/mo · slow + steady, prop-firm-safe';
      case GrowthRiskProfile.standard:
        return '~0.7 %/day · 24 %/mo · balanced risk vs growth';
      case GrowthRiskProfile.aggressive:
        return '~1.5 %/day · 65 %/mo · drawdown spikes more often';
    }
  }
}

final growthRiskProfileProvider =
    StateProvider<GrowthRiskProfile>((_) => GrowthRiskProfile.standard);

class GrowthModeCard extends ConsumerWidget {
  const GrowthModeCard({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final snapshot = ref.watch(accountSnapshotProvider).valueOrNull;
    final starting = ref.watch(growthStartingBalanceProvider);
    final target = ref.watch(growthTargetBalanceProvider);
    final profile = ref.watch(growthRiskProfileProvider);
    final currency = snapshot?.currency == 'EUR' ? '€' : (snapshot?.currency ?? r'$');

    final currentEquity = snapshot?.equity ?? starting;
    final multiplier = starting > 0 ? currentEquity / starting : 0.0;

    // Days-to-target projection. Compounding at the chosen daily
    // rate: target = current × (1 + rate)^d → d = log(target/current) / log(1+rate).
    int? etaDays;
    if (currentEquity > 0 && target > currentEquity && profile.dailyRate > 0) {
      etaDays = (math.log(target / currentEquity) / math.log(1 + profile.dailyRate))
          .ceil();
    }

    return SectionCard(
      title: 'Growth Mode · ML-driven small-account multiplier',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Headline math line — punchy single sentence.
          _HeadlineRow(
            currency: currency,
            starting: starting,
            currentEquity: currentEquity,
            multiplier: multiplier,
          ),
          const SizedBox(height: 12),
          // Inputs: starting balance + target balance.
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: _CurrencyField(
                  label: 'Started with',
                  currency: currency,
                  value: starting,
                  onChanged: (v) =>
                      ref.read(growthStartingBalanceProvider.notifier).state = v,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _CurrencyField(
                  label: 'Target',
                  currency: currency,
                  value: target,
                  onChanged: (v) =>
                      ref.read(growthTargetBalanceProvider.notifier).state = v,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          // Risk-profile chips — drives the ETA math.
          Row(
            children: [
              for (final p in GrowthRiskProfile.values)
                Padding(
                  padding: const EdgeInsets.only(right: 6),
                  child: _RiskChip(
                    label: p.label,
                    selected: p == profile,
                    onTap: () =>
                        ref.read(growthRiskProfileProvider.notifier).state = p,
                  ),
                ),
              const Spacer(),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            profile.tagline,
            style: const TextStyle(
              fontSize: 10,
              color: ForexAiTokens.textFaint,
            ),
          ),
          const SizedBox(height: 10),
          // Projection.
          _ProjectionLine(
            currency: currency,
            target: target,
            currentEquity: currentEquity,
            etaDays: etaDays,
          ),
          const SizedBox(height: 8),
          const Text(
            'Powered by NeoEthos Discovery (GA over 33-model ensemble) '
            '+ risk-aware Auto-Trader. Targets and pace are projections, '
            'not promises — real growth depends on regime + discipline.',
            style: TextStyle(
              fontSize: 10,
              color: ForexAiTokens.textFaint,
            ),
          ),
        ],
      ),
    );
  }
}

class _HeadlineRow extends StatelessWidget {
  final String currency;
  final double starting;
  final double currentEquity;
  final double multiplier;
  const _HeadlineRow({
    required this.currency,
    required this.starting,
    required this.currentEquity,
    required this.multiplier,
  });

  @override
  Widget build(BuildContext context) {
    final color = multiplier > 1.0
        ? ForexAiTokens.buy
        : (multiplier < 1.0 ? ForexAiTokens.sell : ForexAiTokens.textPrimary);
    return Wrap(
      spacing: 6,
      runSpacing: 4,
      crossAxisAlignment: WrapCrossAlignment.end,
      children: [
        Text(
          '$currency${_short(starting)}',
          style: const TextStyle(
            fontSize: 14,
            color: ForexAiTokens.textMuted,
          ),
        ),
        const Text('→', style: TextStyle(color: ForexAiTokens.textMuted)),
        Text(
          '$currency${_short(currentEquity)}',
          style: const TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.w800,
            color: ForexAiTokens.textPrimary,
          ),
        ),
        const SizedBox(width: 4),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.16),
            borderRadius: BorderRadius.circular(4),
          ),
          child: Text(
            '×${multiplier.toStringAsFixed(2)}',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: color,
            ),
          ),
        ),
      ],
    );
  }
}

class _CurrencyField extends StatefulWidget {
  final String label;
  final String currency;
  final double value;
  final ValueChanged<double> onChanged;
  const _CurrencyField({
    required this.label,
    required this.currency,
    required this.value,
    required this.onChanged,
  });

  @override
  State<_CurrencyField> createState() => _CurrencyFieldState();
}

class _CurrencyFieldState extends State<_CurrencyField> {
  late final TextEditingController _ctrl = TextEditingController(
    text: widget.value.toStringAsFixed(0),
  );

  @override
  void didUpdateWidget(covariant _CurrencyField old) {
    super.didUpdateWidget(old);
    if (old.value != widget.value &&
        double.tryParse(_ctrl.text.trim()) != widget.value) {
      _ctrl.text = widget.value.toStringAsFixed(0);
    }
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: _ctrl,
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
      inputFormatters: [
        FilteringTextInputFormatter.allow(RegExp(r'[0-9.,]')),
      ],
      decoration: InputDecoration(
        labelText: widget.label,
        prefixText: '${widget.currency} ',
        isDense: true,
        border: const OutlineInputBorder(),
      ),
      onSubmitted: (v) {
        final parsed = double.tryParse(v.replaceAll(',', ''));
        if (parsed != null && parsed > 0) widget.onChanged(parsed);
      },
      onChanged: (v) {
        final parsed = double.tryParse(v.replaceAll(',', ''));
        if (parsed != null && parsed > 0) widget.onChanged(parsed);
      },
    );
  }
}

class _RiskChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;
  const _RiskChip({
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
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: selected
              ? ForexAiTokens.accent.withValues(alpha: 0.18)
              : ForexAiTokens.surfaceBg,
          border: Border.all(
            color: selected ? ForexAiTokens.accent : ForexAiTokens.border,
          ),
          borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            color: selected ? ForexAiTokens.accent : ForexAiTokens.textPrimary,
          ),
        ),
      ),
    );
  }
}

class _ProjectionLine extends StatelessWidget {
  final String currency;
  final double target;
  final double currentEquity;
  final int? etaDays;
  const _ProjectionLine({
    required this.currency,
    required this.target,
    required this.currentEquity,
    required this.etaDays,
  });

  @override
  Widget build(BuildContext context) {
    String body;
    Color tone;
    if (target <= currentEquity) {
      body = 'Target reached — set a higher one to keep compounding.';
      tone = ForexAiTokens.buy;
    } else if (etaDays == null || etaDays! <= 0) {
      body = 'Pick a risk profile to see a projection.';
      tone = ForexAiTokens.textMuted;
    } else {
      final weeks = (etaDays! / 7).round();
      final months = (etaDays! / 30).round();
      final eta = etaDays! < 30
          ? '~$etaDays days'
          : (etaDays! < 180 ? '~$weeks weeks' : '~$months months');
      body =
          'At current pace, you reach $currency${_short(target)} in $eta (≈${etaDays!} days compounding).';
      tone = ForexAiTokens.textPrimary;
    }
    return Text(
      body,
      style: TextStyle(fontSize: 12, color: tone, fontWeight: FontWeight.w500),
    );
  }
}

/// Compact currency formatter for the headline row — no library
/// dependency, locale-agnostic. Keeps the digit count tight so the
/// "→ ×N.NN" tag stays on the same line at common window widths.
String _short(double v) {
  if (v >= 1000000) {
    return '${(v / 1000000).toStringAsFixed(2)}M';
  }
  if (v >= 1000) {
    return '${(v / 1000).toStringAsFixed(v >= 10000 ? 1 : 2)}k';
  }
  if (v < 1) return v.toStringAsFixed(4);
  return v.toStringAsFixed(2);
}
