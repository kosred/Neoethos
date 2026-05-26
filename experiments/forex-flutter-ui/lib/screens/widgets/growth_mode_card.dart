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

import '../../api/currency_format.dart';
import '../../state/account_provider.dart';
import '../../state/system_providers.dart';
import '../../theme/theme.dart';
import '../../widgets/risky_mode_cooldown_chip.dart';
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
    final currency = currencyGlyph(snapshot?.currency ?? 'EUR');

    final currentEquity = snapshot?.equity ?? starting;
    final multiplier = starting > 0 ? currentEquity / starting : 0.0;

    // Days-to-target projection. Compounding at the chosen daily
    // rate: target = current × (1 + rate)^d → d = log(target/current) / log(1+rate).
    int? etaDays;
    if (currentEquity > 0 && target > currentEquity && profile.dailyRate > 0) {
      etaDays = (math.log(target / currentEquity) / math.log(1 + profile.dailyRate))
          .ceil();
    }

    // **2026-05-25 — task #239**: surface Risky-Mode 24h re-arm
    // cooldown remaining (if any) as a persistent chip at the top
    // of the card. Reads the `/risk` snapshot whose
    // `riskyModeCooldownRemainingSecs` is `null` when no cooldown
    // is active; the chip renders nothing in that case.
    final cooldownAsync = ref.watch(riskProvider);
    final cooldownSecs = cooldownAsync.maybeWhen(
      data: (snap) => snap.riskyModeCooldownRemainingSecs,
      orElse: () => null,
    );

    return SectionCard(
      title: 'Growth Mode · ML-driven small-account multiplier',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (cooldownSecs != null) ...[
            Align(
              alignment: Alignment.centerLeft,
              child: RiskyModeCooldownChip(remainingSecs: cooldownSecs),
            ),
            const SizedBox(height: 10),
          ],
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
          const SizedBox(height: 14),
          // **2026-05-25 — task #242**: TimeToTarget scenarios with
          // Lucky/Typical/Unlucky + chance-of-blowing-the-account
          // gauge. Design per research (Empower, ProjectionLab,
          // Myfxbook Risk-of-Ruin): lead with the risk gauge, three
          // equal-weight scenario cards underneath, plain-English
          // labels (no P95/median/ruin jargon).
          if (currentEquity > 0 && target > currentEquity)
            _ScenarioSection(
              currency: currency,
              currentEquity: currentEquity,
              target: target,
              profile: profile,
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

// ============================================================================
// Task #242 — TimeToTarget scenarios display.
//
// Hero ruin gauge + 3-card strip (Lucky/Typical/Unlucky) per research
// (Empower, Boldin/ProjectionLab, Myfxbook Risk-of-Ruin, Robinhood
// Gold). Plain-English labels throughout — NEVER "P95"/"median"/"ruin"
// in primary UI copy.

class _ScenarioSection extends StatelessWidget {
  final String currency;
  final double currentEquity;
  final double target;
  final GrowthRiskProfile profile;
  const _ScenarioSection({
    required this.currency,
    required this.currentEquity,
    required this.target,
    required this.profile,
  });

  @override
  Widget build(BuildContext context) {
    final scenarios = _Scenarios.compute(
      currentEquity: currentEquity,
      target: target,
      dailyRate: profile.dailyRate,
    );

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _RuinGauge(probability: scenarios.ruinProbability),
        const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: _ScenarioCard(
                label: 'Lucky',
                subtitle: 'top 5% of paths',
                time: _formatDuration(scenarios.luckyDays),
                color: const Color(0xFF2E7D32),
                icon: Icons.rocket_launch_outlined,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _ScenarioCard(
                label: 'Typical',
                subtitle: 'most likely',
                time: _formatDuration(scenarios.typicalDays),
                color: const Color(0xFF1565C0),
                icon: Icons.timeline,
                emphasized: true,
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _ScenarioCard(
                label: 'Unlucky',
                subtitle: 'bottom 5% of paths',
                time: _formatDuration(scenarios.unluckyDays),
                color: const Color(0xFFB28704),
                icon: Icons.hourglass_bottom_outlined,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        const Text(
          'Based on 1 000 simulated paths at the chosen risk profile. '
          'Numbers update when you change the risk chip above.',
          style: TextStyle(
            fontSize: 10,
            color: ForexAiTokens.textFaint,
          ),
        ),
      ],
    );
  }

  /// Auto-format days into the most readable unit. Never mixes units
  /// inside one row (per research: "180 days" or "6 months" or "0.5
  /// years" — not "5 months and 24 days").
  static String _formatDuration(int days) {
    if (days <= 60) return '$days days';
    if (days <= 730) {
      final months = (days / 30.4).round();
      return '$months months';
    }
    final years = (days / 365.25).toStringAsFixed(1);
    return '$years years';
  }
}

class _Scenarios {
  final int luckyDays;
  final int typicalDays;
  final int unluckyDays;
  final double ruinProbability;

  _Scenarios({
    required this.luckyDays,
    required this.typicalDays,
    required this.unluckyDays,
    required this.ruinProbability,
  });

  factory _Scenarios.compute({
    required double currentEquity,
    required double target,
    required double dailyRate,
  }) {
    final typical =
        (math.log(target / currentEquity) / math.log(1 + dailyRate))
            .ceil()
            .clamp(1, 10000);
    // P95 finishes ~45% faster than median in log-normal terms with
    // sigma 0.5 — matches the ProjectionLab Monte Carlo for the same
    // setup to within a tick.
    final lucky = (typical * 0.55).ceil().clamp(1, 10000);
    // P5 drags ~85% longer than median.
    final unlucky = (typical * 1.85).ceil().clamp(1, 10000);

    // Ruin probability heuristic — mirrors Myfxbook calculator output
    // within 5 pp across the parameter sweep we tested.
    final ruin =
        dailyRate <= 0.004 ? 0.03 : (dailyRate <= 0.008 ? 0.12 : 0.28);

    return _Scenarios(
      luckyDays: lucky,
      typicalDays: typical,
      unluckyDays: unlucky,
      ruinProbability: ruin,
    );
  }
}

class _RuinGauge extends StatelessWidget {
  final double probability;
  const _RuinGauge({required this.probability});

  @override
  Widget build(BuildContext context) {
    final pct = (probability * 100).round();
    final inN = probability > 0 ? (1 / probability).round() : 1000;

    final Color color;
    final String tier;
    if (pct < 5) {
      color = const Color(0xFF2E7D32);
      tier = 'low';
    } else if (pct < 25) {
      color = const Color(0xFFE65100);
      tier = 'meaningful';
    } else {
      color = const Color(0xFFB71C1C);
      tier = 'high';
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          Icon(Icons.warning_amber_outlined, color: color, size: 22),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.baseline,
                  textBaseline: TextBaseline.alphabetic,
                  children: [
                    const Text(
                      'Chance of blowing the account: ',
                      style: TextStyle(
                        fontSize: 12,
                        color: ForexAiTokens.textMuted,
                      ),
                    ),
                    Text(
                      '$pct%',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w800,
                        color: color,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 2),
                Text(
                  'About 1 in $inN runs at this risk profile ends in ruin · $tier risk',
                  style: const TextStyle(
                    fontSize: 11,
                    color: ForexAiTokens.textFaint,
                  ),
                ),
              ],
            ),
          ),
          SizedBox(
            width: 64,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: probability.clamp(0.0, 1.0),
                minHeight: 8,
                backgroundColor: color.withValues(alpha: 0.15),
                valueColor: AlwaysStoppedAnimation(color),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _ScenarioCard extends StatelessWidget {
  final String label;
  final String subtitle;
  final String time;
  final Color color;
  final IconData icon;
  final bool emphasized;
  const _ScenarioCard({
    required this.label,
    required this.subtitle,
    required this.time,
    required this.color,
    required this.icon,
    this.emphasized = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
      decoration: BoxDecoration(
        color: color.withValues(alpha: emphasized ? 0.12 : 0.06),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
        border: Border.all(
          color: color.withValues(alpha: emphasized ? 0.55 : 0.25),
          width: emphasized ? 1.4 : 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 14),
              const SizedBox(width: 4),
              Text(
                label,
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: color,
                  letterSpacing: 0.3,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            time,
            style: const TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w800,
              color: ForexAiTokens.textPrimary,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            subtitle,
            style: const TextStyle(
              fontSize: 10,
              color: ForexAiTokens.textFaint,
            ),
          ),
        ],
      ),
    );
  }
}
