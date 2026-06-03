// Growth Mode — the "small-account multiplier" panel.
//
// Growth Mode IS the Risky-Mode challenge surface (small bankroll →
// large target, autonomous compounding). The forward projection — how
// many days to target, and the chance of blowing the account — is NO
// LONGER a hardcoded UI heuristic: it is computed by the LIVE engine
// (`risky_mode.rs::time_to_target_scenarios`, exposed at
// `GET /risky/scenarios`). The UI invents no growth rates and no ruin
// numbers; it only renders what the engine returns.
//
// This is a SEPARATE mode from Prop-Firm-Passing (conservative,
// daily-loss-capped). There is deliberately NO "prop-firm-safe" option
// on this card — mixing the two is what the operator flagged. The Risky
// band is 30%–50% per-trade by design and the ruin gauge shows the
// engine's honest (high) estimate.
//
// The starting/target inputs are local what-if state for now; persisting
// them to the single config (UI/TUI-editable) folds into the broader
// config-consolidation work.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../api/currency_format.dart';
import '../../l10n/app_localizations.dart';
import '../../state/account_provider.dart';
import '../../state/system_providers.dart';
import '../../theme/theme.dart';
import '../../widgets/risky_mode_cooldown_chip.dart';
import '../_placeholder.dart';

/// User-set "where I started" balance — anchors the multiplier.
/// Defaults to €100 because that's the operator's stated starter
/// scenario ("from 100 euros to thousands").
final growthStartingBalanceProvider = StateProvider<double>((_) => 100.0);

/// User-set target balance for the projection line.
final growthTargetBalanceProvider = StateProvider<double>((_) => 10000.0);

/// Per-trade aggression for the Risky/Growth projection. Maps to the
/// engine's signed Risky band (`risky_mode.rs` MIN 0.30 / MAX 0.50) —
/// these are NOT arbitrary daily-growth guesses; the projection itself
/// is computed server-side. Risky/Growth Mode is high-risk BY DESIGN —
/// there is no "prop-firm-safe" option here (that belongs to the
/// separate, mutually-exclusive Prop-Firm-Passing mode).
enum GrowthAggression { steady, balanced, aggressive }

extension GrowthAggressionExt on GrowthAggression {
  /// Per-trade risk fraction sent to `/risky/scenarios` (the server
  /// clamps to the engine band). steady = band min, aggressive = band max.
  double get riskFraction {
    switch (this) {
      case GrowthAggression.steady:
        return 0.30;
      case GrowthAggression.balanced:
        return 0.40;
      case GrowthAggression.aggressive:
        return 0.50;
    }
  }

  /// Localized short label for the aggression chip.
  String label(AppLocalizations l10n) {
    switch (this) {
      case GrowthAggression.steady:
        return l10n.growthCardAggSteady;
      case GrowthAggression.balanced:
        return l10n.growthCardAggBalanced;
      case GrowthAggression.aggressive:
        return l10n.growthCardAggAggressive;
    }
  }

  /// Localized honest copy — Risky/Growth Mode is high-risk throughout the band.
  String tagline(AppLocalizations l10n) {
    switch (this) {
      case GrowthAggression.steady:
        return l10n.growthCardTaglineSteady;
      case GrowthAggression.balanced:
        return l10n.growthCardTaglineBalanced;
      case GrowthAggression.aggressive:
        return l10n.growthCardTaglineAggressive;
    }
  }
}

final growthAggressionProvider =
    StateProvider<GrowthAggression>((_) => GrowthAggression.balanced);

class GrowthModeCard extends ConsumerWidget {
  const GrowthModeCard({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final snapshot = ref.watch(accountSnapshotProvider).valueOrNull;
    final starting = ref.watch(growthStartingBalanceProvider);
    final target = ref.watch(growthTargetBalanceProvider);
    final aggression = ref.watch(growthAggressionProvider);
    final currency = currencyGlyph(snapshot?.currency ?? 'EUR');

    final currentEquity = snapshot?.equity ?? starting;
    final multiplier = starting > 0 ? currentEquity / starting : 0.0;

    // Risky-Mode 24h re-arm cooldown remaining (if any) as a persistent
    // chip at the top. `riskyModeCooldownRemainingSecs` is null when no
    // cooldown is active; the chip renders nothing in that case.
    final cooldownAsync = ref.watch(riskProvider);
    final cooldownSecs = cooldownAsync.maybeWhen(
      data: (snap) => snap.riskyModeCooldownRemainingSecs,
      orElse: () => null,
    );

    return SectionCard(
      title: l10n.growthCardTitle,
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
          // Headline math line — punchy single sentence (live equity).
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
                  label: l10n.growthCardStartedWith,
                  currency: currency,
                  value: starting,
                  onChanged: (v) =>
                      ref.read(growthStartingBalanceProvider.notifier).state = v,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _CurrencyField(
                  label: l10n.growthCardTarget,
                  currency: currency,
                  value: target,
                  onChanged: (v) =>
                      ref.read(growthTargetBalanceProvider.notifier).state = v,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          // Aggression chips — drive the per-trade risk fraction sent to
          // the engine's projection endpoint.
          Row(
            children: [
              for (final a in GrowthAggression.values)
                Padding(
                  padding: const EdgeInsets.only(right: 6),
                  child: _RiskChip(
                    label: a.label(l10n),
                    selected: a == aggression,
                    onTap: () =>
                        ref.read(growthAggressionProvider.notifier).state = a,
                  ),
                ),
              const Spacer(),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            aggression.tagline(l10n),
            style: const TextStyle(
              fontSize: 10,
              color: NeoethosTokens.textFaint,
            ),
          ),
          const SizedBox(height: 12),
          // Projection — computed by the engine, not the UI.
          if (target <= currentEquity)
            Text(
              l10n.growthCardTargetReached,
              style: const TextStyle(
                fontSize: 12,
                color: NeoethosTokens.buy,
                fontWeight: FontWeight.w500,
              ),
            )
          else if (currentEquity <= 0)
            Text(
              l10n.growthCardSetStarting,
              style: const TextStyle(
                  fontSize: 12, color: NeoethosTokens.textMuted),
            )
          else
            _ScenarioSection(
              currentEquity: currentEquity,
              target: target,
              riskFraction: aggression.riskFraction,
            ),
          const SizedBox(height: 8),
          Text(
            l10n.growthCardFooter,
            style: const TextStyle(
              fontSize: 10,
              color: NeoethosTokens.textFaint,
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
        ? NeoethosTokens.buy
        : (multiplier < 1.0 ? NeoethosTokens.sell : NeoethosTokens.textPrimary);
    return Wrap(
      spacing: 6,
      runSpacing: 4,
      crossAxisAlignment: WrapCrossAlignment.end,
      children: [
        Text(
          '$currency${_short(starting)}',
          style: const TextStyle(
            fontSize: 14,
            color: NeoethosTokens.textMuted,
          ),
        ),
        const Text('→', style: TextStyle(color: NeoethosTokens.textMuted)),
        Text(
          '$currency${_short(currentEquity)}',
          style: const TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.w800,
            color: NeoethosTokens.textPrimary,
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
      borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: selected
              ? NeoethosTokens.accent.withValues(alpha: 0.18)
              : NeoethosTokens.surfaceBg,
          border: Border.all(
            color: selected ? NeoethosTokens.accent : NeoethosTokens.border,
          ),
          borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            color: selected ? NeoethosTokens.accent : NeoethosTokens.textPrimary,
          ),
        ),
      ),
    );
  }
}

/// Compact currency formatter for the headline row — no library
/// dependency, locale-agnostic.
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
// TimeToTarget scenarios — now fed by the LIVE engine (`/risky/scenarios`).
//
// Hero ruin gauge + 3-card strip (Lucky / Typical / Slow). Every number
// here comes from `risky_mode.rs::time_to_target_scenarios`; the UI
// computes nothing.

class _ScenarioSection extends ConsumerWidget {
  final double currentEquity;
  final double target;
  final double riskFraction;
  const _ScenarioSection({
    required this.currentEquity,
    required this.target,
    required this.riskFraction,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final async = ref.watch(riskyScenariosProvider((
      startingUsd: currentEquity,
      targetUsd: target,
      riskFraction: riskFraction,
    )));
    return async.when(
      loading: () => const Padding(
        padding: EdgeInsets.symmetric(vertical: 14),
        child: Center(
          child: SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
        ),
      ),
      error: (_, __) => Text(
        l10n.growthCardProjectionUnavailable,
        style: const TextStyle(
          fontSize: 11,
          color: NeoethosTokens.textFaint,
          fontStyle: FontStyle.italic,
        ),
      ),
      data: (s) => Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _RuinGauge(probability: s.ruinProbability),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _ScenarioCard(
                  label: l10n.growthCardScenLucky,
                  subtitle: l10n.growthCardScenLuckySub,
                  time: _formatDays(l10n, s.bestCaseDays),
                  color: const Color(0xFF2E7D32),
                  icon: Icons.rocket_launch_outlined,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _ScenarioCard(
                  label: l10n.growthCardScenTypical,
                  subtitle: l10n.growthCardScenTypicalSub,
                  time: _formatDays(l10n, s.expectedDays),
                  color: const Color(0xFF1565C0),
                  icon: Icons.timeline,
                  emphasized: true,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _ScenarioCard(
                  label: l10n.growthCardScenSlow,
                  subtitle: l10n.growthCardScenSlowSub,
                  time: _formatDays(l10n, s.conservativeDays),
                  color: const Color(0xFFB28704),
                  icon: Icons.hourglass_bottom_outlined,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            l10n.growthCardComputedBy(
              (s.riskFraction * 100).round(),
              (s.winRate * 100).round(),
              s.rewardToRisk.toStringAsFixed(1),
              s.tradesPerDay.round(),
            ),
            style: const TextStyle(
              fontSize: 10,
              color: NeoethosTokens.textFaint,
            ),
          ),
        ],
      ),
    );
  }
}

/// Auto-format engine days into the most readable unit. `null` = the
/// configured edge can't reach target on average (non-positive growth).
String _formatDays(AppLocalizations l10n, int? days) {
  if (days == null) return l10n.growthCardDaysNotReachable;
  if (days <= 0) return l10n.growthCardDaysNow;
  if (days <= 60) return l10n.growthCardDaysDays(days);
  if (days <= 730) {
    final months = (days / 30.4).round();
    return l10n.growthCardDaysMonths(months);
  }
  final years = (days / 365.25).toStringAsFixed(1);
  return l10n.growthCardDaysYears(years);
}

class _RuinGauge extends StatelessWidget {
  final double probability;
  const _RuinGauge({required this.probability});

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final pct = (probability * 100).round();
    final inN = probability > 0 ? (1 / probability).round() : 1000;

    final Color color;
    final String tier;
    if (pct < 5) {
      color = const Color(0xFF2E7D32);
      tier = l10n.growthCardRuinTierLow;
    } else if (pct < 25) {
      color = const Color(0xFFE65100);
      tier = l10n.growthCardRuinTierMeaningful;
    } else {
      color = const Color(0xFFB71C1C);
      tier = l10n.growthCardRuinTierHigh;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
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
                    Text(
                      l10n.growthCardRuinChance,
                      style: const TextStyle(
                        fontSize: 12,
                        color: NeoethosTokens.textMuted,
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
                  l10n.growthCardRuinDetail(inN, tier),
                  style: const TextStyle(
                    fontSize: 11,
                    color: NeoethosTokens.textFaint,
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
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
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
              color: NeoethosTokens.textPrimary,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            subtitle,
            style: const TextStyle(
              fontSize: 10,
              color: NeoethosTokens.textFaint,
            ),
          ),
        ],
      ),
    );
  }
}
