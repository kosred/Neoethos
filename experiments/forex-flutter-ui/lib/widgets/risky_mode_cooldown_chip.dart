// Risky Mode 24h re-arm cooldown countdown chip + lockout modal.
//
// **2026-05-25 — task #239**: surfaces the
// `riskyModeCooldownRemainingSecs` field on `/risk` so the operator
// sees exactly when Risky Mode becomes re-armable after a
// kill-switch trip.
//
// Research-derived design (cross-checked with Robinhood PDT
// lockouts, Discord cooldowns, Binance liquidation countdowns):
//   - Persistent chip with relative + absolute time:
//       "Re-arm in 23h 47m"   (primary)
//       "Available at 16:23 tomorrow (local)"   (secondary)
//   - Traffic-light progression:
//       red while >1h remains
//       amber inside the final hour
//       green pulse at zero
//   - Tick every 60s globally (cheap on stationary widget),
//     per-second only inside the final 60s (the only time the
//     operator looks at it intently).
//   - Blocking modal if operator clicks "Arm Risky Mode" while
//     locked. Modal copy names the trigger, the unlock time, and
//     the policy ("This 24h cooldown is enforced and cannot be
//     overridden"). NO override link — the whole point of a
//     cooldown is non-negotiable.
//
// Anti-patterns avoided:
//   - "1d 0h" precision (looks frozen for an hour at a time).
//   - Absolute-only time without "in X" (forces mental arithmetic).
//   - Per-second ticking for 24h (battery waste).
//   - Override/contact-support link (invites operator to argue).

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../l10n/app_localizations.dart';
import '../theme/theme.dart';

/// Compact chip showing the remaining cooldown. Drop it next to the
/// Risky Mode arm button (Dashboard's GrowthModeCard, Risk screen).
class RiskyModeCooldownChip extends StatefulWidget {
  /// Seconds remaining as reported by `/risk`. `null` = no cooldown
  /// active; the widget renders nothing.
  final int? remainingSecs;

  const RiskyModeCooldownChip({super.key, required this.remainingSecs});

  @override
  State<RiskyModeCooldownChip> createState() => _RiskyModeCooldownChipState();
}

class _RiskyModeCooldownChipState extends State<RiskyModeCooldownChip> {
  Timer? _timer;
  int? _localCountdown;

  @override
  void initState() {
    super.initState();
    _localCountdown = widget.remainingSecs;
    _startTimer();
  }

  @override
  void didUpdateWidget(covariant RiskyModeCooldownChip oldWidget) {
    super.didUpdateWidget(oldWidget);
    // When the backend reports a fresh value (e.g. a new kill-switch
    // trip restarted the 24h timer), re-anchor locally and resume
    // ticking from there.
    if (widget.remainingSecs != oldWidget.remainingSecs) {
      _localCountdown = widget.remainingSecs;
      _restartTimer();
    }
  }

  void _startTimer() {
    final remaining = _localCountdown;
    if (remaining == null || remaining <= 0) return;

    // Per-second ticks in the final minute; per-minute ticks otherwise.
    // Per the research: per-second ticking for a 24h timer wastes CPU
    // and battery on a stationary widget. The operator only cares
    // about per-second precision when it's about to fire.
    final tickInterval =
        remaining <= 60 ? const Duration(seconds: 1) : const Duration(seconds: 60);
    _timer = Timer.periodic(tickInterval, (_) {
      setState(() {
        final next = (_localCountdown ?? 0) - tickInterval.inSeconds;
        _localCountdown = next > 0 ? next : 0;
      });
      // Switch to per-second ticking when we cross the 60s boundary.
      if (tickInterval.inSeconds > 1 && (_localCountdown ?? 0) <= 60) {
        _restartTimer();
      }
      // Stop ticking once we hit zero — the cooldown is over and the
      // chip flips to green. The next /risk refresh will set
      // remainingSecs = null which dismisses the chip entirely.
      if ((_localCountdown ?? 0) <= 0) {
        _timer?.cancel();
        _timer = null;
      }
    });
  }

  void _restartTimer() {
    _timer?.cancel();
    _timer = null;
    _startTimer();
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final secs = _localCountdown;
    if (secs == null || secs < 0) return const SizedBox.shrink();

    // Traffic-light progression. Threshold rationale:
    //   red    > 1h    → "you have a long wait, don't keep checking"
    //   amber  ≤ 1h    → "almost there, you can plan around this"
    //   green  = 0     → "armable now" (brief, until /risk refreshes)
    final Color background;
    final Color text;
    if (secs <= 0) {
      background = const Color(0xFF1B5E20); // green
      text = Colors.white;
    } else if (secs < 3600) {
      background = const Color(0xFFE65100); // amber
      text = Colors.white;
    } else {
      background = const Color(0xFFB71C1C); // red
      text = Colors.white;
    }

    return Tooltip(
      message: _modalCopy(l10n, secs),
      preferBelow: false,
      child: InkWell(
        onTap: () => _showLockoutModal(context, secs),
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: background,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                secs <= 0 ? Icons.lock_open : Icons.lock_clock,
                color: text,
                size: 14,
              ),
              const SizedBox(width: 6),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    secs <= 0
                        ? l10n.riskyChipReArmAvailable
                        : l10n.riskyChipReArmIn(_formatRelative(secs)),
                    style: TextStyle(
                      color: text,
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 0.2,
                    ),
                  ),
                  if (secs > 0)
                    Text(
                      l10n.riskyChipAvailableAt(_formatAbsolute(l10n, secs)),
                      style: TextStyle(
                        color: text.withValues(alpha: 0.8),
                        fontSize: 10,
                      ),
                    ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  /// "23h 47m" / "47m" / "23s" — auto-precision based on magnitude.
  /// Never shows "1d 0h" (that looks frozen for an hour at a time).
  static String _formatRelative(int secs) {
    if (secs < 60) return '${secs}s';
    if (secs < 3600) return '${(secs ~/ 60)}m';
    final hours = secs ~/ 3600;
    final mins = (secs % 3600) ~/ 60;
    return mins > 0 ? '${hours}h ${mins}m' : '${hours}h';
  }

  /// "16:23 tomorrow" / "16:23 today" / "16:23 on May 27".
  /// Always operator-local time — never UTC for a user-facing
  /// countdown.
  static String _formatAbsolute(AppLocalizations l10n, int secs) {
    final eta = DateTime.now().add(Duration(seconds: secs));
    final now = DateTime.now();
    final formatter = DateFormat('HH:mm');
    final clock = formatter.format(eta);
    final today = DateTime(now.year, now.month, now.day);
    final tomorrow = today.add(const Duration(days: 1));
    final etaDay = DateTime(eta.year, eta.month, eta.day);
    if (etaDay == today) return l10n.riskyChipToday(clock);
    if (etaDay == tomorrow) return l10n.riskyChipTomorrow(clock);
    final dateFormatter = DateFormat('MMM d');
    return l10n.riskyChipOnDate(clock, dateFormatter.format(eta));
  }

  static String _modalCopy(AppLocalizations l10n, int secs) {
    return l10n.riskyChipModalCopy(_formatAbsolute(l10n, secs));
  }

  void _showLockoutModal(BuildContext context, int secs) {
    final l10n = AppLocalizations.of(context)!;
    showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Row(
          children: [
            const Icon(Icons.lock_clock, color: Color(0xFFB71C1C)),
            const SizedBox(width: 8),
            Text(l10n.riskyChipModalTitle),
          ],
        ),
        content: SizedBox(
          width: 480,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(_modalCopy(l10n, secs)),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFFB71C1C).withValues(alpha: 0.08),
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: const Color(0xFFB71C1C).withValues(alpha: 0.3),
                  ),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      l10n.riskyChipWhyCooldown,
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      l10n.riskyChipWhyBody,
                      style: const TextStyle(
                        color: NeoethosTokens.textMuted,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: Text(l10n.riskyChipOk),
          ),
        ],
      ),
    );
  }
}
