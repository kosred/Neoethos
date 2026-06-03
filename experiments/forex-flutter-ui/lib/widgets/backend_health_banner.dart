// BackendHealthBanner — red strip across the top of AppShell when
// the Rust backend isn't responding to `/healthz`.
//
// State source: `backendHealthProvider` (see
// startup/backend_watchdog.dart). The banner ONLY renders when
// `state.isDegraded` is true — healthy steady state is
// `SizedBox.shrink` so there's zero vertical cost.
//
// Layout: matches the `pending_actions_banner.dart` convention so
// the two stack predictably when both are active (rare but
// possible — e.g. the operator confirmed an action just as the
// backend crashed; the action is now pending against a dead
// backend and both banners render in sequence).

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../startup/backend_watchdog.dart';
import '../theme/theme.dart';
import 'backend_diagnostics_dialog.dart';

class BackendHealthBanner extends ConsumerWidget {
  const BackendHealthBanner({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final health = ref.watch(backendHealthProvider);
    if (!health.isDegraded) return const SizedBox.shrink();

    final attempts = health.respawnAttempts;
    final attemptSuffix = attempts == 0
        ? ''
        : attempts == 1
            ? ' (restart #1 in progress)'
            : ' (restart #$attempts in progress)';

    return Material(
      color: NeoethosTokens.sell.withValues(alpha: 0.16),
      child: InkWell(
        onTap: () => showBackendDiagnosticsDialog(context),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(
            horizontal: NeoethosTokens.spLg,
            vertical: NeoethosTokens.spSm,
          ),
          decoration: const BoxDecoration(
            border: Border(
              bottom: BorderSide(color: NeoethosTokens.sell, width: 1),
            ),
          ),
          child: Row(
            children: [
              const SizedBox(
                width: 14,
                height: 14,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor:
                      AlwaysStoppedAnimation<Color>(NeoethosTokens.sell),
                ),
              ),
              const SizedBox(width: NeoethosTokens.spSm),
              Expanded(
                child: Text(
                  'Backend reconnecting…$attemptSuffix',
                  style: const TextStyle(
                    fontSize: NeoethosTokens.fsBody,
                    fontWeight: FontWeight.w700,
                    color: NeoethosTokens.sell,
                  ),
                ),
              ),
              const Text(
                'Click for diagnostics',
                style: TextStyle(
                  fontSize: NeoethosTokens.fsCaption,
                  color: NeoethosTokens.textMuted,
                ),
              ),
              const SizedBox(width: NeoethosTokens.spXs),
              const Icon(
                Icons.chevron_right,
                size: 16,
                color: NeoethosTokens.textMuted,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
