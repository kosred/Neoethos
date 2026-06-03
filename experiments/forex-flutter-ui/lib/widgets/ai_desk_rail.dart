// AI Desk right-rail — persistent right column on the trading-focused
// screens (Market Watch, Strategy Lab, Positions).
//
// **F-322 (2026-05-29 rebuild)**: signature feature of the Codex
// mockup. The full-screen AI Desk tab is the rich version; this rail
// is its condensed always-visible mirror so the operator can see at a
// glance:
//   - how many models are loaded (and when last trained)
//   - last walk-forward accuracy
//   - any pending LLM-proposed action awaiting Confirm / Reject
//   - the symbols the engine has trained on (discovery targets)
//
// Width is fixed at 280 px when expanded and collapses to a 36 px
// strip with a chevron button when toggled off (via the
// `aiDeskRailVisibleProvider` state). The collapsed strip preserves a
// tiny "pending action" badge so the operator never misses a fresh
// proposal even when they hid the rail to focus on the chart.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
import '../state/nav.dart';
import '../state/pending_actions_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import 'news_panel.dart';

/// Persists "is the rail expanded" across rebuilds within the session.
/// Defaults to expanded — the rail is the whole point of the rebuild
/// so it should be on by default on first launch.
final aiDeskRailVisibleProvider = StateProvider<bool>((ref) => true);

class AiDeskRail extends ConsumerWidget {
  const AiDeskRail({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final expanded = ref.watch(aiDeskRailVisibleProvider);
    if (!expanded) return const _CollapsedStrip();
    return Container(
      width: 280,
      decoration: const BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border(left: BorderSide(color: NeoethosTokens.border)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _Header(
            onCollapse: () =>
                ref.read(aiDeskRailVisibleProvider.notifier).state = false,
            onOpenFull: () =>
                ref.read(activeTabProvider.notifier).state = 'AiDesk',
          ),
          const Divider(height: 1, color: NeoethosTokens.border),
          const Expanded(
            child: SingleChildScrollView(
              padding: EdgeInsets.symmetric(
                horizontal: NeoethosTokens.spMd,
                vertical: NeoethosTokens.spMd,
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  NewsPanel(),
                  SizedBox(height: NeoethosTokens.spMd),
                  _ModelsLoadedSection(),
                  SizedBox(height: NeoethosTokens.spMd),
                  _PerformanceSection(),
                  SizedBox(height: NeoethosTokens.spMd),
                  _ProposedActionSection(),
                  SizedBox(height: NeoethosTokens.spMd),
                  _DiscoveryTargetsSection(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _CollapsedStrip extends ConsumerWidget {
  const _CollapsedStrip();
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final pendingCount = ref
            .watch(pendingActionsProvider)
            .valueOrNull
            ?.where((a) => a.status == 'pending')
            .length ??
        0;
    return Container(
      width: 36,
      decoration: const BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border(left: BorderSide(color: NeoethosTokens.border)),
      ),
      child: Column(
        children: [
          IconButton(
            tooltip: 'Expand AI Desk',
            iconSize: 18,
            onPressed: () =>
                ref.read(aiDeskRailVisibleProvider.notifier).state = true,
            icon: const Icon(Icons.chevron_left,
                color: NeoethosTokens.textMuted),
          ),
          const SizedBox(height: 4),
          RotatedBox(
            quarterTurns: 3,
            child: Text(
              'AI DESK',
              style: TextStyle(
                fontSize: NeoethosTokens.fsCaption - 1,
                fontWeight: FontWeight.w800,
                letterSpacing: 2,
                color: NeoethosTokens.textFaint.withValues(alpha: 0.8),
              ),
            ),
          ),
          if (pendingCount > 0) ...[
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.symmetric(
                horizontal: 4,
                vertical: 2,
              ),
              decoration: BoxDecoration(
                color: NeoethosTokens.warning.withValues(alpha: 0.18),
                border: Border.all(
                  color: NeoethosTokens.warning.withValues(alpha: 0.6),
                ),
                borderRadius: BorderRadius.circular(3),
              ),
              child: Text(
                '$pendingCount',
                style: const TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                  color: NeoethosTokens.warning,
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

class _Header extends StatelessWidget {
  final VoidCallback onCollapse;
  final VoidCallback onOpenFull;
  const _Header({required this.onCollapse, required this.onOpenFull});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(
        horizontal: NeoethosTokens.spMd,
        vertical: 6,
      ),
      child: Row(
        children: [
          Container(
            width: 22,
            height: 22,
            decoration: BoxDecoration(
              color: NeoethosTokens.accent.withValues(alpha: 0.18),
              border: Border.all(
                color: NeoethosTokens.accent.withValues(alpha: 0.55),
              ),
              borderRadius: BorderRadius.circular(5),
            ),
            alignment: Alignment.center,
            child: const Text(
              '✺',
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w700,
                color: NeoethosTokens.accent,
              ),
            ),
          ),
          const SizedBox(width: 8),
          const Expanded(
            child: Text(
              'AI Desk',
              style: TextStyle(
                fontSize: NeoethosTokens.fsBody + 1,
                fontWeight: FontWeight.w700,
                color: NeoethosTokens.textPrimary,
              ),
            ),
          ),
          IconButton(
            tooltip: 'Open full AI Desk',
            iconSize: 16,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
            onPressed: onOpenFull,
            icon: const Icon(Icons.open_in_full,
                color: NeoethosTokens.textMuted),
          ),
          IconButton(
            tooltip: 'Collapse',
            iconSize: 18,
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(minWidth: 28, minHeight: 28),
            onPressed: onCollapse,
            icon: const Icon(Icons.chevron_right,
                color: NeoethosTokens.textMuted),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Section primitives
// ---------------------------------------------------------------------------

class _SectionCard extends StatelessWidget {
  final String title;
  final Widget body;
  final String? subtitle;
  const _SectionCard({
    required this.title,
    required this.body,
    this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(NeoethosTokens.spMd),
      decoration: BoxDecoration(
        color: NeoethosTokens.appBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: NeoethosTokens.fsCaption,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.6,
                    color: NeoethosTokens.textMuted,
                  ),
                ),
              ),
              if (subtitle != null)
                Text(
                  subtitle!,
                  style: const TextStyle(
                    fontSize: NeoethosTokens.fsCaption,
                    color: NeoethosTokens.textFaint,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 8),
          body,
        ],
      ),
    );
  }
}

class _KvRow extends StatelessWidget {
  final String label;
  final String value;
  final Color? valueColor;
  const _KvRow(this.label, this.value, {this.valueColor});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          Expanded(
            child: Text(
              label,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsCaption,
                color: NeoethosTokens.textMuted,
              ),
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontSize: NeoethosTokens.fsCaption,
              fontWeight: FontWeight.w700,
              color: valueColor ?? NeoethosTokens.textPrimary,
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Models Loaded
// ---------------------------------------------------------------------------

class _ModelsLoadedSection extends ConsumerWidget {
  const _ModelsLoadedSection();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(intelligenceProvider);
    return _SectionCard(
      title: 'MODELS LOADED',
      subtitle: async.maybeWhen(
        data: (snap) => '${snap.artifactCount}',
        orElse: () => '—',
      ),
      body: async.when(
        loading: () => const _Skeleton(lines: 2),
        error: (_, __) => const _ErrorLine(),
        data: (snap) {
          if (snap.artifactCount == 0) {
            return const Text(
              'No models trained yet — run Strategy Lab → Training.',
              style: TextStyle(
                fontSize: NeoethosTokens.fsCaption,
                color: NeoethosTokens.textFaint,
                height: 1.4,
              ),
            );
          }
          // Show first 4 artifact names — the full list lives in the AI
          // Desk full-screen tab.
          final sample = snap.artifacts.take(4).toList();
          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              for (final name in sample)
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2),
                  child: Row(
                    children: [
                      const Icon(
                        Icons.circle,
                        size: 6,
                        color: NeoethosTokens.buy,
                      ),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          name,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: NeoethosTokens.fsCaption,
                            color: NeoethosTokens.textPrimary,
                            fontFeatures: [FontFeature.tabularFigures()],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              if (snap.artifactCount > sample.length)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(
                    '+${snap.artifactCount - sample.length} more',
                    style: const TextStyle(
                      fontSize: NeoethosTokens.fsCaption,
                      color: NeoethosTokens.textFaint,
                    ),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

class _PerformanceSection extends ConsumerWidget {
  const _PerformanceSection();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(intelligenceProvider);
    return _SectionCard(
      title: 'PERFORMANCE',
      body: async.when(
        loading: () => const _Skeleton(lines: 2),
        error: (_, __) => const _ErrorLine(),
        data: (snap) {
          final acc = snap.walkforwardAvgAccuracy;
          final splits = snap.walkforwardSplits ?? 0;
          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _KvRow(
                'WFA splits',
                splits == 0 ? '—' : '$splits',
              ),
              _KvRow(
                'Avg accuracy',
                acc == null
                    ? '—'
                    : '${(acc * 100).toStringAsFixed(1)}%',
                valueColor: acc == null
                    ? null
                    : acc >= 0.55
                        ? NeoethosTokens.buy
                        : acc >= 0.50
                            ? NeoethosTokens.warning
                            : NeoethosTokens.sell,
              ),
              if (snap.artifactCount == 0 && acc == null)
                const Padding(
                  padding: EdgeInsets.only(top: 6),
                  child: Text(
                    'Run validation to populate metrics.',
                    style: TextStyle(
                      fontSize: NeoethosTokens.fsCaption,
                      color: NeoethosTokens.textFaint,
                      height: 1.4,
                    ),
                  ),
                ),
            ],
          );
        },
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Proposed Action
// ---------------------------------------------------------------------------

class _ProposedActionSection extends ConsumerWidget {
  const _ProposedActionSection();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(pendingActionsProvider);
    return _SectionCard(
      title: 'PROPOSED ACTION',
      body: async.when(
        loading: () => const _Skeleton(lines: 3),
        error: (_, __) => const _ErrorLine(),
        data: (actions) {
          final pending =
              actions.where((a) => a.status == 'pending').toList();
          if (pending.isEmpty) {
            return const Text(
              'No pending proposals.',
              style: TextStyle(
                fontSize: NeoethosTokens.fsCaption,
                color: NeoethosTokens.textFaint,
                height: 1.4,
              ),
            );
          }
          // Show only the most recent proposal — anything more goes
          // into the persistent banner that AppShell renders at the
          // top of the dock. The rail shows the headline action.
          final top = pending.first;
          return _ProposalCard(action: top);
        },
      ),
    );
  }
}

class _ProposalCard extends ConsumerWidget {
  final PendingAction action;
  const _ProposalCard({required this.action});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final headline = _formatHeadline(action);
    return Container(
      padding: const EdgeInsets.all(NeoethosTokens.spSm),
      decoration: BoxDecoration(
        color: NeoethosTokens.accentMuted,
        border: Border.all(
          color: NeoethosTokens.accent.withValues(alpha: 0.5),
        ),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            headline,
            style: const TextStyle(
              fontSize: NeoethosTokens.fsBody,
              fontWeight: FontWeight.w800,
              color: NeoethosTokens.textPrimary,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            action.reason,
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: NeoethosTokens.fsCaption,
              color: NeoethosTokens.textMuted,
              height: 1.4,
            ),
          ),
          const SizedBox(height: NeoethosTokens.spSm),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: () => _reject(ref),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: NeoethosTokens.sell,
                    side: BorderSide(
                      color: NeoethosTokens.sell.withValues(alpha: 0.55),
                    ),
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    minimumSize: const Size(0, 28),
                  ),
                  child: const Text(
                    'Reject',
                    style: TextStyle(
                      fontSize: NeoethosTokens.fsCaption,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 6),
              Expanded(
                child: FilledButton(
                  onPressed: () => _confirm(ref),
                  style: FilledButton.styleFrom(
                    backgroundColor: NeoethosTokens.buy,
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    minimumSize: const Size(0, 28),
                  ),
                  child: const Text(
                    'Confirm',
                    style: TextStyle(
                      fontSize: NeoethosTokens.fsCaption,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  String _formatHeadline(PendingAction a) {
    switch (a.kindTag) {
      case 'close_position':
        final pos = a.positionId ?? 0;
        final sym = a.symbolHint ?? '';
        return sym.isEmpty
            ? 'Close position #$pos'
            : 'Close $sym (#$pos)';
      default:
        return a.kindTag.isEmpty ? 'Proposed action' : a.kindTag;
    }
  }

  Future<void> _confirm(WidgetRef ref) async {
    final client = ref.read(backendClientProvider);
    await client.confirmPendingAction(action.id);
    await ref.read(pendingActionsProvider.notifier).refreshNow();
  }

  Future<void> _reject(WidgetRef ref) async {
    final client = ref.read(backendClientProvider);
    await client.rejectPendingAction(action.id);
    await ref.read(pendingActionsProvider.notifier).refreshNow();
  }
}

// ---------------------------------------------------------------------------
// Discovery Targets (symbols trained on)
// ---------------------------------------------------------------------------

class _DiscoveryTargetsSection extends ConsumerWidget {
  const _DiscoveryTargetsSection();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(intelligenceProvider);
    return _SectionCard(
      title: 'TRAINED ON',
      subtitle: async.maybeWhen(
        data: (snap) => '${snap.discoveryTargets.length}',
        orElse: () => '—',
      ),
      body: async.when(
        loading: () => const _Skeleton(lines: 2),
        error: (_, __) => const _ErrorLine(),
        data: (snap) {
          if (snap.discoveryTargets.isEmpty) {
            return const Text(
              'No discovery targets yet.',
              style: TextStyle(
                fontSize: NeoethosTokens.fsCaption,
                color: NeoethosTokens.textFaint,
                height: 1.4,
              ),
            );
          }
          return Wrap(
            spacing: 4,
            runSpacing: 4,
            children: [
              for (final t in snap.discoveryTargets.take(8))
                _SymbolChip(label: '${t.symbol} ${t.baseTf}'),
              if (snap.discoveryTargets.length > 8)
                _SymbolChip(
                  label: '+${snap.discoveryTargets.length - 8}',
                  faded: true,
                ),
            ],
          );
        },
      ),
    );
  }
}

class _SymbolChip extends StatelessWidget {
  final String label;
  final bool faded;
  const _SymbolChip({required this.label, this.faded = false});
  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: NeoethosTokens.appBg,
        border: Border.all(
          color: NeoethosTokens.border,
        ),
        borderRadius: BorderRadius.circular(3),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: NeoethosTokens.fsCaption - 1,
          fontFamily: 'monospace',
          fontWeight: FontWeight.w600,
          color: faded
              ? NeoethosTokens.textFaint
              : NeoethosTokens.textPrimary,
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Skeletons
// ---------------------------------------------------------------------------

class _Skeleton extends StatelessWidget {
  final int lines;
  const _Skeleton({required this.lines});
  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        for (var i = 0; i < lines; i++)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 3),
            child: Container(
              height: 10,
              decoration: BoxDecoration(
                color: NeoethosTokens.appBg,
                borderRadius: BorderRadius.circular(3),
              ),
            ),
          ),
      ],
    );
  }
}

class _ErrorLine extends StatelessWidget {
  const _ErrorLine();
  @override
  Widget build(BuildContext context) => const Text(
        'Backend offline — retrying…',
        style: TextStyle(
          fontSize: NeoethosTokens.fsCaption,
          color: NeoethosTokens.textFaint,
          fontStyle: FontStyle.italic,
        ),
      );
}
