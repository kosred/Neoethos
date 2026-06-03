import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../l10n/app_localizations.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/backend_error_widget.dart';
import '_placeholder.dart';

/// Intelligence — shows the current model swarm inventory: which
/// artifacts are on disk, the discovery targets the last run picked,
/// and walkforward metrics if available. Read-only mirror of the
/// `models/` directory; lifecycle control lives in the Training
/// screen.

class IntelligenceScreen extends ConsumerWidget {
  const IntelligenceScreen({super.key});
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final async = ref.watch(intelligenceProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ViewHeader(
            title: l10n.intelligenceTitle,
            subtitle: l10n.intelligenceSubtitle,
          ),
          async.when(
            data: (s) => _Body(snapshot: s),
            loading: () => const _Loading(),
            error: (err, _) => BackendErrorWidget(
                    error: err, title: l10n.intelligenceUnavailable),
          ),
        ],
      ),
    );
  }
}

class _Body extends StatelessWidget {
  final IntelligenceSnapshot snapshot;
  const _Body({required this.snapshot});

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final dtFmt = DateFormat('yyyy-MM-dd HH:mm');
    final lastTouched = snapshot.lastTouchedUnixMs == null
        ? '—'
        : dtFmt.format(DateTime.fromMillisecondsSinceEpoch(
            snapshot.lastTouchedUnixMs!));
    final avgAcc = snapshot.walkforwardAvgAccuracy;
    final accStr = (avgAcc == null || avgAcc == 0.0)
        ? l10n.intelligenceWalkforwardNotRun
        : '${(avgAcc * 100).toStringAsFixed(2)} %';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: l10n.intelligenceInventory,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row(l10n.intelligenceModelsDirectory, snapshot.modelsDir),
              _Row(
                l10n.intelligenceDirectoryExists,
                snapshot.modelsDirExists
                    ? l10n.intelligenceYes
                    : l10n.intelligenceNo,
                accent: snapshot.modelsDirExists
                    ? NeoethosTokens.buy
                    : NeoethosTokens.sell,
              ),
              _Row(l10n.intelligenceArtifactCount,
                  '${snapshot.artifactCount}'),
              _Row(l10n.intelligenceLastTouched, lastTouched),
              _Row(l10n.intelligenceWalkforwardSplits,
                  '${snapshot.walkforwardSplits ?? 0}'),
              _Row(l10n.intelligenceWalkforwardAvgAccuracy, accStr),
            ],
          ),
        ),
        if (snapshot.artifacts.isNotEmpty)
          SectionCard(
            title: l10n.intelligenceModelArtifacts,
            child: Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final a in snapshot.artifacts)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 3,
                    ),
                    decoration: BoxDecoration(
                      color: NeoethosTokens.surfaceBg,
                      border: Border.all(color: NeoethosTokens.border),
                      borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
                    ),
                    child: Text(
                      a,
                      style: const TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: NeoethosTokens.textPrimary,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        SectionCard(
          title: l10n.intelligenceDiscoveryTargets,
          child: snapshot.discoveryTargets.isEmpty
              ? Text(
                  l10n.intelligenceNoTargets,
                  style: const TextStyle(
                    color: NeoethosTokens.textMuted,
                    fontSize: 12,
                  ),
                )
              : Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    for (final t in snapshot.discoveryTargets)
                      _TargetRow(target: t),
                  ],
                ),
        ),
      ],
    );
  }
}

class _TargetRow extends StatelessWidget {
  final DiscoveryTarget target;
  const _TargetRow({required this.target});
  @override
  Widget build(BuildContext context) {
    final sharpe = target.sharpe == null
        ? '—'
        : target.sharpe!.toStringAsFixed(2);
    final winRate = target.winRate == null
        ? '—'
        : '${(target.winRate! * 100).toStringAsFixed(1)} %';
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        children: [
          SizedBox(
            width: 90,
            child: Text(
              '${target.symbol}/${target.baseTf}',
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: NeoethosTokens.accent,
              ),
            ),
          ),
          Expanded(
            child: Text(
              target.strategyId,
              style: const TextStyle(
                fontSize: 12,
                color: NeoethosTokens.textPrimary,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          SizedBox(
            width: 70,
            child: Text(
              'sh $sharpe',
              style: const TextStyle(
                fontSize: 11,
                color: NeoethosTokens.textMuted,
              ),
              textAlign: TextAlign.right,
            ),
          ),
          SizedBox(
            width: 70,
            child: Text(
              'wr $winRate',
              style: const TextStyle(
                fontSize: 11,
                color: NeoethosTokens.textMuted,
              ),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
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
            Expanded(
              child: Text(
                value,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: accent ?? NeoethosTokens.textPrimary,
                ),
                overflow: TextOverflow.ellipsis,
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
          AppLocalizations.of(context)!.intelligenceScanningModels,
          style: const TextStyle(
              color: NeoethosTokens.textMuted, fontSize: 12),
        ),
      );
}

