import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
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
    final async = ref.watch(intelligenceProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Intelligence',
            subtitle: 'Trained model artifacts · discovery targets',
          ),
          async.when(
            data: (s) => _Body(snapshot: s),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
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
    final dtFmt = DateFormat('yyyy-MM-dd HH:mm');
    final lastTouched = snapshot.lastTouchedUnixMs == null
        ? '—'
        : dtFmt.format(DateTime.fromMillisecondsSinceEpoch(
            snapshot.lastTouchedUnixMs!));
    final avgAcc = snapshot.walkforwardAvgAccuracy;
    final accStr = (avgAcc == null || avgAcc == 0.0)
        ? '— (walkforward has not run yet)'
        : '${(avgAcc * 100).toStringAsFixed(2)} %';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: 'Inventory',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row('Models directory', snapshot.modelsDir),
              _Row(
                'Directory exists',
                snapshot.modelsDirExists ? 'YES' : 'NO',
                accent: snapshot.modelsDirExists
                    ? ForexAiTokens.buy
                    : ForexAiTokens.sell,
              ),
              _Row('Artifact count', '${snapshot.artifactCount}'),
              _Row('Last touched', lastTouched),
              _Row('Walkforward splits',
                  '${snapshot.walkforwardSplits ?? 0}'),
              _Row('Walkforward avg accuracy', accStr),
            ],
          ),
        ),
        if (snapshot.artifacts.isNotEmpty)
          SectionCard(
            title: 'Model artifacts',
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
                      color: ForexAiTokens.surfaceBg,
                      border: Border.all(color: ForexAiTokens.border),
                      borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
                    ),
                    child: Text(
                      a,
                      style: const TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: ForexAiTokens.textPrimary,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        SectionCard(
          title: 'Discovery targets',
          child: snapshot.discoveryTargets.isEmpty
              ? const Text(
                  'No model_targets.json found yet. Run Discovery once '
                  '(Strategy Discovery Engine screen) and the picked '
                  'portfolio will land here.',
                  style: TextStyle(
                    color: ForexAiTokens.textMuted,
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
                color: ForexAiTokens.accent,
              ),
            ),
          ),
          Expanded(
            child: Text(
              target.strategyId,
              style: const TextStyle(
                fontSize: 12,
                color: ForexAiTokens.textPrimary,
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
                color: ForexAiTokens.textMuted,
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
                color: ForexAiTokens.textMuted,
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
                  color: ForexAiTokens.textMuted,
                ),
              ),
            ),
            Expanded(
              child: Text(
                value,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: accent ?? ForexAiTokens.textPrimary,
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
  Widget build(BuildContext context) => const Padding(
        padding: EdgeInsets.symmetric(vertical: 16),
        child: Text(
          'Scanning models directory…',
          style: TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
        ),
      );
}

class _Error extends StatelessWidget {
  final String error;
  const _Error({required this.error});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Text(
          'Backend unreachable: $error',
          style: const TextStyle(color: ForexAiTokens.sell, fontSize: 12),
        ),
      );
}
