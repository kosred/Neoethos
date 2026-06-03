import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/backend_error_widget.dart';
import '_placeholder.dart';
import 'widgets/engine_controls.dart';

class TrainingScreen extends ConsumerWidget {
  const TrainingScreen({super.key});
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(enginesProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Model Swarm Training',
            subtitle: 'Ensemble training pipeline',
          ),
          async.when(
            data: (e) => EngineControls(
              kind: 'Training',
              running: e.trainingRunning,
              state: e.training,
              summary: e.trainingSummary,
              start: ({String? symbol, String? baseTf}) =>
                  ref.read(backendClientProvider).startTraining(
                        symbol: symbol,
                        baseTf: baseTf,
                      ),
              stop: () => ref.read(backendClientProvider).stopTraining(),
              onChanged: () => ref.invalidate(enginesProvider),
              description:
                  'Training drives the 33-model ensemble pipeline (tree, '
                  'evolutionary, reinforcement-learning, statistical, '
                  'anomaly-detection families) over the symbol/timeframe '
                  'you chose. Per-epoch loss + accuracy stream into the '
                  'sectioned log under the TRAINING section.\n\n'
                  'Auto-start: if you launch Discovery first and it '
                  'finishes cleanly (Succeeded), Training kicks off '
                  'automatically against the same (symbol, timeframe) — '
                  'the natural pipeline sequence. Manual Start here is '
                  'still available for re-training without re-running '
                  'Discovery.',
            ),
            loading: () => const _Loading(),
            error: (err, _) => BackendErrorWidget(
                    error: err, title: 'Training status unavailable'),
          ),
        ],
      ),
    );
  }
}

class _Loading extends StatelessWidget {
  const _Loading();
  @override
  Widget build(BuildContext context) => const Padding(
        padding: EdgeInsets.symmetric(vertical: 16),
        child: Text(
          'Loading engine state…',
          style: TextStyle(color: NeoethosTokens.textMuted, fontSize: 12),
        ),
      );
}

