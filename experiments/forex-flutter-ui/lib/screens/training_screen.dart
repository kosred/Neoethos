import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

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
            data: (e) => _Body(state: e.training),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
          ),
        ],
      ),
    );
  }
}

class _Body extends StatelessWidget {
  final String state;
  const _Body({required this.state});
  @override
  Widget build(BuildContext context) {
    final running = state.toLowerCase() == 'running';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: 'Current Job',
          child: Row(
            children: [
              Container(
                width: 10,
                height: 10,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: running ? ForexAiTokens.buy : ForexAiTokens.textFaint,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                state,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: running ? ForexAiTokens.buy : ForexAiTokens.textPrimary,
                ),
              ),
            ],
          ),
        ),
        const SectionCard(
          title: 'How training works',
          child: Text(
            'Training drives the 33-model ensemble pipeline (tree, '
            'evolutionary, reinforcement-learning, statistical, '
            'anomaly-detection families) over the discovery output. '
            'Per-epoch loss / accuracy will stream here once the '
            'POST /engines/training/start endpoint ships.',
            style: TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
          ),
        ),
      ],
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
