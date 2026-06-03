import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../l10n/app_localizations.dart';
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
    final l10n = AppLocalizations.of(context)!;
    final async = ref.watch(enginesProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ViewHeader(
            title: l10n.trainingTitle,
            subtitle: l10n.trainingSubtitle,
          ),
          async.when(
            data: (e) => EngineControls(
              kind: l10n.engineTraining,
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
              description: l10n.trainingEngineControlsDescription,
            ),
            loading: () => const _Loading(),
            error: (err, _) => BackendErrorWidget(
                    error: err, title: l10n.trainingStatusUnavailable),
          ),
        ],
      ),
    );
  }
}

class _Loading extends StatelessWidget {
  const _Loading();
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 16),
        child: Text(
          AppLocalizations.of(context)!.trainingLoadingEngineState,
          style: const TextStyle(
              color: NeoethosTokens.textMuted, fontSize: 12),
        ),
      );
}

