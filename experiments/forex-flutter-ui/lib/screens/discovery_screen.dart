import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';
import 'widgets/engine_controls.dart';

class DiscoveryScreen extends ConsumerWidget {
  const DiscoveryScreen({super.key});
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(enginesProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Strategy Discovery Engine',
            subtitle: 'Genetic search → portfolio',
          ),
          async.when(
            data: (e) => EngineControls(
              kind: 'Discovery',
              running: e.discoveryRunning,
              state: e.discovery,
              summary: e.discoverySummary,
              start: ({String? symbol, String? baseTf}) =>
                  ref.read(backendClientProvider).startDiscovery(
                        symbol: symbol,
                        baseTf: baseTf,
                      ),
              stop: () => ref.read(backendClientProvider).stopDiscovery(),
              onChanged: () => ref.invalidate(enginesProvider),
              description:
                  'Discovery runs a genetic algorithm over the configured '
                  'symbol/timeframe to evolve a portfolio of candidate '
                  'strategies. The Rust engine drives '
                  'population/generations/novelty internally — defaults '
                  'come from config.yaml. Once a run completes, the '
                  'selected portfolio lands in models_targets.json and '
                  'Training picks it up automatically.',
            ),
            loading: () => const _Loading(),
            error: (err, _) =>
                _Error(error: err is DioException ? _formatDio(err) : '$err'),
          ),
        ],
      ),
    );
  }
}

String _formatDio(DioException e) {
  final body = e.response?.data;
  if (body is Map && body['error'] is String) return body['error'] as String;
  return e.message ?? e.toString();
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
