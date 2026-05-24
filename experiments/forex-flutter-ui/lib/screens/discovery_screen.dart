import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';
import 'widgets/engine_controls.dart';

class DiscoveryScreen extends ConsumerStatefulWidget {
  const DiscoveryScreen({super.key});

  @override
  ConsumerState<DiscoveryScreen> createState() => _DiscoveryScreenState();
}

class _DiscoveryScreenState extends ConsumerState<DiscoveryScreen> {
  // #194: Advanced GA knobs. Defaults reflect the engine's
  // `DiscoveryConfig::default()` so the "Estimated time" string is
  // accurate from the first render. The user can drag any slider to
  // override; sending `null` keeps the engine default.
  int _population = 100;
  int _generations = 5;
  int _maxIndicators = 12;
  int _targetCandidates = 200;
  int _portfolioSize = 100;
  bool _advancedOpen = false;

  /// Rough ETA in seconds. Empirical: ~1.2 ms per candidate evaluation
  /// on the Ryzen 7 5700U / Iris Xe baseline; multiplied by an MTF
  /// fan-out of ~6 (M1 base + 3 higher TFs evaluated twice for cross-
  /// confirmation + portfolio diversification scoring). The constant
  /// is intentionally pessimistic — over-promising "5 minutes" and
  /// delivering "12 minutes" is much worse than the reverse.
  int _estimatedSeconds() {
    const msPerEval = 1.2;
    const fanOut = 6.0;
    final evals = _population * _generations * _targetCandidates;
    final ms = evals * msPerEval * fanOut;
    return (ms / 1000).round();
  }

  String _formatEta(int seconds) {
    if (seconds < 60) return '~${seconds}s';
    if (seconds < 3600) {
      final m = seconds ~/ 60;
      final s = seconds % 60;
      return '~${m}m ${s.toString().padLeft(2, '0')}s';
    }
    final h = seconds ~/ 3600;
    final m = (seconds % 3600) ~/ 60;
    return '~${h}h ${m.toString().padLeft(2, '0')}m';
  }

  @override
  Widget build(BuildContext context) {
    final async = ref.watch(enginesProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Strategy Discovery Engine',
            subtitle: 'Genetic search → portfolio',
          ),
          _hyperparamsCard(),
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
                        population: _population,
                        generations: _generations,
                        maxIndicators: _maxIndicators,
                        targetCandidates: _targetCandidates,
                        portfolioSize: _portfolioSize,
                      ),
              stop: () => ref.read(backendClientProvider).stopDiscovery(),
              onChanged: () => ref.invalidate(enginesProvider),
              description:
                  'Discovery runs a genetic algorithm over the configured '
                  'symbol/timeframe to evolve a portfolio of candidate '
                  'strategies. Tune the GA knobs above before pressing '
                  'Start; defaults come from config.yaml. Once a run '
                  'completes, the selected portfolio lands in '
                  'models_targets.json and Training picks it up '
                  'automatically.',
            ),
            loading: () => const _Loading(),
            error: (err, _) =>
                _Error(error: err is DioException ? _formatDio(err) : '$err'),
          ),
        ],
      ),
    );
  }

  Widget _hyperparamsCard() {
    final eta = _formatEta(_estimatedSeconds());
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: ExpansionTile(
        title: const Text(
          'Advanced: GA hyperparameters',
          style: TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
        ),
        subtitle: Text(
          'Estimated runtime: $eta · pop × gen × candidates = '
          '${_population * _generations * _targetCandidates} evaluations',
          style: const TextStyle(fontSize: 11, color: ForexAiTokens.textMuted),
        ),
        initiallyExpanded: _advancedOpen,
        onExpansionChanged: (v) => setState(() => _advancedOpen = v),
        children: [
          _slider(
            label: 'Population',
            value: _population,
            min: 20,
            max: 500,
            divisions: 24,
            onChanged: (v) => setState(() => _population = v.round()),
            hint: 'Number of genomes per generation. Larger = '
                'more exploration but slower.',
          ),
          _slider(
            label: 'Generations',
            value: _generations,
            min: 1,
            max: 50,
            divisions: 49,
            onChanged: (v) => setState(() => _generations = v.round()),
            hint: 'Evolutionary rounds. Diminishing returns past ~20 '
                'for short-horizon strategies.',
          ),
          _slider(
            label: 'Max indicators per strategy',
            value: _maxIndicators,
            min: 3,
            max: 30,
            divisions: 27,
            onChanged: (v) => setState(() => _maxIndicators = v.round()),
            hint: 'Cap on indicator features per candidate. Higher = '
                'more expressive but easier to overfit.',
          ),
          _slider(
            label: 'Target candidates',
            value: _targetCandidates,
            min: 50,
            max: 1000,
            divisions: 19,
            onChanged: (v) => setState(() => _targetCandidates = v.round()),
            hint: 'Number of evaluated candidates to keep before '
                'portfolio selection.',
          ),
          _slider(
            label: 'Portfolio size',
            value: _portfolioSize,
            min: 5,
            max: 200,
            divisions: 39,
            onChanged: (v) => setState(() => _portfolioSize = v.round()),
            hint: 'Final selected portfolio size — what Training will '
                'pick up.',
          ),
        ],
      ),
    );
  }

  Widget _slider({
    required String label,
    required int value,
    required double min,
    required double max,
    required int divisions,
    required ValueChanged<double> onChanged,
    required String hint,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  '$label: $value',
                  style: const TextStyle(fontSize: 12),
                ),
              ),
            ],
          ),
          Slider(
            value: value.toDouble(),
            min: min,
            max: max,
            divisions: divisions,
            label: value.toString(),
            onChanged: onChanged,
          ),
          Text(
            hint,
            style: const TextStyle(
              fontSize: 10,
              color: ForexAiTokens.textMuted,
            ),
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
