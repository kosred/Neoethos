import 'dart:async';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart' show EnginesSnapshot;
import '../api/error_translation.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/backend_error_widget.dart';
import '../widgets/multi_symbol_picker.dart';
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
  // These are only the pre-config fallbacks shown for the split-second
  // before the user's saved config arrives; build() overwrites them from
  // `settingsProvider` (the single source of truth). Kept non-tiny so a
  // run started before settings load is never a shallow 5-generation one.
  int _population = 100;
  int _generations = 100;
  int _maxIndicators = 12;
  int _targetCandidates = 200;
  int _portfolioSize = 100;
  bool _advancedOpen = false;
  // Pull the GA knobs from the user's config exactly once (no hardcoded
  // shallow defaults — the operator's saved values are authoritative).
  bool _loadedConfigDefaults = false;

  // #264 (2026-05-29): Higher-timeframe multi-select. Defaults mirror
  // the backend's `DEFAULT_HIGHER_TFS = ["M5", "M15", "H1"]` in
  // `engines_control.rs:66` so an operator who never touches this
  // chip row sees the same behaviour as the pre-#264 build. Picking
  // any non-default subset overrides only the higher-TF context for
  // the next discovery run.
  final Set<String> _higherTfs = {'M5', 'M15', 'H1'};

  // #265 (2026-05-29): Sequential multi-pair queue. When the queue is
  // empty (`_symbolQueue.isEmpty`), the screen renders the existing
  // single-pair `EngineControls` widget unchanged — the operator picks
  // one symbol + base TF in that card and presses Start as before.
  //
  // The moment the operator adds a symbol to the queue, the screen
  // switches to queue mode: `EngineControls` is hidden, the queue
  // panel takes over rendering of "Current Job" + Start/Stop, and
  // pressing "Run queue" fires off discovery for the first symbol,
  // polls `enginesProvider` (the existing 2 s `/engines/status`
  // poller) until the engine state goes terminal, then advances to
  // the next symbol. Cancelling mid-queue calls `/engines/discovery/
  // stop` on the active run and aborts the loop.
  //
  // Why front-end orchestration instead of a backend queue endpoint:
  // the backend already enforces "one discovery at a time" via
  // `engine_state(JobKind::Discovery) == Running` → 409. Letting the
  // UI drive sequencing keeps the backend stateless w.r.t. the
  // queue and lets the operator reorder / drop symbols mid-flight
  // without a second server API surface to maintain.
  final List<String> _symbolQueue = [];
  // State-owned so the field doesn't lose its cursor / get re-created
  // on every rebuild. Disposed in `dispose()` below.
  final TextEditingController _queueSymbolController = TextEditingController();
  String _queueBaseTf = 'M1';
  bool _queueRunning = false;
  bool _queueAbortRequested = false;
  int _queueCursor = -1; // -1 = idle; otherwise index of in-flight symbol.
  String _queueSummary = '';

  @override
  void dispose() {
    _queueSymbolController.dispose();
    super.dispose();
  }

  /// Rough ETA in seconds. Empirical: ~1.2 ms per candidate evaluation
  /// on the Ryzen 7 5700U / Iris Xe baseline; multiplied by an MTF
  /// fan-out of ~6 (M1 base + 3 higher TFs evaluated twice for cross-
  /// confirmation + portfolio diversification scoring). The constant
  /// is intentionally pessimistic — over-promising "5 minutes" and
  /// delivering "12 minutes" is much worse than the reverse.
  int _estimatedSeconds() {
    const msPerEval = 1.2;
    // #264: fan-out tracks the actual higher-TF set size + 1 for the
    // base TF. With the legacy `_higherTfs = {M5,M15,H1}` we still get
    // the familiar ~6× constant (3 higher TFs × 2 cross-confirm passes
    // = 6); narrowing the chip selection now also narrows the ETA.
    final fanOut = (_higherTfs.length + 1) * 1.5;
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
    // Single source of truth: seed the GA hyperparameters from the user's
    // saved config the first time settings arrive. No hardcoded shallow
    // defaults — the operator's values (e.g. generations=1000) flow
    // straight through. The user can still override via the sliders below.
    final settingsAsync = ref.watch(settingsProvider);
    if (!_loadedConfigDefaults) {
      settingsAsync.whenData((s) {
        _loadedConfigDefaults = true;
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (!mounted) return;
          setState(() {
            _population = s.searchPopulation.clamp(20, 1000).toInt();
            _generations = s.searchGenerations.clamp(5, 2000).toInt();
            if (s.searchMaxIndicators > 0) {
              _maxIndicators = s.searchMaxIndicators.clamp(3, 30).toInt();
            }
            _portfolioSize = s.searchPortfolioSize.clamp(5, 200).toInt();
          });
        });
      });
    }
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Strategy Discovery Engine',
            subtitle: 'Genetic search → portfolio',
          ),
          _hyperparamsCard(),
          // F-14: compact live-stats panel. Renders only while a
          // discovery run is active (counters non-empty); otherwise
          // collapses to SizedBox.shrink so it never adds dead space.
          _DiscoveryLiveStats(snapshot: async.valueOrNull),
          _queueCard(async),
          // Single-pair controls render only when the queue is empty.
          // Once the operator adds a symbol to the queue, the queue
          // panel above owns the entire Start/Stop affordance to
          // avoid two competing "Start" buttons fighting over the
          // single backend Discovery slot.
          if (_symbolQueue.isEmpty)
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
                          // #264: thread the chip selection through
                          // so single-pair runs respect the operator's
                          // higher-TF picks too — not just queue mode.
                          higherTfs: _higherTfs.toList(),
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
                    'automatically.\n\n'
                    'Queue mode: add 2+ symbols to the queue above to run '
                    'discovery sequentially across multiple pairs without '
                    'baby-sitting each one.',
              ),
              loading: () => const _Loading(),
              error: (err, _) => BackendErrorWidget(
                error: err,
                title: 'Discovery status unavailable',
              ),
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
          'Advanced: GA hyperparameters + higher timeframes',
          style: TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
        ),
        subtitle: Text(
          'Estimated runtime: $eta · pop × gen × candidates = '
          '${_population * _generations * _targetCandidates} evaluations · '
          '${_higherTfs.length} higher-TF',
          style: const TextStyle(fontSize: 11, color: ForexAiTokens.textMuted),
        ),
        initiallyExpanded: _advancedOpen,
        onExpansionChanged: (v) => setState(() => _advancedOpen = v),
        children: [
          _higherTfChips(),
          _slider(
            label: 'Population',
            value: _population,
            min: 20,
            max: 1000,
            divisions: 49,
            onChanged: (v) => setState(() => _population = v.round()),
            hint: 'Number of genomes per generation. Larger = '
                'more exploration but slower.',
          ),
          _slider(
            label: 'Generations',
            value: _generations,
            min: 5,
            max: 2000,
            divisions: 399,
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

  /// #264: Higher-timeframe chip row. Each chip is a `FilterChip`
  /// over the canonical-timeframes list (from
  /// `/broker/timeframes`, which mirrors
  /// `neoethos_core::CANONICAL_TIMEFRAMES`). Empty selection is
  /// rejected client-side — sending an empty list to the server
  /// would fall back to DEFAULT_HIGHER_TFS silently, which is
  /// confusing UX (operator thinks "no higher TFs" but actually
  /// gets {M5,M15,H1}). We keep at least one chip selected at all
  /// times.
  Widget _higherTfChips() {
    final tfAsync = ref.watch(brokerTimeframesProvider);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Higher timeframes (multi-select)',
            style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 4),
          const Text(
            'Discovery cross-confirms entries against these higher TFs in '
            'addition to the base TF. Default mirrors backend '
            'DEFAULT_HIGHER_TFS (M5/M15/H1).',
            style: TextStyle(fontSize: 10, color: ForexAiTokens.textMuted),
          ),
          const SizedBox(height: 8),
          tfAsync.when(
            data: (list) {
              if (list.isEmpty) {
                return const Text(
                  '/broker/timeframes returned an empty list — check '
                  'neoethos_core::CANONICAL_TIMEFRAMES.',
                  style: TextStyle(fontSize: 11, color: ForexAiTokens.sell),
                );
              }
              return Wrap(
                spacing: 6,
                runSpacing: 4,
                children: [
                  for (final tf in list)
                    FilterChip(
                      label: Text(
                        tf,
                        style: const TextStyle(fontSize: 11),
                      ),
                      selected: _higherTfs.contains(tf),
                      visualDensity: VisualDensity.compact,
                      materialTapTargetSize:
                          MaterialTapTargetSize.shrinkWrap,
                      onSelected: _queueRunning
                          ? null
                          : (sel) {
                              setState(() {
                                if (sel) {
                                  _higherTfs.add(tf);
                                } else if (_higherTfs.length > 1) {
                                  // Refuse to deselect the LAST chip
                                  // (empty list would silently fall
                                  // back to the server default).
                                  _higherTfs.remove(tf);
                                } else {
                                  ScaffoldMessenger.of(context)
                                      .showSnackBar(const SnackBar(
                                    content: Text(
                                      'Keep at least one higher TF — '
                                      'empty selection would silently '
                                      'fall back to the server default.',
                                    ),
                                    duration: Duration(seconds: 2),
                                  ));
                                }
                              });
                            },
                    ),
                ],
              );
            },
            loading: () => const Text(
              'Loading /broker/timeframes…',
              style: TextStyle(fontSize: 11, color: ForexAiTokens.textMuted),
            ),
            error: (err, _) => Text(
              'Cannot fetch timeframes from broker: $err',
              style: const TextStyle(fontSize: 11, color: ForexAiTokens.warning),
            ),
          ),
        ],
      ),
    );
  }

  /// #265: Sequential multi-pair queue. Render order:
  ///   1. Header + short explainer
  ///   2. Add-symbol input row (text field + Base TF dropdown + Add)
  ///   3. Queue list (chips with [x] removal)
  ///   4. Run / Stop queue buttons + cursor progress
  ///   5. Live status line from the running discovery
  Widget _queueCard(AsyncValue<EnginesSnapshot> enginesAsync) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.queue, size: 16),
                const SizedBox(width: 6),
                const Text(
                  'Multi-pair queue (sequential)',
                  style:
                      TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
                ),
                const Spacer(),
                if (_symbolQueue.isNotEmpty)
                  TextButton.icon(
                    onPressed: _queueRunning
                        ? null
                        : () => setState(() {
                              _symbolQueue.clear();
                              _queueCursor = -1;
                            }),
                    icon: const Icon(Icons.clear_all, size: 14),
                    label: const Text('Clear queue'),
                  ),
              ],
            ),
            const SizedBox(height: 4),
            const Text(
              'Add 2+ symbols to run discovery sequentially. Each symbol '
              'uses the same base TF + higher-TF set + GA knobs. The '
              'next symbol starts only after the previous one reaches '
              'a terminal state (Succeeded / Failed / Cancelled).',
              style: TextStyle(fontSize: 11, color: ForexAiTokens.textMuted),
            ),
            const SizedBox(height: 10),
            _queueAddRow(),
            const SizedBox(height: 8),
            _queueListChips(),
            if (_symbolQueue.isNotEmpty) ...[
              const SizedBox(height: 12),
              _queueControlsRow(enginesAsync),
              if (_queueRunning || _queueSummary.isNotEmpty) ...[
                const SizedBox(height: 8),
                _queueProgressRow(enginesAsync),
              ],
            ],
          ],
        ),
      ),
    );
  }

  Widget _queueAddRow() {
    final tfAsync = ref.watch(brokerTimeframesProvider);
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          flex: 3,
          child: TextField(
            controller: _queueSymbolController,
            enabled: !_queueRunning,
            textCapitalization: TextCapitalization.characters,
            decoration: const InputDecoration(
              labelText: 'Symbol (e.g. EURUSD)',
              isDense: true,
              border: OutlineInputBorder(),
              helperText: 'Type the broker ticker exactly. Validation '
                  'happens server-side at run time.',
            ),
            onSubmitted: (_) => _addToQueue(),
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          flex: 2,
          child: tfAsync.when(
            data: (list) {
              if (list.isEmpty) {
                return const Text(
                  'TF list empty',
                  style: TextStyle(fontSize: 11, color: ForexAiTokens.sell),
                );
              }
              final current =
                  list.contains(_queueBaseTf) ? _queueBaseTf : list.first;
              if (current != _queueBaseTf) {
                WidgetsBinding.instance.addPostFrameCallback((_) {
                  if (mounted) setState(() => _queueBaseTf = current);
                });
              }
              return InputDecorator(
                decoration: const InputDecoration(
                  labelText: 'Base TF',
                  isDense: true,
                  border: OutlineInputBorder(),
                ),
                child: DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    value: current,
                    isDense: true,
                    isExpanded: true,
                    items: [
                      for (final tf in list)
                        DropdownMenuItem(value: tf, child: Text(tf)),
                    ],
                    onChanged: _queueRunning
                        ? null
                        : (v) {
                            if (v != null) setState(() => _queueBaseTf = v);
                          },
                  ),
                ),
              );
            },
            loading: () => const Text(
              'Loading TFs…',
              style: TextStyle(fontSize: 11, color: ForexAiTokens.textMuted),
            ),
            error: (err, _) => Text(
              'Could not load timeframes — ${describeError(err)}. '
              'Authenticate in Broker Setup, then refresh.',
              style: const TextStyle(fontSize: 11, color: ForexAiTokens.warning),
            ),
          ),
        ),
        const SizedBox(width: 8),
        FilledButton.icon(
          onPressed: _queueRunning ? null : _addToQueue,
          icon: const Icon(Icons.add, size: 16),
          label: const Text('Add'),
        ),
        const SizedBox(width: 8),
        // F-337: multi-select pairs from the broker catalog (checkboxes +
        // search) instead of typing tickers one by one.
        OutlinedButton.icon(
          onPressed: _queueRunning ? null : _browsePairs,
          icon: const Icon(Icons.checklist, size: 16),
          label: const Text('Browse'),
        ),
      ],
    );
  }

  /// F-337: open the multi-select catalog picker, seeded with the current
  /// queue, and replace the queue with the operator's checked selection.
  Future<void> _browsePairs() async {
    final picked = await showMultiSymbolPicker(
      context,
      preselected: _symbolQueue.toSet(),
    );
    if (picked == null || !mounted) return;
    setState(() {
      _symbolQueue
        ..clear()
        ..addAll(picked);
    });
  }

  Widget _queueListChips() {
    if (_symbolQueue.isEmpty) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 4),
        child: Text(
          'Queue empty — single-pair controls are active below.',
          style: TextStyle(
            fontSize: 11,
            color: ForexAiTokens.textFaint,
            fontStyle: FontStyle.italic,
          ),
        ),
      );
    }
    return Wrap(
      spacing: 6,
      runSpacing: 4,
      children: [
        for (var i = 0; i < _symbolQueue.length; i++)
          _queueSymbolChip(i, _symbolQueue[i]),
      ],
    );
  }

  Widget _queueSymbolChip(int index, String symbol) {
    final isActive = _queueRunning && index == _queueCursor;
    final isDone = _queueRunning && index < _queueCursor;
    final color = isActive
        ? ForexAiTokens.buy
        : isDone
            ? ForexAiTokens.textMuted
            : null;
    return Chip(
      avatar: Container(
        width: 18,
        height: 18,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: isActive
              ? ForexAiTokens.buy
              : isDone
                  ? ForexAiTokens.textMuted
                  : ForexAiTokens.textFaint,
        ),
        alignment: Alignment.center,
        child: Text(
          '${index + 1}',
          style: const TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
      ),
      label: Text(
        symbol,
        style: TextStyle(
          fontSize: 12,
          fontWeight: isActive ? FontWeight.w700 : FontWeight.w500,
          color: color,
          decoration: isDone ? TextDecoration.lineThrough : null,
        ),
      ),
      deleteIcon: const Icon(Icons.close, size: 14),
      onDeleted: _queueRunning
          ? null
          : () => setState(() => _symbolQueue.removeAt(index)),
      visualDensity: VisualDensity.compact,
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
    );
  }

  Widget _queueControlsRow(AsyncValue<EnginesSnapshot> enginesAsync) {
    final backendRunning =
        enginesAsync.valueOrNull?.discoveryRunning ?? false;
    return Row(
      children: [
        FilledButton.icon(
          onPressed: (_queueRunning || backendRunning) ? null : _runQueue,
          icon: const Icon(Icons.play_arrow, size: 16),
          label: Text(_symbolQueue.length == 1
              ? 'Run 1 symbol'
              : 'Run queue (${_symbolQueue.length} symbols)'),
        ),
        const SizedBox(width: 8),
        OutlinedButton.icon(
          style: OutlinedButton.styleFrom(
            foregroundColor:
                _queueRunning ? const Color(0xFFB71C1C) : null,
            side: BorderSide(
              color: _queueRunning
                  ? const Color(0xFFB71C1C)
                  : ForexAiTokens.textFaint,
            ),
          ),
          onPressed: _queueRunning ? _abortQueue : null,
          icon: const Icon(Icons.stop, size: 16),
          label: const Text('Cancel queue'),
        ),
      ],
    );
  }

  Widget _queueProgressRow(AsyncValue<EnginesSnapshot> enginesAsync) {
    final progress = (_queueCursor + 1).clamp(0, _symbolQueue.length);
    final percent =
        _symbolQueue.isEmpty ? 0.0 : progress / _symbolQueue.length;
    final liveSummary = enginesAsync.valueOrNull?.discoverySummary ?? '';
    final activeSymbol =
        (_queueCursor >= 0 && _queueCursor < _symbolQueue.length)
            ? _symbolQueue[_queueCursor]
            : '';
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          _queueRunning
              ? 'Queue progress: ${_queueCursor + 1}/${_symbolQueue.length}'
                  '${activeSymbol.isEmpty ? '' : ' — $activeSymbol'}'
              : _queueSummary,
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
          ),
        ),
        if (_queueRunning) ...[
          const SizedBox(height: 4),
          LinearProgressIndicator(value: percent, minHeight: 3),
          if (liveSummary.isNotEmpty) ...[
            const SizedBox(height: 6),
            Text(
              liveSummary,
              style: const TextStyle(
                fontSize: 11,
                color: ForexAiTokens.textMuted,
              ),
            ),
          ],
        ],
      ],
    );
  }

  void _addToQueue() {
    final symbol = _queueSymbolController.text.trim().toUpperCase();
    if (symbol.isEmpty) return;
    if (_symbolQueue.contains(symbol)) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('$symbol is already in the queue.'),
          duration: const Duration(seconds: 2),
        ),
      );
      return;
    }
    setState(() {
      _symbolQueue.add(symbol);
      _queueSymbolController.clear();
    });
  }

  /// Orchestrate the queue. For each symbol:
  ///   1. POST `/engines/discovery/start` with the symbol + shared
  ///      base TF + higher TFs + GA knobs.
  ///   2. Poll `enginesProvider` (the existing 2 s `/engines/status`
  ///      poller) until `discoveryRunning` flips false — the backend
  ///      went terminal (Succeeded / Failed / Cancelled, all OK to
  ///      advance on).
  ///   3. If the operator clicked Cancel mid-flight, break.
  Future<void> _runQueue() async {
    if (_symbolQueue.isEmpty || _queueRunning) return;
    if (_higherTfs.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
        content: Text('Select at least one higher TF before starting.'),
        duration: Duration(seconds: 2),
      ));
      return;
    }

    setState(() {
      _queueRunning = true;
      _queueAbortRequested = false;
      _queueCursor = 0;
      _queueSummary = '';
    });

    final client = ref.read(backendClientProvider);
    var aborted = false;
    var failed = <String>[];

    for (var i = 0; i < _symbolQueue.length; i++) {
      if (_queueAbortRequested || !mounted) {
        aborted = true;
        break;
      }
      setState(() => _queueCursor = i);
      final symbol = _symbolQueue[i];

      try {
        await client.startDiscovery(
          symbol: symbol,
          baseTf: _queueBaseTf,
          higherTfs: _higherTfs.toList(),
          population: _population,
          generations: _generations,
          maxIndicators: _maxIndicators,
          targetCandidates: _targetCandidates,
          portfolioSize: _portfolioSize,
        );
      } on DioException catch (e) {
        failed.add('$symbol: ${describeError(e)}');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(
            content: Text(
              'Discovery stopped on $symbol — ${describeError(e)}.',
            ),
            backgroundColor: const Color(0xFFB71C1C),
            duration: const Duration(seconds: 4),
          ));
        }
        break;
      } catch (e) {
        failed.add('$symbol: ${describeError(e)}');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(SnackBar(
            content: Text(
              'Discovery stopped on $symbol — ${describeError(e)}.',
            ),
            backgroundColor: const Color(0xFFB71C1C),
            duration: const Duration(seconds: 4),
          ));
        }
        break;
      }

      // Force an immediate refresh so the UI flips from Idle → Running
      // within ~250 ms instead of waiting up to 2 s for the next poll.
      ref.invalidate(enginesProvider);

      // Wait for terminal. The first poll often returns the OLD
      // (Idle) snapshot because the engine hasn't flipped yet; gate
      // on having seen at least one Running observation before we
      // start treating "not running" as terminal. Cap the wait at
      // 2 hours per symbol so a permanently-stuck engine eventually
      // releases the queue.
      var sawRunning = false;
      final loopStart = DateTime.now();
      final deadline = loopStart.add(const Duration(hours: 2));
      // Grace window after `start` for the engine to flip its
      // /engines/status from Idle → Running. If we never observe
      // Running within this window, assume the start was a no-op
      // (404, port flap, etc.) and break out so the queue advances
      // instead of hanging on a phantom run.
      const flipGrace = Duration(seconds: 30);
      while (mounted && !_queueAbortRequested) {
        if (DateTime.now().isAfter(deadline)) {
          failed.add('$symbol: 2h queue-slot deadline exceeded');
          break;
        }
        await Future<void>.delayed(const Duration(seconds: 2));
        ref.invalidate(enginesProvider);
        // Read the LATEST cached value rather than awaiting the future
        // again — the autoDispose timer re-fetches every 2 s on its
        // own, so the cache is fresh-enough for the loop.
        final snap = ref.read(enginesProvider).valueOrNull;
        if (snap == null) continue;
        if (snap.discoveryRunning) {
          sawRunning = true;
          continue;
        }
        if (sawRunning) {
          // Engine went from Running → terminal. Record the final
          // state label so the queue summary shows what happened.
          if (mounted) {
            setState(() => _queueSummary =
                '$symbol → ${snap.discovery} · ${snap.discoverySummary}');
          }
          break;
        }
        if (DateTime.now().difference(loopStart) > flipGrace) {
          failed.add(
              '$symbol: engine never flipped to Running within '
              '${flipGrace.inSeconds}s — check backend logs');
          break;
        }
      }
    }

    if (mounted) {
      setState(() {
        _queueRunning = false;
        _queueCursor = -1;
        _queueSummary = aborted
            ? 'Queue cancelled (${_symbolQueue.length} symbols staged)'
            : failed.isEmpty
                ? 'Queue complete — ran ${_symbolQueue.length} symbols '
                    'on $_queueBaseTf'
                : 'Queue partial — ${failed.length} failures: '
                    '${failed.join('; ')}';
      });
      // A clean run can flash by in 4s, but a failure now carries an
      // actionable diagnosis (which funnel stage rejected everything +
      // the fix) — give the operator time to read it, and a dismiss
      // action so it isn't a transient flash.
      final hadFailures = !aborted && failed.isNotEmpty;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(_queueSummary),
          duration: Duration(seconds: hadFailures ? 18 : 4),
          backgroundColor: hadFailures ? const Color(0xFF7F1D1D) : null,
          showCloseIcon: hadFailures,
        ),
      );
    }
  }

  Future<void> _abortQueue() async {
    setState(() => _queueAbortRequested = true);
    try {
      await ref.read(backendClientProvider).stopDiscovery();
    } on DioException {
      // The cooperative-cancel path already retries internally; if
      // /stop itself errors there's nothing more we can do here.
    }
    ref.invalidate(enginesProvider);
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
            onChanged: _queueRunning ? null : onChanged,
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

/// F-14: compact LIVE STATS panel for an in-flight discovery run.
///
/// Reads the same `EnginesSnapshot` the rest of the screen already
/// polls (no extra request). It surfaces the named counters the
/// backend streams (`generation`, `population`, `candidates`,
/// `filtered_candidates`, `portfolio`, `archived_profitable`, …) as a
/// chip grid with friendly labels, plus a humanised stage line and a
/// thin progress bar from `discoveryPercent`. Renders nothing when the
/// run is idle / has emitted no counters yet.
class _DiscoveryLiveStats extends StatelessWidget {
  final EnginesSnapshot? snapshot;
  const _DiscoveryLiveStats({required this.snapshot});

  /// Snake_case counter name → friendly label. Anything not in the map
  /// falls back to a title-cased version of the raw name, so a new
  /// backend counter still renders (just with a generic label) instead
  /// of being silently dropped.
  static const Map<String, String> _counterLabels = {
    'generation': 'Generation',
    'generations': 'Generations',
    'population': 'Population',
    'candidates': 'Candidates',
    'filtered_candidates': 'Filtered',
    'portfolio': 'Portfolio',
    'archived_profitable': 'Archived profitable',
    'evaluated': 'Evaluated',
    'survivors': 'Survivors',
  };

  /// Pairing rules: render "current / total" when both halves of a
  /// counter pair are present (e.g. `generation` 3 of `generations`
  /// 5 → "Generation · 3 / 5"). Maps the "current" name to its
  /// "total" name.
  static const Map<String, String> _counterPairTotals = {
    'generation': 'generations',
  };

  String _humaniseStage(String raw) {
    if (raw.isEmpty) return '';
    final cleaned = raw.replaceAll('_', ' ').replaceAll('-', ' ').trim();
    if (cleaned.isEmpty) return '';
    return cleaned[0].toUpperCase() + cleaned.substring(1);
  }

  String _labelFor(String name) {
    final mapped = _counterLabels[name];
    if (mapped != null) return mapped;
    // Fallback: title-case the snake_case name.
    return name
        .split('_')
        .where((w) => w.isNotEmpty)
        .map((w) => w[0].toUpperCase() + w.substring(1))
        .join(' ');
  }

  @override
  Widget build(BuildContext context) {
    final snap = snapshot;
    // Only render for an active run with at least one counter to show.
    if (snap == null ||
        !snap.discoveryRunning ||
        snap.discoveryCounters.isEmpty) {
      return const SizedBox.shrink();
    }

    final byName = <String, int>{
      for (final c in snap.discoveryCounters) c.name: c.value,
    };
    // Names consumed as the "total" half of a pair so we don't also
    // render them as a standalone chip.
    final consumedTotals = <String>{};
    for (final entry in _counterPairTotals.entries) {
      if (byName.containsKey(entry.key) && byName.containsKey(entry.value)) {
        consumedTotals.add(entry.value);
      }
    }

    final chips = <Widget>[];
    for (final c in snap.discoveryCounters) {
      if (consumedTotals.contains(c.name)) continue;
      final totalName = _counterPairTotals[c.name];
      final hasPair = totalName != null && byName.containsKey(totalName);
      final valueText =
          hasPair ? '${c.value} / ${byName[totalName]}' : '${c.value}';
      chips.add(_StatChip(label: _labelFor(c.name), value: valueText));
    }

    if (chips.isEmpty) return const SizedBox.shrink();

    final stage = _humaniseStage(snap.discoveryStage);
    final percent = snap.discoveryPercent.clamp(0.0, 1.0);
    final pctLabel = '${(percent * 100).toStringAsFixed(0)}%';

    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(
                  Icons.insights,
                  size: 16,
                  color: ForexAiTokens.accent,
                ),
                const SizedBox(width: 6),
                const Text(
                  'Live stats',
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: ForexAiTokens.fsBody,
                    letterSpacing: 0.4,
                    color: ForexAiTokens.textPrimary,
                  ),
                ),
                const Spacer(),
                Text(
                  pctLabel,
                  style: const TextStyle(
                    fontSize: ForexAiTokens.fsCaption,
                    fontWeight: FontWeight.w700,
                    color: ForexAiTokens.accent,
                    fontFeatures: [FontFeature.tabularFigures()],
                  ),
                ),
              ],
            ),
            if (stage.isNotEmpty) ...[
              const SizedBox(height: 4),
              Text(
                stage,
                style: const TextStyle(
                  fontSize: ForexAiTokens.fsCaption,
                  color: ForexAiTokens.textMuted,
                ),
              ),
            ],
            const SizedBox(height: 8),
            ClipRRect(
              borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
              child: LinearProgressIndicator(
                value: percent,
                minHeight: 4,
                backgroundColor: ForexAiTokens.appBg,
                valueColor: const AlwaysStoppedAnimation<Color>(
                  ForexAiTokens.accent,
                ),
              ),
            ),
            const SizedBox(height: 10),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              children: chips,
            ),
          ],
        ),
      ),
    );
  }
}

/// One labelled stat tile inside the F-14 live-stats panel.
class _StatChip extends StatelessWidget {
  final String label;
  final String value;
  const _StatChip({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: ForexAiTokens.appBg,
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label.toUpperCase(),
            style: const TextStyle(
              fontSize: ForexAiTokens.fsCaption - 1,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.6,
              color: ForexAiTokens.textFaint,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: const TextStyle(
              fontSize: ForexAiTokens.fsBody,
              fontWeight: FontWeight.w800,
              color: ForexAiTokens.textPrimary,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
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
          style: TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
        ),
      );
}

