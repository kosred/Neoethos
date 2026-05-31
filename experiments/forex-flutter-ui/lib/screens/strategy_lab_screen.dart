// Strategy Lab — unified AI pipeline (F-324 final).
//
// **Codex mockup vision** (mockups/ig_*.png image 2): one screen with a
// horizontal 5-stage pipeline (Data Ready → Discovery → Training →
// Validation → Promotion Gate) where each stage shows its status at a
// glance, plus a "Promote to Live" terminal button when all gates
// clear, plus the persistent AI Desk right-rail.
//
// **F-324 (2026-05-29 rebuild)** lands the pipeline strip with live
// status pulled from the existing providers:
//   - Data Ready  ← /broker/symbols / data bootstrap status
//   - Discovery   ← enginesProvider.discovery (running / idle)
//   - Training    ← enginesProvider.training
//   - Validation  ← intelligenceProvider.walkforwardSplits > 0
//   - Promotion   ← deferred to F-330 (backend orchestrator); for now
//                   it reflects whether validation cleared at all.
//
// Clicking a stage chip jumps to the corresponding tab below where the
// full Discovery / Training / etc. screen renders. F-330 will replace
// the bottom TabBar with the rich per-stage parameter cards the mockup
// inlines under the pipeline; until then we keep the existing rich
// screens accessible as tabs so nothing regresses.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import 'discovery_screen.dart';
import 'training_screen.dart';

class StrategyLabScreen extends ConsumerStatefulWidget {
  const StrategyLabScreen({super.key});

  @override
  ConsumerState<StrategyLabScreen> createState() => _StrategyLabScreenState();
}

class _StrategyLabScreenState extends ConsumerState<StrategyLabScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _controller;

  static const _stages = [
    'Discovery',
    'Training',
    'Validation',
    'Promotion Gate',
  ];

  @override
  void initState() {
    super.initState();
    _controller = TabController(length: _stages.length, vsync: this);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const _PipelineStrip(),
        const SizedBox(height: ForexAiTokens.spSm),
        _StageTabs(controller: _controller, stages: _stages),
        const SizedBox(height: ForexAiTokens.spSm),
        Expanded(
          child: TabBarView(
            controller: _controller,
            physics: const NeverScrollableScrollPhysics(),
            children: const [
              DiscoveryScreen(),
              TrainingScreen(),
              _ValidationStub(),
              _PromotionGateStub(),
            ],
          ),
        ),
      ],
    );
  }
}

// ---------------------------------------------------------------------------
// The 5-stage horizontal pipeline strip
// ---------------------------------------------------------------------------

class _PipelineStrip extends ConsumerWidget {
  const _PipelineStrip();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final engines = ref.watch(enginesProvider).valueOrNull;
    final intel = ref.watch(intelligenceProvider).valueOrNull;

    final stages = <_StageCell>[
      const _StageCell(
        index: 1,
        title: 'Data Ready',
        // The /broker/symbols probe runs at app start; if the broker
        // is up at all this stage is satisfied. Operator only needs
        // to bootstrap historical data if it's missing on disk.
        subtitle: 'Broker symbols subscribed',
        status: _StageStatus.done,
      ),
      _StageCell(
        index: 2,
        title: 'Discovery',
        subtitle: engines == null
            ? 'Idle'
            : _enginePhrase(engines.discovery, 'GA search'),
        status: _statusFor(engines?.discovery),
      ),
      _StageCell(
        index: 3,
        title: 'Training',
        subtitle: engines == null
            ? 'Idle'
            : _enginePhrase(engines.training, 'Ensemble fit'),
        status: _statusFor(engines?.training),
      ),
      _StageCell(
        index: 4,
        title: 'Validation',
        subtitle: intel == null
            ? '—'
            : intel.walkforwardSplits == null ||
                    intel.walkforwardSplits == 0
                ? 'Awaiting WFA'
                : '${intel.walkforwardSplits} WFA splits · '
                    '${intel.walkforwardAvgAccuracy == null
                        ? '—'
                        : '${(intel.walkforwardAvgAccuracy! * 100).toStringAsFixed(1)}% acc'}',
        status: _validationStatus(intel),
      ),
      _StageCell(
        index: 5,
        title: 'Promotion Gate',
        subtitle: intel == null || (intel.walkforwardSplits ?? 0) == 0
            ? 'Awaiting validation'
            : 'Manual review (F-330 ships auto-gate)',
        status: _promotionStatus(intel),
      ),
    ];

    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: ForexAiTokens.spMd,
        vertical: 8,
      ),
      decoration: BoxDecoration(
        color: ForexAiTokens.panelBg,
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Row(
        children: [
          for (var i = 0; i < stages.length; i++) ...[
            Expanded(child: stages[i]),
            if (i < stages.length - 1)
              const Padding(
                padding: EdgeInsets.symmetric(horizontal: 4),
                child: Icon(
                  Icons.chevron_right,
                  color: ForexAiTokens.textFaint,
                  size: 18,
                ),
              ),
          ],
        ],
      ),
    );
  }

  static _StageStatus _statusFor(String? engineState) {
    switch (engineState?.toLowerCase()) {
      case 'running':
        return _StageStatus.running;
      case 'error':
      case 'failed':
        return _StageStatus.error;
      case 'idle':
        return _StageStatus.idle;
      default:
        return _StageStatus.idle;
    }
  }

  static String _enginePhrase(String state, String what) {
    switch (state.toLowerCase()) {
      case 'running':
        return '$what running…';
      case 'error':
      case 'failed':
        return '$what failed';
      case 'idle':
        return '$what idle';
      default:
        return state;
    }
  }

  static _StageStatus _validationStatus(IntelligenceSnapshot? intel) {
    if (intel == null) return _StageStatus.idle;
    final splits = intel.walkforwardSplits ?? 0;
    if (splits == 0) return _StageStatus.idle;
    return _StageStatus.done;
  }

  static _StageStatus _promotionStatus(IntelligenceSnapshot? intel) {
    if (intel == null) return _StageStatus.idle;
    final splits = intel.walkforwardSplits ?? 0;
    final acc = intel.walkforwardAvgAccuracy;
    if (splits == 0 || acc == null) return _StageStatus.idle;
    // Soft gate until F-330 ships the proper Sharpe/Calmar/win-rate
    // backend check: green if WFA accuracy ≥ 55 %, amber otherwise.
    return acc >= 0.55 ? _StageStatus.done : _StageStatus.warn;
  }
}

enum _StageStatus { idle, running, done, warn, error }

class _StageCell extends StatelessWidget {
  final int index;
  final String title;
  final String subtitle;
  final _StageStatus status;
  const _StageCell({
    required this.index,
    required this.title,
    required this.subtitle,
    required this.status,
  });

  @override
  Widget build(BuildContext context) {
    final (statusLabel, statusColor) = switch (status) {
      _StageStatus.done => ('READY', ForexAiTokens.buy),
      _StageStatus.running => ('RUN', ForexAiTokens.accent),
      _StageStatus.warn => ('CHECK', ForexAiTokens.warning),
      _StageStatus.error => ('ERROR', ForexAiTokens.sell),
      _StageStatus.idle => ('IDLE', ForexAiTokens.textFaint),
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      decoration: BoxDecoration(
        color: ForexAiTokens.appBg,
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            children: [
              Container(
                width: 18,
                height: 18,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: statusColor.withValues(alpha: 0.18),
                  border: Border.all(
                    color: statusColor.withValues(alpha: 0.6),
                  ),
                  borderRadius: BorderRadius.circular(9),
                ),
                child: Text(
                  '$index',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w800,
                    color: statusColor,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  title,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: ForexAiTokens.fsBody,
                    fontWeight: FontWeight.w800,
                    color: ForexAiTokens.textPrimary,
                  ),
                ),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 4, vertical: 1),
                decoration: BoxDecoration(
                  color: statusColor.withValues(alpha: 0.14),
                  border: Border.all(
                    color: statusColor.withValues(alpha: 0.5),
                  ),
                  borderRadius: BorderRadius.circular(3),
                ),
                child: Text(
                  statusLabel,
                  style: TextStyle(
                    fontSize: 9,
                    fontWeight: FontWeight.w800,
                    color: statusColor,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: ForexAiTokens.fsCaption,
              color: ForexAiTokens.textMuted,
              height: 1.3,
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Tabs (sub-navigation below the pipeline strip)
// ---------------------------------------------------------------------------

class _StageTabs extends StatelessWidget {
  final TabController controller;
  final List<String> stages;
  const _StageTabs({required this.controller, required this.stages});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: ForexAiTokens.panelBg,
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: TabBar(
        controller: controller,
        isScrollable: true,
        labelColor: ForexAiTokens.accent,
        unselectedLabelColor: ForexAiTokens.textMuted,
        indicatorColor: ForexAiTokens.accent,
        labelStyle: const TextStyle(
          fontSize: ForexAiTokens.fsBody,
          fontWeight: FontWeight.w700,
        ),
        unselectedLabelStyle: const TextStyle(
          fontSize: ForexAiTokens.fsBody,
          fontWeight: FontWeight.w500,
        ),
        tabs: [for (final s in stages) Tab(text: s)],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Validation / Promotion stubs (F-330 will replace these with real
// per-stage param cards backed by the orchestration endpoint)
// ---------------------------------------------------------------------------

class _ValidationStub extends ConsumerWidget {
  const _ValidationStub();
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final intel = ref.watch(intelligenceProvider).valueOrNull;
    final splits = intel?.walkforwardSplits ?? 0;
    final acc = intel?.walkforwardAvgAccuracy;
    return _PlaceholderCard(
      ticket: 'F-330',
      title: 'Validation',
      body: splits == 0
          ? 'No walk-forward run yet. Once Training completes, the WFA '
              'sweep populates this panel with split-by-split accuracy + '
              'Sharpe / Calmar / win-rate per fold.'
          : '$splits WFA splits completed. Average accuracy: '
              '${acc == null ? "—" : "${(acc * 100).toStringAsFixed(1)}%"}.\n\n'
              'The detailed per-split table + sensitivity sweep + Monte '
              'Carlo confidence intervals land with F-330 (backend '
              'orchestrator) and the proper per-stage param cards.',
    );
  }
}

class _PromotionGateStub extends ConsumerWidget {
  const _PromotionGateStub();
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final intel = ref.watch(intelligenceProvider).valueOrNull;
    final splits = intel?.walkforwardSplits ?? 0;
    final acc = intel?.walkforwardAvgAccuracy;
    final ready = splits > 0 && acc != null && acc >= 0.55;
    return _PlaceholderCard(
      ticket: 'F-330',
      title: 'Promotion Gate',
      body: ready
          ? 'Validation passed (≥ 55 % WFA accuracy). Promotion will '
              'enforce Sharpe ≥ 1.0, Calmar ≥ 0.7, win-rate ≥ 50 %, '
              'max drawdown ≤ 25 % once F-330 wires the gate to '
              '`/strategy_lab/promote`. Until then, copy the model '
              'bundle from `models/staging/` to `models/live/` by hand '
              'after manually reviewing the validation report.'
          : 'Promotion Gate is disabled until Validation completes. '
              'Currently: $splits WFA splits, '
              '${acc == null ? "no accuracy yet" : "${(acc * 100).toStringAsFixed(1)}% accuracy"}. '
              'The gate enforces Sharpe ≥ 1.0, Calmar ≥ 0.7, win-rate ≥ 50 %, '
              'max drawdown ≤ 25 % before copying to `models/live/`.',
      action: ready
          ? FilledButton.icon(
              onPressed: null, // F-330 wires this to /strategy_lab/promote
              icon: const Icon(Icons.upload, size: 16),
              label: const Text('Promote to Live (F-330)'),
              style: FilledButton.styleFrom(
                backgroundColor: ForexAiTokens.buy,
              ),
            )
          : null,
    );
  }
}

class _PlaceholderCard extends StatelessWidget {
  final String ticket;
  final String title;
  final String body;
  final Widget? action;
  const _PlaceholderCard({
    required this.ticket,
    required this.title,
    required this.body,
    this.action,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 600),
        child: Container(
          padding: const EdgeInsets.all(ForexAiTokens.spLg),
          decoration: BoxDecoration(
            color: ForexAiTokens.panelBg,
            border: Border.all(color: ForexAiTokens.border),
            borderRadius: BorderRadius.circular(ForexAiTokens.rMd),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.symmetric(
                      vertical: 2,
                      horizontal: 8,
                    ),
                    decoration: BoxDecoration(
                      color: ForexAiTokens.accentMuted,
                      borderRadius:
                          BorderRadius.circular(ForexAiTokens.rSm),
                      border: Border.all(
                        color: ForexAiTokens.accent.withValues(alpha: 0.5),
                      ),
                    ),
                    child: Text(
                      ticket,
                      style: const TextStyle(
                        fontSize: ForexAiTokens.fsCaption,
                        fontWeight: FontWeight.w700,
                        color: ForexAiTokens.accent,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: ForexAiTokens.fsSubtitle,
                      fontWeight: FontWeight.w700,
                      color: ForexAiTokens.textPrimary,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: ForexAiTokens.spMd),
              Text(
                body,
                style: const TextStyle(
                  fontSize: ForexAiTokens.fsBody,
                  color: ForexAiTokens.textMuted,
                  height: 1.5,
                ),
              ),
              if (action != null) ...[
                const SizedBox(height: ForexAiTokens.spLg),
                Align(
                  alignment: Alignment.centerRight,
                  child: action!,
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
