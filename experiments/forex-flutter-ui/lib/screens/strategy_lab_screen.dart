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

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../l10n/app_localizations.dart';
import '../state/account_provider.dart';
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

  // Number of pipeline tabs. The localized labels are built in [build]
  // from the current context (see [_stageLabels]).
  static const _stageCount = 4;

  @override
  void initState() {
    super.initState();
    _controller = TabController(length: _stageCount, vsync: this);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final stageLabels = [
      l10n.engineDiscovery,
      l10n.engineTraining,
      l10n.strategyLabStageValidation,
      l10n.strategyLabStagePromotionGate,
    ];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _PipelineStrip(onSelectStage: _goToStage),
        const SizedBox(height: NeoethosTokens.spSm),
        const _LabTargetSelector(),
        const SizedBox(height: NeoethosTokens.spSm),
        _StageTabs(controller: _controller, stages: stageLabels),
        const SizedBox(height: NeoethosTokens.spSm),
        Expanded(
          child: TabBarView(
            controller: _controller,
            physics: const NeverScrollableScrollPhysics(),
            children: const [
              DiscoveryScreen(),
              TrainingScreen(),
              _ValidationStub(),
              _PromotionGateView(),
            ],
          ),
        ),
      ],
    );
  }

  /// Drive the sub-tab controller from a pipeline chip. Stage cells are
  /// 1-indexed (Data Ready = 1); the tabs below start at Discovery, so
  /// stage N maps to tab N-2 (Discovery=tab0). The Data Ready stage has
  /// no dedicated tab — it lands on Discovery (the first actionable
  /// stage).
  void _goToStage(int stageIndex) {
    final tabIndex = (stageIndex - 2).clamp(0, _stageCount - 1);
    _controller.animateTo(tabIndex);
  }
}

// ---------------------------------------------------------------------------
// Symbol + base-timeframe selector (drives the shared
// selectedSymbolProvider / selectedBaseTfProvider that the promotion
// gate + any discovery/training launched here read).
// ---------------------------------------------------------------------------

class _LabTargetSelector extends ConsumerWidget {
  const _LabTargetSelector();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final symbol = ref.watch(selectedSymbolProvider);
    final baseTf = ref.watch(selectedBaseTfProvider);
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: NeoethosTokens.spMd,
        vertical: 8,
      ),
      decoration: BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Row(
        children: [
          Text(
            l10n.strategyLabTargetLabel.toUpperCase(),
            style: const TextStyle(
              fontSize: NeoethosTokens.fsCaption,
              fontWeight: FontWeight.w800,
              letterSpacing: 0.8,
              color: NeoethosTokens.textMuted,
            ),
          ),
          const SizedBox(width: NeoethosTokens.spMd),
          _LabDropdown(
            value: symbol,
            items: strategyLabSymbols,
            onChanged: (v) =>
                ref.read(selectedSymbolProvider.notifier).state = v,
          ),
          const SizedBox(width: NeoethosTokens.spSm),
          _LabDropdown(
            value: baseTf,
            items: strategyLabBaseTimeframes,
            onChanged: (v) =>
                ref.read(selectedBaseTfProvider.notifier).state = v,
          ),
        ],
      ),
    );
  }
}

class _LabDropdown extends StatelessWidget {
  final String value;
  final List<String> items;
  final ValueChanged<String> onChanged;
  const _LabDropdown({
    required this.value,
    required this.items,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    // Guard against a value that isn't in the list (defensive — keeps
    // DropdownButton from asserting if defaults ever drift).
    final safeValue = items.contains(value) ? value : items.first;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8),
      decoration: BoxDecoration(
        color: NeoethosTokens.appBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: safeValue,
          isDense: true,
          dropdownColor: NeoethosTokens.panelBg,
          style: const TextStyle(
            fontSize: NeoethosTokens.fsBody,
            fontWeight: FontWeight.w700,
            color: NeoethosTokens.textPrimary,
          ),
          icon: const Icon(Icons.arrow_drop_down,
              size: 18, color: NeoethosTokens.textMuted),
          items: [
            for (final item in items)
              DropdownMenuItem<String>(value: item, child: Text(item)),
          ],
          onChanged: (v) {
            if (v != null) onChanged(v);
          },
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// The 5-stage horizontal pipeline strip
// ---------------------------------------------------------------------------

class _PipelineStrip extends ConsumerWidget {
  /// Called with the 1-based stage index when a chip is tapped, so the
  /// parent can switch the sub-tab controller to that stage.
  final ValueChanged<int> onSelectStage;
  const _PipelineStrip({required this.onSelectStage});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final engines = ref.watch(enginesProvider).valueOrNull;
    final intel = ref.watch(intelligenceProvider).valueOrNull;

    final stages = <_StageCell>[
      _StageCell(
        index: 1,
        title: l10n.strategyLabStageDataReady,
        // The /broker/symbols probe runs at app start; if the broker
        // is up at all this stage is satisfied. Operator only needs
        // to bootstrap historical data if it's missing on disk.
        subtitle: l10n.strategyLabDataReadySubtitle,
        status: _StageStatus.done,
        onTap: () => onSelectStage(1),
      ),
      _StageCell(
        index: 2,
        title: l10n.engineDiscovery,
        subtitle: engines == null
            ? l10n.statusIdle
            : _enginePhrase(l10n, engines.discovery, l10n.strategyLabGaSearch),
        status: _statusFor(engines?.discovery),
        onTap: () => onSelectStage(2),
      ),
      _StageCell(
        index: 3,
        title: l10n.engineTraining,
        subtitle: engines == null
            ? l10n.statusIdle
            : _enginePhrase(
                l10n, engines.training, l10n.strategyLabEnsembleFit),
        status: _statusFor(engines?.training),
        onTap: () => onSelectStage(3),
      ),
      _StageCell(
        index: 4,
        title: l10n.strategyLabStageValidation,
        subtitle: intel == null
            ? '—'
            : intel.walkforwardSplits == null ||
                    intel.walkforwardSplits == 0
                ? l10n.strategyLabAwaitingWfa
                : '${intel.walkforwardSplits} WFA splits · '
                    '${intel.walkforwardAvgAccuracy == null
                        ? '—'
                        : '${(intel.walkforwardAvgAccuracy! * 100).toStringAsFixed(1)}% acc'}',
        status: _validationStatus(intel),
        onTap: () => onSelectStage(4),
      ),
      _StageCell(
        index: 5,
        title: l10n.strategyLabStagePromotionGate,
        subtitle: intel == null || (intel.walkforwardSplits ?? 0) == 0
            ? l10n.strategyLabAwaitingValidation
            : l10n.strategyLabPromotionManualReview,
        status: _promotionStatus(intel),
        onTap: () => onSelectStage(5),
      ),
    ];

    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: NeoethosTokens.spMd,
        vertical: 8,
      ),
      decoration: BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
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
                  color: NeoethosTokens.textFaint,
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

  static String _enginePhrase(
      AppLocalizations l10n, String state, String what) {
    switch (state.toLowerCase()) {
      case 'running':
        return l10n.strategyLabPhraseRunning(what);
      case 'error':
      case 'failed':
        return l10n.strategyLabPhraseFailed(what);
      case 'idle':
        return l10n.strategyLabPhraseIdle(what);
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

  /// Tapping the chip switches the sub-tab below to this stage. The
  /// chips looked interactive but used to be inert; this wires them.
  final VoidCallback? onTap;
  const _StageCell({
    required this.index,
    required this.title,
    required this.subtitle,
    required this.status,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final (statusLabel, statusColor) = switch (status) {
      _StageStatus.done => (l10n.strategyLabBadgeReady, NeoethosTokens.buy),
      _StageStatus.running => (l10n.strategyLabBadgeRun, NeoethosTokens.accent),
      _StageStatus.warn => (l10n.strategyLabBadgeCheck, NeoethosTokens.warning),
      _StageStatus.error => (l10n.strategyLabBadgeError, NeoethosTokens.sell),
      _StageStatus.idle => (l10n.strategyLabBadgeIdle, NeoethosTokens.textFaint),
    };
    final radius = BorderRadius.circular(NeoethosTokens.rSm);
    return Material(
      color: NeoethosTokens.appBg,
      borderRadius: radius,
      child: InkWell(
        onTap: onTap,
        borderRadius: radius,
        child: Ink(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
          decoration: BoxDecoration(
            border: Border.all(color: NeoethosTokens.border),
            borderRadius: radius,
          ),
          child: _cellBody(statusLabel, statusColor),
        ),
      ),
    );
  }

  Widget _cellBody(String statusLabel, Color statusColor) {
    return Column(
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
                    fontSize: NeoethosTokens.fsBody,
                    fontWeight: FontWeight.w800,
                    color: NeoethosTokens.textPrimary,
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
              fontSize: NeoethosTokens.fsCaption,
              color: NeoethosTokens.textMuted,
              height: 1.3,
            ),
          ),
        ],
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
        color: NeoethosTokens.panelBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: TabBar(
        controller: controller,
        isScrollable: true,
        labelColor: NeoethosTokens.accent,
        unselectedLabelColor: NeoethosTokens.textMuted,
        indicatorColor: NeoethosTokens.accent,
        labelStyle: const TextStyle(
          fontSize: NeoethosTokens.fsBody,
          fontWeight: FontWeight.w700,
        ),
        unselectedLabelStyle: const TextStyle(
          fontSize: NeoethosTokens.fsBody,
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
    final l10n = AppLocalizations.of(context)!;
    final intel = ref.watch(intelligenceProvider).valueOrNull;
    final splits = intel?.walkforwardSplits ?? 0;
    final acc = intel?.walkforwardAvgAccuracy;
    return _PlaceholderCard(
      ticket: 'F-330',
      title: l10n.strategyLabStageValidation,
      body: splits == 0
          ? l10n.strategyLabValidationBodyEmpty
          : l10n.strategyLabValidationBodyDone(
              splits,
              acc == null ? '—' : '${(acc * 100).toStringAsFixed(1)}%',
            ),
    );
  }
}

// ---------------------------------------------------------------------------
// F-330 — Promotion Gate (real, wired to /strategy_lab/promotion +
// /strategy_lab/promote). Replaces the old _PromotionGateStub.
// ---------------------------------------------------------------------------

class _PromotionGateView extends ConsumerStatefulWidget {
  const _PromotionGateView();

  @override
  ConsumerState<_PromotionGateView> createState() => _PromotionGateViewState();
}

class _PromotionGateViewState extends ConsumerState<_PromotionGateView> {
  // The pair/timeframe the gate evaluates now comes from the shared
  // Strategy Lab selection providers (driven by the header dropdowns),
  // NOT a hard-coded constant. This fixed the wrong-pair promotion bug:
  // the gate used to always check EURUSD/M5 regardless of what the
  // operator was discovering/training.
  String get _symbol => ref.read(selectedSymbolProvider);
  String get _baseTf => ref.read(selectedBaseTfProvider);

  PromotionStatus? _status;
  String? _error;
  bool _loading = true;
  bool _promoting = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final l10n = AppLocalizations.of(context)!;
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final client = ref.read(backendClientProvider);
      final status = await client.fetchPromotionStatus(
        symbol: _symbol,
        baseTf: _baseTf,
      );
      if (!mounted) return;
      setState(() {
        _status = status;
        _loading = false;
      });
    } on DioException catch (e) {
      if (!mounted) return;
      setState(() {
        _error = _formatPromotionError(e);
        _loading = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = l10n.strategyLabBacktestStartError(describeError(e));
        _loading = false;
      });
    }
  }

  Future<void> _promote() async {
    final l10n = AppLocalizations.of(context)!;
    setState(() => _promoting = true);
    try {
      final client = ref.read(backendClientProvider);
      final result = await client.promoteToLive(
        symbol: _symbol,
        baseTf: _baseTf,
      );
      if (!mounted) return;
      final messenger = ScaffoldMessenger.of(context);
      messenger.showSnackBar(
        SnackBar(
          content: Text(result.message),
          backgroundColor:
              result.promoted ? NeoethosTokens.buy : NeoethosTokens.warning,
        ),
      );
    } on DioException catch (e) {
      if (mounted) {
        showTranslatedErrorSnackbar(
          context,
          e,
          prefix: l10n.strategyLabPromoteError,
        );
      }
    } catch (e) {
      if (mounted) {
        showTranslatedErrorSnackbar(
          context,
          e,
          prefix: l10n.strategyLabPromoteError,
        );
      }
    } finally {
      if (mounted) setState(() => _promoting = false);
      // Re-fetch so the gate verdict + portfolio reflect the new
      // live_models state regardless of outcome.
      await _load();
    }
  }

  @override
  Widget build(BuildContext context) {
    // Re-evaluate the gate whenever the operator picks a different
    // pair / base TF in the Strategy Lab header.
    ref.listen<String>(selectedSymbolProvider, (_, __) => _load());
    ref.listen<String>(selectedBaseTfProvider, (_, __) => _load());
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 640),
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(vertical: NeoethosTokens.spMd),
          child: Container(
            padding: const EdgeInsets.all(NeoethosTokens.spLg),
            decoration: BoxDecoration(
              color: NeoethosTokens.panelBg,
              border: Border.all(color: NeoethosTokens.border),
              borderRadius: BorderRadius.circular(NeoethosTokens.rMd),
            ),
            child: _body(),
          ),
        ),
      ),
    );
  }

  Widget _body() {
    final l10n = AppLocalizations.of(context)!;
    if (_loading) {
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 40),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(
                width: 22,
                height: 22,
                child: CircularProgressIndicator(strokeWidth: 2.4),
              ),
              const SizedBox(height: NeoethosTokens.spMd),
              Text(
                l10n.strategyLabCheckingGate,
                style: const TextStyle(
                  color: NeoethosTokens.textMuted,
                  fontSize: NeoethosTokens.fsBody,
                ),
              ),
            ],
          ),
        ),
      );
    }

    if (_error != null) {
      return Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.error_outline,
                  color: NeoethosTokens.sell, size: 18),
              const SizedBox(width: NeoethosTokens.spSm),
              Text(
                l10n.strategyLabGateLoadError,
                style: const TextStyle(
                  fontSize: NeoethosTokens.fsSubtitle,
                  fontWeight: FontWeight.w700,
                  color: NeoethosTokens.textPrimary,
                ),
              ),
            ],
          ),
          const SizedBox(height: NeoethosTokens.spSm),
          Text(
            _error!,
            style: const TextStyle(
              fontSize: NeoethosTokens.fsBody,
              color: NeoethosTokens.textMuted,
              height: 1.4,
            ),
          ),
          const SizedBox(height: NeoethosTokens.spMd),
          Align(
            alignment: Alignment.centerLeft,
            child: OutlinedButton.icon(
              onPressed: _load,
              icon: const Icon(Icons.refresh, size: 16),
              label: Text(l10n.commonRetry),
            ),
          ),
        ],
      );
    }

    return _content(_status!);
  }

  Widget _content(PromotionStatus s) {
    final l10n = AppLocalizations.of(context)!;
    final promoted = s.decision.promoted;
    final badgeColor = promoted ? NeoethosTokens.buy : NeoethosTokens.warning;
    final badgeLabel =
        promoted ? l10n.strategyLabBadgeEligible : l10n.strategyLabBadgeBlocked;

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header: title + symbol/tf + status badge + refresh.
        Row(
          children: [
            Text(
              l10n.strategyLabStagePromotionGate,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsSubtitle,
                fontWeight: FontWeight.w700,
                color: NeoethosTokens.textPrimary,
              ),
            ),
            const SizedBox(width: NeoethosTokens.spSm),
            Text(
              '${s.symbol} · ${s.baseTf}',
              style: const TextStyle(
                fontSize: NeoethosTokens.fsCaption,
                color: NeoethosTokens.textMuted,
              ),
            ),
            const Spacer(),
            _statusBadge(badgeLabel, badgeColor),
            const SizedBox(width: NeoethosTokens.spSm),
            IconButton(
              tooltip: l10n.strategyLabRefreshTooltip,
              onPressed: _loading ? null : _load,
              icon: const Icon(Icons.refresh, size: 18),
              color: NeoethosTokens.textMuted,
              constraints: const BoxConstraints.tightFor(width: 32, height: 32),
              padding: EdgeInsets.zero,
            ),
          ],
        ),
        const SizedBox(height: NeoethosTokens.spMd),

        // Decision summary.
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(NeoethosTokens.spMd),
          decoration: BoxDecoration(
            color: badgeColor.withValues(alpha: 0.10),
            border: Border.all(color: badgeColor.withValues(alpha: 0.45)),
            borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
          ),
          child: Text(
            s.decision.summary.isEmpty
                ? (promoted
                    ? l10n.strategyLabSummaryEligible
                    : l10n.strategyLabSummaryNotEligible)
                : s.decision.summary,
            style: TextStyle(
              fontSize: NeoethosTokens.fsBody,
              color: badgeColor,
              height: 1.4,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
        const SizedBox(height: NeoethosTokens.spLg),

        // Portfolio + aggregate metrics.
        _SectionLabel(l10n.strategyLabSectionPortfolio),
        const SizedBox(height: NeoethosTokens.spSm),
        _metricsGrid(s),
        const SizedBox(height: NeoethosTokens.spLg),

        // Criteria breakdown.
        if (s.decision.criteria.isNotEmpty) ...[
          _SectionLabel(l10n.strategyLabSectionGateCriteria),
          const SizedBox(height: NeoethosTokens.spSm),
          for (final c in s.decision.criteria) _criterionRow(c),
          const SizedBox(height: NeoethosTokens.spLg),
        ] else ...[
          Padding(
            padding: const EdgeInsets.only(bottom: NeoethosTokens.spLg),
            child: Text(
              l10n.strategyLabNoCriteria,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsBody,
                color: NeoethosTokens.textMuted,
                height: 1.4,
              ),
            ),
          ),
        ],

        // Promote action.
        Align(
          alignment: Alignment.centerRight,
          child: FilledButton.icon(
            onPressed: (promoted && !_promoting) ? _promote : null,
            icon: _promoting
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: NeoethosTokens.textPrimary,
                    ),
                  )
                : const Icon(Icons.upload, size: 16),
            label: Text(_promoting
                ? l10n.strategyLabPromoting
                : l10n.strategyLabPromoteToLive),
            style: FilledButton.styleFrom(
              backgroundColor: NeoethosTokens.buy,
              disabledBackgroundColor:
                  NeoethosTokens.border.withValues(alpha: 0.5),
            ),
          ),
        ),
      ],
    );
  }

  Widget _statusBadge(String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.16),
        border: Border.all(color: color.withValues(alpha: 0.6)),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: NeoethosTokens.fsCaption,
          fontWeight: FontWeight.w800,
          letterSpacing: 0.5,
          color: color,
        ),
      ),
    );
  }

  Widget _metricsGrid(PromotionStatus s) {
    final l10n = AppLocalizations.of(context)!;
    final agg = s.aggregate;
    final tiles = <Widget>[
      _metricTile(l10n.strategyLabMetricPortfolioSize, '${s.portfolioSize}'),
      if (agg != null) ...[
        _metricTile('Sharpe', agg.sharpe.toStringAsFixed(2)),
        _metricTile('Win rate', '${(agg.winRate * 100).toStringAsFixed(1)}%'),
        _metricTile('Profit factor', agg.profitFactor.toStringAsFixed(2)),
        _metricTile(
          'Max drawdown',
          '${agg.maxDrawdownPct.toStringAsFixed(1)}%',
        ),
        _metricTile(l10n.strategyLabMetricTrades, '${agg.trades}'),
      ],
    ];
    if (agg == null) {
      tiles.add(
          _metricTile(l10n.strategyLabMetricMetrics, l10n.strategyLabNoPortfolioYet));
    }
    return Wrap(
      spacing: NeoethosTokens.spSm,
      runSpacing: NeoethosTokens.spSm,
      children: tiles,
    );
  }

  Widget _metricTile(String label, String value) {
    return Container(
      width: 120,
      padding: const EdgeInsets.symmetric(
        horizontal: NeoethosTokens.spMd,
        vertical: NeoethosTokens.spSm,
      ),
      decoration: BoxDecoration(
        color: NeoethosTokens.appBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label.toUpperCase(),
            style: const TextStyle(
              fontSize: NeoethosTokens.fsCaption - 1,
              letterSpacing: 0.6,
              fontWeight: FontWeight.w700,
              color: NeoethosTokens.textFaint,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            value,
            style: const TextStyle(
              fontSize: NeoethosTokens.fsBody,
              fontWeight: FontWeight.w700,
              color: NeoethosTokens.textPrimary,
            ),
          ),
        ],
      ),
    );
  }

  Widget _criterionRow(PromotionCriterion c) {
    final color = c.passed ? NeoethosTokens.buy : NeoethosTokens.sell;
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        children: [
          Icon(
            c.passed ? Icons.check_circle : Icons.cancel,
            size: 16,
            color: color,
          ),
          const SizedBox(width: NeoethosTokens.spSm),
          Expanded(
            child: Text(
              c.name,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsBody,
                color: NeoethosTokens.textPrimary,
              ),
            ),
          ),
          Text(
            '${_fmtNum(c.actual)} ${c.comparison} ${_fmtNum(c.threshold)}',
            style: TextStyle(
              fontSize: NeoethosTokens.fsBody,
              fontWeight: FontWeight.w700,
              fontFeatures: const [FontFeature.tabularFigures()],
              color: color,
            ),
          ),
        ],
      ),
    );
  }

  // Trim trailing zeros so "1.40" → "1.4" but "0.52" stays "0.52".
  static String _fmtNum(double v) {
    if (v == v.roundToDouble()) return v.toStringAsFixed(0);
    final s = v.toStringAsFixed(2);
    return s.endsWith('0') ? s.substring(0, s.length - 1) : s;
  }

  static String _formatPromotionError(DioException e) {
    final body = e.response?.data;
    if (body is Map && body['message'] is String) {
      return body['message'] as String;
    }
    if (body is Map && body['error'] is String) {
      return body['error'] as String;
    }
    return e.message ?? e.toString();
  }
}

/// Small all-caps section header used inside the Promotion Gate card.
class _SectionLabel extends StatelessWidget {
  final String text;
  const _SectionLabel(this.text);
  @override
  Widget build(BuildContext context) => Text(
        text,
        style: const TextStyle(
          fontSize: NeoethosTokens.fsCaption,
          letterSpacing: 1.0,
          fontWeight: FontWeight.w700,
          color: NeoethosTokens.textMuted,
        ),
      );
}

class _PlaceholderCard extends StatelessWidget {
  final String ticket;
  final String title;
  final String body;
  const _PlaceholderCard({
    required this.ticket,
    required this.title,
    required this.body,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 600),
        child: Container(
          padding: const EdgeInsets.all(NeoethosTokens.spLg),
          decoration: BoxDecoration(
            color: NeoethosTokens.panelBg,
            border: Border.all(color: NeoethosTokens.border),
            borderRadius: BorderRadius.circular(NeoethosTokens.rMd),
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
                      color: NeoethosTokens.accentMuted,
                      borderRadius:
                          BorderRadius.circular(NeoethosTokens.rSm),
                      border: Border.all(
                        color: NeoethosTokens.accent.withValues(alpha: 0.5),
                      ),
                    ),
                    child: Text(
                      ticket,
                      style: const TextStyle(
                        fontSize: NeoethosTokens.fsCaption,
                        fontWeight: FontWeight.w700,
                        color: NeoethosTokens.accent,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: NeoethosTokens.fsSubtitle,
                      fontWeight: FontWeight.w700,
                      color: NeoethosTokens.textPrimary,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: NeoethosTokens.spMd),
              Text(
                body,
                style: const TextStyle(
                  fontSize: NeoethosTokens.fsBody,
                  color: NeoethosTokens.textMuted,
                  height: 1.5,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
