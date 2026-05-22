import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

class RiskScreen extends ConsumerWidget {
  const RiskScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(riskProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Risk Settings',
            subtitle:
                'Prop-firm caps · enforced by the Rust trading session',
          ),
          async.when(
            data: (r) => _Body(snapshot: r),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
          ),
        ],
      ),
    );
  }
}

class _Body extends StatelessWidget {
  final RiskSnapshot snapshot;
  const _Body({required this.snapshot});

  @override
  Widget build(BuildContext context) {
    final pctFmt = NumberFormat.percentPattern('en_US')
      ..maximumFractionDigits = 2
      ..minimumFractionDigits = 2;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: 'Drawdown Limits',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row('Daily drawdown limit',
                  pctFmt.format(snapshot.dailyDrawdownLimit)),
              _Row('Total drawdown limit',
                  pctFmt.format(snapshot.totalDrawdownLimit)),
            ],
          ),
        ),
        SectionCard(
          title: 'Per-Trade Risk',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row('Current per-trade risk',
                  pctFmt.format(snapshot.riskPerTrade)),
              _Row('Min allowed', pctFmt.format(snapshot.minRiskPerTrade)),
              _Row('Max allowed', pctFmt.format(snapshot.maxRiskPerTrade)),
              _Row('Max lot size',
                  '${snapshot.maxLotSize.toStringAsFixed(2)} lots'),
            ],
          ),
        ),
        SectionCard(
          title: 'Safety Rails',
          child: _Row(
            'Stop-loss required',
            snapshot.requireStopLoss ? 'YES (enforced)' : 'NO (relaxed)',
            accent: snapshot.requireStopLoss
                ? ForexAiTokens.buy
                : ForexAiTokens.warning,
          ),
        ),
        const SectionCard(
          title: 'Editing',
          child: Text(
            'Read-only in this build. Live edits land when the '
            'POST /risk endpoint ships. Until then, edit config.yaml '
            'and restart neoethos-app --server.',
            style: TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
          ),
        ),
      ],
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
            Text(
              value,
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: accent ?? ForexAiTokens.textPrimary,
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
          'Loading risk caps…',
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
