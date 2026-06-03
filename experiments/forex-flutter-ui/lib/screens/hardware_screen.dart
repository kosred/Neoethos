import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../l10n/app_localizations.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/backend_error_widget.dart';
import '_placeholder.dart';

class HardwareScreen extends ConsumerWidget {
  const HardwareScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final async = ref.watch(hardwareProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          ViewHeader(
            title: l10n.hardwareTitle,
            subtitle: l10n.hardwareSubtitle,
          ),
          async.when(
            data: (h) => _Body(snapshot: h),
            loading: () => const _Loading(),
            error: (err, _) => BackendErrorWidget(
                    error: err, title: l10n.hardwareStatusUnavailable),
          ),
        ],
      ),
    );
  }
}

class _Body extends StatelessWidget {
  final HardwareSnapshot snapshot;
  const _Body({required this.snapshot});

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final mbFmt = NumberFormat('#,##0', 'en_US');
    final pctFmt = NumberFormat.percentPattern('en_US')
      ..maximumFractionDigits = 1;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: 'CPU',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row(l10n.hardwareRowModel, snapshot.cpuModel),
              _Row(
                'Cores',
                l10n.hardwareCoresValue(
                  snapshot.cpuCoresPhysical,
                  snapshot.cpuCoresLogical,
                ),
              ),
              _Row(
                l10n.hardwareRowAverageLoad,
                pctFmt.format(snapshot.cpuLoadAvg),
                accent: snapshot.cpuLoadAvg > 0.85
                    ? NeoethosTokens.sell
                    : snapshot.cpuLoadAvg > 0.50
                        ? NeoethosTokens.warning
                        : NeoethosTokens.buy,
              ),
            ],
          ),
        ),
        SectionCard(
          title: l10n.hardwareMemory,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row(l10n.hardwareRowTotal, '${mbFmt.format(snapshot.ramTotalMb)} MB'),
              _Row(l10n.hardwareRowUsed, '${mbFmt.format(snapshot.ramUsedMb)} MB'),
              _Row(l10n.hardwareRowAvailable,
                  '${mbFmt.format(snapshot.ramAvailableMb)} MB'),
            ],
          ),
        ),
        SectionCard(
          title: 'GPU',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row(l10n.hardwareRowDetected,
                  snapshot.gpuAvailable ? l10n.hardwareYes : l10n.hardwareNo),
              _Row(l10n.hardwareRowName, snapshot.gpuName),
            ],
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
              width: 140,
              child: Text(
                label,
                style: const TextStyle(
                  fontSize: 12,
                  color: NeoethosTokens.textMuted,
                ),
              ),
            ),
            Expanded(
              child: Text(
                value,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: accent ?? NeoethosTokens.textPrimary,
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
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16),
      child: Row(
        children: [
          const SizedBox(
            width: 14,
            height: 14,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
          const SizedBox(width: 8),
          Text(
            l10n.hardwareProbing,
            style: const TextStyle(
                color: NeoethosTokens.textMuted, fontSize: 12),
          ),
        ],
      ),
    );
  }
}

