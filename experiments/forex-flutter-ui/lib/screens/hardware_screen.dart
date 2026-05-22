import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

class HardwareScreen extends ConsumerWidget {
  const HardwareScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(hardwareProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Hardware',
            subtitle: 'CPU / RAM / GPU snapshot from the Rust backend',
          ),
          async.when(
            data: (h) => _Body(snapshot: h),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
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
              _Row('Model', snapshot.cpuModel),
              _Row(
                'Cores',
                '${snapshot.cpuCoresPhysical} physical · '
                    '${snapshot.cpuCoresLogical} logical',
              ),
              _Row(
                'Average load',
                pctFmt.format(snapshot.cpuLoadAvg),
                accent: snapshot.cpuLoadAvg > 0.85
                    ? ForexAiTokens.sell
                    : snapshot.cpuLoadAvg > 0.50
                        ? ForexAiTokens.warning
                        : ForexAiTokens.buy,
              ),
            ],
          ),
        ),
        SectionCard(
          title: 'Memory',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row('Total', '${mbFmt.format(snapshot.ramTotalMb)} MB'),
              _Row('Used', '${mbFmt.format(snapshot.ramUsedMb)} MB'),
              _Row('Available', '${mbFmt.format(snapshot.ramAvailableMb)} MB'),
            ],
          ),
        ),
        SectionCard(
          title: 'GPU',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row('Detected', snapshot.gpuAvailable ? 'Yes' : 'No'),
              _Row('Name', snapshot.gpuName),
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
                  color: ForexAiTokens.textMuted,
                ),
              ),
            ),
            Expanded(
              child: Text(
                value,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: accent ?? ForexAiTokens.textPrimary,
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
  Widget build(BuildContext context) => const Padding(
        padding: EdgeInsets.symmetric(vertical: 16),
        child: Row(
          children: [
            SizedBox(
              width: 14,
              height: 14,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
            SizedBox(width: 8),
            Text(
              'Probing hardware…',
              style: TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
            ),
          ],
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
