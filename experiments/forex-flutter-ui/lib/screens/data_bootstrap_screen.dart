import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

class DataBootstrapScreen extends ConsumerWidget {
  const DataBootstrapScreen({super.key});
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final async = ref.watch(dataBootstrapProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Data Bootstrap',
            subtitle: 'Local OHLCV inventory · historical download',
          ),
          async.when(
            data: (d) => _Body(snapshot: d),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
          ),
        ],
      ),
    );
  }
}

class _Body extends StatelessWidget {
  final DataBootstrapSnapshot snapshot;
  const _Body({required this.snapshot});

  @override
  Widget build(BuildContext context) {
    final dtFmt = DateFormat('yyyy-MM-dd HH:mm');
    final mtime = snapshot.lastTouchedUnixMs == null
        ? '—'
        : dtFmt.format(
            DateTime.fromMillisecondsSinceEpoch(snapshot.lastTouchedUnixMs!));
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: 'Inventory',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _Row('Data directory', snapshot.dataDir),
              _Row(
                'Directory exists',
                snapshot.dataDirExists ? 'YES' : 'NO',
                accent: snapshot.dataDirExists
                    ? ForexAiTokens.buy
                    : ForexAiTokens.sell,
              ),
              _Row('Files', '${snapshot.fileCount}'),
              _Row('Last touched', mtime),
              _Row(
                'Symbols mapped',
                snapshot.symbols.isEmpty
                    ? '(none)'
                    : '${snapshot.symbols.length}',
              ),
            ],
          ),
        ),
        if (snapshot.symbols.isNotEmpty)
          SectionCard(
            title: 'Symbol directories',
            child: Wrap(
              spacing: 6,
              runSpacing: 6,
              children: [
                for (final s in snapshot.symbols)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 3,
                    ),
                    decoration: BoxDecoration(
                      color: ForexAiTokens.surfaceBg,
                      border: Border.all(color: ForexAiTokens.border),
                      borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
                    ),
                    child: Text(
                      s,
                      style: const TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: ForexAiTokens.textPrimary,
                      ),
                    ),
                  ),
              ],
            ),
          ),
        const SectionCard(
          title: 'Historical download',
          child: Text(
            'Pull additional history from the connected broker by '
            'running:\n\n'
            '    neoethos-cli history --symbol EURUSD --from 2023-01-01\n\n'
            'The in-app download button arrives when the POST '
            '/data/bootstrap/fetch endpoint ships. Until then, files '
            'placed in the data directory above are picked up '
            'automatically by discovery/training without restart.',
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
              width: 160,
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
        child: Text(
          'Scanning data directory…',
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
