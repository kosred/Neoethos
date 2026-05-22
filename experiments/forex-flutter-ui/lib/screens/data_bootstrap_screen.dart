import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

/// Fallback timeframe list shown ONLY while /broker/timeframes is
/// in-flight. Real choices come from `brokerTimeframesProvider`
/// (= `neoethos_core::CANONICAL_TIMEFRAMES`). Earlier revisions had a
/// 9-entry hardcoded list here that was missing M3 + H12 — that
/// drift is exactly why this is server-driven now.
const _fallbackTimeframes = <String>['M1', 'H1', 'D1'];

class DataBootstrapScreen extends ConsumerWidget {
  const DataBootstrapScreen({super.key});
  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final inventory = ref.watch(dataBootstrapProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'Data Bootstrap',
            subtitle: 'Local OHLCV inventory · historical download',
          ),
          inventory.when(
            data: (d) => _Body(snapshot: d),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
          ),
        ],
      ),
    );
  }
}

class _Body extends ConsumerStatefulWidget {
  final DataBootstrapSnapshot snapshot;
  const _Body({required this.snapshot});

  @override
  ConsumerState<_Body> createState() => _BodyState();
}

class _BodyState extends ConsumerState<_Body> {
  final _symbolCtrl = TextEditingController(text: 'EURUSD');
  String _timeframe = 'H1';
  // Default: last 12 months — operator can shrink or extend (years if
  // the broker supports it).
  DateTime _fromDate =
      DateTime.now().toUtc().subtract(const Duration(days: 365));
  DateTime _toDate = DateTime.now().toUtc();
  bool _busy = false;
  String? _lastResult;

  @override
  void dispose() {
    _symbolCtrl.dispose();
    super.dispose();
  }

  Future<void> _pickDate({required bool isFrom}) async {
    final initial = isFrom ? _fromDate : _toDate;
    final picked = await showDatePicker(
      context: context,
      initialDate: initial.toLocal(),
      firstDate: DateTime.utc(2005, 1, 1),
      lastDate: DateTime.now().toUtc().add(const Duration(days: 1)),
    );
    if (picked == null || !mounted) return;
    setState(() {
      if (isFrom) {
        _fromDate = DateTime.utc(picked.year, picked.month, picked.day);
      } else {
        _toDate = DateTime.utc(picked.year, picked.month, picked.day);
      }
    });
  }

  Future<void> _onDownload() async {
    final symbol = _symbolCtrl.text.trim().toUpperCase();
    if (symbol.isEmpty) return;
    if (_toDate.isBefore(_fromDate)) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          backgroundColor: ForexAiTokens.sell,
          content: Text('From-date must be before to-date'),
        ),
      );
      return;
    }

    setState(() {
      _busy = true;
      _lastResult = null;
    });
    try {
      final r = await ref.read(backendClientProvider).fetchHistoricalData(
            symbol: symbol,
            timeframe: _timeframe,
            fromMs: _fromDate.millisecondsSinceEpoch,
            toMs: _toDate.millisecondsSinceEpoch,
          );
      final count = (r['barCount'] as num?)?.toInt() ?? 0;
      final hasMore = r['hasMore'] == true;
      final path = (r['writtenPath'] as String?) ?? '';
      final tail = hasMore
          ? ' — broker says hasMore=true; widen range or split into chunks'
          : '';
      setState(() => _lastResult =
          '$count bars written to $path$tail');
      ref.invalidate(dataBootstrapProvider);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: ForexAiTokens.buy,
          content: Text('Downloaded $count $symbol $_timeframe bars'),
          duration: const Duration(seconds: 3),
        ),
      );
    } on DioException catch (e) {
      final body = e.response?.data;
      final msg = (body is Map && body['error'] is String)
          ? body['error'] as String
          : e.message ?? e.toString();
      setState(() => _lastResult = 'Failed: $msg');
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: ForexAiTokens.sell,
          content: Text('Download failed: $msg'),
          duration: const Duration(seconds: 5),
        ),
      );
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final snapshot = widget.snapshot;
    final dtFmt = DateFormat('yyyy-MM-dd HH:mm');
    final mtime = snapshot.lastTouchedUnixMs == null
        ? '—'
        : dtFmt.format(
            DateTime.fromMillisecondsSinceEpoch(snapshot.lastTouchedUnixMs!));
    final dateFmt = DateFormat('yyyy-MM-dd');

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
        SectionCard(
          title: 'Download history from broker',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Pull as many years of OHLCV as the broker has on file. '
                'Symbol is anything in the broker catalog (forex pairs, '
                'metals, indices). Wide ranges may need multiple downloads '
                '— if the response shows hasMore=true, narrow the window '
                'and re-fetch the missing tail.',
                style: TextStyle(
                  color: ForexAiTokens.textMuted,
                  fontSize: 12,
                ),
              ),
              const SizedBox(height: 10),
              Row(
                children: [
                  SizedBox(
                    width: 140,
                    child: TextField(
                      controller: _symbolCtrl,
                      enabled: !_busy,
                      textCapitalization: TextCapitalization.characters,
                      inputFormatters: [
                        FilteringTextInputFormatter.allow(
                          RegExp(r'[A-Za-z0-9.]'),
                        ),
                      ],
                      decoration: const InputDecoration(
                        labelText: 'Symbol',
                        isDense: true,
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Consumer(
                    builder: (ctx, ref, _) {
                      final tfs = ref
                          .watch(brokerTimeframesProvider)
                          .maybeWhen(
                            data: (list) =>
                                list.isEmpty ? _fallbackTimeframes : list,
                            orElse: () => _fallbackTimeframes,
                          );
                      // If the live list doesn't include the saved
                      // pick (e.g. canonical contract changed),
                      // anchor on the first available value so
                      // DropdownButton doesn't assert.
                      final current = tfs.contains(_timeframe)
                          ? _timeframe
                          : tfs.first;
                      if (current != _timeframe) {
                        WidgetsBinding.instance.addPostFrameCallback(
                          (_) {
                            if (mounted) {
                              setState(() => _timeframe = current);
                            }
                          },
                        );
                      }
                      return DropdownButton<String>(
                        value: current,
                        items: [
                          for (final tf in tfs)
                            DropdownMenuItem(value: tf, child: Text(tf)),
                        ],
                        onChanged: _busy
                            ? null
                            : (v) {
                                if (v != null) setState(() => _timeframe = v);
                              },
                      );
                    },
                  ),
                ],
              ),
              const SizedBox(height: 10),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _busy ? null : () => _pickDate(isFrom: true),
                      icon: const Icon(Icons.calendar_today, size: 14),
                      label: Text('From: ${dateFmt.format(_fromDate)}'),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _busy ? null : () => _pickDate(isFrom: false),
                      icon: const Icon(Icons.calendar_today, size: 14),
                      label: Text('To: ${dateFmt.format(_toDate)}'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  FilledButton.icon(
                    onPressed: _busy ? null : _onDownload,
                    icon: const Icon(Icons.cloud_download, size: 18),
                    label: const Text('Download'),
                  ),
                  if (_busy) ...[
                    const SizedBox(width: 12),
                    const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  ],
                ],
              ),
              if (_lastResult != null) ...[
                const SizedBox(height: 10),
                Text(
                  _lastResult!,
                  style: const TextStyle(
                    fontSize: 11,
                    color: ForexAiTokens.textMuted,
                  ),
                ),
              ],
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
