import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../state/account_provider.dart';
import '../widgets/backend_error_widget.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/symbol_picker.dart';
import '_placeholder.dart';

// Data Bootstrap screen — local OHLCV inventory + per-symbol
// historical-download form.
//
// Timeframe dropdown is sourced EXCLUSIVELY from
// `brokerTimeframesProvider` (= `neoethos_core::CANONICAL_TIMEFRAMES`
// over the wire). No hardcoded fallback list — earlier revisions had
// one that drifted out of sync (missed M3 + H12 when the contract
// gained them) and that's exactly the bug we don't want to ship
// again. The dropdown stays disabled until the server replies.

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
            error: (err, _) => BackendErrorWidget(error: err, title: 'Data tools unavailable'),
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
  // Symbol + timeframe both come from broker-backed pickers.
  String _symbol = 'EURUSD';
  String _timeframe = 'H1';
  // Default: last 12 months — operator can shrink or extend (years if
  // the broker supports it).
  DateTime _fromDate =
      DateTime.now().toUtc().subtract(const Duration(days: 365));
  DateTime _toDate = DateTime.now().toUtc();
  bool _busy = false;
  String? _lastResult;

  // #192: local-file import state. Separate from the download state
  // so the user can have a download in-flight while typing an import
  // path (or vice versa). Path is a free-text field today; a real
  // file-picker drop-zone is a follow-up (needs `file_picker` dep).
  final _importPathCtrl = TextEditingController();
  bool _importBusy = false;
  String? _importResult;

  // Data-directory editor state. The backend already accepts an
  // absolute `data_dir` (POST /settings) — the only reason this lives
  // here (and not just buried in Settings → App) is discoverability:
  // an operator landing on the Data tab to bootstrap history is
  // exactly who needs to repoint the folder when the inventory shows
  // the wrong symbol count. Prefilled from the snapshot in initState.
  late final TextEditingController _dataDirCtrl;
  bool _dataDirBusy = false;

  @override
  void initState() {
    super.initState();
    _dataDirCtrl = TextEditingController(text: widget.snapshot.dataDir);
  }

  Future<void> _onApplyDataDir() async {
    final dir = _dataDirCtrl.text.trim();
    if (dir.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          backgroundColor: NeoethosTokens.sell,
          content: Text('Data directory cannot be blank'),
        ),
      );
      return;
    }
    setState(() => _dataDirBusy = true);
    try {
      await ref.read(backendClientProvider).saveSettings(dataDir: dir);
      // Re-scan so Inventory + symbol count reflect the new folder.
      ref.invalidate(dataBootstrapProvider);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: NeoethosTokens.buy,
          content: Text('Data directory set to $dir'),
          duration: const Duration(seconds: 3),
        ),
      );
    } on DioException catch (e) {
      if (!mounted) return;
      showTranslatedErrorSnackbar(context, e, prefix: 'Save failed');
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: NeoethosTokens.sell,
          content: Text('Data directory could not be saved — ${describeError(e)}. Make sure the path exists and the app can write to it.'),
        ),
      );
    } finally {
      if (mounted) setState(() => _dataDirBusy = false);
    }
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

  Future<void> _onImport() async {
    final path = _importPathCtrl.text.trim();
    if (path.isEmpty) return;
    final symbol = _symbol.trim().toUpperCase();
    if (symbol.isEmpty) return;
    setState(() {
      _importBusy = true;
      _importResult = null;
    });
    try {
      final r = await ref.read(backendClientProvider).importLocalFile(
            sourcePath: path,
            symbol: symbol,
            timeframe: _timeframe,
          );
      final written = (r['writtenPath'] as String?) ?? '';
      final fmt = (r['sourceFormat'] as String?) ?? '?';
      setState(
          () => _importResult = 'Imported $fmt → $written');
      ref.invalidate(dataBootstrapProvider);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: NeoethosTokens.buy,
          content: Text('Imported $symbol $_timeframe from $fmt'),
          duration: const Duration(seconds: 3),
        ),
      );
    } on DioException catch (e) {
      final msg = describeError(e);
      setState(() => _importResult = 'Failed: $msg');
      if (!mounted) return;
      showTranslatedErrorSnackbar(context, e, prefix: 'Import failed');
    } finally {
      if (mounted) setState(() => _importBusy = false);
    }
  }

  Future<void> _onDownload() async {
    final symbol = _symbol.trim().toUpperCase();
    if (symbol.isEmpty) return;
    if (_toDate.isBefore(_fromDate)) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          backgroundColor: NeoethosTokens.sell,
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
      final oldestMs = (r['oldestMs'] as num?)?.toInt();
      String d(DateTime t) =>
          '${t.year}-${t.month.toString().padLeft(2, '0')}-${t.day.toString().padLeft(2, '0')}';
      final buf = StringBuffer('$count bars written to $path');
      if (oldestMs != null && count > 0) {
        final oldest = DateTime.fromMillisecondsSinceEpoch(oldestMs);
        final years = _toDate.difference(oldest).inDays / 365.0;
        buf.write(
            '\nOldest bar: ${d(oldest)} (~${years.toStringAsFixed(1)}y deep).');
        // Did the broker stop well short of the requested start?
        if (oldest.difference(_fromDate).inDays > 30) {
          buf.write(
              '\n⚠ cTrader returned only back to ${d(oldest)}, not your '
              'requested ${d(_fromDate)} — cTrader history is depth-limited '
              '(~2-3y for most symbols). For deeper history, import an '
              'MT5/CSV/Parquet file below.');
        }
      }
      if (hasMore) {
        buf.write('\n(broker hasMore=true — widen range or split into chunks)');
      }
      setState(() => _lastResult = buf.toString());
      ref.invalidate(dataBootstrapProvider);
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: NeoethosTokens.buy,
          content: Text('Downloaded $count $symbol $_timeframe bars'),
          duration: const Duration(seconds: 3),
        ),
      );
    } on DioException catch (e) {
      final msg = describeError(e);
      setState(() => _lastResult = 'Failed: $msg');
      if (!mounted) return;
      showTranslatedErrorSnackbar(context, e, prefix: 'Download failed');
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

    final dataDirExists = snapshot.dataDirExists;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: 'Data directory',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                "Point this at your historical-data folder (the one "
                "containing symbol=EURUSD/ etc.). Relative paths resolve "
                "against the app's working directory; use an absolute "
                "path to be sure.",
                style: TextStyle(
                  color: NeoethosTokens.textMuted,
                  fontSize: 12,
                ),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: _dataDirCtrl,
                enabled: !_dataDirBusy,
                style: const TextStyle(
                  fontFamily: 'monospace',
                  fontSize: 13,
                  color: NeoethosTokens.textPrimary,
                ),
                decoration: const InputDecoration(
                  labelText: 'Folder path',
                  hintText: r'Absolute path, e.g. C:\Users\you\forex-ai\data',
                  isDense: true,
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 6),
              // Live status: reflects the *currently scanned* folder
              // (the snapshot), which updates after Apply re-fetches.
              Text(
                dataDirExists
                    ? '✓ ${snapshot.symbols.length} symbols found'
                    : '✗ directory not found / empty',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: dataDirExists
                      ? NeoethosTokens.buy
                      : NeoethosTokens.sell,
                ),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  FilledButton.icon(
                    onPressed: _dataDirBusy ? null : _onApplyDataDir,
                    icon: const Icon(Icons.folder_open, size: 18),
                    label: const Text('Apply'),
                  ),
                  if (_dataDirBusy) ...[
                    const SizedBox(width: 12),
                    const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  ],
                ],
              ),
            ],
          ),
        ),
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
                    ? NeoethosTokens.buy
                    : NeoethosTokens.sell,
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
                      color: NeoethosTokens.surfaceBg,
                      border: Border.all(color: NeoethosTokens.border),
                      borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
                    ),
                    child: Text(
                      s,
                      style: const TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: NeoethosTokens.textPrimary,
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
                  color: NeoethosTokens.textMuted,
                  fontSize: 12,
                ),
              ),
              const SizedBox(height: 10),
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Expanded(
                    flex: 3,
                    child: SymbolPicker(
                      value: _symbol,
                      enabled: !_busy,
                      onChanged: (v) => setState(() => _symbol = v),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    flex: 2,
                    child: TimeframePicker(
                      value: _timeframe,
                      enabled: !_busy,
                      onChanged: (v) => setState(() => _timeframe = v),
                    ),
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
                    color: NeoethosTokens.textMuted,
                  ),
                ),
              ],
            ],
          ),
        ),
        SectionCard(
          // #192: import the user's own data files into Vortex layout.
          // The backend auto-detects format from the extension (csv, tsv,
          // parquet, json, jsonl, arrow, ipc, feather). This unblocks the
          // "I have years of MT4/MT5 history exported as CSV, don't make
          // me re-download" workflow.
          title: 'Import a local OHLCV file',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Paste the full path to a CSV, TSV, Parquet, JSON, JSONL '
                'or Arrow/IPC file. The Symbol + Timeframe selected above '
                'decide where the converted Vortex file lands on disk.',
                style: TextStyle(
                  color: NeoethosTokens.textMuted,
                  fontSize: 12,
                ),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: _importPathCtrl,
                enabled: !_importBusy,
                decoration: const InputDecoration(
                  labelText: 'Source path',
                  hintText:
                      r'C:\Users\you\Downloads\EURUSD_H1_2023.csv',
                  isDense: true,
                  border: OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  FilledButton.icon(
                    onPressed: _importBusy ? null : _onImport,
                    icon: const Icon(Icons.file_upload, size: 18),
                    label: const Text('Import file'),
                  ),
                  if (_importBusy) ...[
                    const SizedBox(width: 12),
                    const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  ],
                ],
              ),
              if (_importResult != null) ...[
                const SizedBox(height: 10),
                Text(
                  _importResult!,
                  style: const TextStyle(
                    fontSize: 11,
                    color: NeoethosTokens.textMuted,
                  ),
                ),
              ],
            ],
          ),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _importPathCtrl.dispose();
    _dataDirCtrl.dispose();
    super.dispose();
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
  Widget build(BuildContext context) => const Padding(
        padding: EdgeInsets.symmetric(vertical: 16),
        child: Text(
          'Scanning data directory…',
          style: TextStyle(color: NeoethosTokens.textMuted, fontSize: 12),
        ),
      );
}

