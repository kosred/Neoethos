// BackendDiagnosticsDialog — the operator-facing window onto the
// supervisor + watchdog state.
//
// Surfaces:
//   - Current backend health (online / reconnecting)
//   - Last seen timestamp
//   - Respawn attempt counter (lifetime of this Flutter session)
//   - Tail of supervisor.log (last 200 lines)
//   - "Restart backend" button (force-kills the child and respawns
//     via BackendSupervisor.restartBackend — bypassing the
//     watchdog's 15 s backoff because the operator explicitly
//     asked for it)
//   - "Copy log path" — convenience for users sending the log to
//     support manually
//
// Entry points:
//   - Click on the red BackendHealthBanner (the obvious, contextual
//     path).
//   - The TopBar's monitor-heart icon (always-on diagnostics entry,
//     not just when something's broken).

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/error_translation.dart';
import '../startup/backend_supervisor.dart';
import '../startup/backend_watchdog.dart';
import '../theme/theme.dart';

Future<void> showBackendDiagnosticsDialog(BuildContext context) async {
  await showDialog<void>(
    context: context,
    builder: (_) => const _BackendDiagnosticsDialog(),
  );
}

class _BackendDiagnosticsDialog extends ConsumerStatefulWidget {
  const _BackendDiagnosticsDialog();

  @override
  ConsumerState<_BackendDiagnosticsDialog> createState() =>
      _BackendDiagnosticsDialogState();
}

class _BackendDiagnosticsDialogState
    extends ConsumerState<_BackendDiagnosticsDialog> {
  // Snapshot the log on first build so the dialog doesn't scroll-jitter
  // as new lines stream in. The operator can hit refresh to reload.
  String _logTail = '';
  bool _restarting = false;

  @override
  void initState() {
    super.initState();
    _loadLogTail();
  }

  void _loadLogTail() {
    setState(() {
      _logTail = BackendSupervisor.instance.tailLog(maxLines: 200);
    });
  }

  Future<void> _restartBackend() async {
    setState(() => _restarting = true);
    try {
      await ref.read(backendHealthProvider.notifier).manualRestart();
      // Reload the log so the operator sees the restart line(s) the
      // supervisor just appended.
      _loadLogTail();
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          backgroundColor: ForexAiTokens.accent,
          content: Text(
            'Backend restart requested — watchdog will confirm in ~3 s.',
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: ForexAiTokens.sell,
          content: Text(
            'Restart command failed — ${describeError(e)}. '
            'Try ending the neoethos-core process in Task Manager, '
            'then relaunch.',
          ),
        ),
      );
    } finally {
      if (mounted) setState(() => _restarting = false);
    }
  }

  Future<void> _copyLogPath() async {
    final path = BackendSupervisor.instance.logFilePath;
    await Clipboard.setData(ClipboardData(text: path));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        backgroundColor: ForexAiTokens.buy,
        content: Text('Log path copied to clipboard.'),
        duration: Duration(seconds: 2),
      ),
    );
  }

  Future<void> _openLogFolder() async {
    final path = BackendSupervisor.instance.logFilePath;
    try {
      if (Platform.isWindows) {
        await Process.run('explorer.exe', ['/select,', path]);
      } else if (Platform.isMacOS) {
        await Process.run('open', ['-R', path]);
      } else {
        final dir = path.substring(0, path.lastIndexOf('/'));
        await Process.run('xdg-open', [dir]);
      }
    } catch (_) {
      // Silent — Copy Path is the fallback.
    }
  }

  @override
  Widget build(BuildContext context) {
    final health = ref.watch(backendHealthProvider);
    final logPath = BackendSupervisor.instance.logFilePath;
    final pid = BackendSupervisor.instance.childPid;

    final statusLabel = health.status == BackendHealthStatus.online
        ? 'Online'
        : 'Reconnecting…';
    final statusColor = health.status == BackendHealthStatus.online
        ? ForexAiTokens.buy
        : ForexAiTokens.sell;

    final lastSeen = health.lastSeenAt;
    final lastSeenLabel = lastSeen == null
        ? 'never'
        : DateFormat('HH:mm:ss').format(lastSeen.toLocal());

    return AlertDialog(
      title: const Text('Backend diagnostics'),
      content: SizedBox(
        width: 640,
        height: 480,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Status row
            Row(
              children: [
                Icon(
                  health.status == BackendHealthStatus.online
                      ? Icons.check_circle
                      : Icons.error,
                  size: 18,
                  color: statusColor,
                ),
                const SizedBox(width: 6),
                Text(
                  statusLabel,
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: ForexAiTokens.fsBody,
                    color: statusColor,
                  ),
                ),
                const SizedBox(width: ForexAiTokens.spLg),
                _MetaItem(label: 'Last /healthz OK', value: lastSeenLabel),
                const SizedBox(width: ForexAiTokens.spLg),
                _MetaItem(
                  label: 'Consecutive failures',
                  value: '${health.consecutiveFailures}',
                ),
                const SizedBox(width: ForexAiTokens.spLg),
                _MetaItem(
                  label: 'Respawn attempts',
                  value: '${health.respawnAttempts}',
                ),
                const SizedBox(width: ForexAiTokens.spLg),
                _MetaItem(
                  label: 'Child PID',
                  value: pid == null ? '—' : '$pid',
                ),
              ],
            ),
            const SizedBox(height: ForexAiTokens.spMd),
            // Log path bar
            Row(
              children: [
                const Icon(Icons.description,
                    size: 14, color: ForexAiTokens.textMuted),
                const SizedBox(width: 6),
                Expanded(
                  child: SelectableText(
                    logPath,
                    style: const TextStyle(
                      fontSize: 11,
                      fontFamily: 'Consolas',
                      color: ForexAiTokens.textMuted,
                    ),
                  ),
                ),
                IconButton(
                  onPressed: _copyLogPath,
                  tooltip: 'Copy log path',
                  icon: const Icon(Icons.copy, size: 14),
                ),
                IconButton(
                  onPressed: _openLogFolder,
                  tooltip: 'Show in file manager',
                  icon: const Icon(Icons.folder_open, size: 14),
                ),
                IconButton(
                  onPressed: _loadLogTail,
                  tooltip: 'Refresh log',
                  icon: const Icon(Icons.refresh, size: 14),
                ),
              ],
            ),
            const SizedBox(height: ForexAiTokens.spXs),
            // Log tail
            Expanded(
              child: Container(
                decoration: BoxDecoration(
                  color: ForexAiTokens.appBg,
                  border: Border.all(color: ForexAiTokens.border),
                  borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
                ),
                padding: const EdgeInsets.all(ForexAiTokens.spSm),
                child: Scrollbar(
                  child: SingleChildScrollView(
                    reverse: true,
                    child: SelectableText(
                      _logTail.isEmpty
                          ? '<log empty — backend has not written anything yet>'
                          : _logTail,
                      style: const TextStyle(
                        fontSize: 11,
                        fontFamily: 'Consolas',
                        color: ForexAiTokens.textPrimary,
                        height: 1.35,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed:
              _restarting ? null : () => Navigator.of(context).pop(),
          child: const Text('Close'),
        ),
        FilledButton.icon(
          onPressed: _restarting ? null : _restartBackend,
          icon: _restarting
              ? const SizedBox(
                  width: 14,
                  height: 14,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.restart_alt, size: 16),
          label: Text(_restarting ? 'Restarting…' : 'Restart backend'),
        ),
      ],
    );
  }
}

class _MetaItem extends StatelessWidget {
  final String label;
  final String value;
  const _MetaItem({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label.toUpperCase(),
          style: const TextStyle(
            fontSize: ForexAiTokens.fsCaption - 1,
            letterSpacing: 0.8,
            fontWeight: FontWeight.w700,
            color: ForexAiTokens.textMuted,
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            fontSize: ForexAiTokens.fsBody,
            fontWeight: FontWeight.w700,
            color: ForexAiTokens.textPrimary,
          ),
        ),
      ],
    );
  }
}
