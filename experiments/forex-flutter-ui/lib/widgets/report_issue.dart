// Report Issue — one-click diagnostic bundle + mailto: flow.
//
// End users of NeoEthos cannot rebuild the app. Whenever something
// breaks they need a single button that:
//   1. Collects today's logs + redacted config + system info into a
//      .zip on their Desktop (server-side, via POST /diagnostics/report).
//   2. Opens their default mail client prefilled to
//      konstantinoskokkinos1982@gmail.com with the file path
//      mentioned so they can drag-attach the zip.
//   3. Copies the path to the clipboard as a backup for users
//      whose default mail client doesn't open from a mailto: URL.
//
// `mailto:` attachments don't work cross-platform (RFC-3986 doesn't
// allow them, and only Outlook honours non-standard attachment
// extensions). So our pattern is: open mailto: with the path
// mentioned in the body, plus a "Copy path" button so the user can
// paste into their mail client's attach dialog.

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
import '../theme/theme.dart';

/// Show the Report Issue dialog. Call from any catch block, button,
/// or auto-trigger (e.g. backend supervisor.log panic).
///
/// `prefillDescription` lets the caller seed the description box
/// with the error string they already had — saves the user from
/// re-typing it. `category` becomes the email-subject suffix.
Future<void> showReportIssueDialog(
  BuildContext context, {
  String prefillDescription = '',
  String category = '',
}) async {
  await showDialog<void>(
    context: context,
    builder: (ctx) => _ReportIssueDialog(
      prefillDescription: prefillDescription,
      category: category,
    ),
  );
}

class _ReportIssueDialog extends ConsumerStatefulWidget {
  final String prefillDescription;
  final String category;
  const _ReportIssueDialog({
    required this.prefillDescription,
    required this.category,
  });

  @override
  ConsumerState<_ReportIssueDialog> createState() => _ReportIssueDialogState();
}

class _ReportIssueDialogState extends ConsumerState<_ReportIssueDialog> {
  late final TextEditingController _descCtrl =
      TextEditingController(text: widget.prefillDescription);
  bool _building = false;
  DiagnosticReport? _result;
  Object? _error;

  @override
  void dispose() {
    _descCtrl.dispose();
    super.dispose();
  }

  Future<void> _generate() async {
    setState(() {
      _building = true;
      _error = null;
    });
    try {
      final r = await ref.read(backendClientProvider).requestDiagnosticReport(
            userDescription: _descCtrl.text.trim(),
            category: widget.category,
          );
      if (!mounted) return;
      setState(() => _result = r);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e);
    } finally {
      if (mounted) setState(() => _building = false);
    }
  }

  /// Open the user's default mail client with the prefilled
  /// recipient + subject + body. Uses platform shell out — no
  /// url_launcher dep.
  Future<void> _openMail() async {
    final r = _result;
    if (r == null) return;
    final encodedSubject = Uri.encodeComponent(r.emailSubject);
    final encodedBody = Uri.encodeComponent(r.emailBody);
    final mailto =
        'mailto:${r.emailRecipient}?subject=$encodedSubject&body=$encodedBody';
    try {
      if (Platform.isWindows) {
        // `start` is a cmd builtin — handles the protocol via the
        // default mailto: handler.
        await Process.run('cmd', ['/c', 'start', '', mailto]);
      } else if (Platform.isMacOS) {
        await Process.run('open', [mailto]);
      } else {
        await Process.run('xdg-open', [mailto]);
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: NeoethosTokens.warning,
          content: Text(
            'Could not open mail client ($e). Email '
            '${r.emailRecipient} manually and attach ${r.zipPath}.',
          ),
        ),
      );
    }
  }

  Future<void> _copyPath() async {
    final r = _result;
    if (r == null) return;
    await Clipboard.setData(ClipboardData(text: r.zipPath));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        backgroundColor: NeoethosTokens.buy,
        content: Text('File path copied — paste it into your mail attach dialog'),
        duration: Duration(seconds: 3),
      ),
    );
  }

  Future<void> _revealOnDesktop() async {
    final r = _result;
    if (r == null) return;
    try {
      if (Platform.isWindows) {
        // /select shows the file highlighted inside Explorer.
        await Process.run('explorer.exe', ['/select,', r.zipPath]);
      } else if (Platform.isMacOS) {
        await Process.run('open', ['-R', r.zipPath]);
      } else {
        // Best-effort on Linux: open the enclosing folder.
        final dir = r.zipPath.substring(0, r.zipPath.lastIndexOf('/'));
        await Process.run('xdg-open', [dir]);
      }
    } catch (_) {
      // Silent — Copy Path is the reliable fallback.
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Report an issue'),
      content: SizedBox(
        width: 480,
        child: SingleChildScrollView(
          child: _result == null
              ? _buildForm()
              : _buildResult(_result!),
        ),
      ),
      actions: _result == null
          ? [
              TextButton(
                onPressed: _building ? null : () => Navigator.pop(context),
                child: const Text('Cancel'),
              ),
              FilledButton.icon(
                onPressed: _building ? null : _generate,
                icon: _building
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.bug_report, size: 16),
                label: Text(_building ? 'Bundling…' : 'Generate report'),
              ),
            ]
          : [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Close'),
              ),
              OutlinedButton.icon(
                onPressed: _revealOnDesktop,
                icon: const Icon(Icons.folder_open, size: 16),
                label: const Text('Show file'),
              ),
              OutlinedButton.icon(
                onPressed: _copyPath,
                icon: const Icon(Icons.copy, size: 16),
                label: const Text('Copy path'),
              ),
              FilledButton.icon(
                onPressed: _openMail,
                icon: const Icon(Icons.mail, size: 16),
                label: const Text('Open mail'),
              ),
            ],
    );
  }

  Widget _buildForm() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Something not working? Click Generate report and we will '
          'collect a small bundle (today\'s logs + redacted config + '
          'system info) into a .zip on your Desktop. Then we open your '
          'mail client so you can send it to NeoEthos support.',
          style: TextStyle(fontSize: 12, color: NeoethosTokens.textPrimary),
        ),
        const SizedBox(height: 10),
        TextField(
          controller: _descCtrl,
          minLines: 3,
          maxLines: 6,
          decoration: const InputDecoration(
            labelText: 'What were you doing when it broke? (optional)',
            border: OutlineInputBorder(),
            isDense: true,
            hintText:
                'e.g. "Tried to BUY EURUSD 0.1 lot, got Order failed: '
                'Markets are closed" — but I checked, market should '
                'be open',
          ),
        ),
        const SizedBox(height: 8),
        const Text(
          'Privacy: client_secret + access_token are masked before '
          'they enter the zip. The bundle never leaves your machine '
          'unless you send it.',
          style: TextStyle(
            fontSize: 10,
            color: NeoethosTokens.textFaint,
          ),
        ),
        if (_error != null) ...[
          const SizedBox(height: 10),
          Text(
            'Bundle failed: $_error',
            style: const TextStyle(fontSize: 11, color: NeoethosTokens.sell),
          ),
        ],
      ],
    );
  }

  Widget _buildResult(DiagnosticReport r) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.check_circle, size: 18, color: NeoethosTokens.buy),
            const SizedBox(width: 6),
            Text(
              'Bundle ready · ${r.sizeLabel}',
              style: const TextStyle(
                fontWeight: FontWeight.w700,
                color: NeoethosTokens.buy,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        const Text(
          'File on your Desktop:',
          style: TextStyle(fontSize: 11, color: NeoethosTokens.textMuted),
        ),
        SelectableText(
          r.zipPath,
          style: const TextStyle(
            fontSize: 11,
            color: NeoethosTokens.textPrimary,
            fontFamily: 'Consolas',
          ),
        ),
        const SizedBox(height: 10),
        const Text(
          'What\'s inside:',
          style: TextStyle(fontSize: 11, color: NeoethosTokens.textMuted),
        ),
        const SizedBox(height: 4),
        Wrap(
          spacing: 4,
          runSpacing: 4,
          children: [
            for (final f in r.filesIncluded)
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: NeoethosTokens.surfaceBg,
                  border: Border.all(color: NeoethosTokens.border),
                  borderRadius: BorderRadius.circular(3),
                ),
                child: Text(
                  f,
                  style: const TextStyle(
                    fontSize: 10,
                    fontFamily: 'Consolas',
                    color: NeoethosTokens.textPrimary,
                  ),
                ),
              ),
          ],
        ),
        const SizedBox(height: 12),
        Text(
          'Recipient: ${r.emailRecipient}',
          style: const TextStyle(
            fontSize: 11,
            color: NeoethosTokens.textMuted,
          ),
        ),
        const SizedBox(height: 8),
        const Text(
          'Click "Open mail" — your default mail client opens with '
          'the recipient + subject prefilled. Drag the zip from your '
          'Desktop into the message before you send.',
          style: TextStyle(
            fontSize: 11,
            color: NeoethosTokens.textPrimary,
          ),
        ),
      ],
    );
  }
}
