// Shared error panel for AsyncValue error states (F-346).
//
// Replaces the ~11 copy-pasted `_Error` widgets that each dumped
// "Backend unreachable: <raw Dart/Dio exception>" — a wall of HTTP
// internals with no path forward. This one runs the error through
// [describeError] (translation-aware, strips Dio noise, prefers the
// backend's friendly `translation.message`), explains where to look,
// and offers a one-tap engine restart — the actual fix for most of
// these states.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/error_translation.dart';
import '../l10n/app_localizations.dart';
import '../startup/backend_watchdog.dart';
import '../theme/theme.dart';

class BackendErrorWidget extends ConsumerWidget {
  final Object error;

  /// One-line context for THIS screen, e.g. "Settings couldn't load".
  /// The translated detail renders beneath it. When null, a localized
  /// generic "couldn't reach the engine" title is used.
  final String? title;

  /// Show the "Restart engine" button. Leave true for connectivity /
  /// backend-down errors (the common case); set false for errors a
  /// restart won't fix (bad input, validation).
  final bool showRestart;

  const BackendErrorWidget({
    super.key,
    required this.error,
    this.title,
    this.showRestart = true,
  });

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final l10n = AppLocalizations.of(context)!;
    final detail = describeError(error);
    final resolvedTitle = title ?? l10n.backendErrorDefaultTitle;
    return Padding(
      padding: const EdgeInsets.all(NeoethosTokens.spMd),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Padding(
                padding: EdgeInsets.only(top: 1),
                child: Icon(Icons.error_outline,
                    size: 18, color: NeoethosTokens.sell),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  resolvedTitle,
                  style: const TextStyle(
                    fontSize: NeoethosTokens.fsBody,
                    fontWeight: FontWeight.w700,
                    color: NeoethosTokens.textPrimary,
                  ),
                ),
              ),
            ],
          ),
          if (detail.isNotEmpty) ...[
            const SizedBox(height: 6),
            Text(
              detail,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsCaption,
                height: 1.45,
                color: NeoethosTokens.textMuted,
              ),
            ),
          ],
          const SizedBox(height: 4),
          Text(
            l10n.backendErrorCheckIndicator,
            style: const TextStyle(
              fontSize: NeoethosTokens.fsCaption,
              height: 1.45,
              color: NeoethosTokens.textFaint,
            ),
          ),
          if (showRestart) ...[
            const SizedBox(height: 10),
            OutlinedButton.icon(
              onPressed: () {
                ref.read(backendHealthProvider.notifier).manualRestart();
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(l10n.backendErrorRestarting),
                    duration: const Duration(seconds: 3),
                  ),
                );
              },
              icon: const Icon(Icons.refresh, size: 16),
              label: Text(l10n.backendErrorRestartEngine),
            ),
          ],
        ],
      ),
    );
  }
}
