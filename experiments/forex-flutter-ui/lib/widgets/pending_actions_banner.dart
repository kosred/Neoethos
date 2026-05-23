// Banner for LLM-proposed trade-management actions awaiting operator
// confirmation (#136 Phase B).
//
// Reads `pendingActionsProvider` and surfaces the oldest PENDING
// action as a fat banner above the dock area. The operator clicks:
//
//   - **Confirm** → POST /actions/{id}/confirm → backend dispatches
//     the close-position broker call → status flips to executed/failed
//     → banner disappears (or briefly shows the post-execution note).
//   - **Reject**  → small dialog for an optional reason → POST
//     /actions/{id}/reject → status flips to rejected → banner gone.
//
// "Close entire" actions ship `volume_units == 0` over the wire; the
// backend rejects a confirm with code:missing_volume to force the UI
// to look up the actual position volume. We satisfy that by reading
// the matching position from `accountSnapshotProvider` and sending
// its `volumeUnits` as the override. If no matching position is
// found (race against a close that happened via another channel), we
// surface a friendly error and let the operator click Reject.
//
// When no pending actions exist (the common case), this widget
// renders an empty `SizedBox.shrink()` and takes zero vertical
// space — the dock content shifts up naturally.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
import '../state/pending_actions_provider.dart';
import '../theme/theme.dart';

class PendingActionsBanner extends ConsumerWidget {
  const PendingActionsBanner({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final asyncActions = ref.watch(pendingActionsProvider);
    final actions = asyncActions.valueOrNull ?? const <PendingAction>[];

    // Pick the oldest pending action — operator answers them in the
    // order the LLM proposed them. `list_all` on the Rust side
    // returns newest-first, so we reverse to get oldest-first.
    PendingAction? next;
    for (final a in actions.reversed) {
      if (a.isPending) {
        next = a;
        break;
      }
    }

    if (next == null) {
      return const SizedBox.shrink();
    }

    return _BannerCard(action: next);
  }
}

class _BannerCard extends ConsumerStatefulWidget {
  final PendingAction action;
  const _BannerCard({required this.action});

  @override
  ConsumerState<_BannerCard> createState() => _BannerCardState();
}

class _BannerCardState extends ConsumerState<_BannerCard> {
  bool _submitting = false;
  String? _error;

  Future<void> _handleConfirm() async {
    setState(() {
      _submitting = true;
      _error = null;
    });
    try {
      final client = ref.read(backendClientProvider);
      // For "close entire" proposals (volume_units == 0) we look
      // up the position's actual volume from the latest account
      // snapshot and pass it as override; otherwise the backend
      // rejects with code:missing_volume.
      int? override;
      if (widget.action.kindTag == 'close_position' &&
          (widget.action.volumeUnits ?? 0) <= 0) {
        final snap = ref.read(accountSnapshotProvider).valueOrNull;
        final posId = widget.action.positionId;
        Position? match;
        if (snap != null && posId != null) {
          for (final p in snap.positions) {
            if (p.positionId == posId) {
              match = p;
              break;
            }
          }
        }
        if (match == null) {
          setState(() {
            _submitting = false;
            _error =
                'Position #${posId ?? 0} not found in current snapshot — '
                'it may have already been closed. Click Reject to dismiss.';
          });
          return;
        }
        override = match.volumeUnits;
      }
      final resp = await client.confirmPendingAction(
        widget.action.id,
        volumeUnitsOverride: override,
      );
      // Backend returns `ok:true` on a clean execution; anything
      // else is a 4xx/5xx with `error` + `code` — surface inline
      // so the operator knows what went wrong without digging into
      // logs.
      final ok = resp['ok'] == true;
      if (!ok) {
        final code = resp['code']?.toString() ?? 'unknown';
        final err = resp['error']?.toString() ?? 'confirm failed';
        if (mounted) {
          setState(() {
            _submitting = false;
            _error = '[$code] $err';
          });
        }
        return;
      }
      await ref.read(pendingActionsProvider.notifier).refreshNow();
      // Also refresh the account snapshot — the close just executed,
      // the position list should drop the row immediately rather
      // than waiting 5 s for the next account poll.
      await ref.read(accountSnapshotProvider.notifier).refreshNow();
    } on Exception catch (e) {
      if (mounted) {
        setState(() {
          _submitting = false;
          _error = e.toString();
        });
      }
    }
  }

  Future<void> _handleReject() async {
    final reason = await showDialog<String?>(
      context: context,
      builder: (_) => const _RejectReasonDialog(),
    );
    // Dialog returns null when the user dismisses without clicking
    // Reject; treat that as a "never mind" and leave the banner up.
    if (reason == null) return;
    setState(() {
      _submitting = true;
      _error = null;
    });
    try {
      final client = ref.read(backendClientProvider);
      final resp = await client.rejectPendingAction(
        widget.action.id,
        reason: reason.isEmpty ? null : reason,
      );
      final ok = resp['ok'] == true;
      if (!ok) {
        final code = resp['code']?.toString() ?? 'unknown';
        final err = resp['error']?.toString() ?? 'reject failed';
        if (mounted) {
          setState(() {
            _submitting = false;
            _error = '[$code] $err';
          });
        }
        return;
      }
      await ref.read(pendingActionsProvider.notifier).refreshNow();
    } on Exception catch (e) {
      if (mounted) {
        setState(() {
          _submitting = false;
          _error = e.toString();
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final a = widget.action;
    final secondsLeft = a.secondsUntilExpiry();
    // Backend TTL is 60 s — show the countdown so the operator
    // knows they're about to time out.
    final countdownLabel =
        secondsLeft <= 0 ? 'expiring' : '${secondsLeft}s left';

    return Container(
      margin: const EdgeInsets.only(bottom: ForexAiTokens.spSm),
      padding: const EdgeInsets.all(ForexAiTokens.spMd),
      decoration: BoxDecoration(
        color: ForexAiTokens.accentSoft,
        border: Border.all(color: ForexAiTokens.accent),
        borderRadius: BorderRadius.circular(ForexAiTokens.rMd),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Icon(
                Icons.psychology_alt_outlined,
                color: ForexAiTokens.accent,
                size: 20,
              ),
              const SizedBox(width: ForexAiTokens.spSm),
              const Expanded(
                child: Text(
                  'LLM proposal awaiting your decision',
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    color: ForexAiTokens.textPrimary,
                    fontSize: ForexAiTokens.fsSubtitle,
                  ),
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: ForexAiTokens.spSm,
                  vertical: 2,
                ),
                decoration: BoxDecoration(
                  color: secondsLeft <= 10
                      ? ForexAiTokens.warning
                      : ForexAiTokens.accentMuted,
                  borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
                ),
                child: Text(
                  countdownLabel,
                  style: TextStyle(
                    color: secondsLeft <= 10
                        ? Colors.black
                        : ForexAiTokens.textPrimary,
                    fontSize: ForexAiTokens.fsCaption,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: ForexAiTokens.spSm),
          Text(
            a.summary,
            style: const TextStyle(
              color: ForexAiTokens.textPrimary,
              fontSize: ForexAiTokens.fsBody,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            a.reason.isEmpty ? '(no rationale provided)' : a.reason,
            style: const TextStyle(
              color: ForexAiTokens.textMuted,
              fontSize: ForexAiTokens.fsBody,
              fontStyle: FontStyle.italic,
            ),
          ),
          if (_error != null) ...[
            const SizedBox(height: ForexAiTokens.spSm),
            Container(
              padding: const EdgeInsets.all(ForexAiTokens.spSm),
              decoration: BoxDecoration(
                color: ForexAiTokens.surfaceBg,
                border: Border.all(color: ForexAiTokens.sell),
                borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
              ),
              child: Text(
                _error!,
                style: const TextStyle(
                  color: ForexAiTokens.sellStrong,
                  fontSize: ForexAiTokens.fsCaption,
                ),
              ),
            ),
          ],
          const SizedBox(height: ForexAiTokens.spMd),
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              OutlinedButton(
                onPressed: _submitting ? null : _handleReject,
                style: OutlinedButton.styleFrom(
                  foregroundColor: ForexAiTokens.sell,
                  side: const BorderSide(color: ForexAiTokens.sell),
                  minimumSize: const Size(0, ForexAiTokens.btnHeight),
                ),
                child: const Text('Reject'),
              ),
              const SizedBox(width: ForexAiTokens.spSm),
              FilledButton(
                onPressed: _submitting ? null : _handleConfirm,
                style: FilledButton.styleFrom(
                  backgroundColor: ForexAiTokens.accent,
                  foregroundColor: ForexAiTokens.textPrimary,
                  minimumSize: const Size(0, ForexAiTokens.btnHeight),
                ),
                child: _submitting
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: ForexAiTokens.textPrimary,
                        ),
                      )
                    : const Text('Confirm'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// Modal that collects an optional free-form rejection reason. Pops
/// back the typed string on the Reject button, `null` on Cancel /
/// dismiss / backdrop tap. An empty-string return means "Reject
/// without a reason" — the API call still goes through.
class _RejectReasonDialog extends StatefulWidget {
  const _RejectReasonDialog();

  @override
  State<_RejectReasonDialog> createState() => _RejectReasonDialogState();
}

class _RejectReasonDialogState extends State<_RejectReasonDialog> {
  final _controller = TextEditingController();

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: ForexAiTokens.panelBg,
      title: const Text(
        'Reject proposal',
        style: TextStyle(color: ForexAiTokens.textPrimary),
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Optional reason (surfaced back to the LLM for context):',
            style: TextStyle(
              color: ForexAiTokens.textMuted,
              fontSize: ForexAiTokens.fsBody,
            ),
          ),
          const SizedBox(height: ForexAiTokens.spSm),
          TextField(
            controller: _controller,
            autofocus: true,
            maxLines: 3,
            style: const TextStyle(color: ForexAiTokens.textPrimary),
            decoration: const InputDecoration(
              hintText: 'e.g. market is too thin to close here',
              hintStyle: TextStyle(color: ForexAiTokens.textFaint),
              filled: true,
              fillColor: ForexAiTokens.surfaceBg,
              border: OutlineInputBorder(),
            ),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(null),
          child: const Text(
            'Cancel',
            style: TextStyle(color: ForexAiTokens.textMuted),
          ),
        ),
        FilledButton(
          style: FilledButton.styleFrom(
            backgroundColor: ForexAiTokens.sell,
          ),
          onPressed: () =>
              Navigator.of(context).pop(_controller.text.trim()),
          child: const Text('Reject'),
        ),
      ],
    );
  }
}
