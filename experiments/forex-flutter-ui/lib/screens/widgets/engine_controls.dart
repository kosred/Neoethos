// EngineControls — reusable Start/Stop card for the Discovery and
// Training screens. Both engines share the same UX pattern: pick a
// symbol + timeframe, click Start, watch the status flip from Idle
// to Running with a live one-line summary, click Stop to cancel.
//
// **2026-05-25 — task #240**: 3-state Stop UX per research
// (SageMaker, W&B, Ray Tune, HF Trainer): Running -> Cancelling ->
// Stopped. The Stop click is cooperative — backend signals the
// cancel flag, worker yields at next epoch / generation boundary
// (up to ~30s lag inherent to GA / training loops). The button
// flips to "Cancelling… Ns" with a spinner so the operator sees
// the click registered. A SECOND click inside the Cancelling
// window opens a confirm-hard-abort dialog (covers the case where
// the worker is genuinely stuck and the operator wants to force-
// kill).
//
// The widget is dumb otherwise: parent passes the current state, the
// start/stop callbacks, and a refetch trigger.

import 'dart:async';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';

import '../../api/error_translation.dart';
import '../../theme/theme.dart';
import '../../widgets/symbol_picker.dart';
import '../_placeholder.dart';

typedef EngineStart = Future<Map<String, dynamic>> Function({
  String? symbol,
  String? baseTf,
});

typedef EngineStop = Future<Map<String, dynamic>> Function();

class EngineControls extends StatefulWidget {
  /// Display label (e.g. "Discovery", "Training") — used in button
  /// labels and snack-bar copy.
  final String kind;

  /// Whether the engine is currently running (drives the
  /// Start-disabled / Stop-enabled state).
  final bool running;

  /// Raw state string from /engines/status: "Idle" / "Running" /
  /// "Succeeded" / "Failed" / "Cancelled".
  final String state;

  /// One-line progress summary from the engine's latest snapshot.
  final String summary;

  /// POST handler — start a fresh job.
  final EngineStart start;

  /// POST handler — request cancellation of the current job.
  final EngineStop stop;

  /// Tell the parent to refetch /engines/status so the UI catches up.
  final VoidCallback onChanged;

  /// Free-text description shown under the controls.
  final String description;

  const EngineControls({
    super.key,
    required this.kind,
    required this.running,
    required this.state,
    required this.summary,
    required this.start,
    required this.stop,
    required this.onChanged,
    required this.description,
  });

  @override
  State<EngineControls> createState() => _EngineControlsState();
}

class _EngineControlsState extends State<EngineControls> {
  // Symbol + timeframe now flow through the broker-backed pickers
  // instead of free-text TextFields. Defaults stay as before so the
  // dashboard "Start" affordance still has a sane initial pick.
  String _symbol = 'EURUSD';
  String _timeframe = 'M1';

  bool _busy = false;

  // **2026-05-25 — task #240 3-state UX**: when the operator clicks
  // Stop we remember the click time + spin up a 1-Hz ticker so the
  // button label can render "Cancelling… 47s" with live elapsed
  // counter. The backend's cooperative cancel takes effect at the
  // next epoch / generation boundary (typically 5-30 s for the
  // smaller TFs, up to a couple of minutes for full discovery
  // sweeps). The elapsed counter proves the click registered even
  // when the worker hasn't yielded yet.
  DateTime? _cancelClickedAt;
  Timer? _cancelTicker;

  bool get _isCancelling => _cancelClickedAt != null && widget.running;
  int get _cancelElapsedSecs {
    final started = _cancelClickedAt;
    if (started == null) return 0;
    return DateTime.now().difference(started).inSeconds;
  }

  @override
  void didUpdateWidget(covariant EngineControls old) {
    super.didUpdateWidget(old);
    // Worker yielded → engine no longer running → exit Cancelling state.
    if (old.running && !widget.running) {
      _cancelTicker?.cancel();
      _cancelTicker = null;
      _cancelClickedAt = null;
    }
  }

  @override
  void dispose() {
    _cancelTicker?.cancel();
    super.dispose();
  }

  Future<void> _onStart() async {
    setState(() => _busy = true);
    try {
      await widget.start(
        symbol: _symbol,
        baseTf: _timeframe,
      );
      widget.onChanged();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              '${widget.kind} started for $_symbol $_timeframe',
            ),
            duration: const Duration(seconds: 2),
          ),
        );
      }
    } on DioException catch (e) {
      _showError(e);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _onStop() async {
    // SECOND click inside the Cancelling window opens hard-abort confirm.
    if (_isCancelling) {
      await _confirmHardAbort();
      return;
    }
    setState(() {
      _busy = true;
      _cancelClickedAt = DateTime.now();
    });
    _cancelTicker?.cancel();
    _cancelTicker = Timer.periodic(
      const Duration(seconds: 1),
      (_) => setState(() {}),
    );
    try {
      final r = await widget.stop();
      widget.onChanged();
      if (mounted) {
        final wasRunning = r['running'] == true;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              wasRunning
                  ? 'Cancellation queued — '
                      '${widget.kind} will stop at the next epoch/generation boundary'
                  : '${widget.kind} was not running',
            ),
            duration: const Duration(seconds: 3),
          ),
        );
      }
    } on DioException catch (e) {
      _showError(e);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _confirmHardAbort() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Row(
          children: [
            Icon(Icons.warning_amber, color: Color(0xFFB71C1C)),
            SizedBox(width: 8),
            Text('Force stop?'),
          ],
        ),
        content: SizedBox(
          width: 480,
          child: Text(
            '${widget.kind} has been cancelling for $_cancelElapsedSecs s '
            'but has not yielded yet. Force-stopping kills the worker '
            'thread without checkpointing — any partial GA elites / '
            'training progress will be lost.\n\n'
            'Only do this if you are sure the worker is stuck. Otherwise '
            'just wait — the next epoch / generation boundary may be '
            'imminent.',
            style: const TextStyle(color: NeoethosTokens.textMuted, fontSize: 13),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(false),
            child: const Text('Keep waiting'),
          ),
          FilledButton.icon(
            style: FilledButton.styleFrom(
              backgroundColor: const Color(0xFFB71C1C),
            ),
            onPressed: () => Navigator.of(ctx).pop(true),
            icon: const Icon(Icons.stop_circle, size: 16),
            label: const Text('Force stop'),
          ),
        ],
      ),
    );
    if (confirm == true) {
      // Re-invoke the same /stop endpoint with the cancel flag already
      // set; the backend treats a re-issued stop as a hard-abort
      // request when the existing cancel hasn't taken effect.
      try {
        await widget.stop();
        widget.onChanged();
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Force-stop signal sent.'),
              duration: Duration(seconds: 2),
            ),
          );
        }
      } on DioException catch (e) {
        _showError(e);
      }
    }
  }

  void _showError(DioException e) {
    if (!mounted) return;
    showTranslatedErrorSnackbar(context, e, prefix: widget.kind);
  }

  @override
  Widget build(BuildContext context) {
    final runningColor =
        widget.running ? NeoethosTokens.buy : NeoethosTokens.textFaint;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: 'Current Job',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Container(
                    width: 10,
                    height: 10,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: runningColor,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text(
                    widget.state,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: widget.running
                          ? NeoethosTokens.buy
                          : NeoethosTokens.textPrimary,
                    ),
                  ),
                ],
              ),
              if (widget.summary.isNotEmpty) ...[
                const SizedBox(height: 6),
                Text(
                  widget.summary,
                  style: const TextStyle(
                    fontSize: 12,
                    color: NeoethosTokens.textMuted,
                  ),
                ),
              ],
            ],
          ),
        ),
        SectionCard(
          title: 'Controls',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Symbol: type-ahead from /broker/symbols. Forex-only
                  // filter on by default since Discovery + Training are
                  // wired for FX-shaped pairs (the strategy heuristics
                  // and tick value math assume forex semantics).
                  Expanded(
                    flex: 3,
                    child: SymbolPicker(
                      value: _symbol,
                      enabled: !widget.running && !_busy,
                      onChanged: (v) => setState(() => _symbol = v),
                    ),
                  ),
                  const SizedBox(width: 12),
                  // Timeframe: dropdown over /broker/timeframes.
                  Expanded(
                    flex: 2,
                    child: TimeframePicker(
                      value: _timeframe,
                      enabled: !widget.running && !_busy,
                      onChanged: (v) => setState(() => _timeframe = v),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  FilledButton.icon(
                    onPressed:
                        (widget.running || _busy || _isCancelling) ? null : _onStart,
                    icon: const Icon(Icons.play_arrow, size: 18),
                    label: Text('Start ${widget.kind}'),
                  ),
                  const SizedBox(width: 8),
                  // **Task #240 3-state Stop button**:
                  //   Idle      -> disabled
                  //   Running   -> red outline "Stop" (cooperative cancel)
                  //   Cancelling-> amber filled "Cancelling… Ns" (second
                  //                click opens force-stop confirm)
                  _isCancelling
                      ? FilledButton.icon(
                          style: FilledButton.styleFrom(
                            backgroundColor: const Color(0xFFE65100),
                          ),
                          onPressed: _busy ? null : _onStop,
                          icon: const SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          ),
                          label: Text(
                            'Cancelling… ${_cancelElapsedSecs}s',
                          ),
                        )
                      : OutlinedButton.icon(
                          style: OutlinedButton.styleFrom(
                            foregroundColor: widget.running
                                ? const Color(0xFFB71C1C)
                                : null,
                            side: BorderSide(
                              color: widget.running
                                  ? const Color(0xFFB71C1C)
                                  : NeoethosTokens.textFaint,
                            ),
                          ),
                          onPressed:
                              (!widget.running || _busy) ? null : _onStop,
                          icon: const Icon(Icons.stop, size: 18),
                          label: const Text('Stop'),
                        ),
                ],
              ),
              if (_isCancelling) ...[
                const SizedBox(height: 6),
                const Text(
                  'Worker yields at the next epoch / generation boundary '
                  '(typically within 30 s). Click again to force-stop '
                  'if it stalls.',
                  // `const` here is redundant — the enclosing `const Text(...)`
                  // already inherits const-ness to nested literals.
                  style: TextStyle(
                    fontSize: 11,
                    color: NeoethosTokens.textMuted,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
              if (_busy) ...[
                const SizedBox(height: 8),
                const LinearProgressIndicator(minHeight: 2),
              ],
            ],
          ),
        ),
        SectionCard(
          title: 'How ${widget.kind.toLowerCase()} works',
          child: Text(
            widget.description,
            style: const TextStyle(
              color: NeoethosTokens.textMuted,
              fontSize: 12,
            ),
          ),
        ),
      ],
    );
  }
}
