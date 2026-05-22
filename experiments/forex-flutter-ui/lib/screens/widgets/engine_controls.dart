// EngineControls — reusable Start/Stop card for the Discovery and
// Training screens. Both engines share the same UX pattern: pick a
// symbol + timeframe, click Start, watch the status flip from Idle
// to Running with a live one-line summary, click Stop to cancel.
//
// The widget is dumb: parent passes the current state, the start/stop
// callbacks, and a refetch trigger. We don't own any state ourselves —
// that lives in `enginesProvider` which is invalidated on every
// successful start/stop so the status row reflects reality.

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../theme/theme.dart';
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
  final TextEditingController _symbolCtrl =
      TextEditingController(text: 'EURUSD');
  final TextEditingController _tfCtrl = TextEditingController(text: 'M1');

  bool _busy = false;

  @override
  void dispose() {
    _symbolCtrl.dispose();
    _tfCtrl.dispose();
    super.dispose();
  }

  Future<void> _onStart() async {
    setState(() => _busy = true);
    try {
      await widget.start(
        symbol: _symbolCtrl.text.trim(),
        baseTf: _tfCtrl.text.trim(),
      );
      widget.onChanged();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              '${widget.kind} started for '
              '${_symbolCtrl.text.trim().toUpperCase()} '
              '${_tfCtrl.text.trim().toUpperCase()}',
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
    setState(() => _busy = true);
    try {
      final r = await widget.stop();
      widget.onChanged();
      if (mounted) {
        final wasRunning = r['running'] == true;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              wasRunning
                  ? '${widget.kind} cancellation requested — '
                      'will stop at next checkpoint'
                  : '${widget.kind} was not running',
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

  void _showError(DioException e) {
    if (!mounted) return;
    final body = e.response?.data;
    final msg = (body is Map && body['error'] is String)
        ? body['error'] as String
        : e.message ?? e.toString();
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        backgroundColor: ForexAiTokens.sell,
        content: Text('${widget.kind}: $msg'),
        duration: const Duration(seconds: 4),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final runningColor =
        widget.running ? ForexAiTokens.buy : ForexAiTokens.textFaint;

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
                          ? ForexAiTokens.buy
                          : ForexAiTokens.textPrimary,
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
                    color: ForexAiTokens.textMuted,
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
                children: [
                  SizedBox(
                    width: 140,
                    child: TextField(
                      controller: _symbolCtrl,
                      enabled: !widget.running && !_busy,
                      decoration: const InputDecoration(
                        labelText: 'Symbol',
                        isDense: true,
                        border: OutlineInputBorder(),
                      ),
                      inputFormatters: [
                        FilteringTextInputFormatter.allow(
                          RegExp(r'[A-Za-z0-9]'),
                        ),
                      ],
                      textCapitalization: TextCapitalization.characters,
                    ),
                  ),
                  const SizedBox(width: 12),
                  SizedBox(
                    width: 100,
                    child: TextField(
                      controller: _tfCtrl,
                      enabled: !widget.running && !_busy,
                      decoration: const InputDecoration(
                        labelText: 'Timeframe',
                        isDense: true,
                        border: OutlineInputBorder(),
                      ),
                      inputFormatters: [
                        FilteringTextInputFormatter.allow(
                          RegExp(r'[A-Za-z0-9]'),
                        ),
                      ],
                      textCapitalization: TextCapitalization.characters,
                    ),
                  ),
                  const SizedBox(width: 16),
                  FilledButton.icon(
                    onPressed:
                        (widget.running || _busy) ? null : _onStart,
                    icon: const Icon(Icons.play_arrow, size: 18),
                    label: Text('Start ${widget.kind}'),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed:
                        (!widget.running || _busy) ? null : _onStop,
                    icon: const Icon(Icons.stop, size: 18),
                    label: const Text('Stop'),
                  ),
                ],
              ),
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
              color: ForexAiTokens.textMuted,
              fontSize: 12,
            ),
          ),
        ),
      ],
    );
  }
}
