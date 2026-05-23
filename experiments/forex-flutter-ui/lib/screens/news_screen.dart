// News — local Gemma-4-powered symbol news summary.
//
// Replaces the previous PendingStub. Pick a symbol, tap "Summarise"
// and the server runs the prompt through the local Gemma runtime.
// Read-only — no live news feed (yet); the LLM works from its
// training data, which is good for "what drives EURUSD" types of
// questions and explicit about uncertainty for very recent events.

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/symbol_picker.dart';
import '_placeholder.dart';

class NewsScreen extends ConsumerStatefulWidget {
  const NewsScreen({super.key});

  @override
  ConsumerState<NewsScreen> createState() => _NewsScreenState();
}

class _NewsScreenState extends ConsumerState<NewsScreen> {
  // Symbol comes from SymbolPicker (broker catalog typeahead) so the
  // operator can summarise news for any of the ~830 instruments the
  // broker offers, not just the four we used to hardcode in a chip.
  String _symbol = 'EURUSD';
  String? _result;
  int _elapsedMs = 0;
  String _lastSymbol = '';
  bool _busy = false;

  Future<void> _fetch() async {
    final symbol = _symbol.trim().toUpperCase();
    if (symbol.isEmpty || _busy) return;
    setState(() {
      _busy = true;
      _result = null;
      _lastSymbol = symbol;
    });
    try {
      final r =
          await ref.read(backendClientProvider).gemmaNews(symbol: symbol);
      if (!mounted) return;
      setState(() {
        _result = r.response.trim();
        _elapsedMs = r.elapsedMs;
      });
    } on DioException catch (e) {
      final msg = describeError(e);
      if (!mounted) return;
      setState(() => _result = 'Error: $msg');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final status = ref.watch(gemmaStatusProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'News',
            subtitle: 'Per-symbol summary via local Gemma-4',
          ),
          status.when(
            data: (s) => s.ready ? _readyUi(s) : _installHint(s),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
          ),
        ],
      ),
    );
  }

  Widget _installHint(GemmaStatusSnapshot s) {
    return SectionCard(
      title: 'Local LLM not ready',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            s.message,
            style: const TextStyle(
              fontSize: 12,
              color: ForexAiTokens.warning,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'News uses the same local inference path as AI Helper. '
            'Open AI Helper → follow the install instructions, then come '
            'back here.',
            style: TextStyle(
              fontSize: 11,
              color: ForexAiTokens.textMuted,
            ),
          ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            onPressed: () => ref.invalidate(gemmaStatusProvider),
            icon: const Icon(Icons.refresh, size: 16),
            label: const Text('Re-check status'),
          ),
        ],
      ),
    );
  }

  Widget _readyUi(GemmaStatusSnapshot s) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: 'Pick a symbol',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
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
                  Padding(
                    padding: const EdgeInsets.only(top: 8),
                    child: FilledButton.icon(
                      onPressed: _busy ? null : _fetch,
                      icon: _busy
                          ? const SizedBox(
                              width: 14,
                              height: 14,
                              child:
                                  CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.auto_awesome, size: 16),
                      label: Text(_busy ? 'Summarising…' : 'Summarise'),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              Text(
                'Model: ${s.expectedFilename} · n_ctx ${s.nCtx}',
                style: const TextStyle(
                  fontSize: 11,
                  color: ForexAiTokens.textMuted,
                ),
              ),
            ],
          ),
        ),
        if (_result != null)
          SectionCard(
            title: 'Summary — $_lastSymbol',
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SelectableText(
                  _result!,
                  style: const TextStyle(
                    fontSize: 12,
                    color: ForexAiTokens.textPrimary,
                  ),
                ),
                if (_elapsedMs > 0) ...[
                  const SizedBox(height: 6),
                  Text(
                    'generated in $_elapsedMs ms',
                    style: const TextStyle(
                      fontSize: 10,
                      color: ForexAiTokens.textFaint,
                    ),
                  ),
                ],
              ],
            ),
          )
        else if (!_busy)
          const SectionCard(
            title: 'Tip',
            child: Text(
              'Local Gemma works from its training cutoff — great for '
              '"what typically drives EURUSD" or "explain rate-hike '
              'transmission to GBP", less reliable for events from the '
              'last 24 hours. A live news-feed integration is the next '
              'iteration.',
              style: TextStyle(
                color: ForexAiTokens.textMuted,
                fontSize: 12,
              ),
            ),
          ),
      ],
    );
  }
}

class _Loading extends StatelessWidget {
  const _Loading();
  @override
  Widget build(BuildContext context) => const Padding(
        padding: EdgeInsets.symmetric(vertical: 16),
        child: Text(
          'Checking local LLM status…',
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
