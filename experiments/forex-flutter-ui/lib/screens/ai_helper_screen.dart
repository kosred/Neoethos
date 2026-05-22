// AI Helper — local Gemma-4 chat assistant.
//
// Defensive design: probe /gemma/status on mount, show install-mode
// banner when the runtime or the GGUF is missing, switch to chat-mode
// only when the backend reports ready. Inference is blocking
// (5–60s on CPU); we render a spinner + the elapsed-time hint while
// waiting.

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

class _Message {
  final bool fromUser;
  final String text;
  final int elapsedMs;
  const _Message(
      {required this.fromUser, required this.text, this.elapsedMs = 0});
}

class AiHelperScreen extends ConsumerStatefulWidget {
  const AiHelperScreen({super.key});

  @override
  ConsumerState<AiHelperScreen> createState() => _AiHelperScreenState();
}

class _AiHelperScreenState extends ConsumerState<AiHelperScreen> {
  final _inputCtrl = TextEditingController();
  final _scroll = ScrollController();
  final List<_Message> _messages = [];
  bool _busy = false;

  @override
  void dispose() {
    _inputCtrl.dispose();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _send() async {
    final prompt = _inputCtrl.text.trim();
    if (prompt.isEmpty || _busy) return;
    setState(() {
      _messages.add(_Message(fromUser: true, text: prompt));
      _busy = true;
      _inputCtrl.clear();
    });
    _scrollToBottom();
    try {
      final r = await ref.read(backendClientProvider).gemmaChat(
            prompt: prompt,
            maxTokens: 512,
          );
      if (!mounted) return;
      setState(() {
        _messages.add(_Message(
          fromUser: false,
          text: r.response.trim(),
          elapsedMs: r.elapsedMs,
        ));
      });
    } on DioException catch (e) {
      final body = e.response?.data;
      final msg = (body is Map && body['error'] is String)
          ? body['error'] as String
          : e.message ?? e.toString();
      if (!mounted) return;
      setState(() {
        _messages.add(_Message(fromUser: false, text: 'Error: $msg'));
      });
    } finally {
      if (mounted) setState(() => _busy = false);
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) {
        _scroll.animateTo(
          _scroll.position.maxScrollExtent,
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final status = ref.watch(gemmaStatusProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'AI Helper',
            subtitle: 'Local Gemma-4 · no API key · no quota',
          ),
          status.when(
            data: (s) => s.ready ? _chatUi(s) : _installUi(s),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
          ),
        ],
      ),
    );
  }

  Widget _installUi(GemmaStatusSnapshot s) {
    return SectionCard(
      title: 'Gemma 4 not ready',
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
          const SizedBox(height: 12),
          if (!s.runtimeCompiledIn)
            const _Kv(
              k: 'Rebuild',
              v: 'cargo build -p neoethos-app --release --features gemma-backend',
            ),
          if (s.runtimeCompiledIn && !s.modelFilePresent) ...[
            const _Kv(k: 'Download URL', v: ''),
            SelectableText(
              s.downloadUrl,
              style: const TextStyle(
                fontSize: 11,
                color: ForexAiTokens.accent,
              ),
            ),
            const SizedBox(height: 6),
            _Kv(k: 'Save as', v: 'resources/models/${s.expectedFilename}'),
            _Kv(
              k: 'Expected size',
              v: '≈${(s.expectedSizeBytes / 1024 / 1024 / 1024).toStringAsFixed(1)} GB',
            ),
          ],
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

  Widget _chatUi(GemmaStatusSnapshot s) {
    final size = MediaQuery.of(context).size;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: 'Ready',
          child: Row(
            children: [
              Container(
                width: 10,
                height: 10,
                decoration: const BoxDecoration(
                  color: ForexAiTokens.buy,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  '${s.expectedFilename} · n_ctx ${s.nCtx} · '
                  '${(s.sizeBytes / 1024 / 1024 / 1024).toStringAsFixed(2)} GB',
                  style: const TextStyle(
                    fontSize: 11,
                    color: ForexAiTokens.textMuted,
                  ),
                ),
              ),
            ],
          ),
        ),
        SectionCard(
          title: 'Chat',
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              SizedBox(
                height: (size.height - 360).clamp(220, 800),
                child: _messages.isEmpty
                    ? const Center(
                        child: Text(
                          'Ask anything. Trading questions, market\n'
                          'history, strategy ideas — runs entirely\n'
                          'on your hardware.',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 12,
                            color: ForexAiTokens.textMuted,
                          ),
                        ),
                      )
                    : ListView.builder(
                        controller: _scroll,
                        itemCount: _messages.length,
                        itemBuilder: (ctx, i) => _MessageBubble(
                          message: _messages[i],
                        ),
                      ),
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _inputCtrl,
                      enabled: !_busy,
                      maxLines: null,
                      onSubmitted: (_) => _send(),
                      decoration: const InputDecoration(
                        hintText: 'Type your question…',
                        isDense: true,
                        border: OutlineInputBorder(),
                      ),
                      inputFormatters: const [],
                    ),
                  ),
                  const SizedBox(width: 8),
                  FilledButton.icon(
                    onPressed: _busy ? null : _send,
                    icon: _busy
                        ? const SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.send, size: 16),
                    label: Text(_busy ? 'Generating…' : 'Send'),
                  ),
                ],
              ),
              if (_busy) ...[
                const SizedBox(height: 4),
                const Text(
                  'First call after server start loads the GGUF '
                  '(~5–30 s). Subsequent calls reuse the in-memory '
                  'model and respond in a few seconds for short prompts.',
                  style: TextStyle(
                    fontSize: 10,
                    color: ForexAiTokens.textFaint,
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

class _MessageBubble extends StatelessWidget {
  final _Message message;
  const _MessageBubble({required this.message});

  @override
  Widget build(BuildContext context) {
    final isUser = message.fromUser;
    final bg = isUser
        ? ForexAiTokens.accent.withValues(alpha: 0.18)
        : ForexAiTokens.surfaceBg;
    final border = isUser
        ? ForexAiTokens.accent.withValues(alpha: 0.45)
        : ForexAiTokens.border;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 56,
            child: Text(
              isUser ? 'you' : 'gemma',
              style: TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.w700,
                color: isUser ? ForexAiTokens.accent : ForexAiTokens.buy,
              ),
            ),
          ),
          Expanded(
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: bg,
                border: Border.all(color: border),
                borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SelectableText(
                    message.text,
                    style: const TextStyle(
                      fontSize: 12,
                      color: ForexAiTokens.textPrimary,
                    ),
                  ),
                  if (message.elapsedMs > 0)
                    Padding(
                      padding: const EdgeInsets.only(top: 4),
                      child: Text(
                        '${message.elapsedMs} ms',
                        style: const TextStyle(
                          fontSize: 9,
                          color: ForexAiTokens.textFaint,
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Kv extends StatelessWidget {
  final String k;
  final String v;
  const _Kv({required this.k, required this.v});
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 110,
              child: Text(
                k,
                style: const TextStyle(
                  fontSize: 11,
                  color: ForexAiTokens.textMuted,
                ),
              ),
            ),
            if (v.isNotEmpty)
              Expanded(
                child: SelectableText(
                  v,
                  style: const TextStyle(
                    fontSize: 11,
                    color: ForexAiTokens.textPrimary,
                    fontWeight: FontWeight.w600,
                  ),
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
          'Probing local LLM…',
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
