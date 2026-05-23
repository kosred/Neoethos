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
import '../api/error_translation.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/report_issue.dart';
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
      final msg = describeError(e);
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
    // Two distinct branches:
    //  * Runtime NOT compiled in: this is a packaging defect —
    //    end users CAN'T fix it by themselves (they don't have a
    //    cargo toolchain). Surface the Report Issue flow instead of
    //    a useless cargo command line they can't act on.
    //  * Runtime compiled in, model missing: we CAN help. Show the
    //    download button + progress, calling /gemma/download under
    //    the hood. This is the first-launch fallback when the NSIS
    //    install-time fetch was skipped or interrupted.
    if (!s.runtimeCompiledIn) {
      return SectionCard(
        title: 'Gemma 4 runtime not compiled in this build',
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
            const Text(
              'This installer was built without the local LLM '
              'feature compiled in — so AI Helper + News can\'t '
              'run on this machine. This is a build-side issue, '
              'not something you can fix locally. Please send us '
              'a diagnostic bundle and we\'ll ship a corrected '
              'installer.',
              style: TextStyle(
                fontSize: 12,
                color: ForexAiTokens.textPrimary,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                FilledButton.icon(
                  onPressed: () => showReportIssueDialog(
                    context,
                    prefillDescription:
                        'AI Helper screen reports the Gemma 4 runtime '
                        'is not compiled into this installer. Backend '
                        'message: ${s.message}',
                    category: 'Gemma runtime missing',
                  ),
                  icon: const Icon(Icons.bug_report, size: 16),
                  label: const Text('Report issue'),
                ),
                const SizedBox(width: 8),
                OutlinedButton.icon(
                  onPressed: () => ref.invalidate(gemmaStatusProvider),
                  icon: const Icon(Icons.refresh, size: 16),
                  label: const Text('Re-check status'),
                ),
              ],
            ),
          ],
        ),
      );
    }
    // Runtime is compiled in — show the live downloader.
    return _GemmaDownloader(status: s);
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

/// First-launch downloader card. Shown by AI Helper when the Gemma
/// runtime is compiled in but the GGUF file is missing — typically
/// the case after a Lite-SKU install where the NSIS install-time
/// fetch was skipped or interrupted. Hits the new POST /gemma/download
/// endpoint and renders live progress via gemmaDownloadStatusProvider.
class _GemmaDownloader extends ConsumerStatefulWidget {
  final GemmaStatusSnapshot status;
  const _GemmaDownloader({required this.status});

  @override
  ConsumerState<_GemmaDownloader> createState() => _GemmaDownloaderState();
}

class _GemmaDownloaderState extends ConsumerState<_GemmaDownloader> {
  bool _starting = false;

  Future<void> _start() async {
    setState(() => _starting = true);
    try {
      await ref.read(backendClientProvider).startGemmaDownload();
      // Force an immediate status refresh so the UI flips from
      // "Start Download" to the progress bar without waiting for
      // the next poll tick.
      ref.invalidate(gemmaDownloadStatusProvider);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: ForexAiTokens.sell,
          content: Text('Download start failed: $e'),
        ),
      );
    } finally {
      if (mounted) setState(() => _starting = false);
    }
  }

  Future<void> _cancel() async {
    try {
      await ref.read(backendClientProvider).cancelGemmaDownload();
      ref.invalidate(gemmaDownloadStatusProvider);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: ForexAiTokens.sell,
          content: Text('Cancel failed: $e'),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.status;
    final dlAsync = ref.watch(gemmaDownloadStatusProvider);
    final approxGiB = (s.expectedSizeBytes / (1024 * 1024 * 1024)).toStringAsFixed(1);

    return SectionCard(
      title: 'Gemma 4 model not on disk',
      child: dlAsync.when(
        data: (dl) {
          if (dl.isDownloading) return _progressView(dl);
          if (dl.isCompleted) return _completedView(dl);
          if (dl.isFailed) return _failedView(dl, approxGiB);
          if (dl.isCancelled) return _cancelledView(approxGiB);
          // idle — show the start button
          return _startView(approxGiB, s);
        },
        loading: () => const Padding(
          padding: EdgeInsets.symmetric(vertical: 16),
          child: Text(
            'Checking download status…',
            style: TextStyle(fontSize: 12, color: ForexAiTokens.textMuted),
          ),
        ),
        error: (err, _) => _startView(approxGiB, s),
      ),
    );
  }

  Widget _startView(String approxGiB, GemmaStatusSnapshot s) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'The AI Helper + News features need the local Gemma 4 model. '
          'Click below to fetch it (~$approxGiB GiB) from HuggingFace. '
          'The download runs in the background; you can keep using the '
          'rest of NeoEthos while it finishes.',
          style: const TextStyle(
            fontSize: 12,
            color: ForexAiTokens.textPrimary,
          ),
        ),
        const SizedBox(height: 8),
        _Kv(k: 'File', v: s.expectedFilename),
        _Kv(k: 'Size', v: '≈ $approxGiB GiB'),
        const SizedBox(height: 12),
        Row(
          children: [
            FilledButton.icon(
              onPressed: _starting ? null : _start,
              icon: _starting
                  ? const SizedBox(
                      width: 14,
                      height: 14,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.cloud_download, size: 16),
              label: Text(_starting ? 'Starting…' : 'Download Gemma 4'),
            ),
            const SizedBox(width: 8),
            OutlinedButton.icon(
              onPressed: () => ref.invalidate(gemmaStatusProvider),
              icon: const Icon(Icons.refresh, size: 16),
              label: const Text('Re-check status'),
            ),
          ],
        ),
      ],
    );
  }

  Widget _progressView(GemmaDownloadStatus dl) {
    final mb = (dl.bytesDone / (1024 * 1024)).toStringAsFixed(1);
    final totalMb = dl.bytesTotal > 0
        ? (dl.bytesTotal / (1024 * 1024)).toStringAsFixed(1)
        : '?';
    final percent = dl.fraction == null
        ? '—'
        : '${(dl.fraction! * 100).toStringAsFixed(1)} %';
    final mins = (dl.elapsedSeconds ~/ 60).toString().padLeft(2, '0');
    final secs = (dl.elapsedSeconds % 60).toString().padLeft(2, '0');
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Downloading the Gemma 4 GGUF from HuggingFace.',
          style: const TextStyle(fontSize: 12, color: ForexAiTokens.textPrimary),
        ),
        const SizedBox(height: 10),
        LinearProgressIndicator(
          value: dl.fraction,
          minHeight: 6,
          backgroundColor: ForexAiTokens.border,
        ),
        const SizedBox(height: 8),
        Text(
          '$mb / $totalMb MB · $percent · elapsed $mins:$secs',
          style: const TextStyle(fontSize: 11, color: ForexAiTokens.textMuted),
        ),
        const SizedBox(height: 12),
        OutlinedButton.icon(
          onPressed: _cancel,
          icon: const Icon(Icons.cancel_outlined, size: 16),
          label: const Text('Cancel download'),
        ),
      ],
    );
  }

  Widget _completedView(GemmaDownloadStatus dl) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(Icons.check_circle, size: 18, color: ForexAiTokens.buy),
            const SizedBox(width: 6),
            const Text(
              'Download complete',
              style: TextStyle(
                fontWeight: FontWeight.w700,
                color: ForexAiTokens.buy,
              ),
            ),
          ],
        ),
        if (dl.writtenPath != null) ...[
          const SizedBox(height: 6),
          SelectableText(
            dl.writtenPath!,
            style: const TextStyle(fontSize: 10, color: ForexAiTokens.textMuted),
          ),
        ],
        const SizedBox(height: 10),
        FilledButton.icon(
          onPressed: () {
            ref.invalidate(gemmaStatusProvider);
            ref.invalidate(gemmaDownloadStatusProvider);
          },
          icon: const Icon(Icons.refresh, size: 16),
          label: const Text('Reload AI Helper'),
        ),
      ],
    );
  }

  Widget _failedView(GemmaDownloadStatus dl, String approxGiB) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Download failed: ${dl.error ?? "unknown error"}',
          style: const TextStyle(color: ForexAiTokens.sell, fontSize: 12),
        ),
        const SizedBox(height: 10),
        Row(
          children: [
            FilledButton.icon(
              onPressed: _starting ? null : _start,
              icon: const Icon(Icons.refresh, size: 16),
              label: const Text('Retry'),
            ),
            const SizedBox(width: 8),
            Text(
              '($approxGiB GiB)',
              style: const TextStyle(
                fontSize: 11,
                color: ForexAiTokens.textMuted,
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _cancelledView(String approxGiB) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'Download cancelled. The partial file has been cleaned up.',
          style: TextStyle(color: ForexAiTokens.textMuted, fontSize: 12),
        ),
        const SizedBox(height: 10),
        FilledButton.icon(
          onPressed: _starting ? null : _start,
          icon: const Icon(Icons.cloud_download, size: 16),
          label: Text('Restart download ($approxGiB GiB)'),
        ),
      ],
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
