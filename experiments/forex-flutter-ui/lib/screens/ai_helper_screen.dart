// AI Helper — ChatGPT-subscription-backed chat assistant.
//
// Defensive design: probe /auth/codex/status on mount, show the
// Connect CTA when the operator hasn't linked their ChatGPT
// subscription yet, switch to chat-mode only when authenticated.
// Inference is blocking; we render a spinner while waiting.

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:url_launcher/url_launcher.dart';

import '../api/backend_client.dart';
import '../api/error_translation.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '_placeholder.dart';

class _Message {
  final bool fromUser;
  final String text;
  const _Message({required this.fromUser, required this.text});
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
      final r = await ref.read(backendClientProvider).codexChat(
            prompt: prompt,
            maxTokens: 512,
          );
      if (!mounted) return;
      setState(() {
        _messages.add(_Message(fromUser: false, text: r.response.trim()));
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
    final status = ref.watch(codexStatusProvider);
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const ViewHeader(
            title: 'AI Helper',
            subtitle: 'ChatGPT subscription · no API key',
          ),
          status.when(
            data: (s) =>
                s.authenticated ? _chatUi(s) : _ConnectCard(status: s),
            loading: () => const _Loading(),
            error: (err, _) => _Error(error: err.toString()),
          ),
        ],
      ),
    );
  }

  Widget _chatUi(CodexStatusSnapshot s) {
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
                  s.email != null && s.email!.isNotEmpty
                      ? 'Signed in as ${s.email}'
                      : 'Signed in via ChatGPT subscription',
                  style: const TextStyle(
                    fontSize: 11,
                    color: ForexAiTokens.textMuted,
                  ),
                ),
              ),
              TextButton.icon(
                onPressed: _busy ? null : _logout,
                icon: const Icon(Icons.logout, size: 14),
                label: const Text('Sign out'),
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
                          'history, strategy ideas — proxied through\n'
                          'your ChatGPT subscription.',
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
            ],
          ),
        ),
      ],
    );
  }

  Future<void> _logout() async {
    try {
      await ref.read(backendClientProvider).logoutCodex();
      ref.invalidate(codexStatusProvider);
      if (!mounted) return;
      setState(_messages.clear);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: ForexAiTokens.sell,
          content: Text('Logout failed: $e'),
        ),
      );
    }
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
              isUser ? 'you' : 'codex',
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
              child: SelectableText(
                message.text,
                style: const TextStyle(
                  fontSize: 12,
                  color: ForexAiTokens.textPrimary,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// "Connect ChatGPT" CTA shown when the operator hasn't linked their
/// subscription. Clicking Connect kicks off the PKCE flow on the
/// backend, opens the authorize URL in the system browser, then
/// polls `/auth/codex/status` every ~1 s until `authenticated` flips
/// true.
class _ConnectCard extends ConsumerStatefulWidget {
  final CodexStatusSnapshot status;
  const _ConnectCard({required this.status});

  @override
  ConsumerState<_ConnectCard> createState() => _ConnectCardState();
}

class _ConnectCardState extends ConsumerState<_ConnectCard> {
  bool _starting = false;
  bool _polling = false;
  String? _error;

  Future<void> _start() async {
    setState(() {
      _starting = true;
      _error = null;
    });
    try {
      final start = await ref.read(backendClientProvider).startCodexLogin();
      if (start.authorizeUrl.isEmpty) {
        throw StateError('Backend returned empty authorize URL');
      }
      final uri = Uri.parse(start.authorizeUrl);
      final launched = await launchUrl(
        uri,
        mode: LaunchMode.externalApplication,
      );
      if (!launched) {
        throw StateError('Could not open browser for ${start.authorizeUrl}');
      }
      if (!mounted) return;
      setState(() => _polling = true);
      // Poll status every second until authenticated or the widget
      // disposes. The backend's PKCE callback flips the flag, then
      // the next poll picks it up and the parent rebuilds into the
      // chat UI via the codexStatusProvider invalidate.
      while (mounted && _polling) {
        await Future<void>.delayed(const Duration(seconds: 1));
        if (!mounted) return;
        try {
          final s = await ref.read(backendClientProvider).fetchCodexStatus();
          if (s.authenticated) {
            ref.invalidate(codexStatusProvider);
            return;
          }
          if (s.lastError != null && s.lastError!.isNotEmpty) {
            if (!mounted) return;
            setState(() {
              _error = s.lastError;
              _polling = false;
            });
            return;
          }
        } catch (_) {
          // Transient error — keep polling.
        }
      }
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) {
        setState(() {
          _starting = false;
          _polling = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.status;
    return SectionCard(
      title: 'Connect your ChatGPT subscription',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'AI Helper proxies through your ChatGPT subscription — no '
            'OpenAI API key required. Click Connect to sign in with '
            'your ChatGPT account; we never see or store your '
            'password.',
            style: TextStyle(
              fontSize: 12,
              color: ForexAiTokens.textPrimary,
            ),
          ),
          if (s.authPath.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              'Auth state will be saved to: ${s.authPath}',
              style: const TextStyle(
                fontSize: 10,
                color: ForexAiTokens.textMuted,
              ),
            ),
          ],
          if (_error != null) ...[
            const SizedBox(height: 8),
            Text(
              'Connection error: $_error',
              style: const TextStyle(
                color: ForexAiTokens.sell,
                fontSize: 11,
              ),
            ),
          ] else if (s.lastError != null && s.lastError!.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              'Previous attempt failed: ${s.lastError}',
              style: const TextStyle(
                color: ForexAiTokens.warning,
                fontSize: 11,
              ),
            ),
          ],
          const SizedBox(height: 12),
          Row(
            children: [
              FilledButton.icon(
                onPressed: (_starting || _polling) ? null : _start,
                icon: (_starting || _polling)
                    ? const SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.link, size: 16),
                label: Text(
                  _polling
                      ? 'Waiting for browser…'
                      : (_starting ? 'Starting…' : 'Connect ChatGPT'),
                ),
              ),
              const SizedBox(width: 8),
              OutlinedButton.icon(
                onPressed: () => ref.invalidate(codexStatusProvider),
                icon: const Icon(Icons.refresh, size: 16),
                label: const Text('Re-check status'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _Loading extends StatelessWidget {
  const _Loading();
  @override
  Widget build(BuildContext context) => const Padding(
        padding: EdgeInsets.symmetric(vertical: 16),
        child: Text(
          'Checking ChatGPT subscription…',
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
