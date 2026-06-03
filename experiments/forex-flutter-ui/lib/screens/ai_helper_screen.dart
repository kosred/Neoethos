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
import '../l10n/app_localizations.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import '../widgets/backend_error_widget.dart';
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
    final l10n = AppLocalizations.of(context)!;
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
        _messages.add(
            _Message(fromUser: false, text: l10n.aiHelperErrorPrefix(msg)));
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
    final l10n = AppLocalizations.of(context)!;
    final status = ref.watch(codexStatusProvider);
    // Pin to the available height — AI Desk gives this screen a bounded
    // box (Expanded → TabBarView), so the chat input row stays visible at
    // the bottom instead of being pushed below the fold by a
    // MediaQuery-sized message box. Non-chat states stay scrollable.
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        ViewHeader(
          title: l10n.aiHelperTitle,
          subtitle: l10n.aiHelperSubtitle,
        ),
        Expanded(
          child: status.when(
            data: (s) => s.authenticated
                ? _chatUi(s)
                : SingleChildScrollView(child: _ConnectCard(status: s)),
            loading: () => const _Loading(),
            error: (err, _) => SingleChildScrollView(
                child: BackendErrorWidget(
                    error: err, title: l10n.aiHelperDeskUnavailable)),
          ),
        ),
      ],
    );
  }

  Widget _chatUi(CodexStatusSnapshot s) {
    final l10n = AppLocalizations.of(context)!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionCard(
          title: l10n.aiHelperReady,
          child: Row(
            children: [
              Container(
                width: 10,
                height: 10,
                decoration: const BoxDecoration(
                  color: NeoethosTokens.buy,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  s.email != null && s.email!.isNotEmpty
                      ? l10n.aiHelperSignedInAs(s.email!)
                      : l10n.aiHelperSignedInSubscription,
                  style: const TextStyle(
                    fontSize: 11,
                    color: NeoethosTokens.textMuted,
                  ),
                ),
              ),
              TextButton.icon(
                onPressed: _busy ? null : _logout,
                icon: const Icon(Icons.logout, size: 14),
                label: Text(l10n.aiHelperSignOut),
              ),
            ],
          ),
        ),
        Expanded(
          // Custom expanding card (not SectionCard) so the message list
          // fills all available height and the input row pins to the
          // bottom. SectionCard lays its child at natural height, which
          // would collapse the inner Expanded.
          child: Container(
            padding: const EdgeInsets.all(NeoethosTokens.spMd),
            margin: const EdgeInsets.only(top: NeoethosTokens.spSm),
            decoration: BoxDecoration(
              color: NeoethosTokens.surfaceAlt,
              border: Border.all(color: NeoethosTokens.border),
              borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  l10n.aiHelperChat,
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: NeoethosTokens.textPrimary,
                  ),
                ),
                const SizedBox(height: NeoethosTokens.spXs),
                Expanded(
                  child: _messages.isEmpty
                    ? Center(
                        child: Text(
                          l10n.aiHelperEmptyHint,
                          textAlign: TextAlign.center,
                          style: const TextStyle(
                            fontSize: 12,
                            color: NeoethosTokens.textMuted,
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
                      decoration: InputDecoration(
                        hintText: l10n.aiHelperInputHint,
                        isDense: true,
                        border: const OutlineInputBorder(),
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
                    label: Text(_busy ? l10n.aiHelperGenerating : l10n.aiHelperSend),
                  ),
                ],
              ),
            ],
          ),
          ),
        ),
      ],
    );
  }

  Future<void> _logout() async {
    final l10n = AppLocalizations.of(context)!;
    try {
      await ref.read(backendClientProvider).logoutCodex();
      ref.invalidate(codexStatusProvider);
      if (!mounted) return;
      setState(_messages.clear);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: NeoethosTokens.sell,
          content: Text(l10n.aiHelperSignOutFailed(describeError(e))),
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
    final l10n = AppLocalizations.of(context)!;
    final isUser = message.fromUser;
    final bg = isUser
        ? NeoethosTokens.accent.withValues(alpha: 0.18)
        : NeoethosTokens.surfaceBg;
    final border = isUser
        ? NeoethosTokens.accent.withValues(alpha: 0.45)
        : NeoethosTokens.border;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 56,
            child: Text(
              isUser ? l10n.aiHelperRoleYou : 'codex',
              style: TextStyle(
                fontSize: 10,
                fontWeight: FontWeight.w700,
                color: isUser ? NeoethosTokens.accent : NeoethosTokens.buy,
              ),
            ),
          ),
          Expanded(
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: bg,
                border: Border.all(color: border),
                borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
              ),
              child: SelectableText(
                message.text,
                style: const TextStyle(
                  fontSize: 12,
                  color: NeoethosTokens.textPrimary,
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
    final l10n = AppLocalizations.of(context)!;
    setState(() {
      _starting = true;
      _error = null;
    });
    try {
      final start = await ref.read(backendClientProvider).startCodexLogin();
      if (start.authorizeUrl.isEmpty) {
        throw StateError(l10n.aiHelperConnectEmptyUrl);
      }
      final uri = Uri.parse(start.authorizeUrl);
      final launched = await launchUrl(
        uri,
        mode: LaunchMode.externalApplication,
      );
      if (!launched) {
        throw StateError(l10n.aiHelperConnectBrowserFailed(start.authorizeUrl));
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
    final l10n = AppLocalizations.of(context)!;
    final s = widget.status;
    return SectionCard(
      title: l10n.aiHelperConnectTitle,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            l10n.aiHelperConnectBody,
            style: const TextStyle(
              fontSize: 12,
              color: NeoethosTokens.textPrimary,
            ),
          ),
          if (s.authPath.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              l10n.aiHelperConnectAuthPath(s.authPath),
              style: const TextStyle(
                fontSize: 10,
                color: NeoethosTokens.textMuted,
              ),
            ),
          ],
          if (_error != null) ...[
            const SizedBox(height: 8),
            Text(
              l10n.aiHelperConnectError(_error!),
              style: const TextStyle(
                color: NeoethosTokens.sell,
                fontSize: 11,
              ),
            ),
          ] else if (s.lastError != null && s.lastError!.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              l10n.aiHelperConnectPreviousFailed(s.lastError!),
              style: const TextStyle(
                color: NeoethosTokens.warning,
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
                      ? l10n.aiHelperConnectWaiting
                      : (_starting
                          ? l10n.aiHelperConnectStarting
                          : l10n.aiHelperConnectButton),
                ),
              ),
              const SizedBox(width: 8),
              OutlinedButton.icon(
                onPressed: () => ref.invalidate(codexStatusProvider),
                icon: const Icon(Icons.refresh, size: 16),
                label: Text(l10n.aiHelperRecheckStatus),
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
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 16),
        child: Text(
          AppLocalizations.of(context)!.aiHelperCheckingSubscription,
          style: const TextStyle(color: NeoethosTokens.textMuted, fontSize: 12),
        ),
      );
}

