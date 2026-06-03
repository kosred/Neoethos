// AI news desk panel — lives in the AI Desk right-rail.
//
// Renders the `/news/feed` payload: a Codex-written market briefing on
// top (collapsible) + an AUTO-SCROLLING list of public-RSS headlines.
// Tapping a headline expands it INLINE (the blurb + an "Open in browser"
// button) without pushing a route or stealing focus from the rest of the
// rail — exactly the operator's spec: "opens the news without losing
// focus from the rest of the panels, and can open more in browser by
// getting the link".
//
// Distribution-safe: the headlines come from the backend (no API key),
// and the AI briefing reuses the operator's ChatGPT-subscription Codex
// link. A user with only a ChatGPT login sees the full experience; a
// user who hasn't connected Codex still sees the live headlines plus a
// one-line hint on how to enable the briefing.

import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:url_launcher/url_launcher.dart';

import '../api/backend_client.dart';
import '../l10n/app_localizations.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';

class NewsPanel extends ConsumerStatefulWidget {
  const NewsPanel({super.key});

  @override
  ConsumerState<NewsPanel> createState() => _NewsPanelState();
}

class _NewsPanelState extends ConsumerState<NewsPanel> {
  /// Fixed height of the auto-scrolling headline list inside the 280 px
  /// rail. Tall enough to show ~4 headlines, short enough to leave room
  /// for the AI-Desk sections below it.
  static const double _listHeight = 230;

  final ScrollController _scroll = ScrollController();
  Timer? _ticker;

  /// Auto-scroll pauses while the pointer is over the list or an item is
  /// expanded, so a reading operator is never yanked away mid-headline.
  bool _hovering = false;
  int? _expanded; // index of the inline-expanded headline, if any
  bool _briefingExpanded = false;
  bool _refreshing = false;

  @override
  void initState() {
    super.initState();
    // ~10 px/s creep: slow enough to read, brisk enough to feel "live".
    _ticker = Timer.periodic(const Duration(milliseconds: 40), (_) {
      if (!_scroll.hasClients) return;
      if (_hovering || _expanded != null) return;
      final max = _scroll.position.maxScrollExtent;
      if (max <= 0) return;
      var next = _scroll.offset + 0.4;
      if (next >= max) next = 0; // loop back to the top
      _scroll.jumpTo(next.clamp(0.0, max));
    });
  }

  @override
  void dispose() {
    _ticker?.cancel();
    _scroll.dispose();
    super.dispose();
  }

  Future<void> _refresh() async {
    if (_refreshing) return;
    setState(() => _refreshing = true);
    // force=true makes the backend rebuild + re-cache a fresh feed; the
    // invalidate then re-reads that fresh result through the provider.
    try {
      await ref.read(backendClientProvider).fetchNewsFeed(force: true);
    } catch (_) {
      // Swallow — the provider re-fetch surfaces any error state in-UI.
    }
    ref.invalidate(newsFeedProvider);
    if (mounted) setState(() => _refreshing = false);
  }

  Future<void> _openInBrowser(String link) async {
    if (link.isEmpty) return;
    final uri = Uri.tryParse(link);
    if (uri == null) return;
    await launchUrl(uri, mode: LaunchMode.externalApplication);
  }

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    final async = ref.watch(newsFeedProvider);
    final autoScrolling = !_hovering && _expanded == null;
    return Container(
      padding: const EdgeInsets.all(NeoethosTokens.spMd),
      decoration: BoxDecoration(
        color: NeoethosTokens.appBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  l10n.newsTitle,
                  style: const TextStyle(
                    fontSize: NeoethosTokens.fsCaption,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.6,
                    color: NeoethosTokens.textMuted,
                  ),
                ),
              ),
              _LiveDot(active: autoScrolling),
              const SizedBox(width: 7),
              GestureDetector(
                onTap: _refresh,
                behavior: HitTestBehavior.opaque,
                child: _refreshing
                    ? const SizedBox(
                        width: 13,
                        height: 13,
                        child: CircularProgressIndicator(
                          strokeWidth: 1.6,
                          color: NeoethosTokens.textMuted,
                        ),
                      )
                    : const Icon(
                        Icons.refresh,
                        size: 15,
                        color: NeoethosTokens.textMuted,
                      ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          async.when(
            loading: () => const _NewsSkeleton(),
            error: (_, __) => Text(
              l10n.newsOffline,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsCaption,
                color: NeoethosTokens.textFaint,
                fontStyle: FontStyle.italic,
              ),
            ),
            data: _buildContent,
          ),
        ],
      ),
    );
  }

  Widget _buildContent(NewsFeed feed) {
    final l10n = AppLocalizations.of(context)!;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (feed.aiAvailable && feed.aiSummary.isNotEmpty)
          _briefing(feed.aiSummary)
        else if (feed.items.isNotEmpty)
          _connectHint(),
        if (feed.notice.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Text(
              feed.notice,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsCaption - 1,
                color: NeoethosTokens.textFaint,
                height: 1.3,
              ),
            ),
          ),
        if (feed.items.isEmpty)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8),
            child: Text(
              l10n.newsNoHeadlines,
              style: const TextStyle(
                fontSize: NeoethosTokens.fsCaption,
                color: NeoethosTokens.textFaint,
              ),
            ),
          )
        else
          MouseRegion(
            onEnter: (_) => setState(() => _hovering = true),
            onExit: (_) => setState(() => _hovering = false),
            child: SizedBox(
              height: _listHeight,
              child: ListView.builder(
                controller: _scroll,
                physics: const ClampingScrollPhysics(),
                padding: EdgeInsets.zero,
                itemCount: feed.items.length,
                itemBuilder: (_, i) => _NewsRow(
                  item: feed.items[i],
                  expanded: _expanded == i,
                  onTap: () => setState(
                    () => _expanded = _expanded == i ? null : i,
                  ),
                  onOpen: () => _openInBrowser(feed.items[i].link),
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _briefing(String summary) {
    final l10n = AppLocalizations.of(context)!;
    final collapsed = !_briefingExpanded;
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(NeoethosTokens.spSm),
      decoration: BoxDecoration(
        color: NeoethosTokens.accentMuted,
        border: Border.all(color: NeoethosTokens.accent.withValues(alpha: 0.4)),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              const Icon(Icons.auto_awesome, size: 12, color: NeoethosTokens.accent),
              const SizedBox(width: 5),
              Expanded(
                child: Text(
                  l10n.newsAiBriefing,
                  style: const TextStyle(
                    fontSize: NeoethosTokens.fsCaption - 1,
                    fontWeight: FontWeight.w800,
                    letterSpacing: 0.5,
                    color: NeoethosTokens.accent,
                  ),
                ),
              ),
              GestureDetector(
                onTap: () =>
                    setState(() => _briefingExpanded = !_briefingExpanded),
                behavior: HitTestBehavior.opaque,
                child: Icon(
                  collapsed ? Icons.expand_more : Icons.expand_less,
                  size: 16,
                  color: NeoethosTokens.textMuted,
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            summary,
            maxLines: collapsed ? 3 : null,
            overflow: collapsed ? TextOverflow.ellipsis : TextOverflow.visible,
            style: const TextStyle(
              fontSize: NeoethosTokens.fsCaption,
              color: NeoethosTokens.textPrimary,
              height: 1.45,
            ),
          ),
        ],
      ),
    );
  }

  Widget _connectHint() {
    final l10n = AppLocalizations.of(context)!;
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(NeoethosTokens.spSm),
      decoration: BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Text(
        l10n.newsConnectHint,
        style: const TextStyle(
          fontSize: NeoethosTokens.fsCaption - 1,
          color: NeoethosTokens.textFaint,
          height: 1.4,
        ),
      ),
    );
  }
}

/// One headline row — collapsed shows title (2 lines) + source/age;
/// expanded reveals the blurb and the "Open in browser" action.
class _NewsRow extends StatelessWidget {
  final NewsItem item;
  final bool expanded;
  final VoidCallback onTap;
  final VoidCallback onOpen;
  const _NewsRow({
    required this.item,
    required this.expanded,
    required this.onTap,
    required this.onOpen,
  });

  @override
  Widget build(BuildContext context) {
    final l10n = AppLocalizations.of(context)!;
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          GestureDetector(
            onTap: onTap,
            behavior: HitTestBehavior.opaque,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: 5,
                      height: 5,
                      margin: const EdgeInsets.only(top: 5, right: 7),
                      decoration: const BoxDecoration(
                        color: NeoethosTokens.accent,
                        shape: BoxShape.circle,
                      ),
                    ),
                    Expanded(
                      child: Text(
                        item.title,
                        maxLines: expanded ? null : 2,
                        overflow: expanded
                            ? TextOverflow.visible
                            : TextOverflow.ellipsis,
                        style: TextStyle(
                          fontSize: NeoethosTokens.fsCaption,
                          fontWeight:
                              expanded ? FontWeight.w700 : FontWeight.w600,
                          color: NeoethosTokens.textPrimary,
                          height: 1.35,
                        ),
                      ),
                    ),
                  ],
                ),
                Padding(
                  padding: const EdgeInsets.only(left: 12, top: 2),
                  child: Text(
                    '${item.source}${_age(context, item.publishedMs)}',
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      fontSize: NeoethosTokens.fsCaption - 2,
                      color: NeoethosTokens.textFaint,
                    ),
                  ),
                ),
              ],
            ),
          ),
          if (expanded) ...[
            if (item.blurb.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(left: 12, top: 5),
                child: Text(
                  item.blurb,
                  style: const TextStyle(
                    fontSize: NeoethosTokens.fsCaption - 1,
                    color: NeoethosTokens.textMuted,
                    height: 1.4,
                  ),
                ),
              ),
            Padding(
              padding: const EdgeInsets.only(left: 12, top: 6),
              child: Align(
                alignment: Alignment.centerLeft,
                child: OutlinedButton.icon(
                  onPressed: item.link.isEmpty ? null : onOpen,
                  icon: const Icon(Icons.open_in_new, size: 13),
                  label: Text(
                    l10n.newsOpenInBrowser,
                    style: const TextStyle(
                      fontSize: NeoethosTokens.fsCaption - 1,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: NeoethosTokens.accent,
                    side: BorderSide(
                      color: NeoethosTokens.accent.withValues(alpha: 0.5),
                    ),
                    padding:
                        const EdgeInsets.symmetric(horizontal: 9, vertical: 3),
                    minimumSize: const Size(0, 26),
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                ),
              ),
            ),
          ],
          const Padding(
            padding: EdgeInsets.only(top: 6),
            child: Divider(height: 1, color: NeoethosTokens.border),
          ),
        ],
      ),
    );
  }

  /// Compact relative-age suffix (" · 2h", " · now", "" when undated).
  String _age(BuildContext context, int? ms) {
    if (ms == null || ms <= 0) return '';
    final l10n = AppLocalizations.of(context)!;
    final then = DateTime.fromMillisecondsSinceEpoch(ms);
    final d = DateTime.now().difference(then);
    if (d.isNegative) return '';
    if (d.inMinutes < 1) return ' · ${l10n.newsAgeNow}';
    if (d.inMinutes < 60) return ' · ${l10n.newsAgeMinutes(d.inMinutes)}';
    if (d.inHours < 24) return ' · ${l10n.newsAgeHours(d.inHours)}';
    return ' · ${l10n.newsAgeDays(d.inDays)}';
  }
}

/// Small dot that glows accent-green while the feed is auto-scrolling and
/// dims to faint when paused (hover / expanded).
class _LiveDot extends StatelessWidget {
  final bool active;
  const _LiveDot({required this.active});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 6,
      height: 6,
      decoration: BoxDecoration(
        color: active ? NeoethosTokens.buy : NeoethosTokens.textFaint,
        shape: BoxShape.circle,
      ),
    );
  }
}

class _NewsSkeleton extends StatelessWidget {
  const _NewsSkeleton();

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        for (var i = 0; i < 4; i++)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Container(
              height: 10,
              decoration: BoxDecoration(
                color: NeoethosTokens.panelBg,
                borderRadius: BorderRadius.circular(3),
              ),
            ),
          ),
      ],
    );
  }
}
