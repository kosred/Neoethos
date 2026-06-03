// Sidebar — left rail with the 6 canonical top-level tabs
// (Dashboard, Market Watch, Strategy Lab, Positions, AI Desk, Settings).
//
// **F-321 (2026-05-29 rebuild)**: was a 14-panel grouped nav
// (Trading / AI Engine / System sections with letter-spaced section
// dividers). Codex mockup replaced that with a 6-tab flat list — fewer
// destinations, each one a richer consolidated screen. The old
// section-divider styling is gone; only the active-state left border
// + accent-muted background carries over.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../l10n/app_localizations.dart';
import '../state/nav.dart';
import '../theme/theme.dart';

/// Maps a stable [NavTab.id] to its localized sidebar label. The English
/// fallbacks in nav.dart's `kNavTabs` stay as-is for non-localized contexts
/// (analytics, breadcrumbs, last-active-tab persistence); only the visible
/// rail label is localized here.
String _navTitle(AppLocalizations l10n, String id) {
  switch (id) {
    case 'Dashboard':
      return l10n.navDashboard;
    case 'MarketWatch':
      return l10n.navMarketWatch;
    case 'StrategyLab':
      return l10n.navStrategyLab;
    case 'Positions':
      return l10n.navPositions;
    case 'AiDesk':
      return l10n.navAiDesk;
    case 'Settings':
      return l10n.navSettings;
    default:
      return id;
  }
}

class Sidebar extends ConsumerWidget {
  const Sidebar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final active = ref.watch(activeTabProvider);
    return Container(
      width: NeoethosTokens.sidebarWidth,
      decoration: const BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border(right: BorderSide(color: NeoethosTokens.border)),
      ),
      padding: const EdgeInsets.symmetric(
        vertical: NeoethosTokens.spMd,
        horizontal: NeoethosTokens.spSm,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Brand block at the top — mirrors the Codex mockup which
          // anchors a small wordmark above the nav rather than relying
          // on the TopBar alone.
          const _BrandBlock(),
          const SizedBox(height: NeoethosTokens.spMd),
          Expanded(
            child: ListView(
              padding: EdgeInsets.zero,
              children: [
                for (final tab in kNavTabs)
                  _NavItem(
                    tab: tab,
                    active: active == tab.id,
                    onTap: () => ref
                        .read(activeTabProvider.notifier)
                        .state = tab.id,
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _BrandBlock extends StatelessWidget {
  const _BrandBlock();
  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(
        left: 4,
        right: 4,
        top: 2,
        bottom: 2,
      ),
      child: Row(
        children: [
          // Tiny mark — keeps the rail visually anchored without
          // duplicating the topbar brand. Plain glyph, no asset
          // dependency so the bundle stays small.
          Container(
            width: 22,
            height: 22,
            decoration: BoxDecoration(
              color: NeoethosTokens.accent.withValues(alpha: 0.18),
              border: Border.all(
                color: NeoethosTokens.accent.withValues(alpha: 0.55),
              ),
              borderRadius: BorderRadius.circular(6),
            ),
            alignment: Alignment.center,
            child: const Text(
              '✦',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: NeoethosTokens.accent,
              ),
            ),
          ),
          const SizedBox(width: 8),
          const Expanded(
            child: Text(
              'NeoEthos',
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                fontSize: NeoethosTokens.fsBody + 1,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.3,
                color: NeoethosTokens.textPrimary,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _NavItem extends StatelessWidget {
  final NavTab tab;
  final bool active;
  final VoidCallback onTap;
  const _NavItem({
    required this.tab,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      cursor: SystemMouseCursors.click,
      child: GestureDetector(
        onTap: onTap,
        child: Tooltip(
          message: tab.description,
          waitDuration: const Duration(milliseconds: 600),
          child: Container(
            height: 34,
            margin: const EdgeInsets.symmetric(vertical: 1),
            padding: const EdgeInsets.only(left: 10, right: 8),
            decoration: BoxDecoration(
              color: active
                  ? NeoethosTokens.accentMuted
                  : Colors.transparent,
              borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
              border: active
                  ? const Border(
                      left: BorderSide(
                        color: NeoethosTokens.accent,
                        width: 3,
                      ),
                    )
                  : null,
            ),
            child: Row(
              children: [
                SizedBox(
                  width: 22,
                  child: Text(
                    tab.icon,
                    style: TextStyle(
                      fontSize: 15,
                      color: active
                          ? NeoethosTokens.accent
                          : NeoethosTokens.textFaint,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    _navTitle(AppLocalizations.of(context)!, tab.id),
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      fontSize: NeoethosTokens.fsBody,
                      fontWeight: active ? FontWeight.w600 : FontWeight.w500,
                      color: active
                          ? NeoethosTokens.textPrimary
                          : NeoethosTokens.textMuted,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
