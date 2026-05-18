// Sidebar — left rail with the 14-panel nav grouped by Trading /
// AI Engine / System (mirrors the mockup's .sidebar block).

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../state/nav.dart';
import '../theme/theme.dart';

class Sidebar extends ConsumerWidget {
  const Sidebar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final active = ref.watch(activeTabProvider);
    final groups = [NavGroup.trading, NavGroup.aiEngine, NavGroup.system];
    return Container(
      width: ForexAiTokens.sidebarWidth,
      decoration: const BoxDecoration(
        color: ForexAiTokens.panelBg,
        border: Border(right: BorderSide(color: ForexAiTokens.border)),
      ),
      padding: const EdgeInsets.symmetric(
        vertical: ForexAiTokens.spMd,
        horizontal: ForexAiTokens.spSm,
      ),
      child: ListView(
        children: [
          for (var i = 0; i < groups.length; i++) ...[
            if (i > 0) const SizedBox(height: ForexAiTokens.spLg),
            _SectionLabel(label: navGroupLabel(groups[i])),
            ...kNavTabs
                .where((t) => t.group == groups[i])
                .map((t) => _NavItem(
                      tab: t,
                      active: active == t.id,
                      onTap: () => ref
                          .read(activeTabProvider.notifier)
                          .state = t.id,
                    )),
          ],
        ],
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final String label;
  const _SectionLabel({required this.label});
  @override
  Widget build(BuildContext context) {
    // Letter-spaced uppercase divider, matches the mockup.
    return Container(
      padding: const EdgeInsets.all(ForexAiTokens.spXs),
      margin: const EdgeInsets.only(bottom: ForexAiTokens.spXs),
      decoration: const BoxDecoration(
        border: Border(bottom: BorderSide(color: ForexAiTokens.border)),
      ),
      child: Text(
        label.split('').join(' '),
        style: const TextStyle(
          fontSize: ForexAiTokens.fsCaption - 1,
          letterSpacing: 1.4,
          color: ForexAiTokens.textFaint,
          fontWeight: FontWeight.w700,
        ),
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
        child: Container(
          height: 28,
          padding: const EdgeInsets.only(left: 12),
          decoration: BoxDecoration(
            color: active
                ? ForexAiTokens.accentMuted
                : Colors.transparent,
            borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
            border: active
                ? const Border(
                    left: BorderSide(
                      color: ForexAiTokens.accent,
                      width: 3,
                    ),
                  )
                : null,
          ),
          child: Row(
            children: [
              SizedBox(
                width: 20,
                child: Text(
                  tab.icon,
                  style: TextStyle(
                    fontSize: 14,
                    color: active
                        ? ForexAiTokens.accent
                        : ForexAiTokens.textFaint,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                tab.title,
                style: TextStyle(
                  fontSize: ForexAiTokens.fsBody,
                  color: active
                      ? ForexAiTokens.textPrimary
                      : ForexAiTokens.textMuted,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
