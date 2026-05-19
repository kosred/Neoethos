// TopBar — brand + status badges + ribbon items + auto pills.
// Mirrors the .topbar block in mockups/ui_mockup.html.

import 'package:flutter/material.dart';

import '../theme/theme.dart';

class TopBar extends StatelessWidget {
  const TopBar({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: ForexAiTokens.topbarHeight,
      padding: const EdgeInsets.symmetric(horizontal: ForexAiTokens.spLg),
      decoration: const BoxDecoration(
        // The earlier scaffold set `color:` directly on the Container
        // AND a `BoxDecoration` — Flutter 3.44+ trips an assert because
        // `color` is shorthand for `BoxDecoration(color: …)` and the
        // two can't coexist. Fold the panel background into the
        // decoration so the bottom-border stays + the bg colour stays.
        color: ForexAiTokens.panelBg,
        border: Border(
          bottom: BorderSide(color: ForexAiTokens.border),
        ),
      ),
      // Three-zone Row: brand + badges on the left, ribbon items in
      // the middle (horizontally scrollable so they don't push the
      // right zone off-screen on narrow viewports), action pills +
      // icons on the right.
      child: Row(
        children: [
          const Text(
            'forex-ai',
            style: TextStyle(
              fontSize: ForexAiTokens.fsSubtitle + 1,
              fontWeight: FontWeight.w700,
              color: ForexAiTokens.textPrimary,
            ),
          ),
          const SizedBox(width: ForexAiTokens.spSm),
          const _Badge(label: 'PRO', kind: _BadgeKind.pro),
          const SizedBox(width: ForexAiTokens.spXs),
          const _Badge(label: 'LIVE', kind: _BadgeKind.live),
          const _VSep(),
          // Ribbon — placeholder values. Wired to provider in
          // a follow-up commit; for now they read the same
          // mockup figures. Wrapped in Expanded + horizontal
          // SingleChildScrollView so the ribbon shrinks gracefully
          // on narrow viewports without overflowing.
          Expanded(
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: const [
                  _RibbonItem(label: 'Balance', value: '\$10,000.00'),
                  _RibbonItem(
                    label: 'Equity',
                    value: '\$10,243.55',
                    valueAccent: _ValueAccent.success,
                  ),
                  _RibbonItem(label: 'Free Margin', value: '\$9,762.40'),
                ],
              ),
            ),
          ),
          // Right-side actions (fixed width, pinned to right).
          const _AutoPill(label: 'Auto-Discover', on: true),
          const SizedBox(width: ForexAiTokens.spXs),
          const _AutoPill(label: 'Auto-Train', on: false),
          const SizedBox(width: ForexAiTokens.spSm),
          IconButton(
            onPressed: () {},
            tooltip: 'Notifications',
            icon: const Icon(Icons.notifications_none,
                color: ForexAiTokens.textMuted),
          ),
          IconButton(
            onPressed: () {},
            tooltip: 'Command palette (⌘K)',
            icon: const Icon(Icons.search, color: ForexAiTokens.textMuted),
          ),
        ],
      ),
    );
  }
}

enum _BadgeKind { pro, live, offline, local, blackout, idle }

class _Badge extends StatelessWidget {
  final String label;
  final _BadgeKind kind;
  const _Badge({required this.label, required this.kind});

  @override
  Widget build(BuildContext context) {
    final (fg, bg, border) = switch (kind) {
      _BadgeKind.pro => (
          ForexAiTokens.accent,
          const Color(0x29002962), // alpha-blended approx
          ForexAiTokens.accent.withValues(alpha: 0.6),
        ),
      _BadgeKind.live => (
          ForexAiTokens.buy,
          ForexAiTokens.buy.withValues(alpha: 0.16),
          ForexAiTokens.buy.withValues(alpha: 0.6),
        ),
      _BadgeKind.offline => (
          ForexAiTokens.sell,
          ForexAiTokens.sell.withValues(alpha: 0.16),
          ForexAiTokens.sell.withValues(alpha: 0.6),
        ),
      _BadgeKind.local => (
          ForexAiTokens.warning,
          ForexAiTokens.warning.withValues(alpha: 0.16),
          ForexAiTokens.warning.withValues(alpha: 0.6),
        ),
      _BadgeKind.blackout => (
          ForexAiTokens.sell,
          ForexAiTokens.sell.withValues(alpha: 0.16),
          ForexAiTokens.sell.withValues(alpha: 0.6),
        ),
      _BadgeKind.idle => (
          ForexAiTokens.textFaint,
          ForexAiTokens.textFaint.withValues(alpha: 0.16),
          ForexAiTokens.textFaint.withValues(alpha: 0.6),
        ),
    };
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 2, horizontal: 8),
      decoration: BoxDecoration(
        color: bg,
        border: Border.all(color: border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: ForexAiTokens.fsCaption - 0.5,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.4,
          color: fg,
        ),
      ),
    );
  }
}

class _VSep extends StatelessWidget {
  const _VSep();
  @override
  Widget build(BuildContext context) {
    return Container(
      width: 1,
      height: 28,
      color: ForexAiTokens.border,
      margin: const EdgeInsets.symmetric(horizontal: ForexAiTokens.spSm),
    );
  }
}

enum _ValueAccent { plain, success, danger, accent }

class _RibbonItem extends StatelessWidget {
  final String label;
  final String value;
  final _ValueAccent valueAccent;
  const _RibbonItem({
    required this.label,
    required this.value,
    this.valueAccent = _ValueAccent.plain,
  });

  @override
  Widget build(BuildContext context) {
    final color = switch (valueAccent) {
      _ValueAccent.plain => ForexAiTokens.textPrimary,
      _ValueAccent.success => ForexAiTokens.buy,
      _ValueAccent.danger => ForexAiTokens.sell,
      _ValueAccent.accent => ForexAiTokens.accent,
    };
    return Padding(
      padding: const EdgeInsets.only(right: ForexAiTokens.spLg),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            label.toUpperCase(),
            style: const TextStyle(
              fontSize: ForexAiTokens.fsCaption - 1,
              letterSpacing: 0.8,
              fontWeight: FontWeight.w700,
              color: ForexAiTokens.textMuted,
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontSize: ForexAiTokens.fsBody,
              fontWeight: FontWeight.w700,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}

class _AutoPill extends StatelessWidget {
  final String label;
  final bool on;
  const _AutoPill({required this.label, required this.on});

  @override
  Widget build(BuildContext context) {
    final fg = on ? ForexAiTokens.buy : ForexAiTokens.textFaint;
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 3, horizontal: 8),
      decoration: BoxDecoration(
        color: fg.withValues(alpha: 0.15),
        border: Border.all(color: fg.withValues(alpha: 0.55)),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: ForexAiTokens.fsCaption,
          fontWeight: FontWeight.w700,
          color: fg,
        ),
      ),
    );
  }
}
