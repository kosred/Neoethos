// TopBar — brand + connection badges + live ribbon items.
//
// Mirrors the .topbar block in mockups/ui_mockup.html, but every
// numeric / status value now reads from `accountSnapshotProvider`
// instead of hardcoded mockup figures.
//
// Render rules:
//   - Brand:            always "NeoEthos" (post-rebrand).
//   - LIVE / OFFLINE:   derives from `accountSnapshotProvider` state.
//                         data        → green LIVE
//                         BrokerNot…  → muted CONNECTING
//                         other error → red OFFLINE
//                         loading     → faint CONNECTING (first launch)
//   - Ribbon (Balance/Equity/Free Margin): em-dash when no data,
//     real numbers once the bridge has filled the cache, last-known
//     numbers preserved during transient refresh errors so the
//     operator doesn't lose situational awareness.
//   - Auto pills:       static for now (Discovery / Training engine
//                       state lives behind endpoints that ship in a
//                       follow-up session).

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

import '../api/backend_client.dart';
import '../state/account_provider.dart';
import '../theme/theme.dart';

class TopBar extends ConsumerWidget {
  const TopBar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final asyncSnapshot = ref.watch(accountSnapshotProvider);

    // Connection-state badge. We collapse the four AsyncValue states
    // into the three badge tints the design system supports.
    final (badgeLabel, badgeKind) = switch (asyncSnapshot) {
      AsyncData() => ('LIVE', _BadgeKind.live),
      AsyncError(error: final e) when e is BrokerNotReadyException =>
        ('CONNECTING', _BadgeKind.idle),
      AsyncError() => ('OFFLINE', _BadgeKind.offline),
      _ => ('CONNECTING', _BadgeKind.idle),
    };

    final snap = asyncSnapshot.valueOrNull;
    final currencySymbol = snap?.currency == 'EUR' ? '€' : r'$';
    final fmt = NumberFormat.currency(symbol: currencySymbol, decimalDigits: 2);
    String ribbonValue(double? v) => v == null ? '—' : fmt.format(v);
    final equityAccent = snap == null
        ? _ValueAccent.plain
        : snap.equity > snap.balance
            ? _ValueAccent.success
            : snap.equity < snap.balance
                ? _ValueAccent.danger
                : _ValueAccent.plain;

    return Container(
      height: ForexAiTokens.topbarHeight,
      padding: const EdgeInsets.symmetric(horizontal: ForexAiTokens.spLg),
      decoration: const BoxDecoration(
        color: ForexAiTokens.panelBg,
        border: Border(bottom: BorderSide(color: ForexAiTokens.border)),
      ),
      child: Row(
        children: [
          const Text(
            'NeoEthos',
            style: TextStyle(
              fontSize: ForexAiTokens.fsSubtitle + 1,
              fontWeight: FontWeight.w700,
              color: ForexAiTokens.textPrimary,
            ),
          ),
          const SizedBox(width: ForexAiTokens.spSm),
          const _Badge(label: 'PRO', kind: _BadgeKind.pro),
          const SizedBox(width: ForexAiTokens.spXs),
          _Badge(label: badgeLabel, kind: badgeKind),
          const _VSep(),
          Expanded(
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  _RibbonItem(
                    label: 'Balance',
                    value: ribbonValue(snap?.balance),
                  ),
                  _RibbonItem(
                    label: 'Equity',
                    value: ribbonValue(snap?.equity),
                    valueAccent: equityAccent,
                  ),
                  _RibbonItem(
                    label: 'Free Margin',
                    value: ribbonValue(snap?.freeMargin),
                  ),
                  _RibbonItem(
                    label: 'Open',
                    value: snap == null ? '—' : '${snap.positions.length}',
                    valueAccent: snap != null && snap.positions.isNotEmpty
                        ? _ValueAccent.accent
                        : _ValueAccent.plain,
                  ),
                ],
              ),
            ),
          ),
          // Auto-Discover / Auto-Train pills stay static for now —
          // engine-state endpoints land in a follow-up session.
          const _AutoPill(label: 'Auto-Discover', on: false),
          const SizedBox(width: ForexAiTokens.spXs),
          const _AutoPill(label: 'Auto-Train', on: false),
          const SizedBox(width: ForexAiTokens.spSm),
          IconButton(
            onPressed: () => ref
                .read(accountSnapshotProvider.notifier)
                .refreshNow(),
            tooltip: 'Refresh account snapshot now',
            icon: const Icon(Icons.refresh, color: ForexAiTokens.textMuted),
          ),
          IconButton(
            onPressed: () {},
            tooltip: 'Notifications (TODO)',
            icon: const Icon(Icons.notifications_none,
                color: ForexAiTokens.textMuted),
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
          const Color(0x29002962),
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
  Widget build(BuildContext context) => Container(
        width: 1,
        height: 28,
        color: ForexAiTokens.border,
        margin: const EdgeInsets.symmetric(horizontal: ForexAiTokens.spSm),
      );
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
