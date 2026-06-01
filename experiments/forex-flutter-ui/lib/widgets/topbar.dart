// TopBar — brand + connection badges + live ribbon items.
//
// Mirrors the .topbar block in mockups/ui_mockup.html, but every
// numeric / status value now reads from `accountSnapshotProvider`
// instead of hardcoded mockup figures.
//
// Render rules:
//   - Brand:            always "NeoEthos".
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
import '../api/currency_format.dart';
import '../screens/help_screen.dart';
import '../state/account_provider.dart';
import '../state/system_providers.dart';
import '../theme/theme.dart';
import 'backend_diagnostics_dialog.dart';
import 'report_issue.dart';

class TopBar extends ConsumerWidget {
  const TopBar({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final asyncSnapshot = ref.watch(accountSnapshotProvider);
    final brokerStatus = ref.watch(brokerStatusProvider);
    final engines = ref.watch(enginesProvider).valueOrNull;
    final discoveryOn = (engines?.discovery.toLowerCase() ?? 'idle') == 'running';
    final trainingOn = (engines?.training.toLowerCase() ?? 'idle') == 'running';

    // **2026-05-26 fix (Κωνσταντίνος)**: badge previously always said
    // "LIVE" whenever AccountSnapshot had data, regardless of whether
    // the broker session was Demo or Live. Now reads /broker/status's
    // `environment` field directly: shows "DEMO" for Demo accounts,
    // "LIVE" only for real-money accounts. Account-snapshot
    // availability still drives the CONNECTING/OFFLINE states for
    // back-compat with the no-broker-status case.
    final (badgeLabel, badgeKind) = switch (brokerStatus) {
      AsyncData(value: final b) when b.connected =>
        b.environment.toLowerCase() == 'live'
            ? ('LIVE', _BadgeKind.live)
            : ('DEMO', _BadgeKind.idle),
      AsyncData() => ('OFFLINE', _BadgeKind.offline),
      AsyncError() => ('OFFLINE', _BadgeKind.offline),
      _ => switch (asyncSnapshot) {
          AsyncData() => ('CONNECTING', _BadgeKind.idle),
          AsyncError(error: final e) when e is BrokerNotReadyException =>
            ('CONNECTING', _BadgeKind.idle),
          AsyncError() => ('OFFLINE', _BadgeKind.offline),
          _ => ('CONNECTING', _BadgeKind.idle),
        },
    };

    final snap = asyncSnapshot.valueOrNull;
    final currencySymbol = currencyGlyph(snap?.currency ?? 'EUR');
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
          const SizedBox(width: ForexAiTokens.spXs),
          const _AccountSwitcher(),
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
          // Auto-Discover / Auto-Train pills are now backed by
          // `/engines/status`. Until the POST start/stop endpoints
          // land they're read-only mirrors of whatever the bridge is
          // doing, but they no longer lie about state.
          _AutoPill(label: 'Auto-Discover', on: discoveryOn),
          const SizedBox(width: ForexAiTokens.spXs),
          _AutoPill(label: 'Auto-Train', on: trainingOn),
          const SizedBox(width: ForexAiTokens.spSm),
          IconButton(
            onPressed: () => ref
                .read(accountSnapshotProvider.notifier)
                .refreshNow(),
            tooltip: 'Refresh account snapshot now',
            icon: const Icon(Icons.refresh, color: ForexAiTokens.textMuted),
          ),
          // Help — F1 keyboard shortcut also opens this. Greek + English
          // docs across 6 sections (Welcome, Trading, AI Engine, Risk,
          // Shortcuts, FAQ). Lifted from the Codex mockup in F-329.
          IconButton(
            onPressed: () => showHelpDialog(context),
            tooltip: 'Help (F1) — Welcome guide, trading, AI, risk, FAQs',
            icon: const Icon(Icons.help_outline,
                color: ForexAiTokens.textMuted),
          ),
          // Report Issue — single, always-visible entry point. End
          // users can't rebuild the app, so this is the canonical way
          // to bundle today's logs + redacted config and email them
          // to NeoEthos support. Wired here so the button is reachable
          // from any screen, even when a panel further down is the one
          // that broke.
          IconButton(
            onPressed: () => showReportIssueDialog(context),
            tooltip: 'Report an issue — bundles logs + emails support',
            icon: const Icon(Icons.bug_report_outlined,
                color: ForexAiTokens.textMuted),
          ),
          // Backend diagnostics — supervisor.log tail + Restart button.
          // Always visible (not gated on degraded state) so the
          // operator can audit a healthy backend too.
          IconButton(
            onPressed: () => showBackendDiagnosticsDialog(context),
            tooltip: 'Backend diagnostics — view supervisor log + restart',
            icon: const Icon(Icons.monitor_heart_outlined,
                color: ForexAiTokens.textMuted),
          ),
          // **2026-05-25 — task #241**: dead "Notifications (TODO)"
          // button removed. The Backend diagnostics icon above is the
          // canonical "something happened, see what" entry-point; the
          // PendingActionsBanner handles LLM-proposed trades; the
          // BackendHealthBanner handles connectivity. A separate
          // top-level notifications inbox would be 4th-tier UX clutter
          // until we have a real notification stream to populate it.
        ],
      ),
    );
  }
}

/// Accessible cTrader account switcher in the top bar (operator
/// feedback: the 7-account picker was buried under Settings → App). Lists
/// every account the OAuth grant exposes (DEMO/LIVE badged), marks the
/// active one, and switches on tap. The switch reorders
/// broker_credentials.toml server-side and takes effect on the next
/// backend start — the SnackBar says so.
class _AccountSwitcher extends ConsumerWidget {
  const _AccountSwitcher();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final accounts =
        ref.watch(brokerAccountsProvider).valueOrNull?.accounts ?? const [];
    final currentId =
        ref.watch(brokerStatusProvider).valueOrNull?.accountId ?? '';
    if (accounts.isEmpty) {
      // No OAuth grant loaded yet — nothing to switch between.
      return const SizedBox.shrink();
    }
    return PopupMenuButton<String>(
      tooltip: 'Switch cTrader account (applies on restart)',
      offset: const Offset(0, 36),
      color: ForexAiTokens.panelBg,
      onSelected: (id) => _select(context, ref, id),
      itemBuilder: (_) => [
        for (final a in accounts)
          PopupMenuItem<String>(
            value: a.accountId,
            height: 38,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _Badge(
                  label: (a.isLive ?? false) ? 'LIVE' : 'DEMO',
                  kind: (a.isLive ?? false)
                      ? _BadgeKind.live
                      : _BadgeKind.idle,
                ),
                const SizedBox(width: 8),
                Text(
                  a.accountId,
                  style: TextStyle(
                    fontSize: ForexAiTokens.fsBody,
                    fontWeight: a.accountId == currentId
                        ? FontWeight.w800
                        : FontWeight.w500,
                    color: ForexAiTokens.textPrimary,
                  ),
                ),
                if (a.accountId == currentId) ...[
                  const SizedBox(width: 8),
                  const Icon(Icons.check, size: 14, color: ForexAiTokens.buy),
                ],
              ],
            ),
          ),
      ],
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 3, horizontal: 8),
        decoration: BoxDecoration(
          color: ForexAiTokens.appBg,
          border: Border.all(color: ForexAiTokens.border),
          borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.account_circle_outlined,
                size: 14, color: ForexAiTokens.textMuted),
            const SizedBox(width: 5),
            Text(
              currentId.isEmpty ? 'Account' : currentId,
              style: const TextStyle(
                fontSize: ForexAiTokens.fsCaption,
                fontWeight: FontWeight.w700,
                color: ForexAiTokens.textPrimary,
              ),
            ),
            const Icon(Icons.arrow_drop_down,
                size: 16, color: ForexAiTokens.textMuted),
          ],
        ),
      ),
    );
  }

  Future<void> _select(BuildContext context, WidgetRef ref, String id) async {
    final messenger = ScaffoldMessenger.of(context);
    try {
      final r = await ref
          .read(backendClientProvider)
          .selectBrokerAccount(accountId: id);
      final ok = r['ok'] == true;
      ref.invalidate(brokerStatusProvider);
      ref.invalidate(accountSnapshotProvider);
      messenger.showSnackBar(SnackBar(
        backgroundColor: ok ? ForexAiTokens.buy : ForexAiTokens.warning,
        duration: const Duration(seconds: 5),
        content: Text(ok
            ? 'Active account → $id. Restart NeoEthos to apply.'
            : 'Account select returned an unexpected response.'),
      ));
    } catch (e) {
      messenger.showSnackBar(SnackBar(
        backgroundColor: ForexAiTokens.sell,
        duration: const Duration(seconds: 5),
        content: Text('Account switch failed: $e'),
      ));
    }
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
