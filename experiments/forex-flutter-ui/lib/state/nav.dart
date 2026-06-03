// Navigation state — which top-level panel is currently shown.
//
// **F-321 (2026-05-29 rebuild)**: collapsed from 14 tabs in 3 groups
// (Trading / AI Engine / System) down to **6 flat tabs** matching the
// Codex UI mockup at mockups/ig_*.png. The old tabs survive as content
// nested *inside* the new consolidated screens (e.g. Chart + Markets +
// Execution + News all live under Market Watch now; Discovery +
// Training + Validation + Promotion live under Strategy Lab).
//
// Help is reachable via F1 (global Shortcut wired in AppShell) and via
// a sub-tab inside Settings — not a top-level sidebar entry. The old
// `NavGroup` enum is gone; the sidebar is a flat list now.

import 'package:flutter_riverpod/flutter_riverpod.dart';

class NavTab {
  final String id;
  final String icon;
  final String title;
  final String description;
  const NavTab(this.id, this.icon, this.title, this.description);
}

/// The 6 canonical top-level tabs in the sidebar, in display order.
///
/// These IDs are persisted across restarts (see #24), so renaming any
/// of them silently breaks last-active-tab restore for existing users.
const List<NavTab> kNavTabs = [
  NavTab('Dashboard', '▦', 'Dashboard',
      "Account equity, engine health, today's PnL"),
  NavTab('MarketWatch', '⌖', 'Market Watch',
      'Symbols + live quotes + per-symbol strategy/confidence/auto + '
          'open positions + pending orders'),
  NavTab('StrategyLab', '⚗', 'Strategy Lab',
      'Unified pipeline: Data Ready → Discovery → Training → '
          'Validation → Promotion Gate'),
  NavTab('Positions', '◫', 'Positions',
      'Open positions, pending orders, fills, recent activity log'),
  NavTab('AiDesk', '✺', 'AI Desk',
      'Ensemble state, predictions, proposed actions, AI assistant '
          '(also rendered as right-rail on Market Watch / Strategy Lab)'),
  NavTab('Settings', '⚙', 'Settings',
      'Broker, risk, hardware, data bootstrap, advanced knobs, help'),
];

/// Currently selected nav tab id (defaults to Dashboard).
///
/// Persisted across restarts via #24. If the persisted value points to
/// an ID that no longer exists (e.g. someone upgraded from the
/// pre-F-321 14-tab layout and had 'Discovery' as their last tab),
/// `navTabById` silently falls back to Dashboard via `orElse`.
final activeTabProvider = StateProvider<String>((ref) => 'Dashboard');

/// Resolver — useful for screens that need their own tab metadata
/// (e.g. for breadcrumb rendering or analytics). Falls back to the
/// Dashboard tab when an unknown ID is passed in.
NavTab navTabById(String id) =>
    kNavTabs.firstWhere((t) => t.id == id, orElse: () => kNavTabs.first);

/// Map legacy pre-F-321 tab IDs to their new home, so persisted
/// last-active-tab values from older installs land on the right
/// consolidated screen instead of silently bouncing to Dashboard.
///
/// Returns `id` unchanged when it's already a valid new-tab ID.
String migrateLegacyNavId(String id) {
  switch (id) {
    // Trading group → MarketWatch (chart/markets/execution/news fold
    // in there) or Positions (TradeWatch becomes Positions).
    case 'Chart':
    case 'Markets':
    case 'Execution':
    case 'News':
      return 'MarketWatch';
    case 'TradeWatch':
      return 'Positions';
    // AI Engine group → StrategyLab (Discovery + Training merge into
    // the pipeline) or AiDesk (Intelligence + AiHelper).
    case 'Discovery':
    case 'Training':
      return 'StrategyLab';
    case 'Intelligence':
    case 'AiHelper':
      return 'AiDesk';
    // System group → Settings consumes all of these as sub-tabs.
    case 'BrokerSetup':
    case 'DataBootstrap':
    case 'Hardware':
    case 'Risk':
      return 'Settings';
    default:
      return id;
  }
}
