// The grid shell — TopBar + Sidebar + Dock + StatusBar.
//
// **F-321 (2026-05-29 rebuild)**: was a 14-tab dock that routed each
// sidebar entry to its own screen widget. Now there are 6 top-level
// tabs (Dashboard, Market Watch, Strategy Lab, Positions, AI Desk,
// Settings) and the dock simply hands each one its consolidated
// screen. The breadcrumb (which read `navGroupLabel(tab.group)` over
// `tab.title`) is gone — `NavGroup` was deleted and the sidebar
// itself is the canonical "where am I" indicator.
//
// **F-323 (next, this rebuild cycle)**: this shell will grow a third
// column on the right for the AI Desk persistent right-rail on the
// trading-focused screens (Market Watch, Strategy Lab, Positions).
// For now the layout is unchanged from the pre-F-321 grid.

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../startup/backend_watchdog.dart';
import '../state/nav.dart';
import '../theme/theme.dart';
import 'ai_desk_rail.dart';
import 'backend_health_banner.dart';
import 'pending_actions_banner.dart';
import 'sidebar.dart';
import 'topbar.dart';
import 'statusbar.dart';
import '../screens/dashboard_screen.dart';
import '../screens/help_screen.dart';
import '../screens/market_watch_screen.dart';
import '../screens/strategy_lab_screen.dart';
import '../screens/positions_screen.dart';
import '../screens/ai_desk_screen.dart';
import '../screens/settings_screen.dart';

/// F1 (and the `?` key) opens the Help screen from anywhere in the app.
/// Wired in [AppShell] via the Shortcuts/Actions pair.
class _ShowHelpIntent extends Intent {
  const _ShowHelpIntent();
}

class AppShell extends ConsumerWidget {
  const AppShell({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    // Legacy IDs (Chart, Discovery, BrokerSetup, …) may have been
    // persisted by an older install. Migrate them transparently so the
    // user lands on the right consolidated screen instead of bouncing
    // to Dashboard via `navTabById`'s `orElse` fallback.
    final persistedId = ref.watch(activeTabProvider);
    final activeId = migrateLegacyNavId(persistedId);

    // Eagerly materialise the watchdog provider so it starts polling
    // on the first frame, even if the BackendHealthBanner below
    // collapses to SizedBox.shrink (healthy path). Without this read
    // the Notifier wouldn't `build()` until the banner first decided
    // to render — which would never happen on a healthy machine and
    // we'd never poll at all.
    ref.watch(backendHealthProvider);
    return Scaffold(
      backgroundColor: ForexAiTokens.appBg,
      body: Shortcuts(
        // F1 + `?` both open Help. The `?` shortcut requires Shift on a
        // US keyboard, so we register both the bare `?` key and the
        // Shift+/ combo so it works regardless of how Flutter resolves
        // the keyboard layout on Windows.
        shortcuts: <LogicalKeySet, Intent>{
          LogicalKeySet(LogicalKeyboardKey.f1): const _ShowHelpIntent(),
          LogicalKeySet(LogicalKeyboardKey.question): const _ShowHelpIntent(),
          LogicalKeySet(LogicalKeyboardKey.shift, LogicalKeyboardKey.slash):
              const _ShowHelpIntent(),
        },
        child: Actions(
          actions: <Type, Action<Intent>>{
            _ShowHelpIntent: CallbackAction<_ShowHelpIntent>(
              onInvoke: (_) {
                showHelpDialog(context);
                return null;
              },
            ),
          },
          // F-339: wrap the whole shell in a SelectionArea so the operator
          // can select + copy ANY text (balances, account IDs, errors,
          // log lines, config values). Flutter Text is non-selectable by
          // default — without this, nothing in the app could be copied.
          // The chart opens as a separate route, so its drag-to-pan isn't
          // affected by this selection layer.
          child: Focus(
            autofocus: true,
            child: SelectionArea(
              child: _ShellGrid(activeId: activeId),
            ),
          ),
        ),
      ),
    );
  }
}

class _ShellGrid extends StatelessWidget {
  final String activeId;
  const _ShellGrid({required this.activeId});

  /// Tabs that get the AI Desk right-rail (F-322). Per the Codex mockup
  /// the rail rides along on the trading-focused screens — Market
  /// Watch, Strategy Lab, Positions — but not on the dedicated AI Desk
  /// tab (that tab IS the full version) or on Settings/Dashboard.
  static const _railTabs = {'MarketWatch', 'StrategyLab', 'Positions'};

  @override
  Widget build(BuildContext context) {
    final showRail = _railTabs.contains(activeId);
    return Column(
      children: [
        const TopBar(),
        // Backend connectivity banner sits BELOW the TopBar so the
        // brand + LIVE/OFFLINE badges remain anchored, and ABOVE
        // the sidebar+dock split so it spans the full window width
        // (the user can't miss it). Zero height in the healthy
        // steady state — no layout penalty.
        const BackendHealthBanner(),
        Expanded(
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Sidebar(),
              Expanded(
                child: Container(
                  color: ForexAiTokens.appBg,
                  padding: const EdgeInsets.all(ForexAiTokens.spSm),
                  child: _DockArea(activeId: activeId),
                ),
              ),
              // F-322 AI Desk right-rail. Renders only on the trading
              // screens; the rail itself decides whether to render as
              // a 280 px full panel or a 36 px collapsed strip.
              if (showRail) const AiDeskRail(),
            ],
          ),
        ),
        const StatusBar(),
      ],
    );
  }
}

class _DockArea extends StatelessWidget {
  final String activeId;
  const _DockArea({required this.activeId});

  @override
  Widget build(BuildContext context) {
    final tab = navTabById(activeId);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Banner for LLM-proposed actions awaiting Confirm/Reject
        // (#136 Phase B). Renders as SizedBox.shrink when the
        // queue is empty, so non-LLM users see no UI difference.
        const PendingActionsBanner(),
        // Single-line header. The old breadcrumb (Group › Tab) is gone
        // because there are no groups any more; the active tab in the
        // sidebar already tells the user where they are. We keep a
        // small description line for context — particularly useful on
        // the consolidated screens where the title alone ("Strategy
        // Lab") doesn't say what's inside.
        Padding(
          padding: const EdgeInsets.only(bottom: 4),
          child: Row(
            children: [
              Text(
                tab.title,
                style: const TextStyle(
                  fontSize: ForexAiTokens.fsSubtitle,
                  fontWeight: FontWeight.w700,
                  color: ForexAiTokens.textPrimary,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  tab.description,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: ForexAiTokens.fsCaption,
                    color: ForexAiTokens.textMuted,
                  ),
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: Container(
            decoration: BoxDecoration(
              color: ForexAiTokens.panelBg,
              border: Border.all(color: ForexAiTokens.border),
              borderRadius: BorderRadius.circular(ForexAiTokens.rMd),
            ),
            padding: const EdgeInsets.all(ForexAiTokens.spLg),
            child: _viewForId(activeId),
          ),
        ),
      ],
    );
  }

  Widget _viewForId(String id) {
    switch (id) {
      case 'Dashboard':
        // DashboardScreen owns its own SingleChildScrollView (F-328) so
        // the test harness — which doesn't wrap screens in the shell —
        // doesn't overflow on the 6-card grid. The shell-level wrapper
        // is gone to avoid nesting two scrollviews.
        return const DashboardScreen();
      case 'MarketWatch':
        // Market Watch composes Markets + Chart + Execution + News as
        // internal tabs (transitional until F-325 lands the unified
        // multi-symbol table). The composer manages its own scroll
        // surfaces per sub-tab, so no outer SingleChildScrollView.
        return const MarketWatchScreen();
      case 'StrategyLab':
        return const StrategyLabScreen();
      case 'Positions':
        return const PositionsScreen();
      case 'AiDesk':
        return const AiDeskScreen();
      case 'Settings':
        // Settings is a TabBar wrapper (F-327) — it needs unbounded
        // height for its TabBarView, so no outer SingleChildScrollView.
        return const SettingsScreen();
      default:
        return const SingleChildScrollView(child: DashboardScreen());
    }
  }
}
