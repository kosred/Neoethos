// The grid shell — TopBar + Sidebar + Dock + StatusBar.
//
// Matches mockups/ui_mockup.html .app layout:
//   grid-template-rows:    var(--topbar-h) 1fr var(--statusbar-h)
//   grid-template-columns: var(--sidebar-w) 1fr
//   grid-template-areas:   "topbar topbar"
//                          "sidebar dock"
//                          "statusbar statusbar"

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../state/nav.dart';
import '../theme/theme.dart';
import 'sidebar.dart';
import 'topbar.dart';
import 'statusbar.dart';
import '../screens/dashboard_screen.dart';
import '../screens/chart_screen.dart';
import '../screens/markets_screen.dart';
import '../screens/execution_screen.dart';
import '../screens/news_screen.dart';
import '../screens/trade_watch_screen.dart';
import '../screens/discovery_screen.dart';
import '../screens/training_screen.dart';
import '../screens/intelligence_screen.dart';
import '../screens/broker_setup_screen.dart';
import '../screens/data_bootstrap_screen.dart';
import '../screens/hardware_screen.dart';
import '../screens/risk_screen.dart';
import '../screens/settings_screen.dart';

class AppShell extends ConsumerWidget {
  const AppShell({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final activeId = ref.watch(activeTabProvider);
    return Scaffold(
      backgroundColor: ForexAiTokens.appBg,
      body: Column(
        children: [
          const TopBar(),
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
              ],
            ),
          ),
          const StatusBar(),
        ],
      ),
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
        // Breadcrumb
        Padding(
          padding: const EdgeInsets.only(bottom: 2),
          child: Row(
            children: [
              Text(
                navGroupLabel(tab.group),
                style: const TextStyle(
                  fontSize: 11,
                  color: ForexAiTokens.textMuted,
                ),
              ),
              const Padding(
                padding: EdgeInsets.symmetric(horizontal: 6),
                child: Text('›',
                    style: TextStyle(color: ForexAiTokens.textFaint)),
              ),
              Text(
                tab.title,
                style: const TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                  color: ForexAiTokens.textPrimary,
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
            child: SingleChildScrollView(child: _viewForId(activeId)),
          ),
        ),
      ],
    );
  }

  Widget _viewForId(String id) {
    switch (id) {
      case 'Dashboard':
        return const DashboardScreen();
      case 'Chart':
        return const ChartScreen();
      case 'Markets':
        return const MarketsScreen();
      case 'Execution':
        return const ExecutionScreen();
      case 'News':
        return const NewsScreen();
      case 'TradeWatch':
        return const TradeWatchScreen();
      case 'Discovery':
        return const DiscoveryScreen();
      case 'Training':
        return const TrainingScreen();
      case 'Intelligence':
        return const IntelligenceScreen();
      case 'BrokerSetup':
        return const BrokerSetupScreen();
      case 'DataBootstrap':
        return const DataBootstrapScreen();
      case 'Hardware':
        return const HardwareScreen();
      case 'Risk':
        return const RiskScreen();
      case 'Settings':
        return const SettingsScreen();
      default:
        return const DashboardScreen();
    }
  }
}
