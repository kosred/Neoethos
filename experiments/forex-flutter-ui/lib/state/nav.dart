// Navigation state — which panel is currently shown.
//
// 14 panels grouped under Trading / AI Engine / System (mirrors
// the sidebar in mockups/ui_mockup.html).

import 'package:flutter_riverpod/flutter_riverpod.dart';

enum NavGroup { trading, aiEngine, system }

class NavTab {
  final String id;
  final NavGroup group;
  final String icon;
  final String title;
  final String description;
  const NavTab(this.id, this.group, this.icon, this.title, this.description);
}

const List<NavTab> kNavTabs = [
  // Trading
  NavTab('Dashboard', NavGroup.trading, '▦', 'Dashboard',
      'Account equity, open positions, engine status'),
  NavTab('Chart', NavGroup.trading, '📈', 'Chart',
      'TradingView-style price chart with bid/ask'),
  NavTab('Markets', NavGroup.trading, '≡', 'Markets',
      'Symbol list with live quotes'),
  NavTab('Execution', NavGroup.trading, '↹', 'Order Ticket',
      'Place / modify / cancel orders'),
  NavTab('News', NavGroup.trading, '📰', 'News',
      'LLM-curated news + blackout filter'),
  NavTab('TradeWatch', NavGroup.trading, '◫', 'Trade Watch',
      'Compact trade watch strip'),

  // AI Engine
  NavTab('Discovery', NavGroup.aiEngine, '✦', 'Discovery',
      'Genetic strategy search → portfolio'),
  NavTab('Training', NavGroup.aiEngine, '⊛', 'Training',
      'AI ensemble training pipeline'),
  NavTab('Intelligence', NavGroup.aiEngine, '✺', 'Intelligence',
      'AI model insights & explainability'),

  // System
  NavTab('BrokerSetup', NavGroup.system, '🔌', 'Broker Setup',
      'cTrader / DXTrade credentials & OAuth'),
  NavTab('DataBootstrap', NavGroup.system, '⤓', 'Data Bootstrap',
      'Historical data download / migration'),
  NavTab('Hardware', NavGroup.system, '▤', 'Hardware',
      'CPU / GPU / RAM detection & overrides'),
  NavTab('Risk', NavGroup.system, '⚠', 'Risk Settings',
      'Prop-firm risk rules & guard-rails'),
  NavTab('Settings', NavGroup.system, '⚙', 'Settings', 'App-wide settings'),
];

/// Currently selected nav tab id (defaults to Dashboard).
final activeTabProvider = StateProvider<String>((ref) => 'Dashboard');

/// Resolver — useful for screens that need their own tab metadata.
NavTab navTabById(String id) =>
    kNavTabs.firstWhere((t) => t.id == id, orElse: () => kNavTabs.first);

/// Group label helper (the sidebar renders one section per group).
String navGroupLabel(NavGroup g) {
  switch (g) {
    case NavGroup.trading:
      return 'TRADING';
    case NavGroup.aiEngine:
      return 'AI ENGINE';
    case NavGroup.system:
      return 'SYSTEM';
  }
}
