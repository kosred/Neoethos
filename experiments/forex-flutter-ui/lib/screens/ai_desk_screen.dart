// AI Desk — ensemble state + AI assistant (F-321 rebuild).
//
// **Codex mockup vision** (mockups/ig_*.png both images, right side):
// AI Desk is *both* a top-level tab AND a persistent right-rail on
// the Market Watch / Strategy Lab / Positions screens. The tab gives
// the operator the full-screen view (models loaded, predictions,
// proposed actions, risk/physics/compute sub-tabs); the right-rail
// is a condensed mirror for at-a-glance use.
//
// **This file (transitional)**: F-321 surfaces the existing
// Intelligence + AI Helper screens as internal tabs. F-322 will build
// the right-rail widget, and F-330 will surface the unified
// "Proposed Action" card with Review & Confirm.

import 'package:flutter/material.dart';

import 'ai_helper_screen.dart';
import 'intelligence_screen.dart';
import '../theme/theme.dart';

class AiDeskScreen extends StatefulWidget {
  const AiDeskScreen({super.key});

  @override
  State<AiDeskScreen> createState() => _AiDeskScreenState();
}

class _AiDeskScreenState extends State<AiDeskScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _controller;

  static const _sections = [
    ('Intelligence',
        'Ensemble signals, feature importances, model artifacts'),
    ('Assistant', 'ChatGPT-subscription chat assistant with tool calls'),
  ];

  @override
  void initState() {
    super.initState();
    _controller = TabController(length: _sections.length, vsync: this);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _AiDeskSubNav(controller: _controller, sections: _sections),
        const SizedBox(height: NeoethosTokens.spSm),
        Expanded(
          child: TabBarView(
            controller: _controller,
            physics: const NeverScrollableScrollPhysics(),
            children: const [
              IntelligenceScreen(),
              AiHelperScreen(),
            ],
          ),
        ),
      ],
    );
  }
}

class _AiDeskSubNav extends StatelessWidget {
  final TabController controller;
  final List<(String, String)> sections;
  const _AiDeskSubNav({
    required this.controller,
    required this.sections,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: TabBar(
        controller: controller,
        isScrollable: true,
        labelColor: NeoethosTokens.accent,
        unselectedLabelColor: NeoethosTokens.textMuted,
        indicatorColor: NeoethosTokens.accent,
        labelStyle: const TextStyle(
          fontSize: NeoethosTokens.fsBody,
          fontWeight: FontWeight.w700,
        ),
        unselectedLabelStyle: const TextStyle(
          fontSize: NeoethosTokens.fsBody,
          fontWeight: FontWeight.w500,
        ),
        tabs: [
          for (final (label, tooltip) in sections)
            Tooltip(
              message: tooltip,
              waitDuration: const Duration(milliseconds: 600),
              child: Tab(text: label),
            ),
        ],
      ),
    );
  }
}
