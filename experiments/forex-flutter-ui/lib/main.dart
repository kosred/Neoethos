// forex-ai Flutter front-end entry point.
//
// Multi-platform desktop (Windows/macOS/Linux) + mobile target.
// Pure thin client over the Rust backend — no business logic
// lives in Dart. Layout + design tokens mirror
// mockups/ui_mockup.html.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'theme/theme.dart';
import 'widgets/app_shell.dart';

void main() {
  runApp(const ProviderScope(child: ForexAiApp()));
}

class ForexAiApp extends StatelessWidget {
  const ForexAiApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'forex-ai',
      debugShowCheckedModeBanner: false,
      theme: buildForexAiTheme(),
      home: const AppShell(),
    );
  }
}
