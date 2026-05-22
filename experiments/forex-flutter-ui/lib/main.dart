// NeoEthos Flutter front-end entry point.
//
// Multi-platform desktop (Windows/macOS/Linux) + mobile target.
// Pure thin client over the Rust backend — no business logic
// lives in Dart. Layout + design tokens mirror
// mockups/ui_mockup.html.
//
// Startup sequence:
//   1. Initialise Flutter binding.
//   2. Ensure the Rust backend (`neoethos-app --server`) is running
//      on 127.0.0.1:7423. If it isn't, spawn it as a child process.
//      This gives the operator the double-click experience: launch
//      one .exe, both halves come up.
//   3. Run the Flutter UI.

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'startup/backend_supervisor.dart';
import 'theme/theme.dart';
import 'widgets/app_shell.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Fire-and-forget: spawn the backend if it isn't already up. The
  // supervisor is non-blocking — UI renders immediately and the
  // existing AsyncValue.error states cover the (very short) window
  // where the server hasn't bound the port yet.
  await BackendSupervisor.instance.ensureRunning();
  runApp(const ProviderScope(child: NeoEthosApp()));
}

class NeoEthosApp extends StatelessWidget {
  const NeoEthosApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NeoEthos',
      debugShowCheckedModeBanner: false,
      theme: buildForexAiTheme(),
      home: const AppShell(),
    );
  }
}
