// neoethos Flutter front-end entry point.
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

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'startup/backend_supervisor.dart';
import 'theme/theme.dart';
import 'widgets/app_shell.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // #176: spawn the backend if it isn't already up. The supervisor
  // returns FALSE when another NeoEthos instance is already alive
  // (an existing healthy /healthz response). We exit immediately in
  // that case to avoid two competing UI windows — the existing
  // window stays in focus, the duplicate shell vanishes.
  final shouldContinue = await BackendSupervisor.instance.ensureRunning();
  if (!shouldContinue) {
    // The existing instance owns the UI; exiting with code 0 keeps
    // the OS from logging a crash and stops the duplicate cold.
    exit(0);
  }
  runApp(const ProviderScope(child: NeoethosApp()));
}

class NeoethosApp extends StatelessWidget {
  const NeoethosApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'neoethos',
      debugShowCheckedModeBanner: false,
      theme: buildForexAiTheme(),
      home: const AppShell(),
    );
  }
}
