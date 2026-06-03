import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Holds the active UI [Locale]. Supported: `en` (default) and `el` (Greek).
///
/// Stage 1a (2026-06-03) keeps this in memory — the Settings language picker
/// switches it for the running session. Persistence across restarts is wired
/// in Stage 1b from the Rust backend config (the `/settings` endpoint), which
/// keeps the app's "config is the single source of truth" model rather than
/// introducing `shared_preferences` for a one-off UI preference.
class LocaleNotifier extends StateNotifier<Locale> {
  LocaleNotifier() : super(const Locale('en'));

  /// All language codes the UI ships translations for. Order is the display
  /// order in the Settings picker.
  static const supportedCodes = <String>['en', 'el'];

  /// Switch the active locale. [code] is an ISO-639-1 language code
  /// (`'en'` or `'el'`); unknown codes are ignored so a stale persisted
  /// value can never wedge the UI into an unsupported locale.
  void setLanguage(String code) {
    if (supportedCodes.contains(code)) {
      state = Locale(code);
    }
  }
}

/// The single source of truth for the active UI locale. `MaterialApp.locale`
/// in `main.dart` watches this; the Settings picker writes it.
final localeProvider = StateNotifierProvider<LocaleNotifier, Locale>(
  (ref) => LocaleNotifier(),
);
