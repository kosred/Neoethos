import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'account_provider.dart' show backendClientProvider;

/// Holds the active UI [Locale]. Supported: `en` (default) and `el` (Greek).
///
/// Persistence (Stage 1b, 2026-06-03) lives in the Rust backend config
/// (`system.ui_locale`, surfaced via `/settings`) — the app's single source of
/// truth — rather than a separate Flutter store. On construction we fetch the
/// persisted locale (the backend supervisor has already brought the server up
/// by the time the UI builds); the Settings picker writes it back through
/// `saveSettings(uiLocale: ...)`.
class LocaleNotifier extends StateNotifier<Locale> {
  LocaleNotifier(this._ref) : super(const Locale('en')) {
    _loadFromConfig();
  }

  /// Test-only seam: build the notifier WITHOUT the startup `/settings` fetch
  /// the default constructor performs. Widget tests pump the whole app, and
  /// that fetch otherwise leaves a Dio request timer pending past teardown
  /// (tripping Flutter's "A Timer is still pending…" invariant). Production
  /// always uses the default constructor — this path never runs outside tests.
  @visibleForTesting
  LocaleNotifier.noFetch(this._ref) : super(const Locale('en'));

  final Ref _ref;

  /// All language codes the UI ships translations for. Order is the display
  /// order in the Settings picker.
  static const supportedCodes = <String>['en', 'el'];

  /// Pull the persisted locale from `system.ui_locale` (via `/settings`) once
  /// at startup. Failures are swallowed — the `'en'` default still renders and
  /// the picker keeps working; the next save persists the operator's choice.
  Future<void> _loadFromConfig() async {
    try {
      final s = await _ref.read(backendClientProvider).fetchSettings();
      if (supportedCodes.contains(s.localeCode)) {
        state = Locale(s.localeCode);
      }
    } catch (_) {
      // Backend not reachable yet — keep the default locale.
    }
  }

  /// Switch the active locale. [code] is an ISO-639-1 language code
  /// (`'en'` or `'el'`); unknown codes are ignored so a stale persisted value
  /// can never wedge the UI into an unsupported locale. This only updates the
  /// in-memory state; the Settings picker is responsible for persisting the
  /// choice to the backend config.
  void setLanguage(String code) {
    if (supportedCodes.contains(code)) {
      state = Locale(code);
    }
  }
}

/// The single source of truth for the active UI locale. `MaterialApp.locale`
/// in `main.dart` watches this; the Settings language picker writes it (and
/// persists it to the backend config via `saveSettings`).
final localeProvider = StateNotifierProvider<LocaleNotifier, Locale>(
  (ref) => LocaleNotifier(ref),
);
