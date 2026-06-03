// Server-error translation helper. The Rust backend's
// `crate::app_services::ctrader_errors::translate_anyhow` injects a
// `translation` object into every 502 BAD_GATEWAY body when it can
// extract a recognised cTrader error code. This file is the Flutter
// counterpart — pull the structured payload out of a DioException
// and turn it into a friendly snackbar with optional action button.
//
// Why structured: the bare `error` string still contains the raw
// upstream wording (`cTrader execution rejected: status=Failed
// code=Some("MARKET_CLOSED") description=...`). End users have no
// idea what that means. The `translation.message` field is plain
// English; `translation.actionLabel`/`actionTarget` tells the UI
// what button to render and where it should send the user.

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';

import '../theme/theme.dart';
import '../widgets/report_issue.dart';

/// Mirror of `crate::app_services::ctrader_errors::TranslatedError`.
class TranslatedError {
  /// Raw upstream code, e.g. `CH_ACCESS_TOKEN_INVALID`.
  final String code;

  /// Friendly English copy meant for users.
  final String message;

  /// Optional CTA label, e.g. `"Re-authenticate"`. Null when the
  /// error has no clear next action.
  final String? actionLabel;

  /// Where the CTA should send the user. Conventional values:
  ///   * `broker_setup`  → /broker-setup
  ///   * `settings`      → /settings
  ///   * `risk`          → /risk-settings
  ///   * `data_bootstrap`→ /data-bootstrap
  ///   * `reauth`        → POST /broker/reauth directly (no nav)
  ///   * null            → no CTA
  final String? actionTarget;

  /// `info` / `warning` / `error` / `critical`. Drives banner color.
  final String severity;

  const TranslatedError({
    required this.code,
    required this.message,
    required this.actionLabel,
    required this.actionTarget,
    required this.severity,
  });

  factory TranslatedError.fromJson(Map<String, dynamic> j) => TranslatedError(
        code: (j['code'] as String?) ?? '',
        message: (j['message'] as String?) ?? '',
        actionLabel: j['actionLabel'] as String?,
        actionTarget: j['actionTarget'] as String?,
        severity: (j['severity'] as String?) ?? 'error',
      );

  Color get tone {
    switch (severity) {
      case 'info':
        return NeoethosTokens.textMuted;
      case 'warning':
        return NeoethosTokens.warning;
      case 'critical':
      case 'error':
      default:
        return NeoethosTokens.sell;
    }
  }
}

/// Pull a translation block out of a DioException body, if present.
/// Returns null when (a) it's not a DioException with a structured
/// body, or (b) the body has no `translation` field (i.e. the server
/// couldn't recognise the upstream error code).
TranslatedError? extractTranslation(Object error) {
  if (error is! DioException) return null;
  final data = error.response?.data;
  if (data is! Map) return null;
  final t = data['translation'];
  if (t is! Map) return null;
  return TranslatedError.fromJson(Map<String, dynamic>.from(t));
}

/// One-stop message extractor. Returns the best human-readable
/// description of a failure: translation message first, then the
/// raw `error` field from the body, finally the Dio exception
/// message. Use this whenever a screen needs a single string to
/// display.
String describeError(Object error) {
  final t = extractTranslation(error);
  if (t != null && t.message.isNotEmpty) return t.message;
  if (error is DioException) {
    final data = error.response?.data;
    if (data is Map && data['error'] is String) {
      return data['error'] as String;
    }
    return error.message ?? error.toString();
  }
  return error.toString();
}

/// Show a SnackBar with the translated message, colored by severity.
/// Falls back to a plain `<prefix>: <error>` line when no
/// translation is available. The optional CTA button is rendered
/// purely as a hint — wiring it to actual navigation is deferred to
/// each caller (the app's tab/screen navigation isn't go_router-based
/// at the moment, so a generic navigation helper here would be
/// brittle). Callers that care can read [extractTranslation] and
/// handle `actionTarget` themselves.
///
/// Special case: when the translated severity is `critical` AND no
/// caller-supplied `onAction` is provided, the snackbar's action
/// button switches to a "Report" button that opens the
/// diagnostic-bundle / email-support dialog. End users can't fix
/// catastrophic backend failures themselves, so we ALWAYS give them
/// a one-tap path to get logs into our inbox.
void showTranslatedErrorSnackbar(
  BuildContext context,
  Object error, {
  String prefix = 'Failed',
  VoidCallback? onAction,
}) {
  final translation = extractTranslation(error);
  final body = describeError(error);
  final bg = translation?.tone ?? NeoethosTokens.sell;

  SnackBarAction? action;
  if (translation != null &&
      translation.actionLabel != null &&
      onAction != null) {
    action = SnackBarAction(
      label: translation.actionLabel!,
      textColor: Colors.white,
      onPressed: onAction,
    );
  } else if (translation?.severity == 'critical') {
    // Fallback for catastrophic errors with no caller-handled CTA.
    // Pre-fill the description with the prefix + body so support
    // sees the exact wording the user hit.
    action = SnackBarAction(
      label: 'Report',
      textColor: Colors.white,
      onPressed: () => showReportIssueDialog(
        context,
        prefillDescription: '$prefix: $body',
        category: translation?.code ?? 'critical',
      ),
    );
  }

  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(
      backgroundColor: bg,
      content: Text('$prefix: $body'),
      duration: const Duration(seconds: 6),
      action: action,
    ),
  );
}
