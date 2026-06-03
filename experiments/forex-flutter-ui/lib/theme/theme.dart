// Design tokens mirrored from mockups/ui_mockup.html
// (kept locally in Flutter after the legacy Rust UI removal).
//
// Single source of truth for color / typography / spacing /
// dimensions so every widget reads from `NeoethosTheme.of(context)`
// instead of hard-coding hex values.
//
// TradingView-style dark theme. Light theme is intentionally out
// of scope for v0.4 — operators run this at night, dark is the
// default and only mode.

import 'package:flutter/material.dart';

/// All raw token values from the HTML mockup. Mirror the CSS
/// custom-property names so a developer can grep across either
/// codebase by token name.
class NeoethosTokens {
  // Surfaces
  static const Color appBg = Color(0xFF0E1116);
  static const Color panelBg = Color(0xFF161B22);
  static const Color surfaceBg = Color(0xFF1C2230);
  static const Color surfaceAlt = Color(0xFF22293A);
  static const Color chartBg = Color(0xFF0E1116);
  static const Color grid = Color(0xFF1F2430);

  // Borders
  static const Color border = Color(0xFF2A2F3A);
  static const Color borderStrong = Color(0xFF3A404D);

  // Text
  static const Color textPrimary = Color(0xFFE6EAF2);
  static const Color textMuted = Color(0xFF9AA4B2);
  static const Color textFaint = Color(0xFF5C6473);

  // Brand / accent
  static const Color accent = Color(0xFF2962FF);
  static const Color accentHover = Color(0xFF1E53E5);
  static const Color accentMuted = Color(0xFF1E2A4A);
  static const Color accentSoft = Color(0xFF161F36);

  // Buy / sell — TradingView convention (teal-green for buy, red for sell).
  static const Color buy = Color(0xFF26A69A);
  static const Color buyStrong = Color(0xFF00C853);
  static const Color sell = Color(0xFFEF5350);
  static const Color sellStrong = Color(0xFFFF1744);
  static const Color warning = Color(0xFFF4B400);

  // Spacing scale (px)
  static const double spXs = 4;
  static const double spSm = 8;
  static const double spMd = 12;
  static const double spLg = 16;
  static const double spXl = 24;

  // Type scale (px)
  static const double fsCaption = 11;
  static const double fsBody = 13;
  static const double fsSubtitle = 15;
  static const double fsTitle = 20;

  // Radius (px)
  static const double rSm = 4;
  static const double rMd = 6;
  static const double rLg = 8;

  // Layout dimensions (px)
  static const double topbarHeight = 44;
  static const double statusbarHeight = 22;
  static const double sidebarWidth = 220;
  static const double tabStripHeight = 22;
  static const double btnHeight = 32;
  static const double btnHeightSm = 24;
  static const double rowHeight = 24;
}

/// Materialised dark theme that every screen reads through
/// `Theme.of(context)`. Wraps the raw tokens into Flutter's
/// `ThemeData` so widgets can stay token-agnostic when they only
/// need the standard surface / on-surface / primary axes.
ThemeData buildNeoethosTheme() {
  const scheme = ColorScheme(
    brightness: Brightness.dark,
    primary: NeoethosTokens.accent,
    onPrimary: NeoethosTokens.textPrimary,
    secondary: NeoethosTokens.accentHover,
    onSecondary: NeoethosTokens.textPrimary,
    error: NeoethosTokens.sell,
    onError: NeoethosTokens.textPrimary,
    surface: NeoethosTokens.panelBg,
    onSurface: NeoethosTokens.textPrimary,
    surfaceContainerHighest: NeoethosTokens.surfaceAlt,
    outline: NeoethosTokens.border,
    outlineVariant: NeoethosTokens.borderStrong,
  );

  const textTheme = TextTheme(
    // Title row of a panel (e.g. "Operator Overview").
    titleLarge: TextStyle(
      fontSize: NeoethosTokens.fsTitle,
      fontWeight: FontWeight.w700,
      color: NeoethosTokens.textPrimary,
    ),
    titleMedium: TextStyle(
      fontSize: NeoethosTokens.fsSubtitle,
      fontWeight: FontWeight.w700,
      color: NeoethosTokens.textPrimary,
    ),
    bodyMedium: TextStyle(
      fontSize: NeoethosTokens.fsBody,
      color: NeoethosTokens.textPrimary,
    ),
    bodySmall: TextStyle(
      fontSize: NeoethosTokens.fsCaption,
      color: NeoethosTokens.textMuted,
    ),
    labelSmall: TextStyle(
      fontSize: NeoethosTokens.fsCaption - 1,
      letterSpacing: 1.0,
      fontWeight: FontWeight.w700,
      color: NeoethosTokens.textMuted,
    ),
  );

  return ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    scaffoldBackgroundColor: NeoethosTokens.appBg,
    canvasColor: NeoethosTokens.panelBg,
    colorScheme: scheme,
    textTheme: textTheme,
    fontFamily: 'Segoe UI', // closest Windows match to the mockup's stack
    dividerColor: NeoethosTokens.border,
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: NeoethosTokens.accent,
        foregroundColor: NeoethosTokens.textPrimary,
        textStyle: const TextStyle(
          fontSize: NeoethosTokens.fsBody,
          fontWeight: FontWeight.w700,
        ),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
        ),
        minimumSize: const Size(0, NeoethosTokens.btnHeight),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: NeoethosTokens.surfaceBg,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
        borderSide: const BorderSide(color: NeoethosTokens.border),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
        borderSide: const BorderSide(color: NeoethosTokens.border),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
        borderSide: const BorderSide(color: NeoethosTokens.accent),
      ),
      contentPadding: const EdgeInsets.symmetric(
        horizontal: NeoethosTokens.spMd,
        vertical: NeoethosTokens.spSm,
      ),
      labelStyle: const TextStyle(
        color: NeoethosTokens.textMuted,
        fontSize: NeoethosTokens.fsBody,
      ),
    ),
    cardTheme: CardThemeData(
      color: NeoethosTokens.surfaceBg,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(NeoethosTokens.rMd),
        side: const BorderSide(color: NeoethosTokens.border),
      ),
    ),
  );
}
