// Help screen — F1 globally, also reachable from the TopBar help icon
// and from the Settings → Help sub-tab.
//
// **F-329 (2026-05-29 rebuild)**: until now NeoEthos had no real
// in-app help — only short inline accordions inside Advanced
// Settings. The user explicitly called this out as the #1 frustration
// ("Info είναι άχρηστο, no real Help section"). This screen lifts the
// rich content authored in `mockups/ui_mockup.html` (lines 4530-4670)
// into 6 native Flutter sections with a Greek ⇄ English toggle.
//
// Sections:
//   1. Welcome              — recommended new-user 7-step flow
//   2. Trading              — order ticket, indicators, drawing tools
//   3. AI Engine            — Discovery + Training explained
//   4. Risk                 — Standard / Prop firm / 20-pip modes
//   5. Keyboard shortcuts   — Ctrl+K, F1, right-click menus
//   6. FAQ                  — common questions

import 'package:flutter/material.dart';

import '../l10n/app_localizations.dart';
import '../theme/theme.dart';

/// Public entrypoint — opens the Help screen as a full-screen dialog,
/// usable from any screen / keyboard shortcut / topbar button.
Future<void> showHelpDialog(BuildContext context, {String section = 'welcome'}) {
  return Navigator.of(context).push(
    MaterialPageRoute(
      fullscreenDialog: true,
      builder: (_) => HelpScreen(initialSection: section),
    ),
  );
}

class HelpScreen extends StatefulWidget {
  final String initialSection;
  const HelpScreen({super.key, this.initialSection = 'welcome'});

  @override
  State<HelpScreen> createState() => _HelpScreenState();
}

class _HelpScreenState extends State<HelpScreen> {
  late String _lang;
  late String _section;

  @override
  void initState() {
    super.initState();
    _lang = 'el'; // Default to Greek since the operator is Greek-speaking.
    _section = widget.initialSection;
  }

  Map<String, _HelpSection> _docFor(BuildContext context) =>
      _lang == 'el' ? _helpContentEl : _helpContentEn(context);

  @override
  Widget build(BuildContext context) {
    final sections = _docFor(context);
    final active = sections[_section] ?? sections.values.first;
    return Scaffold(
      backgroundColor: NeoethosTokens.appBg,
      appBar: AppBar(
        backgroundColor: NeoethosTokens.panelBg,
        elevation: 0,
        title: Row(
          children: [
            Text(
              AppLocalizations.of(context)!.helpTitle,
              style: const TextStyle(
                color: NeoethosTokens.textPrimary,
                fontSize: NeoethosTokens.fsSubtitle,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(width: NeoethosTokens.spLg),
            _LangToggle(
              lang: _lang,
              onChanged: (v) => setState(() => _lang = v),
            ),
          ],
        ),
        iconTheme: const IconThemeData(color: NeoethosTokens.textPrimary),
      ),
      body: Row(
        children: [
          _SectionRail(
            sections: sections,
            active: _section,
            onTap: (key) => setState(() => _section = key),
          ),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(
                horizontal: NeoethosTokens.spLg + 8,
                vertical: NeoethosTokens.spLg,
              ),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 760),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      active.title,
                      style: const TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.w700,
                        color: NeoethosTokens.textPrimary,
                      ),
                    ),
                    const SizedBox(height: NeoethosTokens.spMd),
                    for (final block in active.blocks) ...[
                      block.build(context),
                      const SizedBox(height: NeoethosTokens.spSm),
                    ],
                    const SizedBox(height: NeoethosTokens.spXl),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Language toggle
// ---------------------------------------------------------------------------

class _LangToggle extends StatelessWidget {
  final String lang;
  final ValueChanged<String> onChanged;
  const _LangToggle({required this.lang, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        border: Border.all(color: NeoethosTokens.border),
        borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _LangChip(label: 'Ελληνικά', code: 'el', active: lang == 'el',
              onTap: () => onChanged('el')),
          _LangChip(label: 'English', code: 'en', active: lang == 'en',
              onTap: () => onChanged('en')),
        ],
      ),
    );
  }
}

class _LangChip extends StatelessWidget {
  final String label;
  final String code;
  final bool active;
  final VoidCallback onTap;
  const _LangChip({
    required this.label,
    required this.code,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: active ? NeoethosTokens.accentMuted : Colors.transparent,
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: NeoethosTokens.fsCaption,
            fontWeight: active ? FontWeight.w700 : FontWeight.w500,
            color: active ? NeoethosTokens.accent : NeoethosTokens.textMuted,
          ),
        ),
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Left section rail
// ---------------------------------------------------------------------------

class _SectionRail extends StatelessWidget {
  final Map<String, _HelpSection> sections;
  final String active;
  final ValueChanged<String> onTap;
  const _SectionRail({
    required this.sections,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 220,
      decoration: const BoxDecoration(
        color: NeoethosTokens.panelBg,
        border: Border(right: BorderSide(color: NeoethosTokens.border)),
      ),
      padding: const EdgeInsets.symmetric(
        vertical: NeoethosTokens.spMd,
        horizontal: NeoethosTokens.spSm,
      ),
      child: ListView(
        children: [
          for (final entry in sections.entries)
            InkWell(
              onTap: () => onTap(entry.key),
              child: Container(
                margin: const EdgeInsets.symmetric(vertical: 2),
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                decoration: BoxDecoration(
                  color: active == entry.key
                      ? NeoethosTokens.accentMuted
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
                  border: active == entry.key
                      ? const Border(
                          left: BorderSide(
                            color: NeoethosTokens.accent,
                            width: 3,
                          ),
                        )
                      : null,
                ),
                child: Text(
                  entry.value.title,
                  style: TextStyle(
                    fontSize: NeoethosTokens.fsBody,
                    fontWeight: active == entry.key
                        ? FontWeight.w700
                        : FontWeight.w500,
                    color: active == entry.key
                        ? NeoethosTokens.textPrimary
                        : NeoethosTokens.textMuted,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// ---------------------------------------------------------------------------
// Content data classes
// ---------------------------------------------------------------------------

class _HelpSection {
  final String title;
  final List<_HelpBlock> blocks;
  const _HelpSection({required this.title, required this.blocks});
}

abstract class _HelpBlock {
  Widget build(BuildContext context);
}

class _H3 implements _HelpBlock {
  final String text;
  const _H3(this.text);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 16, bottom: 6),
        child: Text(
          text,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            color: NeoethosTokens.textPrimary,
          ),
        ),
      );
}

class _H4 implements _HelpBlock {
  final String text;
  const _H4(this.text);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 10, bottom: 4),
        child: Text(
          text,
          style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w700,
            color: NeoethosTokens.textPrimary,
          ),
        ),
      );
}

class _P implements _HelpBlock {
  /// Spans of (text, isBold). Plain paragraph if all isBold = false.
  final List<(String, bool)> spans;
  const _P(this.spans);
  // Convenience: a plain paragraph with a single non-bold run.
  // Kept const-friendly by inlining the list at the call site as
  // `const _P([(text, false)])` rather than via a factory (factories
  // can't be const in Dart 3 unless they redirect to another const
  // constructor with the same shape, which doesn't fit here).
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Text.rich(
          TextSpan(
            children: [
              for (final (text, bold) in spans)
                TextSpan(
                  text: text,
                  style: TextStyle(
                    fontSize: 13.5,
                    height: 1.55,
                    fontWeight: bold ? FontWeight.w700 : FontWeight.w400,
                    color: NeoethosTokens.textPrimary,
                  ),
                ),
            ],
          ),
        ),
      );
}

class _UL implements _HelpBlock {
  /// Items as spans (same shape as _P).
  final List<List<(String, bool)>> items;
  const _UL(this.items);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            for (final spans in items)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 3),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Padding(
                      padding: EdgeInsets.only(top: 4, right: 8),
                      child: Icon(
                        Icons.circle,
                        size: 6,
                        color: NeoethosTokens.accent,
                      ),
                    ),
                    Expanded(
                      child: Text.rich(
                        TextSpan(
                          children: [
                            for (final (text, bold) in spans)
                              TextSpan(
                                text: text,
                                style: TextStyle(
                                  fontSize: 13.5,
                                  height: 1.55,
                                  fontWeight: bold
                                      ? FontWeight.w700
                                      : FontWeight.w400,
                                  color: NeoethosTokens.textPrimary,
                                ),
                              ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      );
}

class _OL implements _HelpBlock {
  /// Numbered list with the same span shape as _UL.
  final List<List<(String, bool)>> items;
  const _OL(this.items);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            for (var i = 0; i < items.length; i++)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 3),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    SizedBox(
                      width: 24,
                      child: Padding(
                        padding: const EdgeInsets.only(top: 1),
                        child: Text(
                          '${i + 1}.',
                          style: const TextStyle(
                            fontSize: 13.5,
                            fontWeight: FontWeight.w700,
                            color: NeoethosTokens.accent,
                          ),
                        ),
                      ),
                    ),
                    Expanded(
                      child: Text.rich(
                        TextSpan(
                          children: [
                            for (final (text, bold) in items[i])
                              TextSpan(
                                text: text,
                                style: TextStyle(
                                  fontSize: 13.5,
                                  height: 1.55,
                                  fontWeight: bold
                                      ? FontWeight.w700
                                      : FontWeight.w400,
                                  color: NeoethosTokens.textPrimary,
                                ),
                              ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      );
}

class _Tip implements _HelpBlock {
  final String text;
  const _Tip(this.text);
  @override
  Widget build(BuildContext context) => Container(
        margin: const EdgeInsets.symmetric(vertical: 10),
        padding: const EdgeInsets.all(NeoethosTokens.spMd),
        decoration: BoxDecoration(
          color: NeoethosTokens.warning.withValues(alpha: 0.12),
          border: Border.all(
            color: NeoethosTokens.warning.withValues(alpha: 0.55),
          ),
          borderRadius: BorderRadius.circular(NeoethosTokens.rSm),
        ),
        child: Text(
          text,
          style: const TextStyle(
            fontSize: 13.5,
            height: 1.5,
            color: NeoethosTokens.warning,
            fontWeight: FontWeight.w600,
          ),
        ),
      );
}

class _KbdRow implements _HelpBlock {
  final List<String> keys;
  final String description;
  const _KbdRow(this.keys, this.description);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 6),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: 160,
              child: Wrap(
                spacing: 4,
                children: [
                  for (var i = 0; i < keys.length; i++) ...[
                    _KbdChip(keys[i]),
                    if (i < keys.length - 1)
                      const Padding(
                        padding: EdgeInsets.symmetric(horizontal: 2),
                        child: Text('+',
                            style: TextStyle(
                              color: NeoethosTokens.textFaint,
                              fontWeight: FontWeight.w600,
                            )),
                      ),
                  ],
                ],
              ),
            ),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.only(top: 2, left: 8),
                child: Text(
                  description,
                  style: const TextStyle(
                    fontSize: 13.5,
                    height: 1.5,
                    color: NeoethosTokens.textMuted,
                  ),
                ),
              ),
            ),
          ],
        ),
      );
}

class _KbdChip extends StatelessWidget {
  final String label;
  const _KbdChip(this.label);
  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
        decoration: BoxDecoration(
          color: NeoethosTokens.appBg,
          border: Border.all(color: NeoethosTokens.border),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Text(
          label,
          style: const TextStyle(
            fontFamily: 'monospace',
            fontSize: 12,
            fontWeight: FontWeight.w700,
            color: NeoethosTokens.textPrimary,
          ),
        ),
      );
}

// ---------------------------------------------------------------------------
// Greek content (canonical — extended from mockup with NeoEthos specifics)
// ---------------------------------------------------------------------------

final Map<String, _HelpSection> _helpContentEl = {
  'welcome': const _HelpSection(
    title: 'Καλωσήρθες',
    blocks: [
      _H3('NeoEthos — Σύντομη περιγραφή'),
      _P([
        ('Το NeoEthos είναι ένα ', false),
        ('Rust trading terminal', true),
        (' με ενσωματωμένο σύστημα τεχνητής νοημοσύνης. Συνδέεται με broker (cTrader, σύντομα DXtrade), προσφέρει χειροκίνητες συναλλαγές με indicators και drawing tools, και έχει δύο AI engines: ', false),
        ('Discovery', true),
        (' (γενετική αναζήτηση στρατηγικών) και ', false),
        ('Training', true),
        (' (ensemble ML training). Όλα τα δεδομένα έρχονται από τον broker σου — τίποτα δεν είναι hardcoded.', false),
      ]),
      _H4('Προτεινόμενη ροή για νέους χρήστες'),
      _OL([
        [('Άνοιξε λογαριασμό Demo', true), (' σε έναν cTrader broker (Pepperstone, IC Markets, FxPro, Vantage, Axi)', false)],
        [('Σύνδεσε τον broker', true), (' από το Settings → Account → "Connect with cTrader"', false)],
        [('Διάλεξε pairs', true), (' στο Market Watch — ξεκίνα με EUR/USD ή GBP/USD', false)],
        [('Δοκίμασε χειροκίνητη συναλλαγή', true), (' με 0.01 lots για να βεβαιωθείς ότι όλα δουλεύουν', false)],
        [('Τρέξε Strategy Lab → Discovery', true), (' για να βρεις στρατηγικές (10-30 λεπτά)', false)],
        [('Συνέχισε στο Training', true), (' πάνω στις στρατηγικές που βρέθηκαν', false)],
        [('Επικύρωσε για 2 εβδομάδες', true), (' σε demo πριν σκεφτείς live', false)],
      ]),
      _Tip('Σπουδαία υπενθύμιση: 75-90% των retail FX traders χάνουν χρήματα. Πάντα δοκίμασε για εβδομάδες σε Demo πριν αυξήσεις το ρίσκο.'),
    ],
  ),
  'trading': const _HelpSection(
    title: 'Trading',
    blocks: [
      _H3('Χειροκίνητο Trading'),
      _P([
        ('Στο ', false),
        ('Market Watch → Order Ticket', true),
        (' εμφανίζονται BUY (πράσινο) και SELL (κόκκινο) κουμπιά με τις τρέχουσες τιμές bid/ask. Πριν στείλεις την εντολή:', false),
      ]),
      _UL([
        [('Order type: ', true), ('Market (άμεση), Limit (περιμένει χαμηλότερη τιμή για buy), Stop (περιμένει υψηλότερη)', false)],
        [('Lots: ', true), ('το μέγεθος της θέσης. 0.01 lot = 1.000 μονάδες (micro lot). Για EUR/USD ένα pip ≈ \$0.10 ανά 0.01 lot', false)],
        [('Auto Lot: ', true), ('αυτόματος υπολογισμός lot από το % ρίσκο που ορίζεις στο Settings → Risk', false)],
        [('SL/TP (ATR): ', true), ('Stop Loss και Take Profit βασισμένα σε volatility (ATR)', false)],
        [('R:R: ', true), ('αναλογία Risk/Reward — 1:2 σημαίνει για κάθε \$1 ρίσκο στοχεύεις \$2', false)],
      ]),
      _H4('Indicators (Chart)'),
      _P([
        ('Στο γράφημα έχεις ', false),
        ('overlays', true),
        (' (MA, EMA, Bollinger Bands, VWAP, ATR) και ', false),
        ('oscillators', true),
        (' (RSI, MACD, Stochastic). Όλοι υπολογίζονται από το vector_ta crate — δεν υπάρχει manual reimplementation.', false),
      ]),
    ],
  ),
  'chart': const _HelpSection(
    title: 'Γράφημα',
    blocks: [
      _H3('Ζωντανά δεδομένα — όχι αποθηκευμένα'),
      _P([
        ('Το γράφημα δείχνει ', false),
        ('ζωντανά κεριά κατευθείαν από τον broker', true),
        (' (cTrader trendbars), όχι κάποιο παλιό snapshot από τον δίσκο. Το τελευταίο κερί κινείται σε πραγματικό χρόνο από το spot stream.', false),
      ]),
      _H4('Scroll-back — δες χρόνια πίσω (όπως στο TradingView)'),
      _P([
        ('Σύρε το γράφημα ', false),
        ('αριστερά', true),
        (' πέρα από το παλιότερο φορτωμένο κερί και το NeoEthos κατεβάζει αυτόματα την επόμενη σελίδα παλιότερων κεριών ζωντανά από τον broker. ', false),
        ('Μένουν μόνο στη μνήμη — δεν γεμίζουν τον δίσκο.', true),
        (' Μπορείς να γυρίσεις 2 χρόνια πίσω χωρίς κανένα byte στον δίσκο.', false),
      ]),
      _Tip('Για πολύ μακρινό scroll-back σε M1 (εκατομμύρια κεριά), γύρνα σε μεγαλύτερο timeframe (H1/H4/D1) — όπως κάνει και κάθε professional terminal.'),
      _H4('Χειρισμός'),
      _UL([
        [('Zoom: ', true), ('scroll / pinch πάνω στο γράφημα', false)],
        [('Crosshair OHLC: ', true), ('long-press (κράτα πατημένο) για να δεις open/high/low/close σε κάθε κερί', false)],
        [('Overlays / sub-panels: ', true), ('κουμπιά MA · BOLL (πάνω) και MACD · KDJ · RSI · WR (κάτω) — υπολογίζονται live στα broker δεδομένα', false)],
        [('Account switcher: ', true), ('αλλαγή λογαριασμού (DEMO/LIVE) από το dropdown στο top bar', false)],
      ]),
    ],
  ),
  'ai': const _HelpSection(
    title: 'AI Engine',
    blocks: [
      _H3('Strategy Lab — Πώς δουλεύει το pipeline'),
      _P([
        ('Το Strategy Lab είναι ', false),
        ('μία ενιαία ροή 5 σταδίων', true),
        (': Data Ready → Discovery → Training → Validation → Promotion Gate. Κάθε στάδιο έχει δικές του παραμέτρους και output, και ο επόμενος καταναλώνει αυτόματα τον προηγούμενο.', false),
      ]),
      _H4('Discovery — Γενετική αναζήτηση στρατηγικών'),
      _P([('Δημιουργεί τυχαίες στρατηγικές με συνδυασμούς indicators, τις αξιολογεί σε ιστορικά δεδομένα, διατηρεί τις καλύτερες και τις διασταυρώνει σε νέες γενιές.', false)]),
      _UL([
        [('Population: ', true), ('πόσες στρατηγικές σε κάθε γενιά (50-200)', false)],
        [('Generations: ', true), ('πόσες γενιές αναζήτησης (10-50)', false)],
        [('Max indicators: ', true), ('μέγιστο μήκος στρατηγικής (5-15)', false)],
        [('Target candidates: ', true), ('πόσες κερδοφόρες στρατηγικές θέλεις να μαζέψεις', false)],
        [('Correlation threshold: ', true), ('όσο χαμηλότερο, τόσο πιο διαφορετικές οι στρατηγικές μεταξύ τους', false)],
      ]),
      _H4('Training — Multi-symbol ML ensemble'),
      _P([
        ('Παίρνει τα features (indicators, OHLCV, volatility, session, news) και εκπαιδεύει 30 μοντέλα ταυτόχρονα — Tree (LightGBM, XGBoost, CatBoost), Deep classifiers (MLP, KAN, TabNet), Time-series (NBeats, TiDE, PatchTST, TimesNet, Transformer), Meta (logistic, calibration, stacker, HMM regime), RL+Exit (DQN, ExitAgent). ', false),
        ('Genetic/NeuroEvo/NEAT', true),
        (' είναι αποκλειστικά στο Discovery — όχι ψηφοφόροι του ensemble.', false),
      ]),
      _UL([
        [('Walk-forward validation: ', true), ('εκπαιδεύει σε ένα παράθυρο, ελέγχει στο επόμενο, μετακινείται μπροστά', false)],
        [('Soft voting ensemble: ', true), ('30 μοντέλα ψηφίζουν με βάρη — προβλέπει buy/sell/neutral', false)],
        [('Cross-pair features: ', true), ('το μοντέλο χρησιμοποιεί άλλα pairs σαν context', false)],
      ]),
      _H4('Validation + Promotion Gate'),
      _P([('Πριν κάποιο trained ensemble φτάσει σε live, περνάει από Walk-Forward Analysis, Monte Carlo sensitivity sweeps, και ένα Promotion Gate που ελέγχει Sharpe, Calmar, win rate και drawdown. Αν δεν περάσει, δεν προωθείται στον φάκελο live models.', false)]),
    ],
  ),
  'risk': const _HelpSection(
    title: 'Risk',
    blocks: [
      _H3('Διαχείριση Ρίσκου — 3 modes'),
      _UL([
        [('🏦 Standard account: ', true), ('για κανονικούς λογαριασμούς. Ρίσκο 0.5-2% ανά συναλλαγή, daily drawdown ≤ 5%.', false)],
        [('🏆 Prop firm challenge: ', true), ('για challenges (FTMO, FundedNext κλπ.). Σκληρά όρια daily/total drawdown με preset για το challenge σου.', false)],
        [('⚡ Risky Mode (account multiplication): ', true), ('aggressive compounding από μικρό αρχικό κεφάλαιο (π.χ. \$20 → \$50.000). Έχει calculator που εκτιμά time-to-target percentiles (p10/p50/p90) και probability-of-ruin με Brownian Barrier model.', false)],
      ]),
      _H4('Πώς λειτουργεί το Risky Mode'),
      _P([
        ('Το Risky Mode ', false),
        ('ΔΕΝ', true),
        (' είναι "20 pip challenge" hardcoded. Είναι Kelly-aligned compounding με 11 stages από \$20 → \$50K, λογαριθμικό taper, time-to-target ETA με Brownian Barrier inversion και Beasley-Springer-Moro inverse-normal-CDF. Στόχος: να μεγαλώσει τον λογαριασμό όσο γίνεται πιο γρήγορα.', false),
      ]),
      _UL([
        [('Kill-Switch: ', true), ('3 tiers (Soft/Hard/Catastrophic) — αυτόματο block manual orders + 24h cooldown', false)],
        [('Stage budget: ', true), ('κάθε stage έχει όριο R per trade — όχι πάνω από αυτό, ποτέ', false)],
        [('Time-to-target: ', true), ('εμφανίζει p10/p50/p90 σενάρια — βλέπεις τη χειρότερη και την καλύτερη εκδοχή', false)],
      ]),
      _Tip('⚠ 75-90% των retail FX traders χάνουν χρήματα. Risky Mode = aggressive ≠ ασφαλές. Πάντα δοκίμασε για εβδομάδες σε Demo.'),
    ],
  ),
  'data': const _HelpSection(
    title: 'Δεδομένα & Δίσκος',
    blocks: [
      _H3('Από πού έρχονται τα δεδομένα'),
      _P([
        ('Υπάρχουν ', false),
        ('δύο ξεχωριστές πηγές', true),
        (' και είναι σημαντικό να ξέρεις τη διαφορά:', false),
      ]),
      _UL([
        [('Ζωντανά (broker): ', true), ('Γράφημα, Market Watch, indicators, scroll-back. Έρχονται κατευθείαν από τον broker και ', false), ('ΔΕΝ αποθηκεύονται στον δίσκο.', true)],
        [('Τοπικός Vortex cache: ', true), ('Μόνο το Discovery και το Training χρειάζονται αποθηκευμένο ιστορικό για backtest/εκπαίδευση — ', false), ('αυτό', true), (' γράφεται στον δίσκο.', false)],
      ]),
      _H4('Τι γράφει στον δίσκο'),
      _UL([
        [('Data Bootstrap', true), (' (Settings → Data): ρητό κατέβασμα ιστορικού παραθύρου για ένα symbol/timeframe.', false)],
        [('Discovery auto-fetch: ', true), ('αν δεν υπάρχει αρκετό ιστορικό όταν τρέχεις discovery, το κατεβάζει αυτόματα (~N χρόνια, ελέγχεται από το NEOETHOS_BOT_MIN_HISTORY_YEARS) και το αποθηκεύει.', false)],
        [('data_dir: ', true), ('ο φάκελος αποθήκευσης — ορίζεται στο Settings → Data.', false)],
      ]),
      _Tip('Το να βλέπεις γράφημα — ακόμα και να γυρνάς χρόνια πίσω — ΔΕΝ γεμίζει τον δίσκο. Μόνο το Data Bootstrap και το Discovery αποθηκεύουν ιστορικά, επειδή το backtest/training τα χρειάζεται.'),
      _H4('Υποστηριζόμενα symbols'),
      _P([
        ('Η λίστα symbols έρχεται από τον broker, αλλά περιορίζεται σε ', false),
        ('forex, μέταλλα, δείκτες και εμπορεύματα', true),
        (' — οι μετοχές και τα ETF του broker αποκλείονται αυτόματα (η μηχανή δεν τα διαπραγματεύεται).', false),
      ]),
    ],
  ),
  'shortcuts': const _HelpSection(
    title: 'Συντομεύσεις',
    blocks: [
      _H3('Πληκτρολόγιο'),
      _KbdRow(['F1'], 'Άνοιγμα Help (αυτό το παράθυρο)'),
      _KbdRow(['Ctrl', 'K'],
          'Command palette — αναζήτηση σε tabs / symbols / actions (έρχεται στο F1-323)'),
      _KbdRow(['Esc'], 'Κλείνει modal / palette / context menu'),
      _KbdRow(['?'], 'Άνοιγμα Help (alternative)'),
      _KbdRow(['↑', '↓', '↵'], 'Πλοήγηση μέσα σε palette αναζήτησης'),
      _KbdRow(['Right-click'],
          'Context menu σε symbol chips, timeframes, chart'),
      _H3('Right-click γρήγορες ενέργειες'),
      _UL([
        [('Σε ', false), ('symbol chip', true), (' (Discovery/Training): Show on chart · Open Order Ticket · Add to favorites · Symbol specs · Remove', false)],
        [('Σε ', false), ('timeframe chip', true), (': Set as chart TF · Custom interval · Remove', false)],
        [('Σε ', false), ('chart', true), (': Drawing tools (trend line, fib retracement, rectangle, text)', false)],
      ]),
    ],
  ),
  'faq': const _HelpSection(
    title: 'FAQ',
    blocks: [
      _H3('Συχνές ερωτήσεις'),
      _P([
        ('Είναι ασφαλές το AI auto-trading;', true),
      ]),
      _P([('Όχι από μόνο του. Το auto trade πατιέται μόνο μετά από εβδομάδες επικύρωσης σε demo. Πάντα έχει ενεργό το Risk Guard και το Risky Mode kill-switch ώστε να μπλοκάρει υπερβολικά ρίσκα.', false)]),
      _P([
        ('Πόση RAM/CPU χρειάζεται;', true),
      ]),
      _P([('Για βασικό use: 4 cores, 8 GB RAM. Για Discovery + Training παράλληλα: 8-16 cores, 16-32 GB RAM. GPU (Vulkan, NVIDIA CUDA ή AMD ROCm) ωφελεί τα deep timeseries μοντέλα (Transformer, PatchTST, TimesNet).', false)]),
      _P([
        ('Τι είναι το Risky Mode;', true),
      ]),
      _P([('Aggressive compounding μέθοδος όπου ξεκινάς με μικρό κεφάλαιο (π.χ. \$20) και στοχεύεις σε μεγάλο target μέσω καθημερινής αύξησης lot size. Δεν είναι hardcoded "20 pips" — είναι Kelly-aligned και έχει risk-of-ruin probability calculator.', false)]),
      _P([
        ('Πού αποθηκεύονται τα credentials;', true),
      ]),
      _P([('Στο τοπικό σου δίσκο (Windows Credential Manager + broker_credentials.toml). Ποτέ δεν στέλνονται σε εξωτερικό server. Το NeoEthos δεν έχει cloud sync.', false)]),
      _P([
        ('Γιατί έχω "no strategies found" στο Discovery;', true),
      ]),
      _P([('Πλέον το NeoEthos σου λέει ', false), ('ακριβώς ποιο στάδιο', true), (' του pipeline έκοψε τα πάντα και τι να κάνεις — διάβασε το μήνυμα αποτυχίας (π.χ. "stage \'passed_quality\' rejected 412 of 412 — lower min Sharpe / win-rate, or enable opportunistic mode"). Οι συνηθισμένες αιτίες: (α) λίγα bars — τρέξε Data Bootstrap ή άσε το auto-fetch, (β) πολύ αυστηρά thresholds — χαλάρωσέ τα στο στάδιο που αναφέρει το μήνυμα, (γ) λάθος account currency — δες Settings → Account.', false)]),
    ],
  ),
};

// ---------------------------------------------------------------------------
// English content (shorter — covers the essentials)
// ---------------------------------------------------------------------------

Map<String, _HelpSection> _helpContentEn(BuildContext context) {
  final l10n = AppLocalizations.of(context)!;
  return {
    'welcome': _HelpSection(
      title: l10n.helpWelcomeTitle,
      blocks: [
        _H3(l10n.helpWelcomeOverviewHeading),
        _P([
          (l10n.helpWelcomeOverviewP1, false),
          ('Rust-based trading terminal', true),
          (l10n.helpWelcomeOverviewP2, false),
          ('Discovery', true),
          (l10n.helpWelcomeOverviewP3, false),
          ('Training', true),
          (l10n.helpWelcomeOverviewP4, false),
        ]),
        _H4(l10n.helpWelcomeFlowHeading),
        _OL([
          [(l10n.helpWelcomeStep1Label, true), (l10n.helpWelcomeStep1Body, false)],
          [(l10n.helpWelcomeStep2Label, true), (' Settings → Account → "Connect with cTrader"', false)],
          [(l10n.helpWelcomeStep3Label, true), (l10n.helpWelcomeStep3Body, false)],
          [(l10n.helpWelcomeStep4Label, true), (l10n.helpWelcomeStep4Body, false)],
          [('Run Strategy Lab → Discovery', true), (l10n.helpWelcomeStep5Body, false)],
          [(l10n.helpWelcomeStep6Label, true), (l10n.helpWelcomeStep6Body, false)],
          [(l10n.helpWelcomeStep7Label, true), (l10n.helpWelcomeStep7Body, false)],
        ]),
        _Tip(l10n.helpWelcomeTip),
      ],
    ),
    'trading': _HelpSection(
      title: 'Trading',
      blocks: [
        _H3(l10n.helpTradingManualHeading),
        _P([
          (l10n.helpTradingManualP1, false),
          ('Market Watch → Order Ticket', true),
          (l10n.helpTradingManualP2, false),
        ]),
        _UL([
          [('Order type: ', true), (l10n.helpTradingOrderType, false)],
          [('Lots: ', true), (l10n.helpTradingLots, false)],
          [('Auto Lot: ', true), (l10n.helpTradingAutoLot, false)],
          [('SL/TP (ATR): ', true), (l10n.helpTradingSlTp, false)],
          [('R:R: ', true), (l10n.helpTradingRr, false)],
        ]),
      ],
    ),
    'chart': _HelpSection(
      title: l10n.chartTitle,
      blocks: [
        _H3(l10n.helpChartLiveHeading),
        _P([
          (l10n.helpChartLiveP1, false),
          (l10n.helpChartLiveP1Bold, true),
          (l10n.helpChartLiveP2, false),
        ]),
        _H4(l10n.helpChartScrollbackHeading),
        _P([
          (l10n.helpChartScrollbackP1, false),
          (l10n.helpChartScrollbackLeft, true),
          (l10n.helpChartScrollbackP2, false),
          (l10n.helpChartScrollbackP3Bold, true),
          (l10n.helpChartScrollbackP4, false),
        ]),
        _Tip(l10n.helpChartTip),
        _H4(l10n.helpChartControlsHeading),
        _UL([
          [(l10n.helpChartZoomLabel, true), (l10n.helpChartZoomBody, false)],
          [(l10n.helpChartCrosshairLabel, true), (l10n.helpChartCrosshairBody, false)],
          [(l10n.helpChartOverlaysLabel, true), (l10n.helpChartOverlaysBody, false)],
          [(l10n.helpChartAccountSwitcherLabel, true), (l10n.helpChartAccountSwitcherBody, false)],
        ]),
      ],
    ),
    'ai': _HelpSection(
      title: l10n.helpAiTitle,
      blocks: [
        _H3(l10n.helpAiPipelineHeading),
        _P([
          (l10n.helpAiPipelineP1, false),
        ]),
        _H4(l10n.helpAiDiscoveryHeading),
        _P([(l10n.helpAiDiscoveryP1, false)]),
        _H4(l10n.helpAiTrainingHeading),
        _P([
          (l10n.helpAiTrainingP1, false),
          (l10n.helpAiTrainingP1Bold, true),
        ]),
        _H4(l10n.helpAiValidationHeading),
        _P([(l10n.helpAiValidationP1, false)]),
      ],
    ),
    'risk': _HelpSection(
      title: l10n.helpRiskTitle,
      blocks: [
        _H3(l10n.helpRiskHeading),
        _UL([
          [(l10n.helpRiskStandardLabel, true), (l10n.helpRiskStandardBody, false)],
          [(l10n.helpRiskPropFirmLabel, true), (l10n.helpRiskPropFirmBody, false)],
          [(l10n.helpRiskRiskyLabel, true), (l10n.helpRiskRiskyBody, false)],
        ]),
        _H4(l10n.helpRiskRiskyHeading),
        _P([
          (l10n.helpRiskRiskyP1, false),
          ('NOT', true),
          (l10n.helpRiskRiskyP2, false),
        ]),
        _Tip(l10n.helpRiskTip),
      ],
    ),
    'data': _HelpSection(
      title: l10n.helpDataTitle,
      blocks: [
        _H3(l10n.helpDataSourceHeading),
        _P([
          (l10n.helpDataSourceP1, false),
          (l10n.helpDataSourceP1Bold, true),
          (l10n.helpDataSourceP2, false),
        ]),
        _UL([
          [(l10n.helpDataLiveLabel, true), (l10n.helpDataLiveBody, false), (l10n.helpDataLiveBodyBold, true)],
          [(l10n.helpDataCacheLabel, true), (l10n.helpDataCacheBody, false), (l10n.helpDataCacheBodyBold, true), (l10n.helpDataCacheBody2, false)],
        ]),
        _H4(l10n.helpDataWritesHeading),
        _UL([
          [('Data Bootstrap', true), (l10n.helpDataBootstrapBody, false)],
          [('Discovery auto-fetch: ', true), (l10n.helpDataAutofetchBody, false)],
          [('data_dir: ', true), (l10n.helpDataDirBody, false)],
        ]),
        _Tip(l10n.helpDataTip),
        _H4(l10n.helpDataSymbolsHeading),
        _P([
          (l10n.helpDataSymbolsP1, false),
          (l10n.helpDataSymbolsP1Bold, true),
          (l10n.helpDataSymbolsP2, false),
        ]),
      ],
    ),
    'shortcuts': _HelpSection(
      title: l10n.helpShortcutsTitle,
      blocks: [
        _H3(l10n.helpShortcutsKeyboardHeading),
        _KbdRow(const ['F1'], l10n.helpShortcutF1),
        _KbdRow(const ['Ctrl', 'K'], l10n.helpShortcutCtrlK),
        _KbdRow(const ['Esc'], l10n.helpShortcutEsc),
        _KbdRow(const ['?'], l10n.helpShortcutQuestion),
        _KbdRow(const ['↑', '↓', '↵'], l10n.helpShortcutArrows),
        _KbdRow(const ['Right-click'], l10n.helpShortcutRightClick),
      ],
    ),
    'faq': _HelpSection(
      title: l10n.helpFaqTitle,
      blocks: [
        _H3(l10n.helpFaqHeading),
        _P([(l10n.helpFaq1Question, true)]),
        _P([(l10n.helpFaq1Answer, false)]),
        _P([(l10n.helpFaq2Question, true)]),
        _P([(l10n.helpFaq2Answer, false)]),
        _P([(l10n.helpFaq3Question, true)]),
        _P([(l10n.helpFaq3Answer, false)]),
        _P([(l10n.helpFaq4Question, true)]),
        _P([(l10n.helpFaq4Answer, false)]),
      ],
    ),
  };
}
