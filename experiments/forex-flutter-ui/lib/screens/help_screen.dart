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

  Map<String, _HelpSection> get _doc =>
      _lang == 'el' ? _helpContentEl : _helpContentEn;

  @override
  Widget build(BuildContext context) {
    final sections = _doc;
    final active = sections[_section] ?? sections.values.first;
    return Scaffold(
      backgroundColor: ForexAiTokens.appBg,
      appBar: AppBar(
        backgroundColor: ForexAiTokens.panelBg,
        elevation: 0,
        title: Row(
          children: [
            const Text(
              'NeoEthos · Help',
              style: TextStyle(
                color: ForexAiTokens.textPrimary,
                fontSize: ForexAiTokens.fsSubtitle,
                fontWeight: FontWeight.w700,
              ),
            ),
            const SizedBox(width: ForexAiTokens.spLg),
            _LangToggle(
              lang: _lang,
              onChanged: (v) => setState(() => _lang = v),
            ),
          ],
        ),
        iconTheme: const IconThemeData(color: ForexAiTokens.textPrimary),
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
                horizontal: ForexAiTokens.spLg + 8,
                vertical: ForexAiTokens.spLg,
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
                        color: ForexAiTokens.textPrimary,
                      ),
                    ),
                    const SizedBox(height: ForexAiTokens.spMd),
                    for (final block in active.blocks) ...[
                      block.build(context),
                      const SizedBox(height: ForexAiTokens.spSm),
                    ],
                    const SizedBox(height: ForexAiTokens.spXl),
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
        border: Border.all(color: ForexAiTokens.border),
        borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
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
          color: active ? ForexAiTokens.accentMuted : Colors.transparent,
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: ForexAiTokens.fsCaption,
            fontWeight: active ? FontWeight.w700 : FontWeight.w500,
            color: active ? ForexAiTokens.accent : ForexAiTokens.textMuted,
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
        color: ForexAiTokens.panelBg,
        border: Border(right: BorderSide(color: ForexAiTokens.border)),
      ),
      padding: const EdgeInsets.symmetric(
        vertical: ForexAiTokens.spMd,
        horizontal: ForexAiTokens.spSm,
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
                      ? ForexAiTokens.accentMuted
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
                  border: active == entry.key
                      ? const Border(
                          left: BorderSide(
                            color: ForexAiTokens.accent,
                            width: 3,
                          ),
                        )
                      : null,
                ),
                child: Text(
                  entry.value.title,
                  style: TextStyle(
                    fontSize: ForexAiTokens.fsBody,
                    fontWeight: active == entry.key
                        ? FontWeight.w700
                        : FontWeight.w500,
                    color: active == entry.key
                        ? ForexAiTokens.textPrimary
                        : ForexAiTokens.textMuted,
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
            color: ForexAiTokens.textPrimary,
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
            color: ForexAiTokens.textPrimary,
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
                    color: ForexAiTokens.textPrimary,
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
                        color: ForexAiTokens.accent,
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
                                  color: ForexAiTokens.textPrimary,
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
                            color: ForexAiTokens.accent,
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
                                  color: ForexAiTokens.textPrimary,
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
        padding: const EdgeInsets.all(ForexAiTokens.spMd),
        decoration: BoxDecoration(
          color: ForexAiTokens.warning.withValues(alpha: 0.12),
          border: Border.all(
            color: ForexAiTokens.warning.withValues(alpha: 0.55),
          ),
          borderRadius: BorderRadius.circular(ForexAiTokens.rSm),
        ),
        child: Text(
          text,
          style: const TextStyle(
            fontSize: 13.5,
            height: 1.5,
            color: ForexAiTokens.warning,
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
                              color: ForexAiTokens.textFaint,
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
                    color: ForexAiTokens.textMuted,
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
          color: ForexAiTokens.appBg,
          border: Border.all(color: ForexAiTokens.border),
          borderRadius: BorderRadius.circular(4),
        ),
        child: Text(
          label,
          style: const TextStyle(
            fontFamily: 'monospace',
            fontSize: 12,
            fontWeight: FontWeight.w700,
            color: ForexAiTokens.textPrimary,
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
      _P([('Συνήθως είναι ένα από: (α) λίγα bars στο data dir — τρέξε Data Bootstrap πρώτα, (β) πολύ αυστηρά Promotion thresholds — άνοιξε το Strategy Lab → Validation και ελέγξε ποιες μετρικές κόβουν τις στρατηγικές, (γ) λάθος account currency — δες Settings → Account.', false)]),
    ],
  ),
};

// ---------------------------------------------------------------------------
// English content (shorter — covers the essentials)
// ---------------------------------------------------------------------------

final Map<String, _HelpSection> _helpContentEn = {
  'welcome': const _HelpSection(
    title: 'Welcome',
    blocks: [
      _H3('NeoEthos — Quick overview'),
      _P([
        ('NeoEthos is a ', false),
        ('Rust-based trading terminal', true),
        (' with built-in AI. It connects to a broker (cTrader; DXtrade soon), supports manual trading with indicators and drawing tools, and has two AI engines: ', false),
        ('Discovery', true),
        (' (genetic strategy search) and ', false),
        ('Training', true),
        (' (ensemble ML training). Every value comes from the broker — nothing is hardcoded.', false),
      ]),
      _H4('Recommended flow for new users'),
      _OL([
        [('Open a Demo account', true), (' at a cTrader broker (Pepperstone, IC Markets, FxPro, Vantage, Axi)', false)],
        [('Connect the broker', true), (' via Settings → Account → "Connect with cTrader"', false)],
        [('Pick pairs', true), (' in Market Watch — start with EUR/USD or GBP/USD', false)],
        [('Place a small manual trade', true), (' (0.01 lots) to confirm everything works', false)],
        [('Run Strategy Lab → Discovery', true), (' to find strategies (10-30 minutes)', false)],
        [('Run Training', true), (' on those strategies', false)],
        [('Validate for 2 weeks', true), (' on demo before considering live', false)],
      ]),
      _Tip('Reminder: 75-90% of retail FX traders lose money. Always validate on demo for weeks before increasing risk.'),
    ],
  ),
  'trading': const _HelpSection(
    title: 'Trading',
    blocks: [
      _H3('Manual trading'),
      _P([
        ('The ', false),
        ('Market Watch → Order Ticket', true),
        (' shows BUY (green) and SELL (red) buttons with the current bid/ask. Before submitting:', false),
      ]),
      _UL([
        [('Order type: ', true), ('Market (immediate), Limit (wait for better entry), Stop (wait for breakout)', false)],
        [('Lots: ', true), ('position size. 0.01 lot = 1,000 units (micro lot). For EUR/USD a pip ≈ \$0.10 per 0.01 lot', false)],
        [('Auto Lot: ', true), ('automatic lot sizing from your risk %', false)],
        [('SL/TP (ATR): ', true), ('Stop Loss & Take Profit based on volatility', false)],
        [('R:R: ', true), ('Risk/Reward ratio — 1:2 means for every \$1 risk you target \$2 profit', false)],
      ]),
    ],
  ),
  'ai': const _HelpSection(
    title: 'AI Engine',
    blocks: [
      _H3('Strategy Lab — How the pipeline works'),
      _P([
        ('One unified 5-stage flow: Data Ready → Discovery → Training → Validation → Promotion Gate. Each stage has its own params and the next consumes the previous automatically.', false),
      ]),
      _H4('Discovery — Genetic strategy search'),
      _P([('Creates random strategies from indicator combinations, evaluates them on history, keeps the best and crosses them over generations.', false)]),
      _H4('Training — Multi-model ensemble'),
      _P([
        ('Features (indicators, OHLCV, volatility, session, news) feed 30 models in parallel — Tree (LightGBM, XGBoost, CatBoost), Deep (MLP, KAN, TabNet), Time-series (NBeats, TiDE, PatchTST, TimesNet, Transformer), Meta (logistic, calibration, stacker, HMM regime), RL+Exit (DQN, ExitAgent). ', false),
        ('Genetic/NeuroEvo/NEAT are Discovery-only — not ensemble voters.', true),
      ]),
      _H4('Validation + Promotion Gate'),
      _P([('Before a trained ensemble reaches live, it passes Walk-Forward Analysis, Monte Carlo sensitivity sweeps, and a Promotion Gate that checks Sharpe, Calmar, win rate and drawdown.', false)]),
    ],
  ),
  'risk': const _HelpSection(
    title: 'Risk',
    blocks: [
      _H3('Risk management — 3 modes'),
      _UL([
        [('🏦 Standard account: ', true), ('regular retail. Risk 0.5-2% per trade, daily DD ≤ 5%.', false)],
        [('🏆 Prop firm challenge: ', true), ('hard daily/total drawdown limits (FTMO, FundedNext etc.).', false)],
        [('⚡ Risky Mode (account multiplication): ', true), ('aggressive compounding from a small balance (\$20 → \$50K). Includes time-to-target percentiles (p10/p50/p90) and probability-of-ruin via Brownian Barrier.', false)],
      ]),
      _H4('How Risky Mode works'),
      _P([
        ('Risky Mode is ', false),
        ('NOT', true),
        (' a hardcoded "20 pip challenge". It is Kelly-aligned compounding with 11 logarithmic stages from \$20 → \$50K. Time-to-target ETA via Brownian inversion + Beasley-Springer-Moro inverse-normal-CDF (~1e-5 accuracy). Goal: grow the account as fast as safely possible.', false),
      ]),
      _Tip('⚠ 75-90% of retail FX traders lose money. Risky Mode = aggressive ≠ safe. Always validate on demo for weeks first.'),
    ],
  ),
  'shortcuts': const _HelpSection(
    title: 'Shortcuts',
    blocks: [
      _H3('Keyboard'),
      _KbdRow(['F1'], 'Open Help (this window)'),
      _KbdRow(['Ctrl', 'K'],
          'Command palette — search tabs / symbols / actions (lands in F1-323)'),
      _KbdRow(['Esc'], 'Close any modal / palette / context menu'),
      _KbdRow(['?'], 'Show this help'),
      _KbdRow(['↑', '↓', '↵'], 'Navigate inside the search palette'),
      _KbdRow(['Right-click'],
          'Context menu on chips, symbols, timeframes, chart'),
    ],
  ),
  'faq': const _HelpSection(
    title: 'FAQ',
    blocks: [
      _H3('Frequent questions'),
      _P([('Is AI auto-trading safe?', true)]),
      _P([('Not on its own. Only enable auto-trade after weeks of demo validation. Risk Guard and the Risky Mode kill-switch enforce drawdown limits at all times.', false)]),
      _P([('How much RAM/CPU?', true)]),
      _P([('Basic: 4 cores, 8 GB. Discovery + Training in parallel: 8-16 cores, 16-32 GB. GPU (Vulkan, CUDA, ROCm) benefits the deep timeseries models.', false)]),
      _P([('Where are my credentials stored?', true)]),
      _P([('Locally on disk (Windows Credential Manager + broker_credentials.toml). Never sent to an external server. NeoEthos has no cloud sync.', false)]),
      _P([('Why does Discovery say "no strategies found"?', true)]),
      _P([('Usually one of: (a) too few bars in the data dir — run Data Bootstrap first, (b) Promotion thresholds too strict — check Strategy Lab → Validation, (c) wrong account currency — check Settings → Account.', false)]),
    ],
  ),
};
