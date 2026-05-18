//! Anchor corpus — the curated in-scope / out-of-scope reference
//! sentences the embedding gate measures against.
//!
//! Phase G2 — 40+40 anchors live, multilingual (EN+EL).
//!
//! ## Curation principles
//!
//! - **In-scope**: questions about forex-ai itself — bot, models,
//!   trades, positions, broker setup, wizard, risk config, news.
//! - **Out-of-scope**: general chat, jokes, world news / politics
//!   not tied to trading, personal advice, creative writing.
//! - **Balanced EN / EL**: roughly half-and-half so the gate
//!   handles either language without bias.
//! - **No PII**: anchors are generic. The embedder MUST NOT
//!   memorise the operator's broker username / account codes.
//! - **≤ 200 chars per sentence**: keeps embedding cost
//!   bounded and lets the anchor file render in a list view.

use forex_core::{HasSchemaVersion, SchemaVersion, default_v1};
use serde::{Deserialize, Serialize};

pub const GEMMA_ANCHORS_SCHEMA_VERSION: SchemaVersion = SchemaVersion::new(1);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnchorCorpus {
    #[serde(default = "default_v1")]
    pub schema_version: SchemaVersion,
    pub revision: String,
    pub in_scope: Vec<AnchorSentence>,
    pub out_of_scope: Vec<AnchorSentence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnchorSentence {
    pub text: String,
    #[serde(default)]
    pub lang: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
}

impl HasSchemaVersion for AnchorCorpus {
    const CURRENT: SchemaVersion = GEMMA_ANCHORS_SCHEMA_VERSION;
    fn schema_version(&self) -> SchemaVersion {
        self.schema_version
    }
}

/// Helper to keep the anchor table compact.
fn a(text: &str, lang: &str, category: &str) -> AnchorSentence {
    AnchorSentence {
        text: text.to_string(),
        lang: Some(lang.to_string()),
        category: Some(category.to_string()),
    }
}

impl AnchorCorpus {
    /// Built-in placeholder (5+5) shipped with G0 — kept for
    /// backward-compat with anything that referenced the small
    /// fixture explicitly. New code should use
    /// `g2_curated_v1()` instead.
    pub fn g0_placeholder() -> Self {
        Self {
            schema_version: GEMMA_ANCHORS_SCHEMA_VERSION,
            revision: "g0-placeholder".to_string(),
            in_scope: vec![
                a("Show me my open positions", "en", "about_positions"),
                a(
                    "What does the ensemble predict for EUR/USD?",
                    "en",
                    "about_models",
                ),
                a(
                    "How do I configure my cTrader broker?",
                    "en",
                    "about_broker_setup",
                ),
                a(
                    "Πώς ρυθμίζω τον DXtrade broker;",
                    "el",
                    "about_broker_setup",
                ),
                a(
                    "Γιατί απορρίφθηκε η τελευταία μου εντολή;",
                    "el",
                    "about_trades",
                ),
            ],
            out_of_scope: vec![
                a(
                    "What's the weather like today?",
                    "en",
                    "off_topic_smalltalk",
                ),
                a("Tell me a joke", "en", "off_topic_entertainment"),
                a("Help me write a poem", "en", "off_topic_creative"),
                a(
                    "Πες μου την ιστορία της Ελλάδας",
                    "el",
                    "off_topic_education",
                ),
                a("Συγκρίνε τους πολιτικούς", "el", "off_topic_politics"),
            ],
        }
    }

    /// G2 curated corpus — 40 in-scope + 40 out-of-scope
    /// sentences spanning trading / broker / models / wizard /
    /// risk plus a wide off-topic surface. Reviewed once;
    /// re-curate when the bot grows new surfaces.
    pub fn g2_curated_v1() -> Self {
        Self {
            schema_version: GEMMA_ANCHORS_SCHEMA_VERSION,
            revision: "g2-curated-v1".to_string(),
            in_scope: in_scope_anchors_v1(),
            out_of_scope: out_of_scope_anchors_v1(),
        }
    }
}

fn in_scope_anchors_v1() -> Vec<AnchorSentence> {
    vec![
        // ── Positions / orders ────────────────────────────
        a("Show me my open positions", "en", "positions"),
        a("List all my pending orders for today", "en", "orders"),
        a("What's my current PnL on EUR/USD?", "en", "positions"),
        a("Has my last submitted order filled?", "en", "orders"),
        a("Close my XAU/USD position at market", "en", "orders"),
        a("Δείξε μου τις ανοιχτές θέσεις μου", "el", "positions"),
        a("Ποιες εντολές μου είναι εκκρεμείς;", "el", "orders"),
        a(
            "Ποιο είναι το PnL μου σήμερα στο GBP/USD;",
            "el",
            "positions",
        ),
        // ── Models / ensemble ─────────────────────────────
        a(
            "What does the ensemble predict for EUR/USD on H1?",
            "en",
            "models",
        ),
        a("Why did the model vote short on USD/JPY?", "en", "models"),
        a(
            "Which expert had the highest confidence last bar?",
            "en",
            "models",
        ),
        a("Show me recent predictions for AUD/USD", "en", "models"),
        a("Τι προβλέπει το ensemble για το EUR/USD;", "el", "models"),
        a(
            "Ποιο μοντέλο ψήφισε long στο τελευταίο bar;",
            "el",
            "models",
        ),
        a(
            "Δείξε μου τις τελευταίες προβλέψεις του ensemble",
            "el",
            "models",
        ),
        // ── Broker setup ──────────────────────────────────
        a(
            "How do I configure my cTrader broker?",
            "en",
            "broker_setup",
        ),
        a(
            "Where do I enter my DXtrade username and password?",
            "en",
            "broker_setup",
        ),
        a("Why is my cTrader OAuth failing?", "en", "broker_setup"),
        a("Walk me through the broker wizard", "en", "broker_setup"),
        a("Πώς συνδέω τον cTrader broker μου;", "el", "broker_setup"),
        a("Πού βάζω τα στοιχεία του DXtrade;", "el", "broker_setup"),
        a(
            "Πώς ελέγχω αν είμαι συνδεδεμένος στον broker;",
            "el",
            "broker_setup",
        ),
        // ── Risk / Risky Mode ─────────────────────────────
        a("Explain Risky Mode to me", "en", "risk"),
        a("What's my max drawdown limit right now?", "en", "risk"),
        a("How does the news blackout window work?", "en", "risk"),
        a(
            "Why was my manual order rejected by Risky Mode?",
            "en",
            "risk",
        ),
        a("Πώς λειτουργεί το Risky Mode;", "el", "risk"),
        a("Ποιο είναι το όριο απωλειών για σήμερα;", "el", "risk"),
        a("Γιατί απορρίφθηκε η manual εντολή μου;", "el", "risk"),
        // ── Wizard / config ───────────────────────────────
        a("How do I reset the wizard?", "en", "wizard"),
        a("Walk me through the autonomy risk step", "en", "wizard"),
        a(
            "What does the autonomous-only contract mean?",
            "en",
            "wizard",
        ),
        a("Πώς επαναφέρω τον wizard;", "el", "wizard"),
        a("Τι κάνει το autonomous-only contract;", "el", "wizard"),
        // ── Quotes / market data ──────────────────────────
        a("What's the current EUR/USD bid?", "en", "quotes"),
        a("Show me the last hour of XAU/USD bars", "en", "quotes"),
        a("Is the market open right now?", "en", "quotes"),
        a("Ποια είναι η τρέχουσα τιμή του χρυσού;", "el", "quotes"),
        a("Δείξε μου το τελευταίο bar του USD/JPY", "el", "quotes"),
        // ── Operations / health ───────────────────────────
        a("Is the autonomous trader still running?", "en", "ops"),
        a("Show me the recent app log entries", "en", "ops"),
        a("Why did the training job fail?", "en", "ops"),
        a("Ποια είναι η κατάσταση του autonomous trader;", "el", "ops"),
    ]
}

fn out_of_scope_anchors_v1() -> Vec<AnchorSentence> {
    vec![
        // ── General small-talk ────────────────────────────
        a("What's the weather like today?", "en", "smalltalk"),
        a("How are you doing today?", "en", "smalltalk"),
        a("What time is it in Tokyo?", "en", "smalltalk"),
        a("Tell me about your day", "en", "smalltalk"),
        a("Τι κάνεις σήμερα;", "el", "smalltalk"),
        a("Τι ώρα είναι στη Νέα Υόρκη;", "el", "smalltalk"),
        a("Πώς πάει η μέρα σου;", "el", "smalltalk"),
        // ── Entertainment / creative ──────────────────────
        a("Tell me a joke", "en", "entertainment"),
        a("Help me write a poem about the sea", "en", "creative"),
        a("Recommend a movie to watch tonight", "en", "entertainment"),
        a("Write a short story about a dragon", "en", "creative"),
        a("Πες μου ένα ανέκδοτο", "el", "entertainment"),
        a("Γράψε μου ένα ποίημα για τη θάλασσα", "el", "creative"),
        a("Πρότεινε μου μία ταινία", "el", "entertainment"),
        // ── World / general knowledge ─────────────────────
        a("Who won the World Cup in 2018?", "en", "trivia"),
        a("Explain quantum mechanics to me", "en", "education"),
        a("Tell me about ancient Rome", "en", "education"),
        a("What is the capital of Brazil?", "en", "trivia"),
        a("Ποιος κέρδισε το Μουντιάλ το 2018;", "el", "trivia"),
        a("Εξήγησέ μου τη θεωρία της σχετικότητας", "el", "education"),
        a("Πες μου για την αρχαία Ελλάδα", "el", "education"),
        // ── Politics / opinion ────────────────────────────
        a(
            "What do you think about the upcoming election?",
            "en",
            "politics",
        ),
        a("Compare the two main political parties", "en", "politics"),
        a("Should I vote for candidate X?", "en", "politics"),
        a("Συγκρίνε τους πολιτικούς αρχηγούς", "el", "politics"),
        a("Ποιο κόμμα προτείνεις να ψηφίσω;", "el", "politics"),
        // ── Personal life advice ──────────────────────────
        a("Should I break up with my partner?", "en", "personal"),
        a("Help me plan my career change", "en", "personal"),
        a("I'm feeling sad today, what should I do?", "en", "personal"),
        a(
            "My friend is upset, how should I help them?",
            "en",
            "personal",
        ),
        a("Είμαι λυπημένος, τι να κάνω;", "el", "personal"),
        a("Πρέπει να αλλάξω δουλειά;", "el", "personal"),
        // ── Programming / unrelated tech ──────────────────
        a("Write a Python function for me", "en", "tech_general"),
        a(
            "How do I install Docker on my laptop?",
            "en",
            "tech_general",
        ),
        a("Debug this JavaScript code", "en", "tech_general"),
        a("Πώς εγκαθιστώ τον Docker;", "el", "tech_general"),
        a("Γράψε μου ένα Python script", "el", "tech_general"),
        // ── Recipe / lifestyle ────────────────────────────
        a("Give me a recipe for moussaka", "en", "lifestyle"),
        a("How do I cook a steak?", "en", "lifestyle"),
        a("Πώς φτιάχνω παστίτσιο;", "el", "lifestyle"),
        a("Δώσε μου μια συνταγή για μουσακά", "el", "lifestyle"),
        a("Recommend a workout routine", "en", "lifestyle"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g0_placeholder_has_both_sides() {
        let c = AnchorCorpus::g0_placeholder();
        assert!(!c.in_scope.is_empty());
        assert!(!c.out_of_scope.is_empty());
    }

    #[test]
    fn g2_curated_corpus_has_at_least_40_in_each_side() {
        let c = AnchorCorpus::g2_curated_v1();
        assert!(
            c.in_scope.len() >= 40,
            "expected ≥40 in-scope anchors, got {}",
            c.in_scope.len()
        );
        assert!(
            c.out_of_scope.len() >= 40,
            "expected ≥40 out-of-scope anchors, got {}",
            c.out_of_scope.len()
        );
    }

    #[test]
    fn g2_corpus_balances_languages() {
        let c = AnchorCorpus::g2_curated_v1();
        let count_lang = |pool: &[AnchorSentence], lang: &str| {
            pool.iter()
                .filter(|s| s.lang.as_deref() == Some(lang))
                .count()
        };
        // At least 30% of each side is Greek (EL anchors).
        let in_el = count_lang(&c.in_scope, "el");
        let out_el = count_lang(&c.out_of_scope, "el");
        assert!(
            in_el >= c.in_scope.len() * 3 / 10,
            "EL in-scope underrepresented: {in_el} of {}",
            c.in_scope.len()
        );
        assert!(
            out_el >= c.out_of_scope.len() * 3 / 10,
            "EL out-of-scope underrepresented: {out_el} of {}",
            c.out_of_scope.len()
        );
    }

    #[test]
    fn every_anchor_text_under_200_chars() {
        let c = AnchorCorpus::g2_curated_v1();
        for s in c.in_scope.iter().chain(c.out_of_scope.iter()) {
            assert!(s.text.chars().count() <= 200, "anchor too long: {}", s.text);
        }
    }

    #[test]
    fn every_anchor_carries_a_category_tag() {
        let c = AnchorCorpus::g2_curated_v1();
        for s in c.in_scope.iter().chain(c.out_of_scope.iter()) {
            assert!(s.category.is_some(), "missing category on: {}", s.text);
        }
    }

    #[test]
    fn pre_versioning_anchors_default_to_v1() {
        let raw = r#"{ "revision": "pre-v1", "in_scope": [], "out_of_scope": [] }"#;
        let parsed: AnchorCorpus = serde_json::from_str(raw).expect("de");
        assert_eq!(parsed.schema_version, SchemaVersion::new(1));
    }

    #[test]
    fn corpus_round_trips_through_json() {
        let c = AnchorCorpus::g2_curated_v1();
        let s = serde_json::to_string(&c).expect("ser");
        let back: AnchorCorpus = serde_json::from_str(&s).expect("de");
        assert_eq!(back, c);
    }

    #[test]
    fn has_schema_version_trait_returns_field() {
        assert_eq!(
            AnchorCorpus::g2_curated_v1().schema_version(),
            GEMMA_ANCHORS_SCHEMA_VERSION
        );
    }

    #[test]
    fn g2_corpus_categories_cover_known_surfaces() {
        // Pin that the curated corpus mentions the bot surfaces
        // the helper needs to talk about. If any of these
        // categories disappear from the corpus, the gate's
        // recall on those surfaces will silently drop.
        let c = AnchorCorpus::g2_curated_v1();
        let cats: std::collections::HashSet<&str> = c
            .in_scope
            .iter()
            .filter_map(|s| s.category.as_deref())
            .collect();
        for required in &[
            "positions",
            "orders",
            "models",
            "broker_setup",
            "risk",
            "wizard",
            "quotes",
            "ops",
        ] {
            assert!(
                cats.contains(required),
                "in-scope corpus missing category: {required}"
            );
        }
    }
}
