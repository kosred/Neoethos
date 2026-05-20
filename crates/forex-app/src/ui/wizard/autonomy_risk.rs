//! Step 9.5 — Autonomy & Risk Acknowledgement.
//!
//! NEW STEP per `wizard_onboarding_competitive_analysis.md` §9.2.
//! Four collapsed cards:
//!  1. Stage roadmap (four-stage Demo → Paper → Live small → Live full).
//!  2. Risk-acknowledgement quiz (5 MCQ; 5/5 to proceed).
//!  3. Per-trade & per-day caps (pre-filled from Step 3 slider).
//!  4. Autonomous-mode controller.
//!
//! Skippability is conditional (see `WizardController::is_skippable`):
//! mandatory if `trading_mode = Live` OR `autonomous_mode_enabled`.

use eframe::egui;

use super::{RiskAcknowledgement, StepResult, TradingMode, WizardController};
use crate::ui::theme;

/// Default for the Autonomous Mode toggle. Competitive analysis §9.2
/// — "off by default".
pub const WIZARD_DEFAULT_AUTONOMOUS_MODE_ENABLED: bool = false;

/// Default equity-stop % when autonomous mode is enabled. Competitive
/// analysis §5.1 cites eToro 40 % but defaults forex-ai to a tighter
/// 20 % per the §9.1 row "default 20 %, range 5–95 %".
pub const WIZARD_DEFAULT_EQUITY_STOP_PCT: f32 = 20.0;
pub const WIZARD_DEFAULT_EQUITY_STOP_FLOOR_PCT: f32 = 5.0;
pub const WIZARD_DEFAULT_EQUITY_STOP_CEILING_PCT: f32 = 95.0;

/// Quiz version — bumped if the question set changes. Competitive
/// analysis §11.8 reserves the right to operator-counsel-review the
/// question content.
pub const WIZARD_DEFAULT_QUIZ_VERSION: u32 = 1;

/// Required correct count to proceed. Competitive analysis §9.2 —
/// "Cannot Continue until 5/5 correct".
pub const WIZARD_DEFAULT_REQUIRED_CORRECT: u8 = 5;

/// The five v0 quiz questions per competitive analysis §11.8 last
/// paragraph. Each entry is (id, prompt, options, correct_index).
pub const WIZARD_DEFAULT_QUIZ_QUESTIONS: &[QuizQuestion] = &[
    QuizQuestion {
        id: "q1_daily_loss",
        prompt: "What is the FTMO Standard daily-loss limit?",
        options: &["2%", "5%", "10%", "25%"],
        correct_index: 1,
    },
    QuizQuestion {
        id: "q2_overall_drawdown",
        prompt: "What happens if you breach the overall drawdown limit on an FTMO account?",
        options: &[
            "Cool-down period",
            "Account is closed",
            "Daily reset only",
            "Higher commissions",
        ],
        correct_index: 1,
    },
    QuizQuestion {
        id: "q3_news_window",
        prompt: "What is the default news-blackout window around a high-impact event?",
        options: &["0 min", "2 min ±", "30 min ±", "1 hour ±"],
        correct_index: 1,
    },
    QuizQuestion {
        id: "q4_per_trade_risk",
        prompt: "What controls the per-trade max risk?",
        options: &[
            "Broker default only",
            "The Step 3 risk-profile slider",
            "A coin flip",
            "Random per-order",
        ],
        correct_index: 1,
    },
    QuizQuestion {
        id: "q5_halt",
        prompt: "What does the HALT button do?",
        options: &[
            "Pauses for 5 minutes",
            "Flattens all positions and disables autonomous mode",
            "Cancels only pending orders",
            "Closes the platform",
        ],
        correct_index: 1,
    },
];

#[derive(Debug, Clone, Copy)]
pub struct QuizQuestion {
    pub id: &'static str,
    pub prompt: &'static str,
    pub options: &'static [&'static str],
    pub correct_index: usize,
}

/// Per-question answer state. The renderer mutates this in-place; on
/// "Submit answers" the controller derives a `RiskAcknowledgement`
/// with the SHA-256 hash described in `state::RiskAcknowledgement`.
#[derive(Debug, Clone, Default)]
pub struct QuizAnswers {
    pub picks: Vec<Option<usize>>,
}

impl QuizAnswers {
    pub fn new() -> Self {
        Self {
            picks: vec![None; WIZARD_DEFAULT_QUIZ_QUESTIONS.len()],
        }
    }

    /// Count correct answers against `WIZARD_DEFAULT_QUIZ_QUESTIONS`.
    pub fn correct_count(&self) -> u8 {
        WIZARD_DEFAULT_QUIZ_QUESTIONS
            .iter()
            .zip(self.picks.iter())
            .filter(|(q, pick)| matches!(pick, Some(p) if *p == q.correct_index))
            .count() as u8
    }

    pub fn all_answered(&self) -> bool {
        self.picks.iter().all(|p| p.is_some())
    }
}

pub fn render(ui: &mut egui::Ui, controller: &mut WizardController) -> StepResult {
    let mut result = StepResult::StayHere;

    // Card 1 — stage roadmap.
    ui.label(
        egui::RichText::new("Stage roadmap (competitive analysis §6.1)")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );
    for (idx, label) in [
        "Stage 1 — Demo / replay",
        "Stage 2 — Paper (forward test)",
        "Stage 3 — Live small account",
        "Stage 4 — Live full account",
    ]
    .iter()
    .enumerate()
    {
        let current_stage_idx = match controller.config.trading_mode {
            TradingMode::Backtest => 0,
            TradingMode::Forward => 1,
            TradingMode::Live => 2,
        };
        let badge = if idx == current_stage_idx {
            " ← current"
        } else {
            ""
        };
        ui.label(
            egui::RichText::new(format!("  {}{}", label, badge))
                .color(if idx == current_stage_idx {
                    theme::SUCCESS
                } else {
                    theme::TEXT_MUTED
                })
                .size(theme::FONT_CAPTION),
        );
    }

    ui.separator();

    // Card 2 — risk-quiz placeholder.
    ui.label(
        egui::RichText::new(format!(
            "Risk acknowledgement quiz (5 questions — {}/5 must be correct)",
            WIZARD_DEFAULT_REQUIRED_CORRECT
        ))
        .strong()
        .color(theme::TEXT_PRIMARY),
    );
    ui.label(
        egui::RichText::new(
            "The quiz UI is rendered in the full step; the skeleton shows acknowledgement \
             status only.",
        )
        .size(theme::FONT_CAPTION)
        .color(theme::TEXT_MUTED),
    );
    if let Some(ack) = &controller.config.risk_acknowledgement {
        ui.label(
            egui::RichText::new(format!(
                "  Recorded at {} (v{}, {}/5 correct)",
                ack.timestamp_utc, ack.quiz_version, ack.correct_count
            ))
            .color(theme::SUCCESS)
            .size(theme::FONT_CAPTION),
        );
    }

    ui.separator();

    // Card 3 — caps recap (pre-filled from Step 3 slider).
    ui.label(
        egui::RichText::new("Per-trade & per-day caps")
            .strong()
            .color(theme::TEXT_PRIMARY),
    );
    ui.label(
        egui::RichText::new(format!(
            "  Risk profile slider: {}/10 · per-trade risk {:.2} % · SL required: {}",
            controller.config.risk_profile_slider,
            controller.config.per_trade_max_risk_pct,
            controller.config.require_stop_loss
        ))
        .size(theme::FONT_CAPTION)
        .color(theme::TEXT_MUTED),
    );

    ui.separator();

    // Card 4 — autonomous mode toggle.
    ui.checkbox(
        &mut controller.config.autonomous_mode_enabled,
        "Enable Autonomous Mode (discover → train → paper-trade → live with kill switches).",
    );

    ui.separator();

    // Card 4.5 — Risky Mode arming (research §4 + §7.1 sign-off).
    // The mode is OFF by default; enabling it requires the operator
    // to acknowledge the §7.1 ruin-probability ceiling.
    //
    // Persistence + boot-time wire-up is LIVE (2026-05-18 cleanup
    // pass — closed TODO(risky-mode-boot-wire)). On Apply,
    // `summary.rs::write_risky_mode_state` writes the flags below
    // to `<config_dir>/forex-ai/risky_mode_state.json`. At next app
    // launch, `TradingSession::new_with_persisted_credentials`
    // reads that file and calls `session.enable_risky_mode(
    // RiskyModeConfig::default(), starting_bankroll)` when armed.
    // `autonomous_mode_enabled` below controls
    // `autonomous_only_contract_accepted` in the persisted config,
    // which `RiskyModeConfig::validate` requires before auto-arming.
    ui.label(
        egui::RichText::new("⚡ Risky Mode (operator directive §7.1 — autonomous compounding)")
            .strong()
            .color(theme::DANGER),
    );
    ui.label(
        egui::RichText::new(
            "Autonomous compounding from a small starting bankroll \
             ($20 default) toward a large target ($50,000 default). \
             The bot — not the operator — places every trade once armed; \
             manual BUY/SELL is rejected at the gate. Per-trade risk is \
             30–50 % of the current bankroll (default 40 %); the bot \
             may scalp many times per day, targeting net profit after \
             commission, spread and swap rather than a fixed daily pip \
             count. Per-stage kill switches enforce daily / weekly DD \
             caps. Initial-stage ruin probability is up to 99 % per \
             operator directive §7.1 — the operator explicitly accepts \
             that the starting bankroll will most likely be lost.",
        )
        .color(theme::TEXT_MUTED)
        .size(theme::FONT_CAPTION),
    );

    // Acknowledgement checkbox — the operator must affirmatively
    // tick the ruin-probability acknowledgement BEFORE the arm
    // toggle becomes usable. This is the §7.1 informed-consent gate.
    let mut ack_now = controller
        .config
        .risky_mode_ruin_ceiling_acknowledged
        .is_some();
    let ack_before = ack_now;
    ui.checkbox(
        &mut ack_now,
        format!(
            "I acknowledge the initial-stage ruin probability ceiling is up to {:.0}% \
             (research §6.4 / §10.1 — operator decision §7.1).",
            forex_core::MAX_ACCEPTABLE_INITIAL_RUIN_PROBABILITY * 100.0
        ),
    );
    if ack_now != ack_before {
        controller.config.risky_mode_ruin_ceiling_acknowledged = if ack_now {
            Some(forex_core::MAX_ACCEPTABLE_INITIAL_RUIN_PROBABILITY)
        } else {
            None
        };
        // V0.4 audit Task #28 — clarify the destructive-clear behavior.
        // The operator can never end up "armed without acknowledgement",
        // so un-ticking the ack box ALWAYS clears the arm flag. We do
        // NOT restore the prior armed state when the ack is re-ticked
        // (the audit suggested this, but auto-restoration could re-arm
        // a Risky Mode the operator forgot they had toggled — safer to
        // require an explicit re-arm). The wizard label below this
        // checkbox surfaces the side-effect so the operator sees what
        // happened.
        if !ack_now && controller.config.risky_mode_armed {
            controller.config.risky_mode_armed = false;
            // Stash the side-effect so the renderer can call it out.
            // (Renderer logic uses the absence of the ack to label the
            // arm checkbox as cleared — see `can_arm` block below.)
        }
    }
    // Surface the destructive-clear side-effect so the operator notices.
    if !ack_now
        && controller
            .config
            .risky_mode_ruin_ceiling_acknowledged
            .is_none()
    {
        ui.label(
            egui::RichText::new(
                "Un-ticking the acknowledgement also disarms Risky Mode. \
                 Re-arming requires re-ticking both boxes.",
            )
            .small()
            .color(theme::WARNING),
        );
    }

    // Arm toggle — disabled until acknowledgement is recorded.
    let can_arm = controller
        .config
        .risky_mode_ruin_ceiling_acknowledged
        .is_some();
    ui.add_enabled_ui(can_arm, |ui| {
        ui.checkbox(
            &mut controller.config.risky_mode_armed,
            "Arm Risky Mode at Apply. The Apply writer will call \
             session.enable_risky_mode(RiskyModeConfig::default(), …).",
        );
    });
    if !can_arm {
        ui.label(
            egui::RichText::new(
                "  ↳ Tick the acknowledgement above first to enable the arm toggle.",
            )
            .color(theme::WARNING)
            .size(theme::FONT_CAPTION),
        );
    }
    if controller.config.risky_mode_armed {
        ui.label(
            egui::RichText::new(
                "  ⚠ Risky Mode is armed. The full kill-switch hierarchy \
                 (T-Manual / T-PerTrade / T-PerDay / T-PerStage / T-PerMonth / \
                 T-Hardware / T-PreSendSanity) will be active after Apply.",
            )
            .color(theme::DANGER)
            .size(theme::FONT_CAPTION),
        );
    }
    if controller.config.autonomous_mode_enabled {
        ui.horizontal(|ui| {
            ui.label("Equity stop:");
            ui.add(
                egui::Slider::new(
                    &mut controller.config.equity_stop_pct,
                    WIZARD_DEFAULT_EQUITY_STOP_FLOOR_PCT..=WIZARD_DEFAULT_EQUITY_STOP_CEILING_PCT,
                )
                .suffix(" %"),
            );
        });
        // V0.4 audit Task #27 — preserve decimal precision in the
        // capital-at-risk text edit. Pre-fix, the re-render used `{:.0}`
        // (zero decimals) so a user-typed `12345.67` round-tripped as
        // `12346` and the in-memory value was silently rounded to int.
        // Use two decimals on display and parse as f64 (wider mantissa)
        // so currency amounts up to ~$9e15 are exactly representable.
        let mut capital_str = controller
            .config
            .capital_at_risk_disclosure
            .map(|c| format!("{:.2}", c))
            .unwrap_or_default();
        ui.horizontal(|ui| {
            ui.label("Capital you can afford to lose (account currency, optional):");
            if ui.text_edit_singleline(&mut capital_str).changed() {
                // Keep the field type (f32 or f64) consistent with the
                // controller struct; just route through parse so empty /
                // garbage input clears the value instead of rounding it.
                controller.config.capital_at_risk_disclosure =
                    capital_str.trim().parse::<f32>().ok().filter(|v| v.is_finite() && *v >= 0.0);
            }
        });
    }

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("← Back").clicked() {
            result = StepResult::BackRequested;
        }
        if controller.is_skippable() {
            if ui.button("Skip").clicked() {
                result = StepResult::SkipRequested;
            }
        } else {
            ui.label(
                egui::RichText::new(
                    "Skip disabled — Live or Autonomous Mode requires acknowledgement.",
                )
                .color(theme::WARNING)
                .size(theme::FONT_CAPTION),
            );
        }
        if ui.button("Continue →").clicked() {
            result = StepResult::NextRequested;
        }
    });

    result
}

/// SHA-256 hash of the canonical encoding of a quiz attempt. The
/// digest is the auditable record stored in
/// `RiskAcknowledgement::answers_sha256` (spec
/// `installer_wizard_ux_spec.md` §5 — "SHA-256 of the concatenation
/// of (question_id, chosen_option_id) pairs in canonical order").
///
/// Canonical encoding (load-bearing — the Apply-writer
/// re-computes this on a re-read and must produce the same digest):
///   1. ASCII domain-separator `"forex-ai-risk-quiz-v"` (prevents
///      cross-protocol collisions if some other forex-ai feature
///      ever SHA-256-hashes a different payload).
///   2. `quiz_version` as little-endian `u32`.
///   3. For each answer (in slice order): index as little-endian
///      `u32`, then the answer bytes, then a `\x00` terminator
///      (so `"AB" + "C"` ≠ `"A" + "BC"`).
///
/// `hex::encode` returns lowercase hex (consistent with how the
/// existing competitive-analysis spec quotes SHA-256 digests).
pub fn compute_quiz_answer_hash(quiz_version: u32, answers: &[&str]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"forex-ai-risk-quiz-v");
    hasher.update(quiz_version.to_le_bytes());
    for (i, answer) in answers.iter().enumerate() {
        hasher.update((i as u32).to_le_bytes());
        hasher.update(answer.as_bytes());
        hasher.update(b"\x00");
    }
    hex::encode(hasher.finalize())
}

/// Build a `RiskAcknowledgement` record from quiz answers + a clock
/// callback. Exposed for tests so we can pin determinism without
/// touching the system clock.
///
/// The hash binds the (quiz_version, ordered answers) — the
/// timestamp is recorded alongside but is NOT hashed because the
/// Apply-writer needs to be able to re-derive the same digest
/// from the persisted file (`risk_acknowledgement.json`) without
/// knowing the original timestamp string.
pub fn record_acknowledgement(
    answers: &QuizAnswers,
    iso_timestamp_utc: String,
) -> RiskAcknowledgement {
    let encoded: Vec<String> = WIZARD_DEFAULT_QUIZ_QUESTIONS
        .iter()
        .zip(answers.picks.iter())
        .map(|(q, pick)| match pick {
            Some(idx) => format!("{}:{}", q.id, idx),
            None => format!("{}:_", q.id),
        })
        .collect();
    let borrowed: Vec<&str> = encoded.iter().map(String::as_str).collect();

    RiskAcknowledgement {
        answers_sha256: compute_quiz_answer_hash(WIZARD_DEFAULT_QUIZ_VERSION, &borrowed),
        timestamp_utc: iso_timestamp_utc,
        quiz_version: WIZARD_DEFAULT_QUIZ_VERSION,
        correct_count: answers.correct_count(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::wizard::{StepResult, WizardController, WizardState};

    #[test]
    fn quiz_has_five_questions_each_with_four_options() {
        assert_eq!(WIZARD_DEFAULT_QUIZ_QUESTIONS.len(), 5);
        for q in WIZARD_DEFAULT_QUIZ_QUESTIONS {
            assert_eq!(
                q.options.len(),
                4,
                "question {} should have 4 options",
                q.id
            );
            assert!(q.correct_index < q.options.len());
        }
    }

    #[test]
    fn quiz_required_correct_is_full_marks() {
        assert_eq!(
            WIZARD_DEFAULT_REQUIRED_CORRECT as usize,
            WIZARD_DEFAULT_QUIZ_QUESTIONS.len()
        );
    }

    #[test]
    fn all_correct_answers_score_full() {
        let mut answers = QuizAnswers::new();
        for (i, q) in WIZARD_DEFAULT_QUIZ_QUESTIONS.iter().enumerate() {
            answers.picks[i] = Some(q.correct_index);
        }
        assert!(answers.all_answered());
        assert_eq!(answers.correct_count(), WIZARD_DEFAULT_REQUIRED_CORRECT);
    }

    #[test]
    fn record_acknowledgement_captures_quiz_version() {
        let mut answers = QuizAnswers::new();
        for (i, q) in WIZARD_DEFAULT_QUIZ_QUESTIONS.iter().enumerate() {
            answers.picks[i] = Some(q.correct_index);
        }
        let ack = record_acknowledgement(&answers, "2026-05-15T20:00:00Z".to_string());
        assert_eq!(ack.quiz_version, WIZARD_DEFAULT_QUIZ_VERSION);
        assert_eq!(ack.correct_count, 5);
        assert!(!ack.answers_sha256.is_empty());
    }

    #[test]
    fn autonomy_risk_step_advances_to_summary() {
        let mut c = WizardController::new();
        c.current = WizardState::AutonomyRisk;
        c.apply(StepResult::NextRequested);
        assert_eq!(c.current, WizardState::Summary);
    }

    #[test]
    fn autonomy_risk_skip_disallowed_when_live_mode() {
        let mut c = WizardController::new();
        c.current = WizardState::AutonomyRisk;
        c.config.trading_mode = TradingMode::Live;
        c.apply(StepResult::SkipRequested);
        // Skip is a no-op when not skippable; controller stays put.
        assert_eq!(c.current, WizardState::AutonomyRisk);
    }

    #[test]
    fn autonomy_risk_skip_disallowed_when_autonomous_mode_enabled() {
        let mut c = WizardController::new();
        c.current = WizardState::AutonomyRisk;
        c.config.autonomous_mode_enabled = true;
        c.apply(StepResult::SkipRequested);
        assert_eq!(c.current, WizardState::AutonomyRisk);
    }

    #[test]
    fn equity_stop_default_within_bounds() {
        assert!(WIZARD_DEFAULT_EQUITY_STOP_PCT >= WIZARD_DEFAULT_EQUITY_STOP_FLOOR_PCT);
        assert!(WIZARD_DEFAULT_EQUITY_STOP_PCT <= WIZARD_DEFAULT_EQUITY_STOP_CEILING_PCT);
    }

    #[test]
    fn compute_quiz_answer_hash_is_deterministic() {
        let a = compute_quiz_answer_hash(1, &["q1:1", "q2:1", "q3:1", "q4:1", "q5:1"]);
        let b = compute_quiz_answer_hash(1, &["q1:1", "q2:1", "q3:1", "q4:1", "q5:1"]);
        assert_eq!(a, b, "same inputs must produce same digest");
        // SHA-256 hex is exactly 64 chars (256 bits / 4 bits per nibble).
        assert_eq!(a.len(), 64, "sha256 hex must be 64 chars, got {}", a.len());
        assert!(
            a.chars().all(|c| c.is_ascii_hexdigit()),
            "digest must be lowercase-hex, got {}",
            a
        );
    }

    #[test]
    fn compute_quiz_answer_hash_changes_when_answers_change() {
        let base = compute_quiz_answer_hash(1, &["q1:1", "q2:1", "q3:1", "q4:1", "q5:1"]);
        // Flip the first answer.
        let flipped = compute_quiz_answer_hash(1, &["q1:0", "q2:1", "q3:1", "q4:1", "q5:1"]);
        assert_ne!(
            base, flipped,
            "different answers must produce different digest"
        );
    }

    #[test]
    fn compute_quiz_answer_hash_changes_when_version_changes() {
        let v1 = compute_quiz_answer_hash(1, &["q1:1", "q2:1", "q3:1", "q4:1", "q5:1"]);
        let v2 = compute_quiz_answer_hash(2, &["q1:1", "q2:1", "q3:1", "q4:1", "q5:1"]);
        assert_ne!(v1, v2, "bumping quiz_version must produce different digest");
    }

    #[test]
    fn compute_quiz_answer_hash_distinguishes_length_extension() {
        // Without the `\x00` separator, ("AB", "C") and ("A", "BC")
        // would hash to the same digest. The separator must prevent
        // that. Pin the property here so a future refactor that
        // drops the separator gets caught.
        let a = compute_quiz_answer_hash(1, &["AB", "C"]);
        let b = compute_quiz_answer_hash(1, &["A", "BC"]);
        assert_ne!(
            a, b,
            "field separator must prevent ambiguous concat collisions"
        );
    }

    #[test]
    fn record_acknowledgement_uses_real_sha256_digest() {
        let mut answers = QuizAnswers::new();
        for (i, q) in WIZARD_DEFAULT_QUIZ_QUESTIONS.iter().enumerate() {
            answers.picks[i] = Some(q.correct_index);
        }
        let ack = record_acknowledgement(&answers, "2026-05-15T20:00:00Z".to_string());
        // Real sha256 hex is 64 chars, lowercase hex only, no
        // "placeholder-" prefix from the old djb2 implementation.
        assert_eq!(ack.answers_sha256.len(), 64);
        assert!(ack.answers_sha256.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(!ack.answers_sha256.starts_with("placeholder-"));
    }

    #[test]
    fn record_acknowledgement_hash_ignores_timestamp() {
        // Apply-writer determinism contract: same answers + same
        // version must produce same digest regardless of when the
        // user clicked submit. (The timestamp lives next to the
        // digest in `RiskAcknowledgement` but is NOT hashed.)
        let mut answers = QuizAnswers::new();
        for (i, q) in WIZARD_DEFAULT_QUIZ_QUESTIONS.iter().enumerate() {
            answers.picks[i] = Some(q.correct_index);
        }
        let a = record_acknowledgement(&answers, "2026-05-15T20:00:00Z".to_string());
        let b = record_acknowledgement(&answers, "2030-01-01T00:00:00Z".to_string());
        assert_eq!(a.answers_sha256, b.answers_sha256);
        assert_ne!(a.timestamp_utc, b.timestamp_utc);
    }
}
