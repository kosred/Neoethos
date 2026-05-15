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
        let badge = if idx == current_stage_idx { " ← current" } else { "" };
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
        let mut capital_str = controller
            .config
            .capital_at_risk_disclosure
            .map(|c| format!("{:.0}", c))
            .unwrap_or_default();
        ui.horizontal(|ui| {
            ui.label("Capital you can afford to lose (account currency, optional):");
            if ui.text_edit_singleline(&mut capital_str).changed() {
                controller.config.capital_at_risk_disclosure = capital_str.parse::<f32>().ok();
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
                egui::RichText::new("Skip disabled — Live or Autonomous Mode requires acknowledgement.")
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

/// Build a `RiskAcknowledgement` record from quiz answers + a clock
/// callback. Exposed for tests so we can pin determinism without
/// touching the system clock.
pub fn record_acknowledgement(
    answers: &QuizAnswers,
    iso_timestamp_utc: String,
) -> RiskAcknowledgement {
    let mut hasher_input = String::new();
    for (q, pick) in WIZARD_DEFAULT_QUIZ_QUESTIONS.iter().zip(answers.picks.iter()) {
        hasher_input.push_str(q.id);
        hasher_input.push(':');
        if let Some(idx) = pick {
            hasher_input.push_str(&idx.to_string());
        } else {
            hasher_input.push('_');
        }
        hasher_input.push(';');
    }
    hasher_input.push_str(&iso_timestamp_utc);

    RiskAcknowledgement {
        // The wizard skeleton uses a hex digest of a stable hasher;
        // the real implementation should use SHA-256 via the
        // `sha2` crate. Spec §5 names SHA-256 explicitly;
        // TODO(wizard-sha256-hasher) wires `sha2::Sha256`.
        answers_sha256: format!("placeholder-{:x}", djb2(&hasher_input)),
        timestamp_utc: iso_timestamp_utc,
        quiz_version: WIZARD_DEFAULT_QUIZ_VERSION,
        correct_count: answers.correct_count(),
    }
}

fn djb2(bytes: &str) -> u64 {
    let mut hash: u64 = 5381;
    for c in bytes.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u64);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::wizard::{StepResult, WizardController, WizardState};

    #[test]
    fn quiz_has_five_questions_each_with_four_options() {
        assert_eq!(WIZARD_DEFAULT_QUIZ_QUESTIONS.len(), 5);
        for q in WIZARD_DEFAULT_QUIZ_QUESTIONS {
            assert_eq!(q.options.len(), 4, "question {} should have 4 options", q.id);
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
}
