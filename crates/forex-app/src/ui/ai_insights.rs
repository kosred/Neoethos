use egui::{Color32, RichText, Ui, widgets::ProgressBar};

/// Dashboard widget that surfaces the active ensemble's prediction
/// probabilities and the news / consistency / rate-limit compliance
/// status of the model runtime.
///
/// V0.4 audit Task #34 — every field intentionally defaults to `None`
/// and the renderer shows an "Unavailable" placeholder for each one
/// until a producer wires real data in. The producer is the live
/// inference loop (Task #7 — `auto_trade_producer`) which currently
/// runs only as a stub. Once that producer is alive it MUST update
/// these fields on every closed-bar prediction so the operator can see
/// the AI's actual confidence (not a default 0.0 that looks like a
/// false signal).
///
/// **Do not add a `Default` that pre-fills these to dummy values.**
/// The whole point of `Option<f32>` here is "we have no data" vs
/// "the data is exactly 0%". A hardcoded default would erase that
/// distinction and let stale defaults look like real signals.
#[derive(Default, Clone, Debug)]
pub struct AiInsightsPanel {
    pub prob_buy: Option<f32>,
    pub prob_sell: Option<f32>,
    pub prob_neutral: Option<f32>,
    pub news_blackout_active: Option<bool>,
    pub consistency_status: Option<String>,
    pub rate_limit_latency_ms: Option<f32>,
}

impl AiInsightsPanel {
    pub fn new() -> Self {
        Self {
            prob_buy: None,
            prob_sell: None,
            prob_neutral: None,
            news_blackout_active: None,
            consistency_status: None,
            rate_limit_latency_ms: None,
        }
    }

    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading("AI Ensemble Insights");
        ui.add_space(8.0);

        ui.label(RichText::new("Market Direction Probability (predict_proba tensors)").strong());
        if let (Some(prob_buy), Some(prob_sell), Some(prob_neutral)) =
            (self.prob_buy, self.prob_sell, self.prob_neutral)
        {
            ui.horizontal(|ui| {
                ui.label("Buy Confidence: ");
                ui.add(
                    ProgressBar::new(prob_buy)
                        .text(format!("{:.1}%", prob_buy * 100.0))
                        .fill(Color32::from_rgb(40, 200, 40)),
                );
            });

            ui.horizontal(|ui| {
                ui.label("Sell Confidence:");
                ui.add(
                    ProgressBar::new(prob_sell)
                        .text(format!("{:.1}%", prob_sell * 100.0))
                        .fill(Color32::from_rgb(200, 40, 40)),
                );
            });

            ui.horizontal(|ui| {
                ui.label("Neutral/Hold:   ");
                ui.add(
                    ProgressBar::new(prob_neutral)
                        .text(format!("{:.1}%", prob_neutral * 100.0))
                        .fill(Color32::from_rgb(150, 150, 150)),
                );
            });
        } else {
            ui.label(
                RichText::new(
                    "Prediction probabilities are not available from the active model runtime yet.",
                )
                .color(Color32::from_rgb(180, 180, 180)),
            );
        }

        ui.add_space(12.0);
        ui.label(RichText::new("Compliance & Filtering").strong());
        match self.news_blackout_active {
            Some(true) => ui.label(
                RichText::new("News Blackout: ACTIVE").color(Color32::from_rgb(255, 120, 120)),
            ),
            Some(false) => ui.label(
                RichText::new("News Blackout: SAFE").color(Color32::from_rgb(100, 255, 100)),
            ),
            None => ui.label(
                RichText::new("News Blackout: Unavailable").color(Color32::from_rgb(180, 180, 180)),
            ),
        };
        match &self.consistency_status {
            Some(status) => ui.label(
                RichText::new(format!("Consistency Capper: {status}"))
                    .color(Color32::from_rgb(100, 255, 100)),
            ),
            None => ui.label(
                RichText::new("Consistency Capper: Unavailable")
                    .color(Color32::from_rgb(180, 180, 180)),
            ),
        };
        match self.rate_limit_latency_ms {
            Some(latency) => ui.label(
                RichText::new(format!("Rate Limiting: {:.2}ms latency", latency))
                    .color(Color32::from_rgb(100, 255, 100)),
            ),
            None => ui.label(
                RichText::new("Rate Limiting: Unavailable").color(Color32::from_rgb(180, 180, 180)),
            ),
        };
    }
}
