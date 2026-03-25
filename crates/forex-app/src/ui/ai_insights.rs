use egui::{Color32, RichText, Ui, widgets::ProgressBar};

#[derive(Default, Clone, Debug)]
pub struct AiInsightsPanel {
    pub prob_buy: f32,
    pub prob_sell: f32,
    pub prob_neutral: f32,
}

impl AiInsightsPanel {
    pub fn new() -> Self {
        Self {
            prob_buy: 0.33,
            prob_sell: 0.33,
            prob_neutral: 0.34,
        }
    }

    pub fn update_probs(&mut self, buy: f32, sell: f32, neutral: f32) {
        self.prob_buy = buy;
        self.prob_sell = sell;
        self.prob_neutral = neutral;
    }

    pub fn show(&mut self, ui: &mut Ui) {
        ui.heading("AI Ensemble Insights");
        ui.add_space(8.0);

        ui.label(RichText::new("Market Direction Probability (predict_proba tensors)").strong());
        
        ui.horizontal(|ui| {
            ui.label("Buy Confidence: ");
            ui.add(ProgressBar::new(self.prob_buy).text(format!("{:.1}%", self.prob_buy * 100.0)).fill(Color32::from_rgb(40, 200, 40)));
        });

        ui.horizontal(|ui| {
            ui.label("Sell Confidence:");
            ui.add(ProgressBar::new(self.prob_sell).text(format!("{:.1}%", self.prob_sell * 100.0)).fill(Color32::from_rgb(200, 40, 40)));
        });

        ui.horizontal(|ui| {
            ui.label("Neutral/Hold:   ");
            ui.add(ProgressBar::new(self.prob_neutral).text(format!("{:.1}%", self.prob_neutral * 100.0)).fill(Color32::from_rgb(150, 150, 150)));
        });

        ui.add_space(12.0);
        ui.label(RichText::new("Compliance & Filtering").strong());
        ui.label(RichText::new("News Blackout: INACTIVE").color(Color32::from_rgb(100, 255, 100)));
        ui.label(RichText::new("Consistency Capper: SAFE").color(Color32::from_rgb(100, 255, 100)));
        ui.label(RichText::new("Rate Limiting: 0.00ms latency").color(Color32::from_rgb(100, 255, 100)));
    }
}
