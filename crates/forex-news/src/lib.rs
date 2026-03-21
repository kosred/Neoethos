pub mod openai;
pub mod perplexity;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewsEvent {
    pub id: String,
    pub title: String,
    pub body: String,
    pub timestamp: DateTime<Utc>,
    pub source: String,
    pub symbol: Option<String>,
    pub tier: i32, // 1 = High impact, 2 = Medium, 3 = Low
    pub sentiment: f64, // -1.0 to 1.0
}
