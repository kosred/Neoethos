#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubsystemSection {
    System,
    App,
    Cli,
    Discovery,
    Training,
    Bindings,
}

impl SubsystemSection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "SYSTEM",
            Self::App => "APP",
            Self::Cli => "CLI",
            Self::Discovery => "DISCOVERY",
            Self::Training => "TRAINING",
            Self::Bindings => "BINDINGS",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionedRunRecord {
    pub run_id: String,
    pub parent_run_id: Option<String>,
    pub started_at: String,
    pub finished_at: String,
    pub subsystem: SubsystemSection,
    pub operation: String,
    pub status: String,
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    pub error_code: Option<String>,
    pub message: String,
    pub body: String,
}
