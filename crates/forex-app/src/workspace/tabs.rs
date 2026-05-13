#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkspaceTab {
    Dashboard,
    Chart,
    Watchlist,
    Execution,
    News,
    BottomStrip,
    Discovery,
    Training,
    Runtime,
    BrokerSetup,
    Intelligence,
    DataBootstrap,
    Hardware,
    Risk,
    Settings,
}

/// Top-level navigation grouping. The left sidebar uses these to bucket
/// the 15 tabs into 3 collapsible sections so the operator does not
/// have to scan a flat list mid-session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkspaceGroup {
    Trading,
    AiEngine,
    System,
}

impl WorkspaceGroup {
    pub fn title(self) -> &'static str {
        match self {
            Self::Trading => "Trading",
            Self::AiEngine => "AI Engine",
            Self::System => "System",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Trading => "Live execution, charts, order entry, news",
            Self::AiEngine => "Strategy discovery, model training, AI insights",
            Self::System => "Broker, data, hardware, risk, settings",
        }
    }

    pub fn ordered() -> &'static [Self] {
        &[Self::Trading, Self::AiEngine, Self::System]
    }
}

impl WorkspaceTab {
    pub fn title(self) -> &'static str {
        match self {
            Self::Dashboard => "Dashboard",
            Self::Chart => "Chart",
            Self::Watchlist => "Markets",
            Self::Execution => "Order Ticket",
            Self::News => "News",
            Self::BottomStrip => "Trade Watch",
            Self::Discovery => "Discovery",
            Self::Training => "Training",
            Self::Runtime => "Runtime",
            Self::BrokerSetup => "Broker Setup",
            Self::Intelligence => "Intelligence",
            Self::DataBootstrap => "Data Bootstrap",
            Self::Hardware => "Hardware",
            Self::Risk => "Risk Settings",
            Self::Settings => "Settings",
        }
    }

    /// Short hover description shown next to the title in the sidebar.
    pub fn description(self) -> &'static str {
        match self {
            Self::Dashboard => "Account equity, open positions, engine status",
            Self::Chart => "TradingView-style price chart with bid/ask",
            Self::Watchlist => "Symbol list with live quotes",
            Self::Execution => "Place / modify / cancel orders",
            Self::News => "LLM-curated news + blackout filter",
            Self::BottomStrip => "Compact trade watch strip",
            Self::Discovery => "Genetic strategy search → portfolio",
            Self::Training => "AI ensemble training pipeline",
            Self::Runtime => "Connection & session diagnostics",
            Self::BrokerSetup => "cTrader / DXTrade credentials & OAuth",
            Self::Intelligence => "AI model insights & explainability",
            Self::DataBootstrap => "Historical data download / migration",
            Self::Hardware => "CPU / GPU / RAM detection & overrides",
            Self::Risk => "Prop-firm risk rules & guard-rails",
            Self::Settings => "App-wide settings",
        }
    }

    /// Single-glyph icon for the sidebar nav item. Unicode-only (no
    /// font dependency); chosen to be visually consistent with what
    /// cTrader / TradingView use for the same concepts. Two characters
    /// wide so the icon column lines up cleanly across all rows.
    pub fn icon(self) -> &'static str {
        match self {
            Self::Dashboard => "▦",
            Self::Chart => "📈",
            Self::Watchlist => "≡",
            Self::Execution => "↹",
            Self::News => "📰",
            Self::BottomStrip => "◫",
            Self::Discovery => "✦",
            Self::Training => "⊛",
            Self::Runtime => "◉",
            Self::BrokerSetup => "🔌",
            Self::Intelligence => "✺",
            Self::DataBootstrap => "⤓",
            Self::Hardware => "▤",
            Self::Risk => "⚠",
            Self::Settings => "⚙",
        }
    }

    pub fn group(self) -> WorkspaceGroup {
        match self {
            Self::Dashboard
            | Self::Chart
            | Self::Watchlist
            | Self::Execution
            | Self::News
            | Self::BottomStrip => WorkspaceGroup::Trading,
            Self::Discovery | Self::Training | Self::Intelligence => WorkspaceGroup::AiEngine,
            Self::Runtime
            | Self::BrokerSetup
            | Self::DataBootstrap
            | Self::Hardware
            | Self::Risk
            | Self::Settings => WorkspaceGroup::System,
        }
    }

    /// All tabs in the order they appear under their group in the sidebar.
    pub fn all_for_group(group: WorkspaceGroup) -> &'static [Self] {
        match group {
            WorkspaceGroup::Trading => &[
                Self::Dashboard,
                Self::Chart,
                Self::Watchlist,
                Self::Execution,
                Self::News,
                Self::BottomStrip,
            ],
            WorkspaceGroup::AiEngine => &[Self::Discovery, Self::Training, Self::Intelligence],
            WorkspaceGroup::System => &[
                Self::Runtime,
                Self::BrokerSetup,
                Self::DataBootstrap,
                Self::Hardware,
                Self::Risk,
                Self::Settings,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_tab_labels_are_stable() {
        assert_eq!(WorkspaceTab::Dashboard.title(), "Dashboard");
        assert_eq!(WorkspaceTab::Chart.title(), "Chart");
        assert_eq!(WorkspaceTab::Watchlist.title(), "Markets");
        assert_eq!(WorkspaceTab::Execution.title(), "Order Ticket");
        assert_eq!(WorkspaceTab::News.title(), "News");
        assert_eq!(WorkspaceTab::BottomStrip.title(), "Trade Watch");
        assert_eq!(WorkspaceTab::Discovery.title(), "Discovery");
        assert_eq!(WorkspaceTab::Training.title(), "Training");
        assert_eq!(WorkspaceTab::Runtime.title(), "Runtime");
        assert_eq!(WorkspaceTab::BrokerSetup.title(), "Broker Setup");
        assert_eq!(WorkspaceTab::Intelligence.title(), "Intelligence");
        assert_eq!(WorkspaceTab::DataBootstrap.title(), "Data Bootstrap");
        assert_eq!(WorkspaceTab::Hardware.title(), "Hardware");
        assert_eq!(WorkspaceTab::Risk.title(), "Risk Settings");
        assert_eq!(WorkspaceTab::Settings.title(), "Settings");
    }
}
