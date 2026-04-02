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
}

impl WorkspaceTab {
    pub fn title(self) -> &'static str {
        match self {
            Self::Dashboard => "Dashboard",
            Self::Chart => "Chart",
            Self::Watchlist => "Watchlist",
            Self::Execution => "Execution",
            Self::News => "News",
            Self::BottomStrip => "Bottom Strip",
            Self::Discovery => "Discovery",
            Self::Training => "Training",
            Self::Runtime => "Runtime",
            Self::BrokerSetup => "Broker Setup",
            Self::Intelligence => "Intelligence",
            Self::DataBootstrap => "Data Bootstrap",
            Self::Hardware => "Hardware",
            Self::Risk => "Risk Settings",
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
        assert_eq!(WorkspaceTab::Watchlist.title(), "Watchlist");
        assert_eq!(WorkspaceTab::Execution.title(), "Execution");
        assert_eq!(WorkspaceTab::News.title(), "News");
        assert_eq!(WorkspaceTab::BottomStrip.title(), "Bottom Strip");
        assert_eq!(WorkspaceTab::Discovery.title(), "Discovery");
        assert_eq!(WorkspaceTab::Training.title(), "Training");
        assert_eq!(WorkspaceTab::Runtime.title(), "Runtime");
        assert_eq!(WorkspaceTab::BrokerSetup.title(), "Broker Setup");
        assert_eq!(WorkspaceTab::Intelligence.title(), "Intelligence");
        assert_eq!(WorkspaceTab::DataBootstrap.title(), "Data Bootstrap");
        assert_eq!(WorkspaceTab::Hardware.title(), "Hardware");
        assert_eq!(WorkspaceTab::Risk.title(), "Risk Settings");
    }
}
