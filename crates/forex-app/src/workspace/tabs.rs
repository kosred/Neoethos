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
    SystemStatus,
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
            Self::SystemStatus => "System Status",
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
        assert_eq!(WorkspaceTab::SystemStatus.title(), "System Status");
        assert_eq!(WorkspaceTab::Hardware.title(), "Hardware");
        assert_eq!(WorkspaceTab::Risk.title(), "Risk Settings");
    }
}
