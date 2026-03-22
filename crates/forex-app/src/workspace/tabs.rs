#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkspaceTab {
    Chart,
    Watchlist,
    Execution,
    News,
    BottomStrip,
    Discovery,
    Training,
    System,
}

impl WorkspaceTab {
    pub fn title(self) -> &'static str {
        match self {
            Self::Chart => "Chart",
            Self::Watchlist => "Watchlist",
            Self::Execution => "Execution",
            Self::News => "News",
            Self::BottomStrip => "Bottom Strip",
            Self::Discovery => "Discovery",
            Self::Training => "Training",
            Self::System => "System",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_tab_labels_are_stable() {
        assert_eq!(WorkspaceTab::Chart.title(), "Chart");
        assert_eq!(WorkspaceTab::Watchlist.title(), "Watchlist");
        assert_eq!(WorkspaceTab::Execution.title(), "Execution");
        assert_eq!(WorkspaceTab::News.title(), "News");
        assert_eq!(WorkspaceTab::BottomStrip.title(), "Bottom Strip");
        assert_eq!(WorkspaceTab::Discovery.title(), "Discovery");
        assert_eq!(WorkspaceTab::Training.title(), "Training");
        assert_eq!(WorkspaceTab::System.title(), "System");
    }
}
