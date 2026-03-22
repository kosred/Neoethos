use crate::workspace::WorkspaceTab;
use egui_dock::{DockState, NodeIndex};

pub struct WorkspaceState {
    dock_state: DockState<WorkspaceTab>,
}

impl WorkspaceState {
    pub fn new() -> Self {
        let mut dock_state = DockState::new(vec![WorkspaceTab::Chart]);
        let surface = dock_state.main_surface_mut();

        let [chart_node, _watchlist_node] = surface.split_left(
            NodeIndex::root(),
            0.18,
            vec![WorkspaceTab::Watchlist, WorkspaceTab::System],
        );

        let [chart_node, _right_node] = surface.split_right(
            chart_node,
            0.27,
            vec![WorkspaceTab::Execution, WorkspaceTab::News],
        );

        let _ = surface.split_below(
            chart_node,
            0.28,
            vec![
                WorkspaceTab::BottomStrip,
                WorkspaceTab::Discovery,
                WorkspaceTab::Training,
            ],
        );

        Self {
            dock_state,
        }
    }

    pub fn dock_state_mut(&mut self) -> &mut DockState<WorkspaceTab> {
        &mut self.dock_state
    }

    #[cfg(test)]
    pub fn flattened_titles(&self) -> Vec<String> {
        self.dock_state
            .iter_all_tabs()
            .map(|(_, tab)| tab.title().to_string())
            .collect()
    }

    #[cfg(test)]
    pub fn main_tab_title(&self) -> &'static str {
        if self
            .dock_state
            .find_main_surface_tab(&WorkspaceTab::Chart)
            .is_some()
        {
            WorkspaceTab::Chart.title()
        } else {
            "Unknown"
        }
    }
}

impl Default for WorkspaceState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_workspace_includes_expected_tabs() {
        let workspace = WorkspaceState::default();
        let tabs = workspace.flattened_titles();

        assert!(tabs.contains(&"Chart".to_string()));
        assert!(tabs.contains(&"Watchlist".to_string()));
        assert!(tabs.contains(&"Execution".to_string()));
        assert!(tabs.contains(&"News".to_string()));
        assert!(tabs.contains(&"Bottom Strip".to_string()));
        assert!(tabs.contains(&"Discovery".to_string()));
        assert!(tabs.contains(&"Training".to_string()));
        assert!(tabs.contains(&"System".to_string()));
    }

    #[test]
    fn default_workspace_keeps_chart_as_main_center_tab() {
        let workspace = WorkspaceState::default();

        assert_eq!(workspace.main_tab_title(), "Chart");
    }
}
