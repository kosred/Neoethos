use crate::workspace::WorkspaceTab;
use egui_dock::{DockState, NodeIndex};

pub struct WorkspaceState {
    dock_state: DockState<WorkspaceTab>,
}

impl WorkspaceState {
    pub fn new() -> Self {
        let mut dock_state = DockState::new(vec![WorkspaceTab::Chart]);
        let surface = dock_state.main_surface_mut();

        // Left panel: core trading monitors only — nav dropdown handles the rest
        let [chart_node, _left_node] = surface.split_left(
            NodeIndex::root(),
            0.18,
            vec![WorkspaceTab::Watchlist, WorkspaceTab::Dashboard],
        );

        // Right: execution on top, system/setup panels below
        let [chart_node, right_top_node] = surface.split_right(
            chart_node,
            0.26,
            vec![WorkspaceTab::Execution, WorkspaceTab::News],
        );
        let _ = surface.split_below(
            right_top_node,
            0.50,
            vec![
                WorkspaceTab::BrokerSetup,
                WorkspaceTab::Runtime,
                WorkspaceTab::Intelligence,
                WorkspaceTab::DataBootstrap,
                WorkspaceTab::Hardware,
                WorkspaceTab::Risk,
                WorkspaceTab::Settings,
            ],
        );

        // Bottom of chart: job monitors
        let _ = surface.split_below(
            chart_node,
            0.22,
            vec![
                WorkspaceTab::BottomStrip,
                WorkspaceTab::Discovery,
                WorkspaceTab::Training,
            ],
        );

        Self { dock_state }
    }

    pub fn dock_state_mut(&mut self) -> &mut DockState<WorkspaceTab> {
        &mut self.dock_state
    }

    pub fn focus_tab(&mut self, tab: WorkspaceTab) {
        if let Some((surface_index, node_index, tab_index)) = self.dock_state.find_tab(&tab) {
            self.dock_state
                .set_active_tab((surface_index, node_index, tab_index));
            self.dock_state
                .set_focused_node_and_surface((surface_index, node_index));
        }
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

        // Trading monitors (left panel)
        assert!(tabs.contains(&"Markets".to_string()));
        assert!(tabs.contains(&"Dashboard".to_string()));
        // Chart (center)
        assert!(tabs.contains(&"Chart".to_string()));
        // Job monitors (center-bottom)
        assert!(tabs.contains(&"Trade Watch".to_string()));
        assert!(tabs.contains(&"Discovery".to_string()));
        assert!(tabs.contains(&"Training".to_string()));
        // Execution (right-top)
        assert!(tabs.contains(&"Order Ticket".to_string()));
        assert!(tabs.contains(&"News".to_string()));
        // System panels (right-bottom)
        assert!(tabs.contains(&"Broker Setup".to_string()));
        assert!(tabs.contains(&"Runtime".to_string()));
        assert!(tabs.contains(&"Intelligence".to_string()));
        assert!(tabs.contains(&"Data Bootstrap".to_string()));
        assert!(tabs.contains(&"Hardware".to_string()));
        assert!(tabs.contains(&"Risk Settings".to_string()));
        assert!(tabs.contains(&"Settings".to_string()));
    }

    #[test]
    fn default_workspace_keeps_chart_as_main_center_tab() {
        let workspace = WorkspaceState::default();

        assert_eq!(workspace.main_tab_title(), "Chart");
    }

    #[test]
    fn workspace_can_focus_existing_tab() {
        let mut workspace = WorkspaceState::default();

        workspace.focus_tab(WorkspaceTab::Settings);

        assert!(
            workspace
                .dock_state
                .find_main_surface_tab(&WorkspaceTab::Settings)
                .is_some()
        );
    }
}
