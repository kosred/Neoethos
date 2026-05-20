use crate::workspace::WorkspaceTab;
use egui_dock::{DockState, NodeIndex};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Note — minimal workspace persistence.
///
/// We persist the LAST ACTIVE TAB across restarts so the operator
/// reopens the app on the panel they were last using. The full dock
/// layout (panel splits, sizes, floating windows) is intentionally
/// NOT persisted in V0.4 — egui_dock's `DockState` serialisation
/// requires the `serde` feature and round-trip is fragile when the
/// tab enum changes between releases (which it has, every patch).
/// V0.5 follow-up: gate full layout persistence behind a feature flag
/// once the tab set has stabilised.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceStateFile {
    pub last_active_tab: WorkspaceTab,
}

impl WorkspaceStateFile {
    /// Path the persistence file lives at, mirroring the wizard's
    /// pattern: `<data_path>/forex-ai/workspace_state.json`. Returns
    /// `None` if the data path is empty (e.g. wizard not finished).
    pub fn path(data_path: &Path) -> Option<PathBuf> {
        if data_path.as_os_str().is_empty() {
            return None;
        }
        Some(data_path.join("forex-ai").join("workspace_state.json"))
    }

    /// Load if it exists; return `None` on missing / unreadable /
    /// corrupt files so the caller can fall back to defaults rather
    /// than crashing on a malformed state file.
    pub fn load_if_present(data_path: &Path) -> Option<Self> {
        let path = Self::path(data_path)?;
        let body = std::fs::read_to_string(&path).ok()?;
        match serde_json::from_str(&body) {
            Ok(v) => Some(v),
            Err(err) => {
                tracing::warn!(
                    target: "forex_app::workspace",
                    path = %path.display(),
                    error = %err,
                    "workspace_state.json could not be parsed; ignoring"
                );
                None
            }
        }
    }

    /// Save best-effort — log on failure but don't crash the shutdown
    /// path. Atomic write via temp + rename so a crash mid-write
    /// doesn't leave an unparseable file (load_if_present logs and
    /// ignores corruption, but atomicity avoids the corrupt state in
    /// the first place).
    pub fn save_best_effort(&self, data_path: &Path) {
        let Some(path) = Self::path(data_path) else {
            return;
        };
        if let Some(parent) = path.parent()
            && std::fs::create_dir_all(parent).is_err()
        {
            return;
        }
        let Ok(body) = serde_json::to_string_pretty(self) else {
            return;
        };
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, &body).is_err() {
            return;
        }
        if let Err(err) = std::fs::rename(&tmp, &path) {
            tracing::warn!(
                target: "forex_app::workspace",
                path = %path.display(),
                error = %err,
                "workspace_state.json atomic-rename failed; state will not persist"
            );
        }
    }
}

pub struct WorkspaceState {
    dock_state: DockState<WorkspaceTab>,
    /// Last tab the operator selected via the sidebar — used to paint
    /// the sidebar's active-row accent stripe. Falls back to the
    /// initial Chart tab so the sidebar always highlights something.
    active_tab: WorkspaceTab,
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
                // Note — AiHelper was enumerated in
                // `tabs.rs::WorkspaceTab` and listed in the sidebar's
                // `all_for_group(AiEngine)`, but missing here meant
                // clicking it in the sidebar did nothing (`find_tab`
                // returned None → silent no-op). Adding it to the
                // bottom-right system stack so it has a real home.
                WorkspaceTab::AiHelper,
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

        Self {
            dock_state,
            active_tab: WorkspaceTab::Chart,
        }
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
            self.active_tab = tab;
        }
    }

    /// Last tab the operator selected via the sidebar — used by the
    /// sidebar to paint the active-row accent stripe.
    pub fn active_tab(&self) -> Option<WorkspaceTab> {
        Some(self.active_tab)
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
        // Task #17 regression — AiHelper must be in the layout so the
        // sidebar nav can actually focus it.
        assert!(tabs.contains(&"AI Helper".to_string()));
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
