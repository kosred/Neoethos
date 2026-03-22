use crate::app_services::{
    discovery::DiscoveryJobHandle,
    trading::TradingSession,
    training::TrainingJobHandle,
    ServiceEvent,
};
use crate::app_state::AppState;
use crate::ui;
use crate::workspace::{WorkspaceState, WorkspaceTab};
use eframe::egui;
use egui_dock::{DockArea, Style, TabViewer};
use tokio::sync::mpsc;

pub struct WorkspaceViewer<'a> {
    pub state: &'a mut AppState,
    pub trading_session: &'a mut TradingSession,
    pub tx: &'a mpsc::UnboundedSender<ServiceEvent>,
    pub discovery_handle: &'a mut Option<DiscoveryJobHandle>,
    pub training_handle: &'a mut Option<TrainingJobHandle>,
    refresh_requested: bool,
}

impl<'a> WorkspaceViewer<'a> {
    pub fn new(
        state: &'a mut AppState,
        trading_session: &'a mut TradingSession,
        tx: &'a mpsc::UnboundedSender<ServiceEvent>,
        discovery_handle: &'a mut Option<DiscoveryJobHandle>,
        training_handle: &'a mut Option<TrainingJobHandle>,
    ) -> Self {
        Self {
            state,
            trading_session,
            tx,
            discovery_handle,
            training_handle,
            refresh_requested: false,
        }
    }

    pub fn refresh_requested(&self) -> bool {
        self.refresh_requested
    }
}

impl TabViewer for WorkspaceViewer<'_> {
    type Tab = WorkspaceTab;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            WorkspaceTab::Chart => {
                ui::trading::chart_panel::render(ui, self.state, self.trading_session)
            }
            WorkspaceTab::Watchlist => {
                ui::trading::watchlist_panel::render(ui, self.state, self.trading_session)
            }
            WorkspaceTab::Execution => {
                ui::trading::execution_panel::render(ui, self.state, self.trading_session)
            }
            WorkspaceTab::News => ui::trading::news_panel::render(ui, self.state),
            WorkspaceTab::BottomStrip => {
                ui::trading::bottom_strip::render(ui, self.state, self.trading_session)
            }
            WorkspaceTab::Discovery => {
                ui::discovery::render(ui, self.state, self.tx, self.discovery_handle)
            }
            WorkspaceTab::Training => {
                ui::training::render(ui, self.state, self.tx, self.training_handle)
            }
            WorkspaceTab::System => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let refresh = ui::system_status::render(
                        ui,
                        self.state,
                        self.trading_session.is_connected(),
                    );
                    if refresh {
                        self.refresh_requested = true;
                    }
                    ui.separator();
                    ui::hardware::render(ui, &mut self.state.hardware);
                    ui.separator();
                    ui::risk::render(ui, &mut self.state.risk);
                });
            }
        }
    }

    fn closeable(&mut self, _tab: &mut Self::Tab) -> bool {
        false
    }
}

pub fn render_workspace(
    ui: &mut egui::Ui,
    workspace: &mut WorkspaceState,
    viewer: &mut WorkspaceViewer<'_>,
) {
    let style = Style::from_egui(ui.style().as_ref());
    DockArea::new(workspace.dock_state_mut())
        .style(style)
        .show_add_buttons(false)
        .show_close_buttons(false)
        .show_inside(ui, viewer);
}
