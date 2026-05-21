use crate::app_services::{
    ServiceEvent, discovery::DiscoveryJobHandle, trading::TradingSession,
    training::TrainingJobHandle,
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
    pub tx: &'a mpsc::Sender<ServiceEvent>,
    pub discovery_handle: &'a mut Option<DiscoveryJobHandle>,
    pub training_handle: &'a mut Option<TrainingJobHandle>,
    refresh_requested: bool,
}

impl<'a> WorkspaceViewer<'a> {
    pub fn new(
        state: &'a mut AppState,
        trading_session: &'a mut TradingSession,
        tx: &'a mpsc::Sender<ServiceEvent>,
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
            WorkspaceTab::Dashboard => {
                let auto_trade = self.state.auto_trade_enabled;
                let balance = self.state.account_balance;
                let equity = self.state.account_equity;
                let discovery_snap = self.state.discovery_job.clone();
                let training_snap = self.state.training_job.clone();
                let ai_snap = self.state.ai_insights_panel.clone();
                let panel = &mut self.state.dashboard_panel;
                panel.show(
                    ui,
                    ui::dashboard::DashboardInputs {
                        auto_trade_enabled: auto_trade,
                        account_balance: balance,
                        account_equity: equity,
                        discovery_job: discovery_snap.as_ref(),
                        training_job: training_snap.as_ref(),
                        ai_insights: &ai_snap,
                    },
                );
            }
            WorkspaceTab::Chart => {
                ui::trading::chart_panel::render(ui, self.state, self.trading_session, self.tx)
            }
            WorkspaceTab::Watchlist => {
                ui::trading::watchlist_panel::render(
                    ui,
                    self.state,
                    self.trading_session,
                    self.tx,
                );
            }
            WorkspaceTab::Execution => {
                ui::trading::execution_panel::render(ui, self.state, self.trading_session, self.tx)
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
            WorkspaceTab::Runtime => {
                let refresh = ui::system::runtime::render(ui, self.state, self.trading_session);
                if refresh {
                    self.refresh_requested = true;
                }
            }
            WorkspaceTab::BrokerSetup => {
                ui::system::brokers::render(ui, self.state, self.trading_session, self.tx);
            }
            WorkspaceTab::Intelligence => {
                ui::system::intelligence::render(ui, self.state, self.tx);
            }
            WorkspaceTab::AiHelper => {
                ui::ai_helper::render(ui, &mut self.state.ai_helper_panel);
            }
            WorkspaceTab::DataBootstrap => {
                ui::system::bootstrap::render(ui, self.state, self.trading_session, self.tx);
            }
            WorkspaceTab::Hardware => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui::hardware::render(ui, &mut self.state.hardware);
                });
            }
            WorkspaceTab::Risk => {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui::risk::render(ui, &mut self.state.risk);
                });
            }
            WorkspaceTab::Settings => {
                ui::settings::render(ui, self.state, self.tx);
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
    let style = trading_dock_style(ui.style().as_ref());
    // egui_dock 0.16 deprecation: `show_window_close_buttons` /
    // `show_window_collapse_buttons` are scheduled to be renamed in
    // egui_dock 0.17; the suggested replacements
    // (`show_leaf_close_buttons` / `show_leaf_collapse_buttons`)
    // don't exist in 0.16 yet. Keep the deprecated builder methods
    // until Task #65 (workspace dep upgrade) lands.
    #[allow(deprecated)]
    DockArea::new(workspace.dock_state_mut())
        .style(style)
        .show_add_buttons(false)
        .show_close_buttons(false)
        .show_window_close_buttons(false)
        .show_window_collapse_buttons(false)
        .show_leaf_close_all_buttons(false)
        .show_leaf_collapse_buttons(false)
        .show_inside(ui, viewer);
}

fn trading_dock_style(style: &egui::Style) -> Style {
    let mut dock_style = Style::from_egui(style);
    dock_style.dock_area_padding = Some(egui::Margin::same(0));
    dock_style.main_surface_border_stroke = egui::Stroke::new(1.0, ui::theme::BORDER);
    dock_style.main_surface_border_rounding = egui::CornerRadius::same(ui::theme::RADIUS_SM);
    dock_style.separator.width = 1.0;
    dock_style.separator.extra_interact_width = 4.0;
    dock_style.separator.color_idle = ui::theme::BORDER;
    dock_style.separator.color_hovered = ui::theme::ACCENT.linear_multiply(0.65);
    dock_style.separator.color_dragged = ui::theme::ACCENT;

    // The dock's per-panel tab strip is now a quiet header — the
    // sidebar is the primary nav, so this strip should sit lightly on
    // top of each panel without competing for attention.
    dock_style.tab_bar.bg_fill = ui::theme::PANEL_BG;
    dock_style.tab_bar.height = 22.0;
    dock_style.tab_bar.corner_radius = egui::CornerRadius::ZERO;
    dock_style.tab_bar.hline_color = ui::theme::BORDER;
    // Task #76 — was `false`, so when the System group docked 6+ tabs
    // into the bottom-right pane the labels got truncated to single
    // glyphs and operators couldn't tell which tab they were on. Flip
    // to `true` so overflowing tab strips become horizontally
    // scrollable instead of squashed.
    dock_style.tab_bar.show_scroll_bar_on_overflow = true;

    dock_style.tab.minimum_width = Some(60.0);
    dock_style.tab.tab_body.bg_fill = ui::theme::PANEL_BG;
    dock_style.tab.tab_body.stroke = egui::Stroke::new(1.0, ui::theme::BORDER);
    dock_style.tab.tab_body.corner_radius = egui::CornerRadius::same(ui::theme::RADIUS_SM);
    dock_style.tab.tab_body.inner_margin = egui::Margin::same(ui::theme::SPACE_SM as i8);

    // Inactive tabs are very quiet, hovered lifts gently, the active
    // tab gets the violet accent so the operator can still see what
    // panel is in focus.
    dock_style.tab.active.bg_fill = ui::theme::SURFACE_BG;
    dock_style.tab.active.text_color = ui::theme::TEXT_PRIMARY;
    dock_style.tab.active.outline_color = ui::theme::BORDER;
    dock_style.tab.inactive.bg_fill = ui::theme::PANEL_BG;
    dock_style.tab.inactive.text_color = ui::theme::TEXT_FAINT;
    dock_style.tab.inactive.outline_color = egui::Color32::TRANSPARENT;
    dock_style.tab.hovered.bg_fill = ui::theme::SURFACE_BG;
    dock_style.tab.hovered.text_color = ui::theme::TEXT_PRIMARY;
    dock_style.tab.focused.bg_fill = ui::theme::SURFACE_BG;
    dock_style.tab.focused.text_color = ui::theme::ACCENT;
    dock_style.tab.focused.outline_color = ui::theme::ACCENT;

    dock_style
}
