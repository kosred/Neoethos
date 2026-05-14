use crate::app_services::trading::TradingSession;
use crate::app_state::AppState;
use crate::ui::components::{
    DashboardCard, render_report, render_status_badge, render_summary_cards, render_view_header,
};
use crate::ui::system::shared::{failed_bootstrap_snapshot, parse_bootstrap_list};
use crate::ui::theme;
use eframe::egui;

/// In-memory cache of the most recent `DatasetDiscovery` run so the
/// preview survives across egui frames without re-walking the disk on
/// every redraw. Held in a `OnceLock<Mutex<_>>` because egui handlers
/// are called from a single thread but we keep the lock pattern for
/// safety if the page is ever extracted into a worker.
static DISCOVERY_CACHE: std::sync::OnceLock<
    std::sync::Mutex<Option<(std::path::PathBuf, forex_data::DatasetDiscovery)>>,
> = std::sync::OnceLock::new();

fn discovery_cache(
) -> &'static std::sync::Mutex<Option<(std::path::PathBuf, forex_data::DatasetDiscovery)>> {
    DISCOVERY_CACHE.get_or_init(|| std::sync::Mutex::new(None))
}

pub fn render(
    ui: &mut egui::Ui,
    state: &mut AppState,
    session: &mut TradingSession,
    tx: &tokio::sync::mpsc::Sender<crate::app_services::ServiceEvent>,
) {
    egui::ScrollArea::vertical().show(ui, |ui| {
        let summary_cards = vec![
            DashboardCard {
                label: "Data Root".to_string(),
                value: state.runtime.data_dir.display().to_string(),
            },
            DashboardCard {
                label: "Pairs".to_string(),
                value: state.bootstrap_form.pairs_input.clone(),
            },
            DashboardCard {
                label: "Timeframes".to_string(),
                value: state.bootstrap_form.timeframes_input.clone(),
            },
            DashboardCard {
                label: "Years".to_string(),
                value: state.bootstrap_form.years.to_string(),
            },
        ];

        render_view_header(
            ui,
            "Data Bootstrap",
            "Prepare local vortex data without mixing broker setup or intelligence controls into the same page.",
        );
        ui.separator();
        render_summary_cards(ui, "Bootstrap Snapshot", &summary_cards);

        ui.add_space(10.0);
        theme::section_frame(ui.style()).show(ui, |ui| {
            render_status_badge(ui, "Bootstrap", state.bootstrap_job.as_ref());
            ui.add_space(8.0);
            ui.strong("Historical Data Import");
            ui.label(
                "Fetch missing historical bars into the local vortex cache for research and training.",
            );

            // ── Data folder + browse + discovery preview (2026-05-14) ──
            //
            // Operator concern: "Ένα θέμα του UI και του cli είναι ότι
            // δεν βοηθά τον χρήστη να πλοηγηθεί στα αρχεία και να δώσει
            // πιθανό φάκελο που υπάρχουν υποφακέλους με τα δεδομένα."
            //
            // We surface a native folder picker so the user does not
            // have to hand-type the data-root path, then run
            // `DatasetDiscovery::scan` against the chosen folder and
            // show a compact inline summary (file count by format,
            // symbol count, timeframes, skipped count) so they can
            // confirm before kicking off the import.
            let mut data_dir_text = state.runtime.data_dir.display().to_string();
            ui.horizontal(|ui| {
                ui.label("Data Folder");
                let edit = ui.text_edit_singleline(&mut data_dir_text);
                if edit.changed() {
                    state.runtime.data_dir = std::path::PathBuf::from(&data_dir_text);
                }
                if ui.button("Browse…").clicked() {
                    let start_dir = if state.runtime.data_dir.is_dir() {
                        state.runtime.data_dir.clone()
                    } else {
                        std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
                    };
                    if let Some(folder) = rfd::FileDialog::new()
                        .set_title("Select data folder")
                        .set_directory(&start_dir)
                        .pick_folder()
                    {
                        state.runtime.data_dir = folder.clone();
                        // Eagerly run discovery so the preview shows
                        // immediately after the picker closes.
                        if let Ok(report) = forex_data::DatasetDiscovery::scan(&folder) {
                            if let Ok(mut cache) = discovery_cache().lock() {
                                *cache = Some((folder, report));
                            }
                        }
                    }
                }
                if ui.button("Scan").clicked()
                    && let Ok(report) =
                        forex_data::DatasetDiscovery::scan(&state.runtime.data_dir)
                {
                    if let Ok(mut cache) = discovery_cache().lock() {
                        *cache = Some((state.runtime.data_dir.clone(), report));
                    }
                }
            });

            // Inline discovery preview: a 3–4 row compact table.
            if let Ok(guard) = discovery_cache().lock() {
                if let Some((scanned_root, report)) = guard.as_ref() {
                    if *scanned_root == state.runtime.data_dir {
                        ui.add_space(4.0);
                        theme::section_frame(ui.style()).show(ui, |ui| {
                            ui.strong("Dataset Discovery");
                            if report.is_empty() && report.skipped.is_empty() {
                                ui.colored_label(
                                    egui::Color32::from_rgb(220, 120, 60),
                                    format!(
                                        "no data files found at depth ≤ {}",
                                        forex_data::MAX_WALK_DEPTH
                                    ),
                                );
                            } else {
                                egui::Grid::new("bootstrap_discovery_grid")
                                    .num_columns(2)
                                    .spacing([16.0, 4.0])
                                    .show(ui, |ui| {
                                        let formats: Vec<String> = report
                                            .format_counts()
                                            .into_iter()
                                            .map(|(f, n)| format!("{}: {}", f.as_str(), n))
                                            .collect();
                                        ui.label("Files");
                                        ui.label(format!(
                                            "{}  ({})",
                                            report.entries.len(),
                                            formats.join(", ")
                                        ));
                                        ui.end_row();

                                        let symbols = report.symbols();
                                        ui.label("Symbols");
                                        ui.label(format!(
                                            "{}  ({})",
                                            symbols.len(),
                                            if symbols.len() > 6 {
                                                format!(
                                                    "{}, …",
                                                    symbols
                                                        .iter()
                                                        .take(6)
                                                        .cloned()
                                                        .collect::<Vec<_>>()
                                                        .join(", ")
                                                )
                                            } else {
                                                symbols.join(", ")
                                            }
                                        ));
                                        ui.end_row();

                                        ui.label("Timeframes");
                                        ui.label(report.timeframes().join(", "));
                                        ui.end_row();

                                        if !report.skipped.is_empty() {
                                            let summary: Vec<String> = report
                                                .skip_counts_by_category()
                                                .into_iter()
                                                .map(|(cat, n)| format!("{n} {cat}"))
                                                .collect();
                                            ui.label("Skipped");
                                            // Warning chip when any file
                                            // is skipped — operator must
                                            // see this before confirming.
                                            ui.colored_label(
                                                egui::Color32::from_rgb(220, 160, 40),
                                                format!(
                                                    "{}  ({})",
                                                    report.skipped.len(),
                                                    summary.join("; ")
                                                ),
                                            );
                                            ui.end_row();
                                        }
                                    });
                            }
                        });
                    }
                }
            }
            ui.add_space(6.0);

            ui.horizontal(|ui| {
                ui.label("Pairs");
                ui.text_edit_singleline(&mut state.bootstrap_form.pairs_input);
            });
            ui.horizontal(|ui| {
                ui.label("Timeframes");
                ui.text_edit_singleline(&mut state.bootstrap_form.timeframes_input);
            });
            ui.horizontal(|ui| {
                ui.label("Years");
                ui.add(egui::DragValue::new(&mut state.bootstrap_form.years).range(1..=25));
            });

            if ui.button("Start cTrader Bootstrap").clicked() {
                let symbols = parse_bootstrap_list(&state.bootstrap_form.pairs_input);
                let timeframes = parse_bootstrap_list(&state.bootstrap_form.timeframes_input);
                if symbols.is_empty() || timeframes.is_empty() {
                    state.bootstrap_job = Some(failed_bootstrap_snapshot(anyhow::anyhow!(
                        "bootstrap requires at least one symbol and one timeframe"
                    )));
                } else {
                    if let Err(err) = session.start_ctrader_bootstrap_batch(
                        state.runtime.data_dir.clone(),
                        symbols,
                        timeframes,
                        state.bootstrap_form.years,
                        tx.clone(),
                    ) {
                        let msg = format!("Bootstrap launch failed: {err}");
                        tracing::warn!(target: "forex_app::ui::system::bootstrap", "{}", msg);
                        state.bootstrap_job =
                            Some(failed_bootstrap_snapshot(anyhow::anyhow!(msg)));
                    }
                }
            }
        });

        if let Some(snapshot) = state.bootstrap_job.as_ref() {
            ui.add_space(10.0);
            theme::section_frame(ui.style()).show(ui, |ui| {
                render_report(ui, snapshot);
            });
        }
    });
}
