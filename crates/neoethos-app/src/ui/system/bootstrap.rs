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
    std::sync::Mutex<Option<(std::path::PathBuf, neoethos_data::DatasetDiscovery)>>,
> = std::sync::OnceLock::new();

fn discovery_cache()
-> &'static std::sync::Mutex<Option<(std::path::PathBuf, neoethos_data::DatasetDiscovery)>> {
    DISCOVERY_CACHE.get_or_init(|| std::sync::Mutex::new(None))
}

/// Task #9 — per-source `DatasetDiscovery` cache for the
/// `external_sources` list. Keyed by `PathBuf` so each source's
/// preview is independent (an MT4 dump that has 12 CSVs and a
/// Parquet archive that has 400 files don't have to be re-walked
/// together every time the user adds a new source). A
/// `HashMap<PathBuf, DatasetDiscovery>` keeps memory bounded —
/// scan results are cheap (just file metadata + classification).
static EXTERNAL_SOURCE_CACHE: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, neoethos_data::DatasetDiscovery>>,
> = std::sync::OnceLock::new();

fn external_source_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<std::path::PathBuf, neoethos_data::DatasetDiscovery>,
> {
    EXTERNAL_SOURCE_CACHE
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
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
                        if let Ok(report) = neoethos_data::DatasetDiscovery::scan(&folder) {
                            if let Ok(mut cache) = discovery_cache().lock() {
                                *cache = Some((folder, report));
                            }
                        }
                    }
                }
                if ui.button("Scan").clicked()
                    && let Ok(report) =
                        neoethos_data::DatasetDiscovery::scan(&state.runtime.data_dir)
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
                                        neoethos_data::MAX_WALK_DEPTH
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

                                // Task #74 — show ACTUAL skipped paths in a
                                // collapsible section, grouped by reason
                                // category, with a per-bucket sample cap so
                                // a 1000-file dataset doesn't blow up the
                                // panel. Operators previously saw only
                                // "Skipped 229 (89 unknown_extension; 140
                                // unsupported_timeframe)" without any way
                                // to learn WHICH files were missed.
                                if !report.skipped.is_empty() {
                                    egui::CollapsingHeader::new(
                                        format!(
                                            "Why were {} files skipped?",
                                            report.skipped.len()
                                        ),
                                    )
                                    .id_salt("bootstrap_skipped_details")
                                    .default_open(false)
                                    .show(ui, |ui| {
                                        const SAMPLE_PER_BUCKET: usize = 15;
                                        use std::collections::BTreeMap;
                                        let mut buckets: BTreeMap<&'static str, Vec<&neoethos_data::SkippedFile>> = BTreeMap::new();
                                        for entry in &report.skipped {
                                            buckets
                                                .entry(entry.reason.category())
                                                .or_default()
                                                .push(entry);
                                        }
                                        for (category, files) in buckets {
                                            ui.label(
                                                egui::RichText::new(format!(
                                                    "▸ {} ({})",
                                                    category,
                                                    files.len()
                                                ))
                                                .strong(),
                                            );
                                            for skipped in files.iter().take(SAMPLE_PER_BUCKET) {
                                                let detail = match &skipped.reason {
                                                    neoethos_data::SkipReason::UnknownExtension(ext) => format!("ext={ext}"),
                                                    neoethos_data::SkipReason::UnsupportedTimeframe(tf) => format!("tf={tf}"),
                                                    neoethos_data::SkipReason::TooLarge(n) => format!("size={n}B"),
                                                    neoethos_data::SkipReason::Unreadable(err) => format!("read err: {err}"),
                                                };
                                                ui.small(format!(
                                                    "    {}    [{detail}]",
                                                    skipped.path.display(),
                                                ));
                                            }
                                            if files.len() > SAMPLE_PER_BUCKET {
                                                ui.small(
                                                    egui::RichText::new(format!(
                                                        "    … and {} more in this bucket",
                                                        files.len() - SAMPLE_PER_BUCKET
                                                    ))
                                                    .italics(),
                                                );
                                            }
                                        }
                                    });
                                }
                            }
                        });
                    }
                }
            }
            ui.add_space(6.0);

            // ── Task #9: additional source folders ────────────────
            // Operator can list more than one source folder (e.g. an
            // MT4 history dump + a Spotware Parquet archive). Each
            // source gets its own format-auto-detected discovery
            // preview so the operator can confirm before importing.
            theme::section_frame(ui.style()).show(ui, |ui| {
                ui.strong("Additional Source Folders");
                ui.label(
                    egui::RichText::new(
                        "Add MT4 exports, Spotware Parquet dumps, CSV archives, etc. Each \
                         folder is scanned independently — format is auto-detected per file.",
                    )
                    .small()
                    .color(theme::TEXT_MUTED),
                );
                ui.add_space(4.0);
                let mut to_remove: Option<usize> = None;
                let mut to_rescan: Option<usize> = None;
                for (idx, source) in state.bootstrap_form.external_sources.iter().enumerate() {
                    ui.horizontal(|ui| {
                        ui.label(format!("{}.", idx + 1));
                        ui.label(
                            egui::RichText::new(source.display().to_string())
                                .monospace()
                                .small(),
                        );
                        if ui.small_button("⟳ Rescan").clicked() {
                            to_rescan = Some(idx);
                        }
                        if ui.small_button("✕ Remove").clicked() {
                            to_remove = Some(idx);
                        }
                    });
                    // Per-source compact summary (auto-detected formats).
                    if let Ok(cache) = external_source_cache().lock()
                        && let Some(report) = cache.get(source)
                    {
                        let formats: Vec<String> = report
                            .format_counts()
                            .into_iter()
                            .map(|(f, n)| format!("{}:{}", f.as_str(), n))
                            .collect();
                        ui.label(
                            egui::RichText::new(format!(
                                "    {} files · formats: {} · skipped: {}",
                                report.entries.len(),
                                if formats.is_empty() {
                                    "(none detected)".to_string()
                                } else {
                                    formats.join(", ")
                                },
                                report.skipped.len(),
                            ))
                            .small()
                            .color(theme::TEXT_MUTED),
                        );
                    }
                }
                if let Some(idx) = to_remove {
                    let removed = state.bootstrap_form.external_sources.remove(idx);
                    if let Ok(mut cache) = external_source_cache().lock() {
                        cache.remove(&removed);
                    }
                }
                if let Some(idx) = to_rescan
                    && let Some(source) = state.bootstrap_form.external_sources.get(idx).cloned()
                    && let Ok(report) = neoethos_data::DatasetDiscovery::scan(&source)
                    && let Ok(mut cache) = external_source_cache().lock()
                {
                    cache.insert(source, report);
                }
                ui.add_space(2.0);
                if ui.button("➕ Add source folder").clicked() {
                    let start_dir = state
                        .bootstrap_form
                        .external_sources
                        .last()
                        .filter(|p| p.is_dir())
                        .cloned()
                        .or_else(|| {
                            state
                                .runtime
                                .data_dir
                                .parent()
                                .map(std::path::Path::to_path_buf)
                        })
                        .unwrap_or_else(|| {
                            std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
                        });
                    if let Some(folder) = rfd::FileDialog::new()
                        .set_title("Select additional source folder")
                        .set_directory(&start_dir)
                        .pick_folder()
                    {
                        // De-dupe: don't add the same path twice, and
                        // don't add the primary data_dir as a source.
                        if folder == state.runtime.data_dir
                            || state.bootstrap_form.external_sources.contains(&folder)
                        {
                            tracing::info!(
                                target: "neoethos_app::bootstrap::multi_source",
                                path = %folder.display(),
                                "operator picked an already-listed source; ignoring duplicate"
                            );
                        } else {
                            // Eagerly scan so the preview shows when the
                            // picker closes (same UX as the primary
                            // data_dir Browse… button).
                            if let Ok(report) = neoethos_data::DatasetDiscovery::scan(&folder)
                                && let Ok(mut cache) = external_source_cache().lock()
                            {
                                cache.insert(folder.clone(), report);
                            }
                            state.bootstrap_form.external_sources.push(folder);
                        }
                    }
                }
            });

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
                        tracing::warn!(target: "neoethos_app::ui::system::bootstrap", "{}", msg);
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
