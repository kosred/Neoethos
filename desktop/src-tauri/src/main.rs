// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    if std::env::args().any(|argument| argument == "--startup-diagnostics") {
        app_lib::startup_diagnostics();
    }
    app_lib::run();
}
