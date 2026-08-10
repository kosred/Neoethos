//! NeoEthos desktop (Tauri) — single-process Rust core + web UI.
//!
//! PoC commands read the operator's on-disk Vortex data directly through the
//! existing `neoethos-data` / `neoethos-core` crates — NO separate backend
//! process, NO HTTP, NO port 7423, NO supervisor/watchdog. The whole class of
//! Flutter+HTTP bugs (spawn spirals, /healthz timeouts, SSE EventFluxException,
//! config seeded to a different dir) cannot exist here: one process, one CWD.

use std::path::PathBuf;
use std::sync::OnceLock;

use serde::Serialize;

mod broker;

/// The ABSOLUTE `config.yaml` this process reads and writes, resolved once by
/// [`prepare_data_root`] before anything else runs.
///
/// **Audit #125 (2026-08-09).** This used to be the bare relative string
/// `"config.yaml"`, handed to `install_config_path` and to `Settings::from_yaml`.
/// A relative path resolves against the process CWD, and the CWD is only correct
/// because `prepare_data_root` chdirs into the data root — a `set_current_dir`
/// that fails is logged and then ignored (see the two `eprintln!` arms below),
/// so the app kept running and read a `config.yaml` that was not there. Combined
/// with #289 (`ModelsConfig::default()` re-arms the export gates the operator
/// deliberately disarmed on 2026-06-06) a CWD accident was a SILENT CHANGE OF
/// TRADING POLICY — and since 2026-08-09 a money gate reads its setting through
/// this same path (`server/orders.rs:63-75`, `require_stop_loss`).
///
/// The resolution is now absolute and recorded here; the load is fail-loud.
static RESOLVED_CONFIG_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Record the absolute config path for the rest of the process. First install
/// wins, mirroring `server::state::install_config_path`.
fn record_config_path(path: PathBuf) {
    // Absolutise a relative path against the CWD *now*, while the CWD is still
    // the one that made it correct. Deliberately NOT `fs::canonicalize`: on
    // Windows that returns a `\\?\` UNC-prefixed path, which is correct but
    // leaks into every log line and error message the operator reads. And the
    // file may legitimately not exist yet (a first-run seed that failed), which
    // `canonicalize` treats as an error.
    let absolute = if path.is_absolute() {
        path
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(&path))
            .unwrap_or(path)
    };
    let _ = RESOLVED_CONFIG_PATH.set(absolute);
}

/// Stop the app with a visible, actionable message instead of starting under a
/// configuration the operator never wrote.
///
/// **Audit #125/#289 — this is a DELIBERATE fail-closed.** The previous
/// behaviour (`unwrap_or_else(|_| Settings::default())`) started the app under
/// `Settings::default()`, which is a *different trading policy*: different
/// export gates (#289), different risk defaults, and — since 2026-08-09 — a
/// different answer for the `require_stop_loss` money gate that
/// `server/orders.rs:63-75` reads through this path. A silently different
/// policy is strictly worse than an app that will not start, because the
/// operator cannot see it.
///
/// Never called for a *missing* seed on first run: `prepare_data_root` seeds
/// `config.yaml` from the bundle before this point, and a seed failure is
/// itself reported here rather than swallowed.
fn fatal_config_error(what: &str, detail: &str) -> ! {
    let path = RESOLVED_CONFIG_PATH
        .get()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "<unresolved>".to_string());
    let message = format!(
        "NeoEthos cannot start: {what}.\n\n\
         Config file: {path}\n\
         Details: {detail}\n\n\
         The app refuses to start on default settings, because the defaults are \
         NOT your settings — they re-arm export gates and risk limits you \
         changed deliberately, and they would decide real trades.\n\n\
         Fix or restore that file (or delete it to be re-seeded from the \
         bundled defaults on the next launch), then start NeoEthos again."
    );
    eprintln!("FATAL: {message}");
    rfd::MessageDialog::new()
        .set_level(rfd::MessageLevel::Error)
        .set_title("NeoEthos — configuration could not be loaded")
        .set_description(message.as_str())
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
    std::process::exit(2);
}

/// In-process backend: the full neoethos-app axum API, served on an ephemeral
/// loopback port inside THIS process (a tokio task, not a separate exe). It
/// starts with the app and dies with it — no supervisor, no spawn, no fixed
/// port. The web UI reads the port via the `api_base` command and calls the
/// same ~50 handlers the old Flutter client used. One binary, one process.
mod backend {
    use std::net::TcpListener;
    use std::sync::OnceLock;

    use neoethos_app::server;

    static API_PORT: OnceLock<u16> = OnceLock::new();

    /// Bind an ephemeral loopback port *synchronously* (so the port is known
    /// before the window loads), then serve the full API on it.
    pub fn start() {
        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(l) => l,
            Err(e) => {
                eprintln!("FATAL: could not bind the in-process backend: {e}");
                return;
            }
        };
        let port = listener.local_addr().map(|a| a.port()).unwrap_or(0);
        let _ = API_PORT.set(port);
        eprintln!("in-process backend bound on 127.0.0.1:{port}");
        // Publish the ephemeral API port to a well-known file so the P2P mesh
        // sidecar can find the desktop app automatically (the port is random;
        // this is how `neoethos-mesh` locates the local /federation/* bridge).
        let _ = std::fs::write(
            std::env::temp_dir().join("neoethos_api_port"),
            port.to_string(),
        );

        // ── Config resolution + load happen HERE, synchronously, on the
        // setup thread — BEFORE the server task spawns and before any dialog
        // would be racing a window. Audit #125/#289: a config miss must stop
        // the app, not silently become a different trading policy.
        let config_path = super::RESOLVED_CONFIG_PATH.get().cloned().unwrap_or_else(|| {
            // Unreachable in `run()` (prepare_data_root always records one),
            // but a future caller must not silently inherit the CWD.
            super::fatal_config_error(
                "the data root was never established, so no config.yaml path exists",
                "internal ordering error: backend::start() ran before prepare_data_root()",
            )
        });
        eprintln!("config → {}", config_path.display());
        // Same process-wide install the CLI/main.rs perform, now with the
        // ABSOLUTE path so `/settings` GET+POST and `Settings::load` cannot
        // diverge if anything later changes the working directory.
        server::state::install_config_path(config_path.clone());
        // F-005: say which env-var overrides are live before anything
        // acts on them. The desktop shell reads the same env as the
        // headless binary — NEOETHOS_USER_DATA_DIR in particular, which
        // relocates the whole data root — so it needs the same line in
        // its log.
        neoethos_core::env_overrides::log_active_overrides_at_startup();
        // Audit S05: install EVERY runtime override from settings, exactly
        // as the headless main.rs does, via the one shared installer. The
        // desktop previously installed NONE, so config.yaml runtime knobs
        // (search population, hardware CPU budget, feature normalization,
        // tree threads, app-server runtime) were silently ignored here.
        //
        // Audit #125: this was `.unwrap_or_else(|_| Settings::default())`.
        // `Settings::default()` is NOT the shipped configuration — #289
        // records that `ModelsConfig::default()` still encodes the
        // pre-2026-06-06 posture and re-arms `require_walkforward_for_export`
        // + `prop_firm_min_pass_rate`, i.e. it changes WHICH STRATEGIES MAY
        // REACH LIVE. Swallowing the load error swapped the operator's
        // trading policy for a different one without a word. It is now fatal.
        let settings = match neoethos_core::Settings::from_yaml(&config_path) {
            Ok(s) => s,
            Err(e) => super::fatal_config_error(
                &format!("could not load {}", config_path.display()),
                &format!("{e:#}"),
            ),
        };
        neoethos_app::install_runtime_overrides_from_settings(&settings);

        tauri::async_runtime::spawn(async move {
            let state = server::state::AppApiState::new();
            server::state::install_account_refresh_trigger(state.account_refresh_tx_clone());
            server::bridge::spawn(state.clone());
            // Autonomous LLM supervisor heartbeat (no-op until enabled in the UI).
            neoethos_app::app_services::supervisor::spawn(state.clone());
            // Session-aware spread sampler (broker's real per-hour spreads).
            neoethos_app::app_services::spread_stats::spawn();
            // Auto-cull retirement → rediscovery queue drainer (Settings-gated).
            neoethos_app::app_services::rediscovery::spawn(state.clone());
            if let Err(e) = server::serve_on(listener, state).await {
                eprintln!("in-process backend exited: {e:#}");
            }
        });
    }

    pub fn base_url() -> String {
        format!("http://127.0.0.1:{}", API_PORT.get().copied().unwrap_or(0))
    }

    /// True when a Discovery/Training/Autopilot engine is currently running.
    /// Used by the window-close guard: closing the app hard-kills every
    /// worker thread ("close = stop"), which on 2026-07-20 destroyed a
    /// ~30-hour discovery run at 95.5% via an accidental close. Implemented
    /// as a tiny loopback HTTP probe of our own in-process API (std-only —
    /// no HTTP-client dependency in the shell). Fail-open: if the probe
    /// can't answer quickly, we do NOT block the close.
    pub fn engine_running() -> bool {
        use std::io::{Read, Write};
        let Some(&port) = API_PORT.get() else {
            return false;
        };
        let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
        let timeout = std::time::Duration::from_millis(700);
        let Ok(mut stream) = std::net::TcpStream::connect_timeout(&addr, timeout) else {
            return false;
        };
        let _ = stream.set_read_timeout(Some(timeout));
        let _ = stream.set_write_timeout(Some(timeout));
        let request = format!(
            "GET /engines/status HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
        );
        if stream.write_all(request.as_bytes()).is_err() {
            return false;
        }
        let mut body = String::new();
        let _ = stream.take(64 * 1024).read_to_string(&mut body);
        // {"discovery":"Running",...} / training / autoTrader — any engine.
        body.contains("\"Running")
    }
}

/// Base URL of the in-process backend (e.g. `http://127.0.0.1:54321`). The web
/// UI fetches this once at startup and uses it for every backend call.
#[tauri::command]
fn api_base() -> String {
    backend::base_url()
}

/// MCP sidecar lifecycle (task #33 — full MCP support). The isolated
/// `neoethos-mcp` binary connects to the operator's configured MCP servers
/// (cTrader remote, MT5 bridges, web search, …) and exposes their tools on
/// 127.0.0.1:7431 for the Supervisor's approval-gated actions. Best-effort:
/// a missing binary just logs — the app never depends on it.
mod mcp_sidecar {
    use std::sync::Mutex;

    static CHILD: Mutex<Option<std::process::Child>> = Mutex::new(None);

    /// Locate + spawn the sidecar. Called AFTER `prepare_data_root` so the
    /// CWD is the per-user data root — `mcp_servers.json` lives there and
    /// the sidecar's relative default resolves to it.
    pub fn start() {
        let bin_name = if cfg!(windows) {
            "neoethos-mcp.exe"
        } else {
            "neoethos-mcp"
        };
        // Search every place the bundler might have put the sidecar, in
        // priority order, so it is found regardless of how the installer laid
        // it out (audit follow-up: a locally-built 0.5.3 packaged before the
        // sidecar finished building shipped without it beside the exe).
        let mut checked: Vec<std::path::PathBuf> = Vec::new();
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Some(p) = std::env::var("NEOETHOS_MCP_PATH")
            .ok()
            .map(std::path::PathBuf::from)
        {
            candidates.push(p);
        }
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(dir) = exe_path.parent() {
                candidates.push(dir.join(bin_name)); // beside the exe
                candidates.push(dir.join("resources").join(bin_name)); // tauri resources subdir
            }
        }
        let exe = candidates.into_iter().find(|p| {
            let hit = p.exists();
            if !hit {
                checked.push(p.clone());
            }
            hit
        });
        let Some(exe) = exe else {
            eprintln!(
                "MCP sidecar binary '{bin_name}' not found — MCP tools unavailable this session. \
                 Set NEOETHOS_MCP_PATH to its full path, or reinstall from a build that bundles \
                 it. Checked: {}",
                checked
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            return;
        };
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("--config").arg("mcp_servers.json");
        // Windows: spawn WITHOUT a console window (2026-07-20). The sidecar is
        // a console binary, so it opened a visible cmd window the operator had
        // to leave alone — closing it kills the sidecar and every MCP tool with
        // it. CREATE_NO_WINDOW keeps the process running fully headless, so
        // there is nothing to close by accident. Its logs still reach the app's
        // stderr/log file.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        match cmd.spawn() {
            Ok(child) => {
                eprintln!("MCP sidecar started: {} (pid {})", exe.display(), child.id());
                if let Ok(mut slot) = CHILD.lock() {
                    *slot = Some(child);
                }
            }
            Err(e) => eprintln!("MCP sidecar failed to start ({}): {e}", exe.display()),
        }
    }

    /// Kill the sidecar. Called from the window-close handler BEFORE the
    /// hard process exit — a plain `std::process::exit` would orphan it.
    pub fn stop() {
        if let Ok(mut slot) = CHILD.lock() {
            if let Some(mut child) = slot.take() {
                let _ = child.kill();
            }
        }
    }
}

/// P2P mesh sidecar lifecycle — makes the mesh a FIRST-CLASS, app-managed
/// feature instead of a binary the operator must launch by hand. The isolated
/// `neoethos-mesh` binary (its OWN workspace — the engine never links iroh)
/// joins the serverless swarm, discovers peers automatically, and bridges work
/// to this app's local `/federation/*` API. Because pooling compute over the
/// open internet is a conscious privacy/network choice, it is STRICTLY OPT-IN:
/// the sidecar only runs when the operator enabled the mesh (a `mesh_enabled`
/// marker in the data root, flipped by the Federation UI toggle). Best-effort,
/// exactly like `mcp_sidecar`: a missing binary just logs and the app never
/// depends on it.
mod mesh_sidecar {
    use std::sync::Mutex;

    static CHILD: Mutex<Option<std::process::Child>> = Mutex::new(None);

    /// Opt-in marker in the data root (CWD after `prepare_data_root`). Its mere
    /// presence means the operator turned the mesh ON. A marker file keeps this
    /// desktop-shell-local and independent of the engine's `Settings` struct.
    fn enabled_marker() -> std::path::PathBuf {
        std::path::PathBuf::from("mesh_enabled")
    }

    pub fn is_enabled() -> bool {
        enabled_marker().exists()
    }

    /// Whether the child process is currently alive (reaps it if it exited).
    pub fn is_running() -> bool {
        let Ok(mut slot) = CHILD.lock() else {
            return false;
        };
        match slot.as_mut() {
            Some(child) => match child.try_wait() {
                Ok(None) => true,             // still running
                _ => {
                    *slot = None; // exited or errored → clear the dead handle
                    false
                }
            },
            None => false,
        }
    }

    /// Locate the sidecar binary — the SAME search order as `mcp_sidecar` so it
    /// is found however the installer laid it out (env override, beside the exe,
    /// or a `resources/` subdir).
    fn locate() -> Option<std::path::PathBuf> {
        let bin_name = if cfg!(windows) {
            "neoethos-mesh.exe"
        } else {
            "neoethos-mesh"
        };
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Some(p) = std::env::var("NEOETHOS_MESH_PATH")
            .ok()
            .map(std::path::PathBuf::from)
        {
            candidates.push(p);
        }
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(dir) = exe_path.parent() {
                candidates.push(dir.join(bin_name));
                candidates.push(dir.join("resources").join(bin_name));
            }
        }
        candidates.into_iter().find(|p| p.exists())
    }

    /// Start the sidecar IF the operator opted in and it isn't already running.
    /// Called at startup (a no-op when disabled) and by the UI toggle. MUST run
    /// after `backend::start` so the ephemeral API-port file the mesh reads to
    /// find our `/federation` bridge already exists.
    pub fn start() {
        if !is_enabled() || is_running() {
            return;
        }
        let Some(exe) = locate() else {
            eprintln!(
                "mesh sidecar 'neoethos-mesh' not found — mesh unavailable this session. \
                 Set NEOETHOS_MESH_PATH to its full path, or reinstall from a build that \
                 bundles it beside the app."
            );
            return;
        };
        let mut cmd = std::process::Command::new(&exe);
        // The mesh auto-resolves its --app-url from temp/neoethos_api_port; we
        // only pin its identity + swarm state under the data root so a stable
        // node id survives restarts.
        cmd.arg("--data-dir").arg("mesh");
        // Windows: run fully headless (no console window to close by accident) —
        // same rationale as the MCP sidecar.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        match cmd.spawn() {
            Ok(child) => {
                eprintln!("mesh sidecar started: {} (pid {})", exe.display(), child.id());
                if let Ok(mut slot) = CHILD.lock() {
                    *slot = Some(child);
                }
            }
            Err(e) => eprintln!("mesh sidecar failed to start ({}): {e}", exe.display()),
        }
    }

    /// Kill the sidecar. Called from the window-close handler BEFORE the hard
    /// process exit (a plain `process::exit` would orphan it).
    pub fn stop() {
        if let Ok(mut slot) = CHILD.lock() {
            if let Some(mut child) = slot.take() {
                let _ = child.kill();
            }
        }
    }

    /// Persist the operator's opt-in choice and start/stop the sidecar to match.
    pub fn set_enabled(enabled: bool) {
        if enabled {
            let _ = std::fs::write(enabled_marker(), "1");
            start();
        } else {
            let _ = std::fs::remove_file(enabled_marker());
            stop();
        }
    }
}

/// Mesh on/off state for the Federation UI toggle.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MeshStatus {
    /// The operator has opted in (the sidecar auto-starts with the app).
    enabled: bool,
    /// The sidecar process is alive right now.
    running: bool,
}

/// Current mesh opt-in + process state (the Federation panel polls this).
#[tauri::command]
fn mesh_status() -> MeshStatus {
    MeshStatus {
        enabled: mesh_sidecar::is_enabled(),
        running: mesh_sidecar::is_running(),
    }
}

/// Flip the mesh on/off from the UI: persists the opt-in and starts/stops the
/// sidecar immediately (no app restart needed).
#[tauri::command]
fn mesh_set_enabled(enabled: bool) -> MeshStatus {
    mesh_sidecar::set_enabled(enabled);
    MeshStatus {
        enabled: mesh_sidecar::is_enabled(),
        running: mesh_sidecar::is_running(),
    }
}

/// Reveal a file or folder in the OS file manager (Windows Explorer). Files are
/// highlighted via `/select,`; folders open directly. Lets the user find any
/// data/model/log the app stores with one click.
#[tauri::command]
fn open_path(path: String) -> Result<(), String> {
    let p = std::path::Path::new(&path);
    let mut cmd = std::process::Command::new("explorer");
    if p.is_file() {
        cmd.arg("/select,").arg(&path);
    } else {
        cmd.arg(&path);
    }
    // explorer.exe returns non-zero exit codes even on success; ignore status.
    cmd.spawn().map(|_| ()).map_err(|e| e.to_string())
}

/// Resolve the data root the same way the engine does: the operator's
/// `config.yaml` `system.data_dir`. Falls back to `<per-user data root>/data`
/// (audit M08: the old fallback was the developer's workstation path baked
/// into every shipped binary — meaningless on any other machine).
fn resolve_data_root() -> PathBuf {
    // Audit #125, same family: read the config THIS PROCESS resolved, not
    // whatever `Settings::load()` picks. In a dev/portable launch the engine
    // reads the repo's `config.yaml` while `Settings::load()` reads
    // `%LOCALAPPDATA%\neoethos\config.yaml` — so the chart/symbol commands
    // could show a different dataset than the one the engine trades on.
    let from_resolved = RESOLVED_CONFIG_PATH
        .get()
        .and_then(|p| neoethos_core::Settings::from_yaml(p).ok());
    if let Some(s) = from_resolved {
        let d = s.system.data_dir.clone();
        if d.exists() {
            return d;
        }
    } else if let Ok(s) = neoethos_core::Settings::load() {
        let d = s.system.data_dir.clone();
        if d.exists() {
            return d;
        }
    }
    // Same root `prepare_data_root` establishes: the directory holding the
    // per-user config.yaml (CWD in dev/portable launches, the OS-standard
    // per-user dir otherwise), plus `/data`.
    neoethos_core::config::user_config_path()
        .parent()
        .map(|p| p.join("data"))
        .unwrap_or_else(|| PathBuf::from("data"))
}

/// Establish the per-user data root the engine reads (config.yaml + data/ +
/// cache/ + models/), the cross-platform STANDARD way — no hardcoded paths, so
/// it works for any user on any machine:
///   1. `NEOETHOS_USER_DATA_DIR` — explicit override (dev points it at a repo).
///   2. The current dir IF it already holds config.yaml (dev/portable launch).
///   3. OS-standard per-user dir: `%LOCALAPPDATA%\neoethos` /
///      `~/.local/share/neoethos` / `~/Library/Application Support/neoethos`
///      (via `neoethos_core::config::user_config_path()`).
/// On first run it SEEDS config.yaml (+ default symbol costs) from the bundled
/// read-only defaults, then chdirs to the root so every relative read resolves.
fn prepare_data_root(app: &tauri::App) {
    use tauri::Manager;

    let overridden = std::env::var("NEOETHOS_USER_DATA_DIR")
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .is_some();

    // Dev/portable: launched from a dir that already has config.yaml → keep it.
    if !overridden && std::path::Path::new("config.yaml").exists() {
        eprintln!("data root → current dir (config.yaml present)");
        record_config_path(PathBuf::from("config.yaml"));
        return;
    }

    // Installed launch with an arbitrary CWD (Start-Menu / Explorer shortcut
    // without a "Start in" directory): the installer lays config.yaml BESIDE
    // the exe. Prefer that root before the per-user fallback — otherwise the
    // fallback lands in an EMPTY per-user dir and every engine call fails
    // with "data directory does not exist" even though the real root is
    // right next to the binary (observed live 2026-07-20).
    if !overridden
        && let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
        && dir.join("config.yaml").exists()
    {
        match std::env::set_current_dir(dir) {
            Ok(()) => {
                eprintln!("data root → exe dir (config.yaml present): {}", dir.display());
                record_config_path(dir.join("config.yaml"));
                return;
            }
            Err(e) => eprintln!("set working dir {} failed: {e}", dir.display()),
        }
    }

    // Override-aware canonical path (user_config_path honours the override).
    let cfg_path = neoethos_core::config::user_config_path();
    let root = cfg_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    if let Err(e) = std::fs::create_dir_all(&root) {
        eprintln!("could not create data root {}: {e}", root.display());
    }

    // First run: seed the editable config + default symbol costs from the
    // bundled read-only defaults so a fresh install works out of the box.
    //
    // Audit #125: a seed failure used to be one `eprintln!` on a console
    // nobody sees, after which the app started on `Settings::default()`. Both
    // failure arms now say plainly that the app will refuse to start, and
    // `backend::start` enforces it.
    if !cfg_path.exists() {
        match app
            .path()
            .resolve("resources/config.yaml", tauri::path::BaseDirectory::Resource)
        {
            Ok(res) => match std::fs::copy(&res, &cfg_path) {
                Ok(_) => eprintln!("seeded default config → {}", cfg_path.display()),
                Err(e) => eprintln!(
                    "seed config {} → {} FAILED: {e} — the app will refuse to start rather \
                     than run on built-in defaults",
                    res.display(),
                    cfg_path.display()
                ),
            },
            Err(e) => eprintln!(
                "bundled resources/config.yaml could not be resolved: {e} — the app will \
                 refuse to start rather than run on built-in defaults"
            ),
        }
        let data = root.join("data");
        let _ = std::fs::create_dir_all(&data);
        if let Ok(res) = app
            .path()
            .resolve("resources/symbol_metadata.json", tauri::path::BaseDirectory::Resource)
        {
            let dst = data.join("symbol_metadata.json");
            if !dst.exists() {
                let _ = std::fs::copy(&res, &dst);
            }
        }
    }

    if let Err(e) = std::env::set_current_dir(&root) {
        // Audit #125: this failure used to be survivable-looking. It is not —
        // every relative read in the engine resolves against the CWD. The
        // config path recorded below is ABSOLUTE, so the config half is now
        // immune; the data half still depends on the chdir, so say so.
        eprintln!(
            "set working dir {} failed: {e} — relative data/cache/model paths will resolve \
             against the process CWD instead",
            root.display()
        );
    }
    record_config_path(cfg_path);
    eprintln!("data root → {}", root.display());
}

#[derive(Serialize)]
struct AppInfo {
    version: String,
    data_root: String,
    data_root_exists: bool,
}

/// App identity + resolved data root (shown in the status bar so the operator
/// always knows which dataset the charts come from).
#[tauri::command]
fn app_info() -> AppInfo {
    let root = resolve_data_root();
    AppInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        data_root_exists: root.exists(),
        data_root: root.display().to_string(),
    }
}

/// Symbols present on disk (e.g. EURUSD, GBPUSD, XAUUSD …).
#[tauri::command]
async fn list_symbols() -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(|| {
        let root = resolve_data_root();
        neoethos_data::discover_symbols(&root).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Timeframes available for a symbol (M1, M5, H1, …), ordered.
#[tauri::command]
async fn list_timeframes(symbol: String) -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = resolve_data_root();
        neoethos_data::discover_timeframes(&root, &symbol).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

/// One OHLC bar in the shape TradingView Lightweight Charts wants:
/// `time` is a UTC timestamp in SECONDS (Vortex stores ms → /1000).
#[derive(Serialize)]
struct Candle {
    time: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

/// Trailing `limit` candles for (symbol, timeframe), read straight from the
/// Vortex file. Returned ascending by time (the loader normalises order).
#[tauri::command]
async fn chart(
    symbol: String,
    timeframe: String,
    limit: Option<usize>,
) -> Result<Vec<Candle>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = resolve_data_root();
        let ohlcv = neoethos_data::load_symbol_timeframe(&root, &symbol, &timeframe)
            .map_err(|e| e.to_string())?;
        let n = ohlcv.close.len();
        if n == 0 {
            return Ok::<Vec<Candle>, String>(Vec::new());
        }
        let ts = ohlcv.timestamp.clone().unwrap_or_default();
        let take = limit.unwrap_or(1500).min(n);
        let start = n - take;
        let mut out = Vec::with_capacity(take);
        let mut last_t = i64::MIN;
        for i in start..n {
            let t = ts.get(i).copied().unwrap_or(0) / 1000; // ms → s
            // Lightweight Charts requires strictly-ascending unique times.
            if t <= last_t {
                continue;
            }
            last_t = t;
            out.push(Candle {
                time: t,
                open: ohlcv.open[i],
                high: ohlcv.high[i],
                low: ohlcv.low[i],
                close: ohlcv.close[i],
            });
        }
        Ok(out)
    })
    .await
    .map_err(|e| e.to_string())?
}

/// Open a native OS file picker for a data file to import (CSV/Parquet/TSV).
/// Returns the chosen absolute path, or `None` if the user cancelled — the
/// webview's `<input type=file>` can't expose a real path, so the import needs
/// this native dialog.
#[tauri::command]
async fn pick_data_file() -> Result<Option<String>, String> {
    let file = rfd::AsyncFileDialog::new()
        .set_title("Choose a data file to import")
        .add_filter("Data files", &["csv", "tsv", "parquet", "txt"])
        .add_filter("All files", &["*"])
        .pick_file()
        .await;
    Ok(file.map(|f| f.path().to_string_lossy().to_string()))
}

/// How much local history exists for a symbol on a given timeframe — so the
/// Discovery pre-flight can show the operator EXACTLY what's about to be
/// searched (years of data + bar count per pair) before they start.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SymbolCoverage {
    symbol: String,
    bars: usize,
    first_ms: i64,
    last_ms: i64,
    years: f64,
}

#[tauri::command]
async fn data_coverage(
    symbols: Vec<String>,
    timeframe: String,
) -> Result<Vec<SymbolCoverage>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = resolve_data_root();
        let out = symbols
            .iter()
            .map(|sym| match neoethos_data::load_symbol_timeframe(&root, sym, &timeframe) {
                Ok(o) => {
                    let ts = o.timestamp.unwrap_or_default();
                    let first = ts.first().copied().unwrap_or(0);
                    let last = ts.last().copied().unwrap_or(0);
                    // 365.25 d/yr in ms.
                    let years = if last > first {
                        (last - first) as f64 / 31_557_600_000.0
                    } else {
                        0.0
                    };
                    SymbolCoverage { symbol: sym.clone(), bars: o.close.len(), first_ms: first, last_ms: last, years }
                }
                Err(_) => SymbolCoverage { symbol: sym.clone(), bars: 0, first_ms: 0, last_ms: 0, years: 0.0 },
            })
            .collect::<Vec<_>>();
        Ok(out)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // STANDARD per-user data root (no hardcoded paths) + first-run seed.
            prepare_data_root(app);
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            // Start the full neoethos-app API in-process (Discovery, Training,
            // Risk, Journal, News, Intelligence, Data, Hardware, Autonomous,
            // Codex, …) — every old Flutter feature, reachable from the new UI.
            backend::start();
            // Start the live cTrader spot-price streamer (best-effort; no-op
            // until the broker is authenticated). Feeds the in-process server's
            // /live/spots/stream (SSE) that the UI subscribes to.
            broker::start_spot_streamer();
            // MCP sidecar (best-effort): exposes configured MCP servers' tools
            // to the Supervisor's approval-gated actions on 127.0.0.1:7431.
            mcp_sidecar::start();
            // P2P mesh sidecar (best-effort, OPT-IN): auto-starts only if the
            // operator enabled the mesh in the Federation panel. Runs AFTER
            // backend::start so the API-port file it reads already exists.
            mesh_sidecar::start();
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // in-process backend base URL + file manager
            api_base,
            open_path,
            // P2P mesh sidecar opt-in toggle (Federation panel)
            mesh_status,
            mesh_set_enabled,
            // local vortex data
            app_info,
            list_symbols,
            list_timeframes,
            chart,
            pick_data_file,
            data_coverage,
            // live cTrader broker (in-process, auto-auth)
            broker::broker_status,
            broker::broker_chart,
            broker::broker_accounts,
            broker::select_account,
            broker::account_snapshot,
            broker::place_order,
            broker::close_position,
            broker::reauth_broker,
            broker::refresh_broker_costs,
        ])
        .on_window_event(|_window, event| {
            // Heavy Discovery/Training work runs on tokio blocking threads. A
            // tight CPU loop there can keep the process alive past window close
            // — the operator reported a stuck search that only a full REBOOT
            // stopped. When the operator closes the window, hard-exit the whole
            // process so EVERY worker thread dies immediately: "close = stop",
            // guaranteed, no orphaned CPU, no reboot. Broker positions are held
            // server-side by cTrader, so exiting never touches live orders.
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                // Accidental-close guard (2026-07-20): a ~30h discovery run
                // was killed at 95.5% by an unintended window close. When an
                // engine is running, ask before dying; fail-open if the
                // probe can't answer so a wedged backend never traps the
                // operator in an unclosable window.
                if backend::engine_running() {
                    let close_anyway = rfd::MessageDialog::new()
                        .set_level(rfd::MessageLevel::Warning)
                        .set_title("NeoEthos — engine running")
                        .set_description(
                            "A Discovery/Training/Autopilot engine is RUNNING.\n\
                             Closing the app will KILL it and all unsaved \
                             progress will be lost (results are only written \
                             at the very end of a run).\n\nClose anyway?",
                        )
                        .set_buttons(rfd::MessageButtons::YesNo)
                        .show();
                    if !matches!(close_anyway, rfd::MessageDialogResult::Yes) {
                        api.prevent_close();
                        return;
                    }
                }
                // Kill the sidecar children FIRST — process::exit would
                // orphan them (they're independent OS processes).
                mcp_sidecar::stop();
                mesh_sidecar::stop();
                std::process::exit(0);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
