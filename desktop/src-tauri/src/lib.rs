//! NeoEthos desktop (Tauri) — single-process Rust core + web UI.
//!
//! PoC commands read the operator's on-disk Vortex data directly through the
//! existing `neoethos-data` / `neoethos-core` crates — NO separate backend
//! process, NO HTTP, NO port 7423, NO supervisor/watchdog. The whole class of
//! Flutter+HTTP bugs (spawn spirals, /healthz timeouts, SSE EventFluxException,
//! config seeded to a different dir) cannot exist here: one process, one CWD.

use std::path::PathBuf;
use std::sync::OnceLock;

use neoethos_core::execution_budget::{
    InstalledExecutionBudget, StartupEvent, StartupRuntimeKind, StartupTrace,
    format_startup_diagnostics, parse_parent_cpu_assignment,
};
use serde::{Deserialize, Serialize};
use tauri::Manager;

const EMBEDDED_CONFIG_YAML: &[u8] = include_bytes!("../resources/config.yaml");
const EMBEDDED_SYMBOL_METADATA_JSON: &[u8] = include_bytes!("../resources/symbol_metadata.json");

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

        // Config resolution, load, budget installation, and runtime override
        // installation have already completed on the initial synchronous
        // thread. Reloading here used to create a second policy boundary after
        // Tauri had already initialized its async executor.
        let config_path = super::RESOLVED_CONFIG_PATH
            .get()
            .cloned()
            .unwrap_or_else(|| {
                super::fatal_config_error(
                    "the data root was never established, so no config.yaml path exists",
                    "internal ordering error: backend::start() ran before desktop preflight",
                )
            });
        eprintln!("preloaded config → {}", config_path.display());

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
    pub fn start(resource_dir: Option<&std::path::Path>) {
        let bin_name = if cfg!(windows) {
            "neoethos-mcp.exe"
        } else {
            "neoethos-mcp"
        };
        // Search the path reported by Tauri first, then the two development /
        // portable layouts. Linux deb/rpm resources live under
        // `/usr/lib/NeoEthos`, not beside the `/usr/bin` executable, so current
        // executable discovery alone is not a packaged-runtime contract.
        //
        // `NEOETHOS_MCP_PATH` used to be consulted first. It is DELETED: an
        // environment variable that redirects which BINARY this shell launches
        // is not operator configuration, it is an invisible substitution — it
        // lived in no config file and in no knob catalog, so a stale export
        // could point the app at a different sidecar with nothing on screen to
        // say so. The search order below is deterministic and is what every
        // real install uses; `warn_retired_env_vars` names the variable if it
        // is still set.
        let mut checked: Vec<std::path::PathBuf> = Vec::new();
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Some(resource_dir) = resource_dir {
            candidates.push(resource_dir.join(bin_name));
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
                 Reinstall from a build that bundles it in Tauri's resource directory. \
                 Portable/development builds may put it beside the app executable. \
                 Checked: {}",
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
                eprintln!(
                    "MCP sidecar started: {} (pid {})",
                    exe.display(),
                    child.id()
                );
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
                Ok(None) => true, // still running
                _ => {
                    *slot = None; // exited or errored → clear the dead handle
                    false
                }
            },
            None => false,
        }
    }

    /// Locate the sidecar binary — the SAME search order as `mcp_sidecar`:
    /// Tauri's authoritative resource directory first, followed by portable
    /// layouts beside the executable or in its `resources/` child.
    ///
    /// `NEOETHOS_MESH_PATH` is DELETED, for the same reason as
    /// `NEOETHOS_MCP_PATH`: redirecting which binary joins a P2P swarm is not
    /// something an environment variable should be able to do silently.
    fn locate(resource_dir: Option<&std::path::Path>) -> Option<std::path::PathBuf> {
        let bin_name = if cfg!(windows) {
            "neoethos-mesh.exe"
        } else {
            "neoethos-mesh"
        };
        let mut candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Some(resource_dir) = resource_dir {
            candidates.push(resource_dir.join(bin_name));
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
    pub fn start(resource_dir: Option<&std::path::Path>) {
        if !is_enabled() || is_running() {
            return;
        }
        let Some(exe) = locate(resource_dir) else {
            eprintln!(
                "mesh sidecar 'neoethos-mesh' not found — mesh unavailable this session. \
                 Reinstall from a build that bundles it in Tauri's resource directory. \
                 Portable/development builds may put it beside the executable."
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
                eprintln!(
                    "mesh sidecar started: {} (pid {})",
                    exe.display(),
                    child.id()
                );
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
    pub fn set_enabled(enabled: bool, resource_dir: Option<&std::path::Path>) {
        if enabled {
            let _ = std::fs::write(enabled_marker(), "1");
            start(resource_dir);
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
fn mesh_set_enabled(app: tauri::AppHandle, enabled: bool) -> MeshStatus {
    let resource_dir = app.path().resource_dir().ok();
    mesh_sidecar::set_enabled(enabled, resource_dir.as_deref());
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
fn prepare_data_root() -> Result<PathBuf, String> {
    // ONE reader for this override, and it is not here. `user_config_path()`
    // (which this function calls below) already resolves it through
    // `env_overrides::user_data_dir_override()`; this shell used to read the
    // same variable a SECOND time with its own trim/empty rules. Two readers
    // of one name is how the two-Settings-in-one-process defect was born, so
    // this asks the same function core asks. Same semantics, one source.
    //
    // The variable itself is not this change's to retire: `config.rs::
    // user_config_path()` still consults it, and it is the last env var that
    // can change the answer to "which config file?" (routed as A9).
    let overridden = neoethos_core::env_overrides::user_data_dir_override().is_some();

    // Dev/portable: launched from a dir that already has config.yaml → keep it.
    if !overridden && std::path::Path::new("config.yaml").exists() {
        eprintln!("data root → current dir (config.yaml present)");
        record_config_path(PathBuf::from("config.yaml"));
        return RESOLVED_CONFIG_PATH
            .get()
            .cloned()
            .ok_or_else(|| "failed to record current-directory config path".to_string());
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
                eprintln!(
                    "data root → exe dir (config.yaml present): {}",
                    dir.display()
                );
                record_config_path(dir.join("config.yaml"));
                return RESOLVED_CONFIG_PATH.get().cloned().ok_or_else(|| {
                    "failed to record executable-directory config path".to_string()
                });
            }
            Err(e) => return Err(format!("set working dir {} failed: {e}", dir.display())),
        }
    }

    // Override-aware canonical path (user_config_path honours the override).
    let cfg_path = neoethos_core::config::user_config_path();
    let root = cfg_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    std::fs::create_dir_all(&root)
        .map_err(|error| format!("could not create data root {}: {error}", root.display()))?;

    // First run: seed the editable config + default symbol costs from the
    // bundled read-only defaults so a fresh install works out of the box.
    //
    // Audit #125: a seed failure used to be one `eprintln!` on a console
    // nobody sees, after which the app started on `Settings::default()`. Both
    // failure arms now say plainly that the app will refuse to start, and
    // `backend::start` enforces it.
    if !cfg_path.exists() {
        write_new_seed(&cfg_path, EMBEDDED_CONFIG_YAML)?;
        eprintln!("seeded embedded default config → {}", cfg_path.display());
    }
    let data = root.join("data");
    std::fs::create_dir_all(&data).map_err(|error| {
        format!(
            "could not create seed data directory {}: {error}",
            data.display()
        )
    })?;
    let metadata_path = data.join("symbol_metadata.json");
    if !metadata_path.exists() {
        write_new_seed(&metadata_path, EMBEDDED_SYMBOL_METADATA_JSON)?;
    }

    std::env::set_current_dir(&root).map_err(|error| {
        format!(
            "set working dir {} failed: {error}; refusing to resolve data/cache/model paths \
             against a different directory",
            root.display()
        )
    })?;
    record_config_path(cfg_path);
    eprintln!("data root → {}", root.display());
    RESOLVED_CONFIG_PATH
        .get()
        .cloned()
        .ok_or_else(|| "failed to record resolved config path".to_string())
}

fn write_new_seed(path: &std::path::Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create first-run seed {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("write first-run seed {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("sync first-run seed {}: {error}", path.display()))
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

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExactDatasetGenerationReceipt {
    #[serde(deserialize_with = "deserialize_canonical_dataset_identity")]
    dataset_identity: neoethos_data::CanonicalDatasetIdentity,
    generation: String,
}

fn deserialize_canonical_dataset_identity<'de, D>(
    deserializer: D,
) -> Result<neoethos_data::CanonicalDatasetIdentity, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let encoded = String::deserialize(deserializer)?;
    neoethos_data::CanonicalDatasetIdentity::from_path_component(&encoded).map_err(|error| {
        serde::de::Error::custom(format!(
            "invalid canonical dataset identity {encoded:?}: {error}"
        ))
    })
}

fn verify_selected_generation_receipt(
    selection: &ExactDatasetGenerationReceipt,
    actual_generation: &str,
) -> Result<(), String> {
    if selection.generation != actual_generation {
        return Err(format!(
            "selected generation receipt for {} is stale or invalid: selected {:?}, current {:?}; refresh the Data inventory and select the exact current generation",
            selection.dataset_identity.to_path_component(),
            selection.generation,
            actual_generation,
        ));
    }
    Ok(())
}

fn load_exact_dataset_generation(
    root: &std::path::Path,
    selection: &ExactDatasetGenerationReceipt,
) -> Result<neoethos_data::CanonicalOhlcvFrame, String> {
    let loaded = neoethos_data::load_canonical_timeframe(root, &selection.dataset_identity)
        .map_err(|error| format!("{error:#}"))?;
    verify_selected_generation_receipt(selection, loaded.artifact().generation_id())?;
    Ok(loaded)
}

/// Return the timeframe bound into one exact, fully verified generation.
/// Dataset inventory is the only multi-identity listing API; this command
/// never expands an identity from symbol text.
#[tauri::command]
async fn list_timeframes(selection: ExactDatasetGenerationReceipt) -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = resolve_data_root();
        let loaded = load_exact_dataset_generation(&root, &selection)?;
        Ok(vec![
            loaded.artifact().frame_timeframe().as_str().to_owned(),
        ])
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

/// Trailing `limit` candles from one exact, fully verified canonical Vortex
/// generation. Returned ascending by time (the loader normalises order).
#[tauri::command]
async fn chart(
    selection: ExactDatasetGenerationReceipt,
    limit: Option<usize>,
) -> Result<Vec<Candle>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = resolve_data_root();
        let loaded = load_exact_dataset_generation(&root, &selection)?;
        let ohlcv = loaded.ohlcv();
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

struct ImportPickerFilter {
    name: &'static str,
    extensions: &'static [&'static str],
}

const IMPORT_PICKER_FILTERS: &[ImportPickerFilter] = &[
    ImportPickerFilter {
        name: "CSV",
        extensions: &["csv"],
    },
    ImportPickerFilter {
        name: "TSV",
        extensions: &["tsv"],
    },
    ImportPickerFilter {
        name: "JSON array",
        extensions: &["json"],
    },
    ImportPickerFilter {
        name: "JSON Lines",
        extensions: &["jsonl", "ndjson"],
    },
    ImportPickerFilter {
        name: "Parquet",
        extensions: &["parquet"],
    },
    ImportPickerFilter {
        name: "Arrow IPC file",
        extensions: &["arrow", "feather", "ipc"],
    },
    ImportPickerFilter {
        name: "Arrow IPC stream",
        extensions: &["arrows", "ipcstream"],
    },
    ImportPickerFilter {
        name: "Vortex",
        extensions: &["vortex", "vtx"],
    },
];

/// Open a native OS file picker for one of the eight explicit import formats.
/// Returns the chosen absolute path, or `None` if the user cancelled — the
/// webview's `<input type=file>` can't expose a real path, so the import needs
/// this native dialog.
#[tauri::command]
async fn pick_data_file() -> Result<Option<String>, String> {
    let mut dialog = rfd::AsyncFileDialog::new().set_title("Choose a data file to import");
    for filter in IMPORT_PICKER_FILTERS {
        dialog = dialog.add_filter(filter.name, filter.extensions);
    }
    let file = dialog.pick_file().await;
    Ok(file.map(|f| f.path().to_string_lossy().to_string()))
}

/// How much local history exists for a symbol on a given timeframe — so the
/// Discovery pre-flight can show the operator EXACTLY what's about to be
/// searched (years of data + bar count per pair) before they start.
#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum SymbolCoverage {
    Verified {
        #[serde(rename = "datasetIdentity")]
        dataset_identity: String,
        generation: String,
        symbol: String,
        bars: usize,
        #[serde(rename = "firstMs")]
        first_ms: i64,
        #[serde(rename = "lastMs")]
        last_ms: i64,
        years: f64,
    },
    Failed {
        #[serde(rename = "datasetIdentity")]
        dataset_identity: String,
        generation: String,
        symbol: String,
        error: SymbolCoverageError,
    },
}

#[derive(Serialize)]
struct SymbolCoverageError {
    kind: &'static str,
    detail: String,
}

fn summarize_exact_dataset_coverage<E>(
    selection: &ExactDatasetGenerationReceipt,
    result: Result<neoethos_data::CanonicalOhlcvFrame, E>,
) -> SymbolCoverage
where
    E: std::fmt::Display,
{
    let dataset_identity = selection.dataset_identity.to_path_component();
    let generation = selection.generation.clone();
    let symbol = selection.dataset_identity.symbol_name().to_owned();
    match result {
        Ok(loaded) => {
            let ohlcv = loaded.ohlcv();
            let bars = ohlcv.close.len();
            let timestamps = ohlcv.timestamp.as_deref().unwrap_or_default();
            let first_ms = timestamps.first().copied().unwrap_or(0);
            let last_ms = timestamps.last().copied().unwrap_or(0);
            // 365.25 d/yr in ms.
            let years = if last_ms > first_ms {
                (last_ms - first_ms) as f64 / 31_557_600_000.0
            } else {
                0.0
            };
            SymbolCoverage::Verified {
                dataset_identity,
                generation,
                symbol,
                bars,
                first_ms,
                last_ms,
                years,
            }
        }
        Err(error) => SymbolCoverage::Failed {
            dataset_identity,
            generation,
            symbol,
            error: SymbolCoverageError {
                kind: "load_failed",
                detail: format!("{error:#}"),
            },
        },
    }
}

#[tauri::command]
async fn data_coverage(
    selections: Vec<ExactDatasetGenerationReceipt>,
) -> Result<Vec<SymbolCoverage>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let root = resolve_data_root();
        let out = selections
            .iter()
            .map(|selection| {
                let result = load_exact_dataset_generation(&root, selection);
                summarize_exact_dataset_coverage(selection, result)
            })
            .collect::<Vec<_>>();
        Ok(out)
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg(test)]
mod gate1_desktop_contract_tests {
    use super::*;
    use std::fmt;

    fn exact_receipt(generation: &str) -> ExactDatasetGenerationReceipt {
        ExactDatasetGenerationReceipt {
            dataset_identity: neoethos_data::CanonicalDatasetIdentity::external(
                "desktop-contract-test",
                "EURUSD",
                neoethos_data::CanonicalTimeframe::M5,
                neoethos_data::BarTimestampConvention::BarOpen,
            )
            .expect("the fixture identity must satisfy the canonical contract"),
            generation: generation.to_owned(),
        }
    }

    #[test]
    fn native_picker_advertises_exactly_the_eight_explicit_import_formats() {
        let advertised = IMPORT_PICKER_FILTERS
            .iter()
            .map(|filter| (filter.name, filter.extensions))
            .collect::<Vec<_>>();

        assert_eq!(
            advertised,
            vec![
                ("CSV", &["csv"][..]),
                ("TSV", &["tsv"][..]),
                ("JSON array", &["json"][..]),
                ("JSON Lines", &["jsonl", "ndjson"][..]),
                ("Parquet", &["parquet"][..]),
                ("Arrow IPC file", &["arrow", "feather", "ipc"][..]),
                ("Arrow IPC stream", &["arrows", "ipcstream"][..]),
                ("Vortex", &["vortex", "vtx"][..]),
            ]
        );
        assert!(
            advertised.iter().all(|(_, extensions)| {
                !extensions.contains(&"txt") && !extensions.contains(&"*")
            })
        );
    }

    struct AlternateDiagnostic;

    impl fmt::Display for AlternateDiagnostic {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            if formatter.alternate() {
                formatter.write_str(
                    "canonical dataset rejected: ambiguous identity: found 2 verified sources",
                )
            } else {
                formatter.write_str("canonical dataset rejected")
            }
        }
    }

    #[test]
    fn coverage_loader_failure_is_typed_and_keeps_the_full_diagnostic() {
        let selection = exact_receipt("g1-selected-generation");
        let expected_identity = selection.dataset_identity.to_path_component();
        let coverage = summarize_exact_dataset_coverage(
            &selection,
            Err::<neoethos_data::CanonicalOhlcvFrame, _>(AlternateDiagnostic),
        );

        match coverage {
            SymbolCoverage::Failed {
                dataset_identity,
                generation,
                symbol,
                error,
            } => {
                assert_eq!(dataset_identity, expected_identity);
                assert_eq!(generation, "g1-selected-generation");
                assert_eq!(symbol, "EURUSD");
                assert_eq!(error.kind, "load_failed");
                assert_eq!(
                    error.detail,
                    "canonical dataset rejected: ambiguous identity: found 2 verified sources"
                );
            }
            SymbolCoverage::Verified { .. } => {
                panic!("a loader error must never be represented as verified empty coverage")
            }
        }
    }

    #[test]
    fn generation_receipt_requires_an_exact_case_sensitive_match() {
        let selection = exact_receipt("g1-AbCd");

        verify_selected_generation_receipt(&selection, "g1-AbCd")
            .expect("the selected and current generation are identical");

        let error = verify_selected_generation_receipt(&selection, "g1-abcd")
            .expect_err("even a case-only change must reject the stale receipt");
        assert!(error.contains(&selection.dataset_identity.to_path_component()));
        assert!(error.contains("g1-AbCd"));
        assert!(error.contains("g1-abcd"));
        assert!(error.contains("refresh the Data inventory"));
    }
}

/// Environment variables this shell used to obey and no longer does, paired
/// with what replaced them.
///
/// NOTHING branches on this list — it is a report, not a decision. It exists
/// because an operator who exported a name that used to work deserves to be
/// told the name is dead, rather than watch the app behave as if he had set
/// nothing.
const RETIRED_ENV_VARS: &[(&str, &str)] = &[
    (
        "NEOETHOS_MCP_PATH",
        "put neoethos-mcp[.exe] beside the app executable, or in its resources/ directory",
    ),
    (
        "NEOETHOS_MESH_PATH",
        "put neoethos-mesh[.exe] beside the app executable, or in its resources/ directory",
    ),
];

/// Name every retired variable still set in this process's environment, state
/// its replacement, and say plainly that the value was ignored.
fn warn_retired_env_vars() {
    for (name, replacement) in RETIRED_ENV_VARS {
        let Some(value) = std::env::var_os(name) else {
            continue;
        };
        eprintln!(
            "IGNORED ENV VAR: {name}={} — this variable was deleted. Nothing read it, and \
             NOTHING in this session was changed by it. Instead: {replacement}.",
            value.to_string_lossy()
        );
    }
}

struct DesktopRuntimeGuard {
    runtime: tokio::runtime::Runtime,
    worker_threads: usize,
}

impl DesktopRuntimeGuard {
    fn build(worker_threads: usize) -> Result<Self, String> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(worker_threads)
            .enable_all()
            .build()
            .map_err(|error| format!("build managed desktop Tokio runtime: {error}"))?;
        Ok(Self {
            runtime,
            worker_threads,
        })
    }

    fn install_for_tauri(&self) {
        // Tauri requires this call before `Builder` and documents that the
        // underlying Tokio runtime must not be dropped while its global handle
        // exists. `run` keeps this guard through the complete event loop.
        tauri::async_runtime::set(self.runtime.handle().clone());
    }
}

struct PreparedDesktopStartup {
    runtime_guard: DesktopRuntimeGuard,
    installed_budget: &'static InstalledExecutionBudget,
    trace: StartupTrace,
}

fn prepare_desktop_startup(raw_args: &[String]) -> Result<PreparedDesktopStartup, String> {
    let mut trace = StartupTrace::default();
    // Tauri and Tokio both create threads. Linux SourceSeal signal ownership
    // must be installed on the initial thread before either runtime exists.
    neoethos_data::initialize_source_seal_before_runtime().map_err(|error| error.to_string())?;
    trace
        .record(StartupEvent::ImportSignalPreflightCompleted)
        .map_err(|error| error.to_string())?;
    warn_retired_env_vars();
    let config_path = prepare_data_root()?;
    trace
        .record(StartupEvent::ConfigurationSeededOrLocated)
        .map_err(|error| error.to_string())?;
    let settings = neoethos_core::Settings::from_yaml(&config_path)
        .map_err(|error| format!("load {}: {error:#}", config_path.display()))?;
    trace
        .record(StartupEvent::ConfigurationLoaded)
        .map_err(|error| error.to_string())?;

    let parent_cpu_assignment =
        parse_parent_cpu_assignment(raw_args).map_err(|error| error.to_string())?;
    trace
        .record(StartupEvent::ParentCpuCapParsed)
        .map_err(|error| error.to_string())?;
    let coordination_scope = if parent_cpu_assignment.is_some() {
        neoethos_core::execution_budget::CoordinationScope::ManagedProcessTree
    } else {
        neoethos_core::execution_budget::CoordinationScope::ProcessLocal
    };
    let budget_inputs = neoethos_core::ExecutionBudgetInputs::from_settings_and_parent(
        &settings,
        parent_cpu_assignment.map(|limit| limit.get()),
        coordination_scope,
    )
    .map_err(|error| error.to_string())?;
    budget_inputs
        .clone()
        .resolve()
        .map_err(|error| error.to_string())?;
    trace
        .record(StartupEvent::CpuBudgetResolved)
        .map_err(|error| error.to_string())?;
    let installed_budget =
        neoethos_core::execution_budget::install_process_budget(budget_inputs.request().clone())
            .map_err(|error| error.to_string())?;
    trace
        .record(StartupEvent::CpuBudgetInstalled)
        .map_err(|error| error.to_string())?;

    neoethos_app::server::state::install_config_path(config_path);
    neoethos_core::env_overrides::log_active_overrides_at_startup();
    neoethos_app::install_runtime_overrides_from_settings(&settings);
    trace
        .record(StartupEvent::RuntimeSettingsInstalled)
        .map_err(|error| error.to_string())?;

    let worker_threads = installed_budget.resolved().effective_worker_limit.get();
    let runtime_guard = DesktopRuntimeGuard::build(worker_threads)?;
    trace
        .record(StartupEvent::TokioRuntimeBuilt)
        .map_err(|error| error.to_string())?;
    runtime_guard.install_for_tauri();
    trace
        .record(StartupEvent::TauriAsyncRuntimeInstalled)
        .map_err(|error| error.to_string())?;

    Ok(PreparedDesktopStartup {
        runtime_guard,
        installed_budget,
        trace,
    })
}

/// Perform the real preflight and runtime installation but stop before opening
/// a window. The direct exit preserves Tauri's no-runtime-drop contract.
pub fn startup_diagnostics() -> ! {
    let raw_args: Vec<String> = std::env::args().collect();
    match prepare_desktop_startup(&raw_args) {
        Ok(startup) => {
            eprintln!(
                "{}",
                format_startup_diagnostics(
                    "neoethos-desktop",
                    startup.installed_budget,
                    StartupRuntimeKind::Tauri,
                    Some(startup.runtime_guard.worker_threads),
                    &startup.trace,
                )
            );
            std::mem::forget(startup.runtime_guard);
            std::process::exit(0);
        }
        Err(error) => {
            eprintln!("desktop startup preflight failed: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let raw_args: Vec<String> = std::env::args().collect();
    let mut startup = prepare_desktop_startup(&raw_args)
        .unwrap_or_else(|error| fatal_config_error("desktop startup preflight failed", &error));
    startup
        .trace
        .record(StartupEvent::ApplicationBuilderStarted)
        .unwrap_or_else(|error| {
            fatal_config_error("desktop startup order is invalid", &error.to_string())
        });
    eprintln!(
        "{}",
        format_startup_diagnostics(
            "neoethos-desktop",
            startup.installed_budget,
            StartupRuntimeKind::Tauri,
            Some(startup.runtime_guard.worker_threads),
            &startup.trace,
        )
    );

    let run_result = tauri::Builder::default()
        .setup(|app| {
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
            let resource_dir = app.path().resource_dir().ok();
            mcp_sidecar::start(resource_dir.as_deref());
            // P2P mesh sidecar (best-effort, OPT-IN): auto-starts only if the
            // operator enabled the mesh in the Federation panel. Runs AFTER
            // backend::start so the API-port file it reads already exists.
            mesh_sidecar::start(resource_dir.as_deref());
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
        .run(tauri::generate_context!());

    // `tauri::async_runtime::set` owns a process-global handle. Keep its
    // underlying runtime alive until process teardown, including error exit.
    std::mem::forget(startup.runtime_guard);
    run_result.expect("error while running tauri application");
}
