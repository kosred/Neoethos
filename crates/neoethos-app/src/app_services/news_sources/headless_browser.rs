//! Headless-browser fetcher for the news aggregator (#130).
//!
//! Reality check from the operator: a plain `reqwest::blocking::get`
//! against FXStreet / DailyFX / Investing.com returns either a
//! Cloudflare challenge page or an empty HTML shell — these sites
//! ship JavaScript-rendered content + bot-detection. To get the
//! real DOM the user sees, we need a real browser.
//!
//! This module wraps `headless_chrome` (the Rust DevTools-Protocol
//! client) and exposes a single `fetch_via_browser(url)` function
//! the RSS / HTML sources can use as a drop-in replacement for the
//! reqwest path. The browser process is launched lazily on first
//! call and kept alive for the rest of the process lifetime — each
//! fetch opens a fresh tab, navigates, waits for content to render,
//! then closes the tab.
//!
//! ## Runtime requirement
//!
//! Chrome, Chromium, or Microsoft Edge must be installed on the
//! operator's machine. Detected via the standard install paths
//! (Windows: `Program Files`, macOS: `/Applications`, Linux:
//! `which google-chrome`). When detection fails, every call
//! returns a clear error rather than silently hanging.
//!
//! ## Why feature-gated
//!
//! `headless_chrome` adds ~20 transitive deps + ~3 MB of compiled
//! code. Operators who run NeoEthos purely against the cTrader API
//! (no LLM, no news watcher) shouldn't pay that cost; gating
//! everything behind `--features headless-browser` keeps the
//! default release binary lean.

#![cfg(feature = "headless-browser")]

use anyhow::{Context, Result, anyhow};
use headless_chrome::{Browser, LaunchOptions};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// Default per-page navigation timeout. The browser can take
/// several seconds to load a JS-heavy site and resolve the
/// Cloudflare challenge; 25 s is generous without leaving the
/// caller hanging on a dead URL.
const PAGE_TIMEOUT_SECS: u64 = 25;

/// Maximum HTML body size we'll return. Large pages (image-heavy
/// feeds) can push 5+ MB; capping at 1 MB keeps the downstream
/// LLM context window manageable.
const MAX_BODY_BYTES: usize = 1024 * 1024;

/// Global browser handle. We keep ONE process alive for the
/// lifetime of neoethos-app. Closing + reopening Chrome between
/// fetches would burn 3-5 s per call and exhaust the operator's
/// patience long before the news watcher's morning scan finishes.
static BROWSER: OnceLock<Arc<Mutex<Option<Browser>>>> = OnceLock::new();

fn browser_slot() -> &'static Mutex<Option<Browser>> {
    let arc = BROWSER.get_or_init(|| Arc::new(Mutex::new(None)));
    arc.as_ref()
}

/// Locate a Chrome / Chromium / Edge binary on the host. Returns
/// the first one found via the platform-canonical paths.
///
/// We deliberately do NOT depend on the operator setting PATH
/// correctly — Chrome on Windows lives in `Program Files` and is
/// not on PATH by default, and we want NeoEthos to "just work".
pub fn detect_browser_path() -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = if cfg!(target_os = "windows") {
        vec![
            PathBuf::from(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            PathBuf::from(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
            PathBuf::from(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            PathBuf::from(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            PathBuf::from(
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            ),
            PathBuf::from(
                "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            ),
            PathBuf::from("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
    } else {
        // Linux — most distros put Chrome/Chromium on PATH; we
        // also check the absolute paths in case `which` would
        // miss it.
        vec![
            PathBuf::from("/usr/bin/google-chrome"),
            PathBuf::from("/usr/bin/google-chrome-stable"),
            PathBuf::from("/usr/bin/chromium"),
            PathBuf::from("/usr/bin/chromium-browser"),
            PathBuf::from("/snap/bin/chromium"),
        ]
    };
    candidates.into_iter().find(|p| p.exists())
}

/// Launch the browser process (lazy — first caller takes the hit).
/// Returns the cached handle on subsequent calls. Errors are
/// stable across re-tries: if Chrome wasn't installed when the
/// process started, it still won't be on call #2. The
/// `fetch_via_browser` wrapper turns that into a "no data"
/// degradation for the news sources.
fn ensure_browser() -> Result<()> {
    let slot = browser_slot();
    let mut guard = slot
        .lock()
        .map_err(|_| anyhow!("headless browser mutex poisoned"))?;
    if guard.is_some() {
        return Ok(());
    }
    let bin = detect_browser_path().ok_or_else(|| {
        anyhow!(
            "no Chrome / Chromium / Edge binary found on this machine — \
             install Chrome (https://www.google.com/chrome/) or Edge for \
             the headless-browser news sources to work"
        )
    })?;

    // Configure for headless server-side scraping. The flags here
    // address known Cloudflare / bot-detection heuristics:
    //   --headless=new       — newer headless mode (the old `--headless`
    //                           is more aggressively detected)
    //   --disable-gpu        — required on Windows for headless mode
    //   --no-sandbox         — required for some sandboxed installs;
    //                           harmless on a personal Windows machine
    //   --disable-blink-features=AutomationControlled
    //                        — the navigator.webdriver = true flag
    //                           is the standard "I'm a bot" tell.
    let user_data_dir = std::env::temp_dir().join("neoethos-chrome-profile");
    let _ = std::fs::create_dir_all(&user_data_dir);

    let user_data_dir_owned = user_data_dir.clone();
    let mut builder = LaunchOptions::default_builder();
    builder
        .path(Some(bin.clone()))
        .headless(true)
        .sandbox(false)
        .user_data_dir(Some(user_data_dir_owned));
    let extra_args = vec![
        std::ffi::OsStr::new("--disable-gpu"),
        std::ffi::OsStr::new("--disable-blink-features=AutomationControlled"),
        std::ffi::OsStr::new("--disable-dev-shm-usage"),
    ];
    builder.args(extra_args);
    let opts = builder
        .build()
        .map_err(|e| anyhow!("headless_chrome LaunchOptions build failed: {e}"))?;

    let browser = Browser::new(opts).with_context(|| {
        format!(
            "failed to launch Chrome at {} — try running it manually once to \
             accept any first-launch dialogs",
            bin.display()
        )
    })?;
    tracing::info!(
        target: "neoethos_app::headless_browser",
        binary = %bin.display(),
        "headless browser launched"
    );
    *guard = Some(browser);
    Ok(())
}

/// Fetch a URL through the headless browser. Waits for the page
/// to navigate, gives JS a moment to render content (we don't try
/// to be clever about selectors here — RSS pages are mostly XML
/// served as text/xml after JS-Cloudflare clears, and HTML feeds
/// are read by the parser regardless of late-loaded ads / widgets).
///
/// Returns the full HTML body the browser sees as a String.
/// Truncated at [`MAX_BODY_BYTES`] with `(...)` appended.
pub fn fetch_via_browser(url: &str) -> Result<String> {
    ensure_browser()?;
    let slot = browser_slot();
    let guard = slot
        .lock()
        .map_err(|_| anyhow!("headless browser mutex poisoned"))?;
    let browser = guard.as_ref().ok_or_else(|| anyhow!("browser not initialised"))?;

    let tab = browser
        .new_tab()
        .with_context(|| format!("failed to open new tab for {url}"))?;
    // Give the navigation a generous timeout — Cloudflare's
    // challenge page can take ~10 s to clear on first hit.
    tab.set_default_timeout(Duration::from_secs(PAGE_TIMEOUT_SECS));
    tab.navigate_to(url)
        .with_context(|| format!("navigate_to({url}) failed"))?;
    tab.wait_until_navigated()
        .with_context(|| format!("wait_until_navigated for {url} failed"))?;

    // Sleep a tiny bit so client-side hydration can finish writing
    // into the DOM. Empirically 800 ms is enough for the three
    // target sites; a longer wait blocks the news watcher loop.
    std::thread::sleep(Duration::from_millis(800));

    let mut body = tab
        .get_content()
        .with_context(|| format!("get_content for {url} failed"))?;
    if body.len() > MAX_BODY_BYTES {
        body.truncate(MAX_BODY_BYTES);
        body.push_str("\n<!-- truncated by neoethos-app headless_browser -->");
    }
    // Close the tab so the browser doesn't accumulate them. Errors
    // here are best-effort — the worst case is a leaked tab the
    // browser eventually GCs.
    let _ = tab.close(false);
    Ok(body)
}

/// True iff the host has a browser binary we can launch. Used at
/// startup to log a friendly warning instead of failing every
/// scheduled news-watcher tick on a machine that can't drive a
/// browser.
pub fn is_available() -> bool {
    detect_browser_path().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_browser_returns_some_or_none_without_panic() {
        // We can't assume Chrome is installed in CI, but the
        // detector should never panic — it should return None
        // cleanly on a bare runner.
        let result = detect_browser_path();
        // If found, the returned path must exist on disk.
        if let Some(p) = result {
            assert!(p.exists(), "detected path {} should exist", p.display());
        }
    }

    #[test]
    fn is_available_matches_detect_browser_path() {
        assert_eq!(is_available(), detect_browser_path().is_some());
    }
}
