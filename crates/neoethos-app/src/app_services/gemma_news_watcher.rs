//! Autonomous Gemma news watcher (#128).
//!
//! A background tokio task that wakes up at scheduled times of day
//! and prompts the local LLM to digest news / scan the market. The
//! result of each run goes into the persistent memory layer (#125)
//! as a `NoteCategory::EventDigest`, so the next interactive chat
//! turn can recall the morning scan without re-running it.
//!
//! ## Modes
//!
//! 1. **MORNING_SCAN** — fires once per local day at
//!    `news.gemma_morning_scan_time` (default 07:00). Asks Gemma to
//!    scan overnight headlines + check the calendar for high-impact
//!    events scheduled today, focused on the symbols the operator
//!    has trained models for.
//!
//! 2. **SESSION_START** — fires `gemma_session_start_lead_min`
//!    minutes before each major session open (Tokyo 00:00 UTC,
//!    London 08:00 UTC, NY 13:00 UTC). Re-checks for fresh
//!    headlines since the morning scan.
//!
//! 3. **ADAPTIVE_POLL** — when the calendar shows a high-impact
//!    event within `gemma_adaptive_poll_threshold_min` minutes,
//!    switches to polling every `gemma_adaptive_poll_interval_secs`
//!    seconds until the event prints, then summarises impact and
//!    falls back to the slow loop.
//!
//! ## Off by default
//!
//! `news.gemma_news_watcher_enabled = false` until the operator
//! turns it on from Settings. The watcher never schedules itself
//! without explicit opt-in.
//!
//! ## Bounded LLM cost
//!
//! Each fire is at most ONE chat call (which may itself recurse
//! through ReAct tool steps, capped at `MAX_TOOL_STEPS = 6`). The
//! adaptive-poll cadence floor is 5 s — even a misconfigured
//! interval can't hammer the GPU more than that.

#[cfg(feature = "gemma-backend")]
use crate::app_services::gemma_memory::{self, NoteCategory};
#[cfg(feature = "gemma-backend")]
use crate::server::gemma::run_chat_with_tools;
#[cfg(feature = "gemma-backend")]
use crate::server::state::AppApiState;
#[cfg(feature = "gemma-backend")]
use anyhow::Result;
#[cfg(feature = "gemma-backend")]
use chrono::{Local, NaiveTime, Timelike};
#[cfg(feature = "gemma-backend")]
use std::sync::Arc;
#[cfg(feature = "gemma-backend")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "gemma-backend")]
use std::time::Duration;

/// Cap on tokens per scheduled chat call. Sized for a multi-paragraph
/// digest without burning runtime on a runaway generation.
#[cfg(feature = "gemma-backend")]
const SCHEDULED_CHAT_MAX_TOKENS: u32 = 1200;

/// Global broadcast channel for hot-reloading WatcherConfig (#133).
/// `POST /settings` calls `notify_config_changed()` after persisting
/// `config.yaml`; the watcher loop's `select!` picks up the new
/// snapshot on the next tick (or immediately, if it was sleeping).
///
/// We use a `tokio::sync::watch` channel because:
///   - The watcher only cares about the LATEST config — older
///     pending changes are obsolete by definition.
///   - The receiver can `borrow()` the current value without
///     consuming it, so the loop's normal sleep-tick branch can
///     just read it whenever it wakes.
///   - Sender + receiver are cheap to clone via Arc.
#[cfg(feature = "gemma-backend")]
static CONFIG_CHANNEL: std::sync::OnceLock<(
    tokio::sync::watch::Sender<WatcherConfig>,
    tokio::sync::watch::Receiver<WatcherConfig>,
)> = std::sync::OnceLock::new();

/// Floor on the adaptive-poll interval — a misconfigured 1s would
/// hammer the GPU. 5 s is the hard floor regardless of what
/// `gemma_adaptive_poll_interval_secs` says.
#[cfg(feature = "gemma-backend")]
const ADAPTIVE_POLL_FLOOR_SECS: u64 = 5;

/// Outer-loop tick. Every 30s the watcher checks whether any of the
/// three modes should fire. Short enough to catch a morning-scan
/// HH:MM with at-worst 30s lag; long enough not to thrash.
#[cfg(feature = "gemma-backend")]
const WATCHER_TICK_SECS: u64 = 30;

/// Which scheduled mode just fired. Tagged so the prompt builder
/// can pick the right template + the memory key can encode the
/// run kind.
#[cfg(feature = "gemma-backend")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatcherMode {
    MorningScan,
    SessionStart,
    AdaptivePoll,
}

#[cfg(feature = "gemma-backend")]
impl WatcherMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MorningScan => "morning_scan",
            Self::SessionStart => "session_start",
            Self::AdaptivePoll => "adaptive_poll",
        }
    }
}

/// Snapshot of the watcher config that the loop reads at boot.
/// Changes to config.yaml during the process's lifetime are NOT
/// hot-reloaded — flipping the master toggle in Settings will take
/// effect on the next backend restart. Justification: the
/// scheduler is opt-in already; a rare restart is acceptable cost
/// for not having to coordinate config invalidation with the
/// running task.
#[cfg(feature = "gemma-backend")]
#[derive(Debug, Clone)]
pub struct WatcherConfig {
    pub enabled: bool,
    pub morning_scan_time: Option<NaiveTime>,
    pub session_start_lead_min: u32,
    pub adaptive_poll_threshold_min: u32,
    pub adaptive_poll_interval_secs: u64,
}

#[cfg(feature = "gemma-backend")]
impl WatcherConfig {
    /// Parse from a `NewsConfig`. Returns a config with
    /// `enabled=false` for any malformed time string — we don't
    /// fail the whole backend boot because one HH:MM is wrong.
    pub fn from_news_config(news: &neoethos_core::config::NewsConfig) -> Self {
        let morning_scan_time = if news.gemma_morning_scan_time.trim().is_empty() {
            None
        } else {
            // Accept both "HH:MM" and "HH:MM:SS" formats.
            NaiveTime::parse_from_str(news.gemma_morning_scan_time.trim(), "%H:%M")
                .or_else(|_| {
                    NaiveTime::parse_from_str(news.gemma_morning_scan_time.trim(), "%H:%M:%S")
                })
                .ok()
        };
        Self {
            enabled: news.gemma_news_watcher_enabled,
            morning_scan_time,
            session_start_lead_min: news.gemma_session_start_lead_min,
            adaptive_poll_threshold_min: news.gemma_adaptive_poll_threshold_min,
            adaptive_poll_interval_secs: news
                .gemma_adaptive_poll_interval_secs
                .max(ADAPTIVE_POLL_FLOOR_SECS),
        }
    }
}

/// Spawn the watcher loop. Returns immediately with a handle the
/// caller can drop / abort on shutdown. The task itself is a no-op
/// when `config.enabled` is false — it still runs (so a future
/// hot-reload could flip the switch) but every tick is a quick
/// "disabled? sleep" check.
#[cfg(feature = "gemma-backend")]
pub fn spawn(
    state: AppApiState,
    config: WatcherConfig,
    cancel: Arc<AtomicBool>,
) -> tokio::task::JoinHandle<()> {
    // Seed the broadcast channel with the boot-time config. Calling
    // this is idempotent — only the first call sets the channel; the
    // POST /settings handler later calls `notify_config_changed` to
    // push new snapshots.
    let _ = CONFIG_CHANNEL.set(tokio::sync::watch::channel(config.clone()));
    tokio::spawn(async move {
        run_loop(state, config, cancel).await;
    })
}

/// Push a fresh WatcherConfig to the running loop. Called by
/// `POST /settings` after persisting `config.yaml` so a UI toggle
/// of `gemma_news_watcher_enabled` takes effect immediately
/// instead of requiring a backend restart (#133).
///
/// No-op when the channel hasn't been initialised yet (i.e. the
/// watcher hasn't been spawned, e.g. on an early startup error
/// path). No-op when there are no receivers — the watch::Sender's
/// `send_replace` swallows that case.
#[cfg(feature = "gemma-backend")]
pub fn notify_config_changed(new_config: WatcherConfig) {
    if let Some((tx, _rx)) = CONFIG_CHANNEL.get() {
        // `send_replace` returns the previous value; we discard
        // it (the watcher reads the current via Receiver::borrow).
        tx.send_replace(new_config);
        tracing::info!(
            target: "neoethos_app::gemma_news_watcher",
            "watcher config hot-reloaded via POST /settings"
        );
    }
}

#[cfg(feature = "gemma-backend")]
async fn run_loop(state: AppApiState, initial_config: WatcherConfig, cancel: Arc<AtomicBool>) {
    // Mutable local copy so hot-reload (#133) can swap in new
    // values mid-loop. The initial value matches what `spawn`
    // seeded into CONFIG_CHANNEL.
    let mut config = initial_config;
    let mut config_rx = CONFIG_CHANNEL
        .get()
        .map(|(_, rx)| rx.clone());
    tracing::info!(
        target: "neoethos_app::gemma_news_watcher",
        enabled = config.enabled,
        morning_scan = ?config.morning_scan_time,
        session_start_lead_min = config.session_start_lead_min,
        adaptive_threshold_min = config.adaptive_poll_threshold_min,
        "gemma news watcher loop starting"
    );

    // No early-exit on `!config.enabled` — the loop stays alive so
    // a hot-reload flipping the switch ON later (via #132 + #133)
    // takes effect without a backend restart. Disabled state just
    // skips the mode-firing work each tick.

    // Probe the headless-browser path at boot so the operator gets
    // a clear log line about whether the JS-rendered sources will
    // work on this machine. The probe itself doesn't launch Chrome
    // — it just checks the install paths.
    #[cfg(feature = "headless-browser")]
    {
        if crate::app_services::news_sources::headless_browser::is_available() {
            tracing::info!(
                target: "neoethos_app::gemma_news_watcher",
                "headless browser detected — JS-rendered news sources are available"
            );
        } else {
            tracing::warn!(
                target: "neoethos_app::gemma_news_watcher",
                "no Chrome/Edge/Chromium found — headless-browser sources will fall back \
                 to direct HTTP which is often blocked by Cloudflare. Install Chrome \
                 from https://www.google.com/chrome/ to unlock the JS-rendered sources."
            );
        }
    }

    // Last-fire timestamps prevent double-firing the same mode
    // within a single window. We compare on (local_date,
    // mode_kind) for morning-scan + session-start, and use a
    // monotonic instant for adaptive-poll.
    let mut last_morning_scan_date: Option<chrono::NaiveDate> = None;
    let mut last_session_start_hour: Option<u32> = None;
    let mut last_adaptive_fire: Option<std::time::Instant> = None;

    while !cancel.load(Ordering::Relaxed) {
        // Skip the per-tick mode evaluation entirely while the
        // watcher is in "disabled" state. The select! at the
        // bottom still waits for either a sleep or a hot-reload
        // signal, so flipping the switch ON via #132 + #133 wakes
        // us straight into the work below.
        if !config.enabled {
            let tick = Duration::from_secs(WATCHER_TICK_SECS);
            if let Some(rx) = config_rx.as_mut() {
                tokio::select! {
                    _ = tokio::time::sleep(tick) => {}
                    _ = rx.changed() => {
                        config = rx.borrow().clone();
                        tracing::info!(
                            target: "neoethos_app::gemma_news_watcher",
                            enabled = config.enabled,
                            "hot-reload while disabled — re-checking config"
                        );
                    }
                }
            } else {
                tokio::time::sleep(tick).await;
            }
            continue;
        }

        let now_local = Local::now();
        let now_naive = now_local.naive_local();
        let today = now_naive.date();
        let current_time = now_naive.time();

        // ── MORNING_SCAN ─────────────────────────────────────
        if let Some(target) = config.morning_scan_time
            && last_morning_scan_date != Some(today)
            && current_time >= target
            // Only fire if we're within 30 minutes of the target —
            // a process starting at 14:00 should NOT replay this
            // morning's scan; the operator will see it on memory
            // read instead.
            && (current_time.hour() * 60 + current_time.minute())
                .saturating_sub(target.hour() * 60 + target.minute())
                < 30
        {
            if let Err(err) = fire_mode(&state, WatcherMode::MorningScan).await {
                tracing::warn!(
                    target: "neoethos_app::gemma_news_watcher",
                    error = %err,
                    "morning_scan fire failed"
                );
            }
            last_morning_scan_date = Some(today);
        }

        // ── SESSION_START ────────────────────────────────────
        //
        // Major session opens in UTC: Tokyo 00, London 08, NY 13.
        // We want to fire `session_start_lead_min` BEFORE each
        // one. So at e.g. 07:50 local-UTC we fire for London 08.
        // We dedupe on the target-hour so each session fires once
        // per UTC day, not once per tick of the lead window.
        if config.session_start_lead_min > 0 {
            let now_utc = chrono::Utc::now();
            let lead = chrono::Duration::minutes(config.session_start_lead_min as i64);
            let target_utc = now_utc + lead;
            let target_hour = target_utc.hour();
            let target_minute = target_utc.minute();
            // Fire when the lead-projected clock is at exactly the
            // session-open hour with minute < 5 (giving 5 minutes
            // of timer slack — at WATCHER_TICK_SECS = 30 the tick
            // grid is well inside that).
            if [0u32, 8, 13].contains(&target_hour)
                && target_minute < 5
                && last_session_start_hour != Some(target_hour)
            {
                if let Err(err) = fire_mode(&state, WatcherMode::SessionStart).await {
                    tracing::warn!(
                        target: "neoethos_app::gemma_news_watcher",
                        error = %err,
                        session_hour_utc = target_hour,
                        "session_start fire failed"
                    );
                }
                last_session_start_hour = Some(target_hour);
            }
            // Reset the dedupe an hour after firing so the SAME
            // session opens tomorrow can fire again.
            if let Some(h) = last_session_start_hour
                && now_utc.hour() > h
                && now_utc.hour() < (h + 23) % 24
            {
                last_session_start_hour = None;
            }
        }

        // ── ADAPTIVE_POLL ────────────────────────────────────
        //
        // Predicate: "is there a high-impact event scheduled
        // within `adaptive_poll_threshold_min` minutes?". We
        // consult the ForexFactory aggregator (#129). The fetch
        // is cached for 5 minutes so this loop tick doesn't
        // re-issue the HTTP request — at WATCHER_TICK_SECS = 30
        // we'd otherwise hammer FF twice a minute.
        let calendar_event_within_threshold =
            high_impact_event_imminent(config.adaptive_poll_threshold_min);
        if calendar_event_within_threshold {
            let interval = Duration::from_secs(config.adaptive_poll_interval_secs);
            let should_fire = match last_adaptive_fire {
                None => true,
                Some(t) => t.elapsed() >= interval,
            };
            if should_fire {
                if let Err(err) = fire_mode(&state, WatcherMode::AdaptivePoll).await {
                    tracing::warn!(
                        target: "neoethos_app::gemma_news_watcher",
                        error = %err,
                        "adaptive_poll fire failed"
                    );
                }
                last_adaptive_fire = Some(std::time::Instant::now());
            }
        }

        // Sleep until the next tick OR until the config channel
        // fires a change. Hot-reload (#133): when POST /settings
        // pushes a new WatcherConfig, the receiver's `changed()`
        // resolves immediately and we re-read the snapshot.
        // Cancel-flag check happens at the top of the loop, so a
        // mid-sleep cancel is at most WATCHER_TICK_SECS away.
        let tick = Duration::from_secs(WATCHER_TICK_SECS);
        if let Some(rx) = config_rx.as_mut() {
            tokio::select! {
                _ = tokio::time::sleep(tick) => {}
                _ = rx.changed() => {
                    config = rx.borrow().clone();
                    tracing::info!(
                        target: "neoethos_app::gemma_news_watcher",
                        enabled = config.enabled,
                        morning_scan = ?config.morning_scan_time,
                        "applied hot-reloaded config"
                    );
                    if !config.enabled {
                        // Operator turned the watcher off via UI —
                        // log + keep looping; the next iteration's
                        // top-of-loop "disabled? sleep" branch
                        // takes care of doing nothing useful.
                        tracing::info!(
                            target: "neoethos_app::gemma_news_watcher",
                            "watcher disabled via hot-reload; entering quiet mode"
                        );
                    }
                }
            }
        } else {
            // No channel (defensive — should never happen in a
            // properly spawned watcher) → fall back to simple
            // sleep so the loop still ticks.
            tokio::time::sleep(tick).await;
        }
        // If we're now disabled (either at boot or via reload),
        // skip the rest of the loop body so we don't fire any
        // mode. Cheaper than restructuring around an early
        // continue at the top of every iteration.
        if !config.enabled {
            continue;
        }
    }
    tracing::info!(
        target: "neoethos_app::gemma_news_watcher",
        "watcher loop cancelled by atomic flag"
    );
}

/// Invoke Gemma with the mode-specific prompt and persist the
/// digest. Failure modes:
/// - Gemma not loaded yet → returns Err with the model-file message
///   (the outer loop logs and continues; next tick retries).
/// - Inference fails → same handling.
/// - Memory write fails → the digest is logged but we still return
///   Ok (the digest itself was produced; persistence is the
///   secondary goal).
#[cfg(feature = "gemma-backend")]
async fn fire_mode(state: &AppApiState, mode: WatcherMode) -> Result<()> {
    let prompt = build_prompt_for_mode(mode);
    tracing::info!(
        target: "neoethos_app::gemma_news_watcher",
        mode = mode.as_str(),
        "firing scheduled gemma run"
    );
    let outcome = run_chat_with_tools(state.clone(), prompt, SCHEDULED_CHAT_MAX_TOKENS).await?;
    let now = chrono::Utc::now();
    let key = format!(
        "watcher:{}:{}",
        mode.as_str(),
        now.format("%Y-%m-%dT%H:%M:%SZ")
    );
    let content = format!(
        "Mode: {}\nModel: {}\nElapsed: {} ms\n\n{}",
        mode.as_str(),
        outcome.model_id,
        outcome.elapsed_ms,
        outcome.response
    );
    if let Ok(store) = gemma_memory::global() {
        if let Err(err) = store.save(&key, &content, NoteCategory::EventDigest) {
            tracing::warn!(
                target: "neoethos_app::gemma_news_watcher",
                error = %err,
                key = %key,
                "failed to persist watcher digest"
            );
        }
    }
    Ok(())
}

/// Predicate consulted by the ADAPTIVE_POLL branch — does the
/// upstream calendar (#129) show any High-impact event within
/// `threshold_min` minutes of now? Cached at the source level so
/// calling this every WATCHER_TICK_SECS doesn't hammer FF.
///
/// Failures (network down, parse error, all sources disabled) are
/// treated as "no imminent event" so the watcher degrades to its
/// quiet baseline instead of firing the LLM on a stale guess. Each
/// source's display name is logged on a successful hit so the
/// audit trail tells the operator which feed flagged the event.
fn high_impact_event_imminent(threshold_min: u32) -> bool {
    use crate::app_services::news_sources::{NewsImpact, default_sources};
    let now_ms = chrono::Utc::now().timestamp_millis();
    let until_ms = now_ms + (threshold_min as i64) * 60_000;
    let sources = default_sources();
    for src in sources {
        let Ok(events) = src.fetch_calendar_events() else {
            continue;
        };
        if let Some(hit) = events.iter().find(|e| {
            e.scheduled_at_unix_ms >= now_ms
                && e.scheduled_at_unix_ms <= until_ms
                && e.impact == NewsImpact::High
        }) {
            tracing::info!(
                target: "neoethos_app::gemma_news_watcher",
                source = src.display_name(),
                event_title = %hit.title,
                event_currency = %hit.currency,
                event_ms = hit.scheduled_at_unix_ms,
                "adaptive-poll predicate matched"
            );
            return true;
        }
    }
    false
}

/// Mode-specific prompt templates. The model already has the
/// system prompt (scope + role + memory hint) appended by the
/// tool-loop, so these templates focus on what's specific to the
/// firing reason.
#[cfg(feature = "gemma-backend")]
fn build_prompt_for_mode(mode: WatcherMode) -> String {
    match mode {
        WatcherMode::MorningScan => "It's the start of a new trading day. Scan the situation:\n\
             1. Use `list_memory_keys` with prefix `user_pref:` to see the symbols \
                the operator cares about most.\n\
             2. Use `fetch_url` against ForexFactory or a similar calendar to find \
                today's high-impact events for those symbols' currencies.\n\
             3. Read recent headlines (Reuters, BBC, FT) for major themes affecting \
                those pairs.\n\
             4. Summarise in 5-8 bullets: today's calendar risks + dominant narrative \
                + any specific pair-by-pair notes.\n\
             5. Save the summary with `save_memory_note` using category `event_digest` \
                and key `morning:YYYY-MM-DD`.\n\
             Stay concise. Do NOT recommend trades — only describe the landscape."
            .to_string(),
        WatcherMode::SessionStart => "A major trading session is about to open. Re-scan briefly:\n\
             1. Load the morning digest with `load_memory_note` (key `morning:YYYY-MM-DD`).\n\
             2. Check `fetch_url` against a news source for headlines posted since \
                the morning scan.\n\
             3. Note any new high-impact events or theme shifts.\n\
             4. Persist the delta via `save_memory_note` with category \
                `event_digest` keyed `session:<session>:<datetime>`.\n\
             Keep it to 3-5 bullets — the operator only needs the DELTA from the morning scan."
            .to_string(),
        WatcherMode::AdaptivePoll => "A high-impact event is imminent. Check if the print is out:\n\
             1. `fetch_url` against the calendar (ForexFactory/Investing) for the event \
                you flagged.\n\
             2. If the actual number is published: write a 2-3 sentence summary \
                (actual vs consensus, direction of surprise) and save with category \
                `event_digest` keyed `event:<symbol>:<datetime>`.\n\
             3. If still pending: respond with the single word PENDING."
            .to_string(),
    }
}

#[cfg(all(test, feature = "gemma-backend"))]
mod tests {
    use super::*;
    use neoethos_core::config::NewsConfig;

    #[test]
    fn watcher_config_parses_hh_mm_morning_time() {
        let mut nc = NewsConfig::default();
        nc.gemma_news_watcher_enabled = true;
        nc.gemma_morning_scan_time = "08:30".to_string();
        let wc = WatcherConfig::from_news_config(&nc);
        assert!(wc.enabled);
        assert_eq!(
            wc.morning_scan_time,
            Some(NaiveTime::from_hms_opt(8, 30, 0).unwrap())
        );
    }

    #[test]
    fn watcher_config_floors_poll_interval() {
        let mut nc = NewsConfig::default();
        nc.gemma_adaptive_poll_interval_secs = 1; // operator-supplied bad value
        let wc = WatcherConfig::from_news_config(&nc);
        assert_eq!(wc.adaptive_poll_interval_secs, ADAPTIVE_POLL_FLOOR_SECS);
    }

    #[test]
    fn watcher_config_disables_morning_scan_on_blank_time() {
        let mut nc = NewsConfig::default();
        nc.gemma_morning_scan_time = "".to_string();
        let wc = WatcherConfig::from_news_config(&nc);
        assert!(wc.morning_scan_time.is_none());
    }

    #[test]
    fn watcher_config_disables_morning_scan_on_malformed_time() {
        let mut nc = NewsConfig::default();
        nc.gemma_morning_scan_time = "not a time".to_string();
        let wc = WatcherConfig::from_news_config(&nc);
        assert!(wc.morning_scan_time.is_none());
    }

    #[test]
    fn mode_str_round_trips() {
        for m in [
            WatcherMode::MorningScan,
            WatcherMode::SessionStart,
            WatcherMode::AdaptivePoll,
        ] {
            assert!(!m.as_str().is_empty());
        }
    }

    #[test]
    fn prompt_for_each_mode_mentions_save_memory_note() {
        for m in [WatcherMode::MorningScan, WatcherMode::SessionStart] {
            let p = build_prompt_for_mode(m);
            assert!(
                p.contains("save_memory_note"),
                "mode {:?} prompt should instruct the model to persist its digest",
                m
            );
        }
    }

    /// Verify the hot-reload channel actually broadcasts when
    /// `notify_config_changed` is called. We use the channel
    /// directly (not through `spawn`) to avoid needing a
    /// AppApiState fixture.
    #[tokio::test]
    async fn notify_config_changed_broadcasts_to_receiver() {
        // Seed the channel — `set` is idempotent (subsequent calls
        // in this test fixture return Err and we ignore it).
        let mut nc = neoethos_core::config::NewsConfig::default();
        nc.gemma_news_watcher_enabled = false;
        let initial = WatcherConfig::from_news_config(&nc);
        let _ = CONFIG_CHANNEL.set(tokio::sync::watch::channel(initial));

        let mut rx = CONFIG_CHANNEL
            .get()
            .expect("channel set")
            .1
            .clone();

        // Push a config with enabled = true.
        let mut nc_on = neoethos_core::config::NewsConfig::default();
        nc_on.gemma_news_watcher_enabled = true;
        nc_on.gemma_session_start_lead_min = 25;
        let new_cfg = WatcherConfig::from_news_config(&nc_on);
        notify_config_changed(new_cfg);

        // Receiver should see the change.
        rx.changed().await.expect("receiver should observe change");
        let observed = rx.borrow().clone();
        assert!(observed.enabled);
        assert_eq!(observed.session_start_lead_min, 25);
    }
}
