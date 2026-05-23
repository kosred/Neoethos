//! Persistent memory layer for the local Gemma runtime.
//!
//! Gemma is stateless between `/gemma/chat` calls — each turn loses
//! whatever context the prior conversation built. For the
//! autonomous-mode work (#127 trade explanations, #128 news watcher)
//! the model needs durable storage it can write to once and read
//! from across sessions: "I already digested the ECB statement at
//! 14:15 GMT, summary: …", "user prefers 50-pip stops on EURUSD",
//! "morning scan 2026-05-23: NFP is +220k vs +180k consensus, USD
//! bid".
//!
//! ## Shape
//!
//! - SQLite file at
//!   `<dirs::data_dir>/neoethos/gemma_memory.db`. One DB per OS user.
//! - One table `notes(key TEXT PRIMARY KEY, content TEXT NOT NULL,
//!   category TEXT NOT NULL, created_at INTEGER NOT NULL,
//!   updated_at INTEGER NOT NULL)`.
//! - Categories drive eviction policy:
//!   - `user_pref`     — never evicted, user said so.
//!   - `event_digest`  — kept 90 days, then auto-purged.
//!   - `trade_explanation` — kept 30 days.
//!   - `scratch`       — FIFO eviction once total scratch rows > 200.
//!   The eviction sweep runs on every save (cheap — DELETE … LIMIT).
//!
//! ## Why SQLite, not a flat file
//!
//! Atomic writes (no half-written notes if the process dies during a
//! save), proper indexing for "list keys with prefix X", and a real
//! query language for the eventual "find every note about EURUSD
//! from the last 14 days" use case. `rusqlite` is already in the
//! workspace dep tree (neoethos-core uses it for run-record
//! storage); zero new transitive deps.
//!
//! ## Concurrency
//!
//! Gemma chat is single-flight (one inference at a time, see the
//! `OnceLock<Arc<Mutex>>` in `server/gemma.rs`). Memory writes run
//! inside that lock so a single `rusqlite::Connection` is enough —
//! no pool, no Arc<Mutex<Connection>>. The connection is held by
//! `MemoryStore` which lives as a module-local `OnceLock`.

use anyhow::{Context, Result, anyhow};
use rusqlite::{Connection, OptionalExtension, params};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Note category — drives eviction policy. Stored as the string
/// `as_str()` returns so SQL queries can WHERE on category names
/// without needing a numeric mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteCategory {
    UserPref,
    EventDigest,
    TradeExplanation,
    Scratch,
}

impl NoteCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UserPref => "user_pref",
            Self::EventDigest => "event_digest",
            Self::TradeExplanation => "trade_explanation",
            Self::Scratch => "scratch",
        }
    }

    /// Parse from the string the tool's caller (the model) provides.
    /// Permissive on case + minor variants so a small LLM doesn't
    /// have to memorise an exact constant.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "user_pref" | "userpref" | "preference" | "pref" => Some(Self::UserPref),
            "event_digest" | "eventdigest" | "news" | "event" => Some(Self::EventDigest),
            "trade_explanation" | "trade" | "fill" | "explanation" => {
                Some(Self::TradeExplanation)
            }
            "scratch" | "note" | "temp" => Some(Self::Scratch),
            _ => None,
        }
    }
}

/// A single stored note as read back from the DB.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Note {
    pub key: String,
    pub content: String,
    pub category: String,
    pub created_at_unix_ms: i64,
    pub updated_at_unix_ms: i64,
}

/// Maximum scratch rows kept before FIFO eviction kicks in. Sized
/// for "the model wrote ~10 notes per session × 20 sessions" with
/// headroom. Adjust upward if real usage hits the ceiling.
const SCRATCH_MAX_ROWS: usize = 200;

/// Eviction TTLs in seconds. The cleanup query uses
/// `created_at_unix_ms < (now - TTL)` so anything older is purged.
const EVENT_DIGEST_TTL_SECS: i64 = 90 * 86_400;
const TRADE_EXPLANATION_TTL_SECS: i64 = 30 * 86_400;

pub struct MemoryStore {
    conn: std::sync::Mutex<Connection>,
}

impl MemoryStore {
    /// Open or create the DB at the canonical path. Idempotent —
    /// safe to call from multiple places; reuses the same connection
    /// because callers go through `global()` below.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create memory-DB directory at {}",
                    parent.display()
                )
            })?;
        }
        let conn = Connection::open(path)
            .with_context(|| format!("failed to open SQLite DB at {}", path.display()))?;
        // Single-table schema; primary key on `key` so the LLM can
        // overwrite an existing note by re-saving it with the same
        // key (which is what the `save_memory_note` tool does on
        // every call — idempotent semantics match the model's
        // mental model better than "fail on duplicate").
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS notes (
                key            TEXT PRIMARY KEY,
                content        TEXT NOT NULL,
                category       TEXT NOT NULL,
                created_at     INTEGER NOT NULL,
                updated_at     INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_notes_category ON notes (category);
            CREATE INDEX IF NOT EXISTS idx_notes_updated  ON notes (updated_at);",
        )
        .context("failed to initialise notes table")?;
        Ok(Self {
            conn: std::sync::Mutex::new(conn),
        })
    }

    /// Upsert a note. Idempotent: re-saving the same key replaces
    /// content + category and bumps `updated_at`. Runs eviction
    /// pass on every save so the DB stays bounded without a
    /// separate background sweeper.
    pub fn save(&self, key: &str, content: &str, category: NoteCategory) -> Result<()> {
        let key = key.trim();
        if key.is_empty() {
            anyhow::bail!("memory key cannot be blank");
        }
        if content.is_empty() {
            anyhow::bail!("memory content cannot be blank — use forget() to remove");
        }
        let now = current_unix_ms();
        let conn = self.conn.lock().map_err(|_| anyhow!("memory DB lock poisoned"))?;
        conn.execute(
            "INSERT INTO notes (key, content, category, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(key) DO UPDATE SET
                content    = excluded.content,
                category   = excluded.category,
                updated_at = excluded.updated_at",
            params![key, content, category.as_str(), now],
        )
        .context("failed to upsert memory note")?;

        // Eviction: oldest scratch rows beyond SCRATCH_MAX_ROWS,
        // and TTL-expired event_digest/trade_explanation rows.
        // user_pref is never evicted.
        let scratch_count: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM notes WHERE category = ?1",
                params![NoteCategory::Scratch.as_str()],
                |r| r.get::<_, i64>(0),
            )
            .map(|n| n as usize)
            .unwrap_or(0);
        if scratch_count > SCRATCH_MAX_ROWS {
            let overflow = scratch_count - SCRATCH_MAX_ROWS;
            conn.execute(
                "DELETE FROM notes WHERE rowid IN (
                    SELECT rowid FROM notes
                    WHERE category = ?1
                    ORDER BY updated_at ASC
                    LIMIT ?2
                 )",
                params![NoteCategory::Scratch.as_str(), overflow as i64],
            )
            .context("scratch eviction failed")?;
        }
        let cutoff_event = now - EVENT_DIGEST_TTL_SECS * 1_000;
        let cutoff_trade = now - TRADE_EXPLANATION_TTL_SECS * 1_000;
        conn.execute(
            "DELETE FROM notes WHERE category = ?1 AND created_at < ?2",
            params![NoteCategory::EventDigest.as_str(), cutoff_event],
        )
        .context("event_digest TTL eviction failed")?;
        conn.execute(
            "DELETE FROM notes WHERE category = ?1 AND created_at < ?2",
            params![NoteCategory::TradeExplanation.as_str(), cutoff_trade],
        )
        .context("trade_explanation TTL eviction failed")?;
        Ok(())
    }

    /// Read a single note by key. `None` when absent.
    pub fn load(&self, key: &str) -> Result<Option<Note>> {
        let conn = self.conn.lock().map_err(|_| anyhow!("memory DB lock poisoned"))?;
        let note = conn
            .query_row(
                "SELECT key, content, category, created_at, updated_at
                 FROM notes WHERE key = ?1",
                params![key.trim()],
                |row| {
                    Ok(Note {
                        key: row.get(0)?,
                        content: row.get(1)?,
                        category: row.get(2)?,
                        created_at_unix_ms: row.get(3)?,
                        updated_at_unix_ms: row.get(4)?,
                    })
                },
            )
            .optional()
            .context("memory load query failed")?;
        Ok(note)
    }

    /// List keys, optionally filtered by prefix or category. Sorted
    /// by most-recently-updated first so the model sees the freshest
    /// notes first when it scans memory.
    pub fn list(
        &self,
        prefix: Option<&str>,
        category: Option<NoteCategory>,
        limit: usize,
    ) -> Result<Vec<Note>> {
        let conn = self.conn.lock().map_err(|_| anyhow!("memory DB lock poisoned"))?;
        let mut sql = String::from(
            "SELECT key, content, category, created_at, updated_at FROM notes WHERE 1=1",
        );
        let mut bind: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(p) = prefix {
            sql.push_str(" AND key LIKE ?");
            bind.push(Box::new(format!("{}%", p.trim())));
        }
        if let Some(c) = category {
            sql.push_str(" AND category = ?");
            bind.push(Box::new(c.as_str().to_string()));
        }
        sql.push_str(" ORDER BY updated_at DESC LIMIT ?");
        bind.push(Box::new(limit as i64));

        let mut stmt = conn.prepare(&sql).context("memory list prepare failed")?;
        let params_iter: Vec<&dyn rusqlite::ToSql> =
            bind.iter().map(|b| b.as_ref() as &dyn rusqlite::ToSql).collect();
        let rows = stmt
            .query_map(params_iter.as_slice(), |row| {
                Ok(Note {
                    key: row.get(0)?,
                    content: row.get(1)?,
                    category: row.get(2)?,
                    created_at_unix_ms: row.get(3)?,
                    updated_at_unix_ms: row.get(4)?,
                })
            })
            .context("memory list query failed")?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.context("memory list row decode failed")?);
        }
        Ok(out)
    }

    /// Remove a single note. Returns whether anything was deleted.
    /// Idempotent — deleting a missing key is fine, returns `false`.
    pub fn forget(&self, key: &str) -> Result<bool> {
        let conn = self.conn.lock().map_err(|_| anyhow!("memory DB lock poisoned"))?;
        let n = conn
            .execute("DELETE FROM notes WHERE key = ?1", params![key.trim()])
            .context("memory forget failed")?;
        Ok(n > 0)
    }

    /// Test-only escape hatch — opens an in-memory DB so tests don't
    /// touch the operator's real notes file.
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("in-memory SQLite open failed")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS notes (
                key            TEXT PRIMARY KEY,
                content        TEXT NOT NULL,
                category       TEXT NOT NULL,
                created_at     INTEGER NOT NULL,
                updated_at     INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_notes_category ON notes (category);
            CREATE INDEX IF NOT EXISTS idx_notes_updated  ON notes (updated_at);",
        )?;
        Ok(Self {
            conn: std::sync::Mutex::new(conn),
        })
    }
}

/// Canonical on-disk path for the memory DB. Honours
/// `NEOETHOS_GEMMA_MEMORY_PATH` env override for tests / CI; falls
/// back to `<data_dir>/neoethos/gemma_memory.db`. We use `data_dir`
/// (not `config_dir` which the broker credentials file uses)
/// because this is operational state, not user-edited config.
pub fn default_memory_path() -> PathBuf {
    if let Ok(custom) = std::env::var("NEOETHOS_GEMMA_MEMORY_PATH") {
        if !custom.trim().is_empty() {
            return PathBuf::from(custom);
        }
    }
    let base = dirs::data_dir().unwrap_or_else(|| {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(".local")
    });
    base.join("neoethos").join("gemma_memory.db")
}

/// Process-global store. Initialised lazily on first use so a
/// gemma-disabled build pays nothing for the SQLite open.
static GLOBAL: OnceLock<MemoryStore> = OnceLock::new();

/// Lazy global accessor. The first call opens the SQLite file (or
/// re-uses an existing one); subsequent calls return the cached
/// handle.
pub fn global() -> Result<&'static MemoryStore> {
    if let Some(s) = GLOBAL.get() {
        return Ok(s);
    }
    let store = MemoryStore::open(&default_memory_path())?;
    // Race-tolerant init: if two threads call `global()` at the same
    // time, the loser's store is dropped and we use the winner's.
    let _ = GLOBAL.set(store);
    Ok(GLOBAL.get().expect("just initialised"))
}

fn current_unix_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_then_load_roundtrips() {
        let store = MemoryStore::open_in_memory().expect("open");
        store
            .save("user_pref:risk_tolerance", "low", NoteCategory::UserPref)
            .expect("save");
        let loaded = store.load("user_pref:risk_tolerance").expect("load").expect("some");
        assert_eq!(loaded.content, "low");
        assert_eq!(loaded.category, "user_pref");
        assert!(loaded.created_at_unix_ms > 0);
        assert_eq!(loaded.created_at_unix_ms, loaded.updated_at_unix_ms);
    }

    #[test]
    fn upsert_replaces_content_and_bumps_updated_at() {
        let store = MemoryStore::open_in_memory().expect("open");
        store.save("k", "v1", NoteCategory::Scratch).expect("save 1");
        let first = store.load("k").expect("load").expect("some");
        // Sleep 2ms so the clock advances.
        std::thread::sleep(std::time::Duration::from_millis(2));
        store.save("k", "v2", NoteCategory::Scratch).expect("save 2");
        let second = store.load("k").expect("load").expect("some");
        assert_eq!(second.content, "v2");
        assert!(second.updated_at_unix_ms > first.updated_at_unix_ms);
        // created_at is preserved across upserts (the SQL uses
        // excluded.updated_at only).
        assert_eq!(second.created_at_unix_ms, first.created_at_unix_ms);
    }

    #[test]
    fn list_filters_by_prefix_and_orders_recent_first() {
        let store = MemoryStore::open_in_memory().expect("open");
        store.save("trade:1", "a", NoteCategory::TradeExplanation).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        store.save("trade:2", "b", NoteCategory::TradeExplanation).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        store.save("pref:foo", "c", NoteCategory::UserPref).unwrap();

        let trades = store.list(Some("trade:"), None, 100).expect("list");
        assert_eq!(trades.len(), 2);
        // Most-recent first.
        assert_eq!(trades[0].key, "trade:2");
        assert_eq!(trades[1].key, "trade:1");

        let by_cat = store
            .list(None, Some(NoteCategory::UserPref), 100)
            .expect("list");
        assert_eq!(by_cat.len(), 1);
        assert_eq!(by_cat[0].key, "pref:foo");
    }

    #[test]
    fn forget_removes_and_returns_bool() {
        let store = MemoryStore::open_in_memory().expect("open");
        store.save("ephemeral", "x", NoteCategory::Scratch).unwrap();
        assert!(store.forget("ephemeral").expect("forget"));
        assert!(store.load("ephemeral").expect("load").is_none());
        // Idempotent — second forget returns false but doesn't error.
        assert!(!store.forget("ephemeral").expect("forget"));
    }

    #[test]
    fn scratch_evicts_oldest_beyond_cap() {
        let store = MemoryStore::open_in_memory().expect("open");
        // Stuff in cap + 5 scratch entries.
        for i in 0..(SCRATCH_MAX_ROWS + 5) {
            store
                .save(
                    &format!("s:{i}"),
                    "x",
                    NoteCategory::Scratch,
                )
                .expect("save");
            // Small delay to ensure deterministic updated_at ordering.
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let all = store
            .list(None, Some(NoteCategory::Scratch), 1_000)
            .expect("list");
        // Eviction runs on every save — final count is at most cap.
        assert!(all.len() <= SCRATCH_MAX_ROWS);
        // The 5 oldest (s:0..s:4) should be gone.
        for i in 0..5 {
            assert!(
                store.load(&format!("s:{i}")).expect("load").is_none(),
                "s:{i} should have been evicted"
            );
        }
        // The newest survived.
        assert!(store.load(&format!("s:{}", SCRATCH_MAX_ROWS + 4)).unwrap().is_some());
    }

    #[test]
    fn user_pref_never_evicted_by_scratch_pressure() {
        let store = MemoryStore::open_in_memory().expect("open");
        store.save("pref:sticky", "forever", NoteCategory::UserPref).unwrap();
        for i in 0..(SCRATCH_MAX_ROWS + 10) {
            store.save(&format!("scratch:{i}"), "x", NoteCategory::Scratch).unwrap();
        }
        assert!(store.load("pref:sticky").expect("load").is_some());
    }

    #[test]
    fn category_parse_handles_aliases_and_case() {
        assert_eq!(NoteCategory::parse("user_pref"), Some(NoteCategory::UserPref));
        assert_eq!(NoteCategory::parse("PREF"), Some(NoteCategory::UserPref));
        assert_eq!(NoteCategory::parse("event"), Some(NoteCategory::EventDigest));
        assert_eq!(NoteCategory::parse("trade"), Some(NoteCategory::TradeExplanation));
        assert_eq!(NoteCategory::parse("note"), Some(NoteCategory::Scratch));
        assert_eq!(NoteCategory::parse("blah"), None);
    }

    #[test]
    fn save_rejects_blank_key_or_content() {
        let store = MemoryStore::open_in_memory().expect("open");
        assert!(store.save("", "v", NoteCategory::Scratch).is_err());
        assert!(store.save("   ", "v", NoteCategory::Scratch).is_err());
        assert!(store.save("k", "", NoteCategory::Scratch).is_err());
    }
}
