use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubsystemSection {
    System,
    App,
    Cli,
    Discovery,
    Training,
    Bindings,
}

impl SubsystemSection {
    pub fn ordered() -> Vec<Self> {
        vec![
            Self::System,
            Self::App,
            Self::Cli,
            Self::Discovery,
            Self::Training,
            Self::Bindings,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "SYSTEM",
            Self::App => "APP",
            Self::Cli => "CLI",
            Self::Discovery => "DISCOVERY",
            Self::Training => "TRAINING",
            Self::Bindings => "BINDINGS",
        }
    }

    fn parse(raw: &str) -> Result<Self> {
        match raw.trim() {
            "SYSTEM" => Ok(Self::System),
            "APP" => Ok(Self::App),
            "CLI" => Ok(Self::Cli),
            "DISCOVERY" => Ok(Self::Discovery),
            "TRAINING" => Ok(Self::Training),
            "BINDINGS" => Ok(Self::Bindings),
            other => Err(anyhow!("unknown subsystem section: {other}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionedRunRecord {
    pub run_id: String,
    pub parent_run_id: Option<String>,
    pub started_at: String,
    pub finished_at: String,
    pub subsystem: SubsystemSection,
    pub operation: String,
    pub status: String,
    pub symbol: Option<String>,
    pub timeframe: Option<String>,
    pub error_code: Option<String>,
    pub message: String,
    pub body: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SectionSlot {
    pub current: Option<SectionedRunRecord>,
    pub previous: Option<SectionedRunRecord>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalSectionedLog {
    sections: HashMap<SubsystemSection, SectionSlot>,
}

impl CanonicalSectionedLog {
    pub fn new() -> Self {
        let mut sections = HashMap::new();
        for section in SubsystemSection::ordered() {
            sections.insert(section, SectionSlot::default());
        }
        Self { sections }
    }

    pub fn section_order(&self) -> Vec<SubsystemSection> {
        SubsystemSection::ordered()
    }

    pub fn section(&self, section: SubsystemSection) -> Option<&SectionSlot> {
        self.sections.get(&section)
    }

    pub fn update_section(&mut self, section: SubsystemSection, record: SectionedRunRecord) {
        let slot = self.sections.entry(section).or_default();
        slot.previous = slot.current.take();
        slot.current = Some(record);
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        for (index, section) in SubsystemSection::ordered().iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            let slot = self.sections.get(section).cloned().unwrap_or_default();
            out.push_str(&format!("===== SECTION {} =====\n", section.as_str()));
            Self::render_slot(&mut out, "CURRENT", slot.current.as_ref());
            Self::render_slot(&mut out, "PREVIOUS", slot.previous.as_ref());
        }
        out
    }

    pub fn parse(raw: &str) -> Result<Self> {
        let mut lines = raw.lines().peekable();
        let mut parsed = Self::new();

        for expected_section in SubsystemSection::ordered() {
            while matches!(lines.peek(), Some(line) if line.trim().is_empty()) {
                lines.next();
            }

            let header = lines.next().with_context(|| {
                format!("missing section header for {}", expected_section.as_str())
            })?;
            let section_name = header
                .strip_prefix("===== SECTION ")
                .and_then(|rest| rest.strip_suffix(" ====="))
                .ok_or_else(|| anyhow!("invalid section header: {header}"))?;
            let parsed_section = SubsystemSection::parse(section_name)?;
            if parsed_section != expected_section {
                return Err(anyhow!(
                    "section order mismatch: expected {} got {}",
                    expected_section.as_str(),
                    parsed_section.as_str()
                ));
            }

            let current = Self::parse_slot(&mut lines, "CURRENT")?;
            let previous = Self::parse_slot(&mut lines, "PREVIOUS")?;

            parsed
                .sections
                .insert(parsed_section, SectionSlot { current, previous });
        }

        Ok(parsed)
    }

    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read canonical log {}", path.display()))?;
        Self::parse(&raw)
            .with_context(|| format!("failed to parse canonical log {}", path.display()))
    }

    fn render_slot(out: &mut String, slot_name: &str, record: Option<&SectionedRunRecord>) {
        out.push_str(&format!("--- {} ---\n", slot_name));
        match record {
            Some(record) => {
                out.push_str(&format!("run_id={}\n", record.run_id));
                out.push_str(&format!(
                    "parent_run_id={}\n",
                    record.parent_run_id.as_deref().unwrap_or("")
                ));
                out.push_str(&format!("started_at={}\n", record.started_at));
                out.push_str(&format!("finished_at={}\n", record.finished_at));
                out.push_str(&format!("subsystem={}\n", record.subsystem.as_str()));
                out.push_str(&format!("operation={}\n", record.operation));
                out.push_str(&format!("status={}\n", record.status));
                out.push_str(&format!(
                    "symbol={}\n",
                    record.symbol.as_deref().unwrap_or("")
                ));
                out.push_str(&format!(
                    "timeframe={}\n",
                    record.timeframe.as_deref().unwrap_or("")
                ));
                out.push_str(&format!(
                    "error_code={}\n",
                    record.error_code.as_deref().unwrap_or("")
                ));
                out.push_str(&format!("message={}\n", record.message));
                out.push_str(&format!("body={}\n", record.body));
            }
            None => out.push_str("<empty>\n"),
        }
    }

    fn parse_slot<'a, I>(
        lines: &mut std::iter::Peekable<I>,
        slot_name: &str,
    ) -> Result<Option<SectionedRunRecord>>
    where
        I: Iterator<Item = &'a str>,
    {
        let header = lines
            .next()
            .with_context(|| format!("missing slot header for {slot_name}"))?;
        let expected_header = format!("--- {} ---", slot_name);
        if header != expected_header {
            return Err(anyhow!(
                "invalid slot header for {slot_name}: expected {expected_header}, got {header}"
            ));
        }

        let Some(next_line) = lines.next() else {
            return Err(anyhow!("missing slot body for {slot_name}"));
        };
        if next_line == "<empty>" {
            return Ok(None);
        }

        let run_id = Self::parse_field(next_line, "run_id")?;
        let parent_run_id = Self::empty_to_none(Self::parse_next_field(lines, "parent_run_id")?);
        let started_at = Self::parse_next_field(lines, "started_at")?;
        let finished_at = Self::parse_next_field(lines, "finished_at")?;
        let subsystem = SubsystemSection::parse(&Self::parse_next_field(lines, "subsystem")?)?;
        let operation = Self::parse_next_field(lines, "operation")?;
        let status = Self::parse_next_field(lines, "status")?;
        let symbol = Self::empty_to_none(Self::parse_next_field(lines, "symbol")?);
        let timeframe = Self::empty_to_none(Self::parse_next_field(lines, "timeframe")?);
        let error_code = Self::empty_to_none(Self::parse_next_field(lines, "error_code")?);
        let message = Self::parse_next_field(lines, "message")?;
        let body = Self::parse_next_field(lines, "body")?;

        Ok(Some(SectionedRunRecord {
            run_id,
            parent_run_id,
            started_at,
            finished_at,
            subsystem,
            operation,
            status,
            symbol,
            timeframe,
            error_code,
            message,
            body,
        }))
    }

    fn parse_next_field<'a, I>(lines: &mut I, key: &str) -> Result<String>
    where
        I: Iterator<Item = &'a str>,
    {
        let line = lines
            .next()
            .with_context(|| format!("missing field {key}"))?;
        Self::parse_field(line, key)
    }

    fn parse_field(line: &str, key: &str) -> Result<String> {
        line.strip_prefix(&format!("{key}="))
            .map(ToString::to_string)
            .ok_or_else(|| anyhow!("invalid field for {key}: {line}"))
    }

    fn empty_to_none(raw: String) -> Option<String> {
        if raw.is_empty() { None } else { Some(raw) }
    }

    fn recovery_record(target: SubsystemSection, error: &anyhow::Error) -> SectionedRunRecord {
        let now = Utc::now().to_rfc3339();
        SectionedRunRecord {
            run_id: format!("system-recovery-{}", target.as_str().to_lowercase()),
            parent_run_id: None,
            started_at: now.clone(),
            finished_at: now,
            subsystem: SubsystemSection::System,
            operation: "sectioned_log_recovery".to_string(),
            status: "DEGRADED".to_string(),
            symbol: None,
            timeframe: None,
            error_code: Some("MALFORMED_CANONICAL_LOG".to_string()),
            message: format!(
                "recovered malformed canonical log while updating {}",
                target.as_str()
            ),
            body: error.to_string(),
        }
    }
}

impl Default for CanonicalSectionedLog {
    fn default() -> Self {
        Self::new()
    }
}

struct LockFileGuard {
    path: PathBuf,
}

impl LockFileGuard {
    fn acquire(log_path: &Path) -> Result<Self> {
        let lock_path = PathBuf::from(format!("{}.lock", log_path.display()));
        for _ in 0..100 {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(_) => return Ok(Self { path: lock_path }),
                Err(err) if err.kind() == ErrorKind::AlreadyExists => {
                    sleep(Duration::from_millis(10));
                }
                Err(err) => {
                    return Err(err).with_context(|| {
                        format!(
                            "failed to create canonical log lock {}",
                            lock_path.display()
                        )
                    });
                }
            }
        }

        Err(anyhow!(
            "timed out acquiring canonical log lock {}",
            lock_path.display()
        ))
    }
}

impl Drop for LockFileGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub fn update_section_file(
    path: impl AsRef<Path>,
    section: SubsystemSection,
    record: SectionedRunRecord,
) -> Result<CanonicalSectionedLog> {
    let path = path.as_ref();
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create canonical log parent directory {}",
                parent.display()
            )
        })?;
    }

    let _lock = LockFileGuard::acquire(path)?;
    let mut log = match fs::read_to_string(path) {
        Ok(raw) => match CanonicalSectionedLog::parse(&raw) {
            Ok(existing) => existing,
            Err(err) => {
                let mut recovered = CanonicalSectionedLog::new();
                recovered.update_section(
                    SubsystemSection::System,
                    CanonicalSectionedLog::recovery_record(section, &err),
                );
                recovered
            }
        },
        Err(err) if err.kind() == ErrorKind::NotFound => CanonicalSectionedLog::new(),
        Err(err) => {
            return Err(err)
                .with_context(|| format!("failed to read canonical log {}", path.display()));
        }
    };

    log.update_section(section, record);

    let tmp_path = PathBuf::from(format!("{}.tmp", path.display()));
    fs::write(&tmp_path, log.render()).with_context(|| {
        format!(
            "failed to write canonical log temp file {}",
            tmp_path.display()
        )
    })?;
    fs::rename(&tmp_path, path).with_context(|| {
        format!(
            "failed to atomically replace canonical log {} with {}",
            path.display(),
            tmp_path.display()
        )
    })?;

    Ok(log)
}
