#[path = "../src/sectioned_log.rs"]
mod sectioned_log;

use sectioned_log::{CanonicalSectionedLog, SectionedRunRecord, SubsystemSection};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn sample_record(run_id: &str, operation: &str, status: &str, message: &str) -> SectionedRunRecord {
    SectionedRunRecord {
        run_id: run_id.to_string(),
        parent_run_id: None,
        started_at: "2026-03-21T12:00:00Z".to_string(),
        finished_at: "2026-03-21T12:00:01Z".to_string(),
        subsystem: SubsystemSection::Training,
        operation: operation.to_string(),
        status: status.to_string(),
        symbol: Some("EURUSD".to_string()),
        timeframe: Some("M1".to_string()),
        error_code: None,
        message: message.to_string(),
        body: format!("body for {run_id}"),
    }
}

fn unique_temp_path(test_name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "forex_core_sectioned_log_{}_{}_{}.log",
        test_name,
        std::process::id(),
        nonce
    ))
}

#[test]
fn create_canonical_log_from_empty_state() {
    let log = CanonicalSectionedLog::new();

    let rendered = log.render();
    let parsed = CanonicalSectionedLog::parse(&rendered).expect("rendered log should parse");

    assert_eq!(parsed.section_order(), SubsystemSection::ordered());
    for section in SubsystemSection::ordered() {
        let entry = parsed.section(section).expect("all default sections should exist");
        assert!(entry.current.is_none(), "section should start empty");
        assert!(entry.previous.is_none(), "section should start empty");
    }
}

#[test]
fn update_one_section_preserves_other_sections() {
    let mut log = CanonicalSectionedLog::new();
    let training_record = sample_record("training-1", "train", "SUCCESS", "training ok");
    let discovery_record = SectionedRunRecord {
        subsystem: SubsystemSection::Discovery,
        ..sample_record("discovery-1", "discover", "SUCCESS", "discovery ok")
    };

    log.update_section(SubsystemSection::Training, training_record.clone());
    log.update_section(SubsystemSection::Discovery, discovery_record.clone());

    let parsed = CanonicalSectionedLog::parse(&log.render()).expect("rendered log should parse");
    let training = parsed
        .section(SubsystemSection::Training)
        .expect("training section should exist");
    let discovery = parsed
        .section(SubsystemSection::Discovery)
        .expect("discovery section should exist");

    assert_eq!(training.current.as_ref(), Some(&training_record));
    assert_eq!(discovery.current.as_ref(), Some(&discovery_record));
    assert!(training.previous.is_none());
    assert!(discovery.previous.is_none());
}

#[test]
fn update_rotates_current_into_previous() {
    let mut log = CanonicalSectionedLog::new();
    let first = sample_record("training-1", "train", "FAILED", "first run");
    let second = sample_record("training-2", "train", "SUCCESS", "second run");

    log.update_section(SubsystemSection::Training, first.clone());
    log.update_section(SubsystemSection::Training, second.clone());

    let training = log
        .section(SubsystemSection::Training)
        .expect("training section should exist");
    assert_eq!(training.current.as_ref(), Some(&second));
    assert_eq!(training.previous.as_ref(), Some(&first));
}

#[test]
fn update_keeps_only_two_runs_per_section() {
    let mut log = CanonicalSectionedLog::new();
    let first = sample_record("training-1", "train", "FAILED", "first run");
    let second = sample_record("training-2", "train", "SUCCESS", "second run");
    let third = sample_record("training-3", "train", "DEGRADED", "third run");

    log.update_section(SubsystemSection::Training, first);
    log.update_section(SubsystemSection::Training, second.clone());
    log.update_section(SubsystemSection::Training, third.clone());

    let training = log
        .section(SubsystemSection::Training)
        .expect("training section should exist");
    assert_eq!(training.current.as_ref(), Some(&third));
    assert_eq!(training.previous.as_ref(), Some(&second));
}

#[test]
fn update_section_file_rewrites_only_target_section() {
    let path = unique_temp_path("rewrite_only_target");
    let discovery_record = SectionedRunRecord {
        subsystem: SubsystemSection::Discovery,
        ..sample_record("discovery-1", "discover", "SUCCESS", "discovery ok")
    };
    let first_training = sample_record("training-1", "train", "FAILED", "first train");
    let second_training = sample_record("training-2", "train", "SUCCESS", "second train");

    sectioned_log::update_section_file(&path, SubsystemSection::Discovery, discovery_record.clone())
        .expect("first write should succeed");
    sectioned_log::update_section_file(&path, SubsystemSection::Training, first_training.clone())
        .expect("second write should succeed");
    sectioned_log::update_section_file(&path, SubsystemSection::Training, second_training.clone())
        .expect("third write should succeed");

    let parsed = CanonicalSectionedLog::read_from_path(&path).expect("canonical log should load");
    let discovery = parsed
        .section(SubsystemSection::Discovery)
        .expect("discovery section should exist");
    let training = parsed
        .section(SubsystemSection::Training)
        .expect("training section should exist");

    assert_eq!(discovery.current.as_ref(), Some(&discovery_record));
    assert_eq!(discovery.previous, None);
    assert_eq!(training.current.as_ref(), Some(&second_training));
    assert_eq!(training.previous.as_ref(), Some(&first_training));

    let _ = fs::remove_file(path);
}

#[test]
fn update_section_file_recovers_malformed_file_and_records_system_event() {
    let path = unique_temp_path("malformed_recovery");
    fs::write(&path, "not a valid canonical sectioned log").expect("should write malformed file");
    let training = sample_record("training-1", "train", "FAILED", "broken input recovery");

    sectioned_log::update_section_file(&path, SubsystemSection::Training, training.clone())
        .expect("recovery write should succeed");

    let parsed = CanonicalSectionedLog::read_from_path(&path).expect("recovered log should load");
    let system = parsed
        .section(SubsystemSection::System)
        .expect("system section should exist");
    let training_section = parsed
        .section(SubsystemSection::Training)
        .expect("training section should exist");

    let recovery = system
        .current
        .as_ref()
        .expect("system current should contain recovery record");
    assert_eq!(training_section.current.as_ref(), Some(&training));
    assert_eq!(recovery.subsystem, SubsystemSection::System);
    assert_eq!(recovery.status, "DEGRADED");
    assert!(
        recovery.message.contains("recovered malformed canonical log"),
        "unexpected recovery message: {}",
        recovery.message
    );

    let _ = fs::remove_file(path);
}
