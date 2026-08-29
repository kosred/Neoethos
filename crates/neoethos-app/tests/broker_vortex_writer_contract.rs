use anyhow::Result;
use neoethos_app::app_services::bootstrap_writer::{
    BrokerTrendbarStreamRequest, CTraderTrendbarProvenanceV1, publish_broker_trendbar_chunks,
};
use neoethos_data::core::dataset_manifest::read_current_manifest;
use neoethos_data::{
    BarTimestampConvention, CTraderEnvironment, CanonicalDatasetIdentity, CanonicalOhlcvChunk,
    CanonicalTimeframe, CanonicalVolumeChunk, load_canonical_timeframe,
    read_vortex_i64_projection_range,
};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const BASE_MS: i64 = 1_700_000_040_000;
const RETRIEVED_UNIX_MS: u64 = 1_800_000_000_000;
const WINDOWS_MAIN_STACK_CHILD: &str = "NEOETHOS_BROKER_VORTEX_WINDOWS_STACK_CHILD";

struct TemporaryRoot(PathBuf);

impl TemporaryRoot {
    fn new(test_name: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        Self(std::env::temp_dir().join(format!(
            "neoethos_broker_vortex_{test_name}_{}_{}",
            std::process::id(),
            nonce
        )))
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TemporaryRoot {
    fn drop(&mut self) {
        if self.0.parent() == Some(std::env::temp_dir().as_path()) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn broker_identity() -> CanonicalDatasetIdentity {
    broker_identity_for(CanonicalTimeframe::M1)
}

fn broker_identity_for(timeframe: CanonicalTimeframe) -> CanonicalDatasetIdentity {
    CanonicalDatasetIdentity::ctrader(
        CTraderEnvironment::Demo,
        "demo.ctraderapi.com",
        7_001,
        14,
        "EURUSD",
        timeframe,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid canonical broker identity")
}

#[derive(Clone)]
struct TestBar {
    timestamp_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: i64,
}

fn sample_bars() -> Vec<TestBar> {
    vec![
        TestBar {
            timestamp_ms: BASE_MS,
            open: 1.100_000_000_000_000_1,
            high: 1.200_000_000_000_000_2,
            low: 1.000_000_000_000_000_2,
            close: 1.150_000_000_000_000_1,
            volume: 0,
        },
        TestBar {
            timestamp_ms: BASE_MS + 60_000,
            open: 1.150_000_000_000_1,
            high: 1.250_000_000_000_000_2,
            low: 1.050_000_000_000_000_3,
            close: 1.200_000_000_000_000_2,
            volume: 16_777_217,
        },
    ]
}

fn one_week_of_m1_bars() -> Vec<TestBar> {
    (0..7 * 24 * 60)
        .map(|row| {
            let center = 1.1 + (row % 1_000) as f64 * 0.000_000_1;
            TestBar {
                timestamp_ms: BASE_MS + i64::from(row) * 60_000,
                open: center,
                high: center + 0.000_2,
                low: center - 0.000_2,
                close: center + 0.000_05,
                volume: i64::from(row % 10_000),
            }
        })
        .collect()
}

fn publish<'a>(
    root: &'a Path,
    identity: &'a CanonicalDatasetIdentity,
    bars: &'a [TestBar],
    expected_generation: Option<&'a str>,
) -> Result<neoethos_data::core::dataset_manifest::PublishResult> {
    let chunks = bars
        .chunks(5_000)
        .map(|bars| {
            Ok(CanonicalOhlcvChunk {
                timestamp_ms: bars.iter().map(|bar| bar.timestamp_ms).collect(),
                open: bars.iter().map(|bar| bar.open).collect(),
                high: bars.iter().map(|bar| bar.high).collect(),
                low: bars.iter().map(|bar| bar.low).collect(),
                close: bars.iter().map(|bar| bar.close).collect(),
                volume: CanonicalVolumeChunk::Int64(bars.iter().map(|bar| bar.volume).collect()),
            })
        })
        .collect::<Vec<_>>();
    publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
        configured_root: root,
        identity,
        expected_generation,
        requested_from_ms: BASE_MS - 60_000,
        requested_to_ms: bars[bars.len() - 1].timestamp_ms + 60_000,
        retrieved_unix_ms: RETRIEVED_UNIX_MS,
        returned_from_ms: bars[0].timestamp_ms,
        returned_to_ms: bars[bars.len() - 1].timestamp_ms,
        row_count: bars.len() as u64,
        chunks,
    })
}

#[test]
fn broker_writer_week_child_uses_windows_main_thread_sized_stack() {
    if std::env::var_os(WINDOWS_MAIN_STACK_CHILD).is_none() {
        return;
    }

    let root = TemporaryRoot::new("week_child");
    let root_path = root.path().to_path_buf();
    let identity = broker_identity();
    let bars = one_week_of_m1_bars();
    let publication = std::thread::Builder::new()
        .name("broker-vortex-windows-stack".to_owned())
        .stack_size(1024 * 1024)
        .spawn(move || publish(&root_path, &identity, &bars, None))
        .expect("spawn bounded-stack broker Vortex publication")
        .join()
        .expect("broker Vortex publication must not overflow")
        .expect("publish one canonical M1 week");
    assert_eq!(publication.manifest().row_count(), 7 * 24 * 60);
}

#[test]
fn broker_writer_publishes_one_week_on_windows_main_thread_sized_stack() {
    let status = std::process::Command::new(
        std::env::current_exe().expect("current broker Vortex integration-test executable"),
    )
    .arg("--exact")
    .arg("broker_writer_week_child_uses_windows_main_thread_sized_stack")
    .arg("--nocapture")
    .arg("--test-threads=1")
    .env(WINDOWS_MAIN_STACK_CHILD, "1")
    .status()
    .expect("spawn bounded-stack broker Vortex publication child");
    assert!(
        status.success(),
        "one-week broker Vortex publication overflowed or failed: {status}"
    );
}

#[test]
fn broker_writer_reopens_exact_identity_provenance_and_raw_int64_volume() -> Result<()> {
    let root = TemporaryRoot::new("canonical_reopen");
    let identity = broker_identity();
    let bars = sample_bars();

    let publication = publish(root.path(), &identity, &bars, None)?;
    let manifest = read_current_manifest(root.path(), &identity)?;

    assert_eq!(manifest.identity(), &identity);
    assert_eq!(manifest.generation_id(), publication.generation());
    assert_eq!(manifest.row_count(), 2);
    assert!(manifest.generation_path().is_file());
    assert!(!root.path().join("symbol=EURUSD").exists());

    let provenance = CTraderTrendbarProvenanceV1::from_envelope(manifest.provenance())?;
    assert_eq!(provenance.dataset_identity(), &identity);
    assert_eq!(
        provenance.requested_range_ms(),
        (BASE_MS - 60_000, BASE_MS + 120_000)
    );
    assert_eq!(provenance.returned_range_ms(), (BASE_MS, BASE_MS + 60_000));
    assert_eq!(provenance.row_count(), 2);
    assert_eq!(provenance.retrieved_unix_ms(), RETRIEVED_UNIX_MS);

    let raw_volume = read_vortex_i64_projection_range(
        manifest.generation_path(),
        "volume",
        0..u64::try_from(bars.len())?,
    )?;
    assert_eq!(raw_volume, vec![0, 16_777_217]);

    let reopened = load_canonical_timeframe(root.path(), &identity)?;
    assert_eq!(reopened.artifact().identity(), &identity);
    assert_eq!(
        reopened.artifact().generation_id(),
        publication.generation()
    );
    assert_eq!(
        reopened.ohlcv().timestamp,
        Some(vec![BASE_MS, BASE_MS + 60_000])
    );
    assert_eq!(reopened.ohlcv().volume, Some(vec![0.0, 16_777_217.0]));
    assert_eq!(reopened.ohlcv().open[0].to_bits(), bars[0].open.to_bits());
    Ok(())
}

#[test]
fn broker_writer_preserves_native_h4_phase_without_epoch_reanchoring() -> Result<()> {
    const H4_MS: i64 = 4 * 60 * 60 * 1_000;
    const NATIVE_FIRST_H4_MS: i64 = 1_451_844_000_000;

    let root = TemporaryRoot::new("native_h4_phase");
    let identity = broker_identity_for(CanonicalTimeframe::H4);
    let mut bars = sample_bars();
    for (row, bar) in bars.iter_mut().enumerate() {
        bar.timestamp_ms = NATIVE_FIRST_H4_MS + i64::try_from(row)? * H4_MS;
    }
    let chunks = vec![Ok(CanonicalOhlcvChunk {
        timestamp_ms: bars.iter().map(|bar| bar.timestamp_ms).collect(),
        open: bars.iter().map(|bar| bar.open).collect(),
        high: bars.iter().map(|bar| bar.high).collect(),
        low: bars.iter().map(|bar| bar.low).collect(),
        close: bars.iter().map(|bar| bar.close).collect(),
        volume: CanonicalVolumeChunk::Int64(bars.iter().map(|bar| bar.volume).collect()),
    })];

    publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        requested_from_ms: NATIVE_FIRST_H4_MS,
        requested_to_ms: NATIVE_FIRST_H4_MS + 2 * H4_MS,
        retrieved_unix_ms: RETRIEVED_UNIX_MS,
        returned_from_ms: bars[0].timestamp_ms,
        returned_to_ms: bars[1].timestamp_ms,
        row_count: 2,
        chunks,
    })?;

    let reopened = load_canonical_timeframe(root.path(), &identity)?;
    assert_eq!(
        reopened.ohlcv().timestamp,
        Some(vec![NATIVE_FIRST_H4_MS, NATIVE_FIRST_H4_MS + H4_MS])
    );
    Ok(())
}

#[test]
fn broker_writer_preserves_native_h4_weekend_dst_gap_without_resampling() -> Result<()> {
    const NATIVE_FRIDAY_H4_OPEN_MS: i64 = 1_515_312_000_000;
    const NATIVE_WEEKEND_GAP_MS: i64 = 51 * 60 * 60 * 1_000;

    let root = TemporaryRoot::new("native_h4_weekend_gap");
    let identity = broker_identity_for(CanonicalTimeframe::H4);
    let mut bars = sample_bars();
    bars[0].timestamp_ms = NATIVE_FRIDAY_H4_OPEN_MS;
    bars[1].timestamp_ms = NATIVE_FRIDAY_H4_OPEN_MS + NATIVE_WEEKEND_GAP_MS;
    let chunks = vec![Ok(CanonicalOhlcvChunk {
        timestamp_ms: bars.iter().map(|bar| bar.timestamp_ms).collect(),
        open: bars.iter().map(|bar| bar.open).collect(),
        high: bars.iter().map(|bar| bar.high).collect(),
        low: bars.iter().map(|bar| bar.low).collect(),
        close: bars.iter().map(|bar| bar.close).collect(),
        volume: CanonicalVolumeChunk::Int64(bars.iter().map(|bar| bar.volume).collect()),
    })];

    publish_broker_trendbar_chunks(BrokerTrendbarStreamRequest {
        configured_root: root.path(),
        identity: &identity,
        expected_generation: None,
        requested_from_ms: bars[0].timestamp_ms,
        requested_to_ms: bars[1].timestamp_ms + 4 * 60 * 60 * 1_000,
        retrieved_unix_ms: RETRIEVED_UNIX_MS,
        returned_from_ms: bars[0].timestamp_ms,
        returned_to_ms: bars[1].timestamp_ms,
        row_count: 2,
        chunks,
    })?;

    let reopened = load_canonical_timeframe(root.path(), &identity)?;
    assert_eq!(
        reopened.ohlcv().timestamp,
        Some(vec![
            NATIVE_FRIDAY_H4_OPEN_MS,
            NATIVE_FRIDAY_H4_OPEN_MS + NATIVE_WEEKEND_GAP_MS,
        ])
    );
    Ok(())
}

#[test]
fn broker_writer_cas_rejects_stale_generation_without_orphaning_output() -> Result<()> {
    let root = TemporaryRoot::new("cas");
    let identity = broker_identity();
    let bars = sample_bars();

    let first = publish(root.path(), &identity, &bars, None)?;
    let mut updated = bars.clone();
    updated[1].close = 1.21;
    let second = publish(root.path(), &identity, &updated, Some(first.generation()))?;
    assert_eq!(second.previous_generation(), Some(first.generation()));
    let generation_count_before_conflict = generation_count(root.path(), &identity)?;

    let mut stale = bars;
    stale[1].close = 1.22;
    let error = publish(root.path(), &identity, &stale, Some(first.generation()))
        .expect_err("stale broker publication must fail CAS");

    assert!(format!("{error:#}").contains("generation conflict"));
    assert_eq!(
        generation_count(root.path(), &identity)?,
        generation_count_before_conflict
    );
    assert_eq!(
        read_current_manifest(root.path(), &identity)?.generation_id(),
        second.generation()
    );
    assert_eq!(
        load_canonical_timeframe(root.path(), &identity)?
            .ohlcv()
            .close[1]
            .to_bits(),
        updated[1].close.to_bits()
    );
    Ok(())
}

fn generation_count(root: &Path, identity: &CanonicalDatasetIdentity) -> Result<usize> {
    Ok(root
        .join(identity.to_path_component())
        .read_dir()?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("g1-"))
        .count())
}
