use neoethos_data::{Ohlcv, load_vortex, write_ohlcv_vortex, write_vortex_array};
use tempfile::tempdir;
use vortex_array::IntoArray;
use vortex_array::arrays::{PrimitiveArray, StructArray};

const BASE_MS: i64 = 1_700_000_040_000;

fn ohlcv(timestamps: Vec<i64>) -> Ohlcv {
    let len = timestamps.len();
    Ohlcv {
        timestamp: Some(timestamps),
        open: vec![1.1; len],
        high: vec![1.2; len],
        low: vec![1.0; len],
        close: vec![1.15; len],
        volume: None,
    }
}

#[test]
fn canonical_writer_rejects_unit_inference_duplicates_and_sort_repair() {
    let root = tempdir().expect("temporary root");
    for (index, timestamps) in [
        vec![1_700_000_040, 1_700_000_100],
        vec![BASE_MS * 1_000_000, (BASE_MS + 60_000) * 1_000_000],
        vec![BASE_MS, BASE_MS],
        vec![BASE_MS + 60_000, BASE_MS],
    ]
    .into_iter()
    .enumerate()
    {
        let path = root.path().join(format!("invalid-{index}.vortex"));
        assert!(
            write_ohlcv_vortex(&path, &ohlcv(timestamps)).is_err(),
            "invalid timestamp source must not be inferred, sorted or deduplicated"
        );
        assert!(!path.exists());
    }
}

#[test]
fn canonical_reader_rejects_legacy_nanosecond_physical_values() {
    let root = tempdir().expect("temporary root");
    let path = root.path().join("legacy-ns.vortex");
    write_raw_vortex(
        &path,
        vec![BASE_MS * 1_000_000, (BASE_MS + 60_000) * 1_000_000],
        vec![1.1, 1.1],
    );
    assert!(load_vortex(&path).is_err());
}

#[test]
fn canonical_reader_rejects_non_positive_prices_instead_of_dropping_rows() {
    let root = tempdir().expect("temporary root");
    let path = root.path().join("zero-price.vortex");
    write_raw_vortex(&path, vec![BASE_MS, BASE_MS + 60_000], vec![0.0, 1.1]);
    assert!(load_vortex(&path).is_err());
}

fn write_raw_vortex(path: &std::path::Path, timestamps: Vec<i64>, open: Vec<f64>) {
    let len = timestamps.len();
    let array = StructArray::from_fields(&[
        (
            "timestamp",
            PrimitiveArray::from_iter(timestamps).into_array(),
        ),
        ("open", PrimitiveArray::from_iter(open).into_array()),
        (
            "high",
            PrimitiveArray::from_iter(vec![1.2_f64; len]).into_array(),
        ),
        (
            "low",
            PrimitiveArray::from_iter(vec![1.0_f64; len]).into_array(),
        ),
        (
            "close",
            PrimitiveArray::from_iter(vec![1.15_f64; len]).into_array(),
        ),
    ])
    .expect("raw Vortex fixture")
    .into_array();
    write_vortex_array(path, array).expect("raw fixture write");
}
