use std::error::Error;

#[cfg(not(target_arch = "wasm32"))]
use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::io;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, LazyLock, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use vortex_array::ToCanonical;
#[cfg(not(target_arch = "wasm32"))]
use vortex_array::dtype::{DType, PType};
#[cfg(not(target_arch = "wasm32"))]
use vortex_array::scalar_fn::session::ScalarFnSession;
#[cfg(not(target_arch = "wasm32"))]
use vortex_array::session::ArraySession;
#[cfg(not(target_arch = "wasm32"))]
use vortex_array::stream::ArrayStreamExt;
#[cfg(not(target_arch = "wasm32"))]
use vortex_file::OpenOptionsSessionExt;
#[cfg(not(target_arch = "wasm32"))]
use vortex_io::runtime::BlockingRuntime;
#[cfg(not(target_arch = "wasm32"))]
use vortex_io::runtime::current::CurrentThreadRuntime;
#[cfg(not(target_arch = "wasm32"))]
use vortex_io::session::{RuntimeSession, RuntimeSessionExt};
#[cfg(not(target_arch = "wasm32"))]
use vortex_layout::session::LayoutSession;
#[cfg(not(target_arch = "wasm32"))]
use vortex_session::VortexSession;

#[cfg(not(target_arch = "wasm32"))]
static VORTEX_RUNTIME: LazyLock<CurrentThreadRuntime> = LazyLock::new(CurrentThreadRuntime::new);

#[cfg(not(target_arch = "wasm32"))]
static VORTEX_SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let mut session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<LayoutSession>()
        .with::<ScalarFnSession>()
        .with::<RuntimeSession>()
        .with_handle(VORTEX_RUNTIME.handle());
    vortex_file::register_default_encodings(&mut session);
    session
});

#[cfg(not(target_arch = "wasm32"))]
static VORTEX_CANDLE_CACHE: LazyLock<Mutex<HashMap<PathBuf, Arc<Candles>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Clone, Copy)]
pub struct CandleFieldFlags {
    pub open: bool,
    pub high: bool,
    pub low: bool,
    pub close: bool,
    pub volume: bool,
}

#[derive(Debug, Clone)]
pub struct Candles {
    pub timestamp: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub fields: CandleFieldFlags,
    pub hl2: Vec<f64>,
    pub hlc3: Vec<f64>,
    pub ohlc4: Vec<f64>,
    pub hlcc4: Vec<f64>,
}

impl Candles {
    pub fn new(
        timestamp: Vec<i64>,
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
    ) -> Self {
        let mut candles = Candles {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            fields: CandleFieldFlags {
                open: true,
                high: true,
                low: true,
                close: true,
                volume: true,
            },
            hl2: Vec::new(),
            hlc3: Vec::new(),
            ohlc4: Vec::new(),
            hlcc4: Vec::new(),
        };

        candles.precompute_fields();

        candles
    }

    pub fn new_with_fields(
        timestamp: Vec<i64>,
        open: Vec<f64>,
        high: Vec<f64>,
        low: Vec<f64>,
        close: Vec<f64>,
        volume: Vec<f64>,
        fields: CandleFieldFlags,
    ) -> Self {
        let mut candles = Candles {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            fields,
            hl2: Vec::new(),
            hlc3: Vec::new(),
            ohlc4: Vec::new(),
            hlcc4: Vec::new(),
        };

        candles.precompute_fields();

        candles
    }

    pub fn get_timestamp(&self) -> Result<&[i64], Box<dyn Error>> {
        Ok(&self.timestamp)
    }

    fn compute_hl2(&self) -> Vec<f64> {
        self.high
            .iter()
            .zip(self.low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect()
    }

    fn compute_hlc3(&self) -> Vec<f64> {
        self.high
            .iter()
            .zip(self.low.iter())
            .zip(self.close.iter())
            .map(|((&h, &l), &c)| (h + l + c) / 3.0)
            .collect()
    }

    fn compute_ohlc4(&self) -> Vec<f64> {
        self.open
            .iter()
            .zip(self.high.iter())
            .zip(self.low.iter())
            .zip(self.close.iter())
            .map(|(((&o, &h), &l), &c)| (o + h + l + c) / 4.0)
            .collect()
    }

    fn compute_hlcc4(&self) -> Vec<f64> {
        self.high
            .iter()
            .zip(self.low.iter())
            .zip(self.close.iter())
            .map(|((&h, &l), &c)| (h + l + 2.0 * c) / 4.0)
            .collect()
    }

    pub fn get_calculated_field(&self, field: &str) -> Result<&[f64], Box<dyn std::error::Error>> {
        match field.to_lowercase().as_str() {
            "hl2" => Ok(&self.hl2),
            "hlc3" => Ok(&self.hlc3),
            "ohlc4" => Ok(&self.ohlc4),
            "hlcc4" => Ok(&self.hlcc4),
            _ => Err(format!("Invalid calculated field: {}", field).into()),
        }
    }

    pub fn select_candle_field(&self, field: &str) -> Result<&[f64], Box<dyn std::error::Error>> {
        match field.to_lowercase().as_str() {
            "open" => Ok(&self.open),
            "high" => Ok(&self.high),
            "low" => Ok(&self.low),
            "close" => Ok(&self.close),
            "volume" => Ok(&self.volume),
            _ => Err(format!("Invalid field: {}", field).into()),
        }
    }

    fn precompute_fields(&mut self) {
        let len = self.high.len();
        let mut hl2 = Vec::with_capacity(len);
        let mut hlc3 = Vec::with_capacity(len);
        let mut ohlc4 = Vec::with_capacity(len);
        let mut hlcc4 = Vec::with_capacity(len);

        for i in 0..len {
            let o = self.open[i];
            let h = self.high[i];
            let l = self.low[i];
            let c = self.close[i];

            hl2.push((h + l) / 2.0);
            hlc3.push((h + l + c) / 3.0);
            ohlc4.push((o + h + l + c) / 4.0);
            hlcc4.push((h + l + 2.0 * c) / 4.0);
        }

        self.hl2 = hl2;
        self.hlc3 = hlc3;
        self.ohlc4 = ohlc4;
        self.hlcc4 = hlcc4;
    }
}

/// Read the immutable native test/benchmark dataset directly from Vortex.
///
/// VectorTA is a compute library, not an import boundary. User-provided CSV,
/// TSV, JSON, Parquet, or Arrow data is converted once by NeoEthos' admitted
/// importer; indicator tests and benchmarks reopen only the resulting Vortex
/// generation. No source-format parsing or implicit f32 widening happens here.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_candles_from_vortex(file_path: &str) -> Result<Arc<Candles>, Box<dyn Error>> {
    let key = PathBuf::from(file_path);
    let mut cache = VORTEX_CANDLE_CACHE.lock().map_err(|_| {
        io::Error::other("immutable Vortex candle cache was poisoned by a prior panic")
    })?;
    if let Some(candles) = cache.get(&key) {
        return Ok(Arc::clone(candles));
    }

    // Hold the small cache lock through the first decode. This is deliberate:
    // concurrent test workers must not stampede the single Vortex runtime and
    // decode the same immutable generation dozens of times.
    let candles = Arc::new(read_candles_from_vortex_uncached(file_path)?);
    cache.insert(key, Arc::clone(&candles));
    Ok(candles)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_candles_from_vortex_uncached(file_path: &str) -> Result<Candles, Box<dyn Error>> {
    let vortex_file =
        VORTEX_RUNTIME.block_on(VORTEX_SESSION.open_options().open_path(file_path))?;
    let stream = vortex_file.scan()?.into_array_stream()?;
    let array = VORTEX_RUNTIME.block_on(stream.read_all())?;
    if !matches!(array.dtype(), DType::Struct(..)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Vortex candle fixture must be a struct, got {}",
                array.dtype()
            ),
        )
        .into());
    }
    let fields = array.to_struct();

    let field = |name: &str| -> Result<vortex_array::ArrayRef, Box<dyn Error>> {
        Ok(fields.unmasked_field_by_name(name)?.clone())
    };
    let f64_field = |name: &str| -> Result<Vec<f64>, Box<dyn Error>> {
        let array = field(name)?;
        if !matches!(array.dtype(), DType::Primitive(PType::F64, _)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Vortex candle column '{name}' must be physical f64, got {}",
                    array.dtype()
                ),
            )
            .into());
        }
        if !array.all_valid()? {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Vortex candle column '{name}' contains nulls"),
            )
            .into());
        }
        Ok(array.to_primitive().as_slice::<f64>().to_vec())
    };

    let timestamp = field("timestamp")?;
    if !matches!(timestamp.dtype(), DType::Primitive(PType::I64, _)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Vortex candle timestamp must be physical i64 milliseconds, got {}",
                timestamp.dtype()
            ),
        )
        .into());
    }
    if !timestamp.all_valid()? {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Vortex candle timestamp contains nulls",
        )
        .into());
    }

    let candles = Candles::new(
        timestamp.to_primitive().as_slice::<i64>().to_vec(),
        f64_field("open")?,
        f64_field("high")?,
        f64_field("low")?,
        f64_field("close")?,
        f64_field("volume")?,
    );
    let rows = candles.close.len();
    if rows == 0
        || candles.timestamp.len() != rows
        || candles.open.len() != rows
        || candles.high.len() != rows
        || candles.low.len() != rows
        || candles.volume.len() != rows
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Vortex candle fixture has empty or mismatched column lengths",
        )
        .into());
    }
    if candles
        .timestamp
        .windows(2)
        .any(|window| window[0] >= window[1])
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Vortex candle timestamps must be strictly increasing i64 milliseconds",
        )
        .into());
    }
    Ok(candles)
}

pub fn source_type<'a>(candles: &'a Candles, source: &str) -> &'a [f64] {
    if source.eq_ignore_ascii_case("open") {
        &candles.open
    } else if source.eq_ignore_ascii_case("high") {
        &candles.high
    } else if source.eq_ignore_ascii_case("low") {
        &candles.low
    } else if source.eq_ignore_ascii_case("close") {
        &candles.close
    } else if source.eq_ignore_ascii_case("volume") {
        &candles.volume
    } else if source.eq_ignore_ascii_case("hl2") {
        &candles.hl2
    } else if source.eq_ignore_ascii_case("hlc3") {
        &candles.hlc3
    } else if source.eq_ignore_ascii_case("ohlc4") {
        &candles.ohlc4
    } else if source.eq_ignore_ascii_case("hlcc4") || source.eq_ignore_ascii_case("hlcc") {
        &candles.hlcc4
    } else {
        eprintln!("Warning: Invalid price source '{source}'. Defaulting to 'close'.");
        &candles.close
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn test_vortex_fixture_is_decoded_once_across_concurrent_readers() {
        let source = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        let copy = std::env::temp_dir().join(format!(
            "vector-ta-vortex-cache-{}-{unique}.vortex",
            std::process::id()
        ));
        std::fs::copy(source, &copy).expect("copy the Vortex fixture for an isolated cache key");

        let readers = 16;
        let barrier = Arc::new(Barrier::new(readers));
        let path = copy.to_string_lossy().into_owned();
        let mut handles = Vec::with_capacity(readers);
        for _ in 0..readers {
            let barrier = Arc::clone(&barrier);
            let path = path.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                read_candles_from_vortex(&path).expect("concurrent Vortex fixture read")
            }));
        }

        let first = handles
            .remove(0)
            .join()
            .expect("first fixture reader must not panic");
        for handle in handles {
            let candle_handle = handle.join().expect("fixture reader must not panic");
            assert!(
                Arc::ptr_eq(&first, &candle_handle),
                "immutable fixture readers must share one decoded allocation"
            );
        }
        std::fs::remove_file(&copy).expect("remove the isolated Vortex cache fixture");
    }

    #[test]
    fn test_vortex_fixture_round_trip() {
        let candles = read_candles_from_vortex("src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex")
            .expect("load the immutable Vortex candle fixture");

        assert_eq!(candles.timestamp.len(), 15_577);
        assert_eq!(candles.timestamp.first(), Some(&1_500_854_400_000));
        assert_eq!(candles.timestamp.last(), Some(&1_725_148_800_000));
        assert_eq!(candles.close.last(), Some(&58_655.0));
        assert_eq!(candles.hlcc4.last(), Some(&58_711.25));
    }

    #[test]
    fn test_field_congruency() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles =
            read_candles_from_vortex(file_path).expect("failed to load Vortex test fixture");

        let len = candles.timestamp.len();
        assert_eq!(candles.open.len(), len, "Open length mismatch");
        assert_eq!(candles.high.len(), len, "High length mismatch");
        assert_eq!(candles.low.len(), len, "Low length mismatch");
        assert_eq!(candles.close.len(), len, "Close length mismatch");
        assert_eq!(candles.volume.len(), len, "Volume length mismatch");
    }

    #[test]
    fn test_calculated_fields_accuracy() {
        let file_path = "src/data/2018-09-01-2024-Bitfinex_Spot-4h.vortex";
        let candles =
            read_candles_from_vortex(file_path).expect("failed to load Vortex test fixture");

        let hl2 = candles
            .get_calculated_field("hl2")
            .expect("Failed to get HL2");
        let hlc3 = candles
            .get_calculated_field("hlc3")
            .expect("Failed to get HLC3");
        let ohlc4 = candles
            .get_calculated_field("ohlc4")
            .expect("Failed to get OHLC4");
        let hlcc4 = candles
            .get_calculated_field("hlcc4")
            .expect("Failed to get HLCC4");

        let len = candles.timestamp.len();
        assert_eq!(hl2.len(), len, "HL2 length mismatch");
        assert_eq!(hlc3.len(), len, "HLC3 length mismatch");
        assert_eq!(ohlc4.len(), len, "OHLC4 length mismatch");
        assert_eq!(hlcc4.len(), len, "HLCC4 length mismatch");

        let expected_last_5_hl2 = [59166.0, 59244.5, 59118.0, 59146.5, 58767.5];
        let expected_last_5_hlc3 = [59205.7, 59223.3, 59091.7, 59149.3, 58730.0];
        let expected_last_5_ohlc4 = [59221.8, 59238.8, 59114.3, 59121.8, 58836.3];
        let expected_last_5_hlcc4 = [59225.5, 59212.8, 59078.5, 59150.8, 58711.3];

        fn compare_last_five(actual: &[f64], expected: &[f64], field_name: &str) {
            let start = actual.len().saturating_sub(5);
            let actual_slice = &actual[start..];
            for (i, (&a, &e)) in actual_slice.iter().zip(expected.iter()).enumerate() {
                let diff = (a - e).abs();
                assert!(
                    diff < 1e-1,
                    "Mismatch in {} at last-5 index {}: expected {}, got {}",
                    field_name,
                    i,
                    e,
                    a
                );
            }
        }
        compare_last_five(hl2, &expected_last_5_hl2, "HL2");
        compare_last_five(hlc3, &expected_last_5_hlc3, "HLC3");
        compare_last_five(ohlc4, &expected_last_5_ohlc4, "OHLC4");
        compare_last_five(hlcc4, &expected_last_5_hlcc4, "HLCC4");
    }

    #[test]
    fn test_precompute_fields_direct() {
        let timestamp = vec![1, 2, 3];
        let open = vec![100.0, 200.0, 300.0];
        let high = vec![110.0, 220.0, 330.0];
        let low = vec![90.0, 180.0, 270.0];
        let close = vec![105.0, 190.0, 310.0];
        let volume = vec![1000.0, 2000.0, 3000.0];

        let candles = Candles::new(timestamp, open, high, low, close, volume);

        let hl2 = candles.get_calculated_field("hl2").unwrap();
        assert_eq!(hl2, &[100.0, 200.0, 300.0]);

        let hlc3 = candles.get_calculated_field("hlc3").unwrap();
        let expected_hlc3 = &[101.6667, 196.6667, 303.3333];
        for (actual, expected) in hlc3.iter().zip(expected_hlc3.iter()) {
            assert!((actual - expected).abs() < 1e-4);
        }

        let ohlc4 = candles.get_calculated_field("ohlc4").unwrap();
        let expected_ohlc4 = &[101.25, 197.5, 302.5];
        for (actual, expected) in ohlc4.iter().zip(expected_ohlc4.iter()) {
            assert!((actual - expected).abs() < 1e-4);
        }

        let hlcc4 = candles.get_calculated_field("hlcc4").unwrap();
        let expected_hlcc4 = &[102.5, 195.0, 305.0];
        for (actual, expected) in hlcc4.iter().zip(expected_hlcc4.iter()) {
            assert!((actual - expected).abs() < 1e-4);
        }
    }
}
