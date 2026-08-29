#![cfg(feature = "cuda-build-native")]

//! The f64 half of the device vocabulary.
//!
//! # Why this file exists
//!
//! `device_types.rs` contains, verbatim, zero occurrences of the token `f64`.
//! Every shared device view it defines — `CudaDeviceSliceF32Ref`,
//! `CudaDeviceMatrixF32Ref`, `CudaDeviceOhlcvRef`, … — is f32. The dispatch
//! layer built on top of it (`IndicatorCudaDataRef`, `IndicatorCudaSeries`) is
//! therefore f32 end to end, which is why every caller that goes through
//! `compute_cuda` / `compute_cuda_device` gets f32 results no matter how many
//! `*_f64` kernels the crate ships.
//!
//! This module adds the mirror-image f64 vocabulary. It is strictly ADDITIVE:
//! nothing in `device_types.rs` is changed or removed, because 180 wrappers
//! and the whole generated f32 dispatcher depend on those types.
//!
//! # Why f64 and not f32
//!
//! Indicator values feed a threshold comparison of the form
//! `combined >= long_threshold`. A one-ULP move flips a trade, so precision
//! here is not a quality knob — it decides which trades exist. The f32 lane
//! was measured 54% wrong at 200k bars on the backtest, and every CPU
//! indicator in this crate returns `f64`, so f64 is the reference on both
//! sides and the only lane whose result can be compared to it.
//!
//! # Ownership model, copied deliberately from the f32 side
//!
//! * `CudaDeviceVectorF64` / `CudaDeviceMatrixF64` OWN device memory. They are
//!   `CudaDeviceVector<f64>` / `CudaDeviceMatrix<f64>` — the generic types in
//!   `device_types.rs` were already generic over `T: DeviceCopy`, so the
//!   allocation, lifetime and context handling are shared code, not a copy.
//! * `*F64Ref` types BORROW: a raw device pointer, a length and a device id,
//!   `Copy`, no lifetime. They are validated on construction (non-null when
//!   non-empty, matching lengths, matching device) and can only be built from
//!   an owner via `as_view` or, unsafely, from raw parts.

use super::device_types::{CudaDeviceViewError, ensure_same_device, ensure_same_len};
use super::{CudaDeviceMatrix, CudaDeviceVector};

fn validate_raw_slice(device_ptr: u64, len: usize) -> Result<(), CudaDeviceViewError> {
    if len > 0 && device_ptr == 0 {
        return Err(CudaDeviceViewError::NullPointerWithNonZeroLength);
    }
    Ok(())
}

fn validate_matrix_len(rows: usize, cols: usize) -> Result<usize, CudaDeviceViewError> {
    rows.checked_mul(cols)
        .ok_or(CudaDeviceViewError::MatrixLenOverflow)
}

/// A borrowed, device-resident `[f64]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceSliceF64Ref {
    device_ptr: u64,
    len: usize,
    device_id: u32,
}

impl CudaDeviceSliceF64Ref {
    /// # Safety
    /// `device_ptr` must point to at least `len` `f64` elements resident on
    /// CUDA device `device_id`, and must outlive every use of this view.
    pub unsafe fn from_raw_parts(
        device_ptr: u64,
        len: usize,
        device_id: u32,
    ) -> Result<Self, CudaDeviceViewError> {
        validate_raw_slice(device_ptr, len)?;
        Ok(Self {
            device_ptr,
            len,
            device_id,
        })
    }

    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

/// A borrowed, device-resident row-major `f64` matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceMatrixF64Ref {
    device_ptr: u64,
    rows: usize,
    cols: usize,
    device_id: u32,
}

impl CudaDeviceMatrixF64Ref {
    /// # Safety
    /// See [`CudaDeviceSliceF64Ref::from_raw_parts`]; the element count is
    /// `rows * cols`.
    pub unsafe fn from_raw_parts(
        device_ptr: u64,
        rows: usize,
        cols: usize,
        device_id: u32,
    ) -> Result<Self, CudaDeviceViewError> {
        let len = validate_matrix_len(rows, cols)?;
        validate_raw_slice(device_ptr, len)?;
        Ok(Self {
            device_ptr,
            rows,
            cols,
            device_id,
        })
    }

    #[inline]
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows.saturating_mul(self.cols)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.device_id
    }
}

/// Owned device-resident `Vec<f64>` equivalent.
pub type CudaDeviceVectorF64 = CudaDeviceVector<f64>;
/// Owned device-resident row-major `f64` matrix.
pub type CudaDeviceMatrixF64 = CudaDeviceMatrix<f64>;

impl CudaDeviceVector<f64> {
    #[inline]
    pub fn device_ptr_f64(&self) -> u64 {
        self.buffer().as_device_ptr().as_raw() as u64
    }

    pub fn as_view_f64(&self) -> CudaDeviceSliceF64Ref {
        CudaDeviceSliceF64Ref {
            device_ptr: self.device_ptr_f64(),
            len: self.len(),
            device_id: self.device_id(),
        }
    }
}

impl CudaDeviceMatrix<f64> {
    #[inline]
    pub fn device_ptr_f64(&self) -> u64 {
        self.buffer().as_device_ptr().as_raw() as u64
    }

    pub fn as_view_f64(&self) -> CudaDeviceMatrixF64Ref {
        CudaDeviceMatrixF64Ref {
            device_ptr: self.device_ptr_f64(),
            rows: self.rows(),
            cols: self.cols(),
            device_id: self.device_id(),
        }
    }
}

/// Borrowed device-resident OHLC in f64, with an optional explicit price
/// source. `prices()` resolves `source` first and falls back to `close`,
/// matching the CPU side's `source_type(candles, source)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceOhlcF64Ref {
    open: CudaDeviceSliceF64Ref,
    high: CudaDeviceSliceF64Ref,
    low: CudaDeviceSliceF64Ref,
    close: CudaDeviceSliceF64Ref,
    source: Option<CudaDeviceSliceF64Ref>,
}

impl CudaDeviceOhlcF64Ref {
    pub fn new(
        open: CudaDeviceSliceF64Ref,
        high: CudaDeviceSliceF64Ref,
        low: CudaDeviceSliceF64Ref,
        close: CudaDeviceSliceF64Ref,
        source: Option<CudaDeviceSliceF64Ref>,
    ) -> Result<Self, CudaDeviceViewError> {
        let len = close.len();
        let device_id = close.device_id();
        ensure_same_len("ohlc_f64.open", len, open.len())?;
        ensure_same_len("ohlc_f64.high", len, high.len())?;
        ensure_same_len("ohlc_f64.low", len, low.len())?;
        ensure_same_device("ohlc_f64.open", device_id, open.device_id())?;
        ensure_same_device("ohlc_f64.high", device_id, high.device_id())?;
        ensure_same_device("ohlc_f64.low", device_id, low.device_id())?;
        if let Some(source) = source {
            ensure_same_len("ohlc_f64.source", len, source.len())?;
            ensure_same_device("ohlc_f64.source", device_id, source.device_id())?;
        }
        Ok(Self {
            open,
            high,
            low,
            close,
            source,
        })
    }

    #[inline]
    pub fn open(&self) -> CudaDeviceSliceF64Ref {
        self.open
    }
    #[inline]
    pub fn high(&self) -> CudaDeviceSliceF64Ref {
        self.high
    }
    #[inline]
    pub fn low(&self) -> CudaDeviceSliceF64Ref {
        self.low
    }
    #[inline]
    pub fn close(&self) -> CudaDeviceSliceF64Ref {
        self.close
    }
    #[inline]
    pub fn source(&self) -> Option<CudaDeviceSliceF64Ref> {
        self.source
    }
    /// The single price series a price-source indicator should read.
    #[inline]
    pub fn prices(&self) -> CudaDeviceSliceF64Ref {
        self.source.unwrap_or(self.close)
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.close.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
    #[inline]
    pub fn device_id(&self) -> u32 {
        self.close.device_id()
    }
}

/// Borrowed device-resident OHLCV in f64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceOhlcvF64Ref {
    open: CudaDeviceSliceF64Ref,
    high: CudaDeviceSliceF64Ref,
    low: CudaDeviceSliceF64Ref,
    close: CudaDeviceSliceF64Ref,
    volume: CudaDeviceSliceF64Ref,
    source: Option<CudaDeviceSliceF64Ref>,
}

impl CudaDeviceOhlcvF64Ref {
    pub fn new(
        open: CudaDeviceSliceF64Ref,
        high: CudaDeviceSliceF64Ref,
        low: CudaDeviceSliceF64Ref,
        close: CudaDeviceSliceF64Ref,
        volume: CudaDeviceSliceF64Ref,
        source: Option<CudaDeviceSliceF64Ref>,
    ) -> Result<Self, CudaDeviceViewError> {
        let len = close.len();
        let device_id = close.device_id();
        ensure_same_len("ohlcv_f64.open", len, open.len())?;
        ensure_same_len("ohlcv_f64.high", len, high.len())?;
        ensure_same_len("ohlcv_f64.low", len, low.len())?;
        ensure_same_len("ohlcv_f64.volume", len, volume.len())?;
        ensure_same_device("ohlcv_f64.open", device_id, open.device_id())?;
        ensure_same_device("ohlcv_f64.high", device_id, high.device_id())?;
        ensure_same_device("ohlcv_f64.low", device_id, low.device_id())?;
        ensure_same_device("ohlcv_f64.volume", device_id, volume.device_id())?;
        if let Some(source) = source {
            ensure_same_len("ohlcv_f64.source", len, source.len())?;
            ensure_same_device("ohlcv_f64.source", device_id, source.device_id())?;
        }
        Ok(Self {
            open,
            high,
            low,
            close,
            volume,
            source,
        })
    }

    #[inline]
    pub fn open(&self) -> CudaDeviceSliceF64Ref {
        self.open
    }
    #[inline]
    pub fn high(&self) -> CudaDeviceSliceF64Ref {
        self.high
    }
    #[inline]
    pub fn low(&self) -> CudaDeviceSliceF64Ref {
        self.low
    }
    #[inline]
    pub fn close(&self) -> CudaDeviceSliceF64Ref {
        self.close
    }
    #[inline]
    pub fn volume(&self) -> CudaDeviceSliceF64Ref {
        self.volume
    }
    #[inline]
    pub fn source(&self) -> Option<CudaDeviceSliceF64Ref> {
        self.source
    }
    #[inline]
    pub fn prices(&self) -> CudaDeviceSliceF64Ref {
        self.source.unwrap_or(self.close)
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.close.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
    #[inline]
    pub fn device_id(&self) -> u32 {
        self.close.device_id()
    }
}

/// Borrowed device-resident (high, low) pair in f64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceHighLowF64Ref {
    high: CudaDeviceSliceF64Ref,
    low: CudaDeviceSliceF64Ref,
}

impl CudaDeviceHighLowF64Ref {
    pub fn new(
        high: CudaDeviceSliceF64Ref,
        low: CudaDeviceSliceF64Ref,
    ) -> Result<Self, CudaDeviceViewError> {
        ensure_same_len("high_low_f64.low", high.len(), low.len())?;
        ensure_same_device("high_low_f64.low", high.device_id(), low.device_id())?;
        Ok(Self { high, low })
    }

    #[inline]
    pub fn high(&self) -> CudaDeviceSliceF64Ref {
        self.high
    }
    #[inline]
    pub fn low(&self) -> CudaDeviceSliceF64Ref {
        self.low
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.high.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.high.is_empty()
    }
    #[inline]
    pub fn device_id(&self) -> u32 {
        self.high.device_id()
    }
}

/// Borrowed device-resident (typical price, volume) pair in f64.
///
/// Named `CloseVolume` to mirror the f32 side, but the first series is
/// whatever the CPU reference uses as the money-flow price — for `mfi` that is
/// `hlc3`, NOT close. Feeding close here computes a DIFFERENT indicator, not a
/// less precise one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaDeviceCloseVolumeF64Ref {
    close: CudaDeviceSliceF64Ref,
    volume: CudaDeviceSliceF64Ref,
}

impl CudaDeviceCloseVolumeF64Ref {
    pub fn new(
        close: CudaDeviceSliceF64Ref,
        volume: CudaDeviceSliceF64Ref,
    ) -> Result<Self, CudaDeviceViewError> {
        ensure_same_len("close_volume_f64.volume", close.len(), volume.len())?;
        ensure_same_device(
            "close_volume_f64.volume",
            close.device_id(),
            volume.device_id(),
        )?;
        Ok(Self { close, volume })
    }

    #[inline]
    pub fn close(&self) -> CudaDeviceSliceF64Ref {
        self.close
    }
    #[inline]
    pub fn volume(&self) -> CudaDeviceSliceF64Ref {
        self.volume
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.close.len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
    #[inline]
    pub fn device_id(&self) -> u32 {
        self.close.device_id()
    }
}

/// Owned device-resident OHLCV in f64 — the residency root for a frame.
///
/// One of these is uploaded once per frame and every indicator in the sweep
/// reads it in place, so there is no device→host→device round trip between
/// indicators. That round trip, not the kernel time, is the cost that makes a
/// host stage expensive.
pub struct CudaDeviceOhlcvF64 {
    pub open: CudaDeviceVectorF64,
    pub high: CudaDeviceVectorF64,
    pub low: CudaDeviceVectorF64,
    pub close: CudaDeviceVectorF64,
    pub volume: CudaDeviceVectorF64,
    pub source: Option<CudaDeviceVectorF64>,
}

impl std::fmt::Debug for CudaDeviceOhlcvF64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDeviceOhlcvF64")
            .field("len", &self.len())
            .field("device_id", &self.close.device_id())
            .field("has_source", &self.source.is_some())
            .finish()
    }
}

impl CudaDeviceOhlcvF64 {
    pub fn new(
        open: CudaDeviceVectorF64,
        high: CudaDeviceVectorF64,
        low: CudaDeviceVectorF64,
        close: CudaDeviceVectorF64,
        volume: CudaDeviceVectorF64,
        source: Option<CudaDeviceVectorF64>,
    ) -> Result<Self, CudaDeviceViewError> {
        // Validate through the borrowed view so the owned and borrowed forms
        // can never disagree about what "consistent" means.
        let _ = CudaDeviceOhlcvF64Ref::new(
            open.as_view_f64(),
            high.as_view_f64(),
            low.as_view_f64(),
            close.as_view_f64(),
            volume.as_view_f64(),
            source.as_ref().map(|s| s.as_view_f64()),
        )?;
        Ok(Self {
            open,
            high,
            low,
            close,
            volume,
            source,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.close.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }

    #[inline]
    pub fn device_id(&self) -> u32 {
        self.close.device_id()
    }

    pub fn as_view(&self) -> CudaDeviceOhlcvF64Ref {
        CudaDeviceOhlcvF64Ref {
            open: self.open.as_view_f64(),
            high: self.high.as_view_f64(),
            low: self.low.as_view_f64(),
            close: self.close.as_view_f64(),
            volume: self.volume.as_view_f64(),
            source: self.source.as_ref().map(|s| s.as_view_f64()),
        }
    }
}
