#[cfg(feature = "cuda-build-native")]
pub mod alma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod apo_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod decycler_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod dema_wrapper;
pub mod dma_wrapper;
pub mod edcf_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ehlers_itrend_wrapper;
pub mod ehlers_kama_wrapper;
pub mod ehlers_pma_wrapper;
pub mod ehma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ema_wrapper;
pub mod fwma_wrapper;
pub mod gaussian_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod highpass2_wrapper;
pub mod hwma_wrapper;
pub mod jma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod jsa_wrapper;
pub mod maaq_wrapper;
pub mod mama_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod mwdx_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod otto_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod pma_wrapper;
pub mod pwma_wrapper;
pub mod reflex_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod sama_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod sgf_wrapper;
pub mod smma_wrapper;
pub mod sqwma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod srwma_wrapper;
pub mod swma_wrapper;
pub mod tema_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod tilson_wrapper;
pub mod trima_wrapper;
pub mod uma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vama_wrapper;
pub mod vwap_wrapper;
pub mod vwma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod wilders_wrapper;

pub mod buff_averages_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod cora_wave_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod correlation_cycle_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod cwma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ehlers_ecema_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod epma_wrapper;
pub mod frama_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod highpass_wrapper;
pub mod hma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod kama_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod linearreg_intercept_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod linearreg_slope_wrapper;
pub mod linreg_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ma_selector;
#[cfg(feature = "cuda-build-native")]
pub mod mab_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod macz_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod nama_wrapper;
pub mod nma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod ott_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod rsmk_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod sinwma_wrapper;
pub mod sma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod supersmoother_3_pole_wrapper;
pub mod supersmoother_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod tradjema_wrapper;
pub mod trendflex_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod trix_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod tsf_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vidya_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vlma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod volume_adjusted_ma_wrapper;
pub mod vpwma_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod vram_ma;
#[cfg(feature = "cuda-build-native")]
pub mod wclprice_wrapper;
#[cfg(feature = "cuda-build-native")]
pub mod wma_wrapper;
pub mod zlema_wrapper;

pub use alma_wrapper::{CudaAlma, DeviceArrayF32};
#[cfg(feature = "cuda-build-native")]
pub use apo_wrapper::{CudaApo, CudaApoError};
pub use buff_averages_wrapper::{CudaBuffAverages, CudaBuffAveragesError};
#[cfg(feature = "cuda-build-native")]
pub use cora_wave_wrapper::{CudaCoraWave, CudaCoraWaveError};
#[cfg(feature = "cuda-build-native")]
pub use correlation_cycle_wrapper::{
    BatchKernelPolicy as CorrelationCycleBatchKernelPolicy, CudaCorrelationCycle,
    CudaCorrelationCyclePolicy, DeviceCorrelationCycleQuad,
    ManySeriesKernelPolicy as CorrelationCycleManySeriesKernelPolicy,
};
#[cfg(feature = "cuda-build-native")]
pub use cwma_wrapper::{
    BatchKernelPolicy, BatchThreadsPerOutput, CudaCwma, CudaCwmaPolicy, ManySeriesKernelPolicy,
};
#[cfg(feature = "cuda-build-native")]
pub use decycler_wrapper::{CudaDecycler, CudaDecyclerError};
#[cfg(feature = "cuda-build-native")]
pub use dema_wrapper::{CudaDema, CudaDemaError};
pub use dma_wrapper::CudaDma;
pub use edcf_wrapper::CudaEdcf;
#[cfg(feature = "cuda-build-native")]
pub use ehlers_ecema_wrapper::CudaEhlersEcema;
#[cfg(feature = "cuda-build-native")]
pub use ehlers_itrend_wrapper::{
    BatchKernelPolicy as EhlersItrendBatchKernelPolicy,
    BatchThreadsPerOutput as EhlersItrendBatchThreadsPerOutput, CudaEhlersITrend,
    CudaEhlersITrendError, CudaEhlersITrendPolicy,
    ManySeriesKernelPolicy as EhlersItrendManySeriesKernelPolicy,
};
pub use ehlers_kama_wrapper::CudaEhlersKama;
pub use ehlers_pma_wrapper::{CudaEhlersPma, DeviceEhlersPmaPair};
pub use ehma_wrapper::CudaEhma;
#[cfg(feature = "cuda-build-native")]
pub use ema_wrapper::{CudaEma, CudaEmaError};
#[cfg(feature = "cuda-build-native")]
pub use epma_wrapper::CudaEpma;
pub use frama_wrapper::{CudaFrama, CudaFramaError};
pub use fwma_wrapper::CudaFwma;
pub use gaussian_wrapper::CudaGaussian;
#[cfg(feature = "cuda-build-native")]
pub use highpass_wrapper::CudaHighpass;
#[cfg(feature = "cuda-build-native")]
pub use highpass2_wrapper::{CudaHighPass2, CudaHighPass2Error};
pub use hma_wrapper::{CudaHma, CudaHmaError};
pub use hwma_wrapper::CudaHwma;
pub use jma_wrapper::CudaJma;
#[cfg(feature = "cuda-build-native")]
pub use jsa_wrapper::{CudaJsa, CudaJsaError};
#[cfg(feature = "cuda-build-native")]
pub use kama_wrapper::CudaKama;
#[cfg(feature = "cuda-build-native")]
pub use linearreg_intercept_wrapper::{CudaLinregIntercept, CudaLinregInterceptError};
#[cfg(feature = "cuda-build-native")]
pub use linearreg_slope_wrapper::{CudaLinearregSlope, CudaLinearregSlopeError};
pub use linreg_wrapper::{CudaLinreg, CudaLinregError};
#[cfg(feature = "cuda-build-native")]
pub use ma_selector::{CudaMaData, CudaMaDeviceDataRef, CudaMaSelector, CudaMaSelectorError};
pub use maaq_wrapper::CudaMaaq;
#[cfg(feature = "cuda-build-native")]
pub use mab_wrapper::{CudaMab, CudaMabError};
pub use mama_wrapper::{CudaMama, DeviceMamaPair};
#[cfg(feature = "cuda-build-native")]
pub use mwdx_wrapper::{CudaMwdx, CudaMwdxError};
#[cfg(feature = "cuda-build-native")]
pub use nama_wrapper::CudaNama;
pub use nma_wrapper::{CudaNma, CudaNmaError};
#[cfg(feature = "cuda-build-native")]
pub use otto_wrapper::{CudaOtto, CudaOttoError, CudaOttoPolicy};
#[cfg(feature = "cuda-build-native")]
pub use pma_wrapper::{CudaPma, CudaPmaError, DevicePmaPair, benches as pma_benches};
pub use pwma_wrapper::CudaPwma;
pub use reflex_wrapper::CudaReflex;
#[cfg(feature = "cuda-build-native")]
pub use rsmk_wrapper::{CudaRsmk, CudaRsmkError};
#[cfg(feature = "cuda-build-native")]
pub use sama_wrapper::{CudaSama, CudaSamaError};
#[cfg(feature = "cuda-build-native")]
pub use sgf_wrapper::CudaSgf;
#[cfg(feature = "cuda-build-native")]
pub use sinwma_wrapper::CudaSinwma;
pub use sma_wrapper::{CudaSma, CudaSmaError};
pub use smma_wrapper::CudaSmma;
pub use sqwma_wrapper::CudaSqwma;
#[cfg(feature = "cuda-build-native")]
pub use srwma_wrapper::{CudaSrwma, CudaSrwmaError};
#[cfg(feature = "cuda-build-native")]
pub use supersmoother_3_pole_wrapper::CudaSupersmoother3Pole;
pub use supersmoother_wrapper::{CudaSuperSmoother, CudaSuperSmootherError};
pub use swma_wrapper::CudaSwma;
pub use tema_wrapper::CudaTema;
#[cfg(feature = "cuda-build-native")]
pub use tilson_wrapper::{CudaTilson, CudaTilsonError};
#[cfg(feature = "cuda-build-native")]
pub use tradjema_wrapper::CudaTradjema;
pub use trendflex_wrapper::{CudaTrendflex, CudaTrendflexError};
pub use trima_wrapper::CudaTrima;
pub use uma_wrapper::{
    BatchKernelPolicy as UmaBatchKernelPolicy, CudaUma, CudaUmaPolicy,
    ManySeriesKernelPolicy as UmaManySeriesKernelPolicy,
};
#[cfg(feature = "cuda-build-native")]
pub use vama_wrapper::{
    BatchKernelPolicy as VamaBatchKernelPolicy, CudaVama, CudaVamaError, CudaVamaPolicy,
    ManySeriesKernelPolicy as VamaManySeriesKernelPolicy,
};
#[cfg(feature = "cuda-build-native")]
pub use vidya_wrapper::{CudaVidya, CudaVidyaError};
#[cfg(feature = "cuda-build-native")]
pub use volume_adjusted_ma_wrapper::{
    CudaVama as CudaVolumeAdjustedMa, CudaVamaError as CudaVolumeAdjustedMaError,
};
pub use vpwma_wrapper::{CudaVpwma, CudaVpwmaError};
pub use vwap_wrapper::CudaVwap;
pub use vwma_wrapper::CudaVwma;
#[cfg(feature = "cuda-build-native")]
pub use wclprice_wrapper::CudaWclprice;
#[cfg(feature = "cuda-build-native")]
pub use wilders_wrapper::{CudaWilders, CudaWildersError};
#[cfg(feature = "cuda-build-native")]
pub use wma_wrapper::{CudaWma, CudaWmaError};
pub use zlema_wrapper::{CudaZlema, CudaZlemaError};

#[cfg(feature = "cuda-build-native")]
pub use macz_wrapper::{CudaMacz, CudaMaczError};
#[cfg(feature = "cuda-build-native")]
pub use ott_wrapper::{CudaOtt, CudaOttError, benches as ott_benches};
#[cfg(feature = "cuda-build-native")]
pub use trix_wrapper::{CudaTrix, CudaTrixError};
pub use tsf_wrapper::{CudaTsf, CudaTsfError};
#[cfg(feature = "cuda-build-native")]
pub use vlma_wrapper::{CudaVlma, CudaVlmaError};
