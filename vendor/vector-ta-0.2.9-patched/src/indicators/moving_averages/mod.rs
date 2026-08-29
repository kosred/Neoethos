pub mod alma;
pub mod buff_averages;
pub mod corrected_moving_average;
pub mod cwma;
pub mod dema;
pub mod dma;
pub mod edcf;
pub mod ehlers_ecema;
pub mod ehlers_itrend;
pub mod ehlers_kama;
pub mod ehlers_pma;
pub mod ehlers_undersampled_double_moving_average;
pub mod ehma;
pub mod elastic_volume_weighted_moving_average;
pub mod ema;
pub mod ema_deviation_corrected_t3;
pub mod epma;
pub mod frama;
pub mod fwma;
pub mod gaussian;
pub mod highpass;
pub mod highpass_2_pole;
pub mod hma;
pub mod hwma;
pub mod jma;
pub mod jsa;
pub mod kama;
pub mod linreg;
pub mod logarithmic_moving_average;
pub mod ma;
pub mod ma_batch;
pub mod ma_stream;
pub mod maaq;
pub mod mama;
pub mod mwdx;
pub mod n_order_ema;
pub mod nama;
pub mod nma;
pub mod param_schema;
pub mod pwma;
pub mod reflex;
pub mod registry;
pub mod sama;
pub mod sgf;
pub mod sinwma;
pub mod sma;
pub mod smma;
pub mod sqwma;
pub mod srwma;
pub mod supersmoother;
pub mod supersmoother_3_pole;
pub mod swma;
pub mod tema;
pub mod tilson;
pub mod tradjema;
pub mod trendflex;
pub mod trima;
pub mod uma;
pub mod volatility_adjusted_ma;
pub mod volume_adjusted_ma;
pub mod vpwma;
pub mod vwap;
pub mod vwma;
pub mod wave_smoother;
pub mod wilders;
pub mod wma;
pub mod zlema;

pub use corrected_moving_average::corrected_moving_average_into;
pub use corrected_moving_average::{
    CorrectedMovingAverageBatchBuilder, CorrectedMovingAverageBatchOutput,
    CorrectedMovingAverageBatchRange, CorrectedMovingAverageBuilder, CorrectedMovingAverageData,
    CorrectedMovingAverageError, CorrectedMovingAverageInput, CorrectedMovingAverageOutput,
    CorrectedMovingAverageParams, CorrectedMovingAverageStream, corrected_moving_average,
    corrected_moving_average_batch_par_slice, corrected_moving_average_batch_slice,
    corrected_moving_average_batch_with_kernel, corrected_moving_average_into_slice,
    corrected_moving_average_with_kernel, expand_grid_corrected_moving_average,
};
pub use cwma::{CwmaInput, CwmaOutput, CwmaParams, cwma};
pub use dma::{
    DmaBatchBuilder, DmaBatchOutput, DmaBatchRange, DmaBuilder, DmaData, DmaError, DmaInput,
    DmaOutput, DmaParams, DmaStream, dma, dma_batch_with_kernel, dma_into_slice, dma_with_kernel,
};

pub use edcf::{EdcfInput, EdcfOutput, EdcfParams, edcf};
pub use ehlers_kama::{EhlersKamaInput, EhlersKamaOutput, EhlersKamaParams, ehlers_kama};
pub use ehlers_pma::{
    EhlersPmaBatchRange, EhlersPmaInput, EhlersPmaOutput, EhlersPmaParams, ehlers_pma,
    expand_grid as expand_grid_ehlers_pma,
};
pub use ehlers_undersampled_double_moving_average::{
    EhlersUndersampledDoubleMovingAverageBatchRange, EhlersUndersampledDoubleMovingAverageInput,
    EhlersUndersampledDoubleMovingAverageOutput, EhlersUndersampledDoubleMovingAverageParams,
    ehlers_undersampled_double_moving_average,
    expand_grid_ehlers_undersampled_double_moving_average,
};
pub use elastic_volume_weighted_moving_average::{
    ElasticVolumeWeightedMovingAverageBatchRange, ElasticVolumeWeightedMovingAverageInput,
    ElasticVolumeWeightedMovingAverageOutput, ElasticVolumeWeightedMovingAverageParams,
    elastic_volume_weighted_moving_average, expand_grid_elastic_volume_weighted_moving_average,
};
pub use logarithmic_moving_average::logarithmic_moving_average_into;
pub use logarithmic_moving_average::{
    LogarithmicMovingAverageBatchBuilder, LogarithmicMovingAverageBatchOutput,
    LogarithmicMovingAverageBatchRange, LogarithmicMovingAverageBuilder,
    LogarithmicMovingAverageData, LogarithmicMovingAverageError, LogarithmicMovingAverageInput,
    LogarithmicMovingAverageOutput, LogarithmicMovingAverageParams, LogarithmicMovingAverageStream,
    expand_grid_logarithmic_moving_average, logarithmic_moving_average,
    logarithmic_moving_average_batch_par_slice, logarithmic_moving_average_batch_slice,
    logarithmic_moving_average_batch_with_kernel, logarithmic_moving_average_into_slice,
    logarithmic_moving_average_with_kernel,
};
pub use uma::{UmaInput, UmaOutput, UmaParams, uma};
pub use volatility_adjusted_ma::{
    VamaInput as VolatilityAdjustedMaInput, VamaOutput as VolatilityAdjustedMaOutput,
    VamaParams as VolatilityAdjustedMaParams, vama as volatility_adjusted_ma,
};
pub use volume_adjusted_ma::{
    VolumeAdjustedMa as volume_adjusted_ma, VolumeAdjustedMaInput, VolumeAdjustedMaOutput,
    VolumeAdjustedMaParams,
};

pub use ehma::{
    EhmaBatchBuilder, EhmaBatchOutput, EhmaBatchRange, EhmaBuilder, EhmaData, EhmaError, EhmaInput,
    EhmaOutput, EhmaParams, EhmaStream, ehma, ehma_batch_inner_into, ehma_batch_par_slice,
    ehma_batch_slice, ehma_batch_with_kernel, ehma_batch_with_kernel_slice, ehma_into_slice,
    ehma_with_kernel,
};

pub use nama::{
    NamaBatchBuilder, NamaBatchOutput, NamaBatchRange, NamaBuilder, NamaData, NamaError, NamaInput,
    NamaOutput, NamaParams, NamaStream, nama, nama_batch_with_kernel, nama_into_slice,
    nama_with_kernel,
};

pub use n_order_ema::{
    NOrderEmaBatchBuilder, NOrderEmaBatchOutput, NOrderEmaBatchRange, NOrderEmaBuilder,
    NOrderEmaData, NOrderEmaError, NOrderEmaIirStyle, NOrderEmaInput, NOrderEmaOutput,
    NOrderEmaParams, NOrderEmaStream, NOrderEmaStyle, expand_grid_n_order_ema, n_order_ema,
    n_order_ema_batch_from_input_with_kernel, n_order_ema_batch_with_kernel,
    n_order_ema_into_slice, n_order_ema_with_kernel,
};

pub use n_order_ema::n_order_ema_into;

pub use sama::{
    SamaBatchBuilder, SamaBatchOutput, SamaBatchRange, SamaBuilder, SamaData, SamaError, SamaInput,
    SamaOutput, SamaParams, SamaStream, sama, sama_batch_par_slice, sama_batch_slice,
    sama_batch_with_kernel, sama_into_slice, sama_with_kernel,
};

pub use sgf::{
    SgfBatchBuilder, SgfBatchOutput, SgfBatchRange, SgfBuilder, SgfData, SgfError, SgfInput,
    SgfOutput, SgfParams, SgfStream, expand_grid as expand_grid_sgf, sgf, sgf_batch_into_slice,
    sgf_batch_par_slice, sgf_batch_slice, sgf_batch_with_kernel, sgf_into_slice, sgf_with_kernel,
};
