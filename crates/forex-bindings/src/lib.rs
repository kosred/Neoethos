use pyo3::prelude::*;

pub mod utils;
pub mod conformal;
pub mod core;
pub mod inference;
pub mod indicators;
pub mod executor;
pub mod monitoring;
pub mod models;
pub mod data;
pub mod search;
pub mod evaluation;

#[cfg(feature = "burn-backend")]
pub mod burn_bindings;

#[pymodule]
fn forex_bindings(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Monitoring & Management
    m.add_class::<monitoring::ConsistencyTracker>()?;
    m.add_class::<monitoring::MetaController>()?;
    m.add_class::<monitoring::ConceptDriftMonitor>()?;
    
    // Risk & Execution
    m.add_class::<executor::RiskManager>()?;
    m.add_class::<executor::OrderExecutor>()?;
    
    // Core & Hardware
    m.add_class::<core::ForexCore>()?;
    
    // Prediction & Inference
    m.add_class::<conformal::ConformalGate>()?;
    
    #[cfg(feature = "onnx")]
    m.add_class::<inference::ModelEngine>()?;
    
    // Strategy Search & Evolution
    m.add_function(wrap_pyfunction!(search::search_evolve_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(search::search_evolve_gpu_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(search::search_discovery_ohlcv, m)?)?;
    
    // Data Loading & Features
    m.add_function(wrap_pyfunction!(data::load_symbol_frames, m)?)?;
    m.add_function(wrap_pyfunction!(data::load_symbol_features, m)?)?;
    m.add_function(wrap_pyfunction!(data::load_strategy_signals, m)?)?;
    
    // Utility Indicators & Math
    m.add_function(wrap_pyfunction!(evaluation::infer_stop_target_pips_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(evaluation::fast_evaluate_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(evaluation::batch_evaluate_strategies, m)?)?;
    m.add_function(wrap_pyfunction!(evaluation::evaluate_population_talib_ohlcv, m)?)?;
    m.add_function(wrap_pyfunction!(evaluation::evaluate_population_core_py, m)?)?;
    m.add_function(wrap_pyfunction!(evaluation::trade_journal_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(evaluation::triple_barrier_labels, m)?)?;
    m.add_function(wrap_pyfunction!(evaluation::quick_backtest_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(evaluation::talib_bulk_signals_ohlcv, m)?)?;

    // Utility Functions
    m.add_function(wrap_pyfunction!(executor::compute_position_size_lots, m)?)?;
    m.add_function(wrap_pyfunction!(executor::pip_size_from_symbol, m)?)?;
    m.add_function(wrap_pyfunction!(executor::infer_pip_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(utils::rank_scores_desc, m)?)?;

    // Data Alignment & Time
    m.add_function(wrap_pyfunction!(data::derive_time_index_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(data::count_weekday_trading_days, m)?)?;
    m.add_function(wrap_pyfunction!(data::align_ffill_values_by_ns, m)?)?;
    m.add_function(wrap_pyfunction!(data::align_exact_values_by_ns, m)?)?;
    m.add_function(wrap_pyfunction!(data::align_feature_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(data::sorted_index_order, m)?)?;
    m.add_function(wrap_pyfunction!(data::aggregate_news_features, m)?)?;
    m.add_function(wrap_pyfunction!(data::aggregate_news_activation, m)?)?;

    // Indicator logic
    m.add_function(wrap_pyfunction!(indicators::extract_regime_features, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::remap_labels_neutral_buy_sell, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::remap_labels_sell_neutral_buy, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::pad_probs_neutral_buy_sell, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::margins_to_probs, m)?)?;
    m.add_function(wrap_pyfunction!(utils::probs_to_signals, m)?)?;
    m.add_function(wrap_pyfunction!(utils::threshold_signals_and_accuracy, m)?)?;
    m.add_function(wrap_pyfunction!(evaluation::aggregate_prop_score_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(utils::balanced_class_weights, m)?)?;
    m.add_function(wrap_pyfunction!(utils::sample_weights_from_labels, m)?)?;
    m.add_function(wrap_pyfunction!(utils::sort_rows_with_labels_by_index, m)?)?;
    m.add_function(wrap_pyfunction!(utils::sort_dedup_rows_by_index, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::causal_tanh_zscore_py, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::detect_divergence_py, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::vortex_indicator_py, m)?)?;
    m.add_function(wrap_pyfunction!(indicators::fisher_transform_py, m)?)?;

    // Machine Learning Models
    #[cfg(feature = "lightgbm")]
    m.add_class::<models::LightGBMModel>()?;
    #[cfg(feature = "xgboost")]
    {
        m.add_class::<models::XGBoostModel>()?;
        m.add_class::<models::XGBoostRFModel>()?;
        m.add_class::<models::XGBoostDARTModel>()?;
    }
    #[cfg(feature = "catboost")]
    {
        m.add_class::<models::CatBoostModel>()?;
        m.add_class::<models::CatBoostAltModel>()?;
    }
    m.add_class::<models::MLPModel>()?;
    m.add_class::<models::GeneticModel>()?;

    // Burn deep learning models (pure Rust)
    #[cfg(feature = "burn-backend")]
    burn_bindings::register_burn_models(m)?;

    Ok(())
}
