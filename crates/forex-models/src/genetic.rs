// Genetic Strategy Expert
// Ported from src/forex_bot/models/genetic.py
//
// Genetic Algorithm for evolving TA-Lib indicator combinations.
// Uses OHLC data (metadata) to generate signals, not just pre-computed features.

use anyhow::{Context, Result};
use ndarray::Array2;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use std::path::Path;
use tracing::info;

/// GeneticStrategyExpert - Wrapper for Python genetic.py module
///
/// Evolutionary algorithm that discovers optimal TA-Lib indicator combinations.
/// Supports both internal evolution and loading pre-discovered strategies from JSON.
pub struct GeneticStrategyExpert {
    py_expert: Option<Py<PyAny>>,
    population_size: usize,
    generations: usize,
    max_indicators: usize,
}

impl GeneticStrategyExpert {
    fn series_to_numpy(py: Python<'_>, series: &Series) -> Result<Py<PyAny>> {
        let numpy = PyModule::import(py, "numpy")?;
        let array = match series.dtype() {
            DataType::Float32 => {
                let values: Vec<f32> = series.f32()?.into_iter().map(|v| v.unwrap_or(0.0)).collect();
                numpy.call_method1("array", (values,))?
            }
            DataType::Float64 => {
                let values: Vec<f64> = series.f64()?.into_iter().map(|v| v.unwrap_or(0.0)).collect();
                numpy.call_method1("array", (values,))?
            }
            DataType::Int8 => {
                let values: Vec<i8> = series.i8()?.into_iter().map(|v| v.unwrap_or(0)).collect();
                numpy.call_method1("array", (values,))?
            }
            DataType::Int16 => {
                let values: Vec<i16> = series.i16()?.into_iter().map(|v| v.unwrap_or(0)).collect();
                numpy.call_method1("array", (values,))?
            }
            DataType::Int32 => {
                let values: Vec<i32> = series.i32()?.into_iter().map(|v| v.unwrap_or(0)).collect();
                numpy.call_method1("array", (values,))?
            }
            DataType::Int64 => {
                let values: Vec<i64> = series.i64()?.into_iter().map(|v| v.unwrap_or(0)).collect();
                numpy.call_method1("array", (values,))?
            }
            DataType::UInt8 => {
                let values: Vec<u8> = series.u8()?.into_iter().map(|v| v.unwrap_or(0)).collect();
                numpy.call_method1("array", (values,))?
            }
            DataType::UInt16 => {
                let values: Vec<u16> = series.u16()?.into_iter().map(|v| v.unwrap_or(0)).collect();
                numpy.call_method1("array", (values,))?
            }
            DataType::UInt32 => {
                let values: Vec<u32> = series.u32()?.into_iter().map(|v| v.unwrap_or(0)).collect();
                numpy.call_method1("array", (values,))?
            }
            DataType::UInt64 => {
                let values: Vec<u64> = series.u64()?.into_iter().map(|v| v.unwrap_or(0)).collect();
                numpy.call_method1("array", (values,))?
            }
            DataType::Boolean => {
                let values: Vec<bool> = series.bool()?.into_iter().map(|v| v.unwrap_or(false)).collect();
                numpy.call_method1("array", (values,))?
            }
            _ => {
                let casted = series.cast(&DataType::Float64)?;
                let values: Vec<f64> = casted.f64()?.into_iter().map(|v| v.unwrap_or(0.0)).collect();
                numpy.call_method1("array", (values,))?
            }
        };
        Ok(array.unbind())
    }

    fn dataframe_to_bridge_frame(
        py: Python<'_>,
        genetic_module: &Bound<'_, PyModule>,
        df: &DataFrame,
        attrs: Option<Bound<'_, PyDict>>,
    ) -> Result<Py<PyAny>> {
        let frame_cls = genetic_module
            .getattr("_BridgeFrame")
            .context("_BridgeFrame helper not found in forex_bot.models.genetic")?;
        let data = PyDict::new(py);
        for col in df.get_columns() {
            let series = col.as_materialized_series();
            let array = Self::series_to_numpy(py, series)?;
            data.set_item(col.name().as_str(), array.bind(py))?;
        }
        let numpy = PyModule::import(py, "numpy")?;
        let index = numpy.call_method1("arange", (df.height(),))?;
        let frame = match attrs {
            Some(frame_attrs) => frame_cls.call1((data, index, frame_attrs))?,
            None => frame_cls.call1((data, index))?,
        };
        Ok(frame.unbind())
    }

    /// Create new Genetic Strategy Expert
    /// Python lines 15-30
    pub fn new(population_size: usize, generations: usize, max_indicators: usize) -> Result<Self> {
        let py_expert = Python::attach(|py| {
            // Import the Python module
            let genetic_module = PyModule::import(py, "forex_bot.models.genetic")
                .context("Failed to import forex_bot.models.genetic")?;

            // Get the GeneticStrategyExpert class
            let genetic_class = genetic_module
                .getattr("GeneticStrategyExpert")
                .context("GeneticStrategyExpert class not found")?;

            // Create instance with kwargs
            let kwargs = PyDict::new(py);
            kwargs.set_item("population_size", population_size)?;
            kwargs.set_item("generations", generations)?;
            kwargs.set_item("max_indicators", max_indicators)?;

            // Instantiate
            let py_expert = genetic_class.call((), Some(&kwargs))?;

            Ok::<Py<PyAny>, anyhow::Error>(py_expert.into())
        })?;

        Ok(Self {
            py_expert: Some(py_expert),
            population_size,
            generations,
            max_indicators,
        })
    }

    /// Fit the genetic model
    /// Python lines 54-203
    ///
    /// This will:
    /// 1. Try to load pre-discovered strategies from Discovery Engine JSON
    /// 2. Fall back to internal evolution if no cached strategies found
    pub fn fit(
        &mut self,
        x: &DataFrame,
        y: &Series,
        metadata: Option<&DataFrame>,
        symbol: Option<&str>,
    ) -> Result<()> {
        Python::attach(|py| {
            let genetic_module = PyModule::import(py, "forex_bot.models.genetic")
                .context("Failed to import forex_bot.models.genetic")?;
            let expert = self
                .py_expert
                .as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);
            let x_frame = Self::dataframe_to_bridge_frame(py, &genetic_module, x, None)?;
            let y_np = Self::series_to_numpy(py, y)?;

            let metadata_frame = if let Some(meta) = metadata {
                let attrs = PyDict::new(py);
                if let Some(symbol) = symbol {
                    if !symbol.trim().is_empty() {
                        attrs.set_item("symbol", symbol)?;
                    }
                }
                Some(Self::dataframe_to_bridge_frame(py, &genetic_module, meta, Some(attrs))?)
            } else {
                None
            };

            // Call fit method
            let kwargs = PyDict::new(py);
            kwargs.set_item("metadata", metadata_frame)?;
            expert.call_method("fit", (x_frame, y_np), Some(&kwargs))?;

            info!(
                "Genetic expert fitted ({} generations, population {}, max_indicators {})",
                self.generations, self.population_size, self.max_indicators
            );

            Ok(())
        })
    }

    /// Predict probabilities
    /// Python lines 204-236
    ///
    /// Returns 3-class probabilities: [Neutral, Buy, Sell]
    /// Uses portfolio voting across multiple evolved strategies
    pub fn predict_proba(
        &self,
        x: &DataFrame,
        metadata: Option<&DataFrame>,
        symbol: Option<&str>,
    ) -> Result<Array2<f32>> {
        Python::attach(|py| {
            let genetic_module = PyModule::import(py, "forex_bot.models.genetic")
                .context("Failed to import forex_bot.models.genetic")?;
            let expert = self
                .py_expert
                .as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);
            let x_frame = Self::dataframe_to_bridge_frame(py, &genetic_module, x, None)?;
            let metadata_frame = if let Some(meta) = metadata {
                let attrs = PyDict::new(py);
                if let Some(symbol) = symbol {
                    if !symbol.trim().is_empty() {
                        attrs.set_item("symbol", symbol)?;
                    }
                }
                Some(Self::dataframe_to_bridge_frame(py, &genetic_module, meta, Some(attrs))?)
            } else {
                None
            };

            // Call predict_proba
            let kwargs = PyDict::new(py);
            kwargs.set_item("metadata", metadata_frame)?;
            let result = expert.call_method("predict_proba", (x_frame,), Some(&kwargs))?;

            // Convert numpy array to ndarray
            let _numpy = PyModule::import(py, "numpy")?;
            let result_list: Vec<Vec<f32>> = result.call_method0("tolist")?.extract()?;

            let nrows = result_list.len();
            let ncols = if nrows > 0 { result_list[0].len() } else { 3 };

            let mut flat = Vec::new();
            for row in result_list {
                flat.extend(row);
            }

            let array = Array2::from_shape_vec((nrows, ncols), flat)?;

            Ok(array)
        })
    }

    /// Save the model
    /// Python lines 238-244
    pub fn save(&self, path: &Path) -> Result<()> {
        Python::attach(|py| {
            let expert = self
                .py_expert
                .as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);

            expert.call_method1("save", (path.to_string_lossy().as_ref(),))?;

            info!("Saved genetic expert to: {:?}", path);

            Ok(())
        })
    }

    /// Load the model
    /// Python lines 246-256
    pub fn load(&mut self, path: &Path) -> Result<()> {
        Python::attach(|py| {
            let expert = self
                .py_expert
                .as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);

            expert.call_method1("load", (path.to_string_lossy().as_ref(),))?;

            info!("Loaded genetic expert from: {:?}", path);

            Ok(())
        })
    }

    /// Get the number of strategies in the portfolio
    pub fn portfolio_size(&self) -> Result<usize> {
        Python::attach(|py| {
            let expert = self
                .py_expert
                .as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);

            let portfolio = expert.getattr("portfolio")?;
            let size: usize = portfolio.len()?;

            Ok(size)
        })
    }

    /// Get the best gene's fitness score
    pub fn best_fitness(&self) -> Result<f64> {
        Python::attach(|py| {
            let expert = self
                .py_expert
                .as_ref()
                .context("Genetic expert not initialized")?
                .bind(py);

            let best_gene = expert.getattr("best_gene")?;

            if best_gene.is_none() {
                return Ok(0.0);
            }

            let fitness: f64 = best_gene.getattr("fitness")?.extract()?;

            Ok(fitness)
        })
    }
}

// ============================================================================
// SUMMARY
// ============================================================================
//
// Genetic Strategy Expert - PyO3 Wrapper
//
// FEATURES:
// ✅ Wraps Python GeneticStrategyExpert via PyO3
// ✅ Loads pre-discovered strategies from Discovery Engine JSON
// ✅ Falls back to internal evolution (50 population, 10 generations)
// ✅ Portfolio voting across multiple evolved strategies
// ✅ Smooth 3-class probability output: [Neutral, Buy, Sell]
// ✅ Save/Load support via joblib
//
// BRIDGE TO DISCOVERY ENGINE:
// - Loads from cache/talib_knowledge.json
// - Supports symbol-specific caches (cache/talib_knowledge_EURUSD.json)
// - Supports opportunistic strategies (cache/talib_knowledge_opportunistic.json)
// - Validates all loaded genes before use
//
// INTERNAL EVOLUTION:
// - Generates random TA-Lib indicator combinations
// - Evaluates fitness via Sharpe ratio on historical data
// - Evolves over N generations via genetic algorithm
// - Clamps thresholds to avoid degenerate strategies
//
// This module bridges the gap between:
// 1. Python's rich TA-Lib ecosystem (50+ indicators)
// 2. Rust's performance and type safety
// 3. Pre-discovered optimal strategies from external analysis
//

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    use std::ffi::CString;

    fn run_py(py: Python<'_>, code: &str, locals: &Bound<'_, PyDict>) -> PyResult<()> {
        let code = CString::new(code).expect("python code should not contain embedded nulls");
        py.run(code.as_c_str(), Some(locals), Some(locals))
    }

    #[test]
    fn test_genetic_bridge_avoids_pandas_and_passes_frame_like_metadata() -> Result<()> {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            run_py(
                py,
                r#"
import sys, types, numpy as np
_orig_pkg = sys.modules.get("forex_bot")
_orig_models_pkg = sys.modules.get("forex_bot.models")
_orig_genetic = sys.modules.get("forex_bot.models.genetic")
_orig_pandas = sys.modules.get("pandas")

class _PandasTrap(types.SimpleNamespace):
    def __getattr__(self, name):
        raise AssertionError("pandas bridge must not be used")

sys.modules["pandas"] = _PandasTrap()

class _BridgeFrame:
    def __init__(self, data, index=None, attrs=None):
        self._data = {str(k): np.asarray(v) for k, v in data.items()}
        self.columns = list(self._data.keys())
        n = len(next(iter(self._data.values()))) if self._data else 0
        self.index = np.arange(n, dtype=np.int64) if index is None else np.asarray(index)
        self.attrs = dict(attrs or {})
    @property
    def empty(self):
        return len(self.index) <= 0
    def __len__(self):
        return int(len(self.index))
    def __getitem__(self, key):
        return self._data[str(key)]
    def copy(self):
        out = _BridgeFrame({k: np.asarray(v).copy() for k, v in self._data.items()}, np.asarray(self.index).copy(), dict(self.attrs))
        return out

class FakeGeneticStrategyExpert:
    def __init__(self, population_size=0, generations=0, max_indicators=0):
        self.population_size = population_size
        self.generations = generations
        self.max_indicators = max_indicators
    def fit(self, x, y, metadata=None):
        assert hasattr(x, "columns")
        assert hasattr(x, "index")
        assert isinstance(y, np.ndarray)
        assert hasattr(metadata, "copy")
        assert set(str(c).lower() for c in metadata.columns) >= {"open", "high", "low", "close"}
    def predict_proba(self, x, metadata=None):
        assert hasattr(x, "__getitem__")
        assert hasattr(x, "columns")
        assert metadata is None or hasattr(metadata, "__getitem__")
        return np.tile(np.array([[1/3, 1/3, 1/3]], dtype=np.float32), (len(x), 1))

fake_module = types.SimpleNamespace(
    GeneticStrategyExpert=FakeGeneticStrategyExpert,
    _BridgeFrame=_BridgeFrame,
)
pkg = sys.modules.get("forex_bot") or types.ModuleType("forex_bot")
models_pkg = sys.modules.get("forex_bot.models") or types.ModuleType("forex_bot.models")
setattr(pkg, "models", models_pkg)
setattr(models_pkg, "genetic", fake_module)
sys.modules["forex_bot"] = pkg
sys.modules["forex_bot.models"] = models_pkg
sys.modules["forex_bot.models.genetic"] = fake_module
"#,
                &locals,
            )?;

            let mut expert = GeneticStrategyExpert::new(8, 2, 3)?;
            let x = DataFrame::new(vec![
                Series::new("f1".into(), vec![1.0f64, 2.0, 3.0]).into(),
                Series::new("f2".into(), vec![4.0f64, 5.0, 6.0]).into(),
            ])?;
            let y = Series::new("label".into(), vec![1.0f64, 0.0, -1.0]);
            let meta = DataFrame::new(vec![
                Series::new("open".into(), vec![1.0f64, 1.1, 1.2]).into(),
                Series::new("high".into(), vec![1.2f64, 1.3, 1.4]).into(),
                Series::new("low".into(), vec![0.9f64, 1.0, 1.1]).into(),
                Series::new("close".into(), vec![1.1f64, 1.2, 1.3]).into(),
            ])?;

            let fit_result = expert.fit(&x, &y, Some(&meta), Some("EURUSD"));
            let probs_result = expert.predict_proba(&x, Some(&meta), Some("EURUSD"));

            let _ = run_py(
                py,
                r#"
import sys
if _orig_pkg is None:
    sys.modules.pop("forex_bot", None)
else:
    sys.modules["forex_bot"] = _orig_pkg
if _orig_models_pkg is None:
    sys.modules.pop("forex_bot.models", None)
else:
    sys.modules["forex_bot.models"] = _orig_models_pkg
if _orig_genetic is None:
    sys.modules.pop("forex_bot.models.genetic", None)
else:
    sys.modules["forex_bot.models.genetic"] = _orig_genetic
if _orig_pandas is None:
    sys.modules.pop("pandas", None)
else:
    sys.modules["pandas"] = _orig_pandas
"#,
                &locals,
            );

            fit_result?;
            let probs = probs_result?;
            assert_eq!(probs.shape(), &[3, 3]);
            Ok(())
        })
    }
}
