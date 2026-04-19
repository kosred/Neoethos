// ONNX Export for Ultra-Fast Inference
// Ported from src/forex_bot/models/onnx_exporter.py
//
// Converts trained models to ONNX format for:
// - 10-100x faster inference in production
// - Cross-platform deployment (Windows -> Linux)
// - GPU acceleration support
// - Reduced memory footprint

use anyhow::{Context, Result};
use ndarray::Array2;
#[cfg(feature = "onnx-export-bridge")]
use pyo3::prelude::*;
#[cfg(feature = "onnx-export-bridge")]
use pyo3::types::{PyDict, PyModule};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
#[cfg(feature = "onnx-export-bridge")]
use tracing::{info, warn};

#[cfg(feature = "onnx-export-bridge")]
use crate::registry::get_model_capability;
#[cfg(feature = "onnx-export-bridge")]
use crate::runtime::capabilities::ModelFamily;

// ============================================================================
// ONNX EXPORT METADATA
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    pub path: PathBuf,
    pub classes: Option<Vec<i32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportManifest {
    pub models: HashMap<String, ExportMetadata>,
}

// ============================================================================
// ONNX EXPORTER - PYTHON WRAPPER
// ============================================================================

/// ONNXExporter - Wraps Python onnx_exporter.py for PyTorch model export
/// Tree models (LightGBM, XGBoost, CatBoost) export directly via Rust
pub struct ONNXExporter {
    #[cfg(feature = "onnx-export-bridge")]
    py_exporter: Option<Py<PyAny>>,
    #[cfg_attr(not(feature = "onnx-export-bridge"), allow(dead_code))]
    models_dir: PathBuf,
    #[cfg_attr(not(feature = "onnx-export-bridge"), allow(dead_code))]
    onnx_dir: PathBuf,
}

impl ONNXExporter {
    #[cfg(feature = "onnx-export-bridge")]
    fn resolve_export_route(&self, name: &str, model: &Bound<'_, PyAny>) -> Result<ExportRoute> {
        if let Some(capability) = get_model_capability(name) {
            return match capability.family {
                ModelFamily::Tree => {
                    if capability.name.contains("lightgbm") || capability.name.contains("lgbm") {
                        Ok(ExportRoute::LightGbm)
                    } else if capability.name.contains("xgboost") || capability.name.contains("xgb")
                    {
                        Ok(ExportRoute::XgBoost)
                    } else if capability.name.contains("catboost")
                        || capability.name.contains("cat")
                    {
                        Ok(ExportRoute::CatBoost)
                    } else {
                        Err(anyhow::anyhow!(
                            "tree model `{}` has no ONNX export route",
                            capability.name
                        ))
                    }
                }
                ModelFamily::Deep
                | ModelFamily::Forecasting
                | ModelFamily::Exit
                | ModelFamily::Evolutionary
                | ModelFamily::Rl => Ok(ExportRoute::PyTorch),
                ModelFamily::Meta | ModelFamily::Adaptive | ModelFamily::Anomaly => {
                    Err(anyhow::anyhow!(
                        "model `{}` ({:?}) does not currently expose a supported ONNX export path",
                        capability.name,
                        capability.family
                    ))
                }
            };
        }

        let model_type_str = model.get_type().name()?.to_string();
        let normalized_type = model_type_str.to_ascii_lowercase();
        let normalized_name = name.to_ascii_lowercase();
        if model.hasattr("state_dict")?
            || normalized_type.contains("mlp")
            || normalized_type.contains("nbeats")
            || normalized_type.contains("tide")
            || normalized_type.contains("tabnet")
            || normalized_type.contains("kan")
            || normalized_type.contains("transformer")
            || normalized_type.contains("patch")
            || normalized_type.contains("timesnet")
        {
            Ok(ExportRoute::PyTorch)
        } else if normalized_name.contains("lightgbm") || normalized_name.contains("lgbm") {
            Ok(ExportRoute::LightGbm)
        } else if normalized_name.contains("xgboost") || normalized_name.contains("xgb") {
            Ok(ExportRoute::XgBoost)
        } else if normalized_name.contains("catboost") || normalized_name.contains("cat") {
            Ok(ExportRoute::CatBoost)
        } else {
            Err(anyhow::anyhow!(
                "unable to resolve ONNX export route for model `{name}` of runtime type `{model_type_str}`"
            ))
        }
    }

    #[cfg(feature = "onnx-export-bridge")]
    fn array2_to_numpy(py: Python<'_>, sample_input: &Array2<f32>) -> PyResult<Py<PyAny>> {
        let numpy = PyModule::import(py, "numpy")?;
        let shape = (sample_input.nrows(), sample_input.ncols());
        let flat: Vec<f32> = sample_input.iter().copied().collect();
        let np_array = numpy
            .call_method1("array", (flat,))?
            .call_method1("reshape", (shape,))?;
        Ok(np_array.unbind())
    }

    /// Create new ONNX exporter
    pub fn new(models_dir: impl AsRef<Path>) -> Result<Self> {
        let models_dir = models_dir.as_ref().to_path_buf();
        let onnx_dir = models_dir.join("onnx");

        // Create ONNX directory
        std::fs::create_dir_all(&onnx_dir).context("Failed to create ONNX directory")?;

        #[cfg(feature = "onnx-export-bridge")]
        let py_exporter = Python::attach(|py| {
            let exporter_module = PyModule::import(py, "forex_bot.models.onnx_exporter")
                .context("Failed to import forex_bot.models.onnx_exporter")?;

            let exporter_class = exporter_module
                .getattr("ONNXExporter")
                .context("ONNXExporter class not found")?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("models_dir", models_dir.to_string_lossy().as_ref())?;

            let py_exporter = exporter_class.call((), Some(&kwargs))?;

            Ok::<Py<PyAny>, anyhow::Error>(py_exporter.into())
        })?;

        // Initialize Python exporter
        Ok(Self {
            #[cfg(feature = "onnx-export-bridge")]
            py_exporter: Some(py_exporter),
            models_dir,
            onnx_dir,
        })
    }

    /// Export PyTorch neural network model to ONNX
    /// Delegates to Python torch.onnx.export via PyO3
    #[cfg(feature = "onnx-export-bridge")]
    pub fn export_pytorch_model(
        &self,
        name: &str,
        model: &Py<PyAny>,
        sample_input: &Array2<f32>,
    ) -> Result<PathBuf> {
        Python::attach(|py| {
            let exporter = self
                .py_exporter
                .as_ref()
                .context("ONNX exporter not initialized")?
                .bind(py);

            let np_array = Self::array2_to_numpy(py, sample_input)?;
            let result = exporter.call_method1("_export_pytorch_model", (name, model, np_array))?;

            // Extract path from result
            let path_str: String = result.extract()?;
            Ok(PathBuf::from(path_str))
        })
    }

    /// Export LightGBM model to ONNX using onnxmltools
    /// Python lines 209-233
    #[cfg(feature = "onnx-export-bridge")]
    pub fn export_lightgbm_model(&self, name: &str, sample_input: &Array2<f32>) -> Result<PathBuf> {
        Python::attach(|py| {
            let exporter = self
                .py_exporter
                .as_ref()
                .context("ONNX exporter not initialized")?
                .bind(py);
            let np_array = Self::array2_to_numpy(py, sample_input)?;

            // Load the saved LightGBM model from disk
            let model_path = self.models_dir.join(format!("{}.joblib", name));
            let joblib = PyModule::import(py, "joblib")?;
            let model = joblib.call_method1("load", (model_path.to_string_lossy().as_ref(),))?;

            // Export via Python
            let result =
                exporter.call_method1("_export_lightgbm_model", (name, model, np_array))?;

            if result.is_none() {
                return Err(anyhow::anyhow!("LightGBM ONNX export returned None"));
            }

            let path_str: String = result.extract()?;
            Ok(PathBuf::from(path_str))
        })
    }

    /// Export XGBoost model to ONNX using onnxmltools
    /// Python lines 235-259
    #[cfg(feature = "onnx-export-bridge")]
    pub fn export_xgboost_model(&self, name: &str, sample_input: &Array2<f32>) -> Result<PathBuf> {
        Python::attach(|py| {
            let exporter = self
                .py_exporter
                .as_ref()
                .context("ONNX exporter not initialized")?
                .bind(py);
            let np_array = Self::array2_to_numpy(py, sample_input)?;

            let model_path = self.models_dir.join(format!("{}.joblib", name));
            let joblib = PyModule::import(py, "joblib")?;
            let model = joblib.call_method1("load", (model_path.to_string_lossy().as_ref(),))?;

            let result = exporter.call_method1("_export_xgboost_model", (name, model, np_array))?;

            if result.is_none() {
                return Err(anyhow::anyhow!("XGBoost ONNX export returned None"));
            }

            let path_str: String = result.extract()?;
            Ok(PathBuf::from(path_str))
        })
    }

    /// Export CatBoost model to ONNX using CatBoost's native exporter
    /// Python lines 193-207
    #[cfg(feature = "onnx-export-bridge")]
    pub fn export_catboost_model(&self, name: &str, sample_input: &Array2<f32>) -> Result<PathBuf> {
        Python::attach(|py| {
            let exporter = self
                .py_exporter
                .as_ref()
                .context("ONNX exporter not initialized")?
                .bind(py);
            let np_array = Self::array2_to_numpy(py, sample_input)?;

            let model_path = self.models_dir.join(format!("{}.joblib", name));
            let joblib = PyModule::import(py, "joblib")?;
            let model = joblib.call_method1("load", (model_path.to_string_lossy().as_ref(),))?;

            let result =
                exporter.call_method1("_export_catboost_model", (name, model, np_array))?;

            if result.is_none() {
                return Err(anyhow::anyhow!("CatBoost ONNX export returned None"));
            }

            let path_str: String = result.extract()?;
            Ok(PathBuf::from(path_str))
        })
    }

    /// Export all models to ONNX format
    /// Python lines 61-131
    #[cfg(feature = "onnx-export-bridge")]
    pub fn export_all(
        &self,
        models: HashMap<String, Py<PyAny>>,
        sample_input: &Array2<f32>,
    ) -> Result<ExportManifest> {
        let mut exported = HashMap::new();

        for (name, model) in models.iter() {
            match self.export_model(name, model, sample_input) {
                Ok(metadata) => {
                    info!("Exported {} to ONNX: {:?}", name, metadata.path);
                    exported.insert(name.clone(), metadata);
                }
                Err(e) => {
                    warn!("ONNX export skipped for {}: {}", name, e);
                }
            }
        }

        // Save manifest
        let manifest = ExportManifest { models: exported };
        let manifest_path = self.onnx_dir.join("export_manifest.json");
        let json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, json)?;

        info!("Exported {} models to ONNX", manifest.models.len());

        Ok(manifest)
    }

    /// Export a single model (auto-detects type)
    #[cfg(feature = "onnx-export-bridge")]
    fn export_model(
        &self,
        name: &str,
        model: &Py<PyAny>,
        sample_input: &Array2<f32>,
    ) -> Result<ExportMetadata> {
        Python::attach(|py| {
            let model_bound = model.bind(py);
            let model_type_str = model_bound.get_type().name()?.to_string();
            let route = self.resolve_export_route(name, model_bound)?;

            info!("Exporting {} (type: {})", name, model_type_str);

            let path = match route {
                ExportRoute::PyTorch => self.export_pytorch_model(name, model, sample_input)?,
                ExportRoute::LightGbm => self.export_lightgbm_model(name, sample_input)?,
                ExportRoute::XgBoost => self.export_xgboost_model(name, sample_input)?,
                ExportRoute::CatBoost => self.export_catboost_model(name, sample_input)?,
            };

            // Extract classes if available
            let classes = Self::extract_classes(model_bound)?;

            Ok(ExportMetadata { path, classes })
        })
    }

    /// Extract class labels from model
    #[cfg(feature = "onnx-export-bridge")]
    fn extract_classes(model: &Bound<'_, PyAny>) -> Result<Option<Vec<i32>>> {
        Python::attach(|_py| {
            // Try to get classes_ attribute
            if let Ok(classes_attr) = model.getattr("classes_") {
                if let Ok(classes_list) = classes_attr.extract::<Vec<i32>>() {
                    return Ok(Some(classes_list));
                }
            }

            // Try underlying model.model attribute
            if let Ok(base_model) = model.getattr("model") {
                if let Ok(classes_attr) = base_model.getattr("classes_") {
                    if let Ok(classes_list) = classes_attr.extract::<Vec<i32>>() {
                        return Ok(Some(classes_list));
                    }
                }
            }

            Ok(None)
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "onnx-export-bridge")]
enum ExportRoute {
    PyTorch,
    LightGbm,
    XgBoost,
    CatBoost,
}

// ============================================================================
// HELPER: ADD ONNX EXPORT TO NEURAL NETWORK WRAPPERS
// ============================================================================

/// Trait for models that can be exported to ONNX
pub trait ONNXExportable {
    /// Export this model to ONNX format
    fn export_to_onnx(
        &self,
        name: &str,
        sample_input: &Array2<f32>,
        output_path: &Path,
    ) -> Result<()>;
}

// NOTE: We'll implement this trait for MLPExpert, NBeatsExpert, etc. in neural_networks.rs
// For now, this module provides the infrastructure

// ============================================================================
// SUMMARY
// ============================================================================
//
// This module provides ONNX export functionality for all model types:
//
// EXPORT STRATEGY:
// - PyTorch models (MLP, NBeats, TiDE, TabNet, KAN): torch.onnx.export via PyO3
// - Tree models (LightGBM, XGBoost, CatBoost): Native library exporters via PyO3
// - Output: .onnx files + export_manifest.json
//
// BENEFITS:
// ✅ 10-100x faster inference than Python models
// ✅ Cross-platform deployment (train on Windows, deploy on Linux)
// ✅ GPU acceleration support via ONNX Runtime CUDA provider
// ✅ Reduced memory footprint (compiled graphs)
// ✅ No Python runtime needed for inference
//
// WORKFLOW:
// 1. Train models in Rust (using PyO3 wrappers)
// 2. Export to ONNX using this module
// 3. Load in production using ONNXInferenceEngine (lib.rs)
// 4. Ultra-fast inference with ONNX Runtime
//

#[cfg(test)]
mod tests {
    #[cfg(feature = "onnx-export-bridge")]
    use super::*;
    #[cfg(feature = "onnx-export-bridge")]
    use ndarray::array;
    #[cfg(feature = "onnx-export-bridge")]
    use pyo3::types::PyDict;
    #[cfg(feature = "onnx-export-bridge")]
    use std::ffi::CString;

    #[cfg(feature = "onnx-export-bridge")]
    fn run_py(py: Python<'_>, code: &str, locals: &Bound<'_, PyDict>) -> PyResult<()> {
        let code = CString::new(code).expect("python code should not contain embedded nulls");
        py.run(code.as_c_str(), Some(locals), Some(locals))
    }

    #[cfg(feature = "onnx-export-bridge")]
    #[test]
    fn test_export_pytorch_model_avoids_pandas_bridge() -> Result<()> {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            run_py(
                py,
                r#"
import sys, types, numpy as np
_orig_pandas = sys.modules.get("pandas")

class _PandasTrap(types.SimpleNamespace):
    def DataFrame(self, *args, **kwargs):
        raise AssertionError("pandas DataFrame bridge must not be used")

sys.modules["pandas"] = _PandasTrap()

class FakeExporter:
    def _export_pytorch_model(self, name, model, sample_input):
        assert isinstance(sample_input, np.ndarray)
        return "ok.onnx"

exporter = FakeExporter()
model = object()
"#,
                &locals,
            )?;

            let exporter_obj = locals
                .get_item("exporter")?
                .expect("exporter should be defined")
                .unbind();
            let model_obj = locals
                .get_item("model")?
                .expect("model should be defined")
                .unbind();

            let exporter = ONNXExporter {
                py_exporter: Some(exporter_obj),
                models_dir: PathBuf::new(),
                onnx_dir: PathBuf::new(),
            };

            let sample_input = array![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
            let result = exporter.export_pytorch_model("mlp", &model_obj, &sample_input);

            let _ = run_py(
                py,
                r#"
import sys
if _orig_pandas is None:
    sys.modules.pop("pandas", None)
else:
    sys.modules["pandas"] = _orig_pandas
"#,
                &locals,
            );

            let path = result?;
            assert_eq!(path, PathBuf::from("ok.onnx"));
            Ok(())
        })
    }
}
