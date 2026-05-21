use anyhow::{Context, Result, bail};
use ndarray::Array2;
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use tracing::{info, warn};

pub struct ONNXInferenceEngine {
    sessions: HashMap<String, Mutex<Session>>,
    model_outputs: HashMap<String, String>,
    expected_feature_counts: HashMap<String, usize>,
}

fn map_ort_error(error: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{error}")
}

impl ONNXInferenceEngine {
    pub fn new() -> Result<Self> {
        let committed = ort::init()
            .with_name("neoethos_models_ort")
            .with_execution_providers([ep::CUDA::default().build(), ep::CPU::default().build()])
            .commit();
        if !committed {
            warn!("ORT environment already configured; keeping existing ONNX Runtime settings");
        }

        Ok(Self {
            sessions: HashMap::new(),
            model_outputs: HashMap::new(),
            expected_feature_counts: HashMap::new(),
        })
    }

    pub fn load_models(&mut self, models_dir: impl AsRef<Path>) -> Result<()> {
        let models_dir = models_dir.as_ref();
        if !models_dir.exists() {
            warn!("Models directory not found: {:?}", models_dir);
            return Ok(());
        }

        let onnx_dir = models_dir.join("onnx");
        if !onnx_dir.exists() {
            warn!("ONNX directory not found: {:?}", onnx_dir);
            return Ok(());
        }

        for entry in std::fs::read_dir(onnx_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "onnx") {
                let name = match path.file_stem().and_then(|s| s.to_str()) {
                    Some(stem) if !stem.is_empty() => stem.to_string(),
                    _ => {
                        warn!("Skipping ONNX file with non-utf8 or empty stem: {:?}", path);
                        continue;
                    }
                };
                if let Err(e) = self.load_model(&name, &path) {
                    warn!("Failed to load model {}: {}", name, e);
                }
            }
        }

        Ok(())
    }

    pub fn load_model(&mut self, name: &str, path: &Path) -> Result<()> {
        self.load_model_with_feature_count(name, path, None)
    }

    pub fn load_model_with_feature_count(
        &mut self,
        name: &str,
        path: &Path,
        expected_feature_count: Option<usize>,
    ) -> Result<()> {
        let mut builder = Session::builder().map_err(map_ort_error)?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(map_ort_error)?;
        builder = builder.with_intra_threads(4).map_err(map_ort_error)?;
        builder = builder.with_inter_threads(4).map_err(map_ort_error)?;
        let session = builder
            .commit_from_file(path)
            .map_err(map_ort_error)
            .context(format!("Failed to load model {}", name))?;

        let outputs = session.outputs();
        let mut proba_output_name = String::new();

        for out in outputs {
            if out.name().to_lowercase().contains("prob") {
                proba_output_name = out.name().to_string();
                break;
            }
        }
        if proba_output_name.is_empty() {
            if let Some(last) = outputs.last() {
                proba_output_name = last.name().to_string();
            } else {
                warn!("ONNX model '{}' has no outputs: {}", name, path.display());
            }
        }

        if let Some(feature_count) = expected_feature_count {
            self.expected_feature_counts
                .insert(name.to_string(), feature_count);
        }
        self.sessions.insert(name.to_string(), Mutex::new(session));
        self.model_outputs
            .insert(name.to_string(), proba_output_name);
        info!("Loaded ONNX model: {}", name);
        Ok(())
    }

    pub fn predict_proba(&self, model_name: &str, features: &Array2<f32>) -> Result<Array2<f32>> {
        if let Some(expected_feature_count) = self.expected_feature_counts.get(model_name).copied()
            && features.ncols() != expected_feature_count
        {
            anyhow::bail!(
                "ONNX inference feature mismatch: model expects {} features, got {}",
                expected_feature_count,
                features.ncols()
            );
        }
        let output_name = self
            .model_outputs
            .get(model_name)
            .context("Output name not found")?
            .clone();
        if output_name.is_empty() {
            bail!("ONNX model {model_name} has no output tensor name");
        }

        let session = self
            .sessions
            .get(model_name)
            .context(format!("Model {} not loaded", model_name))?;
        let mut session = session
            .lock()
            .map_err(|_| anyhow::anyhow!("ONNX session lock poisoned for {model_name}"))?;

        let input_shape = [features.nrows(), features.ncols()];
        let contiguous_features;
        let input_slice = match features.as_slice_memory_order() {
            Some(slice) => slice,
            None => {
                contiguous_features = features.iter().copied().collect::<Vec<_>>();
                contiguous_features.as_slice()
            }
        };
        let input_tensor =
            TensorRef::from_array_view((input_shape, input_slice)).map_err(map_ort_error)?;
        let outputs = session
            .run(ort::inputs![input_tensor])
            .map_err(map_ort_error)?;
        let output_value = outputs.get(output_name).context("Output tensor missing")?;
        let (output_shape, output_data) = output_value
            .try_extract_tensor::<f32>()
            .map_err(map_ort_error)?;

        let dims = output_shape
            .iter()
            .map(|dim| {
                if *dim < 0 {
                    bail!("ONNX output contains dynamic dimension {dim}");
                }
                Ok(*dim as usize)
            })
            .collect::<Result<Vec<_>>>()?;
        match dims.as_slice() {
            [rows] => Ok(Array2::from_shape_vec((*rows, 1), output_data.to_vec())?),
            [rows, cols] => Ok(Array2::from_shape_vec(
                (*rows, *cols),
                output_data.to_vec(),
            )?),
            _ => bail!("ONNX probability output must be rank 1 or 2, got {dims:?}"),
        }
    }
}
