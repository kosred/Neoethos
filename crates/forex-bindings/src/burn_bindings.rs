use forex_models::burn_models::*;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Helper macro to create PyO3 wrappers for each Burn model
macro_rules! burn_model_wrapper {
    (
        $py_name:ident,
        $config_type:ident,
        $model_type:ident,
        $default_hidden:expr,
        $default_layers:expr
    ) => {
        #[pyclass(unsendable, module = "forex_bindings")]
        pub struct $py_name {
            model: Option<$model_type<TrainBackend>>,
            input_dim: usize,
            hidden_dim: usize,
            n_classes: usize,
            lr: f64,
            batch_size: usize,
            max_epochs: usize,
            patience: usize,
        }

        #[pymethods]
        impl $py_name {
            #[new]
            #[pyo3(signature = (
                        input_dim=96,
                        hidden_dim=$default_hidden,
                        n_classes=3,
                        lr=1e-3,
                        batch_size=64,
                        max_epochs=100,
                        patience=8,
                    ))]
            fn new(
                input_dim: usize,
                hidden_dim: usize,
                n_classes: usize,
                lr: f64,
                batch_size: usize,
                max_epochs: usize,
                patience: usize,
            ) -> Self {
                Self {
                    model: None,
                    input_dim,
                    hidden_dim,
                    n_classes,
                    lr,
                    batch_size,
                    max_epochs,
                    patience,
                }
            }

            fn fit<'py>(
                &mut self,
                _py: Python<'py>,
                features: PyReadonlyArray2<'py, f32>,
                labels: PyReadonlyArray1<'py, i32>,
            ) -> PyResult<f64> {
                let x = features.as_array().to_owned();
                let y: Vec<i32> = labels.as_array().iter().copied().collect();

                // Auto-detect input_dim from features
                self.input_dim = x.ncols();

                let device = <TrainBackend as burn::tensor::backend::Backend>::Device::default();
                let config = $config_type::new(self.input_dim)
                    .with_hidden_dim(self.hidden_dim)
                    .with_n_classes(self.n_classes);
                let model = config.init::<TrainBackend>(&device);

                let train_config = TrainConfig {
                    lr: self.lr,
                    batch_size: self.batch_size,
                    max_epochs: self.max_epochs,
                    patience: self.patience,
                    n_classes: self.n_classes,
                    ..TrainConfig::default()
                };

                let (trained, best_loss) =
                    train_model::<TrainBackend, _>(model, &x, &y, &train_config).map_err(
                        |err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()),
                    )?;
                self.model = Some(trained);
                Ok(best_loss as f64)
            }

            fn predict_proba<'py>(
                &self,
                py: Python<'py>,
                features: PyReadonlyArray2<'py, f32>,
            ) -> PyResult<Bound<'py, PyArray2<f32>>> {
                let x = features.as_array().to_owned();
                let model = self.model.as_ref().ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "Model not trained yet. Call fit() first.",
                    )
                })?;
                let probs = predict_proba::<TrainBackend, _>(model, &x, self.batch_size).map_err(
                    |err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()),
                )?;
                Ok(probs.into_pyarray(py))
            }
        }
    };
}

burn_model_wrapper!(BurnMLPModel, BurnMLPConfig, BurnMLP, 256, 3);
burn_model_wrapper!(BurnNBeatsModel, BurnNBeatsConfig, BurnNBeats, 64, 3);
burn_model_wrapper!(BurnTiDEModel, BurnTiDEConfig, BurnTiDE, 128, 2);
burn_model_wrapper!(BurnKANModel, BurnKANConfig, BurnKAN, 32, 2);
burn_model_wrapper!(
    BurnTransformerModel,
    BurnTransformerConfig,
    BurnTransformer,
    128,
    4
);

// TabNet needs special handling
#[pyclass(unsendable, name = "BurnTabNetModel", module = "forex_bindings")]
pub struct BurnTabNetModel {
    model: Option<BurnTabNet<TrainBackend>>,
    input_dim: usize,
    hidden_dim: usize,
    n_classes: usize,
    lr: f64,
    batch_size: usize,
    max_epochs: usize,
    patience: usize,
}

#[pymethods]
impl BurnTabNetModel {
    #[new]
    #[pyo3(signature = (input_dim=96, hidden_dim=64, n_classes=3, lr=2e-3, batch_size=64, max_epochs=100, patience=8))]
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        n_classes: usize,
        lr: f64,
        batch_size: usize,
        max_epochs: usize,
        patience: usize,
    ) -> Self {
        Self {
            model: None,
            input_dim,
            hidden_dim,
            n_classes,
            lr,
            batch_size,
            max_epochs,
            patience,
        }
    }

    fn fit<'py>(
        &mut self,
        _py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
        labels: PyReadonlyArray1<'py, i32>,
    ) -> PyResult<f64> {
        let x = features.as_array().to_owned();
        let y: Vec<i32> = labels.as_array().iter().copied().collect();
        self.input_dim = x.ncols();

        let device = <TrainBackend as burn::tensor::backend::Backend>::Device::default();
        let config = BurnTabNetConfig::new(self.input_dim)
            .with_hidden_dim(self.hidden_dim)
            .with_n_classes(self.n_classes);
        let model = config.init::<TrainBackend>(&device);

        let train_config = TrainConfig {
            lr: self.lr,
            batch_size: self.batch_size,
            max_epochs: self.max_epochs,
            patience: self.patience,
            n_classes: self.n_classes,
            ..TrainConfig::default()
        };
        let (trained, best_loss) = train_model::<TrainBackend, _>(model, &x, &y, &train_config)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
        self.model = Some(trained);
        Ok(best_loss as f64)
    }

    fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        features: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let x = features.as_array().to_owned();
        let model = self.model.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Model not trained. Call fit() first.",
            )
        })?;
        let probs = predict_proba::<TrainBackend, _>(model, &x, self.batch_size)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(err.to_string()))?;
        Ok(probs.into_pyarray(py))
    }
}

pub fn register_burn_models(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BurnMLPModel>()?;
    m.add_class::<BurnNBeatsModel>()?;
    m.add_class::<BurnTiDEModel>()?;
    m.add_class::<BurnTabNetModel>()?;
    m.add_class::<BurnKANModel>()?;
    m.add_class::<BurnTransformerModel>()?;
    Ok(())
}
