use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;

#[derive(Debug, Clone)]
pub enum CalibrationModel {
    Constant(f64),
    Platt { a: f64, b: f64 },
}

#[pyclass(name = "ProbabilityCalibrator", module = "forex_bindings")]
pub struct ProbabilityCalibrator {
    #[pyo3(get, set)]
    pub method: String,
    #[pyo3(get, set)]
    pub fitted: bool,
    pub models: Vec<CalibrationModel>,
}

#[pymethods]
impl ProbabilityCalibrator {
    #[new]
    #[pyo3(signature = (method="platt"))]
    pub fn new(method: &str) -> Self {
        Self {
            method: method.to_lowercase(),
            fitted: false,
            models: Vec::new(),
        }
    }

    pub fn fit<'py>(
        &mut self,
        _py: Python<'py>,
        probs: PyReadonlyArray2<'py, f64>,
        y_true: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<bool> {
        let p = probs.as_array();
        let y = y_true.as_array();

        if p.shape()[0] != y.len() || p.shape()[1] < 3 {
            return Ok(false);
        }

        self.models.clear();
        for cls in 0..3 {
            let mut x_cls = Vec::with_capacity(y.len());
            let mut y_cls = Vec::with_capacity(y.len());

            for i in 0..y.len() {
                let prob = p[[i, cls]].clamp(1e-6, 1.0 - 1e-6);
                let logit = (prob / (1.0 - prob)).ln();
                x_cls.push(logit);
                let target = if y[i] == cls as i64 || (y[i] == -1 && cls == 2) {
                    1.0
                } else {
                    0.0
                };
                y_cls.push(target);
            }

            let (a, b) = fit_simple_logistic(&x_cls, &y_cls);
            self.models.push(CalibrationModel::Platt { a, b });
        }

        self.fitted = true;
        Ok(true)
    }

    pub fn predict_proba<'py>(
        &self,
        py: Python<'py>,
        probs: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let p = probs.as_array();
        let n_rows = p.shape()[0];
        let mut out = Array2::<f64>::zeros((n_rows, 3));

        if !self.fitted || self.models.len() < 3 {
            for i in 0..n_rows {
                for j in 0..3 {
                    out[[i, j]] = p[[i, j]];
                }
            }
            return Ok(out.to_pyarray(py));
        }

        for cls in 0..3 {
            let model = &self.models[cls];
            for i in 0..n_rows {
                let prob = p[[i, cls]].clamp(1e-6, 1.0 - 1e-6);
                match model {
                    CalibrationModel::Constant(c) => out[[i, cls]] = *c,
                    CalibrationModel::Platt { a, b } => {
                        let logit = (prob / (1.0 - prob)).ln();
                        out[[i, cls]] = 1.0 / (1.0 + (-(a * logit + b)).exp());
                    }
                }
            }
        }

        for i in 0..n_rows {
            let sum: f64 = (0..3).map(|j| out[[i, j]]).sum();
            if sum > 0.0 {
                for j in 0..3 {
                    out[[i, j]] /= sum;
                }
            }
        }

        Ok(out.to_pyarray(py))
    }
}

fn fit_simple_logistic(x: &[f64], y: &[f64]) -> (f64, f64) {
    let mut a = 1.0;
    let mut b = 0.0;
    let lr = 0.1;
    for _ in 0..100 {
        let mut grad_a = 0.0;
        let mut grad_b = 0.0;
        for i in 0..x.len() {
            let pred = 1.0 / (1.0 + (-(a * x[i] + b)).exp());
            let err = pred - y[i];
            grad_a += err * x[i];
            grad_b += err;
        }
        a -= lr * grad_a / x.len() as f64;
        b -= lr * grad_b / x.len() as f64;
    }
    (a, b)
}
