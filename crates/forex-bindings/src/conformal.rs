use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyAny;

#[pyclass(name = "ConformalGate", module = "forex_bindings")]
pub struct ConformalGate {
    #[pyo3(get, set)]
    pub alpha: f64,
    #[pyo3(get, set)]
    pub qhat: f64,
    #[pyo3(get, set)]
    pub fitted: bool,
    #[pyo3(get, set)]
    pub n_calib: usize,
}

#[pymethods]
impl ConformalGate {
    #[new]
    #[pyo3(signature = (alpha=0.10))]
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            qhat: 1.0,
            fitted: false,
            n_calib: 0,
        }
    }

    pub fn fit<'py>(
        &mut self,
        py: Python<'py>,
        probs: &Bound<'py, PyAny>,
        y_true: &Bound<'py, PyAny>,
    ) -> PyResult<bool> {
        let np = py.import("numpy")?;
        let probs_array_any = np.getattr("asarray")?.call1((probs,))?;
        let probs_array_f64 = probs_array_any.call_method1("astype", ("float64",))?;
        let probs_array: PyReadonlyArray2<'py, f64> = probs_array_f64.extract()?;

        let labels_array_any = np.getattr("asarray")?.call1((y_true,))?;
        let labels_array_i64 = labels_array_any.call_method1("astype", ("int64",))?;
        let labels_array: PyReadonlyArray1<'py, i64> = labels_array_i64.extract()?;

        let p = probs_array.as_array();
        if p.ndim() != 2 || p.shape()[1] < 3 {
            return Ok(false);
        }

        let y_raw = labels_array.as_array();
        let n = usize::min(y_raw.len(), p.shape()[0]);
        if n < 64 {
            return Ok(false);
        }

        let alpha = self.alpha.clamp(1e-6, 0.99);
        let q_level = (((n + 1) as f64) * (1.0 - alpha)).ceil() / (n as f64);
        let q_level = q_level.clamp(0.0, 1.0);

        let mut scores = Vec::with_capacity(n);
        for row in 0..n {
            let y = match y_raw[row] {
                -1 => 2usize,
                value if value < 0 => 0usize,
                value if value > 2 => 2usize,
                value => value as usize,
            };
            let prob = p[[row, y]].clamp(1e-8, 1.0);
            scores.push(1.0 - prob);
        }
        scores.sort_by(|a, b| a.total_cmp(b));
        let idx = ((q_level * (n as f64)).ceil() as isize - 1).clamp(0, (n - 1) as isize) as usize;
        self.qhat = scores[idx].clamp(0.0, 1.0);
        self.fitted = true;
        self.n_calib = n;
        Ok(true)
    }

    pub fn prediction_set<'py>(
        &self,
        py: Python<'py>,
        probs_row: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<usize>> {
        let np = py.import("numpy")?;
        let row_any = np.getattr("asarray")?.call1((probs_row,))?;
        let row_f64 = row_any.call_method1("astype", ("float64",))?;
        let row_array: PyReadonlyArray1<'py, f64> = row_f64.extract()?;
        let row = row_array.as_array();
        if row.len() < 3 {
            return Ok(vec![0, 1, 2]);
        }

        let mut probs = [0.0_f64; 3];
        for idx in 0..3 {
            probs[idx] = row[idx].clamp(1e-8, 1.0);
        }

        let mut keep: Vec<usize> = probs
            .iter()
            .enumerate()
            .filter_map(|(idx, prob)| {
                if (1.0 - *prob) <= self.qhat {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        if keep.is_empty() {
            let best = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            keep.push(best);
        }
        Ok(keep)
    }

    #[pyo3(signature = (probs_row, min_set_size=3))]
    pub fn should_abstain<'py>(
        &self,
        py: Python<'py>,
        probs_row: &Bound<'py, PyAny>,
        min_set_size: usize,
    ) -> PyResult<(bool, usize)> {
        if !self.fitted {
            return Ok((false, 1));
        }
        let keep = self.prediction_set(py, probs_row)?;
        let size = keep.len();
        Ok((size >= usize::max(1, min_set_size), size))
    }
}
