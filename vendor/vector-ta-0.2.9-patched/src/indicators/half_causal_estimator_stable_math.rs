#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct StablePopulationMoments {
    count: usize,
    mean: f64,
    m2: f64,
}

impl StablePopulationMoments {
    #[cfg(test)]
    #[inline]
    pub(crate) fn clear(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }

    #[inline]
    pub(crate) fn add(&mut self, value: f64) {
        let next_count = self.count + 1;
        let delta = value - self.mean;
        self.mean += delta / next_count as f64;
        let delta_after_mean = value - self.mean;
        self.m2 += delta * delta_after_mean;
        self.count = next_count;
    }

    #[inline]
    pub(crate) fn count(&self) -> usize {
        self.count
    }

    #[inline]
    pub(crate) fn mean(&self) -> Option<f64> {
        (self.count != 0).then_some(self.mean)
    }

    #[inline]
    pub(crate) fn population_stdev(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let variance = (self.m2 / self.count as f64).max(0.0);
        Some(variance.sqrt())
    }

    #[inline]
    pub(crate) fn creator_inverse_cv(&self, maximum_adjust_factor: f64) -> f64 {
        let Some(mean) = self.mean() else {
            return 1.0;
        };
        if mean == 0.0 {
            return 1.0;
        }
        let Some(stdev) = self.population_stdev() else {
            return 1.0;
        };
        let confidence = 1.0 - (stdev / mean).clamp(0.0, 1.0) * maximum_adjust_factor;
        if confidence.is_nan() { 1.0 } else { confidence }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NeumaierSum {
    sum: f64,
    correction: f64,
}

impl NeumaierSum {
    #[inline]
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    #[inline]
    pub(crate) fn add_weighted(&mut self, value: f64, confidence: f64, coefficient: f64) {
        let scaled = value * confidence;
        let term = scaled * coefficient;
        self.add(term);
    }

    #[inline]
    pub(crate) fn total(self) -> f64 {
        self.sum + self.correction
    }
}

#[cfg(test)]
mod tests {
    use super::{NeumaierSum, StablePopulationMoments};

    #[test]
    fn chronological_welford_matches_biased_population_definition() {
        let mut moments = StablePopulationMoments::default();
        for value in [1.0, 2.0, 3.0, 4.0] {
            moments.add(value);
        }

        assert_eq!(moments.count(), 4);
        assert_eq!(moments.mean(), Some(2.5));
        assert_eq!(moments.population_stdev(), Some(1.25_f64.sqrt()));
    }

    #[test]
    fn creator_cv_uses_tiny_nonzero_mean_but_falls_back_for_exact_zero() {
        let mut tiny = StablePopulationMoments::default();
        tiny.add(1.0e-16);
        tiny.add(2.0e-16);
        let confidence = tiny.creator_inverse_cv(1.0);
        assert!((confidence - (2.0 / 3.0)).abs() <= 8.0 * f64::EPSILON);
        assert_ne!(confidence, 1.0);

        let mut zero = StablePopulationMoments::default();
        zero.add(-1.0);
        zero.add(1.0);
        assert_eq!(zero.mean(), Some(0.0));
        assert_eq!(zero.creator_inverse_cv(1.0), 1.0);
    }

    #[test]
    fn neumaier_recovers_low_term_lost_by_plain_accumulation() {
        let mut sum = NeumaierSum::default();
        for value in [1.0e16, 1.0, -1.0e16] {
            sum.add(value);
        }
        assert_eq!(sum.total(), 1.0);
    }

    #[test]
    fn weighted_add_uses_the_creator_term_shape() {
        let mut sum = NeumaierSum::default();
        sum.add_weighted(3.0, 0.5, 0.25);
        assert_eq!(sum.total(), 0.375);
    }

    #[test]
    fn clear_restarts_the_population_schedule() {
        let mut moments = StablePopulationMoments::default();
        moments.add(8.0);
        moments.clear();
        moments.add(5.0);
        assert_eq!(moments.count(), 1);
        assert_eq!(moments.mean(), Some(5.0));
        assert_eq!(moments.population_stdev(), Some(0.0));
    }
}
