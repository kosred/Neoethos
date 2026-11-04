//! Shared fixed-risk break-even/trailing geometry, extracted from Trader's
//! Position. This computes a candidate price only; callers still own causal
//! closed-bar timing, quote fills, position occupancy and monetary economics.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrailingPolicy {
    pub be_trigger_r: f64,
    pub stop_multiplier: f64,
    pub min_lock_pips: f64,
    pub pip_size: f64,
}

impl TrailingPolicy {
    pub fn new(
        be_trigger_r: f64,
        stop_multiplier: f64,
        min_lock_pips: f64,
        pip_size: f64,
    ) -> Option<Self> {
        let finite_positive = |value: f64| value.is_finite() && value > 0.0;
        if !finite_positive(be_trigger_r)
            || !finite_positive(stop_multiplier)
            || !finite_positive(pip_size)
            || !min_lock_pips.is_finite()
            || min_lock_pips < 0.0
        {
            return None;
        }
        Some(Self {
            be_trigger_r,
            stop_multiplier,
            min_lock_pips,
            pip_size,
        })
    }

    /// Return a newly tightened stop, or None when unarmed/unchanged. Direction
    /// is +1 for long and -1 for short. Keep the original operation order so
    /// existing Position results retain their f64 arithmetic and thresholds.
    pub fn next_stop_price(
        &self,
        entry: f64,
        original_stop: f64,
        direction: i8,
        high: f64,
        low: f64,
        current: Option<f64>,
    ) -> Option<f64> {
        let stop_dist = (entry - original_stop).abs();
        if !stop_dist.is_finite() || stop_dist <= 0.0 {
            return None;
        }
        let trigger = self.be_trigger_r * stop_dist;
        let lock = self.min_lock_pips * self.pip_size;
        let candidate = match direction {
            1 => {
                if (high - entry) < trigger {
                    return None;
                }
                (high - self.stop_multiplier * stop_dist).max(entry + lock)
            }
            -1 => {
                if (entry - low) < trigger {
                    return None;
                }
                (low + self.stop_multiplier * stop_dist).min(entry - lock)
            }
            _ => return None,
        };
        // The minimum lock can exceed the R trigger on a narrow stop. Never
        // install a long stop above the observed high (or a short stop below
        // the low): the locked profit has not been reached yet.
        let reached = if direction == 1 {
            candidate <= high
        } else {
            candidate >= low
        };
        if !candidate.is_finite() || !reached {
            return None;
        }
        let better = match (current, direction) {
            (None, _) => true,
            (Some(previous), 1) => candidate > previous,
            (Some(previous), -1) => candidate < previous,
            _ => false,
        };
        better.then_some(candidate)
    }
}

#[cfg(test)]
mod tests {
    use super::TrailingPolicy;

    #[test]
    fn a_profit_lock_must_be_reached_before_it_can_arm() {
        // Binary-exact prices: 1R is 0.125, but the lock requires 0.25.
        let policy = TrailingPolicy::new(1.0, 1.0, 2.0, 0.125).unwrap();
        for direction in [1_i8, -1] {
            let entry = 1.25;
            let original_stop = entry - f64::from(direction) * 0.125;
            assert_eq!(
                policy.next_stop_price(entry, original_stop, direction, 1.375, 1.125, None),
                None,
                "the R trigger alone cannot authorize an unreached lock ({direction})"
            );
            assert_eq!(
                policy.next_stop_price(entry, original_stop, direction, 1.5, 1.0, None),
                Some(entry + f64::from(direction) * 0.25),
                "the lock arms when the actual excursion reaches it ({direction})"
            );
        }
    }

    #[test]
    fn binary_exact_long_short_lock_and_ratchet_geometry() {
        let policy = TrailingPolicy::new(1.0, 1.0, 1.0, 0.125).unwrap();
        assert_eq!(
            policy.next_stop_price(1.25, 1.0, 1, 1.375, 1.25, None),
            None
        );
        assert_eq!(
            policy.next_stop_price(1.25, 1.0, 1, 1.5, 1.25, None),
            Some(1.375)
        );
        assert_eq!(
            policy.next_stop_price(1.25, 1.0, 1, 1.75, 1.25, Some(1.375)),
            Some(1.5)
        );
        assert_eq!(
            policy.next_stop_price(1.25, 1.0, 1, 1.5, 1.25, Some(1.5)),
            None
        );
        assert_eq!(
            policy.next_stop_price(1.25, 1.5, -1, 1.25, 1.0, None),
            Some(1.125)
        );
        assert_eq!(
            policy.next_stop_price(1.25, 1.5, -1, 1.25, 0.75, Some(1.125)),
            Some(1.0)
        );
        assert_eq!(
            policy.next_stop_price(1.25, 1.5, -1, 1.25, 1.0, Some(1.0)),
            None
        );
    }
}
