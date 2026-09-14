//! Exact selection/holdout row authority shared by resident screening and
//! downstream population sizing.

#[cfg(any(test, target_os = "linux"))]
use crate::discovery::Stage1Window;

#[cfg(any(test, target_os = "linux"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidentFeatureScreeningScopeV2 {
    parent_row_count: u64,
    selection_row_start: u64,
    selection_row_end: u64,
    holdout_row_start: u64,
    holdout_row_end: u64,
}

#[cfg(any(test, target_os = "linux"))]
impl ResidentFeatureScreeningScopeV2 {
    pub(crate) const fn from_screening_plan_v2(
        parent_row_count: u64,
        selection_row_start: u64,
        selection_row_end: u64,
        holdout_row_start: u64,
        holdout_row_end: u64,
    ) -> Self {
        Self {
            parent_row_count,
            selection_row_start,
            selection_row_end,
            holdout_row_start,
            holdout_row_end,
        }
    }

    pub(crate) const fn parent_row_count(self) -> u64 {
        self.parent_row_count
    }

    pub(crate) fn selection_range_v3(self) -> Result<std::ops::Range<usize>, &'static str> {
        self.validate_partition_v2()?;
        Ok(usize::try_from(self.selection_row_start)
            .map_err(|_| "resident selection start does not fit usize")?
            ..usize::try_from(self.selection_row_end)
                .map_err(|_| "resident selection end does not fit usize")?)
    }

    fn validate_partition_v2(self) -> Result<(), &'static str> {
        if self.parent_row_count == 0
            || self.selection_row_start >= self.selection_row_end
            || self.selection_row_end != self.holdout_row_start
            || self.holdout_row_start >= self.holdout_row_end
            || self.holdout_row_end != self.parent_row_count
        {
            return Err(
                "resident screening scope is not an exact selection/holdout partition of the resident parent",
            );
        }
        Ok(())
    }

    pub(crate) fn resolve_stage1_v2(
        self,
        stage1_pct: f64,
        stage1_window: Stage1Window,
    ) -> Result<ResolvedResidentSelectionStage1ScopeV2, &'static str> {
        self.validate_partition_v2()?;
        if !stage1_pct.is_finite() || !(0.0..=1.0).contains(&stage1_pct) || stage1_pct == 0.0 {
            return Err("resident Stage1 percentage must be finite and in (0, 1]");
        }
        let selection_rows = self.selection_row_end - self.selection_row_start;
        let stage1_len = ((selection_rows as f64 * stage1_pct) as u64).min(selection_rows);
        if stage1_len == 0 {
            return Err("resident Stage1 percentage resolves to an empty selection window");
        }
        let (stage1_row_start, stage1_row_end) = match stage1_window {
            Stage1Window::MostRecent => {
                (self.selection_row_end - stage1_len, self.selection_row_end)
            }
            Stage1Window::Earliest => (
                self.selection_row_start,
                self.selection_row_start + stage1_len,
            ),
        };
        Ok(ResolvedResidentSelectionStage1ScopeV2 {
            stage1_row_start,
            stage1_row_end,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResolvedResidentSelectionStage1ScopeV2 {
    stage1_row_start: u64,
    stage1_row_end: u64,
}

#[cfg(any(test, target_os = "linux"))]
impl ResolvedResidentSelectionStage1ScopeV2 {
    pub(crate) const fn stage1_row_start(self) -> u64 {
        self.stage1_row_start
    }

    pub(crate) const fn stage1_row_end(self) -> u64 {
        self.stage1_row_end
    }
}

/// Metadata-only elapsed-time authority for the exact Stage1 interval. Its
/// production constructor reads endpoints from Data's immutable pinned base;
/// callers cannot substitute a goal horizon or a full-source timestamp span.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ResidentSelectionStage1TimeScopeV2 {
    parent_row_count: u64,
    rows: ResolvedResidentSelectionStage1ScopeV2,
    pinned_source_sha256: [u8; 32],
    first_timestamp_ms: i64,
    last_timestamp_ms: i64,
    elapsed_ms: i64,
}

impl ResidentSelectionStage1TimeScopeV2 {
    #[cfg(all(feature = "gpu-cuda", any(test, target_os = "linux")))]
    pub(crate) fn from_pinned_preparation_v2(
        scope: ResidentFeatureScreeningScopeV2,
        stage1_pct: f64,
        stage1_window: Stage1Window,
        prepared: &neoethos_data::PreparedGpuOnlyFeatureMaterializationV3,
    ) -> anyhow::Result<Self> {
        let rows = scope
            .resolve_stage1_v2(stage1_pct, stage1_window)
            .map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            scope.parent_row_count() == prepared.workspace_extent().row_count(),
            "Stage1 timestamp scope parent rows drifted from pinned Data"
        );
        let start = usize::try_from(rows.stage1_row_start())?;
        let end = usize::try_from(rows.stage1_row_end())?;
        let (first_ms, last_ms) = prepared.pinned_base_timestamp_bounds_ms_v1(start, end)?;
        Self::from_pinned_bounds_v2(
            scope.parent_row_count(),
            rows,
            prepared.pinned_source_projection_v1().identity_sha256(),
            first_ms,
            last_ms,
        )
        .map_err(anyhow::Error::msg)
    }

    #[cfg(any(test, target_os = "linux"))]
    fn from_pinned_bounds_v2(
        parent_row_count: u64,
        rows: ResolvedResidentSelectionStage1ScopeV2,
        pinned_source_sha256: [u8; 32],
        first_timestamp_ms: i64,
        last_timestamp_ms: i64,
    ) -> Result<Self, &'static str> {
        if rows.stage1_row_end > parent_row_count
            || rows
                .stage1_row_end
                .checked_sub(rows.stage1_row_start)
                .is_none_or(|n| n < 2)
            || pinned_source_sha256 == [0; 32]
        {
            return Err("Stage1 time authority requires a pinned nonempty multi-row interval");
        }
        let elapsed_ms = last_timestamp_ms
            .checked_sub(first_timestamp_ms)
            .filter(|elapsed| *elapsed > 0)
            .ok_or("Stage1 time authority requires a positive representable timestamp span")?;
        Ok(Self {
            parent_row_count,
            rows,
            pinned_source_sha256,
            first_timestamp_ms,
            last_timestamp_ms,
            elapsed_ms,
        })
    }

    pub(crate) fn validate_binding_v2(
        self,
        parent_row_count: u64,
        row_start: u64,
        row_end: u64,
        pinned_source_sha256: [u8; 32],
    ) -> Result<(), &'static str> {
        if self.parent_row_count != parent_row_count
            || self.rows.stage1_row_start != row_start
            || self.rows.stage1_row_end != row_end
            || self.pinned_source_sha256 != pinned_source_sha256
        {
            return Err("Stage1 time authority drifted from the exact source or evaluation rows");
        }
        Ok(())
    }

    pub(crate) fn evaluation_span_days(self) -> f64 {
        self.elapsed_ms as f64 / 86_400_000.0
    }

    pub(crate) const fn timestamp_bounds_ms(self) -> (i64, i64) {
        (self.first_timestamp_ms, self.last_timestamp_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_selection_range_is_the_exact_screened_suffix_not_the_holdout() {
        let scope = ResidentFeatureScreeningScopeV2::from_screening_plan_v2(100, 20, 80, 80, 100);
        assert_eq!(scope.selection_range_v3().unwrap(), 20..80);
        assert!(
            ResidentFeatureScreeningScopeV2::from_screening_plan_v2(100, 20, 80, 81, 100)
                .selection_range_v3()
                .is_err()
        );
    }

    fn screening_scope_fixture_v2(
        selection_row_start: u64,
        selection_row_end: u64,
        holdout_row_start: u64,
    ) -> ResidentFeatureScreeningScopeV2 {
        ResidentFeatureScreeningScopeV2::from_screening_plan_v2(
            1_000,
            selection_row_start,
            selection_row_end,
            holdout_row_start,
            1_000,
        )
    }

    #[test]
    fn stage1_is_resolved_inside_the_capped_selection_suffix() {
        let scope = screening_scope_fixture_v2(200, 800, 800);
        assert_eq!(scope.parent_row_count(), 1_000);
        let recent = scope
            .resolve_stage1_v2(0.25, Stage1Window::MostRecent)
            .expect("most-recent Stage1 inside selection");
        assert_eq!(
            (recent.stage1_row_start(), recent.stage1_row_end()),
            (650, 800)
        );

        let earliest = scope
            .resolve_stage1_v2(0.25, Stage1Window::Earliest)
            .expect("earliest Stage1 inside selection");
        assert_eq!(
            (earliest.stage1_row_start(), earliest.stage1_row_end()),
            (200, 350)
        );
    }

    #[test]
    fn detached_holdout_partition_is_rejected() {
        let error = screening_scope_fixture_v2(200, 800, 799)
            .resolve_stage1_v2(0.25, Stage1Window::MostRecent)
            .expect_err("detached screening scope must fail closed");
        assert!(error.contains("exact selection/holdout partition"));
    }

    #[test]
    fn empty_stage1_resolution_is_rejected() {
        let error = screening_scope_fixture_v2(799, 800, 800)
            .resolve_stage1_v2(0.01, Stage1Window::MostRecent)
            .expect_err("floor-rounded empty Stage1 must fail closed");
        assert!(error.contains("empty selection window"));
    }

    #[test]
    fn stage1_time_uses_exact_capped_window_endpoints_including_calendar_gaps() {
        let scope = ResidentFeatureScreeningScopeV2::from_screening_plan_v2(10, 2, 8, 8, 10);
        // Irregular actual timestamps deliberately differ from rows * timeframe.
        let timestamps = [
            0,
            1,
            2,
            3,
            4,
            86_400_000,
            345_600_000,
            345_630_000,
            900_000_000,
            999_000_000,
        ];
        for (window, start, end) in [
            (Stage1Window::MostRecent, 5, 8),
            (Stage1Window::Earliest, 2, 5),
        ] {
            let rows = scope.resolve_stage1_v2(0.5, window).unwrap();
            assert_eq!(
                (rows.stage1_row_start(), rows.stage1_row_end()),
                (start as u64, end as u64)
            );
            let time = ResidentSelectionStage1TimeScopeV2::from_pinned_bounds_v2(
                10,
                rows,
                [7; 32],
                timestamps[start],
                timestamps[end - 1],
            )
            .unwrap();
            assert_eq!(
                time.timestamp_bounds_ms(),
                (timestamps[start], timestamps[end - 1])
            );
            assert_eq!(
                time.evaluation_span_days().to_bits(),
                ((timestamps[end - 1] - timestamps[start]) as f64 / 86_400_000.0).to_bits()
            );
            time.validate_binding_v2(10, start as u64, end as u64, [7; 32])
                .unwrap();
            assert!(time.validate_binding_v2(10, 0, 8, [7; 32]).is_err());
            assert!(
                time.validate_binding_v2(10, start as u64, end as u64, [8; 32])
                    .is_err()
            );
        }
    }

    #[test]
    fn stage1_time_rejects_missing_order_overflow_and_single_sample() {
        let rows = ResolvedResidentSelectionStage1ScopeV2 {
            stage1_row_start: 2,
            stage1_row_end: 4,
        };
        for (first, last) in [(10, 10), (11, 10), (i64::MIN, i64::MAX)] {
            assert!(
                ResidentSelectionStage1TimeScopeV2::from_pinned_bounds_v2(
                    5, rows, [7; 32], first, last
                )
                .is_err()
            );
        }
        let single = ResolvedResidentSelectionStage1ScopeV2 {
            stage1_row_start: 2,
            stage1_row_end: 3,
        };
        assert!(
            ResidentSelectionStage1TimeScopeV2::from_pinned_bounds_v2(5, single, [7; 32], 1, 2)
                .is_err()
        );
        assert!(
            ResidentSelectionStage1TimeScopeV2::from_pinned_bounds_v2(3, rows, [7; 32], 1, 2)
                .is_err()
        );
        assert!(
            ResidentSelectionStage1TimeScopeV2::from_pinned_bounds_v2(5, rows, [0; 32], 1, 2)
                .is_err()
        );
    }
}
