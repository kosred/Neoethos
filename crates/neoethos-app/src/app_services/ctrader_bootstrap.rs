// 2026-08-08 dead-code purge: this module once carried the chunked-bootstrap
// lane (plan_bootstrap_chunks / bootstrap_with_fetcher /
// bootstrap_from_ctrader_history) whose only trigger was the test harness —
// production /data/bootstrap is a filesystem scan. The lane was deleted, and
// with it its now-orphaned private support code: the local-coverage
// inspection helpers (clean_normalized_bars, trailing_year_range_ns,
// inspect_local_bar_coverage/-_or_empty, CoverageSegment,
// LocalCoverageReport and the fx-session gap math), which had zero
// consumers outside this file after the lane came down. What remains is the
// one live DTO: `NormalizedBar`, the normalized OHLCV row shared by
// `bootstrap_writer` (vortex writes) and `broker_api` (history import).

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedBar {
    pub timestamp_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}
