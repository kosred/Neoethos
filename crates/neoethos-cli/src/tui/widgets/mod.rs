//! Custom widgets that the standard ratatui set does not cover.

pub mod kpi;
// `sparkline` was a real widget for fitness/equity curves but had no
// consumer wired up (no page rendered a fitness history). Deleted with
// #200 — re-introduce alongside the Strategies-page fitness panel that
// actually needs it; otherwise it's dead code that the warning system
// keeps re-flagging.
