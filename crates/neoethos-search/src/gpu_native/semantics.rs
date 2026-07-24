//! Versioned canonical trading and discovery semantics descriptors.
//!
//! The hashes in this module describe behaviour, not source files or prose. A
//! semantic change must update the corresponding descriptor/version and will
//! therefore be visible in benchmark and validation artifacts.

use neoethos_core::utils::fnv1a64;

pub const TRADING_SEMANTICS_VERSION: u32 = 1;
pub const DISCOVERY_SEMANTICS_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TradingSemanticsDescriptor {
    pub version: u32,
    pub signal_threshold_rule: &'static str,
    pub smc_gate_rule: &'static str,
    pub entry_timing_rule: &'static str,
    pub same_bar_precedence: &'static str,
    pub fixed_stop_rule: &'static str,
    pub adaptive_stop_rule: &'static str,
    pub trailing_rule: &'static str,
    pub break_even_rule: &'static str,
    pub cost_rule: &'static str,
    pub sizing_rule: &'static str,
    pub equity_drawdown_rule: &'static str,
    pub calendar_boundary_rule: &'static str,
    pub maximum_hold_rule: &'static str,
}

impl TradingSemanticsDescriptor {
    pub const fn canonical() -> Self {
        Self {
            version: TRADING_SEMANTICS_VERSION,
            signal_threshold_rule: "inclusive-long-ge-inclusive-short-le-v1",
            smc_gate_rule: "enabled-weight-sum-capped-threshold-directional-match-v1",
            entry_timing_rule: "prior-bar-signal-next-bar-fill-v1",
            same_bar_precedence: "canonical-stop-target-precedence-v1",
            fixed_stop_rule: "entry-captured-fixed-pips-v1",
            adaptive_stop_rule: "entry-captured-volatility-base-times-gene-multiplier-v1",
            trailing_rule: "canonical-monotonic-trailing-v1",
            break_even_rule: "canonical-r-multiple-trigger-v1",
            cost_rule: "spread-commission-swap-conversion-v1",
            sizing_rule: "confidence-scaled-equity-risk-band-v1",
            equity_drawdown_rule: "trade-equity-peak-drawdown-v1",
            calendar_boundary_rule: "typed-month-day-prop-firm-boundaries-v1",
            maximum_hold_rule: "configured-bars-forced-exit-v1",
        }
    }

    pub fn canonical_bytes(self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32(&mut out, self.version);
        for field in [
            self.signal_threshold_rule,
            self.smc_gate_rule,
            self.entry_timing_rule,
            self.same_bar_precedence,
            self.fixed_stop_rule,
            self.adaptive_stop_rule,
            self.trailing_rule,
            self.break_even_rule,
            self.cost_rule,
            self.sizing_rule,
            self.equity_drawdown_rule,
            self.calendar_boundary_rule,
            self.maximum_hold_rule,
        ] {
            push_str(&mut out, field);
        }
        out
    }

    pub fn hash(self) -> u64 {
        fnv1a64(&self.canonical_bytes())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiscoverySemanticsDescriptor {
    pub version: u32,
    pub fitness_formula: &'static str,
    pub rank_quantization: &'static str,
    pub tie_breaking: &'static str,
    pub candidate_identity: &'static str,
    pub filtering_policy: &'static str,
    pub validation_verdict_policy: &'static str,
    pub walkforward_policy: &'static str,
    pub cpcv_policy: &'static str,
    pub pbo_policy: &'static str,
    pub survivor_selection_policy: &'static str,
    pub portfolio_selection_policy: &'static str,
}

impl DiscoverySemanticsDescriptor {
    pub const fn canonical() -> Self {
        Self {
            version: DISCOVERY_SEMANTICS_VERSION,
            fitness_formula: "mode-scoped-canonical-fitness-v1",
            rank_quantization: "integer-rank-key-explicit-scales-v1",
            tie_breaking: "rank-fields-signature-canonical-gene-bytes-trial-id-v1",
            candidate_identity: "semantic-gene-bytes-plus-stable-trial-id-v1",
            filtering_policy: "typed-discovery-filter-profile-v1",
            validation_verdict_policy: "wf-and-cpcv-and-pbo-v1",
            walkforward_policy: "purged-embargoed-splits-v1",
            cpcv_policy: "typed-combinatorial-purged-cv-v1",
            pbo_policy: "cscv-logit-rank-v1",
            survivor_selection_policy: "typed-evolution-selection-policy-v1",
            portfolio_selection_policy: "quality-then-correlation-diversification-v1",
        }
    }

    pub fn canonical_bytes(self) -> Vec<u8> {
        let mut out = Vec::new();
        push_u32(&mut out, self.version);
        for field in [
            self.fitness_formula,
            self.rank_quantization,
            self.tie_breaking,
            self.candidate_identity,
            self.filtering_policy,
            self.validation_verdict_policy,
            self.walkforward_policy,
            self.cpcv_policy,
            self.pbo_policy,
            self.survivor_selection_policy,
            self.portfolio_selection_policy,
        ] {
            push_str(&mut out, field);
        }
        out
    }

    pub fn hash(self) -> u64 {
        fnv1a64(&self.canonical_bytes())
    }
}

pub fn trading_semantics_hash() -> u64 {
    TradingSemanticsDescriptor::canonical().hash()
}

pub fn discovery_semantics_hash() -> u64 {
    DiscoverySemanticsDescriptor::canonical().hash()
}

pub fn semantics_hash_hex(hash: u64) -> String {
    format!("{hash:016x}")
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_str(out: &mut Vec<u8>, value: &str) {
    push_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_hashes_are_stable_and_distinct() {
        let trading = trading_semantics_hash();
        let discovery = discovery_semantics_hash();
        assert_eq!(trading, TradingSemanticsDescriptor::canonical().hash());
        assert_eq!(discovery, DiscoverySemanticsDescriptor::canonical().hash());
        assert_ne!(trading, discovery);
        assert_eq!(semantics_hash_hex(trading).len(), 16);
    }

    #[test]
    fn descriptor_change_changes_hash() {
        let canonical = TradingSemanticsDescriptor::canonical();
        let changed = TradingSemanticsDescriptor {
            same_bar_precedence: "different-rule",
            ..canonical
        };
        assert_ne!(canonical.hash(), changed.hash());
    }

    #[test]
    fn bytes_use_explicit_little_endian_version_and_lengths() {
        let bytes = TradingSemanticsDescriptor::canonical().canonical_bytes();
        assert_eq!(&bytes[..4], &TRADING_SEMANTICS_VERSION.to_le_bytes());
        let first_length = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        assert_eq!(
            first_length,
            TradingSemanticsDescriptor::canonical()
                .signal_threshold_rule
                .len()
        );
    }
}
