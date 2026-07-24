//! Correctness-first contracts and executable reference walks for Prototype B/C.
//!
//! These routines deliberately make no performance claim. They define the exact
//! subset that the future GPU kernels must reproduce and provide deterministic
//! host references for the twelve-level parity harness.

use crate::gpu_native::engine::{EngineCapabilities, EngineStatus};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrototypeKind {
    AExactPersistent,
    BWarpCooperative,
    CSparseFirstHit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionDirection {
    Long,
    Short,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SameBarPrecedence {
    StopFirst,
    TargetFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitReason {
    StopLoss,
    TakeProfit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FirstHit {
    pub exit_bar: usize,
    pub reason: ExitReason,
}

#[derive(Debug, Clone, Copy)]
pub struct PricePath<'a> {
    pub highs: &'a [f64],
    pub lows: &'a [f64],
}

#[derive(Debug, Clone, Copy)]
pub struct FirstHitRequest {
    pub entry_bar: usize,
    pub last_bar: usize,
    pub direction: PositionDirection,
    pub stop_price: f64,
    pub target_price: f64,
    pub same_bar_precedence: SameBarPrecedence,
}

#[derive(Debug, Clone, Copy)]
pub struct EntryEvent {
    pub candidate_id: u64,
    pub scenario_id: u64,
    pub request: FirstHitRequest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseOutcome {
    pub candidate_id: u64,
    pub scenario_id: u64,
    pub hit: Option<FirstHit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubgroupSupport {
    pub available: bool,
    pub width: Option<u32>,
}

impl SubgroupSupport {
    pub const fn unavailable() -> Self {
        Self { available: false, width: None }
    }

    pub const fn known(width: u32) -> Self {
        Self { available: true, width: Some(width) }
    }

    pub fn usable_width(self) -> Option<usize> {
        let width = self.width?;
        if self.available && matches!(width, 8 | 16 | 32 | 64) {
            Some(width as usize)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StrategyPathProfile {
    pub fixed_at_entry: bool,
    pub adaptive_at_entry: bool,
    pub break_even: bool,
    pub trailing: bool,
    pub prop_firm_state: bool,
    pub event_density: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingReason {
    SparseStaticStops,
    DenseStaticStopsWithSubgroup,
    PathDependentStateRequiresExactWalk,
    InvalidProfileRequiresExactWalk,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutingDecision {
    pub prototype: PrototypeKind,
    pub reason: RoutingReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrototypeError {
    EmptyPricePath,
    PriceLengthMismatch { highs: usize, lows: usize },
    InvalidWindow { entry_bar: usize, last_bar: usize, rows: usize },
    NonFiniteLevel,
    UnsupportedSubgroupWidth(Option<u32>),
}

pub fn prototype_b_status(subgroup: SubgroupSupport) -> EngineStatus {
    if subgroup.usable_width().is_some() {
        EngineStatus::NotBenchmarked
    } else {
        EngineStatus::UnsupportedCapability
    }
}

pub const fn prototype_c_status() -> EngineStatus {
    EngineStatus::NotBenchmarked
}

pub const fn prototype_b_capabilities() -> EngineCapabilities {
    EngineCapabilities {
        fixed_stops: true,
        adaptive_stops: true,
        break_even: false,
        trailing: false,
        prop_firm_state: false,
        device_filtering: false,
        compact_readback: true,
    }
}

pub const fn prototype_c_capabilities() -> EngineCapabilities {
    EngineCapabilities {
        fixed_stops: true,
        adaptive_stops: true,
        break_even: false,
        trailing: false,
        prop_firm_state: false,
        device_filtering: false,
        compact_readback: true,
    }
}

pub fn route_strategy(profile: StrategyPathProfile, subgroup: SubgroupSupport) -> RoutingDecision {
    if profile.break_even || profile.trailing || profile.prop_firm_state {
        return RoutingDecision {
            prototype: PrototypeKind::AExactPersistent,
            reason: RoutingReason::PathDependentStateRequiresExactWalk,
        };
    }

    let static_stop = profile.fixed_at_entry ^ profile.adaptive_at_entry;
    let density_valid = profile.event_density.is_finite()
        && (0.0..=1.0).contains(&profile.event_density);
    if !static_stop || !density_valid {
        return RoutingDecision {
            prototype: PrototypeKind::AExactPersistent,
            reason: RoutingReason::InvalidProfileRequiresExactWalk,
        };
    }

    if profile.event_density <= 0.125 {
        RoutingDecision {
            prototype: PrototypeKind::CSparseFirstHit,
            reason: RoutingReason::SparseStaticStops,
        }
    } else if subgroup.usable_width().is_some() {
        RoutingDecision {
            prototype: PrototypeKind::BWarpCooperative,
            reason: RoutingReason::DenseStaticStopsWithSubgroup,
        }
    } else {
        RoutingDecision {
            prototype: PrototypeKind::AExactPersistent,
            reason: RoutingReason::InvalidProfileRequiresExactWalk,
        }
    }
}

pub fn reference_first_hit(path: PricePath<'_>, request: FirstHitRequest) -> Result<Option<FirstHit>, PrototypeError> {
    validate_request(path, request)?;
    for bar in request.entry_bar.saturating_add(1)..=request.last_bar {
        if let Some(reason) = hit_on_bar(path.highs[bar], path.lows[bar], request) {
            return Ok(Some(FirstHit { exit_bar: bar, reason }));
        }
    }
    Ok(None)
}

pub fn prototype_b_first_hit(
    path: PricePath<'_>,
    request: FirstHitRequest,
    subgroup: SubgroupSupport,
) -> Result<Option<FirstHit>, PrototypeError> {
    validate_request(path, request)?;
    let width = subgroup
        .usable_width()
        .ok_or(PrototypeError::UnsupportedSubgroupWidth(subgroup.width))?;

    let first_bar = request.entry_bar.saturating_add(1);
    let mut earliest: Option<FirstHit> = None;
    for lane in 0..width {
        let mut bar = first_bar.saturating_add(lane);
        while bar <= request.last_bar {
            if let Some(reason) = hit_on_bar(path.highs[bar], path.lows[bar], request) {
                let lane_hit = FirstHit { exit_bar: bar, reason };
                if earliest.map(|current| lane_hit.exit_bar < current.exit_bar).unwrap_or(true) {
                    earliest = Some(lane_hit);
                }
                break;
            }
            bar = bar.saturating_add(width);
        }
    }
    Ok(earliest)
}

pub fn prototype_c_event_first_hit(
    path: PricePath<'_>,
    events: &[EntryEvent],
) -> Result<Vec<SparseOutcome>, PrototypeError> {
    validate_path(path)?;
    events
        .iter()
        .map(|event| {
            reference_first_hit(path, event.request).map(|hit| SparseOutcome {
                candidate_id: event.candidate_id,
                scenario_id: event.scenario_id,
                hit,
            })
        })
        .collect()
}

fn validate_path(path: PricePath<'_>) -> Result<(), PrototypeError> {
    if path.highs.is_empty() || path.lows.is_empty() {
        return Err(PrototypeError::EmptyPricePath);
    }
    if path.highs.len() != path.lows.len() {
        return Err(PrototypeError::PriceLengthMismatch { highs: path.highs.len(), lows: path.lows.len() });
    }
    Ok(())
}

fn validate_request(path: PricePath<'_>, request: FirstHitRequest) -> Result<(), PrototypeError> {
    validate_path(path)?;
    if !request.stop_price.is_finite() || !request.target_price.is_finite() {
        return Err(PrototypeError::NonFiniteLevel);
    }
    if request.entry_bar >= request.last_bar || request.last_bar >= path.highs.len() {
        return Err(PrototypeError::InvalidWindow {
            entry_bar: request.entry_bar,
            last_bar: request.last_bar,
            rows: path.highs.len(),
        });
    }
    Ok(())
}

fn hit_on_bar(high: f64, low: f64, request: FirstHitRequest) -> Option<ExitReason> {
    if !high.is_finite() || !low.is_finite() {
        return None;
    }
    let (stop_hit, target_hit) = match request.direction {
        PositionDirection::Long => (low <= request.stop_price, high >= request.target_price),
        PositionDirection::Short => (high >= request.stop_price, low <= request.target_price),
    };

    match (stop_hit, target_hit, request.same_bar_precedence) {
        (true, true, SameBarPrecedence::StopFirst) | (true, false, _) => Some(ExitReason::StopLoss),
        (true, true, SameBarPrecedence::TargetFirst) | (false, true, _) => Some(ExitReason::TakeProfit),
        (false, false, _) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path<'a>(highs: &'a [f64], lows: &'a [f64]) -> PricePath<'a> {
        PricePath { highs, lows }
    }

    fn long_request(precedence: SameBarPrecedence) -> FirstHitRequest {
        FirstHitRequest {
            entry_bar: 0,
            last_bar: 5,
            direction: PositionDirection::Long,
            stop_price: 95.0,
            target_price: 105.0,
            same_bar_precedence: precedence,
        }
    }

    #[test]
    fn prototype_b_matches_serial_oracle_for_supported_widths() {
        let highs = [100.0, 102.0, 103.0, 106.0, 104.0, 101.0];
        let lows = [100.0, 99.0, 98.0, 97.0, 96.0, 94.0];
        let request = long_request(SameBarPrecedence::StopFirst);
        let expected = reference_first_hit(path(&highs, &lows), request).unwrap();
        for width in [8, 16, 32, 64] {
            assert_eq!(
                prototype_b_first_hit(path(&highs, &lows), request, SubgroupSupport::known(width)).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn same_bar_precedence_is_exact() {
        let highs = [100.0, 106.0];
        let lows = [100.0, 94.0];
        let mut request = FirstHitRequest {
            entry_bar: 0,
            last_bar: 1,
            direction: PositionDirection::Long,
            stop_price: 95.0,
            target_price: 105.0,
            same_bar_precedence: SameBarPrecedence::StopFirst,
        };
        assert_eq!(reference_first_hit(path(&highs, &lows), request).unwrap().unwrap().reason, ExitReason::StopLoss);
        request.same_bar_precedence = SameBarPrecedence::TargetFirst;
        assert_eq!(reference_first_hit(path(&highs, &lows), request).unwrap().unwrap().reason, ExitReason::TakeProfit);
    }

    #[test]
    fn prototype_c_preserves_event_identity_and_order() {
        let highs = [100.0, 101.0, 106.0, 103.0, 104.0, 105.0];
        let lows = [100.0, 99.0, 98.0, 94.0, 96.0, 97.0];
        let events = [
            EntryEvent { candidate_id: 11, scenario_id: 101, request: long_request(SameBarPrecedence::StopFirst) },
            EntryEvent {
                candidate_id: 22,
                scenario_id: 202,
                request: FirstHitRequest {
                    direction: PositionDirection::Short,
                    stop_price: 105.0,
                    target_price: 95.0,
                    ..long_request(SameBarPrecedence::StopFirst)
                },
            },
        ];
        let outcomes = prototype_c_event_first_hit(path(&highs, &lows), &events).unwrap();
        assert_eq!((outcomes[0].candidate_id, outcomes[0].scenario_id), (11, 101));
        assert_eq!((outcomes[1].candidate_id, outcomes[1].scenario_id), (22, 202));
    }

    #[test]
    fn routing_is_conservative_and_typed() {
        let path_dependent = route_strategy(
            StrategyPathProfile { fixed_at_entry: true, trailing: true, event_density: 0.01, ..StrategyPathProfile::default() },
            SubgroupSupport::known(32),
        );
        assert_eq!(path_dependent.prototype, PrototypeKind::AExactPersistent);

        let sparse = route_strategy(
            StrategyPathProfile { fixed_at_entry: true, event_density: 0.05, ..StrategyPathProfile::default() },
            SubgroupSupport::unavailable(),
        );
        assert_eq!(sparse.prototype, PrototypeKind::CSparseFirstHit);

        let dense = route_strategy(
            StrategyPathProfile { adaptive_at_entry: true, event_density: 0.5, ..StrategyPathProfile::default() },
            SubgroupSupport::known(32),
        );
        assert_eq!(dense.prototype, PrototypeKind::BWarpCooperative);
    }

    #[test]
    fn unsupported_subgroup_returns_typed_error() {
        let highs = [100.0, 101.0];
        let lows = [100.0, 99.0];
        let request = FirstHitRequest { last_bar: 1, ..long_request(SameBarPrecedence::StopFirst) };
        assert_eq!(
            prototype_b_first_hit(path(&highs, &lows), request, SubgroupSupport::known(12)),
            Err(PrototypeError::UnsupportedSubgroupWidth(Some(12)))
        );
        assert_eq!(prototype_b_status(SubgroupSupport::known(12)), EngineStatus::UnsupportedCapability);
    }
}
