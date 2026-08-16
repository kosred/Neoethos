use std::fmt;

/// Wire/schema version of the release gate that prevents NeoEthos from
/// inventing broker financial inputs.
pub const BROKER_FINANCIAL_TRUTH_SCHEMA_VERSION_V1: u16 = 1;

/// Stable machine-readable error code used by API/CLI/tests.
pub const BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1: &str =
    "BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1";

/// A financial operation that is forbidden until its broker evidence is
/// complete. Signal-only feature work is deliberately not represented here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BrokerFinancialOperationV1 {
    HistoricalEvaluation,
    HistoricalReplay,
    Promotion,
    RiskyMode,
    PropFirmMode,
    LiveTrading,
    LiveRiskAndPnl,
}

impl BrokerFinancialOperationV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HistoricalEvaluation => "historical_evaluation",
            Self::HistoricalReplay => "historical_replay",
            Self::Promotion => "promotion",
            Self::RiskyMode => "risky_mode",
            Self::PropFirmMode => "prop_firm_mode",
            Self::LiveTrading => "live_trading",
            Self::LiveRiskAndPnl => "live_risk_and_pnl",
        }
    }
}

/// Evidence required by the approved broker-truth boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MissingBrokerFinancialEvidenceV1 {
    SynchronizedHistoricalBidAsk,
    SynchronizedConversionLegs,
    ExactProtoOASymbolContract,
    BrokerPositionUnrealizedPnl,
    CloseDealReconciliation,
}

impl MissingBrokerFinancialEvidenceV1 {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SynchronizedHistoricalBidAsk => "synchronized_historical_bid_ask",
            Self::SynchronizedConversionLegs => "synchronized_conversion_legs",
            Self::ExactProtoOASymbolContract => "exact_proto_oa_symbol_contract",
            Self::BrokerPositionUnrealizedPnl => "broker_position_unrealized_pnl",
            Self::CloseDealReconciliation => "close_deal_reconciliation",
        }
    }
}

const CURRENT_MISSING_EVIDENCE: [MissingBrokerFinancialEvidenceV1; 5] = [
    MissingBrokerFinancialEvidenceV1::SynchronizedHistoricalBidAsk,
    MissingBrokerFinancialEvidenceV1::SynchronizedConversionLegs,
    MissingBrokerFinancialEvidenceV1::ExactProtoOASymbolContract,
    MissingBrokerFinancialEvidenceV1::BrokerPositionUnrealizedPnl,
    MissingBrokerFinancialEvidenceV1::CloseDealReconciliation,
];

/// Current release capability. There is intentionally no public constructor
/// for a verified permit: the separate broker-evidence phase must add the
/// typed artifacts and validation before this boundary can open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrokerFinancialTruthCapabilityV1 {
    schema_version: u16,
}

impl BrokerFinancialTruthCapabilityV1 {
    pub const fn current() -> Self {
        Self {
            schema_version: BROKER_FINANCIAL_TRUTH_SCHEMA_VERSION_V1,
        }
    }

    pub const fn schema_version(self) -> u16 {
        self.schema_version
    }

    pub const fn missing_evidence(self) -> &'static [MissingBrokerFinancialEvidenceV1] {
        &CURRENT_MISSING_EVIDENCE
    }

    pub fn require(
        self,
        operation: BrokerFinancialOperationV1,
    ) -> Result<BrokerFinancialTruthPermitV1, BrokerFinancialTruthErrorV1> {
        Err(BrokerFinancialTruthErrorV1 {
            operation,
            schema_version: self.schema_version,
            missing: &CURRENT_MISSING_EVIDENCE,
        })
    }
}

/// Proof token reserved for the future evidence-backed implementation.
/// External code cannot construct it and therefore cannot bypass the gate.
#[derive(Debug)]
pub struct BrokerFinancialTruthPermitV1 {
    operation: BrokerFinancialOperationV1,
}

impl BrokerFinancialTruthPermitV1 {
    /// Resolve one pip from the exact, provenance-bound broker symbol contract.
    ///
    /// The current release deliberately has no such contract and cannot create
    /// a permit. Keep this second fail-closed boundary so a future change to
    /// [`BrokerFinancialTruthCapabilityV1::require`] cannot accidentally make
    /// symbol-name pip heuristics executable. The broker-evidence phase must
    /// replace this refusal with a lookup bound to the permit's captured
    /// `ProtoOASymbol` evidence.
    pub fn exact_pip_size_v1(
        &self,
        _symbol: &str,
    ) -> Result<f64, BrokerFinancialTruthErrorV1> {
        Err(BrokerFinancialTruthErrorV1::unavailable_for(self.operation))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrokerFinancialTruthErrorV1 {
    operation: BrokerFinancialOperationV1,
    schema_version: u16,
    missing: &'static [MissingBrokerFinancialEvidenceV1],
}

impl BrokerFinancialTruthErrorV1 {
    pub const fn unavailable_for(operation: BrokerFinancialOperationV1) -> Self {
        Self {
            operation,
            schema_version: BROKER_FINANCIAL_TRUTH_SCHEMA_VERSION_V1,
            missing: &CURRENT_MISSING_EVIDENCE,
        }
    }

    pub const fn operation(self) -> BrokerFinancialOperationV1 {
        self.operation
    }

    pub const fn schema_version(self) -> u16 {
        self.schema_version
    }

    pub const fn missing_evidence(self) -> &'static [MissingBrokerFinancialEvidenceV1] {
        self.missing
    }
}

impl fmt::Display for BrokerFinancialTruthErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1} schema_version={} operation={} missing=",
            self.schema_version,
            self.operation.as_str()
        )?;
        for (index, item) in self.missing.iter().enumerate() {
            if index > 0 {
                f.write_str(",")?;
            }
            f.write_str(item.as_str())?;
        }
        Ok(())
    }
}

impl std::error::Error for BrokerFinancialTruthErrorV1 {}

pub const fn current_broker_financial_truth_capability_v1() -> BrokerFinancialTruthCapabilityV1 {
    BrokerFinancialTruthCapabilityV1::current()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_release_has_no_fabricated_success_state() {
        let capability = current_broker_financial_truth_capability_v1();
        let error = capability
            .require(BrokerFinancialOperationV1::HistoricalEvaluation)
            .expect_err("the broker evidence phase has not run");

        assert_eq!(error.schema_version(), 1);
        assert_eq!(error.missing_evidence(), CURRENT_MISSING_EVIDENCE);
        assert!(error.to_string().starts_with(
            "BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1 schema_version=1 operation=historical_evaluation"
        ));
    }

    #[test]
    fn legacy_risk_cost_accessors_require_a_broker_truth_permit() {
        let risk = crate::Settings::default().risk;

        let spread = risk.session_spread_pips();
        let spread_error = spread.expect_err("session spread arithmetic must be sealed");
        assert!(spread_error.contains(BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1));

        let commission = risk.round_trip_commission_per_lot();
        let commission_error = commission.expect_err("commission arithmetic must be sealed");
        assert!(
            commission_error
                .to_string()
                .contains(BROKER_FINANCIAL_TRUTH_UNAVAILABLE_V1)
        );
    }
}
