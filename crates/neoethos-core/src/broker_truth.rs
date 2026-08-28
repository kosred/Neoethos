//! Core compatibility re-export for the dependency-leaf broker-truth boundary.
//!
//! The authority lives below core/search/app so evidence producers and exact
//! replay consumers can share one contract without a dependency cycle.

pub use neoethos_broker_truth::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_release_has_no_fabricated_success_state() {
        let capability = current_broker_financial_truth_capability_v1();
        let error = capability
            .require(BrokerFinancialOperationV1::HistoricalEvaluation)
            .expect_err("the semantic broker evidence phase has not run");

        assert_eq!(error.schema_version(), 1);
        assert_eq!(error.missing_evidence().len(), 5);
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
