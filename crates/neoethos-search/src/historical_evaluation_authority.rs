use std::sync::Arc;

use anyhow::Result;

use crate::canonical_trendbar_research::{
    CanonicalTrendbarResearchExecutionContractV3, active_canonical_trendbar_research_execution_v3,
};

#[derive(Debug, Clone)]
pub(crate) enum HistoricalEvaluationAuthorityV1 {
    BrokerFinancialTruth,
    CanonicalTrendbarResearch(Arc<CanonicalTrendbarResearchExecutionContractV3>),
}

pub(crate) fn require_historical_evaluation_authority_v1() -> Result<HistoricalEvaluationAuthorityV1>
{
    if let Some(contract) = active_canonical_trendbar_research_execution_v3() {
        contract.validate()?;
        return Ok(HistoricalEvaluationAuthorityV1::CanonicalTrendbarResearch(
            contract,
        ));
    }
    neoethos_core::current_broker_financial_truth_capability_v1()
        .require(neoethos_core::BrokerFinancialOperationV1::HistoricalEvaluation)
        .map_err(anyhow::Error::new)?;
    Ok(HistoricalEvaluationAuthorityV1::BrokerFinancialTruth)
}

pub(crate) fn active_research_contract_v1()
-> Option<Arc<CanonicalTrendbarResearchExecutionContractV3>> {
    match require_historical_evaluation_authority_v1().ok()? {
        HistoricalEvaluationAuthorityV1::CanonicalTrendbarResearch(contract) => Some(contract),
        HistoricalEvaluationAuthorityV1::BrokerFinancialTruth => None,
    }
}
