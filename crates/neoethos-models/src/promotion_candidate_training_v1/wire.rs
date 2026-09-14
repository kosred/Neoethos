//! Versioned transport only: V1 stays byte-for-byte inline; V2 shares one
//! receipt inside the same handoff; V3 losslessly compresses that one receipt.
//! No filesystem lookup or ambient authority participates in expansion.

use super::*;
use neoethos_search::canonical_trendbar_research::CanonicalTrendbarResearchExecutionContractRefV1;
use neoethos_search::data_selection::{
    CanonicalSearchArtifactScopeRefV1, CanonicalSearchArtifactScopeV2,
};
use serde::{Deserializer, Serializer};

/// Keep exactly the original four-field ordering. The in-memory V2 locked
/// object owns a receipt for independent typed use, but this wire never does.
#[derive(Serialize)]
struct LockedPortfolioWrite<'a> {
    schema: &'a str,
    version: u16,
    canonical_json: &'a str,
    identity_sha256: &'a str,
}

impl<'a> From<&'a PromotionCandidateLockedPortfolioV1> for LockedPortfolioWrite<'a> {
    fn from(value: &'a PromotionCandidateLockedPortfolioV1) -> Self {
        Self {
            schema: &value.schema,
            version: value.version,
            canonical_json: &value.canonical_json,
            identity_sha256: &value.identity_sha256,
        }
    }
}

#[derive(Serialize)]
struct HandoffWriteV1<'a> {
    schema: &'a str,
    version: u16,
    canonical_series: &'a CanonicalDatasetSeriesReceiptV1,
    #[serde(serialize_with = "serialize_timeframe_v1")]
    base_timeframe: CanonicalTimeframe,
    search_input_receipt: &'a CanonicalSearchInputReceiptV2,
    screening_contract: &'a CanonicalTrendbarResearchExecutionContractV3,
    locked_portfolio: LockedPortfolioWrite<'a>,
    oos_cutoff_ms: i64,
    purge_bars: usize,
    broker_authority: &'a PromotionCandidateBrokerAuthorityIdentityV1,
    training_config: &'a PromotionCandidateTrainingConfigIdentityV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    discovery_holdout_scope: Option<&'a CanonicalSearchArtifactScopeV2>,
}

#[derive(Serialize)]
struct HandoffWriteV2<'a> {
    schema: &'a str,
    version: u16,
    canonical_series: &'a CanonicalDatasetSeriesReceiptV1,
    #[serde(serialize_with = "serialize_timeframe_v1")]
    base_timeframe: CanonicalTimeframe,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_input_receipt: Option<&'a CanonicalSearchInputReceiptV2>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compressed_search_input_receipt: Option<&'a codec::CompressedReceipt>,
    screening_contract: CanonicalTrendbarResearchExecutionContractRefV1,
    locked_portfolio: LockedPortfolioWrite<'a>,
    oos_cutoff_ms: i64,
    purge_bars: usize,
    broker_authority: &'a PromotionCandidateBrokerAuthorityIdentityV1,
    training_config: &'a PromotionCandidateTrainingConfigIdentityV1,
    #[serde(skip_serializing_if = "Option::is_none")]
    discovery_holdout_scope: Option<CanonicalSearchArtifactScopeRefV1>,
}

impl Serialize for PromotionCandidateTrainingHandoffV1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::Error;
        match (self.schema.as_str(), self.version) {
            (HANDOFF_SCHEMA_V1, SCHEMA_VERSION_V1) => {
                if self.locked_portfolio.schema != LOCKED_PORTFOLIO_SCHEMA_V1
                    || self.locked_portfolio.version != SCHEMA_VERSION_V1
                    || self.locked_portfolio.shared_receipt.is_some()
                    || self.compressed_search_input_receipt.is_some()
                {
                    return Err(S::Error::custom(
                        "legacy handoff cannot contain a shared locked portfolio",
                    ));
                }
                HandoffWriteV1 {
                    schema: &self.schema,
                    version: self.version,
                    canonical_series: &self.canonical_series,
                    base_timeframe: self.base_timeframe,
                    search_input_receipt: &self.search_input_receipt,
                    screening_contract: &self.screening_contract,
                    locked_portfolio: (&self.locked_portfolio).into(),
                    oos_cutoff_ms: self.oos_cutoff_ms,
                    purge_bars: self.purge_bars,
                    broker_authority: &self.broker_authority,
                    training_config: &self.training_config,
                    discovery_holdout_scope: self.discovery_holdout_scope.as_ref(),
                }
                .serialize(serializer)
            }
            (HANDOFF_SCHEMA_V2, SCHEMA_VERSION_V2) | (HANDOFF_SCHEMA_V3, SCHEMA_VERSION_V3) => {
                if self.locked_portfolio.schema != LOCKED_PORTFOLIO_SCHEMA_V2
                    || self.locked_portfolio.version != SCHEMA_VERSION_V2
                    || self.locked_portfolio.shared_receipt.as_deref()
                        != Some(&self.search_input_receipt)
                    || self.compressed_search_input_receipt.is_some()
                        != (self.version == SCHEMA_VERSION_V3)
                {
                    return Err(S::Error::custom(
                        "shared handoff locked portfolio is not bound to its outer receipt",
                    ));
                }
                if let Some(compressed) = &self.compressed_search_input_receipt {
                    compressed
                        .validate_against(&self.search_input_receipt)
                        .map_err(S::Error::custom)?;
                }
                // The refs validate the original contracts before eliding their
                // receipts. Do not call whole-handoff validate here: it itself
                // serializes for its bounded size check.
                self.screening_contract
                    .validate_against_receipt(&self.search_input_receipt)
                    .map_err(S::Error::custom)?;
                if let Some(scope) = &self.discovery_holdout_scope {
                    scope
                        .validate_against_receipt(&self.search_input_receipt)
                        .map_err(S::Error::custom)?;
                }
                HandoffWriteV2 {
                    schema: &self.schema,
                    version: self.version,
                    canonical_series: &self.canonical_series,
                    base_timeframe: self.base_timeframe,
                    search_input_receipt: (self.version == SCHEMA_VERSION_V2)
                        .then_some(&self.search_input_receipt),
                    compressed_search_input_receipt: self.compressed_search_input_receipt.as_ref(),
                    screening_contract:
                        CanonicalTrendbarResearchExecutionContractRefV1::from_contract(
                            &self.screening_contract,
                        )
                        .map_err(S::Error::custom)?,
                    locked_portfolio: (&self.locked_portfolio).into(),
                    oos_cutoff_ms: self.oos_cutoff_ms,
                    purge_bars: self.purge_bars,
                    broker_authority: &self.broker_authority,
                    training_config: &self.training_config,
                    discovery_holdout_scope: self
                        .discovery_holdout_scope
                        .as_ref()
                        .map(CanonicalSearchArtifactScopeRefV1::from_scope)
                        .transpose()
                        .map_err(S::Error::custom)?,
                }
                .serialize(serializer)
            }
            _ => Err(S::Error::custom(
                "unsupported promotion-candidate handoff schema/version",
            )),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LockedPortfolioRead {
    schema: String,
    version: u16,
    canonical_json: String,
    identity_sha256: String,
}

// These alternatives are local to the small reference fields on the new wire.
// The outer receipt is deserialized directly once, never buffered in an
// untagged whole-artifact enum or expanded through serde_json::Value.
#[derive(Deserialize)]
#[serde(untagged)]
enum ScreeningContractRead {
    Shared(CanonicalTrendbarResearchExecutionContractRefV1),
    Inline(CanonicalTrendbarResearchExecutionContractV3),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum HoldoutScopeRead {
    Shared(CanonicalSearchArtifactScopeRefV1),
    Inline(CanonicalSearchArtifactScopeV2),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HandoffRead {
    schema: String,
    version: u16,
    canonical_series: CanonicalDatasetSeriesReceiptV1,
    #[serde(deserialize_with = "deserialize_timeframe_v1")]
    base_timeframe: CanonicalTimeframe,
    #[serde(default, deserialize_with = "deserialize_present")]
    search_input_receipt: Option<CanonicalSearchInputReceiptV2>,
    #[serde(default, deserialize_with = "deserialize_present")]
    compressed_search_input_receipt: Option<codec::CompressedReceipt>,
    screening_contract: ScreeningContractRead,
    locked_portfolio: LockedPortfolioRead,
    oos_cutoff_ms: i64,
    purge_bars: usize,
    broker_authority: PromotionCandidateBrokerAuthorityIdentityV1,
    training_config: PromotionCandidateTrainingConfigIdentityV1,
    #[serde(default)]
    discovery_holdout_scope: Option<HoldoutScopeRead>,
}

// Optional by version, but an explicitly present null is not a valid receipt.
fn deserialize_present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

impl<'de> Deserialize<'de> for PromotionCandidateTrainingHandoffV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error;
        let wire = HandoffRead::deserialize(deserializer)?;
        let search_input_receipt = match (
            wire.schema.as_str(),
            wire.version,
            wire.search_input_receipt,
            wire.compressed_search_input_receipt.as_ref(),
        ) {
            (HANDOFF_SCHEMA_V1, SCHEMA_VERSION_V1, Some(receipt), None)
            | (HANDOFF_SCHEMA_V2, SCHEMA_VERSION_V2, Some(receipt), None) => receipt,
            (HANDOFF_SCHEMA_V3, SCHEMA_VERSION_V3, None, Some(compressed)) => {
                compressed.decode().map_err(D::Error::custom)?
            }
            _ => {
                return Err(D::Error::custom(
                    "handoff version requires exactly its own receipt representation",
                ));
            }
        };
        let (screening_contract, discovery_holdout_scope, shared_receipt) =
            match (wire.schema.as_str(), wire.version) {
                (HANDOFF_SCHEMA_V1, SCHEMA_VERSION_V1) => {
                    if wire.locked_portfolio.schema != LOCKED_PORTFOLIO_SCHEMA_V1
                        || wire.locked_portfolio.version != SCHEMA_VERSION_V1
                    {
                        return Err(D::Error::custom(
                            "legacy handoff has a non-legacy locked portfolio",
                        ));
                    }
                    let ScreeningContractRead::Inline(contract) = wire.screening_contract else {
                        return Err(D::Error::custom(
                            "legacy handoff requires an inline screening contract",
                        ));
                    };
                    let holdout = match wire.discovery_holdout_scope {
                        None => None,
                        Some(HoldoutScopeRead::Inline(scope)) => Some(scope),
                        Some(HoldoutScopeRead::Shared(_)) => {
                            return Err(D::Error::custom(
                                "legacy handoff cannot contain a shared holdout reference",
                            ));
                        }
                    };
                    (contract, holdout, None)
                }
                (HANDOFF_SCHEMA_V2, SCHEMA_VERSION_V2) | (HANDOFF_SCHEMA_V3, SCHEMA_VERSION_V3) => {
                    if wire.locked_portfolio.schema != LOCKED_PORTFOLIO_SCHEMA_V2
                        || wire.locked_portfolio.version != SCHEMA_VERSION_V2
                    {
                        return Err(D::Error::custom(
                            "shared handoff requires the compact locked portfolio schema",
                        ));
                    }
                    let ScreeningContractRead::Shared(contract) = wire.screening_contract else {
                        return Err(D::Error::custom(
                            "shared handoff cannot contain an inline screening receipt",
                        ));
                    };
                    let contract = contract
                        .attach(&search_input_receipt)
                        .map_err(D::Error::custom)?;
                    let holdout = match wire.discovery_holdout_scope {
                        None => None,
                        Some(HoldoutScopeRead::Shared(scope)) => Some(
                            scope
                                .attach(&search_input_receipt)
                                .map_err(D::Error::custom)?,
                        ),
                        Some(HoldoutScopeRead::Inline(_)) => {
                            return Err(D::Error::custom(
                                "shared handoff cannot contain an inline holdout receipt",
                            ));
                        }
                    };
                    (
                        contract,
                        holdout,
                        Some(Box::new(search_input_receipt.clone())),
                    )
                }
                _ => {
                    return Err(D::Error::custom(
                        "unsupported promotion-candidate handoff schema/version",
                    ));
                }
            };
        // Preserve deferred validation of legacy in-memory fixtures. Ref
        // attachment above is only the new transport boundary; all portfolio,
        // cutoff, configuration and size checks remain in parent validate().
        Ok(Self {
            schema: wire.schema,
            version: wire.version,
            canonical_series: wire.canonical_series,
            base_timeframe: wire.base_timeframe,
            search_input_receipt,
            compressed_search_input_receipt: wire.compressed_search_input_receipt,
            screening_contract,
            locked_portfolio: PromotionCandidateLockedPortfolioV1 {
                schema: wire.locked_portfolio.schema,
                version: wire.locked_portfolio.version,
                canonical_json: wire.locked_portfolio.canonical_json,
                identity_sha256: wire.locked_portfolio.identity_sha256,
                shared_receipt,
            },
            oos_cutoff_ms: wire.oos_cutoff_ms,
            purge_bars: wire.purge_bars,
            broker_authority: wire.broker_authority,
            training_config: wire.training_config,
            discovery_holdout_scope,
        })
    }
}
