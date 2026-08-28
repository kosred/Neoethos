//! Canonical dataset identity and temporal contracts shared by every NeoEthos workspace.

#![forbid(unsafe_code)]

mod identity;
mod temporal;

pub use identity::{
    CTraderEnvironment, CanonicalDatasetIdentity, CanonicalDatasetScope, DatasetIdentityError,
};
pub use temporal::{
    BarTimestampConvention, CANONICAL_TIMEFRAMES, CanonicalTimeframe, TemporalContractError,
};
