//! Shared contract surface for artifact provenance, live-readiness, and runtime semantics.

mod envelope;
mod error;
mod live;
mod primitives;
mod temporal;

#[cfg(test)]
mod tests;

pub use envelope::*;
pub use error::*;
pub use live::*;
pub use primitives::*;
pub use temporal::*;

/// Stable schema version for the shared cross-subsystem artifact contract.
pub const ARTIFACT_SCHEMA_VERSION: u16 = 1;
