//! Schema versioning for every operator-facing persisted contract.
//!
//! Phase D4. Establishes a uniform versioning pattern that the
//! Flutter rewrite (and any future client) can rely on when
//! reading on-disk artifacts the Rust backend wrote.
//!
//! ## The pattern
//!
//! Every versioned contract carries a `schema_version: u32` field
//! at the top of the struct. The field defaults to **v1** when
//! missing (for backward compatibility with pre-versioning files
//! already on operators' disks), so adding the field to an
//! existing struct is a NON-BREAKING change.
//!
//! When the on-disk schema changes (fields renamed/removed/typed
//! differently in a way `#[serde(default)]` can't bridge), the
//! struct's `CURRENT_SCHEMA_VERSION` constant bumps and a
//! migration function converts the older variant to the newer.
//!
//! On load:
//! - `version > MAX_READABLE` → fail-loud ("please update the app")
//! - `version < CURRENT` → run forward migration
//! - `version == CURRENT` → no-op
//!
//! On save:
//! - Always write `version == CURRENT`
//!
//! ## What's covered today
//!
//! - `neoethos_app::app_services::broker_config::BrokerSettingsState`
//!   (`broker_credentials.toml`)
//! - `neoethos_core::system::HardwareProfile` (`hardware_profile.json`)
//! - `neoethos_models::runtime::RuntimeArtifactMetadata` (embedded in
//!   each expert's saved-artifact metadata.json)
//! - `neoethos_core::symbol_metadata::SymbolMetadataTable`
//!   (`symbol_metadata.json`)
//!
//! ## What's NOT covered (deferred follow-ups)
//!
//! - `WizardStateFile` (`wizard_state.json`) — already has a
//!   `version: u32` field but it's a re-run counter, not a schema
//!   version. A small re-rename in a follow-up commit
//!   disambiguates without breaking compat.
//! - `config.yaml` — YAML rather than JSON/TOML; the existing
//!   `#[serde(default)]` on every field gives ad-hoc forward
//!   compat. Adding strict versioning to YAML is its own
//!   conversation with the operator.
//! - SQLite-backed contracts (`alpha_strategies`,
//!   `live_metrics`, `cycle_metrics`) — these get a SQLite
//!   `schema_version` row in a meta table when D4.x lands; SQL
//!   migrations are a different beast from struct migrations.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Newtype carrying a schema version. Wrapping primitive `u32`
/// in a tuple struct lets the compiler distinguish a schema
/// version from any other `u32` (counters, indices, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SchemaVersion(pub u32);

impl SchemaVersion {
    pub const fn new(version: u32) -> Self {
        Self(version)
    }
    pub const fn value(self) -> u32 {
        self.0
    }
}

impl fmt::Display for SchemaVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Error returned when a persisted file's schema version is
/// outside the range the current binary can read.
///
/// The two-side check matters:
/// - `TooNew` → caller should prompt the operator to update
///   the app (the file came from a newer release).
/// - `TooOld` → migration was attempted but no migration path
///   exists for that specific older version (some intermediate
///   migration is missing).
#[derive(Debug, Clone)]
pub enum SchemaVersionError {
    /// The on-disk version is newer than the maximum this binary
    /// understands. The operator probably ran a newer build
    /// previously and then downgraded.
    TooNew {
        contract: &'static str,
        found: SchemaVersion,
        max_supported: SchemaVersion,
    },
    /// The on-disk version is older AND no migration is wired up
    /// to bring it forward. Distinct from "older but migrated
    /// successfully" — that path returns Ok(_).
    UnsupportedOldVersion {
        contract: &'static str,
        found: SchemaVersion,
        oldest_migratable: SchemaVersion,
    },
}

impl fmt::Display for SchemaVersionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooNew {
                contract,
                found,
                max_supported,
            } => write!(
                f,
                "schema version mismatch for {contract}: file is {found} but this build \
                 only reads up to {max_supported}. Please update the app."
            ),
            Self::UnsupportedOldVersion {
                contract,
                found,
                oldest_migratable,
            } => write!(
                f,
                "schema version mismatch for {contract}: file is {found} but the oldest \
                 version this build can migrate from is {oldest_migratable}. \
                 No migration path exists."
            ),
        }
    }
}

impl std::error::Error for SchemaVersionError {}

/// Trait implemented by every versioned persisted contract.
/// Provides a uniform way for IO helpers to query / enforce
/// schema-version invariants without hand-coded constants at
/// every call site.
pub trait HasSchemaVersion {
    /// Current schema version this build writes when saving.
    const CURRENT: SchemaVersion;
    /// Maximum on-disk version this build can read. Usually
    /// `CURRENT`; subclasses can set this lower if a particular
    /// future-compat policy is required.
    const MAX_READABLE: SchemaVersion = Self::CURRENT;
    /// Oldest on-disk version this build can migrate forward from.
    /// Files older than this are rejected with
    /// [`SchemaVersionError::UnsupportedOldVersion`].
    const OLDEST_MIGRATABLE: SchemaVersion = SchemaVersion::new(1);

    /// Schema version of THIS particular value. Implementations
    /// return the value of their `schema_version` field; the
    /// trait method exists so generic helpers can read it without
    /// knowing the concrete struct layout.
    fn schema_version(&self) -> SchemaVersion;
}

/// Validate that a loaded value's schema_version is within the
/// readable range. Caller should run this AFTER deserialisation
/// + AFTER any migrations, to enforce the post-migration version
/// is the current one.
pub fn check_schema_version_readable<T: HasSchemaVersion>(
    value: &T,
    contract: &'static str,
) -> std::result::Result<(), SchemaVersionError> {
    let version = value.schema_version();
    if version > T::MAX_READABLE {
        return Err(SchemaVersionError::TooNew {
            contract,
            found: version,
            max_supported: T::MAX_READABLE,
        });
    }
    if version < T::OLDEST_MIGRATABLE {
        return Err(SchemaVersionError::UnsupportedOldVersion {
            contract,
            found: version,
            oldest_migratable: T::OLDEST_MIGRATABLE,
        });
    }
    Ok(())
}

/// Convenience: serde-default helper that returns
/// `SchemaVersion(1)`. Used as `#[serde(default = "default_v1")]`
/// on the `schema_version` field of newly-versioned contracts so
/// existing on-disk files (without the field) are treated as v1.
pub fn default_v1() -> SchemaVersion {
    SchemaVersion::new(1)
}

/// Convenience anyhow-flavoured wrapper around
/// [`check_schema_version_readable`] for call sites that already
/// use `anyhow::Result`.
pub fn ensure_schema_version_readable<T: HasSchemaVersion>(
    value: &T,
    contract: &'static str,
) -> Result<()> {
    check_schema_version_readable(value, contract).map_err(anyhow::Error::from)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// In-test struct simulating a contract that uses the trait.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct DummyContract {
        #[serde(default = "default_v1")]
        schema_version: SchemaVersion,
        payload: String,
    }

    impl HasSchemaVersion for DummyContract {
        const CURRENT: SchemaVersion = SchemaVersion::new(3);
        const MAX_READABLE: SchemaVersion = SchemaVersion::new(3);
        // Override to demonstrate the trait's three-version policy.
        const OLDEST_MIGRATABLE: SchemaVersion = SchemaVersion::new(1);

        fn schema_version(&self) -> SchemaVersion {
            self.schema_version
        }
    }

    #[test]
    fn schema_version_round_trips_through_json() {
        let v = SchemaVersion::new(7);
        let json = serde_json::to_string(&v).expect("ser");
        assert_eq!(json, "7");
        let parsed: SchemaVersion = serde_json::from_str(&json).expect("de");
        assert_eq!(parsed, v);
    }

    #[test]
    fn schema_version_display_uses_v_prefix() {
        assert_eq!(SchemaVersion::new(1).to_string(), "v1");
        assert_eq!(SchemaVersion::new(42).to_string(), "v42");
    }

    #[test]
    fn default_v1_returns_version_one() {
        assert_eq!(default_v1(), SchemaVersion::new(1));
    }

    #[test]
    fn deserialise_without_version_field_defaults_to_v1() {
        // Pre-versioning files won't carry `schema_version`; the
        // `#[serde(default = "default_v1")]` attribute kicks in.
        let raw = r#"{"payload":"hello"}"#;
        let dc: DummyContract = serde_json::from_str(raw).expect("de");
        assert_eq!(dc.schema_version, SchemaVersion::new(1));
        assert_eq!(dc.payload, "hello");
    }

    #[test]
    fn deserialise_with_explicit_version_keeps_it() {
        let raw = r#"{"schema_version":2,"payload":"v2 data"}"#;
        let dc: DummyContract = serde_json::from_str(raw).expect("de");
        assert_eq!(dc.schema_version, SchemaVersion::new(2));
    }

    #[test]
    fn check_passes_when_version_in_range() {
        let dc = DummyContract {
            schema_version: SchemaVersion::new(2),
            payload: "ok".to_string(),
        };
        assert!(check_schema_version_readable(&dc, "DummyContract").is_ok());
    }

    #[test]
    fn check_rejects_too_new_version_with_clear_error() {
        let dc = DummyContract {
            schema_version: SchemaVersion::new(99),
            payload: "from future".to_string(),
        };
        let err =
            check_schema_version_readable(&dc, "DummyContract").expect_err("must reject too-new");
        match err {
            SchemaVersionError::TooNew {
                contract,
                found,
                max_supported,
            } => {
                assert_eq!(contract, "DummyContract");
                assert_eq!(found, SchemaVersion::new(99));
                assert_eq!(max_supported, SchemaVersion::new(3));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn check_rejects_too_old_unsupported_version() {
        // For DummyContract, OLDEST_MIGRATABLE is 1 — so v0 must
        // be rejected.
        let dc = DummyContract {
            schema_version: SchemaVersion::new(0),
            payload: "ancient".to_string(),
        };
        let err =
            check_schema_version_readable(&dc, "DummyContract").expect_err("must reject too-old");
        match err {
            SchemaVersionError::UnsupportedOldVersion { found, .. } => {
                assert_eq!(found, SchemaVersion::new(0));
            }
            other => panic!("wrong variant: {other:?}"),
        }
    }

    #[test]
    fn error_message_includes_contract_and_versions() {
        let dc = DummyContract {
            schema_version: SchemaVersion::new(99),
            payload: "".to_string(),
        };
        let err = check_schema_version_readable(&dc, "DummyContract").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("DummyContract"));
        assert!(msg.contains("v99"));
        assert!(msg.contains("v3"));
    }

    #[test]
    fn ensure_returns_anyhow_error_on_mismatch() {
        let dc = DummyContract {
            schema_version: SchemaVersion::new(99),
            payload: "".to_string(),
        };
        assert!(ensure_schema_version_readable(&dc, "DummyContract").is_err());
    }
}
