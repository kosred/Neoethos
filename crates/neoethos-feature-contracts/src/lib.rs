//! Canonical feature-plan identity and concrete dataset provenance contracts.

#![forbid(unsafe_code)]

mod identity;
mod source_manifest;

pub use identity::{
    DatasetFeatureArtifactProvenanceIdentityV1, DatasetFeatureArtifactProvenanceV1,
    FeatureContractError, FeatureNodeV1, FeatureOperationTagV1, FeatureOutputV1,
    FeatureParameterV1, FeatureParameterValueV1, FeaturePlanIdentityV1, FeaturePlanV1,
    SourceArtifactBindingV1, SourceSegmentV1,
};
pub use source_manifest::{
    RelevantDependencySetIdentityV1, RelevantDependencySetV1, RelevantDependencySourceKindV1,
    RelevantDependencyV1, SemanticSourceEntryV1, SemanticSourceKindV1,
    SemanticSourceManifestIdentityV1, SemanticSourceManifestV1, SemanticSourceSetIdentityV1,
    SemanticSourceSetV1,
};
