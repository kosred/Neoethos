use std::collections::HashSet;

use sha2::{Digest, Sha256};

use crate::FeatureContractError;
use crate::identity::hex;

const SOURCE_MANIFEST_DOMAIN_V1: &[u8] = b"neoethos.semantic-source-manifest.v1\0";
const SOURCE_MANIFEST_VERSION_V1: u16 = 1;
const RELEVANT_DEPENDENCIES_DOMAIN_V1: &[u8] = b"neoethos.relevant-dependencies.v1\0";
const RELEVANT_DEPENDENCIES_VERSION_V1: u16 = 1;
const SOURCE_SET_DOMAIN_V1: &[u8] = b"neoethos.semantic-source-set.v1\0";
const SOURCE_SET_VERSION_V1: u16 = 1;
const MAX_PATH_BYTES: usize = 4 * 1024;
const MAX_FIELD_BYTES: usize = 4 * 1024;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SemanticSourceKindV1 {
    Utf8Text = 1,
    RawBinary = 2,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GeneratedSourceV1 {
    generator_path: String,
    input_paths: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticSourceEntryV1 {
    path: String,
    payload_kind: SemanticSourceKindV1,
    payload_hash: [u8; 32],
    generated: Option<GeneratedSourceV1>,
}

impl SemanticSourceEntryV1 {
    pub fn from_bytes(
        path: impl Into<String>,
        payload_kind: SemanticSourceKindV1,
        payload: &[u8],
    ) -> Result<Self, FeatureContractError> {
        Self::build(path, payload_kind, payload, None)
    }

    pub fn generated(
        path: impl Into<String>,
        payload_kind: SemanticSourceKindV1,
        payload: &[u8],
        generator_path: impl Into<String>,
        mut input_paths: Vec<String>,
    ) -> Result<Self, FeatureContractError> {
        let generator_path = generator_path.into();
        validate_repository_path(&generator_path)?;
        for input in &input_paths {
            validate_repository_path(input)?;
        }
        input_paths.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        reject_duplicates(
            "generated input path",
            input_paths.iter().map(String::as_str),
        )?;
        Self::build(
            path,
            payload_kind,
            payload,
            Some(GeneratedSourceV1 {
                generator_path,
                input_paths,
            }),
        )
    }

    fn build(
        path: impl Into<String>,
        payload_kind: SemanticSourceKindV1,
        payload: &[u8],
        generated: Option<GeneratedSourceV1>,
    ) -> Result<Self, FeatureContractError> {
        let path = path.into();
        validate_repository_path(&path)?;
        let canonical_payload = canonical_payload(payload_kind, payload)?;
        Ok(Self {
            path,
            payload_kind,
            payload_hash: Sha256::digest(&canonical_payload).into(),
            generated,
        })
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub const fn payload_kind(&self) -> SemanticSourceKindV1 {
        self.payload_kind
    }

    pub const fn payload_hash(&self) -> &[u8; 32] {
        &self.payload_hash
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SemanticSourceManifestIdentityV1([u8; 32]);

impl SemanticSourceManifestIdentityV1 {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticSourceManifestV1 {
    entries: Vec<SemanticSourceEntryV1>,
    canonical_bytes: Vec<u8>,
    identity: SemanticSourceManifestIdentityV1,
}

impl SemanticSourceManifestV1 {
    pub fn new(mut entries: Vec<SemanticSourceEntryV1>) -> Result<Self, FeatureContractError> {
        if entries.is_empty() {
            return Err(FeatureContractError::new(
                "semantic source manifest must not be empty",
            ));
        }
        entries.sort_by(|left, right| left.path.as_bytes().cmp(right.path.as_bytes()));
        reject_duplicates(
            "semantic source path",
            entries.iter().map(|entry| entry.path.as_str()),
        )?;
        let mut case_folded = HashSet::with_capacity(entries.len());
        for entry in &entries {
            if !case_folded.insert(entry.path.to_lowercase()) {
                return Err(FeatureContractError::new(format!(
                    "semantic source paths collide under case folding at `{}`",
                    entry.path
                )));
            }
        }
        let canonical_bytes = encode_source_manifest(&entries)?;
        let identity = SemanticSourceManifestIdentityV1(Sha256::digest(&canonical_bytes).into());
        Ok(Self {
            entries,
            canonical_bytes,
            identity,
        })
    }

    pub fn entries(&self) -> &[SemanticSourceEntryV1] {
        &self.entries
    }

    pub const fn identity(&self) -> SemanticSourceManifestIdentityV1 {
        self.identity
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RelevantDependencySourceV1 {
    Registry {
        registry_url: String,
    },
    Git {
        repository_url: String,
        immutable_revision: String,
    },
    RepositoryPath {
        path: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RelevantDependencySourceKindV1 {
    Registry,
    Git,
    RepositoryPath,
}

impl RelevantDependencySourceV1 {
    const fn tag(&self) -> u8 {
        match self {
            Self::Registry { .. } => 1,
            Self::Git { .. } => 2,
            Self::RepositoryPath { .. } => 3,
        }
    }

    fn canonical_identity(&self) -> String {
        match self {
            Self::Registry { registry_url } => registry_url.clone(),
            Self::Git {
                repository_url,
                immutable_revision,
            } => format!("{repository_url}#{immutable_revision}"),
            Self::RepositoryPath { path } => path.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelevantDependencyV1 {
    package_name: String,
    resolved_version: String,
    source: RelevantDependencySourceV1,
    checksum_or_source_manifest_hash: [u8; 32],
    enabled_features: Vec<String>,
}

impl RelevantDependencyV1 {
    pub fn registry(
        package_name: impl Into<String>,
        resolved_version: impl Into<String>,
        registry_url: impl Into<String>,
        checksum: [u8; 32],
        enabled_features: Vec<String>,
    ) -> Result<Self, FeatureContractError> {
        let registry_url = registry_url.into();
        validate_canonical_url("registry URL", &registry_url)?;
        Self::build(
            package_name,
            resolved_version,
            RelevantDependencySourceV1::Registry { registry_url },
            checksum,
            enabled_features,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn git(
        package_name: impl Into<String>,
        resolved_version: impl Into<String>,
        repository_url: impl Into<String>,
        immutable_revision: impl Into<String>,
        source_manifest_hash: [u8; 32],
        enabled_features: Vec<String>,
    ) -> Result<Self, FeatureContractError> {
        let repository_url = repository_url.into();
        let immutable_revision = immutable_revision.into();
        validate_canonical_url("Git repository URL", &repository_url)?;
        if !matches!(immutable_revision.len(), 40 | 64)
            || !immutable_revision
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(FeatureContractError::new(
                "Git dependency requires an exact 40- or 64-hex immutable revision",
            ));
        }
        Self::build(
            package_name,
            resolved_version,
            RelevantDependencySourceV1::Git {
                repository_url,
                immutable_revision: immutable_revision.to_ascii_lowercase(),
            },
            source_manifest_hash,
            enabled_features,
        )
    }

    pub fn repository_path(
        package_name: impl Into<String>,
        resolved_version: impl Into<String>,
        path: impl Into<String>,
        source_manifest_hash: [u8; 32],
        enabled_features: Vec<String>,
    ) -> Result<Self, FeatureContractError> {
        let path = path.into();
        validate_repository_path(&path)?;
        Self::build(
            package_name,
            resolved_version,
            RelevantDependencySourceV1::RepositoryPath { path },
            source_manifest_hash,
            enabled_features,
        )
    }

    fn build(
        package_name: impl Into<String>,
        resolved_version: impl Into<String>,
        source: RelevantDependencySourceV1,
        checksum_or_source_manifest_hash: [u8; 32],
        mut enabled_features: Vec<String>,
    ) -> Result<Self, FeatureContractError> {
        let package_name = package_name.into();
        let resolved_version = resolved_version.into();
        validate_field("dependency package name", &package_name)?;
        validate_field("dependency resolved version", &resolved_version)?;
        for feature in &enabled_features {
            validate_field("dependency feature", feature)?;
        }
        enabled_features.sort_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
        reject_duplicates(
            "dependency feature",
            enabled_features.iter().map(String::as_str),
        )?;
        Ok(Self {
            package_name,
            resolved_version,
            source,
            checksum_or_source_manifest_hash,
            enabled_features,
        })
    }

    fn sort_key(&self) -> (&str, u8, String) {
        (
            &self.package_name,
            self.source.tag(),
            self.source.canonical_identity(),
        )
    }

    pub fn package_name(&self) -> &str {
        &self.package_name
    }

    pub fn resolved_version(&self) -> &str {
        &self.resolved_version
    }

    pub const fn source_kind(&self) -> RelevantDependencySourceKindV1 {
        match self.source {
            RelevantDependencySourceV1::Registry { .. } => RelevantDependencySourceKindV1::Registry,
            RelevantDependencySourceV1::Git { .. } => RelevantDependencySourceKindV1::Git,
            RelevantDependencySourceV1::RepositoryPath { .. } => {
                RelevantDependencySourceKindV1::RepositoryPath
            }
        }
    }

    pub fn source_identity(&self) -> String {
        self.source.canonical_identity()
    }

    pub const fn checksum_or_source_manifest_hash(&self) -> &[u8; 32] {
        &self.checksum_or_source_manifest_hash
    }

    pub fn enabled_features(&self) -> &[String] {
        &self.enabled_features
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RelevantDependencySetIdentityV1([u8; 32]);

impl RelevantDependencySetIdentityV1 {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelevantDependencySetV1 {
    entries: Vec<RelevantDependencyV1>,
    canonical_bytes: Vec<u8>,
    identity: RelevantDependencySetIdentityV1,
}

impl RelevantDependencySetV1 {
    pub fn new(mut entries: Vec<RelevantDependencyV1>) -> Result<Self, FeatureContractError> {
        entries.sort_by(|left, right| left.sort_key().cmp(&right.sort_key()));
        let mut identities = HashSet::with_capacity(entries.len());
        for entry in &entries {
            let identity = (
                entry.package_name.clone(),
                entry.source.tag(),
                entry.source.canonical_identity(),
            );
            if !identities.insert(identity) {
                return Err(FeatureContractError::new(format!(
                    "duplicate relevant dependency source for `{}`",
                    entry.package_name
                )));
            }
        }
        let canonical_bytes = encode_relevant_dependencies(&entries)?;
        let identity = RelevantDependencySetIdentityV1(Sha256::digest(&canonical_bytes).into());
        Ok(Self {
            entries,
            canonical_bytes,
            identity,
        })
    }

    pub fn entries(&self) -> &[RelevantDependencyV1] {
        &self.entries
    }

    pub const fn identity(&self) -> RelevantDependencySetIdentityV1 {
        self.identity
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SemanticSourceSetIdentityV1([u8; 32]);

impl SemanticSourceSetIdentityV1 {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    pub fn to_hex(self) -> String {
        hex(&self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticSourceSetV1 {
    sources: SemanticSourceManifestV1,
    dependencies: RelevantDependencySetV1,
    canonical_bytes: Vec<u8>,
    identity: SemanticSourceSetIdentityV1,
}

impl SemanticSourceSetV1 {
    pub fn new(sources: SemanticSourceManifestV1, dependencies: RelevantDependencySetV1) -> Self {
        let mut canonical_bytes = Vec::with_capacity(SOURCE_SET_DOMAIN_V1.len() + 68);
        canonical_bytes.extend_from_slice(SOURCE_SET_DOMAIN_V1);
        canonical_bytes.extend_from_slice(&SOURCE_SET_VERSION_V1.to_be_bytes());
        canonical_bytes.extend_from_slice(sources.identity().as_bytes());
        canonical_bytes.extend_from_slice(dependencies.identity().as_bytes());
        let identity = SemanticSourceSetIdentityV1(Sha256::digest(&canonical_bytes).into());
        Self {
            sources,
            dependencies,
            canonical_bytes,
            identity,
        }
    }

    pub const fn identity(&self) -> SemanticSourceSetIdentityV1 {
        self.identity
    }

    pub fn sources(&self) -> &SemanticSourceManifestV1 {
        &self.sources
    }

    pub fn dependencies(&self) -> &RelevantDependencySetV1 {
        &self.dependencies
    }

    pub fn canonical_bytes(&self) -> &[u8] {
        &self.canonical_bytes
    }
}

fn canonical_payload(
    kind: SemanticSourceKindV1,
    payload: &[u8],
) -> Result<Vec<u8>, FeatureContractError> {
    match kind {
        SemanticSourceKindV1::Utf8Text => {
            let text = std::str::from_utf8(payload).map_err(|error| {
                FeatureContractError::new(format!("semantic text source is not UTF-8: {error}"))
            })?;
            let mut canonical = Vec::with_capacity(payload.len());
            let bytes = text.as_bytes();
            let mut index = 0usize;
            while index < bytes.len() {
                if bytes[index] == b'\r' {
                    canonical.push(b'\n');
                    index += 1;
                    if index < bytes.len() && bytes[index] == b'\n' {
                        index += 1;
                    }
                } else {
                    canonical.push(bytes[index]);
                    index += 1;
                }
            }
            Ok(canonical)
        }
        SemanticSourceKindV1::RawBinary => Ok(payload.to_vec()),
    }
}

fn encode_source_manifest(
    entries: &[SemanticSourceEntryV1],
) -> Result<Vec<u8>, FeatureContractError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(SOURCE_MANIFEST_DOMAIN_V1);
    bytes.extend_from_slice(&SOURCE_MANIFEST_VERSION_V1.to_be_bytes());
    push_count(&mut bytes, entries.len(), "semantic source entries")?;
    for entry in entries {
        push_text(&mut bytes, &entry.path)?;
        bytes.push(entry.payload_kind as u8);
        bytes.extend_from_slice(&entry.payload_hash);
        match &entry.generated {
            Some(generated) => {
                bytes.push(1);
                push_text(&mut bytes, &generated.generator_path)?;
                push_count(&mut bytes, generated.input_paths.len(), "generated inputs")?;
                for input in &generated.input_paths {
                    push_text(&mut bytes, input)?;
                }
            }
            None => bytes.push(0),
        }
    }
    Ok(bytes)
}

fn encode_relevant_dependencies(
    entries: &[RelevantDependencyV1],
) -> Result<Vec<u8>, FeatureContractError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(RELEVANT_DEPENDENCIES_DOMAIN_V1);
    bytes.extend_from_slice(&RELEVANT_DEPENDENCIES_VERSION_V1.to_be_bytes());
    push_count(&mut bytes, entries.len(), "relevant dependency entries")?;
    for entry in entries {
        push_tagged_text(&mut bytes, 1, &entry.package_name)?;
        bytes.push(2);
        bytes.push(entry.source.tag());
        match &entry.source {
            RelevantDependencySourceV1::Registry { registry_url } => {
                push_tagged_text(&mut bytes, 3, registry_url)?;
            }
            RelevantDependencySourceV1::Git {
                repository_url,
                immutable_revision,
            } => {
                push_tagged_text(&mut bytes, 4, repository_url)?;
                push_tagged_text(&mut bytes, 5, immutable_revision)?;
            }
            RelevantDependencySourceV1::RepositoryPath { path } => {
                push_tagged_text(&mut bytes, 6, path)?;
            }
        }
        push_tagged_text(&mut bytes, 7, &entry.resolved_version)?;
        bytes.push(8);
        bytes.extend_from_slice(&entry.checksum_or_source_manifest_hash);
        bytes.push(9);
        push_count(
            &mut bytes,
            entry.enabled_features.len(),
            "dependency features",
        )?;
        for feature in &entry.enabled_features {
            push_text(&mut bytes, feature)?;
        }
    }
    Ok(bytes)
}

fn validate_repository_path(path: &str) -> Result<(), FeatureContractError> {
    validate_field("repository-relative path", path)?;
    if path.starts_with('/')
        || path.starts_with('\\')
        || path.contains('\\')
        || path.contains(':')
        || path
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | ".."))
    {
        return Err(FeatureContractError::new(format!(
            "semantic source path `{path}` is not canonical repository-relative UTF-8"
        )));
    }
    if path.len() > MAX_PATH_BYTES {
        return Err(FeatureContractError::new(
            "semantic source path exceeds the byte limit",
        ));
    }
    Ok(())
}

fn validate_canonical_url(field: &str, url: &str) -> Result<(), FeatureContractError> {
    validate_field(field, url)?;
    if !(url.starts_with("https://") || url.starts_with("http://"))
        || url.chars().any(char::is_whitespace)
        || url.ends_with('/')
    {
        return Err(FeatureContractError::new(format!(
            "{field} must be a canonical absolute HTTP(S) URL without a trailing slash"
        )));
    }
    Ok(())
}

fn validate_field(field: &str, value: &str) -> Result<(), FeatureContractError> {
    if value.is_empty() {
        return Err(FeatureContractError::new(format!(
            "{field} must not be empty"
        )));
    }
    if value.len() > MAX_FIELD_BYTES {
        return Err(FeatureContractError::new(format!(
            "{field} exceeds {MAX_FIELD_BYTES} bytes"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(FeatureContractError::new(format!(
            "{field} contains a control character"
        )));
    }
    Ok(())
}

fn reject_duplicates<'a>(
    field: &str,
    values: impl IntoIterator<Item = &'a str>,
) -> Result<(), FeatureContractError> {
    let mut seen = HashSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(FeatureContractError::new(format!(
                "duplicate {field} `{value}`"
            )));
        }
    }
    Ok(())
}

fn push_count(bytes: &mut Vec<u8>, count: usize, field: &str) -> Result<(), FeatureContractError> {
    let count = u32::try_from(count)
        .map_err(|_| FeatureContractError::new(format!("{field} count exceeds u32")))?;
    bytes.extend_from_slice(&count.to_be_bytes());
    Ok(())
}

fn push_tagged_text(bytes: &mut Vec<u8>, tag: u8, value: &str) -> Result<(), FeatureContractError> {
    bytes.push(tag);
    push_text(bytes, value)
}

fn push_text(bytes: &mut Vec<u8>, value: &str) -> Result<(), FeatureContractError> {
    let length = u32::try_from(value.len())
        .map_err(|_| FeatureContractError::new("canonical source field exceeds u32 bytes"))?;
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
    Ok(())
}
