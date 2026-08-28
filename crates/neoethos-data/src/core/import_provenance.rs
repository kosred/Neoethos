//! Typed, canonical import provenance carried inside the generic dataset envelope.

use crate::core::dataset_manifest::ProducerProvenanceEnvelopeV1;
use anyhow::{Context, Result, bail};
use neoethos_dataset_contracts::CanonicalDatasetIdentity;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

const DOMAIN: &[u8] = b"neoethos.import-provenance.v1\0";
const VERSION: u16 = 1;
const MAX_TEXT_BYTES: usize = 512;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ImportSourceFormat {
    Csv = 1,
    Tsv = 2,
    JsonArray = 3,
    JsonLines = 4,
    Parquet = 5,
    ArrowIpcFile = 6,
    ArrowIpcStream = 7,
    Vortex = 8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportSourceFormatError {
    value: String,
}

impl fmt::Display for ImportSourceFormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid explicit import source format {:?}; expected one of csv, tsv, json-array, json-lines, parquet, arrow-ipc-file, arrow-ipc-stream, vortex",
            self.value
        )
    }
}

impl std::error::Error for ImportSourceFormatError {}

impl ImportSourceFormat {
    pub const ALL: [Self; 8] = [
        Self::Csv,
        Self::Tsv,
        Self::JsonArray,
        Self::JsonLines,
        Self::Parquet,
        Self::ArrowIpcFile,
        Self::ArrowIpcStream,
        Self::Vortex,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Csv => "csv",
            Self::Tsv => "tsv",
            Self::JsonArray => "json-array",
            Self::JsonLines => "json-lines",
            Self::Parquet => "parquet",
            Self::ArrowIpcFile => "arrow-ipc-file",
            Self::ArrowIpcStream => "arrow-ipc-stream",
            Self::Vortex => "vortex",
        }
    }

    /// Classify an explicit import candidate from its filename extension.
    /// Runtime discovery never calls this function. Ambiguous `.ipc` paths
    /// deliberately select the file route; callers importing an IPC stream
    /// must declare `arrow-ipc-stream` or use the unambiguous `.arrows` suffix.
    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension
            .trim_start_matches('.')
            .to_ascii_lowercase()
            .as_str()
        {
            "csv" => Some(Self::Csv),
            "tsv" | "tab" => Some(Self::Tsv),
            "json" => Some(Self::JsonArray),
            "jsonl" | "ndjson" => Some(Self::JsonLines),
            "parquet" | "pq" => Some(Self::Parquet),
            "arrow" | "feather" | "ipc" => Some(Self::ArrowIpcFile),
            "arrows" | "ipcstream" => Some(Self::ArrowIpcStream),
            "vortex" | "vtx" => Some(Self::Vortex),
            _ => None,
        }
    }

    fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            1 => Ok(Self::Csv),
            2 => Ok(Self::Tsv),
            3 => Ok(Self::JsonArray),
            4 => Ok(Self::JsonLines),
            5 => Ok(Self::Parquet),
            6 => Ok(Self::ArrowIpcFile),
            7 => Ok(Self::ArrowIpcStream),
            8 => Ok(Self::Vortex),
            _ => bail!("unknown import source format tag {tag}"),
        }
    }
}

impl fmt::Display for ImportSourceFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ImportSourceFormat {
    type Err = ImportSourceFormatError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|format| format.as_str() == value)
            .ok_or_else(|| ImportSourceFormatError {
                value: value.to_owned(),
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VolumeMappingV1 {
    Absent,
    SourceFloat64,
    ExactUnsignedInteger {
        bit_width: u8,
        unit: String,
    },
    ExactSignedInteger {
        bit_width: u8,
        unit: String,
    },
    ExactDecimal {
        precision: u8,
        scale: i8,
        unit: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportProvenanceV1 {
    selected_format: ImportSourceFormat,
    detected_format: ImportSourceFormat,
    source_sha256: [u8; 32],
    source_size: u64,
    stable_source_identity: String,
    dataset_identity: CanonicalDatasetIdentity,
    imported_unix_ms: u64,
    volume_mapping: VolumeMappingV1,
}

impl ImportProvenanceV1 {
    pub const SCHEMA_ID: &'static str = "neoethos.import-provenance.v1";

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        selected_format: ImportSourceFormat,
        detected_format: ImportSourceFormat,
        source_sha256: [u8; 32],
        source_size: u64,
        stable_source_identity: impl Into<String>,
        dataset_identity: CanonicalDatasetIdentity,
        imported_unix_ms: u64,
        volume_mapping: VolumeMappingV1,
    ) -> Result<Self> {
        let provenance = Self {
            selected_format,
            detected_format,
            source_sha256,
            source_size,
            stable_source_identity: stable_source_identity.into(),
            dataset_identity,
            imported_unix_ms,
            volume_mapping,
        };
        provenance.validate()?;
        Ok(provenance)
    }

    pub const fn selected_format(&self) -> ImportSourceFormat {
        self.selected_format
    }
    pub const fn detected_format(&self) -> ImportSourceFormat {
        self.detected_format
    }
    pub const fn source_sha256(&self) -> &[u8; 32] {
        &self.source_sha256
    }
    pub const fn source_size(&self) -> u64 {
        self.source_size
    }
    pub fn stable_source_identity(&self) -> &str {
        &self.stable_source_identity
    }
    pub const fn dataset_identity(&self) -> &CanonicalDatasetIdentity {
        &self.dataset_identity
    }
    pub const fn imported_unix_ms(&self) -> u64 {
        self.imported_unix_ms
    }
    pub const fn volume_mapping(&self) -> &VolumeMappingV1 {
        &self.volume_mapping
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(256);
        bytes.extend_from_slice(DOMAIN);
        bytes.extend_from_slice(&VERSION.to_be_bytes());
        bytes.push(self.selected_format as u8);
        bytes.push(self.detected_format as u8);
        bytes.extend_from_slice(&self.source_sha256);
        bytes.extend_from_slice(&self.source_size.to_be_bytes());
        push_text(&mut bytes, &self.stable_source_identity);
        push_text(&mut bytes, &self.dataset_identity.to_path_component());
        bytes.extend_from_slice(&self.imported_unix_ms.to_be_bytes());
        encode_volume_mapping(&mut bytes, &self.volume_mapping);
        bytes
    }

    pub fn to_envelope(&self) -> Result<ProducerProvenanceEnvelopeV1> {
        self.validate()?;
        ProducerProvenanceEnvelopeV1::new(Self::SCHEMA_ID, self.canonical_bytes())
    }

    pub fn from_envelope(envelope: &ProducerProvenanceEnvelopeV1) -> Result<Self> {
        envelope.validate()?;
        if envelope.schema_id() != Self::SCHEMA_ID {
            bail!(
                "import provenance schema mismatch: expected {}, got {}",
                Self::SCHEMA_ID,
                envelope.schema_id()
            );
        }
        Self::from_canonical_bytes(envelope.canonical_payload())
    }

    fn validate(&self) -> Result<()> {
        if self.selected_format != self.detected_format {
            bail!(
                "declared import format {} disagrees with detected format {}",
                self.selected_format.as_str(),
                self.detected_format.as_str()
            );
        }
        validate_text("stable source identity", &self.stable_source_identity)?;
        validate_text(
            "dataset identity",
            &self.dataset_identity.to_path_component(),
        )?;
        validate_volume_mapping(&self.volume_mapping)?;
        Ok(())
    }

    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(bytes);
        cursor.require_exact(DOMAIN, "domain")?;
        let version = cursor.read_u16("version")?;
        if version != VERSION {
            bail!("unsupported import provenance version {version}");
        }
        let selected_format = ImportSourceFormat::from_tag(cursor.read_u8("selected format")?)?;
        let detected_format = ImportSourceFormat::from_tag(cursor.read_u8("detected format")?)?;
        let source_sha256 = cursor.read_exact_array::<32>("source sha256")?;
        let source_size = cursor.read_u64("source size")?;
        let stable_source_identity = cursor.read_text("stable source identity")?;
        let identity_path = cursor.read_text("dataset identity")?;
        let dataset_identity = CanonicalDatasetIdentity::from_path_component(&identity_path)
            .context("decode dataset identity from import provenance")?;
        let imported_unix_ms = cursor.read_u64("import timestamp")?;
        let volume_mapping = decode_volume_mapping(&mut cursor)?;
        if !cursor.is_empty() {
            bail!("import provenance has trailing bytes");
        }
        let provenance = Self {
            selected_format,
            detected_format,
            source_sha256,
            source_size,
            stable_source_identity,
            dataset_identity,
            imported_unix_ms,
            volume_mapping,
        };
        provenance.validate()?;
        if provenance.canonical_bytes() != bytes {
            bail!("import provenance bytes are not canonical");
        }
        Ok(provenance)
    }
}

fn validate_text(field: &str, value: &str) -> Result<()> {
    if value.is_empty() || value.len() > MAX_TEXT_BYTES || value.chars().any(char::is_control) {
        bail!("invalid {field}");
    }
    Ok(())
}

fn validate_volume_mapping(mapping: &VolumeMappingV1) -> Result<()> {
    match mapping {
        VolumeMappingV1::Absent | VolumeMappingV1::SourceFloat64 => Ok(()),
        VolumeMappingV1::ExactUnsignedInteger { bit_width, unit }
        | VolumeMappingV1::ExactSignedInteger { bit_width, unit } => {
            if !matches!(*bit_width, 8 | 16 | 32 | 64) {
                bail!("invalid volume integer bit width {bit_width}");
            }
            validate_text("volume unit", unit)
        }
        VolumeMappingV1::ExactDecimal {
            precision,
            scale,
            unit,
        } => {
            if *precision == 0 || *precision > 38 {
                bail!("invalid volume decimal precision {precision}");
            }
            if scale.unsigned_abs() > *precision {
                bail!("invalid volume decimal scale {scale}");
            }
            validate_text("volume unit", unit)
        }
    }
}

fn push_text(bytes: &mut Vec<u8>, value: &str) {
    let length = u32::try_from(value.len()).expect("validated provenance text fits u32");
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn encode_volume_mapping(bytes: &mut Vec<u8>, mapping: &VolumeMappingV1) {
    match mapping {
        VolumeMappingV1::Absent => bytes.push(0),
        VolumeMappingV1::SourceFloat64 => bytes.push(1),
        VolumeMappingV1::ExactUnsignedInteger { bit_width, unit } => {
            bytes.push(2);
            bytes.push(*bit_width);
            push_text(bytes, unit);
        }
        VolumeMappingV1::ExactSignedInteger { bit_width, unit } => {
            bytes.push(3);
            bytes.push(*bit_width);
            push_text(bytes, unit);
        }
        VolumeMappingV1::ExactDecimal {
            precision,
            scale,
            unit,
        } => {
            bytes.push(4);
            bytes.push(*precision);
            bytes.push(scale.to_be_bytes()[0]);
            push_text(bytes, unit);
        }
    }
}

fn decode_volume_mapping(cursor: &mut Cursor<'_>) -> Result<VolumeMappingV1> {
    match cursor.read_u8("volume mapping")? {
        0 => Ok(VolumeMappingV1::Absent),
        1 => Ok(VolumeMappingV1::SourceFloat64),
        2 => Ok(VolumeMappingV1::ExactUnsignedInteger {
            bit_width: cursor.read_u8("volume bit width")?,
            unit: cursor.read_text("volume unit")?,
        }),
        3 => Ok(VolumeMappingV1::ExactSignedInteger {
            bit_width: cursor.read_u8("volume bit width")?,
            unit: cursor.read_text("volume unit")?,
        }),
        4 => Ok(VolumeMappingV1::ExactDecimal {
            precision: cursor.read_u8("volume precision")?,
            scale: i8::from_be_bytes([cursor.read_u8("volume scale")?]),
            unit: cursor.read_text("volume unit")?,
        }),
        tag => bail!("unknown volume mapping tag {tag}"),
    }
}

struct Cursor<'a> {
    remaining: &'a [u8],
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }
    fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }
    fn take(&mut self, length: usize, field: &str) -> Result<&'a [u8]> {
        if self.remaining.len() < length {
            bail!("import provenance is truncated at {field}");
        }
        let (value, rest) = self.remaining.split_at(length);
        self.remaining = rest;
        Ok(value)
    }
    fn require_exact(&mut self, expected: &[u8], field: &str) -> Result<()> {
        if self.take(expected.len(), field)? != expected {
            bail!("invalid import provenance {field}");
        }
        Ok(())
    }
    fn read_u8(&mut self, field: &str) -> Result<u8> {
        Ok(self.take(1, field)?[0])
    }
    fn read_u16(&mut self, field: &str) -> Result<u16> {
        Ok(u16::from_be_bytes(self.read_exact_array(field)?))
    }
    fn read_u64(&mut self, field: &str) -> Result<u64> {
        Ok(u64::from_be_bytes(self.read_exact_array(field)?))
    }
    fn read_exact_array<const N: usize>(&mut self, field: &str) -> Result<[u8; N]> {
        self.take(N, field)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid {field}"))
    }
    fn read_text(&mut self, field: &str) -> Result<String> {
        let length = usize::try_from(u32::from_be_bytes(
            self.read_exact_array(&format!("{field} length"))?,
        ))
        .context("provenance text length does not fit usize")?;
        if length > MAX_TEXT_BYTES {
            bail!("{field} exceeds maximum length");
        }
        let value = std::str::from_utf8(self.take(length, field)?)
            .with_context(|| format!("{field} is not UTF-8"))?;
        validate_text(field, value)?;
        Ok(value.to_owned())
    }
}
