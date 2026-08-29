use std::error::Error;
use std::fmt;
use std::str;

use crate::{BarTimestampConvention, CanonicalTimeframe};

const IDENTITY_DOMAIN_V1: &[u8] = b"neoethos.canonical-dataset-identity.v1\0";
const IDENTITY_VERSION_V1: u16 = 1;
const PATH_PREFIX_V1: &str = "d1-";
const MAX_PATH_COMPONENT_BYTES: usize = 240;
const MAX_TEXT_BYTES: usize = 128;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DatasetIdentityError {
    detail: String,
}

impl DatasetIdentityError {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl fmt::Display for DatasetIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.detail)
    }
}

impl Error for DatasetIdentityError {}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CTraderEnvironment {
    Live = 1,
    Demo = 2,
}

impl CTraderEnvironment {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Demo => "demo",
        }
    }

    fn from_tag(tag: u8) -> Result<Self, DatasetIdentityError> {
        match tag {
            1 => Ok(Self::Live),
            2 => Ok(Self::Demo),
            _ => Err(DatasetIdentityError::new(format!(
                "unknown cTrader environment tag {tag}"
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanonicalDatasetScope {
    External {
        source_namespace: String,
    },
    CTrader {
        environment: CTraderEnvironment,
        server: String,
        account_id: i64,
        symbol_id: i64,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalDatasetIdentity {
    scope: CanonicalDatasetScope,
    symbol_name: String,
    timeframe: CanonicalTimeframe,
    bar_timestamp_convention: BarTimestampConvention,
}

impl CanonicalDatasetIdentity {
    pub fn external(
        source_namespace: impl Into<String>,
        symbol_name: impl Into<String>,
        timeframe: CanonicalTimeframe,
        bar_timestamp_convention: BarTimestampConvention,
    ) -> Result<Self, DatasetIdentityError> {
        let identity = Self {
            scope: CanonicalDatasetScope::External {
                source_namespace: source_namespace.into(),
            },
            symbol_name: symbol_name.into(),
            timeframe,
            bar_timestamp_convention,
        };
        identity.validate()?;
        Ok(identity)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ctrader(
        environment: CTraderEnvironment,
        server: impl Into<String>,
        account_id: i64,
        symbol_id: i64,
        symbol_name: impl Into<String>,
        timeframe: CanonicalTimeframe,
        bar_timestamp_convention: BarTimestampConvention,
    ) -> Result<Self, DatasetIdentityError> {
        let identity = Self {
            scope: CanonicalDatasetScope::CTrader {
                environment,
                server: server.into(),
                account_id,
                symbol_id,
            },
            symbol_name: symbol_name.into(),
            timeframe,
            bar_timestamp_convention,
        };
        identity.validate()?;
        Ok(identity)
    }

    pub const fn scope(&self) -> &CanonicalDatasetScope {
        &self.scope
    }

    pub fn symbol_name(&self) -> &str {
        &self.symbol_name
    }

    pub const fn timeframe(&self) -> CanonicalTimeframe {
        self.timeframe
    }

    pub const fn bar_timestamp_convention(&self) -> BarTimestampConvention {
        self.bar_timestamp_convention
    }

    pub const fn is_broker_real(&self) -> bool {
        matches!(self.scope, CanonicalDatasetScope::CTrader { .. })
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(128);
        bytes.extend_from_slice(IDENTITY_DOMAIN_V1);
        bytes.extend_from_slice(&IDENTITY_VERSION_V1.to_be_bytes());
        match &self.scope {
            CanonicalDatasetScope::External { source_namespace } => {
                bytes.push(1);
                push_text(&mut bytes, source_namespace);
            }
            CanonicalDatasetScope::CTrader {
                environment,
                server,
                account_id,
                symbol_id,
            } => {
                bytes.push(2);
                bytes.push(*environment as u8);
                push_text(&mut bytes, server);
                bytes.extend_from_slice(&account_id.to_be_bytes());
                bytes.extend_from_slice(&symbol_id.to_be_bytes());
            }
        }
        push_text(&mut bytes, &self.symbol_name);
        bytes.push(self.timeframe.identity_tag());
        bytes.push(self.bar_timestamp_convention.identity_tag());
        bytes
    }

    pub fn to_path_component(&self) -> String {
        let encoded = encode_base32hex_lower(&self.canonical_bytes());
        format!("{PATH_PREFIX_V1}{encoded}")
    }

    pub fn from_path_component(component: &str) -> Result<Self, DatasetIdentityError> {
        if component.len() > MAX_PATH_COMPONENT_BYTES {
            return Err(DatasetIdentityError::new(
                "dataset path component is too long",
            ));
        }
        let encoded = component
            .strip_prefix(PATH_PREFIX_V1)
            .ok_or_else(|| DatasetIdentityError::new("dataset path component has no d1 prefix"))?;
        if encoded.is_empty() {
            return Err(DatasetIdentityError::new(
                "dataset path component has no payload",
            ));
        }
        let bytes = decode_base32hex_lower(encoded)?;
        let identity = Self::from_canonical_bytes(&bytes)?;
        if identity.to_path_component() != component {
            return Err(DatasetIdentityError::new(
                "dataset path component is not canonically encoded",
            ));
        }
        Ok(identity)
    }

    /// Decode the exact versioned canonical byte representation.
    ///
    /// This is the only binary reconstruction path used by higher-level
    /// provenance contracts; it re-encodes and compares before accepting so
    /// non-canonical or trailing data never becomes a second identity form.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DatasetIdentityError> {
        let mut cursor = Cursor::new(bytes);
        cursor.require_exact(IDENTITY_DOMAIN_V1, "identity domain")?;
        let version = cursor.read_u16("identity version")?;
        if version != IDENTITY_VERSION_V1 {
            return Err(DatasetIdentityError::new(format!(
                "unsupported dataset identity version {version}"
            )));
        }
        let scope_tag = cursor.read_u8("scope tag")?;
        let scope = match scope_tag {
            1 => CanonicalDatasetScope::External {
                source_namespace: cursor.read_text("source namespace")?,
            },
            2 => CanonicalDatasetScope::CTrader {
                environment: CTraderEnvironment::from_tag(cursor.read_u8("environment tag")?)?,
                server: cursor.read_text("server")?,
                account_id: cursor.read_i64("account id")?,
                symbol_id: cursor.read_i64("symbol id")?,
            },
            _ => {
                return Err(DatasetIdentityError::new(format!(
                    "unknown dataset scope tag {scope_tag}"
                )));
            }
        };
        let symbol_name = cursor.read_text("symbol name")?;
        let timeframe = CanonicalTimeframe::from_ctrader_protocol_code(i32::from(
            cursor.read_u8("timeframe tag")?,
        ))
        .map_err(|error| DatasetIdentityError::new(error.to_string()))?;
        let bar_timestamp_convention = BarTimestampConvention::from_identity_tag(
            cursor.read_u8("bar timestamp convention tag")?,
        )
        .map_err(|error| DatasetIdentityError::new(error.to_string()))?;
        if !cursor.is_empty() {
            return Err(DatasetIdentityError::new(
                "dataset identity has trailing bytes",
            ));
        }
        let identity = Self {
            scope,
            symbol_name,
            timeframe,
            bar_timestamp_convention,
        };
        identity.validate()?;
        if identity.canonical_bytes() != bytes {
            return Err(DatasetIdentityError::new(
                "dataset identity bytes are not canonical",
            ));
        }
        Ok(identity)
    }

    fn validate(&self) -> Result<(), DatasetIdentityError> {
        validate_text("symbol name", &self.symbol_name)?;
        if !self.bar_timestamp_convention.is_canonical_bar_open() {
            return Err(DatasetIdentityError::new(format!(
                "canonical dataset identity requires bar_open, got {}",
                self.bar_timestamp_convention
            )));
        }
        match &self.scope {
            CanonicalDatasetScope::External { source_namespace } => {
                validate_text("external source namespace", source_namespace)?;
            }
            CanonicalDatasetScope::CTrader {
                server,
                account_id,
                symbol_id,
                ..
            } => {
                validate_text("cTrader server", server)?;
                if *account_id <= 0 {
                    return Err(DatasetIdentityError::new(
                        "cTrader account id must be positive",
                    ));
                }
                if *symbol_id <= 0 {
                    return Err(DatasetIdentityError::new(
                        "cTrader symbol id must be positive",
                    ));
                }
            }
        }
        let path_len = PATH_PREFIX_V1.len() + encoded_base32hex_len(self.canonical_bytes().len());
        if path_len > MAX_PATH_COMPONENT_BYTES {
            return Err(DatasetIdentityError::new(format!(
                "encoded dataset identity path is {path_len} bytes; maximum is {MAX_PATH_COMPONENT_BYTES}"
            )));
        }
        Ok(())
    }
}

fn validate_text(field: &str, value: &str) -> Result<(), DatasetIdentityError> {
    if value.trim().is_empty() {
        return Err(DatasetIdentityError::new(format!(
            "{field} must not be empty"
        )));
    }
    if value.len() > MAX_TEXT_BYTES {
        return Err(DatasetIdentityError::new(format!(
            "{field} is too long: {} bytes",
            value.len()
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(DatasetIdentityError::new(format!(
            "{field} contains a control character"
        )));
    }
    Ok(())
}

fn push_text(bytes: &mut Vec<u8>, value: &str) {
    let length = u32::try_from(value.len()).expect("validated identity text fits in u32");
    bytes.extend_from_slice(&length.to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn encoded_base32hex_len(byte_len: usize) -> usize {
    byte_len.saturating_mul(8).div_ceil(5)
}

fn encode_base32hex_lower(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 32] = b"0123456789abcdefghijklmnopqrstuv";
    let mut encoded = String::with_capacity(encoded_base32hex_len(bytes.len()));
    let mut accumulator = 0_u32;
    let mut bits = 0_u8;
    for &byte in bytes {
        accumulator = (accumulator << 8) | u32::from(byte);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            let index = ((accumulator >> bits) & 0x1f) as usize;
            encoded.push(ALPHABET[index] as char);
        }
    }
    if bits > 0 {
        let index = ((accumulator << (5 - bits)) & 0x1f) as usize;
        encoded.push(ALPHABET[index] as char);
    }
    encoded
}

fn decode_base32hex_lower(encoded: &str) -> Result<Vec<u8>, DatasetIdentityError> {
    let mut decoded = Vec::with_capacity(encoded.len().saturating_mul(5) / 8);
    let mut accumulator = 0_u32;
    let mut bits = 0_u8;
    for byte in encoded.bytes() {
        let value = match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'v' => 10 + byte - b'a',
            _ => {
                return Err(DatasetIdentityError::new(
                    "dataset path contains a non-base32hex-lower character",
                ));
            }
        };
        accumulator = (accumulator << 5) | u32::from(value);
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            decoded.push(((accumulator >> bits) & 0xff) as u8);
        }
    }
    if bits > 0 && (accumulator & ((1_u32 << bits) - 1)) != 0 {
        return Err(DatasetIdentityError::new(
            "dataset path has non-zero trailing base32 bits",
        ));
    }
    if encode_base32hex_lower(&decoded) != encoded {
        return Err(DatasetIdentityError::new(
            "dataset path is not canonical base32hex-lower",
        ));
    }
    Ok(decoded)
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

    fn take(&mut self, length: usize, field: &str) -> Result<&'a [u8], DatasetIdentityError> {
        if self.remaining.len() < length {
            return Err(DatasetIdentityError::new(format!(
                "dataset identity is truncated at {field}"
            )));
        }
        let (value, rest) = self.remaining.split_at(length);
        self.remaining = rest;
        Ok(value)
    }

    fn require_exact(&mut self, expected: &[u8], field: &str) -> Result<(), DatasetIdentityError> {
        if self.take(expected.len(), field)? != expected {
            return Err(DatasetIdentityError::new(format!(
                "dataset identity has invalid {field}"
            )));
        }
        Ok(())
    }

    fn read_u8(&mut self, field: &str) -> Result<u8, DatasetIdentityError> {
        Ok(self.take(1, field)?[0])
    }

    fn read_u16(&mut self, field: &str) -> Result<u16, DatasetIdentityError> {
        let bytes: [u8; 2] = self
            .take(2, field)?
            .try_into()
            .map_err(|_| DatasetIdentityError::new(format!("invalid {field}")))?;
        Ok(u16::from_be_bytes(bytes))
    }

    fn read_i64(&mut self, field: &str) -> Result<i64, DatasetIdentityError> {
        let bytes: [u8; 8] = self
            .take(8, field)?
            .try_into()
            .map_err(|_| DatasetIdentityError::new(format!("invalid {field}")))?;
        Ok(i64::from_be_bytes(bytes))
    }

    fn read_text(&mut self, field: &str) -> Result<String, DatasetIdentityError> {
        let length_bytes: [u8; 4] = self
            .take(4, field)?
            .try_into()
            .map_err(|_| DatasetIdentityError::new(format!("invalid {field} length")))?;
        let length = usize::try_from(u32::from_be_bytes(length_bytes))
            .map_err(|_| DatasetIdentityError::new(format!("invalid {field} length")))?;
        if length > MAX_TEXT_BYTES {
            return Err(DatasetIdentityError::new(format!(
                "{field} length exceeds the contract maximum"
            )));
        }
        let bytes = self.take(length, field)?;
        let value = str::from_utf8(bytes)
            .map_err(|_| DatasetIdentityError::new(format!("{field} is not UTF-8")))?;
        validate_text(field, value)?;
        Ok(value.to_owned())
    }
}
