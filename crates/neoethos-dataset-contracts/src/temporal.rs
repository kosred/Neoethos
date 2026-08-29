use std::error::Error;
use std::fmt;
use std::str::FromStr;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalContractError {
    kind: &'static str,
    value: String,
}

impl TemporalContractError {
    fn new(kind: &'static str, value: impl Into<String>) -> Self {
        Self {
            kind,
            value: value.into(),
        }
    }
}

impl fmt::Display for TemporalContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid {}: {:?}", self.kind, self.value)
    }
}

impl Error for TemporalContractError {}

/// The exact `ProtoOATrendbarPeriod` set supported by cTrader Open API.
///
/// The discriminants are part of the canonical dataset-identity byte contract.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CanonicalTimeframe {
    M1 = 1,
    M2 = 2,
    M3 = 3,
    M4 = 4,
    M5 = 5,
    M10 = 6,
    M15 = 7,
    M30 = 8,
    H1 = 9,
    H4 = 10,
    H12 = 11,
    D1 = 12,
    W1 = 13,
    MN1 = 14,
}

/// Exact cTrader `ProtoOATrendbarPeriod` labels in protocol order.
///
/// Kept in this dependency-free leaf so config, broker, data, UI, mesh and GPU
/// code cannot grow competing private lists.
pub const CANONICAL_TIMEFRAMES: &[&str] = &[
    "M1", "M2", "M3", "M4", "M5", "M10", "M15", "M30", "H1", "H4", "H12", "D1", "W1", "MN1",
];

impl CanonicalTimeframe {
    pub const ALL: [Self; 14] = [
        Self::M1,
        Self::M2,
        Self::M3,
        Self::M4,
        Self::M5,
        Self::M10,
        Self::M15,
        Self::M30,
        Self::H1,
        Self::H4,
        Self::H12,
        Self::D1,
        Self::W1,
        Self::MN1,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::M1 => "M1",
            Self::M2 => "M2",
            Self::M3 => "M3",
            Self::M4 => "M4",
            Self::M5 => "M5",
            Self::M10 => "M10",
            Self::M15 => "M15",
            Self::M30 => "M30",
            Self::H1 => "H1",
            Self::H4 => "H4",
            Self::H12 => "H12",
            Self::D1 => "D1",
            Self::W1 => "W1",
            Self::MN1 => "MN1",
        }
    }

    pub const fn ctrader_protocol_code(self) -> i32 {
        self as u8 as i32
    }

    pub fn from_ctrader_protocol_code(code: i32) -> Result<Self, TemporalContractError> {
        match code {
            1 => Ok(Self::M1),
            2 => Ok(Self::M2),
            3 => Ok(Self::M3),
            4 => Ok(Self::M4),
            5 => Ok(Self::M5),
            6 => Ok(Self::M10),
            7 => Ok(Self::M15),
            8 => Ok(Self::M30),
            9 => Ok(Self::H1),
            10 => Ok(Self::H4),
            11 => Ok(Self::H12),
            12 => Ok(Self::D1),
            13 => Ok(Self::W1),
            14 => Ok(Self::MN1),
            _ => Err(TemporalContractError::new(
                "cTrader timeframe protocol code",
                code.to_string(),
            )),
        }
    }

    /// Fixed epoch-grid duration for minute/hour frames.
    ///
    /// Daily, weekly, and monthly frames are calendar contracts. Returning a
    /// fabricated fixed duration for them is deliberately impossible here.
    pub const fn fixed_duration_ms(self) -> Option<i64> {
        const MINUTE_MS: i64 = 60_000;
        match self {
            Self::M1 => Some(MINUTE_MS),
            Self::M2 => Some(2 * MINUTE_MS),
            Self::M3 => Some(3 * MINUTE_MS),
            Self::M4 => Some(4 * MINUTE_MS),
            Self::M5 => Some(5 * MINUTE_MS),
            Self::M10 => Some(10 * MINUTE_MS),
            Self::M15 => Some(15 * MINUTE_MS),
            Self::M30 => Some(30 * MINUTE_MS),
            Self::H1 => Some(60 * MINUTE_MS),
            Self::H4 => Some(240 * MINUTE_MS),
            Self::H12 => Some(720 * MINUTE_MS),
            Self::D1 | Self::W1 | Self::MN1 => None,
        }
    }

    pub const fn identity_tag(self) -> u8 {
        self as u8
    }
}

impl fmt::Display for CanonicalTimeframe {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for CanonicalTimeframe {
    type Err = TemporalContractError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|timeframe| timeframe.as_str() == value)
            .ok_or_else(|| TemporalContractError::new("canonical timeframe", value))
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BarTimestampConvention {
    BarOpen = 1,
    BarClose = 2,
    BarEnd = 3,
    Unknown = 4,
}

impl BarTimestampConvention {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BarOpen => "bar_open",
            Self::BarClose => "bar_close",
            Self::BarEnd => "bar_end",
            Self::Unknown => "unknown",
        }
    }

    pub const fn is_canonical_bar_open(self) -> bool {
        matches!(self, Self::BarOpen)
    }

    pub const fn identity_tag(self) -> u8 {
        self as u8
    }

    pub fn from_identity_tag(tag: u8) -> Result<Self, TemporalContractError> {
        match tag {
            1 => Ok(Self::BarOpen),
            2 => Ok(Self::BarClose),
            3 => Ok(Self::BarEnd),
            4 => Ok(Self::Unknown),
            _ => Err(TemporalContractError::new(
                "bar timestamp convention tag",
                tag.to_string(),
            )),
        }
    }
}

impl fmt::Display for BarTimestampConvention {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for BarTimestampConvention {
    type Err = TemporalContractError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "bar_open" => Ok(Self::BarOpen),
            "bar_close" => Ok(Self::BarClose),
            "bar_end" => Ok(Self::BarEnd),
            "unknown" => Ok(Self::Unknown),
            _ => Err(TemporalContractError::new(
                "bar timestamp convention",
                value,
            )),
        }
    }
}
