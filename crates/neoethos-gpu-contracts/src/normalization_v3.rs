//! Backend-neutral Search normalization policy codes. Data resolves names using
//! its canonical CPU classifier; this module deliberately contains no name list.

use sha2::{Digest, Sha256};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchNormalizationColumnModeV3 {
    Robust = 0,
    Binary = 1,
    SignedState = 2,
    SignedContinuous = 3,
}

impl SearchNormalizationColumnModeV3 {
    pub const fn wire_code(self) -> u8 {
        self as u8
    }
}

/// Verify the native six-u64 metadata transport. This digest intentionally is
/// not Data's name-aware fitted-state identity and grants no source authority.
pub fn resident_normalization_fit_metadata_sha256_v3(words: &[u64]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"neoethos.resident-robust-normalization.fit-metadata.semantic-v2\0");
    for word in words {
        hash.update(word.to_be_bytes());
    }
    hash.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_normalization_policy_codes_are_exact_single_bytes() {
        assert_eq!(std::mem::size_of::<SearchNormalizationColumnModeV3>(), 1);
        assert_eq!(
            [
                SearchNormalizationColumnModeV3::Robust,
                SearchNormalizationColumnModeV3::Binary,
                SearchNormalizationColumnModeV3::SignedState,
                SearchNormalizationColumnModeV3::SignedContinuous
            ]
            .map(|m| m.wire_code()),
            [0, 1, 2, 3]
        );
    }
}
