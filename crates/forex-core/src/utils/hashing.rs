//! Shared FNV-1a hashing primitives.
//!
//! Phase 63 extraction: previously the FNV-1a constants and helpers were
//! duplicated across `forex-core::contracts::temporal`,
//! `forex-search::artifact_io`, and
//! `forex-search::genetic::evolution_math`. They now live here so every
//! caller produces the same byte-for-byte hash.

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001B3;

/// Compute the FNV-1a 64-bit hash of `bytes` starting from the canonical
/// offset basis.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_update(FNV_OFFSET, bytes)
}

/// Continue an FNV-1a 64-bit hash from `seed`. Use this when a rolling
/// signature is built incrementally (e.g. the seen-signature ledger in
/// the genetic search).
pub fn fnv1a64_update(seed: u64, bytes: &[u8]) -> u64 {
    let mut hash = seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_offset_basis() {
        assert_eq!(fnv1a64(&[]), FNV_OFFSET);
    }

    #[test]
    fn fnv1a64_matches_known_vector() {
        // FNV-1a of "foobar" — canonical reference vector from
        // http://www.isthe.com/chongo/tech/comp/fnv/
        assert_eq!(fnv1a64(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn update_continues_from_seed() {
        let prefix = fnv1a64(b"foo");
        let combined = fnv1a64_update(prefix, b"bar");
        assert_eq!(combined, fnv1a64(b"foobar"));
    }
}
