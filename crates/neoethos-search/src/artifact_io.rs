#[cfg(test)]
pub use neoethos_core::storage::json::temporary_path;
// `write_bytes_atomic` was re-exported here for the binary trial-returns
// payload, back when that file was written in one shot at the end of the run.
// It is gone because the writer is now STREAMING: it appends each chunk and
// seeks back to patch the header's trial count, so a killed process leaves a
// shorter file that still parses. A whole-file temp+rename cannot express that,
// and an unused re-export carrying a comment claiming a live consumer is the
// same silent-drift defect in miniature — it reads as wiring when it is none.
pub use neoethos_core::storage::json::{read_json, stable_json_hash, write_json_atomic};
pub use neoethos_core::utils::fnv1a64;


/// The single lower-hex encoder for Search identity and receipt digests.
/// Accepts both raw fixed-size SHA-256 arrays and borrowed digest buffers.
pub(crate) fn hex_lower(bytes: impl AsRef<[u8]>) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let bytes = bytes.as_ref();
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod hex_tests {
    use super::hex_lower;

    #[test]
    fn lower_hex_is_identical_for_sha256_arrays_and_slices() {
        assert_eq!(hex_lower([0_u8, 15, 16, 255]), "000f10ff");
        assert_eq!(hex_lower(&[0_u8, 15, 16, 255]), "000f10ff");
        assert_eq!(hex_lower([0_u8; 32]), "0".repeat(64));
        assert_eq!(hex_lower([255_u8; 32]), "f".repeat(64));
        assert_eq!(hex_lower([]), "");
    }
}
