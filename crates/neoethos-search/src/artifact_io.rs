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
