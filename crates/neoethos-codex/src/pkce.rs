//! PKCE (RFC 7636) primitives.
//!
//! The flow is straightforward and the RFC is short, so this file
//! mirrors it directly:
//!
//! 1. Generate a high-entropy `code_verifier`. The RFC requires 43–128
//!    URL-safe characters (`[A-Z][a-z][0-9]-._~`). We use 64 random
//!    bytes → base64url-no-padding, which yields ~86 chars.
//! 2. Derive `code_challenge = BASE64URL(SHA256(code_verifier))`.
//! 3. Send `code_challenge` + `code_challenge_method=S256` to the
//!    authorize endpoint, keep `code_verifier` secret in process
//!    memory until the redirect lands.
//! 4. Send `code_verifier` to the token endpoint alongside the
//!    authorization code. The issuer recomputes the challenge,
//!    compares to what we sent, and only mints tokens if they match.
//!
//! No `code_verifier` ever touches the disk. We hold it in a
//! [`PkceChallenge`] that lives for the duration of one OAuth attempt
//! and gets dropped (and zeroised on Drop ideally — but `String`
//! doesn't zeroise, and pulling `zeroize` in for ~80 bytes of memory
//! isn't worth the dep churn) once the flow finishes.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand::RngCore;
use sha2::{Digest, Sha256};

/// One PKCE attempt — paired `code_verifier` + `code_challenge`.
///
/// Construct with [`PkceChallenge::generate`]; the only consumers are
/// [`crate::oauth::AuthorizationRequest::build_authorize_url`] (reads
/// `code_challenge`) and [`crate::oauth::exchange_code`] (reads
/// `code_verifier`).
#[derive(Debug, Clone)]
pub struct PkceChallenge {
    pub code_verifier: String,
    pub code_challenge: String,
}

impl PkceChallenge {
    /// Produce a fresh verifier + challenge pair.
    ///
    /// Uses `rand::rngs::OsRng` (via `rng()`) so the verifier is
    /// cryptographically strong. The 64 raw bytes turn into 86 chars
    /// after base64url-no-pad encoding, comfortably inside the
    /// RFC 7636 limit (43..=128).
    pub fn generate() -> Self {
        let mut bytes = [0u8; 64];
        rand::rng().fill_bytes(&mut bytes);
        let code_verifier = URL_SAFE_NO_PAD.encode(bytes);

        let digest = Sha256::digest(code_verifier.as_bytes());
        let code_challenge = URL_SAFE_NO_PAD.encode(digest);

        Self {
            code_verifier,
            code_challenge,
        }
    }

    /// PKCE method identifier we send in the authorize request.
    /// RFC 7636 also allows `plain` but we never want that — `S256`
    /// is mandatory for any client that can compute SHA-256, which
    /// we definitely can.
    pub fn method(&self) -> &'static str {
        "S256"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_verifier_has_legal_length() {
        let p = PkceChallenge::generate();
        // 64 bytes → ceil(64 * 4 / 3) = 86 chars (no padding).
        assert_eq!(p.code_verifier.len(), 86);
        // RFC 7636 § 4.1: 43..=128 chars.
        assert!((43..=128).contains(&p.code_verifier.len()));
    }

    #[test]
    fn challenge_is_sha256_of_verifier() {
        let p = PkceChallenge::generate();
        let recomputed = URL_SAFE_NO_PAD.encode(Sha256::digest(p.code_verifier.as_bytes()));
        assert_eq!(p.code_challenge, recomputed);
    }

    #[test]
    fn each_call_produces_a_fresh_pair() {
        let a = PkceChallenge::generate();
        let b = PkceChallenge::generate();
        assert_ne!(a.code_verifier, b.code_verifier);
        assert_ne!(a.code_challenge, b.code_challenge);
    }
}
