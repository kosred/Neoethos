//! Compile-time embedded cTrader Open API credentials.
//!
//! The three constants in this module are baked into the binary at build time
//! by `build.rs`. They serve as the **final fallback** in the credential
//! resolution chain, so that a distributed binary (`.exe` sent to a friend)
//! works out-of-the-box: the recipient only needs to press "Connect" and
//! authenticate with their own cTID in the browser — no manual configuration.
//!
//! # Resolution order (highest priority first)
//!
//! 1. `$FOREX_AI_BROKER_CREDENTIALS_PATH` (runtime env override)
//! 2. `%APPDATA%\forex-ai\broker_credentials.toml` (per-user persistent file)
//! 3. `<cwd>\.local\forex-ai\broker_credentials.toml` (dev fallback)
//! 4. **These constants** (compile-time embedded — this module)
//!
//! If a user-level file exists with non-empty credentials, it wins and these
//! constants are never used at runtime.
//!
//! # Security note
//!
//! The Client ID and Secret are cTrader Open API *application* credentials,
//! not user credentials. They identify the app on the cTrader authorize page
//! and carry no access to funds or account data on their own. They can be
//! rotated at any time via the cTrader Open API portal if needed.
include!(concat!(env!("OUT_DIR"), "/embedded_credentials.rs"));
