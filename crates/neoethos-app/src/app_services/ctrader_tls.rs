//! Shared TLS provider setup for cTrader transports.

use std::sync::Once;

/// rustls 0.23 cannot infer a process default when both `ring` and
/// `aws-lc-rs` providers are enabled through the workspace dependency
/// graph. cTrader transports use tungstenite/rustls builders, so select a
/// provider before the first TLS client config is built.
pub fn ensure_ctrader_rustls_provider() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_ctrader_rustls_provider_is_idempotent() {
        ensure_ctrader_rustls_provider();
        ensure_ctrader_rustls_provider();

        assert!(rustls::crypto::CryptoProvider::get_default().is_some());
    }
}
