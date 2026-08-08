//! Safety regression: the control plane must refuse a non-loopback `--base-url`
//! unless `--allow-remote` is given. It forwards the bearer token to that host
//! and trusts its demo answers to unlock trading, so a remote target would
//! defeat both token secrecy and the "localhost only" posture.

use neoethos_mcp::enforce_loopback;

#[test]
fn loopback_hosts_are_accepted() {
    for url in [
        "http://127.0.0.1:7423",
        "http://localhost:7423",
        "http://[::1]:7423",
        "http://127.0.0.1:7423/",
    ] {
        enforce_loopback(url, false)
            .unwrap_or_else(|e| panic!("{url} is loopback and must be accepted: {e}"));
    }
}

#[test]
fn remote_hosts_are_refused_by_default() {
    for url in [
        "http://8.8.8.8:7423",
        "http://example.com:7423",
        "http://192.168.1.50:7423", // LAN is still not loopback
        "https://api.some-broker.example",
    ] {
        let err = enforce_loopback(url, false)
            .expect_err(&format!("{url} is non-loopback and must be refused"));
        let msg = err.to_string();
        assert!(
            msg.contains("non-loopback") && msg.contains("--allow-remote"),
            "refusal for {url} must name the reason and the override, got: {msg}"
        );
    }
}

#[test]
fn remote_hosts_are_allowed_only_with_the_explicit_flag() {
    enforce_loopback("http://8.8.8.8:7423", true)
        .expect("a non-loopback host is permitted once --allow-remote is set");
}

#[test]
fn a_garbage_url_is_refused() {
    assert!(enforce_loopback("not a url", false).is_err());
    assert!(enforce_loopback("", false).is_err());
}
