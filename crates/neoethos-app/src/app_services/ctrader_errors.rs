//! cTrader error code → user-friendly message + actionable CTA.
//!
//! The cTrader Open API answers failures with structured `errorCode`
//! strings (e.g. `CH_ACCESS_TOKEN_INVALID`, `MARKET_CLOSED`,
//! `RET_ACCOUNT_DISABLED`). They're machine-readable, but no end user
//! should ever see "CH_ACCESS_TOKEN_INVALID" in a banner — they have
//! no idea what to do with it. This module is the one place we
//! translate those codes into plain English (+ optional next-action
//! hints for the UI) so every route handler can `?` an anyhow error,
//! map it through `translate_anyhow`, and emit a consistent JSON
//! shape the Flutter side knows how to render.
//!
//! Wire shape: `TranslatedError { code, message, action_label,
//! action_target, severity }`. The frontend looks at `severity` for
//! banner colour, `message` for the body, and renders a button when
//! `action_label`/`action_target` are populated.
//!
//! Adding a new code: extend the `match` in `translate_code`. Keep
//! the English copy under ~120 characters so it fits inside a
//! single-line snackbar/banner without scrolling.

use serde::Serialize;

/// What the UI banner shows. All strings are pre-translated to
/// English here; Greek (or any other locale) translation can layer
/// on top in the frontend by switching on `code`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranslatedError {
    /// Original upstream code, e.g. `CH_ACCESS_TOKEN_INVALID`. The
    /// frontend may match on this for locale-specific copy or
    /// behaviour, but never displays it raw.
    pub code: String,
    /// Friendly English message. Trader-facing, no jargon.
    pub message: String,
    /// One-line CTA label, e.g. `"Re-authenticate"`. None when the
    /// error has no obvious user action.
    pub action_label: Option<String>,
    /// Where the CTA should send the user. Conventional values:
    ///   * `"broker_setup"`  — open the Broker Setup screen
    ///   * `"settings"`      — open Settings (credentials form)
    ///   * `"reauth"`        — fire POST /broker/reauth directly
    ///   * `"data_bootstrap"`— open Data Bootstrap screen
    ///   * `"none"`          — informational only, no nav
    pub action_target: Option<String>,
    /// "info" / "warning" / "error" / "critical". Drives banner
    /// colour on the frontend. `critical` = red + cannot dismiss
    /// until acted upon, `error` = red + dismissable, `warning` =
    /// amber, `info` = grey.
    pub severity: &'static str,
}

/// Map a raw cTrader error code to its translation. Used by the
/// HTTP handlers when they get a 502 from the broker — they call
/// `translate_anyhow(err)` which extracts the code and runs it
/// through this lookup.
pub fn translate_code(code: &str) -> TranslatedError {
    // Codes are quoted from real cTrader Open API responses observed
    // in production. Sources:
    //   - Spotware docs (https://help.ctrader.com/open-api/error-codes/)
    //   - Live error captures from this session's logs.
    let upper = code.trim().to_ascii_uppercase();
    let (message, action_label, action_target, severity) = match upper.as_str() {
        // ── Authentication / authorization ─────────────────────────
        "CH_ACCESS_TOKEN_INVALID" | "CH_INVALID_ACCESS_TOKEN" | "INVALID_REQUEST" => (
            "Your cTrader session has expired or the account ID doesn't match \
             the one the OAuth token was granted for. Re-authenticate to refresh.",
            Some("Re-authenticate"),
            Some("broker_setup"),
            "warning",
        ),
        "CH_ACCESS_TOKEN_EXPIRED" => (
            "OAuth token expired. Re-authenticate to get a fresh one.",
            Some("Re-authenticate"),
            Some("broker_setup"),
            "warning",
        ),
        "CH_ACCOUNT_NOT_AUTHORIZED" | "RET_ACCOUNT_NOT_AUTHORIZED" => (
            "This account wasn't included in the OAuth consent. Open Broker \
             Setup and re-authenticate, ticking the account you want to use.",
            Some("Re-authenticate"),
            Some("broker_setup"),
            "warning",
        ),
        "RET_ACCOUNT_DISABLED" | "ACCOUNT_DISABLED" => (
            "The broker has disabled this account. Contact your broker to \
             reactivate it.",
            None,
            None,
            "error",
        ),
        "CH_CLIENT_AUTH_FAILURE" => (
            "Client ID or Client Secret is wrong. Open Settings, paste the \
             values from the Spotware Open API portal, and Save.",
            Some("Open Settings"),
            Some("settings"),
            "error",
        ),

        // ── Order placement & execution ────────────────────────────
        "MARKET_CLOSED" | "TRADING_BAD_VOLUME" | "MARKET_NOT_OPEN" => (
            "Markets are closed right now (weekend, holiday, or session gap). \
             Your order will be rejected until the session re-opens.",
            None,
            None,
            "info",
        ),
        "TRADING_DISABLED" => (
            "Trading is disabled on this account. Check with your broker if \
             this is a prop-firm rules pause, a margin call, or a permanent \
             restriction.",
            None,
            None,
            "error",
        ),
        "SYMBOL_NOT_FOUND" | "SYMBOL_HAS_HOLIDAY" => (
            "That symbol isn't in the broker's catalog for this account. \
             Pick another from the Markets list.",
            None,
            None,
            "warning",
        ),
        "INSUFFICIENT_FUNDS" | "RET_TOO_LOW_MARGIN" => (
            "Not enough free margin for this trade. Reduce the volume, widen \
             the stop-loss, or deposit funds.",
            None,
            None,
            "warning",
        ),
        "INVALID_VOLUME" | "TRADING_DISABLED_BY_INVESTOR_PASSWORD" => (
            "Volume is outside the broker's allowed range for this symbol. \
             Check the Symbol Info panel for min/max lot sizes.",
            None,
            None,
            "warning",
        ),
        "INVALID_PRICE" | "PRICE_OFF" => (
            "Limit/stop price is too far from market or violates a minimum \
             stops-level. Move it closer and re-submit.",
            None,
            None,
            "warning",
        ),
        "ORDER_NOT_FOUND" => (
            "That pending order has already been cancelled or filled.",
            None,
            None,
            "info",
        ),
        "POSITION_NOT_FOUND" => (
            "That position has already been closed.",
            None,
            None,
            "info",
        ),

        // ── Risk / prop-firm gate ──────────────────────────────────
        "RISK_EXCEEDED" | "RET_LIMITS_EXCEEDED" => (
            "Risk caps would be breached by this trade. Check Risk Settings \
             (daily DD, total DD, per-trade risk%, max lot).",
            Some("Open Risk Settings"),
            Some("risk"),
            "warning",
        ),

        // ── Data / catalog ─────────────────────────────────────────
        "NO_HISTORICAL_DATA" => (
            "The broker has no historical bars for that symbol/timeframe \
             window. Try a smaller range or a different timeframe.",
            Some("Open Data Bootstrap"),
            Some("data_bootstrap"),
            "info",
        ),

        // ── Network / transport ────────────────────────────────────
        "TIMED_OUT" | "TIMEOUT" => (
            "The broker took too long to respond. Likely a transient network \
             blip — try again in a moment.",
            None,
            None,
            "warning",
        ),
        "CH_RATE_LIMIT_EXCEEDED" | "RATE_LIMIT_EXCEEDED" => (
            "You're sending requests too fast for the broker's rate limit. \
             Wait 10–30 seconds and retry.",
            None,
            None,
            "warning",
        ),

        // ── Catch-all ──────────────────────────────────────────────
        _ => (
            // Surface the raw code so we can grow this table over
            // time. The frontend banner shows the friendly intro +
            // the raw code in small text so the user can copy-paste
            // it into a support request.
            //
            // Severity is `critical` here — unknown codes are exactly
            // the ones support most needs the diagnostic bundle for,
            // and the Flutter snackbar wiring automatically renders a
            // "Report" button on critical severity that opens the
            // logs-to-email dialog. End users can't read these codes
            // and we can't fix them blind, so we funnel every one of
            // them into the email flow.
            "Unexpected broker error. Click Report to send us a \
             diagnostic bundle so we can look at exactly what \
             happened.",
            None,
            None,
            "critical",
        ),
    };

    TranslatedError {
        code: upper,
        message: message.to_string(),
        action_label: action_label.map(|s| s.to_string()),
        action_target: action_target.map(|s| s.to_string()),
        severity,
    }
}

/// Convenience: pull a cTrader error code out of an anyhow error
/// message and translate it. Returns `None` if no recognisable
/// `errorCode=...` substring is found, which signals "this isn't a
/// broker error, render the message as-is".
pub fn translate_anyhow(err: &anyhow::Error) -> Option<TranslatedError> {
    let s = err.to_string();
    extract_error_code(&s).map(|code| translate_code(&code))
}

/// Pull a cTrader-style error code from a free-form error string.
/// We scan for either `errorCode=XXX` (JSON-extracted) or just an
/// uppercase token of the right shape that follows "code=" or "code:".
pub(crate) fn extract_error_code(s: &str) -> Option<String> {
    // Common shapes seen in our codebase:
    //   * `cTrader execution rejected: status=Failed code=Some("MARKET_CLOSED") description=...`
    //   * `errorCode=CH_ACCESS_TOKEN_INVALID description=...`
    //   * `{"errorCode":"CH_ACCESS_TOKEN_INVALID","description":"Invalid access token"}`
    // The match below covers all three.
    if let Some(rest) = s.find("errorCode=").map(|i| &s[i + "errorCode=".len()..]) {
        return Some(rest_to_token(rest));
    }
    if let Some(rest) = s.find("\"errorCode\":\"").map(|i| &s[i + "\"errorCode\":\"".len()..]) {
        return Some(rest.split('"').next().unwrap_or("").to_string());
    }
    if let Some(rest) = s.find("code=Some(\"").map(|i| &s[i + "code=Some(\"".len()..]) {
        return Some(rest.split('"').next().unwrap_or("").to_string());
    }
    None
}

/// Helper: read an UPPERCASE_SNAKE token from the start of a slice,
/// stopping at the first non-identifier character.
fn rest_to_token(rest: &str) -> String {
    rest.chars()
        .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_known_codes() {
        let t = translate_code("CH_ACCESS_TOKEN_INVALID");
        assert_eq!(t.action_target.as_deref(), Some("broker_setup"));
        assert_eq!(t.severity, "warning");
        assert!(t.message.contains("expired"));
    }

    #[test]
    fn unknown_code_falls_through_with_severity() {
        let t = translate_code("ZZZ_NEVER_SEEN_THIS");
        assert_eq!(t.code, "ZZZ_NEVER_SEEN_THIS");
        // Critical — Flutter renders a Report button on critical
        // severity that opens the email-logs flow. End users can't
        // act on unknown broker codes, so we route them all to us.
        assert_eq!(t.severity, "critical");
        assert!(t.action_label.is_none());
    }

    #[test]
    fn extracts_code_from_protocol_buffers_shape() {
        let s = "cTrader execution rejected: status=Failed code=Some(\"MARKET_CLOSED\") description=Some(\"...\")";
        assert_eq!(extract_error_code(s).as_deref(), Some("MARKET_CLOSED"));
    }

    #[test]
    fn extracts_code_from_json_shape() {
        let s = r#"{"errorCode":"CH_ACCESS_TOKEN_INVALID","description":"Invalid access token"}"#;
        assert_eq!(
            extract_error_code(s).as_deref(),
            Some("CH_ACCESS_TOKEN_INVALID")
        );
    }

    #[test]
    fn extracts_code_from_kv_shape() {
        let s = "broker said errorCode=RET_ACCOUNT_DISABLED description=...";
        assert_eq!(
            extract_error_code(s).as_deref(),
            Some("RET_ACCOUNT_DISABLED")
        );
    }

    #[test]
    fn returns_none_for_non_broker_errors() {
        let s = "io error: file not found";
        assert_eq!(extract_error_code(s), None);
    }
}
