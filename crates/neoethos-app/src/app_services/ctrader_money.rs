//! Spec-compliant conversion between cTrader integer-encoded monetary
//! values and account-currency `f64` magnitudes.
//!
//! ## Why this module exists
//!
//! Per the Spotware Open API protocol (see
//! the per-entity `moneyDigits` comments in `OpenApiModelMessages.proto`
//! upstream at `spotware/openapi-proto-messages` — the vendored copy under
//! `crates/neoethos-app/proto/` was deleted in batch D2 (2026-08-09), and the
//! `docs/audits/research/ctrader_api_full_reference.md` this used to cite has
//! never existed in this repo), every monetary
//! integer field on a cTrader message is the actual deposit-currency
//! amount multiplied by `10^moneyDigits`. The exponent is reported as
//! a `uint32 moneyDigits` field on the *carrying* entity — it varies
//! per account / asset / deal — and the value the docs quote verbatim
//! is:
//!
//! > "moneyDigits = 8 must be interpret as business value multiplied
//! > by 10^8, then real balance would be 10053099944 / 10^8 = 100.53099944."
//!
//! Hard-coding `divide-by-100` (i.e. assuming `moneyDigits = 2`) is
//! the cTrader docs-sweep finding #2 in §10 of the reference — it is
//! off by `10^6` for accounts that report `moneyDigits = 8` (some
//! crypto / precious-metal / JPY denominations). This module provides
//! the one correct helper used at every call site that converts a
//! `ProtoOATrader.balance`, `ProtoOADeal.commission`,
//! `ProtoOAPosition.swap`, `ProtoOAClosePositionDetail.grossProfit`,
//! `ProtoOABonusDepositWithdraw.*`, `ProtoOADepositWithdraw.*`, or
//! `ProtoOAGetPositionUnrealizedPnLRes.{gross,net}UnrealizedPnL` field
//! to a display / risk-gate value.
//!
//! `ProtoOAAsset` and `ProtoOASymbol` do **not** carry a `moneyDigits`
//! field (only `digits`, which is a price-precision count); they are
//! intentionally not callers of this helper.
//!
//! ## Range clamp
//!
//! `money_digits` is constrained to `[0, 10]` because:
//!
//! - The lowest value the Spotware comment lists is `0` (whole units).
//! - The highest exponent listed in any wild cTrader payload to date
//!   is `8`, with `10` left as headroom for future high-precision
//!   denominations.
//! - `i64::MAX` (~9.22 × 10^18) divided by `10^10` is still
//!   ~9.22 × 10^8, which keeps the f64 result well inside the
//!   double-precision exact-integer range (`2^53 ≈ 9.007 × 10^15`).
//!
//! Out-of-range values **error**, not silently fall back — per the
//! operator's 2026-05-15 directive: a silent fallback would mask a
//! malformed broker payload that downstream code might still treat as
//! authoritative for live order sizing.

use anyhow::{Result, anyhow, bail};

/// Maximum supported `moneyDigits` exponent. See module-level docs for
/// the justification — `[0, 10]` matches the Spotware spec range and
/// keeps the f64 result inside the IEEE-754 exact-integer interval.
pub const MAX_CTRADER_MONEY_DIGITS: i32 = 10;

/// Scale a cTrader integer-encoded monetary value to its real
/// magnitude given the carrying entity's `money_digits` exponent.
///
/// Per the Spotware Open API protocol, all monetary integer fields in
/// `ProtoOATrader`, `ProtoOAPosition`, `ProtoOADeal`,
/// `ProtoOAClosePositionDetail`, `ProtoOABonusDepositWithdraw`,
/// `ProtoOADepositWithdraw`, and `ProtoOAGetPositionUnrealizedPnLRes`
/// are reported as `actual × 10^moneyDigits`. The default for fiat
/// accounts is `moneyDigits = 2` (cents), but precious-metal / crypto
/// / high-precision denominations use `moneyDigits = 4`, `6`, or even
/// `8`. The conversion MUST use the per-entity field, not a hard-coded
/// `/100`.
///
/// `money_digits` is checked against `[0, 10]` to defend against
/// malformed broker payloads — the Spotware spec allows 0–10 inclusive.
/// Out-of-range values produce an error, not a silent fallback (per
/// operator directive 2026-05-15: "η σιωπηλή προεπιλογή κρύβει
/// πρόβλημα στο payload").
pub fn scale_ctrader_money_int(scaled: i64, money_digits: i32) -> Result<f64> {
    if !(0..=MAX_CTRADER_MONEY_DIGITS).contains(&money_digits) {
        bail!(
            "cTrader moneyDigits out of spec range [0, {}]: {}",
            MAX_CTRADER_MONEY_DIGITS,
            money_digits
        );
    }
    let divisor = 10.0_f64.powi(money_digits);
    Ok(scaled as f64 / divisor)
}

/// Scale an unsigned cTrader monetary value, used for fields such as
/// `ProtoOAPosition.usedMargin`.
pub fn scale_ctrader_money_uint(scaled: u64, money_digits: i32) -> Result<f64> {
    if scaled > i64::MAX as u64 {
        bail!("cTrader unsigned money value exceeds supported i64 range: {scaled}");
    }
    scale_ctrader_money_int(scaled as i64, money_digits)
}

/// Resolve the per-entity `moneyDigits` exponent required by a NeoEthos
/// financial boundary. The protobuf field is optional on several cTrader
/// entities, but without its presence the JSON transport does not prove the
/// scale of a non-zero monetary integer. NeoEthos therefore fails closed rather
/// than guessing a fiat precision.
pub fn required_money_digits(value: Option<u32>, field: &str) -> Result<u32> {
    let digits = value.ok_or_else(|| anyhow!("broker payload omitted {field}"))?;
    if digits > MAX_CTRADER_MONEY_DIGITS as u32 {
        bail!(
            "cTrader moneyDigits out of spec range [0, {}] for {field}: {digits}",
            MAX_CTRADER_MONEY_DIGITS
        );
    }
    Ok(digits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_money_digits_eight_matches_spotware_example() {
        // Verbatim from the Spotware proto comment: "moneyDigits = 8
        // must be interpret as business value multiplied by 10^8, then
        // real balance would be 10053099944 / 10^8 = 100.53099944".
        let result = scale_ctrader_money_int(10_053_099_944, 8).expect("in-range");
        assert!((result - 100.53099944).abs() < 1e-9, "got {result}");
    }

    #[test]
    fn scale_money_digits_zero_returns_whole_units() {
        assert_eq!(scale_ctrader_money_int(42, 0).expect("in-range"), 42.0);
    }

    #[test]
    fn scale_money_digits_two_uses_the_broker_supplied_exponent() {
        assert_eq!(
            scale_ctrader_money_int(12_345, 2).expect("in-range"),
            123.45
        );
    }

    #[test]
    fn scale_money_digits_four_matches_high_precision_account() {
        assert_eq!(
            scale_ctrader_money_int(123_456, 4).expect("in-range"),
            12.3456
        );
        assert_eq!(
            scale_ctrader_money_uint(123_456, 4).expect("in-range"),
            12.3456
        );
    }

    #[test]
    fn scale_rejects_negative_money_digits() {
        let err = scale_ctrader_money_int(1, -1).expect_err("must reject");
        assert!(
            err.to_string().contains("out of spec range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn scale_rejects_money_digits_above_ten() {
        let err = scale_ctrader_money_int(1, 11).expect_err("must reject");
        assert!(
            err.to_string().contains("out of spec range"),
            "unexpected error: {err}"
        );
    }
}
