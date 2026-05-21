//! Spec-compliant conversion between cTrader integer-encoded monetary
//! values and account-currency `f64` magnitudes.
//!
//! ## Why this module exists
//!
//! Per the Spotware Open API protocol (see
//! `docs/audits/research/ctrader_api_full_reference.md` §5.14 and the
//! per-entity `moneyDigits` comments in
//! `crates/neoethos-app/proto/OpenApiModelMessages.proto`), every monetary
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

use anyhow::{Result, bail};

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

/// Resolve a required per-entity `moneyDigits` exponent from broker
/// payloads. Missing values keep the legacy fiat fallback but log
/// loudly so malformed live payloads are visible.
pub fn required_money_digits(value: Option<u32>, field: &str) -> u32 {
    value.unwrap_or_else(|| {
        tracing::error!(
            target: "neoethos_app::ctrader",
            field,
            "broker payload omitted required money_digits; defaulting to 2 \
             (silent default to 0 would mis-scale monetary values)"
        );
        2
    })
}

/// Inverse of [`scale_ctrader_money_int`] for outgoing values
/// (e.g. when a future code path wants to emit a scaled monetary limit
/// in the cTrader wire format).
///
/// Returns an error when:
/// - `money_digits` is outside `[0, 10]`,
/// - `actual` is non-finite (NaN / ±inf),
/// - the scaled product cannot fit in an `i64`.
///
/// `#[allow(dead_code)]`: inverse of the scaled→display path that
/// IS hot. Today the SDK receives prices and never sends, but the
/// inverse will be reached when order-submission paths start quoting
/// in display units. Tests below pin every numeric edge case.
#[allow(dead_code)]
pub fn unscale_to_ctrader_money_int(actual: f64, money_digits: i32) -> Result<i64> {
    if !(0..=MAX_CTRADER_MONEY_DIGITS).contains(&money_digits) {
        bail!(
            "cTrader moneyDigits out of spec range [0, {}]: {}",
            MAX_CTRADER_MONEY_DIGITS,
            money_digits
        );
    }
    if !actual.is_finite() {
        bail!("cannot unscale non-finite value: {}", actual);
    }
    let multiplier = 10.0_f64.powi(money_digits);
    let scaled = (actual * multiplier).round();
    if !scaled.is_finite() || scaled.abs() >= i64::MAX as f64 {
        bail!("scaled value overflows i64: {}", scaled);
    }
    Ok(scaled as i64)
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
    fn scale_money_digits_two_matches_legacy_cents_behaviour() {
        // The pre-fix code path hard-coded `value as f64 / 100.0`,
        // which is exactly `scale_ctrader_money_int(_, 2)` for fiat
        // accounts. Pin the equivalence so the migration cannot
        // regress the default-currency case.
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

    #[test]
    fn unscale_roundtrips_for_every_in_range_money_digits() {
        // For each supported exponent, an integer value scaled and
        // then unscaled must reproduce the original i64. We use a
        // small integer so the `actual × 10^d` product stays inside
        // f64's exact-integer interval (`2^53`).
        for d in 0..=MAX_CTRADER_MONEY_DIGITS {
            let original: i64 = 123_456_789;
            let scaled = scale_ctrader_money_int(original, d).expect("in-range");
            let unscaled = unscale_to_ctrader_money_int(scaled, d).expect("in-range");
            assert_eq!(
                unscaled, original,
                "roundtrip failure at money_digits={d}: {scaled} → {unscaled}"
            );
        }
    }

    #[test]
    fn unscale_rejects_non_finite_input() {
        let err = unscale_to_ctrader_money_int(f64::NAN, 2).expect_err("must reject");
        assert!(
            err.to_string().contains("non-finite"),
            "unexpected error: {err}"
        );
        let err = unscale_to_ctrader_money_int(f64::INFINITY, 2).expect_err("must reject");
        assert!(
            err.to_string().contains("non-finite"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unscale_rejects_out_of_range_money_digits() {
        assert!(unscale_to_ctrader_money_int(1.0, -1).is_err());
        assert!(unscale_to_ctrader_money_int(1.0, 11).is_err());
    }

    #[test]
    fn unscale_rejects_i64_overflow() {
        // f64::MAX * 10^2 obviously can't fit in i64.
        let err = unscale_to_ctrader_money_int(f64::MAX, 2).expect_err("must reject");
        assert!(
            err.to_string().contains("overflows"),
            "unexpected error: {err}"
        );
    }
}
