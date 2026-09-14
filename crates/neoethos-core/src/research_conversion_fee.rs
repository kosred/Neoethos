//! Versioned conversion-fee assumptions for numerical research, never broker settlement.
//!
//! cTrader describes the rate as a percentage of realized gross P&L, charged
//! only when quote and deposit assets differ. Gross P&L excludes swaps and
//! commissions (https://help.ctrader.com/open-api/model-messages/ and
//! https://help.ctrader.com/ctrader-web/interface/trade-watch/).
//! Those sources do not specify the negative-P&L formula or settlement rounding.
//! V1 explicitly assumes a debit on absolute realized price gross, with no
//! rounding. Actual execution must retain the broker-reported monetary fee.

use serde::{Deserialize, Serialize};

/// Mandatory wire identity: no default may reinterpret older net-discount results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResearchPnlConversionFeePolicyV1 {
    #[serde(rename = "absolute_realized_price_gross_debit_v1")]
    AbsoluteRealizedPriceGrossDebitV1,
}

pub fn validate_conversion_fee_rate_v1(rate: f64) -> Result<(), &'static str> {
    if !rate.is_finite() || !(0.0..1.0).contains(&rate) {
        return Err("pnl_conversion_fee_rate must be finite and in [0, 1)");
    }
    Ok(())
}

/// Resolve applicability separately from the symbol's published rate.
pub fn effective_conversion_fee_rate_v1(
    quoted_rate: f64,
    quote_currency: &str,
    account_currency: &str,
) -> Result<f64, &'static str> {
    validate_conversion_fee_rate_v1(quoted_rate)?;
    for currency in [quote_currency, account_currency] {
        if currency.len() != 3 || !currency.bytes().all(|byte| byte.is_ascii_uppercase()) {
            return Err("conversion fee requires exact uppercase quote and account currencies");
        }
    }
    Ok(if quote_currency == account_currency {
        0.0
    } else {
        quoted_rate
    })
}

impl ResearchPnlConversionFeePolicyV1 {
    /// `price_gross` includes the modeled executable spread but no commission/swap.
    pub fn debit(self, price_gross: f64, effective_rate: f64) -> Result<f64, &'static str> {
        validate_conversion_fee_rate_v1(effective_rate)?;
        if !price_gross.is_finite() {
            return Err("conversion fee price gross must be finite");
        }
        // rate < 1 and finite gross make this product finite, including losses.
        Ok(price_gross.abs() * effective_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_fee_is_a_debit_for_profit_loss_and_zero() {
        let policy = ResearchPnlConversionFeePolicyV1::AbsoluteRealizedPriceGrossDebitV1;
        for (gross, expected) in [(100.0, 0.5_f64), (-100.0, 0.5), (0.0, 0.0), (-0.0, 0.0)] {
            assert_eq!(
                policy.debit(gross, 0.005).unwrap().to_bits(),
                expected.to_bits()
            );
        }
        assert!(policy.debit(f64::MAX, 0.999).unwrap().is_finite());
        for invalid in [f64::NAN, f64::INFINITY, -0.01, 1.0] {
            assert!(policy.debit(100.0, invalid).is_err());
        }
        assert!(policy.debit(f64::NEG_INFINITY, 0.01).is_err());
    }

    #[test]
    fn effective_rate_is_zero_only_for_known_matching_currencies() {
        assert_eq!(
            effective_conversion_fee_rate_v1(0.005, "USD", "USD"),
            Ok(0.0)
        );
        assert_eq!(
            effective_conversion_fee_rate_v1(0.005, "USD", "EUR"),
            Ok(0.005)
        );
        assert!(effective_conversion_fee_rate_v1(0.005, "", "").is_err());
        assert!(effective_conversion_fee_rate_v1(f64::NAN, "USD", "USD").is_err());
    }

    #[test]
    fn policy_wire_does_not_accept_the_old_net_discount() {
        let policy = ResearchPnlConversionFeePolicyV1::AbsoluteRealizedPriceGrossDebitV1;
        assert_eq!(
            serde_json::to_value(policy).unwrap(),
            "absolute_realized_price_gross_debit_v1"
        );
        assert!(
            serde_json::from_value::<ResearchPnlConversionFeePolicyV1>(serde_json::json!(
                "multiply_net_pnl_by_one_minus_rate"
            ))
            .is_err()
        );
    }
}
