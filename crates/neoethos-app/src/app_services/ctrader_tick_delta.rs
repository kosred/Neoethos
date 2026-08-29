use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DecodedCTraderTick {
    pub(crate) timestamp_ms: i64,
    pub(crate) price: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CTraderTickDeltaError {
    TimestampOverflow {
        row: usize,
    },
    RawPriceOverflow {
        row: usize,
    },
    NegativeTimestamp {
        row: usize,
        timestamp_ms: i64,
    },
    NonPositiveRawPrice {
        row: usize,
        raw_price: i64,
    },
    NotStrictlyNewestFirst {
        row: usize,
        previous_timestamp_ms: i64,
        timestamp_ms: i64,
    },
    InvalidConvertedPrice {
        row: usize,
        price: f64,
    },
}

impl fmt::Display for CTraderTickDeltaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimestampOverflow { row } => {
                write!(
                    formatter,
                    "cTrader tick row {row} timestamp delta overflows i64"
                )
            }
            Self::RawPriceOverflow { row } => {
                write!(
                    formatter,
                    "cTrader tick row {row} price delta overflows i64"
                )
            }
            Self::NegativeTimestamp { row, timestamp_ms } => write!(
                formatter,
                "cTrader tick row {row} decoded to negative timestamp {timestamp_ms}"
            ),
            Self::NonPositiveRawPrice { row, raw_price } => write!(
                formatter,
                "cTrader tick row {row} decoded to non-positive raw price {raw_price}"
            ),
            Self::NotStrictlyNewestFirst {
                row,
                previous_timestamp_ms,
                timestamp_ms,
            } => write!(
                formatter,
                "cTrader tick row {row} is not strictly newest-first: previous={previous_timestamp_ms}, decoded={timestamp_ms}"
            ),
            Self::InvalidConvertedPrice { row, price } => write!(
                formatter,
                "cTrader tick row {row} converted to invalid price {price}"
            ),
        }
    }
}

impl Error for CTraderTickDeltaError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CTraderTickResponseIdentityError {
    AccountMismatch { expected: i64, actual: i64 },
    ClientMessageMismatch { expected: String, actual: String },
}

impl fmt::Display for CTraderTickResponseIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AccountMismatch { expected, actual } => write!(
                formatter,
                "cTrader tick-data account mismatch: expected {expected}, received {actual}"
            ),
            Self::ClientMessageMismatch { expected, actual } => write!(
                formatter,
                "cTrader tick-data client message mismatch: expected {expected:?}, received {actual:?}"
            ),
        }
    }
}

impl Error for CTraderTickResponseIdentityError {}

/// Validate the response fields which the official `ProtoOAGetTickDataRes`
/// schema actually carries. It does not carry `symbolId`; the symbol remains
/// bound to the request whose exact `clientMsgId` this response echoes.
pub(crate) fn validate_ctrader_tick_response_identity(
    expected_account_id: i64,
    expected_client_msg_id: &str,
    actual_account_id: i64,
    actual_client_msg_id: &str,
) -> Result<(), CTraderTickResponseIdentityError> {
    if actual_account_id != expected_account_id {
        return Err(CTraderTickResponseIdentityError::AccountMismatch {
            expected: expected_account_id,
            actual: actual_account_id,
        });
    }
    if actual_client_msg_id != expected_client_msg_id {
        return Err(CTraderTickResponseIdentityError::ClientMessageMismatch {
            expected: expected_client_msg_id.to_owned(),
            actual: actual_client_msg_id.to_owned(),
        });
    }
    Ok(())
}

/// Decode `ProtoOATickData` in broker wire order. The first timestamp and
/// price are absolute; every later value is a signed delta added to the
/// previously decoded value. The broker returns newest first, which is
/// validated here before the caller may reverse the complete result into
/// canonical ascending order.
pub(crate) fn decode_ctrader_tick_deltas<I, F>(
    wire_rows: I,
    mut convert_price: F,
) -> Result<Vec<DecodedCTraderTick>, CTraderTickDeltaError>
where
    I: IntoIterator<Item = (i64, i64)>,
    F: FnMut(i64) -> f64,
{
    let iterator = wire_rows.into_iter();
    let mut decoded = Vec::with_capacity(iterator.size_hint().0);
    let mut previous_timestamp: Option<i64> = None;
    let mut previous_raw_price: Option<i64> = None;

    for (row, (timestamp_wire, price_wire)) in iterator.enumerate() {
        let timestamp_ms = match previous_timestamp {
            None => timestamp_wire,
            Some(previous) => previous
                .checked_add(timestamp_wire)
                .ok_or(CTraderTickDeltaError::TimestampOverflow { row })?,
        };
        if timestamp_ms < 0 {
            return Err(CTraderTickDeltaError::NegativeTimestamp { row, timestamp_ms });
        }
        if let Some(previous) = previous_timestamp {
            if timestamp_ms >= previous {
                return Err(CTraderTickDeltaError::NotStrictlyNewestFirst {
                    row,
                    previous_timestamp_ms: previous,
                    timestamp_ms,
                });
            }
        }

        let raw_price = match previous_raw_price {
            None => price_wire,
            Some(previous) => previous
                .checked_add(price_wire)
                .ok_or(CTraderTickDeltaError::RawPriceOverflow { row })?,
        };
        if raw_price <= 0 {
            return Err(CTraderTickDeltaError::NonPositiveRawPrice { row, raw_price });
        }
        let price = convert_price(raw_price);
        if !price.is_finite() || price <= 0.0 {
            return Err(CTraderTickDeltaError::InvalidConvertedPrice { row, price });
        }

        previous_timestamp = Some(timestamp_ms);
        previous_raw_price = Some(raw_price);
        decoded.push(DecodedCTraderTick {
            timestamp_ms,
            price,
        });
    }

    Ok(decoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_json_wss_signed_deltas_reconstruct_newest_first_ticks() {
        // Evidence schema: `ten_year_tick_samples[0].raw_wire_preview` from
        // `ctrader-direct-14tf-sample.json`; SHA-256
        // D465484FD0BF6F5DC9F1D3CA945020C172C4C547D347B774A29566F3B2E9AF14.
        // The artifact contains no raw account id or credentials; identity is
        // hashed. First values are absolute; later values are signed deltas.
        let wire_rows = [
            (1_471_446_060_064, 112_742),
            (-1_708, 1),
            (-805, -2),
            (-194, -1),
            (-192, -1),
            (-187, 1),
        ];

        let decoded = decode_ctrader_tick_deltas(wire_rows, |raw| raw as f64 / 100_000.0)
            .expect("captured cTrader deltas must decode");

        assert_eq!(
            decoded
                .iter()
                .map(|tick| tick.timestamp_ms)
                .collect::<Vec<_>>(),
            vec![
                1_471_446_060_064,
                1_471_446_058_356,
                1_471_446_057_551,
                1_471_446_057_357,
                1_471_446_057_165,
                1_471_446_056_978,
            ]
        );
        assert_eq!(
            decoded
                .iter()
                .map(|tick| (tick.price * 100_000.0).round() as i64)
                .collect::<Vec<_>>(),
            vec![112_742, 112_743, 112_741, 112_740, 112_739, 112_740]
        );
        assert!(
            decoded
                .windows(2)
                .all(|pair| pair[0].timestamp_ms > pair[1].timestamp_ms),
            "decoded wire order must be strictly newest-first before ascending reversal"
        );
        assert!(
            decoded
                .iter()
                .all(|tick| tick.price.is_finite() && tick.price > 0.0)
        );
    }

    #[test]
    fn signed_delta_overflow_fails_without_partial_output() {
        let timestamp_error =
            decode_ctrader_tick_deltas([(i64::MAX, 100), (1, -1)], |raw| raw as f64)
                .expect_err("timestamp overflow must fail");
        assert_eq!(
            timestamp_error,
            CTraderTickDeltaError::TimestampOverflow { row: 1 }
        );

        let price_error = decode_ctrader_tick_deltas([(10, i64::MAX), (-1, 1)], |raw| raw as f64)
            .expect_err("price overflow must fail");
        assert_eq!(
            price_error,
            CTraderTickDeltaError::RawPriceOverflow { row: 1 }
        );
    }

    #[test]
    fn invalid_wire_order_and_prices_fail_closed() {
        let order_error = decode_ctrader_tick_deltas([(100, 100), (0, -1)], |raw| raw as f64)
            .expect_err("equal decoded timestamps are not strictly newest-first");
        assert_eq!(
            order_error,
            CTraderTickDeltaError::NotStrictlyNewestFirst {
                row: 1,
                previous_timestamp_ms: 100,
                timestamp_ms: 100,
            }
        );

        assert!(matches!(
            decode_ctrader_tick_deltas([(-1, 100)], |raw| raw as f64),
            Err(CTraderTickDeltaError::NegativeTimestamp { row: 0, .. })
        ));
        assert!(matches!(
            decode_ctrader_tick_deltas([(100, 0)], |raw| raw as f64),
            Err(CTraderTickDeltaError::NonPositiveRawPrice { row: 0, .. })
        ));
        assert!(matches!(
            decode_ctrader_tick_deltas([(100, 100)], |_| f64::NAN),
            Err(CTraderTickDeltaError::InvalidConvertedPrice { row: 0, .. })
        ));
    }

    #[test]
    fn empty_broker_result_remains_an_empty_complete_decode() {
        let decoded =
            decode_ctrader_tick_deltas(std::iter::empty::<(i64, i64)>(), |raw| raw as f64)
                .expect("an empty tickData list is a valid empty broker result");
        assert!(decoded.is_empty());
    }

    #[test]
    fn response_identity_requires_exact_account_and_client_message() {
        validate_ctrader_tick_response_identity(7, "ticks-1", 7, "ticks-1")
            .expect("matching request identity");

        assert_eq!(
            validate_ctrader_tick_response_identity(7, "ticks-1", 8, "ticks-1")
                .expect_err("account mismatch must fail"),
            CTraderTickResponseIdentityError::AccountMismatch {
                expected: 7,
                actual: 8,
            }
        );
        assert_eq!(
            validate_ctrader_tick_response_identity(7, "ticks-1", 7, "ticks-2")
                .expect_err("client message mismatch must fail"),
            CTraderTickResponseIdentityError::ClientMessageMismatch {
                expected: "ticks-1".to_owned(),
                actual: "ticks-2".to_owned(),
            }
        );
    }

    #[test]
    fn parser_wiring_validates_wire_order_then_reverses_without_sort_repair() {
        let source = include_str!("ctrader_data.rs");
        let parser = source
            .split("pub fn parse_tick_data_response")
            .nth(1)
            .and_then(|tail| tail.split("pub fn load_historical_bars_only(").next())
            .expect("tick parser source");
        let identity_check = parser
            .find("validate_ctrader_tick_response_identity")
            .expect("exact account/client-message identity check");
        let decode = parser
            .find("decode_ctrader_tick_deltas")
            .expect("signed-delta decoder call");
        let reverse = parser.find("ticks.reverse()").expect("ascending reverse");

        assert!(identity_check < decode && decode < reverse);
        assert!(parser.contains("symbol_id: symbol.symbol_id"));
        assert!(!parser.contains("sort_by_key"));
        assert!(!parser.contains("previous -"));
    }

    #[test]
    fn parser_uses_the_official_tick_response_identity_fields() {
        let source = include_str!("ctrader_data.rs");
        let envelope = source
            .split("struct TickDataEnvelope")
            .nth(1)
            .and_then(|tail| tail.split("struct TickDataPayload").next())
            .expect("tick envelope source");
        let payload = source
            .split("struct TickDataPayload")
            .nth(1)
            .and_then(|tail| tail.split("struct TickPayload").next())
            .expect("tick payload source");

        assert!(envelope.contains("client_msg_id: String"));
        assert!(!envelope.contains("client_msg_id: Option"));
        assert!(payload.contains("ctid_trader_account_id: i64"));
        assert!(!payload.contains("ctid_trader_account_id: Option"));
        assert!(payload.contains("has_more: bool"));
        assert!(payload.contains("tick_data: Vec<TickPayload>"));
        assert!(!payload.contains("symbol_id"));
        assert!(!payload.contains("#[serde(rename = \"tickData\", default)]"));
    }
}
