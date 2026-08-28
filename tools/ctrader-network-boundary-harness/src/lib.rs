pub mod app_services;

#[cfg(test)]
mod harness_contract_tests {
    #[test]
    fn real_money_digits_boundary_accepts_broker_precision() {
        assert_eq!(
            crate::app_services::ctrader_money::required_money_digits(
                Some(8),
                "ProtoOAAsset.moneyDigits",
            )
            .expect("Spotware precision inside the documented range"),
            8
        );
    }

    #[test]
    fn typed_cancellation_classifier_finds_direct_and_wrapped_network_cancellation() {
        use crate::app_services::ctrader_historical_admission::{
            CTraderIoBoundaryError, CTraderIoPhase, HistoricalPublicationStartError,
            HistoricalRequestCancelled, is_historical_request_cancelled,
        };

        let direct = anyhow::Error::new(HistoricalRequestCancelled)
            .context("settings boundary after CPU admission");
        assert!(is_historical_request_cancelled(direct.as_ref()));

        let io_error = std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            CTraderIoBoundaryError::Cancelled {
                phase: CTraderIoPhase::ResponseRead,
            },
        );
        let wrapped = anyhow::Error::new(tungstenite::Error::Io(io_error))
            .context("persistent historical page receive");
        assert!(is_historical_request_cancelled(wrapped.as_ref()));

        let publication_boundary = anyhow::Error::new(HistoricalPublicationStartError::Cancelled(
            HistoricalRequestCancelled,
        ))
        .context("atomic publication transition");
        assert!(is_historical_request_cancelled(
            publication_boundary.as_ref()
        ));

        let deadline = anyhow::Error::new(CTraderIoBoundaryError::DeadlineExceeded {
            phase: CTraderIoPhase::ResponseRead,
            timeout: std::time::Duration::from_millis(25),
        })
        .context("deadline is not operator cancellation");
        assert!(!is_historical_request_cancelled(deadline.as_ref()));
    }
}
