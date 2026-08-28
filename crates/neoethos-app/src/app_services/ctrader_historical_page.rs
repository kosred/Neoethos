use std::error::Error;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HistoricalPageClientMsgIdOverflow;

impl fmt::Display for HistoricalPageClientMsgIdOverflow {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("cTrader historical page clientMsgId counter overflowed")
    }
}

impl Error for HistoricalPageClientMsgIdOverflow {}

/// Deterministic, connection-local request correlation for one historical
/// capture. The counter is never shared or reset while a session is alive.
pub(crate) struct HistoricalPageClientMsgIds {
    next_index: u64,
}

impl HistoricalPageClientMsgIds {
    pub(crate) fn new() -> Self {
        Self { next_index: 0 }
    }

    pub(crate) fn next_trendbars(&mut self) -> Result<String, HistoricalPageClientMsgIdOverflow> {
        let index = self.next_index;
        self.next_index = index
            .checked_add(1)
            .ok_or(HistoricalPageClientMsgIdOverflow)?;
        Ok(format!("history-trendbars-page-{index}"))
    }
}

/// Testable lifecycle seam for one persistent cTrader historical socket.
/// Production and fakes execute the same ordered state machine; only the wire
/// operations and bounded page type differ.
pub(crate) trait CTraderPersistentHistoricalWire {
    type Error: From<HistoricalPageClientMsgIdOverflow>;
    type Page;
    type PageRequest;

    fn connect(&mut self) -> Result<(), Self::Error>;
    fn application_auth(&mut self) -> Result<(), Self::Error>;
    fn account_auth(&mut self) -> Result<(), Self::Error>;
    fn symbols_list(&mut self) -> Result<(), Self::Error>;
    fn symbol_detail(&mut self) -> Result<(), Self::Error>;
    fn trendbars(
        &mut self,
        client_msg_id: String,
        request: Self::PageRequest,
    ) -> Result<Self::Page, Self::Error>;
}

pub(crate) struct CTraderPersistentHistoricalSession<W> {
    wire: W,
    page_client_msg_ids: HistoricalPageClientMsgIds,
}

impl<W: CTraderPersistentHistoricalWire> CTraderPersistentHistoricalSession<W> {
    pub(crate) fn establish(mut wire: W) -> Result<Self, W::Error> {
        wire.connect()?;
        wire.application_auth()?;
        wire.account_auth()?;
        wire.symbols_list()?;
        wire.symbol_detail()?;
        Ok(Self {
            wire,
            page_client_msg_ids: HistoricalPageClientMsgIds::new(),
        })
    }

    pub(crate) fn next_trendbars(&mut self, request: W::PageRequest) -> Result<W::Page, W::Error> {
        let client_msg_id = self.page_client_msg_ids.next_trendbars()?;
        self.wire.trendbars(client_msg_id, request)
    }

    #[cfg(any(test, feature = "broker-history-service"))]
    pub(crate) fn wire(&self) -> &W {
        &self.wire
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FakeFailure {
        ClientMsgIdOverflow,
        Page,
        Cancelled,
        BlockedPayloadType,
    }

    impl From<HistoricalPageClientMsgIdOverflow> for FakeFailure {
        fn from(_: HistoricalPageClientMsgIdOverflow) -> Self {
            Self::ClientMsgIdOverflow
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum FakeEvent {
        Connect,
        ApplicationAuth,
        AccountAuth,
        SymbolsList,
        SymbolDetail,
        Page {
            request: usize,
            client_msg_id: String,
        },
    }

    struct FakePage {
        request: usize,
        outstanding: Arc<AtomicUsize>,
    }

    impl Drop for FakePage {
        fn drop(&mut self) {
            self.outstanding.fetch_sub(1, Ordering::SeqCst);
        }
    }

    struct FakeWire {
        events: Vec<FakeEvent>,
        fail_page: Option<(usize, FakeFailure)>,
        outstanding_pages: Arc<AtomicUsize>,
    }

    impl FakeWire {
        fn new(fail_page: Option<(usize, FakeFailure)>) -> Self {
            Self {
                events: Vec::new(),
                fail_page,
                outstanding_pages: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl CTraderPersistentHistoricalWire for FakeWire {
        type Error = FakeFailure;
        type Page = FakePage;
        type PageRequest = usize;

        fn connect(&mut self) -> Result<(), Self::Error> {
            self.events.push(FakeEvent::Connect);
            Ok(())
        }

        fn application_auth(&mut self) -> Result<(), Self::Error> {
            self.events.push(FakeEvent::ApplicationAuth);
            Ok(())
        }

        fn account_auth(&mut self) -> Result<(), Self::Error> {
            self.events.push(FakeEvent::AccountAuth);
            Ok(())
        }

        fn symbols_list(&mut self) -> Result<(), Self::Error> {
            self.events.push(FakeEvent::SymbolsList);
            Ok(())
        }

        fn symbol_detail(&mut self) -> Result<(), Self::Error> {
            self.events.push(FakeEvent::SymbolDetail);
            Ok(())
        }

        fn trendbars(
            &mut self,
            client_msg_id: String,
            request: Self::PageRequest,
        ) -> Result<Self::Page, Self::Error> {
            assert_eq!(
                self.outstanding_pages.load(Ordering::SeqCst),
                0,
                "the prior bounded page escaped into the next request"
            );
            self.events.push(FakeEvent::Page {
                request,
                client_msg_id,
            });
            if let Some((failed_request, failure)) = self.fail_page
                && failed_request == request
            {
                return Err(failure);
            }
            self.outstanding_pages.fetch_add(1, Ordering::SeqCst);
            Ok(FakePage {
                request,
                outstanding: Arc::clone(&self.outstanding_pages),
            })
        }
    }

    fn run_fake_capture(
        session: &mut CTraderPersistentHistoricalSession<FakeWire>,
        pages: usize,
        published: &mut bool,
    ) -> Result<(), FakeFailure> {
        for request in 0..pages {
            let page = session.next_trendbars(request)?;
            assert_eq!(page.request, request);
            drop(page);
        }
        *published = true;
        Ok(())
    }

    #[test]
    fn page_client_message_ids_are_unique_and_deterministic() {
        let mut ids = HistoricalPageClientMsgIds::new();
        assert_eq!(
            ids.next_trendbars().expect("page zero"),
            "history-trendbars-page-0"
        );
        assert_eq!(
            ids.next_trendbars().expect("page one"),
            "history-trendbars-page-1"
        );
        assert_eq!(
            ids.next_trendbars().expect("page two"),
            "history-trendbars-page-2"
        );
    }

    #[test]
    fn page_client_message_id_overflow_fails_closed() {
        let mut ids = HistoricalPageClientMsgIds {
            next_index: u64::MAX,
        };
        assert_eq!(
            ids.next_trendbars().expect_err("overflow must fail"),
            HistoricalPageClientMsgIdOverflow
        );
    }

    #[test]
    fn behavioral_session_connects_authenticates_resolves_once_for_many_pages() {
        let mut session = CTraderPersistentHistoricalSession::establish(FakeWire::new(None))
            .expect("authenticated fake session");
        let outstanding = Arc::clone(&session.wire().outstanding_pages);
        let mut published = false;

        run_fake_capture(&mut session, 3, &mut published).expect("three-page capture");

        assert!(published);
        assert_eq!(outstanding.load(Ordering::SeqCst), 0);
        assert_eq!(
            session.wire().events,
            [
                FakeEvent::Connect,
                FakeEvent::ApplicationAuth,
                FakeEvent::AccountAuth,
                FakeEvent::SymbolsList,
                FakeEvent::SymbolDetail,
                FakeEvent::Page {
                    request: 0,
                    client_msg_id: "history-trendbars-page-0".to_owned(),
                },
                FakeEvent::Page {
                    request: 1,
                    client_msg_id: "history-trendbars-page-1".to_owned(),
                },
                FakeEvent::Page {
                    request: 2,
                    client_msg_id: "history-trendbars-page-2".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn page_cancel_and_blocked_failures_stop_later_sends_and_publication() {
        for failure in [
            FakeFailure::Page,
            FakeFailure::Cancelled,
            FakeFailure::BlockedPayloadType,
        ] {
            let mut session =
                CTraderPersistentHistoricalSession::establish(FakeWire::new(Some((1, failure))))
                    .expect("authenticated fake session");
            let mut published = false;

            assert_eq!(
                run_fake_capture(&mut session, 3, &mut published),
                Err(failure)
            );
            assert!(!published);
            assert_eq!(
                session
                    .wire()
                    .events
                    .iter()
                    .filter(|event| matches!(event, FakeEvent::Page { .. }))
                    .count(),
                2,
                "a page was sent after the terminal failure"
            );
            assert!(
                !session
                    .wire()
                    .events
                    .iter()
                    .any(|event| matches!(event, FakeEvent::Page { request: 2, .. }))
            );
        }
    }

    #[test]
    fn authenticated_history_session_uses_one_socket_and_one_page_response() {
        let source = include_str!("ctrader_data.rs");
        let production_wire = source
            .split(
                "impl CTraderPersistentHistoricalWire for ProductionCTraderPersistentHistoricalWire",
            )
            .nth(1)
            .and_then(|tail| {
                tail.split("pub(crate) struct CTraderAuthenticatedHistoricalSession")
                    .next()
            })
            .expect("production persistent wire implementation");
        let session = source
            .split("pub(crate) struct CTraderAuthenticatedHistoricalSession")
            .nth(1)
            .and_then(|tail| tail.split("pub fn load_historical_bars_only(").next())
            .expect("authenticated history session source");

        assert_eq!(
            production_wire
                .matches("connect_session(Some(&self.cancellation))")
                .count(),
            1
        );
        assert!(session.contains(
            "CTraderPersistentHistoricalSession<ProductionCTraderPersistentHistoricalWire>"
        ));
        assert!(session.contains("CTraderPersistentHistoricalSession::establish(wire)"));
        assert!(!session.contains("send_sequence("));
        assert!(production_wire.contains("ensure_ctrader_response_account_id"));
        assert!(production_wire.contains("parse_trendbars_response"));
        assert!(production_wire.contains("send_historical_session_request"));
        assert!(source.contains("CTraderOpenApiSessionResponse::Expected"));
        assert!(source.contains("CTraderOpenApiSessionResponse::BrokerError"));
        assert!(!session.contains("Vec<String>"));
        assert!(!session.contains("Vec<Vec"));
    }

    #[test]
    fn broker_download_uses_one_persistent_session_and_checks_cancel_before_publish() {
        let source = include_str!("../../../neoethos-broker-history/src/service.rs");
        let download = source
            .split("pub(crate) fn capture_with_connector_and_publication_hook")
            .nth(1)
            .and_then(|tail| tail.split("pub fn capture_historical_generation").next())
            .expect("shared broker historical capture implementation");

        let connect = download
            .find(".connect_authenticated(")
            .expect("one persistent authenticated session");
        let page_loop = download.find("while cursor_to").expect("bounded page loop");
        let page_send = download
            .find(".next_page(HistoricalPageRequest")
            .expect("persistent page request");
        let final_cancel = download
            .rfind("ensure_not_cancelled(cancellation)")
            .expect("final cancellation gate");
        let publication_gate = download
            .find("active_fetch.begin_publication()")
            .expect("atomic cancel-to-publication transition");
        let publish = download
            .find("publish_history(")
            .expect("atomic publication");

        assert_eq!(download.matches(".connect_authenticated(").count(), 1);
        assert!(connect < page_loop && page_loop < page_send);
        assert!(
            page_send < final_cancel
                && final_cancel < publication_gate
                && publication_gate < publish
        );
        assert!(download.contains("publication_permit.run_id()"));
        assert!(download.contains("timeframe: request.timeframe"));
        assert!(!download.contains("load_historical_bars_only("));
        assert!(!download.contains("Vec<HistoricalPage>"));
    }
}
