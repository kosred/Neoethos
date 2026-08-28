use super::*;
use crate::awaiting_quote_coverage_v1::{
    AutoresearchNonterminalBoundaryErrorV1, AutoresearchRunOutcomeV1,
    QuoteCoverageProviderOutcomeV1, QuoteCoverageProviderV1, QuoteCoverageReadyV1,
    QuoteCoverageRequestV1, QuoteCoverageStateV1, QuoteCoverageWaitReasonV1,
};
use crate::journal::{NamedValues, Record};
use crate::session::{
    DatasetReceiptV1, DirectTimeframeReceiptV1, InSampleWindowV1, SessionId, SessionWriter,
};
use neoethos_data::{BarTimestampConvention, CanonicalDatasetIdentity, CanonicalTimeframe};
use std::cell::Cell;

fn dataset_receipt() -> DatasetReceiptV1 {
    let identity = CanonicalDatasetIdentity::external(
        "awaiting-quote-coverage-test",
        "EURUSD",
        CanonicalTimeframe::M1,
        BarTimestampConvention::BarOpen,
    )
    .expect("valid dataset identity");
    DatasetReceiptV1::new(
        identity.clone(),
        vec![DirectTimeframeReceiptV1 {
            dataset_identity: identity,
            manifest_schema_id: "neoethos.dataset-manifest.v1".to_owned(),
            manifest_sha256: [3; 32],
            generation_id: "generation-awaiting-coverage".to_owned(),
            vortex_sha256: [4; 32],
            row_count: 1_001,
            timestamp_start_ms: 0,
            timestamp_end_ms: 1_000,
        }],
        InSampleWindowV1 {
            start_ms: 0,
            end_exclusive_ms: 801,
        },
        OosWindow {
            start_ms: 801,
            end_ms: 1_000,
        },
    )
    .expect("valid dataset receipt")
}

fn request(candidate: &str, portfolio_digest_byte: char) -> QuoteCoverageRequestV1 {
    QuoteCoverageRequestV1::new(
        SessionId::parse("ar-awaiting-coverage").expect("valid session id"),
        OosWindow {
            start_ms: 801,
            end_ms: 1_000,
        },
        SweepId(7),
        3,
        candidate.to_owned(),
        dataset_receipt(),
        portfolio_digest_byte.to_string().repeat(64),
        "effective-search-config-v1".to_owned(),
        1,
        2,
        true,
    )
    .expect("valid bounded quote-coverage request")
}

fn opened(request: &QuoteCoverageRequestV1) -> Record {
    Record::SessionOpened {
        session_id: request.session_id().clone(),
        session_seed: 7,
        goal_hash: "fnv64:0000000000000001".to_owned(),
        judge_hash: "fnv64:0000000000000002".to_owned(),
        cost_hash: "fnv64:0000000000000003".to_owned(),
        goals: NamedValues(Vec::new()),
        scenarios_source: "test".to_owned(),
        judge: NamedValues(Vec::new()),
        costs: NamedValues(Vec::new()),
        oos_window: request.window(),
        priors: "test".to_owned(),
        budget: NamedValues(Vec::new()),
        identity_source: "test".to_owned(),
        symbol: "EURUSD".to_owned(),
        dataset_receipt: request.dataset_receipt().clone(),
    }
}

fn writer(request: &QuoteCoverageRequestV1) -> (tempfile::TempDir, SessionWriter) {
    let dir = tempfile::tempdir().expect("temporary session directory");
    let mut writer = SessionWriter::open(dir.path().join("journal.jsonl"))
        .expect("open isolated journal writer");
    writer.append(opened(request)).expect("open session");
    (dir, writer)
}

struct ReadyProvider {
    ready: QuoteCoverageReadyV1,
    calls: Cell<usize>,
}

impl QuoteCoverageProviderV1 for ReadyProvider {
    fn provide_quote_coverage_v1(
        &mut self,
        _request: &QuoteCoverageRequestV1,
    ) -> Result<QuoteCoverageProviderOutcomeV1, crate::QuoteCoverageProviderErrorV1> {
        self.calls.set(self.calls.get() + 1);
        Ok(QuoteCoverageProviderOutcomeV1::Ready(self.ready.clone()))
    }
}

#[test]
fn missing_provider_persists_nonterminal_request_without_spending_oos() {
    let request = request("fnv64:0000000000000007", 'a');
    let (_dir, mut writer) = writer(&request);

    let outcome = advance_quote_coverage_boundary_v1(&mut writer, Some(request.clone()), None)
        .expect("missing coverage is a nonterminal boundary");
    let AutoresearchRunOutcomeV1::AwaitingQuoteCoverage(awaiting) = outcome else {
        panic!("missing provider must return AwaitingQuoteCoverage")
    };
    assert_eq!(awaiting.request(), &request);
    assert_eq!(awaiting.reason(), QuoteCoverageWaitReasonV1::NoProvider);
    assert_eq!(writer.session().oos_touches_spent, 0);
    assert!(writer.session().sweeps.is_empty());
    assert!(writer.session().stopped.is_none());
    assert!(matches!(
        writer.session().quote_coverage_state(),
        Some(QuoteCoverageStateV1::Awaiting(observed)) if observed == &request
    ));
}

#[test]
fn pending_and_cancelled_resumes_are_idempotent_and_do_no_new_work() {
    for (provider, expected_reason) in [
        (
            Some(QuoteCoverageProviderOutcomeV1::Pending),
            QuoteCoverageWaitReasonV1::Pending,
        ),
        (
            Some(QuoteCoverageProviderOutcomeV1::Cancelled),
            QuoteCoverageWaitReasonV1::Cancelled,
        ),
    ] {
        struct FixedProvider(Option<QuoteCoverageProviderOutcomeV1>);
        impl QuoteCoverageProviderV1 for FixedProvider {
            fn provide_quote_coverage_v1(
                &mut self,
                _request: &QuoteCoverageRequestV1,
            ) -> Result<QuoteCoverageProviderOutcomeV1, crate::QuoteCoverageProviderErrorV1>
            {
                Ok(self.0.take().expect("one provider call"))
            }
        }

        let request = request("fnv64:0000000000000007", 'a');
        let (_dir, mut writer) = writer(&request);
        advance_quote_coverage_boundary_v1(&mut writer, Some(request.clone()), None)
            .expect("initial pause");
        let records_before = writer.journal().records().len();
        let mut provider = FixedProvider(provider);
        let outcome = advance_quote_coverage_boundary_v1(&mut writer, None, Some(&mut provider))
            .expect("retryable provider outcome");
        let AutoresearchRunOutcomeV1::AwaitingQuoteCoverage(awaiting) = outcome else {
            panic!("pending/cancelled must remain Awaiting")
        };
        assert_eq!(awaiting.request(), &request);
        assert_eq!(awaiting.reason(), expected_reason);
        assert_eq!(writer.journal().records().len(), records_before);
        assert_eq!(writer.session().oos_touches_spent, 0);
        assert!(writer.session().sweeps.is_empty());
    }
}

#[test]
fn exact_ready_resume_keeps_the_candidate_and_touch_budget_unchanged() {
    let request = request("fnv64:0000000000000007", 'a');
    let ready =
        QuoteCoverageReadyV1::new(&request, "b".repeat(64)).expect("exact coverage receipt");
    let (_dir, mut writer) = writer(&request);
    advance_quote_coverage_boundary_v1(&mut writer, Some(request.clone()), None)
        .expect("initial pause");
    let records_before = writer.journal().records().len();
    let mut provider = ReadyProvider {
        ready: ready.clone(),
        calls: Cell::new(0),
    };

    let outcome = advance_quote_coverage_boundary_v1(&mut writer, None, Some(&mut provider))
        .expect("exact ready resume");
    let AutoresearchRunOutcomeV1::QuoteCoverageReady(boundary) = outcome else {
        panic!("exact coverage must return QuoteCoverageReady")
    };
    assert_eq!(boundary.request(), &request);
    assert_eq!(boundary.coverage(), &ready);
    assert_eq!(provider.calls.get(), 1);
    assert_eq!(writer.journal().records().len(), records_before + 1);
    assert_eq!(writer.session().oos_touches_spent, 0);
    assert!(writer.session().sweeps.is_empty());

    let records_after_ready = writer.journal().records().len();
    let resumed = advance_quote_coverage_boundary_v1(&mut writer, None, None)
        .expect("ready state reopens without provider or candidate work");
    assert!(matches!(
        resumed,
        AutoresearchRunOutcomeV1::QuoteCoverageReady(_)
    ));
    assert_eq!(writer.journal().records().len(), records_after_ready);
    assert_eq!(writer.session().oos_touches_spent, 0);
}

#[test]
fn wrong_ready_identity_fails_closed_and_preserves_the_exact_waiting_request() {
    let request = request("fnv64:0000000000000007", 'a');
    let foreign = self::request("fnv64:0000000000000008", 'c');
    let foreign_ready = QuoteCoverageReadyV1::new(&foreign, "d".repeat(64))
        .expect("individually valid foreign coverage");
    let (_dir, mut writer) = writer(&request);
    advance_quote_coverage_boundary_v1(&mut writer, Some(request.clone()), None)
        .expect("initial pause");
    let records_before = writer.journal().records().len();
    let mut provider = ReadyProvider {
        ready: foreign_ready,
        calls: Cell::new(0),
    };

    let error = advance_quote_coverage_boundary_v1(&mut writer, None, Some(&mut provider))
        .expect_err("foreign Ready evidence must fail closed");
    assert!(format!("{error:#}").contains("different quote-coverage request"));
    assert_eq!(writer.journal().records().len(), records_before);
    assert_eq!(writer.session().oos_touches_spent, 0);
    assert!(matches!(
        writer.session().quote_coverage_state(),
        Some(QuoteCoverageStateV1::Awaiting(observed)) if observed == &request
    ));
}

#[test]
fn crash_reopen_is_exact_and_corrupt_duplicate_is_rejected_by_the_fold() {
    let request = request("fnv64:0000000000000007", 'a');
    let dir = tempfile::tempdir().expect("temporary session directory");
    let path = dir.path().join("journal.jsonl");
    {
        let mut writer = SessionWriter::open(&path).expect("open writer");
        writer.append(opened(&request)).expect("open session");
        advance_quote_coverage_boundary_v1(&mut writer, Some(request.clone()), None)
            .expect("persist wait before crash");
    }
    let writer = SessionWriter::open(&path).expect("reopen exact awaiting session");
    assert_eq!(writer.session().oos_touches_spent, 0);
    assert!(matches!(
        writer.session().quote_coverage_state(),
        Some(QuoteCoverageStateV1::Awaiting(observed)) if observed == &request
    ));
    drop(writer);

    let foreign = self::request("fnv64:0000000000000008", 'c');
    let records = vec![
        opened(&request),
        Record::AwaitingQuoteCoverageV1 {
            request: request.clone(),
        },
        Record::AwaitingQuoteCoverageV1 { request: foreign },
    ];
    let error = crate::Session::fold(&records)
        .expect_err("a second candidate cannot overwrite the durable wait");
    assert!(format!("{error:#}").contains("different quote-coverage request"));
}

#[test]
fn old_terminal_wrapper_fails_closed_on_both_nonterminal_boundaries() {
    let request = request("fnv64:0000000000000007", 'a');
    let awaiting =
        crate::AwaitingQuoteCoverageV1::new(request.clone(), QuoteCoverageWaitReasonV1::NoProvider);
    let ready =
        QuoteCoverageReadyV1::new(&request, "b".repeat(64)).expect("exact coverage receipt");
    let ready_boundary =
        crate::QuoteCoverageReadyBoundaryV1::new(request, ready).expect("matching ready boundary");

    for outcome in [
        AutoresearchRunOutcomeV1::AwaitingQuoteCoverage(awaiting),
        AutoresearchRunOutcomeV1::QuoteCoverageReady(ready_boundary),
    ] {
        let error = terminal_only_v1(outcome)
            .expect_err("the legacy SessionVerdict API cannot pretend a boundary is terminal");
        assert!(
            error
                .downcast_ref::<AutoresearchNonterminalBoundaryErrorV1>()
                .is_some(),
            "legacy wrapper must return its typed fail-closed error: {error:#}"
        );
    }
}

#[test]
fn request_shape_is_bounded_single_batch_and_fixed_stop_only() {
    let valid = request("fnv64:0000000000000007", 'a');
    assert!(valid.compact_json_byte_count() <= crate::MAX_QUOTE_COVERAGE_REQUEST_BYTES_V1);
    for (batch_count, gene_count, fixed_stop_only) in
        [(0, 2, true), (2, 2, true), (1, 0, true), (1, 2, false)]
    {
        let error = QuoteCoverageRequestV1::new(
            valid.session_id().clone(),
            valid.window(),
            valid.sweep(),
            valid.slot(),
            valid.candidate_config_hash().to_owned(),
            valid.dataset_receipt().clone(),
            valid.portfolio_identity_sha256().to_owned(),
            valid.effective_search_config_hash().to_owned(),
            batch_count,
            gene_count,
            fixed_stop_only,
        )
        .expect_err("unsupported V1 shape must fail before Awaiting or OOS touch");
        assert!(matches!(
            error.code(),
            crate::QuoteCoverageErrorCodeV1::UnsupportedV1Shape
        ));
    }
}
