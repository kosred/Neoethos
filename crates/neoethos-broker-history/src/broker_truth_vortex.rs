//! Exact Vortex encoders for broker-financial capture rows.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use neoethos_broker_truth::{
    BrokerFinancialTruthArtifactSourceV1, BrokerFinancialTruthBundleManifestV1,
    BrokerFinancialTruthBundleManifestV2, BrokerFinancialTruthVortexSchemaV1,
    ExactCapturedEvidencePairV1, ExactConversionLegEvidenceV1, ExactConversionLegEvidenceV2,
    ExactConversionRouteEvidenceV1, ExactConversionRouteEvidenceV2,
    ExactDealReconciliationEvidenceV2, ExactQuoteSideEvidenceV1, ExactQuoteSideEvidenceV2,
    ExactSymbolContractEvidenceV2, ImmutableVortexArtifactV1, ReviewedQuoteReplayRuleEvidenceV2,
    SynchronizedBidAskEvidenceV1, SynchronizedBidAskEvidenceV2,
};
use vortex_array::IntoArray;
use vortex_array::arrays::{PrimitiveArray, StructArray, VarBinArray};

use crate::broker_truth_capture::{
    BrokerEvidenceRowKindV2, BrokerFinancialTruthCaptureRequestV1,
    BrokerFinancialTruthCaptureRequestV2, CapturedBrokerEvidenceRowV1, CapturedBrokerEvidenceRowV2,
    ValidatedBrokerTruthCaptureV1, ValidatedBrokerTruthCaptureV2,
    ValidatedCloseDealReconciliationV2, ValidatedEvidencePairV1, ValidatedEvidencePairV2,
    ValidatedQuoteSideV1, ValidatedQuoteSideV2, ValidatedSynchronizedQuotesV1,
    ValidatedSynchronizedQuotesV2,
};

static NEXT_CAPTURE_WORK_ID: AtomicU64 = AtomicU64::new(0);

pub(crate) struct EncodedBrokerTruthBundleV1 {
    manifest: BrokerFinancialTruthBundleManifestV1,
    sources: Vec<BrokerFinancialTruthArtifactSourceV1>,
    _work_directory: CaptureWorkDirectoryV1,
}

impl EncodedBrokerTruthBundleV1 {
    pub(crate) const fn manifest(&self) -> &BrokerFinancialTruthBundleManifestV1 {
        &self.manifest
    }

    pub(crate) fn sources(&self) -> &[BrokerFinancialTruthArtifactSourceV1] {
        &self.sources
    }
}

pub(crate) fn encode_broker_truth_capture_v1(
    request: &BrokerFinancialTruthCaptureRequestV1,
    captured: &ValidatedBrokerTruthCaptureV1,
    work_parent: &Path,
) -> Result<EncodedBrokerTruthBundleV1> {
    let work_directory = CaptureWorkDirectoryV1::create(work_parent)?;
    let mut encoder = ExactVortexEncoderV1::new(work_directory.path());

    let primary_quotes = encoder.encode_synchronized_quotes(
        "primary",
        &captured.primary_quotes,
        request.window(),
    )?;

    let mut conversion_routes = Vec::with_capacity(captured.conversion_routes.len());
    for (route_index, route) in captured.conversion_routes.iter().enumerate() {
        let mut legs = Vec::with_capacity(route.legs.len());
        for (leg_index, leg) in route.legs.iter().enumerate() {
            let quotes = encoder.encode_synchronized_quotes(
                &format!("conversion-{route_index:03}-leg-{leg_index:03}"),
                &leg.quotes,
                request.window(),
            )?;
            legs.push(ExactConversionLegEvidenceV1::new(
                leg.request.from_asset_id(),
                leg.request.from_asset_name(),
                leg.request.to_asset_id(),
                leg.request.to_asset_name(),
                quotes,
            )?);
        }
        conversion_routes.push(ExactConversionRouteEvidenceV1::new(
            route.request.purpose(),
            route.request.from_asset_id(),
            route.request.from_asset_name(),
            route.request.to_asset_id(),
            route.request.to_asset_name(),
            legs,
        )?);
    }

    let symbol_contracts = encoder.encode_evidence_pair(
        "symbol-contracts",
        &captured.symbol_contracts,
        BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV1,
        BrokerFinancialTruthVortexSchemaV1::CTraderSymbolContractsDecodedV1,
    )?;
    let unrealized_pnl = encoder.encode_evidence_pair(
        "position-unrealized-pnl",
        &captured.unrealized_pnl,
        BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlResponsesRawV1,
        BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlDecodedV1,
    )?;
    let close_deal_reconciliation = encoder.encode_evidence_pair(
        "close-deal-reconciliation",
        &captured.close_deal_reconciliation,
        BrokerFinancialTruthVortexSchemaV1::CTraderDealResponsesRawV1,
        BrokerFinancialTruthVortexSchemaV1::CTraderCloseDealReconciliationDecodedV1,
    )?;
    let manifest = BrokerFinancialTruthBundleManifestV1::new(
        request.binding().clone(),
        primary_quotes,
        conversion_routes,
        symbol_contracts,
        unrealized_pnl,
        close_deal_reconciliation,
    )?;

    Ok(EncodedBrokerTruthBundleV1 {
        manifest,
        sources: encoder.sources,
        _work_directory: work_directory,
    })
}

pub(crate) struct EncodedBrokerTruthBundleV2 {
    manifest: BrokerFinancialTruthBundleManifestV2,
    sources: Vec<BrokerFinancialTruthArtifactSourceV1>,
    _work_directory: CaptureWorkDirectoryV1,
}

impl EncodedBrokerTruthBundleV2 {
    pub(crate) const fn manifest(&self) -> &BrokerFinancialTruthBundleManifestV2 {
        &self.manifest
    }

    pub(crate) fn sources(&self) -> &[BrokerFinancialTruthArtifactSourceV1] {
        &self.sources
    }
}

pub(crate) fn encode_broker_truth_capture_v2(
    request: &BrokerFinancialTruthCaptureRequestV2,
    captured: &ValidatedBrokerTruthCaptureV2,
    work_parent: &Path,
) -> Result<EncodedBrokerTruthBundleV2> {
    let work_directory = CaptureWorkDirectoryV1::create(work_parent)?;
    let mut encoder = ExactVortexEncoderV1::new(work_directory.path());

    let primary_quotes =
        encoder.encode_synchronized_quotes_v2("primary-v2", &captured.primary_quotes)?;

    let mut conversion_routes = Vec::with_capacity(captured.conversion_routes.len());
    for (route_index, route) in captured.conversion_routes.iter().enumerate() {
        let mut legs = Vec::with_capacity(route.legs.len());
        for (leg_index, leg) in route.legs.iter().enumerate() {
            let quotes = encoder.encode_synchronized_quotes_v2(
                &format!("conversion-v2-{route_index:03}-leg-{leg_index:03}"),
                &leg.quotes,
            )?;
            legs.push(ExactConversionLegEvidenceV2::new(
                leg.request.from_asset_id(),
                leg.request.from_asset_name(),
                leg.request.to_asset_id(),
                leg.request.to_asset_name(),
                quotes,
            )?);
        }
        conversion_routes.push(ExactConversionRouteEvidenceV2::new(
            route.request.purpose(),
            route.request.from_asset_id(),
            route.request.from_asset_name(),
            route.request.to_asset_id(),
            route.request.to_asset_name(),
            legs,
        )?);
    }

    let exact_symbol_contracts = encoder.encode_symbol_contracts_v2(&captured.symbol_contracts)?;
    let broker_position_unrealized_pnl = encoder.encode_evidence_pair_v2(
        "position-unrealized-pnl-v2",
        &captured.unrealized_pnl,
        BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlResponsesRawV2,
        BrokerFinancialTruthVortexSchemaV1::CTraderUnrealizedPnlDecodedV2,
    )?;
    let close_deal_reconciliation = encoder.encode_close_deal_reconciliation_v2(
        request.window(),
        &captured.close_deal_reconciliation,
    )?;
    let manifest = BrokerFinancialTruthBundleManifestV2::new(
        request.binding().clone(),
        primary_quotes,
        conversion_routes,
        exact_symbol_contracts,
        broker_position_unrealized_pnl,
        close_deal_reconciliation,
    )?;

    Ok(EncodedBrokerTruthBundleV2 {
        manifest,
        sources: encoder.sources,
        _work_directory: work_directory,
    })
}

struct ExactVortexEncoderV1<'a> {
    root: &'a Path,
    sources: Vec<BrokerFinancialTruthArtifactSourceV1>,
}

impl<'a> ExactVortexEncoderV1<'a> {
    fn new(root: &'a Path) -> Self {
        Self {
            root,
            sources: Vec::new(),
        }
    }

    fn encode_synchronized_quotes(
        &mut self,
        stem: &str,
        quotes: &ValidatedSynchronizedQuotesV1,
        window: neoethos_broker_truth::EvidenceWindowV1,
    ) -> Result<SynchronizedBidAskEvidenceV1> {
        let bid_pair = self.encode_quote_side(&format!("{stem}-bid"), &quotes.bid)?;
        let ask_pair = self.encode_quote_side(&format!("{stem}-ask"), &quotes.ask)?;
        let synchronization = self.encode_evidence_pair(
            &format!("{stem}-quote-synchronization"),
            &quotes.synchronization,
            BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV1,
            BrokerFinancialTruthVortexSchemaV1::CTraderQuoteReplayRulesDecodedV1,
        )?;
        let bid = ExactQuoteSideEvidenceV1::new(
            quotes.bid.request.side(),
            quotes.instrument.symbol_id(),
            quotes.instrument.symbol_name(),
            quotes.instrument.base_asset_id(),
            quotes.instrument.quote_asset_id(),
            window,
            window,
            bid_pair,
        )?;
        let ask = ExactQuoteSideEvidenceV1::new(
            quotes.ask.request.side(),
            quotes.instrument.symbol_id(),
            quotes.instrument.symbol_name(),
            quotes.instrument.base_asset_id(),
            quotes.instrument.quote_asset_id(),
            window,
            window,
            ask_pair,
        )?;
        Ok(SynchronizedBidAskEvidenceV1::new(
            bid,
            ask,
            synchronization,
        )?)
    }

    fn encode_quote_side(
        &mut self,
        stem: &str,
        quote: &ValidatedQuoteSideV1,
    ) -> Result<ExactCapturedEvidencePairV1> {
        let raw_name = format!("{stem}-pages-raw.vortex");
        let decoded_name = format!("{stem}-ticks-decoded.vortex");
        let raw = self.write_artifact(
            &raw_name,
            BrokerFinancialTruthVortexSchemaV1::CTraderTickPagesRawV1,
            quote.pages_newest_first.len(),
            raw_tick_pages_array(quote)?,
        )?;
        let decoded = self.write_artifact(
            &decoded_name,
            BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV1,
            quote.ticks_ascending.len(),
            decoded_ticks_array(quote)?,
        )?;
        Ok(ExactCapturedEvidencePairV1::new(raw, decoded))
    }

    fn encode_evidence_pair(
        &mut self,
        stem: &str,
        pair: &ValidatedEvidencePairV1,
        raw_schema: BrokerFinancialTruthVortexSchemaV1,
        decoded_schema: BrokerFinancialTruthVortexSchemaV1,
    ) -> Result<ExactCapturedEvidencePairV1> {
        let raw_name = format!("{stem}-raw.vortex");
        let decoded_name = format!("{stem}-decoded.vortex");
        let raw = self.write_artifact(
            &raw_name,
            raw_schema,
            pair.raw_envelopes.len(),
            broker_evidence_rows_array(&pair.raw_envelopes)?,
        )?;
        let decoded = self.write_artifact(
            &decoded_name,
            decoded_schema,
            pair.decoded_records.len(),
            broker_evidence_rows_array(&pair.decoded_records)?,
        )?;
        Ok(ExactCapturedEvidencePairV1::new(raw, decoded))
    }

    fn encode_synchronized_quotes_v2(
        &mut self,
        stem: &str,
        quotes: &ValidatedSynchronizedQuotesV2,
    ) -> Result<SynchronizedBidAskEvidenceV2> {
        let bid = self.encode_quote_side_v2(&format!("{stem}-bid"), &quotes.bid)?;
        let ask = self.encode_quote_side_v2(&format!("{stem}-ask"), &quotes.ask)?;

        let observations_name = format!("{stem}-quote-session-observations-raw.vortex");
        let observations_raw = self.write_artifact(
            &observations_name,
            BrokerFinancialTruthVortexSchemaV1::CTraderQuoteSessionObservationsRawV2,
            quotes.synchronization.evidence.raw_envelopes.len(),
            broker_evidence_rows_array_v2(&quotes.synchronization.evidence.raw_envelopes)?,
        )?;
        let rules_name = format!("{stem}-reviewed-quote-replay-rules-decoded.vortex");
        let rules_decoded = self.write_artifact(
            &rules_name,
            BrokerFinancialTruthVortexSchemaV1::CTraderReviewedQuoteReplayRulesDecodedV2,
            quotes.synchronization.evidence.decoded_records.len(),
            broker_evidence_rows_array_v2(&quotes.synchronization.evidence.decoded_records)?,
        )?;
        let replay_rule = ReviewedQuoteReplayRuleEvidenceV2::new(
            quotes.synchronization.review_identity.clone(),
            observations_raw,
            rules_decoded,
        )?;
        Ok(SynchronizedBidAskEvidenceV2::new(bid, ask, replay_rule)?)
    }

    fn encode_quote_side_v2(
        &mut self,
        stem: &str,
        quote: &ValidatedQuoteSideV2,
    ) -> Result<ExactQuoteSideEvidenceV2> {
        let raw_name = format!("{stem}-request-pages-raw.vortex");
        let raw_pages = self.write_artifact(
            &raw_name,
            BrokerFinancialTruthVortexSchemaV1::CTraderTickRequestPagesRawV2,
            quote.pages_newest_first.len(),
            raw_tick_pages_array_v2(quote)?,
        )?;
        let decoded_name = format!("{stem}-ticks-decoded.vortex");
        let decoded_ticks = self.write_artifact(
            &decoded_name,
            BrokerFinancialTruthVortexSchemaV1::CTraderTicksDecodedV2,
            quote.ticks_ascending.len(),
            decoded_ticks_array_v2(quote)?,
        )?;
        Ok(ExactQuoteSideEvidenceV2::new(
            quote.request.side(),
            quote.request.instrument().symbol_id(),
            quote.request.instrument().symbol_name(),
            quote.request.instrument().base_asset_id(),
            quote.request.instrument().quote_asset_id(),
            quote.request.window(),
            quote.request_chunks_newest_first.clone(),
            raw_pages,
            decoded_ticks,
        )?)
    }

    fn encode_evidence_pair_v2(
        &mut self,
        stem: &str,
        pair: &ValidatedEvidencePairV2,
        raw_schema: BrokerFinancialTruthVortexSchemaV1,
        decoded_schema: BrokerFinancialTruthVortexSchemaV1,
    ) -> Result<ExactCapturedEvidencePairV1> {
        let raw_name = format!("{stem}-raw.vortex");
        let decoded_name = format!("{stem}-decoded.vortex");
        let raw = self.write_artifact(
            &raw_name,
            raw_schema,
            pair.raw_envelopes.len(),
            broker_evidence_rows_array_v2(&pair.raw_envelopes)?,
        )?;
        let decoded = self.write_artifact(
            &decoded_name,
            decoded_schema,
            pair.decoded_records.len(),
            broker_evidence_rows_array_v2(&pair.decoded_records)?,
        )?;
        Ok(ExactCapturedEvidencePairV1::new(raw, decoded))
    }

    fn encode_symbol_contracts_v2(
        &mut self,
        pair: &ValidatedEvidencePairV2,
    ) -> Result<ExactSymbolContractEvidenceV2> {
        let light_raw = evidence_rows_of_kind_v2(
            &pair.raw_envelopes,
            BrokerEvidenceRowKindV2::LightSymbolResponse,
        );
        let full_raw =
            evidence_rows_of_kind_v2(&pair.raw_envelopes, BrokerEvidenceRowKindV2::SymbolResponse);
        let asset_raw = evidence_rows_of_kind_v2(
            &pair.raw_envelopes,
            BrokerEvidenceRowKindV2::AccountAssetResponse,
        );
        let trader_raw = evidence_rows_of_kind_v2(
            &pair.raw_envelopes,
            BrokerEvidenceRowKindV2::TraderAccountResponse,
        );

        let light_symbol_responses_raw = self.write_artifact(
            "light-symbol-responses-v2-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderLightSymbolResponsesRawV2,
            light_raw.len(),
            broker_evidence_rows_array_v2(&light_raw)?,
        )?;
        let full_symbol_responses_raw = self.write_artifact(
            "full-symbol-responses-v2-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderSymbolResponsesRawV2,
            full_raw.len(),
            broker_evidence_rows_array_v2(&full_raw)?,
        )?;
        let account_asset_responses_raw = self.write_artifact(
            "account-asset-responses-v2-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderAccountAssetResponsesRawV2,
            asset_raw.len(),
            broker_evidence_rows_array_v2(&asset_raw)?,
        )?;
        let trader_account_responses_raw = self.write_artifact(
            "trader-account-responses-v2-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderTraderAccountResponsesRawV2,
            trader_raw.len(),
            broker_evidence_rows_array_v2(&trader_raw)?,
        )?;
        let contracts_decoded = self.write_artifact(
            "symbol-money-contracts-v2-decoded.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderSymbolMoneyContractsDecodedV2,
            pair.decoded_records.len(),
            broker_evidence_rows_array_v2(&pair.decoded_records)?,
        )?;

        Ok(ExactSymbolContractEvidenceV2::new(
            light_symbol_responses_raw,
            full_symbol_responses_raw,
            account_asset_responses_raw,
            trader_account_responses_raw,
            contracts_decoded,
        )?)
    }

    fn encode_close_deal_reconciliation_v2(
        &mut self,
        requested_window: neoethos_broker_truth::EvidenceWindowV1,
        capture: &ValidatedCloseDealReconciliationV2,
    ) -> Result<ExactDealReconciliationEvidenceV2> {
        let reconcile_responses_raw = self.write_artifact(
            "reconcile-responses-v2-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderReconcileResponsesRawV2,
            1,
            broker_evidence_rows_array_v2(std::slice::from_ref(&capture.reconcile_raw))?,
        )?;
        let deal_pages_raw = self.write_artifact(
            "deal-pages-v2-raw.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderDealPagesRawV2,
            capture.deal_pages_newest_first.len(),
            raw_deal_pages_array_v2(capture)?,
        )?;
        let reconciliation_decoded = self.write_artifact(
            "close-deal-reconciliation-v2-decoded.vortex",
            BrokerFinancialTruthVortexSchemaV1::CTraderCloseDealReconciliationDecodedV2,
            capture.decoded_records.len(),
            broker_evidence_rows_array_v2(&capture.decoded_records)?,
        )?;
        Ok(ExactDealReconciliationEvidenceV2::new(
            requested_window,
            capture.deal_request_chunk.clone(),
            reconcile_responses_raw,
            deal_pages_raw,
            reconciliation_decoded,
        )?)
    }

    fn write_artifact(
        &mut self,
        relative_path: &str,
        schema: BrokerFinancialTruthVortexSchemaV1,
        expected_rows: usize,
        array: vortex_array::ArrayRef,
    ) -> Result<ImmutableVortexArtifactV1> {
        let expected_rows = u64::try_from(expected_rows)
            .context("broker truth Vortex row count does not fit u64")?;
        if expected_rows == 0 {
            bail!("cannot encode an empty broker truth Vortex artifact");
        }
        let path = self.root.join(relative_path);
        neoethos_data::core::vortex_io::write_vortex_array(&path, array)
            .with_context(|| format!("failed to write exact Vortex artifact {relative_path}"))?;
        let actual_rows = neoethos_data::core::vortex_io::read_vortex_row_count(&path)
            .with_context(|| format!("failed to reopen Vortex footer for {relative_path}"))?;
        if actual_rows != expected_rows {
            bail!(
                "Vortex artifact {relative_path} row count {actual_rows} differs from exact expected {expected_rows}"
            );
        }
        let artifact =
            ImmutableVortexArtifactV1::from_file(relative_path, schema, expected_rows, &path)?;
        self.sources.push(BrokerFinancialTruthArtifactSourceV1::new(
            relative_path,
            path,
        )?);
        Ok(artifact)
    }
}

fn raw_tick_pages_array(quote: &ValidatedQuoteSideV1) -> Result<vortex_array::ArrayRef> {
    let page_sequence = (0..quote.pages_newest_first.len())
        .map(|index| index as u64)
        .collect::<Vec<_>>();
    let account_ids = quote
        .pages_newest_first
        .iter()
        .map(|page| page.account_id)
        .collect::<Vec<_>>();
    let symbol_ids = quote
        .pages_newest_first
        .iter()
        .map(|page| page.symbol_id)
        .collect::<Vec<_>>();
    let sides = quote
        .pages_newest_first
        .iter()
        .map(|page| quote_side_code(page.side))
        .collect::<Vec<_>>();
    let client_msg_ids = quote
        .pages_newest_first
        .iter()
        .map(|page| page.client_msg_id.as_str())
        .collect::<Vec<_>>();
    let requested_from = quote
        .pages_newest_first
        .iter()
        .map(|page| page.requested_window.from_unix_ms_inclusive())
        .collect::<Vec<_>>();
    let requested_to = quote
        .pages_newest_first
        .iter()
        .map(|page| page.requested_window.to_unix_ms_exclusive())
        .collect::<Vec<_>>();
    let first_timestamp = quote
        .pages_newest_first
        .iter()
        .map(|page| {
            page.ticks
                .first()
                .expect("validated non-empty page")
                .timestamp_ms
        })
        .collect::<Vec<_>>();
    let last_timestamp = quote
        .pages_newest_first
        .iter()
        .map(|page| {
            page.ticks
                .last()
                .expect("validated non-empty page")
                .timestamp_ms
        })
        .collect::<Vec<_>>();
    let tick_count = quote
        .pages_newest_first
        .iter()
        .map(|page| page.ticks.len() as u64)
        .collect::<Vec<_>>();
    let has_more = quote
        .pages_newest_first
        .iter()
        .map(|page| u8::from(page.has_more))
        .collect::<Vec<_>>();
    let raw_response_json = quote
        .pages_newest_first
        .iter()
        .map(|page| page.raw_response_json.as_str())
        .collect::<Vec<_>>();

    Ok(StructArray::from_fields(&[
        (
            "page_sequence",
            PrimitiveArray::from_iter(page_sequence).into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter(account_ids).into_array(),
        ),
        (
            "symbol_id",
            PrimitiveArray::from_iter(symbol_ids).into_array(),
        ),
        ("quote_side", PrimitiveArray::from_iter(sides).into_array()),
        (
            "client_msg_id",
            VarBinArray::from(client_msg_ids).into_array(),
        ),
        (
            "requested_from_unix_ms_inclusive",
            PrimitiveArray::from_iter(requested_from).into_array(),
        ),
        (
            "requested_to_unix_ms_exclusive",
            PrimitiveArray::from_iter(requested_to).into_array(),
        ),
        (
            "first_tick_timestamp_ms",
            PrimitiveArray::from_iter(first_timestamp).into_array(),
        ),
        (
            "last_tick_timestamp_ms",
            PrimitiveArray::from_iter(last_timestamp).into_array(),
        ),
        (
            "decoded_tick_count",
            PrimitiveArray::from_iter(tick_count).into_array(),
        ),
        ("has_more", PrimitiveArray::from_iter(has_more).into_array()),
        (
            "raw_response_json",
            VarBinArray::from(raw_response_json).into_array(),
        ),
    ])
    .context("failed to construct raw cTrader tick-page Vortex table")?
    .into_array())
}

fn decoded_ticks_array(quote: &ValidatedQuoteSideV1) -> Result<vortex_array::ArrayRef> {
    let row_count = quote.ticks_ascending.len();
    let account_ids = vec![quote.request.account_id(); row_count];
    let symbol_ids = vec![quote.request.instrument().symbol_id(); row_count];
    let sides = vec![quote_side_code(quote.request.side()); row_count];
    Ok(StructArray::from_fields(&[
        (
            "page_sequence",
            PrimitiveArray::from_iter(quote.ticks_ascending.iter().map(|row| row.page_sequence))
                .into_array(),
        ),
        (
            "row_sequence_in_page",
            PrimitiveArray::from_iter(quote.ticks_ascending.iter().map(|row| row.row_sequence))
                .into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter(account_ids).into_array(),
        ),
        (
            "symbol_id",
            PrimitiveArray::from_iter(symbol_ids).into_array(),
        ),
        ("quote_side", PrimitiveArray::from_iter(sides).into_array()),
        (
            "timestamp_ms",
            PrimitiveArray::from_iter(quote.ticks_ascending.iter().map(|row| row.timestamp_ms))
                .into_array(),
        ),
        (
            "price",
            PrimitiveArray::from_iter(quote.ticks_ascending.iter().map(|row| row.price))
                .into_array(),
        ),
    ])
    .context("failed to construct decoded cTrader tick Vortex table")?
    .into_array())
}

fn broker_evidence_rows_array(
    rows: &[CapturedBrokerEvidenceRowV1],
) -> Result<vortex_array::ArrayRef> {
    let evidence_kinds = rows.iter().map(|row| row.kind.code()).collect::<Vec<_>>();
    let has_symbol_id = rows
        .iter()
        .map(|row| u8::from(row.symbol_id.is_some()))
        .collect::<Vec<_>>();
    let symbol_ids = rows
        .iter()
        .map(|row| row.symbol_id.unwrap_or(0))
        .collect::<Vec<_>>();
    let has_quote_side = rows
        .iter()
        .map(|row| u8::from(row.quote_side.is_some()))
        .collect::<Vec<_>>();
    let quote_sides = rows
        .iter()
        .map(|row| row.quote_side.map_or(0, quote_side_code))
        .collect::<Vec<_>>();
    let has_requested_window = rows
        .iter()
        .map(|row| u8::from(row.requested_window.is_some()))
        .collect::<Vec<_>>();
    let requested_from = rows
        .iter()
        .map(|row| {
            row.requested_window
                .map_or(0, |window| window.from_unix_ms_inclusive())
        })
        .collect::<Vec<_>>();
    let requested_to = rows
        .iter()
        .map(|row| {
            row.requested_window
                .map_or(0, |window| window.to_unix_ms_exclusive())
        })
        .collect::<Vec<_>>();
    let client_msg_ids = rows
        .iter()
        .map(|row| row.client_msg_id.as_str())
        .collect::<Vec<_>>();
    let payload_json = rows
        .iter()
        .map(|row| row.payload_json.as_str())
        .collect::<Vec<_>>();
    Ok(StructArray::from_fields(&[
        (
            "sequence",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.sequence)).into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.account_id)).into_array(),
        ),
        (
            "evidence_kind",
            PrimitiveArray::from_iter(evidence_kinds).into_array(),
        ),
        (
            "has_symbol_id",
            PrimitiveArray::from_iter(has_symbol_id).into_array(),
        ),
        (
            "symbol_id",
            PrimitiveArray::from_iter(symbol_ids).into_array(),
        ),
        (
            "has_quote_side",
            PrimitiveArray::from_iter(has_quote_side).into_array(),
        ),
        (
            "quote_side",
            PrimitiveArray::from_iter(quote_sides).into_array(),
        ),
        (
            "has_requested_window",
            PrimitiveArray::from_iter(has_requested_window).into_array(),
        ),
        (
            "requested_from_unix_ms_inclusive",
            PrimitiveArray::from_iter(requested_from).into_array(),
        ),
        (
            "requested_to_unix_ms_exclusive",
            PrimitiveArray::from_iter(requested_to).into_array(),
        ),
        (
            "client_msg_id",
            VarBinArray::from(client_msg_ids).into_array(),
        ),
        (
            "payload_type",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.payload_type)).into_array(),
        ),
        ("payload_json", VarBinArray::from(payload_json).into_array()),
    ])
    .context("failed to construct broker evidence Vortex table")?
    .into_array())
}

fn raw_tick_pages_array_v2(quote: &ValidatedQuoteSideV2) -> Result<vortex_array::ArrayRef> {
    let pages = &quote.pages_newest_first;
    Ok(StructArray::from_fields(&[
        (
            "chunk_sequence",
            PrimitiveArray::from_iter(pages.iter().map(|page| page.chunk_sequence)).into_array(),
        ),
        (
            "page_sequence_in_chunk",
            PrimitiveArray::from_iter(pages.iter().map(|page| page.page_sequence_in_chunk))
                .into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter(pages.iter().map(|page| page.account_id)).into_array(),
        ),
        (
            "symbol_id",
            PrimitiveArray::from_iter(pages.iter().map(|page| page.symbol_id)).into_array(),
        ),
        (
            "quote_side",
            PrimitiveArray::from_iter(pages.iter().map(|page| quote_side_code(page.side)))
                .into_array(),
        ),
        (
            "client_msg_id",
            VarBinArray::from(
                pages
                    .iter()
                    .map(|page| page.client_msg_id.as_str())
                    .collect::<Vec<_>>(),
            )
            .into_array(),
        ),
        (
            "chunk_from_unix_ms_inclusive",
            PrimitiveArray::from_iter(
                pages
                    .iter()
                    .map(|page| page.requested_chunk_window.from_unix_ms_inclusive()),
            )
            .into_array(),
        ),
        (
            "chunk_to_unix_ms_exclusive",
            PrimitiveArray::from_iter(
                pages
                    .iter()
                    .map(|page| page.requested_chunk_window.to_unix_ms_exclusive()),
            )
            .into_array(),
        ),
        (
            "page_from_unix_ms_inclusive",
            PrimitiveArray::from_iter(
                pages
                    .iter()
                    .map(|page| page.requested_page_window.from_unix_ms_inclusive()),
            )
            .into_array(),
        ),
        (
            "page_to_unix_ms_exclusive",
            PrimitiveArray::from_iter(
                pages
                    .iter()
                    .map(|page| page.requested_page_window.to_unix_ms_exclusive()),
            )
            .into_array(),
        ),
        (
            "first_tick_timestamp_ms",
            PrimitiveArray::from_iter(pages.iter().map(|page| {
                page.ticks
                    .first()
                    .expect("validated non-empty V2 quote page")
                    .timestamp_ms
            }))
            .into_array(),
        ),
        (
            "last_tick_timestamp_ms",
            PrimitiveArray::from_iter(pages.iter().map(|page| {
                page.ticks
                    .last()
                    .expect("validated non-empty V2 quote page")
                    .timestamp_ms
            }))
            .into_array(),
        ),
        (
            "decoded_tick_count",
            PrimitiveArray::from_iter(pages.iter().map(|page| page.ticks.len() as u64))
                .into_array(),
        ),
        (
            "has_more",
            PrimitiveArray::from_iter(pages.iter().map(|page| u8::from(page.has_more)))
                .into_array(),
        ),
        (
            "raw_response_json",
            VarBinArray::from(
                pages
                    .iter()
                    .map(|page| page.raw_response_json.as_str())
                    .collect::<Vec<_>>(),
            )
            .into_array(),
        ),
    ])
    .context("failed to construct V2 raw cTrader tick request-page Vortex table")?
    .into_array())
}

fn decoded_ticks_array_v2(quote: &ValidatedQuoteSideV2) -> Result<vortex_array::ArrayRef> {
    let row_count = quote.ticks_ascending.len();
    Ok(StructArray::from_fields(&[
        (
            "chunk_sequence",
            PrimitiveArray::from_iter(quote.ticks_ascending.iter().map(|row| row.chunk_sequence))
                .into_array(),
        ),
        (
            "page_sequence_in_chunk",
            PrimitiveArray::from_iter(
                quote
                    .ticks_ascending
                    .iter()
                    .map(|row| row.page_sequence_in_chunk),
            )
            .into_array(),
        ),
        (
            "row_sequence_in_page",
            PrimitiveArray::from_iter(
                quote
                    .ticks_ascending
                    .iter()
                    .map(|row| row.row_sequence_in_page),
            )
            .into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter(vec![quote.request.account_id(); row_count]).into_array(),
        ),
        (
            "symbol_id",
            PrimitiveArray::from_iter(vec![quote.request.instrument().symbol_id(); row_count])
                .into_array(),
        ),
        (
            "quote_side",
            PrimitiveArray::from_iter(vec![quote_side_code(quote.request.side()); row_count])
                .into_array(),
        ),
        (
            "timestamp_ms",
            PrimitiveArray::from_iter(quote.ticks_ascending.iter().map(|row| row.timestamp_ms))
                .into_array(),
        ),
        (
            "price",
            PrimitiveArray::from_iter(quote.ticks_ascending.iter().map(|row| row.price))
                .into_array(),
        ),
    ])
    .context("failed to construct V2 decoded cTrader tick Vortex table")?
    .into_array())
}

fn evidence_rows_of_kind_v2(
    rows: &[CapturedBrokerEvidenceRowV2],
    kind: BrokerEvidenceRowKindV2,
) -> Vec<CapturedBrokerEvidenceRowV2> {
    rows.iter()
        .filter(|row| row.kind == kind)
        .cloned()
        .collect()
}

fn broker_evidence_rows_array_v2(
    rows: &[CapturedBrokerEvidenceRowV2],
) -> Result<vortex_array::ArrayRef> {
    let evidence_kinds = rows.iter().map(|row| row.kind.code()).collect::<Vec<_>>();
    let has_symbol_id = rows
        .iter()
        .map(|row| u8::from(row.symbol_id.is_some()))
        .collect::<Vec<_>>();
    let symbol_ids = rows
        .iter()
        .map(|row| row.symbol_id.unwrap_or(0))
        .collect::<Vec<_>>();
    let has_quote_side = rows
        .iter()
        .map(|row| u8::from(row.quote_side.is_some()))
        .collect::<Vec<_>>();
    let quote_sides = rows
        .iter()
        .map(|row| row.quote_side.map_or(0, quote_side_code))
        .collect::<Vec<_>>();
    let has_requested_window = rows
        .iter()
        .map(|row| u8::from(row.requested_window.is_some()))
        .collect::<Vec<_>>();
    let requested_from = rows
        .iter()
        .map(|row| {
            row.requested_window
                .map_or(0, |window| window.from_unix_ms_inclusive())
        })
        .collect::<Vec<_>>();
    let requested_to = rows
        .iter()
        .map(|row| {
            row.requested_window
                .map_or(0, |window| window.to_unix_ms_exclusive())
        })
        .collect::<Vec<_>>();
    Ok(StructArray::from_fields(&[
        (
            "sequence",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.sequence)).into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.account_id)).into_array(),
        ),
        (
            "evidence_kind",
            PrimitiveArray::from_iter(evidence_kinds).into_array(),
        ),
        (
            "has_symbol_id",
            PrimitiveArray::from_iter(has_symbol_id).into_array(),
        ),
        (
            "symbol_id",
            PrimitiveArray::from_iter(symbol_ids).into_array(),
        ),
        (
            "has_quote_side",
            PrimitiveArray::from_iter(has_quote_side).into_array(),
        ),
        (
            "quote_side",
            PrimitiveArray::from_iter(quote_sides).into_array(),
        ),
        (
            "has_requested_window",
            PrimitiveArray::from_iter(has_requested_window).into_array(),
        ),
        (
            "requested_from_unix_ms_inclusive",
            PrimitiveArray::from_iter(requested_from).into_array(),
        ),
        (
            "requested_to_unix_ms_exclusive",
            PrimitiveArray::from_iter(requested_to).into_array(),
        ),
        (
            "client_msg_id",
            VarBinArray::from(
                rows.iter()
                    .map(|row| row.client_msg_id.as_str())
                    .collect::<Vec<_>>(),
            )
            .into_array(),
        ),
        (
            "payload_type",
            PrimitiveArray::from_iter(rows.iter().map(|row| row.payload_type)).into_array(),
        ),
        (
            "payload_json",
            VarBinArray::from(
                rows.iter()
                    .map(|row| row.payload_json.as_str())
                    .collect::<Vec<_>>(),
            )
            .into_array(),
        ),
    ])
    .context("failed to construct V2 broker evidence Vortex table")?
    .into_array())
}

fn raw_deal_pages_array_v2(
    capture: &ValidatedCloseDealReconciliationV2,
) -> Result<vortex_array::ArrayRef> {
    let pages = &capture.deal_pages_newest_first;
    let has_events = pages
        .iter()
        .map(|page| u8::from(!page.deal_execution_timestamps_ms.is_empty()))
        .collect::<Vec<_>>();
    let first_event = pages
        .iter()
        .map(|page| {
            page.deal_execution_timestamps_ms
                .first()
                .copied()
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let last_event = pages
        .iter()
        .map(|page| {
            page.deal_execution_timestamps_ms
                .last()
                .copied()
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    Ok(StructArray::from_fields(&[
        (
            "chunk_sequence",
            PrimitiveArray::from_iter(vec![0_u64; pages.len()]).into_array(),
        ),
        (
            "page_sequence_in_chunk",
            PrimitiveArray::from_iter(pages.iter().map(|page| page.page_sequence)).into_array(),
        ),
        (
            "account_id",
            PrimitiveArray::from_iter(pages.iter().map(|page| page.account_id)).into_array(),
        ),
        (
            "client_msg_id",
            VarBinArray::from(
                pages
                    .iter()
                    .map(|page| page.client_msg_id.as_str())
                    .collect::<Vec<_>>(),
            )
            .into_array(),
        ),
        (
            "chunk_from_unix_ms_inclusive",
            PrimitiveArray::from_iter(vec![
                capture
                    .deal_request_chunk
                    .requested_window()
                    .from_unix_ms_inclusive();
                pages.len()
            ])
            .into_array(),
        ),
        (
            "chunk_to_unix_ms_exclusive",
            PrimitiveArray::from_iter(vec![
                capture
                    .deal_request_chunk
                    .requested_window()
                    .to_unix_ms_exclusive();
                pages.len()
            ])
            .into_array(),
        ),
        (
            "page_from_unix_ms_inclusive",
            PrimitiveArray::from_iter(
                pages
                    .iter()
                    .map(|page| page.requested_window.from_unix_ms_inclusive()),
            )
            .into_array(),
        ),
        (
            "page_to_unix_ms_exclusive",
            PrimitiveArray::from_iter(
                pages
                    .iter()
                    .map(|page| page.requested_window.to_unix_ms_exclusive()),
            )
            .into_array(),
        ),
        (
            "max_rows",
            PrimitiveArray::from_iter(pages.iter().map(|page| page.max_rows)).into_array(),
        ),
        (
            "has_events",
            PrimitiveArray::from_iter(has_events).into_array(),
        ),
        (
            "first_deal_execution_timestamp_ms",
            PrimitiveArray::from_iter(first_event).into_array(),
        ),
        (
            "last_deal_execution_timestamp_ms",
            PrimitiveArray::from_iter(last_event).into_array(),
        ),
        (
            "decoded_deal_count",
            PrimitiveArray::from_iter(
                pages
                    .iter()
                    .map(|page| page.deal_execution_timestamps_ms.len() as u64),
            )
            .into_array(),
        ),
        (
            "has_more",
            PrimitiveArray::from_iter(pages.iter().map(|page| u8::from(page.has_more)))
                .into_array(),
        ),
        (
            "raw_response_json",
            VarBinArray::from(
                pages
                    .iter()
                    .map(|page| page.raw_response_json.as_str())
                    .collect::<Vec<_>>(),
            )
            .into_array(),
        ),
    ])
    .context("failed to construct V2 raw paged DealList Vortex table")?
    .into_array())
}

const fn quote_side_code(side: neoethos_broker_truth::QuoteSideV1) -> u8 {
    match side {
        neoethos_broker_truth::QuoteSideV1::Bid => 0,
        neoethos_broker_truth::QuoteSideV1::Ask => 1,
    }
}

struct CaptureWorkDirectoryV1 {
    path: PathBuf,
}

impl CaptureWorkDirectoryV1 {
    fn create(parent: &Path) -> Result<Self> {
        match fs::symlink_metadata(parent) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    bail!(
                        "broker truth capture work parent is not a regular directory: {}",
                        parent.display()
                    );
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(parent).with_context(|| {
                    format!(
                        "failed to create broker truth work parent {}",
                        parent.display()
                    )
                })?;
                let metadata = fs::symlink_metadata(parent)?;
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    bail!(
                        "created broker truth work parent is not a regular directory: {}",
                        parent.display()
                    );
                }
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to inspect broker truth work parent {}",
                        parent.display()
                    )
                });
            }
        }

        for _ in 0..64 {
            let nonce = NEXT_CAPTURE_WORK_ID.fetch_add(1, Ordering::Relaxed);
            let path = parent.join(format!(
                ".broker-truth-capture-{}-{nonce}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("failed to create exact capture work dir {}", path.display())
                    });
                }
            }
        }
        Err(anyhow!(
            "could not allocate a unique exact broker truth capture work directory"
        ))
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for CaptureWorkDirectoryV1 {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_dir_all(&self.path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "neoethos_broker_history::broker_truth_capture",
                path = %self.path.display(),
                error = %error,
                "failed to remove exact broker truth capture work directory"
            );
        }
    }
}
