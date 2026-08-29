//! The real executor: one search = one streaming working-set sweep on the card.
//!
//! This is the only place in the crate that reads a bar. It exists behind
//! [`super::SweepExecutor`] so the state machine — the part that has to be
//! provably resumable and provably unable to reach a broker — can be exercised
//! end to end without a GPU, a dataset or a feature cube.
//!
//! ## Two decisions made here, once, loudly
//!
//! **1. Every search is charged at the PESSIMISTIC edge of the frozen cost
//! band.** `evaluation_spread_pips` is set to `cost_band_pips.1` and
//! `evaluation_commission_per_trade` to `0.0`, so the whole round trip is the
//! pessimistic edge and `StrategyMetrics::profit_per_trade` **is**
//! `E_screen_pess` with no unit conversion anywhere. This is not the loop
//! varying a frozen field: the value is fully determined by `cost_band_pips`,
//! which no `SearchConfigDelta` can touch. It is `docs/autoresearch-loop.md` §8
//! — *"charge the COST BAND, not a point estimate"* — applied at the only place
//! where it costs nothing: a result that survives only at the optimistic edge is
//! not a result, so the loop simply never computes one.
//!
//! **2. The loaded span STOPS before the OOS window.** The dataset is sliced by
//! time at `oos_start_ms` before a single feature is built, and
//! [`super::SweepExecutor::windows`] reports the bounds it actually used so the
//! startup self-check can compare them. A leaked out-of-sample window cannot be
//! un-leaked.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(feature = "gpu-cuda")]
use neoethos_search::data_selection::CanonicalSearchInput;
use neoethos_search::discovery::DiscoveryConfig;
#[cfg(test)]
use neoethos_search::genetic::Gene;
use neoethos_search::{
    CanonicalSearchArtifactScopeV2, CanonicalSearchInputReceiptV2, CanonicalSearchRunInputV2,
    CanonicalSearchWindowRoleV1, LockedPortfolioOuterHoldoutReplaySetV1,
};

use crate::QuoteValidatedOosTouchEvidenceV1;
use crate::journal::{CostBandCounts, OosWindow};
use crate::session::{DatasetReceiptV1, DirectTimeframeReceiptV1, InSampleWindowV1, SweepId};

use super::{SearchOutcome, SearchRequest, SweepExecutor};

/// The fraction of the series, by time, held out and touched once.
const OOS_FRACTION: f64 = 0.20;

/// One search = one discovery run over a bounded working-set sweep.
pub struct StreamingSweepExecutor {
    /// Typed base frame clipped before OOS, retaining an exact consumed source
    /// segment into its immutable full-generation artifact.
    in_sample_base: neoethos_data::CanonicalOhlcvFrame,
    /// The full dataset, kept for the single out-of-sample evaluation.
    full: neoethos_data::SymbolDataset,
    #[cfg(feature = "gpu-cuda")]
    data_root: PathBuf,
    #[cfg(feature = "gpu-cuda")]
    pinned_series_receipt: neoethos_data::CanonicalDatasetSeriesReceiptV1,
    base_timeframe: String,
    higher_timeframes: Vec<String>,
    search_span_ms: (i64, i64),
    oos_window: OosWindow,
    oos_bars: usize,
    bars_per_expected_trade: f64,
    streaming_requested: bool,
    dataset_receipt: DatasetReceiptV1,
    /// Installed only by an explicit caller after the finalist is locked. The
    /// ordinary trendbar-only `run()` route leaves this absent and OOS
    /// preflight refuses before spending the touch.
    quote_validated_oos_replay: Option<LockedPortfolioOuterHoldoutReplaySetV1>,
    /// Where per-search discovery ledgers are written before their matrices are
    /// copied into the session store.
    scratch_root: PathBuf,
}

/// Owns one exact disposable discovery-ledger slot.
///
/// The session-store copies live outside this path and are never touched. The
/// guard removes only its leaf slot on normal return, error return, and panic
/// unwind. Process abort/kill cannot run `Drop`; the next retry still clears
/// this same exact slot before reuse.
struct ScratchSlotGuard {
    path: PathBuf,
}

impl ScratchSlotGuard {
    fn prepare(scratch_root: &Path, request: &SearchRequest<'_>) -> Result<Self> {
        let path = scratch_root
            .join(request.dataset_receipt.identity().as_str())
            .join(request.session_id.as_str())
            .join(request.sweep.to_string())
            .join(format!("slot_{:03}", request.slot));
        anyhow::ensure!(
            path.starts_with(scratch_root) && path != scratch_root,
            "refusing scratch ledger path outside the autoresearch scratch root: {}",
            path.display()
        );

        let guard = Self { path };
        if guard.path.exists() {
            std::fs::remove_dir_all(&guard.path).with_context(|| {
                format!(
                    "emptying the scratch ledger directory {} before {} slot {} runs. It is \
                     disposable, and a stale retry matrix would otherwise be attributed to this \
                     search.",
                    guard.path.display(),
                    request.sweep,
                    request.slot
                )
            })?;
        }
        std::fs::create_dir_all(&guard.path).with_context(|| {
            format!(
                "creating the contained scratch ledger slot {}",
                guard.path.display()
            )
        })?;
        Ok(guard)
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchSlotGuard {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.path)
            && error.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                target: "neoethos_autoresearch::streaming",
                path = %self.path.display(),
                error = %error,
                "failed to remove the exact disposable scratch slot during cleanup; the next \
                 retry will clear this same slot before reuse"
            );
        }
    }
}

impl StreamingSweepExecutor {
    /// Load the data, derive the OOS window, and bound the search span.
    pub fn resolve(
        settings: &neoethos_core::config::Settings,
        base_config: &DiscoveryConfig,
        dataset_identity: &neoethos_data::CanonicalDatasetIdentity,
    ) -> Result<Self> {
        let root = settings.system.data_dir.clone();
        let symbol = dataset_identity.symbol_name().to_owned();
        let base_timeframe = dataset_identity.timeframe().as_str().to_owned();
        anyhow::ensure!(
            base_config.evaluation_symbol.eq_ignore_ascii_case(&symbol),
            "autoresearch config symbol {} disagrees with selected canonical dataset identity symbol {symbol}",
            base_config.evaluation_symbol
        );
        anyhow::ensure!(
            base_config.timeframe_label == base_timeframe,
            "autoresearch config timeframe {} disagrees with selected canonical dataset identity timeframe {base_timeframe}",
            base_config.timeframe_label
        );

        let mut required_timeframes = vec![dataset_identity.timeframe()];
        for label in &base_config.higher_timeframes {
            let timeframe = label
                .parse::<neoethos_data::CanonicalTimeframe>()
                .with_context(|| format!("unsupported canonical higher timeframe {label}"))?;
            if !required_timeframes.contains(&timeframe) {
                required_timeframes.push(timeframe);
            }
        }

        let inventory = neoethos_data::discover_canonical_dataset_identities(&root, &symbol)
            .with_context(|| {
                format!(
                    "inventorying exact canonical dataset series {} under {}",
                    dataset_identity.to_path_component(),
                    root.display()
                )
            })?;
        for timeframe in &required_timeframes {
            let matches = inventory
                .iter()
                .filter(|candidate| {
                    candidate.timeframe() == *timeframe
                        && candidate.scope() == dataset_identity.scope()
                        && candidate.symbol_name() == dataset_identity.symbol_name()
                        && candidate.bar_timestamp_convention()
                            == dataset_identity.bar_timestamp_convention()
                })
                .count();
            anyhow::ensure!(
                matches == 1,
                "missing direct canonical timeframe {timeframe} for selected series {}; import/download required",
                dataset_identity.to_path_component()
            );
        }

        let mut selected = Vec::with_capacity(required_timeframes.len());
        for timeframe in &required_timeframes {
            let identity = inventory
                .iter()
                .find(|candidate| {
                    candidate.timeframe() == *timeframe
                        && candidate.scope() == dataset_identity.scope()
                        && candidate.symbol_name() == dataset_identity.symbol_name()
                        && candidate.bar_timestamp_convention()
                            == dataset_identity.bar_timestamp_convention()
                })
                .context("verified direct canonical timeframe disappeared from inventory")?;
            let manifest = neoethos_data::core::dataset_manifest::read_current_manifest_metadata(
                &root, identity,
            )?;
            selected.push(neoethos_data::SelectedDatasetGenerationV1::from_manifest(
                &manifest,
            )?);
        }
        let anchor = selected
            .iter()
            .find(|selected| selected.identity() == dataset_identity)
            .cloned()
            .context("exact autoresearch selection lost its anchor generation")?;
        let pinned_series_receipt =
            neoethos_data::CanonicalDatasetSeriesReceiptV1::new(anchor, selected)?;
        let pinned_series =
            neoethos_data::pin_exact_canonical_series_v1(&root, pinned_series_receipt.clone())?;

        #[cfg(feature = "gpu-cuda")]
        let full = {
            let pinned_series = std::cell::RefCell::new(Some(pinned_series));
            neoethos_search::dispatch_canonical_discovery_data_preparation_v3(
                |no_physical_gpu_admission| {
                    let pinned_series = pinned_series
                        .borrow_mut()
                        .take()
                        .context("autoresearch resolve pin was already consumed")?;
                    let dataset = pinned_series
                        .into_cpu_dataset_after_no_physical_gpu_v1(&no_physical_gpu_admission)?;
                    Ok((dataset, no_physical_gpu_admission))
                },
                |dataset, _no_physical_gpu_admission| Ok(dataset),
                || {
                    anyhow::bail!(
                        "autoresearch cannot seal the complete native Discovery workspace; refusing host OHLCV materialization on a physical GPU"
                    )
                },
                |_admitted_native_run| {
                    let _pinned_series = pinned_series
                        .borrow_mut()
                        .take()
                        .context("autoresearch resolve pin was already consumed")?;
                    anyhow::bail!(
                        "autoresearch native Data materialization is unreachable before workspace sealing"
                    )
                },
            )?
        };
        #[cfg(not(feature = "gpu-cuda"))]
        let full = pinned_series.into_cpu_dataset_without_native_adapter_v1()?;
        neoethos_data::require_direct_timeframes(&full, dataset_identity, &required_timeframes)
            .with_context(|| {
                format!(
                    "verifying direct canonical timeframes for selected series {}",
                    dataset_identity.to_path_component()
                )
            })?;

        let base = full.timeframe(&base_timeframe).ok_or_else(|| {
            anyhow::anyhow!(
                "{symbol} has no {base_timeframe} frame; available: {:?}",
                full.timeframes()
            )
        })?;
        let timestamps = base.timestamp.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "{symbol} {base_timeframe} has no timestamp column, so the out-of-sample window \
                 cannot be defined BY TIME. It will not be defined by row index instead: a row \
                 split across a data gap is not a time split, and the whole point of the window \
                 is that nothing in the search has seen it."
            )
        })?;
        if timestamps.len() < 100 {
            bail!(
                "{symbol} {base_timeframe} has only {} bars — too few to split into a search span \
                 and an out-of-sample window",
                timestamps.len()
            );
        }

        // The OOS window is the final OOS_FRACTION of the span BY TIME.
        let first = *timestamps.first().expect("non-empty");
        let last = *timestamps.last().expect("non-empty");
        let span = (last - first).max(1);
        let oos_start_ms = last - ((span as f64) * OOS_FRACTION) as i64;
        let oos_window = OosWindow {
            start_ms: oos_start_ms,
            end_ms: last,
        };
        let oos_bars = timestamps.iter().filter(|t| **t >= oos_start_ms).count();

        // The exact half-open split is `[first, oos_start_ms)` for IS and
        // `[oos_start_ms, last]` for the single OOS touch. The typed window
        // retains the pinned generation and records original row offsets.
        let in_sample_base = full
            .canonical_frame(&base_timeframe)?
            .prefix_before_timestamp_ms(oos_start_ms)
            .context("clipping the direct base generation to the in-sample prefix")?;
        let in_sample_timestamps = in_sample_base
            .ohlcv()
            .timestamp
            .as_deref()
            .context("the typed in-sample base frame has no timestamps")?;
        let search_span_ms = (
            *in_sample_timestamps
                .first()
                .context("empty in-sample prefix")?,
            *in_sample_timestamps
                .last()
                .context("empty in-sample prefix")?,
        );

        let direct_timeframes = required_timeframes
            .iter()
            .map(|timeframe| {
                full.source_artifacts
                    .get(timeframe.as_str())
                    .with_context(|| {
                        format!("direct timeframe {timeframe} lost its canonical artifact")
                    })
                    .and_then(DirectTimeframeReceiptV1::from_artifact)
            })
            .collect::<Result<Vec<_>>>()?;
        let dataset_receipt = DatasetReceiptV1::new(
            dataset_identity.clone(),
            direct_timeframes,
            InSampleWindowV1 {
                start_ms: first,
                end_exclusive_ms: oos_start_ms,
            },
            oos_window,
        )?;

        // Bars per expected trade, from the operator's own trade-rate floor and
        // the base frame's own bar spacing. Derived, so the OOS length check
        // means the same thing on M5 and on H1.
        let bars_per_day = bars_per_day(timestamps);
        let bars_per_expected_trade = if base_config.min_trades_per_day > 0.0 {
            bars_per_day / base_config.min_trades_per_day
        } else {
            bars_per_day
        };

        let scratch_root = std::env::temp_dir().join("neoethos-autoresearch");
        std::fs::create_dir_all(&scratch_root)
            .with_context(|| format!("creating the scratch root {}", scratch_root.display()))?;

        Ok(Self {
            in_sample_base,
            full,
            #[cfg(feature = "gpu-cuda")]
            data_root: root,
            #[cfg(feature = "gpu-cuda")]
            pinned_series_receipt,
            base_timeframe,
            higher_timeframes: base_config.higher_timeframes.clone(),
            search_span_ms,
            oos_window,
            oos_bars,
            bars_per_expected_trade,
            streaming_requested: true,
            dataset_receipt,
            quote_validated_oos_replay: None,
            scratch_root,
        })
    }

    /// Supply one sealed, immutable replay set for the single final OOS touch.
    /// It is consumed on evaluation and cannot be reused for another touch.
    pub fn install_quote_validated_oos_replay_v1(
        &mut self,
        replay_set: LockedPortfolioOuterHoldoutReplaySetV1,
    ) -> Result<()> {
        anyhow::ensure!(
            self.quote_validated_oos_replay.is_none(),
            "a quote-validated OOS replay set is already installed; replacement is refused"
        );
        self.quote_validated_oos_replay = Some(replay_set);
        Ok(())
    }

    /// The pessimistic-edge cost model, applied to a resolved config.
    ///
    /// Refuses rather than guessing when the band is absent: the startup
    /// self-check already aborts on a band that cannot discriminate, so reaching
    /// here without one means the check was bypassed.
    fn charge_pessimistic_edge(config: &DiscoveryConfig) -> Result<DiscoveryConfig> {
        let Some((_, pessimistic)) = config.cost_band_pips else {
            bail!(
                "no cost_band_pips on the resolved configuration, so the pessimistic edge cannot \
                 be charged. The startup self-check aborts on this; reaching a search without a \
                 band means it was bypassed."
            );
        };
        Ok(DiscoveryConfig {
            evaluation_spread_pips: pessimistic,
            evaluation_commission_per_trade: 0.0,
            ..config.clone()
        })
    }

    /// One options constructor for both the IS-only preflight and the real OOS
    /// evaluation. The preflight changes only the input time boundary; feature
    /// plan, higher-timeframe selection, and frozen normalization fit stay
    /// identical.
    fn oos_feature_build_options(&self) -> neoethos_data::FeatureBuildOptions {
        neoethos_data::FeatureBuildOptions {
            higher_tfs: self.higher_timeframes.clone(),
            normalization_training_rows: Some(0..self.in_sample_base.len()),
            ..neoethos_data::FeatureBuildOptions::default()
        }
    }
}

fn bars_per_day(timestamps: &[i64]) -> f64 {
    const MS_PER_DAY: f64 = 86_400_000.0;
    let first = *timestamps.first().unwrap_or(&0);
    let last = *timestamps.last().unwrap_or(&0);
    let days = ((last - first).max(1) as f64 / MS_PER_DAY).max(1.0);
    (timestamps.len() as f64 / days).max(1.0)
}

/// Prove that search's value-bound feature receipt names the same immutable
/// direct artifacts frozen into the autoresearch session.
///
/// The two receipts have different jobs and neither replaces the other:
/// [`DatasetReceiptV1`] owns the full session window contract, while
/// [`CanonicalSearchInputReceiptV2`] owns the exact feature plan, provenance,
/// and consumed source segments for this search call. This bridge rejects any
/// anchor, generation, manifest, or Vortex substitution before discovery.
pub(super) fn validate_search_receipt_against_dataset_receipt(
    dataset_receipt: &DatasetReceiptV1,
    search_receipt: &CanonicalSearchInputReceiptV2,
) -> Result<()> {
    let search_anchor = search_receipt
        .validate()
        .context("validating the canonical search input receipt")?;
    anyhow::ensure!(
        search_anchor == dataset_receipt.anchor_identity,
        "canonical search receipt anchor {} disagrees with frozen autoresearch anchor {}",
        search_anchor.to_path_component(),
        dataset_receipt.anchor_identity.to_path_component()
    );

    let bindings = search_receipt.source_bindings();
    anyhow::ensure!(
        bindings.len() == dataset_receipt.direct_timeframes.len(),
        "canonical search receipt carries {} direct source bindings, but frozen autoresearch receipt {} carries {} direct timeframe artifacts",
        bindings.len(),
        dataset_receipt.identity(),
        dataset_receipt.direct_timeframes.len()
    );

    for direct in &dataset_receipt.direct_timeframes {
        let direct_identity = direct.dataset_identity.to_path_component();
        let mut matching = bindings
            .iter()
            .filter(|binding| binding.dataset_identity() == direct_identity.as_str());
        let binding = matching.next().with_context(|| {
            format!("canonical search receipt omitted frozen direct artifact {direct_identity}")
        })?;
        anyhow::ensure!(
            matching.next().is_none(),
            "canonical search receipt repeats frozen direct artifact {direct_identity}"
        );
        anyhow::ensure!(
            binding.manifest_schema_id() == direct.manifest_schema_id.as_str(),
            "canonical search receipt manifest schema {} for {direct_identity} disagrees with frozen schema {}",
            binding.manifest_schema_id(),
            direct.manifest_schema_id
        );
        let expected_manifest_sha256 = sha256_hex(&direct.manifest_sha256);
        anyhow::ensure!(
            binding.manifest_sha256() == expected_manifest_sha256,
            "canonical search receipt manifest SHA-256 {} for {direct_identity} disagrees with frozen SHA-256 {expected_manifest_sha256}",
            binding.manifest_sha256()
        );
        anyhow::ensure!(
            binding.generation_id() == direct.generation_id.as_str(),
            "canonical search receipt generation {} for {direct_identity} disagrees with frozen generation {}",
            binding.generation_id(),
            direct.generation_id
        );
        let expected_vortex_sha256 = sha256_hex(&direct.vortex_sha256);
        anyhow::ensure!(
            binding.vortex_sha256() == expected_vortex_sha256,
            "canonical search receipt Vortex SHA-256 {} for {direct_identity} disagrees with frozen SHA-256 {expected_vortex_sha256}",
            binding.vortex_sha256()
        );
        for segment in binding.segments() {
            anyhow::ensure!(
                segment.row_end() <= direct.row_count
                    && segment.timestamp_start_ms() >= direct.timestamp_start_ms
                    && segment.timestamp_end_ms() <= direct.timestamp_end_ms,
                "canonical search receipt consumed a source segment outside frozen direct artifact {direct_identity}"
            );
            anyhow::ensure!(
                segment.timestamp_end_ms() < dataset_receipt.in_sample_window.end_exclusive_ms,
                "canonical search receipt for {direct_identity} reaches timestamp {} at/after frozen OOS cutoff {}",
                segment.timestamp_end_ms(),
                dataset_receipt.in_sample_window.end_exclusive_ms
            );
        }
    }
    Ok(())
}

fn sha256_hex(value: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(value.len() * 2);
    for &byte in value {
        encoded.push(DIGITS[(byte >> 4) as usize] as char);
        encoded.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    encoded
}

/// Reproduce discovery's effective config identity after its hardware-aware
/// population resolution. The proposal stamp is intentionally not used here:
/// `population_auto` can resolve to a different population for different batch
/// feature counts, and the result/ledger hash must name what actually ran.
fn effective_search_config_hash(
    config: &DiscoveryConfig,
    evaluated_window: &neoethos_search::CanonicalSearchEvaluatedWindowV1,
    effective_feature_count: usize,
) -> Result<String> {
    let resolved = resolve_population_for_exact_window(
        config,
        evaluated_window,
        effective_feature_count,
        neoethos_search::eval::gpu_submission_ceiling,
    )?;
    // Production attribution still crosses the broker-truth capability gate.
    // The pure population resolver above exists only so its window arithmetic
    // can be tested without installing a fake financial-truth provider.
    let pip_value_per_lot = resolved
        .try_evaluation_config(None)
        .context("resolving the effective search config's pip value per lot")?
        .pip_value_per_lot;
    neoethos_search::run_identity::config_hash_for(
        &resolved,
        pip_value_per_lot,
        neoethos_data::current_data_runtime_overrides().normalize_features,
    )
    .context("deriving the exact effective discovery/ledger config hash")
}

fn resolve_population_for_exact_window(
    config: &DiscoveryConfig,
    evaluated_window: &neoethos_search::CanonicalSearchEvaluatedWindowV1,
    effective_feature_count: usize,
    submission_ceiling: impl FnOnce(usize, usize) -> Option<usize>,
) -> Result<DiscoveryConfig> {
    let evaluated_rows_u64 = evaluated_window
        .row_end()
        .checked_sub(evaluated_window.row_start())
        .context("exact evaluated selection window has a reversed source-row range")?;
    let evaluated_rows = usize::try_from(evaluated_rows_u64)
        .context("exact evaluated selection row count does not fit this platform")?;
    let stage1_pct = if config.runtime_overrides.funnel_stage1_pct.is_finite() {
        config.runtime_overrides.funnel_stage1_pct.clamp(0.01, 1.0)
    } else {
        0.25
    };
    let stage1_rows = ((evaluated_rows as f64 * stage1_pct) as usize).min(evaluated_rows);
    let resolved_population = if config.population_auto {
        submission_ceiling(stage1_rows, effective_feature_count)
            .map(|fits| fits.min(16_384).max(config.population))
            .unwrap_or(config.population)
    } else {
        config.population
    };
    Ok(DiscoveryConfig {
        population: resolved_population,
        ..config.clone()
    })
}

#[derive(Debug)]
struct ProjectedPromotionBatch<'a> {
    binding: &'a super::PromotionBatchBindingV5,
    features: neoethos_data::FeatureFrame,
}

/// Project every promotion binding by its own local feature names.
///
/// This is deliberately the only production projection path used by both the
/// pre-OOS preflight and the actual OOS consumer. There is no positional or
/// canonical-index fallback: if one recorded local name cannot be reproduced,
/// the entire portfolio is refused.
fn project_promotion_batches<'a>(
    raw: &neoethos_data::FeatureFrame,
    portfolio: &'a super::PromotionPortfolio,
) -> Result<Vec<ProjectedPromotionBatch<'a>>> {
    portfolio
        .batch_bindings
        .iter()
        .map(|binding| {
            let features = neoethos_search::live_portfolio::project_features_to_effective(
                raw,
                &binding.feature_names,
            )
            .with_context(|| {
                format!(
                    "projecting promotion batch ordinal {} cursor {} onto its {} exact local \
                     feature names",
                    binding.ordinal,
                    binding.cursor,
                    binding.feature_names.len()
                )
            })?;
            Ok(ProjectedPromotionBatch { binding, features })
        })
        .collect()
}

impl SweepExecutor for StreamingSweepExecutor {
    fn describe(&self) -> String {
        format!(
            "streaming working-set sweep on {} {} (+{:?}); search span {}..{}, OOS {}..{} \
             ({} bars, touched once); every search charged at the PESSIMISTIC edge of the frozen \
             cost band",
            self.full.symbol,
            self.base_timeframe,
            self.higher_timeframes,
            self.search_span_ms.0,
            self.search_span_ms.1,
            self.oos_window.start_ms,
            self.oos_window.end_ms,
            self.oos_bars
        )
    }

    fn streaming_requested(&self) -> bool {
        self.streaming_requested
    }

    fn windows(&self) -> Result<((i64, i64), OosWindow, usize, f64)> {
        Ok((
            self.search_span_ms,
            self.oos_window,
            self.oos_bars,
            self.bars_per_expected_trade,
        ))
    }

    fn dataset_receipt(&self) -> &DatasetReceiptV1 {
        &self.dataset_receipt
    }

    fn expected_effective_search_config_hash(
        &self,
        config: &DiscoveryConfig,
        binding: &super::PromotionBatchBindingV5,
    ) -> Result<String> {
        let charged = Self::charge_pessimistic_edge(config)?;
        effective_search_config_hash(
            &charged,
            &binding.evaluated_window,
            binding.feature_names.len(),
        )
    }

    fn execute(&mut self, request: &SearchRequest<'_>) -> Result<SearchOutcome> {
        let began = std::time::Instant::now();
        let config = Self::charge_pessimistic_edge(request.config)?;
        anyhow::ensure!(
            request.dataset_receipt == &self.dataset_receipt,
            "search request dataset receipt {} disagrees with executor receipt {}",
            request.dataset_receipt.identity(),
            self.dataset_receipt.identity()
        );

        // Each search owns one exact scratch slot, so parallel searches cannot
        // overwrite one another. The RAII guard clears a stale retry before
        // use, then removes only this disposable leaf on every unwind path.
        // Durable matrices and promotion evidence are copied into the session
        // store, outside this guard's scope.
        let scratch_slot = ScratchSlotGuard::prepare(&self.scratch_root, request)?;
        let ledger_dir = scratch_slot.path();
        let scratch_manifest = ScratchLedgerManifestV1::new(request);
        let scratch_manifest_path = ledger_dir.join(SCRATCH_LEDGER_MANIFEST_FILE);
        write_json_atomically(&scratch_manifest_path, &scratch_manifest).with_context(|| {
            format!(
                "writing exact scratch attribution manifest {}",
                scratch_manifest_path.display()
            )
        })?;
        let config = DiscoveryConfig {
            discovery_ledger_enabled: true,
            discovery_ledger_cache_dir: ledger_dir.to_string_lossy().to_string(),
            ..config
        };

        let base = self.in_sample_base.ohlcv();

        let plan = neoethos_search::orchestration::StreamingPlan::streaming(0);
        #[cfg(not(feature = "gpu-cuda"))]
        let dataset = &self.full;
        let base_tf = self.base_timeframe.clone();
        let permutation = request.permutation;
        let feature_options = neoethos_data::FeatureBuildOptions {
            higher_tfs: self.higher_timeframes.clone(),
            normalization_training_rows: Some(0..base.len()),
            ..neoethos_data::FeatureBuildOptions::default()
        };
        let in_sample_end_exclusive_ms = self.dataset_receipt.in_sample_window.end_exclusive_ms;
        let exact_anchor_identity = self.dataset_receipt.anchor_identity.clone();
        let exact_dataset_receipt = &self.dataset_receipt;
        let mut last_search_receipt = None;
        let mut last_search_config_hash = None;

        #[cfg(feature = "gpu-cuda")]
        let outcome = neoethos_search::orchestration::run_prepared_streaming_working_set_v3(
            &plan,
            base.close.len(),
            |_batch| {
                neoethos_data::pin_exact_canonical_series_v1(
                    &self.data_root,
                    self.pinned_series_receipt.clone(),
                )
            },
            |batch, pinned_series, no_physical_gpu_admission| {
                let batch_dataset = pinned_series
                    .into_cpu_dataset_after_no_physical_gpu_v1(&no_physical_gpu_admission)?;
                let batch_in_sample_base = batch_dataset
                    .canonical_frame(&base_tf)?
                    .prefix_before_timestamp_ms(in_sample_end_exclusive_ms)
                    .context("clip pinned autoresearch batch base before the OOS window")?;
                let mut frame = neoethos_data::with_extended_sweep_working_set(batch, || {
                    neoethos_data::prepare_multitimeframe_features_before_with_options(
                        &batch_dataset,
                        &base_tf,
                        &feature_options,
                        in_sample_end_exclusive_ms,
                    )
                })?;
                if let Some(permutation) = permutation {
                    apply_shuffle_control(&permutation, &mut frame).context(
                        "applying the shuffle control's permutation to the feature block",
                    )?;
                }
                let input = CanonicalSearchInput::from_prepared_canonical_frame(
                    exact_anchor_identity.clone(),
                    batch_in_sample_base,
                    frame,
                )?;
                let receipt = input
                    .receipt()
                    .context("seal prepared autoresearch batch receipt")?;
                validate_search_receipt_against_dataset_receipt(exact_dataset_receipt, &receipt)?;
                Ok((input, no_physical_gpu_admission))
            },
            |_batch| {
                anyhow::bail!(
                    "autoresearch cannot seal the complete native Discovery workspace yet; refusing host feature materialization on a physical GPU"
                )
            },
            |_batch, _pinned_series, _admitted_native_run| {
                anyhow::bail!(
                    "autoresearch native Data materialization is unreachable before workspace sealing"
                )
            },
            |prepared| {
                let result =
                    neoethos_search::run_prepared_canonical_discovery_with_holdout_and_progress_v3(
                        prepared,
                        &config,
                        neoethos_search::PropFirmRiskRules::default(),
                        |_| {},
                    )?;
                let expected_effective = effective_search_config_hash(
                    &config,
                    result.selection_scope()?.evaluated_window(),
                    result.effective_feature_names.len(),
                )?;
                anyhow::ensure!(
                    result.search_config_hash == expected_effective,
                    "discovery result effective config {} disagrees with the exact request/ledger config {expected_effective}; refusing the batch before promotion evidence is written",
                    result.search_config_hash
                );
                last_search_receipt = Some(result.search_input_receipt.clone());
                last_search_config_hash = Some(result.search_config_hash.clone());
                Ok(result)
            },
        );

        #[cfg(not(feature = "gpu-cuda"))]
        let outcome = neoethos_search::orchestration::run_streaming_working_set(
            &plan,
            base.close.len(),
            // The ONLY sanctioned build entry point: it installs the batch as
            // the working set and restores the previous one afterwards, even on
            // panic.
            |batch| {
                let mut frame = neoethos_data::with_extended_sweep_working_set(batch, || {
                    neoethos_data::prepare_multitimeframe_features_before_with_options(
                        dataset,
                        &base_tf,
                        &feature_options,
                        in_sample_end_exclusive_ms,
                    )
                })?;
                // THE SHUFFLE CONTROL, applied at the only correct point: after
                // the features exist and before the search sees them. Prices,
                // labels, costs, exit geometry, gene encoding and the GA seed
                // are untouched — only the relationship between the features
                // and the future is destroyed.
                if let Some(permutation) = permutation {
                    apply_shuffle_control(&permutation, &mut frame).context(
                        "applying the shuffle control's permutation to the feature block",
                    )?;
                }
                Ok(frame)
            },
            |features| {
                let search_receipt = CanonicalSearchInputReceiptV2::from_feature_frame(
                    &exact_anchor_identity,
                    features,
                )
                .context("binding the feature frame to its exact canonical search receipt")?;
                validate_search_receipt_against_dataset_receipt(
                    exact_dataset_receipt,
                    &search_receipt,
                )?;
                let run_input =
                    CanonicalSearchRunInputV2::new(search_receipt, features, &self.in_sample_base)
                        .context(
                            "binding exact canonical search provenance to the typed OHLCV artifact",
                        )?;
                let result = neoethos_search::run_discovery_cycle_with_holdout(
                    &run_input,
                    &config,
                    neoethos_search::PropFirmRiskRules::default(),
                )?;
                let expected_effective = effective_search_config_hash(
                    &config,
                    result.selection_scope()?.evaluated_window(),
                    result.effective_feature_names.len(),
                )?;
                anyhow::ensure!(
                    result.search_config_hash == expected_effective,
                    "discovery result effective config {} disagrees with the exact request/ledger config {expected_effective}; refusing the batch before promotion evidence is written",
                    result.search_config_hash
                );
                last_search_receipt = Some(result.search_input_receipt.clone());
                last_search_config_hash = Some(result.search_config_hash.clone());
                Ok(result)
            },
        );

        let wall_ms = began.elapsed().as_millis() as u64;

        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(err) => {
                // A per-SEARCH failure. Named, counted, and the sweep continues:
                // one bad region must not throw away the regions already
                // searched.
                return Ok(SearchOutcome {
                    slot: request.slot,
                    config_hash: request.config_hash.to_string(),
                    trials_offered: 0,
                    statistics: neoethos_search::deflated::TrialStatisticsReport::unreadable(
                        format!("the search itself failed: {err:#}"),
                    ),
                    // `discriminates: false` on a failed search, so S3 refuses
                    // it. A failure must never inherit a passing band verdict.
                    cost_band: CostBandCounts::default(),
                    rejections: Vec::new(),
                    survivors: 0,
                    e_screen_pess: None,
                    n_trades: 0,
                    champion_returns: Vec::new(),
                    champion_period_keys: Vec::new(),
                    champion_strategy_id: String::new(),
                    streamed: false,
                    batch_columns: 0,
                    next_cursor: 0,
                    wall_ms,
                    error: Some(format!("{err:#}")),
                });
            }
        };

        // Tie the champion to the exact batch receipt that produced its local
        // search result. In a streaming sweep every working set has a distinct
        // feature-plan/provenance receipt, so selecting a receipt by position
        // after selecting metrics would be an identity swap.
        let mut best: Option<(
            &neoethos_search::quality::StrategyMetrics,
            &CanonicalSearchInputReceiptV2,
            &str,
        )> = None;
        let mut survivors = 0usize;
        for batch in &outcome.batches {
            for metrics in &batch.result.quality_metrics {
                survivors += 1;
                if best.is_none_or(|(current, _, _)| {
                    metrics.profit_per_trade > current.profit_per_trade
                }) {
                    best = Some((
                        metrics,
                        &batch.result.search_input_receipt,
                        batch.result.search_config_hash.as_str(),
                    ));
                }
            }
        }
        let (evidence_receipt, evidence_config_hash) = best
            .map(|(_, receipt, config_hash)| (receipt, config_hash))
            .or_else(|| {
                last_search_receipt
                    .as_ref()
                    .zip(last_search_config_hash.as_deref())
            })
            .context(
                "the streaming search completed without retaining an exact canonical search receipt/config binding",
            )?;

        // ── S5 COLLECT ──────────────────────────────────────────────────────
        //
        // ATTRIBUTION FIRST. Everything below is read back out of
        // `discovery_ledger_cache_dir`, and until this check existed it was read
        // back BY PATH ALONE. `TrialReturnsManifest::config_hash` carries the
        // identity of the run that wrote the matrix and its doc comment names
        // this exact attack; the loop simply never read it.
        //
        // The full autoresearch attribution manifest above already binds the
        // receipt, session and slot. This writer-owned trial manifest is a
        // second independent stamp binding the search configuration itself.
        //
        // `None` is NOT a match. It means the writer had no stamp, which is
        // "cannot be attributed" — and an unattributable matrix is exactly as
        // unusable as a foreign one.
        validate_scratch_manifest(&scratch_manifest_path, &scratch_manifest)?;
        let manifest = neoethos_search::trial_returns::load_manifest(
            &config.discovery_ledger_cache_dir,
            &config.evaluation_symbol,
            &config.timeframe_label,
            evidence_receipt,
            evidence_config_hash,
        )
        .context("loading the exact receipt/config-bound trial-returns manifest")?;
        let receipt_bound_binary_path = neoethos_search::trial_returns::binary_path(
            &config.discovery_ledger_cache_dir,
            evidence_receipt,
            evidence_config_hash,
        )
        .context("resolving the exact receipt/config-bound trial-returns binary")?;
        let attribution = attribute_manifest(
            manifest
                .as_ref()
                .map(|manifest| Some(manifest.config_hash.as_str())),
            evidence_config_hash,
            &config.discovery_ledger_cache_dir,
        );

        // `trials_offered` is the honest denominator the writer recorded — every
        // candidate the screen was offered, not the survivors. It is read from
        // the manifest rather than counted here, so the loop's N and the DSR's N
        // are the same number from the same source. An unattributable manifest
        // contributes ZERO: its number belongs to some other run, and carrying
        // it would put another session's trials into this session's N.
        let trials_offered = match (&attribution, &manifest) {
            (Ok(()), Some(m)) => m.trials_offered,
            _ => 0,
        };

        let read = match &attribution {
            // NOT read at all. Reading it would also copy it into the session
            // store, where `pbo_session` and the promotion path both read it
            // back — which is how a foreign matrix would outlive the scratch
            // directory that explains it.
            Err(why) => Err(anyhow::anyhow!("{why}")),
            Ok(()) => neoethos_search::deflated::read_matrix(
                &config.discovery_ledger_cache_dir,
                &config.evaluation_symbol,
                &config.timeframe_label,
                evidence_receipt,
                evidence_config_hash,
            ),
        };

        let statistics = match read {
            Ok(matrix) => {
                // Copy the binary into the session store BEFORE the scratch dir
                // is reused, so the promotion candidate's evidence survives.
                //
                // THE RESULT IS CHECKED. It used to be `let _ = …`, which is
                // the crate's `unused_must_use = "deny"` being defeated by hand
                // on the one line whose comment claims the evidence survives:
                // scratch is disposable, so a copy that silently did not happen
                // is a matrix that may be gone by the time anything reads it —
                // and `pbo_session` and the promotion path both read it. A failure here is a disk
                // fault, not a research result, so it fails the SWEEP loudly
                // rather than producing a row whose evidence quietly is not
                // there.
                if let Some(parent) = request.trial_returns_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::copy(&receipt_bound_binary_path, &request.trial_returns_path).with_context(|| {
                    format!(
                        "copying {} slot {}'s trial-returns matrix from {} into the session store \
                         at {}. The scratch directory is disposable, so a matrix that is not \
                         copied is not durable evidence — and \
                         the session champion matrix, pbo_session and the promotion path all read \
                         it back from the session store.",
                        request.sweep,
                        request.slot,
                        receipt_bound_binary_path.display(),
                        request.trial_returns_path.display()
                    )
                })?;
                neoethos_search::deflated::analyse_matrix(&matrix, trials_offered)
            }
            Err(err) => {
                neoethos_search::deflated::TrialStatisticsReport::unreadable(format!("{err:#}"))
            }
        };

        let rejections = outcome
            .batches
            .iter()
            .filter_map(|b| b.result.funnel_profile.as_ref())
            .flat_map(|profile| {
                profile
                    .stages
                    .iter()
                    .flat_map(|stage| stage.top_reasons.iter().cloned())
            })
            .fold(HashMap::<String, usize>::new(), |mut acc, (name, count)| {
                *acc.entry(name).or_default() += count;
                acc
            })
            .into_iter()
            .collect::<Vec<_>>();

        // The champion's return series is read out of the SAME directory, so it
        // is governed by the SAME attribution. A series that cannot be
        // attributed must not reach the session champion matrix: it is the
        // series `pbo_session` is computed from, and a foreign one there would
        // move the number that gates every promotion.
        let (champion_returns, champion_period_keys, champion_strategy_id) =
            match (&attribution, best) {
                (Ok(()), Some((metrics, _, _))) => {
                    champion_series(&config, evidence_receipt, evidence_config_hash, metrics)
                }
                (Err(_), Some((metrics, _, _))) => {
                    (Vec::new(), Vec::new(), metrics.strategy_id.clone())
                }
                (_, None) => (Vec::new(), Vec::new(), String::new()),
            };

        // ── S5 COLLECT: THE PROMOTION EVIDENCE ──────────────────────────────
        //
        // The genes this search selected, with the names their indices address,
        // written into the SESSION STORE under this slot's `config_hash`.
        //
        // It is written HERE — by the search, at the moment the genes exist —
        // and not reconstructed later, because later there is nothing to
        // reconstruct from: scratch is disposable and a retry can replace it.
        // A promotion candidate can be a sweep that ran
        // days earlier and is resumed into, and the one out-of-sample touch is
        // not a thing to be attempted on evidence that may or may not still be
        // on disk.
        //
        // A search that selected nothing writes NOTHING, so "no artifact" and
        // "an artifact that says nothing survived" are the same observable and
        // the promotion path refuses on both rather than evaluating an empty
        // portfolio and reporting its zero trades as a result.
        let genes_persisted = persist_promotion_evidence(request, &outcome)?;
        tracing::debug!(
            target: "neoethos_autoresearch::streaming",
            sweep = %request.sweep,
            slot = request.slot,
            config_hash = %request.config_hash,
            genes = genes_persisted,
            path = %request.promotion_evidence_path.display(),
            "promotion evidence written"
        );

        let band_discriminates = neoethos_search::discovery::cost_band_discriminates(
            request.config.cost_band_pips,
            neoethos_search::run_identity::cost_pips_round_trip(
                request.config.evaluation_spread_pips,
                request.config.evaluation_commission_per_trade,
                request
                    .config
                    .try_evaluation_config(None)?
                    .pip_value_per_lot,
            ),
        );
        let cost_band = aggregate_cost_band_censuses_v1(
            outcome
                .batches
                .iter()
                .map(|batch| &batch.result.cost_band_census),
            band_discriminates,
        )?;

        Ok(SearchOutcome {
            slot: request.slot,
            config_hash: request.config_hash.to_string(),
            trials_offered,
            statistics,
            // These are the Search-owned measured totals, aggregated across
            // the exact streaming batches. Zero/unmeasured stays a refusal;
            // only an actual `survives` count can clear Stage-1 S3.
            cost_band,
            rejections,
            survivors,
            e_screen_pess: best.map(|(metrics, _, _)| metrics.profit_per_trade),
            n_trades: best
                .map(|(metrics, _, _)| metrics.total_trades)
                .unwrap_or(0),
            champion_returns,
            champion_period_keys,
            champion_strategy_id,
            streamed: outcome.streamed,
            batch_columns: outcome.batch_columns,
            next_cursor: outcome.next_cursor,
            wall_ms,
            error: None,
        })
    }

    /// Everything this executor can refuse WITHOUT reading an out-of-sample bar.
    ///
    /// The runner calls this BEFORE journalling `OosTouchSpent`, so each of
    /// these refusals leaves the single window unspent and a later resume — with
    /// adaptive stops off, or with the frame restored — can still use it. Every
    /// check here is repeated inside [`Self::evaluate_oos`], which is what makes
    /// skipping the preflight expensive rather than dangerous.
    fn oos_preflight(&self, portfolio: &super::PromotionPortfolio) -> Result<()> {
        anyhow::ensure!(
            portfolio.dataset_receipt == self.dataset_receipt,
            "promotion portfolio receipt {} disagrees with executor receipt {}",
            portfolio.dataset_receipt.identity(),
            self.dataset_receipt.identity()
        );
        anyhow::ensure!(
            portfolio.batch_bindings.len() == 1,
            "quote-validated OOS V1 accepts exactly one immutable finalist batch binding; multi-batch replay requires a separately versioned contract"
        );
        // The adaptive-stop hazard, refused rather than approximated.
        //
        // When adaptive stops are installed, a gene's effective SL is
        // `stop_vol_mult x` the dataset's per-bar volatility rather than its own
        // `sl_pips`, and `neoethos-search` deliberately keeps the resolver
        // private (`GeneEvalSettingsResolver`) so no caller can screen a
        // DIFFERENT strategy from the one that was scored — measured, the
        // divergence was 30,331 trades against 1,727 on one signal. There is no
        // public entry point that resolves them from outside the crate, so
        // rather than reconstruct the rule here and quietly evaluate a
        // different strategy, this refuses and names what is missing.
        //
        // It refuses in the PREFLIGHT because it is a property of the build and
        // not of the candidate: it would have been true before the touch was
        // journalled as spent, and spending the window on a call that cannot
        // return is how an irreplaceable resource gets consumed by a bug.
        if neoethos_search::stop_target::adaptive_stops_enabled() {
            bail!(
                "adaptive stops are installed, and there is no PUBLIC neoethos-search entry point \
                 that resolves a gene's effective stop from outside the crate \
                 (`GeneEvalSettingsResolver` is pub(crate)). Reconstructing the rule here would \
                 evaluate a DIFFERENT strategy from the one that was scored — measured at 30,331 \
                 trades against 1,727 on one signal — and it would do so on the one window this \
                 whole design exists to protect. REQUIRED ELSEWHERE: a public \
                 `neoethos_search::evaluate_portfolio_on_window(genes, features, ohlcv, config, \
                 from_ms, to_ms) -> Vec<Trade>` that routes through the same resolver the GA \
                 scored with. Until it exists, promotion is refused rather than approximated."
            );
        }
        if portfolio.batch_bindings.is_empty() || portfolio.gene_count == 0 {
            bail!(
                "{} slot {} carries no exact batch-bound genes, so there is nothing to evaluate out of sample",
                portfolio.sweep,
                portfolio.slot
            );
        }
        if portfolio.batch_bindings.iter().any(|binding| {
            binding.genes.iter().any(|tagged| {
                !tagged.gene.stop_vol_mult.is_finite() || tagged.gene.stop_vol_mult != 0.0
            })
        }) {
            bail!(
                "quote-coverage V1 accepts fixed-stop finalists only; every gene must carry finite stop_vol_mult=0 before a request can be persisted"
            );
        }
        let base = self
            .full
            .timeframe(&self.base_timeframe)
            .ok_or_else(|| anyhow::anyhow!("the dataset lost its {} frame", self.base_timeframe))?;
        let timestamps = base
            .timestamp
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("the base frame has no timestamp column"))?;
        if !timestamps.iter().any(|t| *t >= self.oos_window.start_ms) {
            bail!(
                "no bar falls inside the out-of-sample window {}..{}",
                self.oos_window.start_ms,
                self.oos_window.end_ms
            );
        }

        // Reproduce only the in-sample prefix before spending the touch. The
        // `_before_` entry point clips every direct timeframe before feature
        // construction, and the frozen normalization range also ends in IS;
        // no OOS price/feature value participates in this projection check.
        let options = self.oos_feature_build_options();
        let preflight_raw = neoethos_data::prepare_multitimeframe_features_before_with_options(
            &self.full,
            &self.base_timeframe,
            &options,
            self.oos_window.start_ms,
        )
        .context(
            "building the strict in-sample-only feature frame for promotion projection preflight",
        )?;
        anyhow::ensure!(
            preflight_raw
                .timestamps
                .iter()
                .all(|timestamp| *timestamp < self.oos_window.start_ms),
            "promotion projection preflight produced a feature timestamp at/after OOS start {}; \
             refusing before the touch is spent",
            self.oos_window.start_ms
        );
        project_promotion_batches(&preflight_raw, portfolio).context(
            "promotion batch projection failed on the strict in-sample prefix; the OOS window is \
             NOT spent",
        )?;
        Ok(())
    }

    fn evaluate_oos(
        &mut self,
        sweep: SweepId,
        slot: usize,
        config: &DiscoveryConfig,
        portfolio: &super::PromotionPortfolio,
    ) -> Result<QuoteValidatedOosTouchEvidenceV1> {
        // Re-asserted rather than assumed: the runner calls this first, and a
        // check that only runs when somebody remembers to call it is not a
        // check.
        self.oos_preflight(portfolio)?;
        if portfolio.sweep != sweep || portfolio.slot != slot {
            bail!(
                "the evidence handed to the out-of-sample evaluation says it is {} slot {}, but \
                 the touch is being spent on {sweep} slot {slot}. Refusing to evaluate one \
                 configuration's genes under another's name.",
                portfolio.sweep,
                portfolio.slot
            );
        }

        let config = Self::charge_pessimistic_edge(config)?;

        // Features are built over the FULL series on purpose: every indicator
        // here is causal, so a value at bar `i` is a function of bars `<= i`
        // only. Building them over the whole series and then reading ONLY the
        // out-of-sample tail leaks nothing forward, and it gives the tail the
        // same warm-up the search had — which slicing first would not.
        // The exact same feature plan/options used by the IS-only preflight.
        // Normalization, when enabled, is fitted only on the exact IS base
        // rows. OOS rows are transformed by that frozen fit, never included in
        // its estimation.
        let options = self.oos_feature_build_options();
        let raw = neoethos_data::prepare_multitimeframe_features_with_options(
            &self.full,
            &self.base_timeframe,
            &options,
        )?;

        // THE GENES ARE PROJECTED BY BATCH-LOCAL NAME, NEVER BY POSITION.
        //
        // A gene's `indices` are positions into the feature list its search
        // actually used — after discovery's prefilter and after the streaming
        // loop's canonical remap. They are NOT positions into whatever
        // `prepare_multitimeframe_features` builds here, and the two lists agree
        // only by accident. Reading column 47 of the wrong list evaluates a
        // different strategy and calls the answer out of sample, which is the
        // one error this window cannot survive — so the projection is by name,
        // through the same `project_features_to_effective` the live path uses,
        // and a name that is absent is a REFUSAL.
        let mut projected_batches = project_promotion_batches(&raw, portfolio)?;
        anyhow::ensure!(
            projected_batches.len() == 1,
            "quote-validated OOS V1 requires exactly one PromotionPortfolio batch"
        );
        let projected = projected_batches
            .pop()
            .ok_or_else(|| anyhow::anyhow!("the PromotionPortfolio has no projected batch"))?;
        let binding = projected.binding;
        let expected_selection_config_hash = effective_search_config_hash(
            &config,
            &binding.evaluated_window,
            binding.feature_names.len(),
        )?;
        anyhow::ensure!(
            binding.search_config_hash == expected_selection_config_hash,
            "batch ordinal {} cursor {} carries effective search config {} but OOS reconstructed {expected_selection_config_hash}",
            binding.ordinal,
            binding.cursor,
            binding.search_config_hash
        );
        let features = projected.features;
        let full_base = self.full.canonical_frame(&self.base_timeframe)?;
        let timestamps = full_base
            .ohlcv()
            .timestamp
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("the canonical base frame has no timestamps"))?;
        let oos_start = timestamps
            .iter()
            .position(|timestamp| *timestamp >= self.oos_window.start_ms)
            .ok_or_else(|| anyhow::anyhow!("the canonical base frame has no OOS start row"))?;
        let oos_end = timestamps.partition_point(|timestamp| *timestamp <= self.oos_window.end_ms);
        anyhow::ensure!(oos_start < oos_end, "the canonical OOS row range is empty");

        let oos_search_receipt = CanonicalSearchInputReceiptV2::from_feature_frame(
            &self.dataset_receipt.anchor_identity,
            &features,
        )
        .context("binding the full causal OOS feature frame to its exact canonical receipt")?;
        validate_search_receipt_against_dataset_receipt(
            &self.dataset_receipt,
            &oos_search_receipt,
        )?;
        let oos_run_input =
            CanonicalSearchRunInputV2::new(oos_search_receipt, &features, &full_base)
                .context("binding the final OOS feature frame to canonical trendbars")?;
        let oos_scope = CanonicalSearchArtifactScopeV2::from_run_input_range(
            CanonicalSearchWindowRoleV1::Holdout,
            &oos_run_input,
            oos_start..oos_end,
        )
        .context("binding the exact final OOS canonical-bar window")?;
        let oos_effective_search_config_hash = effective_search_config_hash(
            &config,
            oos_scope.evaluated_window(),
            binding.feature_names.len(),
        )?;
        let ordered_signals = binding
            .genes
            .iter()
            .map(|tagged| {
                neoethos_search::genetic::signals_for_gene(&features, &tagged.gene)
                    .context("building exact batch-bound OOS signals_for_gene")
            })
            .collect::<Result<Vec<_>>>()?;
        let promotion_portfolio_sha256 =
            neoethos_search::canonical_locked_portfolio_identity_sha256_v1(portfolio)
                .map_err(anyhow::Error::new)?;
        let expected_canonical_search_input_receipt_sha256 =
            oos_scope.receipt().identity_sha256()?;
        let expected_holdout_scope_identity_sha256 = oos_scope.identity_sha256()?;
        let evaluation = config.try_evaluation_config(None)?;
        let replay_set = self.quote_validated_oos_replay.take().ok_or_else(|| {
            anyhow::anyhow!("the preflight-approved sealed quote replay set disappeared")
        })?;
        let quote_validated_outer_holdout =
            neoethos_search::evaluate_locked_portfolio_outer_holdout_v1(
                portfolio,
                &ordered_signals,
                &oos_effective_search_config_hash,
                &oos_scope,
                config.initial_balance,
                evaluation.pip_value_per_lot,
                replay_set,
            )
            .map_err(anyhow::Error::new)?;
        crate::evaluate_quote_validated_oos_touch_v1(
            &portfolio.session_id,
            sweep,
            slot,
            &portfolio.config_hash,
            &self.dataset_receipt,
            self.oos_window,
            &promotion_portfolio_sha256,
            &expected_canonical_search_input_receipt_sha256,
            &oos_effective_search_config_hash,
            &expected_holdout_scope_identity_sha256,
            quote_validated_outer_holdout,
            config.initial_balance,
        )
        .map_err(anyhow::Error::new)
    }
}

fn aggregate_cost_band_censuses_v1<'a>(
    censuses: impl IntoIterator<Item = &'a neoethos_search::discovery::CostBandCensus>,
    discriminates: bool,
) -> Result<CostBandCounts> {
    let mut aggregate = neoethos_search::discovery::CostBandCensus::default();
    for census in censuses {
        aggregate.survives = aggregate
            .survives
            .checked_add(census.survives)
            .context("aggregating measured cost-band survivors")?;
        aggregate.optimistic_edge_only = aggregate
            .optimistic_edge_only
            .checked_add(census.optimistic_edge_only)
            .context("aggregating optimistic-edge-only cost-band candidates")?;
        aggregate.fails = aggregate
            .fails
            .checked_add(census.fails)
            .context("aggregating failed cost-band candidates")?;
        aggregate.unmeasured = aggregate
            .unmeasured
            .checked_add(census.unmeasured)
            .context("aggregating unmeasured cost-band candidates")?;
        aggregate.not_discriminating = aggregate
            .not_discriminating
            .checked_add(census.not_discriminating)
            .context("aggregating non-discriminating cost-band candidates")?;
    }
    Ok(CostBandCounts::from_census(&aggregate, discriminates))
}

const SCRATCH_LEDGER_MANIFEST_SCHEMA: &str = "neoethos.autoresearch.scratch-ledger-manifest.v1";
const SCRATCH_LEDGER_MANIFEST_FILE: &str = "autoresearch-attribution.json";

/// Full attribution for one disposable search ledger. The directory is keyed
/// by receipt digest + session id, while this manifest exact-compares the full
/// receipt so a digest/path collision can only refuse, never cross-adopt data.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct ScratchLedgerManifestV1 {
    schema: String,
    session_id: crate::session::SessionId,
    dataset_receipt: DatasetReceiptV1,
    sweep: SweepId,
    slot: usize,
    config_hash: String,
}

impl ScratchLedgerManifestV1 {
    fn new(request: &SearchRequest<'_>) -> Self {
        Self {
            schema: SCRATCH_LEDGER_MANIFEST_SCHEMA.to_owned(),
            session_id: (*request.session_id).clone(),
            dataset_receipt: (*request.dataset_receipt).clone(),
            sweep: request.sweep,
            slot: request.slot,
            config_hash: request.config_hash.to_owned(),
        }
    }
}

fn validate_scratch_manifest(
    path: &std::path::Path,
    expected: &ScratchLedgerManifestV1,
) -> Result<()> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading scratch attribution manifest {}", path.display()))?;
    let observed: ScratchLedgerManifestV1 = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing scratch attribution manifest {}", path.display()))?;
    anyhow::ensure!(
        observed == *expected,
        "scratch attribution manifest {} does not exactly name session {}, receipt {}, {} slot {}, and config {}",
        path.display(),
        expected.session_id,
        expected.dataset_receipt.identity(),
        expected.sweep,
        expected.slot,
        expected.config_hash
    );
    Ok(())
}

/// Can the ledger under `cache_dir` be attributed to the search that is asking?
///
/// The manifest is keyed only on `{SYMBOL}_{TF}.trial_returns.json` — a path,
/// not an identity — and `TrialReturnsManifest::config_hash` is the identity
/// that goes with it. Pure, so the decision is testable without a card, a
/// dataset or a feature cube.
///
/// `stamp` is `None` when no manifest is present at all, and `Some(None)` when
/// one is present but carries no `config_hash`.
///
/// **`Some(None)` is not a match.** It means the writer had no stamp, which is
/// "cannot be attributed"; an unattributable matrix is exactly as unusable as
/// a foreign one, and treating it as a match is how another configuration's
/// numbers would enter this one.
fn attribute_manifest(
    stamp: Option<Option<&str>>,
    expected_config_hash: &str,
    cache_dir: &str,
) -> Result<(), String> {
    let Some(stamp) = stamp else {
        return Err(format!(
            "no trial-returns manifest under {cache_dir}, so nothing read back from that \
             directory can be attributed to this search. DSR and PBO are NOT computable and are \
             refused rather than computed from whatever the directory happens to hold."
        ));
    };
    match stamp {
        Some(hash) if hash == expected_config_hash => Ok(()),
        Some(other) => Err(format!(
            "the trial-returns matrix under {cache_dir} was written by configuration {other}, not \
             by this search's {expected_config_hash}. The enclosing autoresearch manifest binds \
             receipt/session/slot, and this independent configuration stamp still disagrees. \
             Refused: a statistic deflated against another run's trials is not this run's statistic."
        )),
        None => Err(format!(
            "the trial-returns manifest under {cache_dir} carries NO config_hash, so the matrix \
             beside it cannot be attributed to this search. `None` means the writer had no stamp; \
             it does not mean 'matches'."
        )),
    }
}

/// The champion's per-month return series — this search's candidate row for the
/// session champion matrix (`pbo_session`).
///
/// Read from the trial-returns matrix the run just wrote, which is where the
/// per-trial series already lives; nothing is recomputed, so the row and the
/// statistics beside it describe the same trials.
///
/// Only ever called with the exact receipt/config whose strict manifest
/// [`attribute_manifest`] has attributed to this search. The strict matrix
/// reader independently revalidates that manifest before decoding the payload.
fn champion_series(
    config: &DiscoveryConfig,
    evidence_receipt: &CanonicalSearchInputReceiptV2,
    config_hash: &str,
    metrics: &neoethos_search::quality::StrategyMetrics,
) -> (Vec<f64>, Vec<i64>, String) {
    let Ok(matrix) = neoethos_search::deflated::read_matrix(
        &config.discovery_ledger_cache_dir,
        &config.evaluation_symbol,
        &config.timeframe_label,
        evidence_receipt,
        config_hash,
    ) else {
        return (Vec::new(), Vec::new(), metrics.strategy_id.clone());
    };
    let row = matrix
        .rows
        .iter()
        .find(|r| r.strategy_id == metrics.strategy_id);
    match row {
        Some(row) => (
            row.returns.clone(),
            matrix.period_keys.clone(),
            metrics.strategy_id.clone(),
        ),
        None => (Vec::new(), Vec::new(), metrics.strategy_id.clone()),
    }
}

/// Persist what this search selected, so the one out-of-sample touch has
/// something to evaluate. Returns how many genes were written.
///
/// **Every gene stays inside its batch binding.** Its local indices address that
/// batch result's `effective_feature_names`, and the same binding carries the
/// exact receipt, digest, evaluated window, effective config, ordinal, and
/// source cursor. There is no parallel flattened portfolio in schema v5.
///
/// A run that selected nothing writes NOTHING and says so with `0`. That is the
/// honest observable: the promotion path refuses on a missing artifact, and an
/// artifact recording an empty portfolio would be a promise of evidence that
/// contains none.
fn persist_promotion_evidence(
    request: &SearchRequest<'_>,
    outcome: &neoethos_search::orchestration::StreamingRunOutcome<
        neoethos_search::discovery::DiscoveryResult,
    >,
) -> Result<usize> {
    let selected = outcome
        .batches
        .iter()
        .filter(|batch| !batch.result.portfolio.is_empty())
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Ok(0);
    }
    let mut batch_bindings = Vec::with_capacity(selected.len());
    for (ordinal, batch) in selected.into_iter().enumerate() {
        anyhow::ensure!(
            batch.result.portfolio.len() == batch.canonical_portfolio.len(),
            "streaming batch cursor {} selected {} local genes but remapped {} canonical genes; refusing incomplete promotion evidence",
            batch.cursor,
            batch.result.portfolio.len(),
            batch.canonical_portfolio.len()
        );
        validate_search_receipt_against_dataset_receipt(
            request.dataset_receipt,
            &batch.result.search_input_receipt,
        )?;
        let scope = batch
            .result
            .selection_scope()
            .with_context(|| {
                format!(
                    "binding streaming batch cursor {} to its exact evaluated scope",
                    batch.cursor
                )
            })?
            .clone();
        let receipt_sha256 = batch.result.search_input_receipt_sha256()?;
        anyhow::ensure!(
            scope.receipt() == &batch.result.search_input_receipt
                && scope.receipt_sha256() == receipt_sha256,
            "streaming batch cursor {} produced an internally inconsistent receipt/scope digest",
            batch.cursor
        );
        let genes = batch
            .result
            .portfolio
            .iter()
            .cloned()
            .map(|gene| super::TaggedPromotionGeneV4 {
                batch_ordinal: ordinal,
                source_cursor: batch.cursor,
                gene,
            })
            .collect();
        batch_bindings.push(super::PromotionBatchBindingV5 {
            ordinal,
            cursor: batch.cursor,
            search_input_receipt: batch.result.search_input_receipt.clone(),
            receipt_sha256,
            evaluated_window: scope.evaluated_window().clone(),
            search_config_hash: batch.result.search_config_hash.clone(),
            feature_names: batch.result.effective_feature_names.clone(),
            genes,
        });
    }
    let written = batch_bindings
        .iter()
        .map(|binding| binding.genes.len())
        .sum();
    let portfolio = super::PromotionPortfolio {
        schema: super::PROMOTION_EVIDENCE_SCHEMA.to_string(),
        session_id: request.session_id.clone(),
        sweep: request.sweep,
        slot: request.slot,
        // THE STAMP. It is the slot's `config_hash` exactly as the runner handed
        // it in — never recomputed here — so the loader's check compares the
        // same number the journal recorded rather than two independent
        // derivations that could drift apart.
        config_hash: request.config_hash.to_string(),
        dataset_receipt: (*request.dataset_receipt).clone(),
        streamed: outcome.streamed,
        batch_count: batch_bindings.len(),
        gene_count: written,
        batch_bindings,
    };
    write_json_atomically(&request.promotion_evidence_path, &portfolio).with_context(|| {
        format!(
            "writing {} slot {}'s promotion evidence ({written} genes across {} exact batch bindings). \
             Without it the single out-of-sample touch has nothing to evaluate, and this path \
             will NOT re-search to find something later.",
            request.sweep,
            request.slot,
            portfolio.batch_count
        )
    })?;
    Ok(written)
}

/// Write JSON through a temporary file and a rename, fsyncing the bytes before
/// the rename.
///
/// A half-written promotion artifact is worse than a missing one: the missing
/// one is refused by name, the half-written one is a parse error at the moment
/// the window is about to be spent.
fn write_json_atomically<T: serde::Serialize>(path: &std::path::Path, value: &T) -> Result<()> {
    use std::io::Write as _;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let tmp = path.with_extension("tmp");
    let bytes =
        serde_json::to_vec(value).with_context(|| format!("serialising {}", path.display()))?;
    {
        let mut file =
            std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("writing {}", tmp.display()))?;
        file.flush()?;
        file.sync_all()
            .with_context(|| format!("fsyncing {}", tmp.display()))?;
    }
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Apply a control sweep's permutation to a feature block, in place.
///
/// **The features move; nothing else does.** `timestamps` are copied across
/// untouched and prices, labels, costs, exit geometry and the GA seed are not
/// reachable from here at all — that asymmetry *is* the experiment. Every
/// feature keeps its marginal distribution and (under `CircularRotation`) its
/// autocorrelation; the only thing destroyed is its alignment with the future.
///
/// The frame is `[samples × features]` and dense in ROW-major order, hence
/// [`crate::shuffle::BlockLayout::RowMajor`]. A layout mistake here would rotate
/// across features instead of across time and quietly produce a control that is
/// not a control. Values and validity reasons take the exact same permutation,
/// while the existing frame keeps its plan, provenance, generation leases, and
/// row origin. It never falls back to the unpermuted frame — a control that
/// silently ran live data is the one result that would poison the null.
fn apply_shuffle_control(
    permutation: &crate::shuffle::FeaturePermutation,
    frame: &mut neoethos_data::FeatureFrame,
) -> Result<()> {
    let rows = frame.n_samples();
    let cols = frame.n_features();
    if rows == 0 || cols == 0 {
        bail!(
            "the shuffle control was handed a {rows}x{cols} feature block. There is nothing to \
             permute, so the control sweep would be a second live run wearing the control's label."
        );
    }

    let dense = frame
        .to_dense_samples_major()
        .context("materializing the exact f64/validity feature frame for the shuffle control")?;
    let values = dense.values.as_slice().ok_or_else(|| {
        anyhow::anyhow!(
            "the dense f64 [samples x features] block is not in standard row-major order, so the \
             shuffle would move data across features instead of across time"
        )
    })?;
    let validity = dense.validity.as_slice().ok_or_else(|| {
        anyhow::anyhow!(
            "the dense validity [samples x features] block is not in standard row-major order, \
             so it cannot follow the exact same shuffle as its f64 values"
        )
    })?;

    let shuffled_values = permutation
        .apply(values, rows, cols, crate::shuffle::BlockLayout::RowMajor)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let shuffled_validity = permutation
        .apply(validity, rows, cols, crate::shuffle::BlockLayout::RowMajor)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    let columns = frame
        .names
        .iter()
        .enumerate()
        .map(|(column, name)| {
            let values = (0..rows)
                .map(|row| shuffled_values[row * cols + column])
                .collect();
            let validity = (0..rows)
                .map(|row| shuffled_validity[row * cols + column])
                .collect();
            neoethos_data::FeatureColumnF64::new(name.clone(), values, validity)
        })
        .collect::<Result<Vec<_>>>()?;

    // Replace only the physical values. The frame object itself stays in
    // place, so its exact FeaturePlan, source provenance, source-generation
    // leases, timestamps, names, and row origin remain untouched.
    frame.data = neoethos_data::FeatureData::InMemory(columns);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// THE PROMOTION EVIDENCE, TESTED FROM BOTH ENDS
//
// These tests exist because the defect they cover was not a wrong number: the
// read site existed, the comment above it described what it read, and NOTHING
// IN THE WORKSPACE EVER WROTE THE FILE. A unit test of the reader alone would
// have passed — it would have asserted that a missing file errors. So the
// writer and the reader are exercised against each other, and every refusal is
// asserted by the words it uses: a refusal nobody can recognise is a crash with
// better manners.
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{PROMOTION_EVIDENCE_SCHEMA, PromotionPortfolio, load_promotion_portfolio};

    #[test]
    fn streaming_bridge_preserves_and_aggregates_measured_cost_band_censuses() {
        let censuses = [
            neoethos_search::discovery::CostBandCensus {
                survives: 1,
                optimistic_edge_only: 2,
                fails: 3,
                unmeasured: 4,
                not_discriminating: 5,
            },
            neoethos_search::discovery::CostBandCensus {
                survives: 6,
                optimistic_edge_only: 7,
                fails: 8,
                unmeasured: 9,
                not_discriminating: 10,
            },
        ];

        let counts = aggregate_cost_band_censuses_v1(censuses.iter(), true)
            .expect("bounded measured census aggregation");
        assert_eq!(counts.survives, 7);
        assert_eq!(counts.optimistic_edge_only, 9);
        assert_eq!(counts.fails, 11);
        assert_eq!(counts.unmeasured, 13);
        assert_eq!(counts.not_discriminating, 15);
        assert!(counts.discriminates);
    }

    fn dataset_receipt(generation_id: &str) -> DatasetReceiptV1 {
        let identity = neoethos_data::CanonicalDatasetIdentity::external(
            "autoresearch-streaming-test",
            "EURUSD",
            neoethos_data::CanonicalTimeframe::M1,
            neoethos_data::BarTimestampConvention::BarOpen,
        )
        .expect("test dataset identity");
        DatasetReceiptV1::new(
            identity.clone(),
            vec![DirectTimeframeReceiptV1 {
                dataset_identity: identity,
                manifest_schema_id: "neoethos.dataset-manifest.v1".to_owned(),
                manifest_sha256: [3; 32],
                generation_id: generation_id.to_owned(),
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
        .expect("test dataset receipt")
    }

    fn canonical_search_receipt(
        dataset_receipt: &DatasetReceiptV1,
        generation_id: &str,
    ) -> neoethos_search::CanonicalSearchInputReceiptV2 {
        let direct = &dataset_receipt.direct_timeframes[0];
        let value = serde_json::json!({
            "schema_version": 2,
            "anchor_dataset_identity": dataset_receipt.anchor_identity.to_path_component(),
            "feature_plan_identity": "00".repeat(32),
            "feature_provenance_identity": "01".repeat(32),
            "feature_content_sha256": "02".repeat(32),
            "feature_execution": {
                "schema_version": 1,
                "compute_policy": "auto",
                "vector_ta_math_authority": "neoethos.vector-ta.cpu-f64-exact-bits.v1",
                "selected_lane": "cpu_scalar"
            },
            "source_bindings": [{
                "source_node_id": "base",
                "dataset_identity": direct.dataset_identity.to_path_component(),
                "manifest_schema_id": direct.manifest_schema_id.as_str(),
                "manifest_sha256": "03".repeat(32),
                "generation_id": generation_id,
                "vortex_sha256": "04".repeat(32),
                "bar_timestamp_convention": direct
                    .dataset_identity
                    .bar_timestamp_convention()
                    .to_string(),
                "segments": [{
                    "row_start": 0,
                    "row_end": 801,
                    "timestamp_start_ms": 0,
                    "timestamp_end_ms": 800
                }]
            }]
        });
        neoethos_search::CanonicalSearchInputReceiptV2::from_json_bytes(
            &serde_json::to_vec(&value).expect("serialize canonical search receipt fixture"),
        )
        .expect("valid canonical search receipt fixture")
    }

    #[test]
    fn search_receipt_must_match_the_frozen_session_anchor_and_generation() {
        let frozen = dataset_receipt("generation-a");
        let exact = canonical_search_receipt(&frozen, "generation-a");
        validate_search_receipt_against_dataset_receipt(&frozen, &exact)
            .expect("the exact anchor and generation must validate");

        let substituted_generation = canonical_search_receipt(&frozen, "generation-b");
        let generation_error =
            validate_search_receipt_against_dataset_receipt(&frozen, &substituted_generation)
                .expect_err("a substituted generation must fail closed");
        let generation_error = format!("{generation_error:#}");
        assert!(
            generation_error.contains("generation-a"),
            "{generation_error}"
        );
        assert!(
            generation_error.contains("generation-b"),
            "{generation_error}"
        );

        let mut substituted_anchor: serde_json::Value =
            serde_json::from_slice(&exact.to_json_bytes().expect("serialize exact receipt"))
                .expect("parse exact receipt fixture");
        let foreign_anchor = neoethos_data::CanonicalDatasetIdentity::external(
            "other-source",
            "EURUSD",
            neoethos_data::CanonicalTimeframe::M1,
            neoethos_data::BarTimestampConvention::BarOpen,
        )
        .expect("foreign anchor identity")
        .to_path_component();
        substituted_anchor["anchor_dataset_identity"] = foreign_anchor.clone().into();
        substituted_anchor["source_bindings"][0]["dataset_identity"] = foreign_anchor.into();
        let substituted_anchor = neoethos_search::CanonicalSearchInputReceiptV2::from_json_bytes(
            &serde_json::to_vec(&substituted_anchor).expect("serialize substituted anchor"),
        )
        .expect("substituted anchor remains structurally valid");
        let anchor_error =
            validate_search_receipt_against_dataset_receipt(&frozen, &substituted_anchor)
                .expect_err("a substituted anchor must fail closed");
        assert!(format!("{anchor_error:#}").contains("anchor"));
    }

    fn gene(indices: &[usize]) -> Gene {
        Gene {
            indices: indices.to_vec(),
            weights: vec![1.0; indices.len()],
            strategy_id: "s-1".to_string(),
            ..Gene::default()
        }
    }

    fn session_id() -> crate::session::SessionId {
        crate::session::SessionId::parse("ar-streaming-test").unwrap()
    }

    const PROPOSAL_CONFIG_HASH: &str = "fnv64:aaaaaaaaaaaaaaaa";
    const EFFECTIVE_SEARCH_CONFIG_HASH: &str = "fnv64:1111111111111111";

    #[test]
    fn population_auto_resolution_uses_the_exact_stored_selection_row_count() {
        let config = DiscoveryConfig {
            population: 10,
            population_auto: true,
            ..DiscoveryConfig::default()
        };
        let selection_window = neoethos_search::CanonicalSearchEvaluatedWindowV1::new(
            neoethos_search::CanonicalSearchWindowRoleV1::InSample,
            0,
            640,
            0,
            639,
        )
        .expect("exact stored 80% selection window");
        let pre_holdout_window = neoethos_search::CanonicalSearchEvaluatedWindowV1::new(
            neoethos_search::CanonicalSearchWindowRoleV1::DiscoveryInput,
            0,
            801,
            0,
            800,
        )
        .expect("pre-holdout frame window");
        let card_ceiling = |stage1_rows, _feature_count| Some(stage1_rows);

        let exact =
            resolve_population_for_exact_window(&config, &selection_window, 1, card_ceiling)
                .expect("exact selection population");
        let wrong_full_count =
            resolve_population_for_exact_window(&config, &pre_holdout_window, 1, card_ceiling)
                .expect("pre-holdout population");

        assert_eq!(exact.population, 160);
        assert_eq!(wrong_full_count.population, 200);
        assert_ne!(
            exact.population, wrong_full_count.population,
            "population_auto must resolve from the exact 80% selection rows, never the pre-holdout frame length"
        );
    }

    fn canonical_search_receipt_with_plan(
        dataset_receipt: &DatasetReceiptV1,
        plan_byte: u8,
    ) -> neoethos_search::CanonicalSearchInputReceiptV2 {
        let direct = &dataset_receipt.direct_timeframes[0];
        let value = serde_json::json!({
            "schema_version": 2,
            "anchor_dataset_identity": dataset_receipt.anchor_identity.to_path_component(),
            "feature_plan_identity": format!("{plan_byte:02x}").repeat(32),
            "feature_provenance_identity": "01".repeat(32),
            "feature_content_sha256": "02".repeat(32),
            "feature_execution": {
                "schema_version": 1,
                "compute_policy": "auto",
                "vector_ta_math_authority": "neoethos.vector-ta.cpu-f64-exact-bits.v1",
                "selected_lane": "cpu_scalar"
            },
            "source_bindings": [{
                "source_node_id": "base",
                "dataset_identity": direct.dataset_identity.to_path_component(),
                "manifest_schema_id": direct.manifest_schema_id.as_str(),
                "manifest_sha256": "03".repeat(32),
                "generation_id": direct.generation_id.as_str(),
                "vortex_sha256": "04".repeat(32),
                "bar_timestamp_convention": direct
                    .dataset_identity
                    .bar_timestamp_convention()
                    .to_string(),
                "segments": [{
                    "row_start": 0,
                    "row_end": 801,
                    "timestamp_start_ms": 0,
                    "timestamp_end_ms": 800
                }]
            }]
        });
        neoethos_search::CanonicalSearchInputReceiptV2::from_json_bytes(
            &serde_json::to_vec(&value).expect("serialize canonical search receipt fixture"),
        )
        .expect("valid canonical search receipt fixture")
    }

    fn v5_batch_binding(
        frozen: &DatasetReceiptV1,
        ordinal: usize,
        cursor: usize,
        plan_byte: u8,
    ) -> crate::runner::PromotionBatchBindingV5 {
        let receipt = canonical_search_receipt_with_plan(frozen, plan_byte);
        let receipt_sha256 = receipt.identity_sha256().expect("receipt identity");
        let evaluated_window = neoethos_search::CanonicalSearchEvaluatedWindowV1::new(
            neoethos_search::CanonicalSearchWindowRoleV1::InSample,
            0,
            640,
            0,
            639,
        )
        .expect("exact in-sample selection window");
        let scope =
            neoethos_search::CanonicalSearchArtifactScopeV2::new(receipt.clone(), evaluated_window)
                .expect("exact stored selection scope");
        crate::runner::PromotionBatchBindingV5 {
            ordinal,
            cursor,
            search_input_receipt: receipt,
            receipt_sha256,
            evaluated_window: scope.evaluated_window().clone(),
            search_config_hash: EFFECTIVE_SEARCH_CONFIG_HASH.to_owned(),
            feature_names: vec![format!("batch_{ordinal}_rsi_14")],
            genes: vec![crate::runner::TaggedPromotionGeneV4 {
                batch_ordinal: ordinal,
                source_cursor: cursor,
                gene: gene(&[0]),
            }],
        }
    }

    fn v5_discovery_result(
        frozen: &DatasetReceiptV1,
        plan_byte: u8,
        feature_name: &str,
    ) -> neoethos_search::discovery::DiscoveryResult {
        let receipt = canonical_search_receipt_with_plan(frozen, plan_byte);
        let selection_window = neoethos_search::CanonicalSearchEvaluatedWindowV1::new(
            neoethos_search::CanonicalSearchWindowRoleV1::InSample,
            0,
            640,
            0,
            639,
        )
        .expect("exact in-sample selection window");
        let holdout_window = neoethos_search::CanonicalSearchEvaluatedWindowV1::new(
            neoethos_search::CanonicalSearchWindowRoleV1::Holdout,
            640,
            801,
            640,
            800,
        )
        .expect("exact internal holdout window");
        let selection_scope =
            neoethos_search::CanonicalSearchArtifactScopeV2::new(receipt.clone(), selection_window)
                .expect("exact selection scope");
        let holdout_scope =
            neoethos_search::CanonicalSearchArtifactScopeV2::new(receipt.clone(), holdout_window)
                .expect("exact holdout scope");
        neoethos_search::discovery::DiscoveryResult {
            search_input_receipt: receipt,
            selection_scope,
            holdout_scope: Some(holdout_scope),
            search_config_hash: EFFECTIVE_SEARCH_CONFIG_HASH.to_owned(),
            cost_band_by_strategy: Vec::new(),
            cost_band_census: neoethos_search::discovery::CostBandCensus::default(),
            portfolio: vec![gene(&[0])],
            candidates: Vec::new(),
            quality_metrics: Vec::new(),
            logged_trades: Vec::new(),
            effective_feature_names: vec![feature_name.to_owned()],
            effective_smc_gate_threshold: 0.0,
            validation_gates: neoethos_search::discovery::DiscoveryValidationGates::pending(),
            canonical_backtest_artifacts: Vec::new(),
            walkforward_validation_artifacts: Vec::new(),
            forward_test_validation_artifacts: Vec::new(),
            prop_firm_validation_artifacts: Vec::new(),
            funnel_profile: None,
        }
    }

    fn v5_portfolio() -> PromotionPortfolio {
        let frozen = dataset_receipt("generation-a");
        let batch_bindings = vec![
            v5_batch_binding(&frozen, 0, 0, 0),
            v5_batch_binding(&frozen, 1, 64, 2),
        ];
        PromotionPortfolio {
            schema: PROMOTION_EVIDENCE_SCHEMA.to_owned(),
            session_id: session_id(),
            sweep: SweepId(3),
            slot: 7,
            config_hash: PROPOSAL_CONFIG_HASH.to_owned(),
            dataset_receipt: frozen,
            streamed: true,
            batch_count: 2,
            gene_count: 2,
            batch_bindings,
        }
    }

    #[test]
    fn v5_loader_accepts_two_ordered_exact_batch_bindings() {
        let dir = tempfile::tempdir().unwrap();
        let written = v5_portfolio();
        let path = write_json_fixture(dir.path(), &written);

        let loaded = load_promotion_portfolio(
            &path,
            &written.session_id,
            written.sweep,
            written.slot,
            &written.config_hash,
            &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
            &written.dataset_receipt,
        )
        .expect("every exact batch binding must validate before OOS");

        assert_eq!(loaded.batch_bindings.len(), 2);
        assert_eq!(loaded.batch_bindings[0].cursor, 0);
        assert_eq!(loaded.batch_bindings[1].cursor, 64);
    }

    #[test]
    fn v5_loader_refuses_a_missing_batch_binding() {
        let mut portfolio = v5_portfolio();
        portfolio.batch_bindings.pop();
        let error = portfolio
            .assert_bindings(
                &session_id(),
                SweepId(3),
                7,
                PROPOSAL_CONFIG_HASH,
                &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
                &dataset_receipt("generation-a"),
                Path::new("promotion.json"),
            )
            .expect_err("declared batch count without its binding must fail closed");
        assert!(format!("{error:#}").contains("batch_count"));
    }

    #[test]
    fn v5_loader_refuses_swapped_batch_order_and_cursor_binding() {
        let mut portfolio = v5_portfolio();
        portfolio.batch_bindings.swap(0, 1);
        let error = portfolio
            .assert_bindings(
                &session_id(),
                SweepId(3),
                7,
                PROPOSAL_CONFIG_HASH,
                &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
                &dataset_receipt("generation-a"),
                Path::new("promotion.json"),
            )
            .expect_err("batch order is evidence and must not be swappable");
        assert!(format!("{error:#}").contains("ordinal"));
    }

    #[test]
    fn v5_loader_refuses_same_dataset_with_a_different_feature_plan() {
        let mut portfolio = v5_portfolio();
        portfolio.batch_bindings[0].search_input_receipt =
            canonical_search_receipt_with_plan(&portfolio.dataset_receipt, 9);
        let error = portfolio
            .assert_bindings(
                &session_id(),
                SweepId(3),
                7,
                PROPOSAL_CONFIG_HASH,
                &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
                &dataset_receipt("generation-a"),
                Path::new("promotion.json"),
            )
            .expect_err("a different feature plan cannot inherit the old receipt digest");
        assert!(format!("{error:#}").contains("SHA-256"));
    }

    #[test]
    fn v5_loader_refuses_wrong_evaluated_window() {
        let mut portfolio = v5_portfolio();
        portfolio.batch_bindings[0].evaluated_window =
            neoethos_search::CanonicalSearchEvaluatedWindowV1::new(
                neoethos_search::CanonicalSearchWindowRoleV1::Holdout,
                0,
                801,
                0,
                800,
            )
            .expect("structurally valid but semantically wrong window");
        let error = portfolio
            .assert_bindings(
                &session_id(),
                SweepId(3),
                7,
                PROPOSAL_CONFIG_HASH,
                &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
                &dataset_receipt("generation-a"),
                Path::new("promotion.json"),
            )
            .expect_err("a holdout window cannot replace the exact in-sample selection scope");
        assert!(format!("{error:#}").contains("window"));
    }

    #[test]
    fn v5_loader_refuses_wrong_effective_search_config() {
        let mut portfolio = v5_portfolio();
        portfolio.batch_bindings[1].search_config_hash = "fnv64:2222222222222222".to_owned();
        let error = portfolio
            .assert_bindings(
                &session_id(),
                SweepId(3),
                7,
                PROPOSAL_CONFIG_HASH,
                &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
                &dataset_receipt("generation-a"),
                Path::new("promotion.json"),
            )
            .expect_err("one foreign effective config must fail the whole artifact");
        assert!(format!("{error:#}").contains("2222222222222222"));
    }

    #[test]
    fn v5_json_refuses_a_flattened_untagged_gene() {
        let mut value = serde_json::to_value(v5_portfolio()).expect("serialize v5 fixture");
        value["genes"] = serde_json::to_value(vec![gene(&[0])]).expect("serialize flat gene");
        let error = serde_json::from_value::<PromotionPortfolio>(value)
            .expect_err("v5 has no parallel flat gene path");
        assert!(error.to_string().contains("unknown field `genes`"));
    }

    #[test]
    fn v5_nonempty_production_writer_round_trips_two_exact_batches() {
        let dir = tempfile::tempdir().unwrap();
        let frozen = dataset_receipt("generation-a");
        let config = DiscoveryConfig::default();
        let session_id = session_id();
        let path = dir.path().join("promotion").join("slot_007.json");
        let request = SearchRequest {
            session_id: &session_id,
            dataset_receipt: &frozen,
            sweep: SweepId(3),
            slot: 7,
            config: &config,
            config_hash: PROPOSAL_CONFIG_HASH,
            trial_returns_path: dir.path().join("unused.bin"),
            promotion_evidence_path: path.clone(),
            permutation: None,
        };
        let outcome = neoethos_search::orchestration::StreamingRunOutcome {
            batches: vec![
                neoethos_search::orchestration::StreamingBatchSurvivor {
                    cursor: 0,
                    pairs: 1,
                    result: v5_discovery_result(&frozen, 0, "positive_signal"),
                    canonical_portfolio: vec![gene(&[0])],
                },
                neoethos_search::orchestration::StreamingBatchSurvivor {
                    cursor: 64,
                    pairs: 1,
                    result: v5_discovery_result(&frozen, 2, "negative_signal"),
                    canonical_portfolio: vec![gene(&[1])],
                },
            ],
            canonical: neoethos_search::orchestration::batch_ledger::CanonicalFeatureIndex::new(),
            ledger: neoethos_search::orchestration::batch_ledger::StreamingRunLedger::new(),
            streamed: true,
            next_cursor: 128,
            space_len: 128,
            batch_columns: 64,
        };

        assert_eq!(
            persist_promotion_evidence(&request, &outcome).expect("real non-empty writer"),
            2
        );
        let bytes = std::fs::read(&path).expect("read exact writer bytes");
        let json: serde_json::Value =
            serde_json::from_slice(&bytes).expect("parse exact writer bytes");
        assert!(json.get("genes").is_none(), "v5 must not flatten genes");
        assert_eq!(json["batch_bindings"].as_array().unwrap().len(), 2);

        let loaded = load_promotion_portfolio(
            &path,
            &session_id,
            SweepId(3),
            7,
            PROPOSAL_CONFIG_HASH,
            &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
            &frozen,
        )
        .expect("strict loader must accept the real writer's bytes");
        assert_eq!(loaded.batch_count, 2);
        assert_eq!(loaded.gene_count, 2);
        for (ordinal, expected_cursor) in [0, 64].into_iter().enumerate() {
            let binding = &loaded.batch_bindings[ordinal];
            assert_eq!(binding.ordinal, ordinal);
            assert_eq!(binding.cursor, expected_cursor);
            assert_eq!(binding.genes[0].batch_ordinal, ordinal);
            assert_eq!(binding.genes[0].source_cursor, expected_cursor);
            assert_eq!(
                binding.receipt_sha256,
                binding
                    .search_input_receipt
                    .identity_sha256()
                    .expect("exact receipt digest")
            );
            assert_eq!(
                binding.evaluated_window.role(),
                neoethos_search::CanonicalSearchWindowRoleV1::InSample
            );
            assert_eq!(binding.search_config_hash, EFFECTIVE_SEARCH_CONFIG_HASH);
        }
        assert_eq!(loaded.batch_bindings[0].feature_names, ["positive_signal"]);
        assert_eq!(loaded.batch_bindings[1].feature_names, ["negative_signal"]);
    }

    #[test]
    fn v5_oos_projection_uses_each_batch_local_index_and_missing_name_refuses_preflight() {
        use neoethos_data::{FeatureCellValidity, FeatureColumnF64};

        let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(4);
        let columns = vec![
            FeatureColumnF64::new(
                "positive_signal",
                vec![1.0, 2.0, 3.0, 4.0],
                vec![FeatureCellValidity::Valid; 4],
            )
            .expect("positive test feature"),
            FeatureColumnF64::new(
                "negative_signal",
                vec![-1.0, -2.0, -3.0, -4.0],
                vec![FeatureCellValidity::Valid; 4],
            )
            .expect("negative test feature"),
        ];
        let raw = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            timestamps, columns,
        )
        .expect("typed raw OOS feature frame");
        let mut portfolio = v5_portfolio();
        portfolio.batch_bindings[0].feature_names = vec!["positive_signal".to_owned()];
        portfolio.batch_bindings[1].feature_names = vec!["negative_signal".to_owned()];

        let projected = project_promotion_batches(&raw, &portfolio)
            .expect("both exact batch-local projections must resolve");
        assert_eq!(projected.len(), 2);
        assert_eq!(projected[0].binding.ordinal, 0);
        assert_eq!(projected[0].features.names, ["positive_signal"]);
        assert_eq!(projected[1].binding.ordinal, 1);
        assert_eq!(projected[1].features.names, ["negative_signal"]);
        let positive = neoethos_search::genetic::signals_for_gene(
            &projected[0].features,
            &projected[0].binding.genes[0].gene,
        )
        .expect("evaluate batch zero's local index zero");
        let negative = neoethos_search::genetic::signals_for_gene(
            &projected[1].features,
            &projected[1].binding.genes[0].gene,
        )
        .expect("evaluate batch one's local index zero");
        assert_eq!(positive, vec![1, 1, 1, 1]);
        assert_eq!(negative, vec![-1, -1, -1, -1]);

        portfolio.batch_bindings[1].feature_names = vec!["missing_signal".to_owned()];
        let error = project_promotion_batches(&raw, &portfolio)
            .expect_err("a missing local name must refuse in preflight before OOS is spent");
        let rendered = format!("{error:#}");
        assert!(rendered.contains("batch ordinal 1"), "{rendered}");
        assert!(rendered.contains("cursor 64"), "{rendered}");
        assert!(rendered.contains("missing_signal"), "{rendered}");
    }

    fn write_json_fixture(dir: &Path, value: &PromotionPortfolio) -> PathBuf {
        let path = dir.join("promotion").join("slot_007.json");
        write_json_atomically(&path, value).expect("the artifact must be writable");
        path
    }

    // ── C1: the ledger is attributed, not merely located ────────────────────

    #[test]
    fn a_matrix_written_by_this_search_is_attributed_to_it() {
        attribute_manifest(Some(Some("fnv64:mine")), "fnv64:mine", "/scratch")
            .expect("the hashes agree");
    }

    #[test]
    fn a_foreign_configuration_matrix_is_refused_and_names_both_stamps() {
        let err = attribute_manifest(Some(Some("fnv64:theirs")), "fnv64:mine", "/scratch")
            .expect_err("a foreign matrix must not be adopted");
        assert!(err.contains("fnv64:theirs"), "{err}");
        assert!(err.contains("fnv64:mine"), "{err}");
        assert!(err.contains("configuration stamp still disagrees"), "{err}");
    }

    #[test]
    fn an_unstamped_manifest_cannot_be_attributed_and_is_never_treated_as_a_match() {
        let err = attribute_manifest(Some(None), "fnv64:mine", "/scratch")
            .expect_err("None means 'cannot be attributed', not 'matches'");
        assert!(err.contains("NO config_hash"), "{err}");
    }

    #[test]
    fn an_absent_manifest_refuses_rather_than_reading_whatever_is_there() {
        let err = attribute_manifest(None, "fnv64:mine", "/scratch")
            .expect_err("no manifest is no attribution");
        assert!(err.contains("NOT computable"), "{err}");
    }

    fn portfolio(config_hash: &str) -> PromotionPortfolio {
        let mut portfolio = v5_portfolio();
        portfolio.config_hash = config_hash.to_owned();
        portfolio
    }

    fn write(dir: &std::path::Path, value: &PromotionPortfolio) -> PathBuf {
        let path = dir.join("promotion").join("slot_007.json");
        write_json_atomically(&path, value).expect("the artifact must be writable");
        path
    }

    #[test]
    fn shuffle_moves_values_with_validity_and_preserves_the_exact_artifact_contract() {
        use neoethos_data::{FeatureCellValidity, FeatureColumnF64};

        let timestamps = neoethos_data::test_fixtures::canonical_test_timestamps(8);
        let columns = vec![
            FeatureColumnF64::new(
                "feature_a",
                vec![10.0, f64::NAN, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                vec![
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::Warmup,
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::Valid,
                ],
            )
            .expect("typed feature A"),
            FeatureColumnF64::new(
                "feature_b",
                vec![20.0, 21.0, 22.0, 23.0, f64::NAN, 25.0, 26.0, 27.0],
                vec![
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::AlignmentMissing,
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::Valid,
                    FeatureCellValidity::Valid,
                ],
            )
            .expect("typed feature B"),
        ];
        let mut frame = neoethos_data::test_fixtures::ctrader_test_feature_frame_from_columns(
            timestamps, columns,
        )
        .expect("typed feature frame");
        let original = frame.clone();
        let dense = original
            .to_dense_samples_major()
            .expect("materialize original frame");
        let permutation = crate::shuffle::FeaturePermutation::draw(
            crate::shuffle::ControlKind::CircularRotation,
            &crate::session::Session::default(),
            crate::session::BlockId(2),
        );
        let expected_values = permutation
            .apply(
                dense.values.as_slice().expect("contiguous values"),
                original.n_samples(),
                original.n_features(),
                crate::shuffle::BlockLayout::RowMajor,
            )
            .expect("expected value permutation");
        let expected_validity = permutation
            .apply(
                dense.validity.as_slice().expect("contiguous validity"),
                original.n_samples(),
                original.n_features(),
                crate::shuffle::BlockLayout::RowMajor,
            )
            .expect("expected validity permutation");

        apply_shuffle_control(&permutation, &mut frame).expect("typed shuffle control");

        let observed = frame
            .to_dense_samples_major()
            .expect("materialize shuffled frame");
        assert_eq!(
            observed
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected_values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            observed.validity.as_slice(),
            Some(expected_validity.as_slice())
        );
        assert_eq!(frame.timestamps, original.timestamps);
        assert_eq!(frame.names, original.names);
        assert_eq!(frame.plan_identity(), original.plan_identity());
        assert_eq!(frame.provenance_identity(), original.provenance_identity());
        assert!(std::ptr::eq(frame.plan(), original.plan()));
        assert!(std::ptr::eq(frame.provenance(), original.provenance()));
        assert!(matches!(
            frame.data,
            neoethos_data::FeatureData::InMemory(_)
        ));
    }

    #[test]
    fn the_evidence_the_executor_writes_is_the_evidence_the_runner_loads() {
        // The whole defect in one test: a writer and a reader that agree.
        let dir = tempfile::tempdir().unwrap();
        let written = portfolio("fnv64:aaaa");
        let path = write(dir.path(), &written);

        let loaded = load_promotion_portfolio(
            &path,
            &written.session_id,
            SweepId(3),
            7,
            "fnv64:aaaa",
            &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
            &written.dataset_receipt,
        )
        .expect("the artifact it just wrote must load");
        assert_eq!(loaded, written);
        assert_eq!(loaded.gene_count, 2);
        assert_eq!(loaded.batch_bindings.len(), 2);
    }

    #[test]
    fn a_missing_artifact_is_a_named_refusal_that_leaves_the_window_unspent() {
        let dir = tempfile::tempdir().unwrap();
        let err = load_promotion_portfolio(
            &dir.path().join("promotion").join("slot_007.json"),
            &session_id(),
            SweepId(3),
            7,
            "fnv64:aaaa",
            &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
            &dataset_receipt("generation-a"),
        )
        .expect_err("a missing artifact must refuse");
        let text = format!("{err:#}");
        assert!(text.contains("UNSPENT"), "{text}");
        assert!(text.contains("SESSION STORE"), "{text}");
    }

    #[test]
    fn the_stamp_is_what_makes_a_file_at_the_right_path_into_evidence() {
        // A file at the right path is not proof that it describes the right
        // configuration: a resumed session, a re-run sweep or a copied directory
        // all put bytes there. The stamp is checked, and it is checked BEFORE
        // the window is spent.
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), &portfolio("fnv64:aaaa"));
        let err = load_promotion_portfolio(
            &path,
            &session_id(),
            SweepId(3),
            7,
            "fnv64:bbbb",
            &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
            &dataset_receipt("generation-a"),
        )
        .expect_err("a foreign stamp must refuse");
        let text = format!("{err:#}");
        assert!(text.contains("fnv64:bbbb"), "{text}");
        assert!(text.contains("NOT spent"), "{text}");
    }

    #[test]
    fn promotion_refuses_genes_selected_under_a_different_dataset_receipt() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), &portfolio("fnv64:aaaa"));
        let err = load_promotion_portfolio(
            &path,
            &session_id(),
            SweepId(3),
            7,
            "fnv64:aaaa",
            &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
            &dataset_receipt("generation-b"),
        )
        .expect_err("a foreign generation receipt must refuse");
        assert!(format!("{err:#}").contains("different generations"));
    }

    #[test]
    fn promotion_refuses_evidence_from_a_different_session_before_oos() {
        let dir = tempfile::tempdir().unwrap();
        let mut foreign = portfolio("fnv64:aaaa");
        foreign.session_id = crate::session::SessionId::parse("ar-foreign-session").unwrap();
        let path = write(dir.path(), &foreign);
        let expected_session = crate::session::SessionId::parse("ar-current-session").unwrap();

        let err = load_promotion_portfolio(
            &path,
            &expected_session,
            SweepId(3),
            7,
            "fnv64:aaaa",
            &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
            &dataset_receipt("generation-a"),
        )
        .expect_err("promotion evidence from another session must refuse before OOS");
        let rendered = format!("{err:#}");
        assert!(rendered.contains("ar-foreign-session"), "got: {rendered}");
        assert!(rendered.contains("ar-current-session"), "got: {rendered}");
        assert!(rendered.contains("NOT spent"), "got: {rendered}");
    }

    #[test]
    fn scratch_manifest_exactly_binds_session_receipt_slot_and_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = neoethos_search::discovery::DiscoveryConfig::default();
        let session_id = crate::session::SessionId::parse("ar-scratch-manifest-test").unwrap();
        let receipt = dataset_receipt("generation-a");
        let request = SearchRequest {
            session_id: &session_id,
            dataset_receipt: &receipt,
            sweep: SweepId(3),
            slot: 7,
            config: &config,
            config_hash: "fnv64:aaaa",
            trial_returns_path: dir.path().join("unused.bin"),
            promotion_evidence_path: dir.path().join("unused.json"),
            permutation: None,
        };
        let written = ScratchLedgerManifestV1::new(&request);
        let path = dir.path().join(SCRATCH_LEDGER_MANIFEST_FILE);
        write_json_atomically(&path, &written).unwrap();
        validate_scratch_manifest(&path, &written).expect("exact manifest");

        let foreign_receipt = dataset_receipt("generation-b");
        let foreign_request = SearchRequest {
            dataset_receipt: &foreign_receipt,
            ..request
        };
        let expected = ScratchLedgerManifestV1::new(&foreign_request);
        validate_scratch_manifest(&path, &expected)
            .expect_err("the same path cannot adopt a foreign receipt");
    }

    #[test]
    fn one_slots_evidence_is_never_read_as_anothers() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(dir.path(), &portfolio("fnv64:aaaa"));
        let err = load_promotion_portfolio(
            &path,
            &session_id(),
            SweepId(3),
            8,
            "fnv64:aaaa",
            &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
            &dataset_receipt("generation-a"),
        )
        .expect_err("a slot mismatch must refuse");
        assert!(format!("{err:#}").contains("wearing one path"));
    }

    #[test]
    fn an_empty_portfolio_is_not_a_promotion_candidate() {
        let mut empty = portfolio("fnv64:aaaa");
        empty.batch_bindings.clear();
        empty.batch_count = 0;
        empty.gene_count = 0;
        let err = empty
            .assert_bindings(
                &session_id(),
                SweepId(3),
                7,
                "fnv64:aaaa",
                &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
                &dataset_receipt("generation-a"),
                std::path::Path::new("x.json"),
            )
            .expect_err("nothing to evaluate must refuse");
        assert!(format!("{err:#}").contains("nothing to evaluate"));
    }

    #[test]
    fn genes_without_the_names_their_indices_address_are_refused() {
        // An index without a name is a number that reads whatever happens to be
        // in that position — which, on the one out-of-sample window, is a
        // different strategy reported under this one's name.
        let mut nameless = portfolio("fnv64:aaaa");
        nameless.batch_bindings[0].feature_names.clear();
        let err = nameless
            .assert_bindings(
                &session_id(),
                SweepId(3),
                7,
                "fnv64:aaaa",
                &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
                &dataset_receipt("generation-a"),
                std::path::Path::new("x.json"),
            )
            .expect_err("no names must refuse");
        assert!(format!("{err:#}").contains("no local feature names"));

        let mut out_of_range = portfolio("fnv64:aaaa");
        out_of_range.batch_bindings[0].genes[0].gene = gene(&[0, 9]);
        let err = out_of_range
            .assert_bindings(
                &session_id(),
                SweepId(3),
                7,
                "fnv64:aaaa",
                &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
                &dataset_receipt("generation-a"),
                std::path::Path::new("x.json"),
            )
            .expect_err("an index past the names must refuse");
        assert!(format!("{err:#}").contains("internally inconsistent"));
    }

    #[test]
    fn a_schema_this_build_does_not_know_is_refused_rather_than_guessed() {
        let future = PromotionPortfolio {
            schema: "neoethos.autoresearch.promotion_evidence.v99".to_string(),
            ..portfolio("fnv64:aaaa")
        };
        let err = future
            .assert_bindings(
                &session_id(),
                SweepId(3),
                7,
                "fnv64:aaaa",
                &|_| Ok(EFFECTIVE_SEARCH_CONFIG_HASH.to_owned()),
                &dataset_receipt("generation-a"),
                std::path::Path::new("x.json"),
            )
            .expect_err("an unknown schema must refuse");
        assert!(format!("{err:#}").contains("v99"));
    }

    #[test]
    fn scratch_slot_guard_removes_only_its_slot_and_preserves_durable_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let scratch_root = dir.path().join("scratch");
        let unrelated = scratch_root.join("unrelated").join("keep.txt");
        std::fs::create_dir_all(unrelated.parent().unwrap()).unwrap();
        std::fs::write(&unrelated, b"keep").unwrap();

        let durable = dir.path().join("session-store").join("trial-returns.bin");
        std::fs::create_dir_all(durable.parent().unwrap()).unwrap();
        std::fs::write(&durable, b"durable evidence").unwrap();

        let config = neoethos_search::discovery::DiscoveryConfig::default();
        let session_id = session_id();
        let receipt = dataset_receipt("generation-a");
        let request = SearchRequest {
            session_id: &session_id,
            dataset_receipt: &receipt,
            sweep: SweepId(3),
            slot: 7,
            config: &config,
            config_hash: "fnv64:aaaa",
            trial_returns_path: durable.clone(),
            promotion_evidence_path: dir.path().join("session-store").join("promotion.json"),
            permutation: None,
        };

        let slot_path;
        {
            let guard = ScratchSlotGuard::prepare(&scratch_root, &request).unwrap();
            slot_path = guard.path().to_path_buf();
            std::fs::write(guard.path().join("scratch.bin"), b"temporary").unwrap();
            assert!(slot_path.exists());
        }

        assert!(
            !slot_path.exists(),
            "the exact disposable slot must be gone"
        );
        assert_eq!(std::fs::read(&unrelated).unwrap(), b"keep");
        assert_eq!(std::fs::read(&durable).unwrap(), b"durable evidence");
    }

    #[test]
    fn scratch_slot_guard_cleans_up_on_error_return() {
        let dir = tempfile::tempdir().unwrap();
        let scratch_root = dir.path().join("scratch");
        let config = neoethos_search::discovery::DiscoveryConfig::default();
        let session_id = session_id();
        let receipt = dataset_receipt("generation-a");
        let request = SearchRequest {
            session_id: &session_id,
            dataset_receipt: &receipt,
            sweep: SweepId(6),
            slot: 9,
            config: &config,
            config_hash: "fnv64:error",
            trial_returns_path: dir.path().join("durable.bin"),
            promotion_evidence_path: dir.path().join("promotion.json"),
            permutation: None,
        };
        let expected_slot = scratch_root
            .join(receipt.identity().as_str())
            .join(session_id.as_str())
            .join(SweepId(6).to_string())
            .join("slot_009");

        let result = (|| -> Result<()> {
            let guard = ScratchSlotGuard::prepare(&scratch_root, &request)?;
            std::fs::write(guard.path().join("scratch.bin"), b"temporary")?;
            anyhow::bail!("simulated executor error")
        })();

        assert!(format!("{:#}", result.unwrap_err()).contains("simulated executor error"));
        assert!(
            !expected_slot.exists(),
            "RAII must remove the exact scratch slot on Result::Err"
        );
    }

    #[test]
    fn scratch_slot_guard_cleans_up_during_panic_unwind() {
        let dir = tempfile::tempdir().unwrap();
        let scratch_root = dir.path().join("scratch");
        let config = neoethos_search::discovery::DiscoveryConfig::default();
        let session_id = session_id();
        let receipt = dataset_receipt("generation-a");
        let request = SearchRequest {
            session_id: &session_id,
            dataset_receipt: &receipt,
            sweep: SweepId(8),
            slot: 11,
            config: &config,
            config_hash: "fnv64:panic",
            trial_returns_path: dir.path().join("durable.bin"),
            promotion_evidence_path: dir.path().join("promotion.json"),
            permutation: None,
        };
        let expected_slot = scratch_root
            .join(receipt.identity().as_str())
            .join(session_id.as_str())
            .join(SweepId(8).to_string())
            .join("slot_011");

        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let guard = ScratchSlotGuard::prepare(&scratch_root, &request).unwrap();
            std::fs::write(guard.path().join("scratch.bin"), b"temporary").unwrap();
            panic!("simulated executor panic");
        }));

        assert!(unwind.is_err(), "the simulated panic must unwind");
        assert!(
            !expected_slot.exists(),
            "RAII must remove the exact scratch slot during unwind"
        );
        assert!(scratch_root.exists(), "cleanup must not remove its root");
    }

    #[test]
    fn nothing_is_written_for_a_search_that_selected_nothing() {
        // "No artifact" and "an artifact recording an empty portfolio" must be
        // the same observable, so the promotion path has exactly one thing to
        // refuse instead of two.
        let dir = tempfile::tempdir().unwrap();
        let config = neoethos_search::discovery::DiscoveryConfig::default();
        let path = dir.path().join("promotion").join("slot_007.json");
        let session_id = crate::session::SessionId::parse("ar-streaming-test").unwrap();
        let receipt = dataset_receipt("generation-a");
        let request = SearchRequest {
            session_id: &session_id,
            dataset_receipt: &receipt,
            sweep: SweepId(3),
            slot: 7,
            config: &config,
            config_hash: "fnv64:aaaa",
            trial_returns_path: dir.path().join("unused.bin"),
            promotion_evidence_path: path.clone(),
            permutation: None,
        };
        let outcome: neoethos_search::orchestration::StreamingRunOutcome<
            neoethos_search::discovery::DiscoveryResult,
        > = neoethos_search::orchestration::StreamingRunOutcome {
            batches: Vec::new(),
            canonical: neoethos_search::orchestration::batch_ledger::CanonicalFeatureIndex::new(),
            ledger: neoethos_search::orchestration::batch_ledger::StreamingRunLedger::new(),
            streamed: false,
            next_cursor: 0,
            space_len: 0,
            batch_columns: 0,
        };
        assert_eq!(persist_promotion_evidence(&request, &outcome).unwrap(), 0);
        assert!(
            !path.exists(),
            "a search that selected nothing must write nothing"
        );
    }
}
