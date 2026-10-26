# GA fitness normalization: controlled Rust A/B, 26 October 2026

## Reproducible run

- **Successful GitHub Actions run:** https://github.com/kosred/Neoethos/actions/runs/40247532231
- **Experiment source:** `crates/neoethos-gpu-contracts/examples/normalization_ablation.rs`
- **Log artifact:** `normalization-ablation-results` in the linked workflow run.
- **Scope:** experimental branch only. No production search/scorer change, no master, no broker orders, no market-data read, final OOS untouched.

Six candidate selection methods on identical 400-candidate populations:
1. **baseline_all_finite_minmax:** host-GA equivalent: min-max normalized fitness of all finite scores (including -100 zero-trade sentinels), then 80% normalized fitness + 20% novelty.
2. **raw_only:** fitness only, no novelty.
3. **raw_plus_0p2_novelty:** unnormalized raw fitness + 0.2*normalized novelty.
4. **active_minmax:** bounds calculated from finite actually-trading candidates only; no-traders excluded from ranking.
5. **active_percentile:** percentile rank of finite actually-trading candidates, combined 80/20 with novelty, no-traders excluded.
6. **active_winsorized_5_95:** bounds clipped to eligible 5th–95th percentiles before 80/20 combination; no-traders excluded.

64 reproducible seeds, two **actual Rust NeoEthos scoring formulas** (Risky goal V6 / PropFirm V4), five shares of zero-trade candidates: 0%, 5%, 25%, 60%, 90%. This gives 640 **independent seeded population/profile simulations**, with six sorting methods applied to each = 3,840 selection analyses, each at population 400 (1,536,000 candidate-method comparisons; rescoring operates per profile).

**Essential scope limitation:** Each input candidate's 11 metrics was SYNTHETIC and constructed with a positive correlation between latent quality and modeled net. The native Rust named scorer was called exactly, but no market price series, realized broker P&L, strategy evolution or future-window validation was used. Selection of high hypothetical net is guaranteed to be aligned by the fixture; these are selection-pressure/ordering tests, not evidence of profitable trading.

## Selected results: 60% zero-trade population

All entries mean over 64 seeds; TOP 20 selected out of 400.

| Mode | Selection | Mean native raw fitness of selected 20 | Positive *synthetic* net among selected 20 | Overlap with native raw top 20 |
|---|---|---:|---:|---:|
| Risky V6 | **current baseline** | 0.016794 | 12.69 / 20 | **2.77 / 20** |
| Risky V6 | raw fitness only | 0.076922 | 20.00 / 20 | 20.00 / 20 |
| Risky V6 | raw + novelty | 0.059387 | 19.97 / 20 | 7.52 / 20 |
| Risky V6 | active-only min-max | 0.073319 | 20.00 / 20 | 13.94 / 20 |
| Risky V6 | **active-only percentile** | **0.074348** | **20.00 / 20** | **15.09 / 20** |
| Risky V6 | active-only winsorized | 0.073487 | 20.00 / 20 | 14.16 / 20 |
| PropFirm V4 | current baseline | -0.331261 | 13.84 / 20 | 3.38 / 20 |
| PropFirm V4 | raw fitness only | 0.376406 | 20.00 / 20 | 20.00 / 20 |
| PropFirm V4 | raw + novelty | 0.368490 | 20.00 / 20 | 17.62 / 20 |
| PropFirm V4 | active-only min-max | 0.329384 | 20.00 / 20 | 14.16 / 20 |
| PropFirm V4 | **active-only percentile** | **0.338507** | **20.00 / 20** | **14.98 / 20** |
| PropFirm V4 | active-only winsorized | 0.329383 | 20.00 / 20 | 14.33 / 20 |

## Robustness across fractions (Risky V6)

Overlap with raw-fitness TOP 20:

| Zero-trade share | Current baseline | Active percentile |
|---|---:|---:|
| 0% | 9.62/20 | 10.77/20 |
| 5% | 1.30/20 | 11.19/20 |
| 25% | 1.09/20 | 11.80/20 |
| 60% | 2.77/20 | 15.09/20 |
| 90% | 10.28/20 | 18.83/20 |

The 90%-nontrader case has only ~40 eligible candidates; the absolute overlap rises mechanically because the eligible pool is small. Do not interpret a high overlap there as superior generalization.

## Additional mathematical invariants demonstrated by executable assertions

- With novelty weight **zero**, min-max normalization preserves exact native raw-fitness ordering.
- Mixing novelty into **raw unnormalized** scores changes selection when raw scoring units are multiplied by 100, even though individual relative fitness rankings are unchanged. This disproves the idea that simply removing 0–1 normalization is a generally scale-invariant fix.
- The code replicates the host search engine's all-finite min-max rank formula, but **does not** execute the real end-to-end GA or native CUDA scoring kernels.
- With novelty activated, no-trade -100 can force almost all viable native fitness into a narrow ~1.0 interval while normalized novelty continues to span 0–1.

## Newly confirmed conceptual issue

`crates/neoethos-search/src/genetic/search_engine.rs` builds `index_sets` from `gene.indices` and calls `exact_sparse_jaccard_knn_novelty_v1` on **sets of indicators**. This measures structural/feature-set diversity, not independent evidence of behavioral diversity in trades or returns. It may reward two materially similar trading behaviors with different indicator sets. Establishing actual behavioral novelty would require signal/trade-overlap/outcome metrics on the same allowed training data.

## Recommendations and experiment restrictions

1. **Do not remove all normalization while retaining the same additive novelty bonus.** Raw-score units differ between Risky V5/V6 and PropFirm V4 and across account capital profiles.
2. Preferred candidate for an **isolated branch-only implementation** is eligible-only, percentile-based fitness rank, with no-trade/no-finite candidates excluded from the selection fitness population; retain their deterministic rescue/immigrant exploration mechanisms. Compare with active-only min-max and raw-only control.
3. Do not let a percentile-based score be mistaken for actual monetary return, and do not allow a negative net candidate to be silently relabeled profitable because of a high percentile among worse negative candidates. Archive and promotion monetary thresholds remain separately enforced.
4. Investigate risk-normalized trade expectancy (net R/margin-of-safety), actual trading behavioral novelty, capped volume incentives and PropFirm PF=1 discontinuity in separate A/B axes.
5. Confirm native **CUDA** scoring and normalization contract independently. Its separate kernels may behave differently and need parity tests; no CUDA experiment ran here.
6. **Next valid edge test:** compare baseline vs new candidate selection in the true Rust search with the same pre-OOS broker-authoritative EURUSD bars, seeds, limits, spread/swap/commission, and genuine dynamic lot. Track net R expectancy, fraction with positive broker-cost-adjusted IS and *new* embargoed inner-forward folds, PF, DD, trade count, genetic diversity and survivorship. The already-inspected internal forward window is not pristine; **never use final reserved 20% for tuning**.

**Conclusion:** The 0–1 scale is not inherently the error; **the choice of min-max bounds and the units used when blending feature novelty with financial fitness** demonstrably corrupt selection priority in these controlled input scenarios. Synthetic ranking tests do not prove profitable trading or justify disabling risk limits.
