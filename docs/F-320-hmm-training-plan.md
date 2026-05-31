# F-320 — Wire RegimeHmmExpert into the training pipeline

**Status:** diagnosed, fix deferred to a fresh-context session (the HMM
feature-extraction step is subtle enough that doing it wrong silently
trains a garbage model — worse than the current "not trained" state).

## Diagnosis (confirmed 2026-05-31)

The 3-state HMM regime classifier `RegimeHmmExpert` has a **complete
inference path** (F-231):

- `HmmRegimeAdapter` (`ExpertModel`) — `meta_adapters.rs:513`
- `HmmRegimeLoader` (`ExpertLoader`) — `meta_adapters.rs:552`
- registered in the bootstrap registry (`meta_adapters.rs:582`)
- listed in `DEFAULT_BOOTSTRAP_EXPERT_NAMES` as `"hmm_regime"`
  (`ensemble_inference/bootstrap.rs`)
- `RegimeHmmExpert::train(...)` / `save_to_path` / `load_from_artifact`
  all exist (`forecasting/hmm_regime.rs:155`)

…but there is **no training dispatch** for it:

- `ModelType` enum (`parallel_trainer.rs:310`) has **no `HmmRegime`
  variant** (31 variants, none HMM).
- `training_orchestrator.rs` never calls `RegimeHmmExpert::train` +
  `save_to_path`.

**Consequence (not a crash):** the bootstrap loader scans
`models/<symbol>/<tf>/hmm_regime/`, finds nothing (training never wrote
it), and reports it `missing`/`degraded` per the partial-load policy.
The ensemble runs fine on the other 28 experts — `hmm_regime` simply
never votes. So this is an **enhancement** (light up a dormant model),
not a regression.

## Fix plan

1. **Add the enum variant.** `parallel_trainer.rs:310` — add
   `HmmRegime` to `ModelType`.

2. **Map the name.** `training_orchestrator.rs:~1773` (the
   `&str -> ModelType` match) — add
   `"hmm_regime" => Ok(ModelType::HmmRegime)`.

3. **Capability lists.** `training_orchestrator.rs:~2512` and `~2538`
   are `matches!(...)` filters (which model types are
   classifier-shaped / meta-shaped). The HMM is **unsupervised**
   (no labels) — decide carefully which list(s) it belongs in, or give
   it its own branch. Getting this wrong makes the orchestrator either
   skip it or feed it labels it doesn't want.

4. **Training dispatch.** `training_orchestrator.rs:~2618` (next to the
   `ModelType::ElasticNet => {...}` arm) — add
   `ModelType::HmmRegime => {...}`:
   - **CRITICAL:** `RegimeHmmExpert::train(observations, feature_columns,
     config)` wants `observations: &Array2<f64>` with **exactly
     `FEATURE_DIM` columns** and `feature_columns.len() == FEATURE_DIM`
     (`hmm_regime.rs:160-178` `bail!`s otherwise). `FEATURE_DIM` is the
     HMM's own feature set (log-return + volatility/range features —
     read `hmm_regime.rs` for the canonical column order). **Do NOT pass
     the generic classifier feature matrix** — extract/compute the HMM's
     specific columns from the bars, in the right order.
   - Then `RegimeHmmExpert::save_to_path(<models_dir>/<sym>/<tf>/hmm_regime)`
     so the loader finds it.

5. **Test.** Mirror the existing round-trip in
   `meta_adapters.rs:697-713` (build a minimal `RegimeHmmExpert` via
   `train`, save, `load_from_artifact`, assert `predict_proba` works).
   Add an orchestrator-level test that `hmm_regime` produces an artifact
   after a training run on a small fixture.

## Acceptance

After the fix, a training run writes `models/<sym>/<tf>/hmm_regime/`,
the bootstrap reports `hmm_regime` as **loaded** (not missing), and the
soft-voting ensemble count goes from 28 → 29 loaded experts on a fully
trained symbol.
