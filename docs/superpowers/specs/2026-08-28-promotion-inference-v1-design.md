# Promotion Inference V1 Design

Date: 2026-08-28

Status: approved for local RED preparation; authoritative source/Cargo work remains blocked until the P4A owner releases the shared window.

## Goal

Open one already-installed P3 promotion-candidate model ensemble from its immutable candidate root and `PromotionCandidateTrainingManifestV1`, using only an explicit immutable `PromotionInferencePolicyV1`. Return a move-only verified carrier that owns every loaded expert in memory for later P4C consumption.

This scope does not promote the candidate, mint promotion authority, evaluate OOS, or change live-trading behavior. The P3 artifact remains `ResearchOnly` and `NotPromotionEligible`.

## Non-goals and fail-closed boundaries

- No `Settings::load()` or other ambient settings lookup.
- No ambient/global models path, symbol resolver, current pointer, or registry bootstrap that derives paths or voting policy from settings.
- No partial ensemble, degraded expert, missing voter, unexpected voter, or gene-only fallback.
- No caller-constructible or serializable verified carrier.
- No deferred model-file reads after the opener succeeds.
- No mutation of legacy global ensemble-loading behavior.

## Public boundary

```rust
pub fn open_installed_promotion_ensemble_v1(
    candidate_root: &Path,
    manifest: PromotionCandidateTrainingManifestV1,
    policy: PromotionInferencePolicyV1,
) -> Result<VerifiedPromotionEnsembleV1, PromotionEnsembleOpenRefusalV1>;
```

The manifest and policy are consumed. The result owns the manifest, the reopened P3 handoff, the immutable policy, and the fully loaded `SoftVotingEnsemble`. It exposes only borrowing getters needed by P4C. It has private fields, no public constructor, and no `Clone`, `Copy`, `Serialize`, or `Deserialize` implementation.

## Explicit policy

`PromotionInferencePolicyV1` is created only through a checked constructor and binds:

- schema/version and a deterministic policy identity SHA;
- exact P3 handoff identity SHA;
- exact candidate tree SHA;
- ordered required model inventory;
- role-aware soft-voting policy version;
- explicit anomaly blend knees.

Each ordered model requirement binds:

- canonical model name;
- exact P3 model-tree SHA;
- artifact feature-schema hash;
- exact ordered feature-column vector;
- expected `BackendKind`;
- exact planned-backend string;
- expected `TrainingPrecision`;
- expected `ExpertOutputKind`;
- finite, positive voting weight.

Policy construction rejects empty or duplicate model names, duplicate feature columns, empty feature vectors, invalid hashes, non-finite/non-positive weights, degraded/unavailable backends, unknown blend version, and invalid anomaly-knee ordering. Internal validation recomputes the identity so in-module mutation cannot bypass the seal.

## Artifact boundary

For each manifest-listed model directory, the opener reads the bounded typed `model_runtime_artifact.json` as `ModelRuntimeArtifact<TrainingRuntimeProfile>`. It validates provenance explicitly after deserialization, including artifact kind, runtime-safety report, model/symbol/timeframe binding, feature-schema identity, backend, planned backend, and planned precision.

The opener never accepts arbitrary directory traversal. Candidate and model relative paths must match the P3 canonical layout exactly:

```text
<candidate_root>/<candidate_relative_dir>/<symbol>/<base_timeframe>/<model_name>
```

The runtime-artifact reader enforces a fixed byte cap and rejects non-files, symlinks, malformed JSON, trailing data, and schema/provenance mismatches.

## Open and verification order

1. Validate and rehash the policy.
2. Derive the candidate directory only from `candidate_root` plus the P3 manifest relative path.
3. Enumerate the exact manifest/policy model inventory and classify missing and unexpected entries before the generic tree check.
4. Call `manifest.verify_installed(candidate_root)`.
5. Reopen the P3 handoff and bind its identity, dataset generation/series/search receipt, locked finalist portfolio, cutoff/purge, broker-authority identity, runtime/model config, and planned model order through the frozen P3 receipt.
6. Require exact order equality across handoff planned models, manifest model artifacts, and policy requirements. Require exact per-model tree SHA and canonical relative directory.
7. Read and validate each typed runtime artifact.
8. Build the default loader registry, which itself performs no settings lookup. Unit tests use a private injectable registry seam.
9. Call the registry partial-load primitive only as a low-level loader, then upgrade its result to exact all-or-refuse semantics: no missing, degraded, unexpected, reordered, wrong-name, or wrong-output expert.
10. Compare each in-memory expert's exact ordered feature columns against policy and recompute the stable feature-schema hash.
11. Construct `SoftVotingEnsemble` directly from the explicit policy weights and anomaly knees; never call a settings-derived voting helper.
12. Call `manifest.verify_installed(candidate_root)` again after loading to close load-time mutation races.
13. Return the move-only carrier owning all in-memory experts.

## Race and mutation behavior

The P3 verification before and after model loading binds the complete candidate tree. Any file, model tree, handoff, or evidence mutation during the open is refused. The result does not retain a path-based lazy-loader capability, so a successful caller cannot trigger later artifact re-reads.

## Errors

`PromotionEnsembleOpenRefusalV1` uses typed, fail-closed variants for invalid policy, handoff mismatch, candidate-tree mismatch, missing model, unexpected model, runtime-artifact refusal, degraded or missing loader outcome, loaded-model mismatch, feature-order/schema mismatch, backend/precision/output drift, and post-load tree mutation. Details remain bounded and do not expose untrusted full paths.

## RED matrix

The behavior test must independently prove:

1. exact success using a real P3-installed candidate and private fake expert loaders;
2. missing required model refusal;
3. unexpected model directory refusal;
4. degraded or partial loader refusal;
5. installed model-file mutation refusal;
6. whole-tree/evidence mutation refusal;
7. handoff identity mismatch refusal;
8. policy identity mutation refusal;
9. feature-column order drift refusal;
10. backend or planned-backend drift refusal;
11. precision drift refusal;
12. expert-output-kind drift refusal;
13. invalid or drifted voting weight refusal;
14. post-load mutation refusal through a deterministic test hook.

The independent source contract must prove that the opener and carrier are real production types, no forbidden ambient API is referenced, explicit voting construction is present, P3 verification occurs both before and after loading, exact inventory/error paths are wired, and the carrier cannot be cloned, serialized, or publicly constructed.

## File and scope split

- `src/promotion_inference_v1.rs`: policy, refusal types, carrier, opener orchestration; fewer than 900 changed lines.
- `src/promotion_inference_v1/artifact.rs`: bounded artifact and exact-inventory helpers; fewer than 900 changed lines.
- `src/promotion_inference_v1_tests.rs`: in-module behavior RED/GREEN; fewer than 900 changed lines.
- `tests/promotion_inference_v1_source_contract.rs`: independent production-source ratchets; fewer than 900 changed lines.
- `src/lib.rs`: minimal private module plus approved public reexports only.

No remote source or Cargo action is permitted until the P4A owner explicitly releases the shared window.
