# GPU Rollout Report (forex-models)

## Current status

| Model family | GPU training/execution status | Precision status | Notes |
|---|---|---|---|
| Burn deep models (`mlp`, `nbeats`, `tide`, `tabnet`, `kan`, `transformer`, `patchtst`, `timesnet`) | **Partial** | **Partial** | GPU device policy + runtime backend wiring exists, with precision policy recorded (`training_precision`, `training_precision_reason`). |
| RL (`dqn`) | **Partial** | **Partial** | CUDA path exists; runtime now surfaces requested precision mismatch as degraded reason, but precision is not yet persisted as first-class training metadata like Burn. |
| Statistical meta (`elasticnet`, `bayes_logit`, etc.) | **CPU-only** | **CPU-only** | Runtime explicitly reports CPU fallback when GPU is requested. |
| Adaptive (`online_pa`, `online_hoeffding`) | **CPU-only** | **CPU-only** | Runtime backend remains CPU and reports degraded reason on GPU requests. |
| Evolutionary (`neuro_evo`, `neat`, `genetic`) | **CPU-only/CPU-fallback** | **CPU-only** | Runtime backend constants and fallback backends are CPU. |
| Tree models (LightGBM/XGBoost/CatBoost/Sklears) | **Mixed / backend-dependent** | **Mixed** | Device policy normalization/metadata improvements landed, but no unified precision-tier contract across all trees yet. |
| Anomaly (`isolation_forest`) | **Mostly CPU-fallback** | **CPU-only** | Runtime degraded reporting is in place; full GPU path not uniformly present. |

## What changed now (start from this report)

1. Added env-controlled model support flags for Burn precision gating:
   - `FOREX_BURN_MODEL_SUPPORTS_BF16` (default: `true`)
   - `FOREX_BURN_MODEL_SUPPORTS_FP8` (default: `false`)
   - `FOREX_BURN_MODEL_SUPPORTS_BF4` (default: `false`)
2. Added normalized precision-policy resolution path via runtime capabilities:
   - model-scoped precision key: `FOREX_BOT_<MODEL>_TRAIN_PRECISION`
   - fallback keys: `FOREX_BOT_TRAIN_PRECISION`, then `FOREX_TRAIN_PRECISION`
3. This keeps current safe defaults while allowing staged enablement per deployment environment.
4. RL runtime now appends a precision-unavailable degraded reason when non-fp32 precision is requested (first parity step with Burn precision policy surface).

## Next implementation steps (priority)

1. **RL parity with Burn metadata**: persist `training_precision` and `training_precision_reason` in RL training artifacts/metadata.
2. **Tree precision contract**: define precision capabilities per tree backend and persist effective precision in runtime metadata.
3. **CPU-only families** (statistical/adaptive/evolutionary/anomaly):
   - keep truthful degraded reasons,
   - add explicit `effective_precision=fp32` in runtime metadata for consistency.
4. Add an integration test matrix (CPU / CUDA / GPU-requested-on-CPU) for all families.
