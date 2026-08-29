# GPU Starvation and Residency Design

## Goal

Remove the host-driven starvation that makes the RTX 3090 appear slower than the CPU in NeoEthos SEARCH and model TRAINING, while preserving every seed, ordering rule, numerical result, validity result, and fail-closed financial-truth boundary.

## Confirmed evidence

### SEARCH

- The immutable parent dataset is uploaded once per run, but each generation still performs CPU gene/scenario packing, changing-input H2D, compatibility evaluation, an explicit host wait, full metric-row D2H, and CPU ranking/evolution.
- `population_reduce_kernel` maps one thread to one scenario and walks bars serially. With population 100 and block size 32, the dominant generation kernel launches only four blocks on an RTX 3090 with 82 SMs.
- The compatibility path allocates and clears diagnostic outcome storage that production generation fitness does not need.
- Metrics-only population primitives and native resident ranking/selection/offspring primitives exist, but the high-level production resident-generation owner is missing from the current VPS source snapshot.
- The existing CLI benchmark correctly refuses to run without the sealed broker-financial-truth authority. Performance work must not bypass that gate.

### TRAINING

- The M15 run spends about 88 seconds in CPU data and feature preparation before model training begins.
- Burn training gathers each minibatch on the CPU, constructs/uploads feature and label tensors per batch, recreates class-weight tensors per batch, and reads a scalar loss back before backward.
- Validation tensors are re-uploaded each epoch. HPO trials and the final refit are sequential.
- The controlled production-shape Nsight receipt contains 2,272 H2D copies, 2,874 very short kernels, 136 CUDA module loads, 133 event synchronizations, and only 0.101 MB D2H. Useful kernel time is only about 15 ms of roughly 34.7 seconds wall time.

## Common diagnosis

The shared defect is an insufficient asynchronous GPU work queue. SEARCH manifests it mainly as too few independent blocks plus host-owned generation boundaries; TRAINING manifests it mainly as per-batch tensor construction, synchronization, cold graph/module specialization, and serial jobs. The literal transfer pattern is different, so the two implementations require separate plans and separate parity gates.

## Design principles

1. Change one bounded behavior at a time.
2. Write and observe a failing regression test before production edits.
3. Keep the same data, seed, population, generation count, batch order, and model parameters for every before/after comparison.
4. Treat cold-start and warmed steady-state as separate measurements.
5. Never infer GPU utilization from `nvidia-smi` alone. Archive Nsight Systems, Nsight Compute, CUDA resource, parity, and wall-time receipts.
6. CUDA core count is not a scenario count. Occupancy decisions are based on blocks, warps, waves per SM, registers, shared memory, and measured scheduler activity.
7. Do not weaken the broker-financial-truth gate to obtain a benchmark.

## Sequential implementation

### Stage S1: SEARCH metrics-only compatibility removal

Replace the production generation's diagnostic compatibility evaluation with a one-shot metrics-only result owner. The temporary S1 boundary may still perform one explicit terminal wait and one metric-row D2H because CPU ranking remains unchanged, but it must allocate no outcome ledger, launch no outcome-seed kernel, and read no accepted-trade scalar. This isolates and measures the cost removed before the larger resident-generation integration.

S1 passes only with exact metric parity and an RTX receipt proving zero diagnostic outcome bytes, zero outcome-seed launches, zero accepted-trade D2H, and reduced wall time or a documented neutral result.

### Stage S2: SEARCH fully resident generation

Connect metrics-only evaluation directly to the existing device ranking, selection, deduplication, crossover, mutation, and offspring stores through stream-ordered event ownership. Genes, metrics, scratch, and generation state stay device-resident across all generations. No per-generation host wait, metric D2H, CPU ranking, or CPU evolution is permitted. The host receives only the bounded final result and audited final diagnostics.

S2 must preserve exact generation identities and final selected content for the same seed. If a current CPU algorithm has no exact device implementation, the route fails closed rather than silently crossing to CPU.

### Stage S3: SEARCH occupancy and workload geometry

After S2 parity, profile the unchanged population first. Then increase independent in-flight scenarios only where they are already semantically required, such as folds, costs, robustness treatments, or independent searches. Do not increase GA population merely to make utilization look high. Record block count, waves per SM, achieved occupancy, SM activity, idle-gap distribution, and end-to-end throughput.

### Stage T1: TRAINING fit-resident tensors

Upload contiguous train features, labels, validation features, validation labels, and class weights once per fit. Select/gather minibatches on device. Accumulate train loss on device and read only one train scalar plus one validation scalar per epoch. Keep the exact shuffled batch order and optimizer schedule.

### Stage T2: TRAINING warm graph and concurrent work

Reuse same-shape compiled/module state, remove repeated refit preparation where mathematically identical, and schedule independent model/fold jobs concurrently within a measured VRAM budget. HPO semantics and the final refit remain unchanged unless a separate correctness design explicitly changes them.

## Shared profiler receipt

Each stage archives cold and warm Nsight captures with NVTX phase ranges and reports:

- wall time and useful GPU active time;
- GPU idle-gap maximum and p50/p95;
- kernel count and duration p50/p95;
- H2D/D2H count, bytes, and time;
- blocking synchronization count and time;
- copy/compute overlap;
- grid/block geometry and waves per SM;
- theoretical and achieved occupancy;
- SM, scheduler, DRAM, and PCIe activity;
- exact output parity and input/config identities.

Project targets inside GPU-designated phases are: no unexpected D2H or blocking synchronization in inner loops, no invariant-tensor re-upload after warm-up, transfer time at most 10% of phase wall time, useful GPU duty cycle at least 80%, and idle-gap p95 at most 1 ms. These are NeoEthos acceptance targets, not universal NVIDIA rules.

## Rollback and VPS discipline

The VPS snapshot has no `.git` directory. Before every production edit, record SHA-256 and retain an exact preimage. Apply only a reviewed patch, verify the postimage, and keep a reverse patch. A failed compile, parity test, sanitizer, or profiler gate restores the preimage before any later stage begins.
