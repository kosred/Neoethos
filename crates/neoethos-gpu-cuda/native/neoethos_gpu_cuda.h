#pragma once
#include <cstddef>
#include <cstdint>

#define NEOETHOS_GPU_ABI_VERSION 4u

namespace neoethos::resident_generation_v1 {
struct NeoResidentGenerationAllocationReceiptV1;
struct NeoResidentGenerationPlanV1;
struct NeoResidentGenerationRunV1;
}  // namespace neoethos::resident_generation_v1

namespace neoethos::resident_generation_v2 {
struct NeoResidentGenerationGeneViewV2;
}  // namespace neoethos::resident_generation_v2

namespace neoethos::resident_scoring_novelty_v1 {
struct NeoResidentScoringNoveltyAllocationReceiptV1;
struct NeoResidentScoringNoveltyPlanV1;
struct NeoResidentScoringNoveltyRunV1;
}  // namespace neoethos::resident_scoring_novelty_v1

namespace neoethos::resident_search_generation_v2 {
struct NeoResidentScoringPopulationSourceV2;
}  // namespace neoethos::resident_search_generation_v2

extern "C" {

struct CUctx_st;
struct CUstream_st;
struct CUevent_st;

struct NeoBufferRef {
  std::uint64_t offset;
  std::uint64_t len;
};

struct NeoHandleToken {
  std::uint64_t session_id;
  std::uint32_t backend_id;
  std::uint32_t device_id;
  std::uint64_t generation;
  std::uint32_t buffer_kind;
  std::uint32_t reserved;
};

struct NeoDatasetHeader {
  std::uint32_t abi_version;
  std::uint32_t flags;
  std::uint64_t row_count;
  std::uint32_t feature_count;
  std::int32_t price_scale_exp;
  NeoBufferRef timestamps;
  NeoBufferRef open;
  NeoBufferRef high;
  NeoBufferRef low;
  NeoBufferRef close;
  NeoBufferRef features;
  NeoBufferRef months;
  NeoBufferRef days;
};

struct NeoGeneDescriptor {
  std::uint64_t candidate_id;
  std::uint32_t term_offset;
  std::uint32_t term_count;
  double long_threshold;
  double short_threshold;
  std::int64_t stop_ticks;
  std::int64_t target_ticks;
  double stop_vol_multiplier;
  std::uint32_t flags;
  std::uint64_t reserved;
};

struct NeoScenarioDescriptor {
  std::uint64_t base_candidate_id;
  std::uint64_t scenario_id;
  std::uint64_t rng_counter;
  std::uint64_t window_offset;
  std::uint32_t window_len;
  std::uint32_t scenario_type;
  std::int32_t spread_ticks;
  std::int32_t slippage_ticks;
  std::int64_t commission_micros;
  std::uint32_t perturbation_offset;
  std::uint32_t perturbation_count;
  std::uint64_t reserved;
};

struct NeoTradeOutcome {
  std::uint64_t candidate_id;
  std::uint64_t scenario_id;
  std::uint32_t entry_bar;
  std::uint32_t exit_bar;
  std::uint32_t exit_reason;
  std::int32_t direction;
  std::int64_t pnl_micros;
  std::int64_t equity_after_micros;
  std::uint64_t reserved;
};

struct NeoMetrics {
  std::uint64_t candidate_id;
  std::uint64_t scenario_id;
  double net_profit;
  double max_drawdown;
  double sharpe;
  double profit_factor;
  double win_rate;
  std::uint64_t trade_count;
  double monthly_target_hit_rate;
  std::uint32_t flags;
  std::uint32_t reserved;
};

struct NeoPropFirmState {
  std::int64_t equity_micros;
  std::int64_t peak_equity_micros;
  std::int64_t day_start_equity_micros;
  std::int64_t month_start_equity_micros;
  std::int64_t current_day_id;
  std::int64_t current_month_id;
  std::uint32_t trading_days;
  std::uint32_t flags;
};

struct NeoFirstHitEvent {
  std::uint32_t entry_bar;
  std::uint32_t last_bar;
  std::int32_t direction;
  std::int32_t precedence;
  double stop_price;
  double target_price;
};

struct NeoFirstHitResult {
  std::int32_t exit_bar;
  std::int32_t exit_reason;
};

/// Fixed-width settings shared by the Prototype B population entry points.
/// Mirrors `neoethos_gpu_contracts::device::NeoPopulationSettings` exactly.
struct NeoPopulationSettings {
  std::uint32_t abi_version;
  std::uint32_t flags;
  std::uint32_t max_hold_bars;
  std::uint32_t min_hold_bars;
  std::uint32_t max_trades_per_day;
  std::uint32_t month_capacity;
  std::int64_t gap_threshold_ms;
  double initial_equity;
  double pip_value;
  double spread_pips;
  double commission_per_trade;
  double pip_value_per_lot;
  double swap_long_pips_per_day;
  double swap_short_pips_per_day;
  double pnl_conversion_fee_rate;
  double risk_per_trade_min;
  double risk_per_trade_max;
  double high_quality_confidence;
  double adaptive_rr;
  // Trailing stop, so the kernel simulates the strategy the CPU does.
  std::uint32_t trailing_enabled;
  std::uint32_t _trailing_pad;
  double trailing_atr_multiplier;
  double trailing_be_trigger_r;
  double trailing_min_lock_pips;
  /* Spread in pips per liquidity window, resolved from the entry bar's UTC
     hour. When no session profile is configured the host writes spread_pips
     into all three, so the lookup returns the scalar and the arithmetic is
     bit-identical to the single-value form. */
  double spread_pips_asian;
  double spread_pips_overlap;
  double spread_pips_late_ny;
};

struct NeoPopulationEvent {
  std::uint64_t candidate_id;
  std::uint64_t scenario_id;
  std::uint32_t entry_bar;
  std::uint32_t last_bar;
  std::int32_t direction;
  std::uint32_t precedence;
  double stop_price;
  double target_price;
  double entry_price;
};

struct NeoPopulationOutcome {
  std::uint64_t candidate_id;
  std::uint64_t scenario_id;
  std::int32_t exit_bar;
  std::int32_t exit_reason;
  std::int32_t entry_bar;
  std::int32_t pad;
  double mfe;
  double mae;
  // Price the position actually closed at. Rebuilding it from the exit reason
  // works only while the levels are fixed at entry; a trailing stop moves.
  double exit_price;
  double pnl;
  double r_multiple;
};

struct NeoPopulationMetricRow {
  std::uint64_t candidate_id;
  std::uint64_t scenario_id;
  double values[11];
};

struct NeoPopulationCounters {
  std::uint64_t event_count;
  std::uint64_t accepted_trade_count;
  std::uint64_t kernel_submissions;
  std::uint64_t synchronization_events;
  std::uint64_t dataset_upload_bytes;
  std::uint64_t gene_upload_bytes;
  std::uint64_t scenario_upload_bytes;
  std::uint64_t compact_readback_bytes;
  std::uint64_t full_readback_bytes;
  std::uint64_t reserved[3];
};

/// Host-side description of one logical dataset upload. All arrays are
/// `header.row_count` long except `indicators` (feature-major
/// `feature_count * row_count`), `smc_rows` (`row_count * 11`) and the optional
/// adaptive stop base, whose length is stated separately.
///
/// `indicators` IS AND STAYS FEATURE-MAJOR. The Prototype B walk reads it
/// bar-major, and `upload_dataset` transposes it once on the device into a
/// buffer of its own; the feature-major staging copy is freed before that call
/// returns. This is stated here because the layout the device prefers changed
/// and this contract did not — the CPU oracle, the parity fixtures and
/// prototypes A and C all still build and consume feature-major, and a caller
/// that "helpfully" pre-transposes would silently evaluate a transposed matrix.
struct NeoPopulationDatasetView {
  NeoDatasetHeader header;
  const double* close;
  const double* high;
  const double* low;
  const double* indicators;
  const std::int64_t* months;
  const std::int64_t* days;
  const std::int64_t* timestamps;
  const std::int8_t* smc_rows;
  // f64 on purpose: the canonical adaptive-at-entry stop distance is f64 on the
  // host, and narrowing it here would silently break exact parity.
  const double* adaptive_base_pips;
  std::size_t adaptive_base_pips_len;
};

/// Immutable parent bytes for the V1 resident route. Unlike the compatibility
/// dataset view, it cannot contain view-local adaptive settings.
struct NeoPopulationParentDatasetV1 {
  NeoDatasetHeader header;
  const double* close;
  const double* high;
  const double* low;
  const double* indicators_feature_major;
  const std::int64_t* months;
  const std::int64_t* days;
  const std::int64_t* timestamps;
  const std::int8_t* smc_rows;
};

#define NEO_POPULATION_PARENT_OWNED_V1 0u
#define NEO_POPULATION_PARENT_BORROWED_RESIDENT_V3 1u
#define NEO_POPULATION_STREAM_OWNED 0u
#define NEO_POPULATION_STREAM_BORROWED 1u

/// Immediate gpu-cuda-owned bind descriptor for a sealed V3 resident store.
/// The native session borrows every listed parent pointer and the admitted run
/// stream. Rust retains their opaque owners until a consumer completion event
/// proves that no queued population read can still reach them.
struct NeoPopulationResidentFeatureStoreV3 {
  std::uint32_t abi_version;
  std::uint32_t selected_device_ordinal;
  std::uint64_t row_count;
  std::uint32_t feature_count;
  std::uint32_t smc_slots;
  std::uint16_t compute_capability_major;
  std::uint16_t compute_capability_minor;
  std::uint32_t reserved;
  std::uint64_t packed_validity_bytes;
  const double* close;
  const double* high;
  const double* low;
  const double* indicators_bar_major;
  const std::uint8_t* indicators_validity_u4;
  const std::int64_t* months;
  const std::int64_t* days;
  const std::int64_t* timestamps;
  const std::int8_t* smc_rows;
  CUctx_st* admitted_primary_context;
  CUstream_st* admitted_run_stream;
  CUevent_st* ready_event;
  std::uint8_t device_uuid[16];
  std::uint8_t admission_identity_sha256[32];
  std::uint8_t canonical_content_merkle[32];
  std::uint64_t allocator_context_reserve_bytes;
  std::uint8_t run_stream_process_token_v3[32];
};

static_assert(sizeof(NeoPopulationResidentFeatureStoreV3) == 256,
              "resident feature-store V3 ABI changed");
static_assert(alignof(NeoPopulationResidentFeatureStoreV3) == 8,
              "resident feature-store V3 alignment changed");
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3,
                       allocator_context_reserve_bytes) == 216,
              "resident feature-store reserve offset changed");
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3,
                       run_stream_process_token_v3) == 224,
              "resident feature-store stream-token offset changed");
static_assert(offsetof(NeoPopulationResidentFeatureStoreV3,
                       run_stream_process_token_v3) + 32 ==
                  sizeof(NeoPopulationResidentFeatureStoreV3),
              "resident feature-store V3 has unratcheted trailing padding");

#define NEO_POPULATION_VIEW_FULL 0u
#define NEO_POPULATION_VIEW_CONTIGUOUS_RANGE 1u
#define NEO_POPULATION_VIEW_ORDERED_INDICES 2u
#define NEO_POPULATION_TIMESTAMP_CANONICAL 0u
#define NEO_POPULATION_TIMESTAMP_DISABLED_INDEX_DELTA 1u

/// One view over an already uploaded immutable parent. Full/range use scalar
/// offsets. Only ordered views supply a compact u64 index map. Adaptive values
/// are view-local and must cover exactly `row_count` rows when present.
struct NeoPopulationEvaluationViewV1 {
  std::uint32_t abi_version;
  std::uint32_t view_kind;
  std::uint64_t parent_row_count;
  std::uint64_t range_start;
  std::uint64_t row_count;
  const std::uint64_t* ordered_indices;
  std::size_t ordered_index_count;
  std::uint32_t timestamp_mode;
  const double* adaptive_base_pips;
  std::size_t adaptive_base_pips_len;
};

/// Canonical, view-local adaptive-stop base recipe for one resident V3 parent.
///
/// This descriptor contains control scalars only. High/low/close are read from
/// the already-bound resident parent on its admitted stream, and the resulting
/// f64 base series is written directly to the population session's retained
/// adaptive buffer. No host price/base pointer is part of this ABI.
struct NeoResidentAdaptiveBaseRequestV1 {
  std::uint32_t abi_version;
  std::uint32_t view_kind;
  std::uint64_t parent_row_count;
  std::uint64_t view_start;
  std::uint64_t view_row_count;
  std::uint32_t vol_window;
  std::uint32_t vol_horizon_bars;
  std::uint32_t tail_window;
  std::uint32_t tail_quantile_index;
  std::uint64_t tail_step;
  std::uint64_t tail_max_bars;
  double pip_size;
  double stop_k_vol;
  double stop_k_tail;
  double meta_label_min_dist;
};

/// Exact transfer and synchronization facts for one native resident session.
/// Metric rows and diagnostics are intermediate/full-population D2H classes;
/// neither is a compact-final result. The accepted-trade total is the scalar
/// control-plane D2H performed by the current wait boundary.
struct NeoPopulationResidencyCountersV1 {
  std::uint64_t parent_upload_count;
  std::uint64_t parent_upload_bytes;
  std::uint64_t view_binding_count;
  std::uint64_t full_binding_count;
  std::uint64_t range_binding_count;
  std::uint64_t ordered_binding_count;
  std::uint64_t ordered_index_upload_bytes;
  std::uint64_t adaptive_upload_bytes;
  std::uint64_t stream_creation_count;
  std::uint64_t explicit_synchronization_count;
  std::uint64_t metric_rows_readback_count;
  std::uint64_t metric_rows_readback_rows;
  std::uint64_t metric_rows_readback_bytes;
  std::uint64_t diagnostic_readback_count;
  std::uint64_t diagnostic_readback_rows;
  std::uint64_t diagnostic_readback_bytes;
  std::uint64_t accepted_trade_total_readback_count;
  std::uint64_t accepted_trade_total_readback_bytes;
};

/// Fixed-width control-plane receipt for one metrics-only launch whose metric
/// rows remain resident on the population session's native CUDA stream.
///
/// This value is not a device pointer and cannot authorize a detached read. The
/// Rust owner binds it to the borrowed `PopulationSession`; the next resident
/// GPU stage must consume the event as a stream dependency. Diagnostic outcome
/// and accepted-total extents are explicitly zero in this mode.
/// `total_device_bytes` is the exact sum of the three evaluation workspaces and
/// the scenario SoA named below. Immutable parent/gene buffers are charged by
/// their own run-level receipt rather than silently folded into this sub-plan.
struct NeoPopulationResidentMetricsHandleV1 {
  std::uint32_t abi_version;
  std::uint32_t reserved;
  std::uint64_t event_id;
  std::uint64_t scenario_count;
  std::uint64_t month_capacity;
  std::uint64_t metric_rows_bytes;
  std::uint64_t monthly_pnls_bytes;
  std::uint64_t month_start_equities_bytes;
  std::uint64_t scenario_descriptor_bytes;
  std::uint64_t total_device_bytes;
  std::uint64_t outcome_bytes;
  std::uint64_t accepted_trade_total_bytes;
};

/// One bounded terminal result from a strict metrics-only launch. This is the
/// sole host readback permitted at the end of the current one-scenario V1
/// research seam: no outcome, diagnostic, or accepted-trade buffer exists.
struct NeoPopulationTerminalCompactResultV1 {
  std::uint32_t abi_version;
  std::uint32_t reserved;
  std::uint64_t event_id;
  std::uint64_t scenario_count;
  NeoPopulationMetricRow metric_row;
  std::uint64_t terminal_synchronization_count;
  std::uint64_t terminal_readback_count;
  std::uint64_t terminal_readback_rows;
  std::uint64_t terminal_readback_bytes;
};

/// Metadata for the single bounded host transfer that terminates a strict
/// metrics-only launch. The metric rows themselves use NeoPopulationReadback;
/// this fixed-width value proves the synchronized event and exact transfer.
struct NeoPopulationHostMetricsResultV1 {
  std::uint32_t abi_version;
  std::uint32_t reserved;
  std::uint64_t event_id;
  std::uint64_t scenario_count;
  std::uint64_t terminal_synchronization_count;
  std::uint64_t terminal_readback_count;
  std::uint64_t terminal_readback_rows;
  std::uint64_t terminal_readback_bytes;
};

/// Immutable identity of the physical CUDA device selected by one population
/// session. The UUID and name are fixed-width raw bytes so the Rust/C ABI has
/// no allocator, locale, or null-termination dependency.
struct NeoPopulationDeviceIdentityV1 {
  std::uint32_t selected_device_ordinal;
  std::uint32_t compute_capability_major;
  std::uint32_t compute_capability_minor;
  std::uint32_t multiprocessor_count;
  std::uint64_t total_global_memory_bytes;
  std::int32_t pci_domain_id;
  std::int32_t pci_bus_id;
  std::int32_t pci_device_id;
  std::uint8_t uuid[16];
  std::uint8_t name[256];
};

/// Canonical gene batch. `descriptors` carries identity and thresholds; all
/// floating signal inputs (CSR weights, stop/target/multiplier arrays, SMC
/// weights and gate) retain the exact f64 host precision through device math.
struct NeoPopulationGeneView {
  const NeoGeneDescriptor* descriptors;
  std::size_t count;
  const std::int32_t* offsets;
  const std::int32_t* indices;
  const double* weights;
  std::size_t term_count;
  const double* stop_pips;
  const double* target_pips;
  const double* stop_vol_multipliers;
  const std::int8_t* smc_flags;
  const double* smc_weights;
  double gate_threshold;
  std::uint32_t smc_gate_disabled;
};

/// The work list. `count` is the number of THREADS the walk launches and the
/// number of metric rows `read_metrics` returns — it is NOT required to equal
/// the uploaded gene count.
///
/// It used to be. That equality is why a screen wanting 101 treatments of one
/// gene had to clone the gene 101 times, and why the Monte-Carlo pass staged
/// 17 400 gene clones on the host and sent them in six launches. Each descriptor
/// now carries its own gene index, window, costs and perturbation counter, so
/// 174 genes and 17 574 scenarios go up together in one launch.
///
/// Every field of `NeoScenarioDescriptor` is read by the device. The contract:
///
///   * `base_candidate_id` — index into the uploaded gene array. MUST be inside
///     it; the upload refuses anything else, because an out-of-range value is an
///     out-of-bounds read of thresholds and CSR offsets that still produces a
///     plausible metric row.
///   * `scenario_id` — opaque, returned in the metric row so a mixed array can
///     be demultiplexed without relying on position.
///   * `window_offset` / `window_len` — bars `[offset, offset+len)`. A `len` of
///     0 means "to the end of the series". `offset = 0, len = bars` is the whole
///     series and is bit-identical to the pre-scenario walk.
///   * `scenario_type` — 0 base, 1 device-perturbed Monte-Carlo, 2 cost.
///   * `spread_ticks` / `slippage_ticks` — MILLIPIPS (thousandths of a pip).
///     `-1` in `spread_ticks` means "no override, use the settings' per-bar
///     spread"; a sentinel rather than 0 because charging NO spread is a
///     legitimate thing to ask. `slippage_ticks` has no sentinel: 0 means none,
///     and it is the only value the CPU engine can mirror.
///   * `commission_micros` — millionths of one account-currency unit per lot.
///     `-1` means "no override".
///   * `rng_counter` — the perturbation stream, read only for type 1.
///   * `perturbation_offset` / `perturbation_count` / `reserved` — still unused.
///
/// The integer cost fields are converted by DIVISION by their scale (1000 and
/// 1e6), never by multiplication by a reciprocal, because the scales are exactly
/// representable and their reciprocals are not. The host refuses to build a
/// descriptor whose cost does not survive that round trip rather than rounding
/// it, so a spread the descriptor cannot carry is an error the operator reads
/// and never a launch that quietly charged a different number.
struct NeoPopulationScenarioView {
  const NeoScenarioDescriptor* descriptors;
  std::size_t count;
};

/// One row per SCENARIO, in scenario order, each carrying both the gene's
/// `candidate_id` and its own `scenario_id`. `capacity` must therefore cover the
/// uploaded scenario count, which is the gene count only when the caller
/// uploaded the identity descriptor array.
struct NeoPopulationReadback {
  NeoPopulationMetricRow* rows;
  std::size_t capacity;
  std::size_t* written;
};

/// Diagnostic-only readback of the device outcome stream. It exists so a parity
/// failure on rented hardware can be localized; it must never run inside a timed
/// benchmark repetition.
///
/// `capacity` is a RANGE REQUEST, not a minimum. The outcome array is
/// scenario-major with `kMaxTradesPerCandidate` slots each, so the first
/// `capacity` records are exactly the trades of the first
/// `capacity / kMaxTradesPerCandidate` scenarios, and `written` says how many
/// were copied. It used to be an exact-fit requirement, which forced the host to
/// allocate for the whole array — 163.8 M records at a 20 000-scenario launch.
///
/// `events` MAY BE NULL. Nothing emits events any more (the reduce opens
/// positions from the signal), so a non-null pointer is merely memset to zero;
/// pass null to skip 56 B per slot of host allocation that carries nothing.
/// `outcomes` and `written` are required.
struct NeoPopulationDiagnosticReadback {
  NeoPopulationEvent* events;
  NeoPopulationOutcome* outcomes;
  std::size_t capacity;
  std::size_t* written;
};

struct NeoCudaPopulationSession;

#define NEO_POPULATION_STATUS_OK 0
#define NEO_POPULATION_STATUS_UNSUPPORTED (-1)
#define NEO_POPULATION_STATUS_NULL_SESSION (-30)
#define NEO_POPULATION_STATUS_ABI_MISMATCH (-31)
#define NEO_POPULATION_STATUS_INVALID_ARGUMENT (-32)
#define NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE (-33)
#define NEO_POPULATION_STATUS_ALLOCATION_FAILED (-34)
#define NEO_POPULATION_STATUS_TRANSFER_FAILED (-35)
#define NEO_POPULATION_STATUS_LAUNCH_FAILED (-36)
#define NEO_POPULATION_STATUS_EVENT_CAPACITY (-37)
#define NEO_POPULATION_STATUS_MISSING_UPLOAD (-38)
#define NEO_POPULATION_STATUS_READBACK_CAPACITY (-39)
#define NEO_POPULATION_STATUS_SYNC_FAILED (-40)
#define NEO_POPULATION_STATUS_UNKNOWN_EVENT (-41)
#define NEO_POPULATION_STATUS_DATASET_REUPLOAD (-42)
#define NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH (-43)
#define NEO_POPULATION_STATUS_WORKSPACE_PLAN_MISMATCH (-44)
#define NEO_POPULATION_STATUS_STRICT_RESIDENT_IN_FLIGHT (-45)
#define NEO_POPULATION_STATUS_STRICT_RESIDENT_POISONED (-46)
#define NEO_POPULATION_STATUS_ADAPTIVE_BASE_DEGENERATE (-47)
#define NEO_POPULATION_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN (-48)
#define NEO_POPULATION_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN (-49)

#define NEO_CUDA_DEVICE_PROBE_OK 0
#define NEO_CUDA_DEVICE_PROBE_INVALID_OUTPUT (-50)
#define NEO_CUDA_DEVICE_PROBE_ADAPTER_UNAVAILABLE (-51)

/// `max_events` is VESTIGIAL: it must be non-zero, and the device ignores it.
///
/// It once sized a per-session event buffer. Nothing has allocated that buffer
/// since the reduce started opening positions from the signal directly, and the
/// only kernel that filled it is not launched. The parameter is kept so the ABI
/// and every caller stay unchanged; it is documented here rather than removed
/// so no future sizing arithmetic budgets device memory for it again.
NeoCudaPopulationSession* neoethos_gpu_cuda_population_create(
    std::uint32_t abi_version,
    std::int32_t device,
    std::size_t max_events,
    std::int32_t* status);
std::int32_t neoethos_gpu_cuda_population_upload_dataset(
    NeoCudaPopulationSession* session,
    const NeoPopulationDatasetView* dataset);
std::int32_t neoethos_gpu_cuda_population_upload_parent_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationParentDatasetV1* parent);
NeoCudaPopulationSession* neoethos_gpu_cuda_population_bind_resident_feature_store_v3(
    const NeoPopulationResidentFeatureStoreV3* resident,
    std::int32_t* status);
std::int32_t neoethos_gpu_cuda_population_bind_view_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationEvaluationViewV1* view);
std::int32_t neoethos_gpu_cuda_population_bind_resident_adaptive_view_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationEvaluationViewV1* view,
    const NeoResidentAdaptiveBaseRequestV1* request);
#if defined(NEOETHOS_CUDA_DEVICE_FIXTURES_V2)
std::int32_t neoethos_gpu_cuda_population_copy_resident_adaptive_base_fixture_v1(
    NeoCudaPopulationSession* session,
    double* host_values,
    std::size_t value_count);
#endif
std::int32_t neoethos_gpu_cuda_population_read_residency_counters_v1(
    NeoCudaPopulationSession* session,
    NeoPopulationResidencyCountersV1* counters);
std::int32_t neoethos_gpu_cuda_population_read_device_identity_v1(
    NeoCudaPopulationSession* session,
    NeoPopulationDeviceIdentityV1* identity);
std::int32_t neoethos_gpu_cuda_population_upload_genes(
    NeoCudaPopulationSession* session,
    const NeoPopulationGeneView* genes);
std::int32_t neoethos_gpu_cuda_population_upload_scenarios(
    NeoCudaPopulationSession* session,
    const NeoPopulationScenarioView* scenarios);
std::int32_t neoethos_gpu_cuda_population_upload_resident_scenarios_v2(
    NeoCudaPopulationSession* session,
    const NeoPopulationScenarioView* scenarios,
    std::uint64_t planned_population);
std::int32_t neoethos_gpu_cuda_population_create_resident_generation_run_v2(
    NeoCudaPopulationSession* session,
    const neoethos::resident_generation_v1::NeoResidentGenerationPlanV1* plan,
    neoethos::resident_generation_v1::NeoResidentGenerationAllocationReceiptV1* allocation,
    neoethos::resident_generation_v1::NeoResidentGenerationRunV1** run);
std::int32_t neoethos_gpu_cuda_population_release_resident_generation_run_v2(
    NeoCudaPopulationSession* session,
    neoethos::resident_generation_v1::NeoResidentGenerationRunV1* run);
std::int32_t neoethos_gpu_cuda_population_create_unbound_resident_scoring_run_v2(
    NeoCudaPopulationSession* session,
    const neoethos::resident_scoring_novelty_v1::NeoResidentScoringNoveltyPlanV1* plan,
    neoethos::resident_scoring_novelty_v1::NeoResidentScoringNoveltyAllocationReceiptV1*
        allocation,
    neoethos::resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1** run);
std::int32_t neoethos_gpu_cuda_population_release_resident_scoring_run_v2(
    NeoCudaPopulationSession* session,
    neoethos::resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1* run);
std::int32_t neoethos_gpu_cuda_population_export_resident_scoring_source_v2(
    NeoCudaPopulationSession* session,
    const NeoPopulationResidentMetricsHandleV1* resident_metrics,
    std::uint64_t expected_population,
    std::uint64_t expected_feature_count,
    std::uint32_t expected_max_terms,
    neoethos::resident_search_generation_v2::NeoResidentScoringPopulationSourceV2* source);
std::int32_t neoethos_gpu_cuda_population_finish_resident_scoring_source_v2(
    NeoCudaPopulationSession* session,
    const NeoPopulationResidentMetricsHandleV1* resident_metrics);
std::int32_t neoethos_gpu_cuda_population_enqueue_resident_gene_metrics_v2(
    NeoCudaPopulationSession* session,
    const neoethos::resident_generation_v2::NeoResidentGenerationGeneViewV2* genes,
    const NeoPopulationSettings* settings,
    NeoPopulationResidentMetricsHandleV1* resident_metrics,
    NeoPopulationCounters* counters);
std::int32_t neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationSettings* settings,
    NeoPopulationResidentMetricsHandleV1* resident_metrics,
    NeoPopulationCounters* counters);
std::int32_t neoethos_gpu_cuda_population_consume_terminal_compact_result_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationResidentMetricsHandleV1* resident_metrics,
    NeoPopulationTerminalCompactResultV1* compact_result);
std::int32_t neoethos_gpu_cuda_population_consume_host_metrics_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationResidentMetricsHandleV1* resident_metrics,
    NeoPopulationReadback* readback,
    NeoPopulationHostMetricsResultV1* result);
std::int32_t neoethos_gpu_cuda_population_abandon_resident_metrics_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationResidentMetricsHandleV1* resident_metrics);
/// Compatibility/DeviceParityOnly. Strict resident production uses the
/// metrics-only enqueue above and never waits or reads metric rows on the host.
std::int32_t neoethos_gpu_cuda_population_b_evaluate(
    NeoCudaPopulationSession* session,
    const NeoPopulationSettings* settings,
    std::uint64_t* event_id,
    NeoPopulationCounters* counters);
/// Compatibility/DeviceParityOnly host synchronization.
std::int32_t neoethos_gpu_cuda_population_wait(
    NeoCudaPopulationSession* session,
    std::uint64_t event_id);
/// Compatibility/DeviceParityOnly full metric-row D2H readback.
std::int32_t neoethos_gpu_cuda_population_read_metrics(
    NeoCudaPopulationSession* session,
    NeoPopulationReadback* readback);
/// Compatibility/DeviceParityOnly diagnostic D2H readback.
std::int32_t neoethos_gpu_cuda_population_read_diagnostics(
    NeoCudaPopulationSession* session,
    NeoPopulationDiagnosticReadback* readback);
void neoethos_gpu_cuda_population_destroy(NeoCudaPopulationSession* session);
int32_t neoethos_gpu_cuda_population_destroy_terminal_checked_v2(
    NeoCudaPopulationSession* session);

std::uint32_t neoethos_gpu_cuda_abi_version();
std::int32_t neoethos_gpu_cuda_runtime_available();
/// Fallible exact CUDA enumeration. CUDA success writes the exact count,
/// including zero. CUDA errors are returned unchanged and never become zero.
std::int32_t neoethos_gpu_cuda_probe_device_count_v1(std::uint32_t* out_count);
/// Number of visible CUDA devices, or 0 when the runtime is unavailable.
std::int32_t neoethos_gpu_cuda_device_count();
std::uint64_t neoethos_gpu_cuda_device_free_memory(std::int32_t device);
std::int32_t neoethos_gpu_cuda_smoke(const std::uint32_t* input,
                                     std::uint32_t* output,
                                     std::size_t len);
std::int32_t neoethos_gpu_cuda_warp_first_hit(
    const double* highs,
    const double* lows,
    std::size_t rows,
    const NeoFirstHitEvent* events,
    NeoFirstHitResult* results,
    std::size_t event_count);
}
