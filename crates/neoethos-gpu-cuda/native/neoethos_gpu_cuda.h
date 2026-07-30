#pragma once
#include <cstddef>
#include <cstdint>

#define NEOETHOS_GPU_ABI_VERSION 2u

extern "C" {

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
  float long_threshold;
  float short_threshold;
  std::int64_t stop_ticks;
  std::int64_t target_ticks;
  float stop_vol_multiplier;
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
  float stop_price;
  float target_price;
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
struct NeoPopulationDatasetView {
  NeoDatasetHeader header;
  const double* close;
  const double* high;
  const double* low;
  const float* indicators;
  const std::int64_t* months;
  const std::int64_t* days;
  const std::int64_t* timestamps;
  const std::int8_t* smc_rows;
  // f64 on purpose: the canonical adaptive-at-entry stop distance is f64 on the
  // host, and narrowing it here would silently break exact parity.
  const double* adaptive_base_pips;
  std::size_t adaptive_base_pips_len;
};

/// Canonical gene batch. `descriptors` carries identity and thresholds; the
/// f64 stop/target/multiplier arrays carry the exact host precision that the
/// canonical CPU semantics require, so no ticks-to-pips rounding is introduced.
struct NeoPopulationGeneView {
  const NeoGeneDescriptor* descriptors;
  std::size_t count;
  const std::int32_t* offsets;
  const std::int32_t* indices;
  const float* weights;
  std::size_t term_count;
  const double* stop_pips;
  const double* target_pips;
  const double* stop_vol_multipliers;
  const std::int8_t* smc_flags;
  const float* smc_weights;
  float gate_threshold;
  std::uint32_t smc_gate_disabled;
};

struct NeoPopulationScenarioView {
  const NeoScenarioDescriptor* descriptors;
  std::size_t count;
};

struct NeoPopulationReadback {
  NeoPopulationMetricRow* rows;
  std::size_t capacity;
  std::size_t* written;
};

/// Diagnostic-only readback of the device event/outcome streams. It exists so a
/// parity failure on rented hardware can be localized; it must never run inside
/// a timed benchmark repetition.
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

NeoCudaPopulationSession* neoethos_gpu_cuda_population_create(
    std::uint32_t abi_version,
    std::int32_t device,
    std::size_t max_events,
    std::int32_t* status);
std::int32_t neoethos_gpu_cuda_population_upload_dataset(
    NeoCudaPopulationSession* session,
    const NeoPopulationDatasetView* dataset);
std::int32_t neoethos_gpu_cuda_population_upload_genes(
    NeoCudaPopulationSession* session,
    const NeoPopulationGeneView* genes);
std::int32_t neoethos_gpu_cuda_population_upload_scenarios(
    NeoCudaPopulationSession* session,
    const NeoPopulationScenarioView* scenarios);
std::int32_t neoethos_gpu_cuda_population_b_evaluate(
    NeoCudaPopulationSession* session,
    const NeoPopulationSettings* settings,
    std::uint64_t* event_id,
    NeoPopulationCounters* counters);
std::int32_t neoethos_gpu_cuda_population_wait(
    NeoCudaPopulationSession* session,
    std::uint64_t event_id);
std::int32_t neoethos_gpu_cuda_population_read_metrics(
    NeoCudaPopulationSession* session,
    NeoPopulationReadback* readback);
std::int32_t neoethos_gpu_cuda_population_read_diagnostics(
    NeoCudaPopulationSession* session,
    NeoPopulationDiagnosticReadback* readback);
void neoethos_gpu_cuda_population_destroy(NeoCudaPopulationSession* session);

std::uint32_t neoethos_gpu_cuda_abi_version();
std::int32_t neoethos_gpu_cuda_runtime_available();
/// Number of visible CUDA devices, or 0 when the runtime is unavailable.
std::int32_t neoethos_gpu_cuda_device_count();
std::uint64_t neoethos_gpu_cuda_device_free_memory(std::int32_t device);
std::int32_t neoethos_gpu_cuda_smoke(const std::uint32_t* input,
                                     std::uint32_t* output,
                                     std::size_t len);
std::int32_t neoethos_gpu_cuda_warp_first_hit(
    const float* highs,
    const float* lows,
    std::size_t rows,
    const NeoFirstHitEvent* events,
    NeoFirstHitResult* results,
    std::size_t event_count);
}