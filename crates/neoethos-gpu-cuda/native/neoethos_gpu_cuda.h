#pragma once
#include <cstddef>
#include <cstdint>

#define NEOETHOS_GPU_ABI_VERSION 1u

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

std::uint32_t neoethos_gpu_cuda_abi_version();
std::int32_t neoethos_gpu_cuda_runtime_available();
std::int32_t neoethos_gpu_cuda_smoke(const std::uint32_t* input,
                                     std::uint32_t* output,
                                     std::size_t len);
}
