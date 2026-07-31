// Persistent native-CUDA Prototype B population engine.
//
// One session owns one non-default stream, one logical dataset upload and every
// device workspace. `evaluate` runs the complete canonical chain on that
// stream: signal synthesis, causal entry emission, warp-cooperative first-hit
// search and the exact cost/sizing/metric reduction. The host boundary is the
// compact metric readback; no candidate-dependent work returns to the CPU.
//
// The semantics reproduced here are the canonical ones expressed by the
// validation oracle in `prototype_population_oracle.rs`. Any divergence is a
// correctness failure, not a tuning opportunity.

#include <cstdio>
#include <cstdlib>
#include "neoethos_gpu_cuda.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstring>
#include <new>

namespace {

constexpr int kDirectionLong = 1;
constexpr int kDirectionShort = -1;
constexpr int kExitNone = 0;
constexpr int kExitStop = 1;
constexpr int kExitTarget = 2;
constexpr int kExitMaxHold = 3;
constexpr int kExitGap = 4;
constexpr unsigned kPrecedenceStopFirst = 0u;
constexpr std::uint32_t kFlagRiskBasedSizing = 1u;
constexpr int kSmcSlots = 11;
constexpr int kEmitBlock = 256;
constexpr int kReduceBlock = 128;
// Trade slots per candidate. Measured need is ~3 000 trades per gene over
// 439 315 bars; this leaves room for the densest genes the search produces
// without sizing device memory by candidate entries, of which there are a
// hundred times more.
constexpr unsigned long long kMaxTradesPerCandidate = 8192ull;
constexpr int kSignalBlock = 256;

// Priority inside a single bar: gap beats stop, stop beats target, target beats
// the max-hold exit. This mirrors the canonical resolver, where a gap drains
// every active event before level checks and the max-hold sweep only fires for
// events that are still active.
__host__ __device__ inline int exit_priority(int reason) {
  switch (reason) {
    case kExitGap:
      return 0;
    case kExitStop:
      return 1;
    case kExitTarget:
      return 2;
    case kExitMaxHold:
      return 3;
    default:
      return INT_MAX;
  }
}

__device__ inline double guarded_pip(double pip_value) {
  return (fabs(pip_value) < 1.0e-12) ? 1.0e-12 : pip_value;
}

__device__ inline double sanitize(double value) {
  return isfinite(value) ? value : 0.0;
}

struct DeviceDataset {
  const double* close;
  const double* high;
  const double* low;
  const float* indicators;
  const long long* months;
  const long long* days;
  const long long* timestamps;
  const signed char* smc_rows;
  const double* adaptive_base_pips;
  int has_adaptive_base;
  int bars;
  int feature_count;
};

struct DeviceGenes {
  const unsigned long long* candidate_ids;
  const int* offsets;
  const int* indices;
  const float* weights;
  const float* long_thresholds;
  const float* short_thresholds;
  const double* stop_pips;
  const double* target_pips;
  const double* stop_vol_multipliers;
  const signed char* smc_flags;
  const float* smc_weights;
  float gate_threshold;
  int smc_gate_disabled;
  int population;
};

// ---------------------------------------------------------------------------
// Stage 1: signal synthesis (candidate x bar)
// ---------------------------------------------------------------------------

__global__ void population_signal_kernel(DeviceDataset dataset,
                                         DeviceGenes genes,
                                         signed char* signal_values,
                                         float* signal_confidences) {
  const long long total =
      static_cast<long long>(genes.population) * static_cast<long long>(dataset.bars);
  for (long long flat = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
       flat < total;
       flat += static_cast<long long>(blockDim.x) * gridDim.x) {
    const int candidate = static_cast<int>(flat / dataset.bars);
    const int bar = static_cast<int>(flat - static_cast<long long>(candidate) * dataset.bars);

    // Terms accumulate in ascending CSR order, matching the canonical f32
    // accumulation order bit for bit.
    float combined = 0.0f;
    const int start = genes.offsets[candidate];
    const int end = genes.offsets[candidate + 1];
    for (int term = start; term < end; ++term) {
      const int feature = genes.indices[term];
      combined += genes.weights[term] *
                  dataset.indicators[static_cast<long long>(feature) * dataset.bars + bar];
    }

    const float long_threshold = genes.long_thresholds[candidate];
    const float short_threshold = genes.short_thresholds[candidate];
    signed char signal = 0;
    if (combined >= long_threshold) {
      signal = 1;
    } else if (combined <= short_threshold) {
      signal = -1;
    }

    signed char emitted = 0;
    float confidence = 0.0f;
    if (signal != 0) {
      float gap = fabsf(long_threshold - short_threshold);
      if (!(gap > 1.0e-6f)) {
        gap = 1.0e-6f;
      }
      const float margin =
          (signal == 1) ? (combined - long_threshold) : (short_threshold - combined);
      confidence = fminf(fmaxf(margin / gap, 0.0f), 1.0f);

      float active_sum = 0.0f;
      for (int slot = 0; slot < kSmcSlots; ++slot) {
        if (genes.smc_flags[static_cast<long long>(candidate) * kSmcSlots + slot] != 0) {
          active_sum += genes.smc_weights[slot];
        }
      }
      if (genes.smc_gate_disabled != 0) {
        active_sum = 0.0f;
      }
      const float gate = fminf(genes.gate_threshold, active_sum);

      bool passes_gate = true;
      if (active_sum > 0.0f) {
        float score = 0.0f;
        for (int slot = 0; slot < kSmcSlots; ++slot) {
          const signed char flag =
              genes.smc_flags[static_cast<long long>(candidate) * kSmcSlots + slot];
          if (flag == 0) {
            continue;
          }
          const signed char row =
              dataset.smc_rows[static_cast<long long>(bar) * kSmcSlots + slot];
          if (slot == 5) {
            if (row == 1) {
              score += genes.smc_weights[slot];
            }
          } else if (row == signal) {
            score += genes.smc_weights[slot];
          }
        }
        passes_gate = score >= gate;
      }
      if (passes_gate) {
        emitted = signal;
      } else {
        confidence = 0.0f;
      }
    }

    signal_values[flat] = emitted;
    signal_confidences[flat] = confidence;
  }
}

// ---------------------------------------------------------------------------
// Stage 2: causal entry emission (candidate-major, bar ascending)
// ---------------------------------------------------------------------------

__device__ inline void entry_stop_target_pips(const DeviceDataset& dataset,
                                              const DeviceGenes& genes,
                                              const NeoPopulationSettings& settings,
                                              int candidate,
                                              int signal_bar,
                                              double* stop_pips,
                                              double* target_pips) {
  const double multiplier = genes.stop_vol_multipliers[candidate];
  if (multiplier > 0.0 && dataset.has_adaptive_base != 0 && signal_bar < dataset.bars) {
    const double distance = dataset.adaptive_base_pips[signal_bar];
    const double stop = multiplier * distance;
    const double target = settings.adaptive_rr * stop;
    if (isfinite(stop) && stop > 0.0 && isfinite(target) && target > 0.0) {
      *stop_pips = stop;
      *target_pips = target;
      return;
    }
  }
  *stop_pips = genes.stop_pips[candidate];
  *target_pips = genes.target_pips[candidate];
}

__global__ void population_count_events_kernel(DeviceDataset dataset,
                                               DeviceGenes genes,
                                               const signed char* signal_values,
                                               unsigned int* event_counts) {
  __shared__ unsigned int block_total;
  const int candidate = blockIdx.x;
  if (candidate >= genes.population) {
    return;
  }
  if (threadIdx.x == 0) {
    block_total = 0u;
  }
  __syncthreads();

  unsigned int local = 0u;
  const long long base = static_cast<long long>(candidate) * dataset.bars;
  for (int bar = 1 + threadIdx.x; bar < dataset.bars; bar += blockDim.x) {
    if (signal_values[base + bar - 1] != 0) {
      local += 1u;
    }
  }
  atomicAdd(&block_total, local);
  __syncthreads();
  if (threadIdx.x == 0) {
    event_counts[candidate] = block_total;
  }
}

// Deterministic single-block exclusive scan over the per-candidate counts. The
// population axis is small relative to the bar axis, so one sequential pass is
// both exact and cheap; it never truncates.
__global__ void population_scan_offsets_kernel(const unsigned int* event_counts,
                                               unsigned long long* event_offsets,
                                               int population) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  unsigned long long running = 0ull;
  for (int candidate = 0; candidate < population; ++candidate) {
    event_offsets[candidate] = running;
    running += static_cast<unsigned long long>(event_counts[candidate]);
  }
  event_offsets[population] = running;
}

__global__ void population_emit_events_kernel(DeviceDataset dataset,
                                              DeviceGenes genes,
                                              NeoPopulationSettings settings,
                                              const signed char* signal_values,
                                              const unsigned long long* event_offsets,
                                              const unsigned long long* scenario_ids,
                                              NeoPopulationEvent* events,
                                              unsigned long long max_events,
                                              int* overflow_flag) {
  __shared__ unsigned int scan[kEmitBlock];
  __shared__ unsigned int chunk_base;

  const int candidate = blockIdx.x;
  if (candidate >= genes.population) {
    return;
  }
  if (threadIdx.x == 0) {
    chunk_base = 0u;
  }
  __syncthreads();

  const unsigned long long candidate_base = event_offsets[candidate];
  const unsigned long long candidate_id = genes.candidate_ids[candidate];
  const unsigned long long scenario_id = scenario_ids[candidate];
  const long long signal_base = static_cast<long long>(candidate) * dataset.bars;
  const int last_dataset_bar = dataset.bars - 1;
  const double pip = guarded_pip(settings.pip_value);
  const double half_spread_price = settings.spread_pips * 0.5 * pip;

  for (int chunk_start = 1; chunk_start < dataset.bars; chunk_start += kEmitBlock) {
    const int bar = chunk_start + static_cast<int>(threadIdx.x);
    unsigned int flag = 0u;
    signed char direction = 0;
    if (bar < dataset.bars) {
      direction = signal_values[signal_base + bar - 1];
      flag = (direction != 0) ? 1u : 0u;
    }
    scan[threadIdx.x] = flag;
    __syncthreads();

    // Hillis-Steele inclusive scan, converted to an exclusive offset below.
    for (unsigned int stride = 1u; stride < kEmitBlock; stride <<= 1) {
      unsigned int addend = 0u;
      if (threadIdx.x >= stride) {
        addend = scan[threadIdx.x - stride];
      }
      __syncthreads();
      scan[threadIdx.x] += addend;
      __syncthreads();
    }

    const unsigned int inclusive = scan[threadIdx.x];
    const unsigned int exclusive = inclusive - flag;
    if (flag != 0u) {
      const unsigned long long slot = candidate_base + chunk_base + exclusive;
      if (slot >= max_events) {
        atomicExch(overflow_flag, 1);
      } else {
        const int signal_bar = bar - 1;
        const double entry_price =
            dataset.close[bar] + static_cast<double>(direction) * half_spread_price;
        double stop_pips = 0.0;
        double target_pips = 0.0;
        entry_stop_target_pips(dataset, genes, settings, candidate, signal_bar, &stop_pips,
                               &target_pips);
        double stop_price = 0.0;
        double target_price = 0.0;
        if (direction == kDirectionLong) {
          stop_price = entry_price - stop_pips * pip;
          target_price = entry_price + target_pips * pip;
        } else {
          stop_price = entry_price + stop_pips * pip;
          target_price = entry_price - target_pips * pip;
        }
        int last_bar = last_dataset_bar;
        if (settings.max_hold_bars > 0u) {
          const unsigned int hold = settings.max_hold_bars > settings.min_hold_bars
                                        ? settings.max_hold_bars
                                        : settings.min_hold_bars;
          const long long candidate_last = static_cast<long long>(bar) + hold;
          last_bar = candidate_last < last_dataset_bar ? static_cast<int>(candidate_last)
                                                       : last_dataset_bar;
        }

        NeoPopulationEvent event;
        event.candidate_id = candidate_id;
        event.scenario_id = scenario_id;
        event.entry_bar = static_cast<std::uint32_t>(bar);
        event.last_bar = static_cast<std::uint32_t>(last_bar);
        event.direction = static_cast<int>(direction);
        event.precedence = kPrecedenceStopFirst;
        event.stop_price = stop_price;
        event.target_price = target_price;
        // Excursion is measured against the fill, and the first-hit walk is the
        // only stage that sees every bar the position was open, so the entry
        // travels with the event rather than being re-derived downstream.
        event.entry_price = entry_price;
        events[slot] = event;
      }
    }
    __syncthreads();
    if (threadIdx.x == kEmitBlock - 1) {
      chunk_base += inclusive;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Stage 3: warp-cooperative first hit (one warp per event)
// ---------------------------------------------------------------------------

__global__ void population_gap_flags_kernel(DeviceDataset dataset,
                                            NeoPopulationSettings settings,
                                            unsigned char* gap_flags) {
  for (int bar = blockIdx.x * blockDim.x + threadIdx.x; bar < dataset.bars;
       bar += blockDim.x * gridDim.x) {
    unsigned char flag = 0u;
    if (bar > 0 && settings.gap_threshold_ms > 0) {
      const long long previous = dataset.timestamps[bar - 1];
      const long long current = dataset.timestamps[bar];
      if (current > previous && (current - previous) >= settings.gap_threshold_ms) {
        flag = 1u;
      }
    }
    gap_flags[bar] = flag;
  }
}

__global__ void population_first_hit_kernel(DeviceDataset dataset,
                                            NeoPopulationSettings settings,
                                            const NeoPopulationEvent* events,
                                            const unsigned char* gap_flags,
                                            NeoPopulationOutcome* outcomes,
                                            unsigned long long event_count) {
  const unsigned int warps_per_block = blockDim.x / warpSize;
  const unsigned int warp_in_block = threadIdx.x / warpSize;
  const unsigned int lane = threadIdx.x % warpSize;
  // One warp per event for the parallel search, one THREAD per event for the
  // trailing walk.
  //
  // The trailing walk is sequential — the level a bar is tested against depends
  // on every bar before it — so it uses no warp cooperation at all. Mapping it a
  // warp each left 31 of 32 lanes idle, throwing away 32x of the machine on the
  // path production actually takes. The launcher sizes the grid to match.
  const bool sequential = (settings.trailing_enabled != 0u);
  const unsigned long long event_index =
      sequential ? (static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x)
                 : (static_cast<unsigned long long>(blockIdx.x) * warps_per_block + warp_in_block);
  if (event_index >= event_count) {
    return;
  }

  const NeoPopulationEvent event = events[event_index];
  if (lane == 0u) {
    NeoPopulationOutcome outcome;
    outcome.candidate_id = event.candidate_id;
    outcome.scenario_id = event.scenario_id;
    outcome.exit_bar = -1;
    outcome.exit_reason = kExitNone;
    outcome.entry_bar = static_cast<int>(event.entry_bar);
    outcome.pad = 0;
    outcome.mfe = 0.0;
    outcome.mae = 0.0;
    // Was left unset when the field was added — uninitialised device memory
    // written straight into the reducer's exit-price fallback.
    outcome.exit_price = 0.0;
    outcome.pnl = 0.0;
    outcome.r_multiple = 0.0;
    outcomes[event_index] = outcome;
  }

  const int entry_bar = static_cast<int>(event.entry_bar);
  int last_bar = static_cast<int>(event.last_bar);
  if (last_bar > dataset.bars - 1) {
    last_bar = dataset.bars - 1;
  }
  if (entry_bar >= last_bar) {
    // The canonical resolver never schedules such an event.
    return;
  }

  const unsigned int min_hold = settings.min_hold_bars > 0u ? settings.min_hold_bars : 1u;
  const long long level_activation = static_cast<long long>(entry_bar) + min_hold;
  long long max_hold_exit = -1;
  if (settings.max_hold_bars > 0u) {
    const unsigned int hold = settings.max_hold_bars > settings.min_hold_bars
                                  ? settings.max_hold_bars
                                  : settings.min_hold_bars;
    const long long scheduled = static_cast<long long>(entry_bar) + hold;
    if (scheduled <= last_bar) {
      max_hold_exit = scheduled;
    }
  }

  // ── Trailing stop: one lane, walking forward ────────────────────────────
  //
  // The parallel search below examines bars independently, which a trailing
  // stop makes impossible: the level a bar is tested against depends on every
  // bar before it. So the kernel had no trailing at all while the CPU engine
  // has it on by default, and the two lanes were evaluating different
  // strategies — invisible, because the parity fixtures set it off.
  //
  // Lane 0 walks in order, exactly as eval.rs does; the rest idle. That wastes
  // 31/32 of the warp, and is still the cheaper shape: a trailing stop closes
  // trades in a bar or two instead of running thousands of bars toward a
  // distant target, so this walk is short where the parallel search was long.
  //
  // The order is load-bearing and mirrors the CPU: apply the trail set by
  // PRIOR bars, test this bar against it, and only then let this bar's extreme
  // ratchet the trail for future bars. Letting a bar's own high move the stop
  // its own low is checked against is what the CPU comment calls reward-
  // hackable — the GA found it and produced never-lose genes.
  if (sequential) {
    int exit_bar = -1;
    int exit_reason = kExitNone;
    double exit_price = 0.0;
    double fav = 0.0;
    double adv = 0.0;
    {
      const double pip_seq = guarded_pip(settings.pip_value);
      const bool is_long = (event.direction == kDirectionLong);
      const double stop_distance = fabs(event.entry_price - event.stop_price);
      const double arm_at = settings.trailing_be_trigger_r * stop_distance;
      const double give_back = settings.trailing_atr_multiplier * stop_distance;
      const double lock = settings.trailing_min_lock_pips * pip_seq;
      double trail = 0.0;  // 0.0 is the unset sentinel, as on the CPU
      for (int bar = entry_bar + 1; bar <= last_bar; ++bar) {
        const double high = dataset.high[bar];
        const double low = dataset.low[bar];
        const double moved =
            is_long ? (high - event.entry_price) : (event.entry_price - low);
        const double against =
            is_long ? (event.entry_price - low) : (high - event.entry_price);
        if (moved > fav) { fav = moved; }
        if (against > adv) { adv = against; }

        if (gap_flags[bar] != 0u) {
          exit_bar = bar;
          exit_reason = kExitGap;
          exit_price = dataset.close[bar];
          break;
        }
        double stop = event.stop_price;
        if (trail > 0.0 &&
            ((is_long && trail > stop) || (!is_long && trail < stop))) {
          stop = trail;
        }
        if (bar >= static_cast<int>(level_activation)) {
          if (is_long ? (low <= stop) : (high >= stop)) {
            exit_bar = bar;
            exit_reason = kExitStop;
            exit_price = stop;
            break;
          }
          if (is_long ? (high >= event.target_price)
                      : (low <= event.target_price)) {
            exit_bar = bar;
            exit_reason = kExitTarget;
            exit_price = event.target_price;
            break;
          }
        }
        if (max_hold_exit >= 0 && bar == max_hold_exit) {
          exit_bar = bar;
          exit_reason = kExitMaxHold;
          exit_price = dataset.close[bar];
          break;
        }
        if (moved >= arm_at) {
          const double candidate =
              is_long ? fmax(high - give_back, event.entry_price + lock)
                      : fmin(low + give_back, event.entry_price - lock);
          if (trail == 0.0 ||
              (is_long ? candidate > trail : candidate < trail)) {
            trail = candidate;
          }
        }
      }
      NeoPopulationOutcome outcome;
      outcome.candidate_id = event.candidate_id;
      outcome.scenario_id = event.scenario_id;
      outcome.exit_bar = exit_bar;
      outcome.exit_reason = (exit_bar < 0) ? kExitNone : exit_reason;
      outcome.entry_bar = entry_bar;
      outcome.pad = 0;
      // Excursion in the same units the parallel path reports: money per lot.
      outcome.mfe = (fav > 0.0) ? fav / pip_seq * settings.pip_value_per_lot : 0.0;
      outcome.mae = (adv > 0.0) ? adv / pip_seq * settings.pip_value_per_lot : 0.0;
      outcome.exit_price = exit_price;
      // P&L is the reducer's, which owns sizing and carry.
      outcome.pnl = 0.0;
      outcome.r_multiple = 0.0;
      outcomes[event_index] = outcome;
    }
    return;
  }

  int best_bar = INT_MAX;
  int best_priority = INT_MAX;
  int best_reason = kExitNone;
  // ── Stop at the first tile that contains an exit ─────────────────────────
  //
  // This loop used to run to `last_bar` for every event, keeping the minimum.
  // With no holding cap `last_bar` is the end of the series, so an event
  // entering early scanned hundreds of thousands of bars to find an exit
  // twenty bars away — and every in-signal bar emits an event. That makes the
  // search quadratic in bars where the CPU walk is linear, which is why a 3090
  // measured 0.39 M candidate-bars/s against 21 M/s on a CPU, and why the bench
  // hits 47 M/s on the same kernel: `gpu_bench_prepare` hardcodes
  // `max_hold_bars: 12`, so its scans are twelve bars long.
  //
  // Each iteration has the warp cover one contiguous tile of `warpSize` bars,
  // so if any lane finds an exit in this tile, every earlier bar has already
  // been examined and found clean — the earliest exit in the whole range is in
  // this tile. Finishing the tile and stopping therefore returns exactly what
  // scanning to the end returned.
  //
  // The bound is uniform across the warp and every lane reaches the vote, so
  // the ballot is not reading from exited threads.
  for (int base = entry_bar + 1; base <= last_bar; base += warpSize) {
    const int bar = base + static_cast<int>(lane);
    int reason = kExitNone;
    if (bar > last_bar) {
      // Past the end: this lane contributes nothing, but still votes.
    } else if (gap_flags[bar] != 0u) {
      reason = kExitGap;
    } else {
      bool stop_hit = false;
      bool target_hit = false;
      if (bar >= level_activation) {
        const double high = dataset.high[bar];
        const double low = dataset.low[bar];
        if (event.direction == kDirectionLong) {
          stop_hit = low <= event.stop_price;
          target_hit = high >= event.target_price;
        } else {
          stop_hit = high >= event.stop_price;
          target_hit = low <= event.target_price;
        }
      }
      if (stop_hit && target_hit) {
        reason = (event.precedence == kPrecedenceStopFirst) ? kExitStop : kExitTarget;
      } else if (stop_hit) {
        reason = kExitStop;
      } else if (target_hit) {
        reason = kExitTarget;
      } else if (max_hold_exit >= 0 && bar == max_hold_exit) {
        reason = kExitMaxHold;
      }
    }
    if (reason != kExitNone) {
      const int priority = exit_priority(reason);
      if (bar < best_bar || (bar == best_bar && priority < best_priority)) {
        best_bar = bar;
        best_priority = priority;
        best_reason = reason;
      }
    }
    if (__any_sync(0xffffffffu, reason != kExitNone)) {
      break;
    }
  }

  const unsigned mask = 0xffffffffu;
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    const int other_bar = __shfl_down_sync(mask, best_bar, offset);
    const int other_priority = __shfl_down_sync(mask, best_priority, offset);
    const int other_reason = __shfl_down_sync(mask, best_reason, offset);
    if (other_bar < best_bar ||
        (other_bar == best_bar && other_priority < best_priority)) {
      best_bar = other_bar;
      best_priority = other_priority;
      best_reason = other_reason;
    }
  }

  // ── Excursion, bounded by the exit the warp just agreed on ──────────────
  //
  // Deliberately a second pass rather than accumulation during the search: the
  // exit is not known until the reduction above completes, and excursion past
  // the exit is not part of the trade. Every lane now knows `best_bar`, so the
  // same stride re-walks only the bars the position was actually open.
  //
  // The CPU walk (eval.rs) updates these on every open bar *before* testing for
  // an exit, so the exit bar itself counts, and the range is
  // [entry_bar + 1, exit]. Both start at zero and only ever rise, so a trade
  // that never moves favourably reports 0 rather than a negative excursion.
  const int excursion_last = (best_bar == INT_MAX) ? last_bar : best_bar;
  const double pip = guarded_pip(settings.pip_value);
  double best_fav = 0.0;
  double best_adv = 0.0;
  for (int bar = entry_bar + 1 + static_cast<int>(lane); bar <= excursion_last;
       bar += warpSize) {
    const double high = dataset.high[bar];
    const double low = dataset.low[bar];
    double fav = 0.0;
    double adv = 0.0;
    if (event.direction == kDirectionLong) {
      fav = high - event.entry_price;
      adv = event.entry_price - low;
    } else {
      fav = event.entry_price - low;
      adv = high - event.entry_price;
    }
    const double fav_money = fav / pip * settings.pip_value_per_lot;
    const double adv_money = adv / pip * settings.pip_value_per_lot;
    if (fav_money > best_fav) {
      best_fav = fav_money;
    }
    if (adv_money > best_adv) {
      best_adv = adv_money;
    }
  }
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    const double other_fav = __shfl_down_sync(mask, best_fav, offset);
    const double other_adv = __shfl_down_sync(mask, best_adv, offset);
    if (other_fav > best_fav) {
      best_fav = other_fav;
    }
    if (other_adv > best_adv) {
      best_adv = other_adv;
    }
  }

  if (lane == 0u) {
    NeoPopulationOutcome outcome;
    outcome.candidate_id = event.candidate_id;
    outcome.scenario_id = event.scenario_id;
    outcome.exit_bar = (best_bar == INT_MAX) ? -1 : best_bar;
    outcome.exit_reason = (best_bar == INT_MAX) ? kExitNone : best_reason;
    outcome.entry_bar = entry_bar;
    outcome.pad = 0;
    outcome.mfe = best_fav;
    outcome.mae = best_adv;
    // Reported rather than left for the reducer to rebuild. With fixed levels
    // the two agree; with a trailing stop only the kernel knows where the stop
    // had ratcheted to, so the field has to exist for either path to be right.
    outcome.exit_price =
        (best_bar == INT_MAX)
            ? 0.0
            : ((best_reason == kExitStop)
                   ? event.stop_price
                   : ((best_reason == kExitTarget) ? event.target_price
                                                   : dataset.close[best_bar]));
    // P&L is settled by the reducer, which owns position sizing and carry.
    outcome.pnl = 0.0;
    outcome.r_multiple = 0.0;
    outcomes[event_index] = outcome;
  }
}

// ---------------------------------------------------------------------------
// Stage 4: exact per-candidate cost, sizing and metric reduction
// ---------------------------------------------------------------------------

__device__ inline void finalize_daily_drawdown_segment(double day_peak,
                                                       double day_low,
                                                       double* max_daily_drawdown) {
  if (day_peak > 0.0) {
    const double drawdown = (day_peak - day_low) / day_peak;
    if (drawdown > *max_daily_drawdown) {
      *max_daily_drawdown = drawdown;
    }
  }
}

__device__ inline void update_realized_risk(double equity,
                                            double* peak_equity,
                                            double* day_peak,
                                            double* day_low,
                                            double* max_drawdown,
                                            double* max_daily_drawdown) {
  if (equity > *peak_equity) {
    *peak_equity = equity;
  }
  if (equity > *day_peak) {
    finalize_daily_drawdown_segment(*day_peak, *day_low, max_daily_drawdown);
    *day_peak = equity;
    *day_low = equity;
  } else if (equity < *day_low) {
    *day_low = equity;
  }
  if (*peak_equity > 0.0) {
    const double drawdown = (*peak_equity - equity) / *peak_equity;
    if (drawdown > *max_drawdown) {
      *max_drawdown = drawdown;
    }
  }
}

__device__ inline double risk_based_position_lots(double confidence,
                                                  double equity,
                                                  double stop_pips,
                                                  const NeoPopulationSettings& settings) {
  confidence = fmin(fmax(confidence, 0.0), 1.0);
  double confidence_scale = 1.0;
  if (isfinite(settings.high_quality_confidence) && settings.high_quality_confidence > 0.0) {
    confidence_scale = fmin(confidence / settings.high_quality_confidence, 1.0);
  }
  const double risk =
      settings.risk_per_trade_min +
      (settings.risk_per_trade_max - settings.risk_per_trade_min) * confidence_scale;
  const double denominator = fmax(stop_pips, 1.0) * settings.pip_value_per_lot;
  double lots = 0.0;
  if (equity > 0.0 && fabs(denominator) > 1.0e-12 && isfinite(denominator)) {
    lots = risk * equity / denominator;
  }
  if (!isfinite(lots)) {
    return 0.0;
  }
  return fmin(fmax(lots, 0.0), 100.0);
}

__device__ inline double apply_carry_and_conversion(double gross_pnl_scaled,
                                                    double lots,
                                                    int direction,
                                                    long long entry_timestamp,
                                                    long long exit_timestamp,
                                                    const NeoPopulationSettings& settings) {
  double overnight_days = 0.0;
  if (exit_timestamp > entry_timestamp && entry_timestamp > 0) {
    overnight_days = static_cast<double>(exit_timestamp - entry_timestamp) / 86400000.0;
  }
  const double swap_pips = (direction == kDirectionLong) ? settings.swap_long_pips_per_day
                                                         : settings.swap_short_pips_per_day;
  const double with_carry =
      gross_pnl_scaled + swap_pips * overnight_days * settings.pip_value_per_lot * lots;
  if (isfinite(settings.pnl_conversion_fee_rate) && settings.pnl_conversion_fee_rate > 0.0 &&
      settings.pnl_conversion_fee_rate < 1.0) {
    return with_carry * (1.0 - settings.pnl_conversion_fee_rate);
  }
  return with_carry;
}

// Every outcome starts defined. The reduce fills in the ones that become
// trades, and a candidate entry that never opens a position must still read as
// "no exit" rather than as whatever the buffer held last generation.
__global__ void population_seed_outcomes_kernel(NeoPopulationOutcome* outcomes,
                                                unsigned long long event_count) {
  const unsigned long long index =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= event_count) {
    return;
  }
  NeoPopulationOutcome outcome;
  outcome.candidate_id = 0ull;
  outcome.scenario_id = 0ull;
  outcome.exit_bar = -1;
  outcome.exit_reason = kExitNone;
  outcome.entry_bar = -1;
  outcome.pad = 0;
  outcome.mfe = 0.0;
  outcome.mae = 0.0;
  outcome.exit_price = 0.0;
  outcome.pnl = 0.0;
  outcome.r_multiple = 0.0;
  outcomes[index] = outcome;
}

__global__ void population_reduce_kernel(DeviceDataset dataset,
                                         DeviceGenes genes,
                                         NeoPopulationSettings settings,
                                         const signed char* signal_values,
                                         const float* signal_confidences,
                                         const unsigned char* gap_flags,
                                         NeoPopulationOutcome* outcomes,
                                         const unsigned long long* event_offsets,
                                         const unsigned long long* scenario_ids,
                                         double* monthly_pnls,
                                         double* month_start_equities,
                                         NeoPopulationMetricRow* rows,
                                         unsigned long long* accepted_trade_total) {
  const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= genes.population) {
    return;
  }

  const int month_capacity = static_cast<int>(settings.month_capacity);
  double* monthly = monthly_pnls + static_cast<long long>(candidate) * month_capacity;
  double* month_start = month_start_equities + static_cast<long long>(candidate) * month_capacity;
  const double initial_equity = settings.initial_equity;
  for (int index = 0; index < month_capacity; ++index) {
    monthly[index] = 0.0;
    month_start[index] = initial_equity;
  }

  // A fixed slice per candidate instead of a prefix sum over emitted events.
  //
  // The offsets existed because every candidate entry got a slot, and the count
  // varied. Only trades are recorded now, and a gene makes about 3 000 of them
  // over 439 315 bars — so a flat slice is both simpler and far smaller: 2.4 GB
  // for 4 096 candidates against the 180 GB the entry-indexed buffers would
  // need, which is why the population was splitting 4 096 -> 128.
  //
  // Overrunning the slice drops trades rather than corrupting a neighbour's,
  // and is reported through the diagnostics so a silent truncation cannot pass
  // for a strategy that simply traded less.
  const unsigned long long range_start =
      static_cast<unsigned long long>(candidate) * kMaxTradesPerCandidate;
  const unsigned long long range_end = range_start + kMaxTradesPerCandidate;
  const long long confidence_base = static_cast<long long>(candidate) * dataset.bars;

  double equity = initial_equity;
  double peak_equity = initial_equity;
  double max_drawdown = 0.0;
  long long trade_count = 0;
  long long wins = 0;
  double gross_profit = 0.0;
  double gross_loss = 0.0;
  unsigned long long accepted_trades = 0ull;

  long long last_month = -1;
  double current_month_pnl = 0.0;
  double current_month_start_equity = initial_equity;
  long long month_ptr = -1;

  long long last_day = -1;
  double day_peak = equity;
  double day_low = equity;
  double max_daily_drawdown = 0.0;
  unsigned int day_trade_count = 0u;

  bool has_position = false;
  NeoPopulationEvent position_event;
  double position_entry_price = 0.0;
  double position_lots = 0.0;
  unsigned long long position_index = 0ull;
  double position_stop_pips = 0.0;
  // Exit detection moved here from the first-hit kernel, so the state it used
  // to carry per event now lives with the position: where the trail has
  // ratcheted to, and how far the trade has run either way.
  double position_trail = 0.0;
  double position_fav = 0.0;
  double position_adv = 0.0;
  int position_min_hold_bar = 0;
  int position_max_hold_bar = -1;

  const double pip = guarded_pip(settings.pip_value);
  const long long signal_base = static_cast<long long>(candidate) * dataset.bars;
  const double half_spread_cost = settings.spread_pips * 0.5 * settings.pip_value_per_lot;
  const double half_spread_price = settings.spread_pips * 0.5 * pip;
  unsigned long long cursor = range_start;

  for (int bar = 1; bar < dataset.bars; ++bar) {

    const long long month = dataset.months[bar];
    if (month != last_month) {
      if (last_month != -1) {
        month_ptr += 1;
        if (month_ptr < month_capacity) {
          monthly[month_ptr] = current_month_pnl;
          month_start[month_ptr] = current_month_start_equity;
        }
      }
      current_month_pnl = 0.0;
      current_month_start_equity = equity;
      last_month = month;
    }

    const long long day = dataset.days[bar];
    if (day != last_day) {
      if (last_day != -1) {
        finalize_daily_drawdown_segment(day_peak, day_low, &max_daily_drawdown);
      }
      last_day = day;
      day_peak = equity;
      day_low = equity;
      day_trade_count = 0u;
    }

    bool continue_bar = false;
    if (has_position) {
      // Straight from the flag. Asking whether a precomputed outcome happened
      // to name this bar was only ever a proxy for reading it.
      const bool exited_on_gap = gap_flags[bar] != 0u;
      if (exited_on_gap) {
        double exit_price = dataset.close[bar];
        double price_pnl = 0.0;
        if (position_event.direction == kDirectionLong) {
          price_pnl = (exit_price - position_entry_price) / pip * settings.pip_value_per_lot;
        } else {
          price_pnl = (position_entry_price - exit_price) / pip * settings.pip_value_per_lot;
        }
        const double gross_scaled =
            price_pnl * position_lots -
            (settings.commission_per_trade + half_spread_cost) * position_lots;
        const long long entry_timestamp = dataset.timestamps[position_event.entry_bar];
        const long long exit_timestamp = dataset.timestamps[bar];
        const double pnl = apply_carry_and_conversion(gross_scaled, position_lots,
                                                      position_event.direction, entry_timestamp,
                                                      exit_timestamp, settings);
        equity += pnl;
        // The per-trade record is completed here because this is the only place
        // that knows position size, carry and the conversion fee. R-multiple
        // mirrors eval.rs exactly — realised P&L over the entry stop distance,
        // guarded against a zero denominator — so it stays comparable with the
        // CPU trade list rather than merely plausible.
        outcomes[position_index].exit_bar = bar;
        outcomes[position_index].exit_reason = kExitGap;
        outcomes[position_index].entry_bar = position_event.entry_bar;
        outcomes[position_index].exit_price = exit_price;
        outcomes[position_index].mfe =
            position_fav > 0.0 ? position_fav / pip * settings.pip_value_per_lot : 0.0;
        outcomes[position_index].mae =
            position_adv > 0.0 ? position_adv / pip * settings.pip_value_per_lot : 0.0;
        outcomes[position_index].pnl = pnl;
        outcomes[position_index].r_multiple =
            pnl / fmax(position_stop_pips * settings.pip_value_per_lot, 1.0e-9);
        current_month_pnl += pnl;
        trade_count += 1;
        if (pnl > 0.0) {
          wins += 1;
          gross_profit += pnl;
        } else {
          gross_loss += fabs(pnl);
        }
        update_realized_risk(equity, &peak_equity, &day_peak, &day_low, &max_drawdown,
                             &max_daily_drawdown);
        has_position = false;
      } else {
        const double low = dataset.low[bar];
        const double high = dataset.high[bar];
        double worst = 0.0;
        double best = 0.0;
        if (position_event.direction == kDirectionLong) {
          worst = (low - position_entry_price) / pip * settings.pip_value_per_lot;
          best = (high - position_entry_price) / pip * settings.pip_value_per_lot;
        } else {
          worst = (position_entry_price - high) / pip * settings.pip_value_per_lot;
          best = (position_entry_price - low) / pip * settings.pip_value_per_lot;
        }
        worst *= position_lots;
        best *= position_lots;

        if (equity + best > peak_equity) {
          peak_equity = equity + best;
        }
        if (equity + best > day_peak) {
          finalize_daily_drawdown_segment(day_peak, day_low, &max_daily_drawdown);
          day_peak = equity + best;
          day_low = equity + worst;
        } else if (equity + worst < day_low) {
          day_low = equity + worst;
        }
        if (peak_equity > 0.0) {
          const double drawdown = (peak_equity - (equity + worst)) / peak_equity;
          if (drawdown > max_drawdown) {
            max_drawdown = drawdown;
          }
        }

        // ── Exit, decided on this bar ────────────────────────────────────
        //
        // Same order as the CPU walk in eval.rs: the trail set by PRIOR bars is
        // what this bar is tested against, and only after the test does this
        // bar's extreme move it. Letting a bar's own high move the stop its own
        // low is checked against is reward-hackable — the GA found it once and
        // produced never-lose genes.
        const bool is_long_pos = position_event.direction == kDirectionLong;
        double active_stop = position_event.stop_price;
        if (position_trail > 0.0 && ((is_long_pos && position_trail > active_stop) ||
                                     (!is_long_pos && position_trail < active_stop))) {
          active_stop = position_trail;
        }
        int exit_reason_now = kExitNone;
        double exit_price_now = 0.0;
        if (bar >= position_min_hold_bar) {
          if (is_long_pos ? (low <= active_stop) : (high >= active_stop)) {
            exit_reason_now = kExitStop;
            exit_price_now = active_stop;
          } else if (is_long_pos ? (high >= position_event.target_price)
                                 : (low <= position_event.target_price)) {
            exit_reason_now = kExitTarget;
            exit_price_now = position_event.target_price;
          }
        }
        if (exit_reason_now == kExitNone && position_max_hold_bar >= 0 &&
            bar >= position_max_hold_bar) {
          exit_reason_now = kExitMaxHold;
          exit_price_now = dataset.close[bar];
        }
        // Excursion accumulates on every open bar including this one, matching
        // the CPU, which updates before testing for an exit.
        {
          const double moved = is_long_pos ? (high - position_entry_price)
                                           : (position_entry_price - low);
          const double against = is_long_pos ? (position_entry_price - low)
                                             : (high - position_entry_price);
          if (moved > position_fav) {
            position_fav = moved;
          }
          if (against > position_adv) {
            position_adv = against;
          }
        }
        if (exit_reason_now == kExitNone && settings.trailing_enabled != 0u) {
          const double stop_distance = fabs(position_entry_price - position_event.stop_price);
          const double moved = is_long_pos ? (high - position_entry_price)
                                           : (position_entry_price - low);
          if (moved >= settings.trailing_be_trigger_r * stop_distance) {
            const double give_back = settings.trailing_atr_multiplier * stop_distance;
            const double lock = settings.trailing_min_lock_pips * pip;
            const double candidate =
                is_long_pos ? fmax(high - give_back, position_entry_price + lock)
                            : fmin(low + give_back, position_entry_price - lock);
            if (position_trail == 0.0 ||
                (is_long_pos ? candidate > position_trail : candidate < position_trail)) {
              position_trail = candidate;
            }
          }
        }
        if (exit_reason_now != kExitNone) {
          // The kernel reports where the position actually closed. A trailing
          // stop moves, so rebuilding this from `position_event.stop_price`
          // would price every trailed exit at the original stop and understate
          // the win. Zero means "not reported" — outcomes from before the field
          // existed — and those fall back to the levels below.
          const double exit_price = exit_price_now;
          {
            double price_pnl = 0.0;
            if (position_event.direction == kDirectionLong) {
              price_pnl = (exit_price - position_entry_price) / pip * settings.pip_value_per_lot;
            } else {
              price_pnl = (position_entry_price - exit_price) / pip * settings.pip_value_per_lot;
            }
            const double gross_scaled =
                price_pnl * position_lots -
                (settings.commission_per_trade + half_spread_cost) * position_lots;
            const long long entry_timestamp = dataset.timestamps[position_event.entry_bar];
            const long long exit_timestamp = dataset.timestamps[bar];
            const double pnl =
                apply_carry_and_conversion(gross_scaled, position_lots, position_event.direction,
                                           entry_timestamp, exit_timestamp, settings);
            equity += pnl;
            current_month_pnl += pnl;
            trade_count += 1;
            if (pnl > 0.0) {
              wins += 1;
              gross_profit += pnl;
            } else {
              gross_loss += fabs(pnl);
            }
            // The whole trade record is written here now. Nothing upstream knows
            // the exit any more, so nothing upstream can fill this in.
            outcomes[position_index].exit_bar = bar;
            outcomes[position_index].exit_reason = exit_reason_now;
            outcomes[position_index].entry_bar = position_event.entry_bar;
            outcomes[position_index].exit_price = exit_price;
            outcomes[position_index].mfe =
                position_fav > 0.0 ? position_fav / pip * settings.pip_value_per_lot : 0.0;
            outcomes[position_index].mae =
                position_adv > 0.0 ? position_adv / pip * settings.pip_value_per_lot : 0.0;
            outcomes[position_index].pnl = pnl;
            outcomes[position_index].r_multiple =
                pnl / fmax(position_stop_pips * settings.pip_value_per_lot, 1.0e-9);
          }
          update_realized_risk(equity, &peak_equity, &day_peak, &day_low, &max_drawdown,
                               &max_daily_drawdown);
          has_position = false;
        }
        continue_bar = true;
      }
    }
    if (continue_bar) {
      continue;
    }

    // ── Entry, read straight from the signal ──────────────────────────────
    //
    // This used to consult a materialised event. Everything it took from one is
    // available here: the signal says the direction, the bar says when, and
    // `entry_stop_target_pips` is the same call the emit kernel made with the
    // same gene and bar — so the levels are identical, not merely equivalent.
    const int signal_bar = bar - 1;
    const signed char signal_here = signal_values[signal_base + signal_bar];
    if (signal_here != 0) {
      ++cursor;
      if (settings.max_trades_per_day > 0u && day_trade_count >= settings.max_trades_per_day) {
        continue;
      }
      const int direction = signal_here > 0 ? kDirectionLong : kDirectionShort;
      const double entry_price =
          dataset.close[bar] + static_cast<double>(direction) * half_spread_price;
      double entry_stop_pips = 0.0;
      double entry_target_pips = 0.0;
      entry_stop_target_pips(dataset, genes, settings, candidate, signal_bar, &entry_stop_pips,
                             &entry_target_pips);
      NeoPopulationEvent event;
      event.candidate_id = static_cast<unsigned long long>(candidate);
      event.scenario_id = scenario_ids[candidate];
      event.entry_bar = static_cast<unsigned int>(bar);
      event.last_bar = static_cast<unsigned int>(dataset.bars - 1);
      event.direction = direction;
      event.precedence = kPrecedenceStopFirst;
      event.entry_price = entry_price;
      event.stop_price = direction == kDirectionLong ? entry_price - entry_stop_pips * pip
                                                     : entry_price + entry_stop_pips * pip;
      event.target_price = direction == kDirectionLong ? entry_price + entry_target_pips * pip
                                                       : entry_price - entry_target_pips * pip;
      const double stop_pips = entry_stop_pips;
      double lots = 1.0;
      if ((settings.flags & kFlagRiskBasedSizing) != 0u) {
        lots = risk_based_position_lots(
            static_cast<double>(signal_confidences[confidence_base + signal_bar]), equity,
            stop_pips, settings);
      }
      position_event = event;
      position_entry_price = entry_price;
      position_lots = lots;
      // `cursor` already advanced past this trade's slot.
      position_index = cursor - 1ull;
      if (position_index >= range_end) {
        // Out of slots. Keep simulating so equity and drawdown stay honest —
        // the trade still happened — but do not write past this candidate's
        // slice into the next one's.
        position_index = range_end - 1ull;
      }
      position_stop_pips = stop_pips;
      position_trail = 0.0;
      position_fav = 0.0;
      position_adv = 0.0;
      {
        const unsigned int min_hold = settings.min_hold_bars > 0u ? settings.min_hold_bars : 1u;
        position_min_hold_bar = static_cast<int>(event.entry_bar) + static_cast<int>(min_hold);
        if (settings.max_hold_bars > 0u) {
          const unsigned int hold = settings.max_hold_bars > settings.min_hold_bars
                                        ? settings.max_hold_bars
                                        : settings.min_hold_bars;
          position_max_hold_bar = static_cast<int>(event.entry_bar) + static_cast<int>(hold);
        } else {
          position_max_hold_bar = -1;
        }
      }
      has_position = true;
      day_trade_count += 1u;
      accepted_trades += 1ull;
    }
  }

  if (last_day != -1) {
    finalize_daily_drawdown_segment(day_peak, day_low, &max_daily_drawdown);
  }

  const double net_profit = equity - initial_equity;
  const double win_rate =
      trade_count > 0 ? static_cast<double>(wins) / static_cast<double>(trade_count) : 0.0;
  double profit_factor = 0.0;
  if (gross_loss > 0.0) {
    profit_factor = gross_profit / gross_loss;
  } else if (gross_profit > 0.0) {
    profit_factor = 10.0;
  }
  const double expectancy =
      trade_count > 0 ? net_profit / static_cast<double>(trade_count) : 0.0;

  // Completed months only, matching `completed_month_pnls`.
  long long limit = -1;
  if (month_ptr >= 0 && month_capacity > 0) {
    limit = month_ptr < static_cast<long long>(month_capacity - 1)
                ? month_ptr
                : static_cast<long long>(month_capacity - 1);
  }
  double monthly_mean = 0.0;
  double monthly_std = 0.0;
  if (limit >= 0) {
    const long long count = limit + 1;
    long long finite_count = 0;
    double sum = 0.0;
    for (long long index = 0; index <= limit; ++index) {
      if (isfinite(monthly[index])) {
        sum += monthly[index];
        finite_count += 1;
      }
    }
    if (count >= 2 && finite_count >= 2) {
      monthly_mean = sum / static_cast<double>(finite_count);
      double variance = 0.0;
      for (long long index = 0; index <= limit; ++index) {
        if (isfinite(monthly[index])) {
          const double delta = monthly[index] - monthly_mean;
          variance += delta * delta;
        }
      }
      variance /= static_cast<double>(finite_count - 1);
      monthly_std = sqrt(fmax(variance, 0.0));
    } else {
      monthly_mean = 0.0;
      monthly_std = 0.0;
    }
  }
  if (!isfinite(monthly_mean) || !isfinite(monthly_std)) {
    monthly_mean = 0.0;
    monthly_std = 0.0;
  }

  const double sharpe = monthly_std > 0.0 ? (monthly_mean / monthly_std) * 3.4641 : 0.0;
  double consistency = 0.0;
  if (monthly_std > 0.0) {
    consistency = fmin(fmax(monthly_mean / monthly_std, 0.0), 1.0);
  } else if (monthly_mean > 0.0 && limit < 1) {
    consistency = 1.0;
  }

  double monthly_target_hit_rate = 0.0;
  if (limit >= 0) {
    long long hits = 0;
    long long counted = 0;
    for (long long index = 0; index <= limit; ++index) {
      const double base = month_start[index];
      if (base > 0.0) {
        counted += 1;
        if (monthly[index] / base >= 0.04) {
          hits += 1;
        }
      }
    }
    if (counted > 0) {
      monthly_target_hit_rate = static_cast<double>(hits) / static_cast<double>(counted);
    }
  }

  NeoPopulationMetricRow row;
  row.candidate_id = genes.candidate_ids[candidate];
  row.scenario_id = scenario_ids[candidate];
  row.values[0] = sanitize(net_profit);
  row.values[1] = sanitize(sharpe);
  row.values[2] = sanitize(peak_equity);
  row.values[3] = sanitize(max_drawdown);
  row.values[4] = sanitize(win_rate);
  row.values[5] = sanitize(profit_factor);
  row.values[6] = sanitize(expectancy);
  row.values[7] = sanitize(monthly_target_hit_rate);
  row.values[8] = static_cast<double>(trade_count);
  row.values[9] = sanitize(consistency);
  row.values[10] = sanitize(max_daily_drawdown);
  rows[candidate] = row;

  atomicAdd(reinterpret_cast<unsigned long long*>(accepted_trade_total), accepted_trades);
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

template <typename T>
std::int32_t device_alloc(T** pointer, std::size_t count) {
  if (count == 0) {
    *pointer = nullptr;
    return NEO_POPULATION_STATUS_OK;
  }
  if (cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(T)) != cudaSuccess) {
    *pointer = nullptr;
    return NEO_POPULATION_STATUS_ALLOCATION_FAILED;
  }
  return NEO_POPULATION_STATUS_OK;
}

template <typename T>
void device_free(T*& pointer) {
  if (pointer != nullptr) {
    cudaFree(pointer);
    pointer = nullptr;
  }
}

}  // namespace

struct NeoCudaPopulationSession {
  int device = 0;
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
  unsigned long long max_events = 0ull;
  unsigned long long next_event_id = 1ull;
  unsigned long long pending_event_id = 0ull;
  bool has_dataset = false;
  bool has_genes = false;
  bool has_scenarios = false;
  bool metrics_ready = false;
  int bars = 0;
  int feature_count = 0;
  int population = 0;
  int month_capacity = 0;
  // What the workspace was actually built for.
  //
  // The signal, confidence and outcome arrays are sized population * bars at
  // allocation, and every kernel indexes them by the CURRENT population, which
  // `upload_genes` overwrites on each call. The reuse test compared only
  // `signal_values == nullptr` and `month_capacity`, so a session built for a
  // small population and reused for a large one wrote past the end of all
  // three — into `monthly_pnls` and `month_start_equities`, which are the
  // arrays sharpe and consistency are computed from, and `sanitize()` then
  // turns any non-finite consequence into 0.0.
  int workspace_population = 0;
  int workspace_bars = 0;
  unsigned long long emitted_events = 0ull;
  unsigned long long accepted_trades = 0ull;
  std::uint64_t dataset_upload_bytes = 0ull;
  std::uint64_t gene_upload_bytes = 0ull;
  std::uint64_t scenario_upload_bytes = 0ull;
  std::uint64_t kernel_submissions = 0ull;
  std::uint64_t synchronization_events = 0ull;

  double* close = nullptr;
  double* high = nullptr;
  double* low = nullptr;
  float* indicators = nullptr;
  long long* months = nullptr;
  long long* days = nullptr;
  long long* timestamps = nullptr;
  signed char* smc_rows = nullptr;
  double* adaptive_base_pips = nullptr;
  int has_adaptive_base = 0;
  unsigned char* gap_flags = nullptr;

  unsigned long long* candidate_ids = nullptr;
  int* gene_offsets = nullptr;
  int* gene_indices = nullptr;
  float* gene_weights = nullptr;
  float* long_thresholds = nullptr;
  float* short_thresholds = nullptr;
  double* stop_pips = nullptr;
  double* target_pips = nullptr;
  double* stop_vol_multipliers = nullptr;
  signed char* smc_flags = nullptr;
  float* smc_weights = nullptr;
  float gate_threshold = 0.0f;
  int smc_gate_disabled = 0;

  unsigned long long* scenario_ids = nullptr;

  signed char* signal_values = nullptr;
  float* signal_confidences = nullptr;
  unsigned int* event_counts = nullptr;
  unsigned long long* event_offsets = nullptr;
  NeoPopulationEvent* events = nullptr;
  NeoPopulationOutcome* outcomes = nullptr;
  double* monthly_pnls = nullptr;
  double* month_start_equities = nullptr;
  NeoPopulationMetricRow* metric_rows = nullptr;
  unsigned long long* accepted_trade_total = nullptr;
  int* overflow_flag = nullptr;

  void release_workspace() {
    device_free(signal_values);
    device_free(signal_confidences);
    device_free(event_counts);
    device_free(event_offsets);
    device_free(events);
    device_free(outcomes);
    device_free(monthly_pnls);
    device_free(month_start_equities);
    device_free(metric_rows);
    device_free(accepted_trade_total);
    device_free(overflow_flag);
  }

  void release() {
    release_workspace();
    device_free(close);
    device_free(high);
    device_free(low);
    device_free(indicators);
    device_free(months);
    device_free(days);
    device_free(timestamps);
    device_free(smc_rows);
    device_free(adaptive_base_pips);
    device_free(gap_flags);
    device_free(candidate_ids);
    device_free(gene_offsets);
    device_free(gene_indices);
    device_free(gene_weights);
    device_free(long_thresholds);
    device_free(short_thresholds);
    device_free(stop_pips);
    device_free(target_pips);
    device_free(stop_vol_multipliers);
    device_free(smc_flags);
    device_free(smc_weights);
    device_free(scenario_ids);
    if (event != nullptr) {
      cudaEventDestroy(event);
      event = nullptr;
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }
};

namespace {

std::int32_t copy_to_device(void* destination,
                            const void* source,
                            std::size_t bytes,
                            cudaStream_t stream) {
  if (bytes == 0) {
    return NEO_POPULATION_STATUS_OK;
  }
  if (cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream) !=
      cudaSuccess) {
    return NEO_POPULATION_STATUS_TRANSFER_FAILED;
  }
  return NEO_POPULATION_STATUS_OK;
}

}  // namespace

extern "C" NeoCudaPopulationSession* neoethos_gpu_cuda_population_create(
    std::uint32_t abi_version,
    std::int32_t device,
    std::size_t max_events,
    std::int32_t* status) {
  const auto fail = [&](std::int32_t code) -> NeoCudaPopulationSession* {
    if (status != nullptr) {
      *status = code;
    }
    return nullptr;
  };
  if (abi_version != NEOETHOS_GPU_ABI_VERSION) {
    return fail(NEO_POPULATION_STATUS_ABI_MISMATCH);
  }
  if (max_events == 0) {
    return fail(NEO_POPULATION_STATUS_INVALID_ARGUMENT);
  }
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return fail(NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE);
  }
  if (device < 0 || device >= device_count) {
    return fail(NEO_POPULATION_STATUS_INVALID_ARGUMENT);
  }
  if (cudaSetDevice(device) != cudaSuccess) {
    return fail(NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE);
  }

  auto* session = new (std::nothrow) NeoCudaPopulationSession();
  if (session == nullptr) {
    return fail(NEO_POPULATION_STATUS_ALLOCATION_FAILED);
  }
  session->device = device;
  session->max_events = static_cast<unsigned long long>(max_events);
  if (cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreateWithFlags(&session->event, cudaEventDisableTiming) != cudaSuccess) {
    session->release();
    delete session;
    return fail(NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE);
  }
  if (status != nullptr) {
    *status = NEO_POPULATION_STATUS_OK;
  }
  return session;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_upload_dataset(
    NeoCudaPopulationSession* session,
    const NeoPopulationDatasetView* dataset) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (dataset == nullptr || dataset->close == nullptr || dataset->high == nullptr ||
      dataset->low == nullptr || dataset->indicators == nullptr || dataset->months == nullptr ||
      dataset->days == nullptr || dataset->timestamps == nullptr ||
      dataset->smc_rows == nullptr) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (dataset->header.abi_version != NEOETHOS_GPU_ABI_VERSION) {
    return NEO_POPULATION_STATUS_ABI_MISMATCH;
  }
  if (session->has_dataset) {
    return NEO_POPULATION_STATUS_DATASET_REUPLOAD;
  }
  const std::size_t bars = static_cast<std::size_t>(dataset->header.row_count);
  const std::size_t features = static_cast<std::size_t>(dataset->header.feature_count);
  if (bars == 0 || features == 0) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (dataset->adaptive_base_pips != nullptr && dataset->adaptive_base_pips_len != bars) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }

  std::int32_t status = NEO_POPULATION_STATUS_OK;
  status = device_alloc(&session->close, bars);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->high, bars);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->low, bars);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->indicators, features * bars);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->months, bars);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->days, bars);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->timestamps, bars);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->smc_rows, bars * kSmcSlots);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->gap_flags, bars);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  if (dataset->adaptive_base_pips != nullptr) {
    status = device_alloc(&session->adaptive_base_pips, bars);
    if (status != NEO_POPULATION_STATUS_OK) return status;
  }

  const std::size_t double_bytes = bars * sizeof(double);
  const std::size_t i64_bytes = bars * sizeof(long long);
  std::uint64_t uploaded = 0;
  status = copy_to_device(session->close, dataset->close, double_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->high, dataset->high, double_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->low, dataset->low, double_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->indicators, dataset->indicators,
                          features * bars * sizeof(float), session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->months, dataset->months, i64_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->days, dataset->days, i64_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->timestamps, dataset->timestamps, i64_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->smc_rows, dataset->smc_rows,
                          bars * kSmcSlots * sizeof(signed char), session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  uploaded = static_cast<std::uint64_t>(3 * double_bytes + 3 * i64_bytes +
                                        features * bars * sizeof(float) +
                                        bars * kSmcSlots * sizeof(signed char));
  if (dataset->adaptive_base_pips != nullptr) {
    status = copy_to_device(session->adaptive_base_pips, dataset->adaptive_base_pips,
                            bars * sizeof(double), session->stream);
    if (status != NEO_POPULATION_STATUS_OK) return status;
    uploaded += static_cast<std::uint64_t>(bars * sizeof(double));
    session->has_adaptive_base = 1;
  }

  session->bars = static_cast<int>(bars);
  session->feature_count = static_cast<int>(features);
  session->dataset_upload_bytes = uploaded;
  session->has_dataset = true;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_upload_genes(
    NeoCudaPopulationSession* session,
    const NeoPopulationGeneView* genes) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (!session->has_dataset) {
    return NEO_POPULATION_STATUS_MISSING_UPLOAD;
  }
  if (genes == nullptr || genes->descriptors == nullptr || genes->offsets == nullptr ||
      genes->stop_pips == nullptr || genes->target_pips == nullptr ||
      genes->stop_vol_multipliers == nullptr || genes->smc_flags == nullptr ||
      genes->smc_weights == nullptr || genes->count == 0) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (genes->term_count > 0 && (genes->indices == nullptr || genes->weights == nullptr)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }

  const std::size_t population = genes->count;
  const std::size_t terms = genes->term_count;
  // Host-side staging of the descriptor-derived arrays keeps the device layout
  // flat and coalesced without changing canonical identity or ordering.
  auto* candidate_ids = new (std::nothrow) unsigned long long[population];
  auto* long_thresholds = new (std::nothrow) float[population];
  auto* short_thresholds = new (std::nothrow) float[population];
  if (candidate_ids == nullptr || long_thresholds == nullptr || short_thresholds == nullptr) {
    delete[] candidate_ids;
    delete[] long_thresholds;
    delete[] short_thresholds;
    return NEO_POPULATION_STATUS_ALLOCATION_FAILED;
  }
  for (std::size_t index = 0; index < population; ++index) {
    candidate_ids[index] = genes->descriptors[index].candidate_id;
    long_thresholds[index] = genes->descriptors[index].long_threshold;
    short_thresholds[index] = genes->descriptors[index].short_threshold;
  }

  const auto cleanup_host = [&]() {
    delete[] candidate_ids;
    delete[] long_thresholds;
    delete[] short_thresholds;
  };

  device_free(session->candidate_ids);
  device_free(session->gene_offsets);
  device_free(session->gene_indices);
  device_free(session->gene_weights);
  device_free(session->long_thresholds);
  device_free(session->short_thresholds);
  device_free(session->stop_pips);
  device_free(session->target_pips);
  device_free(session->stop_vol_multipliers);
  device_free(session->smc_flags);
  device_free(session->smc_weights);

  std::int32_t status = NEO_POPULATION_STATUS_OK;
  const auto guard = [&](std::int32_t code) {
    if (code != NEO_POPULATION_STATUS_OK) {
      status = code;
    }
    return status == NEO_POPULATION_STATUS_OK;
  };

  if (!guard(device_alloc(&session->candidate_ids, population)) ||
      !guard(device_alloc(&session->gene_offsets, population + 1)) ||
      !guard(device_alloc(&session->gene_indices, terms)) ||
      !guard(device_alloc(&session->gene_weights, terms)) ||
      !guard(device_alloc(&session->long_thresholds, population)) ||
      !guard(device_alloc(&session->short_thresholds, population)) ||
      !guard(device_alloc(&session->stop_pips, population)) ||
      !guard(device_alloc(&session->target_pips, population)) ||
      !guard(device_alloc(&session->stop_vol_multipliers, population)) ||
      !guard(device_alloc(&session->smc_flags, population * kSmcSlots)) ||
      !guard(device_alloc(&session->smc_weights, kSmcSlots))) {
    cleanup_host();
    return status;
  }

  if (!guard(copy_to_device(session->candidate_ids, candidate_ids,
                            population * sizeof(unsigned long long), session->stream)) ||
      !guard(copy_to_device(session->gene_offsets, genes->offsets,
                            (population + 1) * sizeof(int), session->stream)) ||
      !guard(copy_to_device(session->gene_indices, genes->indices, terms * sizeof(int),
                            session->stream)) ||
      !guard(copy_to_device(session->gene_weights, genes->weights, terms * sizeof(float),
                            session->stream)) ||
      !guard(copy_to_device(session->long_thresholds, long_thresholds,
                            population * sizeof(float), session->stream)) ||
      !guard(copy_to_device(session->short_thresholds, short_thresholds,
                            population * sizeof(float), session->stream)) ||
      !guard(copy_to_device(session->stop_pips, genes->stop_pips, population * sizeof(double),
                            session->stream)) ||
      !guard(copy_to_device(session->target_pips, genes->target_pips,
                            population * sizeof(double), session->stream)) ||
      !guard(copy_to_device(session->stop_vol_multipliers, genes->stop_vol_multipliers,
                            population * sizeof(double), session->stream)) ||
      !guard(copy_to_device(session->smc_flags, genes->smc_flags,
                            population * kSmcSlots * sizeof(signed char), session->stream)) ||
      !guard(copy_to_device(session->smc_weights, genes->smc_weights,
                            kSmcSlots * sizeof(float), session->stream))) {
    cleanup_host();
    return status;
  }

  if (cudaStreamSynchronize(session->stream) != cudaSuccess) {
    cleanup_host();
    return NEO_POPULATION_STATUS_TRANSFER_FAILED;
  }
  cleanup_host();

  session->population = static_cast<int>(population);
  session->gate_threshold = genes->gate_threshold;
  session->smc_gate_disabled = static_cast<int>(genes->smc_gate_disabled);
  session->gene_upload_bytes = static_cast<std::uint64_t>(
      population * (sizeof(unsigned long long) + 2 * sizeof(float) + 3 * sizeof(double) +
                    kSmcSlots * sizeof(signed char)) +
      (population + 1) * sizeof(int) + terms * (sizeof(int) + sizeof(float)) +
      kSmcSlots * sizeof(float));
  session->has_genes = true;
  session->has_scenarios = false;
  session->metrics_ready = false;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_upload_scenarios(
    NeoCudaPopulationSession* session,
    const NeoPopulationScenarioView* scenarios) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (!session->has_genes) {
    return NEO_POPULATION_STATUS_MISSING_UPLOAD;
  }
  if (scenarios == nullptr || scenarios->descriptors == nullptr) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (scenarios->count != static_cast<std::size_t>(session->population)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }

  const std::size_t population = scenarios->count;
  auto* scenario_ids = new (std::nothrow) unsigned long long[population];
  if (scenario_ids == nullptr) {
    return NEO_POPULATION_STATUS_ALLOCATION_FAILED;
  }
  for (std::size_t index = 0; index < population; ++index) {
    scenario_ids[index] = scenarios->descriptors[index].scenario_id;
  }

  device_free(session->scenario_ids);
  std::int32_t status = device_alloc(&session->scenario_ids, population);
  if (status != NEO_POPULATION_STATUS_OK) {
    delete[] scenario_ids;
    return status;
  }
  status = copy_to_device(session->scenario_ids, scenario_ids,
                          population * sizeof(unsigned long long), session->stream);
  if (status != NEO_POPULATION_STATUS_OK) {
    delete[] scenario_ids;
    return status;
  }
  if (cudaStreamSynchronize(session->stream) != cudaSuccess) {
    delete[] scenario_ids;
    return NEO_POPULATION_STATUS_TRANSFER_FAILED;
  }
  delete[] scenario_ids;

  session->scenario_upload_bytes =
      static_cast<std::uint64_t>(population * sizeof(unsigned long long));
  session->has_scenarios = true;
  session->metrics_ready = false;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_b_evaluate(
    NeoCudaPopulationSession* session,
    const NeoPopulationSettings* settings,
    std::uint64_t* event_id,
    NeoPopulationCounters* counters) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (settings == nullptr || event_id == nullptr) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (!session->has_dataset || !session->has_genes || !session->has_scenarios) {
    return NEO_POPULATION_STATUS_MISSING_UPLOAD;
  }
  if (settings->abi_version != NEOETHOS_GPU_ABI_VERSION) {
    return NEO_POPULATION_STATUS_ABI_MISMATCH;
  }
  if (settings->month_capacity == 0u) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }

  const int bars = session->bars;
  const int population = session->population;
  const int month_capacity = static_cast<int>(settings->month_capacity);
  const std::size_t signal_slots =
      static_cast<std::size_t>(population) * static_cast<std::size_t>(bars);

  if (session->signal_values == nullptr || session->month_capacity != month_capacity ||
      session->workspace_population != population || session->workspace_bars != bars) {
    session->release_workspace();
    std::int32_t status = NEO_POPULATION_STATUS_OK;
    const auto guard = [&](std::int32_t code) {
      if (code != NEO_POPULATION_STATUS_OK) {
        status = code;
      }
      return status == NEO_POPULATION_STATUS_OK;
    };
    if (!guard(device_alloc(&session->signal_values, signal_slots)) ||
        !guard(device_alloc(&session->signal_confidences, signal_slots)) ||
        !guard(device_alloc(&session->event_counts, static_cast<std::size_t>(population))) ||
        !guard(device_alloc(&session->event_offsets,
                            static_cast<std::size_t>(population) + 1)) ||
        // No event buffer at all, and outcomes sized by the trades a candidate
        // can record rather than by every bar it might have entered on. On M3
        // that is 2.4 GB against 180 GB, which is the whole reason the
        // population was splitting 4 096 -> 128.
        !guard(device_alloc(&session->outcomes,
                            static_cast<std::size_t>(population) *
                                static_cast<std::size_t>(kMaxTradesPerCandidate))) ||
        !guard(device_alloc(&session->monthly_pnls,
                            static_cast<std::size_t>(population) * month_capacity)) ||
        !guard(device_alloc(&session->month_start_equities,
                            static_cast<std::size_t>(population) * month_capacity)) ||
        !guard(device_alloc(&session->metric_rows, static_cast<std::size_t>(population))) ||
        !guard(device_alloc(&session->accepted_trade_total, 1)) ||
        !guard(device_alloc(&session->overflow_flag, 1))) {
      session->release_workspace();
      return status;
    }
    session->month_capacity = month_capacity;
    session->workspace_population = population;
    session->workspace_bars = bars;
  }

  if (cudaMemsetAsync(session->accepted_trade_total, 0, sizeof(unsigned long long),
                      session->stream) != cudaSuccess ||
      cudaMemsetAsync(session->overflow_flag, 0, sizeof(int), session->stream) != cudaSuccess) {
    return NEO_POPULATION_STATUS_TRANSFER_FAILED;
  }

  DeviceDataset dataset;
  dataset.close = session->close;
  dataset.high = session->high;
  dataset.low = session->low;
  dataset.indicators = session->indicators;
  dataset.months = session->months;
  dataset.days = session->days;
  dataset.timestamps = session->timestamps;
  dataset.smc_rows = session->smc_rows;
  dataset.adaptive_base_pips = session->adaptive_base_pips;
  dataset.has_adaptive_base = session->has_adaptive_base;
  dataset.bars = bars;
  dataset.feature_count = session->feature_count;

  DeviceGenes genes;
  genes.candidate_ids = session->candidate_ids;
  genes.offsets = session->gene_offsets;
  genes.indices = session->gene_indices;
  genes.weights = session->gene_weights;
  genes.long_thresholds = session->long_thresholds;
  genes.short_thresholds = session->short_thresholds;
  genes.stop_pips = session->stop_pips;
  genes.target_pips = session->target_pips;
  genes.stop_vol_multipliers = session->stop_vol_multipliers;
  genes.smc_flags = session->smc_flags;
  genes.smc_weights = session->smc_weights;
  genes.gate_threshold = session->gate_threshold;
  genes.smc_gate_disabled = session->smc_gate_disabled;
  genes.population = population;

  const unsigned int signal_blocks = static_cast<unsigned int>(
      (signal_slots + kSignalBlock - 1) / kSignalBlock);
  // ── Per-stage timing ────────────────────────────────────────────────────
  //
  // Three times today a bottleneck was named from reading the code and twice it
  // was the wrong one. Wall-clock for the whole evaluation cannot tell signal
  // from emit from first-hit from reduce, so every guess costs a rebuild and a
  // rented card. This costs one env var and a handful of events.
  //
  // Off unless NEOETHOS_GPU_STAGE_TIMING is set: the syncs it needs would
  // serialise the stream, which is exactly what you do not want in production
  // and exactly what you do want when measuring.
  const bool stage_timing = (std::getenv("NEOETHOS_GPU_STAGE_TIMING") != nullptr);
  cudaEvent_t stage_marks[7];
  int stage_mark = 0;
  auto mark_stage = [&]() {
    if (!stage_timing || stage_mark >= 7) {
      return;
    }
    cudaEventCreate(&stage_marks[stage_mark]);
    cudaEventRecord(stage_marks[stage_mark], session->stream);
    stage_mark += 1;
  };
  mark_stage();

  population_signal_kernel<<<signal_blocks == 0u ? 1u : signal_blocks, kSignalBlock, 0,
                             session->stream>>>(dataset, genes, session->signal_values,
                                                session->signal_confidences);
  const unsigned int gap_blocks = static_cast<unsigned int>((bars + 255) / 256);
  population_gap_flags_kernel<<<gap_blocks == 0u ? 1u : gap_blocks, 256, 0, session->stream>>>(
      dataset, *settings, session->gap_flags);
  mark_stage();
  // Not launched: nothing consumes the event stream now that the reduce
  // opens positions from the signal directly.
  //   population_count_events_kernel<<<static_cast<unsigned int>(population), kEmitBlock, 0,
  //                                    session->stream>>>(dataset, genes, session->signal_values,
  //                                                       session->event_counts);
  mark_stage();
  // Not launched: nothing consumes the event stream now that the reduce
  // opens positions from the signal directly.
  //   population_scan_offsets_kernel<<<1, 1, 0, session->stream>>>(
  //       session->event_counts, session->event_offsets, population);
  mark_stage();
  // Not launched: nothing consumes the event stream now that the reduce
  // opens positions from the signal directly.
  //   population_emit_events_kernel<<<static_cast<unsigned int>(population), kEmitBlock, 0,
  //                                   session->stream>>>(dataset, genes, *settings,
  //                                                      session->signal_values,
  //                                                      session->event_offsets,
  //                                                      session->scenario_ids, session->events,
  //                                                      session->max_events,
  //                                                      session->overflow_flag);
  mark_stage();
  session->kernel_submissions += 5;

  // No event total to read and no capacity to guard: the reduce opens positions
  // from the signal, so there is nothing whose size the host has to check before
  // launching. That readback was a stream synchronization every generation, and
  // the capacity it guarded is what split the population 4 096 -> 128 on M3.
  //
  // The overflow flag still matters — it reports a gene whose trades exceeded
  // its slice — but it is read with the metrics at the end rather than blocking
  // here.
  const unsigned long long trade_slots =
      static_cast<unsigned long long>(population) * kMaxTradesPerCandidate;
  {
    const unsigned int seed_threads = 256u;
    const unsigned long long seed_blocks = (trade_slots + seed_threads - 1ull) / seed_threads;
    population_seed_outcomes_kernel<<<static_cast<unsigned int>(seed_blocks), seed_threads, 0,
                                      session->stream>>>(session->outcomes, trade_slots);
    session->kernel_submissions += 1;
  }
  mark_stage();

  const unsigned int reduce_blocks =
      static_cast<unsigned int>((population + kReduceBlock - 1) / kReduceBlock);
  population_reduce_kernel<<<reduce_blocks == 0u ? 1u : reduce_blocks, kReduceBlock, 0,
                             session->stream>>>(dataset, genes, *settings,
                                                session->signal_values,
                                                session->signal_confidences,
                                                session->gap_flags, session->outcomes, session->event_offsets,
                                                session->scenario_ids, session->monthly_pnls,
                                                session->month_start_equities,
                                                session->metric_rows,
                                                session->accepted_trade_total);
  mark_stage();
  if (stage_timing) {
    cudaStreamSynchronize(session->stream);
    static const char* kStageNames[6] = {"signal", "gap_flags", "count_events",
                                         "scan_offsets", "emit+first_hit", "reduce"};
    std::fprintf(stderr, "[gpu-stage-timing] population=%lld bars=%lld events=%llu\n",
                 static_cast<long long>(population), static_cast<long long>(bars),
                 static_cast<unsigned long long>(trade_slots));
    for (int i = 1; i < stage_mark; ++i) {
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, stage_marks[i - 1], stage_marks[i]);
      std::fprintf(stderr, "[gpu-stage-timing]   %-14s %8.1f ms\n",
                   (i - 1) < 6 ? kStageNames[i - 1] : "?", ms);
    }
    for (int i = 0; i < stage_mark; ++i) {
      cudaEventDestroy(stage_marks[i]);
    }
  }
  session->kernel_submissions += 1;

  if (cudaEventRecord(session->event, session->stream) != cudaSuccess) {
    return NEO_POPULATION_STATUS_LAUNCH_FAILED;
  }
  if (cudaGetLastError() != cudaSuccess) {
    return NEO_POPULATION_STATUS_LAUNCH_FAILED;
  }

  // No events are emitted any more. The diagnostic keeps its meaning — how much
  // work the population generated — by reporting the trade slots it could use,
  // rather than a count of candidate entries that no longer exist.
  session->emitted_events = trade_slots;
  session->pending_event_id = session->next_event_id;
  session->next_event_id += 1ull;
  session->metrics_ready = false;
  *event_id = session->pending_event_id;

  if (counters != nullptr) {
    std::memset(counters, 0, sizeof(NeoPopulationCounters));
    counters->event_count = trade_slots;
    counters->kernel_submissions = session->kernel_submissions;
    counters->synchronization_events = session->synchronization_events;
    counters->dataset_upload_bytes = session->dataset_upload_bytes;
    counters->gene_upload_bytes = session->gene_upload_bytes;
    counters->scenario_upload_bytes = session->scenario_upload_bytes;
  }
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_wait(NeoCudaPopulationSession* session,
                                                          std::uint64_t event_id) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (event_id == 0ull || event_id != session->pending_event_id) {
    return NEO_POPULATION_STATUS_UNKNOWN_EVENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  if (cudaEventSynchronize(session->event) != cudaSuccess) {
    return NEO_POPULATION_STATUS_SYNC_FAILED;
  }
  if (cudaGetLastError() != cudaSuccess) {
    return NEO_POPULATION_STATUS_LAUNCH_FAILED;
  }
  unsigned long long accepted = 0ull;
  if (cudaMemcpy(&accepted, session->accepted_trade_total, sizeof(unsigned long long),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return NEO_POPULATION_STATUS_TRANSFER_FAILED;
  }
  session->accepted_trades = accepted;
  session->synchronization_events += 1;
  session->metrics_ready = true;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_read_metrics(
    NeoCudaPopulationSession* session,
    NeoPopulationReadback* readback) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (readback == nullptr || readback->rows == nullptr || readback->written == nullptr) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (!session->metrics_ready) {
    return NEO_POPULATION_STATUS_MISSING_UPLOAD;
  }
  const std::size_t rows = static_cast<std::size_t>(session->population);
  if (readback->capacity < rows) {
    return NEO_POPULATION_STATUS_READBACK_CAPACITY;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  if (cudaMemcpy(readback->rows, session->metric_rows, rows * sizeof(NeoPopulationMetricRow),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return NEO_POPULATION_STATUS_TRANSFER_FAILED;
  }
  *readback->written = rows;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_read_diagnostics(
    NeoCudaPopulationSession* session,
    NeoPopulationDiagnosticReadback* readback) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (readback == nullptr || readback->events == nullptr || readback->outcomes == nullptr ||
      readback->written == nullptr) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (!session->metrics_ready) {
    return NEO_POPULATION_STATUS_MISSING_UPLOAD;
  }
  const std::size_t count = static_cast<std::size_t>(session->emitted_events);
  if (readback->capacity < count) {
    return NEO_POPULATION_STATUS_READBACK_CAPACITY;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  if (count > 0) {
    // There is no event stream any more — the reduce opens positions from the
    // signal — so the events half is zeroed rather than copied from a buffer
    // that is not allocated. Callers that want entries read them from the
    // outcomes, which now carry `entry_bar` for exactly this reason.
    std::memset(readback->events, 0, count * sizeof(NeoPopulationEvent));
    if (cudaMemcpy(readback->outcomes, session->outcomes, count * sizeof(NeoPopulationOutcome),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return NEO_POPULATION_STATUS_TRANSFER_FAILED;
    }
  }
  *readback->written = count;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" void neoethos_gpu_cuda_population_destroy(NeoCudaPopulationSession* session) {
  if (session == nullptr) {
    return;
  }
  cudaSetDevice(session->device);
  session->release();
  delete session;
}
