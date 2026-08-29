// Persistent native-CUDA Prototype B population engine.
//
// One session owns one non-default stream, one logical dataset upload and every
// device workspace. `evaluate` runs the complete canonical chain on that
// stream. P1-C still reads every population metric row back to the host and
// records that intermediate D2H boundary explicitly; P1-E must remove it
// before the run can claim final-result-only readback.
//
// The chain used to be five kernels passing `population * bars` arrays between
// them. It is now two: a bar-parallel gap-flag pass, and ONE walk kernel that
// synthesises the signal, opens and closes the position, and reduces to
// metrics — all in registers, one thread per scenario. Nothing candidate-sized
// crosses a kernel boundary any more, which took the per-scenario device cost
// from 4 811 048 B to 593 768 B and let the population stop being bounded by
// memory at all.
//
// The sealed V1 parent keeps the exact feature-major matrix resident and the
// walk indexes it through the bound parent/view descriptor. The legacy upload
// ABI still transposes its compatibility-only dataset into bar-major storage.
//
// The semantics reproduced here are the canonical ones expressed by the
// validation oracle in `prototype_population_oracle.rs`. Any divergence is a
// correctness failure, not a tuning opportunity.

#include "neoethos_gpu_cuda.h"

#include <cuda_runtime.h>

#include <climits>
#include <cstring>
#include <new>

namespace {

/// Spread in pips for one bar, from its UTC hour.
///
/// Mirrors `SessionSpreadProfile::spread_pips_at` and
/// `BacktestSettings::spread_pips_for_bar` term for term: the same hour
/// arithmetic, the same half-open ranges, and the same `timestamp_ms > 0`
/// guard that makes a missing timestamp fall back to the scalar. The CPU has
/// resolved spread per bar since the profile type existed; the device charged
/// one number at every hour, so turning the profile on would have made the two
/// lanes disagree with no test to catch it — every parity fixture leaves the
/// profile unset.
///
/// With no profile the host writes `spread_pips` into all three buckets, so
/// this returns the scalar and the result is bit-identical.
__device__ inline double spread_pips_for_bar(const NeoPopulationSettings& settings,
                                             std::int64_t timestamp_ms) {
  if (timestamp_ms <= 0) {
    return settings.spread_pips;
  }
  const std::int64_t secs = timestamp_ms / 1000LL - (timestamp_ms % 1000LL < 0 ? 1LL : 0LL);
  std::int64_t hour = secs / 3600LL - (secs % 3600LL < 0 ? 1LL : 0LL);
  hour = ((hour % 24LL) + 24LL) % 24LL;
  if (hour >= 7 && hour < 16) {
    return settings.spread_pips_overlap;
  }
  if (hour >= 16 && hour < 22) {
    return settings.spread_pips_late_ny;
  }
  return settings.spread_pips_asian;
}

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
// Trade slots per candidate. Measured need is ~3 000 trades per gene over
// 439 315 bars; this leaves room for the densest genes the search produces
// without sizing device memory by candidate entries, of which there are a
// hundred times more.
constexpr unsigned long long kMaxTradesPerCandidate = 8192ull;

// Square tile for the one-off feature-major -> bar-major transpose. 32x32 keeps
// both the read and the write coalesced; the +1 pad breaks shared-bank
// conflicts on the transposed read.
constexpr int kTransposeTile = 32;

// ── THE SCENARIO IS THE UNIT OF WORK ─────────────────────────────────────────
//
// A launch used to evaluate a GENE ARRAY under ONE settings struct. One thread
// per gene, one metric row per gene, and anything that wanted the same genes
// under a different treatment needed a different launch: the Monte-Carlo screen
// staged 17 400 perturbed CLONES on the host and sent them in six chunks, and
// the cost-sensitivity screen sent the same 174 genes a seventh time with two
// scalars changed.
//
// A launch now evaluates a SCENARIO ARRAY. One thread per scenario, one metric
// row per scenario, and each scenario names its own gene, its own window, its
// own costs and its own perturbation counter. The measured quality screen — 174
// candidates x (100 Monte-Carlo + 1 sensitivity) — is one array and one launch.
//
// Every existing caller uploads exactly one BASE scenario per gene with zeroed
// fields, and that case is bit-identical to what ran before: the walk bounds
// resolve to `1 .. bars`, both cost overrides resolve to the settings values,
// and no perturbation is applied. That identity is the parity floor for this
// whole change, and it is the property the 147-test suite must confirm.
//
// The codes and scales below are mirrored in
// `crates/neoethos-search/src/gpu_native/scenario.rs` and checked against THIS
// FILE by `the_scenario_type_codes_match_the_kernel` and
// `the_fixed_point_scales_match_the_kernel`, which parse these declarations.
// Two languages agreeing about a number by convention is how the retry-smaller
// path silently stopped working; these are checked.
constexpr unsigned kScenarioBase = 0u;
constexpr unsigned kScenarioPerturb = 1u;
constexpr unsigned kScenarioCost = 2u;

// Fixed-point scales for the descriptor's integer cost fields.
//
// The conversion is a DIVISION by the scale, never a multiplication by its
// reciprocal: 1000.0 is exactly representable and 1e-3 is not, so `1400 /
// 1000.0` is the correctly-rounded double nearest 1.4 — the same one the host's
// literal parses to — while `1400 * 1e-3` has no such guarantee. The host
// refuses any cost that does not round-trip through this exact division rather
// than rounding it, so a spread the descriptor cannot carry becomes an error
// the operator reads and never a launch that quietly charged a different one.
constexpr double kTicksPerPip = 1000.0;
constexpr double kMicrosPerUnit = 1000000.0;

// "No override — use the settings value."
//
// A sentinel rather than zero, because ZERO IS A LEGITIMATE OVERRIDE: "what
// does this strategy look like paying no spread at all" is a question the
// sensitivity screen may ask, and a zero meaning "unset" makes it unaskable.
// A cost cannot be negative, so -1 is free.
constexpr int kNoTickOverride = -1;
constexpr long long kNoMicroOverride = -1;

// ── The perturbation generator ───────────────────────────────────────────────
//
// THIS IS NOT ChaCha8 AND IT IS NOT TRYING TO BE.
//
// The shipped Monte-Carlo screen perturbs on the HOST with
// `rand_chacha::ChaCha8Rng::seed_from_u64(...)`, and that lane remains the
// default and the reference. ChaCha8 is a stream cipher: the 4 097th draw
// requires the 4 096 before it, and `rand`'s `random_range` consumes a variable
// number of words per draw through rejection sampling. A GPU thread cannot walk
// that stream, so reproducing it here is not a matter of effort — it is a
// different shape of computation.
//
// What runs here instead is counter-based: the k-th draw of stream c is a pure
// function `hash(c, k)`, computable by any thread in O(1) with no state. It is
// fully deterministic and fully reproducible; it is simply a DIFFERENT SEQUENCE.
//
// That difference is why the host lane is the default, why turning this on is an
// explicit act, and why the CPU fallback runs THIS generator (not the gene as
// uploaded) when a device-perturbation scenario falls back — see
// `scenario.rs::perturbed_gene`. A fallback that silently evaluated the
// unperturbed gene would report a Monte-Carlo pass count computed from no
// perturbation at all, with every downstream number still looking plausible.
//
// splitmix64 is used because it is four shift/multiply/xor operations on one
// 64-bit integer with no state, no tables and no branches, so the Rust and CUDA
// transcriptions are character-for-character the same arithmetic. Unsigned
// overflow is defined in both languages and is what `wrapping_*` means there.
__device__ inline unsigned long long splitmix64(unsigned long long value) {
  unsigned long long z = value + 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/// The `draw`-th uniform of stream `counter`, in [0, 1).
///
/// Exact on both lanes and rounding-free: `mixed >> 11` is strictly below 2^53
/// so the conversion to double is exact, and 1/2^53 is exactly representable so
/// the multiply is exact too. That is what makes a CUDA transcription
/// bit-identical to the Rust one rather than merely close.
__device__ inline double perturb_unit(unsigned long long counter, unsigned long long draw) {
  const unsigned long long mixed =
      splitmix64(counter ^ splitmix64(draw * 0x9E3779B97F4A7C15ull));
  return static_cast<double>(mixed >> 11) * (1.0 / 9007199254740992.0);
}

/// `1 + U(-amplitude, amplitude)`.
///
/// Two separate operations on purpose: `-fmad=false` (build.rs) forbids the
/// compiler contracting either `2*u - 1` or `1 + m*a` into an FMA, so this
/// rounds at the same two places the Rust mirror does. Without that flag the two
/// would be identical on paper and not in practice.
__device__ inline double perturb_factor(unsigned long long counter,
                                        unsigned long long draw,
                                        double amplitude) {
  const double unit = perturb_unit(counter, draw);
  const double centred = 2.0 * unit - 1.0;
  return 1.0 + centred * amplitude;
}

constexpr double kMcThresholdAmplitude = 0.15;
constexpr double kMcWeightAmplitude = 0.20;
constexpr double kMcStopAmplitude = 0.25;
// Draw indices. The weights occupy a variable-length block, so the stop and
// target indices move with the gene's term count — getting that wrong would not
// fail, it would perturb by a different and equally plausible number.
constexpr unsigned long long kDrawLongThreshold = 0ull;
constexpr unsigned long long kDrawShortThreshold = 1ull;
constexpr unsigned long long kDrawWeightBase = 2ull;

// ── The reduce block size is CHOSEN, not fixed ───────────────────────────────
//
// It was a constant 128, and that constant is a large part of why the card
// measured at 2.6 % of capacity.
//
// The walk is one thread per scenario and cannot be anything else: equity,
// drawdown, the month and day cursors, the open position and the trailing stop
// are a serial recurrence over bars, so bar b cannot be evaluated before b-1.
// Splitting a scenario across lanes would change the order in which those
// values are produced, which is precisely what parity forbids. The parallelism
// therefore comes from the number of scenarios in flight, and the only knob the
// launch itself owns is how those threads are distributed.
//
// Distribution mattered more than anyone measured. At the population the old
// memory budget allowed — 3 300 candidates — a 128-thread block yields 26
// blocks. An RTX 3090 has 82 SMs, so 56 of them received NO WORK AT ALL and the
// remaining 26 ran one block each: 3 300 threads against ~126 000 resident
// capacity, 2.6 %, each thread serially walking 843 456 bars. A 2 s nvidia-smi
// sampler reports that as "busy".
//
// Ampere hosts at most 16 blocks per SM, so a 32-thread block caps residency at
// 512 threads/SM, 64 at 1 024, and 128 reaches the 1 536-thread hardware limit.
// The rule below therefore takes the SMALLEST block that still has room for the
// requested population, which is the one that spreads a small population over
// the most SMs, and steps up only when the population would otherwise exceed
// that block's residency ceiling.
constexpr int kReduceBlockSmall = 32;
constexpr int kReduceBlockMedium = 64;
constexpr int kReduceBlockLarge = 128;
constexpr int kAmpereBlocksPerSm = 16;

__host__ inline int choose_reduce_block(int population, int sm_count) {
  const int sms = sm_count > 0 ? sm_count : 1;
  if (population <= sms * kReduceBlockSmall * kAmpereBlocksPerSm) {
    return kReduceBlockSmall;
  }
  if (population <= sms * kReduceBlockMedium * kAmpereBlocksPerSm) {
    return kReduceBlockMedium;
  }
  return kReduceBlockLarge;
}

__device__ inline double guarded_pip(double pip_value) {
  return (fabs(pip_value) < 1.0e-12) ? 1.0e-12 : pip_value;
}

__device__ inline double sanitize(double value) {
  return isfinite(value) ? value : 0.0;
}

__device__ inline double invalid_monthly_return_sharpe_v1() {
  return -__longlong_as_double(static_cast<long long>(0x7ff0000000000000ULL));
}

struct DeviceDataset {
  const double* close;
  const double* high;
  const double* low;
  // BAR-MAJOR: element (bar, feature) lives at `bar * feature_count + feature`.
  //
  // The host contract is unchanged — `NeoPopulationDatasetView::indicators` is
  // still feature-major, the CPU oracle, the parity fixtures and prototypes A
  // and C all still see feature-major — and the device transposes once at
  // upload. The field is RENAMED rather than reinterpreted so that any reader
  // that was not updated fails to compile instead of silently reading the
  // matrix through the wrong stride.
  //
  // Why: every thread in a reduce block is at the same bar (the branchy exit
  // logic changes WHAT is computed, not WHICH bar), and each thread wants up to
  // 16 features of that bar. Feature-major puts those at a 3.37 MB stride —
  // 64 distinct cache lines per warp-bar. Bar-major makes the whole feature row
  // of one bar a single contiguous 512-byte run that the warp shares.
  const double* indicators_bar_major;
  const double* indicators_feature_major;
  const std::int64_t* months;
  const std::int64_t* days;
  const std::int64_t* timestamps;
  const signed char* smc_rows;
  const unsigned long long* view_indices;
  const double* adaptive_base_pips;
  int has_adaptive_base;
  int bars;
  int parent_rows;
  int feature_count;
  int view_kind;
  int view_start;
  int timestamp_mode;
};

__device__ inline int population_parent_row(const DeviceDataset& dataset, int view_row) {
  if (dataset.view_kind == static_cast<int>(NEO_POPULATION_VIEW_ORDERED_INDICES)) {
    return static_cast<int>(dataset.view_indices[view_row]);
  }
  return dataset.view_start + view_row;
}

__device__ inline std::int64_t population_timestamp_at(const DeviceDataset& dataset,
                                                       int view_row) {
  if (dataset.timestamp_mode ==
      static_cast<int>(NEO_POPULATION_TIMESTAMP_DISABLED_INDEX_DELTA)) {
    return 0ll;
  }
  return dataset.timestamps[population_parent_row(dataset, view_row)];
}

__device__ inline double population_feature_at(const DeviceDataset& dataset,
                                                int view_row,
                                                int feature) {
  const int parent_row = population_parent_row(dataset, view_row);
  if (dataset.indicators_feature_major != nullptr) {
    return dataset.indicators_feature_major[
        static_cast<long long>(feature) * dataset.parent_rows + parent_row];
  }
  return dataset.indicators_bar_major[
      static_cast<long long>(parent_row) * dataset.feature_count + feature];
}

struct DeviceGenes {
  const unsigned long long* candidate_ids;
  const int* offsets;
  const int* indices;
  const double* weights;
  const double* long_thresholds;
  const double* short_thresholds;
  const double* stop_pips;
  const double* target_pips;
  const double* stop_vol_multipliers;
  const signed char* smc_flags;
  const double* smc_weights;
  double gate_threshold;
  int smc_gate_disabled;
  // No `population`. It bounded the old `candidate >= genes.population` early
  // return, and the thread's identity is the SCENARIO now — the gene index
  // comes from the descriptor and is validated against the uploaded population
  // twice before it reaches the device (the Rust wrapper names the offending
  // index, the native upload refuses the batch). Keeping an extent the device
  // never reads is exactly the shape of the phantom event buffer that had the
  // host reserving VRAM for an allocation that did not exist.
};

/// The work list. One entry per thread of the walk.
///
/// Struct-of-arrays rather than an array of `NeoScenarioDescriptor`: threads in
/// a block read the same field of consecutive scenarios, so SoA makes each read
/// one coalesced run where AoS would stride 64 bytes and touch a separate cache
/// line per lane. The host descriptor layout is unchanged — it is unpacked once
/// at upload, on the host, exactly as the gene descriptors already are.
struct DeviceScenarios {
  /// Which gene this scenario evaluates. Validated on the host to be inside the
  /// uploaded population; an out-of-range value here is an out-of-bounds read of
  /// thresholds and CSR offsets that would still produce a metric row.
  const unsigned long long* base_candidate_ids;
  /// Reported back in the metric row, so the host can demultiplex a mixed array
  /// without relying on position.
  const unsigned long long* ids;
  /// Stream for `perturb_factor`. Only read for `kScenarioPerturb`.
  const unsigned long long* rng_counters;
  const unsigned long long* window_offsets;
  /// 0 means "to the end of the series".
  const unsigned int* window_lens;
  const unsigned int* types;
  /// Millipips; `kNoTickOverride` leaves the settings' per-bar spread in place.
  const int* spread_ticks;
  /// Millipips of adverse fill, applied at entry and at exit. 0 is the default
  /// and takes a branch that adds no arithmetic at all, so the CPU engine —
  /// which models no slippage — stays bit-identical.
  const int* slippage_ticks;
  /// Micro-units per lot; `kNoMicroOverride` leaves the settings' commission.
  const long long* commission_micros;
  int count;
};

// ---------------------------------------------------------------------------
// Stage 0: one-off layout change, feature-major -> bar-major
// ---------------------------------------------------------------------------

/// `dst[bar * features + feature] = src[feature * bars + bar]`.
///
/// Pure data movement: no arithmetic touches the values, so every double the
/// walk later reads is bit-identical to the one the feature-major kernel read.
/// Runs once per dataset upload, on the session stream, before anything else
/// can observe the matrix.
__global__ void transpose_indicators_to_bar_major(const double* __restrict__ source,
                                                  double* __restrict__ destination,
                                                  int bars,
                                                  int features) {
  __shared__ double tile[kTransposeTile][kTransposeTile + 1];

  const int bar_base = static_cast<int>(blockIdx.x) * kTransposeTile;
  const int feature_base = static_cast<int>(blockIdx.y) * kTransposeTile;

  {
    const int bar = bar_base + static_cast<int>(threadIdx.x);
    const int feature = feature_base + static_cast<int>(threadIdx.y);
    if (bar < bars && feature < features) {
      tile[threadIdx.y][threadIdx.x] =
          source[static_cast<long long>(feature) * bars + bar];
    }
  }
  // Every thread reaches this barrier: the bounds tests above guard the
  // memory access, never the control flow.
  __syncthreads();
  {
    const int feature = feature_base + static_cast<int>(threadIdx.x);
    const int bar = bar_base + static_cast<int>(threadIdx.y);
    if (bar < bars && feature < features) {
      destination[static_cast<long long>(bar) * features + feature] =
          tile[threadIdx.x][threadIdx.y];
    }
  }
}

// ---------------------------------------------------------------------------
// Stage 1: signal synthesis — FUSED INTO THE WALK, computed in registers
// ---------------------------------------------------------------------------
//
// This was `population_signal_kernel`, a `population x bars` grid writing
// `signal_values` (i8) and `signal_confidences` (f32). Those two arrays existed
// for one reason: to carry a value from that kernel to the reduce. At 843 456
// bars they cost 5 bytes per candidate-bar = 4 217 280 B per candidate, which
// is 87.7 % of the 4 811 048 B the host budgeted per scenario — the sole reason
// a 24 GB card resolved to 3 316 scenarios and the host grew a six-chunk loop.
//
// Synthesis is pointwise in the bar: nothing it computes depends on any other
// bar. A thread already walking bars ascending can therefore produce the value
// in registers at the moment it is consumed, and the arrays disappear. Per
// scenario the device cost falls from 4 811 048 B to 593 768 B with the trade
// slots kept (3 944 B if they are ever dropped).
//
// PARITY. The fusion is bit-identical BY CONSTRUCTION, not by measurement:
//   * the CSR terms accumulate in the same ascending order into the same f64
//     accumulator — no reassociation, no tree reduction, no lane splitting;
//   * `-fmad=false` (build.rs:92, with its own measured justification) forbids
//     the compiler from contracting `weights[t] * indicator + acc` into an FMA,
//     which is the only way the same source order could still produce a
//     different bit pattern;
//   * the threshold comparisons, the confidence clamp, the SMC active sum and
//     the SMC score loop are transcribed term for term, slot order included;
//   * bar-major is a transpose of the input, so the double VALUE at every
//     (bar, feature) is unchanged.
//
// What IS hoisted out of the bar loop is exactly the part that has no bar in
// it: the CSR window, the two thresholds, the threshold gap, the SMC active sum
// and the gate. Hoisting a loop-invariant expression does not reassociate
// anything — the same additions happen in the same order, once per candidate
// instead of once per candidate-bar.

/// The per-candidate half of signal synthesis, computed once per thread.
struct SignalPlan {
  int term_start;
  int term_end;
  double long_threshold;
  double short_threshold;
  /// `fabs(long - short)` in the canonical f64 threshold domain.
  double gap;
  /// Sum of the weights of this candidate's active SMC slots, ascending.
  double active_sum;
  /// `fmin(gate_threshold, active_sum)`.
  double gate;
  /// Bit `s` set iff `smc_flags[candidate * kSmcSlots + s] != 0`. Read once
  /// instead of eleven strided bytes per bar.
  unsigned smc_mask;
  /// Non-zero when this scenario perturbs the gene, in which case every CSR
  /// weight read below is scaled by `perturb_factor(rng_counter, 2 + t, ...)`.
  /// Zero on every existing path, and the branch it guards adds no arithmetic.
  int perturbed;
  unsigned long long rng_counter;
};

/// The bar-independent half of synthesis, and the whole of the perturbation.
///
/// `perturbed` / `counter` come from the scenario, not the gene. The thresholds
/// are perturbed HERE, before the gap is taken from them, because the host lane
/// perturbs the gene and then computes the gap from the perturbed thresholds —
/// taking the gap first would divide the confidence by the wrong number and
/// change every position size in the run.
__device__ inline SignalPlan build_signal_plan(const DeviceGenes& genes,
                                               int candidate,
                                               int perturbed,
                                               unsigned long long counter) {
  SignalPlan plan;
  plan.term_start = genes.offsets[candidate];
  plan.term_end = genes.offsets[candidate + 1];
  plan.long_threshold = genes.long_thresholds[candidate];
  plan.short_threshold = genes.short_thresholds[candidate];
  plan.perturbed = perturbed;
  plan.rng_counter = counter;
  if (perturbed != 0) {
    plan.long_threshold *=
        perturb_factor(counter, kDrawLongThreshold, kMcThresholdAmplitude);
    plan.short_threshold *=
        perturb_factor(counter, kDrawShortThreshold, kMcThresholdAmplitude);
  }

  double gap = fabs(plan.long_threshold - plan.short_threshold);
  if (!(gap > 1.0e-6)) {
    gap = 1.0e-6;
  }
  plan.gap = gap;

  // Same ascending slot order and the same f64 accumulator as the kernel this
  // replaces; only the number of times it runs changed.
  unsigned mask = 0u;
  double active_sum = 0.0;
  for (int slot = 0; slot < kSmcSlots; ++slot) {
    if (genes.smc_flags[static_cast<long long>(candidate) * kSmcSlots + slot] != 0) {
      mask |= (1u << slot);
      active_sum += genes.smc_weights[slot];
    }
  }
  if (genes.smc_gate_disabled != 0) {
    active_sum = 0.0;
  }
  plan.smc_mask = mask;
  plan.active_sum = active_sum;
  plan.gate = fmin(genes.gate_threshold, active_sum);
  return plan;
}

/// The per-bar half. Returns the emitted direction (-1, 0, +1) and writes the
/// confidence, exactly as the pair of device arrays used to carry them.
__device__ inline signed char synthesize_signal(const DeviceDataset& dataset,
                                                const DeviceGenes& genes,
                                                const SignalPlan& plan,
                                                int bar,
                                                double* confidence_out) {
  // Terms accumulate in ascending CSR order, matching the canonical f64
  // accumulation order bit for bit.
  double combined = 0.0;
  if (plan.perturbed == 0) {
    for (int term = plan.term_start; term < plan.term_end; ++term) {
      const int feature = genes.indices[term];
      combined += genes.weights[term] * population_feature_at(dataset, bar, feature);
    }
  } else {
    // The perturbed loop is written out rather than folded into the one above
    // with a conditional factor, so the unperturbed path — every existing
    // caller, and the one the 147 parity tests exercise — keeps EXACTLY the
    // instruction sequence it had and performs no unnecessary perturbation
    // arithmetic.
    //
    // Term ordinal, not term index: the draw stream is per gene, so a gene's
    // first term is always draw 2 whatever its offset into the shared CSR
    // arrays is. Using `term` directly would make a gene's perturbation depend
    // on how many terms the genes before it had.
    for (int term = plan.term_start; term < plan.term_end; ++term) {
      const int feature = genes.indices[term];
      const unsigned long long ordinal =
          static_cast<unsigned long long>(term - plan.term_start);
      const double weight = genes.weights[term] *
                            perturb_factor(plan.rng_counter, kDrawWeightBase + ordinal,
                                           kMcWeightAmplitude);
      combined += weight * population_feature_at(dataset, bar, feature);
    }
  }

  signed char signal = 0;
  if (combined >= plan.long_threshold) {
    signal = 1;
  } else if (combined <= plan.short_threshold) {
    signal = -1;
  }

  if (signal == 0) {
    *confidence_out = 0.0;
    return 0;
  }

  const double margin =
      (signal == 1) ? (combined - plan.long_threshold) : (plan.short_threshold - combined);
  const double confidence = fmin(fmax(margin / plan.gap, 0.0), 1.0);

  bool passes_gate = true;
  if (plan.active_sum > 0.0) {
    double score = 0.0;
    for (int slot = 0; slot < kSmcSlots; ++slot) {
      if (((plan.smc_mask >> slot) & 1u) == 0u) {
        continue;
      }
      const int parent_row = population_parent_row(dataset, bar);
      const signed char row =
          dataset.smc_rows[static_cast<long long>(parent_row) * kSmcSlots + slot];
      if (slot == 5) {
        if (row == 1) {
          score += genes.smc_weights[slot];
        }
      } else if (row == signal) {
        score += genes.smc_weights[slot];
      }
    }
    passes_gate = score >= plan.gate;
  }

  if (!passes_gate) {
    *confidence_out = 0.0;
    return 0;
  }
  *confidence_out = confidence;
  return signal;
}

// ---------------------------------------------------------------------------
// Stage 2: entry levels
// ---------------------------------------------------------------------------

__device__ inline void entry_stop_target_pips(const DeviceDataset& dataset,
                                              const DeviceGenes& genes,
                                              const NeoPopulationSettings& settings,
                                              const SignalPlan& plan,
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
  double stop = genes.stop_pips[candidate];
  double target = genes.target_pips[candidate];
  if (plan.perturbed != 0) {
    // The draws come AFTER the weights, so their indices depend on this gene's
    // term count — which is why the plan carries the CSR window rather than the
    // caller passing a constant. The finite-and-positive guards mirror the host
    // screen's exactly: a gene with no fixed stop must not acquire one by being
    // multiplied, and NaN * factor is a stop the walk would then act on.
    //
    // Note the adaptive branch above returns first. That is correct and matches
    // the host: when volatility-scaled stops are active the gene's fixed pips
    // are not used at all, so perturbing them changes nothing on either lane.
    const unsigned long long terms =
        static_cast<unsigned long long>(plan.term_end - plan.term_start);
    if (isfinite(stop) && stop > 0.0) {
      stop *= perturb_factor(plan.rng_counter, kDrawWeightBase + terms, kMcStopAmplitude);
    }
    if (isfinite(target) && target > 0.0) {
      target *= perturb_factor(plan.rng_counter, kDrawWeightBase + terms + 1ull, kMcStopAmplitude);
    }
  }
  *stop_pips = stop;
  *target_pips = target;
}

// The event-stream kernels are GONE: population_count_events_kernel,
// population_scan_offsets_kernel and population_emit_events_kernel.
//
// All three consumed `signal_values`, which no longer exists, and none of them
// has been launched since the reduce started opening positions from the signal
// directly. They were the last readers of the intermediate arrays this change
// deletes. Compiled-but-unlaunched kernels are exactly what kept the host
// budgeting VRAM for a buffer with no allocation (the `session->events`
// phantom), so they are removed rather than left as documentation of a design
// that is no longer in the file.

// ---------------------------------------------------------------------------
// Stage 3: per-bar gap flags (bar-parallel, candidate-independent)
// ---------------------------------------------------------------------------

__global__ void population_gap_flags_kernel(DeviceDataset dataset,
                                            NeoPopulationSettings settings,
                                            unsigned char* gap_flags) {
  for (int bar = blockIdx.x * blockDim.x + threadIdx.x; bar < dataset.bars;
       bar += blockDim.x * gridDim.x) {
    unsigned char flag = 0u;
    if (bar > 0 && settings.gap_threshold_ms > 0) {
      const std::int64_t previous = population_timestamp_at(dataset, bar - 1);
      const std::int64_t current = population_timestamp_at(dataset, bar);
      if (current > previous && (current - previous) >= settings.gap_threshold_ms) {
        flag = 1u;
      }
    }
    gap_flags[bar] = flag;
  }
}

// population_first_hit_kernel is GONE too.
//
// It resolved one event per warp (or per thread with a trailing stop) out of an
// event stream, and no event stream has been produced since the reduce took
// over exit detection — every launch site for it disappeared with the emit
// kernel. Its exit-priority helper went with it. Everything it did now happens
// inline in the reduce, against the position it is holding, which is what
// removed the per-event outcome buffer in the first place.

// ---------------------------------------------------------------------------
// Stage 4: THE WALK — synthesis, entry, exit, cost, sizing and metrics, in one
//          kernel, one thread per scenario
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
                                                    std::int64_t entry_timestamp,
                                                    std::int64_t exit_timestamp,
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

__device__ NeoPopulationOutcome* diagnostic_outcome_slot_v1(
    NeoPopulationOutcome* outcomes,
    unsigned long long position_index,
    unsigned long long range_end) {
  if (outcomes == nullptr || position_index >= range_end) {
    return nullptr;
  }
  return outcomes + position_index;
}

// ── One thread per scenario, and the synthesis is inside it ─────────────────
//
// Two parameters are gone: `signal_values` and `signal_confidences`. They were
// a `population * bars` handoff from a kernel that no longer exists, and the
// thread that consumed them is the thread that can produce them — see
// `synthesize_signal`. A third, `event_offsets`, was passed and never read.
//
// The mapping stays one thread per scenario because the walk below is a serial
// recurrence over bars (equity, drawdown, the month and day cursors, the open
// position, the ratcheting trail). What changed is how many of these threads
// can exist at once: the per-scenario device cost fell from 4 811 048 B to
// 593 768 B, so the same 24 GB now holds tens of thousands of scenarios where
// it held 3 316. `choose_reduce_block` then spreads them over every SM instead
// of leaving 56 of 82 idle. See the comment on that function.
__global__ void population_reduce_kernel(DeviceDataset dataset,
                                         DeviceGenes genes,
                                         DeviceScenarios scenarios,
                                         NeoPopulationSettings settings,
                                         const unsigned char* gap_flags,
                                         NeoPopulationOutcome* outcomes,
                                         double* monthly_pnls,
                                         double* month_start_equities,
                                         NeoPopulationMetricRow* rows,
                                         unsigned long long* accepted_trade_total) {
  // ONE THREAD PER SCENARIO, not per gene.
  //
  // Every workspace slice below is indexed by `scenario`, and the gene arrays by
  // `candidate` — the two are equal only when the caller uploaded the identity
  // descriptor array, which is what makes that case bit-identical to the
  // pre-scenario engine. A mixed array (100 Monte-Carlo runs and one cost
  // variant of the same gene) shares one `candidate` across 101 scenarios, and
  // each of those must get its own trade slice, its own monthly buckets and its
  // own metric row.
  const int scenario = blockIdx.x * blockDim.x + threadIdx.x;
  if (scenario >= scenarios.count) {
    return;
  }
  const int candidate = static_cast<int>(scenarios.base_candidate_ids[scenario]);
  const unsigned scenario_type = scenarios.types[scenario];
  const unsigned long long scenario_id = scenarios.ids[scenario];
  const bool diagnostics_enabled = outcomes != nullptr;

  // Perturbation is a property of the SCENARIO. `kScenarioPerturb` is the only
  // type that reads `rng_counters`, and no existing caller emits it: the shipped
  // Monte-Carlo screen perturbs on the host with ChaCha8 and uploads the clones.
  // See the note on `perturb_factor` for why the two lanes draw the same
  // distribution and not the same numbers.
  const int perturbed = (scenario_type == kScenarioPerturb) ? 1 : 0;
  const unsigned long long rng_counter = scenarios.rng_counters[scenario];

  // ── Window ────────────────────────────────────────────────────────────────
  //
  // `window_offset + 1` reproduces the old `bar = 1` exactly when the offset is
  // zero, which it is for every caller today: the signal is taken from the close
  // of the previous bar, so the first bar that can OPEN a position is one past
  // the first bar of the window. A zero `window_len` means "to the end", so a
  // caller that leaves the field alone still walks the whole series.
  const int window_offset = static_cast<int>(scenarios.window_offsets[scenario]);
  const unsigned int requested_len = scenarios.window_lens[scenario];
  const int window_len = (requested_len == 0u) ? (dataset.bars - window_offset)
                                               : static_cast<int>(requested_len);
  const int walk_begin = window_offset + 1;
  int walk_end = window_offset + window_len;
  if (walk_end > dataset.bars) {
    walk_end = dataset.bars;
  }

  // ── Cost overrides ────────────────────────────────────────────────────────
  //
  // Resolved once per thread. `kNoTickOverride` / `kNoMicroOverride` leave the
  // settings' own values in place, and that is the path every existing caller
  // takes — the branch below is uniform across the warp and adds no arithmetic
  // to the walk. Zero is a REAL override ("charge no spread"), which is why the
  // sentinel is -1 and not 0.
  const int spread_ticks = scenarios.spread_ticks[scenario];
  const int has_spread_override = (spread_ticks != kNoTickOverride) ? 1 : 0;
  const double spread_override_pips =
      has_spread_override ? (static_cast<double>(spread_ticks) / kTicksPerPip) : 0.0;
  const long long commission_micros = scenarios.commission_micros[scenario];
  const double commission_per_trade =
      (commission_micros != kNoMicroOverride)
          ? (static_cast<double>(commission_micros) / kMicrosPerUnit)
          : settings.commission_per_trade;
  // Slippage has no counterpart in the CPU engine, so 0 must cost nothing at
  // all — not `+ 0.0`, which is a real add. Every use below is inside
  // `if (has_slippage)`.
  const int slippage_ticks = scenarios.slippage_ticks[scenario];
  const int has_slippage = (slippage_ticks != 0) ? 1 : 0;

  // The bar-independent half of signal synthesis: CSR window, thresholds,
  // threshold gap, SMC active sum and gate — plus, for a perturbed scenario,
  // the perturbed thresholds the gap is then taken from. Computed once here
  // instead of once per bar, which is loop-invariant hoisting and not a
  // reassociation — the same additions, in the same order, fewer times.
  const SignalPlan signal_plan = build_signal_plan(genes, candidate, perturbed, rng_counter);

  const int month_capacity = static_cast<int>(settings.month_capacity);
  double* monthly = monthly_pnls + static_cast<long long>(scenario) * month_capacity;
  double* month_start = month_start_equities + static_cast<long long>(scenario) * month_capacity;
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
  // Overrunning the slice drops trades from the DIAGNOSTIC outcome array rather
  // than corrupting a neighbour's. Equity, drawdown and the trade count are
  // unaffected — the walk keeps simulating, it just stops recording — so no
  // metric silently changes.
  //
  // This said the overrun "is reported through the diagnostics". It was not:
  // the flag that would have carried it was allocated, memset and read by
  // nothing. What actually reports it is metric slot 8 against
  // `kMaxTradesPerCandidate`, which the host already logs every launch as
  // `peak_fill_pct` — a real measurement rather than an allocated intention.
  const unsigned long long range_start =
      static_cast<unsigned long long>(scenario) * kMaxTradesPerCandidate;
  const unsigned long long range_end = range_start + kMaxTradesPerCandidate;

  double equity = initial_equity;
  double peak_equity = initial_equity;
  double max_drawdown = 0.0;
  long long trade_count = 0;
  long long wins = 0;
  double gross_profit = 0.0;
  double gross_loss = 0.0;
  unsigned long long accepted_trades = 0ull;

  std::int64_t last_month = -1;
  double current_month_pnl = 0.0;
  double current_month_start_equity = initial_equity;
  std::int64_t month_ptr = -1;

  std::int64_t last_day = -1;
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
  // Resolved per entry bar below, not hoisted: the value depends on the bar's
  // hour once a session profile is in play.
  unsigned long long cursor = range_start;

  const double slippage_price =
      has_slippage ? (static_cast<double>(slippage_ticks) / kTicksPerPip * pip) : 0.0;

  for (int bar = walk_begin; bar < walk_end; ++bar) {
    // Per-bar spread, resolved where the CPU resolves it — at the top of the
    // bar loop, from this bar's timestamp. An entry therefore pays the spread
    // of the bar it opens on and an exit pays the spread of the bar it closes
    // on, which is what `fast_evaluate_strategy_core` does and what a broker
    // does. With no session profile all three buckets hold `spread_pips`, so
    // both values equal the old hoisted scalar exactly.
    //
    // A cost scenario replaces the whole per-bar resolution with its own flat
    // number. That is what the sensitivity screen means by "wider spread": the
    // CPU path it replaces set `settings.spread_pips` and left the profile
    // unset, which resolves to that scalar at every hour.
    const double bar_spread_pips =
        has_spread_override ? spread_override_pips
                            : spread_pips_for_bar(settings, population_timestamp_at(dataset, bar));
    const double half_spread_price = bar_spread_pips * 0.5 * pip;
    const double half_spread_cost = bar_spread_pips * 0.5 * settings.pip_value_per_lot;

    const int parent_bar = population_parent_row(dataset, bar);
    const std::int64_t month = dataset.months[parent_bar];
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

    const std::int64_t day = dataset.days[parent_bar];
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
        double exit_price = dataset.close[parent_bar];
        // Adverse by construction: a long is filled lower than it asked, a
        // short higher. Guarded rather than added-with-zero so a scenario that
        // wants no slippage — which is every scenario the CPU engine can
        // mirror — executes the identical instruction sequence.
        if (has_slippage) {
          exit_price -= static_cast<double>(position_event.direction) * slippage_price;
        }
        double price_pnl = 0.0;
        if (position_event.direction == kDirectionLong) {
          price_pnl = (exit_price - position_entry_price) / pip * settings.pip_value_per_lot;
        } else {
          price_pnl = (position_entry_price - exit_price) / pip * settings.pip_value_per_lot;
        }
        const double gross_scaled =
            price_pnl * position_lots -
            (commission_per_trade + half_spread_cost) * position_lots;
        const std::int64_t entry_timestamp =
            population_timestamp_at(dataset, position_event.entry_bar);
        const std::int64_t exit_timestamp = population_timestamp_at(dataset, bar);
        const double pnl = apply_carry_and_conversion(gross_scaled, position_lots,
                                                      position_event.direction, entry_timestamp,
                                                      exit_timestamp, settings);
        equity += pnl;
        // The per-trade record is completed here because this is the only place
        // that knows position size, carry and the conversion fee. R-multiple
        // mirrors eval.rs exactly — realised P&L over the entry stop distance,
        // guarded against a zero denominator — so it stays comparable with the
        // CPU trade list rather than merely plausible.
        NeoPopulationOutcome* diagnostic_outcome =
            diagnostic_outcome_slot_v1(outcomes, position_index, range_end);
        if (diagnostic_outcome != nullptr) {
          diagnostic_outcome->exit_bar = bar;
          diagnostic_outcome->exit_reason = kExitGap;
          diagnostic_outcome->entry_bar = position_event.entry_bar;
          diagnostic_outcome->exit_price = exit_price;
          diagnostic_outcome->mfe =
              position_fav > 0.0 ? position_fav / pip * settings.pip_value_per_lot : 0.0;
          diagnostic_outcome->mae =
              position_adv > 0.0 ? position_adv / pip * settings.pip_value_per_lot : 0.0;
          diagnostic_outcome->pnl = pnl;
          diagnostic_outcome->r_multiple =
              pnl / fmax(position_stop_pips * settings.pip_value_per_lot, 1.0e-9);
        }
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
        const double low = dataset.low[parent_bar];
        const double high = dataset.high[parent_bar];
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
          exit_price_now = dataset.close[parent_bar];
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
          double exit_price = exit_price_now;
          if (has_slippage) {
            exit_price -= static_cast<double>(position_event.direction) * slippage_price;
          }
          {
            double price_pnl = 0.0;
            if (position_event.direction == kDirectionLong) {
              price_pnl = (exit_price - position_entry_price) / pip * settings.pip_value_per_lot;
            } else {
              price_pnl = (position_entry_price - exit_price) / pip * settings.pip_value_per_lot;
            }
            const double gross_scaled =
                price_pnl * position_lots -
                (commission_per_trade + half_spread_cost) * position_lots;
            const std::int64_t entry_timestamp =
                population_timestamp_at(dataset, position_event.entry_bar);
            const std::int64_t exit_timestamp = population_timestamp_at(dataset, bar);
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
            NeoPopulationOutcome* diagnostic_outcome =
                diagnostic_outcome_slot_v1(outcomes, position_index, range_end);
            if (diagnostic_outcome != nullptr) {
              diagnostic_outcome->exit_bar = bar;
              diagnostic_outcome->exit_reason = exit_reason_now;
              diagnostic_outcome->entry_bar = position_event.entry_bar;
              diagnostic_outcome->exit_price = exit_price;
              diagnostic_outcome->mfe =
                  position_fav > 0.0 ? position_fav / pip * settings.pip_value_per_lot : 0.0;
              diagnostic_outcome->mae =
                  position_adv > 0.0 ? position_adv / pip * settings.pip_value_per_lot : 0.0;
              diagnostic_outcome->pnl = pnl;
              diagnostic_outcome->r_multiple =
                  pnl / fmax(position_stop_pips * settings.pip_value_per_lot, 1.0e-9);
            }
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

    // ── Entry, from a signal synthesised HERE ─────────────────────────────
    //
    // This read `signal_values[signal_base + signal_bar]` — a 2.78 GB device
    // array whose only purpose was to bring this one byte from another kernel,
    // with `signal_confidences` (11.1 GB) alongside it for the sizing call
    // below. Both are gone; the value is computed in registers at the moment it
    // is used, from the same CSR terms in the same ascending f64 order.
    //
    // Computed HERE, after the exit block, rather than at the top of the loop:
    // synthesis is a pure function of (candidate, bar) with no side effects, so
    // evaluating it only on the bars that can actually open a position is
    // bit-identical to evaluating it on every bar — and a bar spent holding a
    // position now costs nothing. The old kernel had no choice; it had to fill
    // the whole array.
    //
    // `signal_bar` is `bar - 1` exactly as before: the decision is taken on the
    // close of the previous bar and filled on this one. Bar `bars - 1` is
    // therefore never synthesised, which is correct — an entry there would have
    // no bar to fill on. The old kernel computed it and nothing read it.
    const int signal_bar = bar - 1;
    double signal_confidence_here = 0.0;
    const signed char signal_here =
        synthesize_signal(dataset, genes, signal_plan, signal_bar, &signal_confidence_here);
    if (signal_here != 0) {
      if (diagnostics_enabled) {
        ++cursor;
      }
      if (settings.max_trades_per_day > 0u && day_trade_count >= settings.max_trades_per_day) {
        continue;
      }
      const int direction = signal_here > 0 ? kDirectionLong : kDirectionShort;
      double entry_price =
          dataset.close[parent_bar] + static_cast<double>(direction) * half_spread_price;
      if (has_slippage) {
        entry_price += static_cast<double>(direction) * slippage_price;
      }
      double entry_stop_pips = 0.0;
      double entry_target_pips = 0.0;
      entry_stop_target_pips(dataset, genes, settings, signal_plan, candidate, signal_bar,
                             &entry_stop_pips, &entry_target_pips);
      NeoPopulationEvent event;
      event.candidate_id = static_cast<unsigned long long>(candidate);
      event.scenario_id = scenario_id;
      event.entry_bar = static_cast<unsigned int>(bar);
      // The last bar of THIS scenario's window, which for a full window is
      // `bars - 1` exactly as before.
      event.last_bar = static_cast<unsigned int>(walk_end - 1);
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
        lots = risk_based_position_lots(signal_confidence_here, equity, stop_pips, settings);
      }
      position_event = event;
      position_entry_price = entry_price;
      position_lots = lots;
      if (diagnostics_enabled) {
        // `cursor` already advanced past this trade's slot.
        position_index = cursor - 1ull;
        if (position_index >= range_end) {
          // Out of slots. Keep simulating so equity and drawdown stay honest —
          // the trade still happened — but do not write past this candidate's
          // slice into the next one's.
          position_index = range_end - 1ull;
        }
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
      if (accepted_trade_total != nullptr) {
        accepted_trades += 1ull;
      }
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

  bool monthly_return_inputs_valid = true;
  double monthly_return_mean = 0.0;
  double monthly_return_std = 0.0;
  if (limit >= 0) {
    const long long count = limit + 1;
    double monthly_return_sum = 0.0;
    for (long long index = 0; index <= limit; ++index) {
      const double start_equity = month_start[index];
      if (!isfinite(monthly[index]) || !isfinite(start_equity) || start_equity <= 0.0) {
        monthly_return_inputs_valid = false;
      } else {
        const double period_return = monthly[index] / start_equity;
        if (!isfinite(period_return)) {
          monthly_return_inputs_valid = false;
        } else {
          monthly_return_sum += period_return;
          if (!isfinite(monthly_return_sum)) {
            monthly_return_inputs_valid = false;
          }
        }
      }
    }
    if (monthly_return_inputs_valid && count >= 2) {
      monthly_return_mean = monthly_return_sum / static_cast<double>(count);
      if (!isfinite(monthly_return_mean)) {
        monthly_return_inputs_valid = false;
      }
    }
    if (monthly_return_inputs_valid && count >= 2) {
      double monthly_return_variance = 0.0;
      for (long long index = 0; index <= limit; ++index) {
        const double start_equity = month_start[index];
        const double period_return = monthly[index] / start_equity;
        const double delta = period_return - monthly_return_mean;
        const double squared_delta = delta * delta;
        if (!isfinite(squared_delta)) {
          monthly_return_inputs_valid = false;
        } else {
          monthly_return_variance += squared_delta;
          if (!isfinite(monthly_return_variance)) {
            monthly_return_inputs_valid = false;
          }
        }
      }
      if (monthly_return_inputs_valid) {
        monthly_return_std =
            sqrt(fmax(monthly_return_variance / static_cast<double>(count - 1), 0.0));
        if (!isfinite(monthly_return_std)) {
          monthly_return_inputs_valid = false;
        }
      }
    }
  }

  double sharpe = monthly_return_inputs_valid
                       ? (monthly_return_std > 0.0
                              ? (monthly_return_mean / monthly_return_std) * 3.4641
                              : 0.0)
                       : invalid_monthly_return_sharpe_v1();
  if (!isfinite(sharpe) && sharpe != invalid_monthly_return_sharpe_v1()) {
    sharpe = invalid_monthly_return_sharpe_v1();
  }
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
  // The gene's identity and the scenario's, separately. A mixed array shares one
  // `candidate_id` across every scenario of the same gene, and the host demuxes
  // by `scenario_id` — which is why both are carried and neither is a position.
  row.candidate_id = genes.candidate_ids[candidate];
  row.scenario_id = scenario_id;
  row.values[0] = sanitize(net_profit);
  row.values[1] = sharpe;
  row.values[2] = sanitize(peak_equity);
  row.values[3] = sanitize(max_drawdown);
  row.values[4] = sanitize(win_rate);
  row.values[5] = sanitize(profit_factor);
  row.values[6] = sanitize(expectancy);
  row.values[7] = sanitize(monthly_target_hit_rate);
  row.values[8] = static_cast<double>(trade_count);
  row.values[9] = sanitize(consistency);
  row.values[10] = sanitize(max_daily_drawdown);
  rows[scenario] = row;

  if (accepted_trade_total != nullptr) {
    atomicAdd(reinterpret_cast<unsigned long long*>(accepted_trade_total), accepted_trades);
  }
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
  if (count > SIZE_MAX / sizeof(T)) {
    *pointer = nullptr;
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
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

enum class PopulationWorkspaceModeV1 : std::uint32_t {
  Uninitialized = 0u,
  CompatibilityDeviceParityOnly = 1u,
  StrictMetricsOnly = 2u,
};

enum class PopulationStrictExecutionStateV1 : std::uint32_t {
  StrictIdle = 0u,
  InFlight = 1u,
  Poisoned = 2u,
};

struct NeoCudaPopulationSession {
  int device = 0;
  cudaStream_t stream = nullptr;
  std::uint32_t stream_ownership = NEO_POPULATION_STREAM_OWNED;
  std::uint32_t parent_ownership = NEO_POPULATION_PARENT_OWNED_V1;
  cudaEvent_t event = nullptr;
  // No `max_events`. The session used to carry the event-buffer capacity the
  // creator asked for, but the buffer it sized has not been allocated since the
  // reduce started opening positions from the signal directly. The create ABI
  // still takes the argument — see the note there — and the device no longer
  // stores it, because a field nothing reads is how a phantom budget survives.
  unsigned long long next_event_id = 1ull;
  unsigned long long pending_event_id = 0ull;
  bool has_parent_v1 = false;
  bool has_dataset = false;
  bool has_genes = false;
  bool has_scenarios = false;
  bool metrics_ready = false;
  int bars = 0;
  int parent_rows = 0;
  int feature_count = 0;
  int view_kind = static_cast<int>(NEO_POPULATION_VIEW_FULL);
  int view_start = 0;
  int timestamp_mode = static_cast<int>(NEO_POPULATION_TIMESTAMP_CANONICAL);
  int population = 0;
  /// Threads the walk launches, and the extent of every workspace array.
  ///
  /// This used to be implicitly equal to `population` — one scenario per gene,
  /// enforced at upload. It no longer is: a quality screen uploads 174 genes and
  /// 17 574 scenarios. `population` still sizes the GENE arrays and still bounds
  /// `base_candidate_id`; this sizes everything the walk writes.
  int scenario_count = 0;
  int month_capacity = 0;
  /// SM count of `device`, read once at create and used to choose the reduce
  /// block size. Querying it per launch would be a device call on the critical
  /// path for a number that cannot change.
  int sm_count = 0;
  // What the workspace was actually built for.
  //
  // The outcome array is sized `scenario_count * kMaxTradesPerCandidate` at
  // allocation, and every kernel indexes it by the CURRENT scenario count,
  // which `upload_scenarios` overwrites on each call. The reuse test compared
  // only `metric_rows == nullptr` and `month_capacity`, so a session built for
  // a small workload and reused for a large one wrote past the end of it —
  // into `monthly_pnls` and `month_start_equities`, which are the arrays
  // sharpe and consistency are computed from, and `sanitize()` then turns any
  // non-finite consequence into 0.0.
  int workspace_scenarios = 0;
  int workspace_bars = 0;
  PopulationWorkspaceModeV1 workspace_mode = PopulationWorkspaceModeV1::Uninitialized;
  PopulationStrictExecutionStateV1 strict_execution_state =
      PopulationStrictExecutionStateV1::StrictIdle;
  // How many outcome records `read_diagnostics` may copy back.
  //
  // Named for an event stream that no longer exists. It is NOT a count of
  // emitted events — nothing emits events — it is `population *
  // kMaxTradesPerCandidate`, the extent of the outcome array the reduce writes
  // into. The name survives because it is the ABI's `NeoPopulationCounters::
  // event_count` and the diagnostic readback's bound, both of which the bench
  // and the prototype engines still consume.
  unsigned long long emitted_events = 0ull;
  unsigned long long accepted_trades = 0ull;
  std::uint64_t dataset_upload_bytes = 0ull;
  std::uint64_t gene_upload_bytes = 0ull;
  std::uint64_t scenario_upload_bytes = 0ull;
  std::uint64_t kernel_submissions = 0ull;
  std::uint64_t synchronization_events = 0ull;
  NeoPopulationResidencyCountersV1 residency_counters{};
  NeoPopulationDeviceIdentityV1 device_identity{};

  double* close = nullptr;
  double* high = nullptr;
  double* low = nullptr;
  // BAR-MAJOR, `bar * feature_count + feature`. The host still hands over a
  // feature-major matrix; `upload_dataset` transposes it once on the device and
  // frees the feature-major staging copy, so nothing feature-major stays
  // resident. Renamed from `indicators` deliberately: a reader that was not
  // updated must fail to compile rather than silently stride the wrong way.
  double* indicators_bar_major = nullptr;
  // Packed two-u4 validity codes are retained with the sealed V3 parent. The
  // current population kernel reads values only, but the pointer and exact
  // extent remain bound so later strict validation cannot detach validity from
  // the feature matrix it describes.
  unsigned char* indicators_validity_u4 = nullptr;
  std::size_t indicators_validity_u4_bytes = 0;
  // Feature-major immutable parent for the V1 route. It is consumed directly
  // through `population_feature_at`; no staging transpose or second copy.
  double* indicators_feature_major = nullptr;
  std::int64_t* months = nullptr;
  std::int64_t* days = nullptr;
  std::int64_t* timestamps = nullptr;
  signed char* smc_rows = nullptr;
  unsigned long long* view_indices = nullptr;
  std::size_t view_indices_capacity = 0;
  double* adaptive_base_pips = nullptr;
  std::size_t adaptive_base_pips_capacity = 0;
  int has_adaptive_base = 0;
  unsigned char* gap_flags = nullptr;

  unsigned long long* candidate_ids = nullptr;
  int* gene_offsets = nullptr;
  int* gene_indices = nullptr;
  double* gene_weights = nullptr;
  double* long_thresholds = nullptr;
  double* short_thresholds = nullptr;
  double* stop_pips = nullptr;
  double* target_pips = nullptr;
  double* stop_vol_multipliers = nullptr;
  signed char* smc_flags = nullptr;
  double* smc_weights = nullptr;
  double gate_threshold = 0.0;
  int smc_gate_disabled = 0;

  // The work list, unpacked from `NeoScenarioDescriptor` into struct-of-arrays
  // at upload. Was a single `scenario_ids` array, because `scenario_id` was the
  // only field the device read.
  unsigned long long* scenario_base_candidate_ids = nullptr;
  unsigned long long* scenario_ids = nullptr;
  unsigned long long* scenario_rng_counters = nullptr;
  unsigned long long* scenario_window_offsets = nullptr;
  unsigned int* scenario_window_lens = nullptr;
  unsigned int* scenario_types = nullptr;
  int* scenario_spread_ticks = nullptr;
  int* scenario_slippage_ticks = nullptr;
  long long* scenario_commission_micros = nullptr;

  // No `signal_values`, no `signal_confidences`, no `event_counts`, no
  // `event_offsets`, no `events`.
  //
  // The first two were a `population * bars` handoff between two kernels and
  // are now computed in registers inside the walk (`synthesize_signal`). At
  // 843 456 bars they were 2.78 GB and 11.1 GB at 3 300 candidates — 87.7 % of
  // everything the host budgeted per scenario, and the entire reason a 24 GB
  // card resolved to 3 316 scenarios and the caller grew a six-chunk loop.
  //
  // `event_counts` and `event_offsets` belonged to the count/scan/emit chain,
  // which had no launch site; `event_offsets` was still being passed to the
  // reduce as an unread parameter, pointing at memory nothing had ever written.
  // `events` had a declaration and a free and no allocation at all.
  //
  // A buffer that only appears in a struct, an allocation list and a free is
  // indistinguishable from a real one to anyone sizing memory against it. That
  // is exactly how the host came to reserve VRAM for a phantom.
  NeoPopulationOutcome* outcomes = nullptr;
  double* monthly_pnls = nullptr;
  double* month_start_equities = nullptr;
  NeoPopulationMetricRow* metric_rows = nullptr;
  unsigned long long* accepted_trade_total = nullptr;

  void release_scenarios() {
    device_free(scenario_base_candidate_ids);
    device_free(scenario_ids);
    device_free(scenario_rng_counters);
    device_free(scenario_window_offsets);
    device_free(scenario_window_lens);
    device_free(scenario_types);
    device_free(scenario_spread_ticks);
    device_free(scenario_slippage_ticks);
    device_free(scenario_commission_micros);
  }

  void release_workspace() {
    device_free(outcomes);
    device_free(monthly_pnls);
    device_free(month_start_equities);
    device_free(metric_rows);
    device_free(accepted_trade_total);
    // The EXTENTS die with the memory they describe.
    //
    // Leaving them set is safe today only because `metric_rows == nullptr` is
    // the FIRST term of the re-allocation predicate, so the stale numbers are
    // never consulted. Reorder that predicate — or free `metric_rows` from any
    // other place — and a session claims a workspace it does not own, which is
    // an out-of-bounds write into freed device memory rather than an error.
    workspace_scenarios = 0;
    workspace_bars = 0;
    month_capacity = 0;
    // Deliberately retain `workspace_mode`. A run-owned session may grow or be
    // destroyed, but it may never relabel compatibility allocations as strict
    // resident authority (or vice versa).
  }

  void release() {
    release_workspace();
    if (parent_ownership == NEO_POPULATION_PARENT_OWNED_V1) {
      device_free(close);
      device_free(high);
      device_free(low);
      device_free(indicators_bar_major);
      device_free(indicators_feature_major);
      device_free(months);
      device_free(days);
      device_free(timestamps);
      device_free(smc_rows);
    } else {
      close = nullptr;
      high = nullptr;
      low = nullptr;
      indicators_bar_major = nullptr;
      indicators_feature_major = nullptr;
      months = nullptr;
      days = nullptr;
      timestamps = nullptr;
      smc_rows = nullptr;
    }
    indicators_validity_u4 = nullptr;
    indicators_validity_u4_bytes = 0;
    device_free(view_indices);
    view_indices_capacity = 0;
    device_free(adaptive_base_pips);
    adaptive_base_pips_capacity = 0;
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
    release_scenarios();
    if (event != nullptr) {
      cudaEventDestroy(event);
      event = nullptr;
    }
    if (stream != nullptr && stream_ownership == NEO_POPULATION_STREAM_OWNED) {
      cudaStreamDestroy(stream);
    }
    stream = nullptr;
  }
};

namespace {

bool strict_population_work_blocks_host_boundary_v1(const NeoCudaPopulationSession* session) {
  return session != nullptr &&
         session->strict_execution_state != PopulationStrictExecutionStateV1::StrictIdle;
}

std::int32_t strict_population_host_boundary_status_v1(
    const NeoCudaPopulationSession* session) {
  return session->strict_execution_state == PopulationStrictExecutionStateV1::Poisoned
             ? NEO_POPULATION_STATUS_STRICT_RESIDENT_POISONED
             : NEO_POPULATION_STATUS_STRICT_RESIDENT_IN_FLIGHT;
}

struct GeneHostStagingV1 {
  unsigned long long* candidate_ids = nullptr;
  int* offsets = nullptr;
  int* indices = nullptr;
  double* weights = nullptr;
  double* long_thresholds = nullptr;
  double* short_thresholds = nullptr;
  double* stop_pips = nullptr;
  double* target_pips = nullptr;
  double* stop_vol_multipliers = nullptr;
  signed char* smc_flags = nullptr;
  double* smc_weights = nullptr;

  ~GeneHostStagingV1() {
    delete[] candidate_ids;
    delete[] offsets;
    delete[] indices;
    delete[] weights;
    delete[] long_thresholds;
    delete[] short_thresholds;
    delete[] stop_pips;
    delete[] target_pips;
    delete[] stop_vol_multipliers;
    delete[] smc_flags;
    delete[] smc_weights;
  }
};

struct ScenarioHostStagingV1 {
  unsigned long long* base_ids = nullptr;
  unsigned long long* ids = nullptr;
  unsigned long long* counters = nullptr;
  unsigned long long* offsets = nullptr;
  unsigned int* lens = nullptr;
  unsigned int* types = nullptr;
  int* spreads = nullptr;
  int* slippages = nullptr;
  long long* commissions = nullptr;

  ~ScenarioHostStagingV1() {
    delete[] base_ids;
    delete[] ids;
    delete[] counters;
    delete[] offsets;
    delete[] lens;
    delete[] types;
    delete[] spreads;
    delete[] slippages;
    delete[] commissions;
  }
};

void CUDART_CB release_gene_host_staging_v1(void* opaque) {
  delete static_cast<GeneHostStagingV1*>(opaque);
}

void CUDART_CB release_scenario_host_staging_v1(void* opaque) {
  delete static_cast<ScenarioHostStagingV1*>(opaque);
}

template <typename T>
std::int32_t release_staging_after_stream_v1(cudaStream_t stream,
                                             T* staging,
                                             cudaHostFn_t release) {
  if (cudaLaunchHostFunc(stream, release, staging) == cudaSuccess) {
    return NEO_POPULATION_STATUS_OK;
  }
  // The enqueue failed after earlier copies may have been accepted. This is an
  // error-only teardown barrier: never free a host source while DMA can still
  // read it, and never count the failed path as resident success evidence.
  cudaStreamSynchronize(stream);
  delete staging;
  return NEO_POPULATION_STATUS_TRANSFER_FAILED;
}

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

bool checked_add_u64(std::uint64_t& total, std::uint64_t value) {
  if (value > UINT64_MAX - total) {
    return false;
  }
  total += value;
  return true;
}

bool hash_is_nonzero_v3(const std::uint8_t* hash, std::size_t bytes) {
  if (hash == nullptr || bytes == 0) {
    return false;
  }
  for (std::size_t index = 0; index < bytes; ++index) {
    if (hash[index] != 0u) {
      return true;
    }
  }
  return false;
}

template <typename T>
std::int32_t ensure_device_capacity_v3(T** pointer,
                                       std::size_t* capacity,
                                       std::size_t required) {
  if (required <= *capacity) {
    return NEO_POPULATION_STATUS_OK;
  }
  device_free(*pointer);
  *capacity = 0;
  const auto status = device_alloc(pointer, required);
  if (status == NEO_POPULATION_STATUS_OK) {
    *capacity = required;
  }
  return status;
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
  // `max_events` is vestigial and the device ignores it.
  //
  // It sized an event buffer that is no longer allocated. The parameter stays
  // in the C ABI — removing it would break every caller and the version bump
  // buys nothing — and the non-zero check stays with it so a caller passing 0
  // still gets the same refusal it always did. What changed is that the value
  // is not stored: nothing downstream may consult it and mistake it for a
  // budget that is being enforced.
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

  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
      properties.major < 0 || properties.minor < 0 || properties.multiProcessorCount <= 0 ||
      properties.totalGlobalMem == 0) {
    return fail(NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE);
  }

  auto* session = new (std::nothrow) NeoCudaPopulationSession();
  if (session == nullptr) {
    return fail(NEO_POPULATION_STATUS_ALLOCATION_FAILED);
  }
  session->device = device;
  session->sm_count = properties.multiProcessorCount;
  session->device_identity.selected_device_ordinal = static_cast<std::uint32_t>(device);
  session->device_identity.compute_capability_major =
      static_cast<std::uint32_t>(properties.major);
  session->device_identity.compute_capability_minor =
      static_cast<std::uint32_t>(properties.minor);
  session->device_identity.multiprocessor_count =
      static_cast<std::uint32_t>(properties.multiProcessorCount);
  session->device_identity.total_global_memory_bytes =
      static_cast<std::uint64_t>(properties.totalGlobalMem);
  session->device_identity.pci_domain_id = properties.pciDomainID;
  session->device_identity.pci_bus_id = properties.pciBusID;
  session->device_identity.pci_device_id = properties.pciDeviceID;
#if CUDART_VERSION >= 10000
  std::memcpy(session->device_identity.uuid,
              properties.uuid.bytes,
              sizeof(session->device_identity.uuid));
#endif
  std::memcpy(session->device_identity.name,
              properties.name,
              sizeof(session->device_identity.name));
  if (cudaStreamCreateWithFlags(&session->stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreateWithFlags(&session->event, cudaEventDisableTiming) != cudaSuccess) {
    session->release();
    delete session;
    return fail(NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE);
  }
  session->residency_counters.stream_creation_count = 1ull;
  if (status != nullptr) {
    *status = NEO_POPULATION_STATUS_OK;
  }
  return session;
}

extern "C" NeoCudaPopulationSession*
neoethos_gpu_cuda_population_bind_resident_feature_store_v3(
    const NeoPopulationResidentFeatureStoreV3* resident,
    std::int32_t* status) {
  const auto fail = [&](std::int32_t code) -> NeoCudaPopulationSession* {
    if (status != nullptr) {
      *status = code;
    }
    return nullptr;
  };
  if (resident == nullptr || resident->abi_version != NEOETHOS_GPU_ABI_VERSION) {
    return fail(resident == nullptr ? NEO_POPULATION_STATUS_INVALID_ARGUMENT
                                    : NEO_POPULATION_STATUS_ABI_MISMATCH);
  }
  if (resident->reserved != 0u || resident->row_count == 0ull ||
      resident->feature_count == 0u || resident->smc_slots != kSmcSlots ||
      resident->row_count > static_cast<std::uint64_t>(INT_MAX) ||
      resident->feature_count > static_cast<std::uint32_t>(INT_MAX) ||
      resident->row_count > static_cast<std::uint64_t>(SIZE_MAX) /
                                static_cast<std::uint64_t>(resident->feature_count) ||
      resident->row_count * static_cast<std::uint64_t>(resident->feature_count) >
          static_cast<std::uint64_t>(SIZE_MAX) ||
      resident->close == nullptr || resident->high == nullptr || resident->low == nullptr ||
      resident->indicators_bar_major == nullptr ||
      resident->indicators_validity_u4 == nullptr || resident->months == nullptr ||
      resident->days == nullptr || resident->timestamps == nullptr ||
      resident->smc_rows == nullptr || resident->admitted_primary_context == nullptr ||
      resident->admitted_run_stream == nullptr || resident->ready_event == nullptr ||
      !hash_is_nonzero_v3(resident->admission_identity_sha256,
                          sizeof(resident->admission_identity_sha256)) ||
      !hash_is_nonzero_v3(resident->canonical_content_merkle,
                          sizeof(resident->canonical_content_merkle)) ||
      !hash_is_nonzero_v3(resident->device_uuid, sizeof(resident->device_uuid))) {
    return fail(NEO_POPULATION_STATUS_INVALID_ARGUMENT);
  }
  const std::uint64_t cells =
      resident->row_count * static_cast<std::uint64_t>(resident->feature_count);
  const std::uint64_t logical_validity_bytes = cells / 2ull + cells % 2ull;
  if (logical_validity_bytes > UINT64_MAX - 3ull ||
      resident->packed_validity_bytes != (logical_validity_bytes + 3ull) / 4ull * 4ull) {
    return fail(NEO_POPULATION_STATUS_INVALID_ARGUMENT);
  }

  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device < 0 ||
      static_cast<std::uint32_t>(current_device) != resident->selected_device_ordinal) {
    return fail(NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE);
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, current_device) != cudaSuccess ||
      properties.multiProcessorCount <= 0 || properties.totalGlobalMem == 0 ||
      properties.major != static_cast<int>(resident->compute_capability_major) ||
      properties.minor != static_cast<int>(resident->compute_capability_minor)) {
    return fail(NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE);
  }
#if CUDART_VERSION >= 10000
  if (std::memcmp(properties.uuid.bytes, resident->device_uuid, sizeof(resident->device_uuid)) !=
      0) {
    return fail(NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE);
  }
#else
  return fail(NEO_POPULATION_STATUS_UNSUPPORTED);
#endif

  auto* session = new (std::nothrow) NeoCudaPopulationSession();
  if (session == nullptr) {
    return fail(NEO_POPULATION_STATUS_ALLOCATION_FAILED);
  }
  session->device = current_device;
  session->stream = resident->admitted_run_stream;
  session->stream_ownership = NEO_POPULATION_STREAM_BORROWED;
  session->parent_ownership = NEO_POPULATION_PARENT_BORROWED_RESIDENT_V3;
  const auto fail_session = [&](std::int32_t code) -> NeoCudaPopulationSession* {
    session->release();
    delete session;
    return fail(code);
  };
  if (cudaEventCreateWithFlags(&session->event, cudaEventDisableTiming) != cudaSuccess) {
    return fail_session(NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE);
  }
  const auto parent_rows = static_cast<std::size_t>(resident->row_count);
  const auto gap_status = device_alloc(&session->gap_flags, parent_rows);
  if (gap_status != NEO_POPULATION_STATUS_OK) {
    return fail_session(gap_status);
  }

  session->close = const_cast<double*>(resident->close);
  session->high = const_cast<double*>(resident->high);
  session->low = const_cast<double*>(resident->low);
  session->indicators_bar_major =
      const_cast<double*>(resident->indicators_bar_major);
  session->indicators_validity_u4 =
      const_cast<unsigned char*>(resident->indicators_validity_u4);
  session->indicators_validity_u4_bytes =
      static_cast<std::size_t>(resident->packed_validity_bytes);
  session->months = const_cast<std::int64_t*>(resident->months);
  session->days = const_cast<std::int64_t*>(resident->days);
  session->timestamps = const_cast<std::int64_t*>(resident->timestamps);
  session->smc_rows = const_cast<signed char*>(resident->smc_rows);
  session->parent_rows = static_cast<int>(resident->row_count);
  session->feature_count = static_cast<int>(resident->feature_count);
  session->sm_count = properties.multiProcessorCount;
  session->has_parent_v1 = true;
  session->device_identity.selected_device_ordinal = resident->selected_device_ordinal;
  session->device_identity.compute_capability_major = resident->compute_capability_major;
  session->device_identity.compute_capability_minor = resident->compute_capability_minor;
  session->device_identity.multiprocessor_count =
      static_cast<std::uint32_t>(properties.multiProcessorCount);
  session->device_identity.total_global_memory_bytes =
      static_cast<std::uint64_t>(properties.totalGlobalMem);
  session->device_identity.pci_domain_id = properties.pciDomainID;
  session->device_identity.pci_bus_id = properties.pciBusID;
  session->device_identity.pci_device_id = properties.pciDeviceID;
  std::memcpy(session->device_identity.uuid,
              resident->device_uuid,
              sizeof(session->device_identity.uuid));
  std::memcpy(session->device_identity.name,
              properties.name,
              sizeof(session->device_identity.name));

  // This is deliberately the final fallible operation. Once the ready-event
  // dependency is accepted, returning the armed session transfers all native
  // and imported lifetimes together to Rust; no error path can free borrowed
  // parent storage after queuing a possible read.
  if (cudaStreamWaitEvent(session->stream, resident->ready_event, 0) != cudaSuccess) {
    return fail_session(NEO_POPULATION_STATUS_SYNC_FAILED);
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
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
  }
  if (session->parent_ownership == NEO_POPULATION_PARENT_BORROWED_RESIDENT_V3) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
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
  session->parent_ownership = NEO_POPULATION_PARENT_OWNED_V1;

  std::int32_t status = NEO_POPULATION_STATUS_OK;
  status = device_alloc(&session->close, bars);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->high, bars);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->low, bars);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->indicators_bar_major, features * bars);
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
  const std::size_t i64_bytes = bars * sizeof(std::int64_t);
  std::uint64_t uploaded = 0;
  status = copy_to_device(session->close, dataset->close, double_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->high, dataset->high, double_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->low, dataset->low, double_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  // ── Indicators: uploaded feature-major, kept bar-major ────────────────────
  //
  // The host contract does not move. `NeoPopulationDatasetView::indicators` is
  // still `feature_count * row_count` in feature-major order, which is what the
  // CPU oracle builds, what the parity fixtures assert against and what
  // prototypes A and C consume — so none of them is touched by this.
  //
  // The device wants the opposite. Every thread of a reduce block is at the
  // same bar and each wants up to 16 of that bar's features; feature-major puts
  // those 3.37 MB apart, which is up to 64 distinct cache lines per warp-bar.
  // Bar-major makes the whole feature row of a bar one contiguous 512-byte run
  // that the warp shares.
  //
  // So the feature-major bytes land in a STAGING buffer, one kernel transposes
  // them into the permanent one, and the staging buffer is freed before this
  // function returns. Steady-state resident memory is one copy, not two; the
  // transient second copy exists for the duration of one launch on a card that
  // has just been asked for the same amount again for the permanent buffer, so
  // if the transient one cannot be had, the dataset was never going to fit.
  {
    double* staging = nullptr;
    status = device_alloc(&staging, features * bars);
    if (status != NEO_POPULATION_STATUS_OK) return status;
    status = copy_to_device(staging, dataset->indicators, features * bars * sizeof(double),
                            session->stream);
    if (status != NEO_POPULATION_STATUS_OK) {
      device_free(staging);
      return status;
    }
    const dim3 transpose_block(kTransposeTile, kTransposeTile);
    const dim3 transpose_grid(
        static_cast<unsigned int>((bars + kTransposeTile - 1) / kTransposeTile),
        static_cast<unsigned int>((features + kTransposeTile - 1) / kTransposeTile));
    transpose_indicators_to_bar_major<<<transpose_grid, transpose_block, 0, session->stream>>>(
        staging, session->indicators_bar_major, static_cast<int>(bars),
        static_cast<int>(features));
    session->kernel_submissions += 1;
    // One synchronize per DATASET, not per launch: the staging buffer cannot be
    // freed until the transpose has read it, and this is also what makes every
    // other copy above complete before any evaluate can observe them.
    if (cudaStreamSynchronize(session->stream) != cudaSuccess) {
      device_free(staging);
      return NEO_POPULATION_STATUS_TRANSFER_FAILED;
    }
    session->synchronization_events += 1;
    device_free(staging);
    if (cudaGetLastError() != cudaSuccess) {
      return NEO_POPULATION_STATUS_LAUNCH_FAILED;
    }
  }
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
                                        features * bars * sizeof(double) +
                                        bars * kSmcSlots * sizeof(signed char));
  if (dataset->adaptive_base_pips != nullptr) {
    status = copy_to_device(session->adaptive_base_pips, dataset->adaptive_base_pips,
                            bars * sizeof(double), session->stream);
    if (status != NEO_POPULATION_STATUS_OK) return status;
    uploaded += static_cast<std::uint64_t>(bars * sizeof(double));
    session->has_adaptive_base = 1;
  }

  session->bars = static_cast<int>(bars);
  session->parent_rows = static_cast<int>(bars);
  session->feature_count = static_cast<int>(features);
  session->view_kind = static_cast<int>(NEO_POPULATION_VIEW_FULL);
  session->view_start = 0;
  session->timestamp_mode = static_cast<int>(NEO_POPULATION_TIMESTAMP_CANONICAL);
  session->dataset_upload_bytes = uploaded;
  session->has_dataset = true;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_upload_parent_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationParentDatasetV1* parent) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
  }
  if (session->parent_ownership == NEO_POPULATION_PARENT_BORROWED_RESIDENT_V3) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (parent == nullptr || parent->close == nullptr || parent->high == nullptr ||
      parent->low == nullptr || parent->indicators_feature_major == nullptr ||
      parent->months == nullptr || parent->days == nullptr || parent->timestamps == nullptr ||
      parent->smc_rows == nullptr) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (parent->header.abi_version != NEOETHOS_GPU_ABI_VERSION) {
    return NEO_POPULATION_STATUS_ABI_MISMATCH;
  }
  if (session->has_parent_v1 || session->has_dataset) {
    return NEO_POPULATION_STATUS_DATASET_REUPLOAD;
  }
  const std::uint64_t parent_rows_u64 = parent->header.row_count;
  const std::uint64_t features_u64 = parent->header.feature_count;
  if (parent_rows_u64 == 0ull || features_u64 == 0ull ||
      parent_rows_u64 > static_cast<std::uint64_t>(INT_MAX) ||
      features_u64 > static_cast<std::uint64_t>(INT_MAX) ||
      features_u64 > static_cast<std::uint64_t>(SIZE_MAX) / parent_rows_u64 ||
      features_u64 * parent_rows_u64 >
          static_cast<std::uint64_t>(SIZE_MAX / sizeof(double)) ||
      parent_rows_u64 > static_cast<std::uint64_t>(SIZE_MAX / kSmcSlots)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  session->parent_ownership = NEO_POPULATION_PARENT_OWNED_V1;

  const std::size_t parent_rows = static_cast<std::size_t>(parent_rows_u64);
  const std::size_t features = static_cast<std::size_t>(features_u64);
  const std::size_t feature_values = features * parent_rows;
  std::int32_t status = device_alloc(&session->close, parent_rows);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->high, parent_rows);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->low, parent_rows);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->indicators_feature_major, feature_values);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->months, parent_rows);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->days, parent_rows);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->timestamps, parent_rows);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->smc_rows, parent_rows * kSmcSlots);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = device_alloc(&session->view_indices, parent_rows);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  session->view_indices_capacity = parent_rows;
  status = device_alloc(&session->adaptive_base_pips, parent_rows);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  session->adaptive_base_pips_capacity = parent_rows;
  status = device_alloc(&session->gap_flags, parent_rows);
  if (status != NEO_POPULATION_STATUS_OK) return status;

  const std::size_t price_bytes = parent_rows * sizeof(double);
  const std::size_t index_bytes = parent_rows * sizeof(std::int64_t);
  const std::size_t feature_bytes = feature_values * sizeof(double);
  const std::size_t smc_bytes = parent_rows * kSmcSlots * sizeof(signed char);
  status = copy_to_device(session->close, parent->close, price_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->high, parent->high, price_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->low, parent->low, price_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->indicators_feature_major,
                          parent->indicators_feature_major,
                          feature_bytes,
                          session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->months, parent->months, index_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->days, parent->days, index_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->timestamps, parent->timestamps, index_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;
  status = copy_to_device(session->smc_rows, parent->smc_rows, smc_bytes, session->stream);
  if (status != NEO_POPULATION_STATUS_OK) return status;

  std::uint64_t parent_upload_bytes = 0ull;
  if (!checked_add_u64(parent_upload_bytes, static_cast<std::uint64_t>(price_bytes)) ||
      !checked_add_u64(parent_upload_bytes, static_cast<std::uint64_t>(price_bytes)) ||
      !checked_add_u64(parent_upload_bytes, static_cast<std::uint64_t>(price_bytes)) ||
      !checked_add_u64(parent_upload_bytes, static_cast<std::uint64_t>(index_bytes)) ||
      !checked_add_u64(parent_upload_bytes, static_cast<std::uint64_t>(index_bytes)) ||
      !checked_add_u64(parent_upload_bytes, static_cast<std::uint64_t>(index_bytes)) ||
      !checked_add_u64(parent_upload_bytes, static_cast<std::uint64_t>(feature_bytes)) ||
      !checked_add_u64(parent_upload_bytes, static_cast<std::uint64_t>(smc_bytes))) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  session->parent_rows = static_cast<int>(parent_rows);
  session->feature_count = static_cast<int>(features);
  session->dataset_upload_bytes = parent_upload_bytes;
  session->has_parent_v1 = true;
  session->residency_counters.parent_upload_count = 1ull;
  session->residency_counters.parent_upload_bytes = parent_upload_bytes;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_bind_view_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationEvaluationViewV1* view) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
  }
  if (view == nullptr || !session->has_parent_v1) {
    return NEO_POPULATION_STATUS_MISSING_UPLOAD;
  }
  if (view->abi_version != NEOETHOS_GPU_ABI_VERSION) {
    return NEO_POPULATION_STATUS_ABI_MISMATCH;
  }
  if (view->parent_row_count != static_cast<std::uint64_t>(session->parent_rows) ||
      view->row_count == 0ull || view->row_count > static_cast<std::uint64_t>(INT_MAX) ||
      view->timestamp_mode > NEO_POPULATION_TIMESTAMP_DISABLED_INDEX_DELTA ||
      (view->adaptive_base_pips == nullptr && view->adaptive_base_pips_len != 0) ||
      (view->adaptive_base_pips != nullptr &&
       view->adaptive_base_pips_len != static_cast<std::size_t>(view->row_count))) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (session->pending_event_id != 0ull && !session->metrics_ready) {
    return NEO_POPULATION_STATUS_SYNC_FAILED;
  }

  const std::size_t rows = static_cast<std::size_t>(view->row_count);
  switch (view->view_kind) {
    case NEO_POPULATION_VIEW_FULL:
      if (view->range_start != 0ull || view->row_count != view->parent_row_count ||
          view->ordered_indices != nullptr || view->ordered_index_count != 0) {
        return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
      }
      break;
    case NEO_POPULATION_VIEW_CONTIGUOUS_RANGE:
      if (view->range_start >= view->parent_row_count ||
          view->row_count > view->parent_row_count - view->range_start ||
          view->ordered_indices != nullptr || view->ordered_index_count != 0) {
        return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
      }
      break;
    case NEO_POPULATION_VIEW_ORDERED_INDICES:
      if (view->range_start != 0ull || view->ordered_indices == nullptr ||
          view->ordered_index_count != rows) {
        return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
      }
      for (std::size_t index = 0; index < rows; ++index) {
        if (view->ordered_indices[index] >= view->parent_row_count ||
            (index > 0 && view->ordered_indices[index - 1] >= view->ordered_indices[index])) {
          return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
        }
      }
      break;
    default:
      return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  const std::uint64_t ordered_upload_bytes =
      view->view_kind == NEO_POPULATION_VIEW_ORDERED_INDICES
          ? static_cast<std::uint64_t>(rows * sizeof(unsigned long long))
          : 0ull;
  const std::uint64_t adaptive_upload_bytes =
      view->adaptive_base_pips != nullptr
          ? static_cast<std::uint64_t>(rows * sizeof(double))
          : 0ull;
  const std::uint64_t kind_binding_count =
      view->view_kind == NEO_POPULATION_VIEW_FULL
          ? session->residency_counters.full_binding_count
          : (view->view_kind == NEO_POPULATION_VIEW_CONTIGUOUS_RANGE
                 ? session->residency_counters.range_binding_count
                 : session->residency_counters.ordered_binding_count);
  if (session->residency_counters.view_binding_count == UINT64_MAX ||
      kind_binding_count == UINT64_MAX ||
      ordered_upload_bytes >
          UINT64_MAX - session->residency_counters.ordered_index_upload_bytes ||
      adaptive_upload_bytes >
          UINT64_MAX - session->residency_counters.adaptive_upload_bytes) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  std::int32_t status = NEO_POPULATION_STATUS_OK;
  if (view->view_kind == NEO_POPULATION_VIEW_ORDERED_INDICES) {
    status = ensure_device_capacity_v3(&session->view_indices,
                                       &session->view_indices_capacity,
                                       rows);
    if (status != NEO_POPULATION_STATUS_OK) return status;
    const std::size_t bytes = rows * sizeof(unsigned long long);
    status = copy_to_device(session->view_indices, view->ordered_indices, bytes, session->stream);
    if (status != NEO_POPULATION_STATUS_OK) return status;
    session->residency_counters.ordered_index_upload_bytes += ordered_upload_bytes;
  }
  if (view->adaptive_base_pips != nullptr) {
    status = ensure_device_capacity_v3(&session->adaptive_base_pips,
                                       &session->adaptive_base_pips_capacity,
                                       rows);
    if (status != NEO_POPULATION_STATUS_OK) return status;
    const std::size_t bytes = rows * sizeof(double);
    status = copy_to_device(session->adaptive_base_pips,
                            view->adaptive_base_pips,
                            bytes,
                            session->stream);
    if (status != NEO_POPULATION_STATUS_OK) return status;
    session->residency_counters.adaptive_upload_bytes += adaptive_upload_bytes;
  }

  session->view_kind = static_cast<int>(view->view_kind);
  session->view_start = static_cast<int>(view->range_start);
  session->timestamp_mode = static_cast<int>(view->timestamp_mode);
  session->bars = static_cast<int>(rows);
  session->has_adaptive_base = view->adaptive_base_pips != nullptr ? 1 : 0;
  session->has_dataset = true;
  session->has_scenarios = false;
  session->scenario_count = 0;
  session->metrics_ready = false;
  session->pending_event_id = 0ull;
  session->residency_counters.view_binding_count += 1ull;
  if (view->view_kind == NEO_POPULATION_VIEW_FULL) {
    session->residency_counters.full_binding_count += 1ull;
  } else if (view->view_kind == NEO_POPULATION_VIEW_CONTIGUOUS_RANGE) {
    session->residency_counters.range_binding_count += 1ull;
  } else {
    session->residency_counters.ordered_binding_count += 1ull;
  }
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_read_residency_counters_v1(
    NeoCudaPopulationSession* session,
    NeoPopulationResidencyCountersV1* counters) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
  }
  if (counters == nullptr) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  *counters = session->residency_counters;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_read_device_identity_v1(
    NeoCudaPopulationSession* session,
    NeoPopulationDeviceIdentityV1* identity) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
  }
  if (identity == nullptr) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  *identity = session->device_identity;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_upload_genes(
    NeoCudaPopulationSession* session,
    const NeoPopulationGeneView* genes) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
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
  if (population == SIZE_MAX || population > SIZE_MAX / kSmcSlots ||
      terms > SIZE_MAX / sizeof(double)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  // Bar-major turned an out-of-range feature index from LOUD into SILENT.
  //
  // Feature-major addressed `indicators[feature * bars + bar]`, so a `feature`
  // equal to `feature_count` read past the end of a `feature_count * bars`
  // buffer — an out-of-bounds access compute-sanitizer catches immediately.
  // Bar-major addresses `indicators_bar_major[bar * feature_count + feature]`,
  // where the same index resolves to `(bar + 1) * feature_count + 0`: in
  // bounds, a perfectly plausible double belonging to the NEXT bar, and nobody
  // ever knows. The hardware was a second, independent detector and it is gone.
  //
  // So the check moves here, once per upload rather than once per term-read.
  // `PopulationGeneView::validate` refuses the same thing on the host; this is
  // what makes the native side safe on its own instead of trusting the wrapper.
  for (std::size_t term = 0; term < terms; ++term) {
    const int feature = genes->indices[term];
    if (feature < 0 || feature >= session->feature_count) {
      return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
    }
  }
  // Host-side staging of the descriptor-derived arrays keeps the device layout
  // flat and coalesced without changing canonical identity or ordering.
  auto* staging = new (std::nothrow) GeneHostStagingV1();
  if (staging == nullptr) {
    return NEO_POPULATION_STATUS_ALLOCATION_FAILED;
  }
  staging->candidate_ids = new (std::nothrow) unsigned long long[population];
  staging->offsets = new (std::nothrow) int[population + 1];
  if (terms > 0) {
    staging->indices = new (std::nothrow) int[terms];
    staging->weights = new (std::nothrow) double[terms];
  }
  staging->long_thresholds = new (std::nothrow) double[population];
  staging->short_thresholds = new (std::nothrow) double[population];
  staging->stop_pips = new (std::nothrow) double[population];
  staging->target_pips = new (std::nothrow) double[population];
  staging->stop_vol_multipliers = new (std::nothrow) double[population];
  staging->smc_flags = new (std::nothrow) signed char[population * kSmcSlots];
  staging->smc_weights = new (std::nothrow) double[kSmcSlots];
  if (staging->candidate_ids == nullptr || staging->offsets == nullptr ||
      (terms > 0 && (staging->indices == nullptr || staging->weights == nullptr)) ||
      staging->long_thresholds == nullptr || staging->short_thresholds == nullptr ||
      staging->stop_pips == nullptr || staging->target_pips == nullptr ||
      staging->stop_vol_multipliers == nullptr || staging->smc_flags == nullptr ||
      staging->smc_weights == nullptr) {
    delete staging;
    return NEO_POPULATION_STATUS_ALLOCATION_FAILED;
  }
  for (std::size_t index = 0; index < population; ++index) {
    staging->candidate_ids[index] = genes->descriptors[index].candidate_id;
    staging->long_thresholds[index] = genes->descriptors[index].long_threshold;
    staging->short_thresholds[index] = genes->descriptors[index].short_threshold;
  }
  std::memcpy(staging->offsets, genes->offsets, (population + 1) * sizeof(int));
  if (terms > 0) {
    std::memcpy(staging->indices, genes->indices, terms * sizeof(int));
    std::memcpy(staging->weights, genes->weights, terms * sizeof(double));
  }
  std::memcpy(staging->stop_pips, genes->stop_pips, population * sizeof(double));
  std::memcpy(staging->target_pips, genes->target_pips, population * sizeof(double));
  std::memcpy(staging->stop_vol_multipliers,
              genes->stop_vol_multipliers,
              population * sizeof(double));
  std::memcpy(staging->smc_flags,
              genes->smc_flags,
              population * kSmcSlots * sizeof(signed char));
  std::memcpy(staging->smc_weights, genes->smc_weights, kSmcSlots * sizeof(double));

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
    delete staging;
    return status;
  }

  if (!guard(copy_to_device(session->candidate_ids, staging->candidate_ids,
                            population * sizeof(unsigned long long), session->stream)) ||
      !guard(copy_to_device(session->gene_offsets, staging->offsets,
                            (population + 1) * sizeof(int), session->stream)) ||
      !guard(copy_to_device(session->gene_indices, staging->indices, terms * sizeof(int),
                            session->stream)) ||
      !guard(copy_to_device(session->gene_weights, staging->weights, terms * sizeof(double),
                            session->stream)) ||
      !guard(copy_to_device(session->long_thresholds, staging->long_thresholds,
                            population * sizeof(double), session->stream)) ||
      !guard(copy_to_device(session->short_thresholds, staging->short_thresholds,
                            population * sizeof(double), session->stream)) ||
      !guard(copy_to_device(session->stop_pips, staging->stop_pips,
                            population * sizeof(double),
                            session->stream)) ||
      !guard(copy_to_device(session->target_pips, staging->target_pips,
                            population * sizeof(double), session->stream)) ||
      !guard(copy_to_device(session->stop_vol_multipliers,
                            staging->stop_vol_multipliers,
                            population * sizeof(double), session->stream)) ||
      !guard(copy_to_device(session->smc_flags, staging->smc_flags,
                            population * kSmcSlots * sizeof(signed char), session->stream)) ||
      !guard(copy_to_device(session->smc_weights, staging->smc_weights,
                            kSmcSlots * sizeof(double), session->stream))) {
    cudaStreamSynchronize(session->stream);
    delete staging;
    return status;
  }
  status = release_staging_after_stream_v1(session->stream, staging,
                                            release_gene_host_staging_v1);
  if (status != NEO_POPULATION_STATUS_OK) return status;

  session->population = static_cast<int>(population);
  session->gate_threshold = genes->gate_threshold;
  session->smc_gate_disabled = static_cast<int>(genes->smc_gate_disabled);
  session->gene_upload_bytes = static_cast<std::uint64_t>(
      population * (sizeof(unsigned long long) + 5 * sizeof(double) +
                    kSmcSlots * sizeof(signed char)) +
      (population + 1) * sizeof(int) + terms * (sizeof(int) + sizeof(double)) +
      kSmcSlots * sizeof(double));
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
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
  }
  if (!session->has_genes) {
    return NEO_POPULATION_STATUS_MISSING_UPLOAD;
  }
  if (scenarios == nullptr || scenarios->descriptors == nullptr || scenarios->count == 0) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  // The count is NO LONGER required to equal the population.
  //
  // That equality was the whole reason a screen wanting 101 treatments of one
  // gene had to clone the gene 101 times and send 101 gene descriptors. What
  // must still hold is that every scenario names a gene that EXISTS and a window
  // that is inside the series — both checked below, per descriptor, because
  // neither is something the kernel can detect for itself: an out-of-range gene
  // index is an out-of-bounds read of thresholds and CSR offsets that still
  // produces a plausible metric row.
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }

  const std::size_t count = scenarios->count;
  const auto population = static_cast<unsigned long long>(session->population);
  const auto bars = static_cast<unsigned long long>(session->bars);
  for (std::size_t index = 0; index < count; ++index) {
    const NeoScenarioDescriptor& descriptor = scenarios->descriptors[index];
    if (descriptor.base_candidate_id >= population) {
      return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
    }
    if (descriptor.window_offset >= bars) {
      return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
    }
    const unsigned long long length =
        descriptor.window_len == 0u ? (bars - descriptor.window_offset)
                                    : static_cast<unsigned long long>(descriptor.window_len);
    if (descriptor.window_offset + length > bars) {
      return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
    }
    if (descriptor.scenario_type != kScenarioBase &&
        descriptor.scenario_type != kScenarioPerturb &&
        descriptor.scenario_type != kScenarioCost) {
      return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
    }
    // Only the "no override" sentinel may be negative. A stray -2 would divide
    // to -0.002 pips and charge a NEGATIVE spread — free money, and a screen
    // that reports every strategy as robust.
    if (descriptor.spread_ticks < kNoTickOverride ||
        descriptor.commission_micros < kNoMicroOverride) {
      return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
    }
    // Slippage has NO sentinel — 0 already means "none" — so its bound is 0.
    // It is applied SIGNED (`entry_price += direction * slippage_price`,
    // `exit_price -= direction * slippage_price`), so a negative value is a
    // favourable fill on the entry and on the exit both. This was the one cost
    // field neither validator checked.
    if (descriptor.slippage_ticks < 0) {
      return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
    }
  }

  // Host staging, struct-of-arrays, exactly as `upload_genes` stages its
  // descriptor-derived arrays: the device wants each field contiguous so a warp
  // reads one cache line per field instead of one per lane.
  auto* staging = new (std::nothrow) ScenarioHostStagingV1();
  if (staging == nullptr) {
    return NEO_POPULATION_STATUS_ALLOCATION_FAILED;
  }
  staging->base_ids = new (std::nothrow) unsigned long long[count];
  staging->ids = new (std::nothrow) unsigned long long[count];
  staging->counters = new (std::nothrow) unsigned long long[count];
  staging->offsets = new (std::nothrow) unsigned long long[count];
  staging->lens = new (std::nothrow) unsigned int[count];
  staging->types = new (std::nothrow) unsigned int[count];
  staging->spreads = new (std::nothrow) int[count];
  staging->slippages = new (std::nothrow) int[count];
  staging->commissions = new (std::nothrow) long long[count];
  if (staging->base_ids == nullptr || staging->ids == nullptr || staging->counters == nullptr ||
      staging->offsets == nullptr || staging->lens == nullptr || staging->types == nullptr ||
      staging->spreads == nullptr || staging->slippages == nullptr ||
      staging->commissions == nullptr) {
    delete staging;
    return NEO_POPULATION_STATUS_ALLOCATION_FAILED;
  }
  for (std::size_t index = 0; index < count; ++index) {
    const NeoScenarioDescriptor& descriptor = scenarios->descriptors[index];
    staging->base_ids[index] = descriptor.base_candidate_id;
    staging->ids[index] = descriptor.scenario_id;
    staging->counters[index] = descriptor.rng_counter;
    staging->offsets[index] = descriptor.window_offset;
    staging->lens[index] = descriptor.window_len;
    staging->types[index] = descriptor.scenario_type;
    staging->spreads[index] = descriptor.spread_ticks;
    staging->slippages[index] = descriptor.slippage_ticks;
    staging->commissions[index] = descriptor.commission_micros;
  }

  session->release_scenarios();
  std::int32_t status = NEO_POPULATION_STATUS_OK;
  const auto guard = [&](std::int32_t code) {
    if (code != NEO_POPULATION_STATUS_OK) {
      status = code;
    }
    return status == NEO_POPULATION_STATUS_OK;
  };
  if (!guard(device_alloc(&session->scenario_base_candidate_ids, count)) ||
      !guard(device_alloc(&session->scenario_ids, count)) ||
      !guard(device_alloc(&session->scenario_rng_counters, count)) ||
      !guard(device_alloc(&session->scenario_window_offsets, count)) ||
      !guard(device_alloc(&session->scenario_window_lens, count)) ||
      !guard(device_alloc(&session->scenario_types, count)) ||
      !guard(device_alloc(&session->scenario_spread_ticks, count)) ||
      !guard(device_alloc(&session->scenario_slippage_ticks, count)) ||
      !guard(device_alloc(&session->scenario_commission_micros, count))) {
    delete staging;
    session->release_scenarios();
    return status;
  }

  const std::size_t u64_bytes = count * sizeof(unsigned long long);
  const std::size_t u32_bytes = count * sizeof(unsigned int);
  const std::size_t i32_bytes = count * sizeof(int);
  const std::size_t i64_bytes = count * sizeof(long long);
  if (!guard(copy_to_device(session->scenario_base_candidate_ids, staging->base_ids, u64_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_ids, staging->ids, u64_bytes, session->stream)) ||
      !guard(copy_to_device(session->scenario_rng_counters, staging->counters, u64_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_window_offsets, staging->offsets, u64_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_window_lens, staging->lens, u32_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_types, staging->types, u32_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_spread_ticks, staging->spreads, i32_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_slippage_ticks, staging->slippages, i32_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_commission_micros, staging->commissions, i64_bytes,
                            session->stream))) {
    cudaStreamSynchronize(session->stream);
    delete staging;
    return status;
  }
  status = release_staging_after_stream_v1(session->stream, staging,
                                            release_scenario_host_staging_v1);
  if (status != NEO_POPULATION_STATUS_OK) return status;

  session->scenario_count = static_cast<int>(count);
  session->scenario_upload_bytes =
      static_cast<std::uint64_t>(4 * u64_bytes + 2 * u32_bytes + 2 * i32_bytes + i64_bytes);
  session->has_scenarios = true;
  session->metrics_ready = false;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_upload_resident_scenarios_v2(
    NeoCudaPopulationSession* session,
    const NeoPopulationScenarioView* scenarios,
    std::uint64_t planned_population) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (session->has_genes || planned_population == 0 ||
      planned_population > static_cast<std::uint64_t>(INT_MAX) ||
      (session->resident_planned_population_v2 != 0 &&
       planned_population !=
           static_cast<std::uint64_t>(session->resident_planned_population_v2))) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  const int prior_population = session->population;
  session->population = static_cast<int>(planned_population);
  // Reuse the byte-identical scenario validator/uploader. The temporary flag is
  // host-local admission state only; no gene pointer is allocated or uploaded.
  session->has_genes = true;
  const std::int32_t status =
      neoethos_gpu_cuda_population_upload_scenarios(session, scenarios);
  session->has_genes = false;
  if (status != NEO_POPULATION_STATUS_OK) {
    session->population = prior_population;
    return status;
  }
  session->uses_resident_gene_view_v2 = true;
  session->gene_upload_bytes = 0;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t
neoethos_gpu_cuda_population_reserve_resident_search_runtime_v2(
    void* opaque_session,
    neoethos::resident_search_generation_v2::NeoResidentSearchRuntimeFactsV2* facts) {
  auto* session = static_cast<NeoCudaPopulationSession*>(opaque_session);
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (facts == nullptr || session->resident_search_runtime_reserved_v2 ||
      session->resident_generation_run_v2 != nullptr ||
      session->resident_scoring_run_v2 != nullptr ||
      strict_population_work_blocks_host_boundary_v1(session)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  const std::uint64_t ordinal =
      session->next_resident_search_admission_ordinal_v2;
  const std::int32_t status =
      read_resident_search_runtime_facts_v2(session, ordinal, facts);
  if (status != NEO_POPULATION_STATUS_OK) {
    return status;
  }
  session->resident_search_runtime_facts_v2 = *facts;
  session->resident_search_runtime_reserved_v2 = true;
  ++session->next_resident_search_admission_ordinal_v2;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t
neoethos_gpu_cuda_population_query_resident_search_combined_v2(
    void* opaque_session,
    const neoethos::resident_generation_v1::NeoResidentGenerationPlanV1*
        generation_plan,
    const neoethos::resident_scoring_novelty_v1::
        NeoResidentScoringNoveltyPlanV1* scoring_plan,
    const neoethos::resident_search_generation_v2::
        NeoResidentSearchRuntimeFactsV2* expected_runtime,
    neoethos::resident_search_generation_v2::
        NeoResidentSearchCombinedAdmissionV2* admission) {
  using namespace neoethos::resident_generation_v1;
  using namespace neoethos::resident_scoring_novelty_v1;
  using namespace neoethos::resident_search_generation_v2;
  auto* session = static_cast<NeoCudaPopulationSession*>(opaque_session);
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (generation_plan == nullptr || scoring_plan == nullptr ||
      expected_runtime == nullptr || admission == nullptr ||
      !session->resident_search_runtime_reserved_v2 ||
      !runtime_facts_equal_v2(session->resident_search_runtime_facts_v2,
                              *expected_runtime) ||
      session->resident_generation_run_v2 != nullptr ||
      session->resident_scoring_run_v2 != nullptr || !session->has_dataset ||
      generation_plan->logical_population_count == 0 ||
      generation_plan->logical_population_count > INT_MAX ||
      generation_plan->logical_population_count !=
          scoring_plan->logical_population_count ||
      generation_plan->feature_count != scoring_plan->feature_count ||
      generation_plan->max_terms_per_gene != scoring_plan->max_terms_per_gene ||
      generation_plan->feature_count !=
          static_cast<std::uint64_t>(session->feature_count)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  NeoResidentSearchRuntimeFactsV2 current_runtime{};
  std::int32_t status = read_resident_search_runtime_facts_v2(
      session, expected_runtime->run_admission_ordinal, &current_runtime);
  if (status != NEO_POPULATION_STATUS_OK ||
      !runtime_facts_equal_v2(current_runtime, *expected_runtime)) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  std::size_t same_context_free = 0;
  std::size_t same_context_total = 0;
  // This is the sole free-memory snapshot for both device stores.
  if (cudaMemGetInfo(&same_context_free, &same_context_total) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  NeoResidentGenerationAllocationReceiptV1 generation{};
  status = calculate_resident_generation_allocation_v2(
      generation_plan, session->stream, same_context_free,
      expected_runtime->allocator_context_reserve_bytes, &generation);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  NeoResidentScoringNoveltyAllocationReceiptV1 scoring{};
  status = calculate_resident_scoring_allocation_v2(
      scoring_plan, session->stream, same_context_free,
      expected_runtime->allocator_context_reserve_bytes, &scoring);
  if (status != NEO_SCORING_STATUS_OK_V1) {
    return status;
  }
  if (generation.total_device_bytes > UINT64_MAX - scoring.total_device_bytes) {
    return NEO_POPULATION_STATUS_ALLOCATION_FAILED;
  }
  const std::uint64_t total_device_bytes =
      generation.total_device_bytes + scoring.total_device_bytes;
  if (expected_runtime->allocator_context_reserve_bytes > same_context_free ||
      total_device_bytes >
          same_context_free -
              expected_runtime->allocator_context_reserve_bytes) {
    return NEO_POPULATION_STATUS_ALLOCATION_FAILED;
  }
  *admission = {};
  admission->abi_version = NEO_RESIDENT_SEARCH_GENERATION_ABI_V2;
  admission->free_memory_snapshot_count = 1u;
  admission->generation_allocation_count = 1u;
  admission->scoring_allocation_count = 1u;
  admission->terminal_host_allocation_count = 1u;
  admission->terminal_host_receipt_bytes =
      sizeof(neoethos::resident_generation_v2::
                 NeoResidentSearchTerminalReceiptV2);
  admission->same_context_free_bytes = same_context_free;
  admission->same_context_total_bytes = same_context_total;
  admission->full_discovery_reserve_bytes =
      expected_runtime->allocator_context_reserve_bytes;
  admission->generation_device_bytes = generation.total_device_bytes;
  admission->scoring_device_bytes = scoring.total_device_bytes;
  admission->total_device_bytes = total_device_bytes;
  admission->pool_reserved_current_bytes =
      expected_runtime->pool_reserved_current_bytes;
  admission->pool_used_current_bytes =
      expected_runtime->pool_used_current_bytes;
  admission->runtime = *expected_runtime;
  admission->generation = generation;
  admission->scoring = scoring;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t
neoethos_gpu_cuda_population_create_resident_search_combined_v2(
    void* opaque_session,
    const neoethos::resident_generation_v1::NeoResidentGenerationPlanV1*
        generation_plan,
    const neoethos::resident_scoring_novelty_v1::
        NeoResidentScoringNoveltyPlanV1* scoring_plan,
    const neoethos::resident_search_generation_v2::
        NeoResidentSearchCombinedAdmissionV2* admission,
    neoethos::resident_generation_v1::NeoResidentGenerationRunV1** generation,
    neoethos::resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1**
        scoring) {
  using namespace neoethos::resident_generation_v1;
  using namespace neoethos::resident_generation_v2;
  using namespace neoethos::resident_scoring_novelty_v1;
  using namespace neoethos::resident_search_generation_v2;
  auto* session = static_cast<NeoCudaPopulationSession*>(opaque_session);
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (generation_plan == nullptr || scoring_plan == nullptr ||
      admission == nullptr || generation == nullptr || scoring == nullptr ||
      *generation != nullptr || *scoring != nullptr ||
      !session->resident_search_runtime_reserved_v2 ||
      session->strict_execution_state !=
          PopulationStrictExecutionStateV1::StrictIdle ||
      session->resident_generation_run_v2 != nullptr ||
      session->resident_scoring_run_v2 != nullptr ||
      session->generation_ready_event_v2 != nullptr ||
      session->scoring_ready_event_v2 != nullptr ||
      session->resident_search_terminal_host_receipt_v2 != nullptr ||
      admission->abi_version != NEO_RESIDENT_SEARCH_GENERATION_ABI_V2 ||
      admission->flags != 0u || admission->free_memory_snapshot_count != 1u ||
      admission->generation_allocation_count != 1u ||
      admission->scoring_allocation_count != 1u ||
      admission->terminal_host_allocation_count != 1u ||
      admission->terminal_host_receipt_bytes !=
          sizeof(NeoResidentSearchTerminalReceiptV2) ||
      admission->generation_device_bytes !=
          admission->generation.total_device_bytes ||
      admission->scoring_device_bytes != admission->scoring.total_device_bytes ||
      admission->generation_device_bytes >
          UINT64_MAX - admission->scoring_device_bytes ||
      admission->total_device_bytes !=
          admission->generation_device_bytes + admission->scoring_device_bytes ||
      admission->full_discovery_reserve_bytes !=
          admission->runtime.allocator_context_reserve_bytes ||
      !bytes_nonzero_v2(admission->receipt_identity_sha256,
                        sizeof(admission->receipt_identity_sha256)) ||
      !runtime_facts_equal_v2(session->resident_search_runtime_facts_v2,
                              admission->runtime)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  NeoResidentSearchRuntimeFactsV2 current_runtime{};
  std::int32_t status = read_resident_search_runtime_facts_v2(
      session, admission->runtime.run_admission_ordinal, &current_runtime);
  if (status != NEO_POPULATION_STATUS_OK ||
      !runtime_facts_equal_v2(current_runtime, admission->runtime)) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  // The combined receipt is already sealed here. Only now may events, pinned
  // terminal storage, or either device arena be allocated. Keep every stage
  // local until the complete owner graph exists, so no partial owner is ever
  // published through either the session or an output parameter.
  cudaEvent_t created_generation_ready_event = nullptr;
  cudaEvent_t created_scoring_ready_event = nullptr;
  NeoResidentSearchTerminalReceiptV2* created_terminal_host_receipt = nullptr;
  NeoResidentGenerationRunV1* created_generation = nullptr;
  NeoResidentScoringNoveltyRunV1* created_scoring = nullptr;
  const auto unwind_combined_create = [&](std::int32_t primary_status,
                                          bool stream_state_unknown) {
    bool cleanup_complete = true;
    if (created_scoring != nullptr) {
      const std::int32_t cleanup_status =
          enqueue_resident_scoring_release_v2(created_scoring);
      if (cleanup_status == NEO_SCORING_STATUS_OK_V1) {
        created_scoring = nullptr;
      } else {
        cleanup_complete = false;
        if (cleanup_status ==
            NEO_SCORING_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2) {
          primary_status = NEO_POPULATION_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN;
          stream_state_unknown = true;
        }
      }
    }
    if (created_generation != nullptr) {
      const std::int32_t cleanup_status =
          enqueue_resident_generation_release_v1(created_generation);
      if (cleanup_status == NEO_RESIDENT_STATUS_OK_V1) {
        created_generation = nullptr;
      } else {
        cleanup_complete = false;
        if (cleanup_status ==
            NEO_RESIDENT_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2) {
          primary_status = NEO_POPULATION_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN;
          stream_state_unknown = true;
        }
      }
    }
    if (created_generation == nullptr &&
        created_terminal_host_receipt != nullptr) {
      if (cudaFreeHost(created_terminal_host_receipt) == cudaSuccess) {
        created_terminal_host_receipt = nullptr;
      } else {
        cleanup_complete = false;
      }
    }
    if (created_scoring == nullptr && created_scoring_ready_event != nullptr) {
      if (cudaEventDestroy(created_scoring_ready_event) == cudaSuccess) {
        created_scoring_ready_event = nullptr;
      } else {
        cleanup_complete = false;
      }
    }
    if (created_generation == nullptr &&
        created_generation_ready_event != nullptr) {
      if (cudaEventDestroy(created_generation_ready_event) == cudaSuccess) {
        created_generation_ready_event = nullptr;
      } else {
        cleanup_complete = false;
      }
    }
    *generation = nullptr;
    *scoring = nullptr;
    if (cleanup_complete && !stream_state_unknown) {
      session->resident_search_runtime_reserved_v2 = false;
      session->resident_search_runtime_facts_v2 = {};
      session->strict_execution_state =
          PopulationStrictExecutionStateV1::StrictIdle;
      return primary_status;
    }
    // Preserve every resource whose release could not be proven. A poisoned
    // session is leak-only and cannot reuse these handles or the admitted
    // stream, avoiding both UAF and silent partial-owner publication.
    session->generation_ready_event_v2 = created_generation_ready_event;
    session->scoring_ready_event_v2 = created_scoring_ready_event;
    session->resident_search_terminal_host_receipt_v2 =
        created_terminal_host_receipt;
    session->resident_generation_run_v2 = created_generation;
    session->resident_scoring_run_v2 = created_scoring;
    session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;
    if (primary_status == NEO_POPULATION_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN ||
        primary_status ==
            NEO_POPULATION_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN) {
      return primary_status;
    }
    return NEO_POPULATION_STATUS_LAUNCH_FAILED;
  };
  cudaEvent_t attempted_generation_ready_event = nullptr;
  if (cudaEventCreateWithFlags(&attempted_generation_ready_event,
                               cudaEventDisableTiming) != cudaSuccess) {
    // A Runtime API error may be an earlier asynchronous fault. Never publish,
    // destroy, or retry an attempted output handle whose creation is unknown.
    return unwind_combined_create(
        NEO_POPULATION_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN, true);
  }
  created_generation_ready_event = attempted_generation_ready_event;
  cudaEvent_t attempted_scoring_ready_event = nullptr;
  if (cudaEventCreateWithFlags(&attempted_scoring_ready_event,
                               cudaEventDisableTiming) != cudaSuccess) {
    return unwind_combined_create(
        NEO_POPULATION_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN, true);
  }
  created_scoring_ready_event = attempted_scoring_ready_event;
  NeoResidentSearchTerminalReceiptV2* attempted_terminal_host_receipt = nullptr;
  if (cudaHostAlloc(reinterpret_cast<void**>(&attempted_terminal_host_receipt),
                    sizeof(NeoResidentSearchTerminalReceiptV2),
                    cudaHostAllocPortable) != cudaSuccess) {
    return unwind_combined_create(
        NEO_POPULATION_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN, true);
  }
  created_terminal_host_receipt = attempted_terminal_host_receipt;
  if (cudaEventRecord(session->event, session->stream) != cudaSuccess) {
    return unwind_combined_create(NEO_POPULATION_STATUS_LAUNCH_FAILED, true);
  }
  NeoResidentGenerationPopulationSessionImportV1 generation_import{};
  generation_import.abi_version = NEO_RESIDENT_GENERATION_ABI_V1;
  generation_import.selected_cuda_ordinal =
      admission->runtime.selected_cuda_ordinal;
  generation_import.admitted_run_stream = session->stream;
  generation_import.resident_parent_ready_event = session->event;
  generation_import.generation_ready_event = created_generation_ready_event;
  generation_import.population_lifetime_owner = session;
  generation_import.full_discovery_reserve_bytes =
      admission->full_discovery_reserve_bytes;
  std::memcpy(generation_import.cuda_device_identity_sha256,
              scoring_plan->cuda_device_identity_sha256, 32);
  std::memcpy(generation_import.primary_context_identity_sha256,
              scoring_plan->primary_context_identity_sha256, 32);
  std::memcpy(generation_import.run_stream_identity_sha256,
              scoring_plan->run_stream_identity_sha256, 32);
  std::memcpy(generation_import.cuda_build_manifest_sha256,
              generation_plan->cuda_build_manifest_sha256, 32);
  std::memcpy(generation_import.resident_input_content_sha256,
              generation_plan->strategy_gene_schema_sha256, 32);
  status = create_resident_generation_run_from_import_v1(
      &generation_import, generation_plan, &admission->generation,
      &created_generation);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    const bool stream_state_unknown =
        status == NEO_RESIDENT_STATUS_CUDA_ERROR_V1 ||
        status == NEO_RESIDENT_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2 ||
        status ==
            NEO_RESIDENT_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2;
    return unwind_combined_create(population_status_from_generation_v2(status),
                                  stream_state_unknown);
  }
  status = bind_resident_search_terminal_receipt_v2(
      created_generation, created_terminal_host_receipt);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return unwind_combined_create(status, false);
  }
  NeoResidentScoringAdmissionV2 scoring_admission{};
  scoring_admission.abi_version = 2u;
  scoring_admission.selected_cuda_ordinal =
      admission->runtime.selected_cuda_ordinal;
  scoring_admission.admitted_run_stream = session->stream;
  scoring_admission.scoring_novelty_ready_event =
      created_scoring_ready_event;
  scoring_admission.full_discovery_reserve_bytes =
      admission->full_discovery_reserve_bytes;
  std::memcpy(scoring_admission.cuda_device_identity_sha256,
              scoring_plan->cuda_device_identity_sha256, 32);
  std::memcpy(scoring_admission.primary_context_identity_sha256,
              scoring_plan->primary_context_identity_sha256, 32);
  std::memcpy(scoring_admission.run_stream_identity_sha256,
              scoring_plan->run_stream_identity_sha256, 32);
  status = create_unbound_resident_scoring_run_v2(
      &scoring_admission, scoring_plan, &admission->scoring, &created_scoring);
  if (status != NEO_SCORING_STATUS_OK_V1) {
    const bool stream_state_unknown =
        status == NEO_SCORING_STATUS_CUDA_ERROR_V1 ||
        status == NEO_SCORING_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2 ||
        status == NEO_SCORING_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2;
    return unwind_combined_create(population_status_from_scoring_v2(status),
                                  stream_state_unknown);
  }
  session->generation_ready_event_v2 = created_generation_ready_event;
  session->scoring_ready_event_v2 = created_scoring_ready_event;
  session->resident_search_terminal_host_receipt_v2 =
      created_terminal_host_receipt;
  session->resident_generation_run_v2 = created_generation;
  session->resident_scoring_run_v2 = created_scoring;
  *generation = created_generation;
  *scoring = created_scoring;
  session->resident_planned_population_v2 =
      static_cast<int>(generation_plan->logical_population_count);
  session->population = session->resident_planned_population_v2;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_create_resident_generation_run_v2(
    NeoCudaPopulationSession* session,
    const neoethos::resident_generation_v1::NeoResidentGenerationPlanV1* plan,
    neoethos::resident_generation_v1::NeoResidentGenerationAllocationReceiptV1* allocation,
    neoethos::resident_generation_v1::NeoResidentGenerationRunV1** run) {
  using namespace neoethos::resident_generation_v1;
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
  }
  if (plan == nullptr || allocation == nullptr || run == nullptr || *run != nullptr ||
      session->resident_generation_run_v2 != nullptr || !session->has_dataset ||
      plan->logical_population_count == 0 || plan->logical_population_count > INT_MAX ||
      plan->feature_count != static_cast<std::uint64_t>(session->feature_count) ||
      (session->resident_planned_population_v2 != 0 &&
       plan->logical_population_count !=
           static_cast<std::uint64_t>(session->resident_planned_population_v2))) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  if (session->generation_ready_event_v2 == nullptr) {
    cudaEvent_t attempted_generation_ready_event = nullptr;
    if (cudaEventCreateWithFlags(&attempted_generation_ready_event,
                                 cudaEventDisableTiming) != cudaSuccess) {
      session->strict_execution_state =
          PopulationStrictExecutionStateV1::Poisoned;
      return NEO_POPULATION_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN;
    }
    session->generation_ready_event_v2 = attempted_generation_ready_event;
  }
  if (cudaEventRecord(session->event, session->stream) != cudaSuccess) {
    session->strict_execution_state =
        PopulationStrictExecutionStateV1::Poisoned;
    return NEO_POPULATION_STATUS_LAUNCH_FAILED;
  }

  NeoResidentGenerationPopulationSessionImportV1 import{};
  import.abi_version = NEO_RESIDENT_GENERATION_ABI_V1;
  import.selected_cuda_ordinal = static_cast<std::uint32_t>(session->device);
  import.admitted_run_stream = session->stream;
  import.resident_parent_ready_event = session->event;
  import.generation_ready_event = session->generation_ready_event_v2;
  import.population_lifetime_owner = session;
  import.full_discovery_reserve_bytes = 0;
  std::memcpy(import.cuda_device_identity_sha256, plan->run_identity_sha256, 32);
  std::memcpy(import.primary_context_identity_sha256, plan->plan_identity_sha256, 32);
  std::memcpy(import.run_stream_identity_sha256, plan->generation_semantics_sha256, 32);
  std::memcpy(import.cuda_build_manifest_sha256, plan->cuda_build_manifest_sha256, 32);
  std::memcpy(import.resident_input_content_sha256,
              plan->strategy_gene_schema_sha256, 32);

  std::int32_t status = query_resident_generation_allocation_v1(&import, plan, allocation);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    return status;
  }
  status = create_resident_generation_run_from_import_v1(&import, plan, allocation, run);
  if (status != NEO_RESIDENT_STATUS_OK_V1) {
    if (*run != nullptr) {
      // A failed creator may never transfer a retryable device identity. Keep
      // an unexpected host tombstone only to block reuse; never free it again.
      session->resident_generation_run_v2 = *run;
      session->strict_execution_state =
          PopulationStrictExecutionStateV1::Poisoned;
      return NEO_POPULATION_STATUS_LAUNCH_FAILED;
    }
    if (status == NEO_RESIDENT_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2 ||
        status ==
            NEO_RESIDENT_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2) {
      session->strict_execution_state =
          PopulationStrictExecutionStateV1::Poisoned;
    }
    return population_status_from_generation_v2(status);
  }
  session->resident_generation_run_v2 = *run;
  session->resident_planned_population_v2 =
      static_cast<int>(plan->logical_population_count);
  session->population = session->resident_planned_population_v2;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_release_resident_generation_run_v2(
    NeoCudaPopulationSession* session,
    neoethos::resident_generation_v1::NeoResidentGenerationRunV1* run) {
  using namespace neoethos::resident_generation_v1;
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (run == nullptr || session->resident_generation_run_v2 != run ||
      strict_population_work_blocks_host_boundary_v1(session)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (session->resident_search_terminal_host_receipt_v2 != nullptr) {
    const std::int32_t detach_status =
        detach_resident_search_terminal_receipt_v2(
            run, session->resident_search_terminal_host_receipt_v2);
    if (detach_status != NEO_RESIDENT_STATUS_OK_V1) {
      return detach_status;
    }
    if (cudaFreeHost(session->resident_search_terminal_host_receipt_v2) !=
        cudaSuccess) {
      session->strict_execution_state =
          PopulationStrictExecutionStateV1::Poisoned;
      return NEO_POPULATION_STATUS_LAUNCH_FAILED;
    }
    session->resident_search_terminal_host_receipt_v2 = nullptr;
  }
  const std::int32_t status = enqueue_resident_generation_release_v1(run);
  if (status == NEO_RESIDENT_STATUS_OK_V1) {
    session->resident_generation_run_v2 = nullptr;
    session->resident_search_runtime_reserved_v2 = false;
  } else {
    session->strict_execution_state =
        PopulationStrictExecutionStateV1::Poisoned;
  }
  return population_status_from_generation_v2(status);
}

extern "C" std::int32_t
neoethos_gpu_cuda_population_create_unbound_resident_scoring_run_v2(
    NeoCudaPopulationSession* session,
    const neoethos::resident_scoring_novelty_v1::NeoResidentScoringNoveltyPlanV1* plan,
    neoethos::resident_scoring_novelty_v1::NeoResidentScoringNoveltyAllocationReceiptV1*
        allocation,
    neoethos::resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1** run) {
  using namespace neoethos::resident_scoring_novelty_v1;
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (plan == nullptr || allocation == nullptr || run == nullptr ||
      *run != nullptr || session->resident_scoring_run_v2 != nullptr ||
      session->resident_generation_run_v2 == nullptr ||
      strict_population_work_blocks_host_boundary_v1(session) ||
      plan->logical_population_count !=
          static_cast<std::uint64_t>(session->resident_planned_population_v2) ||
      plan->feature_count != static_cast<std::uint64_t>(session->feature_count)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  if (session->scoring_ready_event_v2 == nullptr) {
    cudaEvent_t attempted_scoring_ready_event = nullptr;
    if (cudaEventCreateWithFlags(&attempted_scoring_ready_event,
                                 cudaEventDisableTiming) != cudaSuccess) {
      session->strict_execution_state =
          PopulationStrictExecutionStateV1::Poisoned;
      return NEO_POPULATION_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN;
    }
    session->scoring_ready_event_v2 = attempted_scoring_ready_event;
  }
  NeoResidentScoringAdmissionV2 admission{};
  admission.abi_version = 2u;
  admission.selected_cuda_ordinal = static_cast<std::uint32_t>(session->device);
  admission.admitted_run_stream = session->stream;
  admission.scoring_novelty_ready_event = session->scoring_ready_event_v2;
  admission.full_discovery_reserve_bytes = 0;
  std::memcpy(admission.cuda_device_identity_sha256,
              plan->cuda_device_identity_sha256, 32);
  std::memcpy(admission.primary_context_identity_sha256,
              plan->primary_context_identity_sha256, 32);
  std::memcpy(admission.run_stream_identity_sha256,
              plan->run_stream_identity_sha256, 32);
  std::int32_t status =
      query_resident_scoring_admission_v2(&admission, plan, allocation);
  if (status != NEO_SCORING_STATUS_OK_V1) {
    return status;
  }
  status = create_unbound_resident_scoring_run_v2(
      &admission, plan, allocation, run);
  if (status != NEO_SCORING_STATUS_OK_V1) {
    if (*run != nullptr) {
      // Defensive only: creators must return null on failure. If an older ABI
      // violates that rule, retain the opaque tombstone and forbid any retry.
      session->resident_scoring_run_v2 = *run;
      session->strict_execution_state =
          PopulationStrictExecutionStateV1::Poisoned;
      return NEO_POPULATION_STATUS_LAUNCH_FAILED;
    }
    if (status == NEO_SCORING_STATUS_ASYNC_FREE_OUTCOME_UNKNOWN_V2 ||
        status ==
            NEO_SCORING_STATUS_ASYNC_ALLOCATION_OUTCOME_UNKNOWN_V2) {
      session->strict_execution_state =
          PopulationStrictExecutionStateV1::Poisoned;
    }
    return population_status_from_scoring_v2(status);
  }
  session->resident_scoring_run_v2 = *run;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t
neoethos_gpu_cuda_population_release_resident_scoring_run_v2(
    NeoCudaPopulationSession* session,
    neoethos::resident_scoring_novelty_v1::NeoResidentScoringNoveltyRunV1* run) {
  using namespace neoethos::resident_scoring_novelty_v1;
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (run == nullptr || session->resident_scoring_run_v2 != run ||
      strict_population_work_blocks_host_boundary_v1(session)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  const std::int32_t status = enqueue_resident_scoring_release_v2(run);
  if (status == NEO_SCORING_STATUS_OK_V1) {
    session->resident_scoring_run_v2 = nullptr;
  } else {
    session->strict_execution_state =
        PopulationStrictExecutionStateV1::Poisoned;
  }
  return population_status_from_scoring_v2(status);
}

extern "C" std::int32_t
neoethos_gpu_cuda_population_export_resident_scoring_source_v2(
    NeoCudaPopulationSession* session,
    const NeoPopulationResidentMetricsHandleV1* resident_metrics,
    std::uint64_t expected_population,
    std::uint64_t expected_feature_count,
    std::uint32_t expected_max_terms,
    neoethos::resident_search_generation_v2::
        NeoResidentScoringPopulationSourceV2* source) {
  using namespace neoethos::resident_search_generation_v2;
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (resident_metrics == nullptr || source == nullptr ||
      session->strict_execution_state !=
          PopulationStrictExecutionStateV1::InFlight ||
      session->strict_receipt_token != resident_metrics ||
      session->workspace_mode != PopulationWorkspaceModeV1::StrictMetricsOnly ||
      session->resident_generation_run_v2 == nullptr ||
      session->resident_scoring_run_v2 == nullptr ||
      session->stream == nullptr || session->event == nullptr ||
      session->scoring_ready_event_v2 == nullptr ||
      session->metric_rows == nullptr || session->scenario_ids == nullptr ||
      expected_population == 0 || expected_population > INT_MAX ||
      expected_population !=
          static_cast<std::uint64_t>(session->workspace_scenarios) ||
      expected_population !=
          static_cast<std::uint64_t>(session->scenario_count) ||
      expected_population !=
          static_cast<std::uint64_t>(session->resident_planned_population_v2) ||
      expected_feature_count !=
          static_cast<std::uint64_t>(session->feature_count) ||
      expected_max_terms == 0 || expected_max_terms > expected_feature_count ||
      session->allocator_context_reserve_bytes_v3 == 0ull ||
      resident_metrics->scenario_count != expected_population ||
      resident_metrics->event_id != session->pending_event_id) {
    session->strict_execution_state =
        PopulationStrictExecutionStateV1::Poisoned;
    return NEO_POPULATION_STATUS_WORKSPACE_PLAN_MISMATCH;
  }
  std::memset(source, 0, sizeof(*source));
  source->abi_version = NEO_RESIDENT_SEARCH_GENERATION_ABI_V2;
  source->selected_cuda_ordinal = static_cast<std::uint32_t>(session->device);
  source->admitted_run_stream = session->stream;
  source->metrics_ready_event = session->event;
  source->scoring_ready_event = session->scoring_ready_event_v2;
  source->receipt_token = resident_metrics;
  source->population_lifetime_owner = session;
  source->metric_rows_device = session->metric_rows;
  static_assert(CHAR_BIT == 8);
  static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t));
  static_assert(alignof(unsigned long long) == alignof(std::uint64_t));
  static_assert(ULLONG_MAX == UINT64_MAX);
  // The allocation is raw CUDA storage written as unsigned 64-bit scenario
  // identities. Linux names the two equal representations differently, so the
  // one authority cast lives only at this private ABI bridge.
  source->expected_scenario_ids_device =
      reinterpret_cast<const std::uint64_t*>(session->scenario_ids);
  source->logical_population_count = expected_population;
  source->feature_count = expected_feature_count;
  source->max_terms_per_gene = expected_max_terms;
  source->full_discovery_reserve_bytes =
      session->allocator_context_reserve_bytes_v3;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t
neoethos_gpu_cuda_population_finish_resident_scoring_source_v2(
    NeoCudaPopulationSession* session,
    const NeoPopulationResidentMetricsHandleV1* resident_metrics) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (resident_metrics == nullptr ||
      session->strict_execution_state !=
          PopulationStrictExecutionStateV1::InFlight ||
      session->strict_receipt_token != resident_metrics ||
      resident_metrics->event_id != session->pending_event_id) {
    session->strict_execution_state =
        PopulationStrictExecutionStateV1::Poisoned;
    return NEO_POPULATION_STATUS_WORKSPACE_PLAN_MISMATCH;
  }
  session->strict_receipt_token = nullptr;
  session->pending_event_id = 0;
  session->metrics_ready = false;
  session->strict_execution_state = PopulationStrictExecutionStateV1::StrictIdle;
  return NEO_POPULATION_STATUS_OK;
}

namespace {

enum class PopulationEvaluationModeV1 : std::uint32_t {
  CompatibilityDeviceParity = 0u,
  StrictMetricsOnly = 1u,
};

constexpr std::uint64_t kPopulationScenarioDeviceBytesV1 =
    4ull * sizeof(unsigned long long) + 2ull * sizeof(unsigned int) + 2ull * sizeof(int) +
    sizeof(long long);
static_assert(kPopulationScenarioDeviceBytesV1 == 56ull);

struct MetricsOnlyBytePlanV1 {
  std::uint64_t metric_rows_bytes = 0ull;
  std::uint64_t monthly_pnls_bytes = 0ull;
  std::uint64_t month_start_equities_bytes = 0ull;
  std::uint64_t scenario_descriptor_bytes = 0ull;
  std::uint64_t total_device_bytes = 0ull;
};

bool checked_mul_u64(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t* product) {
  if (product == nullptr || (rhs != 0ull && lhs > UINT64_MAX / rhs)) {
    return false;
  }
  *product = lhs * rhs;
  return true;
}

bool metrics_only_byte_plan_v1(int scenario_count,
                               int month_capacity,
                               MetricsOnlyBytePlanV1* plan) {
  if (plan == nullptr || scenario_count <= 0 || month_capacity <= 0) {
    return false;
  }
  const auto scenarios = static_cast<std::uint64_t>(scenario_count);
  const auto months = static_cast<std::uint64_t>(month_capacity);
  std::uint64_t monthly_elements = 0ull;
  if (!checked_mul_u64(scenarios, months, &monthly_elements) ||
      !checked_mul_u64(scenarios, sizeof(NeoPopulationMetricRow), &plan->metric_rows_bytes) ||
      !checked_mul_u64(monthly_elements, sizeof(double), &plan->monthly_pnls_bytes) ||
      !checked_mul_u64(monthly_elements, sizeof(double), &plan->month_start_equities_bytes) ||
      !checked_mul_u64(scenarios, kPopulationScenarioDeviceBytesV1,
                       &plan->scenario_descriptor_bytes)) {
    return false;
  }
  plan->total_device_bytes = 0ull;
  return checked_add_u64(plan->total_device_bytes, plan->metric_rows_bytes) &&
         checked_add_u64(plan->total_device_bytes, plan->monthly_pnls_bytes) &&
         checked_add_u64(plan->total_device_bytes, plan->month_start_equities_bytes) &&
         checked_add_u64(plan->total_device_bytes, plan->scenario_descriptor_bytes);
}

std::int32_t ensure_compatibility_workspace_v1(NeoCudaPopulationSession* session,
                                               int scenario_count,
                                               int month_capacity,
                                               int bars) {
  if (session->workspace_mode == PopulationWorkspaceModeV1::StrictMetricsOnly) {
    return NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH;
  }
  if (session->workspace_mode == PopulationWorkspaceModeV1::Uninitialized) {
    session->workspace_mode = PopulationWorkspaceModeV1::CompatibilityDeviceParityOnly;
  }
  if (session->workspace_mode != PopulationWorkspaceModeV1::CompatibilityDeviceParityOnly) {
    return NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH;
  }

  const auto scenarios = static_cast<std::size_t>(scenario_count);
  const auto months = static_cast<std::size_t>(month_capacity);
  if (scenarios > SIZE_MAX / static_cast<std::size_t>(kMaxTradesPerCandidate) ||
      scenarios > SIZE_MAX / months) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  const bool incomplete = session->outcomes == nullptr || session->monthly_pnls == nullptr ||
                          session->month_start_equities == nullptr ||
                          session->metric_rows == nullptr ||
                          session->accepted_trade_total == nullptr;
  if (incomplete || session->month_capacity != month_capacity ||
      session->workspace_scenarios < scenario_count) {
    session->release_workspace();
    std::int32_t status = NEO_POPULATION_STATUS_OK;
    const auto guard = [&](std::int32_t code) {
      if (code != NEO_POPULATION_STATUS_OK) {
        status = code;
      }
      return status == NEO_POPULATION_STATUS_OK;
    };
    if (!guard(device_alloc(&session->outcomes,
                            scenarios * static_cast<std::size_t>(kMaxTradesPerCandidate))) ||
        !guard(device_alloc(&session->monthly_pnls, scenarios * months)) ||
        !guard(device_alloc(&session->month_start_equities, scenarios * months)) ||
        !guard(device_alloc(&session->metric_rows, scenarios)) ||
        !guard(device_alloc(&session->accepted_trade_total, 1))) {
      session->release_workspace();
      return status;
    }
    session->month_capacity = month_capacity;
    session->workspace_scenarios = scenario_count;
    session->workspace_bars = bars;
  }
  return NEO_POPULATION_STATUS_OK;
}

std::int32_t ensure_metrics_only_workspace_v1(NeoCudaPopulationSession* session,
                                              int scenario_count,
                                              int month_capacity,
                                              int bars) {
  if (session->workspace_mode == PopulationWorkspaceModeV1::CompatibilityDeviceParityOnly) {
    return NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH;
  }
  if (session->workspace_mode == PopulationWorkspaceModeV1::Uninitialized) {
    session->workspace_mode = PopulationWorkspaceModeV1::StrictMetricsOnly;
  }
  if (session->workspace_mode != PopulationWorkspaceModeV1::StrictMetricsOnly) {
    return NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH;
  }

  if (session->metric_rows != nullptr) {
    if (session->workspace_scenarios != scenario_count ||
        session->month_capacity != month_capacity) {
      return NEO_POPULATION_STATUS_WORKSPACE_PLAN_MISMATCH;
    }
    if (session->monthly_pnls == nullptr || session->month_start_equities == nullptr ||
        session->outcomes != nullptr || session->accepted_trade_total != nullptr) {
      return NEO_POPULATION_STATUS_WORKSPACE_PLAN_MISMATCH;
    }
    return NEO_POPULATION_STATUS_OK;
  }
  if (session->monthly_pnls != nullptr || session->month_start_equities != nullptr ||
      session->outcomes != nullptr || session->accepted_trade_total != nullptr ||
      session->workspace_scenarios != 0 || session->month_capacity != 0) {
    return NEO_POPULATION_STATUS_WORKSPACE_PLAN_MISMATCH;
  }

  const auto scenarios = static_cast<std::size_t>(scenario_count);
  const auto months = static_cast<std::size_t>(month_capacity);
  if (scenarios > SIZE_MAX / months) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  std::int32_t status = NEO_POPULATION_STATUS_OK;
  const auto guard = [&](std::int32_t code) {
    if (code != NEO_POPULATION_STATUS_OK) {
      status = code;
    }
    return status == NEO_POPULATION_STATUS_OK;
  };
  if (!guard(device_alloc(&session->monthly_pnls, scenarios * months)) ||
      !guard(device_alloc(&session->month_start_equities, scenarios * months)) ||
      !guard(device_alloc(&session->metric_rows, scenarios))) {
    session->release_workspace();
    return status;
  }
  session->month_capacity = month_capacity;
  session->workspace_scenarios = scenario_count;
  session->workspace_bars = bars;
  return NEO_POPULATION_STATUS_OK;
}

std::int32_t enqueue_population_evaluation_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationSettings* settings,
    PopulationEvaluationModeV1 mode,
    NeoPopulationResidentMetricsHandleV1* resident_metrics,
    std::uint64_t* compatibility_event_id,
    NeoPopulationCounters* counters) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
  }
  if (settings == nullptr ||
      (mode == PopulationEvaluationModeV1::StrictMetricsOnly && resident_metrics == nullptr) ||
      (mode == PopulationEvaluationModeV1::CompatibilityDeviceParity &&
       compatibility_event_id == nullptr)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (!session->has_dataset || !session->has_genes || !session->has_scenarios) {
    return NEO_POPULATION_STATUS_MISSING_UPLOAD;
  }
  if (settings->abi_version != NEOETHOS_GPU_ABI_VERSION) {
    return NEO_POPULATION_STATUS_ABI_MISMATCH;
  }
  if (settings->month_capacity == 0u || settings->month_capacity > INT_MAX) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }

  const int bars = session->bars;
  // What the launch is SIZED by. Equal to `population` for every caller that
  // uploads one scenario per gene, which is what makes that case identical to
  // the pre-scenario engine; larger for a quality screen that asks for many
  // treatments of the same genes.
  const int scenario_count = session->scenario_count;
  const int month_capacity = static_cast<int>(settings->month_capacity);

  DeviceDataset dataset;
  dataset.close = session->close;
  dataset.high = session->high;
  dataset.low = session->low;
  dataset.indicators_bar_major = session->indicators_bar_major;
  dataset.indicators_feature_major = session->indicators_feature_major;
  dataset.months = session->months;
  dataset.days = session->days;
  dataset.timestamps = session->timestamps;
  dataset.smc_rows = session->smc_rows;
  dataset.view_indices = session->view_indices;
  dataset.adaptive_base_pips = session->adaptive_base_pips;
  dataset.has_adaptive_base = session->has_adaptive_base;
  dataset.bars = bars;
  dataset.parent_rows = session->parent_rows;
  dataset.feature_count = session->feature_count;
  dataset.view_kind = session->view_kind;
  dataset.view_start = session->view_start;
  dataset.timestamp_mode = session->timestamp_mode;

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

  DeviceScenarios scenario_view;
  scenario_view.base_candidate_ids = session->scenario_base_candidate_ids;
  scenario_view.ids = session->scenario_ids;
  scenario_view.rng_counters = session->scenario_rng_counters;
  scenario_view.window_offsets = session->scenario_window_offsets;
  scenario_view.window_lens = session->scenario_window_lens;
  scenario_view.types = session->scenario_types;
  scenario_view.spread_ticks = session->scenario_spread_ticks;
  scenario_view.slippage_ticks = session->scenario_slippage_ticks;
  scenario_view.commission_micros = session->scenario_commission_micros;
  scenario_view.count = scenario_count;

  const unsigned int gap_blocks = static_cast<unsigned int>((bars + 255) / 256);
  const int reduce_block = choose_reduce_block(scenario_count, session->sm_count);
  const unsigned int reduce_blocks =
      static_cast<unsigned int>((scenario_count + reduce_block - 1) / reduce_block);
  unsigned long long trade_slots = 0ull;
  MetricsOnlyBytePlanV1 resident_plan;

  if (mode == PopulationEvaluationModeV1::StrictMetricsOnly) {
    const std::int32_t status =
        ensure_metrics_only_workspace_v1(session, scenario_count, month_capacity, bars);
    if (status != NEO_POPULATION_STATUS_OK) {
      return status;
    }
    if (!metrics_only_byte_plan_v1(session->workspace_scenarios, session->month_capacity,
                                   &resident_plan)) {
      return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
    }
    if (resident_plan.scenario_descriptor_bytes != session->scenario_upload_bytes) {
      return NEO_POPULATION_STATUS_WORKSPACE_PLAN_MISMATCH;
    }
    session->strict_execution_state = PopulationStrictExecutionStateV1::InFlight;
    population_gap_flags_kernel<<<gap_blocks == 0u ? 1u : gap_blocks, 256, 0,
                                  session->stream>>>(dataset, *settings, session->gap_flags);
    session->kernel_submissions += 1;
    population_reduce_kernel<<<reduce_blocks == 0u ? 1u : reduce_blocks, reduce_block, 0,
                               session->stream>>>(dataset, genes, scenario_view, *settings,
                                                  session->gap_flags, nullptr,
                                                  session->monthly_pnls,
                                                  session->month_start_equities,
                                                  session->metric_rows, nullptr);
    session->kernel_submissions += 1;
  } else {
    if (session->workspace_mode == PopulationWorkspaceModeV1::StrictMetricsOnly) {
      return NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH;
    }
    const std::int32_t status =
        ensure_compatibility_workspace_v1(session, scenario_count, month_capacity, bars);
    if (status != NEO_POPULATION_STATUS_OK) {
      return status;
    }
    if (cudaMemsetAsync(session->accepted_trade_total, 0, sizeof(unsigned long long),
                        session->stream) != cudaSuccess) {
      return NEO_POPULATION_STATUS_TRANSFER_FAILED;
    }
    population_gap_flags_kernel<<<gap_blocks == 0u ? 1u : gap_blocks, 256, 0,
                                  session->stream>>>(dataset, *settings, session->gap_flags);
    session->kernel_submissions += 1;
    trade_slots = static_cast<unsigned long long>(scenario_count) * kMaxTradesPerCandidate;
    const unsigned int seed_threads = 256u;
    const unsigned long long seed_blocks = (trade_slots + seed_threads - 1ull) / seed_threads;
    population_seed_outcomes_kernel<<<static_cast<unsigned int>(seed_blocks), seed_threads, 0,
                                      session->stream>>>(session->outcomes, trade_slots);
    session->kernel_submissions += 1;
    population_reduce_kernel<<<reduce_blocks == 0u ? 1u : reduce_blocks, reduce_block, 0,
                               session->stream>>>(dataset, genes, scenario_view, *settings,
                                                  session->gap_flags, session->outcomes,
                                                  session->monthly_pnls,
                                                  session->month_start_equities,
                                                  session->metric_rows,
                                                  session->accepted_trade_total);
    session->kernel_submissions += 1;
  }

  if (cudaEventRecord(session->event, session->stream) != cudaSuccess) {
    if (mode == PopulationEvaluationModeV1::StrictMetricsOnly) {
      session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;
    }
    return NEO_POPULATION_STATUS_LAUNCH_FAILED;
  }
  if (cudaGetLastError() != cudaSuccess) {
    if (mode == PopulationEvaluationModeV1::StrictMetricsOnly) {
      session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;
    }
    return NEO_POPULATION_STATUS_LAUNCH_FAILED;
  }

  // Not a count of anything emitted: this is the extent of the outcome array,
  // and it is what bounds `read_diagnostics`. Nothing may read it as "how full
  // the card got" — every candidate is given the same slice whether it trades
  // once or fills it. The host-side percentage that used to be logged from it
  // measured a reservation against a phantom budget and is gone.
  session->emitted_events = trade_slots;
  session->pending_event_id = session->next_event_id;
  session->next_event_id += 1ull;
  session->metrics_ready = false;

  if (mode == PopulationEvaluationModeV1::StrictMetricsOnly) {
    std::memset(resident_metrics, 0, sizeof(NeoPopulationResidentMetricsHandleV1));
    resident_metrics->abi_version = NEOETHOS_GPU_ABI_VERSION;
    resident_metrics->event_id = session->pending_event_id;
    resident_metrics->scenario_count = static_cast<std::uint64_t>(session->workspace_scenarios);
    resident_metrics->month_capacity = static_cast<std::uint64_t>(session->month_capacity);
    resident_metrics->metric_rows_bytes = resident_plan.metric_rows_bytes;
    resident_metrics->monthly_pnls_bytes = resident_plan.monthly_pnls_bytes;
    resident_metrics->month_start_equities_bytes = resident_plan.month_start_equities_bytes;
    resident_metrics->scenario_descriptor_bytes = resident_plan.scenario_descriptor_bytes;
    resident_metrics->total_device_bytes = resident_plan.total_device_bytes;
    resident_metrics->outcome_bytes = 0ull;
    resident_metrics->accepted_trade_total_bytes = 0ull;
  } else {
    *compatibility_event_id = session->pending_event_id;
  }

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

}  // namespace

extern "C" std::int32_t neoethos_gpu_cuda_population_b_enqueue_metrics_only_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationSettings* settings,
    NeoPopulationResidentMetricsHandleV1* resident_metrics,
    NeoPopulationCounters* counters) {
  return enqueue_population_evaluation_v1(session, settings,
                                          PopulationEvaluationModeV1::StrictMetricsOnly,
                                          resident_metrics, nullptr, counters);
}

extern "C" std::int32_t neoethos_gpu_cuda_population_consume_terminal_compact_result_v1(
    NeoCudaPopulationSession* session,
    const NeoPopulationResidentMetricsHandleV1* resident_metrics,
    NeoPopulationTerminalCompactResultV1* compact_result) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  const auto poison = [&](std::int32_t status) {
    session->strict_execution_state = PopulationStrictExecutionStateV1::Poisoned;
    return status;
  };
  if (session->strict_execution_state != PopulationStrictExecutionStateV1::InFlight) {
    return session->strict_execution_state == PopulationStrictExecutionStateV1::Poisoned
               ? NEO_POPULATION_STATUS_STRICT_RESIDENT_POISONED
               : NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (resident_metrics == nullptr || compact_result == nullptr) {
    return poison(NEO_POPULATION_STATUS_INVALID_ARGUMENT);
  }
  MetricsOnlyBytePlanV1 expected_plan;
  if (session->workspace_mode != PopulationWorkspaceModeV1::StrictMetricsOnly ||
      session->workspace_scenarios != 1 || session->scenario_count != 1 ||
      session->month_capacity <= 0 || session->metric_rows == nullptr ||
      session->monthly_pnls == nullptr || session->month_start_equities == nullptr ||
      session->outcomes != nullptr || session->accepted_trade_total != nullptr ||
      session->event == nullptr ||
      !metrics_only_byte_plan_v1(1, session->month_capacity, &expected_plan) ||
      resident_metrics->abi_version != NEOETHOS_GPU_ABI_VERSION ||
      resident_metrics->reserved != 0u || resident_metrics->event_id == 0ull ||
      resident_metrics->event_id != session->pending_event_id ||
      resident_metrics->scenario_count != 1ull ||
      resident_metrics->month_capacity !=
          static_cast<std::uint64_t>(session->month_capacity) ||
      resident_metrics->metric_rows_bytes != sizeof(NeoPopulationMetricRow) ||
      resident_metrics->metric_rows_bytes != expected_plan.metric_rows_bytes ||
      resident_metrics->monthly_pnls_bytes != expected_plan.monthly_pnls_bytes ||
      resident_metrics->month_start_equities_bytes !=
          expected_plan.month_start_equities_bytes ||
      resident_metrics->scenario_descriptor_bytes !=
          expected_plan.scenario_descriptor_bytes ||
      resident_metrics->total_device_bytes != expected_plan.total_device_bytes ||
      resident_metrics->outcome_bytes != 0ull ||
      resident_metrics->accepted_trade_total_bytes != 0ull) {
    return poison(NEO_POPULATION_STATUS_WORKSPACE_PLAN_MISMATCH);
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return poison(NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE);
  }
  if (cudaEventSynchronize(session->event) != cudaSuccess) {
    return poison(NEO_POPULATION_STATUS_SYNC_FAILED);
  }
  if (cudaGetLastError() != cudaSuccess) {
    return poison(NEO_POPULATION_STATUS_LAUNCH_FAILED);
  }
  std::memset(compact_result, 0, sizeof(NeoPopulationTerminalCompactResultV1));
  if (cudaMemcpy(&compact_result->metric_row, session->metric_rows,
                 sizeof(NeoPopulationMetricRow), cudaMemcpyDeviceToHost) != cudaSuccess) {
    return poison(NEO_POPULATION_STATUS_TRANSFER_FAILED);
  }
  compact_result->abi_version = NEOETHOS_GPU_ABI_VERSION;
  compact_result->event_id = resident_metrics->event_id;
  compact_result->scenario_count = 1ull;
  compact_result->terminal_synchronization_count = 1ull;
  compact_result->terminal_readback_count = 1ull;
  compact_result->terminal_readback_rows = 1ull;
  compact_result->terminal_readback_bytes = sizeof(NeoPopulationMetricRow);
  session->metrics_ready = false;
  session->pending_event_id = 0ull;
  session->strict_execution_state = PopulationStrictExecutionStateV1::StrictIdle;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_b_evaluate(
    NeoCudaPopulationSession* session,
    const NeoPopulationSettings* settings,
    std::uint64_t* event_id,
    NeoPopulationCounters* counters) {
  return enqueue_population_evaluation_v1(session, settings,
                                          PopulationEvaluationModeV1::CompatibilityDeviceParity,
                                          nullptr, event_id, counters);
}

extern "C" std::int32_t neoethos_gpu_cuda_population_wait(NeoCudaPopulationSession* session,
                                                           std::uint64_t event_id) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
  }
  if (session->workspace_mode != PopulationWorkspaceModeV1::CompatibilityDeviceParityOnly) {
    return NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH;
  }
  if (event_id == 0ull || event_id != session->pending_event_id) {
    return NEO_POPULATION_STATUS_UNKNOWN_EVENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  constexpr std::uint64_t accepted_readback_bytes = sizeof(unsigned long long);
  if (session->synchronization_events == UINT64_MAX ||
      session->residency_counters.explicit_synchronization_count == UINT64_MAX ||
      session->residency_counters.accepted_trade_total_readback_count == UINT64_MAX ||
      accepted_readback_bytes >
          UINT64_MAX - session->residency_counters.accepted_trade_total_readback_bytes) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaEventSynchronize(session->event) != cudaSuccess) {
    return NEO_POPULATION_STATUS_SYNC_FAILED;
  }
  if (cudaGetLastError() != cudaSuccess) {
    return NEO_POPULATION_STATUS_LAUNCH_FAILED;
  }
  // P1-C accounts for this per-evaluation scalar D2H exactly. It is not a
  // final compact result: P1-E must remove this host barrier so resident
  // evaluation can flow directly into device-side rank/select/evolution.
  unsigned long long accepted = 0ull;
  if (cudaMemcpy(&accepted, session->accepted_trade_total, sizeof(unsigned long long),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return NEO_POPULATION_STATUS_TRANSFER_FAILED;
  }
  session->accepted_trades = accepted;
  session->synchronization_events += 1;
  session->residency_counters.explicit_synchronization_count += 1ull;
  session->residency_counters.accepted_trade_total_readback_count += 1ull;
  session->residency_counters.accepted_trade_total_readback_bytes +=
      accepted_readback_bytes;
  session->metrics_ready = true;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_read_metrics(
    NeoCudaPopulationSession* session,
    NeoPopulationReadback* readback) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
  }
  if (session->workspace_mode != PopulationWorkspaceModeV1::CompatibilityDeviceParityOnly) {
    return NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH;
  }
  if (readback == nullptr || readback->rows == nullptr || readback->written == nullptr) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (!session->metrics_ready) {
    return NEO_POPULATION_STATUS_MISSING_UPLOAD;
  }
  // One row per SCENARIO, not per gene. This is an intermediate full-population
  // D2H boundary, not a compact-final readback: 17 574 rows for 174 genes in
  // the quality screen.
  const std::size_t rows = static_cast<std::size_t>(session->scenario_count);
  if (readback->capacity < rows) {
    return NEO_POPULATION_STATUS_READBACK_CAPACITY;
  }
  if (rows > SIZE_MAX / sizeof(NeoPopulationMetricRow)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t bytes = rows * sizeof(NeoPopulationMetricRow);
  const auto rows_u64 = static_cast<std::uint64_t>(rows);
  const auto bytes_u64 = static_cast<std::uint64_t>(bytes);
  if (static_cast<std::size_t>(rows_u64) != rows ||
      static_cast<std::size_t>(bytes_u64) != bytes ||
      session->residency_counters.metric_rows_readback_count == UINT64_MAX ||
      rows_u64 > UINT64_MAX - session->residency_counters.metric_rows_readback_rows ||
      bytes_u64 > UINT64_MAX - session->residency_counters.metric_rows_readback_bytes) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  if (cudaMemcpy(readback->rows, session->metric_rows, bytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    return NEO_POPULATION_STATUS_TRANSFER_FAILED;
  }
  *readback->written = rows;
  session->residency_counters.metric_rows_readback_count += 1ull;
  session->residency_counters.metric_rows_readback_rows += rows_u64;
  session->residency_counters.metric_rows_readback_bytes += bytes_u64;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" std::int32_t neoethos_gpu_cuda_population_read_diagnostics(
    NeoCudaPopulationSession* session,
    NeoPopulationDiagnosticReadback* readback) {
  if (session == nullptr) {
    return NEO_POPULATION_STATUS_NULL_SESSION;
  }
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    return strict_population_host_boundary_status_v1(session);
  }
  if (session->workspace_mode != PopulationWorkspaceModeV1::CompatibilityDeviceParityOnly) {
    return NEO_POPULATION_STATUS_WORKSPACE_MODE_MISMATCH;
  }
  if (readback == nullptr || readback->outcomes == nullptr || readback->written == nullptr) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (!session->metrics_ready) {
    return NEO_POPULATION_STATUS_MISSING_UPLOAD;
  }
  // A PREFIX is a well-defined answer here, so a smaller buffer is a range
  // request rather than an error. The outcome array is scenario-major with
  // `kMaxTradesPerCandidate` slots each, so the first `capacity` records are
  // exactly the trades of the first `capacity / kMaxTradesPerCandidate`
  // scenarios. Refusing instead forced the host to allocate for the WHOLE
  // array — at ~20 000 scenarios that is 163.8 M records, ~21 GB across the two
  // vectors, on the rented box where a parity failure is investigated.
  const std::size_t available = static_cast<std::size_t>(session->emitted_events);
  const std::size_t count = readback->capacity < available ? readback->capacity : available;
  if (count > SIZE_MAX / sizeof(NeoPopulationOutcome)) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t bytes = count * sizeof(NeoPopulationOutcome);
  const auto count_u64 = static_cast<std::uint64_t>(count);
  const auto bytes_u64 = static_cast<std::uint64_t>(bytes);
  if (static_cast<std::size_t>(count_u64) != count ||
      static_cast<std::size_t>(bytes_u64) != bytes ||
      session->residency_counters.diagnostic_readback_count == UINT64_MAX ||
      count_u64 > UINT64_MAX - session->residency_counters.diagnostic_readback_rows ||
      bytes_u64 > UINT64_MAX - session->residency_counters.diagnostic_readback_bytes) {
    return NEO_POPULATION_STATUS_INVALID_ARGUMENT;
  }
  if (cudaSetDevice(session->device) != cudaSuccess) {
    return NEO_POPULATION_STATUS_DEVICE_UNAVAILABLE;
  }
  if (count > 0) {
    // There is no event stream any more — the reduce opens positions from the
    // signal — so the events half is zeroed rather than copied from a buffer
    // that is not allocated. A caller that passes NULL is saying it knows that
    // and does not want 56 B per slot of zeros; callers that want entries read
    // them from the outcomes, which carry `entry_bar` for exactly this reason.
    if (readback->events != nullptr) {
      std::memset(readback->events, 0, count * sizeof(NeoPopulationEvent));
    }
    if (cudaMemcpy(readback->outcomes, session->outcomes, bytes, cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      return NEO_POPULATION_STATUS_TRANSFER_FAILED;
    }
  }
  *readback->written = count;
  session->residency_counters.diagnostic_readback_count += 1ull;
  session->residency_counters.diagnostic_readback_rows += count_u64;
  session->residency_counters.diagnostic_readback_bytes += bytes_u64;
  return NEO_POPULATION_STATUS_OK;
}

extern "C" void neoethos_gpu_cuda_population_destroy(NeoCudaPopulationSession* session) {
  if (session == nullptr) {
    return;
  }
  if (strict_population_work_blocks_host_boundary_v1(session)) {
    // Leak-only fail-closed teardown. `release()` uses synchronous CUDA frees;
    // those cannot run until a future resident stage consumes the event on the
    // same stream and atomically clears strict execution state.
    return;
  }
  cudaSetDevice(session->device);
  session->release();
  delete session;
}
