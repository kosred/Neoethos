// Persistent native-CUDA Prototype B population engine.
//
// One session owns one non-default stream, one logical dataset upload and every
// device workspace. `evaluate` runs the complete canonical chain on that
// stream. The host boundary is the compact metric readback; no
// candidate-dependent work returns to the CPU.
//
// The chain used to be five kernels passing `population * bars` arrays between
// them. It is now two: a bar-parallel gap-flag pass, and ONE walk kernel that
// synthesises the signal, opens and closes the position, and reduces to
// metrics — all in registers, one thread per scenario. Nothing candidate-sized
// crosses a kernel boundary any more, which took the per-scenario device cost
// from 4 811 048 B to 593 768 B and let the population stop being bounded by
// memory at all.
//
// The indicator matrix is uploaded feature-major (the host contract) and
// transposed once, on the device, into bar-major (what the walk reads).
//
// The semantics reproduced here are the canonical ones expressed by the
// validation oracle in `prototype_population_oracle.rs`. Any divergence is a
// correctness failure, not a tuning opportunity.

#include <cstdio>
#include <cstdlib>
#include "neoethos_gpu_cuda.h"

#include <cuda_runtime.h>

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
                                             long long timestamp_ms) {
  if (timestamp_ms <= 0) {
    return settings.spread_pips;
  }
  const long long secs = timestamp_ms / 1000LL - (timestamp_ms % 1000LL < 0 ? 1LL : 0LL);
  long long hour = secs / 3600LL - (secs % 3600LL < 0 ? 1LL : 0LL);
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
  // of one bar a single contiguous 256-byte run that the warp shares.
  const float* indicators_bar_major;
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
/// Pure data movement: no arithmetic touches the values, so every float the
/// walk later reads is bit-identical to the one the feature-major kernel read.
/// Runs once per dataset upload, on the session stream, before anything else
/// can observe the matrix.
__global__ void transpose_indicators_to_bar_major(const float* __restrict__ source,
                                                  float* __restrict__ destination,
                                                  int bars,
                                                  int features) {
  __shared__ float tile[kTransposeTile][kTransposeTile + 1];

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
//   * the CSR terms accumulate in the same ascending order into the same f32
//     accumulator — no reassociation, no tree reduction, no lane splitting;
//   * `-fmad=false` (build.rs:92, with its own measured justification) forbids
//     the compiler from contracting `weights[t] * indicator + acc` into an FMA,
//     which is the only way the same source order could still produce a
//     different bit pattern;
//   * the threshold comparisons, the confidence clamp, the SMC active sum and
//     the SMC score loop are transcribed term for term, slot order included;
//   * bar-major is a transpose of the input, so the float VALUE at every
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
  float long_threshold;
  float short_threshold;
  /// `fabsf(long - short)`, clamped exactly as the original did.
  float gap;
  /// Sum of the weights of this candidate's active SMC slots, ascending.
  float active_sum;
  /// `fminf(gate_threshold, active_sum)`.
  float gate;
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
    // Formed in double and narrowed once, exactly as the Rust mirror does:
    // both lanes hold the parameter in f32 and the factor in f64.
    plan.long_threshold = static_cast<float>(
        static_cast<double>(plan.long_threshold) *
        perturb_factor(counter, kDrawLongThreshold, kMcThresholdAmplitude));
    plan.short_threshold = static_cast<float>(
        static_cast<double>(plan.short_threshold) *
        perturb_factor(counter, kDrawShortThreshold, kMcThresholdAmplitude));
  }

  float gap = fabsf(plan.long_threshold - plan.short_threshold);
  if (!(gap > 1.0e-6f)) {
    gap = 1.0e-6f;
  }
  plan.gap = gap;

  // Same ascending slot order and the same f32 accumulator as the kernel this
  // replaces; only the number of times it runs changed.
  unsigned mask = 0u;
  float active_sum = 0.0f;
  for (int slot = 0; slot < kSmcSlots; ++slot) {
    if (genes.smc_flags[static_cast<long long>(candidate) * kSmcSlots + slot] != 0) {
      mask |= (1u << slot);
      active_sum += genes.smc_weights[slot];
    }
  }
  if (genes.smc_gate_disabled != 0) {
    active_sum = 0.0f;
  }
  plan.smc_mask = mask;
  plan.active_sum = active_sum;
  plan.gate = fminf(genes.gate_threshold, active_sum);
  return plan;
}

/// The per-bar half. Returns the emitted direction (-1, 0, +1) and writes the
/// confidence, exactly as the pair of device arrays used to carry them.
__device__ inline signed char synthesize_signal(const DeviceDataset& dataset,
                                                const DeviceGenes& genes,
                                                const SignalPlan& plan,
                                                int bar,
                                                float* confidence_out) {
  // Terms accumulate in ascending CSR order, matching the canonical f32
  // accumulation order bit for bit.
  float combined = 0.0f;
  const long long feature_row = static_cast<long long>(bar) * dataset.feature_count;
  if (plan.perturbed == 0) {
    for (int term = plan.term_start; term < plan.term_end; ++term) {
      const int feature = genes.indices[term];
      combined += genes.weights[term] * dataset.indicators_bar_major[feature_row + feature];
    }
  } else {
    // The perturbed loop is written out rather than folded into the one above
    // with a conditional factor, so the unperturbed path — every existing
    // caller, and the one the 147 parity tests exercise — keeps EXACTLY the
    // instruction sequence it had. A `weight * factor` with `factor == 1.0`
    // is not the identity on a float: it forces a round trip through double
    // and back, and `-fmad=false` cannot help with a rounding that the source
    // asked for.
    //
    // Term ordinal, not term index: the draw stream is per gene, so a gene's
    // first term is always draw 2 whatever its offset into the shared CSR
    // arrays is. Using `term` directly would make a gene's perturbation depend
    // on how many terms the genes before it had.
    for (int term = plan.term_start; term < plan.term_end; ++term) {
      const int feature = genes.indices[term];
      const unsigned long long ordinal =
          static_cast<unsigned long long>(term - plan.term_start);
      const float weight = static_cast<float>(
          static_cast<double>(genes.weights[term]) *
          perturb_factor(plan.rng_counter, kDrawWeightBase + ordinal, kMcWeightAmplitude));
      combined += weight * dataset.indicators_bar_major[feature_row + feature];
    }
  }

  signed char signal = 0;
  if (combined >= plan.long_threshold) {
    signal = 1;
  } else if (combined <= plan.short_threshold) {
    signal = -1;
  }

  if (signal == 0) {
    *confidence_out = 0.0f;
    return 0;
  }

  const float margin =
      (signal == 1) ? (combined - plan.long_threshold) : (plan.short_threshold - combined);
  float confidence = fminf(fmaxf(margin / plan.gap, 0.0f), 1.0f);

  bool passes_gate = true;
  if (plan.active_sum > 0.0f) {
    float score = 0.0f;
    for (int slot = 0; slot < kSmcSlots; ++slot) {
      if (((plan.smc_mask >> slot) & 1u) == 0u) {
        continue;
      }
      const signed char row = dataset.smc_rows[static_cast<long long>(bar) * kSmcSlots + slot];
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
    *confidence_out = 0.0f;
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
      const long long previous = dataset.timestamps[bar - 1];
      const long long current = dataset.timestamps[bar];
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
                            : spread_pips_for_bar(settings, dataset.timestamps[bar]);
    const double half_spread_price = bar_spread_pips * 0.5 * pip;
    const double half_spread_cost = bar_spread_pips * 0.5 * settings.pip_value_per_lot;

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

    // ── Entry, from a signal synthesised HERE ─────────────────────────────
    //
    // This read `signal_values[signal_base + signal_bar]` — a 2.78 GB device
    // array whose only purpose was to bring this one byte from another kernel,
    // with `signal_confidences` (11.1 GB) alongside it for the sizing call
    // below. Both are gone; the value is computed in registers at the moment it
    // is used, from the same CSR terms in the same ascending f32 order.
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
    float signal_confidence_here = 0.0f;
    const signed char signal_here =
        synthesize_signal(dataset, genes, signal_plan, signal_bar, &signal_confidence_here);
    if (signal_here != 0) {
      ++cursor;
      if (settings.max_trades_per_day > 0u && day_trade_count >= settings.max_trades_per_day) {
        continue;
      }
      const int direction = signal_here > 0 ? kDirectionLong : kDirectionShort;
      double entry_price =
          dataset.close[bar] + static_cast<double>(direction) * half_spread_price;
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
        // Same widening the array read performed: the confidence is produced as
        // f32 and consumed as f64, so the f32 value and the conversion are both
        // where they were.
        lots = risk_based_position_lots(static_cast<double>(signal_confidence_here), equity,
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
  // The gene's identity and the scenario's, separately. A mixed array shares one
  // `candidate_id` across every scenario of the same gene, and the host demuxes
  // by `scenario_id` — which is why both are carried and neither is a position.
  row.candidate_id = genes.candidate_ids[candidate];
  row.scenario_id = scenario_id;
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
  rows[scenario] = row;

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
  // No `max_events`. The session used to carry the event-buffer capacity the
  // creator asked for, but the buffer it sized has not been allocated since the
  // reduce started opening positions from the signal directly. The create ABI
  // still takes the argument — see the note there — and the device no longer
  // stores it, because a field nothing reads is how a phantom budget survives.
  unsigned long long next_event_id = 1ull;
  unsigned long long pending_event_id = 0ull;
  bool has_dataset = false;
  bool has_genes = false;
  bool has_scenarios = false;
  bool metrics_ready = false;
  int bars = 0;
  int feature_count = 0;
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

  double* close = nullptr;
  double* high = nullptr;
  double* low = nullptr;
  // BAR-MAJOR, `bar * feature_count + feature`. The host still hands over a
  // feature-major matrix; `upload_dataset` transposes it once on the device and
  // frees the feature-major staging copy, so nothing feature-major stays
  // resident. Renamed from `indicators` deliberately: a reader that was not
  // updated must fail to compile rather than silently stride the wrong way.
  float* indicators_bar_major = nullptr;
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
  }

  void release() {
    release_workspace();
    device_free(close);
    device_free(high);
    device_free(low);
    device_free(indicators_bar_major);
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
    release_scenarios();
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

  auto* session = new (std::nothrow) NeoCudaPopulationSession();
  if (session == nullptr) {
    return fail(NEO_POPULATION_STATUS_ALLOCATION_FAILED);
  }
  session->device = device;
  // Read once. `choose_reduce_block` needs it on every launch and it cannot
  // change; a failed query leaves 0, which that function treats as 1 and so
  // degrades to the smallest block rather than to an undefined grid.
  {
    int sm_count = 0;
    if (cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device) ==
        cudaSuccess) {
      session->sm_count = sm_count;
    }
  }
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
  const std::size_t i64_bytes = bars * sizeof(long long);
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
  // Bar-major makes the whole feature row of a bar one contiguous 256-byte run
  // that the warp shares.
  //
  // So the feature-major bytes land in a STAGING buffer, one kernel transposes
  // them into the permanent one, and the staging buffer is freed before this
  // function returns. Steady-state resident memory is one copy, not two; the
  // transient second copy exists for the duration of one launch on a card that
  // has just been asked for the same amount again for the permanent buffer, so
  // if the transient one cannot be had, the dataset was never going to fit.
  {
    float* staging = nullptr;
    status = device_alloc(&staging, features * bars);
    if (status != NEO_POPULATION_STATUS_OK) return status;
    status = copy_to_device(staging, dataset->indicators, features * bars * sizeof(float),
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
  // Bar-major turned an out-of-range feature index from LOUD into SILENT.
  //
  // Feature-major addressed `indicators[feature * bars + bar]`, so a `feature`
  // equal to `feature_count` read past the end of a `feature_count * bars`
  // buffer — an out-of-bounds access compute-sanitizer catches immediately.
  // Bar-major addresses `indicators_bar_major[bar * feature_count + feature]`,
  // where the same index resolves to `(bar + 1) * feature_count + 0`: in
  // bounds, a perfectly plausible float belonging to the NEXT bar, and nobody
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
  auto* base_ids = new (std::nothrow) unsigned long long[count];
  auto* ids = new (std::nothrow) unsigned long long[count];
  auto* counters = new (std::nothrow) unsigned long long[count];
  auto* offsets = new (std::nothrow) unsigned long long[count];
  auto* lens = new (std::nothrow) unsigned int[count];
  auto* types = new (std::nothrow) unsigned int[count];
  auto* spreads = new (std::nothrow) int[count];
  auto* slippages = new (std::nothrow) int[count];
  auto* commissions = new (std::nothrow) long long[count];
  const auto cleanup_host = [&]() {
    delete[] base_ids;
    delete[] ids;
    delete[] counters;
    delete[] offsets;
    delete[] lens;
    delete[] types;
    delete[] spreads;
    delete[] slippages;
    delete[] commissions;
  };
  if (base_ids == nullptr || ids == nullptr || counters == nullptr || offsets == nullptr ||
      lens == nullptr || types == nullptr || spreads == nullptr || slippages == nullptr ||
      commissions == nullptr) {
    cleanup_host();
    return NEO_POPULATION_STATUS_ALLOCATION_FAILED;
  }
  for (std::size_t index = 0; index < count; ++index) {
    const NeoScenarioDescriptor& descriptor = scenarios->descriptors[index];
    base_ids[index] = descriptor.base_candidate_id;
    ids[index] = descriptor.scenario_id;
    counters[index] = descriptor.rng_counter;
    offsets[index] = descriptor.window_offset;
    lens[index] = descriptor.window_len;
    types[index] = descriptor.scenario_type;
    spreads[index] = descriptor.spread_ticks;
    slippages[index] = descriptor.slippage_ticks;
    commissions[index] = descriptor.commission_micros;
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
    cleanup_host();
    session->release_scenarios();
    return status;
  }

  const std::size_t u64_bytes = count * sizeof(unsigned long long);
  const std::size_t u32_bytes = count * sizeof(unsigned int);
  const std::size_t i32_bytes = count * sizeof(int);
  const std::size_t i64_bytes = count * sizeof(long long);
  if (!guard(copy_to_device(session->scenario_base_candidate_ids, base_ids, u64_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_ids, ids, u64_bytes, session->stream)) ||
      !guard(copy_to_device(session->scenario_rng_counters, counters, u64_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_window_offsets, offsets, u64_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_window_lens, lens, u32_bytes, session->stream)) ||
      !guard(copy_to_device(session->scenario_types, types, u32_bytes, session->stream)) ||
      !guard(copy_to_device(session->scenario_spread_ticks, spreads, i32_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_slippage_ticks, slippages, i32_bytes,
                            session->stream)) ||
      !guard(copy_to_device(session->scenario_commission_micros, commissions, i64_bytes,
                            session->stream))) {
    cleanup_host();
    return status;
  }
  if (cudaStreamSynchronize(session->stream) != cudaSuccess) {
    cleanup_host();
    return NEO_POPULATION_STATUS_TRANSFER_FAILED;
  }
  cleanup_host();

  session->scenario_count = static_cast<int>(count);
  session->scenario_upload_bytes =
      static_cast<std::uint64_t>(4 * u64_bytes + 2 * u32_bytes + 2 * i32_bytes + i64_bytes);
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
  // What the launch is SIZED by. Equal to `population` for every caller that
  // uploads one scenario per gene, which is what makes that case identical to
  // the pre-scenario engine; larger for a quality screen that asks for many
  // treatments of the same genes.
  const int scenario_count = session->scenario_count;
  const int month_capacity = static_cast<int>(settings->month_capacity);

  // Grow the workspace; never merely match it.
  //
  // This read `workspace_population != population`. The host sizes a launch to
  // free VRAM and then splits recursively, so the launch size changes on almost
  // every call — and an equality test turns every one of those into
  // `release_workspace()` followed by a full re-allocation. At 843 456 bars
  // and ~3 300 candidates that was signal_values 2.78 GB, signal_confidences
  // 11.1 GB and outcomes 1.95 GB freed and taken again, 15.9 GB of churn where
  // each `cudaFree` is an implicit device-wide synchronise. Nothing about the
  // work required it. Two of those three arrays no longer exist at all, so the
  // worst case this guards is now 1.95 GB — but the guard is what makes the
  // recursive split free, and it is what lets the split happen at all.
  //
  // A SMALLER workload is safe inside a workspace built for a larger one
  // precisely because every kernel indexes by the CURRENT scenario count: the
  // reduce returns early for `scenario >= scenarios.count`, `read_metrics`
  // copies `session->scenario_count` rows, and the outcome slice is
  // `scenario * kMaxTradesPerCandidate`. All of them address a PREFIX of the
  // resident arrays, which is in bounds by construction.
  //
  // The direction is the whole point, and only one direction is sound. Reusing
  // a workspace for MORE scenarios is the out-of-bounds write this field was
  // introduced to stop (see the field comment on `workspace_scenarios`) — a
  // session built for 256 once ran 25 600. So the test is `<`, never `!=` and
  // never `>`.
  //
  // `month_capacity` and `bars` stay EXACT comparisons on purpose: they are
  // strides, not extents. `monthly_pnls` is indexed `candidate *
  // month_capacity`, so a changed stride re-addresses the entire array rather
  // than shortening it, and having spare room at the end proves nothing about
  // where element (c, i) lands. `bars` no longer strides any workspace array —
  // deleting the signal columns removed the only two — but it still decides
  // what the walk reads, so a changed bar count is still a different workspace.
  //
  // The existence probe moved from `signal_values` to `metric_rows` for the
  // obvious reason: the array it used to name is gone.
  if (session->metric_rows == nullptr || session->month_capacity != month_capacity ||
      session->workspace_scenarios < scenario_count || session->workspace_bars != bars) {
    session->release_workspace();
    std::int32_t status = NEO_POPULATION_STATUS_OK;
    const auto guard = [&](std::int32_t code) {
      if (code != NEO_POPULATION_STATUS_OK) {
        status = code;
      }
      return status == NEO_POPULATION_STATUS_OK;
    };
    if (// No signal or confidence columns, no event buffer, and outcomes sized
        // by the trades a candidate can record rather than by every bar it
        // might have entered on. The per-scenario device cost is now
        // 8192 * 72 + 3 944 = 593 768 B, of which the 3 944 B is the monthly
        // buckets and the metric row; it was 4 811 048 B.
        !guard(device_alloc(&session->outcomes,
                            static_cast<std::size_t>(scenario_count) *
                                static_cast<std::size_t>(kMaxTradesPerCandidate))) ||
        !guard(device_alloc(&session->monthly_pnls,
                            static_cast<std::size_t>(scenario_count) * month_capacity)) ||
        !guard(device_alloc(&session->month_start_equities,
                            static_cast<std::size_t>(scenario_count) * month_capacity)) ||
        !guard(device_alloc(&session->metric_rows, static_cast<std::size_t>(scenario_count))) ||
        !guard(device_alloc(&session->accepted_trade_total, 1))) {
      session->release_workspace();
      return status;
    }
    session->month_capacity = month_capacity;
    session->workspace_scenarios = scenario_count;
    session->workspace_bars = bars;
  }

  if (cudaMemsetAsync(session->accepted_trade_total, 0, sizeof(unsigned long long),
                      session->stream) != cudaSuccess) {
    return NEO_POPULATION_STATUS_TRANSFER_FAILED;
  }

  DeviceDataset dataset;
  dataset.close = session->close;
  dataset.high = session->high;
  dataset.low = session->low;
  dataset.indicators_bar_major = session->indicators_bar_major;
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

  // ── Per-stage timing ────────────────────────────────────────────────────
  //
  // Three times a bottleneck was named from reading the code and twice it was
  // the wrong one. Wall-clock for the whole evaluation cannot tell one stage
  // from another, so every guess costs a rebuild and a rented card. This costs
  // one env var and a handful of events.
  //
  // Off unless NEOETHOS_GPU_STAGE_TIMING is set: the syncs it needs would
  // serialise the stream, which is exactly what you do not want in production
  // and exactly what you do want when measuring.
  //
  // Three stages are gone from the table because they are gone from the file:
  // signal synthesis is now inside the reduce, and count/scan/emit had no
  // launch site. What remains is what actually runs.
  const bool stage_timing = (std::getenv("NEOETHOS_GPU_STAGE_TIMING") != nullptr);
  cudaEvent_t stage_marks[4];
  int stage_mark = 0;
  auto mark_stage = [&]() {
    if (!stage_timing || stage_mark >= 4) {
      return;
    }
    cudaEventCreate(&stage_marks[stage_mark]);
    cudaEventRecord(stage_marks[stage_mark], session->stream);
    stage_mark += 1;
  };
  mark_stage();

  const unsigned int gap_blocks = static_cast<unsigned int>((bars + 255) / 256);
  population_gap_flags_kernel<<<gap_blocks == 0u ? 1u : gap_blocks, 256, 0, session->stream>>>(
      dataset, *settings, session->gap_flags);
  session->kernel_submissions += 1;
  mark_stage();

  // No event total to read and no capacity to guard: the reduce opens positions
  // from the signal, so there is nothing whose size the host has to check before
  // launching. That readback was a stream synchronization every generation, and
  // the capacity it guarded is what split the population 4 096 -> 128 on M3.
  //
  // The overflow flag still matters — it reports a gene whose trades exceeded
  // its slice — but it is read with the metrics at the end rather than blocking
  // here.
  const unsigned long long trade_slots =
      static_cast<unsigned long long>(scenario_count) * kMaxTradesPerCandidate;
  {
    const unsigned int seed_threads = 256u;
    const unsigned long long seed_blocks = (trade_slots + seed_threads - 1ull) / seed_threads;
    population_seed_outcomes_kernel<<<static_cast<unsigned int>(seed_blocks), seed_threads, 0,
                                      session->stream>>>(session->outcomes, trade_slots);
    session->kernel_submissions += 1;
  }
  mark_stage();

  // Block size chosen from the population and the card, not fixed at 128.
  // See `choose_reduce_block`: 128 left 56 of an RTX 3090's 82 SMs with no
  // block at all at the populations the old memory budget allowed.
  const int reduce_block = choose_reduce_block(scenario_count, session->sm_count);
  const unsigned int reduce_blocks =
      static_cast<unsigned int>((scenario_count + reduce_block - 1) / reduce_block);
  population_reduce_kernel<<<reduce_blocks == 0u ? 1u : reduce_blocks, reduce_block, 0,
                             session->stream>>>(dataset, genes, scenario_view, *settings,
                                                session->gap_flags, session->outcomes,
                                                session->monthly_pnls,
                                                session->month_start_equities,
                                                session->metric_rows,
                                                session->accepted_trade_total);
  mark_stage();
  if (stage_timing) {
    cudaStreamSynchronize(session->stream);
    static const char* kStageNames[3] = {"gap_flags", "seed_outcomes", "reduce+signal"};
    // Both counts, because they are no longer the same number and the ratio is
    // the whole point: 174 genes carrying 17 574 scenarios is one launch where
    // it used to be seven.
    std::fprintf(stderr,
                 "[gpu-stage-timing] genes=%lld scenarios=%lld bars=%lld trade_slots=%llu "
                 "block=%d grid=%u sms=%d\n",
                 static_cast<long long>(population), static_cast<long long>(scenario_count),
                 static_cast<long long>(bars),
                 static_cast<unsigned long long>(trade_slots), reduce_block, reduce_blocks,
                 session->sm_count);
    for (int i = 1; i < stage_mark; ++i) {
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, stage_marks[i - 1], stage_marks[i]);
      std::fprintf(stderr, "[gpu-stage-timing]   %-14s %8.1f ms\n",
                   (i - 1) < 3 ? kStageNames[i - 1] : "?", ms);
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

  // Not a count of anything emitted: this is the extent of the outcome array,
  // and it is what bounds `read_diagnostics`. Nothing may read it as "how full
  // the card got" — every candidate is given the same slice whether it trades
  // once or fills it. The host-side percentage that used to be logged from it
  // measured a reservation against a phantom budget and is gone.
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
  // One row per SCENARIO, not per gene. Equal for a caller that uploads the
  // identity descriptor array; 17 574 rows for 174 genes in the quality screen.
  const std::size_t rows = static_cast<std::size_t>(session->scenario_count);
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
