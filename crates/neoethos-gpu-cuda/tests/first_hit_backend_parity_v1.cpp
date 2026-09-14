// Test-only analytical CPU reference versus the EXACT production first-hit C ABI.
// Link GPU builds with native/prototype_b.cu (CUDA) or its official HIPIFY output
// (HIP), never stub.cpp. No kernels are implemented or included in this harness.
// Define exactly one backend below. CPU mode is reference-only, not a fallback.
// GPU mode requires --device N and exits nonzero if the device/API is unavailable.
// Scope excludes timestamp/session-gap handling, fills, PnL and full-bot parity:
// this API has only high/low prices, so its gap cases are price jumps only.

#include "neoethos_gpu_cuda.h"

#if (defined(NEOETHOS_PARITY_CPU) + defined(NEOETHOS_PARITY_CUDA) + \
     defined(NEOETHOS_PARITY_HIP)) != 1
#error "Select exactly one explicit first-hit test backend"
#endif
#if defined(NEOETHOS_PARITY_CUDA)
#include <cuda_runtime.h>
#elif defined(NEOETHOS_PARITY_HIP)
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

static_assert(sizeof(NeoFirstHitEvent) == 32 && alignof(NeoFirstHitEvent) == 8);
static_assert(offsetof(NeoFirstHitEvent, stop_price) == 16);
static_assert(offsetof(NeoFirstHitEvent, target_price) == 24);
static_assert(sizeof(NeoFirstHitResult) == 8);
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);

#if defined(NEOETHOS_PARITY_CUDA)
constexpr const char* backend = "cuda";
#elif defined(NEOETHOS_PARITY_HIP)
constexpr const char* backend = "hip";
#else
constexpr const char* backend = "cpu";
#endif

std::string json_string(const std::string& value) {
  std::string result = "\"";
  for (const unsigned char ch : value) {
    if (ch == '"' || ch == '\\') result += '\\';
    if (ch < 32) {
      const char digits[] = "0123456789abcdef";
      result += "\\u00";
      result += digits[ch >> 4];
      result += digits[ch & 15];
    } else {
      result += static_cast<char>(ch);
    }
  }
  return result + '"';
}

struct Case {
  std::string name;
  std::vector<double> high = std::vector<double>(97, 100.5);
  std::vector<double> low = std::vector<double>(97, 99.5);
  NeoFirstHitEvent event{0, 96, 1, 0, 95.0, 105.0};
  NeoFirstHitResult known{-1, 0};
};

Case make_case(std::string name, int direction, int precedence = 0) {
  Case value;
  value.name = std::move(name);
  value.event.direction = direction;
  value.event.precedence = precedence;
  if (direction == -1) {
    value.event.stop_price = 105.0;
    value.event.target_price = 95.0;
  }
  return value;
}

// Test data construction only: set one threshold contact; do not derive answers
// using the production per-bar predicate or its lane/reduction implementation.
void contact(Case& value, std::size_t bar, int reason) {
  const double level = reason == 1 ? value.event.stop_price : value.event.target_price;
  const bool high_contact = (value.event.direction == 1) == (reason == 2);
  (high_contact ? value.high : value.low).at(bar) = level;
}

std::vector<Case> cases() {
  std::vector<Case> result;
  for (const int direction : {1, -1}) {
    const std::string side = direction == 1 ? "long_" : "short_";
    // Every logical lane, including lane31 on an AMD wave64 device, must reach
    // lane0's decision. Alternating outcomes exercise both threshold predicates.
    for (int lane = 0; lane < 32; ++lane) {
      auto value = make_case(side + "lane_" + std::to_string(lane), direction);
      const int bar = lane + 1;
      const int reason = lane % 2 == 0 ? 1 : 2;
      contact(value, static_cast<std::size_t>(bar), reason);
      value.known = {bar, reason};
      result.push_back(std::move(value));
    }
    for (const int bar : {33, 63, 64, 65, 95, 96}) {
      auto value = make_case(side + "inclusive_tail_" + std::to_string(bar), direction);
      value.event.last_bar = static_cast<std::uint32_t>(bar);
      contact(value, static_cast<std::size_t>(bar), 2);
      value.known = {bar, 2};
      result.push_back(std::move(value));
    }
    for (const int precedence : {0, 1}) {
      auto value = make_case(side + "same_bar_tie_" + std::to_string(precedence), direction, precedence);
      value.high[5] = 105.0;
      value.low[5] = 95.0;
      value.known = {5, precedence == 0 ? 1 : 2};
      result.push_back(std::move(value));
    }
    {
      auto value = make_case(side + "earliest_beats_reason_and_lane", direction);
      contact(value, 2, 2);
      contact(value, 31, 1);
      contact(value, 33, 1);
      value.known = {2, 2};
      result.push_back(std::move(value));
    }
    for (const int last : {1, 31, 32, 65}) {
      auto value = make_case(side + "no_hit_" + std::to_string(last), direction);
      value.event.last_bar = static_cast<std::uint32_t>(last);
      result.push_back(std::move(value));
    }
    {
      auto value = make_case(side + "entry_bar_is_excluded", direction);
      value.high[0] = 110.0;
      value.low[0] = 90.0;
      result.push_back(std::move(value));
    }
    {
      auto value = make_case(side + "after_horizon_is_excluded", direction);
      value.event.last_bar = 31;
      contact(value, 32, 1);
      result.push_back(std::move(value));
    }
    for (const int reason : {1, 2}) {
      auto value = make_case(side + "price_jump_" + std::to_string(reason), direction);
      const bool above = (direction == 1) == (reason == 2);
      value.high[7] = above ? 111.0 : 91.0;
      value.low[7] = above ? 109.0 : 89.0;
      value.known = {7, reason};
      result.push_back(std::move(value));
    }
    for (const int reason : {1, 2}) {
      auto value = make_case(side + "f64_boundary_" + std::to_string(reason), direction);
      const bool above = (direction == 1) == (reason == 2);
      std::fill(value.high.begin(), value.high.end(), above ? 0.5 : 1.5);
      std::fill(value.low.begin(), value.low.end(), above ? 0.5 : 1.5);
      value.event.stop_price = reason == 1 ? 1.0 : (above ? 0.0 : 2.0);
      value.event.target_price = reason == 2 ? 1.0 : (above ? 0.0 : 2.0);
      auto& prices = above ? value.high : value.low;
      prices[1] = std::nextafter(1.0, above ? 0.0 : 2.0);
      prices[2] = std::nextafter(1.0, above ? 2.0 : 0.0);
      value.event.last_bar = 2;
      value.known = {2, reason};
      result.push_back(std::move(value));
    }
    for (const int entry : {7, 32}) {
      auto value = make_case(side + "nonzero_entry_" + std::to_string(entry), direction);
      value.event.entry_bar = static_cast<std::uint32_t>(entry);
      contact(value, static_cast<std::size_t>(entry), 1);
      contact(value, static_cast<std::size_t>(entry + 32), 2);
      value.known = {entry + 32, 2};
      result.push_back(std::move(value));
    }
  }
  return result;
}

// Independent sequential oracle: locate the first stop contact and first target
// contact separately, then compare their chronological positions. It contains no
// lane partition, shuffle, parallel minimum, or copied production helper.
NeoFirstHitResult cpu_reference(const Case& value) {
  const auto& event = value.event;
  if (value.high.size() != value.low.size() || event.entry_bar >= event.last_bar ||
      event.last_bar >= value.high.size() || (event.direction != 1 && event.direction != -1) ||
      (event.precedence != 0 && event.precedence != 1)) {
    throw std::runtime_error("fixture outside production Rust first-hit domain");
  }
  int stop = INT_MAX;
  int target = INT_MAX;
  for (std::size_t bar = event.entry_bar + 1u; bar <= event.last_bar; ++bar) {
    if (!std::isfinite(value.high[bar]) || !std::isfinite(value.low[bar]) ||
        value.low[bar] > value.high[bar]) throw std::runtime_error("invalid fixture prices");
    if (stop == INT_MAX && (event.direction == 1 ? value.low[bar] <= event.stop_price
                                               : value.high[bar] >= event.stop_price)) {
      stop = static_cast<int>(bar);
    }
  }
  for (std::size_t bar = event.entry_bar + 1u; bar <= event.last_bar; ++bar) {
    if (target == INT_MAX && (event.direction == 1 ? value.high[bar] >= event.target_price
                                                 : value.low[bar] <= event.target_price)) {
      target = static_cast<int>(bar);
    }
  }
  if (stop == INT_MAX && target == INT_MAX) return {-1, 0};
  if (stop < target || (stop == target && event.precedence == 0)) return {stop, 1};
  return {target, 2};
}

bool equal(NeoFirstHitResult left, NeoFirstHitResult right) {
  return left.exit_bar == right.exit_bar && left.exit_reason == right.exit_reason;
}

std::string hex_word(std::uint64_t word) {
  std::string result(16, '0');
  const char digits[] = "0123456789abcdef";
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[result.size() - 1 - index] = digits[word & 15u];
    word >>= 4;
  }
  return result;
}

std::uint64_t double_bits(double value) {
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Reproducibility fingerprint, NOT a cryptographic authority. Canonical words
// are u64 little-endian: case count, then each name length/UTF-8 bytes, row count,
// event fields (signed fields sign-extended), stop/target f64 bits, highs, lows.
// The orchestration receipt additionally pins the exact source/tool SHA256.
std::string input_identity(const std::vector<Case>& fixtures) {
  std::uint64_t hash = 14695981039346656037ull;
  const auto byte = [&](unsigned char value) { hash = (hash ^ value) * 1099511628211ull; };
  const auto word = [&](std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index) byte(static_cast<unsigned char>(value >> (8 * index)));
  };
  word(fixtures.size());
  for (const auto& value : fixtures) {
    word(value.name.size());
    for (unsigned char ch : value.name) byte(ch);
    word(value.high.size());
    word(value.event.entry_bar);
    word(value.event.last_bar);
    word(static_cast<std::uint64_t>(static_cast<std::int64_t>(value.event.direction)));
    word(static_cast<std::uint64_t>(static_cast<std::int64_t>(value.event.precedence)));
    word(double_bits(value.event.stop_price));
    word(double_bits(value.event.target_price));
    for (double price : value.high) word(double_bits(price));
    for (double price : value.low) word(double_bits(price));
  }
  return hex_word(hash);
}

void metadata_prefix(const std::vector<Case>& fixtures) {
  std::cout << "{\"type\":\"metadata\",\"schema\":\"neoethos.first-hit-backend-parity.v1\",\"backend\":"
            << json_string(backend) << ",\"fixtures\":" << fixtures.size()
            << ",\"cases\":" << 2 * fixtures.size()
            << ",\"scope\":\"first_hit_discrete_decisions_only\",\"input_identity\":{"
               "\"algorithm\":\"fnv1a64_le_v1_noncryptographic\",\"value\":"
            << json_string(input_identity(fixtures)) << "}";
}

void emit_case(const Case& value, NeoFirstHitResult actual, int pass) {
  std::cout << "{\"type\":\"first_hit\",\"id\":"
            << json_string(value.name + (pass == 0 ? "/forward" : "/reverse"))
            << ",\"inputs\":{\"rows\":" << value.high.size()
            << ",\"entry_bar\":" << value.event.entry_bar << ",\"last_bar\":" << value.event.last_bar
            << ",\"direction\":" << value.event.direction << ",\"precedence\":" << value.event.precedence
            << ",\"stop_f64_bits\":" << json_string(hex_word(double_bits(value.event.stop_price)))
            << ",\"target_f64_bits\":" << json_string(hex_word(double_bits(value.event.target_price)))
            << "},\"expected\":{\"exit_bar\":" << value.known.exit_bar
            << ",\"exit_reason\":" << value.known.exit_reason
            << "},\"result\":{\"exit_bar\":" << actual.exit_bar
            << ",\"exit_reason\":" << actual.exit_reason << "}}\n";
}

#if !defined(NEOETHOS_PARITY_CPU)
void select_device(int ordinal, const std::vector<Case>& fixtures) {
  int count = 0;
  int runtime = 0;
  int driver = 0;
#if defined(NEOETHOS_PARITY_CUDA)
  cudaDeviceProp properties{};
  const auto check = [](cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  };
  check(cudaGetDeviceCount(&count));
  if (count <= 0 || ordinal >= count) throw std::runtime_error("requested CUDA device unavailable");
  check(cudaSetDevice(ordinal));
  check(cudaGetDeviceProperties(&properties, ordinal));
  check(cudaRuntimeGetVersion(&runtime));
  check(cudaDriverGetVersion(&driver));
  const std::string architecture = "sm_" + std::to_string(properties.major) + std::to_string(properties.minor);
#else
  hipDeviceProp_t properties{};
  const auto check = [](hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
  };
  check(hipGetDeviceCount(&count));
  if (count <= 0 || ordinal >= count) throw std::runtime_error("requested HIP device unavailable");
  check(hipSetDevice(ordinal));
  check(hipGetDeviceProperties(&properties, ordinal));
  check(hipRuntimeGetVersion(&runtime));
  check(hipDriverGetVersion(&driver));
  const std::string architecture = properties.gcnArchName;
#endif
  metadata_prefix(fixtures);
  std::cout << ",\"role\":\"production_kernel\",\"device\":{\"count\":" << count
            << ",\"ordinal\":" << ordinal << ",\"name\":" << json_string(properties.name)
            << ",\"architecture\":" << json_string(architecture)
            << ",\"warp_size\":" << properties.warpSize
            << ",\"runtime_version\":" << runtime << ",\"driver_version\":" << driver << "}}\n";
}
#endif

int run(int argc, char** argv) {
  const auto fixtures = cases();
  for (const auto& value : fixtures) {
    if (!equal(cpu_reference(value), value.known)) {
      throw std::runtime_error("analytical known-answer mismatch: " + value.name);
    }
  }
#if defined(NEOETHOS_PARITY_CPU)
  if (argc != 1) throw std::runtime_error("CPU-reference mode takes no device argument");
  (void)argv;
  metadata_prefix(fixtures);
  std::cout << ",\"role\":\"independent_reference\",\"device\":null}\n";
  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t slot = 0; slot < fixtures.size(); ++slot) {
      const auto& value = fixtures[pass == 0 ? slot : fixtures.size() - 1 - slot];
      emit_case(value, cpu_reference(value), pass);
    }
  }
  std::cout << "{\"type\":\"summary\",\"cases\":" << 2 * fixtures.size()
            << ",\"failures\":0,\"device_executed\":false,\"status\":\"reference_passed\"}\n";
#else
  if (argc != 3 || std::string(argv[1]) != "--device") throw std::runtime_error("required: --device N");
  char* end = nullptr;
  errno = 0;
  const long ordinal = std::strtol(argv[2], &end, 10);
  if (errno || end == argv[2] || *end != '\0' || ordinal < 0 || ordinal > INT_MAX) {
    throw std::runtime_error("invalid device ordinal");
  }
  select_device(static_cast<int>(ordinal), fixtures);  // No CPU fallback.
  std::vector<double> highs;
  std::vector<double> lows;
  std::vector<NeoFirstHitEvent> events;
  std::vector<NeoFirstHitResult> expected;
  std::vector<int> bases;
  for (const auto& value : fixtures) {
    if (highs.size() + value.high.size() >= static_cast<std::size_t>(INT_MAX)) {
      throw std::runtime_error("test batch exceeds native integer indexing");
    }
    const auto base = static_cast<std::uint32_t>(highs.size());
    highs.insert(highs.end(), value.high.begin(), value.high.end());
    lows.insert(lows.end(), value.low.begin(), value.low.end());
    auto event = value.event;
    event.entry_bar += base;
    event.last_bar += base;
    events.push_back(event);
    auto answer = value.known;
    if (answer.exit_bar >= 0) answer.exit_bar += static_cast<int>(base);
    expected.push_back(answer);
    bases.push_back(static_cast<int>(base));
  }
  std::size_t mismatches = 0;
  // Forward and reverse event order exercise independent blocks and output
  // placement, without changing any data or deriving an answer from a GPU run.
  for (int pass = 0; pass < 2; ++pass) {
    if (pass == 1) std::reverse(events.begin(), events.end());
    std::vector<NeoFirstHitResult> actual(events.size(), {INT_MIN, -999});
    const auto status = neoethos_gpu_cuda_warp_first_hit(
        highs.data(), lows.data(), highs.size(), events.data(), actual.data(), events.size());
    if (status != 0) throw std::runtime_error("production first-hit API status " + std::to_string(status));
    for (std::size_t slot = 0; slot < actual.size(); ++slot) {
      const std::size_t index = pass == 0 ? slot : actual.size() - 1 - slot;
      const bool matched = equal(actual[slot], expected[index]);
      mismatches += matched ? 0u : 1u;
      const int local_bar = actual[slot].exit_bar >= 0 ? actual[slot].exit_bar - bases[index]
                                                     : actual[slot].exit_bar;
      emit_case(fixtures[index], {local_bar, actual[slot].exit_reason}, pass);
    }
  }
  std::cout << "{\"type\":\"summary\",\"cases\":" << 2 * fixtures.size()
            << ",\"failures\":" << mismatches << ",\"device_executed\":true,\"production_calls\":2"
            << ",\"status\":" << json_string(mismatches == 0 ? "subset_parity_passed" : "failed") << "}\n";
  if (mismatches != 0) return 1;
#endif
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "{\"type\":\"error\",\"schema\":\"neoethos.first-hit-backend-parity.v1\",\"backend\":"
              << json_string(backend) << ",\"status\":\"failed\",\"error\":"
              << json_string(error.what()) << "}\n";
    return 1;
  }
}
