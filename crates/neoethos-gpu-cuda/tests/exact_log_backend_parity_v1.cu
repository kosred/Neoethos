// Production exact-log CPU/CUDA/HIP parity probe; never a CPU fallback.
// Select exactly one backend macro. For HIP, supply the validated HIPIFY
// production-header directory via -I; for CUDA, use the original native dir.
// Run: exact_log_backend_parity_v1 <fixture.csv> [device ordinal, default 0].
#if defined(NEOETHOS_PARITY_HIP) == defined(NEOETHOS_PARITY_CUDA)
#error "Select exactly one of NEOETHOS_PARITY_HIP and NEOETHOS_PARITY_CUDA"
#endif
#if defined(NEOETHOS_PARITY_HIP)
#include <hip/hip_runtime.h>
using RuntimeError = hipError_t;
using RuntimeStream = hipStream_t;
using RuntimeProperties = hipDeviceProp_t;
#define runtime_success hipSuccess
#define runtime_set_device hipSetDevice
#define runtime_get_properties hipGetDeviceProperties
#define runtime_error_string hipGetErrorString
#define runtime_malloc hipMalloc
#define runtime_free hipFree
#define runtime_copy hipMemcpy
#define runtime_h2d hipMemcpyHostToDevice
#define runtime_d2h hipMemcpyDeviceToHost
#define runtime_stream_create hipStreamCreate
#define runtime_stream_destroy hipStreamDestroy
#define runtime_stream_sync hipStreamSynchronize
#define runtime_last_error hipGetLastError
constexpr const char* BACKEND = "hip";
#else
#include <cuda_runtime.h>
using RuntimeError = cudaError_t;
using RuntimeStream = cudaStream_t;
using RuntimeProperties = cudaDeviceProp;
#define runtime_success cudaSuccess
#define runtime_set_device cudaSetDevice
#define runtime_get_properties cudaGetDeviceProperties
#define runtime_error_string cudaGetErrorString
#define runtime_malloc cudaMalloc
#define runtime_free cudaFree
#define runtime_copy cudaMemcpy
#define runtime_h2d cudaMemcpyHostToDevice
#define runtime_d2h cudaMemcpyDeviceToHost
#define runtime_stream_create cudaStreamCreate
#define runtime_stream_destroy cudaStreamDestroy
#define runtime_stream_sync cudaStreamSynchronize
#define runtime_last_error cudaGetLastError
constexpr const char* BACKEND = "cuda";
#endif

#include "resident_exact_log_v3.cuh"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::uint64_t SENTINEL = 0x0123456789abcdefULL;
enum Operation : std::uint32_t { Log, Add, Sub, Mul, Div, Unfused, Fused };
struct Input {
  std::uint64_t a, b, c;
  std::uint32_t operation;
};
struct Output {
  std::uint64_t bits;
  std::uint32_t accepted;
};
struct Case {
  std::string id, operation;
  Input input{};
  bool expected_accepted = true;
  bool has_exact = false, has_accuracy = false;
  std::uint64_t exact = 0, rounded = 0;
};

__global__ void production_exact_log_parity_kernel_v1(
    const Input* inputs, Output* outputs, std::size_t count) {
  const std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (index >= count) return;
  const Input input = inputs[index];
  const double a = __longlong_as_double(static_cast<long long>(input.a));
  const double b = __longlong_as_double(static_cast<long long>(input.b));
  const double c = __longlong_as_double(static_cast<long long>(input.c));
  double result = __longlong_as_double(static_cast<long long>(SENTINEL));
  bool accepted = true;
  using namespace neoethos_exact_math_v3;
  switch (input.operation) {
    case Log: accepted = exact_log_positive_f64_v3(a, &result); break;
    case Add: result = add_rn_v3(a, b); break;
    case Sub: result = sub_rn_v3(a, b); break;
    case Mul: result = mul_rn_v3(a, b); break;
    case Div: result = div_rn_v3(a, b); break;
    case Unfused: result = add_rn_v3(mul_rn_v3(a, b), c); break;
    case Fused: result = __fma_rn(a, b, c); break;
    default: accepted = false; break;
  }
  outputs[index] = {
      static_cast<std::uint64_t>(__double_as_longlong(result)),
      accepted ? 1u : 0u};
}

std::uint64_t parse_bits(const std::string& text) {
  if (text.size() != 16 || text.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos)
    throw std::runtime_error("invalid 16-digit bits: " + text);
  return std::stoull(text, nullptr, 16);
}
std::string hex(std::uint64_t bits) {
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << bits;
  return out.str();
}
std::string json_string(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
    else if (c < 0x20u)
      out << "\\u00" << std::hex << std::setfill('0') << std::setw(2)
          << static_cast<unsigned>(c) << std::dec;
    else out << static_cast<char>(c);
  }
  out << '"';
  return out.str();
}
std::uint64_t ordered(std::uint64_t bits) {
  return bits >> 63 == 0 ? bits | (1ULL << 63) : ~bits;
}
bool finite_bits(std::uint64_t bits) {
  return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}
void check(RuntimeError result, const char* action) {
  if (result != runtime_success)
    throw std::runtime_error(std::string(action) + ": " + runtime_error_string(result));
}
std::vector<Case> read_cases(const char* path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open fixture");
  std::vector<Case> cases;
  std::set<std::string> ids;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    if (cases.size() == 4096) throw std::runtime_error("fixture exceeds 4096 cases");
    std::vector<std::string> fields;
    std::istringstream row(line);
    for (std::string field; std::getline(row, field, ',');) fields.push_back(field);
    if (line.back() == ',' || fields.size() != 7 || fields[0].empty()
        || fields[0].find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
            != std::string::npos || !ids.insert(fields[0]).second)
      throw std::runtime_error("malformed/duplicate vector: " + line);
    Case item;
    item.id = fields[0];
    item.operation = fields[1];
    const char* names[] = {"log", "add", "sub", "mul", "div", "unfused", "fused"};
    std::uint32_t op = 0;
    while (op != 7 && item.operation != names[op]) ++op;
    if (op == 7) throw std::runtime_error("unknown operation");
    item.input = {parse_bits(fields[2]), parse_bits(fields[3]), parse_bits(fields[4]), op};
    item.expected_accepted = fields[5] != "reject";
    if (!item.expected_accepted && (op != Log || fields[6] != "-"))
      throw std::runtime_error("only log-domain rejection is supported");
    item.has_exact = item.expected_accepted && fields[5] != "-";
    if (item.has_exact) item.exact = parse_bits(fields[5]);
    item.has_accuracy = fields[6] != "-";
    if (item.has_accuracy) {
      item.rounded = parse_bits(fields[6]);
      if (op != Log || !finite_bits(item.rounded))
        throw std::runtime_error("invalid log accuracy reference");
    }
    if (item.expected_accepted && !item.has_exact && !item.has_accuracy)
      throw std::runtime_error("missing independent expectation");
    cases.push_back(item);
  }
  if (!file.eof()) throw std::runtime_error("fixture read failed");
  if (cases.empty()) throw std::runtime_error("empty fixture");
  return cases;
}
}  // namespace

int main(int argc, char** argv) {
  bool device_executed = false;
  std::size_t cases_count = 0;
  // Keep DMA-facing host storage outside the exception-unwind scope too.
  std::vector<Input> inputs;
  std::vector<Output> outputs;
  try {
    if (argc != 2 && argc != 3)
      throw std::runtime_error("usage: exact_log_backend_parity_v1 <fixture.csv> [device ordinal]");
    int device = 0;
    if (argc == 3) {
      const std::string text(argv[2]);
      if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("device ordinal must be nonnegative decimal");
      const auto parsed = std::stoull(text);
      if (parsed > static_cast<unsigned>(std::numeric_limits<int>::max()))
        throw std::runtime_error("device ordinal overflow");
      device = static_cast<int>(parsed);
    }
    const auto cases = read_cases(argv[1]);
    cases_count = cases.size();
    for (const auto& item : cases) inputs.push_back(item.input);
    outputs.resize(cases.size());
    check(runtime_set_device(device), "select actual device");
    RuntimeProperties properties{};
    check(runtime_get_properties(&properties, device), "query actual device");
#if defined(NEOETHOS_PARITY_HIP)
    const std::string architecture = properties.gcnArchName;
#else
    const std::string architecture =
        "sm_" + std::to_string(properties.major) + std::to_string(properties.minor);
#endif
    std::cout << "{\"type\":\"metadata\",\"schema\":\"neoethos.exact-log-backend-parity.v1\","
                 "\"backend\":\"" << BACKEND << "\",\"role\":\"production_device\","
                 "\"device_name\":" << json_string(properties.name)
              << ",\"device_ordinal\":" << device
              << ",\"architecture\":" << json_string(architecture)
              << ",\"warp_size\":" << properties.warpSize
              << ",\"cases\":" << cases_count << "}\n";
    RuntimeStream stream{};
    check(runtime_stream_create(&stream), "create stream");
    Input* device_inputs = nullptr;
    Output* device_outputs = nullptr;
    check(runtime_malloc(reinterpret_cast<void**>(&device_inputs), inputs.size() * sizeof(Input)),
          "allocate runtime inputs");
    check(runtime_malloc(reinterpret_cast<void**>(&device_outputs), outputs.size() * sizeof(Output)),
          "allocate results");
    // Runtime device pointers prevent constant-folding the device's data.
    // These synchronous copies retain host buffers for the entire transfer.
    check(runtime_copy(device_inputs, inputs.data(), inputs.size() * sizeof(Input), runtime_h2d),
          "copy runtime inputs");
    production_exact_log_parity_kernel_v1<<<(inputs.size() + 127) / 128, 128, 0, stream>>>(
        device_inputs, device_outputs, inputs.size());
    check(runtime_last_error(), "launch production math probe");
    check(runtime_stream_sync(stream), "complete production math probe");
    device_executed = true;
    check(runtime_copy(outputs.data(), device_outputs, outputs.size() * sizeof(Output), runtime_d2h),
          "read completed results");
    check(runtime_free(device_outputs), "release results");
    check(runtime_free(device_inputs), "release inputs");
    check(runtime_stream_destroy(stream), "release stream");

    std::size_t failures = 0;
    for (std::size_t i = 0; i < cases.size(); ++i) {
      const auto& item = cases[i];
      const auto result = outputs[i];
      const bool accepted = result.accepted != 0u;
      bool passed = result.accepted <= 1u && accepted == item.expected_accepted;
      if (!item.expected_accepted) passed = passed && result.bits == SENTINEL;
      if (item.has_exact) passed = passed && result.bits == item.exact;
      std::uint64_t ulp = 0;
      if (item.has_accuracy) {
        const auto actual_order = ordered(result.bits), expected_order = ordered(item.rounded);
        ulp = actual_order >= expected_order ? actual_order - expected_order : expected_order - actual_order;
        passed = passed && accepted && finite_bits(result.bits) && ulp <= 1;
      }
      if (!passed) ++failures;
      std::cout << "{\"type\":\"case\",\"id\":\"" << item.id
                << "\",\"operation\":\"" << item.operation
                << "\",\"input_bits\":\"" << hex(item.input.a)
                << "\",\"b_bits\":\"" << hex(item.input.b)
                << "\",\"c_bits\":\"" << hex(item.input.c)
                << "\",\"accepted\":" << (accepted ? "true" : "false")
                << ",\"output_bits\":\"" << hex(result.bits)
                << "\",\"accuracy_ulp\":" << (item.has_accuracy ? std::to_string(ulp) : "null")
                << ",\"passed\":" << (passed ? "true" : "false") << "}\n";
    }
    std::cout << "{\"type\":\"summary\",\"cases\":" << cases.size()
              << ",\"failures\":" << failures << ",\"device_executed\":true}\n";
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    // On an ambiguous runtime failure, terminate the standalone process rather
    // than freeing buffers that an unproven asynchronous operation may retain.
    std::cerr << error.what() << "\n";
    std::cout << "{\"type\":\"summary\",\"cases\":" << cases_count
              << ",\"failures\":1,\"device_executed\":"
              << (device_executed ? "true" : "false") << "}\n";
    std::cerr.flush();
    std::cout.flush();
    std::_Exit(2);
  }
}
