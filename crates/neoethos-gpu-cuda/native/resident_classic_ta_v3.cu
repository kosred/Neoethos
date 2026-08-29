#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr unsigned int kThreadsV3 = 256U;
constexpr std::size_t kMaxResidentClassicTaBatchColumnsV3 = 64U;
constexpr unsigned char kValidV3 = 0U;
constexpr unsigned char kWarmupV3 = 1U;
constexpr unsigned char kNonFiniteV3 = 7U;
// Exact FeatureCellValidity::{Valid, Warmup, NonFinite, ComputeFailure}
// numeric authority is carried by the recipe's 0..=9 all-NaN reason byte.

__device__ __forceinline__ double canonical_nan_v3() {
  return __longlong_as_double(static_cast<long long>(0x7ff8000000000000ULL));
}

__global__ void resident_classic_derived_inputs_f64_v3(
    const double* high, const double* low, const double* close,
    std::size_t rows, double* hlc3, double* hl2, double* hlcc4) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          static_cast<std::size_t>(threadIdx.x);
  if (row >= rows) {
    return;
  }
  const double row_high = high[row];
  const double row_low = low[row];
  const double row_close = close[row];
  // These are the literal vector-ta Candles operation orders. Native builds
  // compile with --fmad=false, so no platform-dependent contraction can turn
  // either expression into a different feature value.
  hl2[row] = (row_high + row_low) / 2.0;
  hlc3[row] = (row_high + row_low + row_close) / 3.0;
  hlcc4[row] = (row_high + row_low + 2.0 * row_close) / 4.0;
}

__global__ void resident_classic_fill_nan_f64_v3(double* values,
                                                   std::size_t cells) {
  const std::size_t cell = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           static_cast<std::size_t>(threadIdx.x);
  if (cell < cells) {
    values[cell] = canonical_nan_v3();
  }
}

__global__ void resident_classic_first_finite_f64_v3(
    const std::uint64_t* value_addresses, const std::uint64_t* value_offsets,
    std::size_t rows, std::size_t columns,
    unsigned long long* first_finite_rows, unsigned int* device_error) {
  const std::size_t column = static_cast<std::size_t>(blockIdx.y);
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          static_cast<std::size_t>(threadIdx.x);
  if (column >= columns || row >= rows) {
    return;
  }
  const auto address = static_cast<unsigned long long>(value_addresses[column]);
  const std::size_t offset = static_cast<std::size_t>(value_offsets[column]);
  const std::size_t size_max_v3 = ~std::size_t{0};
  if (address == 0ULL || address % alignof(double) != 0ULL ||
      offset > size_max_v3 - rows) {
    atomicCAS(device_error, 0U, 2U);
    return;
  }
  const auto* values = reinterpret_cast<const double*>(address);
  if (isfinite(values[offset + row])) {
    atomicMin(first_finite_rows + column,
              static_cast<unsigned long long>(row));
  }
}

__global__ void resident_classic_validity_u8_v3(
    const std::uint64_t* value_addresses, const std::uint64_t* value_offsets,
    const unsigned long long* first_finite_rows,
    const unsigned char* all_nan_validity_codes, std::size_t rows,
    std::size_t columns, unsigned char* validity_u8,
    unsigned int* device_error) {
  const std::size_t column = static_cast<std::size_t>(blockIdx.y);
  const std::size_t row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                          static_cast<std::size_t>(threadIdx.x);
  if (column >= columns || row >= rows) {
    return;
  }
  const unsigned char all_nan_code = all_nan_validity_codes[column];
  if (all_nan_code > 9U) {
    atomicCAS(device_error, 0U, 1U);
    validity_u8[column * rows + row] = 0xffU;
    return;
  }
  const auto address = static_cast<unsigned long long>(value_addresses[column]);
  const std::size_t offset = static_cast<std::size_t>(value_offsets[column]);
  const std::size_t size_max_v3 = ~std::size_t{0};
  if (address == 0ULL || address % alignof(double) != 0ULL ||
      offset > size_max_v3 - rows) {
    atomicCAS(device_error, 0U, 2U);
    validity_u8[column * rows + row] = 0xffU;
    return;
  }
  auto* values = reinterpret_cast<double*>(address);
  const std::size_t value_index = offset + row;
  const std::size_t validity_index = column * rows + row;
  const double value = values[value_index];
  if (isinf(value)) {
    atomicCAS(device_error, 0U, 3U);
    validity_u8[validity_index] = 0xffU;
    values[value_index] = canonical_nan_v3();
    return;
  }
  const unsigned long long first_finite = first_finite_rows[column];
  const std::uint64_t u64_max_v3 = ~std::uint64_t{0};
  unsigned char validity = kValidV3;
  if (first_finite == static_cast<unsigned long long>(u64_max_v3)) {
    validity = all_nan_code;
  } else if (isfinite(value)) {
    validity = kValidV3;
  } else if (row < static_cast<std::size_t>(first_finite) && isnan(value)) {
    validity = kWarmupV3;
  } else {
    validity = kNonFiniteV3;
  }
  validity_u8[validity_index] = validity;
  if (validity != kValidV3) {
    // FeatureColumnF64::new is the scalar authority: every typed-invalid cell
    // carries the canonical quiet-NaN payload. Perform that canonicalization
    // in place without ever downloading the output matrix.
    values[value_index] = canonical_nan_v3();
  }
}

bool grid_for_rows_v3(std::size_t rows, unsigned int* grid_x) {
  if (grid_x == nullptr || rows == 0) {
    return false;
  }
  const std::size_t blocks =
      rows / static_cast<std::size_t>(kThreadsV3) +
      (rows % static_cast<std::size_t>(kThreadsV3) != 0 ? 1U : 0U);
  if (blocks == 0 ||
      blocks > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max())) {
    return false;
  }
  int device = -1;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
      blocks > static_cast<std::size_t>(properties.maxGridSize[0])) {
    return false;
  }
  *grid_x = static_cast<unsigned int>(blocks);
  return true;
}

bool grid_for_matrix_v3(std::size_t rows, std::size_t columns,
                        dim3* grid) {
  if (grid == nullptr || columns == 0 ||
      columns > kMaxResidentClassicTaBatchColumnsV3) {
    return false;
  }
  unsigned int grid_x = 0;
  if (!grid_for_rows_v3(rows, &grid_x)) {
    return false;
  }
  int device = -1;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
      columns > static_cast<std::size_t>(properties.maxGridSize[1])) {
    return false;
  }
  *grid = dim3(grid_x, static_cast<unsigned int>(columns), 1U);
  return true;
}

}  // namespace

extern "C" int neoethos_resident_classic_derived_inputs_f64_v3(
    const double* high, const double* low, const double* close,
    std::size_t rows, double* hlc3, double* hl2, double* hlcc4,
    cudaStream_t stream) {
  unsigned int grid_x = 0;
  if (high == nullptr || low == nullptr || close == nullptr || hlc3 == nullptr ||
      hl2 == nullptr || hlcc4 == nullptr || stream == nullptr ||
      !grid_for_rows_v3(rows, &grid_x)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  resident_classic_derived_inputs_f64_v3<<<grid_x, kThreadsV3, 0, stream>>>(
      high, low, close, rows, hlc3, hl2, hlcc4);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int neoethos_resident_classic_fill_nan_f64_v3(
    double* values, std::size_t cells, cudaStream_t stream) {
  unsigned int grid_x = 0;
  if (values == nullptr || stream == nullptr ||
      !grid_for_rows_v3(cells, &grid_x)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  resident_classic_fill_nan_f64_v3<<<grid_x, kThreadsV3, 0, stream>>>(
      values, cells);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int neoethos_resident_classic_validity_u8_v3(
    const std::uint64_t* value_addresses, const std::uint64_t* value_offsets,
    const unsigned char* all_nan_validity_codes, std::size_t rows,
    std::size_t columns, unsigned long long* first_finite_rows,
    unsigned char* validity_u8, unsigned int* device_error,
    cudaStream_t stream) {
  dim3 grid{};
  if (value_addresses == nullptr || value_offsets == nullptr ||
      all_nan_validity_codes == nullptr || first_finite_rows == nullptr ||
      validity_u8 == nullptr || device_error == nullptr || stream == nullptr ||
      columns == 0 || columns > kMaxResidentClassicTaBatchColumnsV3 ||
      rows > std::numeric_limits<std::size_t>::max() / columns ||
      !grid_for_matrix_v3(rows, columns, &grid)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t first_finite_bytes =
      columns * sizeof(unsigned long long);
  cudaError_t status = cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = cudaMemsetAsync(first_finite_rows, 0xff, first_finite_bytes, stream);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resident_classic_first_finite_f64_v3<<<grid, kThreadsV3, 0, stream>>>(
      value_addresses, value_offsets, rows, columns, first_finite_rows,
      device_error);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resident_classic_validity_u8_v3<<<grid, kThreadsV3, 0, stream>>>(
      value_addresses, value_offsets, first_finite_rows,
      all_nan_validity_codes, rows, columns, validity_u8, device_error);
  return static_cast<int>(cudaGetLastError());
}
