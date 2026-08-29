#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr std::size_t kMaxBatchColumnsV2 = 64U;
constexpr std::size_t kFitWordsV2 = 6U;
constexpr std::uint64_t kCanonicalNanBitsV2 = 0x7ff8000000000000ULL;
constexpr std::uint64_t kSortSentinelBitsV2 = 0x7fffffffffffffffULL;
constexpr std::uint64_t kMadToSigmaBitsV2 = 0x3ff7b8bac710cb29ULL;
constexpr std::uint64_t kRustEpsilonBitsV2 = 0x3cb0000000000000ULL;
constexpr std::uint64_t kClipBitsV2 = 0x4024000000000000ULL;
constexpr unsigned char kValidityValidV2 = 0U;
constexpr unsigned char kValidityDegenerateV2 = 6U;
constexpr unsigned char kValidityNonFiniteV2 = 7U;
constexpr unsigned int kControlNoValidTrainingCellV2 = 1U << 1U;
constexpr unsigned int kControlValidTrainingCellNonFiniteV2 = 1U << 2U;
constexpr unsigned int kKernelThreadsV2 = 256U;
constexpr unsigned int kMaxPortableBlocksV2 = 65535U;
constexpr unsigned int kSha256BlockBytesV2 = 64U;
__device__ __constant__ unsigned char kFitDigestDomainV2[] =
    "neoethos.resident-robust-normalization.fit-metadata.semantic-v2";

__device__ __constant__ std::uint32_t kSha256RoundConstantsV2[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

struct RobustSha256StateV2 {
  std::uint32_t words[8];
  unsigned char block[kSha256BlockBytesV2];
  unsigned int block_len;
  std::uint64_t total_bytes;
};

enum FitWordV2 : std::size_t {
  kFitTrainingStartV2 = 0U,
  kFitTrainingEndV2,
  kFitMedianBitsV2,
  kFitScaleBitsV2,
  kFitValidTrainingCellsV2,
  kFitDegenerateV2,
};

__device__ __forceinline__ std::uint32_t robust_rotate_right_v2(
    std::uint32_t value, unsigned int shift) {
  return (value >> shift) | (value << (32U - shift));
}

__device__ __forceinline__ void robust_sha256_initialize_v2(
    RobustSha256StateV2* state) {
  state->words[0] = 0x6a09e667U;
  state->words[1] = 0xbb67ae85U;
  state->words[2] = 0x3c6ef372U;
  state->words[3] = 0xa54ff53aU;
  state->words[4] = 0x510e527fU;
  state->words[5] = 0x9b05688cU;
  state->words[6] = 0x1f83d9abU;
  state->words[7] = 0x5be0cd19U;
  state->block_len = 0U;
  state->total_bytes = 0U;
}

__device__ __forceinline__ void robust_sha256_compress_v2(
    RobustSha256StateV2* state) {
  std::uint32_t schedule[64];
  for (unsigned int index = 0U; index < 16U; ++index) {
    const unsigned int offset = index * 4U;
    schedule[index] =
        (static_cast<std::uint32_t>(state->block[offset]) << 24U) |
        (static_cast<std::uint32_t>(state->block[offset + 1U]) << 16U) |
        (static_cast<std::uint32_t>(state->block[offset + 2U]) << 8U) |
        static_cast<std::uint32_t>(state->block[offset + 3U]);
  }
  for (unsigned int index = 16U; index < 64U; ++index) {
    const std::uint32_t x = schedule[index - 15U];
    const std::uint32_t y = schedule[index - 2U];
    const std::uint32_t sigma0 = robust_rotate_right_v2(x, 7U) ^
                                 robust_rotate_right_v2(x, 18U) ^ (x >> 3U);
    const std::uint32_t sigma1 = robust_rotate_right_v2(y, 17U) ^
                                 robust_rotate_right_v2(y, 19U) ^ (y >> 10U);
    schedule[index] = schedule[index - 16U] + sigma0 +
                      schedule[index - 7U] + sigma1;
  }
  std::uint32_t a = state->words[0];
  std::uint32_t b = state->words[1];
  std::uint32_t c = state->words[2];
  std::uint32_t d = state->words[3];
  std::uint32_t e = state->words[4];
  std::uint32_t f = state->words[5];
  std::uint32_t g = state->words[6];
  std::uint32_t h = state->words[7];
  for (unsigned int index = 0U; index < 64U; ++index) {
    const std::uint32_t sum1 = robust_rotate_right_v2(e, 6U) ^
                               robust_rotate_right_v2(e, 11U) ^
                               robust_rotate_right_v2(e, 25U);
    const std::uint32_t choice = (e & f) ^ ((~e) & g);
    const std::uint32_t temp1 = h + sum1 + choice +
                                kSha256RoundConstantsV2[index] +
                                schedule[index];
    const std::uint32_t sum0 = robust_rotate_right_v2(a, 2U) ^
                               robust_rotate_right_v2(a, 13U) ^
                               robust_rotate_right_v2(a, 22U);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temp2 = sum0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  state->words[0] += a;
  state->words[1] += b;
  state->words[2] += c;
  state->words[3] += d;
  state->words[4] += e;
  state->words[5] += f;
  state->words[6] += g;
  state->words[7] += h;
  state->block_len = 0U;
}

__device__ __forceinline__ void robust_sha256_update_byte_v2(
    RobustSha256StateV2* state, unsigned char byte) {
  state->block[state->block_len++] = byte;
  ++state->total_bytes;
  if (state->block_len == kSha256BlockBytesV2) {
    robust_sha256_compress_v2(state);
  }
}

__device__ __forceinline__ void robust_sha256_update_bytes_v2(
    RobustSha256StateV2* state, const unsigned char* bytes,
    std::size_t byte_count) {
  for (std::size_t index = 0U; index < byte_count; ++index) {
    robust_sha256_update_byte_v2(state, bytes[index]);
  }
}

__device__ __forceinline__ void robust_sha256_update_u64_be_v2(
    RobustSha256StateV2* state, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    robust_sha256_update_byte_v2(
        state, static_cast<unsigned char>((value >> shift) & 0xffU));
  }
}

__device__ __forceinline__ void robust_sha256_finalize_v2(
    RobustSha256StateV2* state, unsigned char* digest) {
  const std::uint64_t message_bits = state->total_bytes << 3U;
  state->block[state->block_len++] = 0x80U;
  if (state->block_len > 56U) {
    while (state->block_len < kSha256BlockBytesV2) {
      state->block[state->block_len++] = 0U;
    }
    robust_sha256_compress_v2(state);
  }
  while (state->block_len < 56U) {
    state->block[state->block_len++] = 0U;
  }
  for (int shift = 56; shift >= 0; shift -= 8) {
    state->block[state->block_len++] =
        static_cast<unsigned char>((message_bits >> shift) & 0xffU);
  }
  robust_sha256_compress_v2(state);
  for (unsigned int word = 0U; word < 8U; ++word) {
    digest[word * 4U] = static_cast<unsigned char>(state->words[word] >> 24U);
    digest[word * 4U + 1U] =
        static_cast<unsigned char>(state->words[word] >> 16U);
    digest[word * 4U + 2U] =
        static_cast<unsigned char>(state->words[word] >> 8U);
    digest[word * 4U + 3U] =
        static_cast<unsigned char>(state->words[word]);
  }
}

__device__ __forceinline__ double robust_from_bits_v2(std::uint64_t bits) {
  return __longlong_as_double(static_cast<long long>(bits));
}

__device__ __forceinline__ std::uint64_t robust_to_bits_v2(double value) {
  return static_cast<std::uint64_t>(__double_as_longlong(value));
}

__device__ __forceinline__ bool robust_finite_bits_v2(std::uint64_t bits) {
  return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

__device__ __forceinline__ double robust_abs_v2(double value) {
  return robust_from_bits_v2(robust_to_bits_v2(value) &
                             0x7fffffffffffffffULL);
}

// Exact mirror of Rust f64::total_cmp: positive encodings are compared as
// signed i64 unchanged, while every negative encoding flips its lower 63 bits.
__device__ __forceinline__ std::int64_t robust_total_cmp_key_v2(
    std::uint64_t bits) {
  const std::uint64_t negative_mask =
      (bits >> 63U) != 0U ? 0x7fffffffffffffffULL : 0ULL;
  return static_cast<std::int64_t>(bits ^ negative_mask);
}

__device__ __forceinline__ unsigned int robust_atomic_load_word_v2(
    const unsigned int* word) {
  return atomicCAS(const_cast<unsigned int*>(word), 0U, 0U);
}

__device__ __forceinline__ unsigned char robust_validity_at_v2(
    const unsigned char* validity_u4, std::size_t cell) {
  const std::size_t byte_index = cell / 2U;
  const std::size_t word_index = byte_index / sizeof(unsigned int);
  const unsigned int byte_in_word =
      static_cast<unsigned int>(byte_index % sizeof(unsigned int));
  const auto* word =
      reinterpret_cast<const unsigned int*>(validity_u4) + word_index;
  const unsigned int packed_word = robust_atomic_load_word_v2(word);
  const unsigned char packed = static_cast<unsigned char>(
      (packed_word >> (byte_in_word * 8U)) & 0xffU);
  return (cell & 1U) == 0U
             ? static_cast<unsigned char>(packed & 0x0fU)
             : static_cast<unsigned char>((packed >> 4U) & 0x0fU);
}

__device__ __forceinline__ void robust_write_validity_v2(
    unsigned char* validity_u4, std::size_t cell, unsigned char code) {
  const std::size_t byte_index = cell / 2U;
  const std::size_t word_index = byte_index / sizeof(unsigned int);
  const unsigned int byte_in_word =
      static_cast<unsigned int>(byte_index % sizeof(unsigned int));
  const unsigned int nibble_in_byte = static_cast<unsigned int>(cell & 1U);
  const unsigned int shift = byte_in_word * 8U + nibble_in_byte * 4U;
  const unsigned int mask = 0x0fU << shift;
  auto* word = reinterpret_cast<unsigned int*>(validity_u4) + word_index;
  unsigned int observed = robust_atomic_load_word_v2(word);
  while (true) {
    const unsigned int replacement =
        (observed & ~mask) | (static_cast<unsigned int>(code) << shift);
    const unsigned int previous = atomicCAS(word, observed, replacement);
    if (previous == observed) {
      return;
    }
    observed = previous;
  }
}

__device__ __forceinline__ double robust_median_sorted_v2(
    const std::uint64_t* sorted_bits, std::size_t count) {
  const std::size_t mid = count / 2U;
  if ((count & 1U) == 0U) {
    const double low = robust_from_bits_v2(sorted_bits[mid - 1U]);
    const double high = robust_from_bits_v2(sorted_bits[mid]);
    return __dadd_rn(__dmul_rn(low, 0.5), __dmul_rn(high, 0.5));
  }
  return robust_from_bits_v2(sorted_bits[mid]);
}

__global__ void robust_fill_training_v2(
    const double* bar_major_values,
    const unsigned char* bar_major_validity_u4, std::size_t rows,
    std::size_t columns, std::size_t training_start,
    std::size_t training_end, std::size_t column_start,
    std::size_t batch_columns, std::size_t padded_training_rows,
    std::uint64_t* sort_scratch_bits, unsigned int* control_error) {
  const std::size_t training_len = training_end - training_start;
  const std::size_t work = batch_columns * padded_training_rows;
  std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  while (index < work) {
    const std::size_t local_column = index / padded_training_rows;
    const std::size_t training_offset = index % padded_training_rows;
    std::uint64_t value_bits = kSortSentinelBitsV2;
    if (training_offset < training_len) {
      const std::size_t row = training_start + training_offset;
      const std::size_t column = column_start + local_column;
      const std::size_t cell = row * columns + column;
      if (robust_validity_at_v2(bar_major_validity_u4, cell) ==
          kValidityValidV2) {
        const std::uint64_t candidate = robust_to_bits_v2(bar_major_values[cell]);
        if (robust_finite_bits_v2(candidate)) {
          value_bits = candidate;
        } else {
          atomicOr(control_error, kControlValidTrainingCellNonFiniteV2);
        }
      }
    }
    sort_scratch_bits[index] = value_bits;
    index += stride;
  }
}

__global__ void robust_bitonic_stage_v2(
    std::uint64_t* sort_scratch_bits, std::size_t batch_columns,
    std::size_t padded_training_rows, std::size_t merge_width,
    std::size_t compare_stride) {
  const std::size_t work = batch_columns * padded_training_rows;
  std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  while (index < work) {
    const std::size_t local_slot = index % padded_training_rows;
    const std::size_t partner_slot = local_slot ^ compare_stride;
    if (partner_slot > local_slot) {
      const std::size_t segment_start = index - local_slot;
      const std::size_t partner_index = segment_start + partner_slot;
      const std::uint64_t left = sort_scratch_bits[index];
      const std::uint64_t right = sort_scratch_bits[partner_index];
      const std::int64_t left_key = robust_total_cmp_key_v2(left);
      const std::int64_t right_key = robust_total_cmp_key_v2(right);
      const bool ascending = (local_slot & merge_width) == 0U;
      const bool exchange =
          ascending ? left_key > right_key : left_key < right_key;
      if (exchange) {
        sort_scratch_bits[index] = right;
        sort_scratch_bits[partner_index] = left;
      }
    }
    index += stride;
  }
}

__global__ void robust_summarize_values_v2(
    const std::uint64_t* sort_scratch_bits, std::size_t column_start,
    std::size_t batch_columns, std::size_t training_start,
    std::size_t training_end, std::size_t padded_training_rows,
    std::uint64_t* fit_metadata_words, unsigned int* control_error) {
  std::size_t local_column =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  while (local_column < batch_columns) {
    const std::uint64_t* sorted =
        sort_scratch_bits + local_column * padded_training_rows;
    std::size_t valid_count = 0U;
    while (valid_count < padded_training_rows &&
           sorted[valid_count] != kSortSentinelBitsV2) {
      ++valid_count;
    }
    const std::size_t metadata_offset =
        (column_start + local_column) * kFitWordsV2;
    fit_metadata_words[metadata_offset + kFitTrainingStartV2] =
        static_cast<std::uint64_t>(training_start);
    fit_metadata_words[metadata_offset + kFitTrainingEndV2] =
        static_cast<std::uint64_t>(training_end);
    fit_metadata_words[metadata_offset + kFitValidTrainingCellsV2] =
        static_cast<std::uint64_t>(valid_count);
    if (valid_count == 0U) {
      fit_metadata_words[metadata_offset + kFitMedianBitsV2] =
          kCanonicalNanBitsV2;
      fit_metadata_words[metadata_offset + kFitScaleBitsV2] =
          kCanonicalNanBitsV2;
      fit_metadata_words[metadata_offset + kFitDegenerateV2] = 1U;
      atomicOr(control_error, kControlNoValidTrainingCellV2);
      local_column += stride;
      continue;
    }

    const double median = robust_median_sorted_v2(sorted, valid_count);
    double max_abs = 0.0;
    double sum = 0.0;
    for (std::size_t index = 0U; index < valid_count; ++index) {
      const double value = robust_from_bits_v2(sorted[index]);
      const double absolute = robust_abs_v2(value);
      max_abs = absolute > max_abs ? absolute : max_abs;
      sum = __dadd_rn(sum, value);
    }
    const double count = static_cast<double>(valid_count);
    const double mean = __ddiv_rn(sum, count);
    double variance_sum = 0.0;
    for (std::size_t index = 0U; index < valid_count; ++index) {
      const double delta =
          __dsub_rn(robust_from_bits_v2(sorted[index]), mean);
      variance_sum =
          __dadd_rn(variance_sum, __dmul_rn(delta, delta));
    }
    const double fallback_scale =
        __dsqrt_rn(__ddiv_rn(variance_sum, count));
    const double scale_anchor = max_abs > 1.0 ? max_abs : 1.0;
    const double scale_floor = __dmul_rn(
        __dmul_rn(32.0, robust_from_bits_v2(kRustEpsilonBitsV2)),
        scale_anchor);
    fit_metadata_words[metadata_offset + kFitMedianBitsV2] =
        robust_to_bits_v2(median);
    // This slot temporarily retains the population-standard-deviation
    // fallback until robust_finalize_fit_v2 selects the exact final scale.
    fit_metadata_words[metadata_offset + kFitScaleBitsV2] =
        robust_to_bits_v2(fallback_scale);
    fit_metadata_words[metadata_offset + kFitDegenerateV2] =
        robust_to_bits_v2(scale_floor);
    local_column += stride;
  }
}

__global__ void robust_make_deviations_v2(
    std::uint64_t* sort_scratch_bits, const std::uint64_t* fit_metadata_words,
    std::size_t column_start, std::size_t batch_columns,
    std::size_t padded_training_rows) {
  const std::size_t work = batch_columns * padded_training_rows;
  std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  while (index < work) {
    const std::size_t local_column = index / padded_training_rows;
    const std::size_t local_slot = index % padded_training_rows;
    const std::size_t metadata_offset =
        (column_start + local_column) * kFitWordsV2;
    const std::size_t valid_count = static_cast<std::size_t>(
        fit_metadata_words[metadata_offset + kFitValidTrainingCellsV2]);
    if (local_slot < valid_count) {
      const double value = robust_from_bits_v2(sort_scratch_bits[index]);
      const double median = robust_from_bits_v2(
          fit_metadata_words[metadata_offset + kFitMedianBitsV2]);
      sort_scratch_bits[index] =
          robust_to_bits_v2(robust_abs_v2(__dsub_rn(value, median)));
    } else {
      sort_scratch_bits[index] = kSortSentinelBitsV2;
    }
    index += stride;
  }
}

__global__ void robust_finalize_fit_v2(
    const std::uint64_t* sorted_deviation_bits, std::size_t column_start,
    std::size_t batch_columns, std::size_t padded_training_rows,
    std::uint64_t* fit_metadata_words) {
  std::size_t local_column =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  while (local_column < batch_columns) {
    const std::size_t metadata_offset =
        (column_start + local_column) * kFitWordsV2;
    const std::size_t valid_count = static_cast<std::size_t>(
        fit_metadata_words[metadata_offset + kFitValidTrainingCellsV2]);
    if (valid_count != 0U) {
      const std::uint64_t* sorted =
          sorted_deviation_bits + local_column * padded_training_rows;
      const double mad = robust_median_sorted_v2(sorted, valid_count);
      const double mad_scale = __dmul_rn(
          mad, robust_from_bits_v2(kMadToSigmaBitsV2));
      const double fallback_scale = robust_from_bits_v2(
          fit_metadata_words[metadata_offset + kFitScaleBitsV2]);
      const double scale_floor = robust_from_bits_v2(
          fit_metadata_words[metadata_offset + kFitDegenerateV2]);
      const double scale =
          mad_scale > scale_floor ? mad_scale : fallback_scale;
      const bool degenerate =
          !robust_finite_bits_v2(robust_to_bits_v2(scale)) ||
          scale <= scale_floor;
      fit_metadata_words[metadata_offset + kFitScaleBitsV2] =
          robust_to_bits_v2(scale);
      fit_metadata_words[metadata_offset + kFitDegenerateV2] =
          degenerate ? 1U : 0U;
    }
    local_column += stride;
  }
}

__global__ void robust_apply_in_place_v2(
    double* bar_major_values, unsigned char* bar_major_validity_u4,
    std::size_t rows, std::size_t columns, std::size_t column_start,
    std::size_t batch_columns, const std::uint64_t* fit_metadata_words) {
  const std::size_t work = rows * batch_columns;
  std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  while (index < work) {
    const std::size_t row = index / batch_columns;
    const std::size_t local_column = index % batch_columns;
    const std::size_t column = column_start + local_column;
    const std::size_t cell = row * columns + column;
    const std::size_t metadata_offset = column * kFitWordsV2;
    const unsigned char validity =
        robust_validity_at_v2(bar_major_validity_u4, cell);
    if (validity != kValidityValidV2) {
      bar_major_values[cell] = robust_from_bits_v2(kCanonicalNanBitsV2);
    } else if (fit_metadata_words[metadata_offset + kFitDegenerateV2] != 0U) {
      robust_write_validity_v2(bar_major_validity_u4, cell,
                               kValidityDegenerateV2);
      bar_major_values[cell] = robust_from_bits_v2(kCanonicalNanBitsV2);
    } else {
      const double median = robust_from_bits_v2(
          fit_metadata_words[metadata_offset + kFitMedianBitsV2]);
      const double scale = robust_from_bits_v2(
          fit_metadata_words[metadata_offset + kFitScaleBitsV2]);
      const double normalized =
          __ddiv_rn(__dsub_rn(bar_major_values[cell], median), scale);
      if (robust_finite_bits_v2(robust_to_bits_v2(normalized))) {
        const double clip = robust_from_bits_v2(kClipBitsV2);
        bar_major_values[cell] =
            normalized < -clip ? -clip : (normalized > clip ? clip : normalized);
      } else {
        robust_write_validity_v2(bar_major_validity_u4, cell,
                                 kValidityNonFiniteV2);
        bar_major_values[cell] = robust_from_bits_v2(kCanonicalNanBitsV2);
      }
    }
    index += stride;
  }
}

__global__ void robust_fit_metadata_sha256_v2(
    const std::uint64_t* fit_metadata_words,
    std::size_t fit_metadata_word_count, std::uint64_t* digest_scratch_words) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }
  RobustSha256StateV2 state;
  robust_sha256_initialize_v2(&state);
  robust_sha256_update_bytes_v2(&state, kFitDigestDomainV2,
                                sizeof(kFitDigestDomainV2));
  for (std::size_t index = 0U; index < fit_metadata_word_count; ++index) {
    robust_sha256_update_u64_be_v2(&state, fit_metadata_words[index]);
  }
  robust_sha256_finalize_v2(
      &state, reinterpret_cast<unsigned char*>(digest_scratch_words));
}

unsigned int robust_blocks_v2(std::size_t work) {
  const std::size_t needed =
      work / kKernelThreadsV2 + (work % kKernelThreadsV2 != 0U ? 1U : 0U);
  return static_cast<unsigned int>(
      needed < kMaxPortableBlocksV2 ? needed : kMaxPortableBlocksV2);
}

int robust_launch_status_v2() {
  return static_cast<int>(cudaGetLastError());
}

int robust_launch_sort_v2(std::uint64_t* sort_scratch_bits,
                          std::size_t batch_columns,
                          std::size_t padded_training_rows,
                          cudaStream_t stream) {
  const std::size_t work = batch_columns * padded_training_rows;
  const unsigned int blocks = robust_blocks_v2(work);
  for (std::size_t merge_width = 2U;
       merge_width <= padded_training_rows; merge_width <<= 1U) {
    for (std::size_t compare_stride = merge_width >> 1U;
         compare_stride != 0U; compare_stride >>= 1U) {
      robust_bitonic_stage_v2<<<blocks, kKernelThreadsV2, 0, stream>>>(
          sort_scratch_bits, batch_columns, padded_training_rows, merge_width,
          compare_stride);
      const int status = robust_launch_status_v2();
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
    if (merge_width == padded_training_rows) {
      break;
    }
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

extern "C" int neoethos_resident_robust_normalize_bar_major_f64_u4_v2(
    double* bar_major_values, unsigned char* bar_major_validity_u4,
    std::size_t packed_validity_allocated_bytes, std::size_t rows,
    std::size_t columns, std::size_t training_start,
    std::size_t training_end, std::size_t padded_training_rows,
    std::uint64_t* sort_scratch_bits, std::size_t sort_scratch_slots,
    std::uint64_t* fit_metadata_words, std::size_t fit_metadata_word_count,
    unsigned int* control_error, cudaStream_t stream) {
  const std::size_t canonical_training_end = static_cast<std::size_t>(
      std::floor(static_cast<double>(rows) * (1.0 - 0.2)));
  if (bar_major_values == nullptr || bar_major_validity_u4 == nullptr ||
      sort_scratch_bits == nullptr || fit_metadata_words == nullptr ||
      control_error == nullptr || stream == nullptr || rows == 0U ||
      columns == 0U || training_start >= training_end || training_end > rows ||
      training_start != 0U || training_end != canonical_training_end ||
      canonical_training_end < 64U || canonical_training_end >= rows ||
      padded_training_rows < training_end - training_start ||
      (padded_training_rows & (padded_training_rows - 1U)) != 0U ||
      columns > std::numeric_limits<std::size_t>::max() / rows ||
      columns > std::numeric_limits<std::size_t>::max() / kFitWordsV2 ||
      reinterpret_cast<std::uintptr_t>(bar_major_validity_u4) %
              alignof(unsigned int) !=
          0U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t max_batch_columns =
      columns < kMaxBatchColumnsV2 ? columns : kMaxBatchColumnsV2;
  const std::size_t cells = rows * columns;
  const std::size_t packed_validity_logical_bytes =
      cells / 2U + (cells % 2U != 0U ? 1U : 0U);
  if (packed_validity_logical_bytes >
      std::numeric_limits<std::size_t>::max() - 3U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t expected_packed_validity_allocated_bytes =
      ((packed_validity_logical_bytes + 3U) / 4U) * 4U;
  if (max_batch_columns >
          std::numeric_limits<std::size_t>::max() / padded_training_rows ||
      sort_scratch_slots != max_batch_columns * padded_training_rows ||
      fit_metadata_word_count != columns * kFitWordsV2 ||
      packed_validity_allocated_bytes !=
          expected_packed_validity_allocated_bytes ||
      packed_validity_allocated_bytes % alignof(unsigned int) != 0U ||
      sort_scratch_slots < 4U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  for (std::size_t column_start = 0U; column_start < columns;
       column_start += kMaxBatchColumnsV2) {
    const std::size_t remaining = columns - column_start;
    const std::size_t batch_columns =
        remaining < kMaxBatchColumnsV2 ? remaining : kMaxBatchColumnsV2;
    const std::size_t sort_work = batch_columns * padded_training_rows;
    const unsigned int sort_blocks = robust_blocks_v2(sort_work);
    robust_fill_training_v2<<<sort_blocks, kKernelThreadsV2, 0, stream>>>(
        bar_major_values, bar_major_validity_u4, rows, columns, training_start,
        training_end, column_start, batch_columns, padded_training_rows,
        sort_scratch_bits, control_error);
    int status = robust_launch_status_v2();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    status = robust_launch_sort_v2(sort_scratch_bits, batch_columns,
                                   padded_training_rows, stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }

    const unsigned int column_blocks = robust_blocks_v2(batch_columns);
    robust_summarize_values_v2<<<column_blocks, kKernelThreadsV2, 0, stream>>>(
        sort_scratch_bits, column_start, batch_columns, training_start,
        training_end, padded_training_rows, fit_metadata_words, control_error);
    status = robust_launch_status_v2();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    robust_make_deviations_v2<<<sort_blocks, kKernelThreadsV2, 0, stream>>>(
        sort_scratch_bits, fit_metadata_words, column_start, batch_columns,
        padded_training_rows);
    status = robust_launch_status_v2();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    status = robust_launch_sort_v2(sort_scratch_bits, batch_columns,
                                   padded_training_rows, stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    robust_finalize_fit_v2<<<column_blocks, kKernelThreadsV2, 0, stream>>>(
        sort_scratch_bits, column_start, batch_columns, padded_training_rows,
        fit_metadata_words);
    status = robust_launch_status_v2();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }

    const std::size_t apply_work = rows * batch_columns;
    robust_apply_in_place_v2<<<robust_blocks_v2(apply_work), kKernelThreadsV2,
                               0, stream>>>(
        bar_major_values, bar_major_validity_u4, rows, columns, column_start,
        batch_columns, fit_metadata_words);
    status = robust_launch_status_v2();
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  robust_fit_metadata_sha256_v2<<<1, 1, 0, stream>>>(
      fit_metadata_words, fit_metadata_word_count, sort_scratch_bits);
  const int digest_status = robust_launch_status_v2();
  if (digest_status != static_cast<int>(cudaSuccess)) {
    return digest_status;
  }
  return static_cast<int>(cudaSuccess);
}
