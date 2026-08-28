#include <cuda.h>  // exact Driver/Runtime primary-context interop
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

constexpr unsigned int kSha256BlockBytes = 64;
constexpr unsigned int kSha256Threads = 128;
constexpr unsigned int kMaxPortableBlocks = 65535;
constexpr unsigned int kLayoutTileRows = 32;
constexpr unsigned int kLayoutTileColumns = 32;
constexpr unsigned int kLayoutBlockRows = 8;
constexpr std::size_t kCanonicalMerkleChunkRowsV3 = 4096;

// Every V3 SHA domain includes its implicit trailing C NUL. This deliberately
// separates leaves, internal nodes and the final shape-bound root.
__device__ __constant__ unsigned char kCanonicalMerkleLeafDomainV3[] =
    "neoethos.canonical-feature-content.merkle.leaf.v3";
__device__ __constant__ unsigned char kCanonicalMerkleNodeDomainV3[] =
    "neoethos.canonical-feature-content.merkle.node.v3";
__device__ __constant__ unsigned char kCanonicalMerkleRootDomainV3[] =
    "neoethos.canonical-feature-content.merkle.root.v3";

__device__ __constant__ std::uint32_t kSha256RoundConstants[64] = {
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

struct Sha256StateV1 {
    std::uint32_t words[8];
    unsigned char block[kSha256BlockBytes];
    unsigned int block_len;
    std::uint64_t total_bytes;
};

__device__ __forceinline__ std::uint32_t rotate_right(std::uint32_t value,
                                                       unsigned int shift) {
    return (value >> shift) | (value << (32U - shift));
}

__device__ __forceinline__ void sha256_initialize(Sha256StateV1* state) {
    state->words[0] = 0x6a09e667U;
    state->words[1] = 0xbb67ae85U;
    state->words[2] = 0x3c6ef372U;
    state->words[3] = 0xa54ff53aU;
    state->words[4] = 0x510e527fU;
    state->words[5] = 0x9b05688cU;
    state->words[6] = 0x1f83d9abU;
    state->words[7] = 0x5be0cd19U;
    state->block_len = 0;
    state->total_bytes = 0;
}

__device__ __forceinline__ void sha256_compress(Sha256StateV1* state) {
    std::uint32_t schedule[64];
    for (unsigned int index = 0; index < 16; ++index) {
        const unsigned int offset = index * 4U;
        schedule[index] =
            (static_cast<std::uint32_t>(state->block[offset]) << 24U) |
            (static_cast<std::uint32_t>(state->block[offset + 1U]) << 16U) |
            (static_cast<std::uint32_t>(state->block[offset + 2U]) << 8U) |
            static_cast<std::uint32_t>(state->block[offset + 3U]);
    }
    for (unsigned int index = 16; index < 64; ++index) {
        const std::uint32_t x = schedule[index - 15U];
        const std::uint32_t y = schedule[index - 2U];
        const std::uint32_t sigma0 =
            rotate_right(x, 7U) ^ rotate_right(x, 18U) ^ (x >> 3U);
        const std::uint32_t sigma1 =
            rotate_right(y, 17U) ^ rotate_right(y, 19U) ^ (y >> 10U);
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

    for (unsigned int index = 0; index < 64; ++index) {
        const std::uint32_t sum1 =
            rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
        const std::uint32_t choice = (e & f) ^ ((~e) & g);
        const std::uint32_t temp1 = h + sum1 + choice +
                                    kSha256RoundConstants[index] + schedule[index];
        const std::uint32_t sum0 =
            rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
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
    state->block_len = 0;
}

__device__ __forceinline__ void sha256_update_byte(Sha256StateV1* state,
                                                    unsigned char byte) {
    state->block[state->block_len++] = byte;
    ++state->total_bytes;
    if (state->block_len == kSha256BlockBytes) {
        sha256_compress(state);
    }
}

__device__ __forceinline__ void sha256_update_bytes(Sha256StateV1* state,
                                                     const unsigned char* bytes,
                                                     std::size_t byte_len) {
    for (std::size_t index = 0; index < byte_len; ++index) {
        sha256_update_byte(state, bytes[index]);
    }
}

__device__ __forceinline__ void sha256_update_u64_le(Sha256StateV1* state,
                                                      std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64U; shift += 8U) {
        sha256_update_byte(state,
                           static_cast<unsigned char>((value >> shift) & 0xffU));
    }
}

__device__ __forceinline__ void sha256_finalize(Sha256StateV1* state,
                                                 unsigned char* digest) {
    const std::uint64_t message_bits = state->total_bytes << 3U;
    state->block[state->block_len++] = 0x80U;
    if (state->block_len > 56U) {
        while (state->block_len < kSha256BlockBytes) {
            state->block[state->block_len++] = 0;
        }
        sha256_compress(state);
    }
    while (state->block_len < 56U) {
        state->block[state->block_len++] = 0;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        state->block[state->block_len++] =
            static_cast<unsigned char>((message_bits >> shift) & 0xffU);
    }
    sha256_compress(state);

    for (unsigned int word = 0; word < 8U; ++word) {
        digest[word * 4U] = static_cast<unsigned char>(state->words[word] >> 24U);
        digest[word * 4U + 1U] =
            static_cast<unsigned char>(state->words[word] >> 16U);
        digest[word * 4U + 2U] =
            static_cast<unsigned char>(state->words[word] >> 8U);
        digest[word * 4U + 3U] = static_cast<unsigned char>(state->words[word]);
    }
}

__global__ void pack_sources_to_bar_major_f64_u4_v3(
    const std::uint64_t* source_addresses,
    const std::uint64_t* source_offsets,
    const std::uint64_t* source_validity_addresses,
    const std::uint64_t* source_validity_offsets,
    std::size_t rows,
    std::size_t source_columns,
    std::size_t destination_columns,
    std::size_t destination_column_start,
    std::uint64_t* search_bar_major_value_bits,
    unsigned char* search_bar_major_validity_u4,
    unsigned int* validity_code_error) {
    __shared__ std::uint64_t
        shared_value_bits[kLayoutTileColumns][kLayoutTileRows + 1U];
    __shared__ unsigned char
        shared_validity[kLayoutTileColumns][kLayoutTileRows + 1U];

    const std::size_t column_lane = threadIdx.x;
    const std::size_t row_group = threadIdx.y;
    const std::size_t column_start =
        static_cast<std::size_t>(blockIdx.x) * kLayoutTileColumns;
    const std::size_t row_tile_count =
        (rows - 1U) / kLayoutTileRows + 1U;
    for (std::size_t row_tile = blockIdx.y; row_tile < row_tile_count;
         row_tile += gridDim.y) {
        const std::size_t row_start = row_tile * kLayoutTileRows;

        // Every warp has one fixed source-column lane-group and reads 32
        // consecutive rows from that producer allocation. The [32][33]
        // padding removes shared-memory bank conflicts during the transpose.
        for (std::size_t column_lane_load = row_group;
             column_lane_load < kLayoutTileColumns;
             column_lane_load += kLayoutBlockRows) {
            const std::size_t source_column =
                column_start + column_lane_load;
            const std::size_t row = row_start + column_lane;
            if (source_column < source_columns && row < rows) {
                const std::uint64_t* source_value_bits =
                    reinterpret_cast<const std::uint64_t*>(
                        static_cast<std::uintptr_t>(
                            source_addresses[source_column]));
                const unsigned char* source_validity =
                    reinterpret_cast<const unsigned char*>(
                        static_cast<std::uintptr_t>(
                            source_validity_addresses[source_column]));
                shared_value_bits[column_lane_load][column_lane] =
                    source_value_bits[static_cast<std::size_t>(
                                          source_offsets[source_column]) +
                                      row];
                shared_validity[column_lane_load][column_lane] =
                    source_validity[static_cast<std::size_t>(
                                        source_validity_offsets[source_column]) +
                                    row];
            }
        }
        __syncthreads();

        // Every warp now stores 32 adjacent feature columns for one bar, so
        // the final Search bar-major value writes are fully coalesced. One
        // unique even logical cell also writes its low-nibble-first u4 pair;
        // a tile/row boundary partner is read directly from its producer.
        const std::size_t source_column = column_start + column_lane;
        const std::size_t destination_column =
            destination_column_start + source_column;
        for (std::size_t row_lane = row_group; row_lane < kLayoutTileRows;
             row_lane += kLayoutBlockRows) {
            const std::size_t row = row_start + row_lane;
            if (source_column < source_columns && row < rows) {
                const std::size_t cell =
                    row * destination_columns + destination_column;
                search_bar_major_value_bits[cell] =
                    shared_value_bits[column_lane][row_lane];
                if ((cell & 1U) == 0U) {
                    const unsigned char low =
                        shared_validity[column_lane][row_lane];
                    unsigned char high = 0U;
                    bool partner_is_in_this_batch = false;
                    if (cell + 1U < rows * destination_columns) {
                        const std::size_t partner_row =
                            destination_column + 1U < destination_columns
                                ? row
                                : row + 1U;
                        const std::size_t partner_destination_column =
                            destination_column + 1U < destination_columns
                                ? destination_column + 1U
                                : 0U;
                        partner_is_in_this_batch =
                            partner_destination_column >=
                                destination_column_start &&
                            partner_destination_column <
                                destination_column_start + source_columns;
                        if (partner_is_in_this_batch) {
                            const std::size_t partner_source_column =
                                partner_destination_column -
                                destination_column_start;
                            if (partner_row == row &&
                                partner_source_column == source_column + 1U &&
                                column_lane + 1U < kLayoutTileColumns) {
                                high = shared_validity[column_lane + 1U]
                                                      [row_lane];
                            } else {
                                const unsigned char* partner_validity =
                                    reinterpret_cast<const unsigned char*>(
                                        static_cast<std::uintptr_t>(
                                            source_validity_addresses[
                                                partner_source_column]));
                                high = partner_validity[
                                    static_cast<std::size_t>(
                                        source_validity_offsets[
                                            partner_source_column]) +
                                    partner_row];
                            }
                        }
                    }
                    if (partner_is_in_this_batch && (low > 9U || high > 9U)) {
                        atomicExch(validity_code_error, 1U);
                    }
                    if (partner_is_in_this_batch) {
                        search_bar_major_validity_u4[cell / 2U] =
                            static_cast<unsigned char>(
                                (low & 0x0fU) | ((high & 0x0fU) << 4U));
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ void pack_batch_boundary_validity_u4_v3(
    const std::uint64_t* source_validity_addresses,
    const std::uint64_t* source_validity_offsets,
    std::size_t rows,
    std::size_t source_columns,
    std::size_t destination_columns,
    std::size_t destination_column_start,
    unsigned char* search_bar_major_validity_u4,
    unsigned int* validity_code_error) {
    const std::size_t candidates_per_row = source_columns == 1U ? 1U : 2U;
    const std::size_t candidate_count = rows * candidates_per_row;
    std::size_t candidate =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    while (candidate < candidate_count) {
        const std::size_t row = candidate / candidates_per_row;
        const std::size_t boundary = candidate % candidates_per_row;
        const std::size_t source_column =
            boundary == 0U ? 0U : source_columns - 1U;
        const std::size_t destination_column =
            destination_column_start + source_column;
        const std::size_t cell = row * destination_columns + destination_column;
        const std::size_t pair_cell = (cell & 1U) == 0U ? cell + 1U : cell - 1U;
        const bool has_pair = pair_cell < rows * destination_columns;
        const std::size_t pair_destination_column =
            has_pair ? pair_cell % destination_columns : destination_column;
        const bool pair_is_in_this_batch =
            has_pair &&
            pair_destination_column >= destination_column_start &&
            pair_destination_column <
                destination_column_start + source_columns;
        if (!pair_is_in_this_batch) {
            const unsigned char* source_validity =
                reinterpret_cast<const unsigned char*>(
                    static_cast<std::uintptr_t>(
                        source_validity_addresses[source_column]));
            const unsigned char code =
                source_validity[static_cast<std::size_t>(
                                    source_validity_offsets[source_column]) +
                                row];
            if (code > 9U) {
                atomicExch(validity_code_error, 1U);
            }
            const std::size_t byte_index = cell / 2U;
            const std::size_t word_index = byte_index / sizeof(unsigned int);
            const unsigned int byte_in_word =
                static_cast<unsigned int>(byte_index % sizeof(unsigned int));
            const unsigned int nibble_in_byte =
                static_cast<unsigned int>(cell & 1U);
            const unsigned int shift = byte_in_word * 8U + nibble_in_byte * 4U;
            atomicOr(reinterpret_cast<unsigned int*>(
                         search_bar_major_validity_u4) +
                         word_index,
                     static_cast<unsigned int>(code & 0x0fU) << shift);
        }
        candidate += stride;
    }
}

__global__ void canonical_feature_merkle_leaf_sha256_v3(
    const std::int64_t* timestamps,
    std::size_t rows,
    std::size_t columns,
    const std::uint64_t* name_offsets,
    const unsigned char* name_bytes,
    const std::uint64_t* search_bar_major_value_bits,
    const unsigned char* search_bar_major_validity_u4,
    std::size_t timestamp_chunk_count,
    std::size_t leaf_count,
    unsigned char* leaf_digests) {
    std::size_t leaf =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    while (leaf < leaf_count) {
        const bool is_timestamp = leaf < timestamp_chunk_count;
        const std::size_t feature_leaf =
            is_timestamp ? 0 : leaf - timestamp_chunk_count;
        const std::size_t column =
            is_timestamp ? 0 : feature_leaf % columns;
        const std::size_t chunk =
            is_timestamp ? leaf : feature_leaf / columns;
        const std::size_t row_start = chunk * kCanonicalMerkleChunkRowsV3;
        const std::size_t rows_left = rows - row_start;
        const std::size_t row_count =
            rows_left < kCanonicalMerkleChunkRowsV3
                ? rows_left
                : kCanonicalMerkleChunkRowsV3;

        Sha256StateV1 state{};
        sha256_initialize(&state);
        sha256_update_bytes(&state, kCanonicalMerkleLeafDomainV3,
                            sizeof(kCanonicalMerkleLeafDomainV3));
        sha256_update_byte(&state, is_timestamp ? 0U : 1U);
        sha256_update_u64_le(&state, static_cast<std::uint64_t>(leaf));
        sha256_update_u64_le(&state, static_cast<std::uint64_t>(row_start));
        sha256_update_u64_le(&state, static_cast<std::uint64_t>(row_count));
        if (is_timestamp) {
            for (std::size_t row = row_start; row < row_start + row_count;
                 ++row) {
                sha256_update_u64_le(
                    &state, static_cast<std::uint64_t>(timestamps[row]));
            }
        } else {
            sha256_update_u64_le(&state, static_cast<std::uint64_t>(column));
            const std::uint64_t name_start = name_offsets[column];
            const std::uint64_t name_end = name_offsets[column + 1U];
            const std::uint64_t name_len = name_end - name_start;
            sha256_update_u64_le(&state, name_len);
            sha256_update_bytes(&state, name_bytes + name_start,
                                static_cast<std::size_t>(name_len));
            for (std::size_t row = row_start; row < row_start + row_count;
                 ++row) {
                const std::size_t cell = row * columns + column;
                sha256_update_u64_le(&state,
                                     search_bar_major_value_bits[cell]);
                const unsigned char packed_validity =
                    search_bar_major_validity_u4[cell / 2U];
                const unsigned char logical_validity =
                    (cell & 1U) == 0U
                        ? packed_validity & 0x0fU
                        : static_cast<unsigned char>(packed_validity >> 4U);
                sha256_update_byte(&state, logical_validity);
            }
        }
        sha256_finalize(&state, leaf_digests + leaf * 32U);
        leaf += stride;
    }
}

__global__ void canonical_feature_merkle_reduce_sha256_v3(
    const unsigned char* input_digests,
    std::size_t input_count,
    std::uint64_t level,
    unsigned char* output_digests) {
    const std::size_t output_count = input_count / 2U + input_count % 2U;
    std::size_t node =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    while (node < output_count) {
        const std::size_t left = node * 2U;
        const bool has_right = left + 1U < input_count;
        Sha256StateV1 state{};
        sha256_initialize(&state);
        sha256_update_bytes(&state, kCanonicalMerkleNodeDomainV3,
                            sizeof(kCanonicalMerkleNodeDomainV3));
        sha256_update_u64_le(&state, level);
        sha256_update_u64_le(&state, static_cast<std::uint64_t>(node));
        sha256_update_byte(&state, has_right ? 2U : 1U);
        sha256_update_bytes(&state, input_digests + left * 32U, 32U);
        if (has_right) {
            sha256_update_bytes(&state, input_digests + (left + 1U) * 32U,
                                32U);
        }
        sha256_finalize(&state, output_digests + node * 32U);
        node += stride;
    }
}

__global__ void canonical_feature_merkle_root_sha256_v3(
    const unsigned char* tree_root,
    std::size_t rows,
    std::size_t columns,
    std::size_t timestamp_chunk_count,
    std::size_t leaf_count,
    unsigned char* digest) {
    if (threadIdx.x != 0U || blockIdx.x != 0U) {
        return;
    }
    Sha256StateV1 state{};
    sha256_initialize(&state);
    sha256_update_bytes(&state, kCanonicalMerkleRootDomainV3,
                        sizeof(kCanonicalMerkleRootDomainV3));
    sha256_update_u64_le(&state, static_cast<std::uint64_t>(rows));
    sha256_update_u64_le(&state, static_cast<std::uint64_t>(columns));
    sha256_update_u64_le(
        &state, static_cast<std::uint64_t>(kCanonicalMerkleChunkRowsV3));
    sha256_update_u64_le(&state,
                         static_cast<std::uint64_t>(timestamp_chunk_count));
    sha256_update_u64_le(&state, static_cast<std::uint64_t>(leaf_count));
    sha256_update_bytes(&state, tree_root, 32U);
    sha256_finalize(&state, digest);
}

int launch_status() {
    return static_cast<int>(cudaGetLastError());
}

int validate_current_device_grid(unsigned int grid_x, unsigned int grid_y) {
    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    if (grid_x == 0U || grid_y == 0U ||
        grid_x > static_cast<unsigned int>(properties.maxGridSize[0]) ||
        grid_y > static_cast<unsigned int>(properties.maxGridSize[1])) {
        return static_cast<int>(cudaErrorInvalidConfiguration);
    }
    return static_cast<int>(cudaSuccess);
}

}  // namespace

extern "C" int neoethos_resident_initialize_validity_u4_v3(
    unsigned char* search_bar_major_validity_u4,
    std::size_t logical_bytes,
    std::size_t allocated_bytes,
    unsigned int* validity_code_error,
    CUstream stream) {
    if (stream == nullptr || search_bar_major_validity_u4 == nullptr ||
        validity_code_error == nullptr || logical_bytes == 0U ||
        allocated_bytes < logical_bytes ||
        allocated_bytes % sizeof(unsigned int) != 0U) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t status = cudaMemsetAsync(
        search_bar_major_validity_u4, 0, allocated_bytes,
        reinterpret_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) {
        return static_cast<int>(status);
    }
    status = cudaMemsetAsync(validity_code_error, 0,
                             sizeof(unsigned int),
                             reinterpret_cast<cudaStream_t>(stream));
    return static_cast<int>(status);
}

extern "C" int neoethos_resident_pack_batch_to_bar_major_f64_u4_v3(
    const std::uint64_t* source_addresses,
    const std::uint64_t* source_offsets,
    const std::uint64_t* source_validity_addresses,
    const std::uint64_t* source_validity_offsets,
    std::size_t rows,
    std::size_t source_columns,
    std::size_t destination_columns,
    std::size_t destination_column_start,
    double* search_bar_major_values,
    unsigned char* search_bar_major_validity_u4,
    unsigned int* validity_code_error,
    CUstream stream) {
    if (stream == nullptr || source_addresses == nullptr ||
        source_offsets == nullptr || source_validity_addresses == nullptr ||
        source_validity_offsets == nullptr ||
        search_bar_major_values == nullptr ||
        search_bar_major_validity_u4 == nullptr ||
        validity_code_error == nullptr || rows == 0 || source_columns == 0 ||
        destination_columns == 0 ||
        destination_column_start > destination_columns ||
        source_columns > destination_columns - destination_column_start ||
        destination_columns >
            std::numeric_limits<std::size_t>::max() / rows) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const std::size_t column_tiles =
        (source_columns - 1U) / kLayoutTileColumns + 1U;
    const std::size_t row_tiles = (rows - 1U) / kLayoutTileRows + 1U;
    if (column_tiles > std::numeric_limits<unsigned int>::max()) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const unsigned int grid_rows = static_cast<unsigned int>(
        row_tiles < kMaxPortableBlocks ? row_tiles : kMaxPortableBlocks);
    const dim3 grid(static_cast<unsigned int>(column_tiles), grid_rows, 1U);
    const dim3 block(kLayoutTileColumns, kLayoutBlockRows, 1U);
    int status = validate_current_device_grid(grid.x, grid.y);
    if (status != static_cast<int>(cudaSuccess)) {
        return status;
    }
    pack_sources_to_bar_major_f64_u4_v3<<<
        grid, block, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        source_addresses, source_offsets, source_validity_addresses,
        source_validity_offsets, rows, source_columns, destination_columns,
        destination_column_start,
        reinterpret_cast<std::uint64_t*>(search_bar_major_values),
        search_bar_major_validity_u4, validity_code_error);
    status = launch_status();
    if (status != static_cast<int>(cudaSuccess)) {
        return status;
    }

    const std::size_t candidates_per_row = source_columns == 1U ? 1U : 2U;
    if (rows > std::numeric_limits<std::size_t>::max() /
                   candidates_per_row) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const std::size_t candidate_count = rows * candidates_per_row;
    constexpr std::size_t boundary_threads = 128U;
    const std::size_t boundary_blocks_required =
        candidate_count / boundary_threads +
        (candidate_count % boundary_threads != 0U ? 1U : 0U);
    const unsigned int boundary_blocks = static_cast<unsigned int>(
        boundary_blocks_required < kMaxPortableBlocks
            ? boundary_blocks_required
            : kMaxPortableBlocks);
    status = validate_current_device_grid(boundary_blocks, 1U);
    if (status != static_cast<int>(cudaSuccess)) {
        return status;
    }
    pack_batch_boundary_validity_u4_v3<<<
        boundary_blocks, boundary_threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        source_validity_addresses, source_validity_offsets, rows,
        source_columns, destination_columns, destination_column_start,
        search_bar_major_validity_u4, validity_code_error);
    return launch_status();
}

extern "C" int neoethos_resident_canonical_merkle_sha256_v3(
    const std::int64_t* timestamps,
    std::size_t rows,
    std::size_t columns,
    const std::uint64_t* name_offsets,
    const unsigned char* name_bytes,
    const double* search_bar_major_values,
    const unsigned char* search_bar_major_validity_u4,
    unsigned char* merkle_scratch_a,
    unsigned char* merkle_scratch_b,
    std::size_t merkle_scratch_digest_capacity,
    unsigned char* digest,
    CUstream stream) {
    if (stream == nullptr || timestamps == nullptr || name_offsets == nullptr ||
        name_bytes == nullptr || search_bar_major_values == nullptr ||
        search_bar_major_validity_u4 == nullptr ||
        merkle_scratch_a == nullptr || merkle_scratch_b == nullptr ||
        digest == nullptr || rows == 0 || columns == 0 ||
        columns > std::numeric_limits<std::size_t>::max() / rows) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const std::size_t timestamp_chunk_count =
        (rows - 1U) / kCanonicalMerkleChunkRowsV3 + 1U;
    if (columns == std::numeric_limits<std::size_t>::max()) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const std::size_t producer_count = columns + 1U;
    if (producer_count >
        std::numeric_limits<std::size_t>::max() / timestamp_chunk_count) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const std::size_t leaf_count = timestamp_chunk_count * producer_count;
    if (leaf_count > std::numeric_limits<std::size_t>::max() / 32U ||
        merkle_scratch_digest_capacity < leaf_count) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const std::size_t leaf_blocks_required =
        (leaf_count + kSha256Threads - 1U) / kSha256Threads;
    const unsigned int leaf_blocks = static_cast<unsigned int>(
        leaf_blocks_required < kMaxPortableBlocks ? leaf_blocks_required
                                                  : kMaxPortableBlocks);
    int status = validate_current_device_grid(leaf_blocks, 1U);
    if (status != static_cast<int>(cudaSuccess)) {
        return status;
    }
    canonical_feature_merkle_leaf_sha256_v3<<<
        leaf_blocks, kSha256Threads, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        timestamps, rows, columns, name_offsets, name_bytes,
        reinterpret_cast<const std::uint64_t*>(search_bar_major_values),
        search_bar_major_validity_u4, timestamp_chunk_count, leaf_count,
        merkle_scratch_a);
    status = launch_status();
    if (status != static_cast<int>(cudaSuccess)) {
        return status;
    }

    const unsigned char* current_input = merkle_scratch_a;
    unsigned char* next_output = merkle_scratch_b;
    std::size_t current_count = leaf_count;
    std::uint64_t level = 0U;
    while (current_count > 1U) {
        const std::size_t output_count =
            current_count / 2U + current_count % 2U;
        const std::size_t node_blocks_required =
            (output_count + kSha256Threads - 1U) / kSha256Threads;
        const unsigned int node_blocks = static_cast<unsigned int>(
            node_blocks_required < kMaxPortableBlocks ? node_blocks_required
                                                      : kMaxPortableBlocks);
        status = validate_current_device_grid(node_blocks, 1U);
        if (status != static_cast<int>(cudaSuccess)) {
            return status;
        }
        canonical_feature_merkle_reduce_sha256_v3<<<
            node_blocks, kSha256Threads, 0,
            reinterpret_cast<cudaStream_t>(stream)>>>(
            current_input, current_count, level, next_output);
        status = launch_status();
        if (status != static_cast<int>(cudaSuccess)) {
            return status;
        }
        current_count = output_count;
        current_input = next_output;
        next_output = next_output == merkle_scratch_a ? merkle_scratch_b
                                                      : merkle_scratch_a;
        ++level;
    }

    canonical_feature_merkle_root_sha256_v3<<<
        1U, 32U, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        current_input, rows, columns, timestamp_chunk_count, leaf_count,
        digest);
    return launch_status();
}
