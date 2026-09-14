#pragma once

#include "resident_scoring_novelty_v1_abi.cuh"

namespace neoethos::resident_backend_math_v3 {

// Flags/algorithm identity only, not a claim of cross-device numerical parity.
// The separately bound build manifest identifies compiler, ROCm, target and
// device libraries. CUDA's existing V2 digest remains unchanged.
inline constexpr char kHipMathSemanticsV1[] =
    "neoethos.hip-math.v1;fast-math=false;fp-contract=off;"
    "denormal-fp-math=ieee;denormal-fp-math-f32=ieee;"
    "gpu-flush-denormals-to-zero=false;"
    "fp32-correctly-rounded-divide-sqrt=true;"
    "unsafe-fp-atomics=false;explicit-fma=preserved";
inline constexpr std::uint8_t kHipMathSemanticsSha256V1[32] = {
    0x71, 0x04, 0xe9, 0x27, 0xbf, 0xc4, 0x17, 0x92,
    0x71, 0x8e, 0x78, 0x3d, 0x19, 0x6e, 0x02, 0x72,
    0xe9, 0xee, 0xda, 0xe4, 0x67, 0xe6, 0x9d, 0xed,
    0x85, 0x5d, 0x83, 0xae, 0x2d, 0xf2, 0x57, 0x55};

inline const std::uint8_t* expected_math_semantics_sha256_v3() {
#if defined(__HIP_PLATFORM_AMD__)
  return kHipMathSemanticsSha256V1;
#else
  return resident_scoring_novelty_v1::NEO_RESIDENT_CUDA_MATH_SEMANTICS_SHA256_V2;
#endif
}

}  // namespace neoethos::resident_backend_math_v3
