#pragma once

// Keep scratch-size queries and execution on the same native implementation.
// This is a private namespace selection, not an alias in either vendor's namespace.
#if defined(__HIP_PLATFORM_AMD__)
#include <hipcub/hipcub.hpp>
namespace neoethos_parallel_primitives_v1 = ::hipcub;
#else
#include <cub/cub.cuh>
namespace neoethos_parallel_primitives_v1 = ::cub;
#endif
