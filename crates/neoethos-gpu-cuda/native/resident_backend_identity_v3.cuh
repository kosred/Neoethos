#pragma once

// Source-sharing accessors only. They do not mint an identity or substitute a
// HIP lease for a CUDA context. Each backend has distinctly named wire members
// and its population owner validates the actual runtime authority first.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define NEO_RESIDENT_IDENTITY_HD_V3 __host__ __device__
#else
#define NEO_RESIDENT_IDENTITY_HD_V3
#endif

namespace neoethos::resident_backend_identity_v3 {

template <class T>
NEO_RESIDENT_IDENTITY_HD_V3 decltype(auto) selected_device_ordinal(T& value) {
#if defined(__HIP_PLATFORM_AMD__)
  return (value.selected_hip_ordinal);
#else
  return (value.selected_cuda_ordinal);
#endif
}
template <class T>
NEO_RESIDENT_IDENTITY_HD_V3 decltype(auto) device_identity(T& value) {
#if defined(__HIP_PLATFORM_AMD__)
  return (value.hip_device_identity_sha256);
#else
  return (value.cuda_device_identity_sha256);
#endif
}
template <class T>
NEO_RESIDENT_IDENTITY_HD_V3 decltype(auto) owner_identity(T& value) {
#if defined(__HIP_PLATFORM_AMD__)
  return (value.hip_lease_identity_sha256);
#else
  return (value.primary_context_identity_sha256);
#endif
}
template <class T>
NEO_RESIDENT_IDENTITY_HD_V3 decltype(auto) build_identity(T& value) {
#if defined(__HIP_PLATFORM_AMD__)
  return (value.hip_build_manifest_sha256);
#else
  return (value.cuda_build_manifest_sha256);
#endif
}
template <class T>
NEO_RESIDENT_IDENTITY_HD_V3 decltype(auto) math_identity(T& value) {
#if defined(__HIP_PLATFORM_AMD__)
  return (value.hip_math_flags_sha256);
#else
  return (value.cuda_math_flags_sha256);
#endif
}
template <class T>
inline decltype(auto) archive_owner_identity(T& value) {
#if defined(__HIP_PLATFORM_AMD__)
  return (value.hip_lease_identity);
#else
  return (value.primary_context_identity);
#endif
}
template <class T>
inline decltype(auto) archive_build_identity(T& value) {
#if defined(__HIP_PLATFORM_AMD__)
  return (value.hip_build_identity);
#else
  return (value.cuda_build_identity);
#endif
}
template <class T>
inline bool archive_backend_valid(const T& value) {
#if defined(__HIP_PLATFORM_AMD__)
  return value.backend_kind == 2u;
#else
  return value.reserved == 0u;
#endif
}

}  // namespace neoethos::resident_backend_identity_v3

#undef NEO_RESIDENT_IDENTITY_HD_V3
