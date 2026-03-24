#pragma once

#include <string_view>

#if defined(MANDELBROT_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace exec {
struct ExecBase {};

struct Default : ExecBase {
  static constexpr std::string_view name() { return "Default"; }
};

#if defined(MANDELBROT_HAS_OMP)
struct OMP : ExecBase {
  static constexpr std::string_view name() { return "OMP"; }
};
#endif
} // namespace exec

template <typename T>
concept Execution = std::is_base_of_v<exec::ExecBase, T>;

namespace backend {
struct BackendBase {};

struct Serial : BackendBase {
  /*
   * Get the name of the backend.
   *
   * @returns The name of the serial backend.
   */
  static constexpr std::string_view name() { return "Serial"; }

  /*
   * Check whether the current system supports the serial backend.
   *
   * As this serves as the baseline version, it will always be available.
   *
   * @returns Whether the serial backend is available.
   */
  static bool is_available() { return true; }

  template <Execution Exec>
  /*
   * Check whether this backend supports an execution policy.
   *
   * @tparam The execution policy.
   *
   * @returns Whether this backend supports the execution policy.
   */
  static constexpr bool supports_exec() {
    return true;
  }
};

#if defined(MANDELBROT_HAS_AVX2)
struct AVX2 : BackendBase {
  /*
   * Get the name of the backend.
   *
   * @returns The name of the AVX2 backend.
   */
  static constexpr std::string_view name() { return "AVX2"; }

  /*
   * Check whether the current system supports AVX2.
   *
   * @returns Whether the AVX2 backend is available.
   */
  static bool is_available() { return __builtin_cpu_supports("avx2"); }

  template <Execution Exec>
  /*
   * Check whether this backend supports an execution policy.
   *
   * @tparam The execution policy.
   *
   * @returns Whether this backend supports the execution policy.
   */
  static constexpr bool supports_exec() {
    return true;
  }

  static constexpr unsigned int simd_width = 256; // The SIMD width in bits.
  static constexpr unsigned int simd_width_bytes = simd_width / 8;
};
#endif

#if defined(MANDELBROT_HAS_AVX512)
struct AVX512 : BackendBase {
  /*
   * Get the name of the backend.
   *
   * @returns The name of the serial backend.
   */
  static constexpr std::string_view name() { return "AVX512"; }

  /*
   * Check whether the current system supports AVX512.
   *
   * @returns Whether the AVX512 backend is available.
   */
  static bool is_available() { return __builtin_cpu_supports("avx512f"); }

  template <Execution Exec>
  /*
   * Check whether this backend supports an execution policy.
   *
   * @tparam The execution policy.
   *
   * @returns Whether this backend supports the execution policy.
   */
  static constexpr bool supports_exec() {
    return true;
  }

  static constexpr unsigned int simd_width = 512; // The SIMD width in bits.
  static constexpr unsigned int simd_width_bytes = simd_width / 8;
};
#endif

#if defined(MANDELBROT_HAS_CUDA)
struct CUDA : BackendBase {
  /*
   * Get the name of the backend.
   *
   * @returns The name of the serial backend.
   */
  static constexpr std::string_view name() { return "CUDA"; }

  /*
   * Check whether the current system supports CUDA.
   *
   * @returns Whether CUDA backend is available.
   */
  static bool is_available() {
    int device_count{0};
    cudaError_t err = cudaGetDeviceCount(&device_count);

    return (err == cudaSuccess && device_count > 0);
  }

  template <Execution Exec>
  /*
   * Check whether this backend supports an execution policy.
   *
   * @tparam The execution policy.
   *
   * @returns Whether this backend supports the execution policy.
   */
  static constexpr bool supports_exec() {
    return std::is_same_v<Exec, exec::Default>;
  }
};
#endif
} // namespace backend

template <typename T>
concept Backend = std::is_base_of_v<backend::BackendBase, T>;

template <typename B, typename E>
concept Compatible =
    Backend<B> && Execution<E> && B::template supports_exec<E>();
