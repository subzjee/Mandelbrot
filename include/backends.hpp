#pragma once

#include <string_view>

#if defined(MANDELBROT_HAS_CUDA)
#include <cuda_runtime.h>
#endif

namespace backend {
struct tag {};

struct serial : tag {
  static constexpr std::string_view name() {
    return "Serial";
  }

  static bool is_available() {
    return true;
  }
};

#if defined(MANDELBROT_HAS_OMP)
struct omp : tag {
  static constexpr std::string_view name() {
    return "OMP";
  }

  static bool is_available() {
    return true;
  }
};
#endif

#if defined(MANDELBROT_HAS_AVX2)
struct avx2 : tag {
  static constexpr std::string_view name() {
    return "AVX2";
  }

  static bool is_available() {
    return __builtin_cpu_supports("avx2");
  }

  static constexpr unsigned int simd_width = 256; // The SIMD width in bits.
  static constexpr unsigned int simd_width_bytes = simd_width / 8;
};

#if defined(MANDELBROT_HAS_OMP)
struct avx2_omp : avx2 {
  static constexpr std::string_view name() {
    return "AVX2_OMP";
  }
};
#endif
#endif

#if defined(MANDELBROT_HAS_AVX512)
struct avx512 : tag {
  static constexpr std::string_view name() {
    return "AVX512";
  }

  static bool is_available() {
    return __builtin_cpu_supports("avx512f");
  }

  static constexpr unsigned int simd_width = 512; // The SIMD width in bits.
  static constexpr unsigned int simd_width_bytes = simd_width / 8;
};

#if defined(MANDELBROT_HAS_OMP)
struct avx512_omp : avx512 {
  static constexpr std::string_view name() {
    return "AVX512_OMP";
  }
};
#endif
#endif

#if defined(MANDELBROT_HAS_CUDA)
struct cuda : tag {
  static constexpr std::string_view name() {
    return "CUDA";
  }

  static bool is_available() {
    int device_count{0};
    cudaError_t err = cudaGetDeviceCount(&device_count);

    return (err == cudaSuccess && device_count > 0);
  }
};
#endif
}

template <typename T>
concept Backend = std::is_base_of_v<backend::tag, T>;
