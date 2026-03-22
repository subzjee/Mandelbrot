#pragma once

#include <string>

#if defined(MANDELBROT_HAS_CUDA)
#include <cuda_runtime.h>
#endif

enum class Backend {
  Serial,
#if defined(MANDELBROT_HAS_OMP)
  OMP,
#endif
#if defined(MANDELBROT_HAS_AVX2)
  AVX2,
#if defined(MANDELBROT_HAS_OMP)
  AVX2_OMP,
#endif
#endif
#if defined(MANDELBROT_HAS_AVX512)
  AVX512,
#if defined(MANDELBROT_HAS_OMP)
  AVX512_OMP,
#endif
#endif
#if defined(MANDELBROT_HAS_CUDA)
  CUDA,
#endif
};

inline std::string to_string(Backend backend) {
  switch (backend) {
  case Backend::Serial:
    return "Serial";
#if defined(MANDELBROT_HAS_OMP)
  case Backend::OMP:
    return "OMP";
#endif
#if defined(MANDELBROT_HAS_AVX2)
  case Backend::AVX2:
    return "AVX2";
#if defined(MANDELBROT_HAS_OMP)
  case Backend::AVX2_OMP:
    return "AVX2_OMP";
#endif
#endif
#if defined(MANDELBROT_HAS_AVX512)
  case Backend::AVX512:
    return "AVX512";
#if defined(MANDELBROT_HAS_OMP)
  case Backend::AVX512_OMP:
    return "AVX512_OMP";
#endif
#endif
#if defined(MANDELBROT_HAS_CUDA)
  case Backend::CUDA:
    return "CUDA";
#endif
  default:
    return "Unknown";
  }
}

inline bool is_available(Backend backend) {
  switch (backend) {
  case Backend::Serial:
    return true;
#if defined(MANDELBROT_HAS_OMP)
  case Backend::OMP:
    return true;
#endif
#if defined(MANDELBROT_HAS_AVX2)
  case Backend::AVX2:
    return __builtin_cpu_supports("avx2");
#if defined(MANDELBROT_HAS_OMP)
  case Backend::AVX2_OMP:
    return __builtin_cpu_supports("avx2");
#endif
#endif
#if defined(MANDELBROT_HAS_AVX512)
  case Backend::AVX512:
    return __builtin_cpu_supports("avx512f");
#if defined(MANDELBROT_HAS_OMP)
  case Backend::AVX512_OMP:
    return __builtin_cpu_supports("avx512f");
#endif
#endif
#if defined(MANDELBROT_HAS_CUDA)
  case Backend::CUDA:
  {
    int device_count{0};
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
  }
#endif
  default:
    return false;
  }
}
