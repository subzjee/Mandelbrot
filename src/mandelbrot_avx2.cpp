/*
 * This file contains the AVX2 implementation of the algorithm.
 *
 * Note: The implementation will fallback to the serial implementation, in case
 * that the current CPU does not support AVX2 instructions.
 */

#if defined(__AVX2__)

#include <iostream>

#include <immintrin.h>

#include "mandelbrot.hpp"
#include "mandelbrot_result.hpp"
#include "utility.hpp"

MandelbrotResult mandelbrot_avx2(const std::size_t width,
                                 const std::size_t height, const float real_min,
                                 const float real_max, const float imag_min,
                                 const float imag_max,
                                 const unsigned int max_iterations) {
  if (!__builtin_cpu_supports("avx2")) {
    std::cerr
        << "AVX2 not supported on this CPU. Falling back to serial version.\n";
    return mandelbrot_serial(width, height, real_min, real_max, imag_min,
                             imag_max, max_iterations);
  }

  constexpr std::size_t lanes = utility::avx::simd_width_bytes / sizeof(float);

  std::unique_ptr<unsigned int[]> iterations = std::make_unique<unsigned int[]>(width * height);

  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; col += lanes) {
      const auto [c_real, c_imag] = utility::avx::detail::mapPixelsToComplexPlane(
          row, col, width, height, real_min, real_max, imag_min, imag_max);

      __m256 z_real = _mm256_setzero_ps();
      __m256 z_imag = _mm256_setzero_ps();

      __m256i iter_counts = _mm256_setzero_si256();

      for (unsigned int i = 0; i < max_iterations; ++i) {
        const __m256 norm = utility::avx::detail::norm(z_real, z_imag);

        // Check which pixels have not escaped yet.
        const __m256 active =
            _mm256_cmp_ps(norm, _mm256_set1_ps(4.0f), _CMP_LE_OS);

        // If all pixels have escaped, stop early.
        if (_mm256_movemask_ps(active) == 0) {
          break;
        }

        const __m256i iter_inc = _mm256_castps_si256(active);

        // Only update the iteration count for active pixels.
        iter_counts = _mm256_add_epi32(
            iter_counts, _mm256_and_si256(iter_inc, _mm256_set1_epi32(1)));

        // Calculate the new real parts.
        const __m256 z_real_new =
            _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(z_real, z_real),
                                        _mm256_mul_ps(z_imag, z_imag)),
                          c_real);

        // Calculate the new imaginary parts.
        const __m256 z_imag_new =
            _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(z_real, z_imag),
                                        _mm256_mul_ps(z_real, z_imag)),
                          c_imag);

        // Only update the real and imaginary parts for active pixels.
        z_real = _mm256_blendv_ps(z_real, z_real_new, active);
        z_imag = _mm256_blendv_ps(z_imag, z_imag_new, active);
      }

      alignas(utility::avx::simd_width_bytes) int lane_iters[lanes];
      _mm256_store_si256(reinterpret_cast<__m256i*>(lane_iters), iter_counts);

      for (std::size_t i = 0; i < std::min(lanes, width - col); ++i) {
        iterations[row * width + col + i] = static_cast<unsigned int>(lane_iters[i]);
      }
    }
  }

  return {std::move(iterations), width, height};
}

#endif
