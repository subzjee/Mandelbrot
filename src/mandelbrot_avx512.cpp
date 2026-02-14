/*
 * This file contains the AVX512 implementation of the algorithm.
 *
 * Note: The implementation will fallback to the serial implementation, in case
 * that the current CPU does not support AVX512 instructions.
 */

#if defined(__AVX512F__)

#include <iostream>

#include <immintrin.h>

#include "mandelbrot.hpp"
#include "utility.hpp"

MandelbrotResult mandelbrot_avx512(const std::size_t width,
                                   const std::size_t height,
                                   const float real_min, const float real_max,
                                   const float imag_min, const float imag_max,
                                   const unsigned int max_iterations) {
  if (!__builtin_cpu_supports("avx512f")) {
    std::cerr << "AVX512F not supported on this CPU. Falling back to serial "
                 "version.\n";
    return mandelbrot_serial(width, height, real_min, real_max, imag_min,
                             imag_max, max_iterations);
  }

  constexpr std::size_t lanes =
      utility::avx512::simd_width_bytes / sizeof(float);

  MandelbrotResult result(height, std::vector<unsigned int>(width, 0));

  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; col += lanes) {
      const auto [c_real, c_imag] = utility::avx512::mapPixelsToComplexPlane(
          row, col, width, height, real_min, real_max, imag_min, imag_max);

      __m512 z_real = _mm512_setzero_ps();
      __m512 z_imag = _mm512_setzero_ps();

      __m512i iter_counts = _mm512_setzero_si512();

      for (unsigned int i = 0; i < max_iterations; ++i) {
        const __m512 norm = utility::avx512::norm(z_real, z_imag);

        // Check which pixels have not escaped yet.
        const __mmask16 active =
            _mm512_cmple_ps_mask(norm, _mm512_set1_ps(4.0f));

        // If all pixels have escaped, stop early.
        if (active == 0) {
          break;
        }

        iter_counts = _mm512_mask_add_epi32(iter_counts, active, iter_counts,
                                            _mm512_set1_epi32(1));

        // Calculate the new real parts.
        const __m512 z_real_new =
            _mm512_add_ps(_mm512_sub_ps(_mm512_mul_ps(z_real, z_real),
                                        _mm512_mul_ps(z_imag, z_imag)),
                          c_real);

        // Calculate the new imaginary parts.
        const __m512 z_imag_new =
            _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(z_real, z_imag),
                                        _mm512_mul_ps(z_real, z_imag)),
                          c_imag);

        // Only update the real and imaginary parts for active pixels.
        z_real = _mm512_mask_blend_ps(active, z_real, z_real_new);
        z_imag = _mm512_mask_blend_ps(active, z_imag, z_imag_new);
      }

      alignas(utility::avx512::simd_width_bytes) int lane_iters[lanes];
      _mm512_store_si512(reinterpret_cast<__m512i*>(lane_iters), iter_counts);

      for (std::size_t i = 0; i < std::min(lanes, width - col); ++i) {
        result[row][col + i] = static_cast<unsigned int>(lane_iters[i]);
      }
    }
  }

  return result;
}

#endif
