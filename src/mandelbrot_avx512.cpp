/*
 * This file contains the AVX512 implementation.
 */

#if defined(__AVX512F__)

#include <immintrin.h>

#include "mandelbrot.hpp"
#include "mandelbrot_result.hpp"
#include "utility.hpp"

MandelbrotResult mandelbrot_avx512(const std::size_t width,
                                   const std::size_t height,
                                   const float real_min, const float real_max,
                                   const float imag_min, const float imag_max,
                                   const unsigned int max_iterations) {
  if (!__builtin_cpu_supports("avx512f")) {
    throw std::runtime_error("AVX512 not supported on this CPU.");
  }

  constexpr std::size_t lanes =
      utility::avx512::simd_width_bytes / sizeof(float);

  auto iterations = std::make_unique<unsigned int[]>(width * height);
  auto z_reals = std::make_unique<float[]>(width * height);
  auto z_imags = std::make_unique<float[]>(width * height);

  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; col += lanes) {
      const auto [c_real, c_imag] =
          utility::avx512::detail::mapPixelsToComplexPlane(
              row, col, width, height, real_min, real_max, imag_min, imag_max);

      __m512 z_real = _mm512_setzero_ps();
      __m512 z_imag = _mm512_setzero_ps();

      __m512i iter_counts = _mm512_setzero_epi32();

      for (unsigned int i = 0; i < max_iterations; ++i) {
        const __m512 norm = utility::avx512::detail::norm(z_real, z_imag);

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

      const std::size_t base_idx = row * width + col;
      const std::size_t remaining = std::min(lanes, width - col);
      const __mmask16 store_mask =
          (remaining == lanes) ? 0xFFFF : (1 << remaining) - 1;

      _mm512_mask_storeu_ps(&z_reals[base_idx], store_mask, z_real);
      _mm512_mask_storeu_ps(&z_imags[base_idx], store_mask, z_imag);
      _mm512_mask_storeu_epi32(&iterations[base_idx], store_mask, iter_counts);
    }
  }

  return {std::move(iterations), std::move(z_reals), std::move(z_imags), width,
          height};
}

#endif
