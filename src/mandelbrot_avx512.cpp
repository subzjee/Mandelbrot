/*
 * This file contains the AVX512 implementation.
 */

#if defined(MANDELBROT_HAS_AVX512)

#include <immintrin.h>

#include "backends.hpp"
#include "mandelbrot_engine.hpp"
#include "utility.hpp"

/*
 * Compute the Mandelbrot set with AVX512 acceleration.
 *
 * @returns MandelbrotResult containing iteration and final z-value per pixel.
 */
template <>
MandelbrotResult<backend::AVX512> MandelbrotEngine<backend::AVX512>::compute() {
  constexpr std::size_t lanes = backend::AVX512::alignment / sizeof(float);

  for (std::size_t row = 0; row < m_height; ++row) {
    for (std::size_t col = 0; col < m_width; col += lanes) {
      const auto [c_real, c_imag] = utility::avx512::mapPixelsToComplexPlane(
          row, col, m_width, m_height, m_bounds.real_min, m_bounds.real_max,
          m_bounds.imag_min, m_bounds.imag_max);

      __m512 z_real = _mm512_setzero_ps();
      __m512 z_imag = _mm512_setzero_ps();

      __m512i iter_counts = _mm512_setzero_epi32();

      for (unsigned int i = 0; i < m_max_iterations; ++i) {
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

      const std::size_t base_idx = row * m_width + col;
      const std::size_t remaining = std::min(lanes, m_width - col);
      const __mmask16 store_mask =
          (remaining == lanes) ? 0xFFFF : (1 << remaining) - 1;

      _mm512_mask_store_ps(&m_host.z_reals[base_idx], store_mask, z_real);
      _mm512_mask_store_ps(&m_host.z_imags[base_idx], store_mask, z_imag);
      _mm512_mask_store_epi32(&m_host.iterations[base_idx], store_mask,
                              iter_counts);
    }
  }

  return {m_host, m_width, m_height};
}

#endif
