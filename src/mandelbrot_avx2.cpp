/*
 * This file contains the AVX2 implementation.
 */

#if defined(__AVX2__)

#include <immintrin.h>

#include "mandelbrot_engine.hpp"
#include "utility.hpp"

/*
 * Check whether the AVX2 backend is available.
 *
 * @returns Whether the AVX2 backend is available.
 */
template<>
bool CPUEngine<Backend::AVX2>::is_available() const {
  return __builtin_cpu_supports("avx2");
}

/*
 * Compute the Mandelbrot set with AVX2 acceleration.
 *
 * @returns MandelbrotResult containing iteration and final z-value per pixel.
 */
template<>
MandelbrotResult CPUEngine<Backend::AVX2>::compute() {
  if (!is_available()) {
    throw std::runtime_error("AVX2 not supported on this CPU.");
  }

  constexpr std::size_t lanes = utility::avx::simd_width_bytes / sizeof(float);

  for (std::size_t row = 0; row < m_height; ++row) {
    for (std::size_t col = 0; col < m_width; col += lanes) {
      const auto [c_real, c_imag] =
          utility::avx::detail::mapPixelsToComplexPlane(
              row, col, m_width, m_height, m_bounds.real_min, m_bounds.real_max, m_bounds.imag_min, m_bounds.imag_max);

      __m256 z_real = _mm256_setzero_ps();
      __m256 z_imag = _mm256_setzero_ps();

      __m256i iter_counts = _mm256_setzero_si256();

      for (unsigned int i = 0; i < m_max_iterations; ++i) {
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

      const std::size_t base_idx = row * m_width + col;
      const std::size_t remaining = std::min(lanes, m_width - col);

      if (remaining == lanes) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&m_iterations[base_idx]),
                            iter_counts);
        _mm256_storeu_ps(&m_z_reals[base_idx], z_real);
        _mm256_storeu_ps(&m_z_imags[base_idx], z_imag);
      } else {
        // AVX2 doesn't have masked stores so we have to manually copy the
        // remaining elements if it doesn't fit perfectly in a lane.
        alignas(utility::avx::simd_width_bytes) int lane_iters[lanes];
        alignas(utility::avx::simd_width_bytes) float lane_real[lanes];
        alignas(utility::avx::simd_width_bytes) float lane_imag[lanes];

        _mm256_store_si256(reinterpret_cast<__m256i*>(lane_iters), iter_counts);
        _mm256_store_ps(lane_real, z_real);
        _mm256_store_ps(lane_imag, z_imag);

        for (std::size_t i = 0; i < std::min(lanes, m_width - col); ++i) {
          m_iterations[base_idx + i] = static_cast<unsigned int>(lane_iters[i]);
          m_z_reals[base_idx + i] = lane_real[i];
          m_z_imags[base_idx + i] = lane_imag[i];
        }
      }
    }
  }

  return {m_iterations, m_z_reals, m_z_imags, m_width,
          m_height};
}

#endif
