#if defined(__AVX2__)

#include <immintrin.h>

#include "mandelbrot.hpp"
#include "utility.hpp"

MandelbrotResult mandelbrot_avx2(std::size_t width, std::size_t height,
                           float real_min, float real_max, float imag_min,
                           float imag_max, unsigned int max_iterations) {
  constexpr std::size_t lanes = utility::avx::simd_width_bytes / sizeof(float);

  MandelbrotResult result(height, std::vector<unsigned int>(width, 0));

  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; col += lanes) {
      const auto [c_real, c_imag] = utility::avx::mapPixelsToComplexPlane(
          row, col, width, height, real_min, real_max, imag_min, imag_max);

      __m256 z_real = _mm256_setzero_ps();
      __m256 z_imag = _mm256_setzero_ps();

      __m256i iter_counts = _mm256_setzero_si256();

      for (unsigned int i = 0; i < max_iterations; ++i) {
        __m256 norm = utility::avx::norm(z_real, z_imag);

        // Check which pixels have not escaped yet.
        __m256 active = _mm256_cmp_ps(norm, _mm256_set1_ps(4.0f), _CMP_LE_OS);

        // If all pixels have escaped, stop early.
        if (_mm256_movemask_ps(active) == 0)
          break;

        __m256i iter_inc =
            _mm256_castps_si256(active);

        // Only update the iteration count for active pixels.
        iter_counts = _mm256_add_epi32(
            iter_counts, _mm256_and_si256(iter_inc, _mm256_set1_epi32(1)));

        // Calculate the new real parts.
        __m256 z_real_new =
            _mm256_add_ps(_mm256_sub_ps(_mm256_mul_ps(z_real, z_real),
                                        _mm256_mul_ps(z_imag, z_imag)),
                          c_real);

        // Calculate the new imaginary parts.
        __m256 z_imag_new =
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
        result[row][col + i] = static_cast<unsigned int>(lane_iters[i]);
      }
    }
  }

  return result;
}

#endif
