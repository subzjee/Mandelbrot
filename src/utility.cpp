/*
 * This file contains the implementations for the utility functions.
 */

#include "utility.hpp"

using namespace utility;

#if defined(__AVX__)
__m256 avx::mapRowToImagAxis(const std::size_t row, const std::size_t height,
                             const float imag_min, const float imag_max) {
  const float imag =
      utility::mapIndexToBoundedAxis(row, height, imag_max, imag_min);

  return _mm256_set1_ps(imag);
}

__m256 avx::mapColumnsToRealAxis(const std::size_t col, const std::size_t width,
                                 const float real_min, const float real_max) {
  const __m256 col_indices =
      _mm256_set_ps(static_cast<float>(col + 7), static_cast<float>(col + 6),
                    static_cast<float>(col + 5), static_cast<float>(col + 4),
                    static_cast<float>(col + 3), static_cast<float>(col + 2),
                    static_cast<float>(col + 1), static_cast<float>(col + 0));

  const float real_scale =
      (real_max - real_min) / (static_cast<float>(width - 1));

  const __m256 reals = _mm256_fmadd_ps(col_indices, _mm256_set1_ps(real_scale),
                                       _mm256_set1_ps(real_min));

  return reals;
}

std::pair<__m256, __m256>
avx::mapPixelsToComplexPlane(const std::size_t row, const std::size_t col,
                             const std::size_t width, const std::size_t height,
                             const float real_min, const float real_max,
                             const float imag_min, const float imag_max) {
  const __m256 reals =
      avx::mapColumnsToRealAxis(col, width, real_min, real_max);
  const __m256 imags = avx::mapRowToImagAxis(row, height, imag_min, imag_max);

  return {reals, imags};
}
#endif

#if defined(__AVX512F__)
__m512 avx512::mapRowToImagAxis(const std::size_t row, const std::size_t height,
                                const float imag_min, const float imag_max) {
  const float imag =
      utility::mapIndexToBoundedAxis(row, height, imag_max, imag_min);

  return _mm512_set1_ps(imag);
}

__m512 avx512::mapColumnsToRealAxis(const std::size_t col,
                                    const std::size_t width,
                                    const float real_min,
                                    const float real_max) {
  const __m512 col_indices =
      _mm512_set_ps(static_cast<float>(col + 15), static_cast<float>(col + 14),
                    static_cast<float>(col + 13), static_cast<float>(col + 12),
                    static_cast<float>(col + 11), static_cast<float>(col + 10),
                    static_cast<float>(col + 9), static_cast<float>(col + 8),
                    static_cast<float>(col + 7), static_cast<float>(col + 6),
                    static_cast<float>(col + 5), static_cast<float>(col + 4),
                    static_cast<float>(col + 3), static_cast<float>(col + 2),
                    static_cast<float>(col + 1), static_cast<float>(col + 0));

  const float real_scale =
      (real_max - real_min) / (static_cast<float>(width - 1));

  const __m512 reals = _mm512_fmadd_ps(col_indices, _mm512_set1_ps(real_scale),
                                       _mm512_set1_ps(real_min));

  return reals;
}

std::pair<__m512, __m512> avx512::mapPixelsToComplexPlane(
    const std::size_t row, const std::size_t col, const std::size_t width,
    const std::size_t height, const float real_min, const float real_max,
    const float imag_min, const float imag_max) {
  const __m512 reals =
      avx512::mapColumnsToRealAxis(col, width, real_min, real_max);
  const __m512 imags =
      avx512::mapRowToImagAxis(row, height, imag_min, imag_max);

  return {reals, imags};
}
#endif
