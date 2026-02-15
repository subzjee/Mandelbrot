/*
 * This file contains the implementations for the AVX512 utility functions.
 *
 * The header can be found in: include/utility.hpp
 */

#if defined(__AVX512F__)

#include "utility.hpp"

namespace utility::avx512::detail {
__m512 mapRowToImagAxis(const std::size_t row, const std::size_t height,
                                const float imag_min, const float imag_max) {
  const float imag =
      utility::detail::mapIndexToBoundedAxis(row, height, imag_max, imag_min);

  return _mm512_set1_ps(imag);
}

__m512 mapColumnsToRealAxis(const std::size_t col,
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

std::pair<__m512, __m512> mapPixelsToComplexPlane(
    const std::size_t row, const std::size_t col, const std::size_t width,
    const std::size_t height, const float real_min, const float real_max,
    const float imag_min, const float imag_max) {
  const __m512 reals =
      mapColumnsToRealAxis(col, width, real_min, real_max);
  const __m512 imags =
      mapRowToImagAxis(row, height, imag_min, imag_max);

  return {reals, imags};
}
}
#endif
