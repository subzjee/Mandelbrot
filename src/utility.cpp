#include "utility.hpp"

using namespace utility;

#if defined(__AVX__)
__m256 avx::mapRowToImagAxis(std::size_t row, std::size_t height,
                                  float imag_min, float imag_max) {
  float imag = utility::mapIndexToBoundedAxis(row, height, imag_max, imag_min);
  return _mm256_set1_ps(imag);
}

__m256 avx::mapColumnsToRealAxis(std::size_t col, std::size_t width,
                                 float real_min, float real_max) {
  __m256 col_indices =
      _mm256_set_ps(static_cast<float>(col + 7), static_cast<float>(col + 6),
                    static_cast<float>(col + 5), static_cast<float>(col + 4),
                    static_cast<float>(col + 3), static_cast<float>(col + 2),
                    static_cast<float>(col + 1), static_cast<float>(col + 0));

  float real_scale = (real_max - real_min) / (static_cast<float>(width - 1));

  __m256 reals = _mm256_add_ps(_mm256_mul_ps(col_indices, _mm256_set1_ps(real_scale)), _mm256_set1_ps(real_min));

  return reals;
}

std::pair<__m256, __m256> avx::mapPixelsToComplexPlane(
    std::size_t row, std::size_t col, std::size_t width, std::size_t height,
    float real_min, float real_max, float imag_min, float imag_max) {
  __m256 reals = avx::mapColumnsToRealAxis(col, width, real_min, real_max);
  __m256 imags = avx::mapRowToImagAxis(row, height, imag_min, imag_max);
  return {reals, imags};
}
#endif
