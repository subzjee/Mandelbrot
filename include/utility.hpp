#pragma once

#include <complex>
#include <vector>

#include <immintrin.h>

template <typename T> using Matrix = std::vector<std::vector<T>>;

using MandelbrotResult = Matrix<unsigned int>;

namespace utility {
namespace {
/*
 * Map an index in a 1D structure to a bounded axis linearly.
 *
 * @param idx The index.
 * @param num_spaces The number of spaces to linearly space to bounded axis
 * into.
 * @param start The start of the range.
 * @param end The end of the range.
 *
 * @returns The mapped coordinate on the axis.
 */
constexpr float mapIndexToBoundedAxis(std::size_t idx, std::size_t num_spaces,
                                      float start, float end) {
  return start +
         (static_cast<float>(idx) / static_cast<float>(num_spaces - 1)) *
             (end - start);
}
} // namespace

/*
 * Map a pixel in an image structured as a 1D array to the complex plane.
 *
 * @param row The row of the pixel.
 * @param col The column of the pixel.
 * @param height The height of the image.
 * @param width The width of the image.
 * @param real_min The lower bound of the real axis.
 * @param real_max The upper bound of the real axis.
 * @param imag_min The lower bound of the imaginary axis.
 * @param imag_max The upper bound of the imaginary axis.
 *
 * @returns The mapped position of the pixel on the complex plane.
 */
constexpr std::complex<float>
mapPixelToComplexPlane(std::size_t row, std::size_t col, std::size_t width,
                       std::size_t height, float real_min, float real_max,
                       float imag_min, float imag_max) {
  return {
      mapIndexToBoundedAxis(col, width, real_min, real_max), // Real axis
      mapIndexToBoundedAxis(row, height, imag_max, imag_min) // Imag axis
  };
}
} // namespace utility

#if defined(__AVX__)
namespace utility::avx {
constexpr unsigned int simd_width = 256; // The SIMD width in bits.
constexpr unsigned int simd_width_bytes = simd_width / 8;

/*
 * Map the columns of eight consecutive pixels in the same row starting at
 * column `col` to their real coordinates in the complex plane.
 *
 * @param col The column of the first pixel.
 * @param width The width of the image.
 * @param real_min The lower bound of the real axis.
 * @param real_max The upper bound of the real axis.
 *
 * @returns The real coordinates.
 */
__m256 mapColumnsToRealAxis(std::size_t col, std::size_t width, float real_min,
                            float real_max);

/*
 * Map the row of eight consecutive pixels in the same row to the imaginary
 * axis.
 *
 * Due the pixels being in the same row, the imaginary
 * coordinate will be identical for each pixel.
 *
 * @param row The row index.
 * @param height The height of the image.
 * @param imag_min The lower bound of the imaginary axis.
 * @param imag_max The upper bound of the imaginary axis.
 *
 * @returns The imaginary coordinates.
 */
__m256 mapRowToImagAxis(std::size_t row, std::size_t height, float imag_min,
                        float imag_max);

/*
 * Map eight consecutive pixels in the same row onto the complex plane.
 *
 * @param row The row of the pixels.
 * @param col The column of the first pixel.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param real_min The lower bound of the real axis.
 * @param real_max The upper bound of the real axis.
 * @param imag_min The lower bound of the imaginary axis.
 * @param imag_max The upper bound of the imaginary axis.
 *
 * @returns The mapped positions of the pixels on the complex plane.
 */
std::pair<__m256, __m256>
mapPixelsToComplexPlane(std::size_t row, std::size_t col, std::size_t width,
                        std::size_t height, float real_min, float real_max,
                        float imag_min, float imag_max);

/*
 * Calculate the norms of multiple complex numbers where the real and imaginary
 * parts are represented by two separate AVX registers.
 *
 * The norm of a complex number (a, bi) is a^2 + b^2.
 *
 * @param real The real parts.
 * @param imag The imaginary parts.
 *
 * @returns The norms.
 */
static inline __m256 norm(__m256 real, __m256 imag) {
  return _mm256_add_ps(_mm256_mul_ps(real, real), _mm256_mul_ps(imag, imag));
}
} // namespace utility::avx
#endif

#if defined(__AVX512F__)
namespace utility::avx512 {
constexpr unsigned int simd_width = 512; // The SIMD width in bits.
constexpr unsigned int simd_width_bytes = simd_width / 8;

/*
 * Map the columns of sixteen consecutive pixels in the same row starting at
 * column `col` to their real coordinates in the complex plane.
 *
 * @param col The column of the first pixel.
 * @param width The width of the image.
 * @param real_min The lower bound of the real axis.
 * @param real_max The upper bound of the real axis.
 *
 * @returns The real coordinates.
 */
__m512 mapColumnsToRealAxis(std::size_t col, std::size_t width, float real_min,
                            float real_max);

/*
 * Map the row of sixteen consecutive pixels in the same row to the imaginary
 * axis.
 *
 * Due the pixels being in the same row, the imaginary
 * coordinate will be identical for each pixel.
 *
 * @param row The row index.
 * @param height The height of the image.
 * @param imag_min The lower bound of the imaginary axis.
 * @param imag_max The upper bound of the imaginary axis.
 *
 * @returns The imaginary coordinates.
 */
__m512 mapRowToImagAxis(std::size_t row, std::size_t height, float imag_min,
                        float imag_max);

/*
 * Map sixteen consecutive pixels in the same row onto the complex plane.
 *
 * @param row The row of the pixels.
 * @param col The column of the first pixel.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param real_min The lower bound of the real axis.
 * @param real_max The upper bound of the real axis.
 * @param imag_min The lower bound of the imaginary axis.
 * @param imag_max The upper bound of the imaginary axis.
 *
 * @returns The mapped positions of the pixels on the complex plane.
 */
std::pair<__m512, __m512>
mapPixelsToComplexPlane(std::size_t row, std::size_t col, std::size_t width,
                        std::size_t height, float real_min, float real_max,
                        float imag_min, float imag_max);

/*
 * Calculate the norms of multiple complex numbers where the real and imaginary
 * parts are represented by two separate AVX registers.
 *
 * The norm of a complex number (a, bi) is a^2 + b^2.
 *
 * @param real The real parts.
 * @param imag The imaginary parts.
 *
 * @returns The norms.
 */
static inline __m512 norm(__m512 real, __m512 imag) {
  return _mm512_add_ps(_mm512_mul_ps(real, real), _mm512_mul_ps(imag, imag));
}
} // namespace utility::avx512
#endif
