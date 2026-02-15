/*
 * This file contains the OpenMP implementation of the algorithm.
 */

#if defined(_OPENMP)

#include <complex>

#include "mandelbrot.hpp"
#include "mandelbrot_result.hpp"
#include "utility.hpp"

MandelbrotResult mandelbrot_omp(const std::size_t width,
                                const std::size_t height, const float real_min,
                                const float real_max, const float imag_min,
                                const float imag_max,
                                const unsigned int max_iterations) {
  std::unique_ptr<unsigned int[]> iterations = std::make_unique<unsigned int[]>(width * height);

#pragma omp parallel for collapse(2) schedule(guided)
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      std::complex<float> z{0.0, 0.0};
      const std::complex<float> c = utility::detail::mapPixelToComplexPlane(
          row, col, width, height, real_min, real_max, imag_min, imag_max);

      unsigned int iteration{0};
      while (std::norm(z) <= 4.0 && iteration < max_iterations) {
        z = z * z + c;

        ++iteration;
      }

      iterations[row * width + col] = iteration;
    }
  }

  return {std::move(iterations), width, height};
}

#endif
