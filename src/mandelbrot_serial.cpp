/*
 * This file contains the serial implementation.
 */

#include <complex>

#include "mandelbrot.hpp"
#include "mandelbrot_result.hpp"
#include "utility.hpp"

MandelbrotResult mandelbrot_serial(const std::size_t width,
                                   const std::size_t height,
                                   const float real_min, const float real_max,
                                   const float imag_min, const float imag_max,
                                   const unsigned int max_iterations) {
  auto iterations = std::make_unique<unsigned int[]>(width * height);
  auto z_imag = std::make_unique<float[]>(width * height);
  auto z_real = std::make_unique<float[]>(width * height);

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
      z_real[row * width + col] = z.real();
      z_imag[row * width + col] = z.imag();
    }
  }

  return {std::move(iterations), std::move(z_real), std::move(z_imag), width,
          height};
}
