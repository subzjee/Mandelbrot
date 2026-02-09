#include <complex>
#include <vector>

#include "mandelbrot.hpp"

MandelbrotResult mandelbrot_serial(std::size_t width, std::size_t height,
                                   float real_min, float real_max,
                                   float imag_min, float imag_max,
                                   unsigned int max_iterations) {
  MandelbrotResult result(height, std::vector<unsigned int>(width, 0));

  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      std::complex<float> z{0.0, 0.0};
      const std::complex<float> c = utility::mapPixelToComplexPlane(
          row, col, width, height, real_min, real_max, imag_min, imag_max);

      unsigned int iteration{0};
      while (std::norm(z) <= 4.0 && iteration < max_iterations) {
        z = z * z + c;

        ++iteration;
      }

      result[row][col] = iteration;
    }
  }

  return result;
}
