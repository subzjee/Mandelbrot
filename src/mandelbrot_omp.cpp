#include <complex>

#include "mandelbrot_omp.hpp"
#include "utility.hpp"

void mandelbrotOMP(std::uint8_t *output, std::size_t width, std::size_t height,
                   float x_min, float x_max, float y_min, float y_max,
                   unsigned int max_iterations) {
#pragma omp parallel for collapse(2) schedule(guided)
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      std::complex<float> z{0.0, 0.0};
      std::complex<float> c = mapPixelToComplexPlane(row, col, width, height,
                                                    x_min, x_max, y_min, y_max);

      unsigned int iteration{0};
      while (std::norm(z) <= 4.0 && iteration < max_iterations) {
        z = z * z + c;
        ++iteration;
      }

      output[row * width + col] = static_cast<uint8_t>(
          255.0 * static_cast<float>(iteration) / max_iterations);
    }
  }
}
