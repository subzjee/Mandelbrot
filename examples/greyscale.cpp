#include <opencv2/opencv.hpp>

#include "mandelbrot.hpp"

constexpr std::size_t width = 1920, height = 1080;
constexpr int max_iterations = 1000;
constexpr double real_min = -1.252213542, real_max = -1.22213542; // The bounds of the real axis on the complex plane.
constexpr double imag_min = 0.108567708, imag_max = 0.125442708; // The bounds of the imaginary axis on the complex plane.

int main() {
  MandelbrotResult iterations = mandelbrot_avx2_omp(width, height, real_min, real_max, imag_min, imag_max, max_iterations);

  // Setup the image.
  cv::Mat pixels(height, width, CV_8UC1);

  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      unsigned int iterations = iterations[row][col];
      // The OpenCV material format uses unsigned 8-bit single-channel colors. Therefore, we convert it to `uint8_t`.
      pixels.data[row * width + col] = static_cast<uint8_t>(255.0f * static_cast<float>(iterations) / max_iterations);
    }
  }

  // Save the image.
  cv::imwrite("greyscale_avx2_omp.png", pixels);

  return 0;
}
