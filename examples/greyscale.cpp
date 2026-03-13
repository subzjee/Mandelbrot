/*
 * This example will build a greyscale 1920x1080 image of the Mandelbrot set.
 * The complex plane is bounded by [-1.252213542, -1.22213542] on the real axis
 * and [0.108567708, 0.125442708] on the imaginary axis.
 * The maximum iterations for each pixel is set to 1000.
 *
 * Note: The classic example of a greyscale Mandelbrot visualization colors
 * non-escaped pixels black. This example inverts that by making them white. I
 * found it to be easier to discern details using this color scheme.
 */

#include <opencv2/opencv.hpp>

#include "mandelbrot_renderer.hpp"
#include "mandelbrot_result.hpp"

constexpr std::size_t width = 1920, height = 1080;
constexpr int max_iterations = 1000;
constexpr float
    real_min = -1.252213542f,
    real_max =
        -1.22213542f; // The bounds of the real axis on the complex plane.
constexpr float
    imag_min = 0.108567708f,
    imag_max =
        0.125442708f; // The bounds of the imaginary axis on the complex plane.

int main() {
  cv::Mat pixels(height, width, CV_8UC1);

  auto renderer = create_renderer(width, height, {real_min, real_max, imag_min, imag_max}, max_iterations);
  MandelbrotResult result = renderer->render();

  for (std::size_t row = 0; row < pixels.rows; ++row) {
    for (std::size_t col = 0; col < pixels.cols; ++col) {
      unsigned int iteration = result(row, col).iteration;
      // The OpenCV material format uses unsigned 8-bit single-channel colors.
      // Therefore, we convert it to `uchar`.
      pixels.at<uchar>(row, col) = static_cast<uchar>(
          255.0f * static_cast<float>(iteration) / max_iterations);
    }
  }

  cv::imwrite("greyscale.png", pixels);

  return 0;
}
