/*
 * This example will build a super-sampled RGB 1920x1080 image of the Mandelbrot
 * set. The SSAA factor has been set to 4. The complex plane is bounded by
 * [-1.252213542, -1.22213542] on the real axis and [0.108567708, 0.125442708]
 * on the imaginary axis. The maximum iterations for each pixel is set to 1000.
 */

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>

#include "mandelbrot.hpp"
#include "mandelbrot_result.hpp"

constexpr std::size_t width = 1920, height = 1080;
constexpr unsigned int ssaa_factor = 4;
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
  // Setup the image.
  cv::Mat pixels(height * ssaa_factor, width * ssaa_factor, CV_8UC3);

  MandelbrotResult iterations =
      mandelbrot_avx512_omp(pixels.cols, pixels.rows, real_min, real_max,
                            imag_min, imag_max, max_iterations);

  for (std::size_t row = 0; row < pixels.rows; ++row) {
    for (std::size_t col = 0; col < pixels.cols; ++col) {
      auto&& [iteration, z] = iterations[row][col];

      if (iteration == max_iterations) {
        pixels.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
      } else {
        float smooth = static_cast<float>(iteration) + 1.0f -
                       std::log2(std::log(std::abs(z)));
        float t = std::pow(smooth / max_iterations, 0.5f);

        float freq = 2.0f;
        float r = 0.5f + 0.5f * std::cos(6.28318f * (freq * t + 0.0f));
        float g = 0.5f + 0.5f * std::cos(6.28318f * (freq * t + 0.33f));
        float b = 0.5f + 0.5f * std::cos(6.28318f * (freq * t + 0.67f));

        float brightness = 0.7f + 0.3f * t;
        r *= brightness;
        g *= brightness;
        b *= brightness;

        pixels.at<cv::Vec3b>(row, col) =
            cv::Vec3b(static_cast<unsigned char>(b * 255),
                      static_cast<unsigned char>(g * 255),
                      static_cast<unsigned char>(r * 255));
      }
    }
  }

  cv::resize(pixels, pixels,
             cv::Size(pixels.cols / ssaa_factor, pixels.rows / ssaa_factor), 0,
             0, cv::INTER_AREA);

  // Save the image.
  cv::imwrite("rainbow.png", pixels);

  return 0;
}
