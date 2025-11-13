#include <cstdint>

#include <opencv2/opencv.hpp>

#include "mandelbrot_serial.hpp"

constexpr std::size_t width = 7680, height = 4320;
constexpr int max_iterations = 500;
constexpr double x_min = -1.252213542, x_max = -1.22213542; // The bounds of the real axis on the complex plane.
constexpr double y_min = 0.108567708, y_max = 0.125442708; // The bounds of the imaginary axis on the complex plane.

int main() {
    cv::Mat pixels(height, width, CV_8UC1);

    uint8_t* buffer = pixels.data;
    mandelbrotSerial(buffer, width, height, x_min, x_max, y_min, y_max, max_iterations);

    cv::bitwise_not(pixels, pixels);
    cv::imwrite("mandelbrot.png", pixels);

    return 0;
}
