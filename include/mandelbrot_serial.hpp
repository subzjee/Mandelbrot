#pragma once

#include <cstdint>
#include <concepts>
#include <complex>

#include "utility.hpp"

template <std::floating_point T>
void mandelbrotSerial(std::uint8_t* output, std::size_t width, std::size_t height, T x_min, T x_max, T y_min, T y_max, unsigned int max_iterations) {
    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
            std::complex<T> z{0.0, 0.0};
            std::complex<T> c = mapPixelToComplexPlane(row, col, width, height, x_min, x_max, y_min, y_max);

            unsigned int iteration{0};
            while (std::norm(z) <= 4.0 && iteration < max_iterations) {
                z = z * z + c;

                ++iteration;
            }

            output[row * width + col] = iteration;
        }
    }
}
