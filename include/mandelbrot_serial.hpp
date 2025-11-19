#pragma once

#include <cstdint>
#include <concepts>
#include <complex>

#include "utility.hpp"

template <std::floating_point Real>
void mandelbrotSerial(std::uint8_t* output, std::size_t width, std::size_t height, Real x_min, Real x_max, Real y_min, Real y_max, unsigned int max_iterations) {
    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t col = 0; col < width; ++col) {
            std::complex<Real> z{0.0, 0.0};
            std::complex<Real> c = mapPixelToComplexPlane(row, col, width, height, x_min, x_max, y_min, y_max);

            unsigned int iteration{0};
            while (std::norm(z) <= 4.0 && iteration < max_iterations) {
                z = z * z + c;

                ++iteration;
            }

            output[row * width + col] = static_cast<uint8_t>(255.0 * static_cast<Real>(iteration) / max_iterations);
        }
    }
}
