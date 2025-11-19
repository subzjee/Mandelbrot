#pragma once

#include <concepts>
#include <complex>

template <std::floating_point T>
/*
 * Map an index in a 1D structure to a bounded axis linearly.
 *
 * @param idx The index.
 * @param num_spaces The number of spaces to linearly space to bounded axis into.
 * @param start The start of the range.
 * @param end The end of the range.
 *
 * @returns The mapped coordinate on the axis.
 */
constexpr T mapIndexToBoundedAxis(std::size_t idx, std::size_t num_spaces, T start, T end) {
    return start + (static_cast<T>(idx) / static_cast<T>(num_spaces - 1)) * (end - start);
}

template <std::floating_point T>
/*
 * Map a pixel in an image structured as a 1D array to the complex plane.
 *
 * @param row The row of the pixel.
 * @param col The column of the pixel.
 * @param height The height of the image.
 * @param width The width of the image.
 * @param x_min The left bound of the real axis.
 * @param x_max The right bound of the real axis.
 * @param y_min The lower bound of the imaginary axis.
 * @param y_max The upper bound of the imaginary axis.
 *
 * @returns The mapped position of the pixel on the complex plane.
 */
constexpr std::complex<T> mapPixelToComplexPlane(std::size_t row, std::size_t col, std::size_t width, std::size_t height, T x_min, T x_max, T y_min, T y_max) {
    return {
        mapIndexToBoundedAxis(col, width, x_min, x_max), // Map column onto the real axis.
        mapIndexToBoundedAxis(row, height, y_max, y_min) // Map row onto the imaginary axis.
    };
}
