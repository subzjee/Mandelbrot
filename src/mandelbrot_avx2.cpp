#include "mandelbrot_avx2.hpp"

void mandelbrotAVX2(std::uint8_t* output, std::size_t width, std::size_t height,
                    float x_min, float x_max, float y_min, float y_max,
                    unsigned int max_iterations) {
    static constexpr std::size_t lanes = 32 / sizeof(float); // How many elements can be processed at once.

    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t col = 0; col < width; col += lanes) {

        }
    }
}
