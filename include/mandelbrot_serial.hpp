#pragma once

#include <cstdint>

void mandelbrotSerial(std::uint8_t* output, std::size_t width,
                      std::size_t height, float x_min, float x_max, float y_min,
                      float y_max, unsigned int max_iterations);
