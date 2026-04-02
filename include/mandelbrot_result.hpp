#pragma once

#include <complex>
#include <cstddef>

#include "backends.hpp"
#include "resources.hpp"

struct EscapeResult {
  unsigned int iteration;
  std::complex<float> z;
};

template <Backend B> class MandelbrotResult {
public:
  MandelbrotResult() = default;

  MandelbrotResult(const HostResources<B>& resources, std::size_t width,
                   std::size_t height)
      : m_resources(resources), m_width(width), m_height(height) {};

  /*
   * Get the escape information for the pixel at row `row` and column `col`.
   *
   * @param row The row of the pixel.
   * @param col The column of the pixel.
   *
   * @returns The escape information.
   */
  EscapeResult operator()(std::size_t row, std::size_t col) const noexcept {
    std::size_t idx = row * m_width + col;

    return {m_resources.iterations[idx],
            std::complex<float>{m_resources.z_reals[idx],
                                m_resources.z_imags[idx]}};
  }

private:
  std::size_t m_width;
  std::size_t m_height;

  const HostResources<B>& m_resources;
};
