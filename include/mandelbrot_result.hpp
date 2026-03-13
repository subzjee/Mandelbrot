#pragma once

#include <complex>
#include <cstddef>
#include <memory>

class MandelbrotResult {
  struct EscapeResult {
    unsigned int iteration;
    std::complex<float> z;
  };

public:
  MandelbrotResult(std::shared_ptr<unsigned int[]> iterations,
                   std::shared_ptr<float[]> z_reals,
                   std::shared_ptr<float[]> z_imags, std::size_t width,
                   std::size_t height)
      : m_width(width), m_height(height), m_iters(iterations),
        m_z_real(z_reals), m_z_imag(z_imags) {};

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

    return {m_iters[idx], std::complex<float>{m_z_real[idx], m_z_imag[idx]}};
  }

  /*
   * Get the width of the frame.
   *
   * @returns The width.
   */
  std::size_t width() const noexcept { return m_width; }

  /*
   * Get the height of the frame.
   *
   * @returns The height.
   */
  std::size_t height() const noexcept { return m_height; }

private:
  const std::size_t m_width;
  const std::size_t m_height;

  std::shared_ptr<unsigned int[]> m_iters{nullptr};
  std::shared_ptr<float[]> m_z_real{nullptr};
  std::shared_ptr<float[]> m_z_imag{nullptr};
};
