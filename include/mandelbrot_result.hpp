#pragma once

#include <complex>
#include <cstddef>
#include <memory>
#include <utility>

class MandelbrotResult {
  struct EscapeResult {
    unsigned int iteration;
    std::complex<float> z;
  };

public:
  MandelbrotResult(std::unique_ptr<unsigned int[]> iterations,
                   std::unique_ptr<float[]> z_real,
                   std::unique_ptr<float[]> z_imag, std::size_t width,
                   std::size_t height)
      : m_width(width), m_height(height), m_iters(std::move(iterations)),
        m_z_real(std::move(z_real)), m_z_imag(std::move(z_imag)) {};

  EscapeResult operator()(std::size_t row, std::size_t col) const noexcept {
    std::size_t idx = row * m_width + col;

    return {m_iters[idx], std::complex<float>{m_z_real[idx], m_z_imag[idx]}};
  }

  std::size_t width() const noexcept { return m_width; }
  std::size_t height() const noexcept { return m_height; }

private:
  const std::size_t m_width;
  const std::size_t m_height;

  const std::unique_ptr<unsigned int[]> m_iters{nullptr};
  const std::unique_ptr<float[]> m_z_real{nullptr};
  const std::unique_ptr<float[]> m_z_imag{nullptr};
};
