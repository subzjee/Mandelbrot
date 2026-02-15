#pragma once

#include <cstddef>
#include <memory>
#include <span>

class MandelbrotResult {
public:
  MandelbrotResult(std::unique_ptr<unsigned int[]> iterations,
                   std::size_t width, std::size_t height)
      : m_width(width), m_height(height), m_data(std::move(iterations)) {};

  std::span<const unsigned int> operator[](std::size_t row_idx) const noexcept {
    return {m_data.get() + row_idx * m_width, m_width};
  }

  /*
   * Get the width.
   *
   * @returns The width.
   */
  std::size_t width() const noexcept { return m_width; }

  /*
   * Get the height.
   *
   * @returns The height.
   */
  std::size_t height() const noexcept { return m_height; }

private:
  const std::size_t m_width;
  const std::size_t m_height;
  const std::unique_ptr<unsigned int[]> m_data;
};
