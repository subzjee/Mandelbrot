#pragma once

#include <cstddef>
#include <format>

#if defined(MANDELBROT_HAS_CUDA)
#include <cuda_runtime.h>
#endif

#include "backends.hpp"
#include "mandelbrot_result.hpp"
#include "resources.hpp"

struct ViewBounds {
  ViewBounds(float real_min, float real_max, float imag_min, float imag_max)
      : real_min{real_min}, real_max{real_max}, imag_min{imag_min},
        imag_max{imag_max} {};

  float real_min, real_max;
  float imag_min, imag_max;
};

template <Backend B = backend::Serial, Execution Exec = exec::Default>
  requires Compatible<B, Exec>
class MandelbrotEngine {
public:
  MandelbrotEngine(std::size_t width, std::size_t height,
                   const ViewBounds& bounds, unsigned int max_iterations)
      : m_width{width}, m_height{height}, m_bounds{bounds},
        m_max_iterations{max_iterations}, m_host{width * height},
        m_device{width * height} {
    if (!B::is_available()) {
      throw std::runtime_error(
          std::format("{} backend is not available.", B::name()));
    }
  };

  MandelbrotResult<B> compute();

  void set_bounds(const ViewBounds& bounds) { m_bounds = bounds; }

  MandelbrotEngine(const MandelbrotEngine&) = delete;
  MandelbrotEngine& operator=(const MandelbrotEngine&) = delete;

  MandelbrotEngine(MandelbrotEngine&&) = default;
  MandelbrotEngine& operator=(MandelbrotEngine&&) = default;

  std::size_t width() const noexcept { return m_width; }
  std::size_t height() const noexcept { return m_height; }
  const ViewBounds& bounds() const noexcept { return m_bounds; }

private:
  std::size_t m_width;
  std::size_t m_height;
  ViewBounds m_bounds;
  unsigned int m_max_iterations;

  HostResources<B> m_host;
  [[no_unique_address]] DeviceResources<B> m_device;
};
