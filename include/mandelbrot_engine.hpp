#pragma once

#include <cstddef>
#include <format>
#include <memory>

#if defined(MANDELBROT_HAS_CUDA)
#include <cuda_runtime.h>
#endif

#include "backends.hpp"
#include "mandelbrot_result.hpp"

struct ViewBounds {
  ViewBounds(float real_min, float real_max, float imag_min, float imag_max)
      : real_min{real_min}, real_max{real_max}, imag_min{imag_min},
        imag_max{imag_max} {};

  float real_min, real_max;
  float imag_min, imag_max;
};

template <Backend B> struct DeviceResources {
  explicit DeviceResources(std::size_t) {};
};

#if defined(MANDELBROT_HAS_CUDA)
template <> struct DeviceResources<backend::cuda> {
  explicit DeviceResources(std::size_t size) {
    cudaMalloc(&iterations, size * sizeof(unsigned int));
    cudaMalloc(&z_reals, size * sizeof(float));
    cudaMalloc(&z_imags, size * sizeof(float));
  }

  ~DeviceResources() {
    cudaFree(iterations);
    cudaFree(z_reals);
    cudaFree(z_imags);
  }

  DeviceResources(const DeviceResources&) = delete;
  DeviceResources& operator=(const DeviceResources&) = delete;

  DeviceResources(DeviceResources&& other) noexcept
      : iterations{std::exchange(other.iterations, nullptr)},
        z_reals{std::exchange(other.z_reals, nullptr)},
        z_imags{std::exchange(other.z_imags, nullptr)} {}

  DeviceResources& operator=(DeviceResources&& other) noexcept {
    if (this != &other) {
      cudaFree(iterations);
      cudaFree(z_reals);
      cudaFree(z_imags);

      iterations = std::exchange(other.iterations, nullptr);
      z_reals = std::exchange(other.z_reals, nullptr);
      z_imags = std::exchange(other.z_imags, nullptr);
    }

    return *this;
  }

  unsigned int* iterations;
  float* z_reals;
  float* z_imags;
};
#endif

template <Backend B> class MandelbrotEngine {
public:
  MandelbrotEngine(std::size_t width, std::size_t height,
                   const ViewBounds& bounds, unsigned int max_iterations)
      : m_width{width}, m_height{height}, m_bounds{bounds},
        m_max_iterations{max_iterations}, m_device{width * height} {
    if (!B::is_available()) {
      throw std::runtime_error(
          std::format("{} backend is not available.", B::name()));
    }

    m_iterations = std::make_shared<unsigned int[]>(width * height);
    m_z_reals = std::make_shared<float[]>(width * height);
    m_z_imags = std::make_shared<float[]>(width * height);
  };

  MandelbrotResult compute();

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

  std::shared_ptr<unsigned int[]> m_iterations;
  std::shared_ptr<float[]> m_z_reals;
  std::shared_ptr<float[]> m_z_imags;

  [[no_unique_address]] DeviceResources<B> m_device;
};

/*
 * Create an engine for the specified backend.
 *
 * @tparam backend The backend to use.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param bounds The view bounds (real and imaginary axis limits).
 * @param max_iterations The maximum iterations per pixel.
 *
 * @returns The engine.
 */
template <Backend B = backend::serial>
std::unique_ptr<MandelbrotEngine<B>>
create_engine(const std::size_t width, const std::size_t height,
              const ViewBounds& bounds, const unsigned int max_iterations) {
  return std::make_unique<MandelbrotEngine<B>>(width, height, bounds,
                                               max_iterations);
}
