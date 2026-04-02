#pragma once

#include <vector>

#include "backends.hpp"
#include "utility.hpp"

using utility::AlignedAllocator;

template <Backend B> struct HostResources {
  explicit HostResources(std::size_t n) : iterations{}, z_reals{}, z_imags{} {
    iterations.reserve(n);
    z_reals.reserve(n);
    z_imags.reserve(n);
  };

  std::vector<unsigned int, AlignedAllocator<unsigned int, B::alignment>>
      iterations;
  std::vector<float, AlignedAllocator<float, B::alignment>> z_reals;
  std::vector<float, AlignedAllocator<float, B::alignment>> z_imags;
};

template <Backend B> struct DeviceResources {
  explicit DeviceResources(std::size_t) {};
};

#if defined(MANDELBROT_HAS_CUDA)
template <> struct DeviceResources<backend::CUDA> {
  explicit DeviceResources(std::size_t n) {
    cudaMalloc(&iterations, n * sizeof(unsigned int));
    cudaMalloc(&z_reals, n * sizeof(float));
    cudaMalloc(&z_imags, n * sizeof(float));
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
