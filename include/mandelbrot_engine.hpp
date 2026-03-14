#pragma once

#include <cstddef>
#include <memory>
#include <string>

#if defined(ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#include "mandelbrot_result.hpp"

enum class Backend {
  Serial,
#if defined(_OPENMP)
  OMP,
#endif
#if defined(__AVX2__)
  AVX2,
#if defined(_OPENMP)
  AVX2_OMP,
#endif
#endif
#if defined(__AVX512F__)
  AVX512,
#if defined(_OPENMP)
  AVX512_OMP,
#endif
#endif
#if defined(ENABLE_CUDA)
  CUDA,
#endif
};

template <Backend backend>
concept CPUBackend = (
    backend == Backend::Serial
#if defined(_OPENMP)
    || backend == Backend::OMP
#endif
#if defined(__AVX2__)
    || backend == Backend::AVX2
#if defined(_OPENMP)
    || backend == Backend::AVX2_OMP
#endif
#endif
#if defined(__AVX512F__)
    || backend == Backend::AVX512
#if defined(_OPENMP)
    || backend == Backend::AVX512_OMP
#endif
#endif
    );

inline std::string to_string(Backend backend) {
  switch (backend) {
  case Backend::Serial:
    return "Serial";
#if defined(_OPENMP)
  case Backend::OMP:
    return "OMP";
#endif
#if defined(__AVX2__)
  case Backend::AVX2:
    return "AVX2";
#if defined(_OPENMP)
  case Backend::AVX2_OMP:
    return "AVX2 + OMP";
#endif
#endif
#if defined(__AVX512F__)
  case Backend::AVX512:
    return "AVX512";
#if defined(_OPENMP)
  case Backend::AVX512_OMP:
    return "AVX512 + OMP";
#endif
#endif
#if defined(ENABLE_CUDA)
  case Backend::CUDA:
    return "CUDA";
#endif
  default:
    return "Unknown";
  }
}

struct ViewBounds {
  ViewBounds(float real_min, float real_max, float imag_min, float imag_max)
      : real_min{real_min}, real_max{real_max}, imag_min{imag_min},
        imag_max{imag_max} {};

  float real_min, real_max;
  float imag_min, imag_max;
};

class MandelbrotEngine {
public:
  virtual ~MandelbrotEngine() = default;
  virtual MandelbrotResult compute() = 0;
  virtual bool is_available() const = 0;
  virtual Backend get_backend() const = 0;

  void set_bounds(const ViewBounds& bounds) {
    m_bounds = bounds;
  }

  std::size_t width() const noexcept { return m_width; }
  std::size_t height() const noexcept { return m_height; }
  const ViewBounds& bounds() const noexcept { return m_bounds; }

protected:
  MandelbrotEngine(std::size_t width, std::size_t height,
                     const ViewBounds& bounds, unsigned int max_iterations)
      : m_width{width}, m_height{height}, m_bounds{bounds}, m_max_iterations{max_iterations} {};

  std::size_t m_width;
  std::size_t m_height;
  ViewBounds m_bounds;
  unsigned int m_max_iterations;
};

template <Backend backend>
  requires CPUBackend<backend>
class CPUEngine : public MandelbrotEngine {
public:
  CPUEngine(std::size_t width, std::size_t height, const ViewBounds& bounds,
              unsigned int max_iterations)
  : MandelbrotEngine(width, height, bounds, max_iterations) {
    m_iterations = std::make_shared<unsigned int[]>(width * height);
    m_z_reals = std::make_shared<float[]>(width * height);
    m_z_imags = std::make_shared<float[]>(width * height);
  };

  ~CPUEngine() = default;

  CPUEngine(const CPUEngine&) = delete;
  CPUEngine& operator=(const CPUEngine&) = delete;

  CPUEngine(CPUEngine&&) = default;
  CPUEngine& operator=(CPUEngine&&) = default;

  MandelbrotResult compute() override;
  bool is_available() const override;
  Backend get_backend() const override {
    return backend;
  }

private:
  std::shared_ptr<unsigned int[]> m_iterations;
  std::shared_ptr<float[]> m_z_reals;
  std::shared_ptr<float[]> m_z_imags;
};

#if defined(ENABLE_CUDA)
class CUDAEngine : public MandelbrotEngine {
public:
  CUDAEngine(std::size_t width, std::size_t height, const ViewBounds& bounds,
              unsigned int max_iterations)
  : MandelbrotEngine(width, height, bounds, max_iterations) {
    m_h_iterations = std::make_shared<unsigned int[]>(width * height);
    m_h_z_reals = std::make_shared<float[]>(width * height);
    m_h_z_imags = std::make_shared<float[]>(width * height);

    cudaMalloc(&m_d_iterations, width * height * sizeof(unsigned int));
    cudaMalloc(&m_d_z_reals, width * height * sizeof(float));
    cudaMalloc(&m_d_z_imags, width * height * sizeof(float));
  };

  ~CUDAEngine() {
    cudaFree(m_d_iterations);
    cudaFree(m_d_z_reals);
    cudaFree(m_d_z_imags);
  }

  CUDAEngine(const CUDAEngine&) = delete;
  CUDAEngine& operator=(const CUDAEngine&) = delete;

  CUDAEngine(CUDAEngine&&) = default;
  CUDAEngine& operator=(CUDAEngine&&) = default;

  MandelbrotResult compute() override;
  bool is_available() const override;
  Backend get_backend() const override {
    return Backend::CUDA;
  }

private:
  std::shared_ptr<unsigned int[]> m_h_iterations;
  std::shared_ptr<float[]> m_h_z_reals;
  std::shared_ptr<float[]> m_h_z_imags;

  unsigned int* m_d_iterations;
  float* m_d_z_reals;
  float* m_d_z_imags;
};
#endif

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
template <Backend backend = Backend::Serial>
std::unique_ptr<MandelbrotEngine>
create_engine(const std::size_t width, const std::size_t height,
                const ViewBounds& bounds, const unsigned int max_iterations) {
  if constexpr (CPUBackend<backend>) {
    return std::make_unique<CPUEngine<backend>>(width, height, bounds,
                                                  max_iterations);
  }
#if defined(ENABLE_CUDA)
  else if constexpr (backend == Backend::CUDA) {
    return std::make_unique<CUDAEngine>(width, height, bounds,
                                                   max_iterations);
  }
#endif
};
