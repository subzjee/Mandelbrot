/*
 * This file contains the benchmarks for the various implementations.
 *
 * Benchmarks may be skipped depending on the runtime capability of the system.
 */

#include <format>
#include <string>

#include "benchmark/benchmark.h"

#if defined(ENABLE_GPU)
#include <cuda_runtime.h>
#endif

#include "mandelbrot_renderer.hpp"
#include "mandelbrot_result.hpp"

const ViewBounds bounds{-2.0f, 1.0f, -1.0f, 1.0f};
constexpr unsigned int max_iter = 1000;

template <Backend backend>
void BM_Mandelbrot(benchmark::State& state) {
  const std::size_t width = static_cast<std::size_t>(state.range(0));
  const std::size_t height = static_cast<std::size_t>(state.range(1));

  auto renderer = create_renderer<backend>(width, height, bounds, max_iter);

  if (!renderer || !renderer->is_available()) {
    state.SkipWithError(std::format("Backend {} not available", to_string(backend)));
    return;
  }

  for (auto _ : state) {
    auto result = renderer->render();
  }
}

// Set benchmark image resolutions.
#define COMMON_ARGS                                                            \
  ->Args({640, 480})                                                           \
  ->Args({1280, 720})                                                      \
  ->Args({1920, 1080})                                                     \
  ->Args({3840, 2160})                                                     \
  ->UseRealTime()

#define MANDEL_BENCH(BACKEND)                                                  \
  BENCHMARK(BM_Mandelbrot<Backend::BACKEND>)->Name(#BACKEND) COMMON_ARGS;

MANDEL_BENCH(Serial)

#if defined(_OPENMP)
MANDEL_BENCH(OMP)
#endif

#if defined(__AVX2__)
MANDEL_BENCH(AVX2)
#endif

#if defined(__AVX2__) && defined(_OPENMP)
MANDEL_BENCH(AVX2_OMP)
#endif

#if defined(__AVX512F__)
MANDEL_BENCH(AVX512)
#endif

#if defined(__AVX512F__) && defined(_OPENMP)
MANDEL_BENCH(AVX512_OMP)
#endif

#if defined(ENABLE_CUDA)
MANDEL_BENCH(CUDA)
#endif

BENCHMARK_MAIN();
