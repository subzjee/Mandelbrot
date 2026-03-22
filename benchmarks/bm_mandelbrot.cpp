/*
 * This file contains the benchmarks for the various implementations.
 *
 * Benchmarks may be skipped depending on the runtime capability of the system.
 */

#include <format>

#include "benchmark/benchmark.h"

#include "mandelbrot_engine.hpp"

const ViewBounds bounds{-2.0f, 1.0f, -1.0f, 1.0f};
constexpr unsigned int max_iter = 1000;

template <Backend backend>
void BM_Mandelbrot(benchmark::State& state) {
  const std::size_t width = static_cast<std::size_t>(state.range(0));
  const std::size_t height = static_cast<std::size_t>(state.range(1));

  auto engine = create_engine<backend>(width, height, bounds, max_iter);

  if (!engine || !is_available(backend)) {
    state.SkipWithError(std::format("Backend {} not available", to_string(backend)));
    return;
  }

  for (auto _ : state) {
    auto result = engine->compute();
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

#if defined(MANDELBROT_HAS_OMP)
MANDEL_BENCH(OMP)
#endif

#if defined(MANDELBROT_HAS_AVX2)
MANDEL_BENCH(AVX2)
#endif

#if defined(MANDELBROT_HAS_AVX2) && defined(MANDELBROT_HAS_OMP)
MANDEL_BENCH(AVX2_OMP)
#endif

#if defined(MANDELBROT_HAS_AVX512)
MANDEL_BENCH(AVX512)
#endif

#if defined(MANDELBROT_HAS_AVX512) && defined(MANDELBROT_HAS_OMP)
MANDEL_BENCH(AVX512_OMP)
#endif

#if defined(MANDELBROT_HAS_CUDA)
MANDEL_BENCH(CUDA)
#endif

BENCHMARK_MAIN();
