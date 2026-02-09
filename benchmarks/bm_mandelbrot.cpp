#include <vector>

#include "benchmark/benchmark.h"

#include "mandelbrot.hpp"
#include "utility.hpp"

constexpr float real_min = -2.0f, real_max = 1.0f;
constexpr float imag_min = -1.0f, imag_max = 1.0f;
constexpr unsigned int max_iter = 1000;

template <auto Func> void BM_Mandelbrot(benchmark::State& state) {
  if (Func == mandelbrot_avx2 && !__builtin_cpu_supports("avx2")) {
    state.SkipWithError("AVX2 not supported on this CPU");
    return;
  } else if (Func == mandelbrot_avx512 && !__builtin_cpu_supports("avx512f")) {
    state.SkipWithError("AVX512F not supported on this CPU");
    return;
  }

  const auto width = state.range(0);
  const auto height = state.range(1);

  std::vector<uint8_t> output(width * height * 3);

  for (auto _ : state) {
    MandelbrotResult result =
        Func(width, height, real_min, real_max, imag_min, imag_max, max_iter);
    benchmark::DoNotOptimize(result);
  }
}

#define COMMON_ARGS                                                            \
  ->Args({640, 480})                                                           \
      ->Args({1280, 720})                                                      \
      ->Args({1920, 1080})                                                     \
      ->Args({3840, 2160})                                                     \
      ->UseRealTime()

#define MANDEL_BENCH(NAME, FUNC)                                               \
  BENCHMARK(BM_Mandelbrot<FUNC>)->Name(NAME) COMMON_ARGS;

MANDEL_BENCH("Serial", mandelbrot_serial)

#if defined(_OPENMP)
MANDEL_BENCH("OMP", mandelbrot_omp)
#endif

#if defined(__AVX2__)
MANDEL_BENCH("AVX2", mandelbrot_avx2)
#endif

#if defined(__AVX2__) && defined(_OPENMP)
MANDEL_BENCH("AVX2_OMP", mandelbrot_avx2_omp)
#endif

#if defined(__AVX512F__)
MANDEL_BENCH("AVX512", mandelbrot_avx512)
#endif

#if defined(__AVX512F__) && defined(_OPENMP)
MANDEL_BENCH("AVX512_OMP", mandelbrot_avx512_omp)
#endif

BENCHMARK_MAIN();
