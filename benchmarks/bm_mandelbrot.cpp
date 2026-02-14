/*
 * This file contains the benchmarks for the various implementations.
 *
 * The benchmarks included are: serial, AVX2, AVX512 and then the combination for each
 * with OpenMP.
 *
 * Benchmarks may be skipped depending on the runtime capability of the CPU.
 */

#include <string>
#include <vector>

#include "benchmark/benchmark.h"

#include "mandelbrot.hpp"
#include "mandelbrot_result.hpp"

constexpr float real_min = -2.0f, real_max = 1.0f;
constexpr float imag_min = -1.0f, imag_max = 1.0f;
constexpr unsigned int max_iter = 1000;

enum class Target {
  Generic,
  AVX2,
  AVX512F
};

template <Target target>
struct ExecutionConfig {
  static bool is_available() {
    switch (target) {
      case Target::Generic: return true;
      case Target::AVX2: return __builtin_cpu_supports("avx2");
      case Target::AVX512F: return __builtin_cpu_supports("avx512f");
    }
  }

  static std::string name() {
    switch (target) {
      case Target::Generic: return "Generic";
      case Target::AVX2: return "AVX2";
      case Target::AVX512F: return "AVX512F";
    }
  }
};

template <auto Func, Target target> void BM_Mandelbrot(benchmark::State& state) {
  if (!ExecutionConfig<target>::is_available()) {
    state.SkipWithError(ExecutionConfig<target>::name() + " is not supported on this machine");
    return;
  }

  const std::size_t width = static_cast<std::size_t>(state.range(0));
  const std::size_t height = static_cast<std::size_t>(state.range(1));

  for (auto _ : state) {
    MandelbrotResult result =
        Func(width, height, real_min, real_max, imag_min, imag_max, max_iter);
  }
}

// Set benchmark image resolutions.
#define COMMON_ARGS                                                            \
  ->Args({640, 480})                                                           \
      ->Args({1280, 720})                                                      \
      ->Args({1920, 1080})                                                     \
      ->Args({3840, 2160})                                                     \
      ->UseRealTime()

#define MANDEL_BENCH(NAME, FUNC, TARGET)                                       \
  BENCHMARK(BM_Mandelbrot<FUNC, TARGET>)->Name(NAME) COMMON_ARGS;

MANDEL_BENCH("Serial", mandelbrot_serial, Target::Generic)

#if defined(_OPENMP)
MANDEL_BENCH("OMP", mandelbrot_omp, Target::Generic)
#endif

#if defined(__AVX2__)
MANDEL_BENCH("AVX2", mandelbrot_avx2, Target::AVX2)
#endif

#if defined(__AVX2__) && defined(_OPENMP)
MANDEL_BENCH("AVX2_OMP", mandelbrot_avx2_omp, Target::AVX2)
#endif

#if defined(__AVX512F__)
MANDEL_BENCH("AVX512", mandelbrot_avx512, Target::AVX512F)
#endif

#if defined(__AVX512F__) && defined(_OPENMP)
MANDEL_BENCH("AVX512_OMP", mandelbrot_avx512_omp, Target::AVX512F)
#endif

BENCHMARK_MAIN();
