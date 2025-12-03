#include <vector>

#include "benchmark/benchmark.h"
#include "mandelbrot_serial.hpp"
#include "mandelbrot_omp.hpp"

constexpr float REAL_MIN = -2.0f, REAL_MAX = 1.0f;
constexpr float IMAG_MIN = -1.0f, IMAG_MAX =  1.0f;
constexpr unsigned int MAX_ITER = 1000;

template <auto Func>
inline void BM_Mandelbrot(benchmark::State& state) {
    const auto width  = state.range(0);
    const auto height = state.range(1);

    std::vector<uint8_t> output(width * height * 3);

    for (auto _ : state) {
        Func(output.data(), width, height,
             REAL_MIN, REAL_MAX,
             IMAG_MIN, IMAG_MAX,
             MAX_ITER);
    }
}

#define COMMON_ARGS \
    ->Args({640, 480}) \
    ->Args({1280, 720}) \
    ->Args({1920, 1080}) \
    ->Args({3840, 2160}) \
    ->UseRealTime()

#define MANDEL_BENCH(NAME, FUNC) \
    BENCHMARK(BM_Mandelbrot<FUNC>) \
        ->Name(NAME) \
        COMMON_ARGS;

MANDEL_BENCH("Serial", mandelbrotSerial)
MANDEL_BENCH("OMP",    mandelbrotOMP)

BENCHMARK_MAIN();
