#include <vector>

#include "benchmark/benchmark.h"

#include "mandelbrot_serial.hpp"
#include "mandelbrot_omp.hpp"

constexpr unsigned int MAX_ITER = 1000;

static void BM_Mandelbrot_Serial(benchmark::State& state) {
    const auto width  = state.range(0);
    const auto height = state.range(1);

    std::vector<uint8_t> output(width * height * 3);

    for (auto _ : state) {
        mandelbrotSerial(output.data(), width, height, -2.0, 1.0, -1.0, 1.0, MAX_ITER);
    }
}

static void BM_Mandelbrot_OpenMP(benchmark::State& state) {
    const auto width  = state.range(0);
    const auto height = state.range(1);

    std::vector<uint8_t> output(width * height * 3);

    for (auto _ : state) {
        mandelbrotOMP(output.data(), width, height, -2.0, 1.0, -1.0, 1.0, MAX_ITER);
    }
}

BENCHMARK(BM_Mandelbrot_Serial)
    ->Args({640, 480})
    ->Args({1280, 720})
    ->Args({1920, 1080})
    ->Args({3840, 2160})
    ->UseRealTime();

BENCHMARK(BM_Mandelbrot_OpenMP)
    ->Args({640, 480})
    ->Args({1280, 720})
    ->Args({1920, 1080})
    ->Args({3840, 2160})
    ->UseRealTime();

BENCHMARK_MAIN();
