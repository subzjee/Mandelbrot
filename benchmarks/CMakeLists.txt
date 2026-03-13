set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_LTO ON CACHE BOOL "" FORCE)

include(FetchContent)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG main
)
FetchContent_MakeAvailable(benchmark)

add_executable(bm_mandelbrot bm_mandelbrot.cpp)
add_dependencies(bm_mandelbrot mandelbrot)
configure_target(bm_mandelbrot)
target_link_libraries(bm_mandelbrot PRIVATE mandelbrot benchmark::benchmark)

if(ENABLE_OMP)
    target_link_libraries(bm_mandelbrot PRIVATE OpenMP::OpenMP_CXX)
endif()

if(ENABLE_GPU)
    find_package(CUDAToolkit REQUIRED)

    target_include_directories(bm_mandelbrot PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(bm_mandelbrot PRIVATE ENABLE_GPU)
endif()

add_custom_target(run_benchmarks
    COMMAND bm_mandelbrot
    DEPENDS bm_mandelbrot
    COMMENT "Running Mandelbrot benchmarks"
)
