set(SOURCE_FILES
    mandelbrot_avx2.cpp
    mandelbrot_avx512.cpp
    mandelbrot_serial.cpp
    utility_avx.cpp
    utility_avx512.cpp
)

add_library(mandelbrot)
target_sources(mandelbrot PRIVATE ${SOURCE_FILES})
target_include_directories(mandelbrot
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
configure_target(mandelbrot)

include(CheckCXXSourceCompiles)

set(CMAKE_REQUIRED_FLAGS "-march=native")

check_cxx_source_compiles("
    #if !defined(__AVX2__)
    #error
    #endif
    int main() {}" MANDELBROT_HAS_AVX2)

check_cxx_source_compiles("
    #if !defined(__AVX512F__)
    #error
    #endif
    int main() {}" MANDELBROT_HAS_AVX512)

unset(CMAKE_REQUIRED_FLAGS)

if(MANDELBROT_HAS_AVX2)
    target_compile_definitions(mandelbrot PUBLIC MANDELBROT_HAS_AVX2)
endif()

if(MANDELBROT_HAS_AVX512)
    target_compile_definitions(mandelbrot PUBLIC MANDELBROT_HAS_AVX512)
endif()

if(ENABLE_OMP)
    target_sources(mandelbrot PRIVATE mandelbrot_omp.cpp mandelbrot_avx2_omp.cpp mandelbrot_avx512_omp.cpp)
    target_link_libraries(mandelbrot PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(mandelbrot PUBLIC MANDELBROT_HAS_OMP)
endif()

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    target_sources(mandelbrot PRIVATE mandelbrot_cuda.cu)
    target_link_libraries(mandelbrot PUBLIC CUDA::cudart)
    target_compile_definitions(mandelbrot PUBLIC MANDELBROT_HAS_CUDA)
endif()
