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

if(ENABLE_OMP)
    target_sources(mandelbrot PRIVATE mandelbrot_omp.cpp mandelbrot_avx2_omp.cpp mandelbrot_avx512_omp.cpp)
    target_link_libraries(mandelbrot PUBLIC OpenMP::OpenMP_CXX)
endif()

if(ENABLE_GPU)
    find_package(CUDAToolkit REQUIRED)
    target_sources(mandelbrot PRIVATE mandelbrot_cuda.cu)
    target_link_libraries(mandelbrot PUBLIC CUDA::cudart)
    target_compile_definitions(mandelbrot PUBLIC ENABLE_GPU)
endif()
