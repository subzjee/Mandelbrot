set(SOURCE_FILES
    mandelbrot_avx2.cpp
    mandelbrot_avx2_omp.cpp
    mandelbrot_avx512.cpp
    mandelbrot_avx512_omp.cpp
    mandelbrot_omp.cpp
    mandelbrot_serial.cpp
    utility_avx.cpp
    utility_avx512.cpp
)

add_library(mandelbrot ${SOURCE_FILES})
target_include_directories(mandelbrot PUBLIC ../include)
configure_target(mandelbrot)
