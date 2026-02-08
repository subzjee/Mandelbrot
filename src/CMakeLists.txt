set(SOURCE_FILES
    mandelbrot_omp.cpp
    mandelbrot_serial.cpp
    mandelbrot_avx2.cpp
    utility.cpp
)

add_library(mandelbrot ${SOURCE_FILES})
target_include_directories(mandelbrot PUBLIC ../include)
configure_target(mandelbrot)
