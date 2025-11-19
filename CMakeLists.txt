cmake_minimum_required(VERSION 3.31)

project(mandelbrot LANGUAGES CXX)

option(BUILD_BENCHMARKS "Build Google Benchmark tests" OFF)
option(USE_FAST_MATH "Use ffast-math compiler option" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP)
find_package(OpenCV REQUIRED)

set(WARNING_FLAGS
    -Wall -Wextra -Wpedantic -Wconversion
)

function(configure_target target)
    target_compile_options(${target} PRIVATE ${WARNING_FLAGS} -O3 -march=native -flto)
    target_link_options(${target} PRIVATE -flto)

    if(USE_FAST_MATH)
        message(STATUS "Enabled fast math for ${target}")
        target_compile_options(${target} PRIVATE -ffast-math)
    endif()

    if(OpenMP_CXX_FOUND)
        message(STATUS "Enabled OpenMP for ${target}")
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

add_executable(${PROJECT_NAME} src/main.cpp)
configure_target(${PROJECT_NAME})

target_include_directories(mandelbrot PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(mandelbrot PRIVATE ${OpenCV_LIBS})

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
