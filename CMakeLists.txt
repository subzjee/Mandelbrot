cmake_minimum_required(VERSION 3.31)

project(mandelbrot LANGUAGES CXX)

option(BUILD_BENCHMARKS "Build Google Benchmark tests" OFF)
option(USE_FAST_MATH "Use ffast-math compiler option" ON)
option(BUILD_EXAMPLES "Build examples" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP)

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

add_subdirectory(src)

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS mandelbrot DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES README.md LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/mandelbrot)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mandelbrotConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mandelbrot
)
