cmake_minimum_required(VERSION 3.31)

option(BUILD_BENCHMARKS "Build Google Benchmark benchmarks" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(ENABLE_OMP "Enable OpenMP acceleration" ON)
option(USE_FAST_MATH "Use ffast-math compiler option" ON)

set(MANDELBROT_HAS_OMP ${ENABLE_OMP} CACHE INTERNAL "Whether OpenMP was enabled")

project(mandelbrot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

function(configure_target target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wconversion -march=native -flto=auto)
    target_link_options(${target} PRIVATE -flto=auto)

    if(USE_FAST_MATH)
        message(STATUS "Enabled fast math for ${target}")
        target_compile_options(${target} PRIVATE -ffast-math)
    endif()
endfunction()

if(ENABLE_OMP)
    find_package(OpenMP REQUIRED)
endif()

add_subdirectory(src)

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS mandelbrot
    EXPORT mandelbrotTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT mandelbrotTargets
    FILE mandelbrotTargets.cmake
    NAMESPACE mandelbrot::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mandelbrot
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mandelbrotConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mandelbrotConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mandelbrot
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mandelbrotConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mandelbrot)

install(FILES README.md LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/mandelbrot
)
