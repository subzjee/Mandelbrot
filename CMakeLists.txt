cmake_minimum_required(VERSION 3.31)

option(BUILD_BENCHMARKS "Build Google Benchmark benchmarks" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(ENABLE_OMP "Enable OpenMP acceleration" ON)
option(ENABLE_CUDA "Enable CUDA acceleration" OFF)
option(USE_FAST_MATH "Use ffast-math compiler option" ON)

if(ENABLE_CUDA)
    project(mandelbrot LANGUAGES CXX CUDA)
    message(STATUS "Enabled CUDA acceleration")
else()
    project(mandelbrot LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(MANDELBROT_HAS_OMP ${ENABLE_OMP} "Whether OpenMP was enabled")
set(MANDELBROT_HAS_CUDA ${ENABLE_CUDA} "Whether CUDA was enabled")

function(configure_target target)
    target_compile_options(${target} PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wconversion -march=native -fno-lto>
    )
    target_link_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fno-lto>
    )

    if(USE_FAST_MATH)
        message(STATUS "Enabled fast math for ${target}")
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        )

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
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mandelbrot)
install(TARGETS mandelbrot
    EXPORT mandelbrotTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mandelbrot
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
