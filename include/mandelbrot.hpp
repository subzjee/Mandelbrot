/*
 * This file contains the declarations for the various Mandelbrot functions.
 */

#pragma once
#include "utility.hpp"

MandelbrotResult mandelbrot_serial(std::size_t width, std::size_t height,
                                   float real_min, float real_max,
                                   float imag_min, float imag_max,
                                   unsigned int max_iterations);

#if defined(_OPENMP)
MandelbrotResult mandelbrot_omp(std::size_t width, std::size_t height,
                                float real_min, float real_max, float imag_min,
                                float imag_max, unsigned int max_iterations);
#endif

#if defined(__AVX2__)
MandelbrotResult mandelbrot_avx2(std::size_t width, std::size_t height,
                                 float real_min, float real_max, float imag_min,
                                 float imag_max, unsigned int max_iterations);
#endif

#if defined(__AVX2__) && defined(_OPENMP)
MandelbrotResult mandelbrot_avx2_omp(std::size_t width, std::size_t height,
                                     float real_min, float real_max,
                                     float imag_min, float imag_max,
                                     unsigned int max_iterations);
#endif

#if defined(__AVX512F__)
MandelbrotResult mandelbrot_avx512(std::size_t width, std::size_t height,
                                   float real_min, float real_max,
                                   float imag_min, float imag_max,
                                   unsigned int max_iterations);
#endif

#if defined(__AVX512F__) && defined(_OPENMP)
MandelbrotResult mandelbrot_avx512_omp(std::size_t width, std::size_t height,
                                       float real_min, float real_max,
                                       float imag_min, float imag_max,
                                       unsigned int max_iterations);
#endif
