/*
 * This file contains the declarations for the various Mandelbrot functions.
 */

#pragma once
#include "utility.hpp"

MandelbrotResult mandelbrot_serial(const std::size_t width,
                                   const std::size_t height,
                                   const float real_min, const float real_max,
                                   const float imag_min, const float imag_max,
                                   const unsigned int max_iterations);

#if defined(_OPENMP)
MandelbrotResult mandelbrot_omp(const std::size_t width,
                                const std::size_t height, const float real_min,
                                const float real_max, const float imag_min,
                                const float imag_max,
                                const unsigned int max_iterations);
#endif

#if defined(__AVX2__)
MandelbrotResult mandelbrot_avx2(const std::size_t width,
                                 const std::size_t height, const float real_min,
                                 const float real_max, const float imag_min,
                                 const float imag_max,
                                 const unsigned int max_iterations);
#endif

#if defined(__AVX2__) && defined(_OPENMP)
MandelbrotResult mandelbrot_avx2_omp(const std::size_t width,
                                     const std::size_t height,
                                     const float real_min, const float real_max,
                                     const float imag_min, const float imag_max,
                                     const unsigned int max_iterations);
#endif

#if defined(__AVX512F__)
MandelbrotResult mandelbrot_avx512(const std::size_t width,
                                   const std::size_t height,
                                   const float real_min, const float real_max,
                                   const float imag_min, const float imag_max,
                                   const unsigned int max_iterations);
#endif

#if defined(__AVX512F__) && defined(_OPENMP)
MandelbrotResult
mandelbrot_avx512_omp(const std::size_t width, const std::size_t height,
                      const float real_min, const float real_max,
                      const float imag_min, const float imag_max,
                      const unsigned int max_iterations);
#endif
