/*
 * This file contains the OpenMP implementation.
 */

#if defined(_OPENMP)

#include <complex>

#include "mandelbrot_engine.hpp"
#include "utility.hpp"

/*
 * Check whether the OpenMP backend is available.
 *
 * @returns Whether the OpenMP backend is available.
 */
template<>
bool CPUEngine<Backend::OMP>::is_available() const {
  return true;
}

/*
 * Compute the Mandelbrot set with OpenMP acceleration.
 *
 * @returns MandelbrotResult containing iteration and final z-value per pixel.
 */
template<>
MandelbrotResult CPUEngine<Backend::OMP>::compute() {
#pragma omp parallel for collapse(2) schedule(guided)
  for (std::size_t row = 0; row < m_height; ++row) {
    for (std::size_t col = 0; col < m_width; ++col) {
      std::complex<float> z{0.0f, 0.0f};
      const std::complex<float> c = utility::detail::mapPixelToComplexPlane(
          row, col, m_width, m_height, m_bounds.real_min, m_bounds.real_max, m_bounds.imag_min, m_bounds.imag_max);

      unsigned int iteration{0};
      while (std::norm(z) <= 4.0f && iteration < m_max_iterations) {
        z = z * z + c;

        ++iteration;
      }

      const std::size_t idx = row * m_width + col;

      m_iterations[idx] = iteration;
      m_z_reals[idx] = z.real();
      m_z_imags[idx] = z.imag();
    }
  }

  return {m_iterations, m_z_reals, m_z_imags, m_width,
          m_height};
}

#endif
