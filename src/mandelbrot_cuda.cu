/*
 * This file contains the CUDA implementation.
 */

#if defined(ENABLE_CUDA)

#include <cuda/std/complex>
#include <cuda_runtime.h>

#include "mandelbrot_engine.hpp"

/*
 * CUDA device kernel that performs the calculation.
 *
 * @param iterations_out Output array for the iterations.
 * @param z_reals_out Output array for the real components.
 * @param z_imags_out Output array for the imaginary components.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param real_min The lower bound of the real axis.
 * @param real_max The upper bound of the real axis.
 * @param imag_min The lower bound of the imaginary axis.
 * @param imag_max The upper bound of the imaginary axis.
 * @param max_iterations The maximum iterations for each pixel.
 */
__global__ void mandelbrot_cuda_kernel(
    unsigned int* iterations_out, float* z_reals_out, float* z_imags_out,
    const std::size_t width, const std::size_t height, const float real_min,
    const float real_max, const float imag_min, const float imag_max,
    const unsigned int max_iterations) {
  const std::size_t col = threadIdx.x + blockIdx.x * blockDim.x;
  const std::size_t row = threadIdx.y + blockIdx.y * blockDim.y;

  if (row >= height || col >= width) {
    return;
  }

  cuda::std::complex<float> z{0.0f, 0.0f};
  cuda::std::complex<float> c{
      real_min + (real_max - real_min) * col / (width - 1),
      imag_max + (imag_min - imag_max) * row / (height - 1)};

  unsigned int iteration{0};

  while (cuda::std::norm(z) <= 4.0f && iteration < max_iterations) {
    z = z * z + c;

    ++iteration;
  }

  iterations_out[row * width + col] = iteration;
  z_reals_out[row * width + col] = z.real();
  z_imags_out[row * width + col] = z.imag();
}

/*
 * Check whether the CUDA backend is available.
 *
 * @Whether the CUDA backend is available.
 */
bool CUDAEngine::is_available() const {
  int device_count{0};
  cudaError_t err = cudaGetDeviceCount(&device_count);

  return (err == cudaSuccess && device_count > 0);
}

/*
 * Compute the Mandelbrot set with CUDA acceleration.
 *
 * @returns MandelbrotResult containing iteration and final z-value per pixel.
 */
MandelbrotResult CUDAEngine::compute() {
  if (!is_available()) {
    throw std::runtime_error("CUDA not available.");
  }

  dim3 block_size(16, 16);
  dim3 grid_size((m_width + block_size.x + 1) / block_size.x,
                 (m_height + block_size.y + 1) / block_size.y);

  mandelbrot_cuda_kernel<<<grid_size, block_size>>>(
      m_d_iterations, m_d_z_reals, m_d_z_imags, m_width, m_height, m_bounds.real_min, m_bounds.real_max,
      m_bounds.imag_min, m_bounds.imag_max, m_max_iterations);

  cudaMemcpy(m_h_iterations.get(), m_d_iterations,
             m_width * m_height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(m_h_z_reals.get(), m_d_z_reals, m_width * m_height * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(m_h_z_imags.get(), m_d_z_imags, m_width * m_height * sizeof(float),
             cudaMemcpyDeviceToHost);

  return {m_h_iterations, m_h_z_reals, m_h_z_imags,
          m_width, m_height};
}

#endif
