# **Mandelbrot**
A Mandelbrot set library in C++17 featuring optional OpenMP and AVX2/AVX512 support. It uses the escape-time algorithm to obtain the iteration count.

---

## Motivation

The Mandelbrot set has fascinated me for two reasons:
1. Emergent behavior: Simple iterative rules giving rise to complex patterns is beautiful.
2. Potential for beautiful visualizations: Apart from the mathematical beauty of the Mandelbrot set, it allows for endless visualization strategies that produce stunning and colorful images.

---

## Features
* Serial implementation for a simple and portable fallback.
* Parallel processing with OpenMP for multicore acceleration.
* Vectorization support with AVX2/AVX512 for capable CPUs.
* Works with CMake and is installable as a library.

---

## Requirements
* Clang/GCC with C++17 support.
* CMake 3.31+ for ease-of-building.
* Optional:
  * A compiler supporting OpenMP to enable the OpenMP implementations.
  * AVX2-capable CPU to use the AVX2 implementations.
  * AVX512-capable CPU to use the AVX512 implementations.

---

## Building the library
```bash
git clone https://github.com/subzjee/Mandelbrot
cd Mandelbrot
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Installing the library
Once the library has been built, you can install the library using the following:
```bash
sudo cmake --install build
```
This will install the library to the default system locations. If you wish to install it to a custom location, you can use the `CMAKE_INSTALL_PREFIX` variable:
```bash
sudo cmake --install build --prefix /path/to/install
```

### Benchmarks
This project includes benchmarks making use of Google Benchmark. These benchmarks can be found in the `benchmarks` directory.

To build the benchmarks, add `-DBUILD_BENCHMARKS=ON` while building the library.\
There is a custom target (`run_benchmarks`) to run the benchmarks after building.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
cmake --build build --target run_benchmarks
```

Alternatively, you can run the benchmarks manually:
```bash
./build/benchmarks/bm_mandelbrot
```

---

## Using the library

Once the library has been installed, it can be used in a CMake-based project by using `find_package`.
```cmake
find_package(mandelbrot REQUIRED)
target_link_libraries(<your_project> PRIVATE mandelbrot::mandelbrot)
```

Then, you can include the header and use the library as shown below:

```cpp
#include <iostream>

#include "mandelbrot.hpp"

int main() {
  MandelbrotResult iterations = mandelbrot_serial(1920, 1080, -2.0f, 1.0f, -1.0f, 1.0f, 1000);
  
  // Print the iteration count for each pixel.
  for (std::size_t row = 0; row < iterations.size(); ++row) {
    for (std::size_t col = 0; col < iterations[row]; ++col) {
      std::cout << iterations[row][col] << '\n';
    }
  }
  return 0;
}
```

This example uses the serial implementation to generate the Mandelbrot set for a 1920x1080 image. The complex plane is bounded by [-2.0, 1.0] for the real axis and [-1.0, 1.0] for the imaginary axis, with a maximum of 1000 iterations per pixel.

For more examples, which export it to an actual image, check out the `examples` directory.
To build the examples, add `-DBUILD_EXAMPLES=ON` while building the library.
For ease-of-use, the examples do require OpenCV to be installed.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
./build/examples/greyscale
```

![Greyscale example of the Mandelbrot set](/examples/greyscale.png)

---

## Project Structure

```bash
.
├── benchmarks
│   ├── bm_mandelbrot.cpp           # Benchmarks
│   └── CMakeLists.txt
├── cmake
│   └── mandelbrotConfig.cmake      # Config for `find_package`
├── CMakeLists.txt                  # Root CMake file
├── examples
│   ├── CMakeLists.txt
│   └── greyscale.cpp               # Export as greyscale example
├── include           
│   ├── mandelbrot.hpp              # Main functions
│   └── utility.hpp                 # Helper functions
├── LICENSE
├── README.md
└── src
    ├── CMakeLists.txt
    ├── mandelbrot_avx2.cpp         # AVX2 implementation
    ├── mandelbrot_avx2_omp.cpp     # AVX2 + OpenMP implementation
    ├── mandelbrot_avx512.cpp       # AVX512 implementation 
    ├── mandelbrot_avx512_omp.cpp   # AVX512 + OpenMP implementation 
    ├── mandelbrot_omp.cpp          # OpenMP implementation
    ├── mandelbrot_serial.cpp       # Serial implementation
    └── utility.cpp                 # Helper functions
```

---

## Future work
* Runtime dispatch to select the best implementation.
* Visualization examples.
* CUDA/HIP support for GPU acceleration.
* MSVC support.

---

## Notes

* All implementations other than the serial implementation have been guarded by their appropriate compile-time checks.
* Runtime checks are ran for implementations depending on specific CPU support, such as the AVX2 implementations. It will fallback to the serial implementation. This is to prevent crashing due to unsupported instructions if the compiler has been set to still generate those instructions.
