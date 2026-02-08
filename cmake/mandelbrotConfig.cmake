set(MANDELBROT_INCLUDE_DIRS "@CMAKE_INSTALL_PREFIX@/include")
set(MANDELBROT_LIBRARIES "@CMAKE_INSTALL_PREFIX@/lib/libmandelbrot.a")

add_library(mandelbrot::mandelbrot UNKNOWN IMPORTED)
set_target_properties(mandelbrot::mandelbrot PROPERTIES
    IMPORTED_LOCATION "@CMAKE_INSTALL_PREFIX@/lib/libmandelbrot.a"
    INTERFACE_INCLUDE_DIRECTORIES "@CMAKE_INSTALL_PREFIX@/include"
)
