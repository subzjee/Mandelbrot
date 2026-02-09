find_package(OpenCV REQUIRED)

add_executable(greyscale greyscale.cpp)

target_link_libraries(greyscale PRIVATE mandelbrot ${OpenCV_LIBS})

target_include_directories(greyscale PRIVATE ${OpenCV_INCLUDE_DIRS})

configure_target(greyscale)
