find_package(OpenCV REQUIRED)

foreach(EXAMPLE greyscale rainbow)
  add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
  target_link_libraries(${EXAMPLE} PRIVATE mandelbrot ${OpenCV_LIBS})
  target_include_directories(${EXAMPLE} PRIVATE ${OpenCV_INCLUDE_DIRS})
  configure_target(${EXAMPLE})
endforeach()
