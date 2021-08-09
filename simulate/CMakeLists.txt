cmake_minimum_required (VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(HEADERS
  include/graph/circuit_gen.hpp
  include/graph/graph.hpp
  include/graph/sta.hpp
  include/solve/basic_gaussian.hpp
  include/solve/gaussian_lu.hpp
  include/solve/sycl_gaussian_lu.hpp
  include/solve/matrix.hpp
)

add_executable(simulate main.cpp ${HEADERS})
target_include_directories(simulate PRIVATE include)

if (SYCL_IMPL STREQUAL "COMPUTECPP")
  add_sycl_to_target(TARGET simulate SOURCES main.cpp)
else()
  # target_link_libraries(simulate PRIVATE project_warnings)
endif()

# TODO: Add tests and install targets if needed.
add_subdirectory(test)
