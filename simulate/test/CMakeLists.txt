cmake_minimum_required (VERSION 3.9)

set(TEST_FILES
  system_solver_test.hpp
  sta_test.hpp
  sycl_solver_test.hpp
  comparison_test.hpp
)
# Now simply link against gtest or gtest_main as needed. Eg
add_executable(simulate_test test_main.cpp ${TEST_FILES})
target_include_directories(simulate_test PRIVATE ../include)
add_test(NAME simulate_test COMMAND simulate)
target_link_libraries(simulate_test PRIVATE OpenCL)

if (SYCL_IMPL STREQUAL "COMPUTECPP")
  add_sycl_to_target(TARGET simulate_test SOURCES test_main.cpp)
else()
  # target_link_libraries(simulate_test PRIVATE project_warnings)
endif()
