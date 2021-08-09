cmake_minimum_required (VERSION 3.14)

project("sycl-circuits" CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SYCL_IMPL "DPCPP" CACHE STRING "Which SYCL implementation to use")
set(IMPL_OPTIONS "DPCPP" "COMPUTECPP")
set_property(CACHE SYCL_IMPL PROPERTY STRINGS IMPL_OPTIONS)
if (NOT SYCL_IMPL IN_LIST IMPL_OPTIONS)
  message(FATAL_ERROR "Wrong value of the parameter 'SYCL_IMPL'")
endif()

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

enable_testing()


if (SYCL_IMPL STREQUAL "COMPUTECPP")
	if (NOT DEFINED ComputeCpp_DIR)
		message( FATAL_ERROR "ComputeCpp_DIR must be defined e.g. -DComputeCpp_DIR=\"C:\\Program Files\\Codeplay\\ComputeCpp\"" )
	endif()
		if (WIN32 AND NOT MSVC )
		message( FATAL_ERROR "ComputeCPP requires the MSVC compiler on windows, use the visual studio developer console (or the IDE)" )
	endif()

	list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
	include(FindComputeCpp)

	find_package(ComputeCpp REQUIRED)
endif()


add_subdirectory("simulate")
