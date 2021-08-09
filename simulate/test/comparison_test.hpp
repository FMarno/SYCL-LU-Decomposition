// For a bit of extra assurance that the SYCL implemenation works, we will compare it against the other implementations
#pragma once

#include <optional>
#include <string>

#include "graph/circuit_gen.hpp"
#include "graph/graph.hpp"
#include "graph/sta.hpp"
#include "solve/basic_gaussian.hpp"
#include "solve/gaussian_lu.hpp"
#include "solve/matrix.hpp"
#include "solve/sycl_gaussian_lu.hpp"

std::optional<std::string> comparison_test() {
    // set up SYCL
    sycl::gpu_selector selector;

    sycl::queue q{selector, [](sycl::exception_list el) {
                      for (auto &&ex : el) {
                          try {
                              std::rethrow_exception(ex);
                          } catch (sycl::exception const &e) {
                              std::fputs("Caught asynchronous SYCL exception: ", stdout);
                              std::puts(e.what());
                          }
                      }
                  }};

    using FloatType = double;
    for (int i = 0; i < 10; ++i) {
        // setup a graph to test
        auto graph = Graph::generate_circuit<FloatType>(100, 300);
        auto system = Graph::graph_to_sta_system(graph);
        auto matrix_size = system.rows;

        // run solvers
        auto gaussian_lu_solution = Solve::Gaussian_LU<FloatType>::solve(matrix{system});
        auto SYCL_solution = Solve::SYCL_Gaussian_LU<FloatType>::solve(q, std::move(system));

        // compare
        constexpr FloatType delta = 1e-6; // an allowable difference
        for (size_t n = 0; n < matrix_size; ++n) {
            if (std::abs(gaussian_lu_solution[n] - SYCL_solution[n]) > delta) {
                return "Comparison test "+std::to_string(i)+": values at index " + std::to_string(n) + " doesn't match. expected " +
                       std::to_string(gaussian_lu_solution[n]) + ", got " + std::to_string(SYCL_solution[n]) +
                       ". diff:" + std::to_string(std::abs(SYCL_solution[n] - gaussian_lu_solution[n]));
            }
        }
    }
    return {};
}
