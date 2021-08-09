// test the different linear system solvers
#pragma once

#include <optional>
#include <string>

#include "solve/basic_gaussian.hpp"
#include "solve/gaussian_lu.hpp"
#include "solve/sycl_gaussian_lu.hpp"

using Solve::matrix;
using Solve::row;
template <typename T> using vec = std::vector<T>;

// data type we are testing
using dtype = double;
using test = std::pair<matrix<dtype>, vec<dtype>>;

const auto basic_tests =
    vec<test>{std::make_pair(matrix<dtype>({{6, 2, 8, 26}, {3, 5, 2, 8}, {0, 8, 2, -7}}), vec<dtype>{4, -1, 0.5}),
              std::make_pair(matrix<dtype>({{0.0004, 1.402, 1.406}, {0.4003, -1.502, 2.501}}), vec<dtype>{dtype{10}, 1}),
              std::make_pair(matrix<dtype>({{1, -1, 1, 0}, {1, -1, 2, 2}, {1, 2, 2, 1}}),
                             vec<dtype>{dtype{-7} / dtype{3}, dtype{-1} / dtype{3}, dtype{2}})};

template <typename Solver, typename Det> std::optional<std::string> test_cases(Solver solve, Det determinant, std::vector<test> tests) {
    for (auto &&test : tests) {
        auto det = determinant(matrix{test.first});
        if (!std::isnormal(det)) {
            return "non-normal det";
        }
        auto solutions = solve(matrix{test.first});

        auto expected_solutions = test.second;
        if (solutions.size != expected_solutions.size()) {
            return "solution is wrong shape";
        }
        for (std::size_t i = 0; i < solutions.size; ++i) {
            if (solutions[i] != expected_solutions[i]) {
                return "values[" + std::to_string(i) + "] not same: " + std::to_string(solutions[i]) +
                       "(calculated) != " + std::to_string(expected_solutions[i]) + "(expected)";
            }
        }
    }
    return {};
}

std::optional<std::string> test_system_solvers() {
    auto result = test_cases(&Solve::Basic_Gaussian<dtype>::solve, &Solve::Basic_Gaussian<dtype>::determinant, basic_tests);
    if (result) {
        return "Basic_Gaussian: " + result.value();
    }
    result = test_cases(&Solve::Gaussian_LU<dtype>::solve, &Solve::Gaussian_LU<dtype>::determinant, basic_tests);
    if (result) {
        return "LU_Gaussian: " + result.value();
    }

    // set up SYCL
    sycl::default_selector selector;

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
    result =
        test_cases([&q](auto system) { return Solve::SYCL_Gaussian_LU<dtype>::solve(q, std::move(system)); },
                   [&q](auto system) { return Solve::SYCL_Gaussian_LU<dtype>::determinant(q, std::move(system)); }, basic_tests);
    if (result) {
        return "SYCL_Gaussian: " + result.value();
    }
    return {};
}