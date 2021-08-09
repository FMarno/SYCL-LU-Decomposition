#pragma once
#include "solve/sycl_gaussian_lu.hpp"
#include <optional>
#include <string>

using Floating = double;
std::optional<std::string> test_data_load(sycl::queue &q);
std::optional<std::string> test_back_propagate(sycl::queue &q);
std::optional<std::string> test_forward_propagate(sycl::queue &q);
template <size_t comparisons>
std::optional<std::string> test_magnitude_row_swap(sycl::queue &q);
std::optional<std::string> test_LU_decomposition(sycl::queue &q);
std::optional<std::string> test_swap_row(sycl::queue &q);

std::optional<std::string> sycl_solver_test() {
    sycl::gpu_selector selector;
    sycl::queue q{selector,
                  [](sycl::exception_list el) {
                      for (auto &&ex : el) {
                          try {
                              std::rethrow_exception(ex);
                          } catch (sycl::exception const &e) {
                              std::fputs("Caught asynchronous SYCL exception: ", stdout);
                              std::puts(e.what());
                          }
                      }
                  }};

    std::puts(("\ntesting device: " + q.get_device().get_info<sycl::info::device::name>()).c_str());

    std::vector tests{&test_data_load,       &test_forward_propagate,     &test_back_propagate,
                      &test_swap_row,        &test_magnitude_row_swap<2>, &test_magnitude_row_swap<3>,
                      &test_LU_decomposition};

    for (auto &&test : tests) {
        auto ret = test(q);
        if (ret) {
            return ret;
        }
    }
    return {};
}

// test data loading
std::optional<std::string> test_data_load(sycl::queue &q) {
    using Solver = Solve::SYCL_Gaussian_LU<Floating>;
    // init
    const size_t matrix_size = 3;
    matrix<Floating> test_data{std::vector<std::vector<Floating>>{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}};
    sycl::buffer<Floating, 2> lu_buf{sycl::range<2>{matrix_size, matrix_size + 1}};

    // run test
    Solver::load_LU_data(matrix_size, q, lu_buf, test_data);

    auto host_acc_lu = lu_buf.template get_access<sycl::access::mode::read>();
    // compare
    for (size_t i = 0; i < matrix_size; ++i) {
        for (size_t j = 0; j < matrix_size + 1; ++j) {
            if (test_data.at(i, j) != host_acc_lu[i][j]) {
                auto test_info =
                    "test_data[" + std::to_string(i) + "][" + std::to_string(j) + "] = " + std::to_string(test_data.at(i, j));
                auto buf_info =
                    "acc_lu[" + std::to_string(i) + "][" + std::to_string(j) + "] = " + std::to_string(host_acc_lu[i][j]);
                return "test_data_load: matrix doesn't match lu buffer\n" + test_info + "\n" + buf_info;
            }
        }
    }
    return {};
}

std::optional<std::string> test_back_propagate(sycl::queue &q) {
    using Solver = Solve::SYCL_Gaussian_LU<Floating>;
    // init
    const size_t matrix_size = 3;
    // zero used where the value shouldn't be used
    matrix<Floating> test_data{std::vector<std::vector<Floating>>{{6, 2, 8, 0}, {0, 4, -2, 0}, {0, 0, 6, 0}}};
    sycl::buffer<Floating, 2> lu_buf{sycl::range<2>{matrix_size, matrix_size + 1}};
    Solver::load_LU_data(matrix_size, q, lu_buf, test_data);

    std::vector<Floating> y{26, -5, 3};
    sycl::buffer<Floating, 1> y_buf{y};

    sycl::buffer<Floating, 1> result_buf{sycl::range<1>{matrix_size}};
    sycl::buffer<Floating, 1> sum_buf{sycl::range<1>{matrix_size-1}};
    sycl::buffer<Floating, 1> scratch{sycl::range<1>{(matrix_size-1)/2 +1}};

    // run test
    Solver::back_propagation(matrix_size, q, lu_buf, y_buf, result_buf, sum_buf, scratch);

    std::vector<Floating> expected_solution{4, -1, 0.5};
    auto host_acc_result = result_buf.template get_access<sycl::access::mode::read>();
    // compare
    for (size_t i = 0; i < matrix_size; ++i) {
        if (expected_solution[i] != host_acc_result[i]) {
            auto test_info = "expected_solution[" + std::to_string(i) + "] = " + std::to_string(expected_solution[i]);
            auto buf_info = "acc_result[" + std::to_string(i) + "] = " + std::to_string(host_acc_result[i]);
            return "test_back_propagate: expected solution doesn't match result buffer\n" + test_info + "\n" + buf_info;
        }
    }
    return {};
}

std::optional<std::string> test_forward_propagate(sycl::queue &q) {
    using Solver = Solve::SYCL_Gaussian_LU<Floating>;
    // init
    const size_t matrix_size = 3;
    // zero used where the value shouldn't be used
    matrix<Floating> test_data{std::vector<std::vector<Floating>>{{0, 0, 0, 26}, {0.5, 0, 0, 8}, {0, 2, 0, -7}}};
    sycl::buffer<Floating, 2> lu_buf{sycl::range<2>{matrix_size, matrix_size + 1}};
    Solver::load_LU_data(matrix_size, q, lu_buf, test_data);

    sycl::buffer<Floating, 1> result_buf{sycl::range<1>{matrix_size}};
    sycl::buffer<Floating, 1> sum_buf{sycl::range<1>{matrix_size-1}};
    sycl::buffer<Floating, 1> scratch{sycl::range<1>{(matrix_size-1)/2 +1}};

    // run test
    Solver::forward_propagation(matrix_size, q, lu_buf, result_buf, sum_buf, scratch);

    std::vector<Floating> expected_solution{26, -5, 3};
    auto host_acc_result = result_buf.template get_access<sycl::access::mode::read>();
    // compare
    for (size_t i = 0; i < matrix_size; ++i) {
        if (expected_solution[i] != host_acc_result[i]) {
            auto test_info = "expected_solution[" + std::to_string(i) + "] = " + std::to_string(expected_solution[i]);
            auto buf_info = "acc_result[" + std::to_string(i) + "] = " + std::to_string(host_acc_result[i]);
            return "test_forward_propagate: expected solution doesn't match result buffer\n" + test_info + "\n" + buf_info;
        }
    }
    return {};
}

class SetRow;

std::optional<std::string> test_swap_row(sycl::queue &q) {
    using Solver = Solve::SYCL_Gaussian_LU<Floating>;
    // init
    const size_t matrix_size = 3;
    matrix<Floating> test_data{std::vector<std::vector<Floating>>{{6, 2, 8, 26}, {3, 5, 2, 8}, {0, 8, 2, -7}}};

    sycl::buffer<Floating, 2> lu_buf{sycl::range<2>{matrix_size, matrix_size + 1}};
    Solver::load_LU_data(matrix_size, q, lu_buf, test_data);

    // this will store which row to swap
    sycl::buffer<size_t, 1> target{sycl::range{1}};

    auto do_row_swap = [&q, &target, &lu_buf, matrix_size](size_t row1, size_t row2) {
        // swap row with self
        q.submit([&](sycl::handler &h) {
            auto acc_target = target.template get_access<sycl::access::mode::discard_write>(h);
            h.single_task<class SetRow>([=] { acc_target[0] = row1; });
        });
        Solver::swap_row(matrix_size, q, lu_buf, target, row2);
    };
    std::vector<std::pair<std::size_t, std::size_t>> swaps{{1, 1}, {0, 2}, {2, 1}};

    for (auto swap : swaps) {
        do_row_swap(swap.first, swap.second);
        Solve::swap_row(test_data, swap.first, swap.second);

        auto host_acc_lu = lu_buf.template get_access<sycl::access::mode::read>();
        for (size_t i = 0; i < matrix_size; ++i) {
            for (size_t j = 0; j < matrix_size + 1; ++j) {
                if (test_data.at(i, j) != host_acc_lu[i][j]) {
                    return "test swap_row: element at [" + std::to_string(i) + "][" + std::to_string(j) + "] doesn't match.";
                }
            }
        }
    }
    return {};
}

template <size_t comparisons>
std::optional<std::string> test_magnitude_row_swap(sycl::queue &q) {
  using Solver = Solve::SYCL_Gaussian_LU<Floating, comparisons>;
    // init
    const size_t matrix_size = 3;
    matrix<Floating> test_data{std::vector<std::vector<Floating>>{{100, 100, 100, 26}, {100, 5, 2, 8}, {0, -8, 2, -7}}};

    sycl::buffer<Floating, 2> lu_buf{sycl::range<2>{matrix_size, matrix_size + 1}};
    Solver::load_LU_data(matrix_size, q, lu_buf, test_data);

    sycl::buffer<size_t, 1> indexesA{sycl::range<1>{matrix_size}};
    sycl::buffer<size_t, 1> indexesB{sycl::range<1>{(matrix_size/comparisons)+1}};

    // run
    Solver::magnitude_row_swap(matrix_size, q, lu_buf,indexesA, indexesB, 1);

    // swap started on row two so we expect the third row will be chosen (std::abs(-8) > std::abs(5))
    matrix<Floating> expected_result{std::vector<std::vector<Floating>>{{100, 100, 100, 26}, {0, -8, 2, -7}, {100, 5, 2, 8}}};
    auto host_acc_lu = lu_buf.template get_access<sycl::access::mode::read>();
    for (size_t i = 0; i < matrix_size; ++i) {
        for (size_t j = 0; j < matrix_size + 1; ++j) {
            if (expected_result.at(i, j) != host_acc_lu[i][j]) {
                return "test magnitude row swap: element at [" + std::to_string(i) + "][" + std::to_string(j) +
                       "] doesn't match.";
            }
        }
    }

    return {};
}

std::optional<std::string> test_LU_decomposition(sycl::queue &q) {
    using Solver = Solve::SYCL_Gaussian_LU<Floating>;
    // init
    const size_t matrix_size = 3;
    matrix<Floating> test_data{std::vector<std::vector<Floating>>{{6, 2, 8, 26}, {3, 5, 2, 8}, {0, 8, 2, -7}}};

    sycl::buffer<Floating, 2> lu_buf{sycl::range<2>{matrix_size, matrix_size + 1}};
    Solver::load_LU_data(matrix_size, q, lu_buf, test_data);

    // run
    sycl::buffer<Floating, 1> multipliers{sycl::range<1>{matrix_size - 1}};
    sycl::buffer<size_t, 1> indexesA(sycl::range<1>{matrix_size});
    sycl::buffer<size_t, 1> indexesB(sycl::range<1>{(matrix_size/Solver::magnitude_row_swap_comparisons)+1});
    Solver::get_gaussian_LU(matrix_size, q, lu_buf, multipliers, indexesA, indexesB);

    // check result
    auto host_acc_lu = lu_buf.template get_access<sycl::access::mode::read>();
    matrix<Floating> expected_solution{std::vector<std::vector<Floating>>{{6, 2, 8, 26}, {0, 8, 2, -7}, {0.5, 0.5, -3, 8}}};

    for (size_t a = 0; a < matrix_size; ++a) {
        for (size_t b = 0; b < matrix_size + 1; ++b) {
            if (expected_solution.at(a, b) != host_acc_lu[a][b]) {
                return "test LU decomposition: expected solution[" + std::to_string(a) + "][" + std::to_string(b) +
                       "] = " + std::to_string(expected_solution.at(a, b)) + ", but instead got " +
                       std::to_string(host_acc_lu[a][b]);
            }
        }
    }

    return {};
}
