// Perform Gaussian elimination of a linear system of equations.

#pragma once
#include "matrix.hpp"
#include <cassert>

namespace Solve {
template <typename Floating> struct Basic_Gaussian {

  private:
    static void triangularise(matrix<Floating> &data) {
        const auto matrix_size = data.rows;
        for (size_t n = 0; n < matrix_size - 1; ++n) {
            { // find highest magnitude in this column
                auto biggest_row = n;
                for (auto row = biggest_row + 1; row < matrix_size; ++row) {
                    if (std::abs(data.at(biggest_row, n)) < std::abs(data.at(row, n))) {
                        biggest_row = row;
                    }
                }
                if (data.at(biggest_row, n) == 0) {
                    // at this point the whole column under data[n][n] is empty
                    // if a column swap happens, we will still eventually get to the
                    // column of zeros
                    // no unique solution exists
                } else {
                    swap_row(data, n, biggest_row);
                }
            }

            for (auto row = n + 1; row < matrix_size; ++row) {
                auto multiplier = data.at(row, n) / data.at(n, n);

                // HOT SPOT
                Floating *const target_row = &data.at(row, n + 1);
                Floating const *const current_row = &data.at(n, n + 1);
                const size_t len = (matrix_size + 1) - (n + 1);
                for (size_t i = 0; i < len; ++i) {
                    target_row[i] = target_row[i] - multiplier * current_row[i];
                }
            }
        }
    }

  public:
    static row<Floating> solve(matrix<Floating> data) {
        const auto matrix_size = data.rows;
        check_shape(data);

        triangularise(data);

        // back substitution
        row<Floating> solution(matrix_size);
        solution[matrix_size - 1] = data.at(matrix_size - 1, matrix_size) / data.at(matrix_size - 1, matrix_size - 1);
        for (size_t i = 1; i < matrix_size; ++i) {
            auto idx = matrix_size - 1 - i;
            Floating weighted_sum = 0;
            Floating const *const row = &data.at(idx, idx + 1);
            Floating const *const x = &solution[idx + 1];
            const auto len = matrix_size - (idx + 1);
            for (size_t i = 0; i < len; ++i) {
                weighted_sum += row[i] * x[i];
            }
            solution[idx] = (data.at(idx, matrix_size) - weighted_sum) / data.at(idx, idx);
        }
        return solution;
    }

    // If one row/column of A multiplied by α is added to another row/column to form B, det(B) = det(A)
    // the determinant of a triangular matrix determinant is the product of the diagonal entries
    static Floating determinant(matrix<Floating> data) noexcept {
        triangularise(data);
        auto product = Floating{1};
        for (size_t i = 0; i < data.rows; ++i) {
            product = product * data.at(i, i);
        }
        return product;
    }
};

} // namespace Solve
