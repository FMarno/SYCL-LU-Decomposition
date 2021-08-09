// Perform LU factorisation with the Gaussian method.
// This can be helpful since the RHS of the Ax = b equation isn't needed.

#pragma once

#include "matrix.hpp"
#include <numeric>

namespace Solve {

template <typename Floating> class Gaussian_LU {
  private:
    // compare rows by magnitude
    static size_t magnitude_row_swap(const matrix<Floating> &data, const size_t n) {
        const auto matrix_size = data.rows;
        auto max_magnitude = std::abs(data.at(n, n));
        auto max_index = n;
        for (auto row = n + 1; row < matrix_size; ++row) {
            const auto candidate = std::abs(data.at(row, n));
            if (max_magnitude < candidate) {
                max_magnitude = candidate;
                max_index = row;
            }
        }
        return max_index;
    }

    static void gaussian_LU(matrix<Floating> &data) {
        const auto matrix_size = data.rows;
        for (size_t n = 0; n < matrix_size - 1; ++n) {
            // find highest magnitude in this column
            // First find Markowitz pivot from submatrix
            auto target_row = magnitude_row_swap(data, n);
            if (target_row != n) {
                swap_row(data, n, target_row);
            }

            // now fill in L values in data, and prepare U values in data
            for (auto row = n + 1; row < matrix_size; ++row) {
                auto multiplier = data.at(row, n) / data.at(n, n);
                data.at(row, n) = multiplier;
                // very hot spot
                // let's make this obvious for the compiler
                // These pointers outperformed std::transform in release and debug (especially with std::execution::par_unseq)
                // and vector indexes in debug, similar to vector indexes in release
                Floating *const a = &data.at(row, n + 1);
                Floating const *const b = &data.at(n, n + 1);
                size_t len = matrix_size - (n + 1);

                for (size_t i = 0; i < len; ++i) {
                    a[i] = a[i] - multiplier * b[i];
                }
                // std::transform(&data[row][n + 1], &data[row][matrix_size], &data[n][n + 1], &data[row][n + 1],
                // [multiplier](auto a, auto b){return a - multiplier*b;});
            }
        }
    }

  public:
    static row<Floating> solve(matrix<Floating> data) {
        check_shape(data);
        const auto matrix_size = data.rows;

        gaussian_LU(data);
        row<Floating> y(matrix_size);
        for (size_t n = 0; n < matrix_size; ++n) {
            // const auto sum = std::transform_reduce(begin(y), begin(y) + n, begin(data[n]), Floating{0});
            Floating sum = 0;
            Floating *y_ptr = &y[0];
            Floating *row = &data.at(n, 0);
            for (size_t i = 0; i < n; ++i) {
                sum += y_ptr[i] * row[i];
            }
            y[n] = data.at(n, matrix_size) - sum;
        }

        row<Floating> x(matrix_size);
        x[matrix_size - 1] = y[matrix_size - 1] / data.at(matrix_size - 1, matrix_size - 1);
        for (size_t n = 1; n < matrix_size; ++n) {
            auto idx = matrix_size - 1 - n;
            // const auto sum = std::transform_reduce(begin(x) + idx + 1, end(x), begin(&data.at(idx,idx+1), Floating{0});
            Floating sum = 0;
            Floating *row = &data.at(idx, idx + 1);
            Floating *x_ptr = &x[idx + 1];
            for (size_t i = 0; i < n; ++i) {
                sum += row[i] * x_ptr[i];
            }
            x[idx] = (y[idx] - sum) / data.at(idx, idx);
        }
        return x;
    }

    // a non-singular matrix has an inverse, and is thus solvable.
    // a square matrix is non-singular (solvable), if and only if the determinant
    // is non-zero a matrix and it's transpose has the same determinant.
    // det(AB) = det(A)det(B).
    // det(A^k) = det(A)^k.
    // If two row or columns of A are swapped to form B, det(B) = -det(A)
    // If one row or column of A is multiplied by α to form B, det(B) = αdet(A).
    // If one row/column of A multiplied by α is added to another row/column to form B, det(B) = det(A)
    // the determinant of a triangular matrix determinant is the product of the diagonal entries
    static Floating determinant(matrix<Floating> data) noexcept {
        gaussian_LU(data);
        auto product = Floating{1};
        for (auto i = std::size_t{0}; i < data.rows; ++i) {
            product = product * data.at(i, i);
        }
        return product;
    }
};
} // namespace Solve
