// Defines the storage of the matrix and some helper methods.
#pragma once
#include <iomanip>
#include <vector>

namespace Solve {

template <typename Floating, std::enable_if_t<std::is_floating_point<Floating>::value, bool> = true> struct row {
    std::unique_ptr<Floating[]> data;
    const size_t size;
    explicit row(const size_t s) : size(s) { data = std::make_unique<Floating[]>(s); }

    explicit row(const std::vector<Floating> &d) : size(d.size()) {
        data = std::make_unique<Floating[]>(size);
        std::copy(std::begin(d), std::end(d), begin());
    }
    explicit row(const row &rhs) : size(rhs.size) {
        data = std::make_unique<Floating[]>(size);
        std::copy(std::begin(rhs), std::end(rhs), begin());
    }
    row &operator=(const row &) = delete;
    row(row &&) = default;
    row &operator=(row &&) = default;
    Floating *begin() const { return data.get(); }
    Floating *end() const { return data.get() + size; }

    Floating &operator[](size_t index) { return data.get()[index]; }
    const Floating &operator[](size_t index) const { return data.get()[index]; }
};

template <typename Floating, std::enable_if_t<std::is_floating_point<Floating>::value, bool> = true> struct matrix {
    std::unique_ptr<Floating[]> data; // rows*columns list of data points
    const size_t rows;
    const size_t columns;

    matrix(const size_t rows, const size_t columns) : rows(rows), columns(columns) {
        data = std::make_unique<Floating[]>(rows * columns);
    }
    explicit matrix(const std::vector<std::vector<Floating>> &d) : rows(d.size()), columns(d[0].size()) {
        if (rows + 1 != columns) {
            throw "wrong shape in alloc";
        }

        data = std::make_unique<Floating[]>(rows * columns);
        for (size_t r = 0; r < rows; ++r) {
            std::copy(std::begin(d[r]), std::end(d[r]), data.get() + r * columns);
        }
    }
    explicit matrix(const matrix &rhs) : rows(rhs.rows), columns(rhs.columns) {
        data = std::make_unique<Floating[]>(rows * columns);
        std::copy(rhs.data.get(), rhs.data.get() + rows * columns, data.get());
    }

    matrix &operator=(const matrix &) = delete;
    matrix(matrix &&) = default;
    matrix &operator=(matrix &&) = default;

    Floating &at(const size_t row, const size_t column) {
#ifdef _DEBUG
        if (!(row < rows))
            throw "bad row index";
        if (!(column < columns))
            throw "bad column index";
#endif // _DEBUG

        return data[row * columns + column];
    }
    const Floating &at(const size_t row, const size_t column) const {
#ifdef _DEBUG
        if (!(row < rows))
            throw "bad row index";
        if (!(column < columns))
            throw "bad column index";
#endif // _DEBUG
        return data[row * columns + column];
    }
};

template <typename Floating> void print_mat(const matrix<Floating> &m) {
    // buffering is very important for windows terminal, ss facilitates that
    std::stringstream ss;
    for (auto &&r : m) {
        std::for_each(begin(r), end(r), [&ss](auto v) { ss << std::setw(3) << v << " "; });
        ss << '\n';
    }
    std::string str = ss.str();
    std::puts(str.c_str());
}

// Get the buffer size needed for print_mat_shape
template <typename Floating> size_t print_mat_shape_buffer_size(const matrix<Floating> &m) {
    const size_t rows = m.size();
    const size_t columns = m[0].size();
    return rows * columns + rows + 1 + 2 * (columns + 1);
}

// prints x's where the matrix has a non-zero value
// get the needed buffer size from print_mat_shape_buffer_size
// interest puts a marker above and below a column to help debug
template <typename Floating> void print_mat_shape(const matrix<Floating> &m, char *buffer, const size_t interest) {
    // buffering is very important for windows terminal, in this method we know the size ahead of time
    const size_t rows = m.size();
    const size_t columns = m[0].size();
    size_t n = 0;
    for (size_t i = 0; i < columns; ++i) {
        buffer[n] = '-';
        ++n;
    }
    buffer[n] = '\n';
    ++n;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j) {
            char c = m[i][j] == 0 ? '.' : 'x';
            buffer[n] = c;
            ++n;
        }
        buffer[n] = '\n';
        ++n;
    }
    for (size_t i = 0; i < columns; ++i) {
        buffer[n] = '-';
        ++n;
    }
    buffer[n] = '\n';
    ++n;
    buffer[n] = 0;
    buffer[interest] = '|';
    buffer[rows * columns + rows + columns + 1 + interest] = '|';
    std::puts(buffer);
}

// We are solving the equation Ax = b, with A as a matrix and x & b as vectors.
// We expect a square matrix for A, augmented with b on the end of each row.
template <typename Floating> void check_shape(const matrix<Floating> &data) {
#ifndef NDEBUG
    // TODO boo exceptions
    if (data.columns != data.rows + 1) {
        throw "badly shaped matrix";
    }
#endif // NDEBUG
}

template <typename Floating> void swap_row(matrix<Floating> &data, std::size_t row1, std::size_t row2) {
    // TODO convert to pointers
    for (size_t i = 0; i < data.columns; ++i) {
        std::swap(data.at(row1, i), data.at(row2, i));
    }
}
} // namespace Solve
