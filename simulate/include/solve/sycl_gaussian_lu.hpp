#pragma once

#include "matrix.hpp"
#include <CL/sycl.hpp>

template <typename Floating, size_t sum_comparisons> class PartialSum;
template <typename Floating, size_t magnitude_row_swap_comparisons> class MagnitudeRowReduce;

template <typename Floating, size_t comps>
class SumBufferSwap;
template <typename Floating, size_t comps>
class LoadLU;
template <typename Floating, size_t comps>
class LoadB;
template <typename Floating, size_t comps>
class FillIndexes;
template <typename Floating, size_t comps>
class RowSwap;
template <typename Floating, size_t comps>
class CalcMultipliers;
template <typename Floating, size_t comps>
class FillL;
template <typename Floating, size_t comps>
class FillU;
template <typename Floating, size_t comps>
class ForwardZero;
template <typename Floating, size_t comps>
class ForwardOne;
template <typename Floating, size_t comps>
class ForwardTwo;
template <typename Floating, size_t comps>
class BackwardZero;
template <typename Floating, size_t comps>
class BackwardOne;
template <typename Floating, size_t comps>
class BackwardTwo;

namespace Solve {

  template <typename Floating, size_t comps = 64> struct SYCL_Gaussian_LU {
    static constexpr size_t magnitude_row_swap_comparisons = comps;
    static constexpr size_t sum_comparisons = comps;

    template <size_t dimensions, typename T>
    static void print_buffer(sycl::buffer<T, dimensions> &buf, sycl::range<dimensions> r,
                             sycl::id<dimensions> offset = sycl::id<dimensions>{}) {
        auto acc_buf = buf.template get_access<sycl::access::mode::read>();
        std::stringstream ss;
        if constexpr (dimensions == 1) {
            for (size_t i = 0; i < r[0]; ++i) {
                ss << std::setw(3) << acc_buf[i + offset[0]] << ' ';
            }
        } else if constexpr (dimensions == 2) {
            for (size_t i = 0; i < r[0]; ++i) {
                for (size_t j = 0; j < r[1]; ++j) {
                    ss << std::setw(3) << acc_buf[i + offset[0]][j + offset[1]] << ' ';
                }
                ss << '\n';
            }
        }
        auto str = ss.str();
        // use fputs to avoid newline thing.
        std::puts(str.c_str());
    }

    template <typename Number> static Number branchless_ternary(bool predicate, Number is_true, Number is_false) {
        // cast predicate to 1 for true or 0 for false
        return is_true * predicate + is_false * (!predicate);
    }

    static void swap_row(const size_t matrix_size, sycl::queue &q, sycl::buffer<Floating, 2> &data_buf,
                         sycl::buffer<size_t, 1> &swap, const size_t diagonal) {
        q.submit([&](sycl::handler &h) {
            // TODO try two accessors to data buf, one for each row
            auto acc_data = data_buf.template get_access<sycl::access::mode::read_write>(h);
            auto acc_swap = swap.template get_access<sycl::access::mode::read>(h, sycl::range<1>{1});
            h.parallel_for<class RowSwap<Floating, comps>>(sycl::range<1>{matrix_size + 1}, [=](sycl::id<1> id) {
                auto global_id = id[0];
                // TODO might be faster with no storage, only bit shifting or addition and subtraction
                auto tmp = acc_data[diagonal][global_id];
                acc_data[diagonal][global_id] = acc_data[acc_swap[0]][global_id];
                acc_data[acc_swap[0]][global_id] = tmp;
            });
        });
    }

    // The size of IndexesA should be at least (matrix_size-diagonal)
    // the size of IndexesB should be at least ((matrix_size-diagonal)/magnitude_row_swap_comparisons)+1
    static void magnitude_row_swap(const size_t matrix_size, sycl::queue &q, sycl::buffer<Floating, 2> &data_buf, sycl::buffer<size_t, 1> indexesA, sycl::buffer<size_t, 1> indexesB,
                                   const size_t diagonal) {

        auto items = matrix_size - diagonal;
        auto extra_values = items % magnitude_row_swap_comparisons;
        auto work_items = items / magnitude_row_swap_comparisons + (extra_values ? 1 : 0);
        // create a buffer of indexs from diagonal to matrix_size-1
        // compare all values at data[index][diagonal] by their magnitude
        // reducing the list of indexes until we have the index of the greatest magnitude value
        // swap the current row and the row with the greastest magnitude
        q.submit([&](sycl::handler &h) {
            auto acc_indexes = indexesA.template get_access<sycl::access::mode::write>(h);
            h.parallel_for<class FillIndexes<Floating, comps>>(sycl::range<1>{items}, [=](sycl::id<1> id) { acc_indexes[id] = id[0] + diagonal; });
        });

        auto partial_magnitude_reduce = [&](sycl::buffer<size_t, 1> &read, sycl::buffer<size_t, 1> &write, const size_t items,
                                            const size_t work_items, const size_t extra_values) {
            q.submit([&](sycl::handler &h) {
                auto acc_data = data_buf.template get_access<sycl::access::mode::read>(h);
                auto read_mem = read.template get_access<sycl::access::mode::read>(h);
                auto write_mem = write.template get_access<sycl::access::mode::write>(h);

                h.parallel_for<class MagnitudeRowReduce<Floating, magnitude_row_swap_comparisons>>(sycl::range<1>{work_items}, [=](sycl::id<1> id) {
                    const auto global_id = id[0];
                    auto greatest_idx = read_mem[global_id * magnitude_row_swap_comparisons];
                    auto greatest_mag = acc_data[greatest_idx][diagonal];
                    // abs
                    greatest_mag = greatest_mag * sycl::sign(greatest_mag);
                    const auto comparisons = branchless_ternary((extra_values && global_id == work_items - 1), extra_values,
                                                                magnitude_row_swap_comparisons);
                    for (size_t i = 1; i < comparisons; ++i) {
                        auto other_idx = read_mem[global_id * magnitude_row_swap_comparisons + i];
                        auto other_mag = acc_data[other_idx][diagonal];
                        // abs
                        other_mag = other_mag * sycl::sign(other_mag);
                        bool other_greater = greatest_mag < other_mag;
                        greatest_idx = branchless_ternary(other_greater, other_idx, greatest_idx);
                        greatest_mag = branchless_ternary(other_greater, other_mag, greatest_mag);
                    }
                    write_mem[global_id] = greatest_idx;
                });
            });
        };

        bool write_to_A = false;
        while (items != 1) {
            if (write_to_A) {
                partial_magnitude_reduce(indexesB, indexesA, items, work_items, extra_values);
            } else {
                partial_magnitude_reduce(indexesA, indexesB, items, work_items, extra_values);
            }
            write_to_A = !write_to_A;
            items = work_items;
            extra_values = items % magnitude_row_swap_comparisons;
            work_items = items / magnitude_row_swap_comparisons + (extra_values ? 1 : 0);
        }

        // If write_to_A then the answer is in B[0]
        swap_row(matrix_size, q, data_buf, write_to_A ? indexesB : indexesA, diagonal);
    }

    // multipliers should be at least matrix_size - 1 long
    // indexesA should be at least matrix_size long
    // indexesB should be at least matrix_size/magnitude_row_swap_comparisons + 1  long
    static void get_gaussian_LU(const size_t matrix_size, sycl::queue &q, sycl::buffer<Floating, 2> &data_buf,
                                sycl::buffer<Floating, 1> multipliers, sycl::buffer<size_t, 1> indexesA,
                                sycl::buffer<size_t, 1> indexesB) {
        for (size_t n = 0; n < matrix_size - 1; ++n) {
            magnitude_row_swap(matrix_size, q, data_buf,indexesA, indexesB, n);

            // calculate the multipliers
            const size_t num_of_multipliers = matrix_size - n - 1;
            q.submit([&, n](sycl::handler &h) {
                auto multiplier_range = sycl::range<1>{num_of_multipliers};
                auto acc_multiplier = multipliers.template get_access<sycl::access::mode::discard_write>(h, multiplier_range);
                auto acc_data = data_buf.template get_access<sycl::access::mode::read>(h);
                h.parallel_for<class CalcMultipliers<Floating, comps>>(multiplier_range, [=](sycl::id<1> id) {
                    auto row = n + 1 + id[0];
                    acc_multiplier[id] = acc_data[row][n] / acc_data[n][n];
                });
            });

            // fill in the L values
            q.submit([&, n](sycl::handler &h) {
                auto acc_multiplier = multipliers.template get_access<sycl::access::mode::read>(h);
                auto acc_data = data_buf.template get_access<sycl::access::mode::write>(h);

                auto multiplier_range = sycl::range<1>{num_of_multipliers};
                h.parallel_for<class FillL<Floating, comps>>(multiplier_range, [=](sycl::id<1> id) {
                    const auto row = n + 1 + id[0];
                    acc_data[row][n] = acc_multiplier[id];
                });
            });

            // prepare U values
            q.submit([&, n](sycl::handler &h) {
                auto acc_multiplier = multipliers.template get_access<sycl::access::mode::read>(h);
                auto acc_data = data_buf.template get_access<sycl::access::mode::read_write>(h);
                h.parallel_for<class FillU<Floating, comps>>(sycl::range<2>{num_of_multipliers, num_of_multipliers}, [=](sycl::id<2> id) {
                    auto row = n + 1 + id[0];
                    auto column = n + 1 + id[1];
                    acc_data[row][column] = acc_data[row][column] - acc_multiplier[id[0]] * acc_data[n][column];
                });
            });
        }
    }

    // values buffer should be values_size long
    // scratch buffer should be (values_size/sum_comparisons) +1 long
    static void sycl_sum(sycl::queue &q, sycl::buffer<Floating, 1> &values, size_t values_size, sycl::buffer<Floating,1>& scratch) {
        // the values sums will be written to values, then buffer, then values, then buffer...
        auto items = values_size;
        auto extra_values = items % sum_comparisons;
        auto work_items = items / sum_comparisons + (extra_values ? 1 : 0);

        auto partial_sum = [&q](sycl::buffer<Floating, 1> &read, sycl::buffer<Floating, 1> &write, const size_t items,
                                const size_t work_items, const size_t extra_values) {
            q.submit([&](sycl::handler &h) {
                auto read_mem = read.template get_access<sycl::access::mode::read>(h);
                auto write_mem = write.template get_access<sycl::access::mode::write>(h);

                h.parallel_for<class PartialSum<Floating, sum_comparisons>>(sycl::range<1>{work_items}, [=](sycl::id<1> id) {
                                    Floating sum{0};
                                    const auto global_id = id[0];
                                    const auto comparisons = branchless_ternary((extra_values && global_id == work_items - 1),
                                                                                extra_values, sum_comparisons);
                                    for (size_t i = 0; i < comparisons; ++i) {
                                        auto idx = global_id * sum_comparisons + i;
                                        sum += read_mem[idx];
                                    }
                                    write_mem[global_id] = sum;
                                });
            });
        };

        bool write_to_values = false;
        while (items != 1) {
            if (write_to_values) {
                partial_sum(scratch, values, items, work_items, extra_values);
            } else {
                partial_sum(values, scratch, items, work_items, extra_values);
            }
            write_to_values = !write_to_values;
            items = work_items;
            extra_values = items % sum_comparisons;
            work_items = items / sum_comparisons + (extra_values ? 1 : 0);
        }

        if (write_to_values) {
            // The result is at the start of buffer, we want it at the start of values
            q.submit([&](sycl::handler &h) {
                auto acc_buffer = scratch.template get_access<sycl::access::mode::read>(h, sycl::range<1>{1});
                auto acc_values = values.template get_access<sycl::access::mode::write>(h, sycl::range<1>{1});
                h.single_task<class SumBufferSwap<Floating, comps>>([=] { acc_values[0] = acc_buffer[0]; });
            });
        }
    }

    // Sum buffer should be matrix_size-1 long
    // scratch buffer should be (matrix_size-1)/sum_comparisons +1 long
    static void forward_propagation(const size_t matrix_size, sycl::queue &q, sycl::buffer<Floating, 2> &LU_buf,
                                    sycl::buffer<Floating, 1> &y_buf, sycl::buffer<Floating, 1> &sum_buffer, sycl::buffer<Floating, 1>& scratch) {
        // -----------Calculate y-----------
        q.submit([&](sycl::handler &h) {
            auto acc_lu = LU_buf.template get_access<sycl::access::mode::read>(h);
            auto acc_y = y_buf.template get_access<sycl::access::mode::write>(h);
            h.single_task<class ForwardZero<Floating, comps>>([=]() { acc_y[0] = acc_lu[0][matrix_size]; });
        });
        for (size_t n = 1; n < matrix_size; ++n) {
            q.submit([&](sycl::handler &h) {
                auto acc_LU_line = LU_buf.template get_access<sycl::access::mode::read>(h);
                auto acc_y = y_buf.template get_access<sycl::access::mode::read>(h);
                auto acc_sum = sum_buffer.template get_access<sycl::access::mode::discard_write>(h);
                h.parallel_for<class ForwardOne<Floating, comps>>(sycl::range<1>{n},
                                                 [=](sycl::id<1> id) { acc_sum[id[0]] = acc_y[id[0]] * acc_LU_line[n][id[0]]; });
            });

            sycl_sum(q, sum_buffer, n, scratch);
            q.submit([&, n](sycl::handler &h) {
                auto acc_lu = LU_buf.template get_access<sycl::access::mode::read>(h);
                auto acc_sum = sum_buffer.template get_access<sycl::access::mode::read>(h, sycl::range<1>{1});
                auto acc_y = y_buf.template get_access<sycl::access::mode::write>(h);
                h.single_task<class ForwardTwo<Floating, comps>>([=]() { acc_y[n] = acc_lu[n][matrix_size] - acc_sum[0]; });
            });
        }
    }

    // Sum buffer should be matrix_size-1 long
    // scratch buffer should be (matrix_size-1)/sum_comparisons +1 long
    static void back_propagation(const size_t matrix_size, sycl::queue &q, sycl::buffer<Floating, 2> &LU_buf,
                                 sycl::buffer<Floating, 1> &y_buf, sycl::buffer<Floating, 1> &x_buf,
                                 sycl::buffer<Floating, 1> &sum_buffer, sycl::buffer<Floating, 1> &scratch) {
        q.submit([&](sycl::handler &h) {
            const auto idx = matrix_size - 1;
            auto acc_y = y_buf.template get_access<sycl::access::mode::read>(h);
            auto acc_lu = LU_buf.template get_access<sycl::access::mode::read>(h);
            auto acc_x = x_buf.template get_access<sycl::access::mode::discard_write>(h);
            h.single_task<class BackwardZero<Floating, comps>>([=] { acc_x[idx] = acc_y[idx] / acc_lu[idx][idx]; });
        });
        // -----------Calculate x-----------
        for (size_t n = 1; n < matrix_size; ++n) {
            const auto idx = (matrix_size - 1) - n;
            q.submit([&, n, idx](sycl::handler &h) {
                auto acc_LU_line = LU_buf.template get_access<sycl::access::mode::read>(h);
                auto acc_x = x_buf.template get_access<sycl::access::mode::read>(h);
                auto acc_sum = sum_buffer.template get_access<sycl::access::mode::discard_write>(h);
                h.parallel_for<class BackwardOne<Floating, comps>>(sycl::range<1>{n}, [=](sycl::id<1> id) {
                    acc_sum[id[0]] = acc_x[id[0] + idx + 1] * acc_LU_line[idx][id[0] + idx + 1];
                });
            });
            sycl_sum(q, sum_buffer, n, scratch);
            q.submit([&, idx](sycl::handler &h) {
                auto acc_sum = sum_buffer.template get_access<sycl::access::mode::read>(h);
                auto acc_y = y_buf.template get_access<sycl::access::mode::read>(h);
                auto acc_lu = LU_buf.template get_access<sycl::access::mode::read>(h);
                auto acc_x = x_buf.template get_access<sycl::access::mode::write>(h);
                h.single_task<class BackwardTwo<Floating, comps>>([=] { acc_x[idx] = (acc_y[idx] - acc_sum[0]) / acc_lu[idx][idx]; });
            });
        }
    }

    static void load_LU_data(const size_t matrix_size, sycl::queue &q, sycl::buffer<Floating, 2> &LU_buf,
                             matrix<Floating> &matrix) {
        Floating const *const data = matrix.data.get();
        q.submit([&](sycl::handler &h) {
            auto acc_lu = LU_buf.template get_access<sycl::access::mode::discard_write>(h);
            h.copy(data, acc_lu);
        });
    }

  public:
    static row<Floating> solve(sycl::queue &q, matrix<Floating> data) {
        check_shape(data);
        const auto matrix_size = data.rows;

        // A x = b
        sycl::buffer<Floating, 2> LU_buf(sycl::range<2>(matrix_size, matrix_size + 1));

        sycl::buffer<size_t, 1> indexesA(sycl::range<1>{matrix_size});
        sycl::buffer<size_t, 1> indexesB(sycl::range<1>{(matrix_size/magnitude_row_swap_comparisons)+1});

        //sycl::buffer<Floating, 1> multipliers{sycl::range<1>{matrix_size - 1}};
        sycl::buffer<Floating, 1> y_buf(matrix_size);
        sycl::buffer<Floating, 1> x_buf(matrix_size);
        sycl::buffer<Floating, 1> sum_buf{sycl::range<1>{matrix_size-1}};
        sycl::buffer<Floating, 1> scratch_buf{sycl::range<1>{(matrix_size-1)/sum_comparisons + 1}};

        load_LU_data(matrix_size, q, LU_buf, data);
        get_gaussian_LU(matrix_size, q, LU_buf, y_buf, indexesA, indexesB);
        forward_propagation(matrix_size, q, LU_buf, y_buf, sum_buf, scratch_buf);
        back_propagation(matrix_size, q, LU_buf, y_buf, x_buf, sum_buf, scratch_buf);

        row<Floating> x(matrix_size);
        Floating *const x_ptr = x.data.get();
        q.submit([&](sycl::handler &h) {
            auto acc_x = x_buf.template get_access<sycl::access::mode::read>(h);
            h.copy(acc_x, x_ptr);
        });
        // TODO what is this doing here?
        q.wait_and_throw();
        return x;
    }

    static Floating determinant(sycl::queue &q, matrix<Floating> data) noexcept {
        const auto matrix_size = data.rows;

        sycl::buffer<Floating, 2> LU_buf(sycl::range<2>(matrix_size, matrix_size + 1));
        load_LU_data(matrix_size, q, LU_buf, data);

        sycl::buffer<Floating, 1> multipliers{sycl::range<1>{matrix_size - 1}};
        sycl::buffer<size_t, 1> indexesA(sycl::range<1>{matrix_size});
        sycl::buffer<size_t, 1> indexesB(sycl::range<1>{(matrix_size/magnitude_row_swap_comparisons)+1});
        get_gaussian_LU(matrix_size, q, LU_buf, multipliers, indexesA, indexesB);
        // TODO do this on GPU
        auto s = LU_buf.template get_access<sycl::access::mode::read>();
        auto product = Floating{1};
        for (auto i = std::size_t{0}; i < matrix_size; ++i) {
            product = product * s[i][i];
        }
        return product;
    }
};
} // namespace Solve
