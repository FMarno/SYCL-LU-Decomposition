// convert a list of branches into the Sparse Tableau Analysis form, which can then be solved as a linear system of equations
#pragma once

#include <cassert>
#include <set>

#include "graph/graph.hpp"
#include "graph/sta.hpp"
#include "solve/matrix.hpp"

using Solve::matrix;
using Solve::row;

namespace Graph {

// convert a graph to a linear system
template <typename Floating> matrix<Floating> graph_to_sta_system(const Branches<Floating> &branches) {
    const auto num_of_nodes = count_nodes(branches.paths);
    const auto num_of_branches = branches.paths.size();
    const auto matrix_size = 2 * num_of_branches + num_of_nodes - 1;
    matrix<Floating> system(matrix_size, matrix_size + 1);

    // identity matrix in the middle
    for (size_t i = 0; i < num_of_branches; ++i) {
        system.at(num_of_nodes - 1 + i,num_of_branches + i) = 1;
    }

    auto &paths = branches.paths;
    for (size_t i = 0; i < paths.size(); ++i) {
        // reduced incidence matrix, should be num_of_nodes-1 x num_of_branches
        // also fill in the negated transpose
        auto &path = paths[i];
        if (path.start_node != 0) {
            system.at(path.start_node - 1,i) = Floating{1};
            system.at(num_of_nodes + i - 1,2 * num_of_branches + path.start_node - 1) = Floating{-1};
        }
        if (path.end_node != 0) {
            system.at(path.end_node - 1,i) = Floating{-1};
            system.at(num_of_nodes + i - 1,2 * num_of_branches + path.end_node - 1) = Floating{1};
        }
    }

    auto &components = branches.components;
    for (size_t i = 0; i < components.size(); ++i) {
        // Z,Y, and s
        std::visit(
            [&system, i, num_of_nodes, num_of_branches, matrix_size](auto &&component) {
                using T = std::decay_t<decltype(component)>;
                const auto start = num_of_nodes - 1 + num_of_branches;
                if constexpr (std::is_same_v<T, Resistor<Floating>>) {
                    // Z
                    system.at(start + i, i) = component.value;
                    // Y
                    system.at(start + i, num_of_branches + i) = Floating{-1};
                    // s is 0
                } else if constexpr (std::is_same_v<T, VoltageSource<Floating>>) {
                    system.at(start + i, num_of_branches + i) = Floating{1};
                    system.at(start + i, matrix_size) = component.value;
                } else if constexpr (std::is_same_v<T, CurrentSource<Floating>>) {
                    system.at(start + i, i) = Floating{1};
                    system.at(start + i, matrix_size) = component.value;
                } else if constexpr (std::is_same_v<T, VoltageControlledVoltageSource<Floating>>) {
                    assert(false);
                } else if constexpr (std::is_same_v<T, VoltageControlledCurrentSource<Floating>>) {
                    assert(false);
                } else if constexpr (std::is_same_v<T, CurrentControlledVoltageSource<Floating>>) {
                    assert(false);
                } else if constexpr (std::is_same_v<T, CurrentControlledCurrentSource<Floating>>) {
                    assert(false);
                } else {
                    static_assert(always_false_v<T>, "non-exhaustive");
                }
            },
            components[i]);
    }

    return system;
}

// counts the number of unique nodes, including 0
inline std::size_t count_nodes(const std::vector<Path> &paths) noexcept {
    size_t count = 0;
    std::set<std::size_t> nodes;
    for (auto &&path : paths) {
        if (nodes.insert(path.start_node).second) {
            count += 1;
        }
        if (nodes.insert(path.end_node).second) {
            count += 1;
        }
    }
    return count;
}

// prints out the result of a solve in a nice format
template <typename Floating> void sta_print_solution(const Branches<Floating> &branches, const std::vector<Floating> &solution) {
    auto &paths = branches.paths;
    auto &components = branches.components;
    const auto num_of_nodes = count_nodes(paths);
    const auto num_of_branches = paths.size();
    const auto matrix_size = 2 * num_of_branches + num_of_nodes - 1;
    std::stringstream ss;
    for (size_t i = 0; i < num_of_branches; ++i) {
        ss << paths[i].start_node << " -> " << to_string(components[i]) << " -> " << paths[i].end_node << ": current "
           << solution[i] << "\tvoltage " << solution[num_of_branches + i] << "\n";
    }
    for (size_t n = 1; n < num_of_nodes; ++n) {
        ss << "Node " << n << " voltage: " << solution[matrix_size - num_of_nodes + n] << "\n";
    }
    std::puts(ss.str().c_str());
}
} // namespace Graph
