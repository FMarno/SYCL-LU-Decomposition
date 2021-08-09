// test the creation of a STA system
#pragma once

#include "graph/graph.hpp"
#include "graph/sta.hpp"
#include "solve/matrix.hpp"
#include <optional>
#include <vector>

// data type we are testing
using Floating = double;

using Graph::Branches;
using Graph::CurrentSource;
using Graph::Resistor;
using Graph::VoltageSource;

std::optional<std::string> test_sta() {
    Branches<Floating> branches;
    branches.add(0, 1, Resistor<Floating>{1});
    branches.add(1, 2, Resistor<Floating>{1});
    branches.add(2, 0, VoltageSource<Floating>{5});
    branches.add(0, 1, Resistor<Floating>{1});
    branches.add(0, 2, Resistor<Floating>{1});

    const auto num_branches = 5;   // branches.size();
    const auto num_nodes = 3;      // Graph::count_nodes(branches.paths);
    const auto expected_size = 2*num_branches + (num_nodes-1); // 2*num_branches + num_nodes -1;
    auto system = Graph::graph_to_sta_system(branches);
    if (expected_size != system.rows) {
        return "system: wrong amount of rows: " + std::to_string(expected_size) + " != "+ std::to_string(system.rows);
    };
    // augmented matrix form
    if (expected_size + 1 != system.columns) {
        return "system: wrong amount of columns: " + std::to_string(expected_size + 1) + " != "+ std::to_string(system.columns);
    }

    // clang-format off
    std::vector<std::vector<Floating>> expected_system{
        {-1, 1, 0, -1, 0,   0, 0, 0, 0, 0,    0, 0,   0},
        {0, -1, 1, 0, -1,   0, 0, 0, 0, 0,    0, 0,   0},

        {0, 0, 0, 0, 0,     1, 0, 0, 0, 0,    1, 0,   0},
        {0, 0, 0, 0, 0,     0, 1, 0, 0, 0,    -1, 1,  0},
        {0, 0, 0, 0, 0,     0, 0, 1, 0, 0,    0, -1,  0},
        {0, 0, 0, 0, 0,     0, 0, 0, 1, 0,    1, 0,   0},
        {0, 0, 0, 0, 0,     0, 0, 0, 0, 1,    0, 1,   0},

        {1, 0, 0, 0, 0,     -1, 0, 0, 0, 0,   0, 0,   0},
        {0, 1, 0, 0, 0,     0, -1, 0, 0, 0,   0, 0,   0},
        {0, 0, 0, 0, 0,     0, 0,  1, 0, 0,   0, 0,   5},
        {0, 0, 0, 1, 0,     0, 0, 0, -1, 0,   0, 0,   0},
        {0, 0, 0, 0, 1,     0, 0, 0, 0, -1,   0, 0,   0},
    };
    // clang-format on
    // auto buff = std::make_unique<char[]>(Solve::print_mat_shape_buffer_size(system));
    // Solve::print_mat_shape(system, buff.get(), 0);

    if (expected_size != expected_system.size()) {
        return "expected system: wrong amount of rows: " + std::to_string(expected_size) + " != "+ std::to_string(expected_system.size());
    }
    // augmented matrix form
    if (expected_size + 1 != expected_system[0].size()) {
        return "expected system: wrong amount of columns: " + std::to_string(expected_size + 1) + " != "+ std::to_string(expected_system[0].size());
    }

    for (int i = 0; i < expected_size; ++i) {
        for (int j = 0; j < expected_size + 1; ++j) {
            if (system.at(i,j) != expected_system[i][j]) {
             return "system mismatch at: <" + std::to_string(i) + "," + std::to_string(j) + ">";
            }
        }
    }
    return {}; // success
}
