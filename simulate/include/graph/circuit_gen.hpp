// Generate circuits random circuits. This is used to create large circuits for performance testing.

#pragma once

#include "graph.hpp"
#include <cassert>
#include <random>

namespace Graph {

// Helpful information
// In a connected graph, a cutset is a set of branches that would cause the graph to be disconnected if removed.
// A current source cutset is a cutset in which every branch corresponds to a current source.
// In voltage source loop is a loop in a graph where every branch corresponnds to a voltage source.
// In order for a network to be uniquely solvable, it is necessary (but not sufficient) that:
// 1. the network graph contains no current source cutsets.
// 2. the network graph contains no voltage source loops.
// these are the "consistency requirements"
// this is sufficient for linear resistive circuits with no controlled sources

// NOTE num_of_nodes and num_of_branches are just minimums. It is not expected the actual number of branches will be much
// higher, but a new node is created for each voltage source added, which could be considerable.
template <typename F> Branches<F> generate_circuit(const size_t num_of_nodes, const size_t num_of_branches) {
    // Assert that there are some branches, but not just a self loop
    assert(num_of_branches > 1);
    // Assert that all nodes could be connected
    assert(num_of_nodes <= num_of_branches);

    // random device is used to seed a PRNG (pseudo random number generator) as per
    // https://en.cppreference.com/w/cpp/numeric/random/random_device .
    // Not sure which is best, so choosing default.
    std::random_device random_device;
    std::default_random_engine rng(random_device());
    std::uniform_int_distribution<size_t> type_dist(0, 2);
    std::uniform_real_distribution<F> value_dist(0, 10);

    Branches<F> branches;
    branches.reserve(num_of_branches);

    size_t next_extra_node = num_of_nodes;
    auto add_component = [&branches, &rng, &type_dist, &value_dist, &next_extra_node](size_t start, size_t end) {
        auto choice = type_dist(rng);
        switch (choice) {
        case 0: // Resistor
            branches.add(start, end, Resistor<F>{value_dist(rng)});
            break;
        case 1: // Voltage Source
                // add negligible resistor to avoid voltage loops
            branches.add(start, next_extra_node, VoltageSource<F>{value_dist(rng)});
            branches.add(next_extra_node, end, Resistor<F>{F{0.0001}});
            next_extra_node += 1;
            break;
        default: // Current Source
            branches.add(start, end, CurrentSource<F>{value_dist(rng)});
            // add large resistor in parallel to avoid current source cutsets
            branches.add(start, end, Resistor<F>{1000});
        }
    };

    // first ensure that all the nodes are connected (in a line)
    for (size_t i = 0; i < num_of_nodes - 1; ++i) {
        add_component(i, i + 1);
    }
    // complete the loop
    add_component(num_of_nodes - 1, 0);

    // don't connect anything to the new nodes since they are there to avoid voltage loops
    std::uniform_int_distribution<size_t> node_select(0, num_of_nodes - 1);
    while (branches.paths.size() < num_of_branches) {
        auto start = node_select(rng);
        auto end = node_select(rng);
        // avoid loopback to self
        while (start == end) {
            end = node_select(rng);
        }
        add_component(start, end);
    }

    return branches;
}
} // namespace Graph
