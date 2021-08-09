#include <cstdio>
#include <memory_resource>

#include "graph/circuit_gen.hpp"
#include "graph/graph.hpp"
#include "graph/sta.hpp"
#include "solve/basic_gaussian.hpp"
#include "solve/gaussian_lu.hpp"
#include "solve/matrix.hpp"
#include "solve/sycl_gaussian_lu.hpp"

using Graph::Branches;
using Graph::CurrentSource;
using Graph::Resistor;
using Graph::VoltageSource;

template <typename FloatType, typename Solver> void time_solver(Solver solve) {
    using std::chrono::duration_cast;
    using std::chrono::nanoseconds;
    typedef std::chrono::high_resolution_clock clock;

    for (int n = 0; n < 5; ++n) {
        auto start = clock::now();
        for (int i = 0; i < 10; ++i) {
            auto graph = Graph::generate_circuit<FloatType>(100, 300);
            auto system = Graph::graph_to_sta_system(graph);
            auto solution = solve(std::move(system));
        }
        auto end = clock::now();
        auto time = duration_cast<nanoseconds>(end - start).count() / 1000000000.0;
        std::fputs(std::to_string(time).c_str(), stdout);
        std::puts(" s");
    }
}

int main() {
    using FloatType = double;
    std::puts("Gaussian LU");
    time_solver<FloatType>(&Solve::Gaussian_LU<FloatType>::solve);
    std::puts("Basic Gaussian");
    time_solver<FloatType>(&Solve::Basic_Gaussian<FloatType>::solve);

    // set up SYCL
    sycl::default_selector selector;

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

    std::puts("Gaussian LU");
    time_solver<FloatType>([&q](auto system) { return Solve::SYCL_Gaussian_LU<FloatType>::solve(q, std::move(system)); });
}
