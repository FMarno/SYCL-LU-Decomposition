// Unfortunately I couldn't get gtest working without a c compiler (not that I spent a long time trying)
// cmake was complaining about this because dpcpp only comes with a c++ compiler on windows
// So I have written (hacked together) my own testing suite.
#include "comparison_test.hpp"
#include "sta_test.hpp"
#include "sycl_solver_test.hpp"
#include "system_solver_test.hpp"

int main() {
    auto tests = std::vector{&test_sta, &sycl_solver_test, &test_system_solvers, &comparison_test};
    for (size_t i = 0; i < tests.size(); ++i) {
        auto test_fun = tests[i];
        std::fputs(("running test " + std::to_string(i)).c_str(), stdout);
        try {
            auto ret = test_fun();
            if (ret) {
                std::puts(" -- failed");
                std::puts(ret.value().c_str());
                return 1;
            } else {
                std::puts(" -- passed");
            }
        } catch (std::string s) {
            std::puts("string thrown:");
            std::puts(s.c_str());
            return 2;
        }
    }
    return 0;
}
