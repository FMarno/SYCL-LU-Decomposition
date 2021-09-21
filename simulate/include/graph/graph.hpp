// Branch data structure. A circuit is made of up many branches connected at nodes.
#pragma once
#include <cassert>
#include <sstream>
#include <variant>
#include <vector>

namespace Graph {
  namespace {
    template <class> inline constexpr bool always_false_v = false;
  }

template <typename Floating> struct Resistor { Floating value; };

template <typename Floating> struct VoltageSource { Floating value; };

template <typename Floating> struct CurrentSource { Floating value; };

template <typename Floating> struct VoltageControlledVoltageSource {
    Floating multiplier;
    std::size_t branch; // index of the control branch, branches are 0 indexed
};

template <typename Floating> struct VoltageControlledCurrentSource {
    Floating multiplier;
    std::size_t branch; // index of the control branch, branches are 0 indexed
};

template <typename Floating> struct CurrentControlledVoltageSource {
    Floating multiplier;
    std::size_t branch; // index of the control branch, branches are 0 indexed
};

template <typename Floating> struct CurrentControlledCurrentSource {
    Floating multiplier;
    std::size_t branch; // index of the control branch, branches are 0 indexed
};

// TODO confirm any difference between std::variant and union.
// Docs specify that no extra allocations take place, like a union.
template <typename F>
using Component =
    std::variant<Resistor<F>, VoltageSource<F>, CurrentSource<F>, VoltageControlledVoltageSource<F>,
                 VoltageControlledCurrentSource<F>, CurrentControlledVoltageSource<F>, CurrentControlledCurrentSource<F>>;

template <typename Floating> std::string to_string(Component<Floating> component) noexcept {
    return std::visit(
        [](auto &&arg) {
            using T = std::decay_t<decltype(arg)>;
            std::stringstream ss;
            if constexpr (std::is_same_v<T, Resistor<Floating>>)
                ss << "R [" << arg.value << "]";
            else if constexpr (std::is_same_v<T, VoltageSource<Floating>>)
                ss << "VS [" << arg.value << "]";
            else if constexpr (std::is_same_v<T, CurrentSource<Floating>>)
                ss << "CS [" << arg.value << "]";
            else if constexpr (std::is_same_v<T, VoltageControlledVoltageSource<Floating>>)
                ss << "VCVS [" << arg.multiplier << " x b" << arg.branch << "]";
            else if constexpr (std::is_same_v<T, VoltageControlledCurrentSource<Floating>>)
                ss << "VCCS [" << arg.multiplier << " x b" << arg.branch << "]";
            else if constexpr (std::is_same_v<T, CurrentControlledVoltageSource<Floating>>)
                ss << "CCVS [" << arg.multiplier << " x b" << arg.branch << "]";
            else if constexpr (std::is_same_v<T, CurrentControlledCurrentSource<Floating>>)
                ss << "CCCS [" << arg.multiplier << " x b" << arg.branch << "]";
            else {
                static_assert(always_false_v<T>, "non-exhaustive");
            }
            return ss.str();
        },
        component);
}

struct Path {
    std::size_t start_node;
    std::size_t end_node;

    Path(size_t s, size_t e) : start_node(s), end_node(e) {}
};

template <typename Floating, std::enable_if_t<std::is_floating_point<Floating>::value, bool> = true> struct Branches {
    // When creating the system of equations, the building of the reduced incidence matrix and the building of Z,Y and s can
    // happen seperately. Due to this, paths and components can be in this Structure of Arrays form, allowing for better data
    // layout. Also there was never a single 'Branch' in the code, there was always a list of them.
    std::vector<Path> paths;
    std::vector<Component<Floating>> components;

    Branches(std::vector<Path> ps, std::vector<Component<Floating>> cs) : paths(std::move(ps)), components(std::move(cs)) {
        assert(paths.size() == components.size());
    }

    Branches() = default;

    void add(size_t start, size_t end, Component<Floating> component) {
        paths.emplace_back(start, end);
        components.push_back(component);
    }

    void reserve(size_t capacity) {
        paths.reserve(capacity);
        components.reserve(capacity);
    }

    std::string to_string() const noexcept {
        std::stringstream ss;
        for (size_t i = 0; i != paths.size(); ++i) {
            ss << paths[i].start_node << " -> " << to_string(components[i]) << " -> " << paths[i].end_node << '\n';
        }
        return ss.str();
    }
};
} // namespace Graph
