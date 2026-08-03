#pragma once

#include "ggml/Context.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Tensor.hpp"
#include <functional>
#include <vector>
#include <random>
#include <cstring>

class Runtime {
public:
    template<typename T>
    using Initializer = std::function<std::vector<T>(Tensor, std::mt19937&)>;

    explicit Runtime(Scheduler& scheduler, uint64_t seed = std::random_device{}())
        : context_(scheduler.arena()), scheduler_(scheduler), rng_(seed) {}

    template <class T>
    Tensor create(const Tensor::Shape& shape, const Initializer<T>& initializer) {
        auto tensor = Tensor::empty<T>(*context_, shape);

        inputs_.push_back({tensor, [initializer](Tensor tensor, std::mt19937& rng) {
            auto values = initializer(tensor, rng);

            std::vector<std::byte> bytes(values.size() * sizeof(T));
            std::memcpy(bytes.data(), values.data(), bytes.size());

            return std::move(bytes);
        }});

        return tensor;
    }

    Context& context() {
        return context_;
    }

    std::mt19937& rng() {
        return rng_;
    }

    Runtime(Runtime&) = delete;
    Runtime& operator =(const Runtime&) = delete;

private:
    Context context_;
    Scheduler& scheduler_;
    std::mt19937 rng_;
    std::vector<std::pair<Tensor, Initializer<std::byte>>> inputs_;

    friend class Graph;
};
