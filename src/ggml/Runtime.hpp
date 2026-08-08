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
    using Provider = std::function<std::vector<T>(std::mt19937&)>;

    struct TensorInputHash {
        std::size_t operator()(const Tensor& t) const noexcept {
            return std::hash<ggml_tensor*>()(*t);
        }
    };

    struct TensorInputEqual {
        bool operator()(const Tensor& a, const Tensor& b) const noexcept {
            return *a == *b;
        }
    };

    typedef std::unordered_map<Tensor, Provider<std::byte>, TensorInputHash, TensorInputEqual> Inputs;

    explicit Runtime(Scheduler& scheduler, uint64_t seed = std::random_device{}())
        : context_(scheduler.arena()), scheduler_(scheduler), rng_(seed) {}

    template <class T>
    Tensor create(const Tensor::Shape& shape, const Provider<T>& provider) {
        auto tensor = Tensor::empty<T>(*context(), shape);
        bind(tensor, provider, true);
        return tensor;
    }

    template <class T>
    void bind(Tensor tensor, const Provider<T>& provider, bool once = false) {
        inputs_[tensor] = [this, tensor, provider, once](std::mt19937& rng) {
            auto values = provider(rng);

            std::vector<std::byte> bytes(values.size() * sizeof(T));
            std::memcpy(bytes.data(), values.data(), bytes.size());

            if (once)
                this->unbind(tensor);

            return std::move(bytes);
        };
    }

    void unbind(Tensor tensor) {
        inputs_.erase(tensor);
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
    Inputs inputs_;

    friend class Graph;
};
