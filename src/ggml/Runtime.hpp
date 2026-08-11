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

    explicit Runtime(Scheduler& scheduler, Context& context, uint64_t seed = std::random_device{}())
        : scheduler_(scheduler), context_(context), rng_(seed) {}

    template <class T>
    Tensor create(const Tensor::Shape& shape, const Provider<T>& provider) {
        auto tensor = Tensor::empty<T>(*context(), shape);
        bind(tensor, provider, true);
        return tensor;
    }

    template <class T>
    void bind(Tensor tensor, const Provider<T>& provider, bool once = false) {
        (*tensor)->flags |= GGML_TENSOR_FLAG_INPUT;

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
        (*tensor)->flags &= ~GGML_TENSOR_FLAG_INPUT;
        inputs_.erase(tensor);
    }

    template<class T>
    std::vector<T> value(const Tensor& tensor)
    {
        constexpr auto expected = Tensor::DType<T>::value;

        if (tensor.dtype() != expected)
            throw std::invalid_argument("value(): dtype mismatch: expected " + std::string(ggml_type_name(expected)) + ", but got " + std::string(ggml_type_name(tensor.dtype())));

        std::vector<T> data(
            ggml_nelements(*tensor)
        );

        if (data.size() * sizeof(T) != ggml_nbytes(*tensor))
            throw std::invalid_argument("value(): data size mismatch: expected " + std::to_string(data.size() * sizeof(T)) + ", but got " + std::to_string(ggml_nbytes(*tensor)));

        ggml_backend_tensor_get(
            *tensor,
            data.data(),
            0,
            ggml_nbytes(*tensor)
        );

        return std::move(data);
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
    Scheduler& scheduler_;
    Context& context_;
    std::mt19937 rng_;
    Inputs inputs_;

    friend class Graph;
};
