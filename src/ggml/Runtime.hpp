#pragma once

#include "ggml/Context.hpp"
#include "ggml/Scheduler.hpp"
#include "ggml/Tensor.hpp"
#include <functional>
#include <vector>
#include <random>
#include <cstring>

class Allocator;

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

    typedef std::unordered_map<Tensor, std::pair<Provider<std::byte>, bool>, TensorInputHash, TensorInputEqual> Bindings;

    explicit Runtime(Scheduler& scheduler, Context& context, uint64_t seed = std::random_device{}())
        : scheduler_(scheduler), context_(context), rng_(seed) {}

    Runtime(Runtime& other)
        : scheduler_(other.scheduler_), context_(other.context_), rng_(other.rng_), bindings_(other.bindings_)
    {
    }

    template <typename T>
    void bind(Tensor tensor, const Provider<T>& provider, bool once = false) {
        bindings_[tensor] = std::make_pair([tensor, provider](std::mt19937& rng) {
            auto values = provider(rng);

            if constexpr (std::is_same_v<T, std::byte>)
                return std::move(values);

            // Convert T[] to std::byte[]
            std::vector<std::byte> bytes(values.size() * sizeof(T));
            std::memcpy(bytes.data(), values.data(), bytes.size());
            return bytes;
        }, once);
    }

    void unbind(Tensor tensor) {
        bindings_.erase(tensor);
    }

    template <typename T>
    Tensor create(const Tensor::Shape& shape, const Provider<T>& provider, Allocator* allocator = nullptr) {
        auto tensor = Tensor::empty<T>(*context(), shape, allocator);
        bind(tensor, provider, true);
        return tensor;
    }

    template <typename T>
    Tensor value(const Tensor::Shape& shape, const Provider<T>& provider, Allocator* allocator = nullptr) {
        auto tensor = Tensor::empty<T>(*context(), shape, allocator);
        bind(tensor, provider);
        return tensor;
    }

    /** @brief Creates a tensor of the sequence start, start+step, ..., < stop.
     */
    Tensor arange(float start, float stop, float step = 1.0f) {
        const int64_t size = static_cast<int64_t>(std::ceil((stop - start) / step));

        return create<float>({size}, [start, step, size](std::mt19937&) {
            std::vector<float> values(static_cast<size_t>(size));

            for (size_t i = 0; i < values.size(); ++i)
                values[i] = start + static_cast<float>(i) * step;

            return values;
        });
    }

    void copy(const Tensor& src, const Tensor& dst) {
        ggml_backend_tensor_copy(*src, *dst);
    }

    template<class T>
    std::vector<T> read(const Tensor& tensor) {
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

    void write(const Tensor& tensor, const std::vector<std::byte>& bytes) {
        if (bytes.size() != ggml_nbytes(*tensor))
            throw std::invalid_argument("write(): data size mismatch: expected " + std::to_string(bytes.size()) + ", but got " + std::to_string(ggml_nbytes(*tensor)));

        ggml_backend_tensor_set(
            *tensor,
            bytes.data(),
            0,
            ggml_nbytes(*tensor)
        );
    }

    Scheduler& scheduler() {
        return scheduler_;
    }

    Context& context() {
        return context_;
    }

    std::mt19937& rng() {
        return rng_;
    }

    const Bindings& bindings() const {
        return bindings_;
    }

    Runtime(const Runtime&) = delete;
    Runtime& operator =(const Runtime&) = delete;

private:
    Scheduler& scheduler_;
    Context& context_;
    std::mt19937 rng_;
    Bindings bindings_;
};
