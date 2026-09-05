#pragma once

#include "ggml/Tensor.hpp"
#include <ggml.h>
#include <ggml-backend.h>
#include <vector>
#include <random>
#include <functional>
#include <cstring>

class Context {
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

    Context(size_t capacity = GGML_DEFAULT_GRAPH_SIZE)
        : ctx_(nullptr), metadata_(ggml_tensor_overhead() * capacity + ggml_graph_overhead()), bindings_(), capacity_(capacity)
    {
        ctx_ = ggml_init({
            /*.mem_size   =*/ metadata_.size(),
            /*.mem_buffer =*/ metadata_.data(),
            /*.no_alloc   =*/ true,
        });
    }

    Context(Context&& other)
        : ctx_(other.ctx_), metadata_(std::move(other.metadata_)), bindings_(std::move(other.bindings_)), capacity_(other.capacity_)
    {
        other.ctx_ = nullptr;
    }

    ~Context() {
        if (ctx_ != nullptr)
            ggml_free(ctx_);
    }

    ggml_context* operator *() {
        return ctx_;
    }

    size_t capacity() const {
        return capacity_;
    }

    const Bindings& bindings() const {
        return bindings_;
    }

    template <typename T>
    Tensor create(const Tensor::Shape& shape, const Provider<T>& provider) {
        auto tensor = Tensor::empty<T>(shape).input();
        bind(tensor, provider, true);
        return tensor;
    }

    template <typename T>
    Tensor value(const Tensor::Shape& shape, const Provider<T>& provider) {
        auto tensor = Tensor::empty<T>(shape).input();
        bind(tensor, provider);
        return tensor;
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

    Context(const Context&) = delete;
    Context& operator =(Context&&) = delete;
    Context& operator =(const Context&) = delete;

private:
    ggml_context* ctx_;
    std::vector<std::byte> metadata_;
    Bindings bindings_;
    size_t capacity_;
};
