#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Buffer.hpp"
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

    struct Binding {
        Provider<std::byte> provider;
        bool once;
        bool unbound;
    };

    typedef std::unordered_map<Tensor, Binding, TensorInputHash, TensorInputEqual> Bindings;

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
        : ctx_(other.ctx_), metadata_(std::move(other.metadata_)), bindings_(std::move(other.bindings_)), capacity_(other.capacity_), buffer_(std::move(other.buffer_))
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

    /** @brief Allocates the context's unallocated tensors into a new
     *  buffer owned by the context (meta-aware: for a meta buffer type
     *  the split rules are applied and memory is allocated on each
     *  underlying device).
     *
     *  The buffer persists until release() or the context's destruction,
     *  so the pinned (weights) and the state contexts -- allocated once
     *  on the first run -- stay resident and are shared across all the
     *  computations that use them. Calling allocate() on an already
     *  allocated context is a no-op.
     */
    void allocate(ggml_backend_buffer_type_t buft, ggml_backend_buffer_usage usage) {
        if (buffer_)
            return;

        // If size is 0, there might be no tensors to allocate, or only
        // views that need initialization.
        if (ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_, buft) == 0) {
            init_views();
            return;
        }

        auto buff = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft);

        if (buff != nullptr) {
            // The Meta backend automatically propagates the usage to all
            // its underlying simple device buffers.
            buffer_.emplace(buff, usage);
        }

        // Initialize the views whose source tensor is already allocated
        // (in this context or in another one): they use the source's
        // buffer.
        init_views();
    }

    /** @brief Frees the context's buffer and unassigns its tensors: the
     *  tensors keep their metadata in the context, but their storage is
     *  gone (and can be (re-)allocated later).
     */
    void release() {
        buffer_.reset();
        unassign();
    }

    const std::optional<Buffer>& buffer() const {
        return buffer_;
    }


    const Bindings& bindings() const {
        return bindings_;
    }

    template <typename T>
    Tensor create(const Tensor::Shape& shape, const Provider<T>& provider) {
        Scope scope(*this);
        auto tensor = Tensor::empty<T>(shape).input();
        bind(tensor, provider, true);
        return tensor;
    }

    template <typename T>
    Tensor value(const Tensor::Shape& shape, const Provider<T>& provider) {
        Scope scope(*this);
        auto tensor = Tensor::empty<T>(shape).input();
        bind(tensor, provider);
        return tensor;
    }

    template <typename T>
    void bind(Tensor tensor, const Provider<T>& provider, bool once = false) {
        bindings_[tensor] = {
            [tensor, provider](std::mt19937& rng) {
                auto values = provider(rng);

                if constexpr (std::is_same_v<T, std::byte>)
                    return std::move(values);

                // Convert T[] to std::byte[]
                std::vector<std::byte> bytes(values.size() * sizeof(T));
                std::memcpy(bytes.data(), values.data(), bytes.size());
                return bytes;
            },
            once,
            /*unbound = */false
        };
    }

    void unbind(Tensor tensor) {
        auto it = bindings_.find(tensor);

        // Mark binding as unbound. We need to keep track of all
        // the bindings to support context reallocation.
        if (it != std::end(bindings_))
            it->second.unbound = true;
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

    void reset() {
        for (auto& [tensor, binding] : bindings_)
            binding.unbound = false;
    }

    Context(const Context&) = delete;
    Context& operator =(Context&&) = delete;
    Context& operator =(const Context&) = delete;

private:
    ggml_context* ctx_;
    std::optional<Buffer> buffer_;
    std::vector<std::byte> metadata_;
    Bindings bindings_;
    size_t capacity_;

    void init_views() {
        for (auto t = ggml_get_first_tensor(ctx_); t != nullptr; t = ggml_get_next_tensor(ctx_, t))
            if (t->data == nullptr && t->view_src != nullptr && t->buffer == nullptr)
                ggml_backend_view_init(t);
    }

    /** @brief Resets the tensors' data/buffer pointers: the context's
     *  tensors keep their metadata, but their storage is detached --
     *  preventing any use of freed storage and letting the context be
     *  allocated again later (the ggml allocators skip tensors that
     *  still carry data).
     */
    void unassign() {
        for (auto t = ggml_get_first_tensor(ctx_); t != nullptr; t = ggml_get_next_tensor(ctx_, t)) {
            t->data = nullptr;
            t->buffer = nullptr;
        }
    }
};
