#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Scope.hpp"
#include "ggml/Buffer.hpp"
#include <ggml.h>
#include <ggml-alloc.h>
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
        : ctx_(other.ctx_), metadata_(std::move(other.metadata_)), bindings_(std::move(other.bindings_)), capacity_(other.capacity_), buffer_(std::move(other.buffer_)), gallocr_(other.gallocr_)
    {
        other.ctx_ = nullptr;
        other.gallocr_ = nullptr;
    }

    ~Context() {
        if (gallocr_ != nullptr)
            ggml_gallocr_free(gallocr_);

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

    /** @brief Allocates the context's compute tensors with liveness-aware
     *  graph allocation (ggml_gallocr): a tensor's storage is freed when
     *  the graph no longer consumes it, so the buffer holds only the peak
     *  set of simultaneously-live tensors -- not every tensor of the
     *  graph. Tensors already allocated elsewhere (the weights, the
     *  state) are left untouched; the graph's outputs (set_output) are
     *  never freed and survive until release().
     *
     *  The graph must contain every tensor of this context that is
     *  executed: tensors outside the graph are not allocated (a dead
     *  tensor needs no storage).
     */
    void allocate_graph(ggml_cgraph* graph, ggml_backend_buffer_type_t buft) {
        if (buffer_ || gallocr_)
            throw std::runtime_error("Context::allocate_graph: the context is already allocated");

        gallocr_ = ggml_gallocr_new(buft);

        if (gallocr_ == nullptr || !ggml_gallocr_alloc_graph(gallocr_, graph)) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
            throw std::runtime_error("Context::allocate_graph: ggml_gallocr_alloc_graph failed");
        }

        // Initialize the views whose source tensor is already allocated
        // (in this context or in another one): they use the source's
        // buffer. (The graph allocator covers the views it allocates;
        // this catches the rest -- e.g. a dead view of a weight.)
        init_views();
    }

    /** @brief The size of the context's allocated storage (0 if
     *  unallocated).
     */
    size_t allocated_size() const {
        if (gallocr_ != nullptr)
            return ggml_gallocr_get_buffer_size(gallocr_, 0);

        if (buffer_)
            return buffer_->size();

        return 0;
    }

    /** @brief Frees the context's buffer and unassigns its tensors: the
     *  tensors keep their metadata in the context, but their storage is
     *  gone (and can be (re-)allocated later).
     */
    void release() {
        if (gallocr_ != nullptr) {
            ggml_gallocr_free(gallocr_);
            gallocr_ = nullptr;
        }

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
            once
        };
    }

    void unbind(Tensor tensor) {
        bindings_.erase(tensor);
    }

    Context(const Context&) = delete;
    Context& operator =(Context&&) = delete;
    Context& operator =(const Context&) = delete;

private:
    ggml_context* ctx_;
    std::optional<Buffer> buffer_;
    ggml_gallocr_t gallocr_ = nullptr;
    std::vector<std::byte> metadata_;
    Bindings bindings_;
    size_t capacity_;

    void init_views() {
        for (auto t = ggml_get_first_tensor(ctx_); t != nullptr; t = ggml_get_next_tensor(ctx_, t))
            // Only the views whose source tensor is already allocated can
            // be initialized: a dead tensor (outside the graph of the
            // liveness-aware allocation) has no storage, and so has no
            // view.
            if (t->data == nullptr && t->view_src != nullptr && t->buffer == nullptr && t->view_src->data != nullptr)
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
