#pragma once

#include <ggml.h>
#include <vector>

class Context {
public:
    Context(size_t capacity = GGML_DEFAULT_GRAPH_SIZE)
        : buffer_(ggml_tensor_overhead() * capacity + ggml_graph_overhead()), ctx_(nullptr)
    {
        ctx_ = ggml_init({
            /*.mem_size   =*/ buffer_.size(),
            /*.mem_buffer =*/ buffer_.data(),
            /*.no_alloc   =*/ true,
        });
    }

    Context(Context&& other) : buffer_(std::move(other.buffer_)), ctx_(other.ctx_) {
        other.ctx_ = nullptr;
    }

    ~Context() {
        if (ctx_ != nullptr)
            ggml_free(ctx_);
    }

    ggml_context* operator *() {
        return ctx_;
    }

    Context(Context&) = delete;
    Context& operator =(const Context&) = delete;

private:
    ggml_context* ctx_;
    std::vector<std::byte> buffer_;
};
