#pragma once

#include "ggml.h"

#include "Arena.hpp"

class Context {
public:
    Context(Arena& arena) : ctx_(nullptr) {
        ctx_ = ggml_init({
            /*.mem_size   =*/ arena.size(),
            /*.mem_buffer =*/ arena.data(),
            /*.no_alloc   =*/ true, // the tensors will be allocated later
        });
    }

    Context(Context&& other) : ctx_(other.ctx_) {
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
};
