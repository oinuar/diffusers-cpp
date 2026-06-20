#pragma once

#include "ggml.h"

#include "GGMLArena.hpp"

class GGMLContext {
public:
    GGMLContext(GGMLArena& arena) : ctx_(nullptr) {
        ctx_ = ggml_init({
            /*.mem_size   =*/ arena.size(),
            /*.mem_buffer =*/ arena.data(),
            /*.no_alloc   =*/ true, // the tensors will be allocated later
        });
    }

    GGMLContext(GGMLContext&& other) : ctx_(other.ctx_) {
        other.ctx_ = nullptr;
    }

    ~GGMLContext() {
        if (ctx_ != nullptr)
            ggml_free(ctx_);
    }

    ggml_context* operator *() {
        return ctx_;
    }

    GGMLContext(GGMLContext&) = delete;
    GGMLContext& operator =(const GGMLContext&) = delete;

private:
    ggml_context* ctx_;
};
