#pragma once

#include "ggml.h"
#include "ggml-backend.h"

class GGMLBackend {
public:
    GGMLBackend(enum ggml_backend_dev_type type) : backend_() {
        backend_ = ggml_backend_init_by_type(type, nullptr);
    }

    GGMLBackend(GGMLBackend&& other) : backend_(other.backend_) {
        other.backend_ = nullptr;
    }

    ~GGMLBackend() {
        if (backend_ != nullptr)
            ggml_backend_free(backend_);
    }

    ggml_backend_t operator *() {
        return backend_;
    }

    GGMLBackend(GGMLBackend&) = delete;
    GGMLBackend& operator =(const GGMLBackend&) = delete;
private:
    ggml_backend_t backend_;

    friend class GGMLScheduler;
};