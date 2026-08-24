#pragma once

#include "ggml.h"
#include "ggml-backend.h"

class Backend {
public:
    Backend(enum ggml_backend_dev_type type) : backend_() {
        backend_ = ggml_backend_init_by_type(type, nullptr);
    }

    /** @brief Constructs a backend from a device (e.g. a meta device). */
    Backend(ggml_backend_dev_t dev) : backend_() {
        backend_ = ggml_backend_dev_init(dev, nullptr);
    }

    Backend(Backend&& other) : backend_(other.backend_) {
        other.backend_ = nullptr;
    }

    ~Backend() {
        if (backend_ != nullptr)
            ggml_backend_free(backend_);
    }

    ggml_backend_t operator *() {
        return backend_;
    }

    Backend(Backend&) = delete;
    Backend& operator =(const Backend&) = delete;
private:
    ggml_backend_t backend_;

    friend class Scheduler;
};