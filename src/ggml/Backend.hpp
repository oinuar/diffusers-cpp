#pragma once

#include "ggml/Device.hpp"

class Backend {
public:
    /** @brief Constructs a backend from a device. */
    Backend(Device& device)
        : device_(device), backend_(ggml_backend_dev_init(*device, nullptr))
    {
    }

    ~Backend() {
        if (backend_ != nullptr)
            ggml_backend_free(backend_);
    }

    ggml_backend_t operator *() {
        return backend_;
    }

    Device& device() {
        return device_;
    }

    Backend(Backend&) = delete;
    Backend& operator =(const Backend&) = delete;
private:
    Device& device_;
    ggml_backend_t backend_;
};
