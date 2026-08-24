#pragma once

#include "nn/Visitor.hpp"
#include <ggml-backend.h>
#include <stdexcept>

class Tensor;

class Device : public Visitor {
public:
    explicit Device(ggml_backend_dev_t device) : device_(device) {
        if (device_ == nullptr)
            throw std::runtime_error("No device " + std::string(ggml_backend_dev_name(device)) + " found");
    }

    explicit Device(enum ggml_backend_dev_type type) : Device(ggml_backend_dev_by_type(type)) {}

    ggml_backend_dev_t operator *() const {
        return device_;
    }

    ggml_backend_buffer_type_t buffer_type() const {
        return ggml_backend_dev_buffer_type(device_);
    }

    virtual void split(const Tensor& tensor, int64_t dim) {
        // Omit for singular device
    }

    virtual void mirror(const Tensor& tensor) {
        // Omit for singular device
    }

private:
    ggml_backend_dev_t device_;
};
