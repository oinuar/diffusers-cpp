#pragma once

#include "Tensor.hpp"

class Parameter {
public:
    Parameter() : tensor_(), fn_() {}

    Parameter(std::function<Tensor(ggml_context*)> fn) : tensor_(), fn_() {}

    void assign(ggml_context* ctx) {
        tensor_ = fn_(ctx);
    }

    const Tensor& operator *() const {
        return tensor_;
    }

private:
    Tensor tensor_;
    std::function<Tensor(ggml_context*)> fn_;
};
