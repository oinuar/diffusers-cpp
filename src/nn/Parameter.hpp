#pragma once

#include "nn/Module.hpp"
#include "nn/Visitor.hpp"
#include <optional>
#include "ggml/Tensor.hpp"

class Parameter : public Module {
public:
    Parameter(const Tensor::Shape& shape, std::optional<int64_t> split_dim = std::nullopt)
        : shape_(shape), tensor_(), split_dim_(split_dim)
    {
    }

    Tensor forward() {
        if (!tensor_)
            throw std::runtime_error("Undefined tensor Parameter; did you forget to set it?");

        return tensor_;
    }
    
    void set(Tensor tensor) {
        tensor_ = tensor;
    }

    Tensor operator *() const {
        if (!tensor_)
            throw std::runtime_error("Undefined tensor Parameter: not set");

        return tensor_;
    }

    const Tensor* operator ->() const {
        if (!tensor_)
            throw std::runtime_error("Undefined tensor Parameter: not set");

        return &tensor_;
    }

    const Tensor::Shape& shape() const {
        return shape_;
    }

    std::optional<int64_t> split_dim() const {
        return split_dim_;
    }

    virtual void accept(Visitor& visitor, std::vector<std::string> path) {
        visitor.visit(*this, std::move(path));
    }

private:
    Tensor::Shape shape_;
    Tensor tensor_;
    std::optional<int64_t> split_dim_;
};
