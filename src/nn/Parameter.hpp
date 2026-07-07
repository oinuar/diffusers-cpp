#pragma once

#include "nn/Module.hpp"
#include "nn/Visitor.hpp"
#include "ggml/Tensor.hpp"

class Parameter : public Module {
public:
    Parameter(const Tensor::Shape& shape)
        : tensor_(), shape_(shape)
    {
    }

    Tensor forward() {
        if (tensor_.ndim() == 0)
            throw std::runtime_error("Undefined tensor parameter; did you forget to set it?");

        return tensor_;
    }
    
    void set(Tensor tensor) {
        tensor_ = tensor;
    }

    const Tensor::Shape& shape() const {
        return shape_;
    }

    virtual void accept(Visitor& visitor, std::vector<std::string> path) {
        visitor.visit(*this, std::move(path));
    }

private:
    Tensor::Shape shape_;
    Tensor tensor_;
};
