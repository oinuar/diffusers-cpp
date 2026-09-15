#pragma once

#include "nn/Module.hpp"
#include "nn/Visitor.hpp"
#include "ggml/Tensor.hpp"
#include <optional>

class Parameter : public Module {
public:
    Parameter(const Tensor::Shape& shape, const ggml_type& dtype = Tensor::DType<float>::value)
        : shape_(shape), dtype_(dtype), tensor_()
    {
    }

    Tensor forward(Scope scope) {
        if (!tensor_)
            throw std::runtime_error("Undefined tensor Parameter. Did you forget to set it?");

        scope.runtime().set_param(*tensor_);

        return tensor_;
    }
    
    void set(Tensor tensor, std::optional<std::string> name = std::nullopt) {
        tensor_ = tensor;

        if (name)
            tensor_.name(name->c_str());
    }

    Tensor operator *() const {
        if (!tensor_)
            throw std::runtime_error("Undefined tensor Parameter");

        return tensor_;
    }

    const Tensor* operator ->() const {
        if (!tensor_)
            throw std::runtime_error("Undefined tensor Parameter");

        return &tensor_;
    }

    const Tensor::Shape& shape() const {
        return shape_;
    }

    ggml_type dtype() const {
        return dtype_;
    }

    virtual void accept(Visitor& visitor, std::vector<std::string> path) {
        visitor.visit(*this, std::move(path));
    }

private:
    Tensor::Shape shape_;
    ggml_type dtype_;
    Tensor tensor_;
};
