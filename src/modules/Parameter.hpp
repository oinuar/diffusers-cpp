#pragma once

#include "modules/Module.hpp"
#include "modules/Visitor.hpp"
#include <iostream>

template <size_t N>
class Parameter : public Module {
public:
    Parameter(const std::array<int64_t, N>& shape)
        : tensor_(), shape_(shape)
    {
    }

    Tensor forward() {
        if (tensor_.ndim() != N)
            throw std::runtime_error("Undefined tensor parameter; did you forget to set it?");

        return tensor_;
    }
    
    void set(Tensor tensor) {
        tensor_ = tensor;
    }

    const std::array<int64_t, N>& shape() const {
        return shape_;
    }

    virtual void accept(Visitor& visitor, std::vector<std::string> path) {
        visitor.visit(*this, std::move(path));
    }

private:
    std::array<int64_t, N> shape_;
    Tensor tensor_;
};
