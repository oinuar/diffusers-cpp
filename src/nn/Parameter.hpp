#pragma once

#include "nn/Module.hpp"
#include "nn/Visitor.hpp"
#include "ggml/MetaDevice.hpp"
#include <optional>
#include "ggml/Tensor.hpp"

class Parameter : public Module {
public:
    Parameter(const Tensor::Shape& shape)
        : shape_(shape), tensor_()
    {
    }

    /** @brief Marks this parameter to be sharded across the devices of a meta
     *  device along the given PyTorch axis when materialized into a meta buffer.
     *  The extent along the axis must be divisible by the device count. */
    void set_split(int64_t pytorch_axis) {
        split_axis_ = pytorch_axis;
    }

    /** @brief The PyTorch axis this parameter is sharded along, if any. */
    std::optional<int64_t> split_axis() const {
        return split_axis_;
    }

    /** @brief Registers this parameter's split state with a meta device.
     *  Must be called before the tensor is allocated into a meta buffer. */
    void apply_split(MetaDevice& meta, ggml_tensor* tensor) const {
        if (!split_axis_)
            return;

        const auto ggml_axis = shape_.rank() - 1 - *split_axis_;
        meta.split(tensor, SplitState::split(ggml_axis, shape_[*split_axis_], meta.n_devices()));
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

    virtual void accept(Visitor& visitor, std::vector<std::string> path) {
        visitor.visit(*this, std::move(path));
    }

private:
    Tensor::Shape shape_;
    Tensor tensor_;
    std::optional<int64_t> split_axis_;
};
