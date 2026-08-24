#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Device.hpp"
#include "nn/Parameter.hpp"
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <iostream>

/** @brief A virtual device that shards tensors across N underlying devices
 *  (e.g. multiple GPUs, or two CPUs in a test).
 *
 * Wraps ggml's meta backend. Statically-allocated tensors (weights) are given an
 * explicit SplitState via split() before they are allocated into a meta buffer;
 * compute tensors are split automatically to stay compatible with their operands.
 *
 * @note ggml caches the meta device (and its split-state callback) for the process
 *  lifetime and keeps a pointer to this object's registry. Keep the MetaDevice
 *  alive for as long as the meta device / backend is in use.
 */
class MetaDevice : public Device {
public:
    explicit MetaDevice(std::vector<ggml_backend_dev_t> devices)
        : Device(ggml_backend_meta_device(devices.data(), devices.size(), get_split_state, &specs_)),
          specs_(),
          n_devices_(devices.size())
    {
    }

    static MetaDevice all(enum ggml_backend_dev_type type) {
        std::vector<ggml_backend_dev_t> devices;

        for (auto i = 0; i < ggml_backend_dev_count(); ++i) {
            auto dev = ggml_backend_dev_get(i);

            if (ggml_backend_dev_type(dev) == type)
                devices.push_back(dev);
        }

        return MetaDevice(devices);
    }
    
    virtual void visit(Parameter& parameter, std::vector<std::string>) {
        if (!parameter.split_dim())
            return;

        if (n_devices_ < 1) {
            std::cerr << "Cannot split tensor: n_devices must be >= 1" << std::endl;
            return;
        }

        auto dim = *parameter.split_dim();
        auto extent = parameter.shape()[dim];

        if (extent % n_devices_ != 0) {
            std::cerr << "Cannot split tensor: extent must be divisible by n_devices" << std::endl;
            return;
        }

        ggml_backend_meta_split_state state;
        state.axis = static_cast<ggml_backend_meta_split_axis>(parameter.shape().rank() - 1 - dim);
        state.n_segments = 1;
        state.nr[0] = 1;

        auto per_device = extent / n_devices_;

        for (auto i = 0; i < n_devices_; ++i)
            state.ne[i] = per_device;

        specs_[**parameter] = state;
    }

    virtual void split(const Tensor& tensor, int64_t dim) {
        if (n_devices_ < 1) {
            std::cerr << "Cannot split tensor: n_devices must be >= 1" << std::endl;
            return;
        }

        auto extent = tensor.shape()[dim];

        if (extent % n_devices_ != 0) {
            std::cerr << "Cannot split tensor: extent must be divisible by n_devices" << std::endl;
            return;
        }

        ggml_backend_meta_split_state state;
        state.axis = static_cast<ggml_backend_meta_split_axis>(tensor.shape().rank() - 1 - dim);
        state.n_segments = 1;
        state.nr[0] = 1;

        auto per_device = extent / n_devices_;

        for (auto i = 0; i < n_devices_; ++i)
            state.ne[i] = per_device;

        specs_[*tensor] = state;
    }

    virtual void mirror(const Tensor& tensor) {
        ggml_backend_meta_split_state state;

        state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        state.ne[0] = 0;
        state.nr[0] = 1;
        state.n_segments = 1;

        specs_[*tensor] = state;
    }

    MetaDevice(const MetaDevice&) = delete;
    MetaDevice& operator=(const MetaDevice&) = delete;
    MetaDevice(MetaDevice&&) = delete;
    MetaDevice& operator=(MetaDevice&&) = delete;

private:
    typedef std::unordered_map<const ggml_tensor*, ggml_backend_meta_split_state> SplitSpecs;
    SplitSpecs specs_;
    size_t n_devices_;

    static ggml_backend_meta_split_state get_split_state(const ggml_tensor* tensor, void* ud) {
        auto specs = reinterpret_cast<SplitSpecs*>(ud);
        auto it = specs->find(tensor);

        if (it == specs->end()) {
            ggml_backend_meta_split_state state;

            state.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
            state.ne[0] = 0;
            state.nr[0] = 1;
            state.n_segments = 1;

            return state;
        }

        return it->second;
    }
};
