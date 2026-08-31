#pragma once

#include "ggml/Tensor.hpp"
#include "ggml/Device.hpp"
#include "nn/Parameter.hpp"
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <iostream>

/** @brief A virtual device that shards tensors across N underlying devices
 *  (e.g. multiple GPUs).
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
        : Device(ggml_backend_meta_device(devices.data(), devices.size(), get_split_state, nullptr)),
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

    size_t count() const {
        return n_devices_;
    }

    MetaDevice(const MetaDevice&) = delete;
    MetaDevice& operator=(const MetaDevice&) = delete;
    MetaDevice(MetaDevice&&) = delete;
    MetaDevice& operator=(MetaDevice&&) = delete;

private:
    typedef std::unordered_map<const ggml_tensor*, ggml_backend_meta_split_state> SplitSpecs;
    size_t n_devices_;

    // -----------------------------------------------------------------------------
    // 50/50 tensor split
    //
    // For a tensor split along axis 0:
    //
    //   GPU 0 gets first half
    //   GPU 1 gets second half
    //
    // ggml_backend_meta_split_state::ne is laid out as:
    //   [segment0_dev0, segment0_dev1, ...]
    // -----------------------------------------------------------------------------
    static ggml_backend_meta_split_state get_split_state(const ggml_tensor* tensor, void*) {
        ggml_backend_meta_split_state state = {};

        state.axis = GGML_BACKEND_SPLIT_AXIS_0;
        state.n_segments = 1;
        state.nr[0] = 1;

        const int64_t n = tensor->ne[0];

        state.ne[0] = n / 2;
        state.ne[1] = n - state.ne[0];

        return state;
    }
};
