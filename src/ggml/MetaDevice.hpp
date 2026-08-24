#pragma once

#include "ggml/SplitState.hpp"
#include "ggml-backend.h"
#include <map>
#include <vector>

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
class MetaDevice {
public:
    explicit MetaDevice(std::vector<ggml_backend_dev_t> devices)
        : n_devices_(static_cast<int>(devices.size())),
          dev_(ggml_backend_meta_device(devices.data(), devices.size(), get_split_state, &specs_))
    {
    }

    /** @brief Registers the split state of a statically-allocated tensor (a weight).
     *  Must be called before the tensor is allocated into a meta buffer. */
    void split(const ggml_tensor* tensor, const SplitState& state) {
        specs_[tensor] = state;
    }

    ggml_backend_dev_t operator*() const {
        return dev_;
    }

    ggml_backend_buffer_type_t buffer_type() const {
        return ggml_backend_dev_buffer_type(dev_);
    }

    /** @brief Number of underlying devices the meta device shards across. */
    int n_devices() const {
        return n_devices_;
    }

    /** @brief True if a sharded (non-mirrored) split state is registered for the tensor. */
    bool is_split(const ggml_tensor* tensor) const {
        auto it = specs_.find(tensor);
        return it != specs_.end() && it->second.state.axis != GGML_BACKEND_SPLIT_AXIS_MIRRORED;
    }

    MetaDevice(const MetaDevice&) = delete;
    MetaDevice& operator=(const MetaDevice&) = delete;
    MetaDevice(MetaDevice&&) = delete;
    MetaDevice& operator=(MetaDevice&&) = delete;

private:
    std::map<const ggml_tensor*, SplitState> specs_;
    ggml_backend_dev_t dev_;
    int n_devices_;

    static ggml_backend_meta_split_state get_split_state(const ggml_tensor* tensor, void* ud) {
        const auto& specs = *static_cast<const std::map<const ggml_tensor*, SplitState>*>(ud);
        auto it = specs.find(tensor);
        return it != specs.end() ? it->second.state : SplitState::mirrored().state;
    }
};
